"""Short return gaps, price rounding, and noise-aware performance evidence.

Holiday gaps, low-price tick quantization, and partial price discrepancies
have distinct contracts. Universe timing and input-mass boundaries must not
turn missing evidence into a clean verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit import inputs as qinputs
from qaudit.checks import costs, performance, survivorship
from qaudit.checks.costs import MAX_PLAUSIBLE_ANNUAL_DIV_YIELD
from qaudit.checks.survivorship import (MARKET_CLOSURE_MAX_RUN_BARS,
                                        MISSING_RETURN_FRAC,
                                        TOU_MIN_MEASURABLE_FRAC)
from qaudit.config import AuditConfig
from qaudit.errors import InputValidationError, MisalignedInputError
from qaudit.inputs import BacktestArtifacts
from qaudit.types import Severity, Status

CFG = AuditConfig()

C_PRICES = "costs.price_return_consistency"
C_TURNOVER = "costs.turnover_unrealistic"
S_MISSING = "survivorship.positions_on_missing_returns"
S_TOU = "survivorship.trading_outside_universe"
P_STAB = "performance.ic_stability"
P_REGIME = "performance.ic_regime_concentration"


def _by(results):
    return {r.check: r for r in results}


# ===========================================================================
# 1. positions_on_missing_returns: union-calendar market-holiday NaNs
# ===========================================================================

def _region_book(seed=101, n=300, holidays_per_region=14, hol_run=1):
    """12-name 3-region book on the union calendar: each region's assets
    have NaN returns on that region's local holidays (interior runs of
    ``hol_run`` bars), positions held throughout."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"{reg}{i}" for reg in ("US", "EU", "AP") for i in range(4)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, 12)),
                        index=dates, columns=cols)
    # interior holiday runs, disjoint across regions, away from both ends
    starts = rng.choice(np.arange(10, n - 10 - hol_run),
                        size=3 * holidays_per_region, replace=False)
    for r_idx, reg in enumerate(("US", "EU", "AP")):
        reg_cols = [c for c in cols if c.startswith(reg)]
        for s in starts[r_idx * holidays_per_region:
                        (r_idx + 1) * holidays_per_region]:
            rets.loc[rets.index[s:s + hol_run], reg_cols] = np.nan
    sig = rets.rolling(5, min_periods=5).mean()
    pos = pd.DataFrame(1.0 / 12, index=dates, columns=cols)
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=1)
    art.validate()
    return art.aligned(), rets


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_union_calendar_holiday_book_passes(seed):
    # Interior one-bar market closures affect held cells without implying terminal
    # delisting losses.
    art, _ = _region_book(seed=seed)
    r = _by(survivorship.run(art, CFG))[S_MISSING]
    assert r.status is Status.PASS
    assert r.details["n_market_closure_cells"] > 100
    assert r.details["n_violations"] == 0
    assert "market-holiday" in r.message or "closure" in r.message


def test_terminal_delisting_gap_still_warns():
    # detection guard: a held terminal NaN gap is the real delisting class
    art, rets = _region_book(seed=7, holidays_per_region=2)
    rets.iloc[-60:, 0] = np.nan          # asset dies, positions ride through
    sig = rets.rolling(5, min_periods=5).mean()
    pos = pd.DataFrame(1.0 / 12, index=rets.index, columns=rets.columns)
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=1)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))[S_MISSING]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["n_violations"] >= 60
    assert r.details["frac_missing_return"] > MISSING_RETURN_FRAC
    assert "delisting" in r.message


def test_long_interior_hole_still_warns():
    # detection guard: interior runs longer than the closure cap are data
    # holes / multi-week halts, not holidays - must keep counting (the
    # 50-bar pin in test_qaudit_survivorship.py depends on the same cap)
    art, rets = _region_book(seed=8, holidays_per_region=2,
                             hol_run=MARKET_CLOSURE_MAX_RUN_BARS + 1)
    r = _by(survivorship.run(art, CFG))[S_MISSING]
    # 6 holiday runs x (cap+1) bars x 4 assets = 144 cells, all counted
    assert r.details["n_market_closure_cells"] == 0
    assert r.details["n_violations"] > 100
    assert r.status is Status.WARN


# ===========================================================================
# 2. trading_outside_universe: fabricated decision universe -> SKIP
# ===========================================================================

def _hindsight_book(n=200, lag=1):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-02", periods=n)
    cols = [f"A{i:02d}" for i in range(10)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, 10)),
                        index=dates, columns=cols)
    sig = rets.rolling(5, min_periods=5).mean()
    uni = pd.DataFrame(True, index=dates, columns=cols)
    uni.iloc[50:, 0] = False             # asset 0 leaves the universe
    pos = pd.DataFrame(0.0, index=dates, columns=cols)
    pos.iloc[:, 1] = 0.1
    pos.iloc[150:, 0] = 0.2              # fresh entry long after the exit
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            universe=uni, signal_lag=lag)
    art.validate()
    return art.aligned()


def test_lag_at_or_past_n_periods_skips_not_passes():
    # a hindsight book that FAILs at lag=1 must not PASS at lag >= n-1,
    # where the legal universe is 100% fabricated
    assert _tou(_hindsight_book(lag=1)).status is Status.FAIL
    for lag in (200, 500):
        r = _tou(_hindsight_book(lag=lag))
        assert r.status is Status.SKIP, f"lag={lag}"
        assert "unobservable" in r.message or "unmeasurable" in r.message
        assert r.details["measurable_frac"] == 0.0


def _tou(art):
    return _by(survivorship.run(art, CFG))[S_TOU]


def test_clean_scan_below_measurable_floor_skips():
    # < 50% measurable rows and 0 violations found -> SKIP, not PASS
    rng = np.random.default_rng(3)
    n = 200
    dates = pd.bdate_range("2020-01-02", periods=n)
    cols = list("ABCD")
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, 4)),
                        index=dates, columns=cols)
    sig = rets.rolling(5, min_periods=5).mean()
    uni = pd.DataFrame(True, index=dates, columns=cols)
    pos = pd.DataFrame(0.25, index=dates, columns=cols)

    def _at(lag):
        art = BacktestArtifacts(signals=sig, asset_returns=rets,
                                positions=pos, universe=uni, signal_lag=lag)
        art.validate()
        return _tou(art.aligned())

    r_low = _at(120)                     # measurable 40% < 50% floor
    assert r_low.status is Status.SKIP
    assert f"{TOU_MIN_MEASURABLE_FRAC:.0%}" in r_low.message
    r_ok = _at(80)                       # measurable 60% >= floor
    assert r_ok.status is Status.PASS
    assert r_ok.details["measurable_frac"] == pytest.approx(0.6)
    assert "treated as legal" in r_ok.message


def test_violations_still_fail_below_measurable_floor():
    # fabricated rows can only mask violations: one found in the
    # measurable tail is evidence at any coverage - never SKIP it away
    r = _tou(_hindsight_book(lag=120))   # measurable 40%, entry at row 150
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_violations"] > 0


# ===========================================================================
# 3. price_return_consistency: tick quantization
# ===========================================================================

def _gbm_panel(seed=0, n=600, n_assets=15, lo=2.0, hi=8.0, decimals=2,
               vol=0.015):
    """Full-precision return chain; delivered prices quantized to the tick.
    Returns (rounded_prices, full_precision_returns)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"S{i:02d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0.0002, vol, (n, n_assets)),
                        index=dates, columns=cols)
    p0 = rng.uniform(lo, hi, n_assets)
    chain = p0 * (1.0 + rets).cumprod()
    px = chain.round(decimals)
    return px, rets


def _price_check(px, declared, ppy=252):
    sig = declared.rolling(5, min_periods=5).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=declared, prices=px,
                            periods_per_year=ppy)
    art.validate()
    return _by(costs.run(art.aligned(), CFG))[C_PRICES]


@pytest.mark.parametrize("seed", [1000, 1001, 7000])
def test_cent_rounded_low_price_panel_passes(seed):
    # honest vendor layout (full-precision total returns + 2dp close file)
    # at $2-8 shows median |diff| ~6bp of pure rounding noise; the
    # price-level-aware tolerance must absorb it
    px, rets = _gbm_panel(seed=seed)
    r = _price_check(px, rets)
    assert r.status is Status.PASS, r.message
    assert r.details["on_cent_grid"] is True
    assert r.details["rounding_shaped"] is True
    assert r.details["median_abs_diff"] > 1e-4    # the rounding noise is there
    assert r.details["frac_discrepant"] < 0.01    # ...but tick-bounded
    assert "tick" in r.message or "quantization" in r.message


def test_cent_rounded_panel_date_shift_still_warns():
    # detection guard: a 1-day shift on the same rounded panel medians at
    # ~2x daily vol, 20-100x above tick noise - must not ride the waiver
    px, rets = _gbm_panel(seed=1000)
    r = _price_check(px, rets.shift(1))
    assert r.status is Status.WARN
    # shift medians at ~2x daily vol (~140bp here) - several times the
    # tick scale even on this deliberately low-priced panel
    assert r.details["median_abs_diff"] > 5 * r.details["tick_median_scale"]


def test_cent_rounded_panel_percent_scale_still_warns():
    # Large percent-scaled returns fail the intake floor. Low-volatility scaled panels
    # can clear that floor but must still trigger price-consistency warnings despite
    # tick rounding.
    px, rets = _gbm_panel(seed=1000)
    with pytest.raises(InputValidationError, match="below -1.0"):
        _price_check(px, rets * 100.0)
    px, rets = _gbm_panel(seed=1000, vol=0.002)
    r = _price_check(px, rets * 100.0)
    assert r.status is Status.WARN


def test_same_chain_rounded_panel_passes():
    # honest control: returns derived from the rounded panel itself agree
    # bit-for-bit - no tolerance needed, and none misapplied
    px, _ = _gbm_panel(seed=1001)
    derived = (px / px.shift(1) - 1.0)
    r = _price_check(px, derived)
    assert r.status is Status.PASS
    assert r.details["median_abs_diff"] <= 1e-6


# ===========================================================================
# 4. price_return_consistency: minority-asset mismatch (pervasive prong)
# ===========================================================================

def _mixed_vendor_panel(seed=5, n=300, n_assets=20, n_shifted=9):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    cols = [f"V{i:02d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0.0003, 0.012, (n, n_assets)),
                        index=dates, columns=cols)
    px = pd.DataFrame(rng.uniform(20, 200, n_assets) * (1.0 + rets).cumprod(),
                      index=dates, columns=cols)
    declared = rets.copy()
    declared.iloc[:, :n_shifted] = rets.iloc[:, :n_shifted].shift(1)
    return px, declared


@pytest.mark.parametrize("seed", [5, 6])
def test_minority_asset_date_shift_warns(seed):
    # Shifting 9 of 20 assets affects many cells while leaving the pooled median zero;
    # the pervasiveness check must detect it.
    px, declared = _mixed_vendor_panel(seed=seed)
    r = _price_check(px, declared)
    assert r.status is Status.WARN
    assert r.details["median_abs_diff"] <= 1e-4       # median prong blind...
    assert r.details["frac_discrepant"] >= 0.25       # ...pervasive prong not
    assert r.details["n_assets_discrepant_median"] == 9
    assert "sparse" not in r.message


def test_heavy_payer_coarse_bar_waiver_stays_pass():
    # honest guard for the pervasive prong: monthly heavy payer puts a
    # one-sided sub-cap offset in every cell (frac_discrepant = 1.0) -
    # the discrepant-subset dividend signature must keep waiving it
    rng = np.random.default_rng(9)
    dates = pd.date_range("2015-01-31", periods=60, freq=pd.offsets.MonthEnd())
    cols = [f"F{i:02d}" for i in range(12)]
    price_ret = pd.DataFrame(rng.normal(0.004, 0.04, (60, 12)),
                             index=dates, columns=cols)
    px = pd.DataFrame(100.0 * (1.0 + price_ret).cumprod(),
                      index=dates, columns=cols)
    r = _price_check(px, price_ret + 0.06 / 12.0, ppy=12)
    assert r.status is Status.PASS
    assert r.details["frac_discrepant"] == pytest.approx(1.0)
    assert "dividend" in r.message


def test_sparse_quarterly_dividends_stay_pass():
    # honest guard: ~1.6% of daily cells carry a quarterly dividend point -
    # far under the 25% bar, and the PASS message may keep calling it sparse
    rng = np.random.default_rng(12)
    n, m = 500, 10
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"D{i:02d}" for i in range(m)]
    price_ret = pd.DataFrame(rng.normal(0.0003, 0.012, (n, m)),
                             index=dates, columns=cols)
    px = pd.DataFrame(rng.uniform(30, 90, m) * (1.0 + price_ret).cumprod(),
                      index=dates, columns=cols)
    total = price_ret.copy()
    total.iloc[::63] += 0.01                     # ~1%/quarter dividends
    r = _price_check(px, total)
    assert r.status is Status.PASS
    assert r.details["frac_discrepant"] < 0.05


# ===========================================================================
# 5. ic_stability / ic_regime_concentration: SE-scaled moot gate
# ===========================================================================

def _noise_book(seed, n_periods=750, n_assets=30):
    """No edge by construction: iid N(0,1) signal on iid 1.5%-vol returns.
    Any stability/regime WARN is a false positive by definition."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n_periods)
    cols = [f"A{i}" for i in range(n_assets)]
    rets = pd.DataFrame(0.015 * rng.standard_normal((n_periods, n_assets)),
                        index=dates, columns=cols)
    sig = pd.DataFrame(rng.standard_normal((n_periods, n_assets)),
                       index=dates, columns=cols)
    art = BacktestArtifacts(signals=sig, asset_returns=rets, signal_lag=1)
    art.validate()
    return _by(performance.run(art.aligned(), CFG))


@pytest.mark.parametrize("seed", [90015, 90018, 92001])
def test_noise_book_regime_not_judged(seed):
    # For a noise book, an absolute mean IC near its standard error does not establish
    # a persistent or concentrated edge.
    r = _noise_book(seed)[P_REGIME]
    assert r.status is Status.PASS
    assert "not judged" in r.message
    assert r.details["moot_floor"] >= 2.0 / np.sqrt(29 * 750)


@pytest.mark.parametrize("seed,n_periods,n_assets",
                         [(90004, 750, 30), (91005, 400, 15)])
def test_noise_book_stability_not_judged(seed, n_periods, n_assets):
    r = _noise_book(seed, n_periods, n_assets)[P_STAB]
    assert r.status is Status.PASS
    assert "not judged" in r.message


def test_real_one_year_edge_still_warns_regime():
    # detection guard: IC ~0.03+ concentrated in one calendar year of 3+
    # clears the SE-scaled gate and must still WARN
    rng = np.random.default_rng(21)
    n, m = 780, 30
    dates = pd.bdate_range("2018-01-02", periods=n)
    cols = [f"R{i:02d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, m)), index=dates,
                        columns=cols)
    fwd = rets.shift(-1)
    sig = pd.DataFrame(rng.normal(0.0, 0.02, (n, m)), index=dates,
                       columns=cols)
    mask = dates.year == 2018
    sig.loc[mask] = sig.loc[mask] + fwd.loc[mask].fillna(0.0)
    art = BacktestArtifacts(signals=sig, asset_returns=rets)
    art.validate()
    r = _by(performance.run(art.aligned(), CFG))[P_REGIME]
    assert r.status is Status.WARN
    assert r.details["top_year"] == 2018


# ===========================================================================
# 6a. positions gross-mass gate: pin 0.50 from both sides
# ===========================================================================

def _mass_book(seed=7, n=400, k=12):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    cols = [f"T{i:02d}" for i in range(k)]
    rets = pd.DataFrame(rng.normal(0.0, 0.012, (n, k)),
                        index=dates, columns=cols)
    sig = rets.rolling(10, min_periods=10).mean()
    w = pd.DataFrame(1.0 / k, index=dates, columns=cols)
    w += rng.normal(0.0, 0.005, (n, k))
    return sig, rets, w.clip(lower=0.0), cols


def test_mass_gate_literal_pin():
    # follows the DEFERRED_WARN_RATIO precedent: the calibrated constant is
    # part of the contract - recalibrating it must be a conscious act
    assert qinputs._POSITIONS_MIN_MASS_KEPT == 0.50


def test_midband_suffix_mismatch_rejected():
    # conviction pin inside the designed band: 5/12 columns suffixed,
    # up-weighted 2.1x -> suffixed names carry ~60% of gross (kept ~40%),
    # so the whole (0.081, 0.50) band is covered, not just the 8.1% point
    sig, rets, w, cols = _mass_book(seed=7)
    bad = cols[:5]
    w[bad] *= 2.1
    pos = w.copy()
    pos.columns = [c + " UW Equity" if c in bad else c for c in cols]
    kept = float(pos[[c for c in pos.columns if " " not in c]]
                 .abs().to_numpy().sum() / pos.abs().to_numpy().sum())
    assert 0.35 < kept < 0.45
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos)
    with pytest.raises(MisalignedInputError, match="gross weight"):
        art.validate()


def test_minority_suffix_mismatch_still_validates():
    # honest twin just above the floor: 5/12 equal-weight columns suffixed
    # keeps ~58% of gross - single-name/share-class nits stay tolerated
    sig, rets, w, cols = _mass_book(seed=11)
    pos = w.copy()
    pos.columns = [c + " UW Equity" if c in cols[:5] else c for c in cols]
    kept = float(pos[[c for c in pos.columns if " " not in c]]
                 .abs().to_numpy().sum() / pos.abs().to_numpy().sum())
    assert 0.53 < kept < 0.63
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos)
    art.validate()                        # must not raise


@pytest.mark.parametrize("cash_frac", [0.10, 0.40])
def test_out_of_grid_cash_sleeve_validates(cash_frac):
    # FP-side pin: an honest book carrying a cash/hedge column absent from
    # the signals x asset_returns grid must validate - a floor as high as
    # 0.95 would refuse it
    sig, rets, w, cols = _mass_book(seed=13)
    pos = w * (1.0 - cash_frac)
    pos["CASH_USD"] = cash_frac
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos)
    art.validate()                        # must not raise
    assert list(art.aligned().positions.columns) == cols


# ===========================================================================
# 6b. dividend-yield cap: pin 0.15 inside (0.11, 0.20)
# ===========================================================================

def _income_result(annual_yield):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-31", periods=48, freq=pd.offsets.MonthEnd())
    cols = [f"F{i:02d}" for i in range(15)]
    price_ret = pd.DataFrame(rng.normal(0.004, 0.04, (48, 15)),
                             index=dates, columns=cols)
    px = pd.DataFrame(100.0 * (1.0 + price_ret).cumprod(),
                      index=dates, columns=cols)
    return _price_check(px, price_ret + annual_yield / 12.0, ppy=12)


def test_bdc_mrei_yield_passes_dividend_waiver():
    # pins the cap >= ~0.11: BDC/mREIT universes yield 10-12%; a cap as
    # low as 0.06 would falsely accuse exactly those books
    r = _income_result(0.11)              # 91.7bp/bar < 125bp/bar cap
    assert r.status is Status.PASS
    assert "dividend" in r.message
    assert r.details["q90_signed_diff"] <= 1e-4


def test_twenty_pct_offset_not_laundered_as_dividends():
    # pins the cap < 0.20: a 167bp/bar one-sided return overstatement must
    # WARN, not pass as income (a cap of 0.23 would launder it as income)
    r = _income_result(0.20)
    assert r.status is Status.WARN
    assert r.details["median_abs_diff"] > MAX_PLAUSIBLE_ANNUAL_DIV_YIELD / 12


# ===========================================================================
# 6c. turnover FAIL bar: bracket 252x/yr from below and above
# ===========================================================================

def _turnover_result(one_sided):
    rng = np.random.default_rng(0)
    n = 700
    dates = pd.bdate_range("2018-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(6)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, 6)),
                        index=dates, columns=cols)
    w1 = np.zeros(6); w1[0] = 0.5; w1[1] = 0.5
    w2 = w1.copy(); w2[0] = 0.5 - one_sided; w2[2] = one_sided
    pos = pd.DataFrame([w1 if t % 2 == 0 else w2 for t in range(n)],
                       index=dates, columns=cols)
    sig = rets.rolling(5, min_periods=5).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos)
    art.validate()
    return _by(costs.run(art.aligned(), CFG))[C_TURNOVER]


def test_fail_bar_bracketed_from_below():
    # 0.9/day = 226.8x/yr sits in the calibrated WARN band; pinning it from
    # below keeps the FAIL bar from tightening toward 113x/yr, which would
    # upgrade honest WARN books to FAIL HIGH with a false "entire book
    # replaced every period" message
    r = _turnover_result(0.90)
    assert r.status is Status.WARN
    assert r.details["ann_turnover"] == pytest.approx(226.8, rel=0.02)


def test_fail_bar_bracketed_from_above():
    r = _turnover_result(1.30)            # 327.6x/yr > 252
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert r.details["ann_turnover"] == pytest.approx(327.6, rel=0.02)
