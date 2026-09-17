"""Embargo boundaries, position coverage, and frequency-aware cost checks.

Cost realism uses live trading spans and annualized turnover. Coarse bars
allow bounded dividend differences while scale and timing errors remain
visible. Invalid scalar and period values fail at input validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit._stats import gross_strategy_returns
from qaudit.checks import contamination, costs
from qaudit.config import AuditConfig
from qaudit.errors import InputValidationError, MisalignedInputError
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import momentum_signal, positions_from_signals, \
    simulate_market
from qaudit.types import Severity, Status

CFG = AuditConfig()

C_EMBARGO = "contamination.insufficient_embargo"
C_MISSING = "costs.missing_transaction_costs"
C_TURNOVER = "costs.turnover_unrealistic"
C_SENSITIVITY = "costs.cost_sensitivity"
C_PRICES = "costs.price_return_consistency"


def _by(results):
    return {r.check: r for r in results}


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _hand_artifacts(train, test, *, n=200, label_horizon=1, dates=None):
    dates = pd.bdate_range("2020-01-01", periods=n) if dates is None else dates
    cols = ["X", "Y", "Z"]
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (len(dates), len(cols))),
                        index=dates, columns=cols)
    sig = rets.rolling(5, min_periods=5).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            train_period=train, test_period=test,
                            label_horizon=label_horizon)
    art.validate()
    return art.aligned(), dates


def _alternating_book(dates, n_assets=6, one_sided_to=0.30, ppy=252, seed=0):
    """Book with exact one-sided turnover per period (alternating weights)."""
    rng = np.random.default_rng(seed)
    cols = [f"A{i:03d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (len(dates), n_assets)),
                        index=dates, columns=cols)
    w1 = np.zeros(n_assets); w1[0] = 0.5; w1[1] = 0.5
    w2 = w1.copy(); w2[0] = 0.5 - one_sided_to; w2[2] = one_sided_to
    pos = pd.DataFrame([w1 if t % 2 == 0 else w2 for t in range(len(dates))],
                       index=dates, columns=cols)
    sig = rets.rolling(5, min_periods=5).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            periods_per_year=ppy)
    art.validate()
    return art.aligned()


MONTHLY = pd.date_range("2010-01-31", periods=180, freq=pd.offsets.MonthEnd())
WEEKLY = pd.date_range("2012-01-06", periods=520, freq="W-FRI")
DAILY = pd.bdate_range("2018-01-02", periods=700)


# ===========================================================================
# 1. insufficient_embargo SKIP pointer (data-hole calendar overlap)
# ===========================================================================

def _hole_artifacts():
    dates = pd.bdate_range("2020-01-02", "2020-12-31")
    hole = (dates >= "2020-06-15") & (dates <= "2020-06-30")
    return _hand_artifacts(("2020-01-02", "2020-06-30"),
                           ("2020-06-15", "2020-12-31"),
                           dates=dates[~hole])


def test_data_hole_calendar_overlap_named_directly():
    art, _ = _hole_artifacts()
    res = _by(contamination.run(art, CFG))
    r = res[C_EMBARGO]
    assert r.status is Status.SKIP
    assert "overlap in calendar time" in r.message
    # a pointer to contamination.test_before_train would be wrong here: it
    # PASSes and explains nothing
    assert "test_before_train" not in r.message
    assert res["contamination.test_before_train"].status is Status.PASS
    assert res["contamination.split_overlap"].status is Status.PASS


def test_reversed_windows_keep_correct_pointer():
    # honest guard: test wholly before train - test_before_train WARNs
    # there, so the pointer is right and must stay
    dates = pd.bdate_range("2020-01-01", periods=200)
    art, _ = _hand_artifacts((dates[100], dates[199]), (dates[0], dates[99]))
    res = _by(contamination.run(art, CFG))
    r = res[C_EMBARGO]
    assert r.status is Status.SKIP
    assert "test_before_train" in r.message
    assert res["contamination.test_before_train"].status is Status.WARN


# ===========================================================================
# 2a. positions column-mass gate
# ===========================================================================

def _twelve_asset_book(seed=0):
    dates = pd.bdate_range("2019-01-02", periods=400)
    rng = np.random.default_rng(seed)
    cols = [f"A{i:03d}" for i in range(12)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (len(dates), 12)),
                        index=dates, columns=cols)
    sig = rets.rolling(5, min_periods=5).mean()
    pos = positions_from_signals(sig, 1)
    return sig, rets, pos


def test_vendor_suffix_mismatch_rejected():
    sig, rets, pos = _twelve_asset_book()
    bad = pos.copy()
    bad.columns = [c if i == 3 else c + " US Equity"
                   for i, c in enumerate(bad.columns)]
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=bad)
    with pytest.raises(MisalignedInputError) as exc:
        art.validate()
    msg = str(exc.value)
    assert "artifacts.positions" in msg
    assert "gross weight" in msg
    assert "US Equity" in msg          # names the dropped columns


def test_sub_universe_positions_still_validate():
    # honest guard: a book holding only 4 of 12 grid names has fewer
    # columns, not mismatched ones - 100% of its mass survives alignment
    sig, rets, pos = _twelve_asset_book()
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            positions=pos.iloc[:, :4].copy())
    art.validate()   # must not raise
    assert art.aligned().positions.abs().to_numpy().sum() > 0


def test_all_zero_positions_still_validate():
    # honest guard: a never-traded book has zero mass - the mass gate must
    # not divide by it or reject
    sig, rets, pos = _twelve_asset_book()
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            positions=pos * 0.0)
    art.validate()   # must not raise


# ===========================================================================
# 2b. turnover live-span trim (date-coverage dilution)
# ===========================================================================

def test_low_coverage_book_does_not_dilute_turnover():
    # Zero padding dilutes 40% daily turnover over a 31%-covered live span to 12.5% on
    # the full grid. Judge the live span.
    art = _alternating_book(DAILY, one_sided_to=0.40)
    art.positions.iloc[:int(0.69 * len(DAILY))] = 0.0
    r = _by(costs.run(art, CFG))[C_TURNOVER]
    assert r.status is Status.WARN
    assert r.details["mean_turnover"] == pytest.approx(0.40, abs=0.02)
    assert r.details["n_live_periods"] < r.details["n_grid_periods"]
    assert "live span" in r.message


def test_full_span_book_turnover_unaffected_by_trim():
    # honest guard: first and last rows trade, so the trim is a no-op
    art = _alternating_book(DAILY, one_sided_to=0.15)
    r = _by(costs.run(art, CFG))[C_TURNOVER]
    assert r.status is Status.PASS
    assert r.details["n_live_periods"] == r.details["n_grid_periods"]
    assert r.details["mean_turnover"] == pytest.approx(0.15, abs=0.01)
    assert "live span" not in r.message


def test_interior_flat_stretch_stays_in_the_mean():
    # honest guard: interior zero rows are genuine flat days, not padding -
    # they must keep diluting the mean (only leading/trailing rows trim)
    art = _alternating_book(DAILY, one_sided_to=0.40)
    lo, hi = int(0.2 * len(DAILY)), int(0.8 * len(DAILY))
    art.positions.iloc[lo:hi] = 0.0
    r = _by(costs.run(art, CFG))[C_TURNOVER]
    assert r.details["n_live_periods"] == r.details["n_grid_periods"]
    assert r.details["mean_turnover"] < 0.25   # interior flat days count


# ===========================================================================
# 3. validate() scalar-field garbage -> branded errors
# ===========================================================================

def _scalar_art(**kw):
    dates = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (len(dates), 3)),
                        index=dates, columns=list("XYZ"))
    sig = rets.rolling(5, min_periods=5).mean()
    return BacktestArtifacts(signals=sig, asset_returns=rets, **kw)


@pytest.mark.parametrize("ppy", [None, "252", True, float("nan"),
                                 float("inf"), 0, -252])
def test_bad_periods_per_year_raises_branded(ppy):
    with pytest.raises(InputValidationError, match="periods_per_year"):
        _scalar_art(periods_per_year=ppy).validate()


@pytest.mark.parametrize("dcb", ["5", True, float("nan"), -1.0])
def test_bad_declared_costs_raises_branded(dcb):
    with pytest.raises(InputValidationError, match="declared_costs_bps"):
        _scalar_art(declared_costs_bps=dcb).validate()


def test_real_number_scalars_still_accepted():
    # honest guards: 252.0 works downstream via float(); 5.0 bps
    # and None declarations are the documented contract
    _scalar_art(periods_per_year=252.0, declared_costs_bps=5.0).validate()
    _scalar_art(declared_costs_bps=None).validate()


# ===========================================================================
# 4. NaT period bounds rejected
# ===========================================================================

@pytest.mark.parametrize("period", [(None, None), ("2020-01-02", None),
                                    (np.nan, np.nan)])
@pytest.mark.parametrize("field", ["train_period", "test_period"])
def test_nat_period_bounds_rejected(field, period):
    with pytest.raises(InputValidationError, match=field):
        _scalar_art(**{field: period}).validate()


def test_whole_tuple_none_still_skips_family():
    # honest guard: period=None (undeclared) is the legitimate SKIP path
    art = _scalar_art(train_period=None, test_period=None)
    art.validate()
    res = contamination.run(art.aligned(), CFG)
    assert len(res) == 1
    assert res[0].check == "contamination.split_declared"
    assert res[0].status is Status.SKIP


# ===========================================================================
# 5. insufficient_embargo boundary pin (gap == needed-1 vs gap == needed)
# ===========================================================================

def test_embargo_one_period_short_warns():
    dates = pd.bdate_range("2020-01-01", periods=200)
    art, _ = _hand_artifacts((dates[0], dates[99]), (dates[104], dates[199]),
                             label_horizon=5)
    r = _by(contamination.run(art, CFG))[C_EMBARGO]
    assert r.status is Status.WARN
    assert r.details == {"gap": 4, "needed": 5}


def test_embargo_exactly_met_passes():
    dates = pd.bdate_range("2020-01-01", periods=200)
    art, _ = _hand_artifacts((dates[0], dates[99]), (dates[105], dates[199]),
                             label_horizon=5)
    r = _by(contamination.run(art, CFG))[C_EMBARGO]
    assert r.status is Status.PASS
    assert r.details == {"gap": 5, "needed": 5}


def test_embargo_override_boundary_pair():
    dates = pd.bdate_range("2020-01-01", periods=200)
    cfg = AuditConfig(min_embargo_periods=10)
    art, _ = _hand_artifacts((dates[0], dates[99]), (dates[109], dates[199]))
    r = _by(contamination.run(art, cfg))[C_EMBARGO]
    assert r.status is Status.WARN
    assert r.details == {"gap": 9, "needed": 10}
    art, _ = _hand_artifacts((dates[0], dates[99]), (dates[110], dates[199]))
    r = _by(contamination.run(art, cfg))[C_EMBARGO]
    assert r.status is Status.PASS
    assert r.details == {"gap": 10, "needed": 10}


# ===========================================================================
# 6. implied-cost ceiling pinned within ~20% (500bps constant)
# ===========================================================================

def _cost_book(drag_per_unit: float, seed: int = 4):
    market = simulate_market(seed=seed)
    rets = market["returns"]
    sig = momentum_signal(rets)
    pos = positions_from_signals(sig, lag=1)
    gross = gross_strategy_returns(pos, rets)
    traded = pos.diff().abs().sum(axis=1)
    traded.iloc[0] = pos.iloc[0].abs().sum()
    net = gross - traded * drag_per_unit
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            strategy_returns=net, signal_lag=1)
    art.validate()
    return art.aligned()


def test_600bps_implied_cost_warns_ceiling():
    # 600bps must WARN so the 500bps ceiling is pinned from above (2500bps
    # alone would leave the constant free to drift)
    r = _by(costs.run(_cost_book(0.06), CFG))[C_MISSING]
    assert r.status is Status.WARN
    assert "ceiling" in r.message


def test_420bps_implied_cost_passes_below_ceiling():
    # ...and 420bps must stay a clean PASS, pinning it from below
    r = _by(costs.run(_cost_book(0.042), CFG))[C_MISSING]
    assert r.status is Status.PASS
    assert "ceiling" not in r.message


# ===========================================================================
# 7. turnover_unrealistic judges annualized turnover
# ===========================================================================

def test_honest_monthly_book_passes_turnover():
    # One-sided turnover of 28% per month is about 3.4 times per year and must be
    # judged on that annual basis.
    art = _alternating_book(MONTHLY, one_sided_to=0.28, ppy=12)
    r = _by(costs.run(art, CFG))[C_TURNOVER]
    assert r.status is Status.PASS
    assert r.details["ann_turnover"] == pytest.approx(0.28 * 12, rel=0.1)


def test_monthly_full_replacement_does_not_fail():
    # One-sided turnover of 100% per month is 12 times per year; the annualized
    # boundary determines severity.
    art = _alternating_book(MONTHLY, one_sided_to=0.50, ppy=12)
    art.positions.iloc[1::2] = -art.positions.iloc[1::2] + 0.0  # keep alt
    r = _by(costs.run(art, CFG))[C_TURNOVER]
    assert r.status is not Status.FAIL
    assert r.details["ann_turnover"] < CFG.turnover_daily_warn * 252


def test_daily_detection_unchanged():
    # detection guard: the annualized bars are the exact annualization of
    # the daily defaults, so daily books see bit-identical thresholds
    r = _by(costs.run(_alternating_book(DAILY, one_sided_to=0.40), CFG))[
        C_TURNOVER]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH


def test_weekly_boundary_pins_annualized_bar():
    # long-short weekly book straddling the 63x/yr bar (63/52 = 1.21/period)
    def _ls_book(one_sided):
        rng = np.random.default_rng(3)
        cols = list("ABCD")
        rets = pd.DataFrame(rng.normal(0.0, 0.01, (len(WEEKLY), 4)),
                            index=WEEKLY, columns=cols)
        half = one_sided / 2.0
        w1 = np.array([half, -half, 0.0, 0.0])
        w2 = np.array([0.0, 0.0, half, -half])
        pos = pd.DataFrame([w1 if t % 2 == 0 else w2
                            for t in range(len(WEEKLY))],
                           index=WEEKLY, columns=cols)
        sig = rets.rolling(5, min_periods=5).mean()
        art = BacktestArtifacts(signals=sig, asset_returns=rets,
                                positions=pos, periods_per_year=52)
        art.validate()
        return _by(costs.run(art.aligned(), CFG))[C_TURNOVER]

    assert _ls_book(1.30).status is Status.WARN    # 67.6x/yr > 63
    assert _ls_book(1.15).status is Status.PASS    # 59.8x/yr < 63


# ===========================================================================
# 8. price_return_consistency: dividend signature at coarse bars
# ===========================================================================

def _monthly_income(seed=0, annual_yield=0.05, n=48, n_assets=15, vol=0.04):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n, freq=pd.offsets.MonthEnd())
    cols = [f"F{i:02d}" for i in range(n_assets)]
    price_ret = pd.DataFrame(rng.normal(0.004, vol, (n, n_assets)),
                             index=dates, columns=cols)
    total = price_ret + annual_yield / 12.0
    px = pd.DataFrame(100.0 * (1.0 + price_ret).cumprod(),
                      index=dates, columns=cols)
    return px, price_ret, total


def _price_result(prices, declared_returns, ppy=12):
    sig = declared_returns.rolling(3, min_periods=3).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=declared_returns,
                            prices=prices, periods_per_year=ppy)
    art.validate()
    return _by(costs.run(art.aligned(), CFG))[C_PRICES]


def test_monthly_income_book_passes():
    # A 5% annual monthly payer yields about 42 bp of one-sided difference per bar
    # between close-only prices and total returns.
    px, _, total = _monthly_income()
    r = _price_result(px, total)
    assert r.status is Status.PASS
    assert "dividend" in r.message
    assert r.details["median_abs_diff"] > 1e-4       # the mass is there
    assert r.details["q90_signed_diff"] <= 1e-4      # ...and one-sided


def test_monthly_percent_vs_decimal_still_warns():
    # Use a low-volatility scaled panel to exercise price consistency separately from
    # intake's -100% asset-return floor.
    px, _, total = _monthly_income()
    with pytest.raises(InputValidationError, match="below -1.0"):
        _price_result(px, total * 100.0)
    px, _, total = _monthly_income(vol=0.002)
    r = _price_result(px, total * 100.0)
    assert r.status is Status.WARN


def test_monthly_date_shift_still_warns():
    px, _, total = _monthly_income()
    r = _price_result(px, total.shift(1))
    assert r.status is Status.WARN


def test_total_adjusted_prices_vs_price_only_returns_warns():
    # broken research input: prices carry the dividends, declared returns
    # do not - one-sided positive offset, outside the dividend signature
    _, price_ret, total = _monthly_income()
    px_adj = pd.DataFrame(
        100.0 * (1.0 + total).cumprod(),
        index=total.index, columns=total.columns)
    r = _price_result(px_adj, price_ret)
    assert r.status is Status.WARN


def test_above_cap_yield_still_warns():
    # 24%/yr = 200bp/bar exceeds the 15%/yr plausible-yield cap: the
    # signature must not launder arbitrarily large one-sided offsets
    px, price_ret, _ = _monthly_income()
    r = _price_result(px, price_ret + 0.24 / 12.0)
    assert r.status is Status.WARN


# ===========================================================================
# 9. cost_sensitivity PASS message truthfulness
# ===========================================================================

def _churny_gross(mu):
    # full-swap churn; the held asset returns mu+1% on even bars (X) and
    # mu-1% on odd bars (Y), so gross SR_ann = mu/0.01 * sqrt(252)
    dates = pd.bdate_range("2019-01-02", periods=400)
    cols = ["X", "Y"]
    rets = pd.DataFrame({"X": mu + 0.01, "Y": mu - 0.01},
                        index=dates)[cols]
    pos = pd.DataFrame(
        [(1.0, 0.0) if t % 2 == 0 else (0.0, 1.0)
         for t in range(len(dates))], index=dates, columns=cols)
    sig = rets.rolling(5, min_periods=5).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos)
    art.validate()
    return _by(costs.run(art.aligned(), CFG))[C_SENSITIVITY]


def test_low_gross_flip_message_does_not_claim_no_flip():
    # gross SR ~0.22 (below the 0.5 warn gate) with the sweep deeply
    # negative: verdict stays PASS by design, but the message must not
    # assert "the edge does not flip negative"
    r = _churny_gross(mu=0.0001)
    assert r.status is Status.PASS
    assert 0 < r.details["gross_sr"] <= 0.5
    assert r.details["sr_at_mid_bps"] <= 0
    assert "does not flip" not in r.message
    assert "no material gross edge" in r.message
    assert "25bps" in r.message                     # sweep table retained


def test_negative_gross_message_states_nothing_to_flip():
    r = _churny_gross(mu=-0.0001)
    assert r.status is Status.PASS
    assert r.details["gross_sr"] <= 0
    assert "does not flip" not in r.message
    assert "non-positive" in r.message


def test_no_flip_claim_reserved_for_surviving_edge():
    # honest guard: a low-turnover book whose Sharpe stays positive at the
    # mid sweep point keeps the no-flip claim
    dates = pd.bdate_range("2019-01-02", periods=400)
    cols = ["X", "Y"]
    ret_x = np.where(np.arange(len(dates)) % 2 == 0, 0.011, -0.009)
    rets = pd.DataFrame({"X": ret_x, "Y": 0.0}, index=dates)[cols]
    pos = pd.DataFrame([(1.0, 0.0)] * len(dates), index=dates, columns=cols)
    sig = rets.rolling(5, min_periods=5).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos)
    art.validate()
    r = _by(costs.run(art.aligned(), CFG))[C_SENSITIVITY]
    assert r.status is Status.PASS
    assert r.details["sr_at_mid_bps"] > 0
    assert "does not flip negative" in r.message
