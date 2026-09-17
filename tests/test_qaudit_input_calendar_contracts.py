"""Input calendar, label, dtype, cadence, and check-filter contracts.

Holdover grace follows observed rebalance cadence within calendar bounds.
Membership coverage, nullable numeric types, period frequency, and family
placeholders remain explicit; cost messages reflect sibling verdicts.
"""
from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from qaudit.api import audit
from qaudit.checks import costs, survivorship
from qaudit.checks.survivorship import (HOLDOVER_CADENCE_SLACK_BARS,
                                        HOLDOVER_GRACE_BARS)
from qaudit.config import AuditConfig
from qaudit.errors import InputValidationError, MisalignedInputError
from qaudit.inputs import (_PPY_TOLERANCE_FACTOR, _UNIVERSE_MIN_COVERAGE,
                           BacktestArtifacts)
from qaudit.synthetic import (momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

DEFAULT_DAILY_HOLDOVER_CAP = 252 + HOLDOVER_CADENCE_SLACK_BARS

CFG = AuditConfig()
TOU = "survivorship.trading_outside_universe"


def _by(results):
    return {r.check: r for r in results}


@pytest.fixture(scope="module")
def market():
    return simulate_market(n_assets=12, n_periods=400, seed=17, death_frac=0.0)


@pytest.fixture(scope="module")
def honest(market):
    return momentum_signal(market["returns"])


def _art(market, sig, **kw):
    base = dict(signals=sig, asset_returns=market["returns"],
                positions=positions_from_signals(sig, 1),
                universe=pd.DataFrame(True, index=market["returns"].index,
                                      columns=market["returns"].columns),
                signal_lag=1)
    base.update(kw)
    return BacktestArtifacts(**base)


# ===========================================================================
# 1. cadence-scaled holdover grace
# ===========================================================================

def _rebalanced_book(seed=3, n_assets=20, n_periods=756, cycle=63,
                     zombie=False):
    """Honest PIT book rebalanced every ``cycle`` bars: deleted names get
    zero weight at the next rebalance, never sooner. zombie=True pins one
    column held forever after its universe exit (hindsight book)."""
    rng = np.random.default_rng(seed)
    m = simulate_market(n_assets=n_assets, n_periods=n_periods, seed=seed,
                        death_frac=0.0)
    rets = m["returns"]
    uni = pd.DataFrame(True, index=rets.index, columns=rets.columns)
    for j in range(n_assets):                     # ~6%/yr index deletions
        t = 60
        while t < n_periods:
            if rng.random() < 0.06 / 252:
                out_len = int(rng.integers(126, 400))
                uni.iloc[t:t + out_len, j] = False
                t += out_len
            t += 1
    sig = momentum_signal(rets)
    reb = (np.arange(n_periods) % cycle) == 0
    in_uni = uni.shift(1, fill_value=False)
    lagged = sig.shift(1).where(in_uni)           # PIT decisions only
    r = lagged.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    # Record weights actually held: targets are installed on rebalance bars;
    # between them the book passively drifts and trades zero dollars.
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for t in range(n_periods):
        if reb[t]:
            pos.iloc[t] = w.iloc[t]
        elif t:
            prev = pos.iloc[t - 1].to_numpy()
            rr = rets.iloc[t - 1].to_numpy()
            pos.iloc[t] = prev * (1.0 + rr) / (1.0 + prev @ rr)
    if zombie:
        uni.iloc[300:, 0] = False                 # exits for good...
        pos.iloc[:, 0] = 0.05                     # ...held forever anyway
    return BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                             universe=uni, signal_lag=1)


def test_honest_quarterly_rebalanced_pit_book_passes():
    # A 63-bar rebalance book passively holds exited names until the next rebalance;
    # cadence-aware grace must preserve that behavior.
    art = _rebalanced_book(cycle=63)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))[TOU]
    assert r.status is Status.PASS, r.message
    assert r.details["rebalance_cadence_bars"] == 63
    assert r.details["holdover_grace_bars"] == 63 + HOLDOVER_CADENCE_SLACK_BARS


def test_zombie_on_quarterly_book_still_fails_critical():
    # detection guard: the same cadence gets no shelter for a name held far
    # past its exit with no rebalance response.
    art = _rebalanced_book(zombie=True)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))[TOU]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_zombie_cells"] > 100


def test_pure_zombie_message_does_not_claim_enter_increase_reverse(market, honest):
    # message-truth pin: a conviction made of pure holds must be described
    # as zombie holdings, never as entries/increases/reversals.
    uni = pd.DataFrame(True, index=market["returns"].index,
                       columns=market["returns"].columns)
    uni.iloc[100:, 0] = False
    rets = market["returns"]
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos.iloc[0] = 1.0 / 12
    for t in range(1, len(pos)):
        prev = pos.iloc[t - 1].to_numpy()
        rr = rets.iloc[t - 1].fillna(0.0).to_numpy()
        pos.iloc[t] = prev * (1.0 + rr) / (1.0 + prev @ rr)
    art = _art(market, honest, universe=uni, positions=pos)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))[TOU]
    assert r.status is Status.FAIL
    assert "ENTER, INCREASE, or REVERSE" not in r.message
    assert re.search(r"held more than \d+ bars past", r.message)
    assert "0 entered/increased/sign-flipped" in r.message
    assert r.details["n_active_cells"] == 0
    assert r.details["n_zombie_cells"] == r.details["n_violations"]
    # constant book has no observable cadence -> grace falls back to 21
    assert r.details["rebalance_cadence_bars"] == 0
    assert r.details["holdover_grace_bars"] == HOLDOVER_GRACE_BARS


def test_mixed_prongs_message_names_both(market, honest):
    uni = pd.DataFrame(True, index=market["returns"].index,
                       columns=market["returns"].columns)
    uni.iloc[100:, 0] = False                     # zombie hold on A000
    uni.iloc[100:, 1] = False                     # fresh entry on A001
    pos = pd.DataFrame(1.0 / 12, index=market["returns"].index,
                       columns=market["returns"].columns)
    pos.iloc[:150, 1] = 0.0
    pos.iloc[150:, 1] = 0.05                      # entered 50 bars post-exit
    art = _art(market, honest, universe=uni, positions=pos)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))[TOU]
    assert r.status is Status.FAIL
    assert "ENTER, INCREASE, or REVERSE" in r.message
    assert re.search(r"plus \d+ cells held > \d+ bars past", r.message)
    assert r.details["n_active_cells"] >= 1
    assert r.details["n_zombie_cells"] >= 1


def test_cadence_grace_is_capped(market, honest):
    # A book trading every 300 bars receives at most annual-plus-slack holdover grace,
    # bounded by DEFAULT_DAILY_HOLDOVER_CAP.
    uni = pd.DataFrame(True, index=market["returns"].index,
                       columns=market["returns"].columns)
    uni.iloc[60:, 0] = False
    rets = market["returns"]
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos.iloc[0] = 1.0 / 12
    for t in range(1, len(pos)):
        prev = pos.iloc[t - 1].to_numpy()
        rr = rets.iloc[t - 1].fillna(0.0).to_numpy()
        pos.iloc[t] = prev * (1.0 + rr) / (1.0 + prev @ rr)
        if t == 50:
            pos.iloc[t, 2] = 0.09                 # trade bars 50 and 350
        elif t == 350:
            pos.iloc[t, 2] = 0.07                 # -> cadence 300
    art = _art(market, honest, universe=uni, positions=pos)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))[TOU]
    assert r.status is Status.FAIL                # held ~340 bars past exit
    assert r.details["rebalance_cadence_bars"] == 300
    assert r.details["holdover_grace_bars"] == DEFAULT_DAILY_HOLDOVER_CAP


# ===========================================================================
# 2. universe column-coverage gate
# ===========================================================================

def test_universe_missing_columns_rejected_with_dual_hint(market, honest):
    # Renaming half the membership columns must raise an actionable mismatch error
    # before alignment loses the evidence.
    uni = pd.DataFrame(True, index=market["returns"].index,
                       columns=market["returns"].columns)
    renamed = uni.rename(columns={c: f"{c}.US" for c in uni.columns[:6]})
    art = _art(market, honest, universe=renamed)
    with pytest.raises(MisalignedInputError) as exc:
        art.validate()
    msg = str(exc.value)
    assert "artifacts.universe" in msg
    assert "6 of 12" in msg
    assert "A000" in msg                          # names the uncovered assets
    assert "label mismatches" in msg              # hint 1: align tickers
    assert "all-False" in msg                     # hint 2: declare explicitly
    assert "fillna(False)" in msg                 # names the mechanism


def test_universe_all_nan_column_rejected(market, honest):
    uni = pd.DataFrame(1.0, index=market["returns"].index,
                       columns=market["returns"].columns)
    uni.iloc[:, 3] = np.nan                       # present but no data
    art = _art(market, honest, universe=uni)
    with pytest.raises(MisalignedInputError, match="all-NaN column"):
        art.validate()


def test_explicit_all_false_columns_keep_detection(market, honest):
    # true-positive guard: a book trading names declared never-in-
    # universe (explicit all-False columns) must fail CRITICAL.
    uni = pd.DataFrame(True, index=market["returns"].index,
                       columns=market["returns"].columns)
    uni.iloc[:, :2] = False
    art = _art(market, honest, universe=uni)
    art.validate()                                # explicit declaration: valid
    r = _by(survivorship.run(art.aligned(), CFG))[TOU]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL


# ===========================================================================
# 3. MultiIndex columns rejected at validate()
# ===========================================================================

@pytest.mark.parametrize("outer", ["constant", "varying"])
def test_multiindex_columns_rejected_with_flatten_hint(market, honest, outer):
    # constant outer level = the yf.download [('Close', ticker)] shape, which
    # would otherwise reach costs.price_return_consistency unflattened;
    # varying outer (sector labels) would silently empty it to a bogus SKIP.
    # Both must be rejected at validate().
    px = market["prices"].copy()
    if outer == "constant":
        px.columns = pd.MultiIndex.from_product([["Close"], px.columns])
    else:
        sectors = ["FIN", "TEC"] * (len(px.columns) // 2)
        px.columns = pd.MultiIndex.from_arrays([sectors, px.columns])
    art = _art(market, honest, prices=px)
    with pytest.raises(InputValidationError) as exc:
        art.validate()
    msg = str(exc.value)
    assert "artifacts.prices" in msg
    assert "MultiIndex" in msg
    assert "get_level_values(-1)" in msg          # actionable flatten hint


def test_flat_columns_prices_still_validate(market, honest):
    _art(market, honest, prices=market["prices"]).validate()


# ===========================================================================
# 4. nullable / pyarrow dtypes
# ===========================================================================

def _nullable_book(market, honest):
    pos = positions_from_signals(honest, 1)
    strat = net_returns(pos, market["returns"], 10.0)
    uni = pd.DataFrame(True, index=market["returns"].index,
                       columns=market["returns"].columns)
    return dict(signals=honest, asset_returns=market["returns"],
                positions=pos, strategy_returns=strat, universe=uni,
                prices=market["prices"], signal_lag=1,
                declared_costs_bps=10.0)


def test_convert_dtypes_artifacts_audit_like_float64(market, honest):
    kw = _nullable_book(market, honest)
    base = audit(BacktestArtifacts(**kw))
    kw_n = dict(kw)
    for name in ("signals", "asset_returns", "positions", "universe", "prices"):
        kw_n[name] = kw[name].convert_dtypes()
    kw_n["strategy_returns"] = kw["strategy_returns"].convert_dtypes()
    assert str(kw_n["signals"].dtypes.iloc[0]) == "Float64"    # premise
    assert str(kw_n["universe"].dtypes.iloc[0]) == "boolean"
    rep = audit(BacktestArtifacts(**kw_n))       # must not raise a raw TypeError
    assert ({r.check: r.status for r in rep.results}
            == {r.check: r.status for r in base.results})


def test_inf_in_nullable_frame_still_caught(market, honest):
    # object-backed frames: a to_numpy() yielding dtype=object would skip
    # the inf scan; the coerced scan must still catch it and name the frame.
    rets = market["returns"].convert_dtypes()
    rets.iloc[5, 0] = np.inf
    art = _art(market, honest)
    art = dataclasses.replace(art, asset_returns=rets)
    with pytest.raises(InputValidationError, match="asset_returns.*infinite"):
        art.validate()


def test_nullable_series_and_bool_series_contract(market, honest):
    art = _art(market, honest)
    sr = pd.Series(0.001, index=market["returns"].index).convert_dtypes()
    dataclasses.replace(art, strategy_returns=sr.astype("Float64")).validate()
    bad = pd.Series(True, index=market["returns"].index)
    with pytest.raises(InputValidationError, match="strategy_returns"):
        dataclasses.replace(art, strategy_returns=bad).validate()


def test_pyarrow_backed_frames_validate(market, honest):
    pytest.importorskip("pyarrow")
    art = _art(market, honest)
    art = dataclasses.replace(
        art, asset_returns=market["returns"].astype("float64[pyarrow]"))
    art.validate()                                # must not raise a raw TypeError
    a = art.aligned()
    assert a.asset_returns.dtypes.eq("float64").all()


def test_aligned_coerces_extension_dtypes_to_numpy(market, honest):
    kw = _nullable_book(market, honest)
    for name in ("signals", "asset_returns", "positions", "prices"):
        kw[name] = kw[name].convert_dtypes()
    kw["strategy_returns"] = kw["strategy_returns"].convert_dtypes()
    art = BacktestArtifacts(**kw)
    art.validate()
    a = art.aligned()
    for name in ("signals", "asset_returns", "positions", "prices"):
        assert getattr(a, name).dtypes.eq("float64").all(), name
    assert a.strategy_returns.dtype == "float64"
    assert a.universe.dtypes.eq(bool).all()


# ===========================================================================
# 5. include filter keeps the contamination SKIP placeholder
# ===========================================================================

@pytest.mark.parametrize("sub_id", [
    "contamination.split_overlap", "contamination.insufficient_embargo",
    "contamination.test_before_train", "contamination.split_coverage"])
def test_include_sub_check_without_split_keeps_placeholder(market, honest, sub_id):
    # an empty report (summary CLEAN "0 checks", ok=True) would be a silent
    # fail-open for a CI gate on an explicitly requested check.
    art = _art(market, honest, universe=None, positions=None)
    rep = audit(art, include=[sub_id])
    assert rep.results, f"include=[{sub_id!r}] returned an empty report"
    r = _by(rep.results)["contamination.split_declared"]
    assert r.status is Status.SKIP
    assert "train_period" in r.message            # says exactly what to pass


def test_include_sub_check_with_declared_overlap_still_fails(market, honest):
    idx = market["returns"].index
    art = _art(market, honest, universe=None, positions=None,
               train_period=(idx[0], idx[260]), test_period=(idx[200], idx[-1]))
    rep = audit(art, include=["contamination.split_overlap"])
    r = _by(rep.results)["contamination.split_overlap"]
    assert r.status is Status.FAIL
    assert not rep.ok


def test_include_no_exits_does_not_leak_sibling_skips(market, honest):
    # the re-admission must be surgical: a matched pattern re-admits nothing.
    rep = audit(_art(market, honest), include=["survivorship.no_exits"])
    assert {r.check for r in rep.results} == {"survivorship.no_exits"}


# ===========================================================================
# 6. universe coverage floor: value + mid-band pins
# ===========================================================================

def test_universe_coverage_floor_value_pinned():
    # Pin both the coverage constant and intermediate coverage behavior; too-low
    # thresholds admit fabricated membership.
    assert _UNIVERSE_MIN_COVERAGE == 0.60


def test_mid_band_dense_back_40pct_universe_rejected(market, honest):
    uni = pd.DataFrame(True, index=market["returns"].index,
                       columns=market["returns"].columns)
    uni = uni.iloc[int(0.60 * len(uni)):]         # dense, back 40% only
    art = _art(market, honest, universe=uni)
    with pytest.raises(MisalignedInputError, match="dense but spans"):
        art.validate()


def test_mid_band_two_of_five_day_stamps_rejected(market, honest):
    idx = market["returns"].index
    uni = pd.DataFrame(True, index=idx, columns=market["returns"].columns)
    uni = uni.loc[idx[idx.weekday.isin([1, 3])]]  # ~40% sparse stamping
    art = _art(market, honest, universe=uni)
    with pytest.raises(MisalignedInputError, match="ffill"):
        art.validate()


# ===========================================================================
# 7. periods_per_year vs observed bar frequency
# ===========================================================================

def _ppy_art(idx, ppy, n_assets=8, seed=0):
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(rng.normal(0.0, 0.02, (len(idx), n_assets)),
                        index=idx, columns=[f"A{i:02d}" for i in range(n_assets)])
    sig = rets.rolling(3, min_periods=3).mean()
    return BacktestArtifacts(signals=sig, asset_returns=rets,
                             periods_per_year=ppy, signal_lag=1)


@pytest.mark.parametrize("ppy", [126, 63, 52, 40])
def test_underdeclared_ppy_on_daily_bars_rejected(ppy):
    # the Sharpe-deflation attack: declared 63 on daily bars halves a
    # true-daily-SR-9 leak below the suspicious_sharpe FAIL bar.
    idx = pd.bdate_range("2018-01-02", periods=500)
    with pytest.raises(InputValidationError) as exc:
        _ppy_art(idx, ppy).validate()
    msg = str(exc.value)
    assert f"periods_per_year={ppy}" in msg
    assert re.search(r"~\d+ bars/year", msg)      # states the measured rate


@pytest.mark.parametrize("freq,kw,ppy", [
    ("bdate", dict(periods=500), 252),
    ("bdate", dict(periods=500), 365),            # crypto-style declaration
    ("D", dict(periods=500), 365),
    ("W-FRI", dict(periods=200), 52),
    (pd.offsets.BMonthEnd(), dict(periods=60), 12),
    (pd.offsets.BQuarterEnd(startingMonth=12), dict(periods=40), 4),
])
def test_honest_calendars_with_correct_ppy_validate(freq, kw, ppy):
    idx = (pd.bdate_range("2015-01-02", **kw)
           if isinstance(freq, str) and freq == "bdate"
           else pd.date_range("2015-01-02", freq=freq, **kw))
    _ppy_art(idx, ppy).validate()                 # must not raise


def test_default_ppy_on_weekly_bars_rejected():
    # the other direction: the 252 default left on weekly bars inflates SR
    # by sqrt(252/52) = 2.2x -> false FAILs on honest books.
    idx = pd.date_range("2012-01-06", periods=200, freq="W-FRI")
    with pytest.raises(InputValidationError, match="periods_per_year=252"):
        _ppy_art(idx, 252).validate()


def test_ppy_tolerance_factor_pinned_between_honest_and_attack():
    # calibration band: the widest honest gap is declared-365 on a business-
    # day grid (365/~261 = 1.40x); the nearest attack is 126 on daily bars
    # (~261/126 = 2.07x). The factor must sit strictly between.
    assert 1.45 < _PPY_TOLERANCE_FACTOR < 2.0


# ===========================================================================
# 8. costs.no_cost_declaration wording matches the reconstruction verdict
# ===========================================================================

def _cost_book(seed=2, n_periods=300, n_assets=10):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_periods)
    assets = [f"A{i:02d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n_periods, n_assets)),
                        index=dates, columns=assets)
    sig = rets.rolling(5, min_periods=5).mean()
    pos = positions_from_signals(sig, 1)
    return rets, sig, pos


def _decl_result(rets, sig, pos, strat):
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            strategy_returns=strat, signal_lag=1,
                            declared_costs_bps=None)
    art.validate()
    out = _by(costs.run(art.aligned(), CFG))
    return out["costs.missing_transaction_costs"], out["costs.no_cost_declaration"]


def test_declaration_says_verified_only_on_reconstruction_pass():
    rets, sig, pos = _cost_book()
    from qaudit._stats import gross_strategy_returns
    gross = gross_strategy_returns(pos, rets)
    traded = pos.diff().abs().sum(axis=1)
    traded.iloc[0] = pos.iloc[0].abs().sum()
    r_missing, r_decl = _decl_result(rets, sig, pos, gross - traded * 10e-4)
    assert r_missing.status is Status.PASS
    assert r_decl.status is Status.PASS
    assert "verified against the gross reconstruction" in r_decl.message
    assert r_decl.details["verified_by_reconstruction"] is True
    assert r_decl.details["reconstruction_check_status"] == "pass"


def test_selected_date_strategy_returns_rejected_before_declaration_check():
    # strategy_returns reported only on no-trade dates has hundreds of finite
    # values but interior holes exactly on every trade. Intake must reject the
    # selected-date series before a cost/declaration result can imply it was a
    # contiguous return history.
    rets, sig, pos = _cost_book()
    flip = pd.Series((np.arange(len(rets)) // 5) % 2 * 2 - 1, index=rets.index)
    # Weekly position changes leave enough finite no-trade net dates to clear the
    # 30-observation intake floor while exercising an unmeasurable cost comparison.
    pos = pos.iloc[::5].reindex(rets.index, method="ffill").mul(flip, axis=0)
    traded = pos.diff().abs().sum(axis=1).fillna(0.0)
    strat = pd.Series(0.001, index=rets.index).where(traded == 0.0)
    assert strat.notna().sum() > 200
    with pytest.raises(MisalignedInputError) as ei:
        _decl_result(rets, sig, pos, strat)
    msg = str(ei.value)
    assert "strategy_returns is not dense" in msg
    assert "hide losing returns" in msg


def test_declaration_does_not_claim_verified_when_reconstruction_fails():
    rets, sig, pos = _cost_book()
    from qaudit._stats import gross_strategy_returns
    strat = gross_strategy_returns(pos, rets)     # net == gross: no costs
    r_missing, r_decl = _decl_result(rets, sig, pos, strat)
    assert r_missing.status is Status.FAIL
    assert r_decl.status is Status.PASS           # one defect, one check
    assert "was verified" not in r_decl.message
    assert re.search(r"reconstruction", r_decl.message)
    assert r_decl.details["verified_by_reconstruction"] is False
    assert r_decl.details["reconstruction_check_status"] == "fail"
