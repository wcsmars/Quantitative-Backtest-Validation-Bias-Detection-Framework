"""Detector evasion cases paired with causal and unbiased control books.

Individual check modules exercise dilution, timing, cost, and universe
boundaries. API validation and dispatch are covered where the boundary
cannot be tested inside an individual detector.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from qaudit._stats import forward_returns
from qaudit.api import CHECK_MODULES, MODULE_CHECK_IDS, audit
from qaudit.checks import costs, leakage, lookahead, survivorship
from qaudit.config import AuditConfig
from qaudit.dynamic import probes_shift
from qaudit.errors import InputValidationError, MisalignedInputError
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (make_backtest_func, momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()


@pytest.fixture(scope="module")
def market():
    return simulate_market(seed=2)


@pytest.fixture(scope="module")
def honest(market):
    return momentum_signal(market["returns"])


def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def _arts(market, sig, pos, **kw):
    rets = market["returns"]
    base = dict(signals=sig, asset_returns=rets, positions=pos,
                strategy_returns=net_returns(pos, rets, 10.0),
                universe=market.get("universe"), signal_lag=1,
                declared_costs_bps=10.0)
    base.update(kw)
    art = BacktestArtifacts(**base)
    art.validate()
    return art.aligned()


def _by(results):
    return {r.check: r for r in results}


# ---------------------------------------------------------------------------
# mild same-bar bleed (15% of the book from signal_t, declared lag 1):
#   slips the alignment-profile margin; caught by the partial-corr detector.
# ---------------------------------------------------------------------------

def _blend_positions(sig, w):
    pos = (1 - w) * positions_from_signals(sig, 1) + w * positions_from_signals(sig, 0)
    gross = pos.abs().sum(axis=1)
    return pos.div(gross.replace(0, np.nan), axis=0).fillna(0.0)


def test_mild_same_bar_bleed_caught(market, honest):
    art = _arts(market, honest, _blend_positions(honest, 0.15))
    res = _by(lookahead.run(art, CFG))
    assert res["lookahead.same_bar_bleed"].status is Status.FAIL
    # the coarse alignment profile alone does not see it (that is the point)
    assert res["lookahead.position_signal_alignment"].status is Status.PASS


def test_same_bar_bleed_honest_nonlinear_book_passes(market, honest):
    # top/bottom-decile book: nonlinear but honest (pure f(sig_{t-1}))
    s = honest.shift(1)
    pct = s.rank(axis=1, pct=True)
    pos = (pct >= 0.8).astype(float) - (pct <= 0.2).astype(float)
    gross = pos.abs().sum(axis=1)
    pos = pos.div(gross.replace(0, np.nan), axis=0).fillna(0.0)
    art = _arts(market, honest, pos)
    assert _by(lookahead.run(art, CFG))["lookahead.same_bar_bleed"].status is Status.PASS


# ---------------------------------------------------------------------------
# future-vol sizing (centered 11-bar vol window): honest ranks, leaky sizes.
# ---------------------------------------------------------------------------

def _vol_scaled(base, vol):
    s = base / vol.clip(lower=1e-4)
    gross = s.abs().sum(axis=1)
    return s.div(gross.replace(0, np.nan), axis=0).fillna(0.0)


def test_centered_vol_sizing_caught(market, honest):
    rets = market["returns"]
    pos = _vol_scaled(positions_from_signals(honest, 1),
                      rets.rolling(11, center=True, min_periods=7).std())
    art = _arts(market, honest, pos)
    assert _by(lookahead.run(art, CFG))["lookahead.future_vol_sizing"].status is Status.FAIL


def test_trailing_vol_sizing_passes(market, honest):
    rets = market["returns"]
    pos = _vol_scaled(positions_from_signals(honest, 1),
                      rets.rolling(11, min_periods=7).std().shift(1))
    art = _arts(market, honest, pos)
    assert _by(lookahead.run(art, CFG))["lookahead.future_vol_sizing"].status is Status.PASS


# ---------------------------------------------------------------------------
# diluted tail leak (true next-day return on 5% of high-dispersion dates,
#   noise-diluted to |IC|~0.5): invisible to median/perfect-rank statistics.
# ---------------------------------------------------------------------------

def test_diluted_tail_leak_caught(market, honest):
    rets = market["returns"]
    rng = np.random.default_rng(7)
    fwd = rets.shift(-1)
    noise = pd.DataFrame(rng.normal(size=fwd.shape), index=fwd.index,
                         columns=fwd.columns)
    leak_sig = _zx(fwd) + 1.7 * noise
    hot = fwd.std(axis=1).rank(pct=True) > 0.95
    sig = honest.copy()
    sig[hot] = leak_sig[hot].where(honest[hot].notna())
    art = _arts(market, sig, positions_from_signals(sig, 1))
    res = _by(leakage.run(art, CFG))
    assert res["leakage.ic_outlier_dates"].status is Status.FAIL
    # and the blunter statistics indeed stay quiet - the outlier counter is
    # the only line of defense here
    assert res["leakage.target_correlation"].status is Status.PASS


def test_ic_outlier_dates_clean_passes(market, honest):
    art = _arts(market, honest, positions_from_signals(honest, 1))
    assert _by(leakage.run(art, CFG))["leakage.ic_outlier_dates"].status is Status.PASS


# ---------------------------------------------------------------------------
# deferred-window leak (signal contains compound r over t+2..t+6, skipping
#   t+1) held weekly: IC(h=1) is tiny, PnL is huge.
# ---------------------------------------------------------------------------

def test_deferred_window_leak_caught(market, honest):
    rets = market["returns"]
    f6, f1 = forward_returns(rets, 6), forward_returns(rets, 1)
    slow = (1 + f6).div(1 + f1) - 1
    sig = (0.5 * _zx(slow) + 0.5 * honest.fillna(0)).where(honest.notna())
    art = _arts(market, sig, positions_from_signals(sig, 1))
    res = _by(lookahead.run(art, CFG))["lookahead.deferred_ic_spike"]
    # The full-strength deferred leak must FAIL at CRITICAL severity; a WARN would
    # leave the severity threshold unprotected. The spike and ratio are independently
    # bounded below.
    assert res.status is Status.FAIL
    assert res.severity is Severity.CRITICAL
    spike = abs(res.details["spike_ic"])
    assert spike >= CFG.predictive_ic_fail
    assert abs(res.details["ic1"]) < lookahead.DEFERRED_FAIL_RATIO * spike


def test_deferred_ic_spike_clean_passes(market, honest):
    art = _arts(market, honest, positions_from_signals(honest, 1))
    assert _by(lookahead.run(art, CFG))["lookahead.deferred_ic_spike"].status is Status.PASS


# ---------------------------------------------------------------------------
# burn-in normalization (full-sample z-score applied to the first 40% of
#   the sample only): invisible to tail-only truncation sampling.
# ---------------------------------------------------------------------------

def test_burnin_normalization_caught(market):
    rets = market["returns"]

    def burnin_signal_func(x):
        ma = x.rolling(20, min_periods=20).mean()
        out = _zx(ma)
        cut = int(0.4 * len(x))
        out.iloc[:cut] = ((ma - ma.mean()) / ma.std()).iloc[:cut]
        return out

    sig = burnin_signal_func(rets)
    art = _arts(market, sig, positions_from_signals(sig, 1), signal_input=rets)
    res = _by(probes_shift.run(art, CFG, signal_func=burnin_signal_func,
                               backtest_func=make_backtest_func()))
    assert res["dynamic.rolling_window_integrity"].status is Status.FAIL


# ---------------------------------------------------------------------------
# survivor panel laundered with one token exit: exit rate, not existence.
# ---------------------------------------------------------------------------

def test_token_exit_survivor_panel_caught():
    m = simulate_market(n_assets=60, seed=3, death_frac=0.0)
    total = (1 + m["returns"]).prod()
    keep = total.sort_values(ascending=False).index[:30]
    rets = m["returns"][keep].copy()
    uni = pd.DataFrame(True, index=rets.index, columns=rets.columns)
    uni.iloc[500:, 0] = False
    rets.iloc[500:, 0] = np.nan
    sig = momentum_signal(rets)
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            positions=positions_from_signals(sig, 1),
                            universe=uni, signal_lag=1)
    art.validate()
    res = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert res.status is Status.WARN
    # Pin the rate branch, not just any WARN: both WARN branches interpolate
    # a '0' here, and the token branch always does - it fires only when
    # exit_rate < 1%, so {:.2%} renders '0.xx%' - so a message-substring
    # check alone would be vacuous.
    assert res.details["n_exiting_assets"] == 1
    assert res.details["exit_rate_per_year"] < CFG.min_exit_rate_per_year
    assert "token" in res.message


def test_realistic_exit_rate_passes(market, honest):
    # clean market: ~4 exits over 30 assets x 4 years ~ 3%/year
    art = _arts(market, honest, positions_from_signals(honest, 1))
    assert _by(survivorship.run(art, CFG))["survivorship.no_exits"].status is Status.PASS



# A causal fast-reversal signal can become anti-correlated with its own bar when
# shifted earlier. That shape needs scoped interpretation; a momentum leak whose edge
# evaporates must still fail.

def _reversal_market(n_assets=30, n_periods=750, seed=11, theta=0.35):
    """MA(1) returns with a negative lag-1 loading: genuine 1-day
    cross-sectional mean reversion."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_periods)
    assets = [f"A{i:03d}" for i in range(n_assets)]
    eps = rng.normal(0.0, 0.02, (n_periods, n_assets))
    rets = eps.copy()
    rets[1:] -= theta * eps[:-1]
    return pd.DataFrame(rets, index=dates, columns=assets)


def _reversal_signal(x):
    return -_zx(x)     # causal: the row for date t reads row t only


@pytest.fixture(scope="module")
def reversal_art():
    rets = _reversal_market()
    art = BacktestArtifacts(signals=_reversal_signal(rets), asset_returns=rets,
                            signal_input=rets)
    art.validate()
    return art.aligned()


def test_causal_reversal_alpha_is_not_convicted_by_date_shift(reversal_art):
    res = _by(probes_shift.run(reversal_art, CFG,
                               signal_func=_reversal_signal,
                               backtest_func=make_backtest_func(costs_bps=0.0)))
    r = res[probes_shift.CHECK_DATE_SHIFT]
    assert r.status is not Status.FAIL
    assert r.details["peek_signature"] == "anti_correlated"
    assert r.details["signal_verified_causal"] is True
    # The baseline has a real edge and the earlier-shift collapse is strongly negative,
    # which is compatible with causal reversal.
    assert r.details["curve"]["0"] >= 1.0
    assert r.details["curve"]["-1"] <= -1.0
    # a 1-day reversal alpha is delay-fragile: if the probe warns, it must be
    # the honest rule-(b) ultra-fast-alpha framing, not the peek accusation
    if r.status is Status.WARN:
        assert "delay" in r.message


def test_unverified_reversal_collapse_warns_for_adjudication(reversal_art):
    # same book, no signal_func: the probe cannot tell a causal reversal from
    # a reversed-sign embed - it must ask for adjudication, not FAIL CRITICAL
    res = _by(probes_shift.run(reversal_art, CFG,
                               backtest_func=make_backtest_func(costs_bps=0.0)))
    r = res[probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.WARN
    assert r.details["peek_signature"] == "anti_correlated"
    assert r.details["signal_verified_causal"] is False
    assert "signal_func" in r.remediation


def test_verified_reversal_with_robust_delay_passes_outright(reversal_art):
    """When the anti-correlated collapse is verified benign and the edge
    survives one bar of extra delay, the probe must PASS with the
    explanatory note (the peek_exempt path)."""
    n = len(reversal_art.signals.index)
    pos_sr = pd.Series(0.002 + 0.01 * np.where(np.arange(n) % 2 == 0, 1, -1),
                       index=reversal_art.signals.index)   # ann SR ~ +3.2
    neg_sr = -pos_sr                                       # ann SR ~ -3.2

    def canned_bt(signals, asset_returns):
        for k in range(-CFG.date_shift_max, CFG.date_shift_max + 1):
            ref = reversal_art.signals.shift(k)
            if np.array_equal(signals.to_numpy(), ref.to_numpy(),
                              equal_nan=True):
                return neg_sr if k == -1 else pos_sr
        raise AssertionError("unexpected signal frame passed to backtest_func")

    res = _by(probes_shift.run(reversal_art, CFG,
                               signal_func=_reversal_signal,
                               backtest_func=canned_bt))
    r = res[probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.PASS
    assert r.details["peek_signature"] == "anti_correlated"
    assert "not informative" in r.message


def test_momentum_welded_leak_still_fails_date_shift(market, honest):
    # attack guard: the momentum-flavored welded leak (edge evaporates at
    # k=-1) keeps its CRITICAL FAIL - the reversal exemption must not blunt it
    rets = market["returns"]
    z_fwd = _zx(rets.shift(-1))
    sig = (0.25 * honest.fillna(0.0) + z_fwd).where(honest.notna())
    art = _arts(market, sig, positions_from_signals(sig, 1))
    r = _by(probes_shift.run(art, CFG, backtest_func=make_backtest_func()))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["peek_signature"] == "evaporation"


# Optional artifacts with no calendar overlap must be refused before alignment can
# replace their content with zeros or NaNs.

def _shifted_years(df, years=10):
    out = df.copy()
    out.index = df.index + pd.DateOffset(years=years)
    return out


@pytest.mark.parametrize("field", ["positions", "universe", "strategy_returns"])
def test_disjoint_artifact_dates_raise_misaligned(market, honest, field):
    rets = market["returns"]
    pos = positions_from_signals(honest, 1)
    values = {
        "positions": _shifted_years(pos),
        "universe": _shifted_years(market["universe"]),
        "strategy_returns": _shifted_years(net_returns(pos, rets, 10.0)),
    }
    art = BacktestArtifacts(signals=honest, asset_returns=rets,
                            **{field: values[field]})
    with pytest.raises(MisalignedInputError, match=field):
        art.validate()


def test_partially_overlapping_artifacts_still_validate(market, honest):
    # honest guard: a positions panel covering only the back 60% of the
    # sample is legitimate (e.g. the strategy started later) and must pass
    rets = market["returns"]
    pos = positions_from_signals(honest, 1)
    cut = int(0.4 * len(pos))
    art = BacktestArtifacts(signals=honest, asset_returns=rets,
                            positions=pos.iloc[cut:],
                            strategy_returns=net_returns(pos, rets, 10.0).iloc[cut:])
    art.validate()   # must not raise
    aligned = art.aligned()
    assert aligned.positions.abs().to_numpy().sum() > 0


def test_cost_sensitivity_does_not_pass_on_allnan_sharpe(market, honest):
    # A disjoint positions panel cannot support a cost sweep; the check must decline to
    # judge.
    rets = market["returns"]
    pos = _shifted_years(positions_from_signals(honest, 1))
    art = BacktestArtifacts(signals=honest, asset_returns=rets,
                            positions=pos).aligned()   # deliberately unvalidated
    r = _by(costs.run(art, CFG))["costs.cost_sensitivity"]
    assert r.status is Status.SKIP
    assert "finite" in r.message


# ---------------------------------------------------------------------------
# signal_func coverage floor: a callable returning one correct cell must not
#   satisfy reproducibility (1/1 cells match) nor make the truncation probe
#   vacuous - that would launder a leaky full-sample z-score signal through
#   both.
# ---------------------------------------------------------------------------

def _leaky_full_sample_z(x):
    ma = x.rolling(20, min_periods=20).mean()
    return (ma - ma.mean()) / ma.std()      # full-sample z-score: not causal


def test_one_cell_signal_func_cannot_launder_truncation(market):
    rets = market["returns"]
    sig = _leaky_full_sample_z(rets)

    def one_cell(x):
        return _leaky_full_sample_z(x).iloc[[-1], [0]]

    art = _arts(market, sig, positions_from_signals(sig, 1), signal_input=rets)
    res = _by(probes_shift.run(art, CFG, signal_func=one_cell))
    r_repro = res[probes_shift.CHECK_REPRO]
    r_trunc = res[probes_shift.CHECK_TRUNCATION]
    assert r_repro.status is Status.WARN
    assert r_repro.details["coverage"] < 0.01
    assert "cover" in r_repro.message
    # the truncation probe must refuse the vacuous verdict, not PASS on it
    assert r_trunc.status is Status.SKIP
    assert "cover" in r_trunc.message


def test_full_coverage_signal_func_passes_both_probes(market, honest):
    # honest guard: a full-panel causal signal_func keeps clean PASSes
    art = _arts(market, honest, positions_from_signals(honest, 1),
                signal_input=market["returns"])
    res = _by(probes_shift.run(art, CFG, signal_func=momentum_signal))
    assert res[probes_shift.CHECK_REPRO].status is Status.PASS
    assert res[probes_shift.CHECK_REPRO].details["coverage"] >= 0.999
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.PASS


# Net returns above the supplied gross reconstruction imply negative drag and require a
# mismatched-artifact warning.

def test_net_above_gross_warns_mismatched_artifacts(market, honest):
    rets = market["returns"]
    pos = positions_from_signals(honest, 1)
    # costs_bps=-5 adds 5bps per unit traded: net systematically above gross
    art = _arts(market, honest, pos,
                strategy_returns=net_returns(pos, rets, -5.0))
    r = _by(costs.run(art, CFG))["costs.missing_transaction_costs"]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["implied_bps"] < -0.5
    assert "ABOVE" in r.message


def test_net_below_gross_still_passes_with_positive_implied(market, honest):
    # honest guard: properly charged costs keep the PASS and the implied bps
    art = _arts(market, honest, positions_from_signals(honest, 1))
    r = _by(costs.run(art, CFG))["costs.missing_transaction_costs"]
    assert r.status is Status.PASS
    assert abs(r.details["implied_bps"] - 10.0) < 0.5


# ---------------------------------------------------------------------------
# out-of-universe sign flips: +0.20 -> -0.20 within the 21-bar holdover
#   grace keeps |w| constant, so an increase-only comparison would read it
#   as a "holdover". A flip is fresh opposite-side risk and gets no grace.
# ---------------------------------------------------------------------------

def _flip_frames(n_periods=300, n_assets=12, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_periods)
    assets = [f"A{i:02d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n_periods, n_assets)),
                        index=dates, columns=assets)
    signals = rets.rolling(5, min_periods=5).mean()
    universe = pd.DataFrame(True, index=dates, columns=assets)
    positions = pd.DataFrame(1.0 / n_assets, index=dates, columns=assets)
    positions.iloc[:, 0] = 0.20
    universe.iloc[100:, 0] = False    # A00 exits at bar 100
    return dates, rets, signals, universe, positions


def test_sign_flip_within_grace_fails():
    dates, rets, signals, universe, positions = _flip_frames()
    positions.iloc[103:, 0] = -0.20   # same |w| flip 3 bars into the grace
    positions.iloc[110:, 0] = 0.0
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            positions=positions, universe=universe,
                            signal_lag=1)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))[
        "survivorship.trading_outside_universe"]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_sign_flip_cells"] >= 1
    assert r.details["first_date"] == str(dates[103].date())
    assert "REVERSE" in r.message


def test_same_side_holdover_within_grace_still_passes():
    # honest guard: holding then unwinding the same side through the exit
    # remains normal rebalance latency
    _, rets, signals, universe, positions = _flip_frames()
    rets.iloc[:] = 0.0  # with zero returns, constant weights are genuine holds
    positions.iloc[105:, 0] = 0.10    # partial unwind, same sign
    positions.iloc[110:, 0] = 0.0
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            positions=positions, universe=universe,
                            signal_lag=1)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))[
        "survivorship.trading_outside_universe"]
    assert r.status is Status.PASS
    assert r.details["n_holdover_cells"] > 0


# ---------------------------------------------------------------------------
# API hygiene: AuditConfig value sanity, include/exclude filtering
#   before dispatch (include=['costs'] must not re-run the pipeline), and
#   the module -> check-id map that dispatch filtering relies on.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(sharpe_warn=-5.0),
    dict(n_placebo=-1),
    dict(truncation_sample_dates=0),
    dict(dsr_fail=0.99),              # fail bar above the warn bar
    dict(leak_perfect_ic=1.5),
])
def test_nonsensical_config_values_raise(kwargs):
    name = next(iter(kwargs))
    with pytest.raises(InputValidationError, match=name):
        AuditConfig(**kwargs)


def test_unknown_config_field_raises():
    # A misspelled field must raise, not be silently ignored.
    with pytest.raises(TypeError, match="sharpe_wrn"):
        AuditConfig(sharpe_wrn=4.0)


def test_reasonable_config_overrides_still_construct():
    AuditConfig(sharpe_warn=4.0, n_trials=345, n_placebo=50,
                min_embargo_periods=10, cost_sensitivity_bps=(2.0, 8.0))


def test_include_costs_never_calls_pipeline(market, honest):
    calls = {"n": 0}
    inner = make_backtest_func()

    def counting_bt(signals, asset_returns):
        calls["n"] += 1
        return inner(signals, asset_returns)

    art = _arts(market, honest, positions_from_signals(honest, 1))
    report = audit(art, CFG, include=["costs"], backtest_func=counting_bt)
    assert calls["n"] == 0
    assert report.results
    assert all(r.check.startswith("costs.") for r in report.results)


def test_include_specific_dynamic_check_still_runs_it(market, honest):
    # honest guard: pre-dispatch filtering must not drop a requested check
    calls = {"n": 0}
    inner = make_backtest_func()

    def counting_bt(signals, asset_returns):
        calls["n"] += 1
        return inner(signals, asset_returns)

    art = _arts(market, honest, positions_from_signals(honest, 1))
    report = audit(art, CFG, include=["dynamic.date_shift"],
                   backtest_func=counting_bt)
    checks = {r.check for r in report.results}
    assert "dynamic.date_shift" in checks
    assert calls["n"] > 0
    # ...while the expensive null probes were never dispatched
    assert calls["n"] <= 2 * CFG.date_shift_max + 1


def test_module_check_id_map_matches_emitted_ids(market, honest):
    # drift guard for the dispatch filter: every id a module emits on a
    # fully-loaded run must be listed in MODULE_CHECK_IDS
    art = _arts(market, honest, positions_from_signals(honest, 1),
                signal_input=market["returns"],
                train_period=(market["returns"].index[0],
                              market["returns"].index[500]),
                test_period=(market["returns"].index[510],
                             market["returns"].index[-1]))
    art.validate()
    cfg = AuditConfig(n_placebo=2, n_shuffle=2)
    assert set(MODULE_CHECK_IDS) == set(CHECK_MODULES)
    for mod_name, known_ids in MODULE_CHECK_IDS.items():
        mod = importlib.import_module(mod_name)
        emitted = {r.check for r in mod.run(art, cfg,
                                            signal_func=momentum_signal,
                                            backtest_func=make_backtest_func())}
        assert emitted <= set(known_ids), (
            f"{mod_name} emitted ids missing from MODULE_CHECK_IDS: "
            f"{emitted - set(known_ids)}")
