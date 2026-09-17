"""Signal sensitivity, date-shift interpretation, and callback input isolation.

Causal transformations must not be mistaken for frozen output, and callbacks
must not corrupt inputs shared by subsequent probes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.config import AuditConfig
from qaudit.dynamic import probes_shift
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (make_backtest_func, momentum_signal,
                              net_returns, positions_from_signals,
                              simulate_market)
from qaudit.types import CheckResult, Severity, Status, errored, passed, \
    skipped, warned

CFG = AuditConfig()

MILD_BLEND_W = 0.2      # same-bar MR weight that lands in the ambiguous band


def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def _arts(market, sig, **kw):
    rets = market["returns"]
    pos = positions_from_signals(sig, 1)
    base = dict(signals=sig, asset_returns=rets, positions=pos,
                strategy_returns=net_returns(pos, rets, 10.0),
                universe=market.get("universe"), signal_lag=1,
                declared_costs_bps=10.0, signal_input=rets)
    base.update(kw)
    art = BacktestArtifacts(**base)
    art.validate()
    return art.aligned()


def _by(results):
    out = {r.check: r for r in results}
    assert set(out) == {probes_shift.CHECK_DATE_SHIFT, probes_shift.CHECK_REPRO,
                        probes_shift.CHECK_TRUNCATION,
                        probes_shift.CHECK_SENSITIVITY}
    assert len(results) == 4
    return out


def mild_blend_signal(asset_returns: pd.DataFrame) -> pd.DataFrame:
    mom = momentum_signal(asset_returns)
    mr = -_zx(asset_returns)
    return ((1 - MILD_BLEND_W) * mom + MILD_BLEND_W * mr).where(mom.notna())


@pytest.fixture(scope="module")
def market():
    return simulate_market(seed=0)


@pytest.fixture(scope="module")
def blend_art(market):
    """Ambiguous-band book: date_shift populates signal_verified_causal."""
    return _arts(market, mild_blend_signal(market["returns"]))


# ---------------------------------------------------------------------------
# 1. fabrication check: values, not index labels
# ---------------------------------------------------------------------------

def _master_calendar_func(master_index):
    def func(signal_input):
        # textbook alignment hygiene: every panel lives on the firm calendar
        return mild_blend_signal(signal_input).reindex(master_index)
    return func


def test_master_calendar_reindexer_is_not_fabrication(market, blend_art):
    idx = market["returns"].index
    master = idx.union(pd.bdate_range(idx[-1], periods=150))
    func = _master_calendar_func(master)
    res = _by(probes_shift.run(blend_art, CFG, signal_func=func,
                               backtest_func=make_backtest_func()))
    # All-NaN calendar padding cannot establish replay of a stored signal panel.
    sens = res[probes_shift.CHECK_SENSITIVITY]
    assert sens.status is Status.PASS
    assert res[probes_shift.CHECK_REPRO].status is Status.PASS
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.PASS
    # the causal stamp is granted: ambiguous band downgrades to MEDIUM
    ds = res[probes_shift.CHECK_DATE_SHIFT]
    assert ds.details["signal_verified_causal"] is True
    assert ds.severity is Severity.MEDIUM


def test_untrimmed_replayer_still_fails_fabrication(market):
    sig = mild_blend_signal(market["returns"])
    art = _arts(market, sig)
    stored = sig

    def cheat(signal_input):
        return stored                      # real values beyond any truncation

    r = _by(probes_shift.run(art, CFG, signal_func=cheat))[
        probes_shift.CHECK_SENSITIVITY]
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert "non-NaN values" in r.message
    assert r.details["n_rows_beyond"] > 0


def test_nan_masking_replayer_caught_by_scramble(market):
    """A replayer that NaN-masks rows beyond the input's end passes the
    value-based fabrication check but is value-insensitive - both perturbation
    stages leave it bit-identical, so the scramble arm convicts it."""
    sig = mild_blend_signal(market["returns"])
    art = _arts(market, sig)
    stored = sig

    def cheat(signal_input):
        last = signal_input.index.max()
        return stored.where(
            pd.Series(stored.index <= last, index=stored.index), np.nan)

    r = _by(probes_shift.run(art, CFG, signal_func=cheat))[
        probes_shift.CHECK_SENSITIVITY]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert "BIT-IDENTICAL" in r.message


# ---------------------------------------------------------------------------
# 2. two-stage scramble: honest permutation-invariant overlays pass
# ---------------------------------------------------------------------------

def _broadcast(series, columns):
    return pd.concat({c: series for c in columns}, axis=1)


def breadth_overlay(signal_input):
    b = (signal_input > 0).mean(axis=1)
    return _broadcast(b.rolling(10, min_periods=10).mean() - 0.5,
                      signal_input.columns)


def median_overlay(signal_input):
    med = signal_input.median(axis=1)
    return _broadcast(med.rolling(15, min_periods=15).mean(),
                      signal_input.columns)


def iqr_dispersion_overlay(signal_input):
    iqr = (signal_input.quantile(0.75, axis=1)
           - signal_input.quantile(0.25, axis=1))
    return _broadcast(iqr.rolling(15, min_periods=15).mean(),
                      signal_input.columns)


@pytest.mark.parametrize("func", [breadth_overlay, median_overlay,
                                  iqr_dispersion_overlay],
                         ids=["breadth", "median", "iqr_dispersion"])
def test_perm_invariant_overlay_passes_sensitivity(market, func):
    # A deterministic input transformation can still respond to input values.
    sig = func(market["returns"])
    art = _arts(market, sig)
    r = _by(probes_shift.run(art, CFG, signal_func=func))[
        probes_shift.CHECK_SENSITIVITY]
    assert r.status is Status.PASS
    assert r.details["perm_invariant"] is True
    assert r.details["frac_cells_changed"] > 0.5
    # the message says what was actually measured, both halves
    assert "invariant" in r.message and "magnitude" in r.message


def test_scramble_sensitive_func_unchanged(market):
    sig = momentum_signal(market["returns"])
    art = _arts(market, sig)
    r = _by(probes_shift.run(art, CFG, signal_func=momentum_signal))[
        probes_shift.CHECK_SENSITIVITY]
    assert r.status is Status.PASS
    assert r.details["perm_invariant"] is False


def test_trimmed_replayer_still_warns_two_stage(market):
    # A closure that replays stored output survives both perturbations unchanged and
    # must warn HIGH.
    sig = mild_blend_signal(market["returns"])
    art = _arts(market, sig)
    stored = sig

    def cheat(signal_input):
        return stored.reindex(index=signal_input.index)

    res = _by(probes_shift.run(art, CFG, signal_func=cheat,
                               backtest_func=make_backtest_func()))
    r = res[probes_shift.CHECK_SENSITIVITY]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert "BIT-IDENTICAL" in r.message
    # both perturbations are named: the conviction measured both
    assert "scramble" in r.message and "sign flip" in r.message
    # and the stamp stays revoked
    ds = res[probes_shift.CHECK_DATE_SHIFT]
    assert ds.details["signal_verified_causal"] is False
    assert ds.severity is Severity.HIGH


def test_perm_invariant_reversal_overlay_earns_stamp(market):
    """A causal fast-reversal breadth overlay must pass sensitivity and retain eligibility
    for the causal stamp through the date-shift path."""
    def breadth_reversal(signal_input):
        b = (signal_input > 0).mean(axis=1)
        return _broadcast(-(b.rolling(3, min_periods=3).mean() - 0.5),
                          signal_input.columns)

    sig = breadth_reversal(market["returns"])
    art = _arts(market, sig)
    res = _by(probes_shift.run(art, CFG, signal_func=breadth_reversal,
                               backtest_func=make_backtest_func()))
    assert res[probes_shift.CHECK_SENSITIVITY].status is Status.PASS
    ds = res[probes_shift.CHECK_DATE_SHIFT]
    # whenever date_shift consulted the stamp, it saw a verified signal
    assert ds.details.get("signal_verified_causal") in (None, True)


# ---------------------------------------------------------------------------
# 3. date_shift degenerate-curve guard
# ---------------------------------------------------------------------------

def _fast_signal(rets):
    ma = rets.rolling(2, min_periods=2).mean()
    return _zx(ma)


def _trimmed_fast_book(seed=0):
    market = simulate_market(seed=seed)
    rets = market["returns"]
    sig = _fast_signal(rets)
    ok = sig.notna().any(axis=1)          # trim warmup: no all-NaN real rows
    market = dict(market, returns=rets.loc[ok])
    return market, sig.loc[ok]


def _samebar_engine():
    def engine(signals, asset_returns):
        pos = positions_from_signals(signals, lag=0)      # same-bar bug
        return net_returns(pos, asset_returns, costs_bps=10.0)
    return engine


@pytest.mark.parametrize("degenerate_value", ["zeros", "nans"])
def test_nan_curve_is_skip_not_pass(degenerate_value):
    """A breaker engine rejects all-NaN rows created by shifting, making every off-zero
    Sharpe unmeasurable. This cannot support a graceful-decay claim."""
    market, sig = _trimmed_fast_book()
    art = _arts(market, sig)
    inner = _samebar_engine()

    def breaker(signals, asset_returns):
        if signals.isna().all(axis=1).any():
            if degenerate_value == "zeros":
                return pd.Series(0.0, index=asset_returns.index)
            return pd.Series(np.nan, index=asset_returns.index)
        return inner(signals, asset_returns)

    r = _by(probes_shift.run(art, CFG, backtest_func=breaker))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.SKIP
    assert "signals argument" in r.message
    assert "NaN-padded" in r.message
    assert r.details["n_degenerate_points"] == 2 * CFG.date_shift_max
    # never claim what was not measured
    assert "decays" not in r.message and "peeking" not in r.message


def test_nan_curve_control_without_breaker_still_flagged():
    # paired honest-detection guard: the same same-bar engine without the
    # breaker keeps its WARN (delay-fragility catches the timing bug)
    market, sig = _trimmed_fast_book()
    art = _arts(market, sig)
    r = _by(probes_shift.run(art, CFG, backtest_func=_samebar_engine()))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH


def test_flat_curve_is_skip_not_pass():
    """A backtest callback that ignores signals returns identical Sharpe at every shift and
    cannot establish improvement from peeking."""
    market, sig = _trimmed_fast_book()
    art = _arts(market, sig)

    def ignores_signals(signals, asset_returns):
        return asset_returns.mean(axis=1)

    r = _by(probes_shift.run(art, CFG, backtest_func=ignores_signals))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.SKIP
    assert "flat" in r.message
    assert "signals argument" in r.message


def test_partial_degeneracy_pass_message_is_honest():
    """Engine degenerates only k>0 (head-pad gate): the PASS may claim the
    peek observation but must not claim delay decay it never measured."""
    market, sig = _trimmed_fast_book()
    art = _arts(market, sig)
    honest = make_backtest_func()

    def head_gate(signals, asset_returns):
        if bool(signals.iloc[0].isna().all()):   # shift(k>0) pads the head
            return pd.Series(0.0, index=asset_returns.index)
        return honest(signals, asset_returns)

    r = _by(probes_shift.run(art, CFG, backtest_func=head_gate))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.PASS
    assert r.details["n_degenerate_points"] == CFG.date_shift_max
    assert "delay does not erase" not in r.message
    curve = r.details["curve"]
    assert ("peeking improves" in r.message) == (curve["-1"] >= curve["0"])


def test_honest_engine_pass_message_matches_curve():
    market = simulate_market(seed=9)
    sig = momentum_signal(market["returns"])
    art = _arts(market, sig)
    r = _by(probes_shift.run(art, CFG, backtest_func=make_backtest_func()))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.PASS
    curve = r.details["curve"]
    assert curve["-1"] is not None and curve["1"] is not None
    # every claim in the message is backed by the measured curve
    assert ("peeking improves" in r.message) == (curve["-1"] >= curve["0"])
    assert "delay does not erase" in r.message
    assert r.details["n_degenerate_points"] == 0


# ---------------------------------------------------------------------------
# 4. causal-stamp conjunction: all three legs must be PASS
# ---------------------------------------------------------------------------

def test_warn_repro_skip_trunc_pass_sens_denies_stamp(market, blend_art):
    """A gate accepting repro/trunc `is not FAIL` would grant the stamp on
    this measurement triple and wrongly downgrade the ambiguous WARN/HIGH to
    WARN/MEDIUM with a 'verified causal' message."""
    rets = market["returns"]
    skip_dates = rets.index[40::18]                    # ~5.4% of dates

    def partial_func(x):
        out = mild_blend_signal(x)
        return out.loc[~x.index.isin(skip_dates)]

    res = _by(probes_shift.run(blend_art, CFG, signal_func=partial_func,
                               backtest_func=make_backtest_func()))
    assert res[probes_shift.CHECK_REPRO].status is Status.WARN
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.SKIP
    assert res[probes_shift.CHECK_SENSITIVITY].status is Status.PASS
    ds = res[probes_shift.CHECK_DATE_SHIFT]
    assert ds.status is Status.WARN
    assert ds.severity is Severity.HIGH                # not downgraded
    assert ds.details["signal_verified_causal"] is False
    assert "verified causal" not in ds.message


_S = probes_shift.CHECK_SENSITIVITY
_R = probes_shift.CHECK_REPRO
_T = probes_shift.CHECK_TRUNCATION


def _mk(check: str, status: Status) -> CheckResult:
    if status is Status.PASS:
        return passed(check, "stub")
    if status is Status.WARN:
        return warned(check, "stub")
    if status is Status.SKIP:
        return skipped(check, "stub")
    return errored(check, RuntimeError("stub"))


@pytest.mark.parametrize("statuses,expect", [
    ((Status.PASS, Status.PASS, Status.PASS), True),
    ((Status.WARN, Status.PASS, Status.PASS), False),   # repro leg
    ((Status.SKIP, Status.PASS, Status.PASS), False),
    ((Status.PASS, Status.SKIP, Status.PASS), False),   # truncation leg
    ((Status.PASS, Status.WARN, Status.PASS), False),
    ((Status.PASS, Status.PASS, Status.WARN), False),   # sensitivity leg
    ((Status.PASS, Status.PASS, Status.SKIP), False),
    ((Status.ERROR, Status.ERROR, Status.SKIP), False),
], ids=["all_pass", "repro_warn", "repro_skip", "trunc_skip", "trunc_warn",
        "sens_warn", "sens_skip", "error"])
def test_stamp_gate_requires_all_three_pass(monkeypatch, blend_art,
                                            statuses, expect):
    """Unit-pins run()'s causal_verified conjunction over the status matrix:
    only PASS/PASS/PASS earns the stamp - an 'is not FAIL' gate fails here."""
    rs, ts, ss = statuses
    monkeypatch.setattr(
        probes_shift, "_signal_func_probes",
        lambda artifacts, config, signal_func: (_mk(_R, rs), _mk(_T, ts),
                                                _mk(_S, ss)))
    res = _by(probes_shift.run(blend_art, CFG, signal_func=lambda x: x,
                               backtest_func=make_backtest_func()))
    ds = res[probes_shift.CHECK_DATE_SHIFT]
    assert ds.details["signal_verified_causal"] is expect
    assert ds.severity is (Severity.MEDIUM if expect else Severity.HIGH)


def test_welded_leak_with_slack_func_keeps_critical(market):
    """A welded leak with a slack signal_func must keep FAIL/CRITICAL; a
    weakened stamp gate would downgrade it to WARN/HIGH."""
    market7 = simulate_market(seed=7)
    rets = market7["returns"]
    honest = momentum_signal(rets)
    sig = (0.25 * honest.fillna(0.0) + _zx(rets.shift(-1))).where(honest.notna())
    art = _arts(market7, sig)

    res = _by(probes_shift.run(art, CFG, signal_func=momentum_signal,
                               backtest_func=make_backtest_func()))
    # the honest func does not reproduce the leaky panel -> repro WARN...
    assert res[probes_shift.CHECK_REPRO].status is Status.WARN
    # ...and the un-stamped evaporation collapse stays CRITICAL
    ds = res[probes_shift.CHECK_DATE_SHIFT]
    assert ds.status is Status.FAIL
    assert ds.severity is Severity.CRITICAL


# ---------------------------------------------------------------------------
# 5. defensive copies: mutating callables neither convict nor corrupt
# ---------------------------------------------------------------------------

def _mut_units_func(x):
    """A causal trailing mean converts values in place. Isolated callback inputs must
    prevent cumulative mutation from creating false truncation evidence."""
    x *= 100.0
    return x.rolling(20, min_periods=20).mean()


def test_mutating_causal_signal_func_passes_and_leaves_artifacts_intact(
        market):
    rets = market["returns"]
    sig = _mut_units_func(rets.copy())          # user's own generation run
    art = _arts(market, sig)
    input_before = art.signal_input.copy()

    res1 = _by(probes_shift.run(art, CFG, signal_func=_mut_units_func))
    assert res1[probes_shift.CHECK_REPRO].status is Status.PASS
    assert res1[probes_shift.CHECK_TRUNCATION].status is Status.PASS
    assert res1[probes_shift.CHECK_TRUNCATION].details["n_mismatched"] == 0
    assert res1[probes_shift.CHECK_SENSITIVITY].status is Status.PASS
    # The caller's input must remain bit-identical after probing.
    pd.testing.assert_frame_equal(art.signal_input, input_before)

    # a second run of the same artifacts object reproduces the first
    res2 = _by(probes_shift.run(art, CFG, signal_func=_mut_units_func))
    for cid in (probes_shift.CHECK_REPRO, probes_shift.CHECK_TRUNCATION,
                probes_shift.CHECK_SENSITIVITY):
        assert res2[cid].status is res1[cid].status
        assert res2[cid].details == res1[cid].details


def test_idempotent_mutation_still_cannot_corrupt_artifacts(market):
    """fillna-in-place never triggers the truncation FP - the integrity
    guarantee must hold independently of any conviction."""
    def mut_fill(x):
        x.fillna(0.0, inplace=True)
        return x.rolling(20, min_periods=20).mean()

    rets = market["returns"]
    art = _arts(market, mut_fill(rets.copy()))
    input_before = art.signal_input.copy()
    probes_shift.run(art, CFG, signal_func=mut_fill)
    pd.testing.assert_frame_equal(art.signal_input, input_before)
    assert art.signal_input.isna().any().any()   # the NaNs are still there


def test_mutating_backtest_func_curve_matches_pure_equivalent(market):
    sig = momentum_signal(market["returns"])
    art = _arts(market, sig)
    rets_before = art.asset_returns.copy()
    sig_before = art.signals.copy()

    def mut_bt(signals, asset_returns):
        # in-place lag of both frames
        signals.iloc[:] = signals.shift(1).to_numpy()
        asset_returns.iloc[:] = asset_returns.shift(1).to_numpy()
        pos = positions_from_signals(signals, lag=1)
        return net_returns(pos, asset_returns, costs_bps=10.0)

    def pure_bt(signals, asset_returns):
        s, r = signals.shift(1), asset_returns.shift(1)
        pos = positions_from_signals(s, lag=1)
        return net_returns(pos, r, costs_bps=10.0)

    r_mut = _by(probes_shift.run(art, CFG, backtest_func=mut_bt))[
        probes_shift.CHECK_DATE_SHIFT]
    r_pure = _by(probes_shift.run(art, CFG, backtest_func=pure_bt))[
        probes_shift.CHECK_DATE_SHIFT]
    # Each backtest rerun needs isolated asset returns so in-place shifts cannot
    # accumulate across runs.
    assert r_mut.details["curve"] == r_pure.details["curve"]
    pd.testing.assert_frame_equal(art.asset_returns, rets_before)
    pd.testing.assert_frame_equal(art.signals, sig_before)


def test_run_does_not_mutate_artifacts_with_mutating_callables(market):
    # hygiene pin extending test_run_does_not_mutate_artifacts to hostile
    # callables: nothing the callables do may leak into the artifacts
    rets = market["returns"]
    art = _arts(market, _mut_units_func(rets.copy()))
    before = {n: getattr(art, n).copy()
              for n in ("signals", "asset_returns", "signal_input")}

    def mut_bt(signals, asset_returns):
        signals *= 3.0
        asset_returns *= 3.0
        pos = positions_from_signals(signals, lag=1)
        return net_returns(pos, asset_returns, costs_bps=10.0)

    probes_shift.run(art, CFG, signal_func=_mut_units_func,
                     backtest_func=mut_bt)
    for n, frame in before.items():
        pd.testing.assert_frame_equal(getattr(art, n), frame)
