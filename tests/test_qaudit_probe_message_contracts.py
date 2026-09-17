"""Probe and performance messages reflect measured evidence and callback contracts.

Causal status requires all three signal probes. Backtests return a Series,
IC band wording is conditional, and single-point nulls request more runs
without misreporting healthy callback outputs as degenerate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import performance
from qaudit.config import AuditConfig
from qaudit.dynamic import probes_shift
from qaudit.dynamic.probes_null import run as null_run
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import make_clean
from qaudit.types import Severity, Status

CFG = AuditConfig()
PPY = 252
N_DATES = 1000
DATES = pd.bdate_range("2020-01-02", periods=N_DATES)
COLS = [f"A{i}" for i in range(6)]


# ---------------------------------------------------------------------------
# Helpers: canned-SR-curve backtest_func (infers the shift k from the all-NaN
# head/tail rows signals.shift(k) manufactures, then returns a series with an
# exactly prescribed annualized Sharpe)
# ---------------------------------------------------------------------------

def _series_with_sr(sr: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N_DATES)
    x = (x - x.mean()) / x.std(ddof=1)          # mean 0, std(ddof=1) 1 exactly
    vol = 0.01
    return pd.Series(x * vol + sr / np.sqrt(PPY) * vol, index=DATES)


def _detect_k(sig: pd.DataFrame) -> int:
    allnan = sig.isna().all(axis=1).to_numpy()
    lead = int(np.argmax(~allnan)) if allnan.any() else 0
    trail = int(np.argmax(~allnan[::-1])) if allnan.any() else 0
    return lead if lead else -trail


def _curve_bt(curve: dict[int, float]):
    series = {k: _series_with_sr(v, seed=10 + k) for k, v in curve.items()}

    def bt(sig, rets):
        return series[_detect_k(sig)]
    return bt


@pytest.fixture(scope="module")
def shift_art():
    rng = np.random.default_rng(1)
    sig = pd.DataFrame(rng.standard_normal((N_DATES, 6)), index=DATES,
                       columns=COLS)
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (N_DATES, 6)), index=DATES,
                        columns=COLS)
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            signal_input=rets)
    art.validate()
    return art.aligned()


# anti-correlated collapse: sr(-1) <= -max(1.0, 0.5*sr0) = -1.0
ANTI_CURVE = {-3: 2.0, -2: 2.0, -1: -2.5, 0: 2.0, 1: 1.9, 2: 1.9, 3: 1.8}
# ambiguous collapse: noise floor sqrt(252/1000) ~ 0.50; -1.0 < -0.7 <= -0.50
AMBIG_CURVE = {-3: 2.0, -2: 2.0, -1: -0.7, 0: 2.0, 1: 1.9, 2: 1.9, 3: 1.8}

THREE_PROBES = ("dynamic.signal_reproducibility",
                "dynamic.rolling_window_integrity",
                "dynamic.signal_input_sensitivity")


def _by(results):
    return {r.check: r for r in results}


# ---------------------------------------------------------------------------
# 1. date_shift wording: the causal stamp is three-legged in every message
# ---------------------------------------------------------------------------

def test_anticorr_unverified_remediation_names_all_three_probes(shift_art):
    r = _by(probes_shift.run(shift_art, CFG,
                             backtest_func=_curve_bt(ANTI_CURVE)))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.WARN
    assert r.details["peek_signature"] == "anti_correlated"
    assert "signal_func" in r.remediation
    for probe in THREE_PROBES:
        assert probe in r.remediation
    # sensitivity must positively PASS; the remediation may not promise that
    # two probes alone clear the warning
    assert "PASS" in r.remediation
    assert "reproducibility and truncation probes" not in r.remediation


def test_ambiguous_unverified_remediation_names_all_three_probes(shift_art):
    r = _by(probes_shift.run(shift_art, CFG,
                             backtest_func=_curve_bt(AMBIG_CURVE)))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.WARN
    assert r.details["peek_signature"] == "ambiguous"
    assert "signal_func" in r.remediation
    for probe in THREE_PROBES:
        assert probe in r.remediation
    assert "reproducibility and truncation probes" not in r.remediation


def test_ambiguous_verified_message_credits_all_three_probes(shift_art):
    r = probes_shift._date_shift(shift_art, CFG, _curve_bt(AMBIG_CURVE),
                                 causal_verified=True)
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert "ambiguous" in r.message
    assert "verified causal" in r.message
    assert "same-bar loading" in r.remediation
    # the stamp is credited to all three probes, not two
    assert "input-sensitivity" in r.message
    assert "truncation both pass" not in r.message


def test_peek_exempt_pass_message_credits_all_three_probes(shift_art):
    r = probes_shift._date_shift(shift_art, CFG, _curve_bt(ANTI_CURVE),
                                 causal_verified=True)
    assert r.status is Status.PASS
    assert "verified causal" in r.message
    assert "input-sensitivity" in r.message
    assert "truncation both pass" not in r.message


def test_trimmed_replay_journey_warn_stands_and_names_sensitivity(shift_art):
    """A trimmed-replay signal_func passes reproducibility and truncation
    yet the WARN stands (sensitivity WARNs, withholding the stamp) - the
    remediation must name the third probe instead of promising the warning
    clears."""
    sig = shift_art.signals

    def replay(inp):
        return sig.reindex(inp.index)

    res = _by(probes_shift.run(shift_art, CFG, signal_func=replay,
                               backtest_func=_curve_bt(ANTI_CURVE)))
    assert res[probes_shift.CHECK_REPRO].status is Status.PASS
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.PASS
    assert res[probes_shift.CHECK_SENSITIVITY].status is Status.WARN
    r = res[probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.WARN                 # does not clear
    assert r.details["signal_verified_causal"] is False
    assert "dynamic.signal_input_sensitivity" in r.remediation


def test_honest_curve_still_passes(shift_art):
    # honest-guard: a smoothly decaying curve keeps the honest PASS path
    curve = {-3: 2.2, -2: 2.15, -1: 2.1, 0: 2.0, 1: 1.9, 2: 1.85, 3: 1.8}
    r = _by(probes_shift.run(shift_art, CFG,
                             backtest_func=_curve_bt(curve)))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.PASS


# ---------------------------------------------------------------------------
# 2. date_shift: DataFrame-returning backtest_func -> contract SKIP, not a
#    misdirected "raised at k=-3" ERROR
# ---------------------------------------------------------------------------

def test_dataframe_return_skips_with_series_contract(shift_art):
    net = _series_with_sr(1.5)

    def bt_df(sig, rets):
        return net.to_frame("net")                 # routine user slip

    r = _by(probes_shift.run(shift_art, CFG, backtest_func=bt_df))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.SKIP
    assert "pd.Series" in r.message
    assert "DataFrame" in r.message
    assert f"{N_DATES}x1" in r.message             # concrete shape
    assert "squeeze" in r.message                  # the actual fix
    # no misdiagnosis: nothing raised, no k to blame
    assert "raised" not in r.message
    assert "k=-3" not in r.message


def test_genuinely_raising_backtest_func_still_errors(shift_art):
    def bt_boom(sig, rets):
        raise ValueError("engine exploded")

    r = _by(probes_shift.run(shift_art, CFG, backtest_func=bt_boom))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.ERROR                # honest-guard: real raises
    assert "raised while re-running" in r.message
    assert "engine exploded" in r.message


# ---------------------------------------------------------------------------
# 3. suspicious_ic PASS wording: band claim only inside the band
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ic_art():
    n, n_names = 600, 20
    dates = pd.bdate_range("2020-01-02", periods=n)
    cols = [f"B{i:02d}" for i in range(n_names)]
    rng = np.random.default_rng(2)
    art = BacktestArtifacts(
        signals=pd.DataFrame(rng.standard_normal((n, n_names)), index=dates,
                             columns=cols),
        asset_returns=pd.DataFrame(rng.normal(0.0, 0.01, (n, n_names)),
                                   index=dates, columns=cols))
    art.validate()
    return art.aligned()


def _ic_res(mean_ic: float, art) -> object:
    # fabricated per-date IC with an exact mean (the +-0.001 wobble cancels)
    n = len(art.signals.index)
    ic = pd.Series(mean_ic + 0.001 * np.tile([1.0, -1.0], n // 2),
                   index=art.signals.index)
    return performance._suspicious_ic(ic, art, CFG)


@pytest.mark.parametrize("mean_ic", [0.062, -0.062, 0.004])
def test_pass_outside_band_does_not_claim_band_membership(mean_ic, ic_art):
    r = _ic_res(mean_ic, ic_art)
    assert r.status is Status.PASS                 # 0.062 < 0.08 warn bar
    assert r.details["warn_bar"] == pytest.approx(CFG.predictive_ic_warn)
    assert "within the realistic" not in r.message
    assert f"below the {r.details['warn_bar']:.2f} suspicion bar" in r.message


def test_pass_inside_band_keeps_band_wording(ic_art):
    # In-band horizon-one IC retains the informative 0.01-0.05 band clause.
    r = _ic_res(0.031, ic_art)
    assert r.status is Status.PASS
    assert "within the realistic 0.01-0.05 band" in r.message


# ---------------------------------------------------------------------------
# 4. probes_null: n_placebo=1 / n_shuffle=1 SKIP + probe_health accounting
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean():
    return make_clean()


@pytest.fixture(scope="module")
def null_art(clean):
    return clean.artifacts.aligned()


def test_n_placebo_1_skip_is_truthful(null_art, clean):
    res = _by(null_run(null_art, AuditConfig(n_placebo=1, n_shuffle=2, seed=1),
                       backtest_func=clean.backtest_func))
    for cid in ("dynamic.placebo_pipeline_bias", "dynamic.placebo_percentile"):
        r = res[cid]
        assert r.status is Status.SKIP
        # no false bad-run accounting; the real cause and fix are named
        assert "NaN/degenerate" not in r.message
        assert "AuditConfig.n_placebo" in r.message
        assert "need >= 2" in r.message
        assert r.details["n_bad"] == 0 and r.details["n_total"] == 1
    health = res["dynamic.probe_health"]
    assert health.status is Status.WARN
    assert "AuditConfig.n_placebo=1" in health.message
    assert "NaN/degenerate" not in health.message  # 0 bad runs occurred
    # remediation points at the config count, not at a healthy callable
    assert "AuditConfig" in health.remediation
    assert "backtest_func" not in health.remediation


def test_n_shuffle_1_skip_is_truthful(null_art, clean):
    res = _by(null_run(null_art, AuditConfig(n_placebo=2, n_shuffle=1, seed=1),
                       backtest_func=clean.backtest_func))
    r = res["dynamic.shuffled_labels"]
    assert r.status is Status.SKIP
    assert "NaN/degenerate" not in r.message
    assert "AuditConfig.n_shuffle" in r.message
    health = res["dynamic.probe_health"]
    assert "AuditConfig.n_shuffle=1" in health.message


def test_n_2_boundary_stays_legal_and_runs(null_art, clean):
    # honest-guard: the minimum testable counts keep producing verdicts
    res = _by(null_run(null_art, AuditConfig(n_placebo=2, n_shuffle=2, seed=1),
                       backtest_func=clean.backtest_func))
    for cid in ("dynamic.placebo_pipeline_bias", "dynamic.placebo_percentile",
                "dynamic.shuffled_labels"):
        assert res[cid].status is not Status.SKIP
    assert "dynamic.probe_health" not in res


def test_genuinely_degenerate_runs_keep_bad_run_diagnosis(null_art):
    # honest-guard: real bad-run storms keep the backtest_func-blaming
    # wording and counts (here every re-run crashes: 3/3 bad > 20%)
    def bt_broken(signals, asset_returns):
        raise ValueError("engine exploded")

    res = _by(null_run(null_art, AuditConfig(n_placebo=3, n_shuffle=3, seed=1),
                       backtest_func=bt_broken))
    r = res["dynamic.placebo_pipeline_bias"]
    assert r.status is Status.SKIP
    assert "3/3" in r.message and "NaN/degenerate" in r.message
    assert "fix backtest_func" in r.message
    health = res["dynamic.probe_health"]
    assert health.status is Status.WARN
    assert "3/3 placebo" in health.message
    assert "NaN/degenerate" in health.message
    assert "backtest_func" in health.remediation
    assert health.details["n_placebo_bad"] == 3
    assert health.details["n_shuffle_bad"] == 3
