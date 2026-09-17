"""Tests for qaudit.checks.performance (suspicious performance / overfitting / IC stability).

Calls the module's ``run`` directly on ``case.artifacts.aligned()`` - never through
``qaudit.audit`` - so it stays independent of the other check modules.
"""
from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from qaudit.checks.performance import run
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import make_clean, make_lookahead, make_overfit
from qaudit.types import Severity, Status

ALL_IDS = {
    "performance.suspicious_sharpe",
    "performance.deflated_sharpe",
    "performance.suspicious_ic",
    "performance.ic_stability",
    "performance.ic_regime_concentration",
    "performance.sample_size",
}


def by_id(results):
    out = {r.check: r for r in results}
    assert set(out) == ALL_IDS, "every check id must be reported exactly once"
    assert len(results) == len(ALL_IDS)
    return out


# ---------------------------------------------------------------------------
# Fixtures: each synthetic case is built at most once per module.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_aligned():
    return make_clean().artifacts.aligned()


@pytest.fixture(scope="module")
def clean_results(clean_aligned):
    return by_id(run(clean_aligned, AuditConfig()))


@pytest.fixture(scope="module")
def lookahead_results():
    return by_id(run(make_lookahead().artifacts.aligned(), AuditConfig()))


@pytest.fixture(scope="module")
def overfit_results():
    case = make_overfit()
    assert case.audit_kwargs == {"n_trials": 200}
    cfg = AuditConfig(n_trials=case.audit_kwargs["n_trials"])
    return by_id(run(case.artifacts.aligned(), cfg))


def _tiny_artifacts(n=40, m=8, seed=1, strategy_returns=None, positions=None):
    """Minimal aligned artifacts: random signals/returns, no edge."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    cols = [f"T{i}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, m)), index=dates, columns=cols)
    sigs = pd.DataFrame(rng.normal(0.0, 1.0, (n, m)), index=dates, columns=cols)
    return BacktestArtifacts(signals=sigs, asset_returns=rets,
                             strategy_returns=strategy_returns,
                             positions=positions).aligned()


@pytest.fixture(scope="module")
def tiny_results():
    """No strategy_returns, no positions, only 40 periods."""
    return by_id(run(_tiny_artifacts(), AuditConfig()))


# ---------------------------------------------------------------------------
# 1. performance.suspicious_sharpe
# ---------------------------------------------------------------------------

def test_suspicious_sharpe_fails_on_lookahead(lookahead_results):
    r = lookahead_results["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    # calibrated: net SR ~76 - the concrete number must be in the message
    assert re.search(r"\b7\d\.\d\b", r.message)
    assert "near-arbitrage" in r.message
    assert r.details["sharpe_annualized"] > AuditConfig().sharpe_fail
    assert 70 < r.details["sharpe_annualized"] < 82
    assert 0.99 <= r.details["psr_vs_zero"] <= 1.0
    assert r.remediation


def test_suspicious_sharpe_passes_on_clean(clean_results):
    r = clean_results["performance.suspicious_sharpe"]
    assert r.status is Status.PASS
    # The honest generator charges drift-aware trading costs; its expected net Sharpe
    # is about 1.79.
    assert 1.5 < r.details["sharpe_annualized"] < 2.2
    assert re.search(r"1\.[78]\d", r.message)  # PASS still states what was measured
    assert r.details["returns_source"] == "net strategy_returns"


def test_suspicious_sharpe_modest_negative_sr_passes():
    # An ordinary losing book is not suspicious. Sharpe thresholds use absolute
    # magnitude, so only a modest negative Sharpe provides a valid control; extreme
    # negative cases are covered in test_qaudit_performance_polarity.py.
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2020-01-02", periods=40)
    losing = pd.Series(-0.0005 + rng.normal(0.0, 0.005, 40), index=dates)
    res = by_id(run(_tiny_artifacts(strategy_returns=losing), AuditConfig()))
    r = res["performance.suspicious_sharpe"]
    assert r.status is Status.PASS
    assert -AuditConfig().sharpe_warn < r.details["sharpe_annualized"] < 0
    assert re.search(r"-\d+\.\d", r.message)  # the (negative) SR is stated


def test_suspicious_sharpe_uses_gross_reconstruction(clean_aligned):
    arts = dataclasses.replace(clean_aligned, strategy_returns=None)
    r = by_id(run(arts, AuditConfig()))["performance.suspicious_sharpe"]
    assert r.status is Status.PASS  # clean gross SR ~2.8, below the 3.0 warn line
    assert "gross reconstruction" in r.message
    assert "gross reconstruction" in r.details["returns_source"]
    assert r.details["sharpe_annualized"] > 0


def test_sharpe_checks_skip_without_returns_or_positions(tiny_results):
    for check_id in ("performance.suspicious_sharpe", "performance.deflated_sharpe"):
        r = tiny_results[check_id]
        assert r.status is Status.SKIP
        assert "strategy_returns" in r.message
        assert "positions" in r.message


# ---------------------------------------------------------------------------
# 2. performance.deflated_sharpe
# ---------------------------------------------------------------------------

def test_deflated_sharpe_skips_without_n_trials(clean_results):
    r = clean_results["performance.deflated_sharpe"]
    assert r.status is Status.SKIP
    assert "n_trials" in r.message
    assert "count" in r.message  # "...if you did not count, that is itself the finding"


def test_deflated_sharpe_passes_on_clean_with_12_trials(clean_aligned):
    r = by_id(run(clean_aligned, AuditConfig(n_trials=12)))["performance.deflated_sharpe"]
    assert r.status is Status.PASS
    assert r.details["dsr"] == pytest.approx(0.97, abs=0.02)  # calibrated ~0.974
    assert r.details["n_trials"] == 12
    assert "12" in r.message


def test_deflated_sharpe_flags_overfit_with_200_trials(overfit_results):
    r = overfit_results["performance.deflated_sharpe"]
    assert r.status in (Status.WARN, Status.FAIL)
    assert r.details["dsr"] < 0.6  # calibrated ~0.16: likely selection, not skill
    assert r.details["n_trials"] == 200
    # message must carry the annualized deflation story: 200 trials, sr_star ~1.39 ann
    assert "200" in r.message
    assert re.search(r"1\.3\d", r.message)
    # The selected noise book pays drift-aware trading costs and has expected
    # annualized Sharpe near 0.89.
    assert re.search(r"0\.[89]\d", r.message)
    assert r.remediation
    # details are the full _stats dict (plus annualized extras), floats/ints only
    for key in ("dsr", "sr_period", "sr_star_period", "var_trials",
                "sr_annualized", "sr_star_annualized"):
        assert isinstance(r.details[key], float)
    assert isinstance(r.details["n_obs"], int)


def test_deflated_sharpe_overfit_sharpe_itself_looks_innocent(overfit_results):
    # the whole point of the overfit case: raw SR ~0.89 sails past check 1 ...
    assert overfit_results["performance.suspicious_sharpe"].status is Status.PASS
    # ... and only trial-counting catches it (tested above)


# ---------------------------------------------------------------------------
# 3. performance.suspicious_ic
# ---------------------------------------------------------------------------

def test_suspicious_ic_fails_on_lookahead(lookahead_results):
    r = lookahead_results["performance.suspicious_ic"]
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert re.search(r"0\.95\d", r.message)          # calibrated mean IC ~0.957
    assert "0.01-0.05" in r.message                  # the realistic band for context
    assert re.search(r"t=\d", r.message)             # Newey-West t-stat cited
    assert r.details["mean_ic"] > 0.9
    assert r.details["n_dates"] > 900
    assert r.remediation


def test_suspicious_ic_passes_on_clean(clean_results):
    r = clean_results["performance.suspicious_ic"]
    assert r.status is Status.PASS
    assert 0.02 < r.details["mean_ic"] < 0.045       # calibrated ~0.031
    assert re.search(r"0\.03\d", r.message)          # PASS doubles as health readout
    assert re.search(r"t=\d", r.message)
    assert r.details["nw_tstat"] > 2.0


# ---------------------------------------------------------------------------
# 4. performance.ic_stability
# ---------------------------------------------------------------------------

def _flip_artifacts(seed=3, n_blocks=17, block=63, m=30):
    """Signal = forward return with sign flipping every `block` days plus noise:
    per-date IC ~ +/-0.5 by block, full-sample mean slightly positive (9 of the
    17 blocks are positive), so sign consistency lands near 50%."""
    rng = np.random.default_rng(seed)
    n = n_blocks * block
    dates = pd.bdate_range("2018-01-02", periods=n)
    cols = [f"S{i:02d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, m)), index=dates, columns=cols)
    fwd = rets.shift(-1)
    sign = pd.Series(np.where((np.arange(n) // block) % 2 == 0, 1.0, -1.0),
                     index=dates)
    noise = pd.DataFrame(rng.normal(0.0, 0.0173, (n, m)), index=dates, columns=cols)
    signals = (fwd.mul(sign, axis=0) + noise).fillna(0.0)
    return BacktestArtifacts(signals=signals, asset_returns=rets).aligned()


def test_ic_stability_warns_on_sign_flipping_blocks():
    r = by_id(run(_flip_artifacts(), AuditConfig()))["performance.ic_stability"]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert r.details["consistency"] < AuditConfig().ic_sign_consistency_warn
    assert r.details["consistency"] == pytest.approx(0.50, abs=0.06)
    assert r.details["n_windows"] >= 4
    assert r.details["worst_window_ic"] < 0 < r.details["best_window_ic"]
    assert "sign" in r.message and "%" in r.message
    assert r.remediation


def test_ic_stability_passes_on_clean(clean_results):
    r = clean_results["performance.ic_stability"]
    assert r.status is Status.PASS
    assert r.details["consistency"] >= AuditConfig().ic_sign_consistency_warn
    assert r.details["n_windows"] > 800


def test_ic_stability_not_judged_when_sample_too_short(tiny_results):
    # 40 periods < one 63-period window -> gate PASS with an explicit note
    r = tiny_results["performance.ic_stability"]
    assert r.status is Status.PASS
    assert re.search(r"too weak|too short", r.message)
    assert r.details["n_windows"] < 4


# ---------------------------------------------------------------------------
# 5. performance.ic_regime_concentration
# ---------------------------------------------------------------------------

def _regime_artifacts(seed=4, n=1040, m=30):
    """Pure noise signal except during 2020, where it embeds the forward
    return - all the IC comes from one calendar year."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    cols = [f"R{i:02d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, m)), index=dates, columns=cols)
    fwd = rets.shift(-1)
    signals = pd.DataFrame(rng.normal(0.0, 0.02, (n, m)), index=dates, columns=cols)
    mask = dates.year == 2020
    signals.loc[mask] = signals.loc[mask] + fwd.loc[mask].fillna(0.0)
    return BacktestArtifacts(signals=signals, asset_returns=rets).aligned()


def test_ic_regime_concentration_warns_on_one_year_edge():
    r = by_id(run(_regime_artifacts(), AuditConfig()))["performance.ic_regime_concentration"]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert r.details["top_year"] == 2020
    assert r.details["share"] > AuditConfig().ic_regime_share_warn
    assert "2020" in r.message and "%" in r.message
    assert isinstance(r.details["yearly"], dict) and len(r.details["yearly"]) >= 3
    assert r.remediation


def test_ic_regime_concentration_passes_on_clean(clean_results):
    r = clean_results["performance.ic_regime_concentration"]
    assert r.status is Status.PASS
    assert r.details["share"] <= AuditConfig().ic_regime_share_warn
    assert r.details["n_years"] >= 3


def test_ic_regime_concentration_not_judged_on_short_history(tiny_results):
    # 40 bdays = 1 calendar year -> gate PASS with an explicit note
    r = tiny_results["performance.ic_regime_concentration"]
    assert r.status is Status.PASS
    assert "not judged" in r.message
    assert r.details["n_years"] < 3


# ---------------------------------------------------------------------------
# 6. performance.sample_size
# ---------------------------------------------------------------------------

def test_sample_size_warns_when_underpowered(tiny_results):
    r = tiny_results["performance.sample_size"]
    assert r.status is Status.WARN
    assert r.severity is Severity.LOW
    assert "40" in r.message and str(AuditConfig().min_periods) in r.message
    assert r.details["n_periods"] == 40
    assert r.remediation


def test_sample_size_passes_on_clean(clean_results):
    r = clean_results["performance.sample_size"]
    assert r.status is Status.PASS
    assert r.details["n_periods"] == 1000
    assert "1000" in r.message


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------

def test_run_never_mutates_artifacts(clean_aligned):
    before = {
        "signals": clean_aligned.signals.copy(deep=True),
        "asset_returns": clean_aligned.asset_returns.copy(deep=True),
        "positions": clean_aligned.positions.copy(deep=True),
        "strategy_returns": clean_aligned.strategy_returns.copy(deep=True),
    }
    run(clean_aligned, AuditConfig(n_trials=12))
    pd.testing.assert_frame_equal(clean_aligned.signals, before["signals"])
    pd.testing.assert_frame_equal(clean_aligned.asset_returns, before["asset_returns"])
    pd.testing.assert_frame_equal(clean_aligned.positions, before["positions"])
    pd.testing.assert_series_equal(clean_aligned.strategy_returns,
                                   before["strategy_returns"])


def test_every_fail_or_warn_carries_remediation(lookahead_results, overfit_results,
                                                tiny_results):
    for results in (lookahead_results, overfit_results, tiny_results):
        for r in results.values():
            if r.status in (Status.FAIL, Status.WARN):
                assert r.remediation, f"{r.check} {r.status} lacks remediation"
