"""Tests for qaudit.dynamic.probes_null (placebo + shuffled-label probes).

Calls the module's run() directly on aligned artifacts - never through
qaudit.audit() - and builds each synthetic case at most once per module.
"""
from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from qaudit.config import AuditConfig
from qaudit.dynamic import probes_null
from qaudit.dynamic.probes_null import run
from qaudit.synthetic import make_clean
from qaudit.types import Severity, Status

CHECK_IDS = {
    "dynamic.placebo_pipeline_bias",
    "dynamic.placebo_percentile",
    "dynamic.shuffled_labels",
}


# ---------------------------------------------------------------------------
# Fixtures (module-scoped: build the synthetic case and run the probes once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean():
    return make_clean()


@pytest.fixture(scope="module")
def aligned(clean):
    return clean.artifacts.aligned()


@pytest.fixture(scope="module")
def cfg():
    return AuditConfig(n_placebo=40, n_shuffle=60, seed=1)


@pytest.fixture(scope="module")
def clean_results(aligned, clean, cfg):
    return run(aligned, cfg, backtest_func=clean.backtest_func)


def one(results, check_id):
    matches = [r for r in results if r.check == check_id]
    assert len(matches) == 1, f"expected exactly one {check_id}, got {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# Clean case: false-positive guard for every detector
# ---------------------------------------------------------------------------

def test_clean_emits_exactly_the_three_check_ids(clean_results):
    assert {r.check for r in clean_results} == CHECK_IDS
    assert len(clean_results) == 3


def test_clean_all_pass(clean_results):
    for r in clean_results:
        assert r.status is Status.PASS, f"{r.check}: {r.status} - {r.message}"


def test_clean_placebo_bias_passes_with_cost_drag_note(clean_results, cfg):
    r = one(clean_results, "dynamic.placebo_pipeline_bias")
    # PASS message must still say what was measured
    assert re.search(r"mean annualized SR -?\d+\.\d+", r.message)
    assert "cost drag" in r.message
    assert r.details["n"] == cfg.n_placebo
    # honest pipeline: random signals lose to costs
    assert r.details["mean_null"] < cfg.placebo_null_sharpe_fail
    assert r.details["mean_null"] < 0.0
    assert np.isfinite(r.details["std_null"]) and r.details["std_null"] > 0


def test_clean_percentile_above_bar(clean_results, cfg):
    r = one(clean_results, "dynamic.placebo_percentile")
    assert r.status is Status.PASS
    # calibrated fact: clean net SR 1.79, far above every placebo outcome
    assert r.details["actual_sr"] == pytest.approx(1.79, abs=0.02)
    assert r.details["percentile"] >= cfg.placebo_percentile_warn
    assert r.details["null_q05"] <= r.details["null_q50"] <= r.details["null_q95"]
    assert r.details["actual_sr"] > r.details["null_q95"]


def test_clean_shuffled_labels_significant(clean_results, cfg):
    r = one(clean_results, "dynamic.shuffled_labels")
    assert r.status is Status.PASS
    assert r.details["p_value"] < 0.05
    # best possible p with 60 permutations and zero exceedances
    assert r.details["p_value"] == pytest.approx(1.0 / 61.0)
    assert r.details["mean_null"] < cfg.shuffle_leak_ratio * r.details["actual_sr"]
    assert re.search(r"p=0\.0\d+", r.message)


# ---------------------------------------------------------------------------
# Check 1 positive: a pipeline that manufactures performance
# ---------------------------------------------------------------------------

def test_placebo_pipeline_bias_fails_on_phantom_credit(aligned, clean, cfg):
    bt = clean.backtest_func

    def bt_biased(signals, asset_returns):
        # phantom 5bp/day credit - stand-in for an engine-level accounting bug
        return bt(signals, asset_returns) + 0.0005

    results = run(aligned, cfg, backtest_func=bt_biased)
    r = one(results, "dynamic.placebo_pipeline_bias")
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["mean_null"] >= cfg.placebo_null_sharpe_fail
    assert r.details["t"] >= 3.0
    # message carries the concrete numbers and blames the pipeline
    assert f"{r.details['mean_null']:.2f}" in r.message
    assert "pipeline" in r.message
    assert r.remediation


# ---------------------------------------------------------------------------
# Check 3 positive (leak path): PnL independent of signal/return alignment
# ---------------------------------------------------------------------------

def test_shuffled_labels_fails_on_constant_pnl(aligned, cfg):
    def bt_const(signals, asset_returns):
        return pd.Series(0.001, index=asset_returns.index)

    results = run(aligned, cfg, backtest_func=bt_const)
    r = one(results, "dynamic.shuffled_labels")
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert re.search(r"shuffl", r.message, re.I)
    # actual == every null: p pinned at 1, mean null equals the actual SR
    assert r.details["p_value"] == pytest.approx(1.0)
    assert r.details["mean_null"] == pytest.approx(r.details["actual_sr"], rel=1e-9)
    assert r.remediation
    # the same constant credit also means random signals "earn" it: check 1
    # must call out the manufactured performance too (identical nulls -> t=inf)
    b = one(results, "dynamic.placebo_pipeline_bias")
    assert b.status is Status.FAIL
    assert b.severity is Severity.CRITICAL


# ---------------------------------------------------------------------------
# Check 2 + check 3 WARN paths: honest pipeline, signal with no edge
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def noise_results(aligned, clean, cfg):
    rng = np.random.default_rng(123)
    noise = pd.DataFrame(
        rng.standard_normal(aligned.signals.shape),
        index=aligned.signals.index, columns=aligned.signals.columns,
    ).where(aligned.signals.notna())   # same shape and NaN mask as the real signal
    art = dataclasses.replace(aligned, signals=noise)
    return run(art, cfg, backtest_func=clean.backtest_func)


def test_noise_signal_percentile_warns(noise_results, cfg):
    r = one(noise_results, "dynamic.placebo_percentile")
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["percentile"] < cfg.placebo_percentile_warn
    assert re.search(r"\d+th percentile", r.message)
    assert r.remediation


def test_noise_signal_shuffled_labels_warns(noise_results, cfg):
    r = one(noise_results, "dynamic.shuffled_labels")
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["p_value"] >= cfg.shuffle_pvalue_warn
    assert f"p={r.details['p_value']:.2f}" in r.message
    assert r.remediation


def test_noise_signal_does_not_fail_pipeline_bias(noise_results):
    # honest engine: even with a junk signal, check 1 must stay green
    r = one(noise_results, "dynamic.placebo_pipeline_bias")
    assert r.status is Status.PASS


# ---------------------------------------------------------------------------
# SKIP path: no backtest_func
# ---------------------------------------------------------------------------

def test_skips_without_backtest_func(aligned, cfg):
    results = run(aligned, cfg, backtest_func=None)
    assert {r.check for r in results} == CHECK_IDS
    assert len(results) == 3
    for r in results:
        assert r.status is Status.SKIP
        # skip message must state the exact callable contract
        assert "backtest_func(signals" in r.message
        assert "asset_returns" in r.message
        assert "pd.Series" in r.message


# ---------------------------------------------------------------------------
# Probe health: a backtest_func whose null re-runs are all degenerate
# ---------------------------------------------------------------------------

def test_probe_health_warns_when_null_runs_degenerate(aligned, cfg):
    def bt_broken(signals, asset_returns):
        raise ValueError("engine exploded")

    results = run(aligned, cfg, backtest_func=bt_broken)
    by_id = {r.check: r for r in results}
    # all three contracted ids still present (as SKIPs), never silently omitted
    assert CHECK_IDS <= set(by_id)
    for cid in CHECK_IDS:
        assert by_id[cid].status is Status.SKIP
    health = by_id["dynamic.probe_health"]
    assert health.status is Status.WARN
    assert health.severity is Severity.LOW
    # names the counts
    assert f"{cfg.n_placebo}/{cfg.n_placebo}" in health.message
    assert f"{cfg.n_shuffle}/{cfg.n_shuffle}" in health.message
    assert health.details["n_placebo_bad"] == cfg.n_placebo
    assert health.details["n_shuffle_bad"] == cfg.n_shuffle


# ---------------------------------------------------------------------------
# Determinism: same config/seed -> identical numbers
# ---------------------------------------------------------------------------

def test_deterministic_given_seed(aligned, clean, clean_results):
    cfg2 = AuditConfig(n_placebo=40, n_shuffle=60, seed=1)  # fresh but equal
    rerun = run(aligned, cfg2, backtest_func=clean.backtest_func)
    a = one(clean_results, "dynamic.shuffled_labels")
    b = one(rerun, "dynamic.shuffled_labels")
    assert a.details["p_value"] == b.details["p_value"]
    assert a.details["mean_null"] == b.details["mean_null"]
    pa = one(clean_results, "dynamic.placebo_pipeline_bias")
    pb = one(rerun, "dynamic.placebo_pipeline_bias")
    assert pa.details["mean_null"] == pb.details["mean_null"]
    assert pa.details["t"] == pb.details["t"]


def test_fractional_periods_per_year_reaches_every_pipeline_run(
        aligned, cfg, monkeypatch):
    """Sub-annual data must not be truncated to a zero Sharpe scale."""
    seen: list[float] = []

    def record_pipeline(backtest_func, signals, asset_returns,
                        periods_per_year):
        seen.append(periods_per_year)
        return 0.25, 0.25

    monkeypatch.setattr(probes_null, "_run_pipeline", record_pipeline)
    fractional = dataclasses.replace(aligned, periods_per_year=0.5)
    probes_null.run(fractional, cfg, backtest_func=lambda *_: None)

    assert seen
    assert set(seen) == {0.5}
