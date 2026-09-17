"""Dynamic-probe numeric boundaries and frozen-scaler identification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit import AuditConfig, BacktestArtifacts, audit
from qaudit.dynamic import probes_null
from qaudit.types import Status


@pytest.mark.parametrize("alpha,expected", [
    (0.1, 20), (0.05, 20), (0.01, 100), (0.001, 1000), (0.0, None),
])
def test_ordinary_placebo_resolution_cutoffs(alpha, expected):
    assert probes_null._minimum_placebo_runs(alpha) == expected


def test_decimal_probability_equality_does_not_clear_threshold():
    assert not probes_null._p_clears_alpha(0.05, 1 - 0.95)
    assert not probes_null._p_clears_alpha(0.1, 1 - 0.9)
    assert probes_null._p_clears_alpha(1 / 21, 1 - 0.95)


@pytest.mark.parametrize("percentile", [1 - 1e-10, 1 - 1e-12, 1 - 1e-16])
def test_near_one_percentile_resolution_is_bounded_and_attainable(
        monkeypatch, percentile):
    config = AuditConfig(placebo_percentile_warn=percentile)
    alpha = 1 - config.placebo_percentile_warn
    compare = probes_null._p_clears_alpha
    comparisons = 0

    def bounded_compare(p_value, threshold):
        nonlocal comparisons
        comparisons += 1
        # A deterministic work budget catches an unbounded increment loop
        # without relying on machine speed or hanging the test suite.
        assert comparisons < 200
        return compare(p_value, threshold)

    monkeypatch.setattr(probes_null, "_p_clears_alpha", bounded_compare)
    required = probes_null._minimum_placebo_runs(alpha)
    assert required is not None
    assert compare(1 / (required + 1), alpha)
    assert not compare(1 / required, alpha)
    assert compare(0, alpha)


@pytest.mark.parametrize("nulls,expected", [
    (np.full(20, -np.inf), (-np.inf, -np.inf, -np.inf)),
    # With 21 observations the 5% quantile is exactly the first finite
    # order statistic, so the preceding -inf must not contaminate it.
    (np.r_[-np.inf, np.arange(20.0)], (0.0, 9.0, 18.0)),
    # Between -inf and a finite observation the linear quantile is -inf.
    (np.r_[-np.inf, np.arange(19.0)], (-np.inf, 8.5, 17.05)),
])
def test_constant_loss_quantiles_keep_meaningful_order_statistics(nulls, expected):
    result = probes_null._check_placebo_percentile(
        AuditConfig(), nulls, actual_sr=1.0, n_pos_active=0)
    quantiles = tuple(result.details[k]
                      for k in ("null_q05", "null_q50", "null_q95"))
    assert quantiles == pytest.approx(expected)
    assert not np.isnan(quantiles).any()


def test_finite_placebo_quantile_arithmetic_is_unchanged():
    nulls = np.random.default_rng(71).normal(size=100)
    result = probes_null._check_placebo_percentile(
        AuditConfig(), nulls, actual_sr=2.0, n_pos_active=0)
    quantiles = [result.details[k]
                 for k in ("null_q05", "null_q50", "null_q95")]
    np.testing.assert_array_equal(quantiles, np.percentile(nulls, [5, 50, 95]))


def test_positive_infinite_nulls_still_skip_strategy_percentile():
    result = probes_null._check_placebo_percentile(
        AuditConfig(), np.r_[np.inf, -np.inf, np.zeros(18)],
        actual_sr=1.0, n_pos_active=1)
    assert result.status is Status.SKIP
    assert result.details["n_degenerate"] == 1
    assert "dynamic.placebo_pipeline_bias" in result.message


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(71)
    dates = pd.bdate_range("2019-01-02", periods=600)
    raw = pd.DataFrame(
        np.cumsum(rng.normal(0, 0.05, (600, 6)), axis=0)
        + rng.normal(size=(600, 6)), index=dates)
    # Different finite histories exercise population/sample conversion
    # with per-column counts instead of one shared row count.
    raw.iloc[:40, 0] = np.nan
    raw.iloc[:75, 1] = np.nan
    returns = pd.DataFrame(rng.normal(0, 0.01, (600, 6)), index=dates)
    return raw, returns


def test_constant_loss_callback_does_not_crash_null_module(panels):
    raw, returns = panels
    artifacts = BacktestArtifacts(signals=raw, asset_returns=returns)

    def backtest(signals, asset_returns):
        return pd.Series(-1 / 1024, index=asset_returns.index)

    report = audit(
        artifacts, AuditConfig(n_placebo=20, n_shuffle=20),
        backtest_func=backtest,
        include=["dynamic.placebo", "dynamic.shuffled", "dynamic.probe_health"],
    )
    assert len(report.results) == 3
    assert all(result.status is not Status.ERROR for result in report.results)
    assert report["dynamic.placebo_pipeline_bias"].status is Status.PASS
    assert report["dynamic.placebo_percentile"].details["null_q50"] == -np.inf


def test_public_percentile_probe_reports_impractical_resolution(panels, monkeypatch):
    raw, returns = panels
    artifacts = BacktestArtifacts(signals=raw, asset_returns=returns)
    compare = probes_null._p_clears_alpha
    comparisons = 0

    def bounded_compare(p_value, threshold):
        nonlocal comparisons
        comparisons += 1
        assert comparisons < 200
        return compare(p_value, threshold)

    def backtest(signals, asset_returns):
        return (signals.shift(1).fillna(0) * asset_returns).mean(axis=1)

    monkeypatch.setattr(probes_null, "_p_clears_alpha", bounded_compare)
    report = audit(
        artifacts,
        AuditConfig(n_placebo=20, n_shuffle=20,
                    placebo_percentile_warn=1 - 1e-16),
        backtest_func=backtest, include=["dynamic.placebo_percentile"],
    )
    result = report["dynamic.placebo_percentile"]
    assert result.status is Status.WARN
    assert result.details["resolution_sufficient"] is False
    assert result.details["min_null_runs"] > 10**15


@pytest.mark.parametrize("ddof", [0, 1])
def test_frozen_full_sample_zscore_gets_advisory_for_either_variance(panels, ddof):
    raw, returns = panels
    mu, sd = raw.mean(), raw.std(ddof=ddof)

    def signal_func(inp):
        return (inp - mu) / sd

    artifacts = BacktestArtifacts(signals=signal_func(raw),
                                  asset_returns=returns, signal_input=raw)
    report = audit(artifacts, signal_func=signal_func,
                   include=["dynamic.signal", "dynamic.rolling"])
    assert all(result.status is Status.PASS for result in report.results)
    result = report["dynamic.rolling_window_integrity"]
    assert "z-score" in result.details["fullsample_scaler_fingerprint"]
    assert "ADVISORY" in result.message


@pytest.mark.parametrize("kind", ["expanding", "cross_sectional"])
def test_causal_normalization_does_not_get_full_sample_advisory(panels, kind):
    raw, returns = panels

    def signal_func(inp):
        if kind == "expanding":
            return (inp - inp.expanding(30).mean()) / inp.expanding(30).std(ddof=0)
        return inp.sub(inp.mean(axis=1), axis=0).div(inp.std(axis=1), axis=0)

    artifacts = BacktestArtifacts(signals=signal_func(raw),
                                  asset_returns=returns, signal_input=raw)
    report = audit(artifacts, signal_func=signal_func,
                   include=["dynamic.signal", "dynamic.rolling"])
    assert all(result.status is Status.PASS for result in report.results)
    result = report["dynamic.rolling_window_integrity"]
    assert result.details["fullsample_scaler_fingerprint"] is None
    assert "ADVISORY" not in result.message
