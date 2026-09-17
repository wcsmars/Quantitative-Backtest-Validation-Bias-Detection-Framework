"""Dynamic probes must audit complete returns on the supplied calendar."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit import AuditConfig, BacktestArtifacts, audit
from qaudit._stats import effective_sample_size, probabilistic_sharpe_ratio
from qaudit.dynamic._pipeline import PipelineOutputError, validated_returns
from qaudit.dynamic.probes_shift import _date_shift
from qaudit.synthetic import make_clean
from qaudit.types import Severity, Status


@pytest.fixture(scope="module")
def case():
    return make_clean()


def _malformed(out: pd.Series, defect: str):
    if defect == "rangeindex":
        return out.reset_index(drop=True)
    if defect == "dropped_dates":
        return out.iloc[::2]
    if defect == "missing_losses":
        return out.where(out >= 0)
    if defect == "duplicates":
        return pd.concat([out.iloc[:1], out])
    if defect == "reversed":
        return out.iloc[::-1]
    if defect == "foreign_calendar":
        return out.set_axis(out.index + pd.Timedelta(days=5000))
    if defect == "bool":
        return out >= 0
    if defect == "complex":
        return out.astype(complex) + 1j
    if defect == "infinite":
        out = out.copy()
        out.iloc[10] = np.inf
        return out
    if defect == "list":
        return out.tolist()
    if defect == "short_history":
        return out.iloc[-10:]
    raise AssertionError(defect)


@pytest.mark.parametrize("defect", [
    "rangeindex", "dropped_dates", "missing_losses", "duplicates", "reversed",
    "foreign_calendar", "bool", "complex", "infinite", "list", "short_history",
])
def test_date_shift_rejects_invalid_callback_returns(case, defect):
    def backtest(signals, returns):
        return _malformed(case.backtest_func(signals, returns), defect)

    result = _date_shift(case.artifacts.aligned(), AuditConfig(), backtest)

    assert result.status is Status.ERROR
    assert "invalid net returns" in result.message
    assert "backtest_func output" in result.message
    assert "not trusted" in result.message


@pytest.mark.parametrize("defect", ["rangeindex", "dropped_dates", "short_history"])
def test_null_probes_cannot_pass_on_misaligned_or_selected_dates(case, defect):
    def backtest(signals, returns):
        return _malformed(case.backtest_func(signals, returns), defect)

    report = audit(case.artifacts, AuditConfig(n_placebo=20, n_shuffle=20),
                   backtest_func=backtest,
                   include=["dynamic.placebo", "dynamic.shuffled", "dynamic.probe_health"])

    for check in ("dynamic.placebo_pipeline_bias", "dynamic.placebo_percentile",
                  "dynamic.shuffled_labels"):
        assert report[check].status is Status.SKIP
    health = report["dynamic.probe_health"]
    assert health.status is Status.WARN
    assert health.severity is Severity.HIGH
    assert health.details["n_invalid_outputs"] == 61
    assert "backtest_func output" in health.message
    assert "supplied calendar" in health.remediation


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("padded", [False, True])
def test_callback_returns_allow_dense_warmup_and_early_stop(case, dtype, padded):
    returns = case.artifacts.strategy_returns.astype(dtype).copy()
    if padded:
        returns.iloc[:30] = np.nan
        returns.iloc[-10:] = np.nan
    else:
        returns = returns.iloc[30:-10]

    validated = validated_returns(returns, case.artifacts.asset_returns.index)

    pd.testing.assert_series_equal(validated, returns.astype(float))
    assert validated is not returns


def test_missing_calendar_dates_are_rejected_even_after_dropna(case):
    returns = case.artifacts.strategy_returns.copy()
    returns.iloc[50] = np.nan
    with pytest.raises(PipelineOutputError, match="inside its observed span"):
        validated_returns(returns.dropna(), case.artifacts.asset_returns.index)


def test_exact_minimum_callback_return_span_is_allowed(case):
    returns = case.artifacts.strategy_returns.iloc[-30:]
    pd.testing.assert_series_equal(
        validated_returns(returns, case.artifacts.asset_returns.index), returns)


def test_clean_callback_keeps_existing_dynamic_verdicts(case):
    report = audit(case.artifacts, AuditConfig(n_placebo=20, n_shuffle=20),
                   backtest_func=case.backtest_func,
                   include=["dynamic.placebo", "dynamic.shuffled",
                            "dynamic.probe_health", "dynamic.date_shift"])
    assert not report.errors
    assert not report.warnings
    assert all(result.status is Status.PASS for result in report.results)


def test_reported_psr_uses_same_serial_dependence_adjustment_as_dsr(case):
    from qaudit.checks.performance import _suspicious_sharpe

    block = np.random.default_rng(7).normal(0, 0.01, 50)
    block = block - block.mean() + 0.12 * block.std(ddof=1)
    returns = pd.Series(np.tile(block, 20))
    n_eff = effective_sample_size(returns)["n_eff"]
    result = _suspicious_sharpe(returns, "net strategy_returns",
                                 case.artifacts, AuditConfig())

    corrected = probabilistic_sharpe_ratio(returns, n_eff=n_eff)
    assert result.details["psr_vs_zero"] == corrected
    assert result.details["n_eff"] == n_eff
    assert corrected < probabilistic_sharpe_ratio(returns) - 0.05


def test_dependent_dsr_skip_explains_effective_observation_shortfall():
    index = pd.bdate_range("2020-01-01", periods=1000)
    rng = np.random.default_rng(92)
    signals = pd.DataFrame(rng.normal(size=(1000, 5)), index=index)
    market_returns = pd.DataFrame(rng.normal(0, 0.01, (1000, 5)), index=index)
    returns = pd.Series(0.001 * (0.2 + np.sin(np.arange(1000) * np.pi / 1000)),
                        index=index)
    artifacts = BacktestArtifacts(signals, market_returns, strategy_returns=returns)

    report = audit(artifacts, AuditConfig(n_trials=10),
                   include=["performance.deflated_sharpe"])

    result = report["performance.deflated_sharpe"]
    assert result.status is Status.SKIP
    assert result.details["n_obs"] == 1000
    assert result.details["n_eff"] < 30
    assert "serial dependence" in result.message
    assert "1000 observed" in result.message
    assert "effective independent" in result.message
