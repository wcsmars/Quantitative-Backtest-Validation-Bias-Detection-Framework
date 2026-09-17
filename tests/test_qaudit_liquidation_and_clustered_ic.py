"""Liquidation trade coverage and clustered perfect-IC evidence.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qaudit import AuditConfig, AuditFailure, BacktestArtifacts, audit
from qaudit._stats import forward_returns, gross_strategy_returns, traded_dollars_series
from qaudit.checks.leakage import _spearman_tail_ge
from qaudit.types import Severity, Status


COST_CHECK = "costs.missing_transaction_costs"
PERFECT_CHECK = "leakage.perfect_rank_dates"


def _result(report, check):
    return next(result for result in report.results if result.check == check)


def _liquidating_book(*, start=0, cost_bps=10.0):
    rng = np.random.default_rng(91)
    dates = pd.bdate_range("2024-01-01", periods=200)
    returns = pd.DataFrame(rng.normal(0, 0.01, (200, 10)), index=dates)
    signals = pd.DataFrame(rng.normal(0, 1, (200, 10)), index=dates)
    positions = pd.DataFrame(0.0, index=dates, columns=returns.columns)
    weights = rng.uniform(0, 1, (10, 10))
    weights /= weights.sum(axis=1, keepdims=True)
    positions.iloc[start:start + 100] = np.repeat(weights, 10, axis=0)
    dollars = traded_dollars_series(positions, returns)
    net = gross_strategy_returns(positions, returns) - dollars * cost_bps * 1e-4
    artifacts = BacktestArtifacts(
        signals=signals, asset_returns=returns, positions=positions,
        strategy_returns=net, declared_costs_bps=cost_bps)
    return artifacts, dollars, start + 100


@pytest.mark.parametrize("start,cost_bps", [(0, 10.0), (37, 10.0), (0, 0.0)])
def test_missing_liquidation_cannot_certify_all_traded_dollars(start, cost_bps):
    artifacts, dollars, exit_row = _liquidating_book(start=start, cost_bps=cost_bps)
    net = artifacts.strategy_returns.copy()
    net.iloc[exit_row:] = np.nan
    report = audit(replace(artifacts, strategy_returns=net), include=[COST_CHECK])
    result = _result(report, COST_CHECK)

    # The omission hides a full-book liquidation, well above the allowed 1%.
    assert dollars.iloc[exit_row] == pytest.approx(1.0)
    true_coverage = dollars.iloc[:exit_row].sum() / dollars.sum()
    assert true_coverage < 0.90
    assert result.status is Status.SKIP
    assert result.details["overlap_traded_share"] == pytest.approx(true_coverage)
    assert result.details["n_uncompared_trade_dates"] == 1
    assert "liquidation" in result.message
    with pytest.raises(AuditFailure):
        report.gate(require=[COST_CHECK])


@pytest.mark.parametrize("start", [0, 37])
def test_correct_liquidation_passes_without_later_flat_returns(start):
    artifacts, _, exit_row = _liquidating_book(start=start)
    net = artifacts.strategy_returns.copy()
    net.iloc[:start] = np.nan
    net.iloc[exit_row + 1:] = np.nan
    result = _result(audit(replace(artifacts, strategy_returns=net),
                           include=[COST_CHECK]), COST_CHECK)
    assert result.status is Status.PASS, result.message
    assert result.details["implied_bps"] == pytest.approx(10.0)
    assert result.details["overlap_traded_share"] == pytest.approx(1.0)
    # Entry/exit coverage must not inflate the history floor with flat padding.
    assert result.details["n_live_dates"] == 100
    assert result.details["n_compared_live_dates"] == 100
    assert result.details["n_uncompared_trade_dates"] == 0


def test_reported_but_uncharged_liquidation_still_warns():
    artifacts, _, exit_row = _liquidating_book()
    net = artifacts.strategy_returns.copy()
    net.iloc[exit_row:] = 0.0
    result = _result(audit(replace(artifacts, strategy_returns=net),
                           include=[COST_CHECK]), COST_CHECK)
    assert result.status is Status.WARN
    assert result.severity is Severity.MEDIUM
    assert "do not track dollars traded date by date" in result.message


def test_undefined_final_liquidation_cannot_disappear_from_ledger():
    artifacts, _, exit_row = _liquidating_book()
    returns = artifacts.asset_returns.copy()
    # Exact weights ensure exact zero NAV on the last invested date.
    positions = artifacts.positions.copy()
    positions.iloc[exit_row - 1] = 0.0
    positions.iloc[exit_row - 1, 0] = 1.0
    returns.iloc[exit_row - 1, 0] = -1.0
    net = artifacts.strategy_returns.copy()
    net.iloc[exit_row:] = np.nan
    report = audit(replace(artifacts, positions=positions,
                           asset_returns=returns, strategy_returns=net),
                   include=[COST_CHECK])
    result = _result(report, COST_CHECK)
    assert result.status is Status.FAIL
    assert result.severity is Severity.CRITICAL
    assert result.details["trade_ledger_undefined"] is True
    assert result.details["n_nonfinite_trade_rows"] == 1
    assert result.details["first_nonfinite_trade_date"] == str(returns.index[exit_row].date())
    with pytest.raises(AuditFailure):
        report.gate(require=[COST_CHECK])


def _fixed_signal_artifacts(seed, horizon, *, tied=False):
    dates = pd.bdate_range("2022-01-03", periods=500)
    # Known before the first return: a fixed score per asset has no target access.
    scores = np.arange(8) // 2 if tied else np.arange(8)
    signals = pd.DataFrame(np.tile(scores, (500, 1)), index=dates)
    returns = pd.DataFrame(np.random.default_rng(seed).normal(0, 0.01, (500, 8)),
                           index=dates)
    return BacktestArtifacts(signals=signals, asset_returns=returns,
                              label_horizon=horizon)


@pytest.mark.parametrize("seed,horizon", [
    (7, 21), (18, 21), (24, 21), (26, 21), (50, 21), (53, 21),
    (0, 63), (77, 63), (20, 126), (66, 126),
])
def test_overlapping_labels_do_not_convict_fixed_known_scores(seed, horizon):
    artifacts = _fixed_signal_artifacts(seed, horizon)
    result = _result(audit(artifacts, include=[PERFECT_CHECK]), PERFECT_CHECK)
    assert result.status is Status.PASS, result.message
    assert result.details["cluster_factor"] == pytest.approx(horizon)
    assert result.details["n_perfect"] <= result.details["noise_bound"]


def test_monthly_fixed_scores_pass_the_whole_leakage_family():
    report = audit(_fixed_signal_artifacts(18, 21), include=["leakage"])
    assert len(report.results) == 4
    assert all(result.status is Status.PASS for result in report.results)
    assert _result(report, PERFECT_CHECK).details["n_perfect"] == 15


def test_horizon_one_matches_unclustered_count_bound():
    result = _result(audit(_fixed_signal_artifacts(18, 1),
                           include=[PERFECT_CHECK]), PERFECT_CHECK)
    expected = 499 * 2 * _spearman_tail_ge(8, AuditConfig().leak_perfect_ic)
    unclustered_bound = expected + 4 * np.sqrt(max(expected, 0.25)) + 1
    assert result.status is Status.PASS
    assert result.details["n_perfect"] == 1
    assert result.details["cluster_factor"] == 1.0
    assert result.details["noise_bound"] == round(unclustered_bound, 2)


def test_tied_persistent_scores_keep_their_conditional_null():
    result = _result(audit(_fixed_signal_artifacts(18, 21, tied=True),
                           include=[PERFECT_CHECK]), PERFECT_CHECK)
    assert result.status is Status.PASS, result.message
    assert result.details["cluster_factor"] == 21.0
    assert result.details["expected_noise_dates"] > 0.0


@pytest.mark.parametrize("horizon", [1, 21, 63, 126])
@pytest.mark.parametrize("intermittent", [False, True])
def test_real_target_leaks_remain_detected_across_horizons(horizon, intermittent):
    artifacts = _fixed_signal_artifacts(4, horizon)
    future = forward_returns(artifacts.asset_returns, horizon)
    if intermittent:
        signals = pd.DataFrame(np.random.default_rng(5).normal(0, 1, future.shape),
                                index=future.index, columns=future.columns)
        signals.iloc[::3] = future.iloc[::3]
    else:
        signals = future
    result = _result(audit(replace(artifacts, signals=signals),
                           include=[PERFECT_CHECK]), PERFECT_CHECK)
    assert result.status is Status.FAIL, result.message
    assert result.severity is Severity.CRITICAL
    assert result.details["n_perfect"] > result.details["noise_bound"]
