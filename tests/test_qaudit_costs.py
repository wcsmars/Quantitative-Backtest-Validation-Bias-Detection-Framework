"""Tests for qaudit.checks.costs (transaction-cost realism)."""
import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from qaudit.checks.costs import run
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import make_clean, make_no_costs, net_returns
from qaudit.types import Severity, Status

C1 = "costs.missing_transaction_costs"
C2 = "costs.no_cost_declaration"
C3 = "costs.turnover_unrealistic"
C4 = "costs.cost_sensitivity"
C5 = "costs.price_return_consistency"
ALL_IDS = {C1, C2, C3, C4, C5}


def by_id(results):
    return {r.check: r for r in results}


# ---------------------------------------------------------------------------
# fixtures: build each synthetic case exactly once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return AuditConfig()


@pytest.fixture(scope="module")
def clean_case():
    return make_clean()


@pytest.fixture(scope="module")
def clean_aligned(clean_case):
    return clean_case.artifacts.aligned()


@pytest.fixture(scope="module")
def clean_results(clean_aligned, cfg):
    return by_id(run(clean_aligned, cfg))


@pytest.fixture(scope="module")
def no_costs_results(cfg):
    case = make_no_costs()
    return by_id(run(case.artifacts.aligned(), cfg))


# ---------------------------------------------------------------------------
# module contract
# ---------------------------------------------------------------------------

def test_emits_exactly_the_registered_check_ids(clean_results):
    assert set(clean_results) == ALL_IDS


def test_run_does_not_mutate_artifacts(clean_aligned, cfg):
    pos_before = clean_aligned.positions.copy()
    sr_before = clean_aligned.strategy_returns.copy()
    run(clean_aligned, cfg)
    pd.testing.assert_frame_equal(clean_aligned.positions, pos_before)
    pd.testing.assert_series_equal(clean_aligned.strategy_returns, sr_before)


# ---------------------------------------------------------------------------
# 1. costs.missing_transaction_costs
# ---------------------------------------------------------------------------

def test_no_costs_case_fails_missing_costs(no_costs_results):
    r = no_costs_results[C1]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert re.search(r"identical", r.message)
    assert re.search(r"37", r.message)          # 0.373/day one-sided turnover
    assert r.details["max_abs_diff"] <= 1e-12
    assert 0.30 < r.details["mean_turnover"] < 0.45
    assert r.remediation


def test_clean_passes_with_implied_cost_near_declared(clean_results):
    r = clean_results[C1]
    assert r.status is Status.PASS
    assert 5.0 < r.details["implied_bps"] < 20.0  # true one-way cost is 10bps
    assert abs(r.details["implied_bps"] - 10.0) < 0.5
    assert re.search(r"implied", r.message) and re.search(r"10\.0", r.message)


def test_undercharged_declared_costs_warns(clean_case, cfg):
    # Pipeline charges 2bps but declares 30bps: implied < declared/3.
    art = clean_case.artifacts
    net2 = net_returns(art.positions, art.asset_returns, costs_bps=2.0)
    rigged = dataclasses.replace(
        art, strategy_returns=net2, declared_costs_bps=30.0).aligned()
    r = by_id(run(rigged, cfg))[C1]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert re.search(r"under-?charg", r.message)
    assert re.search(r"30", r.message)
    assert abs(r.details["implied_bps"] - 2.0) < 0.5


def test_overcharged_vs_declared_noted_in_pass(clean_case, cfg):
    # Pipeline charges 10bps but declares only 1bps: implied > 3x declared,
    # still a PASS (costs are charged) but the discrepancy must be noted.
    rigged = dataclasses.replace(
        clean_case.artifacts, declared_costs_bps=1.0).aligned()
    r = by_id(run(rigged, cfg))[C1]
    assert r.status is Status.PASS
    assert re.search(r"3x", r.message) and re.search(r"disagree", r.message)


def test_buy_and_hold_is_graceful_not_fail(cfg):
    # A true buy-and-hold book: enter once, never trade again - weights
    # drift with returns, w[t] = w[t-1]*(1+r[t-1])/(1+port[t-1]). Net
    # bit-identical to gross must not fail: drift-aware turnover is ~0, so
    # there is genuinely nothing to charge. (A constant-weight panel is not
    # buy-and-hold: constant weights are a daily-rebalanced constant-mix
    # book that re-buys against drift every bar, so net == gross draws the
    # missing-costs FAIL there - see tests/test_qaudit_trade_economics.py.)
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=200)
    cols = [f"S{i}" for i in range(4)]
    rets = pd.DataFrame(rng.normal(3e-4, 0.01, (200, 4)),
                        index=dates, columns=cols)
    r = rets.to_numpy()
    w = np.empty_like(r)
    w[0] = 0.25
    for t in range(1, len(r)):                     # pure drift, no trades
        w[t] = w[t - 1] * (1.0 + r[t - 1]) / (1.0 + w[t - 1] @ r[t - 1])
    pos = pd.DataFrame(w, index=dates, columns=cols)
    gross = (pos * rets).sum(axis=1)
    art = BacktestArtifacts(signals=pos.copy(), asset_returns=rets,
                            positions=pos, strategy_returns=gross).aligned()
    res = by_id(run(art, cfg))
    r1 = res[C1]
    assert r1.status is Status.PASS
    assert re.search(r"negligible", r1.message)
    assert r1.details["mean_turnover"] < 0.005     # entry trade only, /200 bars
    assert res[C3].status is Status.PASS
    assert res[C3].details["mean_turnover"] < 1e-9  # zero post-entry trading
    assert res[C4].status is Status.PASS


def test_missing_positions_skips_check1_naming_artifact(clean_case, cfg):
    art = dataclasses.replace(clean_case.artifacts, positions=None,
                              declared_costs_bps=None).aligned()
    r = by_id(run(art, cfg))[C1]
    assert r.status is Status.SKIP
    assert re.search(r"positions", r.message)
    assert re.search(r"strategy_returns", r.message)  # skip names both needs


def test_missing_strategy_returns_skips_check1(clean_case, cfg):
    art = dataclasses.replace(clean_case.artifacts,
                              strategy_returns=None).aligned()
    r = by_id(run(art, cfg))[C1]
    assert r.status is Status.SKIP
    assert re.search(r"strategy_returns", r.message)


# ---------------------------------------------------------------------------
# 2. costs.no_cost_declaration
# ---------------------------------------------------------------------------

def test_no_declaration_but_check1_ran_passes(no_costs_results):
    # no_costs declares nothing, but positions+strategy_returns are present,
    # so declaration is PASS with "verified against reconstruction" wording
    # (the actual free-trading defect is check 1's CRITICAL FAIL).
    r = no_costs_results[C2]
    assert r.status is Status.PASS
    assert re.search(r"reconstruction", r.message)


def test_no_declaration_and_nothing_to_verify_warns(clean_case, cfg):
    art = dataclasses.replace(clean_case.artifacts, positions=None,
                              strategy_returns=None,
                              declared_costs_bps=None).aligned()
    r = by_id(run(art, cfg))[C2]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert re.search(r"no declared costs", r.message)
    assert re.search(r"nothing to verify", r.message)


def test_zero_declared_costs_warns_high(clean_case, cfg):
    art = dataclasses.replace(clean_case.artifacts,
                              declared_costs_bps=0.0).aligned()
    r = by_id(run(art, cfg))[C2]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert re.search(r"zero declared costs", r.message)


def test_clean_declaration_passes_citing_bps(clean_results):
    r = clean_results[C2]
    assert r.status is Status.PASS
    assert re.search(r"10", r.message)
    assert r.details["declared_costs_bps"] == 10.0


# ---------------------------------------------------------------------------
# 3. costs.turnover_unrealistic
# ---------------------------------------------------------------------------

def test_no_costs_turnover_warns(no_costs_results, cfg):
    r = no_costs_results[C3]
    assert r.status is Status.WARN
    assert 0.30 < r.details["mean_turnover"] < 0.45   # calibrated 0.373/day
    assert r.details["mean_turnover"] >= cfg.turnover_daily_warn
    assert re.search(r"/year one-sided", r.message)   # annualized figure
    assert re.search(r"IBKR", r.message)
    assert abs(r.details["ann_turnover"]
               - r.details["mean_turnover"] * 252) < 1e-9
    assert "p95_turnover" in r.details


def test_clean_turnover_passes(clean_results):
    r = clean_results[C3]
    assert r.status is Status.PASS
    assert abs(r.details["mean_turnover"] - 0.144) < 0.02
    assert re.search(r"turnover", r.message)


def test_full_replacement_book_fails_turnover(cfg):
    res = by_id(run(_alternating_book(), cfg))
    r = res[C3]
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert re.search(r"replaced every period", r.message)
    assert r.details["mean_turnover"] >= cfg.turnover_daily_fail


def test_missing_positions_skips_turnover(clean_case, cfg):
    art = dataclasses.replace(clean_case.artifacts, positions=None,
                              declared_costs_bps=None).aligned()
    res = by_id(run(art, cfg))
    assert res[C3].status is Status.SKIP
    assert re.search(r"positions", res[C3].message)
    assert res[C4].status is Status.SKIP
    assert re.search(r"positions", res[C4].message)


# ---------------------------------------------------------------------------
# 4. costs.cost_sensitivity
# ---------------------------------------------------------------------------

def _alternating_book() -> BacktestArtifacts:
    """Whole book flips between two names every period: real gross edge
    (~2e-4/day on ~1e-3 vol => gross SR ~3) but 2.0 dollars traded/period,
    so any realistic one-way cost annihilates it."""
    rng = np.random.default_rng(42)
    n, k = 400, 4
    dates = pd.bdate_range("2019-01-01", periods=n)
    cols = [f"S{i}" for i in range(k)]
    rets = pd.DataFrame(rng.normal(2e-4, 1e-3, (n, k)),
                        index=dates, columns=cols)
    w = np.zeros((n, k))
    w[::2, 0] = 1.0
    w[1::2, 1] = 1.0
    pos = pd.DataFrame(w, index=dates, columns=cols)
    return BacktestArtifacts(signals=pos.copy(), asset_returns=rets,
                             positions=pos).aligned()


def test_cost_fragile_book_warns(cfg):
    r = by_id(run(_alternating_book(), cfg))[C4]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert re.search(r"cost-fragile", r.message)
    assert re.search(r"10bps", r.message)             # the flip point
    assert r.details["gross_sr"] > 0.5
    assert r.details["sr_by_bps"]["10"] <= 0
    assert "0" in r.details["sr_by_bps"]              # table includes 0bps


def test_clean_survives_cost_sweep(clean_results):
    r = clean_results[C4]
    assert r.status is Status.PASS
    table = r.details["sr_by_bps"]
    assert set(table) == {"0", "5", "10", "25"}
    assert table["25"] > 0                            # survives 25bps one-way
    assert table["0"] >= table["25"]                  # monotone-ish drag
    assert re.search(r"25bps", r.message)             # compact table in message
