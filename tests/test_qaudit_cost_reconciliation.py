"""Plausible implied costs, traded-date coverage, and price-return consistency.

Sub-basis-point jitter cannot disguise free trading. Cost comparison requires
measurable trading overlap and respects the declared/implied tolerance.
Price checks distinguish pervasive shifts, scaling, and splits from sparse
dividend or isolated-tick differences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.api import MODULE_CHECK_IDS
from qaudit.checks import costs
from qaudit.checks.costs import (MIN_PLAUSIBLE_ONE_WAY_BPS,
                                 SPLIT_SIZED_DISCREPANCY)
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (make_clean, momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()
C1 = "costs.missing_transaction_costs"
C2 = "costs.no_cost_declaration"
C5 = "costs.price_return_consistency"


@pytest.fixture(scope="module")
def market():
    return simulate_market(seed=2)


@pytest.fixture(scope="module")
def honest(market):
    return momentum_signal(market["returns"])


def _by(results):
    return {r.check: r for r in results}


def _gross(pos, rets):
    return (pos * rets.fillna(0.0)).sum(axis=1)


def _arts(market, sig, pos, **kw):
    base = dict(signals=sig, asset_returns=market["returns"], positions=pos)
    base.update(kw)
    art = BacktestArtifacts(**base)
    art.validate()
    return art.aligned()


# Sub-basis-point jitter cannot disguise free trading.

def _jitter_net(pos, rets, seed=11, scale=1e-6):
    rng = np.random.default_rng(seed)
    return _gross(pos, rets) - np.abs(rng.normal(0.0, scale, len(rets)))


def test_jitter_free_trading_fails_critical(market, honest):
    pos = positions_from_signals(honest, 1)
    art = _arts(market, honest, pos,
                strategy_returns=_jitter_net(pos, market["returns"]))
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert "economically indistinguishable" in r.message
    assert r.details["implied_bps"] < MIN_PLAUSIBLE_ONE_WAY_BPS
    assert "not a cost model" in r.remediation


def test_jitter_with_declared_costs_fails_naming_declaration(market, honest):
    # declaring 10bps while charging jitter is the same evasion - the FAIL
    # must call out that the declared cost was never deducted
    pos = positions_from_signals(honest, 1)
    art = _arts(market, honest, pos,
                strategy_returns=_jitter_net(pos, market["returns"], seed=12),
                declared_costs_bps=10.0)
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert "never deducted" in r.message


def test_honest_costs_still_pass_with_implied_near_declared(market, honest):
    # honest guard: real 10bp charging keeps the PASS and the implied bps
    pos = positions_from_signals(honest, 1)
    art = _arts(market, honest, pos,
                strategy_returns=net_returns(pos, market["returns"], 10.0),
                declared_costs_bps=10.0)
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.PASS
    assert abs(r.details["implied_bps"] - 10.0) < 0.5


def test_honest_cheap_but_plausible_1bp_passes(market, honest):
    # 1bp one-way is cheap but above the floor: must not false-positive
    pos = positions_from_signals(honest, 1)
    art = _arts(market, honest, pos,
                strategy_returns=net_returns(pos, market["returns"], 1.0),
                declared_costs_bps=1.0)
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.PASS
    assert abs(r.details["implied_bps"] - 1.0) < 0.2


def test_paper_trading_declared_zero_passes_c1_warns_c2(market, honest):
    # the documented paper-trading path: declared_costs_bps=0 with net ==
    # gross is a consistent configuration - C1 PASSes, the realism warning
    # lives in costs.no_cost_declaration (HIGH)
    pos = positions_from_signals(honest, 1)
    art = _arts(market, honest, pos,
                strategy_returns=_gross(pos, market["returns"]),
                declared_costs_bps=0.0)
    res = _by(costs.run(art, CFG))
    assert res[C1].status is Status.PASS
    assert "paper-trading" in res[C1].message
    assert res[C2].status is Status.WARN
    assert res[C2].severity is Severity.HIGH


def test_sub_floor_declared_and_charged_consistently_warns_high(market, honest):
    # declared 0.3bp, charged 0.3bp: consistent but implausibly cheap -
    # graded WARN, not the not-charged FAIL and not a silent PASS
    pos = positions_from_signals(honest, 1)
    art = _arts(market, honest, pos,
                strategy_returns=net_returns(pos, market["returns"], 0.3),
                declared_costs_bps=0.3)
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert "plausibility floor" in r.message


def test_bit_identical_no_declaration_still_fails_critical(market, honest):
    # detection-power guard: the original no-costs construction keeps its
    # CRITICAL FAIL with the bit-identity wording
    pos = positions_from_signals(honest, 1)
    art = _arts(market, honest, pos,
                strategy_returns=_gross(pos, market["returns"]))
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert "bit-identical" in r.message


# ---------------------------------------------------------------------------
# Implied cost cannot be disarmed by reporting returns only on no-trade dates
# ---------------------------------------------------------------------------

def _weekly_book_reported_on_no_trade_dates(market, honest, wiggle=False):
    # Between weekly rebalances, weights must drift with returns to represent a non-
    # trading hold. Constant target weights would trade against drift every bar.
    rets = market["returns"]
    tgt = positions_from_signals(honest, 1)
    keep = np.arange(len(tgt)) % 5 == 0
    r = rets.fillna(0.0).to_numpy(dtype=float)
    t_arr = tgt.to_numpy(dtype=float)
    w = np.zeros_like(t_arr)
    w[0] = t_arr[0]
    for t in range(1, len(w)):
        if keep[t]:
            w[t] = t_arr[t]
        else:
            w[t] = (w[t - 1] * (1.0 + r[t - 1])
                    / (1.0 + float(w[t - 1] @ r[t - 1])))
    pos = pd.DataFrame(w, index=tgt.index, columns=tgt.columns)
    if wiggle:                      # knife-edge: one tiny trade in the overlap
        pos.iloc[7, 0] += 1e-6
    keep_s = pd.Series(keep, index=tgt.index)
    net = _jitter_net(pos, rets, seed=13)
    return pos, net.where(~keep_s)  # NaN on every rebalance date


def test_returns_reported_only_on_no_trade_dates_skips_if_validation_bypassed(
        market, honest):
    # validate() rejects this interior-gap series. Drive the check directly
    # to pin defence in depth: an unmeasurable selected-date overlap is an
    # unjudged SKIP, not a warning that satisfies a strict require gate.
    pos, net = _weekly_book_reported_on_no_trade_dates(market, honest)
    art = BacktestArtifacts(signals=honest, asset_returns=market["returns"],
                            positions=pos, strategy_returns=net).aligned()
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.SKIP
    assert "carries no trading" in r.message
    # drift-held rows trade nothing: exact up to float accumulation order
    # between the fixture's loop and the auditor's vectorized drift
    assert r.details["mean_dollars_traded"] < 1e-12
    assert r.details["mean_dollars_all_dates"] > 0.0


def test_knife_edge_single_tiny_trade_still_skips(market, honest):
    # one 1e-6 trade in the overlap makes mean_dollars > 0 strictly (implied
    # would explode instead of NaN) - the coverage gate must still catch it
    pos, net = _weekly_book_reported_on_no_trade_dates(market, honest,
                                                       wiggle=True)
    art = BacktestArtifacts(signals=honest, asset_returns=market["returns"],
                            positions=pos, strategy_returns=net).aligned()
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.SKIP
    assert "carries no trading" in r.message


def test_honest_weekly_rebalance_full_overlap_passes(market, honest):
    # honest guard: same weekly book with returns on every date passes with
    # the right implied cost - the coverage gate keys on overlap trading
    # share, not on rebalance frequency
    rets = market["returns"]
    pos = positions_from_signals(honest, 1)
    keep = pd.Series(np.arange(len(pos)) % 5 == 0, index=pos.index)
    pos = pos.where(keep, np.nan).ffill().fillna(0.0)
    art = _arts(market, honest, pos,
                strategy_returns=net_returns(pos, rets, 10.0),
                declared_costs_bps=10.0)
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.PASS
    assert abs(r.details["implied_bps"] - 10.0) < 1.0


def test_honest_late_start_partial_overlap_passes(market, honest):
    # honest guard: strategy_returns covering only the back 60% of the grid
    # (strategy started later) trades normally on the compared dates
    rets = market["returns"]
    pos = positions_from_signals(honest, 1)
    net = net_returns(pos, rets, 10.0)
    cut = int(0.4 * len(pos))
    art = BacktestArtifacts(signals=honest, asset_returns=rets,
                            positions=pos.iloc[cut:],
                            strategy_returns=net.iloc[cut:],
                            declared_costs_bps=10.0)
    art.validate()
    r = _by(costs.run(art.aligned(), CFG))[C1]
    assert r.status is Status.PASS


# Declared-versus-implied cost tolerance.

def test_undercharged_beyond_factor_still_warns_medium(market, honest):
    # detection-power guard: charging 2bp while declaring 30bp keeps the
    # under-charged WARN (implied is above the floor, so no escalation)
    pos = positions_from_signals(honest, 1)
    art = _arts(market, honest, pos,
                strategy_returns=net_returns(pos, market["returns"], 2.0),
                declared_costs_bps=30.0)
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert "under-charged" in r.message


def test_within_factor_window_passes_by_design(market, honest):
    # charged 25 vs declared 10 sits inside the documented 3x tolerance:
    # PASS (over-charging biases the backtest conservative)
    pos = positions_from_signals(honest, 1)
    art = _arts(market, honest, pos,
                strategy_returns=net_returns(pos, market["returns"], 25.0),
                declared_costs_bps=10.0)
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.PASS


def test_declared_never_deducted_escalates_past_undercharged(market, honest):
    # Charging 0.01 bp against a 10 bp declaration falls below the economic cost floor
    # and must FAIL CRITICAL.
    pos = positions_from_signals(honest, 1)
    art = _arts(market, honest, pos,
                strategy_returns=net_returns(pos, market["returns"], 0.01),
                declared_costs_bps=10.0)
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL


# ---------------------------------------------------------------------------
# costs.price_return_consistency
# ---------------------------------------------------------------------------

def _price_art(market, honest, prices, asset_returns=None):
    rets = market["returns"] if asset_returns is None else asset_returns
    art = BacktestArtifacts(signals=honest, asset_returns=rets, prices=prices)
    art.validate()
    return art.aligned()


def test_consistent_prices_pass(market, honest):
    r = _by(costs.run(_price_art(market, honest, market["prices"]), CFG))[C5]
    assert r.status is Status.PASS
    assert r.details["median_abs_diff"] < 1e-10
    assert "match" in r.message


def test_close_only_prices_vs_total_returns_stay_clean(market, honest):
    # dividends: quarterly 0.5% points where total return > price return -
    # sparse, dividend-sized, and legitimate for close-only price panels
    rets = market["returns"]
    div = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    div.loc[rets.index[::63]] = 0.005
    total = rets + div
    r = _by(costs.run(_price_art(market, honest, market["prices"],
                                 asset_returns=total), CFG))[C5]
    assert r.status is Status.PASS
    assert r.details["frac_discrepant"] > 0.005   # the points are there...
    assert r.details["median_abs_diff"] < 1e-10   # ...but cannot move the median


def test_shifted_returns_warn_high_pervasive(market, honest):
    # off-by-one date mismatch between prices and returns: every cell off by
    # the scale of daily vol
    r = _by(costs.run(_price_art(market, honest, market["prices"],
                                 asset_returns=market["returns"].shift(1)),
                      CFG))[C5]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert "systematically disagree" in r.message
    assert r.details["frac_discrepant"] > 0.25


def test_percent_vs_decimal_scaling_warns(market, honest):
    # Low volatility keeps the percent-scaled panel above the -100% intake floor so
    # this test reaches the price-consistency detector.
    rng = np.random.default_rng(7)
    base = market["returns"]
    rets = pd.DataFrame(rng.normal(0.0, 0.002, base.shape),
                        index=base.index, columns=base.columns)
    prices = 100.0 * (1.0 + rets).cumprod()
    r = _by(costs.run(_price_art(market, honest, prices,
                                 asset_returns=rets * 100.0),
                      CFG))[C5]
    assert r.status is Status.WARN
    assert r.details["median_abs_diff"] > 1e-3


def test_forgotten_split_adjustment_warns_despite_clean_median(market, honest):
    prices = market["prices"].copy()
    for j, t in [(0, 300), (1, 600)]:              # two forgotten 2:1 splits
        prices.iloc[t:, j] = prices.iloc[t:, j] / 2.0
    r = _by(costs.run(_price_art(market, honest, prices), CFG))[C5]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["median_abs_diff"] <= 1e-4    # median alone is blind here
    assert r.details["n_split_sized"] >= 2
    assert r.details["worst_abs_diff"] >= SPLIT_SIZED_DISCREPANCY
    assert "split" in r.message


def test_single_bad_tick_does_not_warn(market, honest):
    # exactly one split-sized cell (a bad last close, so no rebound cell) is
    # left to pass as a possible bad tick - SPLIT_SIZED_MIN_CELLS is 2
    prices = market["prices"].copy()
    prices.iloc[-1, 2] = prices.iloc[-1, 2] * 0.5
    r = _by(costs.run(_price_art(market, honest, prices), CFG))[C5]
    assert r.status is Status.PASS
    assert r.details["n_split_sized"] == 1


def test_no_prices_skips_with_actionable_message(market, honest):
    art = BacktestArtifacts(signals=honest, asset_returns=market["returns"])
    art.validate()
    r = _by(costs.run(art.aligned(), CFG))[C5]
    assert r.status is Status.SKIP
    assert "artifacts.prices" in r.message


def test_price_return_consistency_registered_in_module_check_ids():
    assert C5 in MODULE_CHECK_IDS["qaudit.checks.costs"]


def test_clean_case_full_costs_run_stays_green():
    # end-to-end FP guard: the clean synthetic (prices included) produces no
    # FAIL/WARN anywhere in the costs family
    res = costs.run(make_clean().artifacts.aligned(), CFG)
    bad = [r for r in res if r.status in (Status.FAIL, Status.WARN)]
    assert not bad, [(r.check, r.message) for r in bad]
