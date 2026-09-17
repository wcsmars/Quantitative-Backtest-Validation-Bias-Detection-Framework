"""Small-sample IC floors, implied-cost ceilings, and nullable universe alignment.

Insufficient observations cannot support a suspicion verdict, implausible
transaction costs remain visible, and universe alignment avoids silent
pandas downcasting.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from qaudit._stats import gross_strategy_returns
from qaudit.checks import costs, lookahead
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (make_clean, make_lookahead, momentum_signal,
                              net_returns, positions_from_signals,
                              simulate_market)
from qaudit.types import Status

CFG = AuditConfig()


def _by(results):
    return {r.check: r for r in results}


def _sparse_signal_book(n_live: int, seed: int = 0, n_assets: int = 6,
                        n_periods: int = 300, embed: bool = True):
    """Signals NaN everywhere except n_live dates; on those dates the signal
    either equals the forward return exactly (embed=True - rank-perfect
    |IC| = 1) or is pure noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_periods)
    assets = [f"A{i}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (n_periods, n_assets)),
                        index=dates, columns=assets)
    sig = pd.DataFrame(np.nan, index=dates, columns=assets)
    live = dates[50:50 + n_live]
    fwd = rets.shift(-1)
    for d in live:
        sig.loc[d] = fwd.loc[d] if embed else rng.normal(size=n_assets)
    art = BacktestArtifacts(signals=sig, asset_returns=rets, signal_lag=1)
    art.validate()
    return art.aligned()


# ---------------------------------------------------------------------------
# lookahead small-sample floors: a verdict may not rest on < MIN_IC_DATES
# ---------------------------------------------------------------------------

class TestLookaheadSmallSampleFloors:
    def test_embedded_future_return_two_perfect_dates_skips(self):
        # mean |IC| = 1.0 over 2 dates is indistinguishable from luck; the
        # floor must SKIP rather than FAIL CRITICAL.
        art = _sparse_signal_book(n_live=2, embed=True)
        r = _by(lookahead.run(art, CFG))["lookahead.embedded_future_return"]
        assert r.status is Status.SKIP
        assert str(lookahead.MIN_IC_DATES) in r.message

    def test_embedded_future_return_floor_boundary_still_fails(self):
        # At the floor the same construction is a real detection.
        art = _sparse_signal_book(n_live=lookahead.MIN_IC_DATES, embed=True)
        r = _by(lookahead.run(art, CFG))["lookahead.embedded_future_return"]
        assert r.status is Status.FAIL

    def test_ic_decay_sub_floor_skips(self):
        art = _sparse_signal_book(n_live=10, embed=False)
        r = _by(lookahead.run(art, CFG))["lookahead.ic_decay_signature"]
        assert r.status is Status.SKIP

    def test_deferred_ic_spike_sub_floor_skips_not_passes(self):
        art = _sparse_signal_book(n_live=10, embed=False)
        r = _by(lookahead.run(art, CFG))["lookahead.deferred_ic_spike"]
        assert r.status is Status.SKIP

    def test_honest_dense_book_not_skipped(self):
        case = make_clean(seed=3)
        art = case.artifacts
        art.validate()
        by = _by(lookahead.run(art.aligned(), CFG))
        assert by["lookahead.embedded_future_return"].status is Status.PASS
        assert by["lookahead.ic_decay_signature"].status is Status.PASS

    def test_lookahead_case_still_detected(self):
        case = make_lookahead(seed=3)
        art = case.artifacts
        art.validate()
        by = _by(lookahead.run(art.aligned(), CFG))
        assert by["lookahead.embedded_future_return"].status is Status.FAIL


# ---------------------------------------------------------------------------
# costs: implied one-way cost above the plausibility ceiling is a mismatch
# ---------------------------------------------------------------------------

def _cost_book(drag_per_unit: float, seed: int = 4):
    market = simulate_market(seed=seed)
    rets = market["returns"]
    sig = momentum_signal(rets)
    pos = positions_from_signals(sig, lag=1)
    gross = gross_strategy_returns(pos, rets)
    traded = pos.diff().abs().sum(axis=1)
    traded.iloc[0] = pos.iloc[0].abs().sum()
    net = gross - traded * drag_per_unit
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            strategy_returns=net, signal_lag=1)
    art.validate()
    return art.aligned()


class TestCostsUpperCeiling:
    def test_absurd_implied_cost_warns(self):
        # 2500bps of drag per unit traded: whatever the dollars-traded
        # convention, implied cost lands far above the 500bps ceiling.
        art = _cost_book(drag_per_unit=0.25)
        r = _by(costs.run(art, CFG))["costs.missing_transaction_costs"]
        assert r.status is Status.WARN
        assert "ceiling" in r.message

    def test_honest_cost_book_unaffected(self):
        market = simulate_market(seed=4)
        rets = market["returns"]
        sig = momentum_signal(rets)
        pos = positions_from_signals(sig, lag=1)
        net = net_returns(pos, rets, costs_bps=10.0)
        art = BacktestArtifacts(signals=sig, asset_returns=rets,
                                positions=pos, strategy_returns=net,
                                signal_lag=1, declared_costs_bps=10.0)
        art.validate()
        r = _by(costs.run(art.aligned(), CFG))["costs.missing_transaction_costs"]
        assert r.status is Status.PASS

    def test_overcharge_note_path_survives_below_ceiling(self):
        # Implied ~ 3-8x a small declared cost but under the absolute
        # ceiling: the PASS+note behaviour applies.
        art = _cost_book(drag_per_unit=0.004)  # <= 40bps implied
        art.declared_costs_bps = 5.0
        r = _by(costs.run(art, CFG))["costs.missing_transaction_costs"]
        assert r.status is Status.PASS


# ---------------------------------------------------------------------------
# aligned() universe fill: no FutureWarning, dtype-independent output
# ---------------------------------------------------------------------------

class TestUniverseFillWarningFree:
    def _arts(self, uni):
        market = simulate_market(seed=5, death_frac=0.0)
        rets = market["returns"]
        return BacktestArtifacts(signals=momentum_signal(rets),
                                 asset_returns=rets, universe=uni,
                                 signal_lag=1)

    def test_object_dtype_universe_no_futurewarning(self):
        market = simulate_market(seed=5, death_frac=0.0)
        uni = market["universe"].copy().astype(object)
        uni.iloc[10, 0] = pd.NA                     # object NA cell
        uni = uni.drop(columns=[uni.columns[-1]])   # reindex all-NaN column
        art = self._arts(uni)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            aligned = art.aligned()
        assert aligned.universe.dtypes.eq(bool).all()
        assert not bool(aligned.universe.iloc[10, 0])
        assert not aligned.universe[market["universe"].columns[-1]].any()

    def test_bool_and_object_dtype_align_identically(self):
        market = simulate_market(seed=5, death_frac=0.0)
        uni_bool = market["universe"]
        uni_obj = uni_bool.astype(object)
        a_bool = self._arts(uni_bool).aligned().universe
        a_obj = self._arts(uni_obj).aligned().universe
        pd.testing.assert_frame_equal(a_bool, a_obj)
