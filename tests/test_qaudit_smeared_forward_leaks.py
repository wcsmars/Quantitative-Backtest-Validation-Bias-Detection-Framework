"""Forward information spread across horizons and callback input sensitivity.

Innovation forward-IC mass detects diffuse signals without a single horizon
spike. Perturbation and truncation probes detect callbacks that replay stored
outputs. Honest momentum families and strongly diluted leaks exercise the
boundaries of those diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit import audit
from qaudit.checks import lookahead
from qaudit.config import AuditConfig
from qaudit.dynamic import probes_shift
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (make_backtest_func, make_clean, momentum_signal,
                              net_returns, positions_from_signals,
                              simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()

ATTACK_ALPHA = 0.2355        # net SR ~4.80 on seed 0 (just under sharpe_fail)
ATTACK_HORIZONS = tuple(range(1, 7))


def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def smeared_signal(asset_returns, alpha=ATTACK_ALPHA,
                   horizons=ATTACK_HORIZONS, weights=None):
    honest = momentum_signal(asset_returns)
    hs = list(horizons)
    if weights is None:
        weights = [1.0 / len(hs)] * len(hs)
    leak = sum(w * _zx(asset_returns.shift(-h)) for h, w in zip(hs, weights))
    return (honest + alpha * leak).where(honest.notna())


def _arts(market, sig):
    rets = market["returns"]
    pos = positions_from_signals(sig, 1)
    art = BacktestArtifacts(
        signals=sig, asset_returns=rets, positions=pos,
        strategy_returns=net_returns(pos, rets, 10.0),
        universe=market.get("universe"), signal_lag=1,
        declared_costs_bps=10.0, signal_input=rets)
    art.validate()
    return art.aligned()


@pytest.fixture(scope="module")
def market():
    return simulate_market(seed=0)


@pytest.fixture(scope="module")
def attack(market):
    return smeared_signal(market["returns"])


def _by(results):
    return {r.check: r for r in results}


# ---------------------------------------------------------------------------
# 1. the smeared-leak attack, variant (a): no callables
# ---------------------------------------------------------------------------

class TestAttackNoCallables:
    def test_attack_reaches_the_claimed_sharpe(self, market, attack):
        from qaudit._stats import annualized_sharpe
        pos = positions_from_signals(attack, 1)
        sr = annualized_sharpe(net_returns(pos, market["returns"], 10.0))
        assert 4.5 <= sr < 5.0          # just under sharpe_fail

    def test_smeared_forward_ic_catches_it(self, market, attack):
        art = _arts(market, attack)
        res = _by(lookahead.run(art, CFG))[lookahead.CHECK_SMEAR]
        assert res.status is Status.FAIL
        assert res.severity is Severity.CRITICAL
        d = res.details
        assert d["excess_mass"] > 0.5           # measured ~1.29, bar 0.15
        assert d["nw_tstat"] >= lookahead.SMEAR_TSTAT
        assert f'{d["ic_mass"]:.3f}' in res.message
        assert "horizon" in res.remediation

    def test_only_smeared_forward_ic_flags_the_attack(
            self, market, attack):
        # every other lookahead detector passes the SR-4.8 attack;
        # smeared_forward_ic is the sole detector.
        art = _arts(market, attack)
        res = _by(lookahead.run(art, CFG))
        for check, r in res.items():
            if check != lookahead.CHECK_SMEAR:
                assert r.status is Status.PASS, (check, r.message)

    def test_full_audit_has_the_fail(self, market, attack):
        art = _arts(market, attack)
        rep = audit(art, CFG)
        fails = {r.check for r in rep.results if r.status is Status.FAIL}
        assert lookahead.CHECK_SMEAR in fails


# ---------------------------------------------------------------------------
# 2. variant (b): honest-leaky signal_func -> truncation catches it
# ---------------------------------------------------------------------------

def test_honest_leaky_signal_func_dies_at_truncation(market, attack):
    art = _arts(market, attack)

    def honest_leaky(signal_input):
        return smeared_signal(signal_input)

    res = _by(probes_shift.run(art, CFG, signal_func=honest_leaky,
                               backtest_func=make_backtest_func()))
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.FAIL
    # the honest computation of the leak responds to its input: no cheat flag
    assert res[probes_shift.CHECK_SENSITIVITY].status is Status.PASS


# ---------------------------------------------------------------------------
# 3. variants (c)/(d): closure-cheating signal_funcs
# ---------------------------------------------------------------------------

class TestClosureCheat:
    def test_trimmed_cheat_flagged_insensitive(self, market, attack):
        art = _arts(market, attack)
        stored = attack

        def cheat(signal_input):
            return stored.reindex(index=signal_input.index)

        res = _by(probes_shift.run(art, CFG, signal_func=cheat,
                                   backtest_func=make_backtest_func()))
        # the laundering passes reproducibility and truncation...
        assert res[probes_shift.CHECK_REPRO].status is Status.PASS
        assert res[probes_shift.CHECK_TRUNCATION].status is Status.PASS
        # ...and the sensitivity probe is what catches it
        r = res[probes_shift.CHECK_SENSITIVITY]
        assert r.status is Status.WARN
        assert r.severity is Severity.HIGH
        assert "bit-identical" in r.message.lower() or "BIT-IDENTICAL" in r.message

    def test_untrimmed_cheat_is_a_fail(self, market, attack):
        art = _arts(market, attack)
        stored = attack

        def cheat(signal_input):
            return stored

        res = _by(probes_shift.run(art, CFG, signal_func=cheat,
                                   backtest_func=make_backtest_func()))
        r = res[probes_shift.CHECK_SENSITIVITY]
        assert r.status is Status.FAIL
        assert "after" in r.message.lower()

    def test_full_audit_with_cheat_still_fails_on_smear(self, market, attack):
        art = _arts(market, attack)
        stored = attack

        def cheat(signal_input):
            return stored.reindex(index=signal_input.index)

        rep = audit(art, CFG, signal_func=cheat,
                    backtest_func=make_backtest_func())
        fails = {r.check for r in rep.results if r.status is Status.FAIL}
        assert lookahead.CHECK_SMEAR in fails


# ---------------------------------------------------------------------------
# 4. the causal stamp is revoked when sensitivity flags the callable
# ---------------------------------------------------------------------------

MILD_BLEND_W = 0.2


def mild_blend_signal(asset_returns):
    mom = momentum_signal(asset_returns)
    mr = -_zx(asset_returns)
    return ((1 - MILD_BLEND_W) * mom + MILD_BLEND_W * mr).where(mom.notna())


def test_cheat_func_cannot_buy_the_causal_stamp():
    # A mild same-bar blend book lands in date_shift's ambiguous band. With
    # an honest signal_func the stamp downgrades it to WARN/MEDIUM; a
    # trimmed closure cheat must not buy that exoneration.
    market = simulate_market(seed=0)
    sig = mild_blend_signal(market["returns"])
    art = _arts(market, sig)
    stored = sig

    def cheat(signal_input):
        return stored.reindex(index=signal_input.index)

    res = _by(probes_shift.run(art, CFG, signal_func=cheat,
                               backtest_func=make_backtest_func()))
    assert res[probes_shift.CHECK_REPRO].status is Status.PASS
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.PASS
    assert res[probes_shift.CHECK_SENSITIVITY].status is Status.WARN
    r = res[probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH               # unverified path
    assert r.details["signal_verified_causal"] is False

    # control: the honest signal_func does earn the stamp
    res = _by(probes_shift.run(art, CFG, signal_func=mild_blend_signal,
                               backtest_func=make_backtest_func()))
    assert res[probes_shift.CHECK_SENSITIVITY].status is Status.PASS
    r = res[probes_shift.CHECK_DATE_SHIFT]
    assert r.details["signal_verified_causal"] is True
    assert r.severity is Severity.MEDIUM             # verified path


# ---------------------------------------------------------------------------
# 5. false-positive guards: honest signals stay clean
# ---------------------------------------------------------------------------

def _smear_result(sig, rets):
    art = _arts({"returns": rets, "universe": None}, sig)
    return _by(lookahead.run(art, CFG))[lookahead.CHECK_SMEAR]


class TestHonestFalsePositiveGuards:
    # Causal momentum variants provide controls near the excess-mass warning boundary.

    def test_momentum_default(self):
        m = simulate_market(seed=0)
        r = _smear_result(momentum_signal(m["returns"]), m["returns"])
        assert r.status is Status.PASS

    def test_strong_momentum_2x_loading(self):
        m = simulate_market(seed=22, mom_loading=0.30)
        r = _smear_result(momentum_signal(m["returns"]), m["returns"])
        assert r.status is Status.PASS

    def test_worst_calibrated_honest_case(self):
        # the ceiling case: 15d momentum on a 2x market, seed 12
        m = simulate_market(seed=12, mom_loading=0.30)
        r = _smear_result(momentum_signal(m["returns"], window=15),
                          m["returns"])
        assert r.status is Status.PASS
        # Use a literal 0.10 expectation so changing the implementation's threshold
        # cannot silently change the test's oracle.
        assert r.details["excess_mass"] < 0.10

    def test_mild_blend(self):
        m = simulate_market(seed=26)
        r = _smear_result(mild_blend_signal(m["returns"]), m["returns"])
        assert r.status is Status.PASS

    def test_fast_reversal_style(self):
        m = simulate_market(seed=14)
        rets = m["returns"]
        ma3 = rets.rolling(3, min_periods=3).mean()
        r = _smear_result(_zx(ma3), rets)
        assert r.status is Status.PASS

    def test_sensitivity_passes_honest_funcs(self, market):
        sig = momentum_signal(market["returns"])
        art = _arts(market, sig)
        res = _by(probes_shift.run(art, CFG, signal_func=momentum_signal,
                                   backtest_func=make_backtest_func()))
        assert res[probes_shift.CHECK_SENSITIVITY].status is Status.PASS

    def test_clean_case_end_to_end(self):
        case = make_clean()
        rep = audit(case.artifacts,
                    AuditConfig(seed=1, n_placebo=25, n_shuffle=25),
                    signal_func=case.signal_func,
                    backtest_func=case.backtest_func)
        assert not rep.failures, [str(r) for r in rep.failures]


# ---------------------------------------------------------------------------
# 6. the residual floor and the backstop
# ---------------------------------------------------------------------------

def _dilution_signal(rets, alpha, payoff=(1,), noise_ratio=16.0):
    """The hardest diluted case: h=1 payoff buried in legal
    past-return noise camouflage (dilutes the cleaned residual)."""
    hs = list(payoff)
    leak = sum(1.0 / len(hs) * _zx(rets.shift(-h)) for h in hs)
    for k in range(2, 9):
        leak = leak + (noise_ratio / 7.0) * _zx(rets.shift(k))
    honest = momentum_signal(rets)
    return (honest + alpha * leak).where(honest.notna())


class TestResidualFloor:
    def test_zero_flag_floor_point(self, market):
        # excess just under the WARN bar: SR ~2.6 with zero flags of any
        # kind from the statics - the measured zero-WARN floor.
        from qaudit._stats import annualized_sharpe
        rets = market["returns"]
        sig = _dilution_signal(rets, alpha=0.0102, noise_ratio=16.0)
        art = _arts(market, sig)
        res = _by(lookahead.run(art, CFG))[lookahead.CHECK_SMEAR]
        assert res.status is Status.PASS       # measured excess ~0.097
        pos = positions_from_signals(sig, 1)
        sr = annualized_sharpe(net_returns(pos, rets, 10.0))
        assert sr < 3.0                        # under suspicious_sharpe WARN

    def test_suspicious_sharpe_is_the_backstop(self, market, attack):
        # crank the same attack past the smear bar territory: SR above 5
        # dies at performance.suspicious_sharpe regardless.
        rets = market["returns"]
        sig = smeared_signal(rets, alpha=0.6)
        art = _arts(market, sig)
        rep = audit(art, CFG, include=["performance.suspicious_sharpe"])
        r = _by(rep.results)["performance.suspicious_sharpe"]
        assert r.status is Status.FAIL


# ---------------------------------------------------------------------------
# 7. registry / drift
# ---------------------------------------------------------------------------

def test_new_check_ids_are_registered():
    from qaudit.api import MODULE_CHECK_IDS
    assert lookahead.CHECK_SMEAR in MODULE_CHECK_IDS["qaudit.checks.lookahead"]
    assert (probes_shift.CHECK_SENSITIVITY
            in MODULE_CHECK_IDS["qaudit.dynamic.probes_shift"])
