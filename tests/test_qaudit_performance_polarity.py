"""Performance suspicion thresholds and regime concentration for both edge signs.

Suspicious Sharpe uses absolute magnitude; regime concentration follows the
dominant IC sign only when the mean is distinguishable from noise. Modest
losers and noise books provide false-positive controls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks.performance import _REGIME_MIN_ABS_MEAN_IC, run
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()


@pytest.fixture(scope="module")
def market():
    return simulate_market(seed=2)


@pytest.fixture(scope="module")
def honest(market):
    return momentum_signal(market["returns"])


def _arts(sig, rets, **kw):
    pos = positions_from_signals(sig, 1)
    base = dict(signals=sig, asset_returns=rets, positions=pos,
                strategy_returns=net_returns(pos, rets, 10.0),
                signal_lag=1, declared_costs_bps=10.0)
    base.update(kw)
    return BacktestArtifacts(**base).aligned()


def _by(results):
    return {r.check: r for r in results}


def _leak_signal(rets, honest, flip=False):
    """Classic embedded-future-return leak; optionally sign-flipped."""
    fwd = rets.shift(-1)
    z = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1), axis=0)
    sig = (0.25 * honest.fillna(0.0) + z).where(honest.notna())
    return -sig if flip else sig


# ---------------------------------------------------------------------------
# 1. suspicious_sharpe: negative-SR attacks
# ---------------------------------------------------------------------------

def test_sign_flipped_leak_extreme_negative_sr_fails(market, honest):
    """SR ~ -80 book (leak traded inverted) must FAIL; a negative sign is
    not exoneration."""
    rets = market["returns"]
    art = _arts(_leak_signal(rets, honest, flip=True), rets)
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert r.details["sharpe_annualized"] < -CFG.sharpe_fail
    # the negative branch must diagnose the sign, not just say "leak"
    assert "sign-flip" in r.message or "inverted" in r.message
    assert "polarity" in r.remediation or "sign" in r.remediation


def test_moderate_negative_sr_warns_between_thresholds(market, honest):
    """|SR| in [warn, fail): -3.5 must warn exactly like +3.5 would."""
    rets = market["returns"]
    rng = np.random.default_rng(0)
    strat = pd.Series(rng.normal(-0.0022, 0.010, len(rets)), index=rets.index)
    art = _arts(honest, rets, strategy_returns=strat)
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert -CFG.sharpe_fail < r.details["sharpe_annualized"] <= -CFG.sharpe_warn
    assert "sign-flipped" in r.message or "inverted" in r.message
    assert r.remediation


def test_modest_losing_book_still_passes(market, honest):
    """Honest guard: an ordinary loser (|SR| < warn) stays unsuspicious."""
    rets = market["returns"]
    rng = np.random.default_rng(1)
    strat = pd.Series(rng.normal(-0.0004, 0.008, len(rets)), index=rets.index)
    art = _arts(honest, rets, strategy_returns=strat)
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.PASS
    assert -CFG.sharpe_warn < r.details["sharpe_annualized"] < 0
    assert "nothing suspicious" in r.message


def test_positive_leak_detection_power_unchanged(market, honest):
    """Honest guard on the detector: the +SR leak must still FAIL with the
    near-arbitrage diagnosis."""
    rets = market["returns"]
    art = _arts(_leak_signal(rets, honest), rets)
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert r.details["sharpe_annualized"] > CFG.sharpe_fail
    assert "near-arbitrage" in r.message


def test_clean_positive_book_still_passes(market, honest):
    art = _arts(honest, market["returns"])
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.PASS
    assert 0 < r.details["sharpe_annualized"] < CFG.sharpe_warn


# ---------------------------------------------------------------------------
# 2. ic_regime_concentration: negative-polarity edge
# ---------------------------------------------------------------------------

def _regime_art(sign, seed=4, n=1040, m=30):
    """Noise signal except during 2020, where it embeds sign * forward return:
    the whole (signed) edge lives in one calendar year."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    cols = [f"R{i:02d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, m)), index=dates, columns=cols)
    fwd = rets.shift(-1)
    signals = pd.DataFrame(rng.normal(0.0, 0.02, (n, m)), index=dates, columns=cols)
    mask = dates.year == 2020
    signals.loc[mask] = signals.loc[mask] + sign * fwd.loc[mask].fillna(0.0)
    return BacktestArtifacts(signals=signals, asset_returns=rets).aligned()


def test_negative_regime_concentration_warns():
    """Attack: 99% of a negative cumulative IC from 2020 alone must WARN, not
    exit as 'not judged'."""
    r = _by(run(_regime_art(-1.0), CFG))["performance.ic_regime_concentration"]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert r.details["top_year"] == 2020
    assert r.details["share"] > CFG.ic_regime_share_warn
    assert r.details["direction"] == "negative"
    assert r.details["total_ic"] < 0
    assert "negative" in r.message and "2020" in r.message
    assert r.remediation


def test_positive_regime_detection_power_unchanged():
    r = _by(run(_regime_art(+1.0), CFG))["performance.ic_regime_concentration"]
    assert r.status is Status.WARN
    assert r.details["top_year"] == 2020
    assert r.details["direction"] == "positive"
    assert r.details["share"] > CFG.ic_regime_share_warn


@pytest.mark.parametrize("flip", [False, True], ids=["long-signal", "inverted-signal"])
def test_spread_edge_passes_in_both_polarities(market, honest, flip):
    """Honest guard: a real edge spread across years passes whichever way the
    signal's polarity points - the share computation must be sign-symmetric
    (measured share 0.43 for both on this seed)."""
    sig = -honest if flip else honest
    art = _arts(sig, market["returns"])
    r = _by(run(art, CFG))["performance.ic_regime_concentration"]
    assert r.status is Status.PASS
    assert r.details["share"] <= CFG.ic_regime_share_warn
    assert abs(r.details["mean_ic"]) > _REGIME_MIN_ABS_MEAN_IC  # judged, not gated


def test_pure_noise_gated_by_mean_ic_materiality():
    """Pure noise with mean IC near zero must remain unjudged. Chance sign cannot provide
    evidence of a concentrated edge."""
    rng = np.random.default_rng(0)
    n, m = 1040, 30
    dates = pd.bdate_range("2018-01-02", periods=n)
    cols = [f"R{i:02d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, m)), index=dates, columns=cols)
    sig = pd.DataFrame(rng.normal(0.0, 1.0, (n, m)), index=dates, columns=cols)
    art = BacktestArtifacts(signals=sig, asset_returns=rets).aligned()
    r = _by(run(art, CFG))["performance.ic_regime_concentration"]
    assert r.status is Status.PASS
    assert "not judged" in r.message
    assert abs(r.details["mean_ic"]) <= _REGIME_MIN_ABS_MEAN_IC
    assert "share" not in r.details


# Polarity controls for the remaining performance checks.

def test_suspicious_ic_already_two_sided_on_negative_leak(market, honest):
    """suspicious_ic uses |mean IC|: the inverted leak (mean IC ~ -0.96)
    must FAIL - pinned so the abs() never silently regresses."""
    rets = market["returns"]
    art = _arts(_leak_signal(rets, honest, flip=True), rets)
    r = _by(run(art, CFG))["performance.suspicious_ic"]
    assert r.status is Status.FAIL
    assert r.details["mean_ic"] < -0.9


def test_deflated_sharpe_flags_extreme_negative_sr(market, honest):
    """deflated_sharpe on a negative-SR book FAILs (DSR ~ 0) rather than
    passing - pinned as the family's second line of defense."""
    rets = market["returns"]
    art = _arts(_leak_signal(rets, honest, flip=True), rets)
    r = _by(run(art, AuditConfig(n_trials=20)))["performance.deflated_sharpe"]
    assert r.status is Status.FAIL
    assert r.details["dsr"] < 0.05


def test_ic_stability_judges_stable_negative_ic_book(market, honest):
    """ic_stability measures consistency against the full-sample sign: a
    stable negative-IC book (inverted honest momentum) passes with high
    consistency - polarity does not disable the check."""
    art = BacktestArtifacts(signals=-honest,
                            asset_returns=market["returns"]).aligned()
    r = _by(run(art, CFG))["performance.ic_stability"]
    assert r.status is Status.PASS
    assert r.details["mean_ic"] < -_REGIME_MIN_ABS_MEAN_IC
    assert r.details["consistency"] >= CFG.ic_sign_consistency_warn
