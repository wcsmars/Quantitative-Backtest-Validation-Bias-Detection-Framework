"""IC sample size, breadth noise, and multi-period horizon scaling.

Sparse and overlapping observations need appropriate suspicion thresholds.
Honest stamped alpha and embedded-target controls exercise both sides of
the evidence boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks.lookahead import MIN_IC_DATES
from qaudit.checks.performance import run
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()


def _by(results):
    return {r.check: r for r in results}


def _ic_check(sig, rets, h=1, **kw):
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            label_horizon=h, **kw).aligned()
    return _by(run(art, CFG))["performance.suspicious_ic"]


# ---------------------------------------------------------------------------
# 1. suspicious_ic: small-sample SKIP floor + breadth/depth noise floor
# ---------------------------------------------------------------------------

def _sparse_stamped_book(seed, n_assets=8, n_days=504, stamp_every=21,
                         noise_only=False):
    """Honest monthly-rebalance layout: causal trailing-60d z-score signal
    stamped on ~month-end dates, NaN between (the vendor artifact), iid
    Gaussian returns (zero true predictability)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(4e-4, 0.012, (n_days, n_assets)),
                        index=dates, columns=cols)
    if noise_only:
        daily = pd.DataFrame(rng.normal(0, 1, (n_days, n_assets)),
                             index=dates, columns=cols)
    else:
        ma = rets.rolling(60).mean()
        daily = ma.sub(ma.mean(axis=1), axis=0).div(ma.std(axis=1), axis=0)
    sig = pd.DataFrame(np.nan, index=dates, columns=cols)
    stamps = np.arange(stamp_every - 1, n_days, stamp_every)
    sig.iloc[stamps] = daily.iloc[stamps]
    return sig, rets


def test_sparse_monthly_honest_book_skips_not_convicts():
    """FP guard: ~24 usable dates (< 30) must SKIP and say why rather than
    judge an honest monthly book."""
    sig, rets = _sparse_stamped_book(10000)
    r = _ic_check(sig, rets)
    assert r.status is Status.SKIP
    assert r.details["n_dates"] < MIN_IC_DATES
    assert r.details["min_ic_dates"] == MIN_IC_DATES
    assert str(MIN_IC_DATES) in r.message
    assert "usable IC dates" in r.message
    # the message must cover the sample_size contradiction: usable IC dates
    # are a different count from common calendar periods
    assert "sample_size" in r.message


def test_sparse_monthly_noise_books_all_skip():
    """Ten seeds of the pure-noise monthly layout: none may FAIL or WARN."""
    for seed in range(10010, 10020):
        sig, rets = _sparse_stamped_book(seed, noise_only=True)
        r = _ic_check(sig, rets)
        assert r.status is Status.SKIP, f"seed {seed}: {r.status} {r.message}"


def test_dense_daily_same_breadth_still_judged():
    """Control pinning the mechanism: the same 8-name breadth with a dense
    daily noise signal (~500 usable dates) is judged and passes - the
    driver is the usable-date count, not universe size."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-04", periods=504)
    cols = [f"A{i:02d}" for i in range(8)]
    rets = pd.DataFrame(rng.normal(4e-4, 0.012, (504, 8)),
                        index=dates, columns=cols)
    sig = pd.DataFrame(rng.normal(0, 1, (504, 8)), index=dates, columns=cols)
    r = _ic_check(sig, rets)
    assert r.status is Status.PASS
    assert r.details["n_dates"] >= MIN_IC_DATES


def test_noise_floor_lifts_bars_above_config_on_narrow_sparse_book():
    """Thirty-one usable dates across five names need a breadth/depth noise floor beyond
    the date-count floor. The lifted thresholds must be visible in result details."""
    rng = np.random.default_rng(700000)
    dates = pd.bdate_range("2022-01-03", periods=250)
    cols = [f"A{i}" for i in range(5)]
    rets = pd.DataFrame(rng.normal(4e-4, 0.012, (250, 5)),
                        index=dates, columns=cols)
    sig = pd.DataFrame(np.nan, index=dates, columns=cols)
    stamps = np.linspace(0, 248, 31).astype(int)
    sig.iloc[stamps] = rng.normal(0, 1, (31, 5))
    r = _ic_check(sig, rets)
    assert r.status is Status.PASS
    assert 0.09 < abs(r.details["mean_ic"]) < 0.15   # the unlifted config bars would flag this
    assert r.details["warn_bar"] > CFG.predictive_ic_warn * 2
    assert r.details["fail_bar"] > CFG.predictive_ic_fail * 2
    assert "noise floor" in r.message


def test_sparse_stamped_embedded_leak_still_fails():
    """Attack / detection power: a monthly-stamped embedded-target leak with
    enough usable dates (36 stamps, 12 names) must still FAIL - the lifted
    fail bar (~0.23) sits 4x below the embed's mean |IC| ~0.9."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-02", periods=756)
    cols = [f"A{i:02d}" for i in range(12)]
    rets = pd.DataFrame(rng.normal(4e-4, 0.012, (756, 12)),
                        index=dates, columns=cols)
    fwd = rets.shift(-1)
    sig = pd.DataFrame(np.nan, index=dates, columns=cols)
    stamps = np.linspace(0, 754, 36).astype(int)
    sig.iloc[stamps] = (fwd.iloc[stamps]
                        + rng.normal(0, 0.002, (36, 12))).values
    r = _ic_check(sig, rets)
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert r.details["mean_ic"] > 0.8
    assert r.details["n_dates"] >= MIN_IC_DATES
    assert r.details["fail_bar"] > CFG.predictive_ic_fail  # floor was active
    assert "embeds the target" in r.message


# ---------------------------------------------------------------------------
# 2. suspicious_ic: horizon-aware band
# ---------------------------------------------------------------------------

def test_h1_bars_and_behavior_unchanged():
    """Constraint pin: at h=1 on a dense honest book the multiplier is 1 and
    the effective bars equal the config bars (floors inactive)."""
    mkt = simulate_market(seed=0)
    r = _ic_check(momentum_signal(mkt["returns"]), mkt["returns"], h=1)
    assert r.status is Status.PASS
    assert r.details["horizon_multiplier"] == 1.0
    assert r.details["warn_bar"] == CFG.predictive_ic_warn
    assert r.details["fail_bar"] == CFG.predictive_ic_fail
    assert "0.01-0.05" in r.message


def test_honest_default_market_h10_passes_under_scaled_band():
    """FP guard: seed 6 default-loading momentum at h=10 (mean IC ~0.107,
    above the h=1 band) must PASS under the horizon-scaled band."""
    mkt = simulate_market(seed=6)
    r = _ic_check(momentum_signal(mkt["returns"]), mkt["returns"], h=10)
    assert r.status is Status.PASS
    assert abs(r.details["mean_ic"]) > CFG.predictive_ic_warn  # above the unscaled h=1 bar
    assert r.details["horizon_multiplier"] == pytest.approx(2.5)


@pytest.mark.parametrize("h", [5, 10, 21])
def test_honest_2x_loading_market_never_fails_at_horizon(h):
    """FP guard at the calibration-blessed 2x loading: may WARN (a genuinely
    strong alpha is suspicious-worthy) but must not FAIL - FAIL flips
    report.ok."""
    mkt = simulate_market(seed=0, mom_loading=0.30)
    r = _ic_check(momentum_signal(mkt["returns"]), mkt["returns"], h=h)
    assert r.status is not Status.FAIL
    assert abs(r.details["mean_ic"]) > 0.1  # above the unscaled bars


def test_embedded_h_period_target_leak_still_fails():
    """Attack / detection power: a signal embedding the 10-period compound
    target (mean IC ~1) must FAIL any sane scaled bar."""
    mkt = simulate_market(seed=0)
    rets = mkt["returns"]
    fwd10 = ((1 + rets).rolling(10).apply(np.prod, raw=True) - 1).shift(-10)
    rng = np.random.default_rng(0)
    sig = fwd10 + rng.normal(0, 0.005, rets.shape)
    r = _ic_check(sig, rets, h=10)
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert r.details["mean_ic"] > 0.8
    assert "embeds the target" in r.message


def test_negative_h_period_leak_still_two_sided():
    """An inverted embedded target at horizons above one must fail on absolute mean IC."""
    mkt = simulate_market(seed=0)
    rets = mkt["returns"]
    fwd10 = ((1 + rets).rolling(10).apply(np.prod, raw=True) - 1).shift(-10)
    rng = np.random.default_rng(0)
    sig = -(fwd10 + rng.normal(0, 0.005, rets.shape))
    r = _ic_check(sig, rets, h=10)
    assert r.status is Status.FAIL
    assert r.details["mean_ic"] < -0.8


# ---------------------------------------------------------------------------
# 3. suspicious_sharpe: cost-drag discriminator on the negative branches
# ---------------------------------------------------------------------------

def _churny_costed_book(seed, costs_bps=25.0, n_assets=100, n_days=756):
    """Honest no-edge fast-reversal candidate: 100 diversified names, 1-bar
    lag, ~46%/period one-sided turnover, costs charged and declared. Net SR
    ~ -16 purely from deterministic drag; gross ~ 0."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    cols = [f"A{i:03d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(2e-4, 0.02, (n_days, n_assets)),
                        index=dates, columns=cols)
    sig = -momentum_signal(rets, window=2)
    pos = positions_from_signals(sig, 1)
    return BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                             strategy_returns=net_returns(pos, rets, costs_bps),
                             signal_lag=1, declared_costs_bps=costs_bps
                             ).aligned()


def test_honest_costed_churny_book_passes_as_cost_drag():
    """FP guard: net SR ~ -16 with gross ~ 0 and drag == dollars x 25bp is
    cost drag, not a sign-flipped leak."""
    art = _churny_costed_book(500)
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.PASS
    assert r.details["sharpe_annualized"] < -CFG.sharpe_fail
    assert abs(r.details["gross_sharpe_annualized"]) < CFG.sharpe_warn
    assert r.details["implied_one_way_bps"] == pytest.approx(25.0, abs=0.5)
    assert "cost drag" in r.message
    assert "sign-flip" in r.message  # names what it is not, for the reader
    assert "costs.cost_sensitivity" in r.message


def test_honest_costed_churny_book_warn_range_also_explained():
    """Same book at 10bp (net SR ~ -6.7) must also PASS."""
    art = _churny_costed_book(501, costs_bps=10.0)
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.PASS
    assert r.details["sharpe_annualized"] < -CFG.sharpe_fail
    assert r.details["implied_one_way_bps"] == pytest.approx(10.0, abs=0.5)


def _sign_flip_leak_book(seed):
    mkt = simulate_market(seed=seed)
    rets = mkt["returns"]
    honest = momentum_signal(rets)
    fwd = rets.shift(-1)
    z = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1), axis=0)
    sig = -((0.25 * honest.fillna(0.0) + z).where(honest.notna()))
    pos = positions_from_signals(sig, 1)
    return BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                             strategy_returns=net_returns(pos, rets, 10.0),
                             signal_lag=1, declared_costs_bps=10.0).aligned()


def test_sign_flipped_leak_with_costs_still_fails_with_sign_diagnosis():
    """Attack / detection power: the sign-flipped leak charges the
    same 10bp costs (its drag reconciles!) but gross SR ~ -80 is itself far
    beyond the warn bar - the gross gate keeps the FAIL and its diagnosis."""
    r = _by(run(_sign_flip_leak_book(2), CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert r.details["sharpe_annualized"] < -CFG.sharpe_fail
    assert r.details["gross_sharpe_annualized"] < -CFG.sharpe_fail
    assert "sign-flip" in r.message or "inverted" in r.message


def test_net_unrelated_to_positions_does_not_reconcile():
    """Honest guard on the reconciliation gate: an extreme-negative net
    series unrelated to the positions (gross ~ honest momentum) must stay
    FAIL - (gross - net) does not track dollars traded x declared costs."""
    mkt = simulate_market(seed=2)
    rets = mkt["returns"]
    honest = momentum_signal(rets)
    pos = positions_from_signals(honest, 1)
    rng = np.random.default_rng(7)
    strat = pd.Series(rng.normal(-0.0040, 0.010, len(rets)), index=rets.index)
    art = BacktestArtifacts(signals=honest, asset_returns=rets, positions=pos,
                            strategy_returns=strat, signal_lag=1,
                            declared_costs_bps=10.0).aligned()
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert abs(r.details["gross_sharpe_annualized"]) < CFG.sharpe_warn
    assert "sign-flip" in r.message or "inverted" in r.message
    assert "does not explain" in r.message  # gross was measured and cited


def test_net_only_negative_fail_names_cost_drag_alternative():
    """Without positions the discriminator cannot run: the FAIL stands but
    must name cost drag as the unmeasured alternative and say what to pass
    (it must not assert 'not underperformance' unmeasured)."""
    rng = np.random.default_rng(101)
    dates = pd.bdate_range("2020-01-02", periods=1000)
    cols = [f"A{i}" for i in range(8)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (1000, 8)), index=dates,
                        columns=cols)
    sig = pd.DataFrame(rng.normal(0, 1, (1000, 8)), index=dates, columns=cols)
    rng2 = np.random.default_rng(101)
    strat = pd.Series(-0.0036 + rng2.normal(0, 0.01, 1000), index=dates)
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            strategy_returns=strat).aligned()
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert "cost drag" in r.message
    assert "positions" in r.message
    assert "sign-flipped" in r.message or "inverted" in r.message


# ---------------------------------------------------------------------------
# 4. sharpe_fail boundary pins: the 5.0 bar constrained from both sides
# ---------------------------------------------------------------------------

def _flat_book(mu, seed, n=1000):
    # separate generators: the strat stream is seeded on its own so the SR
    # values quoted in the docstrings below are reproducible; the panels only
    # need to exist
    prng = np.random.default_rng(seed + 7777)
    srng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    cols = [f"A{i}" for i in range(8)]
    rets = pd.DataFrame(prng.normal(0, 0.01, (n, 8)), index=dates, columns=cols)
    sig = pd.DataFrame(prng.normal(0, 1, (n, 8)), index=dates, columns=cols)
    strat = pd.Series(mu + srng.normal(0, 0.01, n), index=dates)
    return BacktestArtifacts(signals=sig, asset_returns=rets,
                             strategy_returns=strat).aligned()


def test_fail_boundary_just_above_bar_positive():
    """SR +5.02 (measured) sits just past sharpe_fail=5.0 and must FAIL -
    pins the boundary within ~0.5%."""
    r = _by(run(_flat_book(0.0030, 11), CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert CFG.sharpe_fail < r.details["sharpe_annualized"] < 5.7
    assert "near-arbitrage" in r.message


def test_warn_boundary_just_below_bar_positive():
    """SR +4.86 (measured) sits just under the 5.0 bar: WARN, not FAIL -
    pins against downward drift of sharpe_fail too."""
    r = _by(run(_flat_book(0.0029, 11), CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.WARN
    assert CFG.sharpe_warn < r.details["sharpe_annualized"] < CFG.sharpe_fail
    assert 4.5 < r.details["sharpe_annualized"] < 5.0  # self-validating band


def test_fail_boundary_negative_branch():
    """SR -6.29 (measured, net-only book): negative FAIL branch trips just
    past the bar in the other polarity."""
    r = _by(run(_flat_book(-0.0036, 101), CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert -7.0 < r.details["sharpe_annualized"] < -CFG.sharpe_fail
    assert "sign-flipped" in r.message or "inverted" in r.message


def test_warn_boundary_negative_branch():
    """SR -4.76 (measured): negative WARN just under the FAIL bar."""
    r = _by(run(_flat_book(-0.0030, 5), CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.WARN
    assert -CFG.sharpe_fail < r.details["sharpe_annualized"] <= -CFG.sharpe_warn
    assert -5.0 < r.details["sharpe_annualized"] < -4.5  # self-validating band
