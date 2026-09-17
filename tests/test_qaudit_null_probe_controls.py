"""Placebo and shuffle probe controls, degenerate samples, and signal consumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.config import AuditConfig
from qaudit.dynamic.probes_null import (
    _LEAK_MIN_ACTUAL_SR, _SR_DEGENERATE_CAP, _check_placebo_bias,
    _check_shuffled_labels, _gate_val, _hedged_resid_sr, _split_degenerate,
    run,
)
from qaudit.inputs import BacktestArtifacts
from qaudit.types import Severity, Status

CFG = AuditConfig(n_placebo=16, n_shuffle=20, seed=1)


# ---------------------------------------------------------------------------
# Synthetic constructions (small: 420 x 12 keeps each run() < 1s)
# ---------------------------------------------------------------------------

def drift_market(seed: int, n_assets: int = 12, n_periods: int = 420,
                 mu: float = 8e-4) -> pd.DataFrame:
    """Ordinary bull sample: common drifting factor, ~20%/yr, EW SR > 1."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_periods)
    beta = rng.uniform(0.8, 1.2, n_assets)
    fac = mu + 0.010 * rng.standard_normal(n_periods)
    idio = 0.015 * rng.standard_normal((n_periods, n_assets))
    return pd.DataFrame(fac[:, None] * beta[None, :] + idio, index=dates,
                        columns=[f"A{i:02d}" for i in range(n_assets)])


def persistent_market(seed: int, alpha_sd: float = 1.5e-3,
                      n_assets: int = 12,
                      n_periods: int = 420) -> pd.DataFrame:
    """Stable per-asset alpha spread (low-vol/quality-like), no drift."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_periods)
    alpha = rng.normal(0.0, alpha_sd, n_assets)
    idio = 0.012 * rng.standard_normal((n_periods, n_assets))
    return pd.DataFrame(alpha[None, :] + idio, index=dates,
                        columns=[f"A{i:02d}" for i in range(n_assets)])


def zscore_signal(rets: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ma = rets.rolling(window, min_periods=window).mean()
    return ma.sub(ma.mean(axis=1), axis=0).div(
        ma.std(axis=1).replace(0.0, np.nan), axis=0)


def neutral_pos(signals: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    r = signals.shift(lag).rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    g = w.abs().sum(axis=1)
    return w.div(g.replace(0.0, np.nan), axis=0).fillna(0.0)


def longonly_pos(signals: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    r = signals.shift(lag).rank(axis=1)
    return r.div(r.sum(axis=1), axis=0).fillna(0.0)


def net_rets(pos: pd.DataFrame, rets: pd.DataFrame,
             bps: float = 10.0) -> pd.Series:
    gross = (pos * rets.fillna(0.0)).sum(axis=1)
    traded = pos.diff().abs().sum(axis=1)
    traded.iloc[0] = pos.iloc[0].abs().sum()
    return gross - traded * (bps * 1e-4)


def probe(sig, rets, bt, cfg=CFG):
    ok = sig.notna().any(axis=1)
    art = BacktestArtifacts(signals=sig.loc[ok], asset_returns=rets.loc[ok],
                            strategy_returns=None)
    art.validate()
    return {r.check: r for r in run(art.aligned(), cfg, backtest_func=bt)}


def bt_longonly(signals, asset_returns):
    return net_rets(longonly_pos(signals), asset_returns)


def bt_neutral(signals, asset_returns):
    return net_rets(neutral_pos(signals), asset_returns)


# ---------------------------------------------------------------------------
# 1. long-only / net-exposed honest book: no FAIL from either null check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 2, 3])
def test_honest_longonly_bull_sample_not_convicted(seed):
    # mu=1.2e-3 (~30%/yr) so the counterfactual raw gate provably fires
    # at this reduced panel size (12 assets x 420 days)
    rets = drift_market(seed, mu=1.2e-3)
    res = probe(zscore_signal(rets), rets, bt_longonly)
    b, s = res["dynamic.placebo_pipeline_bias"], res["dynamic.shuffled_labels"]
    # attack bites: a raw-SR gate fires on this seed - fully-invested
    # placebo books earn the equity premium
    assert b.details["mean_null_raw"] >= CFG.placebo_null_sharpe_fail
    # exposure-aware verdicts: beta carry is not an engine bug
    assert b.status is not Status.FAIL, b.message
    assert s.status is not Status.FAIL, s.message
    # hedged placebo mean is cost drag, not the equity premium
    assert b.details["mean_null"] < CFG.placebo_null_sharpe_fail


def test_honest_longonly_shuffle_counterfactual_pinned():
    # seed chosen so the raw-SR form of the shuffled_labels gate (raw mean
    # >= 0.5 * raw actual with raw actual >= 1.0) fires
    rets = drift_market(3, mu=1.2e-3)
    res = probe(zscore_signal(rets), rets, bt_longonly)
    s = res["dynamic.shuffled_labels"]
    assert s.details["actual_sr_raw"] >= 1.0
    assert (s.details["mean_null_raw"]
            >= CFG.shuffle_leak_ratio * s.details["actual_sr_raw"])
    assert s.status is not Status.FAIL


def test_longonly_phantom_credit_engine_still_fails_bias():
    # paired attack: a 5bp/day credit on the positive leg through the same
    # long-only pipeline must still be detected - hedging removes beta,
    # never a fixed credit
    rets = drift_market(0)

    def bt_credit(signals, asset_returns):
        pos = longonly_pos(signals)
        return net_rets(pos, asset_returns) + \
            pos.clip(lower=0).sum(axis=1) * 5e-4

    res = probe(zscore_signal(rets), rets, bt_credit)
    b = res["dynamic.placebo_pipeline_bias"]
    assert b.status is Status.FAIL
    assert b.severity is Severity.CRITICAL
    assert b.remediation


# ---------------------------------------------------------------------------
# 2. static tilt vs accounting leak (cross-section relabeling null)
# ---------------------------------------------------------------------------

def _static_tilt_result(seed=201):
    rets = persistent_market(seed)
    return probe(zscore_signal(rets, window=60), rets, bt_neutral)


def test_honest_static_tilt_warns_tilt_not_leak():
    res = _static_tilt_result()
    s = res["dynamic.shuffled_labels"]
    # attack bites: survives the date shuffle (date-shuffle leak gate satisfied)
    assert s.details["actual_sr"] >= _LEAK_MIN_ACTUAL_SR
    assert (s.details["mean_null"]
            >= CFG.shuffle_leak_ratio * s.details["actual_sr"])
    # ...but dies under cross-section relabeling -> tilt WARN, not leak FAIL
    assert (s.details["mean_null_xsec"]
            < CFG.shuffle_leak_ratio * s.details["actual_sr"])
    assert s.status is Status.WARN
    assert s.severity is Severity.MEDIUM
    assert "tilt" in s.message
    assert "leak" not in s.message.split("rules out")[0]  # no leak claim
    # remediation points at validation discipline, not plumbing hunts
    assert "deflated" in s.remediation.lower() or "out-of-sample" in s.remediation.lower()
    # the same book must not trip the placebo bias check either
    assert res["dynamic.placebo_pipeline_bias"].status is Status.PASS


def test_per_asset_credit_leak_still_fails_both_nulls():
    # paired attack: a real per-asset fixed credit (no alpha in the data)
    # survives date shuffling and cross-section relabeling -> leak FAIL
    rets = persistent_market(300, alpha_sd=0.0)

    def bt_leak(signals, asset_returns):
        pos = neutral_pos(signals)
        return net_rets(pos, asset_returns) + \
            pos.clip(lower=0).sum(axis=1) * 25e-4

    res = probe(zscore_signal(rets, window=60), rets, bt_leak)
    s = res["dynamic.shuffled_labels"]
    assert s.status is Status.FAIL
    assert s.severity is Severity.CRITICAL
    assert "relabeling" in s.message         # both nulls named
    assert (s.details["mean_null_xsec"]
            >= CFG.shuffle_leak_ratio * s.details["actual_sr"])
    assert s.remediation


def test_leak_gate_boundary_below_min_actual_sr_never_fails():
    # boundary pin for _LEAK_MIN_ACTUAL_SR = 1.0: retained nulls with a
    # sub-1.0 actual must not convict (noise-on-noise ratios)
    cfg = AuditConfig()
    nulls = np.full(20, 0.9)
    r = _check_shuffled_labels(cfg, nulls, np.full(20, 0.9), True,
                               0.999, 0.999, 0.9)
    assert r.status is not Status.FAIL
    r2 = _check_shuffled_labels(cfg, nulls, np.full(20, 0.9), True,
                                1.0, 1.0, 0.9)
    assert r2.status is Status.FAIL   # at the boundary the gate is live


def test_untrusted_xsec_null_downgrades_to_warn_not_fail():
    # date-shuffle survival with an unmeasurable cross-section null cannot
    # distinguish leak from tilt -> WARN(HIGH) saying exactly that
    cfg = AuditConfig()
    r = _check_shuffled_labels(cfg, np.full(20, 2.0), np.array([]), False,
                               2.0, 2.0, 2.0)
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert "cannot distinguish" in r.message


# ---------------------------------------------------------------------------
# 3. degenerate placebo nulls: fail-closed, both float regimes
# ---------------------------------------------------------------------------

def _threshold_credit_engine(credit):
    """Entry threshold on an unstandardized (price-scale) signal + daily
    cash credit on idle capital: unit-variance placebos never trade, so
    every placebo run is the constant riskless credit."""
    def bt(signals, asset_returns):
        active = signals.shift(1).abs() > 25.0
        raw = signals.shift(1).where(active, 0.0)
        gross = raw.abs().sum(axis=1)
        pos = raw.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)
        idle = 1.0 - pos.abs().sum(axis=1).clip(0.0, 1.0)
        return net_rets(pos, asset_returns) + idle * credit
    return bt


def _price_scale_signal(rets, scale=60.0):
    ma = rets.rolling(10, min_periods=10).sum()
    return ma * scale / ma.stack().std()


@pytest.mark.parametrize("credit", [float(2.0 ** -12), 2e-4],
                         ids=["dyadic", "non-dyadic"])
def test_constant_credit_engine_fails_in_both_float_regimes(credit):
    rets = drift_market(0)
    sig = _price_scale_signal(rets)
    res = probe(sig, rets, _threshold_credit_engine(credit))
    b = res["dynamic.placebo_pipeline_bias"]
    assert b.status is Status.FAIL, (b.status, b.message)
    assert b.severity is Severity.CRITICAL
    assert b.details["n_degenerate_pos"] == CFG.n_placebo
    # message reports the degenerate count, never nan/inf t-stat claims
    assert f"{CFG.n_placebo}/{CFG.n_placebo}" in b.message
    assert "nan" not in b.message
    # percentile redirects at the engine instead of blaming the strategy
    p = res["dynamic.placebo_percentile"]
    assert p.status is Status.SKIP
    assert "placebo_pipeline_bias" in p.message
    assert "inf" not in p.message.replace("+inf or", "")  # no inf quantiles


def test_all_inf_null_fails_not_passes():
    cfg = AuditConfig()
    arr = np.full(60, np.inf)
    r = _check_placebo_bias(cfg, arr, arr, 0.9)
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_degenerate_pos"] == 60


def test_single_inf_cannot_disarm_a_biased_finite_null():
    # 49 blatantly biased finite nulls + one +inf: a NaN t-stat must not
    # flip this to PASS; the inf is itself conviction evidence
    cfg = AuditConfig()
    rng = np.random.default_rng(0)
    finite = rng.normal(3.0, 0.3, 49)
    arr = np.append(finite, np.inf)
    r = _check_placebo_bias(cfg, arr, arr, 0.9)
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    # finite-subset context is still reported with real numbers
    assert r.details["n_finite"] == 49
    assert r.details["mean_null"] == pytest.approx(float(finite.mean()))


def test_biased_finite_null_still_fails_without_any_inf():
    # honest-power guard: the ordinary finite gate convicts without any inf
    cfg = AuditConfig()
    rng = np.random.default_rng(1)
    arr = rng.normal(3.0, 0.3, 50)
    r = _check_placebo_bias(cfg, arr, arr, 0.9)
    assert r.status is Status.FAIL
    assert r.details["t"] >= 3.0


def test_all_neg_inf_null_passes_with_finite_arithmetic():
    cfg = AuditConfig()
    arr = np.full(30, -np.inf)
    r = _check_placebo_bias(cfg, arr, arr, 0.5)
    assert r.status is Status.PASS
    assert "nan" not in r.message
    assert r.details["n_degenerate_neg"] == 30


def test_neg_inf_entries_do_not_poison_finite_stats():
    cfg = AuditConfig()
    rng = np.random.default_rng(2)
    finite = rng.normal(-0.4, 0.5, 40)
    arr = np.append(finite, [-np.inf, -np.inf])
    r = _check_placebo_bias(cfg, arr, arr, 0.5)
    assert r.status is Status.PASS
    assert np.isfinite(r.details["mean_null"])
    assert r.details["mean_null"] == pytest.approx(float(finite.mean()))
    assert r.details["n_degenerate_neg"] == 2


def test_honest_clean_null_unaffected_by_degeneracy_machinery():
    # honest-guard: an ordinary slightly-negative null stays a clean PASS
    cfg = AuditConfig()
    rng = np.random.default_rng(3)
    arr = rng.normal(-0.3, 0.6, 40)
    r = _check_placebo_bias(cfg, arr, arr, 0.8)
    assert r.status is Status.PASS
    assert r.details["n_degenerate_pos"] == 0
    assert "cost drag" in r.message


def test_degenerate_cap_boundary():
    core, n_pos, n_neg = _split_degenerate(
        np.array([_SR_DEGENERATE_CAP, _SR_DEGENERATE_CAP + 1.0,
                  -_SR_DEGENERATE_CAP, -_SR_DEGENERATE_CAP - 1.0, np.inf,
                  -np.inf, 2.0]))
    assert n_pos == 2            # cap+1 and +inf; the cap itself is core
    assert n_neg == 2
    assert core.tolist() == [_SR_DEGENERATE_CAP, -_SR_DEGENERATE_CAP, 2.0]


def test_gate_val_mixed_inf_is_retention_not_nan():
    assert _gate_val(np.array([np.inf, -np.inf])) == np.inf
    assert _gate_val(np.array([-np.inf, -np.inf])) == -np.inf
    assert np.isnan(_gate_val(np.array([])))
    assert _gate_val(np.array([1.0, 2.0])) == pytest.approx(1.5)


def test_shuffle_mixed_inf_null_cannot_fail_open():
    # Mixed infinite null outcomes cannot fall through a NaN comparison; a constant-
    # positive replicate must retain the edge.
    cfg = AuditConfig()
    date_ok = np.array([np.inf, -np.inf] + [2.0] * 18)
    xsec_ok = np.array([np.inf] * 20)
    r = _check_shuffled_labels(cfg, date_ok, xsec_ok, True, 2.0, 2.0, 2.0)
    assert r.status is Status.FAIL


# ---------------------------------------------------------------------------
# 3b. hedge helper: dust vs credit vs ordinary residuals
# ---------------------------------------------------------------------------

def _mkt_series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(8e-4 + 0.01 * rng.standard_normal(n), index=idx)


def test_hedged_sr_exact_linear_pnl_is_degenerate_nan():
    mkt = _mkt_series()
    pnl = 2.0 * mkt
    raw = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252))
    assert np.isnan(_hedged_resid_sr(pnl, mkt, raw, 252))


def test_hedged_sr_linear_plus_credit_is_pos_inf():
    mkt = _mkt_series()
    pnl = 2.0 * mkt + 3e-4
    raw = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252))
    assert _hedged_resid_sr(pnl, mkt, raw, 252) == np.inf


def test_hedged_sr_strips_beta_keeps_alpha():
    rng = np.random.default_rng(5)   # independent of the mkt draw (seed 4)
    mkt = _mkt_series(seed=4)
    noise = pd.Series(2e-3 * rng.standard_normal(len(mkt)), index=mkt.index)
    pnl = 0.8 * mkt + noise + 2e-4
    raw = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252))
    h = _hedged_resid_sr(pnl, mkt, raw, 252)
    resid = noise + 2e-4
    expect = float(resid.mean() / resid.std(ddof=1) * np.sqrt(252))
    assert h == pytest.approx(expect, abs=0.15)
    assert abs(h - raw) > 0.5      # the beta carry was actually removed


def test_hedged_sr_index_mismatch_falls_back_to_raw():
    mkt = _mkt_series()
    pnl = pd.Series(np.linspace(1e-4, 5e-4, 50))   # integer index, disjoint
    raw = 1.23
    assert _hedged_resid_sr(pnl, mkt, raw, 252) == raw


# ---------------------------------------------------------------------------
# 4. defensive copies: mutating callables corrupt nothing
# ---------------------------------------------------------------------------

def _book_from(signals, asset_returns):
    r = signals.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    g = w.abs().sum(axis=1)
    pos = w.div(g.replace(0.0, np.nan), axis=0).fillna(0.0)
    return (pos * asset_returns.fillna(0.0)).sum(axis=1)


def bt_inplace_lag(signals, asset_returns):
    # in-place, non-idempotent: a second application would double-lag
    signals.iloc[:] = signals.shift(1)
    return _book_from(signals, asset_returns)


def bt_pure_lag(signals, asset_returns):
    return _book_from(signals.shift(1), asset_returns)


@pytest.fixture(scope="module")
def small_aligned():
    rets = drift_market(7, n_assets=8, n_periods=260)
    sig = zscore_signal(rets)
    ok = sig.notna().any(axis=1)
    art = BacktestArtifacts(signals=sig.loc[ok], asset_returns=rets.loc[ok],
                            strategy_returns=None)
    art.validate()
    return art.aligned()


def test_mutating_backtest_func_leaves_artifacts_bit_identical(small_aligned):
    sig_before = small_aligned.signals.copy(deep=True)
    ret_before = small_aligned.asset_returns.copy(deep=True)
    run(small_aligned, CFG, backtest_func=bt_inplace_lag)
    pd.testing.assert_frame_equal(small_aligned.signals, sig_before,
                                  check_exact=True)
    pd.testing.assert_frame_equal(small_aligned.asset_returns, ret_before,
                                  check_exact=True)


def test_mutating_backtest_func_equals_pure_equivalent(small_aligned):
    res_mut = {r.check: r for r in
               run(small_aligned, CFG, backtest_func=bt_inplace_lag)}
    res_pure = {r.check: r for r in
                run(small_aligned, CFG, backtest_func=bt_pure_lag)}
    for cid in ("dynamic.shuffled_labels", "dynamic.placebo_pipeline_bias",
                "dynamic.placebo_percentile"):
        assert res_mut[cid].status is res_pure[cid].status, cid
        for k, v in res_pure[cid].details.items():
            got = res_mut[cid].details[k]
            if isinstance(v, float) and np.isnan(v):
                assert np.isnan(got), (cid, k)
            else:
                assert got == v, (cid, k, got, v)


def test_idempotent_inplace_mutation_still_leaves_artifacts_intact(small_aligned):
    # fillna/clip-style mutations trigger no verdict change, but the
    # artifact-integrity contract must hold without relying on any FP
    def bt_idem(signals, asset_returns):
        asset_returns.fillna(0.0, inplace=True)
        signals.clip(-3.0, 3.0, inplace=True)
        return _book_from(signals.shift(1), asset_returns)

    sig_before = small_aligned.signals.copy(deep=True)
    ret_before = small_aligned.asset_returns.copy(deep=True)
    run(small_aligned, CFG, backtest_func=bt_idem)
    pd.testing.assert_frame_equal(small_aligned.signals, sig_before,
                                  check_exact=True)
    pd.testing.assert_frame_equal(small_aligned.asset_returns, ret_before,
                                  check_exact=True)


# ---------------------------------------------------------------------------
# determinism of the extended rng stream (placebos -> date -> xsec)
# ---------------------------------------------------------------------------

def test_deterministic_including_xsec_null(small_aligned):
    a = {r.check: r for r in run(small_aligned, CFG, backtest_func=bt_pure_lag)}
    b = {r.check: r for r in
         run(small_aligned, AuditConfig(n_placebo=16, n_shuffle=20, seed=1),
             backtest_func=bt_pure_lag)}
    sa, sb = a["dynamic.shuffled_labels"], b["dynamic.shuffled_labels"]
    for k in ("p_value", "p_value_xsec", "mean_null", "mean_null_xsec",
              "actual_sr"):
        assert sa.details[k] == sb.details[k], k
