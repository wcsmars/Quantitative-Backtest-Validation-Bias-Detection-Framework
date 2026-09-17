"""Tie-aware placebo ranks and nulls that preserve sparse or dependent structure.

A callback that ignores signals cannot beat an identical null. Sparse
placebos preserve event structure and sign support; block shuffles retain
return dependence. Probe diagnostics disclose context and enforce bounded,
statistically usable run counts.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from qaudit import AuditConfig, BacktestArtifacts, audit
from qaudit._stats import traded_dollars_series
from qaudit.config import _MAX_DYNAMIC_RUNS
from qaudit.dynamic.probes_null import (
    _MIN_PLACEBO_RANDOMIZATION_RUNS, _PHI_MIN_OBS,
    _PLACEBO_SPARSE_ACTIVITY, _SHUFFLE_BLOCK_MAX_DIV,
    _SHUFFLE_PHI_NEGLIGIBLE, _auto_block_len, _block_perm,
    _check_placebo_percentile, _placebo_panel, _placebo_panel_sparse,
    _returns_dependence, _returns_phi, _signal_activity, _signal_phi, run)
from qaudit.errors import AuditFailure, InputValidationError
from qaudit.synthetic import make_backtest_func, make_clean, simulate_market
from qaudit.types import Severity, Status

PCT = "dynamic.placebo_percentile"
BIAS = "dynamic.placebo_pipeline_bias"
SHUF = "dynamic.shuffled_labels"


def _by(results):
    return {r.check: r for r in results}


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean():
    return make_clean()


@pytest.fixture(scope="module")
def clean_aligned(clean):
    return clean.artifacts.aligned()


@pytest.fixture(scope="module")
def clean_results(clean_aligned, clean):
    return _by(run(clean_aligned, AuditConfig(n_placebo=40, n_shuffle=60, seed=1),
                   backtest_func=clean.backtest_func))


@pytest.fixture(scope="module")
def flat_market():
    # no alpha at all (mom_loading=0), no deaths: an information-free panel
    m = simulate_market(n_assets=20, n_periods=400, seed=0, mom_loading=0.0,
                        death_frac=0.0)
    return m["returns"]


# Ties are not wins; an identical null does not establish signal use.

def basket_func(signals, asset_returns):
    return asset_returns.mean(axis=1)          # ignores its signals entirely


def test_signal_ignoring_callable_cannot_pass_percentile(flat_market):
    # A null consisting entirely of ties must not produce a winning percentile.
    sig = pd.DataFrame(np.random.default_rng(3).standard_normal(flat_market.shape),
                       index=flat_market.index, columns=flat_market.columns)
    art = BacktestArtifacts(signals=sig, asset_returns=flat_market).aligned()
    res = _by(run(art, AuditConfig(n_placebo=20, n_shuffle=5, seed=0),
                  backtest_func=basket_func))
    r = res[PCT]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["n_tied"] == r.details["n"] == 20
    assert r.details["percentile"] == 0.0                # strict: 0 beaten
    assert "20/20 placebo outcomes equal the actual SR exactly" in r.message
    assert "hedge dust" in r.message                     # dust arm named too
    assert "verify backtest_func actually consumes its signals argument" \
        in r.message
    assert "extend the sample" not in r.remediation      # extending the sample is the wrong remedy
    assert "signals" in r.remediation
    # the sibling verdicts keep their existing (truthful) shape
    assert res[BIAS].status is Status.SKIP and "hedge dust" in res[BIAS].message
    assert "dynamic.probe_health" not in res


def test_signal_ignoring_callable_is_stopped_by_the_gate(flat_market):
    sig = pd.DataFrame(np.random.default_rng(4).standard_normal(flat_market.shape),
                       index=flat_market.index, columns=flat_market.columns)
    rep = audit(BacktestArtifacts(signals=sig, asset_returns=flat_market),
                AuditConfig(n_placebo=10, n_shuffle=5, seed=0),
                backtest_func=basket_func, include=[PCT])
    assert rep[PCT].status is Status.WARN
    with pytest.raises(AuditFailure):
        rep.gate(require=[PCT], warn_severity=Severity.HIGH)


def test_partial_ties_are_not_counted_as_beaten():
    cfg = AuditConfig()
    actual = 1.0
    null = np.array([0.2, 0.5, 1.0, 1.0, 1.3])          # two exact ties
    r = _check_placebo_percentile(cfg, null, actual, 0)
    assert r.details["n_tied"] == 2
    assert r.details["percentile"] == pytest.approx(2 / 5)   # not 4/5
    assert r.status is Status.WARN
    assert "2/5 placebo outcome(s) tie the actual SR exactly" in r.message
    assert "not counted as beaten" in r.message


def test_zero_spread_null_warns_even_without_hedge_dust_flag():
    r = _check_placebo_percentile(AuditConfig(), np.full(6, 0.7), 0.7, 0)
    assert r.status is Status.WARN and r.severity is Severity.HIGH
    assert r.details["n_tied"] == 6
    assert "6/6 placebo outcomes equal the actual SR exactly" in r.message
    assert "hedge dust" not in r.message                 # only the true arm
    assert "consumes its signals argument" in r.message


def test_hedge_dust_flag_alone_warns_with_counts():
    null = np.array([0.1, 0.2, 0.3, 0.4])
    r = _check_placebo_percentile(AuditConfig(), null, 2.0, 0,
                                  hedge_dust=True, n_hedge_dust=4)
    assert r.status is Status.WARN
    assert r.details["n_tied"] == 0
    assert "4/4 placebo re-runs produced PnL exactly linear" in r.message
    assert "equal the actual SR" not in r.message
    assert "consumes its signals argument" in r.message


def test_crashed_placebo_runs_are_not_counted_as_hedge_dust(flat_market):
    # mixed case: 1 of 10 placebo runs crashes (NaN raw and NaN hedged),
    # the other 9 are basket copies (finite raw, NaN hedged = dust). The
    # dust count is taken among the raw-valid runs: "9/9", not "10/10"
    calls = {"n": 0}

    def crash_then_basket(signals, asset_returns):
        calls["n"] += 1
        if calls["n"] == 3:                  # call 1 = actual, 3 = 2nd placebo
            raise ValueError("boom")
        return asset_returns.mean(axis=1)

    sig = pd.DataFrame(np.random.default_rng(3).standard_normal(flat_market.shape),
                       index=flat_market.index, columns=flat_market.columns)
    art = BacktestArtifacts(signals=sig, asset_returns=flat_market).aligned()
    res = _by(run(art, AuditConfig(n_placebo=10, n_shuffle=3, seed=0),
                  backtest_func=crash_then_basket))
    bias, pct = res[BIAS], res[PCT]
    assert bias.status is Status.SKIP
    assert "9/9 placebo re-runs with a finite raw Sharpe" in bias.message
    assert bias.details["n_hedge_dust"] == 9
    assert bias.details["n_raw_valid"] == 9
    assert bias.details["n_total"] == 10
    assert pct.status is Status.WARN and pct.severity is Severity.HIGH
    assert pct.details["n"] == 9 and pct.details["n_tied"] == 9
    assert "9/9 placebo re-runs produced PnL exactly linear" in pct.message
    assert "10/" not in pct.message and "10/" not in bias.message


def test_degenerate_null_skip_path_unchanged():
    null = np.array([0.1, np.inf, 0.3])
    r = _check_placebo_percentile(AuditConfig(), null, 5.0, 0)
    assert r.status is Status.SKIP
    assert "placebo_pipeline_bias" in r.message


def test_coarse_tie_free_null_fails_closed_on_resolution():
    r = _check_placebo_percentile(AuditConfig(), np.array([0.1, 0.2, 0.3]),
                                  1.0, 0)
    assert r.status is Status.WARN and r.severity is Severity.HIGH
    assert r.details["n_tied"] == 0
    assert r.details["p_value"] == pytest.approx(1 / 4)
    assert r.details["resolution_sufficient"] is False
    assert r.details["min_null_runs"] == _MIN_PLACEBO_RANDOMIZATION_RUNS
    assert "Monte Carlo resolution cannot establish an edge" in r.message
    assert "tie" not in r.message


def test_placebo_percentile_uses_plus_one_randomization_p_value():
    # Zero exceedances is never p=0: the observed run contributes the +1.
    null = np.linspace(-2.0, 0.0, 20)
    r = _check_placebo_percentile(AuditConfig(), null, 1.0, 0)
    assert r.status is Status.PASS
    assert r.details["n_exceed"] == 0
    assert r.details["p_value"] == pytest.approx(1 / 21)
    assert r.details["alpha"] == pytest.approx(0.05)
    assert r.details["p_value_method"] == "plus_one_randomization"
    assert r.details["resolution_sufficient"] is True


def test_configured_alpha_requires_attainable_resolution():
    # 20 draws can clear 5%, but cannot clear 1%: p_min = 1/21.
    cfg = AuditConfig(placebo_percentile_warn=0.99)
    r = _check_placebo_percentile(cfg, np.linspace(-2.0, 0.0, 20), 1.0, 0)
    assert r.status is Status.WARN and r.severity is Severity.HIGH
    assert r.details["best_possible_p_value"] == pytest.approx(1 / 21)
    assert r.details["min_null_runs"] == 100
    assert r.details["resolution_sufficient"] is False
    assert "alpha=0.01" in r.message


def test_zero_alpha_is_explicitly_impossible_not_a_pass():
    cfg = AuditConfig(placebo_percentile_warn=1.0)
    r = _check_placebo_percentile(cfg, np.linspace(-2.0, 0.0, 100), 1.0, 0)
    assert r.status is Status.WARN and r.severity is Severity.HIGH
    assert r.details["alpha"] == 0.0
    assert r.details["min_null_runs"] is None
    assert r.details["resolution_sufficient"] is False
    assert "no finite number" in r.message
    assert "placebo_percentile_warn below 1" in r.remediation


def test_clean_pipeline_with_two_placebos_cannot_pass(clean_aligned, clean):
    # End-to-end: two placebos cannot resolve alpha=0.05, so the percentile
    # cannot PASS.
    r = _by(run(clean_aligned,
                AuditConfig(n_placebo=2, n_shuffle=2, seed=1),
                backtest_func=clean.backtest_func))[PCT]
    assert r.status is Status.WARN and r.severity is Severity.HIGH
    assert r.details["n"] == 2
    assert r.details["resolution_sufficient"] is False
    assert r.details["p_value"] >= 1 / 3


def test_clean_book_percentile_pins_unchanged_honest_guard(clean_results):
    # The rank is descriptive only; the decision uses plus-one p.
    r = clean_results[PCT]
    assert r.status is Status.PASS
    assert r.details["n_tied"] == 0
    assert r.details["percentile"] == 1.0
    assert r.details["p_value"] == pytest.approx(1.0 / 41.0)
    assert r.details["actual_sr"] == pytest.approx(1.79, abs=0.02)
    assert "100th percentile" in r.message


# Compare raw performance on both sides; report hedged context separately.

def test_percentile_details_carry_hedged_context(clean_results):
    r = clean_results[PCT]
    assert np.isfinite(r.details["actual_sr_hedged"])
    assert np.isfinite(r.details["null_q50_hedged"])
    # the hedged actual is the shuffled_labels gate value on this book
    assert r.details["actual_sr_hedged"] == pytest.approx(
        clean_results[SHUF].details["actual_sr"])
    # the verdict is still the raw ranking: identical to recomputing it
    assert r.details["percentile"] == 1.0
    assert r.details["null_q50_hedged"] < r.details["actual_sr_hedged"]


def test_hedge_dust_percentile_reports_nan_hedged_context(flat_market):
    sig = pd.DataFrame(np.random.default_rng(5).standard_normal(flat_market.shape),
                       index=flat_market.index, columns=flat_market.columns)
    art = BacktestArtifacts(signals=sig, asset_returns=flat_market).aligned()
    r = _by(run(art, AuditConfig(n_placebo=4, n_shuffle=2, seed=0),
                backtest_func=basket_func))[PCT]
    assert np.isnan(r.details["actual_sr_hedged"])
    assert np.isnan(r.details["null_q50_hedged"])


# Structure-matched placebos for sparse and event signals.

def sparse_event_signal(index, columns, rng, density=0.03, hold=5,
                        off_value=0.0):
    """+-1 (random sign) held for `hold` bars, start prob density/hold per
    (asset, bar); `off_value` elsewhere. Information-free."""
    n_t, n_a = len(index), len(columns)
    x = np.full((n_t, n_a), off_value, dtype=float)
    p_start = density / hold
    for j in range(n_a):
        t = 0
        while t < n_t:
            if rng.random() < p_start:
                x[t:t + hold, j] = 1.0 if rng.random() < 0.5 else -1.0
                t += hold
            else:
                t += 1
    return pd.DataFrame(x, index=index, columns=columns)


def value_weighted_bt(costs_bps=10.0, lag=1):
    """positions.loc[t] = unit-gross weights from signals.loc[t-lag] (flat
    when the lagged row is all zero/NaN); costs on drift-aware traded
    dollars, like the shipped synthetic pipeline."""
    def bt(signals, rets):
        s = signals.shift(lag).fillna(0.0)
        gross = s.abs().sum(axis=1)
        w = s.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)
        pnl = (w * rets.fillna(0.0)).sum(axis=1)
        return pnl - traded_dollars_series(w, rets) * (costs_bps * 1e-4)
    return bt


def _turnover(sig, rets):
    s = sig.shift(1).fillna(0.0)
    gross = s.abs().sum(axis=1)
    w = s.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)
    return float(traded_dollars_series(w, rets).mean())


@pytest.fixture(scope="module")
def sparse_sig(flat_market):
    return sparse_event_signal(flat_market.index, flat_market.columns,
                               np.random.default_rng(123))


def test_activity_routing(flat_market, sparse_sig, clean_aligned):
    assert _signal_activity(sparse_sig) < _PLACEBO_SPARSE_ACTIVITY
    assert _signal_activity(sparse_sig) == pytest.approx(0.0425, abs=1e-4)
    assert _signal_activity(clean_aligned.signals) == 1.0    # dense z-scores
    nan_sig = sparse_event_signal(flat_market.index, flat_market.columns,
                                  np.random.default_rng(1), off_value=np.nan)
    assert _signal_activity(nan_sig) == 1.0     # NaN-encoded: mask matches
    assert _signal_activity(pd.DataFrame(np.nan, index=flat_market.index,
                                         columns=flat_market.columns)) == 1.0


def test_sparse_placebo_matches_turnover_within_25pct(flat_market, sparse_sig):
    real = _turnover(sparse_sig, flat_market)
    rng = np.random.default_rng(0)
    ratios = [_turnover(_placebo_panel_sparse(sparse_sig, rng), flat_market)
              / real for _ in range(6)]
    assert all(0.75 <= x <= 1.25 for x in ratios), ratios
    # the dense AR(1) placebo of the same signal churns >2x more per bar
    # (0.66 vs 0.30 traded dollars, measured) and trades every bar: a
    # cost-drag mismatch that would let a dense null fail open under costs
    ar1 = _turnover(_placebo_panel(sparse_sig, 0.8, np.random.default_rng(0)),
                    flat_market)
    assert ar1 / real > 1.5


def test_sparse_placebo_preserves_structure_and_destroys_alignment(sparse_sig):
    rng = np.random.default_rng(7)
    p = _placebo_panel_sparse(sparse_sig, rng)
    assert p.shape == sparse_sig.shape
    pd.testing.assert_index_equal(p.index, sparse_sig.index)
    real, plac = sparse_sig.to_numpy(), p.to_numpy()
    # NaN mask has no cells here; off cells stay exactly 0.0 and the complete
    # signal matrix is one common circular row rotation.
    assert np.array_equal(np.isnan(real), np.isnan(plac))
    active_real = np.abs(real) > 1e-12
    active_plac = np.abs(plac) > 1e-12
    # All columns share this fixture's availability calendar, so the whole
    # event matrix must be one common row rotation. This preserves every
    # date's cross-sectional breadth and all co-event relationships;
    # independent per-column shifts would preserve neither.
    assert any(np.array_equal(active_plac, np.roll(active_real, k, axis=0))
               for k in range(real.shape[0]))
    assert np.array_equal(np.sort(active_plac.sum(axis=1)),
                          np.sort(active_real.sum(axis=1)))
    assert np.array_equal(active_plac.astype(int).T @ active_plac.astype(int),
                          active_real.astype(int).T @ active_real.astype(int))
    for j in range(real.shape[1]):
        a_r, a_p = np.abs(real[:, j]) > 1e-12, np.abs(plac[:, j]) > 1e-12
        assert a_r.sum() == a_p.sum()
        assert np.all(plac[~a_p, j] == 0.0)
        runs = lambda a: int(np.sum(a & ~np.concatenate(([False], a[:-1]))))
        assert runs(a_r) <= runs(a_p) <= runs(a_r) + 1
        # Values move with their event row: no value appears in the placebo
        # that the original column never had.
        assert set(plac[a_p, j]) <= set(real[a_r, j])
        assert set(np.abs(plac[a_p, j])) <= set(np.abs(real[a_r, j]))
    # alignment destroyed: the panels do not coincide
    assert np.mean((real != 0) & (plac != 0)) < 0.5 * np.mean(real != 0)


def test_sparse_placebo_rotates_staggered_calendars_as_one_panel(flat_market):
    sig = sparse_event_signal(flat_market.index, flat_market.columns,
                              np.random.default_rng(11))
    sig.iloc[200:, 0] = np.nan            # name dies mid-sample
    sig.iloc[:, 1] = 0.0                  # never-active export column
    sig.iloc[:, 2] = np.nan               # never-observed column
    p = _placebo_panel_sparse(sig, np.random.default_rng(0))

    real = sig.to_numpy()
    plac = p.to_numpy()
    # Independent per-column shifts would smear co-events whenever
    # availability calendars differed. Missingness moves with the complete
    # signal row, so even this staggered panel is exactly one common rotation
    # (equal_nan is essential for the dead/staggered columns).
    assert any(np.array_equal(plac, np.roll(real, k, axis=0), equal_nan=True)
               for k in range(real.shape[0]))
    real_active = np.isfinite(real) & (np.abs(real) > 1e-12)
    plac_active = np.isfinite(plac) & (np.abs(plac) > 1e-12)
    assert np.array_equal(np.sort(plac_active.sum(axis=1)),
                          np.sort(real_active.sum(axis=1)))
    assert np.array_equal(plac_active.astype(int).T @ plac_active.astype(int),
                          real_active.astype(int).T @ real_active.astype(int))
    assert np.array_equal(np.isnan(plac).sum(axis=0),
                          np.isnan(real).sum(axis=0))
    assert np.all(p.iloc[:, 1] == 0.0)
    assert p.iloc[:, 2].isna().all()
    assert (np.abs(p.iloc[:, 0]) > 0).sum() == (np.abs(sig.iloc[:, 0]) > 0).sum()


def test_sparse_placebo_deterministic_given_seed(sparse_sig):
    a = _placebo_panel_sparse(sparse_sig, np.random.default_rng(9))
    b = _placebo_panel_sparse(sparse_sig, np.random.default_rng(9))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    c = _placebo_panel_sparse(sparse_sig, np.random.default_rng(10))
    assert not a.equals(c)


def long_only_bt(costs_bps=10.0):
    """Clips negative scores to 0 before weighting: a long-only book."""
    def bt(signals, rets):
        s = signals.shift(1).fillna(0.0).clip(lower=0.0)
        gross = s.abs().sum(axis=1)
        w = s.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)
        pnl = (w * rets.fillna(0.0)).sum(axis=1)
        return pnl - traded_dollars_series(w, rets) * (costs_bps * 1e-4)
    return bt


def test_one_sided_export_gets_a_one_sided_null(flat_market):
    # a 0/1 long-only mask (30% density): the signed bootstrap keeps every
    # placebo cell >= 0, so the null's net exposure matches the book's; a
    # |value| x coin-flip null would be half short and, through a clipping
    # long-only callable, half as dense as the real book
    mask = sparse_event_signal(flat_market.index, flat_market.columns,
                               np.random.default_rng(500), density=0.30,
                               hold=5).abs()
    assert 0.2 < _signal_activity(mask) < _PLACEBO_SPARSE_ACTIVITY
    for seed in range(4):
        p = _placebo_panel_sparse(mask, np.random.default_rng(seed))
        assert (p.to_numpy() >= 0).all()
        assert set(np.unique(p.to_numpy())) <= {0.0, 1.0}
        assert (p.to_numpy() > 0).sum() == (mask.to_numpy() > 0).sum()
    # An information-free long-only mask must still be rejected by the matched null.
    art = BacktestArtifacts(signals=mask, asset_returns=flat_market,
                            signal_lag=1).aligned()
    res = _by(run(art, AuditConfig(seed=0, n_placebo=30, n_shuffle=4),
                  backtest_func=long_only_bt()))
    assert res[BIAS].details["placebo_kind"] == "sparse"
    assert res[PCT].status is Status.WARN, res[PCT].message


def test_sparse_run_is_deterministic_and_names_sparse_panels(flat_market,
                                                             sparse_sig):
    art = BacktestArtifacts(signals=sparse_sig, asset_returns=flat_market,
                            signal_lag=1).aligned()
    bt = value_weighted_bt()
    a = _by(run(art, AuditConfig(n_placebo=12, n_shuffle=4, seed=2),
                backtest_func=bt))
    b = _by(run(art, AuditConfig(n_placebo=12, n_shuffle=4, seed=2),
                backtest_func=bt))
    for cid in (BIAS, PCT, SHUF):
        assert a[cid].status is b[cid].status
        assert a[cid].message == b[cid].message
    assert a[BIAS].details["placebo_kind"] == "sparse"
    assert a[BIAS].details["activity"] == pytest.approx(0.0425, abs=1e-4)
    assert "structure-matched sparse placebo panels" in a[BIAS].message
    assert "4.2% of cells active" in a[BIAS].message
    assert a[BIAS].status is Status.PASS       # honest engine stays green


def test_dense_book_keeps_ar1_path(clean_results):
    r = clean_results[BIAS]
    assert r.details["placebo_kind"] == "ar1"
    assert r.details["activity"] is None
    assert "placebo AR(1) signal panels (phi=" in r.message


def test_calibration_information_free_sparse_signals_warn(flat_market):
    """Twenty information-free sparse signals with 3% density and five-bar holds run
    through a 10 bp value-weighted pipeline. The seeded experiment requires warnings on
    at least 75% of cases; it checks direction with sampling slack rather than asserting
    a universal false-positive rate."""
    bt = value_weighted_bt()
    t0 = time.perf_counter()
    n_warn = 0
    for i in range(20):
        s_i = sparse_event_signal(flat_market.index, flat_market.columns,
                                  np.random.default_rng(1000 + i))
        art = BacktestArtifacts(signals=s_i, asset_returns=flat_market,
                                signal_lag=1).aligned()
        res = _by(run(art, AuditConfig(seed=7 + i, n_placebo=30, n_shuffle=8),
                      backtest_func=bt))
        assert res[BIAS].details["placebo_kind"] == "sparse"
        n_warn += res[PCT].status is Status.WARN
    assert n_warn >= 15, n_warn
    assert time.perf_counter() - t0 < 30


def test_informative_sparse_signal_still_passes_honest_guard(flat_market):
    # a sparse event signal whose sign is right 85% of the time about the
    # next 5 bars' idiosyncratic move: the sparse null must not WARN it
    rng = np.random.default_rng(21)
    idio = flat_market.sub(flat_market.mean(axis=1), axis=0)
    fwd = idio[::-1].rolling(5, min_periods=1).sum()[::-1].shift(-1)
    base = sparse_event_signal(flat_market.index, flat_market.columns, rng)
    starts = (base != 0) & (base.shift(1).fillna(0.0) == 0)
    sig = base.copy()
    for j, col in enumerate(base.columns):
        for t in np.flatnonzero(starts[col].to_numpy()):
            s = np.sign(fwd[col].iloc[t]) if rng.random() < 0.85 else -np.sign(fwd[col].iloc[t])
            sig.iloc[t:t + 5, j] = np.where(base[col].iloc[t:t + 5] != 0, s, 0.0)
    art = BacktestArtifacts(signals=sig, asset_returns=flat_market,
                            signal_lag=1).aligned()
    res = _by(run(art, AuditConfig(n_placebo=30, n_shuffle=8, seed=3),
                  backtest_func=value_weighted_bt()))
    assert res[BIAS].details["placebo_kind"] == "sparse"
    assert res[PCT].status is Status.PASS, res[PCT].message
    assert res[PCT].details["percentile"] == 1.0


# Circular block permutation for dependent date-shuffle nulls.

def test_block_perm_is_a_permutation_with_blocks_intact():
    n, L = 103, 10
    for seed in range(5):
        perm = _block_perm(n, L, np.random.default_rng(seed))
        assert sorted(perm.tolist()) == list(range(n))
        # within a block consecutive entries advance by 1 (mod n): count
        # the breaks - at most ceil(n/L) of them
        breaks = int(np.sum((np.diff(perm) % n) != 1))
        assert breaks <= int(np.ceil(n / L))
        assert breaks >= 2                       # actually shuffled


def test_block_perm_length_one_is_plain_permutation_bit_identical():
    a = _block_perm(500, 1, np.random.default_rng(3))
    b = np.random.default_rng(3).permutation(500)
    assert np.array_equal(a, b)


def test_block_perm_preserves_per_asset_sample_means():
    # any row permutation keeps each column's mean: the leak/tilt gates of
    # shuffled_labels (which reason about preserved per-asset means) are
    # untouched by the block null
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 6))
    perm = _block_perm(300, 20, rng)
    assert np.allclose(x[perm].mean(axis=0), x.mean(axis=0))


@pytest.mark.parametrize("phi, n, expect", [
    (0.0, 1000, 1), (0.05, 1000, 1), (-0.08, 1000, 1),
    (_SHUFFLE_PHI_NEGLIGIBLE - 1e-9, 1000, 1),
    (0.4, 600, 10), (0.5, 600, 12), (-0.5, 600, 12),   # sign-blind
    (0.2, 600, 6), (0.95, 600, 60),                      # clipped to n//10
    (0.5, 15, 1),                                        # < 2 blocks -> plain
    (float("nan"), 600, 1),
])
def test_auto_block_len_policy(phi, n, expect):
    assert _auto_block_len(phi, n) == expect
    assert 1 <= _auto_block_len(phi, n) <= max(1, n // _SHUFFLE_BLOCK_MAX_DIV)


def test_clean_book_auto_L_is_one_and_pins_unchanged(clean_results):
    r = clean_results[SHUF]
    assert r.details["shuffle_block_len"] == 1
    assert r.details["shuffle_block_auto"] is True
    assert abs(r.details["returns_lag1_autocorr"]) < _SHUFFLE_PHI_NEGLIGIBLE
    assert r.status is Status.PASS
    assert r.details["p_value"] == pytest.approx(1.0 / 61.0)


def test_returns_phi_measures_the_panel():
    rng = np.random.default_rng(0)
    iid = pd.DataFrame(rng.standard_normal((800, 10)))
    assert abs(_returns_phi(iid)) < 0.08
    ar = np.empty((800, 10))
    ar[0] = rng.standard_normal(10)
    for t in range(1, 800):
        ar[t] = 0.5 * ar[t - 1] + rng.standard_normal(10)
    assert _returns_phi(pd.DataFrame(ar)) == pytest.approx(0.5, abs=0.08)
    assert _returns_phi(pd.DataFrame(np.zeros((50, 3)))) == 0.0


def test_auto_dependence_cannot_cancel_opposite_signed_ar_assets():
    rng = np.random.default_rng(303)
    n_t, n_a = 1200, 12
    phis = np.r_[np.full(n_a // 2, 0.6), np.full(n_a // 2, -0.6)]
    innovations = rng.standard_normal((n_t, n_a))
    x = np.empty_like(innovations)
    x[0] = innovations[0]
    scale = np.sqrt(1.0 - phis ** 2)
    for t in range(1, n_t):
        x[t] = phis * x[t - 1] + scale * innovations[t]
    rets = pd.DataFrame(x)

    dep = _returns_dependence(rets)
    assert abs(dep["signed"]) < _SHUFFLE_PHI_NEGLIGIBLE  # a signed-only estimate would choose L=1
    assert dep["level_abs"] > 0.5
    assert dep["combined"] == dep["level_abs"]
    assert _auto_block_len(dep["combined"], n_t) > 1


def test_auto_dependence_includes_volatility_clustering():
    rng = np.random.default_rng(404)
    n_t, n_a = 1200, 12
    log_variance = np.zeros((n_t, n_a))
    volatility_shocks = rng.standard_normal((n_t, n_a))
    for t in range(1, n_t):
        log_variance[t] = (0.96 * log_variance[t - 1]
                           + 0.22 * volatility_shocks[t])
    rets = pd.DataFrame(np.exp(0.5 * log_variance)
                        * rng.standard_normal((n_t, n_a)))

    dep = _returns_dependence(rets)
    assert dep["level_abs"] < _SHUFFLE_PHI_NEGLIGIBLE
    assert dep["squared_abs"] > _SHUFFLE_PHI_NEGLIGIBLE
    assert dep["combined"] == dep["squared_abs"]
    assert _auto_block_len(dep["combined"], n_t) > 1


def test_run_reports_honest_auto_block_basis_for_mixed_ar_panel():
    rng = np.random.default_rng(505)
    n_t, n_a = 500, 8
    phis = np.r_[np.full(n_a // 2, 0.5), np.full(n_a // 2, -0.5)]
    e = rng.standard_normal((n_t, n_a)) * 0.01
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, n_t):
        x[t] = phis * x[t - 1] + e[t]
    idx = pd.bdate_range("2020-01-01", periods=n_t)
    cols = [f"A{i}" for i in range(n_a)]
    rets = pd.DataFrame(x, index=idx, columns=cols)
    sig = pd.DataFrame(rng.standard_normal((n_t, n_a)), index=idx,
                       columns=cols)
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            signal_lag=1).aligned()
    r = _by(run(art, AuditConfig(n_placebo=2, n_shuffle=2, seed=1),
                backtest_func=make_backtest_func(lag=1)))[SHUF]

    assert abs(r.details["returns_lag1_autocorr"]) < 0.15
    assert r.details["returns_lag1_abs_autocorr"] > 0.4
    assert r.details["returns_block_dependence"] > 0.4
    assert r.details["shuffle_block_len"] > 1
    assert r.details["shuffle_p_value_method"] == \
        "plus_one_block_randomization"
    assert "not a universal time-series calibration" in \
        r.details["shuffle_null_assumption"]


@pytest.mark.parametrize("n_obs", [1, 2, 3, _PHI_MIN_OBS - 1])
def test_short_columns_do_not_warn_or_enter_the_median(n_obs):
    # 2 observations make numpy's cov emit "Degrees of freedom <= 0" via
    # warnings.warn (fatal under -W error, not an errstate condition) and
    # 3 observations contribute exactly +/-1: both helpers skip columns
    # with fewer than _PHI_MIN_OBS observations
    rng = np.random.default_rng(1)
    short = pd.DataFrame(rng.standard_normal((n_obs, 3)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _returns_phi(short) == 0.0
        assert _signal_phi(short) == 0.0
        # a short column next to long ones is ignored, not averaged in
        ar = np.empty((800, 4))
        ar[0] = rng.standard_normal(4)
        for t in range(1, 800):
            ar[t] = 0.5 * ar[t - 1] + rng.standard_normal(4)
        mixed = pd.DataFrame(ar)
        mixed.iloc[n_obs:, 0] = np.nan
        assert _returns_phi(mixed) == pytest.approx(
            _returns_phi(mixed.iloc[:, 1:]))
        assert _returns_phi(mixed) == pytest.approx(0.5, abs=0.08)
    # exactly _PHI_MIN_OBS observations are measured
    ok = pd.DataFrame(rng.standard_normal((_PHI_MIN_OBS, 2)))
    assert np.isfinite(_returns_phi(ok))


def test_two_bar_asset_in_the_grid_does_not_crash_the_null_probes(clean):
    # A two-bar asset must be excluded from autocorrelation estimation without
    # raising a RuntimeWarning or interrupting the other null checks.
    art0 = clean.artifacts
    rets = art0.asset_returns.copy()
    col = rets.columns[0]
    rets.loc[rets.index[:500], col] = np.nan
    rets.loc[rets.index[502:], col] = np.nan
    art = BacktestArtifacts(signals=art0.signals.copy(), asset_returns=rets,
                            signal_lag=1).aligned()
    assert int(art.asset_returns[col].notna().sum()) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _by(run(art, AuditConfig(n_placebo=3, n_shuffle=3, seed=1),
                      backtest_func=clean.backtest_func))
    for cid in (BIAS, PCT, SHUF):
        assert res[cid].status is not Status.ERROR
    assert res[BIAS].status is Status.PASS
    assert res[SHUF].details["shuffle_block_len"] == 1
    assert abs(res[SHUF].details["returns_lag1_autocorr"]) < _SHUFFLE_PHI_NEGLIGIBLE


def _ar_garch_returns(seed, phi, n_t=600, n_a=20):
    """Idiosyncratic AR(1) with GARCH(1,1) vol plus an AR(1) market factor."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_t)
    z = rng.standard_normal((n_t, n_a))
    base_sig, a, b = 0.015, 0.10, 0.85
    omega = base_sig ** 2 * (1 - a - b)
    h = np.full(n_a, base_sig ** 2)
    eps = np.empty((n_t, n_a))
    for t in range(n_t):
        eps[t] = np.sqrt(h) * z[t]
        h = omega + a * eps[t] ** 2 + b * h
    r = np.empty((n_t, n_a))
    r[0] = eps[0]
    for t in range(1, n_t):
        r[t] = phi * r[t - 1] + eps[t]
    mz = rng.standard_normal(n_t) * 0.008
    m = np.empty(n_t)
    m[0] = mz[0]
    for t in range(1, n_t):
        m[t] = phi * m[t - 1] + mz[t]
    beta = rng.uniform(0.7, 1.3, n_a)
    return pd.DataFrame(r + np.outer(m, beta), index=dates,
                        columns=[f"A{i:02d}" for i in range(n_a)])


def _ar1_signal(seed, phi_s, index, columns):
    rng = np.random.default_rng(10_000 + seed)
    e = rng.standard_normal((len(index), len(columns)))
    x = np.empty_like(e)
    x[0] = e[0]
    sc = np.sqrt(1 - phi_s ** 2)
    for t in range(1, len(index)):
        x[t] = phi_s * x[t - 1] + sc * e[t]
    return pd.DataFrame(x, index=index, columns=columns)


def test_calibration_block_null_reduces_false_pass_on_dependent_returns():
    """Compare block and single-row nulls on no-edge AR(0.5) and GARCH books with
    persistent noise signals. Across the fixed seed sample, automatic blocks must not
    increase false passes and must keep them at or below 15%. Residual volatility and
    signal dependence limit interpretation of this finite-sample experiment."""
    bt = make_backtest_func(lag=1)
    t0 = time.perf_counter()
    rates = {}
    for L in (1, None):
        n_pass = 0
        for seed in range(20):
            rets = _ar_garch_returns(seed, 0.5)
            sig = _ar1_signal(seed, 0.95, rets.index, rets.columns)
            art = BacktestArtifacts(signals=sig, asset_returns=rets,
                                    signal_lag=1).aligned()
            cfg = AuditConfig(seed=1, n_placebo=2, n_shuffle=60,
                              shuffle_block_len=L)
            r = _by(run(art, cfg, backtest_func=bt))[SHUF]
            if L is None:
                # phi_r measures 0.48-0.52 per seed -> L = 12 or 13
                assert 10 <= r.details["shuffle_block_len"] <= 14
                assert r.details["shuffle_block_auto"] is True
                assert r.details["returns_lag1_autocorr"] > _SHUFFLE_PHI_NEGLIGIBLE
            else:
                assert r.details["shuffle_block_len"] == 1
                assert r.details["shuffle_block_auto"] is False
            n_pass += r.status is Status.PASS
        rates[L] = n_pass / 20
    assert rates[None] <= rates[1], rates
    assert rates[None] <= 0.15, rates
    assert time.perf_counter() - t0 < 40


def test_genuine_edge_on_dependent_returns_still_passes_honest_guard():
    # a slow AR(0.95) signal that drives next-bar returns on an AR(0.25)+
    # GARCH panel (measured phi_r 0.26, well clear of the 0.10 bar, so the
    # block path is exercised unambiguously: L = 7): the block null must
    # not WARN a real edge
    rets = _ar_garch_returns(5, 0.25)
    sig = _ar1_signal(5, 0.95, rets.index, rets.columns)
    driven = rets + 0.004 * sig.shift(1).fillna(0.0).to_numpy()
    art = BacktestArtifacts(signals=sig, asset_returns=driven,
                            signal_lag=1).aligned()
    r = _by(run(art, AuditConfig(seed=1, n_placebo=2, n_shuffle=60),
                backtest_func=make_backtest_func(lag=1)))[SHUF]
    assert r.details["returns_lag1_autocorr"] > 0.2
    assert 5 <= r.details["shuffle_block_len"] <= 9, r.details
    assert r.status is Status.PASS, r.message
    assert r.details["p_value"] == pytest.approx(1.0 / 61.0)


def test_explicit_block_len_is_used_and_clipped_to_the_sample(clean_aligned,
                                                              clean):
    # n_shuffle=25: the best reachable p is 1/26 = 0.038 < 0.05
    res = _by(run(clean_aligned, AuditConfig(n_placebo=2, n_shuffle=25, seed=1,
                                             shuffle_block_len=20),
                  backtest_func=clean.backtest_func))[SHUF]
    assert res.details["shuffle_block_len"] == 20
    assert res.details["shuffle_block_auto"] is False
    assert res.status is Status.PASS                     # honest book
    huge = _by(run(clean_aligned, AuditConfig(n_placebo=2, n_shuffle=3, seed=1,
                                              shuffle_block_len=10 ** 6),
                   backtest_func=clean.backtest_func))[SHUF]
    assert huge.details["shuffle_block_len"] == len(clean_aligned.asset_returns)


def test_block_null_keeps_rng_order_deterministic(clean_aligned, clean):
    cfg = dict(n_placebo=3, n_shuffle=8, seed=4, shuffle_block_len=7)
    a = _by(run(clean_aligned, AuditConfig(**cfg), backtest_func=clean.backtest_func))
    b = _by(run(clean_aligned, AuditConfig(**cfg), backtest_func=clean.backtest_func))
    for k in ("p_value", "mean_null", "p_value_xsec", "mean_null_xsec"):
        assert a[SHUF].details[k] == b[SHUF].details[k]


def test_plain_permutation_leak_detection_intact_under_block_null():
    # an accounting leak (constant credit) survives the block null exactly
    # as it survives a row permutation: the FAIL path is not weakened
    c = make_clean()
    art = c.artifacts.aligned()

    def replay(signals, asset_returns):
        return c.backtest_func(art.signals, art.asset_returns)

    r = _by(run(art, AuditConfig(n_placebo=3, n_shuffle=6, seed=1,
                                 shuffle_block_len=25),
                backtest_func=replay))[SHUF]
    assert r.status is Status.FAIL and r.severity is Severity.CRITICAL


# Run-count caps and shuffle-block configuration.

@pytest.mark.parametrize("name", ["n_placebo", "n_shuffle", "date_shift_max",
                                  "truncation_sample_dates"])
def test_run_counts_capped_at_construction(name):
    with pytest.raises(InputValidationError) as ei:
        AuditConfig(**{name: _MAX_DYNAMIC_RUNS + 1})
    assert name in str(ei.value)
    assert str(_MAX_DYNAMIC_RUNS) in str(ei.value)
    assert "re-run" in str(ei.value)
    assert getattr(AuditConfig(**{name: _MAX_DYNAMIC_RUNS}), name) \
        == _MAX_DYNAMIC_RUNS                              # boundary legal
    with pytest.raises(InputValidationError):
        AuditConfig(**{name: 10 ** 9})


def test_run_count_cap_on_mutation_rolls_back():
    c = AuditConfig(n_placebo=50)
    with pytest.raises(InputValidationError):
        c.n_placebo = _MAX_DYNAMIC_RUNS + 1
    assert c.n_placebo == 50
    c.n_shuffle = 500                                     # under the cap
    assert c.n_shuffle == 500


def test_existing_count_validators_untouched_honest_guard():
    assert AuditConfig(n_placebo=50).n_placebo == 50      # pinned
    for bad in (0, -1, 2.5, True, float("inf"), float("nan"), "100"):
        with pytest.raises(InputValidationError):
            AuditConfig(n_placebo=bad)


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "3", float("inf"),
                                 float("nan")])
def test_shuffle_block_len_rejects_nonsense(bad):
    with pytest.raises(InputValidationError, match="shuffle_block_len"):
        AuditConfig(shuffle_block_len=bad)


def test_shuffle_block_len_accepts_none_and_whole_numbers():
    assert AuditConfig().shuffle_block_len is None
    assert AuditConfig(shuffle_block_len=1).shuffle_block_len == 1
    assert AuditConfig(shuffle_block_len=20.0).shuffle_block_len == 20.0
    c = AuditConfig()
    c.shuffle_block_len = 5
    assert c.shuffle_block_len == 5
    with pytest.raises(InputValidationError):
        c.shuffle_block_len = 0
    assert c.shuffle_block_len == 5
