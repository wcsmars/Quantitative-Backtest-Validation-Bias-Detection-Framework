"""Tie-aware rank nulls, overlapping-label dependence, and numeric safety.

Exact tails condition on observed rank multisets and configured thresholds.
Clustered labels widen count bounds while horizon-one behavior stays stable.
Large finite values must neither crash a check nor erase usable evidence;
exact-null construction has a bounded peak-memory budget.
"""
import itertools
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson, spearmanr

from qaudit.checks import leakage
from qaudit.checks.leakage import (
    _OUTLIER_COUNT_ALPHA, _defactored_with_pcs, _distinct_key,
    _exact_spearman_null, _exact_spearman_null_tied, _mc_spearman_null_tied,
    _overlap_cluster_factor, _rank_multiset_key, _spearman_tail_ge,
    _standardized_rowwise_corr, _tie_profile, _tied_two_sided_p, run)
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.types import Severity, Status

CFG = AuditConfig()
OUTLIER = "leakage.ic_outlier_dates"
PERFECT = "leakage.perfect_rank_dates"
IDENTITY = "leakage.signal_target_identity"

ONE_HOT_KEY_8 = tuple(sorted([8] * 7 + [16]))   # 7 zeros (avg rank 4), 1 hot


def _by_check(sig, rets, cfg=CFG, horizon=1):
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            label_horizon=horizon)
    return {r.check: r for r in run(art, cfg)}


def _one_hot_panel(seed, n_dates=500, n_names=8, leak=False):
    """iid one-hot signal; iid gaussian returns. leak=True points the hot
    name at tomorrow's argmax return (a genuine 1-day-lookahead leak)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n_dates)
    cols = [f"A{i}" for i in range(n_names)]
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(n_dates, n_names)),
                        index=idx, columns=cols)
    s = np.zeros((n_dates, n_names))
    if leak:
        fwd = rets.shift(-1)
        hot = np.nan_to_num(fwd.to_numpy(), nan=-1e9).argmax(axis=1)
    else:
        hot = rng.integers(0, n_names, size=n_dates)
    s[np.arange(n_dates), hot] = 1.0
    return pd.DataFrame(s, index=idx, columns=cols), rets


def _noise_panel(seed, n_dates=500, n_names=8, ret_vol=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n_dates)
    cols = [f"A{i}" for i in range(n_names)]
    sig = pd.DataFrame(rng.normal(size=(n_dates, n_names)), idx, cols)
    rets = pd.DataFrame(rng.normal(0, ret_vol, size=(n_dates, n_names)),
                        idx, cols)
    return sig, rets


# ---------------------------------------------------------------------------
# 1. tie-aware exact null
# ---------------------------------------------------------------------------

def test_one_hot_null_atoms_and_tail_exact():
    # A one-hot row at breadth eight has eight correlation atoms and a two-sided tail
    # of 0.5 at 0.40; the distinct-rank tail is about 0.3268.
    vals, tail, pmf = _exact_spearman_null_tied(ONE_HOT_KEY_8,
                                                _distinct_key(8))
    assert vals.size == 8
    assert np.allclose(sorted(np.abs(vals)),
                       [0.0825, 0.0825, 0.2474, 0.2474,
                        0.4124, 0.4124, 0.5774, 0.5774], atol=5e-5)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    two = float(_tied_two_sided_p(ONE_HOT_KEY_8, _distinct_key(8),
                                  0.40, 0.40)[0])
    assert two == pytest.approx(0.5000, abs=1e-12)
    coded = 2.0 * float(_spearman_tail_ge(8, 0.40))
    assert coded == pytest.approx(0.3268, abs=2e-4)   # the distinct-rank approximation


def test_tied_null_matches_brute_force_enumeration():
    # Independent oracle at n=6: a {3,2,1}-tied signal row against a
    # distinct target - enumerate all 720 pairings through
    # scipy.stats.spearmanr and compare the full pmf.
    row = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    tgt = np.arange(6, dtype=float)
    key = _rank_multiset_key(row)
    assert key is not None
    vals, tail, pmf = _exact_spearman_null_tied(key, _distinct_key(6))
    got = {}
    for perm in itertools.permutations(range(6)):
        r = float(spearmanr(row, tgt[list(perm)]).statistic)
        got[round(r, 12)] = got.get(round(r, 12), 0) + 1
    ref_vals = np.array(sorted(got))
    ref_pmf = np.array([got[v] for v in sorted(got)]) / 720.0
    assert vals.size == ref_vals.size
    assert np.allclose(vals, ref_vals, atol=1e-9)
    assert np.allclose(pmf, ref_pmf, atol=1e-12)


def test_tied_null_distinct_keys_reduce_to_fast_path():
    # Both-sides-distinct through the tied builder must reproduce the
    # pinned distinct-rank table (the fast path stays authoritative).
    for n in (5, 8):
        v0, t0, p0 = _exact_spearman_null(n)
        v1, t1, p1 = _exact_spearman_null_tied(_distinct_key(n),
                                               _distinct_key(n))
        assert np.allclose(v0, v1, atol=1e-12)
        assert np.allclose(t0, t1, atol=1e-12)
        assert np.allclose(p0, p1, atol=1e-12)


def test_mc_tied_null_matches_analytic_one_hot_wide():
    # n=12 takes the MC branch. A one-hot row has an analytic null: rho
    # depends only on the hot name's target rank, uniform over 12 values.
    n = 12
    key = tuple(sorted([n] * (n - 1) + [2 * n]))    # zeros avg rank n/2+...
    row = np.zeros(n)
    row[0] = 1.0
    assert _rank_multiset_key(row) == key
    atoms = []
    for r in range(n):
        tgt = np.empty(n)
        tgt[0] = r + 1
        tgt[1:] = np.delete(np.arange(1, n + 1), r)
        atoms.append(float(spearmanr(row, tgt).statistic))
    atoms = np.sort(atoms)
    x = float(np.abs(atoms)[len(atoms) // 2])       # a mid-support atom
    truth = float(np.mean(np.abs(atoms) >= x - 1e-12))
    mc = float(_tied_two_sided_p(key, _distinct_key(n), x, x)[0])
    se = np.sqrt(truth * (1 - truth) / leakage._TIE_MC_DRAWS)
    assert mc == pytest.approx(truth, abs=5 * se + 1e-6)
    # determinism: the seeded draw is reproducible
    assert np.array_equal(_mc_spearman_null_tied(key, _distinct_key(n)),
                          _mc_spearman_null_tied(key, _distinct_key(n)))


def test_rank_multiset_key_semantics():
    # Tie-free rows -> None (distinct fast path); mirror tie patterns must
    # get different keys (their null laws are mirror images, not equal).
    assert _rank_multiset_key(np.array([3.0, 1.0, 2.0])) is None
    low = _rank_multiset_key(np.array([0.0, 5.0, 5.0, 5.0]))   # singleton low
    high = _rank_multiset_key(np.array([9.0, 5.0, 5.0, 5.0]))  # singleton high
    assert low is not None and high is not None and low != high


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_one_hot_noise_panel_not_convicted(seed):
    # Independent one-hot noise has an expected exceedance rate of about 0.5 per date
    # under its tie-aware null.
    sig, rets = _one_hot_panel(seed)
    r = _by_check(sig, rets)[OUTLIER]
    assert r.status is Status.PASS, (seed, r.message)
    d = r.details
    assert d["n_tied_dates"] == d["n_dates"]        # every date one-hot
    assert d["expected_under_null"] == pytest.approx(
        0.5 * d["n_dates"], rel=0.02)
    assert d["observed_outliers"] <= d["noise_bound"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_hot_leak_still_convicted(seed):
    # Honest-power guard: the same tied geometry with a real leak (hot name
    # = tomorrow's argmax return) must still convict - the tie-aware null may
    # not buy the attacker cover (observed ~460 vs bound ~310).
    sig, rets = _one_hot_panel(100 + seed, leak=True)
    r = _by_check(sig, rets)[OUTLIER]
    assert r.status is Status.FAIL, (seed, r.message)
    assert r.severity is Severity.CRITICAL
    assert r.details["observed_outliers"] > r.details["noise_bound"]


def test_tie_free_panel_unaffected_by_tie_machinery():
    # A continuous panel uses no tied tables; expected counts equal the independently
    # reconstructed distinct-rank tails.
    sig, rets = _noise_panel(7)
    r = _by_check(sig, rets)[OUTLIER]
    d = r.details
    assert d["n_tied_dates"] == 0
    assert d["cluster_factor"] == 1.0
    assert r.status is Status.PASS, r.message


def test_tie_profile_masks_and_routes():
    # The profile must key off the jointly-finite cells (the IC's own mask)
    # and omit unmeasurable dates.
    idx = pd.bdate_range("2020-01-02", periods=3)
    cols = list("ABCDEF")
    sig = pd.DataFrame([[1.0, 1.0, 1.0, 2.0, 3.0, 4.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], idx, cols)
    tgt = pd.DataFrame(np.arange(18, dtype=float).reshape(3, 6), idx, cols)
    prof = _tie_profile(sig, tgt)
    assert prof[idx[0]][0] == 6 and prof[idx[0]][1] is not None
    assert prof[idx[1]][0] == 5 and prof[idx[1]][1] is None  # ties masked out
    assert idx[2] not in prof                       # constant row: no IC


# ---------------------------------------------------------------------------
# 2. overlapping-label dependence
# ---------------------------------------------------------------------------

def _block21_t3_panel(seed, T=1500, N=8):
    """Monthly block-constant signal with
    t(3) returns (the raw normal draw is part of the rng stream)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=T)
    cols = [f"A{i}" for i in range(N)]
    rng.normal(size=(T, N))                         # stream compatibility
    sigv = np.repeat(rng.normal(size=(T // 21 + 1, N)), 21, axis=0)[:T]
    rets = pd.DataFrame(rng.standard_t(3, size=(T, N)) * 0.01, idx, cols)
    return pd.DataFrame(sigv, index=idx, columns=cols), rets


def test_cluster_factor_is_exactly_one_at_horizon_one():
    sig, _ = _block21_t3_panel(2001)
    assert _overlap_cluster_factor(sig, 1) == 1.0


def test_cluster_factor_iid_stays_near_poisson():
    # Honest guard: overlap alone (iid ranks) must not buy a material
    # inflation - measured var/mean ~0.5 at H=21, c estimate ~1.07.
    sig, _ = _noise_panel(3, n_dates=1500, n_names=8)
    c = _overlap_cluster_factor(sig, 21)
    assert 1.0 <= c < 1.25, c


def test_cluster_factor_monthly_block_matches_theory():
    # Block-constant monthly ranks: rhoS_k ~ (21-k)/21, so
    # c = 1 + (2/21^2) * sum_{j=1..20} j^2 ~ 14.0, capped at H.
    sig, _ = _block21_t3_panel(2001)
    c21 = _overlap_cluster_factor(sig, 21)
    assert 10.0 < c21 <= 21.0, c21
    c5 = _overlap_cluster_factor(sig, 5)
    assert 3.0 < c5 < 5.0, c5
    assert c5 < c21                                 # monotone in overlap


@pytest.mark.parametrize("seed", [2001, 2015])
def test_monthly_block_monthly_horizon_passes_under_cluster_bound(seed):
    # Overlapping monthly labels need a cluster-scaled bound; an independent Poisson
    # bound understates their count variance.
    sig, rets = _block21_t3_panel(seed)
    r = _by_check(sig, rets, horizon=21)[OUTLIER]
    d = r.details
    prefix_bound = int(poisson.isf(_OUTLIER_COUNT_ALPHA,
                                   d["expected_under_null"]))
    assert d["observed_outliers"] > prefix_bound    # exceeds the independent Poisson bound
    assert r.status is Status.PASS, (seed, r.message)
    assert d["cluster_factor"] > 5.0
    assert d["observed_outliers"] <= d["noise_bound"]


def test_h1_bound_identical_with_and_without_cluster_machinery():
    # H=1 is the pinned regime: the bound expression must collapse to the
    # plain Poisson bound bit-for-bit (c == 1.0 exactly).
    sig, rets = _noise_panel(11)
    d = _by_check(sig, rets, horizon=1)[OUTLIER].details
    assert d["cluster_factor"] == 1.0
    assert d["noise_bound"] == int(poisson.isf(_OUTLIER_COUNT_ALPHA,
                                               d["expected_under_null"]))


def _persist_leak_panel(seed, H=5, T=1500, N=30, leak_frac=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-03", periods=T)
    cols = [f"A{i:02d}" for i in range(N)]
    raw = rng.normal(size=(T, N))
    sig = pd.DataFrame(pd.DataFrame(raw).rolling(10, min_periods=1)
                       .mean().to_numpy(), idx, cols)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(T, N)), idx, cols)
    if leak_frac > 0:
        fwd = rets.rolling(H).sum().shift(-H)       # ~ the H-period target
        z = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1), axis=0)
        pick = rng.random(T) < leak_frac
        sig[pick] = z[pick]
    return sig, rets


def test_persistent_signal_overlap_power_retained_at_h5():
    # Honest-power pair at label_horizon 5 on a persistent (10-day-mean)
    # signal: clean book PASSes under the inflated bound (c ~4), a
    # 10%-of-dates forward-return leak still convicts (obs ~190 vs ~110).
    clean = _by_check(*_persist_leak_panel(11), horizon=5)[OUTLIER]
    assert clean.status is Status.PASS, clean.message
    assert clean.details["cluster_factor"] > 2.5
    leaky = _by_check(*_persist_leak_panel(11, leak_frac=0.10),
                      horizon=5)[OUTLIER]
    assert leaky.status is Status.FAIL, leaky.message
    assert leaky.details["observed_outliers"] > leaky.details["noise_bound"]


# ---------------------------------------------------------------------------
# 3. threshold-aware perfect-rank null
# ---------------------------------------------------------------------------

def test_lowered_perfect_bar_on_noise_passes():
    # At a configured perfect-rank threshold of 0.50, breadth-eight noise needs the
    # matching two-sided null tail, about 0.216 per date.
    sig, rets = _noise_panel(7)
    r = _by_check(sig, rets, AuditConfig(leak_perfect_ic=0.50))[PERFECT]
    assert r.status is Status.PASS, r.message
    d = r.details
    exact = 2.0 * float(_spearman_tail_ge(8, 0.50))
    assert exact == pytest.approx(0.216, abs=5e-4)
    assert d["expected_noise_dates"] == pytest.approx(
        exact * d["n_dates"], abs=0.011)     # details round to 2 decimals
    assert d["n_perfect"] <= d["noise_bound"]


@pytest.mark.parametrize("bar", [0.50, 0.90])
def test_true_leak_convicted_at_lowered_and_default_bar(bar):
    # Honest-power guard: signal = forward return + tiny noise must FAIL
    # at both the lowered and the default bar.
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2018-01-02", periods=500)
    cols = [f"A{i}" for i in range(8)]
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(500, 8)), idx, cols)
    sig = (rets.shift(-1)
           + rng.normal(0, 0.001, size=(500, 8))).fillna(0.0)
    r = _by_check(sig, rets, AuditConfig(leak_perfect_ic=bar))[PERFECT]
    assert r.status is Status.FAIL, (bar, r.message)
    assert r.severity is Severity.CRITICAL


def test_perfect_expected_budgets_wide_books():
    # The exact/t machinery must budget a real expectation at any breadth/bar
    # combination rather than a clipped lookup table.
    sig, rets = _noise_panel(21, n_names=20)
    r = _by_check(sig, rets, AuditConfig(leak_perfect_ic=0.55))[PERFECT]
    d = r.details
    assert d["expected_noise_dates"] > 1.0          # far above any clipped-table value
    assert r.status is Status.PASS, r.message


def test_perfect_rank_tie_aware_at_lowered_bar():
    # tie-aware x lowered-bar interaction: at bar 0.55 the one-hot lattice reaches the
    # bar only through its top atom (|rho| = 0.5774, P = 0.25/date), while
    # the distinct-rank rate is 0.151/date - enough under-budget that the
    # distinct machinery would convict this pure-noise panel (n_perfect 133
    # vs a would-be bound of ~111); the tie-aware budget (124.75 expected,
    # bound ~170) reads it as the null it is.
    sig, rets = _one_hot_panel(5)
    r = _by_check(sig, rets, AuditConfig(leak_perfect_ic=0.55))[PERFECT]
    d = r.details
    assert r.status is Status.PASS, r.message
    assert d["expected_noise_dates"] == pytest.approx(
        0.25 * d["n_dates"], abs=0.011)
    # reconstruct the distinct-rank budget: this panel exceeds it, so the
    # PASS above is the tie-awareness earning its keep, not slack
    exp_d = 2.0 * float(_spearman_tail_ge(8, 0.55)) * d["n_dates"]
    bound_d = exp_d + 4.0 * np.sqrt(max(exp_d, 0.25)) + 1.0
    assert d["n_perfect"] > bound_d
    assert d["n_perfect"] / d["n_dates"] > CFG.leak_perfect_date_frac


def test_default_bar_noise_passes():
    # Pure noise stays clean at the default threshold.
    sig, rets = _noise_panel(3)
    assert _by_check(sig, rets)[PERFECT].status is Status.PASS


# ---------------------------------------------------------------------------
# 4. overflow hazards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mag", [1e154, 1e300])
def test_no_runtime_warnings_on_huge_finite_panels(mag):
    # Huge finite values must not trigger standard-deviation overflow under warnings-
    # as-errors or strict NumPy error handling.
    sig, rets = _noise_panel(777, n_dates=310, n_names=9)
    rets.iloc[40:60, 3:6] = mag
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = _by_check(sig, rets)
    assert set(res) == set(leakage.ALL_CHECKS)
    old = np.seterr(all="raise")
    try:
        res = _by_check(sig, rets)
    finally:
        np.seterr(**old)
    assert set(res) == set(leakage.ALL_CHECKS)


def test_corrupt_cells_do_not_nan_the_defactored_panel():
    # A small corrupt block must not turn the entire defactored panel into NaNs.
    # Nonfinite variance routes to the fallback and preserves usable dates.
    sig, rets = _noise_panel(11, n_dates=400, n_names=12)
    rets.iloc[100:120, 0:4] = 1e300
    star, _ = _defactored_with_pcs(rets.shift(-1))
    finite = np.isfinite(star.to_numpy())
    assert finite.sum() > 0.9 * (400 - 21) * 12
    clean_rows = np.ones(400, dtype=bool)
    clean_rows[100:120] = False
    clean_rows[-1] = False                          # shift(-1) tail NaN
    assert np.isfinite(star.to_numpy()[clean_rows]).all()


def test_power_survives_corrupt_cells():
    # A corrupt block must preserve detection of a 10%-of-dates leak in the clean
    # stretch. The same corruption without a leak provides the paired control.
    sig, rets = _noise_panel(13, n_dates=400, n_names=30)
    fwd = rets.shift(-1)
    z = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1), axis=0)
    rng = np.random.default_rng(99)
    pick = rng.random(400) < 0.10
    sig[pick] = z[pick].where(sig[pick].notna())
    rets.iloc[100:120, 0:4] = 1e300
    r = _by_check(sig, rets)[OUTLIER]
    assert r.status is Status.FAIL, r.message
    sig2, rets2 = _noise_panel(14, n_dates=400, n_names=30)
    rets2.iloc[100:120, 0:4] = 1e300
    r2 = _by_check(sig2, rets2)[OUTLIER]
    assert r2.status is Status.PASS, r2.message


def test_identity_check_scale_safe_and_still_detects():
    # Scale-safety must not suppress detection: an affine copy of the
    # target at corruption scale (1e200 * z(fwd)) is still an identity
    # FAIL, with no RuntimeWarning raised on the way.
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2019-01-02", periods=300)
    cols = [f"A{i}" for i in range(10)]
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(300, 10)), idx, cols)
    fwd = rets.shift(-1)
    sig = (1e200 * fwd.sub(fwd.mean(axis=1), axis=0)
           .div(fwd.std(axis=1), axis=0)).fillna(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        r = _by_check(sig, rets)[IDENTITY]
    assert r.status is Status.FAIL, r.message
    assert r.details["median_corr"] == pytest.approx(1.0, abs=1e-6)


def test_rowwise_corr_unchanged_on_sane_scales():
    # Honest guard: the max-abs pre-scaling must be numerically inert on
    # ordinary data (corr is scale-invariant; agreement to float rounding).
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2019-01-02", periods=50)
    cols = list("ABCDEFG")
    a = pd.DataFrame(rng.normal(size=(50, 7)), idx, cols)
    b = pd.DataFrame(rng.normal(size=(50, 7)), idx, cols)
    got = _standardized_rowwise_corr(a, b)
    for i in range(50):
        ref = np.corrcoef(a.iloc[i], b.iloc[i])[0, 1]
        assert got.iloc[i] == pytest.approx(ref, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. exact-null memory
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="resource module is POSIX-only")
def test_exact_null_peak_memory_bounded():
    # The exact-null build must stay below 250 MiB of additional peak RSS. This catches
    # factorial-size wide-dtype temporaries without depending on an absolute process
    # baseline.
    code = textwrap.dedent("""
        import resource, sys
        import numpy, scipy.stats
        from qaudit.checks.leakage import _exact_spearman_null
        base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _exact_spearman_null(10)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if sys.platform == "darwin" else 1024   # ru_maxrss units
        print((peak - base) * scale)
    """)
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                         text=True, check=True,
                         env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    delta = int(out.stdout.strip())
    assert delta < 250 * 1024 * 1024, \
        f"exact-null build used {delta / 1e6:.0f}MB over baseline " \
        f"(limit: 250 MiB above the imported-module baseline)"


def test_exact_null_tables_regression():
    # The exact-null builder must reproduce: closed-form moments,
    # the pinned n=5 lattice atoms, and the n=8 two-sided tail at the bar.
    for n in (5, 8, 10):
        vals, tail, pmf = _exact_spearman_null(n)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
        assert float((vals * pmf).sum()) == pytest.approx(0.0, abs=1e-12)
        assert float((vals ** 2 * pmf).sum()) == pytest.approx(
            1.0 / (n - 1), abs=1e-12)
    assert _spearman_tail_ge(5, 0.4) == pytest.approx(31.0 / 120.0,
                                                      abs=1e-12)
    assert 2.0 * float(_spearman_tail_ge(8, 0.40)) == pytest.approx(
        0.3268, abs=2e-4)
