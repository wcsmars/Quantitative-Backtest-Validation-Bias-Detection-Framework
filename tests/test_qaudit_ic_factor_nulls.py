"""IC nulls under truncated dispersion, style factors, and sector structure.

Dispersion estimation accounts for truncation. Target-derived principal
components remove material common factors with breadth and sample guards.
Causal factor tilts and injected forward returns test false-positive and
detection boundaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import leakage
from qaudit.checks.leakage import (_defactored_with_pcs,
                                   _null_bulk_abs_z_quantile,
                                   _truncation_consistent_scale)
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import momentum_signal, simulate_market
from qaudit.types import Severity, Status

CFG = AuditConfig()
OUTLIER = "leakage.ic_outlier_dates"
PERFECT = "leakage.perfect_rank_dates"
TARGET = "leakage.target_correlation"
IDENTITY = "leakage.signal_target_identity"


def _by(results):
    return {r.check: r for r in results}


def _run(sig, rets, cfg=CFG):
    art = BacktestArtifacts(signals=sig, asset_returns=rets).aligned()
    return _by(leakage.run(art, cfg))


# Factor-panel builders.

def style_book(seed, n_assets=30, n_periods=1000, style_vol=0.004,
               style_df=5.0, idio_vol=0.02, mkt_vol=0.010):
    """Honest static-characteristic (value/quality) book: the signal is a
    per-name constant - it cannot contain a leak by construction - and the
    returns carry that characteristic as the loading of one style factor
    (that is what 'the value factor exists' means)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_periods)
    cols = [f"S{i:03d}" for i in range(n_assets)]
    char = rng.normal(size=n_assets)
    char -= char.mean()
    beta = rng.uniform(0.7, 1.3, n_assets)
    mkt = rng.normal(2e-4, mkt_vol, n_periods)
    if style_vol <= 0:
        f = np.zeros(n_periods)
    elif style_df is None:
        f = rng.normal(0.0, style_vol, n_periods)
    else:
        f = rng.standard_t(style_df, n_periods)
        f = f / np.sqrt(style_df / (style_df - 2.0)) * style_vol
    idio = rng.normal(0.0, idio_vol, (n_periods, n_assets))
    rets = pd.DataFrame(np.outer(mkt, beta) + np.outer(f, char) + idio,
                        idx, cols)
    sig = pd.DataFrame(np.tile(char, (n_periods, 1)), idx, cols)
    return sig, rets


def sector_book(seed, n_assets=30, n_periods=1000, n_sectors=2,
                sector_vol=0.008, sector_df=3.0, idio_vol=0.02,
                mkt_vol=0.010, smooth=10):
    """Honest trailing-sector-momentum book on a market whose sector-spread
    factor has Student-t shocks. The signal uses returns <= t only."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_periods)
    cols = [f"S{i:03d}" for i in range(n_assets)]
    sector_of = np.arange(n_assets) % n_sectors
    beta = rng.uniform(0.7, 1.3, n_assets)
    mkt = rng.normal(2e-4, mkt_vol, n_periods)
    fs = rng.standard_t(sector_df, (n_periods, n_sectors))
    fs = fs / np.sqrt(sector_df / (sector_df - 2.0)) * sector_vol
    idio = rng.normal(0.0, idio_vol, (n_periods, n_assets))
    rets = pd.DataFrame(np.outer(mkt, beta) + fs[:, sector_of] + idio,
                        idx, cols)
    sec = pd.DataFrame({s: rets.iloc[:, sector_of == s].mean(axis=1)
                        for s in range(n_sectors)})
    sig = pd.DataFrame(sec.rolling(smooth, min_periods=smooth).mean()
                       .to_numpy()[:, sector_of], idx, cols)
    sig = sig + 0.1 * rets.rolling(smooth, min_periods=smooth).mean()
    sig = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1), axis=0)
    return sig, rets


def with_leak(builder, seed, frac, dilution, rng_base=7000):
    """Replace a fraction of a causal factor book's dates with diluted forward returns. The
    resulting intermittent leak must remain detectable after principal-component removal
    and scale widening."""
    sig, rets = builder(seed)
    rng = np.random.default_rng(rng_base + seed)
    fwd = rets.shift(-1)
    leak = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1), axis=0)
    if dilution > 0:
        leak = leak + dilution * pd.DataFrame(
            rng.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
    pick = rng.random(len(rets)) < frac
    sig = sig.copy()
    sig[pick] = leak[pick]
    return sig, rets


def noise_panel(n_dates, n_names, seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-05", periods=n_dates)
    cols = [f"A{i:02d}" for i in range(n_names)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (n_dates, n_names)), idx, cols)
    sig = pd.DataFrame(rng.normal(size=(n_dates, n_names)), idx, cols)
    return sig, rets


# Causal style and sector books must avoid leakage convictions.

@pytest.mark.parametrize("seed", [5005, 5006, 5007, 5008])
def test_honest_style_book_value_vol_not_convicted(seed):
    # A t5 style factor at 0.4% daily volatility contributes about 3% of per-name
    # variance.
    r = _run(*style_book(seed))[OUTLIER]
    assert r.status is Status.PASS, (seed, r.message)


@pytest.mark.parametrize("seed", [5000, 5001, 5002])
def test_honest_style_book_crash_vol_not_convicted(seed):
    # A 0.6% daily style factor exercises stronger common-factor exposure.
    r = _run(*style_book(seed, style_vol=0.006))[OUTLIER]
    assert r.status is Status.PASS, (seed, r.message)


@pytest.mark.parametrize("seed", [5000, 5001, 5006])
def test_honest_gaussian_style_book_not_convicted(seed):
    # A Gaussian factor isolates scale-mixture effects from heavy-tail effects.
    r = _run(*style_book(seed, style_df=None))[OUTLIER]
    assert r.status is Status.PASS, (seed, r.message)


@pytest.mark.parametrize("seed,vol", [(5007, 0.003), (5008, 0.003),
                                      (5011, 0.003), (5037, 0.002)])
def test_honest_weak_style_factor_not_convicted(seed, vol):
    # Weak factor eigenvalues near the Marchenko-Pastur bulk edge exercise the factor-
    # identification boundary.
    r = _run(*style_book(seed, style_vol=vol))[OUTLIER]
    assert r.status is Status.PASS, (seed, vol, r.message)


@pytest.mark.parametrize("seed", [5001, 5004, 5005])
def test_honest_sector_book_t3_not_convicted(seed):
    # Two-sector trailing momentum with a t3 spread at 0.8% daily volatility.
    r = _run(*sector_book(seed))[OUTLIER]
    assert r.status is Status.PASS, (seed, r.message)


@pytest.mark.parametrize("seed,kw", [
    (5000, dict(sector_vol=0.012)), (5004, dict(sector_vol=0.012)),
    (5002, dict(sector_df=1e6)), (5005, dict(sector_df=1e6)),
    (5008, dict(n_sectors=6)), (5017, dict(n_sectors=6)),
])
def test_honest_sector_book_variants_not_convicted(seed, kw):
    # High-volatility, Gaussian, and six-sector variants.
    r = _run(*sector_book(seed, **kw))[OUTLIER]
    assert r.status is Status.PASS, (seed, kw, r.message)


@pytest.mark.parametrize("seed", [5032, 5034])
def test_honest_narrow10_sector_book_not_convicted(seed):
    # Ten names in two five-name sectors exercise the exact-null breadth regime.
    r = _run(*sector_book(seed, n_assets=10, sector_vol=0.006,
                          sector_df=4.0))[OUTLIER]
    assert r.status is Status.PASS, (seed, r.message)


@pytest.mark.parametrize("seed", [5000, 5001, 5002, 5003])
def test_zero_factor_control_still_clean(seed):
    # Constant signals with no style factor provide the baseline null control.
    r = _run(*style_book(seed, style_vol=0.0))[OUTLIER]
    assert r.status is Status.PASS, (seed, r.message)


def test_honest_style_book_full_family_clean():
    # The complete leakage family must remain free of failures on this causal style
    # book.
    res = _run(*style_book(5005))
    for name, r in res.items():
        assert r.status is not Status.FAIL, (name, r.message)


# Leak detection on factor panels.

@pytest.mark.parametrize("seed", [5000, 5002, 5005])
def test_ten_pct_diluted_leak_on_style_panel_caught(seed):
    # A 10%-of-dates diluted leak on a factor-aligned constant signal must be detected.
    # Lower injection fractions can fall below the count budget, an identification
    # limit.
    r = _run(*with_leak(style_book, seed, 0.10, 1.4))[OUTLIER]
    assert r.status is Status.FAIL, (seed, r.message)
    assert r.severity is Severity.CRITICAL


@pytest.mark.parametrize("builder,seed", [(style_book, 5000),
                                          (style_book, 5001),
                                          (sector_book, 5000),
                                          (sector_book, 5001)])
def test_smeared_leak_on_factor_panel_caught(builder, seed):
    # Diluted forward returns injected on 35% of dates must not widen their own null
    # through quiet sub-threshold factor residuals.
    r = _run(*with_leak(builder, seed, 0.35, 1.9))[OUTLIER]
    assert r.status is Status.FAIL, (seed, r.message)
    assert r.details["dispersion_scale"] <= 1.5, r.details


@pytest.mark.parametrize("seed", [5000, 5001])
def test_full_strength_sparse_leak_on_style_panel_family(seed):
    # Full-strength forward-return dates retain detection through the perfect-rank
    # family.
    res = _run(*with_leak(style_book, seed, 0.02, 0.0))
    assert res[PERFECT].status is Status.FAIL, res[PERFECT].message


# ---------------------------------------------------------------------------
# de-factoring behavior contracts
# ---------------------------------------------------------------------------

def test_pc_stage_removes_the_style_factor_and_keeps_idio():
    # The residual must be near-orthogonal to the characteristic (the
    # factor loading) date by date, while preserving the idio component.
    sig, rets = style_book(5005)
    fwd = rets.shift(-1)
    star, n_pcs = _defactored_with_pcs(fwd)
    assert n_pcs >= 1
    char = sig.iloc[0]
    char_ic = star.iloc[:-1].apply(
        lambda row: pd.Series(row).corr(char, method="spearman"), axis=1)
    assert char_ic.abs().median() < 0.20
    keep = fwd.iloc[100].notna()
    same_day = fwd.iloc[100][keep].corr(star.iloc[100][keep],
                                        method="spearman")
    assert same_day > 0.7


def test_pc_stage_silent_on_noise_and_on_heteroskedastic_idio():
    # Eigen gate: pure noise removes nothing (sample PCs are not factors),
    # and heteroskedastic per-name vols (1.5-2.5%/day, simulate_market)
    # must not fake a factor: a spurious PC would eat the diluted-leak
    # detection power, which is why the eigen gate works in correlation
    # space.
    _, rets = noise_panel(1000, 30, 0)
    assert _defactored_with_pcs(rets.shift(-1))[1] == 0
    m = simulate_market(seed=0)
    assert _defactored_with_pcs(m["returns"].shift(-1))[1] == 0


def test_pc_stage_disarms_below_min_dates():
    # Fewer than _PC_MIN_DATES usable dates cannot support fitted principal-component
    # loadings; disable that stage.
    sig, rets = style_book(5005, n_periods=50)
    star, n_pcs = _defactored_with_pcs(rets.shift(-1))
    assert n_pcs == 0


def test_details_and_message_report_the_defactoring():
    r = _run(*style_book(5005))[OUTLIER]
    assert r.details["defactor_pcs"] >= 1
    assert "target-panel PC" in r.message and "de-factored" in r.message
    n = _run(*noise_panel(1000, 8, 0))[OUTLIER]
    assert n.details["defactor_pcs"] == 0
    assert "market beta removed" in n.message
    for k, v in {**r.details, **n.details}.items():
        assert isinstance(v, (int, float, str, list)), (k, type(v))


def test_outlier_check_deterministic_with_pc_stage():
    r1 = _run(*style_book(5005))[OUTLIER]
    r2 = _run(*style_book(5005))[OUTLIER]
    assert (r1.status, r1.message) == (r2.status, r2.message)
    assert r1.details == r2.details


# ---------------------------------------------------------------------------
# Truncation-consistent scale solver
# ---------------------------------------------------------------------------

def test_scale_solver_floor_cap_and_monotone():
    g1 = _null_bulk_abs_z_quantile(30, 0.40, 0.5)
    assert _truncation_consistent_scale(30, 0.40, g1 * 0.9) == 1.0
    assert _truncation_consistent_scale(30, 0.40, g1) == 1.0
    assert _truncation_consistent_scale(30, 0.40, 50.0) == \
        leakage._OUTLIER_MAX_SCALE
    assert _truncation_consistent_scale(30, 0.40, float("nan")) == 1.0
    prev = 0.0
    for obs in (0.70, 0.75, 0.80, 0.85, 0.90):
        s = _truncation_consistent_scale(30, 0.40, obs)
        assert s >= prev, (obs, s, prev)
        prev = s


def test_scale_solver_recovers_true_widening_unlike_naive_ratio():
    # Scale and truncate an exact Spearman null, then recover the scale from its
    # conditional median. Dividing by a fixed unscaled median underestimates widened
    # distributions.
    rng = np.random.default_rng(17)
    n, s_true, bar = 30, 1.6, 0.40
    r = rng.random((200_000, n)).argsort(axis=1).argsort(axis=1)
    i = np.arange(n)
    rho = (1.0 - 6.0 * ((r - i) ** 2).sum(axis=1)
           / (n * (n * n - 1.0))) * s_true
    obs = float(np.median(np.abs(rho[np.abs(rho) < bar]))) * np.sqrt(n - 1.0)
    fixed = _truncation_consistent_scale(n, bar, obs)
    naive = obs / _null_bulk_abs_z_quantile(n, bar, 0.5)
    assert 1.45 <= fixed <= 1.75, fixed
    assert naive < fixed - 0.15, (naive, fixed)


def test_scale_solver_low_breadth_saturation_semantics():
    # At breadth 8 the bar sits at ~1 sigma of the null: the sub-bar
    # median barely identifies s (the g sawtooth spans only ~0.45-0.60),
    # so the solver returns the floor below g(1) and a strongly
    # conservative widening (>= 2, up to the cap) once the observed median
    # exceeds the s=1 plateau - FP-protective in the only direction that
    # matters at breadths where the count test is weak anyway.
    assert _truncation_consistent_scale(8, 0.40, 0.44) == 1.0
    s = _truncation_consistent_scale(8, 0.40, 0.60)
    assert 2.0 <= s <= leakage._OUTLIER_MAX_SCALE, s
    assert _truncation_consistent_scale(8, 0.40, 0.75) == \
        leakage._OUTLIER_MAX_SCALE


# ---------------------------------------------------------------------------
# target_correlation breadth-aware noise-floor multipliers
# ---------------------------------------------------------------------------

def _tc(median_val, breadth, n=200):
    """Deterministic synthetic IC series with |IC| == median_val on every
    date at the given breadth, fed straight to _target_correlation."""
    idx = pd.bdate_range("2015-01-05", periods=n)
    ic = pd.Series([median_val if i % 2 == 0 else -median_val
                    for i in range(n)], index=idx)
    joint = pd.Series(breadth, index=idx)
    return leakage._target_correlation(ic, joint, 1, CFG)


def test_target_correlation_multiplier_boundaries_breadth6():
    # Breadth-6 null median |IC| = 0.6745/sqrt(5) = 0.3016, so warn bar =
    # 2.0 * 0.3016 = 0.603 and fail bar = 3.0 * 0.3016 = 0.905. The three
    # medians bracket both multipliers from both sides: with the mutated
    # 1.0/1.0 floors 0.45 flips to WARN and 0.65 flips to FAIL.
    r = _tc(0.45, 6)
    assert r.status is Status.PASS, r.message
    r = _tc(0.65, 6)
    assert r.status is Status.WARN and r.severity is Severity.HIGH, r.message
    r = _tc(0.95, 6)
    assert r.status is Status.FAIL and r.severity is Severity.CRITICAL, \
        r.message


def test_target_correlation_absolute_floor_boundary_breadth30():
    # At breadth 30 the multiplier term (2*0.127=0.25) sits below the
    # absolute config floor 0.30, which must still govern.
    assert _tc(0.28, 30).status is Status.PASS
    r = _tc(0.32, 30)
    assert r.status is Status.WARN and r.severity is Severity.HIGH


def test_target_correlation_multiplier_is_load_bearing(monkeypatch):
    # Direct mutation evidence: with the warn multiplier weakened to 1.0
    # the 0.45-median honest narrow book flips to a HIGH WARN - the
    # constant, not the absolute floor, is what protects breadth 5-7.
    monkeypatch.setattr(leakage, "_NULL_MEDIAN_WARN_MULT", 1.0)
    assert _tc(0.45, 6).status is Status.WARN


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noise_breadth7_family_clean(seed):
    # At breadth seven, the absolute floor alone cannot protect the null; the breadth-
    # aware multipliers must remain effective.
    res = _run(*noise_panel(1250, 7, seed))
    for name in (OUTLIER, TARGET, IDENTITY):
        assert res[name].status is Status.PASS, (seed, name,
                                                 res[name].message)
    # Breadth seven is below leak_perfect_min_names; more history cannot resolve that
    # structural limitation, so skip.
    perf = res[PERFECT]
    assert perf.status is Status.SKIP, (seed, perf.message)


def test_honest_narrow_book_target_correlation_passes():
    # Honest-guard for the breadth-aware floor: a genuine weak-momentum
    # 6-name book over 10 years (median |IC| ~0.37, i.e. above the naked
    # breadth-6 null median 0.30) must pass target_correlation - only the
    # 2x multiplier keeps the warn bar above it.
    m = simulate_market(n_assets=6, n_periods=2520, seed=0, death_frac=0.0)
    res = _run(momentum_signal(m["returns"]), m["returns"])
    r = res[TARGET]
    assert r.status is Status.PASS, r.message
    assert r.details["median_abs_ic"] > r.details["null_median"]
