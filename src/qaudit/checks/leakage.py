"""Target-leakage checks: is the prediction target sitting inside the signal?

Complements :mod:`qaudit.checks.lookahead` (timing/alignment): these checks
look at the *strength and shape* of the signal-target relation. No amount of
honest feature engineering produces a signal that ranks the forward return
almost perfectly date after date - that shape only appears when the label
window itself (the forward return, or a transform of it) leaked into the
features.

Checks
------
leakage.target_correlation   median per-date |Spearman IC| vs the forward
                             return at ``artifacts.label_horizon``
leakage.perfect_rank_dates   fraction of dates whose |IC| is near-perfect,
                             with a count bound allowing for overlapping
                             labels and persistent signal ranks
leakage.signal_target_identity  per-date Pearson corr of the cross-sectionally
                             standardized signal vs standardized target
                             (detects affine copies of the target)
leakage.ic_outlier_dates     count of dates with |IC| >= a moderate bar
                             (0.40) against the de-factored forward return
                             (market beta plus any significant principal
                             components of the target panel), vs the exact
                             breadth- and tie-aware discrete Spearman null
                             widened to the panel's own robust IC
                             dispersion, with the count bound inflated for
                             overlapping-label dependence - catches
                             *intermittent/diluted* leaks (the target
                             leaking on a subset of dates) that median-based
                             statistics are blind to by construction
"""
from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import poisson, rankdata
from scipy.stats import t as t_dist

from .._stats import MIN_NAMES as _MIN_NAMES
from .._stats import cross_sectional_ic, forward_returns
from ..config import AuditConfig
from ..inputs import BacktestArtifacts
from ..types import CheckResult, Severity, failed, passed, skipped, warned

# Tail probability for clustered perfect-rank counts. Independent labels
# use a four-sigma bound; overlapping labels use a scaled Poisson tail.
_PERFECT_COUNT_ALPHA = 1e-4

# Minimum usable IC dates. The shared _MIN_NAMES controls breadth.
_MIN_USABLE_DATES = 30

# The median IC must clear both absolute bars and these multiples of the
# breadth-implied null median to avoid treating narrow-panel noise as leakage.
_NULL_MEDIAN_WARN_MULT = 2.0

_NULL_MEDIAN_FAIL_MULT = 3.0

# Enumerate the discrete Spearman permutation null through this breadth;
# use a t approximation beyond it. A normal approximation misses discrete
# tail mass on narrow panels and can understate the expected outlier count.
# At the 0.40 bar the exact discrete tail is 8-22% fatter than a normal
# approximation for n <= 10 (0.327 vs 0.290 at n=8), an under-count that
# grows linearly in n_dates while the count bound's slack grows as sqrt.
# The t approximation used beyond n=10 is within ~4% of a Monte Carlo
# null for n=11..30 where the tail is material (x <= 0.45) and ~10% low
# only in the deep tail, where the contribution to the expected count is
# negligible.
_OUTLIER_EXACT_MAX_N = 10

# Poisson upper-tail probability. Under independent dates its variance
# dominates the corresponding Poisson-binomial variance. Overlapping labels
# receive a separate signal-persistence adjustment. The slack also absorbs
# the ~4% t-approximation error and mild residual overdispersion from
# fat-tailed factors (false-positive rate < 0.5% on noise/honest panels).
_OUTLIER_COUNT_ALPHA = 1e-4

# Minimum excess-date fraction required for actionable outlier evidence.
_OUTLIER_MIN_DATE_FRAC = 0.005

# Cap dispersion widening so a broad leak cannot excuse its own tail.
_OUTLIER_MAX_SCALE = 3.0

# Minimum joint observations for estimating market beta; otherwise use 1.
_BETA_MIN_OBS = 20

# Remove beta-residual principal components only above this multiple of
# the Marchenko-Pastur noise edge. Common factors can otherwise make honest
# style/sector signals look leaky. Removing factors also reduces sensitivity
# to leaks aligned with them; see _defactored_with_pcs for the limitations.
_PC_EDGE_MULT = 1.15

# Minimum history for the PC stage; shorter panels use [1, beta] only:
# loadings fit on fewer dates are overfit, so the PC stage disarms.
_PC_MIN_DATES = 60

# Preserve at least this many cross-sectional residual degrees of freedom:
# per date k_t = min(k_sig, n_t - 2 - 3), so at the 5-name floor no PC is
# removed and thin dates drop trailing PCs first.
_PC_MIN_RESIDUAL_DF = 3

# Resolution of the truncation-consistent dispersion-scale solve.
_SCALE_GRID_STEP = 0.005

# Chunk exact-null permutations to bound transient memory.
_EXACT_NULL_CHUNK = 200_000

# Seeded permutation draws for tied nulls past the exact breadth: at
# tail p = 1e-3 (the smallest per-date rate that materially moves a
# 500-date expected count) the MC standard error is ~7% relative,
# inside the _OUTLIER_COUNT_ALPHA slack that also absorbs the ~4%
# t-approximation error; rarer tails contribute negligibly to the
# count bound.
_TIE_MC_DRAWS = 200_000

# Draws per chunk of the tied-null permutation calculation.
_TIE_MC_CHUNK = 50_000

CHECK_TARGET_CORR = "leakage.target_correlation"
CHECK_PERFECT = "leakage.perfect_rank_dates"
CHECK_IDENTITY = "leakage.signal_target_identity"
CHECK_OUTLIER = "leakage.ic_outlier_dates"
ALL_CHECKS = (CHECK_TARGET_CORR, CHECK_PERFECT, CHECK_IDENTITY, CHECK_OUTLIER)

_UNDERPOWER_REMEDIATION = (
    "Extend the overlapping signals/asset_returns history to at least "
    f"{_MIN_USABLE_DATES} usable dates (each date needs >= {_MIN_NAMES} "
    "jointly non-NaN names) before trusting leakage diagnostics."
)


def _breadth_bound_skip(check: str, statistic: str,
                        max_joint: int) -> CheckResult:
    """Breadth-bound (not history-bound) underpower: no date ever reaches
    _MIN_NAMES jointly non-NaN names, so the per-date cross-sectional IC
    exists on zero dates at any history length - the extend-your-history
    WARN remediation is unattainable for a <5-name (pairs/spread) book and
    would repeat on every report forever. Same contract as
    _perfect_rank_dates: an unmeasurable check says so and SKIPs. Unlike
    perfect_rank there is no config escape to offer - _MIN_NAMES is fixed
    by the check design (a Spearman IC on <5 names is noise), so the
    message must not suggest lowering it."""
    return skipped(
        check,
        f"no date has >= {_MIN_NAMES} jointly non-NaN names (max joint "
        f"breadth {max_joint}), so {statistic} is unmeasurable at this "
        f"breadth no matter how long the history - cross-sectional leakage "
        f"statistics need >= {_MIN_NAMES} names per date (fixed by the "
        f"check design, not configurable). For a pairs/spread or other "
        f"<{_MIN_NAMES}-name book, leakage evidence must come from the "
        f"timing checks (lookahead.*, dynamic.*) instead",
        max_joint_names=max_joint, min_names=_MIN_NAMES)


def run(artifacts: BacktestArtifacts, config: AuditConfig, *,
        signal_func=None, backtest_func=None) -> list[CheckResult]:
    """Run all target-leakage checks. Requires only the mandatory artifacts
    (``signals`` and ``asset_returns``); callables are not used here."""
    if artifacts.signals is None:
        return [skipped(c, "pass artifacts.signals (dates x assets DataFrame "
                           "of signal values known at the end of each period) "
                           "to enable this check")
                for c in ALL_CHECKS]
    if artifacts.asset_returns is None:
        return [skipped(c, "pass artifacts.asset_returns (dates x assets "
                           "DataFrame of simple per-period returns) to enable "
                           "this check")
                for c in ALL_CHECKS]

    horizon = int(artifacts.label_horizon)
    fwd = forward_returns(artifacts.asset_returns, horizon)
    ic = cross_sectional_ic(artifacts.signals, fwd, method="spearman",
                            min_names=_MIN_NAMES).dropna()
    joint_names = (artifacts.signals.notna() & fwd.notna()).sum(axis=1)
    # Per-date tie signatures vs the raw forward return: which exact null a
    # date is judged against depends on its observed rank multisets (a tied
    # signal row - one-hot exports, bucketed scores - has a collapsed rho
    # support the distinct-rank tables misstate). ic_outlier_dates builds
    # its own profile against the de-factored target.
    profile_raw = _tie_profile(artifacts.signals, fwd)

    return [
        _target_correlation(ic, joint_names, horizon, config),
        _perfect_rank_dates(ic, joint_names, config, profile_raw,
                            signals=artifacts.signals, horizon=horizon),
        _signal_target_identity(artifacts.signals, fwd, horizon, config),
        _ic_outlier_dates(artifacts.signals, fwd, ic, config, horizon),
    ]


# ---------------------------------------------------------------------------
# 1. leakage.target_correlation
# ---------------------------------------------------------------------------

def _target_correlation(ic: pd.Series, joint_names: pd.Series, horizon: int,
                        config: AuditConfig) -> CheckResult:
    check = CHECK_TARGET_CORR
    n_dates = int(len(ic))
    # Breadth-bound before history-bound: a book whose best date never
    # reaches _MIN_NAMES names has n_dates == 0 at any history length, and
    # the extend-history WARN below would be permanently un-clearable.
    max_joint = int(joint_names.max()) if len(joint_names) else 0
    if max_joint < _MIN_NAMES:
        return _breadth_bound_skip(check, "the median per-date |IC|",
                                   max_joint)
    if n_dates < _MIN_USABLE_DATES:
        return warned(
            check,
            f"only {n_dates} usable IC dates (< {_MIN_USABLE_DATES}) - too "
            f"few to judge target leakage from the median |IC|",
            severity=Severity.LOW,
            remediation=_UNDERPOWER_REMEDIATION,
            n_dates=n_dates,
        )
    median_abs_ic = float(ic.abs().median())
    mean_ic = float(ic.mean())
    # Breadth-aware floor: with n names, pure noise already has median
    # |spearman| ~ 0.6745/sqrt(n-1) (0.26 at 8 names, 0.13 at 30). Fixed
    # thresholds below that floor would cry wolf on honest narrow books.
    n_med = float(joint_names.reindex(ic.index).clip(lower=2).median())
    null_median = 0.6745 / np.sqrt(max(n_med - 1.0, 1.0))
    warn_bar = max(config.leak_median_ic_warn,
                   _NULL_MEDIAN_WARN_MULT * null_median)
    fail_bar = max(config.leak_median_ic_fail,
                   _NULL_MEDIAN_FAIL_MULT * null_median)
    details = dict(median_abs_ic=median_abs_ic, mean_ic=mean_ic,
                   n_dates=n_dates, median_names=n_med,
                   null_median=round(null_median, 3),
                   warn_bar=round(warn_bar, 3), fail_bar=round(fail_bar, 3))
    core = (f"median |per-date Spearman IC| of signals vs {horizon}-period "
            f"forward return = {median_abs_ic:.3f} over {n_dates} dates "
            f"(noise floor at {n_med:.0f} names ~{null_median:.2f})")
    remediation = (
        "Rebuild every feature for date t from data timestamped <= t; drop "
        "or re-lag any feature computed over the label window (returns after "
        "t), then regenerate signals and re-run the backtest."
    )
    if median_abs_ic >= fail_bar:
        return failed(
            check,
            core + "; genuine equity alphas sit near the noise floor "
                   "- the target is leaking into the signal",
            severity=Severity.CRITICAL, remediation=remediation, details=details)
    if median_abs_ic >= warn_bar:
        return warned(
            check,
            core + "; genuine equity alphas sit near the noise floor "
                   "- suspiciously strong, investigate the feature pipeline",
            severity=Severity.HIGH, remediation=remediation, details=details)
    return passed(
        check,
        core + f" - below the {warn_bar:.2f} warn bar "
               f"(mean IC {mean_ic:+.3f})",
        details=details)


# ---------------------------------------------------------------------------
# 2. leakage.perfect_rank_dates
# ---------------------------------------------------------------------------

def _perfect_rank_dates(ic: pd.Series, joint_names: pd.Series,
                        config: AuditConfig,
                        tie_profile: dict[object, tuple[int, _TieKey,
                                                        _TieKey]],
                        *, signals: pd.DataFrame, horizon: int,
                        ) -> CheckResult:
    check = CHECK_PERFECT
    # A near-perfect Spearman on a handful of names is noise, not leakage:
    # P(|rho| >= 0.9) ~ 1.7% with 5 names but ~0.4% with 8. Only dates with
    # enough breadth can testify here.
    min_names = max(int(config.leak_perfect_min_names), _MIN_NAMES)
    # Breadth-bound vs history-bound underpower are different verdicts. A
    # book whose max joint breadth never reaches min_names can never clear
    # the gate - extending history (the WARN remediation below) cannot help,
    # so a permanent WARN would be un-clearable. An unmeasurable check says
    # so instead of warning forever (the lookahead.py SKIP convention).
    # joint_names is per-date over the raw panel, not the column count: a
    # sparse wide panel whose best dates do reach min_names is history-bound
    # and keeps the WARN path.
    max_joint = int(joint_names.max()) if len(joint_names) else 0
    if max_joint < min_names:
        return skipped(
            check,
            f"no date has >= {min_names} jointly observed names (max joint "
            f"breadth {max_joint}) - near-perfect ranks on so few names are "
            f"noise, so the perfect-rank fraction is unmeasurable at this "
            f"breadth no matter how long the history; for a deliberately "
            f"narrow book lower config.leak_perfect_min_names (floor "
            f"{_MIN_NAMES})",
            min_names=min_names, max_joint_names=max_joint)
    wide = ic[joint_names.reindex(ic.index).fillna(0) >= min_names]
    n_dates = int(len(wide))
    if n_dates < _MIN_USABLE_DATES:
        return warned(
            check,
            f"only {n_dates} dates have >= {min_names} jointly observed names "
            f"(< {_MIN_USABLE_DATES} needed) - the panel is too sparse to "
            f"judge the perfect-rank date fraction",
            severity=Severity.LOW,
            remediation=_UNDERPOWER_REMEDIATION,
            n_dates=n_dates, min_names=min_names,
        )
    perfect = wide[wide.abs() >= config.leak_perfect_ic]
    n_perfect = int(len(perfect))
    frac = n_perfect / n_dates
    example_dates = [ts.date().isoformat() for ts in perfect.index[:3]]
    # Poisson guard: even past the breadth gate, noise produces the odd
    # near-perfect date (rate depends on breadth). Fire only when the count
    # clears the null expectation by ~4 sigma as well as the fraction bar.
    # The per-date rate is the null tail at the configured bar from the
    # exact/t machinery (tie-aware - see _tied_two_sided_p): a fixed
    # 0.90-bar table would convict pure noise the moment a user lowered
    # leak_perfect_ic (a legitimate sensitivity probe), and clipping the
    # rate to 0.0 past breadth 13 would zero the wide-book budget. Rates
    # are a function of the panel's breadth/tie structure alone - never of
    # the IC values - so the accused dates cannot shrink their own
    # expectation.
    bar = float(config.leak_perfect_ic)
    p = np.zeros(len(wide))
    fallback_n = joint_names.reindex(wide.index).fillna(0).clip(lower=2)
    groups: dict[tuple[int, _TieKey, _TieKey], list[int]] = {}
    for i, ts in enumerate(wide.index):
        prof = tie_profile.get(ts)
        # every date with an IC has a profile (same mask/ptp logic); the
        # fallback is defensive only (plain distinct-rank read)
        key = prof if prof is not None \
            else (int(fallback_n.iloc[i]), None, None)
        groups.setdefault(key, []).append(i)
    for (n, ku, kv), idxs in groups.items():
        ii = np.asarray(idxs, dtype=int)
        if ku is None and kv is None:
            p[ii] = 2.0 * float(_spearman_tail_ge(int(n), bar))
        else:
            p[ii] = _tied_two_sided_p(
                ku if ku is not None else _distinct_key(int(n)),
                kv if kv is not None else _distinct_key(int(n)), bar, bar)
    expected_noise = float(p.sum())
    noise_bound = expected_noise + 4.0 * np.sqrt(max(expected_noise, 0.25)) + 1.0
    # Adjacent H-period labels share H-1 returns. Persistent signal ranks
    # therefore make chance perfect dates arrive in clusters. Widening only
    # the variance misses the clustered tail's skew; use the established
    # signal-only overlap factor and a scaled-Poisson upper quantile. H=1
    # keeps the exact four-sigma decision, and the clustered bound can
    # only ever widen it.
    c_factor = _overlap_cluster_factor(signals, horizon)
    if c_factor > 1.0:
        clustered_bound = float(poisson.isf(
            _PERFECT_COUNT_ALPHA, max(expected_noise, 1e-9) / c_factor)
            * c_factor)
        noise_bound = max(noise_bound, clustered_bound)
    details = dict(n_perfect=n_perfect, n_dates=n_dates,
                   min_names=min_names, example_dates=example_dates,
                   expected_noise_dates=round(expected_noise, 2),
                   noise_bound=round(noise_bound, 2),
                   cluster_factor=round(c_factor, 2),
                   label_horizon=int(horizon))
    overlap_note = (f"; overlapping-label cluster factor {c_factor:.2f} "
                    f"at label_horizon={horizon}" if c_factor > 1.0 else "")
    if frac > config.leak_perfect_date_frac and n_perfect > noise_bound:
        return failed(
            check,
            f"{n_perfect} of {n_dates} dates rank the future return almost "
            f"perfectly (|IC| >= {config.leak_perfect_ic:.2f}), e.g. "
            f"{', '.join(example_dates)} - the count exceeds the "
            f"noise bound {noise_bound:.1f}{overlap_note}; the target is "
            f"inside the signal",
            severity=Severity.CRITICAL,
            remediation=(
                "Inspect the feature pipeline for the forward return entering "
                "the signal (mis-shifted labels, joins on future dates, the "
                "target used in scaling); fix the shift and regenerate "
                "signals."),
            details=details)
    return passed(
        check,
        f"{n_perfect} of {n_dates} dates have |IC| >= "
        f"{config.leak_perfect_ic:.2f} (fail requires more than "
        f"{config.leak_perfect_date_frac:.1%} of dates and more than "
        f"~{noise_bound:.1f} dates given the noise floor at this breadth"
        f"{overlap_note})",
        details=details)


# ---------------------------------------------------------------------------
# 3. leakage.signal_target_identity
# ---------------------------------------------------------------------------

def _standardized_rowwise_corr(signals: pd.DataFrame,
                               targets: pd.DataFrame) -> pd.Series:
    """Per-date Pearson corr of the cross-sectionally standardized signal row
    vs the standardized target row, computed on jointly non-NaN cells only."""
    common_cols = signals.columns.intersection(targets.columns)
    common_idx = signals.index.intersection(targets.index)
    s = signals.loc[common_idx, common_cols].to_numpy(dtype=float)
    t = targets.loc[common_idx, common_cols].to_numpy(dtype=float)
    out = np.full(len(common_idx), np.nan)
    for i in range(len(common_idx)):
        mask = np.isfinite(s[i]) & np.isfinite(t[i])
        if mask.sum() < _MIN_NAMES:
            continue
        a, b = s[i][mask], t[i][mask]
        # Scale-safe standardization: Pearson corr is invariant to positive
        # per-row rescaling, and a raw .std() overflows to inf at |x| >~
        # 1e154 (squared deviations reach float64 max) - a RuntimeWarning
        # that is fatal under warnings-as-errors / np.seterr("raise")
        # harnesses, where one error wipes every result of the module.
        # Dividing by the row max |value| first keeps every intermediate
        # <= n, so corruption-scale-but-finite panels are measured instead
        # of raising or overflowing.
        am, bm = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
        if am == 0.0 or bm == 0.0:
            continue                    # all-zero row: no dispersion
        a, b = a / am, b / bm
        sd_a, sd_b = a.std(), b.std()
        if sd_a == 0 or sd_b == 0:
            continue
        az = (a - a.mean()) / sd_a
        bz = (b - b.mean()) / sd_b
        out[i] = float(np.mean(az * bz))
    return pd.Series(out, index=common_idx, name="identity_corr")


def _signal_target_identity(signals: pd.DataFrame, fwd: pd.DataFrame,
                            horizon: int, config: AuditConfig) -> CheckResult:
    check = CHECK_IDENTITY
    corr = _standardized_rowwise_corr(signals, fwd).dropna()
    n_dates = int(len(corr))
    # Check-own breadth basis: the rowwise corr masks jointly non-NaN cells
    # of this check's frames, so gate on that same mask (breadth-bound
    # books must SKIP, not WARN forever; see _breadth_bound_skip).
    cols = signals.columns.intersection(fwd.columns)
    idx = signals.index.intersection(fwd.index)
    joint = (signals.loc[idx, cols].notna()
             & fwd.loc[idx, cols].notna()).sum(axis=1)
    max_joint = int(joint.max()) if len(joint) else 0
    if max_joint < _MIN_NAMES:
        return _breadth_bound_skip(
            check, "the per-date signal/target identity correlation",
            max_joint)
    if n_dates < _MIN_USABLE_DATES:
        return warned(
            check,
            f"only {n_dates} usable dates (< {_MIN_USABLE_DATES}) - too few "
            f"to judge signal/target identity",
            severity=Severity.LOW,
            remediation=_UNDERPOWER_REMEDIATION,
            n_dates=n_dates,
        )
    median_corr = float(corr.median())
    # Two-sided on |corr|: an affine map with negative scale is still an
    # affine copy of the target (a sign-flipped leak / inverted long-short
    # convention), which a one-sided gate would PASS at median_corr
    # -0.999. Honest per-date Pearson medians sit at |.| <= 0.07, nowhere
    # near the 0.95 bar, so the abs() gate adds no false-positive surface.
    frac_dates_above = float((corr.abs() >= config.leak_identity_corr).mean())
    # The signed median alone is a knife-edge whenever the leak's sign
    # varies by date: an alternating +/- affine copy with an exactly
    # balanced usable-date count (400 usable dates -> 200/200) measures a
    # signed median of ~0 and would PASS while the message itself prints
    # 100% of dates at |corr| >= 0.95 (its +/-1-date neighbours FAIL at
    # |median| 0.997). The median |corr| closes the hole without touching
    # the bar: honest books top out at 0.53 (5-name 30-date noise worst
    # case; >= 8 names 0.45; momentum panels 0.14 at 30 names, 0.32 at 8
    # names) - all below the 0.95 bar. Disclosed crossing: a static beta
    # signal on a one-factor panel (rets = beta * f_t + e, signal = beta,
    # 500 x 25) measures median |corr| 0.20 at idio/factor vol ratio 1.5
    # (equity-like), 0.41-0.47 at 0.5, 0.65-0.72 at 0.25, 0.91-0.93 at
    # 0.10 and 0.96-0.99 at <= 1/15 (idio 0.002 vs factor 0.03) - there
    # the cross-section is the beta ranking every day with the factor's
    # sign, so the date-varying-sign FAIL is literally true, and
    # target_correlation + perfect_rank_dates already FAIL CRITICAL on
    # the same panel from ratio 0.25 down (this FAIL never stands alone).
    # No equity-like panel comes near the bar.
    median_abs_corr = float(corr.abs().median())
    details = dict(median_corr=median_corr, median_abs_corr=median_abs_corr,
                   frac_dates_above=frac_dates_above, n_dates=n_dates)
    signed_hit = abs(median_corr) >= config.leak_identity_corr
    if signed_hit or median_abs_corr >= config.leak_identity_corr:
        if signed_hit:
            flavor = ("a sign-flipped affine transform of the target"
                      if median_corr < 0
                      else "an affine transform of the target")
        else:
            # Only the magnitude gate fired: |corr| is at the bar on the
            # median date but the sign flips often enough to cancel the
            # signed median - a per-date copy of the target whose sign is
            # itself date-dependent (an alternating or block-negated leak).
            flavor = ("a per-date affine transform of the target with "
                      "date-varying sign (a leak whose sign flips by date, "
                      "or a static loading signal on a single-factor-"
                      "dominated panel)")
        return failed(
            check,
            f"median per-date Pearson corr of the cross-sectionally "
            f"standardized signal vs the {horizon}-period forward return = "
            f"{median_corr:.3f}, median |corr| = {median_abs_corr:.3f} "
            f"({frac_dates_above:.0%} of dates with |corr| >= "
            f"{config.leak_identity_corr:.2f}) - the signal is {flavor}",
            severity=Severity.CRITICAL,
            remediation=(
                "The signal reproduces the target up to shift/scale "
                "(negative or date-varying scale included) - remove the "
                "forward return (or any transform of it) from the feature "
                "set and from any normalization step, then rebuild the "
                "signal causally."),
            details=details)
    return passed(
        check,
        f"median per-date Pearson corr of standardized signal vs "
        f"{horizon}-period forward target = {median_corr:.3f}, median "
        f"|corr| = {median_abs_corr:.3f} over {n_dates} dates "
        f"({frac_dates_above:.0%} of dates with |corr| >= "
        f"{config.leak_identity_corr:.2f})",
        details=details)


# ---------------------------------------------------------------------------
# 4. leakage.ic_outlier_dates
# ---------------------------------------------------------------------------

# Tie signature of a cross-sectional row: sorted multiset of 2x average
# ranks (ints), or None for a tie-free row (the distinct-rank fast path).
_TieKey = tuple[int, ...] | None


@lru_cache(maxsize=1)
def _all_perms(n: int) -> np.ndarray:
    """All n! permutations of 0..n-1 as an int8 array (n <= 13 by design;
    callers stay <= _OUTLIER_EXACT_MAX_N), built by vectorized insertion.
    maxsize=1: panels have near-constant breadth, so one resident table
    (36MB at n=10) serves every tied-null build of a run without holding a
    copy per breadth."""
    perms = np.zeros((1, 1), dtype=np.int8)
    for k in range(1, n):
        m = perms.shape[0]
        new = np.empty((m * (k + 1), k + 1), dtype=np.int8)
        for pos in range(k + 1):
            blk = new[pos * m:(pos + 1) * m]
            blk[:, :pos] = perms[:, :pos]
            blk[:, pos] = k
            blk[:, pos + 1:] = perms[:, pos:]
        perms = new
    return perms


@lru_cache(maxsize=16)
def _exact_spearman_null(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact permutation null of Spearman rho at breadth n (tie-free rows):
    (sorted support, tail P(rho >= support[j]), pmf). Full n! enumeration
    via a vectorized insertion build; the displacement statistic
    S = sum(perm_j - j)^2 is folded per _EXACT_NULL_CHUNK rows in int16
    (per-cell square <= (n-1)^2 = 81 and per-row S <= n(n^2-1)/3 = 330 at
    n = 10 - int16-safe through n = 13) into a bincount table, so the peak
    footprint is the int8 permutation table plus one small chunk: ~52MB
    above the numpy/scipy import baseline at n=10 (a whole-table int64
    cast would need hundreds of MB more), ~0.1s cold, then cached. The
    support is a lattice (rho = 1 - 6*S/(n(n^2-1)), S even), so tails must
    be read off the lattice, not a continuous approximation."""
    perms = _all_perms(n)
    ranks = np.arange(n, dtype=np.int16)
    s_max = n * (n * n - 1) // 3            # full-reversal displacement
    counts = np.zeros(s_max + 1, dtype=np.int64)
    for lo in range(0, perms.shape[0], _EXACT_NULL_CHUNK):
        d = perms[lo:lo + _EXACT_NULL_CHUNK].astype(np.int16) - ranks
        counts += np.bincount(np.einsum("ij,ij->i", d, d),
                              minlength=s_max + 1)
    s_vals = np.nonzero(counts)[0]
    rho = 1.0 - 6.0 * s_vals / (n * (n * n - 1.0))
    order = np.argsort(rho)                 # rho falls as S grows: reverse
    vals = rho[order]
    pmf = (counts[s_vals] / counts.sum())[order]
    tail = pmf[::-1].cumsum()[::-1]
    return vals, tail, pmf


def _distinct_key(n: int) -> tuple[int, ...]:
    """The tie-free rank multiset (2x average ranks = 2, 4, ..., 2n), for
    pairing a tie-free row with a tied one in the tied-null builders."""
    return tuple(range(2, 2 * n + 1, 2))


def _rank_multiset_key(x: np.ndarray) -> _TieKey:
    """Canonical tie signature of a row: the sorted multiset of 2x average
    ranks (integers - average ranks live on the half-integer grid), or None
    for a tie-free row. The permutation null of rho depends on a row only
    through this multiset (positions never matter under a uniform pairing),
    so it is the cache key for the tied-null tables. Note the multiset, not
    the sorted tie-group sizes: {1,3,3,3} (singleton ranked low) and
    {2,2,2,4} (singleton high) are mirror patterns with mirror-image null
    laws - collapsing them to group sizes would conflate the two tails."""
    if np.unique(x).size == x.size:
        return None
    r2 = np.round(2.0 * rankdata(x)).astype(np.int64)
    return tuple(int(v) for v in np.sort(r2))


def _tie_profile(signals: pd.DataFrame, targets: pd.DataFrame,
                 ) -> dict[object, tuple[int, _TieKey, _TieKey]]:
    """Per-date (breadth, signal-row key, target-row key) on the same
    joint-finite mask cross_sectional_ic uses; dates that produce no IC
    (breadth < _MIN_NAMES, constant row) are omitted. A function of the two
    frames' shapes/ties alone - no IC value enters - so routing dates to
    null tables through it cannot let the accused tail excuse itself."""
    common_cols = signals.columns.intersection(targets.columns)
    common_idx = signals.index.intersection(targets.index)
    s = signals.loc[common_idx, common_cols].to_numpy(dtype=float)
    t = targets.loc[common_idx, common_cols].to_numpy(dtype=float)
    prof: dict[object, tuple[int, _TieKey, _TieKey]] = {}
    for i, ts in enumerate(common_idx):
        mask = np.isfinite(s[i]) & np.isfinite(t[i])
        n = int(mask.sum())
        if n < _MIN_NAMES:
            continue
        a, b = s[i][mask], t[i][mask]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        prof[ts] = (n, _rank_multiset_key(a), _rank_multiset_key(b))
    return prof


@lru_cache(maxsize=64)
def _exact_spearman_null_tied(
        key_a: tuple[int, ...], key_b: tuple[int, ...],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact permutation null of Spearman rho - average-rank (Pearson-on-
    ranks) form, matching scipy.stats.spearmanr - for rows with ties,
    conditional on both rows' observed rank multisets. Under independence,
    conditional on the multisets, the pairing is uniform over the n!
    permutations, and rho depends on the pairing only through the integer
    score S' = sum(2u_j * 2v_perm(j)), so a chunked bincount over S'
    enumerates the null exactly. Heavy ties collapse the support (a one-hot
    row at n=8 has 8 atoms and |rho| <= 0.577): the distinct-rank lattice
    misstates such tails by construction (distinct-rank 0.327 vs true
    0.500 for P(|rho| >= 0.40) at n=8) and convicts independent one-hot
    noise. Same memory discipline as _exact_spearman_null: int8 permutation table
    plus one int16 chunk (per-cell product <= 4n^2 = 400, per-row
    S' <= 4n^3 = 4000 at n=10; int16-safe through n=13). ~0.1s per distinct
    key pair at n=10, cached; a panel cycling through more than 64 distinct
    tie patterns at breadth 6-10 pays recompute, not wrong answers."""
    u2 = np.asarray(key_a, dtype=np.int16)
    v2 = np.asarray(key_b, dtype=np.int16)
    n = int(u2.size)
    mu_u, mu_v = float(u2.mean()) / 2.0, float(v2.mean()) / 2.0
    # population sds; constant rows never reach the null machinery
    # (cross_sectional_ic and _tie_profile skip ptp == 0 rows)
    sd_u = float(np.std(u2 / 2.0))
    sd_v = float(np.std(v2 / 2.0))
    perms = _all_perms(n)
    s_max = int(np.sort(u2).astype(np.int64)
                @ np.sort(v2).astype(np.int64))   # rearrangement inequality
    counts = np.zeros(s_max + 1, dtype=np.int64)
    for lo in range(0, perms.shape[0], _EXACT_NULL_CHUNK):
        w = v2[perms[lo:lo + _EXACT_NULL_CHUNK].astype(np.intp)]
        counts += np.bincount(w @ u2, minlength=s_max + 1)
    s_vals = np.nonzero(counts)[0]
    vals = (s_vals / (4.0 * n) - mu_u * mu_v) / (sd_u * sd_v)
    pmf = counts[s_vals] / counts.sum()     # vals rise with S': sorted
    tail = pmf[::-1].cumsum()[::-1]
    return vals, tail, pmf


@lru_cache(maxsize=32)
def _mc_spearman_null_tied(key_a: tuple[int, ...],
                           key_b: tuple[int, ...]) -> np.ndarray:
    """Seeded Monte-Carlo permutation null (sorted rho draws) for tied rows
    past _OUTLIER_EXACT_MAX_N, where the t-approximation is licensed only
    tie-free (a heavily tied row keeps a collapsed, non-t-shaped support at
    any breadth). The seed derives from the key pair (blake2b), so results
    are reproducible across processes and runs; the cache bounds the
    per-run cost to one ~0.2s draw per distinct tie pattern."""
    u = np.asarray(key_a, dtype=float) / 2.0
    v = np.asarray(key_b, dtype=float) / 2.0
    n = int(u.size)
    uz = (u - u.mean()) / np.std(u)
    vz = (v - v.mean()) / np.std(v)
    digest = hashlib.blake2b(repr((key_a, key_b)).encode(),
                             digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    out = np.empty(_TIE_MC_DRAWS)
    for lo in range(0, _TIE_MC_DRAWS, _TIE_MC_CHUNK):
        m = min(_TIE_MC_CHUNK, _TIE_MC_DRAWS - lo)
        idx = rng.random((m, n)).argsort(axis=1)
        out[lo:lo + m] = vz[idx] @ uz / n
    out.sort()
    return out


def _dist_tail_ge(vals: np.ndarray, tail: np.ndarray,
                  xs: np.ndarray) -> np.ndarray:
    """P(rho >= x) read off a discrete null table (atom at x included):
    x at or below the bottom atom reads 1.0, x above the top atom 0.0."""
    j = np.searchsorted(vals, xs - 1e-12)
    return np.where(j < len(vals), tail[np.minimum(j, len(vals) - 1)], 0.0)


def _dist_tail_le(vals: np.ndarray, pmf: np.ndarray,
                  xs: np.ndarray) -> np.ndarray:
    """P(rho <= x) read off a discrete null table (atom at x included). A
    tied null need not be symmetric (both rows tied with mirror-asymmetric
    multisets), so the lower tail is read directly, never mirrored."""
    cdf = pmf.cumsum()
    j = np.searchsorted(vals, xs + 1e-12, side="right") - 1
    return np.where(j >= 0, cdf[np.maximum(j, 0)], 0.0)


def _tied_two_sided_p(key_a: tuple[int, ...], key_b: tuple[int, ...],
                      lo, hi) -> np.ndarray:
    """P(rho >= lo_i) + P(rho <= -hi_i) under the tied null of the key pair:
    exact enumeration at n <= _OUTLIER_EXACT_MAX_N, seeded MC beyond.
    ``lo``/``hi`` are the recentered upper/lower thresholds ((bar -+ c)/s);
    scalars or arrays (returns matching length)."""
    lo_arr = np.atleast_1d(np.asarray(lo, dtype=float))
    hi_arr = np.atleast_1d(np.asarray(hi, dtype=float))
    n = len(key_a)
    if n <= _OUTLIER_EXACT_MAX_N:
        vals, tail, pmf = _exact_spearman_null_tied(key_a, key_b)
        return (_dist_tail_ge(vals, tail, lo_arr)
                + _dist_tail_le(vals, pmf, -hi_arr))
    draws = _mc_spearman_null_tied(key_a, key_b)
    nd = float(draws.size)
    up = (nd - np.searchsorted(draws, lo_arr - 1e-12, side="left")) / nd
    dn = np.searchsorted(draws, -hi_arr + 1e-12, side="right") / nd
    return up + dn


def _spearman_tail_ge(n: int, x) -> float | np.ndarray:
    """P(rho >= x) under the breadth-n independence null for tie-free rows:
    exact discrete (atoms at x included) for n <= _OUTLIER_EXACT_MAX_N,
    else the standard t-approximation. Conditionally exact for any fixed
    distinct-valued rows - only signal-vs-target independence is assumed -
    which is what licenses evaluating it against the de-factored forward
    return. A row with ties has a different conditional null (the
    average-rank rho support collapses: a one-hot row at n=8 keeps
    |rho| <= 0.577 and P(|rho| >= 0.40) = 0.50 vs 0.327 distinct-rank) -
    tied dates are owned by _exact_spearman_null_tied /
    _mc_spearman_null_tied, keyed on the observed rank multisets. Accepts
    a scalar or an array of thresholds (returns matching shape)."""
    scalar = np.isscalar(x)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    n = int(n)
    if n <= _OUTLIER_EXACT_MAX_N:
        vals, tail, _ = _exact_spearman_null(n)
        out = _dist_tail_ge(vals, tail, xs)
    else:
        xc = np.clip(xs, -0.999999, 0.999999)
        out = t_dist.sf(xc * np.sqrt((n - 2.0) / (1.0 - xc * xc)), df=n - 2)
        out = np.where(xs > 1.0, 0.0, np.where(xs <= -1.0, 1.0, out))
    return float(out[0]) if scalar else out


def _null_bulk_abs_z_quantile(n: int, bar: float, q: float) -> float:
    """q-quantile of |rho|*sqrt(n-1) under the independence null conditional
    on |rho| < bar - the consistency constants for the robust dispersion
    scale, which is estimated on sub-bar dates only. Conditioning matters
    twice: (a) the unconditional null |z| median is 0.70-0.80 at n<=10 (not
    the normal 0.6745), and (b) at breadth 6 the 0.40 bar cuts ~42% of the
    null mass, so unconditional constants would misstate the sub-bar
    scale."""
    n = int(n)
    if n <= _OUTLIER_EXACT_MAX_N:
        vals, _, pmf = _exact_spearman_null(n)
        keep = np.abs(vals) < bar - 1e-12
        if not keep.any():                      # bar below the lattice gap;
            return float(np.sqrt(n - 1.0))      # unreachable with sane bars
        av, w = np.abs(vals[keep]), pmf[keep]
        order = np.argsort(av)
        cdf = w[order].cumsum() / w.sum()
        med = float(av[order][min(int(np.searchsorted(cdf, q)), len(av) - 1)])
        return max(med, 1e-3) * float(np.sqrt(n - 1.0))
    # continuous case: solve P(|rho| <= y) = q * P(|rho| < bar) via the
    # one-sided tail Q(y) = (1 - q*(1 - 2*Q(bar))) / 2, inverted through t.
    # float(): _spearman_tail_ge returns a scalar float for scalar input.
    q_y = min((1.0 - q * (1.0 - 2.0 * float(_spearman_tail_ge(n, bar)))) / 2.0,
              0.5 - 1e-9)
    tv = float(t_dist.isf(q_y, df=n - 2))
    rho_q = tv / np.sqrt(n - 2.0 + tv * tv)
    return max(rho_q, 1e-3) * float(np.sqrt(n - 1.0))


def _truncation_consistent_scale(n: int, bar: float,
                                 observed_med_z: float) -> float:
    """Match the observed sub-bar median to a scaled permutation null.

    Solve g(s) = s * m(bar/s) = observed_med_z on a bounded grid, where
    m(b) is the conditional median of |z| given |rho| < b under scale 1.
    Scaling changes which observations survive truncation, so dividing by
    the fixed m(bar) would understate dispersion. Select the first grid
    point reaching the observed median, or the cap if none does.

    The discrete null can produce small lattice irregularities. At narrow
    breadth the conditional median saturates quickly and identifies scale
    poorly; exact tail probabilities and count-bound slack remain needed.
    The cap prevents an arbitrarily wide null from absorbing a broad leak.

    g(s) increases in s (with lattice sawtooth at n <= 10) and saturates
    near bar*sqrt(n-1)/2. The naive ratio observed/m(bar) understates s
    materially: on honest factor books it fits 1.16-1.29 while
    observed/expected counts run 1.5-1.8x, and at breadth ~30 it reads
    1.19/1.31/1.41 for true scales 1.3/1.6/2.0, which this solve recovers
    within the grid step. At breadth <= ~10 the bar sits near one null
    sigma, so the sub-bar median carries little information about s (g
    spans ~5% to saturation) and the exact tails plus the Poisson slack
    carry the calibration."""
    if not np.isfinite(observed_med_z):
        return 1.0
    if observed_med_z <= _null_bulk_abs_z_quantile(n, bar, 0.5):
        return 1.0
    for s in np.arange(1.0 + _SCALE_GRID_STEP,
                       _OUTLIER_MAX_SCALE + 1e-9, _SCALE_GRID_STEP):
        if (s * _null_bulk_abs_z_quantile(n, float(bar / s), 0.5)
                >= observed_med_z):
            return float(s)
    return float(_OUTLIER_MAX_SCALE)


def _tied_bulk_abs_z_band(key_a: tuple[int, ...], key_b: tuple[int, ...],
                          bar: float, q: float) -> tuple[float, float]:
    """(lower, upper) q-quantile atoms of |rho|*sqrt(n-1) under the tied
    null conditional on |rho| < bar - the tied analogue of
    _null_bulk_abs_z_quantile. Sparse tied supports (one-hot at n=8: four
    |z| atoms) put real sample medians between adjacent atoms, so the
    scale solve needs the atom band, not one convention: with equal mass
    on |z| atoms 0.22 and 0.65 the sample median sits near 0.44, and the
    lower-atom convention alone would read that as a ~2x widening that
    does not exist (a phantom that costs only power, but a needless one).
    """
    n = len(key_a)
    if n <= _OUTLIER_EXACT_MAX_N:
        vals, _, pmf = _exact_spearman_null_tied(key_a, key_b)
    else:
        draws = _mc_spearman_null_tied(key_a, key_b)
        vals = draws
        pmf = np.full(draws.size, 1.0 / draws.size)
    keep = np.abs(vals) < bar - 1e-12
    root = float(np.sqrt(n - 1.0))
    if not keep.any():                      # bar below the support gap
        return root, root
    av, w = np.abs(vals[keep]), pmf[keep]
    order = np.argsort(av)
    av, w = av[order], w[order]
    cdf = w.cumsum() / w.sum()
    j_lo = min(int(np.searchsorted(cdf, q)), av.size - 1)
    j_hi = min(int(np.searchsorted(cdf, q, side="right")), av.size - 1)
    return (max(float(av[j_lo]), 1e-3) * root,
            max(float(av[j_hi]), 1e-3) * root)


def _truncation_consistent_scale_tied(key_a: tuple[int, ...],
                                      key_b: tuple[int, ...], bar: float,
                                      observed_med_z: float) -> float:
    """Tied-row analogue of :func:`_truncation_consistent_scale`, solved
    against the modal tie pattern of the bulk (the tie-dominated branch of
    _ic_outlier_dates - the same single-representative approximation the
    distinct path already makes with the median breadth). Same fixed point
    g(s) = s * m(bar/s) = observed over the same [1, _OUTLIER_MAX_SCALE]
    grid; the one structural difference is band acceptance at s = 1: the
    observed bulk median is consistent with the unwidened tied null
    whenever it sits at or below the null median's upper atom (see
    _tied_bulk_abs_z_band), while the grid solve keeps the lower atom so a
    genuine excess over- rather than under-widens (fail-closed both ways:
    exact tie-aware tails at s = 1 are calibrated, and widening only ever
    raises the conviction bound)."""
    if not np.isfinite(observed_med_z):
        return 1.0
    if observed_med_z <= _tied_bulk_abs_z_band(key_a, key_b, bar, 0.5)[1]:
        return 1.0
    for s in np.arange(1.0 + _SCALE_GRID_STEP,
                       _OUTLIER_MAX_SCALE + 1e-9, _SCALE_GRID_STEP):
        if (s * _tied_bulk_abs_z_band(key_a, key_b, float(bar / s), 0.5)[0]
                >= observed_med_z):
            return float(s)
    return float(_OUTLIER_MAX_SCALE)


def _overlap_cluster_factor(signals: pd.DataFrame, horizon: int) -> float:
    """Count-variance inflation factor for overlapping labels
    (label_horizon > 1) riding a persistent signal. Adjacent dates' forward
    returns share horizon-1 constituent periods; when the signal's
    cross-sectional ranks also persist, adjacent ICs are dependent and
    outlier-date exceedances arrive in clusters of up to
    min(persistence, horizon) dates - the Poisson bound (variance = mean)
    understates the count variance and convicts honest books (on monthly
    block-constant signals with t(3) returns at n=8 the count var/mean
    rises from 0.8 at H=1 to ~6 at H=21, with CRITICAL false fails;
    overlap with iid-ranked signals stays Poisson-safe at any horizon,
    var/mean ~0.5 at H=21 - the dependence needs both).

    c = 1 + 2 * sum_{k=1..H-1} ((H-k)/H) * rhoS_k, where rhoS_k is the
    average cross-sectional rank-autocorrelation of the signal at lag k:
    the Bartlett-style variance ratio with label-overlap weights and
    worst-case (perfect) conditional exceedance correlation, so it is
    deliberately conservative (c ~ 14 at monthly-block x H=21 vs measured
    var/mean ~6). Estimated from the signal frame alone - the IC or
    exceedance series may not calibrate its own null (the sub-bar bulk
    rule), and a leaked target cannot inflate its own excuse through this
    factor; same principle as performance.py's _ic_noise_context, which
    thins overlapping-label dates for suspicious_ic. Negative lag
    correlations are clipped at 0 so anti-persistent ranks never shrink
    the bound below Poisson (fail-closed), and c is capped at its
    theoretical maximum H. Returns 1.0 exactly at horizon 1, so H=1
    verdicts are unaffected by construction."""
    if horizon <= 1:
        return 1.0
    x = signals.to_numpy(dtype=float)
    n_dates = x.shape[0]
    # rank each row once over its own finite cells (that is the signal's
    # cross-sectional rank vector); per-lag correlations then run on the
    # jointly finite cells of the pre-ranked rows
    ranks = np.full_like(x, np.nan)
    for i in range(n_dates):
        m = np.isfinite(x[i])
        if int(m.sum()) >= _MIN_NAMES:
            ranks[i, m] = rankdata(x[i, m])
    c = 1.0
    for k in range(1, horizon):
        if k >= n_dates:
            break
        a, b = ranks[k:], ranks[:-k]
        m = np.isfinite(a) & np.isfinite(b)
        cnt = m.sum(axis=1).astype(float)
        rows = cnt >= _MIN_NAMES
        if not rows.any():
            continue
        aa = np.where(m, a, 0.0)[rows]
        bb = np.where(m, b, 0.0)[rows]
        mr = m[rows]
        cr = cnt[rows]
        mu_a = aa.sum(axis=1) / cr
        mu_b = bb.sum(axis=1) / cr
        da = np.where(mr, aa - mu_a[:, None], 0.0)
        db = np.where(mr, bb - mu_b[:, None], 0.0)
        va = (da * da).sum(axis=1)
        vb = (db * db).sum(axis=1)
        good = (va > 0) & (vb > 0)
        if not good.any():
            continue
        cov = (da * db).sum(axis=1)
        rho_k = float(np.mean(cov[good] / np.sqrt(va[good] * vb[good])))
        c += 2.0 * ((horizon - k) / horizon) * float(np.clip(rho_k, 0.0, 1.0))
    return float(min(c, float(horizon)))


def _defactored_with_pcs(fwd: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove common-factor components and return the number of projected PCs.

    First residualize each date on [1, beta], using each asset's full-sample
    slope against the equal-weight market (or beta=1 with insufficient
    history). This projection removes beta-loaded common components,
    including noise carried by the market proxy itself.

    Then project out principal components of the residual target panel
    whose eigenvalues exceed the Marchenko-Pastur noise edge. Require
    _PC_MIN_DATES usable dates and preserve _PC_MIN_RESIDUAL_DF residual
    degrees of freedom. Components are derived from the target alone;
    choosing signal-aligned components would project out the very leak
    being tested. The edge test, rather than an arbitrary factor count,
    determines how many significant common directions are removed.

    This controls false positives from honest factor exposures but costs
    detection power. A leak of a removed factor is invisible here, and
    weak factors below the edge rely on bounded dispersion widening.
    Signals aligned with removed factors can have unusually quiet residual
    ICs, leaving null-count allowance that masks intermittent leakage.
    On factor-dominated books, only a small idiosyncratic part of the leaked
    target survives. Raw-target and timing checks provide complementary
    evidence, without guaranteeing detection of these cases.

    Measured behaviour: the per-date projection matters because the
    equal-weight proxy carries mean-idio noise loading on beta (~10%
    tail-heavy overdispersion on wide-beta books otherwise). Without the
    PC stage honest single-factor style/sector books convict 20-85%; a
    fixed 3-PC cap still convicts honest >= 6-sector basket books 13-50%,
    while edge-adaptive removal convicts none of them. Diluted (10%) leaks
    on idio-dominated books stay detectable with up to 7 PCs removed; on
    factor-dominant books (idio share ~13-30%) a diluted leak survives
    only at residual |IC| ~0.2, under the 0.40 bar, and only full-strength
    leak dates remain visible to the raw-target checks."""
    f = fwd.to_numpy(dtype=float)
    # Corruption-scale-but-finite cells (|x| >~ 1e154, past the intake inf
    # scan) overflow the squared sums below by design: the resulting
    # inf/NaN routes to documented fallbacks (beta = 1.0 via the isfinite
    # gate on var_j, name drops via name_ok, date drops via the IC's
    # finite mask) instead of crashing. errstate declares that intent so
    # warnings-as-errors harnesses (pytest -W error, python -W error,
    # np.seterr(all="raise") in production wrappers) keep the module alive
    # rather than letting one ERROR wipe all four leakage checks. Scope is
    # this arithmetic block only - nothing else in the module is licensed
    # to overflow.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m = fwd.mean(axis=1).to_numpy(dtype=float)   # NaN on all-NaN rows
        valid = np.isfinite(f) & np.isfinite(m)[:, None]
        cnt_j = valid.sum(axis=0).astype(float)
        den_j = np.maximum(cnt_j, 1.0)
        mm = np.where(valid, m[:, None], 0.0)
        ff = np.where(valid, f, 0.0)
        mu_m = mm.sum(axis=0) / den_j
        mu_f = ff.sum(axis=0) / den_j
        dm = np.where(valid, m[:, None] - mu_m, 0.0)
        dfv = np.where(valid, f - mu_f, 0.0)
        var_j = (dm * dm).sum(axis=0) / den_j
        cov_j = (dm * dfv).sum(axis=0) / den_j
        # isfinite: an overflowed market variance (inf) must take the
        # documented beta = 1.0 fallback. inf passes `var_j > 0`, and
        # inf/inf = NaN would poison every name's beta whenever any date's
        # cross mean overflows - a handful of corrupt cells drives the
        # whole de-factored panel to NaN, a silent full-power loss for
        # ic_outlier_dates.
        ok = (cnt_j >= _BETA_MIN_OBS) & (var_j > 0) & np.isfinite(var_j)
        beta = np.where(ok, cov_j / np.where(var_j > 0, var_j, 1.0), 1.0)
        # per-date cross-sectional OLS of fwd on [1, beta] over observed
        # names
        cnt_t = np.maximum(valid.sum(axis=1), 1)[:, None].astype(float)
        my = np.where(valid, f, 0.0).sum(axis=1, keepdims=True) / cnt_t
        mx = np.where(valid, beta[None, :], 0.0).sum(axis=1,
                                                     keepdims=True) / cnt_t
        dy = np.where(valid, f - my, 0.0)
        dx = np.where(valid, beta[None, :] - mx, 0.0)
        sxx = (dx * dx).sum(axis=1, keepdims=True)
        g = np.where(sxx > 0, (dx * dy).sum(axis=1, keepdims=True)
                     / np.where(sxx > 0, sxx, 1.0), 0.0)
        r0 = dy - g * dx        # stage-1 residual; 0 at unobserved cells
        out = np.where(valid, r0, np.nan)
    pcs_used = 0
    if int(valid.any(axis=1).sum()) >= _PC_MIN_DATES:
        # Eigen-analysis in per-name standardized (correlation) space:
        # honest idio vol is heteroskedastic (1.5-2.5%/day spreads are
        # routine), so covariance eigenvalues of high-vol single names
        # clear any homoskedastic noise edge and would be "removed" as
        # factors, eating diluted-leak power. In correlation space the
        # Marchenko-Pastur bulk sits at unit scale regardless of per-name
        # vols, a genuine common factor still stands out
        # (1 + sum_j corr-loading_j^2), and a single-name axis cannot
        # exceed the bulk edge.
        # same errstate contract as stage 1: squared deviations of
        # corruption-scale residuals overflow to inf and are then dropped
        # by the finite-variance mask below - declared, not accidental.
        # np.where evaluates both branches, so the masked arithmetic
        # itself must be licensed too.
        with np.errstate(over="ignore", invalid="ignore"):
            mu_r = np.where(valid, r0, 0.0).sum(axis=0) / den_j
            var_r = (np.where(valid, r0 - mu_r, 0.0) ** 2).sum(axis=0) \
                / den_j
            sd_r = np.sqrt(var_r)
            # Names whose variance is not finite-positive are dropped from
            # the eigen analysis: cells at |r| >= ~1e154 (corruption-scale
            # data that still passes the intake inf scan) overflow the
            # squared deviation to inf, the standardized cell goes NaN,
            # and eigh on a NaN gram raises LinAlgError - one module ERROR
            # wiping all 4 leakage checks. With the mask,
            # |x0| <= sqrt(T) per cell (each deviation is bounded by
            # sd * sqrt(n_obs) once sd is finite), so the gram cannot
            # overflow; the isfinite gate below is the belt. x0's masked
            # branch still evaluates (r0 - mu_r) on the corrupt cells -
            # inside the errstate for that reason.
            name_ok = np.isfinite(sd_r) & (sd_r > 0)
            sd_safe = np.where(name_ok, sd_r, 1.0)
            x0 = np.where(valid & name_ok[None, :],
                          (r0 - mu_r) / sd_safe, 0.0)
        gram = x0.T @ x0
        k_sig = 0
        if np.isfinite(gram).all():
            evals, evecs = np.linalg.eigh(gram)      # ascending
            evals = evals[::-1]
            vecs = evecs[:, ::-1]
            pos = evals[evals > max(float(evals[0]), 0.0) * 1e-12]
            if pos.size:
                # Correlation-matrix Marchenko-Pastur bulk edge, scaled off
                # the median eigenvalue: noise eigenvalues of a T x p
                # standardized panel concentrate below roughly
                # median * (1 + sqrt(p/T))^2; only PCs clearing that by
                # _PC_EDGE_MULT count as real common factors. Every PC that
                # clears the edge is removed (no fixed cap - see the
                # constants block); pos is descending, so the count
                # indexes the contiguous top block.
                p_eff = float(pos.size)
                t_eff = float(np.mean(den_j[cnt_j > 0])) if (cnt_j > 0).any() \
                    else 1.0
                edge = (float(np.median(pos))
                        * (1.0 + np.sqrt(p_eff / max(t_eff, 1.0))) ** 2
                        * _PC_EDGE_MULT)
                k_sig = int((pos > edge).sum())
        if k_sig > 0:
            res = np.full_like(f, np.nan)
            patterns, inv = np.unique(valid, axis=0, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            for pi in range(patterns.shape[0]):
                obs = patterns[pi]
                rows = inv == pi
                n_t = int(obs.sum())
                if n_t == 0:
                    continue        # all-NaN dates stay NaN
                k_t = int(min(k_sig, max(n_t - 2 - _PC_MIN_RESIDUAL_DF, 0)))
                y = r0[np.ix_(rows, obs)]
                if k_t == 0:
                    res[np.ix_(rows, obs)] = y
                    continue
                # project the standardized cross-section (where the
                # loadings live), then map back to return space so the
                # cross-sectional ranks the IC sees stay return-space
                x_dt = np.column_stack([np.ones(n_t), vecs[obs, :k_t]])
                hat = x_dt @ np.linalg.pinv(x_dt)
                y_std = y / sd_safe[None, obs]
                res[np.ix_(rows, obs)] = (y_std - y_std @ hat.T) \
                    * sd_safe[None, obs]
                pcs_used = max(pcs_used, k_t)
            out = res
    return (pd.DataFrame(out, index=fwd.index, columns=fwd.columns),
            pcs_used)


def _ic_outlier_dates(signals: pd.DataFrame, fwd: pd.DataFrame,
                      ic_raw: pd.Series, config: AuditConfig,
                      horizon: int) -> CheckResult:
    """Test for excess moderate-|IC| dates that median-based checks can miss.

    De-factor the target, then compare each usable IC with its breadth-
    and tie-aware permutation null (approximated beyond exact enumeration).
    Center that null on a local median bounded by predictive_ic_warn.
    Estimate residual dispersion using only sub-bar observations and a
    truncation-consistent solve, so accused tail observations cannot set
    their own allowance. For overlapping labels, inflate the count bound
    using signal-rank persistence. Test the observed count against the
    scaled Poisson upper quantile of the summed per-date tail probabilities.
    Factor-removal limitations are documented in _defactored_with_pcs."""
    check = CHECK_OUTLIER
    fwd_star, n_pcs = _defactored_with_pcs(fwd)
    ic = cross_sectional_ic(signals, fwd_star, method="spearman",
                            min_names=_MIN_NAMES).dropna()
    n_dates = int(len(ic))
    # Check-own breadth basis: this check's IC runs against the de-factored
    # target, whose NaN pattern can differ marginally from the raw panel -
    # gate on the fwd_star mask for consistency with the min_names call
    # above (breadth-bound books must SKIP, not WARN forever).
    cols = signals.columns.intersection(fwd_star.columns)
    idx = signals.index.intersection(fwd_star.index)
    joint_star = (signals.loc[idx, cols].notna()
                  & fwd_star.loc[idx, cols].notna()).sum(axis=1)
    max_joint = int(joint_star.max()) if len(joint_star) else 0
    if max_joint < _MIN_NAMES:
        return _breadth_bound_skip(check, "the IC outlier-date count",
                                   max_joint)
    if n_dates < _MIN_USABLE_DATES:
        return warned(
            check,
            f"only {n_dates} usable IC dates (< {_MIN_USABLE_DATES}) - too "
            f"few to judge IC outlier counts",
            severity=Severity.LOW,
            remediation=_UNDERPOWER_REMEDIATION,
            n_dates=n_dates)
    bar = float(config.leak_outlier_ic)
    joint = (signals.notna() & fwd_star.notna()).sum(axis=1)
    names = joint.reindex(ic.index).fillna(0).clip(lower=3).astype(int)
    observed = int((ic.abs() >= bar).sum())
    frac = observed / n_dates
    # Null center: honest predictive strength is regime-varying (momentum
    # earns most of its IC in trending stretches, sharpened further once the
    # market component is de-factored out of the target), and a location
    # mixture is bulk-invisible but tail-heavy - a constant center convicts
    # honest momentum panels whose outlier dates are one-sided, clustered,
    # and just past the bar. So the center is a per-date local rolling
    # median of the IC, hard-clipped to honest strength: a leak would need
    # majority density inside a window to move the median at all, and even
    # then the clip caps its excuse at the honest per-date tail while its
    # count contribution keeps growing.
    win = max(int(config.ic_rolling_window), _MIN_USABLE_DATES)
    min_periods = max(10, win // 3)
    if min_periods > len(ic):
        # No centered window can meet the configured minimum.  This is the
        # same all-zero fallback produced by rolling(...).fillna(0), without
        # passing a potentially unbounded Python integer into pandas.
        centers = pd.Series(0.0, index=ic.index)
    else:
        centers = (ic.rolling(win, center=True, min_periods=min_periods)
                     .median()
                     .clip(-config.predictive_ic_warn,
                           config.predictive_ic_warn)
                     .fillna(0.0))
    # Dispersion from the sub-bar bulk only: dates at/past the bar are the
    # accused and may not calibrate their own null (an every-date leak would
    # otherwise widen the scale until it excused itself), measured at the
    # median of the bulk - not a shoulder quantile (q90): diluted leaks put
    # their sub-bar dates exactly at the bulk's shoulder, so a shoulder
    # estimate would buy the attacker expected-count cover that no
    # measured honest construction needs. The widening solves the 1-d
    # fixed point s * m(bar/s) = observed bulk median (dividing by the
    # fixed constant m(bar) would understate s under any genuinely widened
    # truth - see _truncation_consistent_scale). Leak-robust because the
    # accused tail is excluded entirely and the _OUTLIER_MAX_SCALE cap
    # still binds.
    # Per-date tie signatures vs the de-factored target: the exact null a
    # date is judged against depends on its observed rank multisets. Every
    # ic date has a profile entry (same mask/ptp logic as the IC); the
    # fallback is defensive (plain distinct-rank read).
    profile = _tie_profile(signals, fwd_star)
    nn = names.to_numpy()
    prof_list: list[tuple[int, _TieKey, _TieKey]] = [
        profile.get(ts, (int(nn[i]), None, None))
        for i, ts in enumerate(ic.index)]
    tied_flags = np.array([ku is not None or kv is not None
                           for _, ku, kv in prof_list])
    tied_series = pd.Series(tied_flags, index=ic.index)
    bulk = ic[ic.abs() < bar]
    # Tied dates are excluded from the widening estimator: the distinct-
    # null consistency constants are licensed on tie-free dates only, and
    # a tied date's |z| lives on a collapsed lattice that corrupts the
    # bulk median in either direction. Tie-free panels have
    # bulk_free == bulk.
    bulk_free = bulk[~tied_series.reindex(bulk.index)]
    if len(bulk_free) >= _MIN_USABLE_DATES:
        z = ((bulk_free - centers.reindex(bulk_free.index)).abs()
             * np.sqrt(names.reindex(bulk_free.index) - 1.0))
        n_med = int(names.reindex(bulk_free.index).median())
        scale = _truncation_consistent_scale(n_med, bar, float(z.median()))
    elif len(bulk) >= _MIN_USABLE_DATES:
        # tie-dominated bulk: solve against the modal tie pattern's exact
        # law instead of silently reading the distinct-rank constants -
        # the same single-representative approximation the tie-free path
        # makes with the median breadth
        z = ((bulk - centers.reindex(bulk.index)).abs()
             * np.sqrt(names.reindex(bulk.index) - 1.0))
        prof_by_ts = dict(zip(ic.index, prof_list))
        modal, _cnt = Counter(prof_by_ts[ts]
                              for ts in bulk.index).most_common(1)[0]
        n_m, ku_m, kv_m = modal
        if ku_m is None and kv_m is None:
            scale = _truncation_consistent_scale(n_m, bar, float(z.median()))
        else:
            scale = _truncation_consistent_scale_tied(
                ku_m if ku_m is not None else _distinct_key(n_m),
                kv_m if kv_m is not None else _distinct_key(n_m),
                bar, float(z.median()))
    else:   # nearly every date is past the bar - no honest bulk exists;
        centers = pd.Series(0.0, index=ic.index)    # no regime credit either:
        scale = 1.0                 # test against the raw independence null
    cn = centers.to_numpy(dtype=float)
    p = np.zeros(n_dates)
    groups: dict[tuple[int, _TieKey, _TieKey], list[int]] = {}
    for i, key in enumerate(prof_list):
        groups.setdefault(key, []).append(i)
    for (n, ku, kv), idxs in groups.items():
        ii = np.asarray(idxs, dtype=int)
        lo = (bar - cn[ii]) / scale
        hi = (bar + cn[ii]) / scale
        if ku is None and kv is None:
            # tie-free: symmetric distinct-rank null (exact <= 10, t above)
            p[ii] = (_spearman_tail_ge(int(n), lo)
                     + _spearman_tail_ge(int(n), hi))
        else:
            ku_eff = ku if ku is not None else _distinct_key(int(n))
            kv_eff = kv if kv is not None else _distinct_key(int(n))
            # Null-calibration floor: the regime-center credit may only
            # ever add budget on tied dates, never subtract it below the
            # exact independence null. On smooth nulls a small center c
            # keeps p(c) >= p(0) (deep tails are convex in the threshold),
            # but on a sparse tied lattice p(c) steps down for any c != 0
            # near an atom boundary - and the rolling-median center
            # estimate jitters off 0 on lattice ICs almost surely, so
            # without the floor, noise-driven centers deflate the one-hot
            # expected count from 0.500 to 0.375/date and under-budget
            # honest tied panels by a quarter.
            p0 = float(_tied_two_sided_p(ku_eff, kv_eff, bar / scale,
                                         bar / scale)[0])
            p[ii] = np.maximum(_tied_two_sided_p(ku_eff, kv_eff, lo, hi),
                               p0)
    expected = float(p.sum())
    # Overlapping-label dependence: at horizon > 1 the exceedances of a
    # persistent signal cluster, so the count bound uses the
    # scaled-Poisson family - counts move in cluster-sized jumps, so
    # Poisson on expected/c scaled back by c keeps both the c-fold
    # variance and the cluster-sized skew. c == 1.0 exactly at horizon 1:
    # the expression collapses to the plain Poisson bound bit-for-bit.
    c_factor = _overlap_cluster_factor(signals, horizon)
    bound = int(poisson.isf(_OUTLIER_COUNT_ALPHA,
                            max(expected, 1e-9) / c_factor) * c_factor)
    example_dates = [ts.date().isoformat()
                     for ts in ic.index[ic.abs() >= bar][:3]]
    removed = ("market beta + top " + str(n_pcs) + " target-panel PC"
               + ("s" if n_pcs != 1 else "")) if n_pcs else "market beta"
    details = dict(observed_outliers=observed,
                   expected_under_null=round(expected, 1),
                   noise_bound=bound, outlier_bar=bar, n_dates=n_dates,
                   mean_ic=round(float(ic.mean()), 4),
                   median_local_center=round(float(centers.median()), 4),
                   max_local_center=round(float(centers.abs().max()), 4),
                   dispersion_scale=round(scale, 2),
                   defactor_pcs=int(n_pcs),
                   n_tied_dates=int(tied_flags.sum()),
                   cluster_factor=round(c_factor, 2),
                   label_horizon=int(horizon),
                   # n_ prefix = count (this dict's convention); example_dates
                   # already carries actual dates, so no unbounded date list
                   n_raw_outlier_dates=int((ic_raw.abs() >= bar).sum()),
                   example_dates=example_dates)
    if observed > bound and frac >= _OUTLIER_MIN_DATE_FRAC:
        return failed(
            check,
            f"{observed} of {n_dates} dates have |IC| >= {bar:.2f} against "
            f"the de-factored forward return ({removed} removed) vs "
            f"~{expected:.1f} expected under the exact breadth- and "
            f"tie-aware null "
            f"(fires above {bound}), e.g. {', '.join(example_dates)} - the "
            f"target leaks into the signal on a subset of dates "
            f"(median-based statistics cannot see intermittent leaks)",
            severity=Severity.CRITICAL,
            remediation=(
                "Find what distinguishes the outlier dates (rebalance days, "
                "data-vendor restatements, events) - some feature is computed "
                "from the label window on exactly those days; re-lag it and "
                "regenerate signals."),
            details=details)
    return passed(
        check,
        f"{observed} of {n_dates} dates have |IC| >= {bar:.2f} against the "
        f"de-factored forward return ({removed} removed) vs ~{expected:.1f} "
        f"expected under the exact breadth- and tie-aware null at the "
        f"panel's robust IC dispersion (x{scale:.2f}) - no excess (fires "
        f"above {bound})",
        details=details)
