"""Static checks for information used before its declared availability.

Signals at t are known at the end of t. With signal_lag=L, they may first
influence positions held during t+L; positions at t earn returns at t.

The eight detectors examine complementary channels:

* position_signal_alignment compares newer signal lags with the declared-lag
  baseline. Both sides need sufficient dates and jointly valid cells.
* embedded_future_return and ic_decay_signature flag unusually large forward
  IC or a concentration of predictive information in the next bar.
* deferred_ic_spike profiles signals that predict later bars while skipping
  the next bar; smeared_forward_ic integrates innovation IC over horizons.
* same_bar_bleed residualizes positions and the same-bar signal against the
  declared timing basis, then tests their remaining dependence. Discretized
  exports require higher thresholds or an inconclusive result because coarse
  signal ranks omit information available to the position-sizing process.
* future_vol_sizing tests position magnitudes against future volatility after
  controlling for strictly trailing, returns-derived risk estimates.
* same_bar_return_loading tests raw position residuals against same-bar
  return ranks, covering magnitude changes that preserve position ranks.

These are statistical diagnostics with explicit limits. Rolling signals can
retain strictly past information in their residuals, so same_bar_bleed uses
additional past-data projections and scopes ambiguous cases to WARN. At
lags of two or more it isolates bar-t use; small partial uses of intervening
signals can remain unidentifiable. Return-orthogonal signal innovations
retain the declared signal-stamp contract. Intermittent or heavily diluted
leaks can fall below the diagnostic thresholds.

HARD LIMITATION: the volatility control grid cannot represent risk information absent
from close-to-close returns, such as implied volatility, OHLC ranges or
intraday realized volatility. Honest sizing on those inputs can resemble
future-volatility sizing and requires a direct audit of the sizing series.
Uniform scaling cancels in cross-sectional ranks. A large contemporaneous
signal/return correlation alone is not evidence of lookahead: trailing
signals may legitimately contain the current return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .._stats import (cross_sectional_ic, fmt_tstat, forward_returns,
                      newey_west_se as _nw_se, newey_west_tstat)
from ..config import AuditConfig
from ..inputs import BacktestArtifacts
from ..types import CheckResult, Severity, failed, passed, skipped, warned

#: Module constant. A signal lagged less than the declared ``signal_lag``
#: must beat the correlation at the declared lag by more than this margin
#: before we call same-bar execution. Absorbs rank-correlation estimation
#: noise on persistent signals (day-over-day signal autocorrelation is
#: typically 0.95+, so neighbouring lags sit within a few 0.01 of each other).
ALIGNMENT_MARGIN = 0.01

#: Minimum assets per date (nonzero position and non-NaN signal) for a
#: per-date cross-sectional correlation to be usable.
_MIN_NAMES = 5

#: Minimum depth and breadth for the declared-lag comparison. A baseline with
#: insufficient valid observations cannot support a verdict. Narrow
#: cross-sections have noisy rank correlations; the cell floor prevents a
#: few minimum-breadth dates from satisfying the depth requirement alone.
#: A per-date Spearman over m names has sd ~ 1/sqrt(m-1) (~0.5 at the
#: 5-name floor vs the 0.01 margin); >= 10 dates caps the standard error
#: of the date-mean at ~sd/3, and 100 cells = 2 x _MIN_NAMES x 10 dates
#: keeps a run of minimum-breadth dates from qualifying alone.
ALIGNMENT_MIN_DATES = 10
ALIGNMENT_MIN_CELLS = 100

#: Minimum usable dates for mean-IC and horizon-profile verdicts. Absolute
#: rank correlation has positive sampling bias on narrow cross-sections,
#: so a few high-IC dates cannot establish embedded future information.
#: Per-date |Spearman| at the 5-name floor has sd ~0.5 and E|IC| ~0.4
#: under pure noise, so a mean over a handful of dates clears even the
#: 0.60 embed threshold by luck: on honest 5-name 250-day books
#: embedded_future_return false-FAILs 37%/17%/8%/2.5% of the time at
#: 1/3/5/10 usable dates and 0.2% at 30; ic_decay_signature false-WARNs
#: 8-10% below the floor vs ~2% at it. 30 also matches the
#: deferred_ic_spike floor. Below this floor the affected detector
#: reports insufficient evidence.
MIN_IC_DATES = 30


def run(artifacts: BacktestArtifacts, config: AuditConfig, *,
        signal_func=None, backtest_func=None) -> list[CheckResult]:
    """Run the lookahead family. ``signal_func``/``backtest_func`` are not
    used by these static checks (the dynamic probes handle those)."""
    return [
        _position_signal_alignment(artifacts, config),
        _same_bar_bleed(artifacts, config),
        _future_vol_sizing(artifacts, config),
        _embedded_future_return(artifacts, config),
        _ic_decay_signature(artifacts, config),
        _deferred_ic_spike(artifacts, config),
        _smeared_forward_ic(artifacts, config),
        _same_bar_return_loading(artifacts, config),
    ]


# ---------------------------------------------------------------------------
# 1. lookahead.position_signal_alignment
# ---------------------------------------------------------------------------

def _alignment_profile(positions: pd.DataFrame, signals: pd.DataFrame,
                       lags: range
                       ) -> tuple[dict[int, float], dict[int, int],
                                  dict[int, int]]:
    """mean over dates of |Spearman corr(positions.loc[t], signals.loc[t-k])|
    for each k in ``lags``. Returns ({k: mean_abs_corr}, {k: n_dates_used},
    {k: n_cells_used}); a lag with no usable dates maps to NaN. Cells count
    the jointly usable (finite nonzero position, finite lagged signal)
    entries on the dates that produced a correlation - the effective sample
    behind each mean."""
    pos = positions.to_numpy(dtype=float)
    sig = signals.to_numpy(dtype=float)
    n = pos.shape[0]
    profile: dict[int, float] = {}
    counts: dict[int, int] = {}
    cells: dict[int, int] = {}
    for k in lags:
        vals = []
        n_cells = 0
        for i in range(k, n):
            p, s = pos[i], sig[i - k]
            mask = np.isfinite(p) & (p != 0.0) & np.isfinite(s)
            if int(mask.sum()) < _MIN_NAMES:
                continue
            a, b = p[mask], s[mask]
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                continue  # degenerate cross-section
            r = sps.spearmanr(a, b).statistic
            if np.isfinite(r):
                vals.append(abs(r))
                n_cells += int(mask.sum())
        profile[k] = float(np.mean(vals)) if vals else float("nan")
        counts[k] = len(vals)
        cells[k] = n_cells
    return profile, counts, cells


def _position_signal_alignment(artifacts: BacktestArtifacts,
                               config: AuditConfig) -> CheckResult:
    check = "lookahead.position_signal_alignment"
    if artifacts.positions is None:
        return skipped(check,
                       "needs positions: pass artifacts.positions (dates x assets "
                       "DataFrame of the weights held during each period) to "
                       "enable the off-by-one execution check.")

    lag = int(artifacts.signal_lag)
    n_rows = min(len(artifacts.positions), len(artifacts.signals))
    if lag >= n_rows:
        return skipped(
            check,
            f"the declared signal_lag ({lag}) leaves zero in-sample rows "
            f"in the {n_rows}-row positions/signals overlap, so its "
            f"alignment baseline is unmeasurable. Pass more pre-history or "
            f"correct artifacts.signal_lag.",
            declared_lag=lag, n_overlap_rows=n_rows,
            min_baseline_dates=ALIGNMENT_MIN_DATES,
            min_baseline_cells=ALIGNMENT_MIN_CELLS,
        )
    lags = range(0, max(3, lag + 1) + 1)
    profile, counts, cells = _alignment_profile(artifacts.positions,
                                                artifacts.signals, lags)

    finite = {k: v for k, v in profile.items() if np.isfinite(v)}
    if not finite:
        return skipped(check,
                       "positions/signals overlap is unusable: no date has >= "
                       f"{_MIN_NAMES} assets with a nonzero position and a "
                       "non-NaN signal. Pass positions on the same asset grid "
                       "as signals to enable this check.")

    best_lag = max(finite, key=lambda k: finite[k])
    corr_at_declared = profile.get(lag, float("nan"))
    profile_str = ", ".join(
        f"k={k}: {profile[k]:.3f}" if np.isfinite(profile[k]) else f"k={k}: n/a"
        for k in lags)
    details = dict(
        alignment_profile={str(k): (float(v) if np.isfinite(v) else None)
                           for k, v in profile.items()},
        n_dates_per_lag={str(k): int(v) for k, v in counts.items()},
        n_cells_per_lag={str(k): int(v) for k, v in cells.items()},
        declared_lag=lag,
        best_lag=int(best_lag),
        margin=ALIGNMENT_MARGIN,
        min_baseline_dates=ALIGNMENT_MIN_DATES,
        min_baseline_cells=ALIGNMENT_MIN_CELLS,
    )

    # The declared-lag baseline needs sufficient jointly valid data before any
    # younger lag can support a comparison. Missing baseline evidence SKIPs.
    if counts[lag] < ALIGNMENT_MIN_DATES or cells[lag] < ALIGNMENT_MIN_CELLS:
        return skipped(
            check,
            f"alignment baseline at the declared lag {lag} is unmeasurable: "
            f"only {counts[lag]} date(s) / {cells[lag]} cells have a nonzero "
            f"position paired with a non-NaN signals.loc[t-{lag}] (need >= "
            f"{ALIGNMENT_MIN_DATES} dates and >= {ALIGNMENT_MIN_CELLS} cells; "
            f"measured profile: {profile_str}). Pass signal history that "
            f"covers the position dates at the declared lag - "
            f"signals.loc[t-{lag}] non-NaN wherever positions.loc[t] is held "
            f"- or correct artifacts.signal_lag.",
            details=details)

    # A younger lag may only overturn the baseline on comparable evidence:
    # the same floors apply, else a single 5-name fluke date at k=0 would
    # outvote a full-sample baseline (the mirror image of the unmeasurable
    # baseline case above: too little data can neither lose nor win the
    # comparison).
    illegal = {k: v for k, v in finite.items()
               if k < lag and counts[k] >= ALIGNMENT_MIN_DATES
               and cells[k] >= ALIGNMENT_MIN_CELLS}
    baseline = corr_at_declared  # finite: rests on >= ALIGNMENT_MIN_DATES dates
    if illegal and max(illegal.values()) > baseline + ALIGNMENT_MARGIN:
        k_bad = max(illegal, key=lambda k: illegal[k])
        when = "the signal of t itself" if k_bad == 0 else f"the signal of t-{k_bad}"
        return failed(
            check,
            f"positions held during t align with {when} "
            f"(corr {illegal[k_bad]:.2f} over {counts[k_bad]} dates at lag "
            f"{k_bad} vs {corr_at_declared:.2f} over {counts[lag]} dates at "
            f"declared lag {lag}) - same-bar execution: the backtest trades on "
            f"a signal it could not have had yet.",
            severity=Severity.CRITICAL,
            remediation=(
                f"Shift position construction by signal_lag: positions.loc[t] "
                f"must be built from signals.loc[t-{lag}] or earlier (e.g. "
                f"positions = weights(signals.shift({lag}))), or declare the "
                f"true lag in artifacts.signal_lag."),
            details=details)

    return passed(
        check,
        f"positions align best with the signal at lag {best_lag} "
        f"(declared lag {lag}); mean |cross-sectional Spearman| profile: "
        f"{profile_str} - no lag younger than {lag} beats the declared lag "
        f"by more than {ALIGNMENT_MARGIN}.",
        severity=Severity.CRITICAL,
        details=details)


# ---------------------------------------------------------------------------
# 2. lookahead.embedded_future_return
# ---------------------------------------------------------------------------

def _embedded_future_return(artifacts: BacktestArtifacts,
                            config: AuditConfig) -> CheckResult:
    check = "lookahead.embedded_future_return"
    horizons = [1]
    if int(artifacts.label_horizon) != 1:
        horizons.append(int(artifacts.label_horizon))

    mean_abs_ic: dict[str, float] = {}
    n_dates: dict[str, int] = {}
    for h in horizons:
        ic = cross_sectional_ic(
            artifacts.signals, forward_returns(artifacts.asset_returns, h))
        ic = ic.dropna()
        mean_abs_ic[str(h)] = float(ic.abs().mean()) if len(ic) else float("nan")
        n_dates[str(h)] = int(len(ic))

    finite = {h: v for h, v in mean_abs_ic.items() if np.isfinite(v)}
    details = dict(mean_abs_ic=mean_abs_ic, n_dates=n_dates,
                   threshold=float(config.future_return_embed_ic),
                   min_ic_dates=MIN_IC_DATES)
    if not finite:
        return skipped(check,
                       "could not compute a single per-date IC (need >= 5 "
                       "assets per date with non-NaN signal and forward "
                       "return); pass wider signals/asset_returns panels to "
                       "enable this check.")

    # Require enough dates to average the positive sampling bias of
    # absolute rank correlation on narrow cross-sections.
    finite = {h: v for h, v in finite.items()
              if n_dates[h] >= MIN_IC_DATES}
    if not finite:
        counts = ", ".join(f"h={h}: {n}" for h, n in n_dates.items())
        return skipped(check,
                       f"too few usable IC dates to separate an embedded "
                       f"return from luck ({counts}; need >= {MIN_IC_DATES} "
                       f"at some horizon) - extend the overlap of signals "
                       f"and asset_returns to enable this check.",
                       details=details)

    worst_h = max(finite, key=lambda h: finite[h])
    worst = finite[worst_h]
    if worst >= config.future_return_embed_ic:
        return failed(
            check,
            f"signal contains the next-period return: mean |per-date Spearman "
            f"IC| vs the h={worst_h} forward return = {worst:.3f} over "
            f"{n_dates[worst_h]} dates (threshold "
            f"{config.future_return_embed_ic:.2f}; mean |IC| has a "
            f"breadth-dependent noise floor even for signal-free inputs, but "
            f"only a signal embedding the return itself reaches this level).",
            severity=Severity.CRITICAL,
            remediation=(
                "Remove the label window from the feature computation: every "
                "input to signals.loc[t] must be observable by the close of "
                "period t. Look for shift(-1)-style bugs, features built from "
                "the target, or timestamps recorded at publication rather "
                "than observation."),
            details=details)

    measured = ", ".join(f"h={h}: {v:.3f} over {n_dates[h]} dates"
                         for h, v in mean_abs_ic.items())
    return passed(
        check,
        f"mean |per-date Spearman IC| vs forward returns ({measured}) stays "
        f"below the embed threshold {config.future_return_embed_ic:.2f} - "
        f"signal does not mechanically contain its future return.",
        severity=Severity.CRITICAL,
        details=details)


# ---------------------------------------------------------------------------
# 3. lookahead.ic_decay_signature
# ---------------------------------------------------------------------------

def _ic_decay_signature(artifacts: BacktestArtifacts,
                        config: AuditConfig) -> CheckResult:
    check = "lookahead.ic_decay_signature"
    fwd1 = forward_returns(artifacts.asset_returns, 1)
    ic1_series = cross_sectional_ic(artifacts.signals, fwd1).dropna()
    # the single-period return of t+2 (not the compound 2-period return)
    ic2_series = cross_sectional_ic(artifacts.signals, fwd1.shift(-1)).dropna()

    if len(ic1_series) < MIN_IC_DATES or len(ic2_series) < MIN_IC_DATES:
        # Below the floor the collapse ratio is noise-on-noise: honest sparse
        # books WARN from luck alone ~8-10% of the time below the floor vs
        # ~2% at it (the ic_decay half of the MIN_IC_DATES calibration).
        return skipped(check,
                       f"too few usable IC dates to judge the decay profile "
                       f"({len(ic1_series)} at t+1, {len(ic2_series)} at "
                       f"t+2; need >= {MIN_IC_DATES} of each) - extend the "
                       f"overlap of signals and asset_returns to enable "
                       f"this check.",
                       n_dates_ic1=int(len(ic1_series)),
                       n_dates_ic2=int(len(ic2_series)),
                       min_ic_dates=MIN_IC_DATES)

    # sort=True: keep the chronological union index pandas 2 produced
    # (pandas 3 deprecates the implicit sort of DatetimeIndex unions).
    paired = pd.concat({"ic1": ic1_series, "ic2": ic2_series}, axis=1,
                       sort=True).dropna()
    n_common = int(len(paired))
    if n_common < MIN_IC_DATES:
        return skipped(
            check,
            f"the t+1 and t+2 IC series have only {n_common} jointly "
            f"measurable calendar dates (need >= {MIN_IC_DATES}; "
            f"individually {len(ic1_series)} and {len(ic2_series)}) - a "
            f"decay ratio across different date samples is not a valid "
            f"horizon comparison. Fill/drop the gapped target dates or "
            f"extend their common overlap.",
            n_dates_ic1=int(len(ic1_series)),
            n_dates_ic2=int(len(ic2_series)), n_common_dates=n_common,
            min_ic_dates=MIN_IC_DATES,
        )

    ic1 = float(paired["ic1"].mean())
    ic2 = float(paired["ic2"].mean())
    ratio = float(abs(ic2) / abs(ic1)) if abs(ic1) > 0 else float("nan")
    details = dict(ic1=ic1, ic2=ic2, ratio=ratio,
                   n_dates_ic1=int(len(ic1_series)),
                   n_dates_ic2=int(len(ic2_series)),
                   n_common_dates=n_common,
                   ic1_trigger=float(config.predictive_ic_fail),
                   collapse_ratio=float(config.ic_decay_collapse_ratio))

    if (abs(ic1) >= config.predictive_ic_fail
            and abs(ic2) < config.ic_decay_collapse_ratio * abs(ic1)):
        return warned(
            check,
            f"predictive information lives entirely in the next bar and "
            f"vanishes at t+2 (mean IC(h=1) = {ic1:.3f}, IC at t+2 = "
            f"{ic2:.3f}, ratio {ratio:.3f} < "
            f"{config.ic_decay_collapse_ratio:.2f}) - pattern of an embedded "
            f"future return rather than a persistent alpha.",
            severity=Severity.HIGH,
            remediation=(
                "Audit the feature pipeline for use of the t+1 bar (target "
                "leakage, shift(-1) bugs, same-bar closes); a genuine alpha "
                "decays gradually across horizons instead of dropping to "
                "zero after exactly one bar."),
            details=details)

    return passed(
        check,
        f"IC decay looks organic: mean IC(h=1) = {ic1:.3f} over "
        f"{n_common} common dates, IC at t+2 = {ic2:.3f} (ratio "
        f"{ratio:.2f}); the collapse signature requires |IC1| >= "
        f"{config.predictive_ic_fail:.2f} with ratio < "
        f"{config.ic_decay_collapse_ratio:.2f}.",
        severity=Severity.HIGH,
        details=details)


# ---------------------------------------------------------------------------
# 4. lookahead.same_bar_bleed
# ---------------------------------------------------------------------------

#: Minimum names for residual-correlation dates (residuals on small
#: cross-sections are too noisy at the 5-name floor used elsewhere).
_MIN_NAMES_PARTIAL = 8
#: Dates of measured-degenerate evidence (monotone books, uniform sizes)
#: required before that evidence alone supports a PASS with no residual
#: statistic behind it.
_MIN_PARTIAL_DATES = 60
#: Minimum dates for a nondegenerate residual-correlation verdict. Between
#: this floor and _MIN_PARTIAL_DATES, require twice the usual t evidence
#: because short samples give unstable HAC estimates. Below it, SKIP.
#: In the short band honest non-monotone books measure mean +0.01..+0.03
#: with worst NW t ~4.8, while planted 15-30% bleeds measure +0.58..+0.77
#: at t >= 17, so doubling the t gate removes the honest tail at no
#: detection cost. Both signs are judged: a negated same-bar component is
#: also lookahead - negated bleeds measure -0.66..-0.92 and honest books
#: are sign-symmetric (|mean| <= 0.098), so abs() keeps the 0.10 margin.
_MIN_PARTIAL_DATES_SHORT = 20
#: Residual variance below this counts as "positions are an exact monotone
#: function of the controlled signal" - a good sign, not a data problem.
_DEGENERATE_VAR = 1e-14

#: Minimum median unique-value fraction for the standard partial-correlation
#: threshold. A bucketed export of a finer causal signal retains predictable
#: within-bucket migration in its rank residual. This can resemble a bleed
#: when positions use the finer score, so coarse exports need a separate gate.
#: Median unique fraction on 25-name exports: sign 0.08, tercile 0.12,
#: quintile 0.20, decile 0.40, 15-ile 0.60, round-to-0.1 0.72; continuous
#: artifacts measure ~1.0. Below 0.8 an honest book built from the finer
#: score measures +0.24 (decile) to +0.42 (tercile) partial corr at NW t
#: 19-47, so the standard 0.10 bar cannot judge there.
BLEED_MIN_UNIQUE_FRAC = 0.8

#: Mildly discretized exports use a higher absolute-correlation threshold and
#: a doubled t co-gate. Results below that threshold remain inconclusive;
#: they cannot exonerate the book because lawful bucket migration and weak
#: bleeds can overlap. Below the unique-fraction floor this check SKIPs.
#: The unique-fraction measure depends on breadth: the same number of
#: buckets can appear nearly continuous on a small cross-section.
#: In [0.45, 0.8) honest bucket-migration books measure |mean| <= 0.172,
#: so 0.35 (~2x that wall) convicts only extreme bleeds: 30% bleeds
#: measure |mean| >= 0.69 and 15% bleeds >= 0.51 (negated 15% bleeds
#: >= 0.39 on round-to-0.1 exports but mostly under the bar at
#: 0.5-unique coarseness - those SKIP). Below 0.45 honest migration reaches
#: 0.26 (decile) to 0.44 (tercile), leaving no bar.
BLEED_DISC_JUDGE_MIN_UNIQUE_FRAC = 0.45
BLEED_DISC_PARTIAL_FAIL = 0.35

#: Conditioning basis for declared lags >= 2. The signal side first uses
#: signal_{t-1}; both sides then project on ranks and raw values of the
#: lag-1 and declared-lag signals, lag-1 returns, and trailing mean returns.
#: Raw values capture additive rolling-window updates that rank columns
#: alone cannot represent. Every control uses information through t-1.
#:
#: This basis isolates bar-t information, not every timing violation: partial
#: use of signals between t-L and t can be absorbed. Full lag replacement
#: is also tested by position_signal_alignment. Short rolling signals can
#: retain past-data structure outside this finite basis, so the separability
#: projection below also governs their verdicts. Lag-1 books retain the
#: signal-stamp test, including signals computed from pre-lagged inputs.
#:
#: Conditioning only on signal_{t-L} leaves r_{t-L+1}..r_{t-1} in the
#: residual, which lawful reversal/fast-momentum sleeves track: honest
#: lag-2/5 multi-sleeve books measure -0.41..+0.27 (|NW t| ~63) that way.
#: Conditioning on signal_{t-1} plus the joint rank+value basis brings
#: them to |p50| <= 0.014, p90 <= 0.026 while 15% same-bar bleeds keep
#: |mean| ~0.45. Short rolling signals (w ~5) still leave ~40-50%
#: F(t-1)-measurable variance, which the separability projection handles.
BLEED_LAG2_TRAIL_WINDOWS = (5, 10)

#: Intermittent dependence can have a small date mean but extreme individual
#: dates. Standardize per-date correlations with Fisher z and require both
#: an extreme-date fraction and count, plus the ordinary t co-gate. This is
#: a WARN diagnostic in the full-support lag-1 band; persistent small
#: correlations alone cannot separate lawful overlays from weak leaks.
#:
#: The lag>=2 joint projection consumes cross-sectional degrees of freedom,
#: so the simple Fisher standardization is unsuitable there. Discretized
#: exports also use their own verdict path because bucket migration widens
#: the correlation distribution. Subthreshold dependence outside this
#: scope remains a limitation of the check.
#:
#: Honest overlays earn |NW t| up to ~14 at |mean| <= 0.10 through
#: consistently small per-date corr, so t alone cannot escalate; an
#: intermittent bleed (~7% of dates) instead concentrates in near-1 dates.
#: Honest lag-1 books measure an extreme fraction <= 0.4% and at most 4
#: extreme dates, so 2% keeps 5x margin and the 5-date floor blocks the
#: honest tail; intermittent bleeds measure 3.6-7.1%. Lag >= 2 honest
#: books reach ~1.5% (the joint projection eats df), too close to gate -
#: lag-1 only.
BLEED_EXTREME_DATE_Z = 4.0
BLEED_EXTREME_FRAC = 0.02
BLEED_EXTREME_MIN_DATES = 5

#: Separate bar-t dependence from lawful past-data structure by projecting
#: both residuals on deeper returns and signals. Rolling-window updates
#: retain dropped bars; a second lawful sleeve can track those bars. The
#: projection depth follows signal dependence and available degrees of
#: freedom. An over-threshold raw statistic with insufficient remaining
#: bar-t evidence receives WARN.
#:
#: This projection governs innovations that measurably contain same-bar
#: return news. Return-orthogonal innovations cannot be reconstructed from
#: these return artifacts and retain the declared signal-stamp test. Only
#: lag-1 books can reach CRITICAL through the fresh-signal projection;
#: at larger declared lags the joint basis leaves insufficient separation
#: from lawful multi-sleeve books. Small blends, rank-preserving changes,
#: and dilution with past-data components can remain below the FAIL bar.
BLEED_SEP_MAX_LAGS = 8
#: Cap for the per-bar return-value ladder and dependence scan. Values take
#: priority in the degrees-of-freedom budget because rolling-window updates
#: are linear in past returns. Rank curvature and signal carryover need
#: additional columns. Signals extending beyond this ladder can retain
#: unrepresented past information; the deeper scan handles locatable bars.
#: 12 reaches one past the dropped bar of a 10-bar rolling signal: a w=10
#: signal with a sub-window overlay leaves lawful dependence on
#: r_{t-9}/r_{t-10}, which an 8-lag ladder cannot span (honest sep +0.33
#: there). Deeper rolling signals (w >= ~15) keep unspanned bars and their
#: honest purged sep rises toward the ~0.42-0.43 plateau the conviction
#: bar is calibrated against.
BLEED_SEP_RET_MAX_LAGS = 12
#: BLEED_SEP_ACF_FLOOR: a rolling signal's dependence/difference profile
#: shows a single ~0.6 spike at its window while smooth EWMA/AR signals
#: stay < ~0.02. BLEED_SEP_FRESH_MIN: rolling/return-derived innovations
#: correlate ~0.7 with same-bar return ranks, return-orthogonal scores and
#: pre-lagged exports ~0.0. BLEED_SEP_PARTIAL_FAIL: lawful lag-1
#: double-sleeve books plateau at purged sep ~0.42-0.43 and 15% same-bar
#: blends measure >= ~0.55, so 0.47 keeps ~1.1x margin each way; at
#: lag >= 2 the honest envelope (~0.43) meets the weakest genuine blends
#: (~0.41), so no purged bar exists and over-bar books WARN.
BLEED_SEP_ACF_FLOOR = 0.10
BLEED_SEP_FRESH_MIN = 0.10
BLEED_SEP_PARTIAL_FAIL = 0.47

#: When the dependence scan saturates, locate deep dropped bars directly
#: from per-name correlations of signal differences with lagged returns.
#: For a rolling window, its difference contains the new and dropped bar.
#: The strongest eligible spikes and their adjacent lags join the projection
#: as return-value columns, within its degrees-of-freedom budget.
#:
#: A saturated signal with no locatable spike cannot establish separation
#: and receives at most WARN through this path. Dropped bars beyond the
#: scan cap are unrepresented. A composite signal with only some locatable
#: spikes can retain an unprojected component; weak genuine blends can also
#: lose correlation when the projection removes their past-data component.
#:
#: Honest adjacent-window twin sleeves (w and w-1) at w >= ~60 measure
#: purged sep +0.53..+0.56 without this step - above the bar and
#: indistinguishable from a 15% blend (+0.54); purging the located dropped
#: bar and its +-1 neighbours takes them to 0.30-0.37 while genuine blends
#: stay >= ~0.46. Three spikes x 3 lags = <= 9 value columns; ranks add no
#: shrink and would crowd out signal-lag columns at m=30.
BLEED_SEP_DEEP_SCAN_MAX = 130
BLEED_SEP_DEEP_BARS_MAX = 3

CHECK_BLEED = "lookahead.same_bar_bleed"
CHECK_VOL_SIZING = "lookahead.future_vol_sizing"


def _piecewise_rank_residual(y: np.ndarray, control: np.ndarray, *,
                             rank_y: bool = True) -> np.ndarray:
    """Rank of ``y`` residualized on a piecewise-linear function of the rank
    of ``control`` (quantile bins, within-bin demean + slope removal).

    A plain rank-OLS residual leaves nonlinear structure behind (a decile
    book is a step function of the lagged signal and its OLS residual still
    correlates with other nonlinear transforms of that signal); plain bin
    demeaning leaves the within-bin slope. The piecewise-linear control
    absorbs any smooth-or-step function of ``control``; a genuine same-bar
    component is orthogonal to ``control`` entirely and survives.

    ``rank_y=False`` residualizes the raw values of ``y`` instead of its
    ranks - magnitude-preserving, for detectors that must see tilts too
    small to reorder the book (a rank residual is exactly zero for any
    tilt that leaves the ordering intact)."""
    n = len(y)
    n_bins = int(np.clip(n // 8, 2, 6))
    rc = sps.rankdata(control).astype(float)
    order = sps.rankdata(control, method="ordinal") - 1
    bins = (order * n_bins // n).astype(int)
    ry = sps.rankdata(y).astype(float) if rank_y else np.asarray(y, dtype=float).copy()
    for b in range(n_bins):
        m = bins == b
        k = int(m.sum())
        if k == 0:
            continue
        yb = ry[m] - ry[m].mean()
        if k >= 3:
            xb = rc[m] - rc[m].mean()
            denom = float(xb @ xb)
            if denom > 0:
                yb = yb - (float(yb @ xb) / denom) * xb
        ry[m] = yb
    return ry


def _joint_rank_value_residual(y: np.ndarray,
                               controls: list[np.ndarray]) -> np.ndarray:
    """Residual of ``y`` on the centered ranks and the standardized raw
    values of every control column, jointly (least squares; collinear
    columns are fine - the residual is projection-unique).

    The rank columns absorb monotone structure; the raw-value columns are
    load-bearing for additive recombinations: a rolling signal updates as
    s_t = s_{t-1} + (new bar - dropped bar)/w, an identity that lives in
    values, not ranks. Joint rank/value conditioning represents additive
    past-return structure that rank-only projections can leave behind."""
    cols = []
    for c in controls:
        rc = sps.rankdata(c).astype(float)
        rc -= rc.mean()
        cols.append(rc)
        v = np.asarray(c, dtype=float) - float(np.mean(c))
        sd = float(v.std())
        cols.append(v / sd if sd > 0 else v)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _value_residual(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    """Residual of ``y`` on an intercept plus the standardized raw values of
    the control columns (least squares; collinear columns are fine).
    Value-only on purpose (1 df per control, vs 2 for rank+value): the
    separability purge runs on top of the piecewise + joint projections, and
    the F(t-1) leftovers it must absorb - dropped window bars, sub-window
    means, EWMA weight profiles - are linear in past bars, which lives in
    values (see BLEED_SEP_MAX_LAGS)."""
    cols: list[np.ndarray] = [np.ones(len(y))]
    for c in controls:
        v = np.asarray(c, dtype=float) - float(np.mean(c))
        sd = float(v.std())
        cols.append(v / sd if sd > 0 else v)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _signal_dependence_window(sig: np.ndarray) -> int:
    """The signal's own measured autocorrelation window: the largest lag
    k <= BLEED_SEP_RET_MAX_LAGS at which the median per-name |autocorr|
    still reaches BLEED_SEP_ACF_FLOOR (min 1). A w-bar rolling signal
    measures ~w-1 (acf decays linearly to 0 at w); an AR(1)/slow signal
    caps at BLEED_SEP_RET_MAX_LAGS. All qualifying lags are scanned (no
    early break): periodic/event-gated signals can carry dependence at k=5
    with none at k=1. Sets the purge's return-ladder reach (window + 2,
    one past the dropped bar); the deeper-signal-lag list is separately
    capped at BLEED_SEP_MAX_LAGS."""
    n, m = sig.shape
    window = 1
    for k in range(1, BLEED_SEP_RET_MAX_LAGS + 1):
        if n <= k + 30:
            break
        cors = []
        a, b = sig[k:], sig[:-k]
        for j in range(m):
            x, y = a[:, j], b[:, j]
            f = np.isfinite(x) & np.isfinite(y)
            if int(f.sum()) < 30:
                continue
            xf, yf = x[f], y[f]
            if xf.std() <= 0 or yf.std() <= 0:
                continue
            c = float(np.corrcoef(xf, yf)[0, 1])
            if np.isfinite(c):
                cors.append(c)
        if cors and abs(float(np.median(cors))) >= BLEED_SEP_ACF_FLOOR:
            window = k
    return window


def _deep_bar_scan(sig: np.ndarray, ret: np.ndarray, lo: int) -> list[int]:
    """Locate deep dropped-bar return lags the saturated purge ladder
    cannot reach (see BLEED_SEP_DEEP_SCAN_MAX): lags j in
    [lo, BLEED_SEP_DEEP_SCAN_MAX] where the median per-name
    |corr(sig_t - sig_{t-1}, r_{t-j})| clears BLEED_SEP_ACF_FLOOR. A
    w-bar rolling signal's one-step difference is ~(r_t - r_{t-w})/w,
    motivating a scan for a localized dropped-bar dependence. Smooth
    signals need not have one. Returns spike lags sorted strongest-first
    (empty = no localizable deep bar)."""
    n, m = sig.shape
    hi = min(BLEED_SEP_DEEP_SCAN_MAX, n - 40)
    found: list[tuple[float, int]] = []
    for j in range(lo, hi + 1):
        x = sig[j:] - sig[j - 1:n - 1]
        y = ret[:n - j]
        cors = []
        for c in range(m):
            f = np.isfinite(x[:, c]) & np.isfinite(y[:, c])
            if int(f.sum()) < 30:
                continue
            xf, yf = x[f, c], y[f, c]
            if xf.std() <= 0 or yf.std() <= 0:
                continue
            cc = float(np.corrcoef(xf, yf)[0, 1])
            if np.isfinite(cc):
                cors.append(cc)
        if cors:
            med = float(np.median(cors))
            if abs(med) >= BLEED_SEP_ACF_FLOOR:
                found.append((abs(med), j))
    found.sort(reverse=True)
    return [j for _, j in found]


def _same_bar_bleed(artifacts: BacktestArtifacts,
                    config: AuditConfig) -> CheckResult:
    check = CHECK_BLEED
    if artifacts.positions is None:
        return skipped(check, "pass artifacts.positions (weights held during "
                              "each period) to enable the same-bar bleed "
                              "detector")
    lag = int(artifacts.signal_lag)
    if lag < 1:
        return skipped(check, "invalid artifacts.signal_lag: qaudit's fixed "
                              "end-of-bar timing contract requires an integer "
                              ">= 1; validate the artifacts and remap any "
                              "pre-bar feed to the prior end-of-bar row")
    pos = artifacts.positions.to_numpy(dtype=float)
    sig = artifacts.signals.to_numpy(dtype=float)
    ret = artifacts.asset_returns.to_numpy(dtype=float)
    if lag >= 2:
        # end-of-(t-1) controls for the lag >= 2 basis (see
        # BLEED_LAG2_TRAIL_WINDOWS): row t of each array is measurable at
        # the end of t-1 (previous-bar return; short trailing means through
        # t-1 - the 5/10-bar pair spans every sub-10-bar trailing-mean
        # window ending anywhere in t-10..t-1 as a linear combination).
        r1m = artifacts.asset_returns.shift(1).to_numpy(dtype=float)
        tms = [artifacts.asset_returns.rolling(win, min_periods=1)
               .mean().shift(1).to_numpy(dtype=float)
               for win in BLEED_LAG2_TRAIL_WINDOWS]
    n = pos.shape[0]
    # the lag >= 2 joint projection spends 2 regression columns per basis
    # control (rank + raw value); the residual needs degrees of freedom
    # past them (mirrors the names_floor logic in _future_vol_sizing)
    names_floor = (_MIN_NAMES_PARTIAL if lag == 1
                   else max(_MIN_NAMES_PARTIAL,
                            2 * (3 + len(BLEED_LAG2_TRAIL_WINDOWS)) + 4))
    # Match the separability projection to the measured signal dependence.
    sep_window = _signal_dependence_window(sig)
    # Reach one past the dropped bar of a (window+1)-bar rolling signal,
    # subject to the return-lag cap.
    ret_lag_max = min(sep_window + 2, BLEED_SEP_RET_MAX_LAGS)
    # A saturated dependence scan may miss a rolling signal's dropped bar.
    # Locate deeper bars and project their neighborhoods; without a located
    # bar this path cannot support a CRITICAL verdict.
    sep_saturated = sep_window >= BLEED_SEP_RET_MAX_LAGS
    deep_bars: list[int] = []
    if lag == 1 and sep_saturated:
        deep_bars = _deep_bar_scan(sig, ret, ret_lag_max + 1)
    sep_spanned = (not sep_saturated) or bool(deep_bars)
    deep_lags: list[int] = []
    for db in deep_bars[:BLEED_SEP_DEEP_BARS_MAX]:
        for jj in (db, db - 1, db + 1):
            if jj > ret_lag_max and jj not in deep_lags:
                deep_lags.append(jj)
    vals = []
    sep_stale: list[float] = []      # purge sparing r_{t-1} (stamp channel)
    sep_fresh: list[float] = []      # purge including r_{t-1} (lag-1 only)
    expl_stale: list[float] = []
    expl_fresh: list[float] = []
    fresh_vals: list[float] = []     # per-date corr(signal innovation, r_t ranks)
    breadths: list[int] = []
    n_monotone = 0
    unique_fracs = []
    for t in range(lag, n):
        p, s0, sl = pos[t], sig[t], sig[t - lag]
        # Zero positions stay in - masking to the traded subset selects
        # on the lagged signal and manufactures spurious partial correlation
        # for nonlinear (e.g. decile) books.
        mask = np.isfinite(p) & np.isfinite(s0) & np.isfinite(sl)
        if lag >= 2:
            mask &= np.isfinite(sig[t - 1]) & np.isfinite(r1m[t])
            for tm in tms:
                mask &= np.isfinite(tm[t])
        m = int(mask.sum())
        if m < names_floor:
            continue
        # cardinality of the audited signal cross-section on this date: a
        # coarse (bucketed) export breaks the innovation premise - see
        # BLEED_MIN_UNIQUE_FRAC.
        unique_fracs.append(
            min(np.unique(s0[mask]).size, np.unique(sl[mask]).size) / m)
        # Project positions on the declared-lag signal. At lag>=2, condition the
        # signal on lag 1 and then jointly project both sides on the strictly
        # past rank/value basis to isolate same-bar information.
        rp = _piecewise_rank_residual(p[mask], sl[mask])
        rs = _piecewise_rank_residual(s0[mask],
                                      sig[t - 1][mask] if lag >= 2
                                      else sl[mask])
        if lag >= 2 and (rp.var() >= _DEGENERATE_VAR
                         and rs.var() >= _DEGENERATE_VAR):
            ctls = [sig[t - 1][mask], sl[mask], r1m[t][mask]]
            ctls += [tm[t][mask] for tm in tms]
            rp = _joint_rank_value_residual(rp, ctls)
            rs = _joint_rank_value_residual(rs, ctls)
        if rp.var() < _DEGENERATE_VAR or rs.var() < _DEGENERATE_VAR:
            n_monotone += 1        # book (or signal) fully explained by the basis
            continue
        r = float(np.corrcoef(rp, rs)[0, 1])
        if not np.isfinite(r):
            continue
        vals.append(r)
        breadths.append(m)
        # Separability pass, same date/cells (see BLEED_SEP_MAX_LAGS):
        # re-project both residuals on the raw values of per-bar return
        # lags (the F(t-1) leftovers of a rolling signal are linear in
        # individual past bars, which no finite set of rolling means
        # spans) plus deeper signal lags (for signals whose past is not
        # return-derived: AR overlays, execution-smoothing on s_{t-2}),
        # skipping columns the lag>=2 joint basis already carries and
        # capped by the cross-sectional df left after the piecewise +
        # joint projections. Two variants at lag 1: the stale purge spares
        # r_{t-1} - a pre-lagged export's fresh bar - so the stamp-binding
        # channel survives; the fresh purge includes it. Which one scopes
        # the verdict is decided by the measured signal freshness below.
        # Every raw-usable date yields a sep value, so the sep series
        # shares the raw series' date count and t-gates.
        rr0 = None
        r0v = ret[t][mask]
        if np.isfinite(r0v).all() and np.ptp(r0v) > 0:
            rr0 = sps.rankdata(r0v).astype(float)
            rr0 -= rr0.mean()
            if rr0.std() > 0:
                fr = float(np.corrcoef(rs, rr0)[0, 1])
                if np.isfinite(fr):
                    fresh_vals.append(fr)
        used_cols = 2 * int(np.clip(m // 8, 2, 6))          # piecewise df
        if lag >= 2:
            used_cols += 2 * (3 + len(BLEED_LAG2_TRAIL_WINDOWS))
        budget = max(1, min(3 * BLEED_SEP_MAX_LAGS, m - used_cols - 5))
        ret_cols: list[np.ndarray] = []
        for j in range(2, ret_lag_max + 1):
            if t - j < 0:
                break
            col = ret[t - j][mask]
            if np.isfinite(col).all():
                ret_cols.append(col)
        # Prioritize return values, located deep bars, deeper signal values,
        # then return ranks. Additive rolling updates need value columns;
        # signal columns cover carryover and ranks cover residual curvature.
        ctls = [c for c in ret_cols]
        for j in deep_lags:
            if t - j < 0:
                continue
            col = ret[t - j][mask]
            if np.isfinite(col).all():
                # Use values to preserve degrees of freedom for the
                # signal-lag controls needed by execution smoothing.
                ctls.append(col)
        for j in range(1, min(sep_window, BLEED_SEP_MAX_LAGS) + 1):
            if t - j < 0:
                break
            if lag >= 2 and j in (1, lag):
                continue                     # already in the joint basis
            col = sig[t - j][mask]
            if np.isfinite(col).all():
                ctls.append(col)
        ctls += [sps.rankdata(c).astype(float) for c in ret_cols]
        ctls = ctls[:budget]
        if not ctls:
            ctls = [sl[mask]]      # floor: value of the declared lag itself

        def _sep(cs: list[np.ndarray]) -> tuple[float, float]:
            rp2 = _value_residual(rp, cs)
            rs2 = _value_residual(rs, cs)
            share = float(np.clip(1.0 - float(rs2.var()) / float(rs.var()),
                                  0.0, 1.0))
            if rs2.var() < _DEGENERATE_VAR or rp2.var() < _DEGENERATE_VAR:
                return 0.0, share   # innovation fully strictly-past-explained
            r2 = float(np.corrcoef(rp2, rs2)[0, 1])
            return (r2 if np.isfinite(r2) else 0.0), share

        v, s = _sep(ctls)
        sep_stale.append(v)
        expl_stale.append(s)
        if lag == 1:
            r1v = ret[t - 1][mask]
            if np.isfinite(r1v).all():
                v, s = _sep(ctls + [r1v,
                                    sps.rankdata(r1v).astype(float)])
            sep_fresh.append(v)
            expl_fresh.append(s)
    series = pd.Series(vals)
    if not len(series) and not n_monotone:
        # Never measured anything: no date had >= names_floor jointly
        # finite cells (signal NaN at the declared lag, universe too
        # small) - must SKIP, not silently pass. At lag >= 2 the floor is
        # higher: the joint end-of-(t-1) projection needs df past its
        # regression columns.
        basis_req = ("" if lag == 1 else
                     f", previous-bar signal and t-1 return history "
                     f"(lag >= 2 basis)")
        return skipped(
            check,
            f"no date has >= {names_floor} assets with a finite "
            f"position, same-bar signal AND lag-{lag} signal{basis_req} - "
            f"the partial-correlation statistic never ran. Pass signal "
            f"history that is non-NaN at the declared lag over the "
            f"traded dates (and >= {names_floor} names per date) "
            f"to enable this check.",
            n_usable=0, n_monotone=0)
    med_unique = float(np.median(unique_fracs))
    if len(series) and med_unique < BLEED_MIN_UNIQUE_FRAC:
        # Bucketed signals retain predictable within-bucket structure. Use the
        # higher discretized threshold and leave weaker evidence inconclusive.
        disc_mean = float(series.mean())
        disc_t = float(newey_west_tstat(series))
        disc_details = dict(
            n_usable=int(len(series)), n_monotone=n_monotone,
            mean_partial_corr=disc_mean, nw_tstat=disc_t,
            median_unique_frac=med_unique,
            min_unique_frac=BLEED_MIN_UNIQUE_FRAC,
            disc_judge_min_unique_frac=BLEED_DISC_JUDGE_MIN_UNIQUE_FRAC,
            disc_partial_fail=BLEED_DISC_PARTIAL_FAIL,
            tstat_gate=2.0 * config.bleed_tstat, declared_lag=lag)
        in_band = med_unique >= BLEED_DISC_JUDGE_MIN_UNIQUE_FRAC
        if (in_band and len(series) >= _MIN_PARTIAL_DATES_SHORT
                and abs(disc_mean) > BLEED_DISC_PARTIAL_FAIL
                and abs(disc_t) > 2.0 * config.bleed_tstat):
            signed = "track" if disc_mean > 0 else "track (with inverted sign)"
            return failed(
                check,
                f"even through the discretized signal export (median "
                f"{med_unique:.2f} unique values per name, < "
                f"{BLEED_MIN_UNIQUE_FRAC:.2f}), positions {signed} the "
                f"same-bar signal beyond the declared lag-{lag} signal "
                f"(mean partial rank corr {disc_mean:+.3f}, NW t "
                f"{fmt_tstat(disc_t)} over {len(series)} dates vs the discretized "
                f"bar {BLEED_DISC_PARTIAL_FAIL:.2f} - honest books built "
                f"from a finer causal score measure at most ~0.17 of "
                f"bucket-migration correlation here) - part of the book is "
                f"built from the signal of the bar it trades",
                severity=Severity.CRITICAL,
                remediation="Find the unlagged path into position "
                            "construction (vectorized joins on t instead "
                            "of t-lag, partial rebalances using the fresh "
                            "signal, risk overlays fed same-day scores) "
                            "and shift it by signal_lag; re-run with the "
                            "continuous pre-bucketing score as "
                            "artifacts.signals to size the bleed "
                            "precisely. A negative partial correlation is "
                            "the same illegal channel with a sign error - "
                            "a negated leak is still a leak.",
                details=disc_details)
        if (in_band and len(series) >= _MIN_PARTIAL_DATES_SHORT
                and abs(disc_mean) > BLEED_DISC_PARTIAL_FAIL):
            # The mean clears its threshold but the t co-gate does not;
            # describe that distinction as unresolved evidence.
            band_note = (
                f"|mean partial rank corr| {abs(disc_mean):.3f} (signed "
                f"{disc_mean:+.3f}, NW t {fmt_tstat(disc_t)}) over {len(series)} "
                f"dates CLEARS the discretized conviction bar "
                f"{BLEED_DISC_PARTIAL_FAIL:.2f}, but the NW t evidence "
                f"({fmt_tstat(disc_t)}, need |t| > {2.0 * config.bleed_tstat:.1f}) "
                f"is insufficient to convict - clustered high-correlation "
                f"dates (an intermittent bleed, e.g. a mid-sample pipeline "
                f"change) deflate the NW t; treat as UNRESOLVED, not clean")
        elif in_band and len(series) >= _MIN_PARTIAL_DATES_SHORT:
            band_note = (
                f"measured mean partial rank corr {disc_mean:+.3f} (NW t "
                f"{fmt_tstat(disc_t)}) over {len(series)} dates stays below the "
                f"discretized conviction bar {BLEED_DISC_PARTIAL_FAIL:.2f}, "
                f"where honest books and sub-extreme bleeds overlap")
        elif in_band:
            band_note = (f"only {len(series)} usable date(s) - too few to "
                         f"judge even the extreme band (need >= "
                         f"{_MIN_PARTIAL_DATES_SHORT})")
        else:
            band_note = (f"below {BLEED_DISC_JUDGE_MIN_UNIQUE_FRAC:.2f} "
                         f"unique-value fraction honest bucket-migration "
                         f"correlation reaches ~0.44, leaving no usable bar")
        return skipped(
            check,
            f"signals look like a discretized export (median "
            f"{med_unique:.2f} unique values per name across dates, < "
            f"{BLEED_MIN_UNIQUE_FRAC:.2f}): bucket-boundary migration makes "
            f"the same-bar residual predictable from t-{lag} data, so only "
            f"the extreme band is judgeable here - {band_note}. "
            f"Pass the continuous pre-bucketing score as artifacts.signals "
            f"to enable the full check.",
            details=disc_details)
    trail_str = "/".join(str(w) for w in BLEED_LAG2_TRAIL_WINDOWS)
    basis = (f"lag-{lag} signal" if lag == 1 else
             f"lag-{lag} signal plus the end-of-(t-1) basis "
             f"(signal[t-1], r[t-1], trailing {trail_str}-bar means)")
    if not len(series):
        # Measured degenerate everywhere. Monotone books are a good sign,
        # but the vouching needs depth: below _MIN_PARTIAL_DATES monotone
        # dates the panel is too small to call anything.
        if n_monotone >= _MIN_PARTIAL_DATES:
            return passed(
                check,
                f"positions are a monotone function of the declared "
                f"{basis} on all {n_monotone} measurable dates - "
                f"no residual channel through which a same-bar bleed could "
                f"flow",
                n_usable=0, n_monotone=n_monotone)
        return skipped(
            check,
            f"positions are a monotone function of the declared {basis} "
            f"on the {n_monotone} measurable date(s), but >= "
            f"{_MIN_PARTIAL_DATES} such dates are needed before that alone "
            f"vouches for the book - extend the overlap of positions and "
            f"signals to enable this check.",
            n_usable=0, n_monotone=n_monotone)
    if len(series) < _MIN_PARTIAL_DATES_SHORT:
        # A tiny computed sample supports neither conviction nor exoneration.
        return skipped(
            check,
            f"only {len(series)} date(s) show residual variation beyond the "
            f"declared lag-{lag} signal (need >= "
            f"{_MIN_PARTIAL_DATES_SHORT}; {n_monotone} monotone dates "
            f"cannot vouch for the residual channel) - extend the overlap "
            f"of positions and signals to enable this check.",
            n_usable=int(len(series)), n_monotone=n_monotone)
    mean = float(series.mean())
    tstat = float(newey_west_tstat(series))
    # Short samples require twice the t evidence. Test both signs because
    # negating a same-bar signal component also violates its declared timing.
    t_gate = config.bleed_tstat * (
        2.0 if len(series) < _MIN_PARTIAL_DATES else 1.0)
    # Count extreme per-date Fisher-z values as well as their date fraction.
    # Intermittent dependence can concentrate in a few dates and leave the
    # overall mean below the ordinary level threshold.
    z_abs = np.abs(np.arctanh(np.clip(series.to_numpy(), -0.999999,
                                      0.999999)))
    z_abs *= np.sqrt(np.maximum(np.asarray(breadths, dtype=float) - 3.0,
                                1.0))
    n_extreme = int((z_abs >= BLEED_EXTREME_DATE_Z).sum())
    extreme_frac = float(n_extreme / len(series))
    # Measure remaining bar-t correlation and the past-explained share.
    # The fresh-signal projection includes r_{t-1}; the stale variant retains
    # that bar so pre-lagged signal exports still respect the stamp contract.
    freshness = (float(np.mean(fresh_vals)) if fresh_vals else 0.0)
    use_fresh = (lag == 1 and abs(freshness) >= BLEED_SEP_FRESH_MIN
                 and len(sep_fresh) == len(sep_stale))
    sep_series = pd.Series(sep_fresh if use_fresh else sep_stale)
    expl_shares = expl_fresh if use_fresh else expl_stale
    sep_mean = float(sep_series.mean())
    sep_t = float(newey_west_tstat(sep_series))
    f_t1_share = float(np.mean(expl_shares)) if expl_shares else 0.0
    details = dict(mean_partial_corr=mean, nw_tstat=tstat,
                   n_usable=int(len(series)), n_monotone=n_monotone,
                   median_unique_frac=med_unique, tstat_gate=t_gate,
                   n_extreme_dates=n_extreme, extreme_frac=extreme_frac,
                   extreme_date_z=BLEED_EXTREME_DATE_Z,
                   extreme_frac_gate=BLEED_EXTREME_FRAC,
                   sep_mean_partial_corr=sep_mean, sep_nw_tstat=sep_t,
                   f_t1_explained_share=f_t1_share,
                   sep_lag_window=sep_window,
                   sep_signal_freshness=freshness,
                   sep_fresh_purge=use_fresh,
                   sep_saturated=sep_saturated,
                   sep_spanned=sep_spanned,
                   sep_deep_bars=deep_bars[:BLEED_SEP_DEEP_BARS_MAX],
                   declared_lag=lag)
    remediation = ("Find the unlagged path into position construction "
                   "(vectorized joins on t instead of t-lag, partial "
                   "rebalances using the fresh signal, risk overlays fed "
                   "same-day scores) and shift it by signal_lag. A "
                   "NEGATIVE partial correlation is the same illegal "
                   "same-bar channel entering with a sign error - check "
                   "for inverted sign conventions in the same-bar join; "
                   "a negated leak is still a leak.")
    if abs(mean) > config.bleed_partial_fail and abs(tstat) > t_gate:
        # Return-linked innovations need additional evidence after removing
        # past-data structure. Return-orthogonal innovations retain the raw
        # signal-stamp test because these artifacts cannot span their inputs.
        signed = ("track" if mean > 0
                  else "track (with inverted sign)")
        core_fail = (
            f"controlling for the declared {basis}, positions still "
            f"{signed} the same-bar signal innovation (mean partial rank "
            f"corr {mean:+.3f}, NW t {fmt_tstat(tstat)} over {len(series)} dates)")
        if abs(freshness) < BLEED_SEP_FRESH_MIN:
            return failed(
                check,
                core_fail + f" - part of the book is built from the signal "
                f"of the bar it trades (signal innovation is "
                f"return-orthogonal: freshness {freshness:+.2f} vs the "
                f"{BLEED_SEP_FRESH_MIN:.2f} gate, so the separability "
                f"purge has no jurisdiction and the signal-stamp contract "
                f"governs)",
                severity=Severity.CRITICAL,
                remediation=remediation,
                details=details)
        if (lag == 1 and sep_spanned
                and abs(sep_mean) > BLEED_SEP_PARTIAL_FAIL
                and abs(sep_t) > t_gate):
            # This conviction path requires lag 1 and a projection that reaches
            # the relevant deep bars. Other cases retain the scoped WARN.
            deep_note = (f" plus located deep dropped bars "
                         f"{sorted(deep_bars[:BLEED_SEP_DEEP_BARS_MAX])}"
                         if deep_bars else "")
            return failed(
                check,
                core_fail + f" - and the bar-t component alone convicts: "
                f"with the F(t-1)-explained share of the innovation "
                f"({f_t1_share:.0%}; per-bar return lags plus deeper "
                f"signal lags out to {sep_window}{deep_note}) removed "
                f"from both sides, the residual still measures "
                f"{sep_mean:+.3f} "
                f"(NW t {fmt_tstat(sep_t)}, vs the {BLEED_SEP_PARTIAL_FAIL:.2f} "
                f"purged bar). The residual dependence survives the "
                f"specified past-data controls - evidence that part "
                f"of the book is built from the signal of the bar it "
                f"trades",
                severity=Severity.CRITICAL,
                remediation=remediation,
                details=details)
        if lag == 1 and not sep_spanned:
            sep_note = (
                f"the purge cannot certify a bar-t component at all: the "
                f"signal's dependence window saturates the "
                f"{BLEED_SEP_RET_MAX_LAGS}-lag scan and no localizable "
                f"deep dropped bar was found within "
                f"{BLEED_SEP_DEEP_SCAN_MAX} lags (see "
                f"BLEED_SEP_DEEP_SCAN_MAX) - the purged statistic "
                f"({sep_mean:+.3f}, NW t {fmt_tstat(sep_t)}) may reflect deep "
                f"strictly-past structure the purge cannot span, which a "
                f"lawful second sleeve tracks legitimately")
        elif lag == 1:
            sep_note = (
                f"the remaining bar-t news component ({sep_mean:+.3f}, "
                f"NW t {fmt_tstat(sep_t)}) does not independently clear the "
                f"{BLEED_SEP_PARTIAL_FAIL:.2f} purged conviction bar; "
                f"the remaining dependence can reflect a lawful second "
                f"sleeve or a partial same-bar blend")
        else:
            sep_note = (
                f"at declared lag {lag} >= 2 the purged statistic "
                f"({sep_mean:+.3f}, NW t {fmt_tstat(sep_t)}) cannot certify a "
                f"bar-t component at all - the joint end-of-(t-1) basis "
                f"consumes the cross-sectional df the purge needs, so "
                f"lawful multi-sleeve dependence and partial same-bar "
                f"blends can remain inseparable")
        return warned(
            check,
            f"positions track the same-bar signal innovation beyond the "
            f"declared {basis} (mean partial rank corr {mean:+.3f}, NW t "
            f"{fmt_tstat(tstat)} over {len(series)} dates, past the "
            f"{config.bleed_partial_fail:.2f} bar) - but {f_t1_share:.0%} "
            f"of that innovation is itself explained by strictly-past "
            f"constructions (per-bar return lags plus deeper signal lags "
            f"out to {sep_window}), and {sep_note}: a lawful second "
            f"sleeve tracking the signal's own past (a multi-horizon "
            f"momentum blend, a short-window reversal overlay) and a "
            f"partial same-bar leak are not separable in this artifact "
            f"set.",
            severity=Severity.HIGH,
            remediation=(
                "If the book blends several sleeves/horizons, declare the "
                "blended CONTINUOUS composite score as artifacts.signals "
                "(an honest composite declaration measures ~0 here; a "
                "rank-sum composite lands in the discretized band instead) "
                "or audit each sleeve separately. If the book truly trades "
                "only the declared signal, treat this as a likely partial "
                "same-bar leak: " + remediation),
            details=details)
    # Escalate concentrated extreme-date dependence within its supported
    # lag-1 scope. Aggregate t evidence alone also occurs with lawful overlays.
    if (lag == 1 and len(series) >= _MIN_PARTIAL_DATES
            and n_extreme >= BLEED_EXTREME_MIN_DATES
            and extreme_frac >= BLEED_EXTREME_FRAC
            and abs(tstat) > config.bleed_tstat):
        return warned(
            check,
            f"mean partial rank corr {mean:+.3f} over {len(series)} dates "
            f"sits below the {config.bleed_partial_fail:.2f} conviction "
            f"bar, but {n_extreme} dates ({extreme_frac:.1%}, vs the "
            f"{BLEED_EXTREME_FRAC:.1%} escalation gate) carry an extreme "
            f"same-bar loading (per-date Fisher z >= "
            f"{BLEED_EXTREME_DATE_Z:.0f}, ~4-sigma under the date null) "
            f"with aggregate NW t {fmt_tstat(tstat)} - the signature of an "
            f"INTERMITTENT same-bar bleed: a leak live on a subset of "
            f"dates dilutes the date-mean but not the extreme-date mass. "
            f"This tail diagnostic warrants investigation; it does not "
            f"establish a universal false-positive rate.",
            severity=Severity.HIGH,
            remediation=remediation + " Inspect the flagged dates for a "
                        "partial pipeline change (the leak may be live "
                        "only after a deploy or only on rebalance dates).",
            details=details)
    if abs(mean) > config.bleed_partial_fail:
        # A mean above the level threshold without the t co-gate is unresolved;
        # report both measurements without claiming the channel is absent.
        return passed(
            check,
            f"|mean partial rank corr| {abs(mean):.3f} (signed {mean:+.3f}) "
            f"over {len(series)} dates clears the "
            f"{config.bleed_partial_fail:.2f} level bar, but the NW t "
            f"evidence ({fmt_tstat(tstat)}, need |t| > {t_gate:.1f}) is "
            f"insufficient to convict - not exonerated: clustered leak "
            f"dates deflate the NW t; treat as unresolved rather than "
            f"clean.",
            details=details, unresolved=True)
    return passed(
        check,
        f"positions carry no same-bar signal information beyond the declared "
        f"{basis} (mean partial rank corr {mean:+.3f}, NW t "
        f"{fmt_tstat(tstat)} over {len(series)} dates; honest books measure ~0)",
        details=details)


# ---------------------------------------------------------------------------
# 5. lookahead.future_vol_sizing
# ---------------------------------------------------------------------------

#: Control future-volatility comparisons for strictly trailing risk state:
#: rolling and EWMA volatility across horizons, downside deviation, absolute
#: returns, and trailing means for leverage-related dependence. A single
#: noisy trailing estimate can leave predictable volatility in the residual.
#: All controls are derived from returns and shifted one bar; using signals
#: as risk controls could let the tested sizing input absorb its own leak.
#: Grid density matters: a single trailing std convicts every honest
#: inverse-vol book under GARCH (-0.11..-0.15); a 10/60/EWMA20 grid still
#: convicts 20d inverse-vol sizing (~-0.07); without semivol, |ret|-EWMA
#: and hl5 controls honest Sortino/semivol and fast-EWMA books fail under
#: GJR leverage.
#:
#: The grid cannot span per-name information absent from close-to-close
#: returns, including implied volatility, OHLC ranges and intraday realized
#: volatility. Lawful sizing on such inputs may load on the residual, so the
#: verdict must retain that limitation: non-return risk inputs are a hard
#: limit, and honest implied-vol sizers measure -0.20..-0.47 (NW t
#: -17..-26), inside the planted-leak band (-0.09..-0.39). Resolving those
#: cases needs the actual risk series and an audit of its timestamps and
#: construction. Uniform index-volatility scaling cancels in
#: cross-sectional ranks.
FWD_VOL_CONTROL_WINDOWS = (15, 20, 30, 40, 60)
FWD_VOL_EWMA_HALFLIVES = (5, 10, 20, 40, 60)
FWD_VOL_SEMIVOL_WINDOWS = (20, 60)
FWD_VOL_ABSRET_HALFLIFE = 15
FWD_VOL_MEAN_HALFLIFE = 20

#: Signed mean partial-correlation thresholds, used with the HAC t co-gate.
#: The positive threshold is higher because proportional trailing-volatility
#: sizing can retain positive estimation bias after projection. Weak forward
#: or centered-window blends can overlap that lawful dependence and remain
#: WARN or unflagged; these constants do not establish universal error rates.
#: Negative side: honest returns-derived trailing books bottom at about
#: -0.07 while planted centered/forward inverse sizers measure -0.10..-0.39,
#: so FAIL at -0.075 sits just past the honest tail and WARN at -0.05 marks
#: the gray band. Positive side: honest proportional-to-trailing-vol books
#: carry up to ~+0.10 of estimation bias, so WARN sits at that ceiling and
#: FAIL at 1.5x it (+0.15); planted proportional forward sizers measure
#: +0.18..+0.43. Weak (25%) forward blends and iid-vol centered sizers
#: overlap the honest positive band and may only WARN.
FWD_VOL_PARTIAL_WARN = -0.05
FWD_VOL_PARTIAL_FAIL = -0.075
FWD_VOL_PARTIAL_WARN_POS = 0.10
FWD_VOL_PARTIAL_FAIL_POS = 0.15


def _rank_residual_multi(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residual of centered rank(y) on the centered ranks of each control
    column (least squares; collinear controls - e.g. three highly correlated
    vol estimates - are fine: the residual is projection-unique)."""
    ry = sps.rankdata(y).astype(float)
    ry -= ry.mean()
    X = np.column_stack([sps.rankdata(controls[:, j]).astype(float)
                         for j in range(controls.shape[1])])
    X -= X.mean(axis=0)
    beta, *_ = np.linalg.lstsq(X, ry, rcond=None)
    return ry - X @ beta


def _future_vol_sizing(artifacts: BacktestArtifacts,
                       config: AuditConfig) -> CheckResult:
    check = CHECK_VOL_SIZING
    if artifacts.positions is None:
        return skipped(check, "pass artifacts.positions (weights held during "
                              "each period) to enable the future-volatility "
                              "sizing detector")
    w = int(config.fwd_vol_window)
    rets = artifacts.asset_returns
    if w + 1 >= len(rets) - w:
        # Candidate t must satisfy w + 1 <= t < n - w.  If that interval is
        # empty, no rolling call is needed; avoiding it also prevents a huge
        # configured window from overflowing pandas' C-sized window parser.
        return skipped(
            check,
            f"the {w}-bar forward/trailing windows leave zero candidate "
            f"dates in this {len(rets)}-row panel - extend the sample or "
            f"reduce AuditConfig.fwd_vol_window.",
            n_usable=0, n_candidates=0, n_below_names=0, n_uniform=0,
            n_explained=0, fwd_vol_window=w,
        )
    fwd_vol = rets.rolling(w, min_periods=w).std().shift(-w)   # vol over t+1..t+w
    # Trailing risk state known at decision time (all <= t-1): a grid of
    # rolling stds, an EWMA std, and the trailing mean return (leverage
    # channel). min_periods=w keeps the usable date range at n - 2w - 1.
    windows = sorted({w, *FWD_VOL_CONTROL_WINDOWS})
    # Each trailing window requires no more observations than its length,
    # including when the requested forward-volatility window is longer.
    controls = [rets.rolling(win, min_periods=min(w, win)).std().shift(1)
                for win in windows]
    controls += [rets.pow(2).ewm(halflife=hl, min_periods=w)
                 .mean().pow(0.5).shift(1)
                 for hl in FWD_VOL_EWMA_HALFLIVES]
    # Downside deviation and absolute-return EWMA capture asymmetric risk
    # state that symmetric volatility and a linear mean need not span.
    downside = rets.where(rets < 0, 0.0)
    controls += [downside.pow(2).rolling(win, min_periods=min(w, win))
                 .mean().pow(0.5).shift(1)
                 for win in FWD_VOL_SEMIVOL_WINDOWS]
    controls.append(rets.abs().ewm(halflife=FWD_VOL_ABSRET_HALFLIFE,
                                   min_periods=w).mean().shift(1))
    # leverage channel: short + smoothed trailing mean returns (GJR-style
    # markets let return levels/signs predict vol innovations; a winner- or
    # loser-tilted honest book otherwise picks that up as "anticipation")
    controls.append(rets.rolling(w, min_periods=w).mean().shift(1))
    controls.append(rets.ewm(halflife=FWD_VOL_MEAN_HALFLIFE, min_periods=w)
                    .mean().shift(1))
    pos = artifacts.positions.to_numpy(dtype=float)
    fv = fwd_vol.to_numpy(dtype=float)
    ctl = np.stack([c.to_numpy(dtype=float) for c in controls], axis=-1)
    # the residual needs a few degrees of freedom past the control count
    names_floor = max(_MIN_NAMES_PARTIAL, ctl.shape[-1] + 3)
    vals = []
    n_candidates = 0
    n_below_names = 0
    n_uniform = 0
    n_explained = 0
    for t in range(w + 1, pos.shape[0] - w):
        n_candidates += 1
        p = np.abs(pos[t])
        mask = ((p > 1e-12) & np.isfinite(fv[t])
                & np.isfinite(ctl[t]).all(axis=-1))
        if int(mask.sum()) < names_floor:
            n_below_names += 1
            continue
        pm = p[mask]
        if np.ptp(pm) == 0:
            n_uniform += 1              # equal-weight date: sizes carry no signal
            continue
        resid_fv = _rank_residual_multi(fv[t][mask], ctl[t][mask])
        resid_p = _rank_residual_multi(pm, ctl[t][mask])
        if resid_fv.var() < _DEGENERATE_VAR:
            n_explained += 1   # forward vol rank-explained by the trailing state
            continue
        if resid_p.var() < _DEGENERATE_VAR:
            n_explained += 1   # sizes rank-explained by the trailing state
            continue
        r = float(np.corrcoef(resid_p, resid_fv)[0, 1])
        if np.isfinite(r):
            vals.append(r)
    series = pd.Series(vals)
    counts = dict(n_usable=int(len(series)), n_candidates=n_candidates,
                  n_below_names=n_below_names, n_uniform=n_uniform,
                  n_explained=n_explained, fwd_vol_window=w)
    if len(series) < _MIN_PARTIAL_DATES_SHORT:
        # Track why dates were lost: only genuinely uniform sizes (or sizes
        # fully rank-explained by the trailing state) justify a PASS with no
        # statistic behind it; a shortfall driven by the sample-range cap
        # (n - 2w - 1 candidates) or the names floor is unmeasurable and
        # must SKIP rather than imply that the unmeasured sizes are uniform.
        if not len(series) and n_uniform + n_explained >= _MIN_PARTIAL_DATES:
            return passed(
                check,
                f"sizes carry no future-vol channel on any measurable date: "
                f"{n_uniform} dates have uniform sizes and {n_explained} "
                f"dates are fully rank-explained by the trailing risk state "
                f"- no room for future-vol sizing",
                details=counts)
        return skipped(
            check,
            f"only {len(series)} of {n_candidates} candidate dates are "
            f"usable for the forward-vol statistic ({n_below_names} below "
            f"the {names_floor}-name floor, {n_uniform} with uniform "
            f"sizes, {n_explained} rank-explained by the trailing state; "
            f"need >= {_MIN_PARTIAL_DATES_SHORT}). The forward/trailing "
            f"{w}-bar windows leave n - {2 * w + 1} candidate dates in an "
            f"n-row panel - extend the sample to enable this check.",
            details=counts)
    mean = float(series.mean())
    tstat = float(newey_west_tstat(series))
    # small-sample co-gate, mirrored from same_bar_bleed: the short band
    # [_MIN_PARTIAL_DATES_SHORT, _MIN_PARTIAL_DATES) demands double the t
    # evidence (planted forward-vol sizers measure |t| >= 100 there).
    t_gate = config.bleed_tstat * (
        2.0 if len(series) < _MIN_PARTIAL_DATES else 1.0)
    details = dict(mean_partial_corr=mean, nw_tstat=tstat,
                   corr_warn=FWD_VOL_PARTIAL_WARN,
                   corr_fail=FWD_VOL_PARTIAL_FAIL,
                   corr_warn_pos=FWD_VOL_PARTIAL_WARN_POS,
                   corr_fail_pos=FWD_VOL_PARTIAL_FAIL_POS,
                   tstat_gate=t_gate, **counts)
    win_str = "/".join(f"{win}d" for win in windows)
    ewma_str = "/".join(f"hl{hl}" for hl in FWD_VOL_EWMA_HALFLIVES)
    semi_str = "/".join(f"{win}d" for win in FWD_VOL_SEMIVOL_WINDOWS)
    core = (f"partial corr of |positions| with forward {w}-bar vol (both "
            f"residualized on the trailing risk state: {win_str}/"
            f"EWMA-{ewma_str} vol + {semi_str} semivol + |ret|-EWMA-"
            f"hl{FWD_VOL_ABSRET_HALFLIFE} + trailing mean return) = "
            f"{mean:+.3f}, NW t {fmt_tstat(tstat)} over {len(series)} dates")
    # Scope the verdict to returns-derived risk models. Per-name risk inputs
    # absent from these artifacts can produce similar residual dependence.
    iv_caveat = (" CAVEAT: this statistic cannot distinguish a forward "
                 "window from honest sizing on per-name risk inputs that "
                 "close-to-close returns cannot reproduce (single-name "
                 "implied vol, intraday-range or high-frequency realized "
                 "vol estimators); if the "
                 "book's risk model is of that class, treat this as "
                 "UNRESOLVED and audit the risk-model series' own timing "
                 "directly (uniform index-vol scaling is unaffected - it "
                 "cancels in the cross-sectional ranks).")
    remediation = ("Position sizes anticipate volatility that has not been "
                   "realized yet - look for centered rolling windows, vol "
                   "computed over the holding period, or risk stats aligned "
                   "off by a window length; replace with trailing (<= t-lag) "
                   "estimates. If sizing already uses only trailing data "
                   "through a non-return channel (per-name implied vol, "
                   "intraday-range or high-frequency realized vol), this "
                   "check cannot exonerate it from returns alone - verify "
                   "the risk-model series is shift(1)-stamped end to end "
                   "instead of degrading it to a return-derived estimate.")
    if mean <= FWD_VOL_PARTIAL_FAIL and tstat <= -t_gate:
        return failed(
            check,
            core + f" - sizes track the part of forward vol that the "
                   f"{ctl.shape[-1]}-model trailing risk grid cannot "
                   f"explain: sizes systematically SHRINK before vol "
                   f"arrives. This exceeds the negative correlation and "
                   f"t thresholds for the RETURNS-DERIVED risk model "
                   f"comparison; it is evidence to investigate the sizing "
                   f"series' timing."
                   + iv_caveat,
            severity=Severity.CRITICAL, remediation=remediation, details=details)
    if mean <= FWD_VOL_PARTIAL_WARN and tstat <= -t_gate:
        return warned(
            check,
            core + " - mildly but consistently negative against the vol "
                   "innovation; verify every vol/risk input into sizing is "
                   "strictly trailing." + iv_caveat,
            severity=Severity.HIGH, remediation=remediation, details=details)
    if mean >= FWD_VOL_PARTIAL_FAIL_POS and tstat >= t_gate:
        return failed(
            check,
            core + " - sizes systematically GROW before vol arrives: "
                   "sizing PROPORTIONAL to volatility that is not realized "
                   "yet is the same lookahead as inverse-sizing on it. "
                   "The positive threshold allows for residual dependence "
                   "from proportional trailing-risk estimates; check the "
                   "sizing series' timing and the risk-model limitations.",
            severity=Severity.CRITICAL, remediation=remediation, details=details)
    if mean >= FWD_VOL_PARTIAL_WARN_POS and tstat >= t_gate:
        return warned(
            check,
            core + " - consistently positive against the vol innovation, "
                   "above the positive warning threshold; "
                   "verify every vol/risk input into sizing is strictly "
                   "trailing (sizes growing before vol arrives)",
            severity=Severity.HIGH, remediation=remediation, details=details)
    # A signed level threshold without the t co-gate remains unresolved.
    # Clustered dependence can weaken HAC evidence; report the measured
    # level rather than implying that future-volatility dependence is absent.
    if mean <= FWD_VOL_PARTIAL_WARN:
        bar, bar_name = ((FWD_VOL_PARTIAL_FAIL, "FAIL")
                         if mean <= FWD_VOL_PARTIAL_FAIL
                         else (FWD_VOL_PARTIAL_WARN, "WARN"))
        return passed(
            check,
            core + f" - the mean sits beyond the {bar:+.3f} {bar_name} bar, "
                   f"but the NW t evidence ({fmt_tstat(tstat)}, need <= "
                   f"{-t_gate:.1f}) is insufficient to convict - NOT "
                   f"exonerated: clustered per-date correlations (an "
                   f"intermittent forward-vol sizer live on a block of "
                   f"dates) deflate the NW t. Lawful trailing-risk "
                   f"dependence can also remain after projection, so "
                   f"this band is unresolved in both directions.",
            details=details, unresolved=True)
    if mean >= FWD_VOL_PARTIAL_WARN_POS:
        bar, bar_name = ((FWD_VOL_PARTIAL_FAIL_POS, "FAIL")
                         if mean >= FWD_VOL_PARTIAL_FAIL_POS
                         else (FWD_VOL_PARTIAL_WARN_POS, "WARN"))
        return passed(
            check,
            core + f" - the mean sits beyond the {bar:+.3f} {bar_name} bar "
                   f"(positive side), but the NW t evidence ({fmt_tstat(tstat)}, "
                   f"need >= {t_gate:.1f}) is insufficient to convict - "
                   f"NOT exonerated: clustered per-date correlations "
                   f"deflate the NW t; treat as unresolved rather than "
                   f"clean.",
            details=details, unresolved=True)
    return passed(
        check,
        core + " - no anticipation signature clears the forward-vol "
               "thresholds in either direction. This comparison controls "
               "for RETURNS-DERIVED trailing-risk estimates; per-name "
               "risk inputs absent from returns, such as implied vol "
               "or range estimates, remain outside its scope. Passing "
               "these thresholds does not establish causal sizing.",
        details=details)


# ---------------------------------------------------------------------------
# 6. lookahead.deferred_ic_spike
# ---------------------------------------------------------------------------

CHECK_DEFERRED = "lookahead.deferred_ic_spike"

#: Ratio of |IC(h=1)| to the max |IC| at h>=2 below which the profile is
#: anomalous (WARN), resp. damning (FAIL). Honest alphas decay with horizon;
#: they do not skip the first bar and spike later - that shape appears when
#: a feature contains returns from a deferred future window (e.g. t+2..t+6).
DEFERRED_WARN_RATIO = 0.50
DEFERRED_FAIL_RATIO = 0.33


def _deferred_ic_spike(artifacts: BacktestArtifacts,
                       config: AuditConfig) -> CheckResult:
    check = CHECK_DEFERRED
    h_max = int(config.deferred_ic_max_horizon)
    if h_max < 2:
        # AuditConfig validates >= 2 at construction, but a crafted config
        # object must degrade to SKIP here, not raise from max() over the
        # empty horizon range 2..h_max, which would turn every lookahead
        # result into a single module ERROR.
        return skipped(
            check,
            f"deferred_ic_max_horizon={h_max} leaves no deferred horizons: "
            f"the spike profile compares IC at t+1 against t+2..t+h_max, so "
            f"it needs config.deferred_ic_max_horizon >= 2 (default 6) to "
            f"run.",
            deferred_ic_max_horizon=h_max)
    fwd1 = forward_returns(artifacts.asset_returns, 1)
    ic_series: dict[int, pd.Series] = {}
    for h in range(1, h_max + 1):
        target = fwd1.shift(-(h - 1))      # the single-period return at t+h
        ic = cross_sectional_ic(artifacts.signals, target,
                                min_names=_MIN_NAMES).dropna()
        if len(ic) < MIN_IC_DATES:
            # An unmeasurable check must say so, not silently pass.
            return skipped(
                check,
                f"only {len(ic)} usable IC dates at horizon {h} (need >= "
                f"{MIN_IC_DATES}) - sample too short to profile IC across "
                f"horizons; extend the overlap of signals and asset_returns "
                f"to enable this check.",
                n_dates=int(len(ic)), horizon=int(h),
                min_ic_dates=MIN_IC_DATES)
        ic_series[h] = ic
    aligned = pd.concat(ic_series, axis=1, sort=True).dropna()
    n_common = int(len(aligned))
    if n_common < MIN_IC_DATES:
        counts = {str(h): int(len(s)) for h, s in ic_series.items()}
        return skipped(
            check,
            f"only {n_common} calendar dates are jointly measurable across "
            f"the h=1..{h_max} IC profile (need >= {MIN_IC_DATES}, though "
            f"each horizon alone clears the floor) - comparing means from "
            f"different date samples can manufacture or hide a deferred "
            f"spike. Fill/drop the gapped target dates or extend their "
            f"common overlap.",
            n_dates_by_horizon=counts, n_common_dates=n_common,
            min_ic_dates=MIN_IC_DATES,
        )
    ics = {h: float(aligned[h].mean()) for h in ic_series}
    ic1 = abs(ics[1])
    spike_h = max(range(2, h_max + 1), key=lambda h: abs(ics[h]))
    spike = abs(ics[spike_h])
    profile = {str(h): round(v, 4) for h, v in ics.items()}
    details = dict(ic_by_horizon=profile, ic1=round(ics[1], 4),
                   spike_horizon=spike_h, spike_ic=round(ics[spike_h], 4),
                   n_common_dates=n_common,
                   n_dates_by_horizon={
                       str(h): int(len(s)) for h, s in ic_series.items()})
    remediation = (
        "A feature contains returns from a deferred future window (e.g. a "
        "'forward-looking' factor, a label smeared over t+2.., or a join "
        "misaligned by several days). Profile each feature's IC by horizon "
        "to find the culprit, re-lag it, and regenerate signals.")
    if spike >= config.predictive_ic_fail and ic1 < DEFERRED_FAIL_RATIO * spike:
        return failed(
            check,
            f"signal predicts the single-period return at t+{spike_h} (mean "
            f"IC {ics[spike_h]:+.3f}) far better than at t+1 ({ics[1]:+.3f}); "
            f"IC by horizon: {profile} - honest alphas decay with horizon, "
            f"they do not skip the first bar and spike later",
            severity=Severity.CRITICAL, remediation=remediation, details=details)
    if spike >= config.predictive_ic_warn and ic1 < DEFERRED_WARN_RATIO * spike:
        return warned(
            check,
            f"IC at t+{spike_h} ({ics[spike_h]:+.3f}) is more than twice the "
            f"IC at t+1 ({ics[1]:+.3f}); IC by horizon: {profile} - verify no "
            f"feature is computed over a deferred future window",
            severity=Severity.HIGH, remediation=remediation, details=details)
    return passed(
        check,
        f"IC decays normally across horizons (t+1: {ics[1]:+.3f}, max at "
        f"t+{spike_h}: {ics[spike_h]:+.3f}) - no deferred-window signature",
        details=details)


# ---------------------------------------------------------------------------
# 7. lookahead.smeared_forward_ic
# ---------------------------------------------------------------------------

CHECK_SMEAR = "lookahead.smeared_forward_ic"

#: Horizons profiled past config.deferred_ic_max_horizon: the smear spreads
#: modest IC over many bars precisely to duck the per-horizon gates, so the
#: mass integrates a few bars beyond the spike window (default H = 6+4).
SMEAR_EXTRA_HORIZONS = 4

#: Trailing-mean control window (matches the synthetic momentum convention;
#: the control need not equal the signal's own window - any monotone map of
#: observables <= t that absorbs honest innovation loadings does the job).
SMEAR_CONTROL_MA = 20

#: Thresholds for integrated absolute innovation IC after subtracting the
#: breadth-implied noise mass. The t co-gate also requires persistent evidence;
#: the mass thresholds alone cannot distinguish a short-sample fluctuation.
#: Honest momentum/reversal/blend families on the synthetic market reach at
#: most ~0.066 excess mass (family p95s <= 0.057); WARN is 1.5x and FAIL
#: 2.3x that ceiling. A smeared leak measures ~1.3 at net SR ~5 and its
#: excess stays >= the realized IC(h=1) however it is spread, so evading
#: the FAIL bar caps the leaked Sharpe.
SMEAR_MASS_WARN = 0.10
SMEAR_MASS_FAIL = 0.15

#: Newey-West t of the sign-aligned pooled per-date innovation IC required
#: alongside the mass bars: the null-mass subtraction centers the statistic
#: but does not shrink its dispersion, so a short/narrow panel could reach
#: the bars on noise alone; real leaks measure t >> 10.
SMEAR_TSTAT = 4.0

#: Per-horizon floor of usable innovation-IC dates below which the mass is
#: not a measurement (and, when the residualization is degenerate instead -
#: signal rank-explained by its own controls - there is no innovation to
#: leak through, which is a PASS, not a SKIP).
SMEAR_MIN_DATES = 60




def _masked_ranks(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row average ranks over the masked cells, 0.0 elsewhere."""
    x = np.where(mask, vals, np.nan)
    r = pd.DataFrame(x).rank(axis=1).to_numpy()
    return np.where(mask, r, 0.0)


def _innovation_forward_ic(signals: pd.DataFrame,
                           asset_returns: pd.DataFrame,
                           h_max: int) -> dict | None:
    """Per-horizon per-date IC of the signal innovation vs forward returns.

    Innovation = residual of the centered-rank cross-section of signal_t on
    the centered ranks of [signal_{t-1}, last-nonzero signal through t-1,
    r_t, ma(r)_t] (batched pseudo-inverse: residuals are projection-unique
    even when controls are collinear, e.g. a momentum signal that is its
    own trailing mean).

    The last-nonzero signal through t-1 controls persistent information in
    event-gated exports. For continuously nonzero signals it duplicates the
    lag-1 signal. On event dates after zero rows it retains the earlier
    signal state, preventing calendar recurrence from being counted as new
    information at multiple forward horizons. The control is strictly past.

    Returns None when any horizon has an underpowered non-degenerate sample;
    an exactly rank-explained horizon may stand on >= SMEAR_MIN_DATES
    degenerate dates, but only when every horizon independently earns one of
    those two forms of support.
    """
    fwd1 = forward_returns(asset_returns, 1)
    cols = signals.columns.intersection(asset_returns.columns)
    idx = signals.index.intersection(asset_returns.index)
    sig = signals.loc[idx, cols].to_numpy(dtype=float)
    ff = (signals.where(signals.abs() > 0.0).ffill()
          .loc[idx, cols].to_numpy(dtype=float))
    ret = asset_returns.loc[idx, cols].to_numpy(dtype=float)
    ma = (asset_returns.rolling(SMEAR_CONTROL_MA, min_periods=1).mean()
          .loc[idx, cols].to_numpy(dtype=float))
    if sig.shape[0] < 2:
        return None
    row_dates = idx[1:]     # decision date of row t across every horizon
    s0, s1 = sig[1:], sig[:-1]
    f1 = np.nan_to_num(ff[:-1], nan=0.0)   # pre-first-nonzero rows: no info
    r0, m0 = ret[1:], ma[1:]
    base = (np.isfinite(s0) & np.isfinite(s1) & np.isfinite(r0)
            & np.isfinite(m0))

    means: dict[int, float] = {}
    ses: dict[int, float] = {}
    series: dict[int, pd.Series] = {}
    n_degenerate_by_horizon: dict[int, int] = {}
    all_degenerate = True
    for h in range(1, h_max + 1):
        f = fwd1.shift(-(h - 1)).loc[idx, cols].to_numpy(dtype=float)[1:]
        mask = base & np.isfinite(f)
        cnt = mask.sum(axis=1)
        ok = cnt >= _MIN_NAMES_PARTIAL
        vals = pd.Series(dtype=float)
        degen = np.zeros(len(row_dates), dtype=bool)
        if ok.any():
            ry = _masked_ranks(s0, mask)
            X = np.stack([_masked_ranks(s1, mask),
                          _masked_ranks(f1, mask),
                          _masked_ranks(r0, mask),
                          _masked_ranks(m0, mask)], axis=-1)
            rf = _masked_ranks(f, mask)
            cntf = np.maximum(cnt, 1).astype(float)[:, None]
            ry = np.where(mask, ry - ry.sum(axis=1)[:, None] / cntf, 0.0)
            rf = np.where(mask, rf - rf.sum(axis=1)[:, None] / cntf, 0.0)
            X = np.where(mask[:, :, None],
                         X - X.sum(axis=1)[:, None, :] / cntf[:, :, None],
                         0.0)
            G = np.einsum("taj,tak->tjk", X, X)
            b = np.einsum("taj,ta->tj", X, ry)
            beta = np.einsum("tjk,tk->tj", np.linalg.pinv(G), b)
            resid = np.where(mask, ry - np.einsum("taj,tj->ta", X, beta), 0.0)
            rvar = (resid ** 2).sum(axis=1) / np.maximum(cnt - 1, 1)
            degen = ok & (rvar < _DEGENERATE_VAR)
            den = np.sqrt((resid ** 2).sum(axis=1) * (rf ** 2).sum(axis=1))
            good = ok & ~degen & (den > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                ic = np.where(good, (resid * rf).sum(axis=1) / den, np.nan)
            # Index by decision date so pooled horizons align the same calendar
            # observations even when each horizon has different missing dates.
            vals = pd.Series(ic[good], index=row_dates[good])
        n_degenerate_h = int(degen.sum())
        n_degenerate_by_horizon[h] = n_degenerate_h
        if len(vals) >= SMEAR_MIN_DATES:
            means[h] = float(vals.mean())
            ses[h] = _nw_se(vals)
            series[h] = vals
            all_degenerate = False
        elif len(vals) == 0 and n_degenerate_h >= SMEAR_MIN_DATES:
            # Exact rank explanation is positive evidence of no remaining
            # innovation, but it is horizon-specific.  Keep its calendar
            # support in the pooled intersection instead of letting the h=1
            # count prematurely certify every later horizon.
            means[h] = 0.0
            ses[h] = 0.0
            series[h] = pd.Series(0.0, index=row_dates[degen], dtype=float)
        else:
            # A mix of many degenerate rows and a small non-degenerate sample
            # is not a clean degeneracy result: those few innovations are
            # exactly where intermittent smeared leakage can hide.
            return None
    mass = float(sum(abs(m) for m in means.values()))
    null_mass = float(np.sqrt(2.0 / np.pi) * sum(ses.values()))
    signs = {h: (1.0 if means[h] >= 0 else -1.0) for h in means}
    # calendar alignment: dropna keeps the dates all horizons measured
    aligned = pd.concat([series[h] * signs[h] for h in series],
                        axis=1, sort=True).dropna()
    n_common = int(len(aligned))
    tstat = (0.0 if all_degenerate else
             (float(newey_west_tstat(aligned.sum(axis=1))) if n_common
              else float("nan")))
    return dict(mass=mass, null_mass=null_mass, excess=mass - null_mass,
                tstat=tstat, means=means,
                n_degenerate=n_degenerate_by_horizon.get(1, 0),
                n_degenerate_by_horizon=n_degenerate_by_horizon,
                all_degenerate=all_degenerate,
                n_dates=int(len(series[1])), n_common_dates=n_common)


def _smeared_forward_ic(artifacts: BacktestArtifacts,
                        config: AuditConfig) -> CheckResult:
    check = CHECK_SMEAR
    h_max = int(config.deferred_ic_max_horizon) + SMEAR_EXTRA_HORIZONS
    st = _innovation_forward_ic(artifacts.signals, artifacts.asset_returns,
                                h_max)
    if st is None:
        # The statistic never ran (no horizon reached the date floor and the
        # degenerate-date count cannot vouch either) - the library contract:
        # an unmeasurable check SKIPs, it does not silently pass (same as
        # its siblings deferred_ic_spike/same_bar_bleed).
        return skipped(
            check,
            f"too few usable innovation-IC dates (need >= {SMEAR_MIN_DATES} "
            f"per horizon out to h={h_max}, with >= {_MIN_NAMES_PARTIAL} "
            f"jointly finite names per date) - extend the overlap of "
            f"signals and asset_returns (roughly >= "
            f"{SMEAR_MIN_DATES + h_max + 1} jointly covered dates) and/or "
            f"widen the universe to >= {_MIN_NAMES_PARTIAL} jointly finite "
            f"names per date to enable this check.",
            smear_min_dates=SMEAR_MIN_DATES,
            min_names_partial=_MIN_NAMES_PARTIAL, h_max=h_max)
    profile = {str(h): round(v, 4) for h, v in st["means"].items()}
    details = dict(ic_mass=round(st["mass"], 4),
                   null_mass=round(st["null_mass"], 4),
                   excess_mass=round(st["excess"], 4),
                   nw_tstat=(round(st["tstat"], 1)
                             if np.isfinite(st["tstat"]) else None),
                   innovation_ic_by_horizon=profile, h_max=h_max,
                   n_dates=st["n_dates"], n_degenerate=st["n_degenerate"],
                   n_degenerate_by_horizon={
                       str(h): int(n) for h, n in
                       st["n_degenerate_by_horizon"].items()},
                   n_common_dates=st["n_common_dates"],
                   mass_warn=SMEAR_MASS_WARN, mass_fail=SMEAR_MASS_FAIL,
                   tstat_gate=SMEAR_TSTAT)
    if st["n_common_dates"] < SMEAR_MIN_DATES:
        # The pooled t-stat needs common dates across every horizon. Adequate
        # support within separate horizons does not guarantee enough overlap.
        return skipped(
            check,
            f"only {st['n_common_dates']} calendar dates are jointly "
            f"measurable across all {h_max} horizons (>= {SMEAR_MIN_DATES} "
            f"needed for the pooled NW t-stat co-gate, though each horizon "
            f"alone has >= {SMEAR_MIN_DATES}) - interior all-NaN "
            f"return/signal dates fragment the horizon overlap. Fill or "
            f"drop the gapped dates (or extend the sample) so the horizons "
            f"share >= {SMEAR_MIN_DATES} common dates to enable this check.",
            details=details)
    if st["all_degenerate"] and st["mass"] == 0.0:
        return passed(
            check,
            f"the signal cross-section is rank-explained by its own lag, "
            f"the same-bar return and the trailing mean at every horizon "
            f"1..{h_max} on at least {SMEAR_MIN_DATES} dates each "
            f"({st['n_common_dates']} common dates) - no innovation exists "
            f"to carry smeared forward information",
            details=details)
    # Use the displayed detail value when formatting mass, keeping the
    # message and details consistent at rounding boundaries.
    excess, tstat, mass = st["excess"], st["tstat"], details["ic_mass"]
    remediation = (
        "The signal's day-to-day innovation carries forward-return "
        "information integrated across horizons - the smeared form of a "
        "lookahead leak (a feature averaging future bars, a label window "
        "smeared over t+1.., a slowly-contaminated feature store). Profile "
        "each feature's innovation IC by horizon (residualize on the "
        "feature's own lag and same-day observables first), find the ones "
        "still predicting many bars out, re-lag or remove them, and "
        "regenerate signals.")
    if excess >= SMEAR_MASS_FAIL and tstat >= SMEAR_TSTAT:
        return failed(
            check,
            f"the signal INNOVATION (residualized on its own lag, its last "
            f"nonzero value, the same-bar return and the trailing "
            f"{SMEAR_CONTROL_MA}-bar mean) "
            f"predicts forward returns with cumulative |IC| mass "
            f"{mass:.3f} over horizons 1..{h_max} ({excess:.3f} above the "
            f"noise mass {st['null_mass']:.3f}; pooled NW t {tstat:.0f}; "
            f"per-horizon: {profile}). The excess mass and pooled t "
            f"clear the diagnostic thresholds after controlling for "
            f"observed past information; inspect the feature pipeline "
            f"for forward-window inputs.",
            severity=Severity.CRITICAL, remediation=remediation, details=details)
    if excess >= SMEAR_MASS_WARN and tstat >= SMEAR_TSTAT:
        return warned(
            check,
            f"innovation forward-IC mass {mass:.3f} over horizons "
            f"1..{h_max} ({excess:.3f} above the noise mass; pooled NW t "
            f"{tstat:.0f}; per-horizon: {profile}) clears the warning "
            f"mass and t thresholds - verify no feature integrates "
            f"future bars and assess whether the controls represent "
            f"the signal's lawful information.",
            severity=Severity.HIGH, remediation=remediation, details=details)
    return passed(
        check,
        f"innovation forward-IC mass {mass:.3f} over horizons 1..{h_max} "
        f"(excess {excess:.3f} over the noise mass {st['null_mass']:.3f}) "
        f"is within honest reach - no smeared forward-information "
        f"signature",
        details=details)


# ---------------------------------------------------------------------------
# 8. lookahead.same_bar_return_loading
# ---------------------------------------------------------------------------

CHECK_RETURN_LOADING = "lookahead.same_bar_return_loading"

#: Absolute mean correlation of magnitude-preserving position residuals
#: with same-bar return ranks. Additive return-based position tilts can
#: preserve rank order, making the magnitude channel necessary alongside
#: rank-based timing checks. Lawful secondary sleeves can also contribute
#: correlation, so the mean and HAC t thresholds are both required.
#: This channel cannot detect a stamped signal bleed orthogonal to the
#: same-bar returns; same_bar_bleed covers that separate signal channel.
#: A tilt of leak * rank_demean(r_t) added to honest weights harvests
#: same-bar dispersion without reordering the book, so rank detectors see
#: nothing at any leak size, while the raw residual's per-date corr with
#: return ranks sits near +0.9 at every tested leak size. Honest undeclared
#: overlays (vol scaling, t-1 side-tilts, second signals, decile/coarse
#: books) measure |mean| <= 0.035; the bars reuse
#: config.predictive_ic_warn/fail (0.08/0.15), >= 2.3x that ceiling. An
#: r_t-excluding score bled at 30% loads only ~+0.01 here - that is
#: same_bar_bleed's channel.
RETURN_LOADING_WARN = 0.08
RETURN_LOADING_FAIL = 0.15

#: Persistence-based WARN escalation for diluted return loading. A t-stat
#: alone grows with sample depth and breadth even for genuine lagged alpha,
#: so escalation additionally requires a mean above the effect-size floor.
#: Only the full-support band qualifies. Persistent dependence below the
#: mean floor is reported as unresolved, leaving sufficiently diluted leaks
#: indistinguishable from a lawful secondary alpha sleeve.
#: Diluted magnitude leaks (a 2% uniform tilt under an inverse-vol overlay,
#: or a leak live on ~7% of dates) measure mean 0.045-0.07 at |NW t| 7-26 -
#: under the level bars. Honest residuals without genuine alpha reach |t|
#: ~3.3, but an undeclared sleeve with genuine t-1 alpha (true IC 0.03-0.05)
#: reaches |t| 5-15 on 10y x 40-60-name panels because t grows ~
#: alpha*sqrt(dates*names); its mean level stays <= ~0.052 across panel
#: geometries while the diluted leaks measure >= ~0.06, so the floor sits
#: at 0.055. Leaks diluted below it with |t| >= 6 are reported unresolved,
#: not WARNed.
RETURN_LOADING_TSTAT_ESC = 6.0
RETURN_LOADING_ESC_MEAN_FLOOR = 0.055

#: A dominant time-static per-asset tilt can inflate residual variance and
#: hide time-varying return loading. Estimate its direction from the mean
#: unit-gross residual and project it out per date. When it explains at
#: least this share, judge the remaining correlation against the level bars.
#: The direction estimate uses the supplied sample; this is a diagnostic
#: projection, not a tradable estimator. Slowly rotating tilts may evade the
#: time-mean estimate and remain a limitation.
#: A leak hidden under a ~90%-of-gross static per-asset tilt deflates the
#: raw corr and its NW t together (both scale ~ leak/tilt) to mean ~0.03 /
#: t ~4-5. With the static direction projected out the leak cells measure
#: purged corr ~+0.9 while honest static-dominated books (static-only,
#: static + fast alpha sleeve, static + vol overlay, year-block rotations)
#: leave <= ~0.04 - under the 0.08 WARN bar.
RETURN_LOADING_STATIC_SHARE = 0.50


def _same_bar_return_loading(artifacts: BacktestArtifacts,
                             config: AuditConfig) -> CheckResult:
    check = CHECK_RETURN_LOADING
    if artifacts.positions is None:
        return skipped(check, "pass artifacts.positions (weights held during "
                              "each period) to enable the same-bar "
                              "return-loading detector")
    lag = int(artifacts.signal_lag)
    if lag < 1:
        return skipped(check, "invalid artifacts.signal_lag: qaudit's fixed "
                              "end-of-bar timing contract requires an integer "
                              ">= 1; validate the artifacts and remap any "
                              "pre-bar feed to the prior end-of-bar row")
    pos = artifacts.positions.to_numpy(dtype=float)
    sig = artifacts.signals.to_numpy(dtype=float)
    ret = artifacts.asset_returns.to_numpy(dtype=float)
    n = pos.shape[0]
    vals = []
    n_explained = 0
    # Retain per-date residuals and unit-gross positions for the static
    # component projection.
    resid_rows: list[np.ndarray] = []
    unit_rows: list[np.ndarray] = []
    row_ts: list[int] = []
    for t in range(lag, n):
        p, sl, r0 = pos[t], sig[t - lag], ret[t]
        # Residualize over the book's own ranking set - every finite
        # position with a finite lag-L signal (zero positions stay in:
        # masking to the traded subset selects on the lagged signal, same
        # rationale as same_bar_bleed). Restricting the fit to the
        # return-finite subset would re-rank the cross-section and leave an
        # honest exact-rank book with a spurious kink residual on
        # universe-transition dates.
        fit_mask = np.isfinite(p) & np.isfinite(sl)
        eval_mask = fit_mask & np.isfinite(r0)
        if int(eval_mask.sum()) < _MIN_NAMES_PARTIAL:
            continue
        pm = p[fit_mask]
        gross = float(np.abs(pm).sum())
        rm = r0[eval_mask]
        if gross <= 0.0 or np.ptp(rm) == 0:
            n_explained += 1            # flat book / degenerate return date
            continue
        # Raw-weight residual, unit-gross normalized (scale-free): a rank
        # residual is exactly zero for any tilt that does not reorder the
        # book, which is precisely how the magnitude leak hides.
        rp_fit = _piecewise_rank_residual(pm / gross, sl[fit_mask],
                                          rank_y=False)
        rp = rp_fit[eval_mask[fit_mask]]
        if rp.var() < _DEGENERATE_VAR:
            n_explained += 1            # book fully explained by lag-L signal
            continue
        rr = sps.rankdata(rm).astype(float)
        rr -= rr.mean()
        r = float(np.corrcoef(rp, rr)[0, 1])
        if np.isfinite(r):
            vals.append(r)
            full = np.full(pos.shape[1], np.nan)
            full[fit_mask] = rp_fit
            resid_rows.append(full)
            ufull = np.full(pos.shape[1], np.nan)
            ufull[fit_mask] = pm / gross
            unit_rows.append(ufull)
            row_ts.append(t)
    series = pd.Series(vals)
    if not len(series) and not n_explained:
        return skipped(
            check,
            f"no usable date for the return-loading statistic (need >= "
            f"{_MIN_NAMES_PARTIAL} assets per date with finite position, "
            f"lag-{lag} signal and same-bar return) - extend the overlap "
            f"of positions, signals and asset_returns to enable this "
            f"check.",
            n_usable=0, n_explained=0)
    if len(series) < _MIN_PARTIAL_DATES_SHORT:
        if n_explained >= _MIN_PARTIAL_DATES:
            # Explained-dominance: magnitudes carry no residual channel on
            # the overwhelming share of dates; the stray remainder
            # (typically universe-transition cross-sections) is reported,
            # not judged.
            stray = (f"; the {len(series)} remaining date(s) carry mean "
                     f"corr {float(series.mean()):+.2f} vs same-bar return "
                     f"ranks - too few to judge"
                     if len(series) else "")
            return passed(
                check,
                f"position magnitudes are fully explained by the declared "
                f"lag-{lag} signal on {n_explained} dates{stray}",
                n_usable=int(len(series)), n_explained=n_explained,
                stray_mean_corr=(float(series.mean()) if len(series)
                                 else None))
        return skipped(
            check,
            f"only {len(series)} date(s) show residual magnitude variation "
            f"beyond the declared lag-{lag} signal, and the "
            f"{n_explained} fully-explained date(s) are below the "
            f"{_MIN_PARTIAL_DATES}-date floor for that evidence to vouch "
            f"alone (need >= {_MIN_PARTIAL_DATES_SHORT} judged dates) - "
            f"extend the overlap of positions, signals and asset_returns "
            f"to enable this check.",
            n_usable=int(len(series)), n_explained=n_explained)
    mean = float(series.mean())
    tstat = float(newey_west_tstat(series))
    # small-sample co-gate, mirrored from same_bar_bleed: the short band
    # demands double the t evidence (the magnitude leak measures |t| in the
    # hundreds at any sample size that reaches the judge).
    t_gate = config.bleed_tstat * (
        2.0 if len(series) < _MIN_PARTIAL_DATES else 1.0)
    # Estimate a static direction from the time-mean unit-gross residual,
    # project it out per date, and measure remaining return-rank correlation.
    # This diagnostic removes variance contributed by an inert basket.
    static_share = float("nan")
    p_mean = float("nan")
    p_t = float("nan")
    p_vals: list[float] = []
    if resid_rows:
        u_static = np.nanmean(np.stack(resid_rows), axis=0)
        shares: list[float] = []
        for full in resid_rows:
            msk = np.isfinite(full)
            x = full[msk]
            u = u_static[msk]
            uc = u - u.mean()
            den = float(uc @ uc)
            resid = x - x.mean()
            if den > 0:
                resid = resid - (float(resid @ uc) / den) * uc
            vx = float(np.var(x))
            shares.append(float(np.clip(1.0 - float(resid.var()) / vx,
                                        0.0, 1.0)) if vx > 0 else 0.0)
        static_share = float(np.mean(shares)) if shares else float("nan")
        if static_share >= RETURN_LOADING_STATIC_SHARE:
            # The residual is mostly one static basket. Its variance also
            # contaminates the piecewise bin fits (bin means/slopes absorb
            # tilt noise, leaving base-book leftovers that dilute the
            # purged corr), so the clean measurement strips the fitted
            # static component from the unit-gross positions first and
            # re-runs the piecewise residualization on what remains.
            for t, ufull in zip(row_ts, unit_rows):
                fmask = np.isfinite(ufull)
                x = ufull[fmask]
                u = u_static[fmask]
                uc = u - u.mean()
                den = float(uc @ uc)
                if den > 0:
                    x = x - (float((x - x.mean()) @ uc) / den) * uc
                slv = sig[t - lag][fmask]
                rp2 = _piecewise_rank_residual(x, slv, rank_y=False)
                emask = np.isfinite(ret[t]) & fmask
                rp2 = rp2[emask[fmask]]
                if rp2.var() < _DEGENERATE_VAR:
                    continue      # purely static book on this date
                rm2 = ret[t][emask]
                if np.ptp(rm2) == 0:
                    continue
                rr2 = sps.rankdata(rm2).astype(float)
                rr2 -= rr2.mean()
                pr = float(np.corrcoef(rp2, rr2)[0, 1])
                if np.isfinite(pr):
                    p_vals.append(pr)
            if len(p_vals) >= _MIN_PARTIAL_DATES_SHORT:
                p_mean = float(pd.Series(p_vals).mean())
                p_t = float(newey_west_tstat(pd.Series(p_vals)))
    details = dict(mean_corr=mean, nw_tstat=tstat,
                   n_usable=int(len(series)), n_explained=n_explained,
                   declared_lag=lag, corr_warn=RETURN_LOADING_WARN,
                   corr_fail=RETURN_LOADING_FAIL, tstat_gate=t_gate,
                   static_share=static_share,
                   static_purged_mean_corr=p_mean,
                   static_purged_nw_tstat=p_t,
                   n_static_purged=int(len(p_vals)),
                   esc_mean_floor=RETURN_LOADING_ESC_MEAN_FLOOR)
    core = (f"corr of the raw position residual (positions minus their "
            f"lag-{lag}-signal explanation, magnitudes preserved) with the "
            f"same-bar return ranks = {mean:+.3f}, NW t {fmt_tstat(tstat)} over "
            f"{len(series)} dates")
    remediation = (
        "Part of the book's MAGNITUDE is built from the returns of the bar "
        "it trades - look for a position optimizer fed contemporaneous "
        "realized returns as expected returns, a same-bar join at the "
        "position-construction stage, or an overlay computed on the "
        "holding-period bar; every input into positions.loc[t] must be "
        "observable by the close of t-signal_lag.")
    if abs(mean) >= RETURN_LOADING_FAIL and abs(tstat) >= t_gate:
        return failed(
            check,
            core + " - position magnitudes load on the very returns they "
                   "earn; an honest residual is a t-1 alpha whose per-date "
                   "IC never sustains this level",
            severity=Severity.CRITICAL, remediation=remediation,
            details=details)
    if abs(mean) >= RETURN_LOADING_WARN and abs(tstat) >= t_gate:
        return warned(
            check,
            core + " - sustained same-bar return alignment above honest "
                   "alpha reach; verify no position input touches the "
                   "holding-period bar",
            severity=Severity.HIGH, remediation=remediation, details=details)
    # When a static basket dominates the residual, judge the projected
    # remainder against the ordinary correlation thresholds.
    if (np.isfinite(static_share)
            and static_share >= RETURN_LOADING_STATIC_SHARE
            and len(p_vals) >= _MIN_PARTIAL_DATES_SHORT
            and np.isfinite(p_mean) and abs(p_mean) >= RETURN_LOADING_WARN
            and abs(p_t) >= t_gate):
        sev = (Severity.CRITICAL if abs(p_mean) >= RETURN_LOADING_FAIL
               else Severity.HIGH)
        verdict = failed if sev is Severity.CRITICAL else warned
        return verdict(
            check,
            core + f" - but {static_share:.0%} of the residual is one "
                   f"TIME-STATIC basket (a fixed characteristic tilt), "
                   f"and with that inert direction projected out the "
                   f"remaining dynamic residual loads on the very returns "
                   f"it earns (corr {p_mean:+.3f}, NW t {fmt_tstat(p_t)} over "
                   f"{len(p_vals)} dates vs the "
                   f"{RETURN_LOADING_WARN:.2f}/{RETURN_LOADING_FAIL:.2f} "
                   f"bars) - a static tilt is F(t-1)-constructible by "
                   f"definition, so it can only DILUTE a same-bar leak's "
                   f"correlation, never explain it",
            severity=sev, remediation=remediation, details=details)
    # Require both persistence and an effect-size floor. Honest secondary
    # alpha also gains t significance with sample size; subfloor alignment
    # is unresolved rather than sufficient evidence of a timing violation.
    if (len(series) >= _MIN_PARTIAL_DATES
            and abs(tstat) >= RETURN_LOADING_TSTAT_ESC):
        if abs(mean) >= RETURN_LOADING_ESC_MEAN_FLOOR:
            return warned(
                check,
                core + f" - the mean sits below the "
                       f"{RETURN_LOADING_WARN:.2f} level bar, but the "
                       f"alignment is both PERSISTENT (|NW t| "
                       f"{fmt_tstat(abs(tstat))} vs the "
                       f"{RETURN_LOADING_TSTAT_ESC:.1f} escalation gate) "
                       f"and ABOVE the effect-size floor (|mean| "
                       f">= {RETURN_LOADING_ESC_MEAN_FLOOR:.3f}). This "
                       f"can indicate a diluted magnitude leak: a risk "
                       f"overlay inflates residual variance and an "
                       f"intermittent leak thins the date-mean. A lawful "
                       f"secondary alpha can also contribute loading. If the "
                       f"book holds an undeclared sleeve with "
                       f"extraordinary genuine alpha, declare it (or the "
                       f"composite) as artifacts.signals to resolve",
                severity=Severity.HIGH, remediation=remediation,
                details=details)
        return passed(
            check,
            core + f" - the same-bar alignment is persistent (|NW t| "
                   f"{fmt_tstat(abs(tstat))} >= "
                   f"{RETURN_LOADING_TSTAT_ESC:.1f}) but its level sits "
                   f"BELOW the effect-size floor (|mean| < "
                   f"{RETURN_LOADING_ESC_MEAN_FLOOR:.3f}), inside the band "
                   f"a lawful secondary alpha occupies: an undeclared "
                   f"honest sleeve with genuine t-1 alpha produces exactly "
                   f"this on ordinary multi-year panels (its NW t grows ~ "
                   f"alpha*sqrt(dates*names)), so persistence alone "
                   f"cannot separate it from a heavily diluted magnitude "
                   f"leak - treat as UNRESOLVED, not exonerated: declare "
                   f"undeclared sleeves (or the composite) as "
                   f"artifacts.signals to resolve",
            details=details, unresolved=True)
    if abs(mean) >= RETURN_LOADING_WARN:
        # A level threshold without the t co-gate is unresolved, so report the
        # measurement without claiming absence of return loading.
        return passed(
            check,
            core + f" - |mean| clears the {RETURN_LOADING_WARN:.2f} level "
                   f"bar but the NW t evidence ({fmt_tstat(tstat)}, need |t| >= "
                   f"{t_gate:.1f}) is insufficient to convict - not "
                   f"exonerated: clustered leak dates deflate the NW t; "
                   f"treat as unresolved rather than clean.",
            details=details, unresolved=True)
    return passed(
        check,
        core + " - no same-bar return loading in position magnitudes "
               "(honest undeclared overlays measure at their t-1 alpha "
               "level, ~0.00-0.04 here)",
        details=details)
