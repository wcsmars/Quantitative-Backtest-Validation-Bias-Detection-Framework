"""Suspicious performance, overfitting, and IC-stability checks.

Check ids
---------
- ``performance.suspicious_sharpe``       (HIGH)   |annualized Sharpe| too extreme to be real
                                                   (negative = sign-flip/inverted convention,
                                                   after cost drag is explained away via the
                                                   gross reconstruction)
- ``performance.deflated_sharpe``         (HIGH)   Sharpe deflated for ``n_trials`` selection
- ``performance.suspicious_ic``           (HIGH)   mean per-date IC vs forward returns too high
                                                   (bars scale with label_horizon; SKIPs below
                                                   30 usable IC dates; breadth/depth noise floor)
- ``performance.ic_stability``            (MEDIUM) rolling-IC sign consistency over time
- ``performance.ic_regime_concentration`` (MEDIUM) one calendar year carrying the IC
- ``performance.sample_size``             (LOW)    enough periods to power the statistics
                                                   (both the common calendar and the finite
                                                   observations of the Sharpe-check series)

Strategy-return source for the Sharpe-based checks: ``artifacts.strategy_returns``
(net) when present; otherwise a gross reconstruction from
``positions x asset_returns`` (flagged as "gross reconstruction" in messages);
otherwise those checks SKIP and say what to pass. suspicious_sharpe SKIPs
below ``_SR_MIN_OBS`` observations and judges a zero-variance series at the
check level (constant nonzero -> FAIL, identically zero -> SKIP) instead of
letting a NaN Sharpe read as a clean SKIP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._stats import (annualized_sharpe, cross_sectional_ic, deflated_sharpe_ratio,
                      effective_sample_size, fmt_tstat, forward_returns,
                      gross_strategy_returns, newey_west_tstat,
                      probabilistic_sharpe_ratio, traded_dollars_series)
from ..types import CheckResult, Severity, failed, passed, skipped, warned
from .costs import MAX_PLAUSIBLE_ONE_WAY_BPS, MIN_PLAUSIBLE_ONE_WAY_BPS
from .lookahead import MIN_IC_DATES

# Stability and regime checks judge an edge only above the larger of an
# absolute IC floor and _IC_MOOT_GATE_Z times the breadth/depth noise SE.
_IC_STABILITY_MIN_ABS_IC = 0.005

# Minimum rolling windows and calendar years needed to judge consistency.
_IC_STABILITY_MIN_WINDOWS = 4

_MIN_YEARS_FOR_REGIME = 3

_REGIME_MIN_ABS_MEAN_IC = 0.005

# Noise-SE multiplier for the stability and regime gates: 2.5 SE is a
# ~1% two-sided false-positive rate under the no-edge null, while an edge
# at the bottom of the realistic band (|mean IC| 0.01) already sits at
# ~1.5 SE at 30 names x 750 days and further out with more data.
_IC_MOOT_GATE_Z = 2.5

# Daily-equity IC reference band used in report wording; not a guarantee.
_REALISTIC_IC_BAND = "0.01-0.05"

_REALISTIC_IC_LO = 0.01

_REALISTIC_IC_HI = 0.05

# Mean IC has approximate SE 1/sqrt((names - 1) * independent dates);
# overlapping label windows reduce the independent-date count. The
# level bars are lifted to at least Z x SE (Z_WARN 3.0 ~ 0.3%, Z_FAIL
# 4.5 ~ 7e-6 two-sided null tail) on top of the MIN_IC_DATES floor: at
# 31 usable dates x 5 names the SE is ~0.09, so the fixed 0.08/0.15 bars
# alone sit at only ~0.9/1.7 SE from the no-edge null, while an embedded
# target (|IC| ~0.9) still clears the lifted bars by 2-4x.
_IC_NOISE_Z_WARN = 3.0

_IC_NOISE_Z_FAIL = 4.5

# Scale daily IC bars by min(sqrt(label_horizon), cap). sqrt(h) is the
# theoretical ceiling for honest growth (an alpha equally predictive of
# every sub-period); decaying real alphas grow less (~1.7-1.9x at h=5-21
# on the bundled honest synthetic market). Cap 2.5 leaves margin above
# that (strongest honest mean IC measured ~0.22 at h=10/21 vs a scaled
# fail bar of 0.375) while an embedded h-period target (IC ~0.9) trips
# any scaled bar; h=1 is unchanged. A ratio-to-IC1 shape test is no
# substitute: an h-window embed inflates IC1 by ~IC_h/sqrt(h),
# reproducing the honest growth shape, so strong honest long-horizon
# books that WARN are adjudicated by hand.
_IC_HORIZON_GROWTH_CAP = 2.5

# Sparse/event-gated exports: cross_sectional_ic drops constant or
# <5-name dates, so the mean IC is conditioned on exactly the dates an
# honest scheduled alpha bets on (conditional IC ~0.10-0.20 where the
# all-dates-diluted equivalent is ~0.02-0.04). Usable-date fraction
# <= 0.35 marks a gated export: measured gated stampings sit at ~0.05
# (monthly) to ~0.20 (weekly, turn-of-month), continuous exports at
# >= ~0.75 even with a 60-bar warmup on 250 bars. A gated book at or
# above the bars gets a scoped WARN, never the continuous-book FAIL
# wording; a continuous book misclassified as gated is only softened
# FAIL -> WARN, never silenced. Above _GATED_EMBED_IC_FAIL x the horizon
# multiplier the plain FAIL still fires: an embedded target's conditional
# IC is ~0.9 and no honest gated alpha measured above ~0.2 (>= 2.5x
# margin). Detector heuristics, not universal calibration.
_GATED_USABLE_FRAC_MAX = 0.35

_GATED_EMBED_IC_FAIL = 0.5

# Negative net Sharpe can reflect transaction-cost drag: per-date
# (gross - net) must track dollars traded x a plausible one-way cost with
# rms residual <= this fraction of the drag's own rms scale. Genuinely
# costed books reconcile at machine precision (~1e-19) and a net series
# unrelated to the positions misses by >100x, so 0.25 only absorbs
# NaN-masking noise and per-name cost dispersion.
_DRAG_RECONCILE_REL_TOL = 0.25

# Minimum overlap for cost-drag reconciliation.
_DRAG_MIN_OBS = 30

# Below 30 finite returns, skip Sharpe judgments: random 5-29-bar slices
# of the bundled honest SR-1.3 book clear the 5.0 FAIL bar a fifth to
# half of the time, a false-positive class the level bars are not
# calibrated for. 30 mirrors the PSR/DSR n >= 30 floor and _DRAG_MIN_OBS.
# Crossing it does not establish power (honest FAIL-bar rate is still
# ~19% at n = 30, ~2% at 60): sample_size also compares the usable
# history with the configured min_periods.
_SR_MIN_OBS = 30

# Treat dispersion this small relative to the absolute mean as constant.
# This absorbs floating-point dust; near-constant series with larger
# dispersion are judged by the ordinary Sharpe thresholds.
_ZERO_VARIANCE_REL_TOL = 1e-12

_SR_SOURCE_SKIP_MSG = (
    "Sharpe-based check needs portfolio returns: pass artifacts.strategy_returns "
    "(net per-period portfolio returns) or artifacts.positions (weights held per "
    "period, which enables a gross reconstruction via positions x asset_returns)"
)


def run(artifacts, config, *, signal_func=None, backtest_func=None) -> list[CheckResult]:
    """Run the performance / overfitting / IC-stability family. Never mutates
    ``artifacts``; input frames arrive pre-aligned by ``api.audit``."""
    strat, source = _strategy_returns_source(artifacts)
    ic = _per_date_ic(artifacts)
    return [
        _suspicious_sharpe(strat, source, artifacts, config),
        _deflated_sharpe(strat, source, artifacts, config),
        _suspicious_ic(ic, artifacts, config),
        _ic_stability(ic, artifacts, config),
        _ic_regime_concentration(ic, artifacts, config),
        _sample_size(strat, source, artifacts, config),
    ]


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------

def _strategy_returns_source(artifacts) -> tuple[pd.Series | None, str | None]:
    """Best available per-period strategy-return series and a label naming it."""
    if artifacts.strategy_returns is not None:
        return artifacts.strategy_returns, "net strategy_returns"
    if artifacts.positions is not None:
        gross = gross_strategy_returns(artifacts.positions, artifacts.asset_returns)
        return gross, "gross reconstruction from positions x asset_returns"
    return None, None


def _net_return_span_gap(strat, source) -> dict | None:
    """Describe missing net returns strictly inside their finite span.

    ``BacktestArtifacts.validate`` rejects this shape at the public intake
    boundary.  The check-level guard is defence in depth for callers that
    invoke this module directly with an already-aligned artifact: silently
    dropping interior NaNs would otherwise let selected dates determine both
    Sharpe verdicts.  Leading/trailing NaNs are valid truncation and ignored.
    Gross reconstructions are outside the strategy_returns density contract.
    """
    if source != "net strategy_returns":
        return None
    s = pd.Series(strat)
    finite = np.isfinite(s.to_numpy(dtype="float64", na_value=np.nan))
    positions = np.flatnonzero(finite)
    if positions.size < 2:
        return None
    first, last = int(positions[0]), int(positions[-1])
    span_finite = finite[first:last + 1]
    missing = int((~span_finite).sum())
    if not missing:
        return None
    gap_index = s.index[first:last + 1][~span_finite]
    return {
        "n_interior_missing": missing,
        "n_reported_span": int(len(span_finite)),
        "first_finite_return": str(s.index[first]),
        "last_finite_return": str(s.index[last]),
        "missing_date_examples": [str(x) for x in gap_index[:3]],
        "returns_source": source,
    }


def _gap_skip(check: str, gap: dict) -> CheckResult:
    return skipped(
        check,
        f"net strategy_returns has {gap['n_interior_missing']} missing/NaN "
        f"date(s) inside its {gap['n_reported_span']}-date finite span "
        f"{gap['first_finite_return']} .. {gap['last_finite_return']} - "
        f"interior gaps can hide losing bars, so this statistic is not "
        f"judged. Report one finite net return on every audited calendar "
        f"date inside the strategy's live span; only leading/trailing "
        f"warmup or truncation gaps may remain.",
        details=gap,
    )


def _per_date_ic(artifacts) -> pd.Series:
    """Per-date Spearman IC of signals vs the legitimate forward-return target."""
    fwd = forward_returns(artifacts.asset_returns,
                          int(artifacts.label_horizon))
    return cross_sectional_ic(artifacts.signals, fwd, method="spearman")


# ---------------------------------------------------------------------------
# 1. performance.suspicious_sharpe
# ---------------------------------------------------------------------------

def _cost_drag_explanation(strat, artifacts, ppy: float):
    """Negative-branch discriminator: can the net book's loss be explained as
    deterministic transaction-cost drag?  Measures the gross reconstruction
    (positions x asset_returns) and reconciles per-period (gross - net)
    against dollars traded x a plausible one-way cost (the declared bps when
    on record, else the implied cost, which must clear the costs-module
    economic-plausibility band). Returns ``(gross_sr, info)``: ``gross_sr``
    is None when the discriminator cannot run (net or positions missing, or
    too little overlap); ``info`` is a dict only when the drag reconciles."""
    if artifacts.strategy_returns is None or artifacts.positions is None:
        return None, None
    gross = gross_strategy_returns(artifacts.positions, artifacts.asset_returns)
    common = gross.index.intersection(strat.index)
    g_all = gross.loc[common]
    n_all = strat.loc[common]
    mask = g_all.notna() & n_all.notna()
    n_common = int(mask.sum())
    if n_common < _DRAG_MIN_OBS:
        return None, None
    g = g_all[mask]
    net = n_all[mask]
    gross_sr = float(annualized_sharpe(g, ppy))
    if not np.isfinite(gross_sr):
        return None, None
    # dollars traded per period, two-sided, first row = entry trade -
    # drift-aware, the _stats.traded_dollars_series convention (trades
    # measured against pre-trade drifted weights, not raw target diffs).
    dollars = traded_dollars_series(artifacts.positions,
                                    artifacts.asset_returns).reindex(g.index)
    # A row after portfolio NAV growth <= 0 has no self-financing pre-trade
    # weights.  Treating that undefined ledger row as zero dollars lets a
    # restarted/wiped book manufacture an exact "cost drag" explanation and
    # turn an extreme negative Sharpe green.  This discriminator is optional:
    # when its ledger is not finite, decline the explanation and let the
    # suspicious-Sharpe verdict stand.
    if not np.isfinite(dollars.to_numpy(dtype=float)).all():
        return gross_sr, None
    drag = g - net
    mean_drag = float(drag.mean())
    mean_dollars = float(dollars.mean())
    declared = artifacts.declared_costs_bps
    if declared is not None and declared > 0:
        c = float(declared) * 1e-4
    else:
        if mean_dollars <= 0 or mean_drag <= 0:
            return gross_sr, None
        c = mean_drag / mean_dollars              # implied one-way cost
    if not (MIN_PLAUSIBLE_ONE_WAY_BPS * 1e-4 <= c
            <= MAX_PLAUSIBLE_ONE_WAY_BPS * 1e-4):
        return gross_sr, None
    resid = (drag - c * dollars).to_numpy(dtype=float)
    rms_resid = float(np.sqrt(np.mean(np.square(resid))))
    scale = c * float(np.sqrt(np.mean(np.square(dollars.to_numpy(dtype=float)))))
    if mean_drag <= 0 or scale <= 0 or rms_resid > _DRAG_RECONCILE_REL_TOL * scale:
        return gross_sr, None
    return gross_sr, dict(implied_one_way_bps=c * 1e4,
                          mean_drag_per_period=mean_drag,
                          mean_dollars_traded=mean_dollars,
                          drag_rms_residual=rms_resid, n_common=n_common)


def _degenerate_series(strat) -> tuple[bool, float, int]:
    """(is_constant, mean, n_obs) of the non-NaN part of ``strat``: a
    zero-variance series (sd == 0, or sd <= _ZERO_VARIANCE_REL_TOL x |mean|
    for float dust) over >= 2 observations. annualized_sharpe returns NaN on
    it, which must not route both Sharpe checks to SKIP and read the family
    CLEAN - a riskless fixed credit is not "undefined", it is unbounded, so
    the verdict is issued here at the check level."""
    x = np.asarray(pd.Series(strat).dropna(), dtype=float)
    n = int(x.size)
    if n < 2:
        return False, float("nan"), n
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    constant = bool(sd == 0.0 or sd <= _ZERO_VARIANCE_REL_TOL * abs(mean))
    return constant, mean, n


def _suspicious_sharpe(strat, source, artifacts, config) -> CheckResult:
    check = "performance.suspicious_sharpe"
    if strat is None:
        return skipped(check, _SR_SOURCE_SKIP_MSG)
    gap = _net_return_span_gap(strat, source)
    if gap is not None:
        return _gap_skip(check, gap)
    # periods_per_year is explicitly a positive real input.  Truncating a
    # sub-annual sampling rate (for example 0.5 observations/year) to zero
    # silently forces every finite Sharpe to 0 and can turn an extreme book
    # into a PASS.
    ppy = float(artifacts.periods_per_year)
    constant, mean, n_obs = _degenerate_series(strat)
    if constant:
        if mean == 0.0:
            # Worded on the source: for the gross reconstruction there was
            # no strategy_returns to blame - every position row is flat.
            return skipped(
                check,
                f"{source} is identically zero over all {n_obs} periods - "
                f"the book never traded (every position row is flat or "
                f"NaN); pass the net-return series of the positions "
                f"actually held",
                n_obs=n_obs, mean_return=0.0, returns_source=source)
        sr_inf = float(np.inf) if mean > 0 else float(-np.inf)
        return failed(
            check,
            f"constant {mean:+.4%}/period return ({source}) with zero "
            f"variance over {n_obs} periods - Sharpe is unbounded; a "
            f"riskless fixed credit/debit is an engine accrual or "
            f"placeholder series, not a strategy",
            severity=Severity.HIGH,
            remediation="pass the net per-period P&L of the actual positions "
                        "(positions x asset_returns minus costs); a constant "
                        "series usually comes from a fee/financing accrual "
                        "column, a placeholder fill, or a returns column that "
                        "was overwritten by its own mean",
            details=dict(sharpe_annualized=sr_inf, mean_return=mean,
                         n_obs=n_obs, periods_per_year=ppy,
                         returns_source=source))
    if n_obs < _SR_MIN_OBS:
        # Not judged: level bars calibrated on multi-year books are noise
        # on a handful of bars (_SR_MIN_OBS mirrors the PSR/DSR n >= 30 floor).
        return skipped(
            check,
            f"only {n_obs} finite return observation(s) ({source}) - below "
            f"the {_SR_MIN_OBS}-period floor at which an annualized Sharpe "
            f"is a measurement rather than noise (random honest 5-29-bar "
            f"slices of the library's own SR-1.3 book clear the default 5.0 "
            f"FAIL bar a fifth to half of the time); not judged. Extend the "
            f"return history - "
            f"deflated_sharpe and PSR apply the same n >= {_SR_MIN_OBS} "
            f"floor, and performance.sample_size reports the shortfall",
            n_obs=n_obs, min_obs=_SR_MIN_OBS, returns_source=source)
    sr = annualized_sharpe(strat, ppy)
    if not np.isfinite(sr):
        return skipped(check,
                       "Sharpe undefined: strategy returns have < 2 non-NaN periods or "
                       "a non-finite dispersion; pass a longer/valid "
                       "artifacts.strategy_returns (or positions) series",
                       n_obs=n_obs, returns_source=source)
    dependence = effective_sample_size(strat)
    n_eff = float(dependence["n_eff"])
    psr0 = float(probabilistic_sharpe_ratio(strat, 0.0, n_eff=n_eff))
    psr_text = (f"{psr0:.3f}" if np.isfinite(psr0)
                else f"undefined ({n_eff:.1f} effective observations)")
    details = dict(sharpe_annualized=float(sr), psr_vs_zero=psr0, n_obs=n_obs,
                   n_eff=n_eff,
                   dependence_variance_ratio=float(dependence["variance_ratio"]),
                   n_dependent_lags=int(dependence["n_dependent_lags"]),
                   periods_per_year=ppy, returns_source=source)
    # "Suspicious" is a statement about |SR| of the gross book: a -6 gross
    # book is exactly as impossible as a +6 one. Net returns are not
    # sign-symmetric - transaction-cost drag is deterministic, so an honest
    # churny no-edge book loses with arbitrary consistency (a 100-name
    # fast-reversal book at 25bp one-way measures net SR ~ -16 with gross ~ 0).
    # The negative branches therefore first try to explain the loss as cost
    # drag via the gross reconstruction; only when that fails - or the gross
    # book is itself beyond the warn bar (a landed sign-flipped leak measures
    # gross SR ~ -80, orders of magnitude apart) - do they diagnose a
    # sign-flipped leak / inverted timing convention.
    gross_sr: float | None = None
    if sr < 0 and abs(sr) >= config.sharpe_warn:
        gross_sr, drag_info = _cost_drag_explanation(strat, artifacts, ppy)
        if gross_sr is not None:
            details["gross_sharpe_annualized"] = gross_sr
        if drag_info is not None and gross_sr is not None \
                and abs(gross_sr) < config.sharpe_warn:
            details.update(drag_info)
            return passed(
                check,
                f"ann net SR {sr:.1f} ({source}) is deterministic cost drag, "
                f"not a sign-flip signature: the gross reconstruction from "
                f"positions x asset_returns has ann SR {gross_sr:+.2f} (inside "
                f"the +/-{config.sharpe_warn:g} honest range) and per-period "
                f"(gross - net) reconciles with dollars traded x "
                f"{drag_info['implied_one_way_bps']:.1f}bps one-way (mean drag "
                f"{drag_info['mean_drag_per_period'] * 1e4:.1f}bp/period over "
                f"{drag_info['n_common']} periods) - no edge, and the book "
                f"churns it away; see costs.turnover_unrealistic and "
                f"costs.cost_sensitivity",
                details=details)
    if abs(sr) >= config.sharpe_fail:
        if sr < 0:
            drag_clause = (
                f"; the gross reconstruction (ann SR {gross_sr:+.1f}) shows "
                f"transaction-cost drag alone does not explain the loss"
                if gross_sr is not None else
                "; deterministic cost drag on a churny book is the alternative "
                "hypothesis - pass artifacts.positions (with strategy_returns) "
                "so the gross reconstruction can separate cost drag from a "
                "sign inversion")
            return failed(
                check,
                f"ann SR {sr:.1f} ({source}) is beyond -{config.sharpe_fail:g} - "
                f"losses this consistent have a mechanical cause: this is the "
                f"signature of a sign-flipped leak or an inverted timing "
                f"convention (PSR vs 0 = {psr_text} over {n_obs} periods)"
                f"{drag_clause}",
                severity=Severity.HIGH,
                remediation="find the sign inversion first: check signal polarity "
                            "(rank direction, long/short legs), then the execution "
                            "convention (positions.loc[t] must earn "
                            "asset_returns.loc[t] using signals.loc[t-signal_lag]); "
                            "a negated leak is still a leak - once the flip is "
                            "found, hunt the lookahead/leakage exactly as for a "
                            "positive-SR book",
                details=details)
        return failed(
            check,
            f"ann SR {sr:.1f} ({source}) for a {ppy:g}-periods/year equity alpha is not "
            f"an edge, it is a bug - sustained SR>{config.sharpe_fail:g} implies "
            f"near-arbitrage (PSR vs 0 = {psr_text} over {n_obs} periods)",
            severity=Severity.HIGH,
            remediation="hunt for lookahead/leakage before anything else: verify the "
                        "execution lag (positions.loc[t] must come from "
                        "signals.loc[t-signal_lag]), check feature windows do not touch "
                        "the label, and confirm costs are charged",
            details=details)
    if abs(sr) >= config.sharpe_warn:
        if sr < 0:
            drag_clause = (
                f" and the gross reconstruction (ann SR {gross_sr:+.1f}) shows "
                f"cost drag alone does not explain the loss"
                if gross_sr is not None else
                " (or deterministic cost drag on a churny book - pass "
                "artifacts.positions with strategy_returns so the gross "
                "reconstruction can separate them)")
            return warned(
                check,
                f"ann SR {sr:.1f} ({source}) is beyond -{config.sharpe_warn:g} - "
                f"strategies do not lose this consistently by accident; treat as a "
                f"sign-flipped leak or inverted execution convention until polarity "
                f"and lag are verified{drag_clause} (PSR vs 0 = {psr_text}, "
                f"{n_obs} periods)",
                severity=Severity.HIGH,
                remediation="verify signal/position polarity and the execution lag "
                            "direction; if the construction really is honest and "
                            "the GROSS book is also extreme, the negated gross "
                            "book is the alpha to audit (negating net returns does "
                            "not refund costs - the toll is paid either way); run "
                            "the lookahead/leakage families and the dynamic probes "
                            "on it, and compare (gross - net) against dollars "
                            "traded x declared costs if turnover is high",
                details=details)
        return warned(
            check,
            f"ann SR {sr:.1f} ({source}) exceeds {config.sharpe_warn:g} - daily equity "
            f"alphas rarely sustain this; treat as presumptively leaky until the lag, "
            f"costs and universe are verified (PSR vs 0 = {psr_text}, {n_obs} periods)",
            severity=Severity.HIGH,
            remediation="re-verify execution lag, transaction costs and point-in-time "
                        "universe; run the lookahead/leakage check families and the "
                        "dynamic probes before trusting this result",
            details=details)
    if sr <= 0:
        return passed(
            check,
            f"ann SR {sr:.2f} ({source}) over {n_obs} periods is not positive and "
            f"|SR| is below the {config.sharpe_warn:g} suspicion threshold - an "
            f"ordinary losing book, nothing suspicious about the level (PSR vs 0 = "
            f"{psr_text})",
            details=details)
    return passed(
        check,
        f"ann SR {sr:.2f} ({source}) over {n_obs} periods is below the "
        f"{config.sharpe_warn:g} suspicion threshold (PSR vs 0 = {psr_text})",
        details=details)


# ---------------------------------------------------------------------------
# 2. performance.deflated_sharpe
# ---------------------------------------------------------------------------

def _deflated_sharpe(strat, source, artifacts, config) -> CheckResult:
    check = "performance.deflated_sharpe"
    if strat is None:
        return skipped(check, _SR_SOURCE_SKIP_MSG)
    if config.n_trials is None:
        return skipped(check,
                       "pass AuditConfig(n_trials=<total configurations ever tried on "
                       "this data, including abandoned ones>) - if you did not count, "
                       "that is itself the finding")
    gap = _net_return_span_gap(strat, source)
    if gap is not None:
        return _gap_skip(check, gap)
    n_trials = int(config.n_trials)
    constant, mean, n_const = _degenerate_series(strat)
    if constant:
        # Same degenerate class as suspicious_sharpe's zero-variance
        # branch: the DSR is undefined on a constant series, but "undefined"
        # must not read as a clean SKIP - the verdict lives next door.
        return skipped(
            check,
            f"deflated Sharpe undefined on a zero-variance series "
            f"(constant {mean:+.4%}/period over {n_const} periods, {source}) "
            f"- see performance.suspicious_sharpe for the verdict on this "
            f"series (a riskless constant return is an accrual or "
            f"placeholder, not a strategy)",
            n_obs=n_const, mean_return=mean, returns_source=source)
    d = deflated_sharpe_ratio(strat, n_trials)
    dsr = float(d["dsr"])
    int_fields = ("n_trials", "n_obs", "n_dependent_lags")
    details = {k: (int(v) if k in int_fields else float(v))
               for k, v in d.items()}
    details.update(returns_source=source)
    if not np.isfinite(dsr):
        if d["n_obs"] >= 30 and d["n_eff"] < 30:
            return skipped(
                check,
                f"deflated Sharpe undefined: serial dependence reduces "
                f"{int(d['n_obs'])} observed return periods to "
                f"{d['n_eff']:.1f} effective independent observations, below "
                f"the required 30; gather more independent history and "
                f"check for repeated or smoothed returns",
                details=details)
        return skipped(check,
                       f"deflated Sharpe undefined: need >= 30 non-NaN return periods "
                       f"with nonzero variance (got {int(d['n_obs'])}); pass a longer "
                       f"strategy_returns/positions history", details=details)
    ann = float(np.sqrt(float(artifacts.periods_per_year)))
    sr_ann = float(d["sr_period"]) * ann
    sr_star_ann = float(d["sr_star_period"]) * ann
    details.update(sr_annualized=sr_ann, sr_star_annualized=sr_star_ann)
    if dsr < config.dsr_fail:
        return failed(
            check,
            f"after {n_trials} trials the expected best luck-only SR is "
            f"{sr_star_ann:.2f} (ann); observed {sr_ann:.2f} ({source}) gives "
            f"DSR {dsr:.2f} - more likely selection than skill",
            severity=Severity.HIGH,
            remediation="shrink the search space and re-validate the chosen "
                        "configuration on genuinely untouched data; keep a registry of "
                        "every trial (including abandoned ones) so n_trials stays honest",
            details=details)
    if dsr < config.dsr_warn:
        return warned(
            check,
            f"DSR {dsr:.2f} < {config.dsr_warn:g}: observed ann SR {sr_ann:.2f} "
            f"({source}) does not clearly beat the expected best of {n_trials} "
            f"luck-only trials ({sr_star_ann:.2f} ann) - cannot rule out selection",
            severity=Severity.HIGH,
            remediation="hold out untouched data for a single confirmatory run, or "
                        "collect more history; do not add trials without recounting "
                        "n_trials",
            details=details)
    return passed(
        check,
        f"DSR {dsr:.3f}: observed ann SR {sr_ann:.2f} ({source}) clears the expected "
        f"best luck-only SR of {sr_star_ann:.2f} (ann) from {n_trials} trials",
        details=details)


# ---------------------------------------------------------------------------
# 3. performance.suspicious_ic
# ---------------------------------------------------------------------------

def _ic_noise_context(ic: pd.Series, artifacts, h: int) -> tuple[float, int]:
    """Breadth and depth behind the mean-IC statistic: median jointly non-NaN
    names per usable IC date, and the count of usable dates whose h-period
    label windows do not overlap (greedy thinning on index positions -
    overlapping targets share returns, serially correlating their noise ICs,
    so only non-overlapping dates count as independent)."""
    fwd = forward_returns(artifacts.asset_returns, h)
    cols = artifacts.signals.columns.intersection(fwd.columns)
    idx = artifacts.signals.index.intersection(fwd.index)
    joint = (artifacts.signals.loc[idx, cols].notna()
             & fwd.loc[idx, cols].notna()).sum(axis=1)
    valid = ic.notna()
    n_med = float(joint.reindex(ic.index[valid]).clip(lower=2).median())
    positions = np.flatnonzero(valid.to_numpy())
    n_eff, last = 0, -h
    for p in positions:
        if int(p) - last >= h:
            n_eff += 1
            last = int(p)
    return n_med, max(n_eff, 1)


def _suspicious_ic(ic, artifacts, config) -> CheckResult:
    check = "performance.suspicious_ic"
    n = int(ic.notna().sum())
    if n == 0:
        return skipped(check,
                       "per-date IC could not be computed on any date - signals and "
                       "asset_returns need >= 5 jointly non-NaN assets per date; check "
                       "column overlap and NaN patterns")
    mean_ic = float(ic.mean())
    t = float(newey_west_tstat(ic))
    h = int(artifacts.label_horizon)
    details = dict(mean_ic=mean_ic, nw_tstat=t, n_dates=n, label_horizon=h,
                   ic_std=float(ic.std()), min_ic_dates=MIN_IC_DATES)
    if n < MIN_IC_DATES:
        # Sparse-stamped books (monthly-rebalance vendor layouts) produce a
        # handful of usable dates; with ~11 monthly stamps on 5-8 names the
        # mean-IC noise SE is ~0.11-0.15, so a mean of noise ICs clears the
        # 0.08/0.15 bars by luck a quarter to a third of the time.
        return skipped(
            check,
            f"only {n} usable IC dates - too few to separate a suspicious IC "
            f"level from luck (|mean IC| = {abs(mean_ic):.3f} over {n} dates "
            f"is not a measurement; need >= {MIN_IC_DATES}). Usable dates "
            f"require >= 5 jointly non-NaN names, a different count from the "
            f"common calendar periods performance.sample_size reports - "
            f"extend the overlap of signals and asset_returns, or stamp the "
            f"signal on more dates",
            details=details)
    n_med, n_eff = _ic_noise_context(ic, artifacts, h)
    mult = min(float(np.sqrt(h)), _IC_HORIZON_GROWTH_CAP)
    se = 1.0 / float(np.sqrt(max(n_med - 1.0, 1.0) * n_eff))
    warn_bar = max(config.predictive_ic_warn * mult, _IC_NOISE_Z_WARN * se)
    fail_bar = max(config.predictive_ic_fail * mult, _IC_NOISE_Z_FAIL * se)
    floor_lifted = warn_bar > config.predictive_ic_warn * mult * (1 + 1e-9)
    # Event-gated stamping: usable dates a small fraction of the grid (the
    # excluded dates are exactly the constant/thin cross-sections a gated
    # export leaves off-schedule) - see _GATED_USABLE_FRAC_MAX.
    n_grid = int(len(ic))
    usable_frac = (n / n_grid) if n_grid else 1.0
    gated = usable_frac <= _GATED_USABLE_FRAC_MAX
    diluted_ic = mean_ic * usable_frac
    details.update(median_names=n_med, n_nonoverlap_dates=n_eff,
                   noise_se=se, horizon_multiplier=mult,
                   warn_bar=warn_bar, fail_bar=fail_bar,
                   n_grid_dates=n_grid, usable_date_frac=round(usable_frac, 3),
                   gated_stamping=gated)
    floor_note = (f" (breadth/depth noise floor: {n_med:.0f} names x {n_eff} "
                  f"non-overlapping dates, SE ~{se:.2f}, lifts the bars to "
                  f"{warn_bar:.2f}/{fail_bar:.2f})" if floor_lifted else "")
    embed_bar = max(fail_bar, _GATED_EMBED_IC_FAIL * mult)
    if gated and warn_bar <= abs(mean_ic) < embed_bar:
        # Scheduled alpha and leakage can overlap in conditional IC. Report
        # a scoped warning here; stronger embedding evidence still reaches
        # the FAIL route below without dilution over unstamped dates.
        details.update(diluted_mean_ic=round(diluted_ic, 4),
                       gated_embed_fail_bar=round(embed_bar, 3))
        return warned(
            check,
            f"|mean IC| vs next-{h}-period return = {abs(mean_ic):.3f} "
            f"(NW t={fmt_tstat(t)}), conditioned on the {n} of {n_grid} grid dates "
            f"({usable_frac:.0%}) where the signal stamps a usable "
            f"cross-section - an event-gated/scheduled export. An honest "
            f"gated alpha concentrates its whole edge on exactly these dates "
            f"(all-dates-diluted equivalent {diluted_ic:+.3f}), so bars "
            f"calibrated for continuously-stamped books do not apply; but a "
            f"target leak stamped only on the same dates would produce the "
            f"same shape, and the IC level alone cannot separate the "
            f"two{floor_note}",
            severity=(Severity.HIGH if abs(mean_ic) >= fail_bar
                      else Severity.MEDIUM),
            remediation="verify every feature on the stamped dates is known "
                        "strictly before that date's label window - the "
                        "gated support is the only place a leak could hide; "
                        "if the gate is a pure calendar rule and the "
                        "features are point-in-time, a conditional IC above "
                        "the continuous-book band is expected, not "
                        "suspicious",
            details=details)
    if abs(mean_ic) >= fail_bar:
        band_clause = (
            f"real daily equity alphas run {_REALISTIC_IC_BAND}" if h == 1 else
            f"honest {h}-period-target alphas run ~{_REALISTIC_IC_LO * mult:.2f}-"
            f"{_REALISTIC_IC_HI * mult:.2f} (daily band {_REALISTIC_IC_BAND} x {mult:.2f} "
            f"horizon growth)")
        return failed(
            check,
            f"|mean IC| vs next-{h}-period return = {abs(mean_ic):.3f} over {n} dates "
            f"(NW t={fmt_tstat(t)}); {band_clause} - the "
            f"signal almost certainly embeds the target{floor_note}",
            severity=Severity.HIGH,
            remediation="inspect signal construction for future data: unlagged joins, "
                        "feature windows overlapping the label horizon, or "
                        "target-derived features; rebuild the signal point-in-time",
            details=details)
    if abs(mean_ic) >= warn_bar:
        band_clause = (
            f"realistic band {_REALISTIC_IC_BAND}" if h == 1 else
            f"realistic band ~{_REALISTIC_IC_LO * mult:.2f}-"
            f"{_REALISTIC_IC_HI * mult:.2f} for "
            f"{h}-period targets, daily band {_REALISTIC_IC_BAND} x {mult:.2f}")
        return warned(
            check,
            f"|mean IC| vs next-{h}-period return = {abs(mean_ic):.3f} over {n} dates "
            f"(NW t={fmt_tstat(t)}) - rare for real alphas ({band_clause}); verify "
            f"feature timestamps against the label window{floor_note}",
            severity=Severity.HIGH,
            remediation="verify every feature is known strictly before the forward-"
                        "return window starts; re-derive the signal from point-in-time "
                        "data and re-measure the IC",
            details=details)
    # The "within the band" clause is a claim about this value: only make it
    # when |mean IC| actually sits inside 0.01-0.05 (an IC of e.g. 0.062 is
    # below the 0.08 warn bar yet outside the band); everything else gets
    # the below-the-bar wording, which is always true.
    if (h == 1 and not floor_lifted and not gated
            and _REALISTIC_IC_LO <= abs(mean_ic) <= _REALISTIC_IC_HI):
        return passed(
            check,
            f"mean IC {mean_ic:+.4f} (NW t={fmt_tstat(t)}) over {n} dates vs {h}-period "
            f"forward returns - within the realistic {_REALISTIC_IC_BAND} band for "
            f"daily equity alphas",
            details=details)
    return passed(
        check,
        f"mean IC {mean_ic:+.4f} (NW t={fmt_tstat(t)}) over {n} dates vs {h}-period forward "
        f"returns - below the {warn_bar:.2f} suspicion bar"
        f"{f' for h={h} targets (daily bar x {mult:.2f} honest horizon growth)' if h > 1 else ''}"
        f"{floor_note}",
        details=details)


# ---------------------------------------------------------------------------
# 4. performance.ic_stability
# ---------------------------------------------------------------------------

def _ic_moot_floor(ic, artifacts) -> float:
    """Effective |mean IC| moot-gate floor for the stability/regime checks:
    the absolute 0.005 floor lifted to _IC_MOOT_GATE_Z x the mean-IC noise
    SE (breadth/depth-scaled, reusing _ic_noise_context exactly as
    suspicious_ic does) - an absolute floor alone is cleared by pure noise
    ~half the time at ordinary breadth (SE 0.0068 at 30 names x 750 days)."""
    n_med, n_eff = _ic_noise_context(ic, artifacts, int(artifacts.label_horizon))
    se = 1.0 / float(np.sqrt(max(n_med - 1.0, 1.0) * n_eff))
    return max(_IC_STABILITY_MIN_ABS_IC, _IC_MOOT_GATE_Z * se)


def _ic_stability(ic, artifacts, config) -> CheckResult:
    check = "performance.ic_stability"
    window = int(config.ic_rolling_window)
    ic_valid = ic.dropna()
    mean_ic = float(ic_valid.mean()) if len(ic_valid) else float("nan")
    # A window longer than the available IC sample has zero complete
    # windows.  Avoid handing an unbounded Python integer to pandas, whose
    # C-sized rolling-window conversion otherwise raises OverflowError.
    roll = (pd.Series(dtype=float)
            if window > len(ic_valid)
            else ic_valid.rolling(window, min_periods=window).mean().dropna())
    n_windows = int(len(roll))
    moot_floor = (_ic_moot_floor(ic, artifacts) if len(ic_valid)
                  else _IC_STABILITY_MIN_ABS_IC)
    if (not np.isfinite(mean_ic) or abs(mean_ic) <= moot_floor
            or n_windows < _IC_STABILITY_MIN_WINDOWS):
        return passed(
            check,
            f"IC too weak/sample too short to judge stability (|mean IC| "
            f"{abs(mean_ic):.4f} <= the {moot_floor:.4f} noise-scaled floor "
            f"(max of {_IC_STABILITY_MIN_ABS_IC:g} absolute and "
            f"{_IC_MOOT_GATE_Z:g} x the breadth/depth mean-IC SE) or only "
            f"{n_windows} rolling {window}-period windows, need >= "
            f"{_IC_STABILITY_MIN_WINDOWS}) - not judged",
            mean_ic=mean_ic, n_windows=n_windows, window=window,
            moot_floor=moot_floor, not_judged=True)
    full_sign = 1.0 if mean_ic > 0 else -1.0
    consistency = float((np.sign(roll.to_numpy(dtype=float)) == full_sign).mean())
    worst = float(roll.min())
    best = float(roll.max())
    details = dict(consistency=consistency, n_windows=n_windows,
                   worst_window_ic=worst, best_window_ic=best,
                   mean_ic=mean_ic, window=window)
    if consistency < config.ic_sign_consistency_warn:
        return warned(
            check,
            f"full-sample IC {mean_ic:+.3f} but only {consistency:.0%} of {n_windows} "
            f"rolling {window}-day windows agree on sign (window IC range "
            f"{worst:+.3f}..{best:+.3f}) - the edge comes and goes",
            severity=Severity.MEDIUM,
            remediation="test sub-period robustness before trusting the full-sample "
                        "stat: identify when the edge is on/off, and either restrict "
                        "the claim to those regimes or find a conditioning variable",
            details=details)
    return passed(
        check,
        f"{consistency:.0%} of {n_windows} rolling {window}-day IC windows match the "
        f"full-sample sign (mean IC {mean_ic:+.3f}, window range "
        f"{worst:+.3f}..{best:+.3f})",
        details=details)


# ---------------------------------------------------------------------------
# 5. performance.ic_regime_concentration
# ---------------------------------------------------------------------------

def _ic_regime_concentration(ic, artifacts, config) -> CheckResult:
    check = "performance.ic_regime_concentration"
    ic_valid = ic.dropna()
    if len(ic_valid):
        yearly = ic_valid.groupby(ic_valid.index.year).sum()
    else:
        yearly = pd.Series(dtype=float)
    yearly_dict = {int(y): float(v) for y, v in yearly.items()}
    n_years = int(len(yearly))
    total = float(yearly.sum()) if n_years else 0.0
    mean_ic = float(ic_valid.mean()) if len(ic_valid) else float("nan")
    # The concentration question is polarity-blind: a negative-IC edge (a
    # signal traded inverted, or a sign-flipped construction) concentrated in
    # one year is exactly as much a regime bet as a positive one, so judge
    # whichever sign the full-sample edge has. Materiality gate on |mean IC|
    # rather than on total > 0: with no real edge, the top same-sign year
    # trivially carries a large share of a near-zero total - pure noise, not a
    # regime finding. The gate is noise-SE-scaled (see _ic_moot_floor).
    moot_floor = (_ic_moot_floor(ic, artifacts) if len(ic_valid)
                  else _REGIME_MIN_ABS_MEAN_IC)
    if (n_years < _MIN_YEARS_FOR_REGIME or not np.isfinite(mean_ic)
            or abs(mean_ic) <= moot_floor):
        return passed(
            check,
            f"regime concentration not judged: {n_years} calendar year(s) with "
            f"|mean IC| {abs(mean_ic):.4f} (cumulative {total:+.2f}); need >= "
            f"{_MIN_YEARS_FOR_REGIME} years and |mean IC| > the "
            f"{moot_floor:.4f} noise-scaled floor (max of "
            f"{_REGIME_MIN_ABS_MEAN_IC:g} absolute and {_IC_MOOT_GATE_Z:g} x "
            f"the breadth/depth mean-IC SE) - no edge detectable above noise "
            f"whose concentration could be judged",
            yearly=yearly_dict, n_years=n_years, total_ic=total,
            mean_ic=mean_ic, moot_floor=moot_floor, not_judged=True)
    sign = 1.0 if total > 0 else -1.0
    direction = "positive" if sign > 0 else "negative"
    aligned = yearly * sign                 # per-year contribution in the edge's direction
    contrib = aligned[aligned > 0]
    top_year = int(contrib.idxmax())
    top = float(yearly.loc[top_year])       # signed, for the message
    same_sign_sum = float(sign * contrib.sum())
    share = float(contrib.max() / contrib.sum())
    details = dict(yearly=yearly_dict, top_year=top_year, share=share,
                   n_years=n_years, total_ic=total, mean_ic=mean_ic,
                   direction=direction)
    if share > config.ic_regime_share_warn:
        polarity_note = ("" if sign > 0 else
                         f" (negative-polarity edge, mean IC {mean_ic:+.3f} - "
                         f"the same regime question applies to the inverted book)")
        return warned(
            check,
            f"{share:.0%} of the cumulative {direction} IC comes from {top_year} "
            f"alone (IC sum {top:+.1f} of the {same_sign_sum:+.1f} same-sign total "
            f"across {n_years} years) - regime bet, not persistent "
            f"alpha{polarity_note}",
            severity=Severity.MEDIUM,
            remediation=f"re-run the backtest excluding {top_year}; if the edge "
                        f"disappears, treat this as a one-regime bet - size it "
                        f"accordingly or condition on a regime indicator",
            details=details)
    return passed(
        check,
        f"cumulative {direction} IC is spread across {n_years} calendar years; the "
        f"largest ({top_year}) carries {share:.0%} of the same-sign total (mean IC "
        f"{mean_ic:+.4f})",
        details=details)


# ---------------------------------------------------------------------------
# 6. performance.sample_size
# ---------------------------------------------------------------------------

def _sample_size(strat, source, artifacts, config) -> CheckResult:
    """Two counts, both against ``config.min_periods``: the common
    signals x asset_returns calendar (what the IC statistics are powered
    on) and the finite observations of the Sharpe-check return source (what
    suspicious_sharpe / deflated_sharpe and the cost reconciliation are
    powered on). A dense 750-date calendar with a net series finite on only
    40 bars must not PASS on the calendar alone."""
    check = "performance.sample_size"
    n = int(len(artifacts.common_index))
    min_p = int(config.min_periods)
    n_sr: int | None = None
    if strat is not None:
        n_sr = int(np.isfinite(
            pd.Series(strat).to_numpy(dtype="float64", na_value=np.nan)).sum())
    sr_label = ("finite strategy_returns observations"
                if source == "net strategy_returns" else
                "finite gross-reconstruction (positions x asset_returns) "
                "return observations")
    details = dict(n_periods=n, min_periods=min_p, n_strategy_obs=n_sr,
                   returns_source=source)
    gap = (_net_return_span_gap(strat, source)
           if strat is not None else None)
    if gap is not None:
        details.update(gap)
        return skipped(
            check,
            f"{n} common periods and {n_sr} finite strategy_returns, but "
            f"{gap['n_interior_missing']} date(s) are missing/NaN inside the "
            f"reported first-to-last finite span - selected-date returns are "
            f"not a dense sample, so sample adequacy is not judged. Report "
            f"one finite net return per audited calendar date throughout the "
            f"live span; leading/trailing warmup truncation may remain.",
            details=details,
        )
    remediation = ("extend the backtest history (or lower the bar consciously "
                   "via AuditConfig(min_periods=...)); statistical checks need "
                   "more periods before their answers mean anything")
    if n < min_p:
        sr_clause = (f" and only {n_sr} {sr_label}"
                     if n_sr is not None and n_sr < min_p else "")
        return warned(
            check,
            f"only {n} common periods{sr_clause} (< {min_p}) - every statistic "
            f"in this module is underpowered at this sample size; PASSes "
            f"elsewhere prove little",
            severity=Severity.LOW, remediation=remediation, details=details)
    if n_sr is not None and n_sr < min_p:
        return warned(
            check,
            f"{n} common periods but only {n_sr} {sr_label} (< {min_p}) - "
            f"the Sharpe-based checks (suspicious_sharpe, deflated_sharpe) and "
            f"the cost reconciliation are powered on those {n_sr} bars, not "
            f"on the {n}-date calendar the IC statistics use; PASSes on the "
            f"return series prove little",
            severity=Severity.LOW,
            remediation="pass a return series that covers the audited "
                        "calendar (or trim signals/asset_returns to the span "
                        "the book actually traded); " + remediation,
            details=details)
    sr_clause = (f" and {n_sr} {sr_label}" if n_sr is not None else "")
    return passed(
        check,
        f"{n} common periods{sr_clause} (>= {min_p}) - adequate sample for the "
        f"statistical checks in this module",
        details=details)
