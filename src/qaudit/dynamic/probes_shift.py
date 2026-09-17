"""Date-shift and truncation probes.

These re-run the user's actual pipeline callables under time perturbation:

- ``dynamic.date_shift``               re-runs ``backtest_func`` with the
  signal shifted -K..+K bars. An honest pipeline improves when (illegally)
  peeking earlier and decays gracefully with extra delay; a leaky one has
  its edge welded to exactly the executed bar. Exception: a causal
  fast-reversal signal collapses deeply negative at k=-1 (the raw shift
  aligns it against its own bar) - that shape is disambiguated from the
  leak signature and cross-checked against the causality probes below.
  A mild causal same-bar loading (e.g. a mean-reversion term whose window
  ends at bar t) collapses only part-way: that ambiguous middle band is
  adjudicated by the signal-probe evidence rather than condemned outright.
  A bar-scheduled book (calendar/event-locked, trading a strictly-spaced
  subset of bars) legitimately collapses under any +-1-bar shift - the
  signal value encodes when to trade, so the shift misaligns the scheduled
  bet instead of adding information; detected from the activity pattern and
  reported as hedged calendar misalignment (WARN), never the categorical
  embedded-return conviction. A held scheduled-refresh book (positions
  never flat, but re-sampled from the signal only on a strict
  weekly/monthly cadence) aliases the same way - the +-1-bar shift rotates
  which staleness of the signal each refresh samples, a genuinely
  different bet for any fast component - and is detected from position
  changes (dominant refresh gap) rather than levels; same hedged WARN.
- ``dynamic.signal_reproducibility``   confirms ``signal_func(signal_input)``
  reproduces the audited signals, so the truncation verdict transfers.
- ``dynamic.rolling_window_integrity`` the truncation probe: recomputes the
  signal on history cut at date t and compares the row at t. Any change
  means the computation reads the future (full-sample z-scores, global
  ranks, centered windows, scalers fitted on all data) - with respect to
  the input argument: constants baked into the callable at definition time
  (e.g. a scaler fitted on the full sample outside signal_func, then
  captured in a closure) are behaviorally indistinguishable from a-priori
  hyperparameters, so no black-box probe can reach them; every "causal"
  verdict below is scoped to heuristic evidence at the sampled dates, and
  a static fingerprint advisory flags the common accidental form.
- ``dynamic.signal_input_sensitivity`` guards the two probes above against
  a signal_func that ignores its input and replays a stored panel (a
  closure cheat launders arbitrarily leaky signals through both): called on
  input truncated at t it must not return non-NaN values on rows after t
  (all-NaN master-calendar padding is tolerated), and its output must
  respond to input perturbation - a cross-sectional value scramble, backed
  by a sign-flip + additive-noise magnitude perturbation for honest
  permutation-invariant statistics (breadth, median, dispersion). A flagged
  callable also loses the causal_verified stamp the date-shift probe
  consults.

Timing convention: see :mod:`qaudit.inputs`. ``signals.shift(k)`` with
positive k delays the signal (always legal); negative k lets the pipeline
see the signal ``|k|`` bars early (peeking).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .._stats import annualized_sharpe
from ..config import AuditConfig
from ..inputs import BacktestArtifacts
from ..types import (CheckResult, Severity, Status, errored, failed, passed,
                     skipped, warned)
from ._pipeline import PipelineOutputError, validated_returns

# Module constants - structural to the probe design;
# every tunable threshold lives in AuditConfig.
PEEK_MIN_BASELINE_SR = 1.0    # rule (a) needs a real baseline edge to judge
PEEK_COLLAPSE_RATIO = 0.5     # sr(-1) < this fraction of sr(0) => edge welded to the bar
PEEK_AMBIG_NOISE_Z = 1.0      # sr(-1) within this many standard errors of 0
                              # (SE(SR_ann) ~ sqrt(ppy/n) under the null) reads
                              # as "collapsed to noise" - the leak-evaporation
                              # signature. Meaningfully below the band, the
                              # shifted book is systematically losing, which a
                              # vanished edge cannot explain...
PEEK_AMBIG_MIN_RATIO = 0.25   # ...but only when that loss rate is also at
                              # least this fraction of the baseline SR does the
                              # collapse carry structure commensurate with the
                              # edge - the ambiguous shape a mild causal
                              # same-bar loading produces (on honest
                              # momentum + mean-reversion blends the
                              # remnant ratio |sr(-1)|/sr(0) falls in roughly
                              # 0.4-1.3). An evaporated leak's k=-1 remnant
                              # (cost drag on a high-turnover book, noise) is
                              # small relative to its inflated sr(0): roughly
                              # 0.01-0.12 of it on welded/target-leak books,
                              # so the 0.25 bar sits in the gap between the
                              # two bands. Between the two gates rule (a) must
                              # adjudicate by signal-probe evidence, not
                              # condemn outright.
DELAY_MIN_BASELINE_SR = 0.5   # rule (b) needs some baseline edge to judge decay
REPRO_MISMATCH_FRAC = 0.01    # tolerated fraction of non-reproduced signal cells
REPRO_MIN_COVERAGE = 0.95     # signal_func's output must cover this fraction of
                              # the audited non-NaN signal cells, else the repro
                              # and truncation verdicts are vacuous: a callable
                              # returning one correct cell would otherwise
                              # launder a leaky full-sample signal through both
REPRO_CONC_MISMATCH_FRAC = 0.05  # per-asset / per-date mismatch concentration
                              # gate: the 1% global tolerance above models
                              # i.i.d. scattered float/re-run noise, so one
                              # wholly replaced asset among 200 (0.5% of
                              # cells) or 3 replaced dates of 730 would ride
                              # it to a panel-wide causal stamp. A single
                              # column/row concentrating >= 5% mismatch of
                              # its own comparable cells is incompatible with
                              # the scattered-noise model (binomial tail ~0
                              # at a 1% base rate) and withholds the stamp.
REPRO_CONC_MIN_CELLS = 20     # a column/row needs at least this many
                              # comparable cells for its own fraction to be
                              # judgeable...
REPRO_CONC_MIN_MISMATCH = 3   # ...and at least this many mismatched cells:
                              # 1-2 strays in a short column are the
                              # scattered noise the global tolerance absorbs
                              # (P[Binom(20, 0.01) >= 3] ~ 1e-3), not a
                              # replaced asset.
FULLSAMPLE_SIG_ATOL = 1e-8    # full-sample-scaler fingerprint exactness: a
                              # per-asset full-sample mean within this of 0
                              # and std within this of 1 (or min/max of 0/1)
                              # across every audited asset only happens when
                              # the whole panel went through a scaler fitted
                              # on the whole sample; an honest expanding or
                              # cross-sectional normalization misses by
                              # orders of magnitude. Advisory only - frozen
                              # fitted constants are black-box
                              # indistinguishable from a-priori
                              # hyperparameters, so this can never be a gate.
FULLSAMPLE_SIG_MIN_OBS = 20   # columns with fewer observations carry no
                              # fingerprint evidence and are excluded.
TRUNCATION_HEAD_SKIP = 0.05   # skip this leading fraction of valid signal dates
                              # (warmup buffer); the rest of the range is sampled
                              # uniformly so early-history leaks are covered too
SENSITIVITY_DATE_FRAC = 0.75  # the sensitivity probe truncates the input this
                              # deep into signal_func's valid output range:
                              # far enough that warmup windows are full, early
                              # enough that rows can exist beyond it
SENSITIVITY_MIN_ASSETS = 3    # the value scramble permutes each input row
                              # across assets; fewer columns leave nothing to
                              # permute and the probe SKIPs
SENSITIVITY_MAG_NOISE_FRAC = 1.0  # stage-2 magnitude perturbation: additive
                              # noise std as a fraction of the panel's own
                              # std, on top of a global sign flip. Sign flip
                              # + noise (never pure scaling, which preserves
                              # breadth) moves every permutation-invariant
                              # row statistic - breadth, median, quantiles.
SCHED_MAX_ACTIVE_FRAC = 0.5   # scheduled-bar carve-out: a book is
                              # bar-scheduled only when it trades at most this
                              # fraction of bars...
SCHED_MIN_GAP = 2             # ...on a strictly-spaced subset (every gap
                              # between active bars >= 2), so a +-1-bar shift
                              # provably moves every bet onto bars the
                              # schedule never trades. For such a signal the
                              # value encodes when to trade: shift(-1) is not
                              # "more information", it misaligns the scheduled
                              # bet, and without the carve-out honest calendar
                              # alphas (day-of-week seasonals) collapse into
                              # the peek rule's FAIL CRITICAL. Clustered
                              # schedules (turn-of-month runs of adjacent
                              # bars) fail the min-gap test and keep the
                              # un-carved rules - they mostly survive the
                              # shift anyway.
SCHED_MIN_ACTIVE_BARS = 10    # fewer active bars cannot establish a schedule
SCHED_POS_TOL = 1e-12         # |position| above this marks a bar active
SCHED_REFRESH_DOMINANT_SHARE = 0.9  # held-refresh detection: a held book
                              # (positions nonzero nearly every bar) that
                              # re-samples the signal on a strict schedule
                              # never reaches the level-based detector
                              # above, yet the +-1-bar shift aliases against
                              # its refresh grid exactly like a sparse
                              # scheduled book, so an honest weekly-refreshed
                              # multi-sleeve book would collapse into the
                              # peek rule's FAIL CRITICAL. Detect from
                              # position changes instead: the dominant
                              # refresh gap must be >= SCHED_MIN_GAP and
                              # cover at least this share of all gaps - the
                              # slack (vs the strict min-gap test) absorbs
                              # sparse off-schedule delisting force-closes,
                              # which split a clean 5-bar cadence into gaps
                              # of 1-4 on a few percent of refreshes without
                              # breaking the cadence. A daily-traded book has
                              # dominant gap 1 and never qualifies; a
                              # daily-resized overlay on a scheduled reselect
                              # changes positions every bar and also never
                              # qualifies (a known residual false-positive
                              # class: it keeps the un-carved rules).

DATE_SHIFT_FLAT_TOL = 1e-12   # SR spread across all shifts at or below this
                              # means the shifted panel never flowed through
                              # backtest_func (honest engines separate curve
                              # points by orders of magnitude more; float
                              # noise alone exceeds this on any real book)

CHECK_DATE_SHIFT = "dynamic.date_shift"
CHECK_REPRO = "dynamic.signal_reproducibility"
CHECK_TRUNCATION = "dynamic.rolling_window_integrity"
CHECK_SENSITIVITY = "dynamic.signal_input_sensitivity"


def run(artifacts: BacktestArtifacts, config: AuditConfig, *,
        signal_func: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        backtest_func: Callable[[pd.DataFrame, pd.DataFrame], pd.Series] | None = None,
        ) -> list[CheckResult]:
    # signal_func probes run first: their verdicts feed the date-shift probe
    # (a k=-1 collapse of a verified-causal signal cannot be an embedded
    # future return - see rule (a) below). The causal stamp requires all
    # three probe verdicts: reproducibility and truncation prove nothing
    # when the callable does not respond to its input (closure cheats), so
    # the sensitivity probe must positively PASS.
    repro_res, trunc_res, sens_res = _signal_func_probes(artifacts, config,
                                                         signal_func)
    causal_verified = (repro_res.status is Status.PASS
                       and trunc_res.status is Status.PASS
                       and sens_res.status is Status.PASS)
    return [_date_shift(artifacts, config, backtest_func,
                        causal_verified=causal_verified),
            repro_res, trunc_res, sens_res]


# ---------------------------------------------------------------------------
# 1. dynamic.date_shift
# ---------------------------------------------------------------------------

def _fmt_sr(v: float) -> str:
    return f"{v:.2f}" if np.isfinite(v) else "nan"


def _strict_gap_schedule(active: pd.Series,
                         source: str) -> dict[str, Any] | None:
    """Sparse-activity detector: activity confined to a sparse,
    strictly-spaced subset of bars (every gap >= SCHED_MIN_GAP), so a
    +-1-bar shift provably lands every bet on bars the schedule never
    trades. Returns the detection details, or None."""
    flags = np.flatnonzero(active.to_numpy())
    n_bars = int(len(active))
    n_active = int(len(flags))
    if n_bars == 0 or n_active < SCHED_MIN_ACTIVE_BARS:
        return None
    frac = n_active / n_bars
    if frac > SCHED_MAX_ACTIVE_FRAC:
        return None
    gaps = np.diff(flags)
    if gaps.size == 0 or int(gaps.min()) < SCHED_MIN_GAP:
        return None
    return dict(schedule_kind="sparse_activity", active_frac=round(frac, 3),
                n_active_bars=n_active, n_bars=n_bars, min_gap=int(gaps.min()),
                activity_source=source)


def _refresh_grid_schedule(positions: pd.DataFrame) -> dict[str, Any] | None:
    """Held-refresh detector: a held book (never flat, so the
    level-based detector above cannot see it) whose positions change only
    on a strictly-cadenced refresh grid. The +-1-bar shift then rotates
    which staleness of the signal each scheduled refresh samples - a
    genuinely different bet for any fast component - instead of adding
    information. The cadence test uses the dominant gap with a share floor
    (not the strict min-gap): sparse off-schedule delisting force-closes
    split a clean weekly cadence into occasional 1-4-bar gaps, and on a
    renormalizing engine those bars also move surviving names' weights, so
    no exits-to-zero exclusion can restore the strict test. Returns the
    detection details, or None."""
    pos = positions.fillna(0.0)
    chg = pos.diff().abs().gt(SCHED_POS_TOL)
    if len(chg):
        chg.iloc[0] = pos.iloc[0].abs().gt(SCHED_POS_TOL)   # entry trade
    refresh = chg.any(axis=1)
    flags = np.flatnonzero(refresh.to_numpy())
    n_bars = int(len(refresh))
    n_refresh = int(len(flags))
    if n_bars == 0 or n_refresh < SCHED_MIN_ACTIVE_BARS:
        return None
    frac = n_refresh / n_bars
    if frac > SCHED_MAX_ACTIVE_FRAC:
        return None
    gaps = np.diff(flags)
    if gaps.size == 0:
        return None
    vals, counts = np.unique(gaps, return_counts=True)
    dom = int(vals[int(np.argmax(counts))])
    share = float(counts.max() / gaps.size)
    if dom < SCHED_MIN_GAP or share < SCHED_REFRESH_DOMINANT_SHARE:
        return None
    return dict(schedule_kind="held_refresh", active_frac=round(frac, 3),
                n_active_bars=n_refresh, n_bars=n_bars, dominant_gap=dom,
                dominant_gap_share=round(share, 3),
                activity_source="position changes")


def _scheduled_activity(artifacts: BacktestArtifacts,
                        base_rets: pd.Series | None) -> dict[str, Any] | None:
    """Detect a bar-scheduled book, in either shape:

    - ``sparse_activity``: calendar/event-locked, trading a
      sparse strictly-spaced subset of bars - nonzero-position bars when
      positions are present, else the zero/nonzero pattern of the baseline
      (k=0) returns Series. The returns fallback is deliberately
      conservative: on a costed book the exit-cost bar is nonzero one bar
      after each bet, breaking the min-gap test, so a costed scheduled
      book without positions keeps the un-carved rules (fails safe).
    - ``held_refresh`` (positions only): a held book whose
      positions change on a strict dominant cadence - the shape the
      level-based test can never reach because a held book is ~100%
      level-active.

    Returns the detection details (``schedule_kind`` says which), or None.
    """
    if artifacts.positions is not None:
        active = (artifacts.positions.abs() > SCHED_POS_TOL).any(axis=1)
        det = _strict_gap_schedule(active, "positions")
        if det is not None:
            return det
        return _refresh_grid_schedule(artifacts.positions)
    if base_rets is not None:
        vals = pd.Series(base_rets).astype(float)
        active = vals.notna() & (vals.abs() > 0.0)
        return _strict_gap_schedule(active, "baseline returns")
    return None


def _date_shift(artifacts: BacktestArtifacts, config: AuditConfig,
                backtest_func, *, causal_verified: bool = False) -> CheckResult:
    if backtest_func is None:
        return skipped(
            CHECK_DATE_SHIFT,
            "pass backtest_func(signals, asset_returns) -> net strategy-returns "
            "Series (applying your production execution lag and costs) to enable "
            "the date-shift probe.")
    k_max = int(config.date_shift_max)
    curve: dict[int, float] = {}
    n_obs = 0                     # baseline sample size, for the SR noise floor
    base_rets: pd.Series | None = None   # k=0 series, for schedule detection
    for k in range(-k_max, k_max + 1):
        try:
            # .copy(): an in-place-mutating callable must not corrupt the
            # shared returns frame across the 2k+1 re-runs (shift(k) already
            # hands a fresh signals frame per call).
            rets = backtest_func(artifacts.signals.shift(k),
                                 artifacts.asset_returns.copy())
            if isinstance(rets, pd.DataFrame):
                # Contract slip, not a crash: the callable ran fine, but a
                # 2-D return breaks the probe's own Sharpe math identically
                # at every k - name the return-type contract instead of
                # blaming the shift.
                n_r, n_c = rets.shape
                return skipped(
                    CHECK_DATE_SHIFT,
                    f"backtest_func returned a DataFrame (shape "
                    f"{n_r}x{n_c}), not the net strategy-returns pd.Series "
                    f"the probe contract requires - the shifted re-runs "
                    f"cannot be Sharpe-judged on a 2-D panel. Return a 1-D "
                    f"Series (for a 1-column frame: "
                    f"net.squeeze('columns')) and re-run.")
            rets = validated_returns(rets, artifacts.asset_returns.index)
            curve[k] = float(annualized_sharpe(rets, artifacts.periods_per_year))
            if k == 0:
                n_obs = int(np.asarray(pd.Series(rets).dropna(), dtype=float).size)
                base_rets = pd.Series(rets)
        except PipelineOutputError as exc:
            res = errored(CHECK_DATE_SHIFT, exc)
            res.message = (
                f"backtest_func returned invalid net returns at shift k={k:+d}: "
                f"{exc} - the date-shift comparison was not trusted")
            return res
        except Exception as exc:  # noqa: BLE001 - a crashing callable must not kill the audit
            res = errored(CHECK_DATE_SHIFT, exc)
            res.message = (
                f"backtest_func raised while re-running with signals shifted "
                f"k={k:+d}: {type(exc).__name__}: {exc} - the probe proves "
                f"nothing until the callable runs on perturbed inputs; make it "
                f"robust to shifted/NaN-padded signal panels")
            return res

    sr_0 = curve.get(0, float("nan"))
    sr_m1 = curve.get(-1, float("nan"))
    sr_p1 = curve.get(1, float("nan"))
    curve_json = {str(k): (float(v) if np.isfinite(v) else None)
                  for k, v in curve.items()}
    curve_str = ", ".join(f"{k:+d}: {_fmt_sr(curve[k])}" for k in sorted(curve))
    details: dict[str, Any] = dict(curve=curve_json,
                                   sr_baseline=curve_json.get("0"),
                                   sr_peek_1=curve_json.get("-1"),
                                   sr_delay_1=curve_json.get("1"),
                                   date_shift_max=k_max)
    # Scheduled-bar carve-out: for a book trading only a strictly-
    # spaced calendar subset of bars, both rules' premises fail - a +-1-bar
    # shift moves every bet onto bars the schedule never trades, so the
    # collapse is calendar misalignment, not evidence the signal embeds the
    # executed bar's return. The carve-out hedges (WARN, never PASS): a
    # same-bar leak deliberately zeroed off-schedule collapses identically.
    sched = _scheduled_activity(artifacts, base_rets)
    if sched is not None:
        details.update(scheduled_activity=sched)
    held_refresh = (sched is not None
                    and sched.get("schedule_kind") == "held_refresh")
    if sched is None:
        sched_str = ""
    elif held_refresh:
        sched_str = (f"{sched['n_active_bars']} of {sched['n_bars']} bars "
                     f"refresh the held positions, dominant refresh gap "
                     f"{sched['dominant_gap']} bars covering "
                     f"{sched['dominant_gap_share']:.0%} of gaps, from "
                     f"{sched['activity_source']}")
    else:
        sched_str = (f"{sched['n_active_bars']} of {sched['n_bars']} bars "
                     f"active, min gap {sched['min_gap']} bars, from "
                     f"{sched['activity_source']}")

    # (a) peeking one bar earlier destroys the strategy. "More information
    # cannot hurt an honest pipeline" is only true of information - and a raw
    # signals.shift(-1) does more than add information: it re-aligns the
    # signal with its own bar. A causal fast-reversal signal
    # (signal_t ~ -zscore_cs(r_t)) becomes mechanically anti-correlated with
    # the very return it earns and collapses to a hugely negative SR without
    # any leak. Disambiguate by the collapse shape: a leak welded to the bar
    # evaporates (SR -> a remnant near 0 and small relative to sr_0 - cost
    # drag on a high-turnover book can pull the remnant a few SR units
    # negative, but never in proportion to the vanished edge); the reversal
    # artifact anti-correlates (the shifted book loses at least as fast as
    # the baseline earns). Between them lies an ambiguous band - sr(-1)
    # meaningfully negative and commensurate with sr_0, yet short of the full
    # anti-correlation bar - where a mild causal same-bar loading (e.g. a
    # mean-reversion term whose window ends at bar t) and a small
    # reversed-sign embed of the executed bar's return produce identical
    # curves: that band is adjudicated by the signal-probe evidence.
    peek_exempt = False
    if (np.isfinite(sr_0) and np.isfinite(sr_m1)
            and sr_0 >= PEEK_MIN_BASELINE_SR
            and sr_m1 < PEEK_COLLAPSE_RATIO * sr_0):
        anti_bar = -max(PEEK_MIN_BASELINE_SR, PEEK_COLLAPSE_RATIO * sr_0)
        noise_floor = PEEK_AMBIG_NOISE_Z * float(
            np.sqrt(artifacts.periods_per_year / max(n_obs, 2)))
        anti_corr = sr_m1 <= anti_bar
        ambiguous_neg = (not anti_corr
                         and sr_m1 <= -max(noise_floor,
                                           PEEK_AMBIG_MIN_RATIO * sr_0))
        # Positive commensurate remnant: an honest one-bar-horizon alpha
        # with signal autocorrelation phi retains ~phi*sr_0 at k=-1 - the
        # fresher signal vintage predicts the next bar, which a fixed-lag
        # pipeline cannot exploit ("more information cannot hurt" holds for
        # optimizers, not fixed mappings). An honest alpha's remnant ratio
        # is of the order of phi, whereas a welded or target leak retains
        # about a percent of its inflated baseline, so the commensurability
        # prong (>= 25% of baseline) is load-bearing: a welded
        # momentum-blend leak's remnant can clear the noise floor alone
        # but never the ratio.
        ambiguous_pos = sr_m1 >= max(noise_floor,
                                     PEEK_AMBIG_MIN_RATIO * sr_0)
        ambiguous = ambiguous_neg or ambiguous_pos
        details.update(peek_signature=("anti_correlated" if anti_corr
                                       else "ambiguous" if ambiguous
                                       else "evaporation"),
                       signal_verified_causal=bool(causal_verified),
                       peek_anti_corr_bar_sr=float(anti_bar),
                       peek_noise_floor_sr=float(noise_floor),
                       n_baseline_obs=n_obs)
        if ambiguous:
            details["peek_ambig_side"] = ("positive_remnant" if ambiguous_pos
                                          else "negative_remnant")
        if anti_corr and causal_verified:
            # Benign reversal artifact: rule (a) is uninformative for this
            # signal shape, but rule (b)'s delay-fragility question still
            # stands on its own merits - fall through to it.
            peek_exempt = True
        elif sched is not None:
            # Scheduled-bar collapse: every k=-1 bet lands on a bar the
            # schedule never trades, so any collapse shape (evaporation,
            # ambiguous, anti-correlated) is the mechanical misalignment an
            # honest calendar alpha produces - never license for the
            # categorical embedded-return conviction. Hedged WARN: a
            # same-bar leak zeroed off-schedule collapses identically, so
            # the causal stamp buys the softer severity, not a PASS.
            stamp_note = (
                " signal_func was verified causal with respect to its "
                "input at the sampled dates (reproducibility, truncation "
                "and input-sensitivity probes all PASS), which "
                "exonerates the signal computation; the schedule's own "
                "provenance stays outside the probes' reach."
                if causal_verified else
                " Without the causal stamp this probe cannot favor either "
                "reading.")
            if held_refresh:
                # Held scheduled-refresh collapse: the book is in
                # the market every bar, but the engine samples the signal
                # only on the refresh grid - shift(-1) rotates which
                # staleness of the signal each refresh trades (a fresh
                # fast-reversal sleeve is a genuinely different, often
                # worse, bet than the 4-bar-stale one production holds), so
                # the collapse is sampling-phase aliasing, not evidence the
                # signal embeds the executed bar's return. Hedged WARN: a
                # leak confined to the refresh bars collapses identically.
                return warned(
                    CHECK_DATE_SHIFT,
                    f"peeking one bar EARLIER collapses the strategy "
                    f"(annualized SR {sr_0:.1f} -> {sr_m1:.1f} at shift "
                    f"k=-1; curve: {curve_str}) - but the book holds its "
                    f"positions and REFRESHES them only on a strict "
                    f"schedule ({sched_str}), so the shift does not add "
                    f"information: it rotates which staleness of the "
                    f"signal each scheduled refresh samples, aliasing any "
                    f"fast component against the refresh grid (a k=-1 "
                    f"refresh trades a genuinely different vintage of the "
                    f"composite). An honest weekly/monthly-refreshed held "
                    f"multi-sleeve book collapses exactly like this - but "
                    f"so would a leak confined to the refresh bars, and "
                    f"this probe cannot tell them apart.{stamp_note}",
                    severity=(Severity.MEDIUM if causal_verified
                              else Severity.HIGH),
                    remediation="Verify the refresh schedule is fixed in "
                                "advance (a pure calendar rebalance rule, "
                                "not derived from the bar's own data); "
                                "pass signal_func and "
                                "artifacts.signal_input so all three "
                                "causality probes can stamp the signal "
                                "causal - the softest reading requires "
                                "that stamp. To judge the shift rules "
                                "un-carved, re-run the audit on a "
                                "daily-rebalanced variant of the same "
                                "composite.",
                    details=details)
            return warned(
                CHECK_DATE_SHIFT,
                f"peeking one bar EARLIER collapses the strategy "
                f"(annualized SR {sr_0:.1f} -> {sr_m1:.1f} at shift k=-1; "
                f"curve: {curve_str}) - but the book trades only a "
                f"strictly-spaced calendar subset of bars ({sched_str}), so "
                f"the shift moves every bet onto bars the schedule never "
                f"trades: for a calendar-keyed signal the value encodes "
                f"WHEN to trade, and shifting it misaligns the scheduled "
                f"bet rather than adding information. An honest scheduled/"
                f"seasonal alpha collapses exactly like this - but so would "
                f"a same-bar leak deliberately zeroed off-schedule, and "
                f"this probe cannot tell them apart.{stamp_note}",
                severity=(Severity.MEDIUM if causal_verified
                          else Severity.HIGH),
                remediation="Verify the trading schedule is genuinely known "
                            "in advance (a pure calendar rule or event "
                            "dates fixed before each bar, not derived from "
                            "the bar's own data); pass signal_func and "
                            "artifacts.signal_input so all three causality "
                            "probes can stamp the signal causal - the "
                            "softest reading requires that stamp.",
                details=details)
        elif anti_corr:
            return warned(
                CHECK_DATE_SHIFT,
                f"peeking one bar EARLIER flips the strategy deeply negative "
                f"(annualized SR {sr_0:.1f} -> {sr_m1:.1f} at shift k=-1; "
                f"curve: {curve_str}) - the shifted book is anti-correlated "
                f"with its own bar. That is what a causal fast-reversal "
                f"signal (signal_t ~ -zscore of bar-t return) does "
                f"mechanically, but it is also what a signal embedding the "
                f"executed bar's return with reversed sign would do; this "
                f"probe alone cannot tell them apart.",
                severity=Severity.HIGH,
                remediation="Pass signal_func and artifacts.signal_input to "
                            "run all three causality probes "
                            "(dynamic.signal_reproducibility, "
                            "dynamic.rolling_window_integrity, "
                            "dynamic.signal_input_sensitivity); the causal "
                            "stamp that clears this warning requires ALL "
                            "THREE to positively PASS - a sensitivity "
                            "WARN/SKIP (an input-insensitive replay) "
                            "withholds it. If all three pass, this collapse "
                            "is the benign reversal artifact.",
                details=details)
        elif ambiguous_pos and causal_verified:
            return warned(
                CHECK_DATE_SHIFT,
                f"peeking one bar EARLIER collapses the strategy to a "
                f"POSITIVE remnant commensurate with its baseline "
                f"(annualized SR {sr_0:.1f} -> {sr_m1:.1f} at shift k=-1, "
                f"remnant ratio {sr_m1 / sr_0:.2f} >= "
                f"{PEEK_AMBIG_MIN_RATIO}; curve: {curve_str}) - an "
                f"ambiguous shape this probe cannot classify alone: an "
                f"honest one-bar-horizon alpha with signal autocorrelation "
                f"phi retains ~phi x baseline at k=-1 (the fresher signal "
                f"vintage predicts the NEXT bar, which the fixed-lag "
                f"pipeline cannot exploit), and a leak blended with an "
                f"honest sleeve leaves the same commensurate remnant when "
                f"the leak evaporates. signal_func was verified causal "
                f"with respect to its input at the sampled dates "
                f"(reproducibility, truncation and input-sensitivity "
                f"probes all PASS), which favors the honest fast-alpha "
                f"reading - but the construction of signal_input itself is "
                f"outside the probes' reach.",
                severity=Severity.MEDIUM,
                remediation="Quantify the signal's autocorrelation and "
                            "horizon: an honest one-bar alpha with lag-1 "
                            "autocorrelation phi should retain ~phi x "
                            "baseline SR at BOTH k=-1 and k=+1. If the "
                            "measured phi does not explain the remnant, "
                            "audit signal_input timestamps and the "
                            "signals/asset_returns calendar for a blended "
                            "leak.",
                details=details)
        elif ambiguous_pos:
            return warned(
                CHECK_DATE_SHIFT,
                f"peeking one bar EARLIER collapses the strategy to a "
                f"POSITIVE remnant commensurate with its baseline "
                f"(annualized SR {sr_0:.1f} -> {sr_m1:.1f} at shift k=-1, "
                f"remnant ratio {sr_m1 / sr_0:.2f} >= "
                f"{PEEK_AMBIG_MIN_RATIO}; curve: {curve_str}) - an "
                f"ambiguous shape: an honest one-bar-horizon alpha with "
                f"signal autocorrelation retains exactly such a remnant "
                f"(the fresher vintage predicts the NEXT bar, which the "
                f"fixed-lag pipeline cannot exploit), and so does a leak "
                f"blended with an honest sleeve; this probe alone cannot "
                f"tell them apart.",
                severity=Severity.HIGH,
                remediation="Pass signal_func and artifacts.signal_input to "
                            "run all three causality probes "
                            "(dynamic.signal_reproducibility, "
                            "dynamic.rolling_window_integrity, "
                            "dynamic.signal_input_sensitivity); the causal "
                            "stamp requires ALL THREE to positively PASS - "
                            "a sensitivity WARN/SKIP withholds it. If all "
                            "three pass, quantify the signal's "
                            "autocorrelation and horizon: an honest "
                            "one-bar alpha with lag-1 autocorrelation phi "
                            "retains ~phi x baseline at k=+-1.",
                details=details)
        elif ambiguous and causal_verified:
            return warned(
                CHECK_DATE_SHIFT,
                f"peeking one bar EARLIER turns the strategy against its own "
                f"bar (annualized SR {sr_0:.1f} -> {sr_m1:.1f} at shift k=-1 "
                f"- below the ~0 noise band (+-{noise_floor:.2f}) but short "
                f"of the full anti-correlation bar ({anti_bar:.1f}); curve: "
                f"{curve_str}) - an ambiguous shape this probe cannot "
                f"classify alone: a causal signal with a small same-bar "
                f"loading (e.g. a mean-reversion term whose window ends at "
                f"bar t) collapses exactly like this when the raw shift "
                f"re-aligns that loading against the earned bar, and so "
                f"would a small reversed-sign embed of the executed bar's "
                f"return. signal_func was verified causal with respect to "
                f"its input at the sampled dates (reproducibility, "
                f"truncation and input-sensitivity probes all PASS), which "
                f"exonerates the signal "
                f"computation - the benign same-bar-loading reading is "
                f"favored, but the timing of signal_input itself is outside "
                f"the probes' reach.",
                severity=Severity.MEDIUM,
                remediation="Quantify the signal's same-bar loading: "
                            "correlate the signal innovation at t with the "
                            "bar-t return; if the implied k=-1 drag matches "
                            "the observed collapse, the shape is the benign "
                            "mechanical artifact. If that loading is ~0, "
                            "audit signal_input timestamps and the "
                            "signals/asset_returns calendar for an "
                            "off-by-one join.",
                details=details)
        elif ambiguous:
            return warned(
                CHECK_DATE_SHIFT,
                f"peeking one bar EARLIER turns the strategy against its own "
                f"bar (annualized SR {sr_0:.1f} -> {sr_m1:.1f} at shift "
                f"k=-1; curve: {curve_str}) - an ambiguous shape: too "
                f"negative for the leak-evaporation signature (SR collapsing "
                f"into the +-{noise_floor:.2f} noise band) yet short of the "
                f"causal fast-reversal signature (SR <= {anti_bar:.1f}). A "
                f"mild causal same-bar loading and a small reversed-sign "
                f"embed of the executed bar's return both produce exactly "
                f"this curve; this probe alone cannot tell them apart.",
                severity=Severity.HIGH,
                remediation="Pass signal_func and artifacts.signal_input to "
                            "run all three causality probes "
                            "(dynamic.signal_reproducibility, "
                            "dynamic.rolling_window_integrity, "
                            "dynamic.signal_input_sensitivity); the causal "
                            "stamp requires ALL THREE to positively PASS - "
                            "a sensitivity WARN/SKIP withholds it. If all "
                            "three pass, the collapse is a benign "
                            "same-bar-loading artifact, not a leak.",
                details=details)
        elif causal_verified:
            return warned(
                CHECK_DATE_SHIFT,
                f"peeking one bar EARLIER erases the edge (annualized SR "
                f"{sr_0:.1f} -> {sr_m1:.1f} at shift k=-1, remnant below "
                f"{PEEK_AMBIG_MIN_RATIO:.0%} of baseline; curve: "
                f"{curve_str}) even though signal_func was verified causal "
                f"with respect to its input at the sampled dates - the edge "
                f"is welded to exactly the executed bar. Two readings "
                f"survive the stamp and this probe cannot separate them: a "
                f"leak outside the signal computation (signal_input "
                f"construction, calendar joins, engine timing), or an "
                f"honest ultra-fast alpha whose autocorrelation is too low "
                f"to leave a commensurate k=-1 remnant.",
                severity=Severity.HIGH,
                remediation="Audit what the truncation probe cannot see: the "
                            "timestamps and construction of signal_input "
                            "itself (pre-shifted columns, publication-time "
                            "joins), calendar alignment between signals and "
                            "asset_returns, and the backtest engine's "
                            "execution timing. Also quantify the signal's "
                            "autocorrelation and horizon: an honest one-bar "
                            "alpha with lag-1 autocorrelation phi retains "
                            "~phi x baseline SR at k=+-1, so a near-zero "
                            "phi is consistent with this collapse.",
                details=details)
        else:
            return failed(
                CHECK_DATE_SHIFT,
                f"peeking one bar EARLIER destroys the strategy (annualized SR "
                f"{sr_0:.1f} -> {sr_m1:.1f} at shift k=-1, remnant below "
                f"{PEEK_AMBIG_MIN_RATIO:.0%} of baseline; curve: {curve_str}) "
                f"- the edge is welded to exactly the executed bar: the "
                f"signature of a signal that embeds that bar's return. (A "
                f"near-zero-autocorrelation ultra-fast alpha leaves the same "
                f"sub-commensurate remnant; only the causality probes can "
                f"adjudicate, and they did not run.)",
                severity=Severity.CRITICAL,
                remediation="Pass signal_func and artifacts.signal_input so "
                            "the causality probes can adjudicate; if all "
                            "three stamp the signal causal, quantify its "
                            "autocorrelation and horizon instead. Otherwise "
                            "rebuild the signal from data available strictly "
                            "before the executed bar: audit every feature "
                            "for use of the bar-t return/price (same-day "
                            "close, unshifted forward returns, labels joined "
                            "on the wrong date) and re-run the backtest with "
                            "the corrected, lagged signal.",
                details=details)

    # (b) one extra bar of delay erases the edge -> ultra-fast alpha or borderline lookahead
    if (np.isfinite(sr_0) and np.isfinite(sr_p1)
            and sr_0 >= DELAY_MIN_BASELINE_SR
            and sr_p1 < config.shift_delay_collapse_ratio * sr_0):
        if sched is not None and not held_refresh:
            # Third reading: on a strictly-spaced scheduled book the
            # delayed bet lands on a bar the schedule never trades - the
            # information (the calendar) was known days in advance, so
            # "ultra-fast alpha" is the wrong frame; the honest question is
            # whether the scheduled bar's fill is achievable at all.
            # held_refresh books deliberately keep the generic HIGH branch
            # below: a held book is in the market every bar, so the
            # sparse-book fill-achievability reading does not transfer -
            # and a genuine same-bar leak on a weekly-held book never
            # trips rule (a) (the embedded bar stays inside the holding
            # window, so sr(-1) barely differs from sr(0)) and is caught
            # only here, so softening this rule for held schedules would
            # be a real false-negative cost.
            return warned(
                CHECK_DATE_SHIFT,
                f"one extra bar of execution delay erases the edge "
                f"(annualized SR {sr_0:.2f} -> {sr_p1:.2f} at shift k=+1, "
                f"ratio {sr_p1 / sr_0:.2f} < "
                f"{config.shift_delay_collapse_ratio}; curve: {curve_str}) "
                f"- on a book trading only a strictly-spaced calendar "
                f"subset of bars ({sched_str}), delay does not lose "
                f"information (the schedule is known in advance): it moves "
                f"the bet off the scheduled bar entirely. A bar-scheduled "
                f"alpha whose scheduled bar simply passes reads like this "
                f"- but so does borderline lookahead welded to that bar.",
                severity=Severity.MEDIUM,
                remediation="Confirm the scheduled bar's fill is physically "
                            "achievable: the signal must be computable "
                            "signal_lag bars before each scheduled bar and "
                            "the order placeable ahead of it. If fills "
                            "realistically land one bar late, the scheduled "
                            "edge does not exist - re-run with signal_lag "
                            "+ 1 and judge that Sharpe instead.",
                details=details)
        return warned(
            CHECK_DATE_SHIFT,
            f"one extra bar of execution delay erases the edge (annualized SR "
            f"{sr_0:.2f} -> {sr_p1:.2f} at shift k=+1, ratio "
            f"{sr_p1 / sr_0:.2f} < {config.shift_delay_collapse_ratio}; curve: "
            f"{curve_str}) - either genuinely ultra-fast alpha or borderline "
            f"lookahead.",
            severity=Severity.HIGH,
            remediation="Confirm the assumed execution timing (signal_lag) is "
                        "physically achievable: can you really compute the signal "
                        "and trade before the bar it earns? If fills arrive one "
                        "bar later than modeled, re-run with signal_lag + 1 and "
                        "judge that Sharpe instead.",
            details=details)

    if peek_exempt:
        return passed(
            CHECK_DATE_SHIFT,
            f"the k=-1 peek rule is not informative for this signal shape: "
            f"shifting the signal one bar earlier makes the book mechanically "
            f"anti-correlated with its own bar (annualized SR {_fmt_sr(sr_0)} "
            f"-> {_fmt_sr(sr_m1)} at k=-1; curve: {curve_str}) - the "
            f"signature of a causal fast-reversal signal, and signal_func was "
            f"independently verified causal with respect to its input at the "
            f"sampled dates (reproducibility, truncation "
            f"and input-sensitivity probes all PASS): no probe evidence "
            f"supports an embedded future return - the collapse is "
            f"consistent with the causal fast-reversal artifact (constants "
            f"baked into the callable and the construction of signal_input "
            f"stay outside the probes' reach).",
            details=details)

    # Degenerate-curve health guard: rules (a)/(b) are the
    # only leak detectors here and both need finite SRs at k=0 and k=-1/+1.
    # When those points are non-finite (an engine's all-or-nothing data gate
    # trips on the |k| all-NaN rows shift(k) manufactures) or the curve is
    # bit-flat (the callable never consumed the shifted panel), a PASS would
    # assert decay/peek behavior that was never observed - SKIP instead.
    finite_curve = [v for v in curve.values() if np.isfinite(v)]
    n_degenerate = len(curve) - len(finite_curve)
    details["n_degenerate_points"] = n_degenerate
    if not np.isfinite(sr_0) or not (np.isfinite(sr_m1) or np.isfinite(sr_p1)):
        return skipped(
            CHECK_DATE_SHIFT,
            f"backtest_func did not produce judgeable shifted re-runs: SR is "
            f"non-finite at k=0 or at both k=-1 and k=+1 "
            f"({n_degenerate}/{len(curve)} curve points degenerate; curve k: "
            f"SR = {curve_str}), so neither the peek rule nor the delay rule "
            f"could measure anything. Verify backtest_func consumes the "
            f"signals argument and tolerates NaN-padded panels - "
            f"signals.shift(k) always creates |k| all-NaN rows at the head "
            f"or tail, and an all-or-nothing data-completeness gate "
            f"degenerates every k != 0 re-run.",
            details=details)
    if (n_degenerate == 0 and len(finite_curve) > 1
            and max(finite_curve) - min(finite_curve) <= DATE_SHIFT_FLAT_TOL):
        return skipped(
            CHECK_DATE_SHIFT,
            f"the SR curve is exactly flat across all {len(curve)} shifts "
            f"(spread <= {DATE_SHIFT_FLAT_TOL:g}; curve k: SR = {curve_str}) "
            f"- shifting the signal changed nothing, so the shifted panel "
            f"never flowed through the strategy and the probe measured "
            f"nothing. Verify backtest_func consumes the signals argument it "
            f"is handed instead of recomputing its book internally.",
            details=details)

    # PASS message claims only what the finite points actually showed.
    claims = []
    if np.isfinite(sr_p1) and sr_0 >= DELAY_MIN_BASELINE_SR:
        claims.append(f"one extra bar of delay does not erase the edge "
                      f"(SR {_fmt_sr(sr_0)} -> {_fmt_sr(sr_p1)} at k=+1)")
    if np.isfinite(sr_m1) and sr_m1 >= sr_0:
        claims.append(f"(illegal) peeking improves it "
                      f"(SR {_fmt_sr(sr_m1)} at k=-1), as an honest "
                      f"pipeline should")
    body = ("; ".join(claims) if claims
            else "neither the peek rule nor the delay rule found the edge "
                 "welded to the executed bar")
    return passed(
        CHECK_DATE_SHIFT,
        f"date-shift curve shows no leak signature (curve k: SR = "
        f"{curve_str}; baseline SR {_fmt_sr(sr_0)}): {body}.",
        details=details)


# ---------------------------------------------------------------------------
# 2 + 3. signal_func probes (share one recomputed signal panel)
# ---------------------------------------------------------------------------

def _signal_func_probes(artifacts: BacktestArtifacts, config: AuditConfig,
                        signal_func
                        ) -> tuple[CheckResult, CheckResult, CheckResult]:
    missing = []
    if signal_func is None:
        missing.append("signal_func")
    if artifacts.signal_input is None:
        missing.append("artifacts.signal_input")
    if missing:
        msg = (f"pass signal_func(signal_input) -> signals DataFrame AND "
               f"artifacts.signal_input (the raw data panel signal_func "
               f"consumes) to enable this check; missing: "
               f"{', '.join(missing)}.")
        return (skipped(CHECK_REPRO, msg), skipped(CHECK_TRUNCATION, msg),
                skipped(CHECK_SENSITIVITY, msg))

    assert artifacts.signal_input is not None  # in `missing` otherwise
    try:
        # .copy(): signal_input is shared by reference with the caller
        # (inputs.aligned() deliberately does not copy it); an in-place-
        # mutating callable must not corrupt the caller's frame or the
        # later probe comparisons.
        full = signal_func(artifacts.signal_input.copy())
    except Exception as exc:  # noqa: BLE001
        return (errored(CHECK_REPRO, exc), errored(CHECK_TRUNCATION, exc),
                skipped(CHECK_SENSITIVITY,
                        "signal_func crashed on the full signal_input - see "
                        "dynamic.signal_reproducibility"))

    # Malformed output guard: a non-DataFrame or duplicated row/column labels
    # make every cell comparison ill-posed (pandas would raise a raw
    # broadcast/reindex error deep inside the coverage math - an opaque
    # ERROR for all three probes). Tell the user what to fix instead of
    # crashing.
    malformed = _malformed_output(full)
    if malformed:
        msg = (f"signal_func(signal_input) returned a malformed output - "
               f"{malformed} - so cells cannot be attributed to a unique "
               f"(date, asset) and neither the reproducibility nor the "
               f"truncation probe can be judged.")
        rem = ("Make signal_func return a DataFrame with one row per date "
               "and one column per asset (deduplicate with "
               "df.loc[~df.index.duplicated()] / "
               "df.loc[:, ~df.columns.duplicated()], or fix the join/concat "
               "that produced the duplicates).")
        return (warned(CHECK_REPRO, msg, severity=Severity.MEDIUM,
                       remediation=rem),
                skipped(CHECK_TRUNCATION,
                        msg + " Fix signal_func's output to enable this "
                              "probe."),
                skipped(CHECK_SENSITIVITY,
                        msg + " Fix signal_func's output to enable this "
                              "probe."))

    # Coverage floor: both probes only prove anything about the cells that
    # signal_func actually returns. Count how many of the audited non-NaN
    # signal cells the output covers with a comparable (non-NaN) value.
    # Reindexing onto the audited grid (instead of intersecting grids) is
    # load-bearing: an output that silently excludes some audited dates/
    # assets must have those cells counted as uncovered - and, below the
    # coverage floor in _reproducibility, as unreproduced.
    sig = artifacts.signals
    audited = sig.notna()
    n_audited = int(audited.to_numpy().sum())
    idx = sig.index.intersection(full.index)
    cols = sig.columns.intersection(full.columns)
    full_grid = full.reindex(index=sig.index, columns=sig.columns)
    n_covered = int((audited & full_grid.notna()).to_numpy().sum())
    coverage = (n_covered / n_audited) if n_audited else 0.0

    return (_reproducibility(artifacts, config, full_grid, idx, cols,
                             coverage, n_audited, n_covered),
            _truncation(artifacts, config, signal_func, full,
                        coverage, n_audited, n_covered),
            _input_sensitivity(artifacts, config, signal_func, full))


def _malformed_output(full) -> str | None:
    """Describe what makes signal_func's output un-auditable, or None."""
    if not isinstance(full, pd.DataFrame):
        return f"a {type(full).__name__}, not a DataFrame"
    parts = []
    dup_rows = full.index[full.index.duplicated()].unique()
    dup_cols = full.columns[full.columns.duplicated()].unique()
    if len(dup_rows):
        parts.append(f"{len(dup_rows)} duplicated index label(s), "
                     f"e.g. {dup_rows[0]!r}")
    if len(dup_cols):
        parts.append(f"{len(dup_cols)} duplicated column label(s), "
                     f"e.g. {dup_cols[0]!r}")
    return " and ".join(parts) or None


def _reproducibility(artifacts: BacktestArtifacts, config: AuditConfig,
                     full_grid: pd.DataFrame, idx: pd.Index, cols: pd.Index,
                     coverage: float, n_audited: int,
                     n_covered: int) -> CheckResult:
    sig = artifacts.signals
    if len(idx) == 0 or len(cols) == 0:
        return warned(
            CHECK_REPRO,
            f"signal_func(signal_input) output shares no (dates x assets) grid "
            f"with artifacts.signals ({len(idx)} common dates, {len(cols)} "
            f"common assets) - the probes below audit signal_func, not the "
            f"backtest you reported.",
            severity=Severity.MEDIUM,
            remediation="Make signal_func return a DataFrame on the same "
                        "date index and asset columns as artifacts.signals.",
            frac_mismatch=1.0, n_cells=0, coverage=0.0,
            n_audited_cells=n_audited, n_covered_cells=0)

    if coverage < REPRO_MIN_COVERAGE:
        return warned(
            CHECK_REPRO,
            f"signal_func(signal_input) covers only {coverage:.1%} of the "
            f"{n_audited} audited non-NaN signal cells with a comparable "
            f"value ({n_covered} covered; floor {REPRO_MIN_COVERAGE:.0%}) - "
            f"whatever it returns may reproduce perfectly, but the "
            f"reproducibility and truncation verdicts transfer only to the "
            f"covered sliver, not to the backtest you reported.",
            severity=Severity.MEDIUM,
            remediation="Make signal_func return the full signal panel: same "
                        "date index and asset columns as artifacts.signals, "
                        "non-NaN wherever the audited signals are non-NaN.",
            frac_mismatch=None, n_cells=0, coverage=coverage,
            n_audited_cells=n_audited, n_covered_cells=n_covered)

    # Compare on the full audited grid, not the grid intersection: an output
    # that omits some audited dates/assets (or returns NaN there) leaves
    # those cells unverifiable, and an unverifiable cell is an unreproduced
    # cell. Intersecting first would let a signal_func exclude up to
    # 1 - REPRO_MIN_COVERAGE = 5% of the audited cells - exactly the leaky
    # ones - from this comparison and still collect the causal stamp with
    # 0.0% mismatch.
    a = sig.to_numpy(dtype=float)
    b = full_grid.to_numpy(dtype=float)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    either = ~(nan_a & nan_b)                      # cells where either side has a value
    one_nan = nan_a ^ nan_b
    close = np.isclose(a, b, rtol=config.truncation_rtol,
                       atol=config.truncation_atol)
    mismatch = one_nan | (~nan_a & ~nan_b & ~close)
    n_cells = int(either.sum())
    frac = float(mismatch.sum() / n_cells) if n_cells else 0.0
    n_uncovered = n_audited - n_covered

    # Concentration gate: the global tolerance
    # models i.i.d. scattered noise, so judge each column (asset) and row
    # (date) against its own comparable cells too - one wholly-replaced
    # asset (or a few whole dates), value-replaced or NaNed-out (one_nan
    # counts), must not ride the panel-wide floor to a causal stamp.
    mm_col = mismatch.sum(axis=0).astype(float)
    mm_row = mismatch.sum(axis=1).astype(float)
    cnt_col = either.sum(axis=0).astype(float)
    cnt_row = either.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_col = np.where(cnt_col > 0, mm_col / np.maximum(cnt_col, 1), 0.0)
        frac_row = np.where(cnt_row > 0, mm_row / np.maximum(cnt_row, 1), 0.0)
    hot_col = ((cnt_col >= REPRO_CONC_MIN_CELLS)
               & (mm_col >= REPRO_CONC_MIN_MISMATCH)
               & (frac_col >= REPRO_CONC_MISMATCH_FRAC))
    hot_row = ((cnt_row >= REPRO_CONC_MIN_CELLS)
               & (mm_row >= REPRO_CONC_MIN_MISMATCH)
               & (frac_row >= REPRO_CONC_MISMATCH_FRAC))
    elig_col = cnt_col >= REPRO_CONC_MIN_CELLS
    elig_row = cnt_row >= REPRO_CONC_MIN_CELLS
    max_col_frac = float(frac_col[elig_col].max()) if elig_col.any() else 0.0
    max_row_frac = float(frac_row[elig_row].max()) if elig_row.any() else 0.0
    conc_details = dict(max_asset_mismatch_frac=max_col_frac,
                        max_date_mismatch_frac=max_row_frac,
                        n_concentrated_assets=int(hot_col.sum()),
                        n_concentrated_dates=int(hot_row.sum()))

    if frac > REPRO_MISMATCH_FRAC:
        uncov = (f", {n_uncovered} of them audited cells the output does "
                 f"not even cover" if n_uncovered else "")
        return warned(
            CHECK_REPRO,
            f"signal_func does not reproduce the audited signals ({frac:.1%} of "
            f"{n_cells} audited-grid cells differ beyond rtol="
            f"{config.truncation_rtol:g}{uncov}) - the probes below audit "
            f"signal_func, not necessarily the backtest you reported.",
            severity=Severity.MEDIUM,
            remediation="Regenerate artifacts.signals from this exact "
                        "signal_func (same parameters, same signal_input "
                        "panel), or pass the signal_func that actually "
                        "produced the audited signals; its output must "
                        "cover every audited non-NaN cell - cells missing "
                        "from the output cannot be verified and count as "
                        "unreproduced.",
            details=conc_details,
            frac_mismatch=frac, n_cells=n_cells, coverage=coverage,
            n_audited_cells=n_audited, n_covered_cells=n_covered)

    if hot_col.any() or hot_row.any():
        if hot_col.any():
            j = int(np.argmax(np.where(hot_col, frac_col, -1.0)))
            worst = (f"asset {sig.columns[j]!r}: {frac_col[j]:.0%} of its "
                     f"{int(cnt_col[j])} comparable cells differ")
        else:
            i = int(np.argmax(np.where(hot_row, frac_row, -1.0)))
            worst = (f"date {pd.Timestamp(sig.index[i]).date()}: "
                     f"{frac_row[i]:.0%} of its {int(cnt_row[i])} comparable "
                     f"cells differ")
        return warned(
            CHECK_REPRO,
            f"signal_func's overall mismatch ({frac:.2%} of {n_cells} "
            f"audited-grid cells) is under the {REPRO_MISMATCH_FRAC:.0%} "
            f"global tolerance, but the mismatch is CONCENTRATED - "
            f"{int(hot_col.sum())} asset(s) and {int(hot_row.sum())} "
            f"date(s) each mismatch >= {REPRO_CONC_MISMATCH_FRAC:.0%} of "
            f"their own comparable cells (worst {worst}). The global "
            f"tolerance models scattered float/re-run noise; a whole "
            f"column/row of mismatch means those audited signals were NOT "
            f"produced by this signal_func, so no causality verdict "
            f"transfers to them (they may embed anything).",
            severity=Severity.MEDIUM,
            remediation="Regenerate the mismatching assets/dates from this "
                        "exact signal_func, or pass the signal_func that "
                        "actually produced every audited column - one "
                        "wholly-different asset can dominate the book while "
                        "staying under the panel-wide mismatch floor.",
            details=conc_details,
            frac_mismatch=frac, n_cells=n_cells, coverage=coverage,
            n_audited_cells=n_audited, n_covered_cells=n_covered)

    return passed(
        CHECK_REPRO,
        f"signal_func reproduces the audited signals: {frac:.2%} of {n_cells} "
        f"comparable cells differ (tolerance rtol={config.truncation_rtol:g}, "
        f"atol={config.truncation_atol:g}; output covers {coverage:.1%} of "
        f"the audited non-NaN cells; max per-asset mismatch "
        f"{max_col_frac:.2%}, max per-date {max_row_frac:.2%} - no "
        f"column/row concentration).",
        details=conc_details,
        frac_mismatch=frac, n_cells=n_cells, coverage=coverage,
        n_audited_cells=n_audited, n_covered_cells=n_covered)


def _fullsample_scaler_fingerprint(sig: pd.DataFrame) -> str | None:
    """Advisory static fingerprint - never a
    gate: constants fitted on the full sample outside signal_func and
    captured in a closure compute honestly on every input, so they pass
    reproducibility, truncation and sensitivity while the audited signals
    are a full-sample normalization (classic lookahead). Frozen fitted
    constants are black-box indistinguishable from a-priori
    hyperparameters, so no behavioral probe can convict; this only flags
    the common accidental construction (sklearn fit-then-closure) for
    human follow-up: per-asset full-sample mean ~ 0 and std ~ 1 (sample
    or population normalization), or min ~ 0 and max ~ 1, to float
    exactness across every audited asset.
    Honest expanding or cross-sectional normalizations miss the exactness
    by orders of magnitude. Returns a short pattern name, or None."""
    a = sig.to_numpy(dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        return None
    cnt = (~np.isnan(a)).sum(axis=0)
    ok = cnt >= FULLSAMPLE_SIG_MIN_OBS
    if int(ok.sum()) < 2:
        return None
    # judge eligible columns only (short columns hit exact 0/1 trivially);
    # any non-conforming eligible column clears the panel - a panel-wide
    # scaler standardizes every column it touches
    b = a[:, ok]
    mu = np.nanmean(b, axis=0)
    sd = np.nanstd(b, axis=0, ddof=1)
    # pandas' default standardization uses sample variance; numpy/scipy
    # and many fitted scalers use population variance. Either convention
    # leaves an exact full-sample fingerprint, including unequal column
    # histories, whose finite observation counts differ.
    sd_population = sd * np.sqrt((cnt[ok] - 1) / cnt[ok])
    unit_sd = ((np.abs(sd - 1.0) <= FULLSAMPLE_SIG_ATOL)
               | (np.abs(sd_population - 1.0) <= FULLSAMPLE_SIG_ATOL))
    if (np.all(np.abs(mu) <= FULLSAMPLE_SIG_ATOL)
            and np.all(unit_sd)):
        return "z-score (per-asset full-sample mean ~ 0, std ~ 1)"
    lo = np.nanmin(b, axis=0)
    hi = np.nanmax(b, axis=0)
    if (np.all(np.abs(lo) <= FULLSAMPLE_SIG_ATOL)
            and np.all(np.abs(hi - 1.0) <= FULLSAMPLE_SIG_ATOL)):
        return "min-max (per-asset full-sample min ~ 0, max ~ 1)"
    return None


def _truncation(artifacts: BacktestArtifacts, config: AuditConfig,
                signal_func, full: pd.DataFrame, coverage: float,
                n_audited: int, n_covered: int) -> CheckResult:
    if coverage < REPRO_MIN_COVERAGE:
        return skipped(
            CHECK_TRUNCATION,
            f"signal_func(signal_input) covers only {coverage:.1%} of the "
            f"{n_audited} audited non-NaN signal cells ({n_covered} covered; "
            f"floor {REPRO_MIN_COVERAGE:.0%}) - a truncation verdict on that "
            f"sliver would be vacuous, not proof the audited signals are "
            f"causal. Make signal_func return the full signal panel to "
            f"enable this probe.",
            coverage=coverage, n_audited_cells=n_audited,
            n_covered_cells=n_covered)

    # Sample across the whole valid signal range (post-warmup), not just the
    # tail: a leak confined to early history (burn-in normalization fitted on
    # the full sample) is invisible to tail-only sampling. Skip only a small
    # head buffer so the truncated recompute has data to work with.
    valid = full.index[full.notna().any(axis=1)]
    candidates = valid[int(np.floor(TRUNCATION_HEAD_SKIP * len(valid))):]
    n_sample = min(int(config.truncation_sample_dates), len(candidates))
    if n_sample == 0:
        return skipped(
            CHECK_TRUNCATION,
            "signal_func(signal_input) produced no non-NaN signal rows - pass "
            "a signal_input panel covering the backtest sample so truncation "
            "dates can be drawn across the signal history.")

    rng = np.random.default_rng(config.seed)
    picks = rng.choice(len(candidates), size=n_sample, replace=False)
    dates = list(candidates[np.sort(picks)])

    n_assets = int(len(full.columns))
    mismatched_dates: list[str] = []
    per_date_max_diff: dict[str, float | None] = {}
    assert artifacts.signal_input is not None  # guarded in _signal_func_probes
    try:
        for t in dates:
            # .copy(): the .loc slice is a view of the caller-shared frame;
            # a mutating callable must not corrupt it between sample dates.
            trunc = signal_func(artifacts.signal_input.loc[:t].copy())
            # Preserve the caller's flat asset-label order.  The public
            # contract permits heterogeneous hashable labels; pandas'
            # default union sorting warns (and raises under ``-W error``)
            # when, for example, a Timestamp and frozenset coexist.  A
            # causal function may legitimately omit an all-NaN late-start
            # asset from an early truncated recomputation, so unequal
            # column indexes are an expected comparison case here.
            cols = full.columns.union(trunc.columns, sort=False)
            full_row = full.loc[t].reindex(cols).to_numpy(dtype=float)
            if t in trunc.index:
                trunc_row = trunc.loc[t].reindex(cols).to_numpy(dtype=float)
            else:
                trunc_row = np.full(len(cols), np.nan)

            nan_f, nan_t = np.isnan(full_row), np.isnan(trunc_row)
            close = np.isclose(full_row, trunc_row,
                               rtol=config.truncation_rtol,
                               atol=config.truncation_atol)
            mm = (nan_f ^ nan_t) | (~nan_f & ~nan_t & ~close)
            if mm.any():
                key = str(pd.Timestamp(t).date())
                mismatched_dates.append(key)
                both = ~nan_f & ~nan_t
                diffs = np.abs(full_row - trunc_row)[both]
                per_date_max_diff[key] = (float(diffs.max())
                                          if diffs.size else None)
    except Exception as exc:  # noqa: BLE001
        return errored(CHECK_TRUNCATION, exc)

    n_mismatched = len(mismatched_dates)
    finite_diffs = [v for v in per_date_max_diff.values() if v is not None]
    max_abs_diff = max(finite_diffs) if finite_diffs else None

    if n_mismatched:
        example = mismatched_dates[0]
        diff_str = (f"max |diff| {max_abs_diff:.1e}" if max_abs_diff is not None
                    else "mismatch in NaN pattern only")
        return failed(
            CHECK_TRUNCATION,
            f"signal at {example} changes when data after that date is removed "
            f"({diff_str} across {n_assets} assets; {n_mismatched}/{n_sample} "
            f"sampled dates affected) - the signal computation reads the "
            f"future. Common causes: full-sample z-score/normalization, "
            f"fitting a scaler on all data, centered rolling windows, global "
            f"ranks.",
            severity=Severity.CRITICAL,
            remediation="Make every step of signal_func causal: trailing (not "
                        "centered) windows; normalize with expanding or "
                        "point-in-time statistics computed from data <= t only; "
                        "rank within each date's cross-section, never across "
                        "the full sample; re-fit any scaler/model inside the "
                        "walk-forward loop.",
            n_sampled=int(n_sample), n_mismatched=int(n_mismatched),
            example_date=example, max_abs_diff=max_abs_diff,
            mismatched_dates=mismatched_dates,
            per_date_max_abs_diff=per_date_max_diff)

    # Scoped PASS: the probe verifies the
    # mapping's dependence on its input argument at the sampled dates only
    # - it cannot see constants baked into the callable (a scaler fitted on
    # the full sample outside signal_func recomputes honestly on every
    # truncated input). The static fingerprint below flags the common
    # accidental form of that seam; advisory only, never a gate.
    fingerprint = _fullsample_scaler_fingerprint(artifacts.signals)
    advisory = ""
    if fingerprint is not None:
        advisory = (
            f" ADVISORY: the audited signals carry the exact fingerprint of "
            f"a full-sample {fingerprint} normalization - if the scaler was "
            f"fitted on the whole sample OUTSIDE signal_func and captured "
            f"as constants, this probe cannot detect it; confirm every "
            f"fitted constant is computed inside signal_func from data <= "
            f"t.")
    return passed(
        CHECK_TRUNCATION,
        f"signal values at {n_sample} sampled dates are identical when "
        f"future data is truncated - no full-sample dependence on the input "
        f"argument detected at those dates (constants baked into the "
        f"callable are outside this probe's reach).{advisory}",
        n_sampled=int(n_sample), n_mismatched=0,
        example_date=None, max_abs_diff=None,
        fullsample_scaler_fingerprint=fingerprint)


# ---------------------------------------------------------------------------
# 4. dynamic.signal_input_sensitivity
# ---------------------------------------------------------------------------

def _values_bit_identical(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """True when every cell matches exactly (NaN == NaN)."""
    try:
        x = a.to_numpy(dtype=float)
        y = b.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return bool(a.equals(b))
    return bool(((x == y) | (np.isnan(x) & np.isnan(y))).all())


def _output_identical(base: pd.DataFrame, other: Any) -> bool:
    """True when ``other`` matches ``base`` exactly (labels and values)."""
    return (isinstance(other, pd.DataFrame)
            and base.index.equals(other.index)
            and base.columns.equals(other.columns)
            and _values_bit_identical(base, other))


def _frac_changed(base: pd.DataFrame, other: Any) -> float:
    """Fraction of cells that differ from ``base`` (1.0 on shape mismatch)."""
    if isinstance(other, pd.DataFrame) and other.shape == base.shape:
        a = base.to_numpy(dtype=float)
        b = other.reindex(index=base.index,
                          columns=base.columns).to_numpy(dtype=float)
        return float((~((a == b) | (np.isnan(a) & np.isnan(b)))).mean())
    return 1.0


def _input_sensitivity(artifacts: BacktestArtifacts, config: AuditConfig,
                       signal_func, full: pd.DataFrame) -> CheckResult:
    """Does signal_func's output actually depend on its input argument?

    A closure-cheating signal_func that ignores signal_input and replays a
    stored panel passes reproducibility trivially, and - if it trims the
    replay to the input's index - passes the truncation probe too,
    laundering an arbitrarily leaky panel through both. Two tells, probed
    at one truncation date t:

    (i)  fabrication: called on input cut at t, the output carries non-NaN
         values on rows after t. No computation of the given input can know
         those values - they come from outside the argument. FAIL. All-NaN
         rows after t are calendar padding (reindexing onto a master
         trading calendar), carry no information, and are tolerated.
    (ii) value-insensitivity: re-called on the same truncated input with
         every row's values permuted across assets. Permutation-invariant
         statistics (breadth, median, quantiles - and the mean, up to float
         summation order) legitimately do not move under that scramble, so
         a bit-identical output triggers a second, magnitude-changing
         perturbation (global sign flip + additive noise). Bit-identical
         under both: WARN, and the causal_verified stamp is withheld - the
         reproducibility/truncation verdicts describe a replay, not a
         computation.
    """
    check = CHECK_SENSITIVITY
    inp = artifacts.signal_input
    assert inp is not None  # guarded in _signal_func_probes
    if inp.shape[1] < SENSITIVITY_MIN_ASSETS:
        return skipped(check,
                       f"signal_input has {inp.shape[1]} column(s); the "
                       f"cross-sectional value scramble needs >= "
                       f"{SENSITIVITY_MIN_ASSETS} assets to perturb - the "
                       f"causal stamp is withheld without this probe.")
    valid = full.index[full.notna().any(axis=1)].intersection(inp.index)
    if not len(valid):
        return skipped(check,
                       "signal_func(signal_input) has no non-NaN rows on "
                       "dates present in signal_input - no truncation date "
                       "exists to probe input sensitivity at.")
    t = valid[int(SENSITIVITY_DATE_FRAC * (len(valid) - 1))]
    t_str = str(pd.Timestamp(t).date())
    trunc_inp = inp.loc[:t]
    try:
        # .copy(): a mutating callable must not corrupt the caller-shared
        # frame, nor trunc_inp itself (compared against the scramble below).
        base = signal_func(trunc_inp.copy())
    except Exception as exc:  # noqa: BLE001 - probes never kill the audit
        return skipped(check,
                       f"signal_func raised on input truncated at {t_str} "
                       f"({type(exc).__name__}: {exc}) - the truncation "
                       f"probe surfaces this; sensitivity not judged.")
    if not isinstance(base, pd.DataFrame):
        return skipped(check,
                       f"signal_func returned a {type(base).__name__}, not "
                       f"a DataFrame - see dynamic.signal_reproducibility.")

    # Fabrication counts informative rows only: an all-NaN row after t is
    # calendar padding (output reindexed onto a master trading calendar),
    # not a replayed value - a stored-panel replay puts real values after t.
    # Consistent with the probe-date selection above, which already filters
    # to notna rows.
    after_t = base.index > t
    informative = base.notna().any(axis=1).to_numpy() & after_t
    beyond = base.index[informative]
    if len(beyond):
        return failed(
            check,
            f"called on signal_input truncated at {t_str}, signal_func "
            f"returned {len(beyond)} row(s) AFTER that date carrying "
            f"non-NaN values (first: {pd.Timestamp(beyond[0]).date()}) - "
            f"values on those dates cannot be computed from the input it "
            f"was given; they are replayed from a stored panel, so every "
            f"reproducibility/truncation verdict is vacuous for this "
            f"callable and the audited signals' construction remains "
            f"unverified.",
            severity=Severity.HIGH,
            remediation="Make signal_func a pure function of its "
                        "signal_input argument: recompute the signal panel "
                        "from the rows it is handed, never return a panel "
                        "captured at definition time.",
            n_rows_beyond=int(len(beyond)),
            n_allnan_rows_beyond=int(after_t.sum() - len(beyond)),
            probe_date=t_str)

    n_base = int(base.notna().to_numpy().sum())
    if n_base == 0:
        return skipped(check,
                       f"signal_func's output on input truncated at {t_str} "
                       f"has no non-NaN cells - nothing to compare under "
                       f"perturbation.")

    rng = np.random.default_rng(config.seed + 1)
    vals = trunc_inp.to_numpy(dtype=float).copy()
    for i in range(vals.shape[0]):
        rng.shuffle(vals[i])
    pert_inp = pd.DataFrame(vals, index=trunc_inp.index,
                            columns=trunc_inp.columns)
    if pert_inp.equals(trunc_inp):
        return skipped(check, "the value scramble left signal_input "
                              "unchanged (constant rows?) - sensitivity "
                              "cannot be probed on this panel.")
    try:
        pert = signal_func(pert_inp)
    except Exception as exc:  # noqa: BLE001
        return skipped(
            check,
            f"signal_func ran on the truncated input but raised on the "
            f"value-perturbed variant ({type(exc).__name__}: {exc}) - "
            f"commonly innocent input validation, but also what a callable "
            f"accepting only its own stored data does, so the causal stamp "
            f"is withheld until the callable runs on perturbed input.",
            probe_date=t_str)

    if not _output_identical(base, pert):
        frac = _frac_changed(base, pert)
        return passed(
            check,
            f"signal_func responds to its input: {frac:.0%} of output cells "
            f"changed under a full-panel value scramble at {t_str} - the "
            f"causality probes audit a real computation of signal_input.",
            frac_cells_changed=frac, probe_date=t_str, perm_invariant=False)

    # Stage 2: the cross-sectional shuffle preserves
    # every permutation-symmetric row statistic (breadth, median, quantiles,
    # and - up to float summation order - the mean), so an honest
    # market-wide/dispersion overlay is exactly invariant to it. Before
    # convicting, perturb magnitudes: a global sign flip plus additive noise
    # scaled to the panel's own std (pure scaling would preserve breadth).
    # A genuine replay is invariant to any input change, so this loses no
    # detection power against the closure cheat.
    rng2 = np.random.default_rng(config.seed + 2)
    base_vals = trunc_inp.to_numpy(dtype=float)
    noise_scale = float(np.nanstd(base_vals))
    if not np.isfinite(noise_scale) or noise_scale == 0.0:
        noise_scale = 1.0
    mag_vals = (-base_vals + SENSITIVITY_MAG_NOISE_FRAC * noise_scale
                * rng2.standard_normal(base_vals.shape))
    mag_inp = pd.DataFrame(mag_vals, index=trunc_inp.index,
                           columns=trunc_inp.columns)
    try:
        mag = signal_func(mag_inp)
    except Exception as exc:  # noqa: BLE001
        return skipped(
            check,
            f"signal_func was invariant to the cross-sectional scramble and "
            f"raised on the sign-flip + noise magnitude perturbation "
            f"({type(exc).__name__}: {exc}) - commonly innocent input "
            f"validation, but also what a callable accepting only its own "
            f"stored data does, so the causal stamp is withheld until the "
            f"callable runs on perturbed input.",
            probe_date=t_str)

    if _output_identical(base, mag):
        return warned(
            check,
            f"perturbing signal_input up to {t_str} (all "
            f"{trunc_inp.shape[0]} rows x {trunc_inp.shape[1]} columns) by "
            f"BOTH a cross-sectional per-row value scramble AND a global "
            f"sign flip + additive noise (std {noise_scale:.2g}) left "
            f"signal_func's output BIT-IDENTICAL across {n_base} non-NaN "
            f"cells - the callable does not respond to its input, so the "
            f"reproducibility and truncation probes verified a replay, not "
            f"the audited signals' construction; the causal stamp is "
            f"revoked.",
            severity=Severity.HIGH,
            remediation="Make signal_func recompute the signal from its "
                        "signal_input argument (a signal genuinely derived "
                        "from the calendar alone should be audited without "
                        "signal_func); a replayed panel's own construction "
                        "is unaudited and may embed anything.",
            n_compared_cells=n_base, probe_date=t_str)

    # Permutation-invariant by design (breadth / median / dispersion
    # market-timing overlays), yet responsive to magnitudes: an honest
    # computation of signal_input, just one the scramble alone cannot judge.
    frac = _frac_changed(base, mag)
    return passed(
        check,
        f"signal_func is invariant to a cross-sectional permutation of its "
        f"input rows - consistent with a market-wide or dispersion "
        f"statistic computed per date, not evidence of a replay - and "
        f"responds to a sign-flip + additive-noise magnitude perturbation "
        f"at {t_str} ({frac:.0%} of output cells changed): the causality "
        f"probes audit a real computation of signal_input.",
        frac_cells_changed=frac, probe_date=t_str, perm_invariant=True)
