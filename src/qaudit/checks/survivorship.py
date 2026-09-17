"""Static survivorship and point-in-time universe checks.

trading_outside_universe judges exposure against the universe at the
position's decision date. Passive drift is not a trade; entries, increases
and reversals need eligibility. Existing holdings can remain through a
bounded rebalance grace, subject to the missing-return rules.

positions_on_missing_returns measures gross exposure whose returns are NaN.
Short interior gaps may be calendar closures when cross-asset evidence and
calendar recurrence support that interpretation. Terminal and long gaps
remain counted. Non-held co-closure and bridge-return checks test whether
an apparent calendar is tied to holdings or removed PnL. Inconclusive
calendar evidence receives a scoped warning when material.

no_exits measures distinct plausibly exiting assets per member-year of
observed universe exposure. Brief out-and-back membership changes are
flickers; sample-edge exits need enough follow-up to assess persistence.
full_history_universe flags near-complete asset histories when no
point-in-time universe artifact is supplied.

Positions at t were decided at t-signal_lag, so trading eligibility uses
universe.shift(signal_lag). Missing delisting outcomes cannot be verified
from the return panel; the exposure disclosure quantifies that limitation.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .._stats import drifted_weights
from ..config import AuditConfig
from ..inputs import BacktestArtifacts
from ..types import CheckResult, Severity, failed, passed, skipped, warned

# Module thresholds and their measurement scope.

# Absolute weight floor for a held position.
POSITION_EPS = 1e-12

# Absolute slack when comparing a held weight with self-financing drift.
# 1e-9 rather than POSITION_EPS: engines that record actually-held weights
# carry NAV-renormalization round-off of ~1e-12..1e-10 in the drift, while
# a genuine top-up is bps of NAV, 5+ orders above. Fresh entries from zero
# still use POSITION_EPS.
HOLD_INCREASE_TOL = max(POSITION_EPS, 1e-9)

# Relative slack for fee or borrow accrual in the NAV denominator.
# It also permits small incremental top-ups during the bounded grace;
# 0.1% compounded over 21 bars is about 2% of the existing position.
HOLD_INCREASE_REL_TOL = 1e-3

# Reference frequency for calendar-time grace constants. Rescale them by
# periods_per_year / 252 so one month of daily bars stays about one month.
DAILY_BARS_PER_YEAR = 252.0

# Minimum grace (in daily bars) for existing holdings awaiting rebalance:
# ~one month covers weekly/monthly rebalance latency. A book with an
# observable slower cadence gets cadence + HOLDOVER_CADENCE_SLACK_BARS
# instead (a quarterly-rebalanced PIT book legally holds a deleted name
# ~63 bars until the next rebalance zeroes it), capped at one annual cycle
# + slack - annual reconstitution (Russell / Jegadeesh-Titman formation)
# is the slowest standard real-world cadence. A name still held past a
# full observed rebalance cycle plus slack exceeds grace at any cadence up
# to annual, so the cap only denies super-annual shelter. An unobservable
# cadence (fewer than two trade bars), or one whose cadence + slack falls
# below this floor, uses the floor, so eternal zombies still convict.
HOLDOVER_GRACE_BARS = 21

# Allowance for decision lag and calendar jitter around observed cadence.
HOLDOVER_CADENCE_SLACK_BARS = 5

# Tolerance on total gross exposure at missing-return cells after eligible
# calendar exclusions: sized to admit ~one holding period per dying/halted
# asset (a weekly-rebalanced book carries an exiting name up to 5 bars),
# not recurring local-holiday NaNs on a union calendar; more means PnL is
# silently zeroed at scale. Exposure weighting preserves concentration
# information; the separate per-bar and terminal gates limit dilution over
# long samples.
MISSING_RETURN_FRAC = 0.01

# Per-date fraction of gross holdings on unexcluded NaN returns. This gate
# catches a concentrated missing bar the whole-sample fraction dilutes.
# 0.25: an equal-weight book needs fewer than ~4 names before one honest
# terminal delisting bar trips it (a delisting at >= 25% of the book is
# material regardless of sample length), while 12-30-name books sit at
# 3-8% per dying name.
MISSING_RETURN_MAX_BAR_GROSS = 0.25

# Cumulative absolute weight on the first NaN bar of terminal return runs,
# in NAV units. Missing delisting outcomes remain unverified even when each
# individual name was diversified on its exit date.
TERMINAL_DELISTING_MAX_GROSS_EXPOSURE = MISSING_RETURN_MAX_BAR_GROSS

# Minimum shared closure dates for a calendar sibling: region siblings on a
# real holiday calendar share every closure date, while independent
# per-name deletions collide on 1-2 dates by chance (worst-PnL days cluster
# on big market days), not a calendar. Shared deletion can still
# manufacture siblings, so the recurrence and corroboration guards apply.
CLOSURE_MIN_SHARED_DATES = 3

# Fraction of live names closed together that triggers the market-wide path,
# where sibling agreement is automatic and only calendar recurrence counts.
# 0.75 rather than higher: sparing a few names to duck it forfeits the
# benefit (spared names keep paying the loss), and honest large-region
# books (a 75%-US global panel) still pass via recurrence.
MARKETWIDE_MIN_CLOSED_FRAC = 0.75

# Circular day-of-year anniversary tolerance: fixed-date holidays drift 0-2
# weekdays via observance shifts and nth-weekday holidays a few days across
# adjacent years; Easter-linked dates drift further and occasionally land
# uncorroborated (~1 date/yr, which the MISSING_RETURN_FRAC tolerance
# absorbs).
MARKETWIDE_RECUR_TOL_DAYS = 5

# Anniversary partners required in sufficiently long samples: a genuine
# holiday recurs in every sampled year, and demanding two partners cuts
# chance matches on deleted dates from roughly a third-to-half to under
# ~10% per date without touching real calendars.
MARKETWIDE_RECUR_MIN_PARTNERS_LONG = 2

# Sample span at which the stronger recurrence partner floor applies.
MARKETWIDE_RECUR_LONG_SPAN_YEARS = 3.0

# Minimum separation for a true anniversary match; adjacent closure bars
# across a year boundary must not count as different annual occurrences.
MARKETWIDE_RECUR_MIN_ANNIV_GAP_DAYS = 300

# A recurrence test needs an anniversary window inside the sample on at
# least one side: 360 = 365 - MARKETWIDE_RECUR_TOL_DAYS. Untestable closure
# mass remains excluded from conviction but produces a scoped warning when
# it exceeds the materiality tolerance.
MARKETWIDE_RECUR_TESTABLE_MARGIN_DAYS = 360

# Annual allowance for testable market-wide closures without recurrence:
# honest ad-hoc full-market closures run ~1/decade (9/11, Sandy) and
# Easter-linked drift leaves ~1 honest holiday/yr outside the anniversary
# window, so more than 1/yr is not a holiday calendar. The count gate
# limits long-sample dilution; unusual legitimate calendars can also
# exceed it, so the result retains a scoped interpretation.
MARKETWIDE_NONRECURRING_MAX_PER_YEAR = 1.0

# Minimum expected non-held co-closures before the deficit guard may fire.
# Expectation under closure/holding independence sums closed * nonheld /
# live; at E >= 8 the honest false-fire probability of the
# NONHELD_COCLOSURE_DEFICIT (0.2) rule is below ~0.5% (Poisson tail).
NONHELD_COCLOSURE_MIN_EXPECTED = 8.0

# Observed/expected floor for non-held co-closures. A holdings-targeted
# deletion can close held names while sparing every non-held peer.
NONHELD_COCLOSURE_DEFICIT = 0.2

# Minimum finite peer returns on each closed and bridge bar for imputing
# the missed factor move. Insufficient peer evidence leaves the run untested.
EXOG_MIN_OPEN_PEERS = 3

# Minimum jointly finite bars for an asset/open-peer factor loading.
EXOG_MIN_BETA_BARS = 60

# Retiming replications on fully observed stretches of the same asset.
# They form a null for destroyed PnL under closure-timing independence.
EXOG_PLACEBOS = 200

# Fixed seed for reproducible placebo retiming.
EXOG_PLACEBO_SEED = 87

# Required downward deviation of destroyed PnL in placebo standard deviations.
EXOG_Z_CRIT = 4.0

# Economic co-gate: destroyed annual return divided by observed annual
# book volatility must also exceed this Sharpe-unit threshold.
EXOG_MIN_DESTROYED_SR = 0.15

# Maximum short interior gap with finite returns on both sides: union-
# calendar holiday clusters run 1-2 bars and Golden Week ~5, so longer
# interior runs are halts, not closures. Closure PnL may be carried in the
# bridge return; terminal and longer gaps remain counted because closure
# accounting cannot be established from this signature.
MARKET_CLOSURE_MAX_RUN_BARS = 5

# Minimum fraction of rows with an in-sample decision universe for a PASS.
# Initial signal_lag rows receive the benefit of the doubt. Measurable
# violations still fail even when coverage is below this floor.
TOU_MIN_MEASURABLE_FRAC = 0.5

# Brief membership exits do not establish persistent attrition: a real exit
# is terminal or stays out until a later reconstitution, and the fastest
# standard cadence is monthly (~21 daily bars), so re-admission within
# ~half that is faster than any membership process. Rescale this window to
# bar frequency; monthly or coarser bars floor to zero. Genuine short
# membership churn is excluded from the rate and warns only when plausible
# attrition is insufficient. Terminal exits also need follow-up beyond
# this window, since a final missing row can mimic an exit.
NO_EXITS_FLICKER_MAX_OUT_DAILY_BARS = 10

# Calendar-time support floor for attrition evidence: real universes lose
# names at a few %/year, so ~a year of history is needed at any bar
# frequency (252 daily bars, 12 monthly bars).
MIN_YEARS_NO_EXITS = 1.0

# Minimum transitions support: very coarse data still needs enough rows.
MIN_BARS_NO_EXITS = 12

# Minimum first-to-last finite-return span fraction for full-history coverage.
FULL_SPAN_COVERAGE = 0.99


def run(artifacts: BacktestArtifacts, config: AuditConfig, *,
        signal_func=None, backtest_func=None) -> list[CheckResult]:
    """Run all survivorship checks. Never mutates ``artifacts``."""
    return [
        _trading_outside_universe(artifacts),
        _positions_on_missing_returns(artifacts),
        _no_exits(artifacts, config),
        _full_history_universe(artifacts, config),
    ]


# ---------------------------------------------------------------------------
# 1. trading_outside_universe (CRITICAL)
# ---------------------------------------------------------------------------

def _trading_outside_universe(artifacts: BacktestArtifacts) -> CheckResult:
    check = "survivorship.trading_outside_universe"
    missing = [name for name in ("universe", "positions")
               if getattr(artifacts, name) is None]
    if missing:
        wanted = " and ".join(f"artifacts.{m}" for m in missing)
        return skipped(
            check,
            f"pass {wanted} to enable this check "
            f"(universe: dates x assets bool, True iff the asset was tradeable "
            f"as known at t; positions: dates x assets weights held during t).",
            missing=missing,
        )

    pos = artifacts.positions
    uni = artifacts.universe
    assert pos is not None and uni is not None  # guaranteed by the skip above
    lag = int(artifacts.signal_lag)
    n_periods = len(pos.index)
    # Fraction of position rows with an in-sample decision universe: rows
    # before `lag` are treated as legal by fabrication, so a clean scan over
    # mostly-fabricated rows proves nothing (an unmeasurable check says so
    # instead of silently passing - the lookahead-module contract).
    measurable_frac = (max(0, n_periods - lag) / n_periods
                       if n_periods else 0.0)
    if lag >= n_periods > 0:
        return skipped(
            check,
            f"signal_lag={lag} >= {n_periods} periods: every in-sample "
            f"position was decided before the sample starts, so the "
            f"decision-time universe is unobservable for 100% of rows - "
            f"out-of-universe trading cannot be measured (it would have "
            f"PASSed over a fully fabricated legal universe). Extend the "
            f"universe/positions history to cover at least signal_lag "
            f"periods before the first position row.",
            n_periods=n_periods, signal_lag=lag,
            measurable_frac=0.0,
        )

    uni_arr = uni.to_numpy(dtype=bool)
    # universe at decision time t - lag; the first `lag` rows have no decision
    # history inside the sample -> treated as legal (True).
    decision_uni = np.ones_like(uni_arr)
    if 0 < lag < n_periods:
        decision_uni[lag:] = uni_arr[:-lag]

    pos_arr = pos.to_numpy(dtype=float)
    held = np.abs(pos_arr) > POSITION_EPS
    # Only new, increased, or reversed exposure while out-of-universe is
    # hindsight; merely holding (or unwinding) a position through a delisting
    # until the next rebalance is normal portfolio mechanics, not lookahead.
    # A sign flip counts as new exposure even at identical magnitude: +0.20
    # -> -0.20 closes the old position and opens fresh opposite-side risk on
    # a name only hindsight says is tradeable. And the hold tolerance is
    # bounded: a "hold" that persists past HOLDOVER_GRACE_BARS after the exit
    # is a zombie position, not rebalance latency.
    #
    # The contract says positions[t] are weights actually held during t.
    # Consequently there is one economically coherent no-trade reference:
    # passive drift, w_drift[t] (_stats.drifted_weights). Repeating yesterday's
    # target is not a second kind of hold: whenever returns disperse, restoring
    # that target buys and sells real dollars (the same convention used by
    # traded_dollars_series and every cost check). Fresh exposure is therefore
    # measured only against drift: |w| above |drift| x
    # (1 + HOLD_INCREASE_REL_TOL) + HOLD_INCREASE_TOL, entry from zero, or a
    # sign reversal. The relative slack absorbs the common engine
    # convention of dividing held weights by a NAV net of a per-bar fee /
    # borrow / cost accrual (weights sit above the exact drift by the drag
    # fraction, ~1e-4/bar) without sheltering a real top-up (bps of NAV,
    # >= 1e-2 relative). Rows where drift is undefined (NAV growth <= 0) have
    # no self-financing continuation; a subsequent nonzero out-of-universe
    # weight is conservatively treated as a fresh entry, never as a fabricated
    # constant-target hold.
    abs_pos = np.abs(pos_arr)
    drift_pos = drifted_weights(pos, artifacts.asset_returns)
    drift_defined = np.isfinite(drift_pos).all(axis=1)
    drift_ref = np.where(np.isfinite(drift_pos), drift_pos, 0.0)

    def _fresh_vs(ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ref_abs = np.abs(ref)
        inc = ((abs_pos > ref_abs * (1.0 + HOLD_INCREASE_REL_TOL)
                + HOLD_INCREASE_TOL)
               | (held & (ref_abs <= POSITION_EPS)))     # entry from zero
        flip = (pos_arr * ref < 0) & (ref_abs > POSITION_EPS)
        return inc, flip

    increased, flipped = _fresh_vs(drift_ref)
    out = ~decision_uni
    bars_out = np.zeros_like(pos_arr, dtype=np.int64)
    for t in range(1, out.shape[0]):
        bars_out[t] = np.where(out[t], bars_out[t - 1] + 1, 0)
    # Holdover grace scales with the book's observed trade cadence: median
    # gap between bars where any recorded weight differs materially from its
    # self-financing drift. Raw row changes are not trades - a genuine
    # buy-and-hold book changes weight every bar as its assets disperse - and
    # repeating an unchanged target can trade every bar. Use the same absolute
    # plus relative storage tolerance as the fresh-exposure test. Book-wide,
    # not per-asset: per-asset inference would let a pinned column stretch its
    # own grace to infinity. Fewer than two trade bars falls back to the
    # 21-daily-bar floor.
    trade_step = np.abs(pos_arr - drift_ref)
    trade_tol = (np.abs(drift_ref) * HOLD_INCREASE_REL_TOL
                 + HOLD_INCREASE_TOL)
    trade_rows = (trade_step > trade_tol).any(axis=1) | ~drift_defined
    if trade_rows.size:
        trade_rows[0] = False       # entry establishes the book, not a cadence
    change_bars = np.flatnonzero(trade_rows)
    cadence = (int(np.median(np.diff(change_bars)))
               if len(change_bars) >= 2 else 0)
    # Rescale daily-bar grace constants to the artifact frequency so
    # rebalance latency is measured in calendar time.
    ppy = float(artifacts.periods_per_year)
    bar_scale = ppy / DAILY_BARS_PER_YEAR
    floor_bars = max(1, int(np.ceil(HOLDOVER_GRACE_BARS * bar_scale)))
    slack_bars = max(1, int(np.ceil(HOLDOVER_CADENCE_SLACK_BARS * bar_scale)))
    cap_bars = int(np.ceil(ppy)) + slack_bars     # one annual cycle + slack
    grace = max(floor_bars, min(cadence + slack_bars, cap_bars))
    # Entries, increases and reversals receive no grace on finite-return
    # bars. On NaN-return bars a privileged holdover may wiggle up
    # (overlapping-tranche books shuffle a decaying dead-name weight by
    # ~25% relative during roll-off, beyond any storage tolerance); that is
    # safe to exempt because no PnL is earnable on a NaN-return bar, and
    # such cells still convict as zombies past grace and are covered by the
    # separate missing-return exposure check.
    ret_na = (artifacts.asset_returns
              .reindex(index=pos.index, columns=pos.columns)
              .isna().to_numpy())
    fresh = increased | flipped
    # Holdover privilege requires a position at the last in-universe bar.
    # Carry that state through the out-run even if a rolling tranche briefly
    # has zero weight. A position first opened outside the universe has no
    # privilege. The initial row receives the same benefit of the doubt as
    # the zero-padded previous-position convention.
    priv = np.zeros_like(held)
    if held.shape[0]:
        priv[0] = held[0]
        for t in range(1, held.shape[0]):
            priv[t] = np.where(out[t], priv[t - 1], held[t])
    active = held & out & fresh & ~(ret_na & priv)
    zombie = held & out & ~active & (bars_out > grace)
    viol = active | zombie
    holdover = held & out & ~viol
    n_active = int(active.sum())
    n_zombie = int(zombie.sum())
    n_violations = int(viol.sum())
    n_holdover = int(holdover.sum())
    n_held = int(held.sum())
    n_flips = int((active & flipped).sum())
    n_deadbar_fresh = int((held & out & fresh & ret_na & priv).sum())
    n_born_outside_deadbar = int((held & out & fresh & ret_na & ~priv).sum())
    details: dict[str, object] = dict(
        n_violations=n_violations, n_active_cells=n_active,
        n_zombie_cells=n_zombie, n_holdover_cells=n_holdover,
        n_sign_flip_cells=n_flips, n_position_cells=n_held,
        n_deadbar_fresh_cells=n_deadbar_fresh,
        n_born_outside_deadbar_cells=n_born_outside_deadbar,
        n_undefined_drift_rows=int((~drift_defined).sum()),
        holdover_grace_bars=int(grace), rebalance_cadence_bars=int(cadence),
        signal_lag=lag, measurable_frac=float(measurable_frac),
    )

    if n_violations == 0:
        # A clean scan is only evidence when most rows were actually
        # measurable; violations (below) are evidence at any coverage -
        # the fabricated-legal rows can only hide them, not invent them.
        if measurable_frac < TOU_MIN_MEASURABLE_FRAC:
            return skipped(
                check,
                f"signal_lag={lag} leaves only {n_periods - lag} of "
                f"{n_periods} position rows ({measurable_frac:.0%}, < the "
                f"{TOU_MIN_MEASURABLE_FRAC:.0%} floor) with an in-sample "
                f"decision universe - the other {lag} rows are treated as "
                f"legal by fabrication, so a clean scan here is not "
                f"evidence (0 violations found on the measurable rows, but "
                f"out-of-universe trading is mostly unmeasurable). Extend "
                f"the history or check the declared signal_lag.",
                details=details,
            )
        deadbar_note = (f", {n_deadbar_fresh} of them accounting wiggles on "
                        f"NaN-return bars during dead-name roll-off - no PnL "
                        f"is earnable there" if n_deadbar_fresh else "")
        note = (f" ({n_holdover} cells merely hold/unwind positions through a "
                f"universe exit until the next rebalance - normal mechanics, "
                f"not hindsight{deadbar_note})" if n_holdover else "")
        fab_note = (f" The first {lag} row(s) predate the in-sample decision "
                    f"history and are treated as legal." if lag else "")
        return passed(
            check,
            f"no position is entered, increased, or sign-flipped on an asset "
            f"outside the universe at decision time (t - signal_lag={lag}) "
            f"across {n_held} nonzero cells over {n_periods} periods{note}."
            f"{fab_note}",
            severity=Severity.CRITICAL,
            details=details,
        )

    t_first, a_first = np.argwhere(viol)[0]
    first_date = str(pos.index[int(t_first)].date())
    first_asset = str(pos.columns[int(a_first)])
    n_assets_affected = int(viol.any(axis=0).sum())
    details.update(first_date=first_date, first_asset=first_asset,
                   n_assets_affected=n_assets_affected)
    # The message names the prong that actually fired: ENTER/INCREASE/
    # REVERSE wording (and the "hindsight universe construction" diagnosis)
    # only for cells with fresh out-of-universe exposure; pure holds past
    # grace are reported as zombie holdings, never as entries.
    # The zombie wording must be true for the book at hand: with an observed
    # cadence the grace already covers one full rebalance cycle plus slack,
    # so exceeding it means the next scheduled rebalance came and went with
    # the dead name still held; a constant book has no cadence to cite.
    if cadence and cadence + slack_bars <= grace:
        cadence_txt = (f"beyond the observed {cadence}-bar rebalance cadence "
                       f"plus {slack_bars}-bar slack")
    elif cadence:
        cadence_txt = (f"beyond the {grace}-bar annual-cycle grace ceiling "
                       f"(observed cadence {cadence} bars is slower than any "
                       f"standard rebalance schedule)")
    else:
        cadence_txt = "no rebalance cadence observable (constant book)"
    if n_active:
        zombie_txt = (f", plus {n_zombie} cells held > {grace} bars past the "
                      f"universe exit ({cadence_txt})"
                      if n_zombie else "")
        msg = (f"{n_active} position-cells ENTER, INCREASE, or REVERSE "
               f"exposure on assets outside the tradeable universe at "
               f"decision time ({n_flips} sign flips; first violation: "
               f"{first_asset} on {first_date}){zombie_txt} - hindsight "
               f"universe construction.")
        remediation = (
            "Rebuild the universe from point-in-time constituent data and "
            "mask position construction with universe.shift(signal_lag): a "
            "weight may only be assigned to names investable when the trade "
            "was decided, never to names known-good only in hindsight."
        )
    else:
        msg = (f"{n_zombie} position-cells are held more than {grace} bars "
               f"past their asset's universe exit - {cadence_txt} - "
               f"(0 entered/increased/sign-flipped while out; first: "
               f"{first_asset} on {first_date}) - zombie holdings on names "
               f"only hindsight keeps tradeable.")
        remediation = (
            "Unwind names that leave the universe at the next rebalance (or "
            "model an explicit exit rule); a holding that persists past the "
            f"{grace}-bar grace is not rebalance latency. The grace already "
            "scales with the book's observed rebalance cadence "
            f"({cadence or 'none observed'} bars), capped at one annual "
            "cycle plus slack."
        )
    return failed(
        check, msg,
        severity=Severity.CRITICAL,
        remediation=remediation,
        details=details,
    )


# ---------------------------------------------------------------------------
# 2. positions_on_missing_returns (HIGH, WARN)
# ---------------------------------------------------------------------------

def _marketwide_recurrence(
        index: pd.Index,
        mw_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Year-over-year recurrence corroboration for market-wide closure dates.

    A real market-wide holiday recurs on (nearly) the same calendar date in
    every sampled year; a deleted crash/worst-PnL row does not. Returns
    ``(ok_by_row, untestable_by_row, n_recurring)`` where ``ok_by_row`` is
    True on rows whose market-wide closure is recurrence-corroborated or
    untestable (no anniversary window inside the sample - excluded from the
    tolerance, so sub-2-year honest books never convict here; the caller
    surfaces untestable mass above the tolerance as a scoped cannot-verify
    WARN rather than an unsupported PASS).
    """
    ok = np.zeros(len(index), dtype=bool)
    untest = np.zeros(len(index), dtype=bool)
    rows = np.flatnonzero(mw_rows)
    if rows.size == 0:
        return ok, untest, 0
    if not isinstance(index, pd.DatetimeIndex):
        # validate() enforces a DatetimeIndex on the supported path; a raw
        # integer grid has no calendar to test - unverifiable, not clean.
        ok[rows] = True
        untest[rows] = True
        return ok, untest, 0
    dts = index[rows]
    start, end = index[0], index[-1]
    span_days = int((end - start).days)
    needed = (MARKETWIDE_RECUR_MIN_PARTNERS_LONG
              if span_days >= 365.25 * MARKETWIDE_RECUR_LONG_SPAN_YEARS
              else 1)
    doy = np.asarray([ts.timetuple().tm_yday for ts in dts], dtype=np.int64)
    yr = np.asarray([ts.year for ts in dts], dtype=np.int64)
    day_ord = np.asarray([int((ts - start).days) for ts in dts],
                         dtype=np.int64)
    n_recurring = 0
    for i, r in enumerate(rows):
        testable = (
            day_ord[i] >= MARKETWIDE_RECUR_TESTABLE_MARGIN_DAYS
            or span_days - day_ord[i] >= MARKETWIDE_RECUR_TESTABLE_MARGIN_DAYS
        )
        if not testable:
            ok[r] = True
            untest[r] = True
            continue
        dd = np.abs(doy - doy[i])
        circ = np.minimum(dd, 365 - dd)          # circular day-of-year
        anniv = (np.abs(day_ord - day_ord[i])
                 >= MARKETWIDE_RECUR_MIN_ANNIV_GAP_DAYS)
        partners = int(((yr != yr[i]) & anniv
                        & (circ <= MARKETWIDE_RECUR_TOL_DAYS)).sum())
        if partners >= needed:
            ok[r] = True
            n_recurring += 1
    return ok, untest, n_recurring


def _nonheld_coclosure_deficit(closure: np.ndarray, held: np.ndarray,
                               open_arr: np.ndarray,
                               mw_rows: np.ndarray) -> tuple[float, int, bool]:
    """Co-closure guard: a real holiday closes held and non-held names alike.

    Under closure/holding independence the expected non-held co-closure
    count over non-market-wide closure dates is sum(n_closed * n_nonheld /
    n_live); a held-targeted deletion observes ~0 there. This is also the
    same-book-sibling corroboration cap: siblings that are all held names
    while non-held names never co-close are the book corroborating itself,
    and their evidence is revoked wholesale. Vacuous (never fires) for
    books that hold the whole panel - those have no non-held names to
    spare and are covered by the market-wide and exogeneity layers
    instead.
    """
    cl_count = closure.sum(axis=1)
    dates = (cl_count > 0) & ~mw_rows
    if not dates.any():
        return 0.0, 0, False
    alive = closure | open_arr
    n_live = alive.sum(axis=1)
    n_nonheld = (alive & ~held).sum(axis=1)
    e_terms = (cl_count[dates] * n_nonheld[dates]
               / np.maximum(n_live[dates], 1))
    expected = float(e_terms.sum())
    observed = int((closure & ~held)[dates].sum())
    fired = (expected >= NONHELD_COCLOSURE_MIN_EXPECTED
             and observed < NONHELD_COCLOSURE_DEFICIT * expected)
    return expected, observed, fired


class _ExogResult(NamedTuple):
    testable: bool
    fired: bool
    z: float
    destroyed_sr: float
    n_runs: int
    tested_cells: np.ndarray


def _bridging_exogeneity(ret_vals: np.ndarray, pos_vals: np.ndarray,
                         runs: list[tuple[int, int, int]],
                         candidate_cells: np.ndarray,
                         ppy: float) -> _ExogResult:
    """Bridging guard (load-bearing): an honest halt conserves cumulative PnL.

    For each corroborated closure run the open-peer median factor implies
    the move the closed name missed; an honest closure realizes that move
    in the post-gap bridging return (bridge excess ~= beta * implied gap
    move - true for real holidays and for 2020-03-style halts on crash
    days), while deletion destroys it (the bridge is an ordinary one-day
    return). The statistic is the total destroyed PnL

        sum over runs of [ sum_t pos*beta*F_hat  -  pos_bridge *
                           (r_bridge - beta*F_hat_bridge) ]

    z-scored against EXOG_PLACEBOS re-timings of the same runs at random
    fully-observed stretches (closure-timing/PnL independence null).
    Honest books - random holiday dates or conserving halts on crash days -
    measure z ~ 0; deleting shared loss days measures z << -EXOG_Z_CRIT with a
    material destroyed-Sharpe. Runs without EXOG_MIN_OPEN_PEERS open peers
    are untestable and keep the benefit of the doubt (a near-empty
    cross-section is market-wide-classified anyway).
    """
    n, m = ret_vals.shape
    empty = np.zeros((n, m), dtype=bool)
    open_arr = np.isfinite(ret_vals)
    open_cnt = open_arr.sum(axis=1)
    fhat = np.full(n, np.nan)
    frows = open_cnt >= EXOG_MIN_OPEN_PEERS
    if frows.any():
        fhat[frows] = np.nanmedian(ret_vals[frows], axis=1)
    fin_f = np.isfinite(fhat)
    if int(fin_f.sum()) < EXOG_MIN_BETA_BARS:
        return _ExogResult(False, False, 0.0, 0.0, 0, empty)
    # per-asset loading on the open-peer median factor (masked OLS slope).
    # Bridge bars (the bar after any closure run) are excluded from the
    # regression: reopening names print multi-day - or, honestly halted,
    # compounded - returns there against a contaminated cross-sectional
    # median, which inflates the closed group's betas and alone fakes a
    # destroyed-PnL signal on a conserving book.
    bridge_rows = np.zeros(n, dtype=bool)
    for _, _, e_run in runs:
        if e_run + 1 < n:
            bridge_rows[e_run + 1] = True
    joint = open_arr & fin_f[:, None] & ~bridge_rows[:, None]
    nj = joint.sum(axis=0).astype(float)
    fcol = np.where(fin_f, fhat, 0.0)
    x = np.where(joint, fcol[:, None], 0.0)
    y = np.where(joint, np.nan_to_num(ret_vals), 0.0)
    sx = x.sum(axis=0)
    sy = y.sum(axis=0)
    sxx = (x * x).sum(axis=0)
    sxy = (x * y).sum(axis=0)
    var_n = nj * sxx - sx * sx
    betas = np.where(
        (nj >= EXOG_MIN_BETA_BARS) & (var_n > 0),
        (nj * sxy - sx * sy) / np.where(var_n > 0, var_n, 1.0),
        np.nan,
    )
    pos_f = np.nan_to_num(pos_vals)
    cpf = np.cumsum(np.where(fin_f[:, None], pos_f * fcol[:, None], 0.0),
                    axis=0)
    cum_open = np.cumsum(open_arr.astype(np.int64), axis=0)
    cum_f = np.cumsum(fin_f.astype(np.int64))

    def _valid_starts(j: int, run_len: int) -> np.ndarray:
        # placebo start s: r_j finite over [s-1, s+L] (a fully-observed
        # stretch) and the factor defined over [s, s+L]
        s = np.arange(1, n - run_len)
        oj = cum_open[:, j]
        w_open = oj[s + run_len] - np.where(s >= 2, oj[s - 2], 0)
        w_f = cum_f[s + run_len] - cum_f[s - 1]
        return s[(w_open == run_len + 2) & (w_f == run_len + 1)]

    def _bridge_factor(s: int, e: int) -> float:
        # Use peers open throughout the gap and bridge. Names reopening
        # with the closed group carry multi-day returns and would distort
        # the single-bar factor comparison even when PnL is conserved.
        peers = open_arr[s:e + 1].all(axis=0) & open_arr[e + 1]
        if int(peers.sum()) < EXOG_MIN_OPEN_PEERS:
            return float("nan")
        return float(np.median(ret_vals[e + 1, peers]))

    tested: list[tuple[int, int, int]] = []
    starts_cache: dict[tuple[int, int], np.ndarray] = {}
    d_real = 0.0
    cells = np.zeros((n, m), dtype=bool)
    for j, s, e in runs:
        if not candidate_cells[s:e + 1, j].any():
            continue
        if np.isnan(betas[j]):
            continue
        run_len = e - s + 1
        if int(cum_f[e + 1] - cum_f[s - 1]) != run_len + 1:
            continue                     # factor unobservable in gap/bridge
        fb = _bridge_factor(s, e)
        if not np.isfinite(fb):
            continue                     # no clean bridge peers: untestable
        key = (j, run_len)
        if key not in starts_cache:
            starts_cache[key] = _valid_starts(j, run_len)
        if starts_cache[key].size == 0:
            continue
        implied = float(betas[j]) * float(cpf[e, j] - cpf[s - 1, j])
        bridge = float(pos_f[e + 1, j]) * (
            float(ret_vals[e + 1, j]) - float(betas[j]) * fb)
        d_real += implied - bridge
        tested.append((j, s, e))
        cells[s:e + 1, j] = True
    if not tested:
        return _ExogResult(False, False, 0.0, 0.0, 0, empty)
    rng = np.random.default_rng(EXOG_PLACEBO_SEED)
    d_plac = np.zeros(EXOG_PLACEBOS)
    for j, s, e in tested:
        run_len = e - s + 1
        starts = starts_cache[(j, run_len)]
        s0 = starts[rng.integers(0, starts.size, EXOG_PLACEBOS)]
        e0 = s0 + run_len - 1
        implied_p = betas[j] * (cpf[e0, j] - cpf[s0 - 1, j])
        # same strict bridge-peer definition as the real runs: peers open
        # over [s0, e0] and at e0+1, falling back to the global median on
        # the (fully-observed, hence rare) short-peer placebo rows
        gap_cnt = cum_open[e0] - cum_open[s0 - 1, :]     # (P, m)
        peers_p = (gap_cnt == run_len) & open_arr[e0 + 1]
        vals_p = np.where(peers_p, ret_vals[e0 + 1], np.nan)
        enough = peers_p.sum(axis=1) >= EXOG_MIN_OPEN_PEERS
        fb_p = np.where(
            enough,
            np.nanmedian(np.where(enough[:, None], vals_p, 0.0), axis=1),
            fhat[e0 + 1],
        )
        bridge_p = pos_f[e0 + 1, j] * (ret_vals[e0 + 1, j]
                                       - betas[j] * fb_p)
        d_plac += implied_p - bridge_p
    sd = float(d_plac.std())
    if not np.isfinite(sd) or sd <= 0.0:
        return _ExogResult(False, False, 0.0, 0.0, len(tested), empty)
    z = float((d_real - float(d_plac.mean())) / sd)
    pnl = (pos_f * np.nan_to_num(ret_vals)).sum(axis=1)
    ann_vol = float(pnl.std()) * float(np.sqrt(ppy))
    destroyed_sr = ((-d_real) * (ppy / n) / ann_vol if ann_vol > 0 else 0.0)
    fired = z <= -EXOG_Z_CRIT and destroyed_sr >= EXOG_MIN_DESTROYED_SR
    return _ExogResult(True, fired, z, float(destroyed_sr), len(tested),
                       cells)


def _positions_on_missing_returns(artifacts: BacktestArtifacts) -> CheckResult:
    check = "survivorship.positions_on_missing_returns"
    if artifacts.positions is None:
        return skipped(
            check,
            "pass artifacts.positions (dates x assets weights held during t) "
            "to enable this check.",
            missing=["positions"],
        )

    pos = artifacts.positions
    held = pos.abs().gt(POSITION_EPS)
    ret_na = artifacts.asset_returns.isna()
    viol_all = held & ret_na
    # Classify short interior gaps before counting missing exposure. Finite
    # returns on both sides permit closure/bridge accounting; terminal and
    # long interior gaps remain counted regardless of calendar evidence.
    closure = np.zeros(ret_na.shape, dtype=bool)
    runs: list[tuple[int, int, int]] = []       # (asset col, first, last bar)
    terminal_start: dict[int, int] = {}         # col -> first bar of a NaN
    #                                             run reaching the sample end
    na_arr = ret_na.to_numpy()
    n_rows = na_arr.shape[0]
    for j in range(na_arr.shape[1]):
        col = na_arr[:, j]
        if not col.any():
            continue
        edges = np.diff(col.astype(np.int8))
        starts = np.flatnonzero(edges == 1) + 1
        ends = np.flatnonzero(edges == -1)       # inclusive last-NaN index
        if col[0]:
            starts = np.concatenate((np.array([0]), starts))
        if col[-1]:
            ends = np.concatenate((ends, np.array([n_rows - 1])))
            terminal_start[j] = int(starts[-1])
        for s, e in zip(starts, ends):
            # runs are maximal, so s > 0 / e < n-1 imply a finite return
            # immediately before/after - the interior (closure) signature
            if s > 0 and e < n_rows - 1 \
                    and e - s + 1 <= MARKET_CLOSURE_MAX_RUN_BARS:
                closure[s:e + 1, j] = True
                runs.append((j, int(s), int(e)))
    # Short gaps need calendar siblings sharing enough closure dates.
    # Shared deletions can also create siblings, so market-wide dates instead
    # need annual recurrence. Other corroboration is removed when non-held
    # co-closures are deficient or the bridge-return test indicates destroyed
    # PnL. Real closures should affect held and non-held names and preserve
    # the missed move in the bridge. Uncorroborated material mass remains
    # ambiguous and receives WARN, including single-name market groups.
    viol_arr = viol_all.to_numpy()
    held_arr = held.to_numpy()
    open_arr = ~na_arr
    guards_fired: list[str] = []
    n_mw_dates = n_mw_recurring = n_mw_untestable = 0
    nonheld_exp, nonheld_obs = 0.0, 0
    exog = _ExogResult(False, False, 0.0, 0.0, 0, np.zeros_like(closure))
    n_stripped = 0
    mw_untest = np.zeros(closure.shape[0], dtype=bool)
    if closure.any():
        cl_int = closure.astype(np.int64)
        shared = cl_int.T @ cl_int        # pairwise co-closure date counts
        np.fill_diagonal(shared, 0)
        partners = (shared >= CLOSURE_MIN_SHARED_DATES).astype(np.int64)
        sib = closure & ((cl_int @ partners) > 0)
        # (a) market-wide dates: recurrence replaces sibling evidence
        cl_count = closure.sum(axis=1)
        n_live = (closure | open_arr).sum(axis=1)
        mw_rows = (cl_count > 0) & (cl_count
                                    >= MARKETWIDE_MIN_CLOSED_FRAC * n_live)
        n_mw_dates = int(mw_rows.sum())
        mw_ok, mw_untest, n_mw_recurring = _marketwide_recurrence(
            viol_all.index, mw_rows)
        n_mw_untestable = int(mw_untest.sum())
        corro_sib = sib & ~mw_rows[:, None]
        corro_mw = closure & (mw_rows & mw_ok)[:, None]
        # (b) corroboration guards over the sibling-corroborated mass
        n_corro_sib_held = int((corro_sib & viol_arr).sum())
        stripped = np.zeros_like(closure)
        if n_corro_sib_held:
            nonheld_exp, nonheld_obs, deficit_fired = \
                _nonheld_coclosure_deficit(closure, held_arr, open_arr,
                                           mw_rows)
            if deficit_fired:
                guards_fired.append("nonheld_coclosure_deficit")
                stripped = corro_sib.copy()
            else:
                exog = _bridging_exogeneity(
                    artifacts.asset_returns.to_numpy(dtype=float),
                    pos.to_numpy(dtype=float), runs, corro_sib & viol_arr,
                    float(artifacts.periods_per_year))
                if exog.fired:
                    guards_fired.append("bridging_exogeneity")
                    stripped = exog.tested_cells & corro_sib
        n_stripped = int((stripped & viol_arr).sum())
        corroborated = (corro_sib & ~stripped) | corro_mw
    else:
        corroborated = closure
        mw_rows = np.zeros(closure.shape[0], dtype=bool)
        mw_ok = np.zeros(closure.shape[0], dtype=bool)
    closure_cells = viol_arr & closure
    corro_cells = viol_arr & corroborated
    uncorro_cells = closure_cells & ~corroborated
    mw_uncorro_cells = closure_cells & (mw_rows & ~mw_ok)[:, None]
    # Market-wide mass lacking an in-sample anniversary window is untestable.
    mw_unverif_cells = closure_cells & (mw_rows & mw_untest)[:, None]
    counted = viol_arr & ~closure
    n_closure = int(closure_cells.sum())
    n_corro = int(corro_cells.sum())
    n_uncorro = int(uncorro_cells.sum())
    n_mw_uncorro = int(mw_uncorro_cells.sum())
    n_mw_unverif = int(mw_unverif_cells.sum())
    n_violations = int(counted.sum())
    n_held = int(held_arr.sum())
    # Weight missing cells by absolute exposure, so a concentrated holding's
    # missing return retains its share of the book's risk.
    pos_gross = np.abs(np.nan_to_num(pos.to_numpy(dtype=float)))
    gross_held = float(pos_gross[held_arr].sum())
    frac = (float(pos_gross[counted].sum()) / gross_held
            if gross_held > 0 else 0.0)
    frac_uncorro = (float(pos_gross[uncorro_cells].sum()) / gross_held
                    if gross_held > 0 else 0.0)
    frac_unverif = (float(pos_gross[mw_unverif_cells].sum()) / gross_held
                    if gross_held > 0 else 0.0)
    # The per-bar gate covers counted gaps and uncorroborated closures.
    # It prevents a concentrated missing date from being diluted by history.
    bar_cells = counted | uncorro_cells
    bar_gross_missing = (pos_gross * bar_cells).sum(axis=1)
    bar_gross_held = (pos_gross * held_arr).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bar_frac = np.where(bar_gross_held > 0,
                            bar_gross_missing / bar_gross_held, 0.0)
    max_bar_frac = float(bar_frac.max()) if bar_frac.size else 0.0
    t_max_bar = int(np.argmax(bar_frac)) if bar_frac.size else 0
    # Measure cumulative exposure on each terminal run's first NaN bar.
    # Its delisting outcome is absent from the panel and remains unverified.
    # Names first bought later within the NaN run are zombie holdings, not
    # positions held through a delisting, and are counted separately.
    n_terminal = 0
    delisting_gross = 0.0
    for asset_col, first_missing_bar in terminal_start.items():
        if held_arr[first_missing_bar, asset_col]:
            n_terminal += 1
            delisting_gross += float(pos_gross[first_missing_bar, asset_col])
    # non-recurring testable market-wide dates carry their own absolute
    # per-year budget (see MARKETWIDE_NONRECURRING_MAX_PER_YEAR)
    n_mw_nonrecurring = n_mw_dates - n_mw_recurring - n_mw_untestable
    years = len(viol_all.index) / float(artifacts.periods_per_year)
    mw_budget = MARKETWIDE_NONRECURRING_MAX_PER_YEAR * years
    corro_details: dict[str, Any] = dict(
        n_terminal_delistings=n_terminal,
        delisting_bar_gross_exposure=float(delisting_gross),
        terminal_delisting_max_gross_exposure=(
            TERMINAL_DELISTING_MAX_GROSS_EXPOSURE),
        terminal_delisting_exposure_unverified=bool(
            n_terminal > 0 and delisting_gross > 0),
        terminal_delisting_exposure_material=bool(
            delisting_gross >= TERMINAL_DELISTING_MAX_GROSS_EXPOSURE),
        n_market_closure_cells=n_closure,
        n_closure_corroborated_cells=n_corro,
        n_closure_uncorroborated_cells=n_uncorro,
        frac_uncorroborated_closure=float(frac_uncorro),
        closure_min_shared_dates=CLOSURE_MIN_SHARED_DATES,
        n_marketwide_closure_dates=n_mw_dates,
        n_marketwide_recurring_dates=n_mw_recurring,
        n_marketwide_untestable_dates=n_mw_untestable,
        n_marketwide_uncorroborated_cells=n_mw_uncorro,
        n_marketwide_unverifiable_cells=n_mw_unverif,
        frac_unverifiable_marketwide=float(frac_unverif),
        n_marketwide_nonrecurring_dates=int(n_mw_nonrecurring),
        marketwide_nonrecurring_budget=float(mw_budget),
        max_bar_gross_missing_frac=float(max_bar_frac),
        max_bar_gross_missing_tol=MISSING_RETURN_MAX_BAR_GROSS,
        marketwide_min_closed_frac=MARKETWIDE_MIN_CLOSED_FRAC,
        nonheld_coclosure_expected=float(nonheld_exp),
        nonheld_coclosure_observed=int(nonheld_obs),
        exog_bridging_z=float(exog.z),
        exog_destroyed_sr=float(exog.destroyed_sr),
        exog_n_runs_tested=int(exog.n_runs),
        exog_testable=bool(exog.testable),
        n_guard_stripped_cells=n_stripped,
        corroboration_guards_fired=list(guards_fired),
    )
    mw_note = (" or a recurring market-wide closure calendar"
               if n_mw_recurring or n_mw_untestable else "")
    closure_note = (
        f" ({n_corro} further held cells sit in interior NaN runs <= "
        f"{MARKET_CLOSURE_MAX_RUN_BARS} bars with finite returns on both "
        f"sides, cross-sectionally corroborated by calendar siblings"
        f"{mw_note} - market-holiday/halt closures on a union calendar, "
        f"whose PnL is realized in the next multi-day return; excluded "
        f"from the tolerance)" if n_corro else "")

    if frac > MISSING_RETURN_FRAC:
        counted_df = pd.DataFrame(counted, index=viol_all.index,
                                  columns=viol_all.columns)
        rows = counted_df.any(axis=1).to_numpy()
        t_first = int(np.argmax(rows))
        first_date = str(counted_df.index[t_first].date())
        first_asset = str(counted_df.columns[
            int(np.argmax(counted_df.iloc[t_first].to_numpy()))])
        return warned(
            check,
            f"{n_violations} of {n_held} nonzero position cells "
            f"({frac:.2%} of gross exposure) sit on NaN asset returns in "
            f"terminal or > "
            f"{MARKET_CLOSURE_MAX_RUN_BARS}-bar gaps (first: {first_asset} "
            f"on {first_date}); their PnL silently becomes 0 - a delisting "
            f"loss you never paid (tolerance "
            f"{MISSING_RETURN_FRAC:.1%}){closure_note}.",
            severity=Severity.HIGH,
            remediation=(
                "Force positions to zero (or model an explicit delisting "
                "return) wherever the asset return is missing; audit the "
                "return panel for holes that positions silently ride through."
            ),
            n_violations=n_violations, n_nonzero_cells=n_held,
            frac_missing_return=float(frac),
            closure_max_run_bars=MARKET_CLOSURE_MAX_RUN_BARS,
            first_date=first_date, first_asset=first_asset,
            **corro_details,
        )
    if frac_uncorro > MISSING_RETURN_FRAC:
        # Scoped verdict, deliberately not a delisting accusation: the
        # statistic cannot separate selective loss-day deletion from honest
        # asset-idiosyncratic closures (a single-name-per-market book has no
        # calendar sibling by construction) or from honest ad-hoc
        # market-wide closures, so it measures and reports the ambiguous
        # component instead of PASSing over it or convicting it. The
        # message names the layer that refused corroboration.
        uncorro_df = pd.DataFrame(uncorro_cells, index=viol_all.index,
                                  columns=viol_all.columns)
        rows = uncorro_df.any(axis=1).to_numpy()
        t_first = int(np.argmax(rows))
        first_date = str(uncorro_df.index[t_first].date())
        first_asset = str(uncorro_df.columns[
            int(np.argmax(uncorro_df.iloc[t_first].to_numpy()))])
        u_strip = n_stripped
        u_mw = n_mw_uncorro
        u_idio = n_uncorro - u_strip - u_mw
        parts: list[str] = []
        if u_idio:
            parts.append(f"{u_idio} on dates no other asset shares as a "
                         f"recurring co-closure")
        if u_mw:
            parts.append(
                f"{u_mw} on market-wide closure dates (>= "
                f"{MARKETWIDE_MIN_CLOSED_FRAC:.0%} of live names closed at "
                f"once) with no year-over-year recurrence - "
                f"{n_mw_dates - n_mw_recurring - n_mw_untestable} of "
                f"{n_mw_dates} such dates have no anniversary partner "
                f"within +/-{MARKETWIDE_RECUR_TOL_DAYS} days in another "
                f"sampled year")
        if u_strip:
            parts.append(
                f"{u_strip} whose calendar-sibling corroboration was "
                f"rejected by the {', '.join(guards_fired)} guard(s)")
        if u_mw >= max(u_idio, u_strip):
            mech = (
                "A real market-wide holiday recurs on (nearly) the same "
                "calendar date every year; non-recurring market-wide gaps "
                "at this scale are indistinguishable from deletion of the "
                "book's worst shared dates, whose PnL silently becomes 0. "
                "This may be honest (ad-hoc exchange closures happen) - "
                "the audit cannot tell from these artifacts alone")
            remediation = (
                "Corroborate the market-wide gap dates against the "
                "exchange holiday calendar; if they are not exchange "
                "holidays, restore the missing rows (or zero positions "
                "across them) - recurring real holidays pass automatically."
            )
        elif u_strip > u_idio:
            why: list[str] = []
            if "nonheld_coclosure_deficit" in guards_fired:
                why.append(
                    f"a real holiday closes held and non-held names alike, "
                    f"but these closures never touch a non-held name "
                    f"({nonheld_obs} non-held co-closures vs "
                    f"~{nonheld_exp:.0f} expected under a book-independent "
                    f"calendar)")
            if "bridging_exogeneity" in guards_fired:
                why.append(
                    f"the peer-implied PnL over the closed bars is never "
                    f"recovered in the post-gap bridging returns (bridging "
                    f"z={exog.z:.1f}, ~{exog.destroyed_sr:.2f} annualized "
                    f"Sharpe units of held-book losses vanish across "
                    f"{exog.n_runs} closure runs; an honest halt conserves "
                    f"cumulative PnL across the gap)")
            mech = (
                "Same-book sibling co-closure is not independent evidence "
                "here: " + "; ".join(why) + ". This is what selective "
                "deletion of shared loss days looks like - the audit "
                "cannot tell it from a broken data join from these "
                "artifacts alone")
            remediation = (
                "Corroborate the gaps against the exchange calendar - "
                "same-book co-closure is not independent evidence; if the "
                "gaps are vendor holes, repair the return panel or zero "
                "positions on them."
            )
        else:
            mech = (
                "A real market holiday closes several names on the same "
                "recurring calendar; asset-idiosyncratic gaps at this "
                "scale are indistinguishable from selective deletion of "
                "loss days, whose PnL silently becomes 0. This may be "
                "honest (one name per market/exchange in the book) - the "
                "audit cannot tell from these artifacts alone")
            remediation = (
                "Corroborate the gaps against the exchange calendar (or "
                "add a sibling name per market so closures are "
                "cross-sectionally confirmed); if the gaps are data holes, "
                "zero positions on them or repair the return panel."
            )
        return warned(
            check,
            f"{n_uncorro} of {n_held} nonzero position cells "
            f"({frac_uncorro:.2%} of gross exposure > tolerance "
            f"{MISSING_RETURN_FRAC:.1%}) sit "
            f"in interior NaN-return runs <= {MARKET_CLOSURE_MAX_RUN_BARS} "
            f"bars without independent closure corroboration: "
            f"{'; '.join(parts)} (first: {first_asset} on {first_date}). "
            f"{mech}{closure_note}.",
            severity=Severity.HIGH,
            remediation=remediation,
            n_violations=n_violations, n_nonzero_cells=n_held,
            frac_missing_return=float(frac),
            closure_max_run_bars=MARKET_CLOSURE_MAX_RUN_BARS,
            first_date=first_date, first_asset=first_asset,
            **corro_details,
        )
    if n_mw_nonrecurring > mw_budget:
        # The annual date-count gate limits dilution in long samples. Excess
        # nonrecurring market-wide closures remain ambiguous from these artifacts.
        return warned(
            check,
            f"{n_mw_nonrecurring} market-wide closure dates (>= "
            f"{MARKETWIDE_MIN_CLOSED_FRAC:.0%} of live names NaN at once) "
            f"recur in NO other sampled year, vs a plausibility budget of "
            f"{mw_budget:.1f} over {years:.1f} years "
            f"({MARKETWIDE_NONRECURRING_MAX_PER_YEAR:.0f}/year - honest "
            f"ad-hoc exchange closures are ~1/decade). The missing mass is "
            f"only {frac_uncorro:.2%} of gross exposure (tolerance "
            f"{MISSING_RETURN_FRAC:.1%}), but non-recurring market-wide "
            f"gaps at this DATE count are indistinguishable from deletion "
            f"of the book's worst shared days diluted over a long sample. "
            f"This may be honest - the audit cannot tell from these "
            f"artifacts alone{closure_note}.",
            severity=Severity.HIGH,
            remediation=(
                "Corroborate the market-wide gap dates against the exchange "
                "holiday calendar; if they are not exchange closures, "
                "restore the missing rows (or zero positions across them) - "
                "recurring real holidays pass automatically."
            ),
            n_violations=n_violations, n_nonzero_cells=n_held,
            frac_missing_return=float(frac),
            closure_max_run_bars=MARKET_CLOSURE_MAX_RUN_BARS,
            **corro_details,
        )
    if frac_unverif > MISSING_RETURN_FRAC:
        # Missing anniversary coverage prevents testing these market-wide dates.
        # Material untestable exposure needs a scoped warning rather than a claim
        # that its closure accounting was verified.
        return warned(
            check,
            f"{n_mw_unverif} of {n_held} nonzero position cells "
            f"({frac_unverif:.2%} of gross exposure > tolerance "
            f"{MISSING_RETURN_FRAC:.1%}) sit on {n_mw_untestable} "
            f"market-wide closure dates whose year-over-year recurrence "
            f"CANNOT be tested: the sample spans too little calendar time "
            f"for any anniversary window (+/-"
            f"{MARKETWIDE_RECUR_TOL_DAYS} days, "
            f"{MARKETWIDE_RECUR_TESTABLE_MARGIN_DAYS}-day margin). These "
            f"look like exchange holidays and are excluded from the "
            f"tolerance, but the audit cannot verify them from these "
            f"artifacts - on a short sample, deleted worst-day rows are "
            f"indistinguishable from a holiday calendar{closure_note}.",
            severity=Severity.HIGH,
            remediation=(
                "Corroborate the market-wide gap dates against the exchange "
                "holiday calendar (or extend the sample past ~2 years so "
                "recurrence becomes testable); real holidays then pass "
                "automatically."
            ),
            n_violations=n_violations, n_nonzero_cells=n_held,
            frac_missing_return=float(frac),
            closure_max_run_bars=MARKET_CLOSURE_MAX_RUN_BARS,
            **corro_details,
        )
    if max_bar_frac > MISSING_RETURN_MAX_BAR_GROSS:
        # A material fraction of one bar's book can be missing even when its
        # aggregate exposure fraction remains under tolerance.
        bad_date = str(viol_all.index[t_max_bar].date()) \
            if isinstance(viol_all.index, pd.DatetimeIndex) \
            else str(viol_all.index[t_max_bar])
        return warned(
            check,
            f"on {bad_date}, {max_bar_frac:.0%} of that bar's gross book "
            f"exposure sits on unexcluded NaN asset returns (per-bar "
            f"tolerance {MISSING_RETURN_MAX_BAR_GROSS:.0%}; aggregate "
            f"{frac:.2%} of gross is under the {MISSING_RETURN_FRAC:.1%} "
            f"tolerance): that bar's PnL on the missing name(s) silently "
            f"became 0 - on a concentrated holding this is a whole-book "
            f"loss (or gain) that never printed{closure_note}.",
            severity=Severity.HIGH,
            remediation=(
                "Force positions to zero (or model an explicit delisting/"
                "halt return) wherever the asset return is missing; a "
                "missing bar on a concentrated holding cannot hide in a "
                "cell-count or aggregate-exposure tolerance."
            ),
            n_violations=n_violations, n_nonzero_cells=n_held,
            frac_missing_return=float(frac),
            closure_max_run_bars=MARKET_CLOSURE_MAX_RUN_BARS,
            **corro_details,
        )
    # The disclosure is in NAV units (sum of |weight| over the delisting
    # bars, i.e. the mass whose delisting return is unverified), not a
    # fraction of the whole-history gross like `frac` above - say so.
    delisting_note = (
        f"; {n_terminal} name(s) delist while held - their delisting-bar "
        f"weights sum to {delisting_gross:.1%} of NAV "
        f"({delisting_gross / n_terminal:.1%} per name on average); the "
        f"delisting return itself is not verified here"
        if delisting_gross > 0 and n_terminal > 0 else "")
    if delisting_gross >= TERMINAL_DELISTING_MAX_GROSS_EXPOSURE:
        return warned(
            check,
            f"{n_terminal} name(s) delist while held; their delisting-bar "
            f"weights sum to {delisting_gross:.1%} of NAV "
            f"({delisting_gross / n_terminal:.1%} per name on average); this "
            f"meets the "
            f"{TERMINAL_DELISTING_MAX_GROSS_EXPOSURE:.0%} material-exposure "
            f"bar, but each delisting return itself is not verified because "
            f"it is absent. The whole-history missing-return ratio "
            f"is only {frac:.4%}; sample-length dilution cannot certify this "
            f"material terminal leg{closure_note}.",
            severity=Severity.HIGH,
            remediation=(
                "Record each terminal outcome explicitly in asset_returns "
                "(bankruptcy, cash acquisition/tender, OTC transfer or other "
                "delisting return), or prove the position was flat before "
                "the first missing-return bar; do not replace a material "
                "terminal PnL leg with NaN/zero."
            ),
            n_violations=n_violations, n_nonzero_cells=n_held,
            frac_missing_return=float(frac),
            closure_max_run_bars=MARKET_CLOSURE_MAX_RUN_BARS,
            **corro_details,
        )
    return passed(
        check,
        f"{n_violations} of {n_held} nonzero position cells sit on NaN asset "
        f"returns in terminal or > {MARKET_CLOSURE_MAX_RUN_BARS}-bar gaps "
        f"({frac:.4%} of gross exposure <= tolerance "
        f"{MISSING_RETURN_FRAC:.1%}) - consistent "
        f"with at most one final delisting bar per dying asset"
        f"{closure_note}{delisting_note}.",
        severity=Severity.HIGH,
        n_violations=n_violations, n_nonzero_cells=n_held,
        frac_missing_return=float(frac),
        closure_max_run_bars=MARKET_CLOSURE_MAX_RUN_BARS,
        **corro_details,
    )


# ---------------------------------------------------------------------------
# 3. no_exits (MEDIUM, WARN)
# ---------------------------------------------------------------------------

def _no_exits(artifacts: BacktestArtifacts, config: AuditConfig) -> CheckResult:
    check = "survivorship.no_exits"
    uni = artifacts.universe
    if uni is None:
        return skipped(
            check,
            "pass artifacts.universe (dates x assets bool, point-in-time "
            "tradeable flags) to enable this check.",
            missing=["universe"],
        )
    n_periods, n_grid_assets = uni.shape
    uni_arr = uni.to_numpy(dtype=bool)
    ppy = float(artifacts.periods_per_year)
    member_years = float(uni_arr.sum()) / ppy
    # All-False padding columns have no member-bar and contribute no
    # evidence about attrition.  Counting them toward the breadth floor lets
    # a one-name universe masquerade as a broad panel and even PASS after one
    # token exit.
    n_assets = int(uni_arr.any(axis=0).sum())
    if n_assets == 0:
        return skipped(
            check,
            f"the universe has no live member-bars ({n_periods} periods x "
            f"{n_grid_assets} grid columns, all False) - there is no "
            f"membership to exit from, so an attrition rate is undefined.",
            n_periods=n_periods, n_assets=0,
            n_grid_assets=n_grid_assets,
            n_never_live_assets=n_grid_assets,
            member_years=0.0,
        )
    if n_assets < config.min_assets_survivorship:
        return skipped(
            check,
            f"only {n_assets} ever-live assets (< "
            f"config.min_assets_survivorship="
            f"{config.min_assets_survivorship}; {n_grid_assets} columns on "
            f"the grid); all-False placeholder columns provide no "
            f"attrition evidence.",
            n_assets=n_assets, n_grid_assets=n_grid_assets,
            n_never_live_assets=n_grid_assets - n_assets,
            n_periods=n_periods, member_years=member_years,
        )
    min_periods = max(int(np.ceil(MIN_YEARS_NO_EXITS * ppy)),
                      MIN_BARS_NO_EXITS)
    if n_periods < min_periods:
        return skipped(
            check,
            f"only {n_periods} periods (< floor {min_periods} bars ~ "
            f"{min_periods / ppy:.1f} years at periods_per_year="
            f"{artifacts.periods_per_year}); sample too short to expect "
            f"universe exits.",
            n_periods=n_periods, min_periods_required=min_periods,
        )

    exits = uni_arr[:-1] & ~uni_arr[1:]          # True -> False transition
    n_exiting_raw = int(exits.any(axis=0).sum())
    n_exit_events = int(exits.sum())
    # Flicker-plausibility prong (see NO_EXITS_FLICKER_MAX_OUT_DAILY_BARS):
    # only exits that are terminal or stay out longer than the window count
    # toward attrition; quick out-and-back transitions are flicker - the
    # signature of scattered NaN membership cells, not delistings.
    flicker_window = int(np.floor(
        NO_EXITS_FLICKER_MAX_OUT_DAILY_BARS * ppy / DAILY_BARS_PER_YEAR))
    n_flicker_events = 0
    n_unverifiable_events = 0
    plausible = np.zeros(n_grid_assets, dtype=bool)
    if flicker_window < 1:
        plausible = exits.any(axis=0)            # coarse bars: prong idle
    else:
        for j in range(n_grid_assets):
            exit_rows = np.flatnonzero(exits[:, j])
            if exit_rows.size == 0:
                continue
            true_rows = np.flatnonzero(uni_arr[:, j])
            for t in exit_rows:
                nxt = int(np.searchsorted(true_rows, t + 1))
                if nxt >= len(true_rows):
                    # terminal: never re-enters - but only plausible when
                    # the sample actually observes more than a flicker
                    # window past the exit. A fabricated exit on the last
                    # bar(s) (one NaN or all-False row at the sample end)
                    # would otherwise mint an unfalsifiable "terminal"
                    # delisting for every name at once.
                    if int(n_periods) - 1 - int(t) > flicker_window:
                        plausible[j] = True
                    else:
                        n_unverifiable_events += 1
                elif int(true_rows[nxt]) - int(t) - 1 > flicker_window:
                    plausible[j] = True          # stayed out long enough
                else:
                    n_flicker_events += 1
    n_exiting = int(plausible.sum())
    years = n_periods / float(artifacts.periods_per_year)
    # Measure exits per member-year of actual universe exposure, so a panel
    # with late listings is not treated as fully populated throughout.
    # Count distinct plausibly exiting assets: repeated exits by one name
    # contribute once, and a token exit does not establish broad attrition.
    if member_years <= 0.0:
        return skipped(
            check,
            f"the universe has no live member-bars ({n_periods} periods x "
            f"{n_assets} names, all False) - there is no membership to "
            f"exit from, so an attrition rate is undefined.",
            n_periods=n_periods, n_assets=n_assets, member_years=0.0,
        )
    exit_rate = n_exiting / member_years
    raw_rate = n_exiting_raw / member_years
    # Retain the per-column-year diagnostic for comparison with membership exposure.
    exit_rate_per_asset_year = (n_exiting / (n_assets * years)
                                if years > 0 else 0.0)
    details = dict(n_exiting_assets=n_exiting, n_assets=n_assets,
                   n_grid_assets=n_grid_assets,
                   n_never_live_assets=n_grid_assets - n_assets,
                   n_periods=n_periods, years=float(years),
                   member_years=float(member_years),
                   exit_rate_per_year=float(exit_rate),
                   exit_rate_per_asset_year=float(exit_rate_per_asset_year),
                   min_exit_rate=config.min_exit_rate_per_year,
                   n_exiting_assets_raw=n_exiting_raw,
                   n_exit_events=n_exit_events,
                   n_flicker_exit_events=n_flicker_events,
                   n_unverifiable_exit_events=n_unverifiable_events,
                   flicker_max_out_bars=flicker_window)
    remediation = (
        "Rebuild the universe from point-in-time constituent history "
        "(including delisted/acquired names) instead of a static "
        "present-day membership list."
    )
    plausible_ok = (n_exiting > 0
                    and exit_rate >= config.min_exit_rate_per_year)
    raw_ok = (n_exiting_raw > 0
              and raw_rate >= config.min_exit_rate_per_year)
    n_implausible = n_flicker_events + n_unverifiable_events
    edge_txt = (f" and {n_unverifiable_events} exit within "
                f"{flicker_window} bar(s) of the sample end (too close to "
                f"the edge to verify)" if n_unverifiable_events else "")
    if n_implausible and raw_ok and not plausible_ok:
        # Raw transitions suggest sufficient attrition, but plausible exits do
        # not. Brief membership gaps and sample-edge exits cannot establish a
        # healthy exit rate, so this discrepancy receives the higher warning.
        return warned(
            check,
            f"universe membership flickers rather than attrites: "
            f"{n_flicker_events} of {n_exit_events} exit event(s) re-enter "
            f"within {flicker_window} bar(s){edge_txt} (a real delisting "
            f"exit is terminal, or stays out at least through the next "
            f"reconstitution), leaving {n_exiting} plausible exiting "
            f"asset(s) ({exit_rate:.2%}/year of live membership over "
            f"{member_years:.1f} member-years vs the "
            f"{config.min_exit_rate_per_year:.0%}/year floor) where the "
            f"raw transition count suggested {n_exiting_raw} "
            f"({raw_rate:.1%}/year) - the signature of scattered universe "
            f"NaN cells decoding to False (fabricated membership), not "
            f"point-in-time attrition.",
            severity=Severity.HIGH,
            remediation=(
                "Resolve NaN membership explicitly (universe.ffill(), or "
                "fillna(False) if the names were genuinely out) and "
                "rebuild the universe from point-in-time constituent "
                "history - quick out-and-back membership blips are data "
                "holes, not delistings."),
            details=details)
    flick_note = (
        f" ({n_flicker_events} of {n_exit_events} exit event(s) re-enter "
        f"within {flicker_window} bar(s){edge_txt} - flicker, not "
        f"attrition - and are excluded from the rate)"
        if n_implausible else "")
    if n_exiting == 0:
        msg = (f"no asset ever plausibly leaves the universe over "
               f"{years:.1f} years x {n_assets} names{flick_note} - real "
               f"point-in-time universes lose names to "
               f"delistings/acquisitions."
               if n_implausible else
               f"no asset ever leaves the universe over {years:.1f} years x "
               f"{n_assets} names - real point-in-time universes lose names "
               f"to delistings/acquisitions.")
        return warned(
            check, msg,
            severity=Severity.MEDIUM, remediation=remediation, details=details)
    if exit_rate < config.min_exit_rate_per_year:
        return warned(
            check,
            f"only {n_exiting} exit(s) across {n_assets} names over "
            f"{member_years:.1f} member-years ({years:.1f} calendar years; "
            f"{exit_rate:.2%}/year of live membership vs the "
            f"{config.min_exit_rate_per_year:.0%}/year plausibility floor; "
            f"real universes lose ~4-8%/year) - membership looks like a "
            f"survivor list with token attrition.{flick_note}",
            severity=Severity.MEDIUM, remediation=remediation, details=details)
    return passed(
        check,
        f"{n_exiting} of {n_assets} assets exit the universe over "
        f"{member_years:.1f} member-years ({exit_rate:.1%}/year of live "
        f"membership, {years:.1f} calendar years) - membership looks "
        f"point-in-time.{flick_note}",
        severity=Severity.MEDIUM, details=details)


# ---------------------------------------------------------------------------
# 4. full_history_universe (HIGH, WARN) - only when universe is None
# ---------------------------------------------------------------------------

def _full_history_universe(artifacts: BacktestArtifacts,
                           config: AuditConfig) -> CheckResult:
    check = "survivorship.full_history_universe"
    if artifacts.universe is not None:
        return skipped(
            check,
            "universe artifact provided - this inference check is superseded "
            "by survivorship.no_exits.",
            superseded=True,   # report.gate() strict mode ignores this SKIP
        )

    rets = artifacts.asset_returns
    n_periods, n_grid_assets = rets.shape
    mask = rets.notna().to_numpy()
    any_valid = mask.any(axis=0)
    # A wholly unobserved column is neither a dead name nor a late listing:
    # it contains no lifecycle evidence at all.  Including it in the
    # denominator lets arbitrary NaN padding dilute an otherwise obvious
    # current-members-only panel below the warning fraction.
    n_assets = int(any_valid.sum())
    if n_assets < config.min_assets_survivorship:
        return skipped(
            check,
            f"only {n_assets} assets contain any observed return (< "
            f"config.min_assets_survivorship="
            f"{config.min_assets_survivorship}; {n_grid_assets} columns on "
            f"the grid); all-NaN placeholder columns provide no history-"
            f"completeness evidence.",
            n_assets=n_assets, n_grid_assets=n_grid_assets,
            n_unobserved_assets=n_grid_assets - n_assets,
        )

    first = mask.argmax(axis=0)
    last = n_periods - 1 - mask[::-1].argmax(axis=0)
    span = (last - first + 1).astype(float)
    coverage = np.where(any_valid, span / n_periods, 0.0)
    frac_full = float((coverage[any_valid] >= FULL_SPAN_COVERAGE).mean())

    if frac_full >= config.full_history_frac_warn:
        return warned(
            check,
            f"{frac_full:.0%} of {n_assets} assets have complete histories "
            f"(first->last non-NaN span >= {FULL_SPAN_COVERAGE:.0%} of "
            f"{n_periods} periods) and no universe was provided - the panel "
            f"looks like today's members projected into the past.",
            severity=Severity.HIGH,
            remediation=(
                "Backtest with point-in-time constituents including dead "
                "names, and pass the resulting universe DataFrame in "
                "artifacts.universe."
            ),
            frac_full_history=frac_full, n_assets=n_assets,
            n_grid_assets=n_grid_assets,
            n_unobserved_assets=n_grid_assets - n_assets,
        )
    return passed(
        check,
        f"only {frac_full:.0%} of {n_assets} assets span >= "
        f"{FULL_SPAN_COVERAGE:.0%} of the sample (threshold "
        f"{config.full_history_frac_warn:.0%}) - dead/late-listed names appear "
        f"present even without a universe artifact.",
        severity=Severity.HIGH,
        frac_full_history=frac_full, n_assets=n_assets,
        n_grid_assets=n_grid_assets,
        n_unobserved_assets=n_grid_assets - n_assets,
    )
