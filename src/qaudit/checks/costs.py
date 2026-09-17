"""Transaction-cost realism checks.

Five static checks (no pipeline re-execution needed):

- ``costs.missing_transaction_costs`` - reconstructs cost-free (gross)
  strategy returns from positions x asset_returns and compares them with the
  declared net ``strategy_returns``. The implied one-way cost per unit traded
  must clear an economic plausibility floor (bit-identity alone is evadable
  with sub-bp jitter), the net-vs-gross overlap must actually contain the
  dates that trade (else implied cost is 0/0 and the comparison is vacuous),
  and the implied cost is cross-checked against
  ``artifacts.declared_costs_bps``; net sitting above gross (negative implied
  cost) is flagged as mismatched artifacts - costs cannot be negative.
  ``declared_costs_bps=0`` is the paper-trading path: net matching
  gross then PASSes here (the declaration says exactly that) while
  ``costs.no_cost_declaration`` carries the HIGH warn about the assumption.
  Honest long-only cash-buffer books whose net includes the cash sleeve's
  yield are recognized before any accusation: when per-date drag
  regresses onto dollars traded with a slope consistent with the declared
  cost and a small constant positive accrual bounded by a plausible yield
  on the uninvested fraction, the verdict is a scoped WARN naming the
  cash-carry reading (declare CASH as an asset column to PASS), not the
  under-charged / mismatched-artifacts conviction. A clean measurement is
  certified only from >= ``COST_RECONCILE_MIN_DATES`` compared dates, and
  as many inside the positions' live span (fewer is a SKIP - definitive
  evidence that costs were absent still FAILs on any overlap) and the PASS
  discloses what share of the book's traded dollars it compared, including
  the liquidation into the first flat row after the live span. A
  clean verdict requires essentially all traded dollars; below
  ``MIN_OVERLAP_TRADED_SHARE`` the check SKIPs so a deployment gate cannot
  mistake a selected-date comparison for verification.
- ``costs.no_cost_declaration`` - is there a cost assumption on record at all?
- ``costs.turnover_unrealistic`` - is the turnover level executable at
  retail/IBKR scale?
- ``costs.cost_sensitivity`` - does the edge survive plausible one-way costs?
- ``costs.price_return_consistency`` - do ``prices`` and ``asset_returns``
  describe the same series? Close-to-close returns derived from the price
  panel are compared cell-wise against asset_returns; a systematic mismatch
  is the adjusted/unadjusted mix-up class that silently mis-sizes every
  downstream dollar figure (share counts, commissions, dollars traded).

Cost math convention (drift-aware): dollars traded during period t
(two-sided, per unit of book) =
``sum_i |w[t]_i - w_drift[t]_i|`` where ``w_drift[t] = w[t-1]*(1+r[t-1]) /
(1 + sum_i w[t-1]_i*r[t-1]_i)`` are the pre-trade drifted weights, with the
first row set to ``positions.iloc[0].abs().sum()`` (the entry trade) - see
:func:`qaudit._stats.traded_dollars_series`. Raw weight diffs are not the
trade ledger: a constant-mix book re-buys against drift every bar (raw
diffs would price that at zero and let a net==gross constant-mix book
through every cost check), while a true buy-and-hold book whose weights
only drift trades nothing after entry.
Two-sided dollars equal 2x the one-sided turnover of
:func:`qaudit._stats.turnover_series`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._stats import (annualized_sharpe, gross_strategy_returns,
                      traded_dollars_series, turnover_series)
from ..config import AuditConfig
from ..inputs import BacktestArtifacts
from ..types import (CheckResult, Severity, Status, failed, passed, skipped,
                     warned)

# Module constants (fixed, not tunable via AuditConfig):
# the cost-sensitivity warning only fires when the gross edge was worth
# having in the first place (annualized gross Sharpe above this floor)...
MIN_GROSS_SR_FOR_SENSITIVITY = 0.5
# ...and implied vs declared costs are called discrepant when they differ by
# more than this factor either way.
DECLARED_IMPLIED_MISMATCH_FACTOR = 3.0
# Net sitting above gross (negative implied cost) beyond this many bps means
# the artifacts do not describe the same backtest - costs cannot be negative.
# The tolerance absorbs reconstruction noise from NaN masking on honest books.
NEGATIVE_IMPLIED_BPS_TOL = 0.5
# Economic diagnostic floor for the daily-equity use case, separate from
# numerical tolerance. A lower assumed rate must be declared explicitly:
# zero selects paper trading, and a positive sub-floor declaration receives
# a scoped warning when it reconciles with the charged cost. This threshold
# is a heuristic, not a universal lower bound on execution costs.
MIN_PLAUSIBLE_ONE_WAY_BPS = 0.5

# Implied one-way cost above 500bps (5% of traded notional) receives a
# mismatch warning: even illiquid small-caps in a crisis run ~50-100bps
# one-way, so the bound leaves ~5x headroom above any real equity
# execution. Check units, rebalance timing and whether net returns
# describe the same positions and asset returns. The bound is a
# plausibility rule for this model, not a claim about every market or
# instrument.
MAX_PLAUSIBLE_ONE_WAY_BPS = 500.0
# config.turnover_daily_warn/_fail are calibrated for daily bars (the field
# names say so): 25%/day warn and 100%/day fail. Executability is a
# per-year property of the book - 25%/period on monthly bars is 3x/year,
# ordinary smart-beta territory that any retail account executes - so the
# verdict compares annualized turnover (mean/period x periods_per_year)
# against the annualized bars (warn 25% x 252 = 63x/yr, fail 100% x 252 =
# 252x/yr one-sided). Daily books see bit-identical thresholds.
TURNOVER_CALIBRATION_PPY = 252.0
# Net returns may include yield on idle cash, which the positions-times-
# returns reconstruction omits. Regress gross-minus-net on traded dollars:
# slope estimates the one-way cost, and a negative intercept implies carry.
# A scoped cash-carry warning requires a declaration-consistent/plausible
# slope, positive carry bounded by the uninvested gross-exposure fraction,
# and small residual variation. This cannot waive an uncharged-cost slope
# or unexplained PnL for a fully invested book.
MAX_PLAUSIBLE_CASH_YIELD_ANNUAL = 0.15  # ceiling on idle-cash yield: covers
#   every modern high-rate regime (T-bills peaked ~5.5% in 2023-24; EM cash
#   rates run higher) with headroom - a PnL stream above 15%/yr on the idle
#   fraction is not cash carry and keeps the mismatched-artifacts verdict.
CASH_CARRY_MAX_RESID_FRAC = 0.25        # residual std / fitted per-bar carry:
#   an honest constant accrual leaves near-zero relative residuals, while a
#   timing/scale mismatch that happens to fit a negative intercept leaves
#   date-level structure the line cannot absorb - 0.25 is economics, not
#   numerics.
CASH_CARRY_MIN_DATES = 60              # minimum dates for the two-parameter fit

# The net-vs-gross overlap must carry a representative share of the book's
# trading: mean dollars traded on the compared (non-NaN) dates below this
# fraction of the all-dates mean means strategy_returns are reported only on
# no-trade dates - implied cost degenerates to 0/0 (NaN, which passes every
# float comparison) and the whole comparison verifies nothing.
MIN_OVERLAP_TRADING_FRAC = 0.10

# The "negligible turnover" shortcut (costs immaterial, nothing to verify)
# needs more than a small mean: the mean over a long live span stays tiny
# while a burst replaces the whole book several times over (a one-bar 100%
# round trip - 2.0 dollars traded - would otherwise read as mean turnover
# 0.0 and certify all five cost checks green with net==gross). The entry
# trade enters the mean (see _mean_one_sided_turnover), and a second
# prong refuses the shortcut when any single post-entry bar turns
# more than this one-sided fraction of the book - such a bar is real
# executable trading whose charging must be measured, not waved through.
NEGLIGIBLE_MAX_ONE_SIDED_PER_BAR = 0.10

# Per-date cost reconciliation: the implied cost is mean(drag)/mean(dollars),
# which any cost stream with the right mean reproduces exactly - a flat
# per-bar deduction on a book that trades every 5th bar, or charges applied
# on shifted dates, lands implied_bps exactly on the declared figure and
# would PASS a mean-only test. Before the final PASS, the per-date residual
# drag_t - implied*dollars_t must stay within this fraction of the fitted
# per-trade drag scale (implied * rms dollars - mirrors performance.py's
# _DRAG_RECONCILE_REL_TOL = 0.25). Failing the prong is a WARN, not a
# FAIL: borrow/financing fees, monthly commission sweeps and non-linear
# impact are legitimate non-per-trade cost models the audit cannot
# distinguish from mis-charging - but they must be declared as such, not
# certified as per-trade charging. Books whose residual decomposes as
# declared-cost + constant cash carry keep the cash-carry scoped WARN.
# Honest per-trade charging measures rms residuals ~1e-15 of scale, so
# 0.25 is economics, not numerics. Fewer than COST_RECONCILE_MIN_DATES
# compared dates cannot resolve per-date structure (mirrors
# performance._DRAG_MIN_OBS) and cannot earn the PASS at all: a
# strategy_returns series reported on one date of a 500-date book clears
# every gate above (the mean-ratio coverage gate is a NaN-disarm, not a
# coverage floor) and would otherwise be certified "declared 10bps ...
# reconciles" off 0.2% of the traded dollars. Below the floor the
# accusatory paths still run on any n_common >= 1 (a 1-date bit-identical
# book still FAILs CRITICAL), but a clean measurement is a SKIP naming the
# missing dates, never a certificate. The floor binds twice: on n_common
# and on the compared dates inside the positions' live span - net reported
# as 0.0 on the flat padding of a short-lived book inflates n_common
# (0 dollars, 0 drag: it constrains nothing) without adding a single traded
# dollar to the certificate.
COST_RECONCILE_REL_TOL = 0.25
COST_RECONCILE_MIN_DATES = 30
# Coverage certificate: a PASS is meaningful only when essentially all
# dollars traded, including final liquidation, enter the net-vs-gross
# comparison.  A lower floor (25%, say) would let a reporter omit most of
# the trading and still satisfy
# ``gate(require=["costs.missing_transaction_costs"])`` with a PASS or LOW
# WARN.  Permit 1% operational/calendar dust, but make any material gap a
# SKIP; honest late-start books remain valid when positions and net returns
# begin together because their own live-span coverage is ~100%.
MIN_OVERLAP_TRADED_SHARE = 0.99

# costs.price_return_consistency tolerances. Close-only prices legitimately
# differ from total returns on dividend ex-dates (dividend-sized, sparse:
# ~4 cells per asset-year, <2% of daily cells), so the primary verdict rides
# on the median cell discrepancy, which sparse points cannot move - while a
# panel-wide systematic mismatch (shifted dates, percent-vs-decimal scaling,
# wrong asset mapping, a different vendor's series) moves the median by the
# scale of daily vol (~100bp). A mismatch confined to a minority of assets
# (mixed-vendor feeds) leaves the median clean, so a second, pervasiveness
# prong counts discrepant cells independently of the median (see
# PRICE_RETURN_PERVASIVE_FRAC below).
PRICE_RETURN_MEDIAN_TOL = 1e-4      # median |derived - declared| per day
PRICE_RETURN_POINT_TOL = 1e-3       # a cell counts as discrepant above 10bp
PRICE_RETURN_PERVASIVE_FRAC = 0.25  # discrepant-cell share that (a) lifts a
                                    # median-prong WARN to HIGH and (b) WARNs
                                    # on its own when the discrepant subset
                                    # lacks the dividend signature - a one-
                                    # bar shift on 9 of 20 assets puts 43% of
                                    # cells at daily-vol scale with median
                                    # 0.00bp, invisible to the median prong
# Tick quantization: real close files are quantized to the $0.01 tick,
# which puts rounding error in every cell of the derived return - per-cell
# |diff| is bounded by ~tick*(1/p_t + 1/p_{t-1})/2 and the median of pure
# cent rounding is 0.293*tick/p (~2.9bp at a $10 price, well above
# PRICE_RETURN_MEDIAN_TOL, so honest cent-quantized sub-$30 universes would
# otherwise warn). When the price panel sits on
# the cent grid and the diffs carry the rounding signature - two-sided
# (rounding is symmetric; dividends and date shifts are not) and cell-wise
# bounded by the per-cell rounding bound - the median tolerance is lifted
# to TICK_MEDIAN_TOL_MULT * median(tick/p_prev) and discrepant cells are
# counted only above their own rounding bound. Detection headroom is huge:
# a 1-day date shift medians at ~2x daily vol (~280bp, 20-100x tick noise
# even at $2 prices), percent-vs-decimal at ~99x, so the lift cannot
# swallow the defect classes this check hunts.
PRICE_TICK = 0.01                   # US equity close-file quantum
# Per-cell quantum inference: a panel-wide cent-grid gate would let one
# split-adjusted asset (rounded raw close / factor -> a tick/factor grid,
# the standard Yahoo-style Adj Close shape) disarm the waiver for the whole
# panel and cause the honest sub-$30 rounding false positive. Each stored
# price therefore gets its own effective quantum:
# the coarsest tick/f grid it sits on, f drawn from plausible cumulative
# split factors (2:1..20:1, 3:2, 5:2). Only f >= 1 (grids at or finer
# than the tick) are candidates - storage rounding coarser than the tick
# is a defect class this check hunts and must not become a waiver.
TICK_SPLIT_FACTORS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
                      10.0, 12.0, 16.0, 20.0)
TICK_GRID_ATOL = 1e-6               # |p/q - round(p/q)| below this counts as
                                    # on the q grid (in grid multiples; the
                                    # panel-wide cent-grid flag uses the same
                                    # tolerance)
TICK_MEDIAN_TOL_MULT = 0.5          # x median(tick/p_prev): pure rounding
                                    # sits at 0.293*tick/p, ~1.7x margin
TICK_CELL_BOUND_MULT = 1.2          # per-cell ceiling multiple: rounding can
                                    # never exceed its half-tick bound, so
                                    # q99 of |diff|/bound above this breaks
                                    # the signature (and a cell only counts
                                    # as discrepant above it on a quantized
                                    # panel)
TICK_TWO_SIDED_MAX_MEDIAN_RATIO = 0.5   # |median signed diff| must stay
                                    # below this fraction of median |diff|:
                                    # symmetric rounding medians near 0,
                                    # dividends/shifts are one-sided
SPARSE_DIVIDEND_FRAC = 0.02         # the docstring's sparse-dividend premise
                                    # (<2% of daily cells): above this the
                                    # PASS message must state the discrepant
                                    # fraction neutrally, not call it
                                    # "sparse"
# No dividend moves a close 15% (specials top out ~10%); forgotten split
# adjustments do (3:2 = -33%, 2:1 = -50%). Two or more split-sized cells is
# an adjusted/unadjusted mix-up even when the median stays clean - one lone
# cell is left to pass as a possible bad tick.
SPLIT_SIZED_DISCREPANCY = 0.15
SPLIT_SIZED_MIN_CELLS = 2
PRICE_CONSISTENCY_MIN_CELLS = 100   # pooled finite cells needed to judge
# Coverage accounting: the pooled compare dropna()s every cell without a
# price, so a panel with prices for 1 of 20 assets (5% of return cells)
# would otherwise issue the unscoped certificate "returns derived from
# artifacts.prices match asset_returns" while a x100 percent-vs-decimal
# defect sat on the other 19 - exactly the mis-sizing class this check
# names. The min-cells floor above is statistical power, not coverage.
# Every verdict reports what fraction of the finite
# asset_returns cells was actually compared (finite only, so dead-asset /
# late-listing NaN stretches - the docstring's ragged-panel design - do
# not read as uncovered); below the floor no PASS is issued: defect prongs
# still run on the covered subset (detection kept), and a clean subset
# earns a scoped MEDIUM WARN, not a certificate. The floor mirrors
# inputs._UNIVERSE_MIN_COVERAGE (0.60).
PRICE_COVERAGE_MIN_FRAC = 0.60
PRICE_COVERAGE_MIN_CELLS_PER_ASSET = 20  # an asset counts as compared with
#   at least this many jointly-observed cells - one stray joint cell is
#   not verification of that asset's series
# The sparse-dividend premise above inverts at coarse bar frequency:
# dividend mass per bar is ~yield / periods_per_year, so on monthly bars a
# 5%-yield monthly payer puts ~42bp in every cell and legitimately moves
# the median. Dividends have a signature the real defects lack: the offset
# is strictly one-sided (derived close-only return = total return - div/p,
# so derived - declared < 0 in every dividend cell) and bounded by a
# plausible per-bar yield - while percent-vs-decimal (~99x return
# magnitude), date shifts (two-sided at per-bar vol scale), and vendor
# mismatches are two-sided or far larger. Cap: 15%/yr covers the highest
# real income universes (BDC/mREIT ~10-12%); per bar that is 125bp at
# ppy=12 and ~6bp at ppy=252.
MAX_PLAUSIBLE_ANNUAL_DIV_YIELD = 0.15

_CHECK_MISSING = "costs.missing_transaction_costs"
_CHECK_DECLARATION = "costs.no_cost_declaration"
_CHECK_TURNOVER = "costs.turnover_unrealistic"
_CHECK_SENSITIVITY = "costs.cost_sensitivity"
_CHECK_PRICE_CONSISTENCY = "costs.price_return_consistency"


def _live_span(positions: pd.DataFrame) -> pd.DataFrame:
    """Trim leading/trailing all-zero (or NaN) rows - the flat padding
    aligned() fabricates for a late-start/early-end book, which otherwise
    dilutes every mean-turnover figure by the coverage fraction. Interior
    zero rows are genuine flat days and stay; a fully nonzero book (first
    and last rows trade) passes through bit-identical."""
    nz = positions.fillna(0.0).ne(0.0).any(axis=1)
    if not bool(nz.any()):
        return positions
    return positions.loc[nz.idxmax():nz[::-1].idxmax()]


def _trade_ledger_issue(positions: pd.DataFrame,
                        asset_returns: pd.DataFrame) -> dict[str, Any] | None:
    """Describe non-finite trade rows through the book's final liquidation.

    The self-financing drift calculation is undefined after portfolio NAV
    growth is non-positive (and can be non-finite when the denominator is
    numerically singular).  Dropping those rows lets every aggregate below
    certify the remaining subset while silently treating a post-wipeout
    restart as free trading, so each ledger-dependent check gates on this
    shared diagnostic first.
    """
    live = _live_span(positions)
    # The first flat row after the live span closes the book. Include it
    # even when its trade amount is NaN: NAV may have been wiped out on
    # the last invested row, leaving the liquidation ledger undefined.
    if len(live):
        start = positions.index.get_loc(live.index[0])
        stop = positions.index.get_loc(live.index[-1]) + 2
        ledger = positions.iloc[start:stop]
    else:
        ledger = live
    dollars = traded_dollars_series(ledger, asset_returns)
    finite = np.isfinite(dollars.to_numpy(dtype=float))
    if bool(finite.all()):
        return None
    bad = dollars.index[~finite]
    first = bad[0]
    first_text = (str(first.date()) if isinstance(first, pd.Timestamp)
                  else str(first))
    return dict(
        n_nonfinite_trade_rows=int((~finite).sum()),
        n_live_periods=int(len(live)),
        n_ledger_periods=int(len(ledger)),
        first_nonfinite_trade_date=first_text,
        trade_ledger_undefined=True,
    )


def _stable_finite_mean(values) -> float:
    """Mean of finite float64 values without overflowing their sum."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return float("nan")
    scale = float(np.max(np.abs(x)))
    if scale == 0.0:
        return 0.0
    return scale * float(np.mean(x / scale))


def _mean_one_sided_turnover(positions: pd.DataFrame,
                             asset_returns: pd.DataFrame) -> float:
    """Mean one-sided turnover over the live span, entry trade included.

    turnover_series NaNs its first row ("no prior book"), so a book whose
    live span is one bar - a 100% round trip, 2.0 dollars traded - would
    average an empty series to 0.0 and certify costs immaterial. Averaging
    the dollars series instead counts the entry bar;
    NAV-wipe NaN rows (undefined weights) are excluded from the mean and
    surfaced by the per-date comparison instead."""
    live = _live_span(positions)
    traded = traded_dollars_series(live, asset_returns).dropna()
    return float(0.5 * _stable_finite_mean(traded)) if len(traded) else 0.0


# ---------------------------------------------------------------------------
# 1. costs.missing_transaction_costs
# ---------------------------------------------------------------------------

def _cash_carry_fit(drag: pd.Series, dollars: pd.Series,
                    positions: pd.DataFrame, declared: float | None,
                    ppy: float) -> dict[str, float] | None:
    """Fit ``drag_t = slope * dollars_traded_t + intercept`` and test the
    honest cash-carry decomposition (see the CASH_CARRY_* constants block):
    slope = truly-charged one-way cost, -intercept = constant per-bar carry
    on the uninvested fraction. Returns the fit details when all waiver
    conditions hold, else None (the accusation branches proceed)."""
    x = dollars.to_numpy(dtype=float)
    y = drag.to_numpy(dtype=float)
    n = int(x.size)
    if n < CASH_CARRY_MIN_DATES:
        return None
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if not np.isfinite(sxx) or sxx <= 0.0:
        return None
    slope = float(xc @ (y - y.mean()) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid_std = float(np.std(y - (slope * x + intercept)))
    slope_bps = 1e4 * slope
    carry = -intercept
    if not (np.isfinite(slope_bps) and np.isfinite(carry)
            and np.isfinite(resid_std)):
        return None
    # (a) the slope must be a real cost model, consistent with the
    # declaration when one exists (a truly-uncharged book fits slope ~ 0)
    if declared is not None and declared > 0:
        if not (declared / DECLARED_IMPLIED_MISMATCH_FACTOR <= slope_bps
                <= declared * DECLARED_IMPLIED_MISMATCH_FACTOR):
            return None
    elif not (MIN_PLAUSIBLE_ONE_WAY_BPS <= slope_bps
              <= MAX_PLAUSIBLE_ONE_WAY_BPS):
        return None
    # (b) a positive constant accrual bounded by a plausible yield on the
    # idle cash the positions leave uninvested (none on a fully-invested
    # or unit-gross long-short book - no waiver there)
    gross_expo = positions.fillna(0.0).abs().sum(axis=1).reindex(drag.index)
    uninvested = float(np.clip(1.0 - float(gross_expo.mean()), 0.0, 1.0))
    carry_cap = MAX_PLAUSIBLE_CASH_YIELD_ANNUAL * uninvested / ppy
    if not 0.0 < carry <= carry_cap:
        return None
    # (c) constant accrual, not noise the line happens to center
    if resid_std > CASH_CARRY_MAX_RESID_FRAC * carry:
        return None
    return dict(slope_bps=slope_bps, carry_per_bar=carry,
                carry_annualized=carry * ppy,
                implied_cash_yield_on_uninvested=(carry * ppy / uninvested),
                uninvested_frac=uninvested, carry_cap_per_bar=carry_cap,
                resid_std=resid_std, n_dates=float(n))


def _check_missing_transaction_costs(art: BacktestArtifacts,
                                     config: AuditConfig) -> CheckResult:
    cid = _CHECK_MISSING
    if art.positions is None or art.strategy_returns is None:
        missing = [name for name, val in (("positions", art.positions),
                                          ("strategy_returns", art.strategy_returns))
                   if val is None]
        return skipped(
            cid,
            f"requires both artifacts.positions and artifacts.strategy_returns "
            f"(missing: {', '.join(missing)}). Pass "
            f"{' and '.join('artifacts.' + m for m in missing)} to verify that "
            f"transaction costs were actually charged.",
            missing=missing,
        )

    ledger_issue = _trade_ledger_issue(art.positions, art.asset_returns)
    if ledger_issue is not None:
        return failed(
            cid,
            f"the self-financing trade ledger is undefined/non-finite on "
            f"{ledger_issue['n_nonfinite_trade_rows']} row(s) across the "
            f"{ledger_issue['n_live_periods']}-bar live book and final "
            f"liquidation (first: "
            f"{ledger_issue['first_nonfinite_trade_date']}). A prior "
            f"portfolio return drove NAV growth to zero/negative or a "
            f"numerically singular value, so post-event weights and costs "
            f"cannot be reconciled; ignoring these rows can make a restarted "
            f"book appear fully costed.",
            severity=Severity.CRITICAL,
            remediation="Fix leverage/return units or terminate the return "
                        "and position series at the wipeout. If the strategy "
                        "was recapitalized, audit each capital episode as a "
                        "separate book with an explicit new entry trade.",
            details=ledger_issue,
        )

    gross = gross_strategy_returns(art.positions, art.asset_returns)
    net = art.strategy_returns
    common = gross.index.intersection(net.index)
    g = gross.loc[common]
    n = net.loc[common]
    mask = pd.Series(
        np.isfinite(g.to_numpy(dtype=float))
        & np.isfinite(n.to_numpy(dtype=float)),
        index=common,
    )
    n_common = int(mask.sum())
    if n_common == 0:
        return skipped(
            cid,
            "strategy_returns share no finite dates with the gross "
            "reconstruction from positions x asset_returns; reindex "
            "artifacts.strategy_returns onto the position dates to enable "
            "this check.",
        )

    with np.errstate(over="ignore", invalid="ignore"):
        diff = g[mask] - n[mask]                  # gross - net (cost drag)
    if not np.isfinite(diff.to_numpy(dtype=float)).all():
        return failed(
            cid,
            "gross - net cost drag is non-finite even though the compared "
            "gross and net returns are individually finite - their "
            "magnitudes overflow float64, so no transaction-cost "
            "reconciliation is numerically meaningful.",
            severity=Severity.CRITICAL,
            remediation="Fix return/weight units and rescale the artifacts "
                        "to finite practical magnitudes before auditing.",
            n_common=n_common, nonfinite_cost_drag=True,
        )
    max_abs_diff = float(diff.abs().max())
    mean_to = _mean_one_sided_turnover(art.positions, art.asset_returns)
    dollars = traded_dollars_series(art.positions, art.asset_returns)
    dollars_cmp = dollars.loc[common][mask]
    mean_dollars = _stable_finite_mean(dollars_cmp)
    mean_dollars_all = _stable_finite_mean(dollars)
    declared = art.declared_costs_bps
    paper_trading = declared is not None and declared == 0

    if mean_to <= config.min_turnover_for_cost_check:
        # Second prong before certifying costs immaterial: the mean is
        # dilution-prone - a burst can replace the book while a long quiet
        # span keeps the average under the bar. Any single post-entry bar
        # above NEGLIGIBLE_MAX_ONE_SIDED_PER_BAR falls through to the
        # actual net-vs-gross measurement instead of the shortcut.
        to_live = turnover_series(_live_span(art.positions),
                                  art.asset_returns).dropna()
        max_bar_to = float(to_live.max()) if len(to_live) else 0.0
        if max_bar_to <= NEGLIGIBLE_MAX_ONE_SIDED_PER_BAR:
            return passed(
                cid,
                f"mean one-sided turnover is {mean_to:.2%}/period (entry "
                f"trade included) - negligible "
                f"(<= {config.min_turnover_for_cost_check:.0%}/period, worst "
                f"bar {max_bar_to:.2%} one-sided), so transaction costs are "
                f"immaterial for this book and the net-vs-gross comparison "
                f"is uninformative (max |net - gross| = {max_abs_diff:.1e} "
                f"over {n_common} dates). NOTE: this shortcut MEASURES "
                f"nothing - cost charging was not verified.",
                mean_turnover=mean_to, max_abs_diff=max_abs_diff,
                n_common=n_common, max_bar_turnover=max_bar_to,
                negligible_turnover=True, unresolved=True,
            )

    # Overlap-coverage gate: the book trades materially somewhere, but do the
    # compared dates trade? If strategy_returns are non-NaN only on no-trade
    # dates, implied cost = 0/0 = NaN, NaN passes every float comparison
    # below, and the whole check would silently verify nothing.
    if mean_dollars <= MIN_OVERLAP_TRADING_FRAC * mean_dollars_all:
        return skipped(
            cid,
            f"the net-vs-gross overlap carries no trading: mean dollars "
            f"traded on the {n_common} compared dates is {mean_dollars:.2e} "
            f"vs {mean_dollars_all:.2e}/period across the full position grid "
            f"(< {MIN_OVERLAP_TRADING_FRAC:.0%} of it) while the book turns "
            f"over {mean_to:.1%}/period overall - the implied cost is "
            f"unmeasurable on these dates, so strategy_returns are reported "
            f"only where nothing was paid (mismatched artifacts or "
            f"cherry-picked reporting dates). Report one finite strategy "
            f"return on every date in the positions' live span and on the "
            f"final liquidation date so traded dates enter the comparison.",
            mean_turnover=mean_to, max_abs_diff=max_abs_diff,
            n_common=n_common, mean_dollars_traded=mean_dollars,
            mean_dollars_all_dates=mean_dollars_all,
            declared_costs_bps=float(declared) if declared is not None else None,
        )

    # Coverage uses every dollar in the ledger, including liquidation on
    # the first flat row after the live span. Subsequent flat padding adds
    # zero dollars, so it cannot dilute coverage. The live-date count stays
    # separate: flat padding must not satisfy the minimum history floor.
    # The mean-ratio gate above is a NaN-disarm (are the compared
    # dates trading at all?), not a coverage floor: one typical date of a
    # 500-date book clears it at ratio ~1 while covering 0.2% of the
    # trading. Every verdict below reports the share; a clean measurement
    # below MIN_OVERLAP_TRADED_SHARE is unjudged and cannot satisfy a strict
    # deployment gate.
    live_idx = _live_span(art.positions).index
    n_live = int(len(live_idx))
    cmp_live_idx = dollars_cmp.index.intersection(live_idx)
    n_cmp_live = int(len(cmp_live_idx))
    total_traded = float(dollars.sum())
    overlap_traded_share = (
        float(dollars_cmp.sum()) / total_traded
        if total_traded > 0 else float("nan"))
    n_uncompared_trade_dates = int(
        ((dollars > 0) & ~dollars.index.isin(dollars_cmp.index)).sum())
    coverage: dict[str, Any] = dict(
        n_live_dates=n_live, n_compared_live_dates=n_cmp_live,
        n_uncompared_trade_dates=n_uncompared_trade_dates,
        overlap_traded_share=overlap_traded_share,
        minimum_verified_traded_share=MIN_OVERLAP_TRADED_SHARE)

    def _coverage_skip(*, details: dict[str, Any] | None = None) -> CheckResult:
        evidence = dict(coverage)
        evidence.update(mean_turnover=mean_to, max_abs_diff=max_abs_diff,
                        n_common=n_common,
                        declared_costs_bps=(float(declared)
                                            if declared is not None else None))
        if details is not None:
            evidence.update(details)
        missing_share = (1.0 - overlap_traded_share
                         if np.isfinite(overlap_traded_share)
                         else float("nan"))
        count_clause = (f"only {n_common} compared dates "
                        f"(< {COST_RECONCILE_MIN_DATES}); "
                        if n_common < COST_RECONCILE_MIN_DATES else "")
        return skipped(
            cid,
            f"net-vs-gross cost charging cannot be verified and is not "
            f"judged: {count_clause}strategy_returns "
            f"covers {overlap_traded_share:.1%} of dollars traded "
            f"(entry, rebalancing and final liquidation), below the "
            f"{MIN_OVERLAP_TRADED_SHARE:.0%} verification floor "
            f"({missing_share:.1%}, {n_uncompared_trade_dates} trading "
            f"date(s) and {n_live - n_cmp_live} live position date(s) are "
            f"unverified). Report one finite strategy return on "
            f"every date on the full live position grid and the final "
            f"liquidation date so essentially all "
            f"traded dollars enter the comparison; leading/trailing "
            f"truncation is valid only when the positions live span is "
            f"truncated with it.",
            details=evidence,
        )

    if max_abs_diff <= config.gross_match_atol:
        if paper_trading:
            if not overlap_traded_share >= MIN_OVERLAP_TRADED_SHARE:
                return _coverage_skip()
            return passed(
                cid,
                f"net returns match the cost-free reconstruction (max "
                f"|net - gross| = {max_abs_diff:.1e} over {n_common} dates) "
                f"and declared_costs_bps=0 says exactly that - consistent "
                f"zero-cost paper-trading configuration; the realism of the "
                f"free-trading assumption itself is flagged by "
                f"costs.no_cost_declaration.",
                mean_turnover=mean_to, max_abs_diff=max_abs_diff,
                n_common=n_common, declared_costs_bps=0.0, **coverage,
            )
        return failed(
            cid,
            f"net returns are bit-identical to the cost-free reconstruction "
            f"(max |net - gross| = {max_abs_diff:.1e} over {n_common} dates) "
            f"while turning over {mean_to:.1%}/period one-sided - no "
            f"transaction costs were charged.",
            severity=Severity.CRITICAL,
            remediation="Deduct transaction costs in the backtest: net_t = "
                        "gross_t - dollars_traded_t * cost_bps * 1e-4, with "
                        "cost_bps covering commission + spread + impact "
                        "(5-10bps one-way is a floor for liquid US equities), "
                        "and set artifacts.declared_costs_bps to the "
                        "assumption used.",
            mean_turnover=mean_to, max_abs_diff=max_abs_diff,
            n_common=n_common,
            declared_costs_bps=(float(declared)
                                if declared is not None else None),
            **coverage,
        )

    # Net detectably differs from gross: infer the charged one-way cost.
    # mean_dollars > 0 is guaranteed by the coverage gate above.
    mean_drag = _stable_finite_mean(diff)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        implied_bps = 1e4 * mean_drag / mean_dollars
    details: dict[str, Any] = dict(
        mean_turnover=mean_to, max_abs_diff=max_abs_diff, n_common=n_common,
        implied_bps=implied_bps, mean_dollars_traded=mean_dollars,
        mean_cost_drag=mean_drag,
        implied_floor_bps=MIN_PLAUSIBLE_ONE_WAY_BPS,
        declared_costs_bps=float(declared) if declared is not None else None,
        **coverage,
    )
    if not np.isfinite(implied_bps):
        return failed(
            cid,
            f"the implied one-way cost is non-finite over {n_common} "
            f"compared dates (mean cost drag {mean_drag:.3g}, mean dollars "
            f"traded {mean_dollars:.3g}) - the artifact magnitudes overflow "
            f"the cost reconciliation, so a finite cost model was not "
            f"verified.",
            severity=Severity.CRITICAL,
            remediation="Fix return/weight units and rescale the artifacts "
                        "to finite practical magnitudes, then re-run the "
                        "cost reconciliation.",
            details=details,
        )

    # Cash-carry discriminator: before accusing the artifacts of
    # mismatch / under-charging, test whether the drag decomposes into the
    # declared cost model plus a constant positive accrual consistent with
    # cash yield on the uninvested fraction - the honest long-only
    # cash-buffer NAV shape the cash-blind gross reconstruction cannot see.
    carry_fit = _cash_carry_fit(diff, dollars_cmp,
                                art.positions, declared,
                                float(art.periods_per_year))

    def _cash_carry_scoped_warn() -> CheckResult:
        assert carry_fit is not None
        details.update(cash_carry_fit=carry_fit)
        declared_txt = (f"declared {declared:g}bps"
                        if declared is not None else "nothing declared")
        return warned(
            cid,
            f"net-vs-gross drag decomposes into a real cost model PLUS a "
            f"constant accrual INTO net: regressing per-date drag "
            f"(gross - net) on dollars traded over {n_common} dates gives "
            f"a charged one-way cost of {carry_fit['slope_bps']:.1f}bps "
            f"({declared_txt}) and a constant "
            f"+{carry_fit['carry_per_bar'] * 1e4:.2f}bp/bar of net return "
            f"from outside positions x asset_returns (residual std "
            f"{carry_fit['resid_std']:.1e}) - consistent with a cash/"
            f"financing sleeve earning "
            f"~{carry_fit['implied_cash_yield_on_uninvested']:.1%}/yr on "
            f"the ~{carry_fit['uninvested_frac']:.0%} of NAV the positions "
            f"leave uninvested, which the cash-blind gross reconstruction "
            f"mis-reads as under-/negative charging (raw implied cost "
            f"{implied_bps:.1f}bps). This check cannot verify the sleeve's "
            f"own return - declare it to close the gap.",
            severity=Severity.MEDIUM,
            remediation="Declare the cash sleeve as an explicit asset "
                        "column so the reconstruction sees it: add a CASH "
                        "column to asset_returns at the per-bar cash rate, "
                        "a constant CASH position equal to the uninvested "
                        "fraction, and an all-NaN CASH signals column - "
                        "the implied cost then matches the declaration and "
                        "this check PASSes. NOTE: a cash column present "
                        "ONLY in positions validates but is silently "
                        "dropped from the audited grid; it must appear in "
                        "asset_returns to count.",
            details=details,
        )

    # Sign check first: a negative implied cost is not a cost model at all.
    if np.isfinite(implied_bps) and implied_bps < -NEGATIVE_IMPLIED_BPS_TOL:
        if not overlap_traded_share >= MIN_OVERLAP_TRADED_SHARE:
            return _coverage_skip(details=details)
        if carry_fit is not None:
            return _cash_carry_scoped_warn()
        return warned(
            cid,
            f"net returns sit ABOVE the cost-free gross reconstruction "
            f"(implied one-way cost {implied_bps:.1f}bps < 0 over {n_common} "
            f"dates) - transaction costs cannot be negative, so positions, "
            f"asset_returns and strategy_returns do not describe the same "
            f"backtest (mismatched artifacts, different rebalance timing, or "
            f"PnL from sources outside positions x asset_returns).",
            severity=Severity.HIGH,
            remediation="Regenerate positions and strategy_returns from the "
                        "same pipeline run so net = gross - costs holds, then "
                        "re-audit; a net series that beats its own gross "
                        "reconstruction cannot be explained by transaction "
                        "costs.",
            details=details,
        )

    # Economic plausibility floor: bit-identity is evadable with sub-bp
    # jitter (net = gross - epsilon), which leaves implied ~ 0bps - not
    # negative, not identical, and otherwise four green cost checks. Real
    # execution cannot be that cheap, so the implied charge must clear the
    # floor before it counts as a cost model.
    if np.isfinite(implied_bps) and implied_bps < MIN_PLAUSIBLE_ONE_WAY_BPS:
        if paper_trading:
            if not overlap_traded_share >= MIN_OVERLAP_TRADED_SHARE:
                return _coverage_skip(details=details)
            return passed(
                cid,
                f"net returns are within noise of the cost-free "
                f"reconstruction (implied one-way cost {implied_bps:.2f}bps "
                f"over {n_common} dates) and declared_costs_bps=0 says "
                f"exactly that - consistent zero-cost paper-trading "
                f"configuration; the realism of the free-trading assumption "
                f"itself is flagged by costs.no_cost_declaration.",
                details=details,
            )
        if carry_fit is not None:
            if not overlap_traded_share >= MIN_OVERLAP_TRADED_SHARE:
                return _coverage_skip(details=details)
            return _cash_carry_scoped_warn()
        if (declared is not None
                and 0 < declared < MIN_PLAUSIBLE_ONE_WAY_BPS
                and implied_bps >= declared / DECLARED_IMPLIED_MISMATCH_FACTOR):
            if not overlap_traded_share >= MIN_OVERLAP_TRADED_SHARE:
                return _coverage_skip(details=details)
            return warned(
                cid,
                f"implied one-way cost {implied_bps:.2f}bps is consistent "
                f"with the declared {declared:g}bps, but both sit below the "
                f"{MIN_PLAUSIBLE_ONE_WAY_BPS:g}bps plausibility floor - only "
                f"the very deepest large-cap/ETF books execute below "
                f"~0.5bp one-way all-in, so the whole result rides on an "
                f"implausibly cheap execution assumption.",
                severity=Severity.HIGH,
                remediation="Justify the sub-floor cost assumption for the "
                            "traded universe (measured half-spreads + "
                            "commissions), or re-run the backtest at a "
                            "defensible one-way cost (5-10bps for liquid US "
                            "equities) and compare.",
                details=details,
            )
        declared_txt = (f" (the declared {declared:g}bps was never deducted)"
                        if declared is not None else "")
        return failed(
            cid,
            f"net returns are economically indistinguishable from gross: "
            f"implied one-way cost {implied_bps:.2f}bps per unit traded "
            f"(< {MIN_PLAUSIBLE_ONE_WAY_BPS:g}bps plausibility floor; even "
            f"the deepest US large-cap/ETF execution costs ~0.5-2bps "
            f"one-way) while turning over {mean_to:.1%}/period over "
            f"{n_common} dates - costs are not actually charged"
            f"{declared_txt}.",
            severity=Severity.CRITICAL,
            remediation="Deduct transaction costs on every dollar traded: "
                        "net_t = gross_t - dollars_traded_t * cost_bps * "
                        "1e-4 with a defensible one-way cost_bps (5-10bps "
                        "floor for liquid US equities), and set "
                        "artifacts.declared_costs_bps to the assumption "
                        "used; sub-bp jitter or rounding noise is not a "
                        "cost model.",
            details=details,
        )

    # The branches above preserve definitive evidence that costs were not
    # charged (FAIL).  Every remaining outcome could otherwise be read as a
    # verification or a scoped warning; refuse to judge it when a material
    # share of traded dollars was withheld from the comparison.
    if not overlap_traded_share >= MIN_OVERLAP_TRADED_SHARE:
        return _coverage_skip(details=details)

    # Upper plausibility bound: a barely-trading overlap can clear the
    # coverage gate yet imply an absurd per-unit charge that would
    # otherwise PASS with at most a note. Declared costs do not excuse it
    # - no declaration makes 5%-of-notional-per-trade execution real.
    if np.isfinite(implied_bps) and implied_bps > MAX_PLAUSIBLE_ONE_WAY_BPS:
        return warned(
            cid,
            f"implied one-way cost {implied_bps:.0f}bps per unit traded "
            f"exceeds the {MAX_PLAUSIBLE_ONE_WAY_BPS:g}bps plausibility "
            f"ceiling (= {MAX_PLAUSIBLE_ONE_WAY_BPS / 1e4:.0%} of notional "
            f"per trade, beyond any real equity execution) over {n_common} "
            f"dates - net and the gross reconstruction do not describe the "
            f"same backtest.",
            severity=Severity.HIGH,
            remediation="Check the units of strategy_returns (percent vs "
                        "decimal), confirm positions/asset_returns/"
                        "strategy_returns come from the same pipeline run, "
                        "and re-audit; a drag this large is a data mismatch, "
                        "not a cost model.",
            details=details,
        )

    note = ""
    if declared is not None and np.isfinite(implied_bps):
        if declared > 0 and implied_bps < declared / DECLARED_IMPLIED_MISMATCH_FACTOR:
            if carry_fit is not None:
                return _cash_carry_scoped_warn()
            return warned(
                cid,
                f"costs declared at {declared:g}bps one-way but the charge "
                f"implied by net-vs-gross is only {implied_bps:.1f}bps "
                f"(< declared/{DECLARED_IMPLIED_MISMATCH_FACTOR:g} over "
                f"{n_common} dates) - costs were declared but under-charged.",
                severity=Severity.MEDIUM,
                remediation="Make the backtest charge the declared cost on "
                            "every dollar traded (including the entry trade), "
                            "or correct artifacts.declared_costs_bps to what "
                            "the pipeline actually deducts.",
                details=details,
            )
        if implied_bps > DECLARED_IMPLIED_MISMATCH_FACTOR * declared:
            note = (f"; note: implied cost is more than "
                    f"{DECLARED_IMPLIED_MISMATCH_FACTOR:g}x the declared "
                    f"{declared:g}bps - the cost model and the declaration "
                    f"disagree")

    # Per-date reconciliation: the mean-based implied cost is blind to
    # per-date structure - a flat per-bar deduction or a shifted-date
    # charge reproduces the right mean exactly. The residual of the
    # per-trade model must be small before the PASS certifies "costs
    # charged per unit traded"; a residual that decomposes as cost +
    # constant cash carry keeps the cash-carry scoped WARN instead.
    c_hat = implied_bps * 1e-4
    d_arr = dollars_cmp.to_numpy(dtype=float)
    resid = diff.to_numpy(dtype=float) - c_hat * d_arr
    ok_r = np.isfinite(resid)
    rms_resid = (float(np.sqrt(np.mean(np.square(resid[ok_r]))))
                 if ok_r.any() else float("nan"))
    resid_scale = c_hat * float(
        np.sqrt(np.nanmean(np.square(d_arr)))) if np.isfinite(c_hat) else 0.0
    details.update(drag_rms_residual=rms_resid,
                   drag_residual_scale=resid_scale,
                   reconcile_rel_tol=COST_RECONCILE_REL_TOL)
    recon_tested = (n_common >= COST_RECONCILE_MIN_DATES and resid_scale > 0
                    and np.isfinite(rms_resid))
    if recon_tested and rms_resid > COST_RECONCILE_REL_TOL * resid_scale:
        if carry_fit is not None:
            return _cash_carry_scoped_warn()
        return warned(
            cid,
            f"costs are charged at the right AVERAGE rate (implied "
            f"{implied_bps:.1f}bps one-way over {n_common} dates) but do "
            f"not track dollars traded date by date: rms of the per-date "
            f"residual drag - implied*dollars is {rms_resid:.2e} vs a "
            f"per-trade drag scale of {resid_scale:.2e} "
            f"(> {COST_RECONCILE_REL_TOL:.0%}) - the charge is not "
            f"per-trade (a flat per-bar deduction, costs applied on "
            f"shifted dates, or fees from outside the trade ledger). "
            f"Legitimate cost models can look like this (borrow/financing "
            f"accrual, monthly commission sweeps, non-linear impact) but "
            f"must be declared as such, not certified as per-trade "
            f"charging.",
            severity=Severity.MEDIUM,
            remediation="Charge costs on each date's dollars traded "
                        "(net_t = gross_t - dollars_traded_t * cost_bps * "
                        "1e-4), or document the non-per-trade cost model "
                        "(financing schedule, sweep dates) so the per-date "
                        "drag can be reconciled against it.",
            details=details,
        )

    # Too few compared dates for a certificate: every accusation
    # above has had its chance on this overlap; a clean measurement from
    # fewer than COST_RECONCILE_MIN_DATES dates cannot verify per-date
    # charging and must not read as one - SKIP, naming what to report.
    # Both counts must clear the floor: n_common includes the flat padding
    # a short-lived book reports 0.0 on (0 dollars, 0 drag - it constrains
    # nothing), so only the compared dates inside the live span carry the
    # certificate.
    if (n_common < COST_RECONCILE_MIN_DATES
            or n_cmp_live < COST_RECONCILE_MIN_DATES):
        if n_common < COST_RECONCILE_MIN_DATES:
            why = (f"only {n_common} compared dates "
                   f"(< {COST_RECONCILE_MIN_DATES})")
            fix = "report artifacts.strategy_returns on the full position grid"
        else:
            why = (f"only {n_cmp_live} of the {n_common} compared dates fall "
                   f"in the positions' live span "
                   f"(< {COST_RECONCILE_MIN_DATES}; the rest are flat padding "
                   f"that trades nothing)")
            fix = "the live span is too short to verify per-date charging"
        return skipped(
            cid,
            f"{why} - implied cost {implied_bps:.1f}bps measured but "
            f"per-date charging cannot be verified; {fix} ({n_live} live "
            f"position dates; the compared dates carry "
            f"{overlap_traded_share:.1%} of traded dollars, including "
            f"final liquidation).",
            details=details,
        )

    declared_txt = (f"declared {declared:g}bps" if declared is not None
                    else "no declared_costs_bps on record")
    recon_txt = (f"; per-date drag reconciles with per-trade charging "
                 f"(rms residual {rms_resid:.1e})" if recon_tested else
                 f"; per-date reconciliation not evaluated (degenerate "
                 f"per-trade drag scale {resid_scale:.1e})")
    verdict_txt = (f"net returns differ from the cost-free reconstruction "
                   f"by an implied one-way cost of {implied_bps:.1f}bps per "
                   f"unit traded over {n_common} dates ({declared_txt}; mean "
                   f"one-sided turnover {mean_to:.1%}/period{recon_txt})"
                   f"{note}")
    coverage_txt = (f"verified on {n_cmp_live} of {n_live} position dates "
                    f"({overlap_traded_share:.0%} of traded dollars)")
    if not recon_tested:
        details["not_judged"] = True   # the per-date prong was not evaluated
    return passed(
        cid,
        f"{verdict_txt}; {coverage_txt}.",
        details=details,
    )


# ---------------------------------------------------------------------------
# 2. costs.no_cost_declaration
# ---------------------------------------------------------------------------

def _check_no_cost_declaration(art: BacktestArtifacts,
                               reconstruction: CheckResult) -> CheckResult:
    cid = _CHECK_DECLARATION
    reconstruction_status = reconstruction.status
    declared = art.declared_costs_bps
    if declared is None:
        if reconstruction_status is not Status.SKIP:
            # Delegation stays a PASS whatever the reconstruction found -
            # one defect, one check: the verdict on the measurement lives in
            # costs.missing_transaction_costs. But the wording must not
            # claim "verified" when that check WARNed the comparison was
            # unmeasurable or FAILed it: only its PASS counts as verified -
            # and not the negligible-turnover PASS, which measures nothing
            # (a net==gross constant-mix book would otherwise get a false
            # "verified" printed off that shortcut).
            negligible = bool(
                reconstruction.details.get("negligible_turnover", False))
            if reconstruction_status is Status.PASS and not negligible:
                return passed(
                    cid,
                    "declared_costs_bps not set, but positions and "
                    "strategy_returns were provided so cost charging was "
                    "verified against the gross reconstruction instead (see "
                    "costs.missing_transaction_costs).",
                    declared_costs_bps=None,
                    verified_by_reconstruction=True,
                    reconstruction_check_status=reconstruction_status.value,
                )
            if reconstruction_status is Status.PASS:
                return passed(
                    cid,
                    "declared_costs_bps not set; "
                    "costs.missing_transaction_costs PASSed only because "
                    "turnover is negligible - the net-vs-gross comparison "
                    "measured nothing, so cost charging was NOT verified "
                    "(it is simply immaterial for this book).",
                    declared_costs_bps=None,
                    verified_by_reconstruction=False,
                    reconstruction_check_status=reconstruction_status.value,
                    unresolved=True,
                )
            return passed(
                cid,
                f"declared_costs_bps not set; cost charging was measured "
                f"against the gross reconstruction - see "
                f"costs.missing_transaction_costs for the verdict (it "
                f"reported {reconstruction_status.value.upper()}, so the "
                f"measurement did NOT verify that costs were charged).",
                declared_costs_bps=None,
                verified_by_reconstruction=False,
                reconstruction_check_status=reconstruction_status.value,
                not_judged=True,
            )
        if art.positions is not None and art.strategy_returns is not None:
            # The artifacts are on record but the reconstruction SKIPped
            # on them (too few compared dates for a certificate,
            # or no jointly non-NaN dates at all) - nothing is declared
            # and nothing verified it, which is the same defect as the
            # missing-artifacts WARN below; it must not read as
            # "artifacts missing", and must never claim verification.
            return warned(
                cid,
                f"no declared costs and the gross reconstruction could not "
                f"verify them: declared_costs_bps is None and "
                f"costs.missing_transaction_costs SKIPped on the provided "
                f"positions/strategy_returns ({reconstruction.message})",
                severity=Severity.MEDIUM,
                remediation="Set artifacts.declared_costs_bps to the one-way "
                            "cost assumption used in the backtest, and "
                            "report artifacts.strategy_returns on the full "
                            "position grid so the implied cost can be "
                            "measured and reconciled per date.",
                declared_costs_bps=None,
                verified_by_reconstruction=False,
                reconstruction_check_status=reconstruction_status.value,
            )
        return warned(
            cid,
            "no declared costs and nothing to verify them against: "
            "declared_costs_bps is None and positions/strategy_returns are "
            "missing, so the audit cannot tell whether trading costs were "
            "charged at all.",
            severity=Severity.MEDIUM,
            remediation="Set artifacts.declared_costs_bps to the one-way cost "
                        "assumption used in the backtest, or pass "
                        "artifacts.positions and artifacts.strategy_returns "
                        "so the implied cost can be measured.",
            declared_costs_bps=None,
            verified_by_reconstruction=False,
        )
    if declared == 0:
        return warned(
            cid,
            "zero declared costs (declared_costs_bps = 0) - the backtest "
            "assumes free trading, which no live execution achieves.",
            severity=Severity.HIGH,
            remediation="Charge a realistic one-way cost (5-10bps floor for "
                        "liquid US equities, more for small caps) and set "
                        "declared_costs_bps accordingly.",
            declared_costs_bps=0.0,
        )
    return passed(
        cid,
        f"one-way transaction costs declared at {declared:g}bps per unit "
        f"traded.",
        declared_costs_bps=float(declared),
    )


# ---------------------------------------------------------------------------
# 3. costs.turnover_unrealistic
# ---------------------------------------------------------------------------

def _check_turnover_unrealistic(art: BacktestArtifacts,
                                config: AuditConfig) -> CheckResult:
    cid = _CHECK_TURNOVER
    if art.positions is None:
        return skipped(
            cid,
            "pass artifacts.positions (weights held during each period) to "
            "measure turnover.",
        )
    # Judge the book over its own live span: aligned() pads a late-start /
    # early-end book with flat rows, which dilutes mean turnover by exactly
    # the coverage fraction (a 31%-coverage book turning 64%/period live
    # would report 20%/period and PASS the 25% bar).
    pos_live = _live_span(art.positions)
    # Which series this check reports, and why it differs from
    # costs.missing_transaction_costs: the level judged here is the
    # post-entry (steady-state) one-sided turnover - turnover_series NaNs
    # the entry bar because a one-off entry trade is a dollars question
    # that annualizing would misprice (a full-book start on a 60-bar book
    # would add +1.7%/period of phantom steady-state turnover). The cost
    # check prices dollars traded, so its mean_turnover counts the entry
    # bar (_mean_one_sided_turnover). Both figures are reported, and a
    # live span with zero post-entry bars SKIPs instead of printing
    # "0.0%/period ... within the bound" over the very round trip the
    # cost check FAILs on.
    ledger_issue = _trade_ledger_issue(art.positions, art.asset_returns)
    if ledger_issue is not None:
        return skipped(
            cid,
            f"turnover is not measurable because the self-financing trade "
            f"ledger is undefined/non-finite on "
            f"{ledger_issue['n_nonfinite_trade_rows']} live row(s) (first: "
            f"{ledger_issue['first_nonfinite_trade_date']}) after NAV "
            f"growth reached zero/negative or became numerically singular; "
            f"see costs.missing_transaction_costs for the CRITICAL verdict.",
            details=ledger_issue,
        )
    to = turnover_series(pos_live, art.asset_returns).dropna()
    n_post = int(len(to))
    mean_to = float(to.mean()) if n_post else 0.0
    p95_to = float(to.quantile(0.95)) if n_post else 0.0
    ann_to = mean_to * art.periods_per_year
    mean_incl = _mean_one_sided_turnover(art.positions, art.asset_returns)
    n_live, n_grid = len(pos_live), len(art.positions)
    span_txt = (f" over the {n_live}-bar live span of the {n_grid}-bar grid"
                if n_live < n_grid else "")
    basis_txt = (f"; {n_post} post-entry bars, entry trade excluded - "
                 f"entry-inclusive mean {mean_incl:.1%}, the figure "
                 f"costs.missing_transaction_costs prices")
    details = dict(mean_turnover=mean_to, p95_turnover=p95_to,
                   ann_turnover=ann_to, n_live_periods=n_live,
                   n_grid_periods=n_grid, n_post_entry_periods=n_post,
                   mean_turnover_entry_inclusive=mean_incl)
    # Annualized bars: the config values are the daily calibration (see
    # TURNOVER_CALIBRATION_PPY) and rescale by bar frequency - a monthly
    # book is not held to 25%/month (= 3x/year, ordinary smart-beta).
    warn_ann = config.turnover_daily_warn * TURNOVER_CALIBRATION_PPY
    fail_ann = config.turnover_daily_fail * TURNOVER_CALIBRATION_PPY

    if n_post == 0:
        return skipped(
            cid,
            f"the book's live span is a single bar of the {n_grid}-bar "
            f"grid: the entry trade (entry-inclusive mean one-sided "
            f"turnover {mean_incl:.1%}) is the only trade, and a turnover "
            f"LEVEL - post-entry trading per period, annualized against "
            f"the {warn_ann:.0f}x/year realism bound - cannot be measured "
            f"from zero post-entry bars. A one-off entry is a dollars "
            f"question: costs.missing_transaction_costs prices it (its "
            f"mean_turnover counts the entry bar).",
            details=details,
        )
    if ann_to >= fail_ann:
        return failed(
            cid,
            f"mean one-sided turnover {mean_to:.1%}/period "
            f"({ann_to:.0f}x/year one-sided{span_txt}{basis_txt}) reaches "
            f"{fail_ann:.0f}x/year - the equivalent of the entire book "
            f"replaced every period at daily frequency; retail/IBKR-scale "
            f"execution cannot support this.",
            severity=Severity.HIGH,
            remediation="Slow the signal (longer horizon / smoothing), trade "
                        "toward targets with a buffer or partial rebalance, "
                        "or re-run with realistic per-trade costs to see the "
                        "surviving edge.",
            details=details,
        )
    if ann_to >= warn_ann:
        return warned(
            cid,
            f"mean one-sided turnover {mean_to:.1%}/period "
            f"({ann_to:.0f}x/year one-sided{span_txt}{basis_txt}) exceeds "
            f"{warn_ann:.0f}x/year (= {config.turnover_daily_warn:.0%}/period "
            f"at daily frequency); retail/IBKR-scale execution cannot "
            f"support this without material slippage.",
            severity=Severity.HIGH,
            remediation="Smooth the signal or rebalance less often; verify "
                        "the edge survives the costs this turnover implies "
                        "(see costs.cost_sensitivity).",
            details=details,
        )
    return passed(
        cid,
        f"mean one-sided turnover {mean_to:.1%}/period "
        f"({ann_to:.1f}x/year one-sided, p95 {p95_to:.1%}{span_txt}"
        f"{basis_txt}) is within the {warn_ann:.0f}x/year realism bound.",
        details=details,
    )


# ---------------------------------------------------------------------------
# 4. costs.cost_sensitivity
# ---------------------------------------------------------------------------

def _check_cost_sensitivity(art: BacktestArtifacts,
                            config: AuditConfig) -> CheckResult:
    cid = _CHECK_SENSITIVITY
    if art.positions is None:
        return skipped(
            cid,
            "pass artifacts.positions (weights held during each period) to "
            "run the cost-sensitivity sweep.",
        )
    ledger_issue = _trade_ledger_issue(art.positions, art.asset_returns)
    if ledger_issue is not None:
        return skipped(
            cid,
            f"cost sensitivity is not measurable because the "
            f"self-financing trade ledger is undefined/non-finite on "
            f"{ledger_issue['n_nonfinite_trade_rows']} live row(s) (first: "
            f"{ledger_issue['first_nonfinite_trade_date']}) after NAV "
            f"growth reached zero/negative or became numerically singular; "
            f"see costs.missing_transaction_costs for the CRITICAL verdict.",
            details=ledger_issue,
        )
    gross = gross_strategy_returns(art.positions, art.asset_returns)
    dollars = traded_dollars_series(art.positions, art.asset_returns)
    raw_sweep = config.cost_sensitivity_bps
    try:
        if not isinstance(raw_sweep, (list, tuple)):
            raise TypeError
        raw_sweep_len = len(raw_sweep)
        configured_sweep = [float(b) for b in raw_sweep]
    except (OverflowError, TypeError, ValueError):
        raw_sweep_len = -1
        configured_sweep = []
    if (len(configured_sweep) != raw_sweep_len
            or not all(np.isfinite(b) for b in configured_sweep)):
        return skipped(
            cid,
            "cost_sensitivity_bps contains a value that cannot be "
            "represented as a finite float, so the cost sweep is not "
            "measurable. Use finite one-way basis-point scenarios.",
            invalid_cost_sensitivity_bps=True,
        )
    sweep = [0.0] + configured_sweep
    sr_by_bps: dict[str, float] = {}
    for bps in sweep:
        net_b = gross - dollars * bps * 1e-4
        sr_by_bps[f"{bps:g}"] = float(annualized_sharpe(
            net_b, periods_per_year=art.periods_per_year))
    gross_sr = sr_by_bps["0"]

    if not np.isfinite(gross_sr):
        return skipped(
            cid,
            f"gross reconstruction from positions x asset_returns has no "
            f"finite Sharpe over {len(gross)} periods (all-NaN, constant, or "
            f"empty) - a cost sweep over it would compare NaNs, which proves "
            f"nothing. Check that positions overlap the signals x "
            f"asset_returns dates/assets with nonzero weights.",
            sr_by_bps=sr_by_bps, gross_sr=gross_sr,
        )

    table_txt = ", ".join(f"{k}bps={v:.2f}" for k, v in sr_by_bps.items())
    if not config.cost_sensitivity_bps:
        # Fail-closed: an empty sweep
        # measures nothing, and a green status must never mean "no sweep
        # run" - a config-loader bug (empty YAML list) would otherwise
        # certify cost robustness it never tested. AuditConfig rejects
        # the empty tuple at construction; this branch is the
        # belt-and-suspenders for configs built around it.
        return skipped(
            cid,
            f"cost_sensitivity_bps is empty - no sweep points configured, "
            f"so cost fragility was NOT tested (gross SR {gross_sr:.2f} "
            f"only). Set AuditConfig.cost_sensitivity_bps to at least one "
            f"one-way cost in bps (default (5, 10, 25)).",
            sr_by_bps=sr_by_bps, gross_sr=gross_sr,
        )

    mid_idx = min(1, len(config.cost_sensitivity_bps) - 1)
    mid_bps = float(config.cost_sensitivity_bps[mid_idx])
    sr_mid = sr_by_bps[f"{mid_bps:g}"]
    details = dict(sr_by_bps=sr_by_bps, gross_sr=gross_sr, mid_bps=mid_bps,
                   sr_at_mid_bps=sr_mid)

    if (np.isfinite(gross_sr) and gross_sr > MIN_GROSS_SR_FOR_SENSITIVITY
            and np.isfinite(sr_mid) and sr_mid <= 0):
        return warned(
            cid,
            f"edge flips negative at {mid_bps:g}bps one-way (gross SR "
            f"{gross_sr:.2f} -> {sr_mid:.2f}; {table_txt}) - cost-fragile.",
            severity=Severity.HIGH,
            remediation="Reduce turnover (slower signal, trade buffers) or "
                        "improve per-trade edge; do not deploy an alpha whose "
                        "sign depends on sub-10bps execution.",
            details=details,
        )
    # The PASS verdict is unconditional below the gross-SR gate, but the
    # message must match the printed sweep: "does not flip" is only claimed
    # when the mid-sweep Sharpe actually stays positive.
    if not np.isfinite(sr_mid):
        msg = (f"Sharpe by one-way cost: {table_txt} - Sharpe at "
               f"{mid_bps:g}bps is not finite, so no flip conclusion is "
               f"drawn.")
    elif sr_mid > 0:
        msg = (f"Sharpe by one-way cost: {table_txt} - the edge does not "
               f"flip negative at {mid_bps:g}bps.")
    elif gross_sr <= 0:
        msg = (f"gross edge is already non-positive (gross SR "
               f"{gross_sr:.2f}), so there is no positive edge for costs to "
               f"flip; Sharpe by one-way cost: {table_txt}.")
    else:   # 0 < gross_sr <= MIN_GROSS_SR_FOR_SENSITIVITY, sr_mid <= 0
        msg = (f"no material gross edge to protect (gross SR {gross_sr:.2f} "
               f"<= the {MIN_GROSS_SR_FOR_SENSITIVITY:g} floor), though the "
               f"sweep does flip negative at {mid_bps:g}bps; Sharpe by "
               f"one-way cost: {table_txt}.")
    return passed(cid, msg, details=details)


# ---------------------------------------------------------------------------
# 5. costs.price_return_consistency
# ---------------------------------------------------------------------------

def _price_quantum_grid(px: pd.DataFrame) -> pd.DataFrame:
    """Per-cell effective price quantum: the coarsest ``PRICE_TICK / f``
    grid each stored price sits on (f in ``TICK_SPLIT_FACTORS``), NaN where
    no candidate matches (full-precision price). Split-adjusted vendor
    closes are rounded raw closes divided by the split factor, so pre-split
    rows sit on a tick/factor grid - per-cell inference lets each asset
    (and each split regime within an asset) carry its own rounding bound
    instead of one off-grid cell disarming the waiver panel-wide."""
    arr = px.to_numpy(dtype=float)
    q = np.full(arr.shape, np.nan)
    finite = np.isfinite(arr)
    for f in TICK_SPLIT_FACTORS:      # ascending f: coarsest grid wins
        quantum = PRICE_TICK / f
        mult = arr / quantum
        on = finite & (np.abs(mult - np.round(mult)) < TICK_GRID_ATOL)
        q[on & np.isnan(q)] = quantum
    return pd.DataFrame(q, index=px.index, columns=px.columns)


def _check_price_return_consistency(art: BacktestArtifacts) -> CheckResult:
    """Do ``prices`` and ``asset_returns`` describe the same series?

    Compares close-to-close returns derived from the price panel
    (p_t / p_{t-1} - 1; no forward-fill across gaps, so dead-asset NaNs stay
    NaN) against asset_returns cell-wise. Close-only prices legitimately lag
    total returns by the dividend on ex-dates - sparse, sub-15% points that
    cannot move the median - so the primary verdict is the median
    discrepancy, plus a split-magnitude prong for the sparse-but-huge
    unadjusted-split case and a pervasiveness prong for mismatches
    confined to a minority of assets, which leave the pooled median clean
    while daily-vol errors cover up to ~50% of cells (mixed-vendor feeds).
    At coarse bar frequency the dividend mass per bar stops being sparse
    (monthly payers on monthly bars put ~yield/12 in every cell), so a
    median offset that carries the dividend signature - strictly one-sided
    negative and below the plausible per-bar yield cap - is passed as
    dividend income rather than warned as a mismatch. Cent-quantized close
    files put symmetric rounding noise scaling as 1/price in every cell
    (~2.9bp at the median for a $10 stock), so on cent-grid panels whose
    diffs carry the rounding signature the median tolerance is
    price-level-aware and discrepant cells are counted only above their
    per-cell rounding bound.
    """
    cid = _CHECK_PRICE_CONSISTENCY
    if art.prices is None:
        return skipped(
            cid,
            "pass artifacts.prices (close panel, dates x assets) to verify "
            "that asset_returns were derived from the same price series - "
            "the adjusted/unadjusted mix-up detector needs both.",
        )
    idx = art.prices.index.intersection(art.asset_returns.index)
    cols = art.prices.columns.intersection(art.asset_returns.columns)
    px = art.prices.loc[idx, cols].astype(float)
    rets = art.asset_returns.loc[idx, cols].astype(float)
    # p_t/p_{t-1} - 1 instead of pct_change(): no fill_method semantics, a
    # NaN gap (delisted asset) yields NaN instead of a spurious gap return.
    derived = (px / px.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
    discrepancy = derived - rets
    # Pool the jointly observed cells positionally rather than through
    # ``DataFrame.stack()``: the stacked (date, asset) index would carry the
    # column labels, and pandas 3's stack refuses missing-like labels
    # (``None``/``NaN``), which are valid, distinct columns here. Every
    # per-cell companion below (rounding bounds, tick scales) is flattened
    # with the same ``cells`` mask so the series stay aligned.
    disc_vals = discrepancy.to_numpy(dtype=float)
    cells = np.flatnonzero(np.isfinite(disc_vals.ravel()))

    def _pooled(frame: pd.DataFrame) -> pd.Series:
        """Cells of ``frame`` (same grid as ``discrepancy``) at ``cells``."""
        return pd.Series(frame.to_numpy(dtype=float).ravel()[cells])

    sd = _pooled(discrepancy)                 # signed: dividends are < 0
    d = sd.abs()
    n_cells = int(len(d))
    if n_cells < PRICE_CONSISTENCY_MIN_CELLS:
        return skipped(
            cid,
            f"prices and asset_returns share only {n_cells} jointly-observed "
            f"return cells (need >= {PRICE_CONSISTENCY_MIN_CELLS}) - too few "
            f"to separate a systematic mismatch from noise. Align the prices "
            f"panel onto the asset_returns grid.",
            n_cells=n_cells,
        )

    # Coverage accounting (see PRICE_COVERAGE_MIN_FRAC): what fraction of
    # the finite asset_returns cells did the pooled compare actually see?
    # The first grid row is excluded from the denominator (derived is
    # structurally NaN there), so full-coverage panels read exactly 1.0.
    ret_finite = rets.iloc[1:].notna()
    n_ret_cells = int(ret_finite.to_numpy().sum())
    frac_covered = (n_cells / n_ret_cells) if n_ret_cells else 0.0
    # Work by physical column rather than grouping the stacked asset-label
    # level.  Flat columns may legally contain heterogeneous hashable labels
    # (for example a Timestamp and a frozenset), which pandas groupby tries
    # to sort by default and raises because those objects are incomparable.
    # Missing-like labels are another reason not to group: ``None`` and
    # ``np.nan`` are distinct, valid columns here but groupby coalesces them
    # into one NA bucket.  Column-wise reductions preserve both identity and
    # position without imposing an undocumented ticker-label restriction.
    per_asset_cmp = discrepancy.notna().sum(axis=0)
    n_assets_total = int((ret_finite.sum(axis=0) > 0).sum())
    n_assets_compared = int(
        (per_asset_cmp >= PRICE_COVERAGE_MIN_CELLS_PER_ASSET).sum())
    full_coverage = n_cells >= n_ret_cells
    scope_txt = ("" if full_coverage else
                 f" Scope: prices cover {n_cells} of {n_ret_cells} finite "
                 f"return cells ({frac_covered:.0%}; {n_assets_compared} of "
                 f"{n_assets_total} assets with >= "
                 f"{PRICE_COVERAGE_MIN_CELLS_PER_ASSET} compared cells) - "
                 f"the verdict applies only to the covered cells.")

    med = float(d.median())
    med_signed = float(sd.median())
    # Tick-quantization handling (see the PRICE_TICK constants block):
    # cent-grid prices put ~0.293*tick/p of rounding error at the median of
    # every honest low-priced panel, so the median tolerance and the
    # discrepant-cell count are made price-level-aware - but only when the
    # diffs actually carry the rounding signature (two-sided, per-cell
    # bounded), which the real defect classes break by 20-100x.
    p_prev = px.shift(1)
    px_finite = px.to_numpy(dtype=float)
    px_finite = px_finite[np.isfinite(px_finite)]
    on_cent_grid = px_finite.size > 0 and bool(
        np.max(np.abs(px_finite * 100.0 - np.round(px_finite * 100.0)))
        < 1e-6)
    # Per-cell quantum: each price carries its own inferred grid,
    # so a split-adjusted asset (half/third-cent grid) neither disarms the
    # waiver for the rest of the panel nor overstates its own bound with
    # the global tick. A return cell is quantized when both endpoint prices
    # sit on an inferred grid; per-cell rounding ceiling
    # |diff| <= ~(q_t/p_t + q_{t-1}/p_{t-1})/2.
    q_grid = _price_quantum_grid(px)
    q_prev = q_grid.shift(1)
    bound = _pooled((q_grid / px + q_prev / p_prev) / 2.0)
    quantized = bound.notna()
    n_quantized = int(quantized.sum())
    frac_quantized = n_quantized / n_cells if n_cells else 0.0
    if n_quantized:
        d_q = d[quantized]
        med_q = float(d_q.median())
        med_signed_q = float(sd[quantized].median())
        ratio_q99 = float((d_q / bound[quantized]).quantile(0.99))
        d_u = d[~quantized]
        # non-quantized cells have no rounding excuse: their own median
        # must already be clean or the lift is withheld (a defect confined
        # to full-precision cells cannot ride the quantized subset's waiver)
        unq_clean = (len(d_u) == 0
                     or float(d_u.median()) <= PRICE_RETURN_MEDIAN_TOL)
        rounding_shaped = (
            abs(med_signed_q) <= TICK_TWO_SIDED_MAX_MEDIAN_RATIO * med_q
            and ratio_q99 <= TICK_CELL_BOUND_MULT
            and unq_clean)
        # quantized cells count as discrepant only above their own rounding
        # bound; non-quantized cells (bound NaN -> 0 here) stay at the
        # strict point tolerance
        disc = d > np.maximum(PRICE_RETURN_POINT_TOL,
                              (TICK_CELL_BOUND_MULT * bound).fillna(0.0))
        tick_med_scale = float(_pooled(q_prev / p_prev)[quantized].median())
    else:
        rounding_shaped = False
        disc = d > PRICE_RETURN_POINT_TOL
        tick_med_scale = float(_pooled(PRICE_TICK / p_prev).median())
    tol_med = (max(PRICE_RETURN_MEDIAN_TOL,
                   TICK_MEDIAN_TOL_MULT * tick_med_scale)
               if rounding_shaped else PRICE_RETURN_MEDIAN_TOL)
    frac_discrepant = float(disc.mean())
    n_split_sized = int((d >= SPLIT_SIZED_DISCREPANCY).sum())
    worst_row, worst_col = np.unravel_index(cells[int(d.to_numpy().argmax())],
                                            disc_vals.shape)
    worst_date, worst_asset = idx[worst_row], cols[worst_col]
    worst = float(d.max())
    q90_signed = float(sd.quantile(0.90))
    div_cap = MAX_PLAUSIBLE_ANNUAL_DIV_YIELD / float(art.periods_per_year)
    details = dict(
        median_abs_diff=med, n_cells=n_cells,
        frac_discrepant=frac_discrepant, n_split_sized=n_split_sized,
        worst_abs_diff=worst, worst_date=str(worst_date),
        worst_asset=str(worst_asset),
        median_tol=tol_med,
        point_tol=PRICE_RETURN_POINT_TOL,
        q90_signed_diff=q90_signed,
        div_cap_per_bar=div_cap,
        median_signed_diff=med_signed,
        on_cent_grid=on_cent_grid,
        rounding_shaped=rounding_shaped,
        tick_median_scale=tick_med_scale,
        frac_cells_quantized=float(frac_quantized),
        n_return_cells=n_ret_cells,
        frac_return_cells_covered=float(frac_covered),
        n_assets_total=n_assets_total,
        n_assets_compared=n_assets_compared,
    )

    # Dividend signature at coarse bar frequency: a one-sided negative
    # offset (derived below declared) bounded by a plausible per-bar yield
    # is regular dividend income on close-only prices - the legitimate
    # combination the docstring promises to tolerate - not a data mismatch.
    # q90 of the signed diff <= median tol establishes one-sidedness (any
    # two-sided defect at per-bar vol scale blows past it); the cap keeps
    # detection of larger systematic errors intact at every frequency.
    dividend_shaped = (med <= div_cap
                       and q90_signed <= PRICE_RETURN_MEDIAN_TOL)
    if med > tol_med and not dividend_shaped:
        pervasive = frac_discrepant >= PRICE_RETURN_PERVASIVE_FRAC
        return warned(
            cid,
            f"close-to-close returns derived from artifacts.prices "
            f"systematically disagree with asset_returns: median |diff| = "
            f"{med * 1e4:.1f}bp/day over {n_cells} cells "
            f"({frac_discrepant:.0%} of cells off by > "
            f"{PRICE_RETURN_POINT_TOL * 1e4:.0f}bp) - the mismatch has "
            f"neither the one-sided sub-{div_cap * 1e4:.0f}bp/bar signature "
            f"of regular dividends on close-only prices nor the symmetric "
            f"per-cell-bounded signature of tick rounding, so prices and "
            f"asset_returns do not describe the same series (adjusted vs "
            f"unadjusted closes, shifted dates, percent-vs-decimal scaling, "
            f"storage rounding coarser than the $0.01 tick, or a different "
            f"vendor's data).",
            severity=Severity.HIGH if pervasive else Severity.MEDIUM,
            remediation="Rebuild prices and asset_returns from the same "
                        "adjustment-consistent source so asset_returns == "
                        "prices.pct_change() up to dividend points; a "
                        "scale/date mismatch here silently mis-sizes every "
                        "dollar-traded, commission, and share-count figure "
                        "downstream.",
            details=details,
        )
    if n_split_sized >= SPLIT_SIZED_MIN_CELLS:
        return warned(
            cid,
            f"median price-vs-return discrepancy is clean "
            f"({med * 1e4:.2f}bp) but {n_split_sized} cells disagree by >= "
            f"{SPLIT_SIZED_DISCREPANCY:.0%} (worst {worst:.1%} on "
            f"{worst_asset} at {pd.Timestamp(worst_date).date()}) - no "
            f"dividend is that large; one of the two series carries "
            f"unadjusted split jumps the other does not.",
            severity=Severity.HIGH,
            remediation="Apply the same split adjustment to prices and "
                        "asset_returns (or to neither) and re-derive; each "
                        "unadjusted split injects a fake ±33-50% return or "
                        "mis-scales every share count computed from the "
                        "price level.",
            details=details,
        )
    # Pervasiveness prong, independent of the pooled median: a mismatch
    # confined to a minority of assets (mixed-vendor / mixed-close-calendar
    # feeds) leaves the pooled median at 0.00bp while daily-vol-scale errors
    # cover up to ~50% of cells - the exact commission-mis-sizing class this
    # check exists for, invisible to the median prong. Only the dividend
    # signature on the discrepant subset (strictly one-sided negative,
    # per-bar-yield-bounded - the coarse-bar heavy-payer layout where
    # frac_discrepant is legitimately 1.0) waives it; tick-rounding cells
    # are already excluded from `disc` on quantized panels.
    if frac_discrepant >= PRICE_RETURN_PERVASIVE_FRAC:
        sub = sd[disc]
        sub_q90_signed = float(sub.quantile(0.90))
        sub_med_abs = float(sub.abs().median())
        subset_dividend_shaped = (sub_q90_signed <= PRICE_RETURN_MEDIAN_TOL
                                  and sub_med_abs <= div_cap)
        per_asset_med = discrepancy.abs().median(axis=0, skipna=True)
        n_assets_bad = int((per_asset_med > PRICE_RETURN_POINT_TOL).sum())
        details.update(discrepant_q90_signed=sub_q90_signed,
                       discrepant_median_abs=sub_med_abs,
                       n_assets_discrepant_median=n_assets_bad,
                       n_assets_any_compared_cell=int(per_asset_med.size))
        if not subset_dividend_shaped:
            return warned(
                cid,
                f"the pooled median |diff| is clean ({med * 1e4:.2f}bp) but "
                f"{frac_discrepant:.0%} of cells disagree by > "
                f"{PRICE_RETURN_POINT_TOL * 1e4:.0f}bp (median |diff| "
                f"{sub_med_abs * 1e4:.0f}bp on the discrepant cells, "
                f"{n_assets_bad} of {per_asset_med.size} assets discrepant "
                f"at their own median) - a mismatch this widespread that "
                f"lacks the one-sided sub-{div_cap * 1e4:.0f}bp/bar dividend "
                f"signature is a prices-vs-returns mismatch confined to a "
                f"subset of assets (mixed vendors, a shifted close calendar "
                f"on some names, or partial adjustment), which the pooled "
                f"median cannot see.",
                severity=(Severity.HIGH
                          if frac_discrepant >= 2 * PRICE_RETURN_PERVASIVE_FRAC
                          else Severity.MEDIUM),
                remediation="Compare derived vs declared returns per asset "
                            "(details name the discrepant-asset count) and "
                            "rebuild the mismatched names from the same "
                            "adjustment-consistent source; every dollar, "
                            "commission, and share-count figure on those "
                            "assets is currently mis-sized.",
                details=details,
            )
    # Fail-closed coverage gate: the defect
    # prongs above ran on whatever was covered - detection kept - but a
    # clean covered subset below the floor earns a scoped WARN, never the
    # unscoped "match" certificate: the mismatch classes this check hunts
    # (percent-vs-decimal, shifted dates, wrong vendor) are invisible on
    # the uncovered cells, and a prices panel that joins only a sliver of
    # the returns grid is itself evidence of a ticker-join/alignment
    # problem.
    if frac_covered < PRICE_COVERAGE_MIN_FRAC:
        return warned(
            cid,
            f"prices cover only {frac_covered:.0%} of the finite "
            f"asset_returns cells ({n_cells} of {n_ret_cells}; "
            f"{n_assets_compared} of {n_assets_total} assets have >= "
            f"{PRICE_COVERAGE_MIN_CELLS_PER_ASSET} jointly-observed cells) "
            f"- the covered subset is clean (median |diff| "
            f"{med * 1e4:.2f}bp) but a systematic mismatch on the other "
            f"{1.0 - frac_covered:.0%} would be invisible to this check, "
            f"so no consistency certificate is issued. Returns that "
            f"continue after prices stop (e.g. a delisting carried as 0.0 "
            f"returns with no price) also count as uncovered - that is a "
            f"disagreement about when the asset existed.",
            severity=Severity.MEDIUM,
            remediation="Align the prices panel onto the asset_returns "
                        "grid (same dates x assets from the same vendor) "
                        "so every asset's returns can be verified against "
                        "its own price series; a partial join usually "
                        "means a ticker-mapping or calendar mismatch.",
            details=details,
        )
    if med > PRICE_RETURN_MEDIAN_TOL and dividend_shaped:
        # median prong waived by the dividend signature
        return passed(
            cid,
            f"returns derived from artifacts.prices sit a one-sided "
            f"{med * 1e4:.1f}bp/bar BELOW asset_returns (90th percentile of "
            f"the signed diff <= {PRICE_RETURN_MEDIAN_TOL * 1e4:.0f}bp, "
            f"within the {div_cap * 1e4:.0f}bp/bar plausible dividend mass "
            f"at periods_per_year={art.periods_per_year:g}) over {n_cells} "
            f"cells - consistent with regular dividend income on close-only "
            f"prices at coarse bar frequency, not a data mismatch."
            f"{scope_txt}",
            details=details,
        )
    if med > PRICE_RETURN_MEDIAN_TOL:
        # rounding_shaped (median prong lifted to tol_med): symmetric,
        # per-cell-bounded grid noise, largest on low-priced names. The
        # wording scopes the verdict honestly: at these price levels the
        # comparison is resolution-limited by tick quantization, so it
        # rules out defects larger than tick noise rather than certifying
        # bit-level agreement.
        return passed(
            cid,
            f"returns derived from artifacts.prices differ from "
            f"asset_returns by median |diff| = {med * 1e4:.1f}bp over "
            f"{n_cells} cells - two-sided (median signed diff "
            f"{med_signed * 1e4:+.2f}bp) and bounded cell-wise by each "
            f"price's own tick grid ({frac_quantized:.0%} of cells "
            f"grid-quantized; rounding tolerance {tol_med * 1e4:.1f}bp): "
            f"consistent with cent-quantization noise on a low-priced "
            f"close file. NOTE: at these price levels the comparison is "
            f"tick-noise-bound - it rules out mismatches larger than the "
            f"rounding scale (date shifts, unit scaling, adjustment "
            f"errors), not sub-tick discrepancies.{scope_txt}",
            details=details,
        )
    sparse_txt = (
        " (sparse, consistent with dividend/split adjustment points on "
        "close-only prices)" if frac_discrepant <= SPARSE_DIVIDEND_FRAC else
        f" - above the {SPARSE_DIVIDEND_FRAC:.0%} sparse-dividend premise "
        f"but below the {PRICE_RETURN_PERVASIVE_FRAC:.0%} pervasive-"
        f"mismatch bar, or dividend-shaped; inspect the discrepant cells "
        f"if this book's universe pays no dividends")
    return passed(
        cid,
        f"returns derived from artifacts.prices match asset_returns: median "
        f"|diff| = {med * 1e4:.2f}bp over {n_cells} cells; "
        f"{frac_discrepant:.1%} of cells differ by > "
        f"{PRICE_RETURN_POINT_TOL * 1e4:.0f}bp{sparse_txt}.{scope_txt}",
        details=details,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def run(artifacts: BacktestArtifacts, config: AuditConfig, *,
        signal_func=None, backtest_func=None) -> list[CheckResult]:
    """Transaction-cost realism checks. Static: ``signal_func`` and
    ``backtest_func`` are accepted for interface uniformity but unused."""
    del signal_func, backtest_func  # static checks only
    r_missing = _check_missing_transaction_costs(artifacts, config)
    r_declaration = _check_no_cost_declaration(artifacts,
                                               reconstruction=r_missing)
    r_turnover = _check_turnover_unrealistic(artifacts, config)
    r_sensitivity = _check_cost_sensitivity(artifacts, config)
    r_prices = _check_price_return_consistency(artifacts)
    return [r_missing, r_declaration, r_turnover, r_sensitivity, r_prices]
