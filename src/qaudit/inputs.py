"""Backtest artifacts: the data contract every check runs against.

Timing convention (one bar = one period, e.g. one trading day)
--------------------------------------------------------------
- ``asset_returns.loc[t]``  = simple return of each asset over the period
  *ending at* t (close_{t-1} -> close_t).
- ``signals.loc[t]``        = signal value *known at the end of* period t.
  With ``signal_lag = L``, the earliest positions this signal may influence
  are the positions held during period t + L.
- ``positions.loc[t]``      = portfolio weights *held during* period t; they
  earn ``asset_returns.loc[t]``. Legality: positions.loc[t] may depend only
  on ``signals.loc[t - signal_lag]`` and earlier.
- ``strategy_returns.loc[t]`` = net portfolio return over period t.
- ``universe.loc[t, a]``    = True iff asset ``a`` was actually tradeable /
  in the investable universe at t (as known *at t*, not as known today).

If your pipeline uses a different convention, remap before auditing -
every detector in this library assumes the above.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .errors import (InputValidationError, MisalignedInputError,
                     MissingArtifactError)

_MIN_OVERLAP = 30
# Finite values at/above this magnitude are corrupt for every artifact this
# package accepts. Keeping the ceiling well below sqrt(float64.max) leaves
# roughly 100 decimal orders of headroom for squares, dot products and the
# positions*returns products used by the checks. This is an arithmetic-safety
# ceiling, not an economic plausibility threshold; ordinary financial data is
# dozens of orders of magnitude smaller. Multi-period return products receive
# an additional overflow guard in ``_stats.forward_returns`` because no
# positive per-cell ceiling can make an arbitrarily long product finite.
_MAX_ABS_VALUE = 1e100
# The universe must cover at least this fraction of the signals x
# asset_returns dates. The contract above defines universe.loc[t, a] per
# trading date; a sparse-stamped universe (a common vendor artifact) shares
# some dates with the grid - passing the zero-overlap gate below - while
# aligned()'s fillna(False) then fabricates "not in universe" on every
# unstamped row. Calibration of the floor: the densest common stamping
# artifact is weekly rebalance rows (1 of 5 trading days = 20% coverage;
# month-end stamping ~4.8%, quarterly ~1.6%), while honest universes sit
# near 100% (exchange-holiday calendar drift costs a few percent) and a
# dense membership history that starts late must stay legal (a dense
# panel covering the back 60% of the sample is a legitimate late-start
# layout). 0.60 sits 3x above the worst stamping artifact and
# at/below every honest layout. positions dates are deliberately not
# floor-gated (only the zero-overlap gate below): a missing positions row
# decodes to 0.0 = flat, a legitimate sparse encoding - the dilution this
# padding causes in mean-turnover figures is handled where it bites, by
# the live-span trim in checks/costs.py.
_UNIVERSE_MIN_COVERAGE = 0.60
# positions column axis is gated: positions columns outside the signals x
# asset_returns grid are silently dropped by aligned(), zeroing the real
# book (a vendor ticker-suffix mismatch on 11 of 12 names leaves 8% of the
# book and a CLEAN report). Honest layouts lose ~0% of gross weight this
# way (a book holding only a sub-universe has fewer columns than the grid,
# not extra ones), so the gate is on the fraction of gross |weight| mass
# surviving the intersection; 0.50 flags wholesale label mismatches while
# leaving single-name/share-class nits alone.
_POSITIONS_MIN_MASS_KEPT = 0.50
# Mean stamp spacing (grid dates per universe row inside the universe's own
# span) above which a coverage shortfall is diagnosed as sparse stamping
# (weekly ~5, monthly ~21) rather than a short-but-dense membership history
# (spacing ~1.0; pure calendar drift stays under ~1.1).
_SPARSE_STAMP_SPACING = 1.5
# Interior universe NaNs are ambiguous: converting them to False can
# fabricate exits and re-entries. Reject panels above this missing-data
# threshold and require explicit membership decisions. Leading/trailing
# NaNs retain their pre-listing/post-delisting interpretation; smaller
# interior gaps remain subject to the downstream survivorship checks.
_UNIVERSE_MAX_INTERIOR_NAN_FRAC = 0.01
# declared periods_per_year vs the bar frequency observed on the signals x
# asset_returns grid (bar count / calendar span): reject when they disagree
# by more than this factor either way. periods_per_year feeds every
# annualization - the suspicious_sharpe SR cap (SR_ann = SR_bar * sqrt(ppy):
# under-declaring 252->63 halves a true-SR-9 leak below the FAIL bar),
# no_exits year counts, and annualized turnover. Calibration: the widest
# honest gap is a 365-day declaration on a business-day grid (365 / ~261
# observed = 1.40x); the nearest wrong declaration is ppy=126 on daily bars
# (261/126 = 2.07x). 1.75 sits between with margin both ways.
_PPY_TOLERANCE_FACTOR = 1.75
# Execution/label offsets ultimately reach Python ranges and pandas rolling
# windows.  Arbitrary-precision integers such as 10**100 are valid ``int``
# instances but crash those consumers during platform-sized conversion.  Ten
# thousand bars is deliberately generous (roughly forty years of daily data)
# while keeping malformed service inputs bounded.
_MAX_PERIOD_OFFSET = 10_000
# This is an operational overflow ceiling, not a realism threshold: one
# million bps already means a 10,000x one-way charge.
_MAX_COST_BPS = 1_000_000.0


def _is_finite_real(value: Any) -> bool:
    """Whether ``value`` is a non-bool real representable as finite float.

    Python integers have arbitrary precision, while every scalar artifact
    consumer eventually converts to float.  Do the conversion defensively at
    the trust boundary so overflow names the offending field.
    """
    if (isinstance(value, bool)
            or not isinstance(value,
                              (int, float, np.integer, np.floating))):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (OverflowError, TypeError, ValueError):
        return False


def _is_binary_object_column(s: pd.Series) -> bool:
    """True for an object column containing only bool/0/1/missing scalars.

    Adding ``None``/``pd.NA`` to an otherwise boolean DataFrame commonly
    produces object dtype.  That is a legitimate universe representation,
    but arbitrary object/string columns remain invalid.
    """
    for value in s.array:
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError):
                return False
            if np.isfinite(number) and number in (0.0, 1.0):
                continue
        return False
    return True


def _frame_to_float(df: pd.DataFrame) -> np.ndarray:
    """Float64 cells with every pandas/Python missing sentinel as NaN."""
    try:
        return df.to_numpy(dtype="float64", na_value=np.nan)
    except TypeError:
        # Object-bool universe columns containing pd.NA/None are the one
        # accepted object case.  pandas does not consistently honor na_value
        # while casting object arrays, so replace missing cells explicitly.
        raw = df.to_numpy(dtype=object)
        return np.where(pd.isna(raw), np.nan, raw).astype("float64")


def _describe_index(idx: pd.Index) -> str:
    if len(idx) == 0:
        return "empty index"
    return f"{len(idx)} rows, {idx[0]} .. {idx[-1]}"


def _check_frame(name: str, df: pd.DataFrame, *,
                 allow_binary_object: bool = False) -> None:
    if not isinstance(df, pd.DataFrame):
        raise InputValidationError(
            f"artifacts.{name} must be a pandas DataFrame (dates x assets), "
            f"got {type(df).__name__}. If you have a single asset, pass a "
            f"one-column DataFrame: series.to_frame(name)."
        )
    if df.empty:
        raise InputValidationError(f"artifacts.{name} is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise InputValidationError(
            f"artifacts.{name} index must be a DatetimeIndex, got "
            f"{type(df.index).__name__}. Convert with "
            f"df.index = pd.to_datetime(df.index)."
        )
    if not df.index.is_monotonic_increasing:
        raise InputValidationError(
            f"artifacts.{name} index is not sorted ascending "
            f"({_describe_index(df.index)}). Sort with df.sort_index()."
        )
    if df.index.has_duplicates:
        dups = df.index[df.index.duplicated()].unique()
        raise InputValidationError(
            f"artifacts.{name} index has {len(dups)} duplicated date(s), "
            f"e.g. {list(dups[:3])}. Deduplicate before auditing - duplicated "
            f"bars silently double positions and returns."
        )
    if isinstance(df.columns, pd.MultiIndex):
        # A stacked column level (yf.download's ('Close', ticker) shape and
        # similar vendor loaders) breaks every check that stacks/iterates
        # columns: DataFrame.stack() stacks only the last level, so a
        # 2-level frame either crashes costs.price_return_consistency
        # (constant outer level) or silently empties it (varying outer
        # level). Reject here - the one gate protects every consumer.
        raise InputValidationError(
            f"artifacts.{name} has MultiIndex columns "
            f"({df.columns.nlevels} levels, e.g. {df.columns[0]!r}); every "
            f"check expects one flat asset level. Flatten with df.columns = "
            f"df.columns.get_level_values(-1) (keep the ticker level)."
        )
    if df.columns.has_duplicates:
        dups = df.columns[df.columns.duplicated()].unique()
        raise InputValidationError(
            f"artifacts.{name} has duplicated column(s): {list(dups[:5])}."
        )
    # Complex dtypes are "numeric" to is_numeric_dtype, so they must be
    # refused before the float64 scan below: to_numpy(dtype="float64") on a
    # complex column emits numpy's ComplexWarning (a hard stop under
    # -W error) while silently discarding the imaginary part - and aligned()
    # passes numpy dtypes through untouched, so a complex panel that slipped
    # past validation would reach the checks intact and crash the lookahead
    # module. No return, signal, price or weight is complex.
    complex_cols = [c for c in df.columns
                    if pd.api.types.is_complex_dtype(df[c].dtype)]
    if complex_cols:
        raise InputValidationError(
            f"artifacts.{name} has complex-dtype column(s): "
            f"{complex_cols[:5]} (e.g. {df[complex_cols[0]].dtype}). Every "
            f"check expects real-valued data; take the real part "
            f"explicitly (df[col] = df[col].apply(np.real)) or fix the "
            f"loader that produced complex values."
        )
    # pd.api.types probes instead of np.issubdtype: the numpy probe raises a
    # raw TypeError on pandas extension dtypes (Float64/Int64/boolean/Arrow -
    # the convert_dtypes() / read_csv(dtype_backend=...) mainstream), which
    # are perfectly auditable numeric data.
    non_numeric = [
        c for c in df.columns
        if (not pd.api.types.is_numeric_dtype(df[c].dtype)
            and not pd.api.types.is_bool_dtype(df[c].dtype)
            and not (allow_binary_object
                     and pd.api.types.is_object_dtype(df[c].dtype)
                     and _is_binary_object_column(df[c])))
    ]
    if non_numeric:
        raise InputValidationError(
            f"artifacts.{name} has non-numeric column(s): {non_numeric[:5]}. "
            f"Drop identifier/text columns before auditing."
        )
    # Coerced inf scan: a nullable frame's bare to_numpy() is object-dtype,
    # which would silently skip the scan; every column is numeric/bool here,
    # so the float coercion (pd.NA -> NaN) always succeeds.
    vals = _frame_to_float(df)
    n_inf = int(np.isinf(vals).sum())
    if n_inf:
        raise InputValidationError(
            f"artifacts.{name} contains {n_inf} infinite value(s) "
            f"(often a divide-by-zero in returns or z-scores). Replace "
            f"with NaN: df.replace([np.inf, -np.inf], np.nan)."
        )
    # Finite-but-huge values exhaust the safety margin for float64 products,
    # squares, and reductions downstream; no real return/signal/price/weight
    # approaches this deliberately extravagant ceiling.
    n_huge = int((np.abs(vals) >= _MAX_ABS_VALUE).sum())
    if n_huge:
        raise InputValidationError(
            f"artifacts.{name} contains {n_huge} value(s) with magnitude "
            f">= {_MAX_ABS_VALUE:g} - magnitudes this large overflow "
            f"or exhaust the safety margin of float64 arithmetic inside "
            f"the checks (products, squares, and reductions compound). "
            f"This is unscaled or corrupt data; rescale or clean it "
            f"before auditing."
        )


def _check_series(name: str, s: pd.Series) -> None:
    if not isinstance(s, pd.Series):
        raise InputValidationError(
            f"artifacts.{name} must be a pandas Series indexed by date, got "
            f"{type(s).__name__}."
        )
    if s.empty:
        raise InputValidationError(f"artifacts.{name} is empty.")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise InputValidationError(
            f"artifacts.{name} index must be a DatetimeIndex, got "
            f"{type(s.index).__name__}."
        )
    if not s.index.is_monotonic_increasing:
        raise InputValidationError(f"artifacts.{name} index is not sorted ascending.")
    if s.index.has_duplicates:
        raise InputValidationError(f"artifacts.{name} index has duplicated dates.")
    # Complex is refused before the float64 scan (see _check_frame): the
    # coercion would emit ComplexWarning and drop the imaginary part silently.
    if pd.api.types.is_complex_dtype(s.dtype):
        raise InputValidationError(
            f"artifacts.{name} has a complex dtype ({s.dtype}); returns are "
            f"real-valued - take the real part explicitly "
            f"(s.apply(np.real)) or fix the loader that produced it."
        )
    # is_numeric_dtype accepts extension dtypes (nullable Float64, Arrow)
    # that np.issubdtype rejects with a raw TypeError; bool is rejected
    # (a bool series is not a return series).
    if (not pd.api.types.is_numeric_dtype(s.dtype)
            or pd.api.types.is_bool_dtype(s.dtype)):
        raise InputValidationError(f"artifacts.{name} must be numeric, got dtype {s.dtype}.")
    svals = s.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(svals).any():
        raise InputValidationError(f"artifacts.{name} contains infinite values.")
    if (np.abs(svals) >= _MAX_ABS_VALUE).any():
        raise InputValidationError(
            f"artifacts.{name} contains values with magnitude >= "
            f"{_MAX_ABS_VALUE:g} - overflow-level magnitudes are unscaled "
            f"or corrupt data; rescale or clean before auditing.")


def _coerce_period(name: str, period: Any) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if period is None:
        return None
    try:
        start, end = period
        start, end = pd.Timestamp(start), pd.Timestamp(end)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            f"artifacts.{name} must be a (start, end) pair coercible to "
            f"timestamps, got {period!r} ({exc})."
        ) from None
    # pd.Timestamp(None)/np.nan coerce to NaT, and every NaT comparison is
    # False - the start>end guard below and every window comparison in the
    # contamination checks would silently degrade to a PASSing 0-period
    # window. Reject the bound here (the one construction chokepoint).
    if pd.isna(start) or pd.isna(end):
        raise InputValidationError(
            f"artifacts.{name} must be two real timestamps, got "
            f"({start}, {end}) - a None/NaN bound usually means a missing "
            f"config key; pass {name}=None (the whole tuple) to declare no "
            f"window instead."
        )
    if (start.tz is None) != (end.tz is None):
        raise InputValidationError(
            f"artifacts.{name} mixes timezone footing between start={start} "
            f"and end={end} (one bound is timezone-naive and the other is "
            f"timezone-aware). Use the same footing for both bounds; naive "
            f"bounds are interpreted in the data index timezone."
        )
    if start > end:
        raise InputValidationError(
            f"artifacts.{name} start {start.date()} is after end {end.date()}."
        )
    return (start, end)


def _reconcile_period_tz(name: str,
                         period: tuple[pd.Timestamp, pd.Timestamp] | None,
                         idx: pd.DatetimeIndex,
                         ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Put period bounds on the same timezone footing as the data index.

    A tz-aware index compared against tz-naive bounds (or vice versa) raises
    a raw TypeError deep inside the contamination checks; reconcile here so
    the comparison is always legal. Naive bounds against an aware index are
    localized into the index timezone (the user wrote dates in the data's
    calendar); aware bounds against a naive index are ambiguous, so that
    combination is rejected with an actionable message.
    """
    if period is None:
        return None
    tz = idx.tz
    out = []
    for label, ts in zip(("start", "end"), period):
        if tz is not None:
            try:
                ts = (ts.tz_localize(tz) if ts.tz is None
                      else ts.tz_convert(tz))
            except Exception as exc:  # pandas timezone backends vary by build
                raise InputValidationError(
                    f"artifacts.{name} {label} ({ts}) cannot be placed in "
                    f"the data index timezone {tz!s}: {type(exc).__name__}: "
                    f"{exc}. Pass an explicit timezone-aware timestamp "
                    f"(including its UTC offset) for an ambiguous/nonexistent "
                    f"daylight-saving wall time."
                ) from None
        elif ts.tz is not None:
            raise InputValidationError(
                f"artifacts.{name} {label} ({ts}) is timezone-aware but the "
                f"data index is timezone-naive. Pass naive timestamps for "
                f"{name}, or localize your data: "
                f"df.index = df.index.tz_localize({str(ts.tz)!r})."
            )
        out.append(ts)
    return (out[0], out[1])


def _denullify_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Extension-dtype (nullable/Arrow) columns -> plain numpy float64
    (pd.NA -> NaN) so every downstream numpy op sees a mainstream dtype;
    numpy-dtype frames pass through untouched (bool signals stay bool)."""
    if not any(isinstance(dt, pd.api.extensions.ExtensionDtype)
               for dt in df.dtypes):
        return df
    return pd.DataFrame(df.to_numpy(dtype="float64", na_value=np.nan),
                        index=df.index, columns=df.columns)


def _denullify_series(s: pd.Series) -> pd.Series:
    if not isinstance(s.dtype, pd.api.extensions.ExtensionDtype):
        return s
    return pd.Series(s.to_numpy(dtype="float64", na_value=np.nan),
                     index=s.index, name=s.name)


@dataclass
class BacktestArtifacts:
    """Everything the auditor knows about one backtest. Only ``signals`` and
    ``asset_returns`` are required; every optional artifact you add unlocks
    more checks (each disabled check SKIPs with a message saying what to pass).
    """

    signals: pd.DataFrame
    asset_returns: pd.DataFrame
    positions: pd.DataFrame | None = None
    strategy_returns: pd.Series | None = None
    universe: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None
    signal_input: pd.DataFrame | None = None  # raw data your signal_func consumes
    train_period: tuple | None = None         # (start, end) inclusive
    test_period: tuple | None = None
    signal_lag: int = 1
    label_horizon: int = 1                    # periods spanned by the prediction target
    declared_costs_bps: float | None = None   # one-way cost per unit traded, in bps
    periods_per_year: float = 252.0

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`InputValidationError` with an actionable message on
        the first structural problem found."""
        for name in ("signals", "asset_returns"):
            if getattr(self, name) is None:
                raise MissingArtifactError(
                    f"artifacts.{name} is required for every audit, got None. "
                    f"Pass a dates x assets DataFrame."
                )
        _check_frame("signals", self.signals)
        _check_frame("asset_returns", self.asset_returns)
        # Value-domain gate on asset_returns only: a simple return cannot
        # lose more than 100% (-1.0 exactly is the wipeout contract and stays
        # valid - forward windows containing it compound to exactly -1.0),
        # so any finite cell below -1.0 is not a return. In practice it is a
        # percent-unit panel (-5.0 meaning -5%; every annualized figure
        # then runs on returns 100x too large) or a negative-price futures
        # bar (WTI 2020-04-20: 18.27 -> -37.63 is a -306% "simple return"
        # that no position P&L convention can carry). Other panels are not
        # floor-gated: signals/prices/positions/signal_input have no such
        # physical bound.
        rvals = self.asset_returns.to_numpy(dtype="float64", na_value=np.nan)
        below = np.isfinite(rvals) & (rvals < -1.0)
        n_below = int(below.sum())
        if n_below:
            flat = np.where(below, rvals, np.nan)
            i, j = np.unravel_index(int(np.nanargmin(flat)), flat.shape)
            raise InputValidationError(
                f"artifacts.asset_returns has {n_below} finite value(s) below "
                f"-1.0 (min {float(rvals[i, j])!r} at "
                f"{self.asset_returns.index[i]} / "
                f"{self.asset_returns.columns[j]!r}). A simple return cannot "
                f"lose more than 100% of the position (-1.0 exactly is the "
                f"wipeout and stays valid). Values like -5.0 usually mean "
                f"the panel is in percent units - divide by 100; a "
                f"negative-price futures bar (a price crossing zero) must be "
                f"set to NaN or clipped to -1.0 before auditing, since no "
                f"position P&L convention can carry it."
            )
        for name in ("positions", "universe", "prices", "signal_input"):
            val = getattr(self, name)
            if val is not None:
                _check_frame(name, val,
                             allow_binary_object=(name == "universe"))
        if self.strategy_returns is not None:
            _check_series("strategy_returns", self.strategy_returns)

        # bool is excluded (as for periods_per_year below): signal_lag=False
        # - a config-loader artifact like YAML `false` - would otherwise
        # validate as lag 0.  Lag zero itself is impossible under this
        # module's fixed timing contract: signals are stamped at the end of
        # period t, while positions.loc[t] were held during period t.  A
        # pre-bar signal feed must be remapped to that end-of-bar convention
        # rather than taken as a reason to weaken the causal contract.
        if (isinstance(self.signal_lag, bool)
                or not isinstance(self.signal_lag, (int, np.integer))
                or self.signal_lag < 1):
            raise InputValidationError(
                f"artifacts.signal_lag must be an int >= 1 (bool is not a "
                f"lag), got {self.signal_lag!r}. Under qaudit's fixed timing "
                f"contract, signal[t] is known only at the end of period t "
                f"and cannot affect positions held during that same period; "
                f"signal_lag=1 means it first affects positions held during "
                f"t+1. If your source stamps a pre-bar signal on t, remap it "
                f"so row t contains the value known at the end of t-1 (or "
                f"shift positions forward one row), then audit with "
                f"signal_lag=1."
            )
        if self.signal_lag > _MAX_PERIOD_OFFSET:
            raise InputValidationError(
                f"artifacts.signal_lag must be <= {_MAX_PERIOD_OFFSET:,} "
                f"periods, got {self.signal_lag!r}. Larger offsets are not "
                f"auditable and make alignment checks expand an effectively "
                f"unbounded lag range."
            )
        if (isinstance(self.label_horizon, bool)
                or not isinstance(self.label_horizon, (int, np.integer))
                or self.label_horizon < 1):
            raise InputValidationError(
                f"artifacts.label_horizon must be an int >= 1 (bool is not a "
                f"horizon), got {self.label_horizon!r} (number of periods "
                f"the prediction target spans)."
            )
        if self.label_horizon > _MAX_PERIOD_OFFSET:
            raise InputValidationError(
                f"artifacts.label_horizon must be <= {_MAX_PERIOD_OFFSET:,} "
                f"periods, got {self.label_horizon!r}. Larger values overflow "
                f"pandas' rolling-window conversion and cannot produce a "
                f"meaningful forward target in a practical audit."
            )
        # Type guards mirror the signal_lag pattern: config-loader garbage
        # (None, "252", "5") must raise the branded error, not a raw
        # TypeError from the comparison below. Real numbers are accepted
        # (every consumer converts via float()); bool is excluded.
        ppy = self.periods_per_year
        if not _is_finite_real(ppy) or ppy <= 0:
            raise InputValidationError(
                f"artifacts.periods_per_year must be a positive real number "
                f"(bars per year: 252 business-daily, 52 weekly, 12 monthly), "
                f"got {ppy!r} - it scales every annualized figure (Sharpe "
                f"caps, annualized turnover, year counts)."
            )
        dcb = self.declared_costs_bps
        if dcb is not None:
            if (not _is_finite_real(dcb) or dcb < 0
                    or float(dcb) > _MAX_COST_BPS):
                raise InputValidationError(
                    f"artifacts.declared_costs_bps must be None or a real "
                    f"number in [0, {_MAX_COST_BPS:g}] (one-way cost per unit "
                    f"traded, in bps - "
                    f"e.g. 5 for 5bps), got {dcb!r}. Convert config-loader "
                    f"strings to numbers before auditing."
                )
        self.train_period = _reconcile_period_tz(
            "train_period",
            _coerce_period("train_period", self.train_period),
            self.signals.index)
        self.test_period = _reconcile_period_tz(
            "test_period",
            _coerce_period("test_period", self.test_period),
            self.signals.index)

        common_idx = self.common_index
        if len(common_idx) < _MIN_OVERLAP:
            raise MisalignedInputError(
                f"signals ({_describe_index(self.signals.index)}) and "
                f"asset_returns ({_describe_index(self.asset_returns.index)}) "
                f"share only {len(common_idx)} date(s); need >= {_MIN_OVERLAP}. "
                f"Check that both use the same calendar/timezone and reindex."
            )
        common_cols = self.common_assets
        if len(common_cols) == 0:
            raise MisalignedInputError(
                f"signals columns ({list(self.signals.columns[:5])}...) and "
                f"asset_returns columns ({list(self.asset_returns.columns[:5])}...) "
                f"share no assets. Align tickers/ids before auditing."
            )

        # periods_per_year vs the observed bar frequency: the declared value
        # scales every annualized figure (the suspicious_sharpe SR cap - the
        # only magnitude backstop against disguised leaks - plus no_exits
        # year counts and annualized turnover), so an under-declared ppy
        # silently deflates the reported Sharpe by sqrt(true/declared).
        # Observed rate = bars / calendar span, continuous across calendars
        # (business-daily ~261, calendar-daily 365, weekly 52, monthly 12).
        # Subtract the integer representation in its native resolution.
        # Timestamp subtraction constructs a pandas Timedelta and overflows
        # for otherwise valid indexes spanning more than roughly 292 years.
        units_per_second = {"s": 1.0, "ms": 1e3,
                            "us": 1e6, "ns": 1e9}
        unit = common_idx.unit
        span_units = int(common_idx.asi8[-1]) - int(common_idx.asi8[0])
        span_days = span_units / units_per_second[unit] / 86400.0
        observed_ppy = (len(common_idx) - 1) * 365.25 / span_days
        declared_ppy = float(self.periods_per_year)
        ppy_ratio = max(declared_ppy / observed_ppy, observed_ppy / declared_ppy)
        if ppy_ratio > _PPY_TOLERANCE_FACTOR:
            raise InputValidationError(
                f"artifacts.periods_per_year={self.periods_per_year} is "
                f"{ppy_ratio:.1f}x off the observed bar frequency: the "
                f"signals x asset_returns grid carries ~{observed_ppy:.0f} "
                f"bars/year ({len(common_idx)} bars over "
                f"{span_days / 365.25:.1f} years; tolerance "
                f"{_PPY_TOLERANCE_FACTOR}x). Every annualized figure (the "
                f"suspicious_sharpe cap, no_exits year counts, annualized "
                f"turnover) scales with it - set periods_per_year to the "
                f"actual bar count per year (252 business-daily, 365 "
                f"calendar-daily, 52 weekly, 12 monthly), or resample the "
                f"panels to the frequency you declared. Do not use the "
                f"REBALANCE count for a book computed on finer bars."
            )

        # Content gate, not just index geometry: every gate above counts
        # dates, so a pair of all-NaN required panels (a NaN-ed vendor
        # load, a join that kept the calendar and dropped the values)
        # would sail through and certify CLEAN - performance.sample_size
        # counts common_index rows, the IC checks route NaN means to
        # "not judged" PASSes, and full_history_universe reads 0% span
        # coverage as 0% survivors. A date is usable only where at least
        # one asset carries a finite signal and a finite return on the
        # same bar. The hard floor is deliberately zero-tolerance only:
        # reject a 100%-NaN required panel and a zero-jointly-finite pair
        # (content that never overlaps), nothing more. A low-but-nonzero
        # usable count is an honest layout - a signal that goes live near
        # the end of a dense book validates with as little as one usable
        # date, and the signal-consuming checks SKIP with measured
        # overlaps; audit() discloses usable counts below _MIN_OVERLAP via
        # the audit.coverage WARN instead, which
        # report.gate(warn_severity=HIGH) turns into a deployment block.
        # No per-cell density floor either: event-gated exports and late
        # listings are legitimate sparse panels.
        sig_finite = np.isfinite(
            self.signals.loc[common_idx, common_cols].to_numpy(
                dtype="float64", na_value=np.nan))
        ret_finite = np.isfinite(
            self.asset_returns.loc[common_idx, common_cols].to_numpy(
                dtype="float64", na_value=np.nan))
        all_nan = [name for name, fin in (("signals", sig_finite),
                                          ("asset_returns", ret_finite))
                   if not fin.any()]
        if all_nan:
            raise InputValidationError(
                f"artifacts.{' and artifacts.'.join(all_nan)} "
                f"{'is' if len(all_nan) == 1 else 'are'} 100% NaN on the "
                f"{len(common_idx)}-date x {len(common_cols)}-asset "
                f"signals x asset_returns grid - there is no content to "
                f"audit, and every check would judge nothing while "
                f"reporting normally. This is usually a broken load/join "
                f"(calendar kept, values dropped); fix the data instead "
                f"of auditing it."
            )
        usable_dates = int((sig_finite & ret_finite).any(axis=1).sum())
        if usable_dates == 0:
            raise MisalignedInputError(
                f"signals and asset_returns are jointly finite (same bar, "
                f"same asset) on 0 of {len(common_idx)} shared date(s) - "
                f"the panels share a calendar but their observable "
                f"content never overlaps, so every check would audit "
                f"nothing. Check that values (not just dates) survived "
                f"your load/join and that both panels cover the same "
                f"live span."
            )

        # Optional artifacts are later reindexed onto the common grid by
        # aligned(): zero date/asset overlap would silently become an
        # all-zero positions panel, an all-False universe, or an all-NaN
        # return series - and every check consuming them would "audit"
        # nothing while reporting normally. Fail loudly instead.
        for name in ("positions", "universe", "prices"):
            df = getattr(self, name)
            if df is None:
                continue
            if len(df.index.intersection(common_idx)) == 0:
                raise MisalignedInputError(
                    f"artifacts.{name} ({_describe_index(df.index)}) shares no "
                    f"dates with the signals x asset_returns grid "
                    f"({_describe_index(common_idx)}) - aligned() would "
                    f"silently zero it out and every check consuming it would "
                    f"audit nothing. Reindex {name} onto the backtest "
                    f"calendar (same timezone/calendar as signals)."
                )
            if len(df.columns.intersection(common_cols)) == 0:
                raise MisalignedInputError(
                    f"artifacts.{name} columns ({list(df.columns[:5])}...) "
                    f"share no assets with the signals x asset_returns grid "
                    f"({list(common_cols[:5])}...). Align tickers/ids before "
                    f"auditing."
                )
        # Beyond zero overlap: a sparse universe (month-end / rebalance-day
        # vendor stamps) overlaps the grid on its stamp dates only, so it
        # survives the gate above, and every unstamped row then becomes
        # all-False in aligned(). Downstream that is not noise, it is
        # inversion: trading_outside_universe mass-flags honest positions
        # (~95% of rows "outside" a monthly-stamped universe) and no_exits
        # sees a spurious True->False exit at every stamp boundary, so a
        # survivor-only list launders itself into a healthy exit rate. The
        # docstring contract requires universe.loc[t, a] per trading date -
        # enforce it with a coverage floor instead of inventing membership.
        if self.universe is not None:
            # Value-domain gate: aligned() decodes membership via
            # astype(bool), so any nonzero numeric silently decodes as
            # True/in-universe - a -1 "removed" sentinel list becomes an
            # all-True survivor universe, and a 0.5-weighted panel
            # becomes binary membership nobody declared. Membership must
            # be binary: {True, False, 1, 0}; NaN cells stay legal and
            # decode to False (out-of-universe) by the aligned()
            # contract.
            uvals = _frame_to_float(self.universe)
            finite_uvals = uvals[np.isfinite(uvals)]
            bad_uvals = finite_uvals[(finite_uvals != 0.0)
                                     & (finite_uvals != 1.0)]
            if bad_uvals.size:
                examples = [float(v) for v in np.unique(bad_uvals)[:5]]
                raise InputValidationError(
                    f"artifacts.universe contains {bad_uvals.size} "
                    f"value(s) outside {{0, 1, True, False, NaN}} "
                    f"(e.g. {examples}) - membership is decoded via "
                    f"astype(bool), so any nonzero value (a -1 'removed' "
                    f"sentinel included) would silently count as "
                    f"in-universe. Re-encode membership explicitly (e.g. "
                    f"universe > 0 for positive-means-member; map "
                    f"sentinel codes yourself) before auditing."
                )
            # Same fabrication along the column axis: a universe artifact
            # that lacks a column for a traded asset (ticker-suffix/share-
            # class join artifacts, a constituents file omitting hedge
            # names) is reindexed to an all-NaN column and fillna(False)d by
            # aligned() - every trade in that asset is then convicted of
            # "hindsight universe construction". Column absence is
            # ambiguous (label mismatch vs genuinely-never-a-member), so
            # membership must be declared explicitly; the floor is 100%
            # because any silently fabricated column mass-flags exactly
            # that asset. A column that is present but all-NaN fabricates
            # identically and is treated the same.
            missing_cols = common_cols.difference(self.universe.columns)
            present = self.universe.columns.intersection(common_cols)
            overlap = self.universe.index.intersection(common_idx)
            # Membership outside the audited calendar cannot resolve an
            # entirely missing column inside it. Inspect the same dates
            # aligned() will retain: a warmup-only stamp otherwise lets a
            # blank live-history column become all-False silently.
            # Select the complete column Index at once: a scalar tuple or
            # frozenset asset id is otherwise interpreted by .loc as a
            # collection of column selectors rather than one asset label.
            membership = self.universe.loc[overlap, present]
            all_nan_cols = list(present[membership.isna().all(axis=0)])
            if len(missing_cols) or all_nan_cols:
                described = []
                if len(missing_cols):
                    described.append(
                        f"no column at all for {list(missing_cols[:5])}"
                        + ("..." if len(missing_cols) > 5 else ""))
                if all_nan_cols:
                    described.append(
                        f"an all-NaN column on the audited calendar for "
                        f"{all_nan_cols[:5]}"
                        + ("..." if len(all_nan_cols) > 5 else ""))
                n_bad = len(missing_cols) + len(all_nan_cols)
                raise MisalignedInputError(
                    f"artifacts.universe has no membership data for {n_bad} "
                    f"of {len(common_cols)} traded assets "
                    f"({'; '.join(described)}) - aligned() would "
                    f"fillna(False) them, silently marking those assets "
                    f"never-in-universe (mass false "
                    f"trading_outside_universe violations). If these are "
                    f"ticker label mismatches (suffixes/share classes), "
                    f"align universe columns to the signals/asset_returns "
                    f"tickers; if the names were genuinely never in the "
                    f"universe, declare that explicitly with all-False "
                    f"columns."
                )
            coverage = len(overlap) / len(common_idx)
            if coverage < _UNIVERSE_MIN_COVERAGE:
                span = common_idx[(common_idx >= overlap[0])
                                  & (common_idx <= overlap[-1])]
                spacing = len(span) / len(overlap)
                if spacing >= _SPARSE_STAMP_SPACING:
                    hint = (
                        f"the universe looks stamped every ~{spacing:.0f} "
                        f"trading days (a month-end/rebalance-date vendor "
                        f"artifact). Forward-fill membership onto the "
                        f"trading calendar before auditing: "
                        f"universe = universe.astype('boolean').reindex("
                        f"dates, method='ffill').fillna(False) with dates = "
                        f"the signals x "
                        f"asset_returns calendar."
                    )
                else:
                    hint = (
                        f"the universe's {len(overlap)}-date history is "
                        f"dense but spans only part of the backtest. Trim "
                        f"signals/asset_returns to the membership history, "
                        f"or extend the universe over the full calendar."
                    )
                raise MisalignedInputError(
                    f"artifacts.universe covers only {len(overlap)} of "
                    f"{len(common_idx)} trading dates ({coverage:.1%} < "
                    f"{_UNIVERSE_MIN_COVERAGE:.0%} floor) on the signals x "
                    f"asset_returns grid - aligned() would fillna(False) "
                    f"the other {len(common_idx) - len(overlap)} rows, "
                    f"silently marking every asset out-of-universe there "
                    f"(mass false trading_outside_universe violations, "
                    f"fake exits in no_exits); {hint}"
                )
            # Interior NaN mass gate (see the _UNIVERSE_MAX_INTERIOR_NAN_
            # FRAC calibration comment): scattered NaN cells strictly
            # inside a column's own finite span decode to False and
            # fabricate membership exits/out-bars at scale. Leading
            # (pre-listing) and trailing (post-delisting) runs decode to
            # False correctly and stay exempt; sub-floor interior NaN
            # stays legal by the aligned() contract.
            gvals = _frame_to_float(self.universe.loc[
                self.universe.index.isin(common_idx), common_cols
            ])
            g_isna = np.isnan(gvals)
            if g_isna.any():
                g_finite = ~g_isna
                n_grows = gvals.shape[0]
                first_fin = g_finite.argmax(axis=0)
                last_fin = n_grows - 1 - g_finite[::-1].argmax(axis=0)
                row_ids = np.arange(n_grows)[:, None]
                interior = (g_isna
                            & (row_ids > first_fin[None, :])
                            & (row_ids < last_fin[None, :]))
                n_interior = int(interior.sum())
                interior_frac = n_interior / gvals.size
                if interior_frac > _UNIVERSE_MAX_INTERIOR_NAN_FRAC:
                    n_cols_hit = int(interior.any(axis=0).sum())
                    raise InputValidationError(
                        f"artifacts.universe has {n_interior} NaN cell(s) "
                        f"({interior_frac:.2%} of the {gvals.size:,} grid-"
                        f"overlap cells, > the "
                        f"{_UNIVERSE_MAX_INTERIOR_NAN_FRAC:.1%} floor) "
                        f"strictly inside the finite membership span of "
                        f"{n_cols_hit} column(s). Each such cell decodes "
                        f"to False in aligned() and fabricates both a "
                        f"membership exit (laundering "
                        f"survivorship.no_exits with fake attrition) and "
                        f"an out-of-universe decision bar. NaN membership "
                        f"at this scale is ambiguous data, not a "
                        f"declaration - resolve NaN membership "
                        f"explicitly: ffill "
                        f"(universe.ffill() carries the last known flag "
                        f"through data holes), or set False deliberately "
                        f"(universe.fillna(False)) if the names were "
                        f"genuinely out. Pre-listing and post-delisting "
                        f"NaN runs are exempt and may stay."
                    )
        # positions column-mass gate (see the _POSITIONS_MIN_MASS_KEPT
        # calibration comment): a wholesale ticker-label mismatch would let
        # aligned() silently drop the real book and audit a sliver. Date
        # sparseness is deliberately not gated here - missing rows decode
        # to flat by contract.
        if self.positions is not None:
            pos_abs = np.abs(self.positions.to_numpy(dtype="float64",
                                                     na_value=0.0))
            total_mass = float(pos_abs.sum())
            kept_cols_mask = self.positions.columns.isin(common_cols)
            kept_mass = float(pos_abs[:, kept_cols_mask].sum())
            # fail-closed: an inf/inf overflow makes the ratio NaN, and
            # `NaN < floor` is False - write the gate as not->=, so an
            # unmeasurable mass ratio rejects instead of slipping through
            if total_mass > 0 and not (
                    kept_mass / total_mass >= _POSITIONS_MIN_MASS_KEPT):
                dropped = self.positions.columns[~kept_cols_mask]
                raise MisalignedInputError(
                    f"artifacts.positions carries only "
                    f"{kept_mass / total_mass:.0%} of its gross weight in "
                    f"columns shared with the signals x asset_returns grid "
                    f"(< {_POSITIONS_MIN_MASS_KEPT:.0%} floor) - aligned() "
                    f"would silently drop the other {len(dropped)} "
                    f"column(s) (e.g. {list(dropped[:5])}) and every "
                    f"cost/turnover check would judge the surviving sliver "
                    f"of the book while reporting normally. If these are "
                    f"ticker label mismatches (vendor suffixes, share "
                    f"classes), align positions columns to the "
                    f"signals/asset_returns tickers before auditing."
                )
        if self.strategy_returns is not None:
            sr_overlap = self.strategy_returns.index.intersection(common_idx)
            if len(sr_overlap) == 0:
                raise MisalignedInputError(
                    f"artifacts.strategy_returns "
                    f"({_describe_index(self.strategy_returns.index)}) shares "
                    f"no dates with the signals x asset_returns grid "
                    f"({_describe_index(common_idx)}) - aligned() would "
                    f"reduce it to all-NaN and the cost/performance checks "
                    f"would audit nothing. Reindex strategy_returns onto the "
                    f"backtest calendar."
                )
            # Content gate, same floor as the signals x asset_returns grid:
            # the return series is what the Sharpe/DSR checks and the cost
            # reconciliation judge, and a series that is finite on a
            # handful of grid dates (5 shared dates, or a 750-date index
            # that is NaN everywhere but 5 bars) would validate and then
            # "certify" a Sharpe over those bars while sample_size counts
            # the 750-date calendar. An honest series that starts late but
            # is dense over its own span (the back 60% of the grid) clears
            # 30 with a wide margin.
            sr_on_grid = self.strategy_returns.reindex(common_idx)
            sr_finite = np.isfinite(
                sr_on_grid.to_numpy(dtype="float64", na_value=np.nan))
            n_sr_finite = int(sr_finite.sum())
            if n_sr_finite < _MIN_OVERLAP:
                raise MisalignedInputError(
                    f"artifacts.strategy_returns "
                    f"({_describe_index(self.strategy_returns.index)}) "
                    f"carries only {n_sr_finite} finite value(s) on the "
                    f"{len(sr_overlap)} date(s) it shares with the "
                    f"{len(common_idx)}-date signals x asset_returns grid; "
                    f"need >= {_MIN_OVERLAP}. aligned() would reduce it to "
                    f"an all-but-NaN series and the cost/performance checks "
                    f"would judge a Sharpe over {n_sr_finite} bars as if it "
                    f"were the book. Reindex strategy_returns onto the "
                    f"backtest calendar (same timezone/calendar as signals), "
                    f"or trim signals/asset_returns to the span the book "
                    f"actually traded."
                )
            # A finite-count floor alone is evadable: a reporter can retain
            # 30+ selected dates (or remove those dates from the Series index
            # altogether), hide every losing period between them, and still
            # have the Sharpe/cost checks judge the surviving observations.
            # Require one finite return on every backtest-calendar date from
            # the first through the last finite reported return.  Leading and
            # trailing NaN runs remain legal warmup / late-start / early-stop
            # truncation because they sit outside that observed span.
            finite_pos = np.flatnonzero(sr_finite)
            first_pos, last_pos = int(finite_pos[0]), int(finite_pos[-1])
            span_finite = sr_finite[first_pos:last_pos + 1]
            n_interior_missing = int((~span_finite).sum())
            if n_interior_missing:
                span_index = common_idx[first_pos:last_pos + 1]
                examples = list(span_index[~span_finite][:3])
                raise MisalignedInputError(
                    f"artifacts.strategy_returns is not dense over its "
                    f"observed span {common_idx[first_pos]} .. "
                    f"{common_idx[last_pos]}: {n_interior_missing} of "
                    f"{len(span_finite)} signals x asset_returns calendar "
                    f"date(s) inside the first-to-last finite return are "
                    f"missing or NaN (e.g. {examples}). Interior gaps can "
                    f"hide losing returns and make Sharpe/cost verification "
                    f"judge selected dates. Report one finite net return per "
                    f"calendar date throughout the strategy's live span. A "
                    f"legitimate warmup, late start, or early stop may remain "
                    f"as leading/trailing NaN dates outside that span; do not "
                    f"drop or blank dates inside it."
                )

    # ------------------------------------------------------------------
    @property
    def common_index(self) -> pd.DatetimeIndex:
        return self.signals.index.intersection(self.asset_returns.index)

    @property
    def common_assets(self) -> pd.Index:
        return self.signals.columns.intersection(self.asset_returns.columns)

    def aligned(self) -> "BacktestArtifacts":
        """Return a copy trimmed to the common (dates x assets) grid of
        signals/asset_returns. positions get NaN->0 (missing = flat); universe
        NaN->False. ``signal_input`` is copied without trimming or dtype
        conversion: raw pipeline input may start earlier for warmup, and
        caller-side mutation must not change later dynamic probes."""
        idx, cols = self.common_index, self.common_assets
        pos = self.positions
        if pos is not None:
            # Nullable booleans represent legitimate 0/1 weights, but their
            # missing cells cannot accept a numeric fill value. Convert to
            # float before filling either existing or reindex-created gaps.
            pos = _denullify_frame(pos.reindex(index=idx, columns=cols)).fillna(0.0)
        uni = self.universe
        if uni is not None:
            # Fill at the numpy level: pandas' fillna/where on object-dtype
            # frames (or the all-NaN columns reindex creates for assets the
            # universe never listed) trip the silent-downcasting
            # FutureWarning; np.where fills identically without it.
            uni = uni.reindex(index=idx, columns=cols)
            vals = uni.to_numpy(dtype=object)
            uni = pd.DataFrame(
                np.where(pd.isna(vals), False, vals).astype(bool),
                index=uni.index, columns=uni.columns)
        prices = self.prices
        if prices is not None:
            prices = _denullify_frame(prices.reindex(index=idx, columns=cols))
        sr = self.strategy_returns
        if sr is not None:
            sr = _denullify_series(sr.reindex(idx))
        # Preserve raw values, dtypes and warmup history, but do not share
        # the caller's mutable frame with callbacks that may close over it.
        signal_input = (None if self.signal_input is None
                        else self.signal_input.copy(deep=True))
        return dataclasses.replace(
            self,
            signals=_denullify_frame(self.signals.loc[idx, cols]),
            asset_returns=_denullify_frame(self.asset_returns.loc[idx, cols]),
            positions=pos, universe=uni, prices=prices, strategy_returns=sr,
            signal_input=signal_input,
        )
