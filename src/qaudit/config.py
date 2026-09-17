"""User-configurable thresholds and resource limits for an audit.

Defaults target daily-bar cross-sectional equity alphas (20-500 names).
Checks also contain fixed statistical floors and scope assumptions; changing
these fields alone does not validate a different frequency or asset class.
Override relevant fields for the intended research setting:

    audit(artifacts, config=AuditConfig(sharpe_warn=4.0, n_trials=345))
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import InputValidationError

# Hard ceiling on every per-run count that multiplies calls into the user's
# ``backtest_func`` / ``signal_func`` (placebo, date-shuffle, relabeling,
# date-shift and rolling-window re-runs). Nothing statistical happens above
# ~1e3 re-runs per null (the empirical p resolution is already 1e-3), and
# an unbounded count is a resource attack on a serving audit: n_placebo =
# 10**9 would run the callable for days. The cap is a plausibility bound,
# not a tuning knob; wall-clock timeouts on the callable itself stay the
# caller's responsibility.
_MAX_DYNAMIC_RUNS = 1_000

# Bounds for the two other list/loop multipliers.  These are deliberately
# generous relative to the daily-equity defaults (6 horizons, 3 sweep
# points) but prevent a malformed service request from turning a linear
# diagnostic into an effectively unbounded job.
_MAX_FORWARD_HORIZON = 252
_MAX_COST_SWEEP_POINTS = 256
_MAX_EXTRA_EXPANDED_NODES = 10_000
# Rolling windows are passed to pandas' C-backed window machinery.  Python
# accepts arbitrarily large integers, but pandas cannot convert values above
# its platform-sized integer range (and raises OverflowError far away from the
# offending config field).  One million bars is already far beyond a
# plausible IC/volatility diagnostic window while leaving ample room for
# intraday data.
_MAX_ROLLING_WINDOW = 1_000_000
# A one-way cost of one million basis points is 10,000x notional.  The ceiling
# is intentionally extravagant; its job is to reject overflow payloads before
# cost-sweep arithmetic/formatting, not to prescribe a realistic execution
# model.
_MAX_COST_BPS = 1_000_000.0


@dataclass
class AuditConfig:
    # -- general -----------------------------------------------------------
    seed: int = 0                       # rng seed for all randomized probes
    min_periods: int = 120              # below this, statistical checks warn/skip
    n_trials: int | None = None         # total configurations tried in this research
                                        # program (for the deflated Sharpe check);
                                        # None -> that check SKIPs and tells you to count

    # -- lookahead ----------------------------------------------------------
    future_return_embed_ic: float = 0.60  # mean |per-date IC| vs next-period return above
                                          # this -> signal almost certainly contains the
                                          # future return itself
    predictive_ic_warn: float = 0.08      # mean daily IC above this is rare for real alphas
    predictive_ic_fail: float = 0.15
    ic_decay_collapse_ratio: float = 0.15 # IC(h=2)/IC(h=1) below this while IC(h=1) is
                                          # extreme -> info concentrated at exactly h=1
    bleed_partial_fail: float = 0.10      # mean partial rank corr of positions vs the
                                          # same-bar signal innovation (honest books ~0;
                                          # a 10%-of-book bleed measures ~+0.5)
    bleed_tstat: float = 4.0              # Newey-West t required alongside the mean
    fwd_vol_window: int = 10              # bars of forward/trailing realized vol compared
    deferred_ic_max_horizon: int = 6      # profile per-horizon IC out to t+this many bars
                                          # for the deferred-window leak signature

    # -- target leakage -----------------------------------------------------
    leak_median_ic_warn: float = 0.30
    leak_median_ic_fail: float = 0.50
    leak_perfect_ic: float = 0.90         # a single date with |IC| >= this is a smoking gun
    leak_perfect_date_frac: float = 0.01  # ...if more than this fraction of dates have one
    leak_perfect_min_names: int = 8       # perfect-rank dates only count when at least this
                                          # many names are jointly observed that date - pure
                                          # noise gives |IC|>=0.9 on ~1.7% of 5-name dates
                                          # (above the 1% fail bar) vs 0.4% at 8 names
    leak_identity_corr: float = 0.95      # per-date pearson corr of standardized signal vs
                                          # target above this -> signal is an affine copy
    leak_outlier_ic: float = 0.40         # per-date |IC| at/above this counts as an outlier
                                          # date for the intermittent-leak counter

    # -- train/test contamination --------------------------------------------
    min_embargo_periods: int | None = None  # None -> defaults to artifacts.label_horizon

    # -- survivorship ---------------------------------------------------------
    full_history_frac_warn: float = 0.95  # fraction of assets alive for the whole sample
    min_assets_survivorship: int = 10     # need at least this many names to judge
    min_exit_rate_per_year: float = 0.01  # universe exits below this rate per member-year
                                          # (distinct plausibly-exiting assets / (live-member
                                          # bars / periods_per_year)) look like token
                                          # attrition on a survivor list (real US equity
                                          # universes lose ~4-8%/year)

    # -- performance / overfitting ---------------------------------------------
    sharpe_warn: float = 3.0              # annualized, net; daily equity alphas above this
    sharpe_fail: float = 5.0              # are presumptively leaky
    dsr_warn: float = 0.95                # deflated-Sharpe prob below this -> cannot rule
    dsr_fail: float = 0.50                # out selection luck; below 0.5 -> likely luck
    ic_rolling_window: int = 63           # periods per rolling-IC window
    ic_sign_consistency_warn: float = 0.60  # fraction of windows matching full-sample sign
    ic_regime_share_warn: float = 0.60    # one calendar year carrying more than this share
                                          # of the summed positive IC -> regime-concentrated

    # -- costs / turnover -------------------------------------------------------
    turnover_daily_warn: float = 0.25     # mean one-sided turnover per period (fraction of book);
                                          # calibrated at daily bars - the check annualizes both
                                          # sides, so non-daily books are judged per-year
    turnover_daily_fail: float = 1.00
    cost_sensitivity_bps: tuple = (5.0, 10.0, 25.0)
    gross_match_atol: float = 1e-12       # net returns matching gross this closely -> no costs
    min_turnover_for_cost_check: float = 0.01

    # -- dynamic probes ----------------------------------------------------------
    n_placebo: int = 100                  # random-signal pipeline re-runs
    n_shuffle: int = 100                  # shuffled-label pipeline re-runs
    placebo_null_sharpe_fail: float = 0.50  # mean placebo Sharpe above this (with t>3)
                                            # -> the pipeline manufactures performance
    placebo_percentile_warn: float = 0.95   # real Sharpe below this placebo quantile
                                            # -> indistinguishable from noise
    shuffle_pvalue_warn: float = 0.05       # empirical p of real Sharpe vs shuffled nulls
    shuffle_leak_ratio: float = 0.50        # mean shuffled Sharpe above this fraction of the
                                            # real one -> mechanical leak in the pipeline
    shuffle_block_len: int | None = None    # date-shuffle null: length (bars) of the
                                            # contiguous row blocks whose order is
                                            # permuted (within-block order kept), so
                                            # serially dependent returns keep their
                                            # short-range structure under the null.
                                            # None -> auto: 1 (a plain row
                                            # permutation) when both the panel's
                                            # median absolute lag-1 return and
                                            # squared-return autocorrelations are
                                            # negligible, else a dependence-derived
                                            # length (probes_null._auto_block_len)
    date_shift_max: int = 3                 # probe signal shifts of +/- 1..this many periods
    shift_delay_collapse_ratio: float = 0.20  # Sharpe at +1 delay below this fraction of
                                              # baseline -> PnL lives at exactly one bar
    truncation_sample_dates: int = 12       # dates re-computed in the rolling-window probe
                                            # (sampled across the whole post-warmup range)
    truncation_rtol: float = 1e-6
    truncation_atol: float = 1e-8

    # escape hatch for experimental checks; never read by built-in ones
    extra: dict[str, Any] = field(default_factory=dict)

    # -- validation --------------------------------------------------------
    _COUNTS = ("min_periods", "fwd_vol_window", "deferred_ic_max_horizon",
               "ic_rolling_window", "min_assets_survivorship", "n_placebo",
               "n_shuffle", "date_shift_max", "truncation_sample_dates",
               "leak_perfect_min_names")
    _POSITIVE = ("bleed_tstat", "sharpe_warn",
                 "sharpe_fail", "turnover_daily_warn", "turnover_daily_fail",
                 "placebo_null_sharpe_fail", "shuffle_leak_ratio")
    # Correlation-valued thresholds live here, not in _POSITIVE: the measured
    # statistics (|Spearman IC|, partial rank corr) are bounded by 1, so a
    # threshold > 1 makes the WARN/FAIL branch unreachable - a typo'd
    # override would silently disarm the detector instead of raising.
    _UNIT_INTERVAL = ("future_return_embed_ic", "leak_perfect_ic",
                      "leak_identity_corr", "leak_outlier_ic",
                      "leak_perfect_date_frac", "full_history_frac_warn",
                      "ic_sign_consistency_warn", "ic_regime_share_warn",
                      "dsr_warn", "dsr_fail", "ic_decay_collapse_ratio",
                      "shift_delay_collapse_ratio", "placebo_percentile_warn",
                      "shuffle_pvalue_warn",
                      "predictive_ic_warn", "predictive_ic_fail",
                      "leak_median_ic_warn", "leak_median_ic_fail",
                      "bleed_partial_fail")
    _NON_NEGATIVE = ("min_exit_rate_per_year", "gross_match_atol",
                     "min_turnover_for_cost_check", "truncation_rtol",
                     "truncation_atol")
    # Counts that each multiply calls into a user callable; capped at
    # _MAX_DYNAMIC_RUNS (date_shift_max runs 2k+1 pipeline calls,
    # truncation_sample_dates re-computes the signal that many times).
    _RUN_COUNTS = ("n_placebo", "n_shuffle", "date_shift_max",
                   "truncation_sample_dates")
    _WARN_BEFORE_FAIL = (("sharpe_warn", "sharpe_fail"),
                         ("predictive_ic_warn", "predictive_ic_fail"),
                         ("leak_median_ic_warn", "leak_median_ic_fail"),
                         ("turnover_daily_warn", "turnover_daily_fail"))

    def __post_init__(self) -> None:
        """Sanity-check every threshold at construction: a config holding
        ``sharpe_warn=-5`` or ``n_placebo=-1`` would not crash any check, it
        would silently mis-audit - so nonsensical values raise here. The
        same validation re-runs on every later field assignment, so a config
        cannot be mutated into an invalid state after construction either."""
        self._validate()
        object.__setattr__(self, "_validation_active", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # Accept loader-friendly lists but store the sweep immutably. Without
        # normalization, ``cfg.cost_sensitivity_bps.extend(...)`` bypasses
        # __setattr__ and its resource cap.
        if name == "cost_sensitivity_bps" and isinstance(value, list):
            value = tuple(value)
        if not getattr(self, "_validation_active", False):
            object.__setattr__(self, name, value)
            return
        old = getattr(self, name, None)
        had = hasattr(self, name)
        object.__setattr__(self, name, value)
        try:
            self._validate()
        except Exception:
            # leave the config in its last valid state
            if had:
                object.__setattr__(self, name, old)
            else:
                object.__delattr__(self, name)
            raise

    def validate(self) -> None:
        """Revalidate current state, including nested mutable ``extra``.

        Normal assignment is validated eagerly; :func:`qaudit.audit` calls
        this again at its trust boundary so in-place nested mutations or
        deliberate ``object.__setattr__`` bypasses cannot disarm a check.
        """
        self._validate()

    def _validate(self) -> None:
        def _bad(name: str, why: str) -> None:
            raise InputValidationError(
                f"AuditConfig.{name} = {getattr(self, name)!r} is invalid: "
                f"{why}.")

        def _is_real(v: Any) -> bool:
            # bool subclasses int, and True would pass every ">= 1" gate.
            # inf also passes ">= 1"/">= 0" (NaN fails them), then crashes
            # whole modules downstream as "OverflowError: cannot convert
            # float infinity to integer" naming no field - so only finite
            # values count. Python ints are finite by construction, but the
            # consumers convert thresholds/sweep points to float (and format
            # them with floating-point format codes), so an arbitrary-precision
            # int that cannot become a finite float is not operationally real.
            if (isinstance(v, bool)
                    or not isinstance(v,
                                      (int, float, np.integer, np.floating))):
                return False
            try:
                return bool(np.isfinite(float(v)))
            except (OverflowError, TypeError, ValueError):
                return False

        def _is_integral(v: Any) -> bool:
            # Counts feed int() truncation at every consumer (config.py
            # embargo_periods, probes_null, probes_shift, performance), so a
            # fractional value would not error - it would silently shrink:
            # min_embargo_periods=0.9 -> int() -> 0 would disarm the embargo
            # check entirely. Integral floats (100.0, np.float64) stay
            # accepted - YAML/JSON loaders emit them.
            return (isinstance(v, (int, np.integer))
                    or (isinstance(v, (float, np.floating))
                        and float(v).is_integer()))

        if not isinstance(self.extra, dict):
            _bad("extra", "must be a dict with string keys")

        extra_nodes = 0

        def _validate_extra(value: Any, path: str, depth: int,
                            stack: set[int]) -> None:
            nonlocal extra_nodes
            extra_nodes += 1
            if extra_nodes > _MAX_EXTRA_EXPANDED_NODES:
                _bad("extra", f"expanded value exceeds "
                              f"{_MAX_EXTRA_EXPANDED_NODES} nodes at {path}; "
                              f"large shared-container graphs can amplify "
                              f"during provenance serialization")
            if depth > 32:
                _bad("extra", f"nesting exceeds 32 containers at {path}; "
                              f"a bounded acyclic config is required for "
                              f"deterministic provenance")
            # ``extra`` is part of the hashed deployment provenance.  An
            # arbitrary object's repr commonly contains a process-specific
            # memory address, so accepting it makes equal configurations
            # hash differently after restart. Keep this escape hatch broad
            # enough for normal JSON/temporal/numpy metadata, but explicit.
            if value is None or isinstance(value, (str, bool)):
                return
            if isinstance(value, (np.datetime64, np.timedelta64,
                                  _dt.datetime, _dt.date, _dt.timedelta)):
                return
            if isinstance(value, (int, np.integer)):
                return
            if isinstance(value, (float, np.floating)):
                if not bool(np.isfinite(value)):
                    _bad("extra", f"contains non-finite numeric value "
                                  f"{value!r} at {path}; NaN/inf collapse "
                                  f"during JSON provenance serialization")
                return
            if not isinstance(value, (dict, list, tuple, set, frozenset)):
                _bad("extra", f"value at {path} has unsupported type "
                              f"{type(value).__module__}."
                              f"{type(value).__qualname__}; use only "
                              f"strings, booleans, finite numbers, temporal "
                              f"scalars, None, and nested dict/list/tuple/"
                              f"set containers so config_hash is stable")
            ident = id(value)
            if ident in stack:
                _bad("extra", f"contains a reference cycle at {path}; "
                              f"cyclic values cannot be serialized into "
                              f"deterministic provenance")
            stack.add(ident)
            try:
                if isinstance(value, dict):
                    for key, child in value.items():
                        if not isinstance(key, str):
                            _bad("extra", f"key {key!r} at {path} has type "
                                          f"{type(key).__name__}; all nested "
                                          f"mapping keys must be strings so "
                                          f"JSON provenance cannot collide "
                                          f"(for example 1 vs '1')")
                        _validate_extra(child, f"{path}.{key}", depth + 1,
                                        stack)
                else:
                    for i, child in enumerate(value):
                        _validate_extra(child, f"{path}[{i}]", depth + 1,
                                        stack)
            finally:
                stack.remove(ident)

        _validate_extra(self.extra, "extra", 0, set())

        if (isinstance(self.seed, bool)
                or not isinstance(self.seed, (int, np.integer))
                or self.seed < 0):
            _bad("seed", "must be a non-negative int (it is fed to "
                         "numpy.random.default_rng by every randomized probe)")

        # Type-gate every numeric field before comparing: config-loader
        # garbage (a string "0.1", None, a bool) would otherwise surface as
        # a raw TypeError from the comparison, naming no field and no fix.
        for name in (self._COUNTS + self._POSITIVE + self._UNIT_INTERVAL
                     + self._NON_NEGATIVE):
            if not _is_real(getattr(self, name)):
                _bad(name, f"must be a finite real number, not "
                           f"{type(getattr(self, name)).__name__} (thresholds "
                           f"are compared numerically against measured "
                           f"statistics, and counts feed int() conversions)")

        for name in self._COUNTS:
            if not _is_integral(getattr(self, name)):
                _bad(name, "must be a whole number - every consumer "
                           "int()-truncates, so a fractional count would "
                           "silently shrink (e.g. 50.7 placebo runs -> 50, "
                           "0.9 -> 0)")
            if not getattr(self, name) >= 1:
                _bad(name, "must be a count >= 1")
        # The deferred-spike profile compares horizon 1 against horizons
        # 2..max; max=1 leaves nothing to compare (an empty max() would
        # crash the lookahead module).
        if not self.deferred_ic_max_horizon >= 2:
            _bad("deferred_ic_max_horizon",
                 "must be >= 2 (the deferred-spike signature compares "
                 "horizon 1 against horizons 2..max)")
        if self.deferred_ic_max_horizon > _MAX_FORWARD_HORIZON:
            _bad("deferred_ic_max_horizon",
                 f"must be <= {_MAX_FORWARD_HORIZON} - each additional "
                 f"horizon performs another full-panel forward-return/IC "
                 f"calculation; larger values are a resource attack, not a "
                 f"plausible leak window")
        for name in ("fwd_vol_window", "ic_rolling_window"):
            if getattr(self, name) > _MAX_ROLLING_WINDOW:
                _bad(name, f"must be <= {_MAX_ROLLING_WINDOW:,} bars - larger "
                           f"values overflow pandas' rolling-window integer "
                           f"conversion and are not a plausible diagnostic "
                           f"window")
        for name in self._POSITIVE:
            if not getattr(self, name) > 0:
                _bad(name, "must be > 0")
        for name in self._UNIT_INTERVAL:
            if not 0 < getattr(self, name) <= 1:
                _bad(name, "must lie in (0, 1]")
        for name in self._NON_NEGATIVE:
            if not getattr(self, name) >= 0:
                _bad(name, "must be >= 0")
        for warn, fail in self._WARN_BEFORE_FAIL:
            if getattr(self, warn) > getattr(self, fail):
                _bad(warn, f"must be <= {fail} ({getattr(self, fail)!r})")
        if self.dsr_fail > self.dsr_warn:
            _bad("dsr_fail", f"must be <= dsr_warn ({self.dsr_warn!r}); DSR "
                             f"fails below the fail bar, warns below the "
                             f"warn bar")
        if self.n_trials is not None and (
                not _is_real(self.n_trials)
                or not _is_integral(self.n_trials)
                or not self.n_trials >= 1):
            _bad("n_trials", "must be None (uncounted -> the deflated-Sharpe "
                             "check SKIPs) or a whole-number count >= 1")
        if self.min_embargo_periods is not None and (
                not _is_real(self.min_embargo_periods)
                or not _is_integral(self.min_embargo_periods)
                or not self.min_embargo_periods >= 0):
            _bad("min_embargo_periods", "must be None (defaults to "
                                        "label_horizon) or a whole number "
                                        ">= 0 of periods - int(0.9) would "
                                        "silently disarm the embargo to 0")
        for name in self._RUN_COUNTS:
            if getattr(self, name) > _MAX_DYNAMIC_RUNS:
                _bad(name, f"must be <= {_MAX_DYNAMIC_RUNS} - every unit is "
                           f"at least one full re-run of a user callable, so "
                           f"an unbounded count is a wall-clock/resource "
                           f"attack, not a statistical setting (the "
                           f"empirical p resolution is already 1e-3 at "
                           f"1000 re-runs)")
        if self.shuffle_block_len is not None and (
                not _is_real(self.shuffle_block_len)
                or not _is_integral(self.shuffle_block_len)
                or not self.shuffle_block_len >= 1):
            _bad("shuffle_block_len", "must be None (auto: 1 on serially "
                                      "independent returns, else a "
                                      "dependence-derived block length) or "
                                      "a whole number of bars >= 1")
        csb = self.cost_sensitivity_bps
        # str is iterable, so an explicit reject - "5,10" must not be swept
        # character by character; a scalar 25.0 would raise a bare
        # "'float' object is not iterable" naming no field.
        if (isinstance(csb, str) or not isinstance(csb, (list, tuple))
                or any(not _is_real(b) for b in csb)):
            _bad("cost_sensitivity_bps",
                 "must be a list/tuple of cost sweep points in bps, each a "
                 "real number >= 0, e.g. (5.0, 10.0, 25.0)")
        if not csb:
            _bad("cost_sensitivity_bps", "must contain at least one sweep "
                 "point - an empty grid cannot test cost fragility")
        if any(not b >= 0 for b in csb):
            _bad("cost_sensitivity_bps", "every sweep point must be >= 0")
        if any(float(b) > _MAX_COST_BPS for b in csb):
            _bad("cost_sensitivity_bps", f"every sweep point must be <= "
                 f"{_MAX_COST_BPS:g} bps - larger costs are economically "
                 f"meaningless and can overflow the sensitivity arithmetic")
        if len(csb) > _MAX_COST_SWEEP_POINTS:
            _bad("cost_sensitivity_bps",
                 f"must contain at most {_MAX_COST_SWEEP_POINTS} points - "
                 f"each point recomputes the strategy Sharpe; use a bounded "
                 f"grid rather than an untrusted arbitrary-length list")

    def embargo_periods(self, label_horizon: int) -> int:
        if self.min_embargo_periods is not None:
            return int(self.min_embargo_periods)
        return int(label_horizon)
