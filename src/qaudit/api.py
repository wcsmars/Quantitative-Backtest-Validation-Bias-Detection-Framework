"""Public entry point: run every registered check against one backtest."""
from __future__ import annotations

import importlib
import struct
import time
from typing import Callable

import numpy as np
import pandas as pd

from .config import AuditConfig
from .errors import CheckRuntimeError, InputValidationError
from .inputs import _MIN_OVERLAP, BacktestArtifacts
from .report import AuditReport, _result_contract_problem
from .types import CheckResult, Severity, Status, errored

# One module per audit family. Each exposes
#   run(artifacts, config, *, signal_func=None, backtest_func=None) -> list[CheckResult]
# and never mutates the artifacts it receives.
CHECK_MODULES: tuple[str, ...] = (
    "qaudit.checks.lookahead",
    "qaudit.checks.leakage",
    "qaudit.checks.contamination",
    "qaudit.checks.survivorship",
    "qaudit.checks.performance",
    "qaudit.checks.costs",
    "qaudit.dynamic.probes_null",
    "qaudit.dynamic.probes_shift",
)

# Every check id each module can emit (SKIP placeholders included). Used to
# skip whole modules before dispatch when include/exclude would drop all of
# their results anyway - e.g. include=["costs"] must not pay for hundreds of
# backtest_func re-runs in the dynamic probes. The registry drift guard in
# tests/ keeps this map in sync with the check modules.
MODULE_CHECK_IDS: dict[str, tuple[str, ...]] = {
    "qaudit.checks.lookahead": (
        "lookahead.position_signal_alignment",
        "lookahead.embedded_future_return",
        "lookahead.ic_decay_signature",
        "lookahead.same_bar_bleed",
        "lookahead.future_vol_sizing",
        "lookahead.deferred_ic_spike",
        "lookahead.smeared_forward_ic",
        "lookahead.same_bar_return_loading"),
    "qaudit.checks.leakage": (
        "leakage.target_correlation",
        "leakage.perfect_rank_dates",
        "leakage.signal_target_identity",
        "leakage.ic_outlier_dates"),
    "qaudit.checks.contamination": (
        "contamination.split_declared",
        "contamination.split_overlap",
        "contamination.insufficient_embargo",
        "contamination.test_before_train",
        "contamination.split_coverage"),
    "qaudit.checks.survivorship": (
        "survivorship.trading_outside_universe",
        "survivorship.positions_on_missing_returns",
        "survivorship.no_exits",
        "survivorship.full_history_universe"),
    "qaudit.checks.performance": (
        "performance.suspicious_sharpe",
        "performance.deflated_sharpe",
        "performance.suspicious_ic",
        "performance.ic_stability",
        "performance.ic_regime_concentration",
        "performance.sample_size"),
    "qaudit.checks.costs": (
        "costs.missing_transaction_costs",
        "costs.no_cost_declaration",
        "costs.turnover_unrealistic",
        "costs.cost_sensitivity",
        "costs.price_return_consistency"),
    "qaudit.dynamic.probes_null": (
        "dynamic.placebo_pipeline_bias",
        "dynamic.placebo_percentile",
        "dynamic.shuffled_labels",
        "dynamic.probe_health"),
    "qaudit.dynamic.probes_shift": (
        "dynamic.date_shift",
        "dynamic.signal_reproducibility",
        "dynamic.rolling_window_integrity",
        "dynamic.signal_input_sensitivity"),
}

# Checks a conforming module may omit. ``dynamic.probe_health`` is emitted
# only when one of the null distributions is unhealthy; every other null
# result is part of the module's fixed contract.
_OPTIONAL_MODULE_CHECK_IDS: dict[str, frozenset[str]] = {
    "qaudit.dynamic.probes_null": frozenset({"dynamic.probe_health"}),
}

# With no declared train/test split, contamination deliberately collapses
# the whole family to one SKIP placeholder. This is the sole supported
# non-empty partial module output; the placeholder makes a strict family gate
# fail closed while telling the caller which artifacts are missing.
_COLLAPSED_MODULE_OUTPUTS: dict[str, tuple[str, Status]] = {
    "qaudit.checks.contamination":
        ("contamination.split_declared", Status.SKIP),
}


# audit.coverage advisory floor: WARN when the signals x asset_returns
# intersection keeps less than this fraction of either required panel's own
# dates or columns - the verdict then judged a sliver of what the caller
# handed in (vendor join accidents, mismatched calendars/tickers).
# Calibration: an honest warmup trim or a late-start join loses well under
# half of one axis, while a join accident that keeps 1 shared asset of 20
# (5%) or 30 shared dates of 500 (6%) certifies a sliver. Advisory WARN,
# not a hard gate: a broad-universe signal panel joined to a narrower
# returns panel is a legitimate layout that must keep auditing.
_COVERAGE_MIN_KEPT_FRAC = 0.50
# Every built-in cross-sectional IC primitive needs at least five jointly
# finite names on a date. If even the widest date falls below that floor,
# the calendar/label intersection can look broad in metadata while every
# cross-sectional detector is structurally unable to judge the book.
_MIN_CROSS_SECTIONAL_ASSETS = 5


def _matches(check_id: str, pattern: str) -> bool:
    """include/require pattern semantics: prefix or substring of the id."""
    return check_id.startswith(pattern) or pattern in check_id


def _snapshot_config(config: AuditConfig) -> AuditConfig:
    """Validated independent config copy used for one audit/module run.

    User callbacks execute in-process and may close over the caller's mutable
    config. Without an entry snapshot, a callback invoked by a late dynamic
    probe can change thresholds after earlier checks ran, making provenance
    claim settings that did not produce those verdicts. A fresh per-module
    copy also prevents an extension module from changing later families.
    """
    import copy

    if not isinstance(config, AuditConfig):
        raise InputValidationError(
            f"audit(config={config!r}) is invalid: config must be an "
            f"AuditConfig instance, not {type(config).__name__}")
    config.validate()
    snapshot = copy.deepcopy(config)
    snapshot.validate()
    return snapshot


def _validate_module_results(mod_name: str,
                             out: object) -> list[CheckResult]:
    """Validate one check module's complete output contract.

    A merely non-empty list is insufficient: if a regression silently drops
    seven of eight lookahead results, a broad deployment gate otherwise sees
    the one surviving PASS and certifies the family. The registry is already
    the dispatch/filter contract, so enforce it at the same boundary.
    """
    if isinstance(out, (str, bytes)) or not hasattr(out, "__iter__"):
        raise TypeError(
            f"{mod_name}.run() must return a list of CheckResult, "
            f"got {type(out).__name__}")
    items = list(out)  # type: ignore[arg-type]
    bad_types = [type(x).__name__ for x in items
                 if not isinstance(x, CheckResult)]
    if bad_types:
        raise TypeError(
            f"{mod_name}.run() returned {len(bad_types)} non-CheckResult "
            f"item(s) of {len(items)} (types: {bad_types[:3]}) - every item "
            f"must be a qaudit.types.CheckResult")
    malformed = [(r.check if isinstance(r.check, str) else repr(r.check),
                  problem)
                 for r in items
                 if (problem := _result_contract_problem(r)) is not None]
    if malformed:
        raise TypeError(
            f"{mod_name}.run() emitted malformed CheckResult value(s) "
            f"{malformed[:3]!r} - status/severity must be enum members, "
            f"message/remediation/check strings, details a dict, and "
            f"reserved verdict flags actual booleans")
    if not items:
        raise ValueError(
            f"{mod_name}.run() emitted zero results - nothing was audited by "
            f"this module (a conforming module SKIPs each check it cannot run "
            f"instead of returning [])")

    actual = [r.check for r in items]
    duplicates = sorted({cid for cid in actual if actual.count(cid) > 1})
    if duplicates:
        raise ValueError(
            f"{mod_name}.run() emitted duplicate check id(s) {duplicates!r} - "
            f"one result per registered check is required")

    expected = MODULE_CHECK_IDS.get(mod_name)
    if expected is None:  # third-party extension: type/non-empty contract only
        return items
    actual_set = set(actual)
    expected_set = set(expected)
    unknown = sorted(actual_set - expected_set)
    if unknown:
        raise ValueError(
            f"{mod_name}.run() emitted unregistered check id(s) {unknown!r}; "
            f"registered ids are {list(expected)!r}")

    collapsed = _COLLAPSED_MODULE_OUTPUTS.get(mod_name)
    if collapsed is not None and len(items) == 1:
        placeholder_id, placeholder_status = collapsed
        if (items[0].check == placeholder_id
                and items[0].status is placeholder_status):
            return items

    required = expected_set - set(_OPTIONAL_MODULE_CHECK_IDS.get(
        mod_name, frozenset()))
    missing = sorted(required - actual_set)
    if missing:
        raise ValueError(
            f"{mod_name}.run() omitted {len(missing)} registered result(s) "
            f"{missing!r} - a partial family output is not auditable; emit a "
            f"PASS/WARN/FAIL/SKIP result for every registered check")
    return items


def _select_modules(include: list[str] | None,
                    exclude: list[str] | None) -> list[str]:
    """Modules that can emit at least one check surviving include/exclude.

    A module absent from the map always runs. The dispatch-all fallback for
    an include pattern matching no *known* id exists only for registries
    that carry a module MODULE_CHECK_IDS does not list (a third-party
    add-on); with every registered module listed, such a pattern is a typo,
    and dispatching all modules for it would waste hundreds of pipeline
    re-runs (dynamic probes) whose results the filters then throw away -
    the typo is reported instead by the per-pattern
    ``audit.filter_pattern_matched_nothing`` ERROR in audit().
    """
    def _kept(cid: str) -> bool:
        if include and not any(_matches(cid, p) for p in include):
            return False
        if exclude and any(cid.startswith(p) for p in exclude):
            return False
        return True

    if include and any(m not in MODULE_CHECK_IDS for m in CHECK_MODULES):
        known = [cid for ids in MODULE_CHECK_IDS.values() for cid in ids]
        for p in include:
            if not any(_matches(c, p) for c in known):
                return list(CHECK_MODULES)
    return [m for m in CHECK_MODULES
            if m not in MODULE_CHECK_IDS
            or any(_kept(cid) for cid in MODULE_CHECK_IDS[m])]

# Dynamic-probe function contracts (documented once, enforced by the probes):
#   signal_func(signal_input: pd.DataFrame) -> pd.DataFrame
#       Recomputes the signal panel from raw input data. Must be causal: the
#       row for date t may use input rows <= t only. The truncation probe
#       verifies exactly this.
#   backtest_func(signals: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series
#       Re-runs the full backtest pipeline (position construction, lags,
#       costs) on the given inputs and returns net strategy returns. It must
#       apply the same execution lag as the production pipeline.


def audit(artifacts: BacktestArtifacts,
          config: AuditConfig | None = None,
          *,
          signal_func: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
          backtest_func: Callable[[pd.DataFrame, pd.DataFrame], pd.Series] | None = None,
          include: list[str] | None = None,
          exclude: list[str] | None = None,
          strict: bool = False) -> AuditReport:
    """Audit one backtest and return an :class:`AuditReport`.

    Parameters
    ----------
    artifacts : the data contract (validated here; see qaudit.inputs docstring
        for the timing convention).
    config : thresholds; defaults to :class:`AuditConfig()`.
    signal_func, backtest_func : optional pipeline callables that unlock the
        dynamic probes (placebo, shuffled-label, date-shift, rolling-window
        truncation). Checks that need a missing callable SKIP with a message.
    include / exclude : filter check families or individual ids,
        e.g. include=["lookahead", "dynamic"] or exclude=["survivorship"].
        An include pattern matches as a prefix or a substring of the check
        id (include=["sharpe"] selects both Sharpe checks); exclude matches
        by prefix only. Modules whose entire output would be filtered away
        are skipped before dispatch, so include=["costs"] never re-runs the
        pipeline callables for the dynamic probes. Filters matching none of
        the known checks do not yield an empty "CLEAN" report: the report
        carries a synthetic ``audit.filter_matched_nothing`` ERROR naming
        the patterns, which trips ``report.ok`` and ``raise_for_failures``.
        A single include pattern matching zero known checks and zero
        results while its siblings matched (the mixed valid+typo case)
        gets its own ``audit.filter_pattern_matched_nothing`` ERROR; an
        exclude pattern matching nothing gets a low-severity
        ``audit.exclude_pattern_matched_nothing`` WARN (an exclude typo
        fails toward more auditing, not less).
    strict : re-raise a crashing check *module* as :class:`CheckRuntimeError`
        instead of recording a Status.ERROR result for it. A crashing user
        callable (``signal_func``/``backtest_func``) is itself audit
        evidence, so the dynamic probes always record it in-report as a
        Status.ERROR (or SKIP) result and never re-raise - a strict-mode
        report can therefore still contain ERROR results.
    """
    if not isinstance(artifacts, BacktestArtifacts):
        raise InputValidationError(
            f"audit(artifacts=...) is invalid: artifacts must be a "
            f"BacktestArtifacts instance, not {type(artifacts).__name__}")
    if type(strict) is not bool:
        raise InputValidationError(
            f"audit(strict=...) is invalid: strict must be an actual "
            f"bool, not {type(strict).__name__}")
    for name, fn in (("signal_func", signal_func),
                     ("backtest_func", backtest_func)):
        if fn is not None and not callable(fn):
            raise InputValidationError(
                f"audit({name}=...) is invalid: {name} must be callable "
                f"or None, not {type(fn).__name__}")

    # str is iterable, so reject it explicitly: exclude="survivorship" would
    # be matched character by character ('s','u','r',...), silently dropping
    # unrelated families (performance.* matches 'p') and flipping report.ok
    # green; include="costs" would dispatch all 8 modules by per-char
    # substring. Mirrors the AuditConfig.cost_sensitivity_bps guard.
    for fname, pats in (("include", include), ("exclude", exclude)):
        if pats is None:
            continue
        if (isinstance(pats, str) or not isinstance(pats, (list, tuple))
                or any(not isinstance(p, str) for p in pats)
                or any(not p.strip() for p in pats)):
            raise InputValidationError(
                f"audit({fname}={pats!r}) is invalid: {fname} must be a "
                f"list/tuple of non-empty check-id prefixes (each a str), e.g. "
                f"{fname}=[\"survivorship\"] - a bare string would be "
                f"matched character by character and an empty pattern would "
                f"match every check.")

    # Keep dispatch, result filtering and provenance on the same entry
    # policy. A user callback or extension can close over the caller's
    # filter lists and mutate them while checks run; without independent
    # copies, that late mutation can erase a failure already produced.
    include = list(include) if include is not None else None
    exclude = list(exclude) if exclude is not None else None
    config = _snapshot_config(config if config is not None else AuditConfig())
    artifacts.validate()
    # Coverage refers to the panels supplied at entry. Callbacks may close
    # over these caller-owned frames and resize them while probes run.
    input_shapes = {name: getattr(artifacts, name).shape
                    for name in ("signals", "asset_returns")}
    aligned = artifacts.aligned()

    modules = (_select_modules(include, exclude) if (include or exclude)
               else list(CHECK_MODULES))
    # Capture reproducibility state before any user code runs. In
    # particular, a callable's counters, caches or RNG state after many
    # probe calls do not describe the state that started this audit.
    provenance = _provenance(
        aligned, config, modules,
        include=include, exclude=exclude, strict=strict,
        signal_func=signal_func, backtest_func=backtest_func)
    results = []
    timings_s: dict[str, float] = {}
    for mod_name in modules:
        family = mod_name.rsplit(".", 2)
        family_id = ".".join(family[-2:])  # e.g. "checks.lookahead"
        short = family[-1]
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(mod_name)
            # A module receives its own copy; ``config`` below remains the
            # entry snapshot used for meta/provenance regardless of callback
            # or extension-side mutation.
            out = mod.run(aligned, _snapshot_config(config),
                          signal_func=signal_func, backtest_func=backtest_func)
            # Dispatch-time output contract (fail-closed): validate both
            # element types and completeness against MODULE_CHECK_IDS. A
            # non-empty but partial family is just as unaudited as [].
            out = _validate_module_results(mod_name, out)
            results.extend(out)
        except Exception as exc:  # noqa: BLE001 - a broken check must not kill the audit
            if strict:
                raise CheckRuntimeError(family_id, exc) from exc
            # The ERROR id must live in the same family namespace as the
            # module's check ids ("dynamic.*" for the probe modules, not
            # "probes_null.*"), or family-prefix include/exclude patterns
            # would not recognize it.
            ids = MODULE_CHECK_IDS.get(mod_name)
            prefix = ids[0].split(".", 1)[0] if ids else short
            err_id = (f"{short}.<module>" if prefix == short
                      else f"{prefix}.<module:{short}>")
            results.append(errored(err_id, exc))
        finally:
            # time-to-crash is diagnostic too, so record in finally
            timings_s[mod_name] = time.perf_counter() - t0

    # ERROR results are exempt from the result-level filters: a module is
    # only dispatched when at least one of its checks survives filtering,
    # so its crash always concerns a requested check - dropping it would
    # flip report.ok to True on a crashed audit.
    if include:
        # A family-level SKIP placeholder (contamination.split_declared
        # standing in for the whole undeclared-split family) carries a
        # different id than the sub-check it stands in for, so the id filter
        # below would erase it and include=["contamination.split_overlap"]
        # would return an empty ok=True report - a silent fail-open on the
        # one requested check. For every include pattern that matches a
        # known id of some module but no emitted result, keep that module's
        # SKIP results visible: the requested check could not run, and the
        # placeholder's message says exactly what to pass.
        readmit: set[str] = set()
        for p in include:
            if any(_matches(r.check, p) for r in results):
                continue
            for ids in MODULE_CHECK_IDS.values():
                if any(_matches(cid, p) for cid in ids):
                    readmit.update(ids)
        results = [r for r in results
                   if r.status is Status.ERROR
                   or any(_matches(r.check, p) for p in include)
                   or (r.status is Status.SKIP and r.check in readmit)]
    pre_exclude = results
    if exclude:
        results = [r for r in results
                   if r.status is Status.ERROR
                   or not any(r.check.startswith(p) for p in exclude)]

    # A typo'd include (or an exclude covering every family) must not go
    # green: an empty filtered report would read "CLEAN - 0 checks", ok=True,
    # and a CI gate on report.ok / raise_for_failures() would pass with zero
    # checks executed. Emit a loud synthetic ERROR naming the patterns
    # instead of raising - the report object stays usable and every gate
    # (ok, raise_for_failures) trips on it.
    if (include or exclude) and not results:
        known = sorted({cid for ids in MODULE_CHECK_IDS.values()
                        for cid in ids})
        families = sorted({cid.split(".", 1)[0] for cid in known})
        results.append(CheckResult(
            "audit.filter_matched_nothing", Status.ERROR,
            f"include={include!r} / exclude={exclude!r} matched 0 of the "
            f"{len(known)} known checks - nothing was audited, so this "
            f"report proves nothing about the backtest",
            severity=Severity.HIGH,
            details={"include": list(include or []),
                     "exclude": list(exclude or []),
                     "known_check_ids": known},
            remediation=("fix the include/exclude patterns (a typo'd family "
                         "name silently matches nothing); valid family "
                         f"prefixes: {', '.join(families)}")))
    elif include or exclude:
        # Per-pattern accounting (the aggregate guard above fires only when
        # the whole report empties): a typo'd pattern riding alongside a
        # valid one would otherwise be silently ignored -
        # include=["costs", "surviorship"] would report CLEAN and never
        # mention the misspelled family (fail-open for exactly the checks
        # the caller thought they requested). A pattern is dead only when
        # it matches zero known ids and zero results. Elif keeps the
        # all-dead case owned by the single aggregate ERROR above.
        known = sorted({cid for ids in MODULE_CHECK_IDS.values()
                        for cid in ids})
        families = sorted({cid.split(".", 1)[0] for cid in known})
        dead_includes = [p for p in (include or [])
                         if not any(_matches(cid, p) for cid in known)
                         and not any(_matches(r.check, p) for r in results)]
        if dead_includes:
            results.append(CheckResult(
                "audit.filter_pattern_matched_nothing", Status.ERROR,
                f"include pattern(s) {dead_includes!r} matched 0 of the "
                f"{len(known)} known checks and 0 results - the checks "
                f"they were meant to select were never audited, while the "
                f"rest of the report looks normal",
                severity=Severity.HIGH,
                details={"dead_include_patterns": dead_includes,
                         "include": list(include or []),
                         "known_check_ids": known},
                remediation=("fix the typo'd include pattern(s) (a typo "
                             "silently matches nothing); valid family "
                             f"prefixes: {', '.join(families)}")))
        # An exclude typo fails toward more auditing (nothing dropped),
        # so a WARN-level note is proportionate - no ERROR, ok untouched.
        dead_excludes = [p for p in (exclude or [])
                         if not any(cid.startswith(p) for cid in known)
                         and not any(r.check.startswith(p)
                                     for r in pre_exclude)]
        if dead_excludes:
            results.append(CheckResult(
                "audit.exclude_pattern_matched_nothing", Status.WARN,
                f"exclude pattern(s) {dead_excludes!r} matched 0 of the "
                f"{len(known)} known checks and excluded 0 results "
                f"(exclude matches by PREFIX only) - nothing was dropped, "
                f"but the filter is not doing what its author intended",
                severity=Severity.LOW,
                details={"dead_exclude_patterns": dead_excludes,
                         "exclude": list(exclude or [])},
                remediation=("fix or remove the exclude pattern(s); valid "
                             f"family prefixes: {', '.join(families)}")))

    # audit.coverage: every verdict above judged the signals x
    # asset_returns intersection. When that intersection discards most of
    # a panel the caller handed in (a 1-of-20-asset vendor join, a
    # 30-of-500-date calendar mismatch), the CLEAN headline certifies a
    # sliver - say so out loud. Appended after the filters, like the
    # synthetic guard above, so include/exclude can never drop it.
    coverage_issues: list[tuple[str, str, int, int, float]] = []
    n_kept_dates = len(aligned.common_index)
    n_kept_assets = len(aligned.common_assets)
    for panel_name in ("signals", "asset_returns"):
        n_dates, n_assets = input_shapes[panel_name]
        for axis, total, kept in (("dates", n_dates, n_kept_dates),
                                  ("columns", n_assets, n_kept_assets)):
            frac = kept / total
            if frac < _COVERAGE_MIN_KEPT_FRAC:
                coverage_issues.append((panel_name, axis, kept, total, frac))
    # Content starvation short of validate()'s hard zero gate: a book with
    # 1..(_MIN_OVERLAP-1) jointly-finite dates validates - a signal live
    # only near the end of a dense book is a legitimate layout and the
    # signal-consuming checks SKIP with measured overlaps - but the
    # statistical checks cannot support a verdict on it, so the report
    # must disclose the starvation rather than headline CLEAN unqualified.
    joint_finite = (
        np.isfinite(aligned.signals.to_numpy(dtype="float64",
                                             na_value=np.nan))
        & np.isfinite(aligned.asset_returns.to_numpy(dtype="float64",
                                                     na_value=np.nan)))
    usable_joint_dates = int(joint_finite.any(axis=1).sum())
    joint_assets_by_date = joint_finite.sum(axis=1)
    max_joint_assets = int(joint_assets_by_date.max())
    median_joint_assets = float(np.median(joint_assets_by_date))
    dates_with_cross_sectional_support = int(
        (joint_assets_by_date >= _MIN_CROSS_SECTIONAL_ASSETS).sum())
    date_starved = usable_joint_dates < _MIN_OVERLAP
    breadth_starved = (
        dates_with_cross_sectional_support < _MIN_OVERLAP
    )
    if coverage_issues or date_starved or breadth_starved:
        parts = [f"{p}.{ax}: kept {k} of {t} ({fr:.0%})"
                 for p, ax, k, t, fr in coverage_issues]
        if date_starved:
            parts.append(f"jointly finite (same bar, same asset) dates: "
                         f"{usable_joint_dates} of {n_kept_dates} - below "
                         f"the {_MIN_OVERLAP}-date support the statistical "
                         f"checks need")
        if breadth_starved:
            parts.append(
                f"dates with at least {_MIN_CROSS_SECTIONAL_ASSETS} jointly "
                f"finite assets: {dates_with_cross_sectional_support} of "
                f"{n_kept_dates} (maximum breadth {max_joint_assets}) - "
                f"below the {_MIN_OVERLAP}-date support cross-sectional "
                f"checks need")
        details: dict[str, object] = {
            "kept_dates": n_kept_dates,
            "kept_assets": n_kept_assets,
            "min_kept_frac": _COVERAGE_MIN_KEPT_FRAC,
            "usable_joint_dates": usable_joint_dates,
            "max_jointly_finite_assets": max_joint_assets,
            "median_jointly_finite_assets": median_joint_assets,
            "dates_with_cross_sectional_support":
                dates_with_cross_sectional_support,
            "min_cross_sectional_assets": _MIN_CROSS_SECTIONAL_ASSETS,
            "shortfalls": {f"{p}.{ax}": {"kept": k, "total": t,
                                         "kept_frac": fr}
                           for p, ax, k, t, fr in coverage_issues}}
        results.append(CheckResult(
            "audit.coverage", Status.WARN,
            f"the audited signals x asset_returns grid covers a sliver of "
            f"the inputs ({'; '.join(parts)}) - every verdict above "
            f"judged only the surviving {n_kept_dates}-date x "
            f"{n_kept_assets}-asset grid, not the panels you passed",
            severity=Severity.HIGH,
            details=details,
            remediation=("align the two panels' calendars and tickers "
                         "(vendor join / timezone / suffix mismatch is the "
                         "usual cause), or trim both to the intended "
                         "sample before auditing, so the certified grid "
                         "is the grid you built; gate deployments with "
                         "report.gate(warn_severity=Severity.HIGH) to "
                         "fail closed on this advisory")))

    meta: dict[str, object] = {
        "n_periods": int(len(aligned.common_index)),
        "n_assets": int(len(aligned.common_assets)),
        "usable_joint_dates": usable_joint_dates,
        "max_jointly_finite_assets": max_joint_assets,
        "median_jointly_finite_assets": median_joint_assets,
        "dates_with_cross_sectional_support":
            dates_with_cross_sectional_support,
        "start": str(aligned.common_index[0].date()) if len(aligned.common_index) else None,
        "end": str(aligned.common_index[-1].date()) if len(aligned.common_index) else None,
        "signal_lag": int(aligned.signal_lag),
        "label_horizon": int(aligned.label_horizon),
        "periods_per_year": float(aligned.periods_per_year),
        "declared_costs_bps": (None if aligned.declared_costs_bps is None
                               else float(aligned.declared_costs_bps)),
        "train_period": (None if aligned.train_period is None else
                         [str(bound) for bound in aligned.train_period]),
        "test_period": (None if aligned.test_period is None else
                        [str(bound) for bound in aligned.test_period]),
        "has_positions": aligned.positions is not None,
        "has_strategy_returns": aligned.strategy_returns is not None,
        "has_universe": aligned.universe is not None,
        "has_signal_func": signal_func is not None,
        "has_backtest_func": backtest_func is not None,
        "n_trials": config.n_trials,
        # Gate contract, intentionally outside the best-effort provenance
        # block: strict gates must still know the selected modules' complete
        # registry if provenance hashing itself degrades. These are the ids
        # the selected modules can/should account for before result filters;
        # dynamic.probe_health remains a diagnostic and is ignored by gate().
        "expected_check_ids": [
            cid for mod_name in modules
            for cid in MODULE_CHECK_IDS.get(mod_name, ())
        ],
    }
    if "error" not in provenance:
        provenance["checks_emitted"] = [r.check for r in results]
        provenance["timings_s"] = {m: round(t, 6)
                                    for m, t in timings_s.items()}
    meta["provenance"] = provenance
    return AuditReport(results=results, meta=meta)


def _panel_provenance_stats(obj) -> dict | None:
    """shape / finite-cell fraction / span for one aligned artifact."""
    if obj is None:
        return None
    vals = obj.to_numpy(dtype="float64", na_value=np.nan)
    return {
        "shape": list(vals.shape),
        "non_null_frac": (float(np.isfinite(vals).mean())
                          if vals.size else 0.0),
        "index_start": str(obj.index[0]) if len(obj.index) else None,
        "index_end": str(obj.index[-1]) if len(obj.index) else None,
    }


# Names of the aligned artifacts that get a content digest. signal_input
# is included because it is the raw pipeline input the dynamic probes feed
# to signal_func: two archives with equal signals but different raw input
# ran different placebo/shuffle probes.
_DIGESTED_PANELS: tuple[str, ...] = (
    "signals", "asset_returns", "positions", "strategy_returns",
    "universe", "prices", "signal_input")


def _framed(tag: bytes, chunks: list[bytes]) -> bytes:
    """Length-prefixed byte framing shared by the provenance digests."""
    out = bytearray(tag)
    for chunk in chunks:
        out += struct.pack(">Q", len(chunk))
        out += chunk
    return bytes(out)


def _panel_digest(obj) -> str | None:
    """sha256 over one aligned artifact's cells and its labels.

    Shapes and coverage alone cannot tell a negated signal panel - or a
    relabelled one - from the original. The digest covers the float64 cell
    bytes (NaN canonicalised so NaN payload bits never differ), the shape,
    and type-and-length-framed index and column labels, so a one-cell edit, a
    column swap, or a date relabel each flip it without collisions between
    labels such as integer ``1`` and string ``"1"``. Cost is O(cells): one
    float64 copy plus one hash pass.

    Non-numeric panels (an object-dtype ``signal_input`` of tickers/text)
    use the same type-framed deterministic scalar encoding as labels. Opaque
    custom values fail provenance closed instead of relying on a potentially
    constant or address-bearing ``repr``.
    """
    if obj is None:
        return None
    import hashlib
    import datetime
    import decimal
    import fractions
    import pathlib
    import uuid

    def _atom_bytes(value, depth: int = 0) -> bytes:
        type_name = (f"{type(value).__module__}."
                     f"{type(value).__qualname__}").encode(
                         "utf-8", "backslashreplace")
        if value is pd.NA or value is pd.NaT:
            return _framed(b"missing:" + type_name, [])
        if value is None or isinstance(value, (bool, int, float, complex,
                                               str, bytes)):
            return _framed(b"scalar:" + type_name, [
                repr(value).encode("utf-8", "backslashreplace")])
        if isinstance(value, np.generic):
            return _framed(b"numpy-scalar:" + type_name, [
                value.dtype.str.encode(),
                repr(value).encode("utf-8", "backslashreplace")])
        if isinstance(value, (pd.Timestamp, pd.Timedelta,
                              datetime.datetime, datetime.date,
                              datetime.time, datetime.timedelta,
                              decimal.Decimal, fractions.Fraction,
                              pathlib.PurePath, uuid.UUID)):
            return _framed(b"stable-value:" + type_name, [
                str(value).encode("utf-8", "backslashreplace")])
        if isinstance(value, pd.Period):
            return _framed(b"pandas-period", [str(value).encode(),
                                                value.freqstr.encode()])
        if isinstance(value, pd.Interval):
            return _framed(b"pandas-interval", [
                _atom_bytes(value.left, depth + 1),
                _atom_bytes(value.right, depth + 1),
                value.closed.encode(),
            ])
        if depth > 32:
            raise TypeError("panel label/value nesting exceeds 32")
        if isinstance(value, tuple):
            return _framed(b"tuple:" + type_name,
                           [_atom_bytes(x, depth + 1) for x in value])
        if isinstance(value, frozenset):
            return _framed(b"frozenset:" + type_name,
                           sorted(_atom_bytes(x, depth + 1) for x in value))
        raise TypeError(
            f"unsupported opaque panel label/value type "
            f"{type(value).__module__}.{type(value).__qualname__}; use "
            f"string/numeric/temporal or tuple labels and deterministic "
            f"plain scalar object cells")

    def _update_labels(h, section: bytes, labels) -> None:
        """Hash a label sequence with explicit type and length framing."""
        h.update(b"\x00" + section + b":")
        h.update(struct.pack(">Q", len(labels)))
        for label in labels:
            value = _atom_bytes(label)
            h.update(struct.pack(">Q", len(value)))
            h.update(value)

    logical_shape = tuple(obj.shape)
    try:
        if isinstance(obj, pd.DataFrame):
            series_and_dtypes = [(obj.iloc[:, i], dtype)
                                 for i, dtype in enumerate(obj.dtypes)]
        else:
            series_and_dtypes = [(obj, obj.dtype)]
        numeric = all(pd.api.types.is_numeric_dtype(dtype)
                      for _, dtype in series_and_dtypes)
        complex_payload = any(pd.api.types.is_complex_dtype(dtype)
                              for _, dtype in series_and_dtypes)
        # float64 is the checks' canonical numeric representation, but it
        # cannot distinguish adjacent integers above 2**53. Provenance promises
        # that a one-cell edit flips the digest, so preserve those rare large
        # integer cells through the exact typed-scalar path below.
        max_exact_float_int = 2 ** 53
        unsafe_integer = any(
            pd.api.types.is_integer_dtype(dtype)
            and bool(((series.dropna() > max_exact_float_int)
                      | (series.dropna() < -max_exact_float_int)).any())
            for series, dtype in series_and_dtypes
        )
        if not numeric or complex_payload or unsafe_integer:
            raise TypeError("typed object payload")
        vals = obj.to_numpy(dtype="float64", na_value=np.nan)
        # Fixed little-endian representation keeps digests stable across
        # CPU-native byte order as well as across contiguous/view layouts.
        vals = np.ascontiguousarray(vals, dtype="<f8")
        # canonical NaN: distinct NaN bit patterns must not split digests
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            # ascontiguousarray may retain a caller-owned array, including
            # a read-only pandas Copy-on-Write view. Own the storage before
            # normalizing NaNs, preserving both the input and hash format.
            vals = vals.copy()
            vals[nan_mask] = np.nan
        kind = b"float64"
    except (TypeError, ValueError):
        # pandas' object hash intentionally normalizes some scalar types
        # (bytes b"a" and str "a" collide). Frame every cell with its type
        # and representation instead; non-numeric panels are rare and this
        # defensive path favors identity over vectorized speed.
        payload = bytearray()
        cells = obj.to_numpy(dtype=object).reshape(-1)
        for cell in cells:
            try:
                missing = pd.isna(cell)
                is_missing = bool(missing) if isinstance(
                    missing, (bool, np.bool_)) else False
            except (TypeError, ValueError):
                is_missing = False
            if is_missing:
                value = _framed(b"missing-cell", [])
            else:
                value = _atom_bytes(cell)
            payload += struct.pack(">Q", len(value)) + value
        vals = np.frombuffer(bytes(payload), dtype=np.uint8)
        kind = b"typed-object-v1"
    h = hashlib.sha256()
    h.update(kind)
    h.update(b"\x00shape:" + repr(logical_shape).encode())
    h.update(b"\x00cells:")
    h.update(vals.tobytes())
    _update_labels(h, b"index", list(obj.index))
    _update_labels(h, b"index-names", list(obj.index.names))
    if isinstance(obj, pd.DataFrame):
        _update_labels(h, b"columns", list(obj.columns))
        _update_labels(h, b"column-names", list(obj.columns.names))
    else:
        _update_labels(h, b"series-name", [getattr(obj, "name", None)])
    return h.hexdigest()


def _callable_label(fn) -> str | None:
    """``"<module>.<qualname>"`` of a pipeline callable, or None.

    Non-identifying by design: two closures built by the same factory with
    different parameters (costs 10 bps vs 0 bps) carry the same label, as
    do two lambdas from the same notebook cell. Hashing ``__code__`` would
    not help (both closures share it) and would present false identity;
    the label is a locator for the human reading the archive, nothing
    more. ``functools.partial`` is unwrapped so the wrapped function's
    name shows instead of ``functools.partial``.
    """
    if fn is None:
        return None
    import functools

    if isinstance(fn, functools.partial):
        return f"functools.partial({_callable_label(fn.func)})"
    module = getattr(fn, "__module__", None) or type(fn).__module__
    qualname = (getattr(fn, "__qualname__", None)
                or getattr(fn, "__name__", None)
                or type(fn).__qualname__)
    return f"{module}.{qualname}"


def _package_digest() -> str:
    """Content digest of the Python source that rendered the verdict.

    Version strings can be reused accidentally in a dirty checkout or a
    rebuilt wheel.  Hashing the installed package tree gives an archive a
    code identity without shelling out to git (which may not exist in a
    wheel/container deployment).
    """
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        body = path.read_bytes()
        h.update(struct.pack(">Q", len(rel)))
        h.update(rel)
        h.update(struct.pack(">Q", len(body)))
        h.update(body)
    return h.hexdigest()


def _callable_digest(fn) -> str | None:
    """Best-effort sha256 fingerprint of callable code and captured state.

    This distinguishes ordinary closures/partials that share a qualname but
    capture different parameters.  It remains a reproducibility aid, not a
    sandbox or a proof about mutable globals/external services.  Only a
    digest is emitted, never the captured values themselves.
    """
    if fn is None:
        return None
    import dataclasses
    import datetime
    import decimal
    import enum
    import fractions
    import functools
    import hashlib
    import marshal
    import pathlib
    import random
    import types
    import uuid

    seen: set[int] = set()
    visited_nodes = 0

    class _UnsupportedDigestState(Exception):
        """Captured state cannot be fingerprinted without false identity."""

    def object_state_bytes(value, depth: int) -> bytes | None:
        """Deterministic visible instance state, including mixed slots."""
        try:
            state = object.__getattribute__(value, "__dict__")
        except (AttributeError, TypeError):
            state = None
        slot_state: dict[str, object] = {}
        for cls in type(value).__mro__:
            declared = cls.__dict__.get("__slots__", ())
            names = (declared,) if isinstance(declared, str) else declared
            if isinstance(names, dict):
                names = names.keys()
            for raw_name in names:
                if not isinstance(raw_name, str) or raw_name in {
                        "__dict__", "__weakref__"}:
                    continue
                attr_name = raw_name
                if raw_name.startswith("__") and not raw_name.endswith("__"):
                    attr_name = f"_{cls.__name__.lstrip('_')}{raw_name}"
                key = f"{cls.__module__}.{cls.__qualname__}.{raw_name}"
                try:
                    slot_state[key] = object.__getattribute__(value, attr_name)
                except AttributeError:
                    slot_state[key] = "<uninitialized-slot>"
        if not isinstance(state, dict) and not slot_state:
            return None
        type_tag = (f"{type(value).__module__}."
                    f"{type(value).__qualname__}").encode()
        return _framed(b"object-state:" + type_tag, [
            value_bytes(state if isinstance(state, dict) else {}, depth + 1),
            value_bytes(slot_state, depth + 1),
        ])

    def value_bytes(value, depth: int = 0) -> bytes:
        nonlocal visited_nodes
        visited_nodes += 1
        if visited_nodes > 100_000:
            raise _UnsupportedDigestState("captured state exceeds node budget")
        if isinstance(value, enum.Enum):
            return _framed(
                f"enum:{type(value).__module__}.{type(value).__qualname__}"
                .encode(), [value.name.encode(),
                            value_bytes(value.value, depth + 1)])
        if value is None or isinstance(value, (bool, int, float, complex,
                                               str, bytes)):
            return _framed(
                f"scalar:{type(value).__module__}.{type(value).__qualname__}"
                .encode(), [repr(value).encode("utf-8", "backslashreplace")])
        if isinstance(value, (datetime.datetime, datetime.date,
                              datetime.time, datetime.timedelta,
                              decimal.Decimal, fractions.Fraction,
                              pathlib.PurePath, uuid.UUID, range)):
            return _framed(
                f"stable-value:{type(value).__module__}."
                f"{type(value).__qualname__}".encode(),
                [str(value).encode("utf-8", "backslashreplace")])
        if isinstance(value, slice):
            return _framed(b"slice", [value_bytes(value.start, depth + 1),
                                      value_bytes(value.stop, depth + 1),
                                      value_bytes(value.step, depth + 1)])
        # Keep deterministic leaves before the cutoff, or two deep captures
        # ending in different scalar values would collide at that boundary.
        # Unsupported deeper structures make the digest unavailable instead
        # of claiming false identity.
        if depth > 64:
            raise _UnsupportedDigestState("captured state exceeds depth limit")
        if isinstance(value, np.generic):
            return value_bytes(value.item(), depth + 1)
        if isinstance(value, (pd.DataFrame, pd.Series)):
            digest = _panel_digest(value) or "none"
            return _framed(b"pandas", [digest.encode()])
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                chunks = [value_bytes(x, depth + 1)
                          for x in value.reshape(-1).tolist()]
            else:
                chunks = [np.ascontiguousarray(value).tobytes()]
            return _framed(b"ndarray", [value.dtype.str.encode(),
                                        repr(value.shape).encode(), *chunks])
        if isinstance(value, type):
            return _framed(b"type", [
                f"{value.__module__}.{value.__qualname__}".encode()])

        ident = id(value)
        if ident in seen:
            raise _UnsupportedDigestState("captured state contains a cycle")
        seen.add(ident)
        try:
            type_tag = (f"{type(value).__module__}."
                        f"{type(value).__qualname__}").encode()
            if isinstance(value, dict):
                pairs = [(value_bytes(k, depth + 1),
                          value_bytes(v, depth + 1))
                         for k, v in value.items()]
                pairs.sort(key=lambda pair: pair[0])
                return _framed(b"dict:" + type_tag,
                              [_framed(b"item", [k, v])
                               for k, v in pairs])
            if isinstance(value, (list, tuple)):
                return _framed(b"sequence:" + type_tag,
                              [value_bytes(x, depth + 1) for x in value])
            if isinstance(value, (set, frozenset)):
                parts = sorted(value_bytes(x, depth + 1) for x in value)
                return _framed(b"set:" + type_tag, parts)
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                parts = [_framed(b"field", [field.name.encode(),
                          value_bytes(getattr(value, field.name), depth + 1)])
                         for field in dataclasses.fields(value)]
                return _framed(b"dataclass:" + type_tag, parts)
            if isinstance(value, np.random.Generator):
                return _framed(b"numpy-generator:" + type_tag, [
                    value_bytes(value.bit_generator.state, depth + 1)])
            if isinstance(value, random.Random):
                return _framed(b"stdlib-random:" + type_tag, [
                    value_bytes(value.getstate(), depth + 1)])
            if callable(value):
                return _framed(b"nested-callable", [callable_bytes(value)])
            visible_state = object_state_bytes(value, depth)
            if visible_state is not None:
                return visible_state
            # repr(object) commonly embeds a process-specific address, while
            # type-only bytes falsely identify behaviorally different state.
            # Returning no digest lets deployment gate() fail closed.
            raise _UnsupportedDigestState(
                f"opaque captured state {type(value).__qualname__}")
        finally:
            seen.discard(ident)

    def normalized_code(code: types.CodeType) -> types.CodeType:
        # co_filename embeds the checkout/venv path; normalize it (including
        # nested comprehensions/functions) so identical callable code hashes
        # identically after a wheel is installed elsewhere.
        constants = tuple(normalized_code(x)
                          if isinstance(x, types.CodeType) else x
                          for x in code.co_consts)
        return code.replace(co_filename="<callable>", co_consts=constants)

    def callable_bytes(callable_obj) -> bytes:
        if isinstance(callable_obj, functools.partial):
            return _framed(b"partial", [callable_bytes(callable_obj.func),
                                        value_bytes(callable_obj.args),
                                        value_bytes(callable_obj.keywords or {})])
        code = getattr(callable_obj, "__code__", None)
        owner = callable_obj
        if code is None:
            owner = getattr(callable_obj, "__call__", callable_obj)
            code = getattr(owner, "__code__", None)
        chunks = [(_callable_label(callable_obj) or "unknown").encode()]
        if code is not None:
            chunks.append(marshal.dumps(normalized_code(code)))
        chunks.append(value_bytes(getattr(owner, "__defaults__", None)))
        chunks.append(value_bytes(getattr(owner, "__kwdefaults__", None)))
        closure = getattr(owner, "__closure__", None) or ()
        closure_values = []
        for cell in closure:
            try:
                closure_values.append(cell.cell_contents)
            except ValueError:  # empty closure cell
                closure_values.append("<empty-cell>")
        chunks.append(value_bytes(closure_values))
        # A bound method's behavior commonly depends on instance fields, but
        # ``method.__dict__`` is empty. Include __self__ so Pipeline(0).run
        # and Pipeline(25).run cannot share a provenance fingerprint.
        chunks.append(value_bytes(getattr(callable_obj, "__self__", None)))
        # A callable instance has no __self__; include its dict and every
        # initialized slot directly. This covers both slots-only and mixed
        # slots+__dict__ pipeline objects.
        own_state = object_state_bytes(callable_obj, 0)
        chunks.append(own_state if own_state is not None
                      else _framed(b"no-own-instance-state", []))
        return _framed(b"callable-v1", chunks)

    try:
        return hashlib.sha256(callable_bytes(fn)).hexdigest()
    except _UnsupportedDigestState:
        return None


def _provenance(aligned: BacktestArtifacts,
                config: AuditConfig,
                modules: list[str],
                *,
                include: list[str] | None,
                exclude: list[str] | None,
                strict: bool,
                signal_func: Callable | None = None,
                backtest_func: Callable | None = None) -> dict:
    """Reproducibility record for ``report.meta["provenance"]``.

    Without it a saved report is un-auditable: nothing says which
    thresholds, which check set, or which library versions produced its
    CLEAN verdict, so a deployment archive cannot distinguish a full
    audit from a filtered one. Everything here except ``timestamp_utc``
    and ``timings_s`` is deterministic for fixed inputs; ``config_hash``
    is the sha256 of the JSON-cleaned config so numpy-vs-python scalar
    overrides hash identically.

    ``artifact_digest`` is a per-panel sha256 over the aligned cells plus
    index/column labels, so the archive also proves *which inputs* the
    verdict was rendered on - a negated or relabelled panel gets a distinct
    provenance record. ``qaudit_source_digest`` identifies the installed
    Python source tree. ``callables`` holds human locators while
    ``callable_digest`` fingerprints ordinary code, defaults, partial
    arguments and closure state. audit() captures these before running
    probes, then fills in ``checks_emitted`` and ``timings_s`` once the
    modules have run. The fingerprint is best-effort: mutable globals and
    external services remain outside an in-process fingerprint.

    Never raises: the audit verdict outranks the provenance nicety, so any
    assembly failure degrades to an ``{"error": ...}`` block - a
    deployment consumer gating on ``provenance["config_hash"]`` presence
    still catches the degradation.
    """
    try:
        import dataclasses
        import hashlib
        import json
        import platform
        from datetime import datetime, timezone

        from .report import _json_clean

        # Read dataclass fields without asdict(): asdict recursively deep-
        # copies shared object graphs before our bounded cleaner sees them.
        # AuditConfig.validate() has already rejected cycles and excessive
        # graph expansion/depth; the private _validation_active guard is not
        # a dataclass field.
        cfg_dict = _json_clean({field.name: getattr(config, field.name)
                                for field in dataclasses.fields(config)})
        cfg_hash = hashlib.sha256(
            json.dumps(cfg_dict, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()
        artifact_parameters = _json_clean({
            "signal_lag": aligned.signal_lag,
            "label_horizon": aligned.label_horizon,
            "periods_per_year": aligned.periods_per_year,
            "declared_costs_bps": aligned.declared_costs_bps,
            "train_period": aligned.train_period,
            "test_period": aligned.test_period,
        })
        artifact_parameters_hash = hashlib.sha256(
            json.dumps(artifact_parameters, sort_keys=True,
                       separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

        deps: dict[str, str] = {}
        # module attributes, not importlib.metadata: qaudit may run
        # un-installed from a source checkout with src/ on sys.path
        for label, getter in (
                ("python", platform.python_version),
                ("numpy", lambda: np.__version__),
                ("pandas", lambda: pd.__version__)):
            try:
                deps[label] = str(getter())
            except Exception:  # noqa: BLE001 - a dep version is a nicety
                deps[label] = "unknown"
        try:  # api.py itself never needs scipy; import lazily and guarded
            import scipy
            deps["scipy"] = str(scipy.__version__)
        except Exception:  # noqa: BLE001
            deps["scipy"] = "unknown"

        try:
            from qaudit import __version__ as qaudit_version
        except Exception:  # noqa: BLE001
            qaudit_version = "unknown"

        return {
            "qaudit_version": qaudit_version,
            "qaudit_source_digest": _package_digest(),
            "config": cfg_dict,
            "config_hash": cfg_hash,
            "checks_selected_modules": list(modules),
            "checks_expected": {m: list(MODULE_CHECK_IDS.get(m, ()))
                                for m in modules},
            "checks_emitted": [],   # filled by audit() after the modules run
            "include": list(include) if include is not None else None,
            "exclude": list(exclude) if exclude is not None else None,
            "strict": bool(strict),
            "timings_s": {},        # filled by audit() after the modules run
            "deps": deps,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_coverage": {
                "signals": _panel_provenance_stats(aligned.signals),
                "asset_returns": _panel_provenance_stats(
                    aligned.asset_returns),
                "positions": _panel_provenance_stats(aligned.positions),
                "strategy_returns": _panel_provenance_stats(
                    aligned.strategy_returns),
                "universe": _panel_provenance_stats(aligned.universe),
                "prices": _panel_provenance_stats(aligned.prices),
            },
            "artifact_digest": {
                name: _panel_digest(getattr(aligned, name, None))
                for name in _DIGESTED_PANELS},
            "artifact_parameters": artifact_parameters,
            "artifact_parameters_hash": artifact_parameters_hash,
            "callables": {
                "signal_func": _callable_label(signal_func),
                "backtest_func": _callable_label(backtest_func),
            },
            "callable_digest": {
                "signal_func": _callable_digest(signal_func),
                "backtest_func": _callable_digest(backtest_func),
            },
        }
    except Exception as exc:  # noqa: BLE001 - verdict outranks provenance
        return {"error": (f"provenance construction failed: "
                          f"{type(exc).__name__}: {exc}")}
