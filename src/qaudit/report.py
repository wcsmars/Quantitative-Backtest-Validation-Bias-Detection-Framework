"""Audit report: aggregation, rendering, and gating."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import AuditFailure, InputValidationError
from .types import CheckResult, Severity, Status

_STATUS_ORDER = [Status.FAIL, Status.ERROR, Status.WARN, Status.PASS, Status.SKIP]

# Diagnostic ids describe the audit itself (input coverage, filter
# accounting, null-probe health), not the backtest: they can never satisfy
# a gate(require=...) pattern in either mode. Otherwise an all-SKIP dynamic
# family whose only spoken result is a LOW dynamic.probe_health WARN (every
# null re-run degenerate) would satisfy gate(require=["dynamic"]).
_DIAGNOSTIC_CHECK_IDS = frozenset({"dynamic.probe_health"})
_DIAGNOSTIC_PREFIXES = ("audit.",)

# Machine-readable verdict flags carried in CheckResult.details:
#   not_judged=True  on a PASS that declined to judge (delegated to a
#                    sibling, sample too short to evaluate) - gate() strict
#                    mode treats it like a SKIP;
#   unresolved=True  on a PASS whose evidence sat above a bar but whose
#                    conviction gate did not clear - advisory APIs keep the
#                    PASS, while gate() requires an explicit opt-in;
#   superseded=True  on a SKIP that is intentional because a better artifact
#                    armed a sibling check - gate() strict mode ignores it.
_FLAG_NOT_JUDGED = "not_judged"
_FLAG_UNRESOLVED = "unresolved"
_FLAG_SUPERSEDED = "superseded"


def _is_diagnostic(check_id: str) -> bool:
    return (check_id in _DIAGNOSTIC_CHECK_IDS
            or check_id.startswith(_DIAGNOSTIC_PREFIXES))


def _flag(r: CheckResult, name: str) -> bool:
    """True iff ``r.details[name]`` is set and truthy."""
    return bool(r.details.get(name, False))


def _result_contract_problem(r: CheckResult) -> str | None:
    """Return why a result is structurally unsafe, or ``None``.

    ``CheckResult`` normalizes enum strings at construction, but remains
    mutable for backward compatibility. Revalidate at module dispatch and
    deployment gating so a later ``result.status = "fail"`` cannot evade
    identity-based selectors, and a truthy string ``superseded="false"``
    cannot turn an unrun check into accepted coverage.
    """
    if not isinstance(r, CheckResult):
        return f"expected CheckResult, got {type(r).__name__}"
    if not isinstance(r.check, str) or not r.check:
        return "check must be a non-empty str"
    if not isinstance(r.status, Status):
        return f"status must be Status, got {type(r.status).__name__}"
    if not isinstance(r.severity, Severity):
        return f"severity must be Severity, got {type(r.severity).__name__}"
    if not isinstance(r.message, str):
        return f"message must be str, got {type(r.message).__name__}"
    if not isinstance(r.remediation, str):
        return (f"remediation must be str, got "
                f"{type(r.remediation).__name__}")
    if not isinstance(r.details, dict):
        return f"details must be dict, got {type(r.details).__name__}"
    for flag_name in (_FLAG_NOT_JUDGED, _FLAG_UNRESOLVED,
                      _FLAG_SUPERSEDED):
        if flag_name in r.details and type(r.details[flag_name]) is not bool:
            return (f"details[{flag_name!r}] must be an actual bool, got "
                    f"{type(r.details[flag_name]).__name__}")
    return None


def _expected_check_ids(meta: dict[str, Any]) -> list[str]:
    """Best-effort registry contract attached by :func:`qaudit.audit`.

    Hand-built ``AuditReport`` instances intentionally have no such contract
    and are gated on their results alone.
    """
    raw = meta.get("expected_check_ids")
    if isinstance(raw, (list, tuple)) and all(isinstance(x, str) for x in raw):
        return list(dict.fromkeys(raw))
    return []


def _json_key(k: Any) -> str:
    """Dict/Series key -> JSON-legal string key."""
    import datetime as _dt

    import pandas as pd

    if isinstance(k, str):
        return k
    if isinstance(k, (pd.Timestamp, _dt.datetime, _dt.date)):
        return k.isoformat()
    try:
        return str(k)
    except Exception:  # noqa: BLE001 - archival fallback must not crash
        return (f"<unrepresentable:{type(k).__module__}."
                f"{type(k).__qualname__}>")


def _json_clean(v: Any, *, _stack: set[int] | None = None,
                _state: list[int] | None = None, _depth: int = 0) -> Any:
    """Arbitrary value -> plain data guaranteed ``json.dumps``-compatible
    (even with ``allow_nan=False``): Timestamps become ISO strings -
    including as dict/Series keys - numpy scalars become Python scalars,
    non-finite floats become None, and anything unrecognized falls back to
    ``repr`` or a deterministic type marker when representation itself is
    broken. Module-level so :func:`qaudit.api._provenance` can hash the
    cleaned config with the exact serialization ``to_dict`` uses."""
    import datetime as _dt
    import math

    import numpy as np
    import pandas as pd

    if _stack is None:
        _stack = set()
    if _state is None:
        _state = [0]
    _state[0] += 1

    def clean(value: Any) -> Any:
        return _json_clean(value, _stack=_stack, _state=_state,
                           _depth=_depth + 1)

    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    # temporal branch before the integer branch: np.timedelta64
    # subclasses np.signedinteger (numpy 1.26), so the int branch
    # would shadow it - int(np.timedelta64(5, "D")) raises TypeError
    # and ns-unit values serialized as bare ints
    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
        return v.isoformat()
    if isinstance(v, (pd.Timedelta, _dt.timedelta, np.datetime64,
                      np.timedelta64)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None

    recursive = isinstance(v, (pd.Series, pd.DataFrame, np.ndarray, dict,
                               set, frozenset, list, tuple))
    if recursive:
        type_name = f"{type(v).__module__}.{type(v).__qualname__}"
        if id(v) in _stack:
            return {"__qaudit_cycle_v1__": type_name}
        # A physically small shared DAG can otherwise expand exponentially.
        # Prune at a deterministic logical-node/depth boundary, while scalar
        # leaves above were still serialized exactly.
        if _depth > 64 or _state[0] > 100_000:
            why = "depth" if _depth > 64 else "node-budget"
            return {"__qaudit_truncated_v1__": f"{why}:{type_name}"}
        _stack.add(id(v))
    try:
        if isinstance(v, pd.Series):
            cleaned = [(_json_key(k),
                        f"{type(k).__module__}.{type(k).__qualname__}",
                        clean(k), clean(x))
                       for k, x in v.items()]
            bases = [base for base, _, _, _ in cleaned]
            if len(set(bases)) == len(bases):
                return {base: value for base, _, _, value in cleaned}
            # Preserve duplicate or stringification-colliding index entries.
            return {"__qaudit_typed_series_v1__": [
                {"position": i, "key_type": key_type, "key": key,
                 "value": value}
                for i, (_, key_type, key, value) in enumerate(cleaned)
            ]}
        if isinstance(v, pd.DataFrame):
            # Positional iloc avoids recursion on duplicated labels. Ordinary
            # unique columns keep the plain object shape; any duplicate
            # switches to an ordered typed record, never a synthetic suffix.
            cleaned = [(_json_key(c),
                        f"{type(c).__module__}.{type(c).__qualname__}",
                        clean(c), clean(v.iloc[:, i]))
                       for i, c in enumerate(v.columns)]
            bases = [base for base, _, _, _ in cleaned]
            if len(set(bases)) == len(bases):
                return {base: value for base, _, _, value in cleaned}
            return {"__qaudit_typed_columns_v1__": [
                {"position": i, "key_type": key_type, "key": key,
                 "value": value}
                for i, (_, key_type, key, value) in enumerate(cleaned)
            ]}
        if isinstance(v, np.ndarray):
            # ``tolist()`` returns a scalar for a zero-dimensional ndarray.
            # Iterating that scalar either crashes (numeric) or corrupts it
            # into characters (string). Re-enter the cleaner once so both 0-D
            # and ordinary nested arrays follow the same recursive path.
            return clean(v.tolist())
        if isinstance(v, dict):
            cleaned = [(_json_key(k),
                        f"{type(k).__module__}.{type(k).__qualname__}",
                        clean(k), clean(x))
                       for k, x in v.items()]
            bases = [base for base, _, _, _ in cleaned]
            if len(set(bases)) == len(bases):
                return {base: value for base, _, _, value in cleaned}
            # Stringification can collide (1 and "1", None and "None").
            import json

            entries = [
                {"key_type": key_type, "key": key, "value": value}
                for _, key_type, key, value in cleaned
            ]
            entries.sort(key=lambda item: json.dumps(
                [item["key_type"], item["key"]], sort_keys=True,
                separators=(",", ":"), allow_nan=False))
            return {"__qaudit_typed_mapping_v1__": entries}
        if isinstance(v, (set, frozenset)):
            # Hash-randomized iteration order must not move config_hash.
            import json

            cleaned = [clean(x) for x in v]
            return sorted(cleaned, key=lambda x: json.dumps(
                x, sort_keys=True, separators=(",", ":"), allow_nan=False))
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        try:
            return repr(v)
        except Exception:  # noqa: BLE001 - archival fallback must not crash
            return {
                "__qaudit_unrepresentable_v1__":
                    f"{type(v).__module__}.{type(v).__qualname__}"
            }
    finally:
        if recursive:
            _stack.remove(id(v))


_PROVENANCE_REQUIRED_KEYS = frozenset({
    "qaudit_version", "qaudit_source_digest", "config", "config_hash",
    "checks_selected_modules", "checks_expected", "checks_emitted",
    "include", "exclude", "strict", "timings_s", "deps",
    "timestamp_utc", "artifact_coverage", "artifact_digest", "callables",
    "callable_digest", "artifact_parameters", "artifact_parameters_hash",
})
_PROVENANCE_PANEL_NAMES = frozenset({
    "signals", "asset_returns", "positions", "strategy_returns",
    "universe", "prices", "signal_input",
})


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value.lower()))


def _provenance_problem(provenance: Any,
                        results: list[CheckResult],
                        meta: dict[str, Any]) -> str | None:
    """Validate the minimum self-consistent deployment provenance record."""
    import hashlib
    import json
    import math

    def _finite_json_number(value: Any) -> bool:
        """True for a non-bool JSON number representable as finite float."""
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            return False

    if not isinstance(provenance, dict):
        return f"malformed provenance: {type(provenance).__name__}"
    if "error" in provenance:
        return str(provenance.get("error") or "construction failed")
    missing = sorted(_PROVENANCE_REQUIRED_KEYS - provenance.keys())
    if missing:
        return f"incomplete provenance; missing fields: {missing}"
    for key in ("qaudit_source_digest", "config_hash"):
        if not _is_sha256(provenance.get(key)):
            return f"provenance field {key!r} is not a sha256 digest"
    config = provenance.get("config")
    if not isinstance(config, dict):
        return "provenance field 'config' must be a dict"
    try:
        config_bytes = json.dumps(
            config, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        return f"provenance config is not canonical JSON: {exc}"
    actual_config_hash = hashlib.sha256(config_bytes).hexdigest()
    if provenance["config_hash"].lower() != actual_config_hash:
        return "provenance config_hash does not match the archived config"

    modules = provenance.get("checks_selected_modules")
    expected = provenance.get("checks_expected")
    emitted = provenance.get("checks_emitted")
    if (not isinstance(modules, list)
            or not all(isinstance(x, str) and x for x in modules)):
        return "provenance checks_selected_modules must be a list of strings"
    if (not isinstance(expected, dict)
            or set(expected) != set(modules)
            or any(not isinstance(ids, list)
                   or not all(isinstance(cid, str) and cid for cid in ids)
                   for ids in expected.values())):
        return "provenance checks_expected is inconsistent with its modules"
    registered = meta.get("expected_check_ids")
    if (not isinstance(registered, list)
            or not all(isinstance(cid, str) and cid for cid in registered)
            or len(registered) != len(set(registered))):
        return "report expected_check_ids is incomplete or malformed"
    flattened = [cid for module in modules for cid in expected[module]]
    if flattened != registered:
        return ("provenance checks_expected does not match the report's "
                "selected registry")
    if emitted != [r.check for r in results]:
        return "provenance checks_emitted does not match this report"
    if type(provenance.get("strict")) is not bool:
        return "provenance field 'strict' must be an actual bool"
    for key in ("include", "exclude"):
        value = provenance.get(key)
        if (value is not None and
                (not isinstance(value, list)
                 or not all(isinstance(x, str) and x for x in value))):
            return f"provenance field {key!r} must be None or a string list"

    timings = provenance.get("timings_s")
    if (not isinstance(timings, dict) or set(timings) != set(modules)
            or any(not _finite_json_number(v) or v < 0
                   for v in timings.values())):
        return "provenance timings_s is inconsistent with its modules"
    deps = provenance.get("deps")
    if (not isinstance(deps, dict)
            or not {"python", "numpy", "pandas", "scipy"} <= set(deps)
            or any(not isinstance(v, str) for v in deps.values())):
        return "provenance dependency versions are incomplete or malformed"
    if not isinstance(provenance.get("qaudit_version"), str):
        return "provenance qaudit_version must be a string"
    if not isinstance(provenance.get("timestamp_utc"), str):
        return "provenance timestamp_utc must be a string"
    coverage = provenance.get("artifact_coverage")
    coverage_names = {"signals", "asset_returns", "positions",
                      "strategy_returns", "universe", "prices"}
    if (not isinstance(coverage, dict)
            or set(coverage) != coverage_names):
        return "provenance artifact_coverage is incomplete or malformed"
    for name, stats in coverage.items():
        if stats is None:
            continue
        if not isinstance(stats, dict) or set(stats) != {
                "shape", "non_null_frac", "index_start", "index_end"}:
            return f"provenance coverage stats for {name!r} are malformed"
        shape = stats["shape"]
        fraction = stats["non_null_frac"]
        if (not isinstance(shape, list) or len(shape) not in (1, 2)
                or any(type(n) is not int or n < 0 for n in shape)
                or not _finite_json_number(fraction)
                or not 0 <= fraction <= 1
                or any(stats[key] is not None
                       and not isinstance(stats[key], str)
                       for key in ("index_start", "index_end"))):
            return f"provenance coverage stats for {name!r} are malformed"

    artifacts = provenance.get("artifact_digest")
    if (not isinstance(artifacts, dict)
            or set(artifacts) != _PROVENANCE_PANEL_NAMES
            or any(v is not None and not _is_sha256(v)
                   for v in artifacts.values())
            or not _is_sha256(artifacts.get("signals"))
            or not _is_sha256(artifacts.get("asset_returns"))):
        return "provenance artifact_digest is incomplete or malformed"
    for name in coverage_names:
        if ((coverage[name] is None) != (artifacts[name] is None)):
            return (f"provenance coverage/digest presence disagrees for "
                    f"artifact {name!r}")

    parameters = provenance.get("artifact_parameters")
    parameter_keys = {
        "signal_lag", "label_horizon", "periods_per_year",
        "declared_costs_bps", "train_period", "test_period",
    }
    if not isinstance(parameters, dict) or set(parameters) != parameter_keys:
        return "provenance artifact_parameters is incomplete or malformed"
    try:
        parameter_bytes = json.dumps(
            parameters, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        return f"provenance artifact_parameters is not canonical JSON: {exc}"
    parameter_hash = provenance.get("artifact_parameters_hash")
    if not isinstance(parameter_hash, str) or not _is_sha256(parameter_hash):
        return "provenance artifact_parameters_hash is not a sha256 digest"
    if parameter_hash.lower() != hashlib.sha256(parameter_bytes).hexdigest():
        return ("provenance artifact_parameters_hash does not match the "
                "archived artifact parameters")
    for key in ("signal_lag", "label_horizon"):
        if type(parameters[key]) is not int or parameters[key] < 1:
            return f"provenance artifact parameter {key!r} is malformed"
    ppy = parameters["periods_per_year"]
    if not _finite_json_number(ppy) or ppy <= 0:
        return "provenance artifact parameter 'periods_per_year' is malformed"
    costs = parameters["declared_costs_bps"]
    if (costs is not None and
            (not _finite_json_number(costs) or costs < 0)):
        return ("provenance artifact parameter 'declared_costs_bps' is "
                "malformed")
    for key in ("train_period", "test_period"):
        period = parameters[key]
        if (period is not None and
                (not isinstance(period, list) or len(period) != 2
                 or not all(isinstance(bound, str) and bound
                            for bound in period))):
            return f"provenance artifact parameter {key!r} is malformed"
    for key in ("callables", "callable_digest"):
        value = provenance.get(key)
        if not isinstance(value, dict) or set(value) != {
                "signal_func", "backtest_func"}:
            return f"provenance field {key!r} is incomplete or malformed"
    if any(v is not None and not isinstance(v, str)
           for v in provenance["callables"].values()):
        return "provenance callable labels must be strings or None"
    if any(v is not None and not _is_sha256(v)
           for v in provenance["callable_digest"].values()):
        return "provenance callable digests must be sha256 strings or None"
    for name in ("signal_func", "backtest_func"):
        if ((provenance["callables"][name] is None)
                != (provenance["callable_digest"][name] is None)):
            return (f"provenance fingerprint for callable {name!r} is "
                    f"unavailable")
    return None


@dataclass
class AuditReport:
    results: list[CheckResult]
    meta: dict[str, Any] = field(default_factory=dict)

    # -- selectors ---------------------------------------------------------
    def by_status(self, status: Status) -> list[CheckResult]:
        return [r for r in self.results if r.status is status]

    @property
    def failures(self) -> list[CheckResult]:
        return self.by_status(Status.FAIL)

    @property
    def warnings(self) -> list[CheckResult]:
        return self.by_status(Status.WARN)

    @property
    def errors(self) -> list[CheckResult]:
        return self.by_status(Status.ERROR)

    @property
    def skips(self) -> list[CheckResult]:
        return self.by_status(Status.SKIP)

    @property
    def passes(self) -> list[CheckResult]:
        return self.by_status(Status.PASS)

    @property
    def unresolved(self) -> list[CheckResult]:
        """PASS results whose details carry ``unresolved=True``: the check
        ran and its evidence sat above a bar, but the conviction gate
        (t-stat, persistence, reconstruction) did not clear. The status
        stays PASS by contract (a PASS is "looked and did not convict",
        never a certificate); the flag says this PASS is not an
        exoneration. :meth:`gate` rejects them by default; callers may make
        the review decision explicit with ``allow_unresolved=True``."""
        return [r for r in self.passes if _flag(r, _FLAG_UNRESOLVED)]

    @property
    def not_judged(self) -> list[CheckResult]:
        """PASS results whose details carry ``not_judged=True``: the check
        declined to judge (delegated its verdict to a sibling, or the
        sample was too short to evaluate). :meth:`gate` strict mode treats
        them like SKIPs - they do not satisfy a ``require`` pattern."""
        return [r for r in self.passes if _flag(r, _FLAG_NOT_JUDGED)]

    @property
    def ok(self) -> bool:
        """True iff nothing failed and nothing crashed.

        Every FAIL counts regardless of severity, so ``ok`` is stricter than
        the default :meth:`raise_for_failures` gate (which ignores FAILs
        below its ``min_severity``): ``ok`` can be False while
        ``raise_for_failures()`` does not raise. Use ``ok`` as the verdict
        and ``raise_for_failures`` as the configurable gate.

        ``ok`` is advisory by design - WARNs and SKIPs never flip it, so
        an all-SKIP report is still ``ok``. A deployment decision that
        must prove specific checks actually ran belongs to :meth:`gate`.
        """
        return (all(_result_contract_problem(r) is None for r in self.results)
                and not self.failures and not self.errors)

    def __getitem__(self, check_id: str) -> CheckResult:
        for r in self.results:
            if r.check == check_id:
                return r
        available = ", ".join(sorted(r.check for r in self.results))
        raise KeyError(f"no check {check_id!r} in report; ran: {available}")

    def find(self, prefix: str) -> list[CheckResult]:
        """All results whose check id starts with ``prefix``."""
        return [r for r in self.results if r.check.startswith(prefix)]

    # -- gating ------------------------------------------------------------
    def raise_for_failures(self, min_severity: Severity = Severity.MEDIUM) -> None:
        """Raise :class:`AuditFailure` if any FAIL at/above ``min_severity``
        (ERROR results count as failures - a crashed check proves nothing).

        FAILs below ``min_severity`` do not raise here but still flip
        :attr:`ok` to False; pass ``min_severity=Severity.INFO`` to make the
        gate exactly as strict as ``ok``."""
        min_severity = Severity(min_severity)
        malformed = [(i, problem)
                     for i, r in enumerate(self.results)
                     if (problem := _result_contract_problem(r)) is not None]
        if malformed:
            detail = "; ".join(f"result[{i}]: {problem}"
                               for i, problem in malformed[:5])
            raise AuditFailure([CheckResult(
                "audit.result_contract_invalid", Status.ERROR,
                f"audit report is structurally invalid: {detail}",
                severity=Severity.HIGH)])
        bad = [r for r in self.failures if r.severity.rank >= min_severity.rank]
        bad += self.errors
        if bad:
            raise AuditFailure(bad)

    def gate(self, *, require: list[str] | None = None,
             min_severity: Severity = Severity.MEDIUM,
             warn_severity: Severity | None = None,
             allow_partial: bool = False,
             allow_unresolved: bool = False) -> None:
        """Fail-closed deployment gate; raises :class:`AuditFailure`.

        ``ok``/``summary()`` are the advisory verdict: they stay green on
        a report where every check SKIPped, because an honest
        minimal-artifact book always carries many SKIPs (e.g.
        ``audit(include=["dynamic"])`` with no callables is "CLEAN - 7
        checks (7 SKIP)", ok=True). A deployment decision must instead
        prove the checks it cares about actually ran. This gate raises
        one exception type for CI to catch, on:

        1. everything ``raise_for_failures(min_severity)`` raises on -
           FAILs at/above ``min_severity`` plus every ERROR;
        2. each ``require`` pattern (matched as prefix or substring of
           the check id, exactly like ``audit(include=...)``) that is
           unmet. A matched result counts as *judged* only when its
           status is WARN, FAIL, or PASS without ``details["not_judged"]``
           or ``details["unresolved"]``. An unresolved PASS deliberately
           does not exonerate a deployment requirement unless the caller
           passes ``allow_unresolved=True``.
           **Strict mode (default, ``allow_partial=False``)**: a pattern
           is unmet if it matches nothing, or if any matched result is a
           SKIP (except SKIPs flagged ``superseded=True`` - intentional,
           a better artifact armed a sibling) or a PASS flagged
           ``not_judged=True``. Every matched check must have run.
           **Permissive mode (``allow_partial=True``)**: a pattern is met
           when at least one matched result is judged, so a family pattern
           passes on one judged sibling while the rest SKIPped. In either
           mode the diagnostic ids (``dynamic.probe_health`` and every
           ``audit.*`` result) never satisfy a pattern: they are
           advisories about the audit, not
           judgments of the backtest. The raise message lists every
           unmet SKIP / not-judged result with its message verbatim
           (each SKIP names the artifact/callable to pass) and, in strict
           mode, says how to satisfy the pattern: supply those artifacts,
           narrow the pattern to exact check ids, or ``allow_partial=True``.
           PASSes flagged ``unresolved=True`` are unmet by default and named
           in the error. ``allow_unresolved=True`` explicitly accepts them;
        3. with ``warn_severity`` given, WARNs at/above it (e.g.
           ``Severity.HIGH`` blocks on high-severity warnings such as the
           synthetic ``audit.coverage`` sliver advisory).

        Example::

            report.gate(require=["lookahead", "leakage", "costs",
                                 "performance.suspicious_sharpe"])
            # an all-SKIP "CLEAN" report fails closed; so does a
            # family whose members partly SKIPped (positions missing)
        """
        if type(allow_partial) is not bool:
            raise InputValidationError(
                f"gate(allow_partial={allow_partial!r}) is invalid: "
                f"allow_partial must be an actual bool, not "
                f"{type(allow_partial).__name__}")
        if type(allow_unresolved) is not bool:
            raise InputValidationError(
                f"gate(allow_unresolved={allow_unresolved!r}) is invalid: "
                f"allow_unresolved must be an actual bool, not "
                f"{type(allow_unresolved).__name__}")
        # str is iterable: require="lookahead" would be matched char by
        # char, each single letter a substring of some id - the gate
        # would silently self-satisfy. Mirror the audit(include=) guard.
        if require is not None and (
                isinstance(require, str)
                or not isinstance(require, (list, tuple))
                or any(not isinstance(p, str) for p in require)
                or any(not p.strip() for p in require)):
            raise InputValidationError(
                f"gate(require={require!r}) is invalid: require must be a "
                f"list/tuple of non-empty check-id patterns (each a str), e.g. "
                f"require=[\"lookahead\", \"leakage\"] - a bare string "
                f"would be matched character by character and an empty "
                f"pattern would match every check.")
        min_severity = Severity(min_severity)
        if warn_severity is not None:
            warn_severity = Severity(warn_severity)

        malformed = [(i, problem)
                     for i, r in enumerate(self.results)
                     if (problem := _result_contract_problem(r)) is not None]
        if not isinstance(self.meta, dict) or malformed:
            pieces: list[str] = []
            if not isinstance(self.meta, dict):
                pieces.append(f"meta is {type(self.meta).__name__}, not dict")
            pieces.extend(f"result[{i}]: {problem}"
                          for i, problem in malformed[:5])
            raise AuditFailure([CheckResult(
                "audit.result_contract_invalid", Status.ERROR,
                "deployment report is structurally invalid: "
                + "; ".join(pieces), severity=Severity.HIGH)])

        bad = [r for r in self.failures
               if r.severity.rank >= min_severity.rank]
        bad += self.errors
        # audit() deliberately preserves diagnostic results if best-effort
        # provenance assembly fails.  A deployment gate must be stricter:
        # it cannot archive a verdict whose inputs/config/code identity are
        # unavailable. Hand-built reports (no provenance key) are gated on
        # their results alone.
        if "provenance" in self.meta:
            provenance = self.meta.get("provenance")
            problem = _provenance_problem(provenance, self.results, self.meta)
            if problem is not None:
                bad.append(CheckResult(
                    "audit.provenance_unavailable", Status.ERROR,
                    f"deployment provenance is unavailable: {problem}",
                    severity=Severity.HIGH))
        if warn_severity is not None:
            bad += [r for r in self.warnings
                    if r.severity.rank >= warn_severity.rank]

        families = sorted({r.check.split(".", 1)[0] for r in self.results})
        for p in (require or []):
            msg = self._require_unmet_message(p, families,
                                              allow_partial=allow_partial,
                                              allow_unresolved=
                                              allow_unresolved)
            if msg is None:
                continue
            # exception payload only, never appended to the report
            bad.append(CheckResult("audit.gate_requirement_unmet",
                                   Status.ERROR, msg,
                                   severity=Severity.HIGH))
        if bad:
            raise AuditFailure(bad)

    def _require_unmet_message(self, p: str, families: list[str], *,
                               allow_partial: bool,
                               allow_unresolved: bool) -> str | None:
        """None when the ``require`` pattern ``p`` is satisfied, else the
        ``audit.gate_requirement_unmet`` message (see :meth:`gate`)."""
        raw_hits = [r for r in self.results
                    if r.check.startswith(p) or p in r.check]
        hits = [r for r in raw_hits if not _is_diagnostic(r.check)]
        expected_hits = [cid for cid in _expected_check_ids(self.meta)
                         if not _is_diagnostic(cid)
                         and (cid.startswith(p) or p in cid)]
        emitted_ids = {r.check for r in hits}
        missing_expected = [cid for cid in expected_hits
                            if cid not in emitted_ids]
        judged = [r for r in hits
                  if r.status in (Status.WARN, Status.FAIL)
                  or (r.status is Status.PASS
                      and not _flag(r, _FLAG_NOT_JUDGED)
                      and (allow_unresolved
                           or not _flag(r, _FLAG_UNRESOLVED)))]
        unmet = [r for r in hits
                 if (r.status is Status.SKIP
                     and not _flag(r, _FLAG_SUPERSEDED))
                 or (r.status is Status.PASS
                     and (_flag(r, _FLAG_NOT_JUDGED)
                          or (not allow_unresolved
                              and _flag(r, _FLAG_UNRESOLVED))))]
        has_unaccepted_unresolved = any(
            r.status is Status.PASS and _flag(r, _FLAG_UNRESOLVED)
            for r in unmet)
        if judged and ((allow_partial and not has_unaccepted_unresolved)
                       or (not unmet and not missing_expected)):
            return None
        n_skip = sum(r.status is Status.SKIP for r in unmet)
        n_nj = sum(r.status is Status.PASS
                   and _flag(r, _FLAG_NOT_JUDGED) for r in unmet)
        n_unresolved = sum(r.status is Status.PASS
                           and _flag(r, _FLAG_UNRESOLVED) for r in unmet)
        mix = " + ".join(x for x in (f"{n_skip} SKIP" if n_skip else "",
                                     f"{n_nj} not-judged PASS" if n_nj
                                     else "",
                                     f"{n_unresolved} unresolved PASS"
                                     if n_unresolved else "") if x)

        def line(r: CheckResult) -> str:
            if r.status is Status.PASS and _flag(r, _FLAG_NOT_JUDGED):
                tag = " (PASS, not judged)"
            elif r.status is Status.PASS and _flag(r, _FLAG_UNRESOLVED):
                tag = " (PASS, unresolved)"
            else:
                tag = ""
            return f"    {r.check}{tag}: {r.message}"

        if not hits and missing_expected:
            detail = "\n".join(
                f"    {cid} (NOT EMITTED): selected by the module registry "
                f"but absent from this report (filtered out or omitted)"
                for cid in missing_expected)
            msg = (f"require pattern {p!r} matched {len(expected_hits)} "
                   f"registered check(s) selected for this audit, but none "
                   f"appear in the report - a result filter or partial module "
                   f"output cannot satisfy a deployment requirement. Missing:"
                   f"\n{detail}")
        elif not hits:
            diag = ""
            if raw_hits:
                diag = (f"; it matched only diagnostic result(s) "
                        f"{', '.join(r.check for r in raw_hits)}, which "
                        f"describe the audit itself and never satisfy a "
                        f"require pattern")
            msg = (f"require pattern {p!r} matched no check in this "
                   f"report ({len(self.results)} result(s); families "
                   f"present: {', '.join(families) or 'none'}){diag} - "
                   f"fix the pattern or audit without include/exclude "
                   f"filters")
        elif not judged and missing_expected:
            # A crashed/collapsed family may emit one ERROR or superseded
            # placeholder while the rest of its selected registry is absent.
            # It already blocks; name every missing result rather than saying
            # misleadingly that zero checks merely SKIPped.
            present = "\n".join(
                f"    {r.check} [{r.status.value.upper()}]: {r.message}"
                for r in hits)
            absent = "\n".join(
                f"    {cid} (NOT EMITTED): selected by the module registry "
                f"but absent from this report"
                for cid in missing_expected)
            msg = (f"require pattern {p!r} matched {len(hits)} emitted "
                   f"check(s), but none judged the backtest and "
                   f"{len(missing_expected)} selected result(s) were not "
                   f"emitted. Present:\n{present}\nMissing:\n{absent}")
        elif not unmet and not missing_expected:
            # superseded SKIPs and/or ERRORs only: nothing judged, nothing
            # to arm - the ERRORs raise on their own; say why the pattern
            # is empty
            detail = "\n".join(
                f"    {r.check} [{r.status.value.upper()}]: {r.message}"
                for r in hits)
            n_err = sum(r.status is Status.ERROR for r in hits)
            hint = ("fix the crash(es) - an ERROR proves nothing"
                    if n_err else
                    "require the check that superseded it instead")
            msg = (f"require pattern {p!r} matched {len(hits)} check(s) "
                   f"but none judged the backtest ({n_err} ERROR, "
                   f"{len(hits) - n_err} superseded SKIP) - {hint}:"
                   f"\n{detail}")
        elif judged:
            # strict mode, partial coverage
            if not missing_expected:
                detail = "\n".join(line(r) for r in unmet)
                msg = (f"require pattern {p!r} matched {len(hits)} check(s) "
                       f"but {len(unmet)} of them never judged the backtest "
                       f"({mix}; {len(judged)} judged) - strict mode requires "
                       f"every matched check to have run. Unmet:\n{detail}\n"
                       f"  To satisfy it: supply the artifacts/callables those "
                       f"messages name, narrow the pattern to the exact ids "
                       f"you intend (judged: "
                       f"{', '.join(repr(r.check) for r in judged)}), or "
                       f"explicitly pass allow_partial=True to "
                       f"accept partial family coverage.")
            else:
                detail_lines = [line(r) for r in unmet]
                detail_lines += [
                    f"    {cid} (NOT EMITTED): selected by the module registry "
                    f"but absent from this report (filtered out or omitted)"
                    for cid in missing_expected]
                detail = "\n".join(detail_lines)
                n_gap = len(unmet) + len(missing_expected)
                missing_clause = (f" + {len(missing_expected)} not-emitted"
                                  if missing_expected else "")
                msg = (f"require pattern {p!r} matched {len(hits)} check(s) "
                       f"but {n_gap} required result(s) did not establish full "
                       f"coverage ({mix or '0 unmet emitted'}{missing_clause}; "
                       f"{len(judged)} judged) - strict mode requires "
                       f"every matched check to have run. Unmet:\n{detail}\n"
                       f"  To satisfy it: supply the artifacts/callables those "
                       f"messages name, narrow the pattern to the exact ids "
                       f"that ran (e.g. {judged[0].check!r}), or pass "
                       f"allow_partial=True to accept a pattern judged by at "
                       f"least one matched check.")
        else:
            detail = "\n".join(line(r) for r in unmet)
            what = ("SKIPped check(s)"
                    if n_nj == 0 and n_unresolved == 0
                    else f"unjudged check(s) ({mix})")
            msg = (f"require pattern {p!r} matched only {what} "
                   f"({len(unmet)}) - the audit never judged it. To arm "
                   f"them, act on:\n{detail}")
        unresolved = [r for r in hits if _flag(r, _FLAG_UNRESOLVED)]
        if unresolved:
            disposition = ("explicitly accepted by allow_unresolved=True"
                           if allow_unresolved else
                           "unmet by default; pass allow_unresolved=True "
                           "only after review")
            msg += (f"\n  Also {len(unresolved)} PASS(es) under {p!r} left "
                    f"unresolved (evidence above a bar, conviction gate not "
                    f"met - not an exoneration; {disposition}): "
                    f"{', '.join(r.check for r in unresolved)}")
        return msg

    # -- rendering -----------------------------------------------------------
    def summary(self) -> str:
        counts = {s: len(self.by_status(s)) for s in _STATUS_ORDER}
        parts = [f"{n} {s.value.upper()}" for s, n in counts.items() if n]
        verdict = "CLEAN" if self.ok else "SUSPECT"
        s = (f"qaudit: {verdict} - {len(self.results)} checks "
             f"({', '.join(parts) if parts else 'none run'})")
        # verdict-flag suffix: PASSes that are not exonerations
        flags = [f"{len(v)} {label}" for v, label in
                 ((self.unresolved, "unresolved"),
                  (self.not_judged, "not judged")) if v]
        if flags:
            s += " - " + ", ".join(flags)
        return s

    def __str__(self) -> str:
        lines = [self.summary()]
        if self.meta:
            n_p = self.meta.get("n_periods")
            n_a = self.meta.get("n_assets")
            if n_p is not None:
                lines.append(f"  sample: {n_p} periods x {n_a} assets")
        for status in _STATUS_ORDER:
            group = sorted(self.by_status(status),
                           key=lambda r: -r.severity.rank)
            for r in group:
                lines.append("  " + str(r).replace("\n", "\n  "))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Report as plain data, guaranteed ``json.dumps``-compatible (even
        with ``allow_nan=False``); see :func:`_json_clean` for the value
        coercions."""
        return {
            "summary": self.summary(),
            "ok": self.ok,
            "meta": _json_clean(self.meta),
            "results": [
                {"check": r.check, "status": r.status.value,
                 "severity": r.severity.value, "message": r.message,
                 "remediation": r.remediation,
                 "details": _json_clean(r.details)}
                for r in self.results
            ],
        }

    def to_html(self, *, title: str = "Backtest validation report") -> str:
        """Standalone HTML with searchable findings and provenance.

        No network access or extra dependencies are required. The report is
        advisory: this method does not call :meth:`gate` or approve deployment.
        Save the returned text with UTF-8 encoding and open it in a browser.
        """
        from datetime import datetime, timezone

        from . import __version__
        from ._html import render_html, report_view

        payload = {
            "schema": "qaudit.report.v1",
            "title": title,
            "qaudit_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cases": [{"name": title,
                       "description": f"{len(self.results)} diagnostic results. "
                                      "Inspect findings and coverage below.",
                       "report": self.to_dict(),
                       "presentation": report_view(self)}],
        }
        return render_html(payload, title=title)

    def to_markdown(self) -> str:
        import html

        def safe_text(value: str, *, line_breaks: bool = True) -> str:
            # Check messages may contain exception text originating in user
            # callables. Raw HTML, images and links here can execute active
            # content or trigger remote tracking in permissive Markdown
            # renderers. Encode HTML and escape Markdown metacharacters before
            # adding our own trusted table/list markup.
            out = html.escape(value, quote=True).replace("\\", "\\\\")
            for char in "`*_{}[]()#+-!>":
                out = out.replace(char, "\\" + char)
            out = out.replace("|", "\\|")
            return out.replace("\n", "<br>") if line_breaks else out

        def safe_check(value: str) -> str:
            # The id is rendered inside code ticks, where ordinary Markdown
            # punctuation is inert. Neutralize only the delimiters/content
            # that can escape that context or split the table.
            return (html.escape(value, quote=True)
                    .replace("\\", "\\\\")
                    .replace("`", "&#96;")
                    .replace("|", "\\|")
                    .replace("\n", " "))

        lines = [f"# Backtest audit - {'CLEAN' if self.ok else 'SUSPECT'}",
                 "", self.summary(), ""]
        lines += ["| status | severity | check | finding |",
                  "|---|---|---|---|"]
        for status in _STATUS_ORDER:
            for r in sorted(self.by_status(status), key=lambda r: -r.severity.rank):
                # newlines end a markdown table row mid-cell (ERROR messages
                # embed exception text verbatim, often multiline for pandas
                # errors); sanitize at this sink only - __str__ indents
                # multiline messages and to_dict keeps the raw string
                msg = safe_text(r.message)
                word = r.status.value.upper()
                if r.status is Status.PASS:
                    if _flag(r, _FLAG_UNRESOLVED):
                        word += " (unresolved)"
                    if _flag(r, _FLAG_NOT_JUDGED):
                        word += " (not judged)"
                check = safe_check(r.check)
                lines.append(f"| {word} | {r.severity.value} "
                             f"| `{check}` | {msg} |")
        # ERROR included for parity with CheckResult.__str__: the synthetic
        # audit.filter_matched_nothing ERROR is the only remediation-carrying
        # ERROR and its fix (the valid family prefixes) must be visible here
        rem = [r for r in self.results
               if r.remediation and r.status in (Status.FAIL, Status.WARN,
                                                 Status.ERROR)]
        if rem:
            lines += ["", "## Remediation", ""]
            for r in rem:
                fix = safe_text(r.remediation)
                check = safe_check(r.check)
                lines.append(f"- **`{check}`** - {fix}")
        return "\n".join(lines)
