"""Exceptions for qaudit.

Every exception message is written to be actionable: it names the offending
artifact/field, quantifies the problem, and says how to fix it.
"""
from __future__ import annotations


class QAuditError(Exception):
    """Base class for all qaudit errors."""


class InputValidationError(QAuditError, ValueError):
    """Backtest artifacts are structurally malformed (wrong type, bad index, infs)."""


class MissingArtifactError(InputValidationError):
    """An artifact required for the requested audit was not provided."""


class MisalignedInputError(InputValidationError):
    """Two artifacts do not share enough dates/assets to audit together."""


class CheckRuntimeError(QAuditError):
    """A check module crashed while running. Only raised with
    ``audit(..., strict=True)``; otherwise the crash is recorded as a
    ``Status.ERROR`` result in the report. Crashes of user-supplied
    callables (signal_func/backtest_func) inside the dynamic probes are
    handled by the probes themselves (ERROR/SKIP results) and are not
    re-raised by strict - strict covers the auditor's own code, not the
    audited pipeline's."""

    def __init__(self, check: str, original: BaseException):
        self.check = check
        self.original = original
        super().__init__(
            f"check {check!r} crashed: {type(original).__name__}: {original}"
        )


class AuditFailure(QAuditError):
    """Raised by :meth:`AuditReport.raise_for_failures` and :meth:`AuditReport.gate`
    when failing checks exist (for ``gate``, also when required checks did not run)."""

    def __init__(self, failures):
        self.failures = list(failures)
        lines = [f"backtest audit failed {len(self.failures)} check(s):"]
        for r in self.failures:
            lines.append(f"  [{r.severity.value}] {r.check}: {r.message}")
            if r.remediation:
                lines.append(f"      fix: {r.remediation}")
        super().__init__("\n".join(lines))
