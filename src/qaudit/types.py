"""Core result types shared by every check module."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """How bad a finding is *if the check fails*.

    CRITICAL  the backtest result is almost certainly invalid
    HIGH      strong evidence the result is inflated or unreliable
    MEDIUM    material risk; investigate before trusting the result
    LOW       hygiene issue; unlikely to change conclusions alone
    INFO      informational only
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
}


class Status(str, Enum):
    PASS = "pass"    # check ran, no problem found
    WARN = "warn"    # suspicious but not conclusive
    FAIL = "fail"    # problem detected
    SKIP = "skip"    # check could not run (missing artifact/function); message says what to pass
    ERROR = "error"  # check itself crashed; message carries the exception


@dataclass
class CheckResult:
    """Outcome of one diagnostic check.

    ``check`` is a stable dotted id, e.g. ``"lookahead.same_bar_bleed"``.
    ``message`` is a one-line human summary that includes the concrete numbers.
    ``details`` holds machine-readable evidence (series, curves, counts).
    ``remediation`` says what to change in the research pipeline.
    """

    check: str
    status: Status
    message: str
    severity: Severity = Severity.MEDIUM
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str = ""

    def __post_init__(self) -> None:
        self.status = Status(self.status)
        self.severity = Severity(self.severity)

    def __str__(self) -> str:
        tag = f"[{self.status.value.upper():5s}|{self.severity.value}]"
        s = f"{tag} {self.check} - {self.message}"
        # ERROR included: errored() never sets a remediation, so exception
        # noise grows no "fix:" line - but the synthetic
        # audit.filter_matched_nothing ERROR carries the valid-family-prefix
        # list, which must reach print(report), not just raise_for_failures
        if self.remediation and self.status in (Status.FAIL, Status.WARN,
                                                Status.ERROR):
            s += f"\n        fix: {self.remediation}"
        return s


# ---------------------------------------------------------------------------
# Factory helpers so check modules stay terse and consistent.
#
# Each factory accepts evidence either as an explicit ``details`` dict, as
# individual ``**extra`` keywords, or both.  The two are merged into
# ``CheckResult.details`` with the ``details`` dict first; explicit keyword
# extras win on key collision.
# ---------------------------------------------------------------------------

def passed(check: str, message: str, *, severity: Severity = Severity.INFO,
           remediation: str = "", details: dict[str, Any] | None = None,
           **extra: Any) -> CheckResult:
    """PASS result. ``details`` merges with ``**extra``; extras win on collision."""
    return CheckResult(check, Status.PASS, message, severity=severity,
                       details={**(details or {}), **extra},
                       remediation=remediation)


def warned(check: str, message: str, *, severity: Severity = Severity.MEDIUM,
           remediation: str = "", details: dict[str, Any] | None = None,
           **extra: Any) -> CheckResult:
    """WARN result. ``details`` merges with ``**extra``; extras win on collision."""
    return CheckResult(check, Status.WARN, message, severity=severity,
                       details={**(details or {}), **extra},
                       remediation=remediation)


def failed(check: str, message: str, *, severity: Severity = Severity.HIGH,
           remediation: str = "", details: dict[str, Any] | None = None,
           **extra: Any) -> CheckResult:
    """FAIL result. ``details`` merges with ``**extra``; extras win on collision."""
    return CheckResult(check, Status.FAIL, message, severity=severity,
                       details={**(details or {}), **extra},
                       remediation=remediation)


def skipped(check: str, message: str, *, details: dict[str, Any] | None = None,
            **extra: Any) -> CheckResult:
    """SKIP result. ``details`` merges with ``**extra``; extras win on collision."""
    return CheckResult(check, Status.SKIP, message, severity=Severity.INFO,
                       details={**(details or {}), **extra})


def errored(check: str, exc: BaseException) -> CheckResult:
    return CheckResult(check, Status.ERROR,
                       f"check crashed: {type(exc).__name__}: {exc}",
                       severity=Severity.HIGH)
