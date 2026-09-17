"""Quantitative Backtest Validation & Bias Detection.

Checks equity backtests for potential lookahead bias, target
leakage, train/test contamination, survivorship bias, missing transaction
costs, unrealistic turnover, unstable IC, and overfitting via trial
multiplicity, plus dynamic probes (placebo signals, shuffled labels,
date shifts, rolling-window truncation) that re-run your actual pipeline.

Quick start::

    from qaudit import audit, BacktestArtifacts

    report = audit(
        BacktestArtifacts(signals=sig, asset_returns=rets,
                          positions=pos, strategy_returns=net,
                          signal_input=raw),  # the raw input my_signal_func reads
        signal_func=my_signal_func,      # optional; with signal_input, unlocks the
                                         # reproducibility/truncation probes
        backtest_func=my_backtest_func,  # optional, unlocks placebo/shuffle/shift
    )
    print(report)
    report.raise_for_failures()   # gate a research pipeline on it
"""
from .api import audit, CHECK_MODULES
from .config import AuditConfig
from .errors import (AuditFailure, CheckRuntimeError, InputValidationError,
                     MisalignedInputError, MissingArtifactError, QAuditError)
from .inputs import BacktestArtifacts
from .report import AuditReport
from .types import CheckResult, Severity, Status

__version__ = "0.1.0"
__author__ = "Chung Shing Mars Wong"

__all__ = [
    "audit", "AuditConfig", "BacktestArtifacts", "AuditReport", "CheckResult",
    "Severity", "Status", "CHECK_MODULES",
    "QAuditError", "InputValidationError", "MisalignedInputError",
    "MissingArtifactError", "CheckRuntimeError", "AuditFailure",
]
