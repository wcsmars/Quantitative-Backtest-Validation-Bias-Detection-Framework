"""Local CSV entry point. Data parsing, audit checks, and reporting stay separate.

No market data is fetched and no strategy is executed by this command. CSV
files describe the artifacts of a backtest the caller has already run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

from . import AuditConfig, AuditFailure, AuditReport, BacktestArtifacts
from . import InputValidationError, Severity, Status, __version__, audit
from .errors import QAuditError


def read_panel(path: Path, *, universe: bool = False) -> pd.DataFrame:
    """Read a numeric date-by-asset CSV without silently repairing its axes.

    Parse headers ourselves: pandas.read_csv normally renames duplicate
    tickers and can infer an index from extra fields, hiding corrupt inputs.
    Only blank cells and the explicit token NaN represent missing numbers.
    Explicit timezone offsets are normalized to UTC (including DST changes);
    dates without a timezone remain naive and cannot be mixed with aware dates.
    """
    dates: list[str] = []
    values: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle, strict=True)
        header = next(rows, [])
        if len(header) < 2 or header[0] != "date":
            raise InputValidationError(
                f"{path}: first row must be date,TICKER,... (at least one asset).")
        columns = header[1:]
        if (len(columns) != len(set(columns))
                or any(not c or c != c.strip() for c in columns)):
            raise InputValidationError(
                f"{path}: asset names must be unique, nonempty, and have no "
                "leading or trailing spaces.")
        for line_number, row in enumerate(rows, 2):
            if len(row) != len(header):
                raise InputValidationError(
                    f"{path}:{line_number}: expected {len(header)} fields, "
                    f"got {len(row)}. Do not omit trailing missing cells.")
            dates.append(row[0])
            parsed: list[float] = []
            for column, cell in zip(columns, row[1:]):
                token = cell.strip()
                if not token or token.lower() == "nan":
                    value = float("nan")
                elif universe and token.lower() in {"true", "false"}:
                    value = float(token.lower() == "true")
                else:
                    try:
                        value = float(token)
                    except ValueError:
                        raise InputValidationError(
                            f"{path}:{line_number}: {column!r} has nonnumeric "
                            f"value {cell!r}; use decimal numbers or blank "
                            "cells for missing data.") from None
                parsed.append(value)
            values.append(parsed)
    if not dates or any(not d.strip() for d in dates):
        raise InputValidationError(f"{path}: data rows need nonempty ISO dates.")
    try:
        # A valid pandas CSV export can contain both -05:00 and -04:00
        # across daylight saving time. Parse in UTC first to avoid pandas'
        # mixed-offset warning/error; separately reject missing offsets so
        # a naive row is never silently guessed to mean UTC.
        index = pd.DatetimeIndex(pd.to_datetime(
            dates, format="ISO8601", errors="raise", utc=True))
        awareness = {pd.Timestamp(value).tzinfo is not None for value in dates}
        if len(awareness) > 1:
            raise ValueError("timezone-aware and timezone-naive dates are mixed; "
                             "use an explicit UTC offset on every row or none")
        if awareness == {False}:
            index = index.tz_localize(None)
    except (ValueError, TypeError) as exc:
        raise InputValidationError(
            f"{path}: use ISO dates such as 2024-01-31, with one consistent "
            f"timezone convention: {exc}") from None
    if index.hasnans:
        raise InputValidationError(f"{path}: every row must have a valid ISO date.")
    return pd.DataFrame(values, index=index.rename("date"), columns=columns)


def read_returns(path: Path) -> pd.Series:
    frame = read_panel(path)
    if list(frame.columns) != ["return"]:
        raise InputValidationError(
            f"{path}: strategy returns need exactly two columns: date,return.")
    return frame["return"]


def gate_problem(report: AuditReport, require: list[str] | None = None) -> str | None:
    """Return an actionable failure, including audits with no judged checks."""
    try:
        report.gate(require=require, min_severity=Severity.INFO,
                    warn_severity=Severity.HIGH)
    except AuditFailure as exc:
        return str(exc)
    judged = [r for r in report.results
              if not (r.check.startswith("audit.") or r.check == "dynamic.probe_health")
              and r.status in (Status.PASS, Status.WARN, Status.FAIL)
              and not r.details.get("not_judged", False)
              and not r.details.get("unresolved", False)]
    if not judged:
        return "No backtest checks reached a judgment; supply the missing artifacts."
    return None


def save_report(report: AuditReport, path: Path) -> None:
    """Save complete JSON evidence, preserving any existing file."""
    payload = json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def print_summary(report: AuditReport, path: Path, problem: str | None) -> None:
    counts = ", ".join(f"{sum(r.status is s for r in report.results)} {s.value}"
                       for s in Status)
    print(f"Audit complete: {counts}")
    print(f"Full evidence and suggested fixes: {path}")
    for result in report.warnings:
        if result.severity.rank < Severity.HIGH.rank:
            print(f"Review {result.check}: {result.message}")
    if problem:
        print(problem, file=sys.stderr)
    else:
        print("No blocking findings in the checks that ran.")
    print("SKIP means untested, not passed. In the CSV command, use --require "
          "to enforce needed coverage.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qaudit",
        description="Audit backtest CSV exports from the command line.",
        epilog="Exit codes: 0 = no blocking findings in judged checks; "
               "1 = findings, check errors or unmet coverage; 2 = invalid input "
               "or output error. CSV audits cannot run pipeline callback probes. "
               "Timezone-aware CSV dates are converted to UTC; use explicit "
               "UTC offsets in train/test bounds when local time matters.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--signals", required=True, type=Path,
                        help="CSV with date,TICKER,...; signals known at date's end")
    parser.add_argument("--asset-returns", required=True, type=Path,
                        help="CSV of period-ending decimal returns (0.01 = 1%%)")
    for name in ("positions", "strategy-returns", "universe", "prices"):
        parser.add_argument(f"--{name}", type=Path,
                            help="date,return CSV" if name == "strategy-returns"
                            else "optional date,TICKER,... CSV")
    parser.add_argument("--cost-bps", type=float, help="one-way costs actually charged")
    parser.add_argument("--trials", type=int, help="total strategy configurations tried")
    parser.add_argument("--signal-lag", type=int, default=1)
    parser.add_argument("--label-horizon", type=int, default=1)
    parser.add_argument("--periods-per-year", type=float, default=252)
    parser.add_argument("--train-period", nargs=2, metavar=("START", "END"))
    parser.add_argument("--test-period", nargs=2, metavar=("START", "END"))
    parser.add_argument("--require", action="append", default=[], metavar="CHECK",
                        help="require every check matching an ID/family to run; repeatable")
    parser.add_argument("--output", type=Path, default=Path("results/audit.json"),
                        help="new JSON file (default: results/audit.json; no overwrite)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise InputValidationError(
                f"Output already exists: {args.output}. Choose a new --output path.")
        if any(not pattern.strip() for pattern in args.require):
            raise InputValidationError("--require needs a nonempty check ID or family.")
        artifacts = BacktestArtifacts(
            signals=read_panel(args.signals),
            asset_returns=read_panel(args.asset_returns),
            positions=read_panel(args.positions) if args.positions else None,
            strategy_returns=read_returns(args.strategy_returns)
                if args.strategy_returns else None,
            universe=read_panel(args.universe, universe=True) if args.universe else None,
            prices=read_panel(args.prices) if args.prices else None,
            declared_costs_bps=args.cost_bps,
            signal_lag=args.signal_lag, label_horizon=args.label_horizon,
            periods_per_year=args.periods_per_year,
            train_period=tuple(args.train_period) if args.train_period else None,
            test_period=tuple(args.test_period) if args.test_period else None,
        )
        report = audit(artifacts, AuditConfig(n_trials=args.trials))
        problem = gate_problem(report, args.require)
        # Keep failed findings too: a nonzero audit should still be reviewable.
        save_report(report, args.output)
        print_summary(report, args.output, problem)
        return 1 if problem else 0
    except (QAuditError, OSError, UnicodeError, csv.Error) as exc:
        print(f"Could not run audit: {exc}", file=sys.stderr)
        return 2


def cli() -> None:
    raise SystemExit(main())
