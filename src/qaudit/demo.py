"""Audit synthetic backtests containing known defects and a clean control.

Run ``python -m qaudit.demo --help`` for cases and export options.
The demo returns a nonzero status if a planted defect is missed, a check
errors, or the clean control fails or has no judged pass. HTML and JSON
exports preserve the evidence from successful and unsuccessful runs.
"""
from __future__ import annotations

import os
import sys

# Direct script execution puts the package directory on sys.path, where
# qaudit/types.py can shadow the standard-library types module. Replace
# that entry with src/ before importing modules that depend on types.
if __package__ in (None, ""):
    _PKG_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or os.curdir) != _PKG_DIR]
    sys.path.insert(0, os.path.dirname(_PKG_DIR))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from qaudit import AuditConfig, __version__, audit, synthetic  # noqa: E402
from qaudit._html import render_html, report_view  # noqa: E402
from qaudit.report import AuditReport  # noqa: E402
from qaudit.types import Status  # noqa: E402

#: statuses that count as "the auditor flagged it"
FLAGGING = (Status.FAIL, Status.WARN)

RULE = "=" * 78
THIN = "-" * 78


def run_case(name: str):
    """Build one synthetic case and audit it with the demo config."""
    case = synthetic.ALL_CASES[name]()
    n_trials = case.audit_kwargs.get("n_trials")
    if n_trials is not None and (
            isinstance(n_trials, bool) or not isinstance(n_trials, int)):
        raise TypeError(
            f"synthetic case {name!r} has invalid n_trials={n_trials!r}")
    config = AuditConfig(seed=1, n_placebo=50, n_shuffle=50,
                         n_trials=n_trials)
    report = audit(case.artifacts, config,
                   signal_func=case.signal_func,
                   backtest_func=case.backtest_func)
    return case, report


def flag_hits(case, report) -> dict[str, bool]:
    """expected_flags prefix -> was any matching check FAIL/WARN?"""
    return {
        prefix: any(r.check.startswith(prefix) and r.status in FLAGGING
                    for r in report.results)
        for prefix in case.expected_flags
    }


def display_status(report) -> str:
    """Presentation status for the demo, distinct from advisory report.ok."""
    return str(report_view(report)["status"])


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface used by the module and entry point."""
    parser = argparse.ArgumentParser(
        prog="qaudit-demo",
        description="Audit every synthetic defect case (or the named subset) "
                    "and check that each planted defect was caught.",
        epilog="cases: " + ", ".join(synthetic.ALL_CASES) + "\n\n"
               "exit status: 0 if demo guards pass; 1 for failed or unjudged "
               "clean control, check errors, missed flags, or export failure; "
               "2 for invalid arguments or an unusable output directory; "
               "130 if interrupted (Ctrl-C).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="*", metavar="CASE",
                        help="synthetic case name(s) to run (default: all "
                             "cases, in the order listed below)")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="suppress full per-check reports and print only the catch matrix",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_cases",
        help="list the available synthetic cases and exit",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--output-dir", type=Path, metavar="PATH",
        help="write an offline HTML report and JSON evidence to a new directory",
    )
    return parser


def output_dir_problem(destination: Path) -> str | None:
    """Explain why ``destination`` cannot receive the export, or ``None``.

    Cheap pre-flight run before the minutes of dynamic probes so an unusable
    path is reported at once (exit 2) rather than after the whole run as an
    export failure (exit 1) with the evidence lost. ``exists()`` follows
    symlinks, so a dangling link is checked with ``is_symlink()`` too; the
    nearest existing ancestor must be a writable directory. The
    ``mkdir(exist_ok=False)`` in :func:`export_reports` stays the
    authoritative guard against races and anything this probe cannot see.
    """
    if destination.is_dir():
        return (f"output directory already exists: {destination}; "
                "choose a new path to preserve previous reports")
    if destination.exists() or destination.is_symlink():
        return (f"output path exists and is not a directory: {destination}; "
                "choose a new path")
    ancestor = destination.absolute().parent
    while not (ancestor.exists() or ancestor.is_symlink()):
        if ancestor.parent == ancestor:
            break
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        return (f"cannot create output directory {destination}: "
                f"{ancestor} is not a directory")
    if not os.access(ancestor, os.W_OK):
        return (f"cannot create output directory {destination}: "
                f"{ancestor} is not writable")
    return None


def export_reports(rows: list[tuple[synthetic.SyntheticBacktest, AuditReport,
                                   dict[str, bool]]],
                   destination: Path, *, demo_passed: bool) -> None:
    """Write both demo artifacts; never overwrite an existing directory."""
    payload = {
        "schema": "qaudit.demo.v1",
        "qaudit_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "demo_passed": demo_passed,
        "cases": [{"name": case.name, "description": case.description,
                   "expected_flags": hits, "report": report.to_dict(),
                   "presentation": report_view(report)}
                  for case, report, hits in rows],
    }
    # Serialize before creating output so a malformed report leaves no files.
    json_text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    html_text = render_html(payload, title="Quantitative Backtest Validation & Bias Detection")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "reports.json").write_text(json_text, encoding="utf-8")
    (destination / "index.html").write_text(html_text, encoding="utf-8")


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)   # -h/--help exits 0 here
    if args.list_cases:
        print("\n".join(synthetic.ALL_CASES))
        return 0

    requested = args.cases or list(synthetic.ALL_CASES)
    unknown = [n for n in requested if n not in synthetic.ALL_CASES]
    if unknown:
        print(f"unknown case(s): {', '.join(unknown)}\n"
              f"available: {', '.join(synthetic.ALL_CASES)}", file=sys.stderr)
        return 2
    # Each case is seeded, so a repeated name would only re-run the same
    # audit, double the runtime and inflate the summary/JSON/HTML counts:
    # run each once, in the order first listed.
    names = list(dict.fromkeys(requested))
    if len(names) < len(requested):
        repeated = [n for n in names if requested.count(n) > 1]
        print(f"ignoring repeated case(s): {', '.join(repeated)}",
              file=sys.stderr)
    if args.output_dir is not None:
        problem = output_dir_problem(args.output_dir)
        if problem is not None:
            print(problem, file=sys.stderr)
            return 2

    rows = []
    t_start = time.perf_counter()
    for i, name in enumerate(names, 1):
        # Progress goes to stderr in both modes: a --summary-only run of all
        # cases would otherwise print nothing for the whole run (about a
        # minute), which reads as a hang. stdout keeps only the results.
        print(f"[{i}/{len(names)}] auditing {name} ...", end="",
              file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        case, report = run_case(name)
        dt = time.perf_counter() - t0
        print(f" {dt:.1f}s", file=sys.stderr, flush=True)

        hits = flag_hits(case, report)
        if not args.summary_only:
            print(RULE)
            print(f"case: {case.name}  ({dt:.1f}s)")
            print(f"  {case.description}")
            print(THIN)
            print(report)
            print(THIN)
            if hits:
                for prefix, ok in hits.items():
                    print(f"  expected flag {prefix!r}: "
                          f"{'CAUGHT' if ok else 'MISSED'}")
            else:
                print("  no flags expected (false-positive guard)")
            print()
        rows.append((case, report, hits))

    # ---- summary matrix ----------------------------------------------------
    print(RULE)
    print(f"summary ({time.perf_counter() - t_start:.0f}s total)")
    print(f"{'case':<22} {'caught/expected':>16}  demo status")
    print(THIN)
    for case, report, hits in rows:
        caught, expected = sum(hits.values()), len(hits)
        verdict = display_status(report)
        note = "" if caught == expected else "   <-- MISSED"
        print(f"{case.name:<22} {f'{caught}/{expected}':>16}  {verdict}{note}")

    rc = 0
    for case, report, hits in rows:
        # ERROR results on any case break the demo's contract: a crashed
        # check proves nothing, so the smoke gate goes red even when no
        # FAIL was recorded.
        errs = report.errors
        if errs:
            print(f"auditor broken: case {case.name!r} produced "
                  f"{len(errs)} ERROR result(s): "
                  f"{', '.join(r.check for r in errs)}", file=sys.stderr)
            rc = 1
        if case.name == "clean":
            fails = report.failures
            incomplete = not report_view(report)["judged_passes"]
            if fails or incomplete:
                # explicit branch, not `assert`: PYTHONOPTIMIZE/-O strips
                # asserts, which would void the documented nonzero-exit
                # contract and print the OK line below with a FAIL present
                print("false-positive guard violated: 'clean' produced "
                      f"{len(fails)} FAIL(s): "
                      f"{', '.join(r.check for r in fails)}"
                      + ("; no check established a judged PASS" if incomplete else ""),
                      file=sys.stderr)
                rc = 1
            elif not errs:
                print(f"clean false-positive guard: OK ({len(report.passes)} "
                      f"PASS, {len(report.warnings)} WARN, "
                      f"{len(report.skips)} SKIP)")
        elif hits and sum(hits.values()) < len(hits):
            missed = [p for p, ok in hits.items() if not ok]
            print(f"detection guard violated: case {case.name!r} missed "
                  f"{len(missed)} of {len(hits)} expected flag(s): "
                  f"{', '.join(missed)}", file=sys.stderr)
            rc = 1
    if args.output_dir is not None:
        try:
            export_reports(rows, args.output_dir, demo_passed=rc == 0)
        except (OSError, TypeError, ValueError) as exc:
            print(f"could not export reports: {exc}", file=sys.stderr)
            return 1
        print(f"HTML report: {(args.output_dir / 'index.html').resolve()}")
        print(f"JSON evidence: {(args.output_dir / 'reports.json').resolve()}")
    return rc


def cli() -> int:
    """Console-script entry point installed as :command:`qaudit-demo`.

    An uncaught Ctrl-C surfaces as a long traceback through the auditor's
    own check code, which reads as "the auditor crashed"; report it in one
    line and exit 130 (the conventional SIGINT status) instead. Only ``cli``
    catches it, so ``main()`` keeps raising for tests and for pytest's own
    abort.
    """
    try:
        return main(sys.argv[1:])
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(cli())
