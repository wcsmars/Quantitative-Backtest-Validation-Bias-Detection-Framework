"""A complete local first run: simulated market -> backtest -> audit -> JSON.

Run from the repository after installing the library:
    python examples/first_audit.py

The simulated strategy is a learning fixture, not a strategy recommendation.
Inspect the exported CSVs, then substitute artifacts from your own backtest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qaudit import AuditConfig, audit
from qaudit.cli import gate_problem, print_summary, save_report
from qaudit.synthetic import make_clean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/first-audit"))
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"{args.output_dir} already exists; choose a new --output-dir.")

    print("Creating a simulated market and a backtest with a one-day signal delay.", flush=True)
    case = make_clean()
    print("Auditing the saved results and rerunning the pipeline under probes...", flush=True)
    report = audit(case.artifacts, AuditConfig(seed=1, n_trials=1,
                                              n_placebo=50, n_shuffle=50),
                   signal_func=case.signal_func, backtest_func=case.backtest_func)
    problem = gate_problem(report)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        # These are exactly the inputs to this audit, in the CLI's CSV format.
        for name in ("signals", "asset_returns", "positions", "strategy_returns",
                     "universe", "prices"):
            panel = getattr(case.artifacts, name)
            if name == "strategy_returns":
                panel = panel.rename("return")
            panel.to_csv(args.output_dir / f"{name}.csv", index_label="date")
        output = args.output_dir / "report.json"
        save_report(report, output)
    except OSError as exc:
        print(f"Could not save the first audit: {exc}", file=sys.stderr)
        return 2
    print_summary(report, output, problem)
    print(f"Inspect the CSV inputs in {args.output_dir}.")
    return 1 if problem else 0


if __name__ == "__main__":
    raise SystemExit(main())
