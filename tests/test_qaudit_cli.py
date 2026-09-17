"""CSV trust boundaries, preserved failure evidence, and local startup."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qaudit import AuditReport, InputValidationError
from qaudit.cli import gate_problem, main, read_panel, read_returns
from qaudit.types import Severity, errored, failed, passed, skipped, warned


def write_csv(tmp_path, text, name="input.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("text,match", [
    ("date,A,A\n2024-01-01,1,2\n", "unique"),
    ("date,A, B\n2024-01-01,1,2\n", "spaces"),
    ("date,A\n2024-01-01,1,2\n", "expected 2 fields"),
    ("date,A,B\n2024-01-01,1\n", "expected 3 fields"),
    ("date,A\n2024-01-01,word\n", "nonnumeric"),
    ("date,A\n01/02/2024,1\n", "ISO dates"),
    ("date,A\n,1\n", "nonempty ISO dates"),
    ("date,A\n", "data rows"),
    ("", "first row"),
])
def test_csv_rejects_ambiguous_inputs_before_alignment(tmp_path, text, match):
    with pytest.raises(InputValidationError, match=match):
        read_panel(write_csv(tmp_path, text))


def test_csv_retains_asset_names_missingness_and_dates(tmp_path):
    path = write_csv(tmp_path, "\ufeffdate,NA,001\n2024-01-01,0.01,\n2024-01-02,-0.02,NaN\n")
    panel = read_panel(path)
    assert list(panel.columns) == ["NA", "001"]
    assert list(panel.index) == list(pd.date_range("2024-01-01", periods=2))
    assert panel["001"].isna().all()
    assert panel["NA"].tolist() == [0.01, -0.02]


def test_csv_accepts_daylight_saving_offsets_from_pandas_export(tmp_path):
    dates = pd.date_range("2024-03-10", periods=2, tz="America/New_York")
    original = pd.DataFrame({"A": [0.01, -0.02]}, index=dates)
    path = tmp_path / "dst.csv"
    original.to_csv(path, index_label="date")

    panel = read_panel(path)

    assert panel.index.equals(dates.tz_convert("UTC").rename("date"))
    assert panel["A"].tolist() == original["A"].tolist()


def test_csv_rejects_mixed_naive_and_aware_dates_without_guessing_timezone(tmp_path):
    path = write_csv(tmp_path, "date,A\n2024-01-01,0.01\n"
                     "2024-01-02T00:00:00+08:00,-0.02\n")
    with pytest.raises(InputValidationError, match="timezone-aware and timezone-naive"):
        read_panel(path)


def test_universe_bool_csv_and_single_return_contract(tmp_path):
    panel = read_panel(write_csv(tmp_path, "date,A,B\n2024-01-01,True,false\n"), universe=True)
    assert panel.iloc[0].tolist() == [1.0, 0.0]
    path = write_csv(tmp_path, "date,return\n2024-01-01,0.01\n")
    assert read_returns(path).iloc[0] == 0.01
    path = write_csv(tmp_path, "date,A,B\n2024-01-01,0.01,0.02\n")
    with pytest.raises(InputValidationError, match="exactly two columns"):
        read_returns(path)


def test_invalid_artifact_index_still_rejected_by_audit(tmp_path, capsys):
    dates = pd.bdate_range("2024-01-01", periods=40)
    panel = pd.DataFrame(np.arange(80).reshape(40, 2) / 1000,
                         index=dates, columns=["A", "B"])
    panel.index = dates[::-1]
    path = tmp_path / "unsorted.csv"
    panel.to_csv(path, index_label="date")
    output = tmp_path / "report.json"
    assert main(["--signals", str(path), "--asset-returns", str(path),
                 "--output", str(output)]) == 2
    assert "not sorted" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("result", [
    failed("costs.test", "unpaid costs", severity=Severity.LOW),
    warned("costs.test", "review needed", severity=Severity.HIGH),
    errored("costs.test", ValueError("broken")),
])
def test_blocked_audit_exports_failed_evidence(tmp_path, monkeypatch, result):
    path = write_csv(tmp_path, "date,A\n2024-01-01,0.01\n")
    report = AuditReport([result])
    monkeypatch.setattr("qaudit.cli.audit", lambda *args: report)
    output = tmp_path / "report.json"
    assert main(["--signals", str(path), "--asset-returns", str(path),
                 "--output", str(output)]) == 1
    assert json.loads(output.read_text()) == report.to_dict()


def test_required_skipped_family_cannot_pass_cli(tmp_path, monkeypatch, capsys):
    path = write_csv(tmp_path, "date,A\n2024-01-01,0.01\n")
    report = AuditReport([passed("costs.test", "ok"), skipped("dynamic.test", "need callback")])
    monkeypatch.setattr("qaudit.cli.audit", lambda *args: report)
    output = tmp_path / "report.json"
    assert main(["--signals", str(path), "--asset-returns", str(path),
                 "--require", "dynamic", "--output", str(output)]) == 1
    assert "need callback" in capsys.readouterr().err
    assert output.is_file()


@pytest.mark.parametrize("results", [[], [skipped("costs.test", "no data")],
                                     [passed("costs.test", "deferred", not_judged=True)],
                                     [passed("dynamic.probe_health", "healthy")]])
def test_empty_or_unjudged_audit_is_not_a_success(results):
    assert gate_problem(AuditReport(results)) is not None


def test_success_with_skips_explains_limited_coverage(tmp_path, monkeypatch, capsys):
    path = write_csv(tmp_path, "date,A\n2024-01-01,0.01\n")
    report = AuditReport([passed("costs.test", "ok"), skipped("dynamic.test", "need callback")])
    monkeypatch.setattr("qaudit.cli.audit", lambda *args: report)
    assert main(["--signals", str(path), "--asset-returns", str(path),
                 "--output", str(tmp_path / "report.json")]) == 0
    text = capsys.readouterr().out
    assert "SKIP means untested" in text
    assert "checks that ran" in text


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch, capsys):
    output = write_csv(tmp_path, "preserve", "report.json")
    def unexpected(*args):
        pytest.fail("Existing output must be checked before audit starts")
    monkeypatch.setattr("qaudit.cli.audit", unexpected)
    assert main(["--signals", "missing.csv", "--asset-returns", "missing.csv",
                 "--output", str(output)]) == 2
    assert output.read_text() == "preserve"
    assert "Output already exists" in capsys.readouterr().err


def test_missing_input_has_actionable_error(tmp_path, capsys):
    assert main(["--signals", str(tmp_path / "missing.csv"), "--asset-returns", "missing.csv",
                 "--output", str(tmp_path / "report.json")]) == 2
    assert "missing.csv" in capsys.readouterr().err


def test_python_module_entrypoint_help():
    # pytest's pythonpath setting is process-local. Put this repository's
    # src/ first so the subprocess exercises the source tree under test
    # (including an uninstalled source archive) rather than any qaudit
    # already installed in the interpreter.
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-m", "qaudit", "--help"],
                            capture_output=True, text=True, check=False, env=env)
    assert result.returncode == 0
    assert "--signals" in result.stdout
    assert "--asset-returns" in result.stdout
