"""Demo CLI progress, interruption, case selection, and output-directory validation.

Progress is emitted per case; interruption exits 130. Repeated cases run once
in input order, and unusable output paths fail before an audit starts.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from qaudit import AuditReport, BacktestArtifacts, demo
from qaudit.synthetic import SyntheticBacktest
from qaudit.types import passed

REPO = Path(__file__).resolve().parents[1]


def case(name="clean"):
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=dates)
    return SyntheticBacktest(name, "Synthetic test case",
                             BacktestArtifacts(frame, frame), expected_flags=())


@pytest.fixture
def fake_runs(monkeypatch):
    """Replace run_case with an instant stand-in; returns the call log."""
    calls: list[str] = []

    def run(name):
        calls.append(name)
        return case(name), AuditReport([passed("lookahead.test", "ok")])

    monkeypatch.setattr(demo, "run_case", run)
    return calls


# 1. progress on stderr, results on stdout -----------------------------------

def test_summary_only_reports_progress_per_case_on_stderr(fake_runs, capsys):
    assert demo.main(["clean", "overfit", "--summary-only"]) == 0
    out, err = capsys.readouterr()
    assert "[1/2] auditing clean ..." in err
    assert "[2/2] auditing overfit ..." in err
    assert err.count("s\n") == 2                    # one elapsed time per case
    # stdout contract: no progress markers, just the table
    assert "auditing" not in out
    assert "summary (" in out
    assert fake_runs == ["clean", "overfit"]


def test_verbose_mode_also_reports_progress(fake_runs, capsys):
    assert demo.main(["clean"]) == 0
    out, err = capsys.readouterr()
    assert "[1/1] auditing clean ..." in err
    assert "case: clean" in out


# 2. Ctrl-C ------------------------------------------------------------------

def test_cli_turns_keyboard_interrupt_into_exit_130(monkeypatch, capsys):
    def interrupted(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(demo, "main", interrupted)
    monkeypatch.setattr(sys, "argv", ["qaudit-demo", "clean"])
    assert demo.cli() == 130
    err = capsys.readouterr().err
    assert "interrupted" in err
    assert "Traceback" not in err


def test_main_still_propagates_keyboard_interrupt(monkeypatch):
    # only cli() swallows it: pytest's own Ctrl-C abort must keep working
    def interrupted(name):
        raise KeyboardInterrupt

    monkeypatch.setattr(demo, "run_case", interrupted)
    with pytest.raises(KeyboardInterrupt):
        demo.main(["clean", "--summary-only"])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGINT semantics")
def test_sigint_mid_run_exits_130_without_traceback():
    # end-to-end through the console entry point: the progress marker must
    # arrive before the case finishes (it is what we wait for), and SIGINT
    # during the audit must yield one line, exit 130, no traceback
    code = ("import sys, time\n"
            "from qaudit import demo\n"
            "demo.run_case = lambda name: time.sleep(120)\n"
            "sys.argv = ['qaudit-demo', 'clean', '--summary-only']\n"
            "sys.exit(demo.cli())\n")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=str(REPO), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stderr is not None
        seen = b""
        deadline = time.monotonic() + 120
        while b"auditing clean ..." not in seen:
            assert time.monotonic() < deadline, seen
            chunk = proc.stderr.read(1)
            assert chunk, f"demo exited early: {seen!r}"
            seen += chunk
        proc.send_signal(signal.SIGINT)
        out, rest = proc.communicate(timeout=120)
    finally:
        proc.kill()
    err = (seen + rest).decode()
    assert proc.returncode == 130, err
    assert "interrupted" in err
    assert "Traceback" not in err
    assert out == b""


def test_help_documents_the_interrupt_exit_status(capsys):
    with pytest.raises(SystemExit):
        demo.main(["--help"])
    assert "130 if interrupted" in capsys.readouterr().out


# 3. repeated case names -----------------------------------------------------

def test_repeated_case_names_run_once_in_first_listed_order(fake_runs, capsys, tmp_path):
    output = tmp_path / "dup"
    assert demo.main(["overfit", "clean", "overfit", "clean", "--summary-only",
                      "--output-dir", str(output)]) == 0
    out, err = capsys.readouterr()
    assert fake_runs == ["overfit", "clean"]
    assert "ignoring repeated case(s): overfit, clean" in err
    payload = json.loads((output / "reports.json").read_text())
    assert [c["name"] for c in payload["cases"]] == ["overfit", "clean"]
    assert out.count("overfit ") == 1


def test_distinct_case_names_are_not_reported_as_repeated(fake_runs, capsys):
    assert demo.main(["clean", "overfit", "--summary-only"]) == 0
    assert "repeated" not in capsys.readouterr().err
    assert fake_runs == ["clean", "overfit"]


def test_unknown_name_still_wins_over_dedup(fake_runs, capsys):
    assert demo.main(["clean", "clean", "bogus"]) == 2
    assert "unknown case(s): bogus" in capsys.readouterr().err
    assert fake_runs == []


# 4. output-dir pre-flight ---------------------------------------------------

def test_output_dir_that_is_a_regular_file_is_named_as_such(fake_runs, capsys, tmp_path):
    blocker = tmp_path / "reports.json"
    blocker.write_text("x")
    assert demo.main(["clean", "--summary-only", "--output-dir", str(blocker)]) == 2
    err = capsys.readouterr().err
    assert "output path exists and is not a directory" in err
    assert "already exists" not in err
    assert fake_runs == []
    assert blocker.read_text() == "x"


def test_existing_directory_keeps_the_preserve_message(fake_runs, capsys, tmp_path):
    assert demo.main(["clean", "--summary-only", "--output-dir", str(tmp_path)]) == 2
    assert "output directory already exists" in capsys.readouterr().err
    assert fake_runs == []


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges")
def test_dangling_symlink_output_dir_is_rejected_before_any_case_runs(fake_runs, capsys,
                                                                     tmp_path):
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "nowhere")
    assert demo.main(["clean", "--summary-only", "--output-dir", str(link)]) == 2
    assert "output path exists and is not a directory" in capsys.readouterr().err
    assert fake_runs == []
    assert not (tmp_path / "nowhere").exists()


def test_non_directory_parent_is_rejected_before_any_case_runs(fake_runs, capsys, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    assert demo.main(["clean", "--summary-only", "--output-dir",
                      str(blocker / "deeper" / "report")]) == 2
    assert "is not a directory" in capsys.readouterr().err
    assert fake_runs == []


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0,
                    reason="permission bits are advisory for root / on Windows")
def test_unwritable_parent_is_rejected_before_any_case_runs(fake_runs, capsys, tmp_path):
    parent = tmp_path / "ro"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        assert demo.main(["clean", "--summary-only", "--output-dir",
                          str(parent / "child")]) == 2
        assert "is not writable" in capsys.readouterr().err
        assert fake_runs == []
    finally:
        parent.chmod(0o700)


def test_fresh_nested_path_under_a_writable_directory_still_exports(fake_runs, tmp_path):
    output = tmp_path / "a" / "b" / "report"
    assert demo.main(["clean", "--summary-only", "--output-dir", str(output)]) == 0
    assert (output / "index.html").exists()
    assert (output / "reports.json").exists()
    assert fake_runs == ["clean"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges")
def test_symlinked_parent_writes_inside_the_link_target(fake_runs, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    assert demo.main(["clean", "--summary-only", "--output-dir", str(link / "out")]) == 0
    assert (target / "out" / "reports.json").exists()


def test_output_dir_problem_is_none_for_a_usable_path(tmp_path):
    assert demo.output_dir_problem(tmp_path / "new") is None
