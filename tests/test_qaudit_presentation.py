"""Portable report evidence, HTML injection boundaries, and CLI exit contracts."""
from __future__ import annotations

import json
from html.parser import HTMLParser

import pandas as pd
import pytest

from qaudit import AuditReport, BacktestArtifacts, demo
from qaudit.synthetic import SyntheticBacktest
from qaudit.types import errored, failed, passed, skipped, warned


class ReportDocument(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.scripts = []
        self.active_tags = []
        self.data = ""
        self.in_data = False
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.scripts.append(attrs)
            self.in_data = attrs.get("id") == "qaudit-data"
        if tag in {"img", "iframe", "object", "embed", "link"}:
            self.active_tags.append((tag, attrs))

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False

    def handle_data(self, text):
        if self.in_data:
            self.data += text


def case(name="clean", *, expected=()):
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=dates)
    return SyntheticBacktest(name, "Synthetic test case", BacktestArtifacts(frame, frame),
                             expected_flags=expected)


def test_report_html_preserves_evidence_without_active_content():
    attack = '</script><img src="https://example.invalid/tracker" onerror="alert(1)">'
    report = AuditReport([
        failed("extension.test", attack, remediation=attack,
               details={"label": attack, "nullable": float("nan")})
    ], meta={"provenance": {"note": attack}})
    document = ReportDocument(report.to_html(title=attack + " __PAYLOAD__ __TITLE__"))
    payload = json.loads(document.data)
    assert payload["cases"][0]["report"] == report.to_dict()
    assert payload["title"] == attack + " __PAYLOAD__ __TITLE__"
    assert payload["cases"][0]["report"]["results"][0]["details"]["nullable"] is None
    assert len(document.scripts) == 2
    assert all("src" not in attrs for attrs in document.scripts)
    assert not document.active_tags


def test_demo_exports_same_complete_evidence_in_html_and_json(monkeypatch, tmp_path):
    clean = case()
    report = AuditReport([passed("lookahead.test", "No defect found")])
    monkeypatch.setattr(demo, "run_case", lambda _: (clean, report))
    output = tmp_path / "shareable"
    assert demo.main(["clean", "--summary-only", "--output-dir", str(output)]) == 0
    payload = json.loads((output / "reports.json").read_text())
    document = ReportDocument((output / "index.html").read_text())
    assert json.loads(document.data) == payload
    assert payload["demo_passed"] is True
    assert payload["cases"][0]["report"] == report.to_dict()


@pytest.mark.parametrize("result", [failed("test.fail", "bad"),
                                    errored("test.error", RuntimeError("bad")),
                                    warned("test.warn", "No judged pass"),
                                    skipped("test.skip", "unavailable")])
def test_failed_clean_guard_still_exports_failed_evidence(monkeypatch, tmp_path, result):
    monkeypatch.setattr(demo, "run_case", lambda _: (case(), AuditReport([result])))
    output = tmp_path / "failure"
    assert demo.main(["clean", "--summary-only", "--output-dir", str(output)]) == 1
    payload = json.loads((output / "reports.json").read_text())
    assert payload["demo_passed"] is False
    assert payload["cases"][0]["report"]["results"][0]["status"] == result.status.value


def test_missed_defect_export_cannot_claim_success(monkeypatch, tmp_path):
    missed = case("overfit", expected=("performance.",))
    monkeypatch.setattr(demo, "run_case", lambda _: (missed, AuditReport([
        passed("performance.test", "Missed the planted defect")
    ])))
    output = tmp_path / "missed"
    assert demo.main(["overfit", "--summary-only", "--output-dir", str(output)]) == 1
    payload = json.loads((output / "reports.json").read_text())
    assert payload["demo_passed"] is False
    assert payload["cases"][0]["expected_flags"] == {"performance.": False}


def test_existing_export_is_preserved_without_running_cases(monkeypatch, tmp_path):
    sentinel = tmp_path / "index.html"
    sentinel.write_text("keep my previous report")
    def unexpected_run(_):
        pytest.fail("An existing output path should be rejected before auditing")
    monkeypatch.setattr(demo, "run_case", unexpected_run)
    assert demo.main(["clean", "--output-dir", str(tmp_path)]) == 2
    assert sentinel.read_text() == "keep my previous report"
    assert not (tmp_path / "reports.json").exists()


def test_export_io_error_returns_nonzero(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(demo, "run_case", lambda _: (case(), AuditReport([
        passed("lookahead.test", "ok")
    ])))
    # The pre-flight rejects a regular-file parent (exit 2, before any case
    # runs; see tests/test_qaudit_demo_cli.py), so this test induces the
    # export-time OSError with a blocker that appears after the pre-flight:
    # the post-run mkdir is the authoritative guard.
    blocking_file = tmp_path / "file"
    real_export = demo.export_reports

    def export_after_race(rows, destination, **kwargs):
        blocking_file.write_text("not a directory")
        return real_export(rows, destination, **kwargs)

    monkeypatch.setattr(demo, "export_reports", export_after_race)
    assert demo.main(["clean", "--summary-only", "--output-dir",
                      str(blocking_file / "report")]) == 1
    assert "could not export reports" in capsys.readouterr().err


@pytest.mark.parametrize("results", [[], [skipped("test.skip", "no artifact")],
                                    [passed("test.pass", "deferred", not_judged=True)],
                                    [passed("test.pass", "inconclusive", unresolved=True)]])
def test_demo_does_not_present_unjudged_reports_as_clean(results):
    assert demo.display_status(AuditReport(results)) == "INCOMPLETE"


def test_html_uses_presentation_verdict_for_warning_only_report():
    report = AuditReport([warned("test.warn", "Review required")])
    payload = json.loads(ReportDocument(report.to_html()).data)
    rendered = payload["cases"][0]
    assert rendered["presentation"]["status"] == "FLAGGED"
    assert "CLEAN" not in rendered["description"]


def test_html_preserves_reserved_flags_in_typed_json_mappings():
    report = AuditReport([passed("test.pass", "No judgment", details={
        "not_judged": True, 1: "number", "1": "string"
    })])
    rendered = json.loads(ReportDocument(report.to_html()).data)["cases"][0]
    assert "__qaudit_typed_mapping_v1__" in rendered["report"]["results"][0]["details"]
    assert rendered["presentation"]["status"] == "INCOMPLETE"
    assert rendered["presentation"]["judged_passes"] == 0
    assert rendered["presentation"]["result_flags"][0]["not_judged"] is True


def test_html_cannot_bless_malformed_reserved_flags():
    report = AuditReport([passed("test.pass", "Malformed", not_judged=0)])
    assert not report.ok
    rendered = json.loads(ReportDocument(report.to_html()).data)["cases"][0]
    assert rendered["presentation"]["status"] == "ERROR"
    assert rendered["presentation"]["judged_passes"] == 0
    assert rendered["presentation"]["contract_problems"]
