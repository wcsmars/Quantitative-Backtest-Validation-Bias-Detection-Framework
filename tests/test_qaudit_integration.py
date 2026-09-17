"""Cross-module integration tests for qaudit.

Runs every synthetic case in :data:`qaudit.synthetic.ALL_CASES` through the
full ``qaudit.audit`` pipeline exactly once (module-scoped fixture) and
verifies, end to end:

1. every planted defect is flagged (FAIL or WARN) under its expected
   check-id prefix;
2. the clean case never FAILs - including specifically under every prefix
   that fires on some *other* case (the paired false-positive guard);
3. the rolling-window leak is caught only by the dynamic truncation probe,
   not by the static leakage/lookahead detectors;
4. report machinery (to_dict/json, to_markdown, __getitem__, summary,
   raise_for_failures) behaves;
5. input-validation errors carry actionable messages;
6. include/exclude filtering and run-to-run determinism.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import qaudit
from qaudit import (AuditConfig, AuditFailure, BacktestArtifacts,
                    InputValidationError, MisalignedInputError, Status)
from qaudit import synthetic

CASE_NAMES = list(synthetic.ALL_CASES)

# Small-but-sufficient dynamic-probe budgets so the whole suite stays fast.
N_PLACEBO = 40
N_SHUFFLE = 60


def _config_for(case: synthetic.SyntheticBacktest) -> AuditConfig:
    return AuditConfig(seed=1, n_placebo=N_PLACEBO, n_shuffle=N_SHUFFLE,
                       n_trials=case.audit_kwargs.get("n_trials"))


@pytest.fixture(scope="module")
def audited() -> dict[str, tuple[synthetic.SyntheticBacktest,
                                 qaudit.AuditReport]]:
    """Every ALL_CASES case (seed=0), built once, plus its audit report."""
    out = {}
    for name, factory in synthetic.ALL_CASES.items():
        case = factory(seed=0)
        report = qaudit.audit(case.artifacts, config=_config_for(case),
                              signal_func=case.signal_func,
                              backtest_func=case.backtest_func)
        out[name] = (case, report)
    return out


# ---------------------------------------------------------------------------
# 1. every planted defect is detected under its expected prefix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CASE_NAMES)
def test_expected_flags_fire(audited, name):
    case, report = audited[name]
    for prefix in case.expected_flags:
        hits = report.find(prefix)
        assert hits, (
            f"case {name!r}: no results at all under expected prefix "
            f"{prefix!r}; ran: {sorted(r.check for r in report.results)}")
        flagged = [r for r in hits
                   if r.status in (Status.FAIL, Status.WARN)]
        assert flagged, (
            f"case {name!r}: planted defect not flagged under {prefix!r}; "
            f"statuses were "
            f"{[(r.check, r.status.value) for r in hits]}")


# ---------------------------------------------------------------------------
# 2. the false-positive guard (clean case)
# ---------------------------------------------------------------------------

def test_clean_has_zero_fail_and_zero_error(audited):
    _, report = audited["clean"]
    assert report.failures == [], (
        "clean case must not FAIL any check; got "
        f"{[(r.check, r.message) for r in report.failures]}")
    assert report.errors == [], (
        "clean case must not ERROR any check; got "
        f"{[(r.check, r.message) for r in report.errors]}")


def test_clean_paired_guard_against_every_detector(audited):
    """For every prefix expected to fire on some other case, the clean
    backtest's matching results must be PASS or SKIP - the detectors must
    not fire on an honest backtest."""
    _, clean_report = audited["clean"]
    for name in CASE_NAMES:
        if name == "clean":
            continue
        case, _ = audited[name]
        for prefix in case.expected_flags:
            hits = clean_report.find(prefix)
            assert hits, (
                f"clean report has no results under {prefix!r} (expected to "
                f"fire on case {name!r}) - cannot pair-guard a detector that "
                f"never ran")
            bad = [r for r in hits
                   if r.status not in (Status.PASS, Status.SKIP)]
            assert not bad, (
                f"detector prefix {prefix!r} (planted in case {name!r}) "
                f"fires on the clean backtest: "
                f"{[(r.check, r.status.value, r.message) for r in bad]}")


# ---------------------------------------------------------------------------
# 3. rolling_window_leak is caught only dynamically
# ---------------------------------------------------------------------------

def test_rolling_window_leak_caught_only_by_truncation_probe(audited):
    """The full-sample z-score leak is invisible to the static leakage /
    embedded-future-return detectors (documented limitation); only the
    dynamic truncation probe sees it."""
    _, report = audited["rolling_window_leak"]

    trunc = report["dynamic.rolling_window_integrity"]
    assert trunc.status is Status.FAIL, (
        f"truncation probe must FAIL on the full-sample z-score leak; got "
        f"{trunc.status.value}: {trunc.message}")

    for static_prefix in ("leakage.", "lookahead.embedded"):
        hits = report.find(static_prefix)
        assert hits, f"no results under static prefix {static_prefix!r}"
        fails = [r for r in hits if r.status is Status.FAIL]
        assert not fails, (
            f"static checks under {static_prefix!r} unexpectedly FAIL on "
            f"the rolling-window leak (they are documented as blind to it): "
            f"{[(r.check, r.message) for r in fails]}")


# ---------------------------------------------------------------------------
# 4. report machinery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CASE_NAMES)
def test_to_dict_json_roundtrip(audited, name):
    _, report = audited[name]
    d = report.to_dict()
    loaded = json.loads(json.dumps(d))
    assert loaded["summary"] == d["summary"]
    assert loaded["ok"] == d["ok"]
    assert [r["check"] for r in loaded["results"]] == \
           [r["check"] for r in d["results"]]
    assert [r["status"] for r in loaded["results"]] == \
           [r["status"] for r in d["results"]]
    assert loaded["meta"]["n_periods"] == d["meta"]["n_periods"]


def test_to_markdown_has_summary_table_header(audited):
    _, report = audited["clean"]
    md = report.to_markdown()
    assert "| status | severity | check | finding |" in md
    assert "|---|---|---|---|" in md


def test_getitem_returns_result_and_keyerror_lists_ids(audited):
    _, report = audited["clean"]
    r = report["performance.sample_size"]
    assert r.check == "performance.sample_size"

    with pytest.raises(KeyError) as ei:
        report["no.such_check"]
    msg = str(ei.value)
    assert "no.such_check" in msg
    # the error must list the available check ids
    assert "performance.sample_size" in msg
    assert "lookahead.position_signal_alignment" in msg


def test_summary_verdicts(audited):
    _, look = audited["lookahead"]
    _, clean = audited["clean"]
    assert "SUSPECT" in look.summary()
    assert "CLEAN" in clean.summary()


def test_raise_for_failures_names_failing_check(audited):
    _, look = audited["lookahead"]
    failing_ids = {r.check for r in look.failures} | \
                  {r.check for r in look.errors}
    assert failing_ids, "lookahead case should have at least one FAIL"
    with pytest.raises(AuditFailure) as ei:
        look.raise_for_failures()
    msg = str(ei.value)
    assert any(cid in msg for cid in failing_ids), (
        f"AuditFailure message names no failing check id; ids={failing_ids}, "
        f"message: {msg}")

    _, clean = audited["clean"]
    clean.raise_for_failures()  # must not raise


# ---------------------------------------------------------------------------
# 5. error-message quality on malformed inputs
# ---------------------------------------------------------------------------

def _panel(n: int = 40, assets=("A", "B", "C", "D", "E", "F"),
           seed: int = 0, index: pd.DatetimeIndex | None = None
           ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = index if index is not None else pd.bdate_range("2021-01-04",
                                                         periods=n)
    return pd.DataFrame(rng.normal(0.0, 0.01, (len(idx), len(assets))),
                        index=idx, columns=list(assets))


def test_unsorted_index_message_suggests_sort_index():
    sig = _panel().iloc[::-1]  # descending dates
    art = BacktestArtifacts(signals=sig, asset_returns=_panel(seed=1))
    with pytest.raises(InputValidationError, match="sort_index"):
        qaudit.audit(art)


def test_rangeindex_message_names_datetimeindex():
    sig = _panel().reset_index(drop=True)  # RangeIndex, not dates
    art = BacktestArtifacts(signals=sig, asset_returns=_panel(seed=1))
    with pytest.raises(InputValidationError, match="DatetimeIndex"):
        qaudit.audit(art)


def test_disjoint_asset_columns_message():
    sig = _panel(assets=("A", "B", "C", "D", "E"))
    rets = _panel(assets=("V", "W", "X", "Y", "Z"), seed=1)
    art = BacktestArtifacts(signals=sig, asset_returns=rets)
    with pytest.raises(MisalignedInputError, match="share no assets"):
        qaudit.audit(art)


def test_infinite_returns_message():
    rets = _panel(seed=1)
    rets.iloc[3, 2] = np.inf
    art = BacktestArtifacts(signals=_panel(), asset_returns=rets)
    with pytest.raises(InputValidationError, match="infinite"):
        qaudit.audit(art)


def test_negative_signal_lag_message():
    art = BacktestArtifacts(signals=_panel(), asset_returns=_panel(seed=1),
                            signal_lag=-1)
    with pytest.raises(InputValidationError, match="signal_lag"):
        qaudit.audit(art)


def test_empty_date_overlap_raises_misaligned():
    sig = _panel(index=pd.bdate_range("2020-01-01", periods=40))
    rets = _panel(index=pd.bdate_range("2022-01-03", periods=40), seed=1)
    art = BacktestArtifacts(signals=sig, asset_returns=rets)
    with pytest.raises(MisalignedInputError):
        qaudit.audit(art)


# ---------------------------------------------------------------------------
# 6. include / exclude filtering
# ---------------------------------------------------------------------------

def test_include_lookahead_only(audited):
    case, _ = audited["clean"]
    report = qaudit.audit(case.artifacts, config=AuditConfig(seed=1),
                          include=["lookahead"])
    assert report.results, "include=['lookahead'] returned no results"
    assert all(r.check.startswith("lookahead.") for r in report.results), (
        f"non-lookahead results leaked through include filter: "
        f"{[r.check for r in report.results]}")


def test_exclude_dynamic_drops_dynamic(audited):
    case, _ = audited["clean"]
    report = qaudit.audit(case.artifacts, config=AuditConfig(seed=1),
                          exclude=["dynamic"])
    assert report.results
    assert not any(r.check.startswith("dynamic") for r in report.results), (
        f"dynamic results survived exclude=['dynamic']: "
        f"{[r.check for r in report.results if r.check.startswith('dynamic')]}")
    # the static families are still there
    assert any(r.check.startswith("lookahead.") for r in report.results)
    assert any(r.check.startswith("costs.") for r in report.results)


# ---------------------------------------------------------------------------
# 7. determinism
# ---------------------------------------------------------------------------

def test_audit_is_deterministic(audited):
    """Same case + same config (fresh AuditConfig, same seed) => identical
    (check, status) pairs, including the randomized dynamic probes."""
    name = "rolling_window_leak"   # exercises both callables + rng probes
    case, first = audited[name]
    second = qaudit.audit(case.artifacts, config=_config_for(case),
                          signal_func=case.signal_func,
                          backtest_func=case.backtest_func)
    assert [(r.check, r.status) for r in first.results] == \
           [(r.check, r.status) for r in second.results]
