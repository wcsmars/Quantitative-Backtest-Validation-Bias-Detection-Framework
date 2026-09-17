"""Tests for qaudit.checks.contamination (train/test split hygiene)."""
import re

import numpy as np
import pandas as pd
import pytest

from qaudit.checks.contamination import run
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import make_clean, make_contaminated_split
from qaudit.types import Severity, Status

ALL_IDS = {
    "contamination.split_declared",
    "contamination.split_overlap",
    "contamination.insufficient_embargo",
    "contamination.test_before_train",
    "contamination.split_coverage",
}


def by_id(results, check_id):
    hits = [r for r in results if r.check == check_id]
    assert len(hits) == 1, f"expected exactly one {check_id}, got {len(hits)}"
    return hits[0]


# ---------------------------------------------------------------------------
# Fixtures: build each synthetic case at most once per module.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_results():
    case = make_clean()
    return run(case.artifacts.aligned(), AuditConfig())


@pytest.fixture(scope="module")
def contaminated_results():
    case = make_contaminated_split()
    return run(case.artifacts.aligned(), AuditConfig())


def _hand_artifacts(train, test, *, n=200, label_horizon=1):
    """Minimal aligned artifacts with hand-picked split windows."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    cols = ["X", "Y", "Z"]
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, len(cols))),
                        index=dates, columns=cols)
    sig = rets.rolling(5, min_periods=5).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            train_period=train, test_period=test,
                            label_horizon=label_horizon)
    art.validate()
    return art.aligned(), dates


# ---------------------------------------------------------------------------
# split_declared / SKIP paths
# ---------------------------------------------------------------------------

def test_both_periods_none_yields_single_skip():
    art, _ = _hand_artifacts(None, None)
    results = run(art, AuditConfig())
    assert len(results) == 1
    r = results[0]
    assert r.check == "contamination.split_declared"
    assert r.status == Status.SKIP
    assert "train_period" in r.message and "test_period" in r.message
    assert "contamination" in r.message  # says what the pass would enable


def test_missing_test_period_skip_names_it():
    dates = pd.bdate_range("2020-01-01", periods=200)
    art, _ = _hand_artifacts((dates[0], dates[99]), None)
    results = run(art, AuditConfig())
    assert len(results) == 1
    r = results[0]
    assert r.check == "contamination.split_declared"
    assert r.status == Status.SKIP
    assert "test_period=(start,end)" in r.message


def test_missing_train_period_skip_names_it():
    dates = pd.bdate_range("2020-01-01", periods=200)
    art, _ = _hand_artifacts(None, (dates[100], dates[199]))
    results = run(art, AuditConfig())
    assert len(results) == 1
    r = results[0]
    assert r.status == Status.SKIP
    assert "train_period=(start,end)" in r.message


def test_declared_split_emits_all_five_ids(clean_results):
    assert {r.check for r in clean_results} == ALL_IDS
    assert len(clean_results) == len(ALL_IDS)


# ---------------------------------------------------------------------------
# split_overlap
# ---------------------------------------------------------------------------

def test_contaminated_split_overlap_fails(contaminated_results):
    r = by_id(contaminated_results, "contamination.split_overlap")
    assert r.status == Status.FAIL
    assert r.severity == Severity.CRITICAL
    # train (dates[0]..dates[700]) vs test (dates[400]..dates[999]) on a
    # 1000-day grid -> indices 400..700 shared = 301 periods.
    assert r.details["n_overlap"] == 301
    assert re.search(r"\b301\b", r.message)
    assert "out-of-sample" in r.message
    # date range appears in the message and the details are JSON-friendly
    assert re.search(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", r.message)
    for key in ("n_overlap", "overlap_start", "overlap_end", "n_train", "n_test"):
        assert key in r.details
    assert r.remediation


def test_overlap_supersedes_embargo(contaminated_results):
    r = by_id(contaminated_results, "contamination.insufficient_embargo")
    assert r.status == Status.SKIP
    assert "overlap" in r.message


def test_clean_split_overlap_passes(clean_results):
    r = by_id(clean_results, "contamination.split_overlap")
    assert r.status == Status.PASS
    assert r.details["n_overlap"] == 0
    # PASS message still states what was measured
    assert re.search(r"\b0\b", r.message)


# ---------------------------------------------------------------------------
# insufficient_embargo
# ---------------------------------------------------------------------------

def test_adjacent_windows_with_horizon_5_warn_embargo():
    dates = pd.bdate_range("2020-01-01", periods=200)
    art, _ = _hand_artifacts((dates[0], dates[99]), (dates[100], dates[199]),
                             label_horizon=5)
    results = run(art, AuditConfig())
    r = by_id(results, "contamination.insufficient_embargo")
    assert r.status == Status.WARN
    assert r.severity == Severity.HIGH
    assert r.details == {"gap": 0, "needed": 5}
    assert "label_horizon=5" in r.message
    assert re.search(r"purge|embargo", r.message, re.I)
    assert r.remediation
    # no overlap here: adjacent, not overlapping
    assert by_id(results, "contamination.split_overlap").status == Status.PASS


def test_embargo_respects_config_override():
    dates = pd.bdate_range("2020-01-01", periods=200)
    art, _ = _hand_artifacts((dates[0], dates[99]), (dates[103], dates[199]),
                             label_horizon=1)
    # gap = 3 periods (indices 100..102); require 10 via config override
    results = run(art, AuditConfig(min_embargo_periods=10))
    r = by_id(results, "contamination.insufficient_embargo")
    assert r.status == Status.WARN
    assert r.details == {"gap": 3, "needed": 10}


def test_clean_embargo_passes(clean_results):
    r = by_id(clean_results, "contamination.insufficient_embargo")
    assert r.status == Status.PASS
    assert r.details["gap"] >= r.details["needed"]
    assert r.details["needed"] == 1  # label_horizon=1 on the clean case


# ---------------------------------------------------------------------------
# test_before_train
# ---------------------------------------------------------------------------

def test_test_window_before_train_warns():
    dates = pd.bdate_range("2020-01-01", periods=200)
    art, _ = _hand_artifacts((dates[100], dates[199]), (dates[0], dates[99]))
    results = run(art, AuditConfig())
    r = by_id(results, "contamination.test_before_train")
    assert r.status == Status.WARN
    assert r.severity == Severity.MEDIUM
    assert re.search(r"AFTER|hindsight", r.message)
    assert "walk-forward" in r.message
    assert r.remediation
    # reversed windows do not overlap, and the embargo check is inapplicable
    assert by_id(results, "contamination.split_overlap").status == Status.PASS
    assert by_id(results, "contamination.insufficient_embargo").status == Status.SKIP


def test_clean_test_before_train_passes(clean_results):
    r = by_id(clean_results, "contamination.test_before_train")
    assert r.status == Status.PASS


# ---------------------------------------------------------------------------
# split_coverage
# ---------------------------------------------------------------------------

def test_short_test_window_warns_coverage():
    dates = pd.bdate_range("2020-01-01", periods=200)
    # 30-period test window (< max(60, 5% of 200) = 60)
    art, _ = _hand_artifacts((dates[0], dates[150]), (dates[160], dates[189]))
    results = run(art, AuditConfig())
    r = by_id(results, "contamination.split_coverage")
    assert r.status == Status.WARN
    assert r.severity == Severity.LOW
    assert r.details["n_test"] == 30
    assert r.details["needed_test"] == 60
    assert re.search(r"\b30\b", r.message) and re.search(r"\b60\b", r.message)
    assert r.remediation


def test_low_joint_coverage_warns():
    dates = pd.bdate_range("2020-01-01", periods=200)
    # train 40 + test 70 = 110 of 200 periods = 55% < 60% joint coverage
    art, _ = _hand_artifacts((dates[0], dates[39]), (dates[100], dates[169]))
    results = run(art, AuditConfig())
    r = by_id(results, "contamination.split_coverage")
    assert r.status == Status.WARN
    assert r.details["n_test"] == 70          # test length itself is fine
    assert r.details["coverage_frac"] == pytest.approx(0.55)
    assert "cover" in r.message


def test_clean_coverage_passes(clean_results):
    r = by_id(clean_results, "contamination.split_coverage")
    assert r.status == Status.PASS
    assert r.details["coverage_frac"] > 0.9
    # PASS message states the measured coverage
    assert "%" in r.message


# ---------------------------------------------------------------------------
# whole-family sanity on the clean case (false-positive guard)
# ---------------------------------------------------------------------------

def test_clean_has_no_fail_or_warn(clean_results):
    bad = [r for r in clean_results if r.status in (Status.FAIL, Status.WARN)]
    assert bad == [], [str(r) for r in bad]
