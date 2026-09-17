"""Tests for qaudit.checks.lookahead (position/signal alignment, embedded
future return, IC-decay signature).

Each synthetic case is built at most once per module (module-scoped
fixtures); the check module is called directly on aligned artifacts -
never through qaudit.audit().
"""
from __future__ import annotations

import dataclasses
import re

import pytest

from qaudit.checks.lookahead import ALIGNMENT_MARGIN, run
from qaudit.config import AuditConfig
from qaudit.synthetic import (
    make_clean,
    make_lookahead,
    make_same_bar_execution,
    make_target_leak,
)
from qaudit.types import Severity, Status

ALL_CHECK_IDS = {
    "lookahead.position_signal_alignment",
    "lookahead.same_bar_bleed",
    "lookahead.future_vol_sizing",
    "lookahead.embedded_future_return",
    "lookahead.ic_decay_signature",
    "lookahead.deferred_ic_spike",
    "lookahead.smeared_forward_ic",
    "lookahead.same_bar_return_loading",
}


def _run_case(case, config=None):
    """Run the module on aligned artifacts, return {check_id: CheckResult}."""
    results = run(case.artifacts.aligned(), config or AuditConfig())
    by_id = {r.check: r for r in results}
    assert set(by_id) == ALL_CHECK_IDS, "module must always emit every check id"
    return by_id


@pytest.fixture(scope="module")
def clean_case():
    return make_clean()


@pytest.fixture(scope="module")
def clean_results(clean_case):
    return _run_case(clean_case)


@pytest.fixture(scope="module")
def same_bar_results():
    return _run_case(make_same_bar_execution())


@pytest.fixture(scope="module")
def lookahead_results():
    return _run_case(make_lookahead())


@pytest.fixture(scope="module")
def target_leak_results():
    return _run_case(make_target_leak())


# ---------------------------------------------------------------------------
# 1. lookahead.position_signal_alignment
# ---------------------------------------------------------------------------

class TestPositionSignalAlignment:
    def test_fails_on_same_bar_execution(self, same_bar_results):
        r = same_bar_results["lookahead.position_signal_alignment"]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        # message must name the offending lag and carry concrete correlations
        assert re.search(r"lag 0", r.message)
        assert re.search(r"1\.00", r.message)          # corr at lag 0
        assert re.search(r"declared lag 1", r.message)
        assert r.remediation and "signal_lag" in r.remediation
        assert r.details["declared_lag"] == 1
        assert r.details["best_lag"] == 0
        profile = r.details["alignment_profile"]
        assert profile["0"] > profile["1"] + ALIGNMENT_MARGIN
        assert profile["0"] == pytest.approx(1.0, abs=1e-6)

    def test_passes_on_clean(self, clean_results):
        r = clean_results["lookahead.position_signal_alignment"]
        assert r.status is Status.PASS
        profile = r.details["alignment_profile"]
        # clean positions are rank weights of signals.shift(1): the profile
        # must peak exactly at the declared lag k=1
        assert r.details["best_lag"] == 1
        assert r.details["declared_lag"] == 1
        assert profile["1"] == pytest.approx(1.0, abs=1e-6)
        assert profile["1"] > profile["0"]
        assert profile["1"] > profile["2"]
        # covers k = 0..max(3, L+1) inclusive
        assert set(profile) == {"0", "1", "2", "3"}
        # pass message still states what was measured
        assert re.search(r"lag 1", r.message)
        assert re.search(r"k=0", r.message) and re.search(r"k=3", r.message)

    def test_skips_without_positions(self, clean_case):
        art = dataclasses.replace(clean_case.artifacts.aligned(), positions=None)
        by_id = {r.check: r for r in run(art, AuditConfig())}
        r = by_id["lookahead.position_signal_alignment"]
        assert r.status is Status.SKIP
        assert "positions" in r.message
        # the other two checks still run without positions
        assert by_id["lookahead.embedded_future_return"].status is Status.PASS
        assert by_id["lookahead.ic_decay_signature"].status is Status.PASS


# ---------------------------------------------------------------------------
# 2. lookahead.embedded_future_return
# ---------------------------------------------------------------------------

class TestEmbeddedFutureReturn:
    def test_fails_on_lookahead(self, lookahead_results):
        r = lookahead_results["lookahead.embedded_future_return"]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        # calibrated: mean |IC| ~ 0.957 - the number must be in the message
        assert re.search(r"0\.9\d\d", r.message)
        assert r.remediation
        assert r.details["mean_abs_ic"]["1"] == pytest.approx(0.957, abs=0.02)
        assert r.details["n_dates"]["1"] > 900

    def test_fails_on_target_leak(self, target_leak_results):
        r = target_leak_results["lookahead.embedded_future_return"]
        assert r.status is Status.FAIL
        assert r.details["mean_abs_ic"]["1"] > 0.9

    def test_passes_on_clean(self, clean_results):
        r = clean_results["lookahead.embedded_future_return"]
        assert r.status is Status.PASS
        # clean momentum: per-date |IC| noise floor ~0.15, far below 0.60
        assert r.details["mean_abs_ic"]["1"] < AuditConfig().future_return_embed_ic
        # pass message states what was measured
        assert re.search(r"h=1", r.message)
        assert re.search(r"0\.\d+", r.message)


# ---------------------------------------------------------------------------
# 3. lookahead.ic_decay_signature
# ---------------------------------------------------------------------------

class TestIcDecaySignature:
    def test_warns_on_lookahead(self, lookahead_results):
        r = lookahead_results["lookahead.ic_decay_signature"]
        assert r.status is Status.WARN
        assert r.severity is Severity.HIGH
        assert re.search(r"t\+2", r.message)
        assert r.remediation
        cfg = AuditConfig()
        assert abs(r.details["ic1"]) >= cfg.predictive_ic_fail
        assert r.details["ratio"] < cfg.ic_decay_collapse_ratio

    def test_passes_on_clean(self, clean_results):
        r = clean_results["lookahead.ic_decay_signature"]
        assert r.status is Status.PASS
        # calibrated: clean IC1 ~ +0.031, persists at t+2 (no collapse)
        assert r.details["ic1"] == pytest.approx(0.031, abs=0.01)
        assert abs(r.details["ic1"]) < AuditConfig().predictive_ic_fail
        assert re.search(r"IC", r.message)

    def test_target_leak_flagged_by_family(self, target_leak_results):
        # at minimum the embedded-future-return detector must FAIL; the decay
        # signature typically corroborates with a WARN
        assert (target_leak_results["lookahead.embedded_future_return"].status
                is Status.FAIL)
        decay = target_leak_results["lookahead.ic_decay_signature"]
        assert decay.status in (Status.WARN, Status.FAIL)


# ---------------------------------------------------------------------------
# Cross-cutting hygiene
# ---------------------------------------------------------------------------

def test_never_mutates_artifacts(clean_case):
    art = clean_case.artifacts.aligned()
    sig_before = art.signals.copy()
    pos_before = art.positions.copy()
    ret_before = art.asset_returns.copy()
    run(art, AuditConfig())
    assert art.signals.equals(sig_before)
    assert art.positions.equals(pos_before)
    assert art.asset_returns.equals(ret_before)


def test_clean_has_no_fail_or_warn(clean_results):
    bad = [r for r in clean_results.values()
           if r.status in (Status.FAIL, Status.WARN)]
    assert not bad, f"clean case must be flag-free, got: {bad}"
