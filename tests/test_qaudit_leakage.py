"""Tests for qaudit.checks.leakage (target-inside-the-features detectors).

Each detector gets a positive test on a rigged synthetic (or a hand-built
frame) plus a false-positive guard on make_clean(). Cases are built once per
module via module-scoped fixtures.
"""
import re

import numpy as np
import pandas as pd
import pytest

from qaudit.checks.leakage import (
    ALL_CHECKS, CHECK_IDENTITY, CHECK_PERFECT, CHECK_TARGET_CORR, run)
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import make_clean, make_lookahead, make_target_leak
from qaudit.types import Severity, Status

CFG = AuditConfig()


def _by_check(results):
    out = {r.check: r for r in results}
    assert set(out) == set(ALL_CHECKS), "every check id must appear exactly once"
    return out


@pytest.fixture(scope="module")
def clean_results():
    case = make_clean()
    return _by_check(run(case.artifacts.aligned(), CFG))


@pytest.fixture(scope="module")
def target_leak_results():
    case = make_target_leak()
    return _by_check(run(case.artifacts.aligned(), CFG))


@pytest.fixture(scope="module")
def lookahead_results():
    case = make_lookahead()
    return _by_check(run(case.artifacts.aligned(), CFG))


# ---------------------------------------------------------------------------
# target_leak: the signal is the forward return plus noise -> all three FAIL
# ---------------------------------------------------------------------------

def test_target_leak_target_correlation_fails(target_leak_results):
    r = target_leak_results[CHECK_TARGET_CORR]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["median_abs_ic"] > 0.9
    assert r.details["n_dates"] > 900
    # message quality: cites the median, the date count, and the honest range
    assert re.search(r"0\.9\d+", r.message), r.message
    assert str(r.details["n_dates"]) in r.message
    assert "noise floor" in r.message
    assert r.remediation


def test_target_leak_perfect_rank_dates_fails(target_leak_results):
    r = target_leak_results[CHECK_PERFECT]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_perfect"] > 0.5 * r.details["n_dates"]
    # an example ISO date must appear in the message
    assert re.search(r"\d{4}-\d{2}-\d{2}", r.message), r.message
    assert len(r.details["example_dates"]) == 3
    assert r.details["example_dates"][0] in r.message
    assert all(isinstance(d, str) for d in r.details["example_dates"])


def test_target_leak_identity_fails(target_leak_results):
    r = target_leak_results[CHECK_IDENTITY]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["median_corr"] >= CFG.leak_identity_corr
    assert 0.0 < r.details["frac_dates_above"] <= 1.0
    assert "affine transform" in r.message


def test_target_leak_details_are_json_friendly(target_leak_results):
    for r in target_leak_results.values():
        for k, v in r.details.items():
            assert isinstance(v, (int, float, str, list)), (r.check, k, type(v))


# ---------------------------------------------------------------------------
# lookahead: signal embeds z-scored next-day return -> check 1 FAILs
# ---------------------------------------------------------------------------

def test_lookahead_target_correlation_fails(lookahead_results):
    r = lookahead_results[CHECK_TARGET_CORR]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["median_abs_ic"] >= CFG.leak_median_ic_fail


# ---------------------------------------------------------------------------
# clean: false-positive guard - all three PASS with an honest weak signal
# ---------------------------------------------------------------------------

def test_clean_all_pass(clean_results):
    for check_id, r in clean_results.items():
        assert r.status is Status.PASS, f"{check_id}: {r.message}"


def test_clean_median_ic_is_noise_level(clean_results):
    r = clean_results[CHECK_TARGET_CORR]
    # per-date |spearman| of a weak signal with ~30 names has median ~0.12,
    # noise-driven - well below the 0.30 warn threshold
    assert r.details["median_abs_ic"] < 0.2
    # PASS message still states what was measured
    assert re.search(r"0\.\d+", r.message)
    assert str(r.details["n_dates"]) in r.message


def test_clean_no_perfect_dates_and_low_identity_corr(clean_results):
    perfect = clean_results[CHECK_PERFECT]
    assert perfect.details["n_perfect"] / perfect.details["n_dates"] \
        <= CFG.leak_perfect_date_frac
    identity = clean_results[CHECK_IDENTITY]
    assert identity.details["median_corr"] < 0.5


# ---------------------------------------------------------------------------
# hand-built tiny frame: signal == next return exactly -> all three FAIL
# ---------------------------------------------------------------------------

def _tiny_frames(n_dates=40, n_assets=8, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    cols = [f"S{i}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n_dates, n_assets)),
                        index=dates, columns=cols)
    return rets


def test_identity_copy_tiny_frame_all_fail():
    rets = _tiny_frames()
    signals = rets.shift(-1)          # signal at t is the return over t+1
    art = BacktestArtifacts(signals=signals, asset_returns=rets)
    results = _by_check(run(art.aligned(), CFG))
    for check_id in ALL_CHECKS:
        assert results[check_id].status is Status.FAIL, check_id
        assert results[check_id].severity is Severity.CRITICAL
    assert results[CHECK_TARGET_CORR].details["median_abs_ic"] == pytest.approx(1.0)
    assert results[CHECK_IDENTITY].details["median_corr"] == pytest.approx(1.0)
    assert results[CHECK_PERFECT].details["n_perfect"] \
        == results[CHECK_PERFECT].details["n_dates"]


def test_run_does_not_mutate_artifacts():
    rets = _tiny_frames()
    signals = rets.shift(-1)
    art = BacktestArtifacts(signals=signals, asset_returns=rets).aligned()
    sig_before = art.signals.copy(deep=True)
    ret_before = art.asset_returns.copy(deep=True)
    run(art, CFG)
    pd.testing.assert_frame_equal(art.signals, sig_before)
    pd.testing.assert_frame_equal(art.asset_returns, ret_before)


# ---------------------------------------------------------------------------
# underpower: fewer than 30 usable dates -> WARN(LOW) stating the count
# ---------------------------------------------------------------------------

def test_underpowered_sample_warns_low():
    rets = _tiny_frames(n_dates=25)
    signals = pd.DataFrame(np.random.default_rng(4).normal(size=rets.shape),
                           index=rets.index, columns=rets.columns)
    art = BacktestArtifacts(signals=signals, asset_returns=rets)
    results = _by_check(run(art.aligned(), CFG))
    for check_id in ALL_CHECKS:
        r = results[check_id]
        assert r.status is Status.WARN, check_id
        assert r.severity is Severity.LOW, check_id
        # 25 dates minus the last (NaN forward return) = 24 usable
        assert "24" in r.message, r.message
        assert r.details["n_dates"] == 24


# ---------------------------------------------------------------------------
# SKIP paths: missing required artifacts are named in the message
# ---------------------------------------------------------------------------

def test_skip_when_signals_missing():
    rets = _tiny_frames()
    art = BacktestArtifacts(signals=None, asset_returns=rets)
    results = _by_check(run(art, CFG))     # cannot aligned() without signals
    for check_id in ALL_CHECKS:
        r = results[check_id]
        assert r.status is Status.SKIP, check_id
        assert re.search(r"artifacts\.signals", r.message), r.message


def test_skip_when_asset_returns_missing():
    rets = _tiny_frames()
    art = BacktestArtifacts(signals=rets.shift(-1), asset_returns=None)
    results = _by_check(run(art, CFG))
    for check_id in ALL_CHECKS:
        r = results[check_id]
        assert r.status is Status.SKIP, check_id
        assert re.search(r"artifacts\.asset_returns", r.message), r.message


# ---------------------------------------------------------------------------
# Regression: sparse panels must not fake perfect-rank leakage
# (noise alone gives |IC| >= 0.9 on ~1.7% of 5-name dates).
# ---------------------------------------------------------------------------

def test_perfect_rank_no_false_positive_on_sparse_noise_panel():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-03", periods=200, freq="B")
    cols = [f"A{i}" for i in range(10)]
    sig = pd.DataFrame(rng.normal(size=(200, 10)), index=idx, columns=cols)
    sig[sig.abs() < 1.0] = np.nan          # ~5-6 observed names per date
    rets = pd.DataFrame(rng.normal(0, 0.01, (200, 10)), index=idx, columns=cols)
    art = BacktestArtifacts(signals=sig, asset_returns=rets).aligned()
    results = {r.check: r for r in run(art, CFG)}
    r = results["leakage.perfect_rank_dates"]
    assert r.status is not Status.FAIL, r.message
    # too few wide dates -> the underpower path, and it says why
    assert re.search(r"jointly observed names|sparse", r.message), r.message


def test_perfect_rank_still_fires_with_eight_names():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2022-01-03", periods=120, freq="B")
    cols = [f"A{i}" for i in range(8)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (120, 8)), index=idx, columns=cols)
    sig = rets.shift(-1)                    # blatant leak, exactly 8 names
    art = BacktestArtifacts(signals=sig, asset_returns=rets).aligned()
    results = {r.check: r for r in run(art, CFG)}
    assert results["leakage.perfect_rank_dates"].status is Status.FAIL


def test_perfect_rank_no_false_positive_on_eight_name_noise():
    # 8 names x 150 dates of noise: the per-date null rate (~0.5%) times few
    # dates makes the 1% *fraction* bar easy to cross by luck - the Poisson
    # count guard must hold the line. Try several seeds.
    idx = pd.date_range("2022-01-03", periods=150, freq="B")
    cols = [f"A{i}" for i in range(8)]
    for seed in range(6):
        rng = np.random.default_rng(seed)
        sig = pd.DataFrame(rng.normal(size=(150, 8)), index=idx, columns=cols)
        rets = pd.DataFrame(rng.normal(0, 0.01, (150, 8)), index=idx, columns=cols)
        art = BacktestArtifacts(signals=sig, asset_returns=rets).aligned()
        results = {r.check: r for r in run(art, CFG)}
        r = results["leakage.perfect_rank_dates"]
        assert r.status is not Status.FAIL, (seed, r.message)
