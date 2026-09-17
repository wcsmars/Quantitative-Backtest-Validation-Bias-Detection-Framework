"""Tests for qaudit.dynamic.probes_shift (date-shift + truncation probes).

Calls the module's run() directly on aligned artifacts - never through
qaudit.audit() - so it stays independent of the other check modules.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from qaudit.config import AuditConfig
from qaudit.dynamic.probes_shift import (CHECK_DATE_SHIFT, CHECK_REPRO,
                                         CHECK_SENSITIVITY, CHECK_TRUNCATION,
                                         run)
from qaudit.synthetic import (make_clean, make_lookahead,
                              make_rolling_window_leak, momentum_signal)
from qaudit.types import Severity, Status

CFG = AuditConfig()


def by_check(results):
    out = {r.check: r for r in results}
    # the module must always emit exactly these four ids, once each
    assert set(out) == {CHECK_DATE_SHIFT, CHECK_REPRO, CHECK_TRUNCATION,
                        CHECK_SENSITIVITY}
    assert len(results) == 4
    return out


# ---------------------------------------------------------------------------
# fixtures: build each synthetic case once per module, run the module once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_case():
    return make_clean()


@pytest.fixture(scope="module")
def clean_results(clean_case):
    art = clean_case.artifacts.aligned()
    return by_check(run(art, CFG, signal_func=clean_case.signal_func,
                        backtest_func=clean_case.backtest_func))


@pytest.fixture(scope="module")
def lookahead_case():
    return make_lookahead()


@pytest.fixture(scope="module")
def lookahead_results(lookahead_case):
    art = lookahead_case.artifacts.aligned()
    return by_check(run(art, CFG, signal_func=lookahead_case.signal_func,
                        backtest_func=lookahead_case.backtest_func))


@pytest.fixture(scope="module")
def rwl_case():
    return make_rolling_window_leak()


@pytest.fixture(scope="module")
def rwl_results(rwl_case):
    art = rwl_case.artifacts.aligned()
    return by_check(run(art, CFG, signal_func=rwl_case.signal_func,
                        backtest_func=rwl_case.backtest_func))


# ---------------------------------------------------------------------------
# dynamic.date_shift
# ---------------------------------------------------------------------------

def test_date_shift_fails_on_lookahead_via_peek_path(lookahead_results):
    r = lookahead_results[CHECK_DATE_SHIFT]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    curve = r.details["curve"]
    sr0, sr_m1 = curve["0"], curve["-1"]
    # calibrated: SR_0 ~76 collapses when the signal is shifted one bar earlier
    assert sr0 > 10.0
    assert sr_m1 < 0.5 * sr0
    # path (a): the message is about peeking earlier, not about delay
    assert "EARLIER" in r.message
    assert "delay" not in r.message.split("-")[0].lower()
    # message quality: carries the concrete baseline SR number
    assert f"{sr0:.1f}" in r.message
    assert r.remediation  # must tell the researcher what to change


def test_date_shift_full_curve_in_details(lookahead_results):
    r = lookahead_results[CHECK_DATE_SHIFT]
    curve = r.details["curve"]
    k_max = CFG.date_shift_max
    assert sorted(curve, key=int) == [str(k) for k in range(-k_max, k_max + 1)]
    for v in curve.values():
        assert v is None or isinstance(v, float)


def test_date_shift_passes_on_clean_and_peeking_helps(clean_results):
    r = clean_results[CHECK_DATE_SHIFT]
    assert r.status is Status.PASS
    curve = r.details["curve"]
    # honest pipeline: illegal peeking helps (calibrated ~17 vs ~1.8 baseline)
    assert curve["-1"] > curve["0"]
    assert 1.0 < curve["0"] < 3.0
    assert curve["-1"] > 5.0
    # PASS message still says what was measured
    assert "curve" in r.message.lower()


def test_date_shift_warn_path_on_ultrafast_edge(clean_case):
    """Hand-rig path (b): a backtest_func whose edge survives (illegal)
    peeking but dies with one extra bar of delay - ultra-fast alpha."""
    art = clean_case.artifacts.aligned()
    honest_bt = clean_case.backtest_func
    baseline = honest_bt(art.signals, art.asset_returns)
    # deterministic zero-mean series -> annualized SR exactly 0.0
    flat = baseline * 0.0
    flat.iloc[::2] += 0.01
    flat.iloc[1::2] -= 0.01

    def brittle_bt(signals, asset_returns):
        for k in range(-CFG.date_shift_max, CFG.date_shift_max + 1):
            ref = art.signals.shift(k)
            if np.array_equal(signals.to_numpy(), ref.to_numpy(),
                              equal_nan=True):
                return baseline if k <= 0 else flat  # delay kills the edge
        raise AssertionError("unexpected signal frame passed to backtest_func")

    r = {c.check: c for c in run(art, CFG, backtest_func=brittle_bt)}[CHECK_DATE_SHIFT]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert "delay" in r.message
    assert re.search(r"\d", r.message)


def test_date_shift_skips_without_backtest_func(clean_case):
    art = clean_case.artifacts.aligned()
    results = by_check(run(art, CFG, signal_func=clean_case.signal_func,
                           backtest_func=None))
    r = results[CHECK_DATE_SHIFT]
    assert r.status is Status.SKIP
    assert "backtest_func" in r.message
    assert "asset_returns" in r.message  # states the exact signature to pass


# ---------------------------------------------------------------------------
# dynamic.signal_reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility_passes_on_clean(clean_results):
    r = clean_results[CHECK_REPRO]
    assert r.status is Status.PASS
    assert r.details["frac_mismatch"] == 0.0
    assert r.details["n_cells"] > 0
    assert "reproduce" in r.message


def test_reproducibility_warns_on_perturbed_signal_func(clean_case):
    art = clean_case.artifacts.aligned()

    def scaled_signal_func(signal_input):
        return momentum_signal(signal_input) * 1.05

    results = by_check(run(art, CFG, signal_func=scaled_signal_func,
                           backtest_func=None))
    r = results[CHECK_REPRO]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert r.details["frac_mismatch"] > 0.01
    # message quality: names signal_func and carries the mismatch percentage
    assert "signal_func" in r.message
    assert re.search(r"\d+(\.\d+)?%", r.message)
    assert r.remediation
    # a uniform 5% rescale is still causal: the truncation probe must PASS
    assert results[CHECK_TRUNCATION].status is Status.PASS


# ---------------------------------------------------------------------------
# dynamic.rolling_window_integrity
# ---------------------------------------------------------------------------

def test_truncation_fails_on_rolling_window_leak(rwl_results):
    r = rwl_results[CHECK_TRUNCATION]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    # message names a concrete date and says the computation reads the future
    assert re.search(r"\d{4}-\d{2}-\d{2}", r.message)
    assert "future" in r.message
    assert re.search(r"max \|diff\| \d", r.message)
    d = r.details
    assert d["n_sampled"] == CFG.truncation_sample_dates
    assert d["n_mismatched"] >= 1
    assert d["example_date"] is not None
    assert d["max_abs_diff"] > 0
    assert r.remediation


def test_truncation_leak_is_statically_invisible(rwl_results):
    """The rolling-window leak is calibrated to be indistinguishable from
    clean statically (SR ~1.28): the date-shift probe must not flag it -
    only the truncation probe catches it."""
    assert rwl_results[CHECK_DATE_SHIFT].status is Status.PASS


def test_truncation_passes_on_clean(rwl_results, clean_results):
    r = clean_results[CHECK_TRUNCATION]
    assert r.status is Status.PASS
    d = r.details
    assert d["n_sampled"] == CFG.truncation_sample_dates
    assert d["n_mismatched"] == 0
    assert d["example_date"] is None and d["max_abs_diff"] is None
    assert "truncated" in r.message
    # paired guard: same check id that FAILs on the rigged case
    assert r.check == rwl_results[CHECK_TRUNCATION].check


def test_truncation_deterministic_under_seed(clean_case):
    art = clean_case.artifacts.aligned()
    r1 = by_check(run(art, CFG, signal_func=clean_case.signal_func))[CHECK_TRUNCATION]
    r2 = by_check(run(art, CFG, signal_func=clean_case.signal_func))[CHECK_TRUNCATION]
    assert r1.details == r2.details


# ---------------------------------------------------------------------------
# SKIP paths for the signal_func probes
# ---------------------------------------------------------------------------

def test_signal_probes_skip_without_signal_func(clean_case):
    art = clean_case.artifacts.aligned()
    results = by_check(run(art, CFG, signal_func=None,
                           backtest_func=None))
    for check_id in (CHECK_REPRO, CHECK_TRUNCATION):
        r = results[check_id]
        assert r.status is Status.SKIP
        # names the missing callable and states both requirements
        assert "signal_func" in r.message
        assert "signal_input" in r.message


def test_signal_probes_skip_without_signal_input(clean_case):
    import dataclasses
    art = dataclasses.replace(clean_case.artifacts.aligned(), signal_input=None)
    results = by_check(run(art, CFG, signal_func=clean_case.signal_func,
                           backtest_func=None))
    for check_id in (CHECK_REPRO, CHECK_TRUNCATION):
        r = results[check_id]
        assert r.status is Status.SKIP
        assert "signal_input" in r.message


# ---------------------------------------------------------------------------
# hygiene: run() must never mutate the artifacts it receives
# ---------------------------------------------------------------------------

def test_run_does_not_mutate_artifacts(clean_case):
    art = clean_case.artifacts.aligned()
    sig_before = art.signals.copy()
    rets_before = art.asset_returns.copy()
    input_before = art.signal_input.copy()
    run(art, CFG, signal_func=clean_case.signal_func,
        backtest_func=clean_case.backtest_func)
    assert art.signals.equals(sig_before)
    assert art.asset_returns.equals(rets_before)
    assert art.signal_input.equals(input_before)


# ---------------------------------------------------------------------------
# A raising backtest_func must yield one clear ERROR result
# naming the callable, not a module-wide crash.
# ---------------------------------------------------------------------------

def test_date_shift_raising_backtest_func_is_graceful(clean_case):
    def exploding(signals, asset_returns):
        raise RuntimeError("user pipeline exploded")

    results = run(clean_case.artifacts.aligned(), CFG, backtest_func=exploding)
    by = {r.check: r for r in results}
    r = by[CHECK_DATE_SHIFT]
    assert r.status is Status.ERROR
    assert re.search(r"backtest_func raised", r.message), r.message
    assert "user pipeline exploded" in r.message
    # the signal_func-based probes are unaffected by the broken backtest_func
    assert by[CHECK_TRUNCATION].status is not Status.ERROR
