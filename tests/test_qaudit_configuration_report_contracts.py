"""Configuration domains, empty audit filtering, and report-message contracts.

Correlation thresholds stay in their valid range, malformed values identify
the field, and demo failure handling survives Python optimization. Empty
reports fail closed, multiline messages render safely, and callback failures
remain evidence even when module execution is strict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit import demo
from qaudit.api import audit
from qaudit.checks import leakage
from qaudit.config import AuditConfig
from qaudit.errors import AuditFailure, InputValidationError
from qaudit.inputs import BacktestArtifacts
from qaudit.report import AuditReport
from qaudit.synthetic import (momentum_signal, positions_from_signals,
                              simulate_market)
from qaudit.types import CheckResult, Severity, Status


@pytest.fixture(scope="module")
def market():
    return simulate_market(n_assets=12, n_periods=400, seed=17, death_frac=0.0)


@pytest.fixture(scope="module")
def art(market):
    sig = momentum_signal(market["returns"])
    a = BacktestArtifacts(signals=sig, asset_returns=market["returns"],
                          positions=positions_from_signals(sig, 1),
                          signal_lag=1)
    a.validate()
    return a


# ===========================================================================
# 1. correlation-valued thresholds > 1 raise (attack)
# ===========================================================================

@pytest.mark.parametrize("kw", [
    dict(predictive_ic_warn=5.0, predictive_ic_fail=7.0),
    dict(predictive_ic_fail=7.0),
    dict(leak_median_ic_warn=3.0, leak_median_ic_fail=9.0),
    dict(leak_median_ic_warn=3.0),
    dict(bleed_partial_fail=4.0),
])
def test_correlation_thresholds_above_one_raise(kw):
    # |Spearman| <= 1, so any of these makes the WARN/FAIL branch unreachable
    name = next(iter(kw))
    with pytest.raises(InputValidationError, match=name):
        AuditConfig(**kw)


def test_mutation_to_disarming_threshold_rejected():
    cfg = AuditConfig()
    with pytest.raises(InputValidationError, match="leak_median_ic_fail"):
        cfg.leak_median_ic_fail = 9.0
    assert cfg.leak_median_ic_fail == 0.50   # left in last valid state


def test_leak_detector_cannot_be_disarmed_by_typo():
    # A pure next-bar target copy must fail. A correlation threshold above one is
    # invalid and cannot disarm detection.
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-02", periods=260)
    cols = [f"A{i:02d}" for i in range(30)]
    rets = pd.DataFrame(rng.standard_normal((260, 30)) * 0.01,
                        index=idx, columns=cols)
    sig = rets.shift(-1).iloc[:-1]
    a = BacktestArtifacts(signals=sig, asset_returns=rets.iloc[:-1],
                          signal_lag=1)
    a.validate()
    res = {r.check: r for r in leakage.run(a.aligned(), AuditConfig())}
    assert res["leakage.target_correlation"].status is Status.FAIL
    with pytest.raises(InputValidationError):
        AuditConfig(leak_median_ic_warn=3.0, leak_median_ic_fail=9.0)


def test_boundary_and_default_correlation_thresholds_still_legal():
    AuditConfig()                                     # defaults construct
    cfg = AuditConfig(predictive_ic_warn=0.5, predictive_ic_fail=1.0,
                      leak_median_ic_warn=0.9, leak_median_ic_fail=1.0,
                      bleed_partial_fail=1.0)         # 1.0 is attainable
    assert cfg.bleed_partial_fail == 1.0


# ===========================================================================
# 2. scalar garbage -> branded InputValidationError naming the field
# ===========================================================================

@pytest.mark.parametrize("kw", [
    dict(predictive_ic_warn="0.1"),
    dict(sharpe_warn=None),
    dict(min_periods="120"),
    dict(bleed_tstat=True),                 # bool passes int comparisons
    dict(n_trials="345"),
    dict(min_embargo_periods="5"),
    dict(cost_sensitivity_bps=25.0),        # scalar for a sequence
    dict(cost_sensitivity_bps="5,10"),      # str is iterable - must reject
    dict(cost_sensitivity_bps=("a", "b")),
    dict(cost_sensitivity_bps=None),
    dict(cost_sensitivity_bps=(5.0, True)),
])
def test_invalid_configuration_error_names_field(kw):
    # Malformed values must raise a field-specific validation error.
    name = next(iter(kw))
    with pytest.raises(InputValidationError, match=name):
        AuditConfig(**kw)


def test_numpy_and_mixed_numeric_types_still_accepted():
    cfg = AuditConfig(sharpe_warn=np.float64(3.5), min_periods=np.int64(100),
                      n_trials=np.int64(345),
                      cost_sensitivity_bps=[5, 10.0, np.float64(25.0)])
    assert cfg.min_periods == 100


# ===========================================================================
# 3. demo clean guard: explicit exit code, no bare assert
# ===========================================================================

def _fake_audit(results):
    def fake(artifacts, config, signal_func=None, backtest_func=None):
        return AuditReport(results=list(results))
    return fake


def test_demo_clean_guard_returns_1_on_fail(monkeypatch, capsys):
    # Failure handling must use an explicit branch that survives PYTHONOPTIMIZE=1 and
    # returns exit status 1.
    monkeypatch.setattr(demo, "audit", _fake_audit([
        CheckResult("lookahead.injected", Status.FAIL, "injected FAIL",
                    severity=Severity.CRITICAL),
        CheckResult("performance.ok", Status.PASS, "ok",
                    severity=Severity.LOW)]))
    rc = demo.main(["clean"])
    out, err = capsys.readouterr()
    assert rc == 1
    assert "false-positive guard violated" in err
    assert "1 FAIL(s)" in err and "lookahead.injected" in err
    assert "OK (" not in out          # the counts line must not print


def test_demo_clean_guard_ok_path(monkeypatch, capsys):
    monkeypatch.setattr(demo, "audit", _fake_audit([
        CheckResult("performance.ok", Status.PASS, "ok",
                    severity=Severity.LOW)]))
    rc = demo.main(["clean"])
    out, err = capsys.readouterr()
    assert rc == 0
    assert "clean false-positive guard: OK (1 PASS, 0 WARN, 0 SKIP)" in out


# ===========================================================================
# 4. include/exclude matching nothing must not go green
# ===========================================================================

@pytest.mark.parametrize("kw", [
    dict(include=["lookahed"]),                       # typo'd family
    dict(include=["lookahead"], exclude=["lookahead"]),
    dict(exclude=["lookahead", "leakage", "contamination", "survivorship",
                  "performance", "costs", "dynamic"]),
])
def test_filters_matching_nothing_are_loud(art, kw):
    # An empty audit must fail closed.
    rep = audit(art, **kw)
    assert len(rep.results) == 1
    r = rep.results[0]
    assert r.check == "audit.filter_matched_nothing"
    assert r.status is Status.ERROR
    assert "matched 0" in r.message
    assert "lookahead" in r.remediation   # names the valid family prefixes
    assert not rep.ok
    assert "SUSPECT" in rep.summary()
    with pytest.raises(AuditFailure):
        rep.raise_for_failures()


def test_valid_include_gets_no_synthetic_error(art):
    rep = audit(art, include=["costs"])
    assert rep.results
    assert all(r.check != "audit.filter_matched_nothing" for r in rep.results)
    assert all(r.check.startswith("costs") for r in rep.results)


def test_valid_exclude_gets_no_synthetic_error(art):
    rep = audit(art, exclude=["survivorship"])
    assert rep.results
    assert not any(r.check.startswith("survivorship") for r in rep.results)
    assert all(r.check != "audit.filter_matched_nothing" for r in rep.results)


def test_skip_placeholder_readmit_not_shadowed(art):
    # Including a known but unrun check retains its family SKIP placeholder rather than
    # producing an empty-filter error.
    rep = audit(art, include=["contamination.split_overlap"])
    assert any(r.check == "contamination.split_declared"
               and r.status is Status.SKIP for r in rep.results)
    assert all(r.check != "audit.filter_matched_nothing" for r in rep.results)


# ===========================================================================
# 5. to_markdown survives multiline messages/remediations
# ===========================================================================

def _multiline_report():
    return AuditReport(results=[
        CheckResult("dynamic.date_shift", Status.ERROR,
                    "backtest_func raised: ValueError: first line\n"
                    "second | line\nthird line",
                    severity=Severity.HIGH),
        CheckResult("lookahead.same_bar_bleed", Status.WARN,
                    "single-line message", severity=Severity.MEDIUM,
                    remediation="step one\nstep two"),
        CheckResult("costs.ok", Status.PASS, "fine", severity=Severity.LOW),
    ])


def test_to_markdown_table_rows_stay_single_lines():
    md = _multiline_report().to_markdown()
    lines = md.splitlines()
    header = lines.index("| status | severity | check | finding |")
    table = lines[header:header + 5]      # header + separator + 3 rows
    assert len(table) == 5
    for ln in table:
        assert ln.startswith("|") and ln.rstrip().endswith("|"), ln
    row = next(ln for ln in table if "dynamic.date_shift" in ln)
    assert "first line<br>second \\| line<br>third line" in row
    # no torn bare fragments anywhere in the document
    assert not any(ln.startswith(("second", "third")) for ln in lines)


def test_to_markdown_remediation_newlines_sanitized():
    md = _multiline_report().to_markdown()
    rem = next(ln for ln in md.splitlines()
               if ln.startswith("- ") and "same_bar_bleed" in ln)
    assert "step one<br>step two" in rem


def test_str_and_to_dict_keep_raw_newlines():
    rep = _multiline_report()
    # __str__ renders continuation lines indented (terminal contract)
    assert "\n  second | line" in str(rep)
    # to_dict carries the raw message for machine consumers
    msg = next(r["message"] for r in rep.to_dict()["results"]
               if r["check"] == "dynamic.date_shift")
    assert "\nsecond | line\n" in msg and "<br>" not in msg


# ===========================================================================
# 6. strict covers check-module crashes only; callable crashes stay in-report
# ===========================================================================

def test_strict_docstring_scopes_module_crashes():
    doc = audit.__doc__
    assert "check *module*" in doc
    assert "never re-raise" in doc        # callable crashes stay in-report


def test_strict_true_records_callable_crash_in_report(art):
    def raiser(signals, asset_returns):
        raise RuntimeError("boom line1\nboom line2")
    rep = audit(art, strict=True, backtest_func=raiser,
                include=["dynamic.date_shift"])
    errs = rep.errors
    assert len(errs) == 1 and errs[0].check == "dynamic.date_shift"
    assert not rep.ok                     # ERROR still flips the verdict
    # and the multiline exception renders as one markdown table row
    torn = [ln for ln in rep.to_markdown().splitlines()
            if ln and not ln.startswith(("|", "#", "-")) and "qaudit" not in ln]
    assert torn == []
