"""Strict family requirements, unresolved flags, and end-of-bar timing.

Every required check needs a usable verdict unless partial coverage is
explicitly accepted. Diagnostic results cannot satisfy a family requirement.
Unresolved and unjudged passes carry flags, and signal lag must be positive.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from qaudit import (AuditConfig, AuditFailure, InputValidationError, Severity,
                    audit, synthetic)
from qaudit.report import AuditReport
from qaudit.types import CheckResult, Status, passed, skipped, warned

def _skips(n: int, fam: str = "dynamic") -> list[CheckResult]:
    return [skipped(f"{fam}.probe{i}", f"pass backtest_func to arm probe{i}")
            for i in range(n)]


# Strict required-check semantics on constructed reports.

def test_attack_family_pattern_with_one_judged_sibling_raises():
    # One judged result cannot satisfy a family with seven skipped siblings.
    rep = AuditReport(results=_skips(7) + [passed("dynamic.x", "fine")])
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=["dynamic"])
    msg = str(ei.value)
    assert "7 of them never judged" in msg and "strict mode" in msg
    for i in range(7):                      # every SKIP message verbatim
        assert f"dynamic.probe{i}: pass backtest_func to arm probe{i}" in msg
    assert "allow_partial=True" in msg and "'dynamic.x'" in msg


def test_allow_partial_restores_the_permissive_rule():
    rep = AuditReport(results=_skips(7) + [passed("dynamic.x", "fine")])
    rep.gate(require=["dynamic"], allow_partial=True)          # silent
    # but an all-SKIP pattern still fails closed in permissive mode
    with pytest.raises(AuditFailure, match="matched only SKIPped"):
        AuditReport(results=_skips(7)).gate(require=["dynamic"],
                                            allow_partial=True)


@pytest.mark.parametrize("allow_partial", [False, True])
def test_attack_probe_health_warn_cannot_satisfy_pattern(allow_partial):
    # A LOW probe-health diagnostic cannot satisfy seven skipped dynamic checks.
    health = warned("dynamic.probe_health", "null re-runs are unreliable",
                    severity=Severity.LOW)
    rep = AuditReport(results=_skips(7) + [health])
    assert rep.ok
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=["dynamic"], warn_severity=Severity.HIGH,
                 allow_partial=allow_partial)
    assert "matched only SKIPped check(s) (7)" in str(ei.value)


@pytest.mark.parametrize("pattern", ["audit", "coverage", "audit.coverage"])
def test_attack_audit_prefix_ids_cannot_satisfy_pattern(pattern):
    cov = warned("audit.coverage", "only 40% of dates kept",
                 severity=Severity.HIGH)
    rep = AuditReport(results=[cov, passed("costs.x", "ok")])
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=[pattern], allow_partial=True)
    msg = str(ei.value)
    assert "matched no check" in msg
    assert "diagnostic" in msg and "audit.coverage" in msg
    # ...while the same WARN still blocks through warn_severity
    with pytest.raises(AuditFailure, match="audit.coverage"):
        rep.gate(warn_severity=Severity.HIGH)


def test_superseded_skip_is_ignored_in_strict_mode():
    sup = skipped("survivorship.full_history_universe",
                  "universe provided - superseded by survivorship.no_exits",
                  superseded=True)
    judged = passed("survivorship.no_exits", "4 of 30 assets exit")
    AuditReport(results=[judged, sup]).gate(require=["survivorship"])
    # the same SKIP without the flag is unmet
    plain = skipped("survivorship.full_history_universe", "needs 10 assets")
    with pytest.raises(AuditFailure, match="never judged"):
        AuditReport(results=[judged, plain]).gate(require=["survivorship"])


def test_pattern_matching_only_a_superseded_skip_still_raises():
    sup = skipped("survivorship.full_history_universe", "superseded",
                  superseded=True)
    rep = AuditReport(results=[sup, passed("survivorship.no_exits", "ok")])
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=["survivorship.full_history_universe"])
    assert "none judged the backtest" in str(ei.value)
    assert "superseded" in str(ei.value)


def test_not_judged_pass_is_treated_like_a_skip_in_strict_mode():
    nj = passed("performance.ic_stability",
                "only 3 windows - stability not judged", not_judged=True)
    ok = passed("performance.sample_size", "1000 bars")
    rep = AuditReport(results=[nj, ok])
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=["performance"])
    msg = str(ei.value)
    assert "1 not-judged PASS" in msg
    assert ("performance.ic_stability (PASS, not judged): only 3 windows"
            in msg)
    rep.gate(require=["performance"], allow_partial=True)      # permissive
    # a pattern resolving only to not-judged PASSes fails in both modes
    with pytest.raises(AuditFailure, match="matched only unjudged"):
        AuditReport(results=[nj]).gate(require=["performance"],
                                       allow_partial=True)


def test_unresolved_pass_requires_explicit_gate_acceptance():
    unres = passed("lookahead.same_bar_bleed",
                   "mean partial corr 0.08 above bar, t=1.2 - UNRESOLVED",
                   unresolved=True)
    with pytest.raises(AuditFailure, match="allow_unresolved=True"):
        AuditReport(results=[unres]).gate(require=["lookahead"])
    AuditReport(results=[unres]).gate(require=["lookahead"],
                                      allow_unresolved=True)
    rep = AuditReport(results=[unres, skipped("lookahead.fvs", "needs pos")])
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=["lookahead"])
    msg = str(ei.value)
    assert "left unresolved" in msg and "lookahead.same_bar_bleed" in msg


def test_error_only_pattern_names_the_crash():
    err = CheckResult("performance.<module>", Status.ERROR, "crashed",
                      severity=Severity.HIGH)
    with pytest.raises(AuditFailure) as ei:
        AuditReport(results=[err]).gate(require=["performance"])
    msg = str(ei.value)
    assert "1 ERROR" in msg and "proves nothing" in msg


def test_gate_still_does_not_mutate_the_report():
    rep = AuditReport(results=_skips(3) + [passed("dynamic.x", "fine")])
    with pytest.raises(AuditFailure):
        rep.gate(require=["dynamic"])
    assert len(rep.results) == 4
    assert all(r.check != "audit.gate_requirement_unmet" for r in rep.results)


# Required-check semantics through complete audit calls.

def _small_case():
    market = synthetic.simulate_market(n_assets=20, n_periods=600, seed=0)
    signals = synthetic.momentum_signal(market["returns"])
    return synthetic._assemble("small_clean", "small clean", market, signals,
                               signal_func=synthetic.momentum_signal,
                               backtest_func=synthetic.make_backtest_func())


def test_attack_dead_backtest_func_dynamic_report_fails_closed():
    # An all-zero callback degenerates every null run. Its probe-health warning cannot
    # satisfy a strict dynamic-family requirement.
    case = _small_case()

    def dead(signals, asset_returns):
        return pd.Series(0.0, index=asset_returns.index)

    rep = audit(case.artifacts, AuditConfig(seed=1, n_placebo=10,
                                            n_shuffle=10),
                include=["dynamic"], backtest_func=dead)
    assert rep.ok
    assert rep["dynamic.probe_health"].status is Status.WARN
    with pytest.raises(AuditFailure, match="never judged"):
        rep.gate(require=["dynamic"], warn_severity=Severity.HIGH)


def test_honest_universe_book_is_silent_under_strict_survivorship():
    # survivorship.full_history_universe SKIPs on purpose when a universe
    # is given (superseded by no_exits): the flag keeps the family green
    case = synthetic.make_clean()
    assert case.artifacts.universe is not None
    rep = audit(case.artifacts, include=["survivorship"])
    r = rep["survivorship.full_history_universe"]
    assert r.status is Status.SKIP and r.details["superseded"] is True
    assert rep.ok and not rep.failures
    rep.gate(require=["survivorship"], warn_severity=Severity.HIGH)


def test_positions_less_book_fails_closed_on_family_patterns():
    # honest but minimal: signals + returns only -> lookahead/costs each
    # have SKIPped siblings; the family pattern is unmet, the exact ids of
    # the checks that ran are not
    case = _small_case()
    art = dataclasses.replace(case.artifacts, positions=None,
                              strategy_returns=None, universe=None)
    rep = audit(art, include=["lookahead", "costs"])
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=["lookahead", "costs"])
    msg = str(ei.value)
    assert "lookahead.position_signal_alignment" in msg
    assert "costs.turnover_unrealistic" in msg
    rep.gate(require=["lookahead.embedded_future_return",
                      "costs.no_cost_declaration"])          # exact ids
    rep.gate(require=["lookahead", "costs"], allow_partial=True)


# Verdict flags in properties, summaries, Markdown, and dictionaries.

def _flagged_report() -> AuditReport:
    return AuditReport(results=[
        passed("lookahead.a", "above bar, t=1.1 - UNRESOLVED", unresolved=True),
        passed("lookahead.b", "above bar, not exonerated", unresolved=True),
        passed("performance.c", "3 windows - not judged", not_judged=True),
        passed("costs.d", "fine"),
        skipped("dynamic.e", "pass backtest_func"),
    ], meta={"n_periods": 100, "n_assets": 5, "signal_lag": 1})


def test_unresolved_and_not_judged_properties():
    rep = _flagged_report()
    assert [r.check for r in rep.unresolved] == ["lookahead.a", "lookahead.b"]
    assert [r.check for r in rep.not_judged] == ["performance.c"]
    assert len(rep.passes) == 4             # they are still PASSes
    assert rep.ok


def test_summary_appends_flag_counts_after_the_stable_prefix():
    rep = _flagged_report()
    s = rep.summary()
    assert s.startswith("qaudit: CLEAN - 5 checks (4 PASS, 1 SKIP)")
    assert s.endswith(" - 2 unresolved, 1 not judged")
    # honest guard: no flags -> no suffix, exact unflagged string
    plain = AuditReport(results=[passed("costs.d", "fine")])
    assert plain.summary() == "qaudit: CLEAN - 1 checks (1 PASS)"


def test_to_markdown_and_to_dict_carry_the_flags():
    rep = _flagged_report()
    md = rep.to_markdown()
    assert "| PASS (unresolved) | info | `lookahead.a` |" in md
    assert "| PASS (not judged) | info | `performance.c` |" in md
    assert "| PASS | info | `costs.d` |" in md
    d = rep.to_dict()
    by = {r["check"]: r for r in d["results"]}
    assert by["lookahead.a"]["status"] == "pass"       # status unchanged
    assert by["lookahead.a"]["details"]["unresolved"] is True
    assert by["performance.c"]["details"]["not_judged"] is True


def test_full_audit_exposes_flag_properties_without_crashing():
    case = synthetic.make_clean()
    rep = audit(case.artifacts, include=["lookahead", "costs", "performance",
                                         "survivorship"])
    assert isinstance(rep.unresolved, list)
    assert isinstance(rep.not_judged, list)
    assert all(r.status is Status.PASS for r in rep.unresolved + rep.not_judged)
    assert rep.summary().startswith("qaudit: ")
    rep.to_markdown()


# A zero signal lag conflicts with the end-of-bar timing contract.

def test_attack_planted_same_bar_book_declared_lag_zero_is_rejected_at_intake():
    case = synthetic.make_same_bar_execution(seed=0)
    art0 = dataclasses.replace(case.artifacts, signal_lag=0)
    with pytest.raises(InputValidationError, match=r"signal_lag.*>= 1") as ei:
        audit(art0, include=["lookahead"])
    assert "remap" in str(ei.value) and "shift positions" in str(ei.value)


def test_honest_declaration_of_the_same_book_has_no_stand_down_note():
    case = synthetic.make_same_bar_execution(seed=0)
    rep = audit(case.artifacts, include=["lookahead"])     # signal_lag=1
    assert "stood down" not in str(rep)
    assert "stood down" not in rep.to_markdown()
    assert not rep.ok and rep.failures        # the planted bug is caught


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_report_never_advertises_a_zero_lag_stand_down(lag):
    rep = AuditReport(results=[passed("lookahead.x", "ok")],
                      meta={"n_periods": 10, "n_assets": 2, "signal_lag": lag})
    assert "same-bar detectors stood down" not in str(rep)
    assert "same-bar detectors stood down" not in rep.to_markdown()
    assert rep.summary() == "qaudit: CLEAN - 1 checks (1 PASS)"


def test_stand_down_note_absent_without_meta():
    rep = AuditReport(results=[passed("lookahead.x", "ok")])
    assert "stood down" not in str(rep)
