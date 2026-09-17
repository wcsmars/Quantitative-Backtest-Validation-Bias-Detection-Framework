"""Finite input coverage, module-output contracts, and fail-closed report gates.

Missing evidence and partial family output cannot satisfy strict deployment
requirements. Tests also cover universe value domains, filter validation,
and complete, deterministic provenance metadata.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import qaudit
import qaudit.api as api
from qaudit import (AuditFailure, BacktestArtifacts, CheckRuntimeError,
                    InputValidationError, MisalignedInputError, Severity,
                    Status, audit)
from qaudit.report import AuditReport
from qaudit.types import CheckResult


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------

def _panels(n=250, k=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"A{i:02d}" for i in range(k)]
    sig = pd.DataFrame(rng.standard_normal((n, k)), index=idx, columns=cols)
    ret = pd.DataFrame(rng.standard_normal((n, k)) * 0.01,
                       index=idx, columns=cols)
    return sig, ret


def _dense_art(seed=0):
    sig, ret = _panels(seed=seed)
    return BacktestArtifacts(signals=sig, asset_returns=ret)


# ===========================================================================
# 1. finite-coverage gate in validate()
# ===========================================================================

def test_all_nan_required_panels_rejected():
    # All-NaN required panels have no finite evidence to audit.
    idx = pd.bdate_range("2020-01-01", periods=250)
    nan = pd.DataFrame(np.nan, index=idx, columns=[f"A{i}" for i in range(5)])
    art = BacktestArtifacts(signals=nan.copy(), asset_returns=nan.copy())
    with pytest.raises(InputValidationError, match="100% NaN"):
        art.validate()


def test_all_nan_signals_only_rejected_naming_the_panel():
    sig, ret = _panels()
    sig.iloc[:, :] = np.nan
    art = BacktestArtifacts(signals=sig, asset_returns=ret)
    with pytest.raises(InputValidationError, match=r"artifacts\.signals is "):
        art.validate()


def test_all_nan_rich_book_rejected():
    # All-NaN optional panels must not create apparent coverage.
    idx = pd.bdate_range("2020-01-01", periods=250)
    cols = [f"A{i}" for i in range(12)]
    nan = pd.DataFrame(np.nan, index=idx, columns=cols)
    art = BacktestArtifacts(
        signals=nan.copy(), asset_returns=nan.copy(), positions=nan.copy(),
        strategy_returns=pd.Series(np.nan, index=idx), prices=nan.copy(),
        train_period=(idx[0], idx[119]), test_period=(idx[130], idx[-1]),
        declared_costs_bps=5.0)
    with pytest.raises(InputValidationError, match="100% NaN"):
        art.validate()


def test_disjoint_finite_content_rejected():
    # each panel has plenty of content, but never on the same date: the
    # calendar overlaps, the observable content does not
    sig, ret = _panels()
    sig.iloc[100:, :] = np.nan      # signals live on dates [0, 100)
    ret.iloc[:100, :] = np.nan      # returns live on dates [100, 250)
    art = BacktestArtifacts(signals=sig, asset_returns=ret)
    with pytest.raises(MisalignedInputError, match="jointly finite"):
        art.validate()


def _sparse_usable_art(n_usable, n=250, k=12, seed=3):
    """signals finite on exactly n_usable evenly spread dates, NaN elsewhere;
    returns dense - usable joint dates == n_usable."""
    sig, ret = _panels(n=n, k=k, seed=seed)
    keep = np.linspace(0, n - 1, n_usable).round().astype(int)
    mask = np.ones(n, dtype=bool)
    mask[keep] = False
    sig.iloc[mask, :] = np.nan
    return BacktestArtifacts(signals=sig, asset_returns=ret)


def test_jointly_finite_hard_gate_is_zero_tolerance_only():
    # Intake rejects zero jointly finite coverage. Low but nonzero coverage remains
    # legal and is disclosed by the audit; individual checks can skip for insufficient
    # support.
    _sparse_usable_art(1).validate()    # must not raise
    _sparse_usable_art(29).validate()   # must not raise


def test_starved_joint_content_disclosed_by_audit_coverage():
    # 1 <= usable < 30: validates, but the report must not headline an
    # unqualified CLEAN - audit.coverage WARNs with the usable count, and
    # gate(warn_severity=HIGH) turns that into a deployment block
    rep = audit(_sparse_usable_art(13))
    r = rep["audit.coverage"]
    assert r.status is Status.WARN and r.severity is Severity.HIGH
    assert "13 of 250" in r.message
    assert r.details["usable_joint_dates"] == 13
    with pytest.raises(AuditFailure, match="audit.coverage"):
        rep.gate(warn_severity=Severity.HIGH)
    # at/above the 30-date support floor with full axes kept: no advisory
    rep_ok = audit(_sparse_usable_art(30))
    assert all(c.check != "audit.coverage" for c in rep_ok.results)


def test_honest_sparse_panels_keep_validating_mc():
    # Event-gated exports with staggered listings are legitimate sparse layouts; the
    # seeded controls must validate.
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sig, ret = _panels(seed=seed)
        gate_mask = np.arange(len(sig)) % 3 != 0     # keep every 3rd date
        sig.iloc[gate_mask, :] = np.nan              # ~83 usable dates
        for j in range(sig.shape[1]):                # late listings
            start = int(rng.integers(0, int(0.4 * len(sig))))
            sig.iloc[:start, j] = np.nan
            ret.iloc[:start, j] = np.nan
        BacktestArtifacts(signals=sig, asset_returns=ret).validate()


# ===========================================================================
# 2. universe value-domain gate
# ===========================================================================

@pytest.mark.parametrize("bad_value", [-1.0, 0.5, 2.0])
def test_nonbinary_universe_values_rejected(bad_value):
    # A -1 membership sentinel must not become True through generic boolean coercion.
    sig, ret = _panels()
    uni = pd.DataFrame(1.0, index=sig.index, columns=sig.columns)
    uni.iloc[50:, 3] = bad_value
    art = BacktestArtifacts(signals=sig, asset_returns=ret, universe=uni)
    with pytest.raises(InputValidationError,
                       match=r"artifacts\.universe .*outside") as ei:
        art.validate()
    assert str(bad_value) in str(ei.value)   # names the offending value


def test_binary_and_nan_universes_still_validate():
    # honest guards: bool, float {0,1}, and scattered-NaN universes all
    # keep validating - NaN cells stay legal and decode to False by the
    # aligned() contract (and are not gated here)
    sig, ret = _panels()
    idx, cols = sig.index, sig.columns

    uni_bool = pd.DataFrame(True, index=idx, columns=cols)
    uni_bool.iloc[100:, 0] = False
    BacktestArtifacts(signals=sig, asset_returns=ret,
                      universe=uni_bool).validate()

    uni_float = pd.DataFrame(1.0, index=idx, columns=cols)
    uni_float.iloc[100:, 0] = 0.0
    BacktestArtifacts(signals=sig, asset_returns=ret,
                      universe=uni_float).validate()

    uni_nan = uni_float.copy()
    uni_nan.iloc[::17, 2] = np.nan          # scattered NaN cells
    art = BacktestArtifacts(signals=sig, asset_returns=ret, universe=uni_nan)
    art.validate()
    assert not art.aligned().universe.iloc[0, 2]   # NaN -> False


# ===========================================================================
# 3. audit.coverage sliver advisory
# ===========================================================================

def _asset_sliver_art(seed=7):
    # Twenty columns per panel share exactly one asset.
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=750)
    sig = pd.DataFrame(rng.normal(size=(750, 20)), index=idx,
                       columns=["SHARED"] + [f"S{i}" for i in range(19)])
    ret = pd.DataFrame(rng.normal(0, 0.01, size=(750, 20)), index=idx,
                       columns=["SHARED"] + [f"R{i}" for i in range(19)])
    return BacktestArtifacts(signals=sig, asset_returns=ret)


def test_asset_sliver_carries_coverage_warn():
    # 1-of-20 assets survives the join: a HIGH WARN must name the discard
    # with the numbers rather than certify CLEAN
    rep = audit(_asset_sliver_art())
    r = rep["audit.coverage"]
    assert r.status is Status.WARN and r.severity is Severity.HIGH
    assert "kept 1 of 20" in r.message
    assert r.details["shortfalls"]["signals.columns"]["kept"] == 1
    assert r.details["shortfalls"]["asset_returns.columns"]["total"] == 20
    assert rep.ok            # advisory contract: WARN never flips ok


def test_date_sliver_dense_block_carries_coverage_warn():
    # Five hundred-date panels share a dense 30-business-day block.
    rng = np.random.default_rng(7)
    full = pd.bdate_range("2018-01-01", periods=1000)
    cols = [f"A{i}" for i in range(12)]
    sig = pd.DataFrame(rng.normal(size=(500, 12)), index=full[:500],
                       columns=cols)
    ret = pd.DataFrame(rng.normal(0, 0.01, size=(500, 12)),
                       index=full[470:970], columns=cols)
    art = BacktestArtifacts(signals=sig, asset_returns=ret)
    art.validate()                       # 30 jointly-finite dates: at floor
    rep = audit(art)
    r = rep["audit.coverage"]
    assert r.status is Status.WARN
    assert "kept 30 of 500" in r.message
    assert rep.meta["n_periods"] == 30


def test_scattered_date_sliver_still_ppy_rejected():
    # the scattered 30-of-500 sliver is rejected by the periods_per_year
    # gate (observed ~15 bars/yr vs declared 252) - pin that protection
    # deliberately so it cannot be optimized away
    rng = np.random.default_rng(7)
    full = pd.bdate_range("2018-01-01", periods=1000)
    cols = [f"A{i}" for i in range(12)]
    sig_idx = full[:500]
    ret_idx = sig_idx[::17].append(full[500:970])   # 30 scattered overlaps
    sig = pd.DataFrame(rng.normal(size=(500, 12)), index=sig_idx,
                       columns=cols)
    ret = pd.DataFrame(rng.normal(0, 0.01, size=(len(ret_idx), 12)),
                       index=ret_idx, columns=cols)
    art = BacktestArtifacts(signals=sig, asset_returns=ret)
    with pytest.raises(InputValidationError, match="periods_per_year"):
        art.validate()


def test_coverage_warn_survives_include_filter():
    # Coverage disclosure is appended after include/exclude filtering so filtered
    # audits retain it.
    rep = audit(_asset_sliver_art(), include=["costs"])
    assert any(r.check == "audit.coverage" for r in rep.results)


def test_dense_book_gets_no_coverage_warn_and_no_module_errors():
    # honest guard: a fully aligned book gets no synthetic coverage result
    # and no dispatch-validation ERRORs
    rep = audit(_dense_art())
    assert all(r.check != "audit.coverage" for r in rep.results)
    assert rep.errors == []


# ===========================================================================
# 4. report.gate() - fail-closed deployment API
# ===========================================================================

def test_gate_all_skip_report_fails_closed():
    # An all-SKIP dynamic report may remain advisory through ok and raise_for_failures;
    # the strict gate must reject its missing evidence.
    rep = audit(_dense_art(), include=["dynamic"])
    assert rep.ok and "CLEAN" in rep.summary()          # advisory path: CLEAN
    rep.raise_for_failures()                            # advisory path: no raise
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=["dynamic"])
    msg = str(ei.value)
    # embeds the SKIP messages verbatim - they name what to pass to arm
    assert "matched only SKIPped" in msg
    assert "backtest_func" in msg or "signal_func" in msg
    assert "dynamic.date_shift" in msg


def test_gate_require_matching_nothing_names_families():
    rep = audit(_dense_art(), include=["costs"])
    with pytest.raises(AuditFailure, match="matched no check"):
        rep.gate(require=["no_such_family_zzz"])
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=["no_such_family_zzz"])
    assert "costs" in str(ei.value)     # names the families present


_DENSE_COSTS_SKIPS = ["costs.missing_transaction_costs",
                      "costs.turnover_unrealistic", "costs.cost_sensitivity",
                      "costs.price_return_consistency"]
_DENSE_LOOKAHEAD_SKIPS = ["lookahead.position_signal_alignment",
                          "lookahead.same_bar_bleed",
                          "lookahead.future_vol_sizing",
                          "lookahead.same_bar_return_loading"]


def test_gate_strict_family_pattern_with_skipped_siblings_raises():
    # Partially judged costs and lookahead families cannot satisfy strict family
    # requirements. A judged exact check ID remains usable.
    rep = audit(_dense_art())
    assert rep.warnings                 # e.g. costs.no_cost_declaration
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=["costs", "performance.sample_size", "lookahead"])
    msg = str(ei.value)
    assert "require pattern 'costs' matched 5 check(s) but 4" in msg
    assert "require pattern 'lookahead' matched 8 check(s) but 4" in msg
    for cid in _DENSE_COSTS_SKIPS + _DENSE_LOOKAHEAD_SKIPS:
        assert cid in msg
    assert "performance.sample_size" not in msg      # judged, not unmet
    assert "allow_partial=True" in msg
    rep.gate(require=["performance.sample_size",
                      "costs.no_cost_declaration"])     # exact ids: silent


def test_gate_allow_partial_satisfied_by_judged_checks_is_silent():
    # honest guard for the permissive mode: PASS/WARN/FAIL all count as
    # "the check ran"; plain WARNs do not raise without warn_severity
    rep = audit(_dense_art())
    assert rep.warnings
    rep.gate(require=["costs", "performance.sample_size", "lookahead"],
             allow_partial=True)


def test_gate_warn_severity_blocks_high_warns():
    rep = audit(_asset_sliver_art())    # carries the HIGH coverage WARN
    rep.gate()                          # default: WARNs never block
    with pytest.raises(AuditFailure, match="audit.coverage"):
        rep.gate(warn_severity=Severity.HIGH)
    rep.gate(warn_severity=Severity.CRITICAL)   # nothing critical: silent


def test_gate_min_severity_and_errors():
    low_fail = CheckResult("costs.x", Status.FAIL, "small", severity=Severity.LOW)
    err = CheckResult("leakage.<module>", Status.ERROR, "boom",
                      severity=Severity.HIGH)
    rep = AuditReport(results=[low_fail])
    rep.gate()                                        # LOW FAIL < MEDIUM bar
    with pytest.raises(AuditFailure):
        rep.gate(min_severity=Severity.LOW)
    with pytest.raises(AuditFailure):                 # ERRORs always raise
        AuditReport(results=[err]).gate()


def test_gate_rejects_bare_string_require():
    # "lookahead" iterated char by char would self-satisfy the gate
    rep = audit(_dense_art(), include=["costs"])
    with pytest.raises(InputValidationError, match="require"):
        rep.gate(require="lookahead")


def test_gate_does_not_mutate_the_report():
    rep = audit(_dense_art(), include=["dynamic"])
    n = len(rep.results)
    with pytest.raises(AuditFailure):
        rep.gate(require=["dynamic"])
    assert len(rep.results) == n
    assert all(r.check != "audit.gate_requirement_unmet" for r in rep.results)


# ===========================================================================
# 5. module run() output validation at dispatch
# ===========================================================================

def test_empty_module_output_is_module_error(monkeypatch):
    monkeypatch.setattr("qaudit.checks.costs.run", lambda *a, **k: [])
    rep = audit(_dense_art(), include=["costs"])
    assert not rep.ok
    (r,) = rep.errors
    assert r.check == "costs.<module>"
    assert "zero results" in r.message


def test_all_modules_empty_is_not_clean(monkeypatch):
    # All modules returning empty output must produce failure evidence.
    for m in api.CHECK_MODULES:
        monkeypatch.setattr(f"{m}.run", lambda *a, **k: [])
    rep = audit(_dense_art())
    assert not rep.ok
    assert len(rep.errors) == len(api.CHECK_MODULES)
    assert "SUSPECT" in rep.summary()


def test_garbage_module_output_is_branded_error(monkeypatch):
    # Malformed module output must be contained before filtering or rendering.
    monkeypatch.setattr("qaudit.checks.costs.run",
                        lambda *a, **k: ["not a CheckResult", 42])
    rep = audit(_dense_art(), include=["costs"])
    (r,) = rep.errors
    assert "non-CheckResult" in r.message and "str" in r.message
    str(rep)                                     # rendering must survive
    json.dumps(rep.to_dict(), allow_nan=False)


def test_str_module_output_is_branded_error(monkeypatch):
    # list("oops") would have extended char by char
    monkeypatch.setattr("qaudit.checks.costs.run", lambda *a, **k: "oops")
    rep = audit(_dense_art(), include=["costs"])
    (r,) = rep.errors
    assert "must return a list" in r.message


def test_bad_module_output_raises_in_strict_mode(monkeypatch):
    monkeypatch.setattr("qaudit.checks.costs.run", lambda *a, **k: [])
    with pytest.raises(CheckRuntimeError, match="zero results"):
        audit(_dense_art(), include=["costs"], strict=True)


# ===========================================================================
# 6. per-pattern filter accounting + dispatch economy
# ===========================================================================

def test_mixed_include_typo_is_loud():
    # A valid include pattern combined with a misspelled family must report the invalid
    # pattern.
    rep = audit(_dense_art(), include=["costs", "surviorship"])
    assert not rep.ok
    r = rep["audit.filter_pattern_matched_nothing"]
    assert r.status is Status.ERROR and r.severity is Severity.HIGH
    assert "surviorship" in r.message
    assert "survivorship" in r.remediation      # lists valid prefixes
    assert any(c.check.startswith("costs.") for c in rep.results)
    with pytest.raises(AuditFailure):
        rep.raise_for_failures()


def test_all_typo_include_still_single_aggregate_error():
    # When every pattern is unmatched, emit one aggregate filter error.
    rep = audit(_dense_art(), include=["surviorship"])
    assert len(rep.results) == 1
    assert rep.results[0].check == "audit.filter_matched_nothing"


def test_unknown_exclude_gets_warn_note_not_error():
    # an exclude typo fails toward more auditing - WARN/LOW, ok untouched
    rep = audit(_dense_art(), exclude=["surviorship"])
    r = rep["audit.exclude_pattern_matched_nothing"]
    assert r.status is Status.WARN and r.severity is Severity.LOW
    assert "surviorship" in r.message
    assert rep.ok
    # ...and the audit itself is complete (nothing was dropped)
    assert any(c.check.startswith("survivorship.") for c in rep.results)


def test_valid_patterns_get_no_synthetic_note():
    # honest guards: correctly spelled filters must stay note-free
    rep_i = audit(_dense_art(), include=["costs"])
    rep_e = audit(_dense_art(), exclude=["survivorship"])
    for rep in (rep_i, rep_e):
        assert all(r.check not in ("audit.filter_pattern_matched_nothing",
                                   "audit.exclude_pattern_matched_nothing",
                                   "audit.filter_matched_nothing")
                   for r in rep.results)


def test_typo_pattern_does_not_dispatch_all_modules():
    # A sibling filter typo must not cause unrelated modules to invoke expensive
    # pipeline callbacks.
    assert api._select_modules(["costs", "surviorship"], None) == \
        ["qaudit.checks.costs"]
    assert api._select_modules(["surviorship"], None) == []


# ===========================================================================
# 7. report.meta["provenance"]
# ===========================================================================

def test_provenance_present_complete_and_json_safe():
    rep = audit(_dense_art())
    prov = rep.meta["provenance"]
    assert prov["qaudit_version"] == qaudit.__version__
    assert len(prov["config_hash"]) == 64
    assert int(prov["config_hash"], 16) >= 0            # hex sha256
    assert prov["checks_selected_modules"] == list(api.CHECK_MODULES)
    assert prov["checks_emitted"] == [r.check for r in rep.results]
    assert set(prov["timings_s"]) == set(api.CHECK_MODULES)
    assert all(t >= 0 for t in prov["timings_s"].values())
    assert set(prov["deps"]) == {"python", "numpy", "pandas", "scipy"}
    assert prov["include"] is None and prov["strict"] is False
    cov = prov["artifact_coverage"]
    assert cov["signals"]["shape"] == [250, 12]
    assert cov["signals"]["non_null_frac"] == pytest.approx(1.0)
    assert cov["positions"] is None
    exp = prov["checks_expected"]["qaudit.checks.costs"]
    assert "costs.cost_sensitivity" in exp
    # Input digests and callable labels accompany the provenance context.
    dig = prov["artifact_digest"]
    assert set(dig) == {"signals", "asset_returns", "positions",
                        "strategy_returns", "universe", "prices",
                        "signal_input"}
    assert len(dig["signals"]) == 64 and dig["positions"] is None
    assert prov["callables"] == {"signal_func": None, "backtest_func": None}
    json.dumps(rep.to_dict(), allow_nan=False)          # round-trip safe


def test_provenance_deterministic_across_runs():
    a = audit(_dense_art()).meta["provenance"]
    b = audit(_dense_art()).meta["provenance"]
    assert a["config_hash"] == b["config_hash"]
    assert a["checks_emitted"] == b["checks_emitted"]
    assert a["config"] == b["config"]
    assert a["artifact_digest"] == b["artifact_digest"]
    # ...and a different book is a different record (tests/
    # test_qaudit_provenance_digests.py holds the full attack set)
    c = audit(_dense_art(seed=1)).meta["provenance"]
    assert c["artifact_digest"]["signals"] != a["artifact_digest"]["signals"]
    assert c["config_hash"] == a["config_hash"]


def test_provenance_config_snapshot_is_fields_only():
    prov = audit(_dense_art(), include=["costs"]).meta["provenance"]
    assert "_validation_active" not in prov["config"]
    assert prov["config"]["sharpe_warn"] == 3.0
    assert prov["config"]["cost_sensitivity_bps"] == [5.0, 10.0, 25.0]
    assert prov["include"] == ["costs"]


def test_provenance_times_a_crashing_module(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaput")
    monkeypatch.setattr("qaudit.checks.costs.run", boom)
    rep = audit(_dense_art(), include=["costs"])
    assert "qaudit.checks.costs" in rep.meta["provenance"]["timings_s"]
    assert rep.errors                       # time-to-crash still recorded


def test_provenance_failure_never_kills_the_audit(monkeypatch):
    # the verdict outranks the nicety: a broken serializer degrades the
    # block to {"error": ...}; a consumer gating on config_hash catches it
    def broken(v):
        raise RuntimeError("serializer down")
    monkeypatch.setattr("qaudit.report._json_clean", broken)
    rep = audit(_dense_art(), include=["costs"])
    prov = rep.meta["provenance"]
    assert "config_hash" not in prov
    assert "provenance construction failed" in prov["error"]
    assert rep.results                      # audit itself unharmed
