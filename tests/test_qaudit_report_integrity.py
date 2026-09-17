"""Dispatch, gate, provenance, and report-rendering integrity.

Coverage includes partial module output, filtered deployment requirements,
unambiguous provenance framing, deterministic serialization, empty patterns,
and HTML from callback exception text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qaudit.api as api
from qaudit import (AuditConfig, AuditFailure, BacktestArtifacts,
                    CheckRuntimeError, InputValidationError, audit)
from qaudit.config import (_MAX_COST_SWEEP_POINTS, _MAX_DYNAMIC_RUNS,
                           _MAX_FORWARD_HORIZON)
from qaudit.report import AuditReport
from qaudit.report import _json_clean
from qaudit.types import CheckResult, Severity, Status, passed
from qaudit.types import failed, skipped


def _dense_art(seed: int = 0) -> BacktestArtifacts:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=250)
    cols = [f"A{i:02d}" for i in range(12)]
    signals = pd.DataFrame(rng.standard_normal((250, 12)),
                           index=idx, columns=cols)
    returns = pd.DataFrame(rng.standard_normal((250, 12)) * 0.01,
                           index=idx, columns=cols)
    return BacktestArtifacts(signals=signals, asset_returns=returns)


def test_nonempty_partial_module_output_is_a_branded_error(monkeypatch):
    monkeypatch.setattr(
        "qaudit.checks.costs.run",
        lambda *args, **kwargs: [passed("costs.no_cost_declaration", "only")],
    )
    report = audit(_dense_art(), include=["costs"])
    assert not report.ok
    (error,) = report.errors
    assert error.check == "costs.<module>"
    assert "partial family output" in error.message
    assert "costs.cost_sensitivity" in error.message


def test_nonempty_partial_module_output_raises_in_strict_mode(monkeypatch):
    monkeypatch.setattr(
        "qaudit.checks.costs.run",
        lambda *args, **kwargs: [passed("costs.no_cost_declaration", "only")],
    )
    with pytest.raises(CheckRuntimeError, match="partial family output"):
        audit(_dense_art(), include=["costs"], strict=True)


def test_filtered_exact_result_cannot_satisfy_broad_family_gate():
    report = audit(_dense_art(), include=["costs.no_cost_declaration"])
    assert [r.check for r in report.results] == ["costs.no_cost_declaration"]
    report.gate(require=["costs.no_cost_declaration"])
    with pytest.raises(AuditFailure) as caught:
        report.gate(require=["costs"])
    message = str(caught.value)
    assert "NOT EMITTED" in message
    assert "costs.cost_sensitivity" in message
    # Deliberate escape hatch remains explicit.
    report.gate(require=["costs"], allow_partial=True)


@pytest.mark.parametrize("which, value", [
    ("include", [""]),
    ("include", ["   "]),
    ("exclude", [""]),
    ("exclude", ["\t"]),
])
def test_audit_rejects_empty_or_whitespace_patterns(which, value):
    kwargs = {which: value}
    with pytest.raises(InputValidationError, match="non-empty"):
        audit(_dense_art(), **kwargs)


@pytest.mark.parametrize("pattern", ["", "   ", "\t"])
def test_gate_rejects_empty_or_whitespace_patterns(pattern):
    report = AuditReport([passed("costs.x", "ok")])
    with pytest.raises(InputValidationError, match="non-empty"):
        report.gate(require=[pattern])


def test_expected_registry_is_outside_best_effort_provenance():
    report = audit(_dense_art(), include=["costs.no_cost_declaration"])
    assert report.meta["expected_check_ids"] == list(
        api.MODULE_CHECK_IDS["qaudit.checks.costs"]
    )


def test_provenance_identifies_package_source_and_callable_code():
    report = audit(_dense_art(), include=["costs"])
    provenance = report.meta["provenance"]
    assert len(provenance["qaudit_source_digest"]) == 64
    assert provenance["qaudit_source_digest"] == api._package_digest()
    assert provenance["callable_digest"] == {
        "signal_func": None, "backtest_func": None,
    }


def test_deployment_gate_blocks_degraded_provenance(monkeypatch):
    monkeypatch.setattr(api, "_package_digest",
                        lambda: (_ for _ in ()).throw(RuntimeError("no source")))
    report = audit(_dense_art(), include=["costs"])
    assert "error" in report.meta["provenance"]
    with pytest.raises(AuditFailure, match="provenance.*unavailable"):
        report.gate()


def test_deployment_gate_blocks_incomplete_or_tampered_provenance():
    report = audit(_dense_art(), include=["costs"])
    report.meta["provenance"] = {}
    with pytest.raises(AuditFailure, match="incomplete provenance"):
        report.gate()

    report = audit(_dense_art(), include=["costs"])
    report.meta["provenance"]["config"]["seed"] = 99
    with pytest.raises(AuditFailure, match="config_hash does not match"):
        report.gate()


def test_deployment_gate_blocks_internally_contradictory_provenance():
    report = audit(_dense_art(), include=["costs"])
    report.meta["provenance"]["callables"]["signal_func"] = "module.func"
    with pytest.raises(AuditFailure, match="fingerprint.*unavailable"):
        report.gate()

    report = audit(_dense_art(), include=["costs"])
    report.meta["provenance"]["artifact_coverage"]["positions"] = dict(
        report.meta["provenance"]["artifact_coverage"]["signals"])
    with pytest.raises(AuditFailure, match="presence disagrees"):
        report.gate()

    report = audit(_dense_art(), include=["costs"])
    report.meta["provenance"]["checks_expected"] = {
        module: []
        for module in report.meta["provenance"]["checks_selected_modules"]}
    with pytest.raises(AuditFailure, match="selected registry"):
        report.gate()

    report = audit(_dense_art(), include=["costs"])
    report.meta["provenance"]["artifact_coverage"]["signals"] = {}
    with pytest.raises(AuditFailure, match="coverage stats.*malformed"):
        report.gate()

    report = audit(_dense_art(), include=["costs"])
    module = report.meta["provenance"]["checks_selected_modules"][0]
    report.meta["provenance"]["timings_s"][module] = float("nan")
    with pytest.raises(AuditFailure, match="timings_s.*inconsistent"):
        report.gate()


def test_panel_digest_distinguishes_label_types():
    idx = pd.date_range("2024-01-01", periods=2)
    integer = pd.DataFrame([[1.0], [2.0]], index=idx, columns=[1])
    string = pd.DataFrame([[1.0], [2.0]], index=idx, columns=["1"])
    assert api._panel_digest(integer) != api._panel_digest(string)


def test_panel_digest_frames_labels_instead_of_joining_with_a_delimiter():
    idx = pd.date_range("2024-01-01", periods=1)
    left = pd.DataFrame([[1.0, 2.0]], index=idx,
                        columns=["a\x1fb", "c"])
    right = pd.DataFrame([[1.0, 2.0]], index=idx,
                         columns=["a", "b\x1fc"])
    assert api._panel_digest(left) != api._panel_digest(right)


def test_panel_digest_distinguishes_object_scalar_types_and_axis_names():
    idx = pd.date_range("2024-01-01", periods=1)
    byte_cell = pd.DataFrame([[b"a"]], index=idx, columns=["x"])
    text_cell = pd.DataFrame([["a"]], index=idx, columns=["x"])
    assert api._panel_digest(byte_cell) != api._panel_digest(text_cell)

    named = pd.DataFrame([[1.0]], index=idx, columns=["x"])
    renamed = named.copy()
    renamed.index.name = "observation_date"
    assert api._panel_digest(named) != api._panel_digest(renamed)
    renamed = named.copy()
    renamed.columns.name = "asset"
    assert api._panel_digest(named) != api._panel_digest(renamed)


def test_panel_digest_rejects_opaque_colliding_labels_fail_closed():
    class Label:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return hash(self.value)

        def __eq__(self, other):
            return isinstance(other, Label) and self.value == other.value

        def __repr__(self):
            return "<asset>"

    idx = pd.date_range("2024-01-01", periods=250)
    label = Label(1)
    signals = pd.DataFrame(np.ones((250, 1)), index=idx, columns=[label])
    returns = pd.DataFrame(np.zeros((250, 1)), index=idx, columns=[label])
    with pytest.raises(TypeError, match="unsupported opaque panel"):
        api._panel_digest(signals)
    report = audit(BacktestArtifacts(signals, returns), include=["costs"])
    assert "error" in report.meta["provenance"]
    with pytest.raises(AuditFailure, match="provenance.*unavailable"):
        report.gate()


def test_callable_digest_includes_bound_instance_state():
    class Pipeline:
        def __init__(self, bps):
            self.bps = bps

        def run(self, signals, returns):
            return returns.mean(axis=1) - self.bps / 10_000

    assert api._callable_digest(Pipeline(0).run) != api._callable_digest(
        Pipeline(25).run)

    class SlottedPipeline:
        __slots__ = ("bps", "__dict__")

        def __init__(self, bps):
            self.bps = bps

        def run(self, signals, returns):
            return returns.mean(axis=1) - self.bps / 10_000

    assert api._callable_digest(
        SlottedPipeline(0).run) != api._callable_digest(
            SlottedPipeline(25).run)

    class CallablePipeline:
        __slots__ = ("bps",)

        def __init__(self, bps):
            self.bps = bps

        def __call__(self, signals, returns):
            return returns.mean(axis=1) - self.bps / 10_000

    assert api._callable_digest(
        CallablePipeline(0)) != api._callable_digest(CallablePipeline(25))


def test_callable_digest_includes_dataclass_type_and_stable_generator_state():
    from dataclasses import dataclass

    @dataclass
    class Left:
        value: int

    @dataclass
    class Right:
        value: int

    def factory(state):
        def pipeline(signals, returns):
            return returns.mean(axis=1) if type(state).__name__ == "Left" \
                else -returns.mean(axis=1)
        return pipeline

    assert api._callable_digest(factory(Left(1))) != api._callable_digest(
        factory(Right(1)))
    assert api._callable_digest(
        np.random.default_rng(7).normal) == api._callable_digest(
            np.random.default_rng(7).normal)

    import random
    assert api._callable_digest(
        random.Random(7).random) == api._callable_digest(
            random.Random(7).random)
    assert api._callable_digest(
        random.Random(7).random) != api._callable_digest(
            random.Random(8).random)


def test_callable_digest_distinguishes_common_immutable_captured_values():
    import datetime
    import decimal
    from pathlib import Path

    def factory(value):
        def pipeline(signals, returns):
            return returns.mean(axis=1) if str(value) else -returns.mean(axis=1)
        return pipeline

    pairs = [
        (datetime.date(2025, 1, 1), datetime.date(2026, 1, 1)),
        (decimal.Decimal("1.25"), decimal.Decimal("2.50")),
        (Path("left/model"), Path("right/model")),
    ]
    for left, right in pairs:
        assert api._callable_digest(factory(left)) != api._callable_digest(
            factory(right))


def test_callable_digest_keeps_deep_scalar_identity_and_fails_opaque_closed():
    def factory(value):
        def pipeline(signals, returns):
            return returns.mean(axis=1) if value else -returns.mean(axis=1)
        return pipeline

    left: object = 1
    right: object = 2
    for _ in range(12):
        left = [left]
        right = [right]
    assert api._callable_digest(factory(left)) != api._callable_digest(
        factory(right))

    opaque_pipeline = factory(object())
    assert api._callable_digest(opaque_pipeline) is None
    report = audit(_dense_art(), include=["dynamic"],
                   backtest_func=opaque_pipeline)
    with pytest.raises(AuditFailure, match="fingerprint.*unavailable"):
        report.gate()


def test_audit_snapshots_config_before_callbacks_mutate_caller_state():
    cfg = AuditConfig(sharpe_fail=5.0, n_placebo=20, n_shuffle=20)
    art = _dense_art()

    def mutating_backtest(signals, returns):
        cfg.sharpe_fail = 99.0
        return returns.mean(axis=1)

    report = audit(art, cfg,
                   include=["performance.suspicious_sharpe", "dynamic"],
                   backtest_func=mutating_backtest)
    assert cfg.sharpe_fail == 99.0
    assert report.meta["provenance"]["config"]["sharpe_fail"] == 5.0


def test_config_set_serialization_is_canonical():
    cfg = AuditConfig(extra={"tags": {"gamma", "alpha", "beta"}})
    first = audit(_dense_art(), cfg, include=["costs"]).meta["provenance"]
    cfg2 = AuditConfig(extra={"tags": set(["beta", "gamma", "alpha"])})
    second = audit(_dense_art(), cfg2, include=["costs"]).meta["provenance"]
    assert first["config"]["extra"]["tags"] == ["alpha", "beta", "gamma"]
    assert first["config_hash"] == second["config_hash"]


@pytest.mark.parametrize("extra", [
    {1: "hidden", "1": "shown"},
    {"nested": {None: "hidden", "None": "shown"}},
])
def test_config_rejects_non_string_mapping_keys_before_hashing(extra):
    with pytest.raises(InputValidationError, match="mapping keys must be strings"):
        AuditConfig(extra=extra)


def test_json_clean_preserves_colliding_keys_in_arbitrary_report_details():
    cleaned = _json_clean({1: "integer", "1": "string"})
    entries = cleaned["__qaudit_typed_mapping_v1__"]
    assert len(entries) == 2
    assert {(entry["key_type"], entry["value"]) for entry in entries} == {
        ("builtins.int", "integer"), ("builtins.str", "string"),
    }


def test_config_rejects_cyclic_extra_and_rolls_back_mutation():
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(InputValidationError, match="reference cycle"):
        AuditConfig(extra=cyclic)
    cfg = AuditConfig(extra={"safe": True})
    with pytest.raises(InputValidationError):
        cfg.extra = cyclic
    assert cfg.extra == {"safe": True}


def test_config_bounds_amplifying_graph_but_allows_small_shared_values():
    # Interned/immutable literals and deliberate small aliases are ordinary
    # loader output and should not be rejected merely because ids match.
    shared = (1, 2)
    AuditConfig(extra={"left": shared, "right": shared})

    # A physically tiny DAG can expand exponentially in JSON. Validation
    # follows logical occurrences but stops at a fixed node budget.
    node: object = [1]
    for _ in range(20):
        node = [node, node]
    with pytest.raises(InputValidationError, match="expanded value exceeds"):
        AuditConfig(extra={"dag": node})


def test_config_rejects_opaque_values():
    with pytest.raises(InputValidationError, match="unsupported type"):
        AuditConfig(extra={"opaque": object()})


def test_mutable_cost_sweep_is_normalized_and_revalidated_at_audit_boundary():
    cfg = AuditConfig(cost_sensitivity_bps=[5.0, 10.0])
    assert cfg.cost_sensitivity_bps == (5.0, 10.0)
    assert isinstance(cfg.cost_sensitivity_bps, tuple)
    object.__setattr__(cfg, "cost_sensitivity_bps",
                       tuple(range(_MAX_COST_SWEEP_POINTS + 1)))
    with pytest.raises(InputValidationError, match="cost_sensitivity_bps"):
        audit(_dense_art(), cfg, include=["costs"])


def test_all_user_controlled_loop_multipliers_are_bounded():
    for name in ("n_placebo", "n_shuffle", "date_shift_max",
                 "truncation_sample_dates"):
        with pytest.raises(InputValidationError, match=name):
            AuditConfig(**{name: _MAX_DYNAMIC_RUNS + 1})
    with pytest.raises(InputValidationError, match="deferred_ic_max_horizon"):
        AuditConfig(deferred_ic_max_horizon=_MAX_FORWARD_HORIZON + 1)
    with pytest.raises(InputValidationError, match="cost_sensitivity_bps"):
        AuditConfig(cost_sensitivity_bps=tuple(
            range(_MAX_COST_SWEEP_POINTS + 1)))


def test_resource_boundaries_remain_legal_and_mutation_rolls_back():
    AuditConfig(n_placebo=_MAX_DYNAMIC_RUNS,
                deferred_ic_max_horizon=_MAX_FORWARD_HORIZON,
                cost_sensitivity_bps=tuple(range(_MAX_COST_SWEEP_POINTS)))
    cfg = AuditConfig(deferred_ic_max_horizon=6)
    with pytest.raises(InputValidationError):
        cfg.deferred_ic_max_horizon = _MAX_FORWARD_HORIZON + 1
    assert cfg.deferred_ic_max_horizon == 6


def test_markdown_escapes_raw_html_from_result_text():
    payload = ('<img src=x onerror="alert(1)"> | next\nline '
               '![track](https://attacker.invalid/pixel) '
               '[click](javascript:alert(1))')
    result = CheckResult(
        "dynamic.`probe`", Status.ERROR, payload,
        severity=Severity.HIGH, remediation=f"remove {payload}",
    )
    markdown = AuditReport([result]).to_markdown()
    assert "<img" not in markdown
    assert "onerror=\"" not in markdown
    assert r"&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;" in markdown
    assert "\\| next<br>line" in markdown
    assert "`dynamic.`probe``" not in markdown
    assert "![track](" not in markdown
    assert "[click](javascript:" not in markdown
    assert r"\!\[track\]\(https://attacker.invalid/pixel\)" in markdown
    assert r"\[click\]\(javascript:alert\(1\)\)" in markdown


def test_json_clean_preserves_series_and_dataframe_label_collisions():
    series = pd.Series(["integer", "string"], index=[1, "1"])
    cleaned_series = _json_clean(series)["__qaudit_typed_series_v1__"]
    assert [x["value"] for x in cleaned_series] == ["integer", "string"]

    frame = pd.DataFrame([[1, 2, 3]], columns=["a.2", "a", "a"])
    cleaned_frame = _json_clean(frame)["__qaudit_typed_columns_v1__"]
    assert [x["value"] for x in cleaned_frame] == [
        {"0": 1}, {"0": 2}, {"0": 3},
    ]


def test_json_clean_marks_cycles_instead_of_crashing_report_archive():
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    report = AuditReport([CheckResult(
        "thirdparty.cyclic", Status.PASS, "ok", details=cyclic,
    )])
    report.gate(require=["thirdparty.cyclic"])
    archived = report.to_dict()
    assert archived["results"][0]["details"] == {
        "self": {"__qaudit_cycle_v1__": "builtins.dict"}}


def test_json_clean_survives_broken_object_repr_in_values_and_keys():
    class ReprBomb:
        def __repr__(self):
            raise RuntimeError("repr exploded")

    bomb = ReprBomb()
    report = AuditReport([CheckResult(
        "thirdparty.repr", Status.PASS, "ok",
        details={"value": bomb, bomb: "key"},
    )])
    archived = report.to_dict()
    details = archived["results"][0]["details"]
    type_name = f"{ReprBomb.__module__}.{ReprBomb.__qualname__}"
    assert details["value"] == {"__qaudit_unrepresentable_v1__": type_name}
    assert details[f"<unrepresentable:{type_name}>"] == "key"


def test_raise_for_failures_normalizes_string_severity_like_gate():
    report = AuditReport([failed("family.medium", "bad",
                                 severity=Severity.MEDIUM)])
    report.raise_for_failures(min_severity="high")  # type: ignore[arg-type]
    with pytest.raises(AuditFailure):
        report.raise_for_failures(min_severity="medium")  # type: ignore[arg-type]


def test_gate_rejects_truthy_string_escape_flags_and_unresolved_partial_mix():
    basic = AuditReport([passed("family.only", "ok")])
    with pytest.raises(InputValidationError, match="allow_partial"):
        basic.gate(require=["family"], allow_partial="false")  # type: ignore[arg-type]
    with pytest.raises(InputValidationError, match="allow_unresolved"):
        basic.gate(require=["family"], allow_unresolved="false")  # type: ignore[arg-type]

    mixed = AuditReport([
        passed("family.judged", "ok"),
        passed("family.open", "needs review",
               details={"unresolved": True}),
        skipped("family.optional", "missing artifact"),
    ])
    with pytest.raises(AuditFailure, match="unresolved"):
        mixed.gate(require=["family"], allow_partial=True)
    mixed.gate(require=["family"], allow_partial=True,
               allow_unresolved=True)


def test_mutated_result_contract_cannot_fail_open_gate_or_ok(monkeypatch):
    complete = [passed(cid, "ok")
                for cid in api.MODULE_CHECK_IDS["qaudit.checks.costs"]]
    bad = failed(complete[0].check, "real failure")
    bad.status = "fail"  # type: ignore[assignment]
    complete[0] = bad
    monkeypatch.setattr("qaudit.checks.costs.run",
                        lambda *args, **kwargs: complete)
    report = audit(_dense_art(), include=["costs"])
    assert not report.ok
    assert report.errors and report.errors[0].check == "costs.<module>"

    direct = failed("costs.x", "real failure")
    direct.status = "fail"  # type: ignore[assignment]
    report = AuditReport([direct])
    assert not report.ok
    with pytest.raises(AuditFailure, match="structurally invalid"):
        report.gate(require=["costs"])
    with pytest.raises(AuditFailure, match="structurally invalid"):
        report.raise_for_failures()
