"""Audit filters, numeric configuration, report conversion, and null-probe boundaries.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from qaudit.api import audit
from qaudit.config import AuditConfig
from qaudit.errors import InputValidationError
from qaudit.inputs import BacktestArtifacts
from qaudit.report import AuditReport
from qaudit.synthetic import (momentum_signal, positions_from_signals,
                              simulate_market)
from qaudit.types import CheckResult, Severity, Status, errored


@pytest.fixture(scope="module")
def market():
    return simulate_market(n_assets=12, n_periods=300, seed=6, death_frac=0.0)


@pytest.fixture(scope="module")
def art(market):
    sig = momentum_signal(market["returns"])
    a = BacktestArtifacts(signals=sig, asset_returns=market["returns"],
                          positions=positions_from_signals(sig, 1),
                          signal_lag=1)
    a.validate()
    return a


# ===========================================================================
# 1. bare-string / non-str include/exclude filters raise (attack)
# ===========================================================================

@pytest.mark.parametrize("kw", [
    dict(exclude="survivorship"),   # chars would match performance.* -> ok=True
    dict(include="costs"),          # per-char substring would run all 8 modules
    dict(include=[3]),  # Invalid filter element type.
    dict(include=[None]),
    dict(exclude=["costs", 7]),
    dict(include={"costs"}),        # set: unordered, not the documented form
])
def test_string_or_nonstr_filters_raise_branded(art, kw):
    with pytest.raises(InputValidationError) as ei:
        audit(art, **kw)
    msg = str(ei.value)
    name = next(iter(kw))
    assert name in msg and "list/tuple" in msg
    assert "character by character" in msg  # says why a bare string is fatal


def test_list_filters_still_work_honest_guard(art):
    rep = audit(art, exclude=["survivorship"])
    # The list form drops only the named family; every other family still runs.
    assert any(r.check.startswith("performance") for r in rep.results)
    assert not any(r.check.startswith("survivorship") for r in rep.results)

    rep = audit(art, include=["costs"])
    assert rep.results
    assert all(r.check.startswith("costs") for r in rep.results)


def test_tuple_filters_accepted(art):
    rep = audit(art, include=("costs",))
    assert rep.results
    assert all(r.check.startswith("costs") for r in rep.results)


# ===========================================================================
# 2. AuditConfig rejects inf in every count-typed field (attack)
# ===========================================================================

_COUNTISH = ("min_periods", "fwd_vol_window", "deferred_ic_max_horizon",
             "ic_rolling_window", "min_assets_survivorship", "n_placebo",
             "n_shuffle", "date_shift_max", "truncation_sample_dates",
             "leak_perfect_min_names", "n_trials", "min_embargo_periods")


@pytest.mark.parametrize("name", _COUNTISH)
@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_config_rejects_inf_at_construction(name, bad):
    # Infinite configuration values must be rejected before consumers convert them to
    # integers.
    with pytest.raises(InputValidationError) as ei:
        AuditConfig(**{name: bad})
    assert name in str(ei.value)


def test_config_rejects_inf_on_mutation_and_keeps_last_valid_state():
    c = AuditConfig()
    before = c.n_trials
    with pytest.raises(InputValidationError):
        c.n_trials = float("inf")
    assert c.n_trials == before
    with pytest.raises(InputValidationError):
        c.ic_rolling_window = float("inf")
    assert c.ic_rolling_window == 63


def test_config_rejects_inf_in_atol_fields():
    # truncation_atol=inf satisfies ">= 0" yet would neutralize the
    # reproducibility check; it must be rejected, not tolerated
    for name in ("truncation_atol", "gross_match_atol"):
        with pytest.raises(InputValidationError):
            AuditConfig(**{name: float("inf")})


@pytest.mark.parametrize("name", ("min_periods", "n_trials", "sharpe_warn"))
def test_config_still_rejects_nan(name):
    # guard against a refactor swapping the comparisons for NaN-tolerant ones
    with pytest.raises(InputValidationError):
        AuditConfig(**{name: float("nan")})


def test_config_float_counts_still_accepted_honest_guard():
    # valid configs may use float counts (int()-truncated downstream);
    # rejecting inf must not tighten the domain to int-only
    c = AuditConfig(ic_rolling_window=63.0, n_trials=345.0)
    assert c.ic_rolling_window == 63.0
    assert c.n_trials == 345.0


# ===========================================================================
# 3. to_dict survives np.timedelta64 scalars and duplicated-column frames
# ===========================================================================

def _report_with(details: dict) -> AuditReport:
    return AuditReport(results=[
        CheckResult("x.y", Status.PASS, "m", severity=Severity.INFO,
                    details=details)])


@pytest.mark.parametrize("unit", ["D", "h", "s", "ms", "us", "ns"])
def test_to_dict_timedelta64_scalar_all_units(unit):
    # Timedelta scalars must preserve their duration across all supported units.
    rep = _report_with({"gap": np.timedelta64(5, unit)})
    d = rep.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["results"][0]["details"]["gap"] == str(np.timedelta64(5, unit))


def test_to_dict_timedelta64_in_meta():
    rep = AuditReport(results=[], meta={"gap": np.timedelta64(3, "D")})
    d = rep.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["meta"]["gap"] == "3 days"


def test_to_dict_duplicated_column_frame_no_recursion():
    # Duplicate DataFrame labels must not cause recursive serialization through label-
    # based selection.
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    rep = _report_with({"frame": df})
    d = rep.to_dict()
    json.dumps(d, allow_nan=False)
    cleaned = d["results"][0]["details"]["frame"]
    # both columns survive in an ordered, typed representation; no generated
    # suffix can overwrite a real user label.
    columns = cleaned["__qaudit_typed_columns_v1__"]
    assert [entry["position"] for entry in columns] == [0, 1]
    assert [entry["key"] for entry in columns] == ["a", "a"]
    assert [list(entry["value"].values()) for entry in columns] == [
        [1, 3], [2, 4],
    ]


def test_to_dict_temporal_and_int_branches_unchanged_honest_guard():
    idx = pd.bdate_range("2024-01-02", periods=3)
    rep = _report_with({
        "ts": pd.Timestamp("2024-01-02"),
        "dt64": np.datetime64("2024-01-02"),
        # Keep exercising the pandas Timedelta cleaner while avoiding the
        # deprecated generic NumPy timedelta unit used by keyword parsing.
        "td": pd.Timedelta(np.timedelta64(2, "D")),
        "n": np.int64(7),
        "frame": pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx),
    })
    d = rep.to_dict()["results"][0]["details"]
    json.dumps(d, allow_nan=False)
    assert d["ts"] == "2024-01-02T00:00:00"
    assert d["dt64"] == "2024-01-02"
    assert d["td"] == "2 days 00:00:00"
    assert d["n"] == 7
    assert list(d["frame"]["a"].values()) == [1.0, 2.0, 3.0]


# ===========================================================================
# 4. ERROR remediation renders in __str__ and to_markdown
# ===========================================================================

def test_filter_matched_nothing_remediation_visible_everywhere(art):
    rep = audit(art, include=["costsz"])
    r = rep["audit.filter_matched_nothing"]
    assert "valid family prefixes" in r.remediation
    # Both human-readable channels must include remediation for the HIGH error.
    assert "valid family prefixes" in str(rep)
    assert "valid family prefixes" in rep.to_markdown()
    assert "## Remediation" in rep.to_markdown()


def test_plain_errored_result_grows_no_fix_line():
    # errored() never sets remediation: exception-noise ERRORs stay terse
    r = errored("leakage.<module>", ValueError("boom"))
    assert "fix:" not in str(r)
    md = AuditReport(results=[r]).to_markdown()
    assert "## Remediation" not in md


def test_fail_warn_remediation_rendering_unchanged_honest_guard():
    f = CheckResult("a.b", Status.FAIL, "bad", severity=Severity.HIGH,
                    remediation="do X")
    w = CheckResult("a.c", Status.WARN, "meh", remediation="do Y")
    p = CheckResult("a.d", Status.PASS, "fine", remediation="ignored")
    assert "fix: do X" in str(f)
    assert "fix: do Y" in str(w)
    assert "fix:" not in str(p)          # PASS remediation still hidden
    md = AuditReport(results=[f, w, p]).to_markdown()
    assert "do X" in md and "do Y" in md and "ignored" not in md


# ===========================================================================
# 5. probes_null hedge-dust SKIP tells the truth
# ===========================================================================

@pytest.fixture(scope="module")
def basket_report():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=200)
    cols = [f"A{i}" for i in range(10)]
    sig = pd.DataFrame(rng.standard_normal((200, 10)), index=idx,
                       columns=cols)
    rets = pd.DataFrame(rng.standard_normal((200, 10)) * 0.012 + 0.0004,
                        index=idx, columns=cols)
    a = BacktestArtifacts(signals=sig, asset_returns=rets)

    def basket_func(signals, asset_returns):
        # ignores signals: every run's PnL is exactly linear in the basket
        return asset_returns.mean(axis=1)

    return audit(a, AuditConfig(n_placebo=5, n_shuffle=5),
                 backtest_func=basket_func, include=["dynamic"])


def test_hedge_dust_skip_message_truthful(basket_report):
    bias = basket_report["dynamic.placebo_pipeline_bias"]
    assert bias.status is Status.SKIP
    # Messages must describe hedge dust without blaming valid callback outputs.
    assert "returned NaN/degenerate" not in bias.message   # runs were valid
    assert "probe_health" not in bias.message              # dangling pointer
    # truthful diagnosis + actionable next step with concrete counts
    assert "hedge dust" in bias.message
    assert "5/5" in bias.message
    assert "consumes its signals argument" in bias.message
    assert bias.details["n_hedge_dust"] == 5
    assert bias.details["n_total"] == 5


def test_hedge_dust_scenario_probe_health_absent_detection_intact(
        basket_report):
    # no distribution was distrusted (raw runs all finite), so probe_health
    # must not appear; the percentile check used those very raw SRs
    assert all(r.check != "dynamic.probe_health"
               for r in basket_report.results)
    assert basket_report["dynamic.placebo_percentile"].status is not Status.SKIP
    # no fail-open: the signal-ignoring callable is still flagged
    assert basket_report["dynamic.shuffled_labels"].status is Status.FAIL


def test_truly_nan_runs_keep_degenerate_blame_honest_guard():
    # _degenerate_skip's message stays correct where it is true: a callable
    # returning NaN series makes the raw runs bad, blame stands, and
    # probe_health is emitted
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2023-01-02", periods=150)
    cols = [f"A{i}" for i in range(6)]
    sig = pd.DataFrame(rng.standard_normal((150, 6)), index=idx, columns=cols)
    rets = pd.DataFrame(rng.standard_normal((150, 6)) * 0.01,
                        index=idx, columns=cols)
    a = BacktestArtifacts(signals=sig, asset_returns=rets)

    def nan_func(signals, asset_returns):
        return pd.Series(np.nan, index=asset_returns.index)

    rep = audit(a, AuditConfig(n_placebo=5, n_shuffle=5),
                backtest_func=nan_func, include=["dynamic"])
    bias = rep["dynamic.placebo_pipeline_bias"]
    assert bias.status is Status.SKIP
    assert "returned NaN/degenerate" in bias.message
    assert any(r.check == "dynamic.probe_health" for r in rep.results)


# ===========================================================================
# 6. bool rejected for signal_lag / label_horizon
# ===========================================================================

def _mini_frames():
    idx = pd.bdate_range("2024-01-02", periods=150)
    cols = ["A", "B", "C"]
    rng = np.random.default_rng(2)
    sig = pd.DataFrame(rng.standard_normal((150, 3)), index=idx, columns=cols)
    rets = pd.DataFrame(rng.standard_normal((150, 3)) * 0.01,
                        index=idx, columns=cols)
    return sig, rets


@pytest.mark.parametrize("kw", [
    dict(signal_lag=False),    # YAML booleans must not masquerade as integers
    dict(signal_lag=True),
    dict(signal_lag=0),        # impossible under the fixed end-of-bar contract
    dict(label_horizon=True),
    dict(label_horizon=False),
])
def test_invalid_lag_horizon_rejected(kw):
    sig, rets = _mini_frames()
    a = BacktestArtifacts(signals=sig, asset_returns=rets, **kw)
    with pytest.raises(InputValidationError) as ei:
        a.validate()
    assert next(iter(kw)) in str(ei.value)


def test_int_and_np_int_lag_still_accepted_honest_guard():
    sig, rets = _mini_frames()
    for lag, hor in ((1, 2), (2, 1), (np.int64(1), np.int64(1))):
        a = BacktestArtifacts(signals=sig, asset_returns=rets,
                              signal_lag=lag, label_horizon=hor)
        a.validate()   # must not raise: honest declarations unchanged
