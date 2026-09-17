"""Core boundary tests: parameter bounds, branded errors, overflow handling."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import qaudit.api as api
from qaudit import (
    AuditConfig,
    AuditFailure,
    BacktestArtifacts,
    InputValidationError,
    Severity,
    audit,
)
from qaudit._stats import expected_max_sharpe, forward_returns
from qaudit.config import _MAX_COST_BPS, _MAX_ROLLING_WINDOW
from qaudit.inputs import _MAX_ABS_VALUE
from qaudit.inputs import _MAX_COST_BPS as _MAX_DECLARED_COST_BPS
from qaudit.inputs import _MAX_PERIOD_OFFSET
from qaudit.report import AuditReport, _json_clean
from qaudit.synthetic import (
    make_backtest_func,
    make_overfit,
    make_smeared_leak,
    momentum_signal,
    net_returns,
    positions_from_signals,
    simulate_market,
)
from qaudit.types import passed


@pytest.fixture
def panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2022-01-03", periods=250)
    columns = ["A", "B", "C"]
    rng = np.random.default_rng(91)
    signals = pd.DataFrame(rng.normal(size=(250, 3)), index=index, columns=columns)
    returns = pd.DataFrame(
        rng.normal(scale=0.01, size=(250, 3)), index=index, columns=columns
    )
    return signals, returns


def test_config_rejects_empty_cost_sweep_at_construction():
    with pytest.raises(InputValidationError, match="at least one sweep point"):
        AuditConfig(cost_sensitivity_bps=())


@pytest.mark.parametrize("field", ["fwd_vol_window", "ic_rolling_window"])
def test_config_bounds_pandas_rolling_windows(field):
    AuditConfig(**{field: _MAX_ROLLING_WINDOW})
    with pytest.raises(InputValidationError, match=field):
        AuditConfig(**{field: _MAX_ROLLING_WINDOW + 1})
    with pytest.raises(InputValidationError, match=field):
        AuditConfig(**{field: 10**1000})


def test_config_bounds_cost_values_and_float_conversion():
    AuditConfig(cost_sensitivity_bps=(_MAX_COST_BPS,))
    with pytest.raises(InputValidationError, match="cost_sensitivity_bps"):
        AuditConfig(cost_sensitivity_bps=(_MAX_COST_BPS + 1,))
    with pytest.raises(InputValidationError, match="cost_sensitivity_bps"):
        AuditConfig(cost_sensitivity_bps=(10**1000,))


@pytest.mark.parametrize("field", ["n_trials", "sharpe_warn"])
def test_config_rejects_arbitrary_precision_values_consumers_cannot_float(field):
    with pytest.raises(InputValidationError, match=field):
        AuditConfig(**{field: 10**1000})


@pytest.mark.parametrize("field", ["signal_lag", "label_horizon"])
def test_artifact_period_offsets_have_operational_ceiling(panels, field):
    signals, returns = panels
    BacktestArtifacts(signals, returns, **{field: _MAX_PERIOD_OFFSET}).validate()
    with pytest.raises(InputValidationError, match=field):
        BacktestArtifacts(
            signals, returns, **{field: _MAX_PERIOD_OFFSET + 1}
        ).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("periods_per_year", 10**1000),
        ("declared_costs_bps", 10**1000),
        ("declared_costs_bps", _MAX_DECLARED_COST_BPS + 1),
    ],
)
def test_artifact_float_scalars_fail_with_branded_error(panels, field, value):
    signals, returns = panels
    with pytest.raises(InputValidationError, match=field):
        BacktestArtifacts(signals, returns, **{field: value}).validate()


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"strict": 1}, "strict"),
        ({"strict": np.bool_(True)}, "strict"),
        ({"signal_func": 7}, "signal_func"),
        ({"backtest_func": "pipeline"}, "backtest_func"),
    ],
)
def test_audit_rejects_invalid_public_argument_types(panels, kwargs, field):
    signals, returns = panels
    with pytest.raises(InputValidationError, match=field):
        audit(BacktestArtifacts(signals, returns), include=["costs"], **kwargs)


def test_audit_rejects_non_artifact_object():
    with pytest.raises(InputValidationError, match="BacktestArtifacts"):
        audit(None)  # type: ignore[arg-type]


def test_audit_coverage_discloses_cross_sectional_breadth_starvation():
    index = pd.bdate_range("2024-01-01", periods=40)
    columns = [f"A{i:03d}" for i in range(100)]
    signals = pd.DataFrame(np.nan, index=index, columns=columns)
    signals.iloc[:, 0] = 1.0
    returns = pd.DataFrame(0.01, index=index, columns=columns)

    report = audit(BacktestArtifacts(signals, returns), include=["leakage"])

    coverage = report["audit.coverage"]
    assert report.ok  # high advisory; deployment policy chooses to block it
    assert coverage.status.value == "warn"
    assert coverage.details["max_jointly_finite_assets"] == 1
    assert coverage.details["dates_with_cross_sectional_support"] == 0
    assert report.meta["max_jointly_finite_assets"] == 1
    assert report.meta["median_jointly_finite_assets"] == 1.0
    with pytest.raises(AuditFailure, match="audit.coverage"):
        report.gate(warn_severity=Severity.HIGH)


def test_one_wide_date_cannot_hide_cross_sectional_date_starvation():
    index = pd.bdate_range("2024-01-01", periods=60)
    columns = [f"A{i:03d}" for i in range(10)]
    signals = pd.DataFrame(np.nan, index=index, columns=columns)
    signals.iloc[:, 0] = 1.0
    signals.iloc[0, :] = 1.0
    returns = pd.DataFrame(0.01, index=index, columns=columns)

    report = audit(BacktestArtifacts(signals, returns), include=["leakage"])

    coverage = report["audit.coverage"]
    assert coverage.status.value == "warn"
    assert coverage.severity is Severity.HIGH
    assert coverage.details["max_jointly_finite_assets"] == 10
    assert coverage.details["dates_with_cross_sectional_support"] == 1
    with pytest.raises(AuditFailure, match="audit.coverage"):
        report.gate(warn_severity=Severity.HIGH)


@pytest.mark.parametrize(
    "value,expected",
    [(np.array(5), 5), (np.array(np.nan), None), (np.array("xy"), "xy")],
)
def test_json_clean_handles_zero_dimensional_arrays(value, expected):
    assert _json_clean(value) == expected
    report = AuditReport([passed("extension.scalar", "ok", details={"value": value})])
    json.dumps(report.to_dict(), allow_nan=False)


def test_panel_digest_distinguishes_large_adjacent_integers():
    index = pd.date_range("2024-01-01", periods=1)
    exact_limit = pd.DataFrame([[2**53]], index=index, columns=["A"], dtype="int64")
    plus_one = pd.DataFrame([[2**53 + 1]], index=index, columns=["A"], dtype="int64")
    plus_two = pd.DataFrame([[2**53 + 2]], index=index, columns=["A"], dtype="int64")
    assert api._panel_digest(exact_limit) != api._panel_digest(plus_one)
    assert api._panel_digest(plus_one) != api._panel_digest(plus_two)


def test_object_boolean_universe_with_missing_cell_is_accepted(panels):
    signals, returns = panels
    universe = pd.DataFrame(True, index=signals.index, columns=signals.columns).astype(
        object
    )
    universe.iloc[10, 0] = pd.NA  # 1/750 interior cells: below ambiguity gate
    artifacts = BacktestArtifacts(signals, returns, universe=universe)
    artifacts.validate()
    aligned = artifacts.aligned()
    assert aligned.universe.dtypes.eq(bool).all()
    assert not bool(aligned.universe.iloc[10, 0])


def test_object_universe_still_rejects_string_codes(panels):
    signals, returns = panels
    universe = pd.DataFrame(True, index=signals.index, columns=signals.columns).astype(
        object
    )
    universe.iloc[10, 0] = "yes"
    with pytest.raises(InputValidationError, match="universe.*non-numeric"):
        BacktestArtifacts(signals, returns, universe=universe).validate()


def test_mixed_timezone_period_bounds_raise_branded_error(panels):
    signals, returns = panels
    period = (signals.index[0], signals.index[20].tz_localize("UTC"))
    with pytest.raises(InputValidationError, match="timezone footing"):
        BacktestArtifacts(signals, returns, train_period=period).validate()


def test_provenance_records_and_hashes_scalar_artifact_contract(panels):
    signals, returns = panels
    artifacts = BacktestArtifacts(
        signals,
        returns,
        train_period=(signals.index[0], signals.index[99]),
        test_period=(signals.index[110], signals.index[-1]),
        signal_lag=2,
        label_horizon=5,
        periods_per_year=252.0,
        declared_costs_bps=7.5,
    )
    report = audit(artifacts, include=["costs"])
    params = report.meta["provenance"]["artifact_parameters"]
    assert params == {
        "signal_lag": 2,
        "label_horizon": 5,
        "periods_per_year": 252.0,
        "declared_costs_bps": 7.5,
        "train_period": [bound.isoformat() for bound in artifacts.train_period],
        "test_period": [bound.isoformat() for bound in artifacts.test_period],
    }
    report.gate()
    params["label_horizon"] = 1
    with pytest.raises(AuditFailure, match="artifact_parameters_hash"):
        report.gate()


def test_provenance_large_integers_fail_as_audit_errors_not_overflow(panels):
    signals, returns = panels
    report = audit(BacktestArtifacts(signals, returns), include=["costs"])
    provenance = report.meta["provenance"]
    params = provenance["artifact_parameters"]
    params["periods_per_year"] = 10**1000
    payload = json.dumps(
        params, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    provenance["artifact_parameters_hash"] = hashlib.sha256(payload).hexdigest()

    with pytest.raises(AuditFailure, match="periods_per_year.*malformed"):
        report.gate()

    report = audit(BacktestArtifacts(signals, returns), include=["costs"])
    provenance = report.meta["provenance"]
    module = next(iter(provenance["timings_s"]))
    provenance["timings_s"][module] = 10**1000
    with pytest.raises(AuditFailure, match="timings_s.*inconsistent"):
        report.gate()


def test_expected_max_sharpe_is_finite_for_large_trial_counts():
    ordinary = expected_max_sharpe(10**10, 0.001)
    enormous = expected_max_sharpe(10**1000, 0.001)
    assert np.isfinite(ordinary)
    assert np.isfinite(enormous)
    assert enormous > ordinary > 0


@pytest.mark.parametrize("horizon", [3, 10**1000])
def test_forward_returns_oversized_horizon_is_exact_all_missing(horizon):
    index = pd.date_range("2024-01-01", periods=3)
    returns = pd.DataFrame(
        [[0.1, -1.0], [0.2, 0.4], [-0.3, np.nan]],
        index=index,
        columns=["A", "B"],
    )
    actual = forward_returns(returns, horizon)
    assert actual.index.equals(returns.index)
    assert actual.columns.equals(returns.columns)
    assert actual.shape == returns.shape
    assert actual.dtypes.eq(float).all()
    assert actual.isna().all().all()


@pytest.mark.parametrize("horizon", [0, -1, 1.0, True, np.bool_(True), "2", None])
def test_forward_returns_rejects_nonpositive_or_noninteger_horizon(horizon):
    returns = pd.DataFrame({"A": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="positive integer"):
        forward_returns(returns, horizon)  # type: ignore[arg-type]


def test_forward_returns_accepts_numpy_integer_horizon():
    returns = pd.DataFrame({"A": [0.1, 0.2, 0.3]})
    expected = pd.DataFrame({"A": [0.56, np.nan, np.nan]})
    pd.testing.assert_frame_equal(forward_returns(returns, np.int64(2)), expected)


def test_forward_returns_fails_closed_before_compound_overflow():
    returns = pd.DataFrame(
        np.full((5, 2), _MAX_ABS_VALUE / 10.0),
        index=pd.date_range("2024-01-01", periods=5),
        columns=["A", "B"],
    )
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="compounding exceeds float64"):
            forward_returns(returns, 4)


def test_audit_records_compound_overflow_as_blocking_error():
    index = pd.bdate_range("2024-01-01", periods=40)
    columns = ["A", "B", "C", "D", "E"]
    signals = pd.DataFrame(
        np.tile(np.arange(5.0), (40, 1)), index=index, columns=columns
    )
    returns = pd.DataFrame(_MAX_ABS_VALUE / 10.0, index=index, columns=columns)
    with np.errstate(all="raise"):
        report = audit(
            BacktestArtifacts(signals, returns, label_horizon=4),
            include=["leakage"],
        )
    assert not report.ok
    assert report.errors
    assert "compounding exceeds float64" in report.errors[0].message


def test_artifact_magnitude_ceiling_leaves_arithmetic_headroom(panels):
    signals, returns = panels
    corrupted = signals.copy()
    corrupted.iloc[10, 0] = _MAX_ABS_VALUE
    with pytest.raises(InputValidationError, match="magnitude"):
        BacktestArtifacts(corrupted, returns).validate()


@pytest.mark.parametrize(
    "bound",
    ["2024-11-03 01:30", "2024-03-10 02:30"],
)
def test_dst_ambiguous_period_bound_raises_branded_error(bound):
    index = pd.date_range("2024-01-01", periods=400, freq="D", tz="America/New_York")
    panel = pd.DataFrame(0.01, index=index, columns=["A", "B"])
    with pytest.raises(InputValidationError, match="explicit timezone-aware"):
        BacktestArtifacts(
            panel,
            panel,
            train_period=(bound, "2024-11-10"),
            periods_per_year=365.0,
        ).validate()


def test_multi_century_datetime_span_does_not_overflow_validation():
    raw = np.linspace(
        int(pd.Timestamp.min.value) + 10_000,
        int(pd.Timestamp.max.value) - 10_000,
        30,
    ).astype("int64")
    index = pd.DatetimeIndex(pd.to_datetime(raw))
    panel = pd.DataFrame(0.01, index=index, columns=["A"])
    # Roughly 0.05 rows/year across the ~584-year supported ns range.
    BacktestArtifacts(panel, panel, periods_per_year=0.05).validate()


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"n_assets": 0}, "n_assets"),
        ({"n_periods": 0}, "n_periods"),
        ({"seed": True}, "seed"),
        ({"seed": -1}, "seed"),
        ({"death_frac": -0.1}, "death_frac"),
        ({"death_frac": 1.1}, "death_frac"),
        ({"death_frac": np.nan}, "death_frac"),
        ({"mom_loading": np.inf}, "mom_loading"),
        ({"mom_loading": 1.000001}, "mom_loading"),
        ({"mom_loading": -1.000001}, "mom_loading"),
        ({"mom_loading": 1e10}, "mom_loading"),
        ({"mom_loading": 1e308}, "mom_loading"),
        ({"n_assets": 10_000, "n_periods": 201}, "cells"),
    ],
)
def test_simulate_market_rejects_invalid_or_unsafe_parameters(kwargs, field):
    with pytest.raises(ValueError, match=field):
        simulate_market(**kwargs)


def test_simulate_market_supports_tiny_positive_panel_without_rng_error():
    market = simulate_market(n_assets=2, n_periods=1, death_frac=1.0)
    assert market["returns"].shape == (1, 2)
    assert market["universe"].shape == (1, 2)


@pytest.mark.parametrize("mom_loading", [-1.0, 1.0])
def test_simulate_market_coefficient_boundaries_have_finite_live_output(
    mom_loading,
):
    market = simulate_market(
        n_assets=8, n_periods=200, mom_loading=mom_loading, death_frac=0.5
    )
    live = market["universe"].to_numpy(dtype=bool)
    assert np.isfinite(market["returns"].to_numpy(dtype=float)[live]).all()
    assert np.isfinite(market["prices"].to_numpy(dtype=float)[live]).all()


@pytest.mark.parametrize("window", [0, -1, 1.5, True, 60_001, 10**1000])
def test_momentum_signal_rejects_invalid_window(window):
    panel = pd.DataFrame({"A": [0.1, 0.2]})
    with pytest.raises(ValueError, match="window must be an integer"):
        momentum_signal(panel, window=window)  # type: ignore[arg-type]


@pytest.mark.parametrize("lag", [-1, 1.5, True, 60_001, 10**1000])
def test_positions_and_backtest_factory_reject_invalid_lag(lag):
    signals = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]})
    with pytest.raises(ValueError, match="lag must be an integer"):
        positions_from_signals(signals, lag=lag)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lag must be an integer"):
        make_backtest_func(lag=lag)  # type: ignore[arg-type]


def test_positions_and_backtest_factory_preserve_same_bar_lag_zero():
    signals = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]})
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, -0.02]})
    direct = positions_from_signals(signals, lag=0)
    via_factory = make_backtest_func(lag=0)(signals, returns)
    assert direct.abs().sum(axis=1).eq(1.0).all()
    assert np.isfinite(via_factory).all()


@pytest.mark.parametrize(
    "costs_bps", [np.nan, np.inf, True, "10", 1_000_001, -1_000_001, 10**1000]
)
def test_net_returns_and_backtest_factory_reject_unsafe_costs(costs_bps):
    signals = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]})
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, -0.02]})
    positions = positions_from_signals(signals)
    with pytest.raises(ValueError, match="costs_bps"):
        net_returns(positions, returns, costs_bps=costs_bps)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="costs_bps"):
        make_backtest_func(costs_bps=costs_bps)  # type: ignore[arg-type]


def test_synthetic_cost_helper_preserves_negative_stress_cases():
    signals = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]})
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, -0.02]})
    positions = positions_from_signals(signals)
    assert np.isfinite(net_returns(positions, returns, costs_bps=-5.0)).all()


@pytest.mark.parametrize("n_candidates", [0, -1, 1.5, True, 1_001])
def test_make_overfit_rejects_invalid_candidate_count_before_simulation(
    n_candidates,
):
    with pytest.raises(ValueError, match="n_candidates"):
        make_overfit(n_candidates=n_candidates)  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [np.nan, np.inf, 1.000001, -1.000001, 1e10, 1e308])
def test_make_smeared_leak_rejects_unsafe_alpha_before_simulation(alpha):
    with pytest.raises(ValueError, match="alpha"):
        make_smeared_leak(alpha=alpha)


def test_make_smeared_leak_boundary_has_no_infinite_outputs():
    case = make_smeared_leak(alpha=1.0)
    assert not np.isinf(case.artifacts.signals.to_numpy(dtype=float)).any()
    assert np.isfinite(case.artifacts.strategy_returns).all()
