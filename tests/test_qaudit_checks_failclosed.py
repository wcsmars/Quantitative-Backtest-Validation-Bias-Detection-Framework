"""Regression tests for checker-level false passes and overflow defenses."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit._stats import (annualized_sharpe, gross_strategy_returns,
                           traded_dollars_series)
from qaudit.checks import costs, leakage, lookahead, performance, survivorship
from qaudit.config import AuditConfig
from qaudit.dynamic import probes_shift
from qaudit.inputs import BacktestArtifacts
from qaudit.types import Severity, Status


def _random_panel(n: int = 100, m: int = 10, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    assets = [f"A{i:02d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, m)),
                        index=dates, columns=assets)
    signals = pd.DataFrame(rng.normal(size=(n, m)),
                           index=dates, columns=assets)
    return rng, rets, signals


def _by_id(results):
    return {result.check: result for result in results}


def _wiped_restarted_book(n: int = 1000) -> BacktestArtifacts:
    """A frequently traded book with a -100% portfolio bar and restart."""
    rng, rets, signals = _random_panel(n=n, m=5, seed=31)
    rets.iloc[:] = 0.0
    rets.iloc[10, 0] = -0.5
    positions = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    positions.iloc[:, 0] = np.where(np.arange(n) % 2 == 0, 2.0, -2.0)
    gross = gross_strategy_returns(positions, rets)
    dollars = traded_dollars_series(positions, rets)
    assert int(dollars.isna().sum()) == 1
    # This is the adversarial treatment: make the undefined restart
    # free, while charging a perfectly reconcilable 50bps everywhere else.
    net = gross - dollars.fillna(0.0) * 50.0e-4
    art = BacktestArtifacts(
        signals=signals,
        asset_returns=rets,
        positions=positions,
        strategy_returns=net,
        declared_costs_bps=50.0,
    )
    art.validate()
    return art.aligned()


def test_fractional_periods_per_year_is_not_truncated_to_zero():
    rng = np.random.default_rng(9)
    dates = pd.date_range("1900-01-01", periods=40, freq="2YS")
    assets = [f"A{i}" for i in range(8)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (40, 8)), dates, assets)
    signals = pd.DataFrame(rng.normal(size=(40, 8)), dates, assets)
    strategy = pd.Series(np.tile([0.010, 0.009], 20), index=dates)
    art = BacktestArtifacts(
        signals=signals,
        asset_returns=rets,
        strategy_returns=strategy,
        periods_per_year=0.5,
    )
    art.validate()

    result = _by_id(performance.run(art.aligned(), AuditConfig(n_trials=2)))
    expected = annualized_sharpe(strategy, periods_per_year=0.5)
    suspicious = result["performance.suspicious_sharpe"]
    deflated = result["performance.deflated_sharpe"]

    assert expected > AuditConfig().sharpe_fail
    assert suspicious.status is Status.FAIL
    assert suspicious.details["sharpe_annualized"] == pytest.approx(expected)
    assert suspicious.details["periods_per_year"] == pytest.approx(0.5)
    assert deflated.details["sr_annualized"] == pytest.approx(expected)


def test_undefined_post_wipeout_ledger_cannot_green_cost_family():
    result = _by_id(costs.run(_wiped_restarted_book(), AuditConfig()))

    missing = result["costs.missing_transaction_costs"]
    assert missing.status is Status.FAIL
    assert missing.severity is Severity.CRITICAL
    assert missing.details["n_nonfinite_trade_rows"] == 1
    assert missing.details["trade_ledger_undefined"] is True
    assert result["costs.turnover_unrealistic"].status is Status.SKIP
    assert result["costs.cost_sensitivity"].status is Status.SKIP


def test_undefined_ledger_cannot_exonerate_extreme_negative_sharpe():
    art = _wiped_restarted_book()
    result = performance._suspicious_sharpe(
        art.strategy_returns, "strategy_returns", art, AuditConfig())

    assert annualized_sharpe(art.strategy_returns, 252) < -10.0
    assert result.status is Status.FAIL
    assert result.severity is Severity.HIGH
    assert "cost drag alone does not explain" in result.message


def test_partial_degeneracy_cannot_hide_intermittent_smeared_signal():
    _, rets, signals = _random_panel(n=300, m=10, seed=123)
    signals = rets.copy()
    # Most dates are exactly explained by the same-bar-return control, but a
    # 30-date island carries fresh returns integrated over future horizons.
    # An h=1-only early return would discard that island and certify PASS;
    # the check must instead decline to judge.
    for t in range(100, 130):
        signals.iloc[t] = sum(rets.shift(-h).iloc[t] for h in range(1, 7))
    art = BacktestArtifacts(signals=signals, asset_returns=rets)
    art.validate()

    result = lookahead._smeared_forward_ic(art.aligned(), AuditConfig())
    assert result.status is Status.SKIP
    assert "too few usable innovation-IC dates" in result.message


def test_horizon_profiles_require_common_calendar_dates():
    rng = np.random.default_rng(133)
    dates = pd.bdate_range("2020-01-02", periods=400)
    assets = [f"A{i:02d}" for i in range(10)]
    rets = pd.DataFrame(np.nan, index=dates, columns=assets)
    signals = pd.DataFrame(0.0, index=dates, columns=assets)

    # Forty h=1 IC dates and forty disjoint h=2 IC dates.  Each marginal
    # sample clears the floor, but no date supports a decay/spike comparison.
    for t in range(10, 170, 4):
        score = rng.normal(size=len(assets))
        signals.iloc[t] = score
        rets.iloc[t + 1] = 0.01 * score          # IC1 = +1
    for j, t in enumerate(range(210, 370, 4)):
        score = rng.normal(size=len(assets))
        signals.iloc[t] = score
        rets.iloc[t + 2] = (0.01 * score if j % 2 == 0
                            else -0.01 * score)   # mean IC2 = 0

    art = BacktestArtifacts(signals=signals, asset_returns=rets)
    art.validate()
    art = art.aligned()
    decay = lookahead._ic_decay_signature(art, AuditConfig())
    deferred = lookahead._deferred_ic_spike(
        art, AuditConfig(deferred_ic_max_horizon=2))

    assert decay.status is Status.SKIP
    assert decay.details["n_dates_ic1"] == 40
    assert decay.details["n_dates_ic2"] == 40
    assert decay.details["n_common_dates"] == 0
    assert deferred.status is Status.SKIP
    assert deferred.details["n_common_dates"] == 0


def test_unvalidated_huge_label_horizon_does_not_crash_check_modules():
    _, rets, signals = _random_panel(n=80, m=10, seed=45)
    # Public audit validation rejects this value.  Direct checker calls still
    # fail closed instead of leaking pandas' C-long OverflowError.
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            label_horizon=10**100)

    perf = _by_id(performance.run(art, AuditConfig()))
    leak = leakage.run(art, AuditConfig())
    ahead = _by_id(lookahead.run(art, AuditConfig()))

    assert perf["performance.suspicious_ic"].status is Status.SKIP
    assert all(result.status is Status.SKIP for result in leak)
    embedded = ahead["lookahead.embedded_future_return"]
    assert embedded.details["n_dates"][str(10**100)] == 0


def test_unvalidated_huge_signal_lag_short_circuits_alignment():
    rng, rets, signals = _random_panel(n=80, m=10, seed=55)
    positions = pd.DataFrame(rng.normal(size=rets.shape),
                             index=rets.index, columns=rets.columns)
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            positions=positions, signal_lag=10**100)

    result = lookahead._position_signal_alignment(art, AuditConfig())
    assert result.status is Status.SKIP
    assert result.details["n_overlap_rows"] == len(rets)


def test_checker_guards_survive_config_validation_bypass():
    rng, rets, signals = _random_panel(n=100, m=20, seed=65)
    positions = pd.DataFrame(rng.normal(size=rets.shape),
                             index=rets.index, columns=rets.columns)
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            positions=positions)

    vol_cfg = AuditConfig()
    object.__setattr__(vol_cfg, "fwd_vol_window", 10**100)
    vol = lookahead._future_vol_sizing(art, vol_cfg)
    assert vol.status is Status.SKIP
    assert vol.details["n_candidates"] == 0

    ic_cfg = AuditConfig()
    object.__setattr__(ic_cfg, "ic_rolling_window", 10**100)
    ic = pd.Series(rng.normal(size=len(rets)), index=rets.index)
    stability = performance._ic_stability(ic, art, ic_cfg)
    assert stability.status is Status.PASS
    assert stability.details["not_judged"] is True
    assert stability.details["n_windows"] == 0
    # leakage.ic_outlier_dates has a separate centered-rolling consumer.
    leakage.run(art, ic_cfg)

    cost_cfg = AuditConfig()
    object.__setattr__(cost_cfg, "cost_sensitivity_bps", (10**1000,))
    sensitivity = costs._check_cost_sensitivity(art, cost_cfg)
    assert sensitivity.status is Status.SKIP
    assert sensitivity.details["invalid_cost_sensitivity_bps"] is True

    object.__setattr__(cost_cfg, "cost_sensitivity_bps", 10)
    scalar = costs._check_cost_sensitivity(art, cost_cfg)
    assert scalar.status is Status.SKIP
    assert scalar.details["invalid_cost_sensitivity_bps"] is True


def test_nonfinite_derived_cost_drag_cannot_pass_as_negligible():
    _, rets, signals = _random_panel(n=100, m=5, seed=81)
    rets.iloc[:] = 0.0
    rets.iloc[:, 0] = 1e308
    positions = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    positions.iloc[:, 0] = 1.0
    net = pd.Series(-1e308, index=rets.index)
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            positions=positions, strategy_returns=net)
    # Public validation rejects these magnitudes; direct checker consumers
    # still must not convert a derived infinity into a green verdict.

    result = costs._check_missing_transaction_costs(art, AuditConfig())
    assert result.status is Status.FAIL
    assert result.severity is Severity.CRITICAL
    assert result.details["nonfinite_cost_drag"] is True


def test_nonfinite_implied_cost_cannot_fall_through_to_pass():
    _, rets, signals = _random_panel(n=100, m=5, seed=82)
    rets.iloc[:] = 0.0
    positions = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    positions.iloc[::2, 0] = 1.0
    rets.iloc[::2, 0] = 1e308
    net = pd.Series(0.0, index=rets.index)
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            positions=positions, strategy_returns=net)
    # Public validation rejects these magnitudes; exercise defense in depth.

    result = costs._check_missing_transaction_costs(art, AuditConfig())
    assert result.status is Status.FAIL
    assert result.severity is Severity.CRITICAL
    assert not np.isfinite(result.details["implied_bps"])


def test_price_consistency_accepts_unique_heterogeneous_asset_labels():
    """Flat asset labels need not be mutually orderable or string-like."""
    rng = np.random.default_rng(83)
    index = pd.bdate_range("2020-01-02", periods=100)
    columns = pd.Index(
        [None, np.nan, pd.Timestamp("2000-01-01"), frozenset({"X"}), ("A", 1)],
        dtype=object,
        tupleize_cols=False,
    )
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, (len(index), len(columns))),
        index=index,
        columns=columns,
    )
    signals = pd.DataFrame(
        rng.normal(size=returns.shape), index=index, columns=columns
    )
    prices = 100.0 * (1.0 + returns).cumprod()
    art = BacktestArtifacts(signals, returns, prices=prices)
    art.validate()

    result = costs._check_price_return_consistency(art.aligned())

    assert result.status is Status.PASS
    assert result.details["n_assets_total"] == len(columns)
    assert result.details["n_assets_compared"] == len(columns)


def test_truncation_accepts_dynamic_heterogeneous_asset_labels():
    """A causal callback may omit a late-start asset until it has data."""
    rng = np.random.default_rng(84)
    index = pd.bdate_range("2020-01-02", periods=120)
    columns = pd.Index(
        [pd.Timestamp("2000-01-01"), frozenset({"X"}), ("A", 1), None, np.nan],
        dtype=object,
        tupleize_cols=False,
    )
    signal_input = pd.DataFrame(
        rng.normal(size=(len(index), len(columns))), index=index, columns=columns
    )
    signal_input.iloc[:90, -1] = np.nan
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, signal_input.shape), index=index, columns=columns
    )

    def causal_dynamic_columns(frame):
        output = frame.rolling(2).mean()
        return output.loc[:, output.notna().any(axis=0)]

    signals = causal_dynamic_columns(signal_input)
    art = BacktestArtifacts(
        signals, returns, signal_input=signal_input
    )
    art.validate()

    _, truncation, _ = probes_shift._signal_func_probes(
        art.aligned(),
        AuditConfig(truncation_sample_dates=len(index)),
        causal_dynamic_columns,
    )

    assert truncation.status is Status.PASS
    assert truncation.details["n_mismatched"] == 0


def test_all_nan_asset_padding_cannot_launder_full_history_panel():
    rng, rets, signals = _random_panel(n=600, m=100, seed=91)
    rets.iloc[:, 10:] = np.nan
    art = BacktestArtifacts(signals=signals, asset_returns=rets)
    art.validate()

    result = survivorship._full_history_universe(art.aligned(), AuditConfig())
    assert result.status is Status.WARN
    assert result.severity is Severity.HIGH
    assert result.details["n_assets"] == 10
    assert result.details["n_grid_assets"] == 100
    assert result.details["n_unobserved_assets"] == 90
    assert result.details["frac_full_history"] == pytest.approx(1.0)


def test_all_false_universe_padding_cannot_fake_breadth_and_attrition():
    _, rets, signals = _random_panel(n=800, m=20, seed=92)
    universe = pd.DataFrame(False, index=rets.index, columns=rets.columns)
    universe.iloc[:400, -1] = True
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            universe=universe)
    art.validate()

    result = survivorship._no_exits(art.aligned(), AuditConfig())
    assert result.status is Status.SKIP
    assert result.details["n_assets"] == 1
    assert result.details["n_grid_assets"] == 20
    assert result.details["n_never_live_assets"] == 19

    # Never-live columns may appear before live columns: the exit scan must
    # retain grid coordinates while using only ever-live names as its rate
    # denominator.
    universe.iloc[:400, -10:] = True
    broad = BacktestArtifacts(signals=signals, asset_returns=rets,
                              universe=universe)
    broad.validate()
    supported = survivorship._no_exits(broad.aligned(), AuditConfig())
    assert supported.status is Status.PASS
    assert supported.details["n_assets"] == 10
    assert supported.details["n_exiting_assets"] == 10
