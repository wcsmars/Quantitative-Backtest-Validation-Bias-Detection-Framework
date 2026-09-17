"""Tests for qaudit.checks.survivorship."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from qaudit.checks.survivorship import (
    MISSING_RETURN_FRAC,
    run,
)
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import make_clean, make_survivorship
from qaudit.types import Severity, Status

CHECK_IDS = {
    "survivorship.trading_outside_universe",
    "survivorship.positions_on_missing_returns",
    "survivorship.no_exits",
    "survivorship.full_history_universe",
}


def _by_id(results):
    out = {r.check: r for r in results}
    assert set(out) == CHECK_IDS, "module must emit every check id exactly once"
    return out


# ---------------------------------------------------------------------------
# Synthetic cases: built once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_results():
    case = make_clean()
    return _by_id(run(case.artifacts.aligned(), AuditConfig()))


@pytest.fixture(scope="module")
def surv_results():
    case = make_survivorship()
    return _by_id(run(case.artifacts.aligned(), AuditConfig()))


# ---------------------------------------------------------------------------
# Hand-built minimal frames
# ---------------------------------------------------------------------------

def _base_frames(n_periods=300, n_assets=12, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_periods)
    assets = [f"A{i:02d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n_periods, n_assets)),
                        index=dates, columns=assets)
    signals = rets.rolling(5, min_periods=5).mean()
    universe = pd.DataFrame(True, index=dates, columns=assets)
    positions = pd.DataFrame(1.0 / n_assets, index=dates, columns=assets)
    return dates, assets, rets, signals, universe, positions


def _aligned(**kwargs) -> BacktestArtifacts:
    art = BacktestArtifacts(**kwargs)
    art.validate()
    return art.aligned()


# ---------------------------------------------------------------------------
# 1. trading_outside_universe
# ---------------------------------------------------------------------------

def test_trading_outside_universe_fails_on_hindsight_universe():
    dates, _, rets, signals, universe, positions = _base_frames()
    rets.iloc[:] = 0.0  # constant weights are a true hold only without drift
    universe.iloc[100:, 0] = False   # A00 leaves the universe at bar 100
    # ...but positions keep holding A00 for the remaining 200 bars - far past
    # the 21-bar rebalance-latency grace, so it counts as a zombie holding.
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=universe, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))["survivorship.trading_outside_universe"]

    assert res.status == Status.FAIL
    assert res.severity == Severity.CRITICAL
    # decision time = t-1: out-of-universe from bar 101, grace expires after
    # HOLDOVER_GRACE_BARS=21 bars -> first flagged bar is 101+21=122
    assert res.details["n_violations"] == 300 - 122
    assert res.details["first_asset"] == "A00"
    assert res.details["first_date"] == str(dates[122].date())
    assert res.details["n_assets_affected"] == 1
    assert res.details["n_holdover_cells"] == 21
    assert "A00" in res.message
    assert res.remediation  # must say what to change


def test_trading_outside_universe_entry_fails_immediately():
    # a position entered while out-of-universe gets no grace at all
    dates, _, rets, signals, universe, positions = _base_frames()
    universe.iloc[100:, 0] = False
    positions.iloc[:150, 0] = 0.0     # flat until bar 150...
    positions.iloc[150:, 0] = 0.05    # ...then a fresh entry, 49 bars after exit
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=universe, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))["survivorship.trading_outside_universe"]
    assert res.status == Status.FAIL
    assert res.details["first_date"] == str(dates[150].date())


def test_trading_outside_universe_first_lag_rows_are_legal():
    # asset out of the universe from bar 0: with signal_lag=1 the shift is NaN
    # on row 0 (treated legal); grace expires 21 bars later, so a constant
    # holding is flagged from bar 1+21=22 on.
    dates, _, rets, signals, universe, positions = _base_frames()
    rets.iloc[:] = 0.0  # isolate the lag/grace path from real rebalancing
    universe.iloc[:, 1] = False
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=universe, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))["survivorship.trading_outside_universe"]
    assert res.status == Status.FAIL
    assert res.details["first_date"] == str(dates[22].date())
    assert res.details["n_violations"] == 300 - 22


def test_trading_outside_universe_holdover_within_grace_passes():
    # weekly-rebalance reality: an exiting name is held a few bars until the
    # next rebalance zeroes it - normal mechanics, must not fail.
    dates, _, rets, signals, universe, positions = _base_frames()
    rets.iloc[:] = 0.0  # unchanged weights equal passive drift in this fixture
    universe.iloc[100:, 0] = False
    positions.iloc[104:, 0] = 0.0     # unwound 4 bars after the exit
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=universe, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))["survivorship.trading_outside_universe"]
    assert res.status == Status.PASS
    assert res.details["n_holdover_cells"] == 3
    assert re.search(r"hold/unwind", res.message)


def test_trading_outside_universe_passes_on_clean(clean_results):
    res = clean_results["survivorship.trading_outside_universe"]
    assert res.status == Status.PASS
    assert res.details["n_violations"] == 0
    # PASS message still states what was measured
    assert re.search(r"\d+ nonzero cells", res.message)


def test_trading_outside_universe_skips_without_universe(surv_results):
    res = surv_results["survivorship.trading_outside_universe"]
    assert res.status == Status.SKIP
    assert "artifacts.universe" in res.message


def test_trading_outside_universe_skips_without_positions():
    _, _, rets, signals, universe, _ = _base_frames()
    arts = _aligned(signals=signals, asset_returns=rets, positions=None,
                    universe=universe, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))["survivorship.trading_outside_universe"]
    assert res.status == Status.SKIP
    assert "artifacts.positions" in res.message


# ---------------------------------------------------------------------------
# 2. positions_on_missing_returns
# ---------------------------------------------------------------------------

def test_positions_on_missing_returns_warns_on_nan_holes():
    dates, _, rets, signals, universe, positions = _base_frames()
    rets.iloc[200:250, 1] = np.nan   # 50 held bars ride through NaN returns
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=universe, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))[
        "survivorship.positions_on_missing_returns"]

    assert res.status == Status.WARN
    assert res.severity == Severity.HIGH
    assert res.details["n_violations"] == 50
    frac = res.details["frac_missing_return"]
    assert frac == pytest.approx(50 / (300 * 12))
    assert frac > MISSING_RETURN_FRAC
    assert re.search(r"\b50\b", res.message)
    assert res.details["first_asset"] == "A01"
    assert res.details["first_date"] == str(dates[200].date())
    assert res.remediation


def test_positions_on_missing_returns_tolerates_clean_delisting_bars(clean_results):
    # clean case: 4 dying assets, each holds exactly one final NaN-return bar
    res = clean_results["survivorship.positions_on_missing_returns"]
    assert res.status == Status.PASS
    assert res.details["n_violations"] == 4
    assert res.details["frac_missing_return"] < MISSING_RETURN_FRAC
    # PASS message mentions the tolerated count
    assert re.search(r"\b4\b", res.message)


def test_positions_on_missing_returns_skips_without_positions():
    _, _, rets, signals, universe, _ = _base_frames()
    arts = _aligned(signals=signals, asset_returns=rets, positions=None,
                    universe=universe, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))[
        "survivorship.positions_on_missing_returns"]
    assert res.status == Status.SKIP
    assert "artifacts.positions" in res.message


# ---------------------------------------------------------------------------
# 3. no_exits
# ---------------------------------------------------------------------------

def test_no_exits_warns_on_static_universe():
    _, _, rets, signals, universe, positions = _base_frames()  # all-True universe
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=universe, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))["survivorship.no_exits"]
    assert res.status == Status.WARN
    assert res.severity == Severity.MEDIUM
    assert res.details["n_exiting_assets"] == 0
    assert re.search(r"no asset ever leaves", res.message)
    assert re.search(r"\b12 names\b", res.message)
    assert res.remediation


def test_no_exits_passes_on_clean(clean_results):
    # 15% of 30 assets die mid-sample -> 4 exits
    res = clean_results["survivorship.no_exits"]
    assert res.status == Status.PASS
    assert res.details["n_exiting_assets"] == 4
    assert re.search(r"\b4\b", res.message)


def test_no_exits_skips_without_universe(surv_results):
    res = surv_results["survivorship.no_exits"]
    assert res.status == Status.SKIP
    assert "artifacts.universe" in res.message


def test_no_exits_skips_on_too_few_assets():
    _, _, rets, signals, universe, positions = _base_frames(n_assets=5)
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=universe, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))["survivorship.no_exits"]
    assert res.status == Status.SKIP
    assert "min_assets_survivorship" in res.message


def test_no_exits_skips_on_short_sample():
    _, _, rets, signals, universe, positions = _base_frames(n_periods=200)
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=universe, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))["survivorship.no_exits"]
    assert res.status == Status.SKIP
    assert "200" in res.message


# ---------------------------------------------------------------------------
# 4. full_history_universe
# ---------------------------------------------------------------------------

def test_full_history_universe_warns_on_survivorship_case(surv_results):
    res = surv_results["survivorship.full_history_universe"]
    assert res.status == Status.WARN
    assert res.severity == Severity.HIGH
    assert res.details["frac_full_history"] == pytest.approx(1.0)
    assert res.details["n_assets"] == 30
    assert re.search(r"100% of 30 assets", res.message)
    assert "universe" in res.message
    assert "point-in-time" in res.remediation


def test_full_history_universe_superseded_when_universe_present(clean_results):
    res = clean_results["survivorship.full_history_universe"]
    assert res.status == Status.SKIP
    assert "no_exits" in res.message


def test_full_history_universe_passes_when_names_die():
    # no universe artifact, but half the names visibly die early ->
    # frac_full_history well below the 0.95 warn threshold
    _, _, rets, signals, _, positions = _base_frames()
    rets.iloc[150:, :6] = np.nan     # 6 of 12 assets stop trading at bar 150
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=None, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))["survivorship.full_history_universe"]
    assert res.status == Status.PASS
    assert res.details["frac_full_history"] == pytest.approx(0.5)


def test_full_history_universe_skips_on_too_few_assets():
    _, _, rets, signals, _, positions = _base_frames(n_assets=5)
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=None, signal_lag=1)
    res = _by_id(run(arts, AuditConfig()))["survivorship.full_history_universe"]
    assert res.status == Status.SKIP
    assert "min_assets_survivorship" in res.message


# ---------------------------------------------------------------------------
# hygiene: run() must not mutate the artifacts it receives
# ---------------------------------------------------------------------------

def test_run_does_not_mutate_artifacts():
    _, _, rets, signals, universe, positions = _base_frames()
    universe.iloc[100:, 0] = False
    arts = _aligned(signals=signals, asset_returns=rets, positions=positions,
                    universe=universe, signal_lag=1)
    before = {
        "signals": arts.signals.copy(),
        "asset_returns": arts.asset_returns.copy(),
        "positions": arts.positions.copy(),
        "universe": arts.universe.copy(),
    }
    run(arts, AuditConfig())
    pd.testing.assert_frame_equal(arts.signals, before["signals"])
    pd.testing.assert_frame_equal(arts.asset_returns, before["asset_returns"])
    pd.testing.assert_frame_equal(arts.positions, before["positions"])
    pd.testing.assert_frame_equal(arts.universe, before["universe"])
