"""Universe calendar coverage and calendar-scaled attrition sample floors.

Sparse membership stamps need explicit alignment before auditing. Attrition
uses elapsed calendar time so monthly data do not require decades of bars.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import survivorship
from qaudit.checks.survivorship import MIN_BARS_NO_EXITS, MIN_YEARS_NO_EXITS
from qaudit.config import AuditConfig
from qaudit.errors import MisalignedInputError
from qaudit.inputs import _UNIVERSE_MIN_COVERAGE, BacktestArtifacts
from qaudit.synthetic import (momentum_signal, positions_from_signals,
                              simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()


def _by(results):
    return {r.check: r for r in results}


@pytest.fixture(scope="module")
def market():
    return simulate_market(n_assets=30, n_periods=1000, seed=5)


@pytest.fixture(scope="module")
def honest(market):
    return momentum_signal(market["returns"])


def _art(market, sig, universe, **kw):
    base = dict(signals=sig, asset_returns=market["returns"],
                positions=positions_from_signals(sig, 1),
                universe=universe, signal_lag=1)
    base.update(kw)
    return BacktestArtifacts(**base)


def _month_end_stamps(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = idx.to_series()
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().sort_values())


# Sparse membership-calendar rejection.

def test_month_end_stamped_universe_rejected(market, honest):
    uni = market["universe"].loc[_month_end_stamps(market["universe"].index)]
    art = _art(market, honest, uni)
    with pytest.raises(MisalignedInputError) as exc:
        art.validate()
    msg = str(exc.value)
    assert "artifacts.universe" in msg
    # actionable: names the fabricated-False mechanism and the ffill fix
    assert "fillna(False)" in msg
    assert "ffill" in msg
    assert re.search(r"\d+ of \d+ trading dates", msg)


def test_weekly_stamped_universe_rejected(market, honest):
    # weekly rebalance-row stamping (~20% coverage) must also fail - the
    # floor is calibrated above the densest common stamping artifact
    idx = market["universe"].index
    uni = market["universe"].loc[idx[idx.weekday == 4]]
    art = _art(market, honest, uni)
    with pytest.raises(MisalignedInputError, match="universe"):
        art.validate()
    # Pin the membership coverage floor within a band: low thresholds admit fabricated
    # missing membership, while values above 0.70 reject the dense late-start control.
    assert 0.5 <= _UNIVERSE_MIN_COVERAGE <= 0.70


def test_sparse_universe_cannot_launder_no_exits(market):
    # A monthly-stamped survivor list fabricates exits at stamp boundaries; validation
    # must reject it before attrition checks.
    m = simulate_market(n_assets=60, seed=3, death_frac=0.0)
    total = (1 + m["returns"]).prod()
    keep = total.sort_values(ascending=False).index[:30]
    rets = m["returns"][keep]
    uni = pd.DataFrame(True, index=rets.index, columns=rets.columns)
    uni = uni.loc[_month_end_stamps(uni.index)]
    sig = momentum_signal(rets)
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            positions=positions_from_signals(sig, 1),
                            universe=uni, signal_lag=1)
    with pytest.raises(MisalignedInputError, match="no_exits"):
        art.validate()


def test_sparse_universe_error_mentions_stamp_spacing(market, honest):
    uni = market["universe"].loc[_month_end_stamps(market["universe"].index)]
    art = _art(market, honest, uni)
    with pytest.raises(MisalignedInputError) as exc:
        art.validate()
    # message estimates the stamping cadence so the user recognises the
    # vendor artifact (~21 trading days for month-end stamps)
    m = re.search(r"stamped every ~(\d+) trading days", str(exc.value))
    assert m and 18 <= int(m.group(1)) <= 23


def test_zero_overlap_gate_still_intact(market, honest):
    # the zero-overlap gate is independent of the coverage gate: fully
    # disjoint dates still raise (not fall through to a divide or a
    # coverage message)
    uni = market["universe"].copy()
    uni.index = uni.index + pd.DateOffset(years=30)
    art = _art(market, honest, uni)
    with pytest.raises(MisalignedInputError, match="shares no"):
        art.validate()


# Dense and partially covered membership controls.

def test_dense_universe_still_validates(market, honest):
    art = _art(market, honest, market["universe"])
    art.validate()   # must not raise
    r = _by(survivorship.run(art.aligned(), CFG))[
        "survivorship.trading_outside_universe"]
    assert r.status is Status.PASS


def test_holiday_drift_universe_still_validates(market, honest):
    # a universe on a real exchange calendar misses a few % of a naive
    # bdate grid - legitimate, far above the floor
    rng = np.random.default_rng(7)
    uni = market["universe"]
    keep = rng.random(len(uni)) > 0.03
    art = _art(market, honest, uni.loc[uni.index[keep]])
    art.validate()   # must not raise


def test_late_starting_dense_universe_still_validates(market, honest):
    # membership history shorter than the panel (dense, back 70%) keeps the
    # same partial-overlap latitude validate() grants positions
    uni = market["universe"]
    art = _art(market, honest, uni.iloc[int(0.30 * len(uni)):])
    art.validate()   # must not raise


def test_sparse_positions_not_coverage_gated(market, honest):
    # Deliberate asymmetry: positions rows missing from the grid decode to
    # 0.0 = flat, a legitimate sparse encoding (and a wrong one gets caught
    # loudly by the gross/strategy_returns cross-checks) - while universe
    # rows decode to False, which fabricates non-membership. So positions
    # keep only the zero-overlap gate.
    pos = positions_from_signals(honest, 1)
    art = _art(market, honest, market["universe"],
               positions=pos.loc[_month_end_stamps(pos.index)])
    art.validate()   # must not raise


def test_hindsight_universe_detection_power_intact(market, honest):
    # a zombie holding far past the 21-bar grace must still fail CRITICAL
    # on a calendar that passes validate()
    uni = market["universe"].copy()
    uni.iloc[:, :] = True
    uni.iloc[500:, 0] = False
    pos = positions_from_signals(honest, 1).copy()
    pos.iloc[:, 0] = 0.05                    # held throughout, incl. post-exit
    art = _art(market, honest, uni, positions=pos)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))[
        "survivorship.trading_outside_universe"]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL


# Attrition sample floors on monthly data.

def _monthly_frames(n_periods, n_assets=30, seed=0, n_exits=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-29", periods=n_periods, freq=pd.offsets.BMonthEnd())
    assets = [f"A{i:02d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0.008, 0.05, (n_periods, n_assets)),
                        index=dates, columns=assets)
    uni = pd.DataFrame(True, index=dates, columns=assets)
    for j in range(n_exits):
        t = int(rng.integers(int(0.3 * n_periods), int(0.9 * n_periods)))
        rets.iloc[t:, j] = np.nan
        uni.iloc[t:, j] = False
    sig = rets.rolling(3, min_periods=3).mean()
    return BacktestArtifacts(
        signals=sig, asset_returns=rets,
        positions=positions_from_signals(sig, 1),
        universe=uni, signal_lag=1, periods_per_year=12)


def test_no_exits_active_on_three_years_of_monthly_bars():
    # 36 monthly bars = 3 years: plenty of calendar time. A raw 252-bar floor
    # would silently SKIP here (disabled until 21 years of monthly bars).
    art = _monthly_frames(36)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.WARN
    assert "no asset ever leaves" in r.message
    assert r.details["years"] == pytest.approx(3.0)


def test_no_exits_activates_at_thirteen_monthly_bars():
    # ~13 months of monthly bars clears the 1-year floor (12 bars at
    # periods_per_year=12). run() is exercised directly: 13 dates is below
    # validate()'s global _MIN_OVERLAP, which is a separate, orthogonal gate.
    r = _by(survivorship.run(_monthly_frames(13).aligned(), CFG))[
        "survivorship.no_exits"]
    assert r.status is Status.WARN


def test_no_exits_below_floor_monthly_still_skips():
    r = _by(survivorship.run(_monthly_frames(11).aligned(), CFG))[
        "survivorship.no_exits"]
    assert r.status is Status.SKIP
    assert r.details["min_periods_required"] == max(
        int(np.ceil(MIN_YEARS_NO_EXITS * 12)), MIN_BARS_NO_EXITS)
    assert "periods_per_year=12" in r.message


def test_no_exits_token_exit_still_caught_on_monthly_bars():
    # detection power at monthly frequency: one token exit over 30 assets x
    # 5 years = 0.67%/yr, below the 1%/yr plausibility floor -> WARN
    art = _monthly_frames(60, n_exits=1)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.WARN
    assert r.details["n_exiting_assets"] == 1
    assert r.details["exit_rate_per_year"] < CFG.min_exit_rate_per_year


# Monthly attrition controls.

def test_no_exits_honest_monthly_universe_passes():
    # 4 of 30 names exit over 3 years (~4.4%/yr) - realistic attrition
    art = _monthly_frames(36, n_exits=4)
    art.validate()
    r = _by(survivorship.run(art.aligned(), CFG))["survivorship.no_exits"]
    assert r.status is Status.PASS
    assert r.details["n_exiting_assets"] == 4


def test_no_exits_daily_floor_unchanged():
    # 1 year of daily bars: the boundary must sit at 252, so calendar scaling
    # leaves the daily calibration unchanged
    def daily(n_periods):
        rng = np.random.default_rng(1)
        dates = pd.bdate_range("2020-01-02", periods=n_periods)
        assets = [f"A{i:02d}" for i in range(12)]
        rets = pd.DataFrame(rng.normal(0.0, 0.01, (n_periods, 12)),
                            index=dates, columns=assets)
        sig = rets.rolling(5, min_periods=5).mean()
        return BacktestArtifacts(
            signals=sig, asset_returns=rets,
            positions=positions_from_signals(sig, 1),
            universe=pd.DataFrame(True, index=dates, columns=assets),
            signal_lag=1)

    at = _by(survivorship.run(daily(252).aligned(), CFG))["survivorship.no_exits"]
    below = _by(survivorship.run(daily(251).aligned(), CFG))["survivorship.no_exits"]
    assert at.status is Status.WARN      # static universe, check active
    assert below.status is Status.SKIP   # guard against short-sample noise
    assert "251" in below.message
