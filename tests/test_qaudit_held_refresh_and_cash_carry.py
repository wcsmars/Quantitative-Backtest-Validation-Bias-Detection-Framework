"""Held-refresh execution schedules and unreported cash carry.

Position changes identify periodic refresh schedules despite occasional
forced closes. Delay-based leakage evidence remains active. Per-date drag
regression may identify bounded cash carry, but the result stays scoped
until cash is supplied as an explicit asset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import costs
from qaudit.config import AuditConfig
from qaudit.dynamic.probes_shift import (CHECK_DATE_SHIFT,
                                         SCHED_REFRESH_DOMINANT_SHARE,
                                         _refresh_grid_schedule,
                                         _strict_gap_schedule)
from qaudit.dynamic.probes_shift import run as shift_run
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import simulate_market
from qaudit.types import Severity, Status

CFG = AuditConfig(n_trials=1)
COSTS_CHECK = "costs.missing_transaction_costs"
PPY = 252
RF = 0.045
COST_BPS = 10.0


# --------------------------------------------------------------- builders --

def zx(df):
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)


def mom(rets, window):
    return rets.rolling(window, min_periods=window).mean()


def rank_weights(lagged_sig):
    r = lagged_sig.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    gross = w.abs().sum(axis=1)
    return w.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)


def net_from(pos, rets, costs_bps=COST_BPS):
    gross = (pos * rets.fillna(0.0)).sum(axis=1)
    traded = pos.diff().abs().sum(axis=1)
    if len(traded):
        traded.iloc[0] = pos.iloc[0].abs().sum()
    return gross - traded * (costs_bps * 1e-4)


def composite_signal_func(rets):
    """Mixed-frequency honest composite: daily 20d momentum + weekly-held
    5d reversal + monthly-held 120d momentum, all causal."""
    daily = zx(mom(rets, 20))
    rev = -zx(mom(rets, 5))
    wmask = np.zeros(len(rets), dtype=bool)
    wmask[4::5] = True
    weekly = rev.where(pd.Series(wmask, index=rets.index), np.nan).ffill()
    slow = zx(mom(rets, 120))
    mmask = np.zeros(len(rets), dtype=bool)
    mmask[20::21] = True
    monthly = slow.where(pd.Series(mmask, index=rets.index), np.nan).ffill()
    comp = (daily + 0.75 * weekly + 0.75 * monthly) / 2.5
    return comp.where(daily.notna())


def held_book(seed=0, weekly_sampled=True, signal_func=composite_signal_func):
    """Honest held book: engine samples the lag-1 signal every 5th bar
    (weekly_sampled) or every bar (daily control), holds rank weights in
    between, force-closes delisted names, charges 10bps."""
    rets = simulate_market(n_assets=40, seed=seed)["returns"]
    sig = signal_func(rets)

    def pos_from(signals):
        lagged = signals.shift(1)
        if not weekly_sampled:
            return rank_weights(lagged)
        mask = np.zeros(len(lagged), dtype=bool)
        mask[4::5] = True
        held = lagged.where(pd.Series(mask, index=lagged.index),
                            np.nan).ffill()
        held = held.where(lagged.notna())     # delisting force-close
        return rank_weights(held)

    def backtest_func(signals, asset_returns):
        return net_from(pos_from(signals), asset_returns)

    pos = pos_from(sig)
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            strategy_returns=net_from(pos, rets),
                            signal_input=rets, signal_lag=1,
                            declared_costs_bps=COST_BPS,
                            periods_per_year=PPY).aligned()
    return art, signal_func, backtest_func


def date_shift(art, bf, sf=None, cfg=CFG):
    res = shift_run(art, cfg, signal_func=sf, backtest_func=bf)
    return {r.check: r for r in res}[CHECK_DATE_SHIFT]


def cash_book(seed=100, cash_frac=0.25, monthly=False, cash_col=False,
              rf=RF, extra_noise_frac=0.0, long_short=False,
              charged_bps=COST_BPS, declared_bps=COST_BPS):
    """Long-only (or unit-gross long-short) book; cash_frac of NAV in cash
    earning rf accrued daily into strategy_returns; charged_bps genuinely
    deducted on every dollar traded; declared_bps on record."""
    rng = np.random.default_rng(seed)
    n, k = 1000, 30
    dates = pd.bdate_range("2020-01-02", periods=n)
    cols = [f"A{i:02d}" for i in range(k)]
    rets = pd.DataFrame(rng.normal(3e-4, 0.015, (n, k)), index=dates,
                        columns=cols)
    sig = rets.rolling(15).mean()
    lag = sig.shift(1)
    r = lag.rank(axis=1)
    if long_short:
        w = r.sub(r.mean(axis=1), axis=0)
        w = w.div(w.abs().sum(axis=1).replace(0.0, np.nan),
                  axis=0).fillna(0.0)
    else:
        top = r.gt(2 * k / 3.0)
        w = top.astype(float)
        w = w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    if monthly:
        keep = np.zeros(n, dtype=bool)
        keep[::21] = True
        w = w.where(pd.Series(keep, index=dates), np.nan).ffill().fillna(0.0)
    if not long_short:
        w = w * (1.0 - cash_frac)
    gross = (w * rets).sum(axis=1)
    traded = w.diff().abs().sum(axis=1)
    traded.iloc[0] = w.iloc[0].abs().sum()
    carry = cash_frac * rf / PPY
    net = gross - traded * charged_bps * 1e-4 + carry
    if extra_noise_frac:
        net = net + rng.normal(0.0, extra_noise_frac * max(carry, 1e-9), n)
    if cash_col:
        rets = rets.copy()
        rets["CASH"] = rf / PPY
        sig = sig.reindex(columns=rets.columns)
        w = w.reindex(columns=rets.columns)
        w["CASH"] = cash_frac
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=w,
                            strategy_returns=net, signal_lag=1,
                            declared_costs_bps=declared_bps,
                            periods_per_year=PPY).aligned()
    return {r_.check: r_ for r_ in costs.run(art, CFG)}[COSTS_CHECK]


# ------------------------------ 1. date_shift held-refresh carve-out (FP) --

def test_held_weekly_refresh_no_stamp_hedged_warn_not_critical():
    # A held-refresh book with only a backtest callback needs a HIGH warning naming
    # schedule aliasing, without a categorical timing accusation.
    art, _, bf = held_book(seed=0)
    r = date_shift(art, bf)
    assert r.status is Status.WARN, r.message
    assert r.severity is Severity.HIGH          # no causal stamp
    assert "REFRESHES them only on a strict schedule" in r.message
    assert "cannot tell them apart" in r.message
    assert "embeds that bar's return" not in r.message
    sched = r.details["scheduled_activity"]
    assert sched["schedule_kind"] == "held_refresh"
    assert sched["activity_source"] == "position changes"
    assert sched["dominant_gap"] == 5
    assert sched["dominant_gap_share"] >= SCHED_REFRESH_DOMINANT_SHARE
    assert sched["active_frac"] <= 0.5


def test_held_weekly_refresh_causal_stamp_buys_medium_never_pass():
    art, sf, bf = held_book(seed=0)
    r = date_shift(art, bf, sf=sf)
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert "verified causal" in r.message
    assert "REFRESHES them only on a strict schedule" in r.message


@pytest.mark.parametrize("seed", [1, 2])
def test_held_weekly_refresh_mc_never_fails(seed):
    # Below-floor books may pass; books clearing the evidence floor need a scoped
    # warning rather than a failure.
    art, _, bf = held_book(seed=seed)
    r = date_shift(art, bf)
    assert r.status is not Status.FAIL, r.message


def test_same_composite_traded_daily_control_passes():
    # Factorization control: the identical composite with a daily engine
    # shows the honest peek-helps curve - the collapse is pure
    # sampling-phase aliasing, so the daily control must stay PASS.
    art, _, bf = held_book(seed=0, weekly_sampled=False)
    r = date_shift(art, bf)
    assert r.status is Status.PASS, r.message


def test_slow_only_weekly_held_control_passes():
    # No fast sleeve -> nothing aliases against the refresh grid: the
    # weekly-held slow book keeps its honest curve and stays PASS (the
    # refresh schedule may be detected, but no rule fires).
    art, _, bf = held_book(seed=0,
                           signal_func=lambda rets: zx(mom(rets, 20)))
    r = date_shift(art, bf)
    assert r.status is Status.PASS, r.message


def test_welded_leak_on_weekly_held_book_keeps_delay_warn_high():
    # FN-cost pin (attack guard): a genuine same-bar leak sampled onto the
    # same weekly-held cadence never trips rule (a) - the embedded bar
    # stays inside the holding window (k=-1 SR ~ k=0) - and is caught by
    # rule (b) as WARN HIGH. The held_refresh carve-out must not soften
    # rule (b) to the sparse-book fill-achievability MEDIUM.
    rets = simulate_market(n_assets=40, seed=0)["returns"]
    leaky = zx(mom(rets, 20)) + 1.5 * zx(rets)   # embeds bar-t return

    def pos_from(signals):
        mask = np.zeros(len(signals), dtype=bool)
        mask[4::5] = True
        held = signals.where(pd.Series(mask, index=signals.index),
                             np.nan).ffill()
        held = held.where(signals.notna())
        return rank_weights(held)               # no lag: the leak pays

    def bf(signals, asset_returns):
        return net_from(pos_from(signals), asset_returns)

    pos = pos_from(leaky)
    art = BacktestArtifacts(signals=leaky, asset_returns=rets,
                            positions=pos,
                            strategy_returns=net_from(pos, rets),
                            signal_lag=1, declared_costs_bps=COST_BPS,
                            periods_per_year=PPY).aligned()
    r = date_shift(art, bf)
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH, r.message
    assert "known in advance" not in r.message   # not the sparse reading
    assert r.details["scheduled_activity"]["schedule_kind"] == "held_refresh"


def test_dense_daily_leak_with_positions_still_fails_critical():
    # Attack guard: a continuous embedded-target book changes positions
    # every bar - dominant change-gap 1 - so the change-based detector
    # must not carve it out and the peek rule keeps CRITICAL.
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2019-01-02", periods=600)
    rets = pd.DataFrame(rng.normal(0, 0.01, (600, 20)), index=dates,
                        columns=[f"A{i}" for i in range(20)])
    signals = rets.shift(-1)

    def bf(sig, asset_returns):
        return (rank_weights(sig.shift(1))
                * asset_returns.fillna(0.0)).sum(axis=1)

    pos = rank_weights(signals.shift(1))
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            positions=pos,
                            strategy_returns=bf(signals, rets),
                            periods_per_year=252).aligned()
    r = date_shift(art, bf)
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert "scheduled_activity" not in r.details


def test_refresh_detector_units():
    # Direct units: weekly refresh grid with sparse delisting-close bars
    # qualifies (dominant gap 5 >= 2, share >= 0.9) even though min gap is
    # 1; a daily-changed book (dominant gap 1) and a <90%-share ragged
    # grid never do; the sparse level-based detector reports its own kind.
    idx = pd.RangeIndex(500)
    held_vals = 0.1 + 1e-3 * np.repeat(np.arange(100), 5)[:500]
    pos = pd.DataFrame({c: held_vals for c in "abcd"}, index=idx)
    pos.iloc[123:, 0] = 0.0    # off-schedule delisting close, held at zero
    det = _refresh_grid_schedule(pos)
    assert det is not None and det["schedule_kind"] == "held_refresh"
    assert det["dominant_gap"] == 5
    daily = pd.DataFrame(np.random.default_rng(0).normal(size=(500, 4)),
                         index=idx, columns=list("abcd"))
    assert _refresh_grid_schedule(daily) is None
    steps = np.cumsum(np.random.default_rng(1).integers(2, 9, 90))
    steps = steps[steps < 500]
    ragged_vals = 0.1 + 1e-3 * np.cumsum(np.isin(np.arange(500), steps))
    ragged = pd.DataFrame({c: ragged_vals for c in "abcd"}, index=idx)
    assert _refresh_grid_schedule(ragged) is None   # no dominant cadence
    sparse = pd.Series(False, index=idx)
    sparse.iloc[::5] = True
    sdet = _strict_gap_schedule(sparse, "positions")
    assert sdet is not None and sdet["schedule_kind"] == "sparse_activity"


# ---------------------------- 2. costs cash-carry discriminator (FP+attack) --

def test_cash_buffer_daily_60pct_scoped_warn_not_mismatch_accusation():
    # Bounded cash carry can explain negative apparent drag; warn at MEDIUM and request
    # an explicit CASH asset column.
    r = cash_book(seed=100, cash_frac=0.60)
    assert (r.status, r.severity) == (Status.WARN, Severity.MEDIUM), r.message
    assert "cash" in r.message
    assert "uninvested" in r.message
    assert "cannot be negative" not in r.message
    assert "do not describe the same backtest" not in r.message
    fit = r.details["cash_carry_fit"]
    assert abs(fit["slope_bps"] - COST_BPS) < 1.0         # true charged cost
    assert abs(fit["implied_cash_yield_on_uninvested"] - RF) < 0.01
    assert "CASH column" in r.remediation or "asset_returns" in r.remediation


def test_cash_buffer_monthly_25pct_is_scoped_warn_not_critical():
    # A monthly allocation with 25% cash can earn enough carry to obscure correctly
    # charged costs.
    for seed in (100, 104):
        r = cash_book(seed=seed, cash_frac=0.25, monthly=True)
        assert (r.status, r.severity) == (Status.WARN, Severity.MEDIUM), \
            (seed, r.status, r.message)
        assert "never deducted" not in r.message
        assert "cash" in r.message


def test_cash_buffer_monthly_40pct_negative_implied_scoped():
    r = cash_book(seed=100, cash_frac=0.40, monthly=True)
    assert (r.status, r.severity) == (Status.WARN, Severity.MEDIUM)
    assert r.details["cash_carry_fit"]["carry_per_bar"] > 0


@pytest.mark.parametrize("cash_frac,expect_pass", [(0.25, True),
                                                   (0.40, False)])
def test_low_cash_daily_books_scoped_or_pass(cash_frac, expect_pass):
    # Per-date reconciliation tolerates immaterial carry. Material carry needs a scoped
    # decomposition warning and an explicit CASH column, so the implied trade cost is
    # not misstated.
    r = cash_book(seed=100, cash_frac=cash_frac)
    if expect_pass:
        assert r.status is Status.PASS, r.message
    else:
        assert (r.status, r.severity) == (Status.WARN, Severity.MEDIUM)
        assert "cash" in r.message
        fit = r.details["cash_carry_fit"]
        assert abs(fit["slope_bps"] - COST_BPS) < 1.0   # true charge recovered
        assert fit["carry_per_bar"] > 0


def test_cash_column_escape_hatch_still_passes():
    r = cash_book(seed=100, cash_frac=0.60, cash_col=True)
    assert r.status is Status.PASS, r.message
    assert abs(r.details["implied_bps"] - COST_BPS) < 0.5


def test_truly_uncharged_book_keeps_fail_critical():
    # Attack guard (slope prong): declared 10bps, nothing deducted, no
    # carry - the drag regression fits slope ~ 0 / intercept ~ 0 and the
    # sub-floor conviction must stand.
    r = cash_book(seed=100, cash_frac=0.0, charged_bps=0.0)
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert "cash" not in r.message


def test_uncharged_book_with_fake_carry_not_waived():
    # Attack guard (slope prong, negative branch): nothing deducted plus a
    # constant boost dressed as cash yield - slope ~ 0 is inconsistent
    # with the declared 10bps, so the mismatch accusation stands.
    r = cash_book(seed=100, cash_frac=0.60, charged_bps=0.0)
    assert (r.status, r.severity) == (Status.WARN, Severity.HIGH), r.message
    assert "cannot be negative" in r.message


def test_unit_gross_long_short_constant_boost_not_waived():
    # Attack guard (uninvested prong): a unit-gross long-short book has no
    # idle cash to earn on (uninvested ~ 0 -> carry cap ~ 0), so a
    # constant PnL boost cannot ride the cash-carry waiver even when costs
    # are charged at the declared rate (monthly-held so the boost actually
    # dominates the drag and trips the negative-implied branch).
    r = cash_book(seed=100, cash_frac=0.60, long_short=True, monthly=True)
    assert (r.status, r.severity) == (Status.WARN, Severity.HIGH), r.message
    assert "cannot be negative" in r.message
    assert "cash_carry_fit" not in r.details


def test_implausible_cash_yield_not_waived():
    # Attack guard (yield-cap prong): a 40%/yr "cash" stream on 25% idle
    # NAV is not cash - carry exceeds the 15%/yr cap on the uninvested
    # fraction and the accusation stands.
    r = cash_book(seed=100, cash_frac=0.25, monthly=True, rf=0.40)
    assert (r.status, r.severity) == (Status.WARN, Severity.HIGH), r.message
    assert "cash_carry_fit" not in r.details


def test_noisy_accrual_not_waived():
    # Attack guard (residual prong): per-date noise at 2x the carry scale
    # breaks the constant-accrual signature - the honest-branch verdict
    # (under-charged WARN MEDIUM) resumes instead of the scoped waiver.
    r = cash_book(seed=100, cash_frac=0.60, extra_noise_frac=2.0)
    assert r.status is Status.WARN
    assert "cash_carry_fit" not in r.details
    assert ("under-charged" in r.message
            or "cannot be negative" in r.message)
