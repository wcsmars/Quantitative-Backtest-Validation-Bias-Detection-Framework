"""Scheduled signals, date-shift ambiguity, and narrow-book performance scope.

Gated signals and scheduled activity need conditional IC and execution
interpretations. Embedded-target leaks remain flagged, while insufficient
breadth and insufficient history retain distinct verdicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import leakage, performance
from qaudit.config import AuditConfig
from qaudit.dynamic.probes_shift import CHECK_DATE_SHIFT
from qaudit.dynamic.probes_shift import run as shift_run
from qaudit.inputs import BacktestArtifacts
from qaudit.types import Severity, Status

CFG = AuditConfig(n_trials=1)
SUSP_IC = "performance.suspicious_ic"

# master business-day calendar; event = Monday (strictly spaced, gap >= 4)
MASTER = pd.bdate_range("2016-01-01", "2024-12-31")
EV = pd.Series(MASTER.dayofweek == 0, index=MASTER)
NXT = pd.Series(np.r_[EV.to_numpy()[1:], False], index=MASTER)  # event-eve


# --------------------------------------------------------------- builders --

def calendar_book(seed=0, strength=0.0035, cost_bps=0.0, n_assets=30,
                  n_periods=1000, leak=False):
    """Day-of-week alpha is estimated causally by an expanding per-asset mean and exported
    only on event eves. Lag-one bets land on scheduled bars. The leak variant
    substitutes the next return on the same support and must remain flagged."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_periods)
    assets = [f"A{i:03d}" for i in range(n_assets)]
    beta = rng.uniform(0.7, 1.3, n_assets)
    mkt = rng.normal(2e-4, 0.010, n_periods)
    sigma = rng.uniform(0.015, 0.025, n_assets)
    load = rng.normal(0.0, 1.0, n_assets)
    ev = EV.reindex(dates).to_numpy()
    nxt = NXT.reindex(dates).to_numpy()
    eps = rng.normal(0.0, 1.0, (n_periods, n_assets)) * sigma
    rets = pd.DataFrame(eps + np.outer(ev.astype(float), load) * strength
                        + np.outer(mkt, beta), index=dates, columns=assets)

    def signal_func(signal_input: pd.DataFrame) -> pd.DataFrame:
        r = signal_input
        ev_loc = EV.reindex(r.index).to_numpy()
        nxt_loc = NXT.reindex(r.index).to_numpy()
        rv = r.to_numpy(dtype=float)
        n, a = rv.shape
        cum_ev = np.zeros(a); cnt_ev = np.zeros(a)
        cum_ot = np.zeros(a); cnt_ot = np.zeros(a)
        est = np.full((n, a), np.nan)
        for t in range(n):
            x = rv[t]
            fin = np.isfinite(x)
            if ev_loc[t]:
                cum_ev[fin] += x[fin]; cnt_ev[fin] += 1
            else:
                cum_ot[fin] += x[fin]; cnt_ot[fin] += 1
            ok = (cnt_ev >= 8) & (cnt_ot >= 8)
            est[t, ok] = cum_ev[ok] / cnt_ev[ok] - cum_ot[ok] / cnt_ot[ok]
        sig = est * nxt_loc[:, None].astype(float)
        return pd.DataFrame(sig, index=r.index, columns=r.columns)

    if leak:   # gated target leak: tomorrow's return, stamped on event eves
        signals = rets.shift(-1).mul(pd.Series(nxt, index=dates)
                                     .astype(float), axis=0)
        signals.iloc[:20] = 0.0     # leave a warmup, like the honest book
    else:
        signals = signal_func(rets)

    def make_positions(sig):
        lagged = sig.shift(1)
        rk = lagged.rank(axis=1)
        w = rk.sub(rk.mean(axis=1), axis=0)
        g = w.abs().sum(axis=1)
        pos = w.div(g.replace(0.0, np.nan), axis=0).fillna(0.0)
        live = (lagged.abs() > 1e-15).any(axis=1)
        return pos.mul(live.astype(float), axis=0)

    def backtest_func(sig, asset_returns):
        pos = make_positions(sig)
        gross = (pos * asset_returns.fillna(0.0)).sum(axis=1)
        traded = pos.diff().abs().sum(axis=1)
        if len(traded):
            traded.iloc[0] = pos.iloc[0].abs().sum()
        return gross - traded * (cost_bps * 1e-4)

    art = BacktestArtifacts(
        signals=signals, asset_returns=rets, positions=make_positions(signals),
        strategy_returns=backtest_func(signals, rets), signal_input=rets,
        signal_lag=1, declared_costs_bps=cost_bps or None,
        periods_per_year=252).aligned()
    return art, signal_func, backtest_func


def pairs_book(seed=0, n_pairs=1, n_periods=1000):
    """Honest pairs/spread book: AR(1) spread, causal
    trailing z-score signal, 1-day lag, costs charged. Breadth 2*n_pairs
    < 5 on every date - structurally below the cross-sectional floor."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_periods)
    rets_cols, sig_cols, pos_cols, assets = {}, {}, {}, []
    for p in range(n_pairs):
        a, b = f"P{p}A", f"P{p}B"
        assets += [a, b]
        common = rng.normal(2e-4, 0.010, n_periods)
        spread = np.zeros(n_periods)
        eps = rng.normal(0, 0.004, n_periods)
        for t in range(1, n_periods):
            spread[t] = 0.97 * spread[t - 1] + eps[t]
        d_spread = np.diff(spread, prepend=0.0)
        ra = common + 0.5 * d_spread + rng.normal(0, 0.003, n_periods)
        rb = common - 0.5 * d_spread + rng.normal(0, 0.003, n_periods)
        rets_cols[a], rets_cols[b] = ra, rb
        cum = pd.Series(np.cumsum(ra - rb), index=dates)
        z = ((cum - cum.rolling(60, min_periods=60).mean())
             / cum.rolling(60, min_periods=60).std())
        sig_cols[a], sig_cols[b] = -z, z
        w = (-z).clip(-1, 1) * 0.5
        pos_cols[a], pos_cols[b] = w, -w
    rets = pd.DataFrame(rets_cols, index=dates)[assets]
    signals = pd.DataFrame(sig_cols, index=dates)[assets]
    pos = pd.DataFrame(pos_cols, index=dates)[assets].shift(1).fillna(0.0)
    gross = (pos * rets).sum(axis=1)
    traded = pos.diff().abs().sum(axis=1)
    traded.iloc[0] = pos.iloc[0].abs().sum()
    return BacktestArtifacts(
        signals=signals, asset_returns=rets, positions=pos,
        strategy_returns=gross - traded * 5e-4, signal_lag=1,
        declared_costs_bps=5.0, periods_per_year=252).aligned()


def perf_check(art, name=SUSP_IC, cfg=CFG):
    return {r.check: r for r in performance.run(art, cfg)}[name]


def date_shift(art, bf, sf=None, cfg=CFG):
    res = shift_run(art, cfg, signal_func=sf, backtest_func=bf)
    return {r.check: r for r in res}[CHECK_DATE_SHIFT]


# ------------------------------------------- 1. suspicious_ic gated books --

@pytest.mark.parametrize("seed", [0, 1, 6])
def test_honest_gated_calendar_book_scoped_never_failed(seed):
    # A gated causal book must not fail; any warning must name the gated interpretation
    # and its supporting measurements.
    art, _, _ = calendar_book(seed=seed, cost_bps=10.0)
    r = perf_check(art)
    assert r.status is not Status.FAIL
    assert "almost certainly embeds" not in r.message
    assert "rare for real alphas" not in r.message
    if r.status is Status.WARN:
        assert r.severity is Severity.MEDIUM  # below the fail bar
        assert "conditioned on" in r.message
        assert "cannot separate" in r.message
        assert r.details["gated_stamping"] is True
        assert "diluted_mean_ic" in r.details
        # the honest gated book's diluted equivalent sits in the honest band
        assert abs(r.details["diluted_mean_ic"]) < 0.06


def test_continuous_export_of_same_score_unchanged():
    # Exporting the same seasonal estimate every day retains the continuous-book path
    # because coverage is high.
    art, sf, _ = calendar_book(seed=0, cost_bps=10.0)
    est = sf(art.asset_returns)                     # gated stamp...
    dense = est.replace(0.0, np.nan).ffill().fillna(0.0)  # ...spread daily
    art2 = BacktestArtifacts(signals=dense, asset_returns=art.asset_returns,
                             periods_per_year=252).aligned()
    r = perf_check(art2)
    assert r.details["gated_stamping"] is False
    assert r.status is Status.PASS
    assert "conditioned on" not in r.message


def test_gated_embedded_target_leak_still_fails():
    # An embedded target on gated support still clears _GATED_EMBED_IC_FAIL and must
    # retain its failure.
    art, _, _ = calendar_book(seed=0, leak=True)
    r = perf_check(art)
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert r.details["gated_stamping"] is True
    assert r.details["mean_ic"] > 0.5
    assert "embeds the target" in r.message


def test_gated_ambiguous_zone_leak_still_warns_high():
    # A diluted gated leak (conditional IC inside the honest-overlap band
    # 0.15-0.5) gets the scoped WARN at HIGH severity - flagged, honestly
    # worded, not silenced and not softened below the fail-bar strength.
    art, _, _ = calendar_book(seed=0)
    rng = np.random.default_rng(21)
    nxt = (art.signals.abs().sum(axis=1) > 0).to_numpy()
    fwd = art.asset_returns.shift(-1)
    noise = pd.DataFrame(rng.normal(0, 0.06, art.signals.shape),
                         index=art.signals.index, columns=art.signals.columns)
    diluted = (fwd + noise).mul(pd.Series(nxt.astype(float),
                                          index=art.signals.index), axis=0)
    art2 = BacktestArtifacts(signals=diluted,
                             asset_returns=art.asset_returns,
                             periods_per_year=252).aligned()
    r = perf_check(art2)
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert 0.15 <= abs(r.details["mean_ic"]) < 0.5, r.details["mean_ic"]
    assert "cannot separate" in r.message


def test_gated_sparse_stamp_skip_floor_preserved():
    # Fewer than MIN_IC_DATES usable dates skips before gated-signal interpretation.
    art, _, _ = calendar_book(seed=0, n_periods=1000)
    keep = art.signals.index.to_period("M")
    first_of_month = ~pd.Series(keep, index=art.signals.index).duplicated()
    stamped = art.signals.index[first_of_month.to_numpy()][:12]
    sparse = pd.DataFrame(np.nan, index=art.signals.index,
                          columns=art.signals.columns)
    rng = np.random.default_rng(3)
    sparse.loc[stamped] = rng.normal(size=(len(stamped),
                                           art.signals.shape[1]))
    art2 = BacktestArtifacts(signals=sparse,
                             asset_returns=art.asset_returns,
                             periods_per_year=252).aligned()
    r = perf_check(art2)
    assert r.status is Status.SKIP
    assert "too few" in r.message


# --------------------------------------- 2. date_shift scheduled carve-out --

def test_scheduled_book_bf_only_zero_cost_never_critical():
    # A scheduled causal book with zero declared cost and only a backtest callback
    # needs a HIGH calendar-scoped warning.
    for seed in (1, 3):
        art, _, bf = calendar_book(seed=seed, cost_bps=0.0)
        r = date_shift(art, bf)
        assert r.status is Status.WARN, r.message
        assert r.severity is Severity.HIGH          # no causal stamp yet
        assert "strictly-spaced calendar subset" in r.message
        assert "embeds that bar's return" not in r.message
        assert "cannot tell them apart" in r.message
        sched = r.details["scheduled_activity"]
        assert sched["activity_source"] == "positions"
        assert sched["min_gap"] >= 2 and sched["active_frac"] <= 0.5


def test_scheduled_book_full_causal_stamp_warn_medium():
    # With signal_func verified causal the carve-out grants the softer
    # severity - still WARN, never PASS (a leak zeroed off-schedule would
    # collapse identically; the stamp exonerates only the computation).
    art, sf, bf = calendar_book(seed=1, cost_bps=0.0)
    r = date_shift(art, bf, sf=sf)
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert "verified causal" in r.message
    assert "strictly-spaced calendar subset" in r.message


def test_scheduled_book_delay_rule_names_third_reading():
    # The delay-rule collapse on a known calendar schedule needs a scoped MEDIUM
    # warning.
    art, sf, bf = calendar_book(seed=0, cost_bps=10.0)
    r = date_shift(art, bf, sf=sf)
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert "known in advance" in r.message
    assert "ultra-fast" not in r.message


def test_dense_same_bar_leak_still_fails_critical():
    # Attack guard: a continuous embedded-target book is active nearly
    # every bar - the carve-out must not touch it and the peek rule keeps
    # its CRITICAL conviction.
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2019-01-02", periods=600)
    rets = pd.DataFrame(rng.normal(0, 0.01, (600, 20)), index=dates,
                        columns=[f"A{i}" for i in range(20)])
    signals = rets.shift(-1)          # signal_t = the return the bet earns

    def bf(sig, asset_returns):
        lagged = sig.shift(1)
        rk = lagged.rank(axis=1)
        w = rk.sub(rk.mean(axis=1), axis=0)
        pos = w.div(w.abs().sum(axis=1).replace(0.0, np.nan),
                    axis=0).fillna(0.0)
        return (pos * asset_returns.fillna(0.0)).sum(axis=1)

    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            strategy_returns=bf(signals, rets),
                            periods_per_year=252).aligned()
    r = date_shift(art, bf)
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert "embeds that bar's return" in r.message
    assert "scheduled_activity" not in r.details


def test_gated_off_schedule_leak_not_exonerated():
    # Arms-race guard: a same-bar leak deliberately zeroed off-schedule
    # rides the carve-out down from CRITICAL, but must still be flagged
    # (WARN HIGH, hedged) - never PASS.
    art, _, bf = calendar_book(seed=2, cost_bps=0.0, leak=True)
    r = date_shift(art, bf)
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert "cannot tell them apart" in r.message


# ------------------------------------ 3. leakage breadth-bound SKIP split --

@pytest.mark.parametrize("n_pairs", [1, 2])
def test_pairs_books_breadth_bound_skip_not_permanent_warn(n_pairs):
    # Insufficient breadth is structural; more history cannot produce usable cross-
    # sectional evidence.
    art = pairs_book(seed=0, n_pairs=n_pairs)
    res = {r.check: r for r in leakage.run(art, CFG)}
    for check in (leakage.CHECK_TARGET_CORR, leakage.CHECK_IDENTITY,
                  leakage.CHECK_OUTLIER):
        r = res[check]
        assert r.status is Status.SKIP, (check, r.message)
        assert "no matter how long the history" in r.message
        assert "Extend the overlapping" not in (r.remediation or "")
        # _MIN_NAMES is fixed by design: no config-lowering escape offered
        assert "config" not in r.message.replace("not configurable", "")
        assert r.details["max_joint_names"] == 2 * n_pairs
    # The perfect-rank sibling retains its own breadth skip.
    assert res[leakage.CHECK_PERFECT].status is Status.SKIP


def test_history_bound_sparsity_still_warns():
    # The WARN + extend-history remediation is the right verdict when the
    # breadth is reachable and only the history is short: 8 names, ~24
    # usable dates.
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=25)
    rets = pd.DataFrame(rng.normal(0, 0.01, (25, 8)), index=dates,
                        columns=[f"A{i}" for i in range(8)])
    signals = pd.DataFrame(rng.normal(size=(25, 8)), index=dates,
                           columns=rets.columns)
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            periods_per_year=252).aligned()
    res = {r.check: r for r in leakage.run(art, CFG)}
    for check in (leakage.CHECK_TARGET_CORR, leakage.CHECK_IDENTITY,
                  leakage.CHECK_OUTLIER):
        r = res[check]
        assert r.status is Status.WARN, (check, r.message)
        assert r.severity is Severity.LOW
        assert "Extend the overlapping" in (r.remediation or "")


def test_breadth_five_panel_unaffected():
    # At exactly _MIN_NAMES the three checks measure normally (no SKIP).
    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2020-01-02", periods=400)
    rets = pd.DataFrame(rng.normal(0, 0.01, (400, 5)), index=dates,
                        columns=[f"A{i}" for i in range(5)])
    signals = pd.DataFrame(rng.normal(size=(400, 5)), index=dates,
                           columns=rets.columns)
    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            periods_per_year=252).aligned()
    res = {r.check: r for r in leakage.run(art, CFG)}
    for check in (leakage.CHECK_TARGET_CORR, leakage.CHECK_IDENTITY,
                  leakage.CHECK_OUTLIER):
        assert res[check].status is Status.PASS, check
