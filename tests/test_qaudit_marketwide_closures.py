"""Shared-calendar missing returns and independent closure corroboration.

Market-wide closures need recurring calendar evidence; held names cannot
corroborate only one another. Non-held co-closures and post-gap return
conservation distinguish plausible closures from deleted portfolio losses.
Short samples retain an explicit unverifiable-closure warning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import survivorship
from qaudit.checks.survivorship import (EXOG_MIN_DESTROYED_SR, EXOG_Z_CRIT,
                                        MARKETWIDE_MIN_CLOSED_FRAC,
                                        MISSING_RETURN_FRAC,
                                        NONHELD_COCLOSURE_MIN_EXPECTED)
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.types import Severity, Status

CFG = AuditConfig()
S_MISSING = "survivorship.positions_on_missing_returns"


def _by(results):
    return {r.check: r for r in results}


def _missing(art):
    return _by(survivorship.run(art, CFG))[S_MISSING]


def _zscore_cs(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def _rankw(df):
    r = df.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    g = w.abs().sum(axis=1)
    return w.div(g.replace(0.0, np.nan), axis=0).fillna(0.0)


def _zw(df):
    z = _zscore_cs(df)
    return z.div(z.abs().sum(axis=1).replace(0.0, np.nan),
                 axis=0).fillna(0.0)


def _gross_sr(pos, rets, ppy=252):
    pnl = (pos * rets.fillna(0.0)).sum(axis=1)
    sd = float(pnl.std())
    return float(pnl.mean()) / sd * float(np.sqrt(ppy)) if sd > 0 else 0.0


def _worst_portfolio_dates(pos, rets, k):
    """The attacker's date selection: k isolated interior worst-PnL dates."""
    port = (pos * rets).sum(axis=1)
    dates = rets.index
    chosen, used = [], set()
    for ts in port.nsmallest(k * 3).index:
        i = dates.get_loc(ts)
        if i <= 1 or i >= len(dates) - 2 or i in used or (i - 1) in used \
                or (i + 1) in used:
            continue
        chosen.append(ts)
        used.add(i)
        if len(chosen) >= k:
            break
    return chosen


# Market-wide shared-date deletion under rank and z-score weight geometries.

def _shared_calendar_deletion_book(seed, n=800, m=25, k=24):
    """Delete all held returns on the worst dates of a causal rank-weighted book. The
    rotating zero-weight median asset stays finite, while roughly 96% of live names
    close together."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(m)]
    beta = rng.uniform(0.7, 1.3, m)
    mkt = rng.normal(2e-4, 0.010, n)
    rets = pd.DataFrame(rng.standard_normal((n, m)) * 0.02
                        + np.outer(mkt, beta), index=dates, columns=cols)
    sig = _zscore_cs(rets.rolling(20, min_periods=20).mean())
    pos = _rankw(sig.shift(1))
    held = pos.abs() > 1e-12
    sab = rets.copy()
    for ts in _worst_portfolio_dates(pos, rets, k):
        row = held.loc[ts]
        sab.loc[ts, row[row].index] = np.nan
    art = BacktestArtifacts(signals=sig, asset_returns=sab, positions=pos,
                            signal_lag=1)
    art.validate()
    return art.aligned(), _gross_sr(pos, rets), _gross_sr(pos, sab)


@pytest.mark.parametrize("seed", [0, 1])
def test_shared_calendar_worst_date_deletion_warns(seed):
    # Shared deletion of losing portfolio dates cannot corroborate itself through
    # affected assets.
    art, sr_clean, sr_attack = _shared_calendar_deletion_book(seed)
    assert sr_attack > sr_clean + 0.5          # the laundering is real
    r = _missing(art)
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["frac_uncorroborated_closure"] > MISSING_RETURN_FRAC
    # the mechanism is named: market-wide dates without recurrence
    assert r.details["n_marketwide_closure_dates"] >= 20
    assert r.details["n_marketwide_uncorroborated_cells"] > 400
    assert "market-wide" in r.message
    assert "recurrence" in r.message or "anniversary" in r.message
    assert "cannot tell" in r.message
    assert "delisting loss" not in r.message   # scoped, not an accusation
    assert r.remediation and "calendar" in r.remediation


def test_zweight_shared_calendar_deletion_warns():
    # Z-score weights hold every name, so held-only deletion removes whole rows. With
    # no non-held controls, the market-wide recurrence layer must detect it.
    rng = np.random.default_rng(100)
    n, m, k = 1000, 40, 15
    dates = pd.bdate_range("2018-01-02", periods=n)
    cols = [f"N{i:03d}" for i in range(m)]
    beta = rng.uniform(0.6, 1.4, m)
    mkt = rng.normal(2e-4, 0.011, n)
    rets = pd.DataFrame(rng.standard_normal((n, m)) * 0.018
                        + np.outer(mkt, beta), index=dates, columns=cols)
    sig = _zscore_cs(rets.rolling(15, min_periods=15).mean())
    pos = _zw(sig.shift(1))
    held = pos.abs() > 1e-12
    sab = rets.copy()
    for ts in _worst_portfolio_dates(pos, rets, k):
        row = held.loc[ts]
        sab.loc[ts, row[row].index] = np.nan
    art = BacktestArtifacts(signals=sig, asset_returns=sab, positions=pos,
                            signal_lag=1)
    art.validate()
    r = _missing(art.aligned())
    assert r.status is Status.WARN
    assert r.details["n_marketwide_closure_dates"] == k
    assert r.details["frac_uncorroborated_closure"] > MISSING_RETURN_FRAC


def _fullrow_crash_book(seed=3, n=1000, m=25, n_crash=20):
    """Net-long book whose signal is computed from the deleted panel at
    declared lag 1 (computing it off the clean panel would trip unrelated
    lookahead FAILs); whole rows deleted on market crash days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    beta = 0.8 + 0.4 * rng.random(m)
    mkt = rng.standard_normal(n) * 0.006 + 4e-4
    crash: list[int] = []
    for d in rng.permutation(np.arange(5, n - 5)):
        if all(abs(int(d) - c) > 1 for c in crash):
            crash.append(int(d))
        if len(crash) >= n_crash:
            break
    crash = sorted(crash)
    for d in crash:
        mkt[d] -= 0.05
    rets = pd.DataFrame(mkt[:, None] * beta[None, :]
                        + rng.standard_normal((n, m)) * 0.010,
                        index=dates, columns=[f"A{i:02d}" for i in range(m)])
    sab = rets.copy()
    sab.iloc[crash] = np.nan                       # full-row deletion
    sig = _zscore_cs(sab.rolling(10, min_periods=5).mean())
    tilt = np.sign(sig.shift(1)).fillna(0.0)
    pos = (1.0 + 0.3 * tilt) / m                   # net-long, all held
    art = BacktestArtifacts(signals=sig, asset_returns=sab, positions=pos,
                            signal_lag=1)
    art.validate()
    return (art.aligned(), len(crash), _gross_sr(pos, rets),
            _gross_sr(pos, sab))


def test_fullrow_crash_day_deletion_warns():
    # whole-row deletion leaves no non-held name and no open peer - only
    # the recurrence layer can catch it, and does
    art, n_crash, sr_clean, sr_attack = _fullrow_crash_book()
    assert sr_attack > sr_clean + 0.5
    r = _missing(art)
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["n_marketwide_closure_dates"] == n_crash
    assert r.details["frac_uncorroborated_closure"] > MISSING_RETURN_FRAC
    assert "market-wide" in r.message


def _marketwide_holiday_rows(dates, anchors):
    """Realize (month, day) anchors on the nearest index date, per year."""
    rows = set()
    for y in sorted(set(dates.year)):
        for mo, dd in anchors:
            target = pd.Timestamp(year=y, month=mo, day=dd)
            i = int(dates.searchsorted(target))
            if i >= len(dates):
                i = len(dates) - 1
            if i > 0 and abs((dates[i - 1] - target).days) < abs(
                    (dates[i] - target).days):
                i -= 1
            if 2 <= i <= len(dates) - 3:
                rows.add(i)
    return sorted(rows)


_ANCHORS = ((1, 1), (2, 18), (4, 7), (5, 27), (7, 4), (9, 3), (11, 26),
            (12, 25))


def _marketwide_holiday_book(n=1008, seed=11, m=25):
    """Honest guard for the recurrence layer: a bdate_range panel whose
    exchange holidays appear as full-row NaNs on a recurring calendar
    (~8 anchors/year), positions held throughout - the standard vendor
    shape for any single-market book."""
    dates = pd.bdate_range("2016-01-04", periods=n)
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(rng.normal(2e-4, 0.01, (len(dates), m)),
                        index=dates, columns=[f"A{i:02d}" for i in range(m)])
    rows = _marketwide_holiday_rows(dates, _ANCHORS)
    rets.iloc[rows] = np.nan
    sig = rets.rolling(10, min_periods=5).mean()
    pos = pd.DataFrame(1.0 / m, index=dates, columns=rets.columns)
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=1)
    art.validate()
    return art.aligned(), len(rows)


def test_recurring_marketwide_holiday_calendar_passes():
    # Recurring full-row holidays over four years must be corroborated with zero
    # uncorroborated mass.
    art, n_hol = _marketwide_holiday_book(n=1008)
    r = _missing(art)
    assert r.status is Status.PASS
    assert r.details["n_marketwide_closure_dates"] == n_hol
    assert r.details["n_marketwide_recurring_dates"] == n_hol
    assert r.details["n_closure_uncorroborated_cells"] == 0
    assert r.details["n_closure_corroborated_cells"] > 500
    assert "market-holiday" in r.message or "closure" in r.message


def test_short_sample_marketwide_holidays_scoped_warn_untestable():
    # A one-year book has no in-sample anniversary evidence. Unverifiable closure mass
    # above tolerance needs a scoped warning requesting exchange-calendar
    # corroboration.
    art, n_hol = _marketwide_holiday_book(n=252)
    r = _missing(art)
    assert r.status is Status.WARN
    assert r.details["n_marketwide_untestable_dates"] == n_hol
    assert r.details["n_marketwide_recurring_dates"] == 0
    assert r.details["n_closure_uncorroborated_cells"] == 0
    assert r.details["n_marketwide_unverifiable_cells"] > 0
    assert "cannot" in r.message.lower()          # scoped, not an accusation
    assert "delisting loss" not in r.message
    assert r.remediation and "calendar" in r.remediation


def test_single_asset_recurring_holidays_pass():
    # a single-asset book has no possible sibling, so its honest recurring
    # exchange holidays (~2.9% of cells) can only be corroborated by the
    # recurrence layer
    dates = pd.bdate_range("2016-01-04", periods=1008)
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(3e-4, 0.01, (len(dates), 1)),
                        index=dates, columns=["ONLY"])
    rows = _marketwide_holiday_rows(dates, _ANCHORS)
    rets.iloc[rows] = np.nan
    sig = rets.rolling(10, min_periods=5).mean()
    pos = pd.DataFrame(1.0, index=dates, columns=["ONLY"])
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=1)
    art.validate()
    r = _missing(art.aligned())
    assert r.status is Status.PASS
    assert r.details["n_closure_uncorroborated_cells"] == 0
    assert r.details["n_marketwide_recurring_dates"] == len(rows)


def test_single_asset_adhoc_gaps_still_warn():
    # detection guard paired with the above: the same single-asset book
    # with 30 non-recurring interior gaps keeps the scoped WARN
    dates = pd.bdate_range("2016-01-04", periods=1008)
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(3e-4, 0.01, (len(dates), 1)),
                        index=dates, columns=["ONLY"])
    rows: list[int] = []
    for d in rng.permutation(np.arange(5, len(dates) - 5)):
        if all(abs(int(d) - x) > 1 for x in rows):
            rows.append(int(d))
        if len(rows) >= 30:
            break
    rets.iloc[sorted(rows)] = np.nan
    sig = rets.rolling(10, min_periods=5).mean()
    pos = pd.DataFrame(1.0, index=dates, columns=["ONLY"])
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=1)
    art.validate()
    r = _missing(art.aligned())
    assert r.status is Status.WARN
    assert r.details["frac_uncorroborated_closure"] > MISSING_RETURN_FRAC
    assert "cannot tell" in r.message


# ===========================================================================
# 2. non-held co-closure deficit (the same-book-sibling corroboration cap)
# ===========================================================================

def test_partial_held_subset_deletion_fires_deficit_guard():
    # attack: delete only the 15 largest-|weight| held names per
    # worst date - 60% of live names, below the market-wide line, siblings
    # fully corroborate - but the closures never touch a non-held cell
    # (observed 0 vs ~14 expected), which no real holiday calendar does
    rng = np.random.default_rng(0)
    n, m, k = 800, 25, 24
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(m)]
    beta = rng.uniform(0.7, 1.3, m)
    mkt = rng.normal(2e-4, 0.010, n)
    rets = pd.DataFrame(rng.standard_normal((n, m)) * 0.02
                        + np.outer(mkt, beta), index=dates, columns=cols)
    sig = _zscore_cs(rets.rolling(20, min_periods=20).mean())
    pos = _rankw(sig.shift(1))
    sab = rets.copy()
    for ts in _worst_portfolio_dates(pos, rets, k):
        sab.loc[ts, pos.loc[ts].abs().nlargest(15).index] = np.nan
    art = BacktestArtifacts(signals=sig, asset_returns=sab, positions=pos,
                            signal_lag=1)
    art.validate()
    r = _missing(art.aligned())
    assert r.status is Status.WARN
    assert "nonheld_coclosure_deficit" in \
        r.details["corroboration_guards_fired"]
    assert r.details["nonheld_coclosure_observed"] == 0
    assert (r.details["nonheld_coclosure_expected"]
            >= NONHELD_COCLOSURE_MIN_EXPECTED)
    assert r.details["n_marketwide_closure_dates"] == 0   # below the line
    assert r.details["frac_uncorroborated_closure"] > MISSING_RETURN_FRAC
    assert "non-held" in r.message


def _rotating_region_book(seed=5, n=400, m=25, hpr=14, held_top=None):
    """Honest guard for the deficit cap: region holiday calendars on a book
    whose held set rotates (rank weights zero the median name; or a top-K
    subset book), so closures hit held and non-held cells alike and the
    expected/observed non-held co-closure counts agree."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"A{i:02d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, m)), index=dates,
                        columns=cols)
    groups = [cols[0:8], cols[8:16], cols[16:25]]
    starts = rng.choice(np.arange(10, n - 11), size=3 * hpr, replace=False)
    for gi, g in enumerate(groups):
        for s in starts[gi * hpr:(gi + 1) * hpr]:
            rets.iloc[s:s + 1, [cols.index(c) for c in g]] = np.nan
    sig = _zscore_cs(rets.rolling(10, min_periods=10).mean())
    if held_top:
        rk = sig.shift(1).rank(axis=1)
        pos = ((rk > m - held_top).astype(float) / held_top).fillna(0.0)
    else:
        pos = _rankw(sig.shift(1))
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=1)
    art.validate()
    return art.aligned()


def test_rotating_rank_book_with_region_holidays_passes():
    # honest guard: the deficit statistic is armed (expected >= the floor)
    # and does not fire - observed tracks expected under a real calendar
    r = _missing(_rotating_region_book())
    assert r.status is Status.PASS
    assert r.details["n_closure_uncorroborated_cells"] == 0
    assert (r.details["nonheld_coclosure_expected"]
            >= NONHELD_COCLOSURE_MIN_EXPECTED)
    obs = r.details["nonheld_coclosure_observed"]
    exp = r.details["nonheld_coclosure_expected"]
    assert obs > 0.5 * exp                       # tracks, no deficit
    assert r.details["corroboration_guards_fired"] == []


def test_subset_held_book_with_region_holidays_passes():
    # A top-eight-of-25 long book holds less than one third of the panel. Regional
    # closures mostly affect non-held names and must corroborate held-name gaps.
    r = _missing(_rotating_region_book(held_top=8))
    assert r.status is Status.PASS
    assert r.details["n_closure_uncorroborated_cells"] == 0
    assert r.details["corroboration_guards_fired"] == []


# ===========================================================================
# 3. bridging exogeneity (the load-bearing statistic) - deletion destroys
#    cumulative PnL, an honest halt (2020-03 confound) conserves it
# ===========================================================================

def _halt_book(seed=2, conserve=True, n=800, m=21, n_crash=26, group=9):
    """Net-long constant book; a fixed 9-name group goes NaN on 26 market
    crash days (-4.5% factor moves). conserve=True compounds the missing
    move into the reopen bar (an honest halt: cumulative PnL conserved);
    conserve=False just deletes it (the crash never happened)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    beta = rng.uniform(0.8, 1.2, m)
    mkt = rng.normal(4e-4, 0.009, n)
    crash: list[int] = []
    for d in rng.permutation(np.arange(5, n - 5)):
        if all(abs(int(d) - c) > 2 for c in crash):
            crash.append(int(d))
        if len(crash) >= n_crash:
            break
    crash = sorted(crash)
    for d in crash:
        mkt[d] -= 0.045
    rets_true = (mkt[:, None] * beta[None, :]
                 + rng.standard_normal((n, m)) * 0.011)
    sab = rets_true.copy()
    for d in crash:
        if conserve:
            sab[d + 1, :group] = (1 + rets_true[d, :group]) * (
                1 + rets_true[d + 1, :group]) - 1
        sab[d, :group] = np.nan
    cols = [f"A{i:02d}" for i in range(m)]
    rets = pd.DataFrame(sab, index=dates, columns=cols)
    sig = rets.rolling(10, min_periods=5).mean()
    pos = pd.DataFrame(1.0 / m, index=dates, columns=cols)
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=1)
    art.validate()
    return art.aligned(), _gross_sr(pos, rets)


@pytest.mark.parametrize("seed", [0, 2])
def test_crash_day_group_deletion_fires_bridging_exogeneity(seed):
    # attack: a fixed sibling group (fully corroborated, below
    # the market-wide line, all names held so the deficit cap is vacuous)
    # deleted exactly on crash days - only the bridging statistic can see
    # that the implied losses are never realized in the reopen bars
    art_del, sr_del = _halt_book(seed=seed, conserve=False)
    art_con, sr_con = _halt_book(seed=seed, conserve=True)
    assert sr_del > sr_con + 0.3                 # deletion inflates SR
    r = _missing(art_del)
    assert r.status is Status.WARN
    assert "bridging_exogeneity" in r.details["corroboration_guards_fired"]
    assert r.details["exog_bridging_z"] <= -EXOG_Z_CRIT
    assert r.details["exog_destroyed_sr"] >= EXOG_MIN_DESTROYED_SR
    assert r.details["frac_uncorroborated_closure"] > MISSING_RETURN_FRAC
    assert "bridging" in r.message
    assert "conserves" in r.message


@pytest.mark.parametrize("seed", [0, 2])
def test_conserving_halts_on_crash_days_pass(seed):
    # The honest confound guard (2020-03 circuit-breaker shape): the same
    # group closes on the same crash days, but the missing move lands in
    # the compounded reopen return - cumulative PnL conserved, so the
    # statistic tests clean and the closures stay excluded
    art, _ = _halt_book(seed=seed, conserve=True)
    r = _missing(art)
    assert r.status is Status.PASS
    assert r.details["exog_testable"] is True
    assert r.details["exog_n_runs_tested"] > 100
    assert r.details["exog_bridging_z"] > -EXOG_Z_CRIT
    assert r.details["exog_destroyed_sr"] < EXOG_MIN_DESTROYED_SR
    assert r.details["n_closure_uncorroborated_cells"] == 0
    assert r.details["corroboration_guards_fired"] == []


# ===========================================================================
# 4. fat-tail dilution: realistic crash-day markets align per-name
#    deletions into a fabricated calendar
# ===========================================================================

def _fattail_deletion_book(seed, n=800, m=25, block=21, n_crash=26):
    """Per-name worst-PnL deletion in a market with crash days. Common crashes align
    deletions across names, so affected siblings alone cannot corroborate the missing
    returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(m)]
    beta = rng.uniform(0.7, 1.3, m)
    mkt = rng.normal(2e-4, 0.010, n)
    cd = rng.choice(np.arange(5, n - 5), size=n_crash, replace=False)
    mkt[cd] -= 0.05
    rets = pd.DataFrame(rng.standard_normal((n, m)) * 0.02
                        + np.outer(mkt, beta), index=dates, columns=cols)
    sig = _zscore_cs(rets.rolling(20, min_periods=20).mean())
    pos = _rankw(sig.shift(1))
    pnl_cell = pos * rets
    sab = rets.copy()
    for j, c in enumerate(cols):
        for blk in range(2, n // block - 1):
            lo, hi = blk * block, (blk + 1) * block
            sab.loc[pnl_cell.iloc[lo:hi, j].idxmin(), c] = np.nan
    art = BacktestArtifacts(signals=sig, asset_returns=sab, positions=pos,
                            signal_lag=1)
    art.validate()
    return art.aligned()


@pytest.mark.parametrize("seed", [0, 1])
def test_fattail_partial_corroboration_stripped(seed):
    # The non-held co-closure deficit must revoke fabricated sibling corroboration so
    # all deleted mass is counted.
    r = _missing(_fattail_deletion_book(seed))
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["corroboration_guards_fired"]           # some guard
    assert r.details["n_closure_corroborated_cells"] == 0    # fully revoked
    assert r.details["frac_uncorroborated_closure"] > 0.04   # not diluted
    assert r.details["n_guard_stripped_cells"] > 300


# ===========================================================================
# 5. layer interaction: honest region books + market-wide holidays coexist
# ===========================================================================

def test_mixed_region_and_recurring_marketwide_book_passes():
    # honest structure crossing: union-calendar region holidays (sibling-
    # corroborated) plus recurring full-row holidays (recurrence-
    # corroborated) in one panel - both layers must corroborate their own
    # mass with no cross-contamination
    art, n_hol = _marketwide_holiday_book(n=1008, seed=21)
    rets = art.asset_returns.copy()
    cols = list(rets.columns)
    rng = np.random.default_rng(9)
    groups = [cols[0:8], cols[8:16], cols[16:25]]
    ok_rows = np.flatnonzero(rets.notna().all(axis=1).to_numpy())
    starts = rng.choice(ok_rows[(ok_rows > 10) & (ok_rows < len(rets) - 11)],
                        size=3 * 10, replace=False)
    for gi, g in enumerate(groups):
        for s in starts[gi * 10:(gi + 1) * 10]:
            rets.iloc[s:s + 1, [cols.index(c) for c in g]] = np.nan
    sig = rets.rolling(10, min_periods=5).mean()
    pos = pd.DataFrame(1.0 / len(cols), index=rets.index, columns=cols)
    art2 = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                             signal_lag=1)
    art2.validate()
    r = _missing(art2.aligned())
    assert r.status is Status.PASS
    assert r.details["n_marketwide_closure_dates"] >= n_hol
    assert r.details["n_closure_uncorroborated_cells"] == 0


def test_marketwide_threshold_is_a_band():
    # calibration invariant pinned as a band: below ~0.7 honest large-
    # region books (e.g. a 2/3-US global panel) would need recurrence for
    # ordinary regional holidays; above ~0.9 an attacker spares a name or
    # two per date to stay under the line at negligible laundering cost
    # (24/25 held names deleted = 96%)
    assert 0.7 <= MARKETWIDE_MIN_CLOSED_FRAC <= 0.9
