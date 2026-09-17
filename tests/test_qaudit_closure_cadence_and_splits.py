"""Closure corroboration, annual holdover cadence, and split-adjusted price rounding.

Shared holiday calendars supply evidence for short gaps. Annual and
overlapping-tranche books retain bounded holdover grace. Split-adjusted
ticks remain recognizable without waiving scale or timing discrepancies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import costs, survivorship
from qaudit.checks.survivorship import (CLOSURE_MIN_SHARED_DATES,
                                        HOLDOVER_CADENCE_SLACK_BARS,
                                        MISSING_RETURN_FRAC)
from qaudit.config import AuditConfig
from qaudit.errors import InputValidationError
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import positions_from_signals, simulate_market
from qaudit.types import Severity, Status

CFG = AuditConfig()

S_MISSING = "survivorship.positions_on_missing_returns"
S_TOU = "survivorship.trading_outside_universe"
C_PRICES = "costs.price_return_consistency"


def _by(results):
    return {r.check: r for r in results}


def _zscore_cs(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def _rankw(df):
    r = df.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    g = w.abs().sum(axis=1)
    return w.div(g.replace(0.0, np.nan), axis=0).fillna(0.0)


# ===========================================================================
# 1. positions_on_missing_returns: closure-exclusion laundering
# ===========================================================================

def _deletion_book(seed, n=800, m=25, block=21, mode="pnl"):
    """Momentum book with each name's worst position-PnL or largest absolute-return day per
    block replaced by an isolated interior NaN."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(m)]
    beta = rng.uniform(0.7, 1.3, m)
    mkt = rng.normal(2e-4, 0.010, n)
    rets = pd.DataFrame(rng.standard_normal((n, m)) * 0.02
                        + np.outer(mkt, beta), index=dates, columns=cols)
    ma = rets.rolling(20, min_periods=20).mean()
    sig = _zscore_cs(ma)
    pos = _rankw(sig.shift(1))
    pnl_cell = pos * rets
    sab = rets.copy()
    for j, c in enumerate(cols):
        for blk in range(2, n // block - 1):
            lo, hi = blk * block, (blk + 1) * block
            sl = (pnl_cell.iloc[lo:hi, j] if mode == "pnl"
                  else -rets.iloc[lo:hi, j].abs())
            sab.loc[sl.idxmin(), c] = np.nan
    strat = (pos * sab.fillna(0.0)).sum(axis=1)      # silent PnL zeroing
    art = BacktestArtifacts(signals=sig, asset_returns=sab, positions=pos,
                            strategy_returns=strat, signal_lag=1)
    art.validate()
    return art.aligned()


def _region_book(seed=101, n=300, holidays_per_region=14,
                 names_per_region=4, n_regions=3):
    """A union-calendar book whose regional assets share missing returns on local holidays."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    m = names_per_region * n_regions
    cols = [f"R{r}N{i}" for r in range(n_regions)
            for i in range(names_per_region)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, m)),
                        index=dates, columns=cols)
    starts = rng.choice(np.arange(10, n - 11),
                        size=n_regions * holidays_per_region, replace=False)
    for r_idx in range(n_regions):
        reg = [c for c in cols if c.startswith(f"R{r_idx}N")]
        for s in starts[r_idx * holidays_per_region:
                        (r_idx + 1) * holidays_per_region]:
            rets.loc[rets.index[s:s + 1], reg] = np.nan
    sig = rets.rolling(5, min_periods=5).mean()
    pos = pd.DataFrame(1.0 / m, index=dates, columns=cols)
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=1)
    art.validate()
    return art.aligned()


def _missing(art):
    return _by(survivorship.run(art, CFG))[S_MISSING]


@pytest.mark.parametrize("seed", [0, 1])
def test_worst_day_deletion_high_mass_warns(seed):
    # Deleted losing observations must leave enough uncorroborated closure mass to
    # warn.
    r = _missing(_deletion_book(seed, block=21))
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["frac_uncorroborated_closure"] > MISSING_RETURN_FRAC
    assert "co-closure" in r.message or "corroborat" in r.message


@pytest.mark.parametrize("seed", [0, 1])
def test_worst_day_deletion_moderated_band_warns(seed):
    # The low-mass deletion case remains above the 1% uncorroborated-gap tolerance and
    # must warn.
    r = _missing(_deletion_book(seed, block=63))
    assert r.status is Status.WARN
    assert r.details["frac_uncorroborated_closure"] > MISSING_RETURN_FRAC


def test_badprint_symmetric_deletion_warns():
    # Filtering extreme absolute-return days also removes PnL from held cells and
    # requires an uncorroborated-gap warning.
    r = _missing(_deletion_book(0, block=21, mode="badprint"))
    assert r.status is Status.WARN
    assert r.details["frac_uncorroborated_closure"] > MISSING_RETURN_FRAC


def test_union_calendar_book_still_passes_fully_corroborated():
    # Each holiday is shared by four region siblings and should be corroborated.
    r = _missing(_region_book(seed=101))
    assert r.status is Status.PASS
    assert r.details["n_closure_uncorroborated_cells"] == 0
    assert r.details["n_closure_corroborated_cells"] > 100
    assert "closure" in r.message or "market-holiday" in r.message


def test_two_name_region_book_passes():
    # honest-structure guard at the sibling floor: regions of just two
    # names still corroborate each other (14 shared dates >= the 3 floor)
    r = _missing(_region_book(seed=8, names_per_region=2, n_regions=3))
    assert r.status is Status.PASS
    assert r.details["n_closure_uncorroborated_cells"] == 0


def test_three_holiday_calendar_still_corroborates():
    # sibling floor pinned from below: a short 3-holiday calendar is
    # exactly CLOSURE_MIN_SHARED_DATES shared dates - must still count
    assert CLOSURE_MIN_SHARED_DATES == 3
    r = _missing(_region_book(seed=11, holidays_per_region=3))
    assert r.status is Status.PASS
    assert r.details["n_closure_uncorroborated_cells"] == 0


def test_single_name_regions_get_scoped_warn_not_accusation():
    # the irreducible ambiguous class: one name per market has no calendar
    # sibling by construction. Verdict scoping: WARN with an honest
    # cannot-separate message, never the "delisting loss" conviction and
    # never a silent PASS (a PASS here is exactly the laundering channel).
    r = _missing(_region_book(seed=7, names_per_region=1, n_regions=6))
    assert r.status is Status.WARN
    assert r.details["n_closure_corroborated_cells"] == 0
    assert "cannot tell" in r.message
    assert "delisting loss" not in r.message
    assert r.remediation and "calendar" in r.remediation


# ===========================================================================
# 2. trading_outside_universe: annual-rebalance cadence
# ===========================================================================

def _annual_book(seed=0, n_assets=30, n_top=20, n=1060, cadence=252,
                 zombie_col=None):
    """Honest annual-reconstitution book (Jegadeesh-Titman style): PIT
    top-K membership recomputed daily, weights from signals.loc[t-1] over
    universe.loc[t-1] at each ``cadence``-bar rebalance, passively drifted,
    exits unwound at the next scheduled rebalance. ``zombie_col`` pins one
    column held forever after an early forced exit (hindsight book)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(2e-4, 0.015, (n, n_assets)),
                        index=dates, columns=cols)
    cap = 100.0 * (1.0 + rets).cumprod() * rng.uniform(0.5, 2.0, n_assets)
    uni = cap.rank(axis=1, ascending=False) <= n_top
    sig = _zscore_cs(rets.rolling(252, min_periods=252).mean())
    pos = pd.DataFrame(0.0, index=dates, columns=cols)
    current = np.zeros(n_assets)
    reb = set(range(253, n, cadence))
    for t in range(253, n):
        if t in reb:
            s = sig.iloc[t - 1]
            elig = uni.iloc[t - 1] & s.notna()
            w = s.where(elig).rank()
            w = w - w.mean()
            g = w.abs().sum()
            current = ((w / g).fillna(0.0).to_numpy() if g > 0
                       else np.zeros(n_assets))
        elif t > 253:
            rr = rets.iloc[t - 1].to_numpy()
            current = current * (1.0 + rr) / (1.0 + current @ rr)
        pos.iloc[t] = current
    if zombie_col is not None:
        uni.iloc[300:, zombie_col] = False        # exits for good...
        # ...but the position established while legal is passively held even
        # through later annual rebalances. Recompute sequentially so this
        # column is exactly its self-financing drift, not a constant-target
        # top-up disguised as a hold.
        pos.iloc[253, zombie_col] = 0.05
        for t in range(254, n):
            prev = pos.iloc[t - 1].to_numpy()
            rr = rets.iloc[t - 1].to_numpy()
            drift = prev * (1.0 + rr) / (1.0 + prev @ rr)
            if t in reb:
                pos.iloc[t, zombie_col] = drift[zombie_col]
            else:
                pos.iloc[t] = drift
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            universe=uni, signal_lag=1)
    art.validate()
    return art.aligned()


def _tou(art):
    return _by(survivorship.run(art, CFG))[S_TOU]


@pytest.mark.parametrize("seed", [0, 1])
def test_honest_annual_reconstitution_book_passes(seed):
    # An annual reconstitution cadence must fit within the bounded holdover grace.
    r = _tou(_annual_book(seed=seed))
    assert r.status is Status.PASS, r.message
    assert r.details["rebalance_cadence_bars"] == 252
    assert (r.details["holdover_grace_bars"]
            == 252 + HOLDOVER_CADENCE_SLACK_BARS)
    assert r.details["n_active_cells"] == 0


def test_annual_cadence_zombie_still_fails_critical():
    # Holding an exited name through multiple annual rebalances exceeds grace. The
    # message must report the observed cadence rather than claim there was no response.
    r = _tou(_annual_book(seed=0, zombie_col=0))
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_zombie_cells"] > 100
    assert "252-bar rebalance cadence" in r.message
    assert "no rebalance response" not in r.message


# ===========================================================================
# 3. trading_outside_universe: overlapping-tranche roll-off
# ===========================================================================

def _tranche_book(seed, n_tranches=5):
    """Standard overlapping-tranche book from the library's own helpers:
    average of positions_from_signals(lag=k), k=1..n_tranches, on
    simulate_market with 15% deaths. Newest information used is t-1."""
    market = simulate_market(seed=seed)
    rets = market["returns"]
    sig = _zscore_cs(rets.rolling(20, min_periods=20).mean())
    books = [positions_from_signals(sig, lag=k)
             for k in range(1, n_tranches + 1)]
    pos = sum(books) / n_tranches
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            universe=market["universe"], signal_lag=1)
    art.validate()
    return art.aligned()


@pytest.mark.parametrize("seed", [0, 2])
def test_overlapping_tranche_book_passes(seed):
    # Rank drift during tranche roll-off can increase a held weight on NaN-return bars
    # without producing earnable PnL.
    r = _tou(_tranche_book(seed))
    assert r.status is Status.PASS, r.message
    assert r.details["n_active_cells"] == 0
    assert r.details["n_deadbar_fresh_cells"] >= 1   # the wiggles are there


def test_finite_return_entry_while_out_still_fails():
    # detection guard: the carve-out is NaN-return-only - a fresh entry
    # that can earn PnL on an out-of-universe name convicts immediately
    art = _tranche_book(0)
    pos = art.positions.copy()
    uni = art.universe
    rets = art.asset_returns
    # pick a name alive to the end (finite returns) and force it out of
    # the universe from bar 500 while entering fresh exposure at bar 520
    alive = [c for c in rets.columns if rets[c].notna().all()]
    c = alive[0]
    uni = uni.copy()
    uni.loc[uni.index[500:], c] = False
    pos.loc[pos.index[520:], c] = 0.30
    art2 = BacktestArtifacts(signals=art.signals, asset_returns=rets,
                             positions=pos, universe=uni, signal_lag=1)
    art2.validate()
    r = _tou(art2.aligned())
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_active_cells"] >= 1
    assert "hindsight" in r.message


def test_nan_return_increases_past_grace_still_convict_as_zombie():
    # detection guard on the carve-out itself: reclassified dead-bar cells
    # stay in the holdover class, so bars_out keeps accruing and a NaN-
    # return name ratcheted upward far past grace is still a zombie FAIL.
    rng = np.random.default_rng(5)
    n, m = 400, 12
    dates = pd.bdate_range("2020-01-02", periods=n)
    cols = [f"A{i:02d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, m)),
                        index=dates, columns=cols)
    uni = pd.DataFrame(True, index=dates, columns=cols)
    uni.iloc[100:, 0] = False
    rets.iloc[100:, 0] = np.nan                   # delisted: NaN returns
    pos = pd.DataFrame(1.0 / m, index=dates, columns=cols)
    ramp = 1.0 / m + 1e-4 * np.arange(1, n - 100 + 1)
    pos.iloc[100:, 0] = ramp                      # increases every bar
    sig = rets.rolling(5, min_periods=5).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            universe=uni, signal_lag=1)
    art.validate()
    r = _tou(art.aligned())
    assert r.status is Status.FAIL
    assert r.details["n_active_cells"] == 0       # carved out (NaN returns)
    assert r.details["n_zombie_cells"] > 100      # ...but never sheltered


# ===========================================================================
# 4. price_return_consistency: per-asset tick waiver
# ===========================================================================

def _split_panel(seed, factor=2.0, split=True, n=800, m=20):
    """Honest vendor layout: raw closes cent-rounded, asset_returns from an
    independent full-precision total-return source. With ``split``, asset 0
    had a ``factor``:1 split and the vendor split-adjusts its pre-split
    rows (rounded raw / factor -> off the cent grid)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(m)]
    p0 = rng.uniform(5.0, 15.0, m)
    rets_true = rng.standard_normal((n, m)) * 0.015 + 3e-4
    pstar = p0 * np.cumprod(1.0 + rets_true, axis=0)
    rets = pd.DataFrame(rets_true, index=dates, columns=cols)
    k = int(0.4 * n)
    raw = pstar.copy()
    if split:
        raw[:k, 0] = pstar[:k, 0] * factor
    stored = np.round(raw * 100.0) / 100.0
    if split:
        stored[:k, 0] = stored[:k, 0] / factor
    px = pd.DataFrame(stored, index=dates, columns=cols)
    return px, rets


def _price_check(px, rets):
    sig = rets.rolling(20, min_periods=20).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=rets, prices=px,
                            signal_lag=1)
    art.validate()
    return _by(costs.run(art.aligned(), CFG))[C_PRICES]


@pytest.mark.parametrize("seed", [0, 1])
def test_one_split_adjusted_asset_does_not_disarm_waiver(seed):
    # A split-adjusted asset may leave the cent grid; it must not invalidate the
    # rounding tolerance for other low-price assets.
    px, rets = _split_panel(seed, factor=2.0)
    r = _price_check(px, rets)
    assert r.status is Status.PASS, r.message
    assert r.details["on_cent_grid"] is False  # Split-adjusted prices need not lie on the cent grid.
    assert r.details["rounding_shaped"] is True
    assert r.details["frac_cells_quantized"] > 0.99
    assert r.details["median_tol"] > 2e-4         # lifted, not the bare 1bp
    assert "tick" in r.message


def test_three_for_two_split_factor_also_recognized():
    # 3:2 splits put pre-split rows on a two-thirds-cent grid - the factor
    # inference must cover non-integer ratios, not just halves
    px, rets = _split_panel(3, factor=1.5)
    r = _price_check(px, rets)
    assert r.status is Status.PASS, r.message
    assert r.details["rounding_shaped"] is True


def test_split_panel_date_shift_still_warns():
    # detection guard: the same treatment panel with a real 1-day shift
    # medians at daily-vol scale - the per-asset waiver must not eat it
    px, rets = _split_panel(0, factor=2.0)
    r = _price_check(px, rets.shift(1))
    assert r.status is Status.WARN
    assert r.details["median_abs_diff"] > 5 * r.details["tick_median_scale"]


def test_split_panel_percent_scale_still_warns():
    # Percent-scaled returns below -100% fail intake. Direct price-check calls must
    # still warn about scaling, including panels whose scaled returns stay above the
    # intake floor.
    px, rets = _split_panel(0, factor=2.0)
    with pytest.raises(InputValidationError, match="percent units"):
        _price_check(px, rets * 100.0)
    sig = rets.rolling(20, min_periods=20).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=rets * 100.0,
                            prices=px, signal_lag=1)
    r = _by(costs.run(art.aligned(), CFG))[C_PRICES]
    assert r.status is Status.WARN


def test_coarser_than_tick_storage_still_warns():
    # detection guard on the factor list's f >= 1 bound: dime-quantized
    # prices sit on the cent grid too, but their errors blow the per-cell
    # cent bound - coarser-than-tick storage must stay a defect, never
    # infer a coarser quantum to excuse it
    px, rets = _split_panel(2, split=False)
    px = (px * 10.0).round(0) / 10.0              # $0.10 storage rounding
    r = _price_check(px, rets)
    assert r.status is Status.WARN
    assert r.details["rounding_shaped"] is False
