"""Interior universe gaps and implausibly brief membership exits.

Scattered missing membership must not fabricate attrition in a survivor-only
panel. The interior-NaN mass gate and calendar-scaled flicker checks preserve
legitimate listing boundaries, long exits, and coarse-frequency reconstitution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit import (AuditFailure, BacktestArtifacts, InputValidationError,
                    Severity, Status, audit)
from qaudit.checks.survivorship import NO_EXITS_FLICKER_MAX_OUT_DAILY_BARS
from qaudit.inputs import _UNIVERSE_MAX_INTERIOR_NAN_FRAC

S_NOEX = "survivorship.no_exits"
S_TOU = "survivorship.trading_outside_universe"
S_MISS = "survivorship.positions_on_missing_returns"


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _book(n=756, m=12, seed=7, start="2020-01-06"):
    """A near-static book rebalanced every five bars."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    cols = [f"A{i:02d}" for i in range(m)]
    # These fixtures isolate membership stamping. Zero dispersion makes the
    # repeated weights genuine holds, so trading_outside_universe is not
    # accidentally testing constant-target rebalancing as a second defect.
    rets = pd.DataFrame(0.0, idx, cols)
    sigs = pd.DataFrame(rng.normal(0, 1, (n, m)), idx, cols)
    pos = pd.DataFrame(1.0 / m, idx, cols)
    pos.iloc[::5] *= 1.02
    return idx, cols, rets, sigs, pos


def _stealth_universe(idx, cols, cells_per_col_step, first=32, stagger=7):
    """A static all-True universe with single-bar NaNs outside rebalance decision bars,
    where the following position bar has no increase."""
    n, m = len(idx), len(cols)
    uni = pd.DataFrame(1.0, index=idx, columns=cols)
    placed = 0
    for j in range(m):
        for t in range(first + stagger * j, n - 2, cells_per_col_step):
            tt = t
            while (tt + 1) % 5 == 0:
                tt += 1
            uni.iloc[tt, j] = np.nan
            placed += 1
    return uni, placed


def _art(sigs, rets, pos, uni, **kw):
    return BacktestArtifacts(signals=sigs, asset_returns=rets, positions=pos,
                             universe=uni, signal_lag=1, **kw)


# ===========================================================================
# 1. attack: off-bar single-cell NaN stealth (above the mass floor) is rejected
# ===========================================================================

def test_stealth_repro_rejected_by_interior_nan_mass_gate():
    # Single-bar missing membership off rebalance dates can fabricate exits without
    # triggering active trading; the membership-gap checks must expose it.
    idx, cols, rets, sigs, pos = _book()
    uni, placed = _stealth_universe(idx, cols, 60)
    assert placed / uni.size > _UNIVERSE_MAX_INTERIOR_NAN_FRAC
    art = _art(sigs, rets, pos, uni)
    with pytest.raises(InputValidationError,
                       match="resolve NaN membership explicitly") as ei:
        art.validate()
    msg = str(ei.value)
    assert "ffill" in msg and "False deliberately" in msg
    assert "no_exits" in msg          # names the check being laundered


def test_flicker_window_dodging_runs_rejected_by_mass_gate():
    # evasion probe: 11-bar interior NaN runs stay out longer than the
    # flicker window (so no_exits would call them plausible) but cost
    # 11 cells per fabricated exit x 12 names = 1.45% > the 1% floor
    idx, cols, rets, sigs, pos = _book()
    rng = np.random.default_rng(3)
    uni = pd.DataFrame(1.0, index=idx, columns=cols)
    for j in range(len(cols)):
        t = int(rng.integers(30, len(idx) - 45))
        uni.iloc[t:t + 11, j] = np.nan
    with pytest.raises(InputValidationError,
                       match="resolve NaN membership explicitly"):
        _art(sigs, rets, pos, uni).validate()


def test_monthly_single_bar_fabrication_rejected_by_mass_gate():
    # at monthly bars the flicker prong stands down (window floors to 0),
    # so the mass gate must be the binding layer: one NaN cell per asset
    # on a 36-bar book is 2.8% of cells
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-31", periods=36, freq=pd.offsets.BMonthEnd())
    cols = [f"A{i:02d}" for i in range(12)]
    rets = pd.DataFrame(rng.normal(0.008, 0.05, (36, 12)), idx, cols)
    sigs = pd.DataFrame(rng.normal(0, 1, (36, 12)), idx, cols)
    pos = pd.DataFrame(1.0 / 12, idx, cols)
    uni = pd.DataFrame(1.0, idx, cols)
    for j in range(12):
        uni.iloc[int(rng.integers(3, 33)), j] = np.nan
    art = _art(sigs, rets, pos, uni, periods_per_year=12)
    with pytest.raises(InputValidationError,
                       match="resolve NaN membership explicitly"):
        art.validate()


# ===========================================================================
# 2. attack: sub-floor stealth is flagged by the flicker prong
# ===========================================================================

def _subfloor_attack(seed=7):
    idx, cols, rets, sigs, pos = _book(seed=seed)
    uni, placed = _stealth_universe(idx, cols, 200, first=40, stagger=11)
    assert placed / uni.size < _UNIVERSE_MAX_INTERIOR_NAN_FRAC
    return _art(sigs, rets, pos, uni), placed


def test_subfloor_stealth_flagged_high_by_flicker_prong():
    # ~0.5% scattered single-bar NaN: legal at the input (below the mass
    # floor, per the pinned NaN-universe contract) but every fabricated
    # exit re-enters next bar - no_exits must WARN HIGH, not PASS
    art, placed = _subfloor_attack()
    rep = audit(art, include=["survivorship"])
    r = rep[S_NOEX]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert "flicker" in r.message
    assert "fabricated membership" in r.message
    assert r.details["n_exiting_assets"] == 0            # plausible exits
    assert r.details["n_exiting_assets_raw"] == 12       # fabricated look
    assert r.details["n_flicker_exit_events"] == placed
    assert r.details["n_flicker_exit_events"] == r.details["n_exit_events"]
    # the rest of the family is green (that is the stealth: only the flicker
    # prong stands between this book and a clean survivorship certificate)
    assert rep[S_TOU].status is Status.PASS
    assert rep[S_MISS].status is Status.PASS
    # advisory contract: WARN never flips ok - the deployment gate blocks
    assert rep.ok
    with pytest.raises(AuditFailure, match="no_exits"):
        rep.gate(warn_severity=Severity.HIGH)


def test_attack_direction_monte_carlo():
    # n=20 randomized placements per band: above-floor variants must all
    # reject at validate(); sub-floor variants must all be flagged by the
    # flicker prong with zero plausible exits
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        idx, cols, rets, sigs, pos = _book(seed=seed)
        n, m = len(idx), len(cols)

        # above the floor: 110-150 scattered single-bar cells (1.2-1.7%)
        uni = pd.DataFrame(1.0, index=idx, columns=cols)
        n_cells = int(rng.integers(110, 151))
        for _ in range(n_cells):
            t = int(rng.integers(5, n - 5))
            while (t + 1) % 5 == 0:
                t += 1
            uni.iloc[t, int(rng.integers(0, m))] = np.nan
        with pytest.raises(InputValidationError,
                           match="resolve NaN membership"):
            _art(sigs, rets, pos, uni).validate()

        # below the floor: 3-5 cells per column (0.4-0.66%)
        uni2 = pd.DataFrame(1.0, index=idx, columns=cols)
        for j in range(m):
            for t in rng.choice(np.arange(5, n - 5), int(rng.integers(3, 6)),
                                replace=False):
                tt = int(t)
                while (tt + 1) % 5 == 0:
                    tt += 1
                uni2.iloc[tt, j] = np.nan
        rep = audit(_art(sigs, rets, pos, uni2), include=["survivorship"])
        r = rep[S_NOEX]
        assert r.status is Status.WARN, f"seed {seed}: not flagged"
        assert r.details["n_exiting_assets"] == 0
        assert r.details["n_flicker_exit_events"] > 0


@pytest.mark.parametrize("encoding", ["nan", "false"])
def test_sample_edge_terminal_fabrication_flagged(encoding):
    # adjacent evasion: one NaN (or explicit all-False) row at the sample
    # end fabricates a "terminal delisting" for every name at once -
    # trailing runs are exempt from the mass gate by design (honest
    # delist-tails), so the prong must refuse to call an exit terminal
    # when the sample observes less than a flicker window past it
    idx, cols, rets, sigs, pos = _book()
    uni = pd.DataFrame(1.0, index=idx, columns=cols)
    uni.iloc[-1, :] = np.nan if encoding == "nan" else 0.0
    rep = audit(_art(sigs, rets, pos, uni), include=["survivorship"])
    r = rep[S_NOEX]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["n_exiting_assets"] == 0
    assert r.details["n_unverifiable_exit_events"] == 12
    assert "sample end" in r.message
    with pytest.raises(AuditFailure, match="no_exits"):
        rep.gate(warn_severity=Severity.HIGH)


def test_terminal_exit_beyond_edge_window_stays_plausible():
    # honest guard for the edge rule: a genuine delisting 30 bars before
    # the sample end is comfortably verifiable - counted as attrition
    idx, cols, rets, sigs, pos = _book()
    n = len(idx)
    uni = pd.DataFrame(1.0, index=idx, columns=cols)
    uni.iloc[n - 30:, 0] = 0.0
    rets, pos = rets.copy(), pos.copy()
    rets.iloc[n - 30:, 0] = np.nan
    pos.iloc[n - 30:, 0] = 0.0
    rep = audit(_art(sigs, rets, pos, uni), include=["survivorship"])
    r = rep[S_NOEX]
    assert r.status is Status.PASS
    assert r.details["n_exiting_assets"] == 1
    assert r.details["n_unverifiable_exit_events"] == 0


# ===========================================================================
# 3. honest guards - layouts the two layers must not flag
# ===========================================================================

def _honest_book(seed, holes_per_col=2):
    """Genuine attrition (2-3 terminal deaths, positions unwound, returns
    NaN after death) plus sub-floor vendor NaN holes placed on rebalance
    bars (the honest counterpart of the attack's off-bar placement)."""
    rng = np.random.default_rng(seed)
    idx, cols, rets, sigs, pos = _book(seed=seed)
    n, m = len(idx), len(cols)
    rets, pos = rets.copy(), pos.copy()
    uni = pd.DataFrame(1.0, index=idx, columns=cols)
    n_deaths = int(rng.integers(2, 4))
    for j in rng.choice(m, n_deaths, replace=False):
        death = int(rng.integers(int(0.3 * n), int(0.9 * n)))
        uni.iloc[death:, j] = 0.0
        rets.iloc[death:, j] = np.nan
        pos.iloc[death:, j] = 0.0
    for j in range(m):
        for _ in range(holes_per_col):
            t = int(rng.integers(2, n - 2))
            t -= t % 5                    # snap to a rebalance bar
            if t >= 2 and np.isfinite(uni.iloc[t, j]) and uni.iloc[t, j] == 1.0:
                uni.iloc[t, j] = np.nan
    return _art(sigs, rets, pos, uni), n_deaths


def test_honest_direction_monte_carlo():
    # n=20: genuine terminal attrition + sub-floor holes on rebalance bars
    # must validate and keep the whole survivorship family green - the
    # holes only flicker; the real deaths carry the rate
    for seed in range(20):
        art, n_deaths = _honest_book(seed)
        rep = audit(art, include=["survivorship"])
        r = rep[S_NOEX]
        assert r.status is Status.PASS, f"seed {seed}: {r.message[:120]}"
        assert r.details["n_exiting_assets"] == n_deaths
        assert rep[S_TOU].status is Status.PASS
        assert rep[S_MISS].status is Status.PASS
        assert rep.ok
        rep.gate(warn_severity=Severity.HIGH)   # deployment-green too


def test_pure_terminal_delisting_universe_unchanged():
    # the classic honest layout: deaths only, no NaN anywhere - the prong
    # must count every terminal exit as plausible, no flicker wording
    art, n_deaths = _honest_book(seed=42, holes_per_col=0)
    rep = audit(art, include=["survivorship"])
    r = rep[S_NOEX]
    assert r.status is Status.PASS
    assert r.details["n_flicker_exit_events"] == 0
    assert "flicker" not in r.message


def test_long_false_out_spell_is_plausible_attrition():
    # a name honestly out (declared False, not NaN) for 6 months then
    # re-admitted: gap 126 bars >> the flicker window - plausible exit
    idx, cols, rets, sigs, pos = _book()
    uni = pd.DataFrame(1.0, index=idx, columns=cols)
    uni.iloc[200:326, 0] = 0.0
    pos = pos.copy()
    pos.iloc[200:326, 0] = 0.0            # book respects the exit
    rep = audit(_art(sigs, rets, pos, uni), include=["survivorship"])
    r = rep[S_NOEX]
    assert r.status is Status.PASS
    assert r.details["n_exiting_assets"] == 1
    assert r.details["n_flicker_exit_events"] == 0


def test_monthly_one_bar_out_spell_stays_plausible():
    # coarse bars: one monthly bar out = a month of calendar time, honest
    # reconstitution churn - the prong stands down (window floors to 0)
    rng = np.random.default_rng(11)
    idx = pd.date_range("2018-01-31", periods=60, freq=pd.offsets.BMonthEnd())
    cols = [f"A{i:02d}" for i in range(12)]
    rets = pd.DataFrame(rng.normal(0.008, 0.05, (60, 12)), idx, cols)
    sigs = pd.DataFrame(rng.normal(0, 1, (60, 12)), idx, cols)
    pos = pd.DataFrame(1.0 / 12, idx, cols)
    uni = pd.DataFrame(1.0, idx, cols)
    uni.iloc[30, 0] = 0.0                 # out one month, back the next
    uni.iloc[40:, 1] = 0.0                # plus one terminal death
    art = _art(sigs, rets, pos, uni, periods_per_year=12)
    rep = audit(art, include=["survivorship"])
    r = rep[S_NOEX]
    assert r.status is Status.PASS
    assert r.details["flicker_max_out_bars"] == 0
    assert r.details["n_exiting_assets"] == 2


def test_weekly_window_boundary():
    # ppy=52 -> window floor(10*52/252) = 2 bars: a 2-bar out-spell is
    # flicker, a 3-bar out-spell is plausible
    assert int(np.floor(NO_EXITS_FLICKER_MAX_OUT_DAILY_BARS * 52 / 252)) == 2
    rng = np.random.default_rng(13)
    idx = pd.date_range("2019-01-04", periods=160, freq="W-FRI")
    cols = [f"A{i:02d}" for i in range(12)]
    rets = pd.DataFrame(rng.normal(0.002, 0.02, (160, 12)), idx, cols)
    sigs = pd.DataFrame(rng.normal(0, 1, (160, 12)), idx, cols)
    pos = pd.DataFrame(1.0 / 12, idx, cols)
    uni = pd.DataFrame(1.0, idx, cols)
    uni.iloc[50:52, 0] = 0.0              # 2 bars out -> flicker
    uni.iloc[80:83, 1] = 0.0              # 3 bars out -> plausible
    art = _art(sigs, rets, pos, uni, periods_per_year=52)
    rep = audit(art, include=["survivorship"])
    r = rep[S_NOEX]
    assert r.details["flicker_max_out_bars"] == 2
    assert r.details["n_flicker_exit_events"] == 1
    assert r.details["n_exiting_assets"] == 1
    assert r.status is Status.PASS        # the plausible exit carries it


# ===========================================================================
# 4. mass-gate boundary and exemptions
# ===========================================================================

def _uni_with_scattered(idx, cols, n_cells, seed=0):
    rng = np.random.default_rng(seed)
    uni = pd.DataFrame(1.0, index=idx, columns=cols)
    n, m = uni.shape
    slots = [(t, j) for j in range(m) for t in range(2, n - 2)]
    for k in rng.choice(len(slots), n_cells, replace=False):
        t, j = slots[int(k)]
        uni.iloc[t, j] = np.nan
    return uni


def test_mass_gate_floor_boundary():
    idx, cols, rets, sigs, pos = _book()
    total = len(idx) * len(cols)                          # 9072 cells
    below = int(np.floor(total * _UNIVERSE_MAX_INTERIOR_NAN_FRAC)) - 2
    above = int(np.ceil(total * _UNIVERSE_MAX_INTERIOR_NAN_FRAC)) + 10
    _art(sigs, rets, pos,
         _uni_with_scattered(idx, cols, below)).validate()      # 0.98% legal
    with pytest.raises(InputValidationError, match="resolve NaN membership"):
        _art(sigs, rets, pos,
             _uni_with_scattered(idx, cols, above)).validate()  # 1.11% rejected


def test_leading_and_trailing_nan_runs_exempt():
    # pre-listing and post-delisting NaN runs decode to False correctly -
    # 3.3%+3.3% of cells in such runs must not trip the interior gate
    idx, cols, rets, sigs, pos = _book()
    uni = pd.DataFrame(1.0, index=idx, columns=cols)
    uni.iloc[:300, 0] = np.nan            # lists late (leading run)
    uni.iloc[456:, 1] = np.nan            # delists (trailing run)
    art = _art(sigs, rets, pos, uni)
    art.validate()                        # must not raise
    aligned = art.aligned()
    assert not aligned.universe.iloc[0, 0]        # still decodes to False
    assert not aligned.universe.iloc[-1, 1]


def test_mass_gate_only_counts_grid_overlap_rows():
    # NaN on universe rows outside the signals x asset_returns grid is
    # dropped by aligned() and must not count toward the mass
    idx, cols, rets, sigs, pos = _book()
    ext = pd.bdate_range(idx[0] - np.timedelta64(400, "D"), periods=250)
    uni_ext = pd.DataFrame(1.0, index=ext.append(idx), columns=cols)
    uni_ext.iloc[10:200, :] = np.nan      # huge NaN mass, all off-grid
    _art(sigs, rets, pos, uni_ext).validate()     # must not raise
