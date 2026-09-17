"""Passive weight drift, delisting exposure, and member-year attrition.

Fresh out-of-universe exposure is measured against self-financing drift with
bounded numeric and fee tolerance. Missing terminal returns remain visible,
and growing universes use live member-years in their attrition denominator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.errors import AuditFailure
from qaudit._stats import drifted_weights, traded_dollars_series, turnover_series
from qaudit.checks import survivorship
from qaudit.checks.survivorship import (
    HOLD_INCREASE_REL_TOL, HOLD_INCREASE_TOL, HOLDOVER_GRACE_BARS,
    POSITION_EPS,
)
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.report import AuditReport
from qaudit.synthetic import make_clean, positions_from_signals
from qaudit.types import Status, Severity

CFG = AuditConfig()
S_TOU = "survivorship.trading_outside_universe"
S_MISS = "survivorship.positions_on_missing_returns"
S_EXIT = "survivorship.no_exits"


def _by(results):
    return {r.check: r for r in results}


def _run(art: BacktestArtifacts):
    art.validate()
    return _by(survivorship.run(art.aligned(), CFG))


# Passive drift and traded-dollar reference arithmetic.

def _reference_traded_dollars(positions, asset_returns):
    """Independent reference arithmetic for the traded-dollar ledger."""
    pos = positions.fillna(0.0)
    r = asset_returns.reindex(index=pos.index, columns=pos.columns).fillna(0.0)
    grow = (pos * (1.0 + r)).shift(1)
    denom = (1.0 + (pos * r).sum(axis=1)).shift(1)
    w_drift = grow.div(denom, axis=0)
    traded = (pos - w_drift).abs().sum(axis=1)
    if len(traded):
        traded.iloc[0] = pos.iloc[0].abs().sum()
        traded[denom <= 0] = np.nan
    return traded


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_traded_dollars_matches_reference_arithmetic(seed):
    a = make_clean(seed).artifacts
    new = traded_dollars_series(a.positions, a.asset_returns).to_numpy()
    ref = _reference_traded_dollars(a.positions, a.asset_returns).to_numpy()
    assert np.array_equal(new, ref, equal_nan=True)          # bit-identical
    to_new = turnover_series(a.positions, a.asset_returns).to_numpy()
    to_ref = (0.5 * _reference_traded_dollars(a.positions, a.asset_returns))
    to_ref.iloc[0] = np.nan
    assert np.array_equal(to_new, to_ref.to_numpy(), equal_nan=True)
    # NaN positions (flat) keep the same contract
    p = a.positions.copy()
    p.iloc[10:20, 0] = np.nan
    assert np.array_equal(
        traded_dollars_series(p, a.asset_returns).to_numpy(),
        _reference_traded_dollars(p, a.asset_returns).to_numpy(),
        equal_nan=True)


def test_drifted_weights_conventions():
    idx = pd.bdate_range("2024-01-01", periods=5)
    pos = pd.DataFrame({"X": [0.6, 0.6, 1.0, 1.0, 0.5],
                        "Y": [0.4, 0.4, 0.0, 0.0, 0.5]}, index=idx)
    r = pd.DataFrame({"X": [0.10, 0.0, -1.0, 0.0, 0.0],
                      "Y": [-0.05, np.nan, 0.0, 0.0, 0.0]}, index=idx)
    w = drifted_weights(pos, r)
    assert w.shape == (5, 2)
    assert np.array_equal(w[0], [0.0, 0.0])                   # no prior book
    port0 = 0.6 * 0.10 + 0.4 * (-0.05)
    assert w[1, 0] == pytest.approx(0.6 * 1.10 / (1 + port0))
    assert w[1, 1] == pytest.approx(0.4 * 0.95 / (1 + port0))
    assert np.array_equal(w[2], [0.6, 0.4])                    # NaN return -> 0
    assert np.isnan(w[3]).all()               # X wiped (-100%): NAV growth 0
    assert np.array_equal(w[4], [1.0, 0.0])                    # flat bar
    # the dollars ledger surfaces the wiped row as NaN, not a fabricated trade
    dollars = traded_dollars_series(pos, r)
    assert np.isnan(dollars.iloc[3]) and dollars.iloc[0] == 1.0
    # asset_returns are reindexed to positions: extra columns/rows ignored
    r2 = r.copy()
    r2["Z"] = 5.0
    assert np.array_equal(drifted_weights(pos, r2), w, equal_nan=True)


# Out-of-universe trading under passive weight drift.

N_T, N_A, EXIT = 400, 10, 100


def _panel(seed, vol=0.02):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_T)
    assets = [f"A{i:02d}" for i in range(N_A)]
    rets = pd.DataFrame(rng.normal(0.0003, vol, (N_T, N_A)), dates, assets)
    sig = pd.DataFrame(rng.normal(size=(N_T, N_A)), dates, assets)
    uni = pd.DataFrame(True, dates, assets)
    uni.iloc[EXIT:, 0] = False                 # A00 leaves at bar 100
    return rets, sig, uni


def _held_book(rets, uni, cadence, daily_fee=0.0):
    """Honest book recorded as actually-held weights: equal weight over the
    PIT universe (t-1) at each rebalance, passive drift in between; the
    dead name is dropped at the next scheduled rebalance. ``daily_fee`` =
    a per-bar management-fee / cost accrual the engine nets into the NAV
    it divides the held weights by (weights sit above the exact drift by
    ~fee relative every bar)."""
    r = rets.to_numpy()
    w = np.zeros((N_T, N_A))
    for t in range(N_T):
        if t % cadence == 0:
            legal = uni.iloc[max(t - 1, 0)].to_numpy()
            w[t] = legal / legal.sum()
        else:
            nav = 1 + w[t - 1] @ r[t - 1] - daily_fee
            w[t] = w[t - 1] * (1 + r[t - 1]) / nav
    return pd.DataFrame(w, rets.index, rets.columns)


def _target_book(rets, uni, cadence):
    """Same book recorded as target weights (constant between rebalances -
    the library's positions_from_signals convention)."""
    w = np.zeros((N_T, N_A))
    for t in range(N_T):
        if t % cadence == 0:
            legal = uni.iloc[max(t - 1, 0)].to_numpy()
            w[t] = legal / legal.sum()
        else:
            w[t] = w[t - 1]
    return pd.DataFrame(w, rets.index, rets.columns)


def _buy_and_hold(rets, unwind_at=None):
    """Pure buy-and-hold: entry at bar 0, zero dollars traded after, the
    dead name sold at ``unwind_at`` (the rest keeps drifting)."""
    r = rets.to_numpy()
    w = np.zeros((N_T, N_A))
    w[0] = 1.0 / N_A
    for t in range(1, N_T):
        w[t] = w[t - 1] * (1 + r[t - 1]) / (1 + w[t - 1] @ r[t - 1])
        if unwind_at is not None and t >= unwind_at:
            w[t, 0] = 0.0
    return pd.DataFrame(w, rets.index, rets.columns)


def _art(rets, sig, uni, pos):
    return BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                             universe=uni, signal_lag=1)


def _tou(rets, sig, uni, pos):
    return _run(_art(rets, sig, uni, pos))[S_TOU]


def test_held_weight_buy_and_hold_sold_inside_grace_passes():
    # True buy-and-hold, A00 sold 9 bars after its first out-of-universe
    # decision bar (inside the 21-bar grace). Zero dollars traded after
    # bar 0; a raw previous-weight comparison would read every bar A00
    # outperformed the book as an "increase".
    rets, sig, uni = _panel(7)
    pos = _buy_and_hold(rets, unwind_at=EXIT + 10)
    dollars = traded_dollars_series(pos, rets).iloc[1:EXIT + 10]
    assert dollars.max() < 1e-9                    # the book trades nothing
    r = _tou(rets, sig, uni, pos)
    assert r.status is Status.PASS, r.message
    assert r.details["n_active_cells"] == 0
    assert r.details["n_zombie_cells"] == 0
    assert r.details["n_holdover_cells"] == 9      # bars 101..109, pure holds
    assert "hold/unwind" in r.message


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_honest_monthly_held_weight_book_passes(seed):
    # The four out-of-universe bars are passive holds with drifting weights.
    rets, sig, uni = _panel(seed, vol=0.012)
    r = _tou(rets, sig, uni, _held_book(rets, uni, 21))
    assert r.status is Status.PASS, r.message
    assert r.details["n_active_cells"] == 0
    assert r.details["n_holdover_cells"] == 4


@pytest.mark.parametrize("cadence", [5, 21])
def test_constant_target_rebalance_outside_universe_fails(cadence):
    # A constant target is not a hold when returns disperse: restoring it
    # buys/sells real dollars. Any top-up of the exited name is illegal even
    # when the scheduled rebalance that finally removes it is inside grace.
    rets, sig, uni = _panel(0)
    pos = _target_book(rets, uni, cadence)
    assert traded_dollars_series(pos, rets).iloc[EXIT + 1:EXIT + 5].sum() > 0
    r = _tou(rets, sig, uni, pos)
    assert r.status is Status.FAIL, r.message
    assert r.severity is Severity.CRITICAL
    assert r.details["n_active_cells"] > 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_held_weight_book_with_nav_fee_drag_passes(seed):
    # Netting a small fee into NAV raises reported held weights relative to the
    # frictionless drift reference without trading. The bounded relative tolerance must
    # absorb that fee effect.
    rets, sig, uni = _panel(seed, vol=0.012)
    pos = _held_book(rets, uni, 21, daily_fee=1e-4)
    drift = drifted_weights(pos, rets)
    excess = (pos.to_numpy() - drift)[EXIT + 1:EXIT + 5, 0]
    assert (excess > HOLD_INCREASE_TOL).all()      # above the absolute tol
    assert (excess < 1e-4 * 0.2).all()             # ... by ~fee x weight
    r = _tou(rets, sig, uni, pos)
    assert r.status is Status.PASS, r.message
    assert r.details["n_active_cells"] == 0
    assert r.details["n_holdover_cells"] == 4


def _topped_up_above_drift(rets, uni, factor):
    """Held-weight monthly book whose dead name is re-bought on every
    out-of-universe bar (102..104) to ``factor`` x its self-financing drift."""
    pos = _held_book(rets, uni, 21)
    r = rets.to_numpy()
    for t in range(EXIT + 2, EXIT + 5):
        prev = pos.iloc[t - 1].to_numpy()
        drift = prev * (1 + r[t - 1]) / (1 + prev @ r[t - 1])
        pos.iloc[t, 0] = np.sign(drift[0]) * abs(drift[0]) * factor
    return pos


@pytest.mark.parametrize("factor", [1.005, 1.02, 1.5])
def test_small_topup_above_relative_tolerance_still_fails(factor):
    # detection guard on the relative slack: a 0.5% top-up of the dead
    # name above drift (5x HOLD_INCREASE_REL_TOL) is ACTIVE on
    # every out-of-universe bar, no grace
    assert factor - 1.0 > 2 * HOLD_INCREASE_REL_TOL
    rets, sig, uni = _panel(3)
    r = _tou(rets, sig, uni, _topped_up_above_drift(rets, uni, factor))
    assert r.status is Status.FAIL, r.message
    assert r.severity is Severity.CRITICAL
    assert r.details["n_active_cells"] == 3            # bars 102, 103, 104
    assert r.details["first_date"] == str(rets.index[EXIT + 2].date())
    assert "hindsight" in r.message


def test_topup_inside_relative_tolerance_is_the_documented_slack():
    # the boundary, pinned so a retune is deliberate: 0.05%/bar above drift
    # (half the relative slack) reads as a hold - bounded abuse of
    # ~1% of the position across the 21-bar grace, then a zombie
    rets, sig, uni = _panel(3)
    r = _tou(rets, sig, uni, _topped_up_above_drift(rets, uni, 1.0005))
    assert r.status is Status.PASS, r.message
    assert r.details["n_active_cells"] == 0
    assert r.details["n_holdover_cells"] == 4


def test_rebuy_above_both_encodings_fails_active():
    # detection guard: a top-up on the dead name that exceeds both its
    # prior target and its drifted weight is fresh exposure under either
    # honest reading -> ACTIVE, FAIL CRITICAL, no grace
    rets, sig, uni = _panel(3)
    pos = _held_book(rets, uni, 21)
    pos.iloc[EXIT + 2:EXIT + 10, 0] *= 1.5          # 50% top-up while out
    r = _tou(rets, sig, uni, pos)
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_active_cells"] > 0
    assert r.details["first_date"] == str(rets.index[EXIT + 2].date())
    assert "hindsight" in r.message


def test_entry_from_zero_while_out_fails_at_position_eps():
    # the entry-from-zero prong keeps POSITION_EPS, not the 1e-9 increase
    # tolerance: a 1e-10 re-entry on a dead name is still an entry
    rets, sig, uni = _panel(3)
    pos = _held_book(rets, uni, 21)
    pos.iloc[EXIT + 5:, 0] = 0.0
    pos.iloc[EXIT + 10:EXIT + 15, 0] = 1e-10
    assert 1e-10 > POSITION_EPS and 1e-10 < HOLD_INCREASE_TOL
    r = _tou(rets, sig, uni, pos)
    assert r.status is Status.FAIL
    assert r.details["n_active_cells"] >= 1
    assert r.details["first_date"] == str(rets.index[EXIT + 10].date())


def test_sign_flip_on_held_weight_book_fails():
    rets, sig, uni = _panel(3)
    pos = _held_book(rets, uni, 21)
    pos.iloc[EXIT + 2:EXIT + 10, 0] *= -1.0         # reverse while out
    r = _tou(rets, sig, uni, pos)
    assert r.status is Status.FAIL
    assert r.details["n_sign_flip_cells"] >= 1
    assert "REVERSE" in r.message


def test_constant_target_pinned_past_grace_is_still_zombie():
    # With zero dispersion, a constant target equals self-financing drift;
    # pinning it past grace is therefore the pure-zombie (not top-up) case.
    dates = pd.bdate_range("2020-01-02", periods=300)
    assets = [f"A{i:02d}" for i in range(12)]
    rets = pd.DataFrame(0.0, dates, assets)
    sig = rets.rolling(5, min_periods=5).mean()
    uni = pd.DataFrame(True, dates, assets)
    uni.iloc[100:, 0] = False
    pos = pd.DataFrame(1.0 / 12, dates, assets)
    r = _run(_art(rets, sig, uni, pos))[S_TOU]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_active_cells"] == 0
    assert r.details["n_zombie_cells"] == 300 - 122
    assert r.details["n_violations"] == 300 - 122
    assert r.details["n_holdover_cells"] == HOLDOVER_GRACE_BARS
    assert "zombie" in r.message


def test_held_weight_zombie_to_end_is_still_convicted():
    # a held-weight book that never sells the dead name: no active cells
    # (nothing is bought) but every bar past grace is a zombie
    rets, sig, uni = _panel(7)
    r = _tou(rets, sig, uni, _buy_and_hold(rets))
    assert r.status is Status.FAIL
    assert r.details["n_active_cells"] == 0
    assert r.details["n_zombie_cells"] > 200
    assert "zombie" in r.message


def test_stored_drift_round_off_is_not_an_increase():
    # engines store held weights to float round-off: +/-1e-12 jitter on the
    # drifted weights must not read as top-ups (HOLD_INCREASE_TOL = 1e-9)
    rets, sig, uni = _panel(11)
    pos = _held_book(rets, uni, 21)
    jit = np.random.default_rng(1).uniform(-1e-12, 1e-12, pos.shape)
    pos = pos + jit
    r = _tou(rets, sig, uni, pos)
    assert r.status is Status.PASS, r.message
    assert r.details["n_active_cells"] == 0


def test_nav_wiped_row_cannot_invent_a_constant_target_hold():
    # A -100% bar on the whole book leaves no self-financing continuation.
    # The next nonzero dead-name weight is a fresh restart, not a fabricated
    # hold against yesterday's target.
    rets, sig, uni = _panel(5)
    pos = pd.DataFrame(0.0, rets.index, rets.columns)
    pos.iloc[:EXIT + 4, 1] = 1.0                  # legal full-weight book
    rets.iloc[EXIT + 3, 1] = -1.0                 # book wiped during bar 103
    pos.iloc[EXIT + 4, 0] = 0.10                  # restart dead A00 next bar
    r = _tou(rets, sig, uni, pos)
    assert r.status is Status.FAIL, r.message
    assert r.details["n_active_cells"] > 0
    assert r.details["n_undefined_drift_rows"] >= 1


def test_exact_constant_short_topup_attack_fails():
    # Constant short target on an exited name: after A00 exits, keep a -50%
    # target while it returns -10%/bar. The short passively drifts to -42.86%,
    # so restoring -50% buys 7.14% NAV of new short exposure each day and
    # books +5% gross/bar on a name the universe says is untradeable.
    t, n = 400, 10
    dates = pd.bdate_range("2020-01-02", periods=t)
    assets = [f"A{i:02d}" for i in range(n)]
    rets = pd.DataFrame(0.0, dates, assets)
    rets.iloc[EXIT:EXIT + 10, 0] = -0.10
    sig = pd.DataFrame(0.0, dates, assets)
    uni = pd.DataFrame(True, dates, assets)
    uni.iloc[EXIT:, 0] = False
    pos = pd.DataFrame(0.0, dates, assets)
    pos.iloc[:EXIT + 10, 0] = -0.50
    pos.iloc[:, 1] = 0.50

    drift = drifted_weights(pos, rets)
    topups = (np.abs(pos.to_numpy()[EXIT + 1:EXIT + 10, 0])
              - np.abs(drift[EXIT + 1:EXIT + 10, 0]))
    assert topups == pytest.approx(np.full(9, 1 / 14))
    assert topups.sum() == pytest.approx(9 / 14)
    assert (pos.iloc[EXIT + 1:EXIT + 10]
            * rets.iloc[EXIT + 1:EXIT + 10]).sum(axis=1).sum() == \
        pytest.approx(0.45)

    r = _tou(rets, sig, uni, pos)
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_active_cells"] == 9
    assert r.details["first_date"] == str(dates[EXIT + 1].date())


# Terminal-delisting exposure disclosure.

def _delisting_book(holdover, n_delist=10, T=750, N=20, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=T)
    assets = [f"S{i:02d}" for i in range(N)]
    rets = pd.DataFrame(rng.normal(0.0, 0.015, (T, N)), dates, assets)
    delist = {f"S{N - n_delist + k:02d}": 300 + 40 * k
              for k in range(n_delist)}
    pos = pd.DataFrame(1.0 / N, dates, assets)
    for a, t0 in delist.items():
        j = rets.columns.get_loc(a)
        rets.iloc[t0:, j] = np.nan
        if holdover is not None:
            pos.iloc[t0 + holdover:, j] = 0.0
    sig = pos.shift(-1).fillna(0.0)
    return BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                             signal_lag=1), delist


def test_material_terminal_delistings_warn_and_block_gate():
    art, delist = _delisting_book(holdover=1)
    r = _run(art)[S_MISS]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["n_terminal_delistings"] == 10
    assert r.details["delisting_bar_gross_exposure"] == pytest.approx(
        10 * (1.0 / 20))                            # 0.05 on each first NaN bar
    assert "10 name(s) delist while held" in r.message
    # the disclosure is in NAV units (sum of delisting-bar weights), not a
    # fraction of the whole-history gross like frac_missing_return
    assert ("their delisting-bar weights sum to 50.0% of NAV "
            "(5.0% per name on average)") in r.message
    assert "of gross exposure sat" not in r.message
    assert "delisting return itself is not verified" in r.message
    assert r.details["terminal_delisting_exposure_unverified"] is True
    assert r.details["terminal_delisting_exposure_material"] is True
    assert r.details["terminal_delisting_max_gross_exposure"] == 0.25
    with pytest.raises(AuditFailure):
        AuditReport(results=[r]).gate(
            require=[S_MISS], warn_severity=Severity.HIGH)
    assert r.details["n_violations"] == 10
    assert r.details["frac_missing_return"] == pytest.approx(
        10 * 0.05 / (750 * 1.0 - sum(750 - t0 - 1 for t0 in delist.values())
                     * 0.05), rel=0.05)


def test_details_present_on_warn_and_zero_without_delistings():
    art, _ = _delisting_book(holdover=None)        # zombie riders -> WARN
    r = _run(art)[S_MISS]
    assert r.status is Status.WARN
    assert r.details["n_terminal_delistings"] == 10
    assert r.details["delisting_bar_gross_exposure"] == pytest.approx(0.5)
    art0, _ = _delisting_book(holdover=1, n_delist=0)
    r0 = _run(art0)[S_MISS]
    assert r0.status is Status.PASS
    assert r0.details["n_terminal_delistings"] == 0
    assert r0.details["delisting_bar_gross_exposure"] == 0.0
    assert "not verified" not in r0.message


def test_delisting_not_held_is_not_counted():
    # a name flattened before its returns go NaN sat on no delisting bar
    art, delist = _delisting_book(holdover=1)
    pos = art.positions.copy()
    a0, t0 = next(iter(delist.items()))
    pos.iloc[t0 - 5:, pos.columns.get_loc(a0)] = 0.0
    art2 = BacktestArtifacts(signals=art.signals, asset_returns=art.asset_returns,
                             positions=pos, signal_lag=1)
    r = _run(art2)[S_MISS]
    assert r.details["n_terminal_delistings"] == 9
    assert r.details["delisting_bar_gross_exposure"] == pytest.approx(0.45)


def test_count_and_exposure_describe_the_same_names():
    # a name flat on its delisting bar but bought later inside the terminal
    # NaN run is a zombie rider (n_violations counts it), not a delisting
    # held through: it is excluded from n_terminal_delistings so the count
    # and the exposure sum (first NaN bar only) describe the same names
    art, delist = _delisting_book(holdover=1)
    pos = art.positions.copy()
    a0, t0 = next(iter(delist.items()))
    j = pos.columns.get_loc(a0)
    pos.iloc[t0 - 5:t0 + 3, j] = 0.0                # flat across the bar
    pos.iloc[t0 + 3:t0 + 6, j] = 0.05                # re-bought inside the run
    art2 = BacktestArtifacts(signals=art.signals, asset_returns=art.asset_returns,
                             positions=pos, signal_lag=1)
    r = _run(art2)[S_MISS]
    assert r.details["n_terminal_delistings"] == 9
    assert r.details["delisting_bar_gross_exposure"] == pytest.approx(0.45)
    assert r.details["n_violations"] == 9 + 3
    assert "9 name(s) delist while held" in r.message
    assert "(5.0% per name on average)" in r.message


def test_clean_case_discloses_its_four_delistings():
    art = make_clean(0).artifacts
    r = _run(art)[S_MISS]
    assert r.status is Status.PASS
    assert r.details["n_terminal_delistings"] == 4
    assert 0 < r.details["delisting_bar_gross_exposure"] < 1.0
    assert "not verified" in r.message


# Attrition measured per live member-year.

PPY = 252


def _exit_art(uni: pd.DataFrame, seed=0, ppy=PPY):
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(rng.normal(0, 0.01, uni.shape), uni.index, uni.columns)
    rets = rets.where(uni)
    sig = rets.rolling(5, min_periods=5).mean()
    pos = positions_from_signals(sig, 1).fillna(0.0)
    return BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                             universe=uni, signal_lag=1, periods_per_year=ppy)


def _growing_universe(n_exits):
    """10 core names for 3 years, 190 names listing in the last month
    (mean live ~14 of 200 columns), ``n_exits`` honest terminal exits among
    the core names far from the sample edge."""
    dates = pd.bdate_range("2019-01-01", periods=756)
    n = len(dates)
    cols = [f"C{i:02d}" for i in range(10)] + [f"IPO{i:03d}" for i in range(190)]
    uni = pd.DataFrame(False, dates, cols)
    uni.iloc[:, :10] = True
    uni.iloc[n - 21:, 10:] = True
    for k in range(n_exits):
        uni.iloc[380 + 40 * k:, k] = False
    return uni


def test_growing_universe_realistic_attrition_passes():
    # false-WARN guard: 3 exits over ~42 member-years is ~7%/yr of live
    # membership; a per-column denominator would read it as
    # 3/(200 x 3y) = 0.5%/yr and warn "token attrition"
    uni = _growing_universe(3)
    r = _run(_exit_art(uni))[S_EXIT]
    member_years = uni.to_numpy(dtype=bool).sum() / PPY
    assert r.status is Status.PASS, r.message
    assert r.details["member_years"] == pytest.approx(member_years)
    assert r.details["exit_rate_per_year"] == pytest.approx(3 / member_years)
    assert 0.06 < r.details["exit_rate_per_year"] < 0.10
    assert r.details["exit_rate_per_asset_year"] == pytest.approx(3 / (200 * 3.0))
    assert r.details["exit_rate_per_asset_year"] < CFG.min_exit_rate_per_year
    assert r.details["years"] == pytest.approx(3.0)
    assert r.details["n_assets"] == 200
    assert "member-years" in r.message


def test_growing_survivor_list_with_zero_exits_still_warns():
    # detection guard on the same shape: zero exits still WARN
    uni = _growing_universe(0)
    r = _run(_exit_art(uni))[S_EXIT]
    assert r.status is Status.WARN
    assert "no asset ever leaves" in r.message


def test_growing_survivor_list_single_exit_has_no_power_documented():
    # no_exits has no power against a growth-shaped survivor list with a
    # single exit: one token exit over ~42 member-years is ~2.3%/yr - above
    # the 1% floor - so the book PASSes under the member-year denominator
    # (a per-column denominator, 1/(200 x 3y) = 0.17%, would WARN here - but
    # only because that denominator is wrong for the honest version of the
    # same shape). No statistical power here.
    uni = _growing_universe(1)
    r = _run(_exit_art(uni))[S_EXIT]
    member_years = uni.to_numpy(dtype=bool).sum() / PPY
    assert r.status is Status.PASS
    assert r.details["n_exiting_assets"] == 1
    assert r.details["exit_rate_per_year"] == pytest.approx(1 / member_years)
    assert 0.02 < r.details["exit_rate_per_year"] < 0.03
    assert r.details["exit_rate_per_asset_year"] < CFG.min_exit_rate_per_year


def test_token_exit_daily_survivor_panel_still_warns():
    # One exit among 30 names over roughly four daily years remains below the 1% annual
    # member-year floor.
    dates = pd.bdate_range("2019-01-02", periods=1000)
    cols = [f"A{i:02d}" for i in range(30)]
    uni = pd.DataFrame(True, dates, cols)
    uni.iloc[500:, 0] = False
    r = _run(_exit_art(uni))[S_EXIT]
    assert r.status is Status.WARN
    assert "token" in r.message
    assert r.details["n_exiting_assets"] == 1
    assert r.details["member_years"] == pytest.approx((30 * 1000 - 500) / PPY)
    assert r.details["exit_rate_per_year"] == pytest.approx(
        1 / ((30 * 1000 - 500) / PPY))
    assert 0.008 < r.details["exit_rate_per_year"] < 0.01


def test_token_exit_monthly_survivor_panel_still_warns():
    # One exit among 30 names over five monthly years remains below the 1% annual
    # member-year floor.
    dates = pd.date_range("2015-01-31", periods=60, freq=pd.offsets.MonthEnd())
    cols = [f"A{i:02d}" for i in range(30)]
    uni = pd.DataFrame(True, dates, cols)
    uni.iloc[30:, 0] = False
    r = _run(_exit_art(uni, ppy=12))[S_EXIT]
    assert r.status is Status.WARN
    assert "token" in r.message
    assert r.details["member_years"] == pytest.approx((30 * 60 - 30) / 12)
    assert 0.006 < r.details["exit_rate_per_year"] < 0.01


def test_static_panel_rates_nearly_unchanged():
    # honest guard: on a full-history panel the two denominators agree to
    # within the exited names' dead bars (4 exits / 30 names / 3y)
    dates = pd.bdate_range("2019-01-01", periods=756)
    cols = [f"A{i:02d}" for i in range(30)]
    uni = pd.DataFrame(True, dates, cols)
    for k, t in enumerate((300, 350, 500, 600)):
        uni.iloc[t:, k] = False
    r = _run(_exit_art(uni))[S_EXIT]
    assert r.status is Status.PASS
    assert r.details["exit_rate_per_year"] == pytest.approx(
        r.details["exit_rate_per_asset_year"], rel=0.10)
    assert r.details["exit_rate_per_year"] > r.details["exit_rate_per_asset_year"]


def test_numerator_is_distinct_assets_not_events():
    # documented numerator: a name that exits and re-enters (beyond the
    # flicker window) three times counts once
    dates = pd.bdate_range("2019-01-01", periods=756)
    cols = [f"A{i:02d}" for i in range(12)]
    uni = pd.DataFrame(True, dates, cols)
    for t in (200, 300, 400):
        uni.iloc[t:t + 30, 0] = False
    uni.iloc[600:, 1] = False
    r = _run(_exit_art(uni))[S_EXIT]
    assert r.details["n_exit_events"] == 4
    assert r.details["n_exiting_assets"] == 2
    assert r.details["exit_rate_per_year"] == pytest.approx(
        2 / (uni.to_numpy(dtype=bool).sum() / PPY))


def test_no_live_membership_skips():
    dates = pd.bdate_range("2019-01-01", periods=300)
    cols = [f"A{i:02d}" for i in range(12)]
    uni = pd.DataFrame(False, dates, cols)
    rets = pd.DataFrame(0.0, dates, cols)
    art = BacktestArtifacts(signals=rets, asset_returns=rets, positions=rets,
                            universe=uni, signal_lag=1)
    r = survivorship._no_exits(art, CFG)
    assert r.status is Status.SKIP
    assert r.details["member_years"] == 0.0
    assert "no live member-bars" in r.message
