"""Drift-aware trade costs, wipeout returns, and dependent-sample Sharpe statistics.

Constant target weights require trades against price drift; cost declarations
must reconcile per date and include entry and liquidation. Forward windows
containing a total loss retain that loss. Return dependence reduces effective
sample size, and partial price coverage cannot certify the whole panel.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qaudit._stats import (deflated_sharpe_ratio, effective_sample_size,
                           expected_max_sharpe, forward_returns,
                           probabilistic_sharpe_ratio, traded_dollars_series,
                           turnover_series)
from qaudit.checks import costs
from qaudit.checks.costs import _check_cost_sensitivity
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()
C1 = "costs.missing_transaction_costs"
C2 = "costs.no_cost_declaration"
C5 = "costs.price_return_consistency"


def _by(results):
    return {r.check: r for r in results}


# ===========================================================================
# 1. drift-aware trade economics: the mechanics
# ===========================================================================

def test_constant_target_through_dispersion_trades_the_drift():
    # A 50/50 book through +20%/-20% returns drifts to 60/40. Restoring the target
    # requires a 10% one-sided trade despite unchanged target weights.
    dates = pd.bdate_range("2020-01-01", periods=3)
    rets = pd.DataFrame([[0.2, -0.2]] * 3, index=dates, columns=["A", "B"])
    pos = pd.DataFrame(0.5, index=dates, columns=["A", "B"])
    to = turnover_series(pos, rets)
    assert np.isnan(to.iloc[0])                      # no prior book
    assert to.iloc[1] == pytest.approx(0.10, abs=1e-12)
    dollars = traded_dollars_series(pos, rets)
    assert dollars.iloc[0] == pytest.approx(1.0, abs=1e-12)   # entry trade
    assert dollars.iloc[1] == pytest.approx(0.20, abs=1e-12)


def test_true_buy_and_hold_trades_nothing_after_entry():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=100)
    rets = pd.DataFrame(rng.normal(0, 0.02, (100, 3)), index=dates,
                        columns=list("ABC"))
    r = rets.to_numpy()
    w = np.empty_like(r)
    w[0] = 1.0 / 3.0
    for t in range(1, len(r)):                       # pure drift
        w[t] = w[t - 1] * (1 + r[t - 1]) / (1 + w[t - 1] @ r[t - 1])
    pos = pd.DataFrame(w, index=dates, columns=rets.columns)
    dollars = traded_dollars_series(pos, rets)
    assert dollars.iloc[0] == pytest.approx(1.0, abs=1e-12)
    assert float(dollars.iloc[1:].abs().max()) < 1e-12


def test_wipeout_asset_renormalizes_survivors():
    # r = -100% zeroes that position; survivors renormalize by NAV growth.
    dates = pd.bdate_range("2020-01-01", periods=2)
    rets = pd.DataFrame([[-1.0, 0.0], [0.0, 0.0]], index=dates,
                        columns=["A", "B"])
    pos = pd.DataFrame(0.5, index=dates, columns=["A", "B"])
    # After a wipeout, surviving exposure renormalizes to one before the 50/50 target
    # is restored.
    dollars = traded_dollars_series(pos, rets)
    assert dollars.iloc[1] == pytest.approx(1.0, abs=1e-12)


def test_nav_wipe_surfaces_nan_not_garbage():
    dates = pd.bdate_range("2020-01-01", periods=3)
    rets = pd.DataFrame({"A": [0.0, -1.0, 0.0]}, index=dates)
    pos = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    dollars = traded_dollars_series(pos, rets)
    assert np.isnan(dollars.iloc[2])   # 1 + port(-100%) = 0: undefined book


# ===========================================================================
# 2. the adversarial constant-mix book and the one-bar round trip
# ===========================================================================

def _constant_mix_dispersion(n=252, k=2, disp=0.2):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.DataFrame(
        [[disp, -disp] if t % 2 == 0 else [-disp, disp] for t in range(n)],
        index=dates, columns=["A", "B"])
    pos = pd.DataFrame(0.5, index=dates, columns=["A", "B"])
    return dates, rets, pos


def test_annualized_turnover_and_drag_exact():
    # Annualized one-sided turnover is 25.2 times and two-sided 10 bp charges create
    # 5.04% annual drag.
    _, rets, pos = _constant_mix_dispersion()
    to = turnover_series(pos, rets).dropna()
    assert float(to.mean()) * 252 == pytest.approx(25.2, abs=1e-9)
    drag = traded_dollars_series(pos, rets).iloc[1:] * 10e-4
    assert float(drag.mean()) * 252 == pytest.approx(0.0504, abs=1e-12)


def test_net_eq_gross_constant_mix_fails_critical():
    # A frequently rebalanced book with net equal to gross must fail missing-costs
    # checks; the declaration message must not imply verified charging.
    _, rets, pos = _constant_mix_dispersion()
    gross = (pos * rets).sum(axis=1)
    art = BacktestArtifacts(signals=pos.copy(), asset_returns=rets,
                            positions=pos, strategy_returns=gross).aligned()
    res = _by(costs.run(art, CFG))
    assert res[C1].status is Status.FAIL
    assert res[C1].severity is Severity.CRITICAL
    assert res[C1].details["mean_turnover"] == pytest.approx(0.10, abs=0.01)
    assert res[C2].details["verified_by_reconstruction"] is False
    assert "NOT verify" in res[C2].message


def test_honest_constant_mix_charging_drift_costs_passes():
    # Honest guard: the same book charging 10bp on its drift trades PASSes
    # with implied == declared exactly (generator and auditor share the
    # drift-aware convention).
    _, rets, pos = _constant_mix_dispersion()
    art = BacktestArtifacts(signals=pos.copy(), asset_returns=rets,
                            positions=pos,
                            strategy_returns=net_returns(pos, rets, 10.0),
                            declared_costs_bps=10.0).aligned()
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.PASS, r.message
    assert r.details["implied_bps"] == pytest.approx(10.0, abs=1e-6)
    assert "reconciles" in r.message


def test_low_dispersion_constant_mix_stays_graceful():
    # Honest guard against an FP storm: a 1%-vol constant-mix book's drift
    # turnover is genuinely negligible (~0.9x/yr) - negligible PASS stays.
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=200)
    rets = pd.DataFrame(rng.normal(3e-4, 0.01, (200, 4)), index=dates,
                        columns=list("ABCD"))
    pos = pd.DataFrame(0.25, index=dates, columns=rets.columns)
    gross = (pos * rets).sum(axis=1)
    art = BacktestArtifacts(signals=pos.copy(), asset_returns=rets,
                            positions=pos, strategy_returns=gross).aligned()
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.PASS
    assert "negligible" in r.message


def _round_trip_art(declared=None):
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=60)
    rets = pd.DataFrame(rng.normal(0, 0.01, (60, 3)), index=dates,
                        columns=list("ABC"))
    pos = pd.DataFrame(0.0, index=dates, columns=rets.columns)
    pos.iloc[30, 0] = 1.0                       # in for one bar, out the next
    gross = (pos * rets.fillna(0.0)).sum(axis=1)
    return BacktestArtifacts(signals=rets.rolling(5).mean(),
                             asset_returns=rets, positions=pos,
                             strategy_returns=gross,
                             declared_costs_bps=declared).aligned()


def test_one_bar_round_trip_registers_as_trading():
    # A one-bar round trip trades two dollars. Its entry belongs in the cost ledger
    # even when the steady-state turnover series is empty.
    r = _by(costs.run(_round_trip_art(), CFG))[C1]
    assert r.status is Status.FAIL, r.message
    assert r.severity is Severity.CRITICAL
    assert r.details["mean_turnover"] >= 0.5    # 1.0 entry dollars / 1 live bar


def test_one_bar_round_trip_paper_trading_still_graceful():
    # Honest guard: the same book declaring 0bps is the documented
    # paper-trading configuration - C1 PASS, realism warning lives in C2.
    res = _by(costs.run(_round_trip_art(declared=0.0), CFG))
    assert res[C1].status is Status.PASS
    assert "paper-trading" in res[C1].message
    assert res[C2].status is Status.WARN


def test_negligible_mean_with_burst_bars_not_certified():
    # Second prong of the negligible gate: a tiny holding plus a 4-bar
    # 100% flip burst keeps the mean under the gate while replacing the
    # book repeatedly - the shortcut must not certify; net==gross FAILs.
    rng = np.random.default_rng(9)
    n = 1000
    dates = pd.bdate_range("2018-01-01", periods=n)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=dates,
                        columns=list("ABC"))
    pos = pd.DataFrame(0.0, index=dates, columns=rets.columns)
    pos["A"] = 0.02                              # small holding, full span
    for t in range(500, 504):                    # burst: full book flips
        pos.iloc[t, :2] = (1.0, 0.0) if t % 2 == 0 else (0.0, 1.0)
    gross = (pos * rets.fillna(0.0)).sum(axis=1)
    art = BacktestArtifacts(signals=rets.rolling(5).mean(),
                            asset_returns=rets, positions=pos,
                            strategy_returns=gross).aligned()
    r = _by(costs.run(art, CFG))[C1]
    assert r.details["mean_turnover"] <= CFG.min_turnover_for_cost_check
    assert r.status is Status.FAIL, r.message   # measured, not waved through
    # honest guard: same book charging per-trade costs passes the measurement
    art_ok = BacktestArtifacts(signals=rets.rolling(5).mean(),
                               asset_returns=rets, positions=pos,
                               strategy_returns=net_returns(pos, rets, 10.0),
                               declared_costs_bps=10.0).aligned()
    r_ok = _by(costs.run(art_ok, CFG))[C1]
    assert r_ok.status is Status.PASS, r_ok.message
    assert r_ok.details["implied_bps"] == pytest.approx(10.0, abs=0.5)


def test_negligible_pass_not_worded_verified():
    # The negligible-turnover shortcut measures nothing, so its PASS must not
    # read "verified against the gross reconstruction". True buy-and-hold
    # book, nothing declared:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=150)
    rets = pd.DataFrame(rng.normal(3e-4, 0.01, (150, 3)), index=dates,
                        columns=list("ABC"))
    r_arr = rets.to_numpy()
    w = np.empty_like(r_arr)
    w[0] = 1.0 / 3.0
    for t in range(1, len(w)):
        w[t] = w[t - 1] * (1 + r_arr[t - 1]) / (1 + w[t - 1] @ r_arr[t - 1])
    pos = pd.DataFrame(w, index=dates, columns=rets.columns)
    gross = (pos * rets).sum(axis=1)
    art = BacktestArtifacts(signals=pos.copy(), asset_returns=rets,
                            positions=pos, strategy_returns=gross).aligned()
    res = _by(costs.run(art, CFG))
    assert res[C1].status is Status.PASS
    assert res[C1].details.get("negligible_turnover") is True
    assert res[C2].status is Status.PASS
    assert res[C2].details["verified_by_reconstruction"] is False
    assert "measured nothing" in res[C2].message


# ===========================================================================
# 3. per-date cost reconciliation (mean-blind cost streams)
# ===========================================================================

@pytest.fixture(scope="module")
def weekly_drift_book():
    """Weekly-rebalanced hold-with-drift book plus its drift-aware dollars."""
    m = simulate_market(seed=41)
    sig = momentum_signal(m["returns"])
    tgt = positions_from_signals(sig, 1)
    keep = np.arange(len(tgt)) % 5 == 0
    r = m["returns"].fillna(0.0).to_numpy()
    ta = tgt.to_numpy(dtype=float)
    w = np.zeros_like(ta)
    w[0] = ta[0]
    for t in range(1, len(w)):
        if keep[t]:
            w[t] = ta[t]
        else:
            w[t] = w[t - 1] * (1 + r[t - 1]) / (1 + w[t - 1] @ r[t - 1])
    pos = pd.DataFrame(w, index=tgt.index, columns=tgt.columns)
    dollars = traded_dollars_series(pos, m["returns"])
    gross = (pos * m["returns"].fillna(0.0)).sum(axis=1)
    return m, sig, pos, dollars, gross


def test_flat_per_bar_charge_warns_not_certified(weekly_drift_book):
    # A flat per-bar deduction can match mean drag without matching trade costs per
    # date. Warn about the unexplained structure rather than certify per-trade
    # charging.
    m, sig, pos, dollars, gross = weekly_drift_book
    flat = gross - float((dollars * 10e-4).mean())
    art = BacktestArtifacts(signals=sig, asset_returns=m["returns"],
                            positions=pos, strategy_returns=flat,
                            declared_costs_bps=10.0).aligned()
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.WARN, r.message
    assert r.severity is Severity.MEDIUM
    assert "date by date" in r.message
    assert (r.details["drag_rms_residual"]
            > r.details["reconcile_rel_tol"] * r.details["drag_residual_scale"])


def test_shifted_date_charge_warns_not_certified(weekly_drift_book):
    # Charging the right cost one bar late has the right mean and the wrong
    # per-date structure.
    m, sig, pos, dollars, gross = weekly_drift_book
    shifted = gross - (dollars * 10e-4).shift(1).fillna(0.0)
    art = BacktestArtifacts(signals=sig, asset_returns=m["returns"],
                            positions=pos, strategy_returns=shifted,
                            declared_costs_bps=10.0).aligned()
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.WARN, r.message
    assert "date by date" in r.message


def test_honest_per_trade_charge_reconciles(weekly_drift_book):
    m, sig, pos, dollars, gross = weekly_drift_book
    art = BacktestArtifacts(signals=sig, asset_returns=m["returns"],
                            positions=pos,
                            strategy_returns=net_returns(pos, m["returns"],
                                                         10.0),
                            declared_costs_bps=10.0).aligned()
    r = _by(costs.run(art, CFG))[C1]
    assert r.status is Status.PASS, r.message
    assert r.details["implied_bps"] == pytest.approx(10.0, abs=1e-6)
    assert (r.details["drag_rms_residual"]
            <= 1e-10 * max(r.details["drag_residual_scale"], 1e-300))


# ===========================================================================
# 4. forward_returns: -100% is a valid observation
# ===========================================================================

def _wipeout_panel():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2019-01-01", periods=260)
    panel = pd.DataFrame(rng.normal(0, 0.02, (260, 6)), index=idx,
                         columns=list("ABCDEF"))
    spots = [(30, 0), (77, 2), (140, 4), (200, 1)]
    for t, j in spots:
        panel.iloc[t, j] = -1.0
    panel.iloc[50:55, 3] = np.nan            # genuinely missing stretch
    return panel, spots


@pytest.mark.parametrize("h", [2, 3, 6])
def test_wipeout_windows_are_exactly_minus_one(h):
    panel, spots = _wipeout_panel()
    fwd = forward_returns(panel, h)
    for t, j in spots:
        for k in range(1, h + 1):
            assert fwd.iloc[t - k, j] == -1.0, (h, t, k)


@pytest.mark.parametrize("h", [2, 3, 6])
def test_clean_windows_match_rolling_product_oracle(h):
    panel, _ = _wipeout_panel()
    fwd = forward_returns(panel, h)
    oracle = ((1 + panel).rolling(h).apply(np.prod, raw=True) - 1).shift(-h)
    both = fwd.notna() & oracle.notna()
    assert np.nanmax(np.abs(fwd[both] - oracle[both]).to_numpy()) < 5e-15
    # no silent drops: every finite oracle cell is finite in fwd too, and
    # the wipeout windows the oracle also resolves agree
    assert int(fwd.notna().to_numpy().sum()) >= int(
        oracle.notna().to_numpy().sum())


def test_wipeout_plus_missing_window_is_still_minus_one():
    # the product contains a zero whatever the missing datum was
    idx = pd.bdate_range("2020-01-01", periods=10)
    panel = pd.DataFrame({"A": [0.01, np.nan, -1.0, 0.01, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0]}, index=idx)
    fwd = forward_returns(panel, 3)
    assert fwd.iloc[0, 0] == -1.0            # window covers NaN and wipeout
    assert np.isfinite(forward_returns(panel, 2).iloc[3, 0])  # clean window


def test_missing_only_window_stays_nan():
    idx = pd.bdate_range("2020-01-01", periods=8)
    panel = pd.DataFrame({"A": [0.01, np.nan, 0.01, 0.01, 0.01, 0.01, 0.01,
                                0.01]}, index=idx)
    fwd = forward_returns(panel, 2)
    assert np.isnan(fwd.iloc[0, 0])          # covers the missing bar
    assert np.isfinite(fwd.iloc[2, 0])


def test_sub_wipeout_corrupt_returns_floored_at_h2():
    # Direct forward-return calls defensively floor sub-total-loss values at horizons
    # above one. Audited asset returns reject these invalid values at intake.
    idx = pd.bdate_range("2020-01-01", periods=6)
    panel = pd.DataFrame({"A": [0.0, -1.5, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    assert forward_returns(panel, 2).iloc[0, 0] == -1.0
    assert forward_returns(panel, 1).iloc[0, 0] == -1.5


# ===========================================================================
# 5. DSR/PSR effective sample size
# ===========================================================================

def test_tiled_returns_do_not_inflate_dsr():
    # Tiling fifty observations twenty times does not create one thousand independent
    # returns. Effective sample size must prevent inflated Sharpe confidence.
    block = np.random.default_rng(7).normal(0.002, 0.0125, 50)
    block = block - block.mean() + 0.159 * block.std(ddof=1)
    tiled = np.tile(block, 20)
    d = deflated_sharpe_ratio(tiled, 100)
    iid_assumption = probabilistic_sharpe_ratio(
        tiled, sr_benchmark=expected_max_sharpe(100, 1.0 / 1000))
    assert iid_assumption > 0.99                    # treating repeated blocks as independent overstates confidence
    assert d["dsr"] < 0.5, d
    assert d["n_eff"] < 150                  # ~103 vs 1000 raw rows
    assert d["dependence_variance_ratio"] > 5


def test_iid_returns_bit_identical_to_bailey_ldp():
    x = np.random.default_rng(123).normal(5e-4, 0.01, 1000)
    d = deflated_sharpe_ratio(x, 50)
    assert d["n_eff"] == 1000.0              # no significant lag fires
    expected = probabilistic_sharpe_ratio(
        x, sr_benchmark=expected_max_sharpe(50, 1.0 / 1000))
    assert d["dsr"] == expected
    assert d["dependence_variance_ratio"] == 1.0


def test_iid_false_detection_rate_negligible():
    hits = sum(effective_sample_size(
        np.random.default_rng(2000 + s).normal(0, 0.01, 500))["n_eff"] < 500
        for s in range(60))
    assert hits <= 6                         # familywise alpha 0.05 per seed


def test_ar1_neff_tracks_theory():
    rng = np.random.default_rng(4)
    eps = rng.normal(0, 1, 4000)
    x = np.empty_like(eps)
    x[0] = eps[0]
    for t in range(1, len(eps)):
        x[t] = 0.5 * x[t - 1] + eps[t]
    ratio = effective_sample_size(x)["n_eff"] / 4000
    assert 0.25 < ratio < 0.45               # theory (1-.5)/(1+.5) = 1/3


def test_negative_dependence_capped_at_n():
    # A mean-reverting book (negative acf) must never get a certificate
    # boost above the IID case.
    z = np.random.default_rng(3).normal(0, 1, 2001)
    ess = effective_sample_size(np.diff(z))  # MA(1), rho1 = -0.5
    assert ess["n_eff"] == 2000.0
    assert ess["variance_ratio"] == 1.0
    assert ess["n_dependent_lags"] >= 1      # detected, then capped


def test_dsr_details_castable_for_wrapper():
    # performance._deflated_sharpe casts every dict value through float()
    d = deflated_sharpe_ratio(np.random.default_rng(0).normal(0, 0.01, 100),
                              10)
    for v in d.values():
        float(v)


def test_psr_neff_below_floor_returns_nan():
    x = np.random.default_rng(1).normal(0, 0.01, 200)
    assert np.isnan(probabilistic_sharpe_ratio(x, 0.0, n_eff=10.0))
    assert np.isfinite(probabilistic_sharpe_ratio(x, 0.0))   # default path


# ===========================================================================
# 6. price_return_consistency coverage
# ===========================================================================

def _sparse_price_arts(cover_assets, scale_uncovered=100.0, n=300, k=20,
                       prices_subset_columns=False, scale_covered=False):
    rng = np.random.default_rng(21)
    dates = pd.bdate_range("2019-01-01", periods=n)
    cols = [f"A{i}" for i in range(k)]
    # Low volatility keeps the scaled panel above intake's -100% floor, allowing the
    # price-consistency check to evaluate the scale mismatch.
    rets = pd.DataFrame(rng.normal(0, 0.002, (n, k)), index=dates,
                        columns=cols)
    prices = 100.0 * (1.0 + rets).cumprod()
    uncovered = [c for c in cols if c not in cover_assets]
    prices[uncovered] = np.nan
    if prices_subset_columns:
        prices = prices[list(cover_assets)]
    if uncovered and scale_uncovered != 1.0:
        rets[uncovered] = rets[uncovered] * scale_uncovered   # percent-vs-decimal
    if scale_covered:
        rets[list(cover_assets)] = rets[list(cover_assets)] * 100.0
    art = BacktestArtifacts(signals=rets.rolling(5).mean(),
                            asset_returns=rets, prices=prices)
    art.validate()
    return art.aligned()


def test_single_covered_asset_cannot_certify_the_panel():
    # Prices covering one of twenty assets cannot certify scaling on the other
    # nineteen.
    art = _sparse_price_arts(["A0"])
    r = _by(costs.run(art, CFG))[C5]
    assert r.status is Status.WARN, r.message
    assert r.severity is Severity.MEDIUM
    assert "cover only" in r.message
    assert r.details["frac_return_cells_covered"] < 0.06
    assert r.details["n_assets_compared"] == 1
    assert r.details["n_assets_total"] == 20


def test_price_columns_absent_variant_same_verdict():
    art = _sparse_price_arts(["A0"], prices_subset_columns=True)
    r = _by(costs.run(art, CFG))[C5]
    assert r.status is Status.WARN
    assert "cover only" in r.message


def test_defect_on_covered_subset_still_detected_first():
    # Detection kept under low coverage: the covered asset itself carries
    # the x100 defect - the systematic-mismatch WARN outranks the soft
    # coverage WARN.
    art = _sparse_price_arts(["A0"], scale_uncovered=1.0, scale_covered=True)
    r = _by(costs.run(art, CFG))[C5]
    assert r.status is Status.WARN
    assert "systematically disagree" in r.message


def test_full_coverage_pass_stays_unscoped():
    art = _sparse_price_arts([f"A{i}" for i in range(20)],
                             scale_uncovered=1.0)
    r = _by(costs.run(art, CFG))[C5]
    assert r.status is Status.PASS
    assert "match" in r.message
    assert "Scope:" not in r.message
    assert r.details["frac_return_cells_covered"] == 1.0


def test_partial_above_floor_pass_is_scoped():
    art = _sparse_price_arts([f"A{i}" for i in range(19)],
                             scale_uncovered=1.0)   # A19 uncovered but honest
    r = _by(costs.run(art, CFG))[C5]
    assert r.status is Status.PASS
    assert "Scope:" in r.message
    assert "19 of 20" in r.message
    assert 0.90 < r.details["frac_return_cells_covered"] < 1.0


# ===========================================================================
# 7. empty cost sweep is a SKIP, not a green check
# ===========================================================================

def test_empty_cost_sweep_skips_not_passes():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2020-01-01", periods=120)
    rets = pd.DataFrame(rng.normal(3e-4, 0.01, (120, 3)), index=dates,
                        columns=list("ABC"))
    pos = pd.DataFrame(1.0 / 3.0, index=dates, columns=rets.columns)
    art = BacktestArtifacts(signals=pos.copy(), asset_returns=rets,
                            positions=pos).aligned()
    # duck config: bypasses AuditConfig validation, which rejects an empty sweep
    cfg = SimpleNamespace(cost_sensitivity_bps=())
    r = _check_cost_sensitivity(art, cfg)   # type: ignore[arg-type]
    assert r.status is Status.SKIP
    assert "empty" in r.message
    assert "NOT tested" in r.message
