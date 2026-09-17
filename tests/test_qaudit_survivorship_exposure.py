"""Exposure-weighted missing returns, calendar grace, and closure-date budgets.

Concentrated losses cannot hide in cell-count averages. Fresh entries outside
the universe have no holdover privilege, and monthly grace uses calendar time.
Unverifiable market-wide closures remain visible in the verdict. Additional
contracts cover whole-number configuration and demo failure exit codes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit import demo
from qaudit.checks import survivorship
from qaudit.checks.survivorship import (
    MISSING_RETURN_FRAC,
    MISSING_RETURN_MAX_BAR_GROSS,
)
from qaudit.config import AuditConfig
from qaudit.errors import InputValidationError
from qaudit.inputs import BacktestArtifacts
from qaudit.report import AuditReport
from qaudit.types import CheckResult, Severity, Status

CFG = AuditConfig()
S_MISS = "survivorship.positions_on_missing_returns"
S_TOU = "survivorship.trading_outside_universe"


def _by(results):
    return {r.check: r for r in results}


def _run(art):
    return _by(survivorship.run(art, CFG))


def _art(sigs, rets, pos, uni=None, lag=1, ppy=252):
    art = BacktestArtifacts(signals=sigs, asset_returns=rets, positions=pos,
                            universe=uni, signal_lag=lag,
                            periods_per_year=ppy)
    art.validate()
    return art.aligned()


def _panel(n, m=12, seed=7, start="2020-01-06"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    cols = [f"A{i:02d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0.0003, 0.012, (n, m)), idx, cols)
    sigs = pd.DataFrame(rng.normal(0, 1, (n, m)), idx, cols)
    return idx, cols, rets, sigs


# ===========================================================================
# 1. exposure-weighted scoring + per-bar gross gate
# ===========================================================================

def _concentrated_book(nan_at, terminal=False):
    """Concentrate the book on one name, with negligible weights elsewhere."""
    idx, cols, rets, sigs = _panel(249)
    pos = pd.DataFrame(1e-6, idx, cols)
    pos["A00"] = 1.0
    pos.iloc[::5, 1:] *= 1.5                      # observable cadence
    rets = rets.copy()
    if terminal:
        rets.iloc[-1, 0] = np.nan                 # terminal delisting-shape
    else:
        rets.iloc[nan_at, 0] = np.nan             # interior closure-shape
    return _art(sigs, rets, pos)


@pytest.mark.parametrize("terminal", [False, True])
def test_full_weight_missing_bar_cannot_hide_in_cell_count(terminal):
    # 1 NaN cell on the 100%-weight name = 0.03% of cells / 0.40% of
    # aggregate gross - both under 1% - yet 100% of that bar's book PnL is
    # missing (a deleted -25% day would never be paid)
    r = _run(_concentrated_book(nan_at=120, terminal=terminal))[S_MISS]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["max_bar_gross_missing_frac"] > 0.99
    assert r.details["max_bar_gross_missing_tol"] == \
        MISSING_RETURN_MAX_BAR_GROSS
    assert r.details["frac_missing_return"] < MISSING_RETURN_FRAC  # diluted
    assert "gross book exposure" in r.message
    assert r.remediation


def test_equal_weight_terminal_delisting_bar_still_passes():
    # honest guard: one dying name in a 12-name equal book carries 1/12 of
    # a single bar's gross - far under the 25% per-bar gate and the 1%
    # aggregate; the "one delisting bar per dying asset" contract survives
    idx, cols, rets, sigs = _panel(300)
    pos = pd.DataFrame(1.0 / 12, idx, cols)
    pos.iloc[::5] *= 1.02
    rets = rets.copy()
    rets.iloc[250:, 0] = np.nan                   # A00 dies (terminal run)
    pos.iloc[251:, 0] = 0.0                       # held exactly one NaN bar
    r = _run(_art(sigs, rets, pos))[S_MISS]
    assert r.status is Status.PASS
    assert r.details["n_violations"] == 1
    assert r.details["max_bar_gross_missing_frac"] == pytest.approx(
        (1.0 / 12) / (1.0 + 0.02 / 5), rel=0.05)


def test_missing_fraction_is_exposure_weighted():
    # Twelve missing observations in a 30%-weight name are only 0.33% of cells but
    # material aggregate exposure. Exposure-weighted scoring must warn.
    idx, cols, rets, sigs = _panel(300)
    pos = pd.DataFrame(0.7 / 11, idx, cols)
    pos["A00"] = 0.30
    pos.iloc[::5] *= 1.02
    rets = rets.copy()
    rets.iloc[100:112, 0] = np.nan                # 12-bar interior hole > 5
    r = _run(_art(sigs, rets, pos))[S_MISS]
    assert r.status is Status.WARN
    assert r.details["n_violations"] == 12
    assert r.details["frac_missing_return"] > MISSING_RETURN_FRAC
    assert r.details["frac_missing_return"] == pytest.approx(
        12 * 0.30 / (300 * 1.0), rel=0.05)


# ===========================================================================
# 2. born-outside positions get no NaN-bar holdover privilege
# ===========================================================================

def _nan_entry_book(pre_held: float):
    """An asset is outside the universe for bars [200, 216); a fresh
    position appears at bar 200 on a NaN-return bar, then rides +2%/bar
    finite returns while still out, all inside the 21-bar grace.
    ``pre_held`` is the weight at the last in-universe bar (0 = the attack:
    born outside; > 0 = an already-held name, the tranche-wiggle FP class).
    """
    idx, cols, rets, sigs = _panel(504)
    m = len(cols)
    uni = pd.DataFrame(True, idx, cols)
    uni.iloc[199:216, 0] = False                  # decision-out [200, 216]
    uni.iloc[-1, -1] = False                      # honest exit elsewhere
    pos = pd.DataFrame(1.0 / m, idx, cols)
    pos.iloc[::5] *= 1.02
    pos.iloc[:, 0] = pre_held
    pos.iloc[200:216, 0] = 0.8
    pos.iloc[216:, 0] = 0.0
    rets = rets.copy()
    rets.iloc[200, 0] = np.nan                    # entry bar earns nothing...
    rets.iloc[201:216, 0] = 0.02                  # ...the ride earns +27%
    return _art(sigs, rets, pos, uni=uni)


def test_born_outside_nan_bar_entry_convicts_immediately():
    # A position first opened outside the universe is an active entry even when that
    # entry bar has no return; it has no holdover grace.
    r = _run(_nan_entry_book(pre_held=0.0))[S_TOU]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_active_cells"] >= 1
    assert r.details["n_born_outside_deadbar_cells"] >= 1
    assert r.details["n_deadbar_fresh_cells"] == 0
    assert "hindsight" in r.message


def test_held_at_exit_nan_bar_wiggle_keeps_carveout():
    # honest guard: the same NaN-bar uptick on a name that was held at its
    # last in-universe bar keeps the holdover privilege (the pinned
    # overlapping-tranche FP class), and unwinding inside grace passes
    idx, cols, rets, sigs = _panel(504)
    m = len(cols)
    uni = pd.DataFrame(True, idx, cols)
    uni.iloc[199:, 0] = False                     # A00 exits for good
    uni.iloc[-1, -1] = False
    pos = pd.DataFrame(1.0 / m, idx, cols)
    pos.iloc[::5] *= 1.02
    pos.iloc[200, 0] = 0.05                       # decaying roll-off...
    pos.iloc[201, 0] = 0.0                        # ...dips through zero...
    pos.iloc[202:204, 0] = 0.02                   # ...wiggles up on NaN bar
    pos.iloc[204:, 0] = 0.0                       # unwound inside grace (21)
    rets = rets.copy()
    rets.iloc[200:204, 0] = np.nan                # dead name: NaN returns
    r = _run(_art(sigs, rets, pos, uni=uni))[S_TOU]
    assert r.status is Status.PASS, r.message
    assert r.details["n_active_cells"] == 0
    assert r.details["n_deadbar_fresh_cells"] >= 1
    assert r.details["n_born_outside_deadbar_cells"] == 0


# ===========================================================================
# 3. calendar-denominated holdover grace
# ===========================================================================

def _monthly_book(zombie_bars: int, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=96, freq=pd.offsets.MonthEnd())
    cols = [f"A{i:02d}" for i in range(12)]
    # Zero dispersion makes the pinned weight a genuine no-trade hold; these
    # tests isolate calendar scaling of the grace rather than target trading.
    rets = pd.DataFrame(0.0, idx, cols)
    sigs = pd.DataFrame(rng.normal(0, 1, (96, 12)), idx, cols)
    uni = pd.DataFrame(True, idx, cols)
    uni.iloc[40:, 0] = False                      # A00 exits at bar 40
    pos = pd.DataFrame(1.0 / 12, idx, cols)
    pos.iloc[::2] *= 1.02                         # ~monthly cadence
    pos.iloc[:, 0] = 1.0 / 12
    pos.iloc[41 + zombie_bars:, 0] = 0.0          # held N months past exit
    return _art(sigs, rets, pos, uni=uni, ppy=12)


def test_monthly_zombie_not_sheltered_by_daily_bar_floor():
    # HOLDOVER_GRACE_BARS=21 is calendar latency (21 daily bars ~ 1 month),
    # not 21 monthly bars; the rescaled floor must convict a 20-month zombie
    r = _run(_monthly_book(zombie_bars=20))[S_TOU]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["holdover_grace_bars"] <= 3   # ~ cadence + 1-bar slack
    assert r.details["n_zombie_cells"] > 10
    assert "zombie" in r.message


def test_monthly_holdover_within_scaled_grace_passes():
    # honest guard: unwinding one bar after the exit is rebalance latency
    # at any bar frequency
    r = _run(_monthly_book(zombie_bars=1))[S_TOU]
    assert r.status is Status.PASS, r.message
    assert r.details["n_holdover_cells"] >= 1


def test_daily_grace_unchanged_at_252_ppy():
    # At 252 periods per year, calendar scaling leaves the 21-bar floor, five-bar
    # slack, and 257-bar cap unchanged.
    idx, cols, rets, sigs = _panel(300)
    rets.iloc[:] = 0.0  # unchanged weights equal self-financing drift
    uni = pd.DataFrame(True, idx, cols)
    uni.iloc[100:, 0] = False
    pos = pd.DataFrame(1.0 / 12, idx, cols)
    pos.iloc[104:, 0] = 0.0
    r = _run(_art(sigs, rets, pos, uni=uni))[S_TOU]
    assert r.status is Status.PASS
    assert r.details["holdover_grace_bars"] == 21


# ===========================================================================
# 4. market-wide deletion: long-sample budget + untestable-band scoped WARN
# ===========================================================================

def _worst_day_deletion_book(n, band_only=False, k=21):
    idx, cols, rets, sigs = _panel(n)
    m = len(cols)
    pos = pd.DataFrame(1.0 / m, idx, cols)
    pos.iloc[::5] *= 1.02
    uni = pd.DataFrame(True, idx, cols)
    uni.iloc[-1, -1] = False
    pnl = (pos.to_numpy() * rets.to_numpy()).sum(axis=1)
    day_ord = np.array([(ts - idx[0]).days for ts in idx])
    span = (idx[-1] - idx[0]).days
    in_band = (day_ord < 360) & (span - day_ord < 360)
    chosen: list[int] = []
    for t in np.argsort(pnl):
        if 1 <= t <= n - 2 and all(abs(t - c) > 1 for c in chosen) \
                and (not band_only or in_band[t]):
            chosen.append(int(t))
        if len(chosen) == k:
            break
    rets = rets.copy()
    rets.iloc[chosen, :] = np.nan                 # full-row worst-day rows
    return _art(sigs, rets, pos, uni=uni)


def test_long_sample_worst_day_deletion_trips_date_budget():
    # Twenty-one deleted worst market days across ten years can fall below a cell-mass
    # tolerance. The non-recurring closure-date budget must remain effective.
    r = _run(_worst_day_deletion_book(n=2520))[S_MISS]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["frac_uncorroborated_closure"] < MISSING_RETURN_FRAC
    assert (r.details["n_marketwide_nonrecurring_dates"]
            > r.details["marketwide_nonrecurring_budget"])
    assert "budget" in r.message
    assert "cannot tell" in r.message              # scoped, not an accusation
    assert "delisting loss" not in r.message
    assert r.remediation and "calendar" in r.remediation


def test_untestable_band_worst_day_deletion_scoped_warns():
    # short-sample twin: on a 350-bar book the recurrence-untestable band
    # covers ~half the sample; 21 worst days deleted inside it (SR 1.66 ->
    # 3.40) must surface as a cannot-verify WARN, not be auto-excluded
    r = _run(_worst_day_deletion_book(n=350, band_only=True))[S_MISS]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["n_marketwide_untestable_dates"] == 21
    assert r.details["frac_unverifiable_marketwide"] > MISSING_RETURN_FRAC
    assert r.details["n_closure_uncorroborated_cells"] == 0  # not convicted
    assert "cannot" in r.message.lower()
    assert r.remediation and "calendar" in r.remediation


def test_clean_short_book_without_closures_passes():
    # honest guard paired with the short-sample twin: the same 350-bar
    # geometry with no missing rows must PASS
    idx, cols, rets, sigs = _panel(350)
    pos = pd.DataFrame(1.0 / 12, idx, cols)
    pos.iloc[::5] *= 1.02
    uni = pd.DataFrame(True, idx, cols)
    uni.iloc[-1, -1] = False
    r = _run(_art(sigs, rets, pos, uni=uni))[S_MISS]
    assert r.status is Status.PASS
    assert r.details["max_bar_gross_missing_frac"] == 0.0
    assert r.details["n_marketwide_nonrecurring_dates"] == 0


@pytest.mark.parametrize("seed", range(5))
def test_honest_holiday_and_delisting_book_calibration(seed):
    # Four years of recurring exchange holidays and small terminal delistings provide a
    # combined control: holidays corroborate, each delisting stays below per-bar
    # materiality, and aggregate missing exposure stays below 1%.
    rng = np.random.default_rng(seed)
    n, m = 1008, 25
    idx = pd.bdate_range("2016-01-04", periods=n)
    cols = [f"A{i:02d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(2e-4, 0.01, (n, m)), idx, cols)
    anchors = ((1, 1), (2, 18), (5, 27), (7, 4), (9, 3), (11, 26), (12, 25))
    for y in sorted(set(idx.year)):
        for mo, dd in anchors:
            i = int(idx.searchsorted(pd.Timestamp(year=y, month=mo, day=dd)))
            if 2 <= i <= n - 3:
                rets.iloc[i] = np.nan
    pos = pd.DataFrame(1.0 / m, idx, cols)
    pos.iloc[::5] *= 1.02
    for j, death in enumerate((600, 700, 800)):   # 3 dying names
        rets.iloc[death:, j] = np.nan
        pos.iloc[death + 1:, j] = 0.0             # one held terminal NaN bar
    uni = pd.DataFrame(True, idx, cols)
    uni.iloc[-1, -1] = False
    sigs = pd.DataFrame(rng.normal(0, 1, (n, m)), idx, cols)
    r = _run(_art(sigs, rets, pos, uni=uni))[S_MISS]
    assert r.status is Status.PASS, r.message
    assert r.details["n_marketwide_nonrecurring_dates"] <= \
        r.details["marketwide_nonrecurring_budget"]
    assert r.details["max_bar_gross_missing_frac"] < \
        MISSING_RETURN_MAX_BAR_GROSS


# ===========================================================================
# 5. fractional counts rejected at construction
# ===========================================================================

@pytest.mark.parametrize("kw", [
    dict(min_embargo_periods=0.9),      # int() -> 0: embargo check disarmed
    dict(n_placebo=50.7),               # int() -> 50: silent shrink
    dict(ic_rolling_window=63.9),
    dict(n_trials=345.5),
    dict(min_periods=120.5),
    dict(truncation_sample_dates=11.99),
])
def test_fractional_counts_rejected_not_truncated(kw):
    name = next(iter(kw))
    with pytest.raises(InputValidationError, match=name):
        AuditConfig(**kw)


def test_integral_valued_counts_still_accepted():
    # Integral floats from JSON/YAML and NumPy integer scalars remain valid counts.
    cfg = AuditConfig(min_embargo_periods=0.0, n_placebo=50.0,
                      ic_rolling_window=np.float64(63.0),
                      n_trials=np.int64(345), min_periods=np.int64(120))
    assert cfg.embargo_periods(label_horizon=5) == 0
    assert cfg.n_placebo == 50.0


def test_fractional_mutation_rejected_and_reverted():
    cfg = AuditConfig()
    with pytest.raises(InputValidationError, match="min_embargo_periods"):
        cfg.min_embargo_periods = 2.5
    assert cfg.min_embargo_periods is None        # last valid state kept


# ===========================================================================
# 6. demo exit code covers ERROR results and missed flags
# ===========================================================================

def _fake_audit(results):
    def fake(artifacts, config, signal_func=None, backtest_func=None):
        return AuditReport(results=list(results))
    return fake


def test_demo_exits_nonzero_on_error_results(monkeypatch, capsys):
    # a crashed check module yields ERROR results; the demo must exit
    # non-zero so a CI smoke gate cannot stay green with the auditor broken
    monkeypatch.setattr(demo, "audit", _fake_audit([
        CheckResult("lookahead.crashed", Status.ERROR, "boom",
                    severity=Severity.HIGH),
        CheckResult("performance.ok", Status.PASS, "ok",
                    severity=Severity.LOW)]))
    rc = demo.main(["clean"])
    out, err = capsys.readouterr()
    assert rc == 1
    assert "auditor broken" in err and "lookahead.crashed" in err
    assert "OK (" not in out          # the guard proved nothing


def test_demo_exits_nonzero_on_missed_expected_flag(monkeypatch, capsys):
    # a defect case whose planted bug goes uncaught must exit non-zero, not
    # merely print "<-- MISSED"
    monkeypatch.setattr(demo, "audit", _fake_audit([
        CheckResult("performance.ok", Status.PASS, "ok",
                    severity=Severity.LOW)]))
    rc = demo.main(["lookahead"])
    _, err = capsys.readouterr()
    assert rc == 1
    assert "detection guard violated" in err
    assert "lookahead" in err


def test_demo_defect_case_caught_still_exits_zero(monkeypatch):
    # honest guard: when every expected flag is caught (and nothing
    # crashed) the demo returns 0
    from qaudit import synthetic
    prefixes = synthetic.ALL_CASES["lookahead"]().expected_flags
    monkeypatch.setattr(demo, "audit", _fake_audit([
        CheckResult(p, Status.FAIL, "caught", severity=Severity.HIGH)
        for p in prefixes]))
    assert demo.main(["lookahead"]) == 0
