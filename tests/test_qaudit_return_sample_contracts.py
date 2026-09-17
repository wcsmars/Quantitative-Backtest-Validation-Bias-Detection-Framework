"""Return sample coverage, constant-return Sharpe, and input value domains.

Performance evidence counts the actual return series. Short series cannot
support Sharpe suspicion; constant nonzero and never-traded series differ.
Complex data and sub-total-loss asset returns fail input validation.
"""
from __future__ import annotations

import dataclasses
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import qaudit
from qaudit.checks.performance import _SR_MIN_OBS, run
from qaudit.config import AuditConfig
from qaudit.errors import (AuditFailure, InputValidationError,
                           MisalignedInputError)
from qaudit.inputs import _MIN_OVERLAP, BacktestArtifacts
from qaudit.synthetic import (make_clean, momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()


def _by(results):
    return {r.check: r for r in results}


@pytest.fixture(scope="module")
def market():
    return simulate_market(n_assets=20, n_periods=750, seed=0)


@pytest.fixture(scope="module")
def book(market):
    rets = market["returns"]
    sig = momentum_signal(rets)
    pos = positions_from_signals(sig, 1)
    return dict(rets=rets, sig=sig, pos=pos, net=net_returns(pos, rets, 10.0),
                universe=market["universe"])


def _art(book, **over):
    kw = dict(signals=book["sig"], asset_returns=book["rets"],
              positions=book["pos"], strategy_returns=book["net"],
              universe=book["universe"], signal_lag=1,
              declared_costs_bps=10.0, periods_per_year=252)
    kw.update(over)
    return BacktestArtifacts(**kw)


@pytest.fixture(scope="module")
def clean_aligned():
    return make_clean().artifacts.aligned()


# Finite strategy-return coverage at input validation.

def test_strategy_returns_five_shared_dates_rejected(book):
    # Five shared net-return dates provide insufficient finite coverage.
    art = _art(book, strategy_returns=book["net"].iloc[400:405])
    with pytest.raises(MisalignedInputError) as ei:
        art.validate()
    msg = str(ei.value)
    assert "strategy_returns" in msg
    assert "only 5 finite value(s)" in msg
    assert f"need >= {_MIN_OVERLAP}" in msg
    assert "Reindex strategy_returns onto the backtest calendar" in msg


def test_strategy_returns_dense_index_but_five_finite_rejected(book):
    # A dense date index with only five finite returns still provides only five
    # observations.
    sr = pd.Series(np.nan, index=book["net"].index)
    sr.iloc[400:405] = book["net"].iloc[400:405]
    with pytest.raises(MisalignedInputError, match="only 5 finite value"):
        _art(book, strategy_returns=sr).validate()


def test_strategy_returns_single_date_rejected(book):
    with pytest.raises(MisalignedInputError, match="only 1 finite value"):
        _art(book, strategy_returns=book["net"].iloc[[400]]).validate()


def test_strategy_returns_floor_boundary(book):
    with pytest.raises(MisalignedInputError, match="only 29 finite value"):
        _art(book, strategy_returns=book["net"].iloc[400:429]).validate()
    _art(book, strategy_returns=book["net"].iloc[400:430]).validate()


def test_strategy_returns_scattered_finite_dates_rejected(book):
    # Attack: keep hundreds of observations but blank every second interior
    # bar (a reporter could target losing bars specifically).  A finite-count
    # floor cannot distinguish this selected sample from a return history.
    sr = book["net"].copy()
    sr.iloc[1:-1:2] = np.nan
    assert sr.notna().sum() > _MIN_OVERLAP
    with pytest.raises(MisalignedInputError) as ei:
        _art(book, strategy_returns=sr).validate()
    msg = str(ei.value)
    assert "strategy_returns is not dense over its observed span" in msg
    assert "inside the first-to-last finite return" in msg
    assert "hide losing returns" in msg
    assert "leading/trailing" in msg


def test_strategy_returns_sparse_index_cannot_hide_calendar_dates(book):
    # Same attack with missing rows rather than explicit NaNs: reindexing to
    # the audited calendar must expose the interior holes.
    sr = book["net"].iloc[::3]
    assert sr.notna().sum() > _MIN_OVERLAP and not sr.isna().any()
    with pytest.raises(MisalignedInputError, match="not dense over its observed span"):
        _art(book, strategy_returns=sr).validate()


def test_strategy_returns_leading_and_trailing_warmup_gaps_validate(book):
    # Honest truncation is outside the finite span and remains legal.
    sr = book["net"].copy()
    sr.iloc[:40] = np.nan
    sr.iloc[-35:] = np.nan
    _art(book, strategy_returns=sr).validate()


def test_strategy_returns_back_60pct_still_validates(book):
    # honest guard: a book that started late but is dense over its own span
    cut = int(0.4 * len(book["net"]))
    art = _art(book, positions=book["pos"].iloc[cut:],
               strategy_returns=book["net"].iloc[cut:])
    art.validate()
    assert art.aligned().strategy_returns.notna().sum() >= 0.6 * 750 - 5


def test_strategy_returns_zero_overlap_wording_intact(book):
    # the zero-overlap gate keeps its own message (fully disjoint dates)
    sr = book["net"].iloc[400:405].copy()
    sr.index = sr.index + pd.DateOffset(years=30)
    with pytest.raises(MisalignedInputError, match="shares no dates"):
        _art(book, strategy_returns=sr).validate()


# Sample size counts the actual return source.

def test_sample_size_warns_on_thin_return_series_over_dense_calendar(book):
    # A dense calendar with returns finite on only forty dates must report the return-
    # series sample size.
    sr = pd.Series(np.nan, index=book["net"].index)
    sr.iloc[400:440] = book["net"].iloc[400:440]
    art = _art(book, strategy_returns=sr).aligned()
    r = _by(run(art, CFG))["performance.sample_size"]
    assert r.status is Status.WARN
    assert r.severity is Severity.LOW
    assert "750 common periods but only 40 finite strategy_returns" in r.message
    assert str(CFG.min_periods) in r.message
    assert r.details["n_periods"] == 750
    assert r.details["n_strategy_obs"] == 40
    assert r.details["returns_source"] == "net strategy_returns"
    assert "suspicious_sharpe" in r.message and "deflated_sharpe" in r.message
    assert r.remediation


def test_sample_size_counts_gross_reconstruction_when_no_net(book):
    art = _art(book, strategy_returns=None).aligned()
    r = _by(run(art, CFG))["performance.sample_size"]
    assert r.status is Status.PASS
    assert "gross reconstruction" in r.details["returns_source"]
    assert r.details["n_strategy_obs"] >= CFG.min_periods
    assert "gross-reconstruction" in r.message


def test_sample_size_pass_states_both_counts(book):
    # honest guard: the full series PASSes and reports both counts
    r = _by(run(_art(book).aligned(), CFG))["performance.sample_size"]
    assert r.status is Status.PASS
    assert r.details["n_periods"] == 750
    assert r.details["n_strategy_obs"] == 750
    assert "750 common periods and 750 finite strategy_returns" in r.message


def test_sample_size_without_return_source_reports_none(book):
    art = BacktestArtifacts(signals=book["sig"],
                            asset_returns=book["rets"]).aligned()
    r = _by(run(art, CFG))["performance.sample_size"]
    assert r.status is Status.PASS
    assert r.details["n_strategy_obs"] is None
    assert r.details["returns_source"] is None
    assert "750 common periods (>= 120)" in r.message


def test_sample_size_short_calendar_and_thin_series_names_both(book):
    idx = book["net"].index[:100]
    sr = pd.Series(np.nan, index=idx)
    sr.iloc[:20] = 0.001
    art = BacktestArtifacts(signals=book["sig"].loc[idx],
                            asset_returns=book["rets"].loc[idx],
                            strategy_returns=sr).aligned()
    r = _by(run(art, CFG))["performance.sample_size"]
    assert r.status is Status.WARN
    assert "only 100 common periods and only 20 finite" in r.message


def test_performance_direct_run_skips_interior_gap_attack(book):
    # Defence in depth: module.run() documents that inputs arrive aligned,
    # but a direct caller must not get Sharpe certificates after bypassing
    # BacktestArtifacts.validate().
    sr = book["net"].copy()
    sr.iloc[200:260:2] = np.nan
    art = _art(book, strategy_returns=sr).aligned()
    res = _by(run(art, AuditConfig(n_trials=20)))
    for cid in ("performance.suspicious_sharpe",
                "performance.deflated_sharpe",
                "performance.sample_size"):
        assert res[cid].status is Status.SKIP, res[cid].message
        assert res[cid].details["n_interior_missing"] == 30
        assert "interior gaps can hide losing bars" in res[cid].message \
            or "selected-date returns" in res[cid].message


@pytest.mark.parametrize("lag", [0, -1])
def test_signal_lag_below_one_rejected_with_remap_guidance(book, lag):
    with pytest.raises(InputValidationError) as ei:
        _art(book, signal_lag=lag).validate()
    msg = str(ei.value)
    assert "signal_lag must be an int >= 1" in msg
    assert "known only at the end of period t" in msg
    assert "remap" in msg and "signal_lag=1" in msg


def test_positive_integer_signal_lags_remain_valid(book):
    for lag in (1, 2, np.int64(3)):
        _art(book, signal_lag=lag).validate()


# Suspicious Sharpe requires at least _SR_MIN_OBS observations.

def test_sr_min_obs_mirrors_the_dsr_psr_floor():
    assert _SR_MIN_OBS == 30 == _MIN_OVERLAP


def test_suspicious_sharpe_skips_fabricated_20_bar_series(book):
    # +5%/bar over 20 bars is ann SR ~50, but 20 bars is not a measurement
    # -> SKIP, not a "near-arbitrage" FAIL
    rng = np.random.default_rng(3)
    sr = pd.Series(np.nan, index=book["net"].index)
    sr.iloc[400:420] = 0.05 + rng.normal(0.0, 0.01, 20)
    art = _art(book, strategy_returns=sr).aligned()
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.SKIP
    assert "only 20 finite return observation" in r.message
    assert f"{_SR_MIN_OBS}-period floor" in r.message
    assert "not judged" in r.message
    assert r.details["n_obs"] == 20
    assert r.details["min_obs"] == _SR_MIN_OBS


def test_suspicious_sharpe_never_fails_honest_short_slices(book):
    # Twenty observations cannot support a stable suspicious-Sharpe verdict.
    statuses = set()
    for start in range(300, 700, 20):
        sr = pd.Series(np.nan, index=book["net"].index)
        sr.iloc[start:start + 20] = book["net"].iloc[start:start + 20]
        art = _art(book, strategy_returns=sr).aligned()
        statuses.add(_by(run(art, CFG))["performance.suspicious_sharpe"].status)
    assert statuses == {Status.SKIP}


def test_suspicious_sharpe_judges_at_exactly_30_obs(book):
    sr = pd.Series(np.nan, index=book["net"].index)
    sr.iloc[400:430] = book["net"].iloc[400:430]
    art = _art(book, strategy_returns=sr).aligned()
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status in (Status.PASS, Status.WARN, Status.FAIL)
    assert r.details["n_obs"] == 30


def test_suspicious_sharpe_full_honest_series_still_passes(book):
    r = _by(run(_art(book).aligned(), CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.PASS
    assert r.details["n_obs"] == 750
    assert 0 < r.details["sharpe_annualized"] < CFG.sharpe_warn


def test_suspicious_sharpe_leak_detection_power_unchanged(book):
    # honest guard on the detector: a 750-bar +5%/bar series still FAILs
    sr = pd.Series(0.05, index=book["net"].index) + np.linspace(0, 0.01, 750)
    r = _by(run(_art(book, strategy_returns=sr).aligned(),
                CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert "near-arbitrage" in r.message


# Zero-variance strategy-return semantics.

def _const_art(clean_aligned, value):
    sr = pd.Series(value, index=clean_aligned.strategy_returns.index)
    return dataclasses.replace(clean_aligned, strategy_returns=sr)


def test_constant_positive_return_fails_high(clean_aligned):
    # attack: +1%/bar riskless has Sharpe NaN; the constant-return branch must
    # FAIL rather than let both checks SKIP into a CLEAN verdict
    res = _by(run(_const_art(clean_aligned, 0.01), AuditConfig(n_trials=50)))
    r = res["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert "constant +1.0000%/period" in r.message
    assert "zero variance over 1000 periods" in r.message
    assert "unbounded" in r.message
    assert "placeholder" in r.message
    assert r.details["sharpe_annualized"] == np.inf
    assert r.details["mean_return"] == pytest.approx(0.01)
    assert r.details["n_obs"] == 1000
    assert r.details["returns_source"] == "net strategy_returns"
    assert r.remediation


def test_constant_negative_return_fails_with_negative_inf(clean_aligned):
    r = _by(run(_const_art(clean_aligned, -0.0005),
                CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert r.details["sharpe_annualized"] == -np.inf
    assert "-0.0500%/period" in r.message


@pytest.mark.parametrize("value", [0.0123, 0.007, 1 / 3, 0.02, 0.1])
def test_constant_with_float_dust_std_still_fails(clean_aligned, value):
    # numpy's std of a constant is not always exactly 0 (float dust ~1e-18)
    r = _by(run(_const_art(clean_aligned, value),
                CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert "zero variance" in r.message


def test_identically_zero_series_skips_as_never_traded(clean_aligned):
    r = _by(run(_const_art(clean_aligned, 0.0), CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.SKIP
    assert "identically zero" in r.message
    assert "never traded" in r.message
    assert r.details["mean_return"] == 0.0


def test_identically_zero_gross_reconstruction_worded_on_source(book):
    # With no strategy_returns and flat positions, the zero series comes from gross
    # reconstruction; the skip message must name that source.
    pos = pd.DataFrame(0.0, index=book["rets"].index,
                       columns=book["rets"].columns)
    art = BacktestArtifacts(signals=book["sig"], asset_returns=book["rets"],
                            positions=pos).aligned()
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.SKIP
    assert r.message.startswith(
        "gross reconstruction from positions x asset_returns is identically "
        "zero")
    assert "strategy_returns is identically zero" not in r.message
    assert "flat or NaN" in r.message
    assert r.details["returns_source"] == (
        "gross reconstruction from positions x asset_returns")
    assert r.details["mean_return"] == 0.0


def test_identically_zero_net_series_worded_on_source(clean_aligned):
    r = _by(run(_const_art(clean_aligned, 0.0), CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.SKIP
    assert r.message.startswith("net strategy_returns is identically zero")


def test_short_series_skip_does_not_interpolate_the_configured_bar(book):
    # The short-sample skip message must describe insufficient evidence without
    # attributing an empirical failure rate to the configured threshold.
    rng = np.random.default_rng(5)
    sr = pd.Series(np.nan, index=book["net"].index)
    sr.iloc[400:420] = 0.05 + rng.normal(0.0, 0.01, 20)
    art = _art(book, strategy_returns=sr).aligned()
    r = _by(run(art, AuditConfig(sharpe_fail=20.0, sharpe_warn=15.0)))[
        "performance.suspicious_sharpe"]
    assert r.status is Status.SKIP
    assert "default 5.0 FAIL bar" in r.message
    assert "20 FAIL bar" not in r.message
    assert "not judged" in r.message


def test_deflated_sharpe_skips_on_constant_pointing_at_verdict(clean_aligned):
    res = _by(run(_const_art(clean_aligned, 0.01), AuditConfig(n_trials=50)))
    r = res["performance.deflated_sharpe"]
    assert r.status is Status.SKIP
    assert "performance.suspicious_sharpe" in r.message
    assert "zero-variance" in r.message
    assert r.details["mean_return"] == pytest.approx(0.01)


def test_near_constant_with_real_dispersion_fails_the_level_bar(clean_aligned):
    # 1% +/- 1e-9 has genuine (tiny) dispersion: not the degenerate class,
    # it FAILs the ordinary bar with an astronomical Sharpe
    rng = np.random.default_rng(1)
    sr = pd.Series(0.01 + rng.uniform(-1e-9, 1e-9, 1000),
                   index=clean_aligned.strategy_returns.index)
    art = dataclasses.replace(clean_aligned, strategy_returns=sr)
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert "near-arbitrage" in r.message
    assert np.isfinite(r.details["sharpe_annualized"])
    assert r.details["sharpe_annualized"] > 1e3


def test_gate_require_performance_raises_on_constant_book():
    case = make_clean(seed=0)
    a = case.artifacts
    sr = pd.Series(0.01, index=a.strategy_returns.index)
    art = dataclasses.replace(a, strategy_returns=sr)
    rep = qaudit.audit(art, AuditConfig(n_trials=50), include=["performance"])
    assert rep.ok is False
    assert rep["performance.suspicious_sharpe"].status is Status.FAIL
    with pytest.raises(AuditFailure, match="performance.suspicious_sharpe"):
        rep.gate(require=["performance"])
    with pytest.raises(AuditFailure, match="zero variance"):
        rep.gate(require=["sharpe"])
    # the inf Sharpe must survive serialization (allow_nan=False)
    d = rep.to_dict()
    json.dumps(d, allow_nan=False)
    sr_row = next(x for x in d["results"]
                  if x["check"] == "performance.suspicious_sharpe")
    assert sr_row["details"]["sharpe_annualized"] is None


def test_constant_gross_reconstruction_is_labelled_as_such(book):
    # a positions panel whose gross P&L is constant: one asset with a
    # constant return, held at a constant weight
    rets = pd.DataFrame(0.0, index=book["rets"].index,
                        columns=book["rets"].columns)
    rets["A000"] = 0.002
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos["A000"] = 1.0
    art = BacktestArtifacts(signals=book["sig"], asset_returns=rets,
                            positions=pos).aligned()
    r = _by(run(art, CFG))["performance.suspicious_sharpe"]
    assert r.status is Status.FAIL
    assert "gross reconstruction" in r.message
    assert "+0.2000%/period" in r.message


def test_clean_book_sharpe_checks_unchanged(clean_aligned):
    # honest guard: the clean fixture's verdicts
    res = _by(run(clean_aligned, AuditConfig(n_trials=12)))
    assert res["performance.suspicious_sharpe"].status is Status.PASS
    assert 1.5 < res["performance.suspicious_sharpe"].details["sharpe_annualized"] < 2.2
    assert res["performance.deflated_sharpe"].status is Status.PASS
    assert res["performance.sample_size"].status is Status.PASS


# Asset returns below total loss.

def test_asset_returns_below_minus_one_rejected(book):
    r2 = book["rets"].copy()
    r2.iloc[100, 3] = -1.5
    r2.iloc[200, 5] = -3.0
    with pytest.raises(InputValidationError) as ei:
        _art(book, asset_returns=r2).validate()
    msg = str(ei.value)
    assert "asset_returns" in msg
    assert "2 finite value(s) below -1.0" in msg
    assert "-3.0" in msg
    assert str(r2.index[200].date()) in msg
    assert repr(r2.columns[5]) in msg
    assert "cannot lose more than 100%" in msg
    assert "percent units" in msg and "divide by 100" in msg
    assert "NaN" in msg and "-1.0 exactly" in msg


def test_percent_unit_panel_rejected(book):
    with pytest.raises(InputValidationError, match="below -1.0"):
        _art(book, asset_returns=book["rets"] * 100.0).validate()


def test_wipeout_minus_one_exactly_stays_valid(book):
    # honest guard: -100% is the wipeout contract (forward windows
    # containing it compound to exactly -1.0)
    r1 = book["rets"].copy()
    r1.iloc[100, 3] = -1.0
    _art(book, asset_returns=r1).validate()


def test_return_floor_applies_only_to_asset_returns(book):
    # signals, positions (a 150% short), prices and signal_input carry no
    # such physical bound
    sig = book["sig"].copy()
    sig.iloc[100, 3] = -5.0
    pos = book["pos"].copy()
    pos.iloc[100, 3] = -1.5
    si = book["rets"] * 100.0
    _art(book, signals=sig, positions=pos, signal_input=si).validate()


def test_return_floor_ignores_nan_cells(book):
    r = book["rets"].copy()
    r.iloc[100, 3] = np.nan
    _art(book, asset_returns=r).validate()


# Complex input dtypes.

@pytest.mark.parametrize("field", ["signals", "asset_returns", "positions",
                                   "signal_input"])
def test_complex_frame_rejected_without_warning(book, field):
    src = book["rets"] if field != "positions" else book["pos"]
    bad = src.astype("complex128") + 1j * 5.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # a ComplexWarning would be fatal
        with pytest.raises(InputValidationError) as ei:
            _art(book, **{field: bad}).validate()
    msg = str(ei.value)
    assert f"artifacts.{field}" in msg
    assert "complex" in msg
    assert "A000" in msg           # names the offending column(s)
    assert "np.real" in msg


def test_complex_single_column_names_that_column(book):
    r = book["rets"].copy()
    r["A007"] = r["A007"].astype("complex128")
    with pytest.raises(InputValidationError, match=r"\['A007'\]"):
        _art(book, asset_returns=r).validate()


def test_complex_strategy_returns_rejected(book):
    bad = book["net"].astype("complex128")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InputValidationError, match="strategy_returns.*complex"):
            _art(book, strategy_returns=bad).validate()


def test_nullable_float_frames_still_validate(book):
    # honest guard: the extension-dtype mainstream is untouched
    _art(book, asset_returns=book["rets"].convert_dtypes(),
         signals=book["sig"].convert_dtypes()).validate()


# Machine-readable not_judged flags on performance passes.

def test_not_judged_passes_carry_the_flag():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-02", periods=40)
    cols = [f"T{i}" for i in range(8)]
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (40, 8)), index=dates, columns=cols)
    sigs = pd.DataFrame(rng.normal(0.0, 1.0, (40, 8)), index=dates, columns=cols)
    res = _by(run(BacktestArtifacts(signals=sigs, asset_returns=rets).aligned(),
                  CFG))
    for cid in ("performance.ic_stability",
                "performance.ic_regime_concentration"):
        r = res[cid]
        assert r.status is Status.PASS
        assert "not judged" in r.message
        assert r.details["not_judged"] is True


def test_judged_passes_do_not_carry_the_flag(clean_aligned):
    res = _by(run(clean_aligned, CFG))
    for cid in ("performance.ic_stability",
                "performance.ic_regime_concentration"):
        r = res[cid]
        if r.status is Status.PASS and "not judged" not in r.message:
            assert "not_judged" not in r.details
