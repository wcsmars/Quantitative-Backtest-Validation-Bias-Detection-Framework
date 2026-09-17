"""Causal-probe coverage, short-horizon alpha ambiguity, and split-window support.

Signal reproducibility must cover each asset and date, including missing
output cells. Date-shift verdicts distinguish causal positive remnants from
embedded-return leaks. Frozen closure scalers remain a stated identification
limit. Split checks require observed train/test windows, and pooled forward
IC statistics align by decision date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qaudit.checks.lookahead as lookahead
from qaudit.checks.contamination import run as contamination_run
from qaudit.config import AuditConfig
from qaudit.dynamic import probes_shift
from qaudit.inputs import BacktestArtifacts
from qaudit.types import Severity, Status

CFG = AuditConfig()

C_IDS = ["contamination.split_declared", "contamination.split_overlap",
         "contamination.insufficient_embargo",
         "contamination.test_before_train", "contamination.split_coverage"]


def _by(results):
    return {r.check: r for r in results}


# ===========================================================================
# reproducibility concentration gate
# ===========================================================================

def _one_asset_leak_book(seed=11, n=750, m=200):
    """Honest 20-day momentum on every asset except A000, whose next-bar return dominates
    the book. One of 200 assets occupies about 0.51% of cells, below the global mismatch
    tolerance."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(n, m)), index=dates,
                        columns=cols)

    def honest_func(inp: pd.DataFrame) -> pd.DataFrame:
        return inp.shift(1).rolling(20).mean()

    signals = honest_func(rets)
    signals["A000"] = rets["A000"].shift(-1) * 25.0

    def bt(sig: pd.DataFrame, r: pd.DataFrame) -> pd.Series:
        return (sig.shift(1) * r).sum(axis=1)

    art = BacktestArtifacts(signals=signals, asset_returns=rets,
                            signal_input=rets)
    art.validate()
    return art, honest_func, bt


def test_one_asset_replacement_blocks_stamp_and_stays_critical():
    art, honest_func, bt = _one_asset_leak_book()
    res = _by(probes_shift.run(art, CFG, signal_func=honest_func,
                               backtest_func=bt))
    r = res[probes_shift.CHECK_REPRO]
    # A small global mismatch cannot hide a fully replaced asset from the concentration
    # gate.
    assert r.status is Status.WARN
    assert r.details["frac_mismatch"] <= probes_shift.REPRO_MISMATCH_FRAC
    assert r.details["n_concentrated_assets"] == 1
    assert (r.details["max_asset_mismatch_frac"]
            >= probes_shift.REPRO_CONC_MISMATCH_FRAC)
    assert "CONCENTRATED" in r.message
    assert "A000" in r.message                      # names the offender
    ds = res[probes_shift.CHECK_DATE_SHIFT]
    assert ds.status is Status.FAIL                 # stamp withheld
    assert ds.severity is Severity.CRITICAL
    assert ds.details["signal_verified_causal"] is False


def test_nan_column_variant_blocks_stamp():
    # NaNing the leaky column clears aggregate coverage and mismatch tolerances; per-
    # column concentration must still reject the causal stamp.
    art, honest_func, _ = _one_asset_leak_book()

    def nan_col_func(inp: pd.DataFrame) -> pd.DataFrame:
        out = honest_func(inp)
        out["A000"] = np.nan
        return out

    res = _by(probes_shift.run(art, CFG, signal_func=nan_col_func))
    r = res[probes_shift.CHECK_REPRO]
    assert r.status is Status.WARN
    assert r.details["coverage"] >= probes_shift.REPRO_MIN_COVERAGE
    assert r.details["frac_mismatch"] <= probes_shift.REPRO_MISMATCH_FRAC
    assert r.details["n_concentrated_assets"] == 1


def test_few_whole_dates_replacement_blocks_stamp():
    # Per-date axis: 3 replaced dates of ~730 is also < 1% globally; each
    # replaced row mismatches ~100% of its own cells.
    rng = np.random.default_rng(200)
    n, m = 750, 200
    dates = pd.bdate_range("2017-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(n, m)), index=dates,
                        columns=cols)

    def honest_func(inp: pd.DataFrame) -> pd.DataFrame:
        return inp.shift(1).rolling(20).mean()

    leaky = honest_func(rets)
    picks = leaky.index[100:400:100]        # 3 whole dates
    leaky.loc[picks] = rets.shift(-1).loc[picks] * 25.0
    art = BacktestArtifacts(signals=leaky, asset_returns=rets,
                            signal_input=rets)
    art.validate()
    r = _by(probes_shift.run(art, CFG, signal_func=honest_func))[
        probes_shift.CHECK_REPRO]
    assert r.status is Status.WARN
    assert r.details["frac_mismatch"] <= probes_shift.REPRO_MISMATCH_FRAC
    assert r.details["n_concentrated_dates"] == 3


def test_scattered_subtolerance_noise_keeps_pass():
    # Scattered independent mismatch near 0.5% is the noise the global tolerance is
    # designed to absorb; the per-axis concentration gate must not fire.
    rng = np.random.default_rng(100)
    n, m = 750, 200
    dates = pd.bdate_range("2017-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(n, m)), index=dates,
                        columns=cols)

    def honest_func(inp: pd.DataFrame) -> pd.DataFrame:
        return inp.shift(1).rolling(20).mean()

    noisy = honest_func(rets).copy()
    mask = rng.random(size=noisy.shape) < 0.005
    vals = noisy.to_numpy(copy=True)
    vals[mask] = vals[mask] + 1.0
    art = BacktestArtifacts(signals=pd.DataFrame(vals, index=dates,
                                                 columns=cols),
                            asset_returns=rets, signal_input=rets)
    art.validate()
    r = _by(probes_shift.run(art, CFG, signal_func=honest_func))[
        probes_shift.CHECK_REPRO]
    assert r.status is Status.PASS
    assert (r.details["max_asset_mismatch_frac"]
            < probes_shift.REPRO_CONC_MISMATCH_FRAC)


def test_clean_repro_pass_reports_concentration_evidence():
    # The PASS message carries the concentration evidence the 1%-tolerance
    # claim rests on.
    rng = np.random.default_rng(5)
    n, m = 400, 20
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"A{i}" for i in range(m)]
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(n, m)), index=dates,
                        columns=cols)

    def honest_func(inp: pd.DataFrame) -> pd.DataFrame:
        return inp.shift(1).rolling(20).mean()

    art = BacktestArtifacts(signals=honest_func(rets), asset_returns=rets,
                            signal_input=rets)
    art.validate()
    r = _by(probes_shift.run(art, CFG, signal_func=honest_func))[
        probes_shift.CHECK_REPRO]
    assert r.status is Status.PASS
    assert r.details["max_asset_mismatch_frac"] == 0.0
    assert r.details["max_date_mismatch_frac"] == 0.0
    assert "max per-asset" in r.message


# ===========================================================================
# positive-remnant ambiguous band
# ===========================================================================

def _ar1_fast_alpha_book(seed=3, n=2000, m=10, phi=0.42, b=0.00021):
    """A causal AR(1) observable with return[t] = b * X[t-1] + noise. The edge has a one-
    bar horizon at lag 1, so an earlier signal leaves a positive remnant proportional to
    persistence."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    cols = [f"A{i}" for i in range(m)]
    e = rng.normal(0, 1, size=(n, m))
    X = np.zeros((n, m))
    for t in range(1, n):
        X[t] = phi * X[t - 1] + e[t]
    X = pd.DataFrame(X, index=dates, columns=cols)
    rets = b * X.shift(1) + pd.DataFrame(rng.normal(0, 0.01, size=(n, m)),
                                         index=dates, columns=cols)
    rets.iloc[0] = 0.0

    def signal_func(inp: pd.DataFrame) -> pd.DataFrame:
        return inp * 1.0

    def bt(sig: pd.DataFrame, r: pd.DataFrame) -> pd.Series:
        return (sig.shift(1) * r).mean(axis=1)

    art = BacktestArtifacts(signals=signal_func(X), asset_returns=rets,
                            signal_input=X)
    art.validate()
    return art, signal_func, bt


def test_causal_fast_alpha_positive_remnant_warns_not_critical():
    art, _, bt = _ar1_fast_alpha_book()
    r = _by(probes_shift.run(art, CFG, backtest_func=bt))[
        probes_shift.CHECK_DATE_SHIFT]
    # A causal one-bar alpha with a positive remnant needs a scoped warning.
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    d = r.details
    assert d["peek_signature"] == "ambiguous"
    assert d["peek_ambig_side"] == "positive_remnant"
    ratio = d["sr_peek_1"] / d["sr_baseline"]
    assert probes_shift.PEEK_AMBIG_MIN_RATIO <= ratio < 0.5
    assert "POSITIVE remnant" in r.message
    assert "NEXT bar" in r.message                  # names the mechanism
    for probe in (probes_shift.CHECK_REPRO, probes_shift.CHECK_TRUNCATION,
                  probes_shift.CHECK_SENSITIVITY):
        assert probe in r.remediation


def test_causal_fast_alpha_stamped_downgrades_to_medium():
    art, sf, bt = _ar1_fast_alpha_book()
    res = _by(probes_shift.run(art, CFG, signal_func=sf, backtest_func=bt))
    for probe in (probes_shift.CHECK_REPRO, probes_shift.CHECK_TRUNCATION,
                  probes_shift.CHECK_SENSITIVITY):
        assert res[probe].status is Status.PASS     # stamp earned honestly
    r = res[probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert r.details["peek_ambig_side"] == "positive_remnant"
    assert "verified causal" in r.message
    assert "autocorrelation" in r.remediation


def test_welded_leak_positive_tiny_remnant_stays_critical():
    # Detection-power guard: a welded leak's k=-1 remnant can be positive
    # and even clear the noise floor, but never the commensurability prong
    # (measured ratio 0.006-0.014 << 0.25) - the conviction must survive
    # the ambiguous band.
    art, _, bt = _one_asset_leak_book(seed=11)
    r = _by(probes_shift.run(art, CFG, backtest_func=bt))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["peek_signature"] == "evaporation"
    assert r.details["sr_peek_1"] > 0               # the remnant is positive
    # the message must not assert the false premise; it must carry the
    # conviction phrase
    assert "More information cannot hurt" not in r.message
    assert "embeds that bar's return" in r.message
    assert "signal_func" in r.remediation           # adjudication path named


# ===========================================================================
# hidden fitted-parameter closures: scoping + fingerprint advisory
# ===========================================================================

def _closure_case(kind):
    rng = np.random.default_rng(7)
    n, m = 600, 6
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"A{i}" for i in range(m)]
    drift = np.cumsum(rng.normal(0, 0.05, size=(n, m)), axis=0)
    raw = pd.DataFrame(drift + rng.normal(0, 1.0, size=(n, m)),
                       index=dates, columns=cols)
    if kind == "zscore":
        mu, sd = raw.mean(), raw.std()
        sf = lambda inp: (inp - mu) / sd            # noqa: E731
    elif kind == "minmax":
        lo, hi = raw.min(), raw.max()
        sf = lambda inp: (inp - lo) / (hi - lo)     # noqa: E731
    else:                                            # honest expanding
        def sf(inp):
            return (inp - inp.expanding(30).mean()) / inp.expanding(30).std()
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(n, m)), index=dates,
                        columns=cols)
    art = BacktestArtifacts(signals=sf(raw), asset_returns=rets,
                            signal_input=raw)
    art.validate()
    return art, sf


@pytest.mark.parametrize("kind,pattern", [("zscore", "z-score"),
                                          ("minmax", "min-max")])
def test_hidden_closure_scaler_flagged_by_fingerprint_advisory(kind, pattern):
    # The stamp is still granted - frozen fitted constants are black-box
    # indistinguishable from a-priori hyperparameters (documented design
    # gap, not behaviorally fixable) - but the truncation PASS carries the
    # static fingerprint advisory naming the seam.
    art, sf = _closure_case(kind)
    res = _by(probes_shift.run(art, CFG, signal_func=sf))
    tr = res[probes_shift.CHECK_TRUNCATION]
    assert tr.status is Status.PASS
    fp = tr.details["fullsample_scaler_fingerprint"]
    assert fp is not None and pattern in fp
    assert "ADVISORY" in tr.message
    assert "OUTSIDE signal_func" in tr.message


def test_honest_expanding_zscore_no_fingerprint():
    art, sf = _closure_case("expanding")
    tr = _by(probes_shift.run(art, CFG, signal_func=sf))[
        probes_shift.CHECK_TRUNCATION]
    assert tr.status is Status.PASS
    assert tr.details["fullsample_scaler_fingerprint"] is None
    assert "ADVISORY" not in tr.message


def test_truncation_pass_wording_is_scoped():
    # the PASS must not make the categorical claim "rolling computations
    # are causal"; it claims only input-argument dependence at the sampled
    # dates.
    art, sf = _closure_case("expanding")
    tr = _by(probes_shift.run(art, CFG, signal_func=sf))[
        probes_shift.CHECK_TRUNCATION]
    assert "rolling computations are causal" not in tr.message
    assert "input" in tr.message
    assert "outside this probe's reach" in tr.message


# ===========================================================================
# contamination: vacuous declarations and the train floor
# ===========================================================================

def _c_artifacts(train, test, n=210, label_horizon=1):
    dates = pd.bdate_range("2020-01-01", periods=n)
    cols = ["X", "Y", "Z"]
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, len(cols))),
                        index=dates, columns=cols)
    sig = rets.rolling(5, min_periods=5).mean()
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            train_period=train, test_period=test,
                            label_horizon=label_horizon)
    art.validate()
    return art.aligned(), dates


@pytest.mark.parametrize("train", [
    ("2010-01-01", "2010-12-31"),                       # before the data
    (pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-05")),  # weekend hole
])
def test_zero_train_periods_downgrades_family(train):
    dates = pd.bdate_range("2020-01-01", periods=210)
    art, _ = _c_artifacts(train, (dates[11], dates[209]))
    res = contamination_run(art, CFG)
    by = _by(res)
    assert [r.check for r in res] == C_IDS              # all five ids, once
    r = by["contamination.split_declared"]
    # No observed training periods must not yield five clean split verdicts.
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    assert r.details["n_train"] == 0
    assert "0" in r.message and "cannot certify" in r.message
    for cid in ("contamination.split_overlap",
                "contamination.insufficient_embargo",
                "contamination.split_coverage"):
        assert by[cid].status is Status.SKIP
        assert "split_declared" in by[cid].message       # pointer
    # declared-timestamps check stays live (it needs no in-data periods)
    assert by["contamination.test_before_train"].status is Status.PASS


def test_future_fake_train_keeps_test_before_train_live():
    # Train declared after the data: split_declared WARNs on the vacuity
    # and test_before_train still lands its hindsight WARN.
    dates = pd.bdate_range("2020-01-01", periods=210)
    art, _ = _c_artifacts(("2030-01-01", "2030-12-31"),
                          (dates[0], dates[198]))
    by = _by(contamination_run(art, CFG))
    assert by["contamination.split_declared"].status is Status.WARN
    assert by["contamination.test_before_train"].status is Status.WARN


def test_zero_test_periods_fails():
    # n_test=0 is harsher than n_train=0: every "OOS" statistic came from
    # no data in this panel - no honest workflow produces that.
    dates = pd.bdate_range("2020-01-01", periods=210)
    art, _ = _c_artifacts((dates[0], dates[99]),
                          ("2030-01-01", "2030-12-31"))
    res = contamination_run(art, CFG)
    by = _by(res)
    assert [r.check for r in res] == C_IDS
    r = by["contamination.split_declared"]
    assert r.status is Status.FAIL
    assert r.severity is Severity.HIGH
    assert r.details["n_test"] == 0
    assert "out-of-sample" in r.message
    assert r.remediation


def test_tiny_train_floor_warns_low():
    # Three training observations require a LOW coverage warning with the measured
    # count.
    dates = pd.bdate_range("2020-01-01", periods=210)
    art, _ = _c_artifacts((dates[0], dates[2]), (dates[11], dates[209]))
    by = _by(contamination_run(art, CFG))
    r = by["contamination.split_coverage"]
    assert r.status is Status.WARN
    assert r.severity is Severity.LOW
    assert r.details["n_train"] == 3
    assert r.details["needed_train"] == 60
    assert "3" in r.message and "60" in r.message
    # zero-train is categorical, small-train is hygiene: family stays live
    assert by["contamination.split_overlap"].status is Status.PASS


def test_healthy_split_stays_all_pass():
    # Honest guard: floors satisfied -> no WARN anywhere.
    dates = pd.bdate_range("2020-01-01", periods=210)
    art, _ = _c_artifacts((dates[0], dates[119]), (dates[125], dates[209]))
    res = contamination_run(art, CFG)
    assert [r.check for r in res] == C_IDS
    assert all(r.status is Status.PASS for r in res)


# ===========================================================================
# smeared IC: calendar alignment of the pooled NW t-stat
# ===========================================================================

def _smear_panel(n_nan, T=300, N=10, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=T)
    cols = [f"A{i}" for i in range(N)]
    signals = pd.DataFrame(rng.normal(size=(T, N)), index=idx, columns=cols)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(T, N)), index=idx,
                        columns=cols)
    if n_nan:
        nan_dates = rng.choice(np.arange(20, T - 20), size=n_nan,
                               replace=False)
        rets.iloc[nan_dates] = np.nan
    return signals, rets


def test_smear_pools_by_calendar_not_ordinal(monkeypatch):
    # Scattered all-NaN dates make ordinal horizon rows refer to different dates. Only
    # the common decision-date intersection may enter the pooled t-statistic.
    signals, rets = _smear_panel(n_nan=25)
    captured = {}
    orig = lookahead.newey_west_tstat

    def spy(x, lags=None):
        s = pd.Series(x)
        captured["n"] = int(s.notna().sum())
        captured["index_type"] = type(s.index).__name__
        captured["monotonic"] = bool(s.index.is_monotonic_increasing)
        return orig(x, lags)

    monkeypatch.setattr(lookahead, "newey_west_tstat", spy)
    st = lookahead._innovation_forward_ic(signals, rets, 10)
    assert st is not None
    assert captured["index_type"] == "DatetimeIndex"    # calendar-indexed, not ordinal
    assert captured["monotonic"]
    assert captured["n"] == 89                          # common decision dates, not 240 ordinal rows
    assert st["n_common_dates"] == 89
    assert st["n_dates"] == 249                         # per-horizon count kept


def test_smear_dense_panel_alignment_unchanged():
    # Honest guard: on a dense panel the good-masks are prefixes, so the
    # calendar-aligned common count equals the longest horizon's own count
    # (ordinal == calendar there, so the pinned dense-panel smear values
    # do not depend on the alignment).
    signals, rets = _smear_panel(n_nan=0)
    st = lookahead._innovation_forward_ic(signals, rets, 10)
    assert st is not None
    # h=10 loses the last 9 forward dates relative to h=1
    assert st["n_common_dates"] == st["n_dates"] - 9
    assert np.isfinite(st["tstat"])


def test_smear_common_floor_skips_fail_closed():
    # Fail-closed gate: every horizon clears SMEAR_MIN_DATES alone, but
    # only 49 dates are jointly measurable - a pooled t presented on that
    # sliver would be dishonest in either direction, so the check SKIPs
    # with the numbers instead of silently passing (or convicting).
    signals, rets = _smear_panel(n_nan=35)
    art = BacktestArtifacts(signals=signals, asset_returns=rets)
    art.validate()
    r = lookahead._smeared_forward_ic(art.aligned(), CFG)
    assert r.status is Status.SKIP
    assert "jointly measurable" in r.message
    assert r.details["n_common_dates"] == 49
    assert r.details["n_common_dates"] < lookahead.SMEAR_MIN_DATES
