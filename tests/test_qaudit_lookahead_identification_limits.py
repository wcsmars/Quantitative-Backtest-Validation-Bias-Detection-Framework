"""Lookahead identification limits and verdicts under incomplete evidence.

Trailing risk controls, lagged multi-sleeve books, and intermittent effects
exercise scope boundaries. Exogenous risk variables cannot be identified
from asset returns alone, and unresolved cases remain explicit in messages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import lookahead
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import momentum_signal, simulate_market
from qaudit.types import Severity, Status

CFG = AuditConfig()


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _rankw(df):
    r = df.rank(axis=1)
    r = r.sub(r.mean(axis=1), axis=0)
    return r.div(r.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


def _norm(pos):
    return pos.div(pos.abs().sum(axis=1).replace(0.0, np.nan),
                   axis=0).fillna(0.0)


def _arts(rets, sig, pos, lag=1):
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=lag)
    art.validate()
    return art.aligned()


def _lag2_book(seed, lag, overlay_fn=None, w=0.20, bleed_w=0.0,
               mom_loading=0.15):
    """A 25-name, 750-date market with rank weights at the declared lag, plus optional
    causal overlays or same-bar blends."""
    mkt = simulate_market(seed=seed, n_assets=25, n_periods=750,
                          mom_loading=mom_loading)
    rets = mkt["returns"]
    sig = momentum_signal(rets)
    pos = _rankw(sig.shift(lag))
    if overlay_fn is not None:
        pos = (1 - w) * pos + w * overlay_fn(rets)
    if bleed_w:
        pos = (1 - abs(bleed_w)) * pos + bleed_w * _rankw(sig)
    return _arts(rets, sig, _norm(pos), lag=lag)


def _vol_overlay_book(seed, frac, leak, channel):
    """A momentum book with trailing inverse-volatility sizing and a same-bar return or
    signal channel active on a fraction of dates."""
    mkt = simulate_market(n_assets=40, n_periods=1200, seed=seed,
                          mom_loading=0.0)
    rets, sig = mkt["returns"], momentum_signal(mkt["returns"])
    lg = sig.shift(1).rank(axis=1)
    w = lg.sub(lg.mean(axis=1), axis=0)
    w = w / rets.rolling(20, min_periods=20).std().shift(1)
    pos = _norm(w)
    if leak:
        src = rets if channel == "ret" else sig
        R = _norm(src.rank(axis=1).sub(src.rank(axis=1).mean(axis=1), axis=0))
        rng = np.random.default_rng(seed * 7 + 1)
        m = np.zeros(len(rets), bool)
        m[rng.choice(len(rets), int(frac * len(rets)), replace=False)] = True
        L = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
        L.iloc[m] = R.iloc[m] * leak
        pos = _norm(pos + L)
    return _arts(rets, sig, pos)


# ---------------------------------------------------------------------------
# fwd_vol_window > 15 must not crash the module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_market():
    # Module-level, built once for the class below: a class-scoped fixture
    # defined as an instance method is deprecated (pytest 9
    # PytestRemovedIn10Warning - every test gets a fresh instance while the
    # fixture runs once per class), and hard-fails on pytest 10.
    return simulate_market(n_assets=20, n_periods=420, seed=5)


class TestFwdVolWindowNoCrash:
    @pytest.mark.parametrize("w", [20, 60])
    def test_window_over_15_runs_all_eight_checks(self, small_market, w):
        # Each rolling control must cap min_periods at its own window length, including
        # short controls under a large configured risk window.
        rets = small_market["returns"]
        sig = momentum_signal(rets)
        art = _arts(rets, sig, _rankw(sig.shift(1)))
        out = lookahead.run(art, AuditConfig(fwd_vol_window=w))
        assert len(out) == 8
        assert all(r.status is not Status.ERROR for r in out)

    def test_planted_forward_sizer_still_convicts_at_w20(self):
        # capping min_periods at each control's window must not cost
        # detection at large configured windows: a pure forward-20-bar
        # inverse sizer still convicts
        # (30 names: the 16-control grid's names floor is 19, so the
        # 20-name crash fixture above is too thin for the statistic -
        # measured here -0.215, NW t -12.8 over 659 dates)
        mkt = simulate_market(n_assets=30, n_periods=700, seed=5)
        rets = mkt["returns"]
        sig = momentum_signal(rets)
        fwd = rets.rolling(20, min_periods=20).std().shift(-20)
        pos = _norm(_rankw(sig.shift(1)) / fwd.clip(lower=1e-4))
        r = lookahead._future_vol_sizing(
            _arts(rets, sig, pos), AuditConfig(fwd_vol_window=20))
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert r.details["mean_partial_corr"] < lookahead.FWD_VOL_PARTIAL_FAIL

    def test_default_window_unchanged(self):
        # min(w, win) must be inert at the calibrated default w=10
        assert int(AuditConfig().fwd_vol_window) == 10
        assert min(10, min(lookahead.FWD_VOL_CONTROL_WINDOWS)) == 10


# ---------------------------------------------------------------------------
# same_bar_bleed at declared lag >= 2: honest multi-component books
# pass, bar-t bleeds still convict, lag-1 stamp-binding intact
# ---------------------------------------------------------------------------

class TestBleedLag2Basis:
    def test_honest_lag2_reversal_overlay_passes(self):
        # A lawful one-day reversal overlay contributes 20% of the book at declared lag
        # two.
        r = lookahead._same_bar_bleed(
            _lag2_book(1000, 2, lambda rets: _rankw(-rets.shift(1))), CFG)
        assert r.status is Status.PASS
        assert abs(r.details["mean_partial_corr"]) < CFG.bleed_partial_fail

    def test_honest_lag2_embargoed_overlay_passes(self):
        # The overlay uses only data through t-2, matching the declared lag-two
        # contract.
        r = lookahead._same_bar_bleed(
            _lag2_book(1000, 2,
                       lambda rets: _rankw(rets.rolling(5).mean().shift(2))),
            CFG)
        assert r.status is Status.PASS

    def test_honest_lag5_batch_book_passes(self):
        # Weekly-batch lag-five book with its overlay embargoed through t-5.
        r = lookahead._same_bar_bleed(
            _lag2_book(1000, 5,
                       lambda rets: _rankw(rets.rolling(5).mean().shift(5))),
            CFG)
        assert r.status is Status.PASS

    def test_genuine_bleed_at_lag2_flagged_scoped(self):
        # At declared lags of two or more, causal multi-sleeve books and same-bar
        # blends can overlap after purging. That statistic alone supports a not-
        # separable WARN, while stamp and discretized paths retain their own evidence
        # rules.
        r = lookahead._same_bar_bleed(_lag2_book(1000, 2, bleed_w=0.15), CFG)
        assert r.status is Status.WARN
        assert r.severity is Severity.HIGH
        assert "not separable" in r.message
        assert r.details["mean_partial_corr"] > 0.3

    def test_negated_bleed_at_lag2_flagged_scoped(self):
        # The two-sided gate retains a warning under lag-two separability limits.
        r = lookahead._same_bar_bleed(_lag2_book(1000, 2, bleed_w=-0.15), CFG)
        assert r.status is Status.WARN
        assert r.details["mean_partial_corr"] < -0.3

    def test_prelagged_export_at_lag1_still_convicts(self):
        # A pre-lagged signal used on its export row violates the declared lag-one end-
        # of-bar contract. Remap it to the prior row or shift positions forward; zero
        # lag is invalid.
        mkt = simulate_market(seed=0, n_assets=25, n_periods=750)
        rets = mkt["returns"]
        export = momentum_signal(rets).shift(1)      # stamped at trade date
        pos = _norm(0.8 * _rankw(export) + 0.2 * _rankw(-rets.shift(1)))
        r = lookahead._same_bar_bleed(_arts(rets, export, pos), CFG)
        assert r.status is Status.FAIL
        assert r.details["mean_partial_corr"] > 0.5

    def test_lag2_basis_constants_pinned(self):
        assert lookahead.BLEED_LAG2_TRAIL_WINDOWS == (5, 10)


# ---------------------------------------------------------------------------
# same_bar_bleed extreme-date escalation (intermittent-dilution FN)
# ---------------------------------------------------------------------------

class TestBleedExtremeDateEscalation:
    def test_intermittent_signal_bleed_escalates(self):
        # Same-bar signal bleed live on 7% of dates measures mean +0.07
        # (under the 0.10 bar) at NW t ~8; the extreme-date mass gate must
        # WARN (measured frac ~0.07 vs honest max 0.004).
        r = lookahead._same_bar_bleed(
            _vol_overlay_book(13, 0.07, 3.0, "sig"), CFG)
        assert r.status is Status.WARN
        assert r.severity is Severity.HIGH
        assert abs(r.details["mean_partial_corr"]) < CFG.bleed_partial_fail
        assert r.details["n_extreme_dates"] >= 40
        assert r.details["extreme_frac"] >= lookahead.BLEED_EXTREME_FRAC
        assert "INTERMITTENT" in r.message

    def test_honest_anti_tilt_high_t_does_not_escalate(self):
        # The causal 0.8*mom20 - 0.2*mom5 book can have a large pooled t-statistic with
        # consistently small per-date correlations. An extreme-date count of zero
        # prevents t-only escalation.
        mkt = simulate_market(seed=11)
        rets = mkt["returns"]
        mom = momentum_signal(rets)
        mom5 = momentum_signal(rets, window=5)
        pos = _norm(0.80 * _rankw(mom.shift(1)) - 0.20 * _rankw(mom5.shift(1)))
        r = lookahead._same_bar_bleed(_arts(rets, mom, pos), CFG)
        assert r.status is Status.PASS
        assert abs(r.details["nw_tstat"]) > 6.0       # t alone would misfire
        assert r.details["n_extreme_dates"] < lookahead.BLEED_EXTREME_MIN_DATES

    def test_escalation_is_lag1_scoped(self):
        # Boundary pin of a documented hole, not an endorsement: the same
        # intermittent bleed declared at lag 2 measures mean +0.066 / ext
        # frac 0.064 / t +8.3 - yet must not escalate, because the lag>=2
        # statistic's 10-column joint projection consumes most of a small
        # cross-section's df and inflates honest Fisher-z tails (honest
        # lag-2/5 overlay books measured extreme frac to 0.0152 vs the
        # 0.02 gate - 1.3x margin, too thin to judge). If this flips to
        # WARN, the escalation was extended past its lag-1 calibration:
        # re-measure honest lag>=2 overlay books (extreme frac up to
        # 0.0152 vs the 0.02 gate) before accepting it.
        mkt = simulate_market(n_assets=40, n_periods=1200, seed=13,
                              mom_loading=0.0)
        rets, sig = mkt["returns"], momentum_signal(mkt["returns"])
        lg = sig.shift(2).rank(axis=1)
        w = lg.sub(lg.mean(axis=1), axis=0)
        w = w / rets.rolling(20, min_periods=20).std().shift(1)
        base = _norm(w)
        R = _norm(sig.rank(axis=1).sub(sig.rank(axis=1).mean(axis=1), axis=0))
        rng = np.random.default_rng(13 * 7 + 1)
        m = np.zeros(len(rets), bool)
        m[rng.choice(len(rets), int(0.07 * len(rets)), replace=False)] = True
        L = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
        L.iloc[m] = R.iloc[m] * 3.0
        pos = _norm(base + L)
        r = lookahead._same_bar_bleed(_arts(rets, sig, pos, lag=2), CFG)
        assert r.status is Status.PASS
        assert r.details["extreme_frac"] > lookahead.BLEED_EXTREME_FRAC

    def test_escalation_constants_pinned(self):
        assert lookahead.BLEED_EXTREME_DATE_Z == 4.0
        assert lookahead.BLEED_EXTREME_FRAC == 0.02
        assert lookahead.BLEED_EXTREME_MIN_DATES == 5


# ---------------------------------------------------------------------------
# same_bar_return_loading t-escalation (diluted magnitude-leak FN)
# ---------------------------------------------------------------------------

class TestReturnLoadingTstatEscalation:
    def test_intermittent_return_leak_escalates(self):
        # return-magnitude leak on 7% of dates, variance-diluted by the
        # honest inverse-vol overlay - mean ~0.06 (under the 0.08 WARN bar)
        # at NW t ~7 must WARN via the t escalation
        r = lookahead._same_bar_return_loading(
            _vol_overlay_book(13, 0.07, 3.0, "ret"), CFG)
        assert r.status is Status.WARN
        assert abs(r.details["mean_corr"]) < lookahead.RETURN_LOADING_WARN
        assert abs(r.details["nw_tstat"]) >= lookahead.RETURN_LOADING_TSTAT_ESC
        assert "escalation gate" in r.message

    def test_uniform_small_tilt_escalates(self):
        # variance-dilution variant: a ~1.5%-gross same-bar rank tilt on
        # every date measures mean ~0.06 at NW t ~13 (a 13-sigma loading
        # under the level bar) - escalates on t alone
        mkt = simulate_market(n_assets=40, n_periods=1200, seed=13,
                              mom_loading=0.0)
        rets, sig = mkt["returns"], momentum_signal(mkt["returns"])
        lg = sig.shift(1).rank(axis=1)
        w = lg.sub(lg.mean(axis=1), axis=0)
        w = w / rets.rolling(20, min_periods=20).std().shift(1)
        R = _norm(rets.rank(axis=1).sub(rets.rank(axis=1).mean(axis=1),
                                        axis=0))
        pos = _norm(_norm(w) + R * 0.015)
        r = lookahead._same_bar_return_loading(_arts(rets, sig, pos), CFG)
        assert r.status is Status.WARN
        assert abs(r.details["mean_corr"]) < lookahead.RETURN_LOADING_WARN
        assert abs(r.details["nw_tstat"]) > 10.0

    def test_honest_vol_overlay_book_stays_clean(self):
        # The identical trailing-volatility overlay without a leak must pass both same-
        # bar checks.
        art = _vol_overlay_book(13, 0.0, 0.0, "ret")
        s = lookahead._same_bar_return_loading(art, CFG)
        b = lookahead._same_bar_bleed(art, CFG)
        assert s.status is Status.PASS
        assert abs(s.details["nw_tstat"]) < lookahead.RETURN_LOADING_TSTAT_ESC
        assert b.status is Status.PASS


# ---------------------------------------------------------------------------
# gray-band SKIP note: message must not claim "stays below" a bar the
# mean cleared
# ---------------------------------------------------------------------------

def _block_disc_bleed(seed, n_dates=90, n_assets=25, w_bleed=0.55,
                      block=(30, 62)):
    """Block bleed construction: 55% same-bar bleed live on a contiguous mid-sample
    block (pipeline change), round-to-0.1 signal export -> judged gray band
    with mean above the 0.35 bar but block-deflated NW t under the doubled
    co-gate."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_dates)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    s = np.zeros((n_dates, n_assets))
    s[0] = rng.standard_normal(n_assets)
    for t in range(1, n_dates):
        s[t] = 0.9 * s[t - 1] + np.sqrt(1 - 0.81) * rng.standard_normal(n_assets)

    def rw(row):
        r = pd.Series(row).rank()
        w = r - r.mean()
        return (w / w.abs().sum()).to_numpy()

    pos = np.zeros_like(s)
    for t in range(1, n_dates):
        honest = rw(s[t - 1])
        if block[0] <= t < block[1]:
            pos[t] = (1 - w_bleed) * honest + w_bleed * rw(s[t])
        else:
            pos[t] = honest
    rets = pd.DataFrame(rng.standard_normal((n_dates, n_assets)) * 0.01,
                        index=idx, columns=cols)
    return _arts(rets,
                 pd.DataFrame(np.round(s, 1), index=idx, columns=cols),
                 pd.DataFrame(pos, index=idx, columns=cols))


class TestGrayBandMessageTruth:
    def test_above_bar_t_failed_says_unresolved(self):
        # seed 1: mean +0.374 > 0.35 with NW t 4.3 < 8.0 -> SKIP; the note
        # must not claim the mean "stays below" the bar it printed itself
        # clearing
        r = lookahead._same_bar_bleed(_block_disc_bleed(1), CFG)
        assert r.status is Status.SKIP
        d = r.details
        assert abs(d["mean_partial_corr"]) > lookahead.BLEED_DISC_PARTIAL_FAIL
        assert abs(d["nw_tstat"]) <= d["tstat_gate"]
        assert "stays below" not in r.message
        assert "CLEARS" in r.message
        assert "UNRESOLVED" in r.message
        # numeric details are independent of the wording branch
        assert d["tstat_gate"] == 2.0 * CFG.bleed_tstat
    # The stays-below wording for means genuinely under the bar is pinned
    # by tests/test_qaudit_lookahead_directionality.py::
    # test_honest_rounded_export_skips_with_measured_stat.


# ---------------------------------------------------------------------------
# evidentiary PASS wording: a mean beyond a verdict bar with the t
# co-gate failed must not read as an affirmative all-clear
# ---------------------------------------------------------------------------

def _garch_returns(rng, n_dates, n_assets):
    a, b, w = 0.09, 0.89, 1e-5
    h = np.full(n_assets, w / (1 - a - b))
    out = np.empty((n_dates, n_assets))
    for t in range(n_dates):
        eps = rng.standard_normal(n_assets)
        out[t] = np.sqrt(h) * eps
        h = w + a * out[t] ** 2 + b * h
    return out


class TestEvidentiaryPassWording:
    def test_fvs_intermittent_sizer_not_exonerated(self):
        # Intermittent forward-vol sizer (deterministic): 100% inverse-forward-
        # vol sizing over a contiguous 30% block, honest trailing sizing
        # elsewhere -> mean -0.096 (beyond the -0.075 FAIL bar) with NW t
        # -3.1 missing the -4.0 co-gate. Status stays PASS (the co-gate is
        # validated FP control) but the message must be evidentiary.
        rng = np.random.default_rng(1)
        n_dates, n_assets = 260, 30
        idx = pd.bdate_range("2023-01-02", periods=n_dates)
        cols = [f"A{i:02d}" for i in range(n_assets)]
        rets = pd.DataFrame(_garch_returns(rng, n_dates, n_assets),
                            index=idx, columns=cols)
        sig = pd.DataFrame(rng.standard_normal((n_dates, n_assets)),
                           index=idx, columns=cols)
        fwd_vol = rets.rolling(10, min_periods=10).std().shift(-10)
        trail = rets.rolling(20, min_periods=10).std().shift(1)
        ranks = sig.rank(axis=1).sub(sig.rank(axis=1).mean(axis=1), axis=0)
        honest = np.sign(ranks) * (ranks.abs() / trail)
        leaky = np.sign(ranks) * (ranks.abs() / fwd_vol)
        b0 = int(n_dates * 0.4)
        b1 = b0 + int(n_dates * 0.3)
        pos = honest.copy()
        pos.iloc[b0:b1] = leaky.iloc[b0:b1]
        r = lookahead._future_vol_sizing(
            _arts(rets, sig, _norm(pos)), CFG)
        assert r.status is Status.PASS
        assert r.details["mean_partial_corr"] <= lookahead.FWD_VOL_PARTIAL_FAIL
        assert abs(r.details["nw_tstat"]) < r.details["tstat_gate"]
        assert "NOT exonerated" in r.message
        assert "no anticipation" not in r.message

    def test_fvs_clean_book_keeps_affirmative_scoped_wording(self):
        # Inside both warning thresholds, affirmative wording remains scoped to
        # returns-derived risk controls.
        mkt = simulate_market(n_assets=30, n_periods=700, seed=5)
        rets = mkt["returns"]
        sig = momentum_signal(rets)
        trail = rets.rolling(20, min_periods=10).std().shift(1)
        pos = _norm(_rankw(sig.shift(1)) / trail.clip(lower=1e-4))
        r = lookahead._future_vol_sizing(_arts(rets, sig, pos), CFG)
        assert r.status is Status.PASS
        assert lookahead.FWD_VOL_PARTIAL_WARN < r.details["mean_partial_corr"] \
            < lookahead.FWD_VOL_PARTIAL_WARN_POS
        assert "no anticipation" in r.message
        assert "RETURNS-DERIVED" in r.message

    def test_bleed_block_leak_above_bar_not_exonerated(self):
        # sibling guard: contiguous-block same-bar bleed on a book with a
        # 15% exogenous overlay (keeps non-block dates measurable) ->
        # mean +0.20 > 0.10 bar with NW t 3.5 < 4.0; per-date extremes
        # exist but the escalation's own t co-gate correctly holds back -
        # the PASS must say unresolved, not "no same-bar information"
        rng = np.random.default_rng(1)
        n_dates, n_assets = 120, 25
        idx = pd.bdate_range("2024-01-02", periods=n_dates)
        cols = [f"A{i:02d}" for i in range(n_assets)]
        s = np.zeros((n_dates, n_assets))
        s[0] = rng.standard_normal(n_assets)
        for t in range(1, n_dates):
            s[t] = 0.9 * s[t - 1] + np.sqrt(1 - 0.81) * rng.standard_normal(n_assets)
        noise = rng.standard_normal((n_dates, n_assets))

        def rw(row):
            r_ = pd.Series(row).rank()
            w_ = r_ - r_.mean()
            return (w_ / w_.abs().sum()).to_numpy()

        pos = np.zeros_like(s)
        for t in range(1, n_dates):
            honest = 0.85 * rw(s[t - 1]) + 0.15 * rw(noise[t - 1])
            if 40 <= t < 70:
                pos[t] = 0.75 * honest + 0.25 * rw(s[t])
            else:
                pos[t] = honest
        rets = pd.DataFrame(rng.standard_normal((n_dates, n_assets)) * 0.01,
                            index=idx, columns=cols)
        r = lookahead._same_bar_bleed(
            _arts(rets, pd.DataFrame(s, index=idx, columns=cols),
                  pd.DataFrame(pos, index=idx, columns=cols)), CFG)
        assert r.status is Status.PASS
        assert abs(r.details["mean_partial_corr"]) > CFG.bleed_partial_fail
        assert abs(r.details["nw_tstat"]) <= r.details["tstat_gate"]
        assert "not exonerated" in r.message
        assert "carry no same-bar signal information" not in r.message

    def test_srl_block_leak_above_bar_not_exonerated(self):
        # sibling guard: return-magnitude leak on a contiguous 25% block of
        # a vol-overlay book -> mean +0.27 > 0.08 with NW t 3.7 < 4.0
        mkt = simulate_market(n_assets=25, n_periods=160, seed=1,
                              mom_loading=0.0)
        rets = mkt["returns"]
        sig = momentum_signal(rets)
        lg = sig.shift(1).rank(axis=1)
        w = lg.sub(lg.mean(axis=1), axis=0)
        w = w / rets.rolling(20, min_periods=20).std().shift(1)
        base = _norm(w)
        R = _norm(rets.rank(axis=1).sub(rets.rank(axis=1).mean(axis=1),
                                        axis=0))
        L = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
        L.iloc[60:100] = R.iloc[60:100] * 1.0
        pos = _norm(base + L)
        r = lookahead._same_bar_return_loading(_arts(rets, sig, pos), CFG)
        assert r.status is Status.PASS
        assert abs(r.details["mean_corr"]) > lookahead.RETURN_LOADING_WARN
        assert abs(r.details["nw_tstat"]) < r.details["tstat_gate"]
        assert "not exonerated" in r.message
        assert "no same-bar return loading" not in r.message


# ---------------------------------------------------------------------------
# future_vol_sizing hard limitation: non-return-derived per-name risk
# inputs (implied vol / range / HF-realized) - a documented
# identification limit
# ---------------------------------------------------------------------------

def _sv_market(seed, n_assets=25, n_periods=800, phi=0.97, tau=0.15,
               mu_vol=0.015):
    """Per-name stochastic-vol market: latent
    log-variance AR(1); the true vol state is observable to the book only
    through a non-return channel."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_periods)
    mu = 2.0 * np.log(mu_vol)
    sd_h = tau / np.sqrt(1.0 - phi * phi)
    h = mu + sd_h * rng.standard_normal(n_assets)
    hs = np.empty((n_periods, n_assets))
    rets = np.empty((n_periods, n_assets))
    for t in range(n_periods):
        hs[t] = h
        rets[t] = np.exp(h / 2.0) * rng.standard_normal(n_assets)
        h = mu + phi * (h - mu) + tau * rng.standard_normal(n_assets)
    cols = [f"A{i:03d}" for i in range(n_assets)]
    return (pd.DataFrame(rets, index=dates, columns=cols),
            pd.DataFrame(np.exp(hs / 2.0), index=dates, columns=cols))


class TestFvsNonReturnRiskLimitation:
    def test_documented_limitation_implied_vol_sizer_still_convicts(self):
        # Known identification limit: causal sizing on an exact exogenous t-1
        # volatility state can be flagged because returns-derived controls cannot span
        # it. Resolving this case requires additional declared risk-model evidence.
        rets, true_vol = _sv_market(500)
        sig = pd.DataFrame(
            np.random.default_rng(501).standard_normal(rets.shape),
            index=rets.index, columns=rets.columns)
        inv = 1.0 / true_vol.shift(1).replace(0.0, np.nan)
        pos = inv.div(inv.abs().sum(axis=1), axis=0).fillna(0.0)
        r = lookahead._future_vol_sizing(_arts(rets, sig, pos), CFG)
        assert r.status is Status.FAIL
        assert r.details["mean_partial_corr"] < lookahead.FWD_VOL_PARTIAL_FAIL
        # ... and the conviction is honest about its own scope:
        assert "RETURNS-DERIVED" in r.message
        assert "implied vol" in r.message
        assert "cannot distinguish" in r.message
        assert "implied vol" in r.remediation

    def test_limitation_documented_in_module_docstring(self):
        doc = lookahead.__doc__
        assert "HARD LIMITATION" in doc
        assert "implied vol" in doc
