"""Same-bar separability, genuine alpha, static tilts, and event-gated signals.

Lagged return dependence and book structure constrain what a partial
correlation proves. Honest constructions and planted blends distinguish
identified timing defects from unresolved structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qaudit.checks import lookahead
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (make_smeared_leak, momentum_signal,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _rankw(df):
    r = df.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    return w.div(w.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


def _norm(pos):
    return pos.div(pos.abs().sum(axis=1).replace(0.0, np.nan),
                   axis=0).fillna(0.0)


def _arts(rets, sig, pos, lag=1):
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=lag)
    art.validate()
    return art.aligned()


def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def _noise_market(seed, n=900, m=40):
    """Pure-noise panel (no cross-sectional alpha): any same-bar coupling
    an honest book shows here is purely structural."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"A{i:03d}" for i in range(m)]
    beta = rng.uniform(0.7, 1.3, m)
    mkt = rng.normal(2e-4, 0.010, n)
    return pd.DataFrame(rng.standard_normal((n, m)) * 0.02
                        + np.outer(mkt, beta), index=dates, columns=cols)


def _rev_multisleeve(seed, w, overlay, lag, n=900, m=40, w_ov=0.3):
    """A rolling-reversal primary and a causal second sleeve, with every input embargoed to
    the declared lag."""
    rets = _noise_market(seed, n, m)
    sig = _zx(-rets.rolling(w, min_periods=w).mean())
    makers = {
        "ewma2rev": lambda r: -r.ewm(halflife=2, min_periods=5).mean(),
        "subrev": lambda r: -r.rolling(max(2, w - 2), min_periods=2).mean(),
        "r1rev": lambda r: -r,
    }
    pos = ((1 - w_ov) * _rankw(sig.shift(lag))
           + w_ov * _rankw(makers[overlay](rets).shift(lag)))
    return _arts(rets, sig, _norm(pos), lag)


# ---------------------------------------------------------------------------
# same_bar_bleed separability scoping
# ---------------------------------------------------------------------------

class TestBleedSeparabilityScoping:
    def test_honest_short_window_multisleeve_lag2_not_critical(self):
        # A causal multi-sleeve book can exceed the raw threshold through lagged
        # innovation content. A purged component below its threshold supports a scoped
        # warning.
        r = lookahead._same_bar_bleed(
            _rev_multisleeve(50, 5, "ewma2rev", lag=2), CFG)
        assert r.status is Status.WARN
        assert r.severity is Severity.HIGH
        assert "not separable" in r.message
        assert "built from the signal of the bar it trades" not in r.message
        d = r.details
        assert abs(d["mean_partial_corr"]) > CFG.bleed_partial_fail
        assert abs(d["sep_mean_partial_corr"]) < lookahead.BLEED_SEP_PARTIAL_FAIL
        assert d["f_t1_explained_share"] > 0.25
        # remediation must point at the honest fix: declare the composite
        assert "composite" in (r.remediation or "")

    def test_honest_short_window_multisleeve_lag1_not_critical(self):
        # The causal lag-one twin has a return-fresh innovation, so the purge includes
        # lag-one returns. Pre-lagged exports retain their separate timestamp rule.
        r = lookahead._same_bar_bleed(
            _rev_multisleeve(50, 5, "subrev", lag=1), CFG)
        assert r.status is Status.WARN
        assert "not separable" in r.message
        assert r.details["sep_fresh_purge"] is True

    def test_honest_w10_subwindow_overlay_not_critical(self):
        # a w=10 reversal with an 8d sub-window overlay at lag 2 needs the
        # return-value ladder to reach window+2 (cap 12) with values ahead
        # of rank columns; a ladder stopping at lag 8 convicts it through
        # the purge (sep +0.33)
        r = lookahead._same_bar_bleed(
            _rev_multisleeve(50, 10, "subrev", lag=2), CFG)
        assert r.status is not Status.FAIL
        assert abs(r.details["sep_mean_partial_corr"]) \
            < lookahead.BLEED_SEP_PARTIAL_FAIL

    def test_honest_momentum_blend_lag1_not_critical(self):
        # Two lawfully lagged momentum sleeves can forecast current returns through
        # trailing information; the lagged-return control grid must explain that
        # coupling.
        mkt = simulate_market(seed=0)
        rets = mkt["returns"]
        mom20 = momentum_signal(rets)
        mom10 = momentum_signal(rets, window=10)
        pos = _norm(0.5 * positions_from_signals(mom20, 1)
                    + 0.5 * positions_from_signals(mom10, 1))
        r = lookahead._same_bar_bleed(_arts(rets, mom20, pos), CFG)
        assert r.status is Status.WARN
        assert r.severity is Severity.HIGH
        assert "not separable" in r.message
        assert r.details["f_t1_explained_share"] > 0.5

    def test_composite_declaration_passes(self):
        # the remediation the WARN points to must actually work: declaring
        # the blended continuous composite score clears the book (measured
        # +0.065..+0.078, under the 0.10 bar)
        mkt = simulate_market(seed=0)
        rets = mkt["returns"]
        mom20 = momentum_signal(rets)
        mom10 = momentum_signal(rets, window=10)
        comp = 0.5 * mom20 + 0.5 * mom10
        pos = _norm(0.5 * positions_from_signals(mom20, 1)
                    + 0.5 * positions_from_signals(mom10, 1))
        r = lookahead._same_bar_bleed(_arts(rets, comp, pos), CFG)
        assert r.status is Status.PASS

    def test_fifteen_pct_bleed_still_critical(self):
        # TP retention: a genuine 15% same-bar blend keeps CRITICAL - the
        # purged bar-t statistic clears its own bar (measured sep +0.56)
        mkt = simulate_market(seed=2)
        rets = mkt["returns"]
        mom = momentum_signal(rets)
        pos = _norm(0.85 * positions_from_signals(mom, 1)
                    + 0.15 * positions_from_signals(mom, 0))
        r = lookahead._same_bar_bleed(_arts(rets, mom, pos), CFG)
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert abs(r.details["sep_mean_partial_corr"]) \
            > lookahead.BLEED_SEP_PARTIAL_FAIL

    def test_negated_bleed_still_critical(self):
        mkt = simulate_market(seed=2)
        rets = mkt["returns"]
        mom = momentum_signal(rets)
        pos = _norm(0.85 * positions_from_signals(mom, 1)
                    - 0.15 * positions_from_signals(mom, 0))
        r = lookahead._same_bar_bleed(_arts(rets, mom, pos), CFG)
        assert r.status is Status.FAIL
        assert r.details["sep_mean_partial_corr"] \
            < -lookahead.BLEED_SEP_PARTIAL_FAIL

    def test_return_orthogonal_bleed_keeps_stamp_verdict(self):
        # A return-orthogonal AR score takes the freshness path; its negated 15% blend
        # must be detected from the raw statistic.
        mkt = simulate_market(seed=2)
        rets = mkt["returns"]
        rng = np.random.default_rng(7)
        eps = rng.standard_normal(rets.shape)
        x = np.empty_like(eps)
        x[0] = eps[0]
        for t in range(1, len(rets)):
            x[t] = 0.9 * x[t - 1] + np.sqrt(1 - 0.81) * eps[t]
        sigx = _zx(pd.DataFrame(x, index=rets.index,
                                columns=rets.columns).where(rets.notna()))
        pos = _norm(0.85 * positions_from_signals(sigx, 1)
                    - 0.15 * positions_from_signals(sigx, 0))
        r = lookahead._same_bar_bleed(_arts(rets, sigx, pos), CFG)
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert "return-orthogonal" in r.message
        assert abs(r.details["sep_signal_freshness"]) \
            < lookahead.BLEED_SEP_FRESH_MIN

    def test_honest_deep_window_heavy_overlay_not_critical(self):
        # A 50/50 mixture of 25-day and 23-day reversal sleeves is nearly collinear and
        # uses only t-1 data. Its elevated purged statistic must not trigger a timing
        # failure.
        r = lookahead._same_bar_bleed(
            _rev_multisleeve(51, 25, "subrev", lag=1, w_ov=0.5), CFG)
        assert r.status is Status.WARN
        assert "not separable" in r.message
        assert abs(r.details["sep_mean_partial_corr"]) \
            < lookahead.BLEED_SEP_PARTIAL_FAIL

    def test_sep_constants_pinned(self):
        assert lookahead.BLEED_SEP_MAX_LAGS == 8
        assert lookahead.BLEED_SEP_RET_MAX_LAGS == 12
        assert lookahead.BLEED_SEP_ACF_FLOOR == 0.10
        assert lookahead.BLEED_SEP_FRESH_MIN == 0.10
        # 0.47: honest lag-1 plateau 0.419 (50/50 w30 double-reversal,
        # x1.12) vs the 15% conviction class at 0.55+ (x1.17); 8% blends
        # straddle the bar by seed (0.47-0.52). Do not move without
        # re-measuring those three classes across seeds (_rev_multisleeve
        # and the 15% blends in this file; an 8% blend built the same way).
        assert lookahead.BLEED_SEP_PARTIAL_FAIL == 0.47


# ---------------------------------------------------------------------------
# same_bar_return_loading: honest genuine-alpha scope on the escalation
# ---------------------------------------------------------------------------

def _ar1_market(seed, n, m, phi):
    """Cross-sectional AR(1) idio market: rank(r_{t-1}) carries genuine
    1-day alpha at IC ~ phi * idio share, inside
    lookahead.RETURN_LOADING_ESC_MEAN_FLOOR (the check's honest
    genuine-alpha band)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)
    assets = [f"S{i:03d}" for i in range(m)]
    beta = rng.uniform(0.6, 1.4, m)
    mkt = rng.normal(3e-4, 0.011, n)
    sigma = rng.uniform(0.012, 0.030, m)
    eps = rng.standard_normal((n, m)) * sigma
    idio = np.zeros((n, m))
    for t in range(1, n):
        idio[t] = phi * idio[t - 1] + eps[t]
    return pd.DataFrame(idio + np.outer(mkt, beta), index=dates,
                        columns=assets)


def _genuine_alpha_book(seed, n, m, phi, w_ov=0.3):
    """Declared trailing-20-day momentum plus a causal fast rank sleeve with genuine one-
    bar alpha. Every position input is available through t-1."""
    rets = _ar1_market(seed, n, m, phi)
    sig = _zx(rets.rolling(20, min_periods=20).mean())
    pos = _norm((1 - w_ov) * _rankw(sig.shift(1))
                + w_ov * _rankw(rets.shift(1)))
    return _arts(rets, sig, pos)


def _vol_overlay_leak_book(seed, frac, leak):
    """An intermittent return-channel leak on a trailing inverse-volatility book."""
    mkt = simulate_market(n_assets=40, n_periods=1200, seed=seed,
                          mom_loading=0.0)
    rets, sig = mkt["returns"], momentum_signal(mkt["returns"])
    lg = sig.shift(1).rank(axis=1)
    w = lg.sub(lg.mean(axis=1), axis=0)
    w = w / rets.rolling(20, min_periods=20).std().shift(1)
    pos = _norm(w)
    if leak:
        R = _norm(rets.rank(axis=1).sub(rets.rank(axis=1).mean(axis=1),
                                        axis=0))
        rng = np.random.default_rng(seed * 7 + 1)
        rows = np.zeros(len(rets), bool)
        rows[rng.choice(len(rets), int(frac * len(rets)),
                        replace=False)] = True
        L = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
        L.iloc[rows] = R.iloc[rows] * leak
        pos = _norm(pos + L)
    return _arts(rets, sig, pos)


class TestReturnLoadingAlphaScope:
    def test_honest_genuine_alpha_10y_passes(self):
        # Long samples can give genuine weak alpha a large t-statistic. If its mean
        # stays inside the causal-alpha band, the pass must state that alignment
        # remains unresolved.
        r = lookahead._same_bar_return_loading(
            _genuine_alpha_book(5001, 2520, 40, 0.03), CFG)
        assert r.status is Status.PASS
        d = r.details
        assert abs(d["nw_tstat"]) >= lookahead.RETURN_LOADING_TSTAT_ESC
        assert abs(d["mean_corr"]) < lookahead.RETURN_LOADING_ESC_MEAN_FLOOR
        assert "UNRESOLVED" in r.message

    def test_honest_genuine_alpha_short_panel_plain_pass(self):
        # calibration-length twin: t below the gate, ordinary PASS wording
        r = lookahead._same_bar_return_loading(
            _genuine_alpha_book(5000, 756, 40, 0.03), CFG)
        assert r.status is Status.PASS

    def test_intermittent_leak_still_escalates(self):
        # The intermittent 7% return-channel leak clears the 0.055 mean floor and must
        # retain its warning.
        r = lookahead._same_bar_return_loading(
            _vol_overlay_leak_book(13, 0.07, 3.0), CFG)
        assert r.status is Status.WARN
        assert abs(r.details["mean_corr"]) \
            >= lookahead.RETURN_LOADING_ESC_MEAN_FLOOR

    def test_floor_boundary_leak_documented_fn(self):
        # A leak tuned inside the causal-alpha band cannot be distinguished from real
        # alpha with these artifacts. Its pass must retain the UNRESOLVED
        # qualification.
        r = lookahead._same_bar_return_loading(
            _vol_overlay_leak_book(14, 0.07, 3.0), CFG)
        assert r.status is Status.PASS
        assert "UNRESOLVED" in r.message
        assert abs(r.details["mean_corr"]) \
            < lookahead.RETURN_LOADING_ESC_MEAN_FLOOR

    def test_esc_constants_pinned(self):
        assert lookahead.RETURN_LOADING_TSTAT_ESC == 6.0
        assert lookahead.RETURN_LOADING_ESC_MEAN_FLOOR == 0.055


# ---------------------------------------------------------------------------
# same_bar_return_loading: static-tilt unmasking
# ---------------------------------------------------------------------------

def _static_tilt_book(seed, tilt, leak, rotate=None):
    """Momentum plus a dominant static per-asset basket, with optional same-bar return-
    magnitude leakage. Annual rotation redraws the basket each calendar year."""
    mkt = simulate_market(30, 1000, seed, 0.15, death_frac=0.12)
    rets, uni = mkt["returns"], mkt["universe"]
    s1 = momentum_signal(rets)
    base = _rankw(s1.shift(1))
    trng = np.random.default_rng(seed + 99)
    if rotate == "year":
        V = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
        years = rets.index.year
        for y in np.unique(years):
            v = trng.standard_normal(rets.shape[1])
            v = v - v.mean()
            v = v / np.abs(v).sum()
            V.loc[years == y, :] = v
    else:
        v = trng.standard_normal(rets.shape[1])
        v = v - v.mean()
        v = v / np.abs(v).sum()
        V = pd.DataFrame(np.tile(v, (len(rets), 1)), index=rets.index,
                         columns=rets.columns)
    V = V.where(uni, 0.0)
    book = base + tilt * V
    if leak:
        rr = rets.rank(axis=1)
        u = rr.sub(rr.mean(axis=1), axis=0)
        u = u.div(u.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        book = book + leak * u
    return _arts(rets, s1, _norm(book))


class TestReturnLoadingStaticUnmask:
    def test_static_tilt_evasion_fails(self):
        # tilt=20, leak=0.65: raw mean +0.024 / t 4.2 threads the
        # honest-max..gate window at SR ~2.5; the residual is ~80% one
        # static basket, and with that inert direction projected out the
        # purged corr measures ~+0.56 - CRITICAL
        r = lookahead._same_bar_return_loading(
            _static_tilt_book(0, 20.0, 0.65), CFG)
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        d = r.details
        assert d["static_share"] >= lookahead.RETURN_LOADING_STATIC_SHARE
        assert abs(d["static_purged_mean_corr"]) \
            >= lookahead.RETURN_LOADING_FAIL

    def test_static_tilt_smaller_leak_still_flagged(self):
        # attacker retuning downward stays flagged (purged corr scales
        # with leak/dynamic-residual, not leak/tilt)
        r = lookahead._same_bar_return_loading(
            _static_tilt_book(1, 20.0, 0.4), CFG)
        assert r.status in (Status.FAIL, Status.WARN)

    def test_honest_static_only_passes(self):
        r = lookahead._same_bar_return_loading(
            _static_tilt_book(0, 20.0, 0.0), CFG)
        assert r.status is Status.PASS

    def test_honest_rotating_tilt_passes(self):
        r = lookahead._same_bar_return_loading(
            _static_tilt_book(0, 20.0, 0.0, rotate="year"), CFG)
        assert r.status is Status.PASS

    def test_rotating_tilt_leak_documented_open_floor(self):
        # Documented open evasion (deliberate): a year-rotating
        # dominant tilt defeats the time-mean static estimate (measured
        # static share ~0.29 < the 0.50 gate), so the same leak under a
        # rotating basket still threads the gates (mean +0.02, t ~3.6).
        # Closing it needs a windowed/blockwise static estimate whose
        # honest FP cost is unmeasured - the comment on
        # RETURN_LOADING_STATIC_SHARE names this floor. This pin exists
        # so any future fix flips it deliberately, and so a silent
        # regression that starts convicting honest rotating-tilt books is
        # noticed (the honest guard above must stay green alongside).
        r = lookahead._same_bar_return_loading(
            _static_tilt_book(0, 20.0, 0.65, rotate="year"), CFG)
        assert r.status is Status.PASS
        assert r.details["static_share"] < lookahead.RETURN_LOADING_STATIC_SHARE

    def test_static_constants_pinned(self):
        assert lookahead.RETURN_LOADING_STATIC_SHARE == 0.50


# ---------------------------------------------------------------------------
# smeared_forward_ic: event-gated honest alphas
# ---------------------------------------------------------------------------

def _dow_gated_book(seed, kind="dow", strength=0.0035, n=1000, m=30):
    """Causal day-of-week or turn-of-month alpha estimated with an expanding event-minus-
    non-event mean and exported only on event eves. Persistent event loadings must not
    be mistaken for smeared future information."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    assets = [f"A{i:03d}" for i in range(m)]
    beta = rng.uniform(0.7, 1.3, m)
    mkt = rng.normal(2e-4, 0.010, n)
    sigma = rng.uniform(0.015, 0.025, m)
    load = rng.normal(0.0, 1.0, m)
    if kind == "dow":
        ev = np.asarray(dates.dayofweek == 0)
    else:
        month = dates.to_period("M")
        ev = np.zeros(n, bool)
        for mo in month.unique():
            idx = np.where(month == mo)[0]
            ev[idx[-2:]] = True
            ev[idx[:2]] = True
    nxt = np.roll(ev, -1)
    nxt[-1] = False
    eps = rng.normal(0.0, 1.0, (n, m)) * sigma
    rets = pd.DataFrame(eps + np.outer(ev.astype(float), load) * strength
                        + np.outer(mkt, beta), index=dates, columns=assets)
    rv = rets.to_numpy()
    cum_ev = np.zeros(m)
    cnt_ev = np.zeros(m)
    cum_ot = np.zeros(m)
    cnt_ot = np.zeros(m)
    est = np.full((n, m), np.nan)
    for t in range(n):
        x = rv[t]
        fin = np.isfinite(x)
        if ev[t]:
            cum_ev[fin] += x[fin]
            cnt_ev[fin] += 1
        else:
            cum_ot[fin] += x[fin]
            cnt_ot[fin] += 1
        ok = (cnt_ev >= 8) & (cnt_ot >= 8)
        est[t, ok] = cum_ev[ok] / cnt_ev[ok] - cum_ot[ok] / cnt_ot[ok]
    sig = pd.DataFrame(est * nxt[:, None].astype(float), index=dates,
                       columns=assets)
    art = BacktestArtifacts(signals=sig, asset_returns=rets, signal_lag=1)
    art.validate()
    return art.aligned()


class TestSmearEventGated:
    def test_honest_weekly_gated_alpha_passes(self):
        # The last-nonzero control must absorb persistent loading in an event-gated
        # causal signal.
        r = lookahead._smeared_forward_ic(_dow_gated_book(0), CFG)
        assert r.status is Status.PASS
        assert r.details["excess_mass"] < lookahead.SMEAR_MASS_WARN

    def test_honest_tom_gated_alpha_passes(self):
        # turn-of-month twin: 4-day event runs (within-run mass at h=1..4)
        r = lookahead._smeared_forward_ic(
            _dow_gated_book(0, kind="tom"), CFG)
        assert r.status is Status.PASS

    def test_diffuse_forward_smear_fails(self):
        # TP retention: a genuinely smeared leak's signal is continuous,
        # so its s1 control already equals the ffill column - the added
        # control cannot absorb the smear (excess stays ~1.3, 20x ceiling)
        r = lookahead._smeared_forward_ic(
            make_smeared_leak(seed=0).artifacts.aligned(), CFG)
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert r.details["excess_mass"] > lookahead.SMEAR_MASS_FAIL

    def test_gated_fresh_forward_leak_still_fails(self):
        # paired attack guard for the ffill control: an event-gated signal
        # whose event-eve stamp adds fresh forward information (mean z of
        # r_{t+1..t+3}, new each event) - the last-nonzero column is
        # F(t-1)-measurable and must not absorb it
        art = _dow_gated_book(0)
        rets = art.asset_returns
        sig = art.signals.copy()
        fwd = (rets.shift(-1) + rets.shift(-2) + rets.shift(-3)) / 3.0
        fz = _zx(fwd)
        live = sig.abs().gt(1e-15).any(axis=1)
        scale = float(np.nanstd(sig.to_numpy()[live.to_numpy()]))
        sig[live] = sig[live] + 0.6 * scale * fz[live]
        leaked = BacktestArtifacts(signals=sig, asset_returns=rets,
                                   signal_lag=1)
        leaked.validate()
        r = lookahead._smeared_forward_ic(leaked.aligned(), CFG)
        assert r.status is Status.FAIL

    def test_ffill_control_documented(self):
        # the identification note must survive doc edits: the innovation
        # docstring names the last-nonzero control and the event-gated FP
        # it closes
        doc = lookahead._innovation_forward_ic.__doc__
        assert "last-nonzero" in doc or "LAST-NONZERO" in doc
        assert "event" in doc.lower()
