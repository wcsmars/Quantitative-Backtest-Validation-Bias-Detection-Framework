"""Deep rolling-signal dependence and same-bar separability.

The dropped-bar scan locates return innovations beyond the shallow lag grid.
When dependence cannot be spanned, the verdict states that limitation.
Adjacent-window causal books and same-bar blends exercise the boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qaudit.checks import lookahead
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.types import Severity, Status

CFG = AuditConfig()


# Shared 750-date, 30-name noise-panel builders.

def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def _rankw(df):
    r = df.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    return w.div(w.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


def _norm(pos):
    return pos.div(pos.abs().sum(axis=1).replace(0.0, np.nan),
                   axis=0).fillna(0.0)


def _noise_market(seed, n=750, m=30):
    """Pure-noise panel (no cross-sectional alpha): any same-bar coupling
    an honest book shows here is purely structural."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-04", periods=n, freq="B")
    cols = [f"A{i:03d}" for i in range(m)]
    beta = rng.uniform(0.7, 1.3, m)
    mkt = rng.normal(2e-4, 0.010, n)
    return pd.DataFrame(rng.standard_normal((n, m)) * 0.02
                        + np.outer(mkt, beta), index=idx, columns=cols)


def _arts(rets, sig, pos, lag=1):
    art = BacktestArtifacts(signals=sig, asset_returns=rets, positions=pos,
                            signal_lag=lag)
    art.validate()
    return art.aligned()


def _twin_book(rets, w, wsub):
    """The causal twin construction: w-bar rolling reversal primary plus an
    adjacent wsub-bar sub-window sleeve, both lagged 1 - positions at t
    depend only on returns through t-1 (zero same-bar information)."""
    sig = _zx(-rets.rolling(w, min_periods=w).mean())
    sub = _zx(-rets.rolling(wsub, min_periods=2).mean())
    pos = _norm(0.5 * _rankw(sig.shift(1)) + 0.5 * _rankw(sub.shift(1)))
    return sig, pos


def _rolling_leak_book(rets, w, frac=0.15):
    """Genuine same-bar bleed: (1-frac) lawful lag-1 + frac same-bar use of
    the same w-bar rolling signal."""
    sig = _zx(-rets.rolling(w, min_periods=w).mean())
    pos = _norm((1 - frac) * _rankw(sig.shift(1)) + frac * _rankw(sig))
    return sig, pos


# ---------------------------------------------------------------------------
# honest side: the causal twin class and its boundary
# ---------------------------------------------------------------------------

class TestDeepTwinHonestGuards:
    def test_twin_construction_has_zero_same_bar_information(self):
        # the honesty proof the verdict pins rest on: replace every return
        # from the cut bar onward with fresh noise - positions through the
        # cut must be bit-identical (both sleeves are lagged 1, so no
        # same-bar data can reach them)
        rets = _noise_market(0)
        _, pos = _twin_book(rets, 60, 59)
        for ci in (300, 550):
            cut = rets.index[ci]
            r2 = rets.copy()
            rng = np.random.default_rng(7000 + ci)
            r2.loc[cut:] = rng.standard_normal(r2.loc[cut:].shape) * 0.02
            _, pos2 = _twin_book(r2, 60, 59)
            assert np.allclose(pos.loc[:cut].to_numpy(),
                               pos2.loc[:cut].to_numpy(), atol=1e-12)

    def test_honest_w60_w59_twin_not_critical(self):
        # The causal twin: unpurged separability measures +0.53..+0.56
        # (seeds 0-3, 100-105), which would convict; the locator finds the
        # shared dropped bar r_{t-60} and the purge removes it - sep drops
        # to ~+0.35..+0.36, under the 0.47 bar, so the verdict is the
        # scoped WARN
        rets = _noise_market(0)
        sig, pos = _twin_book(rets, 60, 59)
        r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
        assert r.status is Status.WARN
        assert r.severity is Severity.HIGH
        assert "built from the signal of the bar it trades" not in r.message
        assert "not separable" in r.message
        d = r.details
        assert d["sep_saturated"] is True
        assert d["sep_spanned"] is True
        # the locator must have found the dropped bar at exactly j=w
        assert 60 in d["sep_deep_bars"]
        assert abs(d["sep_mean_partial_corr"]) \
            < lookahead.BLEED_SEP_PARTIAL_FAIL

    def test_honest_adjacent_twins_across_windows_not_critical(self):
        # Adjacent-window twins across deep windows exercise the dropped-bar locator
        # beyond the shallow dependence grid.
        for w, seed in ((20, 1), (100, 0)):
            rets = _noise_market(seed)
            sig, pos = _twin_book(rets, w, w - 1)
            r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
            assert r.status is Status.WARN, (w, seed)
            assert w in r.details["sep_deep_bars"], (w, seed)
            assert abs(r.details["sep_mean_partial_corr"]) \
                < lookahead.BLEED_SEP_PARTIAL_FAIL, (w, seed)

    def test_differ_by_two_twin_stays_warn(self):
        # Windows differing by two bars remain a causal control after the dropped
        # return is purged.
        rets = _noise_market(0)
        sig, pos = _twin_book(rets, 60, 58)
        r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
        assert r.status is Status.WARN

    def test_small_window_twin_unsaturated_stays_warn(self):
        # w10/w9: dependence window 8 < cap - the primary ladder reaches
        # the dropped bar itself, no deep scan runs (measured sep +0.29)
        rets = _noise_market(0)
        sig, pos = _twin_book(rets, 10, 9)
        r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
        assert r.status is Status.WARN
        d = r.details
        assert d["sep_saturated"] is False
        assert d["sep_deep_bars"] == []
        assert abs(d["sep_mean_partial_corr"]) \
            < lookahead.BLEED_SEP_PARTIAL_FAIL

    def test_honest_smooth_saturated_twin_passes(self):
        # honest side of the fail-safe scope: a smooth saturated signal
        # (EWMA hl40 primary + hl35 sleeve) has no localizable dropped bar
        # (span=False) but also measures raw +0.10 - plain PASS, the
        # scoping never engages
        rets = _noise_market(0)
        sig = _zx(-rets.ewm(halflife=40, min_periods=60).mean())
        sub = _zx(-rets.ewm(halflife=35, min_periods=60).mean())
        pos = _norm(0.5 * _rankw(sig.shift(1)) + 0.5 * _rankw(sub.shift(1)))
        r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
        assert r.status is Status.PASS
        assert r.details["sep_spanned"] is False

    def test_honest_multi_horizon_ensemble_passes(self):
        # Equal-weight 21/63/126/252-bar reversal sleeves, all lagged by one bar.
        rets = _noise_market(0)
        sleeves = [
            _zx(-rets.rolling(w, min_periods=min(w, 20)).mean()).shift(1)
            for w in (21, 63, 126, 252)]
        sig = _zx(-rets.rolling(63, min_periods=63).mean())
        pos = _norm(sum(0.25 * _rankw(s) for s in sleeves))
        r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
        assert r.status is Status.PASS


# ---------------------------------------------------------------------------
# attack side: retention + the documented scope cost
# ---------------------------------------------------------------------------

class TestDeepBleedAttackGuards:
    def test_unsaturated_blend_still_critical(self):
        # genuine 15% same-bar blend on a short-window rolling signal
        # (w=8, dependence window 7 < cap): the purge is unsaturated and
        # the conviction is untouched (measured sep +0.63, both seeds)
        for seed in (0, 1):
            rets = _noise_market(seed)
            sig, pos = _rolling_leak_book(rets, 8)
            r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
            assert r.status is Status.FAIL, seed
            assert r.severity is Severity.CRITICAL, seed
            assert abs(r.details["sep_mean_partial_corr"]) \
                > lookahead.BLEED_SEP_PARTIAL_FAIL, seed

    def test_deep_blend_never_clean(self):
        # 15% blend on the same deep w=60 signal as the honest twin: the
        # locator also purges the leak's r_{t-w} coupling, so the purged
        # statistic straddles the 0.47 bar by seed (measured WARN +0.457 /
        # FAIL +0.483 on seeds 0/1) - flagged at worst, never PASS
        for seed in (0, 1):
            rets = _noise_market(seed)
            sig, pos = _rolling_leak_book(rets, 60)
            r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
            assert r.status in (Status.WARN, Status.FAIL), seed
            assert abs(r.details["mean_partial_corr"]) \
                > CFG.bleed_partial_fail, seed
            assert abs(r.details["sep_mean_partial_corr"]) > 0.40, seed

    def test_deep_blend_conviction_names_located_bars(self):
        # when a deep blend does convict, the CRITICAL message must carry
        # the located dropped bars (auditability of the extended purge)
        rets = _noise_market(1)
        sig, pos = _rolling_leak_book(rets, 60)
        r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
        assert r.status is Status.FAIL
        assert "deep dropped bars" in r.message
        assert 60 in r.details["sep_deep_bars"]

    def test_saturated_no_spike_blend_scoped_warn(self):
        # Documented scope cost of the fail-safe backstop: a genuine 15%
        # blend on a smooth saturated signal (EWMA hl40 - no localizable
        # dropped bar) downgrades to the scoped WARN (measured sep
        # +0.51..+0.54 with span=False): flagged, never clean, never a
        # certified same-bar-execution CRITICAL the purge cannot back
        for seed in (0, 1):
            rets = _noise_market(seed)
            sig = _zx(-rets.ewm(halflife=40, min_periods=60).mean())
            pos = _norm(0.85 * _rankw(sig.shift(1)) + 0.15 * _rankw(sig))
            r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
            assert r.status is Status.WARN, seed
            assert r.severity is Severity.HIGH, seed
            d = r.details
            assert d["sep_saturated"] is True, seed
            assert d["sep_spanned"] is False, seed
            assert "cannot certify" in r.message, seed
            assert abs(d["mean_partial_corr"]) > CFG.bleed_partial_fail, seed

    def test_return_orthogonal_saturated_bleed_keeps_stamp_critical(self):
        # A return-orthogonal AR score takes the freshness path before separability
        # scoping. Saturation without a spike must not hide its same-bar blend.
        rets = _noise_market(0)
        rng = np.random.default_rng(999)
        n, m = rets.shape
        ar = np.zeros((n, m))
        eps = rng.standard_normal((n, m))
        for t in range(1, n):
            ar[t] = 0.97 * ar[t - 1] + eps[t]
        sig = _zx(pd.DataFrame(ar, index=rets.index, columns=rets.columns))
        pos = _norm(0.85 * _rankw(sig.shift(1)) + 0.15 * _rankw(sig))
        r = lookahead._same_bar_bleed(_arts(rets, sig, pos), CFG)
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert "stamp contract governs" in r.message
        assert r.details["sep_spanned"] is False

    def test_deep_constants_pinned(self):
        # Keep explicit boundary expectations for the deep scan and its evidence
        # thresholds.
        assert lookahead.BLEED_SEP_DEEP_SCAN_MAX == 130
        assert lookahead.BLEED_SEP_DEEP_BARS_MAX == 3
        assert lookahead.BLEED_SEP_RET_MAX_LAGS == 12
        assert lookahead.BLEED_SEP_PARTIAL_FAIL == 0.47
