"""Trailing risk-state controls and same-bar leakage measurement boundaries.

Coverage includes persistent-volatility sizing, raw-weight return loading,
discrete exports, short samples, configuration limits, and delayed leaks.
Causal constructions accompany each planted timing defect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.api import MODULE_CHECK_IDS, audit
from qaudit.checks import lookahead
from qaudit.config import AuditConfig
from qaudit.errors import InputValidationError
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()

BLEED = "lookahead.same_bar_bleed"
VOL = "lookahead.future_vol_sizing"
DEFERRED = "lookahead.deferred_ic_spike"
SMEAR = "lookahead.smeared_forward_ic"


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def _renorm(p):
    g = p.abs().sum(axis=1)
    return p.div(g.replace(0.0, np.nan), axis=0).fillna(0.0)


def _arts(rets, sig, pos=None, universe=None, lag=1):
    kw = dict(signals=sig, asset_returns=rets, positions=pos, signal_lag=lag)
    if pos is not None:
        kw["strategy_returns"] = net_returns(pos, rets, 10.0)
        kw["declared_costs_bps"] = 10.0
    if universe is not None:
        kw["universe"] = universe
    art = BacktestArtifacts(**kw)
    art.validate()
    return art.aligned()


def _by(results):
    return {r.check: r for r in results}


def _garch_market(seed, n_assets=30, n_periods=1000, garch_a=0.09,
                  garch_b=0.89, leverage=0.0, df=None, base_vol=0.02):
    """GARCH(1,1) idiosyncratic volatility with optional GJR leverage and t shocks, without
    planted alpha."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_periods)
    assets = [f"A{i:03d}" for i in range(n_assets)]
    beta = rng.uniform(0.7, 1.3, n_assets)
    mkt = rng.normal(2e-4, 0.010, n_periods)
    if df is None:
        z = rng.normal(size=(n_periods, n_assets))
    else:
        z = rng.standard_t(df, size=(n_periods, n_assets))
        z /= np.sqrt(df / (df - 2.0))
    var0 = base_vol ** 2
    omega = var0 * (1.0 - garch_a - garch_b)
    h = np.full(n_assets, var0)
    idio = np.zeros((n_periods, n_assets))
    for t in range(n_periods):
        eps = np.sqrt(h) * z[t]
        idio[t] = eps
        lever = 1.0 + leverage * (eps < 0)
        h = np.clip(omega + garch_a * lever * eps ** 2 + garch_b * h,
                    1e-8, (10 * base_vol) ** 2)
    return pd.DataFrame(idio + np.outer(mkt, beta), index=dates,
                        columns=assets)


def _gap_momentum(rets, window=20):
    """Classic gap momentum: trailing mean excluding the current bar -
    its innovation never contains rets[t], so same_bar_bleed sees no bar-t
    content and the magnitude channel is the only detector."""
    ma = rets.rolling(window, min_periods=window).mean().shift(1)
    return ma.sub(ma.mean(axis=1), axis=0).div(
        ma.std(axis=1).replace(0.0, np.nan), axis=0)


@pytest.fixture(scope="module")
def market0():
    return simulate_market(seed=0)


@pytest.fixture(scope="module")
def market2():
    return simulate_market(seed=2)


# ---------------------------------------------------------------------------
# 1. future_vol_sizing: honest trailing-risk sizing must PASS
# ---------------------------------------------------------------------------

def _inverse_vol_arts(seed, size_win=20, ewma=False, leverage=0.0, df=None):
    rets = _garch_market(seed, leverage=leverage, df=df)
    if ewma:
        v = (rets.pow(2).ewm(halflife=size_win, min_periods=size_win)
             .mean().pow(0.5))
    else:
        v = rets.rolling(size_win, min_periods=size_win).std()
    iv = (1.0 / v).shift(1)                          # decided from <= t-1
    pos = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)  # long-only risk parity
    sig = momentum_signal(rets)
    return _arts(rets, sig, pos)


class TestFutureVolSizingHonestGuards:
    # Persistent-volatility sizing needs multiple trailing risk controls; a single
    # noisy trailing estimate is insufficient.

    def test_garch_20d_inverse_vol_passes(self):
        r = _by(lookahead.run(_inverse_vol_arts(0, size_win=20), CFG))[VOL]
        assert r.status is Status.PASS

    def test_garch_60d_inverse_vol_passes(self):
        r = _by(lookahead.run(_inverse_vol_arts(1, size_win=60), CFG))[VOL]
        assert r.status is Status.PASS

    def test_garch_ewma_inverse_vol_passes(self):
        r = _by(lookahead.run(_inverse_vol_arts(2, size_win=20, ewma=True),
                              CFG))[VOL]
        assert r.status is Status.PASS

    def test_gjr_winner_tilt_never_fails(self):
        # Long-only momentum ranks exercise the leverage channel. Clip extreme t5 GARCH
        # shocks at total loss so this causal control clears intake's asset-return
        # floor.
        rets = _garch_market(3, leverage=1.0, df=5.0).clip(lower=-1.0)
        mom = rets.rolling(20, min_periods=20).mean().shift(1)
        rk = mom.rank(axis=1)
        pos = rk.div(rk.sum(axis=1), axis=0).fillna(0.0)
        r = _by(lookahead.run(_arts(rets, momentum_signal(rets), pos),
                              CFG))[VOL]
        assert r.status is not Status.FAIL

    def test_bars_pinned_literals(self):
        # recalibration must be a conscious edit (mutation guard)
        assert lookahead.FWD_VOL_PARTIAL_WARN == -0.05
        assert lookahead.FWD_VOL_PARTIAL_FAIL == -0.075
        # Proportional-to-future-volatility sizing uses the distinct positive-direction
        # thresholds.
        assert lookahead.FWD_VOL_PARTIAL_WARN_POS == 0.10
        assert lookahead.FWD_VOL_PARTIAL_FAIL_POS == 0.15


class TestFutureVolSizingDetectionPower:
    def test_centered_vol_sizing_still_fails(self, market0):
        rets = market0["returns"]
        sig = momentum_signal(rets)
        vol = rets.rolling(11, center=True, min_periods=7).std()
        pos = _renorm(positions_from_signals(sig, 1) / vol.clip(lower=1e-4))
        r = _by(lookahead.run(_arts(rets, sig, pos,
                                    universe=market0["universe"]), CFG))[VOL]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL

    def test_pure_forward_vol_sizer_fails(self, market0):
        # Forward-volatility sizing provides a strong timing-defect control.
        rets = market0["returns"]
        sig = momentum_signal(rets)
        vol = rets.rolling(10, min_periods=10).std().shift(-10)
        pos = _renorm(positions_from_signals(sig, 1) / vol.clip(lower=1e-4))
        r = _by(lookahead.run(_arts(rets, sig, pos,
                                    universe=market0["universe"]), CFG))[VOL]
        assert r.status is Status.FAIL
        assert r.details["mean_partial_corr"] < lookahead.FWD_VOL_PARTIAL_FAIL


# ---------------------------------------------------------------------------
# 2. future_vol_sizing: short samples SKIP (or judge) - never a false
#    "sizes are ~uniform" PASS
# ---------------------------------------------------------------------------

def _short_panel(seed=0, n=80, n_assets=20):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    cols = [f"A{i}" for i in range(n_assets)]
    scale = 1 + 2 * rng.uniform(size=n_assets)
    rets = pd.DataFrame(rng.normal(0, 0.02, (n, n_assets)) * scale,
                        index=dates, columns=cols)
    sig = pd.DataFrame(rng.normal(size=(n, n_assets)), index=dates,
                       columns=cols)
    return rets, sig


class TestFutureVolSizingShortSample:
    def test_planted_forward_sizer_on_80_rows_fails(self):
        # Fifty-nine usable dates can still identify a strong forward-volatility sizer;
        # low sample count must not imply uniform sizes.
        rets, sig = _short_panel()
        fwd = rets.rolling(10, min_periods=10).std().shift(-10)
        inv = 1.0 / fwd.clip(lower=1e-4)
        pos = inv.div(inv.abs().sum(axis=1), axis=0).fillna(0.0)
        r = _by(lookahead.run(_arts(rets, sig, pos), CFG))[VOL]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert r.details["n_usable"] < lookahead._MIN_PARTIAL_DATES

    def test_honest_trailing_sizer_on_80_rows_not_flagged(self):
        rets, sig = _short_panel()
        tr = rets.rolling(10, min_periods=10).std().shift(1)
        inv = 1.0 / tr.clip(lower=1e-4)
        pos = inv.div(inv.abs().sum(axis=1), axis=0).fillna(0.0)
        r = _by(lookahead.run(_arts(rets, sig, pos), CFG))[VOL]
        assert r.status in (Status.PASS, Status.SKIP)

    def test_tiny_sample_skips_with_reason_breakdown(self):
        rets, sig = _short_panel(n=40)
        tr = rets.rolling(10, min_periods=10).std().shift(1)
        inv = 1.0 / tr.clip(lower=1e-4)
        pos = inv.div(inv.abs().sum(axis=1), axis=0).fillna(0.0)
        r = _by(lookahead.run(_arts(rets, sig, pos), CFG))[VOL]
        assert r.status is Status.SKIP
        assert "candidate dates" in r.message
        assert r.details["n_candidates"] == 40 - 2 * CFG.fwd_vol_window - 1

    def test_equal_weight_book_passes_via_uniform_evidence(self, market0):
        # the only sub-floor PASS is measured uniformity/explained
        rets = market0["returns"]
        sig = momentum_signal(rets)
        pos = pd.DataFrame(1.0 / rets.shape[1], index=rets.index,
                           columns=rets.columns)
        r = _by(lookahead.run(_arts(rets, sig, pos), CFG))[VOL]
        assert r.status is Status.PASS
        assert r.details["n_uniform"] >= lookahead._MIN_PARTIAL_DATES


# ---------------------------------------------------------------------------
# 3. same_bar_return_loading: the magnitude-leak channel
# ---------------------------------------------------------------------------

def _magnitude_leak_arts(market, leak):
    """Add rank-demeaned same-bar return magnitudes to a causal gap-momentum rank book
    without changing its ordering."""
    rets = market["returns"]
    sig = _gap_momentum(rets)
    honest = positions_from_signals(sig, lag=1)
    rk = rets.rank(axis=1)
    tilt = _renorm(rk.sub(rk.mean(axis=1), axis=0))
    pos = honest + leak * tilt
    dec = market["universe"].shift(1, fill_value=False)
    pos = _renorm(pos.where(dec, 0.0))
    return _arts(rets, sig, pos, universe=market["universe"])


class TestSameBarReturnLoading:
    def test_headline_leak_fails_critical(self, market0):
        # leak=0.03 -> net SR 4.65 with every rank check passing (see
        # test_rank_detectors_documented_blind); the raw-weight residual
        # measures corr ~+0.90 vs same-bar return ranks at any leak size
        r = lookahead._same_bar_return_loading(
            _magnitude_leak_arts(market0, 0.03), CFG)
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert r.details["mean_corr"] > 0.5
        assert abs(r.details["nw_tstat"]) > 100

    def test_silent_subthreshold_leak_fails(self, market0):
        # leak=0.004 (net SR ~3.0, invisible to the rank channels)
        r = lookahead._same_bar_return_loading(
            _magnitude_leak_arts(market0, 0.004), CFG)
        assert r.status is Status.FAIL
        assert r.details["mean_corr"] > 0.5

    def test_rank_detectors_documented_blind(self, market0):
        # why the sibling exists: the whole registered lookahead family
        # affirmatively passes the leak book (rank/vol channels untouched).
        art = _magnitude_leak_arts(market0, 0.03)
        by = _by(lookahead.run(art, CFG))
        assert by[BLEED].status is Status.PASS
        assert by["lookahead.position_signal_alignment"].status is Status.PASS
        assert by[VOL].status is Status.PASS

    def test_honest_gap_momentum_baseline_passes(self, market0):
        # leak=0: universe-masked exact rank book -> explained-dominance
        r = lookahead._same_bar_return_loading(
            _magnitude_leak_arts(market0, 0.0), CFG)
        assert r.status is Status.PASS
        assert r.details["n_explained"] >= lookahead._MIN_PARTIAL_DATES

    def test_honest_undeclared_vol_overlay_passes(self):
        # An undeclared overlay based only on t-1 data is a causal residual control.
        m = simulate_market(seed=5)
        rets = m["returns"]
        sig = momentum_signal(rets)
        tv = rets.rolling(15, min_periods=10).std().shift(1)
        pos = _renorm(positions_from_signals(sig, 1) / tv.clip(lower=1e-4))
        r = lookahead._same_bar_return_loading(_arts(rets, sig, pos), CFG)
        assert r.status is Status.PASS
        assert abs(r.details["mean_corr"]) < lookahead.RETURN_LOADING_WARN

    def test_honest_reversal_tilt_passes(self):
        # A t-1 reversal tilt correlates with current returns only through the market's
        # momentum loading.
        m = simulate_market(seed=6)
        rets = m["returns"]
        sig = momentum_signal(rets)
        rk = rets.shift(1).rank(axis=1)
        tilt = _renorm(rk.sub(rk.mean(axis=1), axis=0))
        pos = _renorm(0.7 * positions_from_signals(sig, 1) - 0.3 * tilt)
        r = lookahead._same_bar_return_loading(_arts(rets, sig, pos), CFG)
        assert r.status is Status.PASS

    def test_skips_without_positions_and_rejects_direct_invalid_lag(self,
                                                                   market0):
        rets = market0["returns"]
        sig = _gap_momentum(rets)
        r = lookahead._same_bar_return_loading(_arts(rets, sig, None), CFG)
        assert r.status is Status.SKIP
        pos = positions_from_signals(sig, lag=0)
        invalid = _arts(rets, sig, pos, lag=1)
        invalid.signal_lag = 0  # bypass public validate(): direct-call defence
        r0 = lookahead._same_bar_return_loading(invalid, CFG)
        assert r0.status is Status.SKIP
        assert "invalid" in r0.message and ">= 1" in r0.message

    def test_bars_pinned_literals(self):
        assert lookahead.RETURN_LOADING_WARN == 0.08
        assert lookahead.RETURN_LOADING_FAIL == 0.15

    def test_registered_and_emitted(self, market0):
        rets = market0["returns"]
        sig = _gap_momentum(rets)
        art = _arts(rets, sig, positions_from_signals(sig, 1))
        emitted = {r.check for r in lookahead.run(art, CFG)}
        assert lookahead.CHECK_RETURN_LOADING in emitted
        assert (lookahead.CHECK_RETURN_LOADING
                in MODULE_CHECK_IDS["qaudit.checks.lookahead"])


# ---------------------------------------------------------------------------
# 4. same_bar_bleed: discretized signal exports SKIP, never convict
# ---------------------------------------------------------------------------

def _coarse_export_arts(seed, transform):
    m = simulate_market(n_assets=25, n_periods=320, seed=seed, death_frac=0.0)
    score = momentum_signal(m["returns"])            # continuous causal score
    pos = positions_from_signals(score, lag=1)       # book from t-1 score only
    sig = transform(score).where(score.notna())
    return _arts(m["returns"], sig, pos)


class TestBleedDiscretizedExports:
    # Coarse causal exports require a cardinality-aware skip rather than a partial-
    # correlation conviction.

    def test_sign_export_skips_not_fails(self):
        r = _by(lookahead.run(_coarse_export_arts(
            0, lambda s: s.gt(0).astype(float)), CFG))[BLEED]
        assert r.status is Status.SKIP
        assert "discretized" in r.message
        assert "continuous" in r.message
        assert r.details["median_unique_frac"] < lookahead.BLEED_MIN_UNIQUE_FRAC

    def test_decile_export_skips(self):
        r = _by(lookahead.run(_coarse_export_arts(
            0, lambda s: s.rank(axis=1, pct=True).mul(10)
            .clip(upper=9.999) // 1), CFG))[BLEED]
        assert r.status is Status.SKIP

    def test_continuous_score_still_passes(self):
        r = _by(lookahead.run(_coarse_export_arts(0, lambda s: s),
                              CFG))[BLEED]
        assert r.status is Status.PASS

    def test_gate_does_not_touch_true_positives(self):
        # A continuous signal carrying 15% same-bar information must retain its
        # failure.
        m = simulate_market(seed=4, death_frac=0.0)
        sig = momentum_signal(m["returns"])
        pos = _renorm(0.85 * positions_from_signals(sig, 1)
                      + 0.15 * positions_from_signals(sig, 0))
        r = _by(lookahead.run(_arts(m["returns"], sig, pos), CFG))[BLEED]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert r.details["median_unique_frac"] > lookahead.BLEED_MIN_UNIQUE_FRAC

    def test_unique_frac_gate_pinned(self):
        assert lookahead.BLEED_MIN_UNIQUE_FRAC == 0.8


# ---------------------------------------------------------------------------
# 5. same_bar_bleed: the 1-59-usable band judges (>= 20) or SKIPs - never a
#    false "no measurable same-bar channel" PASS
# ---------------------------------------------------------------------------

def _masked_bleed_arts(seed, live_rows, bleed, n_periods=320):
    m = simulate_market(n_assets=25, n_periods=n_periods, seed=seed,
                        death_frac=0.0)
    rets = m["returns"]
    sig = momentum_signal(rets)
    pos = _renorm((1 - bleed) * positions_from_signals(sig, 1)
                  + bleed * positions_from_signals(sig, 0))
    live = pd.Series(np.arange(len(sig)) >= len(sig) - live_rows,
                     index=sig.index)
    return _arts(rets, sig.where(live, np.nan), pos)


def _honest_tilt_masked_arts(seed, live_rows):
    m = simulate_market(n_assets=25, n_periods=320, seed=seed,
                        death_frac=0.0)
    rets = m["returns"]
    sig = momentum_signal(rets)
    slow = momentum_signal(rets, window=60)
    pos = _renorm(0.7 * positions_from_signals(sig, 1)
                  + 0.3 * positions_from_signals(slow, 1))
    live = pd.Series(np.arange(len(sig)) >= len(sig) - live_rows,
                     index=sig.index)
    return _arts(rets, sig.where(live, np.nan), pos)


class TestBleedShortBand:
    def test_59_usable_30pct_bleed_fails(self):
        # Fifty-nine dates with strong partial correlation provide measurable same-bar
        # evidence.
        r = _by(lookahead.run(_masked_bleed_arts(0, 60, 0.30), CFG))[BLEED]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert r.details["mean_partial_corr"] > 0.5

    def test_short_band_25_usable_bleed_fails(self):
        # A strong short-window blend can clear the doubled small-sample t-statistic
        # gate.
        r = _by(lookahead.run(_masked_bleed_arts(1, 27, 0.30), CFG))[BLEED]
        assert r.status is Status.FAIL
        assert r.details["n_usable"] < lookahead._MIN_PARTIAL_DATES
        assert r.details["tstat_gate"] == 2 * CFG.bleed_tstat

    def test_short_band_honest_tilt_passes(self):
        # A causal non-monotone short book must remain below the doubled t-statistic
        # gate.
        r = _by(lookahead.run(_honest_tilt_masked_arts(0, 27), CFG))[BLEED]
        assert r.status is Status.PASS

    def test_below_judge_floor_skips_both_ways(self):
        # < 20 usable: neither conviction nor exoneration - SKIP for the
        # planted bleed and the honest book alike
        r_bleed = _by(lookahead.run(_masked_bleed_arts(2, 17, 0.30),
                                    CFG))[BLEED]
        assert r_bleed.status is Status.SKIP
        assert "residual variation" in r_bleed.message
        r_honest = _by(lookahead.run(_honest_tilt_masked_arts(2, 17),
                                     CFG))[BLEED]
        assert r_honest.status is Status.SKIP

    def test_monotone_vouching_needs_60_dates(self):
        # An exact-monotone book with too few live dates must skip despite having some
        # monotone observations.
        r = _by(lookahead.run(_masked_bleed_arts(3, 42, 0.0), CFG))[BLEED]
        assert r.status is Status.SKIP
        assert 0 < r.details["n_monotone"] < lookahead._MIN_PARTIAL_DATES

    def test_monotone_dominant_book_still_passes(self, market2):
        rets = market2["returns"]
        sig = momentum_signal(rets)
        r = _by(lookahead.run(
            _arts(rets, sig, positions_from_signals(sig, 1),
                  universe=market2["universe"]), CFG))[BLEED]
        assert r.status is Status.PASS
        assert r.details["n_monotone"] >= lookahead._MIN_PARTIAL_DATES


# ---------------------------------------------------------------------------
# 6. deferred_ic_spike: h_max < 2 cannot kill the module
# ---------------------------------------------------------------------------

class TestDeferredHorizonGuard:
    def test_config_rejects_h1_at_construction(self):
        with pytest.raises(InputValidationError, match="deferred_ic_max_horizon"):
            AuditConfig(deferred_ic_max_horizon=1)

    def test_crafted_config_degrades_to_skip_not_module_error(self):
        # a config object that bypasses validation must cost one check
        # (SKIP), not all 7 (a ValueError from max() over an empty range
        # must not discard the whole family as lookahead.<module> ERROR)
        m = simulate_market(n_assets=15, n_periods=300, seed=777)
        sig = momentum_signal(m["returns"])
        art = _arts(m["returns"], sig, positions_from_signals(sig, 1),
                    universe=m["universe"])
        cfg = AuditConfig()
        object.__setattr__(cfg, "deferred_ic_max_horizon", 1)
        res = lookahead.run(art, cfg)
        assert len(res) >= 7
        by = _by(res)
        assert by[DEFERRED].status is Status.SKIP
        assert "deferred_ic_max_horizon" in by[DEFERRED].message
        assert not any(r.status is Status.ERROR for r in res)

    def test_h2_still_profiles(self):
        m = simulate_market(n_assets=15, n_periods=300, seed=777)
        sig = momentum_signal(m["returns"])
        art = _arts(m["returns"], sig, positions_from_signals(sig, 1),
                    universe=m["universe"])
        res = _by(lookahead.run(art, AuditConfig(deferred_ic_max_horizon=2)))
        assert res[DEFERRED].status in (Status.PASS, Status.WARN, Status.FAIL)
        assert res[DEFERRED].details["spike_horizon"] == 2


# ---------------------------------------------------------------------------
# 7. smeared_forward_ic: unmeasurable panels SKIP
# ---------------------------------------------------------------------------

class TestSmearSkipContract:
    def test_narrow_book_skips(self):
        # 6 names < the 8-name floor: not one innovation-IC date computes
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2022-01-03", periods=300)
        cols = [f"A{i}" for i in range(6)]
        rets = pd.DataFrame(rng.normal(0, 0.01, (300, 6)), index=dates,
                            columns=cols)
        sig = pd.DataFrame(rng.normal(size=(300, 6)), index=dates,
                           columns=cols)
        r = _by(lookahead.run(_arts(rets, sig), CFG))[SMEAR]
        assert r.status is Status.SKIP
        assert r.details["min_names_partial"] == 8
        assert r.details["smear_min_dates"] == lookahead.SMEAR_MIN_DATES

    def test_short_panel_skips(self):
        # 8 names x 70 dates: the h=10 horizon floor cannot be met
        m = simulate_market(n_assets=8, n_periods=70, seed=1, death_frac=0.0)
        rng = np.random.default_rng(0)
        sig = pd.DataFrame(rng.normal(size=m["returns"].shape),
                           index=m["returns"].index,
                           columns=m["returns"].columns)
        r = _by(lookahead.run(_arts(m["returns"], sig), CFG))[SMEAR]
        assert r.status is Status.SKIP

    def test_degenerate_innovation_still_passes(self):
        # signal == same-bar return: rank-explained by its own controls on
        # every date - measured-degenerate is a PASS, not a SKIP
        m = simulate_market(n_assets=20, n_periods=400, seed=5,
                            death_frac=0.0)
        r = _by(lookahead.run(_arts(m["returns"], m["returns"].copy()),
                              CFG))[SMEAR]
        assert r.status is Status.PASS
        assert r.details["n_degenerate"] >= lookahead.SMEAR_MIN_DATES

    def test_narrow_book_audit_reports_skip_not_pass(self):
        rng = np.random.default_rng(2)
        dates = pd.bdate_range("2022-01-03", periods=300)
        cols = [f"A{i}" for i in range(7)]
        rets = pd.DataFrame(rng.normal(0, 0.01, (300, 7)), index=dates,
                            columns=cols)
        sig = pd.DataFrame(rng.normal(size=(300, 7)), index=dates,
                           columns=cols)
        art = BacktestArtifacts(signals=sig, asset_returns=rets, signal_lag=1)
        art.validate()
        rep = audit(art, CFG, include=["lookahead"])
        by = _by(rep.results)
        assert by[SMEAR].status is Status.SKIP
        assert SMEAR in {r.check for r in rep.skips}


# ---------------------------------------------------------------------------
# 8. smeared_forward_ic tier pins: WARN and FAIL bars each constrained
#    from both sides
# ---------------------------------------------------------------------------

def _dilution_signal(rets, alpha, noise_ratio=16.0):
    """A one-bar future payoff buried in legal past-return camouflage near the warning
    boundary."""
    leak = _zx(rets.shift(-1))
    for k in range(2, 9):
        leak = leak + (noise_ratio / 7.0) * _zx(rets.shift(k))
    honest = momentum_signal(rets)
    return (honest + alpha * leak).where(honest.notna())


def _smeared_signal(rets, alpha, horizons=range(1, 7)):
    honest = momentum_signal(rets)
    hs = list(horizons)
    leak = sum((1.0 / len(hs)) * _zx(rets.shift(-h)) for h in hs)
    return (honest + alpha * leak).where(honest.notna())


class TestSmearTierPins:
    def test_warn_tier_pinned(self, market0):
        # measured on this fixture: excess 0.1088, NW t 7.2 - inside
        # [SMEAR_MASS_WARN, SMEAR_MASS_FAIL), so loosening the WARN bar or
        # tightening the FAIL bar breaks this test.
        rets = market0["returns"]
        sig = _dilution_signal(rets, 0.012)
        r = _by(lookahead.run(_arts(rets, sig,
                                    positions_from_signals(sig, 1)),
                              CFG))[SMEAR]
        assert r.status is Status.WARN
        assert r.severity is Severity.HIGH
        assert 0.10 <= r.details["excess_mass"] < 0.15

    def test_fail_bar_mid_band_pinned(self, market0):
        # excess 0.6899, t 41.8, net SR 2.22: pins the FAIL bar from above
        # - loosening it past ~0.69 turns this into a WARN and flips
        # report.ok
        rets = market0["returns"]
        sig = _smeared_signal(rets, 0.035)
        r = _by(lookahead.run(_arts(rets, sig,
                                    positions_from_signals(sig, 1)),
                              CFG))[SMEAR]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert 0.5 < r.details["excess_mass"] < 0.9

    def test_full_audit_mid_band_smear_flips_ok(self, market0):
        # This book stays below the performance gates; the smeared-forward failure
        # alone must keep report.ok false.
        rets = market0["returns"]
        sig = _smeared_signal(rets, 0.035)
        pos = positions_from_signals(sig, 1)
        art = BacktestArtifacts(
            signals=sig, asset_returns=rets, positions=pos,
            strategy_returns=net_returns(pos, rets, 10.0),
            universe=market0["universe"], signal_lag=1,
            declared_costs_bps=10.0, signal_input=rets)
        art.validate()
        rep = audit(art, CFG)
        assert not rep.ok
        assert SMEAR in {r.check for r in rep.failures}

    def test_constants_pinned_literals(self):
        assert lookahead.SMEAR_MASS_WARN == 0.10
        assert lookahead.SMEAR_MASS_FAIL == 0.15
        assert lookahead.SMEAR_TSTAT == 4.0


# ---------------------------------------------------------------------------
# 9. deferred_ic_spike tier pins (the FAIL tier is pinned in
#    tests/test_qaudit_adversarial_cases.py; the WARN band is pinned here)
# ---------------------------------------------------------------------------

class TestDeferredTierPins:
    def test_warn_tier_pinned(self, market2):
        # w=0.18 weld measures spike 0.1012 at h=3, ratio 0.226: inside the
        # WARN band [predictive_ic_warn, predictive_ic_fail) with ratio
        # under DEFERRED_WARN_RATIO - kills WARN-tier drift and FAIL-bar
        # tightening (w=0.5 pins FAIL in test_qaudit_adversarial_cases.py)
        from qaudit._stats import forward_returns
        rets = market2["returns"]
        honest = momentum_signal(rets)
        f6, f1 = forward_returns(rets, 6), forward_returns(rets, 1)
        slow = (1 + f6).div(1 + f1) - 1
        sig = (0.18 * _zx(slow) + 0.82 * honest.fillna(0)).where(honest.notna())
        r = _by(lookahead.run(_arts(rets, sig,
                                    positions_from_signals(sig, 1),
                                    universe=market2["universe"]),
                              CFG))[DEFERRED]
        assert r.status is Status.WARN
        assert r.severity is Severity.HIGH
        spike = abs(r.details["spike_ic"])
        assert CFG.predictive_ic_warn <= spike < CFG.predictive_ic_fail
        assert abs(r.details["ic1"]) < lookahead.DEFERRED_WARN_RATIO * spike

    def test_ratio_constants_pinned_literals(self):
        assert lookahead.DEFERRED_WARN_RATIO == 0.50
        assert lookahead.DEFERRED_FAIL_RATIO == 0.33
