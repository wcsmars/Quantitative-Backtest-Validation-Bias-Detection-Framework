"""Lookahead directionality and causal volatility-control boundaries.

Signed effects require distinct positive and negative evidence thresholds.
Trailing risk families, multi-sleeve books, and planted future information
exercise the boundaries of the same-bar and sizing checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import lookahead
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()

ALIGN = "lookahead.position_signal_alignment"
BLEED = "lookahead.same_bar_bleed"
VOL = "lookahead.future_vol_sizing"
RETLOAD = "lookahead.same_bar_return_loading"


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def _renorm(p):
    g = p.abs().sum(axis=1)
    return p.div(g.replace(0.0, np.nan), axis=0).fillna(0.0)


def _arts(rets, sig, pos, universe=None, lag=1):
    kw = dict(signals=sig, asset_returns=rets, positions=pos, signal_lag=lag,
              strategy_returns=net_returns(pos, rets, 10.0),
              declared_costs_bps=10.0)
    if universe is not None:
        kw["universe"] = universe
    art = BacktestArtifacts(**kw)
    art.validate()
    return art.aligned()


def _by(results):
    return {r.check: r for r in results}


def _ar1_panel(index, columns, phi=0.9, seed=7):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((len(index), len(columns)))
    x = np.empty_like(eps)
    x[0] = eps[0]
    s = np.sqrt(1 - phi * phi)
    for t in range(1, len(index)):
        x[t] = phi * x[t - 1] + s * eps[t]
    return pd.DataFrame(x, index=index, columns=columns)


@pytest.fixture(scope="module")
def market2():
    return simulate_market(seed=2)


@pytest.fixture(scope="module")
def market11():
    return simulate_market(seed=11)


# ---------------------------------------------------------------------------
# 1. ALIGNMENT_MARGIN: exact pin + a persistent-signal FAIL that dies at
#    margin >= ~0.02
# ---------------------------------------------------------------------------

class TestAlignmentMarginPinned:
    def test_margin_literal(self):
        # The docstring rationale is "neighbouring lags sit within a few
        # 0.01 of each other" - the margin is 0.01; loosening it must be a
        # conscious edit here (no other test constrains it).
        assert lookahead.ALIGNMENT_MARGIN == 0.01

    def test_same_bar_execution_on_persistent_signal_fails(self, market11):
        # Sixty-day momentum is highly autocorrelated, so same-bar execution can beat
        # lag-one alignment by only a small margin. The test pins detection near that
        # margin.
        sig60 = momentum_signal(market11["returns"], window=60)
        pos = positions_from_signals(sig60, lag=0)     # same-bar execution
        r = _by(lookahead.run(_arts(market11["returns"], sig60, pos,
                                    universe=market11["universe"]), CFG))[ALIGN]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        prof = r.details["alignment_profile"]
        gap = prof["0"] - prof["1"]
        # the pin only bites while the case sits close over the margin:
        # measured 0.0205 (seed-11 market) - leave drift room, stay < 0.03
        assert lookahead.ALIGNMENT_MARGIN < gap < 0.03
        # Note: this book also (correctly) FAILs same_bar_bleed and
        # same_bar_return_loading - asserted nowhere here on purpose.


# ---------------------------------------------------------------------------
# 2. same_bar_bleed is two-sided: negated bleeds convict; honest books with
#    a significant-but-small negative mean still pass
# ---------------------------------------------------------------------------

class TestBleedTwoSided:
    def test_negated_bleed_on_return_orthogonal_signal_fails(self, market2):
        # AR(1) X-signal (innovation orthogonal to returns) with a -15%
        # same-bar blend measures mean -0.760 / NW t -200; only a two-sided
        # bleed gate flags it.
        rets = market2["returns"]
        sigX = _zx(_ar1_panel(rets.index, rets.columns).where(rets.notna()))
        pos = _renorm(0.85 * positions_from_signals(sigX, 1)
                      - 0.15 * positions_from_signals(sigX, 0))
        by = _by(lookahead.run(_arts(rets, sigX, pos,
                                     universe=market2["universe"]), CFG))
        r = by[BLEED]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert r.details["mean_partial_corr"] < -0.5
        assert r.details["nw_tstat"] < -100
        # sole-detector documentation: the magnitude sibling has no power
        # here (the score carries no same-bar return content)
        assert by[RETLOAD].status is Status.PASS
        assert abs(by[RETLOAD].details["mean_corr"]) < lookahead.RETURN_LOADING_WARN
        # remediation carries the sign-flip clause
        assert "negated leak" in r.remediation

    def test_negated_momentum_bleed_fails(self, market2):
        # sign-flipped twin of the +15% momentum bleed true positive
        rets = market2["returns"]
        mom = momentum_signal(rets)
        pos = _renorm(0.85 * positions_from_signals(mom, 1)
                      - 0.15 * positions_from_signals(mom, 0))
        r = _by(lookahead.run(_arts(rets, mom, pos,
                                    universe=market2["universe"]), CFG))[BLEED]
        assert r.status is Status.FAIL
        assert r.details["mean_partial_corr"] < -0.5

    def test_positive_bleed_still_fails(self, market2):
        # the two-sided gate must keep the positive-side true positive
        rets = market2["returns"]
        mom = momentum_signal(rets)
        pos = _renorm(0.85 * positions_from_signals(mom, 1)
                      + 0.15 * positions_from_signals(mom, 0))
        r = _by(lookahead.run(_arts(rets, mom, pos,
                                    universe=market2["universe"]), CFG))[BLEED]
        assert r.status is Status.FAIL
        assert r.details["mean_partial_corr"] > 0.5

    def test_honest_negative_mean_below_bar_passes(self, market11):
        # A causal anti-tilt combining lagged 20-day and five-day momentum can have a
        # large t-statistic while its mean correlation stays below the effect-size
        # floor. Both gates are required.
        rets = market11["returns"]
        mom = momentum_signal(rets)
        mom5 = momentum_signal(rets, window=5)
        pos = _renorm(0.80 * positions_from_signals(mom, 1)
                      - 0.20 * positions_from_signals(mom5, 1))
        r = _by(lookahead.run(_arts(rets, mom, pos,
                                    universe=market11["universe"]), CFG))[BLEED]
        assert r.status is Status.PASS
        assert r.details["mean_partial_corr"] < -0.05          # genuinely negative
        assert abs(r.details["nw_tstat"]) > CFG.bleed_tstat    # and significant
        assert abs(r.details["mean_partial_corr"]) < CFG.bleed_partial_fail


# ---------------------------------------------------------------------------
# 3. future_vol_sizing: asymmetric/fast trailing estimators are honest -
#    the extended control grid must not convict them (and keeps its teeth)
# ---------------------------------------------------------------------------

def _gjr_t6_market(seed, n_assets=25, n_periods=800):
    """GJR volatility with a=0.05, b=0.90, gamma=0.06 and t6 shocks provides a causal
    leverage-linked volatility control."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_periods)
    a0, a1, b1, gamma = 2e-5, 0.05, 0.90, 0.06
    rets = np.zeros((n_periods, n_assets))
    h = np.full(n_assets, a0 / (1 - a1 - b1 - gamma / 2))
    prev = np.zeros(n_assets)
    for t in range(n_periods):
        z = rng.standard_t(6, n_assets) / np.sqrt(6 / 4)
        eps = np.sqrt(h) * z
        rets[t] = eps
        neg = (prev < 0).astype(float)
        h = a0 + a1 * prev ** 2 + gamma * neg * prev ** 2 + b1 * h
        prev = eps
    return pd.DataFrame(rets, index=dates,
                        columns=[f"A{i:03d}" for i in range(n_assets)])


def _inv_book(risk):
    inv = 1.0 / risk.replace(0.0, np.nan)
    return inv.div(inv.abs().sum(axis=1), axis=0).fillna(0.0)


class TestFutureVolSizingAsymmetricEstimators:
    def test_sortino_semivol_risk_parity_not_flagged(self):
        # honest 20d downside-deviation (Sortino) risk parity, strictly
        # trailing, measures corr -0.086 / NW t -4.5 against the raw grid;
        # the semivol controls must rank-explain (or absorb) it.
        rets = _gjr_t6_market(1000)
        dn = rets.where(rets < 0, 0.0)
        risk = dn.pow(2).rolling(20, min_periods=20).mean().pow(0.5).shift(1)
        r = lookahead._future_vol_sizing(
            _arts(rets, risk.fillna(0.0), _inv_book(risk)), CFG)
        assert r.status is Status.PASS

    def test_fast_riskmetrics_hl5_not_flagged(self):
        # The trailing risk-control grid must include the five-bar EWMA half-life.
        rets = _gjr_t6_market(2000)
        risk = (rets.pow(2).ewm(halflife=5, min_periods=10)
                .mean().pow(0.5).shift(1))
        r = lookahead._future_vol_sizing(
            _arts(rets, risk.fillna(0.0), _inv_book(risk)), CFG)
        assert r.status is Status.PASS

    def test_fast_riskmetrics_hl5_load_bearing_seed(self):
        # This seed exercises the five-bar EWMA control directly: the causal sizing is
        # rank-explained when that control is present, while omitting it leaves a
        # spurious residual beyond the t-statistic gate.
        rets = _gjr_t6_market(1555)
        risk = (rets.pow(2).ewm(halflife=5, min_periods=10)
                .mean().pow(0.5).shift(1))
        r = lookahead._future_vol_sizing(
            _arts(rets, risk.fillna(0.0), _inv_book(risk)), CFG)
        assert r.status is Status.PASS

    def test_fvs_control_grid_pinned(self):
        # Literal grid pins, mirroring test_band_constants_pinned: a
        # refactor must not shrink the control grid silently - each member
        # covers an honest sizing family (see the grid comment).
        assert lookahead.FWD_VOL_EWMA_HALFLIVES == (5, 10, 20, 40, 60)
        assert lookahead.FWD_VOL_CONTROL_WINDOWS == (15, 20, 30, 40, 60)
        assert lookahead.FWD_VOL_SEMIVOL_WINDOWS == (20, 60)
        assert lookahead.FWD_VOL_ABSRET_HALFLIFE == 15
        assert lookahead.FWD_VOL_MEAN_HALFLIFE == 20

    def test_mad_absret_sizing_not_flagged(self):
        # |return|-EWMA (MAD-style) estimator, hl15 - must be spanned by
        # the grid.
        rets = _gjr_t6_market(1234)
        risk = rets.abs().ewm(halflife=15, min_periods=10).mean().shift(1)
        r = lookahead._future_vol_sizing(
            _arts(rets, risk.fillna(0.0), _inv_book(risk)), CFG)
        assert r.status is Status.PASS

    def test_inverse_forward_sizer_still_fails_after_extension(self, market2):
        # the trailing controls must not absorb the leak direction: pure
        # forward-vol inverse sizing still convicts
        rets = market2["returns"]
        sig = momentum_signal(rets)
        fwd = rets.rolling(10, min_periods=10).std().shift(-10)
        pos = _renorm(positions_from_signals(sig, 1) / fwd.clip(lower=1e-4))
        r = lookahead._future_vol_sizing(
            _arts(rets, sig, pos, universe=market2["universe"]), CFG)
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert r.details["mean_partial_corr"] < lookahead.FWD_VOL_PARTIAL_FAIL


# ---------------------------------------------------------------------------
# 4. future_vol_sizing positive branch: proportional-to-future-vol sizing
#    convicts; honest proportional-to-trailing-vol books do not
# ---------------------------------------------------------------------------

class TestFutureVolSizingPositiveBranch:
    def test_proportional_forward_vol_sizing_fails(self, market2):
        # Sizes proportional to future ten-day volatility require the positive-
        # direction detection branch.
        rets = market2["returns"]
        sig = momentum_signal(rets)
        fwd = rets.rolling(10, min_periods=10).std().shift(-10)
        pos = _renorm(positions_from_signals(sig, 1) * fwd.clip(lower=1e-6))
        by = _by(lookahead.run(_arts(rets, sig, pos,
                                     universe=market2["universe"]), CFG))
        r = by[VOL]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert r.details["mean_partial_corr"] > lookahead.FWD_VOL_PARTIAL_FAIL_POS
        assert r.details["nw_tstat"] > CFG.bleed_tstat
        assert "GROW" in r.message

    def test_proportional_forward_vol_sizing_fails_under_garch(self):
        # under the check's own GARCH calibration class the plant is
        # stronger still (+0.28..+0.43 measured)
        rets = _gjr_t6_market(3000)
        sig = momentum_signal(rets)
        fwd = rets.rolling(10, min_periods=10).std().shift(-10)
        pos = _renorm(positions_from_signals(sig, 1) * fwd.clip(lower=1e-6))
        r = lookahead._future_vol_sizing(_arts(rets, sig, pos), CFG)
        assert r.status is Status.FAIL
        assert r.details["mean_partial_corr"] > lookahead.FWD_VOL_PARTIAL_FAIL_POS

    def test_honest_proportional_trailing_vol_passes(self, market2):
        # honest twin: |positions| ~ trailing 11d vol - carries only the
        # small positive EIV bias, far below the positive bars
        rets = market2["returns"]
        sig = momentum_signal(rets)
        trail = rets.rolling(11, min_periods=7).std().shift(1)
        pos = _renorm(positions_from_signals(sig, 1) * trail.clip(lower=1e-6))
        r = lookahead._future_vol_sizing(
            _arts(rets, sig, pos, universe=market2["universe"]), CFG)
        assert r.status is Status.PASS
        assert r.details["mean_partial_corr"] < lookahead.FWD_VOL_PARTIAL_WARN_POS

    def test_honest_proportional_garch_rank_v20_not_failed(self):
        # the honest positive-bias family the bars are calibrated against:
        # rank * trailing 20d vol under GARCH-t5 (measured tail up to
        # ~+0.10) - never FAIL
        rets = _garch_t5_market(306)   # the worst honest seed in the calibration sweep
        sig = momentum_signal(rets)
        v20 = rets.rolling(20, min_periods=20).std().shift(1)
        pos = _renorm(positions_from_signals(sig, 1) * v20.clip(lower=1e-6))
        r = lookahead._future_vol_sizing(_arts(rets, sig, pos), CFG)
        assert r.status is not Status.FAIL

    def test_positive_bars_pinned_and_ordered(self):
        assert lookahead.FWD_VOL_PARTIAL_WARN_POS == 0.10
        assert lookahead.FWD_VOL_PARTIAL_FAIL_POS == 0.15
        # asymmetric on purpose: the honest proportional EIV bias is
        # positive-only, so the positive bars sit above the mirrored
        # negative bars
        assert lookahead.FWD_VOL_PARTIAL_FAIL_POS > -lookahead.FWD_VOL_PARTIAL_FAIL
        assert lookahead.FWD_VOL_PARTIAL_WARN_POS > -lookahead.FWD_VOL_PARTIAL_WARN


def _garch_t5_market(seed, n_assets=30, n_periods=1000, garch_a=0.09,
                     garch_b=0.89, leverage=0.0, df=5.0, base_vol=0.02):
    """GARCH volatility with t5 shocks provides the negative-direction sizing control."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_periods)
    beta = rng.uniform(0.7, 1.3, n_assets)
    mkt = rng.normal(2e-4, 0.010, n_periods)
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
                        columns=[f"A{i:03d}" for i in range(n_assets)])


# ---------------------------------------------------------------------------
# 5. same_bar_bleed judged discretization band: extreme bleeds convict even
#    through mild bucketing; honest discretized exports never do
# ---------------------------------------------------------------------------

def _score_excl_rt(rets, window=20):
    """Trailing mean of returns through t-1 (excludes r_t), z-scored -
    stamped as the end-of-t signal, so using it during t is same-bar
    execution even though the data is older (the timing contract binds the
    stamp)."""
    ma = rets.shift(1).rolling(window, min_periods=window).mean()
    return _zx(ma)


def _disc_market(seed):
    return simulate_market(n_assets=30, n_periods=750, seed=seed,
                           death_frac=0.0)


def _round01(s):
    return ((s / 0.1).round() * 0.1).where(s.notna())


def _bled_book(score, w):
    blend = (1 - w) * score.shift(1) + w * score
    r = blend.rank(axis=1)
    return _renorm(r.sub(r.mean(axis=1), axis=0))


class TestBleedDiscretizedJudgeBand:
    def test_rounded_bleed_of_rt_excluding_score_fails(self):
        # A rounded signal carrying 30% same-bar information must remain detectable
        # despite its reduced cardinality.
        m = _disc_market(101)
        score = _score_excl_rt(m["returns"])
        by = _by(lookahead.run(
            _arts(m["returns"], _round01(score), _bled_book(score, 0.30)),
            CFG))
        r = by[BLEED]
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert r.details["mean_partial_corr"] > lookahead.BLEED_DISC_PARTIAL_FAIL
        assert r.details["median_unique_frac"] < lookahead.BLEED_MIN_UNIQUE_FRAC
        assert "discretized" in r.message
        # the magnitude sibling stays blind here - documents why the rank
        # channel needs the band (score carries no r_t)
        assert by[RETLOAD].status is Status.PASS

    def test_negated_rounded_bleed_fails(self):
        # two-sided inside the band as well
        m = _disc_market(101)
        score = _score_excl_rt(m["returns"])
        pos = _renorm(0.70 * positions_from_signals(score, 1)
                      - 0.30 * positions_from_signals(score, 0))
        r = _by(lookahead.run(
            _arts(m["returns"], _round01(score), pos), CFG))[BLEED]
        assert r.status is Status.FAIL
        assert r.details["mean_partial_corr"] < -lookahead.BLEED_DISC_PARTIAL_FAIL

    def test_honest_rounded_export_skips_with_measured_stat(self):
        # honest book, same rounded export: stays unjudged (SKIP, never
        # PASS - sub-extreme bleeds hide below the bar) and reports the
        # measured statistic instead of a blind refusal
        m = _disc_market(101)
        score = _score_excl_rt(m["returns"])
        pos = positions_from_signals(score, 1)
        r = _by(lookahead.run(
            _arts(m["returns"], _round01(score), pos), CFG))[BLEED]
        assert r.status is Status.SKIP
        assert abs(r.details["mean_partial_corr"]) < lookahead.BLEED_DISC_PARTIAL_FAIL
        assert "below the discretized conviction bar" in r.message
        assert "continuous" in r.message

    def test_deep_coarse_exports_keep_blind_skip(self):
        # sign export (unique frac ~0.08): honest bucket-migration books
        # measure up to ~0.42 there - no usable bar, so both the honest
        # book and (documented residual limitation) a bled book stay SKIP
        m = _disc_market(101)
        score = _score_excl_rt(m["returns"])
        sign_sig = score.gt(0).astype(float).where(score.notna())
        honest = _by(lookahead.run(
            _arts(m["returns"], sign_sig, positions_from_signals(score, 1)),
            CFG))[BLEED]
        assert honest.status is Status.SKIP
        assert honest.details["median_unique_frac"] < \
            lookahead.BLEED_DISC_JUDGE_MIN_UNIQUE_FRAC
        bled = _by(lookahead.run(
            _arts(m["returns"], sign_sig, _bled_book(score, 0.30)),
            CFG))[BLEED]
        assert bled.status is Status.SKIP   # known hole, documented in-module

    def test_band_constants_pinned(self):
        assert lookahead.BLEED_DISC_JUDGE_MIN_UNIQUE_FRAC == 0.45
        assert lookahead.BLEED_DISC_PARTIAL_FAIL == 0.35
        # the band bar must stay above the continuous bar (the band exists
        # because honest migration books measure up to ~0.18 there)
        assert lookahead.BLEED_DISC_PARTIAL_FAIL > CFG.bleed_partial_fail
