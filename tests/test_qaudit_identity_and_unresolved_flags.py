"""Date-varying target identity and machine-readable unresolved lookahead results.

Magnitude correlation detects affine target copies whose signs cancel in
a signed median. Causal factor controls exercise the identification limit.
Unresolved lookahead evidence remains visible to strict report gates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import leakage, lookahead
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import momentum_signal, simulate_market
from qaudit.types import Severity, Status

CFG = AuditConfig()
IDENTITY = leakage.CHECK_IDENTITY
TARGET_CORR = "leakage.target_correlation"
PERFECT_RANK = "leakage.perfect_rank_dates"


# ---------------------------------------------------------------------------
# shared builders (copied from test_qaudit_lookahead_identification_limits.py and
# test_qaudit_lookahead_separability.py so the module is self-contained)
# ---------------------------------------------------------------------------

def _z(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


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


def _garch_returns(rng, n_dates, n_assets):
    a, b, w = 0.09, 0.89, 1e-5
    h = np.full(n_assets, w / (1 - a - b))
    out = np.empty((n_dates, n_assets))
    for t in range(n_dates):
        eps = rng.standard_normal(n_assets)
        out[t] = np.sqrt(h) * eps
        h = w + a * out[t] ** 2 + b * h
    return out


def _ar1_market(seed, n, m, phi):
    """Cross-sectional AR(1) idiosyncratic returns, where lagged return ranks carry genuine
    one-day alpha proportional to persistence and idiosyncratic variance share."""
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
    """Declared trailing-20-day momentum with a lawful fast rank sleeve and genuine one-bar
    alpha. Every position input is available through t-1 at declared lag one."""
    rets = _ar1_market(seed, n, m, phi)
    sig = _z(rets.rolling(20, min_periods=20).mean())
    pos = _norm((1 - w_ov) * _rankw(sig.shift(1))
                + w_ov * _rankw(rets.shift(1)))
    return _arts(rets, sig, pos)


# Date-varying target-identity builders.

def _alt_book(seed: int, n_dates: int, n_names: int,
              eps_sd: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """signals.loc[t] = s_t * z(fwd.loc[t]) + eps with s_t = +1 on even
    rows and -1 on odd rows (label_horizon 1, fwd = asset_returns.shift(-1))
    - an affine copy of the target whose sign alternates by date."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n_dates)
    cols = [f"A{i:02d}" for i in range(n_names)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (n_dates, n_names)), idx, cols)
    fwd = rets.shift(-1)
    eps = pd.DataFrame(rng.normal(0, eps_sd, fwd.shape), idx, cols)
    s = np.where(np.arange(n_dates) % 2 == 0, 1.0, -1.0)
    sig = (_z(fwd).mul(s, axis=0) + eps).where(rets.notna())
    return sig, rets


def _leak_family(sig: pd.DataFrame, rets: pd.DataFrame):
    art = BacktestArtifacts(signals=sig, asset_returns=rets).aligned()
    return {r.check: r for r in leakage.run(art, CFG)}


def _identity(sig: pd.DataFrame, rets: pd.DataFrame):
    return _leak_family(sig, rets)[IDENTITY]


def _static_beta_book(seed: int, ratio: float, n: int = 500, k: int = 25,
                      fvol: float = 0.03
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Honest, leak-free one-factor panel rets = beta * f_t + e with the
    signal equal to the static beta vector every day; ``ratio`` = idio /
    factor vol (1.5 equity-like, 0.25 factor-dominated, 1/15 degenerate).
    The cross-section is the beta ranking every day, signed by f_t."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    cols = [f"A{i:02d}" for i in range(k)]
    beta = rng.normal(1.0, 0.4, k)
    f = rng.normal(0, fvol, n)
    rets = pd.DataFrame(np.outer(f, beta) + rng.normal(0, fvol * ratio, (n, k)),
                        idx, cols)
    sig = pd.DataFrame(np.tile(beta, (n, 1)), idx, cols)
    return sig, rets


def _per_date_corr(sig: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    return leakage._standardized_rowwise_corr(sig, rets.shift(-1)).dropna()


# Magnitude-based signal-target identity.

class TestIdentityMagnitudeGate:
    def test_balanced_alternating_affine_copy_fails(self):
        # balanced case: 401 raw dates -> 400 usable -> 200/200 signs, so
        # the signed median is ~0 while every date sits at |corr| >= 0.95 -
        # a signed-only gate would PASS it
        sig, rets = _alt_book(0, 401, 12)
        corr = _per_date_corr(sig, rets)
        assert len(corr) == 400
        assert int((corr > 0).sum()) == int((corr < 0).sum()) == 200
        r = _identity(sig, rets)
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        d = r.details
        assert abs(d["median_corr"]) < CFG.leak_identity_corr
        assert d["median_abs_corr"] >= CFG.leak_identity_corr
        assert d["frac_dates_above"] == pytest.approx(1.0)
        assert d["n_dates"] == 400
        assert "date-varying sign" in r.message
        assert "sign-flipped" not in r.message
        assert "median |corr| = " in r.message
        assert "unresolved" not in d and "not_judged" not in d

    def test_balanced_by_one_nan_signal_row_fails(self):
        # Blank one signal row to balance a 400-date book at 199 positive and 199
        # negative target-copy observations.
        sig, rets = _alt_book(0, 400, 12)
        sig = sig.copy()
        sig.iloc[100, :] = np.nan
        corr = _per_date_corr(sig, rets)
        assert int((corr > 0).sum()) == int((corr < 0).sum())
        r = _identity(sig, rets)
        assert r.status is Status.FAIL
        assert abs(r.details["median_corr"]) < CFG.leak_identity_corr
        assert r.details["median_abs_corr"] >= CFG.leak_identity_corr
        assert "date-varying sign" in r.message

    def test_unbalanced_alternating_keeps_signed_flavour(self):
        # 400 raw dates -> 399 usable -> 200/199: the signed gate fires
        # first and reports the plain affine-transform wording
        sig, rets = _alt_book(0, 400, 12)
        r = _identity(sig, rets)
        assert r.status is Status.FAIL
        assert r.details["median_corr"] >= CFG.leak_identity_corr
        assert r.details["median_abs_corr"] >= CFG.leak_identity_corr
        assert r.message.endswith("an affine transform of the target")
        assert "date-varying" not in r.message
        assert "sign-flipped" not in r.message

    def test_majority_negated_block_keeps_flipped_flavour(self):
        # 55% of dates negated (block): signed median -0.998 -> the
        # sign-flipped flavour is still the one reported
        sig, rets = _alt_book(1, 500, 30)
        s = np.ones(500)
        s[: int(0.55 * 500)] = -1.0
        rng = np.random.default_rng(1)
        fwd = rets.shift(-1)
        sig = (_z(fwd).mul(s, axis=0)
               + pd.DataFrame(rng.normal(0, 0.05, fwd.shape),
                              rets.index, rets.columns)).where(rets.notna())
        r = _identity(sig, rets)
        assert r.status is Status.FAIL
        assert r.details["median_corr"] <= -CFG.leak_identity_corr
        assert "sign-flipped affine transform of the target" in r.message
        assert "date-varying" not in r.message

    @pytest.mark.parametrize("seed", range(5))
    def test_honest_momentum_panel_passes_well_below_bar(self, seed):
        # the make_clean geometry (30 names x 1000 dates): median |corr|
        # measures 0.13-0.14 across seeds 0-4
        mkt = simulate_market(seed=seed)
        rets = mkt["returns"]
        r = _identity(momentum_signal(rets), rets)
        assert r.status is Status.PASS
        assert r.details["median_abs_corr"] < 0.2
        assert abs(r.details["median_corr"]) < 0.2
        assert "median |corr| = " in r.message
        assert "unresolved" not in r.details
        assert "not_judged" not in r.details

    @pytest.mark.parametrize("seed", range(5))
    def test_honest_momentum_narrow_panel_passes(self, seed):
        # 8 names x 300 dates: noise in the per-date Pearson corr widens
        # the |corr| median to 0.30-0.32 - still a third of the bar
        mkt = simulate_market(n_assets=8, n_periods=300, seed=seed)
        rets = mkt["returns"]
        r = _identity(momentum_signal(rets), rets)
        assert r.status is Status.PASS
        assert r.details["median_abs_corr"] < 0.5

    @pytest.mark.parametrize("seed", range(5))
    def test_honest_noise_narrow_breadth_passes(self, seed):
        # Five names at the minimum breadth and short history provide a noise control
        # for the magnitude gate.
        rng = np.random.default_rng(seed)
        n, k = 60, 5
        idx = pd.bdate_range("2018-01-02", periods=n)
        cols = [f"A{i}" for i in range(k)]
        rets = pd.DataFrame(rng.normal(0, 0.01, (n, k)), idx, cols)
        sig = pd.DataFrame(rng.normal(0, 1, (n, k)), idx, cols)
        r = _identity(sig, rets)
        assert r.status is Status.PASS
        assert r.details["median_abs_corr"] < 0.7

    def test_positive_copy_fails_with_matching_medians(self):
        # +copy: every per-date corr positive, so median |corr| equals the
        # signed median and the classic flavour is reported
        rng = np.random.default_rng(3)
        n, k = 400, 12
        idx = pd.bdate_range("2018-01-02", periods=n)
        cols = [f"A{i:02d}" for i in range(k)]
        rets = pd.DataFrame(rng.normal(0, 0.01, (n, k)), idx, cols)
        fwd = rets.shift(-1)
        sig = (_z(fwd) + pd.DataFrame(rng.normal(0, 0.05, fwd.shape),
                                      idx, cols)).where(rets.notna())
        r = _identity(sig, rets)
        assert r.status is Status.FAIL
        assert r.details["median_corr"] == pytest.approx(
            r.details["median_abs_corr"])
        assert r.message.endswith("an affine transform of the target")

    def test_bar_unchanged(self):
        assert CFG.leak_identity_corr == 0.95


class TestIdentityHonestStaticBetaTilt:
    """A static loading signal on a one-factor panel approaches the identity threshold as
    idiosyncratic noise vanishes. Realistic ratios remain below it; degenerate ratios
    can cross it along with sibling checks."""

    @pytest.mark.parametrize("seed", range(3))
    def test_equity_like_ratio_passes_far_from_bar(self, seed):
        # ratio 1.5: median |corr| 0.20-0.22 - the whole family is clean
        sig, rets = _static_beta_book(seed, 1.5)
        res = _leak_family(sig, rets)
        r = res[IDENTITY]
        assert r.status is Status.PASS
        assert r.details["median_abs_corr"] < 0.3
        assert res[TARGET_CORR].status is Status.PASS
        assert res[PERFECT_RANK].status is Status.PASS

    @pytest.mark.parametrize("seed", range(3))
    def test_factor_dominated_realistic_end_passes(self, seed):
        # The 0.25 idiosyncratic-to-factor ratio stays below the identity threshold;
        # sibling detectors can still flag the factor-dominated panel.
        sig, rets = _static_beta_book(seed, 0.25)
        res = _leak_family(sig, rets)
        r = res[IDENTITY]
        assert r.status is Status.PASS
        assert r.details["median_abs_corr"] < 0.8
        assert abs(r.details["median_corr"]) < CFG.leak_identity_corr
        assert "unresolved" not in r.details

    @pytest.mark.parametrize("seed", range(3))
    def test_degenerate_ratio_never_fails_alone(self, seed):
        # ratio 1/15 (idio 0.002 vs factor 0.03): median |corr| 0.96-0.99
        # trips the magnitude gate with the date-varying-sign wording -
        # disclosed FP; target_correlation and perfect_rank_dates FAIL
        # CRITICAL on the same panel, so the identity FAIL never stands
        # alone
        sig, rets = _static_beta_book(seed, 1 / 15)
        res = _leak_family(sig, rets)
        r = res[IDENTITY]
        assert r.details["median_abs_corr"] >= CFG.leak_identity_corr
        assert abs(r.details["median_corr"]) < CFG.leak_identity_corr
        assert r.status is Status.FAIL
        assert r.severity is Severity.CRITICAL
        assert "date-varying sign" in r.message
        assert "single-factor-dominated panel" in r.message
        for sibling in (TARGET_CORR, PERFECT_RANK):
            assert res[sibling].status is Status.FAIL
            assert res[sibling].severity is Severity.CRITICAL


# Unresolved lookahead fixture builders.

def _bleed_block_book() -> BacktestArtifacts:
    """Same-bar bleed on dates 40-70 with a 15% exogenous overlay. The mean clears its
    effect-size floor while the t-statistic stays below its gate."""
    rng = np.random.default_rng(1)
    n_dates, n_assets = 120, 25
    idx = pd.bdate_range("2024-01-02", periods=n_dates)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    s = np.zeros((n_dates, n_assets))
    s[0] = rng.standard_normal(n_assets)
    for t in range(1, n_dates):
        s[t] = (0.9 * s[t - 1]
                + np.sqrt(1 - 0.81) * rng.standard_normal(n_assets))
    noise = rng.standard_normal((n_dates, n_assets))

    def rw(row):
        r_ = pd.Series(row).rank()
        w_ = r_ - r_.mean()
        return (w_ / w_.abs().sum()).to_numpy()

    pos = np.zeros_like(s)
    for t in range(1, n_dates):
        honest = 0.85 * rw(s[t - 1]) + 0.15 * rw(noise[t - 1])
        pos[t] = (0.75 * honest + 0.25 * rw(s[t]) if 40 <= t < 70
                  else honest)
    rets = pd.DataFrame(rng.standard_normal((n_dates, n_assets)) * 0.01,
                        index=idx, columns=cols)
    return _arts(rets, pd.DataFrame(s, index=idx, columns=cols),
                    pd.DataFrame(pos, index=idx, columns=cols))


def _fvs_intermittent_book(seed: int, inverse: bool,
                           block_frac: float = 0.3) -> BacktestArtifacts:
    """Causal trailing-volatility sizing with a contiguous 30% block using forward ten-bar
    volatility. The inverse and proportional variants exercise the negative and positive
    unresolved bands."""
    rng = np.random.default_rng(seed)
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
    if inverse:
        honest = np.sign(ranks) * (ranks.abs() / trail)
        leaky = np.sign(ranks) * (ranks.abs() / fwd_vol)
    else:
        honest = np.sign(ranks) * (ranks.abs() * trail)
        leaky = np.sign(ranks) * (ranks.abs() * fwd_vol)
    b0 = int(n_dates * 0.4)
    b1 = b0 + int(n_dates * block_frac)
    pos = honest.copy()
    pos.iloc[b0:b1] = leaky.iloc[b0:b1]
    return _arts(rets, sig, _norm(pos))


def _srl_block_book() -> BacktestArtifacts:
    """A return-magnitude leak on dates 60-100 of a volatility-overlay book. Its mean
    clears the effect-size threshold while the t-statistic stays below its gate."""
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
    return _arts(rets, sig, _norm(base + L))


# Hedged lookahead passes carry details["unresolved"] = True.

class TestUnresolvedFlags:
    def test_same_bar_bleed_above_bar_flagged(self):
        r = lookahead._same_bar_bleed(_bleed_block_book(), CFG)
        assert r.status is Status.PASS
        assert abs(r.details["mean_partial_corr"]) > CFG.bleed_partial_fail
        assert abs(r.details["nw_tstat"]) <= r.details["tstat_gate"]
        assert "not exonerated" in r.message
        assert r.details["unresolved"] is True

    def test_future_vol_sizing_negative_side_flagged(self):
        r = lookahead._future_vol_sizing(_fvs_intermittent_book(1, True), CFG)
        assert r.status is Status.PASS
        assert r.details["mean_partial_corr"] <= lookahead.FWD_VOL_PARTIAL_WARN
        assert abs(r.details["nw_tstat"]) < r.details["tstat_gate"]
        assert "NOT exonerated" in r.message
        assert r.details["unresolved"] is True

    def test_future_vol_sizing_positive_side_flagged(self):
        r = lookahead._future_vol_sizing(_fvs_intermittent_book(2, False), CFG)
        assert r.status is Status.PASS
        assert r.details["mean_partial_corr"] >= lookahead.FWD_VOL_PARTIAL_WARN_POS
        assert r.details["nw_tstat"] < r.details["tstat_gate"]
        assert "NOT exonerated" in r.message
        assert "positive side" in r.message
        assert r.details["unresolved"] is True

    def test_return_loading_persistent_in_band_flagged(self):
        # A causal sleeve can clear the t-statistic gate while its mean remains in the
        # causal-alpha band; its pass must state UNRESOLVED.
        r = lookahead._same_bar_return_loading(
            _genuine_alpha_book(5001, 2520, 40, 0.03), CFG)
        assert r.status is Status.PASS
        assert abs(r.details["nw_tstat"]) >= lookahead.RETURN_LOADING_TSTAT_ESC
        assert abs(r.details["mean_corr"]) < lookahead.RETURN_LOADING_ESC_MEAN_FLOOR
        assert "UNRESOLVED" in r.message
        assert r.details["unresolved"] is True

    def test_return_loading_above_bar_flagged(self):
        r = lookahead._same_bar_return_loading(_srl_block_book(), CFG)
        assert r.status is Status.PASS
        assert abs(r.details["mean_corr"]) > lookahead.RETURN_LOADING_WARN
        assert abs(r.details["nw_tstat"]) < r.details["tstat_gate"]
        assert "not exonerated" in r.message
        assert r.details["unresolved"] is True


class TestAffirmativePassesUnflagged:
    """Honest-book guards: the affirmative PASS branches carry no flag."""

    def test_clean_future_vol_sizing_unflagged(self):
        # A causal sizing book retains affirmative wording scoped to returns-derived
        # controls.
        mkt = simulate_market(n_assets=30, n_periods=700, seed=5)
        rets = mkt["returns"]
        sig = momentum_signal(rets)
        trail = rets.rolling(20, min_periods=10).std().shift(1)
        pos = _norm(_rankw(sig.shift(1)) / trail.clip(lower=1e-4))
        r = lookahead._future_vol_sizing(_arts(rets, sig, pos), CFG)
        assert r.status is Status.PASS
        assert "no anticipation" in r.message
        assert "unresolved" not in r.details
        assert "not_judged" not in r.details

    def test_clean_return_loading_short_panel_unflagged(self):
        # Short-sample genuine alpha remains below the t-statistic evidence floor.
        r = lookahead._same_bar_return_loading(
            _genuine_alpha_book(5000, 756, 40, 0.03), CFG)
        assert r.status is Status.PASS
        assert "unresolved" not in r.details
        assert "not_judged" not in r.details

    def test_clean_same_bar_bleed_unflagged(self):
        # An AR(1) signal with a lawful t-1 noise overlay has a residual channel
        # without future information and retains an unflagged pass.
        rng = np.random.default_rng(1)
        n_dates, n_assets = 120, 25
        idx = pd.bdate_range("2024-01-02", periods=n_dates)
        cols = [f"A{i:02d}" for i in range(n_assets)]
        s = np.zeros((n_dates, n_assets))
        s[0] = rng.standard_normal(n_assets)
        for t in range(1, n_dates):
            s[t] = (0.9 * s[t - 1]
                    + np.sqrt(1 - 0.81) * rng.standard_normal(n_assets))
        noise = rng.standard_normal((n_dates, n_assets))
        pos = np.zeros_like(s)
        for t in range(1, n_dates):
            pos[t] = (0.85 * _rankw(pd.DataFrame(s[t - 1:t])).to_numpy()[0]
                      + 0.15 * _rankw(pd.DataFrame(noise[t - 1:t])).to_numpy()[0])
        rets = pd.DataFrame(rng.standard_normal((n_dates, n_assets)) * 0.01,
                            index=idx, columns=cols)
        r = lookahead._same_bar_bleed(
            _arts(rets, pd.DataFrame(s, index=idx, columns=cols),
                     pd.DataFrame(pos, index=idx, columns=cols)), CFG)
        assert r.status is Status.PASS
        assert "carry no same-bar signal information" in r.message
        assert abs(r.details["mean_partial_corr"]) < CFG.bleed_partial_fail
        assert "unresolved" not in r.details
        assert "not_judged" not in r.details
