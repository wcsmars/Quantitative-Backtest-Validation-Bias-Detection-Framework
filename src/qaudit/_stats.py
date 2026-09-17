"""Internal statistics used by the checks.

Self-contained (numpy/pandas/scipy only). PSR/DSR follow Bailey &
Lopez de Prado (2012/2014); all Sharpe inputs to those functions are
per-period (not annualized).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import special as sc
from scipy import stats as sps

EULER_MASCHERONI = 0.5772156649015329

# PSR/DSR assume independent observations. Discount the effective sample
# size when autocorrelations indicate serial dependence, including repeated
# blocks that a short-lag screen can miss. Familywise screening limits the
# effect of noisy autocorrelation estimates on independent samples.
# This correction is a heuristic; it does not establish independence.
DEPENDENCE_MAX_LAG_FRAC = 0.5   # scan autocorrelations out to n/2
DEPENDENCE_ACF_ALPHA = 0.05     # familywise false-detection rate on IID data
DEPENDENCE_MIN_OBS = 30         # minimum observations for the PSR/DSR correction


# ---------------------------------------------------------------------------
# Sharpe machinery
# ---------------------------------------------------------------------------

def annualized_sharpe(returns: pd.Series | np.ndarray,
                      periods_per_year: float = 252.0) -> float:
    """Annualized Sharpe of a per-period return series. NaN if undefined."""
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if x.size < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(x.mean() / sd * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(returns: pd.Series | np.ndarray,
                               sr_benchmark: float = 0.0,
                               n_eff: float | None = None) -> float:
    """PSR: prob that the true per-period Sharpe exceeds ``sr_benchmark``
    (also per-period), adjusting for skew/kurtosis and sample size.

    ``n_eff`` (default: the raw observation count) is the effective sample
    size under serial dependence - see :func:`effective_sample_size`; the
    z-statistic scales with sqrt(n_eff - 1), while the moment estimates
    keep the full sample. n_eff < 30 returns NaN (too dependent to
    certify), matching the raw-n floor."""
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = x.size
    if n < 30:
        return float("nan")
    ne = float(n) if n_eff is None else float(n_eff)
    if not np.isfinite(ne) or ne < 30:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    sr = x.mean() / sd
    g3 = float(sps.skew(x))
    g4 = float(sps.kurtosis(x, fisher=False))
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(ne - 1) / np.sqrt(denom)
    return float(sps.norm.cdf(z))


def effective_sample_size(returns: pd.Series | np.ndarray) -> dict:
    """Effective number of independent observations in a return series.

    Var(mean) of a stationary series is gamma0/n * [1 + 2*sum_{k<n}
    (1 - k/n)*rho_k] (exact identity, Bartlett-in-n weights), so
    n_eff = n / [1 + 2*sum (1 - k/n)*rho_k]. Summing every sample rho_k
    would be pure noise, so only lags whose |rho_k| clears a familywise
    z-bar (alpha = DEPENDENCE_ACF_ALPHA over all scanned lags) enter the
    sum: IID samples keep n_eff == n exactly ~95% of the time, while real
    dependence - AR smoothing at short lags, block/tiling structure at
    any lag up to n/2 - is picked up wherever it lives. n_eff is capped
    at n: significant negative dependence (mean-reverting books) never
    boosts the certificate above the IID case (fail-closed).

    Returns dict(n_obs, n_eff, variance_ratio, n_dependent_lags) - all
    values are finite floats/ints.
    """
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = int(x.size)
    out = {"n_obs": n, "n_eff": float(n), "variance_ratio": 1.0,
           "n_dependent_lags": 0}
    if n < DEPENDENCE_MIN_OBS:
        return out
    e = x - x.mean()
    gamma0 = float(e @ e) / n
    if gamma0 <= 0 or not np.isfinite(gamma0):
        return out
    max_lag = min(n - 2, int(n * DEPENDENCE_MAX_LAG_FRAC))
    if max_lag < 1:
        return out
    # FFT autocovariances (biased /n normalization, same as newey_west_se)
    f = np.fft.rfft(e, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:max_lag + 1] / n
    rho = acov[1:] / gamma0
    z_bar = float(sps.norm.ppf(1.0 - 0.5 * DEPENDENCE_ACF_ALPHA / max_lag))
    sig = np.abs(rho) > z_bar / np.sqrt(n)
    n_sig = int(sig.sum())
    if n_sig == 0:
        return out
    k = np.arange(1, max_lag + 1, dtype=float)
    vr = 1.0 + 2.0 * float(((1.0 - k / n) * rho)[sig].sum())
    vr = max(vr, 1.0)               # cap n_eff at n - never boost above IID
    out.update(n_eff=float(n / vr), variance_ratio=float(vr),
               n_dependent_lags=n_sig)
    return out


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` zero-skill trials
    whose Sharpe estimates have variance ``var_trials`` (per-period)."""
    if n_trials <= 1:
        return 0.0
    # ``ppf(1 - 1/n)`` rounds its probability to exactly 1 once n exceeds
    # roughly 1e16, returning inf; converting a still larger Python integer
    # to float can raise OverflowError.  Work in log-tail space instead.
    log_n = math.log(n_trials)
    q_n = -float(sc.ndtri_exp(-log_n))
    q_ne = -float(sc.ndtri_exp(-log_n - 1.0))
    em = ((1.0 - EULER_MASCHERONI) * q_n
          + EULER_MASCHERONI * q_ne)
    return float(np.sqrt(var_trials) * em)


def deflated_sharpe_ratio(returns: pd.Series | np.ndarray,
                          n_trials: int,
                          var_trials: float | None = None) -> dict:
    """DSR: PSR against the expected best Sharpe of ``n_trials`` skill-less trials.

    ``var_trials`` is the variance of per-period Sharpe estimates across trials;
    if None, uses the zero-skill estimator variance 1/n_eff, where n_eff is
    the dependence-corrected sample size (see :func:`effective_sample_size`):
    trials are independent noise on the same sample, but serially dependent
    returns carry fewer independent observations than rows, so both the
    trial-variance default and the PSR z-statistic use n_eff (a 50-bar
    block tiled to 1000 bars would otherwise claim DSR 0.994 where the
    honest 50-bar answer is 0.083 - n_eff collapses the tiling). IID books
    keep n_eff == n and are bit-identical to the Bailey/Lopez de Prado
    formulas. An explicitly passed ``var_trials`` is honored as given.
    """
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = x.size
    ess = effective_sample_size(x)
    n_eff = float(ess["n_eff"])
    if var_trials is None:
        var_trials = 1.0 / max(n_eff, 2.0)
    sr_star = expected_max_sharpe(n_trials, var_trials)
    dsr = probabilistic_sharpe_ratio(x, sr_benchmark=sr_star, n_eff=n_eff)
    sd = x.std(ddof=1) if n > 1 else float("nan")
    sr = x.mean() / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
    return {"dsr": dsr, "sr_period": float(sr) if np.isfinite(sr) else float("nan"),
            "sr_star_period": sr_star, "n_trials": int(n_trials),
            "var_trials": float(var_trials), "n_obs": int(n),
            "n_eff": n_eff, "dependence_variance_ratio": float(ess["variance_ratio"]),
            "n_dependent_lags": int(ess["n_dependent_lags"])}


# ---------------------------------------------------------------------------
# Information coefficient
# ---------------------------------------------------------------------------

def forward_returns(asset_returns: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """fwd.loc[t] = compound asset return over the ``horizon`` periods after t,
    i.e. the target a signal known at end of t is allowed to predict.

    A -100% period is a valid observation and every window containing it
    compounds to exactly -1.0 (the product contains a zero, whatever else
    the window holds - including missing data). The naive log path breaks
    this: log1p(-1) = -inf and pandas rolling-sum turns any window touching
    -inf into NaN, silently deleting exactly ``horizon`` forward cells per
    wipeout - the observations that carry crash-leak evidence (a
    crash-predicting leak would audit fully PASS because its evidence
    rows vanished). Wipeouts are therefore masked out of the log sum and
    their windows fixed at -1.0. Returns below -100% (corrupt data) are
    floored at the wipeout here as defence in depth; the h=1 path passes
    them through raw. The intake gate on asset_returns (``validate()`` in
    inputs.py) refuses them before any check runs - a return below -100%
    raises InputValidationError - so neither path sees them on the
    supported route.
    """
    if (isinstance(horizon, (bool, np.bool_))
            or not isinstance(horizon, (int, np.integer))
            or horizon < 1):
        raise ValueError(
            f"horizon must be a positive integer number of periods, got "
            f"{horizon!r}")
    # No row can have ``horizon`` future observations when the requested
    # window spans the whole panel or more. Return the mathematically exact
    # all-missing target before handing an arbitrary-precision Python int to
    # pandas' platform-sized rolling-window conversion.
    if horizon >= len(asset_returns):
        return pd.DataFrame(np.nan, index=asset_returns.index,
                            columns=asset_returns.columns, dtype=float)
    if horizon == 1:
        return asset_returns.shift(-1)
    r = asset_returns.clip(lower=-1.0)
    wipe = r <= -1.0                        # NaN <= -1 is False: missing != wipeout
    logs = np.log1p(r.where(~wipe))         # NaN placeholder - no -inf enters rolling
    # Align to each signal date before exponentiating. The first complete
    # backward rolling window contains return row 0 and belongs to signal
    # date -1, outside this panel: it must never trigger an overflow error
    # for targets that only consume rows 1 onward.
    log_sum = logs.rolling(horizon).sum().shift(-horizon)
    # Even individually finite simple returns can have an unrepresentable
    # multi-period product (four 1e100 gains already exceed float64). Letting
    # expm1 emit inf is unsafe: IC consumers discard non-finite targets and
    # can then PASS on only the quiet remainder. Raise before the ufunc instead,
    # so audit() records a blocking module ERROR with the actual cause and
    # warning-as-error / np.seterr deployments behave identically.
    max_log = float(np.log(np.finfo(np.float64).max))
    overflow = log_sum >= max_log
    if bool(overflow.to_numpy().any()):
        n_overflow = int(overflow.to_numpy().sum())
        raise ValueError(
            f"forward return compounding exceeds float64 range in "
            f"{n_overflow} window(s) at horizon={int(horizon)}; rescale or "
            f"clean implausibly large asset returns before auditing")
    # Some NumPy 1.24 builds flag expm1(NaN) as invalid. Missing
    # rolling windows are expected: use zero only during exponentiation,
    # then restore the missing targets while retaining strict error checks.
    with np.errstate(over="raise", invalid="raise"):
        try:
            comp = np.expm1(log_sum.fillna(0.0)).where(log_sum.notna())
        except FloatingPointError as exc:
            raise ValueError(
                f"forward return compounding is numerically undefined at "
                f"horizon={int(horizon)}; clean the asset-return panel before "
                f"auditing") from exc
    comp = comp.mask(
        wipe.astype(float).rolling(horizon).sum().shift(-horizon) > 0, -1.0)
    return comp


#: Per-date breadth floor: the minimum number of jointly non-NaN names a
#: cross-section needs before its Spearman/Pearson IC counts as a
#: measurement (a rank correlation on fewer than five names is noise).
#: Shared by every check that computes per-date cross-sectional statistics.
MIN_NAMES = 5


def cross_sectional_ic(signals: pd.DataFrame, targets: pd.DataFrame,
                       method: str = "spearman",
                       min_names: int = MIN_NAMES) -> pd.Series:
    """Per-date IC between same-index/same-column frames.

    ``targets`` must already be the forward return (see :func:`forward_returns`);
    this function does no shifting of its own. Dates with fewer than
    ``min_names`` jointly finite names (default :data:`MIN_NAMES`) are NaN.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    common_cols = signals.columns.intersection(targets.columns)
    common_idx = signals.index.intersection(targets.index)
    s = signals.loc[common_idx, common_cols].to_numpy(dtype=float)
    t = targets.loc[common_idx, common_cols].to_numpy(dtype=float)
    out = np.full(len(common_idx), np.nan)
    for i in range(len(common_idx)):
        mask = np.isfinite(s[i]) & np.isfinite(t[i])
        if mask.sum() < min_names:
            continue
        a, b = s[i][mask], t[i][mask]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        if method == "spearman":
            r = sps.spearmanr(a, b).statistic
        else:
            r = np.corrcoef(a, b)[0, 1]
        out[i] = r
    return pd.Series(out, index=common_idx, name="ic")


def newey_west_se(x: pd.Series | np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) standard error of the mean of ``x`` (NaNs dropped).

    Bartlett kernel with ``lags`` autocovariance terms; the default
    bandwidth is the usual floor(4 (n/100)^(2/9)), capped at n - 2. NaN
    below 10 observations or when the HAC long-run variance is not
    positive. Exposed on its own because a near-zero mean makes
    mean / newey_west_tstat numerically useless as a route to the SE.
    """
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = v.size
    if n < 10:
        return float("nan")
    if lags is None:
        lags = min(int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))), n - 2)
    e = v - v.mean()
    var = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * float(e[k:] @ e[:-k]) / n
    if var <= 0:
        return float("nan")
    return float(np.sqrt(var / n))


def newey_west_tstat(x: pd.Series | np.ndarray, lags: int | None = None) -> float:
    """t-stat of the mean with Newey-West (HAC) standard errors
    (mean / :func:`newey_west_se`; NaN wherever the SE is)."""
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    se = newey_west_se(v, lags)
    if np.isnan(se):
        return float("nan")
    return float(v.mean() / se)


def fmt_tstat(t: float, big: float = 1e3) -> str:
    """Render a t-statistic for a check message: one decimal in the
    ordinary range, ``:.3g`` once |t| reaches ``big``. A degenerate series
    (a book whose positions are the same-bar signal: partial corr ~1 on
    every date, HAC variance ~1e-30) yields NW t ~1e17, which ``:.1f``
    would print as an 18-digit integer; nan/inf render as such."""
    return f"{t:.1f}" if abs(t) < big else f"{t:.3g}"


# ---------------------------------------------------------------------------
# Portfolio mechanics
# ---------------------------------------------------------------------------

def drifted_weights(positions: pd.DataFrame,
                    asset_returns: pd.DataFrame) -> np.ndarray:
    """Pre-trade drifted weights at the start of every bar, as a
    ``(n_periods, n_assets)`` float array aligned to ``positions``.

    Under the library convention (positions.loc[t] held during t, earning
    asset_returns.loc[t], uninvested residual at zero yield - the same
    convention :func:`gross_strategy_returns` prices), the book that held
    w[t-1] through r[t-1] arrives at the start of t already holding

        w_drift[t] = w[t-1] * (1 + r[t-1]) / (1 + sum_i w[t-1]_i * r[t-1]_i)

    This is the sole self-financing "hold" encoding of a book - what an
    engine that records actually-held weights writes on a bar it did not
    trade. A repeated constant target (w[t] = w[t-1]) is an honest target,
    but not a no-trade hold when returns disperse: restoring it trades real
    dollars. Both :func:`traded_dollars_series` and
    ``survivorship.trading_outside_universe`` therefore measure trades
    against this drifted reference.

    Conventions: ``asset_returns`` is reindexed to ``positions``; NaN
    positions and NaN returns fill to 0 (matching the gross
    reconstruction); r = -100% zeroes that position and renormalizes the
    survivors. Row 0 is all zeros (no prior book - the zero-padded
    previous-position convention). Rows whose NAV growth
    1 + port_ret[t-1] is not strictly positive (book wiped or driven
    negative) have no defined weights and are returned as NaN rows -
    surfaced, never fabricated; callers choose their own fallback.
    """
    pos = positions.fillna(0.0)
    r = asset_returns.reindex(index=pos.index, columns=pos.columns).fillna(0.0)
    grow = (pos * (1.0 + r)).shift(1)       # w[t-1]*(1+r[t-1]), at row t
    denom = (1.0 + (pos * r).sum(axis=1)).shift(1)   # NAV growth during t-1
    # ``copy=True``: under pandas Copy-on-Write (the pandas 3 default)
    # ``to_numpy`` returns a read-only view and the in-place edits below
    # would raise ``ValueError: assignment destination is read-only``.
    w = grow.div(denom, axis=0).to_numpy(dtype=float, copy=True)
    if w.shape[0]:
        w[~(denom.to_numpy(dtype=float) > 0.0)] = np.nan   # NaN denom too
        w[0] = 0.0
    return w


def traded_dollars_series(positions: pd.DataFrame,
                          asset_returns: pd.DataFrame) -> pd.Series:
    """Two-sided dollars traded per period per unit of book NAV, drift-aware.

    The trade at t is |w[t] - w_drift[t]| per asset with the pre-trade
    drifted weights of :func:`drifted_weights`, not |w[t] - w[t-1]|: raw
    diffs report 0 for a constant-mix book that genuinely re-buys against
    drift every bar, and charge phantom trades on a true buy-and-hold book
    whose weights only drift (a constant-target book through +/-20%
    dispersion trades 25.2x/yr one-sided, which raw diffs price at 0).
    First row = the entry trade |w[0]|.sum(). NaN returns fill to 0
    (matching the gross reconstruction); r = -100% zeroes that position
    and renormalizes survivors. Rows where NAV growth 1 + port_ret[t-1]
    <= 0 (book wiped or driven negative) have no defined weights - those
    rows are NaN, surfaced rather than fabricated.
    """
    pos = positions.fillna(0.0)
    drift = drifted_weights(pos, asset_returns)
    if drift.shape[0]:
        # row 0 has no prior book (drifted_weights returns zeros there): set
        # it to NaN so the diff row is an explicit placeholder - pandas'
        # NaN-aware row sum then yields the same numbers as the shift-based
        # formula - and overwrite it with the entry trade below.
        drift[0] = np.nan
    w_drift = pd.DataFrame(drift, index=pos.index, columns=pos.columns)
    traded = (pos - w_drift).abs().sum(axis=1)
    if len(traded):
        traded.iloc[0] = pos.iloc[0].abs().sum()     # entry trade
        undefined = np.isnan(drift).any(axis=1)      # NAV growth <= 0 rows
        undefined[0] = False
        traded[undefined] = np.nan
    return traded


def turnover_series(positions: pd.DataFrame,
                    asset_returns: pd.DataFrame) -> pd.Series:
    """One-sided turnover per period: 0.5 * sum_i |w_it - w_drift_it|, with
    the pre-trade drifted weights of :func:`traded_dollars_series`.

    First period is NaN (no prior book - the entry trade is a dollars
    question, not a turnover level). 1.0 == the whole book replaced.
    """
    to = 0.5 * traded_dollars_series(positions, asset_returns)
    if len(to):
        to.iloc[0] = np.nan
    return to


def gross_strategy_returns(positions: pd.DataFrame,
                           asset_returns: pd.DataFrame) -> pd.Series:
    """Gross (cost-free) portfolio return under the library convention:
    positions.loc[t] are the weights held during period t and earn
    asset_returns.loc[t]."""
    common_idx = positions.index.intersection(asset_returns.index)
    common_cols = positions.columns.intersection(asset_returns.columns)
    p = positions.loc[common_idx, common_cols].fillna(0.0)
    r = asset_returns.loc[common_idx, common_cols].fillna(0.0)
    return (p * r).sum(axis=1)
