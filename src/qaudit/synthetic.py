"""Synthetic backtests with known defects, for tests, demos, and calibration.

Each factory returns a :class:`SyntheticBacktest` bundling artifacts, the
pipeline callables, and ``expected_flags`` - the check-id prefixes that a
correct auditor must flag (FAIL or WARN) on that case. ``make_clean`` must
produce no FAILs at default thresholds: it is the false-positive guard.

The simulated market has genuine, weak cross-sectional momentum: the
next-period idiosyncratic return loads on the trailing 20-day mean return,
so a properly lagged rolling-mean signal earns a realistic IC (~0.02-0.04)
and Sharpe (~1-2) - real enough to pass significance probes, modest enough
to pass the "too good" detectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ._stats import traded_dollars_series
from .inputs import BacktestArtifacts

MOM_WINDOW = 20
COST_BPS = 10.0
_MAX_SYNTHETIC_ASSETS = 10_000
_MAX_SYNTHETIC_PERIODS = 60_000
_MAX_SYNTHETIC_CELLS = 2_000_000
_MAX_SYNTHETIC_CANDIDATES = 1_000
_MAX_SYNTHETIC_COST_BPS = 1_000_000.0
_MAX_ABS_MOM_LOADING = 1.0
_MAX_ABS_SMEAR_ALPHA = 1.0


@dataclass
class SyntheticBacktest:
    name: str
    description: str
    artifacts: BacktestArtifacts
    signal_func: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    backtest_func: Callable[[pd.DataFrame, pd.DataFrame], pd.Series] | None = None
    expected_flags: tuple[str, ...] = ()   # prefixes that must FAIL or WARN
    audit_kwargs: dict[str, object] = field(default_factory=dict)


def _bounded_positive_int(name: str, value: object, maximum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1 or value > maximum):
        raise ValueError(
            f"{name} must be an integer in [1, {maximum:,}], got {value!r}")
    return int(value)


def _bounded_nonnegative_int(name: str, value: object, maximum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0 or value > maximum):
        raise ValueError(
            f"{name} must be an integer in [0, {maximum:,}], got {value!r}")
    return int(value)


def _finite_float(name: str, value: object) -> float:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value,
                              (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        out = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(
            f"{name} must be representable as a finite real number, got "
            f"{value!r}") from None
    if not np.isfinite(out):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return out


def _bounded_cost_bps(value: object) -> float:
    costs_bps = _finite_float("costs_bps", value)
    if abs(costs_bps) > _MAX_SYNTHETIC_COST_BPS:
        raise ValueError(
            f"costs_bps must lie in [-{_MAX_SYNTHETIC_COST_BPS:g}, "
            f"{_MAX_SYNTHETIC_COST_BPS:g}], got {costs_bps!r}")
    return costs_bps


def _bounded_coefficient(name: str, value: object, maximum: float) -> float:
    coefficient = _finite_float(name, value)
    if abs(coefficient) > maximum:
        raise ValueError(
            f"{name} must lie in [-{maximum:g}, {maximum:g}] so the "
            f"synthetic recurrence/mix remains numerically meaningful, got "
            f"{coefficient!r}")
    return coefficient


# ---------------------------------------------------------------------------
# Market simulation
# ---------------------------------------------------------------------------

def simulate_market(n_assets: int = 30, n_periods: int = 1000, seed: int = 0,
                    mom_loading: float = 0.15,
                    death_frac: float = 0.15) -> dict[str, pd.DataFrame]:
    """Panel of daily returns with real momentum and some assets that die.

    Returns dict with 'returns' (NaN after an asset dies), 'universe'
    (True while alive), 'prices'.
    """
    n_assets = _bounded_positive_int(
        "n_assets", n_assets, _MAX_SYNTHETIC_ASSETS)
    n_periods = _bounded_positive_int(
        "n_periods", n_periods, _MAX_SYNTHETIC_PERIODS)
    if n_assets * n_periods > _MAX_SYNTHETIC_CELLS:
        raise ValueError(
            f"n_assets * n_periods must be <= "
            f"{_MAX_SYNTHETIC_CELLS:,} cells for the bundled in-memory "
            f"simulator, got {n_assets:,} * {n_periods:,}")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    seed = int(seed)
    mom_loading = _bounded_coefficient(
        "mom_loading", mom_loading, _MAX_ABS_MOM_LOADING)
    death_frac = _finite_float("death_frac", death_frac)
    if not 0.0 <= death_frac <= 1.0:
        raise ValueError(
            f"death_frac must lie in [0, 1], got {death_frac!r}")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_periods)
    assets = [f"A{i:03d}" for i in range(n_assets)]

    beta = rng.uniform(0.7, 1.3, n_assets)
    mkt = rng.normal(2e-4, 0.010, n_periods)
    sigma = rng.uniform(0.015, 0.025, n_assets)

    idio = np.zeros((n_periods, n_assets))
    eps = rng.normal(0.0, 1.0, (n_periods, n_assets)) * sigma
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, n_periods):
            lo = max(0, t - MOM_WINDOW)
            ma = idio[lo:t].mean(axis=0)
            idio[t] = mom_loading * ma + eps[t]
    if not np.isfinite(idio).all():
        raise ValueError(
            f"mom_loading={mom_loading!r} produced non-finite synthetic "
            f"returns; use a smaller recurrence coefficient")

    rets = pd.DataFrame(idio + np.outer(mkt, beta), index=dates, columns=assets)

    universe = pd.DataFrame(True, index=dates, columns=assets)
    n_dead = int(death_frac * n_assets)
    if n_dead:
        dead = rng.choice(n_assets, size=n_dead, replace=False)
        for j in dead:
            # Keep the interval non-empty even for deliberately tiny unit-test
            # markets; larger panels use the 40%-90% window.
            low = max(1, int(0.4 * n_periods)) if n_periods > 1 else 0
            high = min(n_periods, max(low + 1, int(0.9 * n_periods)))
            death_t = int(rng.integers(low, high))
            rets.iloc[death_t:, j] = np.nan
            universe.iloc[death_t:, j] = False

    with np.errstate(over="ignore", invalid="ignore"):
        prices = 100.0 * (1.0 + rets.fillna(0.0)).cumprod()
    prices = prices.where(universe)
    live = universe.to_numpy(dtype=bool)
    if (not np.isfinite(rets.to_numpy(dtype=float)[live]).all()
            or not np.isfinite(prices.to_numpy(dtype=float)[live]).all()):
        raise ValueError(
            "synthetic market generation produced non-finite live returns "
            "or prices; reduce mom_loading or the requested sample length")
    return {"returns": rets, "universe": universe, "prices": prices}


# ---------------------------------------------------------------------------
# The honest pipeline (shared by most cases)
# ---------------------------------------------------------------------------

def momentum_signal(asset_returns: pd.DataFrame,
                    window: int = MOM_WINDOW) -> pd.DataFrame:
    """Causal signal: trailing mean return, cross-sectionally z-scored per
    date (same-date cross-section only - no future data)."""
    window = _bounded_positive_int(
        "window", window, _MAX_SYNTHETIC_PERIODS)
    ma = asset_returns.rolling(window, min_periods=window).mean()
    mu = ma.mean(axis=1)
    sd = ma.std(axis=1)
    return ma.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)


def positions_from_signals(signals: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """positions.loc[t] = weights from signals.loc[t - lag] (NaN signal = flat):
    cross-sectional rank-demeaned weights, unit gross exposure per date."""
    lag = _bounded_nonnegative_int("lag", lag, _MAX_SYNTHETIC_PERIODS)
    lagged = signals.shift(lag)
    r = lagged.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    gross = w.abs().sum(axis=1)
    return w.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)


def net_returns(positions: pd.DataFrame, asset_returns: pd.DataFrame,
                costs_bps: float = COST_BPS) -> pd.Series:
    """Gross minus per-trade costs on drift-aware dollars traded - the same
    economics the audit reconstructs (qaudit._stats.traded_dollars_series):
    trades are |target - drifted holdings|, and the entry bar pays for the
    whole book. Charging on raw weight diffs instead would leave the honest
    generator and the auditor disagreeing per date on every dispersion bar."""
    costs_bps = _bounded_cost_bps(costs_bps)
    gross = (positions * asset_returns.fillna(0.0)).sum(axis=1)
    traded = traded_dollars_series(positions, asset_returns)
    return gross - traded * (costs_bps * 1e-4)


def make_backtest_func(lag: int = 1, costs_bps: float = COST_BPS
                       ) -> Callable[[pd.DataFrame, pd.DataFrame], pd.Series]:
    lag = _bounded_nonnegative_int("lag", lag, _MAX_SYNTHETIC_PERIODS)
    costs_bps = _bounded_cost_bps(costs_bps)

    def backtest_func(signals: pd.DataFrame,
                      asset_returns: pd.DataFrame) -> pd.Series:
        pos = positions_from_signals(signals, lag=lag)
        return net_returns(pos, asset_returns, costs_bps=costs_bps)
    return backtest_func


def _train_test(dates: pd.DatetimeIndex, gap: int = 5):
    cut = int(0.6 * len(dates))
    return ((dates[0], dates[cut]), (dates[min(cut + gap, len(dates) - 2)], dates[-1]))


def _assemble(name: str, description: str, market: dict,
              signals: pd.DataFrame, *,
              lag: int = 1, costs_bps: float | None = COST_BPS,
              signal_func=None, backtest_func=None,
              expected_flags: tuple[str, ...] = (),
              train_test: bool = True, universe: bool = True,
              **artifact_overrides) -> SyntheticBacktest:
    rets = market["returns"]
    pos = positions_from_signals(signals, lag=lag)
    strat = net_returns(pos, rets, costs_bps=costs_bps or 0.0)
    tr, te = _train_test(rets.index) if train_test else (None, None)
    art = dict(
        signals=signals, asset_returns=rets, positions=pos,
        strategy_returns=strat,
        universe=market["universe"] if universe else None,
        prices=market["prices"] if universe else None,
        signal_input=rets,
        train_period=tr, test_period=te,
        signal_lag=lag, declared_costs_bps=costs_bps,
    )
    art.update(artifact_overrides)
    return SyntheticBacktest(
        name=name, description=description,
        artifacts=BacktestArtifacts(**art),
        signal_func=signal_func, backtest_func=backtest_func,
        expected_flags=expected_flags,
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _doc(func: Callable[..., object]) -> str:
    """Stripped docstring of ``func``, used as the case description."""
    return (func.__doc__ or "").strip()


def make_clean(seed: int = 0) -> SyntheticBacktest:
    """Honest momentum backtest: proper 1-day lag, causal signal, costs
    charged and declared, dying assets present. Must produce zero FAILs."""
    market = simulate_market(seed=seed)
    signals = momentum_signal(market["returns"])
    return _assemble(
        "clean", _doc(make_clean), market, signals,
        signal_func=momentum_signal, backtest_func=make_backtest_func(),
        expected_flags=(),
    )


def make_lookahead(seed: int = 0) -> SyntheticBacktest:
    """Signal contaminated with the next-period return (classic lookahead):
    signal_t = 0.25 * honest_t + z(r_{t+1})."""
    market = simulate_market(seed=seed)
    rets = market["returns"]
    honest = momentum_signal(rets)
    fwd = rets.shift(-1)
    z_fwd = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1), axis=0)
    signals = (0.25 * honest.fillna(0.0) + z_fwd).where(honest.notna())
    return _assemble(
        "lookahead", _doc(make_lookahead), market, signals,
        backtest_func=make_backtest_func(),
        expected_flags=("lookahead.embedded_future_return",
                        "performance.suspicious_sharpe",
                        "performance.suspicious_ic"),
    )


def make_same_bar_execution(seed: int = 0) -> SyntheticBacktest:
    """Off-by-one execution: positions held during t are built from the
    signal of t itself (signal_lag declared 1, actual 0) - the single most
    common backtest bug."""
    market = simulate_market(seed=seed)
    signals = momentum_signal(market["returns"])
    return _assemble(
        "same_bar_execution", _doc(make_same_bar_execution),
        market, signals,
        lag=0,                      # bug: same-bar execution...
        signal_lag=1,               # ...while declaring a 1-day lag
        backtest_func=make_backtest_func(lag=0),
        expected_flags=("lookahead.position_signal_alignment",),
    )


def make_target_leak(seed: int = 0) -> SyntheticBacktest:
    """Target leakage: the signal is the prediction target plus a little
    noise (e.g. a feature computed from the label window)."""
    market = simulate_market(seed=seed)
    rets = market["returns"]
    rng = np.random.default_rng(seed + 1)
    fwd = rets.shift(-1)
    noise = pd.DataFrame(rng.normal(0, 0.2 * fwd.stack().std(), fwd.shape),
                         index=fwd.index, columns=fwd.columns)
    signals = (fwd + noise).where(rets.notna())
    return _assemble(
        "target_leak", _doc(make_target_leak), market, signals,
        backtest_func=make_backtest_func(),
        expected_flags=("leakage.", "lookahead.embedded_future_return"),
    )


def make_rolling_window_leak(seed: int = 0) -> SyntheticBacktest:
    """Subtle rolling-window violation: the signal is z-scored per asset
    using full-sample mean/std, so every historical value quietly depends on
    the future. Static ICs barely move - only the truncation probe sees it."""
    market = simulate_market(seed=seed)
    rets = market["returns"]

    def leaky_signal_func(asset_returns: pd.DataFrame) -> pd.DataFrame:
        ma = asset_returns.rolling(MOM_WINDOW, min_periods=MOM_WINDOW).mean()
        return (ma - ma.mean()) / ma.std()   # full-sample per-asset z-score

    signals = leaky_signal_func(rets)
    return _assemble(
        "rolling_window_leak", _doc(make_rolling_window_leak),
        market, signals,
        signal_func=leaky_signal_func, backtest_func=make_backtest_func(),
        expected_flags=("dynamic.rolling_window_integrity",),
    )


def make_contaminated_split(seed: int = 0) -> SyntheticBacktest:
    """Train/test contamination: the declared test window overlaps the
    training window by more than a year."""
    market = simulate_market(seed=seed)
    dates = market["returns"].index
    signals = momentum_signal(market["returns"])
    return _assemble(
        "contaminated_split", _doc(make_contaminated_split),
        market, signals,
        train_test=False,
        train_period=(dates[0], dates[int(0.7 * len(dates))]),
        test_period=(dates[int(0.4 * len(dates))], dates[-1]),
        backtest_func=make_backtest_func(),
        expected_flags=("contamination.split_overlap",),
    )


def make_survivorship(seed: int = 0) -> SyntheticBacktest:
    """Survivorship bias: universe built from today's index members - every
    loser deleted, every remaining asset has a complete history."""
    market = simulate_market(n_assets=60, seed=seed, death_frac=0.0)
    rets = market["returns"]
    total = (1.0 + rets).prod()
    keep = total.sort_values(ascending=False).index[:30]   # survivors only
    market = {"returns": rets[keep],
              "universe": None, "prices": None}
    signals = momentum_signal(market["returns"])
    return _assemble(
        "survivorship", _doc(make_survivorship), market, signals,
        universe=False,
        backtest_func=make_backtest_func(),
        expected_flags=("survivorship.",),
    )


def make_no_costs(seed: int = 0) -> SyntheticBacktest:
    """High-turnover strategy with zero transaction costs: net returns are
    identical to gross and nothing was declared."""
    market = simulate_market(seed=seed)
    rets = market["returns"]
    ma3 = rets.rolling(3, min_periods=3).mean()          # fast, high-turnover
    signals = ma3.sub(ma3.mean(axis=1), axis=0).div(ma3.std(axis=1), axis=0)
    return _assemble(
        "no_costs", _doc(make_no_costs), market, signals,
        costs_bps=None,
        backtest_func=make_backtest_func(costs_bps=0.0),
        expected_flags=("costs.",),
    )


def make_overfit(seed: int = 0, n_candidates: int = 200) -> SyntheticBacktest:
    """Selection bias: the 'signal' is the best of 200 pure-noise candidates,
    picked on full-sample net Sharpe. Placebo and permutation tests cannot see
    this - only counting trials (the deflated Sharpe check) can. Audit with
    AuditConfig(n_trials=200)."""
    n_candidates = _bounded_positive_int(
        "n_candidates", n_candidates, _MAX_SYNTHETIC_CANDIDATES)
    market = simulate_market(seed=seed, mom_loading=0.0)   # no real alpha
    rets = market["returns"]
    rng = np.random.default_rng(seed + 7)
    bt = make_backtest_func()
    best_sig, best_sr = None, -np.inf
    for _ in range(n_candidates):
        noise = pd.DataFrame(rng.normal(0, 1, rets.shape),
                             index=rets.index, columns=rets.columns)
        # slow-moving noise: low turnover, so costs don't give the game away
        cand = noise.rolling(60, min_periods=60).mean()
        cand = cand.sub(cand.mean(axis=1), axis=0).div(cand.std(axis=1), axis=0)
        cand = cand.where(rets.notna())
        r = bt(cand, rets)
        sd = r.std(ddof=1)
        sr = r.mean() / sd * np.sqrt(252) if sd > 0 else -np.inf
        if sr > best_sr:
            best_sig, best_sr = cand, sr
    case = _assemble(
        "overfit", _doc(make_overfit), market, best_sig,
        backtest_func=bt,
        expected_flags=("performance.deflated_sharpe",),
    )
    case.audit_kwargs = {"n_trials": n_candidates}
    return case


def make_smeared_leak(seed: int = 0, alpha: float = 0.2355) -> SyntheticBacktest:
    """Smeared lookahead: honest momentum plus alpha times the mean of the
    z-scored single-period forward returns over t+1..t+6, alpha tuned to
    net SR ~4.8 (just under the suspicious-Sharpe fail bar). Per-horizon
    ICs stay flat and under every gate; only the innovation forward-IC mass
    (lookahead.smeared_forward_ic) sees the sum. The bundled signal_func
    ignores its input and replays the stored panel trimmed to the input's
    index - the closure cheat that dynamic.signal_input_sensitivity flags."""
    alpha = _bounded_coefficient(
        "alpha", alpha, _MAX_ABS_SMEAR_ALPHA)
    market = simulate_market(seed=seed)
    rets = market["returns"]
    honest = momentum_signal(rets)
    fwd_z = []
    for h in range(1, 7):
        f = rets.shift(-h)
        fwd_z.append(f.sub(f.mean(axis=1), axis=0)
                     .div(f.std(axis=1).replace(0.0, np.nan), axis=0))
    signals = (honest + alpha * sum(fwd_z) / len(fwd_z)).where(honest.notna())
    if np.isinf(signals.to_numpy(dtype=float)).any():
        raise RuntimeError(
            "smeared-leak generator produced infinite signal values; "
            "the bundled synthetic case is not auditable")

    def cheating_signal_func(signal_input: pd.DataFrame) -> pd.DataFrame:
        return signals.reindex(index=signal_input.index)

    return _assemble(
        "smeared_leak", _doc(make_smeared_leak), market, signals,
        signal_func=cheating_signal_func, backtest_func=make_backtest_func(),
        expected_flags=("lookahead.smeared_forward_ic",
                        "dynamic.signal_input_sensitivity"),
    )


ALL_CASES: dict[str, Callable[..., SyntheticBacktest]] = {
    "clean": make_clean,
    "lookahead": make_lookahead,
    "same_bar_execution": make_same_bar_execution,
    "target_leak": make_target_leak,
    "rolling_window_leak": make_rolling_window_leak,
    "contaminated_split": make_contaminated_split,
    "survivorship": make_survivorship,
    "no_costs": make_no_costs,
    "overfit": make_overfit,
    "smeared_leak": make_smeared_leak,
}
