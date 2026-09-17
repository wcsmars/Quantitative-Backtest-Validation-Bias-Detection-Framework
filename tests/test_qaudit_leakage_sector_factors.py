"""Adaptive sector-factor removal and overflow containment in leakage checks.

Causal sector-rotation books exercise the factor-count boundary. Forward
injections retain detection, and huge finite values cannot crash the whole
module or suppress checks on usable observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import leakage
from qaudit.checks.leakage import _defactored_with_pcs
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.types import Severity, Status

CFG = AuditConfig()
OUTLIER = "leakage.ic_outlier_dates"
PERFECT = "leakage.perfect_rank_dates"


def _run(sig, rets, cfg=CFG):
    art = BacktestArtifacts(signals=sig, asset_returns=rets).aligned()
    return {r.check: r for r in leakage.run(art, cfg)}


# Sector-rotation and forward-injection builders.

def sector_rotation_book(seed, n_assets, n_sectors, sector_vol, idio_lo,
                         idio_hi, n_periods=750, mkt_vol=0.010, phi=0.95):
    """Honest multi-sector book: returns = beta*mkt + sector factors +
    idio; signal = strictly-causal exogenous per-sector AR(1) rotation
    score drawn from a disjoint rng stream - independent of every return,
    so zero edge and zero leak by construction. Any FAIL is an FP."""
    rng_r = np.random.default_rng(seed)
    rng_s = np.random.default_rng(seed + 909_000)
    dates = pd.bdate_range("2018-01-02", periods=n_periods)
    sec = rng_r.permutation(np.arange(n_assets) % n_sectors)
    beta = rng_r.uniform(0.7, 1.3, n_assets)
    mkt = mkt_vol * rng_r.standard_normal(n_periods)
    secf = sector_vol * rng_r.standard_normal((n_periods, n_sectors))
    iv = rng_r.uniform(idio_lo, idio_hi, n_assets)
    idio = iv[None, :] * rng_r.standard_normal((n_periods, n_assets))
    rets = beta[None, :] * mkt[:, None] + secf[:, sec] + idio
    x = np.zeros((n_periods, n_sectors))
    x[0] = rng_s.standard_normal(n_sectors)
    innov = np.sqrt(1.0 - phi * phi)
    for t in range(1, n_periods):
        x[t] = phi * x[t - 1] + innov * rng_s.standard_normal(n_sectors)
    sig = x[:, sec] + 0.35 * rng_s.standard_normal((n_periods, n_assets))
    cols = [f"S{sec[i]}N{i:02d}" for i in range(n_assets)]
    return (pd.DataFrame(sig, index=dates, columns=cols),
            pd.DataFrame(rets, index=dates, columns=cols))


ETF_CELL = dict(n_assets=30, n_sectors=6, sector_vol=0.012,
                idio_lo=0.006, idio_hi=0.006)
HET_CELL = dict(n_assets=30, n_sectors=6, sector_vol=0.012,
                idio_lo=0.006, idio_hi=0.012)
STOCK_CELL = dict(n_assets=40, n_sectors=8, sector_vol=0.008,
                  idio_lo=0.015, idio_hi=0.015)


def with_fwd_leak(seed, frac, dilution, rng_base=7000, **kw):
    """Inject diluted forward returns on random dates of the causal sector book.
    Idiosyncratic returns dominate outside extreme factor days, so the leak must survive
    principal-component removal."""
    sig, rets = sector_rotation_book(seed, **kw)
    rng = np.random.default_rng(rng_base + seed)
    fwd = rets.shift(-1)
    leak = fwd.sub(fwd.mean(axis=1), axis=0).div(
        fwd.std(axis=1).replace(0.0, np.nan), axis=0)
    if dilution > 0:
        leak = leak + dilution * pd.DataFrame(
            rng.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
    pick = rng.random(len(rets)) < frac
    sig = sig.copy()
    sig[pick] = leak[pick]
    return sig, rets


def clean_panel(seed=7, n_periods=400, n_assets=12):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n_periods)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (n_periods, n_assets)), idx,
                        cols)
    sig = pd.DataFrame(rng.normal(size=(n_periods, n_assets)), idx, cols)
    return sig, rets


# Causal multi-sector books must not be convicted.

@pytest.mark.parametrize("seed", [1003, 1004, 1006, 1010])
def test_honest_etf_like_six_sector_book_not_convicted(seed):
    # ETF-like factor dominance: sector volatility 1.2% per day versus 0.6%
    # idiosyncratic volatility.
    r = _run(*sector_rotation_book(seed, **ETF_CELL))[OUTLIER]
    assert r.status is Status.PASS, (seed, r.message)


@pytest.mark.parametrize("seed", [2003, 2005, 2008])
def test_honest_hetero_idio_sector_book_not_convicted(seed):
    # Six five-name sectors with heterogeneous idiosyncratic volatility.
    r = _run(*sector_rotation_book(seed, **HET_CELL))[OUTLIER]
    assert r.status is Status.PASS, (seed, r.message)


@pytest.mark.parametrize("seed", [1000, 1005, 1007])
def test_honest_stock_like_control_still_passes(seed):
    # Stock-like idiosyncratic volatility of 1.5% exceeds sector volatility of 0.8%;
    # adaptive factor removal must preserve this control.
    r = _run(*sector_rotation_book(seed, **STOCK_CELL))[OUTLIER]
    assert r.status is Status.PASS, (seed, r.message)


def test_adaptive_count_removes_all_above_edge_sector_factors():
    # Mechanism pin: a 6-sector book has 5 common directions beyond beta;
    # the eigen gate must select them all; a fixed cap of 3 would leave a
    # residual location mixture that convicts honest books. The
    # dispersion scale collapsing to ~1 is the downstream signature.
    sig, rets = sector_rotation_book(1003, **ETF_CELL)
    _, n_pcs = _defactored_with_pcs(rets.shift(-1))
    assert n_pcs == 5, n_pcs
    r = _run(sig, rets)[OUTLIER]
    assert r.details["defactor_pcs"] == 5, r.details
    assert r.details["dispersion_scale"] <= 1.1, r.details


# Forward-injection detection where the available evidence identifies leakage.

@pytest.mark.parametrize("seed", [3000, 3001, 3002, 3003])
def test_ten_pct_diluted_fwd_leak_on_stock_like_book_caught(seed):
    # A diluted leak on 10% of an idiosyncratic-dominated eight-sector panel must
    # survive adaptive removal of all material sector components.
    r = _run(*with_fwd_leak(seed, 0.10, 1.4, **STOCK_CELL))[OUTLIER]
    assert r.status is Status.FAIL, (seed, r.message)
    assert r.severity is Severity.CRITICAL
    assert r.details["defactor_pcs"] >= 5, r.details


@pytest.mark.parametrize("seed", [3000, 3001, 3002])
def test_full_strength_sparse_leak_on_etf_book_family_caught(seed):
    # In a factor-dominated panel, defactoring removes much of a sparse full-strength
    # injection. The raw-target perfect-rank check must retain detection.
    res = _run(*with_fwd_leak(seed, 0.02, 0.0, **ETF_CELL))
    assert res[PERFECT].status is Status.FAIL, (seed, res[PERFECT].message)


# Huge finite values must degrade gracefully without crashing eigendecomposition.

@pytest.mark.parametrize("mag", [1e154, 1e300, -1e300])
def test_huge_finite_returns_do_not_crash_leakage(mag):
    # The module must complete and return all four leakage results.
    sig, rets = clean_panel()
    rets.iloc[40:60, 3:6] = mag
    res = _run(sig, rets)
    assert set(res) == set(leakage.ALL_CHECKS)
    assert all(r.status in (Status.PASS, Status.WARN, Status.FAIL,
                            Status.SKIP) for r in res.values())


def test_huge_finite_single_column_does_not_crash_leakage():
    # Corruption confined to a single column.
    sig, rets = clean_panel(seed=777, n_periods=310, n_assets=9)
    rets.iloc[:, 4] = 1e200
    res = _run(sig, rets)
    assert set(res) == set(leakage.ALL_CHECKS)


def test_huge_finite_values_disarm_pc_stage_not_the_module():
    # The corrupted names' variance overflows to inf; the finite-variance
    # mask must drop them from the eigen analysis (here the row-mean
    # pollution kills every name, so the stage yields 0 PCs) instead of
    # handing eigh a NaN gram.
    _, rets = clean_panel(seed=11, n_periods=400, n_assets=12)
    rets.iloc[100:120, 0:4] = 1e300
    star, n_pcs = _defactored_with_pcs(rets.shift(-1))
    assert n_pcs == 0
    assert star.shape == rets.shape


def test_finite_variance_mask_does_not_disturb_clean_panels():
    # Honest-guard: on a clean factor panel the mask is all-True and the
    # PC stage output is unaffected by the finite-variance guard.
    _, rets = sector_rotation_book(1000, **ETF_CELL)
    star, n_pcs = _defactored_with_pcs(rets.shift(-1))
    assert n_pcs == 5
    assert np.isfinite(star.to_numpy()).sum() == rets.shift(-1).notna() \
        .to_numpy().sum()
