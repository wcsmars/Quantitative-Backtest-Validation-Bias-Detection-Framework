"""Leakage checks across sparse observations, factor breadth, and numeric edge cases.
"""
import numpy as np
import pandas as pd
import pytest

from qaudit.checks import leakage
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.types import Severity, Status

CFG = AuditConfig()
OUTLIER = "leakage.ic_outlier_dates"
PERFECT = "leakage.perfect_rank_dates"
IDENTITY = "leakage.signal_target_identity"


def _run(sig, rets, cfg=CFG):
    art = BacktestArtifacts(signals=sig, asset_returns=rets).aligned()
    return {r.check: r for r in leakage.run(art, cfg)}


def _z(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def _noise(seed, n_dates, n_names):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n_dates)
    cols = [f"A{i:03d}" for i in range(n_names)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (n_dates, n_names)), idx, cols)
    sig = pd.DataFrame(rng.normal(size=(n_dates, n_names)), idx, cols)
    return sig, rets


# ---------------------------------------------------------------------------
# 1. signal_target_identity must be two-sided (negated affine copies)
# ---------------------------------------------------------------------------

def _affine_copy(seed, sign):
    """Honest-looking book whose signal is sign * z(fwd) + small noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=400)
    cols = [f"A{i:02d}" for i in range(12)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (400, 12)), idx, cols)
    fwd = rets.shift(-1)
    eps = pd.DataFrame(rng.normal(0, 0.05, fwd.shape), index=idx, columns=cols)
    sig = (sign * _z(fwd) + eps).where(rets.notna())
    return sig, rets


def test_identity_negated_affine_copy_fails():
    # A sign-flipped affine target copy must be detected through correlation magnitude.
    r = _run(*_affine_copy(0, -1.0))[IDENTITY]
    assert r.status is Status.FAIL, r.message
    assert r.severity is Severity.CRITICAL
    assert r.details["median_corr"] <= -CFG.leak_identity_corr
    assert r.details["frac_dates_above"] > 0.95      # |corr|-based fraction
    assert "sign-flipped affine transform" in r.message, r.message


def test_identity_positive_affine_copy_still_fails_unflipped_wording():
    # Honest-guard for the message split: the positive copy keeps the plain
    # "an affine transform" wording (no sign-flip claim it did not measure).
    r = _run(*_affine_copy(0, +1.0))[IDENTITY]
    assert r.status is Status.FAIL, r.message
    assert r.details["median_corr"] >= CFG.leak_identity_corr
    assert "an affine transform" in r.message
    assert "sign-flipped" not in r.message


def test_identity_two_sided_gate_no_fp_on_noise():
    # The absolute-correlation gate must not convict independent noise.
    for seed in range(3):
        r = _run(*_noise(seed, 400, 8))[IDENTITY]
        assert r.status is Status.PASS, (seed, r.message)
        assert abs(r.details["median_corr"]) < 0.5
        assert r.details["frac_dates_above"] < 0.05


# ---------------------------------------------------------------------------
# 2. _OUTLIER_MIN_DATE_FRAC: wide-book detection pins that box the gate
#    near 0.005
# ---------------------------------------------------------------------------

def _wide_random_date_leak(seed, n_dates=1000, n_names=100, frac=0.03,
                           dilution=0.8):
    """A breadth-100 noise panel with diluted standardized forward returns injected on a
    fraction of dates. The injected absolute IC is about 0.78, above the 0.40 counting
    threshold."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-05", periods=n_dates)
    cols = [f"A{i:03d}" for i in range(n_names)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (n_dates, n_names)), idx, cols)
    honest = pd.DataFrame(rng.normal(size=(n_dates, n_names)), idx, cols)
    fwd = rets.shift(-1)
    noise = pd.DataFrame(rng.normal(size=fwd.shape), index=fwd.index,
                         columns=fwd.columns)
    leak_sig = _z(fwd) + dilution * noise
    pick = rng.random(n_dates) < frac
    sig = honest.copy()
    sig[pick] = leak_sig[pick].where(honest[pick].notna())
    return sig, rets


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_wide_book_intermittent_leak_convicted(seed):
    # Attack pin: observed outliers 25/25/33/30 of 999 (frac 0.025-0.033)
    # vs Poisson bound 2, expected 0.0-0.1 - any actionability gate
    # >= 0.025 would silently PASS these.
    r = _run(*_wide_random_date_leak(seed, frac=0.03))[OUTLIER]
    assert r.status is Status.FAIL, (seed, r.message)
    assert r.severity is Severity.CRITICAL
    d = r.details
    assert d["observed_outliers"] > d["noise_bound"]
    assert d["observed_outliers"] / d["n_dates"] < 0.05  # gate 0.05 kills it


def test_wide_book_sparse_leak_convicted_boxes_gate_low():
    # frac=0.01: observed 9 of 999 (frac 0.0090) - boxes the gate below
    # ~0.009, i.e. near its 0.005 default.
    r = _run(*_wide_random_date_leak(0, frac=0.01))[OUTLIER]
    assert r.status is Status.FAIL, r.message
    assert r.details["observed_outliers"] / r.details["n_dates"] < 0.010


def test_outlier_min_date_frac_gate_is_load_bearing(monkeypatch):
    # Tightening the minimum exceedance fraction tenfold would suppress this wide-book
    # leak; the test protects that gate as well as the count bound.
    monkeypatch.setattr(leakage, "_OUTLIER_MIN_DATE_FRAC", 0.05)
    r = _run(*_wide_random_date_leak(0, frac=0.03))[OUTLIER]
    assert r.status is Status.PASS, r.message


def test_wide_book_honest_noise_passes():
    # Honest guard at breadth 100: expected outliers ~0, none observed
    # beyond the bound.
    r = _run(*_wide_random_date_leak(0, frac=0.0))[OUTLIER]
    assert r.status is Status.PASS, r.message
    assert r.details["observed_outliers"] <= r.details["noise_bound"]


# ---------------------------------------------------------------------------
# 3. perfect_rank_dates: breadth-bound underpower SKIPs; history-bound WARNs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_dates", [300, 2520])
def test_perfect_rank_narrow_universe_skips_regardless_of_history(n_dates):
    # 6 names < leak_perfect_min_names (8): no history length can clear the
    # breadth gate - a permanent LOW WARN asking for more history would be
    # misdirection, so the check SKIPs, naming the real constraint and the
    # config escape hatch.
    r = _run(*_noise(0, n_dates, 6))[PERFECT]
    assert r.status is Status.SKIP, (n_dates, r.message)
    assert r.details["max_joint_names"] == 6
    assert r.details["min_names"] == 8
    assert "jointly observed names" in r.message
    assert "leak_perfect_min_names" in r.message
    # the misdirecting extend-history remediation must not accompany the SKIP
    assert "Extend the overlapping" not in (r.remediation or "")


def test_perfect_rank_short_wide_history_still_warns_extend_history():
    # Honest guard for the split: 8 names but only 20 dates is history-bound
    # - extending history genuinely fixes it, so the WARN + extend-history
    # remediation stays.
    r = _run(*_noise(1, 20, 8))[PERFECT]
    assert r.status is Status.WARN, r.message
    assert r.severity is Severity.LOW
    assert "Extend the overlapping" in r.remediation


def test_perfect_rank_config_escape_hatch_runs_narrow_book():
    # Lowering leak_perfect_min_names to the 5-name floor lets a deliberately
    # narrow book be measured (noise -> PASS).
    r = _run(*_noise(0, 300, 6), cfg=AuditConfig(leak_perfect_min_names=5))[
        PERFECT]
    assert r.status is Status.PASS, r.message


def test_perfect_rank_eight_name_noise_unaffected():
    # Boundary: max joint breadth == min_names is measurable, not a SKIP.
    r = _run(*_noise(2, 300, 8))[PERFECT]
    assert r.status is Status.PASS, r.message


# ---------------------------------------------------------------------------
# 4. details keys: n_raw_outlier_dates is a count, example_dates are dates
# ---------------------------------------------------------------------------

def test_outlier_details_count_key_named_as_count():
    r = _run(*_noise(0, 300, 8))[OUTLIER]
    assert "raw_outlier_dates" not in r.details       # a count must not carry a dates-style key
    assert isinstance(r.details["n_raw_outlier_dates"], int)
    assert all(isinstance(d, str) for d in r.details["example_dates"])
