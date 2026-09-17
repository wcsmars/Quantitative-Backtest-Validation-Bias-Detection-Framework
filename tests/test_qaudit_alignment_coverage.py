"""Declared-lag alignment requires enough jointly observed dates and cells.

Missing declared-lag signals cannot serve as a losing baseline. Alternative
lags need equivalent support, and same-bar checks skip unmeasurable panels
while retaining detection on short but sufficiently supported windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.checks import lookahead
from qaudit.checks.lookahead import (ALIGNMENT_MIN_CELLS, ALIGNMENT_MIN_DATES)
from qaudit.config import AuditConfig
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (make_same_bar_execution, momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()

ALIGN = "lookahead.position_signal_alignment"
BLEED = "lookahead.same_bar_bleed"


@pytest.fixture(scope="module")
def market():
    return simulate_market(seed=11)


@pytest.fixture(scope="module")
def honest(market):
    return momentum_signal(market["returns"])


def _arts(market, sig, pos, **kw):
    rets = market["returns"]
    base = dict(signals=sig, asset_returns=rets, positions=pos,
                strategy_returns=net_returns(pos, rets, 10.0),
                universe=market.get("universe"), signal_lag=1,
                declared_costs_bps=10.0)
    base.update(kw)
    art = BacktestArtifacts(**base)
    art.validate()
    return art.aligned()


def _by(results):
    return {r.check: r for r in results}


def _mask_to_tail(sig, live_rows):
    """NaN out everything but the last ``live_rows`` rows (signal went live
    late - its first valid date sits after positions start)."""
    live = pd.Series(np.arange(len(sig)) >= len(sig) - live_rows, index=sig.index)
    return sig.where(live, np.nan)


# ---------------------------------------------------------------------------
# 1. attack: all-NaN at the declared lag -> SKIP, never CRITICAL FAIL
# ---------------------------------------------------------------------------

def test_late_signal_honest_book_skips_not_fails(market, honest):
    # The causal book's audited signal starts on the final row: lag 1 has no usable
    # baseline dates and lag 0 has one. Insufficient baseline evidence must skip.
    pos = positions_from_signals(honest, lag=1)
    art = _arts(market, _mask_to_tail(honest, 1), pos)
    r = _by(lookahead.run(art, CFG))[ALIGN]
    assert r.status is Status.SKIP
    # actionable: states the measured overlap and both floors
    assert "0 date(s)" in r.message
    assert str(ALIGNMENT_MIN_DATES) in r.message
    assert str(ALIGNMENT_MIN_CELLS) in r.message
    assert "signal_lag" in r.message
    assert r.details["n_dates_per_lag"]["1"] == 0
    assert r.details["n_cells_per_lag"]["1"] == 0
    assert r.details["n_dates_per_lag"]["0"] >= 1


def test_late_signal_whole_module_emits_no_flags(market, honest):
    # the same NaN pattern must not trip any lookahead detector (each either
    # SKIPs or PASSes on explicitly-measured evidence - no -inf/NaN baseline
    # anywhere in the family).
    pos = positions_from_signals(honest, lag=1)
    art = _arts(market, _mask_to_tail(honest, 1), pos)
    res = lookahead.run(art, CFG)
    flagged = [r for r in res if r.status in (Status.FAIL, Status.WARN)]
    assert not flagged, f"honest late-signal book flagged: {flagged}"


def test_mid_sample_nan_cadence_skips_not_fails(market, honest):
    # Even-row positions pair with missing odd-row signals at declared lag 1. Lag 0
    # having more observations cannot compensate for an unmeasurable baseline.
    even = pd.Series(np.arange(len(honest)) % 2 == 0, index=honest.index)
    sig = honest.where(even, np.nan)
    pos = positions_from_signals(honest, lag=1).where(even, 0.0)
    art = _arts(market, sig, pos)
    r = _by(lookahead.run(art, CFG))[ALIGN]
    assert r.status is Status.SKIP
    assert r.details["n_dates_per_lag"]["1"] == 0
    assert r.details["n_dates_per_lag"]["0"] > 100   # k=0 evidence was ample


def test_sub_floor_cells_guard(honest):
    # dates floor met, cells floor not: 6-asset universe, signal live for the
    # last 13 rows -> ~12 baseline dates x 6 names ~ 72 cells < 100.
    market6 = simulate_market(n_assets=6, seed=11, death_frac=0.0)
    sig6 = momentum_signal(market6["returns"])
    pos6 = positions_from_signals(sig6, lag=1)
    art = _arts(market6, _mask_to_tail(sig6, 13), pos6)
    r = _by(lookahead.run(art, CFG))[ALIGN]
    assert r.status is Status.SKIP
    assert r.details["n_dates_per_lag"]["1"] >= ALIGNMENT_MIN_DATES
    assert r.details["n_cells_per_lag"]["1"] < ALIGNMENT_MIN_CELLS
    assert "cells" in r.message


# ---------------------------------------------------------------------------
# 2. sibling: same_bar_bleed on the unmeasurable book -> SKIP, not a false
#    "monotone function ... on 0 dates" PASS
# ---------------------------------------------------------------------------

def test_bleed_skips_when_nothing_measurable(market, honest):
    pos = positions_from_signals(honest, lag=1)
    art = _arts(market, _mask_to_tail(honest, 1), pos)
    r = _by(lookahead.run(art, CFG))[BLEED]
    assert r.status is Status.SKIP
    assert r.details["n_usable"] == 0 and r.details["n_monotone"] == 0
    assert "non-NaN at the declared lag" in r.message


def test_bleed_monotone_pass_path_intact(market, honest):
    # clean book: positions are an exact rank function of the lag-1 signal,
    # so every date is monotone-degenerate - that is a measured good outcome
    # and must stay PASS (the SKIP applies only to 0-usable/0-monotone).
    pos = positions_from_signals(honest, lag=1)
    art = _arts(market, honest, pos)
    r = _by(lookahead.run(art, CFG))[BLEED]
    assert r.status is Status.PASS
    assert r.details["n_monotone"] > 0


def test_bleed_detection_power_intact(market, honest):
    # A 15% same-bar signal contribution must retain detection.
    pos = (0.85 * positions_from_signals(honest, 1)
           + 0.15 * positions_from_signals(honest, 0))
    gross = pos.abs().sum(axis=1)
    pos = pos.div(gross.replace(0, np.nan), axis=0).fillna(0.0)
    art = _arts(market, honest, pos)
    r = _by(lookahead.run(art, CFG))[BLEED]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL


# ---------------------------------------------------------------------------
# 3. honest guards: detection power of the alignment check itself
# ---------------------------------------------------------------------------

def test_same_bar_execution_still_critical():
    case = make_same_bar_execution()
    r = _by(lookahead.run(case.artifacts.aligned(), CFG))[ALIGN]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["best_lag"] == 0
    # the FAIL message carries the evidence depth
    assert "dates at lag 0" in r.message


def test_same_bar_bug_with_small_but_measurable_window_still_fails(market, honest):
    # real bug + short signal history that clears the floor (~15 baseline
    # dates, ~25 names/date): the floor must not eat legitimate detections.
    pos = positions_from_signals(honest, lag=0)          # bug: same-bar
    art = _arts(market, _mask_to_tail(honest, 16), pos)  # declared lag 1
    r = _by(lookahead.run(art, CFG))[ALIGN]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["n_dates_per_lag"]["1"] >= ALIGNMENT_MIN_DATES


def test_clean_full_book_still_passes(market, honest):
    pos = positions_from_signals(honest, lag=1)
    art = _arts(market, honest, pos)
    r = _by(lookahead.run(art, CFG))[ALIGN]
    assert r.status is Status.PASS
    assert r.details["best_lag"] == 1
    # floors are reported for auditability
    assert r.details["min_baseline_dates"] == ALIGNMENT_MIN_DATES
    assert r.details["min_baseline_cells"] == ALIGNMENT_MIN_CELLS
