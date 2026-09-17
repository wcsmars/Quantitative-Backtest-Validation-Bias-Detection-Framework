"""Date-shift ambiguity, full-grid signal reproduction, and callback output validation.

Causal mild blends receive scoped warnings. Missing callback dates count as
unreproduced cells, while genuine sparse panels remain valid. Duplicate
labels and malformed outputs cannot grant a causal-probe stamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.config import AuditConfig
from qaudit.dynamic import probes_shift
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import (make_backtest_func, momentum_signal,
                              net_returns, positions_from_signals,
                              simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()

MILD_BLEND_W = 0.2          # same-bar MR weight that lands in the ambiguous band
AMBIGUOUS_SEEDS = (0, 6)    # market seeds whose k=-1 SR falls between shapes


def _zx(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def _arts(market, sig, **kw):
    rets = market["returns"]
    pos = positions_from_signals(sig, 1)
    base = dict(signals=sig, asset_returns=rets, positions=pos,
                strategy_returns=net_returns(pos, rets, 10.0),
                universe=market.get("universe"), signal_lag=1,
                declared_costs_bps=10.0, signal_input=rets)
    base.update(kw)
    art = BacktestArtifacts(**base)
    art.validate()
    return art.aligned()


def _by(results):
    out = {r.check: r for r in results}
    # the module contract: exactly these four ids, once each
    assert set(out) == {probes_shift.CHECK_DATE_SHIFT, probes_shift.CHECK_REPRO,
                        probes_shift.CHECK_TRUNCATION,
                        probes_shift.CHECK_SENSITIVITY}
    assert len(results) == 4
    return out


def mild_blend_signal(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Honest causal signal: momentum plus a small same-bar mean-reversion
    loading. signals.loc[t] uses returns up to and including t - legal under
    the timing convention (positions.loc[t] use signals.loc[t-1])."""
    mom = momentum_signal(asset_returns)
    mr = -_zx(asset_returns)
    return ((1 - MILD_BLEND_W) * mom + MILD_BLEND_W * mr).where(mom.notna())


# ---------------------------------------------------------------------------
# 1. the ambiguous k=-1 band: honest mild same-bar blend
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", AMBIGUOUS_SEEDS)
def test_mild_blend_unverified_is_warn_not_critical(seed):
    market = simulate_market(seed=seed)
    art = _arts(market, mild_blend_signal(market["returns"]))
    r = _by(probes_shift.run(art, CFG, backtest_func=make_backtest_func()))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.WARN
    assert r.severity is Severity.HIGH
    d = r.details
    assert d["peek_signature"] == "ambiguous"
    assert d["signal_verified_causal"] is False
    # the band is real: meaningfully negative yet short of the anti-corr bar
    assert d["sr_peek_1"] <= -d["peek_noise_floor_sr"]
    assert d["sr_peek_1"] > d["peek_anti_corr_bar_sr"]
    # actionable: says what evidence adjudicates and carries the numbers
    assert "signal_func" in r.remediation
    assert f"{d['sr_baseline']:.1f}" in r.message


def test_mild_blend_verified_causal_downgrades_to_medium():
    market = simulate_market(seed=AMBIGUOUS_SEEDS[0])
    art = _arts(market, mild_blend_signal(market["returns"]))
    res = _by(probes_shift.run(art, CFG, signal_func=mild_blend_signal,
                               backtest_func=make_backtest_func()))
    # the probes really did verify causality (the exculpatory evidence)
    assert res[probes_shift.CHECK_REPRO].status is Status.PASS
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.PASS
    r = res[probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert r.details["peek_signature"] == "ambiguous"
    assert r.details["signal_verified_causal"] is True
    # the message must present the ambiguity, not assert a leak as fact
    assert "ambiguous" in r.message
    assert "verified causal" in r.message
    assert "same-bar loading" in r.remediation


# --- detection-power guards: real leaks keep their CRITICAL -----------------

def test_welded_momentum_leak_still_critical_evaporation():
    # The earlier-shift Sharpe collapses into the noise band; the welded leak must
    # retain its CRITICAL failure.
    market = simulate_market(seed=7)
    rets = market["returns"]
    honest = momentum_signal(rets)
    sig = (0.25 * honest.fillna(0.0) + _zx(rets.shift(-1))).where(honest.notna())
    art = _arts(market, sig)
    r = _by(probes_shift.run(art, CFG, backtest_func=make_backtest_func()))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["peek_signature"] == "evaporation"


def test_cost_dragged_target_leak_still_critical():
    """A pure target leak on a high-turnover book: at k=-1 the foresight is
    gone and transaction costs drag the remnant several SR units negative
    (~-3..-4.7 measured). 'Meaningfully negative' alone must not read as the
    ambiguous shape - the remnant is tiny relative to the SR~100 baseline,
    so the commensurability gate keeps it in evaporation -> CRITICAL."""
    market = simulate_market(seed=3)
    rets = market["returns"]
    rng = np.random.default_rng(4)
    fwd = rets.shift(-1)
    noise = pd.DataFrame(rng.normal(0, 0.2 * fwd.stack().std(), fwd.shape),
                         index=fwd.index, columns=fwd.columns)
    sig = (fwd + noise).where(rets.notna())
    art = _arts(market, sig)
    r = _by(probes_shift.run(art, CFG, backtest_func=make_backtest_func()))[
        probes_shift.CHECK_DATE_SHIFT]
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    d = r.details
    assert d["peek_signature"] == "evaporation"
    # the test only bites if the remnant really is below the noise floor:
    # an absolute-floor rule would have called this "ambiguous"
    assert d["sr_peek_1"] < -d["peek_noise_floor_sr"]


def test_clean_momentum_book_stays_clean():
    market = simulate_market(seed=9)
    art = _arts(market, momentum_signal(market["returns"]))
    res = _by(probes_shift.run(art, CFG, signal_func=momentum_signal,
                               backtest_func=make_backtest_func()))
    for r in res.values():
        assert r.status is Status.PASS


# ---------------------------------------------------------------------------
# 2. coverage-floor slack: excluded audited cells must count as unreproduced
# ---------------------------------------------------------------------------

def _leaky_slack_case(seed=0):
    """Honest momentum panel, except every 21st date carries the next-day
    return (~4.8% of dates, net SR 3.6 vs 1.9 honest). The laundering
    signal_func excludes exactly those dates from its output index."""
    market = simulate_market(seed=seed)
    rets = market["returns"]
    honest = momentum_signal(rets)
    leak_dates = rets.index[30::21]
    mask = pd.Series(rets.index.isin(leak_dates), index=rets.index)
    leaky = honest.copy()
    leaky[mask.values] = (5.0 * _zx(rets.shift(-1)))[mask.values]
    leaky = leaky.where(honest.notna())

    def slack_func(x):
        out = momentum_signal(x)
        return out.loc[~x.index.isin(leak_dates)]

    return market, leaky, slack_func


def test_excluded_leaky_dates_cannot_collect_causal_stamp():
    market, leaky, slack_func = _leaky_slack_case()
    art = _arts(market, leaky)
    res = _by(probes_shift.run(art, CFG, signal_func=slack_func))
    r = res[probes_shift.CHECK_REPRO]
    # Unreported leaky dates count toward reproduction mismatch even if aggregate
    # coverage clears its floor.
    assert r.status is Status.WARN
    assert r.details["frac_mismatch"] > 0.01
    assert r.details["coverage"] >= 0.95        # the floor alone did not catch it
    assert "not even cover" in r.message
    # All three signal probes must pass to grant the stamp. A reproduction warning
    # denies it; the conjunction is also covered in
    # test_qaudit_probe_input_isolation.py.
    assert not (res[probes_shift.CHECK_REPRO].status is Status.PASS
                and res[probes_shift.CHECK_TRUNCATION].status is Status.PASS
                and res[probes_shift.CHECK_SENSITIVITY].status is Status.PASS)


def test_honest_sparse_monthly_panel_keeps_clean_verdict():
    """An audited signal finite only at monthly stamps can be fully reproduced by its
    callback. A causal stamp must remain obtainable for every cell actually used."""
    market = simulate_market(seed=0)
    rets = market["returns"]
    stamps = rets.index[::21]

    def sparse_func(x):
        return momentum_signal(x).where(
            pd.Series(x.index.isin(stamps), index=x.index), np.nan)

    sparse = sparse_func(rets)
    assert sparse.notna().to_numpy().mean() < 0.06      # the panel is sparse
    art = _arts(market, sparse)
    res = _by(probes_shift.run(art, CFG, signal_func=sparse_func))
    assert res[probes_shift.CHECK_REPRO].status is Status.PASS
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.PASS


def test_one_cell_laundering_still_blocked():
    # A one-cell callback cannot reproduce the audited grid and must retain its warning
    # and vacuous-verdict skip.
    market = simulate_market(seed=2)
    rets = market["returns"]
    ma = rets.rolling(20, min_periods=20).mean()
    sig = (ma - ma.mean()) / ma.std()

    def one_cell(x):
        m = x.rolling(20, min_periods=20).mean()
        return ((m - m.mean()) / m.std()).iloc[[-1], [0]]

    art = _arts(market, sig)
    res = _by(probes_shift.run(art, CFG, signal_func=one_cell))
    assert res[probes_shift.CHECK_REPRO].status is Status.WARN
    assert res[probes_shift.CHECK_REPRO].details["coverage"] < 0.01
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.SKIP


# ---------------------------------------------------------------------------
# 3. malformed signal_func output: clean WARN, not a raw pandas crash
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def honest_art():
    market = simulate_market(seed=0)
    return _arts(market, momentum_signal(market["returns"]))


def test_duplicate_column_output_warns_cleanly(honest_art):
    def dup_col(x):
        out = momentum_signal(x)
        return pd.concat([out, out.iloc[:, [0]]], axis=1)

    res = _by(probes_shift.run(honest_art, CFG, signal_func=dup_col))
    r = res[probes_shift.CHECK_REPRO]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert "duplicated column" in r.message
    assert "signal_func" in r.message
    assert r.remediation
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.SKIP


def test_duplicate_index_output_warns_cleanly(honest_art):
    def dup_idx(x):
        out = momentum_signal(x)
        return pd.concat([out, out.iloc[[5]]], axis=0)

    res = _by(probes_shift.run(honest_art, CFG, signal_func=dup_idx))
    r = res[probes_shift.CHECK_REPRO]
    assert r.status is Status.WARN
    assert "duplicated index" in r.message
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.SKIP


def test_non_dataframe_output_warns_cleanly(honest_art):
    res = _by(probes_shift.run(honest_art, CFG,
                               signal_func=lambda x: x.iloc[:, 0]))
    r = res[probes_shift.CHECK_REPRO]
    assert r.status is Status.WARN
    assert "not a DataFrame" in r.message
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.SKIP


def test_malformed_output_never_grants_causal_stamp():
    """Malformed output cannot grant causal_verified. A mild-blend ambiguous book forces
    the date-shift branch to populate and use the stamp, so the assertion exercises the
    actual decision."""
    def dup_col(x):
        out = mild_blend_signal(x)
        return pd.concat([out, out.iloc[:, [0]]], axis=1)

    market = simulate_market(seed=AMBIGUOUS_SEEDS[0])
    art = _arts(market, mild_blend_signal(market["returns"]))
    res = _by(probes_shift.run(art, CFG, signal_func=dup_col,
                               backtest_func=make_backtest_func()))
    assert res[probes_shift.CHECK_REPRO].status is Status.WARN
    assert res[probes_shift.CHECK_TRUNCATION].status is Status.SKIP
    r = res[probes_shift.CHECK_DATE_SHIFT]
    assert r.details["signal_verified_causal"] is False
    assert r.severity is Severity.HIGH          # not the verified MEDIUM path
    assert "verified causal" not in r.message
