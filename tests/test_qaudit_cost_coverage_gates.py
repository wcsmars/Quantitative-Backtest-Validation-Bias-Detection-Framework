"""Cost reconciliation requires sufficient dates and traded-dollar coverage.

Flat padding does not add evidence inside a short live span. Sparse or
materially incomplete comparisons cannot certify costs, while direct
evidence of missing charges remains actionable. Sibling declaration
messages and machine-readable flags reflect the actual measurement.

Every fixture calls aligned() without validate(): intake already refuses
strategy_returns with fewer than 30 finite values, and these tests pin the
check's own verdict independently of that defence in depth.
"""
from __future__ import annotations

import numpy as np
import pytest

from qaudit._stats import gross_strategy_returns, traded_dollars_series
from qaudit.checks import costs
from qaudit.checks.costs import (COST_RECONCILE_MIN_DATES,
                                 MIN_OVERLAP_TRADED_SHARE)
from qaudit.config import AuditConfig
from qaudit.errors import AuditFailure
from qaudit.inputs import BacktestArtifacts
from qaudit.report import AuditReport
from qaudit.synthetic import (momentum_signal, net_returns,
                              positions_from_signals, simulate_market)
from qaudit.types import Severity, Status

CFG = AuditConfig()
C1 = "costs.missing_transaction_costs"
C2 = "costs.no_cost_declaration"


def _by(results):
    return {r.check: r for r in results}


@pytest.fixture(scope="module")
def book():
    """A 500-date honest 10bps book plus its live (trading) dates."""
    mk = simulate_market(n_assets=20, n_periods=500, seed=0, death_frac=0.0)
    rets = mk["returns"]
    sig = momentum_signal(rets)
    pos = positions_from_signals(sig, 1)
    net = net_returns(pos, rets, 10.0)
    dollars = traded_dollars_series(pos, rets)
    live = dollars[dollars > 0].index
    return dict(rets=rets, sig=sig, pos=pos, net=net, dollars=dollars,
                live=live)


def _run(book, sel, *, declared=10.0, sr=None):
    """Audit the book with strategy_returns reported only on ``sel``.
    Drives aligned() directly, bypassing validate()."""
    sr = book["net"].loc[sel] if sr is None else sr
    art = BacktestArtifacts(signals=book["sig"], asset_returns=book["rets"],
                            positions=book["pos"], strategy_returns=sr,
                            declared_costs_bps=declared).aligned()
    return _by(costs.run(art, CFG))


# ---------------------------------------------------------------------------
# (a) too few compared dates: SKIP, never a certificate
# ---------------------------------------------------------------------------

def test_single_date_is_skip_not_pass(book):
    # Only one mid-book date reports returns, with honest 10 bps costs on that date.
    r = _run(book, [book["live"][100]])[C1]
    assert r.status is Status.SKIP, r.message
    assert "1 compared dates" in r.message
    assert "cannot be verified" in r.message
    assert "full live position grid" in r.message
    assert r.details["n_common"] == 1
    assert r.details["implied_bps"] == pytest.approx(10.0, abs=0.5)
    assert r.details["overlap_traded_share"] < 0.01
    assert "superseded" not in r.details       # a real gap, not a hand-off


@pytest.mark.parametrize("k", [5, 10, 29])
def test_below_min_dates_is_skip(book, k):
    r = _run(book, list(book["live"][100:100 + k]))[C1]
    assert r.status is Status.SKIP, r.message
    assert f"only {k} compared dates (< {COST_RECONCILE_MIN_DATES})" in r.message
    assert r.details["n_common"] == k


def test_single_date_bit_identical_still_fails_critical(book):
    # detection-power guard: the accusatory paths run on any overlap
    d = book["live"][100]
    gross = gross_strategy_returns(book["pos"], book["rets"])
    r = _run(book, [d], sr=gross.loc[[d]])[C1]
    assert r.status is Status.FAIL, r.message
    assert r.severity is Severity.CRITICAL
    assert r.details["n_common"] == 1
    assert "overlap_traded_share" in r.details


def test_attack_costs_charged_only_on_reported_dates_not_certified(book):
    # the book is free everywhere except the 5 dates it reports on
    sel = list(book["live"][100:105])
    gross = gross_strategy_returns(book["pos"], book["rets"])
    net_attack = gross.copy()
    net_attack.loc[sel] = book["net"].loc[sel]
    r = _run(book, sel, sr=net_attack.loc[sel])[C1]
    assert r.status is Status.SKIP, r.message


def test_gate_refuses_single_date_book(book):
    # the fail-closed deployment API must raise: the check never judged
    rep = AuditReport(results=list(_run(book, [book["live"][100]]).values()))
    assert rep.ok                              # advisory verdict stays green
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=[C1])
    assert "cannot be verified" in str(ei.value)
    assert "99% verification floor" in str(ei.value)


@pytest.mark.parametrize("shape, declared, status, severity", [
    ("negative", 10.0, Status.SKIP, None),                # incomplete evidence
    ("under_charged", 10.0, Status.SKIP, None),           # incomplete evidence
    ("sub_floor", 10.0, Status.FAIL, Severity.CRITICAL),  # 0.1bp jitter
    ("ceiling", 10.0, Status.SKIP, None),                 # incomplete evidence
    ("paper", 0.0, Status.SKIP, None),                    # declaration != coverage
    ("flat_deduction", 10.0, Status.SKIP, None),          # incomplete evidence
])
def test_five_date_overlap_is_fail_closed(book, shape, declared,
                                          status, severity):
    # Definitive evidence that costs were absent (sub-floor) remains a FAIL.
    # Every inconclusive/scoped verdict is instead SKIP because five selected
    # dates cannot verify the unreported traded dollars.
    sel = list(book["live"][100:105])
    pos, rets = book["pos"], book["rets"]
    gross = gross_strategy_returns(pos, rets)
    series = {
        "negative": gross + 0.001,
        "under_charged": net_returns(pos, rets, 1.0),
        "sub_floor": net_returns(pos, rets, 0.1),
        "ceiling": net_returns(pos, rets, 2000.0),
        "paper": gross,
        "flat_deduction": gross - 0.0003,
    }[shape]
    r = _run(book, sel, declared=declared, sr=series.loc[sel])[C1]
    assert r.status is status, r.message
    if severity is not None:
        assert r.severity is severity
    assert r.details["n_common"] == 5


def _short_live_book(book, n_live, start=200):
    """Positions live on ``n_live`` consecutive dates of the 500-date grid
    (flat elsewhere); honest 10bps net reported on the full grid, 0.0 on
    the padding - so n_common is the whole grid while the live span is
    ``n_live`` dates."""
    pos = book["pos"].fillna(0.0) * 0.0
    pos.iloc[start:start + n_live] = book["pos"].iloc[start:start + n_live]
    net = net_returns(pos, book["rets"], 10.0)
    art = BacktestArtifacts(signals=book["sig"], asset_returns=book["rets"],
                            positions=pos, strategy_returns=net,
                            declared_costs_bps=10.0).aligned()
    return _by(costs.run(art, CFG))[C1]


def test_short_live_span_is_skip_despite_full_grid_net(book):
    # Flat padding adds common dates but no trades or drag; cost certification requires
    # compared dates inside the live span.
    r = _short_live_book(book, 10)
    assert r.status is Status.SKIP, r.message
    nc = r.details["n_common"]
    # the date-count floor alone is cleared; the live-span prong is what SKIPs
    assert nc >= COST_RECONCILE_MIN_DATES
    assert r.details["n_live_dates"] == 10
    assert r.details["n_compared_live_dates"] == 10
    assert r.details["overlap_traded_share"] == pytest.approx(1.0, abs=1e-9)
    assert (f"only 10 of the {nc} compared dates fall in the positions' "
            f"live span (< {COST_RECONCILE_MIN_DATES}") in r.message
    assert "cannot be verified" in r.message
    assert "traded dollars, including final liquidation)" in r.message


def test_live_span_at_floor_with_full_grid_net_passes(book):
    # honest guard: a 40-live-date book reported on the full grid clears
    # both prongs of the floor and is certified at 100% coverage
    r = _short_live_book(book, 40)
    assert r.status is Status.PASS, r.message
    assert r.details["n_compared_live_dates"] == 40
    assert r.details["n_live_dates"] == 40
    assert r.message.endswith(
        "verified on 40 of 40 position dates (100% of traded dollars).")


# ---------------------------------------------------------------------------
# (b) essentially-complete traded-dollar coverage is required
# ---------------------------------------------------------------------------

def test_full_overlap_passes_with_share_in_details(book):
    r = _run(book, list(book["net"].index))[C1]
    assert r.status is Status.PASS, r.message
    assert r.details["overlap_traded_share"] == pytest.approx(1.0, abs=1e-9)
    assert r.details["n_live_dates"] == len(book["live"])
    assert r.details["n_compared_live_dates"] == len(book["live"])
    assert r.message.endswith(
        f"verified on {len(book['live'])} of {len(book['live'])} position "
        f"dates (100% of traded dollars).")
    assert "not_judged" not in r.details and "unresolved" not in r.details


def test_35_of_500_dates_skips_as_unjudged(book):
    r = _run(book, list(book["live"][100:135]))[C1]
    assert r.status is Status.SKIP, r.message
    assert r.details["n_common"] == 35
    assert r.details["overlap_traded_share"] < MIN_OVERLAP_TRADED_SHARE
    assert r.details["implied_bps"] == pytest.approx(10.0, abs=0.5)
    assert "not judged" in r.message
    assert "99% verification floor" in r.message
    assert "unverified" in r.message


def test_incomplete_coverage_blocks_default_strict_gate(book):
    rep = AuditReport(results=list(
        _run(book, list(book["live"][100:135])).values()))
    assert rep.ok
    with pytest.raises(AuditFailure) as ei:
        rep.gate(require=[C1])
    assert "99% verification floor" in str(ei.value)


def test_honest_late_start_60pct_still_passes(book):
    # Returns and positions cover the same final 60% of the grid, so the reported span
    # covers the entire live book.
    pos, net = book["pos"], book["net"]
    cut = int(0.4 * len(pos))
    art = BacktestArtifacts(signals=book["sig"], asset_returns=book["rets"],
                            positions=pos.iloc[cut:],
                            strategy_returns=net.iloc[cut:],
                            declared_costs_bps=10.0)
    art.validate()
    r = _by(costs.run(art.aligned(), CFG))[C1]
    assert r.status is Status.PASS, r.message
    assert r.details["overlap_traded_share"] >= 0.6
    assert r.details["n_live_dates"] == r.details["n_compared_live_dates"]


def test_valid_leading_truncation_with_full_positions_skips_cost_coverage(book):
    # Exact attack through intake: the dense net series reports only the back
    # 60% (a legal leading truncation), but positions prove the book traded in
    # the withheld front 40%. Validation accepts the shape; the cost check
    # must still SKIP and a strict exact-id gate must reject it.
    pos, net = book["pos"], book["net"]
    cut = int(0.4 * len(pos))
    art = BacktestArtifacts(signals=book["sig"], asset_returns=book["rets"],
                            positions=pos, strategy_returns=net.iloc[cut:],
                            declared_costs_bps=10.0)
    art.validate()
    r = _by(costs.run(art.aligned(), CFG))[C1]
    assert r.status is Status.SKIP, r.message
    share = r.details["overlap_traded_share"]
    assert 0.5 < share < 0.7
    assert f"covers {share:.1%} of dollars traded" in r.message
    assert "not judged" in r.message
    assert r.details["n_compared_live_dates"] < r.details["n_live_dates"]
    with pytest.raises(AuditFailure):
        AuditReport(results=[r]).gate(require=[C1])


def test_full_vs_materially_incomplete_share_flips_pass_to_skip(book):
    assert MIN_OVERLAP_TRADED_SHARE == 0.99
    n_live = len(book["live"])
    partial = list(book["live"][: int(0.95 * n_live)])
    r_full = _run(book, list(book["net"].index))[C1]
    r_partial = _run(book, partial)[C1]
    assert r_full.details["overlap_traded_share"] >= MIN_OVERLAP_TRADED_SHARE
    assert r_partial.details["overlap_traded_share"] < MIN_OVERLAP_TRADED_SHARE
    assert r_full.status is Status.PASS, r_full.message
    assert r_partial.status is Status.SKIP, r_partial.message


# ---------------------------------------------------------------------------
# (c) the declaration check never claims verification off a SKIP / WARN
# ---------------------------------------------------------------------------

def test_declaration_warns_when_reconstruction_skipped_on_few_dates(book):
    res = _run(book, list(book["live"][100:105]), declared=None)
    assert res[C1].status is Status.SKIP
    r = res[C2]
    assert r.status is Status.WARN
    assert r.severity is Severity.MEDIUM
    assert "was verified" not in r.message
    assert "verified against" not in r.message
    assert "could not verify" in r.message
    assert "are missing" not in r.message      # the artifacts are present
    assert "SKIPped" in r.message
    assert r.details["verified_by_reconstruction"] is False
    assert r.details["reconstruction_check_status"] == "skip"
    assert r.remediation


def test_declaration_missing_artifacts_warn_unchanged(book):
    # genuinely absent artifacts get the separate missing-artifacts WARN wording
    art = BacktestArtifacts(signals=book["sig"], asset_returns=book["rets"],
                            positions=None, strategy_returns=None,
                            declared_costs_bps=None).aligned()
    r = _by(costs.run(art, CFG))[C2]
    assert r.status is Status.WARN
    assert "nothing to verify them against" in r.message


def test_missing_declaration_warns_when_coverage_is_unjudged(book):
    res = _run(book, list(book["live"][100:135]), declared=None)
    assert res[C1].status is Status.SKIP
    r = res[C2]
    assert r.status is Status.WARN and r.severity is Severity.MEDIUM
    assert "was verified" not in r.message
    assert "could not verify" in r.message
    assert r.details["verified_by_reconstruction"] is False
    assert r.details["reconstruction_check_status"] == "skip"


def test_declaration_verified_only_on_full_pass(book):
    r = _run(book, list(book["net"].index), declared=None)[C2]
    assert r.status is Status.PASS
    assert "verified against the gross reconstruction" in r.message
    assert r.details["verified_by_reconstruction"] is True
    assert "not_judged" not in r.details and "unresolved" not in r.details


def test_declared_costs_pass_unchanged_by_sibling_skip(book):
    # a declaration on record is still a declaration on record
    r = _run(book, [book["live"][100]], declared=10.0)[C2]
    assert r.status is Status.PASS
    assert "declared at 10bps" in r.message


# ---------------------------------------------------------------------------
# (d) verdict flags on the hedged PASS sites
# ---------------------------------------------------------------------------

def _drift_only_book():
    rng = np.random.default_rng(3)
    import pandas as pd
    dates = pd.bdate_range("2020-01-01", periods=150)
    rets = pd.DataFrame(rng.normal(3e-4, 0.01, (150, 3)), index=dates,
                        columns=list("ABC"))
    r_arr = rets.to_numpy()
    w = np.empty_like(r_arr)
    w[0] = 1.0 / 3.0
    for t in range(1, len(w)):
        w[t] = w[t - 1] * (1 + r_arr[t - 1]) / (1 + w[t - 1] @ r_arr[t - 1])
    pos = pd.DataFrame(w, index=dates, columns=rets.columns)
    gross = (pos * rets).sum(axis=1)
    return BacktestArtifacts(signals=pos.copy(), asset_returns=rets,
                             positions=pos, strategy_returns=gross).aligned()


def test_negligible_turnover_pass_flags_unresolved():
    res = _by(costs.run(_drift_only_book(), CFG))
    assert res[C1].status is Status.PASS
    assert res[C1].details["negligible_turnover"] is True
    assert res[C1].details["unresolved"] is True
    assert "not verified" in res[C1].message
    # ...and the declaration's "NOT verified" PASS carries the same flag
    assert res[C2].status is Status.PASS
    assert res[C2].details["unresolved"] is True
    assert res[C2].details["verified_by_reconstruction"] is False


def test_delegation_pass_flags_not_judged_on_fail(book):
    gross = gross_strategy_returns(book["pos"], book["rets"])
    res = _run(book, list(gross.index), declared=None, sr=gross)
    assert res[C1].status is Status.FAIL
    assert res[C2].status is Status.PASS
    assert res[C2].details["not_judged"] is True
