"""Readable diagnostic messages, demo entry points, and compatibility contracts.

Large t-statistics use compact formatting. Turnover and placebo messages name
the evidence actually measured. Demo help, plain-script imports, and
class-scoped pytest fixtures keep their contracts.
"""
from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qaudit import demo
from qaudit._stats import fmt_tstat
from qaudit.checks import costs, leakage, lookahead, performance, survivorship
from qaudit.report import AuditReport
from qaudit.types import errored, passed, warned
from qaudit.config import AuditConfig
from qaudit.dynamic import probes_null, probes_shift
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import momentum_signal, simulate_market
from qaudit.types import Severity, Status

CFG = AuditConfig()
C_COSTS = "costs.missing_transaction_costs"
C_TURNOVER = "costs.turnover_unrealistic"
REPO = Path(__file__).resolve().parents[1]


def _by(results):
    return {r.check: r for r in results}


def _rankw(df):
    r = df.rank(axis=1)
    r = r.sub(r.mean(axis=1), axis=0)
    return r.div(r.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# 1. t-stat rendering
# ---------------------------------------------------------------------------

def test_fmt_tstat_ordinary_range_unchanged():
    # ordinary-magnitude t-stats must render exactly as {t:.1f} would
    assert fmt_tstat(12.34) == "12.3"
    assert fmt_tstat(-3.456) == "-3.5"
    assert fmt_tstat(0.0) == "0.0"
    assert fmt_tstat(999.94) == "999.9"


def test_fmt_tstat_absurd_magnitudes_compact():
    assert fmt_tstat(2.4076487682995837e17) == "2.41e+17"
    assert fmt_tstat(-1.5230974034544682e17) == "-1.52e+17"
    assert fmt_tstat(1e3) == "1e+03"
    assert fmt_tstat(float("nan")) == "nan"
    assert fmt_tstat(float("inf")) == "inf"
    assert fmt_tstat(float("-inf")) == "-inf"


def test_same_bar_bleed_degenerate_book_message_stays_readable():
    # positions == same-bar signal ranks at declared lag 1: partial corr
    # +1.000 on every date, NW t ~1e17 - still CRITICAL, but the message
    # must not carry an 18-digit integer
    mkt = simulate_market(seed=0, n_assets=30, n_periods=400)
    rets = mkt["returns"]
    sig = momentum_signal(rets)
    art = BacktestArtifacts(signals=sig, asset_returns=rets,
                            positions=_rankw(sig), signal_lag=1)
    art.validate()
    r = lookahead._same_bar_bleed(art.aligned(), CFG)
    assert r.status is Status.FAIL
    assert r.severity is Severity.CRITICAL
    assert r.details["nw_tstat"] > 1e6            # the degenerate regime
    assert not re.search(r"\d{8,}", r.message), r.message
    assert re.search(r"NW t [+-]?\d\.\d\de\+\d\d", r.message), r.message


def test_no_raw_one_decimal_tstat_formats_remain():
    # package-wide: no check module may format a t-stat with a bare :.1f
    # (gates and bars are bounded constants and may)
    pat = re.compile(r"\{(?:abs\()?(?:tstat|sep_t|disc_t|p_t|t)\)?:\.1f\}")
    for mod in (lookahead, performance, leakage, survivorship, costs,
                probes_null, probes_shift):
        hits = pat.findall(inspect.getsource(mod))
        assert not hits, (mod.__name__, hits)


# ---------------------------------------------------------------------------
# 2. turnover level: basis stated, zero-post-entry-bar span SKIPs
# ---------------------------------------------------------------------------

def _round_trip_art(declared=None):
    # A one-bar round trip trades two dollars around a one-bar live span.
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=60)
    rets = pd.DataFrame(rng.normal(0, 0.01, (60, 3)), index=dates,
                        columns=list("ABC"))
    pos = pd.DataFrame(0.0, index=dates, columns=rets.columns)
    pos.iloc[30, 0] = 1.0
    gross = (pos * rets.fillna(0.0)).sum(axis=1)
    return BacktestArtifacts(signals=rets.rolling(5).mean(),
                             asset_returns=rets, positions=pos,
                             strategy_returns=gross,
                             declared_costs_bps=declared).aligned()


def _alternating_art(one_sided_to=0.15, n=200):
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=n)
    cols = list("ABCD")
    rets = pd.DataFrame(rng.normal(0.0, 1e-3, (n, 4)), index=dates,
                        columns=cols)
    a = np.array([0.25 + one_sided_to / 2, 0.25 - one_sided_to / 2, 0.25, 0.25])
    b = np.array([0.25 - one_sided_to / 2, 0.25 + one_sided_to / 2, 0.25, 0.25])
    w = np.where((np.arange(n) % 2 == 0)[:, None], a, b)
    pos = pd.DataFrame(w, index=dates, columns=cols)
    gross = (pos * rets).sum(axis=1)
    return BacktestArtifacts(signals=rets.rolling(5).mean(),
                             asset_returns=rets, positions=pos,
                             strategy_returns=gross,
                             declared_costs_bps=5.0).aligned()


def test_one_bar_round_trip_turnover_level_skips_instead_of_zero():
    res = _by(costs.run(_round_trip_art(), CFG))
    assert res[C_COSTS].status is Status.FAIL          # the same trade
    assert res[C_COSTS].details["mean_turnover"] >= 0.5
    r = res[C_TURNOVER]
    assert r.status is Status.SKIP, r.message
    assert r.details["n_post_entry_periods"] == 0
    assert r.details["mean_turnover_entry_inclusive"] >= 0.5
    assert not re.search(r"(?<![\d.])0\.0%", r.message), r.message
    assert "single bar" in r.message
    assert "entry" in r.message and C_COSTS in r.message
    assert "within" not in r.message                   # no realism claim


def test_turnover_level_states_basis_and_entry_inclusive_figure():
    r = _by(costs.run(_alternating_art(0.15), CFG))[C_TURNOVER]
    assert r.status is Status.PASS
    d = r.details
    assert d["mean_turnover"] == pytest.approx(0.15, abs=0.01)
    assert d["n_post_entry_periods"] == d["n_live_periods"] - 1
    # entry bar (|w0| = 1.0 -> 0.5 one-sided) lifts the inclusive mean
    assert d["mean_turnover_entry_inclusive"] > d["mean_turnover"]
    assert d["mean_turnover_entry_inclusive"] == pytest.approx(
        (0.5 + 0.15 * d["n_post_entry_periods"]) / d["n_live_periods"],
        abs=0.005)
    assert "entry trade excluded" in r.message
    assert "post-entry bars" in r.message
    assert C_COSTS in r.message
    assert "live span" not in r.message                # full-span book


def test_turnover_level_basis_clause_on_warn_and_fail_branches():
    warn = _by(costs.run(_alternating_art(0.40), CFG))[C_TURNOVER]
    assert warn.status is Status.WARN
    assert "entry trade excluded" in warn.message
    fail = _by(costs.run(_alternating_art(1.0), CFG))[C_TURNOVER]
    assert fail.status is Status.FAIL
    assert "entry trade excluded" in fail.message


# ---------------------------------------------------------------------------
# 3. placebo drag clause conditional on the sign
# ---------------------------------------------------------------------------

def _placebo_pass(act):
    r = probes_null._check_placebo_bias(CFG, act, act, 0.5)
    assert r.status is Status.PASS, r.message
    return r.message


def test_placebo_pass_drag_clause_only_for_negative_mean():
    noise = np.random.default_rng(1).normal(0, 0.3, 50)
    pos_msg = _placebo_pass(np.full(50, 0.05) + noise)
    assert "slightly negative mean is the expected" not in pos_msg
    assert "non-negative" in pos_msg
    assert f"{CFG.placebo_null_sharpe_fail:.2f} bar" in pos_msg
    neg_msg = _placebo_pass(-np.abs(np.full(50, 0.05) + noise))
    assert "slightly negative mean is the expected transaction-cost drag" \
        in neg_msg
    assert "non-negative" not in neg_msg


def test_placebo_pass_names_the_t_gate_when_mean_clears_the_bar():
    # mean above the 0.50 bar with t < 3: PASS on the t co-gate alone -
    # the message must say so rather than assert a drag
    act = np.full(50, 1.0) + np.random.default_rng(1).normal(0, 3.0, 50)
    assert act.mean() >= CFG.placebo_null_sharpe_fail
    msg = _placebo_pass(act)
    assert "slightly negative" not in msg
    assert "under the 3 gate" in msg
    assert "under the 0.50 bar" not in msg


# ---------------------------------------------------------------------------
# 4. demo CLI
# ---------------------------------------------------------------------------

def test_demo_help_exits_zero_and_lists_cases(capsys):
    from qaudit import synthetic
    with pytest.raises(SystemExit) as exc:
        demo.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in synthetic.ALL_CASES:
        assert name in out
    assert "-h" in out


def test_demo_bogus_case_still_exits_two_with_the_list(capsys):
    from qaudit import synthetic
    assert demo.main(["bogus"]) == 2
    err = capsys.readouterr().err
    assert "unknown case(s): bogus" in err
    for name in synthetic.ALL_CASES:
        assert name in err


def test_demo_parser_defaults_to_all_cases():
    assert demo.build_parser().parse_args([]).cases == []
    assert demo.build_parser().parse_args(["clean", "overfit"]).cases == \
        ["clean", "overfit"]


def test_demo_parser_supports_compact_presentation_mode():
    args = demo.build_parser().parse_args(["--summary-only", "clean"])
    assert args.summary_only is True
    assert args.list_cases is False
    assert args.cases == ["clean"]


def test_demo_list_prints_each_available_case_once(capsys):
    from qaudit import synthetic
    assert demo.main(["--list"]) == 0
    assert capsys.readouterr().out.splitlines() == list(synthetic.ALL_CASES)


def test_demo_display_status_does_not_label_warnings_clean():
    assert demo.display_status(AuditReport([passed("demo.clean", "ok")])) == "CLEAN"
    assert demo.display_status(AuditReport([warned("demo.warning", "review")])) == \
        "FLAGGED"
    assert demo.display_status(AuditReport([errored("demo.error", RuntimeError())])) == \
        "ERROR"


def test_demo_runs_as_plain_script_without_shadowing_stdlib_types():
    # `python3 src/qaudit/demo.py` puts src/qaudit/ at sys.path[0]; a clean
    # interpreter can then resolve `import types` (via re/enum) to
    # qaudit/types.py and die before argparse - --help is the cheapest
    # end-to-end proof that startup survives in this interpreter
    proc = subprocess.run(
        [sys.executable, str(REPO / "src" / "qaudit" / "demo.py"), "--help"],
        cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "same_bar_execution" in proc.stdout
    proc = subprocess.run(
        [sys.executable, str(REPO / "src" / "qaudit" / "demo.py"), "bogus"],
        cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert "available:" in proc.stderr


# ---------------------------------------------------------------------------
# 5. no class-scoped fixtures as instance methods (pytest 9 deprecation)
# ---------------------------------------------------------------------------

def test_no_class_scoped_fixture_defined_as_instance_method():
    pat = re.compile(r"^[ \t]+@pytest\.fixture\([^)]*scope=", re.M)
    offenders = [p.name for p in sorted(REPO.glob("tests/test_qaudit_*.py"))
                 if pat.search(p.read_text())]
    assert not offenders, offenders
