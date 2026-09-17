"""Artifact and callable provenance, deterministic digests, and supported runtimes.

Input cells, typed labels, and callable state contribute to identity. Labels
alone do not identify closures. Metadata degrades explicitly on unsupported
values.
"""
from __future__ import annotations

import functools
import json
import re
import time
import types

import numpy as np
import pandas as pd

import qaudit.api as api
from qaudit import BacktestArtifacts, audit, synthetic

HEX64 = re.compile(r"^[0-9a-f]{64}$")
DIGESTED = ("signals", "asset_returns", "positions", "strategy_returns",
            "universe", "prices", "signal_input")


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _panels(n=250, k=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"A{i:02d}" for i in range(k)]
    sig = pd.DataFrame(rng.standard_normal((n, k)), index=idx, columns=cols)
    ret = pd.DataFrame(rng.standard_normal((n, k)) * 0.01,
                       index=idx, columns=cols)
    return sig, ret


def _prov(art, **kw):
    # contamination is the cheapest module; provenance is module-agnostic
    kw.setdefault("include", ["contamination"])
    return audit(art, **kw).meta["provenance"]


def _strip(prov):
    return {k: v for k, v in prov.items()
            if k not in ("timestamp_utc", "timings_s", "checks_emitted")}


def _make_backtest(cost_bps: float):
    def bt(s, r):
        return (s * r).sum(axis=1) - cost_bps * 1e-4
    return bt


# Distinct inputs and callables must have distinct provenance.

def test_negated_panel_has_distinct_provenance():
    # Negating signals changes their digest while leaving other panel digests
    # unchanged; callable state is recorded separately.
    sig, ret = _panels()
    a = _prov(BacktestArtifacts(signals=sig, asset_returns=ret))
    b = _prov(BacktestArtifacts(signals=-sig, asset_returns=ret))
    assert _strip(a) != _strip(b)
    assert a["artifact_digest"]["signals"] != b["artifact_digest"]["signals"]
    assert (a["artifact_digest"]["asset_returns"]
            == b["artifact_digest"]["asset_returns"])
    # the thresholds did not change, so config_hash must not either
    assert a["config_hash"] == b["config_hash"]


def test_attack_one_cell_change_flips_only_that_panel():
    sig, ret = _panels()
    base = _prov(BacktestArtifacts(signals=sig, asset_returns=ret))
    sig2 = sig.copy()
    sig2.iloc[17, 3] += 1e-12                   # smallest visible edit
    edited = _prov(BacktestArtifacts(signals=sig2, asset_returns=ret))
    assert edited["artifact_digest"]["signals"] != base["artifact_digest"]["signals"]
    assert (edited["artifact_digest"]["asset_returns"]
            == base["artifact_digest"]["asset_returns"])
    # coverage stats alone are blind to this edit; only the digest sees it
    assert edited["artifact_coverage"] == base["artifact_coverage"]


def test_attack_shuffled_column_labels_flip_the_digest():
    # Same cells, permuted column labels: the audited grid is a different
    # asset<->signal pairing, so the digest must not collapse them.
    sig, ret = _panels()
    d0 = api._panel_digest(sig)
    relabelled = sig.copy()
    relabelled.columns = list(sig.columns)[::-1]
    assert api._panel_digest(relabelled) != d0
    # and a physical column permutation (labels follow the cells) too
    assert api._panel_digest(sig[list(sig.columns)[::-1]]) != d0


def test_attack_relabelled_dates_flip_the_digest():
    sig, _ = _panels()
    shifted = sig.copy()
    shifted.index = sig.index + np.timedelta64(1, "D")
    assert api._panel_digest(shifted) != api._panel_digest(sig)


def test_attack_end_to_end_relabel_via_audit():
    # Through audit(): asset_returns relabelled to a reversed ticker order
    # (cells unchanged) is a different book and gets a different digest.
    sig, ret = _panels()
    ret2 = ret.copy()
    ret2.columns = list(ret.columns)[::-1]
    a = _prov(BacktestArtifacts(signals=sig, asset_returns=ret))
    b = _prov(BacktestArtifacts(signals=sig, asset_returns=ret2))
    assert a["artifact_digest"]["asset_returns"] != b["artifact_digest"]["asset_returns"]


# Digest determinism and provenance field contracts.

def test_identical_inputs_digest_identically_across_runs():
    sig, ret = _panels()
    a = _prov(BacktestArtifacts(signals=sig.copy(), asset_returns=ret.copy()))
    b = _prov(BacktestArtifacts(signals=sig.copy(), asset_returns=ret.copy()))
    assert a["artifact_digest"] == b["artifact_digest"]
    assert a["callables"] == b["callables"] == {"signal_func": None,
                                                "backtest_func": None}


def test_nan_cells_digest_deterministically():
    sig, ret = _panels()
    sig.iloc[::7, ::3] = np.nan
    a = api._panel_digest(sig)
    b = api._panel_digest(sig.copy())
    # nullable extension dtype with pd.NA: same cells, same digest
    c = api._panel_digest(sig.astype("Float64"))
    assert a == b == c
    # and a NaN is content: filling it flips the digest
    assert api._panel_digest(sig.fillna(0.0)) != a


def test_digest_block_covers_every_panel_of_a_rich_book():
    art = synthetic.make_clean(seed=3).artifacts
    prov = _prov(art)
    dig = prov["artifact_digest"]
    assert tuple(dig) == DIGESTED
    for name in DIGESTED:
        assert getattr(art, name) is not None
        assert HEX64.match(dig[name]), (name, dig[name])
    # the strategy_returns Series digests like the DataFrames do
    assert dig["strategy_returns"] != dig["signals"]
    # make_clean feeds asset_returns as signal_input: equal content, equal
    # digest - the digest is content-addressed, not slot-addressed
    assert dig["signal_input"] == dig["asset_returns"]


def test_absent_panels_digest_to_none_and_keys_are_stable():
    sig, ret = _panels()
    dig = _prov(BacktestArtifacts(signals=sig, asset_returns=ret))["artifact_digest"]
    assert tuple(dig) == DIGESTED
    assert HEX64.match(dig["signals"]) and HEX64.match(dig["asset_returns"])
    for name in DIGESTED[2:]:
        assert dig[name] is None


def test_non_numeric_panel_digests_via_row_hash_fallback():
    # validate() rejects text/object columns before provenance runs, so
    # the pandas-row-hash fallback is defensive - but it must be correct
    # (deterministic, content-sensitive) should a future artifact slot
    # admit non-numeric data. Exercised directly.
    raw = pd.DataFrame({"ticker": ["X", "Y", "Z"] * 10,
                        "sector": ["a", "b", "c"] * 10,
                        "px": np.arange(30, dtype=float)},
                       index=pd.bdate_range("2020-01-01", periods=30))
    a = api._panel_digest(raw)
    assert HEX64.match(a)
    assert api._panel_digest(raw.copy()) == a
    raw2 = raw.copy()
    raw2.loc[raw2.index[0], "ticker"] = "W"
    assert api._panel_digest(raw2) != a


def test_nullable_extension_signal_input_digests_through_audit():
    # the pandas-2 nullable mainstream (convert_dtypes / Arrow-backed
    # loaders): Float64 with pd.NA is legal signal_input and must digest
    # identically to its float64+NaN twin
    sig, ret = _panels()
    raw = ret.copy()
    raw.iloc[::5, 0] = np.nan
    a = _prov(BacktestArtifacts(signals=sig, asset_returns=ret,
                                signal_input=raw))
    b = _prov(BacktestArtifacts(signals=sig, asset_returns=ret,
                                signal_input=raw.astype("Float64")))
    assert HEX64.match(a["artifact_digest"]["signal_input"])
    assert a["artifact_digest"]["signal_input"] == b["artifact_digest"]["signal_input"]
    assert "error" not in a and "error" not in b


def test_deps_and_config_hash_semantics_unchanged():
    # Dependency context and configuration hashes must remain separate from input
    # digests.
    sig, ret = _panels()
    a = _prov(BacktestArtifacts(signals=sig, asset_returns=ret))
    b = _prov(BacktestArtifacts(signals=sig * 2, asset_returns=ret))
    assert set(a["deps"]) == {"python", "numpy", "pandas", "scipy"}
    assert a["config_hash"] == b["config_hash"]
    assert a["artifact_digest"] != b["artifact_digest"]


def test_provenance_is_json_round_trip_safe_with_new_keys():
    art = synthetic.make_clean(seed=1).artifacts
    rep = audit(art, include=["contamination"],
                backtest_func=_make_backtest(10.0))
    blob = json.dumps(rep.to_dict(), allow_nan=False)
    back = json.loads(blob)["meta"]["provenance"]
    assert back["artifact_digest"] == rep.meta["provenance"]["artifact_digest"]
    assert back["callables"]["backtest_func"].endswith("_make_backtest.<locals>.bt")


# Callable labels describe names; callable digests identify state.

def test_callable_labels_are_module_qualname():
    sig, ret = _panels()
    art = BacktestArtifacts(signals=sig, asset_returns=ret)
    prov = _prov(art, signal_func=np.nansum, backtest_func=_make_backtest(0.0))
    assert prov["callables"] == {
        "signal_func": "numpy.nansum",
        "backtest_func": f"{__name__}._make_backtest.<locals>.bt"}


def test_callable_labels_none_when_absent():
    sig, ret = _panels()
    prov = _prov(BacktestArtifacts(signals=sig, asset_returns=ret))
    assert prov["callables"] == {"signal_func": None, "backtest_func": None}


def test_callable_label_is_not_identity_for_closures():
    # Labels are locators, not identity (provenance keeps `callables`
    # separate from `callable_digest`): two closures from
    # the same factory with different captured parameters carry the same
    # label. Pinned so nobody "fixes" it by hashing __code__ - both share
    # the code object too, and a code hash would present false identity.
    a = api._callable_label(_make_backtest(10.0))
    b = api._callable_label(_make_backtest(0.0))
    assert a == b == f"{__name__}._make_backtest.<locals>.bt"
    assert _make_backtest(10.0).__code__ is _make_backtest(0.0).__code__


def test_callable_digest_covers_closure_and_partial_state():
    a = api._callable_digest(_make_backtest(10.0))
    b = api._callable_digest(_make_backtest(0.0))
    assert HEX64.match(a) and HEX64.match(b) and a != b
    assert api._callable_digest(_make_backtest(10.0)) == a

    # Isolate partial argument identity from dependency implementation
    # details: NumPy 1.24's array-function wrapper closes over itself,
    # intentionally making its captured-state fingerprint unavailable.
    def reducer(values, axis=0):
        return values.sum(axis=axis)

    p0 = functools.partial(reducer, axis=0)
    p1 = functools.partial(reducer, axis=1)
    d0, d1 = api._callable_digest(p0), api._callable_digest(p1)
    assert HEX64.match(d0) and HEX64.match(d1) and d0 != d1


def test_callable_digest_rejects_recursive_capture_instead_of_claiming_identity():
    def recursive(values):
        return recursive(values)

    assert api._callable_digest(recursive) is None


def test_callable_digest_does_not_embed_checkout_filename():
    def identity(x):
        return x

    left_code = identity.__code__.replace(co_filename="/tmp/left/source.py")
    right_code = identity.__code__.replace(co_filename="/opt/right/source.py")
    left = types.FunctionType(left_code, identity.__globals__,
                              name=identity.__name__)
    right = types.FunctionType(right_code, identity.__globals__,
                               name=identity.__name__)
    left.__qualname__ = right.__qualname__ = identity.__qualname__
    assert api._callable_digest(left) == api._callable_digest(right)


def test_callable_label_unwraps_partial_and_survives_odd_callables():
    assert api._callable_label(functools.partial(np.nansum, axis=0)) \
        == "functools.partial(numpy.nansum)"

    class Pipeline:
        def __call__(self, s, r):
            return (s * r).sum(axis=1)

    label = api._callable_label(Pipeline())
    assert label.endswith("Pipeline")           # instance: falls back to type
    assert api._callable_label(None) is None
    assert api._callable_label(lambda s: s).endswith("<lambda>")


# Digest cost and failure containment.

def test_digest_cost_is_negligible_on_1000x30():
    # The timing budget allows normal CI variation while guarding against quadratic or
    # per-cell Python hashing.
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=1000)
    df = pd.DataFrame(rng.standard_normal((1000, 30)), index=idx,
                      columns=[f"A{i:03d}" for i in range(30)])
    api._panel_digest(df)                       # warm
    t0 = time.perf_counter()
    for _ in range(10):
        api._panel_digest(df)
    per_call = (time.perf_counter() - t0) / 10
    assert per_call < 0.05, f"{per_call * 1e3:.1f} ms per 1000x30 digest"


def test_digest_failure_degrades_provenance_not_the_audit(monkeypatch):
    def boom(obj):
        raise RuntimeError("hash unavailable")
    monkeypatch.setattr(api, "_panel_digest", boom)
    sig, ret = _panels()
    rep = audit(BacktestArtifacts(signals=sig, asset_returns=ret),
                include=["contamination"])
    prov = rep.meta["provenance"]
    assert "config_hash" not in prov
    assert "provenance construction failed" in prov["error"]
    assert rep.results                          # verdict unharmed
