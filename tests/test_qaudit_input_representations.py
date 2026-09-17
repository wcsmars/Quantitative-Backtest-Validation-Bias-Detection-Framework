"""Accepted input representations and stable provenance across storage layouts.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from qaudit import BacktestArtifacts, MisalignedInputError, audit
from qaudit.api import _callable_digest, _panel_digest


def _panels(columns=None):
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2024-01-01", periods=120)
    if columns is None:
        columns = list("ABCDEFGH")
    signals = pd.DataFrame(rng.normal(size=(120, 8)),
                           index=index, columns=columns)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(120, 8)),
                           index=index, columns=columns)
    return signals, returns


@pytest.mark.parametrize("gap", ["cell", "date", "column"])
def test_nullable_boolean_positions_audit_as_float_weights(gap):
    signals, returns = _panels()
    positions = (signals.shift(1) > 0).astype("boolean")
    if gap == "cell":
        positions.iloc[0, 0] = pd.NA
    elif gap == "date":
        positions = positions.drop(index=signals.index[10])
    else:
        positions = positions.drop(columns=signals.columns[0])
    original = positions.copy(deep=True)
    artifacts = BacktestArtifacts(signals, returns, positions=positions)
    artifacts.validate()

    aligned = artifacts.aligned()
    expected = positions.astype("Float64").reindex(
        index=signals.index, columns=signals.columns).fillna(0).astype(float)
    pd.testing.assert_frame_equal(aligned.positions, expected)
    report = audit(artifacts, include=["costs"])
    baseline = audit(BacktestArtifacts(signals, returns, positions=expected),
                     include=["costs"])
    assert not report.errors
    assert [(r.check, r.status, r.message) for r in report.results] == [
        (r.check, r.status, r.message) for r in baseline.results]
    pd.testing.assert_frame_equal(positions, original)


@pytest.mark.parametrize("label_kind", ["tuple", "frozenset"])
def test_universe_accepts_flat_iterable_asset_ids(label_kind):
    labels = [("US", i) if label_kind == "tuple" else frozenset({i})
              for i in range(8)]
    columns = pd.Index(labels, tupleize_cols=False)
    assert not isinstance(columns, pd.MultiIndex)
    signals, returns = _panels(columns)
    universe = pd.DataFrame(True, index=signals.index, columns=columns)
    artifacts = BacktestArtifacts(signals, returns, universe=universe)
    artifacts.validate()
    report = audit(artifacts)
    assert not report.errors
    pd.testing.assert_frame_equal(artifacts.aligned().universe, universe)

    # The vectorized lookup must reject missing membership evidence
    # for these labels, using only dates the audit actually retains.
    broken = universe.astype(float)
    broken.iloc[:, 0] = np.nan
    warmup = pd.DataFrame(1.0, index=[signals.index[0] - pd.Timedelta(days=1)],
                          columns=columns)
    broken = pd.concat([warmup, broken])
    with pytest.raises(MisalignedInputError, match="all-NaN column"):
        BacktestArtifacts(signals, returns, universe=broken).validate()


def _signal_with_capture(panel):
    def signal(data):
        return data + panel.iloc[0, 0]
    return signal


def test_copy_on_write_layout_keeps_panel_and_callable_provenance_stable():
    # Copy-on-Write is mandatory in pandas 3; setting the obsolete option
    # there warns, while pandas 2 needs it enabled explicitly.
    cow = (pd.option_context("mode.copy_on_write", True)
           if int(pd.__version__.split(".", 1)[0]) < 3 else nullcontext())
    with cow:
        panel = pd.DataFrame(np.array([[1.0, np.nan], [2.0, 3.0]]),
                             copy=False)
        copied = panel.copy(deep=True)
        pd.testing.assert_frame_equal(panel, copied)
        assert _panel_digest(panel) == _panel_digest(copied)
        first = _signal_with_capture(panel)
        second = _signal_with_capture(copied)
        assert _callable_digest(first) is not None
        assert _callable_digest(first) == _callable_digest(second)

        signals, returns = _panels()
        artifacts = BacktestArtifacts(signals, returns)
        original_report = audit(artifacts, signal_func=first,
                                include=["contamination"])
        copied_report = audit(artifacts, signal_func=second,
                              include=["contamination"])
        original_prov = original_report.meta["provenance"]
        copied_prov = copied_report.meta["provenance"]
        assert "error" not in original_prov
        assert "error" not in copied_prov
        assert original_prov["callable_digest"] == copied_prov["callable_digest"]


@pytest.mark.parametrize("kind", ["frame", "series"])
@pytest.mark.parametrize("readonly", [False, True])
def test_digest_canonicalizes_nan_without_mutating_shared_input(kind, readonly):
    values = np.array([[1.0, np.nan], [2.0, 3.0]])
    # A valid quiet NaN with a non-default payload exposes even an otherwise
    # invisible in-place rewrite of NaN bits in the caller's array.
    values.view(np.uint64)[0, 1] = 0x7FF8000000000042
    before = values.tobytes()
    values.setflags(write=not readonly)
    if kind == "frame":
        panel = pd.DataFrame(values, copy=False)
        canonical = pd.DataFrame([[1.0, np.nan], [2.0, 3.0]])
    else:
        panel = pd.Series(values.reshape(-1), copy=False)
        canonical = pd.Series([1.0, np.nan, 2.0, 3.0])
    assert _panel_digest(panel) == _panel_digest(canonical)
    assert values.tobytes() == before
