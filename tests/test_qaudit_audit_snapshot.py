"""One audit must describe the data and callable state present at entry."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qaudit import AuditConfig, BacktestArtifacts, audit
from qaudit.api import _callable_digest, _panel_digest
from qaudit.types import Status


def _artifacts() -> BacktestArtifacts:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2023-01-01", periods=160)
    columns = list("ABCDEFGH")
    raw = pd.DataFrame(rng.normal(size=(160, 8)), index=index, columns=columns)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(160, 8)),
                           index=index, columns=columns)
    return BacktestArtifacts(signals=raw.copy(), asset_returns=returns,
                             signal_input=raw)


def test_callback_mutation_of_original_raw_input_does_not_change_later_probes():
    artifacts = _artifacts()
    original_raw = artifacts.signal_input
    entry_digest = _panel_digest(original_raw)

    def signal_func(data):
        # A pipeline may update a notebook/cache object it closes over.
        # That caller object must not also be the auditor's raw input.
        original_raw.iloc[:, :] += 1.0
        return data.copy()

    report = audit(artifacts, AuditConfig(truncation_sample_dates=3),
                   signal_func=signal_func,
                   include=["dynamic.signal_reproducibility",
                            "dynamic.rolling_window_integrity"])

    assert report["dynamic.signal_reproducibility"].status is Status.PASS
    assert report["dynamic.rolling_window_integrity"].status is Status.PASS
    assert report.meta["provenance"]["artifact_digest"]["signal_input"] == entry_digest
    assert _panel_digest(original_raw) != entry_digest


def test_callable_fingerprint_records_entry_state_before_probe_calls():
    class Backtest:
        def __init__(self):
            self.calls = 0

        def __call__(self, signals, returns):
            self.calls += 1
            return returns.mean(axis=1)

    backtest = Backtest()
    entry_digest = _callable_digest(backtest)
    report = audit(_artifacts(), AuditConfig(date_shift_max=1),
                   backtest_func=backtest, include=["dynamic.date_shift"])

    assert backtest.calls > 0
    assert entry_digest is not None
    assert _callable_digest(backtest) != entry_digest
    assert report.meta["provenance"]["callable_digest"]["backtest_func"] == entry_digest


def test_callback_mutation_of_original_panel_shape_does_not_crash_coverage():
    artifacts = _artifacts()
    original_signals = artifacts.signals

    def backtest(signals, returns):
        original_signals.drop(index=original_signals.index, inplace=True)
        return returns.mean(axis=1)

    report = audit(artifacts, AuditConfig(date_shift_max=1),
                   backtest_func=backtest, include=["dynamic.date_shift"])

    assert original_signals.empty
    assert report.meta["n_periods"] == 160
    assert not report.find("audit.coverage")
    assert not report.errors
