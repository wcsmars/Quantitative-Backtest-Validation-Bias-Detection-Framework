"""Configuration validation, timezone periods, report serialization, and errors.

Module failures survive filtering. Configuration mutation is validated,
reports remain JSON-safe, and missing artifacts raise the documented error.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from qaudit.api import audit
from qaudit.config import AuditConfig
from qaudit.errors import InputValidationError, MissingArtifactError
from qaudit.inputs import BacktestArtifacts
from qaudit.report import AuditReport
from qaudit.synthetic import momentum_signal, simulate_market
from qaudit.types import CheckResult, Severity, Status


@pytest.fixture(scope="module")
def market():
    return simulate_market(seed=11)


@pytest.fixture(scope="module")
def base_arts(market):
    rets = market["returns"]
    return dict(signals=momentum_signal(rets), asset_returns=rets,
                signal_lag=1)


def _boom(*args, **kwargs):
    raise RuntimeError("synthetic module crash")


# Include/exclude filtering must retain module errors.

class TestErrorSurvivesFiltering:
    def test_dynamic_module_crash_survives_family_include(
            self, base_arts, monkeypatch):
        monkeypatch.setattr("qaudit.dynamic.probes_null.run", _boom)
        rep = audit(BacktestArtifacts(**base_arts), include=["dynamic"])
        assert rep.errors, "module crash vanished under include filtering"
        assert not rep.ok
        (err,) = rep.errors
        assert err.check == "dynamic.<module:probes_null>"
        assert err.check.startswith("dynamic.")

    def test_dynamic_module_crash_survives_specific_check_include(
            self, base_arts, monkeypatch):
        monkeypatch.setattr("qaudit.dynamic.probes_shift.run", _boom)
        rep = audit(BacktestArtifacts(**base_arts),
                    include=["dynamic.date_shift"])
        assert rep.errors, ("crash of the module that would have produced the "
                            "requested check was silently dropped")
        assert not rep.ok

    def test_static_module_crash_survives_exclude(self, base_arts, monkeypatch):
        monkeypatch.setattr("qaudit.checks.costs.run", _boom)
        rep = audit(BacktestArtifacts(**base_arts), exclude=["lookahead"])
        assert any(r.check == "costs.<module>" for r in rep.errors)
        assert not rep.ok

    def test_static_module_crash_id_unchanged(self, base_arts, monkeypatch):
        monkeypatch.setattr("qaudit.checks.costs.run", _boom)
        rep = audit(BacktestArtifacts(**base_arts))
        assert any(r.check == "costs.<module>" for r in rep.errors)

    def test_strict_still_raises(self, base_arts, monkeypatch):
        from qaudit.errors import CheckRuntimeError
        monkeypatch.setattr("qaudit.dynamic.probes_null.run", _boom)
        with pytest.raises(CheckRuntimeError):
            audit(BacktestArtifacts(**base_arts), include=["dynamic"],
                  strict=True)


# Timezone-aware indexes must not crash split checks.

def _localized(df, tz):
    out = df.copy()
    out.index = out.index.tz_localize(tz)
    return out


class TestTimezonePeriods:
    def test_aware_index_naive_periods_runs_clean(self, market):
        rets = _localized(market["returns"], "America/New_York")
        sig = _localized(momentum_signal(market["returns"]), "America/New_York")
        art = BacktestArtifacts(
            signals=sig, asset_returns=rets, signal_lag=1,
            train_period=("2018-01-02", "2019-12-31"),
            test_period=("2020-01-02", "2021-06-30"))
        rep = audit(art, include=["contamination"])
        assert not rep.errors, [r.message for r in rep.errors]
        assert rep.find("contamination.")
        assert art.train_period[0].tz is not None

    def test_aware_index_aware_periods_converted(self, market):
        rets = _localized(market["returns"], "America/New_York")
        sig = _localized(momentum_signal(market["returns"]), "America/New_York")
        art = BacktestArtifacts(
            signals=sig, asset_returns=rets, signal_lag=1,
            train_period=(pd.Timestamp("2018-01-02", tz="UTC"),
                          pd.Timestamp("2019-12-31", tz="UTC")))
        art.validate()
        assert str(art.train_period[0].tz) == "America/New_York"

    def test_naive_index_aware_periods_rejected(self, base_arts):
        art = BacktestArtifacts(
            **base_arts,
            train_period=(pd.Timestamp("2018-01-02", tz="UTC"),
                          pd.Timestamp("2019-12-31", tz="UTC")))
        with pytest.raises(InputValidationError, match="timezone-aware"):
            art.validate()

    def test_validate_is_idempotent_on_aware_index(self, market):
        rets = _localized(market["returns"], "UTC")
        sig = _localized(momentum_signal(market["returns"]), "UTC")
        art = BacktestArtifacts(signals=sig, asset_returns=rets,
                                train_period=("2018-01-02", "2019-12-31"))
        art.validate()
        art.validate()
        assert str(art.train_period[0].tz) == "UTC"


# ---------------------------------------------------------------------------
# AuditConfig: seed validation + no mutation past validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    @pytest.mark.parametrize("bad", ["not-a-seed", -1, True, 1.5, None])
    def test_bad_seed_rejected(self, bad):
        with pytest.raises(InputValidationError, match="seed"):
            AuditConfig(seed=bad)

    def test_numpy_int_seed_accepted(self):
        assert AuditConfig(seed=np.int64(7)).seed == 7

    def test_invalid_mutation_raises_and_reverts(self):
        cfg = AuditConfig()
        with pytest.raises(InputValidationError):
            cfg.sharpe_warn = -5.0
        assert cfg.sharpe_warn == 3.0  # left in last valid state

    def test_mutation_cannot_break_warn_fail_order(self):
        cfg = AuditConfig()
        with pytest.raises(InputValidationError):
            cfg.sharpe_warn = cfg.sharpe_fail + 1.0
        assert cfg.sharpe_warn == 3.0

    def test_valid_mutation_still_allowed(self):
        cfg = AuditConfig()
        cfg.sharpe_warn = 4.0
        assert cfg.sharpe_warn == 4.0


# ---------------------------------------------------------------------------
# to_dict must be json-serializable, strictly (allow_nan=False)
# ---------------------------------------------------------------------------

class TestReportSerialization:
    def test_nasty_details_serialize(self):
        ts = pd.Timestamp("2020-03-15")
        nasty = {
            "series_ts_keys": pd.Series([1.0, np.nan],
                                        index=[ts, ts + np.timedelta64(1, "D")]),
            "frame": pd.DataFrame({"a": [1.0]}, index=[ts]),
            "timestamp": ts,
            "np_bool": np.bool_(True),
            "np_float": np.float64(1.5),
            "nan": float("nan"),
            "inf": np.inf,
            "array": np.array([1.0, np.nan]),
            "date": dt.date(2020, 1, 1),
            # Construct from an explicitly-unitized NumPy scalar.  This
            # preserves the pandas Timedelta serialization branch without
            # relying on NumPy's deprecated generic timedelta unit.
            "timedelta": pd.Timedelta(np.timedelta64(3, "D")),
            "tuple_key": {("a", 1): 2},
            "set": {1, 2},
            "object": object(),
        }
        rep = AuditReport(results=[CheckResult(
            "x.y", Status.PASS, "m", severity=Severity.INFO, details=nasty)])
        payload = json.dumps(rep.to_dict(), allow_nan=False)
        assert "2020-03-15" in payload

    def test_full_audit_report_serializes(self, base_arts):
        rep = audit(BacktestArtifacts(**base_arts))
        json.dumps(rep.to_dict(), allow_nan=False)


# ---------------------------------------------------------------------------
# MissingArtifactError is raised for absent required panels
# ---------------------------------------------------------------------------

class TestMissingArtifact:
    def test_none_signals_raises_missing_artifact(self, market):
        art = BacktestArtifacts(signals=None,
                                asset_returns=market["returns"])
        with pytest.raises(MissingArtifactError, match="signals"):
            art.validate()

    def test_none_asset_returns_raises_missing_artifact(self, market):
        art = BacktestArtifacts(signals=momentum_signal(market["returns"]),
                                asset_returns=None)
        with pytest.raises(MissingArtifactError, match="asset_returns"):
            art.validate()
