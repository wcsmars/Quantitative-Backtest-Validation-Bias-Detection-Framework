"""Finite-magnitude validation and fail-closed position-mass coverage ratios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qaudit.errors import InputValidationError, MisalignedInputError
from qaudit.inputs import BacktestArtifacts
from qaudit.synthetic import momentum_signal, simulate_market


@pytest.fixture(scope="module")
def market():
    return simulate_market(seed=21)


def _arts(market, **overrides):
    rets = market["returns"]
    base = dict(signals=momentum_signal(rets), asset_returns=rets,
                signal_lag=1)
    base.update(overrides)
    return BacktestArtifacts(**base)


class TestHugeMagnitudeRejected:
    def test_huge_frame_value_rejected(self, market):
        rets = market["returns"].copy()
        rets.iloc[50, 0] = 1e154
        art = _arts(market, asset_returns=rets)
        with pytest.raises(InputValidationError, match="magnitude"):
            art.validate()

    def test_huge_negative_value_rejected(self, market):
        sig = momentum_signal(market["returns"]).copy()
        sig.iloc[40, 1] = -1e300
        art = _arts(market, signals=sig)
        with pytest.raises(InputValidationError, match="magnitude"):
            art.validate()

    def test_huge_series_value_rejected(self, market):
        sr = pd.Series(0.001, index=market["returns"].index)
        sr.iloc[10] = 1e200
        art = _arts(market, strategy_returns=sr)
        with pytest.raises(InputValidationError, match="magnitude"):
            art.validate()

    def test_large_but_sane_values_pass(self, market):
        # 1e6 is absurd for a return but far below the magnitude bound - the
        # gate is about float64 arithmetic safety, not plausibility (other
        # checks judge plausibility)
        rets = market["returns"].copy()
        rets.iloc[50, 0] = 1e6
        art = _arts(market, asset_returns=rets)
        art.validate()


class TestMassGateFailClosed:
    def test_zero_kept_mass_rejects(self, market):
        # A zero kept mass with positive total mass must fail coverage. Huge finite
        # weights are rejected independently by the magnitude gate.
        pos = market["returns"].copy() * 0 + 0.1
        pos.columns = [f"X{i}" for i in range(len(pos.columns))]
        art = _arts(market, positions=pos)
        with pytest.raises((MisalignedInputError, InputValidationError)):
            art.validate()
