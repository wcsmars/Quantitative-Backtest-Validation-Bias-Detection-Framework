"""Public artifact isolation and stable audit filtering at API boundaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qaudit.api as api
from qaudit import BacktestArtifacts, MisalignedInputError, audit
from qaudit.types import Status, failed, passed


@pytest.mark.parametrize("outside_stamp", ["before", "after"])
def test_universe_membership_outside_audit_cannot_resolve_missing_live_column(
    outside_stamp: str,
):
    dates = pd.bdate_range("2024-01-02", periods=60)
    columns = [f"A{i:03d}" for i in range(150)]
    panel = pd.DataFrame(0.01, index=dates, columns=columns)
    outside = (dates[0] - pd.offsets.BDay() if outside_stamp == "before"
               else dates[-1] + pd.offsets.BDay())
    universe = pd.DataFrame(1.0, index=dates.union([outside]), columns=columns)
    universe.loc[dates, columns[0]] = np.nan
    artifacts = BacktestArtifacts(panel, panel, universe=universe)

    # One empty column among 150 names is below the interior-NaN fraction
    # threshold; absence of live membership must not be diluted by breadth.
    with pytest.raises(MisalignedInputError, match="all-NaN.*audited calendar"):
        artifacts.validate()

    # Explicit non-membership remains valid even if the name was a member
    # before the audited sample (or becomes one after it).
    universe.loc[dates, columns[0]] = 0.0
    artifacts.validate()
    assert not artifacts.aligned().universe[columns[0]].any()


@pytest.mark.parametrize("mutated_filter", ["include", "exclude"])
def test_audit_filter_policy_is_stable_when_user_code_mutates_caller_lists(
    monkeypatch, mutated_filter: str,
):
    dates = pd.bdate_range("2024-01-02", periods=60)
    panel = pd.DataFrame(0.01, index=dates, columns=list("ABCDEF"))
    include = ["costs"]
    exclude: list[str] = []
    failing_check = "costs.missing_transaction_costs"

    def module_with_side_effect(*args, **kwargs):
        if mutated_filter == "include":
            include[:] = ["costs.no_cost_declaration"]
        else:
            exclude.append(failing_check)
        return [failed(check, "costs were omitted") if check == failing_check
                else passed(check, "ok")
                for check in api.MODULE_CHECK_IDS["qaudit.checks.costs"]]

    monkeypatch.setattr("qaudit.checks.costs.run", module_with_side_effect)
    report = audit(BacktestArtifacts(panel, panel),
                   include=include, exclude=exclude)

    assert report[failing_check].status is Status.FAIL
    assert not report.ok
    assert report.meta["provenance"]["include"] == ["costs"]
    assert report.meta["provenance"]["exclude"] == []
