"""Return contract shared by the dynamic backtest probes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import InputValidationError
from ..inputs import _MIN_OVERLAP, _check_series


class PipelineOutputError(ValueError):
    """A callback ran, but its result cannot represent a per-period backtest."""


def validated_returns(output: object, calendar: pd.DatetimeIndex) -> pd.Series:
    """Validate callback returns on the calendar supplied to that callback.

    Leading/trailing missing returns are allowed for warmup or a shorter
    trading span. Every calendar period between the first and last finite
    return must be present: dropping dates or blanking losses would change
    the sample on which the perturbation probes compare performance.
    """
    try:
        _check_series("strategy_returns", output)  # type: ignore[arg-type]
    except InputValidationError as exc:
        raise PipelineOutputError(
            str(exc).replace("artifacts.strategy_returns", "backtest_func output")
        ) from None
    assert isinstance(output, pd.Series)
    if output.index.hasnans:
        raise PipelineOutputError("backtest_func output index contains NaT dates")
    outside = ~output.index.isin(calendar)
    if bool(outside.any()):
        raise PipelineOutputError(
            f"backtest_func output has {int(outside.sum())} date(s) outside "
            "the supplied backtest calendar; preserve the input dates and timezone"
        )
    result = output.astype(float)
    on_grid = result.reindex(calendar)
    finite = np.isfinite(on_grid.to_numpy(dtype=float))
    locations = np.flatnonzero(finite)
    # An all-missing run has no Sharpe and follows the probes' existing
    # undefined-output SKIP path. A nonempty short sample must not be
    # allowed to produce a trusted performance comparison.
    if 0 < locations.size < _MIN_OVERLAP:
        raise PipelineOutputError(
            f"backtest_func output has only {locations.size} finite return "
            f"observation(s) on the supplied calendar; need >= {_MIN_OVERLAP} "
            "to compare backtest performance. Extend the live return history "
            "or reduce the callback's warmup window"
        )
    if locations.size:
        first, last = int(locations[0]), int(locations[-1])
        missing = int((~finite[first:last + 1]).sum())
        if missing:
            raise PipelineOutputError(
                f"backtest_func output has {missing} missing or NaN calendar "
                "return(s) inside its observed span; return one finite net "
                "return per period, including cash periods, without dropping "
                "or blanking dates between the first and last finite return"
            )
    return result
