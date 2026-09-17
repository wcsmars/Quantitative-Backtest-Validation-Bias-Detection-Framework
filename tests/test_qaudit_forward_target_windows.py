"""Only actual future windows may affect forward-return diagnostics."""
import numpy as np
import pandas as pd
import pytest

from qaudit._stats import forward_returns


def test_unused_leading_window_cannot_raise_forward_return_overflow():
    returns = pd.DataFrame({"A": [1e99, 1e99, 1e99, 1e99, 0.0]})
    with np.errstate(all="raise"):
        result = forward_returns(returns, 4)
    # The return at row 0 never belongs to any future target. Only rows
    # 1..4 contribute to target 0, whose three large gains are representable.
    expected = (1.0 + returns.iloc[1:, 0]).prod() - 1.0
    assert result.iloc[0, 0] == pytest.approx(expected)
    assert result.iloc[1:, 0].isna().all()


def test_actual_future_window_overflow_still_fails_closed():
    returns = pd.DataFrame({"A": [0.0, 1e99, 1e99, 1e99, 1e99]})
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="compounding exceeds float64"):
            forward_returns(returns, 4)
