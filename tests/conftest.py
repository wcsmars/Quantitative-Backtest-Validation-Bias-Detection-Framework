"""Shared reference fixtures for checker tests."""

import pytest


@pytest.fixture
def perfect_rank_null_rates():
    """Approximate two-sided rank-null rates at an absolute IC of 0.90."""
    return {
        5: 0.017, 6: 0.018, 7: 0.007, 8: 0.005, 9: 0.002,
        10: 0.0011, 11: 0.0005, 12: 0.00025, 13: 0.0001,
    }
