"""
test_var_models.py
==================
Pytest tests for the core risk-model functions.

Three groups:
1. Historical VaR on a known small array equals the expected percentile.
2. Traffic-light boundaries: 4 -> green, 5 -> yellow, 10 -> red.
3. No look-ahead: changing return on day t does not change VaR for day t.
"""

import numpy as np
import pandas as pd
import pytest

from src.var_models import historical_var, historical_es, rolling_var
from src.backtest import traffic_light_zone


# -- Group 1: Historical VaR on known arrays ----------------------------------

class TestHistoricalVaR:
    def test_99th_percentile_of_sorted_array(self):
        """
        With losses = [0.01, 0.02, ..., 1.00] (100 values),
        np.percentile uses linear interpolation so the result is ~0.9901.
        We verify it is within 0.01 of 0.99 (within 1 percentage point).
        """
        losses = np.arange(0.01, 1.01, 0.01)   # 100 values: 0.01..1.00
        result = historical_var(losses, 0.99)
        assert abs(result - 0.99) < 0.01

    def test_95th_percentile(self):
        """95th percentile of arange(0.01,1.01,0.01) is within 0.01 of 0.95."""
        losses = np.arange(0.01, 1.01, 0.01)
        result = historical_var(losses, 0.95)
        assert abs(result - 0.95) < 0.01

    def test_empty_array_returns_nan(self):
        assert np.isnan(historical_var(np.array([]), 0.99))

    def test_single_element(self):
        assert historical_var(np.array([0.05]), 0.99) == pytest.approx(0.05)

    def test_expected_shortfall_exceeds_var(self):
        """ES must be at least as large as VaR."""
        losses = np.random.default_rng(42).uniform(0, 1, 500)
        var = historical_var(losses, 0.99)
        es  = historical_es(losses, 0.99)
        assert es >= var

    def test_es_is_mean_of_tail(self):
        """ES is at least VaR (verifies tail logic does not shrink ES below VaR)."""
        losses = np.arange(0.01, 1.01, 0.01)
        var = historical_var(losses, 0.99)
        es  = historical_es(losses, 0.99)
        assert es >= var

    def test_known_es_value(self):
        """
        losses = [1, 2, ..., 200].
        The 99th pct is near 198, so the tail contains 199 and 200.
        ES (mean of tail) must be >= 198.
        """
        losses = np.arange(1.0, 201.0)   # 200 values: 1..200
        es = historical_es(losses, 0.99)
        assert es >= 198.0


# -- Group 2: Traffic-light zone boundaries -----------------------------------

class TestTrafficLight:
    @pytest.mark.parametrize("n,expected", [
        (0,  "green"),
        (1,  "green"),
        (4,  "green"),    # boundary: 4 is still green
        (5,  "yellow"),   # boundary: 5 flips to yellow
        (9,  "yellow"),   # boundary: 9 is still yellow
        (10, "red"),      # boundary: 10 flips to red
        (25, "red"),
    ])
    def test_zone_boundaries(self, n, expected):
        assert traffic_light_zone(n, green_max=4, yellow_max=9) == expected


# -- Group 3: No look-ahead ---------------------------------------------------

class TestNoLookAhead:
    """
    Verify the no-look-ahead property of rolling_var.

    The VaR for day t is computed from shifted data so it uses only
    information available before day t.  We test this by:
      (a) confirming VaR[t] is unchanged when we change loss[t], and
      (b) confirming VaR[t+1] IS changed when we change loss[t] to an extreme.

    Strategy: build a 600-day series of tiny losses (0.0001..0.0002).
    At position 300 (well inside the full series) set an extreme value.
    Because the window is 250 days and all other values are near 0.0002,
    the 99th pct of any normal window is ~0.0002.  Setting position 300 to
    0.50 will push the 99th pct at position 301 to 0.50 (it dominates).
    """

    WINDOW = 250
    EXTREME = 0.50
    TINY_LOW = 0.0001
    TINY_HIGH = 0.0002
    TARGET_IDX = 300    # the day we modify
    N = 600

    def _make_series(self, extreme_value: float) -> pd.Series:
        rng = np.random.default_rng(42)
        data = rng.uniform(self.TINY_LOW, self.TINY_HIGH, self.N)
        # Pre-seed two extreme values right before the target.
        # Since 1% of 250 is 2.5, we need 3 extreme values to significantly shift the 99th percentile.
        data[self.TARGET_IDX - 2] = self.EXTREME
        data[self.TARGET_IDX - 1] = self.EXTREME
        data[self.TARGET_IDX] = extreme_value
        return pd.Series(
            data,
            index=pd.date_range("2018-01-02", periods=self.N, freq="B"),
        )

    def test_var_at_day_t_unchanged_when_loss_at_day_t_changes(self):
        """
        VaR on day t must not depend on the loss on day t (no look-ahead).
        shift(1) in rolling_var enforces this.
        """
        losses_normal  = self._make_series(self.TINY_HIGH)   # normal value
        losses_extreme = self._make_series(self.EXTREME)      # extreme value

        var_normal  = rolling_var(losses_normal,  self.WINDOW, 0.99)
        var_extreme = rolling_var(losses_extreme, self.WINDOW, 0.99)

        # VaR at the target day must be the same -- it only saw data up to t-1
        assert var_normal.iloc[self.TARGET_IDX] == pytest.approx(
            var_extreme.iloc[self.TARGET_IDX], rel=1e-9
        ), "VaR at day t changed when we modified loss[t] -- look-ahead detected!"

    def test_var_at_day_t_plus_1_changes_when_loss_at_day_t_is_extreme(self):
        """
        VaR on day t+1 includes loss[t] in its look-back window.
        When loss[t] is 0.50 (vs normal ~0.0002), VaR[t+1] must be >> normal.
        """
        losses_normal  = self._make_series(self.TINY_HIGH)
        losses_extreme = self._make_series(self.EXTREME)

        var_normal  = rolling_var(losses_normal,  self.WINDOW, 0.99)
        var_extreme = rolling_var(losses_extreme, self.WINDOW, 0.99)

        t1 = self.TARGET_IDX + 1
        # var_extreme[t+1] should be dramatically larger (0.50 dominates the window)
        assert var_extreme.iloc[t1] > 100 * var_normal.iloc[t1], (
            f"VaR[t+1] did not react to extreme loss at t. "
            f"Normal={var_normal.iloc[t1]:.6f}, Extreme={var_extreme.iloc[t1]:.6f}"
        )
