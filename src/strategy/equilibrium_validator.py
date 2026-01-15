"""
Equilibrium Premium/Discount Zone validation for Strategy II.

Replaces FVG/Demand Zone validation with dynamic Fibonacci-based equilibrium zones.
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

from .models import EquilibriumState


class EquilibriumValidator:
    """
    Validates trade setup using Equilibrium Premium/Discount zones.

    For SHORT:
      - Max = swept high (fixed)
      - Min = running lowest after Event B (dynamic)
      - Equilibrium = (Max + Min) / 2
      - Trigger: Price enters PREMIUM area (above equilibrium)

    For LONG:
      - Min = swept low (fixed)
      - Max = running highest after Event B (dynamic)
      - Equilibrium = (Max + Min) / 2
      - Trigger: Price enters DISCOUNT area (below equilibrium)
    """

    def __init__(
        self,
        df_5m: pd.DataFrame,
        swept_level: float,
        entry_direction: str
    ) -> None:
        """
        Initialize equilibrium validator.

        Args:
            df_5m: 5M OHLCV DataFrame with datetime index
            swept_level: Price of the swept liquidity (1H)
            entry_direction: 'long' or 'short'
        """
        self.df_5m = df_5m
        self.swept_level = swept_level
        self.entry_direction = entry_direction

        # Initialize state
        if entry_direction == 'short':
            self._max_level = swept_level  # Fixed
            self._min_level = float('inf')  # Will track running min
        else:  # long
            self._min_level = swept_level  # Fixed
            self._max_level = float('-inf')  # Will track running max

        self._equilibrium = None
        self._in_target_zone = False
        self._trigger_time = None
        self._trigger_price = None

    def validate_equilibrium_zone(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[Tuple[datetime, str, float, EquilibriumState]]:
        """
        Progressive scan for price entering equilibrium zone.

        Scans candle-by-candle updating running min/max and checking
        if price enters the target zone.

        Args:
            start_time: Start of 5M window (after Event B)
            end_time: End of 5M window

        Returns:
            (trigger_time, 'Equilibrium', trigger_price, state) or None
        """
        # Get 5M candles in window
        mask = (self.df_5m.index >= start_time) & (self.df_5m.index <= end_time)
        window_candles = self.df_5m[mask]

        if len(window_candles) == 0:
            return None

        # Progressive scan
        for time_5m in window_candles.index:
            candle = self.df_5m.loc[time_5m]

            # Update running extreme
            self._update_running_extreme(candle['low'], candle['high'])

            # Calculate equilibrium
            self._equilibrium = self._calculate_equilibrium()

            # Check if price enters target zone
            if self._is_in_target_zone(candle):
                self._in_target_zone = True
                self._trigger_time = time_5m

                # Get trigger price based on direction
                if self.entry_direction == 'short':
                    # For SHORT, trigger when price enters premium (above equilibrium)
                    self._trigger_price = candle['high']
                else:
                    # For LONG, trigger when price enters discount (below equilibrium)
                    self._trigger_price = candle['low']

                state = self.get_current_state()
                return (time_5m, 'Equilibrium', self._trigger_price, state)

        return None

    def _update_running_extreme(
        self,
        candle_low: float,
        candle_high: float
    ) -> None:
        """Update running min/max based on new candle data."""
        if self.entry_direction == 'short':
            # For SHORT: track running minimum
            self._min_level = min(self._min_level, candle_low)
        else:  # long
            # For LONG: track running maximum
            self._max_level = max(self._max_level, candle_high)

    def _calculate_equilibrium(self) -> float:
        """Calculate current equilibrium level."""
        return (self._max_level + self._min_level) / 2

    def _is_in_target_zone(self, candle: pd.Series) -> bool:
        """Check if price is in premium (SHORT) or discount (LONG) zone."""
        if self._equilibrium is None:
            return False

        if self.entry_direction == 'short':
            # Premium area = above equilibrium
            return candle['high'] > self._equilibrium
        else:  # long
            # Discount area = below equilibrium
            return candle['low'] < self._equilibrium

    def get_current_state(self) -> EquilibriumState:
        """Return current equilibrium tracking state."""
        return EquilibriumState(
            fixed_level=self.swept_level,
            running_extreme=self._min_level if self.entry_direction == 'short' else self._max_level,
            equilibrium=self._equilibrium if self._equilibrium is not None else 0.0,
            entry_direction=self.entry_direction,
            in_target_zone=self._in_target_zone,
            trigger_time=self._trigger_time,
            trigger_price=self._trigger_price
        )

    def get_equilibrium_level(self) -> Optional[float]:
        """Get current equilibrium level."""
        return self._equilibrium

    def get_zone_bounds(self) -> Tuple[float, float]:
        """Get current min and max levels defining the equilibrium zone."""
        return (self._min_level, self._max_level)
