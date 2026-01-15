"""
Equilibrium Premium/Discount Zone validation for Strategy II.

Replaces FVG/Demand Zone validation with dynamic Fibonacci-based equilibrium zones.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

from indicators.smc_custom import smc_custom
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
            self._max_level = swept_level  # Fixed (swept high)
            self._min_level = float('inf')  # Will track running min from inflection points
        else:  # long
            self._min_level = swept_level  # Fixed (swept low)
            self._max_level = float('-inf')  # Will track running max from inflection points

        # Track confirmed inflection points for running extreme
        self._confirmed_extreme_level = None

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

        Scans candle-by-candle, using INFLECTION POINTS for running extreme.
        Only calculates equilibrium after a confirmed inflection point forms.

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

            # Calculate inflection points progressively (up to current candle)
            df_slice = self.df_5m.loc[:time_5m]
            inflexions = smc_custom.inflexion_points(df_slice)

            # Update running extreme from inflection points AFTER Event B
            self._update_running_extreme_from_inflexions(inflexions, start_time)

            # Only calculate equilibrium if we have a confirmed inflection point
            if self._confirmed_extreme_level is None:
                continue

            # Calculate equilibrium using confirmed inflection point
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

    def _update_running_extreme_from_inflexions(
        self,
        inflexions: pd.DataFrame,
        event_b_time: datetime
    ) -> None:
        """
        Find the lowest/highest inflection point AFTER Event B.

        For SHORT: look for convex (valley) inflection points (type = -1)
        For LONG: look for concave (peak) inflection points (type = 1)

        Args:
            inflexions: DataFrame of inflection points
            event_b_time: Timestamp of Event B (start of search window)
        """
        target_type = -1 if self.entry_direction == 'short' else 1

        # Scan ALL inflection points after Event B to find the extreme
        for i in range(len(inflexions)):
            inflexion_type = inflexions['InflexionType'].iloc[i]
            inflexion_time = self.df_5m.index[i]

            # Must be AFTER Event B and correct type
            if inflexion_time <= event_b_time:
                continue

            if np.isnan(inflexion_type) or inflexion_type != target_type:
                continue

            level = inflexions['Level'].iloc[i]

            # Update if this is a better extreme (lowest for SHORT, highest for LONG)
            if self.entry_direction == 'short':
                if self._confirmed_extreme_level is None or level < self._confirmed_extreme_level:
                    self._confirmed_extreme_level = level
                    self._min_level = level
            else:  # long
                if self._confirmed_extreme_level is None or level > self._confirmed_extreme_level:
                    self._confirmed_extreme_level = level
                    self._max_level = level

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
