"""
Equilibrium Premium/Discount Zone validation for Strategy II.

Replaces FVG/Demand Zone validation with dynamic Fibonacci-based equilibrium zones.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple

from indicators.smc_custom import smc_custom
from .models import EquilibriumState, TrackedFVG


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
        df_mid: pd.DataFrame,
        swept_level: float,
        entry_direction: str,
        fvg_mid: Optional[pd.DataFrame] = None,
        sweep_time: Optional[datetime] = None,
        inflexions_mid: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize equilibrium validator.

        Args:
            df_mid: Mid TF OHLCV DataFrame with datetime index
            swept_level: Price of the swept liquidity (high TF)
            entry_direction: 'long' or 'short'
            fvg_mid: FVG DataFrame for mid TF (optional, for FVG respect validation)
            sweep_time: Liquidity sweep timestamp (used for FVG filtering)
            inflexions_mid: Pre-calculated inflection points (optional, for O(n) optimization)
        """
        self.df_mid = df_mid
        self.swept_level = swept_level
        self.entry_direction = entry_direction
        self.fvg_mid = fvg_mid
        self.sweep_time = sweep_time
        self.inflexions_mid = inflexions_mid

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

        # FVG respect validation tracking
        self._tracked_fvgs: List[TrackedFVG] = []
        self._validation_type: str = 'Equilibrium'

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
        mask = (self.df_mid.index >= start_time) & (self.df_mid.index <= end_time)
        window_candles = self.df_mid[mask]

        if len(window_candles) == 0:
            return None

        # Progressive scan
        for time_5m in window_candles.index:
            candle = self.df_mid.loc[time_5m]

            # Calculate inflection points progressively (up to current candle)
            df_slice = self.df_mid.loc[:time_5m]
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
            inflexion_time = self.df_mid.index[i]

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

    def _update_running_extreme_up_to(self, time_5m: datetime) -> None:
        """
        Update running extreme using pre-calculated inflection points up to time_5m.
        Only considers inflection points after sweep_time.

        This is the O(n) optimized version that uses pre-calculated inflexions_mid
        instead of recalculating inflection points for each candle.

        Args:
            time_5m: Current candle timestamp to scan up to
        """
        if self.inflexions_mid is None or self.sweep_time is None:
            return

        current_idx = self.df_mid.index.get_loc(time_5m)
        sweep_idx = self.df_mid.index.get_loc(self.sweep_time)
        target_type = -1 if self.entry_direction == 'short' else 1

        # Look at inflection points from sweep to current candle
        for i in range(sweep_idx + 1, current_idx + 1):
            inflexion_type = self.inflexions_mid['InflexionType'].iloc[i]

            if pd.isna(inflexion_type) or inflexion_type != target_type:
                continue

            level = self.inflexions_mid['Level'].iloc[i]

            if self.entry_direction == 'short':
                if self._confirmed_extreme_level is None or level < self._confirmed_extreme_level:
                    self._confirmed_extreme_level = level
                    self._min_level = level
            else:
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
            trigger_price=self._trigger_price,
            validation_type=self._validation_type,
            tracked_fvgs=self._tracked_fvgs
        )

    def get_equilibrium_level(self) -> Optional[float]:
        """Get current equilibrium level."""
        return self._equilibrium

    def get_zone_bounds(self) -> Tuple[float, float]:
        """Get current min and max levels defining the equilibrium zone."""
        return (self._min_level, self._max_level)

    def validate_at_candle(
        self,
        time_5m: datetime
    ) -> Optional[Tuple[datetime, str, float, EquilibriumState]]:
        """
        Check single 5M candle for FVG respect OR equilibrium zone entry.

        This is the O(n) optimized version that processes one candle at a time,
        using pre-calculated inflection points instead of recalculating them.

        Args:
            time_5m: The specific candle timestamp to check

        Returns:
            (trigger_time, validation_type, trigger_price, state) or None
        """
        candle = self.df_mid.loc[time_5m]
        current_idx = self.df_mid.index.get_loc(time_5m)

        # 1. Check FVG respect FIRST (takes priority)
        if self.fvg_mid is not None:
            new_fvgs = self.detect_fvg_respect_at_candle(time_5m, current_idx)
            if new_fvgs:  # If any FVGs found
                # Add new FVGs to tracked list
                self._tracked_fvgs.extend(new_fvgs)
                self._validation_type = 'FVG_Respect'
                self._trigger_time = time_5m
                self._trigger_price = candle['high'] if self.entry_direction == 'short' else candle['low']
                self._in_target_zone = True
                return (time_5m, 'FVG_Respect', self._trigger_price, self.get_current_state())

        # 2. Update running extreme from pre-calculated inflection points up to this candle
        self._update_running_extreme_up_to(time_5m)

        # 3. Check equilibrium zone if we have a confirmed inflection
        if self._confirmed_extreme_level is None:
            return None

        self._equilibrium = self._calculate_equilibrium()

        if self._is_in_target_zone(candle):
            self._in_target_zone = True
            self._trigger_time = time_5m
            self._validation_type = 'Equilibrium'
            self._trigger_price = candle['high'] if self.entry_direction == 'short' else candle['low']
            return (time_5m, 'Equilibrium', self._trigger_price, self.get_current_state())

        return None

    def detect_fvg_respect_at_candle(
        self,
        time_5m: datetime,
        current_idx: int
    ) -> List[TrackedFVG]:
        """
        Simplified FVG respect detection - returns ALL matching FVGs.

        Logic:
        1. Filter fvg_mid for FVGs after the liquidity sweep point
        2. Find FVGs where MitigatedIndex == current_idx (being touched NOW)
        3. Exclude if same-candle disrespect (StatusIndex == MitigatedIndex AND Respected == False)
        4. Return ALL valid FVGs as candidates

        Args:
            time_5m: Current 5M candle timestamp
            current_idx: Current candle index in df_mid

        Returns:
            List of TrackedFVG objects (empty list if none found)
        """
        candidates = []

        if self.fvg_mid is None or self.sweep_time is None:
            return candidates

        # Get sweep index
        try:
            sweep_idx = self.df_mid.index.get_loc(self.sweep_time)
            print(f"      [DEBUG] FVG detection at idx {current_idx}: sweep_idx={sweep_idx}")
        except KeyError:
            print(f"      [DEBUG] Could not find sweep_time {self.sweep_time} in df_mid index")
            return candidates

        # Determine target FVG type based on entry direction
        # For LONG: look for bullish FVG (1) being respected (support zone)
        # For SHORT: look for bearish FVG (-1) being respected (resistance zone)
        target_fvg_type = 1 if self.entry_direction == 'long' else -1

        # Vectorized filtering - much faster than Python loop
        # Create position array for filtering (fvg_mid may not have sequential index)
        positions = pd.Series(range(len(self.fvg_mid)), index=self.fvg_mid.index)

        # Build mask for valid FVGs
        mask = (
            (self.fvg_mid['MitigatedIndex'] == current_idx) &  # Being touched NOW
            (self.fvg_mid['FVG'] == target_fvg_type) &          # Correct type
            (positions > sweep_idx)                              # Formed after sweep
        )

        # Apply mask to get potential matches
        potential_matches = self.fvg_mid[mask]

        # Process matches and check for same-candle disrespect
        for idx in potential_matches.index:
            fvg_row = potential_matches.loc[idx]
            fvg_pos = positions.loc[idx]  # Get positional index for TrackedFVG

            # IMPORTANT: Skip if touched AND disrespected on SAME candle
            status_idx = fvg_row['StatusIndex']
            mitigated_idx = fvg_row['MitigatedIndex']
            respected = fvg_row['Respected']
            if not pd.isna(status_idx) and int(status_idx) == int(mitigated_idx) and respected == False:
                print(f"      [DEBUG] FVG at idx {fvg_pos} skipped: same-candle disrespect")
                continue

            # Valid candidate - add to list
            print(f"      [DEBUG] FVG candidate found: fvg_idx={fvg_pos}, top={fvg_row['Top']:.2f}, bottom={fvg_row['Bottom']:.2f}")
            candidates.append(TrackedFVG(
                fvg_index=fvg_pos,
                fvg_type=int(fvg_row['FVG']),
                top=fvg_row['Top'],
                bottom=fvg_row['Bottom'],
                respected_at_index=current_idx,
                respected_at_time=time_5m
            ))

        return candidates

    def get_invalidated_fvgs_at_candle(
        self,
        tracked_fvgs: List[TrackedFVG],
        current_5m_idx: int
    ) -> List[TrackedFVG]:
        """
        Check which tracked FVGs were invalidated at current candle.
        Returns list of FVGs that should be removed from tracking.

        Args:
            tracked_fvgs: List of FVGs being tracked
            current_5m_idx: Current 5M candle index

        Returns:
            List of FVGs that were invalidated at this candle
        """
        invalidated = []

        if self.fvg_mid is None:
            return invalidated

        for fvg in tracked_fvgs:
            fvg_row = self.fvg_mid.iloc[fvg.fvg_index]
            status_idx = fvg_row['StatusIndex']

            if not pd.isna(status_idx) and int(status_idx) == current_5m_idx and fvg_row['Respected'] == False:
                print(f"      [DEBUG] FVG at idx {fvg.fvg_index} invalidated at 5M idx {current_5m_idx}")
                invalidated.append(fvg)

        return invalidated

    def is_fvg_disrespected_at_candle(
        self,
        tracked_fvg: TrackedFVG,
        current_5m_idx: int
    ) -> bool:
        """
        Check if tracked FVG was just disrespected at current 5M candle.
        DEPRECATED: Use get_invalidated_fvgs_at_candle for list-based tracking.

        Args:
            tracked_fvg: The FVG being tracked
            current_5m_idx: Current 5M candle index

        Returns:
            True if FVG was disrespected at this candle
        """
        if self.fvg_mid is None:
            return False

        fvg_row = self.fvg_mid.iloc[tracked_fvg.fvg_index]

        # FVG is disrespected at this candle if:
        # StatusIndex == current_5m_idx AND Respected == False
        status_idx = fvg_row['StatusIndex']
        if pd.isna(status_idx):
            return False

        return int(status_idx) == current_5m_idx and fvg_row['Respected'] == False
