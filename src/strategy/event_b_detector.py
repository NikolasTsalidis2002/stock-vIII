"""
Event B detection (BOS/IFVG) on 5M timeframe.

Handles detection of Break of Structure and Inverse Fair Value Gaps
that signal potential trade entries.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple


class EventBDetector:
    """
    Detects Event B (BOS or IFVG in opposite direction) on 5M.

    Responsibilities:
    - BOS detection in entry direction
    - Inverse FVG detection (FVG that gets disrespected)
    - Combined search helper
    """

    def __init__(
        self,
        df_5m: pd.DataFrame,
        bos_5m: pd.DataFrame,
        fvg_5m: pd.DataFrame,
        inflexions_5m: pd.DataFrame
    ) -> None:
        """
        Initialize with 5M data and indicators.

        Args:
            df_5m: 5M OHLCV DataFrame with datetime index
            bos_5m: Pre-calculated BOS data
            fvg_5m: Pre-calculated FVG data
            inflexions_5m: Pre-calculated inflexion points
        """
        self.df_5m = df_5m
        self.bos_5m = bos_5m
        self.fvg_5m = fvg_5m
        self.inflexions_5m = inflexions_5m

    def detect_bos_opposite_direction(
        self,
        start_time: datetime,
        end_time: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float]]:
        """
        Detect BOS on 5M in the entry direction (Event B).

        Args:
            start_time: Start of 5M analysis window
            end_time: End of 5M analysis window
            entry_direction: Intended entry direction ('long' or 'short')

        Returns:
            (timestamp, 'BOS', price) or None if no BOS found
        """
        # Get 5M data in window
        mask = (self.df_5m.index >= start_time) & (self.df_5m.index <= end_time)
        window_indices = self.df_5m.index[mask]

        if len(window_indices) == 0:
            return None

        # Look for BOS in entry direction
        target_bos = 1 if entry_direction == 'long' else -1

        for time_5m in window_indices:
            idx_5m = self.df_5m.index.get_loc(time_5m)
            bos_value = self.bos_5m['BOS'].iloc[idx_5m]

            if not np.isnan(bos_value) and bos_value == target_bos:
                # Found opposite direction BOS
                level = self.bos_5m['Level'].iloc[idx_5m]
                return (time_5m, 'BOS', level)

        return None

    def detect_ifvg(
        self,
        start_time: datetime,
        end_time: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float]]:
        """
        Detect Inverse FVG on 5M (FVG that gets disrespected).

        An IFVG is an FVG in the opposite direction that gets broken/disrespected,
        confirming the move in the entry direction.

        Args:
            start_time: Start of 5M analysis window
            end_time: End of 5M analysis window
            entry_direction: Intended entry direction ('long' or 'short')

        Returns:
            (timestamp, 'IFVG', price) or None if no IFVG found
        """
        # Get 5M data in window
        mask = (self.df_5m.index >= start_time) & (self.df_5m.index <= end_time)
        window_indices = self.df_5m.index[mask]

        if len(window_indices) == 0:
            return None

        # Determine target FVG type
        # For long entry, we want bearish FVG (-1) that gets disrespected
        # For short entry, we want bullish FVG (1) that gets disrespected
        target_fvg = -1 if entry_direction == 'long' else 1

        # Scan window indices for FVG disrespect
        for time_5m in window_indices:
            idx_5m = self.df_5m.index.get_loc(time_5m)
            local_5_fvg = self.fvg_5m.loc[self.fvg_5m['StatusIndex'] == idx_5m]

            # Check all FVGs to find ones disrespected at this index
            for i in range(len(local_5_fvg)):
                fvg_value = local_5_fvg['FVG'].iloc[i]
                respected = local_5_fvg['Respected'].iloc[i]
                status_index = local_5_fvg['StatusIndex'].iloc[i]

                # Check if:
                # 1. FVG exists at this row
                # 2. It was disrespected (Respected == False)
                # 3. Disrespect happened at current index (StatusIndex == idx_5m)
                # 4. FVG type matches what we need
                if (not np.isnan(fvg_value) and
                    respected == False and
                    status_index == idx_5m and
                    fvg_value == target_fvg):

                    # Found IFVG - find nearest swing for invalidation level
                    # For long (bearish FVG broken): nearest convex (valley) before IFVG
                    # For short (bullish FVG broken): nearest concave (peak) before IFVG
                    target_inflexion_type = -1 if entry_direction == 'long' else 1

                    # Search backwards from current index for nearest inflexion
                    level = None
                    for j in range(idx_5m - 1, -1, -1):
                        inflexion_type = self.inflexions_5m['InflexionType'].iloc[j]
                        if not np.isnan(inflexion_type) and inflexion_type == target_inflexion_type:
                            level = self.inflexions_5m['Level'].iloc[j]
                            break

                    return (time_5m, 'IFVG', level)

        return None

    def search_for_event_b(
        self,
        start_time: datetime,
        end_time: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float]]:
        """
        Combined search for BOS or IFVG (Event B).

        Tries BOS first, then IFVG if no BOS found.

        Args:
            start_time: Start of search window
            end_time: End of search window
            entry_direction: 'long' or 'short'

        Returns:
            (timestamp, event_type, price) or None if not found
        """
        # Try BOS first
        event_b_candidate = self.detect_bos_opposite_direction(start_time, end_time, entry_direction)

        if event_b_candidate is None:
            # Try IFVG
            event_b_candidate = self.detect_ifvg(start_time, end_time, entry_direction)

        return event_b_candidate
