"""
Final confirmation detection (BOS/IFVG) on 1M timeframe.

Handles detection of Break of Structure and Inverse Fair Value Gaps
that provide final entry confirmation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple


class ConfirmationDetector:
    """
    Detects final confirmation (BOS or IFVG) on 1M.

    Responsibilities:
    - BOS detection in entry direction
    - IFVG detection in entry direction
    """

    def __init__(
        self,
        df_1m: pd.DataFrame,
        bos_1m: pd.DataFrame,
        fvg_1m: pd.DataFrame,
        inflexions_1m: pd.DataFrame
    ) -> None:
        """
        Initialize with 1M data and indicators.

        Args:
            df_1m: 1M OHLCV DataFrame with datetime index
            bos_1m: Pre-calculated BOS data
            fvg_1m: Pre-calculated FVG data
            inflexions_1m: Pre-calculated inflexion points
        """
        self.df_1m = df_1m
        self.bos_1m = bos_1m
        self.fvg_1m = fvg_1m
        self.inflexions_1m = inflexions_1m

    def detect_final_confirmation(
        self,
        start_time: datetime,
        end_time: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float]]:
        """
        Detect final confirmation on 1M (BOS or IFVG in entry direction).

        Args:
            start_time: Start of 1M analysis window
            end_time: End of 1M analysis window
            entry_direction: 'long' or 'short'

        Returns:
            (timestamp, confirmation_type, price) where type is 'BOS' or 'IFVG'
        """
        # Get 1M data in window
        mask = (self.df_1m.index >= start_time) & (self.df_1m.index <= end_time)
        window_indices = self.df_1m.index[mask]

        if len(window_indices) == 0:
            return None

        # Target BOS direction
        target_bos = 1 if entry_direction == 'long' else -1

        # Check for BOS
        for time_1m in window_indices:
            idx_1m = self.df_1m.index.get_loc(time_1m)
            bos_value = self.bos_1m['BOS'].iloc[idx_1m]

            if not np.isnan(bos_value) and bos_value == target_bos:
                # Found BOS in entry direction
                level = self.bos_1m['Level'].iloc[idx_1m]
                return (time_1m, 'BOS', level)

        # Check for IFVG (same logic as detect_ifvg_5m)
        # Determine target FVG type (opposite to entry direction)
        # For long entry, we want bearish FVG (-1) that gets disrespected
        # For short entry, we want bullish FVG (1) that gets disrespected
        target_fvg = -1 if entry_direction == 'long' else 1

        # Scan window indices for FVG disrespect
        for time_1m in window_indices:
            idx_1m = self.df_1m.index.get_loc(time_1m)

            # Check all FVGs to find ones disrespected at this index
            for i in range(len(self.fvg_1m)):
                fvg_value = self.fvg_1m['FVG'].iloc[i]
                respected = self.fvg_1m['Respected'].iloc[i]
                status_index = self.fvg_1m['StatusIndex'].iloc[i]

                # Check if:
                # 1. FVG exists at this row
                # 2. It was disrespected (Respected == False)
                # 3. Disrespect happened at current index (StatusIndex == idx_1m)
                # 4. FVG type matches what we need
                if (not np.isnan(fvg_value) and
                    respected == False and
                    status_index == idx_1m and
                    fvg_value == target_fvg):

                    # Found IFVG - FVG disrespected at this candle
                    level = (self.fvg_1m['Top'].iloc[i] + self.fvg_1m['Bottom'].iloc[i]) / 2
                    return (time_1m, 'IFVG', level)

        return None
