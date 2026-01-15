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

    def detect_confirmation_at_candle(
        self,
        time_1m: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float]]:
        """
        Check single 1M candle for confirmation (BOS or IFVG in entry direction).

        Args:
            time_1m: The specific candle timestamp to check
            entry_direction: 'long' or 'short'

        Returns:
            (timestamp, confirmation_type, price) where type is 'BOS' or 'IFVG'
        """
        idx_1m = self.df_1m.index.get_loc(time_1m)

        # Check BOS at this candle
        target_bos = 1 if entry_direction == 'long' else -1
        bos_value = self.bos_1m['BOS'].iloc[idx_1m]

        if not np.isnan(bos_value) and bos_value == target_bos:
            level = self.bos_1m['Level'].iloc[idx_1m]
            return (time_1m, 'BOS', level)

        # Check IFVG at this candle
        # For long entry, we want bearish FVG (-1) that gets disrespected
        # For short entry, we want bullish FVG (1) that gets disrespected
        target_fvg = -1 if entry_direction == 'long' else 1

        # Only check FVGs disrespected at this specific index
        local_fvg = self.fvg_1m.loc[self.fvg_1m['StatusIndex'] == idx_1m]

        for i in range(len(local_fvg)):
            fvg_value = local_fvg['FVG'].iloc[i]
            respected = local_fvg['Respected'].iloc[i]

            if (not np.isnan(fvg_value) and
                respected == False and
                fvg_value == target_fvg):
                # Found IFVG - FVG disrespected at this candle
                level = (local_fvg['Top'].iloc[i] + local_fvg['Bottom'].iloc[i]) / 2
                return (time_1m, 'IFVG', level)

        return None
