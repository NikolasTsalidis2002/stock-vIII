"""
Final confirmation detection (BOS/IFVG) on low timeframe.

Handles detection of Break of Structure and Inverse Fair Value Gaps
that provide final entry confirmation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple


class ConfirmationDetector:
    """
    Detects final confirmation (BOS or IFVG) on low TF.

    Responsibilities:
    - BOS detection in entry direction
    - IFVG detection in entry direction
    """

    def __init__(
        self,
        df_low: pd.DataFrame,
        bos_low: pd.DataFrame,
        fvg_low: pd.DataFrame,
        inflexions_low: pd.DataFrame
    ) -> None:
        """
        Initialize with low TF data and indicators.

        Args:
            df_low: Low TF OHLCV DataFrame with datetime index
            bos_low: Pre-calculated BOS data
            fvg_low: Pre-calculated FVG data
            inflexions_low: Pre-calculated inflexion points
        """
        self.df_low = df_low
        self.bos_low = bos_low
        self.fvg_low = fvg_low
        self.inflexions_low = inflexions_low

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
        idx_1m = self.df_low.index.get_loc(time_1m)

        # Check BOS at this candle
        target_bos = 1 if entry_direction == 'long' else -1
        bos_value = self.bos_low['BOS'].iloc[idx_1m]

        if not np.isnan(bos_value) and bos_value == target_bos:
            level = self.bos_low['Level'].iloc[idx_1m]
            return (time_1m, 'BOS', level)

        # Check IFVG at this candle
        # For long entry, we want bearish FVG (-1) that gets disrespected
        # For short entry, we want bullish FVG (1) that gets disrespected
        target_fvg = -1 if entry_direction == 'long' else 1

        # Only check FVGs disrespected at this specific index
        local_fvg = self.fvg_low.loc[self.fvg_low['StatusIndex'] == idx_1m]

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
