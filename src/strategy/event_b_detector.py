"""
Event B detection (BOS/IFVG) on mid timeframe.

Handles detection of Break of Structure and Inverse Fair Value Gaps
that signal potential trade entries.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple


class EventBDetector:
    """
    Detects Event B (BOS or IFVG in opposite direction) on mid TF.

    Responsibilities:
    - BOS detection in entry direction
    - Inverse FVG detection (FVG that gets disrespected)
    - Combined search helper
    """

    def __init__(
        self,
        df_mid: pd.DataFrame,
        bos_mid: pd.DataFrame,
        fvg_mid: pd.DataFrame,
        inflexions_mid: pd.DataFrame
    ) -> None:
        """
        Initialize with mid TF data and indicators.

        Args:
            df_mid: Mid TF OHLCV DataFrame with datetime index
            bos_mid: Pre-calculated BOS data
            fvg_mid: Pre-calculated FVG data
            inflexions_mid: Pre-calculated inflexion points
        """
        self.df_mid = df_mid
        self.bos_mid = bos_mid
        self.fvg_mid = fvg_mid
        self.inflexions_mid = inflexions_mid

    def detect_bos_at_candle(
        self,
        time_5m: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float]]:
        """
        Check single 5M candle for BOS in entry direction.

        Args:
            time_5m: The specific candle timestamp to check
            entry_direction: Intended entry direction ('long' or 'short')

        Returns:
            (timestamp, 'BOS', price) or None if no BOS at this candle
        """
        idx_5m = self.df_mid.index.get_loc(time_5m)
        target_bos = 1 if entry_direction == 'long' else -1
        bos_value = self.bos_mid['BOS'].iloc[idx_5m]

        if not np.isnan(bos_value) and bos_value == target_bos:
            level = self.bos_mid['Level'].iloc[idx_5m]
            return (time_5m, 'BOS', level)

        return None

    def detect_ifvg_at_candle(
        self,
        time_5m: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float]]:
        """
        Check single 5M candle for Inverse FVG (FVG that gets disrespected).

        An IFVG is an FVG in the opposite direction that gets broken/disrespected,
        confirming the move in the entry direction.

        Args:
            time_5m: The specific candle timestamp to check
            entry_direction: Intended entry direction ('long' or 'short')

        Returns:
            (timestamp, 'IFVG', price) or None if no IFVG at this candle
        """
        idx_5m = self.df_mid.index.get_loc(time_5m)

        # Determine target FVG type
        # For long entry, we want bearish FVG (-1) that gets disrespected
        # For short entry, we want bullish FVG (1) that gets disrespected
        target_fvg = -1 if entry_direction == 'long' else 1

        # Check FVGs disrespected at this specific index
        local_5_fvg = self.fvg_mid.loc[self.fvg_mid['StatusIndex'] == idx_5m]

        for i in range(len(local_5_fvg)):
            fvg_value = local_5_fvg['FVG'].iloc[i]
            respected = local_5_fvg['Respected'].iloc[i]

            # Check if:
            # 1. FVG exists at this row
            # 2. It was disrespected (Respected == False)
            # 3. FVG type matches what we need
            if (not np.isnan(fvg_value) and
                respected == False and
                fvg_value == target_fvg):

                # Found IFVG - find nearest swing for invalidation level
                # For long (bearish FVG broken): nearest convex (valley) before IFVG
                # For short (bullish FVG broken): nearest concave (peak) before IFVG
                target_inflexion_type = -1 if entry_direction == 'long' else 1

                # Search backwards from current index for nearest inflexion
                level = None
                for j in range(idx_5m - 1, -1, -1):
                    inflexion_type = self.inflexions_mid['InflexionType'].iloc[j]
                    if not np.isnan(inflexion_type) and inflexion_type == target_inflexion_type:
                        level = self.inflexions_mid['Level'].iloc[j]
                        break

                return (time_5m, 'IFVG', level)

        return None

    def search_for_event_b_at_candle(
        self,
        time_5m: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float]]:
        """
        Check single candle for Event B (BOS or IFVG).

        Tries BOS first, then IFVG if no BOS found.

        Args:
            time_5m: The specific candle timestamp to check
            entry_direction: 'long' or 'short'

        Returns:
            (timestamp, event_type, price) or None if not found
        """
        # Try BOS first
        result = self.detect_bos_at_candle(time_5m, entry_direction)

        if result is None:
            # Try IFVG
            result = self.detect_ifvg_at_candle(time_5m, entry_direction)

        return result
