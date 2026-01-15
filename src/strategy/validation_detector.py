"""
Validation detection (FVG/Demand Zone) on 5M timeframe.

Handles detection of Fair Value Gaps and Order Blocks (Demand Zones)
that validate the trade setup.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple


class ValidationDetector:
    """
    Detects FVG or Demand Zone validation on 5M.

    Responsibilities:
    - FVG respect detection
    - Order Block (Demand Zone) respect detection
    - Combined search helper
    """

    def __init__(
        self,
        df_5m: pd.DataFrame,
        fvg_5m: pd.DataFrame,
        ob_5m: pd.DataFrame
    ) -> None:
        """
        Initialize with 5M data and indicators.

        Args:
            df_5m: 5M OHLCV DataFrame with datetime index
            fvg_5m: Pre-calculated FVG data
            ob_5m: Pre-calculated Order Block data
        """
        self.df_5m = df_5m
        self.fvg_5m = fvg_5m
        self.ob_5m = ob_5m

    def validate_fvg_or_demand_zone(
        self,
        start_time: datetime,
        end_time: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float, Optional[int]]]:
        """
        Check if price respects FVG or Demand Zone (Order Block) on 5M.

        Args:
            start_time: Start of 5M analysis window
            end_time: End of 5M analysis window
            entry_direction: 'long' or 'short' (determines which zones to check)

        Returns:
            (timestamp, zone_type, price, fvg_index) where zone_type is 'FVG' or 'Demand Zone'
            fvg_index is the original DataFrame index for FVG, None for Demand Zone
        """
        # Get 5M data in window
        mask = (self.df_5m.index >= start_time) & (self.df_5m.index <= end_time)
        window_indices = self.df_5m.index[mask]

        if len(window_indices) == 0:
            return None

        # Determine target FVG
        # For long entry, we want bullish FVG (1)
        # For short entry, we want bearish FVG (-1)
        target_fvg = 1 if entry_direction == 'long' else -1

        # Check FVG respect - progressively scan through window
        for time_5m in window_indices:
            idx_5m = self.df_5m.index.get_loc(time_5m)

            # Filter FVGs that were mitigated at this specific moment
            local_5m_fvg = self.fvg_5m.loc[self.fvg_5m['MitigatedIndex'] == idx_5m]

            # Check all FVGs mitigated at this index
            for i in range(len(local_5m_fvg)):
                fvg_value = local_5m_fvg['FVG'].iloc[i]
                respected = local_5m_fvg['Respected'].iloc[i]
                status_index = local_5m_fvg['StatusIndex'].iloc[i]

                # Check if:
                # 1. FVG exists at this row
                # 2. It was respected (Respected == True)
                # 3. FVG type matches what we need (entry direction)
                if (not np.isnan(fvg_value) and
                    status_index != idx_5m and
                    fvg_value == target_fvg):

                    # Found respected FVG at this moment
                    level = (local_5m_fvg['Top'].iloc[i] + local_5m_fvg['Bottom'].iloc[i]) / 2
                    # Get the original DataFrame index for this FVG
                    original_fvg_index = local_5m_fvg.index[i]
                    return (time_5m, 'FVG', level, original_fvg_index)

        # Check Order Block (Demand Zone) - iterate through window
        for time_5m in window_indices:
            idx_5m = self.df_5m.index.get_loc(time_5m)
            ob_value = self.ob_5m['OB'].iloc[idx_5m]
            if ob_value != 0 and not np.isnan(ob_value):
                # For long entry, we want bullish OB (ob_value = 1)
                # For short entry, we want bearish OB (ob_value = -1)

                if (entry_direction == 'long' and ob_value == 1) or \
                   (entry_direction == 'short' and ob_value == -1):

                    ob_top = self.ob_5m['Top'].iloc[idx_5m]
                    ob_bottom = self.ob_5m['Bottom'].iloc[idx_5m]
                    mitigated_idx = self.ob_5m['MitigatedIndex'].iloc[idx_5m]

                    # Check if OB was respected (touched but not fully mitigated)
                    if mitigated_idx == 0:
                        # Not yet mitigated - check if price is touching it
                        for j in range(idx_5m + 1, min(idx_5m + 10, len(self.df_5m))):
                            if entry_direction == 'long':
                                if self.df_5m['low'].iloc[j] <= ob_top and \
                                   self.df_5m['close'].iloc[j] >= ob_bottom:
                                    # Price touched and bounced - OB respected
                                    level = (ob_top + ob_bottom) / 2
                                    return (self.df_5m.index[j], 'Demand Zone', level, None)
                            else:
                                if self.df_5m['high'].iloc[j] >= ob_bottom and \
                                   self.df_5m['close'].iloc[j] <= ob_top:
                                    # Price touched and bounced - OB respected
                                    level = (ob_top + ob_bottom) / 2
                                    return (self.df_5m.index[j], 'Demand Zone', level, None)

        return None

    def is_fvg_broken(self, fvg_index: int, current_time: datetime) -> bool:
        """
        Check if a specific FVG has been broken.

        Args:
            fvg_index: Original DataFrame index of the FVG
            current_time: Current timestamp

        Returns:
            True if FVG was disrespected
        """
        if fvg_index is None:
            return False

        fvg_respected = self.fvg_5m.loc[fvg_index, 'Respected']
        fvg_status_index = self.fvg_5m.loc[fvg_index, 'StatusIndex']

        # Check if FVG was disrespected and status determination happened by now
        if not np.isnan(fvg_status_index):
            # Map status index to time and check if it's in the past
            status_time = self.df_5m.index[int(fvg_status_index)]
            if status_time <= current_time and fvg_respected == False:
                return True

        return False
