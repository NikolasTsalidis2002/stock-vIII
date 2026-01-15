"""
Liquidity sweep detection and invalidation checking on 1H timeframe.

Handles progressive liquidity sweep detection and validation of sweeps
during the strategy workflow.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List

from indicators.smc_custom import smc_custom
from .models import SweepInfo


class LiquidityDetector:
    """
    Detects liquidity sweeps on 1H timeframe.

    Responsibilities:
    - Progressive liquidity sweep detection
    - Sweep broken/invalidation checking
    """

    def __init__(
        self,
        df_1h: pd.DataFrame,
        inflexions_1h: pd.DataFrame,
        bos_1h: pd.DataFrame
    ) -> None:
        """
        Initialize with 1H data and indicators.

        Args:
            df_1h: 1H OHLCV DataFrame with datetime index
            inflexions_1h: Pre-calculated inflexion points
            bos_1h: Pre-calculated BOS data
        """
        self.df_1h = df_1h
        self.inflexions_1h = inflexions_1h
        self.bos_1h = bos_1h

    def detect_all_sweeps(self) -> List[SweepInfo]:
        """
        Detect ALL liquidity sweeps progressively (EXACT same logic as walkthrough).

        This matches analyze_1h_tv.py lines 52-61:
        - Calculate indicators progressively (candle by candle)
        - Count ALL sweeps where Respected == True at each candle
        - Track when sweeps first appear

        Returns:
            List of SweepInfo objects for ALL sweeps found
        """
        print(f"  Progressively scanning {len(self.df_1h)} candles for liquidity sweeps...")

        all_sweeps = []
        seen_sweeps = set()  # Track (inflexion_idx, sweep_type) to avoid duplicates

        # Scan progressively (EXACT same as walkthrough lines 652-662)
        for i in range(len(self.df_1h)):
            # Get slice up to current candle ONLY (no future data)
            df_slice = self.df_1h.iloc[:i+1]

            # Calculate indicators on this slice (same as walkthrough line 55)
            inflexions_slice = smc_custom.inflexion_points(df_slice)

            # Find ALL sweeps at this moment (same as walkthrough line 61)
            for j in range(len(inflexions_slice)):
                inflexion_type = inflexions_slice['InflexionType'].iloc[j]
                respected = inflexions_slice['Respected'].iloc[j]
                level = inflexions_slice['Level'].iloc[j]

                if not np.isnan(inflexion_type) and respected == True:
                    sweep_type = 'high' if inflexion_type == 1 else 'low'

                    # Check if this is a NEW sweep we haven't seen before
                    sweep_key = (j, sweep_type)
                    if sweep_key not in seen_sweeps:
                        seen_sweeps.add(sweep_key)
                        timestamp = self.df_1h.index[i]
                        print(f"  Found liquidity sweep at candle {i+1}: {timestamp} ({sweep_type}, level: ${level:.2f})")
                        all_sweeps.append(SweepInfo(
                            candle_idx=i,
                            sweep_type=sweep_type,
                            swept_level=level,
                            inflexion_idx=j,
                            timestamp=timestamp
                        ))

        print(f"  Total sweeps found: {len(all_sweeps)}\n")
        return all_sweeps

    def is_sweep_broken_at_time(
        self,
        inflexion_idx: int,
        end_time: datetime,
        current_candle_idx: int
    ) -> bool:
        """
        Check if a sweep has been broken UP TO a specific time (time-aware, no future data).

        This ensures we only use data available at that moment, simulating real-time trading.
        A sweep is broken when its Respected status changes from True to False.

        Args:
            inflexion_idx: Index of the inflexion point in the FULL data
            end_time: Only consider data up to this timestamp (inclusive)
            current_candle_idx: Current 1H candle index

        Returns:
            True if broken (Respected == False) by end_time, False if still valid
        """
        # Only use data UP TO end_time (no future data leakage)
        df_slice = self.df_1h.loc[:end_time]

        # Calculate inflexions on this time-limited slice
        inflexions_slice = smc_custom.inflexion_points(df_slice)

        # Check if this inflexion is broken in the slice
        if inflexion_idx < len(inflexions_slice):
            status_index = inflexions_slice['StatusIndex'].iloc[inflexion_idx]

            # If StatusIndex is greater than current index, the respect/disrespect determination
            # happens in the future (after end_time), so we can't use that information yet
            if not np.isnan(status_index) and status_index > current_candle_idx:
                return False

            # StatusIndex is within our current time window, check the Respected status
            respected = inflexions_slice['Respected'].iloc[inflexion_idx]
            return respected == False

        return False
