"""
Liquidity sweep detection and invalidation checking on high timeframe.

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
    Detects liquidity sweeps on high timeframe.

    Responsibilities:
    - Progressive liquidity sweep detection
    - Sweep broken/invalidation checking
    """

    def __init__(
        self,
        df_high: pd.DataFrame,
        inflexions_high: pd.DataFrame,
        bos_high: pd.DataFrame
    ) -> None:
        """
        Initialize with high timeframe data and indicators.

        Args:
            df_high: High TF OHLCV DataFrame with datetime index
            inflexions_high: Pre-calculated inflexion points
            bos_high: Pre-calculated BOS data
        """
        self.df_high = df_high
        self.inflexions_high = inflexions_high
        self.bos_high = bos_high

    def detect_all_sweeps(self) -> List[SweepInfo]:
        """
        Detect ALL liquidity sweeps progressively (EXACT same logic as walkthrough).

        This matches analyze_1h_tv.py lines 52-61:
        - Calculate indicators progressively (candle by candle)
        - Count ALL sweeps where Respected == True at each candle
        - Track when sweeps first appear
        - DUAL SWEEP: If same candle sweeps both high AND low, use candle color for direction

        Returns:
            List of SweepInfo objects for ALL sweeps found
        """
        print(f"  Progressively scanning {len(self.df_high)} candles for liquidity sweeps...")

        all_sweeps = []
        seen_sweeps = set()  # Track (inflexion_idx, sweep_type) to avoid duplicates

        # Scan progressively (EXACT same as walkthrough lines 652-662)
        for i in range(len(self.df_high)):
            # Get slice up to current candle ONLY (no future data)
            df_slice = self.df_high.iloc[:i+1]

            # Calculate indicators on this slice (same as walkthrough line 55)
            inflexions_slice = smc_custom.inflexion_points(df_slice)

            # Track NEW sweeps at this candle (for dual sweep detection)
            new_high_sweeps = []  # List of (inflexion_idx, level)
            new_low_sweeps = []   # List of (inflexion_idx, level)

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
                        if sweep_type == 'high':
                            new_high_sweeps.append((j, level))
                        else:
                            new_low_sweeps.append((j, level))

            timestamp = self.df_high.index[i]

            # Check for DUAL SWEEP: same candle sweeps both high AND low
            if new_high_sweeps and new_low_sweeps:
                candle = self.df_high.iloc[i]
                is_green = candle['close'] > candle['open']

                # Use closest levels if multiple
                high_inflexion_idx, high_level = new_high_sweeps[0]
                low_inflexion_idx, low_level = new_low_sweeps[0]

                # Direction based on candle color
                if is_green:
                    entry_direction = 'short'  # Green + dual → Short
                else:
                    entry_direction = 'long'   # Red + dual → Long

                sweep_type = f'dual_{entry_direction}'

                print(f"  🔄 DUAL LIQUIDITY SWEEP at candle {i+1}: {timestamp}")
                print(f"     High swept: ${high_level:.2f} (inflexion idx: {high_inflexion_idx})")
                print(f"     Low swept: ${low_level:.2f} (inflexion idx: {low_inflexion_idx})")
                print(f"     Candle color: {'GREEN' if is_green else 'RED'} (close={'>' if is_green else '<'}open)")
                print(f"     → Entry direction: {entry_direction.upper()}")

                # Use the level that matches our direction for swept_level
                # For short: use the high level, For long: use the low level
                if entry_direction == 'short':
                    swept_level = high_level
                    primary_inflexion_idx = high_inflexion_idx
                else:
                    swept_level = low_level
                    primary_inflexion_idx = low_inflexion_idx

                all_sweeps.append(SweepInfo(
                    candle_idx=i,
                    sweep_type=sweep_type,
                    swept_level=swept_level,
                    inflexion_idx=primary_inflexion_idx,
                    timestamp=timestamp,
                    is_dual_sweep=True,
                    dual_high_level=high_level,
                    dual_low_level=low_level,
                    dual_high_inflexion_idx=high_inflexion_idx,
                    dual_low_inflexion_idx=low_inflexion_idx
                ))
            else:
                # Single sweeps (original logic)
                for inflexion_idx, level in new_high_sweeps:
                    print(f"  Found liquidity sweep at candle {i+1}: {timestamp} (high, level: ${level:.2f})")
                    all_sweeps.append(SweepInfo(
                        candle_idx=i,
                        sweep_type='high',
                        swept_level=level,
                        inflexion_idx=inflexion_idx,
                        timestamp=timestamp
                    ))

                for inflexion_idx, level in new_low_sweeps:
                    print(f"  Found liquidity sweep at candle {i+1}: {timestamp} (low, level: ${level:.2f})")
                    all_sweeps.append(SweepInfo(
                        candle_idx=i,
                        sweep_type='low',
                        swept_level=level,
                        inflexion_idx=inflexion_idx,
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
        df_slice = self.df_high.loc[:end_time]

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
