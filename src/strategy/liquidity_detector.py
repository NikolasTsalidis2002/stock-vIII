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
        bos_high: pd.DataFrame,
        proximity_threshold: float = 0.0
    ) -> None:
        """
        Initialize with high timeframe data and indicators.

        Args:
            df_high: High TF OHLCV DataFrame with datetime index
            inflexions_high: Pre-calculated inflexion points
            bos_high: Pre-calculated BOS data
            proximity_threshold: Percentage threshold (as decimal) for near-sweep detection.
                                If price comes within this % of the level, it counts as swept.
                                Default 0.0 = exact touch required.
        """
        self.df_high = df_high
        self.inflexions_high = inflexions_high
        self.bos_high = bos_high
        self.proximity_threshold = proximity_threshold

    def detect_all_sweeps(
        self,
        sweep_type_filter: str = None,
        tradable_start: datetime = None
    ) -> List[SweepInfo]:
        """
        Detect ALL liquidity sweeps progressively (EXACT same logic as walkthrough).

        This matches analyze_1h_tv.py lines 52-61:
        - Calculate indicators progressively (candle by candle)
        - Count ALL sweeps where Respected == True at each candle
        - Track when sweeps first appear
        - DUAL SWEEP: If same candle sweeps both high AND low, use candle color for direction

        Args:
            sweep_type_filter: Optional filter for sweep types:
                - 'high': Only return sweeps of HIGHS (buy-side liquidity)
                - 'low': Only return sweeps of LOWS (sell-side liquidity)
                - 'both' or None: Return all sweeps (default)
            tradable_start: If provided, only return sweeps at or after this timestamp.
                           Sweeps before this time are still detected (for context)
                           but filtered out before return.

        Returns:
            List of SweepInfo objects for ALL sweeps found (filtered to tradable period)
        """
        print(f"  Progressively scanning {len(self.df_high)} candles for liquidity sweeps...")

        all_sweeps = []
        seen_sweeps = set()  # Track (inflexion_idx, sweep_type) to avoid duplicates

        # Scan progressively (EXACT same as walkthrough lines 652-662)
        for i in range(len(self.df_high)):
            # Get slice up to current candle ONLY (no future data)
            df_slice = self.df_high.iloc[:i+1]

            # Calculate indicators on this slice (same as walkthrough line 55)
            inflexions_slice = smc_custom.inflexion_points(df_slice, proximity_threshold=self.proximity_threshold)

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

        print(f"  Total sweeps found: {len(all_sweeps)}")

        # Filter to tradable period if specified
        if tradable_start is not None:
            pre_filter_count = len(all_sweeps)
            all_sweeps = [s for s in all_sweeps if s.timestamp >= tradable_start]
            filtered_out = pre_filter_count - len(all_sweeps)
            if filtered_out > 0:
                print(f"  Filtered out {filtered_out} sweeps before tradable_start ({tradable_start})")
                print(f"  Tradable sweeps: {len(all_sweeps)}")

        # Apply sweep type filter if specified
        if sweep_type_filter and sweep_type_filter != 'both':
            filtered_sweeps = []
            for sweep in all_sweeps:
                # Handle dual sweeps
                if sweep.is_dual_sweep:
                    # Dual sweeps have direction already determined by candle color
                    # dual_short or dual_long
                    if sweep_type_filter == 'high' and sweep.sweep_type == 'dual_short':
                        # SHORT bias looks for HIGH sweeps, dual_short qualifies
                        filtered_sweeps.append(sweep)
                    elif sweep_type_filter == 'low' and sweep.sweep_type == 'dual_long':
                        # LONG bias looks for LOW sweeps, dual_long qualifies
                        filtered_sweeps.append(sweep)
                # Handle single sweeps
                elif sweep.sweep_type == sweep_type_filter:
                    filtered_sweeps.append(sweep)

            print(f"  After {sweep_type_filter} filter: {len(filtered_sweeps)} sweeps\n")
            return filtered_sweeps

        print()
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
        inflexions_slice = smc_custom.inflexion_points(df_slice, proximity_threshold=self.proximity_threshold)

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

    def detect_all_fvg_triggers(
        self,
        fvg_high: pd.DataFrame,
        tradable_start: datetime = None
    ) -> List[SweepInfo]:
        """
        Detect FVG respect events on high TF as Stage 1 triggers.

        FVG respect occurs when price touches an FVG but doesn't break through:
        - Bullish FVG (FVG=1) respected → LONG entry signal
        - Bearish FVG (FVG=-1) respected → SHORT entry signal

        Args:
            fvg_high: Pre-calculated FVG DataFrame for high timeframe
            tradable_start: If provided, only return FVG triggers at or after this timestamp

        Returns:
            List of SweepInfo objects for FVG respect triggers
        """
        print(f"  Scanning {len(fvg_high)} candles for FVG triggers...")

        all_fvg_triggers = []

        # Pre-filter for performance:
        # 1. Has FVG (not NaN)
        # 2. Was touched (MitigatedIndex > 0)
        # 3. NOT immediately disrespected (MitigatedIndex == StatusIndex with Respected == False)
        valid_fvgs = fvg_high[
            fvg_high['FVG'].notna() &
            (fvg_high['MitigatedIndex'] > 0) &
            ~(
                (fvg_high['MitigatedIndex'] == fvg_high['StatusIndex']) &
                (fvg_high['Respected'] == False)
            )
        ]

        for idx, row in valid_fvgs.iterrows():
            fvg_type = row['FVG']
            mitigated_idx = int(row['MitigatedIndex'])
            fvg_top = row['Top']
            fvg_bottom = row['Bottom']

            # The trigger timestamp is when the FVG was touched (MitigatedIndex)
            if mitigated_idx < len(self.df_high):
                timestamp = self.df_high.index[mitigated_idx]
            else:
                continue

            # Determine entry direction based on FVG type
            # Bullish FVG (1) touched = price came down to FVG = LONG
            # Bearish FVG (-1) touched = price came up to FVG = SHORT
            if fvg_type == 1:  # Bullish FVG
                entry_direction = 'long'
                swept_level = fvg_bottom  # Price touched bottom of bullish FVG
            else:  # Bearish FVG (fvg_type == -1)
                entry_direction = 'short'
                swept_level = fvg_top  # Price touched top of bearish FVG

            sweep_type = f'fvg_{entry_direction}'

            print(f"  Found FVG trigger at candle {mitigated_idx+1}: {timestamp} "
                  f"({sweep_type}, FVG range: ${fvg_bottom:.2f}-${fvg_top:.2f})")

            all_fvg_triggers.append(SweepInfo(
                candle_idx=mitigated_idx,
                sweep_type=sweep_type,
                swept_level=swept_level,
                inflexion_idx=idx,  # Store the FVG formation index for invalidation checks
                timestamp=timestamp,
                fvg_formation_idx=idx,
                fvg_top=fvg_top,
                fvg_bottom=fvg_bottom
            ))

        print(f"  Total FVG triggers found: {len(all_fvg_triggers)}")

        # Filter to tradable period if specified
        if tradable_start is not None:
            pre_filter_count = len(all_fvg_triggers)
            all_fvg_triggers = [t for t in all_fvg_triggers if t.timestamp >= tradable_start]
            filtered_out = pre_filter_count - len(all_fvg_triggers)
            if filtered_out > 0:
                print(f"  Filtered out {filtered_out} FVG triggers before tradable_start ({tradable_start})")
                print(f"  Tradable FVG triggers: {len(all_fvg_triggers)}")

        print()
        return all_fvg_triggers

    def is_fvg_broken_at_time(
        self,
        fvg_formation_idx: int,
        fvg_high: pd.DataFrame,
        end_time: datetime
    ) -> bool:
        """
        Check if the FVG that triggered has been disrespected by a specific time.

        An FVG is "broken" when its Respected status is False, meaning price
        closed through the FVG zone after initially touching it.

        Args:
            fvg_formation_idx: Index of the FVG formation in the FVG DataFrame
            fvg_high: Pre-calculated FVG DataFrame for high timeframe
            end_time: Only consider data up to this timestamp (inclusive)

        Returns:
            True if FVG was disrespected by end_time, False if still valid
        """
        if fvg_formation_idx >= len(fvg_high):
            return False

        # Get the current status of the FVG
        respected = fvg_high['Respected'].iloc[fvg_formation_idx]
        status_idx = fvg_high['StatusIndex'].iloc[fvg_formation_idx]

        # If status is False (disrespected), check if it happened before end_time
        if respected == False and not np.isnan(status_idx):
            status_idx = int(status_idx)
            if status_idx < len(self.df_high):
                status_time = self.df_high.index[status_idx]
                # Only consider broken if disrespect happened at or before end_time
                return status_time <= end_time

        return False
