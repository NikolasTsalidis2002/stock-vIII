"""
TSLA Multi-Timeframe Trading Strategy Engine

Implements the strategy defined in strategy.md:
1. 1H Liquidity Sweep detection (trigger)
2. 5M BOS/IFVG in opposite direction (Event B)
3. 5M FVG or Demand Zone validation
4. 1M Final confirmation (BOS or IFVG)
5. Entry signal generation

Uses hybrid indicators:
- Custom BOS + Inflexion Points (more precise)
- SMC FVG, Liquidity, Order Blocks
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smartmoneyconcepts.smc import smc
from smartmoneyconcepts.smc_custom import smc_custom


class StrategyState(Enum):
    """Strategy state machine states."""
    WAITING_FOR_LIQUIDITY_SWEEP = 1
    WAITING_FOR_EVENT_B = 2          # 5M BOS/IFVG opposite direction
    WAITING_FOR_VALIDATION = 3       # 5M FVG/Demand respect
    WAITING_FOR_FINAL_CONFIRMATION = 4  # 1M BOS/IFVG
    ENTRY_SIGNAL_GENERATED = 5


@dataclass
class PartialSetup:
    """Represents a partial trade setup (not all conditions met)."""

    # Timestamps (None if condition not met)
    timestamp_1h_sweep: datetime
    timestamp_5m_event_b: Optional[datetime]
    timestamp_5m_validation: Optional[datetime]
    timestamp_1m_confirmation: Optional[datetime]

    # Prices
    price_1h_sweep: float
    price_5m_event_b: Optional[float]
    price_5m_validation: Optional[float]
    price_1m_confirmation: Optional[float]

    # Trend and direction
    trend_1h_before_sweep: str
    entry_direction: str

    # Conditions met
    condition_liquidity_sweep: str
    condition_event_b: Optional[str]
    condition_validation: Optional[str]
    condition_confirmation: Optional[str]

    # Status tracking
    conditions_met: int  # How many of 4 conditions passed
    failure_reason: str  # Where it failed

    # Metadata for visualization
    indices_1h: Tuple[int, int]
    indices_5m: Optional[Tuple[int, int]]
    indices_1m: Optional[Tuple[int, int]]
    inflexion_idx: int  # Index of the inflexion point for sweep validation


@dataclass
class TradeSignal:
    """Represents a complete trade setup with all conditions met."""

    # Timestamps
    timestamp_1h_sweep: datetime
    timestamp_5m_event_b: datetime
    timestamp_5m_validation: datetime
    timestamp_1m_confirmation: datetime
    timestamp_entry: datetime

    # Trend information
    trend_1h_before_sweep: str  # 'bullish' or 'bearish'
    entry_direction: str        # 'long' or 'short' (opposite of 1h trend)

    # Prices
    price_1h_sweep: float
    price_5m_event_b: float
    price_5m_validation: float
    price_1m_confirmation: float
    price_entry: float

    # Conditions met
    condition_liquidity_sweep: str  # Description of liquidity sweep
    condition_event_b: str          # 'BOS' or 'IFVG'
    condition_validation: str       # 'FVG' or 'Demand Zone'
    condition_confirmation: str     # 'BOS' or 'IFVG'

    # Metadata for visualization
    indices_1h: Tuple[int, int]     # (start, end) indices in 1H data
    indices_5m: Tuple[int, int]     # (start, end) indices in 5M data
    indices_1m: Tuple[int, int]     # (start, end) indices in 1M data


class MultiTimeframeStrategy:
    """
    Main strategy engine implementing multi-timeframe analysis.

    Workflow:
    1. Load data for 1H, 5M, 1M timeframes
    2. Align timeframes (map candles across timeframes)
    3. Scan for liquidity sweeps on 1H
    4. For each sweep, analyze 5M for Event B
    5. For each Event B, validate with FVG/Demand Zone
    6. For each validation, check 1M for final confirmation
    7. Generate trade signals
    """

    def __init__(
        self,
        df_1h: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_1m: pd.DataFrame,
        max_price_deviation_percent: float = 3.0
    ):
        """
        Initialize strategy with multi-timeframe data.

        Args:
            df_1h: 1-hour OHLCV data with 'time' column
            df_5m: 5-minute OHLCV data with 'time' column
            df_1m: 1-minute OHLCV data with 'time' column
            max_price_deviation_percent: Maximum % price can move from sweep before invalidating setup (default: 1.0)
        """
        # Store invalidation threshold
        self.max_price_deviation_percent = max_price_deviation_percent
        # Store original data
        self.df_1h_original = df_1h.copy()
        self.df_5m_original = df_5m.copy()
        self.df_1m_original = df_1m.copy()

        # Prepare indexed versions
        if 'time' in df_1h.columns:
            self.df_1h = df_1h.set_index('time')
        else:
            self.df_1h = df_1h.copy()

        if 'time' in df_5m.columns:
            self.df_5m = df_5m.set_index('time')
        else:
            self.df_5m = df_5m.copy()

        if 'time' in df_1m.columns:
            self.df_1m = df_1m.set_index('time')
        else:
            self.df_1m = df_1m.copy()

        # Ensure datetime index
        self.df_1h.index = pd.to_datetime(self.df_1h.index)
        self.df_5m.index = pd.to_datetime(self.df_5m.index)
        self.df_1m.index = pd.to_datetime(self.df_1m.index)

        # Determine overlapping time period
        self._determine_overlapping_period()

        # Create timeframe mapping
        self._create_timeframe_mapping()

        # Pre-calculate indicators
        self._calculate_indicators()

        # Trade signals found
        self.signals: List[TradeSignal] = []
        self.partial_setups: List[PartialSetup] = []

    def _determine_overlapping_period(self):
        """
        Determine the overlapping time period across all timeframes.

        Only analyze the period where all three timeframes have data.
        """
        # Find latest start time
        start_1h = self.df_1h.index.min()
        start_5m = self.df_5m.index.min()
        start_1m = self.df_1m.index.min()
        self.overlap_start = max(start_1h, start_5m, start_1m)

        # Find earliest end time
        end_1h = self.df_1h.index.max()
        end_5m = self.df_5m.index.max()
        end_1m = self.df_1m.index.max()
        self.overlap_end = min(end_1h, end_5m, end_1m)

        print(f"\n⏱️  Data overlap period:")
        print(f"  Start: {self.overlap_start}")
        print(f"  End:   {self.overlap_end}")
        print(f"  Duration: {self.overlap_end - self.overlap_start}")

        # Filter dataframes to overlapping period
        self.df_1h = self.df_1h.loc[self.overlap_start:self.overlap_end]
        self.df_5m = self.df_5m.loc[self.overlap_start:self.overlap_end]
        self.df_1m = self.df_1m.loc[self.overlap_start:self.overlap_end]

        print(f"\n  Filtered data:")
        print(f"  1H: {len(self.df_1h)} candles")
        print(f"  5M: {len(self.df_5m)} candles")
        print(f"  1M: {len(self.df_1m)} candles")

    def _create_timeframe_mapping(self):
        """
        Create mapping between timeframes.

        For each 1H candle, determine which 5M and 1M candles fall within it.
        For each 5M candle, determine which 1M candles fall within it.
        """
        print("📊 Creating timeframe mapping...")

        # Map 1H → 5M
        self.map_1h_to_5m = {}  # {1h_index: [list of 5m indices]}

        for i, time_1h in enumerate(self.df_1h.index):
            # 1H candle covers [time_1h, time_1h + 1 hour)
            end_time = time_1h + timedelta(hours=1)

            # Find all 5M candles in this range
            mask_5m = (self.df_5m.index >= time_1h) & (self.df_5m.index < end_time)
            indices_5m = self.df_5m.index[mask_5m].tolist()

            self.map_1h_to_5m[i] = indices_5m

        # Map 5M → 1M
        self.map_5m_to_1m = {}  # {5m_index: [list of 1m indices]}

        for i, time_5m in enumerate(self.df_5m.index):
            # 5M candle covers [time_5m, time_5m + 5 minutes)
            end_time = time_5m + timedelta(minutes=5)

            # Find all 1M candles in this range
            mask_1m = (self.df_1m.index >= time_5m) & (self.df_1m.index < end_time)
            indices_1m = self.df_1m.index[mask_1m].tolist()

            self.map_5m_to_1m[i] = indices_1m

        print(f"  ✓ Mapped {len(self.df_1h)} 1H candles")
        print(f"  ✓ Mapped {len(self.df_5m)} 5M candles")
        print(f"  ✓ Mapped {len(self.df_1m)} 1M candles")

    def _calculate_indicators(self):
        """Pre-calculate all indicators for each timeframe."""
        print("\n📈 Calculating indicators...")

        # 1H Indicators (using proper liquidity detection)
        print("  [1H] Calculating indicators...")
        self.swings_1h = smc.swing_highs_lows(self.df_1h, swing_length=3)  # Shorter swing for more swing points
        self.liquidity_1h = smc.liquidity(self.df_1h, self.swings_1h, range_percent=0.01)  # 1% range
        self.inflexions_1h = smc_custom.inflexion_points(self.df_1h)
        self.bos_1h = smc_custom.bos(self.df_1h, self.inflexions_1h, close_break=True)

        # 5M Indicators
        print("  [5M] Calculating indicators...")
        self.swings_5m = smc.swing_highs_lows(self.df_5m, swing_length=30)
        self.inflexions_5m = smc_custom.inflexion_points(self.df_5m)
        self.bos_5m = smc_custom.bos(self.df_5m, self.inflexions_5m, close_break=True)
        self.fvg_5m = smc.fvg(self.df_5m, join_consecutive=True)
        self.ob_5m = smc.ob(self.df_5m, self.swings_5m)  # Order Blocks (Demand Zones)

        # 1M Indicators
        print("  [1M] Calculating indicators...")
        self.inflexions_1m = smc_custom.inflexion_points(self.df_1m)
        self.bos_1m = smc_custom.bos(self.df_1m, self.inflexions_1m, close_break=True)
        self.fvg_1m = smc.fvg(self.df_1m, join_consecutive=True)

        print("  ✓ All indicators calculated\n")

    def get_1h_trend_at_index(self, idx: int) -> Optional[str]:
        """
        Get the trend at a specific 1H index.

        Args:
            idx: Index in 1H dataframe

        Returns:
            'bullish', 'bearish', or None if undefined
        """
        trend_value = self.bos_1h['Trend'].iloc[idx]

        if trend_value == 1:
            return 'bullish'
        elif trend_value == -1:
            return 'bearish'
        else:
            return None

    def detect_all_liquidity_sweeps_1h(self) -> List[Tuple[int, str, float, int]]:
        """
        Detect ALL liquidity sweeps progressively (EXACT same logic as walkthrough).

        This matches analyze_1h_tv.py lines 52-61:
        - Calculate indicators progressively (candle by candle)
        - Count ALL sweeps where Respected == True at each candle
        - Track when sweeps first appear

        Returns:
            List of (candle_idx, sweep_type, swept_level, inflexion_idx) tuples for ALL sweeps found
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
                        all_sweeps.append((i, sweep_type, level, j))  # j = inflexion index

        print(f"  Total sweeps found: {len(all_sweeps)}\n")
        return all_sweeps

    def is_sweep_broken_at_time(self, inflexion_idx: int, end_time: datetime, current_candle_idx: int) -> bool:
        """
        Check if a sweep has been broken UP TO a specific time (time-aware, no future data).

        This ensures we only use data available at that moment, simulating real-time trading.
        A sweep is broken when its Respected status changes from True to False.

        Args:
            inflexion_idx: Index of the inflexion point in the FULL data
            end_time: Only consider data up to this timestamp (inclusive)

        Returns:
            True if broken (Respected == False) by end_time, False if still valid
        """
        # Only use data UP TO end_time (no future data leakage)
        df_slice = self.df_1h.loc[:end_time]

        # Calculate inflexions on this time-limited slice
        inflexions_slice = smc_custom.inflexion_points(df_slice)

        # print inflexions slice for debugging
        # print(inflexions_slice)

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

    def is_price_moved_too_far(
        self,
        current_price: float,
        sweep_price: float,
        entry_direction: str
    ) -> bool:
        """
        Check if price has moved too far in the intended direction.

        If price has already moved beyond the threshold, we've missed the optimal
        entry opportunity and should abandon the setup.

        Args:
            current_price: Current 1H price
            sweep_price: Original liquidity sweep price
            entry_direction: 'long' or 'short'

        Returns:
            True if price moved beyond threshold (setup should be invalidated)
        """
        threshold_percent = self.max_price_deviation_percent / 100

        if entry_direction == 'short':
            # For shorts, invalidate if price dropped below sweep - x%
            threshold_price = sweep_price * (1 - threshold_percent)
            return current_price < threshold_price
        else:  # 'long'
            # For longs, invalidate if price rose above sweep + x%
            threshold_price = sweep_price * (1 + threshold_percent)
            return current_price > threshold_price

    def detect_bos_5m_opposite_direction(
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

        ## for handling debugging purposes for BOS table
        # dymmt_df = self.bos_5m[self.bos_5m['Level'].notna()]
        # print('self.bos_5m: \n', dymmt_df)

        for time_5m in window_indices:
            idx_5m = self.df_5m.index.get_loc(time_5m)
            bos_value = self.bos_5m['BOS'].iloc[idx_5m]

            if not np.isnan(bos_value) and bos_value == target_bos:
                # Found opposite direction BOS
                level = self.bos_5m['Level'].iloc[idx_5m]
                return (time_5m, 'BOS', level)

        return None

    def detect_ifvg_5m(
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

            # print('local_5_fvg: \n', local_5_fvg)
            
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
                    # print('time_5m --> ', time_5m)
                    level = None
                    for j in range(idx_5m - 1, -1, -1):
                        inflexion_type = self.inflexions_5m['InflexionType'].iloc[j]
                        if not np.isnan(inflexion_type) and inflexion_type == target_inflexion_type:
                            level = self.inflexions_5m['Level'].iloc[j]
                            break

                    # # Fallback to FVG midpoint if no inflexion found
                    # if level is None:
                    #     level = (local_5_fvg['Top'].iloc[i] + local_5_fvg['Bottom'].iloc[i]) / 2

                    return (time_5m, 'IFVG', level)

        return None

    def validate_fvg_or_demand_zone_5m(
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

        # Check FVG respect - progressively scan through window (same pattern as detect_ifvg_5m)
        for time_5m in window_indices:
            idx_5m = self.df_5m.index.get_loc(time_5m)

            # Filter FVGs that were mitigated at this specific moment
            local_5m_fvg = self.fvg_5m.loc[self.fvg_5m['MitigatedIndex'] == idx_5m]
            
            # # remove once finished debugging why it doesn't find the FVG in validate event B from 5min
            # if time_5m == datetime(2025, 10, 16, 19, 5):
            #     print('local_5m_fvg: \n', local_5m_fvg)

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

    def detect_final_confirmation_1m(
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

    def get_market_close_for_day(self, timestamp: datetime) -> datetime:
        """
        Get the actual market close time for the trading day containing timestamp.

        Args:
            timestamp: Any timestamp within the trading day

        Returns:
            The last candle timestamp for that trading day (market close time)
        """
        # Extract just the date (ignore time)
        trading_date = timestamp.date()

        # Filter 5M data for candles on this date
        same_day_candles = self.df_5m[self.df_5m.index.date == trading_date]

        if len(same_day_candles) == 0:
            # Fallback: if no data for this day, return end of overlap period
            return self.overlap_end

        # Return the last (maximum) timestamp for this day
        return same_day_candles.index.max()

    def _search_for_event_b_5m(
        self,
        start_5m: datetime,
        end_5m: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float]]:
        """
        Helper method to search for Event B (BOS or IFVG) on 5M timeframe.

        Args:
            start_5m: Start of 5M search window
            end_5m: End of 5M search window
            entry_direction: 'long' or 'short'

        Returns:
            (timestamp, event_type, price) or None if not found
        """
        # Try BOS first
        event_b_candidate = self.detect_bos_5m_opposite_direction(start_5m, end_5m, entry_direction)

        if event_b_candidate is None:
            # Try IFVG
            event_b_candidate = self.detect_ifvg_5m(start_5m, end_5m, entry_direction)

        return event_b_candidate

    def _search_for_validation_5m(
        self,
        start_5m: datetime,
        end_5m: datetime,
        entry_direction: str
    ) -> Optional[Tuple[datetime, str, float, Optional[int]]]:
        """
        Helper method to search for Validation (FVG or Demand Zone) on 5M timeframe.

        Args:
            start_5m: Start of 5M search window
            end_5m: End of 5M search window
            entry_direction: 'long' or 'short'

        Returns:
            (timestamp, zone_type, price, fvg_index) or None if not found
        """
        return self.validate_fvg_or_demand_zone_5m(start_5m, end_5m, entry_direction)

    def _search_for_confirmation_1m(
        self,
        start_1m: datetime,
        end_1m: datetime,
        entry_direction: str,
        time_1h_sweep: datetime,
        sweep_idx: int,
        inflexion_idx: int,
        price_1h_sweep: float,
        price_5m_event_b: float,
        validation_fvg_index: Optional[int]
    ) -> Tuple[Optional[Tuple[datetime, str, float]], bool, bool, bool, datetime]:
        """
        Helper method to search for 1M confirmation with invalidation checks.

        Args:
            start_1m: Start of 1M search window
            end_1m: End of 1M search window
            entry_direction: 'long' or 'short'
            time_1h_sweep: Timestamp of liquidity sweep
            sweep_idx: Index of sweep in 1H data
            inflexion_idx: Index of inflexion point for sweep
            price_1h_sweep: Price at liquidity sweep
            price_5m_event_b: Price at Event B
            validation_fvg_index: FVG index for validation (None if Demand Zone)

        Returns:
            (confirmation_tuple, need_new_event_b, need_new_validation, sweep_broken, last_time_scanned)
        """
        confirmation = None
        need_new_event_b = False
        need_new_validation = False
        sweep_broken = False
        last_time_scanned = start_1m

        # Progressive scan through each 1M candle
        for time_1m_current in self.df_1m.loc[start_1m:end_1m].index:
            last_time_scanned = time_1m_current
            # Check if we've crossed into a new 1H candle
            hours_elapsed = (time_1m_current - time_1h_sweep).total_seconds() / 3600
            current_hour_boundary = hours_elapsed if hours_elapsed % 1 == 0 else int(hours_elapsed) + 1

            if hours_elapsed == current_hour_boundary and hours_elapsed > 0:
                # We've entered a new 1H candle - check if sweep is still valid
                check_time = time_1h_sweep + timedelta(hours=current_hour_boundary)
                current_hour = check_time - timedelta(hours=1)
                current_candle_idx = sweep_idx + int(current_hour_boundary) - 1

                # Check 1: Sweep disrespected?
                if self.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                    print(f"    ⚠️  Sweep broken at {check_time} during 1M scan - abandoning setup")
                    sweep_broken = True
                    break

                # Check 2: Price moved too far?
                current_1h_price = self.df_1h['close'].iloc[current_candle_idx]
                if self.is_price_moved_too_far(current_1h_price, price_1h_sweep, entry_direction):
                    print(f"    ⚠️  Price moved too far during 1M scan ({current_1h_price:.2f} vs sweep {price_1h_sweep:.2f}) - abandoning setup")
                    sweep_broken = True
                    break

            # Check Event B invalidation (check 5M candles)
            current_5m_time = time_1m_current.replace(minute=(time_1m_current.minute // 5) * 5, second=0, microsecond=0)
            if current_5m_time in self.df_5m.index:
                current_5m_close = self.df_5m['close'].loc[current_5m_time]

                # For long: Event B broken if price closes below Event B level
                # For short: Event B broken if price closes above Event B level
                if (entry_direction == 'long' and current_5m_close < price_5m_event_b) or \
                   (entry_direction == 'short' and current_5m_close > price_5m_event_b):
                    print(f"    ⚠️  Event B broken during 1M scan at {current_5m_time} - need new Event B")
                    need_new_event_b = True
                    break

            # Check Validation FVG invalidation (if validation was FVG, not Demand Zone)
            if validation_fvg_index is not None:
                fvg_respected = self.fvg_5m.loc[validation_fvg_index, 'Respected']
                fvg_status_index = self.fvg_5m.loc[validation_fvg_index, 'StatusIndex']

                # Check if FVG was disrespected and status determination happened by now
                if not np.isnan(fvg_status_index):
                    # Map status index to time and check if it's in the past
                    status_time = self.df_5m.index[int(fvg_status_index)]
                    if status_time <= time_1m_current and fvg_respected == False:
                        print(f"    ⚠️  Validation FVG broken at {status_time} - need new validation")
                        need_new_validation = True
                        break

            # Look for confirmation progressively
            confirmation_candidate = self.detect_final_confirmation_1m(
                start_1m,
                time_1m_current,
                entry_direction
            )

            if confirmation_candidate is not None:
                confirmation = confirmation_candidate
                time_1m_confirmation, confirmation_type, price_1m_confirmation = confirmation
                print(f"    ✓ Confirmation ({confirmation_type}) at {time_1m_confirmation}")
                break  # Confirmation found

        return (confirmation, need_new_event_b, need_new_validation, sweep_broken, last_time_scanned)

    def scan_for_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        """
        Scan historical data for complete trade setups.

        Implements invalidation logic:
        - If a 1H candle closes beyond the swept level, sweep is invalidated
        - If a new sweep appears, strategy resets to the new sweep

        Args:
            max_signals: Maximum number of signals to find

        Returns:
            List of TradeSignal objects
        """
        print("🔍 Scanning for trade signals...")
        print(f"  Looking for up to {max_signals} complete setups\n")

        signals = []

        # Example of data overlap period:
        #   Start:    2025-10-01 16:40:00
        #   End:      2025-10-17 21:30:00
        #   Duration: 16 days, 4 hours, 50 minutes
        #
        # Step 1: Detect all liquidity sweeps upfront (same logic as walkthrough)
        #   all_sweeps: List of tuples in the form (candle_idx, sweep_type, swept_level, inflexion_idx)
        #       - candle_idx: Index of the candle where the sweep is detected (e.g., 19 → 20th candle visually)
        #       - sweep_type: Either 'high' or 'low', indicating the direction of the sweep
        #       - swept_level: Price level at which the sweep occurred
        #       - inflexion_idx: Index of the candle marking the origin of the sweep’s inflection point (candle_idx)
        #                        (i.e., where the move leading to the sweep begins)
        #                        (e.g., if candle_idx = 19, inflexion_idx might be 12)
        all_sweeps = self.detect_all_liquidity_sweeps_1h()

        # Track analyzed sweep timestamps to avoid redundant processing
        analyzed_sweep_times = set()

        # Track last analysis end time to ensure temporal consistency
        last_analysis_end_time = None

        # Step 2: Process each sweep in order
        for sweep_idx, sweep_type, swept_level, inflexion_idx in all_sweeps:
            if len(signals) >= max_signals:
                break

            time_1h_sweep = self.df_1h.index[sweep_idx]

            # Skip if we've already analyzed a sweep at this timestamp
            if time_1h_sweep in analyzed_sweep_times:
                print(f"  Skipping duplicate sweep at {time_1h_sweep}")
                continue

            # Skip if sweep time is before or at last analysis end time (temporal consistency)
            if last_analysis_end_time is not None and time_1h_sweep <= last_analysis_end_time:
                print(f"  Skipping sweep at {time_1h_sweep} (overlaps with previous analysis ending at {last_analysis_end_time})")
                continue

            # Mark this timestamp as analyzed
            analyzed_sweep_times.add(time_1h_sweep)

            # Get 1H trend before sweep
            trend_1h = self.get_1h_trend_at_index(sweep_idx)
            if trend_1h is None:
                continue

            price_1h_sweep = swept_level

            print(f"\n  Analyzing sweep at {time_1h_sweep} (1H trend: {trend_1h})")

            # Determine entry direction based on how liquidity was swept
            sweep_candle_close = self.df_1h['close'].iloc[sweep_idx]

            if sweep_candle_close > swept_level:
                # Price closed above swept level → bullish sweep → enter LONG
                entry_direction = 'long'
            else:
                # Price closed below swept level → bearish sweep → enter SHORT
                entry_direction = 'short'

            # Get market close time for this trading day
            market_close = self.get_market_close_for_day(time_1h_sweep)
            market_close_cutoff = market_close - timedelta(minutes=10)

            # Skip sweeps that occur too close to market close (not enough time to develop setup)
            if time_1h_sweep >= market_close_cutoff:
                print(f"    ⚠️  Sweep too close to market close ({market_close}) - skipping")
                continue

            # Step 2 & 3: Progressive 5M scan for Event B → Validation
            # Single loop through 5M candles (state machine approach)
            # Scan until 10 minutes before market close
            start_5m = time_1h_sweep
            end_5m = market_close_cutoff

            # State tracking
            event_b = None
            validation = None
            validation_fvg_index = None  # Track FVG index for validation invalidation
            sweep_broken = False

            # Progressive scan through each 5M candle
            for time_5m_current in self.df_5m.loc[start_5m:end_5m].index:
                # Check if we've crossed into a new 1H candle
                hours_elapsed = (time_5m_current - time_1h_sweep).total_seconds() / 3600
                current_hour_boundary = hours_elapsed if hours_elapsed % 1 == 0 else int(hours_elapsed) + 1

                if hours_elapsed == current_hour_boundary and hours_elapsed > 0:
                    # We've entered a new 1H candle - check if sweep is still valid
                    check_time = time_1h_sweep + timedelta(hours=current_hour_boundary)
                    current_hour = check_time - timedelta(hours=1)
                    current_candle_idx = sweep_idx + int(current_hour_boundary) - 1

                    # Check 1: Sweep disrespected?
                    if self.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                        print(f"    ⚠️  Sweep broken between {current_hour} and {check_time} - abandoning setup")
                        sweep_broken = True
                        break

                    # Check 2: Price moved too far?
                    current_1h_price = self.df_1h['close'].iloc[current_candle_idx]
                    if self.is_price_moved_too_far(current_1h_price, price_1h_sweep, entry_direction):
                        print(f"    ⚠️  Price moved too far ({current_1h_price:.2f} vs sweep {price_1h_sweep:.2f}) between {current_hour} and {check_time}- abandoning setup")
                        sweep_broken = True
                        break

                # Check Event B invalidation (if Event B was found) (only in the IFVG can price_5m_event_b be None - but it should never be the case?)
                if event_b is not None and price_5m_event_b is not None:
                    current_5m_close = self.df_5m['close'].loc[time_5m_current]

                    # For long: Event B broken if price closes below Event B level (new bearish BOS)
                    # For short: Event B broken if price closes above Event B level (new bullish BOS)
                    if (entry_direction == 'long' and current_5m_close < price_5m_event_b) or \
                       (entry_direction == 'short' and current_5m_close > price_5m_event_b):
                        print(f"    ⚠️  Event B broken at {time_5m_current} (price: {current_5m_close:.2f} vs Event B: {price_5m_event_b:.2f}) - resetting")
                        event_b = None
                        validation = None
                        validation_fvg_index = None
                        # Move search window forward to prevent finding the same Event B again
                        start_5m = time_5m_current
                        continue  # Go back to looking for new Event B

                # State 1: Looking for Event B
                if event_b is None:
                    event_b_candidate = self._search_for_event_b_5m(start_5m, time_5m_current, entry_direction)

                    if event_b_candidate is not None:
                        event_b = event_b_candidate
                        time_5m_event_b, event_b_type, price_5m_event_b = event_b
                        print(f"    ✓ Event B ({event_b_type}) at {time_5m_event_b}")

                # State 2: Event B found, looking for Validation
                elif validation is None:
                    validation_candidate = self._search_for_validation_5m(
                        time_5m_event_b,
                        time_5m_current,
                        entry_direction
                    )

                    if validation_candidate is not None:
                        validation = validation_candidate
                        time_5m_validation, validation_type, price_5m_validation, validation_fvg_index = validation
                        print(f"    ✓ Validation ({validation_type}) at {time_5m_validation}")
                        break  # Both Event B and Validation found - move to next step

            # Handle cases where conditions weren't met
            if sweep_broken:
                # Update last analysis end time to when sweep broke
                last_analysis_end_time = time_5m_current if 'time_5m_current' in locals() else end_5m
                continue  # Move to next sweep

            if event_b is None:
                print(f"    ✗ No Event B found on 5M")
                # Save as partial setup
                self.partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_1h_sweep,
                    timestamp_5m_event_b=None,
                    timestamp_5m_validation=None,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_1h_sweep,
                    price_5m_event_b=None,
                    price_5m_validation=None,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_1h,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=None,
                    condition_validation=None,
                    condition_confirmation=None,
                    conditions_met=1,
                    failure_reason="No Event B on 5M",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self.df_1h) - 1, sweep_idx + 5)),
                    indices_5m=None,
                    indices_1m=None,
                    inflexion_idx=inflexion_idx
                ))
                # Update last analysis end time to end of 5M scan window
                last_analysis_end_time = end_5m
                continue

            if validation is None:
                print(f"    ✗ No FVG/Demand Zone validation found")
                time_5m_event_b, event_b_type, price_5m_event_b = event_b
                # Save as partial setup
                self.partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_1h_sweep,
                    timestamp_5m_event_b=time_5m_event_b,
                    timestamp_5m_validation=None,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_1h_sweep,
                    price_5m_event_b=price_5m_event_b,
                    price_5m_validation=None,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_1h,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=event_b_type,
                    condition_validation=None,
                    condition_confirmation=None,
                    conditions_met=2,
                    failure_reason="No FVG/Demand Zone validation",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self.df_1h) - 1, sweep_idx + 5)),
                    indices_5m=(self.df_5m.index.get_loc(time_5m_event_b), self.df_5m.index.get_loc(time_5m_event_b) + 20),
                    indices_1m=None,
                    inflexion_idx=inflexion_idx
                ))
                # Update last analysis end time to end of 5M scan window
                last_analysis_end_time = end_5m
                continue

            # Unpack results (both Event B and Validation found)
            time_5m_event_b, event_b_type, price_5m_event_b = event_b
            time_5m_validation, validation_type, price_5m_validation, validation_fvg_index = validation

            # Step 4: Final confirmation on 1M (progressive scan)
            start_1m = time_5m_validation

            # Use helper method for 1M confirmation search
            confirmation, need_new_event_b, need_new_validation, sweep_broken, last_time_scanned = self._search_for_confirmation_1m(
                start_1m,
                market_close_cutoff,
                entry_direction,
                time_1h_sweep,
                sweep_idx,
                inflexion_idx,
                price_1h_sweep,
                price_5m_event_b,
                validation_fvg_index
            )

            # (Old 1M loop code removed - now using helper method above)

            # Unpack confirmation if found (before checking for breaks/resets)
            if confirmation is not None:
                time_1m_confirmation, confirmation_type, price_1m_confirmation = confirmation

            # Handle sweep broken during 1M scan
            if sweep_broken:
                # Update last analysis end time to where scan actually stopped
                last_analysis_end_time = last_time_scanned
                continue  # Abandon this sweep

            # Handle Event B or Validation broken during 1M scan - go back to 5M scanning
            if need_new_event_b or need_new_validation:
                print(f"    ⏮️  Going back to 5M scanning")

                # Reset state based on what broke
                if need_new_event_b:
                    # Event B broken - reset both Event B and Validation
                    event_b = None
                    validation = None
                    validation_fvg_index = None
                    # Continue 5M scan from where validation occurred (when we entered 1M scan)
                    remaining_5m_start = time_5m_validation
                else:
                    # Validation broken - keep Event B, reset Validation
                    validation = None
                    validation_fvg_index = None
                    # Continue 5M scan from where validation was
                    remaining_5m_start = time_5m_validation

                # Re-search for Event B or Validation on remaining 5M window
                if need_new_event_b:
                    # First, check if sweep is still valid in the re-scan window
                    # Calculate hour boundaries that fall within remaining_5m_start to end_5m
                    start_hours_elapsed = (remaining_5m_start - time_1h_sweep).total_seconds() / 3600
                    end_hours_elapsed = (end_5m - time_1h_sweep).total_seconds() / 3600

                    # Check each hour boundary in this range
                    first_boundary = int(start_hours_elapsed) + 1
                    last_boundary = int(end_hours_elapsed) + 1

                    for boundary_hour in range(first_boundary, last_boundary + 1):
                        check_time = time_1h_sweep + timedelta(hours=boundary_hour)

                        # Skip if check_time is beyond our window
                        if check_time > end_5m:
                            break

                        current_candle_idx = sweep_idx + boundary_hour - 1

                        # Check 1: Sweep disrespected?
                        if self.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                            print(f"    ⚠️  Sweep broken at {check_time} during 5M re-scan - abandoning setup")
                            sweep_broken = True
                            break

                        # Check 2: Price moved too far?
                        current_1h_price = self.df_1h['close'].iloc[current_candle_idx]
                        if self.is_price_moved_too_far(current_1h_price, price_1h_sweep, entry_direction):
                            print(f"    ⚠️  Price moved too far ({current_1h_price:.2f} vs sweep {price_1h_sweep:.2f}) at {check_time} during 5M re-scan - abandoning setup")
                            sweep_broken = True
                            break

                    # If sweep was broken, abandon this sweep
                    if sweep_broken:
                        continue

                    # Search for new Event B
                    event_b_resume = self._search_for_event_b_5m(remaining_5m_start, end_5m, entry_direction)

                    if event_b_resume is not None:
                        event_b = event_b_resume
                        time_5m_event_b, event_b_type, price_5m_event_b = event_b
                        print(f"    ✓ Found new Event B ({event_b_type}) at {time_5m_event_b}")

                        # Now search for validation from this new Event B
                        validation_resume = self._search_for_validation_5m(time_5m_event_b, end_5m, entry_direction)

                        if validation_resume is not None:
                            validation = validation_resume
                            time_5m_validation, validation_type, price_5m_validation, validation_fvg_index = validation
                            print(f"    ✓ Found validation ({validation_type}) at {time_5m_validation}")
                            # Continue to 1M scanning (don't use continue statement, let it fall through)
                        else:
                            print(f"    ✗ No validation found after new Event B")
                            continue  # Abandon this sweep
                    else:
                        print(f"    ✗ No new Event B found - abandoning sweep")
                        continue  # Abandon this sweep

                else:  # need_new_validation
                    # First, check if sweep is still valid in the re-scan window
                    # Calculate hour boundaries that fall within time_5m_event_b to end_5m
                    start_hours_elapsed = (time_5m_event_b - time_1h_sweep).total_seconds() / 3600
                    end_hours_elapsed = (end_5m - time_1h_sweep).total_seconds() / 3600

                    # Check each hour boundary in this range
                    first_boundary = int(start_hours_elapsed) + 1
                    last_boundary = int(end_hours_elapsed) + 1

                    for boundary_hour in range(first_boundary, last_boundary + 1):
                        check_time = time_1h_sweep + timedelta(hours=boundary_hour)

                        # Skip if check_time is beyond our window
                        if check_time > end_5m:
                            break

                        current_candle_idx = sweep_idx + boundary_hour - 1

                        # Check 1: Sweep disrespected?
                        if self.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                            print(f"    ⚠️  Sweep broken at {check_time} during Validation re-scan - abandoning setup")
                            sweep_broken = True
                            break

                        # Check 2: Price moved too far?
                        current_1h_price = self.df_1h['close'].iloc[current_candle_idx]
                        if self.is_price_moved_too_far(current_1h_price, price_1h_sweep, entry_direction):
                            print(f"    ⚠️  Price moved too far ({current_1h_price:.2f} vs sweep {price_1h_sweep:.2f}) at {check_time} during Validation re-scan - abandoning setup")
                            sweep_broken = True
                            break

                    # If sweep was broken, abandon this sweep
                    if sweep_broken:
                        continue

                    # Search for new Validation (Event B is still valid)
                    validation_resume = self._search_for_validation_5m(time_5m_event_b, end_5m, entry_direction)

                    if validation_resume is not None:
                        validation = validation_resume
                        time_5m_validation, validation_type, price_5m_validation, validation_fvg_index = validation
                        print(f"    ✓ Found new validation ({validation_type}) at {time_5m_validation}")
                        # Continue to 1M scanning (don't use continue statement, let it fall through)
                    else:
                        print(f"    ✗ No new validation found - abandoning sweep")
                        continue  # Abandon this sweep

                # If we got here, we found new Event B/Validation - now try 1M confirmation again
                print(f"    ▶️  Resuming 1M scan from {time_5m_validation}")
                start_1m = time_5m_validation

                # Use helper method to search for confirmation with new Event B/Validation
                confirmation_resume, need_new_event_b_again, need_new_validation_again, sweep_broken_resume, last_time_scanned_resume = self._search_for_confirmation_1m(
                    start_1m,
                    market_close_cutoff,
                    entry_direction,
                    time_1h_sweep,
                    sweep_idx,
                    inflexion_idx,
                    price_1h_sweep,
                    price_5m_event_b,
                    validation_fvg_index
                )

                # Handle results from resumed scan
                if sweep_broken_resume:
                    print(f"    ⚠️  Sweep broken during resumed 1M scan - abandoning")
                    last_analysis_end_time = last_time_scanned_resume
                    continue
                elif need_new_event_b_again or need_new_validation_again:
                    print(f"    ⚠️  Event B/Validation broken again during resumed scan - abandoning for now")
                    # Could implement nested loop here, but for now abandon
                    continue
                elif confirmation_resume is not None:
                    # Success! Use the resumed confirmation
                    confirmation = confirmation_resume
                    time_1m_confirmation, confirmation_type, price_1m_confirmation = confirmation
                    print(f"    ✅ Found confirmation after reset!")
                    # Fall through to signal generation below
                else:
                    print(f"    ✗ No confirmation found after reset")
                    continue

            # Handle no confirmation found
            if confirmation is None:
                print(f"    ✗ No 1M confirmation found")
                # Save as partial setup
                self.partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_1h_sweep,
                    timestamp_5m_event_b=time_5m_event_b,
                    timestamp_5m_validation=time_5m_validation,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_1h_sweep,
                    price_5m_event_b=price_5m_event_b,
                    price_5m_validation=price_5m_validation,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_1h,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=event_b_type,
                    condition_validation=validation_type,
                    condition_confirmation=None,
                    conditions_met=3,
                    failure_reason="No 1M confirmation",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self.df_1h) - 1, sweep_idx + 5)),
                    indices_5m=(self.df_5m.index.get_loc(time_5m_event_b), self.df_5m.index.get_loc(time_5m_validation)),
                    indices_1m=(self.df_1m.index.get_loc(start_1m), min(self.df_1m.index.get_loc(start_1m) + 30, len(self.df_1m) - 1)),
                    inflexion_idx=inflexion_idx
                ))
                # Update last analysis end time to where scan actually ended
                last_analysis_end_time = last_time_scanned
                continue

            # Confirmation found (time_1m_confirmation, confirmation_type, price_1m_confirmation already unpacked)

            # All conditions met - generate signal
            signal = TradeSignal(
                timestamp_1h_sweep=time_1h_sweep,
                timestamp_5m_event_b=time_5m_event_b,
                timestamp_5m_validation=time_5m_validation,
                timestamp_1m_confirmation=time_1m_confirmation,
                timestamp_entry=time_1m_confirmation,  # Enter on confirmation
                trend_1h_before_sweep=trend_1h,
                entry_direction=entry_direction,
                price_1h_sweep=price_1h_sweep,
                price_5m_event_b=price_5m_event_b,
                price_5m_validation=price_5m_validation,
                price_1m_confirmation=price_1m_confirmation,
                price_entry=price_1m_confirmation,
                condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                condition_event_b=event_b_type,
                condition_validation=validation_type,
                condition_confirmation=confirmation_type,
                indices_1h=(max(0, sweep_idx - 5), min(len(self.df_1h) - 1, sweep_idx + 5)),
                indices_5m=(
                    self.df_5m.index.get_loc(time_5m_event_b),
                    self.df_5m.index.get_loc(time_5m_validation)
                ),
                indices_1m=(
                    self.df_1m.index.get_loc(start_1m),
                    self.df_1m.index.get_loc(time_1m_confirmation)
                )
            )

            signals.append(signal)
            print(f"    ✅ COMPLETE SIGNAL #{len(signals)} - {entry_direction.upper()} entry")

            # Update last analysis end time to confirmation time
            last_analysis_end_time = time_1m_confirmation

        print(f"\n✅ Found {len(signals)} complete trade signals!")
        print(f"📊 Found {len(self.partial_setups)} partial setups:")
        for i, partial in enumerate(self.partial_setups, 1):
            print(f"   #{i}: {partial.conditions_met}/4 conditions - {partial.failure_reason}")

        self.signals = signals
        return signals
