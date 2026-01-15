"""
Multi-timeframe data management and indicator calculation.

Handles timeframe alignment, mapping between candles, and pre-calculation
of all SMC indicators.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from indicators.smc import smc
from indicators.smc_custom import smc_custom


class TimeframeManager:
    """
    Manages multi-timeframe data alignment and indicator pre-calculation.

    Responsibilities:
    - Store and index DataFrames for 1H, 5M, 1M
    - Determine overlapping periods
    - Create timeframe mappings (1H->5M->1M)
    - Pre-calculate all indicators
    """

    def __init__(
        self,
        df_1h: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_1m: pd.DataFrame
    ) -> None:
        """
        Initialize with multi-timeframe data.

        Args:
            df_1h: 1-hour OHLCV data with 'time' column
            df_5m: 5-minute OHLCV data with 'time' column
            df_1m: 1-minute OHLCV data with 'time' column
        """
        # Store original data
        self.df_1h_original = df_1h.copy()
        self.df_5m_original = df_5m.copy()
        self.df_1m_original = df_1m.copy()

        # Prepare indexed versions
        if 'time' in df_1h.columns:
            self._df_1h = df_1h.set_index('time')
        else:
            self._df_1h = df_1h.copy()

        if 'time' in df_5m.columns:
            self._df_5m = df_5m.set_index('time')
        else:
            self._df_5m = df_5m.copy()

        if 'time' in df_1m.columns:
            self._df_1m = df_1m.set_index('time')
        else:
            self._df_1m = df_1m.copy()

        # Ensure datetime index
        self._df_1h.index = pd.to_datetime(self._df_1h.index)
        self._df_5m.index = pd.to_datetime(self._df_5m.index)
        self._df_1m.index = pd.to_datetime(self._df_1m.index)

        # Overlapping period
        self._overlap_start: Optional[datetime] = None
        self._overlap_end: Optional[datetime] = None

        # Timeframe mappings
        self.map_1h_to_5m: Dict[int, List[datetime]] = {}
        self.map_5m_to_1m: Dict[int, List[datetime]] = {}

        # Initialize
        self._determine_overlapping_period()
        self._create_timeframe_mapping()
        self._calculate_indicators()

    def _determine_overlapping_period(self) -> None:
        """
        Determine the overlapping time period across all timeframes.

        Only analyze the period where all three timeframes have data.
        """
        # Find latest start time
        start_1h = self._df_1h.index.min()
        start_5m = self._df_5m.index.min()
        start_1m = self._df_1m.index.min()
        self._overlap_start = max(start_1h, start_5m, start_1m)

        # Find earliest end time
        end_1h = self._df_1h.index.max()
        end_5m = self._df_5m.index.max()
        end_1m = self._df_1m.index.max()
        self._overlap_end = min(end_1h, end_5m, end_1m)

        print(f"\n⏱️  Data overlap period:")
        print(f"  Start: {self._overlap_start}")
        print(f"  End:   {self._overlap_end}")
        print(f"  Duration: {self._overlap_end - self._overlap_start}")

        # Filter dataframes to overlapping period
        self._df_1h = self._df_1h.loc[self._overlap_start:self._overlap_end]
        self._df_5m = self._df_5m.loc[self._overlap_start:self._overlap_end]
        self._df_1m = self._df_1m.loc[self._overlap_start:self._overlap_end]

        print(f"\n  Filtered data:")
        print(f"  1H: {len(self._df_1h)} candles")
        print(f"  5M: {len(self._df_5m)} candles")
        print(f"  1M: {len(self._df_1m)} candles")

    def _create_timeframe_mapping(self) -> None:
        """
        Create mapping between timeframes.

        For each 1H candle, determine which 5M and 1M candles fall within it.
        For each 5M candle, determine which 1M candles fall within it.
        """
        print("📊 Creating timeframe mapping...")

        # Map 1H → 5M
        self.map_1h_to_5m = {}

        for i, time_1h in enumerate(self._df_1h.index):
            # 1H candle covers [time_1h, time_1h + 1 hour)
            end_time = time_1h + timedelta(hours=1)

            # Find all 5M candles in this range
            mask_5m = (self._df_5m.index >= time_1h) & (self._df_5m.index < end_time)
            indices_5m = self._df_5m.index[mask_5m].tolist()

            self.map_1h_to_5m[i] = indices_5m

        # Map 5M → 1M
        self.map_5m_to_1m = {}

        for i, time_5m in enumerate(self._df_5m.index):
            # 5M candle covers [time_5m, time_5m + 5 minutes)
            end_time = time_5m + timedelta(minutes=5)

            # Find all 1M candles in this range
            mask_1m = (self._df_1m.index >= time_5m) & (self._df_1m.index < end_time)
            indices_1m = self._df_1m.index[mask_1m].tolist()

            self.map_5m_to_1m[i] = indices_1m

        print(f"  ✓ Mapped {len(self._df_1h)} 1H candles")
        print(f"  ✓ Mapped {len(self._df_5m)} 5M candles")
        print(f"  ✓ Mapped {len(self._df_1m)} 1M candles")

    def _calculate_indicators(self) -> None:
        """Pre-calculate all indicators for each timeframe."""
        print("\n📈 Calculating indicators...")

        # 1H Indicators (using proper liquidity detection)
        print("  [1H] Calculating indicators...")
        self.swings_1h = smc.swing_highs_lows(self._df_1h, swing_length=3)
        self.liquidity_1h = smc.liquidity(self._df_1h, self.swings_1h, range_percent=0.01)
        self.inflexions_1h = smc_custom.inflexion_points(self._df_1h)
        self.bos_1h = smc_custom.bos(self._df_1h, self.inflexions_1h, close_break=True)

        # 5M Indicators
        print("  [5M] Calculating indicators...")
        self.swings_5m = smc.swing_highs_lows(self._df_5m, swing_length=30)
        self.inflexions_5m = smc_custom.inflexion_points(self._df_5m)
        self.bos_5m = smc_custom.bos(self._df_5m, self.inflexions_5m, close_break=True)
        self.fvg_5m = smc.fvg(self._df_5m, join_consecutive=True)
        self.ob_5m = smc_custom.ob(self._df_5m, self.bos_5m, self.inflexions_5m)

        # 1M Indicators
        print("  [1M] Calculating indicators...")
        self.inflexions_1m = smc_custom.inflexion_points(self._df_1m)
        self.bos_1m = smc_custom.bos(self._df_1m, self.inflexions_1m, close_break=True)
        self.fvg_1m = smc.fvg(self._df_1m, join_consecutive=True)

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
        same_day_candles = self._df_5m[self._df_5m.index.date == trading_date]

        if len(same_day_candles) == 0:
            # Fallback: if no data for this day, return end of overlap period
            return self._overlap_end

        # Return the last (maximum) timestamp for this day
        return same_day_candles.index.max()

    @property
    def df_1h(self) -> pd.DataFrame:
        """1H DataFrame with datetime index."""
        return self._df_1h

    @property
    def df_5m(self) -> pd.DataFrame:
        """5M DataFrame with datetime index."""
        return self._df_5m

    @property
    def df_1m(self) -> pd.DataFrame:
        """1M DataFrame with datetime index."""
        return self._df_1m

    @property
    def overlap_start(self) -> datetime:
        """Start of overlapping period."""
        return self._overlap_start

    @property
    def overlap_end(self) -> datetime:
        """End of overlapping period."""
        return self._overlap_end
