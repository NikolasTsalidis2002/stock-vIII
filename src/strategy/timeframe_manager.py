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
    - Store and index DataFrames for high, mid, low timeframes
    - Determine overlapping periods
    - Create timeframe mappings (high->mid->low)
    - Pre-calculate all indicators
    """

    def __init__(
        self,
        df_high: pd.DataFrame,
        df_mid: pd.DataFrame,
        df_low: pd.DataFrame,
        timeframe_config: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize with multi-timeframe data.

        Args:
            df_high: High timeframe OHLCV data with 'time' column (e.g., 1H or 4H)
            df_mid: Mid timeframe OHLCV data with 'time' column (e.g., 5M or 15M)
            df_low: Low timeframe OHLCV data with 'time' column (e.g., 1M or 5M)
            timeframe_config: Optional dict with timeframe labels for display
                              e.g., {'high': '4h', 'mid': '15min', 'low': '5min'}
        """
        # Store timeframe config for display purposes
        self.timeframe_config = timeframe_config or {'high': 'HIGH', 'mid': 'MID', 'low': 'LOW'}

        # Store original data with datetime index (for backtester)
        if 'time' in df_high.columns:
            self.df_high_original = df_high.set_index('time').copy()
            self.df_high_original.index = pd.to_datetime(self.df_high_original.index)
        else:
            self.df_high_original = df_high.copy()

        if 'time' in df_mid.columns:
            self.df_mid_original = df_mid.set_index('time').copy()
            self.df_mid_original.index = pd.to_datetime(self.df_mid_original.index)
        else:
            self.df_mid_original = df_mid.copy()

        if 'time' in df_low.columns:
            self.df_low_original = df_low.set_index('time').copy()
            self.df_low_original.index = pd.to_datetime(self.df_low_original.index)
        else:
            self.df_low_original = df_low.copy()

        # Prepare indexed versions (will be filtered to overlap period)
        self._df_high = self.df_high_original.copy()
        self._df_mid = self.df_mid_original.copy()
        self._df_low = self.df_low_original.copy()

        # Overlapping period
        self._overlap_start: Optional[datetime] = None
        self._overlap_end: Optional[datetime] = None

        # Timeframe mappings
        self.map_high_to_mid: Dict[int, List[datetime]] = {}
        self.map_mid_to_low: Dict[int, List[datetime]] = {}

        # Initialize
        self._determine_overlapping_period()
        self._calculate_timeframe_durations()
        self._create_timeframe_mapping()
        self._calculate_indicators()

    def _calculate_timeframe_durations(self) -> None:
        """Calculate the duration of each timeframe from the data."""
        # Calculate duration by looking at median gap between consecutive candles
        if len(self._df_high) > 1:
            high_gaps = pd.Series(self._df_high.index).diff().dropna()
            self._high_duration = high_gaps.median()
        else:
            self._high_duration = timedelta(hours=1)  # fallback

        if len(self._df_mid) > 1:
            mid_gaps = pd.Series(self._df_mid.index).diff().dropna()
            self._mid_duration = mid_gaps.median()
        else:
            self._mid_duration = timedelta(minutes=5)  # fallback

        if len(self._df_low) > 1:
            low_gaps = pd.Series(self._df_low.index).diff().dropna()
            self._low_duration = low_gaps.median()
        else:
            self._low_duration = timedelta(minutes=1)  # fallback

        high_label = self.timeframe_config.get('high', 'HIGH')
        mid_label = self.timeframe_config.get('mid', 'MID')
        low_label = self.timeframe_config.get('low', 'LOW')

        print(f"\n⏱️  Detected timeframe durations:")
        print(f"  {high_label}: {self._high_duration}")
        print(f"  {mid_label}: {self._mid_duration}")
        print(f"  {low_label}: {self._low_duration}")

    def _determine_overlapping_period(self) -> None:
        """
        Determine the overlapping time period across all timeframes.

        Only analyze the period where all three timeframes have data.
        """
        high_label = self.timeframe_config.get('high', 'HIGH')
        mid_label = self.timeframe_config.get('mid', 'MID')
        low_label = self.timeframe_config.get('low', 'LOW')

        # Find latest start time
        start_high = self._df_high.index.min()
        start_mid = self._df_mid.index.min()
        start_low = self._df_low.index.min()
        self._overlap_start = max(start_high, start_mid, start_low)

        # Find earliest end time
        end_high = self._df_high.index.max()
        end_mid = self._df_mid.index.max()
        end_low = self._df_low.index.max()
        self._overlap_end = min(end_high, end_mid, end_low)

        print(f"\n⏱️  Data overlap period:")
        print(f"  Start: {self._overlap_start}")
        print(f"  End:   {self._overlap_end}")
        print(f"  Duration: {self._overlap_end - self._overlap_start}")

        # Filter dataframes to overlapping period
        self._df_high = self._df_high.loc[self._overlap_start:self._overlap_end]
        self._df_mid = self._df_mid.loc[self._overlap_start:self._overlap_end]
        self._df_low = self._df_low.loc[self._overlap_start:self._overlap_end]

        print(f"\n  Filtered data:")
        print(f"  {high_label}: {len(self._df_high)} candles")
        print(f"  {mid_label}: {len(self._df_mid)} candles")
        print(f"  {low_label}: {len(self._df_low)} candles")

    def _create_timeframe_mapping(self) -> None:
        """
        Create mapping between timeframes.

        For each high TF candle, determine which mid TF candles fall within it.
        For each mid TF candle, determine which low TF candles fall within it.
        """
        high_label = self.timeframe_config.get('high', 'HIGH')
        mid_label = self.timeframe_config.get('mid', 'MID')
        low_label = self.timeframe_config.get('low', 'LOW')

        print("📊 Creating timeframe mapping...")

        # Map high → mid
        self.map_high_to_mid = {}

        for i, time_high in enumerate(self._df_high.index):
            # High candle covers [time_high, time_high + high_duration)
            end_time = time_high + self._high_duration

            # Find all mid candles in this range
            mask_mid = (self._df_mid.index >= time_high) & (self._df_mid.index < end_time)
            indices_mid = self._df_mid.index[mask_mid].tolist()

            self.map_high_to_mid[i] = indices_mid

        # Map mid → low
        self.map_mid_to_low = {}

        for i, time_mid in enumerate(self._df_mid.index):
            # Mid candle covers [time_mid, time_mid + mid_duration)
            end_time = time_mid + self._mid_duration

            # Find all low candles in this range
            mask_low = (self._df_low.index >= time_mid) & (self._df_low.index < end_time)
            indices_low = self._df_low.index[mask_low].tolist()

            self.map_mid_to_low[i] = indices_low

        print(f"  ✓ Mapped {len(self._df_high)} {high_label} candles")
        print(f"  ✓ Mapped {len(self._df_mid)} {mid_label} candles")
        print(f"  ✓ Mapped {len(self._df_low)} {low_label} candles")

    def _calculate_indicators(self) -> None:
        """Pre-calculate all indicators for each timeframe."""
        high_label = self.timeframe_config.get('high', 'HIGH')
        mid_label = self.timeframe_config.get('mid', 'MID')
        low_label = self.timeframe_config.get('low', 'LOW')

        print("\n📈 Calculating indicators...")

        # High TF Indicators (using proper liquidity detection)
        print(f"  [{high_label}] Calculating indicators...")
        self.swings_high = smc.swing_highs_lows(self._df_high, swing_length=3)
        self.liquidity_high = smc.liquidity(self._df_high, self.swings_high, range_percent=0.01)
        self.inflexions_high = smc_custom.inflexion_points(self._df_high)
        self.bos_high = smc_custom.bos(self._df_high, self.inflexions_high, close_break=True)

        # Mid TF Indicators
        print(f"  [{mid_label}] Calculating indicators...")
        self.swings_mid = smc.swing_highs_lows(self._df_mid, swing_length=30)
        self.inflexions_mid = smc_custom.inflexion_points(self._df_mid)
        self.bos_mid = smc_custom.bos(self._df_mid, self.inflexions_mid, close_break=True)
        self.fvg_mid = smc.fvg(self._df_mid, join_consecutive=False)
        self.ob_mid = smc_custom.ob(self._df_mid, self.bos_mid, self.inflexions_mid)

        # Low TF Indicators
        print(f"  [{low_label}] Calculating indicators...")
        self.inflexions_low = smc_custom.inflexion_points(self._df_low)
        self.bos_low = smc_custom.bos(self._df_low, self.inflexions_low, close_break=True)
        self.fvg_low = smc.fvg(self._df_low, join_consecutive=False)

        print("  ✓ All indicators calculated\n")

    def get_high_trend_at_index(self, idx: int) -> Optional[str]:
        """
        Get the trend at a specific high TF index.

        Args:
            idx: Index in high TF dataframe

        Returns:
            'bullish', 'bearish', or None if undefined
        """
        trend_value = self.bos_high['Trend'].iloc[idx]

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

        # Filter mid TF data for candles on this date
        same_day_candles = self._df_mid[self._df_mid.index.date == trading_date]

        if len(same_day_candles) == 0:
            # Fallback: if no data for this day, return end of overlap period
            return self._overlap_end

        # Return the last (maximum) timestamp for this day
        return same_day_candles.index.max()

    @property
    def high_duration(self) -> timedelta:
        """Duration of high timeframe candles."""
        return self._high_duration

    @property
    def mid_duration(self) -> timedelta:
        """Duration of mid timeframe candles."""
        return self._mid_duration

    @property
    def low_duration(self) -> timedelta:
        """Duration of low timeframe candles."""
        return self._low_duration

    @property
    def df_high(self) -> pd.DataFrame:
        """High TF DataFrame with datetime index."""
        return self._df_high

    @property
    def df_mid(self) -> pd.DataFrame:
        """Mid TF DataFrame with datetime index."""
        return self._df_mid

    @property
    def df_low(self) -> pd.DataFrame:
        """Low TF DataFrame with datetime index."""
        return self._df_low

    @property
    def overlap_start(self) -> datetime:
        """Start of overlapping period."""
        return self._overlap_start

    @property
    def overlap_end(self) -> datetime:
        """End of overlapping period."""
        return self._overlap_end
