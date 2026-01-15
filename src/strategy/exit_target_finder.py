"""
Exit target detection for Strategy II.

Finds the Order Block at the origin of the liquidity move for take profit calculation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

from .models import ExitTarget


class ExitTargetFinder:
    """
    Finds Order Block at the origin of liquidity move.

    Logic:
    1. Given liquidity sweep at time X
    2. Find the BOS that ended the previous trend and started the current trend
    3. At that BOS origin, find the Order Block (opposite color candles + first same color)
    4. Return OB levels for take profit calculation
    """

    def __init__(
        self,
        df_5m: pd.DataFrame,
        bos_5m: pd.DataFrame,
        inflexions_5m: pd.DataFrame
    ) -> None:
        """
        Initialize exit target finder.

        Args:
            df_5m: 5M OHLCV DataFrame with datetime index
            bos_5m: Pre-calculated BOS data with Trend column
            inflexions_5m: Pre-calculated inflexion points
        """
        self.df_5m = df_5m
        self.bos_5m = bos_5m
        self.inflexions_5m = inflexions_5m

    def find_exit_order_block(
        self,
        sweep_time: datetime,
        entry_direction: str
    ) -> Optional[ExitTarget]:
        """
        Find Order Block at origin of liquidity move.

        For SHORT (swept high):
        1. High was created by upward trend
        2. Find BOS that started the upward trend (ended previous downtrend)
        3. Find lowest point of that previous downtrend
        4. OB = last RED candles + first GREEN at that minimum
        5. TP = Top of OB

        For LONG (swept low):
        1. Low was created by downward trend
        2. Find BOS that started the downward trend (ended previous uptrend)
        3. Find highest point of that previous uptrend
        4. OB = last GREEN candles + first RED at that maximum
        5. TP = Bottom of OB

        Args:
            sweep_time: Time of liquidity sweep
            entry_direction: 'long' or 'short'

        Returns:
            ExitTarget with OB levels and take profit, or None
        """
        # Get sweep index in 5M data
        if sweep_time not in self.df_5m.index:
            # Find nearest 5M time at or before sweep
            mask = self.df_5m.index <= sweep_time
            if not mask.any():
                return None
            sweep_time = self.df_5m.index[mask][-1]

        sweep_idx = self.df_5m.index.get_loc(sweep_time)

        if entry_direction == 'short':
            # For SHORT: swept a HIGH, meaning there was an upward trend
            # Find the BOS that started the upward trend (bullish BOS)
            return self._find_exit_ob_for_short(sweep_idx)
        else:
            # For LONG: swept a LOW, meaning there was a downward trend
            # Find the BOS that started the downward trend (bearish BOS)
            return self._find_exit_ob_for_long(sweep_idx)

    def _find_exit_ob_for_short(self, sweep_idx: int) -> Optional[ExitTarget]:
        """
        Find exit OB for SHORT entry.

        Search backwards from sweep for the bullish BOS that started the uptrend.
        Then find the Order Block at the origin (last RED candles + first GREEN).
        """
        # Search backwards for bullish BOS (where uptrend started)
        for i in range(sweep_idx - 1, -1, -1):
            bos_val = self.bos_5m['BOS'].iloc[i]

            if bos_val == 1:  # Bullish BOS = uptrend started
                # Get TrendStartIndex if available
                trend_start = int(self.bos_5m['TrendStartIndex'].iloc[i])

                # Find the lowest point BEFORE the bullish BOS
                # This is where the previous downtrend bottomed
                search_start = max(0, trend_start - 50)  # Look back up to 50 candles
                search_end = i

                if search_start >= search_end:
                    continue

                # Find minimum in the range before the BOS
                lows_in_range = self.df_5m['low'].iloc[search_start:search_end]
                if len(lows_in_range) == 0:
                    continue

                min_idx = lows_in_range.idxmin()
                min_loc = self.df_5m.index.get_loc(min_idx)

                # Find Order Block at minimum: last RED candles + first GREEN
                ob = self._find_order_block_at_minimum(min_loc)

                if ob is not None:
                    ob_top, ob_bottom, ob_start, ob_end = ob
                    return ExitTarget(
                        ob_top=ob_top,
                        ob_bottom=ob_bottom,
                        ob_start_idx=ob_start,
                        ob_end_idx=ob_end,
                        take_profit=ob_top,  # For SHORT: TP at top of OB
                        bos_idx=i,
                        ob_start_time=self.df_5m.index[ob_start],
                        ob_end_time=self.df_5m.index[ob_end]
                    )

        return None

    def _find_exit_ob_for_long(self, sweep_idx: int) -> Optional[ExitTarget]:
        """
        Find exit OB for LONG entry.

        Search backwards from sweep for the bearish BOS that started the downtrend.
        Then find the Order Block at the origin (last GREEN candles + first RED).
        """
        # Search backwards for bearish BOS (where downtrend started)
        for i in range(sweep_idx - 1, -1, -1):
            bos_val = self.bos_5m['BOS'].iloc[i]

            if bos_val == -1:  # Bearish BOS = downtrend started
                # Get TrendStartIndex if available
                trend_start = int(self.bos_5m['TrendStartIndex'].iloc[i])

                # Find the highest point BEFORE the bearish BOS
                # This is where the previous uptrend topped
                search_start = max(0, trend_start - 50)  # Look back up to 50 candles
                search_end = i

                if search_start >= search_end:
                    continue

                # Find maximum in the range before the BOS
                highs_in_range = self.df_5m['high'].iloc[search_start:search_end]
                if len(highs_in_range) == 0:
                    continue

                max_idx = highs_in_range.idxmax()
                max_loc = self.df_5m.index.get_loc(max_idx)

                # Find Order Block at maximum: last GREEN candles + first RED
                ob = self._find_order_block_at_maximum(max_loc)

                if ob is not None:
                    ob_top, ob_bottom, ob_start, ob_end = ob
                    return ExitTarget(
                        ob_top=ob_top,
                        ob_bottom=ob_bottom,
                        ob_start_idx=ob_start,
                        ob_end_idx=ob_end,
                        take_profit=ob_bottom,  # For LONG: TP at bottom of OB
                        bos_idx=i,
                        ob_start_time=self.df_5m.index[ob_start],
                        ob_end_time=self.df_5m.index[ob_end]
                    )

        return None

    def _find_order_block_at_minimum(
        self,
        min_idx: int
    ) -> Optional[Tuple[float, float, int, int]]:
        """
        Find Order Block at a local minimum.

        For SHORT exit target: Find last RED candles + first GREEN at minimum.
        The OB represents the reversal zone where price bounced.

        Returns:
            (ob_top, ob_bottom, start_idx, end_idx) or None
        """
        # Look for red candles leading to minimum, then first green candle after
        search_start = max(0, min_idx - 20)
        search_end = min(len(self.df_5m) - 1, min_idx + 10)

        # Find red candles before minimum
        red_candle_indices = []
        for i in range(min_idx, search_start - 1, -1):
            candle = self.df_5m.iloc[i]
            if candle['close'] < candle['open']:  # Red candle
                red_candle_indices.append(i)
            elif len(red_candle_indices) > 0:
                # Found non-red candle after finding red ones - stop
                break

        if len(red_candle_indices) == 0:
            return None

        # Find first green candle after the red group
        green_idx = None
        for i in range(min_idx + 1, search_end + 1):
            candle = self.df_5m.iloc[i]
            if candle['close'] > candle['open']:  # Green candle
                green_idx = i
                break

        if green_idx is None:
            # No green candle found, use last red candle range
            ob_start = min(red_candle_indices)
            ob_end = max(red_candle_indices)
        else:
            # OB = red candles + first green
            ob_start = min(red_candle_indices)
            ob_end = green_idx

        # Calculate OB bounds
        ob_candles = self.df_5m.iloc[ob_start:ob_end + 1]
        ob_top = ob_candles['high'].max()
        ob_bottom = ob_candles['low'].min()

        return (ob_top, ob_bottom, ob_start, ob_end)

    def _find_order_block_at_maximum(
        self,
        max_idx: int
    ) -> Optional[Tuple[float, float, int, int]]:
        """
        Find Order Block at a local maximum.

        For LONG exit target: Find last GREEN candles + first RED at maximum.
        The OB represents the reversal zone where price rejected.

        Returns:
            (ob_top, ob_bottom, start_idx, end_idx) or None
        """
        # Look for green candles leading to maximum, then first red candle after
        search_start = max(0, max_idx - 20)
        search_end = min(len(self.df_5m) - 1, max_idx + 10)

        # Find green candles before maximum
        green_candle_indices = []
        for i in range(max_idx, search_start - 1, -1):
            candle = self.df_5m.iloc[i]
            if candle['close'] > candle['open']:  # Green candle
                green_candle_indices.append(i)
            elif len(green_candle_indices) > 0:
                # Found non-green candle after finding green ones - stop
                break

        if len(green_candle_indices) == 0:
            return None

        # Find first red candle after the green group
        red_idx = None
        for i in range(max_idx + 1, search_end + 1):
            candle = self.df_5m.iloc[i]
            if candle['close'] < candle['open']:  # Red candle
                red_idx = i
                break

        if red_idx is None:
            # No red candle found, use last green candle range
            ob_start = min(green_candle_indices)
            ob_end = max(green_candle_indices)
        else:
            # OB = green candles + first red
            ob_start = min(green_candle_indices)
            ob_end = red_idx

        # Calculate OB bounds
        ob_candles = self.df_5m.iloc[ob_start:ob_end + 1]
        ob_top = ob_candles['high'].max()
        ob_bottom = ob_candles['low'].min()

        return (ob_top, ob_bottom, ob_start, ob_end)
