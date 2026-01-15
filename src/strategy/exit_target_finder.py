"""
Exit target detection for Strategy II.

Finds the Order Block at the origin of the liquidity move for take profit calculation.
"""

import pandas as pd
from datetime import datetime
from typing import Optional

from .models import ExitTarget


class ExitTargetFinder:
    """
    Finds Order Block at the origin of liquidity move.

    Logic:
    1. Given liquidity sweep at time X
    2. Find the BOS that started the current trend (using TrendStartIndex)
    3. Use pre-calculated OB data at that origin for Top/Bottom bounds
    4. Return OB levels for take profit calculation
    """

    def __init__(
        self,
        df_mid: pd.DataFrame,
        bos_mid: pd.DataFrame,
        inflexions_mid: pd.DataFrame,
        ob_mid: pd.DataFrame = None
    ) -> None:
        """
        Initialize exit target finder.

        Args:
            df_mid: Mid TF OHLCV DataFrame with datetime index
            bos_mid: Pre-calculated BOS data with TrendStartIndex and Level
            inflexions_mid: Pre-calculated inflexion points
            ob_mid: Pre-calculated Order Block data with Top/Bottom bounds
        """
        self.df_mid = df_mid
        self.bos_mid = bos_mid
        self.inflexions_mid = inflexions_mid
        self.ob_mid = ob_mid

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
        if sweep_time not in self.df_mid.index:
            # Find nearest 5M time at or before sweep
            mask = self.df_mid.index <= sweep_time
            if not mask.any():
                return None
            sweep_time = self.df_mid.index[mask][-1]

        sweep_idx = self.df_mid.index.get_loc(sweep_time)

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
        Use BOS TrendStartIndex and OB data directly instead of manual searching.
        """
        # Search backwards for bullish BOS (where uptrend started)
        for i in range(sweep_idx - 1, -1, -1):
            bos_val = self.bos_mid['BOS'].iloc[i]

            if bos_val == 1:  # Bullish BOS = uptrend started
                trend_start = int(self.bos_mid['TrendStartIndex'].iloc[i])
                level = self.bos_mid['Level'].iloc[i]

                # Find OB where EndIndex matches this BOS index
                ob_top = level
                ob_bottom = level
                if self.ob_mid is not None:
                    ob_match = self.ob_mid[self.ob_mid['EndIndex'] == i]
                    if len(ob_match) > 0:
                        ob_top = ob_match['Top'].iloc[0]
                        ob_bottom = ob_match['Bottom'].iloc[0]

                return ExitTarget(
                    ob_top=ob_top,
                    ob_bottom=ob_bottom,
                    ob_start_idx=trend_start,
                    ob_end_idx=i,
                    take_profit=ob_top,  # For SHORT: TP at top of OB
                    bos_idx=i,
                    ob_start_time=self.df_mid.index[trend_start],
                    ob_end_time=self.df_mid.index[i]
                )

        return None

    def _find_exit_ob_for_long(self, sweep_idx: int) -> Optional[ExitTarget]:
        """
        Find exit OB for LONG entry.

        Search backwards from sweep for the bearish BOS that started the downtrend.
        Use BOS TrendStartIndex and OB data directly instead of manual searching.
        """
        # Search backwards for bearish BOS (where downtrend started)
        # self.bos_mid.to_csv('bos_5m_debug.csv')
        # self.ob_mid.to_csv('ob_5m_debug.csv')
        for i in range(sweep_idx - 1, -1, -1):
            bos_val = self.bos_mid['BOS'].iloc[i]

            if bos_val == -1:  # Bearish BOS = downtrend started
                trend_start = int(self.bos_mid['TrendStartIndex'].iloc[i])
                level = self.bos_mid['Level'].iloc[i]

                # Find OB where EndIndex matches this BOS index
                ob_top = level
                ob_bottom = level
                if self.ob_mid is not None:
                    ob_match = self.ob_mid[self.ob_mid['EndIndex'] == i]
                    if len(ob_match) > 0:
                        ob_top = ob_match['Top'].iloc[0]
                        ob_bottom = ob_match['Bottom'].iloc[0]

                return ExitTarget(
                    ob_top=ob_top,
                    ob_bottom=ob_bottom,
                    ob_start_idx=trend_start,
                    ob_end_idx=i,
                    take_profit=ob_bottom,  # For LONG: TP at bottom of OB
                    bos_idx=i,
                    ob_start_time=self.df_mid.index[trend_start],
                    ob_end_time=self.df_mid.index[i]
                )

        return None

