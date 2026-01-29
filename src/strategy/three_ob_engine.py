"""
3-OB Strategy Engine — Python port of ob_rsi_strategy.pine

State machine:
  State 0: Detect OB A touch & leave + find opposite OB C with >min_dist_pct distance
  State 1: Wait for new OB B (same direction as A) formed after confirmation, price touches it
  State 2: Wait for close past OB B boundary → entry signal

OB creation is dynamic: OBs are built inline at each BOS event, matching Pine Script behavior.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from .models import TradeSignal


@dataclass
class OrderBlock:
    """A single order block derived from BOS."""
    direction: int         # 1 = bullish, -1 = bearish
    top: float
    bottom: float
    start_idx: int         # bar index where OB starts (inflexion)
    bos_idx: int           # bar index of the BOS candle


class ThreeOBEngine:
    """
    Scans OHLCV data for 3-OB strategy signals.

    Reuses smc_custom.inflexion_points and smc_custom.bos for detection,
    then builds OBs dynamically during the bar loop (matching Pine Script).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        close_break: bool = True,
        min_dist_pct: float = 0.10,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.close_break = close_break
        self.min_dist_pct = min_dist_pct

        from src.indicators.smc_custom import smc_custom
        self.inflexions = smc_custom.inflexion_points(self.df)
        self.bos_data = smc_custom.bos(self.df, self.inflexions, close_break=close_break)

    def scan_for_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        """
        Run the 3-state machine across all bars, mirroring Pine Script logic.
        OBs are created dynamically at each BOS event and removed on invalidation.
        """
        df = self.df
        n = len(df)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_ = df['open'].values
        times = df['time'].values

        # BOS arrays for fast access
        bos_values = self.bos_data['BOS'].values
        bos_levels = self.bos_data['Level'].values
        bos_inflx = self.bos_data['BrokenInflexionIndex'].values

        signals: List[TradeSignal] = []
        ob_list: List[OrderBlock] = []

        # --- state variables (long) ---
        long_state = 0
        long_ob_a_idx: Optional[int] = None
        long_ob_a_bottom: Optional[float] = None
        long_ob_a_confirmed_bar: Optional[int] = None
        long_ob_b_idx: Optional[int] = None
        long_ob_c_idx: Optional[int] = None
        long_ob_c_bottom: Optional[float] = None

        # --- state variables (short) ---
        short_state = 0
        short_ob_a_idx: Optional[int] = None
        short_ob_a_top: Optional[float] = None
        short_ob_a_confirmed_bar: Optional[int] = None
        short_ob_b_idx: Optional[int] = None
        short_ob_c_idx: Optional[int] = None
        short_ob_c_top: Optional[float] = None

        def _adjust_idx_after_remove(idx, removed):
            """Adjust a saved ob_list index after an element was removed."""
            if idx is None:
                return None
            if idx == removed:
                return None  # the referenced OB was removed
            if idx > removed:
                return idx - 1
            return idx

        for bar in range(n):
            if len(signals) >= max_signals:
                break

            bar_low = low[bar]
            bar_high = high[bar]
            bar_close = close[bar]
            bar_open = open_[bar]
            break_low = min(bar_close, bar_open)
            break_high = max(bar_close, bar_open)

            # --- Step 1: Create new OB if BOS at this bar ---
            if not np.isnan(bos_values[bar]):
                bos_type = int(bos_values[bar])
                inflexion_idx = int(bos_inflx[bar]) if not np.isnan(bos_inflx[bar]) else bar
                if bos_type == 1:  # Bullish BOS → Bullish OB
                    zone_bottom = float(np.min(low[inflexion_idx:bar + 1]))
                    ob_list.append(OrderBlock(
                        direction=1,
                        top=float(bos_levels[bar]),
                        bottom=zone_bottom,
                        start_idx=inflexion_idx,
                        bos_idx=bar,
                    ))
                elif bos_type == -1:  # Bearish BOS → Bearish OB
                    zone_top = float(np.max(high[inflexion_idx:bar + 1]))
                    ob_list.append(OrderBlock(
                        direction=-1,
                        top=zone_top,
                        bottom=float(bos_levels[bar]),
                        start_idx=inflexion_idx,
                        bos_idx=bar,
                    ))

            # --- Step 2: Iterate OBs in reverse, check invalidation & state machine ---
            long_signal = False
            short_signal = False
            long_sl = None
            short_sl = None

            to_remove: List[int] = []

            for ob_i in range(len(ob_list) - 1, -1, -1):
                ob = ob_list[ob_i]

                # Check invalidation with <= / >= (match Pine)
                invalidated = False
                if ob.direction == 1 and break_low <= ob.bottom:
                    invalidated = True
                elif ob.direction == -1 and break_high >= ob.top:
                    invalidated = True

                if invalidated:
                    # Reset state if relevant OB was invalidated
                    if ob.direction == 1:
                        if long_state in (1, 2) and ob_i == long_ob_a_idx:
                            long_state = 0
                            long_ob_a_idx = None
                            long_ob_a_bottom = None
                            long_ob_a_confirmed_bar = None
                            long_ob_b_idx = None
                        if long_state == 2 and ob_i == long_ob_b_idx:
                            long_state = 1
                            long_ob_b_idx = None
                    if ob.direction == -1:
                        if short_state in (1, 2) and ob_i == short_ob_a_idx:
                            short_state = 0
                            short_ob_a_idx = None
                            short_ob_a_top = None
                            short_ob_a_confirmed_bar = None
                            short_ob_b_idx = None
                        if short_state == 2 and ob_i == short_ob_b_idx:
                            short_state = 1
                            short_ob_b_idx = None
                    if ob_i == long_ob_c_idx:
                        long_ob_c_idx = None
                        long_ob_c_bottom = None
                        long_state = 0
                        long_ob_a_idx = None
                        long_ob_a_bottom = None
                        long_ob_a_confirmed_bar = None
                        long_ob_b_idx = None
                    if ob_i == short_ob_c_idx:
                        short_ob_c_idx = None
                        short_ob_c_top = None
                        short_state = 0
                        short_ob_a_idx = None
                        short_ob_a_top = None
                        short_ob_a_confirmed_bar = None
                        short_ob_b_idx = None

                    to_remove.append(ob_i)
                    continue

                # OB still alive — check touch
                touching = bar_low <= ob.top and bar_high >= ob.bottom
                if bar > 0:
                    was_touching_prev = low[bar - 1] <= ob.top and high[bar - 1] >= ob.bottom
                else:
                    was_touching_prev = False

                # --- LONG strategy (bullish OBs) ---
                if ob.direction == 1:
                    if long_state == 0:
                        left_now = not touching and was_touching_prev
                        if left_now:
                            best_dist = float('inf')
                            best_bear_idx = None
                            best_bear_bot = None
                            for j, ob_j in enumerate(ob_list):
                                if ob_j.direction != -1:
                                    continue
                                if j in to_remove:
                                    continue
                                if ob_j.bottom > bar_close:
                                    dist = ob_j.bottom - bar_close
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_bear_idx = j
                                        best_bear_bot = ob_j.bottom
                            if best_bear_bot is not None and (best_bear_bot - bar_close) / bar_close > self.min_dist_pct:
                                long_state = 1
                                long_ob_a_bottom = ob.bottom
                                long_ob_a_idx = ob_i
                                long_ob_a_confirmed_bar = bar
                                long_ob_c_idx = best_bear_idx
                                long_ob_c_bottom = best_bear_bot

                    elif long_state == 1 and ob_i != long_ob_a_idx:
                        if ob.start_idx > long_ob_a_confirmed_bar and touching:
                            long_state = 2
                            long_ob_b_idx = ob_i

                    elif long_state == 2 and ob_i == long_ob_b_idx:
                        if bar_close > ob.top:
                            long_signal = True
                            long_sl = long_ob_a_bottom

                # --- SHORT strategy (bearish OBs) ---
                if ob.direction == -1:
                    if short_state == 0:
                        left_now = not touching and was_touching_prev
                        if left_now:
                            best_dist = float('inf')
                            best_bull_idx = None
                            best_bull_top = None
                            for j, ob_j in enumerate(ob_list):
                                if ob_j.direction != 1:
                                    continue
                                if j in to_remove:
                                    continue
                                if ob_j.top < bar_close:
                                    dist = bar_close - ob_j.top
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_bull_idx = j
                                        best_bull_top = ob_j.top
                            if best_bull_top is not None and (bar_close - best_bull_top) / bar_close > self.min_dist_pct:
                                short_state = 1
                                short_ob_a_top = ob.top
                                short_ob_a_idx = ob_i
                                short_ob_a_confirmed_bar = bar
                                short_ob_c_idx = best_bull_idx
                                short_ob_c_top = best_bull_top

                    elif short_state == 1 and ob_i != short_ob_a_idx:
                        if ob.start_idx > short_ob_a_confirmed_bar and touching:
                            short_state = 2
                            short_ob_b_idx = ob_i

                    elif short_state == 2 and ob_i == short_ob_b_idx:
                        if bar_close < ob.bottom:
                            short_signal = True
                            short_sl = short_ob_a_top

            # --- Step 3: Remove invalidated OBs and adjust indices ---
            for removed_idx in sorted(to_remove, reverse=True):
                ob_list.pop(removed_idx)
                # Adjust all saved indices
                long_ob_a_idx = _adjust_idx_after_remove(long_ob_a_idx, removed_idx)
                long_ob_b_idx = _adjust_idx_after_remove(long_ob_b_idx, removed_idx)
                long_ob_c_idx = _adjust_idx_after_remove(long_ob_c_idx, removed_idx)
                short_ob_a_idx = _adjust_idx_after_remove(short_ob_a_idx, removed_idx)
                short_ob_b_idx = _adjust_idx_after_remove(short_ob_b_idx, removed_idx)
                short_ob_c_idx = _adjust_idx_after_remove(short_ob_c_idx, removed_idx)

            # --- Execute long entry ---
            if long_signal and long_ob_c_idx is not None and long_ob_c_idx < len(ob_list):
                ob_c = ob_list[long_ob_c_idx]
                ob_c_start = ob_c.start_idx
                lowest_low = np.min(low[ob_c_start:bar + 1]) if ob_c_start <= bar else bar_low
                tp = (lowest_low + long_ob_c_bottom) / 2.0
                if tp > bar_close:
                    entry_time = pd.Timestamp(times[bar])
                    sig = TradeSignal(
                        timestamp_1h_sweep=entry_time,
                        timestamp_5m_event_b=entry_time,
                        timestamp_5m_validation=entry_time,
                        timestamp_1m_confirmation=entry_time,
                        timestamp_entry=entry_time,
                        trend_1h_before_sweep='n/a',
                        entry_direction='long',
                        price_1h_sweep=bar_close,
                        price_5m_event_b=bar_close,
                        price_5m_validation=bar_close,
                        price_1m_confirmation=bar_close,
                        price_entry=bar_close,
                        condition_liquidity_sweep='3-OB State Machine',
                        condition_event_b='OB-A touch',
                        condition_validation='OB-C distance',
                        condition_confirmation='OB-B close break',
                        indices_1h=(bar, bar),
                        indices_5m=(bar, bar),
                        indices_1m=(bar, bar),
                        take_profit_price=tp,
                        stop_loss_price=long_sl,
                    )
                    signals.append(sig)
                long_state = 0
                long_ob_a_idx = None
                long_ob_a_bottom = None
                long_ob_a_confirmed_bar = None
                long_ob_b_idx = None
                long_ob_c_idx = None
                long_ob_c_bottom = None

            # --- Execute short entry ---
            if short_signal and short_ob_c_idx is not None and short_ob_c_idx < len(ob_list):
                ob_c = ob_list[short_ob_c_idx]
                ob_c_start = ob_c.start_idx
                highest_high = np.max(high[ob_c_start:bar + 1]) if ob_c_start <= bar else bar_high
                tp = (highest_high + short_ob_c_top) / 2.0
                if tp < bar_close:
                    entry_time = pd.Timestamp(times[bar])
                    sig = TradeSignal(
                        timestamp_1h_sweep=entry_time,
                        timestamp_5m_event_b=entry_time,
                        timestamp_5m_validation=entry_time,
                        timestamp_1m_confirmation=entry_time,
                        timestamp_entry=entry_time,
                        trend_1h_before_sweep='n/a',
                        entry_direction='short',
                        price_1h_sweep=bar_close,
                        price_5m_event_b=bar_close,
                        price_5m_validation=bar_close,
                        price_1m_confirmation=bar_close,
                        price_entry=bar_close,
                        condition_liquidity_sweep='3-OB State Machine',
                        condition_event_b='OB-A touch',
                        condition_validation='OB-C distance',
                        condition_confirmation='OB-B close break',
                        indices_1h=(bar, bar),
                        indices_5m=(bar, bar),
                        indices_1m=(bar, bar),
                        take_profit_price=tp,
                        stop_loss_price=short_sl,
                    )
                    signals.append(sig)
                short_state = 0
                short_ob_a_idx = None
                short_ob_a_top = None
                short_ob_a_confirmed_bar = None
                short_ob_b_idx = None
                short_ob_c_idx = None
                short_ob_c_top = None

        return signals
