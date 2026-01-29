"""
3-OB Strategy Engine — Python port of ob_rsi_strategy.pine

State machine:
  State 0: Detect OB A touch & leave + find opposite OB C with >min_dist_pct distance
  State 1: Wait for new OB B (same direction as A) formed after confirmation
  State 2: OB-B found, waiting for price to leave and retouch
  State 3: OB-B retouched, waiting for close break → entry signal

OBs are precomputed via smc_custom.ob() and looked up by StartIndex (stable keys).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .models import TradeSignal


@dataclass
class OBRecord:
    """A single order block record from the precomputed OB table."""
    direction: int         # 1 = bullish, -1 = bearish
    top: float
    bottom: float
    start_idx: int         # bar index where OB starts (inflexion)
    bos_idx: int           # bar index of the BOS candle
    status_idx: int        # bar where OB is invalidated (0 = still alive)
    respected: Optional[bool]  # False = invalidated, True = respected, None = pending


class ThreeOBEngine:
    """
    Scans OHLCV data for 3-OB strategy signals.

    Uses precomputed OB table from smc_custom.ob() keyed by StartIndex,
    eliminating fragile list-index juggling.
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
        self.enable_shorts = False

        from src.indicators.smc_custom import smc_custom
        self.inflexions = smc_custom.inflexion_points(self.df)
        self.bos_data = smc_custom.bos(self.df, self.inflexions, close_break=close_break)
        self.ob_table = smc_custom.ob(self.df, self.bos_data)

        # Build OB records dict keyed by (start_idx, bos_idx) — composite key
        # avoids silent overwrites when multiple OBs share the same StartIndex.
        self.ob_records: Dict[tuple, OBRecord] = {}
        # Map bos_idx → list of (start_idx, bos_idx) keys for O(1) activation lookup
        self.bos_to_obs: Dict[int, List[tuple]] = {}

        ob_rows = self.ob_table[self.ob_table['OB'].notna()]
        for _, row in ob_rows.iterrows():
            start = int(row['StartIndex'])
            bos = int(row['BOSIndex'])
            key = (start, bos)
            raw_respected = row.get('Respected', None)
            respected = None if raw_respected is None or (isinstance(raw_respected, float) and np.isnan(raw_respected)) else bool(raw_respected)
            rec = OBRecord(
                direction=int(row['OB']),
                top=float(row['Top']),
                bottom=float(row['Bottom']),
                start_idx=start,
                bos_idx=bos,
                status_idx=int(row.get('StatusIndex', 0)),
                respected=respected,
            )
            self.ob_records[key] = rec
            self.bos_to_obs.setdefault(bos, []).append(key)

    def scan_for_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        """
        Run the 3-state machine across all bars.
        OBs are activated at their BOS bar and removed on invalidation.
        Keys are StartIndex values (stable, never shift).
        """
        signals: List[TradeSignal] = []
        for _bar, snapshot in self.scan_with_snapshots(max_signals=max_signals):
            if snapshot['signal'] is not None:
                signals.append(snapshot['signal'])
        return signals

    def scan_with_snapshots(self, max_signals: int = 10):
        """
        Yield (bar_index, snapshot_dict) for each bar.

        The snapshot contains the current state machine status and any signal
        fired on that bar.  Used by the walkthrough visualizer to show
        per-bar state progression.
        """
        df = self.df
        n = len(df)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_ = df['open'].values
        times = df['time'].values

        signals: List[TradeSignal] = []
        # Active OBs keyed by (start_idx, bos_idx)
        active_obs: Dict[tuple, OBRecord] = {}
        # Track which OBs have had price move away since activation
        ob_has_left: set = set()
        # Accumulated invalidated OB keys (for visualization)
        dead_obs: List[tuple] = []

        # --- state variables (long) ---
        long_state = 0
        long_ob_a_key: Optional[int] = None
        long_ob_a_bottom: Optional[float] = None
        long_ob_a_confirmed_bar: Optional[int] = None
        long_ob_b_key: Optional[int] = None
        long_ob_b_retouched: bool = False
        long_ob_c_key: Optional[int] = None
        long_ob_c_bottom: Optional[float] = None

        # --- state variables (short) ---
        short_state = 0
        short_ob_a_key: Optional[int] = None
        short_ob_a_top: Optional[float] = None
        short_ob_a_confirmed_bar: Optional[int] = None
        short_ob_b_key: Optional[int] = None
        short_ob_b_retouched: bool = False
        short_ob_c_key: Optional[int] = None
        short_ob_c_top: Optional[float] = None

        for bar in range(n):
            if len(signals) >= max_signals:
                break

            bar_low = low[bar]
            bar_high = high[bar]
            bar_close = close[bar]
            bar_open = open_[bar]
            break_low = min(bar_close, bar_open)
            break_high = max(bar_close, bar_open)

            # --- Step 1: Activate new OBs at their BOS bar ---
            if bar in self.bos_to_obs:
                for start_key in self.bos_to_obs[bar]:
                    if start_key in self.ob_records:
                        active_obs[start_key] = self.ob_records[start_key]

            # --- Step 2: Check invalidation & state machine ---
            long_signal = False
            short_signal = False
            long_sl = None
            short_sl = None

            to_remove: List[tuple] = []

            for key, ob in active_obs.items():
                # Use precomputed invalidation from OB table
                invalidated = (ob.status_idx != 0 and not ob.respected and bar >= ob.status_idx)

                if invalidated:
                    # Reset state if relevant OB was invalidated
                    if ob.direction == 1:
                        if long_state in (1, 2, 3) and key == long_ob_a_key:
                            long_state = 0
                            long_ob_a_key = None
                            long_ob_a_bottom = None
                            long_ob_a_confirmed_bar = None
                            long_ob_b_key = None
                            long_ob_b_retouched = False
                        if long_state in (2, 3) and key == long_ob_b_key:
                            long_state = 1
                            long_ob_b_key = None
                            long_ob_b_retouched = False
                    if ob.direction == -1:
                        if short_state in (1, 2) and key == short_ob_a_key:
                            short_state = 0
                            short_ob_a_key = None
                            short_ob_a_top = None
                            short_ob_a_confirmed_bar = None
                            short_ob_b_key = None
                            short_ob_b_retouched = False
                        if short_state == 2 and key == short_ob_b_key:
                            short_state = 1
                            short_ob_b_key = None
                            short_ob_b_retouched = False
                    if key == long_ob_c_key:
                        long_ob_c_key = None
                        long_ob_c_bottom = None
                        long_state = 0
                        long_ob_a_key = None
                        long_ob_a_bottom = None
                        long_ob_a_confirmed_bar = None
                        long_ob_b_key = None
                        long_ob_b_retouched = False
                    if key == short_ob_c_key:
                        short_ob_c_key = None
                        short_ob_c_top = None
                        short_state = 0
                        short_ob_a_key = None
                        short_ob_a_top = None
                        short_ob_a_confirmed_bar = None
                        short_ob_b_key = None
                        short_ob_b_retouched = False

                    to_remove.append(key)
                    continue

                # OB still alive — check touch
                touching = bar_low <= ob.top and bar_high >= ob.bottom
                if bar > 0:
                    was_touching_prev = low[bar - 1] <= ob.top and high[bar - 1] >= ob.bottom
                else:
                    was_touching_prev = False

                # Track whether price has moved away from OB since activation.
                # OB only qualifies for touch-and-leave once it has been "away"
                # on a PRIOR bar (not the current one).
                was_away = key in ob_has_left
                if not was_away and not touching:
                    ob_has_left.add(key)

                # --- LONG strategy (bullish OBs) ---
                if ob.direction == 1:
                    if long_state == 0:
                        retouched = was_away and touching
                        if retouched:
                            best_dist = float('inf')
                            best_bear_key = None
                            best_bear_bot = None
                            for j_key, ob_j in active_obs.items():
                                if ob_j.direction != -1:
                                    continue
                                if j_key in to_remove:
                                    continue
                                if ob_j.bottom > bar_close:
                                    dist = ob_j.bottom - bar_close
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_bear_key = j_key
                                        best_bear_bot = ob_j.bottom
                            if best_bear_bot is not None and (best_bear_bot - bar_close) / bar_close > self.min_dist_pct:
                                long_state = 1
                                long_ob_a_bottom = ob.bottom
                                long_ob_a_key = key
                                long_ob_a_confirmed_bar = bar
                                long_ob_c_key = best_bear_key
                                long_ob_c_bottom = best_bear_bot

                    elif long_state == 1 and key != long_ob_a_key:
                        if ob.start_idx > long_ob_a_confirmed_bar:
                            long_state = 2
                            long_ob_b_key = key

                    elif long_state == 2 and key == long_ob_b_key:
                        b_was_away = long_ob_b_key in ob_has_left
                        if b_was_away and touching:
                            long_ob_b_retouched = True
                            long_state = 3

                    elif long_state == 3 and key == long_ob_b_key:
                        # OB-B was re-touched; now check for green candle
                        if bar_close > bar_open:
                            long_signal = True
                            long_sl = long_ob_a_bottom

                # --- Invalidate long setup if price touches OB-C ---
                if ob.direction == -1 and long_state in (1, 2, 3) and key == long_ob_c_key:
                    if touching:
                        long_state = 0
                        long_ob_a_key = None
                        long_ob_a_bottom = None
                        long_ob_a_confirmed_bar = None
                        long_ob_b_key = None
                        long_ob_b_retouched = False
                        long_ob_c_key = None
                        long_ob_c_bottom = None

                # --- SHORT strategy (bearish OBs) ---
                if self.enable_shorts and ob.direction == -1:
                    if short_state == 0:
                        retouched = was_away and touching
                        if retouched:
                            best_dist = float('inf')
                            best_bull_key = None
                            best_bull_top = None
                            for j_key, ob_j in active_obs.items():
                                if ob_j.direction != 1:
                                    continue
                                if j_key in to_remove:
                                    continue
                                if ob_j.top < bar_close:
                                    dist = bar_close - ob_j.top
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_bull_key = j_key
                                        best_bull_top = ob_j.top
                            if best_bull_top is not None and (bar_close - best_bull_top) / bar_close > self.min_dist_pct:
                                short_state = 1
                                short_ob_a_top = ob.top
                                short_ob_a_key = key
                                short_ob_a_confirmed_bar = bar
                                short_ob_c_key = best_bull_key
                                short_ob_c_top = best_bull_top

                    elif short_state == 1 and key != short_ob_a_key:
                        if ob.start_idx > short_ob_a_confirmed_bar:
                            short_state = 2
                            short_ob_b_key = key

                    elif short_state == 2 and key == short_ob_b_key:
                        b_was_away = short_ob_b_key in ob_has_left
                        if not short_ob_b_retouched:
                            if b_was_away and touching:
                                short_ob_b_retouched = True
                        else:
                            # OB-B was re-touched; now check for red candle
                            if bar_close < bar_open:
                                short_signal = True
                                short_sl = short_ob_a_top

            # Remove invalidated OBs (no index adjustment needed)
            for key in to_remove:
                del active_obs[key]
                ob_has_left.discard(key)
                dead_obs.append(key)

            # --- Execute long entry ---
            bar_signal = None
            if long_signal and long_ob_c_key is not None and long_ob_c_key in active_obs:
                ob_c = active_obs[long_ob_c_key]
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
                    bar_signal = sig
                long_state = 0
                long_ob_a_key = None
                long_ob_a_bottom = None
                long_ob_a_confirmed_bar = None
                long_ob_b_key = None
                long_ob_b_retouched = False
                long_ob_c_key = None
                long_ob_c_bottom = None

            # --- Execute short entry ---
            if self.enable_shorts and short_signal and short_ob_c_key is not None and short_ob_c_key in active_obs:
                ob_c = active_obs[short_ob_c_key]
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
                    bar_signal = sig
                short_state = 0
                short_ob_a_key = None
                short_ob_a_top = None
                short_ob_a_confirmed_bar = None
                short_ob_b_key = None
                short_ob_b_retouched = False
                short_ob_c_key = None
                short_ob_c_top = None

            # --- Build snapshot ---
            snapshot = {
                'long_state': long_state,
                'short_state': short_state,
                'long_ob_a_key': long_ob_a_key,
                'long_ob_b_key': long_ob_b_key,
                'long_ob_c_key': long_ob_c_key,
                'short_ob_a_key': short_ob_a_key,
                'short_ob_b_key': short_ob_b_key,
                'short_ob_c_key': short_ob_c_key,
                'active_ob_keys': list(active_obs.keys()),
                'dead_ob_keys': list(dead_obs),
                'signal': bar_signal,
                'signals_so_far': len(signals),
            }
            yield bar, snapshot
