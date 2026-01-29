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

from .models import TradeSignal, ThreeOBSignalContext


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
        enable_shorts: bool = False,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.close_break = close_break
        self.min_dist_pct = min_dist_pct
        self.enable_shorts = enable_shorts

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

        This is the non-LTF fallback path: when a pending_confirmation is
        emitted and the bar has a matching close color, the signal fires
        directly (green close for long, red close for short).
        """
        df = self.df
        close = df['close'].values
        open_ = df['open'].values
        times = df['time'].values

        signals: List[TradeSignal] = []
        # Track OB-B keys that already fired to avoid duplicate signals
        # (State 3 persists and emits pending_confirmation every bar)
        fired_ob_b_keys: set = set()
        for bar, snapshot in self.scan_with_snapshots(max_signals=max_signals):
            if snapshot['signal'] is not None:
                signals.append(snapshot['signal'])
                continue

            pc = snapshot.get('pending_confirmation')
            if pc is None:
                continue

            if pc['ob_b_key'] in fired_ob_b_keys:
                continue

            bar_close = close[bar]
            bar_open = open_[bar]
            direction = pc['direction']

            # Check close-break: green for long, red for short
            if direction == 'long' and bar_close <= bar_open:
                continue
            if direction == 'short' and bar_close >= bar_open:
                continue

            # Determine TP from OB-C
            ob_c_key = pc['ob_c_key']
            if ob_c_key is None:
                continue

            if direction == 'long':
                tp = pc['ob_c_bottom']
                if tp <= bar_close:
                    continue
            else:
                tp = pc['ob_c_top']
                if tp >= bar_close:
                    continue

            fired_ob_b_keys.add(pc['ob_b_key'])
            entry_time = pd.Timestamp(times[bar])
            sig = TradeSignal(
                timestamp_1h_sweep=entry_time,
                timestamp_5m_event_b=entry_time,
                timestamp_5m_validation=entry_time,
                timestamp_1m_confirmation=entry_time,
                timestamp_entry=entry_time,
                trend_1h_before_sweep='n/a',
                entry_direction=direction,
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
                stop_loss_price=pc['sl'],
            )
            signals.append(sig)
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
        long_ob_a_top: Optional[float] = None
        long_ob_a_confirmed_bar: Optional[int] = None
        long_ob_b_key: Optional[int] = None
        long_ob_b_bottom: Optional[float] = None
        long_ob_b_top: Optional[float] = None
        long_ob_b_retouched: bool = False
        long_ob_b_retouch_bar: Optional[int] = None
        long_ob_c_key: Optional[int] = None
        long_ob_c_bottom: Optional[float] = None
        long_ob_c_top: Optional[float] = None

        # --- state variables (short) ---
        short_state = 0
        short_ob_a_key: Optional[int] = None
        short_ob_a_top: Optional[float] = None
        short_ob_a_bottom: Optional[float] = None
        short_ob_a_confirmed_bar: Optional[int] = None
        short_ob_b_key: Optional[int] = None
        short_ob_b_bottom: Optional[float] = None
        short_ob_b_top: Optional[float] = None
        short_ob_b_retouched: bool = False
        short_ob_b_retouch_bar: Optional[int] = None
        short_ob_c_key: Optional[int] = None
        short_ob_c_top: Optional[float] = None
        short_ob_c_bottom: Optional[float] = None

        reset_cmd = None
        for bar in range(n):
            if len(signals) >= max_signals:
                break

            # Handle reset commands sent via .send()
            if reset_cmd:
                if reset_cmd in ('reset_long', 'reset_both'):
                    long_state = 0
                    long_ob_a_key = None
                    long_ob_a_bottom = None
                    long_ob_a_top = None
                    long_ob_a_confirmed_bar = None
                    long_ob_b_key = None
                    long_ob_b_bottom = None
                    long_ob_b_top = None
                    long_ob_b_retouched = False
                    long_ob_b_retouch_bar = None
                    long_ob_c_key = None
                    long_ob_c_bottom = None
                    long_ob_c_top = None
                if reset_cmd in ('reset_short', 'reset_both'):
                    short_state = 0
                    short_ob_a_key = None
                    short_ob_a_top = None
                    short_ob_a_bottom = None
                    short_ob_a_confirmed_bar = None
                    short_ob_b_key = None
                    short_ob_b_bottom = None
                    short_ob_b_top = None
                    short_ob_b_retouched = False
                    short_ob_b_retouch_bar = None
                    short_ob_c_key = None
                    short_ob_c_top = None
                    short_ob_c_bottom = None
                reset_cmd = None

            bar_low = low[bar]
            bar_high = high[bar]
            bar_close = close[bar]
            bar_open = open_[bar]
            break_low = min(bar_close, bar_open)
            break_high = max(bar_close, bar_open)

            long_status = ""
            short_status = ""

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
                            long_status = "OB-A invalidated — resetting to state 0"
                            long_state = 0
                            long_ob_a_key = None
                            long_ob_a_bottom = None
                            long_ob_a_top = None
                            long_ob_a_confirmed_bar = None
                            long_ob_b_key = None
                            long_ob_b_bottom = None
                            long_ob_b_top = None
                            long_ob_b_retouched = False
                            long_ob_b_retouch_bar = None
                        if long_state in (2, 3) and key == long_ob_b_key:
                            long_status = "OB-B invalidated — back to state 1"
                            long_state = 1
                            long_ob_b_key = None
                            long_ob_b_bottom = None
                            long_ob_b_top = None
                            long_ob_b_retouched = False
                            long_ob_b_retouch_bar = None
                    if ob.direction == -1:
                        if short_state in (1, 2, 3) and key == short_ob_a_key:
                            short_status = "OB-A invalidated — resetting to state 0"
                            short_state = 0
                            short_ob_a_key = None
                            short_ob_a_top = None
                            short_ob_a_bottom = None
                            short_ob_a_confirmed_bar = None
                            short_ob_b_key = None
                            short_ob_b_bottom = None
                            short_ob_b_top = None
                            short_ob_b_retouched = False
                            short_ob_b_retouch_bar = None
                        if short_state in (2, 3) and key == short_ob_b_key:
                            short_status = "OB-B invalidated — back to state 1"
                            short_state = 1
                            short_ob_b_key = None
                            short_ob_b_bottom = None
                            short_ob_b_top = None
                            short_ob_b_retouched = False
                            short_ob_b_retouch_bar = None
                    if key == long_ob_c_key:
                        long_status = "OB-C invalidated — resetting to state 0"
                        long_ob_c_key = None
                        long_ob_c_bottom = None
                        long_ob_c_top = None
                        long_state = 0
                        long_ob_a_key = None
                        long_ob_a_bottom = None
                        long_ob_a_top = None
                        long_ob_a_confirmed_bar = None
                        long_ob_b_key = None
                        long_ob_b_bottom = None
                        long_ob_b_top = None
                        long_ob_b_retouched = False
                        long_ob_b_retouch_bar = None
                    if key == short_ob_c_key:
                        short_status = "OB-C invalidated — resetting to state 0"
                        short_ob_c_key = None
                        short_ob_c_top = None
                        short_ob_c_bottom = None
                        short_state = 0
                        short_ob_a_key = None
                        short_ob_a_top = None
                        short_ob_a_bottom = None
                        short_ob_a_confirmed_bar = None
                        short_ob_b_key = None
                        short_ob_b_bottom = None
                        short_ob_b_top = None
                        short_ob_b_retouched = False
                        short_ob_b_retouch_bar = None

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
                        if not was_away and touching:
                            long_status = "Bullish OB touching — waiting for price to leave first"
                        retouched = was_away and touching
                        if retouched:
                            candidates = []
                            for j_key, ob_j in active_obs.items():
                                if ob_j.direction != -1 or j_key in to_remove:
                                    continue
                                if ob_j.bottom > bar_close:
                                    dist = ob_j.bottom - bar_close
                                    dist_pct = dist / bar_close if bar_close > 0 else 0
                                    candidates.append((dist, dist_pct, j_key, ob_j.bottom, ob_j.top))
                            candidates.sort()  # by distance ascending

                            chosen = None
                            for dist, dist_pct, c_key, c_bot, c_top in candidates:
                                if dist_pct > self.min_dist_pct:
                                    chosen = (dist_pct, c_key, c_bot, c_top)
                                    break

                            if chosen:
                                best_dist_pct, best_bear_key, best_bear_bot, best_bear_top = chosen
                                long_state = 1
                                long_ob_a_bottom = ob.bottom
                                long_ob_a_top = ob.top
                                long_ob_a_key = key
                                long_ob_a_confirmed_bar = bar
                                long_ob_c_key = best_bear_key
                                long_ob_c_bottom = best_bear_bot
                                long_ob_c_top = best_bear_top
                                long_status = f"[State 0→1] OB-A (bull ${ob.bottom:.2f}–${ob.top:.2f}) confirmed, OB-C found {best_dist_pct*100:.1f}% above — close=${bar_close:.2f}, OB-C bottom=${best_bear_bot:.2f}"
                            elif candidates:
                                long_status = f"OB touched & left but all OB-C candidates too close (closest: {candidates[0][1]*100:.1f}% < {self.min_dist_pct*100:.1f}%) — close=${bar_close:.2f}, OB-C bottom=${candidates[0][3]:.2f}"
                            else:
                                long_status = "OB touched & left but no bearish OB above current price"

                    elif long_state == 1 and key != long_ob_a_key:
                        if ob.start_idx > long_ob_a_confirmed_bar:
                            long_state = 2
                            long_ob_b_key = key
                            long_ob_b_bottom = ob.bottom
                            long_ob_b_top = ob.top
                            long_status = f"[State 1→2] OB-B (bull ${ob.bottom:.2f}–${ob.top:.2f}) found after OB-A (${long_ob_a_bottom:.2f}–${long_ob_a_top:.2f})"

                    elif long_state == 2 and key == long_ob_b_key:
                        b_was_away = long_ob_b_key in ob_has_left
                        if not b_was_away and touching:
                            long_status = f"[State 2] OB-B (${long_ob_b_bottom:.2f}–${long_ob_b_top:.2f}) active, waiting for price to leave"
                        if b_was_away and touching:
                            long_ob_b_retouched = True
                            long_ob_b_retouch_bar = bar
                            long_state = 3
                            long_status = f"[State 2→3] OB-B (${long_ob_b_bottom:.2f}–${long_ob_b_top:.2f}) retouched — waiting for LTF confirmation (BOS/IFVG), close=${bar_close:.2f}"

                    elif long_state == 3 and key == long_ob_b_key:
                        long_status = f"[State 3] OB-B (${long_ob_b_bottom:.2f}–${long_ob_b_top:.2f}) retouched — waiting for confirmation"

                # --- Invalidate long setup if price touches OB-C ---
                if ob.direction == -1 and long_state in (1, 2, 3) and key == long_ob_c_key:
                    if touching:
                        long_status = f"OB-C (${long_ob_c_bottom:.2f}–${long_ob_c_top:.2f}) reached — invalidating long setup"
                        long_state = 0
                        long_ob_a_key = None
                        long_ob_a_bottom = None
                        long_ob_a_top = None
                        long_ob_a_confirmed_bar = None
                        long_ob_b_key = None
                        long_ob_b_bottom = None
                        long_ob_b_top = None
                        long_ob_b_retouched = False
                        long_ob_b_retouch_bar = None
                        long_ob_c_key = None
                        long_ob_c_bottom = None
                        long_ob_c_top = None

                # --- SHORT strategy (bearish OBs) ---
                if self.enable_shorts and ob.direction == -1:
                    if short_state == 0:
                        if not was_away and touching:
                            short_status = "Bearish OB touching — waiting for price to leave first"
                        retouched = was_away and touching
                        if retouched:
                            candidates = []
                            for j_key, ob_j in active_obs.items():
                                if ob_j.direction != 1 or j_key in to_remove:
                                    continue
                                if ob_j.top < bar_close:
                                    dist = bar_close - ob_j.top
                                    dist_pct = dist / bar_close if bar_close > 0 else 0
                                    candidates.append((dist, dist_pct, j_key, ob_j.top, ob_j.bottom))
                            candidates.sort()  # by distance ascending

                            chosen = None
                            for dist, dist_pct, c_key, c_top, c_bot in candidates:
                                if dist_pct > self.min_dist_pct:
                                    chosen = (dist_pct, c_key, c_top, c_bot)
                                    break

                            if chosen:
                                best_dist_pct, best_bull_key, best_bull_top, best_bull_bot = chosen
                                short_state = 1
                                short_ob_a_top = ob.top
                                short_ob_a_bottom = ob.bottom
                                short_ob_a_key = key
                                short_ob_a_confirmed_bar = bar
                                short_ob_c_key = best_bull_key
                                short_ob_c_top = best_bull_top
                                short_ob_c_bottom = best_bull_bot
                                short_status = f"[State 0→1] OB-A (bear ${ob.bottom:.2f}–${ob.top:.2f}) confirmed, OB-C found {best_dist_pct*100:.1f}% below — close=${bar_close:.2f}, OB-C top=${best_bull_top:.2f}"
                            elif candidates:
                                short_status = f"OB touched & left but all OB-C candidates too close (closest: {candidates[0][1]*100:.1f}% < {self.min_dist_pct*100:.1f}%) — close=${bar_close:.2f}, OB-C top=${candidates[0][3]:.2f}"
                            else:
                                short_status = "OB touched & left but no bullish OB below current price"

                    elif short_state == 1 and key != short_ob_a_key:
                        if ob.start_idx > short_ob_a_confirmed_bar:
                            short_state = 2
                            short_ob_b_key = key
                            short_ob_b_bottom = ob.bottom
                            short_ob_b_top = ob.top
                            short_status = f"[State 1→2] OB-B (bear ${ob.bottom:.2f}–${ob.top:.2f}) found after OB-A (${short_ob_a_bottom:.2f}–${short_ob_a_top:.2f})"

                    elif short_state == 2 and key == short_ob_b_key:
                        b_was_away = short_ob_b_key in ob_has_left
                        if not b_was_away and touching:
                            short_status = f"[State 2] OB-B (${short_ob_b_bottom:.2f}–${short_ob_b_top:.2f}) active, waiting for price to leave"
                        if b_was_away and touching:
                            short_ob_b_retouched = True
                            short_ob_b_retouch_bar = bar
                            short_state = 3
                            short_status = f"[State 2→3] OB-B (${short_ob_b_bottom:.2f}–${short_ob_b_top:.2f}) retouched — waiting for LTF confirmation (BOS/IFVG), close=${bar_close:.2f}"

                    elif short_state == 3 and key == short_ob_b_key:
                        short_status = f"[State 3] OB-B (${short_ob_b_bottom:.2f}–${short_ob_b_top:.2f}) retouched — waiting for confirmation"

                # --- Invalidate short setup if price touches OB-C ---
                if self.enable_shorts and ob.direction == 1 and short_state in (1, 2, 3) and key == short_ob_c_key:
                    if touching:
                        short_status = f"OB-C (${short_ob_c_bottom:.2f}–${short_ob_c_top:.2f}) reached — invalidating short setup"
                        short_state = 0
                        short_ob_a_key = None
                        short_ob_a_top = None
                        short_ob_a_bottom = None
                        short_ob_a_confirmed_bar = None
                        short_ob_b_key = None
                        short_ob_b_retouched = False
                        short_ob_b_retouch_bar = None
                        short_ob_b_bottom = None
                        short_ob_b_top = None
                        short_ob_c_key = None
                        short_ob_c_top = None
                        short_ob_c_bottom = None

            # Remove invalidated OBs (no index adjustment needed)
            for key in to_remove:
                del active_obs[key]
                ob_has_left.discard(key)
                dead_obs.append(key)

            # --- Execute long entry ---
            bar_signal = None
            if long_signal and long_ob_c_key is not None and long_ob_c_key in active_obs:
                tp = long_ob_c_bottom
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
                long_ob_a_top = None
                long_ob_a_confirmed_bar = None
                long_ob_b_key = None
                long_ob_b_bottom = None
                long_ob_b_top = None
                long_ob_b_retouched = False
                long_ob_b_retouch_bar = None
                long_ob_c_key = None
                long_ob_c_bottom = None
                long_ob_c_top = None

            # --- Execute short entry ---
            if self.enable_shorts and short_signal and short_ob_c_key is not None and short_ob_c_key in active_obs:
                tp = short_ob_c_top
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
                short_ob_a_bottom = None
                short_ob_a_confirmed_bar = None
                short_ob_b_key = None
                short_ob_b_bottom = None
                short_ob_b_top = None
                short_ob_b_retouched = False
                short_ob_b_retouch_bar = None
                short_ob_c_key = None
                short_ob_c_top = None
                short_ob_c_bottom = None

            # --- Build pending_confirmation info ---
            # Emitted on the bar where State 2→3 fires (OB-B retouched)
            pending_confirmation = None
            if long_state == 3 and long_ob_b_retouched:
                pending_confirmation = {
                    'direction': 'long',
                    'ob_b_retouch_bar': long_ob_b_retouch_bar,
                    'ob_a_bottom': long_ob_a_bottom,
                    'ob_a_top': long_ob_a_top,
                    'ob_a_key': long_ob_a_key,
                    'ob_a_confirmed_bar': long_ob_a_confirmed_bar,
                    'ob_b_bottom': long_ob_b_bottom,
                    'ob_b_top': long_ob_b_top,
                    'ob_b_key': long_ob_b_key,
                    'ob_c_bottom': long_ob_c_bottom,
                    'ob_c_top': long_ob_c_top,
                    'ob_c_key': long_ob_c_key,
                    'sl': long_ob_a_bottom,
                }
            if self.enable_shorts and short_state == 3 and short_ob_b_retouched:
                pending_confirmation = {
                    'direction': 'short',
                    'ob_b_retouch_bar': short_ob_b_retouch_bar,
                    'ob_a_bottom': short_ob_a_bottom,
                    'ob_a_top': short_ob_a_top,
                    'ob_a_key': short_ob_a_key,
                    'ob_a_confirmed_bar': short_ob_a_confirmed_bar,
                    'ob_b_bottom': short_ob_b_bottom,
                    'ob_b_top': short_ob_b_top,
                    'ob_b_key': short_ob_b_key,
                    'ob_c_bottom': short_ob_c_bottom,
                    'ob_c_top': short_ob_c_top,
                    'ob_c_key': short_ob_c_key,
                    'sl': short_ob_a_top,
                }

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
                'long_status': long_status,
                'short_status': short_status,
                'pending_confirmation': pending_confirmation,
            }
            reset_cmd = yield bar, snapshot

    def scan_for_signals_with_context(self, max_signals: int = 10) -> List[ThreeOBSignalContext]:
        """
        Like scan_for_signals(), but returns ThreeOBSignalContext objects
        that include OB-A/B/C zone information for each signal.
        """
        df = self.df
        close = df['close'].values
        open_ = df['open'].values
        times = df['time'].values

        contexts: List[ThreeOBSignalContext] = []
        fired_ob_b_keys: set = set()

        for bar, snapshot in self.scan_with_snapshots(max_signals=max_signals):
            # Direct signal from engine (State machine fired internally)
            if snapshot['signal'] is not None:
                sig = snapshot['signal']
                ctx = self._context_from_snapshot(sig, snapshot)
                if ctx is not None:
                    sig.ob_a_id = (ctx.ob_a_top, ctx.ob_a_bottom, ctx.ob_a_start_idx)
                    contexts.append(ctx)
                continue

            pc = snapshot.get('pending_confirmation')
            if pc is None:
                continue

            if pc['ob_b_key'] in fired_ob_b_keys:
                continue

            bar_close = close[bar]
            bar_open = open_[bar]
            direction = pc['direction']

            if direction == 'long' and bar_close <= bar_open:
                continue
            if direction == 'short' and bar_close >= bar_open:
                continue

            ob_c_key = pc['ob_c_key']
            if ob_c_key is None:
                continue

            if direction == 'long':
                tp = pc['ob_c_bottom']
                if tp <= bar_close:
                    continue
            else:
                tp = pc['ob_c_top']
                if tp >= bar_close:
                    continue

            fired_ob_b_keys.add(pc['ob_b_key'])
            entry_time = pd.Timestamp(times[bar])
            sig = TradeSignal(
                timestamp_1h_sweep=entry_time,
                timestamp_5m_event_b=entry_time,
                timestamp_5m_validation=entry_time,
                timestamp_1m_confirmation=entry_time,
                timestamp_entry=entry_time,
                trend_1h_before_sweep='n/a',
                entry_direction=direction,
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
                stop_loss_price=pc['sl'],
            )
            ctx = self._context_from_pending(sig, pc, direction)
            sig.ob_a_id = (ctx.ob_a_top, ctx.ob_a_bottom, ctx.ob_a_start_idx)
            contexts.append(ctx)

        return contexts

    def _context_from_snapshot(self, sig: TradeSignal, snapshot: dict) -> Optional[ThreeOBSignalContext]:
        """Build ThreeOBSignalContext from a snapshot that has a direct signal."""
        direction = sig.entry_direction
        prefix = 'long' if direction == 'long' else 'short'

        ob_a_key = snapshot.get(f'{prefix}_ob_a_key')
        ob_b_key = snapshot.get(f'{prefix}_ob_b_key')
        ob_c_key = snapshot.get(f'{prefix}_ob_c_key')

        # Need all three OB keys to build context
        if ob_a_key is None or ob_b_key is None or ob_c_key is None:
            return None

        ob_a = self.ob_records.get(ob_a_key)
        ob_b = self.ob_records.get(ob_b_key)
        ob_c = self.ob_records.get(ob_c_key)
        if ob_a is None or ob_b is None or ob_c is None:
            return None

        return ThreeOBSignalContext(
            signal=sig,
            ob_a_top=ob_a.top, ob_a_bottom=ob_a.bottom, ob_a_start_idx=ob_a.start_idx,
            ob_b_top=ob_b.top, ob_b_bottom=ob_b.bottom, ob_b_start_idx=ob_b.start_idx,
            ob_c_top=ob_c.top, ob_c_bottom=ob_c.bottom, ob_c_start_idx=ob_c.start_idx,
            direction=direction,
        )

    def _context_from_pending(self, sig: TradeSignal, pc: dict, direction: str) -> ThreeOBSignalContext:
        """Build ThreeOBSignalContext from a pending_confirmation dict."""
        ob_a_key = pc.get('ob_a_key')
        ob_b_key = pc.get('ob_b_key')
        ob_c_key = pc.get('ob_c_key')

        ob_a_start = self.ob_records[ob_a_key].start_idx if ob_a_key and ob_a_key in self.ob_records else 0
        ob_b_start = self.ob_records[ob_b_key].start_idx if ob_b_key and ob_b_key in self.ob_records else 0
        ob_c_start = self.ob_records[ob_c_key].start_idx if ob_c_key and ob_c_key in self.ob_records else 0

        return ThreeOBSignalContext(
            signal=sig,
            ob_a_top=pc.get('ob_a_top', 0), ob_a_bottom=pc.get('ob_a_bottom', 0), ob_a_start_idx=ob_a_start,
            ob_b_top=pc.get('ob_b_top', 0), ob_b_bottom=pc.get('ob_b_bottom', 0), ob_b_start_idx=ob_b_start,
            ob_c_top=pc.get('ob_c_top', 0), ob_c_bottom=pc.get('ob_c_bottom', 0), ob_c_start_idx=ob_c_start,
            direction=direction,
        )
