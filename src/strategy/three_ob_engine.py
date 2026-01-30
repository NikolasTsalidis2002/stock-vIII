"""
3-OB Strategy Engine — Python port of ob_rsi_strategy.pine

State machine (per setup):
  State 0: Detect OB A touch & leave + find opposite OB C with >min_dist_pct distance
  State 1: Wait for new OB B (same direction as A) formed after confirmation
  State 2: OB-B found, waiting for price to leave and retouch
  State 3: OB-B retouched, waiting for close break → entry signal

Multiple setups can progress concurrently for both long and short directions.
OBs are precomputed via smc_custom.ob() and looked up by StartIndex (stable keys).
"""

from dataclasses import dataclass, field
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


@dataclass
class Setup:
    """Tracks a single in-progress 3-OB setup through the state machine."""
    direction: str  # 'long' or 'short'
    state: int = 0  # 0-3

    ob_a_key: Optional[tuple] = None
    ob_a_bottom: Optional[float] = None
    ob_a_top: Optional[float] = None
    ob_a_confirmed_bar: Optional[int] = None

    ob_b_key: Optional[tuple] = None
    ob_b_bottom: Optional[float] = None
    ob_b_top: Optional[float] = None
    ob_b_retouched: bool = False
    ob_b_retouch_bar: Optional[int] = None

    ob_c_key: Optional[tuple] = None
    ob_c_bottom: Optional[float] = None
    ob_c_top: Optional[float] = None


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
        fired_ob_b_keys: set = set()
        for bar, snapshot in self.scan_with_snapshots(max_signals=max_signals):
            if snapshot['signal'] is not None:
                signals.append(snapshot['signal'])
                continue

            all_pendings = snapshot.get('pending_confirmations', [])
            if not all_pendings:
                pc = snapshot.get('pending_confirmation')
                if pc is not None:
                    all_pendings = [pc]

            bar_close = close[bar]
            bar_open = open_[bar]

            for pc in all_pendings:
                if pc['ob_b_key'] in fired_ob_b_keys:
                    continue

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
                signals.append(sig)
        return signals

    def scan_with_snapshots(self, max_signals: int = 10):
        """
        Yield (bar_index, snapshot_dict) for each bar.

        Maintains multiple concurrent setups (long and short) so that
        overlapping OB-A touches can progress through the state machine
        in parallel.
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

        # Multiple concurrent setups
        long_setups: List[Setup] = []
        short_setups: List[Setup] = []

        # Prevent duplicate signals from the same OB-B
        fired_ob_b_keys: set = set()

        reset_cmd = None
        for bar in range(n):
            if len(signals) >= max_signals:
                break

            # Handle reset commands sent via .send()
            if reset_cmd:
                if reset_cmd in ('reset_long', 'reset_both'):
                    long_setups.clear()
                if reset_cmd in ('reset_short', 'reset_both'):
                    short_setups.clear()
                reset_cmd = None

            bar_low = low[bar]
            bar_high = high[bar]
            bar_close = close[bar]
            bar_open = open_[bar]

            long_status = ""
            short_status = ""

            # --- Step 1: Activate new OBs at their BOS bar ---
            if bar in self.bos_to_obs:
                for start_key in self.bos_to_obs[bar]:
                    if start_key in self.ob_records:
                        active_obs[start_key] = self.ob_records[start_key]

            # --- Step 2: Check invalidation & advance setups ---
            to_remove: List[tuple] = []

            for key, ob in active_obs.items():
                invalidated = (ob.status_idx != 0 and not ob.respected and bar >= ob.status_idx)

                if invalidated:
                    # Remove any setup referencing this OB
                    long_setups = [s for s in long_setups if not self._setup_uses_ob(s, key)]
                    short_setups = [s for s in short_setups if not self._setup_uses_ob(s, key)]
                    to_remove.append(key)
                    continue

                # OB still alive — check touch
                touching = bar_low <= ob.top and bar_high >= ob.bottom

                was_away = key in ob_has_left
                if not was_away and not touching:
                    ob_has_left.add(key)

                # --- LONG strategy (bullish OBs) ---
                if ob.direction == 1:
                    # State 0 → 1: any bullish OB retouched can start a NEW setup
                    retouched = was_away and touching
                    if retouched:
                        # Don't start a new setup if this OB-A is already tracked in an active long setup
                        already_tracked = any(s.ob_a_key == key for s in long_setups)
                        if not already_tracked:
                            candidates = []
                            for j_key, ob_j in active_obs.items():
                                if ob_j.direction != -1 or j_key in to_remove:
                                    continue
                                if ob_j.bottom > bar_close:
                                    dist = ob_j.bottom - bar_close
                                    dist_pct = dist / bar_close if bar_close > 0 else 0
                                    candidates.append((dist, dist_pct, j_key, ob_j.bottom, ob_j.top))
                            candidates.sort()

                            chosen = None
                            for dist, dist_pct, c_key, c_bot, c_top in candidates:
                                if dist_pct > self.min_dist_pct:
                                    chosen = (dist_pct, c_key, c_bot, c_top)
                                    break

                            if chosen:
                                best_dist_pct, best_bear_key, best_bear_bot, best_bear_top = chosen
                                new_setup = Setup(
                                    direction='long', state=1,
                                    ob_a_key=key, ob_a_bottom=ob.bottom, ob_a_top=ob.top,
                                    ob_a_confirmed_bar=bar,
                                    ob_c_key=best_bear_key, ob_c_bottom=best_bear_bot, ob_c_top=best_bear_top,
                                )
                                long_setups.append(new_setup)
                                long_status = f"[State 0→1] OB-A (bull ${ob.bottom:.2f}–${ob.top:.2f}) confirmed, OB-C found {best_dist_pct*100:.1f}% above"

                    # Advance existing long setups that reference this OB
                    for s in long_setups:
                        if s.state == 1 and key != s.ob_a_key and ob.start_idx > s.ob_a_confirmed_bar:
                            # Only pick OB-B if setup doesn't already have one
                            if s.ob_b_key is None:
                                s.state = 2
                                s.ob_b_key = key
                                s.ob_b_bottom = ob.bottom
                                s.ob_b_top = ob.top
                                long_status = f"[State 1→2] OB-B (bull ${ob.bottom:.2f}–${ob.top:.2f}) found"

                        elif s.state == 2 and key == s.ob_b_key:
                            b_was_away = s.ob_b_key in ob_has_left
                            if b_was_away and touching:
                                s.ob_b_retouched = True
                                s.ob_b_retouch_bar = bar
                                s.state = 3
                                long_status = f"[State 2→3] OB-B (${s.ob_b_bottom:.2f}–${s.ob_b_top:.2f}) retouched"

                # --- Invalidate long setups if price touches their OB-C ---
                if ob.direction == -1:
                    new_long = []
                    for s in long_setups:
                        if s.ob_c_key == key and touching:
                            long_status = f"OB-C reached — invalidating a long setup"
                        else:
                            new_long.append(s)
                    long_setups = new_long

                # --- SHORT strategy (bearish OBs) ---
                if self.enable_shorts and ob.direction == -1:
                    retouched = was_away and touching
                    if retouched:
                        already_tracked = any(s.ob_a_key == key for s in short_setups)
                        if not already_tracked:
                            candidates = []
                            for j_key, ob_j in active_obs.items():
                                if ob_j.direction != 1 or j_key in to_remove:
                                    continue
                                if ob_j.top < bar_close:
                                    dist = bar_close - ob_j.top
                                    dist_pct = dist / bar_close if bar_close > 0 else 0
                                    candidates.append((dist, dist_pct, j_key, ob_j.top, ob_j.bottom))
                            candidates.sort()

                            chosen = None
                            for dist, dist_pct, c_key, c_top, c_bot in candidates:
                                if dist_pct > self.min_dist_pct:
                                    chosen = (dist_pct, c_key, c_top, c_bot)
                                    break

                            if chosen:
                                best_dist_pct, best_bull_key, best_bull_top, best_bull_bot = chosen
                                new_setup = Setup(
                                    direction='short', state=1,
                                    ob_a_key=key, ob_a_top=ob.top, ob_a_bottom=ob.bottom,
                                    ob_a_confirmed_bar=bar,
                                    ob_c_key=best_bull_key, ob_c_top=best_bull_top, ob_c_bottom=best_bull_bot,
                                )
                                short_setups.append(new_setup)
                                short_status = f"[State 0→1] OB-A (bear ${ob.bottom:.2f}–${ob.top:.2f}) confirmed"

                    for s in short_setups:
                        if s.state == 1 and key != s.ob_a_key and ob.start_idx > s.ob_a_confirmed_bar:
                            if s.ob_b_key is None:
                                s.state = 2
                                s.ob_b_key = key
                                s.ob_b_bottom = ob.bottom
                                s.ob_b_top = ob.top
                                short_status = f"[State 1→2] OB-B (bear ${ob.bottom:.2f}–${ob.top:.2f}) found"

                        elif s.state == 2 and key == s.ob_b_key:
                            b_was_away = s.ob_b_key in ob_has_left
                            if b_was_away and touching:
                                s.ob_b_retouched = True
                                s.ob_b_retouch_bar = bar
                                s.state = 3
                                short_status = f"[State 2→3] OB-B (${s.ob_b_bottom:.2f}–${s.ob_b_top:.2f}) retouched"

                # --- Invalidate short setups if price touches their OB-C ---
                if self.enable_shorts and ob.direction == 1:
                    new_short = []
                    for s in short_setups:
                        if s.ob_c_key == key and touching:
                            short_status = f"OB-C reached — invalidating a short setup"
                        else:
                            new_short.append(s)
                    short_setups = new_short

            # Remove invalidated OBs
            for key in to_remove:
                del active_obs[key]
                ob_has_left.discard(key)
                dead_obs.append(key)

            # --- Build pending_confirmations (one per state-3 setup) ---
            bar_signal = None
            pending_confirmations = []

            for s in long_setups:
                if s.state == 3 and s.ob_b_retouched and s.ob_b_key not in fired_ob_b_keys:
                    pending_confirmations.append({
                        'direction': 'long',
                        'ob_b_retouch_bar': s.ob_b_retouch_bar,
                        'ob_a_bottom': s.ob_a_bottom,
                        'ob_a_top': s.ob_a_top,
                        'ob_a_key': s.ob_a_key,
                        'ob_a_confirmed_bar': s.ob_a_confirmed_bar,
                        'ob_b_bottom': s.ob_b_bottom,
                        'ob_b_top': s.ob_b_top,
                        'ob_b_key': s.ob_b_key,
                        'ob_c_bottom': s.ob_c_bottom,
                        'ob_c_top': s.ob_c_top,
                        'ob_c_key': s.ob_c_key,
                        'sl': s.ob_b_bottom,
                    })

            for s in short_setups:
                if s.state == 3 and s.ob_b_retouched and s.ob_b_key not in fired_ob_b_keys:
                    pending_confirmations.append({
                        'direction': 'short',
                        'ob_b_retouch_bar': s.ob_b_retouch_bar,
                        'ob_a_bottom': s.ob_a_bottom,
                        'ob_a_top': s.ob_a_top,
                        'ob_a_key': s.ob_a_key,
                        'ob_a_confirmed_bar': s.ob_a_confirmed_bar,
                        'ob_b_bottom': s.ob_b_bottom,
                        'ob_b_top': s.ob_b_top,
                        'ob_b_key': s.ob_b_key,
                        'ob_c_bottom': s.ob_c_bottom,
                        'ob_c_top': s.ob_c_top,
                        'ob_c_key': s.ob_c_key,
                        'sl': s.ob_b_top,
                    })

            # For backward compatibility, pick the first pending as 'pending_confirmation'
            pending_confirmation = pending_confirmations[0] if pending_confirmations else None

            # Use first long/short setup for snapshot backward compat
            first_long = long_setups[0] if long_setups else None
            first_short = short_setups[0] if short_setups else None

            # --- Build snapshot ---
            snapshot = {
                'long_state': first_long.state if first_long else 0,
                'short_state': first_short.state if first_short else 0,
                'long_ob_a_key': first_long.ob_a_key if first_long else None,
                'long_ob_b_key': first_long.ob_b_key if first_long else None,
                'long_ob_c_key': first_long.ob_c_key if first_long else None,
                'short_ob_a_key': first_short.ob_a_key if first_short else None,
                'short_ob_b_key': first_short.ob_b_key if first_short else None,
                'short_ob_c_key': first_short.ob_c_key if first_short else None,
                'active_ob_keys': list(active_obs.keys()),
                'dead_ob_keys': list(dead_obs),
                'signal': bar_signal,
                'signals_so_far': len(signals),
                'long_status': long_status,
                'short_status': short_status,
                'pending_confirmation': pending_confirmation,
                'pending_confirmations': pending_confirmations,
                'long_setups_count': len(long_setups),
                'short_setups_count': len(short_setups),
            }
            reset_cmd = yield bar, snapshot

    @staticmethod
    def _setup_uses_ob(setup: 'Setup', key: tuple) -> bool:
        """Check if a setup references the given OB key as A, B, or C."""
        return key in (setup.ob_a_key, setup.ob_b_key, setup.ob_c_key)

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
            if snapshot['signal'] is not None:
                sig = snapshot['signal']
                ctx = self._context_from_snapshot(sig, snapshot)
                if ctx is not None:
                    sig.ob_a_id = (ctx.ob_a_top, ctx.ob_a_bottom, ctx.ob_a_start_idx)
                    contexts.append(ctx)
                continue

            all_pendings = snapshot.get('pending_confirmations', [])
            if not all_pendings:
                pc = snapshot.get('pending_confirmation')
                if pc is not None:
                    all_pendings = [pc]

            bar_close = close[bar]
            bar_open = open_[bar]

            for pc in all_pendings:
                if pc['ob_b_key'] in fired_ob_b_keys:
                    continue

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
