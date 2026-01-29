"""
3-OB Strategy wrapper — single-timeframe interface analogous to MultiTimeframeStrategy.

Supports optional lower-timeframe BOS/IFVG confirmation via ConfirmationDetector.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from .models import TradeSignal
from .three_ob_engine import ThreeOBEngine
from .confirmation_detector import ConfirmationDetector


class ThreeOBStrategy:
    """
    Single-timeframe 3-OB strategy with optional lower TF confirmation.

    If df_confirmation is provided, State 3 uses BOS/IFVG on the lower TF
    instead of the simple green/red candle close check.

    Usage:
        # Old behavior (green/red close):
        strategy = ThreeOBStrategy(df, close_break=True, min_dist_pct=0.10)

        # New behavior (lower TF confirmation):
        strategy = ThreeOBStrategy(df, close_break=True, min_dist_pct=0.10,
                                   df_confirmation=df_5min, confirmation_timeframe='5min')

        signals = strategy.scan_for_signals(max_signals=5)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        close_break: bool = True,
        min_dist_pct: float = 0.10,
        enable_shorts: bool = False,
        df_confirmation: Optional[pd.DataFrame] = None,
        confirmation_timeframe: Optional[str] = None,
    ):
        self.df = df
        self._engine = ThreeOBEngine(df, close_break=close_break, min_dist_pct=min_dist_pct, enable_shorts=enable_shorts)

        # Expose the working DataFrame with datetime index (for Backtester compatibility)
        df_work = self._engine.df.copy()
        if 'time' in df_work.columns:
            df_work = df_work.set_index('time')
        self.df_low = df_work

        # Empty partial_setups for compatibility with visualizer
        self.partial_setups: list = []

        # Set up lower TF confirmation if provided
        self._confirmation_detector: Optional[ConfirmationDetector] = None
        self._df_confirmation: Optional[pd.DataFrame] = None
        self._confirmation_timeframe = confirmation_timeframe

        if df_confirmation is not None and confirmation_timeframe is not None:
            self._df_confirmation = df_confirmation.copy()
            # Compute SMC indicators on confirmation TF
            from src.indicators.smc_custom import smc_custom
            from smartmoneyconcepts import smc

            df_conf_work = self._df_confirmation.copy()
            if 'time' in df_conf_work.columns:
                df_conf_indexed = df_conf_work.set_index('time')
            else:
                df_conf_indexed = df_conf_work

            inflexions_conf = smc_custom.inflexion_points(df_conf_work)
            bos_conf = smc_custom.bos(df_conf_work, inflexions_conf, close_break=close_break)
            fvg_conf = smc.fvg(df_conf_work, join_consecutive=False)

            self._confirmation_detector = ConfirmationDetector(
                df_low=df_conf_indexed,
                bos_low=bos_conf,
                fvg_low=fvg_conf,
                inflexions_low=inflexions_conf,
            )
            print(f"  Lower TF confirmation enabled: {confirmation_timeframe} ({len(df_conf_work)} candles)")

    def scan_for_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        if self._confirmation_detector is None:
            # Fallback: use engine's built-in green/red close check
            return self._engine.scan_for_signals(max_signals=max_signals)

        # Use lower TF confirmation
        return self._scan_with_confirmation(max_signals=max_signals)

    def _scan_with_confirmation(self, max_signals: int = 10) -> List[TradeSignal]:
        """Scan using lower TF BOS/IFVG confirmation instead of green/red close."""
        signals: List[TradeSignal] = []
        engine_df = self._engine.df
        times = engine_df['time'].values
        high = engine_df['high'].values
        low = engine_df['low'].values

        # Track which pending_confirmation retouch bars we've already processed
        processed_retouch_bars = set()

        for bar, snapshot in self._engine.scan_with_snapshots(max_signals=max_signals * 5):
            # If the engine produced a signal via green/red close, skip it —
            # we want lower TF confirmation instead
            pending = snapshot.get('pending_confirmation')
            if pending is None:
                continue

            retouch_bar = pending['ob_b_retouch_bar']
            if retouch_bar in processed_retouch_bars:
                continue
            processed_retouch_bars.add(retouch_bar)

            direction = pending['direction']
            retouch_time = pd.Timestamp(times[retouch_bar])

            # Determine the confirmation window:
            # From retouch candle time to the close of the next higher-TF candle
            if retouch_bar + 1 < len(times):
                window_end = pd.Timestamp(times[retouch_bar + 1])
            else:
                window_end = retouch_time

            # Scan lower TF candles in this window
            confirmation = self._scan_confirmation_window(
                retouch_time, window_end, direction
            )

            if confirmation is not None:
                conf_time, conf_type, conf_price = confirmation
                # Build TP the same way the engine does
                ob_c_key = pending['ob_c_key']
                ob_c_bottom = pending['ob_c_bottom']
                ob_c_top = pending['ob_c_top']
                sl = pending['sl']

                if direction == 'long':
                    ob_c_rec = self._engine.ob_records.get(ob_c_key)
                    ob_c_start = ob_c_rec.start_idx if ob_c_rec else retouch_bar
                    lowest_low = np.min(low[ob_c_start:retouch_bar + 1]) if ob_c_start <= retouch_bar else low[retouch_bar]
                    tp = (lowest_low + ob_c_bottom) / 2.0
                    if tp <= conf_price:
                        continue
                else:
                    ob_c_rec = self._engine.ob_records.get(ob_c_key)
                    ob_c_start = ob_c_rec.start_idx if ob_c_rec else retouch_bar
                    highest_high = np.max(high[ob_c_start:retouch_bar + 1]) if ob_c_start <= retouch_bar else high[retouch_bar]
                    tp = (highest_high + ob_c_top) / 2.0
                    if tp >= conf_price:
                        continue

                sig = TradeSignal(
                    timestamp_1h_sweep=retouch_time,
                    timestamp_5m_event_b=retouch_time,
                    timestamp_5m_validation=retouch_time,
                    timestamp_1m_confirmation=conf_time,
                    timestamp_entry=conf_time,
                    trend_1h_before_sweep='n/a',
                    entry_direction=direction,
                    price_1h_sweep=conf_price,
                    price_5m_event_b=conf_price,
                    price_5m_validation=conf_price,
                    price_1m_confirmation=conf_price,
                    price_entry=conf_price,
                    condition_liquidity_sweep='3-OB State Machine',
                    condition_event_b='OB-A touch',
                    condition_validation='OB-C distance',
                    condition_confirmation=f'{conf_type} ({self._confirmation_timeframe})',
                    indices_1h=(retouch_bar, retouch_bar),
                    indices_5m=(retouch_bar, retouch_bar),
                    indices_1m=(retouch_bar, retouch_bar),
                    take_profit_price=tp,
                    stop_loss_price=sl,
                )
                signals.append(sig)
                if len(signals) >= max_signals:
                    break

        return signals

    def _scan_confirmation_window(
        self,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        direction: str,
    ):
        """Scan lower TF candles in [window_start, window_end] for BOS/IFVG confirmation."""
        df_conf = self._confirmation_detector.df_low
        mask = (df_conf.index >= window_start) & (df_conf.index <= window_end)
        candle_times = df_conf.index[mask]

        for t in candle_times:
            result = self._confirmation_detector.detect_confirmation_at_candle(t, direction)
            if result is not None:
                return result
        return None
