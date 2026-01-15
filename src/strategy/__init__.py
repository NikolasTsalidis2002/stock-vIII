"""
Strategy module for multi-timeframe trading analysis.

Re-exports all public classes for easy importing.

Usage:
    from src.strategy import MultiTimeframeStrategy, PartialSetup, TradeSignal

    # Or use the modular classes directly:
    from src.strategy import TimeframeManager, StrategyEngine
"""

import pandas as pd
from typing import List

from .models import StrategyState, PartialSetup, TradeSignal, SweepInfo, EquilibriumState, ExitTarget
from .timeframe_manager import TimeframeManager
from .liquidity_detector import LiquidityDetector
from .event_b_detector import EventBDetector
from .confirmation_detector import ConfirmationDetector
from .equilibrium_validator import EquilibriumValidator
from .exit_target_finder import ExitTargetFinder
from .strategy_engine import StrategyEngine


class MultiTimeframeStrategy:
    """
    Convenience wrapper for the strategy engine.

    This class provides a simplified interface that handles TimeframeManager
    initialization internally. For more control, use StrategyEngine directly.
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
            max_price_deviation_percent: Maximum % price can move from sweep before invalidating setup
        """
        # Store invalidation threshold
        self.max_price_deviation_percent = max_price_deviation_percent

        # Initialize timeframe manager
        self._tm = TimeframeManager(df_1h, df_5m, df_1m)

        # Initialize strategy engine
        self._engine = StrategyEngine(self._tm, max_price_deviation_percent)

        # Expose attributes for convenience
        self.df_1h = self._tm.df_1h
        self.df_5m = self._tm.df_5m
        self.df_1m = self._tm.df_1m
        self.df_1h_original = self._tm.df_1h_original
        self.df_5m_original = self._tm.df_5m_original
        self.df_1m_original = self._tm.df_1m_original

        # Expose overlap period
        self.overlap_start = self._tm.overlap_start
        self.overlap_end = self._tm.overlap_end

        # Expose mappings
        self.map_1h_to_5m = self._tm.map_1h_to_5m
        self.map_5m_to_1m = self._tm.map_5m_to_1m

        # Expose indicators
        self.swings_1h = self._tm.swings_1h
        self.liquidity_1h = self._tm.liquidity_1h
        self.inflexions_1h = self._tm.inflexions_1h
        self.bos_1h = self._tm.bos_1h
        self.swings_5m = self._tm.swings_5m
        self.inflexions_5m = self._tm.inflexions_5m
        self.bos_5m = self._tm.bos_5m
        self.fvg_5m = self._tm.fvg_5m
        self.ob_5m = self._tm.ob_5m
        self.inflexions_1m = self._tm.inflexions_1m
        self.bos_1m = self._tm.bos_1m
        self.fvg_1m = self._tm.fvg_1m

    def scan_for_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        """
        Scan historical data for complete trade setups.

        Args:
            max_signals: Maximum number of signals to find

        Returns:
            List of TradeSignal objects
        """
        return self._engine.scan_for_signals(max_signals)

    def get_1h_trend_at_index(self, idx: int):
        """Get the trend at a specific 1H index."""
        return self._tm.get_1h_trend_at_index(idx)

    def get_market_close_for_day(self, timestamp):
        """Get the actual market close time for the trading day."""
        return self._tm.get_market_close_for_day(timestamp)

    def is_sweep_broken_at_time(self, inflexion_idx, end_time, current_candle_idx=0):
        """Check if a sweep has been broken up to a specific time."""
        return self._engine._liquidity_detector.is_sweep_broken_at_time(
            inflexion_idx, end_time, current_candle_idx
        )

    def is_price_moved_too_far(self, current_price, sweep_price, entry_direction):
        """Check if price has moved too far in the intended direction."""
        return self._engine._liquidity_detector.is_price_moved_too_far(
            current_price, sweep_price, entry_direction
        )

    def detect_all_liquidity_sweeps_1h(self):
        """Detect ALL liquidity sweeps progressively."""
        return self._engine._liquidity_detector.detect_all_sweeps()

    def detect_bos_5m_opposite_direction(self, start_time, end_time, entry_direction):
        """Detect BOS on 5M in the entry direction."""
        return self._engine._event_b_detector.detect_bos_opposite_direction(
            start_time, end_time, entry_direction
        )

    def detect_ifvg_5m(self, start_time, end_time, entry_direction):
        """Detect Inverse FVG on 5M."""
        return self._engine._event_b_detector.detect_ifvg(
            start_time, end_time, entry_direction
        )

    def detect_final_confirmation_1m(self, start_time, end_time, entry_direction):
        """Detect final confirmation on 1M."""
        return self._engine._confirmation_detector.detect_final_confirmation(
            start_time, end_time, entry_direction
        )

    @property
    def signals(self) -> List[TradeSignal]:
        """All detected trade signals."""
        return self._engine.signals

    @property
    def partial_setups(self) -> List[PartialSetup]:
        """All partial setups (incomplete trades)."""
        return self._engine.partial_setups


__all__ = [
    # Models
    'StrategyState',
    'PartialSetup',
    'TradeSignal',
    'SweepInfo',
    'EquilibriumState',
    'ExitTarget',
    # Managers
    'TimeframeManager',
    # Detectors
    'LiquidityDetector',
    'EventBDetector',
    'ConfirmationDetector',
    'EquilibriumValidator',
    'ExitTargetFinder',
    # Engine
    'StrategyEngine',
    # Convenience wrapper
    'MultiTimeframeStrategy',
]
