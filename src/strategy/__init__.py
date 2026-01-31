"""
Strategy module for multi-timeframe trading analysis.

Re-exports all public classes for easy importing.

Usage:
    from src.strategy import MultiTimeframeStrategy, PartialSetup, TradeSignal

    # Or use the modular classes directly:
    from src.strategy import TimeframeManager, StrategyEngine
"""

import pandas as pd
from typing import List, Optional, Dict

from .models import StrategyState, PartialSetup, TradeSignal, SweepInfo, EquilibriumState, ExitTarget
from .timeframe_manager import TimeframeManager
from .liquidity_detector import LiquidityDetector
from .event_b_detector import EventBDetector
from .confirmation_detector import ConfirmationDetector
from .equilibrium_validator import EquilibriumValidator
from .exit_target_finder import ExitTargetFinder
from .gmm_zone_detector import GMMZoneDetector, GMMZoneInfo
from .strategy_engine import StrategyEngine
from .three_ob_strategy import ThreeOBStrategy
from .magnet_chase_engine import MagnetChaseEngine, MagnetChaseSignal, MagnetChaseResult
from .zone_proximity import (
    ZoneRecord, RankedZone,
    extract_zone_records, get_active_zones,
    rank_zones_by_proximity, select_target_zone,
)


class MultiTimeframeStrategy:
    """
    Convenience wrapper for the strategy engine.

    This class provides a simplified interface that handles TimeframeManager
    initialization internally. For more control, use StrategyEngine directly.
    """

    def __init__(
        self,
        df_high: pd.DataFrame,
        df_mid: pd.DataFrame,
        df_low: pd.DataFrame,
        timeframe_config: Optional[Dict[str, str]] = None,
        use_fvg_validation: bool = True,
        use_equilibrium_validation: bool = True,
        require_fvg_in_equilibrium: bool = False,
        sweep_proximity_threshold: float = 0.0,
        abandon_on_new_sweep: bool = True,
        gmm_config: Optional[Dict] = None,
        high_tf_lookback: int = 300,
        use_fvg_trigger: bool = False
    ):
        """
        Initialize strategy with multi-timeframe data.

        Args:
            df_high: High timeframe OHLCV data with 'time' column (e.g., 1H or 4H)
            df_mid: Mid timeframe OHLCV data with 'time' column (e.g., 5M or 15M)
            df_low: Low timeframe OHLCV data with 'time' column (e.g., 1M or 5M)
            timeframe_config: Optional dict with timeframe labels for display
                              e.g., {'high': '4h', 'mid': '15min', 'low': '5min'}
            use_fvg_validation: If True, check FVG respect in validation (takes priority).
                               If False, skip FVG validation.
            use_equilibrium_validation: If True, check equilibrium zone entry.
                                       If False, skip equilibrium validation.
            require_fvg_in_equilibrium: If True (OVERRIDES other settings), require
                                       an FVG that structurally overlaps with the
                                       equilibrium zone. This is the strictest mode.
            sweep_proximity_threshold: Percentage threshold (as decimal) for near-sweep detection.
                                      If price comes within this % of the level, it counts as swept.
                                      Default 0.0 = exact touch required.
            abandon_on_new_sweep: If True (default), abandons current setup if a new liquidity
                                 sweep occurs during mid/low TF scanning. If False, continues
                                 building the current setup regardless of new sweeps.
            gmm_config: Optional dictionary with GMM zone configuration:
                        - enabled: bool (default False)
                        - lookback_candles: int (default 300)
                        - step: float (default 0.1)
                        - max_components: int (default 3)
                        - recalc_interval: int (default 50)
                        - fib_levels: list (default [1.0, 0.786, ...])
                        - premium_zone: list (default [0.786, 1.0])
                        - discount_zone: list (default [0.0, 0.236])
                        - allow_middle_zone_trades: bool (default False)
                        - take_profit_method: str ('fib' or 'order_block', default 'fib')
            high_tf_lookback: Number of high TF candles to include before the overlap
                             period for better inflection point detection. Default 300.
            use_fvg_trigger: If True, FVG respect on high TF also triggers Stage 1.
                            Bullish FVG respected = LONG, Bearish = SHORT.
        """
        # Initialize timeframe manager
        self._tm = TimeframeManager(
            df_high, df_mid, df_low, timeframe_config,
            high_tf_lookback=high_tf_lookback
        )

        # Initialize strategy engine
        self._engine = StrategyEngine(
            self._tm,
            use_fvg_validation=use_fvg_validation,
            use_equilibrium_validation=use_equilibrium_validation,
            require_fvg_in_equilibrium=require_fvg_in_equilibrium,
            sweep_proximity_threshold=sweep_proximity_threshold,
            abandon_on_new_sweep=abandon_on_new_sweep,
            gmm_config=gmm_config,
            use_fvg_trigger=use_fvg_trigger
        )

        # Expose attributes for convenience
        self.df_high = self._tm.df_high
        self.df_mid = self._tm.df_mid
        self.df_low = self._tm.df_low
        self.df_high_original = self._tm.df_high_original
        self.df_mid_original = self._tm.df_mid_original
        self.df_low_original = self._tm.df_low_original

        # Expose overlap period
        self.overlap_start = self._tm.overlap_start
        self.overlap_end = self._tm.overlap_end
        self.tradable_start = self._tm.tradable_start

        # Expose mappings
        self.map_high_to_mid = self._tm.map_high_to_mid
        self.map_mid_to_low = self._tm.map_mid_to_low

        # Expose indicators
        self.swings_high = self._tm.swings_high
        self.liquidity_high = self._tm.liquidity_high
        self.inflexions_high = self._tm.inflexions_high
        self.bos_high = self._tm.bos_high
        self.fvg_high = self._tm.fvg_high
        self.swings_mid = self._tm.swings_mid
        self.inflexions_mid = self._tm.inflexions_mid
        self.bos_mid = self._tm.bos_mid
        self.fvg_mid = self._tm.fvg_mid
        self.ob_mid = self._tm.ob_mid
        self.inflexions_low = self._tm.inflexions_low
        self.bos_low = self._tm.bos_low
        self.fvg_low = self._tm.fvg_low

        # Expose timeframe config and durations
        self.timeframe_config = self._tm.timeframe_config
        self.high_duration = self._tm.high_duration
        self.mid_duration = self._tm.mid_duration
        self.low_duration = self._tm.low_duration

    def scan_for_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        """
        Scan historical data for complete trade setups.

        Args:
            max_signals: Maximum number of signals to find

        Returns:
            List of TradeSignal objects
        """
        return self._engine.scan_for_signals(max_signals)

    def get_high_trend_at_index(self, idx: int):
        """Get the trend at a specific high TF index."""
        return self._tm.get_high_trend_at_index(idx)

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

    def detect_all_liquidity_sweeps_high(self):
        """Detect ALL liquidity sweeps progressively."""
        return self._engine._liquidity_detector.detect_all_sweeps()

    def detect_bos_mid_opposite_direction(self, start_time, end_time, entry_direction):
        """Detect BOS on mid TF in the entry direction."""
        return self._engine._event_b_detector.detect_bos_opposite_direction(
            start_time, end_time, entry_direction
        )

    def detect_ifvg_mid(self, start_time, end_time, entry_direction):
        """Detect Inverse FVG on mid TF."""
        return self._engine._event_b_detector.detect_ifvg(
            start_time, end_time, entry_direction
        )

    def detect_final_confirmation_low(self, start_time, end_time, entry_direction):
        """Detect final confirmation on low TF."""
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

    @property
    def gmm_zone(self):
        """Current GMM zone detection result (if enabled)."""
        return self._engine.gmm_zone

    @property
    def gmm_enabled(self) -> bool:
        """Whether GMM zone detection is enabled."""
        return self._engine.gmm_enabled


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
    'GMMZoneDetector',
    'GMMZoneInfo',
    # Engine
    'StrategyEngine',
    # Convenience wrappers
    'MultiTimeframeStrategy',
    'ThreeOBStrategy',
    # Magnet Chase
    'MagnetChaseEngine',
    'MagnetChaseSignal',
    'MagnetChaseResult',
    'ZoneRecord',
    'RankedZone',
    'extract_zone_records',
    'get_active_zones',
    'rank_zones_by_proximity',
    'select_target_zone',
]
