"""
Strategy Sweep Visualizer

Unified four-tab visualization for liquidity sweep analysis.
Generates a single HTML file with 4 static tabs showing fixed snapshots
of the complete trade at different stages.

Features:
    - Sweep dropdown to select any detected sweep
    - Four timeframe tabs (1H, 5M Event B, 5M Validation, 1M)
    - Static snapshots instead of frame-by-frame animation
    - SMC indicators: FVG zones, OB zones, BOS lines, liquidity levels
    - Event B highlighting (BOS or IFVG)
    - Equilibrium line for validation
    - TP/SL lines on 1M tab only
    - Disabled tabs for partial setups

Usage:
    from src.strategy_visualizer import StrategySweepVisualizer

    visualizer = StrategySweepVisualizer(strategy, symbol='TSLA')
    output_path = visualizer.run()
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.core import (
    NumpyEncoder,
    generate_candle_data,
    generate_fvg_zones,
    generate_ob_zones,
    generate_bos_lines,
    generate_liquidity_lines,
)

from src.strategy.models import TradeSignal, PartialSetup
from src.indicators.smc_custom import smc_custom
from src.indicators.smc import smc
from src.backtesting.models import TradeResult, TradeOutcome, ExitType, SkippedTrade, SkipReason, SKIP_REASON_MESSAGES, PerformanceMetrics


@dataclass
class SweepEntry:
    """Represents a single sweep for the dropdown."""
    timestamp: datetime
    outcome: str  # "SUCCESS (+$X)", "FAILED (No Event B)", etc.
    is_complete: bool  # True if TradeSignal, False if PartialSetup
    signal: Optional[TradeSignal]
    partial: Optional[PartialSetup]
    entry_direction: str
    conditions_met: int

    @property
    def dropdown_text(self) -> str:
        """Format for dropdown display."""
        ts_str = self.timestamp.strftime('%Y-%m-%d %H:%M')
        return f"{ts_str} - {self.outcome}"


class StrategySweepVisualizer:
    """
    Unified four-tab strategy visualizer for liquidity sweep analysis.

    Generates a single HTML file with 4 static tabs showing fixed snapshots
    of the complete trade at different stages.
    """

    def __init__(
        self,
        strategy,  # MultiTimeframeStrategy - has timeframe_manager, signals, partial_setups
        symbol: str = 'TSLA',
        output_dir: Optional[Path] = None,
        trade_results: Optional[List['TradeResult']] = None,
        skipped_trades: Optional[List['SkippedTrade']] = None,
        performance_metrics: Optional['PerformanceMetrics'] = None
    ):
        """
        Initialize visualizer.

        Args:
            strategy: MultiTimeframeStrategy instance with signals and partial_setups
            symbol: Stock symbol for titles
            output_dir: Custom output directory (defaults to results/)
            trade_results: Optional list of TradeResult objects from backtesting
            skipped_trades: Optional list of SkippedTrade objects for trades that weren't executed
            performance_metrics: Optional PerformanceMetrics from backtesting for Dashboard tab
        """
        self.strategy = strategy
        self.symbol = symbol.upper()
        self.trade_results = trade_results or []
        self.skipped_trades = skipped_trades or []
        self.performance_metrics = performance_metrics

        # Build lookup from signal timestamp to trade result for efficient access
        self._trade_result_lookup: Dict[datetime, TradeResult] = {}
        for result in self.trade_results:
            if result.entry_time:
                self._trade_result_lookup[result.entry_time] = result

        # Build lookup from signal entry time to skipped trade for efficient access
        self._skip_reason_lookup: Dict[datetime, SkippedTrade] = {}
        for skipped in self.skipped_trades:
            if skipped.signal_entry_time:
                self._skip_reason_lookup[skipped.signal_entry_time] = skipped

        if output_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            self.output_dir = project_root / 'results/visualizations'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get timeframe labels from strategy
        self.tf_labels = strategy.timeframe_config or {'high': '1H', 'mid': '5M', 'low': '1M'}

        # Store pre-calculated indicators from strategy
        self.df_high = strategy.df_high
        self.df_mid = strategy.df_mid
        self.df_low = strategy.df_low

        # Indicators
        self.inflexions_high = strategy.inflexions_high
        self.bos_high = strategy.bos_high
        self.inflexions_mid = strategy.inflexions_mid
        self.bos_mid = strategy.bos_mid
        self.fvg_mid = strategy.fvg_mid
        self.ob_mid = strategy.ob_mid
        self.inflexions_low = strategy.inflexions_low
        self.bos_low = strategy.bos_low
        self.fvg_low = strategy.fvg_low

        # Timeframe durations
        self.high_duration = strategy.high_duration
        self.mid_duration = strategy.mid_duration
        self.low_duration = strategy.low_duration

    def _build_sweep_list(self) -> List[SweepEntry]:
        """
        Build unified list of sweeps from signals and partial_setups.

        Returns:
            List of SweepEntry sorted by timestamp_1h_sweep
        """
        sweep_list = []

        # Add complete signals
        for signal in self.strategy.signals:
            outcome = "SUCCESS"
            if hasattr(signal, 'pnl') and signal.pnl is not None:
                outcome = f"SUCCESS (+${signal.pnl:.2f})"

            sweep_list.append(SweepEntry(
                timestamp=signal.timestamp_1h_sweep,
                outcome=outcome,
                is_complete=True,
                signal=signal,
                partial=None,
                entry_direction=signal.entry_direction,
                conditions_met=4
            ))

        # Add partial setups
        for partial in self.strategy.partial_setups:
            outcome = f"FAILED ({partial.failure_reason})"

            sweep_list.append(SweepEntry(
                timestamp=partial.timestamp_1h_sweep,
                outcome=outcome,
                is_complete=False,
                signal=None,
                partial=partial,
                entry_direction=partial.entry_direction,
                conditions_met=partial.conditions_met
            ))

        # Sort by timestamp
        sweep_list.sort(key=lambda x: x.timestamp)

        return sweep_list

    def _get_trade_end_time(self, sweep_entry: SweepEntry, trade_result: Optional[TradeResult] = None) -> datetime:
        """Get the end time for the trade (for static snapshot range).

        Args:
            sweep_entry: The sweep entry being visualized
            trade_result: Optional TradeResult from backtesting with actual exit time

        Returns:
            End time for the chart display range
        """
        # If we have actual backtest exit time, use that + buffer to show full trade
        if trade_result and trade_result.exit_time:
            return trade_result.exit_time + timedelta(minutes=30)

        # Fallback to existing logic for trades without backtest results
        if sweep_entry.is_complete:
            signal = sweep_entry.signal
            # Use confirmation time + some buffer for complete trades
            return signal.timestamp_1m_confirmation + timedelta(hours=2)
        else:
            partial = sweep_entry.partial
            # Use the furthest timestamp available
            if partial.timestamp_1m_confirmation:
                return partial.timestamp_1m_confirmation + timedelta(hours=2)
            elif partial.timestamp_5m_validation:
                return partial.timestamp_5m_validation + timedelta(hours=2)
            elif partial.timestamp_5m_event_b:
                return partial.timestamp_5m_event_b + timedelta(hours=2)
            else:
                return partial.timestamp_1h_sweep + timedelta(hours=6)

    def _generate_tab_1h(self, sweep_entry: SweepEntry, trade_result: Optional[TradeResult] = None) -> Dict:
        """
        Generate 1H timeframe tab data (static snapshot).

        Range: Start of data to end of trade
        Purpose: Show liquidity sweep and overall market structure
        """
        sweep_time = sweep_entry.timestamp
        end_time = self._get_trade_end_time(sweep_entry, trade_result)

        # Find sweep index
        sweep_idx = self.df_high.index.get_indexer([sweep_time], method='nearest')[0]

        # Show context: 50 candles before sweep and up to end of trade
        start_idx = max(0, sweep_idx - 50)

        # Find end index closest to end_time
        end_mask = self.df_high.index <= end_time
        if end_mask.any():
            end_idx = self.df_high.index.get_indexer([end_time], method='nearest')[0]
        else:
            end_idx = len(self.df_high) - 1

        end_idx = min(len(self.df_high) - 1, end_idx + 10)  # Add some buffer

        df_slice = self.df_high.iloc[start_idx:end_idx + 1]

        if len(df_slice) == 0:
            return None

        # Calculate indicators on slice
        inflexions_slice = smc_custom.inflexion_points(df_slice)
        bos_slice = smc_custom.bos(df_slice, inflexions_slice, close_break=True)
        fvg_slice = smc.fvg(df_slice, join_consecutive=True)
        ob_slice = smc_custom.ob(df_slice, bos_slice, inflexions_slice)

        # Generate data using shared visualization functions
        candle_data = generate_candle_data(df_slice)
        fvg_zones = generate_fvg_zones(df_slice, fvg_slice)
        ob_zones = generate_ob_zones(df_slice, ob_slice)
        bos_lines = generate_bos_lines(df_slice, bos_slice, inflexions_slice)
        liquidity_lines = generate_liquidity_lines(df_slice, inflexions_slice)

        # Highlight sweep candle
        sweep_marker = {
            'time': int(sweep_time.timestamp()),
            'position': 'aboveBar' if sweep_entry.entry_direction == 'short' else 'belowBar',
            'color': '#ff0000',
            'shape': 'arrowDown' if sweep_entry.entry_direction == 'short' else 'arrowUp',
            'text': 'SWEEP'
        }

        # Calculate focus window around sweep (for context window feature)
        focus_time = int(sweep_time.timestamp())

        return {
            'timeframe': self.tf_labels.get('high', '1H'),
            'tabName': self.tf_labels.get('high', '1H'),
            'candleData': candle_data,
            'fvgZones': fvg_zones,
            'obZones': ob_zones,
            'bosLines': bos_lines,
            'liquidityLines': liquidity_lines,
            'sweepMarker': sweep_marker,
            'markers': [sweep_marker],
            'equilibriumLine': None,
            'tpLine': None,
            'slLine': None,
            'focusTime': focus_time,
            'contextCandles': 30  # Show 30 candles on each side of focus
        }

    def _generate_tab_5m_event_b(self, sweep_entry: SweepEntry, trade_result: Optional[TradeResult] = None) -> Optional[Dict]:
        """
        Generate 5M Event B tab data (static snapshot).

        Range: 1H liquidity sweep timestamp to end of trade
        Purpose: Show what BOS or IFVG triggered Event B
        """
        # Check if Event B was found
        if sweep_entry.is_complete:
            event_b_time = sweep_entry.signal.timestamp_5m_event_b
            event_b_type = sweep_entry.signal.condition_event_b
        elif sweep_entry.partial and sweep_entry.partial.timestamp_5m_event_b:
            event_b_time = sweep_entry.partial.timestamp_5m_event_b
            event_b_type = sweep_entry.partial.condition_event_b
        else:
            return None  # No Event B found

        sweep_time = sweep_entry.timestamp
        end_time = self._get_trade_end_time(sweep_entry, trade_result)

        # Filter data from sweep to end
        mask = (self.df_mid.index >= sweep_time) & (self.df_mid.index <= end_time)
        df_slice = self.df_mid.loc[mask]

        if len(df_slice) == 0:
            return None

        # Calculate indicators on slice
        inflexions_slice = smc_custom.inflexion_points(df_slice)
        bos_slice = smc_custom.bos(df_slice, inflexions_slice, close_break=True)
        fvg_slice = smc.fvg(df_slice, join_consecutive=False)
        ob_slice = smc_custom.ob(df_slice, bos_slice, inflexions_slice)

        # Generate data with highlighting for Event B
        highlight_time = event_b_time if event_b_type == 'BOS' else None
        fvg_highlight_time = event_b_time if event_b_type == 'IFVG' else None

        candle_data = generate_candle_data(df_slice)
        fvg_zones = generate_fvg_zones(df_slice, fvg_slice, highlight_time=fvg_highlight_time)
        ob_zones = generate_ob_zones(df_slice, ob_slice)
        bos_lines = generate_bos_lines(df_slice, bos_slice, inflexions_slice, highlight_time=highlight_time)
        liquidity_lines = generate_liquidity_lines(df_slice, inflexions_slice)

        # Event B marker
        event_b_marker = {
            'time': int(event_b_time.timestamp()),
            'position': 'aboveBar',
            'color': '#ffff00',  # Yellow highlight
            'shape': 'circle',
            'text': f'EB:{event_b_type}'
        }

        # Focus on Event B time
        focus_time = int(event_b_time.timestamp())

        return {
            'timeframe': self.tf_labels.get('mid', '5M'),
            'tabName': f"{self.tf_labels.get('mid', '5M')} Event B",
            'candleData': candle_data,
            'fvgZones': fvg_zones,
            'obZones': ob_zones,
            'bosLines': bos_lines,
            'liquidityLines': liquidity_lines,
            'sweepMarker': None,
            'markers': [event_b_marker],
            'eventBType': event_b_type,
            'eventBTime': int(event_b_time.timestamp()),
            'equilibriumLine': None,
            'tpLine': None,
            'slLine': None,
            'focusTime': focus_time,
            'contextCandles': 40  # Show 40 candles on each side for 5M
        }

    def _generate_tab_5m_validation(self, sweep_entry: SweepEntry, trade_result: Optional[TradeResult] = None) -> Optional[Dict]:
        """
        Generate 5M Validation tab data (static snapshot).

        Range: Exit Order Block start to end of trade
        Purpose: Show FVG respect or equilibrium zone entry
        """
        # Check if validation was found
        if sweep_entry.is_complete:
            validation_time = sweep_entry.signal.timestamp_5m_validation
            validation_type = sweep_entry.signal.condition_validation
            equilibrium_level = sweep_entry.signal.equilibrium_level
            exit_ob_start_idx = sweep_entry.signal.exit_ob_start_idx
        elif sweep_entry.partial and sweep_entry.partial.timestamp_5m_validation:
            validation_time = sweep_entry.partial.timestamp_5m_validation
            validation_type = sweep_entry.partial.condition_validation
            equilibrium_level = None  # Partial setups may not have this
            exit_ob_start_idx = None
        else:
            return None  # No validation found

        sweep_time = sweep_entry.timestamp
        end_time = self._get_trade_end_time(sweep_entry, trade_result)

        # Determine start time (exit OB start or sweep time)
        if exit_ob_start_idx is not None and exit_ob_start_idx < len(self.df_mid):
            start_time = self.df_mid.index[exit_ob_start_idx]
        else:
            start_time = sweep_time

        # Filter data
        mask = (self.df_mid.index >= start_time) & (self.df_mid.index <= end_time)
        df_slice = self.df_mid.loc[mask]

        if len(df_slice) == 0:
            return None

        # Calculate indicators on slice
        inflexions_slice = smc_custom.inflexion_points(df_slice)
        bos_slice = smc_custom.bos(df_slice, inflexions_slice, close_break=True)
        fvg_slice = smc.fvg(df_slice, join_consecutive=False)
        ob_slice = smc_custom.ob(df_slice, bos_slice, inflexions_slice)

        # Generate data using shared visualization functions
        candle_data = generate_candle_data(df_slice)
        fvg_zones = generate_fvg_zones(df_slice, fvg_slice)
        ob_zones = generate_ob_zones(df_slice, ob_slice)
        bos_lines = generate_bos_lines(df_slice, bos_slice, inflexions_slice)
        liquidity_lines = generate_liquidity_lines(df_slice, inflexions_slice)

        # Validation marker
        validation_marker = {
            'time': int(validation_time.timestamp()),
            'position': 'aboveBar',
            'color': '#9c27b0',  # Purple
            'shape': 'circle',
            'text': f'VAL:{validation_type}'
        }

        # Equilibrium line (if applicable)
        equilibrium_line = None
        if validation_type == 'Equilibrium' and equilibrium_level is not None:
            equilibrium_line = {
                'price': float(equilibrium_level),
                'color': '#9c27b0',  # Purple
                'lineWidth': 2,
                'label': f'EQ: ${equilibrium_level:.2f}'
            }

        # Focus on validation time
        focus_time = int(validation_time.timestamp())

        return {
            'timeframe': self.tf_labels.get('mid', '5M'),
            'tabName': f"{self.tf_labels.get('mid', '5M')} Validation",
            'candleData': candle_data,
            'fvgZones': fvg_zones,
            'obZones': ob_zones,
            'bosLines': bos_lines,
            'liquidityLines': liquidity_lines,
            'sweepMarker': None,
            'markers': [validation_marker],
            'validationType': validation_type,
            'equilibriumLine': equilibrium_line,
            'tpLine': None,
            'slLine': None,
            'focusTime': focus_time,
            'contextCandles': 40  # Show 40 candles on each side for 5M
        }

    def _generate_tab_1m(self, sweep_entry: SweepEntry, trade_result: Optional[TradeResult] = None) -> Optional[Dict]:
        """
        Generate 1M tab data (static snapshot).

        Range: 1H liquidity sweep timestamp to end of trade
        Purpose: Show trade execution with TP/SL levels
        """
        # Check if confirmation was found
        if sweep_entry.is_complete:
            confirmation_time = sweep_entry.signal.timestamp_1m_confirmation
            tp_price = sweep_entry.signal.take_profit_price
            sl_price = sweep_entry.signal.stop_loss_price
        elif sweep_entry.partial and sweep_entry.partial.timestamp_1m_confirmation:
            confirmation_time = sweep_entry.partial.timestamp_1m_confirmation
            tp_price = None
            sl_price = None
        else:
            return None  # No confirmation found

        sweep_time = sweep_entry.timestamp
        end_time = self._get_trade_end_time(sweep_entry, trade_result)

        # Filter data
        mask = (self.df_low.index >= sweep_time) & (self.df_low.index <= end_time)
        df_slice = self.df_low.loc[mask]

        if len(df_slice) == 0:
            return None

        # Calculate indicators on slice
        inflexions_slice = smc_custom.inflexion_points(df_slice)
        bos_slice = smc_custom.bos(df_slice, inflexions_slice, close_break=True)
        fvg_slice = smc.fvg(df_slice, join_consecutive=False)

        # Generate data using shared visualization functions
        candle_data = generate_candle_data(df_slice)
        fvg_zones = generate_fvg_zones(df_slice, fvg_slice)
        ob_zones = []  # Usually not calculated for 1M
        bos_lines = generate_bos_lines(df_slice, bos_slice, inflexions_slice)
        liquidity_lines = generate_liquidity_lines(df_slice, inflexions_slice)

        # Confirmation marker
        confirmation_marker = {
            'time': int(confirmation_time.timestamp()),
            'position': 'belowBar' if sweep_entry.entry_direction == 'long' else 'aboveBar',
            'color': '#00ff00',
            'shape': 'arrowUp' if sweep_entry.entry_direction == 'long' else 'arrowDown',
            'text': 'ENTRY'
        }

        # TP/SL lines
        tp_line = None
        sl_line = None
        if tp_price is not None:
            tp_line = {
                'price': float(tp_price),
                'color': '#089981',  # Green
                'label': f'TP: ${tp_price:.2f}'
            }
        if sl_price is not None:
            sl_line = {
                'price': float(sl_price),
                'color': '#f23645',  # Red
                'label': f'SL: ${sl_price:.2f}'
            }

        # Focus on confirmation/entry time
        focus_time = int(confirmation_time.timestamp())

        return {
            'timeframe': self.tf_labels.get('low', '1M'),
            'tabName': self.tf_labels.get('low', '1M'),
            'candleData': candle_data,
            'fvgZones': fvg_zones,
            'obZones': ob_zones,
            'bosLines': bos_lines,
            'liquidityLines': liquidity_lines,
            'sweepMarker': None,
            'markers': [confirmation_marker],
            'equilibriumLine': None,
            'tpLine': tp_line,
            'slLine': sl_line,
            'focusTime': focus_time,
            'contextCandles': 60  # Show 60 candles on each side for 1M (covers more time)
        }

    def _get_trade_result_for_sweep(self, sweep_entry: SweepEntry) -> Optional[TradeResult]:
        """Get the TradeResult for a sweep entry if available."""
        if not sweep_entry.is_complete or not sweep_entry.signal:
            return None

        # Look up by entry time
        entry_time = sweep_entry.signal.timestamp_entry
        if entry_time in self._trade_result_lookup:
            return self._trade_result_lookup[entry_time]

        # Fallback: look up by confirmation time
        confirmation_time = sweep_entry.signal.timestamp_1m_confirmation
        if confirmation_time in self._trade_result_lookup:
            return self._trade_result_lookup[confirmation_time]

        return None

    def _get_pnl_unavailable_reason(self, sweep_entry: SweepEntry) -> Optional[str]:
        """
        Get human-readable reason why P&L is not available for this sweep.

        Args:
            sweep_entry: The sweep entry to check

        Returns:
            Human-readable reason string, or None if P&L is available
        """
        # Check if this is a partial setup (incomplete)
        if sweep_entry.partial and not sweep_entry.is_complete:
            failure_reason = sweep_entry.partial.failure_reason or "conditions not met"
            return f"Setup incomplete: {failure_reason}"

        # Check if signal was skipped during backtesting
        if sweep_entry.signal:
            entry_time = sweep_entry.signal.timestamp_entry
            if entry_time in self._skip_reason_lookup:
                skip = self._skip_reason_lookup[entry_time]
                return SKIP_REASON_MESSAGES.get(skip.skip_reason, "Trade skipped")

        # No trade results available (backtest not run)
        if not self.trade_results:
            return "Backtest not run"

        return None

    def _generate_tab_pnl(self, sweep_entry: SweepEntry, trade_result: Optional[TradeResult]) -> Optional[Dict]:
        """
        Generate P&L tab data showing unrealized P&L progression.

        Args:
            sweep_entry: The sweep entry being visualized
            trade_result: The TradeResult from backtesting (if available)

        Returns:
            Dict with P&L chart data, or None if no trade result
        """
        if not trade_result:
            return None

        # Check if we have unrealized P&L series data
        if not trade_result.unrealized_pnl_series or len(trade_result.unrealized_pnl_series) == 0:
            return None

        # Convert unrealized P&L series to chart format
        pnl_series = []
        for ts, pnl in trade_result.unrealized_pnl_series:
            pnl_series.append({
                'time': int(ts.timestamp()),
                'value': float(pnl)
            })

        # Calculate TP and SL levels in dollar terms
        tp_dollars = trade_result.reward_amount()
        sl_dollars = -trade_result.risk_amount()

        return {
            'tabName': 'P&L',
            'pnlSeries': pnl_series,
            'tpDollars': tp_dollars,
            'slDollars': sl_dollars,
            'exitPnl': trade_result.pnl_dollars,
            'exitTime': int(trade_result.exit_time.timestamp()) if trade_result.exit_time else None,
            'entryPrice': trade_result.entry_price,
            'direction': trade_result.entry_direction,
            'outcome': trade_result.outcome.value if trade_result.outcome else None,
            'exitType': trade_result.exit_type.value if trade_result.exit_type else None
        }

    def _generate_sweep_data(self, sweep_entry: SweepEntry, sweep_idx: int) -> Dict:
        """
        Generate all tab data for a single sweep (static snapshots).

        Returns dict with data for each of the 5 tabs (including P&L).
        """
        # Get trade result for this sweep (if available from backtesting)
        trade_result = self._get_trade_result_for_sweep(sweep_entry)

        # Generate tabs (pass trade_result so charts extend to actual exit time)
        tab_1h = self._generate_tab_1h(sweep_entry, trade_result)
        tab_5m_event_b = self._generate_tab_5m_event_b(sweep_entry, trade_result)
        tab_5m_validation = self._generate_tab_5m_validation(sweep_entry, trade_result)
        tab_1m = self._generate_tab_1m(sweep_entry, trade_result)
        tab_pnl = self._generate_tab_pnl(sweep_entry, trade_result)

        # Determine active tab (first available tab with most progress)
        if tab_1m:
            active_tab = '1M'
        elif tab_5m_validation:
            active_tab = '5M_VALIDATION'
        elif tab_5m_event_b:
            active_tab = '5M_EVENT_B'
        else:
            active_tab = '1H'

        # Build conditions list
        conditions = [
            {
                'label': f'{self.tf_labels.get("high", "1H")} Sweep',
                'met': True,
                'value': sweep_entry.partial.condition_liquidity_sweep if sweep_entry.partial else sweep_entry.signal.condition_liquidity_sweep
            }
        ]

        # Event B condition
        if sweep_entry.is_complete:
            conditions.append({
                'label': f'{self.tf_labels.get("mid", "5M")} Event B',
                'met': True,
                'value': sweep_entry.signal.condition_event_b
            })
        elif sweep_entry.partial and sweep_entry.partial.condition_event_b:
            conditions.append({
                'label': f'{self.tf_labels.get("mid", "5M")} Event B',
                'met': True,
                'value': sweep_entry.partial.condition_event_b
            })
        else:
            conditions.append({
                'label': f'{self.tf_labels.get("mid", "5M")} Event B',
                'met': False,
                'value': None
            })

        # Validation condition
        if sweep_entry.is_complete:
            conditions.append({
                'label': f'{self.tf_labels.get("mid", "5M")} Validation',
                'met': True,
                'value': sweep_entry.signal.condition_validation
            })
        elif sweep_entry.partial and sweep_entry.partial.condition_validation:
            conditions.append({
                'label': f'{self.tf_labels.get("mid", "5M")} Validation',
                'met': True,
                'value': sweep_entry.partial.condition_validation
            })
        else:
            conditions.append({
                'label': f'{self.tf_labels.get("mid", "5M")} Validation',
                'met': False,
                'value': None
            })

        # Confirmation condition
        if sweep_entry.is_complete:
            conditions.append({
                'label': f'{self.tf_labels.get("low", "1M")} Confirmation',
                'met': True,
                'value': sweep_entry.signal.condition_confirmation
            })
        elif sweep_entry.partial and sweep_entry.partial.condition_confirmation:
            conditions.append({
                'label': f'{self.tf_labels.get("low", "1M")} Confirmation',
                'met': True,
                'value': sweep_entry.partial.condition_confirmation
            })
        else:
            conditions.append({
                'label': f'{self.tf_labels.get("low", "1M")} Confirmation',
                'met': False,
                'value': None
            })

        # Extract P&L data from trade result (if available)
        pnl_dollars = trade_result.pnl_dollars if trade_result else None
        pnl_percent = trade_result.pnl_percent if trade_result else None
        pnl_r = trade_result.pnl_r_multiple if trade_result else None
        exit_type = trade_result.exit_type.value if trade_result and trade_result.exit_type else None
        trade_outcome = trade_result.outcome.value if trade_result and trade_result.outcome else None

        # Get reason why P&L is unavailable (if applicable)
        pnl_unavailable_reason = None
        if pnl_dollars is None:
            pnl_unavailable_reason = self._get_pnl_unavailable_reason(sweep_entry)

        return {
            'sweepIdx': sweep_idx,
            'timestamp': sweep_entry.timestamp.strftime('%Y-%m-%d %H:%M'),
            'outcome': sweep_entry.outcome,
            'direction': sweep_entry.entry_direction.upper(),
            'conditions': conditions,
            'activeTab': active_tab,
            'tab1H': tab_1h,
            'tab5MEventB': tab_5m_event_b,
            'tab5MValidation': tab_5m_validation,
            'tab1M': tab_1m,
            'tabPnL': tab_pnl,
            'entryPrice': sweep_entry.signal.price_entry if sweep_entry.is_complete else None,
            'tpPrice': sweep_entry.signal.take_profit_price if sweep_entry.is_complete else None,
            'slPrice': sweep_entry.signal.stop_loss_price if sweep_entry.is_complete else None,
            # P&L fields from backtesting
            'pnlDollars': pnl_dollars,
            'pnlPercent': pnl_percent,
            'pnlR': pnl_r,
            'exitType': exit_type,
            'tradeOutcome': trade_outcome,
            'pnlUnavailableReason': pnl_unavailable_reason
        }

    def _generate_performance_data(self) -> Optional[Dict]:
        """
        Generate performance data for the Dashboard tab.

        Returns:
            Dict with performance metrics and equity curve data, or None if not available.
        """
        if not self.performance_metrics:
            return None

        metrics = self.performance_metrics

        # Convert equity curve to chart-friendly format
        # Each point is indexed by trade number
        equity_curve_data = []
        for i, capital in enumerate(metrics.equity_curve):
            equity_curve_data.append({
                'trade': i,
                'capital': float(capital)
            })

        # Handle infinity for profit factor display
        profit_factor = metrics.profit_factor
        if profit_factor == float('inf'):
            profit_factor_str = "∞"
            profit_factor_num = None
        else:
            profit_factor_str = f"{profit_factor:.2f}"
            profit_factor_num = float(profit_factor)

        return {
            # KPI card data
            'totalReturn': float(metrics.total_return_percent),
            'winRate': float(metrics.win_rate * 100),
            'profitFactor': profit_factor_num,
            'profitFactorStr': profit_factor_str,
            'maxDrawdownPercent': float(metrics.max_drawdown_percent),
            'maxDrawdownDollars': float(metrics.max_drawdown_dollars),

            # Trade statistics
            'totalTrades': int(metrics.total_trades),
            'winningTrades': int(metrics.winning_trades),
            'losingTrades': int(metrics.losing_trades),

            # Exit breakdown
            'tpExits': int(metrics.tp_exits),
            'slExits': int(metrics.sl_exits),
            'timeoutExits': int(metrics.timeout_exits),
            'bosExits': int(metrics.bos_exits),

            # Risk metrics
            'avgRMultiple': float(metrics.average_r_multiple),
            'maxConsecWins': int(metrics.max_consecutive_wins),
            'maxConsecLosses': int(metrics.max_consecutive_losses),

            # Win/Loss breakdown
            'avgWinDollars': float(metrics.average_win_dollars),
            'avgLossDollars': float(metrics.average_loss_dollars),
            'largestWin': float(metrics.largest_win_dollars),
            'largestLoss': float(metrics.largest_loss_dollars),

            # Capital
            'initialCapital': float(metrics.initial_capital),
            'finalCapital': float(metrics.final_capital),
            'totalPnl': float(metrics.total_pnl_dollars),

            # Direction breakdown
            'longTrades': int(metrics.long_trades),
            'shortTrades': int(metrics.short_trades),
            'longWins': int(metrics.long_wins),
            'shortWins': int(metrics.short_wins),

            # Equity curve data
            'equityCurve': equity_curve_data
        }

    def _create_html_template(self, all_sweeps_data: List[Dict], performance_data: Optional[Dict] = None) -> str:
        """Create the complete HTML template with embedded data."""

        tf_high = self.tf_labels.get('high', '1H')
        tf_mid = self.tf_labels.get('mid', '5M')
        tf_low = self.tf_labels.get('low', '1M')

        # Serialize performance data
        performance_json = json.dumps(performance_data, cls=NumpyEncoder) if performance_data else 'null'

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.symbol} Strategy Sweep Visualization</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #131722;
            color: #d1d4dc;
            height: 100vh;
            overflow: hidden;
        }}
        #app {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}

        /* Header */
        #header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 20px;
            background-color: #1e222d;
            border-bottom: 1px solid #2b2b43;
        }}
        #header h1 {{
            color: #2962ff;
            font-size: 18px;
            font-weight: 600;
        }}
        .header-controls {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .sweep-selector {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .sweep-selector label {{
            color: #787b86;
            font-size: 13px;
        }}
        #sweep-dropdown {{
            padding: 8px 12px;
            background-color: #131722;
            border: 1px solid #363a45;
            color: #d1d4dc;
            border-radius: 4px;
            font-size: 13px;
            min-width: 280px;
            cursor: pointer;
        }}
        #sweep-dropdown:focus {{
            outline: none;
            border-color: #2962ff;
        }}
        .sweep-counter {{
            color: #d1d4dc;
            font-size: 13px;
            min-width: 100px;
            text-align: center;
        }}

        /* Tabs */
        #tabs {{
            display: flex;
            background-color: #1e222d;
            border-bottom: 1px solid #2b2b43;
            padding: 0 20px;
        }}
        .tab {{
            padding: 12px 24px;
            color: #787b86;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }}
        .tab:hover {{
            color: #d1d4dc;
        }}
        .tab.active {{
            color: #2962ff;
            border-bottom-color: #2962ff;
        }}
        .tab.disabled {{
            color: #363a45;
            cursor: not-allowed;
        }}
        .tab.disabled:hover {{
            color: #363a45;
        }}

        /* Main content */
        #main {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        /* Chart area */
        #chart-area {{
            flex: 1;
            position: relative;
        }}
        #chart-container {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
        }}
        #svg-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            z-index: 10;
        }}

        /* Sidebar */
        #sidebar {{
            width: 280px;
            background-color: #1e222d;
            border-left: 1px solid #2b2b43;
            padding: 20px;
            overflow-y: auto;
        }}
        .sidebar-section {{
            margin-bottom: 24px;
        }}
        .sidebar-section h3 {{
            color: #787b86;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        .info-label {{
            color: #787b86;
        }}
        .info-value {{
            color: #d1d4dc;
            font-weight: 500;
        }}
        .info-value.success {{
            color: #089981;
        }}
        .info-value.failed {{
            color: #f23645;
        }}
        .info-value.long {{
            color: #089981;
        }}
        .info-value.short {{
            color: #f23645;
        }}

        /* Conditions list */
        .condition-item {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            font-size: 13px;
        }}
        .condition-check {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            font-size: 12px;
        }}
        .condition-check.met {{
            background-color: rgba(8, 153, 129, 0.2);
            color: #089981;
        }}
        .condition-check.not-met {{
            background-color: rgba(242, 54, 69, 0.2);
            color: #f23645;
        }}
        .condition-label {{
            flex: 1;
        }}
        .condition-value {{
            color: #787b86;
            font-size: 11px;
        }}

        /* Legend */
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 12px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            margin-right: 8px;
            border-radius: 2px;
        }}

        /* Keyboard shortcuts */
        .shortcuts {{
            margin-top: 20px;
            padding-top: 16px;
            border-top: 1px solid #2b2b43;
        }}
        .shortcut-item {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 11px;
            color: #787b86;
        }}
        .shortcut-key {{
            background-color: #131722;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }}

        /* P&L Unavailable message (subtle info style) */
        .pnl-unavailable {{
            color: #6b7280;
            font-size: 12px;
            font-style: italic;
            padding: 8px 0;
            text-align: center;
        }}

        /* Dashboard Styles */
        #dashboard-container {{
            display: none;
            padding: 24px;
            overflow-y: auto;
            height: 100%;
        }}
        #dashboard-container.active {{
            display: block;
        }}
        .dashboard-no-data {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #787b86;
            font-size: 16px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background-color: #1e222d;
            border: 1px solid #2b2b43;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-card.positive {{
            border-color: rgba(8, 153, 129, 0.4);
        }}
        .kpi-card.negative {{
            border-color: rgba(242, 54, 69, 0.4);
        }}
        .kpi-label {{
            color: #787b86;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 28px;
            font-weight: 600;
        }}
        .kpi-value.positive {{
            color: #089981;
        }}
        .kpi-value.negative {{
            color: #f23645;
        }}
        .kpi-value.neutral {{
            color: #d1d4dc;
        }}
        .equity-chart-container {{
            background-color: #1e222d;
            border: 1px solid #2b2b43;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
            height: 300px;
        }}
        .equity-chart-title {{
            color: #787b86;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}
        #equity-chart {{
            width: 100%;
            height: calc(100% - 30px);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }}
        .stats-card {{
            background-color: #1e222d;
            border: 1px solid #2b2b43;
            border-radius: 8px;
            padding: 16px;
        }}
        .stats-card-title {{
            color: #787b86;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2b2b43;
        }}
        .stats-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        .stats-row:last-child {{
            margin-bottom: 0;
        }}
        .stats-label {{
            color: #787b86;
        }}
        .stats-value {{
            color: #d1d4dc;
            font-weight: 500;
        }}
        .stats-value.positive {{
            color: #089981;
        }}
        .stats-value.negative {{
            color: #f23645;
        }}
    </style>
</head>
<body>
    <div id="app">
        <div id="header">
            <h1>{self.symbol} Strategy Visualization</h1>
            <div class="header-controls">
                <div class="sweep-selector">
                    <label>Sweep:</label>
                    <select id="sweep-dropdown"></select>
                </div>
                <span class="sweep-counter" id="sweep-counter">1 / 1</span>
            </div>
        </div>

        <div id="tabs">
            <div class="tab" data-tab="1H" onclick="switchTab('1H')">{tf_high}</div>
            <div class="tab" data-tab="5M_EVENT_B" onclick="switchTab('5M_EVENT_B')">{tf_mid} Event B</div>
            <div class="tab" data-tab="5M_VALIDATION" onclick="switchTab('5M_VALIDATION')">{tf_mid} Validation</div>
            <div class="tab" data-tab="1M" onclick="switchTab('1M')">{tf_low}</div>
            <div class="tab" data-tab="PNL" onclick="switchTab('PNL')">P&L</div>
            <div class="tab" data-tab="DASHBOARD" onclick="switchTab('DASHBOARD')">Dashboard</div>
        </div>

        <div id="main">
            <div id="chart-area">
                <div id="chart-container"></div>
                <div id="dashboard-container"></div>
            </div>

            <div id="sidebar">
                <div class="sidebar-section">
                    <h3>Sweep Information</h3>
                    <div class="info-row">
                        <span class="info-label">Time</span>
                        <span class="info-value" id="sweep-time">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Direction</span>
                        <span class="info-value" id="sweep-direction">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Outcome</span>
                        <span class="info-value" id="sweep-outcome">-</span>
                    </div>
                </div>

                <div class="sidebar-section" id="trade-info" style="display: none;">
                    <h3>Trade Details</h3>
                    <div class="info-row">
                        <span class="info-label">Entry</span>
                        <span class="info-value" id="entry-price">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Take Profit</span>
                        <span class="info-value success" id="tp-price">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Stop Loss</span>
                        <span class="info-value failed" id="sl-price">-</span>
                    </div>
                </div>

                <div class="sidebar-section" id="pnl-info" style="display: none;">
                    <h3>P&L Summary</h3>
                    <div class="info-row">
                        <span class="info-label">Outcome</span>
                        <span class="info-value" id="trade-outcome">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Exit Type</span>
                        <span class="info-value" id="exit-type">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">P&L</span>
                        <span class="info-value" id="pnl-dollars">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">P&L %</span>
                        <span class="info-value" id="pnl-percent">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">R-Multiple</span>
                        <span class="info-value" id="pnl-r">-</span>
                    </div>
                </div>

                <div class="sidebar-section">
                    <h3>Conditions</h3>
                    <div id="conditions-list"></div>
                </div>

                <div class="sidebar-section">
                    <h3>Legend</h3>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: rgba(255, 235, 59, 0.3); border: 1px solid #ffeb3b;"></div>
                        <span>FVG Zone</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: rgba(0, 188, 212, 0.3); border: 1px solid #00bcd4;"></div>
                        <span>Bullish OB</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: rgba(255, 87, 34, 0.3); border: 1px solid #ff5722;"></div>
                        <span>Bearish OB</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #089981;"></div>
                        <span>Bullish BOS</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #f23645;"></div>
                        <span>Bearish BOS</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #ff8c00;"></div>
                        <span>Liquidity Level</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #9c27b0;"></div>
                        <span>Equilibrium</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #ffff00;"></div>
                        <span>Event B Highlight</span>
                    </div>
                </div>

                <div class="shortcuts">
                    <div class="shortcut-item">
                        <span>Previous sweep</span>
                        <span class="shortcut-key">&#8592;</span>
                    </div>
                    <div class="shortcut-item">
                        <span>Next sweep</span>
                        <span class="shortcut-key">&#8594;</span>
                    </div>
                    <div class="shortcut-item">
                        <span>Switch tab</span>
                        <span class="shortcut-key">1 2 3 4 5 6</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embed all sweep data
        const allSweepsData = {json.dumps(all_sweeps_data, cls=NumpyEncoder)};

        // Embed performance data for Dashboard
        const performanceData = {performance_json};

        let currentSweepIdx = 0;
        let currentTab = '1H';
        let chart = null;
        let equityChart = null;
        let candlestickSeries = null;
        let activeLineSeries = [];
        let dashboardRendered = false;

        // Chart options
        const chartOptions = {{
            layout: {{
                background: {{ color: '#131722' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#1e222d' }},
                horzLines: {{ color: '#1e222d' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            rightPriceScale: {{
                borderColor: '#2b2b43',
            }},
            timeScale: {{
                borderColor: '#2b2b43',
                timeVisible: true,
                secondsVisible: false,
            }},
        }};

        // Initialize
        function init() {{
            const chartContainer = document.getElementById('chart-container');
            chart = LightweightCharts.createChart(chartContainer, chartOptions);
            candlestickSeries = chart.addCandlestickSeries({{
                upColor: '#089981',
                downColor: '#f23645',
                borderVisible: false,
                wickUpColor: '#089981',
                wickDownColor: '#f23645',
            }});

            // Create SVG overlay
            const svg = d3.select(chartContainer)
                .append('svg')
                .attr('id', 'svg-overlay')
                .style('position', 'absolute')
                .style('top', '0')
                .style('left', '0')
                .style('pointer-events', 'none');

            // Populate dropdown
            const dropdown = document.getElementById('sweep-dropdown');
            allSweepsData.forEach((sweep, idx) => {{
                const option = document.createElement('option');
                option.value = idx;
                option.textContent = `${{sweep.timestamp}} - ${{sweep.outcome}}`;
                dropdown.appendChild(option);
            }});

            dropdown.addEventListener('change', (e) => {{
                currentSweepIdx = parseInt(e.target.value);
                showSweep(currentSweepIdx);
            }});

            // Keyboard navigation
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'ArrowLeft') prevSweep();
                if (e.key === 'ArrowRight') nextSweep();
                if (e.key === '1') switchTab('1H');
                if (e.key === '2') switchTab('5M_EVENT_B');
                if (e.key === '3') switchTab('5M_VALIDATION');
                if (e.key === '4') switchTab('1M');
                if (e.key === '5') switchTab('PNL');
                if (e.key === '6') switchTab('DASHBOARD');
            }});

            // Resize handler
            const resizeObserver = new ResizeObserver(() => {{
                chart.applyOptions({{
                    width: chartContainer.clientWidth,
                    height: chartContainer.clientHeight
                }});
                redrawOverlays();
            }});
            resizeObserver.observe(chartContainer);

            // Subscribe to chart events
            chart.timeScale().subscribeVisibleLogicalRangeChange(redrawOverlays);

            // Show first sweep
            if (allSweepsData.length > 0) {{
                showSweep(0);
            }}
        }}

        function showSweep(idx) {{
            if (idx < 0 || idx >= allSweepsData.length) return;

            currentSweepIdx = idx;
            const sweep = allSweepsData[idx];

            // Update dropdown
            document.getElementById('sweep-dropdown').value = idx;

            // Update sweep counter
            document.getElementById('sweep-counter').textContent =
                `${{idx + 1}} / ${{allSweepsData.length}}`;

            // Update sidebar
            document.getElementById('sweep-time').textContent = sweep.timestamp;

            const dirEl = document.getElementById('sweep-direction');
            dirEl.textContent = sweep.direction;
            dirEl.className = 'info-value ' + sweep.direction.toLowerCase();

            const outcomeEl = document.getElementById('sweep-outcome');
            outcomeEl.textContent = sweep.outcome;
            outcomeEl.className = 'info-value ' + (sweep.outcome.includes('SUCCESS') ? 'success' : 'failed');

            // Trade details (only for successful signals)
            const tradeInfo = document.getElementById('trade-info');
            if (sweep.entryPrice) {{
                tradeInfo.style.display = 'block';
                document.getElementById('entry-price').textContent = '$' + sweep.entryPrice.toFixed(2);
                document.getElementById('tp-price').textContent = sweep.tpPrice ? '$' + sweep.tpPrice.toFixed(2) : '-';
                document.getElementById('sl-price').textContent = sweep.slPrice ? '$' + sweep.slPrice.toFixed(2) : '-';
            }} else {{
                tradeInfo.style.display = 'none';
            }}

            // P&L Summary (for completed trades with backtest results, or show unavailable reason)
            const pnlInfo = document.getElementById('pnl-info');
            if (sweep.pnlDollars !== null && sweep.pnlDollars !== undefined) {{
                // P&L data available - show full details
                pnlInfo.style.display = 'block';
                pnlInfo.innerHTML = `
                    <h3>P&L Summary</h3>
                    <div class="info-row">
                        <span class="info-label">Outcome</span>
                        <span class="info-value" id="trade-outcome">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Exit Type</span>
                        <span class="info-value" id="exit-type">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">P&L</span>
                        <span class="info-value" id="pnl-dollars">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">P&L %</span>
                        <span class="info-value" id="pnl-percent">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">R-Multiple</span>
                        <span class="info-value" id="pnl-r">-</span>
                    </div>
                `;

                // Outcome
                const outcomeSpan = document.getElementById('trade-outcome');
                outcomeSpan.textContent = sweep.tradeOutcome ? sweep.tradeOutcome.toUpperCase() : '-';
                outcomeSpan.className = 'info-value ' + (sweep.tradeOutcome === 'win' ? 'success' : 'failed');

                // Exit Type
                const exitTypeSpan = document.getElementById('exit-type');
                const exitTypeMap = {{
                    'tp_hit': 'Take Profit',
                    'sl_hit': 'Stop Loss',
                    'timeout': 'Timeout',
                    'bos_exit': 'BOS Exit'
                }};
                exitTypeSpan.textContent = exitTypeMap[sweep.exitType] || sweep.exitType || '-';

                // P&L Dollars
                const pnlDollarsSpan = document.getElementById('pnl-dollars');
                const pnlDollars = sweep.pnlDollars;
                pnlDollarsSpan.textContent = (pnlDollars >= 0 ? '+' : '') + '$' + pnlDollars.toFixed(2);
                pnlDollarsSpan.className = 'info-value ' + (pnlDollars >= 0 ? 'success' : 'failed');

                // P&L Percent
                const pnlPercentSpan = document.getElementById('pnl-percent');
                const pnlPercent = sweep.pnlPercent;
                pnlPercentSpan.textContent = (pnlPercent >= 0 ? '+' : '') + pnlPercent.toFixed(2) + '%';
                pnlPercentSpan.className = 'info-value ' + (pnlPercent >= 0 ? 'success' : 'failed');

                // R-Multiple
                const pnlRSpan = document.getElementById('pnl-r');
                const pnlR = sweep.pnlR;
                pnlRSpan.textContent = (pnlR >= 0 ? '+' : '') + pnlR.toFixed(2) + 'R';
                pnlRSpan.className = 'info-value ' + (pnlR >= 0 ? 'success' : 'failed');
            }} else if (sweep.pnlUnavailableReason) {{
                // P&L not available - show reason
                pnlInfo.style.display = 'block';
                pnlInfo.innerHTML = `
                    <h3>P&L Summary</h3>
                    <div class="pnl-unavailable">${{sweep.pnlUnavailableReason}}</div>
                `;
            }} else {{
                pnlInfo.style.display = 'none';
            }}

            // Update conditions
            const conditionsList = document.getElementById('conditions-list');
            conditionsList.innerHTML = '';
            sweep.conditions.forEach(cond => {{
                const item = document.createElement('div');
                item.className = 'condition-item';
                item.innerHTML = `
                    <div class="condition-check ${{cond.met ? 'met' : 'not-met'}}">
                        ${{cond.met ? '&#10003;' : '&#10007;'}}
                    </div>
                    <div class="condition-label">${{cond.label}}</div>
                    <div class="condition-value">${{cond.value || ''}}</div>
                `;
                conditionsList.appendChild(item);
            }});

            // Update tabs
            updateTabs(sweep);

            // Switch to active tab and show chart
            currentTab = sweep.activeTab;
            showChart(sweep);
        }}

        function updateTabs(sweep) {{
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {{
                const tabId = tab.dataset.tab;
                tab.classList.remove('active', 'disabled');

                // Check if tab has data
                let hasData = false;
                if (tabId === '1H' && sweep.tab1H) hasData = true;
                if (tabId === '5M_EVENT_B' && sweep.tab5MEventB) hasData = true;
                if (tabId === '5M_VALIDATION' && sweep.tab5MValidation) hasData = true;
                if (tabId === '1M' && sweep.tab1M) hasData = true;
                if (tabId === 'PNL' && sweep.tabPnL) hasData = true;
                if (tabId === 'DASHBOARD') hasData = true;  // Dashboard is always available

                if (!hasData) {{
                    tab.classList.add('disabled');
                }}

                if (tabId === sweep.activeTab || (currentTab === 'DASHBOARD' && tabId === 'DASHBOARD')) {{
                    tab.classList.add('active');
                }}
            }});
        }}

        function switchTab(tabId) {{
            const sweep = allSweepsData[currentSweepIdx];

            // Check if tab is disabled
            let hasData = false;
            if (tabId === '1H' && sweep.tab1H) hasData = true;
            if (tabId === '5M_EVENT_B' && sweep.tab5MEventB) hasData = true;
            if (tabId === '5M_VALIDATION' && sweep.tab5MValidation) hasData = true;
            if (tabId === '1M' && sweep.tab1M) hasData = true;
            if (tabId === 'PNL' && sweep.tabPnL) hasData = true;
            if (tabId === 'DASHBOARD') hasData = true;  // Dashboard is always available

            if (!hasData) return;

            // Update tab UI
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
                if (tab.dataset.tab === tabId) tab.classList.add('active');
            }});

            currentTab = tabId;

            // Handle Dashboard tab separately
            if (tabId === 'DASHBOARD') {{
                showDashboard();
            }} else {{
                hideDashboard();
                showChart(sweep);
            }}
        }}

        function showChart(sweep) {{
            // Get tab data for current tab
            let tabData = null;
            if (currentTab === '1H') tabData = sweep.tab1H;
            else if (currentTab === '5M_EVENT_B') tabData = sweep.tab5MEventB;
            else if (currentTab === '5M_VALIDATION') tabData = sweep.tab5MValidation;
            else if (currentTab === '1M') tabData = sweep.tab1M;
            else if (currentTab === 'PNL') tabData = sweep.tabPnL;

            if (!tabData) return;

            // Handle P&L tab separately (line chart, not candlestick)
            if (currentTab === 'PNL') {{
                showPnLChart(sweep.tabPnL);
                return;
            }}

            // Clear existing lines
            clearLineSeries();

            // Update candlesticks
            candlestickSeries.setData(tabData.candleData);

            // Set markers
            const markers = tabData.markers || [];
            if (tabData.sweepMarker) markers.push(tabData.sweepMarker);
            candlestickSeries.setMarkers(markers);

            // Draw BOS lines
            if (tabData.bosLines) {{
                tabData.bosLines.forEach(bos => {{
                    const lineSeries = chart.addLineSeries({{
                        color: bos.color,
                        lineWidth: bos.lineWidth || 2,
                        lineStyle: LightweightCharts.LineStyle.Solid,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    }});
                    lineSeries.setData([
                        {{ time: bos.startTime, value: bos.price }},
                        {{ time: bos.endTime, value: bos.price }}
                    ]);
                    activeLineSeries.push(lineSeries);
                }});
            }}

            // Draw liquidity lines
            if (tabData.liquidityLines) {{
                tabData.liquidityLines.forEach(liq => {{
                    const lineSeries = chart.addLineSeries({{
                        color: liq.color,
                        lineWidth: liq.lineWidth,
                        lineStyle: liq.swept ? LightweightCharts.LineStyle.Solid : LightweightCharts.LineStyle.Dotted,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    }});
                    lineSeries.setData([
                        {{ time: liq.startTime, value: liq.price }},
                        {{ time: liq.endTime, value: liq.price }}
                    ]);
                    activeLineSeries.push(lineSeries);
                }});
            }}

            // Draw equilibrium line (5M Validation tab)
            if (tabData.equilibriumLine) {{
                const eqSeries = chart.addLineSeries({{
                    color: tabData.equilibriumLine.color,
                    lineWidth: tabData.equilibriumLine.lineWidth || 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false,
                    lastValueVisible: true,
                    title: 'EQ',
                }});
                const firstTime = tabData.candleData[0].time;
                const lastTime = tabData.candleData[tabData.candleData.length - 1].time;
                eqSeries.setData([
                    {{ time: firstTime, value: tabData.equilibriumLine.price }},
                    {{ time: lastTime, value: tabData.equilibriumLine.price }}
                ]);
                activeLineSeries.push(eqSeries);
            }}

            // Draw TP/SL lines (1M tab only)
            if (tabData.tpLine) {{
                const tpSeries = chart.addLineSeries({{
                    color: tabData.tpLine.color,
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false,
                    lastValueVisible: true,
                    title: 'TP',
                }});
                const firstTime = tabData.candleData[0].time;
                const lastTime = tabData.candleData[tabData.candleData.length - 1].time;
                tpSeries.setData([
                    {{ time: firstTime, value: tabData.tpLine.price }},
                    {{ time: lastTime, value: tabData.tpLine.price }}
                ]);
                activeLineSeries.push(tpSeries);
            }}

            if (tabData.slLine) {{
                const slSeries = chart.addLineSeries({{
                    color: tabData.slLine.color,
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false,
                    lastValueVisible: true,
                    title: 'SL',
                }});
                const firstTime = tabData.candleData[0].time;
                const lastTime = tabData.candleData[tabData.candleData.length - 1].time;
                slSeries.setData([
                    {{ time: firstTime, value: tabData.slLine.price }},
                    {{ time: lastTime, value: tabData.slLine.price }}
                ]);
                activeLineSeries.push(slSeries);
            }}

            // Apply context window - focus on key event instead of fitting all content
            applyContextWindow(tabData);

            // Redraw overlays (FVG, OB zones)
            setTimeout(redrawOverlays, 50);
        }}

        function applyContextWindow(tabData) {{
            // If we have focus time and context candles, set visible range around focus
            if (tabData.focusTime && tabData.contextCandles && tabData.candleData && tabData.candleData.length > 0) {{
                const focusTime = tabData.focusTime;
                const contextCandles = tabData.contextCandles;

                // Find the index of the candle closest to focus time
                let focusIdx = 0;
                for (let i = 0; i < tabData.candleData.length; i++) {{
                    if (tabData.candleData[i].time >= focusTime) {{
                        focusIdx = i;
                        break;
                    }}
                    focusIdx = i;  // Last candle if focus time is beyond data
                }}

                // Calculate visible range indices
                const startIdx = Math.max(0, focusIdx - contextCandles);
                const endIdx = Math.min(tabData.candleData.length - 1, focusIdx + contextCandles);

                // Set visible logical range (index-based)
                chart.timeScale().setVisibleLogicalRange({{
                    from: startIdx,
                    to: endIdx
                }});
            }} else {{
                // Fallback to fit content if no focus info
                chart.timeScale().fitContent();
            }}
        }}

        function clearLineSeries() {{
            activeLineSeries.forEach(series => {{
                try {{ chart.removeSeries(series); }} catch(e) {{}}
            }});
            activeLineSeries = [];
        }}

        function redrawOverlays() {{
            // Skip overlays for P&L tab (no candlestick zones)
            if (currentTab === 'PNL') return;

            const sweep = allSweepsData[currentSweepIdx];
            let tabData = null;
            if (currentTab === '1H') tabData = sweep.tab1H;
            else if (currentTab === '5M_EVENT_B') tabData = sweep.tab5MEventB;
            else if (currentTab === '5M_VALIDATION') tabData = sweep.tab5MValidation;
            else if (currentTab === '1M') tabData = sweep.tab1M;

            if (!tabData) return;

            const chartContainer = document.getElementById('chart-container');
            const rect = chartContainer.getBoundingClientRect();

            const svg = d3.select('#svg-overlay')
                .attr('width', rect.width)
                .attr('height', rect.height);

            // Clear existing shapes
            svg.selectAll('rect').remove();
            svg.selectAll('text').remove();

            const timeScale = chart.timeScale();

            // Draw FVG zones
            if (tabData.fvgZones) {{
                tabData.fvgZones.forEach(zone => {{
                    const x1 = timeScale.timeToCoordinate(zone.startTime);
                    const x2 = timeScale.timeToCoordinate(zone.endTime);
                    const y1 = candlestickSeries.priceToCoordinate(zone.topPrice);
                    const y2 = candlestickSeries.priceToCoordinate(zone.bottomPrice);

                    if (x1 !== null && x2 !== null && y1 !== null && y2 !== null) {{
                        const x = Math.min(x1, x2);
                        const y = Math.min(y1, y2);
                        const width = Math.abs(x2 - x1);
                        const height = Math.abs(y2 - y1);

                        // Highlighted FVG (Event B IFVG) uses brighter color
                        const fillColor = zone.highlighted ? 'rgba(255, 255, 0, 0.4)' : 'rgba(255, 235, 59, 0.2)';
                        const strokeColor = zone.highlighted ? '#ffff00' : '#ffeb3b';
                        const strokeWidth = zone.highlighted ? 3 : 1;

                        svg.append('rect')
                            .attr('class', 'fvg')
                            .attr('x', x)
                            .attr('y', y)
                            .attr('width', width)
                            .attr('height', height)
                            .attr('fill', fillColor)
                            .attr('stroke', strokeColor)
                            .attr('stroke-width', strokeWidth)
                            .attr('stroke-dasharray', zone.highlighted ? '' : '4,4');
                    }}
                }});
            }}

            // Draw OB zones
            if (tabData.obZones) {{
                tabData.obZones.forEach(zone => {{
                    const x1 = timeScale.timeToCoordinate(zone.startTime);
                    const x2 = timeScale.timeToCoordinate(zone.endTime);
                    const y1 = candlestickSeries.priceToCoordinate(zone.topPrice);
                    const y2 = candlestickSeries.priceToCoordinate(zone.bottomPrice);

                    if (x1 !== null && x2 !== null && y1 !== null && y2 !== null) {{
                        const x = Math.min(x1, x2);
                        const y = Math.min(y1, y2);
                        const width = Math.abs(x2 - x1);
                        const height = Math.abs(y2 - y1);

                        const fillColor = zone.obType === 1 ? 'rgba(0, 188, 212, 0.2)' : 'rgba(255, 87, 34, 0.2)';
                        const strokeColor = zone.obType === 1 ? '#00bcd4' : '#ff5722';

                        svg.append('rect')
                            .attr('class', 'ob')
                            .attr('x', x)
                            .attr('y', y)
                            .attr('width', width)
                            .attr('height', height)
                            .attr('fill', fillColor)
                            .attr('stroke', strokeColor)
                            .attr('stroke-width', 1);
                    }}
                }});
            }}
        }}

        function showPnLChart(pnlData) {{
            // Clear existing series
            clearLineSeries();

            // Clear SVG overlays
            d3.select('#svg-overlay').selectAll('*').remove();

            // Hide candlestick series (we'll use line series for P&L)
            candlestickSeries.setData([]);
            candlestickSeries.setMarkers([]);

            if (!pnlData || !pnlData.pnlSeries || pnlData.pnlSeries.length === 0) {{
                return;
            }}

            // Get time range for horizontal lines
            const firstTime = pnlData.pnlSeries[0].time;
            const lastTime = pnlData.pnlSeries[pnlData.pnlSeries.length - 1].time;

            // Draw P&L line (main series)
            const pnlSeries = chart.addLineSeries({{
                color: '#2962ff',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                priceLineVisible: false,
                lastValueVisible: true,
                title: 'P&L',
            }});
            pnlSeries.setData(pnlData.pnlSeries);
            activeLineSeries.push(pnlSeries);

            // Draw zero line
            const zeroSeries = chart.addLineSeries({{
                color: '#787b86',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                priceLineVisible: false,
                lastValueVisible: false,
            }});
            zeroSeries.setData([
                {{ time: firstTime, value: 0 }},
                {{ time: lastTime, value: 0 }}
            ]);
            activeLineSeries.push(zeroSeries);

            // Draw TP line (green dashed)
            if (pnlData.tpDollars) {{
                const tpSeries = chart.addLineSeries({{
                    color: '#089981',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false,
                    lastValueVisible: true,
                    title: 'TP',
                }});
                tpSeries.setData([
                    {{ time: firstTime, value: pnlData.tpDollars }},
                    {{ time: lastTime, value: pnlData.tpDollars }}
                ]);
                activeLineSeries.push(tpSeries);
            }}

            // Draw SL line (red dashed)
            if (pnlData.slDollars) {{
                const slSeries = chart.addLineSeries({{
                    color: '#f23645',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false,
                    lastValueVisible: true,
                    title: 'SL',
                }});
                slSeries.setData([
                    {{ time: firstTime, value: pnlData.slDollars }},
                    {{ time: lastTime, value: pnlData.slDollars }}
                ]);
                activeLineSeries.push(slSeries);
            }}

            // Add exit marker
            if (pnlData.exitTime && pnlData.exitPnl !== null) {{
                const exitMarkerSeries = chart.addLineSeries({{
                    color: pnlData.exitPnl >= 0 ? '#089981' : '#f23645',
                    lineWidth: 0,
                    priceLineVisible: false,
                    lastValueVisible: false,
                }});
                // Use a point marker approach - add a small data point
                exitMarkerSeries.setData([
                    {{ time: pnlData.exitTime, value: pnlData.exitPnl }}
                ]);
                exitMarkerSeries.setMarkers([{{
                    time: pnlData.exitTime,
                    position: pnlData.exitPnl >= 0 ? 'aboveBar' : 'belowBar',
                    color: pnlData.exitPnl >= 0 ? '#089981' : '#f23645',
                    shape: 'circle',
                    text: 'EXIT'
                }}]);
                activeLineSeries.push(exitMarkerSeries);
            }}

            // Fit content
            chart.timeScale().fitContent();
        }}

        function showDashboard() {{
            // Hide chart container, show dashboard container
            document.getElementById('chart-container').style.display = 'none';
            const dashboardContainer = document.getElementById('dashboard-container');
            dashboardContainer.classList.add('active');

            // Render dashboard content if not already done or if data changed
            if (!dashboardRendered) {{
                renderDashboard();
                dashboardRendered = true;
            }}
        }}

        function hideDashboard() {{
            // Show chart container, hide dashboard container
            document.getElementById('chart-container').style.display = 'block';
            document.getElementById('dashboard-container').classList.remove('active');
        }}

        function renderDashboard() {{
            const container = document.getElementById('dashboard-container');

            if (!performanceData) {{
                container.innerHTML = '<div class="dashboard-no-data">No performance data available. Run backtest to see dashboard.</div>';
                return;
            }}

            const data = performanceData;

            // Format values
            const formatDollars = (val) => {{
                if (val === null || val === undefined) return '-';
                const prefix = val >= 0 ? '+' : '';
                return prefix + '$' + val.toFixed(2).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ',');
            }};

            const formatPercent = (val) => {{
                if (val === null || val === undefined) return '-';
                const prefix = val >= 0 ? '+' : '';
                return prefix + val.toFixed(2) + '%';
            }};

            const formatR = (val) => {{
                if (val === null || val === undefined) return '-';
                const prefix = val >= 0 ? '+' : '';
                return prefix + val.toFixed(2) + 'R';
            }};

            // Calculate direction win rates
            const longWinRate = data.longTrades > 0 ? ((data.longWins / data.longTrades) * 100).toFixed(1) : '0';
            const shortWinRate = data.shortTrades > 0 ? ((data.shortWins / data.shortTrades) * 100).toFixed(1) : '0';

            // Determine value classes
            const returnClass = data.totalReturn >= 0 ? 'positive' : 'negative';
            const winRateClass = data.winRate >= 50 ? 'positive' : 'negative';
            const pfClass = data.profitFactor !== null && data.profitFactor >= 1 ? 'positive' : (data.profitFactor !== null ? 'negative' : 'neutral');
            const ddClass = 'negative';  // Drawdown is always shown as negative

            container.innerHTML = `
                <!-- KPI Cards -->
                <div class="kpi-grid">
                    <div class="kpi-card ${{data.totalReturn >= 0 ? 'positive' : 'negative'}}">
                        <div class="kpi-label">Total Return</div>
                        <div class="kpi-value ${{returnClass}}">${{formatPercent(data.totalReturn)}}</div>
                    </div>
                    <div class="kpi-card ${{data.winRate >= 50 ? 'positive' : 'negative'}}">
                        <div class="kpi-label">Win Rate</div>
                        <div class="kpi-value ${{winRateClass}}">${{data.winRate.toFixed(1)}}%</div>
                    </div>
                    <div class="kpi-card ${{pfClass === 'positive' ? 'positive' : ''}}">
                        <div class="kpi-label">Profit Factor</div>
                        <div class="kpi-value ${{pfClass}}">${{data.profitFactorStr}}</div>
                    </div>
                    <div class="kpi-card negative">
                        <div class="kpi-label">Max Drawdown</div>
                        <div class="kpi-value negative">-${{data.maxDrawdownPercent.toFixed(1)}}%</div>
                    </div>
                </div>

                <!-- Equity Curve Chart -->
                <div class="equity-chart-container">
                    <div class="equity-chart-title">Equity Curve</div>
                    <div id="equity-chart"></div>
                </div>

                <!-- Stats Grid -->
                <div class="stats-grid">
                    <!-- Trade Statistics -->
                    <div class="stats-card">
                        <div class="stats-card-title">Trade Statistics</div>
                        <div class="stats-row">
                            <span class="stats-label">Total Trades</span>
                            <span class="stats-value">${{data.totalTrades}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Winning Trades</span>
                            <span class="stats-value positive">${{data.winningTrades}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Losing Trades</span>
                            <span class="stats-value negative">${{data.losingTrades}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Long Trades</span>
                            <span class="stats-value">${{data.longTrades}} (${{longWinRate}}% WR)</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Short Trades</span>
                            <span class="stats-value">${{data.shortTrades}} (${{shortWinRate}}% WR)</span>
                        </div>
                    </div>

                    <!-- Exit Breakdown -->
                    <div class="stats-card">
                        <div class="stats-card-title">Exit Breakdown</div>
                        <div class="stats-row">
                            <span class="stats-label">TP Exits</span>
                            <span class="stats-value positive">${{data.tpExits}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">SL Exits</span>
                            <span class="stats-value negative">${{data.slExits}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Timeout Exits</span>
                            <span class="stats-value">${{data.timeoutExits}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">BOS Exits</span>
                            <span class="stats-value">${{data.bosExits}}</span>
                        </div>
                    </div>

                    <!-- Risk Metrics -->
                    <div class="stats-card">
                        <div class="stats-card-title">Risk Metrics</div>
                        <div class="stats-row">
                            <span class="stats-label">Average R</span>
                            <span class="stats-value ${{data.avgRMultiple >= 0 ? 'positive' : 'negative'}}">${{formatR(data.avgRMultiple)}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Max Drawdown</span>
                            <span class="stats-value negative">${{formatDollars(-data.maxDrawdownDollars)}} (${{data.maxDrawdownPercent.toFixed(1)}}%)</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Max Consec. Wins</span>
                            <span class="stats-value positive">${{data.maxConsecWins}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Max Consec. Losses</span>
                            <span class="stats-value negative">${{data.maxConsecLosses}}</span>
                        </div>
                    </div>

                    <!-- P&L Breakdown -->
                    <div class="stats-card">
                        <div class="stats-card-title">P&L Breakdown</div>
                        <div class="stats-row">
                            <span class="stats-label">Initial Capital</span>
                            <span class="stats-value">$${{data.initialCapital.toFixed(2).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ',')}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Final Capital</span>
                            <span class="stats-value">$${{data.finalCapital.toFixed(2).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ',')}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Total P&L</span>
                            <span class="stats-value ${{data.totalPnl >= 0 ? 'positive' : 'negative'}}">${{formatDollars(data.totalPnl)}}</span>
                        </div>
                    </div>

                    <!-- Win/Loss Breakdown -->
                    <div class="stats-card">
                        <div class="stats-card-title">Win/Loss Breakdown</div>
                        <div class="stats-row">
                            <span class="stats-label">Average Win</span>
                            <span class="stats-value positive">${{formatDollars(data.avgWinDollars)}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Average Loss</span>
                            <span class="stats-value negative">${{formatDollars(data.avgLossDollars)}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Largest Win</span>
                            <span class="stats-value positive">${{formatDollars(data.largestWin)}}</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Largest Loss</span>
                            <span class="stats-value negative">${{formatDollars(data.largestLoss)}}</span>
                        </div>
                    </div>
                </div>
            `;

            // Render equity curve chart
            renderEquityCurve(data.equityCurve);
        }}

        function renderEquityCurve(equityCurveData) {{
            if (!equityCurveData || equityCurveData.length < 2) return;

            const chartContainer = document.getElementById('equity-chart');
            if (!chartContainer) return;

            // Create TradingView chart for equity curve
            equityChart = LightweightCharts.createChart(chartContainer, {{
                layout: {{
                    background: {{ color: '#1e222d' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: '#2b2b43' }},
                    horzLines: {{ color: '#2b2b43' }},
                }},
                rightPriceScale: {{
                    borderColor: '#2b2b43',
                    scaleMargins: {{
                        top: 0.1,
                        bottom: 0.1,
                    }},
                }},
                timeScale: {{
                    borderColor: '#2b2b43',
                    visible: true,
                    timeVisible: false,
                    tickMarkFormatter: (time) => `Trade ${{time}}`,
                }},
                handleScroll: false,
                handleScale: false,
            }});

            // Create area series for equity curve
            const areaSeries = equityChart.addAreaSeries({{
                topColor: 'rgba(41, 98, 255, 0.4)',
                bottomColor: 'rgba(41, 98, 255, 0.0)',
                lineColor: '#2962ff',
                lineWidth: 2,
                priceLineVisible: false,
                lastValueVisible: true,
            }});

            // Convert equity curve data to chart format
            const chartData = equityCurveData.map(d => ({{
                time: d.trade,
                value: d.capital
            }}));

            areaSeries.setData(chartData);

            // Add initial capital line
            const initialCapital = equityCurveData[0].capital;
            const initCapSeries = equityChart.addLineSeries({{
                color: '#787b86',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                priceLineVisible: false,
                lastValueVisible: false,
            }});
            initCapSeries.setData([
                {{ time: 0, value: initialCapital }},
                {{ time: equityCurveData.length - 1, value: initialCapital }}
            ]);

            equityChart.timeScale().fitContent();
        }}

        function prevSweep() {{
            if (currentSweepIdx > 0) {{
                showSweep(currentSweepIdx - 1);
            }}
        }}

        function nextSweep() {{
            if (currentSweepIdx < allSweepsData.length - 1) {{
                showSweep(currentSweepIdx + 1);
            }}
        }}

        // Initialize on load
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""

    def generate_html(self) -> Path:
        """
        Main entry point - generate the visualization HTML file.

        Returns:
            Path to generated HTML file
        """
        print(f"\n{'='*60}")
        print(f"GENERATING STRATEGY SWEEP VISUALIZATION")
        print(f"{'='*60}\n")

        # Build sweep list
        sweep_list = self._build_sweep_list()
        print(f"Found {len(sweep_list)} total sweeps:")
        print(f"  - {sum(1 for s in sweep_list if s.is_complete)} complete signals")
        print(f"  - {sum(1 for s in sweep_list if not s.is_complete)} partial setups\n")

        if len(sweep_list) == 0:
            print("No sweeps to visualize.")
            return None

        # Generate data for each sweep (static snapshots)
        print("Generating static tab data...")
        all_sweeps_data = []
        for idx, sweep_entry in enumerate(sweep_list):
            print(f"  Processing sweep {idx + 1}/{len(sweep_list)}: {sweep_entry.timestamp}")
            sweep_data = self._generate_sweep_data(sweep_entry, idx)
            all_sweeps_data.append(sweep_data)

        # Generate performance data for Dashboard
        print("Generating performance data for Dashboard...")
        performance_data = self._generate_performance_data()
        if performance_data:
            print(f"  ✓ Performance data generated ({performance_data['totalTrades']} trades)")
        else:
            print("  ⚠ No performance data available (backtest not run)")

        # Generate HTML
        print("\nCreating HTML file...")
        html_content = self._create_html_template(all_sweeps_data, performance_data)

        # Save file
        symbol = self.symbol if '/' not in self.symbol else self.symbol.replace('/','_')
        output_path = self.output_dir / f"{symbol.lower()}_sweeps.html"
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"\nGenerated: {output_path}")
        print(f"File size: {len(html_content) / 1024:.1f} KB")

        return output_path

    def run(self) -> Path:
        """Alias for generate_html() - main entry point."""
        return self.generate_html()
