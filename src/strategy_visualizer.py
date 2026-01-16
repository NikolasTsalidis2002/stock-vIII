"""
Strategy Sweep Visualizer

Unified three-tab visualization for liquidity sweep analysis.
Generates a single HTML file with 1H, 5M, and 1M timeframe tabs
that dynamically show/hide based on strategy progression.

Features:
    - Sweep dropdown to select any detected sweep
    - Three timeframe tabs (1H, 5M, 1M)
    - Frame-by-frame navigation with arrow keys
    - SMC indicators: FVG zones, OB zones, BOS lines, liquidity levels
    - TP/SL lines on 1M tab only
    - NO inflexion point markers (per spec)

Usage:
    from src.strategy_visualizer import StrategySweepVisualizer

    visualizer = StrategySweepVisualizer(strategy, symbol='TSLA')
    output_path = visualizer.run()
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategy.models import TradeSignal, PartialSetup
from src.indicators.smc_custom import smc_custom
from src.indicators.smc import smc


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
    Unified three-tab strategy visualizer for liquidity sweep analysis.

    Generates a single HTML file with 1H, 5M, and 1M timeframe tabs
    that dynamically show/hide based on strategy progression.
    """

    def __init__(
        self,
        strategy,  # MultiTimeframeStrategy - has timeframe_manager, signals, partial_setups
        symbol: str = 'TSLA',
        output_dir: Optional[Path] = None
    ):
        """
        Initialize visualizer.

        Args:
            strategy: MultiTimeframeStrategy instance with signals and partial_setups
            symbol: Stock symbol for titles
            output_dir: Custom output directory (defaults to results/)
        """
        self.strategy = strategy
        self.symbol = symbol.upper()

        if output_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            self.output_dir = project_root / 'results'
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

    def _get_strategy_state(self, sweep_entry: SweepEntry) -> str:
        """
        Determine strategy state for tab activation.

        Returns:
            One of: 'no_sweep', 'sweep_found', 'event_b_found', 'validation_found'
        """
        if sweep_entry.is_complete:
            return 'validation_found'

        conditions = sweep_entry.conditions_met
        if conditions == 1:
            return 'sweep_found'  # Only 1H sweep found
        elif conditions == 2:
            return 'event_b_found'  # Event B found, no validation
        elif conditions >= 3:
            return 'validation_found'  # Validation found (may lack confirmation)

        return 'sweep_found'

    def _generate_candle_data(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to candlestick data format."""
        candle_data = []
        for idx, row in df.iterrows():
            candle_data.append({
                'time': int(idx.timestamp()),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })
        return candle_data

    def _generate_fvg_zones(self, df: pd.DataFrame, fvg: pd.DataFrame) -> List[Dict]:
        """Generate FVG zone data for visualization."""
        fvg_zones = []

        for i in range(len(fvg)):
            if i >= len(df):
                break
            if pd.notna(fvg["FVG"].iloc[i]):
                # Get end index (mitigated or end of data)
                mitigated_idx = fvg["MitigatedIndex"].iloc[i]
                end_idx = int(mitigated_idx) if pd.notna(mitigated_idx) and mitigated_idx != 0 else len(df) - 1
                end_idx = min(end_idx, len(df) - 1)

                fvg_zones.append({
                    'startTime': int(df.index[i].timestamp()),
                    'endTime': int(df.index[end_idx].timestamp()),
                    'topPrice': float(fvg["Top"].iloc[i]),
                    'bottomPrice': float(fvg["Bottom"].iloc[i]),
                    'fvgType': int(fvg["FVG"].iloc[i])  # 1 = bullish, -1 = bearish
                })

        return fvg_zones

    def _generate_ob_zones(self, df: pd.DataFrame, ob: pd.DataFrame) -> List[Dict]:
        """Generate Order Block zone data for visualization."""
        ob_zones = []

        for i in range(len(ob)):
            if i >= len(df):
                break
            if pd.notna(ob["OB"].iloc[i]):
                end_idx = int(ob["EndIndex"].iloc[i]) if pd.notna(ob["EndIndex"].iloc[i]) else len(df) - 1
                end_idx = min(end_idx, len(df) - 1)

                ob_zones.append({
                    'startTime': int(df.index[i].timestamp()),
                    'endTime': int(df.index[end_idx].timestamp()),
                    'topPrice': float(ob["Top"].iloc[i]),
                    'bottomPrice': float(ob["Bottom"].iloc[i]),
                    'obType': int(ob["OB"].iloc[i])  # 1 = bullish, -1 = bearish
                })

        return ob_zones

    def _generate_bos_lines(self, df: pd.DataFrame, bos: pd.DataFrame, inflexions: pd.DataFrame) -> List[Dict]:
        """Generate BOS line data for visualization."""
        bos_lines = []

        for i in range(len(bos)):
            if i >= len(df):
                break
            if pd.notna(bos["BOS"].iloc[i]):
                bos_type = bos["BOS"].iloc[i]
                level = bos["Level"].iloc[i]

                # Find matching inflexion point
                matching_inflexions = []
                for j in range(len(inflexions)):
                    if j >= len(df) or j >= i:
                        continue
                    if (pd.notna(inflexions["Level"].iloc[j]) and
                        inflexions["Level"].iloc[j] == level):
                        matching_inflexions.append(j)

                if len(matching_inflexions) > 0:
                    inflexion_pos = matching_inflexions[-1]
                    bos_lines.append({
                        'startTime': int(df.index[inflexion_pos].timestamp()),
                        'endTime': int(df.index[i].timestamp()),
                        'price': float(level),
                        'color': '#089981' if bos_type == 1 else '#f23645',
                        'label': 'BOS' if bos_type == 1 else 'BOS'
                    })
                else:
                    # Fallback
                    bos_lines.append({
                        'startTime': int(df.index[i].timestamp()),
                        'endTime': int(df.index[i].timestamp()),
                        'price': float(level),
                        'color': '#089981' if bos_type == 1 else '#f23645',
                        'label': 'BOS'
                    })

        return bos_lines

    def _generate_liquidity_lines(self, df: pd.DataFrame, inflexions: pd.DataFrame) -> List[Dict]:
        """
        Generate liquidity lines for visualization.

        Shows respected inflexions (liquidity sweeps) as orange lines with X when swept.
        """
        liquidity_lines = []

        for i in range(len(inflexions)):
            if i >= len(df):
                break
            if pd.notna(inflexions["InflexionType"].iloc[i]):
                inflx_type = inflexions["InflexionType"].iloc[i]
                level = inflexions["Level"].iloc[i]
                respected = inflexions["Respected"].iloc[i]
                status_idx = inflexions["StatusIndex"].iloc[i]

                # Only show liquidity lines for pending or respected inflexions
                if respected is True:  # Liquidity was swept
                    end_idx = int(status_idx) if pd.notna(status_idx) and status_idx != 0 else len(df) - 1
                    end_idx = min(end_idx, len(df) - 1)

                    liquidity_lines.append({
                        'startTime': int(df.index[i].timestamp()),
                        'endTime': int(df.index[end_idx].timestamp()),
                        'price': float(level),
                        'color': '#ff8c00',  # Orange
                        'swept': True,
                        'lineWidth': 2
                    })
                elif respected is None:  # Pending (not yet determined)
                    end_idx = len(df) - 1

                    liquidity_lines.append({
                        'startTime': int(df.index[i].timestamp()),
                        'endTime': int(df.index[end_idx].timestamp()),
                        'price': float(level),
                        'color': '#ff8c00',  # Orange
                        'swept': False,
                        'lineWidth': 1
                    })

        return liquidity_lines

    def _generate_frame_1h(
        self,
        sweep_entry: SweepEntry,
        frame_idx: int
    ) -> Dict:
        """
        Generate 1H timeframe frame data.

        Shows all available historical data with sweep highlighted.
        """
        # Use all available data up to sweep + some context after
        sweep_time = sweep_entry.timestamp

        # Find sweep index
        sweep_idx = self.df_high.index.get_loc(sweep_time) if sweep_time in self.df_high.index else 0

        # Show context: 50 candles before and 20 after (or available)
        start_idx = max(0, sweep_idx - 50)
        end_idx = min(len(self.df_high), sweep_idx + 20)

        df_slice = self.df_high.iloc[start_idx:end_idx + 1]

        # Calculate indicators on slice
        inflexions_slice = smc_custom.inflexion_points(df_slice)
        bos_slice = smc_custom.bos(df_slice, inflexions_slice, close_break=True)
        fvg_slice = smc.fvg(df_slice, join_consecutive=True)
        ob_slice = smc_custom.ob(df_slice, bos_slice, inflexions_slice)

        # Generate data
        candle_data = self._generate_candle_data(df_slice)
        fvg_zones = self._generate_fvg_zones(df_slice, fvg_slice)
        ob_zones = self._generate_ob_zones(df_slice, ob_slice)
        bos_lines = self._generate_bos_lines(df_slice, bos_slice, inflexions_slice)
        liquidity_lines = self._generate_liquidity_lines(df_slice, inflexions_slice)

        # Highlight sweep candle
        sweep_marker = {
            'time': int(sweep_time.timestamp()),
            'position': 'aboveBar' if sweep_entry.entry_direction == 'short' else 'belowBar',
            'color': '#ff0000',
            'shape': 'arrowDown' if sweep_entry.entry_direction == 'short' else 'arrowUp',
            'text': 'SWEEP'
        }

        return {
            'timeframe': self.tf_labels.get('high', '1H'),
            'frameIdx': frame_idx,
            'candleData': candle_data,
            'fvgZones': fvg_zones,
            'obZones': ob_zones,
            'bosLines': bos_lines,
            'liquidityLines': liquidity_lines,
            'sweepMarker': sweep_marker,
            'currentTime': sweep_time.strftime('%Y-%m-%d %H:%M')
        }

    def _generate_frame_5m(
        self,
        sweep_entry: SweepEntry,
        frame_idx: int
    ) -> Optional[Dict]:
        """
        Generate 5M timeframe frame data.

        Data range depends on strategy state:
        - Before Event B: From 1H sweep start to current
        - After Event B: From exit OB start to current
        """
        state = self._get_strategy_state(sweep_entry)

        if state == 'no_sweep':
            return None

        sweep_time = sweep_entry.timestamp

        # Determine data range
        if sweep_entry.is_complete:
            signal = sweep_entry.signal
            # After Event B: use exit OB start if available
            if signal.exit_ob_start_idx is not None:
                start_time = self.df_mid.index[signal.exit_ob_start_idx]
            else:
                start_time = sweep_time
            end_time = signal.timestamp_5m_validation
        else:
            partial = sweep_entry.partial
            if partial.timestamp_5m_event_b is not None:
                # Event B found, use exit OB start if available
                if partial.exit_ob_start_time is not None:
                    start_time = partial.exit_ob_start_time
                else:
                    start_time = partial.timestamp_5m_event_b
                end_time = partial.timestamp_5m_validation or partial.timestamp_5m_event_b
            else:
                # No Event B, show from sweep time
                start_time = sweep_time
                # Find end of day or reasonable window
                end_time = sweep_time + timedelta(hours=6)

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

        # Generate data
        candle_data = self._generate_candle_data(df_slice)
        fvg_zones = self._generate_fvg_zones(df_slice, fvg_slice)
        ob_zones = self._generate_ob_zones(df_slice, ob_slice)
        bos_lines = self._generate_bos_lines(df_slice, bos_slice, inflexions_slice)
        liquidity_lines = self._generate_liquidity_lines(df_slice, inflexions_slice)

        # Event B marker
        event_b_marker = None
        if sweep_entry.is_complete:
            event_b_time = sweep_entry.signal.timestamp_5m_event_b
            if event_b_time in df_slice.index:
                event_b_marker = {
                    'time': int(event_b_time.timestamp()),
                    'position': 'aboveBar',
                    'color': '#00ff00',
                    'shape': 'circle',
                    'text': 'EB'
                }
        elif sweep_entry.partial and sweep_entry.partial.timestamp_5m_event_b:
            event_b_time = sweep_entry.partial.timestamp_5m_event_b
            if event_b_time in df_slice.index:
                event_b_marker = {
                    'time': int(event_b_time.timestamp()),
                    'position': 'aboveBar',
                    'color': '#00ff00',
                    'shape': 'circle',
                    'text': 'EB'
                }

        return {
            'timeframe': self.tf_labels.get('mid', '5M'),
            'frameIdx': frame_idx,
            'candleData': candle_data,
            'fvgZones': fvg_zones,
            'obZones': ob_zones,
            'bosLines': bos_lines,
            'liquidityLines': liquidity_lines,
            'eventBMarker': event_b_marker,
            'currentTime': end_time.strftime('%Y-%m-%d %H:%M')
        }

    def _generate_frame_1m(
        self,
        sweep_entry: SweepEntry,
        frame_idx: int
    ) -> Optional[Dict]:
        """
        Generate 1M timeframe frame data.

        Only available when validation is found.
        Shows TP/SL lines for complete signals.
        """
        state = self._get_strategy_state(sweep_entry)

        if state != 'validation_found':
            return None

        sweep_time = sweep_entry.timestamp

        # Determine data range
        if sweep_entry.is_complete:
            signal = sweep_entry.signal
            start_time = sweep_time
            end_time = signal.timestamp_1m_confirmation
        else:
            partial = sweep_entry.partial
            if partial.indices_1m is not None:
                start_idx, end_idx = partial.indices_1m
                start_time = self.df_low.index[start_idx]
                end_time = self.df_low.index[end_idx]
            else:
                return None

        # Filter data
        mask = (self.df_low.index >= start_time) & (self.df_low.index <= end_time)
        df_slice = self.df_low.loc[mask]

        if len(df_slice) == 0:
            return None

        # Calculate indicators on slice
        inflexions_slice = smc_custom.inflexion_points(df_slice)
        bos_slice = smc_custom.bos(df_slice, inflexions_slice, close_break=True)
        fvg_slice = smc.fvg(df_slice, join_consecutive=False)

        # Generate data
        candle_data = self._generate_candle_data(df_slice)
        fvg_zones = self._generate_fvg_zones(df_slice, fvg_slice)
        ob_zones = []  # Usually not calculated for 1M
        bos_lines = self._generate_bos_lines(df_slice, bos_slice, inflexions_slice)
        liquidity_lines = self._generate_liquidity_lines(df_slice, inflexions_slice)

        # TP/SL lines (only for complete signals)
        tp_line = None
        sl_line = None
        if sweep_entry.is_complete:
            signal = sweep_entry.signal
            if signal.take_profit_price is not None:
                tp_line = {
                    'price': float(signal.take_profit_price),
                    'color': '#089981',  # Green
                    'label': f'TP: ${signal.take_profit_price:.2f}'
                }
            if signal.stop_loss_price is not None:
                sl_line = {
                    'price': float(signal.stop_loss_price),
                    'color': '#f23645',  # Red
                    'label': f'SL: ${signal.stop_loss_price:.2f}'
                }

        # Confirmation marker
        confirmation_marker = None
        if sweep_entry.is_complete:
            conf_time = sweep_entry.signal.timestamp_1m_confirmation
            if conf_time in df_slice.index:
                confirmation_marker = {
                    'time': int(conf_time.timestamp()),
                    'position': 'belowBar' if sweep_entry.entry_direction == 'long' else 'aboveBar',
                    'color': '#00ff00',
                    'shape': 'arrowUp' if sweep_entry.entry_direction == 'long' else 'arrowDown',
                    'text': 'ENTRY'
                }

        return {
            'timeframe': self.tf_labels.get('low', '1M'),
            'frameIdx': frame_idx,
            'candleData': candle_data,
            'fvgZones': fvg_zones,
            'obZones': ob_zones,
            'bosLines': bos_lines,
            'liquidityLines': liquidity_lines,
            'tpLine': tp_line,
            'slLine': sl_line,
            'confirmationMarker': confirmation_marker,
            'currentTime': end_time.strftime('%Y-%m-%d %H:%M')
        }

    def _generate_sweep_frames(self, sweep_entry: SweepEntry, sweep_idx: int) -> Dict:
        """
        Generate all frames for a single sweep.

        Returns dict with frames for each timeframe.
        """
        state = self._get_strategy_state(sweep_entry)

        # Determine active tab based on state
        if state == 'no_sweep':
            active_tab = '1H'
        elif state == 'sweep_found':
            active_tab = '5M'
        elif state == 'event_b_found':
            active_tab = '5M'
        else:  # validation_found
            active_tab = '1M'

        # Generate frames for each timeframe
        frame_1h = self._generate_frame_1h(sweep_entry, sweep_idx)
        frame_5m = self._generate_frame_5m(sweep_entry, sweep_idx)
        frame_1m = self._generate_frame_1m(sweep_entry, sweep_idx)

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

        return {
            'sweepIdx': sweep_idx,
            'timestamp': sweep_entry.timestamp.strftime('%Y-%m-%d %H:%M'),
            'outcome': sweep_entry.outcome,
            'direction': sweep_entry.entry_direction.upper(),
            'state': state,
            'activeTab': active_tab,
            'conditions': conditions,
            'frame1H': frame_1h,
            'frame5M': frame_5m,
            'frame1M': frame_1m,
            'entryPrice': sweep_entry.signal.price_entry if sweep_entry.is_complete else None,
            'tpPrice': sweep_entry.signal.take_profit_price if sweep_entry.is_complete else None,
            'slPrice': sweep_entry.signal.stop_loss_price if sweep_entry.is_complete else None
        }

    def _create_html_template(self, all_sweeps_data: List[Dict]) -> str:
        """Create the complete HTML template with embedded data."""

        tf_high = self.tf_labels.get('high', '1H')
        tf_mid = self.tf_labels.get('mid', '5M')
        tf_low = self.tf_labels.get('low', '1M')

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
        .nav-controls {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .nav-btn {{
            padding: 8px 12px;
            background-color: #2962ff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
        }}
        .nav-btn:hover {{
            background-color: #1e53e5;
        }}
        .nav-btn:disabled {{
            background-color: #2b2b43;
            cursor: not-allowed;
        }}
        .frame-counter {{
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
                <div class="nav-controls">
                    <button class="nav-btn" id="prev-btn" onclick="prevSweep()">&#9664; Prev</button>
                    <span class="frame-counter" id="frame-counter">1 / 1</span>
                    <button class="nav-btn" id="next-btn" onclick="nextSweep()">Next &#9654;</button>
                </div>
            </div>
        </div>

        <div id="tabs">
            <div class="tab" data-tab="1H" onclick="switchTab('1H')">{tf_high}</div>
            <div class="tab" data-tab="5M" onclick="switchTab('5M')">{tf_mid}</div>
            <div class="tab" data-tab="1M" onclick="switchTab('1M')">{tf_low}</div>
        </div>

        <div id="main">
            <div id="chart-area">
                <div id="chart-container"></div>
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

                <div class="sidebar-section">
                    <h3>Conditions</h3>
                    <div id="conditions-list"></div>
                </div>

                <div class="sidebar-section">
                    <h3>Active</h3>
                    <div class="info-row">
                        <span class="info-label">Timeframe</span>
                        <span class="info-value" id="active-tf">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Stage</span>
                        <span class="info-value" id="active-stage">-</span>
                    </div>
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
                        <span class="shortcut-key">1 2 3</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embed all sweep data
        const allSweepsData = {json.dumps(all_sweeps_data, cls=NumpyEncoder)};

        let currentSweepIdx = 0;
        let currentTab = '1H';
        let chart = null;
        let candlestickSeries = null;
        let activeLineSeries = [];

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
                if (e.key === '2') switchTab('5M');
                if (e.key === '3') switchTab('1M');
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

            // Update frame counter
            document.getElementById('frame-counter').textContent =
                `${{idx + 1}} / ${{allSweepsData.length}}`;

            // Update navigation buttons
            document.getElementById('prev-btn').disabled = (idx === 0);
            document.getElementById('next-btn').disabled = (idx === allSweepsData.length - 1);

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

            // Update active info
            document.getElementById('active-tf').textContent = sweep.activeTab;
            const stageMap = {{
                'no_sweep': 'Searching',
                'sweep_found': 'Event B',
                'event_b_found': 'Validation',
                'validation_found': 'Execution'
            }};
            document.getElementById('active-stage').textContent = stageMap[sweep.state] || sweep.state;

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
                if (tabId === '1H' && sweep.frame1H) hasData = true;
                if (tabId === '5M' && sweep.frame5M) hasData = true;
                if (tabId === '1M' && sweep.frame1M) hasData = true;

                if (!hasData) {{
                    tab.classList.add('disabled');
                }}

                if (tabId === sweep.activeTab) {{
                    tab.classList.add('active');
                }}
            }});
        }}

        function switchTab(tabId) {{
            const sweep = allSweepsData[currentSweepIdx];

            // Check if tab is disabled
            let hasData = false;
            if (tabId === '1H' && sweep.frame1H) hasData = true;
            if (tabId === '5M' && sweep.frame5M) hasData = true;
            if (tabId === '1M' && sweep.frame1M) hasData = true;

            if (!hasData) return;

            // Update tab UI
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
                if (tab.dataset.tab === tabId) tab.classList.add('active');
            }});

            currentTab = tabId;
            showChart(sweep);
        }}

        function showChart(sweep) {{
            // Get frame data for current tab
            let frameData = null;
            if (currentTab === '1H') frameData = sweep.frame1H;
            else if (currentTab === '5M') frameData = sweep.frame5M;
            else if (currentTab === '1M') frameData = sweep.frame1M;

            if (!frameData) return;

            // Clear existing lines
            clearLineSeries();

            // Update candlesticks
            candlestickSeries.setData(frameData.candleData);

            // Set markers
            const markers = [];
            if (frameData.sweepMarker) markers.push(frameData.sweepMarker);
            if (frameData.eventBMarker) markers.push(frameData.eventBMarker);
            if (frameData.confirmationMarker) markers.push(frameData.confirmationMarker);
            candlestickSeries.setMarkers(markers);

            // Draw BOS lines
            if (frameData.bosLines) {{
                frameData.bosLines.forEach(bos => {{
                    const lineSeries = chart.addLineSeries({{
                        color: bos.color,
                        lineWidth: 2,
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
            if (frameData.liquidityLines) {{
                frameData.liquidityLines.forEach(liq => {{
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

            // Draw TP/SL lines (1M only)
            if (frameData.tpLine) {{
                const tpSeries = chart.addLineSeries({{
                    color: frameData.tpLine.color,
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false,
                    lastValueVisible: true,
                    title: 'TP',
                }});
                // Extend across visible range
                const firstTime = frameData.candleData[0].time;
                const lastTime = frameData.candleData[frameData.candleData.length - 1].time;
                tpSeries.setData([
                    {{ time: firstTime, value: frameData.tpLine.price }},
                    {{ time: lastTime, value: frameData.tpLine.price }}
                ]);
                activeLineSeries.push(tpSeries);
            }}

            if (frameData.slLine) {{
                const slSeries = chart.addLineSeries({{
                    color: frameData.slLine.color,
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false,
                    lastValueVisible: true,
                    title: 'SL',
                }});
                const firstTime = frameData.candleData[0].time;
                const lastTime = frameData.candleData[frameData.candleData.length - 1].time;
                slSeries.setData([
                    {{ time: firstTime, value: frameData.slLine.price }},
                    {{ time: lastTime, value: frameData.slLine.price }}
                ]);
                activeLineSeries.push(slSeries);
            }}

            // Fit content
            chart.timeScale().fitContent();

            // Redraw overlays (FVG, OB zones)
            setTimeout(redrawOverlays, 50);
        }}

        function clearLineSeries() {{
            activeLineSeries.forEach(series => {{
                try {{ chart.removeSeries(series); }} catch(e) {{}}
            }});
            activeLineSeries = [];
        }}

        function redrawOverlays() {{
            const sweep = allSweepsData[currentSweepIdx];
            let frameData = null;
            if (currentTab === '1H') frameData = sweep.frame1H;
            else if (currentTab === '5M') frameData = sweep.frame5M;
            else if (currentTab === '1M') frameData = sweep.frame1M;

            if (!frameData) return;

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
            if (frameData.fvgZones) {{
                frameData.fvgZones.forEach(zone => {{
                    const x1 = timeScale.timeToCoordinate(zone.startTime);
                    const x2 = timeScale.timeToCoordinate(zone.endTime);
                    const y1 = candlestickSeries.priceToCoordinate(zone.topPrice);
                    const y2 = candlestickSeries.priceToCoordinate(zone.bottomPrice);

                    if (x1 !== null && x2 !== null && y1 !== null && y2 !== null) {{
                        const x = Math.min(x1, x2);
                        const y = Math.min(y1, y2);
                        const width = Math.abs(x2 - x1);
                        const height = Math.abs(y2 - y1);

                        svg.append('rect')
                            .attr('class', 'fvg')
                            .attr('x', x)
                            .attr('y', y)
                            .attr('width', width)
                            .attr('height', height)
                            .attr('fill', 'rgba(255, 235, 59, 0.2)')
                            .attr('stroke', '#ffeb3b')
                            .attr('stroke-width', 1)
                            .attr('stroke-dasharray', '4,4');
                    }}
                }});
            }}

            // Draw OB zones
            if (frameData.obZones) {{
                frameData.obZones.forEach(zone => {{
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

        # Generate frames for each sweep
        print("Generating frame data...")
        all_sweeps_data = []
        for idx, sweep_entry in enumerate(sweep_list):
            print(f"  Processing sweep {idx + 1}/{len(sweep_list)}: {sweep_entry.timestamp}")
            sweep_data = self._generate_sweep_frames(sweep_entry, idx)
            all_sweeps_data.append(sweep_data)

        # Generate HTML
        print("\nCreating HTML file...")
        html_content = self._create_html_template(all_sweeps_data)

        # Save file
        output_path = self.output_dir / f"{self.symbol.lower()}_sweeps.html"
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"\nGenerated: {output_path}")
        print(f"File size: {len(html_content) / 1024:.1f} KB")

        return output_path

    def run(self) -> Path:
        """Alias for generate_html() - main entry point."""
        return self.generate_html()


# Alias for backwards compatibility with existing import
StrategySweepVisualizer = StrategySweepVisualizer
