"""
Strategy Sweep Visualizer

Generates a SINGLE interactive HTML file showing ALL liquidity sweeps.
Features:
- Dropdown to select any sweep (successful or failed)
- Tabs for 1H/5M/1M timeframes
- Frame-by-frame candle navigation with arrow keys
- Clean visualization with zones only (no circle markers)
- All SMC indicators: Order Blocks, FVG, BOS lines, Exit OB

Usage:
    from src.tradingview_strategy_analyzer import StrategySweepVisualizer

    visualizer = StrategySweepVisualizer(strategy)
    visualizer.generate_html()  # Creates results/{symbol}_sweeps.html
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional, Union
from datetime import timedelta
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategy import MultiTimeframeStrategy, TradeSignal, PartialSetup
from src.indicators.smc_custom import smc_custom
from src.indicators.smc import smc


@dataclass
class SweepData:
    """Unified data structure for both complete and partial sweeps."""
    sweep_id: int
    timestamp: pd.Timestamp
    outcome: str  # "SUCCESS (+$X.XX)" or "FAILED (reason)"
    is_success: bool
    entry_direction: str
    conditions_met: int

    # From TradeSignal or PartialSetup
    timestamp_1h_sweep: pd.Timestamp
    timestamp_5m_event_b: Optional[pd.Timestamp]
    timestamp_5m_validation: Optional[pd.Timestamp]
    timestamp_1m_confirmation: Optional[pd.Timestamp]

    # Prices
    price_1h_sweep: float
    take_profit: Optional[float]
    stop_loss: Optional[float]
    profit_loss: Optional[float]

    # Exit OB
    exit_ob_top: Optional[float]
    exit_ob_bottom: Optional[float]

    # Equilibrium
    equilibrium_level: Optional[float]


class StrategySweepVisualizer:
    """
    Generates unified HTML visualization for all liquidity sweeps.
    """

    def __init__(self, strategy: MultiTimeframeStrategy, symbol: str = "STOCK"):
        """
        Initialize visualizer.

        Args:
            strategy: MultiTimeframeStrategy instance with signals and partial_setups
            symbol: Stock symbol for filename
        """
        self.strategy = strategy
        self.symbol = symbol

        # Setup output directory
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        self.output_dir = project_root / 'results'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Combine signals and partials into unified list
        self.sweeps = self._build_sweep_list()

    def _build_sweep_list(self) -> List[SweepData]:
        """Build unified list of all sweeps (successful and failed)."""
        sweeps = []
        sweep_id = 0

        # Add successful trades
        for signal in self.strategy.signals:
            profit = 0.0
            if signal.take_profit_price and signal.price_entry:
                if signal.entry_direction == 'long':
                    profit = signal.take_profit_price - signal.price_entry
                else:
                    profit = signal.price_entry - signal.take_profit_price

            sweeps.append(SweepData(
                sweep_id=sweep_id,
                timestamp=signal.timestamp_1h_sweep,
                outcome=f"SUCCESS (+${profit:.2f})" if profit >= 0 else f"SUCCESS (${profit:.2f})",
                is_success=True,
                entry_direction=signal.entry_direction,
                conditions_met=4,
                timestamp_1h_sweep=signal.timestamp_1h_sweep,
                timestamp_5m_event_b=signal.timestamp_5m_event_b,
                timestamp_5m_validation=signal.timestamp_5m_validation,
                timestamp_1m_confirmation=signal.timestamp_1m_confirmation,
                price_1h_sweep=signal.price_1h_sweep,
                take_profit=signal.take_profit_price,
                stop_loss=signal.stop_loss_price,
                profit_loss=profit,
                exit_ob_top=signal.exit_ob_top,
                exit_ob_bottom=signal.exit_ob_bottom,
                equilibrium_level=signal.equilibrium_level
            ))
            sweep_id += 1

        # Add failed/partial setups
        for partial in self.strategy.partial_setups:
            sweeps.append(SweepData(
                sweep_id=sweep_id,
                timestamp=partial.timestamp_1h_sweep,
                outcome=f"FAILED ({partial.failure_reason})",
                is_success=False,
                entry_direction=partial.entry_direction,
                conditions_met=partial.conditions_met,
                timestamp_1h_sweep=partial.timestamp_1h_sweep,
                timestamp_5m_event_b=partial.timestamp_5m_event_b,
                timestamp_5m_validation=partial.timestamp_5m_validation,
                timestamp_1m_confirmation=partial.timestamp_1m_confirmation,
                price_1h_sweep=partial.price_1h_sweep,
                take_profit=None,
                stop_loss=None,
                profit_loss=None,
                exit_ob_top=partial.exit_ob_top,
                exit_ob_bottom=partial.exit_ob_bottom,
                equilibrium_level=None
            ))
            sweep_id += 1

        # Sort by timestamp
        sweeps.sort(key=lambda x: x.timestamp)

        # Reassign IDs after sorting
        for i, sweep in enumerate(sweeps):
            sweep.sweep_id = i

        return sweeps

    def _generate_frames_for_sweep(self, sweep: SweepData) -> List[Dict[str, Any]]:
        """Generate all frames for a single sweep."""
        frames = []

        # Phase 1: 1H candles leading up to sweep (10 candles before)
        start_1h = sweep.timestamp_1h_sweep - timedelta(hours=10)
        candles_before = self.strategy.df_1h[
            (self.strategy.df_1h.index >= start_1h) &
            (self.strategy.df_1h.index < sweep.timestamp_1h_sweep)
        ]

        for i, ts in enumerate(candles_before.index):
            frames.append(self._create_frame(
                sweep=sweep,
                current_time=ts,
                status=f"Scanning 1H for liquidity sweep... ({i+1}/{len(candles_before)+1})",
                active_tab='1h',
                visible_tabs=['1h'],
                conditions=[False, False, False, False]
            ))

        # Phase 2: Sweep detected
        frames.append(self._create_frame(
            sweep=sweep,
            current_time=sweep.timestamp_1h_sweep,
            status=f"✓ LIQUIDITY SWEEP DETECTED\nDirection: {sweep.entry_direction.upper()}\nPrice: ${sweep.price_1h_sweep:.2f}",
            active_tab='1h',
            visible_tabs=['1h', '5m'],
            conditions=[True, False, False, False]
        ))

        # Phase 3: 5M candles for Event B
        if sweep.timestamp_5m_event_b:
            candles_5m = self.strategy.df_5m[
                (self.strategy.df_5m.index > sweep.timestamp_1h_sweep) &
                (self.strategy.df_5m.index <= sweep.timestamp_5m_event_b)
            ]

            for i, ts in enumerate(candles_5m.index):
                is_event_b = (ts == sweep.timestamp_5m_event_b)
                frames.append(self._create_frame(
                    sweep=sweep,
                    current_time=ts,
                    status="✓ EVENT B DETECTED" if is_event_b else f"Scanning 5M for Event B... ({i+1}/{len(candles_5m)})",
                    active_tab='5m',
                    visible_tabs=['1h', '5m'],
                    conditions=[True, is_event_b, False, False]
                ))
        else:
            # Failed at Event B - show some 5M candles
            end_5m = sweep.timestamp_1h_sweep + timedelta(hours=2)
            candles_5m = self.strategy.df_5m[
                (self.strategy.df_5m.index > sweep.timestamp_1h_sweep) &
                (self.strategy.df_5m.index <= end_5m)
            ]
            for i, ts in enumerate(candles_5m.index[:20]):  # Limit to 20 candles
                frames.append(self._create_frame(
                    sweep=sweep,
                    current_time=ts,
                    status=f"Scanning 5M for Event B... ({i+1})\n✗ {sweep.outcome}",
                    active_tab='5m',
                    visible_tabs=['1h', '5m'],
                    conditions=[True, False, False, False]
                ))

        # Phase 4: 5M candles for Validation
        if sweep.timestamp_5m_validation and sweep.timestamp_5m_event_b:
            candles_5m_val = self.strategy.df_5m[
                (self.strategy.df_5m.index > sweep.timestamp_5m_event_b) &
                (self.strategy.df_5m.index <= sweep.timestamp_5m_validation)
            ]

            for i, ts in enumerate(candles_5m_val.index):
                is_validation = (ts == sweep.timestamp_5m_validation)
                frames.append(self._create_frame(
                    sweep=sweep,
                    current_time=ts,
                    status="✓ EQUILIBRIUM ZONE ENTERED" if is_validation else f"Scanning for validation... ({i+1}/{len(candles_5m_val)})",
                    active_tab='5m',
                    visible_tabs=['1h', '5m', '1m'] if is_validation else ['1h', '5m'],
                    conditions=[True, True, is_validation, False]
                ))

        # Phase 5: 1M candles for Confirmation
        if sweep.timestamp_1m_confirmation and sweep.timestamp_5m_validation:
            candles_1m = self.strategy.df_1m[
                (self.strategy.df_1m.index > sweep.timestamp_5m_validation) &
                (self.strategy.df_1m.index <= sweep.timestamp_1m_confirmation)
            ]

            for i, ts in enumerate(candles_1m.index):
                is_confirm = (ts == sweep.timestamp_1m_confirmation)
                frames.append(self._create_frame(
                    sweep=sweep,
                    current_time=ts,
                    status="✓ CONFIRMATION - TRADE ENTRY!" if is_confirm else f"Scanning 1M for confirmation... ({i+1}/{len(candles_1m)})",
                    active_tab='1m',
                    visible_tabs=['1h', '5m', '1m'],
                    conditions=[True, True, True, is_confirm]
                ))

        # Final frame for successful trades
        if sweep.is_success and sweep.timestamp_1m_confirmation:
            frames.append(self._create_frame(
                sweep=sweep,
                current_time=sweep.timestamp_1m_confirmation + timedelta(minutes=5),
                status=f"TRADE COMPLETE\n{sweep.outcome}\nTP: ${sweep.take_profit:.2f}\nSL: ${sweep.stop_loss:.2f}",
                active_tab='5m',
                visible_tabs=['1h', '5m', '1m'],
                conditions=[True, True, True, True]
            ))

        return frames

    def _create_frame(
        self,
        sweep: SweepData,
        current_time: pd.Timestamp,
        status: str,
        active_tab: str,
        visible_tabs: List[str],
        conditions: List[bool]
    ) -> Dict[str, Any]:
        """Create a single frame with data for all timeframes."""

        frame = {
            'currentTime': current_time.strftime('%Y-%m-%d %H:%M'),
            'currentTimeUnix': int(current_time.timestamp()),
            'status': status,
            'activeTab': active_tab,
            'visibleTabs': visible_tabs,
            'conditions': conditions,
            'timeframes': {}
        }

        # 1H data - always visible, show all data up to current time
        df_1h = self.strategy.df_1h[self.strategy.df_1h.index <= current_time]
        frame['timeframes']['1h'] = self._process_timeframe(df_1h, sweep, '1h', current_time)

        # 5M data - visible after sweep, windowed from sweep time
        if '5m' in visible_tabs and current_time >= sweep.timestamp_1h_sweep:
            df_5m = self.strategy.df_5m[
                (self.strategy.df_5m.index >= sweep.timestamp_1h_sweep) &
                (self.strategy.df_5m.index <= current_time)
            ]
            frame['timeframes']['5m'] = self._process_timeframe(df_5m, sweep, '5m', current_time)
        else:
            frame['timeframes']['5m'] = {'candleData': [], 'obZones': [], 'fvgZones': [], 'bosLines': [], 'stats': {}}

        # 1M data - visible after validation, windowed from validation time
        if '1m' in visible_tabs and sweep.timestamp_5m_validation and current_time >= sweep.timestamp_5m_validation:
            df_1m = self.strategy.df_1m[
                (self.strategy.df_1m.index >= sweep.timestamp_5m_validation) &
                (self.strategy.df_1m.index <= current_time)
            ]
            frame['timeframes']['1m'] = self._process_timeframe(df_1m, sweep, '1m', current_time)
        else:
            frame['timeframes']['1m'] = {'candleData': [], 'obZones': [], 'fvgZones': [], 'bosLines': [], 'stats': {}}

        return frame

    def _process_timeframe(
        self,
        df: pd.DataFrame,
        sweep: SweepData,
        timeframe: str,
        current_time: pd.Timestamp
    ) -> Dict[str, Any]:
        """Process a single timeframe and extract visualization data."""

        result = {
            'candleData': [],
            'obZones': [],
            'fvgZones': [],
            'bosLines': [],
            'stats': {
                'totalCandles': len(df),
                'totalOb': 0,
                'totalFvg': 0,
                'totalBos': 0
            }
        }

        if len(df) == 0:
            return result

        # Convert OHLC data
        for idx, row in df.iterrows():
            result['candleData'].append({
                'time': int(idx.timestamp()),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })

        # Calculate indicators
        if len(df) > 5:  # Need minimum candles for indicators
            try:
                inflexions = smc_custom.inflexion_points(df)
                bos = smc_custom.bos(df, inflexions, close_break=True)
                fvg = smc.fvg(df, join_consecutive=True)
                ob = smc_custom.ob(df, bos, inflexions)

                # Process Order Blocks (NO markers, just zones)
                for i in range(len(ob["OB"])):
                    if not np.isnan(ob["OB"].iloc[i]):
                        ob_type = ob["OB"].iloc[i]
                        end_idx = int(ob["EndIndex"].iloc[i]) if not np.isnan(ob["EndIndex"].iloc[i]) else len(df) - 1

                        result['obZones'].append({
                            'startTime': int(df.index[i].timestamp()),
                            'endTime': int(df.index[min(end_idx, len(df) - 1)].timestamp()),
                            'topPrice': float(ob["Top"].iloc[i]),
                            'bottomPrice': float(ob["Bottom"].iloc[i]),
                            'obType': int(ob_type)
                        })
                        result['stats']['totalOb'] += 1

                # Process FVG zones
                for i in range(len(fvg["FVG"])):
                    if not np.isnan(fvg["FVG"][i]):
                        end_idx = int(fvg["MitigatedIndex"][i]) if fvg["MitigatedIndex"][i] != 0 else len(df) - 1

                        result['fvgZones'].append({
                            'startTime': int(df.index[i].timestamp()),
                            'endTime': int(df.index[min(end_idx, len(df) - 1)].timestamp()),
                            'topPrice': float(fvg["Top"][i]),
                            'bottomPrice': float(fvg["Bottom"][i])
                        })
                        result['stats']['totalFvg'] += 1

                # Process BOS lines
                for i in range(len(bos)):
                    if not np.isnan(bos["BOS"].iloc[i]):
                        bos_type = bos["BOS"].iloc[i]
                        level = bos["Level"].iloc[i]

                        # Find matching inflexion for line start
                        for j in range(len(inflexions)):
                            if (not np.isnan(inflexions["Level"].iloc[j]) and
                                inflexions["Level"].iloc[j] == level and j < i):

                                result['bosLines'].append({
                                    'startTime': int(df.index[j].timestamp()),
                                    'endTime': int(df.index[i].timestamp()),
                                    'price': float(level),
                                    'bosType': int(bos_type)
                                })
                                result['stats']['totalBos'] += 1
                                break
            except Exception as e:
                pass  # Silently handle indicator calculation errors

        # Add Exit OB zone on 5M if available
        if timeframe == '5m' and sweep.exit_ob_top and sweep.exit_ob_bottom:
            result['obZones'].append({
                'startTime': int(sweep.timestamp_1h_sweep.timestamp()),
                'endTime': int(current_time.timestamp()),
                'topPrice': float(sweep.exit_ob_top),
                'bottomPrice': float(sweep.exit_ob_bottom),
                'obType': 'exit'  # Special type for purple styling
            })

        # Add Equilibrium zone on 5M if available
        if timeframe == '5m' and sweep.equilibrium_level and sweep.price_1h_sweep:
            eq_top = max(sweep.equilibrium_level, sweep.price_1h_sweep)
            eq_bottom = min(sweep.equilibrium_level, sweep.price_1h_sweep)
            result['equilibriumZone'] = {
                'topPrice': float(eq_top),
                'bottomPrice': float(eq_bottom),
                'centerPrice': float(sweep.equilibrium_level)
            }

        return result

    def generate_html(self) -> str:
        """Generate the unified HTML file with all sweeps."""

        print(f"\n{'='*60}")
        print(f"Generating Sweep Visualization for {self.symbol}")
        print(f"{'='*60}")
        print(f"  Total sweeps: {len(self.sweeps)}")
        print(f"  Successful: {sum(1 for s in self.sweeps if s.is_success)}")
        print(f"  Failed: {sum(1 for s in self.sweeps if not s.is_success)}")

        # Generate frames for each sweep
        all_sweep_data = []
        for sweep in self.sweeps:
            print(f"\n  Processing sweep {sweep.sweep_id + 1}: {sweep.timestamp.strftime('%Y-%m-%d %H:%M')} - {sweep.outcome[:20]}...")
            frames = self._generate_frames_for_sweep(sweep)

            all_sweep_data.append({
                'sweepId': sweep.sweep_id,
                'label': f"{sweep.timestamp.strftime('%b %d %H:%M')} - {sweep.outcome}",
                'isSuccess': sweep.is_success,
                'direction': sweep.entry_direction,
                'conditionsMet': sweep.conditions_met,
                'takeProfit': sweep.take_profit,
                'stopLoss': sweep.stop_loss,
                'profitLoss': sweep.profit_loss,
                'frames': frames
            })
            print(f"    Generated {len(frames)} frames")

        # Generate HTML
        html = self._create_html(all_sweep_data)

        # Save file
        output_path = self.output_dir / f"{self.symbol}_sweeps.html"
        with open(output_path, 'w') as f:
            f.write(html)

        file_size = len(html) / 1024
        print(f"\n✓ Saved: {output_path}")
        print(f"✓ File size: {file_size:.1f} KB")

        return str(output_path)

    def _create_html(self, all_sweep_data: List[Dict]) -> str:
        """Create the HTML template with embedded data."""

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.symbol} Liquidity Sweeps</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #131722;
            color: #d1d4dc;
            height: 100vh;
            overflow: hidden;
        }}

        #container {{
            display: grid;
            grid-template-rows: 60px 50px 1fr;
            height: 100vh;
        }}

        #header {{
            background: #1e222d;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2b2b43;
        }}

        #header h1 {{
            font-size: 16px;
            color: #2962ff;
        }}

        #controls {{
            display: flex;
            gap: 15px;
            align-items: center;
        }}

        select {{
            background: #1e222d;
            color: #d1d4dc;
            border: 1px solid #363a45;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 13px;
            min-width: 280px;
        }}

        select option.success {{ color: #089981; }}
        select option.failed {{ color: #f23645; }}

        button {{
            background: #2962ff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
        }}

        button:hover {{ background: #1e53e5; }}
        button:disabled {{ background: #2b2b43; cursor: not-allowed; }}

        #frame-info {{
            font-size: 13px;
            min-width: 120px;
            text-align: center;
        }}

        #tab-bar {{
            background: #1e222d;
            padding: 0 20px;
            display: flex;
            align-items: flex-end;
            gap: 5px;
            border-bottom: 1px solid #2b2b43;
        }}

        .tab {{
            padding: 12px 24px;
            background: #131722;
            border: 1px solid #2b2b43;
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            color: #787b86;
            display: none;
        }}

        .tab.visible {{ display: block; }}
        .tab:hover {{ background: #1e222d; color: #d1d4dc; }}
        .tab.active {{ background: #131722; color: #2962ff; border-color: #2962ff; }}

        #main {{
            display: grid;
            grid-template-columns: 1fr 280px;
            overflow: hidden;
        }}

        #chart-area {{
            position: relative;
            padding: 10px;
        }}

        .chart-wrapper {{
            position: absolute;
            top: 10px; left: 10px; right: 10px; bottom: 10px;
            background: #1e222d;
            border-radius: 4px;
            display: none;
        }}

        .chart-wrapper.active {{ display: block; }}

        .chart-header {{
            padding: 12px 16px;
            background: #131722;
            border-bottom: 1px solid #2b2b43;
            font-weight: 600;
            font-size: 14px;
        }}

        .chart-content {{
            position: relative;
            height: calc(100% - 50px);
        }}

        #sidebar {{
            background: #1e222d;
            padding: 20px;
            overflow-y: auto;
            border-left: 1px solid #2b2b43;
        }}

        .sidebar-section {{
            margin-bottom: 20px;
            padding: 15px;
            background: #131722;
            border-radius: 4px;
        }}

        .sidebar-section h3 {{
            font-size: 12px;
            color: #787b86;
            text-transform: uppercase;
            margin-bottom: 12px;
        }}

        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 13px;
        }}

        .stat-label {{ color: #787b86; }}
        .stat-value {{ font-weight: 600; }}
        .stat-value.success {{ color: #089981; }}
        .stat-value.failed {{ color: #f23645; }}

        .condition {{
            margin: 8px 0;
            padding: 8px;
            background: #0c0e12;
            border-radius: 4px;
            border-left: 3px solid #2b2b43;
            font-size: 12px;
        }}

        .condition.met {{ border-left-color: #089981; }}
        .condition.met::before {{ content: "✓ "; color: #089981; }}
        .condition:not(.met)::before {{ content: "○ "; color: #787b86; }}

        #status-box {{
            padding: 12px;
            background: #1a1d24;
            border-radius: 4px;
            font-size: 12px;
            white-space: pre-line;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="header">
            <h1>{self.symbol} Liquidity Sweeps</h1>
            <div id="controls">
                <select id="sweep-select" onchange="selectSweep(this.value)">
                    {self._generate_options(all_sweep_data)}
                </select>
                <button onclick="prevFrame()">◀ Prev</button>
                <span id="frame-info">Frame 1/1</span>
                <button onclick="nextFrame()">Next ▶</button>
            </div>
        </div>

        <div id="tab-bar">
            <div class="tab visible" id="tab-1h" onclick="switchTab('1h')">1H Timeframe</div>
            <div class="tab" id="tab-5m" onclick="switchTab('5m')">5M Timeframe</div>
            <div class="tab" id="tab-1m" onclick="switchTab('1m')">1M Timeframe</div>
        </div>

        <div id="main">
            <div id="chart-area">
                <div class="chart-wrapper active" id="wrapper-1h">
                    <div class="chart-header">1H - Liquidity Sweep Detection</div>
                    <div class="chart-content" id="chart-1h"></div>
                </div>
                <div class="chart-wrapper" id="wrapper-5m">
                    <div class="chart-header">5M - Event B & Validation</div>
                    <div class="chart-content" id="chart-5m"></div>
                </div>
                <div class="chart-wrapper" id="wrapper-1m">
                    <div class="chart-header">1M - Final Confirmation</div>
                    <div class="chart-content" id="chart-1m"></div>
                </div>
            </div>

            <div id="sidebar">
                <div class="sidebar-section">
                    <h3>Status</h3>
                    <div id="status-box">Loading...</div>
                </div>

                <div class="sidebar-section">
                    <h3>Sweep Info</h3>
                    <div class="stat-row">
                        <span class="stat-label">Outcome</span>
                        <span class="stat-value" id="outcome">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Direction</span>
                        <span class="stat-value" id="direction">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Take Profit</span>
                        <span class="stat-value" id="tp">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Stop Loss</span>
                        <span class="stat-value" id="sl">-</span>
                    </div>
                </div>

                <div class="sidebar-section">
                    <h3>Conditions</h3>
                    <div class="condition" id="cond-1">1H Liquidity Sweep</div>
                    <div class="condition" id="cond-2">5M Event B</div>
                    <div class="condition" id="cond-3">5M Validation</div>
                    <div class="condition" id="cond-4">1M Confirmation</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const allSweeps = {json.dumps(all_sweep_data)};

        let currentSweepIdx = 0;
        let currentFrameIdx = 0;
        let currentTab = '1h';

        const charts = {{}};
        const series = {{}};
        const svgOverlays = {{}};
        const lineSeries = {{ '1h': [], '5m': [], '1m': [] }};
        const initialized = {{ '1h': false, '5m': false, '1m': false }};

        function initChart(tf) {{
            if (initialized[tf]) return;

            const container = document.getElementById(`chart-${{tf}}`);
            if (!container) return;

            charts[tf] = LightweightCharts.createChart(container, {{
                layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
                grid: {{ vertLines: {{ color: '#1e222d' }}, horzLines: {{ color: '#1e222d' }} }},
                rightPriceScale: {{ borderColor: '#2b2b43' }},
                timeScale: {{ borderColor: '#2b2b43', timeVisible: true }}
            }});

            series[tf] = charts[tf].addCandlestickSeries({{
                upColor: '#089981',
                downColor: '#f23645',
                borderVisible: false,
                wickUpColor: '#089981',
                wickDownColor: '#f23645'
            }});

            svgOverlays[tf] = d3.select(container)
                .append('svg')
                .style('position', 'absolute')
                .style('top', '0')
                .style('left', '0')
                .style('pointer-events', 'none');

            new ResizeObserver(() => {{
                charts[tf].applyOptions({{ width: container.clientWidth, height: container.clientHeight }});
                drawZones(tf);
            }}).observe(container);

            charts[tf].timeScale().subscribeVisibleLogicalRangeChange(() => drawZones(tf));

            initialized[tf] = true;
        }}

        function drawZones(tf) {{
            if (!svgOverlays[tf] || !charts[tf]) return;

            const svg = svgOverlays[tf];
            const container = document.getElementById(`chart-${{tf}}`);
            const rect = container.getBoundingClientRect();

            svg.attr('width', rect.width).attr('height', rect.height);
            svg.selectAll('rect').remove();

            const sweep = allSweeps[currentSweepIdx];
            const frame = sweep.frames[currentFrameIdx];
            const tfData = frame.timeframes[tf];

            if (!tfData) return;

            const timeScale = charts[tf].timeScale();

            // Draw FVG zones (yellow)
            (tfData.fvgZones || []).forEach(zone => {{
                drawRect(svg, timeScale, series[tf], zone, 'rgba(255, 235, 59, 0.2)', '#ffeb3b', true);
            }});

            // Draw OB zones (cyan/orange/purple)
            (tfData.obZones || []).forEach(zone => {{
                let fill, stroke, dashed = false;
                if (zone.obType === 'exit') {{
                    fill = 'rgba(156, 39, 176, 0.3)';
                    stroke = '#9c27b0';
                    dashed = true;
                }} else if (zone.obType === 1) {{
                    fill = 'rgba(0, 188, 212, 0.2)';
                    stroke = '#00bcd4';
                }} else {{
                    fill = 'rgba(255, 87, 34, 0.2)';
                    stroke = '#ff5722';
                }}
                drawRect(svg, timeScale, series[tf], zone, fill, stroke, dashed);
            }});

            // Draw equilibrium zone (blue)
            if (tfData.equilibriumZone) {{
                const eq = tfData.equilibriumZone;
                const firstCandle = tfData.candleData[0];
                const lastCandle = tfData.candleData[tfData.candleData.length - 1];
                if (firstCandle && lastCandle) {{
                    drawRect(svg, timeScale, series[tf], {{
                        startTime: firstCandle.time,
                        endTime: lastCandle.time,
                        topPrice: eq.topPrice,
                        bottomPrice: eq.bottomPrice
                    }}, 'rgba(33, 150, 243, 0.15)', '#2196f3', false);
                }}
            }}
        }}

        function drawRect(svg, timeScale, priceSeries, zone, fill, stroke, dashed) {{
            const x1 = timeScale.timeToCoordinate(zone.startTime);
            const x2 = timeScale.timeToCoordinate(zone.endTime);
            const y1 = priceSeries.priceToCoordinate(zone.topPrice);
            const y2 = priceSeries.priceToCoordinate(zone.bottomPrice);

            if (x1 === null || x2 === null || y1 === null || y2 === null) return;

            const r = svg.append('rect')
                .attr('x', Math.min(x1, x2))
                .attr('y', Math.min(y1, y2))
                .attr('width', Math.abs(x2 - x1))
                .attr('height', Math.abs(y2 - y1))
                .attr('fill', fill)
                .attr('stroke', stroke)
                .attr('stroke-width', 1);

            if (dashed) r.attr('stroke-dasharray', '4,4');
        }}

        function clearLines(tf) {{
            lineSeries[tf].forEach(s => {{ if (charts[tf]) charts[tf].removeSeries(s); }});
            lineSeries[tf] = [];
        }}

        function updateChart(tf) {{
            if (!charts[tf]) return;

            const sweep = allSweeps[currentSweepIdx];
            const frame = sweep.frames[currentFrameIdx];
            const tfData = frame.timeframes[tf];

            if (tfData && tfData.candleData && tfData.candleData.length > 0) {{
                series[tf].setData(tfData.candleData);
            }} else {{
                series[tf].setData([]);
            }}

            clearLines(tf);

            // Draw BOS lines
            if (tfData && tfData.bosLines) {{
                tfData.bosLines.forEach(bos => {{
                    const ls = charts[tf].addLineSeries({{
                        color: bos.bosType === 1 ? '#089981' : '#f23645',
                        lineWidth: 2,
                        priceLineVisible: false,
                        lastValueVisible: false
                    }});
                    ls.setData([
                        {{ time: bos.startTime, value: bos.price }},
                        {{ time: bos.endTime, value: bos.price }}
                    ]);
                    lineSeries[tf].push(ls);
                }});
            }}

            drawZones(tf);
            charts[tf].timeScale().fitContent();
        }}

        function switchTab(tf) {{
            const sweep = allSweeps[currentSweepIdx];
            const frame = sweep.frames[currentFrameIdx];

            if (!frame.visibleTabs.includes(tf)) return;

            currentTab = tf;

            ['1h', '5m', '1m'].forEach(t => {{
                const tab = document.getElementById(`tab-${{t}}`);
                const wrapper = document.getElementById(`wrapper-${{t}}`);

                tab.classList.toggle('visible', frame.visibleTabs.includes(t));
                tab.classList.toggle('active', t === tf);
                wrapper.classList.toggle('active', t === tf);
            }});

            initChart(tf);
            updateChart(tf);
        }}

        function showFrame() {{
            const sweep = allSweeps[currentSweepIdx];
            const frame = sweep.frames[currentFrameIdx];

            // Update frame info
            document.getElementById('frame-info').textContent = `Frame ${{currentFrameIdx + 1}}/${{sweep.frames.length}}`;

            // Update status
            document.getElementById('status-box').textContent = frame.status;

            // Update sidebar
            document.getElementById('outcome').textContent = sweep.isSuccess ? 'SUCCESS' : 'FAILED';
            document.getElementById('outcome').className = 'stat-value ' + (sweep.isSuccess ? 'success' : 'failed');
            document.getElementById('direction').textContent = sweep.direction.toUpperCase();
            document.getElementById('tp').textContent = sweep.takeProfit ? `$${{sweep.takeProfit.toFixed(2)}}` : '-';
            document.getElementById('sl').textContent = sweep.stopLoss ? `$${{sweep.stopLoss.toFixed(2)}}` : '-';

            // Update conditions
            for (let i = 0; i < 4; i++) {{
                const el = document.getElementById(`cond-${{i + 1}}`);
                el.classList.toggle('met', frame.conditions[i]);
            }}

            // Update tabs and chart
            ['1h', '5m', '1m'].forEach(tf => {{
                document.getElementById(`tab-${{tf}}`).classList.toggle('visible', frame.visibleTabs.includes(tf));
            }});

            if (!frame.visibleTabs.includes(currentTab)) {{
                currentTab = frame.activeTab;
            }}

            switchTab(currentTab);
        }}

        function selectSweep(idx) {{
            currentSweepIdx = parseInt(idx);
            currentFrameIdx = 0;
            showFrame();
        }}

        function nextFrame() {{
            const sweep = allSweeps[currentSweepIdx];
            if (currentFrameIdx < sweep.frames.length - 1) {{
                currentFrameIdx++;
                showFrame();
            }}
        }}

        function prevFrame() {{
            if (currentFrameIdx > 0) {{
                currentFrameIdx--;
                showFrame();
            }}
        }}

        document.addEventListener('keydown', e => {{
            if (e.key === 'ArrowRight') nextFrame();
            if (e.key === 'ArrowLeft') prevFrame();
        }});

        window.onload = () => {{
            initChart('1h');
            showFrame();
        }};
    </script>
</body>
</html>'''

    def _generate_options(self, all_sweep_data: List[Dict]) -> str:
        """Generate HTML options for sweep dropdown."""
        options = []
        for sweep in all_sweep_data:
            css_class = 'success' if sweep['isSuccess'] else 'failed'
            options.append(f'<option value="{sweep["sweepId"]}" class="{css_class}">{sweep["label"]}</option>')
        return '\n'.join(options)
