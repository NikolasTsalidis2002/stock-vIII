"""
TradingView-Based Multi-Timeframe Strategy Visualizer

Generates single HTML file with all 3 timeframes (1H, 5M, 1M) using:
- TradingView lightweight-charts library for professional appearance
- D3.js for FVG rectangles with y-axis scaling support
- Single HTML file with dynamic navigation (no iframe loading)
- Follows FRAME_VISUALIZATION_LOGIC.md windowing rules

Usage:
    analyzer = TradingViewStrategyAnalyzer(strategy)
    analyzer.analyze_partial_setup(partial, setup_num=1)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Optional, List, Dict, Any
from datetime import timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy import PartialSetup, TradeSignal, MultiTimeframeStrategy
from src.indicators.smc_custom import smc_custom
from src.indicators.smc import smc


class TradingViewStrategyAnalyzer:
    """
    Generates professional TradingView-style visualizations for strategy setups.

    Creates a single HTML file with embedded data for all frames, showing
    3 timeframes with proper indicator windowing per FRAME_VISUALIZATION_LOGIC.md
    """

    def __init__(self, strategy: MultiTimeframeStrategy, output_dir: Optional[str] = None):
        """
        Initialize analyzer.

        Args:
            strategy: MultiTimeframeStrategy instance with data and calculated indicators
            output_dir: Directory to save visualizations
        """
        self.strategy = strategy

        if output_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            self.output_dir = project_root / 'results' / 'strategy_examples'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_frame(
        self,
        partial: PartialSetup,
        current_time: pd.Timestamp,
        frame_idx: int,
        status_message: str,
        conditions_met: List[bool]
    ) -> Dict[str, Any]:
        """
        Generate frame data for a specific timestamp.

        Args:
            partial: PartialSetup object
            current_time: Timestamp for this frame
            frame_idx: Frame number
            status_message: Status text to display
            conditions_met: [bool, bool, bool, bool] for 4 conditions

        Returns:
            Dictionary containing all data for this frame
        """
        frame_data = {
            'frameIdx': frame_idx,
            'currentTime': current_time.strftime('%Y-%m-%d %H:%M'),
            'currentTimeUnix': int(current_time.timestamp()),
            'statusMessage': status_message,
            'conditions': conditions_met,
            'timeframes': {}
        }

        # ==================== 1H TIMEFRAME ====================
        # Per FRAME_VISUALIZATION_LOGIC.md: Show ALL data (no windowing)
        df_1h = self.strategy.df_1h[self.strategy.df_1h.index <= current_time]
        df_1h_plot = df_1h  # No windowing for 1H

        frame_data['timeframes']['1h'] = self._process_timeframe(
            df=df_1h_plot,
            full_df=df_1h,  # For indicator calculation
            timeframe_label='1H',
            partial=partial,
            current_time=current_time,
            highlight_sweep=(current_time >= partial.timestamp_1h_sweep)
        )

        # ==================== 5M TIMEFRAME ====================
        # Per FRAME_VISUALIZATION_LOGIC.md: Window from 1H sweep time
        # IMPORTANT: Hide 5M if sweep is broken (setup reset)
        df_5m = self.strategy.df_5m[self.strategy.df_5m.index <= current_time]

        # Check if sweep is still valid at current time
        is_sweep_broken = self.strategy.is_sweep_broken_at_time(partial.inflexion_idx, current_time)

        if is_sweep_broken:
            # Sweep broken - hide 5M timeframe (setup reset)
            df_5m_plot = pd.DataFrame()
        elif len(df_5m) > 0 and current_time >= partial.timestamp_1h_sweep:
            start_ts_5m = partial.timestamp_1h_sweep
            df_5m_plot = df_5m[df_5m.index >= start_ts_5m]
        else:
            df_5m_plot = pd.DataFrame()  # Empty before sweep

        frame_data['timeframes']['5m'] = self._process_timeframe(
            df=df_5m_plot,
            full_df=df_5m_plot,  # Calculate on windowed data
            timeframe_label='5M',
            partial=partial,
            current_time=current_time,
            highlight_event_b=(partial.timestamp_5m_event_b and current_time >= partial.timestamp_5m_event_b),
            highlight_validation=(partial.timestamp_5m_validation and current_time >= partial.timestamp_5m_validation)
        )

        # ==================== 1M TIMEFRAME ====================
        # Per FRAME_VISUALIZATION_LOGIC.md: Window from 5M validation time
        df_1m = self.strategy.df_1m[self.strategy.df_1m.index <= current_time]

        if len(df_1m) > 0 and partial.timestamp_5m_validation and current_time >= partial.timestamp_5m_validation:
            start_ts_1m = partial.timestamp_5m_validation
            df_1m_plot = df_1m[df_1m.index >= start_ts_1m]
        else:
            df_1m_plot = pd.DataFrame()  # Empty before validation

        frame_data['timeframes']['1m'] = self._process_timeframe(
            df=df_1m_plot,
            full_df=df_1m_plot,  # Calculate on windowed data
            timeframe_label='1M',
            partial=partial,
            current_time=current_time,
            highlight_confirmation=(partial.timestamp_1m_confirmation and current_time >= partial.timestamp_1m_confirmation)
        )

        return frame_data

    def _process_timeframe(
        self,
        df: pd.DataFrame,
        full_df: pd.DataFrame,
        timeframe_label: str,
        partial: PartialSetup,
        current_time: pd.Timestamp,
        highlight_sweep: bool = False,
        highlight_event_b: bool = False,
        highlight_validation: bool = False,
        highlight_confirmation: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single timeframe and extract all visualization data.

        Args:
            df: DataFrame slice to display
            full_df: DataFrame for indicator calculation
            timeframe_label: '1H', '5M', or '1M'
            partial: PartialSetup object
            current_time: Current frame timestamp
            highlight_*: Flags for highlighting specific events

        Returns:
            Dictionary with candleData, markers, lines, fvgZones, stats
        """
        result = {
            'candleData': [],
            'inflexionMarkers': [],
            'inflexionLines': [],
            'bosLines': [],
            'fvgZones': [],
            'stats': {
                'totalCandles': len(df),
                'totalInflexions': 0,
                'totalBos': 0,
                'totalFvg': 0
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

        # Calculate indicators on full_df
        if len(full_df) > 0:
            inflexions = smc_custom.inflexion_points(full_df)
            bos = smc_custom.bos(full_df, inflexions, close_break=True)
            fvg = smc.fvg(full_df, join_consecutive=True)

            # Process inflexion markers and lines
            for i in range(len(inflexions)):
                if not np.isnan(inflexions["InflexionType"].iloc[i]):
                    # Check if this index is in the display window
                    if full_df.index[i] not in df.index:
                        continue

                    inflx_type = inflexions["InflexionType"].iloc[i]
                    level = inflexions["Level"].iloc[i]
                    respected = inflexions["Respected"].iloc[i]

                    # Determine if this is a highlighted event
                    is_sweep = False
                    if highlight_sweep and timeframe_label == '1H':
                        try:
                            sweep_idx = full_df.index.get_loc(partial.timestamp_1h_sweep)
                            is_sweep = (i == sweep_idx)
                        except:
                            pass

                    # Marker properties
                    if inflx_type == 1:  # Concave (high)
                        if respected is True:
                            color = '#00ff00'  # Green - liquidity sweep
                            text = '★' if is_sweep else '▼'
                            position = 'aboveBar'
                        elif respected is False:
                            color = '#ff0000'  # Red - broken
                            text = '✕'
                            position = 'aboveBar'
                        else:
                            color = '#ffff00'  # Yellow - pending
                            text = '▼'
                            position = 'aboveBar'
                    else:  # Convex (low)
                        if respected is True:
                            color = '#0080ff'  # Blue - liquidity sweep
                            text = '★' if is_sweep else '▲'
                            position = 'belowBar'
                        elif respected is False:
                            color = '#ff8800'  # Orange - broken
                            text = '✕'
                            position = 'belowBar'
                        else:
                            color = '#ffff00'  # Yellow - pending
                            text = '▲'
                            position = 'belowBar'

                    result['inflexionMarkers'].append({
                        'time': int(full_df.index[i].timestamp()),
                        'position': position,
                        'color': color,
                        'shape': 'circle',
                        'text': text,
                        'size': 15 if is_sweep else 10
                    })

                    # Inflexion level line
                    status_idx = int(inflexions["StatusIndex"].iloc[i]) if inflexions["StatusIndex"].iloc[i] != 0 else len(full_df) - 1

                    if respected is True:
                        line_color = '#00ff00' if inflx_type == 1 else '#0080ff'
                    elif respected is False:
                        line_color = '#ff0000' if inflx_type == 1 else '#ff8800'
                    else:
                        line_color = '#ffff00'

                    result['inflexionLines'].append({
                        'startTime': int(full_df.index[i].timestamp()),
                        'endTime': int(full_df.index[status_idx].timestamp()),
                        'price': float(level),
                        'color': line_color,
                        'lineWidth': 2 if is_sweep else 1,
                        'lineStyle': 'Solid' if is_sweep else 'Dotted'
                    })

                    result['stats']['totalInflexions'] += 1

            # Process BOS lines
            for i in range(len(bos)):
                if not np.isnan(bos["BOS"].iloc[i]):
                    # Check if in display window
                    if full_df.index[i] not in df.index:
                        continue

                    bos_type = bos["BOS"].iloc[i]
                    level = bos["Level"].iloc[i]

                    # Find matching inflexion
                    matching_inflexions = []
                    for j in range(len(inflexions)):
                        if (not np.isnan(inflexions["Level"].iloc[j]) and
                            inflexions["Level"].iloc[j] == level and
                            j < i):
                            matching_inflexions.append(j)

                    if len(matching_inflexions) > 0:
                        inflexion_pos = matching_inflexions[-1]

                        result['bosLines'].append({
                            'startTime': int(full_df.index[inflexion_pos].timestamp()),
                            'endTime': int(full_df.index[i].timestamp()),
                            'price': float(level),
                            'color': '#00ff00' if bos_type == 1 else '#ff0000',
                            'label': 'BOS ↑' if bos_type == 1 else 'BOS ↓',
                            'lineWidth': 2
                        })

                        result['stats']['totalBos'] += 1

            # Process FVG zones
            for i in range(len(fvg["FVG"])):
                if not np.isnan(fvg["FVG"][i]):
                    # Check if in display window
                    if full_df.index[i] not in df.index:
                        continue

                    x1 = int(fvg["MitigatedIndex"][i] if fvg["MitigatedIndex"][i] != 0 else len(full_df) - 1)

                    result['fvgZones'].append({
                        'startTime': int(full_df.index[i].timestamp()),
                        'endTime': int(full_df.index[x1].timestamp()),
                        'topPrice': float(fvg["Top"][i]),
                        'bottomPrice': float(fvg["Bottom"][i])
                    })

                    result['stats']['totalFvg'] += 1

        return result

    def _create_single_html_template(self, all_frames: List[Dict[str, Any]], partial: PartialSetup, setup_num: int) -> str:
        """
        Create single HTML file with all frames embedded and 3 TradingView charts.

        Args:
            all_frames: List of frame data dictionaries
            partial: PartialSetup object for metadata
            setup_num: Setup number

        Returns:
            HTML string
        """
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Partial Setup #{setup_num} - TradingView Walkthrough</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #131722;
            color: #d1d4dc;
            overflow: hidden;
        }}

        #container {{
            display: grid;
            grid-template-rows: 60px 1fr;
            height: 100vh;
        }}

        #header {{
            background-color: #1e222d;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2b2b43;
        }}

        #header h1 {{
            margin: 0;
            font-size: 18px;
            color: #2962ff;
        }}

        #controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        button {{
            background-color: #2962ff;
            color: white;
            border: none;
            padding: 8px 16px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            border-radius: 4px;
        }}

        button:hover {{
            background-color: #1e53e5;
        }}

        button:disabled {{
            background-color: #2b2b43;
            cursor: not-allowed;
        }}

        #frame-info {{
            font-size: 14px;
            color: #d1d4dc;
            font-weight: 600;
            min-width: 150px;
            text-align: center;
        }}

        #goto-input {{
            width: 80px;
            padding: 8px;
            background-color: #1e222d;
            border: 1px solid #363a45;
            color: #d1d4dc;
            border-radius: 4px;
            font-size: 12px;
            text-align: center;
        }}

        #goto-input:focus {{
            outline: none;
            border-color: #2962ff;
        }}

        #main-content {{
            display: grid;
            grid-template-columns: 1fr 300px;
            height: 100%;
            overflow: hidden;
        }}

        #charts-container {{
            display: grid;
            grid-template-rows: 1fr 1fr 1fr;
            gap: 10px;
            padding: 10px;
            overflow-y: auto;
        }}

        .chart-wrapper {{
            position: relative;
            background-color: #1e222d;
            border-radius: 4px;
            min-height: 250px;
        }}

        .chart-header {{
            padding: 8px 12px;
            background-color: #131722;
            border-bottom: 1px solid #2b2b43;
            font-weight: 600;
            font-size: 14px;
        }}

        .chart-content {{
            position: relative;
            height: calc(100% - 40px);
        }}

        #sidebar {{
            background-color: #1e222d;
            padding: 20px;
            overflow-y: auto;
            border-left: 1px solid #2b2b43;
        }}

        .stat-group {{
            margin: 20px 0;
            padding: 15px;
            background-color: #131722;
            border-radius: 4px;
        }}

        .stat-group h2 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #787b86;
            text-transform: uppercase;
        }}

        .stat-item {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 13px;
        }}

        .stat-label {{
            color: #787b86;
        }}

        .stat-value {{
            color: #d1d4dc;
            font-weight: 600;
        }}

        .condition-item {{
            margin: 10px 0;
            padding: 10px;
            background-color: #0c0e12;
            border-radius: 4px;
            border-left: 4px solid #2b2b43;
        }}

        .condition-item.completed {{
            border-left-color: #089981;
        }}

        .condition-header {{
            font-weight: 600;
            margin-bottom: 5px;
        }}

        .condition-details {{
            font-size: 11px;
            color: #787b86;
        }}

        .status-message {{
            margin: 15px 0;
            padding: 12px;
            background-color: #1a1d24;
            border-radius: 4px;
            font-size: 12px;
            white-space: pre-line;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="header">
            <h1>Partial Setup #{setup_num} - Multi-Timeframe Analysis</h1>
            <div id="controls">
                <button id="first-btn" onclick="firstFrame()">⏮ First</button>
                <button id="prev-btn" onclick="prevFrame()">◀ Prev</button>
                <span id="frame-info">Frame 1 / {len(all_frames)}</span>
                <input type="number" id="goto-input" min="1" max="{len(all_frames)}" placeholder="Go to...">
                <button id="goto-btn" onclick="gotoFrame()">Go</button>
                <button id="next-btn" onclick="nextFrame()">Next ▶</button>
                <button id="last-btn" onclick="lastFrame()">Last ⏭</button>
            </div>
        </div>

        <div id="main-content">
            <div id="charts-container">
                <div class="chart-wrapper">
                    <div class="chart-header">1H Timeframe - Liquidity Sweep Detection</div>
                    <div class="chart-content" id="chart-1h"></div>
                </div>
                <div class="chart-wrapper">
                    <div class="chart-header">5M Timeframe - Event B & Validation</div>
                    <div class="chart-content" id="chart-5m"></div>
                </div>
                <div class="chart-wrapper">
                    <div class="chart-header">1M Timeframe - Final Confirmation</div>
                    <div class="chart-content" id="chart-1m"></div>
                </div>
            </div>

            <div id="sidebar">
                <div class="stat-group">
                    <h2>Progress</h2>
                    <div class="stat-item">
                        <span class="stat-label">Time</span>
                        <span class="stat-value" id="current-time">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Conditions Met</span>
                        <span class="stat-value" id="conditions-met">0/4</span>
                    </div>
                </div>

                <div class="stat-group">
                    <h2>Setup Status</h2>
                    <div class="status-message" id="status-message">Loading...</div>
                </div>

                <div class="stat-group" id="conditions-panel">
                    <h2>Strategy Conditions</h2>
                    <div class="condition-item" id="cond-1">
                        <div class="condition-header">☐ 1H Liquidity Sweep</div>
                        <div class="condition-details">-</div>
                    </div>
                    <div class="condition-item" id="cond-2">
                        <div class="condition-header">☐ 5M Event B</div>
                        <div class="condition-details">BOS or IFVG opposite direction</div>
                    </div>
                    <div class="condition-item" id="cond-3">
                        <div class="condition-header">☐ 5M Validation</div>
                        <div class="condition-details">FVG or Demand Zone respect</div>
                    </div>
                    <div class="condition-item" id="cond-4">
                        <div class="condition-header">☐ 1M Confirmation</div>
                        <div class="condition-details">Final BOS or IFVG</div>
                    </div>
                </div>

                <div class="stat-group">
                    <h2>Partial Details</h2>
                    <div class="stat-item">
                        <span class="stat-label">1H Trend</span>
                        <span class="stat-value">{partial.trend_1h_before_sweep.upper()}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Entry Direction</span>
                        <span class="stat-value">{partial.entry_direction.upper()}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Failure Reason</span>
                        <span class="stat-value">{partial.failure_reason}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embed all frame data
        const allFrames = {json.dumps(all_frames)};
        let currentFrameIdx = 0;

        // TradingView chart instances
        let charts = {{}};
        let candlestickSeries = {{}};
        let svgOverlays = {{}};
        let activeLineSeries = {{ '1h': [], '5m': [], '1m': [] }};

        // Initialize charts
        function initializeCharts() {{
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

            // Create charts for each timeframe
            ['1h', '5m', '1m'].forEach(tf => {{
                const container = document.getElementById(`chart-${{tf}}`);
                charts[tf] = LightweightCharts.createChart(container, chartOptions);

                candlestickSeries[tf] = charts[tf].addCandlestickSeries({{
                    upColor: '#089981',
                    downColor: '#f23645',
                    borderVisible: false,
                    wickUpColor: '#089981',
                    wickDownColor: '#f23645',
                }});

                // Create SVG overlay for FVG rectangles
                svgOverlays[tf] = d3.select(container)
                    .append('svg')
                    .style('position', 'absolute')
                    .style('top', '0')
                    .style('left', '0')
                    .style('z-index', '10')
                    .style('pointer-events', 'none');

                // Handle resize
                new ResizeObserver(() => {{
                    charts[tf].applyOptions({{
                        width: container.clientWidth,
                        height: container.clientHeight
                    }});
                    drawFVGRectangles(tf);
                }}).observe(container);

                // Subscribe to scale changes
                charts[tf].timeScale().subscribeVisibleLogicalRangeChange(() => drawFVGRectangles(tf));

                // Mouse events for y-axis scaling
                let isMouseDown = false;
                let redrawTimer = null;

                function throttledRedraw() {{
                    if (redrawTimer) return;
                    redrawTimer = requestAnimationFrame(() => {{
                        drawFVGRectangles(tf);
                        redrawTimer = null;
                    }});
                }}

                container.addEventListener('wheel', throttledRedraw);
                container.addEventListener('mousedown', () => {{ isMouseDown = true; }});
                container.addEventListener('mousemove', () => {{ if (isMouseDown) throttledRedraw(); }});
                container.addEventListener('mouseup', () => {{ isMouseDown = false; throttledRedraw(); }});
                container.addEventListener('mouseleave', () => {{ isMouseDown = false; }});
            }});
        }}

        function drawFVGRectangles(tf) {{
            const svg = svgOverlays[tf];
            const container = document.getElementById(`chart-${{tf}}`);
            const rect = container.getBoundingClientRect();

            svg.attr('width', rect.width).attr('height', rect.height);
            svg.selectAll('rect').remove();

            const frame = allFrames[currentFrameIdx];
            const tfData = frame.timeframes[tf];
            const timeScale = charts[tf].timeScale();

            tfData.fvgZones.forEach(zone => {{
                const x1 = timeScale.timeToCoordinate(zone.startTime);
                const x2 = timeScale.timeToCoordinate(zone.endTime);
                const y1 = candlestickSeries[tf].priceToCoordinate(zone.topPrice);
                const y2 = candlestickSeries[tf].priceToCoordinate(zone.bottomPrice);

                if (x1 !== null && x2 !== null && y1 !== null && y2 !== null) {{
                    const x = Math.min(x1, x2);
                    const y = Math.min(y1, y2);
                    const width = Math.abs(x2 - x1);
                    const height = Math.abs(y2 - y1);

                    svg.append('rect')
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

        function clearLineSeries(tf) {{
            activeLineSeries[tf].forEach(series => {{
                charts[tf].removeSeries(series);
            }});
            activeLineSeries[tf] = [];
        }}

        function updateChart(tf) {{
            const frame = allFrames[currentFrameIdx];
            const tfData = frame.timeframes[tf];

            // Update candlestick data
            if (tfData.candleData.length > 0) {{
                candlestickSeries[tf].setData(tfData.candleData);
                candlestickSeries[tf].setMarkers(tfData.inflexionMarkers);
            }} else {{
                candlestickSeries[tf].setData([]);
                candlestickSeries[tf].setMarkers([]);
            }}

            // Clear and redraw lines
            clearLineSeries(tf);

            // Inflexion lines
            tfData.inflexionLines.forEach(line => {{
                const lineSeries = charts[tf].addLineSeries({{
                    color: line.color,
                    lineWidth: line.lineWidth,
                    lineStyle: line.lineStyle === 'Solid' ?
                        LightweightCharts.LineStyle.Solid :
                        LightweightCharts.LineStyle.Dotted,
                    priceLineVisible: false,
                    lastValueVisible: false,
                }});
                lineSeries.setData([
                    {{ time: line.startTime, value: line.price }},
                    {{ time: line.endTime, value: line.price }}
                ]);
                activeLineSeries[tf].push(lineSeries);
            }});

            // BOS lines
            tfData.bosLines.forEach(bos => {{
                const lineSeries = charts[tf].addLineSeries({{
                    color: bos.color,
                    lineWidth: bos.lineWidth,
                    lineStyle: LightweightCharts.LineStyle.Solid,
                    priceLineVisible: false,
                    lastValueVisible: false,
                }});
                lineSeries.setData([
                    {{ time: bos.startTime, value: bos.price }},
                    {{ time: bos.endTime, value: bos.price }}
                ]);
                activeLineSeries[tf].push(lineSeries);
            }});

            // Redraw FVG rectangles
            drawFVGRectangles(tf);

            // Fit content
            charts[tf].timeScale().fitContent();
        }}

        function showFrame(n) {{
            currentFrameIdx = Math.max(0, Math.min(n, allFrames.length - 1));
            const frame = allFrames[currentFrameIdx];

            // Update all 3 timeframe charts
            ['1h', '5m', '1m'].forEach(tf => {{
                updateChart(tf);
            }});

            // Update sidebar
            document.getElementById('current-time').textContent = frame.currentTime;
            document.getElementById('status-message').textContent = frame.statusMessage;

            const conditionsMet = frame.conditions.filter(c => c).length;
            document.getElementById('conditions-met').textContent = `${{conditionsMet}}/4`;

            // Update condition checkboxes
            for (let i = 0; i < 4; i++) {{
                const condEl = document.getElementById(`cond-${{i+1}}`);
                if (frame.conditions[i]) {{
                    condEl.classList.add('completed');
                    condEl.querySelector('.condition-header').textContent =
                        condEl.querySelector('.condition-header').textContent.replace('☐', '✓');
                }} else {{
                    condEl.classList.remove('completed');
                    condEl.querySelector('.condition-header').textContent =
                        condEl.querySelector('.condition-header').textContent.replace('✓', '☐');
                }}
            }}

            // Update navigation UI
            document.getElementById('frame-info').textContent =
                `Frame ${{currentFrameIdx + 1}} / ${{allFrames.length}}`;
            document.getElementById('prev-btn').disabled = (currentFrameIdx === 0);
            document.getElementById('first-btn').disabled = (currentFrameIdx === 0);
            document.getElementById('next-btn').disabled = (currentFrameIdx === allFrames.length - 1);
            document.getElementById('last-btn').disabled = (currentFrameIdx === allFrames.length - 1);
        }}

        function nextFrame() {{ showFrame(currentFrameIdx + 1); }}
        function prevFrame() {{ showFrame(currentFrameIdx - 1); }}
        function firstFrame() {{ showFrame(0); }}
        function lastFrame() {{ showFrame(allFrames.length - 1); }}
        function gotoFrame() {{
            const input = document.getElementById('goto-input');
            const frameNum = parseInt(input.value);
            if (frameNum >= 1 && frameNum <= allFrames.length) {{
                showFrame(frameNum - 1);
            }}
            input.value = '';
        }}

        // Keyboard navigation
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'ArrowRight') nextFrame();
            if (event.key === 'ArrowLeft') prevFrame();
            if (event.key === 'Home') firstFrame();
            if (event.key === 'End') lastFrame();
        }});

        // Enter key for goto input
        document.getElementById('goto-input').addEventListener('keydown', function(event) {{
            if (event.key === 'Enter') {{
                event.stopPropagation();
                gotoFrame();
            }}
        }});

        // Initialize
        window.onload = function() {{
            initializeCharts();
            showFrame(0);
        }};
    </script>
</body>
</html>
"""

    def analyze_partial_setup(self, partial: PartialSetup, setup_num: int):
        """
        Generate complete walkthrough for a partial setup.

        Args:
            partial: PartialSetup object
            setup_num: Setup number
        """
        print(f"\n{'='*80}")
        print(f"Generating TradingView Walkthrough for Partial Setup #{setup_num}")
        print(f"{'='*80}\n")
        print(f"  Conditions Met: {partial.conditions_met}/4")
        print(f"  Failed At: {partial.failure_reason}")

        # Get timestamps to visualize
        timestamps = self._get_key_timestamps(partial)

        print(f"\n  Generating {len(timestamps)} frames...")

        # Collect all frame data
        all_frames = []
        for i, (ts, status, conditions) in enumerate(timestamps):
            if (i + 1) % 10 == 0:
                print(f"    Frame {i+1}/{len(timestamps)}...")

            frame_data = self.generate_frame(partial, ts, i, status, conditions)
            all_frames.append(frame_data)

        print(f"\n  ✓ Generated {len(timestamps)} frames")

        # Generate single HTML file
        print(f"\n  📝 Creating single HTML file...")
        html_content = self._create_single_html_template(all_frames, partial, setup_num)

        # Save HTML
        output_dir = self.output_dir / f"partial_setup_{setup_num}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "walkthrough_tv.html"

        with open(output_path, 'w') as f:
            f.write(html_content)

        file_size_kb = len(html_content) / 1024

        print(f"  ✓ Saved: {output_path}")
        print(f"  ✓ File size: {file_size_kb:.1f} KB")
        print(f"\n✅ TradingView walkthrough complete for Partial Setup #{setup_num}")

    def _get_key_timestamps(self, partial: PartialSetup) -> List[tuple]:
        """
        Get list of (timestamp, status_message, conditions_met) tuples.

        NOW SHOWS EVERY CANDLE (not just key events) for debugging.

        Args:
            partial: PartialSetup object

        Returns:
            List of (timestamp, status, [bool]*4) tuples
        """
        frames = []

        # ==================== PHASE 1: 1H CANDLES (Before Sweep) ====================
        # Show every 1H candle leading up to the sweep
        start_1h = partial.timestamp_1h_sweep - timedelta(hours=10)  # 10 candles before sweep

        candles_1h = self.strategy.df_1h[
            (self.strategy.df_1h.index >= start_1h) &
            (self.strategy.df_1h.index < partial.timestamp_1h_sweep)
        ]

        for i, ts in enumerate(candles_1h.index):
            frames.append((
                ts,
                f"Scanning 1H for liquidity sweep...\nCandle {i+1}/{len(candles_1h)+1}",
                [False, False, False, False]
            ))

        # ==================== PHASE 2: 1H SWEEP EVENT ====================
        frames.append((
            partial.timestamp_1h_sweep,
            f"✓ LIQUIDITY SWEEP DETECTED!\n\n1H Trend: {partial.trend_1h_before_sweep.upper()}\nPrice: ${partial.price_1h_sweep:.2f}\nEntry Direction: {partial.entry_direction.upper()}\n\n→ Moving to 5M timeframe...",
            [True, False, False, False]
        ))

        # ==================== PHASE 3: 5M CANDLES (After Sweep) ====================
        # Show every 5M candle from sweep time
        if partial.timestamp_5m_event_b:
            # From sweep to Event B
            candles_5m_to_eventb = self.strategy.df_5m[
                (self.strategy.df_5m.index > partial.timestamp_1h_sweep) &
                (self.strategy.df_5m.index <= partial.timestamp_5m_event_b)
            ]

            for i, ts in enumerate(candles_5m_to_eventb.index):
                # Check if this IS the event B candle
                is_event_b = (ts == partial.timestamp_5m_event_b)

                if is_event_b:
                    status = f"✓ EVENT B DETECTED!\n\nType: {partial.condition_event_b}\nPrice: ${partial.price_5m_event_b:.2f}\nOpposite direction confirmed\n\n→ Looking for validation..."
                    conditions = [True, True, False, False]
                else:
                    status = f"Scanning 5M for Event B...\n(BOS or IFVG in opposite direction)\n\n5M Candle {i+1}/{len(candles_5m_to_eventb)}"
                    conditions = [True, False, False, False]

                frames.append((ts, status, conditions))

            # From Event B to Validation (if exists)
            if partial.timestamp_5m_validation:
                candles_5m_to_validation = self.strategy.df_5m[
                    (self.strategy.df_5m.index > partial.timestamp_5m_event_b) &
                    (self.strategy.df_5m.index <= partial.timestamp_5m_validation)
                ]

                for i, ts in enumerate(candles_5m_to_validation.index):
                    is_validation = (ts == partial.timestamp_5m_validation)

                    if is_validation:
                        status = f"✓ VALIDATION CONFIRMED!\n\nType: {partial.condition_validation}\nPrice: ${partial.price_5m_validation:.2f}\nZone respected\n\n→ Moving to 1M timeframe..."
                        conditions = [True, True, True, False]
                    else:
                        status = f"Scanning 5M for validation...\n(FVG or Demand Zone respect)\n\n5M Candle {i+1}/{len(candles_5m_to_validation)}"
                        conditions = [True, True, False, False]

                    frames.append((ts, status, conditions))

        else:
            # No Event B - show 5M candles for some time after sweep
            end_5m = partial.timestamp_1h_sweep + timedelta(hours=2)
            candles_5m = self.strategy.df_5m[
                (self.strategy.df_5m.index > partial.timestamp_1h_sweep) &
                (self.strategy.df_5m.index <= end_5m)
            ]

            for i, ts in enumerate(candles_5m.index):
                frames.append((
                    ts,
                    f"Scanning 5M for Event B...\n(BOS or IFVG in opposite direction)\n\n5M Candle {i+1}/{len(candles_5m)}",
                    [True, False, False, False]
                ))

        # ==================== PHASE 4: 1M CANDLES (After Validation) ====================
        if partial.timestamp_5m_validation:
            # Show every 1M candle from validation time
            end_1m = partial.timestamp_5m_validation + timedelta(minutes=30)

            candles_1m = self.strategy.df_1m[
                (self.strategy.df_1m.index > partial.timestamp_5m_validation) &
                (self.strategy.df_1m.index <= end_1m)
            ]

            for i, ts in enumerate(candles_1m.index):
                frames.append((
                    ts,
                    f"Scanning 1M for final confirmation...\n(BOS or IFVG in entry direction)\n\n1M Candle {i+1}/{len(candles_1m)}",
                    [True, True, True, False]
                ))

        # ==================== PHASE 5: REMAINING 1H CANDLES ====================
        # Show ALL remaining 1H candles until the end of the dataset
        # NOW WITH DYNAMIC SWEEP INVALIDATION CHECK

        # Determine the last event timestamp
        if partial.timestamp_5m_validation:
            last_event = partial.timestamp_5m_validation
        elif partial.timestamp_5m_event_b:
            last_event = partial.timestamp_5m_event_b
        else:
            last_event = partial.timestamp_1h_sweep

        # Get all remaining 1H candles after the last event
        remaining_1h_candles = self.strategy.df_1h[
            self.strategy.df_1h.index > last_event
        ]

        # Track whether sweep has been broken (initially still valid)
        sweep_broken_ts = None  # Timestamp when sweep was first broken

        # Add a frame for each remaining 1H candle
        total_remaining = len(remaining_1h_candles)
        for i, ts in enumerate(remaining_1h_candles.index):
            # Check if the sweep is still valid at this timestamp
            is_sweep_broken = self.strategy.is_sweep_broken_at_time(partial.inflexion_idx, ts)

            # Track when sweep was first broken
            if is_sweep_broken and sweep_broken_ts is None:
                sweep_broken_ts = ts

            if is_sweep_broken:
                # Sweep is broken - reset ALL conditions
                status = f"⚠️ LIQUIDITY SWEEP INVALIDATED!\n\nThe 1H liquidity sweep was broken at {sweep_broken_ts.strftime('%Y-%m-%d %H:%M')}\nPrice closed beyond the swept level\n\n→ Setup has been RESET\n→ Waiting for new liquidity sweep...\n\n1H Candle {i+1}/{total_remaining}"
                conditions = [False, False, False, False]  # ALL conditions reset
            else:
                # Sweep still valid - show normal incomplete setup message
                status = f"❌ SETUP INCOMPLETE\n\n{partial.failure_reason}\n\nConditions met: {partial.conditions_met}/4\n\n1H Sweep: {'✓' if partial.conditions_met >= 1 else '✗'}\n5M Event B: {'✓' if partial.conditions_met >= 2 else '✗'}\n5M Validation: {'✓' if partial.conditions_met >= 3 else '✗'}\n1M Confirmation: {'✓' if partial.conditions_met >= 4 else '✗'}\n\nShowing remaining 1H candles... ({i+1}/{total_remaining})"
                conditions = [
                    True,
                    partial.conditions_met >= 2,
                    partial.conditions_met >= 3,
                    partial.conditions_met >= 4
                ]

            frames.append((ts, status, conditions))

        return frames
