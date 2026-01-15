"""
1H Candle-by-Candle Walkthrough using TradingView Lightweight Charts

Generates a single HTML file with all frames embedded and dynamic navigation.
Uses TradingView lightweight-charts library and D3.js for professional visualization.

Usage:
    python3 scripts/analyze_1h_tv.py

Output:
    - results/1h_tv_walkthrough/walkthrough.html (single file with all data)

Features:
    - Dynamic frame switching without page reloads
    - Arrow key navigation (← → or Home/End)
    - Professional TradingView-style dark theme
    - All SMC indicators: FVG, BOS, Inflexions, Liquidity Sweeps
    - Y-axis scaling support for FVG rectangles using D3.js
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import TSLADataLoader
from smartmoneyconcepts.smc_custom import smc_custom
from smartmoneyconcepts.smc import smc


class TradingViewOneHourAnalyzer:
    """Generates TradingView-style charts for 1H timeframe walkthrough."""

    def __init__(self, output_dir=None):
        if output_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            self.output_dir = project_root / 'results' / '1h_tv_walkthrough'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir = self.output_dir / 'frames'
        self.frame_dir.mkdir(parents=True, exist_ok=True)

    def generate_frame(self, df_slice, candle_idx, total_candles):
        """Generate frame data for candle_idx (returns dictionary, doesn't write file)."""

        # Calculate indicators on the slice
        inflexions = smc_custom.inflexion_points(df_slice)
        bos = smc_custom.bos(df_slice, inflexions, close_break=True)
        fvg = smc.fvg(df_slice, join_consecutive=True)

        # Count statistics
        total_inflexions = inflexions['InflexionType'].notna().sum()
        liquidity_sweeps = ((inflexions['Respected'] == True) & inflexions['InflexionType'].notna()).sum()
        total_bos = bos['BOS'].notna().sum()
        total_fvg = fvg['FVG'].notna().sum()

        # Convert OHLC data to format for TradingView
        candlestick_data = []
        for idx, row in df_slice.iterrows():
            candlestick_data.append({
                'time': int(idx.timestamp()),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })

        # Prepare FVG data
        fvg_zones = []
        for i in range(len(fvg["FVG"])):
            if not np.isnan(fvg["FVG"][i]):
                x1 = int(fvg["MitigatedIndex"][i] if fvg["MitigatedIndex"][i] != 0 else len(df_slice) - 1)
                fvg_zones.append({
                    'startTime': int(df_slice.index[i].timestamp()),
                    'endTime': int(df_slice.index[x1].timestamp()),
                    'topPrice': float(fvg["Top"][i]),
                    'bottomPrice': float(fvg["Bottom"][i])
                })

        # Prepare inflexion points data
        inflexion_markers = []
        for i in range(len(inflexions)):
            if not np.isnan(inflexions["InflexionType"].iloc[i]):
                inflx_type = inflexions["InflexionType"].iloc[i]
                level = inflexions["Level"].iloc[i]
                respected = inflexions["Respected"].iloc[i]

                # Determine marker properties
                if inflx_type == 1:  # Concave
                    if respected is True:
                        color = '#00ff00'  # Green - LIQUIDITY SWEEP
                        text = '★'
                        position = 'aboveBar'
                    elif respected is False:
                        color = '#ff0000'
                        text = '✕'
                        position = 'aboveBar'
                    else:
                        color = '#ffff00'
                        text = '▼'
                        position = 'aboveBar'
                else:  # Convex
                    if respected is True:
                        color = '#0080ff'  # Blue - LIQUIDITY SWEEP
                        text = '★'
                        position = 'belowBar'
                    elif respected is False:
                        color = '#ff8800'
                        text = '✕'
                        position = 'belowBar'
                    else:
                        color = '#ffff00'
                        text = '▲'
                        position = 'belowBar'

                inflexion_markers.append({
                    'time': int(df_slice.index[i].timestamp()),
                    'position': position,
                    'color': color,
                    'shape': 'circle',
                    'text': text
                })

        # Prepare inflexion level lines (horizontal lines from inflexion to status determination)
        inflexion_lines = []
        for i in range(len(inflexions)):
            if not np.isnan(inflexions["InflexionType"].iloc[i]):
                inflx_type = inflexions["InflexionType"].iloc[i]
                level = inflexions["Level"].iloc[i]
                respected = inflexions["Respected"].iloc[i]
                status_idx = int(inflexions["StatusIndex"].iloc[i]) if inflexions["StatusIndex"].iloc[i] != 0 else len(df_slice) - 1

                # Color based on respect status
                if respected is True:
                    line_color = '#00ff00' if inflx_type == 1 else '#0080ff'  # Green for concave, blue for convex
                elif respected is False:
                    line_color = '#ff0000' if inflx_type == 1 else '#ff8800'  # Red/orange for broken
                else:
                    line_color = '#ffff00'  # Yellow for pending

                inflexion_lines.append({
                    'startTime': int(df_slice.index[i].timestamp()),
                    'endTime': int(df_slice.index[status_idx].timestamp()),
                    'price': float(level),
                    'color': line_color,
                    'lineWidth': 2 if respected is True else 1,
                    'lineStyle': 'Solid' if respected is True else 'Dotted'
                })

        # Prepare BOS data
        bos_lines = []
        for i in range(len(bos)):
            if not np.isnan(bos["BOS"].iloc[i]):
                bos_type = bos["BOS"].iloc[i]
                level = bos["Level"].iloc[i]

                # Find the inflexion point that matches this BOS level
                # BOS breaks an inflexion level, so find which inflexion has this exact level
                # inflexions uses integer index, so compare with position i
                matching_inflexions = []
                for j in range(len(inflexions)):
                    if (not np.isnan(inflexions["Level"].iloc[j]) and
                        inflexions["Level"].iloc[j] == level and
                        j < i):  # Inflexion must be before BOS candle
                        matching_inflexions.append(j)

                if len(matching_inflexions) > 0:
                    # Get the most recent matching inflexion
                    inflexion_pos = matching_inflexions[-1]

                    bos_lines.append({
                        'startTime': int(df_slice.index[inflexion_pos].timestamp()),  # Inflexion candle
                        'endTime': int(df_slice.index[i].timestamp()),  # BOS break candle
                        'price': float(level),
                        'color': '#00ff00' if bos_type == 1 else '#ff0000',
                        'label': 'BOS ↑' if bos_type == 1 else 'BOS ↓'
                    })
                else:
                    # Fallback: draw a point if we can't find matching inflexion
                    bos_lines.append({
                        'startTime': int(df_slice.index[i].timestamp()),
                        'endTime': int(df_slice.index[i].timestamp()),
                        'price': float(level),
                        'color': '#00ff00' if bos_type == 1 else '#ff0000',
                        'label': 'BOS ↑' if bos_type == 1 else 'BOS ↓'
                    })

        # Return frame data as dictionary
        return {
            'candleIdx': candle_idx,
            'totalCandles': total_candles,
            'candleData': candlestick_data,
            'fvgZones': fvg_zones,
            'inflexionMarkers': inflexion_markers,
            'inflexionLines': inflexion_lines,
            'bosLines': bos_lines,
            'stats': {
                'totalInflexions': int(total_inflexions),
                'liquiditySweeps': int(liquidity_sweeps),
                'totalBos': int(total_bos),
                'totalFvg': int(total_fvg)
            },
            'currentTime': df_slice.index[-1].strftime('%Y-%m-%d %H:%M')
        }

    def _create_single_html_template(self, all_frames):
        """Create single HTML file with all frames embedded and dynamic navigation."""

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>1H TradingView Walkthrough</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #131722;
            color: #d1d4dc;
        }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #chart {{
            flex: 1;
            position: relative;
        }}
        #chart-container {{
            position: relative;
            width: 100%;
            height: 100%;
        }}
        #svg-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }}
        #sidebar {{
            width: 300px;
            background-color: #1e222d;
            padding: 20px;
            overflow-y: auto;
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 18px;
            color: #2962ff;
        }}
        .controls {{
            margin: 15px 0;
            padding: 15px;
            background-color: #131722;
            border-radius: 4px;
        }}
        button {{
            background-color: #2962ff;
            color: white;
            border: none;
            padding: 8px 15px;
            margin: 3px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            border-radius: 4px;
            width: calc(50% - 8px);
        }}
        button:hover {{
            background-color: #1e53e5;
        }}
        button:disabled {{
            background-color: #2b2b43;
            cursor: not-allowed;
        }}
        .frame-info {{
            text-align: center;
            margin: 10px 0;
            font-size: 14px;
            color: #d1d4dc;
            font-weight: 600;
        }}
        .goto-row {{
            display: flex;
            gap: 5px;
            margin: 10px 0;
        }}
        #goto-input {{
            flex: 1;
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
        .stat-value.highlight {{
            color: #089981;
        }}
        .legend {{
            margin-top: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 12px;
        }}
        .legend-marker {{
            width: 16px;
            height: 16px;
            margin-right: 8px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
        }}
        .help-text {{
            text-align: center;
            color: #787b86;
            font-size: 11px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="chart">
            <div id="chart-container"></div>
        </div>
        <div id="sidebar">
            <h1>1H TIMEFRAME ANALYSIS</h1>

            <div class="controls">
                <div class="frame-info" id="frame-number">Candle 1 of {len(all_frames)}</div>
                <button id="first-btn" onclick="firstFrame()">⏮ First</button>
                <button id="prev-btn" onclick="prevFrame()">◀ Previous</button>
                <button id="next-btn" onclick="nextFrame()">Next ▶</button>
                <button id="last-btn" onclick="lastFrame()">Last ⏭</button>
                <div class="goto-row">
                    <input type="number" id="goto-input" min="1" max="{len(all_frames)}" placeholder="Go to...">
                    <button id="goto-btn" onclick="gotoFrame()">Go</button>
                </div>
                <p class="help-text">Use arrow keys ← → or buttons<br>Home/End for first/last</p>
            </div>

            <div class="stat-group">
                <h2>Progress</h2>
                <div class="stat-item">
                    <span class="stat-label">Time</span>
                    <span class="stat-value" id="current-time">-</span>
                </div>
            </div>

            <div class="stat-group">
                <h2>Indicators Found</h2>
                <div class="stat-item">
                    <span class="stat-label">Inflexions</span>
                    <span class="stat-value" id="total-inflexions">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Liquidity Sweeps</span>
                    <span class="stat-value highlight" id="liquidity-sweeps">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">BOS Events</span>
                    <span class="stat-value" id="total-bos">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">FVG Zones</span>
                    <span class="stat-value" id="total-fvg">0</span>
                </div>
            </div>

            <div class="legend">
                <h2 style="margin: 0 0 10px 0; font-size: 14px; color: #787b86;">LEGEND</h2>
                <div class="legend-item">
                    <div class="legend-marker" style="background-color: #00ff00; color: #000;">★</div>
                    <span>Liquidity Sweep (High)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-marker" style="background-color: #0080ff; color: #fff;">★</div>
                    <span>Liquidity Sweep (Low)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-marker" style="background-color: #ffff00; color: #000;">▼</div>
                    <span>Inflexion (Pending)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-marker" style="background-color: #ff0000; color: #fff;">✕</div>
                    <span>Inflexion (Broken)</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embed all frame data
        const allFrames = {json.dumps(all_frames)};
        let currentFrameIdx = 0;

        // Chart setup
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

        const chartContainer = document.getElementById('chart-container');
        const chart = LightweightCharts.createChart(chartContainer, chartOptions);
        const candlestickSeries = chart.addCandlestickSeries({{
            upColor: '#089981',
            downColor: '#f23645',
            borderVisible: false,
            wickUpColor: '#089981',
            wickDownColor: '#f23645',
        }});

        // Create SVG overlay for FVG rectangles using D3.js
        const svg = d3.select(chartContainer)
            .append('svg')
            .attr('id', 'svg-overlay')
            .style('position', 'absolute')
            .style('top', '0')
            .style('left', '0')
            .style('z-index', '10')
            .style('pointer-events', 'none');

        // Store line series to be able to remove them
        let activeLineSeries = [];

        function updateSVGSize() {{
            const rect = chartContainer.getBoundingClientRect();
            svg.attr('width', rect.width)
               .attr('height', rect.height);
        }}

        function drawFVGRectangles(fvgZones) {{
            updateSVGSize();
            svg.selectAll('rect').remove();

            const timeScale = chart.timeScale();

            fvgZones.forEach(zone => {{
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

        function clearLineSeries() {{
            activeLineSeries.forEach(series => {{
                chart.removeSeries(series);
            }});
            activeLineSeries = [];
        }}

        function drawInflexionLines(inflexionLines) {{
            inflexionLines.forEach(line => {{
                const lineSeries = chart.addLineSeries({{
                    color: line.color,
                    lineWidth: line.lineWidth,
                    lineStyle: line.lineStyle === 'Solid' ? LightweightCharts.LineStyle.Solid : LightweightCharts.LineStyle.Dotted,
                    priceLineVisible: false,
                    lastValueVisible: false,
                }});
                lineSeries.setData([
                    {{ time: line.startTime, value: line.price }},
                    {{ time: line.endTime, value: line.price }}
                ]);
                activeLineSeries.push(lineSeries);
            }});
        }}

        function drawBosLines(bosLines) {{
            bosLines.forEach(bos => {{
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

        function updateStats(stats, currentTime) {{
            document.getElementById('current-time').textContent = currentTime;
            document.getElementById('total-inflexions').textContent = stats.totalInflexions;
            document.getElementById('liquidity-sweeps').textContent = stats.liquiditySweeps;
            document.getElementById('total-bos').textContent = stats.totalBos;
            document.getElementById('total-fvg').textContent = stats.totalFvg;
        }}

        function showFrame(n) {{
            currentFrameIdx = Math.max(0, Math.min(n, allFrames.length - 1));
            const frame = allFrames[currentFrameIdx];

            // Update candlestick data and markers
            candlestickSeries.setData(frame.candleData);
            candlestickSeries.setMarkers(frame.inflexionMarkers);

            // Clear and redraw lines
            clearLineSeries();
            drawInflexionLines(frame.inflexionLines);
            drawBosLines(frame.bosLines);

            // Redraw FVG rectangles
            drawFVGRectangles(frame.fvgZones);

            // Update stats
            updateStats(frame.stats, frame.currentTime);

            // Update navigation UI
            document.getElementById('frame-number').textContent =
                'Candle ' + (currentFrameIdx + 1) + ' of ' + allFrames.length;
            document.getElementById('prev-btn').disabled = (currentFrameIdx === 0);
            document.getElementById('first-btn').disabled = (currentFrameIdx === 0);
            document.getElementById('next-btn').disabled = (currentFrameIdx === allFrames.length - 1);
            document.getElementById('last-btn').disabled = (currentFrameIdx === allFrames.length - 1);

            // Fit content
            chart.timeScale().fitContent();
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

        // Subscribe to chart events to redraw FVG rectangles on zoom/pan
        chart.timeScale().subscribeVisibleLogicalRangeChange(() => {{
            const frame = allFrames[currentFrameIdx];
            drawFVGRectangles(frame.fvgZones);
        }});

        // Add throttled redraw for mouse interactions (y-axis scaling)
        let redrawTimer = null;
        let isMouseDown = false;

        function throttledRedraw() {{
            if (redrawTimer) return;
            redrawTimer = requestAnimationFrame(() => {{
                const frame = allFrames[currentFrameIdx];
                drawFVGRectangles(frame.fvgZones);
                redrawTimer = null;
            }});
        }}

        chartContainer.addEventListener('wheel', throttledRedraw);
        chartContainer.addEventListener('mousedown', () => {{ isMouseDown = true; }});
        chartContainer.addEventListener('mousemove', () => {{ if (isMouseDown) throttledRedraw(); }});
        chartContainer.addEventListener('mouseup', () => {{ isMouseDown = false; throttledRedraw(); }});
        chartContainer.addEventListener('mouseleave', () => {{ isMouseDown = false; }});

        // Handle window resize
        const chartDiv = document.getElementById('chart');
        const resizeObserver = new ResizeObserver(() => {{
            chart.applyOptions({{ width: chartContainer.clientWidth, height: chartContainer.clientHeight }});
            const frame = allFrames[currentFrameIdx];
            drawFVGRectangles(frame.fvgZones);
        }});
        resizeObserver.observe(chartDiv);

        // Load initial frame
        showFrame(0);
    </script>
</body>
</html>
"""

    def run(self, df_1h):
        """Generate single HTML walkthrough with all candles."""
        total_candles = len(df_1h)

        print(f"\n{'='*80}")
        print(f"1H TRADINGVIEW CANDLE-BY-CANDLE WALKTHROUGH")
        print(f"{'='*80}\n")
        print(f"Total candles: {total_candles}")
        print(f"Collecting data for {total_candles} frames...\n")

        all_frames = []
        total_sweeps = 0

        for i in range(total_candles):
            df_slice = df_1h.iloc[:i+1]
            frame_data = self.generate_frame(df_slice, i, total_candles)
            all_frames.append(frame_data)

            if (i + 1) % 10 == 0:
                print(f"  Processed frame {i+1}/{total_candles}")

            if frame_data['stats']['liquiditySweeps'] > total_sweeps:
                print(f"  🎯 Liquidity sweep detected at candle {i+1}!")
                total_sweeps = frame_data['stats']['liquiditySweeps']

        print(f"\n✅ Collected data for {total_candles} frames")
        print(f"🎯 Total liquidity sweeps found: {total_sweeps}")

        # Generate single HTML file with all frames
        print(f"\n📝 Generating single HTML file with embedded data...")
        html_content = self._create_single_html_template(all_frames)
        output_path = self.output_dir / "walkthrough.html"

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"✅ Generated single HTML file: {output_path}")
        print(f"   File size: {len(html_content) / 1024:.1f} KB")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("1H TIMEFRAME TRADINGVIEW ANALYSIS")
    print("="*80 + "\n")

    # Load all timeframes to determine overlap
    print("📂 Loading TSLA data for all timeframes...")
    loader = TSLADataLoader()
    df_1h_full = loader.get_data('1h', force_refresh=False)
    df_5m_full = loader.get_data('5min', force_refresh=False)
    df_1m_full = loader.get_data('1min', force_refresh=False)

    print(f"  ✓ 1H: {len(df_1h_full)} candles")
    print(f"  ✓ 5M: {len(df_5m_full)} candles")
    print(f"  ✓ 1M: {len(df_1m_full)} candles")

    # Determine overlap period
    start_overlap = pd.to_datetime(max(
        df_1h_full['time'].iloc[0],
        df_5m_full['time'].iloc[0],
        df_1m_full['time'].iloc[0]
    ))
    end_overlap = pd.to_datetime(min(
        df_1h_full['time'].iloc[-1],
        df_5m_full['time'].iloc[-1],
        df_1m_full['time'].iloc[-1]
    ))

    print(f"\n⏱️  Data overlap period:")
    print(f"  Start: {start_overlap}")
    print(f"  End:   {end_overlap}")

    # Filter 1H to overlap period
    df_1h = df_1h_full[
        (pd.to_datetime(df_1h_full['time']) >= start_overlap) &
        (pd.to_datetime(df_1h_full['time']) <= end_overlap)
    ].copy()

    print(f"\n  Filtered 1H: {len(df_1h)} candles")

    # Set time as index
    if 'time' in df_1h.columns:
        df_1h = df_1h.set_index('time')

    # Run analysis
    analyzer = TradingViewOneHourAnalyzer()
    analyzer.run(df_1h)

    print("\n" + "="*80)
    print("✅ WALKTHROUGH COMPLETE!")
    print("="*80)
    print(f"\n📍 Open: {analyzer.output_dir / 'walkthrough.html'}")
    print("💡 Single HTML file with dynamic navigation and embedded data!")
    print("⌨️  Use arrow keys ← → or buttons to navigate between candles")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
