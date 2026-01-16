"""
TradingView Candle-by-Candle Walkthrough Visualizer

Unified visualization module for generating TradingView-style walkthroughs
for any timeframe. Creates single HTML files with embedded frame data.

Features:
    - Dynamic frame switching without page reloads
    - Arrow key navigation
    - Professional TradingView-style dark theme
    - All SMC indicators: FVG, BOS, Inflexions, Liquidity Sweeps
    - Y-axis scaling support for FVG rectangles using D3.js

Usage:
    from src.tradingview_visualizer import TradingViewVisualizer

    visualizer = TradingViewVisualizer(timeframe_name="1H", output_subdir="1h_tv_walkthrough")
    visualizer.run(df)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.core import (
    generate_candle_data,
    generate_fvg_zones,
    generate_ob_zones,
    generate_bos_lines,
    generate_inflexion_markers,
    generate_inflexion_lines,
)

from src.indicators.smc_custom import smc_custom
from src.indicators.smc import smc


class TradingViewVisualizer:
    """
    Generates TradingView-style candle-by-candle walkthroughs.

    Unified visualizer that works for any timeframe (1H, 5M, 15M, etc.)
    by parameterizing the timeframe name and output directory.
    """

    def __init__(self, timeframe_name="1H", output_subdir="1h_tv_walkthrough", output_dir=None):
        """
        Initialize visualizer.

        Args:
            timeframe_name: Display name for the timeframe (e.g., "1H", "5M", "15M")
            output_subdir: Subdirectory name in results/ folder
            output_dir: Custom output directory (overrides output_subdir if provided)
        """
        self.timeframe_name = timeframe_name

        if output_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            self.output_dir = project_root / 'results' / 'walkthroughs' / output_subdir
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
        ob = smc_custom.ob(df_slice, bos, inflexions)  # OBs based on broken inflections

        # Count statistics
        total_inflexions = inflexions['InflexionType'].notna().sum()
        liquidity_sweeps = ((inflexions['Respected'] == True) & inflexions['InflexionType'].notna()).sum()
        total_bos = bos['BOS'].notna().sum()
        total_fvg = fvg['FVG'].notna().sum()
        total_ob = ob['OB'].notna().sum()

        # Generate visualization data using shared functions
        candlestick_data = generate_candle_data(df_slice)
        fvg_zones = generate_fvg_zones(df_slice, fvg)
        ob_zones = generate_ob_zones(df_slice, ob)
        inflexion_markers = generate_inflexion_markers(df_slice, inflexions)
        inflexion_lines = generate_inflexion_lines(df_slice, inflexions)
        bos_lines = generate_bos_lines(df_slice, bos, inflexions)

        # Return frame data as dictionary
        return {
            'candleIdx': candle_idx,
            'totalCandles': total_candles,
            'candleData': candlestick_data,
            'fvgZones': fvg_zones,
            'obZones': ob_zones,
            'inflexionMarkers': inflexion_markers,
            'inflexionLines': inflexion_lines,
            'bosLines': bos_lines,
            'stats': {
                'totalInflexions': int(total_inflexions),
                'liquiditySweeps': int(liquidity_sweeps),
                'totalBos': int(total_bos),
                'totalFvg': int(total_fvg),
                'totalOb': int(total_ob)
            },
            'currentTime': df_slice.index[-1].strftime('%Y-%m-%d %H:%M')
        }

    def _create_single_html_template(self, all_frames):
        """Create single HTML file with all frames embedded and dynamic navigation."""

        # Use timeframe_name for title and heading
        title = f"{self.timeframe_name} TradingView Walkthrough"
        heading = f"{self.timeframe_name} TIMEFRAME ANALYSIS"

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
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
            <h1>{heading}</h1>

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
                <div class="stat-item">
                    <span class="stat-label">Order Blocks</span>
                    <span class="stat-value" id="total-ob">0</span>
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
                <div class="legend-item">
                    <div class="legend-marker" style="background-color: rgba(0, 200, 255, 0.4); border: 1px solid #00c8ff;">▭</div>
                    <span>Bullish OB (at Valley)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-marker" style="background-color: rgba(255, 140, 0, 0.4); border: 1px solid #ff8c00;">▭</div>
                    <span>Bearish OB (at Peak)</span>
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
            svg.selectAll('rect.fvg').remove();

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

        function drawOBRectangles(obZones) {{
            updateSVGSize();
            svg.selectAll('rect.ob').remove();

            const timeScale = chart.timeScale();

            obZones.forEach(zone => {{
                const x1 = timeScale.timeToCoordinate(zone.startTime);
                const x2 = timeScale.timeToCoordinate(zone.endTime);
                const y1 = candlestickSeries.priceToCoordinate(zone.topPrice);
                const y2 = candlestickSeries.priceToCoordinate(zone.bottomPrice);

                if (x1 !== null && x2 !== null && y1 !== null && y2 !== null) {{
                    const x = Math.min(x1, x2);
                    const y = Math.min(y1, y2);
                    const width = Math.abs(x2 - x1);
                    const height = Math.abs(y2 - y1);

                    // Bullish OB (at valleys) = cyan/blue, Bearish OB (at peaks) = orange/red
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
            document.getElementById('total-ob').textContent = stats.totalOb;
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

            // Redraw FVG and OB rectangles
            drawFVGRectangles(frame.fvgZones);
            drawOBRectangles(frame.obZones);

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
            drawOBRectangles(frame.obZones);
        }});

        // Add throttled redraw for mouse interactions (y-axis scaling)
        let redrawTimer = null;
        let isMouseDown = false;

        function throttledRedraw() {{
            if (redrawTimer) return;
            redrawTimer = requestAnimationFrame(() => {{
                const frame = allFrames[currentFrameIdx];
                drawFVGRectangles(frame.fvgZones);
                drawOBRectangles(frame.obZones);
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
            drawOBRectangles(frame.obZones);
        }});
        resizeObserver.observe(chartDiv);

        // Load initial frame
        showFrame(0);
    </script>
</body>
</html>
"""

    def run(self, df):
        """Generate single HTML walkthrough with all candles."""
        total_candles = len(df)

        print(f"\n{'='*80}")
        print(f"{self.timeframe_name} TRADINGVIEW CANDLE-BY-CANDLE WALKTHROUGH")
        print(f"{'='*80}\n")
        print(f"Total candles: {total_candles}")
        print(f"Collecting data for {total_candles} frames...\n")

        all_frames = []
        total_sweeps = 0

        for i in range(total_candles):
            df_slice = df.iloc[:i+1]
            frame_data = self.generate_frame(df_slice, i, total_candles)
            all_frames.append(frame_data)

            if (i + 1) % 10 == 0:
                print(f"  Processed frame {i+1}/{total_candles}")

            if frame_data['stats']['liquiditySweeps'] > total_sweeps:
                print(f"  Liquidity sweep detected at candle {i+1}!")
                total_sweeps = frame_data['stats']['liquiditySweeps']

        print(f"\nCollected data for {total_candles} frames")
        print(f"Total liquidity sweeps found: {total_sweeps}")

        # Generate single HTML file with all frames
        print(f"\nGenerating single HTML file with embedded data...")
        html_content = self._create_single_html_template(all_frames)
        output_path = self.output_dir / "walkthrough.html"

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"Generated: {output_path}")
        print(f"File size: {len(html_content) / 1024:.1f} KB")

        return output_path
