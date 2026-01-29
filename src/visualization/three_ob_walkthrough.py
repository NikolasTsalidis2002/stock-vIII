"""
3-OB Strategy TradingView-Style Walkthrough Visualizer

Generates a candle-by-candle HTML walkthrough that shows the 3-OB state machine
progressing through bars.  A sidebar displays a condition checklist that updates
live, and OBs assigned as A/B/C are highlighted with distinct colors and labels.

Uses the same LightweightCharts + D3.js stack as TradingViewVisualizer.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.strategy.three_ob_engine import ThreeOBEngine, OBRecord
from src.visualization.core import NumpyEncoder, generate_candle_data


class ThreeOBWalkthrough:
    """Generates a TradingView-style walkthrough for the 3-OB state machine."""

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = project_root / 'results' / 'walkthroughs' / '3ob_walkthrough'
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        df: pd.DataFrame,
        close_break: bool = True,
        min_dist_pct: float = 0.10,
        max_signals: int = 10,
    ) -> Path:
        """Generate the walkthrough HTML.

        Args:
            df: OHLCV DataFrame with 'time' column.
            close_break: Whether close break is required for BOS.
            min_dist_pct: Minimum distance percentage for OB-C.
            max_signals: Maximum signals before stopping.

        Returns:
            Path to the generated HTML file.
        """
        print("\nInitializing 3-OB engine for walkthrough...")
        engine = ThreeOBEngine(df, close_break=close_break, min_dist_pct=min_dist_pct)

        # Prepare the working df with datetime index for candle data generation
        df_work = engine.df.copy()
        if 'time' in df_work.columns:
            df_work = df_work.set_index('time')

        print("Scanning bars and collecting snapshots...")
        all_frames = []
        total_bars = len(engine.df)

        for bar, snapshot in engine.scan_with_snapshots(max_signals=max_signals):
            frame = self._build_frame(engine, df_work, bar, snapshot)
            all_frames.append(frame)
            if (bar + 1) % 100 == 0:
                print(f"  Processed {bar + 1}/{total_bars} bars")

        print(f"Collected {len(all_frames)} frames")
        print("Generating HTML...")

        html = self._create_html(all_frames)
        output_path = self.output_dir / 'walkthrough.html'
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"Generated: {output_path}")
        print(f"File size: {len(html) / 1024:.1f} KB")
        return output_path

    def _build_frame(
        self,
        engine: ThreeOBEngine,
        df_work: pd.DataFrame,
        bar: int,
        snapshot: dict,
    ) -> dict:
        """Build a single frame dict from engine state and snapshot."""
        df_slice = df_work.iloc[:bar + 1]
        candle_data = generate_candle_data(df_slice)

        # Build OB zone data for all active OBs
        ob_zones = []
        setup_obs = []  # OBs with A/B/C roles

        role_keys = {
            'long_A': snapshot['long_ob_a_key'],
            'long_B': snapshot['long_ob_b_key'],
            'long_C': snapshot['long_ob_c_key'],
            'short_A': snapshot['short_ob_a_key'],
            'short_B': snapshot['short_ob_b_key'],
            'short_C': snapshot['short_ob_c_key'],
        }

        # Collect all role-assigned keys for quick lookup
        assigned_keys = set()
        for k in role_keys.values():
            if k is not None:
                assigned_keys.add(k)

        def _make_ob_zone(key, is_dead=False):
            rec = engine.ob_records.get(key)
            if rec is None:
                return None

            start_time = int(df_work.index[rec.start_idx].timestamp()) if rec.start_idx < len(df_work) else None
            if is_dead:
                # Dead OBs end at their invalidation bar
                end_idx = min(rec.status_idx, len(df_work) - 1) if rec.status_idx > 0 else bar
            else:
                end_idx = bar  # extend to current bar for visualization
            end_time = int(df_work.index[min(end_idx, len(df_work) - 1)].timestamp())

            if start_time is None:
                return None

            # Determine role (only for active OBs)
            role = None
            direction = None
            if not is_dead:
                for role_name, role_key in role_keys.items():
                    if role_key == key:
                        role = role_name.split('_')[1]  # 'A', 'B', or 'C'
                        direction = role_name.split('_')[0]  # 'long' or 'short'
                        break

            return {
                'startTime': start_time,
                'endTime': end_time,
                'topPrice': float(rec.top),
                'bottomPrice': float(rec.bottom),
                'obType': int(rec.direction),
                'role': role,
                'roleDirection': direction,
                'dead': is_dead,
            }

        for key in snapshot['active_ob_keys']:
            zone = _make_ob_zone(key, is_dead=False)
            if zone is None:
                continue
            ob_zones.append(zone)
            if zone['role'] is not None:
                setup_obs.append(zone)

        # Dead (invalidated) OBs
        for key in snapshot.get('dead_ob_keys', []):
            zone = _make_ob_zone(key, is_dead=True)
            if zone is not None:
                ob_zones.append(zone)

        # BOS lines
        bos_lines = []
        bos_rows = engine.bos_data[engine.bos_data['BOS'].notna()]
        for idx, row in bos_rows.iterrows():
            bos_bar_idx = int(idx)
            if bos_bar_idx > bar:
                break
            inflexion_idx = int(row['BrokenInflexionIndex']) if not pd.isna(row.get('BrokenInflexionIndex')) else bos_bar_idx
            if inflexion_idx >= len(df_work) or bos_bar_idx >= len(df_work):
                continue
            level = float(row['Level']) if not pd.isna(row.get('Level')) else 0
            bos_lines.append({
                'startTime': int(df_work.index[inflexion_idx].timestamp()),
                'endTime': int(df_work.index[bos_bar_idx].timestamp()),
                'level': level,
                'direction': int(row['BOS']),
            })

        # Current time string
        current_time = df_work.index[bar].strftime('%Y-%m-%d %H:%M') if bar < len(df_work) else ''

        # Signal info
        sig = snapshot['signal']
        signal_info = None
        if sig is not None:
            signal_info = {
                'direction': sig.entry_direction,
                'price': float(sig.price_entry),
                'tp': float(sig.take_profit_price) if sig.take_profit_price else None,
                'sl': float(sig.stop_loss_price) if sig.stop_loss_price else None,
            }

        return {
            'candleIdx': bar,
            'totalCandles': len(engine.df),
            'candleData': candle_data,
            'obZones': ob_zones,
            'bosLines': bos_lines,
            'setupOBs': setup_obs,
            'currentTime': current_time,
            'longState': snapshot['long_state'],
            'shortState': snapshot['short_state'],
            'signalFired': sig is not None,
            'signalInfo': signal_info,
            'signalsSoFar': snapshot['signals_so_far'],
            'activeOBCount': len(snapshot['active_ob_keys']),
        }

    def _create_html(self, all_frames: list) -> str:
        """Generate the complete HTML walkthrough."""
        frames_json = json.dumps(all_frames, cls=NumpyEncoder)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>3-OB Strategy Walkthrough</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
            width: 320px;
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
        button:hover {{ background-color: #1e53e5; }}
        button:disabled {{ background-color: #2b2b43; cursor: not-allowed; }}
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
        #goto-input:focus {{ outline: none; border-color: #2962ff; }}
        .help-text {{
            text-align: center;
            color: #787b86;
            font-size: 11px;
            margin-top: 10px;
        }}
        .state-group {{
            margin: 15px 0;
            padding: 15px;
            background-color: #131722;
            border-radius: 4px;
        }}
        .state-group h2 {{
            margin: 0 0 10px 0;
            font-size: 13px;
            color: #787b86;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .checklist-item {{
            display: flex;
            align-items: center;
            margin: 6px 0;
            font-size: 13px;
            padding: 4px 0;
        }}
        .check-icon {{
            width: 20px;
            font-size: 14px;
            margin-right: 8px;
            text-align: center;
        }}
        .check-icon.done {{ color: #089981; }}
        .check-icon.pending {{ color: #363a45; }}
        .checklist-label {{ color: #d1d4dc; }}
        .checklist-label.done {{ color: #089981; }}
        .checklist-label.pending {{ color: #787b86; }}
        .signal-counter {{
            margin-top: 15px;
            padding: 12px;
            background-color: #131722;
            border-radius: 4px;
            text-align: center;
        }}
        .signal-counter .count {{
            font-size: 24px;
            font-weight: bold;
            color: #2962ff;
        }}
        .signal-counter .label {{
            font-size: 12px;
            color: #787b86;
            margin-top: 4px;
        }}
        .stat-item {{
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
            font-size: 13px;
        }}
        .stat-label {{ color: #787b86; }}
        .stat-value {{ color: #d1d4dc; font-weight: 600; }}
        .signal-alert {{
            display: none;
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 600;
            text-align: center;
        }}
        .signal-alert.long {{
            background-color: rgba(8, 153, 129, 0.2);
            border: 1px solid #089981;
            color: #089981;
        }}
        .signal-alert.short {{
            background-color: rgba(242, 54, 69, 0.2);
            border: 1px solid #f23645;
            color: #f23645;
        }}
        .legend {{
            margin-top: 15px;
            padding: 10px;
            background-color: #131722;
            border-radius: 4px;
        }}
        .legend h2 {{
            margin: 0 0 8px 0;
            font-size: 13px;
            color: #787b86;
            text-transform: uppercase;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 12px;
        }}
        .legend-swatch {{
            width: 14px;
            height: 14px;
            margin-right: 8px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="chart">
            <div id="chart-container"></div>
        </div>
        <div id="sidebar">
            <h1>3-OB STATE MACHINE</h1>

            <div class="controls">
                <div class="frame-info" id="frame-number">Bar 1 of -</div>
                <button id="first-btn" onclick="firstFrame()">&#9198; First</button>
                <button id="prev-btn" onclick="prevFrame()">&#9664; Previous</button>
                <button id="next-btn" onclick="nextFrame()">Next &#9654;</button>
                <button id="last-btn" onclick="lastFrame()">Last &#9197;</button>
                <div class="goto-row">
                    <input type="number" id="goto-input" min="1" placeholder="Go to...">
                    <button id="goto-btn" onclick="gotoFrame()">Go</button>
                </div>
                <p class="help-text">Arrow keys / Home / End</p>
            </div>

            <div class="state-group">
                <div class="stat-item">
                    <span class="stat-label">Time</span>
                    <span class="stat-value" id="current-time">-</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Active OBs</span>
                    <span class="stat-value" id="active-ob-count">0</span>
                </div>
            </div>

            <div id="signal-alert" class="signal-alert"></div>

            <div class="state-group">
                <h2>Long Setup</h2>
                <div class="checklist-item" id="long-check-0">
                    <span class="check-icon pending">&#9744;</span>
                    <span class="checklist-label pending">OB-A touched &amp; left</span>
                </div>
                <div class="checklist-item" id="long-check-1">
                    <span class="check-icon pending">&#9744;</span>
                    <span class="checklist-label pending">OB-C found (min dist)</span>
                </div>
                <div class="checklist-item" id="long-check-2">
                    <span class="check-icon pending">&#9744;</span>
                    <span class="checklist-label pending">OB-B formed</span>
                </div>
                <div class="checklist-item" id="long-check-3">
                    <span class="check-icon pending">&#9744;</span>
                    <span class="checklist-label pending">OB-B retouched</span>
                </div>
                <div class="checklist-item" id="long-check-4">
                    <span class="check-icon pending">&#9744;</span>
                    <span class="checklist-label pending">Close break past OB-B</span>
                </div>
            </div>

            <div class="state-group">
                <h2>Short Setup</h2>
                <div class="checklist-item" id="short-check-0">
                    <span class="check-icon pending">&#9744;</span>
                    <span class="checklist-label pending">OB-A touched &amp; left</span>
                </div>
                <div class="checklist-item" id="short-check-1">
                    <span class="check-icon pending">&#9744;</span>
                    <span class="checklist-label pending">OB-C found (min dist)</span>
                </div>
                <div class="checklist-item" id="short-check-2">
                    <span class="check-icon pending">&#9744;</span>
                    <span class="checklist-label pending">OB-B formed</span>
                </div>
                <div class="checklist-item" id="short-check-3">
                    <span class="check-icon pending">&#9744;</span>
                    <span class="checklist-label pending">OB-B retouched</span>
                </div>
                <div class="checklist-item" id="short-check-4">
                    <span class="check-icon pending">&#9744;</span>
                    <span class="checklist-label pending">Close break past OB-B</span>
                </div>
            </div>

            <div class="signal-counter">
                <div class="count" id="signal-count">0</div>
                <div class="label">SIGNALS FIRED</div>
            </div>

            <div class="legend">
                <h2>OB Legend</h2>
                <div class="legend-item">
                    <div class="legend-swatch" style="background: rgba(0,188,212,0.3); border: 2px solid #00bcd4;"></div>
                    <span>Bullish OB</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch" style="background: rgba(255,87,34,0.3); border: 2px solid #ff5722;"></div>
                    <span>Bearish OB</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch" style="background: rgba(76,175,80,0.3); border: 2px solid #4caf50;"></div>
                    <span>OB-A (setup)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch" style="background: rgba(33,150,243,0.3); border: 2px solid #2196f3;"></div>
                    <span>OB-B (setup)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch" style="background: rgba(244,67,54,0.3); border: 2px solid #f44336;"></div>
                    <span>OB-C (setup)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch" style="background: rgba(150,150,150,0.07); border: 1px dashed rgba(150,150,150,0.3);"></div>
                    <span>Dead OB (invalidated)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch" style="background: none; border-top: 2px dashed #089981; height: 0; margin-top: 7px;"></div>
                    <span>BOS (bullish)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch" style="background: none; border-top: 2px dashed #f23645; height: 0; margin-top: 7px;"></div>
                    <span>BOS (bearish)</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const allFrames = {frames_json};
        let currentFrameIdx = 0;

        const chartContainer = document.getElementById('chart-container');
        const chart = LightweightCharts.createChart(chartContainer, {{
            layout: {{
                background: {{ color: '#131722' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#1e222d' }},
                horzLines: {{ color: '#1e222d' }},
            }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            rightPriceScale: {{ borderColor: '#2b2b43' }},
            timeScale: {{
                borderColor: '#2b2b43',
                timeVisible: true,
                secondsVisible: false,
            }},
        }});

        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#089981',
            downColor: '#f23645',
            borderVisible: false,
            wickUpColor: '#089981',
            wickDownColor: '#f23645',
        }});

        const svg = d3.select(chartContainer)
            .append('svg')
            .attr('id', 'svg-overlay')
            .style('position', 'absolute')
            .style('top', '0')
            .style('left', '0')
            .style('z-index', '10')
            .style('pointer-events', 'none');

        function updateSVGSize() {{
            const rect = chartContainer.getBoundingClientRect();
            svg.attr('width', rect.width).attr('height', rect.height);
        }}

        // OB role colors
        const roleColors = {{
            'A': {{ fill: 'rgba(76,175,80,0.25)', stroke: '#4caf50' }},
            'B': {{ fill: 'rgba(33,150,243,0.25)', stroke: '#2196f3' }},
            'C': {{ fill: 'rgba(244,67,54,0.25)', stroke: '#f44336' }},
        }};
        const defaultColors = {{
            1:  {{ fill: 'rgba(0,188,212,0.15)', stroke: '#00bcd4' }},
            '-1': {{ fill: 'rgba(255,87,34,0.15)', stroke: '#ff5722' }},
        }};

        function drawOBs(obZones, bosLines) {{
            updateSVGSize();
            svg.selectAll('.ob-rect').remove();
            svg.selectAll('.ob-label').remove();
            svg.selectAll('.bos-line').remove();
            const ts = chart.timeScale();

            obZones.forEach(zone => {{
                const x1 = ts.timeToCoordinate(zone.startTime);
                const x2 = ts.timeToCoordinate(zone.endTime);
                const y1 = candleSeries.priceToCoordinate(zone.topPrice);
                const y2 = candleSeries.priceToCoordinate(zone.bottomPrice);
                if (x1 == null || x2 == null || y1 == null || y2 == null) return;

                const x = Math.min(x1, x2);
                const y = Math.min(y1, y2);
                const w = Math.max(Math.abs(x2 - x1), 4);
                const h = Math.max(Math.abs(y2 - y1), 2);

                let colors;
                let strokeWidth = 1;
                let dashArray = 'none';
                if (zone.dead) {{
                    // Dead OBs: very low opacity, dashed stroke
                    colors = {{
                        fill: zone.obType === 1 ? 'rgba(0,188,212,0.07)' : 'rgba(255,87,34,0.07)',
                        stroke: zone.obType === 1 ? 'rgba(0,188,212,0.3)' : 'rgba(255,87,34,0.3)',
                    }};
                    dashArray = '4,3';
                }} else if (zone.role && roleColors[zone.role]) {{
                    colors = roleColors[zone.role];
                    strokeWidth = 2;
                }} else {{
                    colors = defaultColors[String(zone.obType)] || defaultColors['1'];
                }}

                svg.append('rect')
                    .attr('class', 'ob-rect')
                    .attr('x', x)
                    .attr('y', y)
                    .attr('width', w)
                    .attr('height', h)
                    .attr('fill', colors.fill)
                    .attr('stroke', colors.stroke)
                    .attr('stroke-width', strokeWidth)
                    .attr('stroke-dasharray', dashArray);

                // Draw role label (skip for dead OBs)
                if (zone.role && !zone.dead) {{
                    svg.append('text')
                        .attr('class', 'ob-label')
                        .attr('x', x + 4)
                        .attr('y', y + 14)
                        .attr('fill', colors.stroke)
                        .attr('font-size', '12px')
                        .attr('font-weight', 'bold')
                        .attr('font-family', 'sans-serif')
                        .text(zone.role);
                }}
            }});

            // Draw BOS lines
            if (bosLines) {{
                bosLines.forEach(bos => {{
                    const x1 = ts.timeToCoordinate(bos.startTime);
                    const x2 = ts.timeToCoordinate(bos.endTime);
                    const y = candleSeries.priceToCoordinate(bos.level);
                    if (x1 == null || x2 == null || y == null) return;

                    const color = bos.direction === 1 ? '#089981' : '#f23645';
                    svg.append('line')
                        .attr('class', 'bos-line')
                        .attr('x1', x1)
                        .attr('x2', x2)
                        .attr('y1', y)
                        .attr('y2', y)
                        .attr('stroke', color)
                        .attr('stroke-width', 1.5)
                        .attr('stroke-dasharray', '6,3')
                        .attr('opacity', 0.7);
                }});
            }}
        }}

        function updateChecklist(prefix, state, signalFired) {{
            // States: 0 = nothing, 1 = OB-A+C found, 2 = OB-B found, 3 = OB-B retouched
            // Check 0: OB-A touched & left (state >= 1)
            // Check 1: OB-C found (state >= 1)
            // Check 2: OB-B formed (state >= 2)
            // Check 3: OB-B retouched (state >= 3)
            // Check 4: Close break past OB-B (signal fired)
            const thresholds = [1, 1, 2, 3, 99];  // 99 = signal check
            for (let i = 0; i < 5; i++) {{
                const el = document.getElementById(prefix + '-check-' + i);
                const icon = el.querySelector('.check-icon');
                const label = el.querySelector('.checklist-label');
                let met = false;
                if (i < 4) met = state >= thresholds[i];
                else met = signalFired;

                if (met) {{
                    icon.innerHTML = '&#9745;';
                    icon.className = 'check-icon done';
                    label.className = 'checklist-label done';
                }} else {{
                    icon.innerHTML = '&#9744;';
                    icon.className = 'check-icon pending';
                    label.className = 'checklist-label pending';
                }}
            }}
        }}

        function showFrame(n) {{
            const savedRange = (currentFrameIdx !== n) ? chart.timeScale().getVisibleLogicalRange() : null;
            currentFrameIdx = Math.max(0, Math.min(n, allFrames.length - 1));
            const f = allFrames[currentFrameIdx];

            candleSeries.setData(f.candleData);
            drawOBs(f.obZones, f.bosLines);

            // Update sidebar
            document.getElementById('frame-number').textContent =
                'Bar ' + (currentFrameIdx + 1) + ' of ' + f.totalCandles;
            document.getElementById('current-time').textContent = f.currentTime;
            document.getElementById('active-ob-count').textContent = f.activeOBCount;
            document.getElementById('signal-count').textContent = f.signalsSoFar;

            // Determine if signal fired on this specific bar (for checklist check-3)
            const longSignalFired = f.signalFired && f.signalInfo && f.signalInfo.direction === 'long';
            const shortSignalFired = f.signalFired && f.signalInfo && f.signalInfo.direction === 'short';

            updateChecklist('long', f.longState, longSignalFired);
            updateChecklist('short', f.shortState, shortSignalFired);

            // Signal alert
            const alertEl = document.getElementById('signal-alert');
            if (f.signalFired && f.signalInfo) {{
                const si = f.signalInfo;
                alertEl.style.display = 'block';
                alertEl.className = 'signal-alert ' + si.direction;
                let text = si.direction.toUpperCase() + ' SIGNAL @ $' + si.price.toFixed(2);
                if (si.tp) text += ' | TP: $' + si.tp.toFixed(2);
                if (si.sl) text += ' | SL: $' + si.sl.toFixed(2);
                alertEl.textContent = text;
            }} else {{
                alertEl.style.display = 'none';
            }}

            // Navigation buttons
            document.getElementById('prev-btn').disabled = (currentFrameIdx === 0);
            document.getElementById('first-btn').disabled = (currentFrameIdx === 0);
            document.getElementById('next-btn').disabled = (currentFrameIdx === allFrames.length - 1);
            document.getElementById('last-btn').disabled = (currentFrameIdx === allFrames.length - 1);

            if (savedRange) {{
                chart.timeScale().setVisibleLogicalRange(savedRange);
            }} else {{
                chart.timeScale().fitContent();
            }}
        }}

        function nextFrame() {{ showFrame(currentFrameIdx + 1); }}
        function prevFrame() {{ showFrame(currentFrameIdx - 1); }}
        function firstFrame() {{ showFrame(0); }}
        function lastFrame() {{ showFrame(allFrames.length - 1); }}
        function gotoFrame() {{
            const v = parseInt(document.getElementById('goto-input').value);
            if (v >= 1 && v <= allFrames.length) showFrame(v - 1);
            document.getElementById('goto-input').value = '';
        }}

        document.addEventListener('keydown', function(e) {{
            if (e.target.id === 'goto-input') return;
            if (e.key === 'ArrowRight') nextFrame();
            if (e.key === 'ArrowLeft') prevFrame();
            if (e.key === 'Home') firstFrame();
            if (e.key === 'End') lastFrame();
        }});
        document.getElementById('goto-input').addEventListener('keydown', function(e) {{
            if (e.key === 'Enter') {{ e.stopPropagation(); gotoFrame(); }}
        }});

        // Redraw OBs on zoom/pan/resize
        chart.timeScale().subscribeVisibleLogicalRangeChange(() => {{
            drawOBs(allFrames[currentFrameIdx].obZones, allFrames[currentFrameIdx].bosLines);
        }});

        let redrawTimer = null;
        let isMouseDown = false;
        function throttledRedraw() {{
            if (redrawTimer) return;
            redrawTimer = requestAnimationFrame(() => {{
                drawOBs(allFrames[currentFrameIdx].obZones, allFrames[currentFrameIdx].bosLines);
                redrawTimer = null;
            }});
        }}
        chartContainer.addEventListener('wheel', throttledRedraw);
        chartContainer.addEventListener('mousedown', () => {{ isMouseDown = true; }});
        chartContainer.addEventListener('mousemove', () => {{ if (isMouseDown) throttledRedraw(); }});
        chartContainer.addEventListener('mouseup', () => {{ isMouseDown = false; throttledRedraw(); }});
        chartContainer.addEventListener('mouseleave', () => {{ isMouseDown = false; }});

        new ResizeObserver(() => {{
            chart.applyOptions({{ width: chartContainer.clientWidth, height: chartContainer.clientHeight }});
            drawOBs(allFrames[currentFrameIdx].obZones, allFrames[currentFrameIdx].bosLines);
        }}).observe(document.getElementById('chart'));

        showFrame(0);
    </script>
</body>
</html>"""
