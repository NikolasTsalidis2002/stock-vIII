"""
OB MFE Highlight Viewer — Standalone HTML dashboard.

Full-history zoomable candlestick chart (LightweightCharts + D3.js) with all OB
rectangles overlaid.  OBs with mfe_pct >= threshold are highlighted (green/red);
others are dimmed grey.  An HTML slider controls the MFE threshold interactively.

Usage:
    python3 scripts/ob_mfe_viewer.py
    python3 scripts/ob_mfe_viewer.py --symbol TSLA --timeframe 15min
"""

import sys
import os
import argparse
import json
import webbrowser
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import DataLoader
from indicators import smc, smc_custom
from visualization.core import generate_candle_data, NumpyEncoder

# Reuse zone extraction + MFE computation from smc_dashboard
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from smc_dashboard import extract_zones, compute_reversal_quality


def build_ob_zones_with_mfe(df, ob, zones_df, reversal_data):
    """Build OB zone list with MFE data for the JS side."""
    rev_df = pd.DataFrame(reversal_data)
    zones_df = zones_df.reset_index(drop=True)
    zones_df['mfe_pct'] = rev_df['mfe_pct'].values

    ob_zones = zones_df[zones_df['zone_type'] == 'OB'].copy()

    result = []
    n = len(df)
    for _, row in ob_zones.iterrows():
        ci = int(row['creation_index'])
        if ci >= n:
            continue

        si = row['status_index']
        if si is not None and not (isinstance(si, float) and np.isnan(si)):
            end_idx = min(int(si), n - 1)
        else:
            end_idx = n - 1

        mfe = row['mfe_pct']
        if mfe is not None and isinstance(mfe, float) and np.isnan(mfe):
            mfe = None

        result.append({
            'startTime': int(df.index[ci].timestamp()),
            'endTime': int(df.index[end_idx].timestamp()),
            'topPrice': float(row['top']),
            'bottomPrice': float(row['bottom']),
            'obType': int(row['direction']),
            'mfePct': float(mfe) if mfe is not None else None,
        })

    return result


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OB MFE Highlight — {symbol} {timeframe}</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #131722; color: #d1d4dc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
#controls {{
    display: flex; align-items: center; gap: 16px;
    padding: 10px 20px; background: #1e222d; border-bottom: 1px solid #2b2b43;
}}
#controls label {{ font-size: 14px; }}
#mfe-slider {{ width: 300px; accent-color: #2962ff; }}
#mfe-value {{ font-weight: bold; color: #2962ff; min-width: 50px; }}
#ob-count {{ font-size: 13px; color: #8a8e96; }}
#chart-area {{ position: relative; width: 100%; height: calc(100vh - 50px); }}
#chart-container {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; }}
#svg-overlay {{ position: absolute; top: 0; left: 0; pointer-events: none; z-index: 10; }}
</style>
</head>
<body>
<div id="controls">
    <label>MFE threshold &ge;</label>
    <input type="range" id="mfe-slider" min="0" max="10" step="0.5" value="2.0">
    <span id="mfe-value">2.0%</span>
    <span id="ob-count"></span>
</div>
<div id="chart-area">
    <div id="chart-container"></div>
</div>
<script>
const candleData = {candle_json};
const obZones = {ob_json};

let chart, series;
let mfeThreshold = 2.0;

function init() {{
    const container = document.getElementById('chart-container');

    chart = LightweightCharts.createChart(container, {{
        layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
        grid: {{ vertLines: {{ color: '#1e222d' }}, horzLines: {{ color: '#1e222d' }} }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
        rightPriceScale: {{ borderColor: '#2b2b43', scaleMargins: {{ top: 0.05, bottom: 0.05 }} }},
        timeScale: {{ borderColor: '#2b2b43', timeVisible: true, secondsVisible: false }},
    }});

    series = chart.addCandlestickSeries({{
        upColor: '#089981', downColor: '#f23645',
        borderVisible: false, wickUpColor: '#089981', wickDownColor: '#f23645',
    }});
    series.setData(candleData);

    d3.select(container)
        .append('svg')
        .attr('id', 'svg-overlay')
        .style('position', 'absolute')
        .style('top', '0')
        .style('left', '0')
        .style('pointer-events', 'none');

    chart.timeScale().subscribeVisibleLogicalRangeChange(redrawOverlays);

    const ro = new ResizeObserver(() => {{
        chart.applyOptions({{ width: container.clientWidth, height: container.clientHeight }});
        redrawOverlays();
    }});
    ro.observe(container);

    // Mouse-driven redraws for smooth panning
    let mouseDown = false;
    container.addEventListener('wheel', throttledRedraw);
    container.addEventListener('mousedown', () => {{ mouseDown = true; }});
    container.addEventListener('mousemove', () => {{ if (mouseDown) throttledRedraw(); }});
    container.addEventListener('mouseup', () => {{ mouseDown = false; throttledRedraw(); }});

    // Slider
    const slider = document.getElementById('mfe-slider');
    const valEl = document.getElementById('mfe-value');
    slider.addEventListener('input', () => {{
        mfeThreshold = parseFloat(slider.value);
        valEl.textContent = mfeThreshold.toFixed(1) + '%';
        redrawOverlays();
    }});

    redrawOverlays();
}}

let redrawTimer = null;
function throttledRedraw() {{
    if (redrawTimer) return;
    redrawTimer = requestAnimationFrame(() => {{
        redrawTimer = null;
        redrawOverlays();
    }});
}}

function redrawOverlays() {{
    const container = document.getElementById('chart-container');
    const rect = container.getBoundingClientRect();
    const svg = d3.select('#svg-overlay')
        .attr('width', rect.width)
        .attr('height', rect.height);
    svg.selectAll('*').remove();

    const ts = chart.timeScale();
    let highlighted = 0;

    for (const z of obZones) {{
        const x1 = ts.timeToCoordinate(z.startTime);
        const x2 = ts.timeToCoordinate(z.endTime);
        const y1 = series.priceToCoordinate(z.topPrice);
        const y2 = series.priceToCoordinate(z.bottomPrice);
        if (x1 == null || x2 == null || y1 == null || y2 == null) continue;

        const x = Math.min(x1, x2);
        const y = Math.min(y1, y2);
        const w = Math.max(Math.abs(x2 - x1), 4);
        const h = Math.max(Math.abs(y2 - y1), 2);

        const qualifies = z.mfePct != null && z.mfePct >= mfeThreshold;
        let fill, stroke, strokeW;

        if (qualifies) {{
            highlighted++;
            fill = z.obType === 1 ? 'rgba(38,166,154,0.35)' : 'rgba(239,83,80,0.35)';
            stroke = z.obType === 1 ? 'rgba(38,166,154,0.8)' : 'rgba(239,83,80,0.8)';
            strokeW = 1.5;
        }} else {{
            fill = 'rgba(128,128,128,0.06)';
            stroke = 'none';
            strokeW = 0;
        }}

        svg.append('rect')
            .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
            .attr('fill', fill).attr('stroke', stroke).attr('stroke-width', strokeW);

        if (qualifies && z.mfePct != null) {{
            svg.append('text')
                .attr('x', x + 3).attr('y', y + 12)
                .attr('fill', stroke).attr('font-size', '10px').attr('font-weight', 'bold')
                .text(z.mfePct.toFixed(1) + '%');
        }}
    }}

    document.getElementById('ob-count').textContent =
        highlighted + ' / ' + obZones.length + ' OBs highlighted';
}}

init();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description='OB MFE Highlight Viewer')
    parser.add_argument('--symbol', default='META')
    parser.add_argument('--timeframe', default='15min')
    args = parser.parse_args()

    # Load data and compute indicators
    loader = DataLoader(args.symbol)
    df = loader.get_data(args.timeframe)
    if df is None or df.empty:
        print(f"No data for {args.symbol} {args.timeframe}")
        sys.exit(1)

    inflexions = smc_custom.inflexion_points(df)
    bos = smc_custom.bos(df, inflexions, close_break=True)
    fvg = smc.fvg(df, join_consecutive=True)
    ob = smc_custom.ob(df, bos, inflexions)

    # Extract zones and compute reversal quality / MFE
    zones_df = extract_zones(df, fvg, ob, inflexions)
    reversal_data = compute_reversal_quality(df, zones_df)

    # Ensure df has DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('time')

    # Build OB data with MFE for JS
    ob_data = build_ob_zones_with_mfe(df, ob, zones_df, reversal_data)
    print(f"Total OB zones: {len(ob_data)}")
    candle_data = generate_candle_data(df)

    # Build HTML
    candle_json = json.dumps(candle_data, cls=NumpyEncoder)
    ob_json = json.dumps(ob_data, cls=NumpyEncoder)

    html = HTML_TEMPLATE.format(
        symbol=args.symbol,
        timeframe=args.timeframe,
        candle_json=candle_json,
        ob_json=ob_json,
    )

    # Write output
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'charts')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ob_mfe_highlight.html')
    with open(out_path, 'w') as f:
        f.write(html)

    print(f"Written to {out_path}")
    # webbrowser.open('file://' + os.path.abspath(out_path))


if __name__ == '__main__':
    main()
