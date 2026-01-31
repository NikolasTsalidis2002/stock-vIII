"""
Left/Right Split-View: Candlestick (left) + Zone Histogram (right)

Single HTML page with horizontal layout:
  Left  = LightweightCharts candlestick with SMC indicators (FVG, OB, BOS, Inflexions)
  Right = D3 histogram panel showing exact zone bands at real price ranges + current price line

Bar-by-bar walkthrough: frame N shows candles 0..N.
Arrow key navigation, play/pause, slider.  Full zoom/pan (save/restore visible range).

Usage:
    python3 scripts/price_distribution_indicators.py
    python3 scripts/price_distribution_indicators.py --timeframe 15min --step 5
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
from visualization.core import (
    NumpyEncoder,
    generate_candle_data,
    generate_fvg_zones,
    generate_ob_zones,
    generate_bos_lines,
    generate_inflexion_markers,
    generate_inflexion_lines,
)


def generate_price_levels(df, step=None, max_points_per_candle=50):
    """Generate discrete price levels from candle high-low ranges."""
    lows = df['low'].values
    highs = df['high'].values

    if step is None:
        avg_range = np.mean(highs - lows)
        step = max(avg_range / max_points_per_candle, 0.01)

    n_points = np.minimum(
        np.ceil((highs - lows) / step + 1).astype(int),
        max_points_per_candle
    )
    total_points = n_points.sum()

    all_prices = np.empty(total_points)
    idx = 0
    for low, high, n in zip(lows, highs, n_points):
        all_prices[idx:idx + n] = np.linspace(low, high, n)
        idx += n

    return all_prices


def build_chart_frame(df_slice):
    """Build candlestick + indicator data for a single frame.

    Returns (chart_data, raw_indicators) where raw_indicators contains
    the DataFrames used, so the histogram can reuse the same data.
    """
    inflexions = smc_custom.inflexion_points(df_slice)
    bos = smc_custom.bos(df_slice, inflexions, close_break=True)
    fvg = smc.fvg(df_slice, join_consecutive=True)
    ob = smc_custom.ob(df_slice, bos, inflexions)

    chart_data = {
        'candleData': generate_candle_data(df_slice),
        'fvgZones': generate_fvg_zones(df_slice, fvg),
        'obZones': generate_ob_zones(df_slice, ob),
        'bosLines': generate_bos_lines(df_slice, bos, inflexions),
        'inflexionMarkers': generate_inflexion_markers(df_slice, inflexions),
        'inflexionLines': generate_inflexion_lines(df_slice, inflexions),
    }
    raw_indicators = {
        'fvg': fvg,
        'ob': ob,
        'inflexions': inflexions,
    }
    return chart_data, raw_indicators


def collect_histogram_zones(fvg_data, ob_data, inflexion_data):
    """Collect active zone bands from the same indicator DataFrames used by the chart.

    Uses Respected column directly — these DataFrames were computed on the same
    slice as the chart, so zones match exactly.
    """
    fvg_zones = []
    for i in range(len(fvg_data)):
        if pd.isna(fvg_data['FVG'].iloc[i]):
            continue
        respected = fvg_data['Respected'].iloc[i] if 'Respected' in fvg_data.columns else None
        if respected is False:
            continue
        fvg_zones.append({
            'top': float(fvg_data['Top'].iloc[i]),
            'bottom': float(fvg_data['Bottom'].iloc[i]),
            'mitigated': bool(respected) if respected is not None else False,
        })

    ob_zones = []
    for i in range(len(ob_data)):
        if pd.isna(ob_data['OB'].iloc[i]):
            continue
        respected = ob_data['Respected'].iloc[i] if 'Respected' in ob_data.columns else None
        if respected is False:
            continue
        ob_zones.append({
            'top': float(ob_data['Top'].iloc[i]),
            'bottom': float(ob_data['Bottom'].iloc[i]),
            'mitigated': bool(respected) if respected is not None else False,
        })

    inflexion_levels = []
    for i in range(len(inflexion_data)):
        if pd.isna(inflexion_data['InflexionType'].iloc[i]):
            continue
        respected = inflexion_data['Respected'].iloc[i] if 'Respected' in inflexion_data.columns else None
        if respected is False:
            continue
        level = inflexion_data['Level'].iloc[i]
        if pd.isna(level):
            continue
        inflexion_levels.append({
            'level': float(level),
            'mitigated': bool(respected) if respected is not None else False,
        })

    return {
        'fvgZones': fvg_zones,
        'obZones': ob_zones,
        'inflexionLevels': inflexion_levels,
    }


def build_html(symbol, timeframe, all_frames_json):
    """Return the complete HTML string for the left/right layout page."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{symbol} ({timeframe}) — Price Distribution + Candlestick</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#131722; color:#d1d4dc; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; overflow:hidden; }}
#main {{ display:flex; flex-direction:column; height:100vh; }}
#panels {{ display:flex; flex:1; min-height:0; }}
#chart-panel {{ flex:1; position:relative; min-width:0; }}
#chart-container {{ width:100%; height:100%; position:relative; }}
#svg-overlay {{ position:absolute; top:0; left:0; pointer-events:none; z-index:10; }}
#hist-panel {{ width:180px; position:relative; border-left:1px solid #2b2b43; background:#131722; }}
.hist-label {{ position:absolute; top:4px; left:8px; font-size:11px; color:#787b86; z-index:2; pointer-events:none; }}
.indicator-panel {{ position:relative; border-left:1px solid #2b2b43; background:#131722; overflow:hidden; }}
.divider {{ width:4px; cursor:col-resize; background:#2b2b43; flex-shrink:0; }}
.divider:hover {{ background:#3b3b53; }}
#legend {{ position:absolute; top:20px; left:8px; font-size:10px; z-index:2; display:flex; flex-direction:column; gap:4px; }}
.legend-item {{ display:flex; align-items:center; gap:4px; }}
.legend-swatch {{ width:10px; height:10px; border-radius:2px; }}
#controls {{ display:flex; align-items:center; gap:10px; padding:8px 16px; background:#1e222d; border-top:1px solid #2b2b43; }}
#controls button {{ background:#2962ff; color:#fff; border:none; padding:6px 14px; cursor:pointer; font-size:12px; font-weight:600; border-radius:4px; }}
#controls button:hover {{ background:#1e53e5; }}
#controls button:disabled {{ background:#2b2b43; cursor:not-allowed; }}
#slider {{ flex:1; }}
#frame-info {{ font-size:12px; color:#787b86; white-space:nowrap; }}
</style>
</head>
<body>
<div id="main">
  <div id="panels">
    <div id="chart-panel">
      <div id="chart-container"></div>
    </div>
    <div class="divider" data-left="chart-panel" data-right="hist-panel"></div>
    <div id="hist-panel" style="width:180px">
      <span class="hist-label">Histogram</span>
      <div id="legend">
        <div class="legend-item"><div class="legend-swatch" style="background:#787b86"></div><span style="color:#787b86">Price dist.</span></div>
        <div class="legend-item"><div class="legend-swatch" style="background:#ffffff"></div><span style="color:#ffffff">Price</span></div>
      </div>
    </div>
    <div class="divider" data-left="hist-panel" data-right="fvg-panel"></div>
    <div id="fvg-panel" class="indicator-panel" style="width:60px"><span class="hist-label">FVG</span></div>
    <div class="divider" data-left="fvg-panel" data-right="ob-panel"></div>
    <div id="ob-panel" class="indicator-panel" style="width:60px"><span class="hist-label">OB</span></div>
    <div class="divider" data-left="ob-panel" data-right="swing-panel"></div>
    <div id="swing-panel" class="indicator-panel" style="width:60px"><span class="hist-label">Swing</span></div>
  </div>
  <div id="controls">
    <button id="prev-btn" onclick="prevFrame()">&#9664;</button>
    <button id="play-btn" onclick="togglePlay()">&#9654; Play</button>
    <button id="next-btn" onclick="nextFrame()">&#9654;</button>
    <input type="range" id="slider" min="0" max="0" value="0">
    <span id="frame-info">-</span>
  </div>
</div>

<script>
const allFrames = {all_frames_json};
let currentIdx = 0;
let playing = false;
let playTimer = null;
let savedRange = null;

// ── LightweightCharts setup ──
const chartContainer = document.getElementById('chart-container');
const chart = LightweightCharts.createChart(chartContainer, {{
    layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
    grid: {{ vertLines: {{ color: '#1e222d' }}, horzLines: {{ color: '#1e222d' }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    rightPriceScale: {{ borderColor: '#2b2b43' }},
    timeScale: {{ borderColor: '#2b2b43', timeVisible: true, secondsVisible: false }},
}});
const candleSeries = chart.addCandlestickSeries({{
    upColor: '#089981', downColor: '#f23645', borderVisible: false,
    wickUpColor: '#089981', wickDownColor: '#f23645',
}});

// SVG overlay for FVG/OB rectangles
const svg = d3.select(chartContainer).append('svg')
    .attr('id', 'svg-overlay')
    .style('position', 'absolute').style('top', '0').style('left', '0')
    .style('z-index', '10').style('pointer-events', 'none');

let activeLineSeries = [];

function updateSVGSize() {{
    const r = chartContainer.getBoundingClientRect();
    svg.attr('width', r.width).attr('height', r.height);
}}

function drawRects(zones, cls, defaultFill, defaultStroke) {{
    updateSVGSize();
    svg.selectAll('rect.' + cls).remove();
    const ts = chart.timeScale();
    zones.forEach(z => {{
        const x1 = ts.timeToCoordinate(z.startTime);
        const x2 = ts.timeToCoordinate(z.endTime);
        const y1 = candleSeries.priceToCoordinate(z.topPrice);
        const y2 = candleSeries.priceToCoordinate(z.bottomPrice);
        if (x1 == null || x2 == null || y1 == null || y2 == null) return;
        let fill = defaultFill, stroke = defaultStroke;
        if (cls === 'ob') {{
            fill = z.obType === 1 ? 'rgba(0,188,212,0.2)' : 'rgba(255,87,34,0.2)';
            stroke = z.obType === 1 ? '#00bcd4' : '#ff5722';
        }}
        svg.append('rect').attr('class', cls)
            .attr('x', Math.min(x1,x2)).attr('y', Math.min(y1,y2))
            .attr('width', Math.abs(x2-x1)).attr('height', Math.abs(y2-y1))
            .attr('fill', fill).attr('stroke', stroke).attr('stroke-width', 1)
            .attr('stroke-dasharray', cls === 'fvg' ? '4,4' : 'none');
    }});
}}

function clearLines() {{
    activeLineSeries.forEach(s => chart.removeSeries(s));
    activeLineSeries = [];
}}

function drawLines(lines) {{
    lines.forEach(l => {{
        const s = chart.addLineSeries({{
            color: l.color,
            lineWidth: l.lineWidth || 2,
            lineStyle: (l.lineStyle === 'Solid' || !l.lineStyle) ? LightweightCharts.LineStyle.Solid : LightweightCharts.LineStyle.Dotted,
            priceLineVisible: false, lastValueVisible: false,
        }});
        s.setData([{{ time: l.startTime, value: l.price }}, {{ time: l.endTime, value: l.price }}]);
        activeLineSeries.push(s);
    }});
}}

// ── Multi-panel setup (Histogram + FVG + OB + Swing) ──
const panels = {{}};

function initPanel(containerId, marginConfig, withYAxis) {{
    if (allFrames.length === 0) return null;
    const container = document.getElementById(containerId);
    // Remove old SVG if any
    d3.select(container).select('svg').remove();
    const rect = container.getBoundingClientRect();
    const margin = marginConfig;
    const w = rect.width - margin.left - margin.right;
    const h = rect.height - margin.top - margin.bottom;
    if (w <= 0 || h <= 0) return null;

    const svg = d3.select(container).append('svg')
        .attr('width', rect.width).attr('height', rect.height)
        .append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

    const f0 = allFrames[0];
    const yScale = d3.scaleLinear().domain(f0.histogram.priceRange).range([h, 0]);
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, w]);

    if (withYAxis) {{
        svg.append('g').attr('class', 'y-axis')
            .call(d3.axisLeft(yScale).ticks(10).tickFormat(d3.format('.1f')))
            .selectAll('text').style('fill', '#787b86').style('font-size', '10px');
        svg.selectAll('.y-axis path, .y-axis line').style('stroke', '#2b2b43');
    }}

    svg.append('g').attr('class', 'layer-content');
    svg.append('g').attr('class', 'layer-price-lines');

    return {{ svg, x: xScale, y: yScale, w, h, margin, containerId, withYAxis }};
}}

function initAllPanels() {{
    panels.hist = initPanel('hist-panel', {{ top: 60, right: 12, bottom: 20, left: 50 }}, true);
    panels.fvg = initPanel('fvg-panel', {{ top: 60, right: 4, bottom: 20, left: 4 }}, false);
    panels.ob = initPanel('ob-panel', {{ top: 60, right: 4, bottom: 20, left: 4 }}, false);
    panels.swing = initPanel('swing-panel', {{ top: 60, right: 4, bottom: 20, left: 4 }}, false);
}}

function drawPriceLines(p, hist, showLabels) {{
    if (!p) return;
    const g = p.svg.select('.layer-price-lines');
    g.selectAll('*').remove();

    const priceY = p.y(hist.currentPrice);
    g.append('line')
        .attr('x1', 0).attr('x2', p.w)
        .attr('y1', priceY).attr('y2', priceY)
        .attr('stroke', '#ffffff').attr('stroke-width', 2);
    if (showLabels) {{
        g.append('text')
            .attr('x', p.w - 2).attr('y', priceY - 4)
            .attr('text-anchor', 'end').attr('fill', '#ffffff')
            .attr('font-size', '10px').attr('font-weight', 'bold')
            .text('$' + hist.currentPrice.toFixed(2));
    }}

    if (hist.currentHigh != null) {{
        const highY = p.y(hist.currentHigh);
        g.append('line')
            .attr('x1', 0).attr('x2', p.w)
            .attr('y1', highY).attr('y2', highY)
            .attr('stroke', '#ffffff').attr('stroke-width', 1)
            .attr('stroke-dasharray', '4,3');
        if (showLabels) {{
            g.append('text')
                .attr('x', p.w - 2).attr('y', highY - 4)
                .attr('text-anchor', 'end').attr('fill', '#ffffff')
                .attr('font-size', '9px').text('H ' + hist.currentHigh.toFixed(2));
        }}
    }}
    if (hist.currentLow != null) {{
        const lowY = p.y(hist.currentLow);
        g.append('line')
            .attr('x1', 0).attr('x2', p.w)
            .attr('y1', lowY).attr('y2', lowY)
            .attr('stroke', '#ffffff').attr('stroke-width', 1)
            .attr('stroke-dasharray', '4,3');
        if (showLabels) {{
            g.append('text')
                .attr('x', p.w - 2).attr('y', lowY + 11)
                .attr('text-anchor', 'end').attr('fill', '#ffffff')
                .attr('font-size', '9px').text('L ' + hist.currentLow.toFixed(2));
        }}
    }}
}}

function updateYDomain(hist) {{
    Object.values(panels).forEach(p => {{
        if (!p) return;
        p.y.domain(hist.priceRange);
        if (p.withYAxis) {{
            p.svg.select('.y-axis')
                .call(d3.axisLeft(p.y).ticks(10).tickFormat(d3.format('.1f')))
                .selectAll('text').style('fill', '#787b86').style('font-size', '10px');
            p.svg.selectAll('.y-axis path, .y-axis line').style('stroke', '#2b2b43');
        }}
    }});
}}

function updateHistogram(frame) {{
    const p = panels.hist;
    if (!p) return;
    const hist = frame.histogram;

    updateYDomain(hist);

    // Price distribution bars
    const g = p.svg.select('.layer-content');
    g.selectAll('*').remove();
    const bins = hist.binCenters;
    const nBins = bins.length;
    const binH = p.h / nBins;
    let maxCount = d3.max(hist.priceCounts) || 1;
    p.x.domain([0, maxCount]);

    hist.priceCounts.forEach((c, i) => {{
        if (c <= 0) return;
        g.append('rect')
            .attr('y', p.y(bins[i]) - binH / 2)
            .attr('x', 0)
            .attr('height', Math.max(binH - 0.5, 1))
            .attr('width', Math.max(p.x(c), 0))
            .attr('fill', '#787b86').attr('opacity', 0.25);
    }});

    drawPriceLines(p, hist, true);
}}

function updateFvgPanel(hist) {{
    const p = panels.fvg;
    if (!p) return;
    const g = p.svg.select('.layer-content');
    g.selectAll('*').remove();

    hist.fvgZones.forEach(z => {{
        const y1 = p.y(z.top);
        const y2 = p.y(z.bottom);
        g.append('rect')
            .attr('x', 0).attr('y', Math.min(y1, y2))
            .attr('width', p.w)
            .attr('height', Math.max(Math.abs(y2 - y1), 1))
            .attr('fill', '#ffd700')
            .attr('opacity', z.mitigated ? 0.1 : 0.25);
    }});

    drawPriceLines(p, hist, false);
}}

function updateObPanel(hist) {{
    const p = panels.ob;
    if (!p) return;
    const g = p.svg.select('.layer-content');
    g.selectAll('*').remove();

    hist.obZones.forEach(z => {{
        const y1 = p.y(z.top);
        const y2 = p.y(z.bottom);
        g.append('rect')
            .attr('x', 0).attr('y', Math.min(y1, y2))
            .attr('width', p.w)
            .attr('height', Math.max(Math.abs(y2 - y1), 1))
            .attr('fill', '#9c27b0')
            .attr('opacity', z.mitigated ? 0.1 : 0.25);
    }});

    drawPriceLines(p, hist, false);
}}

function updateSwingPanel(hist) {{
    const p = panels.swing;
    if (!p) return;
    const g = p.svg.select('.layer-content');
    g.selectAll('*').remove();

    hist.inflexionLevels.forEach(z => {{
        const y = p.y(z.level);
        g.append('line')
            .attr('x1', 0).attr('x2', p.w)
            .attr('y1', y).attr('y2', y)
            .attr('stroke', z.mitigated ? '#f23645' : '#00bcd4')
            .attr('stroke-width', z.mitigated ? 1 : 1.5)
            .attr('opacity', z.mitigated ? 0.5 : 1);
    }});

    drawPriceLines(p, hist, false);
}}

function resizePanels() {{
    initAllPanels();
    if (allFrames.length > 0) {{
        const f = allFrames[currentIdx];
        updateHistogram(f);
        updateFvgPanel(f.histogram);
        updateObPanel(f.histogram);
        updateSwingPanel(f.histogram);
    }}
}}

// ── Frame navigation ──
const slider = document.getElementById('slider');
slider.max = allFrames.length - 1;
slider.addEventListener('input', () => showFrame(parseInt(slider.value)));

function showFrame(n) {{
    // Save visible range before changing frame
    if (currentIdx !== n) {{
        try {{ savedRange = chart.timeScale().getVisibleLogicalRange(); }} catch(e) {{ savedRange = null; }}
    }}
    currentIdx = Math.max(0, Math.min(n, allFrames.length - 1));
    const f = allFrames[currentIdx];

    // Candlestick
    candleSeries.setData(f.chart.candleData);
    candleSeries.setMarkers(f.chart.inflexionMarkers);

    // Lines
    clearLines();
    drawLines(f.chart.inflexionLines);
    drawLines(f.chart.bosLines);

    // Rectangles
    drawRects(f.chart.fvgZones, 'fvg', 'rgba(255,235,59,0.2)', '#ffeb3b');
    drawRects(f.chart.obZones, 'ob', '', '');

    // Restore visible range or fit
    if (savedRange) {{
        chart.timeScale().setVisibleLogicalRange(savedRange);
    }} else {{
        chart.timeScale().fitContent();
    }}

    // Panels
    updateHistogram(f);
    updateFvgPanel(f.histogram);
    updateObPanel(f.histogram);
    updateSwingPanel(f.histogram);

    // Controls
    slider.value = currentIdx;
    document.getElementById('frame-info').textContent =
        `Frame ${{currentIdx + 1}} / ${{allFrames.length}}  |  ${{f.chart.candleData.length}} candles`;
    document.getElementById('prev-btn').disabled = currentIdx === 0;
    document.getElementById('next-btn').disabled = currentIdx === allFrames.length - 1;
}}

function nextFrame() {{ showFrame(currentIdx + 1); }}
function prevFrame() {{ showFrame(currentIdx - 1); }}

function togglePlay() {{
    playing = !playing;
    document.getElementById('play-btn').textContent = playing ? '\\u23F8 Pause' : '\\u25B6 Play';
    if (playing) {{
        playTimer = setInterval(() => {{
            if (currentIdx >= allFrames.length - 1) {{ togglePlay(); return; }}
            nextFrame();
        }}, 150);
    }} else {{
        clearInterval(playTimer);
    }}
}}

document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowRight') nextFrame();
    else if (e.key === 'ArrowLeft') prevFrame();
    else if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
    else if (e.key === 'Home') showFrame(0);
    else if (e.key === 'End') showFrame(allFrames.length - 1);
}});

// Redraw overlays on zoom/pan
chart.timeScale().subscribeVisibleLogicalRangeChange(() => {{
    const f = allFrames[currentIdx];
    drawRects(f.chart.fvgZones, 'fvg', 'rgba(255,235,59,0.2)', '#ffeb3b');
    drawRects(f.chart.obZones, 'ob', '', '');
}});

let redrawTimer = null, isMouseDown = false;
function throttledRedraw() {{
    if (redrawTimer) return;
    redrawTimer = requestAnimationFrame(() => {{
        const f = allFrames[currentIdx];
        drawRects(f.chart.fvgZones, 'fvg', 'rgba(255,235,59,0.2)', '#ffeb3b');
        drawRects(f.chart.obZones, 'ob', '', '');
        redrawTimer = null;
    }});
}}
chartContainer.addEventListener('wheel', throttledRedraw);
chartContainer.addEventListener('mousedown', () => {{ isMouseDown = true; }});
chartContainer.addEventListener('mousemove', () => {{ if (isMouseDown) throttledRedraw(); }});
chartContainer.addEventListener('mouseup', () => {{ isMouseDown = false; throttledRedraw(); }});
chartContainer.addEventListener('mouseleave', () => {{ isMouseDown = false; }});

// Resize
const ro = new ResizeObserver(() => {{
    chart.applyOptions({{ width: chartContainer.clientWidth, height: chartContainer.clientHeight }});
    throttledRedraw();
    resizePanels();
}});
ro.observe(document.getElementById('chart-panel'));

// ── Draggable dividers ──
const minWidths = {{ 'chart-panel': 200, 'hist-panel': 80, 'fvg-panel': 30, 'ob-panel': 30, 'swing-panel': 30 }};
document.querySelectorAll('.divider').forEach(div => {{
    div.addEventListener('mousedown', e => {{
        e.preventDefault();
        const leftId = div.dataset.left;
        const rightId = div.dataset.right;
        const leftEl = document.getElementById(leftId);
        const rightEl = document.getElementById(rightId);
        const startX = e.clientX;
        const startLeftW = leftEl.getBoundingClientRect().width;
        const startRightW = rightEl.getBoundingClientRect().width;

        function onMove(ev) {{
            const dx = ev.clientX - startX;
            let newLeft = startLeftW + dx;
            let newRight = startRightW - dx;
            const minL = minWidths[leftId] || 30;
            const minR = minWidths[rightId] || 30;
            if (newLeft < minL) {{ newRight -= (minL - newLeft); newLeft = minL; }}
            if (newRight < minR) {{ newLeft -= (minR - newRight); newRight = minR; }}
            if (newLeft < minL || newRight < minR) return;
            if (leftId === 'chart-panel') {{
                leftEl.style.flex = '0 0 ' + newLeft + 'px';
            }} else {{
                leftEl.style.width = newLeft + 'px';
            }}
            rightEl.style.width = newRight + 'px';
        }}
        function onUp() {{
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            if (leftId === 'chart-panel') {{
                chart.applyOptions({{ width: leftEl.clientWidth, height: leftEl.clientHeight }});
                throttledRedraw();
            }}
            resizePanels();
        }}
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    }});
}});

// Init
initAllPanels();
if (allFrames.length > 0) showFrame(0);
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description='Left/Right Split-View: Candlestick + Zone Histogram')
    parser.add_argument('--symbol', default='TSLA', help='Stock symbol (default: TSLA)')
    parser.add_argument('--timeframe', default='15min', help='Timeframe (default: 15min)')
    parser.add_argument('--bins', type=int, default=80, help='Number of histogram bins (default: 80)')
    parser.add_argument('--swing-length', type=int, default=50, help='Swing length for OB detection (default: 50)')
    parser.add_argument('--step', type=int, default=1, help='Candles to advance per frame (default: 5)')
    args = parser.parse_args()

    # Load data
    loader = DataLoader(symbol=args.symbol)
    ohlc = loader.get_data(args.timeframe)
    if 'time' in ohlc.columns and not isinstance(ohlc.index, pd.DatetimeIndex):
        ohlc = ohlc.set_index('time')
    print(f"Loaded {len(ohlc)} candles for {args.symbol} ({args.timeframe})")

    # Use first quarter only
    quarter = len(ohlc) // 4
    ohlc = ohlc.iloc[:quarter].copy()
    print(f"Using first quarter: {len(ohlc)} candles")

    # Build bar-by-bar frames (every `step` bars)
    total = len(ohlc)
    frame_indices = list(range(0, total, args.step))
    if frame_indices[-1] != total - 1:
        frame_indices.append(total - 1)

    print(f"Building {len(frame_indices)} frames (step={args.step})...")

    all_frames = []
    for count, i in enumerate(frame_indices):
        end = i + 1  # slice 0..i inclusive
        ohlc_slice = ohlc.iloc[:end]

        # Chart data + raw indicators (recomputed per slice)
        chart_data, raw_indicators = build_chart_frame(ohlc_slice)

        # Histogram zone data (from same indicators as chart)
        hist_zones = collect_histogram_zones(
            raw_indicators['fvg'], raw_indicators['ob'], raw_indicators['inflexions']
        )

        # Price distribution histogram (binned from candle ranges)
        price_min = float(ohlc_slice['low'].min())
        price_max = float(ohlc_slice['high'].max())
        margin = (price_max - price_min) * 0.02
        bin_edges = np.linspace(price_min - margin, price_max + margin, args.bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        price_levels = generate_price_levels(ohlc_slice)
        price_counts, _ = np.histogram(price_levels, bins=bin_edges)

        current_price = float(ohlc.iloc[i]['close'])

        hist_zones['binCenters'] = bin_centers.tolist()
        hist_zones['priceCounts'] = price_counts.tolist()
        hist_zones['currentPrice'] = current_price
        hist_zones['currentHigh'] = float(ohlc.iloc[i]['high'])
        hist_zones['currentLow'] = float(ohlc.iloc[i]['low'])
        hist_zones['priceRange'] = [price_min - margin, price_max + margin]

        all_frames.append({
            'chart': chart_data,
            'histogram': hist_zones,
        })

        if (count + 1) % 50 == 0:
            print(f"  Frame {count + 1}/{len(frame_indices)} (bar {i})")

    print(f"Serializing {len(all_frames)} frames to JSON...")
    all_frames_json = json.dumps(all_frames, cls=NumpyEncoder)

    html = build_html(args.symbol, args.timeframe, all_frames_json)

    output_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 'charts', 'price_distribution_indicators.html'
    ))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = len(html) / (1024 * 1024)
    print(f"Saved to {output_path} ({size_mb:.1f} MB)")
    webbrowser.open(f'file://{output_path}')


if __name__ == '__main__':
    main()
