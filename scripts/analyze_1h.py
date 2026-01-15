"""
1H Candle-by-Candle Walkthrough

Generates a frame for every single 1H candle showing the progressive
formation of SMC indicators.

Usage:
    python3 scripts/analyze_1h.py

Output:
    - results/1h_walkthrough/frames/frame_0000.html → frame_XXXX.html
    - results/1h_walkthrough/index.html
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import TSLADataLoader
from src.indicators.smc_custom import smc_custom
from src.indicators.smc import smc


class OneHourAnalyzer:
    """Generates candle-by-candle walkthrough for 1H timeframe."""

    def __init__(self, output_dir=None):
        if output_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            self.output_dir = project_root / 'results' / '1h_walkthrough'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir = self.output_dir / 'frames'
        self.frame_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            'bullish_candle': '#77dd76',
            'bearish_candle': '#ff6962',
            'background': 'rgba(12, 14, 18, 1)',
            'fvg': 'yellow',
            'ob': 'Purple',
        }

    def add_FVG(self, fig, df, fvg_data):
        """Add Fair Value Gap visualization."""
        for i in range(len(fvg_data["FVG"])):
            if not np.isnan(fvg_data["FVG"][i]):
                x1 = int(
                    fvg_data["MitigatedIndex"][i]
                    if fvg_data["MitigatedIndex"][i] != 0
                    else len(df) - 1
                )
                fig.add_shape(
                    type="rect",
                    x0=df.index[i],
                    y0=fvg_data["Top"][i],
                    x1=df.index[x1],
                    y1=fvg_data["Bottom"][i],
                    line=dict(width=0),
                    fillcolor=self.colors['fvg'],
                    opacity=0.2,
                )
                mid_x = round((i + x1) / 2)
                mid_y = (fvg_data["Top"][i] + fvg_data["Bottom"][i]) / 2
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[mid_x]],
                        y=[mid_y],
                        mode="text",
                        text="FVG",
                        textposition="middle center",
                        textfont=dict(color='rgba(255, 255, 255, 0.4)', size=8),
                        showlegend=False,
                    )
                )
        return fig

    def add_inflexions(self, fig, df, inflexions_data):
        """Add inflexion points visualization."""
        for i in range(len(inflexions_data)):
            if not np.isnan(inflexions_data["InflexionType"].iloc[i]):
                inflx_type = inflexions_data["InflexionType"].iloc[i]
                level = inflexions_data["Level"].iloc[i]
                respected = inflexions_data["Respected"].iloc[i]

                # Color based on type and respect status
                if inflx_type == 1:  # Concave (peak/resistance)
                    if respected is True:
                        color = 'rgba(0, 255, 0, 0.8)'  # Green - LIQUIDITY SWEEP
                        symbol = 'star'
                        size = 20
                    elif respected is False:
                        color = 'rgba(255, 0, 0, 0.3)'
                        symbol = 'x'
                        size = 10
                    else:
                        color = 'rgba(255, 255, 0, 0.6)'
                        symbol = 'triangle-down'
                        size = 10
                else:  # Convex (valley/support)
                    if respected is True:
                        color = 'rgba(0, 128, 255, 0.8)'  # Blue - LIQUIDITY SWEEP
                        symbol = 'star'
                        size = 20
                    elif respected is False:
                        color = 'rgba(255, 165, 0, 0.3)'
                        symbol = 'x'
                        size = 10
                    else:
                        color = 'rgba(255, 255, 0, 0.6)'
                        symbol = 'triangle-up'
                        size = 10

                # Add marker
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i]],
                        y=[level],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            size=size,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>Inflexion</b><br>Type: {'Concave' if inflx_type == 1 else 'Convex'}<br>Level: {level:.2f}<br>Respected: {respected}<extra></extra>"
                    )
                )

                # Draw level line
                end_idx = inflexions_data["StatusIndex"].iloc[i] if inflexions_data["StatusIndex"].iloc[i] != 0 else len(df) - 1
                line_width = 3 if respected is True else 1
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i], df.index[int(end_idx)]],
                        y=[level, level],
                        mode='lines',
                        line=dict(color=color, width=line_width, dash='solid' if respected is True else 'dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

        return fig

    def add_bos(self, fig, df, bos_data):
        """Add BOS visualization."""
        for i in range(len(bos_data)):
            if not np.isnan(bos_data["BOS"].iloc[i]):
                bos_type = bos_data["BOS"].iloc[i]
                level = bos_data["Level"].iloc[i]
                broken_idx = int(bos_data["BrokenIndex"].iloc[i])

                color = 'rgba(0, 255, 0, 0.6)' if bos_type == 1 else 'rgba(255, 0, 0, 0.6)'

                # Draw line
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i], df.index[broken_idx]],
                        y=[level, level],
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

                # Add label
                mid_x = round((i + broken_idx) / 2)
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[mid_x]],
                        y=[level],
                        mode='text',
                        text='BOS ↑' if bos_type == 1 else 'BOS ↓',
                        textposition='top center' if bos_type == 1 else 'bottom center',
                        textfont=dict(color=color, size=10, family='Arial Black'),
                        showlegend=False,
                    )
                )

        return fig

    def generate_frame(self, df_slice, candle_idx, total_candles):
        """Generate a single frame for candle_idx."""

        # Calculate indicators on the slice
        inflexions = smc_custom.inflexion_points(df_slice)
        bos = smc_custom.bos(df_slice, inflexions, close_break=True)
        fvg = smc.fvg(df_slice, join_consecutive=True)

        # Count statistics
        total_inflexions = inflexions['InflexionType'].notna().sum()
        liquidity_sweeps = ((inflexions['Respected'] == True) & inflexions['InflexionType'].notna()).sum()
        total_bos = bos['BOS'].notna().sum()
        total_fvg = fvg['FVG'].notna().sum()

        # Create chart
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_slice.index,
                    open=df_slice['open'],
                    high=df_slice['high'],
                    low=df_slice['low'],
                    close=df_slice['close'],
                    increasing_line_color=self.colors['bullish_candle'],
                    decreasing_line_color=self.colors['bearish_candle'],
                    name='TSLA 1H'
                )
            ]
        )

        # Add indicators
        fig = self.add_inflexions(fig, df_slice, inflexions)
        fig = self.add_bos(fig, df_slice, bos)
        fig = self.add_FVG(fig, df_slice, fvg)

        # Add vertical line at current candle
        fig.add_vline(
            x=df_slice.index[-1],
            line_dash="dash",
            line_color='white',
            line_width=2,
        )

        # Layout
        fig.update_layout(
            title=f"1H Analysis - Candle {candle_idx + 1}/{total_candles} - {df_slice.index[-1].strftime('%Y-%m-%d %H:%M')}",
            xaxis_rangeslider_visible=False,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor=self.colors['background'],
            font=dict(color='white', size=11),
            height=800,
            margin=dict(r=250, l=80, t=80, b=50)
        )

        # Status panel
        status_text = f"""
<b style="color: #00ff00">1H TIMEFRAME ANALYSIS</b><br>
<br>
<b>Progress:</b> Candle {candle_idx + 1}/{total_candles}<br>
<br>
<b>Indicators Found:</b><br>
• Inflexions: {total_inflexions}<br>
• <span style="color: #00ff00">Liquidity Sweeps: {liquidity_sweeps}</span><br>
• BOS Events: {total_bos}<br>
• FVG Zones: {total_fvg}<br>
<br>
<b>Legend:</b><br>
<span style="color: #00ff00">★</span> Liquidity Sweep (Respected)<br>
<span style="color: #ffff00">▼</span> Inflexion (Pending)<br>
<span style="color: #ff0000">✕</span> Inflexion (Disrespected)<br>
        """

        fig.add_annotation(
            text=status_text,
            xref='paper',
            yref='paper',
            x=1.15,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            showarrow=False,
            bgcolor='rgba(0, 0, 0, 0.9)',
            bordercolor='#00ff00',
            borderwidth=3,
            borderpad=12,
            font=dict(size=10, color='white', family='monospace'),
            align='left'
        )

        # Save frame
        filename = f"frame_{candle_idx:04d}.html"
        output_path = self.frame_dir / filename
        fig.write_html(str(output_path))

        return liquidity_sweeps

    def create_index(self, total_frames):
        """Create index HTML with navigation."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>1H Candle-by-Candle Walkthrough</title>
    <style>
        body {{
            background-color: #0c0e12;
            color: white;
            font-family: Arial, sans-serif;
            padding: 20px;
            margin: 0;
        }}
        h1 {{
            color: #00ff00;
            text-align: center;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        button {{
            background-color: #1a1d24;
            color: white;
            border: 2px solid #00ff00;
            padding: 10px 20px;
            margin: 0 5px;
            cursor: pointer;
            font-size: 16px;
        }}
        button:hover {{
            background-color: #00ff00;
            color: #000;
        }}
        .frame-container {{
            margin: 20px auto;
            max-width: 1600px;
        }}
        iframe {{
            width: 100%;
            height: 850px;
            border: 2px solid #00ff00;
        }}
        .frame-info {{
            text-align: center;
            margin: 10px 0;
            font-size: 18px;
        }}
        #goto-input {{
            width: 80px;
            padding: 8px;
            background-color: #1a1d24;
            border: 2px solid #00ff00;
            color: white;
            border-radius: 4px;
            font-size: 14px;
            text-align: center;
        }}
        #goto-input:focus {{
            outline: none;
            background-color: #2a2d34;
        }}
    </style>
    <script>
        let currentFrame = 0;
        const totalFrames = {total_frames};

        function showFrame(n) {{
            currentFrame = Math.max(0, Math.min(n, totalFrames - 1));
            const frameFile = 'frames/frame_' + String(currentFrame).padStart(4, '0') + '.html';
            document.getElementById('frame-viewer').src = frameFile;
            document.getElementById('frame-number').textContent = 'Candle ' + (currentFrame + 1) + ' of ' + totalFrames;

            document.getElementById('prev-btn').disabled = (currentFrame === 0);
            document.getElementById('next-btn').disabled = (currentFrame === totalFrames - 1);
        }}

        function nextFrame() {{ showFrame(currentFrame + 1); }}
        function prevFrame() {{ showFrame(currentFrame - 1); }}
        function firstFrame() {{ showFrame(0); }}
        function lastFrame() {{ showFrame(totalFrames - 1); }}
        function gotoFrame() {{
            const input = document.getElementById('goto-input');
            const frameNum = parseInt(input.value);
            if (frameNum >= 1 && frameNum <= totalFrames) {{
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
        document.addEventListener('DOMContentLoaded', function() {{
            document.getElementById('goto-input').addEventListener('keydown', function(event) {{
                if (event.key === 'Enter') {{
                    event.stopPropagation();
                    gotoFrame();
                }}
            }});
        }});
    </script>
</head>
<body onload="showFrame(0)">
    <h1>1H Candle-by-Candle Walkthrough</h1>
    <p style="text-align: center; color: #aaa;">Watch SMC indicators form progressively on the 1H timeframe</p>

    <div class="controls">
        <button id="first-btn" onclick="firstFrame()">⏮ First</button>
        <button id="prev-btn" onclick="prevFrame()">◀ Previous</button>
        <span class="frame-info" id="frame-number">Candle 1 of {total_frames}</span>
        <input type="number" id="goto-input" min="1" max="{total_frames}" placeholder="Go to...">
        <button id="goto-btn" onclick="gotoFrame()">Go</button>
        <button id="next-btn" onclick="nextFrame()">Next ▶</button>
        <button id="last-btn" onclick="lastFrame()">Last ⏭</button>
    </div>

    <div class="frame-container">
        <iframe id="frame-viewer" src="frames/frame_0000.html"></iframe>
    </div>

    <div style="text-align: center; color: #888; margin-top: 20px;">
        Use arrow keys ← → or buttons to navigate | Home/End for first/last candle
    </div>
</body>
</html>
        """

        output_path = self.output_dir / "index.html"
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"\n✅ Index created: {output_path}")
        print(f"   Navigate through {total_frames} candles with arrow keys")

    def run(self, df_1h):
        """Generate walkthrough for all candles."""
        total_candles = len(df_1h)

        print(f"\n{'='*80}")
        print(f"1H CANDLE-BY-CANDLE WALKTHROUGH")
        print(f"{'='*80}\n")
        print(f"Total candles: {total_candles}")
        print(f"Generating {total_candles} frames...\n")

        total_sweeps = 0

        for i in range(total_candles):
            df_slice = df_1h.iloc[:i+1]
            sweeps = self.generate_frame(df_slice, i, total_candles)

            if (i + 1) % 10 == 0:
                print(f"  Generated frame {i+1}/{total_candles}")

            if sweeps > total_sweeps:
                print(f"  🎯 Liquidity sweep detected at candle {i+1}!")
                total_sweeps = sweeps

        print(f"\n✅ Generated {total_candles} frames")
        print(f"🎯 Total liquidity sweeps found: {total_sweeps}")

        self.create_index(total_candles)


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("1H TIMEFRAME ANALYSIS")
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
    analyzer = OneHourAnalyzer()
    analyzer.run(df_1h)

    print("\n" + "="*80)
    print("✅ WALKTHROUGH COMPLETE!")
    print("="*80)
    print(f"\n📍 Open: {analyzer.output_dir / 'index.html'}")
    print("💡 Use arrow keys to navigate through candles!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
