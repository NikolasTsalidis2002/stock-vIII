"""
Frame-by-Frame Visual Analyzer for Partial Setups

Simple approach:
1. Generate PNG frame for each candle/moment
2. Combine frames into GIF or video
3. No complex animations - just static images

Matches the pattern from your other project.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
from datetime import timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy import PartialSetup, MultiTimeframeStrategy
from smartmoneyconcepts.smc_custom import smc_custom
from smartmoneyconcepts.smc import smc


class FrameByFrameAnalyzer:
    """
    Generates frame-by-frame walkthroughs of partial setups.

    Creates static PNG images for each moment, then combines into GIF/video.
    """

    def __init__(self, strategy: MultiTimeframeStrategy, output_dir: Optional[str] = None):
        """
        Initialize frame-by-frame analyzer.

        Args:
            strategy: MultiTimeframeStrategy instance
            output_dir: Directory to save frames
        """
        self.strategy = strategy

        if output_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            self.output_dir = project_root / 'results' / 'strategy_examples'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Frame counter per setup
        self.frame_id = 0

        # Color scheme
        self.colors = {
            'bullish_candle': '#77dd76',
            'bearish_candle': '#ff6962',
            'background': 'rgba(12, 14, 18, 1)',
            'fvg': 'yellow',
            'ob': 'Purple',
            'bos': 'rgba(255, 165, 0, 0.2)',
            'choch': 'rgba(0, 0, 255, 0.2)',
            'liquidity': 'rgba(255, 165, 0, 0.2)',
            'liquidity_swept': 'rgba(255, 0, 0, 0.2)',
            'swing_high': 'rgba(255, 0, 0, 0.2)',
            'swing_low': 'rgba(0, 128, 0, 0.2)',
            'previous_hl': 'rgba(255, 255, 255, 0.2)',
            'sweep': '#FFA500',
            'event_b': '#00FFFF',
            'validation': '#FFFF00',
            'confirmation': '#00FF00',
        }

    def add_FVG(self, fig, df, fvg_data, row, highlight_indices=None):
        """Add Fair Value Gap visualization to a specific subplot row."""
        for i in range(len(fvg_data["FVG"])):
            if not np.isnan(fvg_data["FVG"][i]):
                x1 = int(
                    fvg_data["MitigatedIndex"][i]
                    if fvg_data["MitigatedIndex"][i] != 0
                    else len(df) - 1
                )

                # Highlight if this is a trigger
                opacity = 0.5 if (highlight_indices and i in highlight_indices) else 0.2
                line_width = 2 if (highlight_indices and i in highlight_indices) else 0

                fig.add_shape(
                    type="rect",
                    x0=df.index[i],
                    y0=fvg_data["Top"][i],
                    x1=df.index[x1],
                    y1=fvg_data["Bottom"][i],
                    line=dict(width=line_width, color=self.colors['fvg']),
                    fillcolor=self.colors['fvg'],
                    opacity=opacity,
                    row=row, col=1
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
                    ),
                    row=row, col=1
                )
        return fig

    def add_custom_inflexions(self, fig, df, inflexions_data, row, highlight_indices=None):
        """Add Custom Inflexion Points visualization to a specific subplot row."""
        for i in range(len(inflexions_data)):
            if not np.isnan(inflexions_data["InflexionType"].iloc[i]):
                inflx_type = inflexions_data["InflexionType"].iloc[i]
                level = inflexions_data["Level"].iloc[i]
                respected = inflexions_data["Respected"].iloc[i]

                # Determine color based on type and respect status
                if inflx_type == 1:  # Concave (peak/resistance)
                    if respected is True:
                        color = 'rgba(0, 255, 0, 0.6)'  # Green - respected resistance
                        symbol = 'triangle-down'
                    elif respected is False:
                        color = 'rgba(255, 0, 0, 0.3)'  # Red - disrespected resistance
                        symbol = 'x'
                    else:  # None - pending
                        color = 'rgba(255, 255, 0, 0.6)'  # Yellow - pending
                        symbol = 'triangle-down'
                else:  # Convex (valley/support)
                    if respected is True:
                        color = 'rgba(0, 128, 255, 0.6)'  # Blue - respected support
                        symbol = 'triangle-up'
                    elif respected is False:
                        color = 'rgba(255, 165, 0, 0.3)'  # Orange - disrespected support
                        symbol = 'x'
                    else:  # None - pending
                        color = 'rgba(255, 255, 0, 0.6)'  # Yellow - pending
                        symbol = 'triangle-up'

                # Highlight if this is a trigger (liquidity sweep)
                marker_size = 15 if (highlight_indices and i in highlight_indices) else 10
                line_width = 3 if (highlight_indices and i in highlight_indices) else 1

                # Add marker at inflexion point
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i]],
                        y=[level],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            size=marker_size,
                            color=color,
                            line=dict(width=line_width, color='white')
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>Inflexion</b><br>Type: {'Concave' if inflx_type == 1 else 'Convex'}<br>Level: {level:.2f}<br>Status: {respected}<extra></extra>"
                    ),
                    row=row, col=1
                )

                # Draw horizontal line to show level
                end_idx = inflexions_data["StatusIndex"].iloc[i] if inflexions_data["StatusIndex"].iloc[i] != 0 else len(df) - 1
                line_dash = 'solid' if (highlight_indices and i in highlight_indices) else 'dot'
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i], df.index[int(end_idx)]],
                        y=[level, level],
                        mode='lines',
                        line=dict(color=color, width=2 if (highlight_indices and i in highlight_indices) else 1, dash=line_dash),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=1
                )

        return fig

    def add_custom_bos(self, fig, df, bos_data, row, highlight_indices=None):
        """Add Custom Break of Structure visualization to a specific subplot row."""
        # Add BOS markers and levels
        for i in range(len(bos_data)):
            if not np.isnan(bos_data["BOS"].iloc[i]):
                bos_type = bos_data["BOS"].iloc[i]
                level = bos_data["Level"].iloc[i]
                broken_idx = int(bos_data["BrokenIndex"].iloc[i])

                # Color: green for bullish BOS, red for bearish BOS
                color = 'rgba(0, 255, 0, 0.6)' if bos_type == 1 else 'rgba(255, 0, 0, 0.6)'

                # Highlight if this is a trigger
                line_width = 4 if (highlight_indices and i in highlight_indices) else 2
                font_size = 12 if (highlight_indices and i in highlight_indices) else 10

                # Draw line from inflexion to break point
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i], df.index[broken_idx]],
                        y=[level, level],
                        mode='lines',
                        line=dict(color=color, width=line_width),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=1
                )

                # Add BOS label at midpoint
                mid_x = round((i + broken_idx) / 2)
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[mid_x]],
                        y=[level],
                        mode='text',
                        text='BOS ↑' if bos_type == 1 else 'BOS ↓',
                        textposition='top center' if bos_type == 1 else 'bottom center',
                        textfont=dict(color=color, size=font_size, family='Arial Black'),
                        showlegend=False,
                        hovertemplate=f"<b>BOS</b><br>Type: {'Bullish' if bos_type == 1 else 'Bearish'}<br>Level: {level:.2f}<extra></extra>"
                    ),
                    row=row, col=1
                )

        return fig

    def generate_frame(
        self,
        partial: PartialSetup,
        current_time: pd.Timestamp,
        frame_dir: Path,
        status_message: str,
        conditions_met: List[bool]
    ):
        """
        Generate a single frame showing state at current_time.

        Args:
            partial: PartialSetup object
            current_time: Timestamp for this frame
            frame_dir: Directory to save frame
            status_message: Status text to display
            conditions_met: [bool, bool, bool, bool] for 4 conditions
        """
        # Create 3-panel chart
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=('1H Timeframe', '5M Timeframe', '1M Timeframe'),
            vertical_spacing=0.08,
            row_heights=[0.33, 0.33, 0.34]
        )

        # Get data up to current_time for each timeframe
        df_1h = self.strategy.df_1h[self.strategy.df_1h.index <= current_time]
        df_5m = self.strategy.df_5m[self.strategy.df_5m.index <= current_time]
        df_1m = self.strategy.df_1m[self.strategy.df_1m.index <= current_time]

        # Window around relevant area - following FRAME_VISUALIZATION_LOGIC.md
        # 1H: Show ALL overlap data (no windowing)
        df_1h_plot = df_1h

        # 5M: Start from 1H sweep time (like user's start_ts = one_hour_entry.obj.time)
        # Extend to current time (we already filtered to current_time above)
        if len(df_5m) > 0:
            start_ts_5m = partial.timestamp_1h_sweep  # Start from 1H trigger
            df_5m_plot = df_5m[df_5m.index >= start_ts_5m]
            if len(df_5m_plot) == 0:
                df_5m_plot = df_5m  # Fallback
        else:
            df_5m_plot = df_5m

        # 1M: Start from 5M Event B time (like user's start_ts_1m = self.eb.active("5mI")[0].obj.time)
        if len(df_1m) > 0 and partial.timestamp_5m_event_b:
            start_ts_1m = partial.timestamp_5m_event_b  # Start from 5M Event B trigger
            df_1m_plot = df_1m[df_1m.index >= start_ts_1m]
            if len(df_1m_plot) == 0:
                df_1m_plot = df_1m  # Fallback
        elif len(df_1m) > 0:
            # Before Event B exists, start from 1H sweep
            start_ts_1m = partial.timestamp_1h_sweep
            df_1m_plot = df_1m[df_1m.index >= start_ts_1m]
            if len(df_1m_plot) == 0:
                df_1m_plot = df_1m  # Fallback
        else:
            df_1m_plot = df_1m

        # Add candlesticks for each timeframe
        if len(df_1h_plot) > 0:
            fig.add_trace(
                go.Candlestick(
                    x=df_1h_plot.index,
                    open=df_1h_plot['open'],
                    high=df_1h_plot['high'],
                    low=df_1h_plot['low'],
                    close=df_1h_plot['close'],
                    increasing_line_color=self.colors['bullish_candle'],
                    decreasing_line_color=self.colors['bearish_candle'],
                    showlegend=False
                ),
                row=1, col=1
            )

        if len(df_5m_plot) > 0:
            fig.add_trace(
                go.Candlestick(
                    x=df_5m_plot.index,
                    open=df_5m_plot['open'],
                    high=df_5m_plot['high'],
                    low=df_5m_plot['low'],
                    close=df_5m_plot['close'],
                    increasing_line_color=self.colors['bullish_candle'],
                    decreasing_line_color=self.colors['bearish_candle'],
                    showlegend=False
                ),
                row=2, col=1
            )

        if len(df_1m_plot) > 0:
            fig.add_trace(
                go.Candlestick(
                    x=df_1m_plot.index,
                    open=df_1m_plot['open'],
                    high=df_1m_plot['high'],
                    low=df_1m_plot['low'],
                    close=df_1m_plot['close'],
                    increasing_line_color=self.colors['bullish_candle'],
                    decreasing_line_color=self.colors['bearish_candle'],
                    showlegend=False
                ),
                row=3, col=1
            )

        # Calculate and render indicators for each timeframe
        # 1H Indicators - Use FULL data (no windowing)
        if len(df_1h) > 0:
            # Calculate inflexions (liquidity levels) and BOS on FULL 1H data
            inflexions_1h = smc_custom.inflexion_points(df_1h)
            bos_1h = smc_custom.bos(df_1h, inflexions_1h, close_break=True)
            fvg_1h = smc.fvg(df_1h, join_consecutive=True)

            # Determine which inflexion is the liquidity sweep trigger
            sweep_highlight = []
            if current_time >= partial.timestamp_1h_sweep:
                try:
                    sweep_idx = df_1h.index.get_loc(partial.timestamp_1h_sweep)
                    sweep_highlight = [sweep_idx]
                except:
                    pass

            # Render indicators on row 1 (but only show df_1h_plot window)
            fig = self.add_custom_inflexions(fig, df_1h_plot, inflexions_1h, row=1, highlight_indices=sweep_highlight)
            fig = self.add_custom_bos(fig, df_1h_plot, bos_1h, row=1)
            fig = self.add_FVG(fig, df_1h_plot, fvg_1h, row=1)

        # 5M Indicators - Calculate on WINDOWED data (from 1H sweep time)
        if len(df_5m_plot) > 0:
            # Calculate indicators on df_5m_plot (windowed from sweep time)
            inflexions_5m = smc_custom.inflexion_points(df_5m_plot)
            bos_5m = smc_custom.bos(df_5m_plot, inflexions_5m, close_break=True)
            fvg_5m = smc.fvg(df_5m_plot, join_consecutive=True)

            # Determine which BOS/FVG is the Event B trigger
            eventb_bos_highlight = []
            eventb_fvg_highlight = []
            validation_fvg_highlight = []

            if partial.timestamp_5m_event_b and current_time >= partial.timestamp_5m_event_b:
                try:
                    eventb_idx = df_5m_plot.index.get_loc(partial.timestamp_5m_event_b)
                    if 'BOS' in partial.condition_event_b:
                        eventb_bos_highlight = [eventb_idx]
                    elif 'IFVG' in partial.condition_event_b:
                        eventb_fvg_highlight = [eventb_idx]
                except:
                    pass

            if partial.timestamp_5m_validation and current_time >= partial.timestamp_5m_validation:
                try:
                    val_idx = df_5m_plot.index.get_loc(partial.timestamp_5m_validation)
                    if 'FVG' in partial.condition_validation:
                        validation_fvg_highlight = [val_idx]
                except:
                    pass

            # Merge highlights
            fvg_highlights = list(set(eventb_fvg_highlight + validation_fvg_highlight))

            # Render indicators on row 2 (using df_5m_plot)
            fig = self.add_custom_inflexions(fig, df_5m_plot, inflexions_5m, row=2)
            fig = self.add_custom_bos(fig, df_5m_plot, bos_5m, row=2, highlight_indices=eventb_bos_highlight)
            fig = self.add_FVG(fig, df_5m_plot, fvg_5m, row=2, highlight_indices=fvg_highlights)

        # 1M Indicators - Calculate on WINDOWED data (from 5M validation time)
        if len(df_1m_plot) > 0:
            # Calculate indicators on df_1m_plot (windowed from validation time)
            inflexions_1m = smc_custom.inflexion_points(df_1m_plot)
            bos_1m = smc_custom.bos(df_1m_plot, inflexions_1m, close_break=True)
            fvg_1m = smc.fvg(df_1m_plot, join_consecutive=True)

            # Determine if there's a confirmation trigger
            confirm_bos_highlight = []
            if partial.timestamp_1m_confirmation and current_time >= partial.timestamp_1m_confirmation:
                try:
                    confirm_idx = df_1m_plot.index.get_loc(partial.timestamp_1m_confirmation)
                    confirm_bos_highlight = [confirm_idx]
                except:
                    pass

            # Render indicators on row 3 (using df_1m_plot)
            fig = self.add_custom_inflexions(fig, df_1m_plot, inflexions_1m, row=3)
            fig = self.add_custom_bos(fig, df_1m_plot, bos_1m, row=3, highlight_indices=confirm_bos_highlight)
            fig = self.add_FVG(fig, df_1m_plot, fvg_1m, row=3)

        # Add vertical line showing current time on each panel
        for row in range(1, 4):
            fig.add_vline(
                x=current_time,
                line_dash="dash",
                line_color='white',
                line_width=2,
                row=row,
                col=1
            )

        # Add status annotations on each panel
        # 1H Panel annotation
        if current_time >= partial.timestamp_1h_sweep:
            status_1h = f"✓ Liquidity Sweep\n${partial.price_1h_sweep:.2f}\n{partial.timestamp_1h_sweep.strftime('%H:%M')}"
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref='x domain',
                yref='y domain',
                text=status_1h,
                showarrow=False,
                bgcolor='rgba(255, 165, 0, 0.8)',
                bordercolor='white',
                borderwidth=2,
                font=dict(size=10, color='white', family='monospace'),
                align='left',
                xanchor='left',
                yanchor='top',
                row=1, col=1
            )

        # 5M Panel annotations
        annotations_5m = []
        if partial.timestamp_5m_event_b and current_time >= partial.timestamp_5m_event_b:
            annotations_5m.append(f"✓ Event B: {partial.condition_event_b}")
            annotations_5m.append(f"${partial.price_5m_event_b:.2f} @ {partial.timestamp_5m_event_b.strftime('%H:%M')}")

        if partial.timestamp_5m_validation and current_time >= partial.timestamp_5m_validation:
            annotations_5m.append(f"✓ Validation: {partial.condition_validation}")
            annotations_5m.append(f"${partial.price_5m_validation:.2f} @ {partial.timestamp_5m_validation.strftime('%H:%M')}")

        if annotations_5m:
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref='x2 domain',
                yref='y2 domain',
                text='<br>'.join(annotations_5m),
                showarrow=False,
                bgcolor='rgba(0, 255, 255, 0.8)' if len(annotations_5m) >= 2 else 'rgba(0, 255, 255, 0.6)',
                bordercolor='white',
                borderwidth=2,
                font=dict(size=10, color='white', family='monospace'),
                align='left',
                xanchor='left',
                yanchor='top',
                row=2, col=1
            )

        # 1M Panel annotation
        if partial.timestamp_1m_confirmation and current_time >= partial.timestamp_1m_confirmation:
            status_1m = f"✓ Confirmation\n{partial.condition_confirmation}\n${partial.price_1m_confirmation:.2f}"
            bgcolor = 'rgba(0, 255, 0, 0.8)'
        else:
            status_1m = "✗ Waiting for\nBOS/IFVG confirmation"
            bgcolor = 'rgba(255, 0, 0, 0.6)'

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref='x3 domain',
            yref='y3 domain',
            text=status_1m,
            showarrow=False,
            bgcolor=bgcolor,
            bordercolor='white',
            borderwidth=2,
            font=dict(size=10, color='white', family='monospace'),
            align='left',
            xanchor='left',
            yanchor='top',
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title=f"Frame {self.frame_id:04d} - {current_time.strftime('%Y-%m-%d %H:%M')}",
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            xaxis3_rangeslider_visible=False,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor=self.colors['background'],
            font=dict(color='white', size=11),
            height=1000,
            margin=dict(r=300, l=80, t=80, b=50)
        )

        # Add detailed status panel annotation
        # Build detailed condition status
        cond1_text = f"✓ 1H Liquidity Sweep<br>   ${partial.price_1h_sweep:.2f}<br>   {partial.timestamp_1h_sweep.strftime('%Y-%m-%d %H:%M')}" if conditions_met[0] else "☐ 1H Liquidity Sweep"

        if conditions_met[1] and partial.timestamp_5m_event_b:
            cond2_text = f"✓ 5M Event B: {partial.condition_event_b}<br>   ${partial.price_5m_event_b:.2f}<br>   {partial.timestamp_5m_event_b.strftime('%Y-%m-%d %H:%M')}"
        else:
            cond2_text = "☐ 5M Event B (BOS/IFVG)"

        if conditions_met[2] and partial.timestamp_5m_validation:
            cond3_text = f"✓ 5M Validation: {partial.condition_validation}<br>   ${partial.price_5m_validation:.2f}<br>   {partial.timestamp_5m_validation.strftime('%Y-%m-%d %H:%M')}"
        else:
            cond3_text = "☐ 5M Validation (FVG/DZ)"

        if conditions_met[3] and partial.timestamp_1m_confirmation:
            cond4_text = f"✓ 1M Confirmation: {partial.condition_confirmation}<br>   ${partial.price_1m_confirmation:.2f}<br>   {partial.timestamp_1m_confirmation.strftime('%Y-%m-%d %H:%M')}"
        else:
            cond4_text = "☐ 1M Confirmation (BOS/IFVG)"

        progress = sum(conditions_met)
        conditions_text = f"""
<b>TRADE SETUP PROGRESS</b><br>
<b style="color: {'#00ff00' if progress == 4 else '#ffff00' if progress >= 2 else '#ff6962'}">{progress}/4 Conditions Met</b><br>
<br>
<b>Conditions:</b><br>
{cond1_text}<br>
<br>
{cond2_text}<br>
<br>
{cond3_text}<br>
<br>
{cond4_text}<br>
<br>
<b style="color: {'#00ff00' if progress == 4 else '#ff6962'}">Status:</b><br>
{status_message}
        """

        fig.add_annotation(
            text=conditions_text,
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
            font=dict(size=9, color='white', family='monospace')
        )

        # Save as HTML (simpler, no dependencies)
        filename = f"frame_{self.frame_id:04d}.html"
        output_path = frame_dir / filename

        fig.write_html(str(output_path))
        self.frame_id += 1

    def create_walkthrough(self, partial: PartialSetup, setup_num: int):
        """
        Generate all frames and combine into GIF.

        Args:
            partial: PartialSetup object
            setup_num: Setup number
        """
        print(f"\n{'='*80}")
        print(f"Generating Frame-by-Frame Walkthrough for Partial Setup #{setup_num}")
        print(f"{'='*80}\n")
        print(f"  Conditions Met: {partial.conditions_met}/4")
        print(f"  Failed At: {partial.failure_reason}")

        # Create frame directory
        frame_dir = self.output_dir / f"partial_setup_{setup_num}" / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Reset frame counter
        self.frame_id = 0

        # Get timestamps to visualize
        timestamps = self._get_key_timestamps(partial)

        print(f"\n  Generating {len(timestamps)} frames...")

        # Generate each frame
        for i, (ts, status, conditions) in enumerate(timestamps):
            if (i + 1) % 10 == 0:
                print(f"    Frame {i+1}/{len(timestamps)}...")

            self.generate_frame(partial, ts, frame_dir, status, conditions)

        print(f"\n  ✓ Generated {len(timestamps)} frames")

        # Create index HTML linking all frames
        self._create_index(frame_dir, setup_num, len(timestamps))

        print(f"\n✅ Walkthrough complete for Partial Setup #{setup_num}")

    def _get_key_timestamps(self, partial: PartialSetup) -> List[tuple]:
        """
        Get list of (timestamp, status_message, conditions_met) tuples.

        Args:
            partial: PartialSetup object

        Returns:
            List of (timestamp, status, [bool]*4) tuples
        """
        frames = []

        # Frame 1: Initial state
        start_time = partial.timestamp_1h_sweep - timedelta(hours=3)
        frames.append((
            start_time,
            "Looking for liquidity sweep...",
            [False, False, False, False]
        ))

        # Frame 2: Liquidity sweep
        frames.append((
            partial.timestamp_1h_sweep,
            f"✓ Liquidity Sweep!\nTrend: {partial.trend_1h_before_sweep.upper()}\nMoving to 5M...",
            [True, False, False, False]
        ))

        # Frames 3-N: 5M analysis
        if partial.timestamp_5m_event_b:
            # Before Event B
            before_event = partial.timestamp_5m_event_b - timedelta(minutes=15)
            if before_event > partial.timestamp_1h_sweep:
                frames.append((
                    before_event,
                    "Scanning 5M for Event B...",
                    [True, False, False, False]
                ))

            # Event B moment
            frames.append((
                partial.timestamp_5m_event_b,
                f"✓ Event B: {partial.condition_event_b}!\nOpposite direction confirmed\nLooking for validation...",
                [True, True, False, False]
            ))

        # Validation
        if partial.timestamp_5m_validation:
            frames.append((
                partial.timestamp_5m_validation,
                f"✓ Validation: {partial.condition_validation}!\nZone respected\nMoving to 1M...",
                [True, True, True, False]
            ))

            # 1M candles (every 2nd candle to reduce frame count)
            start_1m = partial.timestamp_5m_validation
            end_1m = start_1m + timedelta(minutes=30)

            candles_1m = self.strategy.df_1m[
                (self.strategy.df_1m.index > start_1m) &
                (self.strategy.df_1m.index <= end_1m)
            ]

            for i, ts in enumerate(candles_1m.index[::2]):  # Every 2nd candle
                frames.append((
                    ts,
                    f"Scanning 1M for confirmation...\nCandle {i*2 + 1}/{len(candles_1m)}",
                    [True, True, True, False]
                ))

        # Final frame
        final_time = partial.timestamp_5m_validation + timedelta(minutes=30) if partial.timestamp_5m_validation else \
                    (partial.timestamp_5m_event_b + timedelta(minutes=15) if partial.timestamp_5m_event_b else \
                     partial.timestamp_1h_sweep + timedelta(hours=1))

        frames.append((
            final_time,
            f"❌ INCOMPLETE\n\n{partial.failure_reason}\n\n{partial.conditions_met}/4 conditions met",
            [
                True,
                partial.conditions_met >= 2,
                partial.conditions_met >= 3,
                partial.conditions_met >= 4
            ]
        ))

        return frames

    def _create_index(self, frame_dir: Path, setup_num: int, total_frames: int):
        """
        Create index HTML with navigation between frames.

        Args:
            frame_dir: Directory containing frame HTMLs
            setup_num: Setup number
            total_frames: Total number of frames
        """
        # Get all frame files
        frame_files = sorted(frame_dir.glob("frame_*.html"))

        if len(frame_files) == 0:
            print("  ⚠️ No frames found")
            return

        # Create index HTML
        output_index = frame_dir.parent / "index.html"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Partial Setup #{setup_num} - Frame-by-Frame Walkthrough</title>
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
            max-width: 1400px;
        }}
        iframe {{
            width: 100%;
            height: 1050px;
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
            document.getElementById('frame-number').textContent = 'Frame ' + (currentFrame + 1) + ' of ' + totalFrames;

            // Update button states
            document.getElementById('prev-btn').disabled = (currentFrame === 0);
            document.getElementById('next-btn').disabled = (currentFrame === totalFrames - 1);
        }}

        function nextFrame() {{
            showFrame(currentFrame + 1);
        }}

        function prevFrame() {{
            showFrame(currentFrame - 1);
        }}

        function firstFrame() {{
            showFrame(0);
        }}

        function lastFrame() {{
            showFrame(totalFrames - 1);
        }}

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
    <h1>Partial Setup #{setup_num} - Frame-by-Frame Walkthrough</h1>

    <div class="controls">
        <button id="first-btn" onclick="firstFrame()">⏮ First</button>
        <button id="prev-btn" onclick="prevFrame()">◀ Previous</button>
        <span class="frame-info" id="frame-number">Frame 1 of {total_frames}</span>
        <input type="number" id="goto-input" min="1" max="{total_frames}" placeholder="Go to...">
        <button id="goto-btn" onclick="gotoFrame()">Go</button>
        <button id="next-btn" onclick="nextFrame()">Next ▶</button>
        <button id="last-btn" onclick="lastFrame()">Last ⏭</button>
    </div>

    <div class="frame-container">
        <iframe id="frame-viewer" src="frames/frame_0000.html"></iframe>
    </div>

    <div style="text-align: center; color: #888; margin-top: 20px;">
        Use arrow keys ← → or buttons to navigate | Home/End for first/last frame
    </div>
</body>
</html>
        """

        with open(output_index, 'w') as f:
            f.write(html_content)

        print(f"\n  ✓ Index created: {output_index}")
        print(f"    {len(frame_files)} frames, navigate with arrows or buttons")

    def analyze_partial_setup(self, partial: PartialSetup, setup_num: int):
        """
        Generate complete frame-by-frame analysis.

        Args:
            partial: PartialSetup object
            setup_num: Setup number
        """
        self.create_walkthrough(partial, setup_num)
