"""
TSLA Smart Money Concepts Visualizer
Generates interactive charts with SMC indicators using Plotly
Adapted from tests/generate_gif.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add parent directory to path to import smc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.smc import smc
from src.indicators.smc_custom import smc_custom


class TSLAVisualizer:
    """
    Visualizes TSLA data with Smart Money Concepts indicators.

    All visualization methods adapted from tests/generate_gif.py
    """

    def __init__(self, results_dir=None):
        """
        Initialize visualizer.

        Args:
            results_dir (str, optional): Directory to save charts.
                                        Defaults to ../results/charts
        """
        if results_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            self.results_dir = project_root / 'results' / 'charts'
        else:
            self.results_dir = Path(results_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            'bullish_candle': '#77dd76',
            'bearish_candle': '#ff6962',
            'fvg': 'yellow',
            'ob': 'Purple',
            'bos': 'rgba(255, 165, 0, 0.2)',
            'choch': 'rgba(0, 0, 255, 0.2)',
            'liquidity': 'rgba(255, 165, 0, 0.2)',
            'liquidity_swept': 'rgba(255, 0, 0, 0.2)',
            'swing_high': 'rgba(255, 0, 0, 0.2)',
            'swing_low': 'rgba(0, 128, 0, 0.2)',
            'previous_hl': 'rgba(255, 255, 255, 0.2)',
            'session': '#16866E',
            'background': 'rgba(12, 14, 18, 1)'
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

    def add_swing_highs_lows(self, fig, df, swing_highs_lows_data):
        """Add Swing Highs and Lows visualization."""
        indexs = []
        level = []
        for i in range(len(swing_highs_lows_data)):
            if not np.isnan(swing_highs_lows_data["HighLow"][i]):
                indexs.append(i)
                level.append(swing_highs_lows_data["Level"][i])

        for i in range(len(indexs) - 1):
            fig.add_trace(
                go.Scatter(
                    x=[df.index[indexs[i]], df.index[indexs[i + 1]]],
                    y=[level[i], level[i + 1]],
                    mode="lines",
                    line=dict(
                        color=(
                            self.colors['swing_low']
                            if swing_highs_lows_data["HighLow"][indexs[i]] == -1
                            else self.colors['swing_high']
                        ),
                    ),
                    showlegend=False,
                )
            )
        return fig

    def add_bos_choch(self, fig, df, bos_choch_data):
        """Add Break of Structure and Change of Character visualization."""
        for i in range(len(bos_choch_data["BOS"])):
            if not np.isnan(bos_choch_data["BOS"][i]):
                mid_x = round((i + int(bos_choch_data["BrokenIndex"][i])) / 2)
                mid_y = bos_choch_data["Level"][i]
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i], df.index[int(bos_choch_data["BrokenIndex"][i])]],
                        y=[bos_choch_data["Level"][i], bos_choch_data["Level"][i]],
                        mode="lines",
                        line=dict(color=self.colors['bos']),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[mid_x]],
                        y=[mid_y],
                        mode="text",
                        text="BOS",
                        textposition="top center" if bos_choch_data["BOS"][i] == 1 else "bottom center",
                        textfont=dict(color="rgba(255, 165, 0, 0.4)", size=8),
                        showlegend=False,
                    )
                )
            if not np.isnan(bos_choch_data["CHOCH"][i]):
                mid_x = round((i + int(bos_choch_data["BrokenIndex"][i])) / 2)
                mid_y = bos_choch_data["Level"][i]
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i], df.index[int(bos_choch_data["BrokenIndex"][i])]],
                        y=[bos_choch_data["Level"][i], bos_choch_data["Level"][i]],
                        mode="lines",
                        line=dict(color=self.colors['choch']),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[mid_x]],
                        y=[mid_y],
                        mode="text",
                        text="CHOCH",
                        textposition="top center" if bos_choch_data["CHOCH"][i] == 1 else "bottom center",
                        textfont=dict(color="rgba(0, 0, 255, 0.4)", size=8),
                        showlegend=False,
                    )
                )
        return fig

    def add_OB(self, fig, df, ob_data):
        """Add Order Blocks visualization."""
        def format_volume(volume):
            if volume >= 1e12:
                return f"{volume / 1e12:.3f}T"
            elif volume >= 1e9:
                return f"{volume / 1e9:.3f}B"
            elif volume >= 1e6:
                return f"{volume / 1e6:.3f}M"
            elif volume >= 1e3:
                return f"{volume / 1e3:.3f}k"
            else:
                return f"{volume:.2f}"

        for i in range(len(ob_data["OB"])):
            if ob_data["OB"][i] == 1:
                x1 = int(
                    ob_data["MitigatedIndex"][i]
                    if ob_data["MitigatedIndex"][i] != 0
                    else len(df) - 1
                )
                fig.add_shape(
                    type="rect",
                    x0=df.index[i],
                    y0=ob_data["Bottom"][i],
                    x1=df.index[x1],
                    y1=ob_data["Top"][i],
                    line=dict(color=self.colors['ob']),
                    fillcolor=self.colors['ob'],
                    opacity=0.2,
                )

                if ob_data["MitigatedIndex"][i] > 0:
                    x_center = df.index[int(i + (ob_data["MitigatedIndex"][i] - i) / 2)]
                else:
                    x_center = df.index[int(i + (len(df) - i) / 2)]

                y_center = (ob_data["Bottom"][i] + ob_data["Top"][i]) / 2
                volume_text = format_volume(ob_data["OBVolume"][i])
                annotation_text = f'OB: {volume_text} ({ob_data["Percentage"][i]:.0f}%)'

                fig.add_annotation(
                    x=x_center,
                    y=y_center,
                    xref="x",
                    yref="y",
                    align="center",
                    text=annotation_text,
                    font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                    showarrow=False,
                )

        for i in range(len(ob_data["OB"])):
            if ob_data["OB"][i] == -1:
                x1 = int(
                    ob_data["MitigatedIndex"][i]
                    if ob_data["MitigatedIndex"][i] != 0
                    else len(df) - 1
                )
                fig.add_shape(
                    type="rect",
                    x0=df.index[i],
                    y0=ob_data["Bottom"][i],
                    x1=df.index[x1],
                    y1=ob_data["Top"][i],
                    line=dict(color=self.colors['ob']),
                    fillcolor=self.colors['ob'],
                    opacity=0.2,
                )

                if ob_data["MitigatedIndex"][i] > 0:
                    x_center = df.index[int(i + (ob_data["MitigatedIndex"][i] - i) / 2)]
                else:
                    x_center = df.index[int(i + (len(df) - i) / 2)]

                y_center = (ob_data["Bottom"][i] + ob_data["Top"][i]) / 2
                volume_text = format_volume(ob_data["OBVolume"][i])
                annotation_text = f'OB: {volume_text} ({ob_data["Percentage"][i]:.0f}%)'

                fig.add_annotation(
                    x=x_center,
                    y=y_center,
                    xref="x",
                    yref="y",
                    align="center",
                    text=annotation_text,
                    font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                    showarrow=False,
                )
        return fig

    def add_liquidity(self, fig, df, liquidity_data):
        """Add Liquidity levels and sweeps visualization."""
        for i in range(len(liquidity_data["Liquidity"])):
            if not np.isnan(liquidity_data["Liquidity"][i]):
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i], df.index[int(liquidity_data["End"][i])]],
                        y=[liquidity_data["Level"][i], liquidity_data["Level"][i]],
                        mode="lines",
                        line=dict(color=self.colors['liquidity']),
                        showlegend=False,
                    )
                )
                mid_x = round((i + int(liquidity_data["End"][i])) / 2)
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[mid_x]],
                        y=[liquidity_data["Level"][i]],
                        mode="text",
                        text="Liquidity",
                        textposition="top center" if liquidity_data["Liquidity"][i] == 1 else "bottom center",
                        textfont=dict(color="rgba(255, 165, 0, 0.4)", size=8),
                        showlegend=False,
                    )
                )
            if liquidity_data["Swept"][i] != 0 and not np.isnan(liquidity_data["Swept"][i]):
                fig.add_trace(
                    go.Scatter(
                        x=[
                            df.index[int(liquidity_data["End"][i])],
                            df.index[int(liquidity_data["Swept"][i])],
                        ],
                        y=[
                            liquidity_data["Level"][i],
                            (
                                df["high"].iloc[int(liquidity_data["Swept"][i])]
                                if liquidity_data["Liquidity"][i] == 1
                                else df["low"].iloc[int(liquidity_data["Swept"][i])]
                            ),
                        ],
                        mode="lines",
                        line=dict(color=self.colors['liquidity_swept']),
                        showlegend=False,
                    )
                )
                mid_x = round((i + int(liquidity_data["Swept"][i])) / 2)
                mid_y = (
                    liquidity_data["Level"][i]
                    + (
                        df["high"].iloc[int(liquidity_data["Swept"][i])]
                        if liquidity_data["Liquidity"][i] == 1
                        else df["low"].iloc[int(liquidity_data["Swept"][i])]
                    )
                ) / 2
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[mid_x]],
                        y=[mid_y],
                        mode="text",
                        text="Liquidity Swept",
                        textposition="top center" if liquidity_data["Liquidity"][i] == 1 else "bottom center",
                        textfont=dict(color="rgba(255, 0, 0, 0.4)", size=8),
                        showlegend=False,
                    )
                )
        return fig

    def add_previous_high_low(self, fig, df, previous_high_low_data):
        """Add Previous High/Low visualization."""
        high = previous_high_low_data["PreviousHigh"]
        low = previous_high_low_data["PreviousLow"]

        high_levels = []
        high_indexes = []
        for i in range(len(high)):
            if not np.isnan(high[i]) and high[i] != (high_levels[-1] if len(high_levels) > 0 else None):
                high_levels.append(high[i])
                high_indexes.append(i)

        low_levels = []
        low_indexes = []
        for i in range(len(low)):
            if not np.isnan(low[i]) and low[i] != (low_levels[-1] if len(low_levels) > 0 else None):
                low_levels.append(low[i])
                low_indexes.append(i)

        for i in range(len(high_indexes) - 1):
            fig.add_trace(
                go.Scatter(
                    x=[df.index[high_indexes[i]], df.index[high_indexes[i + 1]]],
                    y=[high_levels[i], high_levels[i]],
                    mode="lines",
                    line=dict(color=self.colors['previous_hl']),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.index[high_indexes[i + 1]]],
                    y=[high_levels[i]],
                    mode="text",
                    text="PH",
                    textposition="top center",
                    textfont=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                    showlegend=False,
                )
            )

        for i in range(len(low_indexes) - 1):
            fig.add_trace(
                go.Scatter(
                    x=[df.index[low_indexes[i]], df.index[low_indexes[i + 1]]],
                    y=[low_levels[i], low_levels[i]],
                    mode="lines",
                    line=dict(color=self.colors['previous_hl']),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.index[low_indexes[i + 1]]],
                    y=[low_levels[i]],
                    mode="text",
                    text="PL",
                    textposition="bottom center",
                    textfont=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                    showlegend=False,
                )
            )
        return fig

    def add_retracements(self, fig, df, retracements):
        """Add Retracement percentages visualization."""
        for i in range(len(retracements)):
            if (
                (
                    (
                        retracements["Direction"].iloc[i + 1]
                        if i < len(retracements) - 1
                        else 0
                    )
                    != retracements["Direction"].iloc[i]
                    or i == len(retracements) - 1
                )
                and retracements["Direction"].iloc[i] != 0
                and (
                    retracements["Direction"].iloc[i + 1]
                    if i < len(retracements) - 1
                    else retracements["Direction"].iloc[i]
                )
                != 0
            ):
                fig.add_annotation(
                    x=df.index[i],
                    y=(
                        df["high"].iloc[i]
                        if retracements["Direction"].iloc[i] == -1
                        else df["low"].iloc[i]
                    ),
                    xref="x",
                    yref="y",
                    text=f"C:{retracements['CurrentRetracement%'].iloc[i]}%<br>D:{retracements['DeepestRetracement%'].iloc[i]}%",
                    font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                    showarrow=False,
                )
        return fig

    def add_custom_inflexions(self, fig, df, inflexions_data):
        """
        Add Custom Inflexion Points visualization.

        Color coding:
        - Green (concave/peak): Respected resistance
        - Red (concave/peak): Disrespected resistance
        - Blue (convex/valley): Respected support
        - Orange (convex/valley): Disrespected support
        - Yellow: Pending (not yet reached)
        """
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

                # Add marker at inflexion point
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i]],
                        y=[level],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            size=10,
                            color=color,
                            line=dict(width=1, color='white')
                        ),
                        name=f"{'Peak' if inflx_type == 1 else 'Valley'}",
                        showlegend=False,
                        hovertemplate=f"<b>Inflexion</b><br>Type: {'Concave' if inflx_type == 1 else 'Convex'}<br>Level: {level:.2f}<br>Status: {respected}<extra></extra>"
                    )
                )

                # Draw horizontal line to show level
                end_idx = inflexions_data["StatusIndex"].iloc[i] if inflexions_data["StatusIndex"].iloc[i] != 0 else len(df) - 1
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i], df.index[int(end_idx)]],
                        y=[level, level],
                        mode='lines',
                        line=dict(color=color, width=1, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

        return fig

    def add_custom_bos(self, fig, df, bos_data):
        """
        Add Custom Break of Structure visualization.

        Shows:
        - BOS levels with arrows
        - Trend period background shading (green for bullish, red for bearish)
        """
        # Add trend period background shading
        trend_changes = []
        for i in range(len(bos_data)):
            if i == 0 or bos_data["Trend"].iloc[i] != bos_data["Trend"].iloc[i-1]:
                trend_changes.append(i)
        trend_changes.append(len(bos_data) - 1)

        for i in range(len(trend_changes) - 1):
            start_idx = trend_changes[i]
            end_idx = trend_changes[i + 1]
            trend = bos_data["Trend"].iloc[start_idx]

            if trend == 1:  # Bullish
                color = 'rgba(0, 255, 0, 0.05)'
            elif trend == -1:  # Bearish
                color = 'rgba(255, 0, 0, 0.05)'
            else:
                continue

            fig.add_vrect(
                x0=df.index[start_idx],
                x1=df.index[end_idx],
                fillcolor=color,
                layer="below",
                line_width=0,
            )

        # Add BOS markers and levels
        for i in range(len(bos_data)):
            if not np.isnan(bos_data["BOS"].iloc[i]):
                bos_type = bos_data["BOS"].iloc[i]
                level = bos_data["Level"].iloc[i]
                broken_idx = int(bos_data["BrokenIndex"].iloc[i])

                # Color: green for bullish BOS, red for bearish BOS
                color = 'rgba(0, 255, 0, 0.6)' if bos_type == 1 else 'rgba(255, 0, 0, 0.6)'

                # Draw line from inflexion to break point
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

                # Add BOS label at midpoint
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
                        hovertemplate=f"<b>BOS</b><br>Type: {'Bullish' if bos_type == 1 else 'Bearish'}<br>Level: {level:.2f}<extra></extra>"
                    )
                )

        return fig

    def plot_custom_indicators(
        self,
        df,
        close_break=True,
        indicators=None,
        window_size=None
    ):
        """
        Generate chart with custom SMC indicators.

        Args:
            df (pd.DataFrame): OHLCV data with 'time' column
            close_break (bool): Require close/open to break levels for BOS (default: True)
            indicators (list, optional): List of indicators to show. If None, shows all.
                                       Options: 'inflexions', 'bos'
            window_size (int, optional): Only plot last N candles. If None, plots all.

        Returns:
            plotly.graph_objects.Figure: Interactive chart
        """
        # If window_size specified, use only last N candles
        if window_size is not None:
            df = df.iloc[-window_size:].copy()

        # Set time as index if not already
        if 'time' in df.columns:
            df_indexed = df.set_index('time')
        else:
            df_indexed = df.copy()

        # Default: show all custom indicators
        if indicators is None:
            indicators = ['inflexions', 'bos']

        print(f"📊 Generating custom indicator visualization for {len(df_indexed)} candles...")

        # Create candlestick chart
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_indexed.index,
                    open=df_indexed["open"],
                    high=df_indexed["high"],
                    low=df_indexed["low"],
                    close=df_indexed["close"],
                    increasing_line_color=self.colors['bullish_candle'],
                    decreasing_line_color=self.colors['bearish_candle'],
                    name='TSLA'
                )
            ]
        )

        # Calculate and add custom indicators
        if 'inflexions' in indicators or 'bos' in indicators:
            print("  Calculating Custom Inflexion Points...")
            inflexions_data = smc_custom.inflexion_points(df_indexed)

            if 'inflexions' in indicators:
                print("  Adding Custom Inflexion Points...")
                fig = self.add_custom_inflexions(fig, df_indexed, inflexions_data)

            if 'bos' in indicators:
                print("  Calculating Custom BOS...")
                bos_data = smc_custom.bos(df_indexed, inflexions_data, close_break=close_break)
                print("  Adding Custom BOS...")
                fig = self.add_custom_bos(fig, df_indexed, bos_data)

        # Apply styling
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor=self.colors['background'],
            font=dict(color='white'),
            title='TSLA - Custom Smart Money Concepts Analysis',
            xaxis_title='Time',
            yaxis_title='Price',
            hovermode='x unified'
        )

        print("✅ Visualization complete!")
        return fig

    def plot_all_indicators(
        self,
        df,
        swing_length=50,
        previous_tf='1D',
        indicators=None,
        window_size=None
    ):
        """
        Generate chart with all SMC indicators.

        Args:
            df (pd.DataFrame): OHLCV data with 'time' column as index
            swing_length (int): Lookback period for swing highs/lows
            previous_tf (str): Timeframe for previous high/low (1h, 4h, 1day, etc.)
            indicators (list, optional): List of indicators to show. If None, shows all.
                                       Options: 'fvg', 'swing', 'bos_choch', 'ob',
                                               'liquidity', 'previous_hl', 'retracements'
            window_size (int, optional): Only plot last N candles. If None, plots all.

        Returns:
            plotly.graph_objects.Figure: Interactive chart
        """
        # If window_size specified, use only last N candles
        if window_size is not None:
            df = df.iloc[-window_size:].copy()

        # Set time as index if not already
        if 'time' in df.columns:
            df = df.set_index('time')

        # Default: show all indicators
        if indicators is None:
            indicators = ['fvg', 'swing', 'bos_choch', 'ob', 'liquidity', 'previous_hl', 'retracements']

        print(f"📊 Generating visualization for {len(df)} candles...")

        # Create candlestick chart
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    increasing_line_color=self.colors['bullish_candle'],
                    decreasing_line_color=self.colors['bearish_candle'],
                    name='TSLA'
                )
            ]
        )

        # Calculate and add indicators
        if 'fvg' in indicators:
            print("  Adding FVG...")
            fvg_data = smc.fvg(df, join_consecutive=True)
            fig = self.add_FVG(fig, df, fvg_data)

        if any(ind in indicators for ind in ['swing', 'bos_choch', 'ob', 'liquidity', 'retracements']):
            print("  Calculating Swing Highs/Lows...")
            swing_data = smc.swing_highs_lows(df, swing_length=swing_length)

            if 'swing' in indicators:
                print("  Adding Swing Highs/Lows...")
                fig = self.add_swing_highs_lows(fig, df, swing_data)

            if 'bos_choch' in indicators:
                print("  Adding BOS/CHOCH...")
                bos_choch_data = smc.bos_choch(df, swing_data)
                fig = self.add_bos_choch(fig, df, bos_choch_data)

            if 'ob' in indicators:
                print("  Adding Order Blocks...")
                ob_data = smc.ob(df, swing_data)
                fig = self.add_OB(fig, df, ob_data)

            if 'liquidity' in indicators:
                print("  Adding Liquidity...")
                liquidity_data = smc.liquidity(df, swing_data)
                fig = self.add_liquidity(fig, df, liquidity_data)

            if 'retracements' in indicators:
                print("  Adding Retracements...")
                retracements_data = smc.retracements(df, swing_data)
                fig = self.add_retracements(fig, df, retracements_data)

        if 'previous_hl' in indicators:
            print(f"  Adding Previous High/Low ({previous_tf})...")
            previous_hl_data = smc.previous_high_low(df, time_frame=previous_tf)
            fig = self.add_previous_high_low(fig, df, previous_hl_data)

        # Apply styling
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor=self.colors['background'],
            font=dict(color='white'),
            title='TSLA - Smart Money Concepts Analysis',
            xaxis_title='Time',
            yaxis_title='Price',
            hovermode='x unified'
        )

        print("✅ Visualization complete!")
        return fig

    def save_html(self, fig, filename):
        """
        Save figure as interactive HTML.

        Args:
            fig (plotly.graph_objects.Figure): Figure to save
            filename (str): Filename (without path)

        Returns:
            str: Full path to saved file
        """
        filepath = self.results_dir / filename
        fig.write_html(str(filepath))
        print(f"💾 Saved to {filepath}")
        return str(filepath)

    def save_image(self, fig, filename, width=1600, height=900):
        """
        Save figure as static image (PNG).
        Requires kaleido package: pip install kaleido

        Args:
            fig (plotly.graph_objects.Figure): Figure to save
            filename (str): Filename (without path)
            width (int): Image width in pixels
            height (int): Image height in pixels

        Returns:
            str: Full path to saved file
        """
        filepath = self.results_dir / filename
        try:
            fig.write_image(str(filepath), width=width, height=height)
            print(f"💾 Saved to {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"⚠️ Error saving image: {e}")
            print("   Tip: Install kaleido with: pip install kaleido")
            return None


# Example usage
if __name__ == "__main__":
    from data_loader import TSLADataLoader

    # Load data
    loader = TSLADataLoader()
    df = loader.get_data('15min')

    # Create visualizer
    viz = TSLAVisualizer()

    # Plot with all indicators
    fig = viz.plot_all_indicators(df, swing_length=50, window_size=500)

    # Save as HTML
    viz.save_html(fig, 'tsla_15min_analysis.html')

    print("\n✅ Done! Open the HTML file in your browser to view the chart.")
