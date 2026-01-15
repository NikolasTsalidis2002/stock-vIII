"""
Visual Analyzer for Multi-Timeframe Strategy

Generates step-by-step annotated frames showing strategy execution:
1. Frame 1: 1H Liquidity Sweep
2. Frame 2: 5M Event B (BOS/IFVG)
3. Frame 3: 5M Validation (FVG/Demand Zone)
4. Frame 4: 1M Final Confirmation
5. Frame 5: Entry Summary (Multi-panel view)

Each frame is an interactive Plotly HTML chart with annotations.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple
from datetime import timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy import TradeSignal, MultiTimeframeStrategy


class VisualAnalyzer:
    """
    Generates visual walkthroughs of trade setups.

    Creates 5-frame sequence showing:
    1. 1H Liquidity Sweep detection
    2. 5M Event B (BOS/IFVG opposite direction)
    3. 5M FVG/Demand Zone validation
    4. 1M Final confirmation
    5. Entry point summary
    """

    def __init__(self, strategy: MultiTimeframeStrategy, output_dir: Optional[str] = None):
        """
        Initialize visual analyzer.

        Args:
            strategy: MultiTimeframeStrategy instance with calculated indicators
            output_dir: Directory to save output frames (default: results/strategy_examples)
        """
        self.strategy = strategy

        if output_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            self.output_dir = project_root / 'results' / 'strategy_examples'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            'bullish_candle': '#77dd76',
            'bearish_candle': '#ff6962',
            'background': 'rgba(12, 14, 18, 1)',
            'liquidity_sweep': 'rgba(255, 165, 0, 0.8)',
            'event_b_bos': 'rgba(0, 255, 255, 0.6)',
            'event_b_ifvg': 'rgba(255, 0, 255, 0.6)',
            'fvg': 'rgba(255, 255, 0, 0.4)',
            'demand_zone': 'rgba(128, 0, 128, 0.4)',
            'confirmation': 'rgba(0, 255, 0, 0.7)',
            'entry_arrow': '#00ff00',
            'trend_bullish': 'rgba(0, 255, 0, 0.1)',
            'trend_bearish': 'rgba(255, 0, 0, 0.1)',
        }

    def create_candlestick_base(
        self,
        df: pd.DataFrame,
        title: str,
        window: Optional[Tuple[int, int]] = None
    ) -> go.Figure:
        """
        Create base candlestick chart.

        Args:
            df: DataFrame with OHLC data (datetime index)
            title: Chart title
            window: Optional (start_idx, end_idx) to show only subset

        Returns:
            Plotly figure
        """
        if window is not None:
            start_idx, end_idx = window
            df_plot = df.iloc[start_idx:end_idx+1]
        else:
            df_plot = df

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_plot.index,
                    open=df_plot['open'],
                    high=df_plot['high'],
                    low=df_plot['low'],
                    close=df_plot['close'],
                    increasing_line_color=self.colors['bullish_candle'],
                    decreasing_line_color=self.colors['bearish_candle'],
                    name='Price'
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor=self.colors['background'],
            font=dict(color='white', size=12),
            hovermode='x unified',
            height=700
        )

        return fig

    def add_annotation_box(
        self,
        fig: go.Figure,
        text: str,
        x_position: float = 0.5,
        y_position: float = 0.95
    ):
        """
        Add text annotation box to figure.

        Args:
            fig: Plotly figure
            text: Annotation text (supports HTML)
            x_position: X position (0-1, paper coordinates)
            y_position: Y position (0-1, paper coordinates)
        """
        fig.add_annotation(
            text=text,
            xref='paper',
            yref='paper',
            x=x_position,
            y=y_position,
            xanchor='center',
            yanchor='top',
            showarrow=False,
            bgcolor='rgba(0, 0, 0, 0.8)',
            bordercolor='white',
            borderwidth=2,
            borderpad=10,
            font=dict(size=14, color='white', family='Arial Bold')
        )

    def generate_frame_1_liquidity_sweep(
        self,
        signal: TradeSignal,
        trade_num: int
    ) -> str:
        """
        Frame 1: Show 1H liquidity sweep.

        Args:
            signal: TradeSignal object
            trade_num: Trade number for output filename

        Returns:
            Path to saved HTML file
        """
        print("  [Frame 1] Generating 1H Liquidity Sweep visualization...")

        # Get 1H data around sweep
        sweep_idx = self.strategy.df_1h.index.get_loc(signal.timestamp_1h_sweep)
        start_idx = max(0, sweep_idx - 20)
        end_idx = min(len(self.strategy.df_1h) - 1, sweep_idx + 10)

        # Create base chart
        fig = self.create_candlestick_base(
            self.strategy.df_1h,
            "Frame 1: 1H Liquidity Sweep Detection",
            window=(start_idx, end_idx)
        )

        # Highlight trend background before sweep
        trend_color = self.colors['trend_bullish'] if signal.trend_1h_before_sweep == 'bullish' else self.colors['trend_bearish']

        fig.add_vrect(
            x0=self.strategy.df_1h.index[start_idx],
            x1=signal.timestamp_1h_sweep,
            fillcolor=trend_color,
            layer="below",
            line_width=0,
        )

        # Mark liquidity sweep point
        fig.add_trace(
            go.Scatter(
                x=[signal.timestamp_1h_sweep],
                y=[signal.price_1h_sweep],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=20,
                    color=self.colors['liquidity_sweep'],
                    line=dict(width=2, color='white')
                ),
                name='Liquidity Sweep',
                hovertemplate=f"<b>Liquidity Sweep</b><br>Price: {signal.price_1h_sweep:.2f}<extra></extra>"
            )
        )

        # Add vertical line at sweep
        fig.add_vline(
            x=signal.timestamp_1h_sweep,
            line_dash="dash",
            line_color=self.colors['liquidity_sweep'],
            line_width=2
        )

        # Annotation box
        annotation_text = f"""
        <b>STEP 1: 1H LIQUIDITY SWEEP</b><br>
        <br>
        Trend: {signal.trend_1h_before_sweep.upper()}<br>
        Sweep Type: {signal.condition_liquidity_sweep}<br>
        Time: {signal.timestamp_1h_sweep.strftime('%Y-%m-%d %H:%M')}<br>
        Price: ${signal.price_1h_sweep:.2f}<br>
        <br>
        ✓ Condition 1 Met: Liquidity Sweep Detected
        """

        self.add_annotation_box(fig, annotation_text)

        # Save
        output_path = self.output_dir / f"trade_{trade_num}" / "frame_1_liquidity_sweep.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

        print(f"    Saved: {output_path}")
        return str(output_path)

    def generate_frame_2_event_b(
        self,
        signal: TradeSignal,
        trade_num: int
    ) -> str:
        """
        Frame 2: Show 5M Event B (BOS or IFVG).

        Args:
            signal: TradeSignal object
            trade_num: Trade number for output filename

        Returns:
            Path to saved HTML file
        """
        print("  [Frame 2] Generating 5M Event B (BOS/IFVG) visualization...")

        # Get 5M data around Event B
        event_b_idx = self.strategy.df_5m.index.get_loc(signal.timestamp_5m_event_b)
        start_idx = max(0, event_b_idx - 50)
        end_idx = min(len(self.strategy.df_5m) - 1, event_b_idx + 30)

        # Create base chart
        fig = self.create_candlestick_base(
            self.strategy.df_5m,
            "Frame 2: 5M Event B - Opposite Direction Detected",
            window=(start_idx, end_idx)
        )

        # Mark Event B point
        event_b_color = self.colors['event_b_bos'] if signal.condition_event_b == 'BOS' else self.colors['event_b_ifvg']

        fig.add_trace(
            go.Scatter(
                x=[signal.timestamp_5m_event_b],
                y=[signal.price_5m_event_b],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=18,
                    color=event_b_color,
                    line=dict(width=2, color='white')
                ),
                name=f'Event B ({signal.condition_event_b})',
                hovertemplate=f"<b>Event B: {signal.condition_event_b}</b><br>Price: {signal.price_5m_event_b:.2f}<extra></extra>"
            )
        )

        # Add vertical line
        fig.add_vline(
            x=signal.timestamp_5m_event_b,
            line_dash="dash",
            line_color=event_b_color,
            line_width=2
        )

        # Show custom BOS trend if available
        bos_5m_window = self.strategy.bos_5m.loc[
            self.strategy.df_5m.index[start_idx]:self.strategy.df_5m.index[end_idx]
        ]

        # Highlight trend changes
        for i in range(len(bos_5m_window) - 1):
            trend = bos_5m_window['Trend'].iloc[i]
            if trend == 1:
                color = self.colors['trend_bullish']
            elif trend == -1:
                color = self.colors['trend_bearish']
            else:
                continue

            fig.add_vrect(
                x0=bos_5m_window.index[i],
                x1=bos_5m_window.index[i+1],
                fillcolor=color,
                layer="below",
                line_width=0,
            )

        # Annotation
        opposite_direction = "BEARISH" if signal.trend_1h_before_sweep == 'bullish' else "BULLISH"
        annotation_text = f"""
        <b>STEP 2: 5M EVENT B</b><br>
        <br>
        Event Type: {signal.condition_event_b}<br>
        Direction: {opposite_direction} (opposite of 1H)<br>
        Time: {signal.timestamp_5m_event_b.strftime('%Y-%m-%d %H:%M')}<br>
        Price: ${signal.price_5m_event_b:.2f}<br>
        <br>
        ✓ Condition 2 Met: Opposite Direction Confirmed
        """

        self.add_annotation_box(fig, annotation_text)

        # Save
        output_path = self.output_dir / f"trade_{trade_num}" / "frame_2_event_b.html"
        fig.write_html(str(output_path))

        print(f"    Saved: {output_path}")
        return str(output_path)

    def generate_frame_3_validation(
        self,
        signal: TradeSignal,
        trade_num: int
    ) -> str:
        """
        Frame 3: Show 5M FVG or Demand Zone validation.

        Args:
            signal: TradeSignal object
            trade_num: Trade number for output filename

        Returns:
            Path to saved HTML file
        """
        print("  [Frame 3] Generating 5M Validation (FVG/Demand Zone) visualization...")

        # Get 5M data around validation
        validation_idx = self.strategy.df_5m.index.get_loc(signal.timestamp_5m_validation)
        start_idx = max(0, validation_idx - 40)
        end_idx = min(len(self.strategy.df_5m) - 1, validation_idx + 20)

        # Create base chart
        fig = self.create_candlestick_base(
            self.strategy.df_5m,
            "Frame 3: 5M Validation - FVG or Demand Zone Respect",
            window=(start_idx, end_idx)
        )

        # Highlight validation zone
        validation_color = self.colors['fvg'] if signal.condition_validation == 'FVG' else self.colors['demand_zone']

        # Draw rectangle for validation zone (approximate based on price)
        zone_height = signal.price_5m_validation * 0.01  # 1% height
        fig.add_shape(
            type="rect",
            x0=self.strategy.df_5m.index[start_idx],
            y0=signal.price_5m_validation - zone_height,
            x1=signal.timestamp_5m_validation,
            y1=signal.price_5m_validation + zone_height,
            line=dict(width=2, color=validation_color),
            fillcolor=validation_color,
            opacity=0.3,
        )

        # Mark validation point
        fig.add_trace(
            go.Scatter(
                x=[signal.timestamp_5m_validation],
                y=[signal.price_5m_validation],
                mode='markers',
                marker=dict(
                    symbol='hexagon',
                    size=16,
                    color=validation_color,
                    line=dict(width=2, color='white')
                ),
                name=signal.condition_validation,
                hovertemplate=f"<b>{signal.condition_validation} Respected</b><br>Price: {signal.price_5m_validation:.2f}<extra></extra>"
            )
        )

        # Add vertical line
        fig.add_vline(
            x=signal.timestamp_5m_validation,
            line_dash="dash",
            line_color=validation_color,
            line_width=2
        )

        # Annotation
        annotation_text = f"""
        <b>STEP 3: 5M VALIDATION</b><br>
        <br>
        Zone Type: {signal.condition_validation}<br>
        Status: RESPECTED (price bounced)<br>
        Time: {signal.timestamp_5m_validation.strftime('%Y-%m-%d %H:%M')}<br>
        Price: ${signal.price_5m_validation:.2f}<br>
        <br>
        ✓ Condition 3 Met: {signal.condition_validation} Validated
        """

        self.add_annotation_box(fig, annotation_text)

        # Save
        output_path = self.output_dir / f"trade_{trade_num}" / "frame_3_validation.html"
        fig.write_html(str(output_path))

        print(f"    Saved: {output_path}")
        return str(output_path)

    def generate_frame_4_confirmation(
        self,
        signal: TradeSignal,
        trade_num: int
    ) -> str:
        """
        Frame 4: Show 1M final confirmation.

        Args:
            signal: TradeSignal object
            trade_num: Trade number for output filename

        Returns:
            Path to saved HTML file
        """
        print("  [Frame 4] Generating 1M Final Confirmation visualization...")

        # Get 1M data around confirmation
        confirmation_idx = self.strategy.df_1m.index.get_loc(signal.timestamp_1m_confirmation)
        start_idx = max(0, confirmation_idx - 100)
        end_idx = min(len(self.strategy.df_1m) - 1, confirmation_idx + 30)

        # Create base chart
        fig = self.create_candlestick_base(
            self.strategy.df_1m,
            "Frame 4: 1M Final Confirmation - Entry Signal",
            window=(start_idx, end_idx)
        )

        # Mark confirmation point
        fig.add_trace(
            go.Scatter(
                x=[signal.timestamp_1m_confirmation],
                y=[signal.price_1m_confirmation],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if signal.entry_direction == 'long' else 'triangle-down',
                    size=20,
                    color=self.colors['confirmation'],
                    line=dict(width=2, color='white')
                ),
                name=f'{signal.condition_confirmation}',
                hovertemplate=f"<b>Confirmation: {signal.condition_confirmation}</b><br>Price: {signal.price_1m_confirmation:.2f}<extra></extra>"
            )
        )

        # Add vertical line
        fig.add_vline(
            x=signal.timestamp_1m_confirmation,
            line_dash="dash",
            line_color=self.colors['confirmation'],
            line_width=3
        )

        # Add entry arrow
        arrow_y = signal.price_1m_confirmation * 0.98 if signal.entry_direction == 'long' else signal.price_1m_confirmation * 1.02

        fig.add_annotation(
            x=signal.timestamp_1m_confirmation,
            y=arrow_y,
            text="ENTRY",
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor=self.colors['entry_arrow'],
            ax=0,
            ay=40 if signal.entry_direction == 'long' else -40,
            font=dict(size=16, color=self.colors['entry_arrow'], family='Arial Black')
        )

        # Annotation
        annotation_text = f"""
        <b>STEP 4: 1M FINAL CONFIRMATION</b><br>
        <br>
        Confirmation: {signal.condition_confirmation}<br>
        Entry Direction: {signal.entry_direction.upper()}<br>
        Time: {signal.timestamp_1m_confirmation.strftime('%Y-%m-%d %H:%M')}<br>
        Entry Price: ${signal.price_entry:.2f}<br>
        <br>
        ✅ ALL CONDITIONS MET - ENTER {signal.entry_direction.upper()}
        """

        self.add_annotation_box(fig, annotation_text)

        # Save
        output_path = self.output_dir / f"trade_{trade_num}" / "frame_4_confirmation.html"
        fig.write_html(str(output_path))

        print(f"    Saved: {output_path}")
        return str(output_path)

    def generate_frame_5_summary(
        self,
        signal: TradeSignal,
        trade_num: int
    ) -> str:
        """
        Frame 5: Multi-panel summary showing all timeframes.

        Args:
            signal: TradeSignal object
            trade_num: Trade number for output filename

        Returns:
            Path to saved HTML file
        """
        print("  [Frame 5] Generating Entry Summary (Multi-Panel) visualization...")

        # Create subplots (3 rows)
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=('1H: Liquidity Sweep', '5M: Event B + Validation', '1M: Final Confirmation'),
            vertical_spacing=0.08,
            row_heights=[0.33, 0.33, 0.34]
        )

        # Panel 1: 1H
        sweep_idx = self.strategy.df_1h.index.get_loc(signal.timestamp_1h_sweep)
        df_1h_plot = self.strategy.df_1h.iloc[max(0, sweep_idx-15):min(len(self.strategy.df_1h), sweep_idx+10)]

        fig.add_trace(
            go.Candlestick(
                x=df_1h_plot.index,
                open=df_1h_plot['open'],
                high=df_1h_plot['high'],
                low=df_1h_plot['low'],
                close=df_1h_plot['close'],
                increasing_line_color=self.colors['bullish_candle'],
                decreasing_line_color=self.colors['bearish_candle'],
                name='1H',
                showlegend=False
            ),
            row=1, col=1
        )

        # Panel 2: 5M
        event_b_idx = self.strategy.df_5m.index.get_loc(signal.timestamp_5m_event_b)
        validation_idx = self.strategy.df_5m.index.get_loc(signal.timestamp_5m_validation)
        df_5m_plot = self.strategy.df_5m.iloc[max(0, event_b_idx-30):min(len(self.strategy.df_5m), validation_idx+20)]

        fig.add_trace(
            go.Candlestick(
                x=df_5m_plot.index,
                open=df_5m_plot['open'],
                high=df_5m_plot['high'],
                low=df_5m_plot['low'],
                close=df_5m_plot['close'],
                increasing_line_color=self.colors['bullish_candle'],
                decreasing_line_color=self.colors['bearish_candle'],
                name='5M',
                showlegend=False
            ),
            row=2, col=1
        )

        # Panel 3: 1M
        confirmation_idx = self.strategy.df_1m.index.get_loc(signal.timestamp_1m_confirmation)
        df_1m_plot = self.strategy.df_1m.iloc[max(0, confirmation_idx-60):min(len(self.strategy.df_1m), confirmation_idx+20)]

        fig.add_trace(
            go.Candlestick(
                x=df_1m_plot.index,
                open=df_1m_plot['open'],
                high=df_1m_plot['high'],
                low=df_1m_plot['low'],
                close=df_1m_plot['close'],
                increasing_line_color=self.colors['bullish_candle'],
                decreasing_line_color=self.colors['bearish_candle'],
                name='1M',
                showlegend=False
            ),
            row=3, col=1
        )

        # Add vertical lines showing progression
        for row in range(1, 4):
            if row == 1:
                fig.add_vline(x=signal.timestamp_1h_sweep, line_dash="dash", line_color=self.colors['liquidity_sweep'], row=row, col=1)
            elif row == 2:
                fig.add_vline(x=signal.timestamp_5m_event_b, line_dash="dash", line_color=self.colors['event_b_bos'], row=row, col=1)
                fig.add_vline(x=signal.timestamp_5m_validation, line_dash="dash", line_color=self.colors['fvg'], row=row, col=1)
            else:
                fig.add_vline(x=signal.timestamp_1m_confirmation, line_dash="dash", line_color=self.colors['confirmation'], row=row, col=1, line_width=3)

        # Update layout
        fig.update_layout(
            title="Frame 5: Complete Trade Setup - All Timeframes",
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            xaxis3_rangeslider_visible=False,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor=self.colors['background'],
            font=dict(color='white', size=11),
            height=1000
        )

        # Add summary annotation
        summary_text = f"""
        <b>TRADE SUMMARY</b><br>
        <br>
        Entry: {signal.entry_direction.upper()} @ ${signal.price_entry:.2f}<br>
        Time: {signal.timestamp_entry.strftime('%Y-%m-%d %H:%M')}<br>
        <br>
        ✓ 1H Liquidity Sweep: {signal.condition_liquidity_sweep}<br>
        ✓ 5M Event B: {signal.condition_event_b}<br>
        ✓ 5M Validation: {signal.condition_validation}<br>
        ✓ 1M Confirmation: {signal.condition_confirmation}<br>
        """

        fig.add_annotation(
            text=summary_text,
            xref='paper',
            yref='paper',
            x=0.5,
            y=1.08,
            xanchor='center',
            yanchor='top',
            showarrow=False,
            bgcolor='rgba(0, 100, 0, 0.8)',
            bordercolor='white',
            borderwidth=2,
            borderpad=10,
            font=dict(size=13, color='white', family='Arial Bold')
        )

        # Save
        output_path = self.output_dir / f"trade_{trade_num}" / "frame_5_summary.html"
        fig.write_html(str(output_path))

        print(f"    Saved: {output_path}")
        return str(output_path)

    def generate_index_html(self, trade_num: int, signal: TradeSignal):
        """
        Generate index HTML with navigation links between frames.

        Args:
            trade_num: Trade number
            signal: TradeSignal object
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trade #{trade_num} - Strategy Walkthrough</title>
    <style>
        body {{
            background-color: #0c0e12;
            color: white;
            font-family: Arial, sans-serif;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #00ff00;
            border-bottom: 2px solid #00ff00;
            padding-bottom: 10px;
        }}
        .frame-link {{
            display: block;
            background-color: #1a1d24;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            text-decoration: none;
            color: white;
            transition: all 0.3s;
        }}
        .frame-link:hover {{
            background-color: #2a2d34;
            border-color: #00ff00;
            transform: translateX(10px);
        }}
        .frame-number {{
            color: #00ff00;
            font-size: 24px;
            font-weight: bold;
        }}
        .frame-title {{
            font-size: 18px;
            margin: 5px 0;
        }}
        .frame-details {{
            color: #aaa;
            font-size: 14px;
        }}
        .summary {{
            background-color: #1a3a1a;
            border: 2px solid #00ff00;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .summary-item {{
            margin: 8px 0;
        }}
        .checkmark {{
            color: #00ff00;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Trade #{trade_num} - Multi-Timeframe Strategy Walkthrough</h1>

    <div class="summary">
        <h2>Trade Summary</h2>
        <div class="summary-item"><strong>Entry Direction:</strong> {signal.entry_direction.upper()}</div>
        <div class="summary-item"><strong>Entry Price:</strong> ${signal.price_entry:.2f}</div>
        <div class="summary-item"><strong>Entry Time:</strong> {signal.timestamp_entry.strftime('%Y-%m-%d %H:%M')}</div>
        <br>
        <div class="summary-item"><span class="checkmark">✓</span> 1H Liquidity Sweep: {signal.condition_liquidity_sweep}</div>
        <div class="summary-item"><span class="checkmark">✓</span> 5M Event B: {signal.condition_event_b}</div>
        <div class="summary-item"><span class="checkmark">✓</span> 5M Validation: {signal.condition_validation}</div>
        <div class="summary-item"><span class="checkmark">✓</span> 1M Confirmation: {signal.condition_confirmation}</div>
    </div>

    <h2>Step-by-Step Analysis</h2>

    <a href="frame_1_liquidity_sweep.html" class="frame-link">
        <div class="frame-number">Frame 1</div>
        <div class="frame-title">1H Liquidity Sweep Detection</div>
        <div class="frame-details">
            Trend: {signal.trend_1h_before_sweep.upper()} |
            Sweep: {signal.condition_liquidity_sweep} |
            Time: {signal.timestamp_1h_sweep.strftime('%Y-%m-%d %H:%M')}
        </div>
    </a>

    <a href="frame_2_event_b.html" class="frame-link">
        <div class="frame-number">Frame 2</div>
        <div class="frame-title">5M Event B - Opposite Direction</div>
        <div class="frame-details">
            Type: {signal.condition_event_b} |
            Time: {signal.timestamp_5m_event_b.strftime('%Y-%m-%d %H:%M')}
        </div>
    </a>

    <a href="frame_3_validation.html" class="frame-link">
        <div class="frame-number">Frame 3</div>
        <div class="frame-title">5M Validation - FVG/Demand Zone</div>
        <div class="frame-details">
            Zone: {signal.condition_validation} |
            Time: {signal.timestamp_5m_validation.strftime('%Y-%m-%d %H:%M')}
        </div>
    </a>

    <a href="frame_4_confirmation.html" class="frame-link">
        <div class="frame-number">Frame 4</div>
        <div class="frame-title">1M Final Confirmation</div>
        <div class="frame-details">
            Type: {signal.condition_confirmation} |
            Entry: {signal.entry_direction.upper()} @ ${signal.price_entry:.2f} |
            Time: {signal.timestamp_1m_confirmation.strftime('%Y-%m-%d %H:%M')}
        </div>
    </a>

    <a href="frame_5_summary.html" class="frame-link">
        <div class="frame-number">Frame 5</div>
        <div class="frame-title">Complete Summary - All Timeframes</div>
        <div class="frame-details">
            Multi-panel view showing entire trade setup progression
        </div>
    </a>

</body>
</html>
        """

        output_path = self.output_dir / f"trade_{trade_num}" / "index.html"
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"  [Index] Saved: {output_path}")

    def analyze_trade(self, signal: TradeSignal, trade_num: int):
        """
        Generate complete 5-frame visual walkthrough for a trade.

        Args:
            signal: TradeSignal object
            trade_num: Trade number (for output organization)
        """
        print(f"\n{'='*80}")
        print(f"Generating visual walkthrough for Trade #{trade_num}")
        print(f"{'='*80}\n")

        # Generate all 5 frames
        self.generate_frame_1_liquidity_sweep(signal, trade_num)
        self.generate_frame_2_event_b(signal, trade_num)
        self.generate_frame_3_validation(signal, trade_num)
        self.generate_frame_4_confirmation(signal, trade_num)
        self.generate_frame_5_summary(signal, trade_num)

        # Generate index
        self.generate_index_html(trade_num, signal)

        print(f"\n✅ Trade #{trade_num} walkthrough complete!")
        print(f"   Open: {self.output_dir}/trade_{trade_num}/index.html\n")
