"""
Core visualization data generation functions.

Shared utilities for generating chart data in TradingView-compatible format.
Used by both tradingview_visualizer.py and strategy_visualizer.py.

Functions:
    - generate_candle_data: Convert DataFrame to candlestick format
    - generate_fvg_zones: Generate Fair Value Gap zone data
    - generate_ob_zones: Generate Order Block zone data
    - generate_bos_lines: Generate Break of Structure line data
    - generate_inflexion_markers: Generate inflexion point markers
    - generate_inflexion_lines: Generate inflexion level lines
    - generate_liquidity_lines: Generate liquidity sweep lines
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)


def generate_candle_data(df: pd.DataFrame) -> List[Dict]:
    """
    Convert DataFrame to candlestick data format for TradingView charts.

    Args:
        df: DataFrame with OHLC data indexed by datetime

    Returns:
        List of dicts with time, open, high, low, close keys
    """
    candle_data = []
    for idx, row in df.iterrows():
        candle_data.append({
            'time': int(idx.timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })
    return candle_data


def generate_fvg_zones(
    df: pd.DataFrame,
    fvg: pd.DataFrame,
    highlight_time: Optional[datetime] = None
) -> List[Dict]:
    """
    Generate FVG (Fair Value Gap) zone data for visualization.

    Args:
        df: DataFrame with OHLC data indexed by datetime
        fvg: DataFrame with FVG indicator data
        highlight_time: Optional datetime to highlight a specific FVG (e.g., Event B IFVG)

    Returns:
        List of dicts with zone data (startTime, endTime, topPrice, bottomPrice, etc.)
    """
    fvg_zones = []

    for i in range(len(fvg)):
        if i >= len(df):
            break

        # Check for FVG - handle both column styles (numpy array vs pandas series)
        fvg_value = fvg["FVG"].iloc[i] if hasattr(fvg["FVG"], 'iloc') else fvg["FVG"][i]
        if pd.isna(fvg_value) or np.isnan(fvg_value):
            continue

        # Get end index (mitigated or end of data)
        mitigated_idx = fvg["MitigatedIndex"].iloc[i] if hasattr(fvg["MitigatedIndex"], 'iloc') else fvg["MitigatedIndex"][i]
        if pd.notna(mitigated_idx) and mitigated_idx != 0:
            end_idx = int(mitigated_idx)
        else:
            end_idx = len(df) - 1
        end_idx = min(end_idx, len(df) - 1)

        # Check if this FVG should be highlighted (Event B IFVG)
        is_highlighted = False
        if highlight_time is not None and df.index[i] == highlight_time:
            is_highlighted = True

        # Get top and bottom prices
        top_price = fvg["Top"].iloc[i] if hasattr(fvg["Top"], 'iloc') else fvg["Top"][i]
        bottom_price = fvg["Bottom"].iloc[i] if hasattr(fvg["Bottom"], 'iloc') else fvg["Bottom"][i]

        fvg_zones.append({
            'startTime': int(df.index[i].timestamp()),
            'endTime': int(df.index[end_idx].timestamp()),
            'topPrice': float(top_price),
            'bottomPrice': float(bottom_price),
            'fvgType': int(fvg_value),  # 1 = bullish, -1 = bearish
            'highlighted': is_highlighted
        })

    return fvg_zones


def generate_ob_zones(df: pd.DataFrame, ob: pd.DataFrame) -> List[Dict]:
    """
    Generate Order Block zone data for visualization.

    Args:
        df: DataFrame with OHLC data indexed by datetime
        ob: DataFrame with Order Block indicator data

    Returns:
        List of dicts with zone data (startTime, endTime, topPrice, bottomPrice, obType)
    """
    ob_zones = []

    for i in range(len(ob)):
        if i >= len(df):
            break

        ob_value = ob["OB"].iloc[i]
        if pd.isna(ob_value) or np.isnan(ob_value):
            continue

        end_idx_val = ob["EndIndex"].iloc[i]
        if pd.notna(end_idx_val):
            end_idx = int(end_idx_val)
        else:
            end_idx = len(df) - 1
        end_idx = min(end_idx, len(df) - 1)

        ob_zones.append({
            'startTime': int(df.index[i].timestamp()),
            'endTime': int(df.index[end_idx].timestamp()),
            'topPrice': float(ob["Top"].iloc[i]),
            'bottomPrice': float(ob["Bottom"].iloc[i]),
            'obType': int(ob_value)  # 1 = bullish, -1 = bearish
        })

    return ob_zones


def generate_bos_lines(
    df: pd.DataFrame,
    bos: pd.DataFrame,
    inflexions: pd.DataFrame,
    highlight_time: Optional[datetime] = None
) -> List[Dict]:
    """
    Generate BOS (Break of Structure) line data for visualization.

    Args:
        df: DataFrame with OHLC data indexed by datetime
        bos: DataFrame with BOS indicator data
        inflexions: DataFrame with inflexion point data (to find matching levels)
        highlight_time: Optional datetime to highlight a specific BOS (e.g., Event B BOS)

    Returns:
        List of dicts with line data (startTime, endTime, price, color, label, etc.)
    """
    bos_lines = []

    for i in range(len(bos)):
        if i >= len(df):
            break

        bos_value = bos["BOS"].iloc[i]
        if pd.isna(bos_value) or np.isnan(bos_value):
            continue

        bos_type = bos_value
        level = bos["Level"].iloc[i]

        # Find matching inflexion point
        matching_inflexions = []
        for j in range(len(inflexions)):
            if j >= len(df) or j >= i:
                continue
            inflexion_level = inflexions["Level"].iloc[j]
            if pd.notna(inflexion_level) and inflexion_level == level:
                matching_inflexions.append(j)

        # Check if this BOS should be highlighted (Event B BOS)
        is_highlighted = False
        if highlight_time is not None and df.index[i] == highlight_time:
            is_highlighted = True

        # Determine color based on highlight and type
        if is_highlighted:
            color = '#ffff00'  # Yellow for highlighted
        else:
            color = '#089981' if bos_type == 1 else '#f23645'  # Green/red

        line_width = 4 if is_highlighted else 2

        if len(matching_inflexions) > 0:
            # Get the most recent matching inflexion
            inflexion_pos = matching_inflexions[-1]
            bos_lines.append({
                'startTime': int(df.index[inflexion_pos].timestamp()),
                'endTime': int(df.index[i].timestamp()),
                'price': float(level),
                'color': color,
                'lineWidth': line_width,
                'label': 'BOS',
                'highlighted': is_highlighted
            })
        else:
            # Fallback: draw a point if we can't find matching inflexion
            bos_lines.append({
                'startTime': int(df.index[i].timestamp()),
                'endTime': int(df.index[i].timestamp()),
                'price': float(level),
                'color': color,
                'lineWidth': line_width,
                'label': 'BOS',
                'highlighted': is_highlighted
            })

    return bos_lines


def generate_inflexion_markers(df: pd.DataFrame, inflexions: pd.DataFrame) -> List[Dict]:
    """
    Generate inflexion point markers for visualization.

    Used by tradingview_visualizer for detailed inflexion display.

    Args:
        df: DataFrame with OHLC data indexed by datetime
        inflexions: DataFrame with inflexion point data

    Returns:
        List of dicts with marker data (time, position, color, shape, text)
    """
    inflexion_markers = []

    for i in range(len(inflexions)):
        if i >= len(df):
            break

        inflx_type = inflexions["InflexionType"].iloc[i]
        if pd.isna(inflx_type) or np.isnan(inflx_type):
            continue

        respected = inflexions["Respected"].iloc[i]

        # Determine marker properties based on type and respect status
        if inflx_type == 1:  # Concave (high)
            if respected is True:
                color = '#00ff00'  # Green - LIQUIDITY SWEEP
                text = '\u2605'  # Star
                position = 'aboveBar'
            elif respected is False:
                color = '#ff0000'
                text = '\u2715'  # X mark
                position = 'aboveBar'
            else:
                color = '#ffff00'
                text = '\u25bc'  # Down arrow
                position = 'aboveBar'
        else:  # Convex (low)
            if respected is True:
                color = '#0080ff'  # Blue - LIQUIDITY SWEEP
                text = '\u2605'  # Star
                position = 'belowBar'
            elif respected is False:
                color = '#ff8800'
                text = '\u2715'  # X mark
                position = 'belowBar'
            else:
                color = '#ffff00'
                text = '\u25b2'  # Up arrow
                position = 'belowBar'

        inflexion_markers.append({
            'time': int(df.index[i].timestamp()),
            'position': position,
            'color': color,
            'shape': 'circle',
            'text': text
        })

    return inflexion_markers


def generate_inflexion_lines(df: pd.DataFrame, inflexions: pd.DataFrame) -> List[Dict]:
    """
    Generate inflexion level lines (horizontal lines from inflexion to status determination).

    Used by tradingview_visualizer for detailed inflexion display.

    Args:
        df: DataFrame with OHLC data indexed by datetime
        inflexions: DataFrame with inflexion point data

    Returns:
        List of dicts with line data (startTime, endTime, price, color, lineWidth, lineStyle)
    """
    inflexion_lines = []

    for i in range(len(inflexions)):
        if i >= len(df):
            break

        inflx_type = inflexions["InflexionType"].iloc[i]
        if pd.isna(inflx_type) or np.isnan(inflx_type):
            continue

        level = inflexions["Level"].iloc[i]
        respected = inflexions["Respected"].iloc[i]
        status_idx_val = inflexions["StatusIndex"].iloc[i]

        if status_idx_val != 0 and pd.notna(status_idx_val):
            status_idx = int(status_idx_val)
        else:
            status_idx = len(df) - 1
        status_idx = min(status_idx, len(df) - 1)

        # Color based on respect status
        if respected is True:
            line_color = '#00ff00' if inflx_type == 1 else '#0080ff'  # Green for concave, blue for convex
        elif respected is False:
            line_color = '#ff0000' if inflx_type == 1 else '#ff8800'  # Red/orange for broken
        else:
            line_color = '#ffff00'  # Yellow for pending

        inflexion_lines.append({
            'startTime': int(df.index[i].timestamp()),
            'endTime': int(df.index[status_idx].timestamp()),
            'price': float(level),
            'color': line_color,
            'lineWidth': 2 if respected is True else 1,
            'lineStyle': 'Solid' if respected is True else 'Dotted'
        })

    return inflexion_lines


def generate_liquidity_lines(df: pd.DataFrame, inflexions: pd.DataFrame) -> List[Dict]:
    """
    Generate liquidity lines for visualization.

    Shows respected inflexions (liquidity sweeps) as orange lines with X when swept.
    Used by strategy_visualizer for liquidity level display.

    Args:
        df: DataFrame with OHLC data indexed by datetime
        inflexions: DataFrame with inflexion point data

    Returns:
        List of dicts with line data (startTime, endTime, price, color, swept, lineWidth)
    """
    liquidity_lines = []

    for i in range(len(inflexions)):
        if i >= len(df):
            break

        inflx_type = inflexions["InflexionType"].iloc[i]
        if pd.isna(inflx_type) or np.isnan(inflx_type):
            continue

        level = inflexions["Level"].iloc[i]
        respected = inflexions["Respected"].iloc[i]
        status_idx = inflexions["StatusIndex"].iloc[i]

        # Only show liquidity lines for pending or respected inflexions
        if respected is True:  # Liquidity was swept
            if pd.notna(status_idx) and status_idx != 0:
                end_idx = int(status_idx)
            else:
                end_idx = len(df) - 1
            end_idx = min(end_idx, len(df) - 1)

            liquidity_lines.append({
                'startTime': int(df.index[i].timestamp()),
                'endTime': int(df.index[end_idx].timestamp()),
                'price': float(level),
                'color': '#ff8c00',  # Orange
                'swept': True,
                'lineWidth': 2
            })
        elif respected is None:  # Pending (not yet determined)
            end_idx = len(df) - 1

            liquidity_lines.append({
                'startTime': int(df.index[i].timestamp()),
                'endTime': int(df.index[end_idx].timestamp()),
                'price': float(level),
                'color': '#ff8c00',  # Orange
                'swept': False,
                'lineWidth': 1
            })

    return liquidity_lines
