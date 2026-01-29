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

        end_idx_val = ob["StatusIndex"].iloc[i]
        if end_idx_val > 0:
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


def generate_gmm_zones(gmm_zone_info, df: pd.DataFrame) -> Optional[Dict]:
    """
    Generate GMM zone visualization data for dashboard display.

    Creates:
    - 3 horizontal zone bands (premium, middle, discount)
    - 7 Fibonacci level lines
    - Current price indicator with position info
    - Zone labels for chart overlay

    Args:
        gmm_zone_info: GMMZoneInfo object from GMM zone detector
        df: DataFrame with price data (used for time range)

    Returns:
        Dict with zoneBands, fibLevels, currentPriceIndicator, zoneLabels
        or None if gmm_zone_info is None
    """
    if gmm_zone_info is None:
        return None

    # Get time range from DataFrame
    start_time = int(df.index[0].timestamp())
    end_time = int(df.index[-1].timestamp())

    # Zone colors (low opacity for clearer visualization)
    zone_colors = {
        'premium': {
            'fill': 'rgba(244, 67, 54, 0.05)',
            'border': '#f44336'  # Red
        },
        'middle': {
            'fill': 'rgba(255, 193, 7, 0.05)',
            'border': '#ffc107'  # Amber
        },
        'discount': {
            'fill': 'rgba(76, 175, 80, 0.05)',
            'border': '#4caf50'  # Green
        }
    }

    # Fibonacci level colors
    fib_colors = {
        1.0: '#f44336',    # Red
        0.786: '#e91e63',  # Pink
        0.618: '#9c27b0',  # Purple
        0.5: '#ffeb3b',    # Yellow (equilibrium)
        0.382: '#3f51b5',  # Indigo
        0.236: '#4caf50',  # Green
        0.0: '#00bcd4'     # Cyan
    }

    fib_prices = gmm_zone_info.fib_prices

    # Get premium/discount zone boundaries from fib_prices
    # Premium zone: 0.786 - 1.0
    # Middle zone: 0.236 - 0.786
    # Discount zone: 0.0 - 0.236
    premium_top = fib_prices.get(1.0, 0)
    premium_bottom = fib_prices.get(0.786, 0)
    discount_top = fib_prices.get(0.236, 0)
    discount_bottom = fib_prices.get(0.0, 0)

    # Generate zone bands
    zone_bands = [
        {
            'startTime': start_time,
            'endTime': end_time,
            'topPrice': float(premium_top),
            'bottomPrice': float(premium_bottom),
            'zoneType': 'premium',
            'fillColor': zone_colors['premium']['fill'],
            'borderColor': zone_colors['premium']['border']
        },
        {
            'startTime': start_time,
            'endTime': end_time,
            'topPrice': float(premium_bottom),
            'bottomPrice': float(discount_top),
            'zoneType': 'middle',
            'fillColor': zone_colors['middle']['fill'],
            'borderColor': zone_colors['middle']['border']
        },
        {
            'startTime': start_time,
            'endTime': end_time,
            'topPrice': float(discount_top),
            'bottomPrice': float(discount_bottom),
            'zoneType': 'discount',
            'fillColor': zone_colors['discount']['fill'],
            'borderColor': zone_colors['discount']['border']
        }
    ]

    # Generate Fibonacci level lines
    fib_levels = []
    for level in sorted(fib_prices.keys(), reverse=True):
        price = fib_prices[level]
        color = fib_colors.get(level, '#787b86')  # Default gray

        # Equilibrium line (50%) is dashed and wider
        line_width = 2 if level == 0.5 else 1
        line_style = 'Dashed' if level == 0.5 else 'Solid'

        # Create label based on level
        label = f'{level * 100:.1f}%: ${price:.2f}'

        fib_levels.append({
            'startTime': start_time,
            'endTime': end_time,
            'price': float(price),
            'level': level,
            'color': color,
            'label': label,
            'lineWidth': line_width,
            'lineStyle': line_style
        })

    # Current price indicator
    current_price_indicator = {
        'fibPosition': float(gmm_zone_info.current_fib_position),
        'entryBias': gmm_zone_info.entry_bias,
        'sweepTypeFilter': gmm_zone_info.sweep_type_filter,
        'confidence': float(gmm_zone_info.confidence)
    }

    # Zone labels for chart overlay (positioned at mid-point of each zone)
    mid_time = (start_time + end_time) // 2
    zone_labels = [
        {
            'time': mid_time,
            'price': float((premium_top + premium_bottom) / 2),
            'text': 'PREMIUM',
            'color': zone_colors['premium']['border']
        },
        {
            'time': mid_time,
            'price': float(fib_prices.get(0.5, 0)),
            'text': 'EQUILIBRIUM',
            'color': '#ffeb3b'
        },
        {
            'time': mid_time,
            'price': float((discount_top + discount_bottom) / 2),
            'text': 'DISCOUNT',
            'color': zone_colors['discount']['border']
        }
    ]

    return {
        'zoneBands': zone_bands,
        'fibLevels': fib_levels,
        'currentPriceIndicator': current_price_indicator,
        'zoneLabels': zone_labels
    }


def generate_gmm_debug_data(gmm_zone_info, df_window: pd.DataFrame) -> Optional[Dict]:
    """
    Generate GMM debug visualization data for the debug tab.

    Creates data for Plotly visualization showing:
    - Left panel: Candlestick chart of window candles
    - Right panel: Histogram + fitted Gaussian curves
    - Stats panel: Component details and BIC scores

    Args:
        gmm_zone_info: GMMZoneInfo object with debug fields populated
        df_window: DataFrame with OHLCV data for the GMM window

    Returns:
        Dict with debug visualization data, or None if debug fields not populated
    """
    if gmm_zone_info is None:
        return None

    # Check if debug fields are populated
    if gmm_zone_info.price_levels is None or gmm_zone_info.bic_scores is None:
        return None

    # VALIDATION: Verify DataFrame matches price_levels to detect mismatches
    if len(df_window) > 0:
        import warnings
        df_price_min = df_window['low'].min()
        df_price_max = df_window['high'].max()
        price_levels_min = gmm_zone_info.price_levels.min()
        price_levels_max = gmm_zone_info.price_levels.max()
        # Allow 1% tolerance for floating point differences
        if abs(df_price_min - price_levels_min) > 0.01 * df_price_min:
            warnings.warn(
                f"GMM Debug data mismatch: candle min=${df_price_min:.2f}, "
                f"price_levels min=${price_levels_min:.2f}"
            )
        if abs(df_price_max - price_levels_max) > 0.01 * df_price_max:
            warnings.warn(
                f"GMM Debug data mismatch: candle max=${df_price_max:.2f}, "
                f"price_levels max=${price_levels_max:.2f}"
            )

    # Generate candlestick data for window
    candle_data = []
    for idx, row in df_window.iterrows():
        candle_data.append({
            'time': idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx),
            'timestamp': int(idx.timestamp()) if hasattr(idx, 'timestamp') else 0,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })

    # Generate histogram data
    # Create bins for the histogram
    price_levels = gmm_zone_info.price_levels
    num_bins = min(50, len(price_levels) // 20)  # Adaptive bin count
    num_bins = max(20, num_bins)  # At least 20 bins

    hist_counts, bin_edges = np.histogram(price_levels, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    histogram_data = {
        'counts': hist_counts.tolist(),
        'binCenters': bin_centers.tolist(),
        'binEdges': bin_edges.tolist()
    }

    # Generate Gaussian curve data for each component
    # Create smooth price range for plotting curves
    price_min = price_levels.min()
    price_max = price_levels.max()
    price_range = np.linspace(price_min, price_max, 200)

    gaussian_curves = []
    for i in range(gmm_zone_info.n_components):
        mean = gmm_zone_info.all_means[i]
        std = gmm_zone_info.all_stds[i]
        weight = gmm_zone_info.weights[i]

        # Calculate Gaussian PDF (scaled by weight and total count for histogram overlay)
        if std > 0:
            pdf = weight * np.exp(-0.5 * ((price_range - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            # Scale to match histogram
            bin_width = bin_edges[1] - bin_edges[0]
            pdf_scaled = pdf * len(price_levels) * bin_width
        else:
            pdf_scaled = np.zeros_like(price_range)

        gaussian_curves.append({
            'componentIndex': i,
            'mean': float(mean),
            'std': float(std),
            'weight': float(weight),
            'priceMin': gmm_zone_info.component_ranges[i][0],
            'priceMax': gmm_zone_info.component_ranges[i][1],
            'priceRange': price_range.tolist(),
            'pdfValues': pdf_scaled.tolist(),
            'isCurrent': i == gmm_zone_info.current_component
        })

    # Component statistics
    component_stats = []
    for i in range(gmm_zone_info.n_components):
        mean = gmm_zone_info.all_means[i]
        std = gmm_zone_info.all_stds[i]
        weight = gmm_zone_info.weights[i]

        # Get price range for this component (soft assignment)
        # Use confidence > 0.5 threshold to assign prices
        component_stats.append({
            'componentIndex': i,
            'mean': float(mean),
            'std': float(std),
            'weight': float(weight),
            'isCurrent': i == gmm_zone_info.current_component
        })

    # BIC curve data
    bic_data = {
        'components': list(range(1, len(gmm_zone_info.bic_scores) + 1)),
        'scores': [float(s) for s in gmm_zone_info.bic_scores],
        'selectedComponents': gmm_zone_info.n_components
    }

    # Window information
    window_info = {
        'startIdx': gmm_zone_info.window_start_idx,
        'endIdx': gmm_zone_info.window_end_idx,
        'candleCount': gmm_zone_info.window_candle_count,
        'startTime': candle_data[0]['time'] if candle_data else None,
        'endTime': candle_data[-1]['time'] if candle_data else None,
        'pricePointCount': len(price_levels)
    }

    # Current price position info
    current_price_info = {
        'price': float(gmm_zone_info.current_price) if gmm_zone_info.current_price else None,
        'fibPosition': float(gmm_zone_info.current_fib_position),
        'entryBias': gmm_zone_info.entry_bias,
        'sweepTypeFilter': gmm_zone_info.sweep_type_filter,
        'confidence': float(gmm_zone_info.confidence),
        'currentComponent': gmm_zone_info.current_component,
        'zoneTop': float(gmm_zone_info.zone_top),
        'zoneBottom': float(gmm_zone_info.zone_bottom),
        'selectionMethod': gmm_zone_info.selection_method
    }

    # Fib levels for reference lines
    fib_levels_data = []
    for level, price in sorted(gmm_zone_info.fib_prices.items(), reverse=True):
        fib_levels_data.append({
            'level': float(level),
            'price': float(price),
            'label': f'{level * 100:.1f}%'
        })

    return {
        'candleData': candle_data,
        'histogramData': histogram_data,
        'gaussianCurves': gaussian_curves,
        'componentStats': component_stats,
        'bicData': bic_data,
        'windowInfo': window_info,
        'currentPriceInfo': current_price_info,
        'fibLevels': fib_levels_data
    }
