"""
1H Candle-by-Candle Walkthrough using TradingView Lightweight Charts

Generates a single HTML file with all frames embedded and dynamic navigation.

Usage:
    python3 scripts/analyze_1h_tv.py

Output:
    - results/1h_tv_walkthrough/walkthrough.html
"""

import sys
import os
import pandas as pd

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import TSLADataLoader
from src.tradingview_visualizer import TradingViewVisualizer


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("1H TIMEFRAME TRADINGVIEW ANALYSIS")
    print("="*80 + "\n")

    # Load all timeframes to determine overlap
    print("Loading TSLA data for all timeframes...")
    loader = TSLADataLoader()
    df_1h_full = loader.get_data('1h', force_refresh=False)
    df_5m_full = loader.get_data('5min', force_refresh=False)
    df_1m_full = loader.get_data('1min', force_refresh=False)

    print(f"  1H: {len(df_1h_full)} candles")
    print(f"  5M: {len(df_5m_full)} candles")
    print(f"  1M: {len(df_1m_full)} candles")

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

    print(f"\nData overlap period:")
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

    # Run analysis using unified visualizer
    visualizer = TradingViewVisualizer(
        timeframe_name="1H",
        output_subdir="1h_tv_walkthrough"
    )
    output_path = visualizer.run(df_1h)

    print("\n" + "="*80)
    print("WALKTHROUGH COMPLETE!")
    print("="*80)
    print(f"\nOpen: {output_path}")
    print("Use arrow keys or buttons to navigate between candles")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
