"""
5M Candle-by-Candle Walkthrough using TradingView Lightweight Charts

Generates a single HTML file with all frames embedded and dynamic navigation.

Usage:
    python3 scripts/analyze_5m_tv.py                    # Default: TSLA
    python3 scripts/analyze_5m_tv.py --symbol AAPL      # Use different symbol

Output:
    - results/{symbol}_5m_tv_walkthrough/walkthrough.html
"""

import sys
import os
import json
import argparse
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import DataLoader
from src.tradingview_visualizer import TradingViewVisualizer


def load_config():
    """Load configuration from JSON file."""
    config_path = Path(__file__).parent.parent / 'config' / 'strategy_config.json'
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            print(f"Loaded config from: {config_path}")
            return config
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in config file: {e}")
            return {}
    return {}


def main():
    """Main execution."""
    config = load_config()

    parser = argparse.ArgumentParser(
        description='5M Candle-by-Candle Walkthrough using TradingView Charts'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='TSLA',
        help='Stock symbol to analyze (default: TSLA)'
    )

    # Apply JSON config as defaults
    if config:
        parser.set_defaults(symbol=config.get('symbol', 'TSLA'))

    args = parser.parse_args()
    symbol = args.symbol.upper()

    print("\n" + "="*80)
    print(f"{symbol} 5M TIMEFRAME TRADINGVIEW ANALYSIS")
    print("="*80 + "\n")

    # Load all timeframes to determine overlap
    print(f"Loading {symbol} data for all timeframes...")
    loader = DataLoader(symbol=symbol)
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

    # Filter 5M to overlap period
    df_5m = df_5m_full[
        (pd.to_datetime(df_5m_full['time']) >= start_overlap) &
        (pd.to_datetime(df_5m_full['time']) <= end_overlap)
    ].copy()

    print(f"\n  Filtered 5M: {len(df_5m)} candles")

    # Set time as index
    if 'time' in df_5m.columns:
        df_5m = df_5m.set_index('time')

    # Run analysis using unified visualizer
    visualizer = TradingViewVisualizer(
        timeframe_name="5M",
        output_subdir=f"{symbol.lower()}_5m_tv_walkthrough"
    )
    output_path = visualizer.run(df_5m)

    print("\n" + "="*80)
    print("WALKTHROUGH COMPLETE!")
    print("="*80)
    print(f"\nOpen: {output_path}")
    print("Use arrow keys or buttons to navigate between candles")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
