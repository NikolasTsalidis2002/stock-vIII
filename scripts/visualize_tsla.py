"""
Demo script to visualize TSLA with Smart Money Concepts indicators

Usage:
    python3 scripts/visualize_tsla.py                    # Use SMC library indicators (default)
    python3 scripts/visualize_tsla.py --indicator-type smc       # Use SMC library indicators
    python3 scripts/visualize_tsla.py --indicator-type custom    # Use custom indicators
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import TSLADataLoader
from visualizer import TSLAVisualizer


def main():
    """Generate TSLA visualizations with SMC indicators."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Visualize TSLA data with Smart Money Concepts indicators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/visualize_tsla.py                          # Default: SMC library indicators
  python3 scripts/visualize_tsla.py --indicator-type smc     # SMC library indicators
  python3 scripts/visualize_tsla.py --indicator-type custom  # Custom indicators (inflexions + BOS)
        """
    )
    parser.add_argument(
        '--indicator-type',
        type=str,
        choices=['smc', 'custom'],
        default='smc',
        help='Choose indicator type: "smc" for SMC library indicators, "custom" for custom inflexion-based indicators (default: smc)'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"TSLA SMART MONEY CONCEPTS VISUALIZATION - {args.indicator_type.upper()} INDICATORS")
    print("="*80 + "\n")

    # Initialize loader and visualizer
    loader = TSLADataLoader()
    viz = TSLAVisualizer()

    if args.indicator_type == 'smc':
        # Use SMC library indicators
        print("Using SMC Library indicators (FVG, Order Blocks, Swings, BOS/CHOCH, Liquidity)\n")

        # Example 1: 15min timeframe with all indicators
        print("[1/3] Generating 15min chart (last 500 candles)...")
        df_15min = loader.get_data('15min')
        fig_15min = viz.plot_all_indicators(
            df_15min,
            swing_length=50,
            previous_tf='1D',  # pandas format
            window_size=500
        )
        viz.save_html(fig_15min, 'tsla_15min_smc.html')
        print()

        # Example 2: 1h timeframe with selected indicators
        print("[2/3] Generating 1h chart (last 300 candles) - Selected indicators...")
        df_1h = loader.get_data('1h')
        fig_1h = viz.plot_all_indicators(
            df_1h,
            swing_length=30,
            previous_tf='1D',  # pandas format
            indicators=['fvg', 'ob', 'bos_choch', 'liquidity'],
            window_size=300
        )
        viz.save_html(fig_1h, 'tsla_1h_smc.html')
        print()

        # Example 3: 5min timeframe - recent data
        print("[3/3] Generating 5min chart (last 200 candles)...")
        df_5min = loader.get_data('5min')
        fig_5min = viz.plot_all_indicators(
            df_5min,
            swing_length=20,
            previous_tf='1H',  # pandas format
            window_size=200
        )
        viz.save_html(fig_5min, 'tsla_5min_smc.html')
        print()

        print("="*80)
        print("✅ VISUALIZATION COMPLETE!")
        print("="*80)
        print("\nGenerated charts:")
        print("  1. results/charts/tsla_15min_smc.html")
        print("  2. results/charts/tsla_1h_smc.html")
        print("  3. results/charts/tsla_5min_smc.html")
        print("\nOpen these files in your browser to view interactive charts!")
        print("="*80 + "\n")

    else:  # custom indicators
        # Use custom inflexion-based indicators
        print("Using Custom indicators (Inflexion Points + Trend-Aware BOS)\n")

        # Example 1: 15min timeframe with custom indicators
        print("[1/3] Generating 15min chart (last 500 candles)...")
        df_15min = loader.get_data('15min')
        fig_15min = viz.plot_custom_indicators(
            df_15min,
            close_break=True,
            window_size=500
        )
        viz.save_html(fig_15min, 'tsla_15min_custom.html')
        print()

        # Example 2: 1h timeframe with custom indicators
        print("[2/3] Generating 1h chart (last 300 candles)...")
        df_1h = loader.get_data('1h')
        fig_1h = viz.plot_custom_indicators(
            df_1h,
            close_break=True,
            window_size=300
        )
        viz.save_html(fig_1h, 'tsla_1h_custom.html')
        print()

        # Example 3: 5min timeframe - recent data
        print("[3/3] Generating 5min chart (last 200 candles)...")
        df_5min = loader.get_data('5min')
        fig_5min = viz.plot_custom_indicators(
            df_5min,
            close_break=True,
            window_size=200
        )
        viz.save_html(fig_5min, 'tsla_5min_custom.html')
        print()

        print("="*80)
        print("✅ VISUALIZATION COMPLETE!")
        print("="*80)
        print("\nGenerated charts:")
        print("  1. results/charts/tsla_15min_custom.html")
        print("  2. results/charts/tsla_1h_custom.html")
        print("  3. results/charts/tsla_5min_custom.html")
        print("\nOpen these files in your browser to view interactive charts!")
        print("\nColor coding for custom indicators:")
        print("  • Green triangles: Respected resistance (concave)")
        print("  • Blue triangles: Respected support (convex)")
        print("  • Red X: Disrespected resistance")
        print("  • Orange X: Disrespected support")
        print("  • Yellow triangles: Pending inflexions (not yet tested)")
        print("  • Background shading: Green = Bullish trend, Red = Bearish trend")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
