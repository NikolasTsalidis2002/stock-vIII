"""
Compare SMC Library vs Custom Indicators
Side-by-side comparison on TSLA data
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import TSLADataLoader
from smartmoneyconcepts.smc_custom import smc_custom
from smartmoneyconcepts.smc import smc
import pandas as pd


def compare_indicators(timeframe='15min', window_size=500):
    """
    Compare custom indicators vs SMC library indicators.

    Args:
        timeframe: Data timeframe to analyze
        window_size: Number of recent candles to analyze
    """
    print("\n" + "="*80)
    print(f"SMC vs CUSTOM INDICATORS COMPARISON - TSLA {timeframe}")
    print("="*80)

    # Load data
    loader = TSLADataLoader()
    df = loader.get_data(timeframe)

    # Use recent window
    if window_size:
        df = df.iloc[-window_size:].reset_index(drop=True)

    df_indexed = df.set_index('time')

    print(f"\nDataset: {len(df)} candles")
    print(f"Period: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # ==============================================
    # PART 1: Pivot Detection Comparison
    # ==============================================
    print("\n" + "-"*80)
    print("PART 1: PIVOT DETECTION COMPARISON")
    print("-"*80)

    # Custom inflexion points
    print("\n[Custom] Detecting inflexion points (3-point local extrema)...")
    inflexions = smc_custom.inflexion_points(df_indexed)

    # SMC swing highs/lows
    print("[SMC] Detecting swing highs/lows (window-based, length=50)...")
    swings = smc.swing_highs_lows(df_indexed, swing_length=50)

    # Statistics
    custom_total = inflexions["InflexionType"].notna().sum()
    custom_concave = (inflexions['InflexionType'] == 1).sum()
    custom_convex = (inflexions['InflexionType'] == -1).sum()

    smc_total = swings["HighLow"].notna().sum()
    smc_highs = (swings['HighLow'] == 1).sum()
    smc_lows = (swings['HighLow'] == -1).sum()

    print(f"\n{'Metric':<30} {'Custom':>15} {'SMC':>15} {'Difference':>15}")
    print("-"*80)
    print(f"{'Total Pivots':<30} {custom_total:>15} {smc_total:>15} {custom_total-smc_total:>15}")
    print(f"{'Peaks/Highs':<30} {custom_concave:>15} {smc_highs:>15} {custom_concave-smc_highs:>15}")
    print(f"{'Valleys/Lows':<30} {custom_convex:>15} {smc_lows:>15} {custom_convex-smc_lows:>15}")

    # Respect/Disrespect Analysis
    custom_respected = (inflexions['Respected'] == True).sum()
    custom_disrespected = (inflexions['Respected'] == False).sum()
    custom_pending = inflexions['Respected'].isna().sum()

    print(f"\n{'Custom Respect Status':<30} {'Count':>15} {'Percentage':>15}")
    print("-"*80)
    print(f"{'Respected (held)':<30} {custom_respected:>15} {custom_respected/custom_total*100:>14.1f}%")
    print(f"{'Disrespected (broken)':<30} {custom_disrespected:>15} {custom_disrespected/custom_total*100:>14.1f}%")
    print(f"{'Pending (not reached)':<30} {custom_pending:>15} {custom_pending/len(df):>14.1f}%")

    # ==============================================
    # PART 2: BOS Detection Comparison
    # ==============================================
    print("\n" + "-"*80)
    print("PART 2: BREAK OF STRUCTURE COMPARISON")
    print("-"*80)

    # Custom BOS
    print("\n[Custom] Detecting BOS (trend-aware, close/open confirmation)...")
    custom_bos = smc_custom.bos(df_indexed, inflexions, close_break=True)

    # SMC BOS
    print("[SMC] Detecting BOS (pattern-based)...")
    smc_bos = smc.bos_choch(df_indexed, swings, close_break=True)

    # Statistics
    custom_bos_total = custom_bos["BOS"].notna().sum()
    custom_bullish = (custom_bos['BOS'] == 1).sum()
    custom_bearish = (custom_bos['BOS'] == -1).sum()

    smc_bos_total = smc_bos["BOS"].notna().sum()
    smc_bullish = (smc_bos['BOS'] == 1).sum()
    smc_bearish = (smc_bos['BOS'] == -1).sum()

    print(f"\n{'Metric':<30} {'Custom':>15} {'SMC':>15} {'Difference':>15}")
    print("-"*80)
    print(f"{'Total BOS':<30} {custom_bos_total:>15} {smc_bos_total:>15} {custom_bos_total-smc_bos_total:>15}")
    print(f"{'Bullish BOS':<30} {custom_bullish:>15} {smc_bullish:>15} {custom_bullish-smc_bullish:>15}")
    print(f"{'Bearish BOS':<30} {custom_bearish:>15} {smc_bearish:>15} {custom_bearish-smc_bearish:>15}")

    # Trend Analysis (Custom only)
    current_trend = custom_bos['Trend'].iloc[-1]
    trend_name = 'Bullish' if current_trend == 1 else ('Bearish' if current_trend == -1 else 'Undefined')
    bullish_candles = (custom_bos['Trend'] == 1).sum()
    bearish_candles = (custom_bos['Trend'] == -1).sum()
    undefined_candles = (custom_bos['Trend'] == 0).sum()

    print(f"\n{'Custom Trend Analysis':<30} {'Candles':>15} {'Percentage':>15}")
    print("-"*80)
    print(f"{'Bullish periods':<30} {bullish_candles:>15} {bullish_candles/len(df)*100:>14.1f}%")
    print(f"{'Bearish periods':<30} {bearish_candles:>15} {bearish_candles/len(df)*100:>14.1f}%")
    print(f"{'Undefined periods':<30} {undefined_candles:>15} {undefined_candles/len(df)*100:>14.1f}%")
    print(f"\n{'Current Trend:':<30} {trend_name:>15}")

    # ==============================================
    # PART 3: Key Insights
    # ==============================================
    print("\n" + "-"*80)
    print("PART 3: KEY INSIGHTS")
    print("-"*80)

    print("\n✓ PIVOT DETECTION:")
    if custom_total > smc_total:
        ratio = custom_total / smc_total if smc_total > 0 else float('inf')
        print(f"  - Custom detects {ratio:.1f}x MORE pivots than SMC")
        print(f"  - Custom captures ALL local extrema (complete market structure)")
        print(f"  - SMC filters to major swings only (cleaner but less complete)")

    print("\n✓ RESPECT/DISRESPECT TRACKING:")
    print(f"  - {custom_respected} inflexions respected (key support/resistance zones)")
    print(f"  - {custom_disrespected} inflexions broken (historical filtering)")
    print(f"  - Custom provides level effectiveness data (SMC doesn't track this)")

    print("\n✓ BOS DETECTION:")
    if custom_bos_total > smc_bos_total:
        ratio = custom_bos_total / smc_bos_total if smc_bos_total > 0 else float('inf')
        print(f"  - Custom detects {ratio:.1f}x MORE BOS than SMC")
        print(f"  - Custom uses trend-aware logic with explicit state tracking")
        print(f"  - SMC uses 4-swing pattern matching (simpler but less flexible)")

    print("\n✓ TREND AWARENESS:")
    print(f"  - Custom explicitly tracks trend state ({trend_name} currently)")
    print(f"  - Provides trend duration and composition data")
    print(f"  - SMC has no explicit trend tracking")

    # ==============================================
    # PART 4: Recommendations
    # ==============================================
    print("\n" + "-"*80)
    print("PART 4: RECOMMENDATIONS")
    print("-"*80)

    print("\n📊 USE CUSTOM INDICATORS WHEN:")
    print("  ✓ You need complete market structure analysis")
    print("  ✓ You want to track level effectiveness (respect/disrespect)")
    print("  ✓ You need explicit trend state for strategy logic")
    print("  ✓ You want fewer missed signals (catches all pivots)")
    print("  ✓ You're trading lower timeframes (1m, 5m, 15m)")

    print("\n📊 USE SMC INDICATORS WHEN:")
    print("  ✓ You prefer simpler, cleaner charts")
    print("  ✓ You only care about major structure")
    print("  ✓ You're trading higher timeframes (4H, 1D)")
    print("  ✓ You want to filter out market noise")
    print("  ✓ You're a beginner learning structure concepts")

    print("\n💡 HYBRID APPROACH (RECOMMENDED):")
    print("  1. Use Custom inflexions as foundation")
    print("  2. Filter by significance (like SMC swing_length)")
    print("  3. Apply Custom BOS logic on filtered inflexions")
    print("  4. Get precision + noise filtering")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80 + "\n")

    return {
        'inflexions': inflexions,
        'swings': swings,
        'custom_bos': custom_bos,
        'smc_bos': smc_bos,
        'df': df
    }


def main():
    """Run comparison analysis."""

    # Compare on 15min data
    print("\n" + "="*80)
    print("TSLA SMART MONEY CONCEPTS INDICATOR COMPARISON")
    print("="*80)

    results = compare_indicators(timeframe='15min', window_size=500)

    print("\n💾 Results saved in memory. To visualize:")
    print("  - Use results['inflexions'] for custom inflexion data")
    print("  - Use results['swings'] for SMC swing data")
    print("  - Use results['custom_bos'] for custom BOS data")
    print("  - Use results['smc_bos'] for SMC BOS data")
    print("  - Use results['df'] for raw OHLC data")

    return results


if __name__ == "__main__":
    results = main()
