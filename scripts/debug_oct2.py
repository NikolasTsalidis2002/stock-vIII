"""
Debug Oct 2nd 19:30 liquidity sweep detection
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np
from src.data_loader import TSLADataLoader
from smartmoneyconcepts.smc_custom import smc_custom

# Load and filter data (same as strategy)
loader = TSLADataLoader()
df_1h = loader.get_data('1h')
df_1h = df_1h.set_index('time')
df_1h.index = pd.to_datetime(df_1h.index)

# Get overlap period (same as strategy)
overlap_start = pd.Timestamp('2025-09-30 19:14:00')
overlap_end = pd.Timestamp('2025-10-17 17:30:00')
df_1h = df_1h.loc[overlap_start:overlap_end]

# Calculate indicators
inflexions_1h = smc_custom.inflexion_points(df_1h)
bos_1h = smc_custom.bos(df_1h, inflexions_1h, close_break=True)

print("\n" + "="*80)
print("DEBUGGING OCT 2, 2025 19:30 LIQUIDITY SWEEP")
print("="*80)

# Find Oct 2, 19:30
target_time = pd.Timestamp('2025-10-02 19:30:00')
if target_time not in df_1h.index:
    print(f"\nERROR: {target_time} not in dataframe!")
    print("Available times around Oct 2:")
    oct2 = df_1h.loc['2025-10-02']
    for t in oct2.index:
        print(f"  {t}")
else:
    i = df_1h.index.get_loc(target_time)

    print(f"\n📍 Oct 2, 19:30 at index {i}:")
    print(f"   OHLC: O={df_1h['open'].iloc[i]:.2f} H={df_1h['high'].iloc[i]:.2f} L={df_1h['low'].iloc[i]:.2f} C={df_1h['close'].iloc[i]:.2f}")

    # Check inflexion
    inflexion_type = inflexions_1h['InflexionType'].iloc[i]
    level = inflexions_1h['Level'].iloc[i]
    respected = inflexions_1h['Respected'].iloc[i]
    status_idx = inflexions_1h['StatusIndex'].iloc[i]

    if not np.isnan(inflexion_type):
        type_str = "PEAK (concave)" if inflexion_type == 1 else "VALLEY (convex)"
        print(f"\n✅ INFLEXION FOUND:")
        print(f"   Type: {type_str}")
        print(f"   Level: {level:.2f}")
        print(f"   Respected: {respected}")
        print(f"   StatusIndex: {status_idx}")

        if status_idx > 0:
            candles_between = int(status_idx - i)
            print(f"   Candles between inflexion and status determination: {candles_between}")

            if candles_between >= 3:
                print(f"\n✅ PASSES 3-CANDLE MINIMUM FILTER!")
                print(f"   Would be reported as sweep at index {int(status_idx)}: {df_1h.index[int(status_idx)]}")
            else:
                print(f"\n❌ FAILS 3-CANDLE MINIMUM FILTER (only {candles_between} candles)")
    else:
        print(f"\n❌ NO INFLEXION at this candle")

    # Check trend
    trend = bos_1h['Trend'].iloc[i]
    trend_str = 'BULLISH' if trend == 1 else ('BEARISH' if trend == -1 else 'NONE')
    print(f"\n   1H Trend: {trend_str}")

# Now check ALL inflexions with their respected status
print("\n" + "="*80)
print("ALL RESPECTED INFLEXIONS IN DATA:")
print("="*80)

for i in range(len(inflexions_1h)):
    inflexion_type = inflexions_1h['InflexionType'].iloc[i]
    respected = inflexions_1h['Respected'].iloc[i]
    status_idx = inflexions_1h['StatusIndex'].iloc[i]

    if not np.isnan(inflexion_type) and respected == True:
        type_str = "PEAK" if inflexion_type == 1 else "VALLEY"
        level = inflexions_1h['Level'].iloc[i]
        time = df_1h.index[i]

        candles_between = int(status_idx - i) if status_idx > 0 else 0
        sweep_time = df_1h.index[int(status_idx)] if status_idx > 0 and int(status_idx) < len(df_1h) else "N/A"

        passes_filter = candles_between >= 3

        print(f"[{'✅' if passes_filter else '❌'}] {time} ({type_str} @ {level:.2f}) - Swept {candles_between} candles later at {sweep_time}")
