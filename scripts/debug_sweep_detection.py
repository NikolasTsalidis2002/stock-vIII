"""
Debug liquidity sweep detection for Oct 8, 19:30
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np
from src.data_loader import TSLADataLoader
from smartmoneyconcepts.smc_custom import smc_custom

# Load and filter data
loader = TSLADataLoader()
df_1h = loader.get_data('1h')
df_1h = df_1h.set_index('time')
df_1h.index = pd.to_datetime(df_1h.index)

# Get overlap period (same logic as strategy)
overlap_start = pd.Timestamp('2025-09-30 19:10:00')
overlap_end = pd.Timestamp('2025-10-17 17:30:00')
df_1h = df_1h.loc[overlap_start:overlap_end]

# Calculate indicators
inflexions_1h = smc_custom.inflexion_points(df_1h)

print("\n" + "="*80)
print("DEBUGGING LIQUIDITY SWEEP DETECTION")
print("="*80)

# Find Oct 8, 19:30
target_time = pd.Timestamp('2025-10-08 19:30:00')
if target_time not in df_1h.index:
    print(f"\nERROR: {target_time} not in dataframe!")
else:
    i = df_1h.index.get_loc(target_time)

    print(f"\n📍 Checking candle at index {i}: {target_time}")
    print(f"   OHLC: O={df_1h['open'].iloc[i]:.2f} H={df_1h['high'].iloc[i]:.2f} L={df_1h['low'].iloc[i]:.2f} C={df_1h['close'].iloc[i]:.2f}")

    inflexion_type = inflexions_1h['InflexionType'].iloc[i]
    level = inflexions_1h['Level'].iloc[i]
    respected = inflexions_1h['Respected'].iloc[i]

    if not np.isnan(inflexion_type):
        type_str = "PEAK (concave)" if inflexion_type == 1 else "VALLEY (convex)"
        print(f"   Inflexion: {type_str} @ {level:.2f}, Respected: {respected}")

        # Check next 20 candles manually
        print(f"\n🔍 Checking next 20 candles for sweeps of level {level:.2f}:")

        min_candles_later = 3
        swept = False
        respected_manual = True

        for j in range(i + min_candles_later, min(i + 21, len(df_1h))):
            time_j = df_1h.index[j]
            candle_high = df_1h['high'].iloc[j]
            candle_low = df_1h['low'].iloc[j]
            candle_close = df_1h['close'].iloc[j]
            candle_open = df_1h['open'].iloc[j]

            if inflexion_type == 1:  # Checking for sweep above
                if candle_high >= level:
                    swept = True
                    broke = (candle_close > level) or (candle_open > level)
                    print(f"   [{j - i}] {time_j}: H={candle_high:.2f} TOUCHED! Close={candle_close:.2f} Open={candle_open:.2f} Broke={broke}")

                    if broke:
                        respected_manual = False
                        print(f"      ❌ Level was BROKEN (close/open above level)")
                        break
                    else:
                        print(f"      ✅ Level was RESPECTED (touched but didn't close above)")
                        break

        if not swept:
            print(f"   ❌ Level {level:.2f} was NEVER touched in the next {min(20, len(df_1h) - i - min_candles_later)} candles")
        elif swept and respected_manual:
            print(f"\n✅ LIQUIDITY SWEEP DETECTED!")
        elif swept and not respected_manual:
            print(f"\n❌ Level was touched but BROKEN, not a valid sweep")
    else:
        print(f"   No inflexion at this candle")
