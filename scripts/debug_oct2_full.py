"""
Check Oct 2 inflexion in FULL dataset vs FILTERED dataset
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np
from src.data_loader import TSLADataLoader
from smartmoneyconcepts.smc_custom import smc_custom

# Load FULL data (no filtering)
loader = TSLADataLoader()
df_1h_full = loader.get_data('1h')
df_1h_full = df_1h_full.set_index('time')
df_1h_full.index = pd.to_datetime(df_1h_full.index)

print("="*80)
print("FULL DATASET (5000 candles)")
print("="*80)

# Calculate inflexions on FULL dataset
inflexions_full = smc_custom.inflexion_points(df_1h_full)

# Find Oct 2, 19:30
target_time = pd.Timestamp('2025-10-02 19:30:00')
if target_time in df_1h_full.index:
    i = df_1h_full.index.get_loc(target_time)
    print(f"\nOct 2, 19:30 at index {i} in FULL dataset:")
    print(f"   OHLC: O={df_1h_full['open'].iloc[i]:.2f} H={df_1h_full['high'].iloc[i]:.2f} L={df_1h_full['low'].iloc[i]:.2f} C={df_1h_full['close'].iloc[i]:.2f}")

    inflexion_type = inflexions_full['InflexionType'].iloc[i]
    if not np.isnan(inflexion_type):
        type_str = "PEAK" if inflexion_type == 1 else "VALLEY"
        level = inflexions_full['Level'].iloc[i]
        respected = inflexions_full['Respected'].iloc[i]
        status_idx = inflexions_full['StatusIndex'].iloc[i]

        print(f"   ✅ INFLEXION: {type_str} @ {level:.2f}")
        print(f"   Respected: {respected}")
        print(f"   StatusIndex: {status_idx}")

        if status_idx > 0:
            candles_between = int(status_idx - i)
            print(f"   Candles between: {candles_between}")
            if candles_between >= 3:
                print(f"   ✅ PASSES 3-candle filter!")
    else:
        print(f"   ❌ NO inflexion")

# Now check FILTERED dataset
print("\n" + "="*80)
print("FILTERED DATASET (overlap period)")
print("="*80)

overlap_start = pd.Timestamp('2025-09-30 19:14:00')
overlap_end = pd.Timestamp('2025-10-17 17:30:00')
df_1h_filtered = df_1h_full.loc[overlap_start:overlap_end]

inflexions_filtered = smc_custom.inflexion_points(df_1h_filtered)

if target_time in df_1h_filtered.index:
    i = df_1h_filtered.index.get_loc(target_time)
    print(f"\nOct 2, 19:30 at index {i} in FILTERED dataset:")

    inflexion_type = inflexions_filtered['InflexionType'].iloc[i]
    if not np.isnan(inflexion_type):
        type_str = "PEAK" if inflexion_type == 1 else "VALLEY"
        level = inflexions_filtered['Level'].iloc[i]
        respected = inflexions_filtered['Respected'].iloc[i]
        status_idx = inflexions_filtered['StatusIndex'].iloc[i]

        print(f"   ✅ INFLEXION: {type_str} @ {level:.2f}")
        print(f"   Respected: {respected}")
        print(f"   StatusIndex: {status_idx}")
    else:
        print(f"   ❌ NO inflexion")

print("\n" + "="*80)
print("The inflexion detection NEEDS historical context!")
print("Filtering BEFORE calculating indicators removes that context.")
print("="*80)
