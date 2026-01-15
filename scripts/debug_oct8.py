"""
Debug script to inspect Oct 8 data and indicators
"""
import sys
import os

# Add both parent and smartmoneyconcepts to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
from src.data_loader import TSLADataLoader
from smartmoneyconcepts.smc_custom import smc_custom

# Load data
loader = TSLADataLoader()
df_1h = loader.get_data('1h')
df_5m = loader.get_data('5min')

# Set index
df_1h = df_1h.set_index('time')
df_5m = df_5m.set_index('time')
df_1h.index = pd.to_datetime(df_1h.index)
df_5m.index = pd.to_datetime(df_5m.index)

# Calculate inflexions and BOS for 1H
inflexions_1h = smc_custom.inflexion_points(df_1h)
bos_1h = smc_custom.bos(df_1h, inflexions_1h, close_break=True)

# Calculate inflexions and BOS for 5M
inflexions_5m = smc_custom.inflexion_points(df_5m)
bos_5m = smc_custom.bos(df_5m, inflexions_5m, close_break=True)

# Focus on Oct 8, 2025
target_date = pd.Timestamp('2025-10-08')

print("\n" + "="*80)
print("1H DATA AROUND OCT 8, 2025 19:30")
print("="*80)

# Get Oct 8 19:30
oct8_1930 = pd.Timestamp('2025-10-08 19:30:00')
if oct8_1930 in df_1h.index:
    idx = df_1h.index.get_loc(oct8_1930)

    # Show window around it
    start_idx = max(0, idx - 5)
    end_idx = min(len(df_1h), idx + 6)

    for i in range(start_idx, end_idx):
        time = df_1h.index[i]
        o = df_1h['open'].iloc[i]
        h = df_1h['high'].iloc[i]
        l = df_1h['low'].iloc[i]
        c = df_1h['close'].iloc[i]

        inflx_type = inflexions_1h['InflexionType'].iloc[i]
        inflx_level = inflexions_1h['Level'].iloc[i]
        respected = inflexions_1h['Respected'].iloc[i]
        status_idx = inflexions_1h['StatusIndex'].iloc[i]

        trend = bos_1h['Trend'].iloc[i]
        trend_str = 'BULL' if trend == 1 else ('BEAR' if trend == -1 else 'NONE')

        marker = ">>> " if i == idx else "    "

        print(f"{marker}{time}  O:{o:7.2f} H:{h:7.2f} L:{l:7.2f} C:{c:7.2f}  Trend:{trend_str}")

        if not pd.isna(inflx_type):
            type_str = "PEAK" if inflx_type == 1 else "VALLEY"
            print(f"    └─ Inflexion: {type_str} @ {inflx_level:.2f}, Respected: {respected}, StatusIdx: {status_idx}")

print("\n" + "="*80)
print("5M DATA AROUND OCT 8, 2025 20:50")
print("="*80)

# Get Oct 8 20:50
oct8_2050 = pd.Timestamp('2025-10-08 20:50:00')
if oct8_2050 in df_5m.index:
    idx = df_5m.index.get_loc(oct8_2050)

    # Show window around it
    start_idx = max(0, idx - 10)
    end_idx = min(len(df_5m), idx + 11)

    for i in range(start_idx, end_idx):
        time = df_5m.index[i]
        o = df_5m['open'].iloc[i]
        h = df_5m['high'].iloc[i]
        l = df_5m['low'].iloc[i]
        c = df_5m['close'].iloc[i]

        inflx_type = inflexions_5m['InflexionType'].iloc[i]
        inflx_level = inflexions_5m['Level'].iloc[i]

        bos_val = bos_5m['BOS'].iloc[i]
        bos_level = bos_5m['Level'].iloc[i]
        trend = bos_5m['Trend'].iloc[i]
        trend_str = 'BULL' if trend == 1 else ('BEAR' if trend == -1 else 'NONE')

        marker = ">>> " if i == idx else "    "

        print(f"{marker}{time}  O:{o:7.2f} H:{h:7.2f} L:{l:7.2f} C:{c:7.2f}  Trend:{trend_str}", end="")

        if not pd.isna(bos_val):
            bos_str = "BOS_UP" if bos_val == 1 else "BOS_DN"
            print(f"  {bos_str} @ {bos_level:.2f}", end="")

        print()

        if not pd.isna(inflx_type):
            type_str = "PEAK" if inflx_type == 1 else "VALLEY"
            print(f"    └─ Inflexion: {type_str} @ {inflx_level:.2f}")

print("\n" + "="*80)
