"""
Check which candle number (1-90) has the first liquidity sweep
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np
from src.data_loader import TSLADataLoader
from smartmoneyconcepts.smc_custom import smc_custom

# Load and filter data (EXACT same as strategy)
loader = TSLADataLoader()
df_1h_full = loader.get_data('1h')
df_5m_full = loader.get_data('5min')
df_1m_full = loader.get_data('1min')

# Determine overlap (EXACT same as walkthrough)
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

# Filter (EXACT same as walkthrough)
df_1h = df_1h_full[
    (pd.to_datetime(df_1h_full['time']) >= start_overlap) &
    (pd.to_datetime(df_1h_full['time']) <= end_overlap)
].copy()

df_1h = df_1h.set_index('time')
df_1h.index = pd.to_datetime(df_1h.index)

print(f"Filtered to {len(df_1h)} candles")
print(f"From {df_1h.index[0]} to {df_1h.index[-1]}")
print()

# Calculate indicators (EXACT same as walkthrough for each frame)
print("Checking liquidity sweeps at each candle (like walkthrough frames):\n")

for i in range(len(df_1h)):
    df_slice = df_1h.iloc[:i+1]  # Progressive slice (same as walkthrough)
    inflexions = smc_custom.inflexion_points(df_slice)

    # Count sweeps (EXACT same as walkthrough line 62)
    liquidity_sweeps = ((inflexions['Respected'] == True) & inflexions['InflexionType'].notna()).sum()

    if liquidity_sweeps > 0:
        candle_num = i + 1
        timestamp = df_1h.index[i]
        print(f"Candle {candle_num}/90:  {timestamp}  -  {liquidity_sweeps} sweep(s)")

        if candle_num == 17:
            print(f"\n  ^^^ This is the Oct 2, 19:30 frame you showed!")
            print(f"      At this point, {liquidity_sweeps} sweep(s) have been detected\n")
