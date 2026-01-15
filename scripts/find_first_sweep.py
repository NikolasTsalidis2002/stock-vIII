"""
Find the FIRST liquidity sweep in the overlap period
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np
from src.data_loader import TSLADataLoader
from smartmoneyconcepts.smc_custom import smc_custom

# Load FULL data
loader = TSLADataLoader()
df_1h = loader.get_data('1h')
df_1h = df_1h.set_index('time')
df_1h.index = pd.to_datetime(df_1h.index)

# Calculate inflexions on FULL dataset
inflexions = smc_custom.inflexion_points(df_1h)

# Filter to overlap period
overlap_start = pd.Timestamp('2025-09-30 19:14:00')
overlap_end = pd.Timestamp('2025-10-17 17:30:00')

print("="*80)
print("FINDING FIRST LIQUIDITY SWEEP IN OVERLAP PERIOD")
print(f"Period: {overlap_start} to {overlap_end}")
print("="*80)

sweeps_found = []

for i in range(len(df_1h)):
    time = df_1h.index[i]

    # Skip if outside overlap period
    if time < overlap_start or time > overlap_end:
        continue

    inflexion_type = inflexions['InflexionType'].iloc[i]
    respected = inflexions['Respected'].iloc[i]
    status_idx = inflexions['StatusIndex'].iloc[i]

    if not np.isnan(inflexion_type) and respected == True and status_idx > 0:
        sweep_idx = int(status_idx)
        sweep_time = df_1h.index[sweep_idx]

        # Check if sweep also within overlap period
        if sweep_time >= overlap_start and sweep_time <= overlap_end:
            candles_between = sweep_idx - i

            if candles_between >= 3:
                type_str = "HIGH" if inflexion_type == 1 else "LOW"
                level = inflexions['Level'].iloc[i]

                sweeps_found.append({
                    'inflexion_time': time,
                    'sweep_time': sweep_time,
                    'type': type_str,
                    'level': level,
                    'candles_between': candles_between
                })

# Sort by sweep time
sweeps_found.sort(key=lambda x: x['sweep_time'])

print(f"\nFound {len(sweeps_found)} valid sweeps in overlap period:\n")

for i, sweep in enumerate(sweeps_found, 1):
    print(f"{i}. {sweep['type']} sweep at {sweep['sweep_time']}")
    print(f"   Inflexion created at: {sweep['inflexion_time']}")
    print(f"   Level: {sweep['level']:.2f}")
    print(f"   Candles between: {sweep['candles_between']}")
    print()

if len(sweeps_found) > 0:
    print(f"\n✅ FIRST SWEEP: {sweeps_found[0]['type']} at {sweeps_found[0]['sweep_time']}")
    print(f"   (Inflexion was created at {sweeps_found[0]['inflexion_time']})")
