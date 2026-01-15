"""
Find inflexions created BEFORE overlap that get swept WITHIN overlap
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

overlap_start = pd.Timestamp('2025-09-30 19:14:00')
overlap_end = pd.Timestamp('2025-10-17 17:30:00')

print("="*80)
print("FINDING CROSS-PERIOD SWEEPS")
print("="*80)
print(f"Inflexion BEFORE: {overlap_start}")
print(f"Sweep WITHIN: {overlap_start} to {overlap_end}")
print("="*80 + "\n")

cross_sweeps = []

for i in range(len(df_1h)):
    time = df_1h.index[i]

    # Only process inflexions created BEFORE overlap
    if time >= overlap_start:
        continue

    inflexion_type = inflexions['InflexionType'].iloc[i]
    respected = inflexions['Respected'].iloc[i]
    status_idx = inflexions['StatusIndex'].iloc[i]

    if not np.isnan(inflexion_type) and respected == True and status_idx > 0:
        sweep_idx = int(status_idx)
        sweep_time = df_1h.index[sweep_idx]

        # Check if sweep happens WITHIN overlap period
        if sweep_time >= overlap_start and sweep_time <= overlap_end:
            candles_between = sweep_idx - i
            type_str = "HIGH" if inflexion_type == 1 else "LOW"
            level = inflexions['Level'].iloc[i]

            cross_sweeps.append({
                'inflexion_time': time,
                'sweep_time': sweep_time,
                'type': type_str,
                'level': level,
                'candles_between': candles_between
            })

cross_sweeps.sort(key=lambda x: x['sweep_time'])

if len(cross_sweeps) > 0:
    print(f"Found {len(cross_sweeps)} cross-period sweeps:\n")
    for i, sweep in enumerate(cross_sweeps, 1):
        print(f"{i}. {sweep['type']} sweep at {sweep['sweep_time']}")
        print(f"   Inflexion created at: {sweep['inflexion_time']} (BEFORE overlap)")
        print(f"   Level: {sweep['level']:.2f}")
        print(f"   Candles between: {sweep['candles_between']}")
        print()

    print(f"\n✅ These sweeps are visible in the walkthrough but NOT in the strategy!")
    print(f"   The strategy only uses inflexions created within the overlap period.")
else:
    print("No cross-period sweeps found.")
