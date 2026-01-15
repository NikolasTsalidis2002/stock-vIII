"""
Find all liquidity sweeps around Oct 1-3 in FULL dataset
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

print("="*80)
print("ALL RESPECTED INFLEXIONS (LIQUIDITY SWEEPS) AROUND OCT 1-3, 2025")
print("="*80)

# Filter to Oct 1-3 period
start = pd.Timestamp('2025-10-01')
end = pd.Timestamp('2025-10-04')
df_oct = df_1h.loc[start:end]

for time in df_oct.index:
    i = df_1h.index.get_loc(time)

    inflexion_type = inflexions['InflexionType'].iloc[i]
    respected = inflexions['Respected'].iloc[i]

    if not np.isnan(inflexion_type):
        type_str = "PEAK" if inflexion_type == 1 else "VALLEY"
        level = inflexions['Level'].iloc[i]
        status_idx = inflexions['StatusIndex'].iloc[i]

        # Show ALL inflexions, mark which are respected (sweeps)
        marker = "★" if respected == True else ("✕" if respected == False else "●")
        status = "SWEEP" if respected == True else ("BROKEN" if respected == False else "PENDING")

        print(f"{marker} {time}  {type_str} @ {level:.2f}  [{status}]", end="")

        if status_idx > 0:
            status_time = df_1h.index[int(status_idx)]
            candles = int(status_idx - i)
            print(f"  (Status at {status_time}, {candles} candles later)")
        else:
            print()

print("\n" + "="*80)
print("Legend: ★ = Respected (Sweep), ✕ = Broken, ● = Pending")
print("="*80)
