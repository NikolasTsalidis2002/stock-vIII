# Frame-by-Frame Visualization Logic

## Overview

This document describes how data should be calculated and displayed across the three timeframes (1H, 5M, 1M) in the frame-by-frame walkthrough system.

**Key Principle**: Each timeframe only becomes active AFTER its parent timeframe's condition is met. Both calculation and display use the same windowed data slice.

---

## 1H Timeframe

### Status
✅ **Always Active** - This is the root timeframe

### Calculation Window
- **From**: Start of overlap period
- **To**: Current frame time
- **Data Used**: Full `df_1h` filtered to `current_time`

### Display Window
- **From**: Start of overlap period
- **To**: Current frame time
- **Data Shown**: Same as calculation window

### Indicators Calculated
- Inflexion Points (liquidity levels)
- BOS (Break of Structure)
- FVG (Fair Value Gaps)

### Code Pattern
```python
df_1h = self.strategy.df_1h[self.strategy.df_1h.index <= current_time]
df_1h_plot = df_1h  # No windowing needed

inflexions_1h = smc_custom.inflexion_points(df_1h_plot)
bos_1h = smc_custom.bos(df_1h_plot, inflexions_1h)
fvg_1h = smc.fvg(df_1h_plot)
```

---

## 5M Timeframe

### Status
⏸️ **Inactive** until 1H liquidity sweep occurs

### Activation Condition
- After `partial.timestamp_1h_sweep` has occurred

### Calculation Window (AFTER activation)
- **From**: `partial.timestamp_1h_sweep` (1H liquidity sweep time)
- **To**: Current frame time
- **Data Used**: `df_5m` filtered from sweep time → current time

### Display Window (AFTER activation)
- **From**: `partial.timestamp_1h_sweep`
- **To**: Current frame time
- **Data Shown**: Same as calculation window

### Indicators Calculated
- Inflexion Points
- BOS (Break of Structure)
- FVG (Fair Value Gaps)

### Code Pattern
```python
# Get data up to current time
df_5m = self.strategy.df_5m[self.strategy.df_5m.index <= current_time]

# Window from 1H sweep time (both calculation AND display)
start_ts_5m = partial.timestamp_1h_sweep
df_5m_plot = df_5m[df_5m.index >= start_ts_5m]

# Calculate on windowed data
inflexions_5m = smc_custom.inflexion_points(df_5m_plot)
bos_5m = smc_custom.bos(df_5m_plot, inflexions_5m)
fvg_5m = smc.fvg(df_5m_plot)
```

### Important Note
❌ **DO NOT** calculate on full `df_5m`
✅ **DO** calculate on `df_5m_plot` (windowed from sweep time)

This matches the v1.py pattern:
```python
# From v1.py - indicators calculated on slices
self.inf_5m._run_inflexion_handler(slice_5m)
```

---

## 1M Timeframe

### Status
⏸️ **Inactive** until 5M validation occurs

### Activation Condition
- After `partial.timestamp_5m_validation` has occurred

### Calculation Window (AFTER activation)
- **From**: `partial.timestamp_5m_validation` (5M validation time)
- **To**: Current frame time
- **Data Used**: `df_1m` filtered from validation time → current time

### Display Window (AFTER activation)
- **From**: `partial.timestamp_5m_validation`
- **To**: Current frame time
- **Data Shown**: Same as calculation window

### Indicators Calculated
- Inflexion Points
- BOS (Break of Structure)
- FVG (Fair Value Gaps)

### Code Pattern
```python
# Get data up to current time
df_1m = self.strategy.df_1m[self.strategy.df_1m.index <= current_time]

# Window from 5M validation time (both calculation AND display)
if partial.timestamp_5m_validation:
    start_ts_1m = partial.timestamp_5m_validation
    df_1m_plot = df_1m[df_1m.index >= start_ts_1m]
else:
    df_1m_plot = pd.DataFrame()  # Empty if validation hasn't occurred

# Calculate on windowed data
if len(df_1m_plot) > 0:
    inflexions_1m = smc_custom.inflexion_points(df_1m_plot)
    bos_1m = smc_custom.bos(df_1m_plot, inflexions_1m)
    fvg_1m = smc.fvg(df_1m_plot)
```

### Important Note
❌ **DO NOT** calculate on full `df_1m`
✅ **DO** calculate on `df_1m_plot` (windowed from validation time)

---

## Timeline Visualization

```
1H Timeline: [========================================] (full overlap data)
             ^                    ^
             Start                Liquidity Sweep (Oct 8 17:00)

5M Timeline:                      [======================] (from sweep time)
                                  ^             ^
                                  Start         Event B (17:30)

1M Timeline:                                    [========] (from validation)
                                                ^
                                                Start (validation time)
```

---

## Why This Matters

### Problem with Full Dataset Calculation
If we calculate indicators on the full dataset but only display a window:
- FVG boxes from September appear in October view
- Inflexion points from dates outside the visible range clutter the chart
- User sees indicators that have nothing to do with the current trade setup

### Solution with Windowed Calculation
By calculating on the same windowed slice we display:
- Only relevant indicators appear
- Chart is clean and focused on the active trade setup
- Matches the progressive analysis flow from v1.py

---

## Reference Implementation

See `src/animated_analyzer.py` in the `generate_frame()` method for the implementation of this logic.

Follows the pattern from `/Users/nikolastsalidis/Desktop/Projects/stock-v2/Strategies/v1.py`:
```python
# v1.py passes slices to handlers, not full datasets
slice_5m = m_slice.iloc[: m_idx + 1]
self.scan_5mI(slice_5m, ...)  # Calculates on slice

slice_1m = one_m_slice.iloc[: l_idx + 1]
self.scan_1m(slice_1m, ...)   # Calculates on slice
```

---

**Last Updated**: 2025-10-17
**Author**: Claude Code with user guidance
