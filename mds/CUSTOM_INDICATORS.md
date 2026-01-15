# Custom Indicators Documentation
## From stock-v2 Project

This document describes the custom trading indicators developed in the stock-v2 project, which offer more sophisticated market structure analysis compared to the standard Smart Money Concepts library.

---

## Table of Contents
1. [Inflexion Points](#inflexion-points)
2. [Break of Structure (BOS)](#break-of-structure-bos)
3. [Fair Value Gaps (FVG)](#fair-value-gaps-fvg)
4. [Key Differences from SMC Library](#key-differences)

---

## 1. Inflexion Points

### Overview
**File:** `Tools/inflexion.py`

Inflexion points are **local extrema** in price action - peaks (concave) and valleys (convex) that represent significant turning points where price changes direction.

### Types
- **Concave Inflexions** (Peaks): Local maxima using high prices
- **Convex Inflexions** (Valleys): Local minima using low prices

### Detection Algorithm

#### Mathematical Condition
For a 3-point comparison at index `i`:

**Concave (Peak):**
```
high[i] >= high[i-1] AND high[i] > high[i+1]
```

**Convex (Valley):**
```
low[i] <= low[i-1] AND low[i] < low[i+1]
```

### Respect/Disrespect Logic

An inflexion point can be in one of three states:

1. **Respected** (`respected = True`)
   - Price approached the level but did not close beyond it
   - For support (convex): Price wicked down but closed above
   - For resistance (concave): Price wicked up but closed below

2. **Disrespected** (`respected = False`)
   - Price **closed or opened** beyond the inflexion level
   - Once disrespected, the inflexion is marked historically and filtered out
   - For support: `close < level OR open < level`
   - For resistance: `close > level OR open > level`

3. **Pending** (`respected = None`)
   - Not yet reached by price
   - Still active as potential support/resistance

### Key Features

1. **Relative Position Tracking**
   - `above`: Inflexion acts as resistance
   - `below`: Inflexion acts as support
   - `equal`: Inflexion at current price

2. **Historical Filtering**
   - Disrespected inflexions stored in `historically_disrespected_inflexions`
   - Prevents re-detection of broken levels
   - Keeps chart clean and focused on valid support/resistance

3. **Time-Based Keys**
   - Each inflexion identified by: `f"{time}_{type}"`
   - Allows tracking same timestamp with different types

### Code Flow (`Tools/inflexion.py`)

```python
# Detection (lines 82-140)
1. Calculate local extrema using 3-point comparison
2. Check if inflexion was historically disrespected → skip if yes
3. If new inflexion → create Inflexion object
4. Evaluate status (respected/disrespected/pending)
5. Store in inflexion_points list and inflexion_times dict

# Status Evaluation (lines 45-79)
1. Get all future prices after inflexion
2. Check disrespect conditions (close/open beyond level)
3. If disrespected → mark False, add to historical list
4. Check respect conditions (wick touched but didn't close beyond)
5. If respected → mark True
6. Else → mark None (pending)
```

### Data Structure

```python
class Inflexion:
    time: datetime              # Timestamp
    idx: int                    # Index in DataFrame
    time_interval: str          # '15min', '1h', etc.
    price: float                # Level price
    relative_price_position: str # 'above', 'below', 'equal'
    type_: str                  # 'concave' or 'convex'
    respected: bool|None        # True/False/None
    object_type: str           # 'inflection'
```

---

## 2. Break of Structure (BOS)

### Overview
**File:** `Tools/bos.py`

BOS detects when price **breaks through a significant inflexion level**, signaling a **trend change**. This is more sophisticated than typical swing break detection.

### Core Concept

A Break of Structure occurs when:
1. Price is in a defined trend (bullish or bearish)
2. Price closes/opens **beyond** the last significant inflexion in the **opposite direction**
3. This signals a potential trend reversal

### Trend Logic

#### Bullish Trend
- Higher Highs (HH): Each concave inflexion > previous concave inflexion
- Higher Lows (HL): Each convex inflexion > previous convex inflexion
- **BOS when**: Price closes below `last_trend_min` (the lowest convex in the trend)

#### Bearish Trend
- Lower Highs (LH): Each concave inflexion < previous concave inflexion
- Lower Lows (LL): Each convex inflexion < previous convex inflexion
- **BOS when**: Price closes above `last_trend_max` (the highest concave in the trend)

### Detection Algorithm

```python
# From lines 66-129 in bos.py

# Step 1: Check for BOS
if trend == "bearish" and last_trend_max exists:
    if close > last_trend_max.price OR open > last_trend_max.price:
        → BOS detected
        → Switch trend to "bullish"
        → Update last_trend_min and last_trend_max

if trend == "bullish" and last_trend_min exists:
    if close < last_trend_min.price OR open < last_trend_min.price:
        → BOS detected
        → Switch trend to "bearish"
        → Update last_trend_max and last_trend_min
```

### Three-Phase System

#### Phase 1: Seed First Pivots (lines 161-167)
```python
# Establish initial reference points
- Wait for first concave inflexion → set as last_trend_max
- Wait for first convex inflexion → set as last_trend_min
```

#### Phase 2: Establish Initial Trend (lines 169-198)
```python
# Determine first trend direction
If last_trend_max exists:
    If new concave < last_trend_max → trend = "bearish"
    If new concave > last_trend_max → trend = "bullish"

If last_trend_min exists:
    If new convex > last_trend_min → trend = "bullish"
    If new convex < last_trend_min → trend = "bearish"

Create first Trend object with start_idx
```

#### Phase 3: Trend Maintenance (lines 200-226)
```python
# Update trend pivots based on current trend

Bearish Trend:
    - New concave < last_trend_max → update last_trend_max
    - New convex < last_trend_min → update last_trend_min

Bullish Trend:
    - New concave > last_trend_max → update last_trend_max
    - New convex > last_trend_min → update last_trend_min
```

### Key State Variables

```python
last_trend_max      # Highest concave in current trend
last_trend_min      # Lowest convex in current trend
last_max            # Most recent concave (may not be trend-significant)
last_min            # Most recent convex (may not be trend-significant)
trend               # Current trend: "bullish", "bearish", or None
```

### Data Structures

```python
class BOS:
    time: datetime              # Time of inflexion that was broken
    idx: int                    # Index of broken inflexion
    time_interval: str          # Timeframe
    broken_price: float         # Price level that was broken
    relative_price_position: str # 'above' or 'below'
    broken_trend: str           # 'bullish' or 'bearish'
    broken_idx: int            # Index where BOS occurred
    object_type: str           # 'bos'
    respected: bool            # Always False (BOS = disrespect)

class Trend:
    type_: str                  # 'bullish' or 'bearish'
    idx_start: int              # Start of trend
    idx_end: int               # End of trend (None if ongoing)
    n_convex_inflexs: int       # Count of valleys in trend
    n_concave_inflexs: int      # Count of peaks in trend
```

### Critical Difference from SMC

**Your BOS:**
- Based on **inflexion points** (statistically significant local extrema)
- Requires **closing/opening price** to break the level (more confirmative)
- Tracks **trend state** explicitly with pivot updates
- Distinguishes between `last_trend_max/min` (significant) and `last_max/min` (recent)

**SMC BOS:**
- Based on **swing highs/lows** with fixed lookback period
- Any break (not just close/open) can trigger
- Simpler logic without sophisticated trend tracking
- No distinction between trend-significant and recent pivots

---

## 3. Fair Value Gaps (FVG)

### Overview
**File:** `Tools/fairValueGaps.py`

Fair Value Gaps identify **price imbalances** where a candle's body creates a gap between the previous and next candle's wicks.

### Detection Logic

#### For FVG Above Current Price
```python
# Looking backward (lines 46-51)
shifted_highs = high[i-2]  # High from 2 candles ago
current_low = low[i]        # Current low

if current_low > shifted_highs[i-2]:
    → FVG exists at index i-1 (middle candle)
```

#### For FVG Below Current Price
```python
# Looking forward (lines 46-51)
shifted_highs = high[i+2]  # High from 2 candles ahead
current_low = low[i]        # Current low

if current_low > shifted_highs[i+2]:
    → FVG exists at index i+1 (middle candle)
```

### Gap Boundaries Calculation

The middle candle (i) must have its body **within the gap**:

```python
# Lines 75-104
low = high_prices[i-1] or low_prices[i+1]  # depending on direction
high = low_prices[i-1] or high_prices[i+1] # depending on direction

body_top = max(open[i], close[i])
body_bottom = min(open[i], close[i])

# Candle body must be inside the gap
if body_bottom > low or body_top < high:
    skip  # Not a valid FVG

# Calculate actual gap boundaries
if fvg_type == 'above':
    top = min(open[i], original_top)
    bottom = max(close[i], original_bottom)
else:
    top = max(open[i], original_top)
    bottom = min(close[i], original_bottom)
```

### Respect/Disrespect Logic

Similar to inflexions, FVGs track respect status:

```python
# Lines 113-146

# For FVG above (resistance):
disrespected = (close > price_top) | (open > price_top)
respected = high > price_bottom  # touched but didn't close through

# For FVG below (support):
disrespected = (close < price_bottom) | (open < price_bottom)
respected = low < price_top  # touched but didn't close through
```

### Status States

1. **Respected** (`respected = True`)
   - Price entered the gap but didn't close through it
   - Gap acted as intended support/resistance

2. **Disrespected** (`respected = False`)
   - Price closed/opened completely through the gap
   - Gap failed as support/resistance
   - Marked with `end_idx` showing when it was violated

3. **Pending** (`respected = None`)
   - Gap not yet reached by price
   - Still active

### Data Structure

```python
class FairValueGap:
    time: datetime              # Timestamp
    idx: int                    # Middle candle index
    time_interval: str          # Timeframe
    price_top: float            # Upper boundary
    price_bottom: float         # Lower boundary
    relative_price_position: str # 'above' or 'below'
    respected: bool|None        # Status
    end_idx: int               # When disrespected (None if active)
    object_type: str           # 'fvg'
```

---

## 4. Key Differences from SMC Library

### Philosophical Approach

| Aspect | Your Custom Indicators | SMC Library |
|--------|----------------------|-------------|
| **Foundation** | Inflexion points (local extrema) | Swing highs/lows (fixed lookback) |
| **Precision** | Requires close/open to break levels | Any price breach counts |
| **State Tracking** | Explicit respect/disrespect status | Binary existence (mitigated or not) |
| **Trend Definition** | Sequence-based with pivot tracking | Structure break based |
| **Historical Memory** | Filters disrespected levels | Keeps all, marks mitigated |

### BOS Comparison

#### Your Custom BOS
```python
# Advantages:
✓ Uses statistically significant inflexion points
✓ Requires close/open confirmation (reduces false signals)
✓ Explicit trend state machine
✓ Distinguishes trend-significant vs recent pivots
✓ Tracks trend composition (number of concave/convex)

# Logic:
- Detect inflexions using 3-point comparison
- Maintain last_trend_max and last_trend_min
- BOS when price closes beyond opposite trend extreme
- Switch trend and update pivots
```

#### SMC Library BOS
```python
# Characteristics:
- Uses swing_highs_lows with fixed swing_length parameter
- BOS when any price movement breaks swing level
- Less sophisticated trend tracking
- Simpler to understand but less precise

# Logic:
- Find swing highs/lows in window
- Check if current price exceeds previous swing
- Mark as BOS if broken
```

### Inflexion Points vs Swing Highs/Lows

#### Your Inflexion Points
```python
# Detection:
high[i] >= high[i-1] AND high[i] > high[i+1]  # Concave
low[i] <= low[i-1] AND low[i] < low[i+1]      # Convex

# Advantages:
✓ Immediate detection (3-point comparison)
✓ All local extrema captured
✓ No arbitrary lookback parameter
✓ Respect/disrespect tracking
✓ Historical filtering
```

#### SMC Swing Highs/Lows
```python
# Detection:
current_high == max(highs in window of size 2*swing_length)
current_low == min(lows in window of size 2*swing_length)

# Characteristics:
- Requires user-defined swing_length
- Only captures major swings (misses smaller but significant pivots)
- No respect/disrespect concept
- Includes all swings (no filtering)
```

### Fair Value Gaps Comparison

#### Your Custom FVG
```python
# Advantages:
✓ Validates candle body is within gap
✓ Adjusts gap boundaries based on body position
✓ Respect/disrespect tracking
✓ Historical filtering
✓ Separate handling for above/below gaps

# More Conservative:
- Ensures middle candle body stays in gap
- Tighter gap boundaries (adjusted for candle body)
```

#### SMC Library FVG
```python
# Characteristics:
- Simpler gap detection (just checks highs/lows)
- join_consecutive option to merge adjacent gaps
- Less validation on candle structure
- No adjusted boundaries

# More Permissive:
- May detect gaps that aren't true imbalances
- Wider gap ranges
```

---

## Implementation Recommendations

### When to Use Custom Indicators

1. **Inflexion Points** are crucial when:
   - You need precise support/resistance levels
   - You want to filter out broken levels automatically
   - You need to know if levels are being respected
   - You want all significant pivots, not just major swings

2. **Custom BOS** is superior when:
   - You need confirmed trend changes (close/open requirement)
   - You want to track trend state explicitly
   - You need to distinguish between trend-defining and recent pivots
   - You want fewer false signals (more conservative)

3. **Custom FVG** is better when:
   - You need validated gaps (body-in-gap check)
   - You want tighter, more accurate gap boundaries
   - You need to know gap effectiveness (respect status)
   - You want cleaner charts (historical filtering)

### Integration Strategy

```python
# Recommended: Use custom inflexions as foundation
1. Detect inflexion points (Tools/inflexion.py)
2. Build BOS on top of inflexions (Tools/bos.py)
3. Add FVG as additional confirmation (Tools/fairValueGaps.py)

# Strategy Logic:
- Entry: BOS in desired direction + FVG respected as support/resistance
- Stop Loss: Beyond last inflexion point
- Take Profit: At next opposite inflexion or FVG
```

---

## Conclusion

Your custom indicators represent a **more sophisticated and precise** approach to market structure analysis compared to the Smart Money Concepts library. The key advantages are:

1. **Statistical Rigor**: Inflexion points are mathematically defined local extrema
2. **State Tracking**: Explicit respect/disrespect monitoring
3. **Historical Memory**: Automatic filtering of broken levels
4. **Confirmation Requirements**: Close/open prices for stronger signals
5. **Trend Awareness**: Explicit trend state machines

These make the custom indicators more suitable for **professional trading systems** where precision and reduced false signals are critical, at the cost of slightly more complexity.

---

**Next Steps:**
1. Compare custom BOS implementation with SMC BOS in detail
2. Visualize both on same chart to see differences
3. Backtest both approaches to quantify performance differences
4. Consider hybrid approach using custom inflexions with SMC for other indicators
