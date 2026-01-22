# GMM + Fibonacci Zone Trading Strategy

## 1. Strategy Overview

This strategy combines **Gaussian Mixture Model (GMM)** price zone detection with **Fibonacci retracement levels** to determine optimal entry bias (long/short) before applying Smart Money Concepts (SMC) validation.

### Core Concept
1. Use GMM to identify price distribution zones on the 1H timeframe
2. Apply Fibonacci levels within the current zone
3. Trade reversals at zone extremes (premium/discount)
4. Validate with existing SMC confirmation flow

---

## 2. Strategy Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: GMM ZONE DETECTION                   │
│                         (1H Timeframe)                          │
├─────────────────────────────────────────────────────────────────┤
│  • Lookback: X candles (configurable)                           │
│  • Recalculate: Every N candles (configurable)                  │
│  • Identify which zone current price belongs to                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 2: FIBONACCI LEVELS                       │
│                  (Within Current Zone)                          │
├─────────────────────────────────────────────────────────────────┤
│  Fib 1.0 = max(component_prices)  ← Zone Top                    │
│  Fib 0.786                                                      │
│  Fib 0.618                                                      │
│  Fib 0.5   ← Take Profit Target                                 │
│  Fib 0.382                                                      │
│  Fib 0.236                                                      │
│  Fib 0.0 = min(component_prices)  ← Zone Bottom                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 3: ENTRY BIAS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PREMIUM ZONE (0.786 - 1.0):                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Bias: SHORT                                          │    │
│  │  • Look for: Sweep of HIGHS (buy-side liquidity)        │    │
│  │  • Logic: Price at top of range, expect reversal down   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  DISCOUNT ZONE (0.0 - 0.236):                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Bias: LONG                                           │    │
│  │  • Look for: Sweep of LOWS (sell-side liquidity)        │    │
│  │  • Logic: Price at bottom of range, expect reversal up  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  MIDDLE ZONE (0.236 - 0.786):                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Configurable: Skip trades OR allow both directions   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  UNCHARTED TERRITORY (outside all zones):                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • No trades until GMM recalculates                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: LIQUIDITY SWEEP DETECTION                  │
│                       (1H Timeframe)                            │
├─────────────────────────────────────────────────────────────────┤
│  Filter sweeps based on entry bias:                             │
│  • If SHORT bias: Only detect sweeps of HIGHS                   │
│  • If LONG bias: Only detect sweeps of LOWS                     │
│                                                                 │
│  Requirement: Current price must be in premium/discount zone    │
│  (The swept liquidity level itself can be anywhere)             │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: EXISTING VALIDATION FLOW                   │
│                  (Unchanged from current)                       │
├─────────────────────────────────────────────────────────────────┤
│  5min: Event B (BOS/IFVG) in entry direction                    │
│        ↓                                                        │
│  5min: FVG/Equilibrium Validation                               │
│        ↓                                                        │
│  1min: Confirmation (BOS/IFVG)                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 6: EXIT TARGETS                         │
├─────────────────────────────────────────────────────────────────┤
│  Take Profit (configurable):                                    │
│  • "fib": 0.5 Fibonacci level (zone midpoint)                   │
│  • "order_block": Existing Order Block-based TP                 │
│                                                                 │
│  Stop Loss:                                                     │
│  • 1:2 R:R (risk 1 to make 2)                                   │
│  • SL distance = half the distance to TP                        │
│  • SHORT: SL above entry                                        │
│  • LONG: SL below entry                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Entry Rules

### 3.1 SHORT Entry (Premium Zone)
1. GMM identifies current zone
2. Current price is in premium zone (fib >= 0.786)
3. Liquidity sweep of HIGHS detected (buy-side liquidity taken)
4. 5min Event B confirms bearish direction
5. 5min Validation (FVG/Equilibrium) passes
6. 1min Confirmation in bearish direction
7. **ENTRY: SHORT**

### 3.2 LONG Entry (Discount Zone)
1. GMM identifies current zone
2. Current price is in discount zone (fib <= 0.236)
3. Liquidity sweep of LOWS detected (sell-side liquidity taken)
4. 5min Event B confirms bullish direction
5. 5min Validation (FVG/Equilibrium) passes
6. 1min Confirmation in bullish direction
7. **ENTRY: LONG**

---

## 4. Exit Rules

### 4.1 Take Profit
| Method | Description |
|--------|-------------|
| `fib` | 0.5 Fibonacci level (zone midpoint) |
| `order_block` | Existing Order Block-based target |

### 4.2 Stop Loss
- **Calculation**: 1:2 Risk:Reward ratio
- **Formula**: `SL_distance = TP_distance / 2`
- **SHORT**: `SL = entry_price + SL_distance`
- **LONG**: `SL = entry_price - SL_distance`

---

## 5. Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable GMM zone filtering |
| `lookback_candles` | int | 300 | Candles for GMM analysis |
| `timeframe` | str | "1h" | Timeframe for zone detection |
| `step` | float | 0.1 | Price level granularity |
| `max_components` | int | 3 | Max GMM components |
| `recalc_interval` | int | 50 | Recalculate every N candles |
| `fib_levels` | list | [1.0, 0.786, ...] | Fibonacci levels |
| `premium_zone` | list | [0.786, 1.0] | Premium zone boundaries |
| `discount_zone` | list | [0.0, 0.236] | Discount zone boundaries |
| `allow_middle_zone_trades` | bool | false | Trade in middle zone |
| `take_profit_method` | str | "fib" | "fib" or "order_block" |

---

## 6. Code Reuse Strategy

### 6.1 Existing Components (NO CHANGES)
- `TimeframeManager` - Multi-TF data alignment
- `EventBDetector` - 5min Event B detection
- `EquilibriumValidator` - FVG/Equilibrium validation
- `ConfirmationDetector` - 1min confirmation
- All SMC indicators in `smc_custom.py`

### 6.2 Modified Components (MINIMAL CHANGES)
- `LiquidityDetector` - Add `sweep_type_filter` parameter
- `ExitTargetFinder` - Add fib-based TP option
- `StrategyEngine` - Integrate GMM zone detection

### 6.3 New Components
- `GMMZoneDetector` - Adapt from `notebooks/curr.ipynb`

### 6.4 Code from notebooks/curr.ipynb to Reuse
```python
# These functions can be directly copied/adapted:
- generate_price_levels()      # Line ~30
- find_optimal_components()    # Line ~100
- get_component_prices()       # Line ~130
- get_price_distribution()     # Line ~150
```

---

## 7. Visual Representation

```
Price
  │
  │  ══════════════════════════════════════  Fib 1.0 (Zone Top)
  │  ┌─────────────────────────────────────┐
  │  │         PREMIUM ZONE                │ ← SHORT bias
  │  │     (Sweep highs → SHORT)           │   Look for sweep of HIGHS
  │  └─────────────────────────────────────┘
  │  ══════════════════════════════════════  Fib 0.786
  │  ┌─────────────────────────────────────┐
  │  │                                     │
  │  │         MIDDLE ZONE                 │ ← Configurable
  │  │      (Skip or allow both)           │
  │  │                                     │
  │  │  - - - - - - - - - - - - - - - - -  │  Fib 0.5 (TP Target)
  │  │                                     │
  │  └─────────────────────────────────────┘
  │  ══════════════════════════════════════  Fib 0.236
  │  ┌─────────────────────────────────────┐
  │  │        DISCOUNT ZONE                │ ← LONG bias
  │  │     (Sweep lows → LONG)             │   Look for sweep of LOWS
  │  └─────────────────────────────────────┘
  │  ══════════════════════════════════════  Fib 0.0 (Zone Bottom)
  │
  └────────────────────────────────────────────────────────────── Time
```

---

## 8. Example Trade Scenarios

### Scenario A: SHORT in Premium
```
1. GMM detects 2 zones, current price ($420) in Zone 1
2. Zone 1 component_prices: min=$380, max=$440
3. Fib levels: 0.0=$380, 0.5=$410, 1.0=$440
4. Current fib position: ($420-$380)/($440-$380) = 0.67... wait recalculating
   Actually: ($420-$380)/$60 = 0.667 → Middle zone

   Let's say price moves to $435:
   Fib position: ($435-$380)/$60 = 0.917 → Premium zone (>0.786)

5. Sweep of highs detected at $438
6. Bias: SHORT, look for highs ✓
7. 5min Event B confirms bearish
8. Validation passes
9. 1min confirms
10. ENTRY: SHORT at $433
11. TP: $410 (0.5 fib)
12. SL: $433 + ($433-$410)/2 = $433 + $11.5 = $444.5
```

### Scenario B: LONG in Discount
```
1. Same zone: min=$380, max=$440
2. Price drops to $385
3. Fib position: ($385-$380)/$60 = 0.083 → Discount zone (<0.236)
4. Sweep of lows detected at $378
5. Bias: LONG, look for lows ✓
6. 5min Event B confirms bullish
7. Validation passes
8. 1min confirms
9. ENTRY: LONG at $382
10. TP: $410 (0.5 fib)
11. SL: $382 - ($410-$382)/2 = $382 - $14 = $368
```

---

## 9. Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-22 | 1.0 | Initial strategy documentation |
