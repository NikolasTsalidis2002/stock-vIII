# Strategy II: Equilibrium Premium/Discount Zone Strategy

## Overview

This document describes the **Equilibrium Premium/Discount Zone Strategy**, an evolution of the original multi-timeframe strategy. The key change is replacing the FVG/Demand Zone validation step with an **Equilibrium Zone** approach based on Fibonacci retracement principles.

---

## Strategy Flow Comparison

### Original Strategy (Strategy I):
1. 1H Liquidity Sweep → 2. 5M Event B (BOS/IFVG) → 3. 5M FVG/Demand Zone Validation → 4. 1M Confirmation

### New Strategy (Strategy II):
1. 1H Liquidity Sweep → 2. 5M Event B (BOS/IFVG) → 3. **5M Equilibrium Premium/Discount Area** → 4. 1M Confirmation → 5. **Trade with Exit Target**

---

## Detailed Strategy Logic

### Stage 1: 1H Liquidity Sweep (unchanged)
- Detect liquidity sweep on 1H timeframe
- **Bearish sweep (HIGH swept)** → prepare for SHORT
- **Bullish sweep (LOW swept)** → prepare for LONG

### Stage 2: 5M Event B (unchanged)
- After liquidity sweep, wait for 5M BOS or IFVG
- Must be in the **opposite direction** of the swept level (confirming reversal)
- For SHORT: 5M bearish BOS/IFVG
- For LONG: 5M bullish BOS/IFVG

### Stage 3: 5M Equilibrium Validation (NEW - replaces FVG/Demand Zone)

This is the core change in Strategy II.

#### For SHORT:
- **Max** = Swept high (FIXED, never changes)
- **Min** = Running lowest price after 5M Event B (updates dynamically as price makes new lows)
- **Equilibrium** = (Max + Min) / 2 (the 50% Fibonacci level)
- **Premium Area** = Above equilibrium (between equilibrium and max)
- **Trigger**: Price enters premium area → move to 1M confirmation

#### For LONG:
- **Min** = Swept low (FIXED, never changes)
- **Max** = Running highest price after 5M Event B (updates dynamically as price makes new highs)
- **Equilibrium** = (Max + Min) / 2 (the 50% Fibonacci level)
- **Discount Area** = Below equilibrium (between min and equilibrium)
- **Trigger**: Price enters discount area → move to 1M confirmation

#### Dynamic Min/Max Tracking Logic:

The key insight is that the equilibrium zone is calculated **dynamically** as price moves:

1. After 5M Event B, track the running min (for SHORT) or max (for LONG)
2. Each time price makes a new extreme, update the min/max
3. Recalculate equilibrium: `(Max + Min) / 2`
4. Check if current price is in the target zone (premium for SHORT, discount for LONG)
5. Continue until price enters the target zone

```
SHORT Example:
- Swept high at $250 (Max = $250, fixed)
- 5M BOS at $245
- Price drops to $243 → Min = $243, Equilibrium = $246.50, Premium = above $246.50
- Price drops to $240 → Min = $240, Equilibrium = $245.00, Premium = above $245.00
- Price bounces to $242 → Min stays $240 (no new low)
- Price drops to $238 → Min = $238, Equilibrium = $244.00, Premium = above $244.00
- Price rises to $246 → PREMIUM REACHED (above $244) → Move to 1M timeframe
```

### Stage 4: 1M Confirmation (unchanged)
- Once price enters premium/discount area, move to 1M timeframe
- Look for 1M BOS or IFVG in the **entry direction**
- For SHORT: 1M bearish BOS/IFVG
- For LONG: 1M bullish BOS/IFVG

---

## Exit Target: Order Block at Origin

The exit target is an Order Block found on the **5M timeframe** at the origin of the move that created the liquidity.

### For SHORT:
1. The 1H liquidity swept a HIGH at time X
2. This high was created by an **upward trend**
3. Before the upward trend, there was a **downward trend**
4. Find the BOS that ended the downward trend and started the upward trend
5. Find the **lowest price** of that downward trend
6. **Order Block** = Last group of **RED candles** + first **GREEN candle** at that minimum
7. **Take Profit** = Top of this Order Block

### For LONG:
1. The 1H liquidity swept a LOW at time X
2. This low was created by a **downward trend**
3. Before the downward trend, there was an **upward trend**
4. Find the BOS that ended the upward trend and started the downward trend
5. Find the **highest price** of that upward trend
6. **Order Block** = Last group of **GREEN candles** + first **RED candle** at that maximum
7. **Take Profit** = Bottom of this Order Block

---

## Trade Execution

### Entry Price
- Closing price of the candle that causes the 1M BOS/IFVG confirmation

### Take Profit
- **SHORT**: Top of the exit Order Block
- **LONG**: Bottom of the exit Order Block

### Stop Loss (2:1 Risk/Reward)
```
profit_distance = abs(entry_price - take_profit_price)
risk_allowed = profit_distance / 2
stop_loss = entry_price + risk_allowed  (for SHORT)
stop_loss = entry_price - risk_allowed  (for LONG)
```

### Future Enhancement (noted for later):
If the calculated stop loss ends up beyond the swept liquidity level, the trade should be skipped (risk/reward not favorable). **Implement this check after core logic is working.**

---

## Invalidation Conditions

A setup is invalidated and abandoned if:

### 1. Price breaks the swept level
- **SHORT**: Price breaks **above** the swept high
- **LONG**: Price breaks **below** the swept low

### 2. Price breaks the exit Order Block
- **SHORT**: Price breaks **below** the bottom of the exit Order Block (target reached before entry)
- **LONG**: Price breaks **above** the top of the exit Order Block (target reached before entry)

---

## Complete SHORT Example (Step-by-Step)

### Setup Phase:
| Time | Timeframe | Event | Price |
|------|-----------|-------|-------|
| 10:00 | 1H | Liquidity sweep of HIGH | $250 swept |
| 10:15 | 5M | 5M BOS bearish confirmed | $245 |

**Max locked at $250** (the swept high - never changes)

### Equilibrium Tracking Phase (5M):
| Time | Price Action | Running Min | Equilibrium | Premium Area |
|------|--------------|-------------|-------------|--------------|
| 10:20 | Drops to $243 | $243 | $246.50 | Above $246.50 |
| 10:25 | Drops to $240 | $240 | $245.00 | Above $245.00 |
| 10:30 | Bounces to $242 | $240 | $245.00 | Above $245.00 |
| 10:35 | Drops to $238 | $238 | $244.00 | Above $244.00 |
| 10:40 | Bounces to $241 | $238 | $244.00 | Above $244.00 |
| 10:45 | **Rises to $246** | $238 | $244.00 | **IN PREMIUM!** |

**Price enters premium area at 10:45 → Move to 1M timeframe**

### Exit Order Block (found on 5M):
- Looking backwards from liquidity sweep at 10:00
- The upward trend to $250 started from a low around $230
- Found: Last group of red candles + first green candle at that origin
- **Exit Order Block**: Top = $232, Bottom = $230

### 1M Confirmation Phase:
| Time | Event | Price |
|------|-------|-------|
| 10:45:00 | Scanning for BOS/IFVG... | Nothing yet |
| 10:46:00 | Scanning... | Nothing yet |
| 10:47:00 | **1M BOS bearish!** | Candle closes at $245.50 |

**1M Confirmation triggered!**

### Trade Execution:
| Parameter | Calculation | Value |
|-----------|-------------|-------|
| Entry | 1M candle close | $245.50 |
| Take Profit | Top of exit OB | $232.00 |
| Profit Distance | $245.50 - $232.00 | $13.50 |
| Risk Allowed | $13.50 / 2 | $6.75 |
| Stop Loss | $245.50 + $6.75 | $252.25 |

### Final Trade:
- **Direction:** SHORT
- **Entry:** $245.50
- **Stop Loss:** $252.25
- **Take Profit:** $232.00
- **Risk/Reward:** 2:1

---

## Complete LONG Example (Step-by-Step)

### Setup Phase:
| Time | Timeframe | Event | Price |
|------|-----------|-------|-------|
| 14:00 | 1H | Liquidity sweep of LOW | $200 swept |
| 14:10 | 5M | 5M BOS bullish confirmed | $204 |

**Min locked at $200** (the swept low - never changes)

### Equilibrium Tracking Phase (5M):
| Time | Price Action | Running Max | Equilibrium | Discount Area |
|------|--------------|-------------|-------------|---------------|
| 14:15 | Rises to $208 | $208 | $204.00 | Below $204.00 |
| 14:20 | Rises to $212 | $212 | $206.00 | Below $206.00 |
| 14:25 | Pulls back to $210 | $212 | $206.00 | Below $206.00 |
| 14:30 | Rises to $215 | $215 | $207.50 | Below $207.50 |
| 14:35 | Pulls back to $213 | $215 | $207.50 | Below $207.50 |
| 14:40 | **Drops to $205** | $215 | $207.50 | **IN DISCOUNT!** |

**Price enters discount area at 14:40 → Move to 1M timeframe**

### Exit Order Block (found on 5M):
- Looking backwards from liquidity sweep at 14:00
- The downward trend to $200 started from a high around $220
- Found: Last group of green candles + first red candle at that origin
- **Exit Order Block**: Top = $218, Bottom = $216

### 1M Confirmation Phase:
| Time | Event | Price |
|------|-------|-------|
| 14:42:00 | **1M BOS bullish!** | Candle closes at $206.00 |

**1M Confirmation triggered!**

### Trade Execution:
| Parameter | Calculation | Value |
|-----------|-------------|-------|
| Entry | 1M candle close | $206.00 |
| Take Profit | Bottom of exit OB | $216.00 |
| Profit Distance | $216.00 - $206.00 | $10.00 |
| Risk Allowed | $10.00 / 2 | $5.00 |
| Stop Loss | $206.00 - $5.00 | $201.00 |

### Final Trade:
- **Direction:** LONG
- **Entry:** $206.00
- **Stop Loss:** $201.00
- **Take Profit:** $216.00
- **Risk/Reward:** 2:1

---

## Key Differences from Strategy I

| Aspect | Strategy I | Strategy II |
|--------|------------|-------------|
| **Validation** | FVG or Demand Zone respect | Equilibrium Premium/Discount area |
| **Zone Definition** | Static zones from indicators | Dynamic Fibonacci-based zones |
| **Exit Target** | Not explicitly defined | Order Block at origin of liquidity move |
| **Stop Loss** | Not explicitly defined | 2:1 R/R based on take profit |
| **Risk Management** | Manual | Built-in 2:1 ratio |

---

## Visual Reference

See image: `ACodingReflections/Anotes_while_developing.png`

The image shows:
- Blue line at top: Liquidity level (swept high)
- Fibonacci levels (0.618, 0.5, 0.382): The equilibrium zone
- Red shaded area at bottom: Demand zone / exit target Order Block

---

## Implementation Notes

### Files to Create/Modify:
| File | Action |
|------|--------|
| `src/strategy/equilibrium_validator.py` | **CREATE** - New validator |
| `src/strategy/models.py` | **MODIFY** - Add new fields |
| `src/strategy/strategy_engine.py` | **MODIFY** - Integrate validator |
| `scripts/analyze_strategy.py` | **MODIFY** - Add CLI flag |

### Keep Original Code:
The original FVG/Demand Zone validation code should remain intact in `validation_detector.py` for future use. A `--validation-mode` CLI argument will allow switching between strategies.

---

## Summary

Strategy II improves upon Strategy I by:
1. Using dynamic Fibonacci-based equilibrium zones instead of static indicator zones
2. Explicitly defining exit targets using Order Blocks at the origin of the move
3. Building in 2:1 risk/reward ratio for consistent risk management
4. Providing clear invalidation conditions for setup abandonment

The core multi-timeframe structure (1H → 5M → 1M) remains the same, ensuring compatibility with existing infrastructure.
