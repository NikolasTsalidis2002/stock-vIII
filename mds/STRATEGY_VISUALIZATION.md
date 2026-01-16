# Strategy Visualization System

## Overview

This document specifies the visualization system for the Smart Money Concepts trading strategy. The system generates a **single interactive HTML file** with **4 static tabs** that show fixed snapshots of the complete trade at different stages.

**Key Change**: Static snapshots instead of frame-by-frame animation.

---

## Tab System: Static vs Animated

| Before (Old) | After (New) |
|--------------|-------------|
| Frame-by-frame candle progression | Fixed static snapshots |
| 3 tabs (1H, 5M, 1M) | 4 tabs (1H, 5M Event B, 5M Validation, 1M) |
| Prev/Next frame navigation | No frame navigation |
| Dynamic data ranges | Fixed ranges showing complete trade |

---

## 4 Tab Specifications

### Tab 1: 1H (Liquidity Sweep)

**Purpose**: Show the liquidity sweep and overall market structure

**Range**: Start of data to end of trade (where position closed)

**Indicators**:
- Candlesticks (green bullish, red bearish)
- FVG zones (yellow semi-transparent)
- Order Blocks (cyan bullish, orange bearish)
- BOS lines (green/red)
- Liquidity lines (orange, X when swept)
- **Highlight**: The swept liquidity level with SWEEP marker

---

### Tab 2: 5M Event B

**Purpose**: Show what BOS or IFVG triggered Event B

**Range**: 1H liquidity sweep timestamp to end of trade

**Indicators**:
- All standard indicators (FVG, OB, BOS, liquidity)
- **Highlight**: The specific BOS or IFVG that was detected as Event B
  - Marker/annotation showing "Event B: BOS" or "Event B: IFVG"
  - Highlighted with distinct color/style

---

### Tab 3: 5M Validation

**Purpose**: Show FVG respect or equilibrium zone entry

**Range**: Exit Order Block start to end of trade

**Indicators**:
- All standard indicators
- **Highlight for FVG Respect**: The FVG(s) that were respected
- **Highlight for Equilibrium**:
  - Horizontal purple dashed line at equilibrium level
  - Label: "EQ: $XXX.XX"

---

### Tab 4: 1M (Entry & Exit)

**Purpose**: Show trade execution with TP/SL levels

**Range**: 1H liquidity sweep timestamp to end of trade

**Indicators**:
- Candlesticks, FVG, BOS
- **TP Line**: Green dashed horizontal line
- **SL Line**: Red dashed horizontal line
- **Entry marker**: Arrow at confirmation candle

---

## User Interface

```
+-----------------------------------------------------------------+
| HEADER: [Sweep Dropdown v]  Sweep 1/5                           |
+-----------------------------------------------------------------+
| TABS: [ 1H ] [ 5M Event B ] [ 5M Validation ] [ 1M ]            |
+-----------------------------------------------------------------+
| CHART AREA                       | SIDEBAR                      |
| (Static snapshot)                | - Sweep info                 |
|                                  | - Direction                  |
|                                  | - Outcome                    |
|                                  | - Entry/TP/SL                |
|                                  | - Conditions check/X         |
+-----------------------------------------------------------------+
```

**Removed**: Frame counter, Prev/Next frame navigation buttons

---

## Partial Setups Handling

For partial setups (where not all conditions were met):
- **Show all 4 tabs** but disable tabs for stages that weren't reached
- Tab 1 (1H): Always available (sweep was detected)
- Tab 2 (5M Event B): Disabled if no Event B was found
- Tab 3 (5M Validation): Disabled if no validation was found
- Tab 4 (1M): Disabled if no confirmation was found

Disabled tabs shown greyed out with "No data" indication.

---

## Data Requirements from TradeSignal

| Field | Used For |
|-------|----------|
| `timestamp_1h_sweep` | 1H tab highlight, 5M/1M range start |
| `timestamp_5m_event_b` | 5M Event B tab highlight |
| `condition_event_b` | Label ("BOS" or "IFVG") |
| `timestamp_5m_validation` | 5M Validation tab highlight |
| `condition_validation` | Label ("FVG_Respect" or "Equilibrium") |
| `equilibrium_level` | Draw equilibrium line |
| `timestamp_1m_confirmation` | 1M entry marker |
| `take_profit_price` | 1M TP line |
| `stop_loss_price` | 1M SL line |
| `exit_ob_start_idx` | 5M Validation tab range start |

---

## Highlighting Strategy

### Event B Highlighting
- If Event B is BOS: Draw thicker/different colored BOS line (yellow highlight, lineWidth: 4)
- If Event B is IFVG: Highlight that FVG zone differently (brighter color, thicker border)

### Validation Highlighting
- For FVG Respect: Highlight the respected FVG with distinct style
- For Equilibrium: Draw horizontal purple dashed line with "EQ" label

---

## Indicator Visual Reference

| Indicator | Visual Style | Color |
|-----------|-------------|-------|
| Bullish candles | Solid body | Teal/Green (#089981) |
| Bearish candles | Solid body | Red (#f23645) |
| Order Blocks (bullish) | Filled rectangle | Cyan (`rgba(0, 188, 212, 0.2)`) |
| Order Blocks (bearish) | Filled rectangle | Orange (`rgba(255, 87, 34, 0.2)`) |
| Fair Value Gaps | Filled rectangle | Yellow (`rgba(255, 235, 59, 0.2)`) |
| BOS lines | Solid horizontal line | Green (bullish), Red (bearish) |
| Liquidity levels | Horizontal line | Orange (#ff8c00) |
| Liquidity swept | X marker on line | Red |
| TP Level (1M only) | Horizontal dashed line | Green (#089981) |
| SL Level (1M only) | Horizontal dashed line | Red (#f23645) |
| Equilibrium line | Horizontal dashed line | Purple (#9c27b0) |
| Event B highlight | Thicker line/brighter zone | Yellow (#ffff00) |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Arrow Left | Previous sweep |
| Arrow Right | Next sweep |
| 1 | Switch to 1H tab |
| 2 | Switch to 5M Event B tab |
| 3 | Switch to 5M Validation tab |
| 4 | Switch to 1M tab |

---

## Implementation Notes

1. **No Frame Logic**: Remove all frame-by-frame progression code
2. **Static Ranges**: Each tab shows a fixed data range covering the complete trade
3. **Tab State**: Track which tabs are available based on conditions met
4. **Sweep Navigation**: Keep sweep dropdown for switching between different trade setups
5. **Highlighting**: Use distinct visual styles for key events (Event B, validation, entry)
