# Strategy Visualization System

## Overview

This document specifies the visualization system for the Smart Money Concepts trading strategy. The system generates a **single interactive HTML file** that allows you to explore every liquidity sweep detected, see whether it became a successful trade or failed, and step through the setup candle by candle.

---

## Output

**Single file**: `results/{symbol}_sweeps.html`

Example: `results/META_sweeps.html`

---

## User Interface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HEADER                                                                      │
│  ┌──────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ Select Sweep: [Dropdown ▼]       │  │ ◀ Prev │ Frame 15/120 │ Next ▶ │  │
│  │  • Oct 6 15:30 - SUCCESS (+$8)   │  └─────────────────────────────────┘  │
│  │  • Oct 8 17:00 - FAILED (No EB)  │                                       │
│  │  • Oct 10 14:00 - SUCCESS (+$60) │                                       │
│  └──────────────────────────────────┘                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  TABS: [ 1H ] [ 5M ] [ 1M ]                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────┐  ┌─────────────────────┐ │
│  │                                               │  │  SIDEBAR            │ │
│  │              CHART AREA                       │  │                     │ │
│  │                                               │  │  Sweep Time:        │ │
│  │   [Candlesticks]                              │  │  Oct 6 15:30        │ │
│  │   [Order Block Zones - Cyan/Orange]           │  │                     │ │
│  │   [FVG Zones - Yellow]                        │  │  Outcome:           │ │
│  │   [BOS Lines - Green/Red]                     │  │  SUCCESS            │ │
│  │   [Exit OB Zone - Purple]                     │  │  P/L: +$8.37        │ │
│  │                                               │  │                     │ │
│  │   NO CIRCLE MARKERS                           │  │  Conditions:        │ │
│  │   Only rectangular zones and lines            │  │  ✓ 1H Sweep         │ │
│  │                                               │  │  ✓ 5M Event B       │ │
│  └───────────────────────────────────────────────┘  │  ✓ 5M Validation    │ │
│                                                      │  ✓ 1M Confirmation  │ │
│                                                      └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### 1. Sweep Dropdown
- Lists ALL detected liquidity sweeps
- Format: `{timestamp} - {outcome}`
- Outcomes:
  - `SUCCESS (+$X.XX)` - Complete trade that hit TP
  - `FAILED (No Event B)` - Partial setup, failed at stage 2
  - `FAILED (No Validation)` - Partial setup, failed at stage 3
  - `FAILED (No Confirmation)` - Partial setup, failed at stage 4
  - `FAILED (No Exit OB)` - Partial setup, no TP target found

### 2. Timeframe Tabs
- **1H Tab**: Shows liquidity sweep detection
- **5M Tab**: Shows Event B and Equilibrium validation
- **1M Tab**: Shows final confirmation
- Click to switch, one chart visible at a time

### 3. Frame Navigation
- **Arrow keys**: Left/Right to step through candles
- **Buttons**: ◀ Prev | Next ▶
- **Frame counter**: Shows current position (e.g., "Frame 15/120")
- Animation shows how the setup developed over time

### 4. Sidebar
- Current timestamp
- Sweep outcome (SUCCESS/FAILED)
- P/L amount (for successful trades)
- Conditions checklist (✓ or ✗ for each of 4 stages)
- Entry direction (LONG/SHORT)
- TP and SL levels (if applicable)

---

## Visual Elements

### What IS Rendered (Rectangular Zones)

| Element | Color | Style |
|---------|-------|-------|
| Bullish Order Block | Cyan `rgba(0, 188, 212, 0.2)` | Solid border |
| Bearish Order Block | Orange `rgba(255, 87, 34, 0.2)` | Solid border |
| Exit Order Block (TP) | Purple `rgba(156, 39, 176, 0.3)` | Dashed border |
| FVG Zone | Yellow `rgba(255, 235, 59, 0.2)` | Dashed border |
| BOS Line | Green (bullish) / Red (bearish) | Horizontal line |
| Equilibrium Zone | Blue `rgba(33, 150, 243, 0.15)` | Filled rectangle |

### What is NOT Rendered

- ~~Circle markers~~
- ~~Arrow markers~~
- ~~Star symbols~~
- ~~X symbols~~
- ~~Any point-based indicators~~

Only zones (rectangles) and lines are shown. This keeps the chart clean and readable.

---

## Data Sources

### For Successful Trades (TradeSignal)
```python
signal.timestamp_1h_sweep      # When sweep occurred
signal.timestamp_5m_event_b    # When Event B triggered
signal.timestamp_5m_validation # When validation happened
signal.timestamp_1m_confirmation # When confirmation triggered
signal.take_profit_price       # TP level
signal.stop_loss_price         # SL level
signal.exit_ob_top / bottom    # Exit OB zone
signal.equilibrium_level       # Equilibrium zone center
```

### For Failed Setups (PartialSetup)
```python
partial.timestamp_1h_sweep     # When sweep occurred
partial.timestamp_5m_event_b   # None if failed before this
partial.conditions_met         # 1, 2, 3, or 4
partial.failure_reason         # "No Event B found", etc.
partial.exit_ob_top / bottom   # Exit OB if found
```

---

## Frame Generation Logic

### For Each Sweep, Generate Frames:

**Phase 1: Before Sweep (1H)**
- Show 10 candles leading up to sweep
- Status: "Scanning for liquidity sweep..."

**Phase 2: Sweep Detected (1H)**
- Highlight the sweep candle
- Status: "✓ LIQUIDITY SWEEP DETECTED"
- Show exit OB zone if found

**Phase 3: Event B Search (5M)**
- Step through 5M candles after sweep
- Status: "Scanning 5M for Event B..."
- If found: "✓ EVENT B DETECTED"
- If not found (partial): "✗ No Event B - Setup Failed"

**Phase 4: Validation Search (5M)**
- Step through 5M candles after Event B
- Show equilibrium zone
- Status: "Scanning for equilibrium entry..."
- If found: "✓ EQUILIBRIUM ZONE ENTERED"

**Phase 5: Confirmation Search (1M)**
- Step through 1M candles after validation
- Status: "Scanning 1M for confirmation..."
- If found: "✓ CONFIRMATION - TRADE ENTRY"

**Phase 6: Trade Outcome (for successful trades)**
- Show final frame with TP/SL levels
- Status: "TRADE COMPLETE - P/L: +$X.XX"

---

## Windowing Rules

| Timeframe | Window Start | Window End |
|-----------|--------------|------------|
| 1H | 10 candles before sweep | Current frame time |
| 5M | Sweep timestamp | Current frame time |
| 1M | Validation timestamp | Current frame time |

The 5M and 1M charts only become visible after their respective trigger events.

---

## Command

```bash
python3 scripts/analyze_strategy.py --symbol META --visualize
```

Generates: `results/META_sweeps.html`

---

## Technical Implementation

### Files

| File | Purpose |
|------|---------|
| `src/tradingview_strategy_analyzer.py` | Main visualization generator |
| `scripts/analyze_strategy.py` | CLI with `--visualize` flag |
| `src/strategy/models.py` | TradeSignal and PartialSetup dataclasses |

### Libraries

- **TradingView Lightweight Charts** - Candlestick rendering
- **D3.js** - SVG overlay for rectangular zones

### HTML Structure

```html
<!DOCTYPE html>
<html>
<head>
    <script src="lightweight-charts.js"></script>
    <script src="d3.v7.min.js"></script>
</head>
<body>
    <div id="header">
        <select id="sweep-dropdown">...</select>
        <div id="navigation">...</div>
    </div>
    <div id="tabs">...</div>
    <div id="chart-area">...</div>
    <div id="sidebar">...</div>

    <script>
        const allSweeps = [...];  // Embedded JSON data
        // Chart initialization and navigation logic
    </script>
</body>
</html>
```

---

## Summary

- **One HTML file** for all sweeps
- **Dropdown** to select any sweep (timestamp + outcome)
- **Tabs** for 1H/5M/1M timeframes
- **Arrow key navigation** through candles
- **Clean visualization** with zones only (no markers)
- **All SMC indicators** rendered as rectangles and lines

---

**Last Updated**: 2026-01-15
