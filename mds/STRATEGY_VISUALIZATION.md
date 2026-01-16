# Strategy Visualization System

## Overview

This document specifies the visualization system for the Smart Money Concepts trading strategy. The system generates a **single interactive HTML file** with three timeframe tabs that dynamically show/hide based on strategy progression.

---

## Tab System Behavior

### Key Principle: One Tab Active at a Time
- Only ONE tab is visible at any moment
- Tabs automatically switch based on which timeframe has activity
- **Cascading dependency**: Lower timeframes only become active when higher timeframe events are detected

### Tab Activation Rules

| Strategy State | 1H Tab | 5M Tab | 1M Tab |
|----------------|--------|--------|--------|
| No liquidity sweep detected | Active (searching) | Empty | Empty |
| Liquidity sweep found, no Event B | Shows sweep | Active (searching for Event B) | Empty |
| Event B found, no validation | Shows sweep | Active (searching for validation) | Empty |
| Validation found | Shows sweep | Shows Event B + validation | Active (monitoring TP/SL) |

---

## Timeframe Specifications

### 1. Large Time Interval (1H)

**Purpose**: Show overall market structure and liquidity sweep detection

**Indicators Displayed** (SMC indicators - NO inflection points):
- Candlesticks (green bullish, red bearish)
- Order Blocks (rectangular zones)
- Fair Value Gaps (rectangular zones)
- BOS/CHoCH lines (horizontal price lines)
- Liquidity levels (horizontal lines, X when swept)

**Data Range**: ALL available historical data

**Visual Style**: Match `walkthroughs/.../walkthrough.html` using TradingView lightweight-charts:
- Rectangular zones for OBs (cyan bullish, orange bearish)
- Rectangular zones for FVGs (yellow semi-transparent)
- Horizontal lines for BOS/CHoCH/Liquidity
- X markers for swept/invalidated levels
- NO swing point markers (no stars, circles, triangles for inflection points)

---

### 2. Middle Time Interval (5M)

**Purpose**: Show Event B detection and validation (FVG/equilibrium) search

**Indicators Displayed** (same as 1H - NO inflection points):
- Order Blocks (rectangular zones)
- Fair Value Gaps (rectangular zones)
- BOS/CHoCH lines
- Liquidity levels

**Data Range** (DYNAMIC based on strategy stage):

| Stage | Range Start | Range End |
|-------|-------------|-----------|
| Before Event B found | Start of where liquidity sweep 1 started (in 1H) | Current candle |
| After Event B found | Start of the Order Block that would be used to close/exit the operation | Current candle |

**Transition Behavior**:
- When Event B is detected, the chart range shifts to focus on validation search
- The exit Order Block becomes the left boundary of the view

---

### 3. Small Time Interval (1M)

**Purpose**: Monitor trade execution - show proximity to Take Profit and Stop Loss

**Indicators Displayed** (same as 1H + TP/SL - NO inflection points):
- Candlesticks
- Order Blocks, FVGs, BOS/CHoCH, Liquidity levels
- **TP Line**: Horizontal line showing Take Profit level
- **SL Line**: Horizontal line showing Stop Loss level

**Data Range**: From the start of the 1H liquidity sweep

**Special Features**:
- TP/SL lines extend across the entire visible range
- User can visually track how close price is getting to each level
- Real-time (or frame-by-frame in replay) proximity tracking

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
│  TABS: [ 1H ] [ 5M ] [ 1M ]    ← Auto-switches based on active timeframe    │
│         ^^^^                                                                 │
│         Only one tab shown at a time, others greyed out if empty            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────┐  ┌─────────────────────┐ │
│  │                                               │  │  SIDEBAR            │ │
│  │              CHART AREA                       │  │                     │ │
│  │                                               │  │  Sweep Time:        │ │
│  │   [Candlesticks]                              │  │  Oct 6 15:30        │ │
│  │   [Order Block Zones - Cyan/Orange]           │  │                     │ │
│  │   [FVG Zones - Yellow]                        │  │  Outcome:           │ │
│  │   [BOS/CHoCH Lines]                           │  │  SUCCESS            │ │
│  │   [Liquidity Levels - X when swept]           │  │  P/L: +$8.37        │ │
│  │   (NO inflection point markers)               │  │                     │ │
│  │                                               │  │  Conditions:        │ │
│  │   For 1M tab only:                            │  │  ✓ 1H Sweep         │ │
│  │   [TP Line - Green horizontal]                │  │  ✓ 5M Event B       │ │
│  │   [SL Line - Red horizontal]                  │  │  ✓ 5M Validation    │ │
│  │                                               │  │  ✓ 1M Confirmation  │ │
│  └───────────────────────────────────────────────┘  │                     │ │
│                                                      │  Active TF: 5M     │ │
│                                                      │  Stage: Validation │ │
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
  - `FAILED (Hit SL)` - Trade executed but stopped out

### 2. Automatic Tab Switching
- System monitors which timeframe has the current "action"
- Automatically switches tabs when:
  - 1H → 5M: When liquidity sweep is detected
  - 5M → 1M: When Event B and validation are found
  - 1M stays active: Until trade concludes (TP or SL hit)
- User can manually click tabs, but only non-empty tabs are clickable

### 3. Frame Navigation
- **Arrow keys**: Left/Right to step through candles
- **Buttons**: ◀ Prev | Next ▶
- **Frame counter**: Shows current position (e.g., "Frame 15/120")
- Animation shows how the setup developed over time

### 4. Sidebar Information
- Current timestamp
- Sweep outcome (SUCCESS/FAILED)
- P/L amount (for successful trades)
- Conditions checklist (✓ or ✗ for each stage)
- **Active Timeframe indicator**: Shows which TF is currently displayed
- **Current Stage**: Shows where we are in the strategy flow

---

## Indicator Visual Reference

Based on `walkthroughs/.../walkthrough.html` (TradingView lightweight-charts style):

| Indicator | Visual Style | Color |
|-----------|-------------|-------|
| Bullish candles | Solid body | Teal/Green |
| Bearish candles | Solid body | Red |
| Order Blocks (bullish) | Filled rectangle | Cyan (`rgba(0, 188, 212, 0.2)`) |
| Order Blocks (bearish) | Filled rectangle | Orange (`rgba(255, 87, 34, 0.2)`) |
| Fair Value Gaps | Filled rectangle | Yellow (semi-transparent) |
| BOS lines | Solid horizontal line | Green (bullish), Red (bearish) |
| CHoCH lines | Solid horizontal line | Distinct color |
| Liquidity levels | Horizontal line | Orange |
| Liquidity swept | X marker on line | Red |
| TP Level (1M only) | Horizontal dashed line | Green |
| SL Level (1M only) | Horizontal dashed line | Red |

**NOT included**: Swing high/low markers (stars, circles, triangles for inflection points)

---

## Data Range Summary

| Timeframe | Range |
|-----------|-------|
| 1H | All available data |
| 5M (before Event B) | From 1H liquidity sweep start → Current |
| 5M (after Event B) | From exit Order Block start → Current |
| 1M | From 1H liquidity sweep start → Current |

---

## Implementation Notes

1. **Tab State Management**: Track strategy state to determine which tab should be active
2. **Dynamic Ranges**: 5M chart needs logic to detect Event B and switch range boundaries
3. **TP/SL Lines**: Only render on 1M chart, calculate from entry price and risk parameters
4. **Indicator Parity**: Ensure all three timeframes show the same indicator types (except TP/SL on 1M)
5. **Visual Reference**: Use existing `walkthroughs/.../walkthrough.html` as the template for chart styling
6. **No Inflection Points**: Do NOT render swing high/low markers - only zones (OB, FVG) and lines (BOS, liquidity, TP/SL)
