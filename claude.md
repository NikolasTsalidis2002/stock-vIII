# TSLA Smart Money Concepts Trading System

## Quick Start Guide

### 1. Create Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment
```bash
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Strategy Analysis
```bash
# Analyze complete trade setups
python3 scripts/analyze_strategy.py

# Visualize partial setups (incomplete but close)
python3 scripts/analyze_strategy.py --animate-partials
```

### 5. Deactivate Virtual Environment (when done)
```bash
deactivate
```

---

## Project Overview

This project implements an automated trading system for TSLA (Tesla stock) using Smart Money Concepts (SMC) indicators. The system analyzes price action using institutional trading concepts like Order Blocks, Fair Value Gaps, Break of Structure, Change of Character, and Liquidity patterns to generate trading signals.

**Current Status:** Phase 1 - Data & Visualization Setup

**Last Updated:** 2025-10-17

---

## Project Goals

1. **Download & Process TSLA Data**
   - Fetch historical TSLA price data via Twelve Data API
   - Support multiple timeframes (1m, 5m, 15m, 1h, 1d)
   - Cache data locally for efficient testing

2. **Visualize SMC Indicators**
   - Generate interactive charts showing all SMC indicators
   - Validate indicators are working correctly on TSLA data
   - Visual confirmation before strategy implementation

3. **Implement Trading Strategy**
   - Document strategy rules in `strategy.md`
   - Code strategy logic based on SMC indicators
   - Implement entry/exit rules, filters, and risk management

4. **Backtest & Optimize**
   - Run historical backtests
   - Analyze performance metrics
   - Optimize parameters
   - Generate performance reports

---

## Technical Stack

### Core Libraries
- **Smart Money Concepts Library** - Custom implementation in `smartmoneyconcepts/smc.py`
  - Version: 0.0.26
  - Dependencies: pandas ≥2.0.2, numpy ≥1.24.3, numba ≥0.58.1

### Data Source
- **Twelve Data API** - Stock market data provider
  - Using existing `DataHandler` class from previous project
  - API Key: Configured in DataHandler
  - Timezone: Europe/Madrid
  - Max output: 5000 candles per request

### Visualization
- **Plotly** - Interactive charting library
  - Existing visualization functions in `tests/generate_gif.py`
  - Includes renderers for all SMC indicators
  - Exports to HTML for easy viewing

### Backtesting
- **TBD** - To be determined based on strategy complexity
  - Options: backtesting.py, vectorbt, or custom solution

---

## Project Structure

```
smart-money-concepts/
├── claude.md                      # This file - project documentation
├── strategy.md                    # Trading strategy documentation (to create)
│
├── smartmoneyconcepts/            # Core SMC indicator library
│   ├── __init__.py
│   └── smc.py                     # All indicator implementations
│
├── tests/                         # Existing test files
│   ├── generate_gif.py            # Visualization code (reference)
│   └── test_data/EURUSD/          # Sample test data
│
├── src/                           # New - trading system code (to create)
│   ├── data_loader.py             # TSLA data fetching via Twelve Data
│   ├── visualizer.py              # Adapted visualization for TSLA
│   ├── strategy.py                # Trading strategy implementation
│   └── backtester.py              # Backtesting engine
│
├── data/                          # New - cached market data (to create)
│   └── tsla/
│       ├── tsla_1m.csv
│       ├── tsla_5m.csv
│       ├── tsla_15m.csv
│       └── tsla_1h.csv
│
├── notebooks/                     # New - exploration notebooks (to create)
│   └── visualization.ipynb        # Interactive analysis
│
└── results/                       # New - backtest outputs (to create)
    ├── charts/
    └── reports/
```

---

## SMC Indicators Used

### 1. Fair Value Gap (FVG)
**What it is:** A price gap indicating imbalance between buyers and sellers.
- **Bullish FVG:** Previous high < next low (with bullish candle between)
- **Bearish FVG:** Previous low > next high (with bearish candle between)
- **Usage:** Potential reversal or continuation zones

### 2. Order Blocks (OB)
**What it is:** Price ranges where large institutional orders exist.
- Identified at the last candle before strong price moves
- High volume concentration indicates strength
- **Usage:** Support/resistance zones, entry areas

### 3. Break of Structure (BOS)
**What it is:** Price breaks a significant swing high/low in trend direction.
- Confirms trend continuation
- **Bullish BOS:** Price breaks above previous swing high
- **Bearish BOS:** Price breaks below previous swing low

### 4. Change of Character (CHoCH)
**What it is:** Price breaks structure in opposite direction.
- Early signal of potential trend reversal
- **Bullish CHoCH:** Downtrend breaks above swing high
- **Bearish CHoCH:** Uptrend breaks below swing low

### 5. Liquidity
**What it is:** Clusters of highs/lows within tight ranges.
- Represents stop-loss clusters
- Often "swept" before true price moves
- **Liquidity Sweep:** Price takes liquidity then reverses

### 6. Swing Highs/Lows
**What it is:** Significant pivot points in price action.
- Basis for structure analysis (BOS/CHoCH)
- Configurable lookback period

### 7. Previous High/Low
**What it is:** Prior timeframe's high and low levels.
- Key support/resistance levels
- Tracks if levels are broken

### 8. Retracements
**What it is:** Percentage pullback from swing points.
- Measures strength of retracement
- Helps identify optimal entry timing

---

## How to Read SMC Indicator DataFrames

This section explains how to correctly interpret the DataFrames returned by `smc.py`. These columns are easy to misread — pay close attention to what zero and None mean.

### Shared Lifecycle Columns

Most zone-based indicators (FVG, OB, Inflexion) share these columns:

**MitigatedIndex** — The candle index where price first *touched* the zone.
- `0` means "never touched yet" (it does NOT use NaN for this).
- Mitigation does not mean broken. It just means price came back to the zone.

**Respected** — What happened after mitigation. Three possible values:
- `True` — Price touched the zone and bounced (held). The zone worked as support/resistance.
- `False` — Price closed through the zone (broken/disrespected). The zone failed.
- `None` — Price never reached the zone yet (pending).

**StatusIndex** — The candle index where the Respected decision was made.
- If `Respected=False`, StatusIndex points to the candle that broke through.
- If `Respected=True`, StatusIndex points to the mitigation candle.
- If `Respected=None`, StatusIndex is 0.

**Key distinction:** MitigatedIndex and StatusIndex can differ. MitigatedIndex is when price *first touched* the zone. StatusIndex is when the final verdict (respect or disrespect) happened. For disrespected zones, StatusIndex > MitigatedIndex because price touched first, then later broke through.

---

### Per-Indicator Column Reference

#### FVG — `FVG, Top, Bottom, MitigatedIndex, Respected, StatusIndex`

Most rows are NaN. Only rows where `FVG=1` (bullish) or `FVG=-1` (bearish) have data.

- **Bullish FVG (1):** Top = next candle's low, Bottom = previous candle's high. Disrespected if any future close < Bottom.
- **Bearish FVG (-1):** Top = previous candle's low, Bottom = next candle's high. Disrespected if any future close > Top.

#### OB — `OB, Top, Bottom, StartIndex, BOSIndex, MitigatedIndex, Respected, StatusIndex`

Data is stored at the *inflexion* index (StartIndex), not the BOS candle.

- **Bullish OB:** Top = the broken level, Bottom = min(low) from inflexion to BOS.
- **Bearish OB:** Top = max(high) from inflexion to BOS, Bottom = the broken level.
- Disrespect check uses candle *body* (min of close/open), not wicks.

#### Inflexion — `InflexionType, Level, Respected, StatusIndex`

No Top/Bottom — it is a single price `Level` (a point, not a range). No MitigatedIndex column.

- **Peak (1):** Disrespected if any future close/open > Level. Respected if high reaches Level but doesn't close through.
- **Valley (-1):** Disrespected if any future close/open < Level. Respected if low reaches Level but doesn't close through.

#### Liquidity — `Liquidity, Level, End, Swept`

Different pattern — no Respected/MitigatedIndex/StatusIndex columns.

- `Swept` = index of candle that swept the liquidity (`0` if not swept).
- `End` = index of the last swing high/low in the cluster.

---

### Common Recipes

```python
# Was this FVG mitigated?
fvg_data['MitigatedIndex'] != 0

# Is this zone still active (not broken)?
data['Respected'] != False

# Price touched and bounced?
data['Respected'] == True

# Price touched then broke through?
data['Respected'] == False

# How many candles until mitigation?
fvg_data['MitigatedIndex'] - fvg_data.index

# Get all bullish OBs that held
ob_data[(ob_data['OB'] == 1) & (ob_data['Respected'] == True)]

# Was liquidity swept?
liq_data['Swept'] != 0
```

---

## Development Phases

### Phase 1: Data & Visualization ⏳ (Current)
**Goal:** Set up data pipeline and validate indicators on TSLA

**Tasks:**
- [ ] Create `src/data_loader.py` using Twelve Data API
- [ ] Download historical TSLA data (2020-present recommended)
- [ ] Cache data locally
- [ ] Create `src/visualizer.py` adapted from `tests/generate_gif.py`
- [ ] Generate TSLA charts with all SMC indicators
- [ ] Visual validation of indicator functionality

**Deliverable:** Interactive HTML charts showing TSLA with SMC indicators

---

### Phase 2: Strategy Definition 📋 (Next)
**Goal:** Document clear trading rules

**Tasks:**
- [ ] Create `strategy.md` template
- [ ] Define entry conditions
- [ ] Define exit conditions (TP/SL)
- [ ] Define filters and confluence requirements
- [ ] Define risk management rules
- [ ] Get user approval on strategy

**Deliverable:** Complete `strategy.md` document

---

### Phase 3: Implementation & Backtesting 💻 (Future)
**Goal:** Code and test strategy

**Tasks:**
- [ ] Implement strategy logic in `src/strategy.py`
- [ ] Create backtesting framework in `src/backtester.py`
- [ ] Run initial backtests
- [ ] Generate performance reports
- [ ] Analyze results and iterate

**Deliverable:** Backtested strategy with performance metrics

---

### Phase 4: Optimization & Analysis 📊 (Future)
**Goal:** Refine and optimize

**Tasks:**
- [ ] Parameter optimization
- [ ] Timeframe analysis
- [ ] Drawdown analysis
- [ ] Win rate and risk/reward analysis
- [ ] Forward testing preparation

**Deliverable:** Optimized strategy ready for paper trading

---

## Key Files Reference

### Existing Files
- `smartmoneyconcepts/smc.py` (line 1-958) - All SMC indicator implementations
- `tests/generate_gif.py` - Complete visualization code for all indicators
  - `add_FVG()` - Fair Value Gap visualization
  - `add_OB()` - Order Blocks visualization
  - `add_bos_choch()` - BOS/CHoCH visualization
  - `add_liquidity()` - Liquidity and sweeps visualization
  - `add_swing_highs_lows()` - Swing points visualization
  - `add_previous_high_low()` - Previous timeframe levels
  - `add_sessions()` - Trading sessions
  - `add_retracements()` - Retracement percentages

### Files to Create
- `strategy.md` - Trading strategy documentation
- `src/data_loader.py` - TSLA data fetching
- `src/visualizer.py` - TSLA visualization
- `src/strategy.py` - Strategy implementation
- `src/backtester.py` - Backtesting engine

---

## Data Requirements

### TSLA Data Specifications
- **Source:** Twelve Data API
- **Symbol:** TSLA
- **Timeframes:** 1m, 5m, 15m, 1h, 1d (configurable)
- **History:** 2020-01-01 to present (recommended)
- **Format:** OHLCV (Open, High, Low, Close, Volume)
- **Timezone:** Europe/Madrid
- **Storage:** Local CSV cache in `data/tsla/`

### Data Schema
Required DataFrame columns (lowercase):
- `time` - Datetime index
- `open` - Opening price
- `high` - Highest price
- `low` - Lowest price
- `close` - Closing price
- `volume` - Trading volume
- `time_interval` - Timeframe identifier (e.g., "15m")
- `name` - Asset name ("TSLA")

---

## Visualization Specifications

### Chart Features
- **Type:** Candlestick chart (Plotly)
- **Indicators:** All 8 SMC indicators overlaid
- **Interactivity:** Zoom, pan, hover for details
- **Export:** HTML files for easy viewing
- **Color Scheme:**
  - Bullish candles: #77dd76 (green)
  - Bearish candles: #ff6962 (red)
  - FVG: Yellow (opacity 0.2)
  - Order Blocks: Purple (opacity 0.2)
  - BOS: Orange (opacity 0.2)
  - CHoCH: Blue (opacity 0.2)
  - Liquidity: Orange lines
  - Liquidity Swept: Red lines

### Chart Output
- Static HTML files for analysis
- Optional: Animated GIFs for presentations
- Size: Configurable (default 1200x800)

---

## Notes & Decisions

### 2025-10-17
- **Decision:** Use Twelve Data API (from existing DataHandler) instead of yfinance
- **Decision:** Adapt existing Plotly visualization code rather than using mplfinance
- **Decision:** Focus on TSLA as primary asset for initial development
- **Note:** Existing project has complete visualization reference in `tests/generate_gif.py`
- **Note:** Known issue - README mentions failing tests (need to investigate if relevant)

---

## Strategy Development Workflow

1. **Visual Analysis**
   - Generate charts with all indicators
   - Manually identify patterns and setups
   - Note what "good" setups look like

2. **Strategy Definition**
   - Document patterns in `strategy.md`
   - Define clear entry/exit rules
   - Specify confluence requirements

3. **Implementation**
   - Code strategy logic
   - Implement signal generation
   - Add risk management

4. **Testing**
   - Backtest on historical data
   - Analyze win rate, profit factor, drawdown
   - Iterate and refine

5. **Optimization**
   - Parameter tuning
   - Timeframe selection
   - Risk/reward optimization

---

## Resources

### Documentation
- Smart Money Concepts README: `/README.md`
- SMC Source Code: `/smartmoneyconcepts/smc.py`
- Visualization Reference: `/tests/generate_gif.py`

### External References
- Inner Circle Trader (ICT) Concepts
- Smart Money Concepts Trading Methodology
- Institutional Order Flow Analysis

---

## Future Enhancements

- [ ] Multi-asset support (expand beyond TSLA)
- [ ] Real-time data integration
- [ ] Paper trading capability
- [ ] Live trading execution
- [ ] Web dashboard for monitoring
- [ ] Telegram/Discord notifications
- [ ] Machine learning for parameter optimization
- [ ] Portfolio management features

---

## Troubleshooting

### Common Issues
- **API Rate Limits:** Twelve Data free tier has limits - cache data locally
- **Data Gaps:** Handle missing candles gracefully
- **Indicator Lag:** SMC indicators need sufficient lookback - ensure adequate history
- **Performance:** Use cached data for faster iteration during development

---

## Contact & Contribution

- **Author:** Project owner working with Claude Code
- **Purpose:** Educational and personal trading system development
- **Disclaimer:** For educational purposes only. Not financial advice.

---

*This document is a living reference and will be updated throughout development.*
