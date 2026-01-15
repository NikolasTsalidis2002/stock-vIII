# TSLA Smart Money Concepts Trading System

A sophisticated multi-timeframe trading strategy analyzer for Tesla (TSLA) stock using Smart Money Concepts (SMC) indicators to identify high-probability reversal trades based on institutional order flow patterns.

## Overview

This project implements an automated trading signal detection system that analyzes TSLA price action across multiple timeframes (1H, 5M, 1M) using institutional trading concepts. The strategy identifies potential reversal points by detecting liquidity sweeps, break of structure events, and fair value gaps.

**Key Capabilities:**
- Multi-timeframe analysis with synchronized data alignment
- 8+ Smart Money Concepts indicators
- Professional TradingView-style visualizations
- Automated trade signal generation with complete audit trail

## Features

- **Multi-Timeframe Analysis** - Coordinates 1-Hour, 5-Minute, and 1-Minute timeframes for precise entry signals
- **SMC Indicators** - Fair Value Gaps, Order Blocks, Break of Structure, Change of Character, Liquidity detection, and more
- **TradingView Visualizations** - Interactive HTML charts with candle-by-candle walkthroughs
- **Modular Architecture** - Clean separation of concerns with individual detector components
- **Partial Setup Tracking** - Analyzes incomplete setups to understand where patterns fail

## Quick Start

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
# Find complete trade setups
python3 scripts/analyze_strategy.py

# Visualize partial setups (incomplete but close)
python3 scripts/analyze_strategy.py --animate-partials
```

### 5. View Results
Open `results/strategy_examples/master_index.html` in your browser to see all detected trade signals.

### 6. Deactivate (when done)
```bash
deactivate
```

## Trading Strategy

The strategy uses a **5-step progressive signal detection workflow** that validates each stage before advancing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-TIMEFRAME WORKFLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: LIQUIDITY SWEEP (1H)                                   │
│  ├── Detect inflexion points (local peaks/valleys)              │
│  └── Identify when price sweeps past these levels               │
│                         ↓                                        │
│  STEP 2: EVENT B (5M)                                           │
│  ├── Find Break of Structure (BOS) in entry direction           │
│  └── OR find Inverse Fair Value Gap (IFVG)                      │
│                         ↓                                        │
│  STEP 3: VALIDATION (5M)                                        │
│  ├── Validate with Fair Value Gap (FVG)                         │
│  └── OR validate with Demand Zone (Order Block)                 │
│                         ↓                                        │
│  STEP 4: CONFIRMATION (1M)                                      │
│  ├── Detect BOS on 1-minute in entry direction                  │
│  └── OR detect IFVG on 1-minute                                 │
│                         ↓                                        │
│  STEP 5: TRADE ENTRY                                            │
│  └── All 4 conditions met = Complete trade signal               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Entry Direction Logic
- **Bullish Sweep** (price sweeps a low, then closes above) → **LONG** entry
- **Bearish Sweep** (price sweeps a high, then closes below) → **SHORT** entry

### Invalidation Rules
The strategy includes built-in invalidation logic:
- If sweep level is broken: Entire setup abandoned
- If Event B is invalidated: Return to Step 2
- If Validation zone is broken: Return to Step 3
- If price moves >3% from sweep: Setup abandoned

## SMC Indicators

### Fair Value Gap (FVG)
A price gap indicating imbalance between buyers and sellers. Represents zones where institutional orders may cluster.
- **Bullish FVG**: Previous candle high < next candle low
- **Bearish FVG**: Previous candle low > next candle high

### Order Blocks (OB)
Price ranges where large institutional orders exist, identified at the last candle before strong price moves.

### Break of Structure (BOS)
Price breaks a significant swing high/low in trend direction, confirming trend continuation.
- **Bullish BOS**: Price breaks above previous swing high
- **Bearish BOS**: Price breaks below previous swing low

### Change of Character (CHoCH)
Price breaks structure in the opposite direction, signaling potential trend reversal.

### Liquidity
Clusters of swing highs/lows within tight price ranges, representing stop-loss clusters that institutions target.

### Swing Highs/Lows
Significant pivot points in price action, forming the basis for structure analysis.

### Inflexion Points (Custom)
More precise local extrema detection using 3-point statistical comparison for accurate liquidity sweep identification.

## Project Structure

```
smart-money-concepts/
├── scripts/                           # Entry point scripts
│   ├── analyze_strategy.py           # Main strategy analysis
│   ├── analyze_1h_tv.py              # 1H timeframe visualization
│   └── analyze_5m_tv.py              # 5M timeframe visualization
│
├── src/
│   ├── data_loader.py                # TSLA data fetching (Twelve Data API)
│   ├── tradingview_visualizer.py     # TradingView-style visualizations
│   ├── tradingview_strategy_analyzer.py # Multi-TF strategy visualizer
│   │
│   ├── indicators/                   # SMC indicator implementations
│   │   ├── smc.py                    # Standard SMC library
│   │   └── smc_custom.py             # Custom implementations
│   │
│   └── strategy/                     # Modular strategy engine
│       ├── __init__.py               # Public API
│       ├── models.py                 # Data models (TradeSignal, etc.)
│       ├── strategy_engine.py        # Main orchestrator
│       ├── timeframe_manager.py      # Multi-TF data alignment
│       ├── liquidity_detector.py     # 1H sweep detection
│       ├── event_b_detector.py       # 5M BOS/IFVG detection
│       ├── validation_detector.py    # 5M FVG/OB validation
│       └── confirmation_detector.py  # 1M confirmation
│
├── data/tsla/                        # Cached TSLA data
│   ├── tsla_1min.csv
│   ├── tsla_5min.csv
│   ├── tsla_15min.csv
│   ├── tsla_1h.csv
│   └── tsla_1day.csv
│
├── results/                          # Generated outputs
│   ├── strategy_examples/            # Trade signal visualizations
│   ├── 1h_tv_walkthrough/           # 1H candle walkthroughs
│   └── 5m_tv_walkthrough/           # 5M candle walkthroughs
│
├── requirements.txt                  # Python dependencies
└── CLAUDE.md                        # Detailed project documentation
```

## Usage Examples

### Find Complete Trade Setups
```bash
python3 scripts/analyze_strategy.py
```
Scans historical data for complete trade signals where all 4 conditions are met. Generates HTML visualizations in `results/strategy_examples/`.

### Analyze Partial Setups
```bash
python3 scripts/analyze_strategy.py --animate-partials
```
Finds setups that met 1-3 conditions but didn't complete. Useful for understanding where patterns fail and refining the strategy.

### Single Timeframe Visualization
```bash
# Generate 1H candle-by-candle walkthrough
python3 scripts/analyze_1h_tv.py

# Generate 5M candle-by-candle walkthrough
python3 scripts/analyze_5m_tv.py
```
Creates interactive HTML files showing indicators appearing progressively as candles form.

## Data Source

The project uses the **Twelve Data API** to fetch TSLA historical data:
- Supports multiple timeframes: 1min, 5min, 15min, 1h, 1day
- Smart caching: Saves data locally, loads from cache when available
- Update capability: Merges new candles with existing cache
- Data validation: Checks for OHLC integrity, nulls, and duplicates

## Output Format

### Trade Signal
When all conditions are met, a `TradeSignal` object is generated containing:
- Timestamps for each step (1H sweep, 5M Event B, 5M validation, 1M confirmation)
- Entry direction (long/short)
- Entry price
- Condition types (BOS, IFVG, FVG, etc.)
- DataFrame indices for visualization

### Visualizations
- **Master Index**: Grid view of all detected trades with statistics
- **Individual Trade Pages**: 3-panel synchronized view (1H, 5M, 1M)
- **Candle Walkthroughs**: Frame-by-frame analysis with keyboard navigation

## Requirements

- Python 3.8+
- pandas >= 2.0.2
- numpy >= 1.24.3
- plotly
- requests
- numba >= 0.58.1

## Disclaimer

**This project is for educational and research purposes only.**

- This is not financial advice
- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Always do your own research and consider consulting a financial advisor
- The authors are not responsible for any financial losses incurred

## License

This project is for personal use and educational purposes.

## Resources

- [Inner Circle Trader (ICT) Concepts](https://www.youtube.com/@TheInnerCircleTrader)
- Smart Money Concepts Trading Methodology
- Institutional Order Flow Analysis
