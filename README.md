# Smart Money Concepts Trading System

Multi-strategy, multi-asset trading signal analyzer using Smart Money Concepts (SMC) indicators with backtesting, statistical filtering, and interactive TradingView-style visualizations.

## Quick Start

```bash
# 1. Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run 3-OB strategy on default symbol
python3 scripts/analyze_strategy.py

# 4. Run multi-TF strategy with visualization
python3 scripts/analyze_strategy.py --strategy multi-tf --visualize

# 5. Batch comparison across all symbols
python3 scripts/batch_3ob_comparison.py
```

---

## Strategies

### Multi-Timeframe Strategy (`--strategy multi-tf`)

A 5-stage liquidity sweep pipeline that coordinates three timeframes (default 1H / 5M / 1M):

```
 STAGE 1: LIQUIDITY SWEEP (High TF)
 ├── Detect inflexion points (local peaks/valleys)
 └── Identify when price sweeps past these levels
                    ↓
 STAGE 2: EVENT B (Mid TF)
 ├── Find Break of Structure (BOS) in entry direction
 └── OR find Inverse Fair Value Gap (IFVG)
                    ↓
 STAGE 3: VALIDATION (Mid TF)
 ├── Validate with FVG respect
 ├── OR validate with Equilibrium Premium/Discount zone
 └── OR validate with Demand Zone (Order Block)
                    ↓
 STAGE 4: CONFIRMATION (Low TF)
 ├── Detect BOS on lowest TF in entry direction
 └── OR detect IFVG on lowest TF
                    ↓
 STAGE 5: TRADE ENTRY
 └── All 4 conditions met → complete signal
```

- **Bullish sweep** (low swept, closes above) triggers LONG
- **Bearish sweep** (high swept, closes below) triggers SHORT
- Invalidation rules reset to the appropriate stage when conditions break

### 3-OB Strategy (`--strategy 3-ob`)

A 4-state order block state machine tracking OB-A, OB-B, and OB-C:

| State | Description |
|-------|-------------|
| **0** | OB-A discovered + opposite OB-C found |
| **1** | Waiting for OB-B (same direction as OB-A) |
| **2** | Waiting for OB-B retouch |
| **3** | Pending LTF confirmation (BOS or Inverse FVG) |

| Setup | OB-A | OB-C | OB-B | Entry |
|-------|------|------|------|-------|
| Long | Bullish | Bearish (above) | Bullish | LTF bullish BOS or bearish IFVG |
| Short | Bearish | Bullish (below) | Bearish | LTF bearish BOS or bullish IFVG |

When `confirmation_timeframe` is configured (e.g. `"1min"`), the engine drops to the lower timeframe for precise entry at the OB-B retouch window.

### Magnet Chase Strategy (`--strategy magnet-chase`)

A zone-proximity targeting strategy that chases the closest active FVG or OB zone:

1. Scans all active (unmitigated) FVG and OB zones on the configured timeframe
2. Ranks zones by distance from current price, filtering by `max_distance_pct`
3. Prefers OB zones over FVG zones when equidistant (OBs have stronger institutional backing)
4. Enters when price approaches the nearest qualifying zone with LTF confirmation

- Configurable via `magnet_chase` section in `config/strategy_config.json`
- Visualization: `src/visualization/magnet_chase_viewer.py`

---

## TradingView Walkthroughs

Interactive candle-by-candle HTML viewers using TradingView LightweightCharts with arrow-key navigation.

```bash
# 1H walkthrough
python3 scripts/analyze_1h_tv.py --symbol AAPL

# 5M walkthrough
python3 scripts/analyze_5m_tv.py --symbol META

# 3-OB state machine walkthrough
python3 scripts/analyze_strategy.py --walkthrough-3ob
```

- Indicators appear progressively as each candle forms
- OB zones, BOS lines, FVGs rendered as D3 SVG overlays
- Output: `results/<symbol>_<tf>_tv_walkthrough/walkthrough.html`

---

## Strategy Analysis

`scripts/analyze_strategy.py` is the main entry point.

```
Usage: python3 scripts/analyze_strategy.py [OPTIONS]

Options:
  --strategy {multi-tf,3-ob,magnet-chase}  Strategy to run (default: 3-ob)
  --symbol SYMBOL               Asset symbol (default: TSLA)
  --high-tf TF                  High timeframe (default: 1h)
  --mid-tf TF                   Mid timeframe (default: 5min)
  --low-tf TF                   Low timeframe (default: 1min)
  --visualize                   Generate unified HTML visualization
  --no-backtest                 Skip backtest simulation
  --initial-capital FLOAT       Starting capital (default: 10000)
  --hold-overnight              Allow overnight positions
  --export-journal [PATH]       Export trade journal CSV (default: auto path)
  --all-symbols                 Batch mode: run all symbols from config
  --summary-output PATH         Batch summary CSV path
  --walkthrough-3ob             Generate 3-OB candle walkthrough
```

Examples:

```bash
# 3-OB on NVDA with visualization
python3 scripts/analyze_strategy.py --symbol NVDA --visualize

# Multi-TF with custom timeframes
python3 scripts/analyze_strategy.py --strategy multi-tf --high-tf 4h --mid-tf 15min --low-tf 5min

# Magnet Chase strategy with visualization
python3 scripts/analyze_strategy.py --strategy magnet-chase --visualize

# Batch across all configured symbols
python3 scripts/analyze_strategy.py --all-symbols --summary-output results/summary/batch.csv
```

---

## Batch Analysis

`scripts/batch_3ob_comparison.py` runs the 3-OB strategy across all configured symbols with two timeframe combinations (5min/1min and 15min/5min).

```bash
python3 scripts/batch_3ob_comparison.py --output results/batch_3ob_comparison.csv
```

- Auto-downloads missing timeframe data (with rate-limit delays)
- Backtests each symbol + TF combo
- Produces a CSV sorted by return/day with metrics: trades, win rate, profit factor, return%, max drawdown, R-multiple, consecutive wins/losses
- Generates an interactive Plotly HTML report (auto-opens in browser) with:
  - Return/day bar chart by symbol
  - Win rate vs return/day scatter plot
  - 5min/1min vs 15min/5min comparison
  - Top 10 and bottom 10 results table

---

## SMC Indicators

### Standard Library (`src/indicators/smc.py`)

| Indicator | Description |
|-----------|-------------|
| **FVG** | Fair Value Gap — price imbalance zones (bullish/bearish) |
| **OB** | Order Blocks — institutional order zones with volume |
| **BOS** | Break of Structure — trend continuation confirmation |
| **CHoCH** | Change of Character — trend reversal signal |
| **Liquidity** | Swing high/low clusters representing stop-loss pools |
| **Swing Highs/Lows** | Configurable window-based pivot detection |
| **Previous High/Low** | Prior timeframe levels |
| **Retracements** | Fibonacci retracement percentages |

### Custom Indicators (`src/indicators/smc_custom.py`)

| Indicator | Description |
|-----------|-------------|
| **Inflexion Points** | 3-point mathematical extrema (concave peaks, convex valleys) with proximity-based near-sweep detection |
| **Trend-Aware BOS** | Close/open break confirmation with explicit state tracking |

Inflexion point algorithm:
```
Peak:   high[i] >= high[i-1] AND high[i] > high[i+1]
Valley: low[i]  <= low[i-1]  AND low[i]  < low[i+1]
```

---

## Statistical Models

### ARMA Trend Filter (`src/arma_trend_analysis.py`)

Autoregressive Moving Average model used as a trade direction filter.

1. Converts prices to log returns
2. Fits ARMA(p, q) with AIC-based order selection
3. Calculates Signal-to-Noise Ratio: `SNR = |drift| / residual_std`
4. SNR < 0.05 → neutral (no filtering); drift > 0 → bullish (LONG only); drift < 0 → bearish (SHORT only)

Enable via `backtest.trend_filter_enabled` in config.

### GMM Zone Detection (`src/strategy/gmm_zone_detector.py`)

Gaussian Mixture Model for price distribution regime detection with Fibonacci levels.

1. Generates discrete price levels from candle ranges
2. Fits GMM with 1 to `max_components` components
3. Selects optimal count via elbow method or min BIC
4. Calculates Fibonacci levels within the detected zone
5. Classifies price position: premium (0.786-1.0) → SHORT bias, discount (0.0-0.236) → LONG bias

Enable via `gmm_zones.enabled` in config.

---

## Backtesting Engine

Located in `src/backtesting/`. Components: `backtester.py` (orchestrator), `trade_simulator.py` (execution), `performance_tracker.py` (metrics), `trade_journal.py` (CSV export).

### Features

- **Compounding**: full capital deployed per trade, wins increase position size
- **Trailing Stop Loss**: activates after price reaches a configurable % of TP distance, then trails to nearest swing level on every candle
- **Trade Filters**: minimum profit %, minimum R:R ratio, ARMA trend filter, 1H BOS trend filter
- **Intraday Mode**: exits all positions at market close when `hold_overnight` is false
- **Skip Reasons**: logs why trades were filtered (low R:R, wrong trend, insufficient profit)

### Metrics

Total P&L, win rate, profit factor, max drawdown, average R-multiple, max consecutive wins/losses, largest win/loss, long/short breakdown, capital curve.

### Exit Priority

1. Take Profit
2. Stop Loss
3. Trailing SL (if enabled and activated)
4. Market close (if intraday only)

---

## Configuration

All settings live in `config/strategy_config.json`. Key sections:

| Section | Key Settings |
|---------|-------------|
| `symbol` | Asset to analyze (any Twelve Data symbol) |
| `timeframes` | `high`, `mid`, `low` timeframe triplet |
| `three_ob` | `timeframe`, `confirmation_timeframe`, `close_break`, `min_dist_pct`, `enable_shorts` |
| `liquidity` | `sweep_proximity_threshold_percent`, `high_tf_lookback_candles`, `use_fvg_trigger` |
| `validation` | `use_fvg_validation`, `use_equilibrium_validation`, `require_fvg_in_equilibrium` |
| `gmm_zones` | `enabled`, `lookback_candles`, `max_components`, `selection_method`, `take_profit_method` |
| `magnet_chase` | `timeframe`, `confirmation_timeframe`, `max_distance_pct`, zone preference settings |
| `backtest` | `initial_capital`, `trailing_sl_enabled`, `trailing_sl_activation_pct`, `min_rr_ratio`, `trend_filter_enabled` |
| `output` | `visualize`, `export_journal` |

---

## Supported Assets

90+ symbols across 9 categories, all fetched via the Twelve Data API:

| Category | Symbols |
|----------|---------|
| **Commodities** | XAU/USD, XAG/USD, BRENT, WTI, NG |
| **Forex** | EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD, EUR/GBP |
| **Crypto** | BTC/USD, ETH/USD, SOL/USD, XRP/USD, ADA/USD, DOGE/USD, AVAX/USD, LINK/USD |
| **Tech** | AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AMD, INTC, NFLX, SHOP, ABNB, CRM, ORCL, ADBE |
| **Finance** | JPM, BAC, GS, MS, V, MA, PYPL, C |
| **Healthcare** | JNJ, UNH, PFE, ABBV, MRK, LLY |
| **Energy** | XOM, CVX, COP, SLB, EOG |
| **ETFs** | SPY, QQQ, IWM, DIA, GLD, SLV, USO, XLF, XLE, ARKK |
| **Indices** | SPX, NDX, DJI, VIX |

---

## Project Structure

```
smart-money-concepts/
├── scripts/
│   ├── analyze_strategy.py            # Main strategy analyzer (multi-tf & 3-ob)
│   ├── batch_3ob_comparison.py        # Batch comparison across all symbols
│   ├── ob_mfe_viewer.py               # OB MFE highlight chart
│   ├── price_distribution_indicators.py # Split-view walkthrough
│   ├── analyze_1h_tv.py               # 1H TradingView walkthrough
│   └── analyze_5m_tv.py               # 5M TradingView walkthrough
│
├── src/
│   ├── data_loader.py                 # Data fetching (Twelve Data API)
│   ├── tradingview_visualizer.py      # TradingView-style chart generation
│   ├── strategy_visualizer.py         # Multi-TF sweep visualizer
│   ├── arma_trend_analysis.py         # ARMA trend filter
│   │
│   ├── indicators/
│   │   ├── smc.py                     # Standard SMC library (v0.0.26)
│   │   └── smc_custom.py             # Inflexion points, trend-aware BOS
│   │
│   ├── strategy/
│   │   ├── models.py                  # TradeSignal, enums, data classes
│   │   ├── strategy_engine.py         # Multi-TF orchestrator
│   │   ├── timeframe_manager.py       # Multi-TF data alignment
│   │   ├── liquidity_detector.py      # Stage 1: 1H sweep detection
│   │   ├── event_b_detector.py        # Stage 2: BOS/IFVG detection
│   │   ├── confirmation_detector.py   # Stage 4: LTF confirmation
│   │   ├── equilibrium_validator.py   # Premium/discount validation
│   │   ├── exit_target_finder.py      # TP/SL calculation
│   │   ├── gmm_zone_detector.py       # GMM-based zone detection
│   │   ├── three_ob_strategy.py       # 3-OB strategy wrapper
│   │   ├── three_ob_engine.py         # 3-OB state machine
│   │   ├── magnet_chase_engine.py     # Magnet Chase strategy engine
│   │   └── zone_proximity.py          # Zone proximity ranking
│   │
│   ├── backtesting/
│   │   ├── backtester.py              # Backtest orchestrator
│   │   ├── trade_simulator.py         # Trade execution simulator
│   │   ├── performance_tracker.py     # Metrics calculation
│   │   ├── trade_journal.py           # CSV trade log export
│   │   └── models.py                  # TradeResult, PerformanceMetrics
│   │
│   └── visualization/
│       ├── core.py                    # Shared visualization utilities
│       ├── three_ob_walkthrough.py    # 3-OB candle walkthrough
│       ├── three_ob_signal_viewer.py  # 3-OB signal dashboard
│       └── magnet_chase_viewer.py     # Magnet Chase visualization
│
├── config/
│   └── strategy_config.json           # All strategy & backtest settings
│
├── data/                              # Cached market data by symbol
├── results/                           # Generated outputs
│   ├── trades/                        # Trade journal CSVs
│   ├── summary/                       # Batch summary reports
│   ├── visualizations/                # HTML signal viewers
│   └── walkthroughs/                  # Candle-by-candle walkthroughs
│
├── mds/                               # Strategy documentation
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Disclaimer

**This project is for educational and research purposes only.** This is not financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research.

## Resources

- [Inner Circle Trader (ICT) Concepts](https://www.youtube.com/@TheInnerCircleTrader)
- Smart Money Concepts Trading Methodology
- Institutional Order Flow Analysis
