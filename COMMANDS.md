# Commands & Dashboard Reference

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

---

## Strategies

| Strategy | Docs | Run Command |
|----------|------|-------------|
| **Multi-Timeframe** — 1H sweep → 5M Event B → 5M FVG → 1M confirm | `mds/STRATEGY.md` | `python3 scripts/analyze_strategy.py --strategy multi-tf` |
| **Equilibrium Premium/Discount** — 1H sweep → 5M Event B → equilibrium zone → 1M confirm | `mds/STRATEGYII.md` | `python3 scripts/analyze_strategy.py --strategy multi-tf` (with equilibrium validation in config) |
| **3-OB LTF Confirmation** — 3 order block confluence with LTF entry | `mds/THREE_OB_LTF_CONFIRMATION.md` | `python3 scripts/analyze_strategy.py --strategy 3-ob` |
| **Magnet Chase** — Zone proximity targeting (closest active FVG/OB) | `mds/STRATEGY.md` | `python3 scripts/analyze_strategy.py --strategy magnet-chase` |
| **GMM Fibonacci** — GMM distribution + Fibonacci zones | `mds/gmm_fib_strategy.md` | Enable via `gmm_zones.enabled` in config, runs within `analyze_strategy.py` |

---

## Scripts & Commands

### `analyze_strategy.py` — Strategy runner & backtester

```bash
# Run 3-OB strategy (default) on TSLA
python3 scripts/analyze_strategy.py

# Multi-TF strategy with visualization
python3 scripts/analyze_strategy.py --strategy multi-tf --visualize

# Custom symbol, no backtest
python3 scripts/analyze_strategy.py --symbol META --no-backtest

# Batch mode — all symbols from config
python3 scripts/analyze_strategy.py --all-symbols --summary-output results/summary/batch_summary.csv

# 3-OB walkthrough
python3 scripts/analyze_strategy.py --strategy 3-ob --walkthrough-3ob

# Custom timeframes and capital
python3 scripts/analyze_strategy.py --high-tf 1h --mid-tf 5min --low-tf 1min --initial-capital 25000

# Hold overnight (default is intraday exit at 21:59)
python3 scripts/analyze_strategy.py --hold-overnight

# Export trade journal
python3 scripts/analyze_strategy.py --export-journal results/trades/my_trades.csv

# Magnet Chase strategy with visualization
python3 scripts/analyze_strategy.py --strategy magnet-chase --visualize
```

| Flag | Default | Description |
|------|---------|-------------|
| `--strategy` | `3-ob` | `multi-tf`, `3-ob`, or `magnet-chase` |
| `--symbol` | `TSLA` | Stock symbol |
| `--high-tf` | `1h` | High timeframe |
| `--mid-tf` | `5min` | Mid timeframe |
| `--low-tf` | `1min` | Low timeframe |
| `--initial-capital` | `10000.0` | Starting capital (USD) |
| `--no-backtest` | off | Skip backtest simulation |
| `--visualize` | off | Generate HTML visualization |
| `--hold-overnight` | off | Allow overnight holds |
| `--all-symbols` | off | Batch run all config symbols |
| `--summary-output` | `results/summary/batch_summary.csv` | Batch summary path |
| `--export-journal` | `auto` | Trade journal CSV path |
| `--walkthrough-3ob` | off | Candle-by-candle walkthrough |

**Output:** `results/trades/{symbol}_trades.csv`, `results/{symbol}_sweeps.html`

---

### `smc_dashboard.py` — SMC indicator dashboard

```bash
python3 scripts/smc_dashboard.py
python3 scripts/smc_dashboard.py --symbol META --timeframe 1h
python3 scripts/smc_dashboard.py --no-open
```

| Flag | Default | Description |
|------|---------|-------------|
| `--symbol` | `TSLA` | Symbol |
| `--timeframe` | `15min` | Timeframe |
| `--no-open` | off | Don't auto-open browser |

**Output:** `results/charts/smc_dashboard.html` — 3 tabs: zone lifecycle stats, reversal quality, OB analysis

---

### `ob_mfe_viewer.py` — OB MFE highlight chart

```bash
python3 scripts/ob_mfe_viewer.py
python3 scripts/ob_mfe_viewer.py --symbol TSLA --timeframe 15min
```

| Flag | Default | Description |
|------|---------|-------------|
| `--symbol` | `META` | Symbol |
| `--timeframe` | `15min` | Timeframe |

**Output:** `results/charts/ob_mfe_highlight.html` — Zoomable candlestick with OB rectangles, high-MFE OBs highlighted

---

### `price_distribution_indicators.py` — Split-view walkthrough

```bash
python3 scripts/price_distribution_indicators.py
python3 scripts/price_distribution_indicators.py --symbol AAPL --bins 100 --swing-length 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--symbol` | `TSLA` | Symbol |
| `--timeframe` | `15min` | Timeframe |
| `--bins` | `80` | Histogram bins |
| `--swing-length` | `50` | Swing length for OB detection |
| `--step` | `1` | Candles per frame |

**Output:** `results/charts/price_distribution_indicators.html` — Left: candlestick, Right: zone histogram. Arrow key navigation.

---

### `batch_3ob_comparison.py` — Batch 3-OB comparison

```bash
python3 scripts/batch_3ob_comparison.py
python3 scripts/batch_3ob_comparison.py --output results/my_comparison.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `results/batch_3ob_comparison.csv` | Output CSV path |

**Output:** `results/batch_3ob_comparison.csv` + `results/batch_3ob_comparison.html` — All symbols, two timeframe combos, sorted by return/day

---

### `analyze_1h_tv.py` / `analyze_5m_tv.py` — TradingView walkthroughs

```bash
python3 scripts/analyze_1h_tv.py
python3 scripts/analyze_1h_tv.py --symbol META

python3 scripts/analyze_5m_tv.py
python3 scripts/analyze_5m_tv.py --symbol AAPL
```

| Flag | Default | Description |
|------|---------|-------------|
| `--symbol` | `TSLA` | Symbol |

**Output:** `results/{symbol}_{tf}_tv_walkthrough/walkthrough.html`

---

## Dashboards & Visualizations

### SMC Dashboard (`smc_dashboard.html`)
Three-tab dashboard:
- **Zone Lifecycle** — FVG/OB mitigation rates, time-to-mitigation, respect vs disrespect stats
- **Reversal Quality** — How well zones predict reversals, bounce magnitude
- **OB Analysis** — Order block characteristics, MFE distribution, volume profiles

### OB MFE Viewer (`ob_mfe_highlight.html`)
Full-history zoomable candlestick chart with OB rectangles overlaid. High-MFE order blocks are highlighted to show which zones produced the largest favorable excursions.

### Price Distribution Split-View (`price_distribution_indicators.html`)
Left panel: candlestick chart building bar-by-bar. Right panel: zone histogram showing price distribution. Navigate with arrow keys to step through candles.

### Batch Comparison (`batch_3ob_comparison.html`)
Interactive Plotly charts comparing 3-OB strategy performance across all symbols. Sorted by return/day.

### Strategy Visualizations (`results/visualizations/`)
Per-symbol HTML charts showing sweep events, FVG zones, order blocks, and trade entries/exits overlaid on candlestick data.

### TradingView Walkthroughs (`results/{symbol}_*_tv_walkthrough/`)
Candle-by-candle replay using TradingView Lightweight Charts. Single self-contained HTML file per symbol/timeframe.

---

## Configuration

**`config/strategy_config.json`** — Central config for all strategies.

| Section | Purpose |
|---------|---------|
| `symbol` | Default symbol (`TSLA`) |
| `timeframes` | `high`, `mid`, `low` timeframe defaults |
| `gmm_zones` | GMM detection: lookback, step, components, fib levels, zone thresholds |
| `liquidity` | Sweep proximity threshold, lookback, FVG trigger toggle |
| `validation` | FVG/equilibrium validation toggles |
| `three_ob` | 3-OB timeframe, close break, min distance, shorts toggle |
| `magnet_chase` | Magnet Chase timeframe, confirmation TF, max distance %, zone preferences |
| `backtest` | Capital, overnight hold, trailing SL, min profit/RR, trend filters |
| `output` | Visualize toggle, journal export mode |
| `asset_options` | Symbol lists by category (stocks, forex, crypto, ETFs, indices, commodities) |

---

## Output Directory Structure

```
results/
├── charts/                          # Dashboard & viewer HTML files
│   ├── smc_dashboard.html
│   ├── ob_mfe_highlight.html
│   └── price_distribution_indicators.html
├── trades/                          # Trade journal CSVs (per symbol)
│   ├── tsla_trades.csv
│   ├── tsla_3ob_trades.csv
│   └── ...
├── summary/                         # Batch summary CSVs
│   └── batch_summary.csv
├── visualizations/                  # Strategy sweep HTML charts
├── walkthroughs/                    # Walkthrough HTML files
├── batch_3ob_comparison.csv         # Batch comparison data
└── batch_3ob_comparison.html        # Batch comparison charts
```
