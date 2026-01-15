# Command Reference

All available terminal commands for running the trading strategy analysis.

---

## 1. Main Strategy Analysis (`analyze_strategy.py`)

Scans historical data for complete trade setups and generates visual walkthroughs.

```bash
# Basic run (default: TSLA) - scans for setups AND runs backtest
python3 scripts/analyze_strategy.py

# Analyze a different stock (backtest included by default)
python3 scripts/analyze_strategy.py --symbol AAPL
python3 scripts/analyze_strategy.py --symbol META
python3 scripts/analyze_strategy.py --symbol AMZN

# Show partial setups (incomplete but close to triggering)
python3 scripts/analyze_strategy.py --animate-partials

# Skip backtest (only show signals, no P&L simulation)
python3 scripts/analyze_strategy.py --no-backtest

# Custom starting capital for backtest
python3 scripts/analyze_strategy.py --initial-capital 50000

# Export trade journal to CSV (auto-saves to results/{symbol}_trades.csv)
python3 scripts/analyze_strategy.py --export-journal

# Export to custom path
python3 scripts/analyze_strategy.py --export-journal my_trades.csv

# Combine all options
python3 scripts/analyze_strategy.py --symbol NVDA --initial-capital 25000 --animate-partials
```

---

## 2. 1H Timeframe Walkthrough (`analyze_1h_tv.py`)

Generates an interactive candle-by-candle visualization of the 1-hour timeframe.

```bash
# Candle-by-candle 1H visualization (default: TSLA)
python3 scripts/analyze_1h_tv.py

# Different stock
python3 scripts/analyze_1h_tv.py --symbol GOOGL
```

**Output:** `results/{symbol}_1h_tv_walkthrough/walkthrough.html`

---

## 3. 5M Timeframe Walkthrough (`analyze_5m_tv.py`)

Generates an interactive candle-by-candle visualization of the 5-minute timeframe.

```bash
# Candle-by-candle 5M visualization (default: TSLA)
python3 scripts/analyze_5m_tv.py

# Different stock
python3 scripts/analyze_5m_tv.py --symbol MSFT
```

**Output:** `results/{symbol}_5m_tv_walkthrough/walkthrough.html`

---

## Quick Reference Table

| Argument | Script | Description |
|----------|--------|-------------|
| `--symbol TICKER` | All scripts | Use any stock symbol (AAPL, META, NVDA, etc.) |
| `--animate-partials` | analyze_strategy.py | Visualize incomplete trade setups |
| `--no-backtest` | analyze_strategy.py | Skip P&L simulation (backtest runs by default) |
| `--initial-capital N` | analyze_strategy.py | Set starting capital for backtest (default: 10000) |
| `--export-journal [FILE]` | analyze_strategy.py | Save trade log to CSV (default: results/{symbol}_trades.csv) |

---

## Notes

- Data is automatically fetched and cached in `data/{symbol}/` when running with a new symbol
- Cached data is reused on subsequent runs (use `force_refresh=True` in code to bypass)
- All visualizations are saved as interactive HTML files in the `results/` directory
