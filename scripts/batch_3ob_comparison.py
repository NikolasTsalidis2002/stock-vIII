"""
Batch 3-OB Strategy Comparison

Runs the 3-OB strategy on all available symbols with two timeframe combos,
producing a CSV comparison table sorted by return/day.

Usage:
    python3 scripts/batch_3ob_comparison.py
    python3 scripts/batch_3ob_comparison.py --output results/my_comparison.csv
"""

import sys
import os
import json
import csv
import argparse
import time
import traceback
import webbrowser
from pathlib import Path
from datetime import datetime

import numpy as np

# Add src directory and project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import DataLoader
from strategy import ThreeOBStrategy
from backtesting import Backtester
from backtesting.models import SkippedTrade, SkipReason


TF_COMBOS = [
    {"timeframe": "5min", "confirmation_timeframe": "1min", "label": "5min/1min"},
    {"timeframe": "15min", "confirmation_timeframe": "5min", "label": "15min/5min"},
]

CSV_COLUMNS = [
    "symbol", "timeframe_combo",
    "analysis_start", "analysis_end", "trading_days",
    "total_trades", "long_trades", "short_trades",
    "win_rate", "profit_factor",
    "total_return_pct", "return_per_day_pct",
    "max_drawdown_pct",
    "avg_r_multiple",
    "max_consec_wins", "max_consec_losses",
    "avg_win_dollars", "avg_loss_dollars",
    "largest_win", "largest_loss",
    "initial_capital", "final_capital",
    "status", "error",
]


def get_all_symbols_from_config(config):
    symbols = []
    asset_options = config.get("asset_options", {})
    for category, symbol_list in asset_options.items():
        if category.startswith("_"):
            continue
        if isinstance(symbol_list, list):
            symbols.extend(symbol_list)
    return symbols


def get_missing_timeframes(symbol: str, timeframes: list) -> list:
    """Return list of timeframes that are not cached for this symbol."""
    safe_symbol = symbol.lower().replace('/', '_')
    data_dir = Path(__file__).parent.parent / 'data' / safe_symbol
    missing = []
    for tf in timeframes:
        cache_file = data_dir / f"{safe_symbol}_{tf}.csv"
        if not cache_file.exists():
            missing.append(tf)
    return missing


def pre_download_missing_data(symbols, tf_combos):
    """Download missing TF data only for symbols that already have at least one combo cached."""
    # Collect all unique TFs needed
    all_tfs = set()
    for combo in tf_combos:
        all_tfs.add(combo["timeframe"])
        all_tfs.add(combo["confirmation_timeframe"])

    # Only download for symbols that have at least one combo fully cached
    to_download = []
    for symbol in symbols:
        has_any_combo = any(
            len(get_missing_timeframes(symbol, [c["timeframe"], c["confirmation_timeframe"]])) == 0
            for c in tf_combos
        )
        if not has_any_combo:
            continue
        missing = get_missing_timeframes(symbol, list(all_tfs))
        for tf in missing:
            to_download.append((symbol, tf))

    if not to_download:
        print("All timeframe data is cached — no downloads needed.\n")
        return

    print(f"Need to download {len(to_download)} timeframes (~{len(to_download) * 8}s)")
    for i, (symbol, tf) in enumerate(to_download, 1):
        print(f"  [{i}/{len(to_download)}] Downloading {symbol} {tf}...", end=" ", flush=True)
        try:
            loader = DataLoader(symbol)
            loader.get_data(tf, force_refresh=True)
            print("OK")
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")
        if i < len(to_download):
            time.sleep(8)
    print()


def run_single(symbol, tf_combo, config):
    """Run 3-OB strategy for one symbol + TF combo. Returns a result dict."""
    timeframe = tf_combo["timeframe"]
    conf_tf = tf_combo["confirmation_timeframe"]
    label = tf_combo["label"]

    row = {col: "" for col in CSV_COLUMNS}
    row["symbol"] = symbol
    row["timeframe_combo"] = label

    try:
        loader = DataLoader(symbol)

        df_primary = loader.get_data(timeframe, force_refresh=False)
        df_confirmation = loader.get_data(conf_tf, force_refresh=False)

        three_ob_cfg = config.get("three_ob", {})
        strategy = ThreeOBStrategy(
            df_primary,
            close_break=three_ob_cfg.get("close_break", True),
            min_dist_pct=three_ob_cfg.get("min_dist_pct", 1.0) / 100.0,
            enable_shorts=three_ob_cfg.get("enable_shorts", True),
            df_confirmation=df_confirmation,
            confirmation_timeframe=conf_tf,
        )

        signals = strategy.scan_for_signals(max_signals=9999)

        if len(signals) == 0:
            row["status"] = "no_signals"
            # Still fill date range
            row["analysis_start"] = str(df_primary["time"].min().date())
            row["analysis_end"] = str(df_primary["time"].max().date())
            return row

        # Filter signals to confirmation data range
        conf_start = df_confirmation["time"].min()
        conf_end = df_confirmation["time"].max()
        tradeable_signals = [s for s in signals if conf_start <= s.timestamp_entry <= conf_end]

        if len(tradeable_signals) == 0:
            row["status"] = "no_signals"
            row["analysis_start"] = str(df_primary["time"].min().date())
            row["analysis_end"] = str(df_primary["time"].max().date())
            return row

        # Backtest settings
        bt_cfg = config.get("backtest", {})
        initial_capital = bt_cfg.get("initial_capital", 10000.0)

        backtester = Backtester(
            df_low=strategy.df_low,
            initial_capital=initial_capital,
            symbol=symbol,
            intraday_only=not bt_cfg.get("hold_overnight", True),
            min_profit_percent=bt_cfg.get("min_profit_percent", 0.5),
            min_rr_ratio=bt_cfg.get("min_rr_ratio", 2.0),
            trailing_sl_enabled=bt_cfg.get("trailing_sl_enabled", False),
            trailing_sl_activation_pct=bt_cfg.get("trailing_sl_activation_pct", 50.0),
            trailing_sl_swing_length=bt_cfg.get("trailing_sl_swing_length", 5),
        )

        results = backtester.run(tradeable_signals)
        m = backtester._metrics

        if m is None or m.total_trades == 0:
            row["status"] = "no_signals"
            row["analysis_start"] = str(df_primary["time"].min().date())
            row["analysis_end"] = str(df_primary["time"].max().date())
            return row

        # Date range
        start_date = df_primary["time"].min().date()
        end_date = df_primary["time"].max().date()
        trading_days = int(np.busday_count(start_date, end_date))
        if trading_days < 1:
            trading_days = 1

        return_per_day = m.total_return_percent / trading_days if trading_days > 0 else 0.0

        row.update({
            "analysis_start": str(start_date),
            "analysis_end": str(end_date),
            "trading_days": trading_days,
            "total_trades": m.total_trades,
            "long_trades": m.long_trades,
            "short_trades": m.short_trades,
            "win_rate": round(m.win_rate * 100, 2),
            "profit_factor": round(m.profit_factor, 2),
            "total_return_pct": round(m.total_return_percent, 2),
            "return_per_day_pct": round(return_per_day, 4),
            "max_drawdown_pct": round(m.max_drawdown_percent, 2),
            "avg_r_multiple": round(m.average_r_multiple, 2),
            "max_consec_wins": m.max_consecutive_wins,
            "max_consec_losses": m.max_consecutive_losses,
            "avg_win_dollars": round(m.average_win_dollars, 2),
            "avg_loss_dollars": round(m.average_loss_dollars, 2),
            "largest_win": round(m.largest_win_dollars, 2),
            "largest_loss": round(m.largest_loss_dollars, 2),
            "initial_capital": round(m.initial_capital, 2),
            "final_capital": round(m.final_capital, 2),
            "status": "success",
        })
        return row

    except Exception as e:
        row["status"] = "failed"
        row["error"] = f"{type(e).__name__}: {e}"
        return row


def main():
    parser = argparse.ArgumentParser(description="Batch 3-OB strategy comparison across all symbols")
    parser.add_argument("--output", default="results/batch_3ob_comparison.csv", help="Output CSV path")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "strategy_config.json"
    with open(config_path) as f:
        config = json.load(f)

    symbols = get_all_symbols_from_config(config)
    print(f"Found {len(symbols)} symbols across all categories")
    print(f"Running {len(TF_COMBOS)} TF combos per symbol\n")

    # Pre-download missing timeframe data
    pre_download_missing_data(symbols, TF_COMBOS)

    all_rows = []
    total = len(symbols) * len(TF_COMBOS)
    done = 0

    for symbol in symbols:
        for combo in TF_COMBOS:
            done += 1
            print(f"[{done}/{total}] {symbol} {combo['label']}...", end=" ", flush=True)
            row = run_single(symbol, combo, config)
            all_rows.append(row)
            status_msg = row["status"]
            if row.get("error"):
                status_msg += f" ({row['error'][:80]})"
            print(status_msg)

    # Sort by return_per_day_pct descending
    def sort_key(r):
        try:
            return float(r["return_per_day_pct"]) if r["return_per_day_pct"] != "" else float("-inf")
        except (ValueError, TypeError):
            return float("-inf")

    all_rows.sort(key=sort_key, reverse=True)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {output_path}")

    # Print summary table
    success_rows = [r for r in all_rows if r["status"] == "success"]
    if success_rows:
        print(f"\n{'='*120}")
        print(f"TOP RESULTS BY RETURN/DAY (showing {min(len(success_rows), 20)} of {len(success_rows)} successful runs)")
        print(f"{'='*120}")
        print(f"{'Symbol':<12} {'TF Combo':<12} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Return%':>9} {'Ret/Day%':>10} {'MaxDD%':>8} {'AvgR':>7}")
        print(f"{'-'*12} {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*10} {'-'*8} {'-'*7}")
        for r in success_rows[:20]:
            print(
                f"{r['symbol']:<12} {r['timeframe_combo']:<12} "
                f"{r['total_trades']:>7} {float(r['win_rate']):>7.1f} {float(r['profit_factor']):>7.2f} "
                f"{float(r['total_return_pct']):>9.2f} {float(r['return_per_day_pct']):>10.4f} "
                f"{float(r['max_drawdown_pct']):>8.2f} {float(r['avg_r_multiple']):>7.2f}"
            )

    # Summary counts
    statuses = {}
    for r in all_rows:
        s = r["status"]
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nStatus breakdown: {statuses}")

    # Generate visualization
    if success_rows:
        html_path = output_path.with_suffix('.html')
        generate_visualization(success_rows, html_path)


def generate_visualization(rows, html_path):
    """Generate an interactive HTML report with plotly charts from results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            "Return/Day % by Symbol+TF (sorted)",
            "Win Rate vs Return/Day (sized by trades)",
            "Total Return % — 5min/1min vs 15min/5min by Symbol",
            "Top 10 & Bottom 10 Results",
        ],
        row_heights=[0.3, 0.25, 0.25, 0.2],
        specs=[[{"type": "bar"}], [{"type": "scatter"}], [{"type": "bar"}], [{"type": "table"}]],
        vertical_spacing=0.08,
    )

    # Parse floats
    for r in rows:
        for k in ["return_per_day_pct", "win_rate", "total_return_pct", "total_trades",
                   "profit_factor", "max_drawdown_pct", "avg_r_multiple"]:
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                r[k] = 0.0

    # --- Chart 1: Horizontal bar of return/day ---
    sorted_rows = sorted(rows, key=lambda r: r["return_per_day_pct"])
    labels = [f"{r['symbol']} {r['timeframe_combo']}" for r in sorted_rows]
    values = [r["return_per_day_pct"] for r in sorted_rows]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig.add_trace(go.Bar(
        y=labels, x=values, orientation='h', marker_color=colors,
        name="Return/Day %", showlegend=False,
    ), row=1, col=1)

    # --- Chart 2: Scatter — win rate vs return/day ---
    combo_colors = {"5min/1min": "#3498db", "15min/5min": "#e67e22"}
    for combo_label, color in combo_colors.items():
        subset = [r for r in rows if r["timeframe_combo"] == combo_label]
        if not subset:
            continue
        fig.add_trace(go.Scatter(
            x=[r["win_rate"] for r in subset],
            y=[r["return_per_day_pct"] for r in subset],
            mode="markers+text",
            marker=dict(
                size=[max(8, min(40, r["total_trades"])) for r in subset],
                color=color, opacity=0.7,
            ),
            text=[r["symbol"] for r in subset],
            textposition="top center", textfont=dict(size=9),
            name=combo_label,
        ), row=2, col=1)

    # --- Chart 3: Grouped bar — side by side comparison ---
    # Get symbols that have both combos
    sym_combos = {}
    for r in rows:
        sym_combos.setdefault(r["symbol"], {})[r["timeframe_combo"]] = r["total_return_pct"]
    symbols_both = [s for s, combos in sym_combos.items() if len(combos) == 2]
    symbols_both.sort(key=lambda s: sym_combos[s].get("5min/1min", 0), reverse=True)

    for combo_label, color in combo_colors.items():
        fig.add_trace(go.Bar(
            x=symbols_both,
            y=[sym_combos[s].get(combo_label, 0) for s in symbols_both],
            name=combo_label, marker_color=color,
        ), row=3, col=1)

    # --- Chart 4: Table — top 10 + bottom 10 ---
    sorted_by_rpd = sorted(rows, key=lambda r: r["return_per_day_pct"], reverse=True)
    table_rows = sorted_by_rpd[:10] + sorted_by_rpd[-10:] if len(sorted_by_rpd) > 20 else sorted_by_rpd
    headers = ["Symbol", "TF Combo", "Trades", "Win%", "PF", "Return%", "Ret/Day%", "MaxDD%"]
    cell_values = [
        [r["symbol"] for r in table_rows],
        [r["timeframe_combo"] for r in table_rows],
        [int(r["total_trades"]) for r in table_rows],
        [round(r["win_rate"], 1) for r in table_rows],
        [round(r["profit_factor"], 2) for r in table_rows],
        [round(r["total_return_pct"], 2) for r in table_rows],
        [round(r["return_per_day_pct"], 4) for r in table_rows],
        [round(r["max_drawdown_pct"], 2) for r in table_rows],
    ]
    fill_colors = [["#d4edda" if r["return_per_day_pct"] >= 0 else "#f8d7da" for r in table_rows]] * len(headers)

    fig.add_trace(go.Table(
        header=dict(values=headers, fill_color="#343a40", font=dict(color="white", size=12), align="center"),
        cells=dict(values=cell_values, fill_color=fill_colors, align="center", font=dict(size=11)),
    ), row=4, col=1)

    fig.update_layout(
        height=1800, width=1200,
        title_text="Batch 3-OB Strategy Comparison",
        barmode="group",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Return/Day %", row=1, col=1)
    fig.update_xaxes(title_text="Win Rate %", row=2, col=1)
    fig.update_yaxes(title_text="Return/Day %", row=2, col=1)
    fig.update_yaxes(title_text="Total Return %", row=3, col=1)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))
    print(f"Visualization saved to {html_path}")
    webbrowser.open(f"file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
