"""
Strategy Analysis Script

Scans historical stock data for complete trade setups and generates
visual walkthroughs showing the step-by-step strategy execution.

Usage:
    python3 scripts/analyze_strategy.py                         # Default: TSLA with 1h/5min/1min
    python3 scripts/analyze_strategy.py --symbol META           # Use different symbol
    python3 scripts/analyze_strategy.py --symbol META --visualize  # Generate unified HTML
    python3 scripts/analyze_strategy.py --high-tf 4h --mid-tf 15min --low-tf 5min  # Custom timeframes

Output:
    --visualize flag generates: results/{symbol}_sweeps.html
    (Single interactive file with all sweeps, tabs for timeframes, arrow key navigation)
"""

import sys
import os
import json
from pathlib import Path
import argparse

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import DataLoader
from strategy import MultiTimeframeStrategy
from strategy_visualizer import StrategySweepVisualizer
from backtesting import Backtester


def load_config():
    """Load configuration from JSON file.

    Returns:
        dict: Configuration values, or empty dict if file not found.
    """
    config_path = Path(__file__).parent.parent / 'config' / 'strategy_config.json'
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            print(f"Loaded config from: {config_path}")
            return config
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in config file: {e}")
            return {}
    return {}


def generate_master_index(signals, output_dir, symbol='TSLA'):
    """
    Generate master index HTML with links to all trade examples.

    Args:
        signals: List of TradeSignal objects
        output_dir: Output directory path
        symbol: Stock symbol for titles
    """
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{symbol} Strategy Analysis - Trade Examples</title>
    <style>
        body {{
            background-color: #0c0e12;
            color: white;
            font-family: Arial, sans-serif;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #00ff00;
            border-bottom: 3px solid #00ff00;
            padding-bottom: 15px;
            text-align: center;
        }}
        .subtitle {{
            text-align: center;
            color: #aaa;
            margin-bottom: 30px;
        }}
        .strategy-overview {{
            background-color: #1a1d24;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
        }}
        .strategy-overview h2 {{
            color: #00ff00;
            margin-top: 0;
        }}
        .strategy-step {{
            margin: 10px 0;
            padding: 10px;
            background-color: #0c0e12;
            border-left: 4px solid #00ff00;
        }}
        .trades-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .trade-card {{
            background-color: #1a1d24;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 20px;
            text-decoration: none;
            color: white;
            transition: all 0.3s;
        }}
        .trade-card:hover {{
            border-color: #00ff00;
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 255, 0, 0.3);
        }}
        .trade-number {{
            color: #00ff00;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .trade-direction {{
            font-size: 20px;
            margin-bottom: 10px;
        }}
        .trade-direction.long {{
            color: #77dd76;
        }}
        .trade-direction.short {{
            color: #ff6962;
        }}
        .trade-details {{
            color: #aaa;
            font-size: 14px;
            line-height: 1.6;
        }}
        .trade-details-item {{
            margin: 5px 0;
        }}
        .checkmark {{
            color: #00ff00;
        }}
        .stats {{
            background-color: #1a3a1a;
            border: 2px solid #00ff00;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 32px;
            color: #00ff00;
            font-weight: bold;
        }}
        .stat-label {{
            color: #aaa;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>{symbol} Multi-Timeframe Trading Strategy</h1>
    <div class="subtitle">Visual Analysis of Complete Trade Setups</div>

    <div class="strategy-overview">
        <h2>Strategy Overview</h2>
        <p>This strategy uses multi-timeframe analysis to identify high-probability reversal trades:</p>

        <div class="strategy-step">
            <strong>Step 1:</strong> Detect <strong>Liquidity Sweep</strong> on 1H timeframe (trigger event)
        </div>

        <div class="strategy-step">
            <strong>Step 2:</strong> Identify <strong>BOS or IFVG</strong> on 5M in opposite direction (Event B)
        </div>

        <div class="strategy-step">
            <strong>Step 3:</strong> Validate with <strong>Equilibrium Premium/Discount Zone</strong> on 5M
        </div>

        <div class="strategy-step">
            <strong>Step 4:</strong> Confirm with <strong>BOS or IFVG</strong> on 1M in entry direction
        </div>

        <div class="strategy-step">
            <strong>Step 5:</strong> <strong>Enter trade</strong> with exit target (Order Block at origin) and 2:1 R/R
        </div>
    </div>

    <div class="stats">
        <div class="stat-item">
            <div class="stat-value">{len(signals)}</div>
            <div class="stat-label">Trade Setups Found</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{sum(1 for s in signals if s.entry_direction == 'long')}</div>
            <div class="stat-label">Long Entries</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{sum(1 for s in signals if s.entry_direction == 'short')}</div>
            <div class="stat-label">Short Entries</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{sum(1 for s in signals if s.condition_event_b == 'BOS')}</div>
            <div class="stat-label">BOS Events</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{sum(1 for s in signals if s.condition_validation == 'Equilibrium')}</div>
            <div class="stat-label">Equilibrium Validations</div>
        </div>
    </div>

    <h2>Trade Examples</h2>
    <p>Click on any trade card to view the complete step-by-step visual walkthrough:</p>

    <div class="trades-grid">
"""

    # Add trade cards
    for i, signal in enumerate(signals, 1):
        direction_class = "long" if signal.entry_direction == "long" else "short"

        html_content += f"""
        <a href="trade_{i}/index.html" class="trade-card">
            <div class="trade-number">Trade #{i}</div>
            <div class="trade-direction {direction_class}">{signal.entry_direction.upper()} @ ${signal.price_entry:.2f}</div>
            <div class="trade-details">
                <div class="trade-details-item"><strong>Date:</strong> {signal.timestamp_entry.strftime('%Y-%m-%d')}</div>
                <div class="trade-details-item"><strong>Time:</strong> {signal.timestamp_entry.strftime('%H:%M')}</div>
                <div class="trade-details-item"><span class="checkmark">✓</span> 1H: {signal.condition_liquidity_sweep}</div>
                <div class="trade-details-item"><span class="checkmark">✓</span> 5M: {signal.condition_event_b}</div>
                <div class="trade-details-item"><span class="checkmark">✓</span> 5M: {signal.condition_validation}</div>
                <div class="trade-details-item"><span class="checkmark">✓</span> 1M: {signal.condition_confirmation}</div>
            </div>
        </a>
"""

    html_content += f"""
    </div>

    <div style="text-align: center; margin-top: 50px; color: #666;">
        <p>Generated by {symbol} Multi-Timeframe Strategy Analyzer</p>
        <p>Using Custom Inflexion Points + Trend-Aware BOS + SMC Indicators</p>
    </div>

</body>
</html>
    """

    # Save master index
    output_path = Path(output_dir) / "master_index.html"
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"\n📄 Master index saved: {output_path}")


def main():
    """Main execution function."""

    # Load configuration from JSON file
    config = load_config()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze stock trading strategy with multi-timeframe setups',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--no-backtest',
        action='store_true',
        help='Skip backtest simulation (backtest runs by default)'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000.0,
        help='Initial capital for backtesting in USD (default: $10,000)'
    )
    parser.add_argument(
        '--export-journal',
        nargs='?',
        const='auto',
        default='auto',
        help='Export trade journal to CSV (default: results/trades/{symbol}_trades.csv)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='TSLA',
        help='Stock symbol to analyze (default: TSLA)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate unified HTML visualization of all sweeps (results/{symbol}_sweeps.html)'
    )
    parser.add_argument(
        '--high-tf',
        type=str,
        default='1h',
        help='High timeframe (default: 1h). Options: 1min, 5min, 15min, 30min, 1h, 4h, 1day'
    )
    parser.add_argument(
        '--mid-tf',
        type=str,
        default='5min',
        help='Mid timeframe (default: 5min). Options: 1min, 5min, 15min, 30min, 1h, 4h, 1day'
    )
    parser.add_argument(
        '--low-tf',
        type=str,
        default='1min',
        help='Low timeframe (default: 1min). Options: 1min, 5min, 15min, 30min, 1h, 4h, 1day'
    )
    parser.add_argument(
        '--hold-overnight',
        action='store_true',
        help='Allow trades to hold overnight (default: intraday only, exit at market close 21:59)'
    )
    parser.add_argument(
        '--bos-exit',
        action='store_true',
        default=None,
        help='Enable BOS-based early exit: exit when opposing BOS occurs while in profit'
    )
    parser.add_argument(
        '--no-bos-exit',
        action='store_true',
        help='Disable BOS-based early exit (overrides config file)'
    )

    # Apply JSON config as defaults (CLI args will override these)
    if config:
        timeframes = config.get('timeframes', {})
        backtest_cfg = config.get('backtest', {})
        output_cfg = config.get('output', {})

        parser.set_defaults(
            symbol=config.get('symbol', 'TSLA'),
            high_tf=timeframes.get('high', '1h'),
            mid_tf=timeframes.get('mid', '5min'),
            low_tf=timeframes.get('low', '1min'),
            initial_capital=backtest_cfg.get('initial_capital', 10000.0),
            no_backtest=not backtest_cfg.get('enabled', True),
            hold_overnight=backtest_cfg.get('hold_overnight', False),
            bos_exit_enabled=backtest_cfg.get('bos_exit_enabled', False),
            bos_exit_threshold_percent=backtest_cfg.get('bos_exit_threshold_percent', 50.0),
            visualize=output_cfg.get('visualize', False),
            export_journal=output_cfg.get('export_journal', 'auto'),
        )

    args = parser.parse_args()

    symbol = args.symbol.upper()
    high_tf = args.high_tf
    mid_tf = args.mid_tf
    low_tf = args.low_tf

    # Resolve bos_exit setting: CLI flags take precedence over config
    bos_exit_enabled = getattr(args, 'bos_exit_enabled', False)
    bos_exit_threshold_percent = getattr(args, 'bos_exit_threshold_percent', 50.0)
    if args.bos_exit:
        bos_exit_enabled = True
    elif args.no_bos_exit:
        bos_exit_enabled = False

    # Create timeframe config for display
    timeframe_config = {
        'high': high_tf.upper(),
        'mid': mid_tf.upper(),
        'low': low_tf.upper()
    }

    print("\n" + "="*80)
    print(f"{symbol} MULTI-TIMEFRAME STRATEGY ANALYSIS")
    print("="*80)

    # Display configuration summary
    print("\n📋 Configuration:")
    print(f"  Symbol:           {symbol}")
    print(f"  Timeframes:       {high_tf.upper()} (high) / {mid_tf.upper()} (mid) / {low_tf.upper()} (low)")
    print(f"  Initial Capital:  ${args.initial_capital:,.2f}")
    print(f"  Backtest:         {'Disabled' if args.no_backtest else 'Enabled'}")
    print(f"  Hold Overnight:   {'Yes' if args.hold_overnight else 'No (intraday only)'}")
    if bos_exit_enabled:
        print(f"  BOS Exit:         Enabled (exit on opposing BOS when profit >= {bos_exit_threshold_percent:.0f}% of TP)")
    else:
        print(f"  BOS Exit:         Disabled")
    print(f"  Visualization:    {'Enabled' if args.visualize else 'Disabled'}")
    print(f"  Export Journal:   {args.export_journal if args.export_journal else 'Disabled'}")
    print("")

    # Step 1: Load data (force refresh to get latest)
    print(f"📂 Downloading fresh {symbol} data for all timeframes...")
    loader = DataLoader(symbol=symbol)

    df_high = loader.get_data(high_tf, force_refresh=True)
    df_mid = loader.get_data(mid_tf, force_refresh=True)
    df_low = loader.get_data(low_tf, force_refresh=True)

    print(f"  ✓ {high_tf.upper()}:  {len(df_high)} candles")
    print(f"  ✓ {mid_tf.upper()}:  {len(df_mid)} candles")
    print(f"  ✓ {low_tf.upper()}:  {len(df_low)} candles")

    # Step 2: Initialize strategy
    print("\n🔧 Initializing strategy engine...")
    strategy = MultiTimeframeStrategy(df_high, df_mid, df_low, timeframe_config)

    # Step 3: Scan for signals
    print("\n🔍 Scanning for trade setups...")
    signals = strategy.scan_for_signals(max_signals=5)

    # Handle complete signals
    if len(signals) > 0:
        print(f"\n✅ Found {len(signals)} complete trade setups!")
        print("   (Skipping old visualization - use TradingView analyzer instead)")

    # Run backtest if requested
    backtester = None
    if not args.no_backtest and len(signals) > 0:
        print("\n" + "="*80)
        print("RUNNING BACKTEST SIMULATION")
        print("="*80)

        backtester = Backtester.from_strategy(
            strategy,
            initial_capital=args.initial_capital,
            symbol=symbol,
            intraday_only=not args.hold_overnight,
            bos_exit_enabled=bos_exit_enabled,
            bos_exit_threshold_percent=bos_exit_threshold_percent
        )
        _results = backtester.run(signals)
        backtester.print_summary()

        if args.export_journal:
            # Generate default path if 'auto'
            if args.export_journal == 'auto':
                import os
                os.makedirs("results/trades", exist_ok=True)
                journal_path = f"results/trades/{symbol.lower()}_trades.csv"
            else:
                journal_path = args.export_journal
            backtester.export_journal(journal_path)

    elif not args.no_backtest and len(signals) == 0:
        print("\n⚠️  Cannot run backtest - no signals found.")

    # Generate unified visualization if requested
    if args.visualize:
        print("\n" + "="*80)
        print("GENERATING UNIFIED SWEEP VISUALIZATION")
        print("="*80)

        total_sweeps = len(signals) + len(strategy.partial_setups)
        if total_sweeps > 0:
            # Pass trade results and skipped trades from backtester (if available) for P&L display
            trade_results = backtester.results if backtester else []
            skipped_trades = backtester.skipped_trades if backtester else []
            visualizer = StrategySweepVisualizer(
                strategy,
                symbol=symbol,
                trade_results=trade_results,
                skipped_trades=skipped_trades
            )
            output_path = visualizer.generate_html()
            print(f"\n✅ Generated visualization: {output_path}")
            print(f"   Total sweeps: {total_sweeps} ({len(signals)} successful, {len(strategy.partial_setups)} failed)")
            if trade_results:
                print(f"   P&L data included for {len(trade_results)} trades")
            if skipped_trades:
                print(f"   Skipped trade info included for {len(skipped_trades)} signals")
            print("   Open the HTML file in your browser to explore all sweeps!")
        else:
            print("\n⚠️  No sweeps found to visualize.")

    # If no sweeps found at all
    if len(signals) == 0 and len(strategy.partial_setups) == 0:
        print("\n⚠️  No sweeps found in the data.")
        print("   Try with different date ranges or adjust strategy parameters.")
        return

    # Suggest visualization if not already done
    if not args.visualize and (len(signals) > 0 or len(strategy.partial_setups) > 0):
        total = len(signals) + len(strategy.partial_setups)
        print(f"\n💡 Found {total} sweeps. Run with --visualize to generate interactive HTML:"
              f"\n   python3 scripts/analyze_strategy.py --symbol {symbol} --visualize")

    # Summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

    if len(signals) > 0:
        print(f"\n✓ Found {len(signals)} complete trade signals:")
        for i, sig in enumerate(signals, 1):
            print(f"\n  Signal #{i}: {sig.entry_direction.upper()}")
            print(f"    Entry: ${sig.price_entry:.2f}")
            print(f"    Take Profit: ${sig.take_profit_price:.2f}" if sig.take_profit_price else "    Take Profit: N/A")
            print(f"    Stop Loss: ${sig.stop_loss_price:.2f}" if sig.stop_loss_price else "    Stop Loss: N/A")

    if len(strategy.partial_setups) > 0:
        print(f"\n✓ Found {len(strategy.partial_setups)} partial setups (failed to complete)")

    if args.visualize:
        print(f"\n💡 Open results/{symbol.lower()}_sweeps.html in your browser!")
        print("   Use arrow keys ← → to navigate through frames")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
