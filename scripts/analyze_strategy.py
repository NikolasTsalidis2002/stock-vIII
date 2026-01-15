"""
Strategy Analysis Script

Scans historical stock data for complete trade setups and generates
visual walkthroughs showing the step-by-step strategy execution.

Usage:
    python3 scripts/analyze_strategy.py                    # Default: TSLA
    python3 scripts/analyze_strategy.py --symbol AAPL      # Use different symbol
    python3 scripts/analyze_strategy.py --symbol META      # Use different symbol

Output:
    - results/strategy_examples/trade_1/index.html
    - results/strategy_examples/trade_2/index.html
    - results/strategy_examples/trade_3/index.html
    - results/strategy_examples/master_index.html (overview)
"""

import sys
import os
from pathlib import Path
import argparse

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import DataLoader
from strategy import MultiTimeframeStrategy
from tradingview_strategy_analyzer import TradingViewStrategyAnalyzer
from backtesting import Backtester


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

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze stock trading strategy with multi-timeframe setups',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--animate-partials',
        action='store_true',
        help='Generate TradingView frame-by-frame walkthroughs for partial setups (incomplete but close)'
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
        help='Export trade journal to CSV (default: results/{symbol}_trades.csv)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='TSLA',
        help='Stock symbol to analyze (default: TSLA)'
    )

    args = parser.parse_args()

    symbol = args.symbol.upper()

    print("\n" + "="*80)
    print(f"{symbol} MULTI-TIMEFRAME STRATEGY ANALYSIS")
    print("="*80 + "\n")

    # Step 1: Load data (force refresh to get latest)
    print(f"📂 Downloading fresh {symbol} data for all timeframes...")
    loader = DataLoader(symbol=symbol)

    df_1h = loader.get_data('1h', force_refresh=False)
    df_5m = loader.get_data('5min', force_refresh=False)
    df_1m = loader.get_data('1min', force_refresh=False)

    print(f"  ✓ 1H:  {len(df_1h)} candles")
    print(f"  ✓ 5M:  {len(df_5m)} candles")
    print(f"  ✓ 1M:  {len(df_1m)} candles")

    # Step 2: Initialize strategy
    print("\n🔧 Initializing strategy engine...")
    strategy = MultiTimeframeStrategy(df_1h, df_5m, df_1m)

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
            symbol=symbol
        )
        results = backtester.run(signals)
        backtester.print_summary()

        if args.export_journal:
            # Generate default path if 'auto'
            if args.export_journal == 'auto':
                journal_path = f"results/{symbol.lower()}_trades.csv"
            else:
                journal_path = args.export_journal
            backtester.export_journal(journal_path)

    elif not args.no_backtest and len(signals) == 0:
        print("\n⚠️  Cannot run backtest - no signals found.")

    # Handle partial setups if requested
    if args.animate_partials and len(strategy.partial_setups) > 0:
        print("\n🎬 Generating TradingView frame-by-frame walkthroughs for partial setups...")
        tv_analyzer = TradingViewStrategyAnalyzer(strategy)

        for i, partial in enumerate(strategy.partial_setups, 1):
            tv_analyzer.analyze_partial_setup(partial, setup_num=i)

        print("\n✅ TradingView walkthroughs complete!")
        print(f"   Generated {len(strategy.partial_setups)} interactive walkthroughs")

    elif args.animate_partials and len(strategy.partial_setups) == 0:
        print("\n⚠️  No partial setups found to animate.")

    # If no complete signals and no partials animated
    if len(signals) == 0 and not args.animate_partials:
        print("\n⚠️  No complete trade setups found in the data.")
        if len(strategy.partial_setups) > 0:
            print(f"   However, found {len(strategy.partial_setups)} partial setups.")
            print("   Run with --animate-partials to visualize them:")
            print("   python3 scripts/analyze_strategy.py --animate-partials")
        else:
            print("   Try with different date ranges or adjust strategy parameters.")
        return

    # Summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

    if len(signals) > 0:
        print(f"\n✓ Found {len(signals)} signals with exit targets:")
        for i, sig in enumerate(signals, 1):
            print(f"\n  Signal #{i}: {sig.entry_direction.upper()}")
            print(f"    Entry: ${sig.price_entry:.2f}")
            print(f"    Take Profit: ${sig.take_profit_price:.2f}" if sig.take_profit_price else "    Take Profit: N/A")
            print(f"    Stop Loss: ${sig.stop_loss_price:.2f}" if sig.stop_loss_price else "    Stop Loss: N/A")
            print(f"    Equilibrium: ${sig.equilibrium_level:.2f}" if sig.equilibrium_level else "    Equilibrium: N/A")

    if args.animate_partials and len(strategy.partial_setups) > 0:
        print(f"\nGenerated {len(strategy.partial_setups)} TradingView walkthroughs:")
        for i in range(1, len(strategy.partial_setups) + 1):
            tv_path = tv_analyzer.output_dir / f"partial_setup_{i}" / "walkthrough_tv.html"
            print(f"  {i}. {tv_path}")

    if len(signals) > 0:
        print("\n💡 Open the master index in your browser to explore all trades!")
    elif args.animate_partials and len(strategy.partial_setups) > 0:
        print("\n💡 Open the TradingView HTML files to see the professional 3-timeframe walkthrough!")
        print("   Use arrow keys ← → or buttons to navigate between frames")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
