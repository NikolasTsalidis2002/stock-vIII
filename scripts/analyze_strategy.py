"""
Strategy Analysis Script

Scans historical stock data for complete trade setups and generates
visual walkthroughs showing the step-by-step strategy execution.

Usage:
    python3 scripts/analyze_strategy.py                         # Default: TSLA with 1h/5min/1min
    python3 scripts/analyze_strategy.py --symbol META           # Use different symbol
    python3 scripts/analyze_strategy.py --symbol META --visualize  # Generate unified HTML
    python3 scripts/analyze_strategy.py --high-tf 4h --mid-tf 15min --low-tf 5min  # Custom timeframes

Batch Mode:
    python3 scripts/analyze_strategy.py --all-symbols           # Run for all symbols in config
    python3 scripts/analyze_strategy.py --all-symbols --summary-output results/my_batch.csv  # Custom summary path

Output:
    --visualize flag generates: results/{symbol}_sweeps.html
    (Single interactive file with all sweeps, tabs for timeframes, arrow key navigation)

    --all-symbols generates:
    - results/trades/{symbol}_trades.csv (individual trade journals)
    - results/summary/batch_summary.csv (consolidated summary with metrics)
"""

import sys
import os
import json
import csv
from pathlib import Path
import argparse
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import DataLoader
from strategy import MultiTimeframeStrategy, ThreeOBStrategy
from strategy_visualizer import StrategySweepVisualizer
from backtesting import Backtester
from backtesting.models import SkippedTrade, SkipReason
from visualization.three_ob_walkthrough import ThreeOBWalkthrough
from visualization.three_ob_signal_viewer import ThreeOBSignalViewer


def get_all_symbols_from_config(config):
    """Extract all symbols from all categories in config.

    Args:
        config: Configuration dictionary with asset_options containing symbol lists.

    Returns:
        list: All symbols from all categories.
    """
    symbols = []
    asset_options = config.get("asset_options", {})
    for category, symbol_list in asset_options.items():
        if category.startswith("_"):
            continue  # Skip description fields
        if isinstance(symbol_list, list):
            symbols.extend(symbol_list)
    return symbols


def has_required_cache(symbol: str, timeframes: dict) -> bool:
    """Check if symbol has all required timeframe data cached.

    Args:
        symbol: The trading symbol (e.g., 'TSLA', 'BTC/USD').
        timeframes: Dict with 'high', 'mid', 'low' timeframe keys.

    Returns:
        bool: True if all required cache files exist.
    """
    safe_symbol = symbol.lower().replace('/', '_')
    data_dir = Path(__file__).parent.parent / 'data' / safe_symbol

    required_tfs = [timeframes['high'], timeframes['mid'], timeframes['low']]
    for tf in required_tfs:
        cache_file = data_dir / f"{safe_symbol}_{tf}.csv"
        if not cache_file.exists():
            return False
    return True


def generate_summary_csv(results, output_path):
    """Write consolidated batch results to CSV.

    Args:
        results: List of result dictionaries from batch analysis.
        output_path: Path to write the summary CSV.
    """
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'symbol',
        'high_tf_start', 'high_tf_end',
        'mid_tf_start', 'mid_tf_end',
        'low_tf_start', 'low_tf_end',
        'status', 'num_trades', 'long_trades', 'short_trades',
        'total_pnl_dollars', 'total_pnl_percent', 'win_rate',
        'avg_pnl_per_trade', 'best_trade_pnl', 'worst_trade_pnl', 'error'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\n📊 Batch summary saved to: {output_path}")


def run_single_symbol(symbol, config, args):
    """Run analysis for a single symbol and return result dict.

    Args:
        symbol: The symbol to analyze.
        config: Configuration dictionary.
        args: Parsed command-line arguments.

    Returns:
        dict: Result dictionary with metrics or error info.
    """
    # Get timeframe settings
    timeframes = config.get('timeframes', {})
    high_tf = timeframes.get('high', '1h')
    mid_tf = timeframes.get('mid', '5min')
    low_tf = timeframes.get('low', '1min')

    timeframe_config = {
        'high': high_tf.upper(),
        'mid': mid_tf.upper(),
        'low': low_tf.upper()
    }

    # Get validation settings
    validation_cfg = config.get('validation', {})
    use_fvg_validation = validation_cfg.get('use_fvg_validation', True)
    use_equilibrium_validation = validation_cfg.get('use_equilibrium_validation', True)
    require_fvg_in_equilibrium = validation_cfg.get('require_fvg_in_equilibrium', False)
    abandon_on_new_sweep = validation_cfg.get('abandon_on_new_sweep', True)

    # Get liquidity settings
    liquidity_cfg = config.get('liquidity', {})
    sweep_proximity_threshold = liquidity_cfg.get('sweep_proximity_threshold_percent', 0.0) / 100.0

    # Get high TF lookback setting (for extended inflection point detection)
    high_tf_lookback = liquidity_cfg.get('high_tf_lookback_candles', 300)

    # Get FVG trigger setting
    use_fvg_trigger = liquidity_cfg.get('use_fvg_trigger', False)

    # Get GMM zone settings
    gmm_config = config.get('gmm_zones', {})

    # Get backtest settings
    backtest_cfg = config.get('backtest', {})
    initial_capital = backtest_cfg.get('initial_capital', 10000.0)
    hold_overnight = backtest_cfg.get('hold_overnight', False)
    trailing_sl_enabled = backtest_cfg.get('trailing_sl_enabled', False)
    trailing_sl_activation_pct = backtest_cfg.get('trailing_sl_activation_pct', 50.0)
    trailing_sl_swing_length = backtest_cfg.get('trailing_sl_swing_length', 5)
    min_profit_percent = backtest_cfg.get('min_profit_percent', 0.0)
    min_rr_ratio = backtest_cfg.get('min_rr_ratio', 0.0)
    trend_filter_enabled = backtest_cfg.get('trend_filter_enabled', False)
    trend_filter_lookback = backtest_cfg.get('trend_filter_lookback', 500)
    bos_1h_trend_filter_enabled = backtest_cfg.get('bos_1h_trend_filter_enabled', False)

    print(f"\n{'='*60}")
    print(f"Analyzing {symbol}...")
    print(f"{'='*60}")

    # Load data
    loader = DataLoader(symbol=symbol)
    df_high = loader.get_data(high_tf, force_refresh=False)
    df_mid = loader.get_data(mid_tf, force_refresh=False)
    df_low = loader.get_data(low_tf, force_refresh=False)

    print(f"  {high_tf.upper()}: {len(df_high)} candles | {mid_tf.upper()}: {len(df_mid)} candles | {low_tf.upper()}: {len(df_low)} candles")

    # Extract date range from all timeframes
    high_tf_start = df_high['time'].min().strftime('%Y-%m-%d')
    high_tf_end = df_high['time'].max().strftime('%Y-%m-%d')
    mid_tf_start = df_mid['time'].min().strftime('%Y-%m-%d')
    mid_tf_end = df_mid['time'].max().strftime('%Y-%m-%d')
    low_tf_start = df_low['time'].min().strftime('%Y-%m-%d')
    low_tf_end = df_low['time'].max().strftime('%Y-%m-%d')

    # Initialize strategy
    strategy = MultiTimeframeStrategy(
        df_high, df_mid, df_low, timeframe_config,
        use_fvg_validation=use_fvg_validation,
        use_equilibrium_validation=use_equilibrium_validation,
        require_fvg_in_equilibrium=require_fvg_in_equilibrium,
        sweep_proximity_threshold=sweep_proximity_threshold,
        abandon_on_new_sweep=abandon_on_new_sweep,
        gmm_config=gmm_config,
        high_tf_lookback=high_tf_lookback,
        use_fvg_trigger=use_fvg_trigger
    )

    # Scan for signals
    signals = strategy.scan_for_signals(max_signals=999_999)

    if len(signals) == 0:
        print(f"  No signals found for {symbol}")
        return {
            'symbol': symbol,
            'high_tf_start': high_tf_start,
            'high_tf_end': high_tf_end,
            'mid_tf_start': mid_tf_start,
            'mid_tf_end': mid_tf_end,
            'low_tf_start': low_tf_start,
            'low_tf_end': low_tf_end,
            'status': 'success',
            'num_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'total_pnl_dollars': 0.0,
            'total_pnl_percent': 0.0,
            'win_rate': 0.0,
            'avg_pnl_per_trade': 0.0,
            'best_trade_pnl': 0.0,
            'worst_trade_pnl': 0.0,
            'error': ''
        }

    # Run ARMA trend analysis if enabled
    import numpy as np
    trend_signal = None
    if trend_filter_enabled:
        from src.arma_trend_analysis import ARMA
        mid_prices = list(df_mid['close'])
        lookback_prices = mid_prices[-trend_filter_lookback:] if len(mid_prices) >= trend_filter_lookback else mid_prices
        arma_result = ARMA.fit_arma(lookback_prices, auto_select=True, max_order=3)
        trend_signal = arma_result['trend_signal']

    # Run backtest
    backtester = Backtester.from_strategy(
        strategy,
        initial_capital=initial_capital,
        symbol=symbol,
        intraday_only=not hold_overnight,
        min_profit_percent=min_profit_percent,
        min_rr_ratio=min_rr_ratio,
        trend_filter=trend_signal if trend_filter_enabled else None,
        bos_1h_trend_filter_enabled=bos_1h_trend_filter_enabled,
        trailing_sl_enabled=trailing_sl_enabled,
        trailing_sl_activation_pct=trailing_sl_activation_pct,
        trailing_sl_swing_length=trailing_sl_swing_length
    )
    _results = backtester.run(signals)

    # Export trade journal
    os.makedirs("results/trades", exist_ok=True)
    journal_path = f"results/trades/{symbol.lower().replace('/', '_')}_trades.csv"
    backtester.export_journal(journal_path)

    # Extract metrics
    metrics = backtester.metrics
    trade_results = backtester.results

    # Calculate additional metrics
    num_trades = len(trade_results)
    long_trades = sum(1 for t in trade_results if t.entry_direction == 'long')
    short_trades = sum(1 for t in trade_results if t.entry_direction == 'short')

    pnl_values = [t.pnl_dollars for t in trade_results]
    best_trade = max(pnl_values) if pnl_values else 0.0
    worst_trade = min(pnl_values) if pnl_values else 0.0

    total_pnl = metrics.total_pnl_dollars if metrics else 0.0
    total_pnl_pct = metrics.total_return_percent if metrics else 0.0
    win_rate = metrics.win_rate if metrics else 0.0
    avg_pnl = total_pnl / num_trades if num_trades > 0 else 0.0

    print(f"  {symbol}: {num_trades} trades, P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%), Win rate: {win_rate:.1f}%")

    return {
        'symbol': symbol,
        'high_tf_start': high_tf_start,
        'high_tf_end': high_tf_end,
        'mid_tf_start': mid_tf_start,
        'mid_tf_end': mid_tf_end,
        'low_tf_start': low_tf_start,
        'low_tf_end': low_tf_end,
        'status': 'success',
        'num_trades': num_trades,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'total_pnl_dollars': round(total_pnl, 2),
        'total_pnl_percent': round(total_pnl_pct, 2),
        'win_rate': round(win_rate, 1),
        'avg_pnl_per_trade': round(avg_pnl, 2),
        'best_trade_pnl': round(best_trade, 2),
        'worst_trade_pnl': round(worst_trade, 2),
        'error': ''
    }


def run_batch_analysis(config, args):
    """Run analysis for all symbols in config with error handling.

    Only processes symbols with complete cached data (no API calls).

    Args:
        config: Configuration dictionary.
        args: Parsed command-line arguments.

    Returns:
        list: List of result dictionaries for each symbol.
    """
    all_symbols = get_all_symbols_from_config(config)
    results = []

    # Get timeframe settings for cache check
    timeframes_cfg = config.get('timeframes', {})
    timeframes = {
        'high': timeframes_cfg.get('high', '1h'),
        'mid': timeframes_cfg.get('mid', '15min'),
        'low': timeframes_cfg.get('low', '5min')
    }

    # Filter to only symbols with complete cache
    symbols_with_cache = [s for s in all_symbols if has_required_cache(s, timeframes)]
    symbols_without_cache = [s for s in all_symbols if not has_required_cache(s, timeframes)]

    print("\n" + "="*80)
    print("BATCH ANALYSIS - Cache-Only Mode (no API calls)")
    print("="*80)
    print(f"Total configured symbols: {len(all_symbols)}")
    print(f"Symbols with complete cache: {len(symbols_with_cache)}")
    print(f"Symbols skipped (no cache): {len(symbols_without_cache)}")

    if symbols_without_cache:
        print(f"\nSkipping {len(symbols_without_cache)} symbols without cache:")
        for s in symbols_without_cache:
            print(f"  - {s}")
            results.append({
                'symbol': s,
                'high_tf_start': '',
                'high_tf_end': '',
                'mid_tf_start': '',
                'mid_tf_end': '',
                'low_tf_start': '',
                'low_tf_end': '',
                'status': 'skipped',
                'num_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'total_pnl_dollars': 0.0,
                'total_pnl_percent': 0.0,
                'win_rate': 0.0,
                'avg_pnl_per_trade': 0.0,
                'best_trade_pnl': 0.0,
                'worst_trade_pnl': 0.0,
                'error': 'No cached data'
            })

    for i, symbol in enumerate(symbols_with_cache, 1):
        print(f"\n[{i}/{len(symbols_with_cache)}] Processing {symbol}...")
        try:
            result = run_single_symbol(symbol, config, args)
            results.append(result)
        except Exception as e:
            print(f"  Failed to process {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'high_tf_start': '',
                'high_tf_end': '',
                'mid_tf_start': '',
                'mid_tf_end': '',
                'low_tf_start': '',
                'low_tf_end': '',
                'status': 'failed',
                'num_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'total_pnl_dollars': 0.0,
                'total_pnl_percent': 0.0,
                'win_rate': 0.0,
                'avg_pnl_per_trade': 0.0,
                'best_trade_pnl': 0.0,
                'worst_trade_pnl': 0.0,
                'error': str(e)
            })

    return results


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
        '--strategy',
        type=str,
        choices=['multi-tf', '3-ob'],
        # default='multi-tf',
        default='3-ob',
        help='Strategy to run: multi-tf (default multi-timeframe) or 3-ob (3-OB state machine)'
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
        '--all-symbols',
        action='store_true',
        help='Run analysis for all symbols configured in asset_options (batch mode)'
    )
    parser.add_argument(
        '--summary-output',
        type=str,
        default='results/summary/batch_summary.csv',
        help='Path for consolidated summary CSV (default: results/summary/batch_summary.csv)'
    )
    parser.add_argument(
        '--walkthrough-3ob',
        action='store_true',
        help='Generate candle-by-candle HTML walkthrough for the 3-OB state machine'
    )

    # Apply JSON config as defaults (CLI args will override these)
    if config:
        timeframes = config.get('timeframes', {})
        backtest_cfg = config.get('backtest', {})
        output_cfg = config.get('output', {})
        validation_cfg = config.get('validation', {})
        liquidity_cfg = config.get('liquidity', {})

        parser.set_defaults(
            symbol=config.get('symbol', 'TSLA'),
            high_tf=timeframes.get('high', '1h'),
            mid_tf=timeframes.get('mid', '5min'),
            low_tf=timeframes.get('low', '1min'),
            initial_capital=backtest_cfg.get('initial_capital', 10000.0),
            no_backtest=not backtest_cfg.get('enabled', True),
            hold_overnight=backtest_cfg.get('hold_overnight', False),
            trailing_sl_enabled=backtest_cfg.get('trailing_sl_enabled', False),
            trailing_sl_activation_pct=backtest_cfg.get('trailing_sl_activation_pct', 50.0),
            trailing_sl_swing_length=backtest_cfg.get('trailing_sl_swing_length', 5),
            min_profit_percent=backtest_cfg.get('min_profit_percent', 0.0),
            min_rr_ratio=backtest_cfg.get('min_rr_ratio', 0.0),
            trend_filter_enabled=backtest_cfg.get('trend_filter_enabled', False),
            trend_filter_lookback=backtest_cfg.get('trend_filter_lookback', 500),
            bos_1h_trend_filter_enabled=backtest_cfg.get('bos_1h_trend_filter_enabled', False),
            visualize=output_cfg.get('visualize', False),
            export_journal=output_cfg.get('export_journal', 'auto'),
            use_fvg_validation=validation_cfg.get('use_fvg_validation', True),
            use_equilibrium_validation=validation_cfg.get('use_equilibrium_validation', True),
            require_fvg_in_equilibrium=validation_cfg.get('require_fvg_in_equilibrium', False),
            abandon_on_new_sweep=validation_cfg.get('abandon_on_new_sweep', True),
            sweep_proximity_threshold_percent=liquidity_cfg.get('sweep_proximity_threshold_percent', 0.0),
        )

    args = parser.parse_args()

    # Handle batch mode: run for all symbols if --all-symbols is passed
    if args.all_symbols:
        results = run_batch_analysis(config, args)
        generate_summary_csv(results, args.summary_output)

        # Print batch summary
        print("\n" + "="*80)
        print("BATCH ANALYSIS COMPLETE")
        print("="*80)

        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        total_trades = sum(r['num_trades'] for r in results)
        total_pnl = sum(r['total_pnl_dollars'] for r in results)

        print(f"\nSymbols: {len(results)} total ({successful} successful, {failed} failed, {skipped} skipped)")
        print(f"Total trades across all symbols: {total_trades}")
        print(f"Combined P&L: ${total_pnl:,.2f}")
        print(f"\nSummary saved to: {args.summary_output}")
        print(f"Individual trade journals saved to: results/trades/")
        print("="*80 + "\n")
        return

    symbol = args.symbol.upper()
    high_tf = args.high_tf
    mid_tf = args.mid_tf
    low_tf = args.low_tf

    # Resolve trailing SL and other settings from config/CLI
    trailing_sl_enabled = getattr(args, 'trailing_sl_enabled', False)
    trailing_sl_activation_pct = getattr(args, 'trailing_sl_activation_pct', 50.0)
    trailing_sl_swing_length = getattr(args, 'trailing_sl_swing_length', 5)
    min_profit_percent = getattr(args, 'min_profit_percent', 0.0)
    min_rr_ratio = getattr(args, 'min_rr_ratio', 0.0)
    trend_filter_enabled = getattr(args, 'trend_filter_enabled', False)
    trend_filter_lookback = getattr(args, 'trend_filter_lookback', 500)
    bos_1h_trend_filter_enabled = getattr(args, 'bos_1h_trend_filter_enabled', False)

    # Get validation settings from config
    use_fvg_validation = getattr(args, 'use_fvg_validation', True)
    use_equilibrium_validation = getattr(args, 'use_equilibrium_validation', True)
    require_fvg_in_equilibrium = getattr(args, 'require_fvg_in_equilibrium', False)
    abandon_on_new_sweep = getattr(args, 'abandon_on_new_sweep', True)

    # Get liquidity sweep settings from config
    # Convert from percentage to decimal (e.g., 0.5% -> 0.005)
    sweep_proximity_threshold_percent = getattr(args, 'sweep_proximity_threshold_percent', 0.0)
    sweep_proximity_threshold = sweep_proximity_threshold_percent / 100.0

    # Get high TF lookback setting (for extended inflection point detection)
    liquidity_cfg = config.get('liquidity', {})
    high_tf_lookback = liquidity_cfg.get('high_tf_lookback_candles', 300)

    # Get FVG trigger setting
    use_fvg_trigger = liquidity_cfg.get('use_fvg_trigger', False)

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
    if trailing_sl_enabled:
        print(f"  Trailing SL:      Enabled (activation={trailing_sl_activation_pct}% of entry→TP, swing_length={trailing_sl_swing_length})")
    else:
        print(f"  Trailing SL:      Disabled")
    if min_profit_percent > 0:
        print(f"  Min Profit:       {min_profit_percent}% (skip trades below)")
    if min_rr_ratio > 0:
        print(f"  Min R:R Ratio:    {min_rr_ratio:.1f} (skip trades below)")
    if trend_filter_enabled:
        print(f"  Trend Filter:     Enabled (ARMA, lookback={trend_filter_lookback})")
    else:
        print(f"  Trend Filter:     Disabled")
    if bos_1h_trend_filter_enabled:
        print(f"  1H BOS Filter:    Enabled (trades must follow 1H BOS trend)")
    else:
        print(f"  1H BOS Filter:    Disabled")
    # Display validation mode
    if require_fvg_in_equilibrium:
        print(f"  Validation Mode:  FVG-in-Equilibrium (strictest - FVG must overlap equilibrium zone)")
    elif use_fvg_validation and use_equilibrium_validation:
        print(f"  Validation Mode:  FVG Priority with Equilibrium Fallback")
    elif use_fvg_validation:
        print(f"  Validation Mode:  FVG Only")
    elif use_equilibrium_validation:
        print(f"  Validation Mode:  Equilibrium Only")
    else:
        print(f"  Validation Mode:  None (warning: no validation enabled)")
    if sweep_proximity_threshold_percent > 0:
        print(f"  Sweep Proximity:  {sweep_proximity_threshold_percent}% (price within {sweep_proximity_threshold_percent}% of level counts as swept)")
    else:
        print(f"  Sweep Proximity:  Exact touch required (0%)")
    # Display FVG trigger setting
    if use_fvg_trigger:
        print(f"  FVG Trigger:      Enabled (high TF FVG respect triggers Stage 1)")
    else:
        print(f"  FVG Trigger:      Disabled")
    # Display GMM zone settings
    gmm_enabled = config.get('gmm_zones', {}).get('enabled', False)
    if gmm_enabled:
        gmm_cfg = config.get('gmm_zones', {})
        print(f"  GMM Zones:        Enabled")
        print(f"    - Lookback:     {gmm_cfg.get('lookback_candles', 300)} candles")
        print(f"    - Premium zone: >= {gmm_cfg.get('premium_zone', [0.786, 1.0])[0]} fib (SHORT bias)")
        print(f"    - Discount zone: <= {gmm_cfg.get('discount_zone', [0.0, 0.236])[1]} fib (LONG bias)")
        print(f"    - Middle zone:  {'Allow trades' if gmm_cfg.get('allow_middle_zone_trades', False) else 'Skip trades'}")
        print(f"    - TP method:    {gmm_cfg.get('take_profit_method', 'fib')}")
    else:
        print(f"  GMM Zones:        Disabled")
    print(f"  Visualization:    {'Enabled' if args.visualize else 'Disabled'}")
    print(f"  Export Journal:   {args.export_journal if args.export_journal else 'Disabled'}")
    print("")

    # Step 1: Load data (force refresh to get latest)
    print(f"📂 Downloading fresh {symbol} data for all timeframes...")
    loader = DataLoader(symbol=symbol)
    force_refresh = False
    df_high = loader.get_data(high_tf, force_refresh=force_refresh)
    df_mid = loader.get_data(mid_tf, force_refresh=force_refresh)
    df_low = loader.get_data(low_tf, force_refresh=force_refresh)

    print(f"  ✓ {high_tf.upper()}:  {len(df_high)} candles")
    print(f"  ✓ {mid_tf.upper()}:  {len(df_mid)} candles")
    print(f"  ✓ {low_tf.upper()}:  {len(df_low)} candles")

    # Get GMM zone settings
    gmm_config = config.get('gmm_zones', {})

    # ===== 3-OB strategy branch =====
    if args.strategy == '3-ob':
        three_ob_cfg = config.get('three_ob', {})
        three_ob_tf = three_ob_cfg.get('timeframe', mid_tf)
        three_ob_close_break = three_ob_cfg.get('close_break', True)
        three_ob_min_dist = three_ob_cfg.get('min_dist_pct', 10.0) / 100.0
        three_ob_enable_shorts = three_ob_cfg.get('enable_shorts', False)
        three_ob_conf_tf = three_ob_cfg.get('confirmation_timeframe', None)

        print(f"\n🔧 Initializing 3-OB strategy engine (TF: {three_ob_tf})...")
        df_3ob = loader.get_data(three_ob_tf, force_refresh=False)
        # df_3ob = df_3ob[df_3ob.shape[0] * 2 // 3:].reset_index(drop=True)  # take last ~1/3 for faster iteration
        print(f"  ✓ {three_ob_tf.upper()}: {len(df_3ob)} candles")

        # Load lower TF confirmation data if configured
        df_confirmation = None
        if three_ob_conf_tf:
            print(f"  Loading confirmation TF data ({three_ob_conf_tf})...")
            df_confirmation = loader.get_data(three_ob_conf_tf, force_refresh=False)
            print(f"  ✓ {three_ob_conf_tf.upper()}: {len(df_confirmation)} candles")

            # Auto-refresh if confirmation data is stale compared to primary TF
            primary_end = df_3ob['time'].max()
            conf_end = df_confirmation['time'].max()
            if conf_end < primary_end:
                print(f"  ⚠️ Confirmation data ends at {conf_end}, primary TF goes to {primary_end}")
                print(f"  🔄 Refreshing {three_ob_conf_tf} data...")
                df_confirmation = loader.update_cache(three_ob_conf_tf)
                print(f"  ✓ {three_ob_conf_tf.upper()}: {len(df_confirmation)} candles (refreshed)")

        # Align primary TF data to confirmation data coverage
        if df_confirmation is not None and len(df_confirmation) > 0:
            conf_start = df_confirmation['time'].min()
            primary_start = df_3ob['time'].min()
            if primary_start < conf_start:
                original_len = len(df_3ob)
                df_3ob = df_3ob[df_3ob['time'] >= conf_start].reset_index(drop=True)
                print(f"  ⚠️ Trimmed {three_ob_tf} data from {original_len} to {len(df_3ob)} candles to align with {three_ob_conf_tf} coverage ({conf_start})")

        strategy_3ob = ThreeOBStrategy(
            df_3ob,
            close_break=three_ob_close_break,
            min_dist_pct=three_ob_min_dist,
            enable_shorts=three_ob_enable_shorts,
            df_confirmation=df_confirmation,
            confirmation_timeframe=three_ob_conf_tf,
        )

        print("\n🔍 Scanning for 3-OB trade setups...")
        signal_contexts = None
        if args.visualize:
            signal_contexts = strategy_3ob.scan_for_signals_with_context(max_signals=999_999)
            signals = [ctx.signal for ctx in signal_contexts]
        else:
            signals = strategy_3ob.scan_for_signals(max_signals=999_999)

        # Generate walkthrough if requested (before signal check so it runs even with 0 signals)
        if args.walkthrough_3ob:
            print("\n" + "="*80)
            print("GENERATING 3-OB WALKTHROUGH")
            print("="*80)
            walkthrough = ThreeOBWalkthrough()
            output_path = walkthrough.run(
                df_3ob,
                close_break=three_ob_close_break,
                min_dist_pct=three_ob_min_dist,
                max_signals=999_999,
                enable_shorts=three_ob_enable_shorts,
                df_confirmation=df_confirmation,
                confirmation_timeframe=three_ob_conf_tf,
            )
            print(f"\n  Open {output_path} in your browser")
            print("  Use arrow keys to step through candles")

        if len(signals) == 0:
            print("\n⚠️  No 3-OB signals found.")
            return

        print(f"\n✅ Found {len(signals)} 3-OB trade setups!")

        # Run backtest
        if not args.no_backtest:
            print("\n" + "="*80)
            print("RUNNING BACKTEST SIMULATION (3-OB)")
            print("="*80)

            backtester = Backtester(
                df_low=strategy_3ob.df_low,
                initial_capital=args.initial_capital,
                symbol=symbol,
                intraday_only=not args.hold_overnight,
                min_profit_percent=min_profit_percent,
                min_rr_ratio=min_rr_ratio,
                trailing_sl_enabled=trailing_sl_enabled,
                trailing_sl_activation_pct=trailing_sl_activation_pct,
                trailing_sl_swing_length=trailing_sl_swing_length,
            )

            # Filter signals to only those with low-TF data coverage
            no_data_skipped = []
            if df_confirmation is not None and len(df_confirmation) > 0:
                conf_start = df_confirmation['time'].min()
                conf_end = df_confirmation['time'].max()
                tradeable_signals = []
                for s in signals:
                    if conf_start <= s.timestamp_entry <= conf_end:
                        tradeable_signals.append(s)
                    else:
                        no_data_skipped.append(SkippedTrade(
                            signal_entry_time=s.timestamp_entry,
                            skip_reason=SkipReason.NO_LOW_TF_DATA,
                            details=f"Signal at {s.timestamp_entry} outside low-TF data range ({conf_start} to {conf_end})"
                        ))
                if no_data_skipped:
                    print(f"\n  Skipping {len(no_data_skipped)} signals without low-TF data coverage")
                    print(f"  Backtesting {len(tradeable_signals)} signals with low-TF data")
            else:
                tradeable_signals = signals

            _results = backtester.run(tradeable_signals)
            backtester.print_summary()

            if args.export_journal:
                os.makedirs("results/trades", exist_ok=True)
                journal_path = f"results/trades/{symbol.lower()}_3ob_trades.csv" if args.export_journal == 'auto' else args.export_journal
                backtester.export_journal(journal_path)

        # Generate signal viewer visualization if requested
        if args.visualize and signal_contexts:
            print("\n" + "="*80)
            print("GENERATING 3-OB SIGNAL VISUALIZATION")
            print("="*80)

            trade_results = backtester.results if not args.no_backtest and 'backtester' in locals() else []
            perf_metrics = backtester.metrics if not args.no_backtest and 'backtester' in locals() else None
            skipped_trades = (backtester.skipped_trades if not args.no_backtest and 'backtester' in locals() else []) + (no_data_skipped if 'no_data_skipped' in locals() else [])
            viewer = ThreeOBSignalViewer(
                df=df_3ob,
                signal_contexts=signal_contexts,
                symbol=symbol,
                trade_results=trade_results,
                performance_metrics=perf_metrics,
                df_entry=df_confirmation,
                skipped_trades=skipped_trades,
                trailing_sl_activation_pct=trailing_sl_activation_pct,
            )
            viz_path = viewer.generate_html()
            print(f"\n✅ Signal viewer: {viz_path}")
            print("   Use dropdown or arrow keys to browse signals")

        # Summary
        print("\n" + "="*80)
        print("✅ 3-OB ANALYSIS COMPLETE!")
        print("="*80)
        for i, sig in enumerate(signals, 1):
            conf_tag = " [NO CONF]" if "No confirmation" in (sig.condition_confirmation or "") else ""
            print(f"\n  Signal #{i}: {sig.entry_direction.upper()}{conf_tag}")
            print(f"    Entry: ${sig.price_entry:.2f} @ {sig.timestamp_entry}")
            print(f"    TP: ${sig.take_profit_price:.2f}" if sig.take_profit_price else "    TP: N/A")
            print(f"    SL: ${sig.stop_loss_price:.2f}" if sig.stop_loss_price else "    SL: N/A")
        print("="*80 + "\n")
        return

    # Step 2: Initialize strategy
    print("\n🔧 Initializing strategy engine...")
    strategy = MultiTimeframeStrategy(
        df_high, df_mid, df_low, timeframe_config,
        use_fvg_validation=use_fvg_validation,
        use_equilibrium_validation=use_equilibrium_validation,
        require_fvg_in_equilibrium=require_fvg_in_equilibrium,
        sweep_proximity_threshold=sweep_proximity_threshold,
        abandon_on_new_sweep=abandon_on_new_sweep,
        gmm_config=gmm_config,
        high_tf_lookback=high_tf_lookback,
        use_fvg_trigger=use_fvg_trigger
    )

    # Step 3: Scan for signals
    print("\n🔍 Scanning for trade setups...")
    signals = strategy.scan_for_signals(max_signals=999_999)

    # Handle complete signals
    if len(signals) > 0:
        print(f"\n✅ Found {len(signals)} complete trade setups!")
        print("   (Skipping old visualization - use TradingView analyzer instead)")

    # Run ARMA trend analysis if enabled (before backtest, for both backtest and visualization)
    import numpy as np
    trend_signal = None
    arma_data = None
    if trend_filter_enabled:
        from src.arma_trend_analysis import ARMA

        # Use mid timeframe prices for trend calculation
        mid_prices = list(df_mid['close'])
        lookback_prices = mid_prices[-trend_filter_lookback:] if len(mid_prices) >= trend_filter_lookback else mid_prices

        arma_result = ARMA.fit_arma(lookback_prices, auto_select=True, max_order=3)
        trend_signal = arma_result['trend_signal']

        print(f"\n📊 ARMA Trend Analysis:")
        print(f"   Lookback:  {len(lookback_prices)} candles")
        print(f"   Order:     ARMA({arma_result['order'][0]}, {arma_result['order'][1]})")
        print(f"   Drift:     {arma_result['drift']:.6f}")
        print(f"   SNR:       {arma_result['snr']:.4f}")
        print(f"   Trend:     {trend_signal.upper()}")
        if trend_signal != 'neutral':
            print(f"   Filter:    Active - only {trend_signal.upper()} trades allowed")
        else:
            print(f"   Filter:    Inactive (neutral trend - all directions allowed)")
        print("")

        # Prepare ARMA data for visualization
        arma_data = {
            'prices': lookback_prices,
            'log_returns': arma_result['log_returns'].tolist(),
            'fitted_returns': arma_result['fitted_returns'].tolist(),
            'trend_signal': trend_signal,
            'snr': arma_result['snr'],
            'drift': arma_result['drift'],
            'order': list(arma_result['order']),
            'aic': arma_result['aic'],
            'bic': arma_result['bic'],
            'lookback': len(lookback_prices),
            'residual_std': float(np.sqrt(arma_result['sigma2']))
        }

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
            min_profit_percent=min_profit_percent,
            min_rr_ratio=min_rr_ratio,
            trend_filter=trend_signal if trend_filter_enabled else None,
            bos_1h_trend_filter_enabled=bos_1h_trend_filter_enabled,
            trailing_sl_enabled=trailing_sl_enabled,
            trailing_sl_activation_pct=trailing_sl_activation_pct,
            trailing_sl_swing_length=trailing_sl_swing_length
        )
        _results = backtester.run(signals)
        backtester.print_summary()

        if args.export_journal:
            # Generate default path if 'auto'
            if args.export_journal == 'auto':
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
            performance_metrics = backtester.metrics if backtester else None
            visualizer = StrategySweepVisualizer(
                strategy,
                symbol=symbol,
                trade_results=trade_results,
                skipped_trades=skipped_trades,
                performance_metrics=performance_metrics,
                arma_data=arma_data
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
