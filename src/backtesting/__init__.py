"""
Backtesting module for trade simulation and performance analysis.

Usage:
    from src.backtesting import Backtester, TradeResult, PerformanceMetrics

    # Create backtester from strategy
    backtester = Backtester.from_strategy(strategy)

    # Or create directly with 1M data
    backtester = Backtester(df_1m=df_1m)

    # Run backtest
    results = backtester.run(signals)

    # View results
    backtester.print_summary()
    backtester.print_trade_table()
    backtester.export_journal("results/trades.csv")
"""

from .models import TradeResult, TradeOutcome, PerformanceMetrics
from .trade_simulator import TradeSimulator
from .performance_tracker import PerformanceTracker
from .trade_journal import TradeJournal
from .backtester import Backtester

__all__ = [
    'TradeResult',
    'TradeOutcome',
    'PerformanceMetrics',
    'TradeSimulator',
    'PerformanceTracker',
    'TradeJournal',
    'Backtester',
]
