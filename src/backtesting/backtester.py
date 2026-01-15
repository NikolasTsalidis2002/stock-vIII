"""
Main backtesting orchestrator.

Ties together the trade simulator, performance tracker, and trade journal
to provide a complete backtesting solution.
"""

from typing import List, Optional
import pandas as pd

from src.strategy.models import TradeSignal
from .models import TradeResult, PerformanceMetrics
from .trade_simulator import TradeSimulator
from .performance_tracker import PerformanceTracker
from .trade_journal import TradeJournal


class Backtester:
    """
    Complete backtesting solution for the multi-timeframe strategy.

    Usage:
        # Option 1: Integrate with existing strategy
        strategy = MultiTimeframeStrategy(df_1h, df_5m, df_1m)
        signals = strategy.scan_for_signals(max_signals=10)

        backtester = Backtester(df_1m=strategy.df_1m)
        results = backtester.run(signals)
        backtester.print_summary()
        backtester.export_journal("results/backtest_journal.csv")

        # Option 2: Create from strategy directly
        backtester = Backtester.from_strategy(strategy)
        results = backtester.run(signals)
    """

    def __init__(
        self,
        df_1m: pd.DataFrame,
        initial_capital: float = 10000.0,
        max_trade_duration_hours: int = 24,
        symbol: str = 'TSLA'
    ) -> None:
        """
        Initialize backtester.

        Args:
            df_1m: 1M OHLCV DataFrame for exit simulation
            initial_capital: Starting capital (default $10,000)
            max_trade_duration_hours: Max hours before timeout
            symbol: Stock symbol being backtested
        """
        self.df_1m = df_1m
        self.initial_capital = initial_capital
        self.symbol = symbol.upper()

        # Initialize components
        self._simulator = TradeSimulator(
            df_1m=df_1m,
            initial_capital=initial_capital,
            max_trade_duration_hours=max_trade_duration_hours
        )
        self._tracker = PerformanceTracker(initial_capital=initial_capital)
        self._journal = TradeJournal(symbol=self.symbol)

        # Results storage
        self._results: List[TradeResult] = []
        self._metrics: Optional[PerformanceMetrics] = None
        self._signals: List[TradeSignal] = []

    @classmethod
    def from_strategy(
        cls,
        strategy,  # MultiTimeframeStrategy
        initial_capital: float = 10000.0,
        max_trade_duration_hours: int = 24,
        symbol: str = 'TSLA'
    ) -> "Backtester":
        """
        Create backtester directly from strategy instance.

        Args:
            strategy: Initialized MultiTimeframeStrategy
            initial_capital: Starting capital
            max_trade_duration_hours: Max hours before timeout
            symbol: Stock symbol being backtested

        Returns:
            Configured Backtester instance
        """
        return cls(
            df_1m=strategy.df_1m,
            initial_capital=initial_capital,
            max_trade_duration_hours=max_trade_duration_hours,
            symbol=symbol
        )

    def run(self, signals: List[TradeSignal]) -> List[TradeResult]:
        """
        Run backtest on provided signals.

        Args:
            signals: List of TradeSignal objects from strategy

        Returns:
            List of TradeResult objects
        """
        print("\n" + "=" * 70)
        print("BACKTESTING TRADE SIGNALS")
        print("=" * 70)
        print(f"  Signals to test: {len(signals)}")
        print(f"  Initial capital: ${self.initial_capital:,.2f}")
        print(f"  Mode: Full position compounding")
        print("=" * 70)

        self._signals = signals
        self._results = self._simulator.simulate_all(signals)
        self._metrics = self._tracker.calculate_metrics(self._results)

        return self._results

    def print_summary(self) -> None:
        """Print performance summary to console."""
        if self._metrics is None:
            print("No results to summarize. Run backtest first.")
            return

        self._tracker.print_summary(self._metrics)

    def print_trade_table(self, max_rows: int = 20) -> None:
        """Print trade table to console."""
        self._journal.print_trade_table(self._results, max_rows)

    def export_journal(
        self,
        filepath: str,
        include_signals: bool = True
    ) -> None:
        """
        Export trade journal to CSV.

        Args:
            filepath: Path for CSV file
            include_signals: Include original signal details
        """
        self._journal.export_csv(
            results=self._results,
            signals=self._signals if include_signals else None,
            filepath=filepath
        )

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame for analysis."""
        return self._journal.to_dataframe(self._results, self._signals)

    @property
    def results(self) -> List[TradeResult]:
        """List of trade results."""
        return self._results

    @property
    def metrics(self) -> Optional[PerformanceMetrics]:
        """Performance metrics (None if not yet calculated)."""
        return self._metrics

    @property
    def equity_curve(self) -> List[float]:
        """Equity curve from metrics."""
        if self._metrics and self._metrics.equity_curve:
            return self._metrics.equity_curve
        return [self.initial_capital]
