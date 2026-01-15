"""
Performance metrics calculation and display.

Calculates aggregate statistics from trade results.
"""

from typing import List
from .models import TradeResult, TradeOutcome, ExitType, PerformanceMetrics


class PerformanceTracker:
    """
    Calculates and displays performance metrics from trade results.
    """

    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize performance tracker.

        Args:
            initial_capital: Starting capital for metrics
        """
        self.initial_capital = initial_capital

    def calculate_metrics(self, results: List[TradeResult]) -> PerformanceMetrics:
        """
        Calculate aggregate performance metrics from trade results.

        Args:
            results: List of TradeResult objects

        Returns:
            PerformanceMetrics with all calculated statistics
        """
        if not results:
            return self._empty_metrics()

        # Basic counts (based on P&L)
        total_trades = len(results)
        winning_trades = sum(1 for r in results if r.outcome == TradeOutcome.WIN)
        losing_trades = sum(1 for r in results if r.outcome == TradeOutcome.LOSS)

        # Exit type counts
        tp_exits = sum(1 for r in results if r.exit_type == ExitType.TP_HIT)
        sl_exits = sum(1 for r in results if r.exit_type == ExitType.SL_HIT)
        timeout_exits = sum(1 for r in results if r.exit_type == ExitType.TIMEOUT)

        # Win rate (simple: wins / total)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # P&L metrics
        final_capital = results[-1].capital_after if results else self.initial_capital
        total_pnl = final_capital - self.initial_capital
        total_return = (total_pnl / self.initial_capital) * 100

        # Win/Loss breakdown
        wins = [r for r in results if r.outcome == TradeOutcome.WIN]
        losses = [r for r in results if r.outcome == TradeOutcome.LOSS]

        avg_win = sum(r.pnl_dollars for r in wins) / len(wins) if wins else 0.0
        avg_loss = sum(r.pnl_dollars for r in losses) / len(losses) if losses else 0.0
        largest_win = max((r.pnl_dollars for r in wins), default=0.0)
        largest_loss = min((r.pnl_dollars for r in losses), default=0.0)

        # Profit factor
        gross_profit = sum(r.pnl_dollars for r in results if r.pnl_dollars > 0)
        gross_loss = abs(sum(r.pnl_dollars for r in results if r.pnl_dollars < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average R-multiple
        avg_r = sum(r.pnl_r_multiple for r in results) / len(results) if results else 0.0

        # Drawdown calculation
        equity_curve = [self.initial_capital]
        for r in results:
            equity_curve.append(r.capital_after)

        max_dd_dollars, max_dd_percent = self._calculate_max_drawdown(equity_curve)

        # Consecutive wins/losses
        max_consec_wins, max_consec_losses = self._calculate_consecutive_streaks(results)

        # Duration metrics
        durations = [r.duration_minutes for r in results if r.duration_minutes is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # Direction breakdown
        long_trades = [r for r in results if r.entry_direction == 'long']
        short_trades = [r for r in results if r.entry_direction == 'short']
        long_wins = sum(1 for r in long_trades if r.outcome == TradeOutcome.WIN)
        short_wins = sum(1 for r in short_trades if r.outcome == TradeOutcome.WIN)

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            tp_exits=tp_exits,
            sl_exits=sl_exits,
            timeout_exits=timeout_exits,
            win_rate=win_rate,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_pnl_dollars=total_pnl,
            total_return_percent=total_return,
            average_win_dollars=avg_win,
            average_loss_dollars=avg_loss,
            largest_win_dollars=largest_win,
            largest_loss_dollars=largest_loss,
            profit_factor=profit_factor,
            average_r_multiple=avg_r,
            max_drawdown_dollars=max_dd_dollars,
            max_drawdown_percent=max_dd_percent,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            average_trade_duration_minutes=avg_duration,
            long_trades=len(long_trades),
            short_trades=len(short_trades),
            long_wins=long_wins,
            short_wins=short_wins,
            equity_curve=equity_curve
        )

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> tuple:
        """
        Calculate maximum drawdown from equity curve.

        Returns:
            (max_drawdown_dollars, max_drawdown_percent)
        """
        if len(equity_curve) < 2:
            return 0.0, 0.0

        peak = equity_curve[0]
        max_dd_dollars = 0.0
        max_dd_percent = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = peak - value
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0.0

            if drawdown > max_dd_dollars:
                max_dd_dollars = drawdown
                max_dd_percent = drawdown_pct

        return max_dd_dollars, max_dd_percent

    def _calculate_consecutive_streaks(self, results: List[TradeResult]) -> tuple:
        """
        Calculate max consecutive wins and losses.

        Returns:
            (max_consecutive_wins, max_consecutive_losses)
        """
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for r in results:
            if r.outcome == TradeOutcome.WIN:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:  # LOSS
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades."""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            tp_exits=0,
            sl_exits=0,
            timeout_exits=0,
            win_rate=0.0,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_pnl_dollars=0.0,
            total_return_percent=0.0,
            average_win_dollars=0.0,
            average_loss_dollars=0.0,
            largest_win_dollars=0.0,
            largest_loss_dollars=0.0,
            profit_factor=0.0,
            average_r_multiple=0.0,
            max_drawdown_dollars=0.0,
            max_drawdown_percent=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            average_trade_duration_minutes=0.0,
            long_trades=0,
            short_trades=0,
            long_wins=0,
            short_wins=0,
            equity_curve=[self.initial_capital]
        )

    def print_summary(self, metrics: PerformanceMetrics) -> None:
        """Print formatted performance summary to console."""
        print("\n" + "=" * 70)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 70)

        print("\n--- TRADE STATISTICS ---")
        print(f"  Total Trades:       {metrics.total_trades}")
        print(f"  Winning Trades:     {metrics.winning_trades}")
        print(f"  Losing Trades:      {metrics.losing_trades}")
        print(f"  Win Rate:           {metrics.win_rate:.1%}")

        print("\n--- EXIT TYPE BREAKDOWN ---")
        print(f"  TP Exits:           {metrics.tp_exits}")
        print(f"  SL Exits:           {metrics.sl_exits}")
        print(f"  Timeout Exits:      {metrics.timeout_exits}")

        print("\n--- PROFIT & LOSS (Compounding) ---")
        print(f"  Initial Capital:    ${metrics.initial_capital:,.2f}")
        print(f"  Final Capital:      ${metrics.final_capital:,.2f}")
        print(f"  Total P&L:          ${metrics.total_pnl_dollars:+,.2f}")
        print(f"  Total Return:       {metrics.total_return_percent:+.2f}%")
        pf_display = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float('inf') else "INF"
        print(f"  Profit Factor:      {pf_display}")

        print("\n--- WIN/LOSS BREAKDOWN ---")
        print(f"  Average Win:        ${metrics.average_win_dollars:+,.2f}")
        print(f"  Average Loss:       ${metrics.average_loss_dollars:+,.2f}")
        print(f"  Largest Win:        ${metrics.largest_win_dollars:+,.2f}")
        print(f"  Largest Loss:       ${metrics.largest_loss_dollars:+,.2f}")

        print("\n--- RISK METRICS ---")
        print(f"  Average R:          {metrics.average_r_multiple:+.2f}R")
        print(f"  Max Drawdown:       ${metrics.max_drawdown_dollars:,.2f} ({metrics.max_drawdown_percent:.1f}%)")
        print(f"  Max Consec. Wins:   {metrics.max_consecutive_wins}")
        print(f"  Max Consec. Losses: {metrics.max_consecutive_losses}")

        print("\n--- DIRECTION BREAKDOWN ---")
        long_wr = metrics.long_wins / metrics.long_trades * 100 if metrics.long_trades > 0 else 0
        short_wr = metrics.short_wins / metrics.short_trades * 100 if metrics.short_trades > 0 else 0
        print(f"  Long Trades:        {metrics.long_trades} ({long_wr:.0f}% win rate)")
        print(f"  Short Trades:       {metrics.short_trades} ({short_wr:.0f}% win rate)")

        print("\n--- DURATION ---")
        if metrics.average_trade_duration_minutes > 60:
            hours = metrics.average_trade_duration_minutes / 60
            print(f"  Avg Trade Duration: {hours:.1f} hours")
        else:
            print(f"  Avg Trade Duration: {metrics.average_trade_duration_minutes:.0f} minutes")

        print("\n--- EQUITY CURVE ---")
        if metrics.equity_curve and len(metrics.equity_curve) > 1:
            for i, (before, after) in enumerate(zip(metrics.equity_curve[:-1], metrics.equity_curve[1:])):
                pnl = after - before
                print(f"  Trade {i+1}: ${before:>10,.2f} -> ${after:>10,.2f} (${pnl:+,.2f})")

        print("=" * 70 + "\n")
