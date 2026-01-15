"""
Data models for backtesting module.

Contains dataclasses for trade results and performance metrics.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class TradeOutcome(Enum):
    """Possible trade outcomes."""
    WIN = "win"           # Take profit hit first
    LOSS = "loss"         # Stop loss hit first
    TIMEOUT = "timeout"   # Neither TP nor SL hit by end of data


@dataclass
class TradeResult:
    """Complete result of a simulated trade execution."""

    # Trade identification
    signal_index: int                    # Index in the signals list

    # Entry details (from TradeSignal)
    entry_direction: str                 # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    take_profit_price: float
    stop_loss_price: float

    # Exit details (calculated by simulator)
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    outcome: TradeOutcome

    # Position sizing (compounding)
    capital_before: float                # Capital before this trade
    position_size_usd: float             # Same as capital_before (full position)
    shares_traded: float                 # position_size_usd / entry_price

    # P&L calculations
    pnl_dollars: float                   # Actual dollar P&L
    pnl_percent: float                   # Percentage return on this trade
    pnl_r_multiple: float                # P&L in R (risk units)
    capital_after: float                 # Capital after this trade

    # Trade duration
    duration: Optional[timedelta] = None
    duration_minutes: Optional[int] = None

    # Price action during trade
    max_adverse_excursion: float = 0.0   # Worst drawdown during trade
    max_favorable_excursion: float = 0.0 # Best profit during trade

    # Edge case flags
    gap_exit: bool = False               # True if exit was via gap
    exit_candle_idx: Optional[int] = None
    total_candles_in_trade: int = 0

    def risk_amount(self) -> float:
        """Calculate the dollar risk on this trade."""
        return abs(self.entry_price - self.stop_loss_price) * self.shares_traded

    def reward_amount(self) -> float:
        """Calculate the potential dollar reward on this trade."""
        return abs(self.entry_price - self.take_profit_price) * self.shares_traded


@dataclass
class PerformanceMetrics:
    """Aggregate performance statistics across all trades."""

    # Basic counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    timeout_trades: int

    # Win/Loss metrics
    win_rate: float                      # winning_trades / (winning + losing)

    # P&L metrics (compounding)
    initial_capital: float
    final_capital: float
    total_pnl_dollars: float
    total_return_percent: float

    # Win/Loss breakdown
    average_win_dollars: float
    average_loss_dollars: float
    largest_win_dollars: float
    largest_loss_dollars: float

    # Risk-adjusted metrics
    profit_factor: float                 # gross_profit / gross_loss
    average_r_multiple: float            # Average R per trade

    # Drawdown metrics
    max_drawdown_dollars: float
    max_drawdown_percent: float
    max_consecutive_losses: int
    max_consecutive_wins: int

    # Duration metrics
    average_trade_duration_minutes: float

    # Direction breakdown
    long_trades: int
    short_trades: int
    long_wins: int
    short_wins: int

    # Equity curve (list of capital values after each trade)
    equity_curve: list = None
