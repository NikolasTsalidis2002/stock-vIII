"""
Data models for backtesting module.

Contains dataclasses for trade results and performance metrics.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Tuple


class TradeOutcome(Enum):
    """Possible trade outcomes based on P&L."""
    WIN = "win"           # Trade closed with positive P&L
    LOSS = "loss"         # Trade closed with negative or zero P&L


class SkipReason(Enum):
    """Reasons why a trade signal was not executed."""
    NONE = "none"                          # Trade was executed (no skip)
    OVERLAP = "overlap"                    # Another position was open
    MISSING_TP_SL = "missing_tp_sl"        # No TP or SL defined
    AFTER_MARKET_CLOSE = "after_close"     # Entry time after market hours
    PARTIAL_SETUP = "partial_setup"        # Setup conditions not met
    NO_BACKTEST = "no_backtest"            # Backtest not run
    INSUFFICIENT_PROFIT = "insufficient_profit"  # Below min_profit_percent threshold
    INSUFFICIENT_RR = "insufficient_rr"  # Below min_rr_ratio threshold
    AGAINST_TREND = "against_trend"        # Trade direction opposes ARMA trend
    NO_LOW_TF_DATA = "no_low_tf_data"      # No low-timeframe data for backtesting


# Human-readable messages for skip reasons
SKIP_REASON_MESSAGES = {
    SkipReason.NONE: None,
    SkipReason.OVERLAP: "Skipped - another position was open",
    SkipReason.MISSING_TP_SL: "Skipped - missing TP or SL levels",
    SkipReason.AFTER_MARKET_CLOSE: "Skipped - entry after market close",
    SkipReason.PARTIAL_SETUP: "Setup incomplete",
    SkipReason.NO_BACKTEST: "No backtest data available",
    SkipReason.INSUFFICIENT_PROFIT: "Skipped - profit potential below minimum threshold",
    SkipReason.INSUFFICIENT_RR: "Skipped - reward-to-risk ratio below minimum threshold",
    SkipReason.AGAINST_TREND: "Skipped - trade direction opposes ARMA trend",
    SkipReason.NO_LOW_TF_DATA: "Skipped - no low-timeframe data for backtesting",
}


@dataclass
class SkippedTrade:
    """Tracks a signal that was not executed and why."""
    signal_entry_time: datetime
    skip_reason: SkipReason
    details: Optional[str] = None  # Additional context (e.g., failure reason for partials)


class ExitType(Enum):
    """How the trade was exited."""
    TP_HIT = "tp_hit"         # Take profit price was hit
    SL_HIT = "sl_hit"         # Stop loss price was hit
    TIMEOUT = "timeout"       # Neither TP nor SL hit (timeout or end of data)
    TRAILING_SL = "trailing_sl"  # Exited due to trailing stop loss hit


class RejectionReason(Enum):
    """Reasons why a trade signal was rejected before execution (exit_price=None)."""
    NONE = "none"                              # Trade was executed (no rejection)
    AFTER_MARKET_CLOSE = "after_market_close"  # Entry time at/after market close
    ENTRY_NOT_FOUND = "entry_not_found"        # Entry candle not found in data
    ENTRY_ALREADY_HIT_TP_SL = "entry_already_hit_tp_sl"  # Entry candle already hit TP or SL
    NO_DATA_AFTER_ENTRY = "no_data_after_entry"  # No data remaining after entry candle


# Human-readable messages for rejection reasons
REJECTION_REASON_MESSAGES = {
    RejectionReason.NONE: None,
    RejectionReason.AFTER_MARKET_CLOSE: "Entry at/after market close",
    RejectionReason.ENTRY_NOT_FOUND: "Entry candle not found in data",
    RejectionReason.ENTRY_ALREADY_HIT_TP_SL: "Entry candle already touched TP or SL (invalid setup)",
    RejectionReason.NO_DATA_AFTER_ENTRY: "No data remaining after entry candle",
}


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
    outcome: TradeOutcome            # WIN or LOSS based on P&L
    exit_type: ExitType              # How the trade exited (TP_HIT, SL_HIT, TIMEOUT)

    # Position sizing (compounding)
    capital_before: float                # Capital before this trade
    position_size_usd: float             # Same as capital_before (full position)
    shares_traded: float                 # position_size_usd / entry_price

    # P&L calculations
    pnl_dollars: float                   # Actual dollar P&L
    pnl_percent: float                   # Percentage return on this trade
    pnl_r_multiple: float                # P&L in R (risk units)
    capital_after: float                 # Capital after this trade

    # Trade duration (fields with defaults must come after required fields)
    rejection_reason: Optional['RejectionReason'] = None  # Why trade was rejected (if exit_price=None)
    duration: Optional[timedelta] = None
    duration_minutes: Optional[int] = None

    # Price action during trade
    max_adverse_excursion: float = 0.0   # Worst drawdown during trade
    max_favorable_excursion: float = 0.0 # Best profit during trade

    # Edge case flags
    gap_exit: bool = False               # True if exit was via gap
    entry_candle_idx: Optional[int] = None  # 1M candle index at entry
    exit_candle_idx: Optional[int] = None   # 1M candle index at exit
    total_candles_in_trade: int = 0

    # Candle-by-candle unrealized P&L series: [(timestamp, unrealized_pnl_dollars), ...]
    unrealized_pnl_series: List[Tuple[datetime, float]] = None

    def risk_amount(self) -> float:
        """Calculate the dollar risk on this trade."""
        return abs(self.entry_price - self.stop_loss_price) * self.shares_traded

    def reward_amount(self) -> float:
        """Calculate the potential dollar reward on this trade."""
        return abs(self.entry_price - self.take_profit_price) * self.shares_traded

    def max_potential_profit_dollars(self) -> float:
        """Maximum profit if exited at peak favorable excursion."""
        return self.max_favorable_excursion * self.shares_traded

    def profit_left_on_table(self) -> float:
        """Difference between max potential and actual profit."""
        return self.max_potential_profit_dollars() - self.pnl_dollars

    def tp_progress_percent(self) -> float:
        """Percentage of distance to TP reached at best point."""
        distance_to_tp = abs(self.take_profit_price - self.entry_price)
        if distance_to_tp == 0:
            return 0.0
        return (self.max_favorable_excursion / distance_to_tp) * 100


@dataclass
class PerformanceMetrics:
    """Aggregate performance statistics across all trades."""

    # Basic counts
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Exit type counts
    tp_exits: int                        # Trades that hit take profit
    sl_exits: int                        # Trades that hit stop loss
    timeout_exits: int                   # Trades that timed out
    trailing_sl_exits: int                # Trades that exited via trailing stop loss

    # Win/Loss metrics
    win_rate: float                      # winning_trades / total_trades

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
