"""
Trade execution simulator using walk-forward methodology.

Walks through 1M candles from entry timestamp to determine
whether TP or SL was hit first.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from src.strategy.models import TradeSignal
from .models import TradeResult, TradeOutcome


class TradeSimulator:
    """
    Simulates trade execution by walking forward through 1M price data.

    Key features:
    - Uses 1M timeframe for precise exit detection
    - Handles price gaps (open beyond TP/SL)
    - Tracks maximum adverse/favorable excursion
    - Supports compounding position sizing
    """

    def __init__(
        self,
        df_1m: pd.DataFrame,
        initial_capital: float = 10000.0,
        max_trade_duration_hours: int = 24
    ) -> None:
        """
        Initialize trade simulator.

        Args:
            df_1m: 1M OHLCV DataFrame with datetime index
            initial_capital: Starting capital (default $10,000)
            max_trade_duration_hours: Max hours before timeout (default 24h)
        """
        self.df_1m = df_1m
        self.initial_capital = initial_capital
        self.max_trade_duration = timedelta(hours=max_trade_duration_hours)

    def simulate_trade(
        self,
        signal: TradeSignal,
        signal_index: int,
        current_capital: float
    ) -> TradeResult:
        """
        Simulate a single trade from entry to exit.

        Algorithm:
        1. Start at entry timestamp
        2. Walk forward through each 1M candle
        3. For each candle, check if high/low crossed TP or SL
        4. Handle gaps: if candle opens beyond TP/SL, use open price
        5. Track MAE/MFE during walk
        6. Stop when: TP hit, SL hit, timeout, or end of data

        Args:
            signal: TradeSignal with entry/TP/SL details
            signal_index: Index in signals list (for tracking)
            current_capital: Capital available for this trade (compounding)

        Returns:
            TradeResult with complete execution details
        """
        # Extract signal details
        entry_price = signal.price_entry
        entry_time = signal.timestamp_entry
        tp_price = signal.take_profit_price
        sl_price = signal.stop_loss_price
        direction = signal.entry_direction

        # Calculate position size (full capital)
        position_size = current_capital
        shares = position_size / entry_price

        # Initialize tracking variables
        exit_price = None
        exit_time = None
        outcome = TradeOutcome.TIMEOUT
        exit_candle_idx = None
        gap_exit = False

        # Track excursions
        max_favorable = 0.0
        max_adverse = 0.0

        # Find starting point in 1M data
        start_idx = self._find_start_index(entry_time)
        if start_idx is None:
            return self._create_timeout_result(
                signal, signal_index, current_capital, shares, 0.0, 0.0, 0
            )

        # Calculate timeout boundary
        timeout_time = entry_time + self.max_trade_duration

        # Walk forward through 1M candles
        candles_scanned = 0

        for idx in range(start_idx, len(self.df_1m)):
            candle = self.df_1m.iloc[idx]
            candle_time = self.df_1m.index[idx]
            candles_scanned += 1

            # Check timeout
            if candle_time > timeout_time:
                exit_price = candle['close']
                exit_time = candle_time
                exit_candle_idx = idx
                outcome = TradeOutcome.TIMEOUT
                break

            candle_open = candle['open']
            candle_high = candle['high']
            candle_low = candle['low']

            # Check for gap opening beyond TP/SL
            gap_result = self._check_gap_exit(
                candle_open, tp_price, sl_price, direction
            )
            if gap_result is not None:
                exit_price = candle_open
                exit_time = candle_time
                exit_candle_idx = idx
                outcome = gap_result
                gap_exit = True
                break

            # Normal candle processing: check high/low for TP/SL
            tp_hit, sl_hit = self._check_candle_crosses(
                candle_high, candle_low, tp_price, sl_price, direction
            )

            if tp_hit and sl_hit:
                # Both TP and SL touched in same candle - assume worst case (SL hit first)
                exit_price = sl_price
                exit_time = candle_time
                exit_candle_idx = idx
                outcome = TradeOutcome.LOSS
                break
            elif tp_hit:
                exit_price = tp_price
                exit_time = candle_time
                exit_candle_idx = idx
                outcome = TradeOutcome.WIN
                break
            elif sl_hit:
                exit_price = sl_price
                exit_time = candle_time
                exit_candle_idx = idx
                outcome = TradeOutcome.LOSS
                break

            # Update excursions (no exit yet)
            if direction == 'long':
                favorable = candle_high - entry_price
                adverse = entry_price - candle_low
            else:  # short
                favorable = entry_price - candle_low
                adverse = candle_high - entry_price

            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

        # If we exited the loop without finding exit
        if exit_price is None:
            last_candle = self.df_1m.iloc[-1]
            exit_price = last_candle['close']
            exit_time = self.df_1m.index[-1]
            exit_candle_idx = len(self.df_1m) - 1
            outcome = TradeOutcome.TIMEOUT

        # Calculate P&L
        if direction == 'long':
            pnl_dollars = (exit_price - entry_price) * shares
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        else:  # short
            pnl_dollars = (entry_price - exit_price) * shares
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100

        # Calculate R-multiple
        risk_per_share = abs(entry_price - sl_price)
        if risk_per_share > 0:
            pnl_per_share = pnl_dollars / shares
            pnl_r = pnl_per_share / risk_per_share
        else:
            pnl_r = 0.0

        # Calculate duration
        duration = exit_time - entry_time if exit_time else None
        duration_minutes = int(duration.total_seconds() / 60) if duration else None

        # Calculate new capital after trade
        capital_after = current_capital + pnl_dollars

        return TradeResult(
            signal_index=signal_index,
            entry_direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            take_profit_price=tp_price,
            stop_loss_price=sl_price,
            exit_price=exit_price,
            exit_time=exit_time,
            outcome=outcome,
            capital_before=current_capital,
            position_size_usd=position_size,
            shares_traded=shares,
            pnl_dollars=pnl_dollars,
            pnl_percent=pnl_percent,
            pnl_r_multiple=pnl_r,
            capital_after=capital_after,
            duration=duration,
            duration_minutes=duration_minutes,
            max_adverse_excursion=max_adverse,
            max_favorable_excursion=max_favorable,
            gap_exit=gap_exit,
            exit_candle_idx=exit_candle_idx,
            total_candles_in_trade=candles_scanned
        )

    def _find_start_index(self, entry_time: datetime) -> Optional[int]:
        """Find the index in df_1m at or after entry_time."""
        if entry_time in self.df_1m.index:
            return self.df_1m.index.get_loc(entry_time)

        # Find nearest candle at or after entry
        mask = self.df_1m.index >= entry_time
        if not mask.any():
            return None

        start_time = self.df_1m.index[mask][0]
        return self.df_1m.index.get_loc(start_time)

    def _check_gap_exit(
        self,
        candle_open: float,
        tp_price: float,
        sl_price: float,
        direction: str
    ) -> Optional[TradeOutcome]:
        """Check if candle opened beyond TP or SL (gap)."""
        if direction == 'long':
            if candle_open >= tp_price:
                return TradeOutcome.WIN
            elif candle_open <= sl_price:
                return TradeOutcome.LOSS
        else:  # short
            if candle_open <= tp_price:
                return TradeOutcome.WIN
            elif candle_open >= sl_price:
                return TradeOutcome.LOSS
        return None

    def _check_candle_crosses(
        self,
        high: float,
        low: float,
        tp: float,
        sl: float,
        direction: str
    ) -> Tuple[bool, bool]:
        """Check if candle high/low crossed TP or SL."""
        if direction == 'long':
            tp_hit = high >= tp
            sl_hit = low <= sl
        else:  # short
            tp_hit = low <= tp
            sl_hit = high >= sl

        return (tp_hit, sl_hit)

    def _create_timeout_result(
        self,
        signal: TradeSignal,
        signal_index: int,
        current_capital: float,
        shares: float,
        max_favorable: float,
        max_adverse: float,
        candles_scanned: int
    ) -> TradeResult:
        """Create a timeout result when no data available."""
        return TradeResult(
            signal_index=signal_index,
            entry_direction=signal.entry_direction,
            entry_price=signal.price_entry,
            entry_time=signal.timestamp_entry,
            take_profit_price=signal.take_profit_price,
            stop_loss_price=signal.stop_loss_price,
            exit_price=None,
            exit_time=None,
            outcome=TradeOutcome.TIMEOUT,
            capital_before=current_capital,
            position_size_usd=current_capital,
            shares_traded=shares,
            pnl_dollars=0.0,
            pnl_percent=0.0,
            pnl_r_multiple=0.0,
            capital_after=current_capital,
            duration=None,
            duration_minutes=None,
            max_adverse_excursion=max_adverse,
            max_favorable_excursion=max_favorable,
            gap_exit=False,
            exit_candle_idx=None,
            total_candles_in_trade=candles_scanned
        )

    def simulate_all(
        self,
        signals: List[TradeSignal]
    ) -> List[TradeResult]:
        """
        Simulate all trades from a list of signals with compounding.

        Capital compounds: profits from winning trades increase position size
        for subsequent trades, losses decrease it.

        Args:
            signals: List of TradeSignal objects (should be sorted by time)

        Returns:
            List of TradeResult objects
        """
        results = []
        current_capital = self.initial_capital

        # Sort signals by entry time
        sorted_signals = sorted(signals, key=lambda s: s.timestamp_entry)

        print(f"\n  Simulating {len(sorted_signals)} trades with compounding...")
        print(f"  Initial capital: ${self.initial_capital:,.2f}")
        print("-" * 60)

        for i, signal in enumerate(sorted_signals):
            # Skip signals without TP/SL
            if signal.take_profit_price is None or signal.stop_loss_price is None:
                print(f"  Trade {i+1}: SKIPPED - Missing TP or SL")
                continue

            # Simulate this trade with current capital
            result = self.simulate_trade(signal, i, current_capital)
            results.append(result)

            # Update capital for next trade (compounding)
            current_capital = result.capital_after

            # Progress logging
            outcome_symbol = {
                TradeOutcome.WIN: "WIN ",
                TradeOutcome.LOSS: "LOSS",
                TradeOutcome.TIMEOUT: "TIME"
            }

            print(
                f"  Trade {i+1}: {result.entry_direction.upper():5} "
                f"${result.entry_price:>7.2f} -> ${result.exit_price:>7.2f} "
                f"[{outcome_symbol[result.outcome]}] "
                f"P&L: ${result.pnl_dollars:>+8.2f} "
                f"Capital: ${result.capital_after:>10,.2f}"
            )

        print("-" * 60)
        print(f"  Final capital: ${current_capital:,.2f}")

        return results
