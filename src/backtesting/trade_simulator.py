"""
Trade execution simulator using walk-forward methodology.

Walks through low timeframe candles from entry timestamp to determine
whether TP or SL was hit first.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

from indicators.smc import smc

from src.strategy.models import TradeSignal
from .models import (
    TradeResult, TradeOutcome, ExitType, SkipReason, SkippedTrade,
    RejectionReason, REJECTION_REASON_MESSAGES
)


class TradeSimulator:
    """
    Simulates trade execution by walking forward through low timeframe price data.

    Key features:
    - Uses low timeframe for precise exit detection
    - Handles price gaps (open beyond TP/SL)
    - Tracks maximum adverse/favorable excursion
    - Supports compounding position sizing
    """

    def __init__(
        self,
        df_low: pd.DataFrame,
        initial_capital: float = 10000.0,
        max_trade_duration_hours: int = 24,
        intraday_only: bool = True,
        symbol: str = 'TSLA',
        min_profit_percent: float = 0.0,
        min_rr_ratio: float = 0.0,
        trend_filter: str = None,
        bos_1h_trend_filter_enabled: bool = False,
        trailing_sl_enabled: bool = False,
        trailing_sl_step_pct: float = 0.5,
        trailing_sl_swing_length: int = 5
    ) -> None:
        """
        Initialize trade simulator.

        Args:
            df_low: Low timeframe OHLCV DataFrame with datetime index
            initial_capital: Starting capital (default $10,000)
            max_trade_duration_hours: Max hours before timeout (default 24h)
            intraday_only: If True, exit at market close (21:59). If False, allow overnight holding.
            symbol: Stock symbol being backtested (default 'TSLA')
            min_profit_percent: Minimum profit % to accept a trade (default 0.0, disabled)
            trend_filter: ARMA trend direction ('bullish', 'bearish', 'neutral', or None to disable)
            bos_1h_trend_filter_enabled: If True, filter trades to follow 1H BOS trend direction
            trailing_sl_enabled: If True, trail SL to swing levels as profit milestones are reached
            trailing_sl_step_pct: Profit % step that triggers trailing SL update
            trailing_sl_swing_length: Swing detection lookback for trailing SL
        """
        self.df_low = df_low
        self.initial_capital = initial_capital
        self.max_trade_duration = timedelta(hours=max_trade_duration_hours)
        self.intraday_only = intraday_only
        self.symbol = symbol.upper()
        self.min_profit_percent = min_profit_percent
        self.min_rr_ratio = min_rr_ratio
        self.trend_filter = trend_filter  # 'bullish', 'bearish', 'neutral', or None
        self.bos_1h_trend_filter_enabled = bos_1h_trend_filter_enabled

        # Trailing SL configuration
        self.trailing_sl_enabled = trailing_sl_enabled
        self.trailing_sl_step_pct = trailing_sl_step_pct

        # Store data date range for validation
        self.data_start = df_low.index.min()
        self.data_end = df_low.index.max()

        # Setup symbol-specific plots directory
        self._setup_plots_directory()

        # Build lookup: date -> last candle timestamp for that day
        self.last_candle_by_date = df_low.groupby(df_low.index.date).apply(lambda x: x.index.max()).to_dict()

        # Precompute swing levels on df_low for trailing SL
        self.swing_lows: List[Tuple[int, float]] = []   # (index, price)
        self.swing_highs: List[Tuple[int, float]] = []   # (index, price)
        if trailing_sl_enabled:
            self._compute_swing_levels(trailing_sl_swing_length)

    def _setup_plots_directory(self) -> None:
        """Create symbol-specific plots folder and clean existing plots."""
        from pathlib import Path

        self.plots_dir = Path('results/trades') / self.symbol.lower()
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Delete existing plots for this symbol (clean slate for rerun)
        for png_file in self.plots_dir.glob('*.png'):
            png_file.unlink()

    def validate_data_coverage(self, trade_time: datetime) -> bool:
        """
        Check if a trade timestamp falls within the data range.

        Args:
            trade_time: The entry timestamp of the trade

        Returns:
            True if data covers the trade time, False otherwise
        """
        return self.data_start <= trade_time <= self.data_end

    def get_data_range_str(self) -> str:
        """Return a human-readable string of the data date range."""
        return f"{self.data_start.strftime('%Y-%m-%d')} to {self.data_end.strftime('%Y-%m-%d')}"

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

        # Validate data coverage - critical check to prevent data mismatch bugs
        if not self.validate_data_coverage(entry_time):
            raise ValueError(
                f"DATA MISMATCH ERROR: Trade entry time {entry_time} is outside the low timeframe "
                f"data range ({self.get_data_range_str()}). "
                f"This would cause incorrect backtesting results. "
                f"Ensure all timeframe data files cover the same date range."
            )

        # Reject entries at or after market close (dynamically detected from data)
        entry_date = entry_time.date()
        last_candle_of_day = self.last_candle_by_date.get(entry_date)
        if last_candle_of_day and entry_time >= last_candle_of_day:
            return self._create_timeout_result(
                signal, signal_index, current_capital, 0, 0.0, 0.0, 0,
                rejection_reason=RejectionReason.AFTER_MARKET_CLOSE
            )

        # Calculate position size (full capital)
        position_size = current_capital
        shares = position_size / entry_price

        # Initialize tracking variables
        exit_price = None
        exit_time = None
        exit_type = ExitType.TIMEOUT  # How the trade exited
        exit_candle_idx = None
        gap_exit = False

        # Track excursions
        max_favorable = 0.0
        max_adverse = 0.0

        # Track unrealized P&L at each candle
        unrealized_pnl_series = []

        # Trailing SL state
        current_sl = sl_price
        if self.trailing_sl_enabled:
            step_count = 0
            if direction == 'long':
                next_step_threshold = entry_price * (1 + self.trailing_sl_step_pct / 100.0)
            else:
                next_step_threshold = entry_price * (1 - self.trailing_sl_step_pct / 100.0)

        # Find starting point in 1M data
        start_idx = self._find_start_index(entry_time)
        if start_idx is None:
            return self._create_timeout_result(
                signal, signal_index, current_capital, shares, 0.0, 0.0, 0,
                rejection_reason=RejectionReason.ENTRY_NOT_FOUND
            )

        # Check if entry candle already hit TP/SL (reject trade if so)
        entry_candle = self.df_low.iloc[start_idx]
        tp_hit_on_entry, sl_hit_on_entry = self._check_candle_crosses(
            entry_candle['high'],
            entry_candle['low'],
            tp_price,
            sl_price,
            direction
        )
        if tp_hit_on_entry or sl_hit_on_entry:
            # Entry candle already touched target - reject trade
            return self._create_timeout_result(
                signal, signal_index, current_capital, 0, 0.0, 0.0, 0,
                rejection_reason=RejectionReason.ENTRY_ALREADY_HIT_TP_SL
            )

        # Skip entry candle, start walk-forward from next candle
        # (entry is at candle close, so the first unrealized P&L should be from the next candle)
        start_idx += 1
        if start_idx >= len(self.df_low):
            return self._create_timeout_result(
                signal, signal_index, current_capital, shares, 0.0, 0.0, 0,
                rejection_reason=RejectionReason.NO_DATA_AFTER_ENTRY
            )

        # Calculate timeout boundary
        timeout_time = entry_time + self.max_trade_duration

        # Walk forward through 1M candles
        candles_scanned = 0

        for idx in range(start_idx, len(self.df_low)):
            candle = self.df_low.iloc[idx]
            candle_time = self.df_low.index[idx]
            candles_scanned += 1

            # # Check timeout (24h max)
            # if candle_time > timeout_time:
            #     exit_price = candle['close']
            #     exit_time = candle_time
            #     exit_candle_idx = idx
            #     exit_type = ExitType.TIMEOUT
            #     break

            candle_open = candle['open']
            candle_high = candle['high']
            candle_low = candle['low']

            # Update excursions FIRST (before any exit checks)
            # This ensures excursions are tracked even on the exit candle
            if direction == 'long':
                favorable = candle_high - entry_price
                adverse = entry_price - candle_low
                unrealized_pnl = (candle['close'] - entry_price) * shares
            else:  # short
                favorable = entry_price - candle_low
                adverse = candle_high - entry_price
                unrealized_pnl = (entry_price - candle['close']) * shares
            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

            # Record unrealized P&L for this candle
            unrealized_pnl_series.append((candle_time, unrealized_pnl))

            # Check for gap opening beyond TP/SL
            gap_result = self._check_gap_exit(
                candle_open, tp_price, current_sl, direction
            )
            if gap_result is not None:
                exit_price = candle_open
                exit_time = candle_time
                exit_candle_idx = idx
                # If gap hit trailing SL (not original), mark as TRAILING_SL
                if gap_result == ExitType.SL_HIT and current_sl != sl_price:
                    exit_type = ExitType.TRAILING_SL
                else:
                    exit_type = gap_result
                gap_exit = True
                break

            # Normal candle processing: check high/low for TP/SL
            tp_hit, sl_hit = self._check_candle_crosses(
                candle_high, candle_low, tp_price, current_sl, direction
            )

            if tp_hit and sl_hit:
                # Both TP and SL touched in same candle - assume worst case (SL hit first)
                exit_price = current_sl
                exit_time = candle_time
                exit_candle_idx = idx
                exit_type = ExitType.TRAILING_SL if current_sl != sl_price else ExitType.SL_HIT
                break
            elif tp_hit:
                exit_price = tp_price
                exit_time = candle_time
                exit_candle_idx = idx
                exit_type = ExitType.TP_HIT
                break
            elif sl_hit:
                exit_price = current_sl
                exit_time = candle_time
                exit_candle_idx = idx
                exit_type = ExitType.TRAILING_SL if current_sl != sl_price else ExitType.SL_HIT
                break

            # Trailing SL: check if price reached next profit milestone
            if self.trailing_sl_enabled:
                threshold_hit = False
                if direction == 'long':
                    threshold_hit = candle_high >= next_step_threshold
                else:
                    threshold_hit = candle_low <= next_step_threshold

                if threshold_hit:
                    step_count += 1
                    # Find nearest swing level to move SL to
                    new_sl = self._find_trailing_sl_level(
                        idx, entry_price, current_sl, direction
                    )
                    if new_sl is not None:
                        current_sl = new_sl
                    # Calculate next threshold
                    if direction == 'long':
                        next_step_threshold = entry_price * (1 + self.trailing_sl_step_pct * (step_count + 1) / 100.0)
                    else:
                        next_step_threshold = entry_price * (1 - self.trailing_sl_step_pct * (step_count + 1) / 100.0)

            # Check for market close - exit at end of trading day
            # Only applies if intraday_only is True
            # Only check after TP/SL so we can still hit targets on the close candle
            if self.intraday_only:
                entry_date = entry_time.date()
                last_candle_of_day = self.last_candle_by_date.get(entry_date)
                if last_candle_of_day and candle_time >= last_candle_of_day:
                    exit_price = candle['close']
                    exit_time = candle_time
                    exit_candle_idx = idx
                    exit_type = ExitType.TIMEOUT
                    break

        # If we exited the loop without finding exit
        if exit_price is None:
            last_candle = self.df_low.iloc[-1]
            exit_price = last_candle['close']
            exit_time = self.df_low.index[-1]
            exit_candle_idx = len(self.df_low) - 1
            exit_type = ExitType.TIMEOUT

        # Calculate P&L
        if direction == 'long':
            pnl_dollars = (exit_price - entry_price) * shares
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        else:  # short
            pnl_dollars = (entry_price - exit_price) * shares
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100

        # Determine outcome based on P&L (not exit type)
        outcome = TradeOutcome.WIN if pnl_dollars > 0 else TradeOutcome.LOSS

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
            exit_type=exit_type,
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
            entry_candle_idx=start_idx,
            exit_candle_idx=exit_candle_idx,
            total_candles_in_trade=candles_scanned,
            unrealized_pnl_series=unrealized_pnl_series
        )

    def _find_start_index(self, entry_time: datetime) -> Optional[int]:
        """Find the index in df_1m at or after entry_time."""
        if entry_time in self.df_low.index:
            return self.df_low.index.get_loc(entry_time)

        # Find nearest candle at or after entry
        mask = self.df_low.index >= entry_time
        if not mask.any():
            return None

        start_time = self.df_low.index[mask][0]
        return self.df_low.index.get_loc(start_time)

    def _check_gap_exit(
        self,
        candle_open: float,
        tp_price: float,
        sl_price: float,
        direction: str
    ) -> Optional[ExitType]:
        """Check if candle opened beyond TP or SL (gap)."""
        if direction == 'long':
            if candle_open >= tp_price:
                return ExitType.TP_HIT
            elif candle_open <= sl_price:
                return ExitType.SL_HIT
        else:  # short
            if candle_open <= tp_price:
                return ExitType.TP_HIT
            elif candle_open >= sl_price:
                return ExitType.SL_HIT
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

    def _compute_swing_levels(self, swing_length: int) -> None:
        """Precompute swing highs and lows on df_low for trailing SL."""
        swing_hl = smc.swing_highs_lows(self.df_low, swing_length=swing_length)
        for i in range(len(swing_hl)):
            val = swing_hl['HighLow'].iloc[i]
            if val == -1:  # swing low
                self.swing_lows.append((i, self.df_low['low'].iloc[i]))
            elif val == 1:  # swing high
                self.swing_highs.append((i, self.df_low['high'].iloc[i]))

    def _find_trailing_sl_level(
        self,
        current_idx: int,
        entry_price: float,
        current_sl: float,
        direction: str
    ) -> Optional[float]:
        """
        Find the best swing level to move trailing SL to.

        For LONG: find most recent swing low above current_sl and below current price.
        For SHORT: find most recent swing high below current_sl and above current price.

        Returns new SL price with 0.1% buffer, or None if no valid level found.
        """
        buffer_pct = 0.001  # 0.1% buffer

        if direction == 'long':
            best_level = None
            for swing_idx, swing_price in self.swing_lows:
                if swing_idx >= current_idx:
                    break
                if swing_price > current_sl:
                    best_level = swing_price
            if best_level is not None:
                return best_level * (1 - buffer_pct)
        else:  # short
            best_level = None
            for swing_idx, swing_price in self.swing_highs:
                if swing_idx >= current_idx:
                    break
                if swing_price < current_sl:
                    best_level = swing_price
            if best_level is not None:
                return best_level * (1 + buffer_pct)

        return None

    def _create_timeout_result(
        self,
        signal: TradeSignal,
        signal_index: int,
        current_capital: float,
        shares: float,
        max_favorable: float,
        max_adverse: float,
        candles_scanned: int,
        rejection_reason: RejectionReason = RejectionReason.NONE
    ) -> TradeResult:
        """Create a timeout result when trade was rejected before execution.

        Args:
            rejection_reason: Specific reason why the trade was rejected
        """
        return TradeResult(
            signal_index=signal_index,
            entry_direction=signal.entry_direction,
            entry_price=signal.price_entry,
            entry_time=signal.timestamp_entry,
            take_profit_price=signal.take_profit_price,
            stop_loss_price=signal.stop_loss_price,
            exit_price=None,
            exit_time=None,
            outcome=TradeOutcome.LOSS,  # No P&L = LOSS
            exit_type=ExitType.TIMEOUT,
            rejection_reason=rejection_reason,
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
            entry_candle_idx=None,
            exit_candle_idx=None,
            total_candles_in_trade=candles_scanned,
            unrealized_pnl_series=[]
        )

    def simulate_all(
        self,
        signals: List[TradeSignal]
    ) -> Tuple[List[TradeResult], List[SkippedTrade]]:
        """
        Simulate all trades from a list of signals with compounding.

        Capital compounds: profits from winning trades increase position size
        for subsequent trades, losses decrease it.

        Args:
            signals: List of TradeSignal objects (should be sorted by time)

        Returns:
            Tuple of (List of TradeResult objects, List of SkippedTrade objects)
        """
        results = []
        skipped_trades = []
        current_capital = self.initial_capital

        # Sort signals by entry time
        sorted_signals = sorted(signals, key=lambda s: s.timestamp_entry)

        print(f"\n  Simulating {len(sorted_signals)} trades with compounding...")
        print(f"  Initial capital: ${self.initial_capital:,.2f}")
        print(f"  Low TF data range: {self.get_data_range_str()}")
        print("-" * 60)

        for i, signal in enumerate(sorted_signals):
            # Skip signals without TP/SL
            if signal.take_profit_price is None or signal.stop_loss_price is None:
                print(f"  Trade {i+1}: SKIPPED - Missing TP or SL")
                skipped_trades.append(SkippedTrade(
                    signal_entry_time=signal.timestamp_entry,
                    skip_reason=SkipReason.MISSING_TP_SL,
                    details="Take profit or stop loss not defined"
                ))
                continue

            # Skip signals below minimum profit threshold
            if self.min_profit_percent > 0:
                if signal.entry_direction == 'long':
                    expected_profit_pct = ((signal.take_profit_price - signal.price_entry) / signal.price_entry) * 100
                else:  # short
                    expected_profit_pct = ((signal.price_entry - signal.take_profit_price) / signal.price_entry) * 100

                print('signal.take_profit_price --> ', signal.take_profit_price)
                print('signal.price_entry --> ', signal.price_entry)
                print(f"  Trade {i+1} - Expected profit %: {expected_profit_pct:.2f}%")

                if expected_profit_pct < self.min_profit_percent:
                    print(f"  Trade {i+1}: SKIPPED - Profit {expected_profit_pct:.2f}% below {self.min_profit_percent}% threshold")
                    skipped_trades.append(SkippedTrade(
                        signal_entry_time=signal.timestamp_entry,
                        skip_reason=SkipReason.INSUFFICIENT_PROFIT,
                        details=f"Expected profit {expected_profit_pct:.2f}% < {self.min_profit_percent}% minimum"
                    ))
                    continue

            # Skip signals below minimum reward-to-risk ratio
            if self.min_rr_ratio > 0:
                reward = abs(signal.take_profit_price - signal.price_entry)
                risk = abs(signal.price_entry - signal.stop_loss_price)
                if risk > 0:
                    rr_ratio = reward / risk
                else:
                    rr_ratio = 0.0

                if rr_ratio < self.min_rr_ratio:
                    print(f"  Trade {i+1}: SKIPPED - R:R {rr_ratio:.2f} below {self.min_rr_ratio:.1f} threshold")
                    skipped_trades.append(SkippedTrade(
                        signal_entry_time=signal.timestamp_entry,
                        skip_reason=SkipReason.INSUFFICIENT_RR,
                        details=f"R:R ratio {rr_ratio:.2f} < {self.min_rr_ratio:.1f} minimum"
                    ))
                    continue

            # Skip signals that overlap with previous trade (can only be in one position at a time)
            if results:
                previous_result = results[-1]
                if previous_result.exit_time and signal.timestamp_entry < previous_result.exit_time:
                    print(f"  Trade {i+1}: SKIPPED - Overlaps with previous trade")
                    skipped_trades.append(SkippedTrade(
                        signal_entry_time=signal.timestamp_entry,
                        skip_reason=SkipReason.OVERLAP,
                        details=f"Previous trade exits at {previous_result.exit_time}"
                    ))
                    continue

            # Skip signals that go against the ARMA trend filter
            if self.trend_filter and self.trend_filter != 'neutral':
                signal_is_long = signal.entry_direction == 'long'
                if self.trend_filter == 'bullish' and not signal_is_long:
                    print(f"  Trade {i+1}: SKIPPED - SHORT signal rejected (trend is bullish)")
                    skipped_trades.append(SkippedTrade(
                        signal_entry_time=signal.timestamp_entry,
                        skip_reason=SkipReason.AGAINST_TREND,
                        details=f"SHORT signal rejected - ARMA trend is bullish"
                    ))
                    continue
                elif self.trend_filter == 'bearish' and signal_is_long:
                    print(f"  Trade {i+1}: SKIPPED - LONG signal rejected (trend is bearish)")
                    skipped_trades.append(SkippedTrade(
                        signal_entry_time=signal.timestamp_entry,
                        skip_reason=SkipReason.AGAINST_TREND,
                        details=f"LONG signal rejected - ARMA trend is bearish"
                    ))
                    continue

            # Skip signals that go against the 1H BOS trend direction
            if self.bos_1h_trend_filter_enabled:
                signal_is_long = signal.entry_direction == 'long'
                trend_1h = signal.trend_1h_before_sweep  # 'bullish' or 'bearish'

                if trend_1h == 'bullish' and not signal_is_long:
                    print(f"  Trade {i+1}: SKIPPED - SHORT signal rejected (1H BOS trend is bullish)")
                    skipped_trades.append(SkippedTrade(
                        signal_entry_time=signal.timestamp_entry,
                        skip_reason=SkipReason.AGAINST_TREND,
                        details=f"SHORT signal rejected - 1H BOS trend is bullish"
                    ))
                    continue
                elif trend_1h == 'bearish' and signal_is_long:
                    print(f"  Trade {i+1}: SKIPPED - LONG signal rejected (1H BOS trend is bearish)")
                    skipped_trades.append(SkippedTrade(
                        signal_entry_time=signal.timestamp_entry,
                        skip_reason=SkipReason.AGAINST_TREND,
                        details=f"LONG signal rejected - 1H BOS trend is bearish"
                    ))
                    continue

            # Simulate this trade with current capital
            result = self.simulate_trade(signal, i, current_capital)

            # Skip rejected trades (various reasons tracked by rejection_reason)
            if result.exit_price is None:
                # Get human-readable message for the rejection reason
                reason = result.rejection_reason or RejectionReason.NONE
                reason_msg = REJECTION_REASON_MESSAGES.get(reason, "Unknown rejection reason")

                # Map rejection reasons to skip reasons
                skip_reason_map = {
                    RejectionReason.AFTER_MARKET_CLOSE: SkipReason.AFTER_MARKET_CLOSE,
                    RejectionReason.ENTRY_NOT_FOUND: SkipReason.PARTIAL_SETUP,
                    RejectionReason.ENTRY_ALREADY_HIT_TP_SL: SkipReason.PARTIAL_SETUP,
                    RejectionReason.NO_DATA_AFTER_ENTRY: SkipReason.PARTIAL_SETUP,
                }
                skip_reason = skip_reason_map.get(reason, SkipReason.PARTIAL_SETUP)

                # Build detailed message based on rejection reason
                if reason == RejectionReason.AFTER_MARKET_CLOSE:
                    entry_date = signal.timestamp_entry.date()
                    market_close_time = self.last_candle_by_date.get(entry_date)
                    market_close_str = market_close_time.strftime('%H:%M') if market_close_time else "unknown"
                    details = f"Entry time {signal.timestamp_entry.strftime('%H:%M')} at/after market close ({market_close_str})"
                else:
                    details = reason_msg

                print(f"  Trade {i+1}: REJECTED - {reason_msg} ({signal.timestamp_entry})")
                skipped_trades.append(SkippedTrade(
                    signal_entry_time=signal.timestamp_entry,
                    skip_reason=skip_reason,
                    details=details
                ))
                continue

            results.append(result)

            # Update capital for next trade (compounding)
            current_capital = result.capital_after

            # Progress logging
            exit_symbol = {
                ExitType.TP_HIT: "TP",
                ExitType.SL_HIT: "SL",
                ExitType.TIMEOUT: "TO",
                ExitType.TRAILING_SL: "TSL"
            }
            outcome_symbol = "WIN " if result.outcome == TradeOutcome.WIN else "LOSS"

            entry_dt = result.entry_time.strftime('%m/%d %H:%M')
            exit_dt = result.exit_time.strftime('%m/%d %H:%M') if result.exit_time else '??'
            print(
                f"  Trade {i+1}: {entry_dt}->{exit_dt} {result.entry_direction.upper():5} "
                f"${result.entry_price:>7.2f} -> ${result.exit_price:>7.2f} "
                f"[{outcome_symbol}|{exit_symbol[result.exit_type]}] "
                f"P&L: ${result.pnl_dollars:>+8.2f} "
                f"(Max: ${result.max_potential_profit_dollars():>7.2f} Left: ${result.profit_left_on_table():>+7.2f} TP: {result.tp_progress_percent():>5.1f}%) "
                f"Capital: ${result.capital_after:>10,.2f}"
            )

            # # Plot unrealized P&L for this trade
            # self.plot_unrealized_pnl(result, i+1)

        print("-" * 60)
        print(f"  Final capital: ${current_capital:,.2f}")
        if skipped_trades:
            print(f"  Skipped trades: {len(skipped_trades)}")

        return results, skipped_trades

    def plot_unrealized_pnl(self, result: TradeResult, trade_num: int) -> None:
        """Plot unrealized P&L for a single trade."""
        if not result.unrealized_pnl_series:
            return

        times = [t.strftime('%H:%M') for t, _ in result.unrealized_pnl_series]
        pnls = [p for _, p in result.unrealized_pnl_series]

        # Calculate TP/SL in dollar terms
        shares = result.shares_traded
        if result.entry_direction == 'long':
            tp_dollars = (result.take_profit_price - result.entry_price) * shares
            sl_dollars = (result.stop_loss_price - result.entry_price) * shares
        else:
            tp_dollars = (result.entry_price - result.take_profit_price) * shares
            sl_dollars = (result.entry_price - result.stop_loss_price) * shares

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(pnls)), pnls, 'b-', linewidth=1.5, label='Unrealized P&L')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, label='Entry')
        plt.axhline(y=tp_dollars, color='green', linestyle='--', label=f'TP: ${tp_dollars:+.2f}')
        plt.axhline(y=sl_dollars, color='red', linestyle='--', label=f'SL: ${sl_dollars:+.2f}')

        # Mark final exit
        plt.scatter([len(pnls)-1], [pnls[-1]], color='orange', s=100, zorder=5, label=f'Exit: ${pnls[-1]:+.2f}')

        plt.xlabel('Time')
        plt.ylabel('Unrealized P&L ($)')
        plt.title(f'Trade {trade_num} [{self.symbol}] - {result.entry_direction.upper()} @ ${result.entry_price:.2f}')
        plt.legend()
        plt.xticks(range(0, len(times), max(1, len(times)//10)),
                   [times[i] for i in range(0, len(times), max(1, len(times)//10))], rotation=45)
        plt.tight_layout()
        save_path = self.plots_dir / f'trade_{trade_num}_pnl.png'
        plt.savefig(save_path)
        plt.close()
        print(f"    Plot saved: {save_path}")
