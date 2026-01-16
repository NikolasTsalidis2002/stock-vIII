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

from src.strategy.models import TradeSignal
from .models import TradeResult, TradeOutcome, ExitType


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
        bos_exit_enabled: bool = False,
        bos_exit_threshold_percent: float = 50.0,
        bos_mid_df: pd.DataFrame = None,
        df_mid: pd.DataFrame = None,
        symbol: str = 'TSLA'
    ) -> None:
        """
        Initialize trade simulator.

        Args:
            df_low: Low timeframe OHLCV DataFrame with datetime index
            initial_capital: Starting capital (default $10,000)
            max_trade_duration_hours: Max hours before timeout (default 24h)
            intraday_only: If True, exit at market close (21:59). If False, allow overnight holding.
            bos_exit_enabled: If True, exit when opposing BOS signal detected while in profit
            bos_exit_threshold_percent: Min profit as % of TP target before BOS exit allowed (default 50%)
            bos_mid_df: Mid timeframe BOS DataFrame (columns: BOS, Level, etc.)
            df_mid: Mid timeframe OHLCV DataFrame with datetime index
            symbol: Stock symbol being backtested (default 'TSLA')
        """
        self.df_low = df_low
        self.initial_capital = initial_capital
        self.max_trade_duration = timedelta(hours=max_trade_duration_hours)
        self.intraday_only = intraday_only
        self.symbol = symbol.upper()

        # BOS exit configuration
        self.bos_exit_enabled = bos_exit_enabled
        self.bos_exit_threshold_percent = bos_exit_threshold_percent
        self.bos_mid_df = bos_mid_df
        self.df_mid = df_mid
        self.low_to_mid_idx: Dict[datetime, int] = {}

        # Setup symbol-specific plots directory
        self._setup_plots_directory()

        # Build lookup: date -> last candle timestamp for that day
        self.last_candle_by_date = df_low.groupby(df_low.index.date).apply(lambda x: x.index.max()).to_dict()

        # Build 1min->5min timestamp mapping if BOS exit is enabled
        if bos_exit_enabled and bos_mid_df is not None and df_mid is not None:
            self._build_low_to_mid_mapping()

    def _setup_plots_directory(self) -> None:
        """Create symbol-specific plots folder and clean existing plots."""
        from pathlib import Path

        self.plots_dir = Path('results/trades') / self.symbol.lower()
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Delete existing plots for this symbol (clean slate for rerun)
        for png_file in self.plots_dir.glob('*.png'):
            png_file.unlink()

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

        # Reject entries at or after market close (dynamically detected from data)
        entry_date = entry_time.date()
        last_candle_of_day = self.last_candle_by_date.get(entry_date)
        if last_candle_of_day and entry_time >= last_candle_of_day:
            return self._create_timeout_result(
                signal, signal_index, current_capital, 0, 0.0, 0.0, 0
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

        # Find starting point in 1M data
        start_idx = self._find_start_index(entry_time)
        if start_idx is None:
            return self._create_timeout_result(
                signal, signal_index, current_capital, shares, 0.0, 0.0, 0
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
                signal, signal_index, current_capital, 0, 0.0, 0.0, 0
            )

        # Skip entry candle, start walk-forward from next candle
        # (entry is at candle close, so the first unrealized P&L should be from the next candle)
        start_idx += 1
        if start_idx >= len(self.df_low):
            return self._create_timeout_result(
                signal, signal_index, current_capital, shares, 0.0, 0.0, 0
            )

        # Calculate timeout boundary
        timeout_time = entry_time + self.max_trade_duration

        # Walk forward through 1M candles
        candles_scanned = 0

        for idx in range(start_idx, len(self.df_low)):
            candle = self.df_low.iloc[idx]
            candle_time = self.df_low.index[idx]
            candles_scanned += 1

            # Check timeout (24h max)
            if candle_time > timeout_time:
                exit_price = candle['close']
                exit_time = candle_time
                exit_candle_idx = idx
                exit_type = ExitType.TIMEOUT
                break

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
                candle_open, tp_price, sl_price, direction
            )
            if gap_result is not None:
                exit_price = candle_open
                exit_time = candle_time
                exit_candle_idx = idx
                exit_type = gap_result
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
                exit_type = ExitType.SL_HIT
                break
            elif tp_hit:
                exit_price = tp_price
                exit_time = candle_time
                exit_candle_idx = idx
                exit_type = ExitType.TP_HIT
                break
            elif sl_hit:
                exit_price = sl_price
                exit_time = candle_time
                exit_candle_idx = idx
                exit_type = ExitType.SL_HIT
                break

            # Check for BOS-based early exit (opposing BOS when profit >= threshold % of TP)
            if self.bos_exit_enabled:
                bos_exit_price = self._check_bos_exit(
                    candle_time=candle_time,
                    entry_price=entry_price,
                    tp_price=tp_price,
                    current_close=candle['close'],
                    shares=shares,
                    direction=direction
                )
                if bos_exit_price is not None:
                    exit_price = bos_exit_price
                    exit_time = candle_time
                    exit_candle_idx = idx
                    exit_type = ExitType.BOS_EXIT
                    break

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

    def _build_low_to_mid_mapping(self) -> None:
        """
        Build mapping from 1min candle timestamps to corresponding 5min candle indices.

        For each 1min candle, find which 5min candle it belongs to.
        A 1min candle at 09:32 belongs to the 5min candle starting at 09:30.

        Uses a timestamp-to-index lookup for O(1) mapping instead of O(n) search.
        """
        if self.df_mid is None or len(self.df_mid) == 0:
            return

        mid_timestamps = self.df_mid.index.tolist()

        # Calculate mid timeframe duration from data (in minutes)
        if len(mid_timestamps) > 1:
            mid_duration = mid_timestamps[1] - mid_timestamps[0]
            mid_minutes = int(mid_duration.total_seconds() / 60)
        else:
            mid_minutes = 5  # Default assumption

        # Build lookup: mid_timestamp -> index in bos_mid_df
        mid_time_to_idx = {ts: i for i, ts in enumerate(mid_timestamps)}

        # Build mapping for each low timeframe candle
        for low_time in self.df_low.index:
            # Calculate the floor timestamp (start of the mid timeframe candle)
            # e.g., 09:32 -> 09:30 for 5min candles
            minutes_from_midnight = low_time.hour * 60 + low_time.minute
            floored_minutes = (minutes_from_midnight // mid_minutes) * mid_minutes
            floored_time = low_time.replace(
                hour=floored_minutes // 60,
                minute=floored_minutes % 60,
                second=0,
                microsecond=0
            )

            # Look up the corresponding mid candle index
            if floored_time in mid_time_to_idx:
                self.low_to_mid_idx[low_time] = mid_time_to_idx[floored_time]

    def _check_bos_exit(
        self,
        candle_time: datetime,
        entry_price: float,
        tp_price: float,
        current_close: float,
        shares: float,
        direction: str
    ) -> Optional[float]:
        """
        Check if BOS-based exit condition is met.

        Exit conditions:
        1. Unrealized P&L >= threshold % of TP target
        2. Opposing BOS signal detected (bearish BOS for longs, bullish BOS for shorts)

        Uses BOS signals (events) not trend state.

        Args:
            candle_time: Current 1min candle timestamp
            entry_price: Original entry price
            tp_price: Take profit price
            current_close: Current candle close price
            shares: Position size in shares
            direction: 'long' or 'short'

        Returns:
            Exit price (candle close) if BOS exit triggered, None otherwise
        """
        if not self.bos_exit_enabled:
            return None

        if self.bos_mid_df is None or candle_time not in self.low_to_mid_idx:
            return None

        # Get the corresponding mid timeframe index
        mid_idx = self.low_to_mid_idx[candle_time]

        # Calculate unrealized P&L and TP target in dollars
        if direction == 'long':
            unrealized_pnl = (current_close - entry_price) * shares
            tp_dollars = (tp_price - entry_price) * shares
        else:  # short
            unrealized_pnl = (entry_price - current_close) * shares
            tp_dollars = (entry_price - tp_price) * shares

        # Check if profit meets threshold (% of TP target)
        threshold_dollars = tp_dollars * (self.bos_exit_threshold_percent / 100.0)
        if unrealized_pnl < threshold_dollars:
            return None

        # Check for opposing BOS signal (event, not trend state)
        bos_value = self.bos_mid_df['BOS'].iloc[mid_idx]

        # Skip if no BOS signal at this candle
        if pd.isna(bos_value):
            return None

        # Check for opposing BOS
        if direction == 'long':
            # Exit LONG on bearish BOS (-1)
            opposing_bos = (bos_value == -1)
        else:  # short
            # Exit SHORT on bullish BOS (+1)
            opposing_bos = (bos_value == 1)

        if opposing_bos:
            return current_close

        return None

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
            outcome=TradeOutcome.LOSS,  # No P&L = LOSS
            exit_type=ExitType.TIMEOUT,
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

            # Skip signals that overlap with previous trade (can only be in one position at a time)
            if results:
                previous_result = results[-1]
                if previous_result.exit_time and signal.timestamp_entry < previous_result.exit_time:
                    print(f"  Trade {i+1}: SKIPPED - Overlaps with previous trade")
                    continue

            # Simulate this trade with current capital
            result = self.simulate_trade(signal, i, current_capital)

            # Skip rejected trades (entry after market close)
            if result.exit_price is None:
                print(f"  Trade {i+1}: REJECTED - Entry after market close ({signal.timestamp_entry})")
                continue

            results.append(result)

            # Update capital for next trade (compounding)
            current_capital = result.capital_after

            # Progress logging
            exit_symbol = {
                ExitType.TP_HIT: "TP",
                ExitType.SL_HIT: "SL",
                ExitType.TIMEOUT: "TO",
                ExitType.BOS_EXIT: "BOS"
            }
            outcome_symbol = "WIN " if result.outcome == TradeOutcome.WIN else "LOSS"

            print(
                f"  Trade {i+1}: {result.entry_direction.upper():5} "
                f"${result.entry_price:>7.2f} -> ${result.exit_price:>7.2f} "
                f"[{outcome_symbol}|{exit_symbol[result.exit_type]}] "
                f"P&L: ${result.pnl_dollars:>+8.2f} "
                f"(Max: ${result.max_potential_profit_dollars():>7.2f} Left: ${result.profit_left_on_table():>+7.2f} TP: {result.tp_progress_percent():>5.1f}%) "
                f"Capital: ${result.capital_after:>10,.2f}"
            )

            # Plot unrealized P&L for this trade
            self.plot_unrealized_pnl(result, i+1)

        print("-" * 60)
        print(f"  Final capital: ${current_capital:,.2f}")

        return results

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
