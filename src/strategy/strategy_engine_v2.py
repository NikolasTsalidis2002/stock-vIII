"""
Strategy Engine V2 - Equilibrium Premium/Discount Strategy.

Extends base strategy with:
- Equilibrium-based validation (Stage 3)
- Exit target calculation
- Stop loss calculation with 2:1 R/R
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from .models import StrategyState, PartialSetup, TradeSignal, SweepInfo, EquilibriumState, ExitTarget
from .timeframe_manager import TimeframeManager
from .liquidity_detector import LiquidityDetector
from .event_b_detector import EventBDetector
from .equilibrium_validator import EquilibriumValidator
from .exit_target_finder import ExitTargetFinder
from .confirmation_detector import ConfirmationDetector


class StrategyEngineV2:
    """
    Strategy Engine for Strategy II (Equilibrium-based).

    Workflow:
    1. 1H Liquidity Sweep (same as Strategy I)
    2. 5M Event B - BOS/IFVG (same as Strategy I)
    3. 5M Equilibrium Zone - Price enters premium/discount (NEW)
    4. 1M Confirmation - BOS/IFVG (same as Strategy I)
    5. Calculate exit target and stop loss (NEW)
    """

    def __init__(
        self,
        timeframe_manager: TimeframeManager,
        max_price_deviation_percent: float = 3.0
    ) -> None:
        """
        Initialize with pre-configured timeframe manager.

        Args:
            timeframe_manager: TimeframeManager with loaded data
            max_price_deviation_percent: Invalidation threshold
        """
        self._tm = timeframe_manager
        self.max_price_deviation_percent = max_price_deviation_percent

        # Reuse existing detectors from Strategy I
        self._liquidity_detector = LiquidityDetector(
            df_1h=self._tm.df_1h,
            inflexions_1h=self._tm.inflexions_1h,
            bos_1h=self._tm.bos_1h,
            max_price_deviation_percent=max_price_deviation_percent
        )

        self._event_b_detector = EventBDetector(
            df_5m=self._tm.df_5m,
            bos_5m=self._tm.bos_5m,
            fvg_5m=self._tm.fvg_5m,
            inflexions_5m=self._tm.inflexions_5m
        )

        self._confirmation_detector = ConfirmationDetector(
            df_1m=self._tm.df_1m,
            bos_1m=self._tm.bos_1m,
            fvg_1m=self._tm.fvg_1m,
            inflexions_1m=self._tm.inflexions_1m
        )

        # NEW for Strategy II: Exit target finder
        self._exit_target_finder = ExitTargetFinder(
            df_5m=self._tm.df_5m,
            bos_5m=self._tm.bos_5m,
            inflexions_5m=self._tm.inflexions_5m
        )

        # Trade signals found
        self._signals: List[TradeSignal] = []
        self._partial_setups: List[PartialSetup] = []

    def _search_for_confirmation_1m(
        self,
        start_1m: datetime,
        end_1m: datetime,
        entry_direction: str,
        time_1h_sweep: datetime,
        sweep_idx: int,
        inflexion_idx: int,
        price_1h_sweep: float,
        price_5m_event_b: float,
        exit_target: Optional[ExitTarget]
    ) -> Tuple[Optional[Tuple[datetime, str, float]], bool, bool, bool, datetime]:
        """
        Search for 1M confirmation with invalidation checks.

        Strategy II adds additional invalidation check for exit OB break.

        Returns:
            (confirmation_tuple, need_new_event_b, need_new_equilibrium, sweep_broken, last_time_scanned)
        """
        confirmation = None
        need_new_event_b = False
        need_new_equilibrium = False
        sweep_broken = False
        last_time_scanned = start_1m

        # Progressive scan through each 1M candle
        for time_1m_current in self._tm.df_1m.loc[start_1m:end_1m].index:
            last_time_scanned = time_1m_current

            # Check if we've crossed into a new 1H candle
            hours_elapsed = (time_1m_current - time_1h_sweep).total_seconds() / 3600
            current_hour_boundary = hours_elapsed if hours_elapsed % 1 == 0 else int(hours_elapsed) + 1

            if hours_elapsed == current_hour_boundary and hours_elapsed > 0:
                # We've entered a new 1H candle - check if sweep is still valid
                check_time = time_1h_sweep + timedelta(hours=current_hour_boundary)
                current_candle_idx = sweep_idx + int(current_hour_boundary) - 1

                # Check 1: Sweep disrespected?
                if self._liquidity_detector.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                    print(f"    ⚠️  Sweep broken at {check_time} during 1M scan - abandoning setup")
                    sweep_broken = True
                    break

                # Check 2: Price moved too far?
                current_1h_price = self._tm.df_1h['close'].iloc[current_candle_idx]
                if self._liquidity_detector.is_price_moved_too_far(current_1h_price, price_1h_sweep, entry_direction):
                    print(f"    ⚠️  Price moved too far during 1M scan ({current_1h_price:.2f} vs sweep {price_1h_sweep:.2f}) - abandoning setup")
                    sweep_broken = True
                    break

            # Check Event B invalidation (check 5M candles)
            current_5m_time = time_1m_current.replace(minute=(time_1m_current.minute // 5) * 5, second=0, microsecond=0)
            if current_5m_time in self._tm.df_5m.index:
                current_5m_close = self._tm.df_5m['close'].loc[current_5m_time]

                # For long: Event B broken if price closes below Event B level
                # For short: Event B broken if price closes above Event B level
                if (entry_direction == 'long' and current_5m_close < price_5m_event_b) or \
                   (entry_direction == 'short' and current_5m_close > price_5m_event_b):
                    print(f"    ⚠️  Event B broken during 1M scan at {current_5m_time} - need new Event B")
                    need_new_event_b = True
                    break

            # Strategy II: Check if price breaks exit OB (target reached before entry)
            if exit_target is not None:
                current_1m_price = self._tm.df_1m['close'].loc[time_1m_current]
                if entry_direction == 'short' and current_1m_price < exit_target.ob_bottom:
                    print(f"    ⚠️  Price broke exit OB ({current_1m_price:.2f} < {exit_target.ob_bottom:.2f}) - abandoning setup")
                    sweep_broken = True
                    break
                elif entry_direction == 'long' and current_1m_price > exit_target.ob_top:
                    print(f"    ⚠️  Price broke exit OB ({current_1m_price:.2f} > {exit_target.ob_top:.2f}) - abandoning setup")
                    sweep_broken = True
                    break

            # Look for confirmation progressively
            confirmation_candidate = self._confirmation_detector.detect_final_confirmation(
                start_1m,
                time_1m_current,
                entry_direction
            )

            if confirmation_candidate is not None:
                confirmation = confirmation_candidate
                time_1m_confirmation, confirmation_type, price_1m_confirmation = confirmation
                print(f"    ✓ Confirmation ({confirmation_type}) at {time_1m_confirmation}")
                break  # Confirmation found

        return (confirmation, need_new_event_b, need_new_equilibrium, sweep_broken, last_time_scanned)

    def scan_for_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        """
        Scan historical data for complete trade setups using Strategy II logic.

        Key differences from Strategy I:
        - Stage 3: Use EquilibriumValidator instead of ValidationDetector
        - After confirmation: Calculate exit target and stop loss
        - Additional invalidation: Check if price breaks exit OB
        - Skip trades if exit OB not found

        Args:
            max_signals: Maximum number of signals to find

        Returns:
            List of TradeSignal objects
        """
        print("🔍 Scanning for Strategy II trade signals...")
        print(f"  Using Equilibrium Premium/Discount zones")
        print(f"  Looking for up to {max_signals} complete setups\n")

        signals = []

        # Track analyzed sweep timestamps to avoid redundant processing
        analyzed_sweep_times = set()

        # Track last analysis end time to ensure temporal consistency
        last_analysis_end_time = None

        # Step 1: Detect all liquidity sweeps upfront
        all_sweeps = self._liquidity_detector.detect_all_sweeps()

        # Step 2: Process each sweep in order
        for sweep in all_sweeps:
            if len(signals) >= max_signals:
                break

            sweep_idx = sweep.candle_idx
            sweep_type = sweep.sweep_type
            swept_level = sweep.swept_level
            inflexion_idx = sweep.inflexion_idx
            time_1h_sweep = sweep.timestamp

            # Skip if we've already analyzed a sweep at this timestamp
            if time_1h_sweep in analyzed_sweep_times:
                print(f"  Skipping duplicate sweep at {time_1h_sweep}")
                continue

            # Skip if sweep time is before or at last analysis end time
            if last_analysis_end_time is not None and time_1h_sweep <= last_analysis_end_time:
                print(f"  Skipping sweep at {time_1h_sweep} (overlaps with previous analysis)")
                continue

            # Mark this timestamp as analyzed
            analyzed_sweep_times.add(time_1h_sweep)

            # Get 1H trend before sweep
            trend_1h = self._tm.get_1h_trend_at_index(sweep_idx)
            if trend_1h is None:
                continue

            price_1h_sweep = swept_level

            print(f"\n  Analyzing sweep at {time_1h_sweep} (1H trend: {trend_1h})")

            # Determine entry direction based on how liquidity was swept
            sweep_candle_close = self._tm.df_1h['close'].iloc[sweep_idx]

            if sweep_candle_close > swept_level:
                entry_direction = 'long'
            else:
                entry_direction = 'short'

            # Get market close time for this trading day
            market_close = self._tm.get_market_close_for_day(time_1h_sweep)
            market_close_cutoff = market_close - timedelta(minutes=10)

            # Skip sweeps that occur too close to market close
            if time_1h_sweep >= market_close_cutoff:
                print(f"    ⚠️  Sweep too close to market close ({market_close}) - skipping")
                continue

            # Strategy II: Find exit OB upfront
            exit_target = self._exit_target_finder.find_exit_order_block(time_1h_sweep, entry_direction)

            if exit_target is None:
                print(f"    ⚠️  No exit Order Block found - skipping (Strategy II requirement)")
                self._partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_1h_sweep,
                    timestamp_5m_event_b=None,
                    timestamp_5m_validation=None,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_1h_sweep,
                    price_5m_event_b=None,
                    price_5m_validation=None,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_1h,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=None,
                    condition_validation=None,
                    condition_confirmation=None,
                    conditions_met=1,
                    failure_reason="No exit Order Block found",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_1h) - 1, sweep_idx + 5)),
                    indices_5m=None,
                    indices_1m=None,
                    inflexion_idx=inflexion_idx
                ))
                continue

            print(f"    ✓ Exit OB found: top={exit_target.ob_top:.2f}, bottom={exit_target.ob_bottom:.2f}, TP={exit_target.take_profit:.2f}")

            # Step 2: Progressive 5M scan for Event B → Equilibrium
            start_5m = time_1h_sweep
            end_5m = market_close_cutoff

            # State tracking
            event_b = None
            equilibrium_validation = None
            equilibrium_state = None
            sweep_broken = False
            price_5m_event_b = None

            # Progressive scan through each 5M candle
            for time_5m_current in self._tm.df_5m.loc[start_5m:end_5m].index:
                # Check if we've crossed into a new 1H candle
                hours_elapsed = (time_5m_current - time_1h_sweep).total_seconds() / 3600
                current_hour_boundary = hours_elapsed if hours_elapsed % 1 == 0 else int(hours_elapsed) + 1

                if hours_elapsed == current_hour_boundary and hours_elapsed > 0:
                    check_time = time_1h_sweep + timedelta(hours=current_hour_boundary)
                    current_candle_idx = sweep_idx + int(current_hour_boundary) - 1

                    if self._liquidity_detector.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                        print(f"    ⚠️  Sweep broken - abandoning setup")
                        sweep_broken = True
                        break

                    current_1h_price = self._tm.df_1h['close'].iloc[current_candle_idx]
                    if self._liquidity_detector.is_price_moved_too_far(current_1h_price, price_1h_sweep, entry_direction):
                        print(f"    ⚠️  Price moved too far - abandoning setup")
                        sweep_broken = True
                        break

                # Check Event B invalidation
                if event_b is not None and price_5m_event_b is not None:
                    current_5m_close = self._tm.df_5m['close'].loc[time_5m_current]

                    if (entry_direction == 'long' and current_5m_close < price_5m_event_b) or \
                       (entry_direction == 'short' and current_5m_close > price_5m_event_b):
                        print(f"    ⚠️  Event B broken at {time_5m_current} - resetting")
                        event_b = None
                        equilibrium_validation = None
                        equilibrium_state = None
                        start_5m = time_5m_current
                        continue

                # State 1: Looking for Event B
                if event_b is None:
                    event_b_candidate = self._event_b_detector.search_for_event_b(start_5m, time_5m_current, entry_direction)

                    if event_b_candidate is not None:
                        event_b = event_b_candidate
                        time_5m_event_b, event_b_type, price_5m_event_b = event_b
                        print(f"    ✓ Event B ({event_b_type}) at {time_5m_event_b}")

                # State 2: Event B found, looking for Equilibrium zone entry
                elif equilibrium_validation is None:
                    # Create fresh equilibrium validator for this scan
                    equilibrium_validator = EquilibriumValidator(
                        df_5m=self._tm.df_5m,
                        swept_level=swept_level,
                        entry_direction=entry_direction
                    )

                    equilibrium_candidate = equilibrium_validator.validate_equilibrium_zone(
                        time_5m_event_b,
                        time_5m_current
                    )

                    if equilibrium_candidate is not None:
                        time_5m_validation, validation_type, price_5m_validation, equilibrium_state = equilibrium_candidate
                        equilibrium_validation = (time_5m_validation, validation_type, price_5m_validation)
                        print(f"    ✓ Equilibrium zone entered at {time_5m_validation} (eq={equilibrium_state.equilibrium:.2f})")
                        break  # Both Event B and Equilibrium found - move to next step

            # Handle cases where conditions weren't met
            if sweep_broken:
                last_analysis_end_time = time_5m_current if 'time_5m_current' in dir() else end_5m
                continue

            if event_b is None:
                print(f"    ✗ No Event B found on 5M")
                self._partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_1h_sweep,
                    timestamp_5m_event_b=None,
                    timestamp_5m_validation=None,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_1h_sweep,
                    price_5m_event_b=None,
                    price_5m_validation=None,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_1h,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=None,
                    condition_validation=None,
                    condition_confirmation=None,
                    conditions_met=1,
                    failure_reason="No Event B on 5M",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_1h) - 1, sweep_idx + 5)),
                    indices_5m=None,
                    indices_1m=None,
                    inflexion_idx=inflexion_idx
                ))
                last_analysis_end_time = end_5m
                continue

            if equilibrium_validation is None:
                print(f"    ✗ No equilibrium zone entry found")
                time_5m_event_b, event_b_type, price_5m_event_b = event_b
                self._partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_1h_sweep,
                    timestamp_5m_event_b=time_5m_event_b,
                    timestamp_5m_validation=None,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_1h_sweep,
                    price_5m_event_b=price_5m_event_b,
                    price_5m_validation=None,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_1h,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=event_b_type,
                    condition_validation=None,
                    condition_confirmation=None,
                    conditions_met=2,
                    failure_reason="No equilibrium zone entry",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_1h) - 1, sweep_idx + 5)),
                    indices_5m=(self._tm.df_5m.index.get_loc(time_5m_event_b), self._tm.df_5m.index.get_loc(time_5m_event_b) + 20),
                    indices_1m=None,
                    inflexion_idx=inflexion_idx
                ))
                last_analysis_end_time = end_5m
                continue

            # Unpack results (both Event B and Equilibrium found)
            time_5m_event_b, event_b_type, price_5m_event_b = event_b
            time_5m_validation, validation_type, price_5m_validation = equilibrium_validation

            # Step 4: Final confirmation on 1M
            start_1m = time_5m_validation

            confirmation, need_new_event_b, need_new_equilibrium, sweep_broken, last_time_scanned = self._search_for_confirmation_1m(
                start_1m,
                market_close_cutoff,
                entry_direction,
                time_1h_sweep,
                sweep_idx,
                inflexion_idx,
                price_1h_sweep,
                price_5m_event_b,
                exit_target
            )

            # Unpack confirmation if found
            if confirmation is not None:
                time_1m_confirmation, confirmation_type, price_1m_confirmation = confirmation

            # Handle sweep broken during 1M scan
            if sweep_broken:
                last_analysis_end_time = last_time_scanned
                continue

            # Handle Event B or Equilibrium broken during 1M scan
            if need_new_event_b or need_new_equilibrium:
                print(f"    ⚠️  Event B/Equilibrium broken during 1M scan - abandoning for now")
                last_analysis_end_time = last_time_scanned
                continue

            # Handle no confirmation found
            if confirmation is None:
                print(f"    ✗ No 1M confirmation found")
                self._partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_1h_sweep,
                    timestamp_5m_event_b=time_5m_event_b,
                    timestamp_5m_validation=time_5m_validation,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_1h_sweep,
                    price_5m_event_b=price_5m_event_b,
                    price_5m_validation=price_5m_validation,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_1h,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=event_b_type,
                    condition_validation=validation_type,
                    condition_confirmation=None,
                    conditions_met=3,
                    failure_reason="No 1M confirmation",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_1h) - 1, sweep_idx + 5)),
                    indices_5m=(self._tm.df_5m.index.get_loc(time_5m_event_b), self._tm.df_5m.index.get_loc(time_5m_validation)),
                    indices_1m=(self._tm.df_1m.index.get_loc(start_1m), min(self._tm.df_1m.index.get_loc(start_1m) + 30, len(self._tm.df_1m) - 1)),
                    inflexion_idx=inflexion_idx
                ))
                last_analysis_end_time = last_time_scanned
                continue

            # All conditions met - generate Strategy II signal with TP/SL
            signal = TradeSignal(
                timestamp_1h_sweep=time_1h_sweep,
                timestamp_5m_event_b=time_5m_event_b,
                timestamp_5m_validation=time_5m_validation,
                timestamp_1m_confirmation=time_1m_confirmation,
                timestamp_entry=time_1m_confirmation,
                trend_1h_before_sweep=trend_1h,
                entry_direction=entry_direction,
                price_1h_sweep=price_1h_sweep,
                price_5m_event_b=price_5m_event_b,
                price_5m_validation=price_5m_validation,
                price_1m_confirmation=price_1m_confirmation,
                price_entry=price_1m_confirmation,
                condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                condition_event_b=event_b_type,
                condition_validation=validation_type,
                condition_confirmation=confirmation_type,
                indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_1h) - 1, sweep_idx + 5)),
                indices_5m=(
                    self._tm.df_5m.index.get_loc(time_5m_event_b),
                    self._tm.df_5m.index.get_loc(time_5m_validation)
                ),
                indices_1m=(
                    self._tm.df_1m.index.get_loc(start_1m),
                    self._tm.df_1m.index.get_loc(time_1m_confirmation)
                ),
                # Strategy II fields
                take_profit_price=exit_target.take_profit,
                exit_ob_top=exit_target.ob_top,
                exit_ob_bottom=exit_target.ob_bottom,
                exit_ob_start_idx=exit_target.ob_start_idx,
                exit_ob_end_idx=exit_target.ob_end_idx,
                equilibrium_level=equilibrium_state.equilibrium if equilibrium_state else None,
                equilibrium_fixed_level=equilibrium_state.fixed_level if equilibrium_state else None,
                equilibrium_running_extreme=equilibrium_state.running_extreme if equilibrium_state else None,
                strategy_version="II"
            )

            # Calculate stop loss based on 2:1 R/R
            signal.calculate_stop_loss()

            signals.append(signal)
            print(f"    ✅ COMPLETE SIGNAL #{len(signals)} - {entry_direction.upper()} entry")
            print(f"       Entry: {signal.price_entry:.2f}, TP: {signal.take_profit_price:.2f}, SL: {signal.stop_loss_price:.2f}")

            last_analysis_end_time = time_1m_confirmation

        print(f"\n✅ Found {len(signals)} complete Strategy II trade signals!")
        print(f"📊 Found {len(self._partial_setups)} partial setups:")
        for i, partial in enumerate(self._partial_setups, 1):
            print(f"   #{i}: {partial.conditions_met}/4 conditions - {partial.failure_reason}")

        self._signals = signals
        return signals

    @property
    def signals(self) -> List[TradeSignal]:
        """All detected trade signals."""
        return self._signals

    @property
    def partial_setups(self) -> List[PartialSetup]:
        """All partial setups (incomplete trades)."""
        return self._partial_setups
