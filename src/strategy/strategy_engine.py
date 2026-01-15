"""
Main strategy orchestrator implementing scan_for_signals state machine.

Coordinates all detector classes to find complete trade setups across
multiple timeframes.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from .models import StrategyState, PartialSetup, TradeSignal, SweepInfo
from .timeframe_manager import TimeframeManager
from .liquidity_detector import LiquidityDetector
from .event_b_detector import EventBDetector
from .validation_detector import ValidationDetector
from .confirmation_detector import ConfirmationDetector


class StrategyEngine:
    """
    Main strategy orchestrator using state machine pattern.

    Implements the multi-timeframe strategy workflow:
    1. Detect liquidity sweeps on 1H
    2. For each sweep, find Event B on 5M
    3. Validate with FVG/Demand Zone on 5M
    4. Confirm with BOS/IFVG on 1M
    5. Generate trade signals
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

        # Initialize detectors
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

        self._validation_detector = ValidationDetector(
            df_5m=self._tm.df_5m,
            fvg_5m=self._tm.fvg_5m,
            ob_5m=self._tm.ob_5m
        )

        self._confirmation_detector = ConfirmationDetector(
            df_1m=self._tm.df_1m,
            bos_1m=self._tm.bos_1m,
            fvg_1m=self._tm.fvg_1m,
            inflexions_1m=self._tm.inflexions_1m
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
        validation_fvg_index: Optional[int]
    ) -> Tuple[Optional[Tuple[datetime, str, float]], bool, bool, bool, datetime]:
        """
        Search for 1M confirmation with invalidation checks.

        Args:
            start_1m: Start of 1M search window
            end_1m: End of 1M search window
            entry_direction: 'long' or 'short'
            time_1h_sweep: Timestamp of liquidity sweep
            sweep_idx: Index of sweep in 1H data
            inflexion_idx: Index of inflexion point for sweep
            price_1h_sweep: Price at liquidity sweep
            price_5m_event_b: Price at Event B
            validation_fvg_index: FVG index for validation (None if Demand Zone)

        Returns:
            (confirmation_tuple, need_new_event_b, need_new_validation, sweep_broken, last_time_scanned)
        """
        confirmation = None
        need_new_event_b = False
        need_new_validation = False
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

            # Check Validation FVG invalidation (if validation was FVG, not Demand Zone)
            if validation_fvg_index is not None:
                fvg_respected = self._tm.fvg_5m.loc[validation_fvg_index, 'Respected']
                fvg_status_index = self._tm.fvg_5m.loc[validation_fvg_index, 'StatusIndex']

                # Check if FVG was disrespected and status determination happened by now
                if not np.isnan(fvg_status_index):
                    # Map status index to time and check if it's in the past
                    status_time = self._tm.df_5m.index[int(fvg_status_index)]
                    if status_time <= time_1m_current and fvg_respected == False:
                        print(f"    ⚠️  Validation FVG broken at {status_time} - need new validation")
                        need_new_validation = True
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

        return (confirmation, need_new_event_b, need_new_validation, sweep_broken, last_time_scanned)

    def scan_for_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        """
        Scan historical data for complete trade setups.

        Implements invalidation logic:
        - If a 1H candle closes beyond the swept level, sweep is invalidated
        - If a new sweep appears, strategy resets to the new sweep

        Args:
            max_signals: Maximum number of signals to find

        Returns:
            List of TradeSignal objects
        """
        print("🔍 Scanning for trade signals...")
        print(f"  Looking for up to {max_signals} complete setups\n")

        signals = []

        # Step 1: Detect all liquidity sweeps upfront
        all_sweeps = self._liquidity_detector.detect_all_sweeps()

        # Track analyzed sweep timestamps to avoid redundant processing
        analyzed_sweep_times = set()

        # Track last analysis end time to ensure temporal consistency
        last_analysis_end_time = None

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

            # Skip if sweep time is before or at last analysis end time (temporal consistency)
            if last_analysis_end_time is not None and time_1h_sweep <= last_analysis_end_time:
                print(f"  Skipping sweep at {time_1h_sweep} (overlaps with previous analysis ending at {last_analysis_end_time})")
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
                # Price closed above swept level → bullish sweep → enter LONG
                entry_direction = 'long'
            else:
                # Price closed below swept level → bearish sweep → enter SHORT
                entry_direction = 'short'

            # Get market close time for this trading day
            market_close = self._tm.get_market_close_for_day(time_1h_sweep)
            market_close_cutoff = market_close - timedelta(minutes=10)

            # Skip sweeps that occur too close to market close
            if time_1h_sweep >= market_close_cutoff:
                print(f"    ⚠️  Sweep too close to market close ({market_close}) - skipping")
                continue

            # Step 2 & 3: Progressive 5M scan for Event B → Validation
            start_5m = time_1h_sweep
            end_5m = market_close_cutoff

            # State tracking
            event_b = None
            validation = None
            validation_fvg_index = None
            sweep_broken = False

            # Progressive scan through each 5M candle
            for time_5m_current in self._tm.df_5m.loc[start_5m:end_5m].index:
                # Check if we've crossed into a new 1H candle
                hours_elapsed = (time_5m_current - time_1h_sweep).total_seconds() / 3600
                current_hour_boundary = hours_elapsed if hours_elapsed % 1 == 0 else int(hours_elapsed) + 1

                if hours_elapsed == current_hour_boundary and hours_elapsed > 0:
                    # We've entered a new 1H candle - check if sweep is still valid
                    check_time = time_1h_sweep + timedelta(hours=current_hour_boundary)
                    current_candle_idx = sweep_idx + int(current_hour_boundary) - 1

                    # Check 1: Sweep disrespected?
                    if self._liquidity_detector.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                        print(f"    ⚠️  Sweep broken between {check_time - timedelta(hours=1)} and {check_time} - abandoning setup")
                        sweep_broken = True
                        break

                    # Check 2: Price moved too far?
                    current_1h_price = self._tm.df_1h['close'].iloc[current_candle_idx]
                    if self._liquidity_detector.is_price_moved_too_far(current_1h_price, price_1h_sweep, entry_direction):
                        print(f"    ⚠️  Price moved too far ({current_1h_price:.2f} vs sweep {price_1h_sweep:.2f}) - abandoning setup")
                        sweep_broken = True
                        break

                # Check Event B invalidation (if Event B was found)
                if event_b is not None and price_5m_event_b is not None:
                    current_5m_close = self._tm.df_5m['close'].loc[time_5m_current]

                    # For long: Event B broken if price closes below Event B level
                    # For short: Event B broken if price closes above Event B level
                    if (entry_direction == 'long' and current_5m_close < price_5m_event_b) or \
                       (entry_direction == 'short' and current_5m_close > price_5m_event_b):
                        print(f"    ⚠️  Event B broken at {time_5m_current} (price: {current_5m_close:.2f} vs Event B: {price_5m_event_b:.2f}) - resetting")
                        event_b = None
                        validation = None
                        validation_fvg_index = None
                        start_5m = time_5m_current
                        continue

                # State 1: Looking for Event B
                if event_b is None:
                    event_b_candidate = self._event_b_detector.search_for_event_b(start_5m, time_5m_current, entry_direction)

                    if event_b_candidate is not None:
                        event_b = event_b_candidate
                        time_5m_event_b, event_b_type, price_5m_event_b = event_b
                        print(f"    ✓ Event B ({event_b_type}) at {time_5m_event_b}")

                # State 2: Event B found, looking for Validation
                elif validation is None:
                    validation_candidate = self._validation_detector.validate_fvg_or_demand_zone(
                        time_5m_event_b,
                        time_5m_current,
                        entry_direction
                    )

                    if validation_candidate is not None:
                        validation = validation_candidate
                        time_5m_validation, validation_type, price_5m_validation, validation_fvg_index = validation
                        print(f"    ✓ Validation ({validation_type}) at {time_5m_validation}")
                        break  # Both Event B and Validation found - move to next step

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

            if validation is None:
                print(f"    ✗ No FVG/Demand Zone validation found")
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
                    failure_reason="No FVG/Demand Zone validation",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_1h) - 1, sweep_idx + 5)),
                    indices_5m=(self._tm.df_5m.index.get_loc(time_5m_event_b), self._tm.df_5m.index.get_loc(time_5m_event_b) + 20),
                    indices_1m=None,
                    inflexion_idx=inflexion_idx
                ))
                last_analysis_end_time = end_5m
                continue

            # Unpack results (both Event B and Validation found)
            time_5m_event_b, event_b_type, price_5m_event_b = event_b
            time_5m_validation, validation_type, price_5m_validation, validation_fvg_index = validation

            # Step 4: Final confirmation on 1M (progressive scan)
            start_1m = time_5m_validation

            # Use helper method for 1M confirmation search
            confirmation, need_new_event_b, need_new_validation, sweep_broken, last_time_scanned = self._search_for_confirmation_1m(
                start_1m,
                market_close_cutoff,
                entry_direction,
                time_1h_sweep,
                sweep_idx,
                inflexion_idx,
                price_1h_sweep,
                price_5m_event_b,
                validation_fvg_index
            )

            # Unpack confirmation if found
            if confirmation is not None:
                time_1m_confirmation, confirmation_type, price_1m_confirmation = confirmation

            # Handle sweep broken during 1M scan
            if sweep_broken:
                last_analysis_end_time = last_time_scanned
                continue

            # Handle Event B or Validation broken during 1M scan
            if need_new_event_b or need_new_validation:
                print(f"    ⏮️  Going back to 5M scanning")

                # Reset state based on what broke
                if need_new_event_b:
                    event_b = None
                    validation = None
                    validation_fvg_index = None
                    remaining_5m_start = time_5m_validation
                else:
                    validation = None
                    validation_fvg_index = None
                    remaining_5m_start = time_5m_validation

                # Re-search for Event B or Validation
                if need_new_event_b:
                    # Check if sweep is still valid
                    start_hours_elapsed = (remaining_5m_start - time_1h_sweep).total_seconds() / 3600
                    end_hours_elapsed = (end_5m - time_1h_sweep).total_seconds() / 3600

                    first_boundary = int(start_hours_elapsed) + 1
                    last_boundary = int(end_hours_elapsed) + 1

                    for boundary_hour in range(first_boundary, last_boundary + 1):
                        check_time = time_1h_sweep + timedelta(hours=boundary_hour)

                        if check_time > end_5m:
                            break

                        current_candle_idx = sweep_idx + boundary_hour - 1

                        if self._liquidity_detector.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                            print(f"    ⚠️  Sweep broken at {check_time} during 5M re-scan - abandoning setup")
                            sweep_broken = True
                            break

                        current_1h_price = self._tm.df_1h['close'].iloc[current_candle_idx]
                        if self._liquidity_detector.is_price_moved_too_far(current_1h_price, price_1h_sweep, entry_direction):
                            print(f"    ⚠️  Price moved too far at {check_time} during 5M re-scan - abandoning setup")
                            sweep_broken = True
                            break

                    if sweep_broken:
                        continue

                    # Search for new Event B
                    event_b_resume = self._event_b_detector.search_for_event_b(remaining_5m_start, end_5m, entry_direction)

                    if event_b_resume is not None:
                        event_b = event_b_resume
                        time_5m_event_b, event_b_type, price_5m_event_b = event_b
                        print(f"    ✓ Found new Event B ({event_b_type}) at {time_5m_event_b}")

                        validation_resume = self._validation_detector.validate_fvg_or_demand_zone(time_5m_event_b, end_5m, entry_direction)

                        if validation_resume is not None:
                            validation = validation_resume
                            time_5m_validation, validation_type, price_5m_validation, validation_fvg_index = validation
                            print(f"    ✓ Found validation ({validation_type}) at {time_5m_validation}")
                        else:
                            print(f"    ✗ No validation found after new Event B")
                            continue
                    else:
                        print(f"    ✗ No new Event B found - abandoning sweep")
                        continue

                else:  # need_new_validation
                    # Check if sweep is still valid
                    start_hours_elapsed = (time_5m_event_b - time_1h_sweep).total_seconds() / 3600
                    end_hours_elapsed = (end_5m - time_1h_sweep).total_seconds() / 3600

                    first_boundary = int(start_hours_elapsed) + 1
                    last_boundary = int(end_hours_elapsed) + 1

                    for boundary_hour in range(first_boundary, last_boundary + 1):
                        check_time = time_1h_sweep + timedelta(hours=boundary_hour)

                        if check_time > end_5m:
                            break

                        current_candle_idx = sweep_idx + boundary_hour - 1

                        if self._liquidity_detector.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                            print(f"    ⚠️  Sweep broken at {check_time} during Validation re-scan - abandoning setup")
                            sweep_broken = True
                            break

                        current_1h_price = self._tm.df_1h['close'].iloc[current_candle_idx]
                        if self._liquidity_detector.is_price_moved_too_far(current_1h_price, price_1h_sweep, entry_direction):
                            print(f"    ⚠️  Price moved too far at {check_time} during Validation re-scan - abandoning setup")
                            sweep_broken = True
                            break

                    if sweep_broken:
                        continue

                    # Search for new Validation
                    validation_resume = self._validation_detector.validate_fvg_or_demand_zone(time_5m_event_b, end_5m, entry_direction)

                    if validation_resume is not None:
                        validation = validation_resume
                        time_5m_validation, validation_type, price_5m_validation, validation_fvg_index = validation
                        print(f"    ✓ Found new validation ({validation_type}) at {time_5m_validation}")
                    else:
                        print(f"    ✗ No new validation found - abandoning sweep")
                        continue

                # Try 1M confirmation again
                print(f"    ▶️  Resuming 1M scan from {time_5m_validation}")
                start_1m = time_5m_validation

                confirmation_resume, need_new_event_b_again, need_new_validation_again, sweep_broken_resume, last_time_scanned_resume = self._search_for_confirmation_1m(
                    start_1m,
                    market_close_cutoff,
                    entry_direction,
                    time_1h_sweep,
                    sweep_idx,
                    inflexion_idx,
                    price_1h_sweep,
                    price_5m_event_b,
                    validation_fvg_index
                )

                if sweep_broken_resume:
                    print(f"    ⚠️  Sweep broken during resumed 1M scan - abandoning")
                    last_analysis_end_time = last_time_scanned_resume
                    continue
                elif need_new_event_b_again or need_new_validation_again:
                    print(f"    ⚠️  Event B/Validation broken again during resumed scan - abandoning for now")
                    continue
                elif confirmation_resume is not None:
                    confirmation = confirmation_resume
                    time_1m_confirmation, confirmation_type, price_1m_confirmation = confirmation
                    print(f"    ✅ Found confirmation after reset!")
                else:
                    print(f"    ✗ No confirmation found after reset")
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

            # All conditions met - generate signal
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
                )
            )

            signals.append(signal)
            print(f"    ✅ COMPLETE SIGNAL #{len(signals)} - {entry_direction.upper()} entry")

            last_analysis_end_time = time_1m_confirmation

        print(f"\n✅ Found {len(signals)} complete trade signals!")
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
