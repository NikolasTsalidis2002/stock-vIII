"""
Strategy Engine - Equilibrium Premium/Discount Strategy.

Main strategy orchestrator implementing:
- GMM-based zone detection with Fibonacci levels (optional)
- Equilibrium-based validation (Stage 3)
- Exit target calculation
- Stop loss calculation with 2:1 R/R
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .models import StrategyState, PartialSetup, TradeSignal, SweepInfo, EquilibriumState, ExitTarget
from .timeframe_manager import TimeframeManager
from .liquidity_detector import LiquidityDetector
from .event_b_detector import EventBDetector
from .equilibrium_validator import EquilibriumValidator
from .exit_target_finder import ExitTargetFinder
from .confirmation_detector import ConfirmationDetector
from .gmm_zone_detector import GMMZoneDetector, GMMZoneInfo


class StrategyEngine:
    """
    Main strategy orchestrator using state machine pattern.

    Workflow:
    1. High TF Liquidity Sweep
    2. Mid TF Event B - BOS/IFVG
    3. Mid TF Equilibrium Zone - Price enters premium/discount
    4. Low TF Confirmation - BOS/IFVG
    5. Calculate exit target and stop loss
    """

    def __init__(
        self,
        timeframe_manager: TimeframeManager,
        use_fvg_validation: bool = True,
        use_equilibrium_validation: bool = True,
        require_fvg_in_equilibrium: bool = False,
        sweep_proximity_threshold: float = 0.0,
        abandon_on_new_sweep: bool = True,
        gmm_config: Dict = None
    ) -> None:
        """
        Initialize with pre-configured timeframe manager.

        Args:
            timeframe_manager: TimeframeManager with loaded data
            use_fvg_validation: If True, check FVG respect in validation (takes priority).
                               If False, skip FVG validation.
            use_equilibrium_validation: If True, check equilibrium zone entry.
                                       If False, skip equilibrium validation.
            require_fvg_in_equilibrium: If True (OVERRIDES other settings), require
                                       an FVG that structurally overlaps with the
                                       equilibrium zone. This is the strictest mode.
            sweep_proximity_threshold: Percentage threshold (as decimal) for near-sweep detection.
                                      If price comes within this % of the level, it counts as swept.
                                      Default 0.0 = exact touch required.
            abandon_on_new_sweep: If True (default), abandons current setup if a new liquidity
                                 sweep occurs during mid/low TF scanning. If False, continues
                                 building the current setup regardless of new sweeps.
            gmm_config: Optional dictionary with GMM zone configuration:
                        - enabled: bool (default True)
                        - lookback_candles: int (default 300)
                        - step: float (default 0.1)
                        - max_components: int (default 3)
                        - recalc_interval: int (default 50)
                        - fib_levels: list (default [1.0, 0.786, ...])
                        - premium_zone: list (default [0.786, 1.0])
                        - discount_zone: list (default [0.0, 0.236])
                        - allow_middle_zone_trades: bool (default False)
                        - take_profit_method: str ('fib' or 'order_block', default 'fib')
        """
        self._tm = timeframe_manager
        self._use_fvg_validation = use_fvg_validation
        self._use_equilibrium_validation = use_equilibrium_validation
        self._require_fvg_in_equilibrium = require_fvg_in_equilibrium
        self._abandon_on_new_sweep = abandon_on_new_sweep

        # GMM Zone Detector (optional)
        self._gmm_config = gmm_config or {}
        self._gmm_enabled = self._gmm_config.get('enabled', False)
        self._gmm_detector = GMMZoneDetector(self._gmm_config) if self._gmm_enabled else None
        self._current_gmm_zone: Optional[GMMZoneInfo] = None

        # Reuse existing detectors from base strategy
        self._liquidity_detector = LiquidityDetector(
            df_high=self._tm.df_high,
            inflexions_high=self._tm.inflexions_high,
            bos_high=self._tm.bos_high,
            proximity_threshold=sweep_proximity_threshold
        )

        self._event_b_detector = EventBDetector(
            df_mid=self._tm.df_mid,
            bos_mid=self._tm.bos_mid,
            fvg_mid=self._tm.fvg_mid,
            inflexions_mid=self._tm.inflexions_mid
        )

        self._confirmation_detector = ConfirmationDetector(
            df_low=self._tm.df_low,
            bos_low=self._tm.bos_low,
            fvg_low=self._tm.fvg_low,
            inflexions_low=self._tm.inflexions_low
        )

        # NEW for Strategy: Exit target finder
        self._exit_target_finder = ExitTargetFinder(
            df_mid=self._tm.df_mid,
            bos_mid=self._tm.bos_mid,
            inflexions_mid=self._tm.inflexions_mid,
            ob_mid=self._tm.ob_mid
        )

        # Trade signals found
        self._signals: List[TradeSignal] = []
        self._partial_setups: List[PartialSetup] = []

    def _has_new_sweep_occurred(
        self,
        all_sweeps: List[SweepInfo],
        current_sweep_time: datetime,
        check_time: datetime
    ) -> bool:
        """
        Check if a new liquidity sweep has occurred after the current sweep.

        If a new sweep occurs during Stages 2-4, the current setup should be
        abandoned and we should start fresh with the new sweep.

        Args:
            all_sweeps: List of all detected sweeps
            current_sweep_time: Timestamp of the sweep we're currently tracking
            check_time: Current processing time to check against

        Returns:
            True if a new sweep has occurred (setup should be invalidated)
        """
        for sweep in all_sweeps:
            # Check if this sweep occurred after our current sweep but before/at check_time
            if sweep.timestamp > current_sweep_time and sweep.timestamp <= check_time:
                return True
        return False

    def _search_for_confirmation_low(
        self,
        start_low: datetime,
        end_low: datetime,
        entry_direction: str,
        time_high_sweep: datetime,
        sweep_idx: int,
        inflexion_idx: int,
        exit_target: Optional[ExitTarget],
        all_sweeps: List[SweepInfo],
        equilibrium_state: Optional[EquilibriumState] = None
    ) -> Tuple[Optional[Tuple[datetime, str, float]], bool, bool, datetime, bool, Optional[datetime]]:
        """
        Search for low TF confirmation with invalidation checks.

        Strategy adds additional invalidation check for exit OB break,
        new sweep invalidation, and FVG disrespect monitoring.

        Returns:
            (confirmation_tuple, sweep_broken, new_sweep_occurred, last_time_scanned,
             fvg_invalidated, fvg_invalidation_time)
        """
        confirmation = None
        sweep_broken = False
        new_sweep_occurred = False
        last_time_scanned = start_low
        fvg_invalidated = False
        fvg_invalidation_time = None

        # Get the high TF duration for boundary checks
        high_duration = self._tm.high_duration

        # Track last checked 5M index to avoid duplicate checks
        last_checked_5m_idx = None

        # Check if we need to monitor FVG (FVG-based validation)
        tracking_fvg = (
            equilibrium_state is not None and
            equilibrium_state.validation_type == 'FVG_Respect' and
            len(equilibrium_state.tracked_fvgs) > 0
        )

        # Create local copy of tracked FVGs to manage during scan
        active_fvgs = list(equilibrium_state.tracked_fvgs) if tracking_fvg else []

        # Progressive scan through each low TF candle
        for time_low_current in self._tm.df_low.loc[start_low:end_low].index:
            last_time_scanned = time_low_current

            # Check if we've crossed into a new high TF candle
            time_elapsed = time_low_current - time_high_sweep
            candles_elapsed = time_elapsed.total_seconds() / high_duration.total_seconds()
            current_candle_boundary = int(candles_elapsed) + 1 if candles_elapsed % 1 != 0 else int(candles_elapsed)

            if candles_elapsed == current_candle_boundary and candles_elapsed > 0:
                # We've entered a new high TF candle - check if sweep is still valid
                check_time = time_high_sweep + timedelta(seconds=high_duration.total_seconds() * current_candle_boundary)
                current_candle_idx = sweep_idx + int(current_candle_boundary) - 1

                # Check: Sweep disrespected?
                if self._liquidity_detector.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                    print(f"    ⚠️  Sweep broken at {check_time} during low TF scan - abandoning setup")
                    sweep_broken = True
                    break

            # Check: FVG disrespected? (only if tracking FVG-based validation)
            if tracking_fvg and len(active_fvgs) > 0:
                # Map current 1M time to 5M candle index
                # Find the 5M candle that contains this 1M time
                mask_5m = self._tm.df_mid.index <= time_low_current
                if mask_5m.any():
                    current_5m_time = self._tm.df_mid.index[mask_5m][-1]
                    current_5m_idx = self._tm.df_mid.index.get_loc(current_5m_time)

                    # Only check if we've moved to a new 5M candle
                    if last_checked_5m_idx is None or current_5m_idx > last_checked_5m_idx:
                        last_checked_5m_idx = current_5m_idx

                        # Check which FVGs are invalidated at this 5M candle
                        invalidated_fvgs = []
                        for fvg in active_fvgs:
                            fvg_row = self._tm.fvg_mid.iloc[fvg.fvg_index]
                            status_idx = fvg_row['StatusIndex']

                            if not np.isnan(status_idx) and int(status_idx) == current_5m_idx and fvg_row['Respected'] == False:
                                print(f"    [DEBUG] FVG at idx {fvg.fvg_index} invalidated at 5M idx {current_5m_idx}")
                                invalidated_fvgs.append(fvg)

                        # Remove invalidated FVGs from active list
                        for inv_fvg in invalidated_fvgs:
                            active_fvgs.remove(inv_fvg)

                        # If ALL FVGs invalidated, trigger re-search
                        if len(active_fvgs) == 0 and len(invalidated_fvgs) > 0:
                            print(f"    ⚠️  All FVGs disrespected at {current_5m_time} - re-searching for validation")
                            fvg_invalidated = True
                            fvg_invalidation_time = current_5m_time
                            break

            # Check: New sweep occurred? (invalidates current setup)
            if self._abandon_on_new_sweep and self._has_new_sweep_occurred(all_sweeps, time_high_sweep, time_low_current):
                print(f"    ⚠️  New sweep occurred during low TF scan - abandoning current setup")
                new_sweep_occurred = True
                break

            # Strategy: Check if price breaks exit OB (target reached before entry)
            if exit_target is not None:
                current_low_price = self._tm.df_low['close'].loc[time_low_current]
                if entry_direction == 'short' and current_low_price < exit_target.ob_bottom:
                    print(f"    ⚠️  Price broke exit OB ({current_low_price:.2f} < {exit_target.ob_bottom:.2f}) - abandoning setup")
                    sweep_broken = True
                    break
                elif entry_direction == 'long' and current_low_price > exit_target.ob_top:
                    print(f"    ⚠️  Price broke exit OB ({current_low_price:.2f} > {exit_target.ob_top:.2f}) - abandoning setup")
                    sweep_broken = True
                    break

            # Look for confirmation progressively
            confirmation_candidate = self._confirmation_detector.detect_confirmation_at_candle(
                time_low_current,
                entry_direction
            )

            if confirmation_candidate is not None:
                confirmation = confirmation_candidate
                time_low_confirmation, confirmation_type, price_low_confirmation = confirmation
                print(f"    ✓ Confirmation ({confirmation_type}) at {time_low_confirmation}")
                break  # Confirmation found

        return (confirmation, sweep_broken, new_sweep_occurred, last_time_scanned, fvg_invalidated, fvg_invalidation_time)

    def scan_for_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        """
        Scan historical data for complete trade setups using Strategy logic.

        Key differences from base strategy:
        - GMM-based zone detection with Fibonacci levels (optional)
        - Stage 3: Use EquilibriumValidator instead of ValidationDetector
        - After confirmation: Calculate exit target and stop loss
        - Additional invalidation: Check if price breaks exit OB
        - Skip trades if exit OB not found

        Args:
            max_signals: Maximum number of signals to find

        Returns:
            List of TradeSignal objects
        """
        print("🔍 Scanning for Strategy trade signals...")
        print(f"  Using Equilibrium Premium/Discount zones")
        if self._gmm_enabled:
            print(f"  GMM Zone Detection: ENABLED")
            print(f"    - Lookback: {self._gmm_config.get('lookback_candles', 300)} candles")
            print(f"    - Premium zone: >= {self._gmm_config.get('premium_zone', [0.786, 1.0])[0]} fib")
            print(f"    - Discount zone: <= {self._gmm_config.get('discount_zone', [0.0, 0.236])[1]} fib")
            print(f"    - TP method: {self._gmm_config.get('take_profit_method', 'fib')}")
        print(f"  Looking for up to {max_signals} complete setups\n")

        signals = []

        # Track analyzed sweep timestamps to avoid redundant processing
        analyzed_sweep_times = set()

        # Track last analysis end time to ensure temporal consistency
        last_analysis_end_time = None

        # Get the high TF duration for time calculations
        high_duration = self._tm.high_duration

        # GMM Zone Detection Setup (if enabled)
        # Use ORIGINAL (unfiltered) data to access historical candles for lookback
        # Select dataframe based on gmm_zones.timeframe config
        # Use case-insensitive comparison since config may have different casing
        gmm_timeframe = self._gmm_config.get('timeframe', '15min').lower()
        tf_high = self._tm.timeframe_config.get('high', '').lower()
        tf_mid = self._tm.timeframe_config.get('mid', '').lower()
        if gmm_timeframe == tf_high:
            gmm_df = self._tm.df_high_original
        elif gmm_timeframe == tf_mid:
            gmm_df = self._tm.df_mid_original
        else:
            # Fallback to mid if timeframe doesn't match
            gmm_df = self._tm.df_mid_original
            print(f"[GMM] Warning: timeframe '{gmm_timeframe}' doesn't match high ({tf_high}) or mid ({tf_mid}), using mid")

        if self._gmm_enabled and self._gmm_detector:
            # Calculate fixed start index for growing window
            # first_overlap_index = index of tradable_start in GMM timeframe
            tradable_start = self._tm.tradable_start
            gmm_indices = gmm_df.index.get_indexer([tradable_start], method='bfill')
            first_overlap_index = gmm_indices[0] if gmm_indices[0] != -1 else 0

            # Fixed start = first_overlap_index - lookback_candles
            fixed_start_index = max(0, first_overlap_index - self._gmm_detector.lookback_candles)
            self._gmm_detector.set_fixed_start_index(fixed_start_index)

            print(f"[GMM] Tradable start: {tradable_start}")
            print(f"[GMM] First overlap index in GMM TF: {first_overlap_index}")
            print(f"[GMM] Fixed start index: {fixed_start_index}")
            print(f"[GMM] Initial window: {first_overlap_index - fixed_start_index} candles")

        # Step 1: Detect all liquidity sweeps upfront (no GMM filter - we filter per sweep)
        # Get tradable_start from timeframe manager - only trade sweeps within overlap period
        tradable_start = self._tm.tradable_start
        all_sweeps = self._liquidity_detector.detect_all_sweeps(
            sweep_type_filter=None,  # No pre-filtering - GMM filter applied per sweep
            tradable_start=tradable_start
        )

        # Step 2: Process each sweep in order
        for sweep in all_sweeps:
            if len(signals) >= max_signals:
                break

            sweep_idx = sweep.candle_idx
            sweep_type = sweep.sweep_type
            swept_level = sweep.swept_level
            inflexion_idx = sweep.inflexion_idx
            time_high_sweep = sweep.timestamp

            # Initialize GMM zone for this sweep (will be set if GMM enabled)
            sweep_gmm_zone = None

            # Skip if we've already analyzed a sweep at this timestamp
            if time_high_sweep in analyzed_sweep_times:
                print(f"  Skipping duplicate sweep at {time_high_sweep}")
                continue

            # Skip if sweep time is before or at last analysis end time
            if last_analysis_end_time is not None and time_high_sweep <= last_analysis_end_time:
                print(f"  Skipping sweep at {time_high_sweep} (overlaps with previous analysis)")
                continue

            # Mark this timestamp as analyzed
            analyzed_sweep_times.add(time_high_sweep)

            # Get high TF trend before sweep
            trend_high = self._tm.get_high_trend_at_index(sweep_idx)
            if trend_high is None:
                continue

            price_high_sweep = swept_level

            print(f"\n  Analyzing sweep at {time_high_sweep} (high TF trend: {trend_high})")

            # Determine entry direction based on how liquidity was swept
            if sweep.is_dual_sweep:
                # Dual sweep: direction already determined by candle color in detector
                entry_direction = sweep.sweep_type.replace('dual_', '')  # 'dual_short' → 'short'
                print(f"    🔄 Dual liquidity sweep detected")
                print(f"       High level: ${sweep.dual_high_level:.2f}, Low level: ${sweep.dual_low_level:.2f}")
                print(f"       Direction from candle color: {entry_direction.upper()}")
            else:
                # Normal single sweep logic
                sweep_candle_close = self._tm.df_high['close'].iloc[sweep_idx]

                if sweep_candle_close > swept_level:
                    entry_direction = 'long'
                else:
                    entry_direction = 'short'

            # GMM Zone Detection: Calculate zones AT THIS SWEEP's position (growing window)
            if self._gmm_enabled and self._gmm_detector:
                # Find the sweep's index in the GMM timeframe (mid TF)
                # Use the high TF sweep time to find the corresponding GMM TF candle
                gmm_indices = gmm_df.index.get_indexer([time_high_sweep], method='ffill')
                gmm_sweep_idx = gmm_indices[0] if gmm_indices[0] != -1 else len(gmm_df) - 1

                # Calculate GMM zones at this sweep's position
                sweep_gmm_zone = self._gmm_detector.detect_zones(
                    gmm_df,
                    gmm_sweep_idx,
                    swept_level  # Use the swept price level
                )

                if sweep_gmm_zone is None:
                    print(f"    ⚠️  GMM zone detection failed at sweep - skipping")
                    continue

                # Store current GMM zone for this sweep (for visualization)
                self._current_gmm_zone = sweep_gmm_zone

                # Set fib prices for ExitTargetFinder
                use_fib_tp = self._gmm_config.get('take_profit_method', 'fib') == 'fib'
                self._exit_target_finder.set_fib_prices(
                    sweep_gmm_zone.fib_prices,
                    use_fib_tp=use_fib_tp
                )

                gmm_bias = sweep_gmm_zone.entry_bias

                # Check: Is price in middle zone?
                if gmm_bias == 'skip':
                    print(f"\n    [GMM DEBUG] === Skipping Sweep (Middle Zone) ===")
                    print(f"      Sweep at index {gmm_sweep_idx}, price ${swept_level:.2f}")
                    print(f"      Fib position: {sweep_gmm_zone.current_fib_position:.3f} ({sweep_gmm_zone.current_fib_position * 100:.1f}%)")
                    print(f"      Zone range: ${sweep_gmm_zone.zone_bottom:.2f} - ${sweep_gmm_zone.zone_top:.2f}")
                    print(f"      Price in middle zone - skipping (allow_middle=False)")
                    continue

                # Check: Does entry direction match zone bias?
                if gmm_bias not in ['neutral', 'skip'] and entry_direction != gmm_bias:
                    print(f"\n    [GMM DEBUG] === Skipping Sweep (Direction Mismatch) ===")
                    print(f"      Sweep type: {sweep_type}")
                    print(f"      Swept level: ${swept_level:.2f}")
                    print(f"      Entry direction from sweep: {entry_direction}")
                    print(f"      GMM entry bias: {gmm_bias}")
                    print(f"      Fib position: {sweep_gmm_zone.current_fib_position:.3f} ({sweep_gmm_zone.current_fib_position * 100:.1f}%)")
                    print(f"      Zone range: ${sweep_gmm_zone.zone_bottom:.2f} - ${sweep_gmm_zone.zone_top:.2f}")
                    if gmm_bias == 'short':
                        print(f"      Reason: Price in PREMIUM zone (fib >= {self._gmm_config.get('premium_zone', [0.786, 1.0])[0] * 100:.1f}%), only SHORT entries allowed")
                    elif gmm_bias == 'long':
                        print(f"      Reason: Price in DISCOUNT zone (fib <= {self._gmm_config.get('discount_zone', [0.0, 0.236])[1] * 100:.1f}%), only LONG entries allowed")
                    print(f"    ⚠️  Entry direction {entry_direction} doesn't match GMM bias {gmm_bias} - skipping")
                    continue

                # GMM filter passed
                print(f"    ✓ GMM filter passed: {entry_direction.upper()} in {gmm_bias.upper()} zone")

            # Get market close time for this trading day
            market_close = self._tm.get_market_close_for_day(time_high_sweep)
            market_close_cutoff = market_close - timedelta(minutes=10)

            # Calculate when the liquidity hour closes (needed for look-ahead bias fix)
            time_high_end = time_high_sweep + high_duration

            # Skip sweeps where the scan can't start (hour closes at/after market close)
            if time_high_end >= market_close_cutoff:
                print(f"    ⚠️  Liquidity hour closes too close to market close ({market_close}) - skipping")
                continue

            # Exit target will be found after mid TF BOS (Event B) is detected
            exit_target = None

            # Step 2: Progressive mid TF scan for Event B → Equilibrium
            # Find the actual sweep point within the high TF candle (historical lookup - valid)
            # For LONG: find mid TF candle with min low (actual sweep point)
            # For SHORT: find mid TF candle with max high (actual sweep point)
            mask_mid_in_high = (self._tm.df_mid.index >= time_high_sweep) & (self._tm.df_mid.index < time_high_end)
            candles_mid_in_high = self._tm.df_mid.loc[mask_mid_in_high]

            # Find actual sweep point (historical lookup - this is valid)
            if len(candles_mid_in_high) > 0:
                if entry_direction == 'long':
                    # For LONG: swept a LOW, find the mid TF candle with min low
                    actual_sweep_time = candles_mid_in_high['low'].idxmin()
                    actual_sweep_price = self._tm.df_mid.loc[actual_sweep_time, 'low']
                else:
                    # For SHORT: swept a HIGH, find the mid TF candle with max high
                    actual_sweep_time = candles_mid_in_high['high'].idxmax()
                    actual_sweep_price = self._tm.df_mid.loc[actual_sweep_time, 'high']
                print(f"    Actual sweep point: {actual_sweep_time} (price: {actual_sweep_price:.2f})")
            else:
                actual_sweep_time = time_high_sweep  # Fallback
                actual_sweep_price = swept_level  # Fallback to 1H level

            # FIX LOOK-AHEAD BIAS: Calculate when we can actually start scanning
            # We can only know about the sweep AFTER the 1H candle closes (time_high_end)
            scan_start_candidates = self._tm.df_mid.index[self._tm.df_mid.index >= time_high_end]
            if len(scan_start_candidates) == 0:
                print(f"    No 5M candles after liquidity hour closes - skipping")
                continue
            scan_start_time = scan_start_candidates[0]
            print(f"    Scan for Event B starts at: {scan_start_time} (after 1H close at {time_high_end})")

            end_mid = market_close_cutoff

            # State tracking
            event_b = None
            equilibrium_validation = None
            equilibrium_state = None
            equilibrium_validator = None  # Created once when Event B found
            sweep_broken = False
            new_sweep_occurred = False
            price_mid_event_b = None

            # Progressive scan through each mid TF candle (starting AFTER hour closes)
            for time_mid_current in self._tm.df_mid.loc[scan_start_time:end_mid].index:
                # Check if we've crossed into a new high TF candle
                time_elapsed = time_mid_current - time_high_sweep
                candles_elapsed = time_elapsed.total_seconds() / high_duration.total_seconds()
                current_candle_boundary = int(candles_elapsed) + 1 if candles_elapsed % 1 != 0 else int(candles_elapsed)

                if candles_elapsed == current_candle_boundary and candles_elapsed > 0:
                    check_time = time_high_sweep + timedelta(seconds=high_duration.total_seconds() * current_candle_boundary)
                    current_candle_idx = sweep_idx + int(current_candle_boundary) - 1

                    if self._liquidity_detector.is_sweep_broken_at_time(inflexion_idx, check_time, current_candle_idx):
                        print(f"    ⚠️  Sweep broken - abandoning setup")
                        sweep_broken = True
                        break

                # Check: New sweep occurred? (invalidates current setup)
                if self._abandon_on_new_sweep and self._has_new_sweep_occurred(all_sweeps, time_high_sweep, time_mid_current):
                    print(f"    ⚠️  New sweep occurred during mid TF scan - abandoning current setup")
                    new_sweep_occurred = True
                    break

                # State 1: Looking for Event B
                if event_b is None:
                    event_b_candidate = self._event_b_detector.search_for_event_b_at_candle(time_mid_current, entry_direction)

                    if event_b_candidate is not None:
                        event_b = event_b_candidate
                        time_mid_event_b, event_b_type, price_mid_event_b = event_b
                        print(f"    ✓ Event B ({event_b_type}) at {time_mid_event_b}")

                        # Find exit OB using the mid TF BOS timestamp (not sweep time)
                        exit_target = self._exit_target_finder.find_exit_order_block(
                            time_mid_event_b,
                            entry_direction
                        )

                        if exit_target is None:
                            print(f"    ⚠️  No exit Order Block found after Event B - abandoning setup")
                            break  # Will be handled by partial setup creation below

                        print(f"    ✓ Exit OB found: top={exit_target.ob_top:.2f}, bottom={exit_target.ob_bottom:.2f}, TP={exit_target.take_profit:.2f}")

                        # Create validator ONCE with pre-calculated inflexions (O(n) optimization)
                        # Pass sweep_time (actual 5M sweep point) for FVG filtering
                        # Use actual_sweep_price (5M sweep price) instead of swept_level (1H inflexion)
                        equilibrium_validator = EquilibriumValidator(
                            df_mid=self._tm.df_mid,
                            swept_level=actual_sweep_price,  # Use actual 5M sweep price, not 1H inflexion
                            entry_direction=entry_direction,
                            fvg_mid=self._tm.fvg_mid,
                            sweep_time=actual_sweep_time,  # Use actual 5M sweep point for FVG filtering
                            inflexions_mid=self._tm.inflexions_mid,
                            use_fvg_validation=self._use_fvg_validation,
                            use_equilibrium_validation=self._use_equilibrium_validation,
                            require_fvg_in_equilibrium=self._require_fvg_in_equilibrium
                        )
                        print(f"    [DEBUG] EquilibriumValidator created with sweep_time={actual_sweep_time}, actual_sweep_price={actual_sweep_price:.2f}")
                        self._tm.fvg_mid.to_csv("fvg_mid_debug.csv")

                # State 2: Event B found, looking for FVG respect OR Equilibrium zone entry
                elif equilibrium_validation is None and equilibrium_validator is not None:
                    # Use single-candle validation method (O(n) optimization)
                    validation_candidate = equilibrium_validator.validate_at_candle(time_mid_current)

                    if validation_candidate is not None:
                        time_mid_validation, validation_type, price_mid_validation, equilibrium_state = validation_candidate
                        equilibrium_validation = (time_mid_validation, validation_type, price_mid_validation)

                        # Log based on validation type
                        if validation_type == 'FVG_Respect' and len(equilibrium_state.tracked_fvgs) > 0:
                            tracked_fvg = equilibrium_state.tracked_fvgs[0]  # Log first FVG for display
                            print(f"    ✓ FVG respected at {time_mid_validation} (FVG {tracked_fvg.top:.2f}-{tracked_fvg.bottom:.2f}, {len(equilibrium_state.tracked_fvgs)} total)")
                        else:
                            print(f"    ✓ Equilibrium zone entered at {time_mid_validation} (eq={equilibrium_state.equilibrium:.2f})")
                        break  # Both Event B and Validation found - move to next step

            # Handle cases where conditions weren't met
            if sweep_broken:
                last_analysis_end_time = time_mid_current if 'time_mid_current' in dir() else end_mid
                continue

            # Handle new sweep occurred during mid TF scan - abandon but DON'T update last_analysis_end_time
            # The new sweep that triggered abandonment should be processed, not blocked
            if new_sweep_occurred:
                continue

            if event_b is None:
                # No Event B found - create partial setup
                self._partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_high_sweep,
                    timestamp_5m_event_b=None,
                    timestamp_5m_validation=None,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_high_sweep,
                    price_5m_event_b=None,
                    price_5m_validation=None,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_high,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=None,
                    condition_validation=None,
                    condition_confirmation=None,
                    conditions_met=1,
                    failure_reason="No Event B found",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_high) - 1, sweep_idx + 5)),
                    indices_5m=None,
                    indices_1m=None,
                    inflexion_idx=inflexion_idx,
                    exit_ob_start_time=exit_target.ob_start_time if exit_target else None,
                    exit_ob_end_time=exit_target.ob_end_time if exit_target else None,
                    exit_ob_top=exit_target.ob_top if exit_target else None,
                    exit_ob_bottom=exit_target.ob_bottom if exit_target else None,
                    gmm_zone_info=sweep_gmm_zone
                ))
                last_analysis_end_time = end_mid
                continue

            # Event B found but no exit OB - create partial setup with 2 conditions met
            if exit_target is None:
                time_mid_event_b_tmp, event_b_type_tmp, price_mid_event_b_tmp = event_b
                self._partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_high_sweep,
                    timestamp_5m_event_b=time_mid_event_b_tmp,
                    timestamp_5m_validation=None,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_high_sweep,
                    price_5m_event_b=price_mid_event_b_tmp,
                    price_5m_validation=None,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_high,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=event_b_type_tmp,
                    condition_validation=None,
                    condition_confirmation=None,
                    conditions_met=2,
                    failure_reason="No exit Order Block found",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_high) - 1, sweep_idx + 5)),
                    indices_5m=(
                        self._tm.df_mid.index.get_loc(time_mid_event_b_tmp),
                        self._tm.df_mid.index.get_loc(time_mid_event_b_tmp)
                    ),
                    indices_1m=None,
                    inflexion_idx=inflexion_idx,
                    exit_ob_start_time=None,
                    exit_ob_end_time=None,
                    exit_ob_top=None,
                    exit_ob_bottom=None,
                    gmm_zone_info=sweep_gmm_zone
                ))
                last_analysis_end_time = end_mid
                continue

            if equilibrium_validation is None:
                # Print the specific failure reason
                if equilibrium_validator is not None:
                    reason = equilibrium_validator.get_failure_reason()
                    if reason:
                        print(f"    ⚠️  Equilibrium validation failed: {reason}")

                # No FVG respect or equilibrium zone entry - create partial setup
                time_mid_event_b_tmp, event_b_type_tmp, price_mid_event_b_tmp = event_b
                self._partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_high_sweep,
                    timestamp_5m_event_b=time_mid_event_b_tmp,
                    timestamp_5m_validation=None,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_high_sweep,
                    price_5m_event_b=price_mid_event_b_tmp,
                    price_5m_validation=None,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_high,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=event_b_type_tmp,
                    condition_validation=None,
                    condition_confirmation=None,
                    conditions_met=2,
                    failure_reason="No FVG respect or equilibrium zone entry",
                    indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_high) - 1, sweep_idx + 5)),
                    indices_5m=(
                        self._tm.df_mid.index.get_loc(time_mid_event_b_tmp),
                        self._tm.df_mid.index.get_loc(time_mid_event_b_tmp)
                    ),
                    indices_1m=None,
                    inflexion_idx=inflexion_idx,
                    exit_ob_start_time=exit_target.ob_start_time if exit_target else None,
                    exit_ob_end_time=exit_target.ob_end_time if exit_target else None,
                    exit_ob_top=exit_target.ob_top if exit_target else None,
                    exit_ob_bottom=exit_target.ob_bottom if exit_target else None,
                    gmm_zone_info=sweep_gmm_zone
                ))
                last_analysis_end_time = end_mid
                continue

            # Unpack results (both Event B and Validation found)
            time_mid_event_b, event_b_type, price_mid_event_b = event_b
            time_mid_validation, validation_type, price_mid_validation = equilibrium_validation

            # Step 4: Final confirmation on low TF with FVG monitoring
            # Use a loop to handle FVG invalidation and re-search
            max_revalidation_attempts = 3
            revalidation_attempt = 0
            confirmation = None
            final_sweep_broken = False
            final_new_sweep_occurred = False
            final_last_time_scanned = None

            while revalidation_attempt < max_revalidation_attempts:
                # Start searching from the candle AFTER validation (e.g., if validation at 13:05, start at 13:06)
                indices_after_validation = self._tm.df_low.index[self._tm.df_low.index > time_mid_validation]
                if len(indices_after_validation) == 0:
                    # No candles after validation time - cannot search for confirmation
                    break
                start_low = indices_after_validation[0]

                (confirmation, sweep_broken, new_sweep_occurred, last_time_scanned,
                 fvg_invalidated, fvg_invalidation_time) = self._search_for_confirmation_low(
                    start_low,
                    market_close_cutoff,
                    entry_direction,
                    time_high_sweep,
                    sweep_idx,
                    inflexion_idx,
                    exit_target,
                    all_sweeps,
                    equilibrium_state
                )

                final_last_time_scanned = last_time_scanned

                # Handle sweep broken
                if sweep_broken:
                    final_sweep_broken = True
                    break

                # Handle new sweep occurred
                if new_sweep_occurred:
                    final_new_sweep_occurred = True
                    break

                # Handle FVG invalidation - re-search for new validation
                if fvg_invalidated and fvg_invalidation_time is not None:
                    revalidation_attempt += 1
                    if revalidation_attempt >= max_revalidation_attempts:
                        print(f"    ⚠️  Max re-validation attempts reached ({max_revalidation_attempts})")
                        break

                    # Search for new validation (FVG or Equilibrium) from invalidation time
                    new_equilibrium_validator = EquilibriumValidator(
                        df_mid=self._tm.df_mid,
                        swept_level=swept_level,
                        entry_direction=entry_direction,
                        fvg_mid=self._tm.fvg_mid,
                        sweep_time=actual_sweep_time,  # Use actual 5M sweep point for FVG filtering
                        inflexions_mid=self._tm.inflexions_mid,
                        use_fvg_validation=self._use_fvg_validation,
                        use_equilibrium_validation=self._use_equilibrium_validation,
                        require_fvg_in_equilibrium=self._require_fvg_in_equilibrium
                    )

                    # Loop through candles using single-candle method (O(n) optimization)
                    new_validation = None
                    for revalidation_time in self._tm.df_mid.loc[fvg_invalidation_time:market_close_cutoff].index:
                        new_validation = new_equilibrium_validator.validate_at_candle(revalidation_time)
                        if new_validation is not None:
                            break

                    if new_validation is None:
                        # No new validation found after FVG invalidation
                        print(f"    ⚠️  No new validation found after FVG invalidation")
                        validation_type = "FVG_Respect (invalidated)"
                        equilibrium_state = None
                        break

                    # Update validation info for next iteration
                    time_mid_validation, validation_type, price_mid_validation, equilibrium_state = new_validation

                    # Log new validation
                    if validation_type == 'FVG_Respect' and len(equilibrium_state.tracked_fvgs) > 0:
                        tracked_fvg = equilibrium_state.tracked_fvgs[0]  # Log first FVG for display
                        print(f"    ✓ New FVG respected at {time_mid_validation} (FVG {tracked_fvg.top:.2f}-{tracked_fvg.bottom:.2f}, {len(equilibrium_state.tracked_fvgs)} total)")
                    else:
                        print(f"    ✓ Equilibrium zone entered at {time_mid_validation} (eq={equilibrium_state.equilibrium:.2f})")

                    # Continue the while loop to search for confirmation again
                    continue

                # Confirmation found or no more retries needed
                break

            # Unpack confirmation if found
            if confirmation is not None:
                time_low_confirmation, confirmation_type, price_low_confirmation = confirmation

            # Handle sweep broken during low TF scan
            if final_sweep_broken:
                last_analysis_end_time = final_last_time_scanned if final_last_time_scanned else last_time_scanned
                continue

            # Handle new sweep occurred during low TF scan - abandon but DON'T update last_analysis_end_time
            # The new sweep that triggered abandonment should be processed, not blocked
            if final_new_sweep_occurred:
                continue

            # Handle no confirmation found - create partial setup
            if confirmation is None:
                failure_reason = "No low TF confirmation"
                if revalidation_attempt >= max_revalidation_attempts:
                    failure_reason = "FVG validation invalidated, max re-validations reached"
                elif validation_type == "FVG_Respect (invalidated)":
                    failure_reason = "FVG validation invalidated, no new validation found"

                self._partial_setups.append(PartialSetup(
                    timestamp_1h_sweep=time_high_sweep,
                    timestamp_5m_event_b=time_mid_event_b,
                    timestamp_5m_validation=time_mid_validation,
                    timestamp_1m_confirmation=None,
                    price_1h_sweep=price_high_sweep,
                    price_5m_event_b=price_mid_event_b,
                    price_5m_validation=price_mid_validation,
                    price_1m_confirmation=None,
                    trend_1h_before_sweep=trend_high,
                    entry_direction=entry_direction,
                    condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                    condition_event_b=event_b_type,
                    condition_validation=validation_type,
                    condition_confirmation=None,
                    conditions_met=3,
                    failure_reason=failure_reason,
                    indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_high) - 1, sweep_idx + 5)),
                    indices_5m=(
                        self._tm.df_mid.index.get_loc(time_mid_event_b),
                        self._tm.df_mid.index.get_loc(time_mid_validation)
                    ),
                    indices_1m=(
                        self._tm.df_low.index.get_loc(start_low),
                        self._tm.df_low.index.get_loc(final_last_time_scanned if final_last_time_scanned else last_time_scanned)
                    ),
                    inflexion_idx=inflexion_idx,
                    exit_ob_start_time=exit_target.ob_start_time if exit_target else None,
                    exit_ob_end_time=exit_target.ob_end_time if exit_target else None,
                    exit_ob_top=exit_target.ob_top if exit_target else None,
                    exit_ob_bottom=exit_target.ob_bottom if exit_target else None,
                    gmm_zone_info=sweep_gmm_zone
                ))
                last_analysis_end_time = last_time_scanned
                continue
            # print('exit_target.take_profit --> ', exit_target.take_profit)

            # All conditions met - generate Strategy signal with TP/SL
            signal = TradeSignal(
                timestamp_1h_sweep=time_high_sweep,
                timestamp_5m_event_b=time_mid_event_b,
                timestamp_5m_validation=time_mid_validation,
                timestamp_1m_confirmation=time_low_confirmation,
                timestamp_entry=time_low_confirmation,
                trend_1h_before_sweep=trend_high,
                entry_direction=entry_direction,
                price_1h_sweep=price_high_sweep,
                price_5m_event_b=price_mid_event_b,
                price_5m_validation=price_mid_validation,
                price_1m_confirmation=price_low_confirmation,
                price_entry=price_low_confirmation,
                condition_liquidity_sweep=f"{sweep_type} liquidity swept",
                condition_event_b=event_b_type,
                condition_validation=validation_type,
                condition_confirmation=confirmation_type,
                indices_1h=(max(0, sweep_idx - 5), min(len(self._tm.df_high) - 1, sweep_idx + 5)),
                indices_5m=(
                    self._tm.df_mid.index.get_loc(time_mid_event_b),
                    self._tm.df_mid.index.get_loc(time_mid_validation)
                ),
                indices_1m=(
                    self._tm.df_low.index.get_loc(start_low),
                    self._tm.df_low.index.get_loc(time_low_confirmation)
                ),
                # Exit target and equilibrium fields
                take_profit_price=exit_target.take_profit,
                exit_ob_top=exit_target.ob_top,
                exit_ob_bottom=exit_target.ob_bottom,
                exit_ob_start_idx=exit_target.ob_start_idx,
                exit_ob_end_idx=exit_target.ob_end_idx,
                equilibrium_level=equilibrium_state.equilibrium if equilibrium_state else None,
                equilibrium_fixed_level=equilibrium_state.fixed_level if equilibrium_state else None,
                equilibrium_running_extreme=equilibrium_state.running_extreme if equilibrium_state else None,
                # GMM zone info for debug visualization
                gmm_zone_info=sweep_gmm_zone
            )

            # Calculate stop loss based on 2:1 R/R
            signal.calculate_stop_loss()

            signals.append(signal)
            print(f"    ✅ COMPLETE SIGNAL #{len(signals)} - {entry_direction.upper()} entry")
            print(f"       Entry: {signal.price_entry:.2f}, TP: {signal.take_profit_price:.2f}, SL: {signal.stop_loss_price:.2f}")

            last_analysis_end_time = time_low_confirmation

        print(f"\n✅ Found {len(signals)} complete Strategy trade signals!")
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

    @property
    def gmm_zone(self) -> Optional[GMMZoneInfo]:
        """Current GMM zone detection result (if enabled)."""
        return self._current_gmm_zone

    @property
    def gmm_enabled(self) -> bool:
        """Whether GMM zone detection is enabled."""
        return self._gmm_enabled

    @property
    def df_high_original(self) -> pd.DataFrame:
        """Original unfiltered high timeframe DataFrame."""
        return self._tm.df_high_original

    @property
    def df_mid_original(self) -> pd.DataFrame:
        """Original unfiltered mid timeframe DataFrame."""
        return self._tm.df_mid_original
