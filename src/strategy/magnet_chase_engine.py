"""
Magnet Chase Strategy Engine.

Trades toward the closest unmitigated FVG or OB zone.
No liquidity sweep required — pure zone-proximity driven.

Uses:
- 15min (or configurable) for zone detection
- 5min (or configurable) for BOS/IFVG entry confirmation

Reuses existing components:
- EventBDetector for BOS/IFVG detection on confirmation TF
- zone_proximity module for zone ranking
- smc indicators for FVG, OB, BOS computation
"""

import logging
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

from indicators.smc import smc
from indicators.smc_custom import smc_custom

from .zone_proximity import (
    ZoneRecord, RankedZone,
    extract_zone_records, get_active_zones,
    rank_zones_by_proximity, select_target_zone,
    compute_tp_price, compute_trade_direction,
)

logger = logging.getLogger(__name__)


@dataclass
class MagnetChaseSignal:
    """A complete Magnet Chase trade signal."""
    # Target zone info
    target_zone_type: str       # 'FVG' or 'OB'
    target_zone_direction: int  # 1 or -1
    target_zone_top: float
    target_zone_bottom: float
    target_zone_mid: float
    target_zone_side: str       # 'above' or 'below'
    target_distance_pct: float
    target_rank: int

    # Trade parameters
    entry_direction: str        # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    take_profit: float
    stop_loss: float
    rr_ratio: float

    # Confirmation info
    confirmation_type: str      # 'BOS' or 'IFVG'
    confirmation_time: datetime
    confirmation_level: float

    # Detection TF bar index where target was identified
    detection_bar_idx: int

    # Metadata
    target_identified_time: Optional[datetime] = None

    def to_trade_signal(self):
        """Convert to a TradeSignal for use with the Backtester."""
        from .models import TradeSignal
        idx_tuple = (self.detection_bar_idx, self.detection_bar_idx)
        return TradeSignal(
            timestamp_1h_sweep=self.entry_time,
            timestamp_5m_event_b=self.entry_time,
            timestamp_5m_validation=self.entry_time,
            timestamp_1m_confirmation=self.entry_time,
            timestamp_entry=self.entry_time,
            trend_1h_before_sweep='neutral',
            entry_direction=self.entry_direction,
            price_1h_sweep=self.entry_price,
            price_5m_event_b=self.entry_price,
            price_5m_validation=self.entry_price,
            price_1m_confirmation=self.entry_price,
            price_entry=self.entry_price,
            condition_liquidity_sweep='magnet_chase',
            condition_event_b=self.confirmation_type,
            condition_validation=f'{self.target_zone_type}_{self.target_zone_side}',
            condition_confirmation=self.confirmation_type,
            indices_1h=idx_tuple,
            indices_5m=idx_tuple,
            indices_1m=idx_tuple,
            take_profit_price=self.take_profit,
            stop_loss_price=self.stop_loss,
        )


@dataclass
class MagnetChaseResult:
    """Result of running the Magnet Chase engine."""
    signals: List[MagnetChaseSignal] = field(default_factory=list)
    total_zones_scanned: int = 0
    total_targets_identified: int = 0
    skipped_no_confirmation: int = 0
    skipped_low_rr: int = 0
    skipped_zone_mitigated: int = 0


class MagnetChaseEngine:
    """
    Magnet Chase strategy engine.

    Scans bar-by-bar on the detection timeframe for the closest active zone,
    then checks the confirmation timeframe for BOS/IFVG entry signals.
    """

    def __init__(
        self,
        df_detect: pd.DataFrame,
        df_confirm: pd.DataFrame,
        config: Optional[Dict] = None,
    ):
        """
        Initialize with detection and confirmation timeframe data.

        Args:
            df_detect: Detection TF OHLCV data (e.g., 15min) with 'time' column or datetime index.
            df_confirm: Confirmation TF OHLCV data (e.g., 5min) with 'time' column or datetime index.
            config: Optional config dict (from strategy_config.json 'magnet_chase' section).
        """
        config = config or {}
        self.max_distance_pct = config.get('max_distance_pct', 1.0)
        self.min_rr_ratio = config.get('min_rr_ratio', 1.5)
        self.timeout_hours = config.get('timeout_hours', 4)
        self.prefer_ob = config.get('prefer_ob_over_fvg', True)
        self.require_trend_alignment = config.get('require_trend_alignment', False)
        self.confirmation_timeout_hours = config.get('confirmation_timeout_hours', 2)
        self.close_break = config.get('close_break', True)

        # Prepare detection TF
        if 'time' in df_detect.columns:
            self.df_detect = df_detect.set_index('time').copy()
            self.df_detect.index = pd.to_datetime(self.df_detect.index)
        else:
            self.df_detect = df_detect.copy()

        # Prepare confirmation TF
        if 'time' in df_confirm.columns:
            self.df_confirm = df_confirm.set_index('time').copy()
            self.df_confirm.index = pd.to_datetime(self.df_confirm.index)
        else:
            self.df_confirm = df_confirm.copy()

        # Compute indicators on detection TF
        logger.info("Computing indicators on detection TF...")
        self.inflexions_detect = smc_custom.inflexion_points(self.df_detect)
        self.bos_detect = smc_custom.bos(self.df_detect, self.inflexions_detect, close_break=self.close_break)
        self.fvg_detect = smc.fvg(self.df_detect, join_consecutive=True)
        self.ob_detect = smc_custom.ob(self.df_detect, self.bos_detect, self.inflexions_detect)

        # Compute indicators on confirmation TF
        logger.info("Computing indicators on confirmation TF...")
        self.inflexions_confirm = smc_custom.inflexion_points(self.df_confirm)
        self.bos_confirm = smc_custom.bos(self.df_confirm, self.inflexions_confirm, close_break=self.close_break)
        self.fvg_confirm = smc.fvg(self.df_confirm, join_consecutive=True)
        self.swings_confirm = smc.swing_highs_lows(self.df_confirm, swing_length=5)

        # Extract all zone records
        self.fvg_records = extract_zone_records(self.fvg_detect, 'FVG')
        self.ob_records = extract_zone_records(self.ob_detect, 'OB')

        # Precompute last BOS direction on detection TF
        n_detect = len(self.df_detect)
        self._last_bos_dir = np.zeros(n_detect, dtype=int)
        current_dir = 0
        for i in range(n_detect):
            if not pd.isna(self.bos_detect['BOS'].iloc[i]):
                current_dir = int(self.bos_detect['BOS'].iloc[i])
            self._last_bos_dir[i] = current_dir

        # Build time mapping: for each detection bar, find the range of confirmation bars
        self._detect_times = self.df_detect.index.tolist()
        self._confirm_times = self.df_confirm.index.tolist()

    def _find_confirm_range(self, detect_time: datetime, next_detect_time: Optional[datetime]) -> Tuple[int, int]:
        """Find confirmation TF bar indices that fall within a detection TF bar window."""
        confirm_idx = self.df_confirm.index
        start = confirm_idx.searchsorted(detect_time, side='left')
        if next_detect_time is not None:
            end = confirm_idx.searchsorted(next_detect_time, side='left')
        else:
            end = len(confirm_idx)
        return start, end

    def _find_last_swing(self, confirm_bar_idx: int, direction: str) -> Optional[float]:
        """
        Find the last swing high (for short SL) or swing low (for long SL)
        on the confirmation TF before the given bar.
        """
        # swing_highs_lows returns 1 for swing high, -1 for swing low in 'HighLow' column
        target = 1 if direction == 'short' else -1  # short SL = swing high, long SL = swing low
        for i in range(confirm_bar_idx - 1, -1, -1):
            val = self.swings_confirm['HighLow'].iloc[i]
            if not pd.isna(val) and int(val) == target:
                if target == 1:
                    return float(self.df_confirm['high'].iloc[i])
                else:
                    return float(self.df_confirm['low'].iloc[i])
        return None

    def _check_confirmation_at_bar(self, confirm_idx: int, entry_direction: str) -> Optional[Tuple[str, float]]:
        """
        Check a single confirmation TF bar for BOS or IFVG in the entry direction.

        Returns:
            ('BOS', level) or ('IFVG', level) or None
        """
        # Check BOS
        target_bos = 1 if entry_direction == 'long' else -1
        bos_val = self.bos_confirm['BOS'].iloc[confirm_idx]
        if not np.isnan(bos_val) and int(bos_val) == target_bos:
            level = float(self.bos_confirm['Level'].iloc[confirm_idx])
            return ('BOS', level)

        # Check IFVG: for long, we want bearish FVG (-1) disrespected
        # For short, we want bullish FVG (1) disrespected
        target_fvg = -1 if entry_direction == 'long' else 1
        local_fvg = self.fvg_confirm.loc[self.fvg_confirm['StatusIndex'] == confirm_idx]
        for i in range(len(local_fvg)):
            fvg_val = local_fvg['FVG'].iloc[i]
            respected = local_fvg['Respected'].iloc[i]
            if (not np.isnan(fvg_val) and respected == False and int(fvg_val) == target_fvg):
                level = float(self.df_confirm['close'].iloc[confirm_idx])
                return ('IFVG', level)

        return None

    def scan(self, max_signals: int = 50, scan_step: int = 1) -> MagnetChaseResult:
        """
        Scan for Magnet Chase signals.

        Args:
            max_signals: Maximum signals to collect.
            scan_step: Step between detection bars to scan (1 = every bar).

        Returns:
            MagnetChaseResult with signals and stats.
        """
        result = MagnetChaseResult()
        n_detect = len(self.df_detect)
        n_confirm = len(self.df_confirm)

        # Skip first 50 bars to let indicators warm up
        start_bar = 50

        # Track: don't enter a new trade while one is active
        active_trade_exit_time = None

        for bar in range(start_bar, n_detect, scan_step):
            if len(result.signals) >= max_signals:
                break

            detect_time = self._detect_times[bar]

            # Skip if we're inside an active trade
            if active_trade_exit_time is not None and detect_time < active_trade_exit_time:
                continue
            active_trade_exit_time = None

            price = float(self.df_detect['close'].iloc[bar])

            # Step 1: Get active zones
            active_zones = get_active_zones(self.fvg_records, self.ob_records, bar)
            if not active_zones:
                continue

            result.total_zones_scanned += len(active_zones)

            # Step 2: Rank by proximity
            above, below = rank_zones_by_proximity(active_zones, price, self.max_distance_pct)

            # Step 3: Select target
            bos_dir = int(self._last_bos_dir[bar])
            if self.require_trend_alignment:
                # Filter to trend-aligned zones only
                above = [z for z in above if z.zone.direction == bos_dir or bos_dir == 0]
                below = [z for z in below if z.zone.direction == bos_dir or bos_dir == 0]
                # Re-rank
                for i, rz in enumerate(above):
                    rz.rank = i + 1
                for i, rz in enumerate(below):
                    rz.rank = i + 1

            target = select_target_zone(above, below, bos_dir, self.prefer_ob)
            if target is None:
                continue

            result.total_targets_identified += 1

            # Step 4: Determine direction and TP
            direction = compute_trade_direction(target)
            tp = compute_tp_price(target)

            # Step 5: Look for confirmation on 5min TF
            # Find confirmation bars from current detect bar to timeout
            next_bar_time = self._detect_times[bar + 1] if bar + 1 < n_detect else None
            timeout_time = detect_time + timedelta(hours=self.confirmation_timeout_hours)

            # Find all confirmation bars in the window [detect_time, timeout_time]
            confirm_start = self.df_confirm.index.searchsorted(detect_time, side='left')
            confirm_end = self.df_confirm.index.searchsorted(timeout_time, side='right')
            confirm_end = min(confirm_end, n_confirm)

            found_confirmation = False
            for ci in range(confirm_start, confirm_end):
                confirm_time = self._confirm_times[ci]

                # Check if target zone got mitigated before we enter
                # (zone midpoint reached by price on detection TF)
                # Approximate: check if confirm price has crossed the zone
                confirm_price = float(self.df_confirm['close'].iloc[ci])
                zone_mitigated = False
                if direction == 'long' and confirm_price >= target.zone.top:
                    zone_mitigated = True
                elif direction == 'short' and confirm_price <= target.zone.bottom:
                    zone_mitigated = True

                if zone_mitigated:
                    result.skipped_zone_mitigated += 1
                    break

                confirmation = self._check_confirmation_at_bar(ci, direction)
                if confirmation is None:
                    continue

                conf_type, conf_level = confirmation
                entry_price = float(self.df_confirm['close'].iloc[ci])

                # Step 6: Compute SL
                sl = self._find_last_swing(ci, direction)
                if sl is None:
                    # Fallback: use TP distance / min_rr_ratio
                    tp_dist = abs(entry_price - tp)
                    if tp_dist == 0:
                        continue
                    sl_dist = tp_dist / self.min_rr_ratio
                    if direction == 'long':
                        sl = entry_price - sl_dist
                    else:
                        sl = entry_price + sl_dist

                # Validate SL is on the correct side
                if direction == 'long' and sl >= entry_price:
                    sl = entry_price - abs(entry_price - tp) / self.min_rr_ratio
                elif direction == 'short' and sl <= entry_price:
                    sl = entry_price + abs(entry_price - tp) / self.min_rr_ratio

                # Validate TP is on correct side of entry
                if direction == 'long' and tp <= entry_price:
                    continue
                if direction == 'short' and tp >= entry_price:
                    continue

                # Step 7: Check R:R
                tp_dist = abs(entry_price - tp)
                sl_dist = abs(entry_price - sl)
                if sl_dist == 0:
                    continue
                rr = tp_dist / sl_dist
                if rr < self.min_rr_ratio:
                    result.skipped_low_rr += 1
                    found_confirmation = True  # We found one, just skipped it
                    break

                # Signal!
                signal = MagnetChaseSignal(
                    target_zone_type=target.zone.zone_type,
                    target_zone_direction=target.zone.direction,
                    target_zone_top=target.zone.top,
                    target_zone_bottom=target.zone.bottom,
                    target_zone_mid=target.zone.mid,
                    target_zone_side=target.side,
                    target_distance_pct=round(target.distance_pct, 3),
                    target_rank=target.rank,
                    entry_direction=direction,
                    entry_price=round(entry_price, 2),
                    entry_time=confirm_time,
                    take_profit=round(tp, 2),
                    stop_loss=round(sl, 2),
                    rr_ratio=round(rr, 2),
                    confirmation_type=conf_type,
                    confirmation_time=confirm_time,
                    confirmation_level=round(conf_level, 2),
                    detection_bar_idx=bar,
                    target_identified_time=detect_time,
                )
                result.signals.append(signal)
                logger.info(
                    f"Signal: {direction.upper()} @ ${entry_price:.2f} "
                    f"→ TP ${tp:.2f} SL ${sl:.2f} (R:R {rr:.1f}) "
                    f"via {conf_type} on {confirm_time} "
                    f"target={target.zone.zone_type} {target.side} "
                    f"dist={target.distance_pct:.2f}%"
                )

                # Set active trade exit time (timeout)
                active_trade_exit_time = confirm_time + timedelta(hours=self.timeout_hours)
                found_confirmation = True
                break

            if not found_confirmation:
                result.skipped_no_confirmation += 1

        return result
