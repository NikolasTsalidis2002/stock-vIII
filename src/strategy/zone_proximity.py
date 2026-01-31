"""
Zone proximity analysis for the Magnet Chase strategy.

Extracts active FVG/OB zones, ranks them by distance to current price,
and identifies the closest target zone on each side.

Extracted from scripts/smc_dashboard.py (_extract_zone_records,
compute_zone_proximity_analysis) and adapted for bar-by-bar use.
"""

import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ZoneRecord:
    """A single active FVG or OB zone."""
    idx: int              # Row index in the source DataFrame
    zone_type: str        # 'FVG' or 'OB'
    direction: int        # 1 (bullish) or -1 (bearish)
    top: float
    bottom: float
    mid: float
    mit_idx: int          # 0 = not yet mitigated


@dataclass
class RankedZone:
    """A zone with proximity metadata relative to a price."""
    zone: ZoneRecord
    distance_pct: float   # abs(mid - price) / price * 100
    side: str             # 'above' or 'below'
    rank: int             # 1 = closest on this side


def extract_zone_records(zone_df: pd.DataFrame, type_col: str) -> List[ZoneRecord]:
    """
    Extract zone records from an FVG (type_col='FVG') or OB (type_col='OB') DataFrame.

    Args:
        zone_df: DataFrame with columns: type_col, Top, Bottom, MitigatedIndex
        type_col: Column name that holds the zone type value (1/-1 or NaN)

    Returns:
        List of ZoneRecord objects for non-NaN rows.
    """
    records = []
    for i in range(len(zone_df)):
        if pd.isna(zone_df[type_col].iloc[i]):
            continue
        direction = int(zone_df[type_col].iloc[i])
        top = float(zone_df['Top'].iloc[i])
        bottom = float(zone_df['Bottom'].iloc[i])
        mid = (top + bottom) / 2
        mit_raw = zone_df['MitigatedIndex'].iloc[i]
        mit_idx = int(mit_raw) if (not pd.isna(mit_raw) and int(mit_raw) > 0) else 0
        records.append(ZoneRecord(
            idx=i, zone_type=type_col, direction=direction,
            top=top, bottom=bottom, mid=mid, mit_idx=mit_idx,
        ))
    return records


def get_active_zones(
    fvg_records: List[ZoneRecord],
    ob_records: List[ZoneRecord],
    current_bar: int,
) -> List[ZoneRecord]:
    """
    Filter to zones that are active (created on or before current_bar
    and not yet mitigated).

    Args:
        fvg_records: All FVG zone records.
        ob_records: All OB zone records.
        current_bar: The current bar index.

    Returns:
        List of active ZoneRecord objects.
    """
    active = []
    for z in fvg_records + ob_records:
        if z.idx > current_bar:
            continue
        # Active = never mitigated (0) or mitigated after current bar
        if z.mit_idx == 0 or z.mit_idx > current_bar:
            active.append(z)
    return active


def rank_zones_by_proximity(
    active_zones: List[ZoneRecord],
    price: float,
    max_distance_pct: float = 100.0,
    min_distance_pct: float = 0.0,
) -> tuple[List[RankedZone], List[RankedZone]]:
    """
    Rank active zones by distance to current price, split by side.

    Args:
        active_zones: List of active zone records.
        price: Current price.
        max_distance_pct: Maximum distance filter (%).

    Returns:
        (above_ranked, below_ranked) — each sorted by distance ascending,
        rank 1 = closest.
    """
    above = []
    below = []

    for z in active_zones:
        dist_pct = abs(z.mid - price) / price * 100 if price != 0 else 0
        if dist_pct > max_distance_pct:
            continue
        if dist_pct < min_distance_pct:
            continue
        side = 'above' if z.mid > price else 'below'
        rz = RankedZone(zone=z, distance_pct=dist_pct, side=side, rank=0)
        if side == 'above':
            above.append(rz)
        else:
            below.append(rz)

    # Sort by distance and assign ranks
    above.sort(key=lambda rz: rz.distance_pct)
    below.sort(key=lambda rz: rz.distance_pct)
    for i, rz in enumerate(above):
        rz.rank = i + 1
    for i, rz in enumerate(below):
        rz.rank = i + 1

    return above, below


def select_target_zone(
    above: List[RankedZone],
    below: List[RankedZone],
    last_bos_direction: int = 0,
    prefer_ob: bool = True,
) -> Optional[RankedZone]:
    """
    Select the single best target zone.

    Rules:
    1. Pick the closer side's rank-1 zone.
    2. If both sides' rank-1 zones are within 20% distance of each other,
       pick the one aligned with last BOS direction.
    3. If still tied, prefer OB over FVG (if prefer_ob=True).

    Args:
        above: Ranked zones above price (rank 1 = closest).
        below: Ranked zones below price.
        last_bos_direction: Last BOS direction on detection TF (1=bull, -1=bear, 0=none).
        prefer_ob: When tied, prefer OB zones.

    Returns:
        The selected RankedZone, or None if no zones available.
    """
    if not above and not below:
        return None
    if not above:
        return below[0]
    if not below:
        return above[0]

    a = above[0]
    b = below[0]

    # Check if roughly equal distance (within 20%)
    min_dist = min(a.distance_pct, b.distance_pct)
    max_dist = max(a.distance_pct, b.distance_pct)
    roughly_equal = (max_dist - min_dist) / max_dist < 0.20 if max_dist > 0 else True

    if roughly_equal and last_bos_direction != 0:
        # Prefer the side aligned with BOS direction
        # Bullish BOS → price moving up → target above
        # Bearish BOS → price moving down → target below
        if last_bos_direction == 1:
            return a  # above
        else:
            return b  # below

    if roughly_equal and prefer_ob:
        # Prefer OB over FVG
        if a.zone.zone_type == 'OB' and b.zone.zone_type != 'OB':
            return a
        if b.zone.zone_type == 'OB' and a.zone.zone_type != 'OB':
            return b

    # Default: pick whichever is closer
    return a if a.distance_pct <= b.distance_pct else b


def compute_tp_price(target: RankedZone) -> float:
    """
    Compute take profit = near edge of the target zone.

    For a zone above price, near edge = zone Bottom.
    For a zone below price, near edge = zone Top.
    """
    if target.side == 'above':
        return target.zone.bottom
    else:
        return target.zone.top


def compute_trade_direction(target: RankedZone) -> str:
    """
    Determine trade direction from target zone position.
    Zone above → LONG, zone below → SHORT.
    """
    return 'long' if target.side == 'above' else 'short'


class ImpulseTracker:
    """Tracks avg candles from zone creation to max-distance point (tilting point)."""

    def __init__(self):
        self._fvg_counts: List[int] = []
        self._ob_counts: List[int] = []

    def record(self, zone_type: str, candles_to_peak: int):
        if zone_type == 'FVG':
            self._fvg_counts.append(candles_to_peak)
        else:
            self._ob_counts.append(candles_to_peak)

    def avg(self, zone_type: str) -> float:
        counts = self._fvg_counts if zone_type == 'FVG' else self._ob_counts
        return sum(counts) / len(counts) if counts else 0.0
