"""
Data models for the multi-timeframe trading strategy.

Contains all dataclasses and enums used across strategy modules.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple


class StrategyState(Enum):
    """Strategy state machine states."""
    WAITING_FOR_LIQUIDITY_SWEEP = 1
    WAITING_FOR_EVENT_B = 2          # 5M BOS/IFVG opposite direction
    WAITING_FOR_VALIDATION = 3       # 5M FVG/Demand respect
    WAITING_FOR_FINAL_CONFIRMATION = 4  # 1M BOS/IFVG
    ENTRY_SIGNAL_GENERATED = 5


@dataclass
class SweepInfo:
    """Information about a detected liquidity sweep."""
    candle_idx: int
    sweep_type: str  # 'high' or 'low'
    swept_level: float
    inflexion_idx: int
    timestamp: datetime


@dataclass
class PartialSetup:
    """Represents a partial trade setup (not all conditions met)."""

    # Timestamps (None if condition not met)
    timestamp_1h_sweep: datetime
    timestamp_5m_event_b: Optional[datetime]
    timestamp_5m_validation: Optional[datetime]
    timestamp_1m_confirmation: Optional[datetime]

    # Prices
    price_1h_sweep: float
    price_5m_event_b: Optional[float]
    price_5m_validation: Optional[float]
    price_1m_confirmation: Optional[float]

    # Trend and direction
    trend_1h_before_sweep: str
    entry_direction: str

    # Conditions met
    condition_liquidity_sweep: str
    condition_event_b: Optional[str]
    condition_validation: Optional[str]
    condition_confirmation: Optional[str]

    # Status tracking
    conditions_met: int  # How many of 4 conditions passed
    failure_reason: str  # Where it failed

    # Metadata for visualization
    indices_1h: Tuple[int, int]
    indices_5m: Optional[Tuple[int, int]]
    indices_1m: Optional[Tuple[int, int]]
    inflexion_idx: int  # Index of the inflexion point for sweep validation


@dataclass
class EquilibriumState:
    """Tracks the dynamic equilibrium calculation state for Strategy II."""
    fixed_level: float        # Swept level (fixed anchor)
    running_extreme: float    # Running min (SHORT) or max (LONG)
    equilibrium: float        # (fixed_level + running_extreme) / 2
    entry_direction: str      # 'long' or 'short'
    in_target_zone: bool      # True if price in premium/discount
    trigger_time: Optional[datetime] = None  # Time when zone was entered
    trigger_price: Optional[float] = None    # Price when zone was entered


@dataclass
class ExitTarget:
    """Exit target Order Block information for Strategy II."""
    ob_top: float             # Top of exit Order Block
    ob_bottom: float          # Bottom of exit Order Block
    ob_start_idx: int         # Index where OB starts
    ob_end_idx: int           # Index where OB ends
    take_profit: float        # TP price (top for SHORT, bottom for LONG)
    bos_idx: int              # Index of the BOS that ended previous trend
    ob_start_time: Optional[datetime] = None  # Timestamp where OB starts
    ob_end_time: Optional[datetime] = None    # Timestamp where OB ends


@dataclass
class TradeSignal:
    """Represents a complete trade setup with all conditions met."""

    # Timestamps
    timestamp_1h_sweep: datetime
    timestamp_5m_event_b: datetime
    timestamp_5m_validation: datetime
    timestamp_1m_confirmation: datetime
    timestamp_entry: datetime

    # Trend information
    trend_1h_before_sweep: str  # 'bullish' or 'bearish'
    entry_direction: str        # 'long' or 'short' (opposite of 1h trend)

    # Prices
    price_1h_sweep: float
    price_5m_event_b: float
    price_5m_validation: float
    price_1m_confirmation: float
    price_entry: float

    # Conditions met
    condition_liquidity_sweep: str  # Description of liquidity sweep
    condition_event_b: str          # 'BOS' or 'IFVG'
    condition_validation: str       # 'FVG' or 'Demand Zone' or 'Equilibrium'
    condition_confirmation: str     # 'BOS' or 'IFVG'

    # Metadata for visualization
    indices_1h: Tuple[int, int]     # (start, end) indices in 1H data
    indices_5m: Tuple[int, int]     # (start, end) indices in 5M data
    indices_1m: Tuple[int, int]     # (start, end) indices in 1M data

    # Strategy II fields (optional, None for Strategy I)
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    exit_ob_top: Optional[float] = None
    exit_ob_bottom: Optional[float] = None
    exit_ob_start_idx: Optional[int] = None
    exit_ob_end_idx: Optional[int] = None
    equilibrium_level: Optional[float] = None
    equilibrium_fixed_level: Optional[float] = None
    equilibrium_running_extreme: Optional[float] = None
    strategy_version: str = "I"  # "I" or "II"

    def calculate_stop_loss(self) -> None:
        """Calculate stop loss based on 2:1 R/R from take profit."""
        if self.take_profit_price is None or self.price_entry is None:
            return

        profit_distance = abs(self.price_entry - self.take_profit_price)
        risk_distance = profit_distance / 2  # 2:1 R/R

        if self.entry_direction == 'short':
            self.stop_loss_price = self.price_entry + risk_distance
        else:  # long
            self.stop_loss_price = self.price_entry - risk_distance
