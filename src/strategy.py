"""
DEPRECATED: This file exists for backward compatibility only.

The strategy module has been refactored into multiple files under src/strategy/.

Use the new modular imports:
    from src.strategy import MultiTimeframeStrategy, PartialSetup, TradeSignal

Or use the new classes directly:
    from src.strategy import TimeframeManager, StrategyEngine
"""

import warnings

# Re-export from the new module location
from src.strategy import (
    StrategyState,
    PartialSetup,
    TradeSignal,
    SweepInfo,
    TimeframeManager,
    LiquidityDetector,
    EventBDetector,
    ValidationDetector,
    ConfirmationDetector,
    StrategyEngine,
    MultiTimeframeStrategy,
)

# Issue deprecation warning when importing directly from this file
warnings.warn(
    "Importing from 'src.strategy' (the file) is deprecated. "
    "Use 'from src.strategy import ...' (the package) instead. "
    "This file will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'StrategyState',
    'PartialSetup',
    'TradeSignal',
    'SweepInfo',
    'TimeframeManager',
    'LiquidityDetector',
    'EventBDetector',
    'ValidationDetector',
    'ConfirmationDetector',
    'StrategyEngine',
    'MultiTimeframeStrategy',
]
