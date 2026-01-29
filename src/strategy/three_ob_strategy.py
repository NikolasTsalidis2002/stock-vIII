"""
3-OB Strategy wrapper — single-timeframe interface analogous to MultiTimeframeStrategy.
"""

from typing import List, Optional
import pandas as pd

from .models import TradeSignal
from .three_ob_engine import ThreeOBEngine


class ThreeOBStrategy:
    """
    Single-timeframe 3-OB strategy.

    Usage:
        strategy = ThreeOBStrategy(df, close_break=True, min_dist_pct=0.10)
        signals = strategy.scan_for_signals(max_signals=5)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        close_break: bool = True,
        min_dist_pct: float = 0.10,
    ):
        self.df = df
        self._engine = ThreeOBEngine(df, close_break=close_break, min_dist_pct=min_dist_pct)

        # Expose the working DataFrame with datetime index (for Backtester compatibility)
        df_work = self._engine.df.copy()
        if 'time' in df_work.columns:
            df_work = df_work.set_index('time')
        self.df_low = df_work

        # Empty partial_setups for compatibility with visualizer
        self.partial_setups: list = []

    def scan_for_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        return self._engine.scan_for_signals(max_signals=max_signals)
