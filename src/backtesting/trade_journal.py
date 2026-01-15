"""
Trade journal for CSV export and console display.

Handles formatting and exporting trade results.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional

from src.strategy.models import TradeSignal
from .models import TradeResult


class TradeJournal:
    """
    Handles trade journal export and formatting.
    """

    def __init__(self, symbol: str = 'TSLA'):
        """
        Initialize trade journal.

        Args:
            symbol: Stock symbol for this journal
        """
        self.symbol = symbol.upper()

    def to_dataframe(
        self,
        results: List[TradeResult],
        signals: Optional[List[TradeSignal]] = None
    ) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.

        Args:
            results: List of TradeResult objects
            signals: Optional list of original TradeSignal objects

        Returns:
            DataFrame with all trade details
        """
        rows = []

        for result in results:
            row = {
                # Key info first (symbol, outcome, P&L)
                'symbol': self.symbol,
                'outcome': result.outcome.value.upper(),
                'pnl_dollars': round(result.pnl_dollars, 2),
                'pnl_percent': round(result.pnl_percent, 4),

                # Trade identification
                'trade_id': result.signal_index + 1,
                'direction': result.entry_direction.upper(),

                # Entry details
                'entry_time': result.entry_time,
                'entry_price': result.entry_price,

                # Exit details
                'exit_time': result.exit_time,
                'exit_price': result.exit_price,

                # Targets
                'take_profit': result.take_profit_price,
                'stop_loss': result.stop_loss_price,

                # Additional P&L metrics
                'pnl_r_multiple': round(result.pnl_r_multiple, 2),

                # Capital (compounding)
                'capital_before': round(result.capital_before, 2),
                'capital_after': round(result.capital_after, 2),
                'position_size': round(result.position_size_usd, 2),
                'shares_traded': round(result.shares_traded, 4),

                # Duration
                'duration_minutes': result.duration_minutes,
                'candles_in_trade': result.total_candles_in_trade,

                # Excursions
                'max_adverse_excursion': round(result.max_adverse_excursion, 2),
                'max_favorable_excursion': round(result.max_favorable_excursion, 2),

                # Edge cases
                'gap_exit': result.gap_exit,
            }

            # Add signal details if provided
            if signals and result.signal_index < len(signals):
                signal = signals[result.signal_index]
                row.update({
                    'sweep_time': signal.timestamp_1h_sweep,
                    'event_b': signal.condition_event_b,
                    'validation': signal.condition_validation,
                    'confirmation': signal.condition_confirmation,
                    'trend_before_sweep': signal.trend_1h_before_sweep,
                    'equilibrium_level': signal.equilibrium_level,
                })

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by entry time
        if len(df) > 0:
            df = df.sort_values('entry_time').reset_index(drop=True)

        return df

    def export_csv(
        self,
        results: List[TradeResult],
        signals: Optional[List[TradeSignal]],
        filepath: str
    ) -> None:
        """
        Export trade journal to CSV file.

        Args:
            results: List of TradeResult objects
            signals: Optional list of TradeSignal objects
            filepath: Output CSV path
        """
        df = self.to_dataframe(results, signals)

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Export
        df.to_csv(filepath, index=False)
        print(f"\nTrade journal exported: {filepath}")
        print(f"  Total trades: {len(df)}")

    def print_trade_table(
        self,
        results: List[TradeResult],
        max_rows: int = 20
    ) -> None:
        """
        Print formatted trade table to console.
        """
        if not results:
            print("\nNo trades to display.")
            return

        df = self.to_dataframe(results)

        # Select key columns for display
        display_cols = [
            'symbol', 'outcome', 'pnl_dollars', 'pnl_percent',
            'trade_id', 'direction', 'entry_price',
            'exit_price', 'capital_after', 'duration_minutes'
        ]

        # Filter to only existing columns
        display_cols = [c for c in display_cols if c in df.columns]

        display_df = df[display_cols].head(max_rows)

        print("\n" + "=" * 90)
        print("TRADE JOURNAL")
        print("=" * 90)

        # Format for nicer display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(display_df.to_string(index=False))

        if len(df) > max_rows:
            print(f"\n... and {len(df) - max_rows} more trades")

        print("=" * 90)
