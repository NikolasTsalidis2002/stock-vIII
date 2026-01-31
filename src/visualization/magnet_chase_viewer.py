"""
Magnet Chase Signal Viewer — tabbed dashboard visualizer for Magnet Chase strategy signals.

Generates a single HTML file with:
- Signal dropdown + arrow key navigation
- Tabs: Detection TF | Confirmation TF | P&L | Dashboard
- Sidebar: Signal info, Target zone, Trade params, Confirmation, P&L
- D3 SVG overlay for FVG/OB zones, BOS lines, entry arrow, TP/SL

Uses LightweightCharts + D3.js (same stack as three_ob_signal_viewer.py).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategy.magnet_chase_engine import MagnetChaseSignal, MagnetChaseEngine
from src.backtesting.models import TradeResult, PerformanceMetrics
from .core import (
    NumpyEncoder, generate_candle_data, generate_bos_lines,
    generate_fvg_zones, generate_ob_zones, generate_liquidity_lines,
)
from src.indicators.smc_custom import smc_custom
from src.indicators.smc import smc


class MagnetChaseSignalViewer:
    """Generates a full tabbed dashboard viewer for Magnet Chase strategy signals."""

    def __init__(
        self,
        df_detect: pd.DataFrame,
        df_confirm: pd.DataFrame,
        signals: List[MagnetChaseSignal],
        engine: MagnetChaseEngine,
        symbol: str = 'TSLA',
        trade_results: Optional[List[TradeResult]] = None,
        performance_metrics: Optional[PerformanceMetrics] = None,
        bars_around: int = 50,
        trailing_sl_activation_pct: float = 50.0,
        min_rr_ratio: float = 0.0,
    ):
        self.symbol = symbol
        self.trailing_sl_activation_pct = trailing_sl_activation_pct
        self.min_rr_ratio = min_rr_ratio
        self.signals = signals
        self.engine = engine
        self.trade_results = trade_results or []
        self.performance_metrics = performance_metrics
        self.bars_around = bars_around

        # Detection TF dataframe
        self.df_detect = df_detect.copy()
        if 'time' in self.df_detect.columns:
            self.df_detect = self.df_detect.set_index('time')

        # Confirmation TF dataframe
        self.df_confirm = df_confirm.copy()
        if 'time' in self.df_confirm.columns:
            self.df_confirm = self.df_confirm.set_index('time')

    def generate_html(self, output_dir: Optional[str] = None) -> Path:
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            out_dir = project_root / 'results' / 'visualizations'
        else:
            out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        signals_data = self._build_signals_data()
        performance_data = self._generate_performance_data()
        htf_trades_data = self._generate_htf_trades_data()
        html = self._create_html(signals_data, performance_data, htf_trades_data)
        safe_symbol = self.symbol.replace('/', '_')
        output_path = out_dir / f'{safe_symbol.lower()}_magnet_chase_signals.html'
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"Generated: {output_path}")
        return output_path

    def _build_signals_data(self) -> List[Dict]:
        signals_data = []

        # Build lookup of trade results by entry time
        trade_result_map = {}
        for tr in self.trade_results:
            trade_result_map[tr.entry_time] = tr

        # Build reverse lookup: entry_time -> signal index
        self._entry_time_to_signal_idx = {}
        for i, sig in enumerate(self.signals):
            self._entry_time_to_signal_idx[sig.entry_time] = i

        for i, sig in enumerate(self.signals):
            entry_time = sig.entry_time
            trade_result = trade_result_map.get(entry_time, None)

            # Generate tabs
            tab_detect = self._generate_tab_detect(sig, trade_result)
            tab_confirm = self._generate_tab_confirm(sig, trade_result)
            tab_pnl = self._generate_tab_pnl(trade_result)
            tab_sl_history = self._generate_tab_sl_history(trade_result, sig)
            tab_context = self._generate_tab_context(sig, trade_result)

            # P&L info
            pnl_dollars = trade_result.pnl_dollars if trade_result else None
            pnl_percent = trade_result.pnl_percent if trade_result else None
            pnl_r = trade_result.pnl_r_multiple if trade_result else None
            exit_type = trade_result.exit_type.value if trade_result and trade_result.exit_type else None
            trade_outcome = trade_result.outcome.value if trade_result and trade_result.outcome else None

            # Label for dropdown
            direction_label = sig.entry_direction.upper()
            time_label = entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(entry_time, 'strftime') else str(entry_time)
            pnl_label = ''
            if pnl_dollars is not None:
                pnl_label = f" | ${pnl_dollars:+.2f} ({trade_outcome or ''})"
            label = f"#{i+1} {direction_label} @ ${sig.entry_price:.2f} — {time_label}{pnl_label}"

            # Conditions checklist
            conditions = [
                {'label': f'Target Zone ({sig.target_zone_type})', 'met': True, 'value': f'${sig.target_zone_bottom:.2f}-${sig.target_zone_top:.2f}'},
                {'label': f'Zone Side ({sig.target_zone_side})', 'met': True, 'value': f'dist {sig.target_distance_pct:.2f}%'},
                {'label': f'Confirmation ({sig.confirmation_type})', 'met': True, 'value': f'@ ${sig.confirmation_level:.2f}'},
                {'label': f'R:R Ratio (>={self.min_rr_ratio:.1f})', 'met': sig.rr_ratio >= self.min_rr_ratio, 'value': f'{sig.rr_ratio:.1f}'},
            ]

            signals_data.append({
                'label': label,
                'tabDetect': tab_detect,
                'tabConfirm': tab_confirm,
                'tabPnL': tab_pnl,
                'tabSLHistory': tab_sl_history,
                'tabContext': tab_context,
                'direction': sig.entry_direction,
                'entryPrice': float(sig.entry_price),
                'tpPrice': float(sig.take_profit),
                'slPrice': float(sig.stop_loss),
                'conditions': conditions,
                'pnlDollars': float(pnl_dollars) if pnl_dollars is not None else None,
                'pnlPercent': float(pnl_percent) if pnl_percent is not None else None,
                'pnlR': float(pnl_r) if pnl_r is not None else None,
                'exitType': exit_type,
                'tradeOutcome': trade_outcome,
                'timestamp': time_label,
                'entryTime': trade_result.entry_time.strftime('%Y-%m-%d %H:%M') if trade_result else time_label,
                'exitTime': trade_result.exit_time.strftime('%Y-%m-%d %H:%M') if trade_result and trade_result.exit_time else None,
                'exitPrice': float(trade_result.exit_price) if trade_result and trade_result.exit_price else None,
                # Magnet Chase specific sidebar info
                'targetZoneType': sig.target_zone_type,
                'targetZoneSide': sig.target_zone_side,
                'targetDistancePct': round(sig.target_distance_pct, 3),
                'targetRank': sig.target_rank,
                'zoneAgeBars': sig.zone_age_bars,
                'impulseAvgCandles': sig.impulse_avg_candles,
                'rrRatio': round(sig.rr_ratio, 2),
                'confirmationType': sig.confirmation_type,
                'confirmationTime': sig.confirmation_time.strftime('%Y-%m-%d %H:%M') if hasattr(sig.confirmation_time, 'strftime') else str(sig.confirmation_time),
                'confirmationLevel': round(sig.confirmation_level, 2),
            })

        return signals_data

    def _generate_tab_detect(self, sig: MagnetChaseSignal, trade_result: Optional[TradeResult]) -> Dict:
        """Detection TF tab: candlestick chart with FVG/OB zones, highlighted target, TP/SL."""
        entry_time = sig.entry_time
        detect_bar = sig.detection_bar_idx

        # Window: some bars before detection, extend past entry/exit
        pre_padding = 20
        start_idx = max(0, detect_bar - pre_padding)

        entry_idx = self.df_detect.index.get_indexer([entry_time], method='nearest')[0]
        if trade_result and trade_result.exit_time:
            exit_idx = self.df_detect.index.get_indexer([trade_result.exit_time], method='nearest')[0]
            end_idx = min(len(self.df_detect), exit_idx + 5)
        else:
            end_idx = min(len(self.df_detect), entry_idx + self.bars_around)
        end_idx = max(end_idx, entry_idx + 10)

        df_slice = self.df_detect.iloc[start_idx:end_idx]
        candle_data = generate_candle_data(df_slice)

        # Generate FVG and OB zones from engine's precomputed data
        fvg_zones = generate_fvg_zones(self.df_detect.iloc[start_idx:end_idx], self.engine.fvg_detect.iloc[start_idx:end_idx], slice_offset=start_idx)
        ob_zones = generate_ob_zones(self.df_detect.iloc[start_idx:end_idx], self.engine.ob_detect.iloc[start_idx:end_idx], slice_offset=start_idx)

        # Highlighted target zone
        target_zone = {
            'topPrice': float(sig.target_zone_top),
            'bottomPrice': float(sig.target_zone_bottom),
            'startTime': int(df_slice.index[0].timestamp()),
            'endTime': int(df_slice.index[-1].timestamp()),
            'zoneType': sig.target_zone_type,
            'direction': sig.target_zone_direction,
        }

        entry_marker = {
            'time': int(entry_time.timestamp()) if hasattr(entry_time, 'timestamp') else int(pd.Timestamp(entry_time).timestamp()),
            'price': float(sig.entry_price),
            'direction': sig.entry_direction,
        }

        return {
            'tabName': 'Detection TF',
            'candleData': candle_data,
            'fvgZones': fvg_zones,
            'obZones': ob_zones,
            'targetZone': target_zone,
            'entryMarker': entry_marker,
            'tpPrice': float(sig.take_profit),
            'slPrice': float(sig.stop_loss),
            'focusTime': int(entry_time.timestamp()) if hasattr(entry_time, 'timestamp') else int(pd.Timestamp(entry_time).timestamp()),
            'contextCandles': end_idx - start_idx,
        }

    def _generate_tab_confirm(self, sig: MagnetChaseSignal, trade_result: Optional[TradeResult]) -> Optional[Dict]:
        """Confirmation TF tab: candlestick chart with BOS lines, FVG zones, entry marker."""
        entry_time = sig.entry_time
        entry_time_ts = pd.Timestamp(entry_time)

        if self.df_confirm.index.tz is not None and entry_time_ts.tz is None:
            entry_time_ts = entry_time_ts.tz_localize(self.df_confirm.index.tz)
        elif self.df_confirm.index.tz is None and entry_time_ts.tz is not None:
            entry_time_ts = entry_time_ts.tz_localize(None)

        if len(self.df_confirm) == 0:
            return None

        try:
            entry_idx = self.df_confirm.index.get_indexer([entry_time_ts], method='nearest')[0]
        except Exception:
            return None

        if entry_idx < 0 or entry_idx >= len(self.df_confirm):
            return None

        zoom_bars = 40
        start_idx = max(0, entry_idx - zoom_bars)
        end_idx = min(len(self.df_confirm), entry_idx + zoom_bars)
        df_slice = self.df_confirm.iloc[start_idx:end_idx]

        if len(df_slice) == 0:
            return None

        # Use engine's precomputed indicators for the slice
        try:
            bos_slice = self.engine.bos_confirm.iloc[start_idx:end_idx]
            inflexions_slice = self.engine.inflexions_confirm.iloc[start_idx:end_idx]
            fvg_slice = self.engine.fvg_confirm.iloc[start_idx:end_idx]

            bos_lines = generate_bos_lines(df_slice, bos_slice, inflexions_slice)
            fvg_zones = generate_fvg_zones(df_slice, fvg_slice, slice_offset=start_idx)
        except Exception:
            bos_lines = []
            fvg_zones = []

        candle_data = generate_candle_data(df_slice)

        entry_marker = {
            'time': int(entry_time.timestamp()) if hasattr(entry_time, 'timestamp') else int(pd.Timestamp(entry_time).timestamp()),
            'price': float(sig.entry_price),
            'direction': sig.entry_direction,
        }

        return {
            'tabName': 'Confirm TF',
            'candleData': candle_data,
            'bosLines': bos_lines,
            'fvgZones': fvg_zones,
            'entryMarker': entry_marker,
            'tpPrice': float(sig.take_profit),
            'slPrice': float(sig.stop_loss),
            'focusTime': int(entry_time.timestamp()) if hasattr(entry_time, 'timestamp') else int(pd.Timestamp(entry_time).timestamp()),
            'contextCandles': 40,
        }

    def _generate_tab_pnl(self, trade_result: Optional[TradeResult]) -> Optional[Dict]:
        if not trade_result:
            return None
        if not trade_result.unrealized_pnl_series or len(trade_result.unrealized_pnl_series) == 0:
            return None

        pnl_series = []
        for ts, pnl in trade_result.unrealized_pnl_series:
            pnl_series.append({
                'time': int(ts.timestamp()),
                'value': float(pnl)
            })

        tp_dollars = trade_result.reward_amount()
        sl_dollars = -trade_result.risk_amount()

        return {
            'tabName': 'P&L',
            'pnlSeries': pnl_series,
            'tpDollars': tp_dollars,
            'slDollars': sl_dollars,
            'exitPnl': trade_result.pnl_dollars,
            'exitTime': int(trade_result.exit_time.timestamp()) if trade_result.exit_time else None,
            'direction': trade_result.entry_direction,
            'outcome': trade_result.outcome.value if trade_result.outcome else None,
            'exitType': trade_result.exit_type.value if trade_result.exit_type else None,
        }

    def _generate_tab_sl_history(self, trade_result: Optional[TradeResult], sig: MagnetChaseSignal) -> Optional[Dict]:
        if not trade_result:
            return None
        if not trade_result.sl_history or len(trade_result.sl_history) == 0:
            return None

        sl_entries = []
        for i, (ts, price, reason) in enumerate(trade_result.sl_history):
            is_last = (i == len(trade_result.sl_history) - 1)
            sl_entries.append({
                'time': int(ts.timestamp()),
                'price': float(price),
                'reason': reason,
                'isActive': is_last,
            })

        exit_time_unix = int(trade_result.exit_time.timestamp()) if trade_result.exit_time else None

        # Use confirmation TF for finer-grained SL visualization
        sl_df = self.df_confirm
        entry_time = sig.entry_time
        entry_idx = sl_df.index.get_indexer([entry_time], method='nearest')[0]
        pre_padding = 10
        start_idx = max(0, entry_idx - pre_padding)

        if trade_result.exit_time:
            exit_idx = sl_df.index.get_indexer([trade_result.exit_time], method='nearest')[0]
            post_padding = 10
            end_idx = min(len(sl_df), exit_idx + post_padding)
        else:
            end_idx = min(len(sl_df), entry_idx + 60)

        end_idx = max(end_idx, entry_idx + 20)
        df_slice = sl_df.iloc[start_idx:end_idx]
        candle_data = generate_candle_data(df_slice)

        entry_price = trade_result.entry_price
        tp_price = trade_result.take_profit_price
        if trade_result.entry_direction == 'long':
            activation_price = entry_price + (tp_price - entry_price) * (self.trailing_sl_activation_pct / 100.0)
        else:
            activation_price = entry_price - (entry_price - tp_price) * (self.trailing_sl_activation_pct / 100.0)

        return {
            'tabName': 'Stop Loss',
            'slHistory': sl_entries,
            'entryPrice': float(trade_result.entry_price),
            'direction': trade_result.entry_direction,
            'exitTime': exit_time_unix,
            'exitType': trade_result.exit_type.value if trade_result.exit_type else None,
            'candleData': candle_data,
            'activationPrice': float(activation_price),
            'activationPct': self.trailing_sl_activation_pct,
            'tpPrice': float(tp_price),
        }

    def _generate_tab_context(self, sig: MagnetChaseSignal, trade_result: Optional[TradeResult]) -> Dict:
        """Wide-context view on detection TF: all FVG/OB zones, target zone, entry/exit markers."""
        entry_time = sig.entry_time
        detect_bar = sig.detection_bar_idx

        # Window: start well before detection, extend past exit
        pre_padding = 50
        start_idx = max(0, detect_bar - pre_padding)

        entry_idx = self.df_detect.index.get_indexer([entry_time], method='nearest')[0]
        if trade_result and trade_result.exit_time:
            exit_idx = self.df_detect.index.get_indexer([trade_result.exit_time], method='nearest')[0]
            end_idx = min(len(self.df_detect), exit_idx + 10)
        else:
            end_idx = min(len(self.df_detect), entry_idx + 30)
        end_idx = max(end_idx, entry_idx + 20)

        df_slice = self.df_detect.iloc[start_idx:end_idx]
        candle_data = generate_candle_data(df_slice)

        # All FVG and OB zones on the slice
        fvg_zones = generate_fvg_zones(df_slice, self.engine.fvg_detect.iloc[start_idx:end_idx], slice_offset=start_idx)
        ob_zones = generate_ob_zones(df_slice, self.engine.ob_detect.iloc[start_idx:end_idx], slice_offset=start_idx)

        # Compute all OBs from raw indicators for context overlay
        all_obs = []
        try:
            ob_data = self.engine.ob_detect.iloc[start_idx:end_idx]
            ob_vals = ob_data['OB'].values
            ob_tops = ob_data['Top'].values
            ob_bottoms = ob_data['Bottom'].values
            ob_starts = ob_data['StartIndex'].values
            ob_mitigated = ob_data['MitigatedIndex'].values

            ob_respected = ob_data['Respected'].values
            ob_status = ob_data['StatusIndex'].values

            for j in range(len(ob_data)):
                if np.isnan(ob_vals[j]):
                    continue
                s_idx = int(ob_starts[j])
                mit_idx = int(ob_mitigated[j]) if ob_mitigated[j] and not np.isnan(float(ob_mitigated[j])) else 0

                # Determine zone end: use StatusIndex for disrespected, MitigatedIndex for mitigated
                status_val = int(ob_status[j]) if not np.isnan(float(ob_status[j])) else 0
                respected_val = ob_respected[j]

                if respected_val == False and status_val > 0:
                    zone_end_abs = status_val
                elif mit_idx > 0:
                    zone_end_abs = mit_idx
                else:
                    zone_end_abs = start_idx + len(df_slice) - 1

                ob_start_abs = max(0, min(s_idx - start_idx, len(df_slice) - 1))
                ob_end_abs = max(0, min(zone_end_abs - start_idx, len(df_slice) - 1))

                satisfied = bool(mit_idx > 0)
                all_obs.append({
                    'topPrice': float(ob_tops[j]),
                    'bottomPrice': float(ob_bottoms[j]),
                    'startTime': int(df_slice.index[ob_start_abs].timestamp()),
                    'endTime': int(df_slice.index[ob_end_abs].timestamp()),
                    'satisfied': satisfied,
                    'direction': int(ob_vals[j]),
                })
        except Exception:
            pass

        # Highlighted target zone
        target_zone = {
            'topPrice': float(sig.target_zone_top),
            'bottomPrice': float(sig.target_zone_bottom),
            'startTime': int(df_slice.index[0].timestamp()),
            'endTime': int(df_slice.index[-1].timestamp()),
            'zoneType': sig.target_zone_type,
            'direction': sig.target_zone_direction,
        }

        # Entry marker
        entry_marker = {
            'time': int(entry_time.timestamp()) if hasattr(entry_time, 'timestamp') else int(pd.Timestamp(entry_time).timestamp()),
            'price': float(sig.entry_price),
            'direction': sig.entry_direction,
        }

        # Exit marker
        exit_marker = None
        if trade_result and trade_result.exit_time and trade_result.exit_price:
            exit_marker = {
                'time': int(trade_result.exit_time.timestamp()),
                'price': float(trade_result.exit_price),
                'direction': sig.entry_direction,
                'exitType': trade_result.exit_type.value if trade_result.exit_type else None,
                'isWin': trade_result.pnl_dollars >= 0 if trade_result.pnl_dollars is not None else False,
            }

        # Focus window for zoom-in view
        focus_start_idx = max(0, detect_bar - 10)
        if trade_result and trade_result.exit_time:
            focus_exit_idx = self.df_detect.index.get_indexer([trade_result.exit_time], method='nearest')[0]
            focus_end_idx = min(len(self.df_detect) - 1, focus_exit_idx + 5)
        else:
            focus_end_idx = min(len(self.df_detect) - 1, entry_idx + self.bars_around)
        focus_end_idx = max(focus_end_idx, entry_idx + 10)
        focus_end_idx = min(focus_end_idx, len(self.df_detect) - 1)
        focus_start_idx = min(focus_start_idx, len(self.df_detect) - 1)

        return {
            'candleData': candle_data,
            'fvgZones': fvg_zones,
            'obZones': ob_zones,
            'allOBs': all_obs,
            'targetZone': target_zone,
            'entryMarker': entry_marker,
            'exitMarker': exit_marker,
            'tpPrice': float(sig.take_profit),
            'slPrice': float(sig.stop_loss),
            'focusTime': int(entry_time.timestamp()) if hasattr(entry_time, 'timestamp') else int(pd.Timestamp(entry_time).timestamp()),
            'contextCandles': end_idx - start_idx,
            'focusWindow': {
                'startTime': int(self.df_detect.index[focus_start_idx].timestamp()),
                'endTime': int(self.df_detect.index[focus_end_idx].timestamp()),
            },
        }

    def _generate_htf_trades_data(self) -> Optional[Dict]:
        """Full detection TF chart with ALL trades marked (for zoom-out view)."""
        if not self.trade_results:
            return None

        candle_data = generate_candle_data(self.df_detect)
        if not candle_data:
            return None

        markers = []
        trade_lines = []

        for tr in self.trade_results:
            is_long = tr.entry_direction == 'long'
            is_win = tr.pnl_dollars >= 0 if tr.pnl_dollars is not None else False
            win_color = '#089981'
            loss_color = '#f23645'
            outcome_color = win_color if is_win else loss_color

            entry_ts = int(tr.entry_time.timestamp()) if hasattr(tr.entry_time, 'timestamp') else int(pd.Timestamp(tr.entry_time).timestamp())

            markers.append({
                'time': entry_ts,
                'position': 'belowBar' if is_long else 'aboveBar',
                'color': win_color if is_long else loss_color,
                'shape': 'arrowUp' if is_long else 'arrowDown',
                'text': f'${tr.entry_price:.0f}',
            })

            if tr.exit_time and tr.exit_price:
                exit_ts = int(tr.exit_time.timestamp()) if hasattr(tr.exit_time, 'timestamp') else int(pd.Timestamp(tr.exit_time).timestamp())
                label_map = {'tp_hit': 'TP', 'sl_hit': 'SL', 'trailing_sl': 'TSL', 'timeout': 'TIME'}
                exit_label = label_map.get(tr.exit_type.value, tr.exit_type.value.upper()) if tr.exit_type else ''

                markers.append({
                    'time': exit_ts,
                    'position': 'aboveBar' if is_long else 'belowBar',
                    'color': outcome_color,
                    'shape': 'circle',
                    'text': exit_label,
                })

                trade_lines.append({
                    'entryTime': entry_ts,
                    'exitTime': exit_ts,
                    'entryPrice': float(tr.entry_price),
                    'exitPrice': float(tr.exit_price),
                    'color': outcome_color,
                    'isWin': is_win,
                })

        markers.sort(key=lambda m: m['time'])

        # Consecutive win/loss clusters
        trade_clusters = []
        if trade_lines:
            sorted_trades = sorted(trade_lines, key=lambda t: t['entryTime'])
            streak_type = sorted_trades[0]['isWin']
            streak_start = 0
            for i in range(1, len(sorted_trades)):
                if sorted_trades[i]['isWin'] != streak_type:
                    streak_len = i - streak_start
                    if streak_len >= 1:
                        trade_clusters.append({
                            'startTime': sorted_trades[streak_start]['entryTime'],
                            'endTime': sorted_trades[i - 1]['exitTime'],
                            'isWin': streak_type,
                            'count': streak_len,
                        })
                    streak_type = sorted_trades[i]['isWin']
                    streak_start = i
            streak_len = len(sorted_trades) - streak_start
            if streak_len >= 1:
                trade_clusters.append({
                    'startTime': sorted_trades[streak_start]['entryTime'],
                    'endTime': sorted_trades[-1]['exitTime'],
                    'isWin': streak_type,
                    'count': streak_len,
                })

        # BOS lines and liquidity on full detect TF
        bos_lines = []
        liquidity_lines = []
        try:
            bos_lines = generate_bos_lines(self.df_detect, self.engine.bos_detect, self.engine.inflexions_detect)
            liquidity_lines = generate_liquidity_lines(self.df_detect, self.engine.inflexions_detect)
        except Exception:
            pass

        return {
            'candleData': candle_data,
            'markers': markers,
            'tradeLines': trade_lines,
            'bosLines': bos_lines,
            'liquidityLines': liquidity_lines,
            'tradeClusters': trade_clusters,
        }

    def _generate_performance_data(self) -> Optional[Dict]:
        if not self.performance_metrics:
            return None

        metrics = self.performance_metrics

        equity_curve_data = []
        if metrics.equity_curve:
            for i, capital in enumerate(metrics.equity_curve):
                equity_curve_data.append({'trade': i, 'capital': float(capital)})

        profit_factor = metrics.profit_factor
        if profit_factor == float('inf'):
            profit_factor_str = "\u221e"
            profit_factor_num = None
        else:
            profit_factor_str = f"{profit_factor:.2f}"
            profit_factor_num = float(profit_factor)

        source_df = self.df_confirm
        analysis_start = source_df.index.min().strftime('%Y-%m-%d')
        analysis_end = source_df.index.max().strftime('%Y-%m-%d')

        start_date = source_df.index.min().date()
        end_date = source_df.index.max().date()
        trading_days = int(np.busday_count(start_date, end_date))
        if trading_days > 0:
            return_per_day = float(metrics.total_return_percent) / trading_days
        else:
            return_per_day = 0.0

        return {
            'analysisStart': analysis_start,
            'analysisEnd': analysis_end,
            'totalReturn': float(metrics.total_return_percent),
            'returnPerDay': return_per_day,
            'tradingDays': trading_days,
            'winRate': float(metrics.win_rate * 100),
            'profitFactor': profit_factor_num,
            'profitFactorStr': profit_factor_str,
            'maxDrawdownPercent': float(metrics.max_drawdown_percent),
            'maxDrawdownDollars': float(metrics.max_drawdown_dollars),
            'totalTrades': int(metrics.total_trades),
            'winningTrades': int(metrics.winning_trades),
            'losingTrades': int(metrics.losing_trades),
            'tpExits': int(metrics.tp_exits),
            'slExits': int(metrics.sl_exits),
            'timeoutExits': int(metrics.timeout_exits),
            'trailingSlExits': int(metrics.trailing_sl_exits),
            'avgRMultiple': float(metrics.average_r_multiple),
            'maxConsecWins': int(metrics.max_consecutive_wins),
            'maxConsecLosses': int(metrics.max_consecutive_losses),
            'avgWinDollars': float(metrics.average_win_dollars),
            'avgLossDollars': float(metrics.average_loss_dollars),
            'largestWin': float(metrics.largest_win_dollars),
            'largestLoss': float(metrics.largest_loss_dollars),
            'initialCapital': float(metrics.initial_capital),
            'finalCapital': float(metrics.final_capital),
            'totalPnl': float(metrics.total_pnl_dollars),
            'longTrades': int(metrics.long_trades),
            'shortTrades': int(metrics.short_trades),
            'longWins': int(metrics.long_wins),
            'shortWins': int(metrics.short_wins),
            'impulseAvgFVG': round(self.engine.impulse_tracker.avg('FVG'), 1),
            'impulseAvgOB': round(self.engine.impulse_tracker.avg('OB'), 1),
            'equityCurve': equity_curve_data,
            'trades': [
                {
                    'tradeNum': i + 1,
                    'entryTime': result.entry_time.strftime('%Y-%m-%d %H:%M'),
                    'exitTime': result.exit_time.strftime('%Y-%m-%d %H:%M') if result.exit_time else '-',
                    'direction': result.entry_direction,
                    'entryPrice': float(result.entry_price),
                    'exitPrice': float(result.exit_price) if result.exit_price else None,
                    'pnlDollars': float(result.pnl_dollars),
                    'pnlPercent': float(result.pnl_percent),
                    'pnlR': float(result.pnl_r_multiple),
                    'outcome': result.outcome.value if result.outcome else None,
                    'exitType': result.exit_type.value if result.exit_type else None,
                    'durationMin': result.duration_minutes,
                    'signalIdx': self._entry_time_to_signal_idx.get(result.entry_time, -1),
                }
                for i, result in enumerate(self.trade_results)
            ]
        }

    def _create_html(self, signals_data: List[Dict], performance_data: Optional[Dict], htf_trades_data: Optional[Dict] = None) -> str:
        data_json = json.dumps(signals_data, cls=NumpyEncoder)
        performance_json = json.dumps(performance_data, cls=NumpyEncoder) if performance_data else 'null'
        htf_trades_json = json.dumps(htf_trades_data, cls=NumpyEncoder) if htf_trades_data else 'null'

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.symbol} Magnet Chase Signal Viewer</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #131722;
            color: #d1d4dc;
            height: 100vh;
            overflow: hidden;
        }}
        #app {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        #header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 20px;
            background-color: #1e222d;
            border-bottom: 1px solid #2b2b43;
        }}
        #header h1 {{
            color: #2962ff;
            font-size: 18px;
            font-weight: 600;
        }}
        .header-controls {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .signal-selector {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .signal-selector label {{
            color: #787b86;
            font-size: 13px;
        }}
        #signal-select {{
            padding: 8px 12px;
            background-color: #131722;
            border: 1px solid #363a45;
            color: #d1d4dc;
            border-radius: 4px;
            font-size: 13px;
            min-width: 350px;
            cursor: pointer;
        }}
        #signal-select:focus {{ outline: none; border-color: #2962ff; }}
        .signal-count {{
            color: #d1d4dc;
            font-size: 13px;
            min-width: 80px;
            text-align: center;
        }}
        #tabs {{
            display: flex;
            background-color: #1e222d;
            border-bottom: 1px solid #2b2b43;
            padding: 0 20px;
        }}
        .tab {{
            padding: 12px 24px;
            color: #787b86;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }}
        .tab:hover {{ color: #d1d4dc; }}
        .tab.active {{ color: #2962ff; border-bottom-color: #2962ff; }}
        .tab.disabled {{ color: #363a45; cursor: not-allowed; }}
        .tab.disabled:hover {{ color: #363a45; }}
        #main {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        #chart-area {{
            flex: 1;
            position: relative;
        }}
        #chart-container {{
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
        }}
        #svg-overlay {{
            position: absolute;
            top: 0; left: 0;
            pointer-events: none;
            z-index: 10;
        }}
        #dashboard-container {{
            display: none;
            padding: 24px;
            overflow-y: auto;
            height: 100%;
        }}
        #dashboard-container.active {{ display: block; }}
        #sidebar {{
            width: 280px;
            background-color: #1e222d;
            border-left: 1px solid #2b2b43;
            padding: 20px;
            overflow-y: auto;
        }}
        .sidebar-section {{
            margin-bottom: 24px;
        }}
        .sidebar-section h3 {{
            color: #787b86;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        .info-label {{ color: #787b86; }}
        .info-value {{ color: #d1d4dc; font-weight: 500; }}
        .info-value.success {{ color: #089981; }}
        .info-value.failed {{ color: #f23645; }}
        .info-value.long {{ color: #089981; }}
        .info-value.short {{ color: #f23645; }}
        .condition-item {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            font-size: 13px;
        }}
        .condition-check {{
            width: 20px; height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            font-size: 12px;
        }}
        .condition-check.met {{
            background-color: rgba(8, 153, 129, 0.2);
            color: #089981;
        }}
        .condition-check.not-met {{
            background-color: rgba(242, 54, 69, 0.2);
            color: #f23645;
        }}
        .condition-label {{ flex: 1; }}
        .condition-value {{ color: #787b86; font-size: 11px; }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 12px;
        }}
        .legend-color {{
            width: 16px; height: 16px;
            margin-right: 8px;
            border-radius: 2px;
        }}
        .shortcuts {{
            margin-top: 20px;
            padding-top: 16px;
            border-top: 1px solid #2b2b43;
        }}
        .shortcut-item {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 11px;
            color: #787b86;
        }}
        .shortcut-key {{
            background-color: #131722;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }}
        /* Dashboard Styles */
        .dashboard-no-data {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #787b86;
            font-size: 16px;
        }}
        .analysis-period {{
            text-align: center;
            padding: 12px 0;
            margin-bottom: 16px;
            color: #9db2c8;
            font-size: 14px;
        }}
        .analysis-period .period-label {{ color: #6e7a8a; }}
        .analysis-period .period-value {{ color: #d1d4dc; font-weight: 500; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background-color: #1e222d;
            border: 1px solid #2b2b43;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-card.positive {{ border-color: rgba(8, 153, 129, 0.4); }}
        .kpi-card.negative {{ border-color: rgba(242, 54, 69, 0.4); }}
        .kpi-label {{
            color: #787b86;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .kpi-value {{ font-size: 28px; font-weight: 600; }}
        .kpi-value.positive {{ color: #089981; }}
        .kpi-value.negative {{ color: #f23645; }}
        .kpi-value.neutral {{ color: #d1d4dc; }}
        .equity-chart-container {{
            background-color: #1e222d;
            border: 1px solid #2b2b43;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
            height: 300px;
        }}
        .equity-chart-title {{
            color: #787b86;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}
        #equity-chart {{ width: 100%; height: calc(100% - 30px); }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }}
        .stats-card {{
            background-color: #1e222d;
            border: 1px solid #2b2b43;
            border-radius: 8px;
            padding: 16px;
        }}
        .stats-card-title {{
            color: #787b86;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2b2b43;
        }}
        .stats-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        .stats-row:last-child {{ margin-bottom: 0; }}
        .stats-label {{ color: #787b86; }}
        .stats-value {{ color: #d1d4dc; font-weight: 500; }}
        .stats-value.positive {{ color: #089981; }}
        .stats-value.negative {{ color: #f23645; }}
        .transactions-container {{
            margin-top: 20px;
            background-color: #1e222d;
            border: 1px solid #2b2b43;
            border-radius: 8px;
            overflow: hidden;
        }}
        .transactions-header {{
            padding: 12px 16px;
            border-bottom: 1px solid #2b2b43;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .transactions-title {{ color: #d1d4dc; font-size: 14px; font-weight: 600; }}
        .transactions-count {{ color: #787b86; font-size: 12px; }}
        .transactions-table-wrapper {{
            max-height: 400px;
            overflow-y: auto;
        }}
        .transactions-table-wrapper::-webkit-scrollbar {{ width: 8px; }}
        .transactions-table-wrapper::-webkit-scrollbar-track {{ background: #1e222d; }}
        .transactions-table-wrapper::-webkit-scrollbar-thumb {{ background: #363a45; border-radius: 4px; }}
        .transactions-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .transactions-table thead {{ position: sticky; top: 0; z-index: 1; }}
        .transactions-table th {{
            background-color: #131722;
            color: #787b86;
            font-weight: 500;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #2b2b43;
        }}
        .transactions-table th.sortable {{ cursor: pointer; user-select: none; }}
        .transactions-table th.sortable:hover {{ color: #d1d4dc; }}
        .transactions-table th .sort-indicator {{ margin-left: 4px; opacity: 0.5; }}
        .transactions-table th.sort-asc .sort-indicator::after {{ content: '\u25b2'; }}
        .transactions-table th.sort-desc .sort-indicator::after {{ content: '\u25bc'; }}
        .transactions-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #2b2b43;
            color: #d1d4dc;
        }}
        .transactions-table tbody tr {{ cursor: pointer; }}
        .transactions-table tr.win-row {{ background-color: rgba(8, 153, 129, 0.05); }}
        .transactions-table tr.win-row:hover {{ background-color: rgba(8, 153, 129, 0.1); }}
        .transactions-table tr.loss-row {{ background-color: rgba(242, 54, 69, 0.05); }}
        .transactions-table tr.loss-row:hover {{ background-color: rgba(242, 54, 69, 0.1); }}
        .trade-num {{ color: #787b86; }}
        .direction-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .direction-badge.long {{ background-color: rgba(8, 153, 129, 0.2); color: #089981; }}
        .direction-badge.short {{ background-color: rgba(242, 54, 69, 0.2); color: #f23645; }}
        .pnl-value {{ font-weight: 500; }}
        .pnl-value.positive {{ color: #089981; }}
        .pnl-value.negative {{ color: #f23645; }}
        .outcome-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .outcome-badge.win {{ background-color: rgba(8, 153, 129, 0.2); color: #089981; }}
        .outcome-badge.loss {{ background-color: rgba(242, 54, 69, 0.2); color: #f23645; }}
        .exit-type-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }}
        .exit-type-badge.tp {{ background-color: rgba(8, 153, 129, 0.15); color: #089981; }}
        .exit-type-badge.sl {{ background-color: rgba(242, 54, 69, 0.15); color: #f23645; }}
        .exit-type-badge.timeout {{ background-color: rgba(120, 123, 134, 0.15); color: #787b86; }}
    </style>
</head>
<body>
    <div id="app">
        <div id="header">
            <h1>{self.symbol} MAGNET CHASE SIGNALS</h1>
            <div class="header-controls">
                <div class="signal-selector">
                    <label>Signal:</label>
                    <select id="signal-select"></select>
                </div>
                <span class="signal-count" id="signal-count">{len(signals_data)} signals</span>
            </div>
        </div>

        <div id="tabs">
            <div class="tab" data-tab="DETECT" onclick="switchTab('DETECT')">Detection TF</div>
            <div class="tab" data-tab="CONFIRM" onclick="switchTab('CONFIRM')">Confirm TF</div>
            <div class="tab" data-tab="PNL" onclick="switchTab('PNL')">P&L</div>
            <div class="tab" data-tab="SL_HISTORY" onclick="switchTab('SL_HISTORY')">Stop Loss</div>
            <div class="tab" data-tab="CONTEXT" onclick="switchTab('CONTEXT')">Context</div>
            <div class="tab" data-tab="DASHBOARD" onclick="switchTab('DASHBOARD')">Dashboard</div>
        </div>

        <div id="main">
            <div id="chart-area">
                <button id="context-zoom-toggle" style="display:none; position:absolute; top:8px; right:8px; z-index:10; padding:4px 12px; cursor:pointer; background:#2a2a3e; color:#e0e0e0; border:1px solid #555; border-radius:4px; font-size:13px;" onclick="toggleContextZoom()">Zoom Out</button>
                <div id="cluster-threshold-control" style="display:none; position:absolute; top:8px; right:140px; z-index:10; background:#2a2a3e; color:#e0e0e0; border:1px solid #555; border-radius:4px; padding:4px 8px; font-size:13px;">
                    Cluster >= <input id="cluster-threshold" type="number" min="2" max="20" value="3" style="width:36px; background:#1a1a2e; color:#e0e0e0; border:1px solid #555; border-radius:3px; text-align:center; font-size:13px;" onchange="onClusterThresholdChange()">
                </div>
                <div id="chart-container"></div>
                <div id="dashboard-container"></div>
            </div>

            <div id="sidebar">
                <div class="sidebar-section">
                    <h3>Signal Information</h3>
                    <div class="info-row">
                        <span class="info-label">Time</span>
                        <span class="info-value" id="signal-time">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Direction</span>
                        <span class="info-value" id="signal-direction">-</span>
                    </div>
                </div>

                <div class="sidebar-section" id="target-zone-info">
                    <h3>Target Zone</h3>
                    <div class="info-row">
                        <span class="info-label">Type</span>
                        <span class="info-value" id="zone-type">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Side</span>
                        <span class="info-value" id="zone-side">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Distance</span>
                        <span class="info-value" id="zone-distance">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Rank</span>
                        <span class="info-value" id="zone-rank">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Zone Age</span>
                        <span class="info-value" id="zone-age">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Impulse Avg</span>
                        <span class="info-value" id="impulse-avg">-</span>
                    </div>
                </div>

                <div class="sidebar-section" id="trade-info" style="display: none;">
                    <h3>Trade Parameters</h3>
                    <div class="info-row">
                        <span class="info-label">Entry</span>
                        <span class="info-value" id="entry-price">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Take Profit</span>
                        <span class="info-value success" id="tp-price">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Stop Loss</span>
                        <span class="info-value failed" id="sl-price">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">R:R Ratio</span>
                        <span class="info-value" id="rr-ratio">-</span>
                    </div>
                </div>

                <div class="sidebar-section" id="confirm-info">
                    <h3>Confirmation</h3>
                    <div class="info-row">
                        <span class="info-label">Type</span>
                        <span class="info-value" id="confirm-type">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Time</span>
                        <span class="info-value" id="confirm-time">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Level</span>
                        <span class="info-value" id="confirm-level">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Min Wait</span>
                        <span class="info-value" id="confirm-min-wait">-</span>
                    </div>
                </div>

                <div class="sidebar-section" id="pnl-info" style="display: none;">
                    <h3>P&L Summary</h3>
                    <div id="pnl-content"></div>
                </div>

                <div class="sidebar-section">
                    <h3>Conditions</h3>
                    <div id="conditions-list"></div>
                </div>

                <div class="sidebar-section">
                    <h3>Legend</h3>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgba(255,235,59,0.2); border: 2px solid #ffeb3b;"></div>
                        <span>FVG Zone</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgba(156,39,176,0.2); border: 2px solid #9c27b0;"></div>
                        <span>OB Zone</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgba(255,152,0,0.4); border: 3px solid #ff9800;"></div>
                        <span>Target Zone</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: none; border-top: 2px solid #089981; height: 0; margin-top: 7px;"></div>
                        <span>Take Profit</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: none; border-top: 2px dashed #f23645; height: 0; margin-top: 7px;"></div>
                        <span>Stop Loss</span>
                    </div>
                </div>

                <div class="shortcuts">
                    <div class="shortcut-item">
                        <span>Previous signal</span>
                        <span class="shortcut-key">&#8592;</span>
                    </div>
                    <div class="shortcut-item">
                        <span>Next signal</span>
                        <span class="shortcut-key">&#8594;</span>
                    </div>
                    <div class="shortcut-item">
                        <span>Switch tab</span>
                        <span class="shortcut-key">1-6</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const signalsData = {data_json};
        const performanceData = {performance_json};
        const htfTradesData = {htf_trades_json};

        let currentSignalIdx = 0;
        let currentTab = 'DETECT';
        let chart = null;
        let equityChart = null;
        let candlestickSeries = null;
        let activeLineSeries = [];
        let dashboardRendered = false;
        let contextZoomedOut = false;
        let clusterThreshold = 3;

        const chartOptions = {{
            layout: {{
                background: {{ color: '#131722' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#1e222d' }},
                horzLines: {{ color: '#1e222d' }},
            }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            rightPriceScale: {{
                borderColor: '#2b2b43',
                scaleMargins: {{ top: 0.15, bottom: 0.15 }},
                autoScale: true,
            }},
            timeScale: {{
                borderColor: '#2b2b43',
                timeVisible: true,
                secondsVisible: false,
            }},
        }};

        function init() {{
            const chartContainer = document.getElementById('chart-container');
            chart = LightweightCharts.createChart(chartContainer, chartOptions);
            candlestickSeries = chart.addCandlestickSeries({{
                upColor: '#089981',
                downColor: '#f23645',
                borderVisible: false,
                wickUpColor: '#089981',
                wickDownColor: '#f23645',
            }});

            d3.select(chartContainer)
                .append('svg')
                .attr('id', 'svg-overlay')
                .style('position', 'absolute')
                .style('top', '0')
                .style('left', '0')
                .style('pointer-events', 'none');

            const select = document.getElementById('signal-select');
            signalsData.forEach((s, i) => {{
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = s.label;
                select.appendChild(opt);
            }});

            select.addEventListener('change', (e) => {{
                currentSignalIdx = parseInt(e.target.value);
                showSignal(currentSignalIdx);
            }});

            document.addEventListener('keydown', (e) => {{
                if (e.key === 'ArrowLeft' && currentSignalIdx > 0) {{
                    select.value = currentSignalIdx - 1;
                    showSignal(currentSignalIdx - 1);
                }}
                if (e.key === 'ArrowRight' && currentSignalIdx < signalsData.length - 1) {{
                    select.value = currentSignalIdx + 1;
                    showSignal(currentSignalIdx + 1);
                }}
                if (e.key === '1') switchTab('DETECT');
                if (e.key === '2') switchTab('CONFIRM');
                if (e.key === '3') switchTab('PNL');
                if (e.key === '4') switchTab('SL_HISTORY');
                if (e.key === '5') switchTab('CONTEXT');
                if (e.key === '6') switchTab('DASHBOARD');
            }});

            const resizeObserver = new ResizeObserver(() => {{
                chart.applyOptions({{
                    width: chartContainer.clientWidth,
                    height: chartContainer.clientHeight
                }});
                redrawOverlays();
            }});
            resizeObserver.observe(chartContainer);

            chart.timeScale().subscribeVisibleLogicalRangeChange(redrawOverlays);

            if (signalsData.length > 0) showSignal(0);
        }}

        function showSignal(idx) {{
            if (idx < 0 || idx >= signalsData.length) return;
            currentSignalIdx = idx;
            const sig = signalsData[idx];

            document.getElementById('signal-select').value = idx;
            document.getElementById('signal-count').textContent =
                `${{idx + 1}} / ${{signalsData.length}} signals`;

            // Sidebar: signal info
            document.getElementById('signal-time').textContent = sig.timestamp;
            const dirEl = document.getElementById('signal-direction');
            dirEl.textContent = sig.direction.toUpperCase();
            dirEl.className = 'info-value ' + sig.direction;

            // Target zone info
            document.getElementById('zone-type').textContent = sig.targetZoneType;
            document.getElementById('zone-side').textContent = sig.targetZoneSide;
            document.getElementById('zone-distance').textContent = sig.targetDistancePct + '%';
            document.getElementById('zone-rank').textContent = '#' + sig.targetRank;
            document.getElementById('zone-age').textContent = sig.zoneAgeBars + ' bars';
            document.getElementById('impulse-avg').textContent = sig.impulseAvgCandles > 0 ? sig.impulseAvgCandles.toFixed(1) + ' bars' : 'N/A';

            // Trade params
            const tradeInfo = document.getElementById('trade-info');
            if (sig.entryPrice) {{
                tradeInfo.style.display = 'block';
                document.getElementById('entry-price').textContent = '$' + sig.entryPrice.toFixed(2);
                document.getElementById('tp-price').textContent = sig.tpPrice ? '$' + sig.tpPrice.toFixed(2) : '-';
                document.getElementById('sl-price').textContent = sig.slPrice ? '$' + sig.slPrice.toFixed(2) : '-';
                document.getElementById('rr-ratio').textContent = sig.rrRatio + ':1';
            }} else {{
                tradeInfo.style.display = 'none';
            }}

            // Confirmation info
            document.getElementById('confirm-type').textContent = sig.confirmationType;
            document.getElementById('confirm-time').textContent = sig.confirmationTime;
            document.getElementById('confirm-level').textContent = '$' + sig.confirmationLevel.toFixed(2);
            document.getElementById('confirm-min-wait').textContent = sig.impulseAvgCandles > 0 ? sig.impulseAvgCandles + ' bars' : 'N/A';

            // P&L info
            const pnlInfo = document.getElementById('pnl-info');
            const pnlContent = document.getElementById('pnl-content');
            if (sig.pnlDollars !== null && sig.pnlDollars !== undefined) {{
                pnlInfo.style.display = 'block';
                const isWin = sig.pnlDollars >= 0;
                const cls = isWin ? 'success' : 'failed';
                const exitTypeMap = {{
                    'tp_hit': 'Take Profit', 'sl_hit': 'Stop Loss',
                    'timeout': 'Timeout', 'trailing_sl': 'Trailing SL'
                }};
                pnlContent.innerHTML = `
                    <div class="info-row">
                        <span class="info-label">Outcome</span>
                        <span class="info-value ${{cls}}">${{(sig.tradeOutcome || '-').toUpperCase()}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Exit Type</span>
                        <span class="info-value">${{exitTypeMap[sig.exitType] || sig.exitType || '-'}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">P&L</span>
                        <span class="info-value ${{cls}}">${{(sig.pnlDollars >= 0 ? '+' : '') + '$' + sig.pnlDollars.toFixed(2)}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">P&L %</span>
                        <span class="info-value ${{cls}}">${{(sig.pnlPercent >= 0 ? '+' : '') + sig.pnlPercent.toFixed(2) + '%'}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">R-Multiple</span>
                        <span class="info-value ${{cls}}">${{(sig.pnlR >= 0 ? '+' : '') + sig.pnlR.toFixed(2) + 'R'}}</span>
                    </div>
                `;
            }} else {{
                pnlInfo.style.display = 'block';
                pnlContent.innerHTML = '<div style="color: #6b7280; font-size: 12px; font-style: italic; text-align: center; padding: 8px 0;">No backtest data</div>';
            }}

            // Conditions
            const condList = document.getElementById('conditions-list');
            condList.innerHTML = '';
            sig.conditions.forEach(cond => {{
                const item = document.createElement('div');
                item.className = 'condition-item';
                item.innerHTML = `
                    <div class="condition-check ${{cond.met ? 'met' : 'not-met'}}">
                        ${{cond.met ? '&#10003;' : '&#10007;'}}
                    </div>
                    <div class="condition-label">${{cond.label}}</div>
                    <div class="condition-value">${{cond.value || ''}}</div>
                `;
                condList.appendChild(item);
            }});

            updateTabs(sig);

            if (currentTab === 'DASHBOARD') {{
                showDashboard();
            }} else {{
                showChart(sig);
            }}
        }}

        function updateTabs(sig) {{
            document.querySelectorAll('.tab').forEach(tab => {{
                const tabId = tab.dataset.tab;
                tab.classList.remove('active', 'disabled');

                let hasData = false;
                if (tabId === 'DETECT' && sig.tabDetect) hasData = true;
                if (tabId === 'CONFIRM' && sig.tabConfirm) hasData = true;
                if (tabId === 'PNL' && sig.tabPnL) hasData = true;
                if (tabId === 'SL_HISTORY' && sig.tabSLHistory) hasData = true;
                if (tabId === 'CONTEXT' && sig.tabContext) hasData = true;
                if (tabId === 'DASHBOARD') hasData = true;

                if (!hasData) tab.classList.add('disabled');
                if (tabId === currentTab) tab.classList.add('active');
            }});
        }}

        function switchTab(tabId) {{
            const sig = signalsData[currentSignalIdx];
            let hasData = false;
            if (tabId === 'DETECT' && sig.tabDetect) hasData = true;
            if (tabId === 'CONFIRM' && sig.tabConfirm) hasData = true;
            if (tabId === 'PNL' && sig.tabPnL) hasData = true;
            if (tabId === 'SL_HISTORY' && sig.tabSLHistory) hasData = true;
            if (tabId === 'CONTEXT' && sig.tabContext) hasData = true;
            if (tabId === 'DASHBOARD') hasData = true;
            if (!hasData) return;

            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
                if (tab.dataset.tab === tabId) tab.classList.add('active');
            }});
            currentTab = tabId;
            contextZoomedOut = false;
            const zoomBtn = document.getElementById('context-zoom-toggle');
            zoomBtn.style.display = (tabId === 'CONTEXT' && htfTradesData) ? 'block' : 'none';
            zoomBtn.textContent = 'Zoom Out';
            document.getElementById('cluster-threshold-control').style.display = 'none';

            if (tabId === 'DASHBOARD') {{
                showDashboard();
            }} else {{
                hideDashboard();
                document.getElementById('chart-container').style.display = 'block';
                showChart(sig);
            }}
        }}

        function navigateToSignal(signalIdx) {{
            if (signalIdx < 0 || signalIdx >= signalsData.length) return;
            currentSignalIdx = signalIdx;
            document.getElementById('signal-select').value = signalIdx;
            showSignal(signalIdx);
            switchTab('DETECT');
        }}

        function clearLineSeries() {{
            activeLineSeries.forEach(series => {{
                try {{ chart.removeSeries(series); }} catch(e) {{}}
            }});
            activeLineSeries = [];
        }}

        function applyContextWindow(tabData) {{
            if (tabData.focusTime && tabData.contextCandles && tabData.candleData && tabData.candleData.length > 0) {{
                const focusTime = tabData.focusTime;
                let focusIdx = 0;
                for (let i = 0; i < tabData.candleData.length; i++) {{
                    if (tabData.candleData[i].time >= focusTime) {{
                        focusIdx = i;
                        break;
                    }}
                    focusIdx = i;
                }}
                const startIdx = Math.max(0, focusIdx - tabData.contextCandles);
                const endIdx = Math.min(tabData.candleData.length - 1, focusIdx + tabData.contextCandles);
                chart.timeScale().setVisibleLogicalRange({{ from: startIdx, to: endIdx }});
                setTimeout(() => {{
                    chart.priceScale('right').applyOptions({{ autoScale: true }});
                }}, 50);
            }} else {{
                chart.timeScale().fitContent();
            }}
        }}

        function showChart(sig) {{
            let tabData = null;
            if (currentTab === 'DETECT') tabData = sig.tabDetect;
            else if (currentTab === 'CONFIRM') tabData = sig.tabConfirm;
            else if (currentTab === 'PNL') tabData = sig.tabPnL;
            if (currentTab === 'SL_HISTORY') tabData = sig.tabSLHistory;
            else if (currentTab === 'CONTEXT') tabData = sig.tabContext;

            if (!tabData) return;

            if (currentTab === 'PNL') {{
                showPnLChart(tabData);
                return;
            }}

            if (currentTab === 'SL_HISTORY') {{
                showSLHistoryChart(tabData);
                return;
            }}

            clearLineSeries();
            candlestickSeries.setData(tabData.candleData);
            candlestickSeries.setMarkers([]);

            chart.priceScale('right').applyOptions({{
                autoScale: true,
                scaleMargins: {{ top: 0.15, bottom: 0.15 }},
            }});

            // TP/SL lines
            if (tabData.tpPrice != null) {{
                const firstTime = tabData.candleData[0].time;
                const lastTime = tabData.candleData[tabData.candleData.length - 1].time;
                const tpSeries = chart.addLineSeries({{
                    color: '#089981', lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false, lastValueVisible: true, title: 'TP',
                }});
                tpSeries.setData([
                    {{ time: firstTime, value: tabData.tpPrice }},
                    {{ time: lastTime, value: tabData.tpPrice }}
                ]);
                activeLineSeries.push(tpSeries);
            }}

            if (tabData.slPrice != null) {{
                const firstTime = tabData.candleData[0].time;
                const lastTime = tabData.candleData[tabData.candleData.length - 1].time;
                const slSeries = chart.addLineSeries({{
                    color: '#f23645', lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false, lastValueVisible: true, title: 'SL',
                }});
                slSeries.setData([
                    {{ time: firstTime, value: tabData.slPrice }},
                    {{ time: lastTime, value: tabData.slPrice }}
                ]);
                activeLineSeries.push(slSeries);
            }}

            // BOS lines (confirm tab)
            if (tabData.bosLines) {{
                tabData.bosLines.forEach(bos => {{
                    const lineSeries = chart.addLineSeries({{
                        color: bos.color,
                        lineWidth: bos.lineWidth || 2,
                        lineStyle: LightweightCharts.LineStyle.Solid,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    }});
                    lineSeries.setData([
                        {{ time: bos.startTime, value: bos.price }},
                        {{ time: bos.endTime, value: bos.price }}
                    ]);
                    activeLineSeries.push(lineSeries);
                }});
            }}

            applyContextWindow(tabData);
            setTimeout(redrawOverlays, 50);
        }}

        function showPnLChart(pnlData) {{
            clearLineSeries();
            d3.select('#svg-overlay').selectAll('*').remove();
            candlestickSeries.setData([]);
            candlestickSeries.setMarkers([]);

            if (!pnlData || !pnlData.pnlSeries || pnlData.pnlSeries.length === 0) return;

            const firstTime = pnlData.pnlSeries[0].time;
            const lastTime = pnlData.pnlSeries[pnlData.pnlSeries.length - 1].time;

            const pnlSeries = chart.addLineSeries({{
                color: '#2962ff', lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                priceLineVisible: false, lastValueVisible: true, title: 'P&L',
            }});
            pnlSeries.setData(pnlData.pnlSeries);
            activeLineSeries.push(pnlSeries);

            const zeroSeries = chart.addLineSeries({{
                color: '#787b86', lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                priceLineVisible: false, lastValueVisible: false,
            }});
            zeroSeries.setData([
                {{ time: firstTime, value: 0 }},
                {{ time: lastTime, value: 0 }}
            ]);
            activeLineSeries.push(zeroSeries);

            if (pnlData.tpDollars) {{
                const tpSeries = chart.addLineSeries({{
                    color: '#089981', lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false, lastValueVisible: true, title: 'TP',
                }});
                tpSeries.setData([
                    {{ time: firstTime, value: pnlData.tpDollars }},
                    {{ time: lastTime, value: pnlData.tpDollars }}
                ]);
                activeLineSeries.push(tpSeries);
            }}

            if (pnlData.slDollars) {{
                const slSeries = chart.addLineSeries({{
                    color: '#f23645', lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false, lastValueVisible: true, title: 'SL',
                }});
                slSeries.setData([
                    {{ time: firstTime, value: pnlData.slDollars }},
                    {{ time: lastTime, value: pnlData.slDollars }}
                ]);
                activeLineSeries.push(slSeries);
            }}

            if (pnlData.exitTime && pnlData.exitPnl !== null) {{
                const exitMarkerSeries = chart.addLineSeries({{
                    color: pnlData.exitPnl >= 0 ? '#089981' : '#f23645',
                    lineWidth: 0, priceLineVisible: false, lastValueVisible: false,
                }});
                exitMarkerSeries.setData([{{ time: pnlData.exitTime, value: pnlData.exitPnl }}]);
                exitMarkerSeries.setMarkers([{{
                    time: pnlData.exitTime,
                    position: pnlData.exitPnl >= 0 ? 'aboveBar' : 'belowBar',
                    color: pnlData.exitPnl >= 0 ? '#089981' : '#f23645',
                    shape: 'circle', text: 'EXIT'
                }}]);
                activeLineSeries.push(exitMarkerSeries);
            }}

            chart.timeScale().fitContent();

            const pnlValues = pnlData.pnlSeries.map(d => d.value);
            const minPnl = Math.min(...pnlValues, 0);
            const maxPnl = Math.max(...pnlValues, 0);
            const padding = (maxPnl - minPnl) * 0.15 || 10;

            pnlSeries.applyOptions({{
                autoscaleInfoProvider: () => ({{
                    priceRange: {{
                        minValue: minPnl - padding,
                        maxValue: maxPnl + padding,
                    }},
                }}),
            }});

            chart.priceScale('right').applyOptions({{ autoScale: true }});
        }}

        function showSLHistoryChart(slData) {{
            clearLineSeries();
            d3.select('#svg-overlay').selectAll('*').remove();
            candlestickSeries.setData(slData && slData.candleData ? slData.candleData : []);
            candlestickSeries.setMarkers([]);

            if (!slData || !slData.slHistory || slData.slHistory.length === 0) return;

            const history = slData.slHistory;
            const exitTime = slData.exitTime;
            const isLong = slData.direction === 'long';
            const dimColor = isLong ? 'rgba(242,54,69,0.25)' : 'rgba(8,153,129,0.25)';
            const activeColor = isLong ? '#f23645' : '#089981';

            const firstTime = history[0].time;
            const lastTime = exitTime || history[history.length - 1].time;
            const entryLine = chart.addLineSeries({{
                color: '#787b86', lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                priceLineVisible: false, lastValueVisible: true, title: 'Entry',
            }});
            entryLine.setData([
                {{ time: firstTime, value: slData.entryPrice }},
                {{ time: lastTime, value: slData.entryPrice }}
            ]);
            activeLineSeries.push(entryLine);

            if (slData.activationPrice) {{
                const actLine = chart.addLineSeries({{
                    color: '#ff9800', lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dotted,
                    priceLineVisible: false, lastValueVisible: true,
                    title: 'Activation (' + slData.activationPct + '%)',
                }});
                actLine.setData([
                    {{ time: firstTime, value: slData.activationPrice }},
                    {{ time: lastTime, value: slData.activationPrice }}
                ]);
                activeLineSeries.push(actLine);
            }}

            if (slData.tpPrice) {{
                const tpLine = chart.addLineSeries({{
                    color: '#089981', lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dotted,
                    priceLineVisible: false, lastValueVisible: true,
                    title: 'TP',
                }});
                tpLine.setData([
                    {{ time: firstTime, value: slData.tpPrice }},
                    {{ time: lastTime, value: slData.tpPrice }}
                ]);
                activeLineSeries.push(tpLine);
            }}

            for (let i = 0; i < history.length; i++) {{
                const sl = history[i];
                const startTime = sl.time;
                const endTime = (i < history.length - 1) ? history[i + 1].time : (exitTime || sl.time + 3600);
                const isActive = sl.isActive;

                const color = isActive ? activeColor : dimColor;
                const lineWidth = isActive ? 3 : 1;

                const slLine = chart.addLineSeries({{
                    color: color,
                    lineWidth: lineWidth,
                    lineStyle: isActive ? LightweightCharts.LineStyle.Solid : LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false,
                    lastValueVisible: isActive,
                    title: isActive ? 'SL (active)' : '',
                }});
                slLine.setData([
                    {{ time: startTime, value: sl.price }},
                    {{ time: endTime, value: sl.price }}
                ]);
                activeLineSeries.push(slLine);

                const markerSeries = chart.addLineSeries({{
                    color: color, lineWidth: 0,
                    priceLineVisible: false, lastValueVisible: false,
                }});
                markerSeries.setData([{{ time: startTime, value: sl.price }}]);
                markerSeries.setMarkers([{{
                    time: startTime,
                    position: isLong ? 'belowBar' : 'aboveBar',
                    color: isActive ? activeColor : (isLong ? 'rgba(242,54,69,0.6)' : 'rgba(8,153,129,0.6)'),
                    shape: 'circle',
                    text: sl.reason,
                }}]);
                activeLineSeries.push(markerSeries);
            }}

            chart.timeScale().fitContent();
            chart.priceScale('right').applyOptions({{ autoScale: true }});
        }}

        function redrawOverlays() {{
            if (currentTab === 'PNL' || currentTab === 'SL_HISTORY' || currentTab === 'DASHBOARD') return;

            let tabData = null;
            if (currentTab === 'CONTEXT' && contextZoomedOut) {{
                tabData = htfTradesData;
                const sig = signalsData[currentSignalIdx];
                if (sig && sig.tabContext && sig.tabContext.focusWindow) {{
                    tabData = Object.assign({{}}, tabData, {{ focusWindow: sig.tabContext.focusWindow }});
                }}
            }} else {{
                const sig = signalsData[currentSignalIdx];
                if (currentTab === 'DETECT') tabData = sig.tabDetect;
                else if (currentTab === 'CONFIRM') tabData = sig.tabConfirm;
                else if (currentTab === 'CONTEXT') tabData = sig.tabContext;
            }}
            if (!tabData) return;

            const chartContainer = document.getElementById('chart-container');
            const rect = chartContainer.getBoundingClientRect();

            const svg = d3.select('#svg-overlay')
                .attr('width', rect.width)
                .attr('height', rect.height);
            svg.selectAll('*').remove();

            const ts = chart.timeScale();

            // Draw FVG zones
            if (tabData.fvgZones) {{
                tabData.fvgZones.forEach(zone => {{
                    const x1 = ts.timeToCoordinate(zone.startTime);
                    const x2 = ts.timeToCoordinate(zone.endTime);
                    const y1 = candlestickSeries.priceToCoordinate(zone.topPrice);
                    const y2 = candlestickSeries.priceToCoordinate(zone.bottomPrice);
                    if (x1 == null || x2 == null || y1 == null || y2 == null) return;

                    const x = Math.min(x1, x2);
                    const y = Math.min(y1, y2);
                    const w = Math.max(Math.abs(x2 - x1), 4);
                    const h = Math.max(Math.abs(y2 - y1), 2);

                    svg.append('rect')
                        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
                        .attr('fill', 'rgba(255, 235, 59, 0.15)')
                        .attr('stroke', '#ffeb3b')
                        .attr('stroke-width', 1)
                        .attr('stroke-dasharray', '4,4');
                }});
            }}

            // Draw OB zones
            if (tabData.obZones) {{
                tabData.obZones.forEach(zone => {{
                    const x1 = ts.timeToCoordinate(zone.startTime);
                    const x2 = ts.timeToCoordinate(zone.endTime);
                    const y1 = candlestickSeries.priceToCoordinate(zone.topPrice);
                    const y2 = candlestickSeries.priceToCoordinate(zone.bottomPrice);
                    if (x1 == null || x2 == null || y1 == null || y2 == null) return;

                    const x = Math.min(x1, x2);
                    const y = Math.min(y1, y2);
                    const w = Math.max(Math.abs(x2 - x1), 4);
                    const h = Math.max(Math.abs(y2 - y1), 2);

                    svg.append('rect')
                        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
                        .attr('fill', 'rgba(156, 39, 176, 0.15)')
                        .attr('stroke', '#9c27b0')
                        .attr('stroke-width', 1)
                        .attr('stroke-dasharray', '4,4');
                }});
            }}

            // Draw highlighted target zone
            if (tabData.targetZone) {{
                const tz = tabData.targetZone;
                const x1 = ts.timeToCoordinate(tz.startTime);
                const x2 = ts.timeToCoordinate(tz.endTime);
                const y1 = candlestickSeries.priceToCoordinate(tz.topPrice);
                const y2 = candlestickSeries.priceToCoordinate(tz.bottomPrice);
                if (x1 != null && x2 != null && y1 != null && y2 != null) {{
                    const x = Math.min(x1, x2);
                    const y = Math.min(y1, y2);
                    const w = Math.max(Math.abs(x2 - x1), 4);
                    const h = Math.max(Math.abs(y2 - y1), 2);

                    svg.append('rect')
                        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
                        .attr('fill', 'rgba(255, 152, 0, 0.3)')
                        .attr('stroke', '#ff9800')
                        .attr('stroke-width', 3);

                    svg.append('text')
                        .attr('x', x + 4).attr('y', y + 14)
                        .attr('fill', '#ff9800')
                        .attr('font-size', '12px').attr('font-weight', 'bold')
                        .attr('font-family', 'sans-serif')
                        .text('TARGET (' + tz.zoneType + ')');
                }}
            }}

            // Draw entry arrow
            if (tabData.entryMarker) {{
                const em = tabData.entryMarker;
                const xE = ts.timeToCoordinate(em.time);
                const yE = candlestickSeries.priceToCoordinate(em.price);
                if (xE != null && yE != null) {{
                    const isLong = em.direction === 'long';
                    const color = isLong ? '#089981' : '#f23645';
                    const arrowY = isLong ? yE + 15 : yE - 15;
                    const tipY = isLong ? yE + 2 : yE - 2;

                    svg.append('line')
                        .attr('x1', xE).attr('x2', xE)
                        .attr('y1', arrowY).attr('y2', tipY)
                        .attr('stroke', color).attr('stroke-width', 3);

                    const headSize = 6;
                    const headDir = isLong ? -1 : 1;
                    svg.append('polygon')
                        .attr('points',
                            (xE - headSize) + ',' + (tipY + headDir * headSize) + ' ' +
                            xE + ',' + tipY + ' ' +
                            (xE + headSize) + ',' + (tipY + headDir * headSize)
                        )
                        .attr('fill', color);

                    svg.append('text')
                        .attr('x', xE + 10).attr('y', arrowY + 4)
                        .attr('fill', color).attr('font-size', '11px')
                        .attr('font-weight', 'bold').attr('font-family', 'sans-serif')
                        .text(em.direction.toUpperCase() + ' $' + em.price.toFixed(2));
                }}
            }}

            // Draw exit marker
            if (tabData.exitMarker) {{
                const ex = tabData.exitMarker;
                const xEx = ts.timeToCoordinate(ex.time);
                const yEx = candlestickSeries.priceToCoordinate(ex.price);
                if (xEx != null && yEx != null) {{
                    const isLong = ex.direction === 'long';
                    const exitColor = ex.isWin ? '#089981' : '#f23645';
                    const exitLabelMap = {{'tp_hit': 'TP', 'sl_hit': 'SL', 'trailing_sl': 'TSL', 'timeout': 'TIME'}};
                    const exitLabel = exitLabelMap[ex.exitType] || 'EXIT';

                    svg.append('circle')
                        .attr('cx', xEx).attr('cy', yEx).attr('r', 6)
                        .attr('fill', exitColor).attr('stroke', '#fff').attr('stroke-width', 1);

                    svg.append('text')
                        .attr('x', xEx + 10).attr('y', yEx + 4)
                        .attr('fill', exitColor).attr('font-size', '11px')
                        .attr('font-weight', 'bold').attr('font-family', 'sans-serif')
                        .text(exitLabel + ' $' + ex.price.toFixed(2));
                }}
            }}

            // Draw all OBs (Context tab) — background layer
            if (tabData.allOBs) {{
                tabData.allOBs.forEach(ob => {{
                    const x1 = ts.timeToCoordinate(ob.startTime);
                    const x2 = ts.timeToCoordinate(ob.endTime);
                    const y1 = candlestickSeries.priceToCoordinate(ob.topPrice);
                    const y2 = candlestickSeries.priceToCoordinate(ob.bottomPrice);
                    if (x1 == null || x2 == null || y1 == null || y2 == null) return;

                    const x = Math.min(x1, x2);
                    const y = Math.min(y1, y2);
                    const w = Math.max(Math.abs(x2 - x1), 4);
                    const h = Math.max(Math.abs(y2 - y1), 2);

                    const isBull = ob.direction === 1;
                    const sat = ob.satisfied;
                    const fillColor = sat
                        ? (isBull ? 'rgba(0,188,212,0.07)' : 'rgba(255,87,34,0.07)')
                        : (isBull ? 'rgba(0,188,212,0.20)' : 'rgba(255,87,34,0.20)');
                    const strokeColor = isBull ? '#00bcd4' : '#ff5722';
                    const strokeOpacity = sat ? 0.3 : 1.0;
                    const dashArray = sat ? '4,3' : 'none';

                    svg.append('rect')
                        .attr('x', x).attr('y', y).attr('width', w).attr('height', h)
                        .attr('fill', fillColor)
                        .attr('stroke', strokeColor)
                        .attr('stroke-opacity', strokeOpacity)
                        .attr('stroke-width', 1)
                        .attr('stroke-dasharray', dashArray);
                }});
            }}

            // Draw focus window rectangle (Context tab)
            if (tabData.focusWindow) {{
                const x1 = ts.timeToCoordinate(tabData.focusWindow.startTime);
                const x2 = ts.timeToCoordinate(tabData.focusWindow.endTime);
                if (x1 != null && x2 != null) {{
                    const x = Math.min(x1, x2);
                    const w = Math.abs(x2 - x1);
                    svg.append('rect')
                        .attr('x', x).attr('y', 0)
                        .attr('width', w).attr('height', rect.height)
                        .attr('fill', 'rgba(100,180,255,0.08)')
                        .attr('stroke', 'rgba(100,180,255,0.5)')
                        .attr('stroke-width', 2)
                        .attr('stroke-dasharray', '6,4');
                    svg.append('text')
                        .attr('x', x + 4).attr('y', 16)
                        .attr('fill', 'rgba(100,180,255,0.7)')
                        .attr('font-size', '11px').attr('font-weight', 'bold')
                        .attr('font-family', 'sans-serif')
                        .text('Detection view');
                }}
            }}

            // Draw trade clusters (consecutive win/loss streaks)
            if (tabData.tradeClusters) {{
                tabData.tradeClusters.forEach(cl => {{
                    if (cl.count < clusterThreshold) return;
                    const x1 = ts.timeToCoordinate(cl.startTime);
                    const x2 = ts.timeToCoordinate(cl.endTime);
                    if (x1 == null || x2 == null) return;
                    const x = Math.min(x1, x2);
                    const w = Math.max(Math.abs(x2 - x1), 4);
                    const color = cl.isWin ? 'rgba(8,153,129,' : 'rgba(242,54,69,';
                    svg.append('rect')
                        .attr('x', x).attr('y', 0)
                        .attr('width', w).attr('height', rect.height)
                        .attr('fill', color + '0.07)')
                        .attr('stroke', color + '0.4)')
                        .attr('stroke-width', 2)
                        .attr('stroke-dasharray', '6,4');
                    svg.append('text')
                        .attr('x', x + 4).attr('y', 32)
                        .attr('fill', color + '0.8)')
                        .attr('font-size', '11px').attr('font-weight', 'bold')
                        .attr('font-family', 'sans-serif')
                        .text((cl.isWin ? 'W' : 'L') + 'x' + cl.count);
                }});
            }}
        }}

        function toggleContextZoom() {{
            contextZoomedOut = !contextZoomedOut;
            const btn = document.getElementById('context-zoom-toggle');
            btn.textContent = contextZoomedOut ? 'Zoom In' : 'Zoom Out';
            document.getElementById('cluster-threshold-control').style.display = contextZoomedOut ? 'block' : 'none';

            if (contextZoomedOut) {{
                showHTFTradesChart();
            }} else {{
                const sig = signalsData[currentSignalIdx];
                showChart(sig);
            }}
        }}

        function onClusterThresholdChange() {{
            clusterThreshold = parseInt(document.getElementById('cluster-threshold').value) || 3;
            if (contextZoomedOut) {{
                redrawOverlays();
            }}
        }}

        function showHTFTradesChart() {{
            if (!htfTradesData) return;
            clearLineSeries();
            d3.select('#svg-overlay').selectAll('*').remove();

            candlestickSeries.setData(htfTradesData.candleData);
            candlestickSeries.setMarkers(htfTradesData.markers || []);

            chart.priceScale('right').applyOptions({{
                autoScale: true,
                scaleMargins: {{ top: 0.15, bottom: 0.15 }},
            }});

            // Draw entry-to-exit trade lines
            if (htfTradesData.tradeLines) {{
                htfTradesData.tradeLines.forEach(tl => {{
                    const lineSeries = chart.addLineSeries({{
                        color: tl.color,
                        lineWidth: 2,
                        lineStyle: LightweightCharts.LineStyle.Solid,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    }});
                    lineSeries.setData([
                        {{ time: tl.entryTime, value: tl.entryPrice }},
                        {{ time: tl.exitTime, value: tl.exitPrice }}
                    ]);
                    activeLineSeries.push(lineSeries);
                }});
            }}

            // BOS lines
            if (htfTradesData.bosLines) {{
                htfTradesData.bosLines.forEach(bos => {{
                    const lineSeries = chart.addLineSeries({{
                        color: bos.color,
                        lineWidth: bos.lineWidth || 2,
                        lineStyle: LightweightCharts.LineStyle.Solid,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    }});
                    lineSeries.setData([
                        {{ time: bos.startTime, value: bos.price }},
                        {{ time: bos.endTime, value: bos.price }}
                    ]);
                    activeLineSeries.push(lineSeries);
                }});
            }}

            // Liquidity lines
            if (htfTradesData.liquidityLines) {{
                htfTradesData.liquidityLines.forEach(liq => {{
                    const lineSeries = chart.addLineSeries({{
                        color: liq.color,
                        lineWidth: liq.lineWidth,
                        lineStyle: liq.swept ? LightweightCharts.LineStyle.Solid : LightweightCharts.LineStyle.Dotted,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    }});
                    lineSeries.setData([
                        {{ time: liq.startTime, value: liq.price }},
                        {{ time: liq.endTime, value: liq.price }}
                    ]);
                    activeLineSeries.push(lineSeries);
                }});
            }}

            chart.timeScale().fitContent();
            setTimeout(redrawOverlays, 50);
        }}

        // Dashboard functions
        function showDashboard() {{
            document.getElementById('chart-container').style.display = 'none';
            const dc = document.getElementById('dashboard-container');
            dc.classList.add('active');
            if (!dashboardRendered) {{
                renderDashboard();
                dashboardRendered = true;
            }}
        }}

        function hideDashboard() {{
            document.getElementById('chart-container').style.display = 'block';
            document.getElementById('dashboard-container').classList.remove('active');
        }}

        function renderDashboard() {{
            const container = document.getElementById('dashboard-container');
            if (!performanceData) {{
                container.innerHTML = '<div class="dashboard-no-data">No performance data available. Run backtest to see dashboard.</div>';
                return;
            }}
            const data = performanceData;
            const formatDollars = (val) => {{
                if (val === null || val === undefined) return '-';
                const prefix = val >= 0 ? '+' : '';
                return prefix + '$' + val.toFixed(2).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ',');
            }};
            const formatPercent = (val) => {{
                if (val === null || val === undefined) return '-';
                return (val >= 0 ? '+' : '') + val.toFixed(2) + '%';
            }};
            const formatR = (val) => {{
                if (val === null || val === undefined) return '-';
                return (val >= 0 ? '+' : '') + val.toFixed(2) + 'R';
            }};

            const longWinRate = data.longTrades > 0 ? ((data.longWins / data.longTrades) * 100).toFixed(1) : '0';
            const shortWinRate = data.shortTrades > 0 ? ((data.shortWins / data.shortTrades) * 100).toFixed(1) : '0';
            const returnClass = data.totalReturn >= 0 ? 'positive' : 'negative';
            const winRateClass = data.winRate >= 50 ? 'positive' : 'negative';
            const pfClass = data.profitFactor !== null && data.profitFactor >= 1 ? 'positive' : (data.profitFactor !== null ? 'negative' : 'neutral');

            container.innerHTML = `
                <div class="analysis-period">
                    <span class="period-label">Analysis Period:</span>
                    <span class="period-value">${{data.analysisStart}} to ${{data.analysisEnd}}</span>
                </div>
                <div class="kpi-grid">
                    <div class="kpi-card ${{data.totalReturn >= 0 ? 'positive' : 'negative'}}">
                        <div class="kpi-label">Total Return</div>
                        <div class="kpi-value ${{returnClass}}">${{formatPercent(data.totalReturn)}}</div>
                    </div>
                    <div class="kpi-card ${{data.winRate >= 50 ? 'positive' : 'negative'}}">
                        <div class="kpi-label">Win Rate</div>
                        <div class="kpi-value ${{winRateClass}}">${{data.winRate.toFixed(1)}}%</div>
                    </div>
                    <div class="kpi-card ${{pfClass === 'positive' ? 'positive' : ''}}">
                        <div class="kpi-label">Profit Factor</div>
                        <div class="kpi-value ${{pfClass}}">${{data.profitFactorStr}}</div>
                    </div>
                    <div class="kpi-card negative">
                        <div class="kpi-label">Max Drawdown</div>
                        <div class="kpi-value negative">-${{data.maxDrawdownPercent.toFixed(1)}}%</div>
                    </div>
                    <div class="kpi-card ${{data.returnPerDay >= 0 ? 'positive' : 'negative'}}">
                        <div class="kpi-label">Return / Day</div>
                        <div class="kpi-value ${{data.returnPerDay >= 0 ? 'positive' : 'negative'}}">${{data.returnPerDay >= 0 ? '+' : ''}}${{data.returnPerDay.toFixed(3)}}%</div>
                        <div class="kpi-label" style="font-size: 11px; margin-top: 4px;">${{data.tradingDays}} trading days</div>
                    </div>
                </div>
                <div class="equity-chart-container">
                    <div class="equity-chart-title">Equity Curve</div>
                    <div id="equity-chart"></div>
                </div>
                <div class="stats-grid">
                    <div class="stats-card">
                        <div class="stats-card-title">Trade Statistics</div>
                        <div class="stats-row"><span class="stats-label">Total Trades</span><span class="stats-value">${{data.totalTrades}}</span></div>
                        <div class="stats-row"><span class="stats-label">Winning</span><span class="stats-value positive">${{data.winningTrades}}</span></div>
                        <div class="stats-row"><span class="stats-label">Losing</span><span class="stats-value negative">${{data.losingTrades}}</span></div>
                        <div class="stats-row"><span class="stats-label">Long</span><span class="stats-value">${{data.longTrades}} (${{longWinRate}}% WR)</span></div>
                        <div class="stats-row"><span class="stats-label">Short</span><span class="stats-value">${{data.shortTrades}} (${{shortWinRate}}% WR)</span></div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-title">Exit Breakdown</div>
                        <div class="stats-row"><span class="stats-label">TP Exits</span><span class="stats-value positive">${{data.tpExits}}</span></div>
                        <div class="stats-row"><span class="stats-label">SL Exits</span><span class="stats-value negative">${{data.slExits}}</span></div>
                        <div class="stats-row"><span class="stats-label">Timeout</span><span class="stats-value">${{data.timeoutExits}}</span></div>
                        <div class="stats-row"><span class="stats-label">Trailing SL</span><span class="stats-value">${{data.trailingSlExits}}</span></div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-title">Risk Metrics</div>
                        <div class="stats-row"><span class="stats-label">Average R</span><span class="stats-value ${{data.avgRMultiple >= 0 ? 'positive' : 'negative'}}">${{formatR(data.avgRMultiple)}}</span></div>
                        <div class="stats-row"><span class="stats-label">Max Drawdown</span><span class="stats-value negative">${{formatDollars(-data.maxDrawdownDollars)}} (${{data.maxDrawdownPercent.toFixed(1)}}%)</span></div>
                        <div class="stats-row"><span class="stats-label">Max Consec Wins</span><span class="stats-value positive">${{data.maxConsecWins}}</span></div>
                        <div class="stats-row"><span class="stats-label">Max Consec Losses</span><span class="stats-value negative">${{data.maxConsecLosses}}</span></div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-title">P&L Breakdown</div>
                        <div class="stats-row"><span class="stats-label">Initial Capital</span><span class="stats-value">$${{data.initialCapital.toFixed(2).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ',')}}</span></div>
                        <div class="stats-row"><span class="stats-label">Final Capital</span><span class="stats-value">$${{data.finalCapital.toFixed(2).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ',')}}</span></div>
                        <div class="stats-row"><span class="stats-label">Total P&L</span><span class="stats-value ${{data.totalPnl >= 0 ? 'positive' : 'negative'}}">${{formatDollars(data.totalPnl)}}</span></div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-title">Win/Loss Breakdown</div>
                        <div class="stats-row"><span class="stats-label">Average Win</span><span class="stats-value positive">${{formatDollars(data.avgWinDollars)}}</span></div>
                        <div class="stats-row"><span class="stats-label">Average Loss</span><span class="stats-value negative">${{formatDollars(data.avgLossDollars)}}</span></div>
                        <div class="stats-row"><span class="stats-label">Largest Win</span><span class="stats-value positive">${{formatDollars(data.largestWin)}}</span></div>
                        <div class="stats-row"><span class="stats-label">Largest Loss</span><span class="stats-value negative">${{formatDollars(data.largestLoss)}}</span></div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-title">Impulse Duration (debug)</div>
                        <div class="stats-row"><span class="stats-label">Avg FVG candles</span><span class="stats-value">${{data.impulseAvgFVG}}</span></div>
                        <div class="stats-row"><span class="stats-label">Avg OB candles</span><span class="stats-value">${{data.impulseAvgOB}}</span></div>
                    </div>
                </div>
                <div class="transactions-container">
                    <div class="transactions-header">
                        <span class="transactions-title">Transactions</span>
                        <span class="transactions-count" id="transactions-count">0 trades</span>
                    </div>
                    <div class="transactions-table-wrapper">
                        <table class="transactions-table">
                            <thead>
                                <tr>
                                    <th class="sortable" data-sort="tradeNum"># <span class="sort-indicator"></span></th>
                                    <th class="sortable" data-sort="entryTime">Entry <span class="sort-indicator"></span></th>
                                    <th>Dir</th>
                                    <th>P&L</th>
                                    <th>% Growth</th>
                                    <th>Outcome</th>
                                    <th>Exit</th>
                                </tr>
                            </thead>
                            <tbody id="transactions-body"></tbody>
                        </table>
                    </div>
                </div>
            `;

            renderEquityCurve(data.equityCurve);
            renderTransactionsTable(data.trades);
        }}

        function renderEquityCurve(equityCurveData) {{
            if (!equityCurveData || equityCurveData.length < 2) return;
            const chartContainer = document.getElementById('equity-chart');
            if (!chartContainer) return;

            equityChart = LightweightCharts.createChart(chartContainer, {{
                layout: {{ background: {{ color: '#1e222d' }}, textColor: '#d1d4dc' }},
                grid: {{ vertLines: {{ color: '#2b2b43' }}, horzLines: {{ color: '#2b2b43' }} }},
                rightPriceScale: {{ borderColor: '#2b2b43', scaleMargins: {{ top: 0.1, bottom: 0.1 }} }},
                timeScale: {{
                    borderColor: '#2b2b43', visible: true, timeVisible: false,
                    tickMarkFormatter: (time) => `Trade ${{time}}`,
                }},
                localization: {{
                    timeFormatter: (time) => `Trade ${{time}}`,
                }},
                handleScroll: false, handleScale: false,
            }});

            const areaSeries = equityChart.addAreaSeries({{
                topColor: 'rgba(41, 98, 255, 0.4)',
                bottomColor: 'rgba(41, 98, 255, 0.0)',
                lineColor: '#2962ff', lineWidth: 2,
                priceLineVisible: false, lastValueVisible: true,
            }});
            areaSeries.setData(equityCurveData.map(d => ({{ time: d.trade, value: d.capital }})));

            const initialCapital = equityCurveData[0].capital;
            const initCapSeries = equityChart.addLineSeries({{
                color: '#787b86', lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                priceLineVisible: false, lastValueVisible: false,
            }});
            initCapSeries.setData([
                {{ time: 0, value: initialCapital }},
                {{ time: equityCurveData.length - 1, value: initialCapital }}
            ]);
            equityChart.timeScale().fitContent();
        }}

        let transactionsData = [];
        let sortColumn = 'tradeNum';
        let sortDirection = 'asc';

        function renderTransactionsTable(trades) {{
            if (!trades || trades.length === 0) return;
            transactionsData = trades;
            const countEl = document.getElementById('transactions-count');
            if (countEl) countEl.textContent = `${{trades.length}} trade${{trades.length !== 1 ? 's' : ''}}`;
            renderTradeRows(trades);
            setupTableSorting();
        }}

        function renderTradeRows(trades) {{
            const tbody = document.getElementById('transactions-body');
            if (!tbody) return;

            const sortedTrades = [...trades].sort((a, b) => {{
                let valA = a[sortColumn];
                let valB = b[sortColumn];
                if (sortColumn === 'entryTime') {{
                    valA = new Date(valA).getTime();
                    valB = new Date(valB).getTime();
                }}
                return sortDirection === 'asc'
                    ? (valA > valB ? 1 : valA < valB ? -1 : 0)
                    : (valA < valB ? 1 : valA > valB ? -1 : 0);
            }});

            let html = '';
            sortedTrades.forEach(trade => {{
                const isWin = trade.outcome === 'win';
                const rowClass = isWin ? 'win-row' : 'loss-row';
                const dirClass = trade.direction === 'long' ? 'long' : 'short';
                const pnlClass = trade.pnlDollars >= 0 ? 'positive' : 'negative';
                const pnlSign = trade.pnlDollars >= 0 ? '+' : '';
                let exitClass = 'timeout', exitText = 'TIME';
                if (trade.exitType === 'tp_hit') {{ exitClass = 'tp'; exitText = 'TP'; }}
                else if (trade.exitType === 'sl_hit') {{ exitClass = 'sl'; exitText = 'SL'; }}
                else if (trade.exitType === 'trailing_sl') {{ exitClass = 'trailing-sl'; exitText = 'TSL'; }}

                html += `<tr class="${{rowClass}}" onclick="navigateToSignal(${{trade.signalIdx}})">`
                html += `
                    <td class="trade-num">${{trade.tradeNum}}</td>
                    <td>${{trade.entryTime}}</td>
                    <td><span class="direction-badge ${{dirClass}}">${{trade.direction.toUpperCase()}}</span></td>
                    <td><span class="pnl-value ${{pnlClass}}">${{pnlSign}}$${{trade.pnlDollars.toFixed(2)}}</span></td>
                    <td><span class="pnl-value ${{pnlClass}}">${{trade.pnlPercent.toFixed(2)}}%</span></td>
                    <td><span class="outcome-badge ${{isWin ? 'win' : 'loss'}}">${{isWin ? 'WIN' : 'LOSS'}}</span></td>
                    <td><span class="exit-type-badge ${{exitClass}}">${{exitText}}</span></td>
                </tr>`;
            }});
            tbody.innerHTML = html;
        }}

        function setupTableSorting() {{
            const headers = document.querySelectorAll('.transactions-table th.sortable');
            headers.forEach(header => {{
                header.addEventListener('click', () => {{
                    const column = header.dataset.sort;
                    if (column === sortColumn) {{
                        sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
                    }} else {{
                        sortColumn = column;
                        sortDirection = 'asc';
                    }}
                    headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
                    header.classList.add(sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
                    renderTradeRows(transactionsData);
                }});
            }});
            const initialHeader = document.querySelector(`.transactions-table th[data-sort="${{sortColumn}}"]`);
            if (initialHeader) initialHeader.classList.add('sort-asc');
        }}

        // Throttled redraw for scroll/pan
        let redrawTimer = null;
        let isMouseDown = false;
        function throttledRedraw() {{
            if (redrawTimer) return;
            redrawTimer = requestAnimationFrame(() => {{
                redrawOverlays();
                redrawTimer = null;
            }});
        }}

        const chartContainer = document.getElementById('chart-container');
        chartContainer.addEventListener('wheel', throttledRedraw);
        chartContainer.addEventListener('mousedown', () => {{ isMouseDown = true; }});
        chartContainer.addEventListener('mousemove', () => {{ if (isMouseDown) throttledRedraw(); }});
        chartContainer.addEventListener('mouseup', () => {{ isMouseDown = false; throttledRedraw(); }});
        chartContainer.addEventListener('mouseleave', () => {{ isMouseDown = false; }});

        init();
    </script>
</body>
</html>"""
