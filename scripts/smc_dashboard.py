"""
SMC Dashboard — Unified HTML report with tabbed windows.

Tabs:
  1. Zones   — zone lifecycle stats (FVG, OB, Inflexion)
  2. Reversal Quality — MFE analysis per zone type
  3. OB Analysis — Order Block feature analysis
  4. FVG Magnet — which FVG does price move to next?
  5. Price Distribution — on-demand generation via local server

Usage:
    python3 scripts/smc_dashboard.py
    python3 scripts/smc_dashboard.py --timeframe 15min
    python3 scripts/smc_dashboard.py --symbol META --timeframe 1h
    python3 scripts/smc_dashboard.py --no-serve   # static files only, no server
"""

import sys
import os
import argparse
import json
import math
import webbrowser
import threading
import socket
import numpy as np
import pandas as pd
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import DataLoader
from indicators import smc, smc_custom


# ---------------------------------------------------------------------------
# 1. Load & compute indicators
# ---------------------------------------------------------------------------

def load_and_compute(symbol, timeframe):
    loader = DataLoader(symbol)
    df = loader.get_data(timeframe)
    if df is None or df.empty:
        print(f"No data for {symbol} {timeframe}")
        sys.exit(1)

    inflexions = smc_custom.inflexion_points(df)
    bos = smc_custom.bos(df, inflexions, close_break=True)
    fvg = smc.fvg(df, join_consecutive=True)
    ob = smc_custom.ob(df, bos, inflexions)

    return df, fvg, ob, inflexions, bos


# ---------------------------------------------------------------------------
# 2. Zone extraction (from zone_lifecycle_stats.py)
# ---------------------------------------------------------------------------

def extract_zones(df, fvg, ob, inflexions):
    zones = []

    # FVG zones
    for i in range(len(fvg)):
        if pd.isna(fvg['FVG'].iloc[i]):
            continue
        direction = int(fvg['FVG'].iloc[i])
        top = float(fvg['Top'].iloc[i])
        bottom = float(fvg['Bottom'].iloc[i])
        mid = (top + bottom) / 2
        close_at_creation = float(df['close'].iloc[i])
        dist_pct = abs(mid - close_at_creation) / close_at_creation * 100

        mit_idx = fvg['MitigatedIndex'].iloc[i]
        status_idx = fvg['StatusIndex'].iloc[i]
        respected = fvg['Respected'].iloc[i]

        candles_to_mit = (int(mit_idx) - i) if (not pd.isna(mit_idx) and int(mit_idx) > 0) else None
        candles_to_res = (int(status_idx) - i) if (not pd.isna(status_idx) and int(status_idx) > 0) else None

        zones.append({
            'zone_type': 'FVG', 'direction': direction, 'creation_index': i,
            'top': top, 'bottom': bottom, 'level': mid,
            'mitigated_index': int(mit_idx) if (not pd.isna(mit_idx) and int(mit_idx) > 0) else None,
            'status_index': int(status_idx) if (not pd.isna(status_idx) and int(status_idx) > 0) else None,
            'respected': respected,
            'candles_to_mitigation': candles_to_mit,
            'candles_to_resolution': candles_to_res,
            'distance_pct': dist_pct,
        })

    # OB zones
    for i in range(len(ob)):
        if pd.isna(ob['OB'].iloc[i]):
            continue
        direction = int(ob['OB'].iloc[i])
        top = float(ob['Top'].iloc[i])
        bottom = float(ob['Bottom'].iloc[i])
        mid = (top + bottom) / 2
        close_at_creation = float(df['close'].iloc[i])
        dist_pct = abs(mid - close_at_creation) / close_at_creation * 100

        mit_idx = ob['MitigatedIndex'].iloc[i]
        status_idx = ob['StatusIndex'].iloc[i]
        respected = ob['Respected'].iloc[i]

        candles_to_mit = (int(mit_idx) - i) if (not pd.isna(mit_idx) and int(mit_idx) > 0) else None
        candles_to_res = (int(status_idx) - i) if (not pd.isna(status_idx) and int(status_idx) > 0) else None

        zones.append({
            'zone_type': 'OB', 'direction': direction, 'creation_index': i,
            'top': top, 'bottom': bottom, 'level': mid,
            'mitigated_index': int(mit_idx) if (not pd.isna(mit_idx) and int(mit_idx) > 0) else None,
            'status_index': int(status_idx) if (not pd.isna(status_idx) and int(status_idx) > 0) else None,
            'respected': respected,
            'candles_to_mitigation': candles_to_mit,
            'candles_to_resolution': candles_to_res,
            'distance_pct': dist_pct,
        })

    # Inflexion zones
    for i in range(len(inflexions)):
        if pd.isna(inflexions['InflexionType'].iloc[i]):
            continue
        level_val = inflexions['Level'].iloc[i]
        if pd.isna(level_val):
            continue
        direction = int(inflexions['InflexionType'].iloc[i])
        level = float(level_val)
        close_at_creation = float(df['close'].iloc[i])
        dist_pct = abs(level - close_at_creation) / close_at_creation * 100

        status_idx = inflexions['StatusIndex'].iloc[i]
        respected = inflexions['Respected'].iloc[i]
        candles_to_res = (int(status_idx) - i) if (not pd.isna(status_idx) and int(status_idx) > 0) else None

        zones.append({
            'zone_type': 'Inflexion', 'direction': direction, 'creation_index': i,
            'top': level, 'bottom': level, 'level': level,
            'mitigated_index': None,
            'status_index': int(status_idx) if (not pd.isna(status_idx) and int(status_idx) > 0) else None,
            'respected': respected,
            'candles_to_mitigation': candles_to_res,
            'candles_to_resolution': candles_to_res,
            'distance_pct': dist_pct,
        })

    return pd.DataFrame(zones)


# ---------------------------------------------------------------------------
# 3. Zone statistics (from zone_lifecycle_stats.py)
# ---------------------------------------------------------------------------

def compute_lifecycle_stats(zones_df):
    rows = []
    for (zt, d), g in zones_df.groupby(['zone_type', 'direction']):
        n = len(g)
        resolved = g['candles_to_resolution'].dropna()
        mitigated = g['candles_to_mitigation'].dropna()
        respected_count = (g['respected'] == True).sum()
        disrespected_count = (g['respected'] == False).sum()
        pending_count = g['respected'].isna().sum()

        rows.append({
            'zone_type': zt,
            'direction': 'Bull' if d == 1 else 'Bear',
            'count': n,
            'mitigation_rate': f"{len(mitigated)/n*100:.1f}%" if n else '-',
            'respect_rate': f"{respected_count/(respected_count+disrespected_count)*100:.1f}%" if (respected_count+disrespected_count) else '-',
            'pending': pending_count,
            'mean_candles_to_mit': f"{mitigated.mean():.1f}" if len(mitigated) else '-',
            'median_candles_to_mit': f"{mitigated.median():.1f}" if len(mitigated) else '-',
            'std_candles_to_mit': f"{mitigated.std():.1f}" if len(mitigated) > 1 else '-',
        })
    return pd.DataFrame(rows)


def compute_survival_curves(zones_df, max_candles=500):
    curves = {}
    for zt, g in zones_df.groupby('zone_type'):
        times = g['candles_to_mitigation'].dropna().values
        total = len(g)
        if total == 0:
            continue
        x = np.arange(0, max_candles + 1)
        survived = np.array([np.sum(times > t) + (total - len(times)) for t in x]) / total * 100
        curves[zt] = {'x': x.tolist(), 'y': survived.tolist()}
    return curves


def compute_race_analysis(zones_df, df_len, sample_step=50):
    wins = {zt: 0 for zt in zones_df['zone_type'].unique()}
    races = 0

    for bar in range(0, df_len, sample_step):
        active = zones_df[
            (zones_df['creation_index'] <= bar) &
            (zones_df['candles_to_mitigation'].notna()) &
            (zones_df['creation_index'] + zones_df['candles_to_mitigation'] > bar)
        ].copy()

        if active.empty:
            continue

        active_types = active['zone_type'].unique()
        if len(active_types) < 2:
            continue

        races += 1
        active['abs_mit'] = active['creation_index'] + active['candles_to_mitigation']
        first_hit_idx = active['abs_mit'].idxmin()
        winner = active.loc[first_hit_idx, 'zone_type']
        wins[winner] += 1

    rates = {}
    for zt, w in wins.items():
        rates[zt] = w / races * 100 if races > 0 else 0
    return rates, races


def compute_reversal_quality(df, zones_df):
    results = []
    n = len(df)
    highs = df['high'].values
    lows = df['low'].values

    for _, z in zones_df.iterrows():
        rec = {
            'zone_type': z['zone_type'],
            'direction': int(z['direction']),
            'mfe_pct': None,
            'category': 'Pending',
            'time_to_disrespect': None,
        }

        is_inflexion = z['zone_type'] == 'Inflexion'

        if is_inflexion:
            status_idx = z['status_index']
            respected = z['respected']
            if status_idx is None or pd.isna(status_idx):
                results.append(rec)
                continue
            status_idx = int(status_idx)

            if respected is False:
                rec['category'] = 'Direct disrespect'
                rec['mfe_pct'] = 0.0
                rec['time_to_disrespect'] = 0
            elif respected is True:
                level = float(z['level'])
                if status_idx + 1 < n:
                    scan_h = highs[status_idx + 1:]
                    scan_l = lows[status_idx + 1:]
                    if z['direction'] == 1:  # Peak → bearish reversal expected
                        rec['mfe_pct'] = round((level - float(scan_l.min())) / level * 100, 2)
                    else:  # Valley → bullish reversal expected
                        rec['mfe_pct'] = round((float(scan_h.max()) - level) / level * 100, 2)
                else:
                    rec['mfe_pct'] = 0.0
                rec['category'] = 'Respected (held)'
            results.append(rec)
            continue

        # FVG / OB
        mit_idx = z['mitigated_index']
        if mit_idx is None or pd.isna(mit_idx) or int(mit_idx) == 0:
            results.append(rec)
            continue
        mit_idx = int(mit_idx)
        respected = z['respected']
        status_idx = z['status_index']

        if respected is False:
            if status_idx is not None and not pd.isna(status_idx):
                status_idx = int(status_idx)
                rec['time_to_disrespect'] = status_idx - mit_idx

                if mit_idx == status_idx:
                    rec['category'] = 'Direct disrespect'
                    rec['mfe_pct'] = 0.0
                else:
                    ref_high = float(highs[mit_idx])
                    ref_low = float(lows[mit_idx])
                    scan_start = mit_idx + 1
                    scan_end = status_idx

                    if z['direction'] == 1:  # Bullish zone → expect UP
                        max_h = float(highs[scan_start:scan_end + 1].max())
                        mfe_pct = (max_h - ref_high) / ref_high * 100
                    else:  # Bearish zone → expect DOWN
                        min_l = float(lows[scan_start:scan_end + 1].min())
                        mfe_pct = (ref_low - min_l) / ref_low * 100

                    rec['mfe_pct'] = round(mfe_pct, 2)
                    rec['category'] = 'Reversal then disrespect' if rec['mfe_pct'] > 0 else 'No reversal, then disrespect'
        elif respected is True:
            ref_high = float(highs[mit_idx])
            ref_low = float(lows[mit_idx])
            if mit_idx + 1 < n:
                if z['direction'] == 1:
                    max_h = float(highs[mit_idx + 1:].max())
                    rec['mfe_pct'] = round((max_h - ref_high) / ref_high * 100, 2)
                else:
                    min_l = float(lows[mit_idx + 1:].min())
                    rec['mfe_pct'] = round((ref_low - min_l) / ref_low * 100, 2)
            else:
                rec['mfe_pct'] = 0.0
            rec['category'] = 'Respected (held)'

        results.append(rec)

    # Sanitize for JSON
    for r in results:
        if r['time_to_disrespect'] is not None:
            r['time_to_disrespect'] = int(r['time_to_disrespect'])
        if r['mfe_pct'] is not None:
            r['mfe_pct'] = float(r['mfe_pct'])

    return results


def compute_ob_features(df, ob, zones_df, reversal_data=None):
    """Extract per-OB features that might predict respect vs disrespect."""
    # Build lookup from zones_df row index -> reversal category
    _rev_category = {}
    _rev_mfe = {}
    if reversal_data is not None:
        ob_zone_indices = zones_df.index[zones_df['zone_type'] == 'OB'].tolist()
        rev_ob = [r for r in reversal_data if r['zone_type'] == 'OB']
        for idx, rec in zip(ob_zone_indices, rev_ob):
            creation_idx = zones_df.loc[idx, 'creation_index']
            _rev_category[creation_idx] = rec['category']
            _rev_mfe[creation_idx] = rec.get('mfe_pct', None)

    features = []
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values if 'volume' in df.columns else None
    n = len(df)

    # Pre-compute 20-bar rolling average volume
    if volumes is not None:
        vol_series = pd.Series(volumes)
        vol_rolling = vol_series.rolling(20, min_periods=1).mean().values
    else:
        vol_rolling = None

    for i in range(len(ob)):
        if pd.isna(ob['OB'].iloc[i]):
            continue

        direction = int(ob['OB'].iloc[i])
        top = float(ob['Top'].iloc[i])
        bottom = float(ob['Bottom'].iloc[i])
        mid = (top + bottom) / 2
        zone_size_pct = (top - bottom) / mid * 100 if mid != 0 else 0

        start_idx = int(ob['StartIndex'].iloc[i]) if not pd.isna(ob['StartIndex'].iloc[i]) else i
        bos_idx = int(ob['BOSIndex'].iloc[i]) if not pd.isna(ob['BOSIndex'].iloc[i]) else i

        # Impulse strength: price move from start to BOS
        if 0 <= start_idx < n and 0 <= bos_idx < n and closes[start_idx] != 0:
            impulse_strength_pct = abs(closes[bos_idx] - closes[start_idx]) / closes[start_idx] * 100
        else:
            impulse_strength_pct = 0

        # Impulse speed: candles from start to BOS
        impulse_speed = max(bos_idx - start_idx, 1)

        # BOS candle body size
        if 0 <= bos_idx < n and closes[bos_idx] != 0:
            bos_candle_body_pct = abs(closes[bos_idx] - opens[bos_idx]) / closes[bos_idx] * 100
        else:
            bos_candle_body_pct = 0

        # Volume ratio
        if vol_rolling is not None and 0 <= start_idx < n and vol_rolling[start_idx] > 0:
            volume_ratio = volumes[start_idx] / vol_rolling[start_idx]
        else:
            volume_ratio = 1.0

        # Find matching zone in zones_df for distance_pct, candles_to_mitigation, respected
        ob_zones = zones_df[(zones_df['zone_type'] == 'OB') & (zones_df['creation_index'] == i)]
        if not ob_zones.empty:
            z = ob_zones.iloc[0]
            distance_pct = float(z['distance_pct'])
            ctm = z['candles_to_mitigation']
            candles_to_mitigation = int(ctm) if ctm is not None and not (isinstance(ctm, float) and math.isnan(ctm)) else None
            resp = z['respected']
            if resp is None or (isinstance(resp, float) and math.isnan(resp)):
                respected = None
            else:
                respected = bool(resp)
        else:
            distance_pct = 0
            candles_to_mitigation = None
            respected = None

        category = _rev_category.get(i, 'Pending')
        number_decimals = 1
        features.append({
            'direction': direction,
            'zone_size_pct': round(zone_size_pct, number_decimals),
            'impulse_strength_pct': round(impulse_strength_pct, number_decimals),
            'impulse_speed': impulse_speed,
            'bos_candle_body_pct': round(bos_candle_body_pct, number_decimals),
            'volume_ratio': round(volume_ratio, number_decimals),
            'distance_pct': round(distance_pct, number_decimals),
            'candles_to_mitigation': candles_to_mitigation,
            'respected': respected,
            'category': category,
            'mfe_pct': _rev_mfe.get(i, None),
        })
        # print('features --> ', features)
        # raise Exception('stop')

    return features


def _extract_zone_records(zone_df, type_col):
    """Extract zone records from FVG (type_col='FVG') or OB (type_col='OB') DataFrame."""
    records = []
    for i in range(len(zone_df)):
        if pd.isna(zone_df[type_col].iloc[i]):
            continue
        direction = int(zone_df[type_col].iloc[i])
        top = float(zone_df['Top'].iloc[i])
        bottom = float(zone_df['Bottom'].iloc[i])
        mit_raw = zone_df['MitigatedIndex'].iloc[i]
        mit_idx = int(mit_raw) if (not pd.isna(mit_raw) and int(mit_raw) > 0) else 0
        resp_raw = zone_df['Respected'].iloc[i]
        if resp_raw is None or (isinstance(resp_raw, float) and math.isnan(resp_raw)):
            respected = None
        else:
            respected = bool(resp_raw)
        records.append({'idx': i, 'direction': direction, 'top': top, 'bottom': bottom,
                        'mit_idx': mit_idx, 'respected': respected})
    return records


def compute_zone_proximity_analysis(df, zone_records, sample_step=20):
    """Race: when zones exist both above AND below price, which side gets mitigated first?"""
    if not zone_records:
        return {'above_wins': 0, 'below_wins': 0, 'total_races': 0,
                'closer_side_wins': 0, 'farther_side_wins': 0, 'distance_ratios': []}

    z_indices = np.array([r['idx'] for r in zone_records])
    z_tops = np.array([r['top'] for r in zone_records])
    z_bottoms = np.array([r['bottom'] for r in zone_records])
    z_mit_indices = np.array([r['mit_idx'] for r in zone_records])

    closes = df['close'].values
    n = len(df)

    above_wins = 0
    below_wins = 0
    closer_side_wins = 0
    farther_side_wins = 0
    distance_ratios = []

    for bar in range(0, n, sample_step):
        price = closes[bar]
        mask = (z_indices <= bar) & ((z_mit_indices == 0) | (z_mit_indices > bar))
        if not mask.any():
            continue

        active_idx = np.where(mask)[0]
        mids = (z_tops[active_idx] + z_bottoms[active_idx]) / 2
        dists = mids - price

        above_mask = dists > 0
        below_mask = dists < 0

        if not above_mask.any() or not below_mask.any():
            continue

        def closest_mitigated(side_mask):
            side_active = active_idx[side_mask]
            side_dists = np.abs(dists[side_mask])
            closest_order = np.argmin(side_dists)
            ai = side_active[closest_order]
            dist = side_dists[closest_order]
            mit = z_mit_indices[ai]
            if mit > 0 and mit > bar:
                return dist, mit
            return dist, None

        above_dist, above_mit = closest_mitigated(above_mask)
        below_dist, below_mit = closest_mitigated(below_mask)

        if above_mit is None and below_mit is None:
            continue

        if above_mit is not None and (below_mit is None or above_mit < below_mit):
            winner_side = 'above'
            winner_dist = above_dist
            loser_dist = below_dist
        elif below_mit is not None and (above_mit is None or below_mit < above_mit):
            winner_side = 'below'
            winner_dist = below_dist
            loser_dist = above_dist
        else:
            continue

        if winner_side == 'above':
            above_wins += 1
        else:
            below_wins += 1

        if winner_dist <= loser_dist:
            closer_side_wins += 1
        else:
            farther_side_wins += 1

        if loser_dist > 0:
            distance_ratios.append(round(float(winner_dist / loser_dist), 2))

    return {
        'above_wins': above_wins,
        'below_wins': below_wins,
        'total_races': above_wins + below_wins,
        'closer_side_wins': closer_side_wins,
        'farther_side_wins': farther_side_wins,
        'distance_ratios': distance_ratios,
    }


def compute_zone_feature_matrix(df, zone_records, bos):
    """Build a feature matrix for each zone to predict mitigation."""
    closes = df['close'].values
    n = len(df)

    # Precompute last_bos_direction array
    last_bos_dir = np.zeros(n, dtype=int)
    current_dir = 0
    for i in range(n):
        if not pd.isna(bos['BOS'].iloc[i]):
            current_dir = int(bos['BOS'].iloc[i])
        last_bos_dir[i] = current_dir

    features = []
    for fi, f_rec in enumerate(zone_records):
        idx = f_rec['idx']
        top = f_rec['top']
        bottom = f_rec['bottom']
        mid = (top + bottom) / 2
        direction = f_rec['direction']
        price = closes[idx] if idx < n else mid

        size_pct = (top - bottom) / mid * 100 if mid != 0 else 0
        distance_pct = abs(mid - price) / price * 100 if price != 0 else 0
        side = 'above' if mid > price else 'below'
        trend_aligned = 1 if (last_bos_dir[idx] == direction) else 0

        # Count competing active zones and compute proximity rank
        num_competing = 0
        same_side_dists = []
        my_dist = abs(mid - price)

        for fj, other in enumerate(zone_records):
            if fj == fi:
                continue
            if other['idx'] > idx:
                continue
            if other['mit_idx'] > 0 and other['mit_idx'] <= idx:
                continue
            num_competing += 1
            other_mid = (other['top'] + other['bottom']) / 2
            other_side = 'above' if other_mid > price else 'below'
            if other_side == side:
                same_side_dists.append(abs(other_mid - price))

        closer_count = sum(1 for d in same_side_dists if d < my_dist)
        proximity_rank = closer_count + 1

        was_mitigated = f_rec['mit_idx'] > 0
        candles_to_mit = (f_rec['mit_idx'] - idx) if was_mitigated else None

        features.append({
            'size_pct': round(size_pct, 2),
            'distance_pct': round(distance_pct, 2),
            'side': side,
            'direction': direction,
            'trend_aligned': trend_aligned,
            'num_competing': num_competing,
            'proximity_rank': proximity_rank,
            'was_mitigated': was_mitigated,
            'candles_to_mitigation': candles_to_mit,
            'was_respected': f_rec['respected'],
        })

    return features


def compute_distance_vs_speed(zones_df):
    resolved = zones_df[zones_df['candles_to_mitigation'].notna()].copy()
    if resolved.empty:
        return {}
    bins = [0, 0.5, 1, 2, 3, 5, 10, 100]
    labels = ['0-0.5%', '0.5-1%', '1-2%', '2-3%', '3-5%', '5-10%', '10%+']
    resolved['dist_bucket'] = pd.cut(resolved['distance_pct'], bins=bins, labels=labels, right=True)

    result = {}
    for zt, g in resolved.groupby('zone_type'):
        bucket_stats = g.groupby('dist_bucket', observed=True)['candles_to_mitigation'].agg(['mean', 'count'])
        result[zt] = {
            'buckets': bucket_stats.index.astype(str).tolist(),
            'mean_candles': bucket_stats['mean'].tolist(),
            'counts': bucket_stats['count'].tolist(),
        }
    return result


# ---------------------------------------------------------------------------
# 5. Build unified HTML
# ---------------------------------------------------------------------------

def build_html(symbol, timeframe, stats_df, survival_curves, race_rates,
               total_races, dist_vs_speed, zones_df, df=None,
               reversal_data=None, ob_features=None,
               fvg_proximity=None, fvg_features=None,
               ob_proximity=None, ob_magnet_features=None):

    # Zone histogram data
    hist_data = {}
    for zt, g in zones_df.groupby('zone_type'):
        hist_data[zt] = g['candles_to_mitigation'].dropna().tolist()

    stats_html = stats_df.to_html(index=False, classes='stats-table', border=0)

    empty_proximity = {'above_wins': 0, 'below_wins': 0, 'total_races': 0, 'closer_side_wins': 0, 'farther_side_wins': 0, 'distance_ratios': []}
    reversal_json = json.dumps(reversal_data or [])
    ob_features_json = json.dumps(ob_features or [])
    fvg_proximity_json = json.dumps(fvg_proximity or empty_proximity)
    fvg_features_json = json.dumps(fvg_features or [])
    ob_proximity_json = json.dumps(ob_proximity or empty_proximity)
    ob_magnet_features_json = json.dumps(ob_magnet_features or [])

    total_zones = len(zones_df)
    total_candles = len(df) if df is not None else 0
    if df is not None and len(df) > 0 and 'time' in df.columns:
        date_start = pd.to_datetime(df['time'].iloc[0]).strftime('%Y-%m-%d')
        date_end = pd.to_datetime(df['time'].iloc[-1]).strftime('%Y-%m-%d')
        date_range_str = f"{date_start} → {date_end}"
    else:
        date_range_str = ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{symbol} ({timeframe}) — SMC Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{ background:#131722; color:#d1d4dc; font-family:-apple-system,BlinkMacSystemFont,sans-serif; padding:20px; margin:0; }}
h1, h2 {{ color:#e0e3eb; }}
h1 {{ font-size:24px; margin-bottom:4px; }}
.subtitle {{ color:#787b86; font-size:14px; margin-bottom:16px; }}

/* Tabs */
.tab-bar {{ display:flex; gap:0; border-bottom:2px solid #2b2b43; margin-bottom:24px; }}
.tab-btn {{
    background:none; border:none; color:#787b86; font-size:15px; padding:10px 24px;
    cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-2px;
    font-family:inherit; font-weight:600;
}}
.tab-btn:hover {{ color:#d1d4dc; }}
.tab-btn.active {{ color:#2962ff; border-bottom-color:#2962ff; }}
.tab-btn.disabled {{ color:#3b3b4f; cursor:default; }}
.tab-content {{ display:none; }}
.tab-content.active {{ display:block; }}

.section {{ margin-bottom:40px; }}
.chart-row {{ display:flex; flex-wrap:wrap; gap:16px; }}
.chart-box {{ flex:1; min-width:350px; }}
.stats-table {{ border-collapse:collapse; width:100%; margin-bottom:16px; }}
.stats-table th, .stats-table td {{ padding:6px 12px; text-align:left; border-bottom:1px solid #2b2b43; }}
.stats-table th {{ color:#787b86; font-weight:600; }}
.explanation {{ color:#787b86; font-size:14px; max-width:800px; margin-bottom:16px; line-height:1.5; }}
.explanation strong {{ color:#d1d4dc; }}
.placeholder {{ color:#3b3b4f; font-size:18px; text-align:center; padding:80px 20px; }}
</style>
</head>
<body>
<h1>SMC Dashboard</h1>
<p class="subtitle">{symbol} · {timeframe} · {date_range_str} · {total_candles:,} candles · {total_zones} zones</p>

<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('zones')">Zones</button>
  <button class="tab-btn" onclick="showTab('reversal')">Reversal Quality</button>
  <button class="tab-btn" onclick="showTab('obanalysis')">OB Analysis</button>
  <button class="tab-btn" onclick="showTab('fvgmagnet')">FVG Magnet</button>
  <button class="tab-btn" onclick="showTab('obmagnet')">OB Magnet</button>
  <button class="tab-btn" onclick="showTab('pricedist')">Price Distribution</button>
</div>

<!-- ==================== TAB 1: ZONES ==================== -->
<div id="tab-zones" class="tab-content active">

<div class="section">
<h2>Glossary</h2>
<div class="explanation">
<p><strong>FVG (Fair Value Gap)</strong>: A price gap where one candle's range doesn't overlap the next, indicating imbalance. <strong>Bullish FVG</strong> = gap upward, <strong>Bearish FVG</strong> = gap downward.</p>
<p><strong>OB (Order Block)</strong>: The last candle before a strong price move — marks where institutions likely placed large orders. <strong>Bullish OB</strong> = demand zone, <strong>Bearish OB</strong> = supply zone.</p>
<p><strong>Inflexion</strong>: A swing high or swing low pivot point. <strong>Bull (concave)</strong> = resistance at swing high, <strong>Bear (convex)</strong> = support at swing low.</p>
<p><strong>Mitigation</strong> = price returns to touch/fill a zone. <strong>Respected</strong> = zone held. <strong>Disrespected</strong> = price broke through.</p>
</div>
</div>

<div class="section">
<h2>Summary Statistics</h2>
<p class="explanation">Each row is a zone type + direction. <strong>Mitigation rate</strong> = % of zones price returned to. <strong>Respect rate</strong> = of mitigated zones, % where price bounced. <strong>Mean/Median/Std candles</strong> = how many candles to mitigation.</p>
{stats_html}
</div>

<div class="section">
<h2>Distribution of Candles-to-Mitigation</h2>
<p class="explanation">Right-skewed = most zones are filled quickly with a long tail of persistent zones.</p>
<div class="chart-row">
  <div class="chart-box" id="hist-FVG"></div>
  <div class="chart-box" id="hist-OB"></div>
  <div class="chart-box" id="hist-Inflexion"></div>
</div>
</div>

<div class="section">
<h2>Survival Curves (% still unmitigated after N candles)</h2>
<p class="explanation">Steeper initial drop = zone type gets filled faster.</p>
<div id="survival-chart"></div>
</div>

<div class="section">
<h2>Race Analysis — Which zone type is hit first? ({total_races} races)</h2>
<p class="explanation">At regular intervals, check which simultaneously-active zone types exist, then track which price hits first.</p>
<div id="race-chart"></div>
</div>

<div class="section">
<h2>Distance vs Speed — Avg candles to hit by distance bucket</h2>
<p class="explanation">Zones bucketed by distance from price at creation. Taller bars = longer to reach. Small n = interpret cautiously.</p>
<div id="dist-chart"></div>
</div>

</div>

<!-- ==================== TAB 2: REVERSAL QUALITY ==================== -->
<div id="tab-reversal" class="tab-content">

<div class="section">
<h2>Glossary</h2>
<div class="explanation">
<p><strong>MFE (Max Favorable Excursion)</strong>: After a zone is touched (mitigated), how far did price move in the zone's expected direction before the zone was broken?</p>
<p><strong>Reference price</strong>: For FVG/OB — the high (bullish) or low (bearish) of the mitigation candle. For Inflexions — the level itself.</p>
<p><strong>Categories</strong>:</p>
<ul style="margin:4px 0;">
<li><strong>Direct disrespect</strong>: Broken on the same candle it was touched. No reversal opportunity.</li>
<li><strong>No reversal, then disrespect</strong>: Had a window between touch and break, but price never moved in the expected direction.</li>
<li><strong>Reversal then disrespect</strong>: Price did reverse (MFE &gt; 0) but eventually broke through anyway.</li>
<li><strong>Respected (held)</strong>: Zone was touched and never broken. MFE measured to end of data.</li>
<li><strong>Pending</strong>: Zone was never touched.</li>
</ul>
</div>
</div>

<div class="section">
<h2>Outcome Breakdown</h2>
<p class="explanation">Stacked bar showing the percentage of each category per zone type.</p>
<div id="rq-outcome"></div>
</div>

<div class="section">
<h2>MFE Distribution (zones with MFE &gt; 0)</h2>
<p class="explanation">How far price reversed in the expected direction. Only includes zones where a reversal actually occurred.</p>
<div class="chart-row">
  <div class="chart-box" id="rq-mfe-FVG"></div>
  <div class="chart-box" id="rq-mfe-OB"></div>
  <div class="chart-box" id="rq-mfe-Inflexion"></div>
</div>
</div>

<div class="section">
<h2>Avg MFE by Zone Type &amp; Direction</h2>
<p class="explanation">Average MFE for zones with MFE &gt; 0, split by bullish vs bearish.</p>
<div id="rq-mfe-grouped"></div>
</div>

<div class="section">
<h2>Time to Disrespect (candles between mitigation and break)</h2>
<p class="explanation">How long zones survived after first touch before being broken. Only includes disrespected zones with a window (time &gt; 0).</p>
<div class="chart-row">
  <div class="chart-box" id="rq-ttd-FVG"></div>
  <div class="chart-box" id="rq-ttd-OB"></div>
  <div class="chart-box" id="rq-ttd-Inflexion"></div>
</div>
</div>

</div>

<!-- ==================== TAB 3: OB ANALYSIS ==================== -->
<div id="tab-obanalysis" class="tab-content">

<div class="section">
<h2>Glossary</h2>
<div class="explanation">
<p>This tab analyzes <strong>Order Block features</strong> to find what predicts whether an OB will be <strong>respected</strong> (price bounces) vs <strong>disrespected</strong> (price breaks through).</p>
<p><strong>Zone Size %</strong>: Height of the OB zone relative to price. <strong>Impulse Strength %</strong>: Price move from OB creation to BOS confirmation. <strong>Impulse Speed</strong>: Number of candles from OB to BOS. <strong>BOS Candle Body %</strong>: Body size of the BOS candle. <strong>Volume Ratio</strong>: Volume at OB creation vs 20-bar avg. <strong>Distance %</strong>: How far the OB was from price at creation.</p>
</div>
</div>

<div style="margin: 10px 0;">
  <a href="/charts/ob_mfe_highlight.html"
     target="_blank"
     style="display:inline-block; padding:8px 16px; background:#26a69a; color:#fff; border-radius:4px; text-decoration:none; font-size:13px;">
     Open OB MFE Chart ↗
  </a>
</div>

<div style="margin: 10px 0; display:flex; align-items:center; gap:10px;">
  <label style="color:#d1d4dc; font-size:13px;">Min MFE %:</label>
  <input type="range" id="ob-mfe-slider" min="0" max="10" step="0.5" value="0"
         style="width:200px;">
  <span id="ob-mfe-val" style="color:#d1d4dc; font-size:13px;">0.0%</span>
</div>

<div class="section">
<h2>Feature Comparison — Respected vs Disrespected</h2>
<p class="explanation">Overlapping histograms: <span style="color:#26a69a;">green = respected</span>, <span style="color:#ef5350;">red = disrespected</span>. Separation indicates predictive power.</p>
<div class="chart-row">
  <div class="chart-box" id="ob-fc-zone_size_pct"></div>
  <div class="chart-box" id="ob-fc-impulse_strength_pct"></div>
</div>
<div class="chart-row">
  <div class="chart-box" id="ob-fc-impulse_speed"></div>
  <div class="chart-box" id="ob-fc-bos_candle_body_pct"></div>
</div>
<div class="chart-row">
  <div class="chart-box" id="ob-fc-volume_ratio"></div>
  <div class="chart-box" id="ob-fc-distance_pct"></div>
</div>
</div>

<div class="section">
<h2>Respect Rate by Feature Bucket</h2>
<p class="explanation">Bar charts showing what % of OBs in each bucket were respected. Labels show sample size (n=).</p>
<div class="chart-row">
  <div class="chart-box" id="ob-rb-zone_size_pct"></div>
  <div class="chart-box" id="ob-rb-impulse_strength_pct"></div>
</div>
<div class="chart-row">
  <div class="chart-box" id="ob-rb-impulse_speed"></div>
  <div class="chart-box" id="ob-rb-volume_ratio"></div>
</div>
</div>

<div class="section">
<h2>Correlation Heatmap</h2>
<p class="explanation">Pearson correlation between features and the respected outcome (1=respected, 0=disrespected). Values near +1/-1 indicate strong relationships.</p>
<div id="ob-corr-heatmap"></div>
</div>

<div class="section">
<h2>Summary Table — Respected vs Disrespected</h2>
<p class="explanation">Mean/median of each feature split by respected (price bounced off the OB) vs disrespected (price closed through it). <strong>Zone Size %</strong> = OB height as a percentage of price. <strong>Impulse Strength %</strong> = size of the move that created the OB, as % of price. <strong>Impulse Speed</strong> = number of candles from the OB origin to the break-of-structure candle (fewer = more impulsive). <strong>BOS Candle Body %</strong> = body size of the BOS candle as % of price. <strong>Volume Ratio</strong> = volume at the OB relative to recent average. <strong>Distance %</strong> = how far the OB was from price when formed. <strong>Δ Mean</strong> = respected mean minus disrespected mean.</p>
<div id="ob-summary-table"></div>
</div>

</div>

<!-- ==================== TAB 5: FVG MAGNET ==================== -->
<div id="tab-fvgmagnet" class="tab-content">

<div class="section">
<h2>Glossary</h2>
<div class="explanation">
<p>This tab answers: <strong>when multiple FVGs are active, which one does price move to next?</strong></p>
<p><strong>Proximity Rank</strong>: Rank 1 = closest FVG to current price on that side, Rank 2 = second closest, etc.</p>
<p><strong>Trend Aligned</strong>: FVG direction matches the last BOS/CHoCH direction (1 = aligned, 0 = counter-trend).</p>
<p><strong>Mitigated</strong>: Price returned to fill the FVG. <strong>Unmitigated</strong>: FVG never touched.</p>
</div>
</div>

<div class="section">
<h2>Above vs Below Race — Which side gets mitigated first?</h2>
<p class="explanation">For each candle, we find the closest FVG above and below price, then see which one price reaches first — that side "wins." The bar chart shows how often above vs below won. The histogram shows winner_distance / loser_distance: below 1 means price went to the nearer FVG, above 1 means it skipped to the farther one. Most ratios cluster well below 1, confirming price gravitates toward whichever FVG is closer.</p>
<div class="chart-row">
  <div class="chart-box" id="fvgm-proximity"></div>
  <div class="chart-box" id="fvgm-dist-ratio"></div>
</div>
</div>

<div class="section">
<h2>Speed Scatter — Distance vs Candles to Mitigation</h2>
<p class="explanation">Each dot is a mitigated FVG. X = distance from price at creation, Y = how many candles until filled. Color = direction.</p>
<div id="fvgm-speed-scatter"></div>
</div>

<div class="section">
<h2>Summary Table</h2>
<p class="explanation">Mean/median of each feature split by respected (price bounced off the FVG) vs disrespected (price closed through it). <strong>Size %</strong> = FVG height as a percentage of price. <strong>Distance %</strong> = how far the FVG was from price when it formed. <strong># Competing</strong> = number of other active FVGs in the same direction at creation time. <strong>Proximity Rank</strong> = rank among competing FVGs by distance to price (1 = closest). <strong>Trend Aligned</strong> = 1 if the FVG direction matches the higher-timeframe trend, 0 otherwise. <strong>Δ Mean</strong> = respected mean minus disrespected mean (green = respected FVGs score higher).</p>
<div id="fvgm-summary-table"></div>
</div>

</div>

<!-- ==================== TAB 6: OB MAGNET ==================== -->
<div id="tab-obmagnet" class="tab-content">

<div class="section">
<h2>Glossary</h2>
<div class="explanation">
<p>This tab answers: <strong>when multiple OBs are active, which one does price move to next?</strong></p>
<p><strong>Proximity Rank</strong>: Rank 1 = closest OB to current price on that side, Rank 2 = second closest, etc.</p>
<p><strong>Trend Aligned</strong>: OB direction matches the last BOS/CHoCH direction (1 = aligned, 0 = counter-trend).</p>
<p><strong>Mitigated</strong>: Price returned to fill the OB. <strong>Unmitigated</strong>: OB never touched.</p>
</div>
</div>

<div class="section">
<h2>Above vs Below Race — Which side gets mitigated first?</h2>
<p class="explanation">For each candle, we find the closest OB above and below price, then see which one price reaches first — that side "wins." The bar chart shows how often above vs below won. The histogram shows winner_distance / loser_distance: below 1 means price went to the nearer OB, above 1 means it skipped to the farther one.</p>
<div class="chart-row">
  <div class="chart-box" id="obm-proximity"></div>
  <div class="chart-box" id="obm-dist-ratio"></div>
</div>
</div>

<div class="section">
<h2>Speed Scatter — Distance vs Candles to Mitigation</h2>
<p class="explanation">Each dot is a mitigated OB. X = distance from price at creation, Y = how many candles until filled. Color = direction.</p>
<div id="obm-speed-scatter"></div>
</div>

<div class="section">
<h2>Summary Table</h2>
<p class="explanation">Mean/median of each feature split by respected (price bounced off the OB) vs disrespected (price closed through it). <strong>Size %</strong> = OB height as a percentage of price. <strong>Distance %</strong> = how far the OB was from price when it formed. <strong># Competing</strong> = number of other active OBs at creation time. <strong>Proximity Rank</strong> = rank among competing OBs by distance to price (1 = closest). <strong>Trend Aligned</strong> = 1 if the OB direction matches the higher-timeframe trend, 0 otherwise. <strong>Δ Mean</strong> = respected mean minus disrespected mean (green = respected OBs score higher).</p>
<div id="obm-summary-table"></div>
</div>

</div>

<!-- ==================== TAB 4: PRICE DISTRIBUTION ==================== -->
<div id="tab-pricedist" class="tab-content">

<div class="section">
<h2>Price Distribution Generator</h2>
<div class="explanation">
<p>Generate a bar-by-bar walkthrough showing candlestick chart (left) + price distribution histogram and zone bands (right). Adjust parameters below and click <strong>Generate</strong>.</p>
<p><strong>Symbol:</strong> {symbol} &nbsp; <strong>Timeframe:</strong> {timeframe}</p>
</div>
</div>

<div class="section" style="max-width:500px;">
  <div style="margin-bottom:12px;">
    <label style="display:block; color:#787b86; font-size:13px; margin-bottom:4px;">DataFrame size (candles from end of data)</label>
    <input type="range" id="pd-size" min="100" max="5000" step="50" value="1250" style="width:100%;">
    <span id="pd-size-val" style="color:#d1d4dc; font-size:13px;">1250</span>
  </div>
  <div style="margin-bottom:12px;">
    <label style="display:block; color:#787b86; font-size:13px; margin-bottom:4px;">Bins</label>
    <input type="number" id="pd-bins" value="80" min="10" max="500" style="background:#1e222d; color:#d1d4dc; border:1px solid #2b2b43; padding:4px 8px; width:80px;">
  </div>
  <div style="margin-bottom:12px;">
    <label style="display:block; color:#787b86; font-size:13px; margin-bottom:4px;">Swing Length</label>
    <input type="number" id="pd-swing" value="50" min="5" max="200" style="background:#1e222d; color:#d1d4dc; border:1px solid #2b2b43; padding:4px 8px; width:80px;">
  </div>
  <div style="margin-bottom:12px;">
    <label style="display:block; color:#787b86; font-size:13px; margin-bottom:4px;">Step (candles per frame)</label>
    <input type="number" id="pd-step" value="1" min="1" max="50" style="background:#1e222d; color:#d1d4dc; border:1px solid #2b2b43; padding:4px 8px; width:80px;">
  </div>
  <button id="pd-generate-btn" onclick="generatePriceDist()" style="background:#2962ff; color:#fff; border:none; padding:10px 24px; cursor:pointer; font-size:14px; font-weight:600; border-radius:4px;">Generate</button>
  <span id="pd-status" style="color:#787b86; font-size:13px; margin-left:12px;"></span>
</div>

</div>

<script>
// --- Price Distribution generator ---
document.getElementById('pd-size').addEventListener('input', function() {{
    document.getElementById('pd-size-val').textContent = this.value;
}});

function generatePriceDist() {{
    const btn = document.getElementById('pd-generate-btn');
    const status = document.getElementById('pd-status');
    btn.disabled = true;
    status.textContent = 'Generating... (this may take a while)';
    status.style.color = '#f0b90b';

    // Open window immediately (user gesture) to avoid popup blocker
    const newWin = window.open('about:blank', '_blank');
    if (newWin) {{
        newWin.document.write('<html><body style="background:#131722;color:#d1d4dc;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;"><h2>Generating price distribution...</h2></body></html>');
    }}

    const params = new URLSearchParams({{
        symbol: '{symbol}',
        timeframe: '{timeframe}',
        size: document.getElementById('pd-size').value,
        bins: document.getElementById('pd-bins').value,
        swing_length: document.getElementById('pd-swing').value,
        step: document.getElementById('pd-step').value,
    }});

    fetch('/api/generate-price-dist?' + params)
        .then(r => r.json())
        .then(data => {{
            if (data.error) {{
                status.textContent = 'Error: ' + data.error;
                status.style.color = '#ef5350';
                if (newWin) newWin.close();
            }} else {{
                status.textContent = 'Done — opened in new tab';
                status.style.color = '#26a69a';
                if (newWin) newWin.location.href = data.url;
            }}
            btn.disabled = false;
        }})
        .catch(err => {{
            status.textContent = 'Error: ' + err.message;
            status.style.color = '#ef5350';
            if (newWin) newWin.close();
            btn.disabled = false;
        }});
}}

// --- Tab switching ---
function showTab(tabId) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    const tab = document.getElementById('tab-' + tabId);
    if (tab) {{
        tab.classList.add('active');
        // Find the button that triggered this
        document.querySelectorAll('.tab-btn').forEach(btn => {{
            if (btn.getAttribute('onclick').includes("'" + tabId + "'")) btn.classList.add('active');
        }});
        // Trigger Plotly resize for charts that were hidden
        tab.querySelectorAll('[class*="js-plotly-plot"]').forEach(el => Plotly.Plots.resize(el));
    }}
}}

const COLORS = {{FVG:'#f0b90b', OB:'#9b59b6', Inflexion:'#3498db'}};
const BULL = '#26a69a';
const BEAR = '#ef5350';
const plotBg = '#131722';
const paperBg = '#131722';
const gridColor = '#2b2b43';
const fontColor = '#d1d4dc';
const layout_base = {{
    paper_bgcolor: paperBg, plot_bgcolor: plotBg,
    font: {{color: fontColor}},
    xaxis: {{gridcolor: gridColor}},
    yaxis: {{gridcolor: gridColor}},
    margin: {{l:50, r:20, t:40, b:40}},
}};

// ==================== ZONE CHARTS ====================

const histData = {json.dumps(hist_data)};
for (const [zt, vals] of Object.entries(histData)) {{
    const el = document.getElementById('hist-' + zt);
    if (!el || vals.length === 0) continue;
    Plotly.newPlot(el, [{{
        x: vals, type:'histogram', marker:{{color: COLORS[zt]}}, nbinsx:40, name:zt,
    }}], {{
        ...layout_base,
        title: {{text: zt + ' — Candles to Mitigation', font:{{size:14, color:fontColor}}}},
        xaxis: {{...layout_base.xaxis, title:'Candles'}},
        yaxis: {{...layout_base.yaxis, title:'Count'}},
        height: 300,
    }}, {{responsive:true}});
}}

const survivalData = {json.dumps(survival_curves)};
const survTraces = [];
for (const [zt, curve] of Object.entries(survivalData)) {{
    survTraces.push({{
        x: curve.x, y: curve.y, mode:'lines', name: zt,
        line: {{color: COLORS[zt], width:2}},
    }});
}}
Plotly.newPlot('survival-chart', survTraces, {{
    ...layout_base,
    title: {{text:'Zone Survival Curves', font:{{size:14, color:fontColor}}}},
    xaxis: {{...layout_base.xaxis, title:'Candles since creation'}},
    yaxis: {{...layout_base.yaxis, title:'% still unmitigated', range:[0,105]}},
    height: 400,
}}, {{responsive:true}});

const raceRates = {json.dumps(race_rates)};
const raceTypes = Object.keys(raceRates).sort((a,b) => raceRates[b]-raceRates[a]);
Plotly.newPlot('race-chart', [{{
    x: raceTypes, y: raceTypes.map(t => raceRates[t]),
    type:'bar', marker: {{color: raceTypes.map(t => COLORS[t] || '#888')}},
    text: raceTypes.map(t => raceRates[t].toFixed(1) + '%'), textposition:'outside',
}}], {{
    ...layout_base,
    title: {{text:'Race Win Rate — first zone type hit', font:{{size:14, color:fontColor}}}},
    yaxis: {{...layout_base.yaxis, title:'Win Rate %', range:[0, Math.max(...Object.values(raceRates))*1.3]}},
    height: 350,
}}, {{responsive:true}});

const distData = {json.dumps(dist_vs_speed)};
const distTraces = [];
for (const [zt, d] of Object.entries(distData)) {{
    distTraces.push({{
        x: d.buckets, y: d.mean_candles, type:'bar', name: zt,
        marker: {{color: COLORS[zt]}},
        text: d.counts.map(c => 'n=' + c), textposition:'outside',
    }});
}}
Plotly.newPlot('dist-chart', distTraces, {{
    ...layout_base,
    title: {{text:'Avg Candles to Mitigation by Distance at Creation', font:{{size:14, color:fontColor}}}},
    xaxis: {{...layout_base.xaxis, title:'Distance from price at creation'}},
    yaxis: {{...layout_base.yaxis, title:'Avg candles to mitigation'}},
    barmode: 'group', height: 400,
}}, {{responsive:true}});

// ==================== REVERSAL QUALITY CHARTS ====================

const rqData = {reversal_json};

function renderReversalQuality() {{
    const zoneTypes = ['FVG', 'OB', 'Inflexion'];
    const categories = ['Direct disrespect', 'No reversal, then disrespect', 'Reversal then disrespect', 'Respected (held)', 'Pending'];
    const catColors = {{'Direct disrespect':'#b71c1c', 'No reversal, then disrespect':'#e65100', 'Reversal then disrespect':'#f9a825', 'Respected (held)':'#00897b', 'Pending':'#546e7a'}};

    // Outcome stacked bar
    const outcomeTraces = [];
    for (const cat of categories) {{
        const xVals = [];
        const yVals = [];
        for (const zt of zoneTypes) {{
            const total = rqData.filter(r => r.zone_type === zt).length;
            const count = rqData.filter(r => r.zone_type === zt && r.category === cat).length;
            xVals.push(zt);
            yVals.push(total > 0 ? count / total * 100 : 0);
        }}
        outcomeTraces.push({{
            x: xVals, y: yVals, type: 'bar', name: cat,
            marker: {{color: catColors[cat]}},
            text: zoneTypes.map(zt => {{
                const count = rqData.filter(r => r.zone_type === zt && r.category === cat).length;
                return 'n=' + count;
            }}),
            textposition: 'inside',
            hovertemplate: '%{{x}}: %{{y:.1f}}% (%{{text}})<extra>' + cat + '</extra>',
        }});
    }}
    Plotly.newPlot('rq-outcome', outcomeTraces, {{
        ...layout_base,
        title: {{text: 'Outcome Breakdown by Zone Type', font: {{size: 14, color: fontColor}}}},
        xaxis: {{...layout_base.xaxis, title: 'Zone Type'}},
        yaxis: {{...layout_base.yaxis, title: '% of Zones', range: [0, 105]}},
        barmode: 'stack', height: 400,
    }}, {{responsive: true}});

    // MFE histograms per zone type — exclude "Respected (held)" to avoid end-of-data bias
    for (const zt of zoneTypes) {{
        const vals = rqData.filter(r => r.zone_type === zt && r.mfe_pct !== null && r.mfe_pct > 0 && r.category !== 'Respected (held)').map(r => r.mfe_pct);
        const el = document.getElementById('rq-mfe-' + zt);
        if (!el) continue;
        if (vals.length === 0) {{
            el.innerHTML = '<p style="color:#546e7a;text-align:center;padding:40px;">No reversals for ' + zt + '</p>';
            continue;
        }}
        // Use percentile-based x-axis cap to avoid outlier stretch
        const sorted = [...vals].sort((a, b) => a - b);
        const p95 = sorted[Math.floor(sorted.length * 0.95)];
        const xMax = Math.max(p95 * 1.2, sorted[Math.min(5, sorted.length - 1)]);
        const binSize = Math.max((xMax - 0) / 25, 0.05);
        Plotly.newPlot(el, [{{
            x: vals, type: 'histogram', marker: {{color: COLORS[zt]}},
            xbins: {{start: 0, end: xMax, size: binSize}},
        }}], {{
            ...layout_base,
            title: {{text: zt + ' — MFE Distribution (%, disrespected only)', font: {{size: 14, color: fontColor}}}},
            xaxis: {{...layout_base.xaxis, title: 'MFE %', range: [0, xMax], type: 'linear'}},
            yaxis: {{...layout_base.yaxis, title: 'Count'}},
            height: 300,
        }}, {{responsive: true}});
    }}

    // Avg MFE grouped bar by direction
    const mfeTraces = [];
    for (const dir of [1, -1]) {{
        const dirLabel = dir === 1 ? 'Bullish' : 'Bearish';
        const dirColor = dir === 1 ? BULL : BEAR;
        const xVals = [];
        const yVals = [];
        const textVals = [];
        for (const zt of zoneTypes) {{
            const vals = rqData.filter(r => r.zone_type === zt && r.direction === dir && r.mfe_pct !== null && r.mfe_pct > 0 && r.category !== 'Respected (held)').map(r => r.mfe_pct);
            xVals.push(zt);
            yVals.push(vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0);
            textVals.push('n=' + vals.length);
        }}
        mfeTraces.push({{
            x: xVals, y: yVals, type: 'bar', name: dirLabel,
            marker: {{color: dirColor}},
            text: textVals, textposition: 'outside',
        }});
    }}
    Plotly.newPlot('rq-mfe-grouped', mfeTraces, {{
        ...layout_base,
        title: {{text: 'Avg MFE by Zone Type & Direction (disrespected, MFE > 0)', font: {{size: 14, color: fontColor}}}},
        xaxis: {{...layout_base.xaxis, title: 'Zone Type'}},
        yaxis: {{...layout_base.yaxis, title: 'Avg MFE %'}},
        barmode: 'group', height: 400,
    }}, {{responsive: true}});

    // Time to disrespect histograms
    for (const zt of zoneTypes) {{
        const vals = rqData.filter(r => r.zone_type === zt && r.time_to_disrespect !== null && r.time_to_disrespect > 0).map(r => r.time_to_disrespect);
        const el = document.getElementById('rq-ttd-' + zt);
        if (!el) continue;
        if (vals.length === 0) {{
            el.innerHTML = '<p style="color:#546e7a;text-align:center;padding:40px;">No windowed disrespects for ' + zt + '</p>';
            continue;
        }}
        const sortedTtd = [...vals].sort((a, b) => a - b);
        const p95ttd = sortedTtd[Math.floor(sortedTtd.length * 0.95)];
        const ttdMax = Math.max(p95ttd * 1.2, 20);
        Plotly.newPlot(el, [{{
            x: vals, type: 'histogram', marker: {{color: COLORS[zt]}},
            xbins: {{start: 0, end: ttdMax, size: 5}},
        }}], {{
            ...layout_base,
            title: {{text: zt + ' — Time to Disrespect (candles)', font: {{size: 14, color: fontColor}}}},
            xaxis: {{...layout_base.xaxis, title: 'Candles', type: 'linear', range: [0, ttdMax]}},
            yaxis: {{...layout_base.yaxis, title: 'Count'}},
            height: 300,
        }}, {{responsive: true}});
    }}
}}

renderReversalQuality();

// ==================== OB ANALYSIS CHARTS ====================

const obFeatures = {ob_features_json};

function renderOBAnalysis(mfeThreshold) {{
    mfeThreshold = mfeThreshold || 0;
    let groupA, groupB, labelA, labelB;

    if (mfeThreshold > 0) {{
        // MFE-based split: only resolved OBs with non-null mfe_pct
        const resolved = obFeatures.filter(f => f.category !== 'Pending' && f.mfe_pct !== null);
        groupA = resolved.filter(f => f.mfe_pct >= mfeThreshold);
        groupB = resolved.filter(f => f.mfe_pct < mfeThreshold);
        labelA = 'MFE ≥ ' + mfeThreshold.toFixed(1) + '% (n=' + groupA.length + ')';
        labelB = 'MFE < ' + mfeThreshold.toFixed(1) + '% (n=' + groupB.length + ')';
    }} else {{
        // Category-based split (original behaviour)
        groupA = obFeatures.filter(f => f.category === 'Respected (held)' || f.category === 'Reversal then disrespect');
        groupB = obFeatures.filter(f => f.category === 'Direct disrespect' || f.category === 'No reversal, then disrespect');
        labelA = 'Produced reversal (n=' + groupA.length + ')';
        labelB = 'No reversal (n=' + groupB.length + ')';
    }}

    // Aliases for backward compat inside helpers
    const producedReversal = groupA;
    const noReversal = groupB;

    if (producedReversal.length === 0 && noReversal.length === 0) return;

    // --- Feature Comparison Histograms ---
    function plotFeatureComparison(elId, feature, title, xLabel) {{
        const el = document.getElementById(elId);
        if (!el) return;
        const rVals = producedReversal.map(f => f[feature]).filter(v => v !== null);
        const dVals = noReversal.map(f => f[feature]).filter(v => v !== null);
        if (rVals.length === 0 && dVals.length === 0) return;

        const allVals = [...rVals, ...dVals].sort((a, b) => a - b);
        const p95 = allVals[Math.floor(allVals.length * 0.95)];
        const xMax = p95 * 1.1;

        Plotly.newPlot(el, [
            {{x: rVals, type: 'histogram', name: labelA, marker: {{color: 'rgba(38,166,154,0.7)'}}, nbinsx: 30}},
            {{x: dVals, type: 'histogram', name: labelB, marker: {{color: 'rgba(239,83,80,0.7)'}}, nbinsx: 30}},
        ], {{
            ...layout_base,
            title: {{text: title, font: {{size: 14, color: fontColor}}}},
            xaxis: {{...layout_base.xaxis, type: 'linear', title: xLabel, range: [0, xMax]}},
            yaxis: {{...layout_base.yaxis, title: 'Count'}},
            barmode: 'group', height: 300,
            legend: {{x: 0.6, y: 0.95, font: {{size: 11}}}},
        }}, {{responsive: true}});
    }}

    plotFeatureComparison('ob-fc-zone_size_pct', 'zone_size_pct', 'Zone Size %', 'Zone Size %');
    plotFeatureComparison('ob-fc-impulse_strength_pct', 'impulse_strength_pct', 'Impulse Strength %', 'Impulse Strength %');
    plotFeatureComparison('ob-fc-impulse_speed', 'impulse_speed', 'Impulse Speed (candles)', 'Candles');
    plotFeatureComparison('ob-fc-bos_candle_body_pct', 'bos_candle_body_pct', 'BOS Candle Body %', 'Body %');
    plotFeatureComparison('ob-fc-volume_ratio', 'volume_ratio', 'Volume Ratio', 'Ratio');
    plotFeatureComparison('ob-fc-distance_pct', 'distance_pct', 'Distance from Price %', 'Distance %');

    // --- Reversal Rate by Bucket ---
    function plotReversalByBucket(elId, feature, title, edges, labels) {{
        const el = document.getElementById(elId);
        if (!el) return;
        const resolved = mfeThreshold > 0
            ? obFeatures.filter(f => f.category !== 'Pending' && f.mfe_pct !== null)
            : obFeatures.filter(f => f.category !== 'Pending');
        const bucketRates = [];
        const bucketCounts = [];
        for (let b = 0; b < edges.length - 1; b++) {{
            const lo = edges[b];
            const hi = edges[b + 1];
            const inBucket = resolved.filter(f => f[feature] >= lo && f[feature] < hi);
            const nRev = mfeThreshold > 0
                ? inBucket.filter(f => f.mfe_pct >= mfeThreshold).length
                : inBucket.filter(f => f.category === 'Respected (held)' || f.category === 'Reversal then disrespect').length;
            bucketRates.push(inBucket.length > 0 ? nRev / inBucket.length * 100 : 0);
            bucketCounts.push(inBucket.length);
        }}

        Plotly.newPlot(el, [{{
            x: labels, y: bucketRates, type: 'bar',
            marker: {{color: '#9b59b6'}},
            text: bucketCounts.map(c => 'n=' + c), textposition: 'outside',
        }}], {{
            ...layout_base,
            title: {{text: title, font: {{size: 14, color: fontColor}}}},
            xaxis: {{...layout_base.xaxis, title: feature}},
            yaxis: {{...layout_base.yaxis, title: 'Reversal Rate %', range: [0, 105]}},
            height: 300,
        }}, {{responsive: true}});
    }}

    plotReversalByBucket('ob-rb-zone_size_pct', 'zone_size_pct', 'Reversal Rate by Zone Size',
        [0, 0.1, 0.3, 0.5, 1, 2, 100], ['0-0.1%', '0.1-0.3%', '0.3-0.5%', '0.5-1%', '1-2%', '2%+']);
    plotReversalByBucket('ob-rb-impulse_strength_pct', 'impulse_strength_pct', 'Reversal Rate by Impulse Strength',
        [0, 0.5, 1, 2, 5, 10, 100], ['0-0.5%', '0.5-1%', '1-2%', '2-5%', '5-10%', '10%+']);
    plotReversalByBucket('ob-rb-impulse_speed', 'impulse_speed', 'Reversal Rate by Impulse Speed',
        [0, 2, 5, 10, 20, 50, 1000], ['1', '2-4', '5-9', '10-19', '20-49', '50+']);
    plotReversalByBucket('ob-rb-volume_ratio', 'volume_ratio', 'Reversal Rate by Volume Ratio',
        [0, 0.5, 0.8, 1.0, 1.5, 2.0, 100], ['<0.5', '0.5-0.8', '0.8-1.0', '1.0-1.5', '1.5-2.0', '2.0+']);

    // --- Correlation Heatmap ---
    const corrLabel = mfeThreshold > 0 ? 'High MFE' : 'Produced Reversal';
    const featureNames = ['zone_size_pct', 'impulse_strength_pct', 'impulse_speed', 'bos_candle_body_pct', 'volume_ratio', 'distance_pct', 'target'];
    const featureLabels = ['Zone Size %', 'Impulse Str %', 'Impulse Speed', 'BOS Body %', 'Volume Ratio', 'Distance %', corrLabel];
    const resolved = mfeThreshold > 0
        ? obFeatures.filter(f => f.category !== 'Pending' && f.mfe_pct !== null)
        : obFeatures.filter(f => f.category !== 'Pending');
    if (resolved.length > 5) {{
        // Build matrix
        const cols = featureNames.map(fn => {{
            if (fn === 'target') {{
                if (mfeThreshold > 0) return resolved.map(f => f.mfe_pct >= mfeThreshold ? 1 : 0);
                return resolved.map(f => (f.category === 'Respected (held)' || f.category === 'Reversal then disrespect') ? 1 : 0);
            }}
            return resolved.map(f => f[fn] || 0);
        }});

        function pearson(a, b) {{
            const n = a.length;
            if (n === 0) return 0;
            const meanA = a.reduce((s, v) => s + v, 0) / n;
            const meanB = b.reduce((s, v) => s + v, 0) / n;
            let num = 0, denA = 0, denB = 0;
            for (let i = 0; i < n; i++) {{
                const da = a[i] - meanA, db = b[i] - meanB;
                num += da * db;
                denA += da * da;
                denB += db * db;
            }}
            const den = Math.sqrt(denA * denB);
            return den > 0 ? num / den : 0;
        }}

        const corrMatrix = [];
        const annotations = [];
        for (let r = 0; r < featureNames.length; r++) {{
            const row = [];
            for (let c = 0; c < featureNames.length; c++) {{
                const val = pearson(cols[r], cols[c]);
                row.push(val);
                annotations.push({{
                    x: featureLabels[c], y: featureLabels[r],
                    text: val.toFixed(2),
                    showarrow: false, font: {{color: Math.abs(val) > 0.4 ? '#fff' : '#aaa', size: 12}},
                }});
            }}
            corrMatrix.push(row);
        }}

        Plotly.newPlot('ob-corr-heatmap', [{{
            z: corrMatrix, x: featureLabels, y: featureLabels,
            type: 'heatmap',
            colorscale: [[0, '#ef5350'], [0.5, '#1e222d'], [1, '#26a69a']],
            zmin: -1, zmax: 1, showscale: true,
            colorbar: {{title: 'r'}},
        }}], {{
            ...layout_base,
            title: {{text: 'Feature Correlation Matrix', font: {{size: 14, color: fontColor}}}},
            annotations: annotations,
            yaxis: {{...layout_base.yaxis, autorange: 'reversed'}},
            height: 450,
            margin: {{l: 120, r: 20, t: 40, b: 100}},
        }}, {{responsive: true}});
    }}

    // --- Summary Table ---
    const summaryFeatures = ['zone_size_pct', 'impulse_strength_pct', 'impulse_speed', 'bos_candle_body_pct', 'volume_ratio', 'distance_pct'];
    const summaryLabels = ['Zone Size %', 'Impulse Strength %', 'Impulse Speed', 'BOS Candle Body %', 'Volume Ratio', 'Distance %'];

    function mean(arr) {{ return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }}
    function median(arr) {{
        if (arr.length === 0) return 0;
        const s = [...arr].sort((a, b) => a - b);
        const m = Math.floor(s.length / 2);
        return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    }}

    const thA = mfeThreshold > 0 ? 'High MFE' : 'Reversal';
    const thB = mfeThreshold > 0 ? 'Low MFE' : 'No Reversal';
    let tableHtml = '<table class="stats-table"><thead><tr><th>Feature</th><th>' + thA + ' Mean</th><th>' + thA + ' Median</th><th>' + thB + ' Mean</th><th>' + thB + ' Median</th><th>Δ Mean</th></tr></thead><tbody>';
    for (let i = 0; i < summaryFeatures.length; i++) {{
        const fn = summaryFeatures[i];
        const rVals = producedReversal.map(f => f[fn]).filter(v => v !== null);
        const dVals = noReversal.map(f => f[fn]).filter(v => v !== null);
        const rMean = mean(rVals), dMean = mean(dVals);
        const rMed = median(rVals), dMed = median(dVals);
        const delta = rMean - dMean;
        const deltaColor = delta > 0 ? '#26a69a' : delta < 0 ? '#ef5350' : '#787b86';
        tableHtml += '<tr><td>' + summaryLabels[i] + '</td><td>' + rMean.toFixed(3) + '</td><td>' + rMed.toFixed(3) + '</td><td>' + dMean.toFixed(3) + '</td><td>' + dMed.toFixed(3) + '</td><td style="color:' + deltaColor + '">' + (delta >= 0 ? '+' : '') + delta.toFixed(3) + '</td></tr>';
    }}
    tableHtml += '<tr><td colspan="6" style="color:#787b86;">' + labelA + ' | ' + labelB + '</td></tr>';
    tableHtml += '</tbody></table>';
    document.getElementById('ob-summary-table').innerHTML = tableHtml;
}}

renderOBAnalysis(0);

document.getElementById('ob-mfe-slider').addEventListener('input', function() {{
    document.getElementById('ob-mfe-val').textContent = parseFloat(this.value).toFixed(1) + '%';
    renderOBAnalysis(parseFloat(this.value));
}});

// ==================== FVG MAGNET CHARTS ====================

const fvgProximity = {fvg_proximity_json};
const fvgFeatures = {fvg_features_json};

function renderFVGMagnet() {{
    // --- 1. Above vs Below Race ---
    const aw = fvgProximity.above_wins || 0;
    const bw = fvgProximity.below_wins || 0;
    const cw = fvgProximity.closer_side_wins || 0;
    const fw = fvgProximity.farther_side_wins || 0;
    const total = fvgProximity.total_races || 0;

    Plotly.newPlot('fvgm-proximity', [
        {{x: ['Above wins', 'Below wins', 'Closer side wins', 'Farther side wins'],
          y: [aw, bw, cw, fw], type: 'bar',
          marker: {{color: [BULL, BEAR, '#5c6bc0', '#ff8a65']}},
          text: [aw, bw, cw, fw].map(v => total > 0 ? (v/total*100).toFixed(1) + '%' : '0%'),
          textposition: 'outside'}},
    ], {{
        ...layout_base,
        title: {{text: 'Above vs Below Race (n=' + total + ' races)', font: {{size: 14, color: fontColor}}}},
        xaxis: {{...layout_base.xaxis}},
        yaxis: {{...layout_base.yaxis, title: 'Count'}},
        height: 400,
    }}, {{responsive: true}});

    // --- 1b. Distance Ratio Histogram ---
    // For each bar, we find the closest FVG above and the closest FVG below.
    // We then see which FVG price reached first (the "winner").
    // Ratio = winner_distance / loser_distance.
    //   ratio < 1  → price went to the nearer FVG  (closer side won)
    //   ratio = 1  → both FVGs were equidistant
    //   ratio > 1  → price went to the farther FVG (farther side won)
    // Most ratios cluster well below 1, meaning price overwhelmingly
    // moves toward whichever FVG is closer — the "magnet" effect.
    const ratios = fvgProximity.distance_ratios || [];
    const clippedRatios = ratios.filter(r => r >= 0 && r <= 3);
    Plotly.newPlot('fvgm-dist-ratio', [
        {{x: clippedRatios, type: 'histogram', xbins: {{start: 0, end: 3, size: 0.15}},
          marker: {{color: 'rgba(92,107,192,0.7)'}},
          name: 'n=' + ratios.length + ' (shown=' + clippedRatios.length + ')'}},
    ], {{
        ...layout_base,
        title: {{text: 'Winner Distance / Loser Distance', font: {{size: 14, color: fontColor}}}},
        xaxis: {{...layout_base.xaxis, type: 'linear', title: 'Ratio (< 1 = closer side won)', range: [0, 3]}},
        yaxis: {{...layout_base.yaxis, title: 'Count'}},
        shapes: [{{type: 'line', x0: 1, x1: 1, y0: 0, y1: 1, yref: 'paper',
                   line: {{color: 'rgba(255,255,255,0.6)', width: 2, dash: 'dash'}}}}],
        annotations: [{{x: 1, y: 1, yref: 'paper', text: 'ratio=1', showarrow: false,
                        font: {{color: fontColor, size: 10}}, yanchor: 'bottom'}}],
        height: 400,
    }}, {{responsive: true}});

    // --- 3. Speed Scatter ---
    const mitWithSpeed = fvgFeatures.filter(f => f.was_mitigated && f.candles_to_mitigation !== null);
    const bullMit = mitWithSpeed.filter(f => f.direction === 1);
    const bearMit = mitWithSpeed.filter(f => f.direction === -1);

    Plotly.newPlot('fvgm-speed-scatter', [
        {{x: bullMit.map(f => +f.distance_pct), y: bullMit.map(f => +f.candles_to_mitigation),
          mode: 'markers', name: 'Bullish', marker: {{color: BULL, size: 5, opacity: 0.6}}, type: 'scatter'}},
        {{x: bearMit.map(f => +f.distance_pct), y: bearMit.map(f => +f.candles_to_mitigation),
          mode: 'markers', name: 'Bearish', marker: {{color: BEAR, size: 5, opacity: 0.6}}, type: 'scatter'}},
    ], {{
        ...layout_base,
        title: {{text: 'FVG Fill Speed — Distance vs Candles to Mitigation', font: {{size: 14, color: fontColor}}}},
        xaxis: {{...layout_base.xaxis, title: 'Distance from price at creation %', type: 'linear'}},
        yaxis: {{...layout_base.yaxis, title: 'Candles to mitigation', type: 'linear'}},
        height: 400,
    }}, {{responsive: true}});

    function medianArr(arr) {{
        if (arr.length === 0) return 0;
        const s = [...arr].sort((a, b) => a - b);
        const m = Math.floor(s.length / 2);
        return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    }}

    // --- 5. Summary Table ---
    const summFeatures = ['size_pct', 'distance_pct', 'num_competing', 'proximity_rank', 'trend_aligned'];
    const summLabels = ['Size %', 'Distance %', '# Competing', 'Proximity Rank', 'Trend Aligned'];

    function meanF(arr) {{ return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }}

    // Respected vs disrespected split
    const respected = fvgFeatures.filter(f => f.was_respected === true);
    const disrespected = fvgFeatures.filter(f => f.was_respected === false);

    let tbl = '<table class="stats-table"><thead><tr><th>Feature</th><th>Respected Mean</th><th>Respected Median</th><th>Disrespected Mean</th><th>Disrespected Median</th><th>Δ Mean</th></tr></thead><tbody>';
    for (let i = 0; i < summFeatures.length; i++) {{
        const fn = summFeatures[i];
        const rVals = respected.map(f => f[fn]).filter(v => v !== null && v !== undefined);
        const dVals = disrespected.map(f => f[fn]).filter(v => v !== null && v !== undefined);
        const rMean = meanF(rVals), dMean = meanF(dVals);
        const rMed = medianArr(rVals), dMed = medianArr(dVals);
        const delta = rMean - dMean;
        const dColor = delta > 0 ? '#26a69a' : delta < 0 ? '#ef5350' : '#787b86';
        tbl += '<tr><td>' + summLabels[i] + '</td><td>' + rMean.toFixed(2) + '</td><td>' + rMed.toFixed(2) + '</td><td>' + dMean.toFixed(2) + '</td><td>' + dMed.toFixed(2) + '</td><td style="color:' + dColor + '">' + (delta >= 0 ? '+' : '') + delta.toFixed(2) + '</td></tr>';
    }}
    tbl += '<tr><td colspan="6" style="color:#787b86;">Respected: n=' + respected.length + ' | Disrespected: n=' + disrespected.length + '</td></tr>';
    tbl += '</tbody></table>';
    document.getElementById('fvgm-summary-table').innerHTML = tbl;
}}

renderFVGMagnet();

// ==================== OB MAGNET CHARTS ====================

const obProximity = {ob_proximity_json};
const obMagnetFeatures = {ob_magnet_features_json};

function renderOBMagnet() {{
    // --- 1. Above vs Below Race ---
    const aw = obProximity.above_wins || 0;
    const bw = obProximity.below_wins || 0;
    const cw = obProximity.closer_side_wins || 0;
    const fw = obProximity.farther_side_wins || 0;
    const total = obProximity.total_races || 0;

    Plotly.newPlot('obm-proximity', [
        {{x: ['Above wins', 'Below wins', 'Closer side wins', 'Farther side wins'],
          y: [aw, bw, cw, fw], type: 'bar',
          marker: {{color: [BULL, BEAR, '#5c6bc0', '#ff8a65']}},
          text: [aw, bw, cw, fw].map(v => total > 0 ? (v/total*100).toFixed(1) + '%' : '0%'),
          textposition: 'outside'}},
    ], {{
        ...layout_base,
        title: {{text: 'Above vs Below Race (n=' + total + ' races)', font: {{size: 14, color: fontColor}}}},
        xaxis: {{...layout_base.xaxis}},
        yaxis: {{...layout_base.yaxis, title: 'Count'}},
        height: 400,
    }}, {{responsive: true}});

    // --- 1b. Distance Ratio Histogram ---
    const ratios = obProximity.distance_ratios || [];
    const clippedRatios = ratios.filter(r => r >= 0 && r <= 3);
    Plotly.newPlot('obm-dist-ratio', [
        {{x: clippedRatios, type: 'histogram', xbins: {{start: 0, end: 3, size: 0.15}},
          marker: {{color: 'rgba(155,89,182,0.7)'}},
          name: 'n=' + ratios.length + ' (shown=' + clippedRatios.length + ')'}},
    ], {{
        ...layout_base,
        title: {{text: 'Winner Distance / Loser Distance', font: {{size: 14, color: fontColor}}}},
        xaxis: {{...layout_base.xaxis, type: 'linear', title: 'Ratio (< 1 = closer side won)', range: [0, 3]}},
        yaxis: {{...layout_base.yaxis, title: 'Count'}},
        shapes: [{{type: 'line', x0: 1, x1: 1, y0: 0, y1: 1, yref: 'paper',
                   line: {{color: 'rgba(255,255,255,0.6)', width: 2, dash: 'dash'}}}}],
        annotations: [{{x: 1, y: 1, yref: 'paper', text: 'ratio=1', showarrow: false,
                        font: {{color: fontColor, size: 10}}, yanchor: 'bottom'}}],
        height: 400,
    }}, {{responsive: true}});

    // --- 3. Speed Scatter ---
    const mitWithSpeed = obMagnetFeatures.filter(f => f.was_mitigated && f.candles_to_mitigation !== null);
    const bullMit = mitWithSpeed.filter(f => f.direction === 1);
    const bearMit = mitWithSpeed.filter(f => f.direction === -1);

    Plotly.newPlot('obm-speed-scatter', [
        {{x: bullMit.map(f => +f.distance_pct), y: bullMit.map(f => +f.candles_to_mitigation),
          mode: 'markers', name: 'Bullish', marker: {{color: BULL, size: 5, opacity: 0.6}}, type: 'scatter'}},
        {{x: bearMit.map(f => +f.distance_pct), y: bearMit.map(f => +f.candles_to_mitigation),
          mode: 'markers', name: 'Bearish', marker: {{color: BEAR, size: 5, opacity: 0.6}}, type: 'scatter'}},
    ], {{
        ...layout_base,
        title: {{text: 'OB Fill Speed — Distance vs Candles to Mitigation', font: {{size: 14, color: fontColor}}}},
        xaxis: {{...layout_base.xaxis, title: 'Distance from price at creation %', type: 'linear'}},
        yaxis: {{...layout_base.yaxis, title: 'Candles to mitigation', type: 'linear'}},
        height: 400,
    }}, {{responsive: true}});

    function medianArr(arr) {{
        if (arr.length === 0) return 0;
        const s = [...arr].sort((a, b) => a - b);
        const m = Math.floor(s.length / 2);
        return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    }}

    // --- 5. Summary Table ---
    const summFeatures = ['size_pct', 'distance_pct', 'num_competing', 'proximity_rank', 'trend_aligned'];
    const summLabels = ['Size %', 'Distance %', '# Competing', 'Proximity Rank', 'Trend Aligned'];

    function meanF(arr) {{ return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }}

    const respected = obMagnetFeatures.filter(f => f.was_respected === true);
    const disrespected = obMagnetFeatures.filter(f => f.was_respected === false);

    let tbl = '<table class="stats-table"><thead><tr><th>Feature</th><th>Respected Mean</th><th>Respected Median</th><th>Disrespected Mean</th><th>Disrespected Median</th><th>Δ Mean</th></tr></thead><tbody>';
    for (let i = 0; i < summFeatures.length; i++) {{
        const fn = summFeatures[i];
        const rVals = respected.map(f => f[fn]).filter(v => v !== null && v !== undefined);
        const dVals = disrespected.map(f => f[fn]).filter(v => v !== null && v !== undefined);
        const rMean = meanF(rVals), dMean = meanF(dVals);
        const rMed = medianArr(rVals), dMed = medianArr(dVals);
        const delta = rMean - dMean;
        const dColor = delta > 0 ? '#26a69a' : delta < 0 ? '#ef5350' : '#787b86';
        tbl += '<tr><td>' + summLabels[i] + '</td><td>' + rMean.toFixed(2) + '</td><td>' + rMed.toFixed(2) + '</td><td>' + dMean.toFixed(2) + '</td><td>' + dMed.toFixed(2) + '</td><td style="color:' + dColor + '">' + (delta >= 0 ? '+' : '') + delta.toFixed(2) + '</td></tr>';
    }}
    tbl += '<tr><td colspan="6" style="color:#787b86;">Respected: n=' + respected.length + ' | Disrespected: n=' + disrespected.length + '</td></tr>';
    tbl += '</tbody></table>';
    document.getElementById('obm-summary-table').innerHTML = tbl;
}}

renderOBMagnet();

</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SMC Dashboard — Unified Report')
    parser.add_argument('--symbol', default='TSLA', help='Symbol (default: TSLA)')
    parser.add_argument('--timeframe', default='15min', help='Timeframe (default: 15min)')
    parser.add_argument('--no-open', action='store_true', help='Do not open browser')
    parser.add_argument('--no-serve', action='store_true', help='Static files only, no local server')
    args = parser.parse_args()

    print(f"Loading {args.symbol} {args.timeframe} data...")
    df, fvg, ob, inflexions, bos = load_and_compute(args.symbol, args.timeframe)
    print(f"  {len(df)} candles loaded")

    # --- Zones ---
    print("Extracting zones...")
    zones_df = extract_zones(df, fvg, ob, inflexions)
    print(f"  {len(zones_df)} zones extracted")

    if zones_df.empty:
        zones_df = pd.DataFrame()
        stats_df = pd.DataFrame()
        survival = {}
        race_rates, total_races = {}, 0
        dist_speed = {}
    else:
        stats_df = compute_lifecycle_stats(zones_df)
        survival = compute_survival_curves(zones_df)
        race_rates, total_races = compute_race_analysis(zones_df, len(df))
        dist_speed = compute_distance_vs_speed(zones_df)

    # --- Reversal quality ---
    print("Computing reversal quality...")
    reversal_data = compute_reversal_quality(df, zones_df)

    # --- OB Analysis ---
    print("Computing OB features...")
    ob_features = compute_ob_features(df, ob, zones_df, reversal_data=reversal_data)
    print(f"  {len(ob_features)} OBs with features extracted")

    # --- FVG Magnet ---
    print("Computing FVG proximity analysis...")
    fvg_records = _extract_zone_records(fvg, 'FVG')
    fvg_proximity = compute_zone_proximity_analysis(df, fvg_records, sample_step=20)
    fvg_features = compute_zone_feature_matrix(df, fvg_records, bos)
    print(f"  {len(fvg_features)} FVGs with features extracted")

    # --- OB Magnet ---
    print("Computing OB proximity analysis...")
    ob_records = _extract_zone_records(ob, 'OB')
    ob_proximity = compute_zone_proximity_analysis(df, ob_records, sample_step=20)
    ob_magnet_features = compute_zone_feature_matrix(df, ob_records, bos)
    print(f"  {len(ob_magnet_features)} OBs with magnet features extracted")

    # --- Build HTML ---
    html = build_html(args.symbol, args.timeframe, stats_df, survival,
                      race_rates, total_races, dist_speed, zones_df, df=df,
                      reversal_data=reversal_data, ob_features=ob_features,
                      fvg_proximity=fvg_proximity, fvg_features=fvg_features,
                      ob_proximity=ob_proximity, ob_magnet_features=ob_magnet_features)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'charts')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'smc_dashboard.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"Dashboard saved to {out_path}")

    # --- Also generate OB MFE Highlight chart ---
    print("Generating OB MFE highlight chart...")
    from ob_mfe_viewer import build_ob_zones_with_mfe, HTML_TEMPLATE as OB_HTML_TEMPLATE
    from visualization.core import generate_candle_data, NumpyEncoder

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('time')

    ob_data = build_ob_zones_with_mfe(df, ob, zones_df, reversal_data)
    candle_data = generate_candle_data(df)
    ob_html = OB_HTML_TEMPLATE.format(
        symbol=args.symbol,
        timeframe=args.timeframe,
        candle_json=json.dumps(candle_data, cls=NumpyEncoder),
        ob_json=json.dumps(ob_data, cls=NumpyEncoder),
    )
    ob_path = os.path.join(out_dir, 'ob_mfe_highlight.html')
    with open(ob_path, 'w') as f:
        f.write(ob_html)
    print(f"OB MFE chart saved to {ob_path}")

    if args.no_serve:
        if not args.no_open:
            webbrowser.open(f'file://{os.path.abspath(out_path)}')
        return

    # --- Start local server ---
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    serve_dashboard(results_dir, os.path.abspath(out_path), not args.no_open)


def _find_free_port(start=8050, end=8100):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    return start


def serve_dashboard(results_dir, dashboard_path, open_browser=True):
    """Start a local HTTP server serving from results_dir with an API endpoint."""
    port = _find_free_port()

    class DashboardHandler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=results_dir, **kw)

        def do_GET(self):
            parsed = urlparse(self.path)

            # Serve dashboard at root
            if parsed.path == '/':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                with open(dashboard_path, 'rb') as f:
                    self.wfile.write(f.read())
                return

            # API endpoint for price distribution generation
            if parsed.path == '/api/generate-price-dist':
                self._handle_price_dist(parsed)
                return

            # Serve static files from results_dir
            super().do_GET()

        def _handle_price_dist(self, parsed):
            try:
                params = parse_qs(parsed.query)
                symbol = params.get('symbol', ['TSLA'])[0]
                timeframe = params.get('timeframe', ['15min'])[0]
                size = int(params.get('size', ['1250'])[0])
                bins = int(params.get('bins', ['80'])[0])
                swing_length = int(params.get('swing_length', ['50'])[0])
                step = int(params.get('step', ['1'])[0])

                result = _run_price_distribution(symbol, timeframe, size, bins, swing_length, step, results_dir)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())

        def log_message(self, format, *a):
            # Suppress request logs
            pass

    server = HTTPServer(('localhost', port), DashboardHandler)
    url = f'http://localhost:{port}'
    print(f"\nServing dashboard at {url}")
    print("Press Ctrl+C to stop\n")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


def _run_price_distribution(symbol, timeframe, size, bins, swing_length, step, results_dir):
    """Run price distribution generation and return {'url': ...}."""
    from price_distribution_indicators import (
        generate_price_levels,
        build_chart_frame,
        collect_histogram_zones,
        build_html as build_price_dist_html,
    )
    from visualization.core import NumpyEncoder

    loader = DataLoader(symbol=symbol)
    ohlc = loader.get_data(timeframe)
    if ohlc is None or ohlc.empty:
        return {'error': f'No data for {symbol} {timeframe}'}

    if 'time' in ohlc.columns and not isinstance(ohlc.index, pd.DatetimeIndex):
        ohlc = ohlc.set_index('time')

    # Slice to requested size (from end)
    if size < len(ohlc):
        ohlc = ohlc.iloc[-size:].copy()

    total = len(ohlc)
    frame_indices = list(range(0, total, step))
    if frame_indices[-1] != total - 1:
        frame_indices.append(total - 1)

    print(f"[Price Dist] Building {len(frame_indices)} frames for {symbol} {timeframe} ({total} candles, step={step})...")

    all_frames = []
    for count, i in enumerate(frame_indices):
        ohlc_slice = ohlc.iloc[:i + 1]
        chart_data, raw_indicators = build_chart_frame(ohlc_slice)
        hist_zones = collect_histogram_zones(
            raw_indicators['fvg'], raw_indicators['ob'], raw_indicators['inflexions']
        )

        price_min = float(ohlc_slice['low'].min())
        price_max = float(ohlc_slice['high'].max())
        margin = (price_max - price_min) * 0.02
        bin_edges = np.linspace(price_min - margin, price_max + margin, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        price_levels = generate_price_levels(ohlc_slice)
        price_counts, _ = np.histogram(price_levels, bins=bin_edges)

        hist_zones['binCenters'] = bin_centers.tolist()
        hist_zones['priceCounts'] = price_counts.tolist()
        hist_zones['currentPrice'] = float(ohlc.iloc[i]['close'])
        hist_zones['currentHigh'] = float(ohlc.iloc[i]['high'])
        hist_zones['currentLow'] = float(ohlc.iloc[i]['low'])
        hist_zones['priceRange'] = [price_min - margin, price_max + margin]

        all_frames.append({'chart': chart_data, 'histogram': hist_zones})

        if (count + 1) % 50 == 0:
            print(f"  Frame {count + 1}/{len(frame_indices)}")

    all_frames_json = json.dumps(all_frames, cls=NumpyEncoder)
    html = build_price_dist_html(symbol, timeframe, all_frames_json)

    out_dir = os.path.join(results_dir, 'charts')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'price_distribution_indicators.html')
    with open(out_path, 'w') as f:
        f.write(html)

    size_mb = len(html) / (1024 * 1024)
    print(f"[Price Dist] Saved ({size_mb:.1f} MB)")
    return {'url': '/charts/price_distribution_indicators.html'}


if __name__ == '__main__':
    main()
