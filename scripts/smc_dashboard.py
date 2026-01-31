"""
SMC Dashboard — Unified HTML report with tabbed windows.

Tabs:
  1. Zones   — zone lifecycle stats (FVG, OB, Inflexion)
  2. Reversal Quality — MFE analysis per zone type
  3. OB Analysis — Order Block feature analysis

Usage:
    python3 scripts/smc_dashboard.py
    python3 scripts/smc_dashboard.py --timeframe 15min
    python3 scripts/smc_dashboard.py --symbol META --timeframe 1h
"""

import sys
import os
import argparse
import json
import math
import webbrowser
import numpy as np
import pandas as pd

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
    if reversal_data is not None:
        ob_zone_indices = zones_df.index[zones_df['zone_type'] == 'OB'].tolist()
        rev_ob = [r for r in reversal_data if r['zone_type'] == 'OB']
        for idx, rec in zip(ob_zone_indices, rev_ob):
            creation_idx = zones_df.loc[idx, 'creation_index']
            _rev_category[creation_idx] = rec['category']

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
        })
        # print('features --> ', features)
        # raise Exception('stop')

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
               reversal_data=None, ob_features=None):

    # Zone histogram data
    hist_data = {}
    for zt, g in zones_df.groupby('zone_type'):
        hist_data[zt] = g['candles_to_mitigation'].dropna().tolist()

    stats_html = stats_df.to_html(index=False, classes='stats-table', border=0)

    reversal_json = json.dumps(reversal_data or [])
    ob_features_json = json.dumps(ob_features or [])

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
  <a href="file:///Users/nikolastsalidis/Desktop/smart-money-concepts/results/charts/ob_mfe_highlight.html"
     target="_blank"
     style="display:inline-block; padding:8px 16px; background:#26a69a; color:#fff; border-radius:4px; text-decoration:none; font-size:13px;">
     Open OB MFE Chart ↗
  </a>
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
<p class="explanation">Mean and median of each feature, split by outcome. Delta = respected − disrespected.</p>
<div id="ob-summary-table"></div>
</div>

</div>

<script>
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

function renderOBAnalysis() {{
    // Split by reversal quality category (exclude pending)
    const producedReversal = obFeatures.filter(f => f.category === 'Respected (held)' || f.category === 'Reversal then disrespect');
    const noReversal = obFeatures.filter(f => f.category === 'Direct disrespect' || f.category === 'No reversal, then disrespect');

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
            {{x: rVals, type: 'histogram', name: 'Produced reversal (n=' + rVals.length + ')', marker: {{color: 'rgba(38,166,154,0.7)'}}, nbinsx: 30}},
            {{x: dVals, type: 'histogram', name: 'No reversal (n=' + dVals.length + ')', marker: {{color: 'rgba(239,83,80,0.7)'}}, nbinsx: 30}},
        ], {{
            ...layout_base,
            title: {{text: title, font: {{size: 14, color: fontColor}}}},
            xaxis: {{...layout_base.xaxis, title: xLabel, range: [0, xMax]}},
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
        const resolved = obFeatures.filter(f => f.category !== 'Pending');
        const bucketRates = [];
        const bucketCounts = [];
        for (let b = 0; b < edges.length - 1; b++) {{
            const lo = edges[b];
            const hi = edges[b + 1];
            const inBucket = resolved.filter(f => f[feature] >= lo && f[feature] < hi);
            const nRev = inBucket.filter(f => f.category === 'Respected (held)' || f.category === 'Reversal then disrespect').length;
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
    const featureNames = ['zone_size_pct', 'impulse_strength_pct', 'impulse_speed', 'bos_candle_body_pct', 'volume_ratio', 'distance_pct', 'produced_reversal'];
    const featureLabels = ['Zone Size %', 'Impulse Str %', 'Impulse Speed', 'BOS Body %', 'Volume Ratio', 'Distance %', 'Produced Reversal'];
    const resolved = obFeatures.filter(f => f.category !== 'Pending');
    if (resolved.length > 5) {{
        // Build matrix
        const cols = featureNames.map(fn => {{
            if (fn === 'produced_reversal') return resolved.map(f => (f.category === 'Respected (held)' || f.category === 'Reversal then disrespect') ? 1 : 0);
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

    let tableHtml = '<table class="stats-table"><thead><tr><th>Feature</th><th>Reversal Mean</th><th>Reversal Median</th><th>No Reversal Mean</th><th>No Reversal Median</th><th>Δ Mean</th></tr></thead><tbody>';
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
    tableHtml += '<tr><td colspan="6" style="color:#787b86;">Produced reversal: n=' + producedReversal.length + ' | No reversal: n=' + noReversal.length + '</td></tr>';
    tableHtml += '</tbody></table>';
    document.getElementById('ob-summary-table').innerHTML = tableHtml;
}}

renderOBAnalysis();

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

    # --- Build HTML ---
    html = build_html(args.symbol, args.timeframe, stats_df, survival,
                      race_rates, total_races, dist_speed, zones_df, df=df,
                      reversal_data=reversal_data, ob_features=ob_features)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'charts')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'smc_dashboard.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"Dashboard saved to {out_path}")

    # if not args.no_open:
    #     webbrowser.open(f'file://{os.path.abspath(out_path)}')


if __name__ == '__main__':
    main()
