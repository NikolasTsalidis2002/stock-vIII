# Magnet Chase Strategy

## Overview

Trade toward the closest unmitigated FVG or OB. No liquidity sweep required — pure zone-proximity driven.

**Core insight**: Dashboard analysis shows price consistently gravitates to the nearest active zone. The closer side wins the race in the majority of samples, with distance ratios clustering well below 1.0. Proximity rank 1 zones fill fastest and most reliably.

---

## 1. Data-Driven Foundation

These rules come directly from the SMC Dashboard analysis on TSLA 15min data:

| Dashboard Tab | Finding | Strategy Implication |
|---------------|---------|---------------------|
| FVG Magnet — Race | Closer side wins majority of races | Trade toward whichever side has the nearer zone |
| FVG Magnet — Distance Ratio | Ratios cluster < 1 (winner is the closer FVG) | Proximity rank 1 is the best target |
| OB Magnet — Race | Same pattern: closer OB gets mitigated first | OBs are valid chase targets too, not just FVGs |
| FVG/OB Magnet — Speed Scatter | Distance < 1% fills in fewest candles | Prefer zones within 1% of current price |
| OB Analysis — Respect Rate | OBs with higher impulse strength hold better | When choosing between equidistant zones, prefer OBs with strong impulse |

---

## 2. Timeframes

| Role | Timeframe | Purpose |
|------|-----------|---------|
| Zone detection | 15min | Compute all active FVGs and OBs, rank by proximity |
| Entry confirmation | 5min | Wait for BOS or IFVG confirming momentum toward target |

---

## 3. Entry Rules

### Step-by-step:

1. **Compute active zones on 15min**
   - Calculate all FVGs: `smc.fvg(df, join_consecutive=True)`
   - Calculate all OBs: `smc_custom.ob(df, bos, inflexions)`
   - Extract zone records using `_extract_zone_records()` for both FVG and OB
   - Filter to **active zones only**: `MitigatedIndex == 0` (never touched)

2. **Rank zones by proximity**
   - For each active zone, compute distance from current price to zone midpoint: `distance = abs(midpoint - close) / close * 100`
   - Separate into **above** (midpoint > price) and **below** (midpoint < price)
   - On each side, rank by distance (rank 1 = closest)

3. **Select target zone**
   - Target = the **proximity rank 1 zone on the closer side** (smallest absolute distance)
   - If rank 1 above and rank 1 below are roughly equal distance (within 20%), both are valid — pick the one aligned with the most recent BOS direction on 15min

4. **Determine trade direction**
   - Target zone is **above** price → trade direction is **LONG** (price moves up toward zone)
   - Target zone is **below** price → trade direction is **SHORT** (price moves down toward zone)

5. **Wait for 5min confirmation**
   - Drop to 5min timeframe
   - Wait for a **BOS** or **IFVG** in the trade direction:
     - LONG: bullish BOS (+1) or bearish IFVG (bearish FVG that gets disrespected)
     - SHORT: bearish BOS (-1) or bullish IFVG (bullish FVG that gets disrespected)
   - **Entry price** = close of the 5min candle that triggered the BOS/IFVG

6. **Verify R:R before entering**
   - Compute TP distance = `abs(entry_price - tp_price)`
   - Compute SL distance = `abs(entry_price - sl_price)`
   - If `TP distance / SL distance < 1.5` → **skip this trade** (R:R too low)

---

## 4. Exit Rules

### Take Profit (TP)
TP = the **near edge** of the target zone (first boundary price touches):

| Target Zone Type | Target Direction | TP Level |
|-----------------|-----------------|----------|
| Bullish FVG | LONG (zone above) | Zone Bottom |
| Bearish FVG | SHORT (zone below) | Zone Top |
| Bullish OB | LONG (zone above) | Zone Bottom |
| Bearish OB | SHORT (zone below) | Zone Top |

*Rationale: We're trading toward the zone. TP fires when price first touches it — we don't need to wait for a full fill or bounce.*

### Stop Loss (SL)
SL = the **last swing high/low on 5min** behind the entry direction:

- LONG: SL = most recent 5min swing low before entry
- SHORT: SL = most recent 5min swing high before entry

If no clear swing point exists, use a fallback: `SL = entry ± (TP_distance / 1.5)` to enforce minimum 1.5:1 R:R.

### Trade Resolution

| Condition | Result |
|-----------|--------|
| TP hit first | WIN |
| SL hit first | LOSS |
| Both TP and SL on same candle | LOSS (worst-case assumption) |
| 4-hour timeout | Exit at current close price |

---

## 5. Filters

Apply these filters to improve win rate. A trade must pass **all required filters**.

| Filter | Required? | Rule |
|--------|-----------|------|
| Max distance | Yes | Target zone midpoint must be within `max_distance_pct` (default 1.0%) of current price |
| Min distance | Optional | Target zone midpoint must be at least `min_distance_pct` (default 0.0%) from current price. Filters out zones too close to enter. |
| Proximity rank | Yes | Target must be rank 1 on its side (closest zone) |
| Min R:R | Yes | Risk:Reward ≥ 1.5:1 |
| Trend alignment | Optional | Prefer zones where direction matches last 15min BOS direction |
| Zone type preference | Optional | When FVG and OB are equidistant, prefer OB (higher respect rate from dashboard data) |

---

## 6. Invalidation

A setup is abandoned if any of these occur **before entry**:

| Condition | Action |
|-----------|--------|
| Target zone gets mitigated (price fills it) | Find next closest zone or abandon if none within distance filter |
| 15min BOS in opposite direction of trade | Abandon — structure shifted against the trade |
| A closer zone appears on the opposite side | Re-evaluate — the magnet has shifted |
| No 5min BOS/IFVG within 2 hours of target identification | Abandon — momentum not confirming |

After entry, only TP/SL/timeout apply (section 4).

---

## 7. Risk Management

| Parameter | Value |
|-----------|-------|
| Risk per trade | 2% of account |
| Position sizing | `risk_amount / abs(entry - SL)` |
| Max concurrent positions | 2 |
| Min R:R to enter | 1.5:1 |
| Timeout | 4 hours |

---

## 8. Configuration Parameters

Add to `config/strategy_config.json` under `"magnet_chase"`:

```json
{
  "magnet_chase": {
    "timeframe": "15min",
    "confirmation_timeframe": "5min",
    "max_distance_pct": 1.0,
    "min_distance_pct": 0.0,
    "min_rr_ratio": 1.5,
    "timeout_hours": 4,
    "max_concurrent": 2,
    "risk_per_trade_pct": 2.0,
    "prefer_ob_over_fvg": true,
    "require_trend_alignment": false,
    "confirmation_timeout_hours": 2,
    "close_break": true
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeframe` | str | "15min" | Zone detection timeframe |
| `confirmation_timeframe` | str | "5min" | BOS/IFVG confirmation timeframe |
| `max_distance_pct` | float | 1.0 | Max distance from price to target zone midpoint (%) |
| `min_distance_pct` | float | 0.0 | Min distance from price to target zone midpoint (%). 0 = disabled. |
| `min_rr_ratio` | float | 1.5 | Minimum risk:reward ratio to take a trade |
| `timeout_hours` | int | 4 | Close trade if neither TP nor SL hit |
| `max_concurrent` | int | 2 | Max simultaneous open positions |
| `risk_per_trade_pct` | float | 2.0 | Account risk per trade (%) |
| `prefer_ob_over_fvg` | bool | true | When equidistant, prefer OB targets |
| `require_trend_alignment` | bool | false | Only trade zones aligned with 15min BOS |
| `confirmation_timeout_hours` | int | 2 | Abandon if no 5min confirmation within this window |
| `close_break` | bool | true | Require close/open break for BOS (conservative) |

---

## 9. Example Trade — SHORT

### Setup (15min)

| Time | Event | Detail |
|------|-------|--------|
| 10:00 | Zone scan | Active bearish FVG at $248.50–$249.20 (midpoint $248.85), 0.6% below price ($250.40). Active bullish OB at $252.00–$253.10 (midpoint $252.55), 0.9% above price. |
| 10:00 | Target selected | Bearish FVG below is closer (0.6% vs 0.9%) → **target = bearish FVG below** → direction = **SHORT** |
| 10:00 | TP/SL computed | TP = $249.20 (zone Top, near edge). Last 5min swing high = $251.10 → SL = $251.10. R:R = ($250.40 - $249.20) / ($251.10 - $250.40) = 1.20 / 0.70 = 1.71 ✓ |

### Confirmation (5min)

| Time | Event | Detail |
|------|-------|--------|
| 10:15 | 5min BOS | Bearish BOS (-1) on 5min — candle closes at $250.10 |
| 10:15 | **ENTRY** | SHORT at $250.10 |

### Execution

| Time | Event | Price |
|------|-------|-------|
| 10:15 | Entry | $250.10 |
| 10:45 | Price touches FVG top | $249.20 |
| 10:45 | **TP HIT → WIN** | +$0.90 per share |

### Final Trade Summary

| Parameter | Value |
|-----------|-------|
| Direction | SHORT |
| Entry | $250.10 |
| TP | $249.20 |
| SL | $251.10 |
| R:R | 1.71:1 |
| Result | WIN (+0.36%) |

---

## 10. Implementation Notes

**Maximize reuse of existing code:**

| Existing Component | Reuse For |
|-------------------|-----------|
| `TimeframeManager` | Multi-TF data loading and indicator pre-computation |
| `ConfirmationDetector` | 5min BOS/IFVG detection (already does exactly what we need) |
| `EventBDetector` | BOS/IFVG detection logic (reusable for directional confirmation) |
| `_extract_zone_records()` from `smc_dashboard.py` | Extract to `src/strategy/zone_proximity.py` for runtime zone ranking |
| `compute_zone_proximity_analysis()` from `smc_dashboard.py` | Same — extract for runtime use |
| `models.py` | Extend `StrategyState` enum, reuse `TrackedFVG` pattern for zone tracking |
| `ExitTargetFinder` | Adapt for zone-edge TP calculation |
| `config/strategy_config.json` | Add `magnet_chase` section alongside existing configs |

**New code to write:**
- `src/strategy/zone_proximity.py` — extracted from dashboard, provides `get_closest_zones(df, fvg, ob)` returning ranked active zones
- `src/strategy/magnet_chase_engine.py` — thin orchestrator: calls zone_proximity → ConfirmationDetector → computes TP/SL → emits signals

---

## Key Definitions

- **Active zone**: An FVG or OB where `MitigatedIndex == 0` (price has never returned to it)
- **Proximity rank**: Among all active zones on the same side (above or below price), rank 1 = closest to current price
- **Near edge**: The zone boundary that price hits first when approaching from outside. For a zone above price, this is the Bottom. For a zone below price, this is the Top.
- **BOS (Break of Structure)**: A candle close beyond a prior swing high/low, confirming directional momentum
- **IFVG (Inverse Fair Value Gap)**: An FVG that gets disrespected (price closes through it in the opposite direction), acting as a momentum signal

---

*Last updated: 2026-01-31*
