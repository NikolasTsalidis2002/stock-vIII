# 3-OB Strategy: Lower-Timeframe Confirmation

## Overview

The **3-OB (Three Order Block) Strategy** identifies institutional trade setups by tracking a sequence of three Order Blocks — OB-A, OB-B, and OB-C — using a state machine that progresses through states 0→3.

**LTF Confirmation** adds a final precision step: instead of entering on a simple green/red candle close (HTF), the strategy drops to a lower timeframe (e.g., 5min inside a 15min candle) and waits for a **BOS** or **Inverse FVG (IFVG)** to confirm the entry. This produces a more precise entry price and filters out false setups.

When `confirmation_timeframe` is set to `null` in config, the strategy falls back to the original behaviour (green close for longs, red close for shorts).

---

## State Machine Recap (States 0→3)

| State | Name | What Happens | Transition |
|-------|------|--------------|------------|
| **0** | OB-A + OB-C Discovery | Price touches and leaves an Order Block (OB-A). An opposite-direction OB-C is found at least `min_dist_pct`% away. | OB-A confirmed + OB-C found → **State 1** |
| **1** | Waiting for OB-B | A new Order Block forms in the **same direction** as OB-A, after OB-A confirmation. | OB-B detected → **State 2** |
| **2** | Waiting for OB-B Retouch | Price moves away from OB-B, then returns to touch it. | OB-B retouched → **State 3** |
| **3** | Pending Confirmation | With LTF: emit `pending_confirmation` for LTF scan. Without LTF: wait for confirming candle colour. | Confirmation found → **Signal** |

### Direction Rules

| | OB-A | OB-C | OB-B | Confirming Move |
|---|---|---|---|---|
| **Long** | Bullish OB (touched & left) | Bearish OB above price (target) | New bullish OB after OB-A | LTF bullish BOS or bearish IFVG |
| **Short** | Bearish OB (touched & left) | Bullish OB below price (target) | New bearish OB after OB-A | LTF bearish BOS or bullish IFVG |

### Invalidation

- **OB-A invalidated** (price breaks through, not respected) → reset to State 0
- **OB-B invalidated** → fall back to State 1 (keep OB-A/C, find new OB-B)
- **OB-C touched** before entry → full invalidation (target reached prematurely)

---

## LTF Confirmation Step

When the engine reaches State 3, the strategy performs the following sub-steps on the lower timeframe:

### Step 1: Find Inflexion Reference

Between OB-A confirmation and OB-B retouch, find the extreme price point on the HTF:

- **Long:** highest swing high in that range (the peak before price pulled back to OB-B)
- **Short:** lowest swing low in that range (the valley before price bounced to OB-B)

Any LTF confirmation that occurs **before** this inflexion timestamp is ignored. This prevents false signals from the prior swing.

### Step 2: Define Confirmation Window

- **Start:** OB-B retouch candle timestamp
- **End:** next HTF candle close timestamp

Example with 15min HTF: if OB-B retouch happens on the 14:45 candle, the window is `14:45:00` to `15:00:00`.

### Step 3: Scan LTF for BOS or IFVG

Within the confirmation window, iterate through LTF candles (skipping any before the inflexion reference) and check:

1. **BOS (Break of Structure):** A bullish BOS (+1) for longs, bearish BOS (-1) for shorts. Entry price = BOS level.
2. **IFVG (Inverse Fair Value Gap):** A bearish FVG that gets disrespected for longs, a bullish FVG that gets disrespected for shorts. Entry price = LTF close.

The first match triggers the signal.

### Step 4: Generate Signal

- **Entry** = LTF confirmation price (more precise than HTF close)
- **Stop Loss** = OB-A bottom (long) or OB-A top (short)
- **Take Profit** = OB-C boundary:
  - Long: `OB-C_bottom`
  - Short: `OB-C_top`

---

## Flow Diagrams

### Long Setup

```
State 0                State 1           State 2         State 3
┌─────────────┐       ┌───────────┐     ┌────────────┐  ┌──────────────────┐
│ Bull OB-A    │       │ New bull   │     │ Price re-  │  │ LTF scan:        │
│ touched &    │──────▶│ OB-B forms │────▶│ touches    │─▶│  Find inflexion  │
│ left + Bear  │       │ after A    │     │ OB-B       │  │  Window: retouch │
│ OB-C found   │       │            │     │            │  │   → next HTF bar │
│ above (>2%)  │       │            │     │            │  │  BOS(+1) / IFVG? │
└─────────────┘       └───────────┘     └────────────┘  │  → LONG SIGNAL   │
                                                         └──────────────────┘
  OB-C ═══════  (target: bearish OB above)
  OB-A ▓▓▓▓▓▓  (support: bullish OB below)
  OB-B ░░░░░░  (pullback: bullish OB between)
```

### Short Setup

```
State 0                State 1           State 2         State 3
┌─────────────┐       ┌───────────┐     ┌────────────┐  ┌──────────────────┐
│ Bear OB-A    │       │ New bear   │     │ Price re-  │  │ LTF scan:        │
│ touched &    │──────▶│ OB-B forms │────▶│ touches    │─▶│  Find inflexion  │
│ left + Bull  │       │ after A    │     │ OB-B       │  │  Window: retouch │
│ OB-C found   │       │            │     │            │  │   → next HTF bar │
│ below (>2%)  │       │            │     │            │  │  BOS(-1) / IFVG? │
└─────────────┘       └───────────┘     └────────────┘  │  → SHORT SIGNAL  │
                                                         └──────────────────┘
  OB-A ▓▓▓▓▓▓  (resistance: bearish OB above)
  OB-B ░░░░░░  (pullback: bearish OB between)
  OB-C ═══════  (target: bullish OB below)
```

---

## Implementation Details

### Files

| File | Role |
|------|------|
| `src/strategy/three_ob_engine.py` | Core state machine (states 0→3). Yields per-bar snapshots with `pending_confirmation` when State 3 is reached. |
| `src/strategy/three_ob_strategy.py` | Strategy wrapper. Consumes engine snapshots, runs LTF confirmation via `ConfirmationDetector`, produces `TradeSignal` objects. |
| `src/strategy/confirmation_detector.py` | Detects BOS and IFVG on the lower timeframe at a given candle index. |
| `src/visualization/three_ob_walkthrough.py` | HTML walkthrough generator showing state progression, OB zones (colour-coded A/B/C), and signal alerts. |
| `config/strategy_config.json` | Configuration under `"three_ob"` key. |
| `scripts/analyze_strategy.py` | CLI entry point. Use `--strategy 3-ob` and `--walkthrough-3ob`. |

### Configuration (`config/strategy_config.json`)

```json
{
  "three_ob": {
    "timeframe": "15min",
    "confirmation_timeframe": "5min",
    "close_break": true,
    "min_dist_pct": 2.0,
    "enable_shorts": true
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeframe` | `"15min"` | HTF for the state machine |
| `confirmation_timeframe` | `"5min"` | LTF for BOS/IFVG confirmation. Set to `null` to disable and use green/red close instead. |
| `close_break` | `true` | Require close/open break for BOS detection (conservative) |
| `min_dist_pct` | `2.0` | Minimum percentage distance between OB-A and OB-C |
| `enable_shorts` | `true` | Allow short setups |

---

## Logging

The strategy uses Python's `logging` module. Enable with:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Engine Log Messages (per bar)

State transitions are reported in the snapshot `long_status` / `short_status` fields:

```
[State 0→1] OB-A (bull $220.50-$221.30) confirmed, OB-C found 3.2% above
[State 1→2] OB-B (bull $219.80-$220.40) found after OB-A
[State 2→3] OB-B ($219.80-$220.40) retouched — waiting for confirmation
[State 3]   Green close on OB-B → LONG signal — SL=$220.50
```

### Strategy Log Messages (LTF confirmation)

```
Pending confirmation: long @ bar 142 (2025-10-15 14:45), OB-B=[219.80, 220.40]
  Inflexion reference: bar 138 @ 2025-10-15 14:30, price=223.10
  Confirmation window: 2025-10-15 14:45 to 2025-10-15 15:00 (inflexion_after=14:30)
  Confirmation FOUND: BOS @ 2025-10-15 14:50, price=220.15
  SIGNAL: long entry=220.15, TP=225.60, SL=218.90
```

---

## Running

```bash
# Full analysis with LTF confirmation (uses config settings)
python3 scripts/analyze_strategy.py --strategy 3-ob

# Generate interactive HTML walkthrough
python3 scripts/analyze_strategy.py --strategy 3-ob --walkthrough-3ob

# Disable LTF confirmation: set confirmation_timeframe to null in config
```
