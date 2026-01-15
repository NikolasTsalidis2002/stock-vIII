# Setup Exit & Invalidation Conditions Reference

## Setup Lifecycle (4 Stages)

```
Stage 1: Liquidity Sweep (1H)    → inflexion point gets swept
Stage 2: Event B Detection (5M) → BOS or IFVG confirms momentum
Stage 3: Equilibrium Zone (5M)  → price enters premium/discount zone
Stage 4: Final Confirmation (1M) → BOS or IFVG on 1M
```

---

## Exit Conditions During Setup (Stages 1-4)

### KEEP: Sweep Disrespected
- If sweep's Respected status changes to False → **abandon setup entirely**

### KEEP: Sweep Too Close to Market Close
- If sweep occurs within 10 minutes of market close → **abandon setup**
- Not enough time to develop through all 4 stages

### KEEP: Exit Order Block Broken Before Entry
- If price breaks through exit OB (TP zone) BEFORE 1M confirmation
- **Abandon setup** - the move already happened without us

### REMOVED: 3% Price Deviation Check
- ~~Old: Abandon if price moves > 3% from sweep level~~
- **Replaced by:** Exit OB broken check (more logical)

### REMOVED: Event B Broken Check
- ~~Old: Reset Event B search if price breaks Event B level~~
- **Reason:** Breaking Event B is EXPECTED - price should retrace through it to reach equilibrium zone

### CHANGE: No Event B / No Equilibrium / No 1M Confirmation
- ~~Old: Track as PartialSetup~~
- **New: Abandon silently**, don't track as partial

### NEW: New Sweep Invalidates Current Setup
- If ANY new liquidity sweep occurs during Stages 2-4
- **Abandon current setup, start fresh with new sweep**
- Applies to sweeps in ANY direction (long or short)
- Rationale: New inflexion is more recent and relevant to current market structure

---

## Exit Conditions During Trade (Backtesting)

### TP Hit First → WIN
- For LONG: Candle high >= TP price
- For SHORT: Candle low <= TP price

### SL Hit First → LOSS
- For LONG: Candle low <= SL price
- For SHORT: Candle high >= SL price

### Both TP & SL Same Candle → LOSS
- Worst-case assumption - assume SL hit first

### 24 Hour Timeout → TIMEOUT
- Exit at current close price
- Neither TP nor SL hit within 24 hours

---

## Summary Table

| Condition | Action | Stage |
|-----------|--------|-------|
| Sweep disrespected | Abandon | 1-4 |
| Sweep within 10 min of close | Abandon | 1 |
| Exit OB broken before entry | Abandon | 4 |
| ~~3% price deviation~~ | ~~Removed~~ | - |
| ~~Event B broken~~ | ~~Removed~~ | - |
| No Event B by close | Abandon silently | 2 |
| No equilibrium entry by close | Abandon silently | 3 |
| No 1M confirmation by close | Abandon silently | 4 |
| New sweep during setup | Abandon current, start new | 2-4 |

---

*Last updated: 2025-01-15*
