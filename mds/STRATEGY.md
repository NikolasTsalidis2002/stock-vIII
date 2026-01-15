# Trading Strategy Specification

## Timeframes and Event Flow

### 1. One-Hour Timeframe (High-Level Trigger)
- Detect a **Liquidity Sweep** (as defined in Smart Money Concepts).  
- Once a liquidity sweep is confirmed, proceed to the 5-minute timeframe.
- When a 1H candle closes above the swept high level (or below a 
  swept low level), that liquidity sweep becomes invalidated and the strategy 
  should reset to look for a new sweep.
- If a new liquidity sweep occurs on the one-hour timeframe at any point—even after moving to lower timeframes—the strategy fully resets and aligns with the most recent liquidity event.

### 2. Five-Minute Timeframe (Trend Reversal Confirmation)
Perform two sequential checks in this order:

1. **Break of Structure (BOS) / Inverse FVG**
   - Identify a **BOS** in the **opposite direction** of the prior 1-hour trend.  
     - Example: If the 1-hour trend was bullish (uptrend), the 5-minute BOS should be bearish (downward break).  
   - Alternatively, an **IFVG (Inverse Fair Value Gap)** — a fair value gap that gets disrespected — also satisfies this step.  
   - We refer to this reversal BOS/IFVG as **Event B**.

2. **Validation via FVG or Demand Zone**
   - After Event B, monitor price retracement.  
   - Confirm that price **respects** either:
     - a **Fair Value Gap (FVG)**, or  
     - a **Demand Zone**.  
   - This indicates that the market is favoring the new opposite direction.

➡️ Once both steps are complete, proceed to the 1-minute timeframe.

### 3. One-Minute Timeframe (Final Confirmation)
- Detect either:
  - a **Break of Structure (BOS)** in the new direction, or  
  - an **Inverse FVG (IFVG)**.  
- This serves as the **final confirmation signal** to enter the trade.

---

## Trade Execution
- **Entry Condition:**  
  After the 1-minute confirmation (BOS or IFVG), open a trade in the **opposite direction of the original 1-hour trend**.  

- **Example Workflow:**
  1. **1H:** Market trending upward (**Trend A**) → Liquidity Sweep detected.  
  2. **5M:** Price breaks downward (**BOS B**) → Then respects an FVG or Demand Zone.  
  3. **1M:** A final BOS or IFVG confirms bearish bias → Enter **short trade**.  

---

## Key Definitions (for coding clarity)
- **Liquidity Sweep:** Price takes out a previous high/low and then reverses.  
- **Break of Structure (BOS):** A candle close beyond a prior swing high/low.  
- **Fair Value Gap (FVG):** A 3-candle pattern where the middle candle’s wick does not overlap the previous or next candle, creating an inefficiency.  
- **Inverse FVG (IFVG):** A fair value gap that fails (price trades through it in the opposite direction).  
- **Demand Zone:** A previously defined area where aggressive buying occurred, often marked by consolidation before upward movement.  
