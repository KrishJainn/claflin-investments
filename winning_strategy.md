# Winning Strategy Analysis

## 1. The Evolution
We ran two major evolutionary simulations. The second run, with an **Expanded Indicator Universe**, produced a significantly superior strategy.

| Feature | Strategy 1 (Standard) | Strategy 2 (Expanded) |
| :--- | :--- | :--- |
| **DNA ID** | `317f8cf7` | `577d4e8f` |
| **Fitness** | 0.475 | **0.561** |
| **Sharpe Ratio** | 4.30 | **7.95** |
| **Key logic** | Momentum + Mean Reversion | Smart Money + Structural Trends |

## 2. Strategy 2: The "Structural Alpha" Model
The new winner (`577d4e8f`) didn't just look at price speed (Momentum); it looked at **Market Structure** and **Smart Money**.

### Key Drivers (Why it buys)
1.  **Pivot Points (+0.77):**
    - It learned to buy when price bounces off key support levels (Pivots). This allows for much tighter entries than just chasing a moving average.
2.  **Negative Volume Index (NVI) (+0.74):**
    - *Theory:* "Smart money trades on low volume days; crowd trades on high volume days."
    - The bot learned to follow the NVI trend, effectively tracking institutional accumulation.
3.  **Coppock Curve (+0.42):** *[New Indicator]*
    - A long-term momentum oscillator originally designed to identify major market bottoms. It uses this to filter out noise and stay in the major trend.
4.  **Vortex Indicator (+0.32):** *[New Indicator]*
    - Separates positive and negative trend movements. Used as a confirmation signal.

### The Filters (Why it sells)
1.  **Keltner Channels (-0.73):**
    - If price pushes too far outside the Keltner Channel (volatility band), it sells/avoids buy.
2.  **T3 Average (-0.68):**
    - A very smooth moving average. Negative weight suggests it might be using it as a "mean" to revert to, or fading the trend when it deviates too far from T3.

## 3. Why it Improved (Sharpe 4.3 -> 7.95)
Strategy 1 was a **"Chaser"**—it waited for a move to happen (Momentum) then jumped on. It worked, but suffered drawdown when the trend snapped back.

Strategy 2 is a **"Hunter"**:
- It identifies **Support** (Pivots).
- It confirms with **Smart Money flow** (NVI).
- It uses **Coppock** to ensure the long-term tide is rising.

This multi-dimensional approach (Price Structure + Volume + Momentum) filters out almost all "fake outs," leading to the massive jump in Sharpe Ratio.
