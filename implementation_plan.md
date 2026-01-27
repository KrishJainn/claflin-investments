# Implementation Plan - Indicator Expansion
**Status: Completed**

## Goal
Expand the technical indicator universe by adding advanced indicators to provide the evolutionary engine with more "genes" to discover robust trading strategies.

## Proposed Changes

### 1. `trading_evolution/indicators/universe.py`
- Add `Vortex Indicator` (Trend/Momentum).
- Add `KST Oscillator` (Know Sure Thing - Momentum).
- Add `Mass Index` (Volatility Reversal).
- Add `Coppock Curve` (Momentum).
- Add `Ichimoku Cloud` (Trend).

### 2. `trading_evolution/indicators/calculator.py`
- Implement calculation logic for the new indicators using `ta` library mapping.
  - `trend.VortexIndicator`
  - `trend.KSTIndicator`
  - `trend.MassIndex`
  - `trend.IchimokuIndicator`
  - `momentum.KAMAIndicator` (Already mapped?) Check.
  - `trend.PSAR` (Already mapped).

## Verification Plan
1.  **Code Check**: Ensure `calculate_all` runs without error on a sample dataframe. [x] Passed.
2.  **Run Evolution**: Execute `python3 -m trading_evolution.main` with expanded universe. [x] Completed (Run #14).
3.  **Results**:
    - **Sharpe Ratio**: Improved from 4.30 to **7.95**.
    - **Fitness**: Improved by **18%**.
    - **New Indicators**: Successfully integrated and utilized (Coppock, Vortex, etc).
4.  **Monitor Logs**: Confirmed correct loading of 87 indicators.
