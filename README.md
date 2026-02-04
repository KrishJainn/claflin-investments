# 5-Player Trading System with AI Coach

A Gemini-powered trading system where 5 independent players trade with different strategies, and an AI coach continuously optimizes each player based on their performance.

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         5-PLAYER TRADING SYSTEM                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Each player starts with 8-12 indicators and EVOLVES over time:              │
│                                                                              │
│  PLAYER 1 (Aggressive)     → Momentum + Volatility indicators                │
│  PLAYER 2 (Conservative)   → Trend + Moving Average indicators               │
│  PLAYER 3 (Balanced)       → Mix of all categories                           │
│  PLAYER 4 (VolBreakout)    → Volatility + Trend indicators                   │
│  PLAYER 5 (Momentum)       → Pure momentum indicators                        │
│                                                                              │
│  Indicators are NOT hardcoded - they evolve with each backtest!              │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                           AI COACH (Gemini)                                  │
│                                                                              │
│  After every N trading days, Gemini analyzes each player and decides:        │
│                                                                              │
│  ✓ Which indicators to ADD (from 80+ available)                              │
│  ✓ Which indicators to REMOVE (underperformers)                              │
│  ✓ How to ADJUST weights (0.1 to 1.0)                                        │
│  ✓ Entry/exit thresholds                                                     │
│  ✓ Minimum hold period                                                       │
│                                                                              │
│  The coach LEARNS from trade history and improves over time!                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Available Indicators (80+)

The AI Coach can choose from any of these indicators for each player:

| Category | Indicators |
|----------|------------|
| **Momentum** | RSI_7, RSI_14, RSI_21, STOCH_5_3, STOCH_14_3, MACD_12_26_9, CCI_14, CMO_14, WILLR_14, TSI_13_25, ROC_10, ROC_20, MOM_10, MOM_20, KST, COPPOCK, UO_7_14_28, AO_5_34 |
| **Trend** | ADX_14, ADX_20, AROON_14, AROON_25, SUPERTREND_7_3, SUPERTREND_10_2, PSAR, VORTEX_14, LINREG_SLOPE_14 |
| **Volatility** | ATR_14, ATR_20, NATR_14, NATR_20, BBANDS_20_2, KC_20_2, DONCHIAN_20, TRUERANGE, MASS_INDEX |
| **Volume** | OBV, AD, ADOSC_3_10, CMF_20, MFI_14, MFI_20, EFI_13, NVI, PVI |
| **Overlap** | EMA_10, EMA_20, EMA_50, SMA_20, SMA_50, WMA_20, DEMA_20, TEMA_20, HMA_9, VWMA_20, KAMA_20, T3_10 |
| **Other** | ZSCORE_20, ZSCORE_50, PIVOTS |

## Quick Start

1. **Set up environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Gemini API key:**
   ```bash
   echo "GEMINI_API_KEY=your_key_here" > .env
   ```

3. **Run the dashboard:**
   ```bash
   python run_dashboard.py
   # OR
   streamlit run coach_system/dashboard/app.py
   ```

## Project Structure

```
.
├── coach_system/              # Main trading system
│   ├── coaches/
│   │   └── ai_coach.py        # Gemini-powered AI coach
│   ├── llm/
│   │   ├── base.py            # LLM provider abstraction
│   │   └── gemini_provider.py # Gemini implementation
│   └── dashboard/
│       ├── app.py             # Streamlit dashboard
│       └── pages/
│           └── continuous_backtest.py  # 5-player backtest UI
│
├── trading_evolution/         # Core trading framework
│   ├── indicators/            # 80+ technical indicators
│   ├── backtest/              # Backtesting engine
│   └── player/                # Trade execution
│
├── aqtis/
│   └── knowledge/             # Knowledge base ingestion
│
├── data/                      # Market data utilities
├── knowledge_base/            # Trading knowledge docs
└── evolved_player_configs.json # Learned player configs (DYNAMIC!)
```

## How It Works

### 1. Five Independent Players
Each player starts with a unique strategy profile but **evolves independently**:

| Player | Style | Starting Focus | Can Evolve To Use |
|--------|-------|----------------|-------------------|
| Aggressive | High risk, short holds | Momentum indicators | Any indicator |
| Conservative | Low risk, longer holds | Trend indicators | Any indicator |
| Balanced | Medium risk | Mixed indicators | Any indicator |
| VolBreakout | Catches breakouts | Volatility indicators | Any indicator |
| Momentum | Rides trends | Momentum indicators | Any indicator |

### 2. AI Coach Optimization
Every N trading days (configurable), Gemini:
- Analyzes each player's recent trades (wins, losses, P&L)
- Reviews which indicators contributed to wins vs losses
- **Decides which indicators to add** from the 80+ available
- **Removes underperforming indicators**
- Adjusts weights based on performance
- Tunes entry/exit thresholds

### 3. Continuous Learning
- Configs persist in `evolved_player_configs.json`
- Each backtest run continues from where it left off
- Players can have anywhere from 5 to 20+ indicators
- The system gets smarter over time!

## Example: How a Player Evolves

**Run 1 (Initial):**
```
PLAYER_1 (Aggressive): RSI_7, STOCH_5_3, TSI_13_25, CMO_14 (4 indicators)
Win Rate: 42%, P&L: -$2,340
```

**Run 5 (After Coach Optimization):**
```
PLAYER_1 (Aggressive): RSI_7, STOCH_5_3, TSI_13_25, CMO_14, WILLR_14,
                       NATR_14, OBV, MFI_14, ADX_14, DEMA_20 (10 indicators)
Win Rate: 58%, P&L: +$4,120
```

**Run 10 (Further Evolution):**
```
PLAYER_1 (Aggressive): RSI_7, TSI_13_25, NATR_14, TRUERANGE, WILLR_14,
                       OBV, ADX_14, DEMA_20, KAMA_10, ROC_10, CCI_14 (11 indicators)
                       [Removed: STOCH_5_3, CMO_14 | Added: TRUERANGE, KAMA_10, ROC_10, CCI_14]
Win Rate: 62%, P&L: +$6,890
```

## Configuration

Player configs are stored in `evolved_player_configs.json` and update automatically:

```json
{
  "PLAYER_1": {
    "label": "Aggressive",
    "weights": {
      "RSI_7": 1.0,
      "TSI_13_25": 0.95,
      "NATR_14": 0.88,
      // ... more indicators added by coach
    },
    "entry_threshold": 0.31,
    "exit_threshold": -0.18,
    "min_hold_bars": 2
  }
}
```

## Requirements

- Python 3.11+
- Gemini API key (free tier works)
- See `requirements.txt` for dependencies

## License

MIT License - Claflin Investments
