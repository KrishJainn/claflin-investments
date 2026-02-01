"""
Comprehensive Trading Knowledge Base for the AI Coach System.

This module contains curated, actionable trading knowledge specifically
optimized for 15-minute candle intraday trading on NIFTY 50 (Indian NSE stocks).

Knowledge Sources:
- Technical analysis research and backtested strategies
- Mark Douglas: "Trading in the Zone"
- John Murphy: "Technical Analysis of Financial Markets"
- Perry Kaufman: "New Trading Systems and Methods"
- David Aronson: "Evidence-Based Technical Analysis"
- Indian market-specific research (NIFTY 50, FII/DII flows)

All thresholds, settings, and rules are specific and actionable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# SECTION 1: INDICATOR SETTINGS & COMBINATIONS (15-MIN TIMEFRAME)
# =============================================================================

INDICATOR_OPTIMAL_SETTINGS = {
    # --- Trend Indicators ---
    "EMA": {
        "short_term": {"period": 9, "use": "fast signal line"},
        "medium_term": {"period": 21, "use": "trend direction"},
        "long_term": {"period": 50, "use": "major trend filter"},
        "15min_note": "9/21 EMA crossover is primary trend signal on 15-min",
    },
    "SMA": {
        "short_term": {"period": 20, "use": "Bollinger Band center, mean reversion anchor"},
        "medium_term": {"period": 50, "use": "intermediate trend"},
        "long_term": {"period": 200, "use": "major trend filter (use on hourly for 15-min)"},
    },
    "SUPERTREND": {
        "default": {"atr_period": 10, "multiplier": 3.0},
        "aggressive_scalping": {"atr_period": 7, "multiplier": 2.0},
        "conservative": {"atr_period": 14, "multiplier": 4.0},
        "recommended_15min": {"atr_period": 10, "multiplier": 3.0},
        "dual_timeframe_rule": (
            "Check 1-hour SuperTrend direction; only trade 15-min signals "
            "that align with the 1-hour SuperTrend direction"
        ),
    },
    "KAMA": {
        "efficiency_ratio_period": 10,
        "fastest_ema": 2,
        "slowest_ema": 30,
        "note": (
            "Kaufman Adaptive MA: stays flat in noise, follows fast in trend. "
            "ER near 1 = strong trend, ER near 0 = noise/chop. "
            "Use ER < 0.3 to detect chop, ER > 0.6 for strong trend."
        ),
    },

    # --- Momentum Indicators ---
    "RSI": {
        "period": 14,
        "overbought": 70,
        "oversold": 30,
        "strong_trend_overbought": 80,
        "strong_trend_oversold": 20,
        "centerline": 50,
        "bullish_bias_above": 40,
        "bearish_bias_below": 60,
        "faster_alternative": {"period": 9, "use": "more responsive for 15-min"},
    },
    "MACD": {
        "standard": {"fast": 12, "slow": 26, "signal": 9},
        "scalping": {"fast": 3, "slow": 10, "signal": 16},
        "recommended_15min": {"fast": 12, "slow": 26, "signal": 9},
        "signal_rules": {
            "buy": "MACD line crosses above signal line, both below zero = strongest",
            "sell": "MACD line crosses below signal line, both above zero = strongest",
            "histogram": "Rising histogram = strengthening momentum, falling = weakening",
        },
    },
    "STOCHASTIC": {
        "period_k": 14,
        "period_d": 3,
        "overbought": 80,
        "oversold": 20,
    },

    # --- Volatility Indicators ---
    "BOLLINGER_BANDS": {
        "period": 20,
        "std_dev": 2.0,
        "bandwidth_squeeze_threshold": 0.04,  # BBW < 4% indicates squeeze
        "bandwidth_expansion_threshold": 0.10,  # BBW > 10% indicates high volatility
        "squeeze_duration_significant": 15,  # 15+ candles of squeeze = significant
    },
    "ATR": {
        "period": 14,
        "intraday_period": 10,
        "use": "stop loss calculation, volatility filter, position sizing",
    },

    # --- Trend Strength ---
    "ADX": {
        "period": 14,
        "intraday_fast": 7,
        "thresholds": {
            "no_trend": 20,
            "weak_trend": 25,
            "strong_trend": 30,
            "very_strong_trend": 40,
            "extreme_trend": 50,
        },
        "rules": {
            "below_20": "No trend - use mean reversion strategies only",
            "20_to_25": "Weak trend - avoid trend-following, prefer range strategies",
            "25_to_30": "Moderate trend - trend-following acceptable with confirmation",
            "30_to_40": "Strong trend - trend-following optimal, avoid counter-trend",
            "above_40": "Very strong trend - ride it, but watch for exhaustion",
            "above_50": "Extreme - rare on intraday, expect reversal soon",
        },
        "rising_adx": "Trend is strengthening regardless of direction",
        "falling_adx": "Trend is weakening, prepare for range-bound conditions",
    },
    "DMI": {
        "period": 14,
        "crossover_rule": (
            "DI+ crosses above DI- = bullish, DI- crosses above DI+ = bearish. "
            "Only take signals when ADX > 25."
        ),
    },

    # --- Volume Indicators ---
    "VWAP": {
        "note": "Resets daily. Primary intraday fair-value indicator.",
        "bands": {
            "1_std": "First standard deviation - minor S/R",
            "2_std": "Second standard deviation - significant S/R",
            "3_std": "Third standard deviation - extreme, high probability mean reversion",
        },
        "rules": {
            "above_vwap": "Bullish bias for the session",
            "below_vwap": "Bearish bias for the session",
            "touch_and_hold": "Strong confirmation of trend direction",
        },
    },
    "OBV": {
        "note": "On Balance Volume - cumulative volume indicator",
        "interpretation": {
            "rising_with_price": "Confirms uptrend - strong buying pressure",
            "falling_with_price": "Confirms downtrend - strong selling pressure",
            "rising_price_falling_obv": "BEARISH DIVERGENCE - trend weakening",
            "falling_price_rising_obv": "BULLISH DIVERGENCE - accumulation, reversal likely",
            "obv_breakout_before_price": "LEADING SIGNAL - price will follow OBV direction",
        },
    },
}


# =============================================================================
# SECTION 2: INDICATOR COMBINATION STRATEGIES
# =============================================================================

INDICATOR_COMBINATIONS = {
    "TRIPLE_CONFIRMATION": {
        "name": "RSI + MACD + Bollinger Bands (Triple Confirmation)",
        "components": ["RSI_14", "MACD_12_26_9", "BBANDS_20_2"],
        "buy_signal": {
            "condition_1": "Price touches or dips below lower Bollinger Band",
            "condition_2": "RSI falls below 30 then starts turning upward",
            "condition_3": "MACD line crosses above signal line",
            "all_must_align": True,
            "confidence": "HIGH",
            "backtest_win_rate": 0.78,
        },
        "sell_signal": {
            "condition_1": "Price reaches or exceeds upper Bollinger Band",
            "condition_2": "RSI rises above 70 then starts dipping",
            "condition_3": "MACD line crosses below signal line",
            "all_must_align": True,
        },
        "avoid_when": "Bollinger Bands very tight (squeeze) - generates false signals",
    },

    "TREND_MOMENTUM": {
        "name": "SuperTrend + ADX + EMA Cross",
        "components": ["SUPERTREND_10_3", "ADX_14", "EMA_9", "EMA_21"],
        "buy_signal": {
            "condition_1": "SuperTrend turns green (bullish)",
            "condition_2": "ADX > 25 and rising",
            "condition_3": "EMA_9 crosses above EMA_21",
            "confidence": "HIGH for trending markets",
        },
        "filter": "Only take if 1-hour SuperTrend agrees with direction",
    },

    "VWAP_PULLBACK": {
        "name": "VWAP Pullback with RSI Confirmation",
        "components": ["VWAP", "RSI_14", "VOLUME"],
        "buy_signal": {
            "condition_1": "Price above VWAP for most of session (bullish day)",
            "condition_2": "Price pulls back to touch VWAP",
            "condition_3": "RSI between 40-50 (not oversold, just pulling back)",
            "condition_4": "Bullish candle forms on VWAP touch",
            "condition_5": "Volume below average on pullback, rises on bounce",
        },
        "stop_loss": "Just below VWAP",
        "target": "Previous swing high or +1 standard deviation VWAP band",
    },

    "BB_SQUEEZE_BREAKOUT": {
        "name": "Bollinger Band Squeeze + MACD + Volume",
        "components": ["BBANDS_20_2", "MACD_12_26_9", "OBV", "VOLUME"],
        "setup": {
            "condition_1": "Bollinger BandWidth near 6-month low (< 4%)",
            "condition_2": "Squeeze persists for 15+ candles on 15-min chart",
            "condition_3": "Check OBV direction during squeeze for bias",
        },
        "entry": {
            "condition_1": "Price breaks decisively above/below band",
            "condition_2": "Breakout candle has above-average volume (1.5x+)",
            "condition_3": "MACD crossover confirms direction",
            "condition_4": "RSI confirms (above 60 for bullish, below 40 for bearish)",
        },
        "head_fake_warning": (
            "Beware Bollinger head fakes: price breaks one band then reverses. "
            "Wait for candle CLOSE beyond band, not just a wick."
        ),
    },

    "ADX_DMI_CROSSOVER": {
        "name": "ADX + DMI Directional System",
        "components": ["ADX_14", "DI_PLUS", "DI_MINUS"],
        "buy_signal": {
            "condition_1": "DI+ crosses above DI-",
            "condition_2": "ADX > 25 (confirms trend strength)",
            "condition_3": "ADX is rising (trend strengthening)",
        },
        "sell_signal": {
            "condition_1": "DI- crosses above DI+",
            "condition_2": "ADX > 25",
            "condition_3": "ADX is rising",
        },
        "ignore_when": "ADX < 20 (trendless market, crossovers are noise)",
        "pullback_enhancement": (
            "When ADX > 30 and rising, wait for price to pull back to 20-EMA, "
            "then enter in trend direction when price crosses back"
        ),
    },

    "MEAN_REVERSION_SETUP": {
        "name": "RSI + Bollinger Band Mean Reversion",
        "components": ["RSI_14", "BBANDS_20_2", "ADX_14", "VWAP"],
        "conditions": {
            "regime": "ADX < 25 (ranging market ONLY)",
            "entry_long": (
                "Price at or below lower Bollinger Band AND "
                "RSI < 30 AND price near VWAP -2 std dev"
            ),
            "entry_short": (
                "Price at or above upper Bollinger Band AND "
                "RSI > 70 AND price near VWAP +2 std dev"
            ),
            "target": "Middle Bollinger Band (20-SMA) or VWAP",
            "stop": "1.5x ATR beyond entry or outside 3rd std dev band",
        },
        "critical_rule": "NEVER use mean reversion when ADX > 25",
    },
}


# =============================================================================
# SECTION 3: RSI DIVERGENCE RULES
# =============================================================================

RSI_DIVERGENCE_RULES = {
    "regular_bullish": {
        "definition": "Price makes LOWER LOW, RSI makes HIGHER LOW",
        "signal": "REVERSAL - momentum weakening on downside",
        "reliability": "Moderate-High (55-65% win rate)",
        "confirmation_required": [
            "Wait for bullish candlestick pattern (hammer, engulfing, morning star)",
            "RSI must cross back above 30",
            "Volume should increase on the bounce",
            "Wait for candle CLOSE above the prior swing candle",
        ],
        "stop_loss": "Below the most recent swing low",
        "target": "Previous resistance level or Fibonacci 61.8% retracement",
    },
    "regular_bearish": {
        "definition": "Price makes HIGHER HIGH, RSI makes LOWER HIGH",
        "signal": "REVERSAL - momentum weakening on upside",
        "reliability": "Moderate-High",
        "confirmation_required": [
            "Wait for bearish candlestick pattern (shooting star, evening star, engulfing)",
            "RSI must cross back below 70",
            "Volume should decrease on the new high",
        ],
        "stop_loss": "Above the most recent swing high",
    },
    "hidden_bullish": {
        "definition": "Price makes HIGHER LOW, RSI makes LOWER LOW",
        "signal": "CONTINUATION - trend resuming after pullback",
        "reliability": "Good in established uptrends",
        "use_case": "Confirms trend continuation; enter on pullback completion",
    },
    "hidden_bearish": {
        "definition": "Price makes LOWER HIGH, RSI makes HIGHER HIGH",
        "signal": "CONTINUATION - downtrend resuming after bounce",
        "reliability": "Good in established downtrends",
    },
    "multi_timeframe_enhancement": {
        "rule": (
            "Check RSI divergence on 1-hour chart first (overall trend). "
            "Then find precise entry on 15-min chart. "
            "This improves win rate by 15-20% vs single timeframe."
        ),
    },
    "rsi_settings_by_condition": {
        "standard_range": {"period": 14, "overbought": 70, "oversold": 30},
        "strong_trend": {"period": 14, "overbought": 80, "oversold": 20},
        "faster_signals": {"period": 9, "overbought": 70, "oversold": 30},
        "choppy_low_vol": {"period": 14, "overbought": 65, "oversold": 35},
    },
}


# =============================================================================
# SECTION 4: REGIME DETECTION & STRATEGY SELECTION
# =============================================================================

class MarketRegimeType(Enum):
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGE_BOUND = "range_bound"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY_CHOP = "high_volatility_chop"


REGIME_DETECTION_RULES = {
    "primary_indicator": "ADX",
    "secondary_indicators": ["Bollinger BandWidth", "KAMA Efficiency Ratio", "ATR percentile"],

    "regime_classification": {
        MarketRegimeType.STRONG_UPTREND: {
            "adx": "> 30 and rising",
            "di_plus_vs_di_minus": "DI+ > DI- by significant margin",
            "ema": "Price above 9-EMA > 21-EMA > 50-EMA",
            "supertrend": "Green/bullish",
            "strategy": "MOMENTUM - trend following, buy pullbacks to 9/21 EMA",
            "indicators_to_use": ["SuperTrend", "EMA crossover", "VWAP pullback"],
            "indicators_to_avoid": ["Mean reversion RSI", "counter-trend Bollinger"],
        },
        MarketRegimeType.WEAK_UPTREND: {
            "adx": "25-30, may be flat or slightly rising",
            "strategy": "CAUTIOUS MOMENTUM - smaller positions, tighter stops",
            "indicators_to_use": ["VWAP pullback", "EMA support", "RSI > 50 filter"],
        },
        MarketRegimeType.RANGE_BOUND: {
            "adx": "< 25",
            "bollinger_bw": "Moderate (not squeezing, not expanding)",
            "kama_er": "< 0.3 (high noise)",
            "strategy": "MEAN REVERSION - buy at support, sell at resistance",
            "indicators_to_use": ["RSI oversold/overbought", "Bollinger Band bounce",
                                  "VWAP mean reversion", "Stochastic"],
            "indicators_to_avoid": ["Trend-following", "SuperTrend", "MACD crossovers"],
        },
        MarketRegimeType.STRONG_DOWNTREND: {
            "adx": "> 30 and rising",
            "di_plus_vs_di_minus": "DI- > DI+ by significant margin",
            "strategy": "MOMENTUM SHORT - sell rallies, follow SuperTrend",
        },
        MarketRegimeType.HIGH_VOLATILITY_CHOP: {
            "adx": "< 25 but ATR is high",
            "bollinger_bw": "Wide bands but no direction",
            "strategy": "REDUCE POSITION SIZE or STAY OUT",
            "risk_adjustment": "Cut position size by 50%, widen stops by 1.5x",
        },
    },

    "hurst_exponent_guide": {
        "description": "Alternative regime detection using Hurst exponent",
        "H_above_0.5": "Trending/momentum market - use trend-following",
        "H_equals_0.5": "Random walk - no edge, reduce activity",
        "H_below_0.5": "Mean-reverting - use mean reversion strategies",
    },

    "volatility_regime_layers": {
        "layer_1_market_regime": "ADX + 200-period MA slope + VIX equivalent",
        "layer_2_liquidity": "Minimum volume filter, max bid-ask spread < 0.2%",
        "layer_3_volatility": "Only trade when ATR is in 20th-80th percentile of history",
    },
}


# =============================================================================
# SECTION 5: STOP LOSS & TAKE PROFIT RULES
# =============================================================================

RISK_MANAGEMENT_RULES = {
    "atr_based_stops": {
        "atr_period": 14,
        "intraday_atr_period": 10,
        "stop_loss_multiplier": {
            "tight": 1.5,   # For mean reversion trades
            "standard": 2.0,  # For trend-following trades
            "wide": 2.5,    # For breakout trades in volatile conditions
            "very_wide": 3.0,  # For high volatility environments
        },
        "formula": "Stop = Entry Price +/- (ATR * Multiplier)",
    },

    "take_profit_levels": {
        "level_1": {"atr_multiplier": 2.0, "position_pct": 0.50, "note": "Close 50% at 1:1.33 R:R"},
        "level_2": {"atr_multiplier": 3.0, "position_pct": 0.30, "note": "Close 30% at 1:2 R:R"},
        "level_3": {"atr_multiplier": 4.0, "position_pct": 0.20, "note": "Trail remaining 20%"},
    },

    "risk_reward_ratios": {
        "minimum_acceptable": 1.5,
        "standard_target": 2.0,
        "trending_market": 3.0,
        "scalping": 1.0,  # Only if win rate > 70%
        "rule": (
            "Never enter a trade where potential reward is less than 1.5x the risk. "
            "In trending markets, aim for 3:1 or better."
        ),
    },

    "position_sizing": {
        "max_risk_per_trade": 0.02,  # 2% of capital
        "max_risk_in_choppy_market": 0.01,  # 1% in high ADX-chop
        "formula": "Position Size = (Account * Risk%) / (Entry - Stop)",
        "max_concurrent_positions": 5,
        "max_single_position_pct": 0.20,
        "correlation_rule": "Max 2 positions in same sector",
    },

    "trailing_stop_rules": {
        "atr_trail": {
            "method": "Move stop to Entry + 1 ATR after price moves 2 ATR in favor",
            "trail_distance": "2x ATR behind current price",
        },
        "supertrend_trail": {
            "method": "Use SuperTrend line as trailing stop",
            "advantage": "Adapts to volatility automatically",
        },
        "ema_trail": {
            "method": "Trail stop behind 9-EMA or 21-EMA",
            "close_condition": "Close when candle closes beyond EMA",
        },
    },

    "intraday_specific": {
        "max_loss_per_day": 0.05,  # Stop trading after 5% daily loss
        "max_trades_per_day": 8,
        "mandatory_exit_time": "15:15 IST (3:15 PM)",  # Before square-off
        "no_new_trades_after": "14:45 IST (2:45 PM)",  # Unless clear trend
        "gap_risk": "No overnight positions in intraday strategy",
    },
}


# =============================================================================
# SECTION 6: VWAP STRATEGIES (DETAILED)
# =============================================================================

VWAP_STRATEGIES = {
    "pullback_long": {
        "name": "VWAP Pullback Long (Trend Following)",
        "setup": [
            "Stock has been trading ABOVE VWAP for majority of session",
            "Price pulls back to touch or slightly penetrate VWAP",
            "Look for 5-min bull flag pattern forming at VWAP",
        ],
        "entry": "Bullish candle closes on VWAP touch, or break above flag high",
        "stop_loss": "Just below VWAP (0.1-0.2% below)",
        "target": "Previous swing high or VWAP +1 std dev band",
        "volume_filter": "Volume should decrease on pullback, increase on bounce",
        "higher_tf_filter": "Daily trend must be bullish for strongest signals",
    },

    "pullback_short": {
        "name": "VWAP Pullback Short",
        "setup": [
            "Stock has been trading BELOW VWAP for majority of session",
            "Price rallies to touch or slightly exceed VWAP",
        ],
        "entry": "Bearish rejection candle at VWAP",
        "stop_loss": "Just above VWAP",
        "target": "Previous swing low or VWAP -1 std dev band",
    },

    "breakout": {
        "name": "VWAP Breakout (Momentum)",
        "setup": [
            "Price has been below VWAP (building short pressure)",
            "Sudden powerful break above VWAP with volume surge",
        ],
        "entry_aggressive": "On the VWAP break with strong candle",
        "entry_conservative": "Wait for micro-pullback to retest VWAP as support",
        "stop_loss": "Below VWAP",
        "logic": (
            "Short sellers forced to cover + new longs entering = "
            "double/triple buying pressure. Massive supply/demand imbalance."
        ),
    },

    "mean_reversion_bands": {
        "name": "VWAP Standard Deviation Mean Reversion",
        "setup": "Price reaches VWAP +/- 2 or 3 standard deviations",
        "entry": "At 2nd or 3rd std dev band with reversal candle pattern",
        "target": "VWAP center line",
        "rule": "Price tends to snap back to VWAP after touching outer bands",
        "best_when": "ADX < 25 (range-bound sessions)",
    },
}


# =============================================================================
# SECTION 7: BOLLINGER BAND SQUEEZE DETAILED RULES
# =============================================================================

BB_SQUEEZE_RULES = {
    "identification": {
        "bandwidth_formula": "(Upper Band - Lower Band) / Middle Band",
        "squeeze_threshold": "BandWidth near 6-month low",
        "squeeze_pct_threshold": 0.04,  # < 4% BandWidth
        "significant_squeeze_duration": "20-30 candles (5-7.5 hours on 15-min)",
        "longer_squeeze_bigger_breakout": True,
    },

    "breakout_entry_checklist": [
        "1. Bands have been squeezing for 15+ candles",
        "2. Price closes DECISIVELY beyond upper/lower band (not just wick)",
        "3. Breakout candle has volume >= 1.5x 20-period average volume",
        "4. MACD crossover confirms breakout direction",
        "5. RSI confirms: > 60 for bullish, < 40 for bearish",
        "6. Breakout aligns with key support/resistance levels",
    ],

    "direction_prediction_during_squeeze": {
        "use_obv": "Rising OBV during squeeze = likely upside breakout",
        "use_cmf": "Positive Chaikin Money Flow = accumulation, upside bias",
        "use_ad_line": "Rising A/D line = accumulation",
        "rule": (
            "Momentum oscillators and moving averages are USELESS during squeeze. "
            "Use volume-based indicators (OBV, CMF, A/D) to predict direction."
        ),
    },

    "head_fake_defense": {
        "definition": "Price breaks one band, then immediately reverses through the other",
        "defense_1": "Wait for candle CLOSE beyond band, not just intracandle break",
        "defense_2": "Require volume confirmation (weak volume = likely fake)",
        "defense_3": "Use MACD histogram direction as filter",
        "defense_4": "If breakout fails within 3 candles, reverse position",
    },

    "stop_loss": {
        "method": "Place stop at opposite Bollinger Band or 2x ATR",
        "trailing": "Move stop to middle band (20-SMA) after 2 ATR of profit",
    },
}


# =============================================================================
# SECTION 8: SUPERTREND DETAILED RULES (15-MIN)
# =============================================================================

SUPERTREND_RULES = {
    "settings": {
        "15_min_default": {"atr_period": 10, "multiplier": 3.0},
        "15_min_responsive": {"atr_period": 7, "multiplier": 2.0},
        "15_min_smooth": {"atr_period": 14, "multiplier": 4.0},
    },

    "basic_signals": {
        "buy": "SuperTrend line flips from above price to below (turns green)",
        "sell": "SuperTrend line flips from below price to above (turns red)",
        "trailing_stop": "SuperTrend line itself acts as dynamic trailing stop",
    },

    "dual_timeframe_system": {
        "rule": (
            "Plot SuperTrend on BOTH 15-min and 1-hour charts. "
            "Only take 15-min buy signals when 1-hour SuperTrend is also green. "
            "Only take 15-min sell signals when 1-hour SuperTrend is also red."
        ),
        "result": "Significantly reduces false signals and whipsaws",
    },

    "confirmation_with_other_indicators": {
        "with_rsi": "Only take SuperTrend buy when RSI > 50, sell when RSI < 50",
        "with_volume": "Breakout should have volume 1.5x average",
        "with_adx": "Best results when ADX > 25 (trending market)",
    },

    "limitations": {
        "ranging_markets": "Generates many false signals in sideways markets",
        "solution": "Use ADX filter - only trade SuperTrend when ADX > 25",
    },
}


# =============================================================================
# SECTION 9: VOLUME PROFILE & OBV RULES
# =============================================================================

VOLUME_RULES = {
    "obv_interpretation": {
        "trend_confirmation": {
            "both_rising": "OBV + Price both rising = strong uptrend, hold longs",
            "both_falling": "OBV + Price both falling = strong downtrend, hold shorts",
        },
        "divergence_signals": {
            "bearish_divergence": {
                "condition": "Price makes higher high, OBV makes lower high",
                "meaning": "Volume not supporting rally - trend weakening",
                "action": "Prepare to exit longs, watch for reversal confirmation",
                "reliability": "High - volume precedes price",
            },
            "bullish_divergence": {
                "condition": "Price makes lower low, OBV makes higher low",
                "meaning": "Accumulation occurring despite falling prices",
                "action": "Prepare for long entry on reversal confirmation",
            },
            "advanced_breakout": {
                "condition": "OBV breaks to new high while price fails to break prior high",
                "meaning": "Extremely bullish - price will follow OBV upward",
                "action": "Enter long before price breakout",
            },
        },
        "in_trading_range": {
            "rising_obv": "Accumulation - expect upside breakout",
            "falling_obv": "Distribution - expect downside breakout",
        },
    },

    "volume_confirmation_rules": {
        "healthy_uptrend": "Volume higher on up candles than down candles",
        "healthy_downtrend": "Volume higher on down candles than up candles",
        "breakout_volume": "Breakout candle should have >= 1.5x average volume",
        "pullback_volume": "Volume should DECREASE during pullbacks (healthy)",
        "climax_volume": "Extremely high volume after extended trend = potential reversal",
    },

    "volume_profile_intraday": {
        "point_of_control": "Price level with highest volume - strongest S/R",
        "value_area": "Range containing 70% of volume - expect price to stay here",
        "low_volume_nodes": "Price moves quickly through these - potential breakout zones",
        "high_volume_nodes": "Price consolidates here - support/resistance zones",
    },
}


# =============================================================================
# SECTION 10: INTRADAY TIME-OF-DAY RULES (INDIAN MARKET IST)
# =============================================================================

INTRADAY_TIME_RULES = {
    "market_hours": {
        "pre_open": "09:00 - 09:15 IST (order matching, price discovery)",
        "market_open": "09:15 IST",
        "market_close": "15:30 IST",
        "square_off_deadline": "15:15 IST (brokers start auto-squaring)",
    },

    "trading_phases": {
        "phase_1_opening_volatility": {
            "time": "09:15 - 09:30 IST",
            "volatility": "EXTREMELY HIGH",
            "description": (
                "Most volatile 15 minutes. Market reacts to overnight global cues, "
                "GIFT Nifty gap, and pre-market order imbalance. "
                "Sharp unpredictable swings in both directions."
            ),
            "strategy": "EXPERIENCED ONLY - gap plays, opening range formation",
            "15_min_rule": (
                "Wait for the first 15-minute candle to complete (09:15-09:30). "
                "High and low of this candle become initial support/resistance. "
                "Breakout above high = buy signal, breakdown below low = sell signal."
            ),
            "risk": "HIGH - wide stops needed, position size should be smaller",
        },

        "phase_2_morning_momentum": {
            "time": "09:30 - 10:30 IST",
            "volatility": "HIGH",
            "description": (
                "Primary momentum phase. The trend established in first 15 min "
                "often continues or sees significant reversal around 09:45-10:00."
            ),
            "strategy": "MOMENTUM TRADING - opening range breakouts, trend following",
            "rules": [
                "If trend from 09:15 persists past 10:00, it is likely the day's trend",
                "10:00 AM is a common reversal point - watch for rejection patterns",
                "Strongest setups of the day often occur in this window",
            ],
        },

        "phase_3_mid_morning_trend": {
            "time": "10:30 - 11:30 IST",
            "volatility": "MODERATE",
            "description": "Cleaner trends, less noise. Technical signals more reliable.",
            "strategy": "TREND FOLLOWING - pullbacks to EMA/VWAP, continuation patterns",
            "ideal_for": "Beginners - lower noise makes analysis easier",
        },

        "phase_4_lunch_chop": {
            "time": "11:30 - 13:30 IST",
            "volatility": "LOW",
            "description": (
                "Lunch hour doldrums. Volume drops, volatility decreases, "
                "sideways movement, many false signals. Trend indicators flatten."
            ),
            "strategy": "AVOID TRADING or use range-bound strategies only",
            "rules": [
                "Do NOT open new momentum/trend positions",
                "If holding positions, exit via existing rules or hold with wider stops",
                "Mean reversion with small targets can work for experienced traders",
                "Watch for breakout of lunch-hour range after 13:30",
            ],
            "critical_rule": "This is the #1 time to STAY OUT for most traders",
        },

        "phase_5_afternoon_revival": {
            "time": "13:30 - 14:30 IST",
            "volatility": "MODERATE - INCREASING",
            "description": (
                "Market often breaks out of lunch-hour range. "
                "Direction frequently matches the pre-lunch trend."
            ),
            "strategy": "Breakout of lunch range, trend resumption",
        },

        "phase_6_closing_action": {
            "time": "14:30 - 15:15 IST",
            "volatility": "HIGH",
            "description": (
                "Second most volatile period. Institutional position adjustments, "
                "mutual fund trading, index fund rebalancing. "
                "Strong moves often occur here."
            ),
            "strategy": "Quick momentum trades, but plan to EXIT by 15:15",
            "rules": [
                "No new positions after 14:45 unless very clear signal",
                "Square off ALL positions by 15:15 IST",
                "Watch for sharp reversals as traders square off",
            ],
        },
    },

    "gap_opening_strategy": {
        "gap_up": {
            "large_gap_above_1pct": (
                "Wait 15 min for opening range. If gap holds and price stays above "
                "opening candle low, go long targeting gap continuation. "
                "If price falls back into gap, it will likely fill."
            ),
            "small_gap_below_0.5pct": "Often fills within first hour - mean reversion",
        },
        "gap_down": {
            "large_gap_below_1pct": (
                "Wait 15 min. If selling continues, short with stop above opening high. "
                "If bounce occurs within first 30 min, may reverse."
            ),
        },
        "gift_nifty_signal": (
            "GIFT Nifty (formerly SGX Nifty) provides pre-market gap indication. "
            "Use it to prepare gap strategies before 09:15 open."
        ),
        "gap_fill_probability": {
            "small_gaps_under_0.5pct": "70-80% fill within session",
            "medium_gaps_0.5_to_1pct": "50-60% fill within session",
            "large_gaps_above_1pct": "30-40% fill - often continue in gap direction",
        },
    },

    "optimal_15min_candle_strategy": {
        "best_candles_to_trade": [
            "Candle 1 (09:15-09:30): Opening range formation - observe, don't trade",
            "Candle 2 (09:30-09:45): First actionable candle - breakout/breakdown",
            "Candle 3-6 (09:45-10:45): Prime momentum trading window",
            "Candle 7-10 (10:45-11:45): Trend continuation, pullback entries",
            "Candle 11-17 (11:45-13:30): LUNCH - avoid or mean reversion only",
            "Candle 18-21 (13:30-14:45): Afternoon breakout opportunities",
            "Candle 22-24 (14:45-15:15): Closing trades, must exit by candle 24",
        ],
    },
}


# =============================================================================
# SECTION 11: INDIAN MARKET SPECIFIC KNOWLEDGE
# =============================================================================

INDIAN_MARKET_KNOWLEDGE = {
    "nifty_50_characteristics": {
        "top_weight_stocks": {
            "RELIANCE": {"weight_pct": 9.29, "sector": "Energy/Conglomerate"},
            "HDFCBANK": {"weight_pct": 7.05, "sector": "Banking"},
            "BHARTIARTL": {"weight_pct": 5.86, "sector": "Telecom"},
            "TCS": {"weight_pct": 5.69, "sector": "IT"},
            "SBI": {"weight_pct": 4.82, "sector": "Banking"},
            "ICICIBANK": {"weight_pct": 4.81, "sector": "Banking"},
        },
        "sector_concentration": (
            "Banking + Financial Services = ~30% of Nifty 50. "
            "IT = ~15%. Energy = ~12%. These sectors drive index direction."
        ),
        "typical_intraday_range": "100-300 points on Nifty index",
    },

    "fii_dii_impact": {
        "overview": (
            "FII (Foreign Institutional Investors) and DII (Domestic Institutional Investors) "
            "flows are the primary drivers of Indian market direction."
        ),
        "2025_structural_shift": (
            "Historic shift: DIIs overtook FIIs in ownership for first time in 2025. "
            "DII holdings reached 18.26%, FII fell to 16.71%. "
            "Domestic SIP flows provide structural floor to market."
        ),
        "trading_rules": {
            "both_buying": "Strong bullish bias - market likely to rally",
            "both_selling": "Strong bearish bias - market likely to fall",
            "fii_selling_dii_buying": (
                "Mixed - DII cushions FII outflows. Market may consolidate. "
                "In 2025, DIIs absorbed massive FII selling, preventing crashes."
            ),
            "fii_buying_dii_selling": "Often precedes strong short-term rallies",
        },
        "data_source": "NSE website publishes FII/DII data daily after market close",
        "use_for_next_day": (
            "Previous day's FII/DII data informs next day's bias. "
            "Persistent FII selling for 5+ days = bearish setup. "
            "Sudden FII buying after selling streak = potential reversal."
        ),
    },

    "sector_rotation_patterns": {
        "early_bull_market": ["Banking", "IT", "Auto"],
        "mid_bull_market": ["Capital Goods", "Infrastructure", "Real Estate"],
        "late_bull_market": ["FMCG", "Pharma", "Defensives"],
        "bear_market_safe": ["FMCG", "Pharma", "Utilities"],
        "current_dii_favorites_2025": [
            "Financial Services", "FMCG", "Defence", "Healthcare"
        ],
        "current_fii_favorites_2025": [
            "Export-oriented sectors", "Global trade-linked sectors"
        ],
        "intraday_rotation_rule": (
            "Rotate into sectors showing strength in volumes and breakouts. "
            "Avoid sectors where FIIs are consistently selling."
        ),
    },

    "india_specific_patterns": {
        "expiry_day_volatility": (
            "Weekly expiry (Thursday) and monthly expiry (last Thursday) "
            "see elevated volatility, especially in last 2 hours. "
            "Options unwinding creates sharp directional moves."
        ),
        "global_cues": (
            "US markets (close at 04:30 IST / 05:30 IST DST), "
            "Asian markets (Japan opens 05:30 IST, China 06:30 IST) "
            "set the tone for Indian market opening."
        ),
        "rbi_policy_days": "Avoid intraday trading on RBI monetary policy days",
        "budget_day": "Extreme volatility - experienced traders only or stay out",
        "quarterly_results_season": (
            "Stocks reporting results see 3-10% gaps. "
            "Avoid holding intraday positions into results."
        ),
    },
}


# =============================================================================
# SECTION 12: BOOK WISDOM - ACTIONABLE RULES
# =============================================================================

BOOK_WISDOM = {
    "trading_in_the_zone_mark_douglas": {
        "core_principle": "Trading success is 80% psychology and 20% strategy",
        "five_trading_truths": [
            "1. Anything can happen in the market",
            "2. You don't need to know what will happen next to make money",
            "3. There is a random distribution between wins and losses for any edge",
            "4. An edge is nothing more than a higher probability of one thing over another",
            "5. Every moment in the market is unique",
        ],
        "actionable_rules_for_ai_coach": {
            "rule_1": (
                "Treat each trade as an independent event. Don't let the outcome of "
                "trade N affect the sizing or confidence of trade N+1."
            ),
            "rule_2": (
                "Define your edge BEFORE trading. The AI system's edge is the combination "
                "of indicators, regime detection, and risk management. Trust it."
            ),
            "rule_3": (
                "Pre-define risk for every trade. Never enter without a stop loss. "
                "Accept the risk emotionally (for the AI: implement it deterministically)."
            ),
            "rule_4": (
                "Follow rules consistently regardless of recent outcomes. "
                "Don't tighten stops after losses or widen them after wins."
            ),
            "rule_5": (
                "Process over outcome. A good trade that loses money is still a good trade "
                "if it followed the system. A bad trade that makes money is still a bad trade."
            ),
            "rule_6": (
                "Over-analysis is fear in disguise. Once the system has an edge, "
                "execute consistently. Don't add more indicators trying to be 'sure'."
            ),
        },
        "consistency_framework": (
            "Seven principles: Define your edge, pre-define risk, accept risk completely, "
            "act without hesitation, pay yourself as market makes profits available, "
            "monitor yourself for errors, understand absolute necessity of these principles."
        ),
    },

    "technical_analysis_john_murphy": {
        "ten_laws_summary": {
            "1_map_the_trend": "Use weekly and daily charts to identify long-term trend first",
            "2_spot_trend_follow": "Determine trend and follow it. Trade in direction of trend.",
            "3_find_support_resistance": "Best places to buy are near support, sell near resistance",
            "4_know_how_far_pullback": (
                "Use Fibonacci retracements: 33%, 50%, 66%. "
                "50% retracement is most common."
            ),
            "5_draw_trendlines": "Uptrend lines along lows, downtrend lines along highs",
            "6_follow_moving_averages": (
                "20-day for short-term, 50-day for intermediate, 200-day for major. "
                "Key crossovers: 5/20, 20/50, 50/200. Use EMAs for faster signals."
            ),
            "7_use_oscillators": (
                "RSI and Stochastic identify overbought/oversold. "
                "Use when ADX < 25 (ranging markets)."
            ),
            "8_use_macd": (
                "Buy: MACD crosses above signal below zero. "
                "Sell: MACD crosses below signal above zero. "
                "Weekly MACD overrides daily MACD."
            ),
            "9_trending_or_not": (
                "ADX tells you if market is trending. "
                "Rising ADX = use moving averages (trend tools). "
                "Falling ADX = use oscillators (range tools)."
            ),
            "10_confirm_with_volume": (
                "Volume must confirm trends. Rising volume in trend direction = healthy. "
                "Declining volume = trend weakening."
            ),
        },
        "intermarket_analysis": {
            "bonds_vs_stocks": "Falling bond yields often bullish for stocks",
            "dollar_vs_commodities": "Strong dollar often bearish for commodities",
            "commodities_vs_inflation": "Rising commodities signal inflation, affects rate policy",
            "for_indian_market": (
                "US Dollar Index (DXY) inversely correlated with FII flows into India. "
                "Rising DXY = FII outflows = bearish for Nifty."
            ),
        },
    },

    "new_trading_systems_perry_kaufman": {
        "key_concepts": {
            "efficiency_ratio": {
                "formula": "ER = |Price change over N periods| / Sum of |individual changes|",
                "range": "0 to 1",
                "interpretation": {
                    "near_0": "Pure noise - no directional movement, chop",
                    "near_1": "Perfect trend - all movement in one direction",
                    "below_0.3": "Very noisy - avoid trend-following",
                    "above_0.6": "Strong trend - trend-following ideal",
                },
                "use_for_regime": (
                    "ER is superior to ADX for noise detection. "
                    "Use ER to decide between momentum and mean reversion."
                ),
            },
            "adaptive_moving_average_kama": {
                "settings": {"er_period": 10, "fast_ema": 2, "slow_ema": 30},
                "behavior": (
                    "Flat during chop (like it's frozen), fast during trends. "
                    "Best of both worlds vs fixed moving averages."
                ),
                "dual_kama_crossover": (
                    "Fast KAMA (10,2,30) crossing above slow KAMA (20,2,30) = buy. "
                    "Reverse for sell."
                ),
            },
            "noise_in_equity_indices": (
                "Equity index markets have the GREATEST NOISE and WORST trend-following "
                "performance of all sectors. This means: "
                "1) Use adaptive indicators (KAMA) over fixed MAs for NIFTY stocks, "
                "2) Mean reversion may outperform momentum for indices, "
                "3) Wider stops needed to account for noise, "
                "4) Confirm trends with multiple timeframes."
            ),
        },
    },

    "evidence_based_ta_david_aronson": {
        "core_message": (
            "Most traditional technical analysis is not statistically validated. "
            "Traders must apply the scientific method to determine whether any trading "
            "signal has genuine predictive power."
        ),
        "actionable_rules_for_ai_coach": {
            "rule_1_data_mining_bias": (
                "When backtesting many indicator combinations, the best-performing ones "
                "are BIASED upward. Apply corrections: "
                "- Walk-forward testing (out-of-sample validation) "
                "- Multiple comparison corrections (Bonferroni, FDR) "
                "- Monte Carlo permutation tests "
                "- Cross-validation across different time periods and symbols"
            ),
            "rule_2_statistical_significance": (
                "Past performance is NECESSARY but NOT SUFFICIENT. "
                "A strategy must be statistically significant (p < 0.05) "
                "after correcting for data mining to be trusted."
            ),
            "rule_3_cognitive_bias_defense": {
                "confirmation_bias": "Don't only look at winning trades of a strategy",
                "overconfidence": "Even good backtests degrade in live trading",
                "recency_bias": "Recent performance is not more valid than long-term",
                "survivorship_bias": "Only surviving strategies get attention",
            },
            "rule_4_objective_over_subjective": (
                "Use OBJECTIVE, quantifiable rules only. "
                "No subjective chart patterns ('it looks like a head and shoulders'). "
                "Every rule must be programmable and testable."
            ),
            "rule_5_walk_forward_mandate": (
                "NEVER deploy a strategy based solely on in-sample performance. "
                "Require: "
                "- Minimum 3 walk-forward windows "
                "- Consistent performance across windows "
                "- Out-of-sample Sharpe > 0.5"
            ),
        },
    },
}


# =============================================================================
# SECTION 13: COMBINED STRATEGY DECISION MATRIX
# =============================================================================

STRATEGY_DECISION_MATRIX = {
    "description": (
        "Master decision framework: Given current market regime and time of day, "
        "which strategy and indicators should the AI coach select?"
    ),

    "matrix": {
        # (Regime, Time of Day) -> Strategy Config
        ("STRONG_TREND", "MORNING_9:30-10:30"): {
            "strategy": "MOMENTUM",
            "primary_indicators": ["SuperTrend", "EMA_9_21_cross", "ADX"],
            "secondary_indicators": ["VWAP_position", "Volume"],
            "position_size": "FULL (2% risk)",
            "stop_method": "2x ATR or SuperTrend line",
            "target": "3x ATR or trail with SuperTrend",
        },
        ("STRONG_TREND", "MID_DAY_10:30-14:30"): {
            "strategy": "TREND_PULLBACK",
            "primary_indicators": ["VWAP_pullback", "EMA_21_bounce", "RSI_40-60"],
            "position_size": "FULL",
            "stop_method": "Below VWAP or 21-EMA",
        },
        ("STRONG_TREND", "CLOSING_14:30-15:15"): {
            "strategy": "QUICK_MOMENTUM",
            "position_size": "HALF (1% risk) - limited time",
            "must_exit_by": "15:15 IST",
        },

        ("RANGE_BOUND", "MORNING_9:30-10:30"): {
            "strategy": "MEAN_REVERSION",
            "primary_indicators": ["RSI_extremes", "Bollinger_Band_bounce", "VWAP_bands"],
            "position_size": "HALF (1% risk) - range-bound is tricky in morning",
            "stop_method": "1.5x ATR",
            "target": "Middle BB or VWAP center",
        },
        ("RANGE_BOUND", "LUNCH_11:30-13:30"): {
            "strategy": "NO_TRADE or TINY_MEAN_REVERSION",
            "primary_indicators": ["RSI_extremes", "BB_bounce"],
            "position_size": "QUARTER (0.5% risk) or ZERO",
        },
        ("RANGE_BOUND", "CLOSING_14:30-15:15"): {
            "strategy": "BREAKOUT_WATCH",
            "note": "Range may break in closing hour - watch for BB squeeze breakout",
            "position_size": "HALF",
        },

        ("HIGH_VOL_CHOP", "ANY_TIME"): {
            "strategy": "REDUCE_OR_AVOID",
            "position_size": "QUARTER (0.5% risk) maximum",
            "stop_method": "3x ATR (very wide)",
            "note": "High volatility without direction = worst condition for any strategy",
        },

        ("BB_SQUEEZE", "ANY_TIME"): {
            "strategy": "WAIT_FOR_BREAKOUT",
            "primary_indicators": ["BB_bandwidth", "OBV_direction", "Volume_surge"],
            "position_size": "FULL on confirmed breakout",
            "entry_rule": "Only on decisive band break with 1.5x volume",
        },
    },
}


# =============================================================================
# SECTION 14: COACH OPTIMIZATION PARAMETERS
# =============================================================================

COACH_OPTIMIZATION_PARAMS = {
    "indicator_weight_adjustment_rules": {
        "increase_weight_when": [
            "Indicator has been profitable in last 20 trades",
            "Indicator aligns with current market regime",
            "Indicator has low correlation with other active indicators",
            "Indicator signals confirmed by volume",
        ],
        "decrease_weight_when": [
            "Indicator generates false signals in current regime",
            "Indicator is redundant with another active indicator",
            "Indicator has been unprofitable in last 20 trades",
            "Market regime has shifted (e.g., trending to ranging)",
        ],
        "max_weight_change_per_adjustment": 0.10,  # 10% max change
        "min_trades_before_adjusting": 20,
    },

    "regime_transition_rules": {
        "trending_to_ranging": {
            "trigger": "ADX drops below 25 after being above 30",
            "action": [
                "Reduce trend-following indicator weights by 30%",
                "Increase mean-reversion indicator weights by 30%",
                "Tighten stop losses (from 2x ATR to 1.5x ATR)",
                "Reduce position sizes by 25%",
            ],
        },
        "ranging_to_trending": {
            "trigger": "ADX rises above 25 after being below 20, with volume surge",
            "action": [
                "Increase trend-following indicator weights by 30%",
                "Decrease mean-reversion indicator weights by 30%",
                "Widen stop losses (from 1.5x ATR to 2x ATR)",
                "Allow full position sizes",
            ],
        },
        "volatility_spike": {
            "trigger": "ATR increases by 50%+ in 5 periods",
            "action": [
                "Reduce ALL position sizes by 50%",
                "Widen stops by 1.5x",
                "Wait for volatility to stabilize before normal trading",
            ],
        },
    },

    "performance_feedback_loop": {
        "review_frequency": "Every 20 trades or end of trading day",
        "metrics_to_track": [
            "Win rate by indicator combination",
            "Average R:R achieved vs targeted",
            "Win rate by time of day",
            "Win rate by market regime",
            "Maximum adverse excursion (MAE) - how far trades go against before winning",
            "Maximum favorable excursion (MFE) - how far trades go before exiting",
        ],
        "adjustment_triggers": {
            "win_rate_below_40pct": "Review indicator combination, may need regime filter",
            "avg_rr_below_1": "Stops too tight or targets too ambitious",
            "mae_too_high": "Entry timing is poor, need better confirmation",
            "mfe_much_higher_than_actual_profit": "Exiting too early, try trailing stops",
        },
    },

    "anti_overfitting_rules": {
        "max_indicators_in_combination": 5,
        "min_trades_for_statistical_validity": 30,
        "walk_forward_windows": 3,
        "out_of_sample_sharpe_minimum": 0.5,
        "in_sample_vs_oos_degradation_max": 0.40,  # Max 40% degradation acceptable
        "bonferroni_correction": (
            "When testing N indicator combinations, divide significance level by N. "
            "E.g., testing 100 combos: require p < 0.05/100 = 0.0005"
        ),
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_regime_strategy(adx_value: float, adx_rising: bool,
                        er_value: float = None,
                        bbw_value: float = None) -> Dict:
    """
    Determine the optimal strategy based on current market regime indicators.

    Args:
        adx_value: Current ADX value (0-100)
        adx_rising: Whether ADX is rising
        er_value: Kaufman Efficiency Ratio (0-1), optional
        bbw_value: Bollinger BandWidth, optional

    Returns:
        Dictionary with strategy recommendation
    """
    # Check for BB Squeeze first
    if bbw_value is not None and bbw_value < 0.04:
        return {
            "regime": "BB_SQUEEZE",
            "strategy": "WAIT_FOR_BREAKOUT",
            "use_indicators": ["BB_bandwidth", "OBV", "Volume"],
            "position_size_multiplier": 1.0,  # Full on breakout
            "stop_multiplier": 2.0,
        }

    # Strong trend
    if adx_value > 30 and adx_rising:
        return {
            "regime": "STRONG_TREND",
            "strategy": "MOMENTUM",
            "use_indicators": ["SuperTrend", "EMA_cross", "VWAP_pullback", "ADX"],
            "avoid_indicators": ["RSI_mean_reversion", "BB_bounce"],
            "position_size_multiplier": 1.0,
            "stop_multiplier": 2.0,
            "target_multiplier": 3.0,
        }

    # Moderate trend
    if 25 <= adx_value <= 30:
        return {
            "regime": "MODERATE_TREND",
            "strategy": "CAUTIOUS_MOMENTUM",
            "use_indicators": ["EMA_cross", "VWAP", "RSI_centerline"],
            "position_size_multiplier": 0.75,
            "stop_multiplier": 2.0,
            "target_multiplier": 2.5,
        }

    # Range-bound / No trend
    if adx_value < 25:
        # Check if it's high-volatility chop
        if er_value is not None and er_value < 0.15:
            return {
                "regime": "HIGH_NOISE_CHOP",
                "strategy": "REDUCE_OR_AVOID",
                "position_size_multiplier": 0.25,
                "stop_multiplier": 3.0,
                "note": "Extremely noisy - minimal trading recommended",
            }
        return {
            "regime": "RANGE_BOUND",
            "strategy": "MEAN_REVERSION",
            "use_indicators": ["RSI_extremes", "BB_bounce", "VWAP_bands", "Stochastic"],
            "avoid_indicators": ["SuperTrend", "MACD_crossover", "EMA_cross"],
            "position_size_multiplier": 0.5,
            "stop_multiplier": 1.5,
            "target_multiplier": 1.5,
        }

    # Default fallback
    return {
        "regime": "UNCERTAIN",
        "strategy": "REDUCE_SIZE",
        "position_size_multiplier": 0.5,
    }


def get_time_phase(hour: int, minute: int) -> Dict:
    """
    Determine the current trading phase based on IST time.

    Args:
        hour: Hour in IST (0-23)
        minute: Minute (0-59)

    Returns:
        Dictionary with phase information and recommendations
    """
    time_minutes = hour * 60 + minute

    if time_minutes < 9 * 60 + 15:
        return {"phase": "PRE_MARKET", "can_trade": False, "note": "Market not open"}

    if time_minutes < 9 * 60 + 30:
        return {
            "phase": "OPENING_VOLATILITY",
            "can_trade": True,
            "volatility": "EXTREME",
            "position_size_multiplier": 0.5,
            "strategy_preference": "gap_play_or_observe",
            "note": "First 15-min candle forming - observe for opening range",
        }

    if time_minutes < 10 * 60 + 30:
        return {
            "phase": "MORNING_MOMENTUM",
            "can_trade": True,
            "volatility": "HIGH",
            "position_size_multiplier": 1.0,
            "strategy_preference": "momentum_breakout",
            "note": "Prime trading window - strongest signals of the day",
        }

    if time_minutes < 11 * 60 + 30:
        return {
            "phase": "MID_MORNING_TREND",
            "can_trade": True,
            "volatility": "MODERATE",
            "position_size_multiplier": 1.0,
            "strategy_preference": "trend_following_pullback",
            "note": "Clean trends, good for pullback entries",
        }

    if time_minutes < 13 * 60 + 30:
        return {
            "phase": "LUNCH_CHOP",
            "can_trade": False,  # Recommended to avoid
            "volatility": "LOW",
            "position_size_multiplier": 0.25,
            "strategy_preference": "mean_reversion_only_or_avoid",
            "note": "AVOID TRADING - lunch hour chop, false signals likely",
        }

    if time_minutes < 14 * 60 + 30:
        return {
            "phase": "AFTERNOON_REVIVAL",
            "can_trade": True,
            "volatility": "MODERATE",
            "position_size_multiplier": 0.75,
            "strategy_preference": "breakout_of_lunch_range",
            "note": "Market often breaks lunch range, trade in pre-lunch trend direction",
        }

    if time_minutes < 14 * 60 + 45:
        return {
            "phase": "CLOSING_ACTION",
            "can_trade": True,
            "volatility": "HIGH",
            "position_size_multiplier": 0.5,
            "strategy_preference": "quick_momentum",
            "note": "Last trading opportunities, must plan exit by 15:15",
        }

    if time_minutes < 15 * 60 + 15:
        return {
            "phase": "SQUARE_OFF",
            "can_trade": False,
            "volatility": "HIGH",
            "position_size_multiplier": 0.0,
            "strategy_preference": "exit_all",
            "note": "EXIT ALL POSITIONS - square-off deadline approaching",
        }

    return {"phase": "MARKET_CLOSED", "can_trade": False}


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.02,
    regime_multiplier: float = 1.0,
    time_multiplier: float = 1.0,
) -> Dict:
    """
    Calculate position size based on risk management rules.

    Args:
        capital: Total trading capital
        entry_price: Planned entry price
        stop_price: Planned stop loss price
        risk_pct: Maximum risk per trade (default 2%)
        regime_multiplier: Adjustment based on market regime (0.25-1.0)
        time_multiplier: Adjustment based on time of day (0.25-1.0)

    Returns:
        Dictionary with position sizing details
    """
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share == 0:
        return {"error": "Stop price equals entry price"}

    adjusted_risk_pct = risk_pct * regime_multiplier * time_multiplier
    max_risk_amount = capital * adjusted_risk_pct
    shares = int(max_risk_amount / risk_per_share)

    position_value = shares * entry_price
    position_pct = position_value / capital

    # Cap at 20% of capital per position
    if position_pct > 0.20:
        shares = int((capital * 0.20) / entry_price)
        position_value = shares * entry_price
        position_pct = position_value / capital

    return {
        "shares": shares,
        "position_value": position_value,
        "position_pct_of_capital": round(position_pct, 4),
        "risk_amount": round(shares * risk_per_share, 2),
        "risk_pct_of_capital": round((shares * risk_per_share) / capital, 4),
        "adjusted_risk_pct": round(adjusted_risk_pct, 4),
        "regime_multiplier": regime_multiplier,
        "time_multiplier": time_multiplier,
    }


def get_stop_and_target(
    entry_price: float,
    atr_value: float,
    direction: str,  # "long" or "short"
    regime: str = "MODERATE_TREND",
) -> Dict:
    """
    Calculate stop loss and take profit levels based on ATR and regime.

    Args:
        entry_price: Entry price
        atr_value: Current ATR value
        direction: "long" or "short"
        regime: Current market regime

    Returns:
        Dictionary with stop and target levels
    """
    # Regime-based multipliers
    multipliers = {
        "STRONG_TREND": {"stop": 2.0, "tp1": 2.0, "tp2": 3.0, "tp3": 4.0},
        "MODERATE_TREND": {"stop": 2.0, "tp1": 2.0, "tp2": 3.0, "tp3": 3.5},
        "RANGE_BOUND": {"stop": 1.5, "tp1": 1.5, "tp2": 2.0, "tp3": 2.5},
        "HIGH_NOISE_CHOP": {"stop": 3.0, "tp1": 2.0, "tp2": 3.0, "tp3": 4.0},
    }

    m = multipliers.get(regime, multipliers["MODERATE_TREND"])

    if direction == "long":
        stop = entry_price - (atr_value * m["stop"])
        tp1 = entry_price + (atr_value * m["tp1"])
        tp2 = entry_price + (atr_value * m["tp2"])
        tp3 = entry_price + (atr_value * m["tp3"])
    else:  # short
        stop = entry_price + (atr_value * m["stop"])
        tp1 = entry_price - (atr_value * m["tp1"])
        tp2 = entry_price - (atr_value * m["tp2"])
        tp3 = entry_price - (atr_value * m["tp3"])

    return {
        "stop_loss": round(stop, 2),
        "take_profit_1": {"price": round(tp1, 2), "close_pct": 0.50},
        "take_profit_2": {"price": round(tp2, 2), "close_pct": 0.30},
        "take_profit_3": {"price": round(tp3, 2), "close_pct": 0.20},
        "risk_reward_at_tp1": round(m["tp1"] / m["stop"], 2),
        "risk_reward_at_tp2": round(m["tp2"] / m["stop"], 2),
        "risk_reward_at_tp3": round(m["tp3"] / m["stop"], 2),
    }
