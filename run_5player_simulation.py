#!/usr/bin/env python3
"""
5-Player + Coach 60-Day Intraday Simulation.

Runs 5 independent players with diverse evolved strategies on 15-minute
NIFTY 50 candles for 60 days.  An LLM Coach (Gemini) analyses every
player's trades each evening, generates bounded patches, validates them,
and applies improvements overnight.

Usage:
    python run_5player_simulation.py                     # full 60-day run
    python run_5player_simulation.py --days 10           # shorter run
    python run_5player_simulation.py --symbols 10        # fewer symbols
    python run_5player_simulation.py --no-coach          # skip LLM coach
"""

import argparse
import json
import logging
import sys
import time as _time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# ── Core components ──
from trading_evolution.data.intraday import IntradayDataFetcher, IntradayConfig
from trading_evolution.data.cache import DataCache
from trading_evolution.indicators.universe import IndicatorUniverse
from trading_evolution.indicators.calculator import IndicatorCalculator
from trading_evolution.indicators.normalizer import IndicatorNormalizer
from trading_evolution.super_indicator.dna import (
    SuperIndicatorDNA, IndicatorGene, create_dna_from_weights,
)
from trading_evolution.super_indicator.core import SuperIndicator
from trading_evolution.super_indicator.signals import SignalType, PositionState
from trading_evolution.indicators.normalizer import SignalAggregator
from trading_evolution.player.trader import Player
from trading_evolution.player.portfolio import Portfolio, Trade
from trading_evolution.player.risk_manager import RiskManager, RiskParameters
from trading_evolution.player.execution import ExecutionEngine
from trading_evolution.ai_config import AIConfig

# ── Coach components ──
from trading_evolution.ai_coach.post_market_analyzer import PostMarketAnalyzer
from trading_evolution.coach.candidate_generator import (
    CandidateGenerator, MistakePattern, CandidateExperiment,
)
from trading_evolution.coach.patch_language import MistakeType, StrategyPatch

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
# Silence noisy library loggers
for _lib in ("httpx", "google_genai", "urllib3", "yfinance"):
    logging.getLogger(_lib).setLevel(logging.ERROR)
logger = logging.getLogger("5player_sim")
logger.setLevel(logging.INFO)

# ───────────────────────────────────────────────────────────────────────
# LLM Expert Coach — direct Gemini call with full market context
# ───────────────────────────────────────────────────────────────────────
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ───────────────────────────────────────────────────────────────────────
# TRADING KNOWLEDGE BASE — compiled from books, research, backtests
# ───────────────────────────────────────────────────────────────────────
try:
    from trading_evolution.ai_coach.trading_knowledge_base import (
        INDICATOR_OPTIMAL_SETTINGS,
        INDICATOR_COMBINATIONS,
        RSI_DIVERGENCE_RULES,
        REGIME_DETECTION_RULES,
        RISK_MANAGEMENT_RULES,
        VWAP_STRATEGIES,
        BB_SQUEEZE_RULES,
        SUPERTREND_RULES,
        VOLUME_RULES,
        INTRADAY_TIME_RULES,
        INDIAN_MARKET_KNOWLEDGE,
        BOOK_WISDOM,
        STRATEGY_DECISION_MATRIX,
        COACH_OPTIMIZATION_PARAMS,
    )
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False


def build_knowledge_text() -> str:
    """Serialize the trading knowledge base into a compact text for LLM prompts."""
    if not KNOWLEDGE_AVAILABLE:
        return ""
    import json

    sections = []

    # 1. Indicator combinations (most actionable)
    sections.append("## PROVEN INDICATOR COMBINATIONS (Backtested)")
    for name, combo in INDICATOR_COMBINATIONS.items():
        parts = [f"**{combo.get('name', name)}**"]
        parts.append(f"  Components: {combo.get('components', [])}")
        buy = combo.get("buy_signal", combo.get("entry", combo.get("conditions", {})))
        if isinstance(buy, dict):
            for k, v in buy.items():
                parts.append(f"  {k}: {v}")
        if "avoid_when" in combo:
            parts.append(f"  AVOID: {combo['avoid_when']}")
        if "critical_rule" in combo:
            parts.append(f"  CRITICAL: {combo['critical_rule']}")
        sections.append("\n".join(parts))

    # 2. Regime detection rules
    sections.append("\n## REGIME DETECTION RULES")
    for regime, rules in REGIME_DETECTION_RULES.items():
        if isinstance(rules, dict):
            indicators = rules.get("indicators", {})
            strategy = rules.get("strategy", "")
            avoid = rules.get("avoid", "")
            position = rules.get("position_sizing", "")
            line = f"  {regime}: strategy={strategy}"
            if avoid:
                line += f" | AVOID: {avoid}"
            if position:
                line += f" | sizing: {position}"
            sections.append(line)

    # 3. Risk management essentials
    sections.append("\n## RISK MANAGEMENT (from Kaufman & Murphy)")
    rm = RISK_MANAGEMENT_RULES
    atr_rules = rm.get("atr_based_stops", {})
    for regime, rule in atr_rules.items():
        if isinstance(rule, dict):
            sections.append(f"  {regime}: stop={rule.get('stop_distance','?')}, target={rule.get('target','?')}")
    daily = rm.get("daily_risk_limits", {})
    sections.append(f"  Daily limits: max_loss={daily.get('max_daily_loss','5%')}, "
                    f"max_trades={daily.get('max_trades_per_day', 8)}")

    # 4. Intraday time rules (NIFTY-specific)
    sections.append("\n## INTRADAY TIME RULES (IST - Indian Market)")
    phases = INTRADAY_TIME_RULES.get("trading_phases", {})
    for phase_key, phase in phases.items():
        if isinstance(phase, dict):
            t = phase.get("time", "")
            vol = phase.get("volatility", "")
            strat = phase.get("strategy", "")
            sections.append(f"  {t}: vol={vol}, strategy={strat}")

    # 5. Key book insights (actionable rules only)
    sections.append("\n## KEY INSIGHTS FROM TRADING BOOKS")

    # Kaufman - noise
    kaufman = BOOK_WISDOM.get("new_trading_systems_perry_kaufman", {}).get("key_concepts", {})
    noise = kaufman.get("noise_in_equity_indices", "")
    if noise:
        sections.append(f"  Kaufman: {noise}")

    # Murphy - 10 laws condensed
    murphy = BOOK_WISDOM.get("technical_analysis_john_murphy", {}).get("ten_laws_summary", {})
    sections.append("  Murphy's Laws: " + "; ".join(
        f"{k}: {v[:60]}" for k, v in list(murphy.items())[:5]
    ))

    # Douglas - actionable rules
    douglas = BOOK_WISDOM.get("trading_in_the_zone_mark_douglas", {})
    ar = douglas.get("actionable_rules_for_ai_coach", {})
    for rk, rv in list(ar.items())[:3]:
        sections.append(f"  Douglas {rk}: {rv[:80]}")

    # Aronson - anti-overfitting
    aronson = BOOK_WISDOM.get("evidence_based_ta_david_aronson", {})
    ar2 = aronson.get("actionable_rules_for_ai_coach", {})
    for rk, rv in list(ar2.items())[:2]:
        sections.append(f"  Aronson {rk}: {str(rv)[:80]}")

    # 6. Indian market specifics
    sections.append("\n## INDIAN MARKET (NIFTY 50)")
    nifty = INDIAN_MARKET_KNOWLEDGE.get("nifty_50_characteristics", {})
    sectors = INDIAN_MARKET_KNOWLEDGE.get("sector_rotation", {})
    if isinstance(sectors, dict):
        for cycle, info in list(sectors.items())[:3]:
            if isinstance(info, dict):
                leading = info.get("leading_sectors", info.get("sectors", []))
                sections.append(f"  {cycle}: {leading}")

    # 7. Coach optimization params
    sections.append("\n## COACH OPTIMIZATION RULES")
    cop = COACH_OPTIMIZATION_PARAMS
    weight_rules = cop.get("indicator_weight_adjustment_rules", {})
    sections.append("  Increase weight when: " + "; ".join(weight_rules.get("increase_weight_when", [])[:3]))
    sections.append("  Decrease weight when: " + "; ".join(weight_rules.get("decrease_weight_when", [])[:3]))
    anti = cop.get("anti_overfitting_rules", {})
    sections.append(f"  Max indicators: {anti.get('max_indicators_in_combination', 5)}")
    sections.append(f"  Min trades for validity: {anti.get('min_trades_for_statistical_validity', 30)}")

    return "\n".join(sections)


# ───────────────────────────────────────────────────────────────────────
# PRE-SIMULATION STRATEGY OPTIMIZER — uses knowledge to redesign players
# ───────────────────────────────────────────────────────────────────────

STRATEGY_OPTIMIZER_PROMPT = """You are a world-class quantitative trading systems designer.
You have studied every major trading book and have deep expertise in:
- Technical analysis indicator combinations (Murphy, Kaufman, Aronson)
- Market microstructure and 15-minute intraday trading
- Indian equity markets (NIFTY 50)
- Statistical validation and anti-overfitting (Aronson)

## YOUR TRADING KNOWLEDGE
{knowledge_text}

## AVAILABLE INDICATORS (85+ from our library)
MOMENTUM: RSI_7, RSI_14, RSI_21, MACD_12_26_9, MACD_8_17_9, MACD_5_35_5,
  STOCH_14_3, STOCH_5_3, STOCH_21_5, WILLR_14, WILLR_28, CCI_14, CCI_20,
  ROC_10, ROC_20, MOM_10, MOM_20, CMO_14, CMO_20, TSI_13_25, UO_7_14_28,
  AO_5_34, KST, COPPOCK

TREND: ADX_14, ADX_20, AROON_14, AROON_25, SUPERTREND_7_3, SUPERTREND_10_2,
  SUPERTREND_20_3, PSAR, LINREG_SLOPE_14, LINREG_SLOPE_25, ICHIMOKU, VORTEX_14

VOLATILITY: ATR_14, ATR_20, NATR_14, NATR_20, BBANDS_20_2, BBANDS_20_2.5,
  BBANDS_10_1.5, KC_20_2, KC_20_1.5, DONCHIAN_20, DONCHIAN_50, TRUERANGE, MASS_INDEX

VOLUME: OBV, AD, ADOSC_3_10, CMF_20, CMF_21, MFI_14, MFI_20, EFI_13, EFI_20, PVI, NVI

OVERLAP/MA: SMA_10, SMA_20, SMA_50, EMA_9, EMA_10, EMA_20, EMA_50, WMA_10, WMA_20,
  DEMA_10, DEMA_20, TEMA_10, TEMA_20, KAMA_10, KAMA_20, T3_5, T3_10, HMA_9, HMA_16,
  VWMA_10, VWMA_20

OTHER: ZSCORE_20, ZSCORE_50

## SYSTEM MECHANICS
- Super Indicator (SI) = weighted average of normalized indicators (range -1 to +1)
- Entry: LONG when SI > entry_threshold, SHORT when SI < -entry_threshold
- Exit: LONG exits when SI < exit_threshold (negative = wait for reversal)
- Exit: SHORT exits when SI > |exit_threshold|
- Minimum holding period: min_hold_bars (each bar = 15 min)
- EMA50 trend filter gates entries (long only in uptrend, short only in downtrend)
- ATR-based stops (currently 5x ATR)
- EOD flatten at 15:15 IST, no new entries after 14:45 IST

## CURRENT PLAYER: {player_id} ({player_label})
Strategy: {strategy_desc}
Current weights: {current_weights}
Entry threshold: {entry_threshold} | Exit threshold: {exit_threshold}
Min hold bars: {min_hold_bars}

## PREVIOUS PERFORMANCE (if available)
{prev_performance}

## TASK
Redesign this player's strategy using your trading knowledge.
Apply the proven indicator combinations, regime-aware logic, and risk management principles.

CRITICAL DESIGN PRINCIPLES:
1. Use MAX 6-8 indicators (Aronson: more = overfitting)
2. Every indicator must serve a SPECIFIC PURPOSE (trend, momentum, volume, volatility filter)
3. Use PROVEN COMBINATIONS from the knowledge base (Triple Confirmation, Trend Momentum, etc.)
4. Negative weights are allowed — they INVERT the signal (e.g., negative BBANDS = mean reversion)
5. Kaufman says equity indices have the GREATEST NOISE — use adaptive indicators (KAMA, ADX filter)
6. Consider the INTRADAY TIME RULES — lunch hours are choppy, mornings are best
7. Exit threshold should be significantly negative (-0.08 to -0.20) to avoid premature exits
8. Entry threshold: higher = fewer but better trades (0.25-0.40)
9. Min hold bars: 3-5 (45-75 min) to let trades develop through noise

Respond with ONLY valid JSON:
{{
  "strategy_name": "short descriptive name",
  "rationale": "1-2 sentences explaining the design logic",
  "weights": {{
    "INDICATOR_NAME": 0.85,
    "INDICATOR_NAME": -0.60
  }},
  "entry_threshold": 0.30,
  "exit_threshold": -0.12,
  "min_hold_bars": 4
}}
"""


def optimize_player_strategy(pid: str, cfg: Dict, knowledge_text: str,
                             prev_perf: str = "No previous data",
                             api_key: str = "", model_name: str = "") -> Optional[Dict]:
    """Call LLM to redesign a player's strategy using trading knowledge."""
    if not GENAI_AVAILABLE or not api_key:
        return None

    weights_str = "\n".join(
        f"  {k}: {v:.3f}" for k, v in sorted(cfg["weights"].items(), key=lambda x: -x[1])
    )

    prompt = STRATEGY_OPTIMIZER_PROMPT.format(
        knowledge_text=knowledge_text,
        player_id=pid,
        player_label=cfg["label"],
        strategy_desc=cfg.get("original", "Unknown"),
        current_weights=weights_str,
        entry_threshold=cfg["entry_threshold"],
        exit_threshold=cfg["exit_threshold"],
        min_hold_bars=cfg.get("min_hold_bars", 4),
        prev_performance=prev_perf,
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 4096},
        )
        raw = response.text.strip()

        # Clean JSON
        import re
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
        start_idx = raw.find('{')
        end_idx = raw.rfind('}')
        if start_idx != -1 and end_idx != -1:
            raw = raw[start_idx:end_idx+1]

        # Fix common JSON issues from LLM
        raw = re.sub(r',\s*}', '}', raw)
        raw = re.sub(r',\s*]', ']', raw)
        raw = re.sub(r'//[^\n]*', '', raw)

        data = json.loads(raw)
        return data
    except Exception as e:
        print(f"    [Optimizer] Failed for {pid}: {e}")
        return None


EXPERT_COACH_PROMPT = """You are a world-class quantitative portfolio manager and expert stock market analyst.
You are coaching an intraday trading system that trades Indian NIFTY 50 stocks on 15-minute candles.

## YOUR DEEP TRADING KNOWLEDGE (from Murphy, Kaufman, Douglas, Aronson)
{knowledge_text}

## YOUR EXPERTISE
You have deep knowledge of:
- Indian equity markets (NSE, NIFTY 50, sector dynamics)
- Technical analysis (RSI, MACD, Bollinger, SuperTrend, ADX, etc.)
- Market microstructure (15-min bar noise, slippage, bid-ask spread)
- Regime detection (trending, mean-reverting, volatile, low-volatility trap)
- Position management (entry timing, exit logic, holding period, stop placement)
- Proven indicator combinations from backtested research
- Anti-overfitting principles (Aronson): max 6-8 indicators, walk-forward validation

## MARKET DATA TODAY ({analysis_date})
{market_context}

## PLAYER: {player_id} ({player_label})
Strategy: {strategy_desc}
Active indicators: {active_indicators}
Entry threshold: {entry_threshold:.3f} | Exit threshold: {exit_threshold:.3f}
Min hold bars: {min_hold_bars}

Current weights:
{weights_str}

## TODAY'S PERFORMANCE
{trades_summary}

## TRADE DETAILS
{trades_detail}

## TASK
Analyze this player's performance in the context of today's market conditions.
Provide a JSON diagnosis with specific, actionable recommendations.

CRITICAL RULES:
1. Do NOT just say "lower entry threshold" — that causes overtrading
2. Consider the MARKET REGIME — in downtrends, the player should trade less, not more
3. Focus on WEIGHT ADJUSTMENTS for specific indicators based on what's working/failing
4. If the market is choppy, recommend RAISING thresholds or reducing position count
5. Consider if the player's indicator mix matches the current regime

Respond with ONLY valid JSON:
{{
  "market_assessment": {{
    "regime": "trending_up|trending_down|ranging|volatile|choppy",
    "strength": "strong|moderate|weak",
    "key_sectors": ["sector names moving"],
    "recommendation": "brief market view"
  }},
  "mistakes": [
    {{
      "type": "late_entry|early_entry|whipsaw|overtrading|wrong_direction|held_too_long|exited_too_early|stop_too_tight|eod_forced_exit",
      "count": 1,
      "pnl_impact": -500.0,
      "description": "specific description with market context",
      "examples": ["trade_id"]
    }}
  ],
  "weight_recommendations": [
    {{
      "indicator": "RSI_7",
      "current_weight": 0.90,
      "recommended_weight": 0.75,
      "reason": "RSI_7 is generating false signals in this choppy regime"
    }}
  ],
  "threshold_recommendation": {{
    "entry_threshold_change": 0.0,
    "exit_threshold_change": 0.0,
    "reason": "explanation"
  }},
  "regime_advice": "What this player should do differently given the current market"
}}
"""


def build_market_context(market_data: Dict[str, pd.DataFrame],
                         indicator_data: Dict[str, pd.DataFrame],
                         trading_date: date) -> str:
    """Build a comprehensive market context string for the expert coach."""
    lines = []

    # Per-symbol summary
    sym_summaries = []
    for sym in sorted(market_data.keys()):
        df = market_data[sym]
        day_mask = df.index.date == trading_date
        day_bars = df[day_mask]
        if day_bars.empty:
            continue

        open_price = float(day_bars.iloc[0]["open"])
        close_price = float(day_bars.iloc[-1]["close"])
        high = float(day_bars["high"].max())
        low = float(day_bars["low"].min())
        change_pct = (close_price - open_price) / open_price * 100
        intraday_range = (high - low) / open_price * 100

        # Trend from lookback
        lookback = df.loc[df.index.date <= trading_date, "close"]
        trend = "flat"
        if len(lookback) >= 60:
            ema20 = lookback.ewm(span=20, min_periods=20).mean()
            ema50 = lookback.ewm(span=50, min_periods=50).mean()
            if len(ema20.dropna()) >= 1 and len(ema50.dropna()) >= 1:
                if float(ema20.iloc[-1]) > float(ema50.iloc[-1]):
                    trend = "bullish"
                elif float(ema20.iloc[-1]) < float(ema50.iloc[-1]):
                    trend = "bearish"

        # Key indicators from indicator_data
        ind_str = ""
        if sym in indicator_data:
            raw = indicator_data[sym]
            day_ind = raw.loc[raw.index.date == trading_date]
            if not day_ind.empty:
                last_row = day_ind.iloc[-1]
                rsi = last_row.get("RSI_14", None)
                adx = last_row.get("ADX_14", None)
                parts = []
                if pd.notna(rsi):
                    parts.append(f"RSI={rsi:.0f}")
                if pd.notna(adx):
                    parts.append(f"ADX={adx:.0f}")
                if parts:
                    ind_str = f" [{', '.join(parts)}]"

        sym_short = sym.replace(".NS", "")
        sym_summaries.append(
            f"  {sym_short:<12} O={open_price:>8.1f} C={close_price:>8.1f} "
            f"Chg={change_pct:>+5.1f}% Range={intraday_range:.1f}% "
            f"Trend={trend}{ind_str}"
        )

    # Market breadth
    changes = []
    for sym, df in market_data.items():
        day_bars = df[df.index.date == trading_date]
        if not day_bars.empty:
            o = float(day_bars.iloc[0]["open"])
            c = float(day_bars.iloc[-1]["close"])
            changes.append((c - o) / o * 100)

    if changes:
        advancers = sum(1 for c in changes if c > 0)
        decliners = sum(1 for c in changes if c < 0)
        avg_change = np.mean(changes)
        lines.append(f"Market breadth: {advancers} advancers, {decliners} decliners, "
                      f"avg change: {avg_change:+.2f}%")
        if avg_change > 0.5:
            lines.append("Market bias: BULLISH")
        elif avg_change < -0.5:
            lines.append("Market bias: BEARISH")
        else:
            lines.append("Market bias: NEUTRAL/MIXED")

    lines.append("\nPer-stock summary:")
    lines.extend(sym_summaries[:15])
    if len(sym_summaries) > 15:
        lines.append(f"  ... and {len(sym_summaries) - 15} more stocks")

    return "\n".join(lines)

# ───────────────────────────────────────────────────────────────────────
# RAW SI CALCULATOR — bypasses double-tanh compression
#
# The library's SignalAggregator.weighted_average applies tanh(1.5x) to
# the weighted average of already-normalised values.  Since normalisation
# itself uses tanh(0.5·z) for unbounded indicators, the double compression
# squashes realistic SI to ±0.2–0.4, making thresholds unreachable.
#
# This version returns the plain weighted average (still bounded to [-1,1]
# by the normalised inputs).
# ───────────────────────────────────────────────────────────────────────
def raw_weighted_average(normalized_df: pd.DataFrame,
                         weights: Dict[str, float]) -> pd.Series:
    """Weighted average of normalised indicators WITHOUT final tanh."""
    weighted_sum = pd.Series(0.0, index=normalized_df.index)
    total_weight = 0.0
    for name, weight in weights.items():
        if name not in normalized_df.columns or weight == 0:
            continue
        weighted_sum += normalized_df[name].fillna(0) * weight
        total_weight += abs(weight)
    if total_weight == 0:
        return pd.Series(0.0, index=normalized_df.index)
    return (weighted_sum / total_weight).clip(-1.0, 1.0)


# ───────────────────────────────────────────────────────────────────────
# 5 INTRADAY-REDESIGNED strategies (15-min optimised)  — v3
#
# Design principles:
#   1. Fewer indicators (6-10) — less noise aggregation
#   2. Fast-response indicators (RSI_7, STOCH_5_3, EMA_9, HMA_9)
#   3. All positive weights — no inverted signals
#   4. Volume confirmation (OBV / CMF / MFI)
#   5. Volatility gate (ADX / ATR / NATR)
#   6. Raw weighted average (no double tanh) ⇒ SI in ±0.3–0.7 range
#   7. Directional exit: exit long when SI drops below 0, not at a low
#      threshold.  Exit short when SI rises above 0.
#   8. Minimum holding period: 3 bars (45 min) to let trades develop
#   9. Wider ATR stops (3x) for 15m noise
# ───────────────────────────────────────────────────────────────────────
PLAYERS = {
    "PLAYER_1": {
        "dna_id": "intra_agg1",
        "label": "Aggressive",
        "original": "Fast momentum + volume",
        "weights": {
            "RSI_7": 0.90, "STOCH_5_3": 0.85, "TSI_13_25": 0.75,
            "CMO_14": 0.70, "WILLR_14": 0.65, "OBV": 0.60,
            "MFI_14": 0.55, "ADX_14": 0.50, "EMA_9": 0.45,
            "NATR_14": 0.40,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.10,   # exit LONG when SI goes negative; exit SHORT when SI > +0.10
        "min_hold_bars": 4,
    },
    "PLAYER_2": {
        "dna_id": "intra_con2",
        "label": "Conservative",
        "original": "Trend-confirm + volume",
        "weights": {
            "ADX_14": 0.90, "SUPERTREND_7_3": 0.85, "EMA_20": 0.80,
            "AROON_14": 0.70, "CMF_20": 0.65, "RSI_14": 0.60,
            "BBANDS_20_2": 0.55, "OBV": 0.50, "VWMA_10": 0.45,
            "HMA_9": 0.40, "ATR_14": 0.35,
        },
        "entry_threshold": 0.35,
        "exit_threshold": -0.15,   # most patient — holds until clear reversal
        "min_hold_bars": 5,
    },
    "PLAYER_3": {
        "dna_id": "intra_bal3",
        "label": "Balanced",
        "original": "Mean-reversion + momentum",
        "weights": {
            "RSI_14": 0.85, "BBANDS_20_2": 0.80, "STOCH_14_3": 0.75,
            "CMF_20": 0.65, "ZSCORE_20": 0.60, "MFI_20": 0.55,
            "EMA_9": 0.50, "ADX_14": 0.45, "ATR_14": 0.40,
            "TEMA_20": 0.35,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_4": {
        "dna_id": "intra_vol4",
        "label": "VolBreakout",
        "original": "Volatility breakout + trend",
        "weights": {
            "NATR_14": 0.90, "ATR_14": 0.80, "BBANDS_20_2": 0.75,
            "ADX_14": 0.70, "SUPERTREND_7_3": 0.65, "OBV": 0.60,
            "CMF_20": 0.55, "RSI_7": 0.50, "AROON_14": 0.45,
            "PSAR": 0.40,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_5": {
        "dna_id": "intra_mom5",
        "label": "Momentum",
        "original": "Pure momentum + quick exit",
        "weights": {
            "RSI_7": 0.95, "STOCH_5_3": 0.90, "TSI_13_25": 0.80,
            "CMO_14": 0.75, "UO_7_14_28": 0.65, "MFI_14": 0.60,
            "OBV": 0.55, "HMA_9": 0.50, "NATR_14": 0.40,
        },
        "entry_threshold": 0.25,
        "exit_threshold": -0.08,   # quicker exit for momentum
        "min_hold_bars": 3,
    },
}

# Ensure defaults
for p in PLAYERS.values():
    p.setdefault("entry_threshold", 0.30)
    p.setdefault("exit_threshold", -0.10)
    p.setdefault("min_hold_bars", 4)

# Top 25 NIFTY 50 stocks
NIFTY25_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", "NTPC.NS", "POWERGRID.NS",
    "ULTRACEMCO.NS", "TATASTEEL.NS", "ONGC.NS", "JSWSTEEL.NS", "TECHM.NS",
]

NO_ENTRY_AFTER = time(14, 45)   # tighter: 45 min before close
EOD_FLATTEN_TIME = time(15, 15)  # flatten 15 min before close

# ───────────────────────────────────────────────────────────────────────
# PlayerState — per-player mutable state
# ───────────────────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    player_id: str
    dna_id: str
    label: str
    weights: Dict[str, float]
    entry_threshold: float
    exit_threshold: float
    min_hold_bars: int = 3

    # Computed objects
    dna: SuperIndicatorDNA = field(default=None, repr=False)
    si: SuperIndicator = field(default=None, repr=False)
    player: Player = field(default=None, repr=False)

    # Auto-calibrated thresholds
    cal_entry: float = 0.0
    cal_exit: float = 0.0

    # Per-position bar counter {symbol: bars_held}
    bars_held: Dict[str, int] = field(default_factory=dict)

    # Tracking
    daily_trades: List[Dict] = field(default_factory=list)
    all_trades: List[Dict] = field(default_factory=list)
    daily_pnl: List[float] = field(default_factory=list)
    equity_history: List[Dict] = field(default_factory=list)
    patches_applied: List[Dict] = field(default_factory=list)
    strategy_version: str = "v1.0"


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def trade_to_dict(trade: Trade, si_value: float = 0.0) -> Dict:
    """Convert a Trade dataclass to the dict format PostMarketAnalyzer expects."""
    return {
        "trade_id": trade.trade_id,
        "symbol": trade.symbol,
        "side": trade.direction,
        "price": trade.entry_price,
        "exit_price": trade.exit_price,
        "pnl": trade.net_pnl,
        "net_pnl": trade.net_pnl,
        "si_value": trade.signal_at_entry,
        "exit_si": trade.signal_at_exit,
        "timestamp": str(trade.entry_time),
        "exit_time": str(trade.exit_time),
        "entry_reason": "signal",
        "exit_reason": trade.exit_reason,
        "indicator_snapshot": trade.indicator_snapshot or {},
        "atr": trade.atr_at_entry,
    }


# ───────────────────────────────────────────────────────────────────────
# FivePlayerSimulation — main orchestrator
# ───────────────────────────────────────────────────────────────────────

class FivePlayerSimulation:
    """60-day intraday simulation with 5 players + LLM coach."""

    def __init__(self, days: int = 60, max_symbols: int = 25,
                 use_coach: bool = True, coach_interval: int = 1,
                 player_overrides: Dict[str, Dict] = None,
                 run_label: str = "",
                 use_knowledge: bool = False):
        self.days = days
        self.max_symbols = max_symbols
        self.use_coach = use_coach
        self.coach_interval = coach_interval  # run coach every N days
        self.player_overrides = player_overrides or {}
        self.run_label = run_label
        self.use_knowledge = use_knowledge

        # Shared infra (built once in setup)
        self.universe = IndicatorUniverse()
        self.universe.load_all()
        self.calculator = IndicatorCalculator(universe=self.universe)
        self.normalizer = IndicatorNormalizer()

        self.intraday = IntradayDataFetcher(
            config=IntradayConfig(interval="15m"),
            cache=DataCache("data_cache"),
        )

        # Coach
        self.coach = PostMarketAnalyzer(config=AIConfig()) if use_coach else None

        # State
        self.players: Dict[str, PlayerState] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.indicator_data: Dict[str, pd.DataFrame] = {}
        self.trading_days: List[date] = []

    # ── Setup ─────────────────────────────────────────────────────────

    def setup(self):
        """Fetch data, compute indicators, init players."""
        symbols = NIFTY25_SYMBOLS[: self.max_symbols]

        print("=" * 70)
        title = "5-PLAYER + COACH INTRADAY SIMULATION"
        if self.run_label:
            title += f"  —  {self.run_label}"
        print(title)
        print(f"Interval: 15m | Days: {self.days} | Symbols: {len(symbols)}")
        coach_label = 'disabled'
        if self.use_coach:
            coach_label = f"Gemini LLM (every {self.coach_interval} day{'s' if self.coach_interval > 1 else ''})"
        print(f"Coach: {coach_label}")
        print("=" * 70)

        # 1. Fetch 15-min data
        print("\n[Setup] Fetching 15-min intraday data ...")
        raw_data = self.intraday.fetch_multiple(
            symbols=symbols, days=self.days,
        )
        print(f"  Fetched {len(raw_data)} / {len(symbols)} symbols")

        # 2. Normalize columns
        for sym, df in raw_data.items():
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.columns = [c.lower() for c in df.columns]
            # Drop dividends/stock splits columns if present
            for col in ["dividends", "stock splits", "capital gains"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            self.market_data[sym] = df

        # 3. Compute indicators (once, shared)
        print("[Setup] Computing indicators ...")
        for sym, df in self.market_data.items():
            try:
                raw = self.calculator.calculate_all(df)
                raw = self.calculator.rename_to_dna_names(raw)
                self.indicator_data[sym] = raw
            except Exception as e:
                logger.warning(f"Indicator calc failed for {sym}: {e}")
        print(f"  Indicators ready for {len(self.indicator_data)} symbols")

        # 4. Derive trading days
        all_dates = set()
        for df in self.market_data.values():
            all_dates.update(df.index.date)
        self.trading_days = sorted(all_dates)[-self.days:]
        print(f"  Trading days: {len(self.trading_days)} "
              f"({self.trading_days[0]} → {self.trading_days[-1]})")

        # 5. Knowledge-based strategy optimization (runs EVERY run)
        if self.use_knowledge and KNOWLEDGE_AVAILABLE and GENAI_AVAILABLE:
            is_first_run = not self.player_overrides
            label = "Designing" if is_first_run else "Redesigning (with previous run learning)"
            print(f"\n[Knowledge Coach] {label} strategies using trading knowledge base ...")
            knowledge_text = build_knowledge_text()
            ai_cfg = AIConfig()
            api_key = ai_cfg.llm.api_key
            model_name = ai_cfg.llm.model_name

            # Build per-player performance context from previous run
            def _build_prev_perf(pid: str) -> str:
                if is_first_run:
                    return "No previous data (first run)"
                ovr = self.player_overrides.get(pid, {})
                if not ovr:
                    return "No previous data"

                lines = []
                lines.append(f"Previous run results:")
                lines.append(f"  Trades: {ovr.get('total_trades', '?')} | "
                             f"Win rate: {ovr.get('win_rate', '?')}% | "
                             f"Net P&L: ${ovr.get('net_pnl', 0):,.0f} | "
                             f"Sharpe: {ovr.get('sharpe', '?')}")
                lines.append(f"  Avg win: ${ovr.get('avg_win', 0):,.0f} | "
                             f"Avg loss: ${ovr.get('avg_loss', 0):,.0f} | "
                             f"Profit factor: {ovr.get('profit_factor', '?')}")
                lines.append(f"  Exit breakdown: {ovr.get('exit_breakdown', '?')}")

                regimes = ovr.get("regime_distribution", {})
                if regimes:
                    lines.append(f"  Market regimes seen: {regimes}")

                lessons = ovr.get("coach_lessons", [])
                if lessons:
                    lines.append(f"  Coach lessons from previous run:")
                    for i, lesson in enumerate(lessons, 1):
                        lines.append(f"    {i}. {lesson}")

                # Show current weights so optimizer knows what was tried
                prev_weights = ovr.get("weights", {})
                if prev_weights:
                    w_str = ", ".join(f"{k}={v:.2f}" for k, v in
                                     sorted(prev_weights.items(), key=lambda x: -abs(x[1]))[:8])
                    lines.append(f"  Previous weights (top 8): {w_str}")
                    lines.append(f"  Previous thresholds: entry={ovr.get('entry_threshold', '?')}, "
                                 f"exit={ovr.get('exit_threshold', '?')}, "
                                 f"hold={ovr.get('min_hold_bars', '?')} bars")

                lines.append("")
                lines.append("IMPORTANT: Use the above data to IMPROVE on the previous strategy.")
                lines.append("If win rate was low, consider different indicator combinations.")
                lines.append("If profit factor < 1.0, the average winner is smaller than average loser — adjust exits.")
                lines.append("If too many EOD forced exits (E), the strategy holds too long — lower min_hold_bars or tighten exit threshold.")
                lines.append("If too many signal exits (X) with losses, the exit threshold may be too sensitive.")
                lines.append("If the coach identified specific regimes (choppy, trending), optimize for the dominant regime.")

                return "\n".join(lines)

            # Build effective config: use previous weights as starting point if available
            effective_configs = {}
            for pid, cfg in PLAYERS.items():
                ovr = self.player_overrides.get(pid, {})
                effective_configs[pid] = {
                    "label": cfg["label"],
                    "original": cfg.get("original", "Unknown"),
                    "weights": ovr.get("weights", cfg["weights"]),
                    "entry_threshold": ovr.get("entry_threshold", cfg["entry_threshold"]),
                    "exit_threshold": ovr.get("exit_threshold", cfg["exit_threshold"]),
                    "min_hold_bars": ovr.get("min_hold_bars", cfg.get("min_hold_bars", 4)),
                }

            optimized_overrides: Dict[str, Dict] = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(
                        optimize_player_strategy,
                        pid, effective_configs[pid], knowledge_text,
                        _build_prev_perf(pid),
                        api_key, model_name,
                    ): pid
                    for pid in PLAYERS
                }
                for future in as_completed(futures):
                    pid = futures[future]
                    result = future.result()
                    if result and "weights" in result:
                        strategy_name = result.get("strategy_name", "unknown")
                        rationale = result.get("rationale", "")
                        new_weights = result["weights"]
                        entry_t = result.get("entry_threshold", effective_configs[pid]["entry_threshold"])
                        exit_t = result.get("exit_threshold", effective_configs[pid]["exit_threshold"])
                        min_hb = result.get("min_hold_bars", effective_configs[pid].get("min_hold_bars", 4))

                        # Validate weights - only keep known indicators
                        valid_weights = {}
                        for ind, w in new_weights.items():
                            if isinstance(w, (int, float)):
                                valid_weights[ind] = float(w)
                        if len(valid_weights) >= 3:
                            optimized_overrides[pid] = {
                                "weights": valid_weights,
                                "entry_threshold": float(entry_t),
                                "exit_threshold": float(exit_t),
                                "min_hold_bars": int(min_hb),
                            }
                            print(f"  {pid}: {strategy_name} ({len(valid_weights)} indicators)")
                            print(f"    Rationale: {rationale[:100]}")
                            print(f"    Entry: {entry_t:.3f} | Exit: {exit_t:.3f} | Hold: {min_hb} bars")
                            top3 = sorted(valid_weights.items(), key=lambda x: -abs(x[1]))[:3]
                            print(f"    Top indicators: {', '.join(f'{k}={v:.2f}' for k,v in top3)}")
                        else:
                            print(f"  {pid}: Optimization failed (too few valid indicators)")
                    else:
                        print(f"  {pid}: Optimization returned no result")

            # Apply optimized overrides
            if optimized_overrides:
                self.player_overrides = optimized_overrides
                print(f"[Knowledge Coach] {'Designed' if is_first_run else 'Redesigned'} "
                      f"{len(optimized_overrides)}/5 players\n")

        # 6. Initialize players
        print("[Setup] Initializing 5 players ...")
        for pid, cfg in PLAYERS.items():
            self.players[pid] = self._init_player(pid, cfg)
        print(f"  Players ready: {', '.join(self.players)}")

        # 7. Auto-calibrate thresholds
        for st in self.players.values():
            self._auto_calibrate(st)
        print("[Setup] Thresholds auto-calibrated\n")

    def _init_player(self, pid: str, cfg: Dict) -> PlayerState:
        # Use overrides from previous run if available
        ovr = self.player_overrides.get(pid, {})
        weights = dict(ovr.get("weights", cfg["weights"]))
        entry_threshold = ovr.get("entry_threshold", cfg["entry_threshold"])
        exit_threshold = ovr.get("exit_threshold", cfg["exit_threshold"])
        min_hold_bars = ovr.get("min_hold_bars", cfg.get("min_hold_bars", 3))

        dna = create_dna_from_weights(weights)
        si = SuperIndicator(dna, normalizer=self.normalizer)

        portfolio = Portfolio(initial_capital=100_000.0)
        risk = RiskManager(params=RiskParameters(
            max_risk_per_trade=0.015,           # 1.5% risk per trade
            max_position_pct=0.10,              # 10% max in single position
            max_concurrent_positions=3,          # fewer simultaneous positions
            atr_stop_multiplier=5.0,            # 5x ATR stops (1.6% for NIFTY)
            min_risk_reward_ratio=1.5,
        ))
        exe = ExecutionEngine(slippage_pct=0.001, commission_per_share=0.005)
        player = Player(portfolio=portfolio, risk_manager=risk,
                        execution=exe, allow_short=True)
        player.set_dna(cfg["dna_id"], 0, 0)

        return PlayerState(
            player_id=pid,
            dna_id=cfg["dna_id"],
            label=cfg["label"],
            weights=weights,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            min_hold_bars=min_hold_bars,
            dna=dna,
            si=si,
            player=player,
        )

    def get_learned_state(self) -> Dict[str, Dict]:
        """Extract full learned state: weights, thresholds, performance, and coach lessons."""
        state = {}
        for pid, st in self.players.items():
            metrics = st.player.portfolio.get_performance_metrics()

            # Collect all coach advice and regime insights
            coach_lessons = []
            regime_counts = {}
            for patch in st.patches_applied:
                regime = patch.get("regime", "unknown")
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                advice = patch.get("advice", "")
                if advice:
                    coach_lessons.append(advice)

            # Analyze trade patterns
            win_trades = [t for t in st.all_trades if (t.get("pnl") or 0) > 0]
            loss_trades = [t for t in st.all_trades if (t.get("pnl") or 0) <= 0]
            avg_win = sum(t.get("pnl", 0) for t in win_trades) / max(1, len(win_trades))
            avg_loss = sum(t.get("pnl", 0) for t in loss_trades) / max(1, len(loss_trades))

            # Exit reason breakdown
            stop_exits = sum(1 for t in st.all_trades if t.get("exit_reason") == "stop_loss")
            signal_exits = sum(1 for t in st.all_trades if t.get("exit_reason") == "signal")
            eod_exits = sum(1 for t in st.all_trades if t.get("exit_reason") == "eod_flatten")

            state[pid] = {
                "weights": dict(st.weights),
                "entry_threshold": st.entry_threshold,
                "exit_threshold": st.exit_threshold,
                "min_hold_bars": st.min_hold_bars,
                # Performance data
                "total_trades": metrics["total_trades"],
                "win_rate": round(metrics["win_rate"] * 100, 1),
                "net_pnl": round(metrics["net_profit"], 2),
                "sharpe": round(metrics["sharpe_ratio"], 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
                "exit_breakdown": f"S{stop_exits}/X{signal_exits}/E{eod_exits}",
                # Coach learning
                "regime_distribution": regime_counts,
                "coach_lessons": coach_lessons[-5:],  # last 5 lessons
                "patches_applied": len(st.patches_applied),
            }
        return state

    def get_run_summary(self) -> Dict:
        """Get a structured summary of this run's results."""
        summary = {"players": {}, "team_pnl": 0.0}
        for pid, st in self.players.items():
            metrics = st.player.portfolio.get_performance_metrics()
            pnl = metrics["net_profit"]
            summary["players"][pid] = {
                "label": st.label,
                "trades": metrics["total_trades"],
                "win_rate": round(metrics["win_rate"] * 100, 1),
                "net_pnl": round(pnl, 2),
                "sharpe": round(metrics["sharpe_ratio"], 2),
                "max_dd": round(metrics["max_drawdown"] * 100, 1),
                "patches": len(st.patches_applied),
            }
            summary["team_pnl"] += pnl
        summary["team_pnl"] = round(summary["team_pnl"], 2)
        summary["team_return"] = round(summary["team_pnl"] / 500_000 * 100, 2)
        return summary

    def _auto_calibrate(self, st: PlayerState):
        """Auto-calibrate entry/exit thresholds using RAW SI distribution.

        Uses the raw weighted average (no double tanh) so SI values are
        wider and thresholds are achievable.
        """
        all_si: List[float] = []
        for sym, raw in self.indicator_data.items():
            active = [i for i in st.dna.get_active_indicators()
                      if i in raw.columns]
            if not active:
                continue
            try:
                price_s = self.market_data[sym]["close"]
                norm = self.normalizer.normalize_all(
                    raw[active], price_series=price_s,
                )
                if norm.empty:
                    continue
                # Use RAW weighted average — no final tanh
                si_vals = raw_weighted_average(norm, st.weights).dropna()
                all_si.extend(si_vals.values[50:].tolist())
            except Exception:
                continue

        if not all_si:
            st.cal_entry = st.entry_threshold
            st.cal_exit = st.exit_threshold
            return

        arr = np.array(all_si)
        abs_arr = np.abs(arr)
        p50 = float(np.percentile(abs_arr, 50))
        p75 = float(np.percentile(abs_arr, 75))
        p90 = float(np.percentile(abs_arr, 90))
        p95 = float(np.percentile(abs_arr, 95))

        print(f"    {st.player_id} SI dist: p50={p50:.3f} p75={p75:.3f} "
              f"p90={p90:.3f} p95={p95:.3f}  "
              f"mean={np.mean(arr):.3f} std={np.std(arr):.3f}")

        if st.entry_threshold <= p95:
            st.cal_entry = st.entry_threshold
        else:
            st.cal_entry = max(p90, 0.10)

        st.cal_exit = st.exit_threshold

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self):
        self.setup()

        for day_idx, trading_date in enumerate(self.trading_days):
            print(f"\n{'=' * 70}")
            print(f"DAY {day_idx + 1}/{len(self.trading_days)} — "
                  f"{trading_date.strftime('%a %Y-%m-%d')}")
            print(f"{'=' * 70}")

            # Reset daily state
            for st in self.players.values():
                st.daily_trades = []
                st.bars_held = {}  # reset holding counters (positions flattened EOD)

            # INTRADAY: each player trades
            self._simulate_day(trading_date)

            # EOD FLATTEN
            self._eod_flatten(trading_date)

            # Print daily summary
            self._print_daily_summary(day_idx, trading_date)

            # POST-MARKET: Coach analysis + patches
            if self.use_coach and day_idx > 0 and (day_idx % self.coach_interval == 0):
                total_day_trades = sum(
                    len(s.daily_trades) for s in self.players.values()
                )
                if total_day_trades >= 3:
                    self._coach_cycle(trading_date)
                else:
                    print(f"  [Coach] Skipped (only {total_day_trades} trades)")
            elif self.use_coach and day_idx > 0 and (day_idx % self.coach_interval != 0):
                print(f"  [Coach] Skipped (next coach day: {self.coach_interval - (day_idx % self.coach_interval)} day(s))")

        # Final report
        self._final_report()

    # ── Intraday trading ──────────────────────────────────────────────

    def _simulate_day(self, trading_date: date):
        """Run all 5 players through one trading day.

        Key changes in v3:
        - Uses raw_weighted_average (no final tanh) for wider SI range
        - Directional exit: exit long when SI < exit_thresh (default 0.0)
        - Minimum holding period: skip exits until min_hold_bars elapsed
        - Persistence filter: prev bar must confirm direction at 70% threshold
        """
        # Pre-compute lookback data and price series once per symbol
        sym_lookbacks: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}
        for sym in self.indicator_data:
            df = self.market_data[sym]
            raw = self.indicator_data[sym]
            day_mask = df.index.date == trading_date
            day_bars = df[day_mask]
            if day_bars.empty:
                continue
            lookback_raw = raw.loc[raw.index.date <= trading_date]
            price_s = df.loc[df.index.date <= trading_date, "close"]
            sym_lookbacks[sym] = (lookback_raw, price_s, day_bars)

        # Cache normalized data per (symbol, active_cols_key)
        norm_cache: Dict[str, pd.DataFrame] = {}

        for sym, (lookback_raw, price_s, day_bars) in sym_lookbacks.items():
            raw = self.indicator_data[sym]

            for pid, st in self.players.items():
                active = [i for i in st.dna.get_active_indicators()
                          if i in raw.columns]
                if not active:
                    continue

                cache_key = f"{sym}|{'|'.join(sorted(active))}"
                if cache_key in norm_cache:
                    norm = norm_cache[cache_key]
                else:
                    try:
                        norm = self.normalizer.normalize_all(
                            lookback_raw[active], price_series=price_s,
                        )
                        if norm.empty:
                            continue
                        norm_cache[cache_key] = norm
                    except Exception:
                        continue

                try:
                    # ── RAW SI (no double tanh) ──
                    si_series = raw_weighted_average(norm, st.weights).fillna(0.0)
                except Exception:
                    continue

                # ── TREND FILTER (daily bias) ──
                # Computed once per day per symbol using prior close data.
                # Price above EMA50 + EMA50 sloping up → bullish bias
                # Price below EMA50 + EMA50 sloping down → bearish bias
                # Mixed → allow both directions
                close_series = price_s  # full lookback close prices
                trend_bull = True
                trend_bear = True
                if len(close_series) >= 60:
                    ema50 = close_series.ewm(span=50, min_periods=50).mean()
                    valid_ema = ema50.dropna()
                    if len(valid_ema) >= 10:
                        ema_now = valid_ema.iloc[-1]
                        ema_prev = valid_ema.iloc[-10]
                        price_now = close_series.iloc[-1]
                        slope_up = ema_now > ema_prev
                        price_above = price_now > ema_now
                        if slope_up and price_above:
                            trend_bear = False   # strong uptrend: no shorts
                        elif (not slope_up) and (not price_above):
                            trend_bull = False   # strong downtrend: no longs

                prev_si = 0.0
                entry_thresh = st.cal_entry
                exit_thresh = st.cal_exit

                for bar_ts in day_bars.index:
                    bar = day_bars.loc[bar_ts]

                    if bar_ts not in si_series.index:
                        continue
                    si_val = float(si_series.loc[bar_ts])

                    # ── Position state ──
                    pos = st.player.portfolio.get_position(sym)
                    pos_state = PositionState.FLAT
                    if pos:
                        pos_state = (PositionState.LONG
                                     if pos.direction == "LONG"
                                     else PositionState.SHORT)

                    # ── Track holding bars ──
                    if pos_state != PositionState.FLAT:
                        st.bars_held[sym] = st.bars_held.get(sym, 0) + 1
                    else:
                        st.bars_held.pop(sym, None)

                    held = st.bars_held.get(sym, 0)

                    # ── Signal logic ──
                    signal = SignalType.HOLD
                    bar_time = bar_ts.time() if hasattr(bar_ts, "time") else None

                    if pos_state == PositionState.FLAT:
                        # No entries after cutoff
                        if bar_time and bar_time > NO_ENTRY_AFTER:
                            signal = SignalType.HOLD
                        # Persistent bullish + trend filter
                        elif (si_val > entry_thresh
                              and prev_si > entry_thresh * 0.7
                              and trend_bull):
                            signal = SignalType.LONG_ENTRY
                        # Persistent bearish + trend filter
                        elif (si_val < -entry_thresh
                              and prev_si < -entry_thresh * 0.7
                              and trend_bear):
                            signal = SignalType.SHORT_ENTRY

                    elif pos_state == PositionState.LONG:
                        # Exit long when SI drops below exit_threshold
                        if held >= st.min_hold_bars and si_val < exit_thresh:
                            signal = SignalType.LONG_EXIT

                    elif pos_state == PositionState.SHORT:
                        if held >= st.min_hold_bars and si_val > abs(exit_thresh):
                            signal = SignalType.SHORT_EXIT

                    prev_si = si_val

                    # ── ATR ──
                    atr_val = 1.0
                    if bar_ts in raw.index and "ATR_14" in raw.columns:
                        a = raw.loc[bar_ts, "ATR_14"]
                        if pd.notna(a) and a > 0:
                            atr_val = float(a)
                        else:
                            atr_val = float(bar["close"]) * 0.02
                    else:
                        atr_val = float(bar["close"]) * 0.02

                    # ── Execute ──
                    trade = st.player.process_signal(
                        symbol=sym,
                        signal=signal,
                        current_price=float(bar["close"]),
                        timestamp=bar_ts,
                        high=float(bar["high"]),
                        low=float(bar["low"]),
                        atr=atr_val,
                        si_value=si_val,
                    )
                    if trade:
                        td = trade_to_dict(trade, si_val)
                        st.daily_trades.append(td)
                        st.all_trades.append(td)
                        # Reset bar counter on exit
                        if signal in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT):
                            st.bars_held.pop(sym, None)

    # ── EOD flatten ───────────────────────────────────────────────────

    def _eod_flatten(self, trading_date: date):
        for st in self.players.values():
            if st.player.portfolio.num_positions == 0:
                continue
            prices = {}
            for sym in list(st.player.portfolio.positions.keys()):
                df = self.market_data.get(sym)
                if df is not None:
                    day_bars = df[df.index.date == trading_date]
                    if not day_bars.empty:
                        prices[sym] = float(day_bars.iloc[-1]["close"])
            # Build a tz-aware timestamp matching market data
            flatten_ts = pd.Timestamp(
                datetime.combine(trading_date, EOD_FLATTEN_TIME),
                tz="Asia/Kolkata",
            )
            trades = st.player.close_all_positions(
                timestamp=flatten_ts, prices=prices, reason="eod_flatten",
            )
            for t in trades:
                td = trade_to_dict(t, 0.0)
                st.daily_trades.append(td)
                st.all_trades.append(td)

    # ── Coach cycle ───────────────────────────────────────────────────

    def _expert_analyze_player(self, pid: str, st: PlayerState,
                              trading_date: date,
                              market_context: str) -> Tuple[str, Optional[Dict]]:
        """Analyze a player using the Expert Coach with full market context."""
        if not st.daily_trades:
            return pid, None

        if not GENAI_AVAILABLE or not self.coach:
            return pid, None

        # Build trade summary
        day_pnl = sum(t.get("pnl", 0) or 0 for t in st.daily_trades)
        wins = sum(1 for t in st.daily_trades if (t.get("pnl") or 0) > 0)
        losses = len(st.daily_trades) - wins
        trades_summary = (
            f"Trades: {len(st.daily_trades)} | Wins: {wins} | Losses: {losses} | "
            f"Day P&L: ${day_pnl:,.0f}"
        )

        # Format trade details
        trade_lines = []
        for t in st.daily_trades:
            pnl = t.get("pnl", 0) or 0
            result = "WIN" if pnl > 0 else "LOSS"
            line = (f"  {t.get('trade_id', '?')} | {t.get('symbol', '?')} | "
                    f"{t.get('side', '?')} | ${t.get('price', 0):,.2f}")
            if t.get("exit_price"):
                line += f" → ${t['exit_price']:,.2f}"
            line += f" | SI={t.get('si_value', 0):.3f} | P&L: ${pnl:,.0f} {result}"
            line += f" | Exit: {t.get('exit_reason', '?')}"
            trade_lines.append(line)
        trades_detail = "\n".join(trade_lines) if trade_lines else "No trades"

        # Format weights
        weights_lines = [f"  {k}: {v:.3f}" for k, v in sorted(st.weights.items(), key=lambda x: -x[1])]
        weights_str = "\n".join(weights_lines)

        knowledge = build_knowledge_text() if KNOWLEDGE_AVAILABLE else "Not available"

        prompt = EXPERT_COACH_PROMPT.format(
            knowledge_text=knowledge,
            analysis_date=trading_date.isoformat(),
            market_context=market_context,
            player_id=pid,
            player_label=st.label,
            strategy_desc=PLAYERS.get(pid, {}).get("original", "Unknown"),
            active_indicators=", ".join(sorted(st.weights.keys())),
            entry_threshold=st.cal_entry,
            exit_threshold=st.cal_exit,
            min_hold_bars=st.min_hold_bars,
            weights_str=weights_str,
            trades_summary=trades_summary,
            trades_detail=trades_detail,
        )

        try:
            client = genai.Client(api_key=self.coach.config.llm.api_key)
            import signal as _sig

            def _timeout_handler(signum, frame):
                raise TimeoutError("Gemini API call timed out after 90s")
            old_handler = _sig.signal(_sig.SIGALRM, _timeout_handler)
            _sig.alarm(90)  # 90 second timeout
            try:
                response = client.models.generate_content(
                    model=self.coach.config.llm.model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "max_output_tokens": 2048,
                    },
                )
                raw = response.text.strip()
            finally:
                _sig.alarm(0)
                _sig.signal(_sig.SIGALRM, old_handler)

            # Clean JSON
            import re
            if "```" in raw:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if match:
                    raw = match.group(1)
            start_idx = raw.find('{')
            end_idx = raw.rfind('}')
            if start_idx != -1 and end_idx != -1:
                raw = raw[start_idx:end_idx+1]

            # Fix common JSON issues from LLM: trailing commas, comments
            raw = re.sub(r',\s*}', '}', raw)
            raw = re.sub(r',\s*]', ']', raw)
            raw = re.sub(r'//[^\n]*', '', raw)

            data = json.loads(raw)
            return pid, data

        except Exception as e:
            logger.warning(f"  Expert coach failed for {pid}: {e}")
            # Fallback to rule-based
            try:
                rule_analyzer = PostMarketAnalyzer.__new__(PostMarketAnalyzer)
                rule_analyzer.config = self.coach.config
                rule_analyzer._client = None
                rule_analyzer._indicator_scores = []
                diagnosis = rule_analyzer.analyze(
                    trades=st.daily_trades,
                    analysis_date=trading_date,
                    strategy_version=st.strategy_version,
                )
                return pid, diagnosis.to_dict()
            except Exception:
                return pid, None

    def _coach_cycle(self, trading_date: date):
        """Expert Coach: analyze market + all players, then apply patches."""
        # Build market context ONCE (shared across all players)
        market_context = build_market_context(
            self.market_data, self.indicator_data, trading_date
        )

        active_players = [
            (pid, st) for pid, st in self.players.items()
            if st.daily_trades
        ]
        if not active_players:
            return

        print(f"\n  [Expert Coach] Post-market analysis for {len(active_players)} players ...")

        # Run expert analysis sequentially to avoid Gemini rate limits
        diagnoses: Dict[str, Dict] = {}
        for pid, st in active_players:
            pid_out, result = self._expert_analyze_player(pid, st, trading_date, market_context)
            if result:
                diagnoses[pid_out] = result
            _time.sleep(1)  # Rate limit buffer

        # Apply patches based on expert diagnosis
        for pid, diag in diagnoses.items():
            st = self.players[pid]

            # Print market assessment if available
            market_assess = diag.get("market_assessment", {})
            regime = market_assess.get("regime", "unknown")
            regime_advice = diag.get("regime_advice", "")

            mistakes = diag.get("mistakes", [])
            weight_recs = diag.get("weight_recommendations", [])
            thresh_rec = diag.get("threshold_recommendation", {})

            n_mistakes = sum(m.get("count", 1) for m in mistakes)
            print(f"    {pid}: {len(st.daily_trades)} trades, {n_mistakes} mistakes, "
                  f"regime={regime}, {len(weight_recs)} weight recs")

            if regime_advice:
                print(f"    {pid}: Coach says: {regime_advice[:80]}")

            # 1. Apply weight recommendations from expert coach (max 3 per day)
            applied_weight_changes = 0
            for rec in weight_recs[:3]:
                ind = rec.get("indicator", "")
                new_w = rec.get("recommended_weight")
                if ind in st.weights and new_w is not None:
                    old_w = st.weights[ind]
                    # Cap change at ±15% of current weight
                    max_delta = abs(old_w) * 0.15
                    delta = new_w - old_w
                    clamped = max(-max_delta, min(max_delta, delta))
                    st.weights[ind] = round(old_w + clamped, 4)
                    applied_weight_changes += 1

            # 2. Apply threshold recommendation (bounded ±10%)
            entry_change = thresh_rec.get("entry_threshold_change", 0)
            exit_change = thresh_rec.get("exit_threshold_change", 0)
            if entry_change != 0:
                max_entry_delta = st.entry_threshold * 0.10
                clamped_entry = max(-max_entry_delta, min(max_entry_delta, entry_change))
                st.entry_threshold = max(0.10, st.entry_threshold + clamped_entry)

            if exit_change != 0:
                max_exit_delta = abs(st.exit_threshold) * 0.10 + 0.01  # small floor
                clamped_exit = max(-max_exit_delta, min(max_exit_delta, exit_change))
                st.exit_threshold = st.exit_threshold + clamped_exit

            # 3. Rebuild DNA and recalibrate if weights changed
            if applied_weight_changes > 0 or entry_change != 0 or exit_change != 0:
                st.dna = create_dna_from_weights(st.weights)
                st.si = SuperIndicator(st.dna, normalizer=self.normalizer)
                self._auto_calibrate(st)

                st.patches_applied.append({
                    "day": trading_date.isoformat(),
                    "regime": regime,
                    "weight_changes": applied_weight_changes,
                    "entry_change": round(entry_change, 4),
                    "exit_change": round(exit_change, 4),
                    "advice": regime_advice[:100],
                })
                st.strategy_version = f"v1.{len(st.patches_applied)}"
                print(f"    {pid}: Patch applied — {applied_weight_changes} weight changes, "
                      f"regime={regime}")

    # ── Reporting ─────────────────────────────────────────────────────

    def _print_daily_summary(self, day_idx: int, trading_date: date):
        hdr = (f"  {'Player':<12} {'Label':<12} {'Trades':<7} {'W/L':<7} "
               f"{'Exits':>15} {'Day P&L':>10} {'Cum P&L':>10} {'Equity':>12}")
        print(hdr)
        print(f"  {'-' * 88}")

        for pid, st in self.players.items():
            day_pnl = sum(t.get("pnl", 0) or 0 for t in st.daily_trades)
            st.daily_pnl.append(day_pnl)

            equity = st.player.portfolio.get_equity()
            cum_pnl = equity - 100_000
            wins = sum(1 for t in st.daily_trades
                       if (t.get("pnl") or 0) > 0)
            losses = len(st.daily_trades) - wins

            # Exit reason breakdown
            stop_exits = sum(1 for t in st.daily_trades
                             if t.get("exit_reason") == "stop_loss")
            signal_exits = sum(1 for t in st.daily_trades
                               if t.get("exit_reason") == "signal")
            eod_exits = sum(1 for t in st.daily_trades
                            if t.get("exit_reason") == "eod_flatten")
            exit_str = f"S{stop_exits}/X{signal_exits}/E{eod_exits}"

            st.equity_history.append({
                "day": day_idx,
                "date": trading_date.isoformat(),
                "equity": equity,
                "day_pnl": day_pnl,
                "trades": len(st.daily_trades),
                "version": st.strategy_version,
            })

            print(f"  {pid:<12} {st.label:<12} {len(st.daily_trades):<7} "
                  f"{wins}/{losses:<5} "
                  f"{exit_str:>15} "
                  f"${day_pnl:>9,.0f} ${cum_pnl:>9,.0f} "
                  f"${equity:>10,.0f}")

    def _final_report(self):
        print("\n" + "=" * 70)
        print("FINAL SIMULATION RESULTS")
        print("=" * 70)

        results = {
            "simulation": {
                "type": "5_player_60day_intraday",
                "interval": "15m",
                "days": len(self.trading_days),
                "symbols": len(self.market_data),
                "coach": self.use_coach,
                "timestamp": datetime.now().isoformat(),
            },
            "players": {},
        }

        hdr = (f"  {'Player':<12} {'Label':<12} {'Trades':<7} {'WR%':<7} "
               f"{'Net P&L':>10} {'Sharpe':>7} {'MaxDD':>7} {'Patches':>7}")
        print(hdr)
        print(f"  {'-' * 72}")

        team_pnl = 0.0
        for pid, st in self.players.items():
            metrics = st.player.portfolio.get_performance_metrics()

            total = metrics["total_trades"]
            wr = metrics["win_rate"] * 100
            pnl = metrics["net_profit"]
            sharpe = metrics["sharpe_ratio"]
            dd = metrics["max_drawdown"] * 100
            patches = len(st.patches_applied)
            team_pnl += pnl

            print(f"  {pid:<12} {st.label:<12} {total:<7} {wr:<6.1f}% "
                  f"${pnl:>9,.0f} {sharpe:>7.2f} {dd:>6.1f}% {patches:>7}")

            results["players"][pid] = {
                "dna_id": st.dna_id,
                "label": st.label,
                "final_version": st.strategy_version,
                "patches_applied": patches,
                "total_trades": total,
                "win_rate": round(wr, 2),
                "net_profit": round(pnl, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown_pct": round(dd, 2),
                "equity_history": st.equity_history,
                "patches": st.patches_applied,
            }

        print(f"\n  Team total P&L: ${team_pnl:,.0f}")
        print(f"  Capital deployed: $500,000 (5 x $100K)")
        print(f"  Team return: {team_pnl / 500_000 * 100:.2f}%")

        out_path = Path("simulation_5player_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")


# ───────────────────────────────────────────────────────────────────────
# CLI entry
# ───────────────────────────────────────────────────────────────────────

def print_multi_run_summary(all_summaries: List[Dict]):
    """Print a comparison table across all runs."""
    n = len(all_summaries)
    print("\n" + "=" * 90)
    print("MULTI-RUN COMPARISON  —  Coach Learning Across Runs")
    print("=" * 90)

    # Team-level comparison
    print(f"\n{'Run':<8} {'Team P&L':>12} {'Return':>9} {'P1':>10} {'P2':>10} "
          f"{'P3':>10} {'P4':>10} {'P5':>10}")
    print(f"{'─' * 8} {'─' * 12} {'─' * 9} {'─' * 10} {'─' * 10} "
          f"{'─' * 10} {'─' * 10} {'─' * 10}")
    for i, s in enumerate(all_summaries):
        p = s["players"]
        print(f"Run {i+1:<3} ${s['team_pnl']:>10,.0f} {s['team_return']:>7.2f}% "
              f"${p['PLAYER_1']['net_pnl']:>8,.0f} ${p['PLAYER_2']['net_pnl']:>8,.0f} "
              f"${p['PLAYER_3']['net_pnl']:>8,.0f} ${p['PLAYER_4']['net_pnl']:>8,.0f} "
              f"${p['PLAYER_5']['net_pnl']:>8,.0f}")

    # Improvement over runs
    print(f"\n{'─' * 90}")
    print("IMPROVEMENT OVER RUNS")
    print(f"{'─' * 90}")
    baseline = all_summaries[0]
    for i in range(1, n):
        s = all_summaries[i]
        team_delta = s["team_pnl"] - baseline["team_pnl"]
        pct_improve = (team_delta / abs(baseline["team_pnl"])) * 100 if baseline["team_pnl"] != 0 else 0
        # vs previous run
        prev = all_summaries[i - 1]
        vs_prev = s["team_pnl"] - prev["team_pnl"]
        print(f"  Run {i+1} vs Run 1:  ${team_delta:>+10,.0f} ({pct_improve:>+.1f}%)   |   "
              f"vs Run {i}: ${vs_prev:>+10,.0f}")

    # Win rate progression
    print(f"\n{'─' * 90}")
    print("WIN RATE PROGRESSION (%)")
    print(f"{'─' * 90}")
    print(f"{'Run':<8} {'P1':>8} {'P2':>8} {'P3':>8} {'P4':>8} {'P5':>8} {'Avg':>8}")
    print(f"{'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for i, s in enumerate(all_summaries):
        p = s["players"]
        wrs = [p[f"PLAYER_{j}"]["win_rate"] for j in range(1, 6)]
        avg_wr = sum(wrs) / len(wrs)
        print(f"Run {i+1:<3} {wrs[0]:>7.1f}% {wrs[1]:>7.1f}% {wrs[2]:>7.1f}% "
              f"{wrs[3]:>7.1f}% {wrs[4]:>7.1f}% {avg_wr:>7.1f}%")

    # Sharpe progression
    print(f"\n{'─' * 90}")
    print("SHARPE RATIO PROGRESSION")
    print(f"{'─' * 90}")
    print(f"{'Run':<8} {'P1':>8} {'P2':>8} {'P3':>8} {'P4':>8} {'P5':>8} {'Avg':>8}")
    print(f"{'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for i, s in enumerate(all_summaries):
        p = s["players"]
        sharpes = [p[f"PLAYER_{j}"]["sharpe"] for j in range(1, 6)]
        avg_sh = sum(sharpes) / len(sharpes)
        print(f"Run {i+1:<3} {sharpes[0]:>8.2f} {sharpes[1]:>8.2f} {sharpes[2]:>8.2f} "
              f"{sharpes[3]:>8.2f} {sharpes[4]:>8.2f} {avg_sh:>8.2f}")

    # Trade count progression
    print(f"\n{'─' * 90}")
    print("TRADE COUNT PROGRESSION")
    print(f"{'─' * 90}")
    print(f"{'Run':<8} {'P1':>8} {'P2':>8} {'P3':>8} {'P4':>8} {'P5':>8} {'Total':>8}")
    print(f"{'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for i, s in enumerate(all_summaries):
        p = s["players"]
        trades = [p[f"PLAYER_{j}"]["trades"] for j in range(1, 6)]
        print(f"Run {i+1:<3} {trades[0]:>8} {trades[1]:>8} {trades[2]:>8} "
              f"{trades[3]:>8} {trades[4]:>8} {sum(trades):>8}")

    # Best and worst
    best_idx = max(range(n), key=lambda i: all_summaries[i]["team_pnl"])
    worst_idx = min(range(n), key=lambda i: all_summaries[i]["team_pnl"])
    print(f"\n{'─' * 90}")
    print(f"BEST RUN:  Run {best_idx+1}  —  ${all_summaries[best_idx]['team_pnl']:>,.0f} "
          f"({all_summaries[best_idx]['team_return']:.2f}%)")
    print(f"WORST RUN: Run {worst_idx+1}  —  ${all_summaries[worst_idx]['team_pnl']:>,.0f} "
          f"({all_summaries[worst_idx]['team_return']:.2f}%)")
    total_across = sum(s["team_pnl"] for s in all_summaries)
    print(f"TOTAL ACROSS ALL RUNS: ${total_across:>,.0f} "
          f"({total_across / 500_000 * 100:.2f}% of capital)")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="5-Player + Coach Intraday Simulation",
    )
    parser.add_argument("--days", type=int, default=60,
                        help="Trading days to simulate (max 60)")
    parser.add_argument("--symbols", type=int, default=25,
                        help="Number of NIFTY stocks")
    parser.add_argument("--no-coach", action="store_true",
                        help="Disable LLM coach")
    parser.add_argument("--coach-interval", type=int, default=3,
                        help="Run coach every N days (default: 3)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of consecutive runs (coach learning carries over)")
    parser.add_argument("--knowledge", action="store_true",
                        help="Enable knowledge-based strategy optimization (uses trading books/research)")
    args = parser.parse_args()

    num_runs = max(1, args.runs)

    if num_runs == 1:
        # Single run (original behavior)
        sim = FivePlayerSimulation(
            days=min(args.days, 60),
            max_symbols=args.symbols,
            use_coach=not args.no_coach,
            coach_interval=args.coach_interval,
            use_knowledge=args.knowledge,
        )
        sim.run()
        return

    # Multi-run: carry forward coach learning
    all_summaries: List[Dict] = []
    carry_over: Dict[str, Dict] = {}

    for run_idx in range(num_runs):
        print("\n" + "#" * 90)
        print(f"###  STARTING RUN {run_idx + 1} OF {num_runs}  ###")
        if carry_over:
            print(f"###  Carrying forward learned weights from Run {run_idx}  ###")
        print("#" * 90)

        sim = FivePlayerSimulation(
            days=min(args.days, 60),
            max_symbols=args.symbols,
            use_coach=not args.no_coach,
            coach_interval=args.coach_interval,
            player_overrides=carry_over,
            run_label=f"RUN {run_idx + 1}/{num_runs}",
            use_knowledge=args.knowledge,
        )
        sim.run()

        # Collect summary
        summary = sim.get_run_summary()
        summary["run"] = run_idx + 1
        all_summaries.append(summary)

        # Extract learned state for next run
        carry_over = sim.get_learned_state()

        # Save per-run results
        out_path = Path(f"simulation_run{run_idx + 1}_results.json")
        with open(out_path, "w") as f:
            json.dump({
                "run": run_idx + 1,
                "summary": summary,
                "learned_state": {pid: {k: v for k, v in st.items() if k != "weights"}
                                  for pid, st in carry_over.items()},
                "weight_snapshot": carry_over,
            }, f, indent=2, default=str)

    # Print cross-run comparison
    print_multi_run_summary(all_summaries)

    # Save all summaries
    with open("simulation_all_runs_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nAll run summaries saved to simulation_all_runs_summary.json")


if __name__ == "__main__":
    main()
