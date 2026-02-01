"""
Phase 2 — Coach v1: Post-Market Analyzer.

Reads yesterday's (or any day's) trades from the paper trading ledger
and produces a structured JSON diagnosis that the Player can consume.

Output schema:
{
    "analysis_date": "2026-01-28",
    "strategy_version": "v1.0",
    "summary": { ... },
    "winning_patterns": [ ... ],
    "losing_patterns": [ ... ],
    "mistakes": [ { type, count, pnl_impact, examples: [trade_id...] } ],
    "opportunities": [ { hypothesis, expected_mechanism, suggested_change_type } ],
    "news_summary": [ { event, impact_level, affected_symbols, timestamp } ],
    "indicator_diagnosis": { ... },
    "regime_diagnosis": { ... },
    "meta": { ... }
}

If JSON is invalid → regenerate (up to 3 retries).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from ..ai_config import AIConfig, DEFAULT_AI_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema dataclasses — every field is typed so validation is structural
# ---------------------------------------------------------------------------

class MistakeType(str, Enum):
    """Taxonomy of trading mistakes."""
    LATE_ENTRY = "late_entry"
    EARLY_ENTRY = "early_entry"
    WHIPSAW = "whipsaw"
    OVERTRADING = "overtrading"
    NEWS_SHOCK = "news_shock"
    STOP_TOO_TIGHT = "stop_too_tight"
    STOP_TOO_WIDE = "stop_too_wide"
    WRONG_DIRECTION = "wrong_direction"
    HELD_TOO_LONG = "held_too_long"
    EXITED_TOO_EARLY = "exited_too_early"
    INDICATOR_DISAGREEMENT = "indicator_disagreement"
    LOW_VOLATILITY_TRAP = "low_volatility_trap"
    HIGH_VOLATILITY_LOSS = "high_volatility_loss"
    EOD_FORCED_EXIT = "eod_forced_exit"
    COOLDOWN_MISSED_OPPORTUNITY = "cooldown_missed_opportunity"


@dataclass
class Mistake:
    """A single mistake classification."""
    type: str
    count: int
    pnl_impact: float
    description: str
    examples: List[str] = field(default_factory=list)  # trade_ids

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Opportunity:
    """A hypothesis for improvement."""
    hypothesis: str
    expected_mechanism: str
    suggested_change_type: str  # "weight_adjustment", "threshold_change", "risk_param", "filter_add"
    confidence: float = 0.0  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NewsEvent:
    """A market event relevant to the trading day."""
    event: str
    impact_level: str  # "critical", "high", "medium", "low"
    affected_symbols: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndicatorScore:
    """Per-indicator analysis result."""
    name: str
    current_weight: float
    avg_value_winners: float  # avg indicator value on winning trades
    avg_value_losers: float   # avg indicator value on losing trades
    win_correlation: float    # positive = helps, negative = hurts
    trade_count: int          # how many trades had this indicator
    verdict: str              # "helping", "hurting", "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WeightRecommendation:
    """Recommended weight change for an indicator."""
    indicator: str
    current_weight: float
    recommended_weight: float
    change: float             # recommended_weight - current_weight
    reason: str
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoachDiagnosis:
    """
    Complete structured output from the Coach.
    This is the contract between Coach and Player.
    """
    analysis_date: str
    strategy_version: str

    # Summary stats
    summary: Dict[str, Any]

    # Pattern analysis
    winning_patterns: List[Dict[str, Any]]
    losing_patterns: List[Dict[str, Any]]

    # Mistake taxonomy
    mistakes: List[Mistake]

    # Improvement hypotheses
    opportunities: List[Opportunity]

    # News context
    news_summary: List[NewsEvent]

    # Indicator-level diagnosis
    indicator_diagnosis: Dict[str, Any]

    # Per-indicator scores and weight recommendations
    indicator_scores: List[IndicatorScore] = field(default_factory=list)
    weight_recommendations: List[WeightRecommendation] = field(default_factory=list)

    # Regime diagnosis
    regime_diagnosis: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_date": self.analysis_date,
            "strategy_version": self.strategy_version,
            "summary": self.summary,
            "winning_patterns": self.winning_patterns,
            "losing_patterns": self.losing_patterns,
            "mistakes": [m.to_dict() for m in self.mistakes],
            "opportunities": [o.to_dict() for o in self.opportunities],
            "news_summary": [n.to_dict() for n in self.news_summary],
            "indicator_diagnosis": self.indicator_diagnosis,
            "indicator_scores": [s.to_dict() for s in self.indicator_scores],
            "weight_recommendations": [w.to_dict() for w in self.weight_recommendations],
            "regime_diagnosis": self.regime_diagnosis,
            "meta": self.meta,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def validate_dict(cls, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a dictionary matches the CoachDiagnosis schema.
        Returns (is_valid, list_of_errors).
        Only requires 'mistakes' — other fields are optional.
        """
        errors = []
        # Only mistakes is strictly required from LLM
        required_keys = ["analysis_date", "strategy_version", "summary"]
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")

        # Validate mistakes structure (must be present)
        if "mistakes" not in data:
            errors.append("Missing required key: mistakes")
        for i, m in enumerate(data.get("mistakes", [])):
            for req in ("type", "count", "pnl_impact"):
                if req not in m:
                    errors.append(f"mistakes[{i}] missing '{req}'")
            # examples is optional, default to []
            if "examples" in m and not isinstance(m["examples"], list):
                errors.append(f"mistakes[{i}].examples must be a list")

        # Validate opportunities structure (optional)
        for i, o in enumerate(data.get("opportunities", [])):
            for req in ("hypothesis",):
                if req not in o:
                    errors.append(f"opportunities[{i}] missing '{req}'")

        return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# Post-Market Analyzer
# ---------------------------------------------------------------------------

class PostMarketAnalyzer:
    """
    Phase 2 Coach v1: reads a day's trades and produces structured diagnosis.

    Usage:
        analyzer = PostMarketAnalyzer()
        diagnosis = analyzer.analyze(trades, news_items=[], analysis_date=date.today())
        print(diagnosis.to_json())
    """

    MAX_LLM_RETRIES = 3

    # The LLM prompt asks for strict JSON matching our schema
    DIAGNOSIS_PROMPT = """You are an expert quantitative trading coach analyzing an Indian equity intraday session.

## Summary
Date: {analysis_date} | Version: {strategy_version}
Trades: {total_trades} | Winners: {winners} ({win_rate:.1f}%) | Losers: {losers}
Gross P&L: ₹{gross_pnl:,.0f} | Net P&L: ₹{net_pnl:,.0f} | Costs: ₹{total_costs:,.0f}

## Trades
{trades_detail}

## News
{news_context}

## Task
Respond with ONLY valid JSON (no markdown, no text before/after).

Schema:
{{
  "winning_patterns": [{{"pattern":"what winners share","trade_ids":["T0001"],"confidence":0.8}}],
  "losing_patterns": [{{"pattern":"what losers share","trade_ids":["T0002"],"confidence":0.8}}],
  "mistakes": [{{"type":"late_entry|early_entry|whipsaw|overtrading|stop_too_tight|stop_too_wide|wrong_direction|held_too_long|exited_too_early|eod_forced_exit","count":1,"pnl_impact":-500.0,"description":"brief why","examples":["T0002"]}}],
  "opportunities": [{{"hypothesis":"what to improve","expected_mechanism":"how it works","suggested_change_type":"weight_adjustment|threshold_change|risk_param|filter_add","confidence":0.8}}],
  "indicator_insights": {{"top_contributors":["names"],"bottom_contributors":["names"]}},
  "regime_insights": {{"detected_regime":"trending_up|trending_down|ranging|volatile","regime_fit":"good|poor|mixed","suggestion":"brief"}}
}}

Rules: trade_ids must exist in trade list. pnl_impact is negative for losses. Empty arrays OK.
"""

    def __init__(self, config: AIConfig = None):
        self.config = config or DEFAULT_AI_CONFIG
        self._client = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM client."""
        if not GENAI_AVAILABLE:
            logger.warning("google-genai not installed, LLM diagnosis disabled")
            return
        if not self.config.llm.api_key:
            logger.warning("GOOGLE_API_KEY not set, LLM diagnosis disabled")
            return
        try:
            self._client = genai.Client(api_key=self.config.llm.api_key)
            logger.info("PostMarketAnalyzer: Gemini client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        trades: List[Dict[str, Any]],
        news_items: List[Dict[str, Any]] | None = None,
        analysis_date: date | None = None,
        strategy_version: str = "v1.0",
    ) -> CoachDiagnosis:
        """
        Analyze a day's trades and produce a CoachDiagnosis.

        Args:
            trades: list of trade dicts from TradeLedger (LedgerEntry.to_dict())
            news_items: optional list of news dicts (NewsItem.to_dict())
            analysis_date: the trading date being analyzed
            strategy_version: strategy version string

        Returns:
            CoachDiagnosis with validated, structured output
        """
        analysis_date = analysis_date or date.today()
        news_items = news_items or []

        # 1. Compute rule-based summary & mistake taxonomy
        self._indicator_scores = []  # will be populated by _diagnose_indicators
        summary = self._compute_summary(trades)
        rule_mistakes = self._classify_mistakes(trades)
        rule_winning = self._find_winning_patterns(trades)
        rule_losing = self._find_losing_patterns(trades)
        indicator_diag = self._diagnose_indicators(trades)
        weight_recs = self._recommend_weight_changes(trades)
        regime_diag = self._diagnose_regime(trades)

        # 2. Try LLM diagnosis (strict JSON with validation + retry)
        llm_result = None
        if self._client and trades:
            llm_result = self._llm_diagnose(
                trades, news_items, analysis_date, strategy_version, summary
            )

        # 3. Merge: LLM enriches rule-based; rule-based is the fallback
        if llm_result:
            winning_patterns = llm_result.get("winning_patterns", rule_winning)
            losing_patterns = llm_result.get("losing_patterns", rule_losing)
            mistakes = self._merge_mistakes(rule_mistakes, llm_result.get("mistakes", []))
            opportunities = [
                Opportunity(
                    hypothesis=o.get("hypothesis", ""),
                    expected_mechanism=o.get("expected_mechanism", ""),
                    suggested_change_type=o.get("suggested_change_type", ""),
                    confidence=float(o.get("confidence", 0)),
                )
                for o in llm_result.get("opportunities", [])
            ]
            news_summary = [
                self._news_dict_to_event(n) for n in llm_result.get("news_summary", [])
            ]
            if llm_result.get("indicator_insights"):
                indicator_diag.update(llm_result["indicator_insights"])
            if llm_result.get("regime_insights"):
                regime_diag.update(llm_result["regime_insights"])
            source = "llm"
        else:
            winning_patterns = rule_winning
            losing_patterns = rule_losing
            mistakes = rule_mistakes
            opportunities = self._generate_rule_opportunities(summary, rule_mistakes)
            news_summary = [self._news_dict_to_event(n) for n in news_items[:10]]
            source = "rule_based"

        return CoachDiagnosis(
            analysis_date=analysis_date.isoformat(),
            strategy_version=strategy_version,
            summary=summary,
            winning_patterns=winning_patterns,
            losing_patterns=losing_patterns,
            mistakes=mistakes,
            opportunities=opportunities,
            news_summary=news_summary,
            indicator_diagnosis=indicator_diag,
            indicator_scores=self._indicator_scores,
            weight_recommendations=weight_recs,
            regime_diagnosis=regime_diag,
            meta={
                "source": source,
                "trades_analyzed": len(trades),
                "generated_at": datetime.now().isoformat(),
                "llm_model": self.config.llm.model_name if self._client else None,
            },
        )

    # ------------------------------------------------------------------
    # Rule-based analysis (always available, no LLM needed)
    # ------------------------------------------------------------------

    def _compute_summary(self, trades: List[Dict]) -> Dict[str, Any]:
        """Compute basic summary stats from trades."""
        if not trades:
            return {"total_trades": 0, "winners": 0, "losers": 0,
                    "gross_pnl": 0, "net_pnl": 0, "total_costs": 0,
                    "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                    "best_trade": 0, "worst_trade": 0, "profit_factor": 0}

        closed = [t for t in trades if t.get("pnl") is not None]
        if not closed:
            return {"total_trades": len(trades), "winners": 0, "losers": 0,
                    "gross_pnl": 0, "net_pnl": 0, "total_costs": 0,
                    "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                    "best_trade": 0, "worst_trade": 0, "profit_factor": 0}

        pnls = [t["pnl"] for t in closed]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        total_costs = sum(t.get("total_cost", 0) for t in closed)
        gross_pnl = sum(pnls) + total_costs  # approximate gross

        return {
            "total_trades": len(closed),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(closed) * 100 if closed else 0,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(sum(pnls), 2),
            "total_costs": round(total_costs, 2),
            "avg_win": round(sum(winners) / len(winners), 2) if winners else 0,
            "avg_loss": round(sum(losers) / len(losers), 2) if losers else 0,
            "best_trade": round(max(pnls), 2) if pnls else 0,
            "worst_trade": round(min(pnls), 2) if pnls else 0,
            "profit_factor": round(
                abs(sum(winners)) / abs(sum(losers)), 2
            ) if losers and sum(losers) != 0 else float("inf") if winners else 0,
        }

    def _classify_mistakes(self, trades: List[Dict]) -> List[Mistake]:
        """Classify trades into mistake categories using rule-based logic."""
        mistakes: Dict[str, Mistake] = {}
        closed = [t for t in trades if t.get("pnl") is not None]
        losing = [t for t in closed if t["pnl"] < 0]

        if not losing:
            return []

        for trade in losing:
            trade_id = trade.get("trade_id", "?")
            pnl = trade["pnl"]
            exit_reason = trade.get("exit_reason", "")
            si_entry = abs(trade.get("si_value", 0))
            entry_time = self._parse_time(trade.get("timestamp"))
            exit_time = self._parse_time(trade.get("exit_time"))

            classified = False

            # 1. Whipsaw: quick round-trip loss (entered & exited within 2 bars / <10 mins)
            if entry_time and exit_time:
                duration = (exit_time - entry_time).total_seconds()
                if duration < 600:  # < 10 minutes
                    self._add_mistake(mistakes, MistakeType.WHIPSAW, pnl, trade_id,
                                      "Quick reversal: position opened and closed within 10 minutes")
                    classified = True

            # 2. EOD forced exit
            if "End of day" in exit_reason or "EOD" in exit_reason.upper():
                self._add_mistake(mistakes, MistakeType.EOD_FORCED_EXIT, pnl, trade_id,
                                  "Position forced closed at end of day")
                classified = True

            # 3. Stop loss hit — check if stop was too tight
            if "Stop loss" in exit_reason or "stop_loss" in exit_reason.lower():
                self._add_mistake(mistakes, MistakeType.STOP_TOO_TIGHT, pnl, trade_id,
                                  "Stop loss triggered — may indicate stop is too tight for current volatility")
                classified = True

            # 4. Late entry (weak signal at entry)
            if si_entry < 0.5 and not classified:
                self._add_mistake(mistakes, MistakeType.LATE_ENTRY, pnl, trade_id,
                                  "Entered with weak SI signal (< 0.5), likely late to the move")
                classified = True

            # 5. Wrong direction catch-all
            if not classified:
                self._add_mistake(mistakes, MistakeType.WRONG_DIRECTION, pnl, trade_id,
                                  "Trade went against the position — possible misread of market direction")

        # 6. Overtrading: check if too many trades in a short window
        if len(closed) > 0:
            trades_by_hour = self._count_trades_by_hour(closed)
            for hour, count in trades_by_hour.items():
                if count >= 5:  # 5+ trades in a single hour
                    hour_trades = [
                        t for t in closed
                        if self._get_hour(t.get("timestamp")) == hour
                    ]
                    hour_losers = [t for t in hour_trades if t.get("pnl", 0) < 0]
                    if hour_losers:
                        impact = sum(t["pnl"] for t in hour_losers)
                        ids = [t.get("trade_id", "?") for t in hour_losers]
                        self._add_mistake(
                            mistakes, MistakeType.OVERTRADING, impact, ids[0],
                            f"Excessive trading in hour {hour}:00 ({count} trades)"
                        )
                        for tid in ids[1:]:
                            if MistakeType.OVERTRADING.value in mistakes:
                                mistakes[MistakeType.OVERTRADING.value].examples.append(tid)

        return list(mistakes.values())

    def _find_winning_patterns(self, trades: List[Dict]) -> List[Dict]:
        """Rule-based winning pattern detection."""
        closed = [t for t in trades if t.get("pnl") is not None]
        winners = [t for t in closed if t["pnl"] > 0]
        if not winners:
            return []

        patterns = []

        # Pattern: strong SI at entry
        strong_si = [t for t in winners if abs(t.get("si_value", 0)) > 0.75]
        if len(strong_si) >= 2:
            patterns.append({
                "pattern": "Strong SI signal at entry (> 0.75) led to profitable trades",
                "trade_ids": [t.get("trade_id", "?") for t in strong_si],
                "confidence": min(1.0, len(strong_si) / len(winners)),
            })

        # Pattern: time-of-day clustering
        time_clusters = self._cluster_by_time(winners)
        for cluster_label, cluster_trades in time_clusters.items():
            if len(cluster_trades) >= 2:
                patterns.append({
                    "pattern": f"Winning trades clustered in {cluster_label}",
                    "trade_ids": [t.get("trade_id", "?") for t in cluster_trades],
                    "confidence": min(1.0, len(cluster_trades) / len(winners)),
                })

        # Pattern: exit by signal (vs stop/eod)
        signal_exits = [t for t in winners if "signal" in t.get("exit_reason", "").lower()]
        if len(signal_exits) >= 2:
            patterns.append({
                "pattern": "Profitable trades exited by SI signal (not stop/EOD)",
                "trade_ids": [t.get("trade_id", "?") for t in signal_exits],
                "confidence": min(1.0, len(signal_exits) / len(winners)),
            })

        return patterns

    def _find_losing_patterns(self, trades: List[Dict]) -> List[Dict]:
        """Rule-based losing pattern detection."""
        closed = [t for t in trades if t.get("pnl") is not None]
        losers = [t for t in closed if t["pnl"] < 0]
        if not losers:
            return []

        patterns = []

        # Pattern: weak SI
        weak_si = [t for t in losers if abs(t.get("si_value", 0)) < 0.5]
        if len(weak_si) >= 2:
            patterns.append({
                "pattern": "Losing trades had weak SI at entry (< 0.5)",
                "trade_ids": [t.get("trade_id", "?") for t in weak_si],
                "confidence": min(1.0, len(weak_si) / len(losers)),
            })

        # Pattern: same symbol repeated losses
        from collections import Counter
        symbol_counts = Counter(t.get("symbol") for t in losers)
        for sym, cnt in symbol_counts.items():
            if cnt >= 2:
                sym_trades = [t for t in losers if t.get("symbol") == sym]
                patterns.append({
                    "pattern": f"Multiple losses on {sym} ({cnt} trades)",
                    "trade_ids": [t.get("trade_id", "?") for t in sym_trades],
                    "confidence": min(1.0, cnt / len(losers)),
                })

        # Pattern: time clustering
        time_clusters = self._cluster_by_time(losers)
        for cluster_label, cluster_trades in time_clusters.items():
            if len(cluster_trades) >= 2:
                patterns.append({
                    "pattern": f"Losing trades clustered in {cluster_label}",
                    "trade_ids": [t.get("trade_id", "?") for t in cluster_trades],
                    "confidence": min(1.0, len(cluster_trades) / len(losers)),
                })

        return patterns

    def _diagnose_indicators(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        Deep per-indicator analysis.
        Compares each indicator's value distribution between winners and losers.
        Returns overall diagnosis dict + populates self._indicator_scores.
        """
        closed = [t for t in trades if t.get("pnl") is not None]
        if not closed:
            self._indicator_scores = []
            return {"note": "No closed trades to analyze"}

        winners = [t for t in closed if t["pnl"] > 0]
        losers = [t for t in closed if t["pnl"] < 0]

        # Aggregate SI-level stats
        avg_si_winners = (
            sum(abs(t.get("si_value", 0)) for t in winners) / len(winners)
            if winners else 0
        )
        avg_si_losers = (
            sum(abs(t.get("si_value", 0)) for t in losers) / len(losers)
            if losers else 0
        )

        # Per-indicator analysis using indicator_snapshot
        has_snapshots = any(t.get("indicator_snapshot") for t in closed)
        scores: List[IndicatorScore] = []
        helping = []
        hurting = []
        neutral = []

        if has_snapshots:
            # Collect all indicator names
            all_indicators: set = set()
            for t in closed:
                snap = t.get("indicator_snapshot", {})
                all_indicators.update(snap.keys())

            for ind_name in sorted(all_indicators):
                # Gather values for winners and losers
                winner_vals = []
                loser_vals = []
                for t in winners:
                    snap = t.get("indicator_snapshot", {})
                    if ind_name in snap:
                        winner_vals.append(snap[ind_name])
                for t in losers:
                    snap = t.get("indicator_snapshot", {})
                    if ind_name in snap:
                        loser_vals.append(snap[ind_name])

                total_count = len(winner_vals) + len(loser_vals)
                if total_count == 0:
                    continue

                avg_w = sum(winner_vals) / len(winner_vals) if winner_vals else 0
                avg_l = sum(loser_vals) / len(loser_vals) if loser_vals else 0

                # Win correlation: positive weight + higher in winners = helping
                # Measure how much the indicator differs between W and L
                diff = avg_w - avg_l
                # Normalize by the range to get a -1 to 1 correlation proxy
                max_abs = max(abs(avg_w), abs(avg_l), 0.001)
                correlation = diff / max_abs
                correlation = max(-1.0, min(1.0, correlation))

                if abs(correlation) < 0.1:
                    verdict = "neutral"
                    neutral.append(ind_name)
                elif correlation > 0:
                    verdict = "helping"
                    helping.append(ind_name)
                else:
                    verdict = "hurting"
                    hurting.append(ind_name)

                # Get current weight
                current_weight = 0.0
                for t in closed:
                    snap = t.get("indicator_snapshot", {})
                    if ind_name in snap:
                        current_weight = snap[ind_name]
                        break

                scores.append(IndicatorScore(
                    name=ind_name,
                    current_weight=round(current_weight, 4),
                    avg_value_winners=round(avg_w, 4),
                    avg_value_losers=round(avg_l, 4),
                    win_correlation=round(correlation, 4),
                    trade_count=total_count,
                    verdict=verdict,
                ))

            # Sort: hurting first (most negative correlation), then neutral, then helping
            scores.sort(key=lambda s: s.win_correlation)

        self._indicator_scores = scores

        return {
            "avg_si_at_entry_winners": round(avg_si_winners, 4),
            "avg_si_at_entry_losers": round(avg_si_losers, 4),
            "si_discriminative": avg_si_winners > avg_si_losers,
            "total_indicators_analyzed": len(scores),
            "helping": helping[:5],
            "hurting": hurting[:5],
            "neutral_count": len(neutral),
            "suggestion": (
                "SI is discriminating winners from losers — trust strong signals"
                if avg_si_winners > avg_si_losers
                else "SI does not clearly separate winners from losers — consider recalibration"
            ),
        }

    def _recommend_weight_changes(
        self,
        trades: List[Dict],
        strategy_weights: Dict[str, float] | None = None,
    ) -> List[WeightRecommendation]:
        """
        Based on per-indicator scores, recommend weight adjustments.
        Rules:
        - Hurting indicators with |weight| > 0.3: reduce weight by 20%
        - Helping indicators with |weight| < 0.8: increase weight by 10%
        - Neutral indicators with |weight| > 0.5: reduce weight by 10%
        - Cap all changes to max 10% per update (safety)
        """
        if not hasattr(self, "_indicator_scores") or not self._indicator_scores:
            return []

        MAX_CHANGE_PCT = 0.10  # max 10% change per indicator per day

        recs: List[WeightRecommendation] = []

        for score in self._indicator_scores:
            current = score.current_weight
            if current == 0:
                continue

            recommended = current
            reason = ""

            if score.verdict == "hurting":
                # Reduce weight by 20% (capped to MAX_CHANGE_PCT of abs value)
                reduction = abs(current) * 0.20
                reduction = min(reduction, abs(current) * MAX_CHANGE_PCT)
                if current > 0:
                    recommended = current - reduction
                else:
                    recommended = current + reduction  # less negative
                reason = (
                    f"Hurting performance (correlation={score.win_correlation:+.2f}). "
                    f"Winners avg={score.avg_value_winners:.4f}, "
                    f"Losers avg={score.avg_value_losers:.4f}"
                )

            elif score.verdict == "helping" and abs(current) < 0.8:
                # Increase weight by 10% (capped)
                increase = abs(current) * 0.10
                increase = min(increase, abs(current) * MAX_CHANGE_PCT)
                if current > 0:
                    recommended = current + increase
                else:
                    recommended = current - increase  # more negative
                reason = (
                    f"Helping performance (correlation={score.win_correlation:+.2f}). "
                    f"Consider boosting weight."
                )

            elif score.verdict == "neutral" and abs(current) > 0.5:
                # Slightly reduce — not contributing
                reduction = abs(current) * 0.10
                reduction = min(reduction, abs(current) * MAX_CHANGE_PCT)
                if current > 0:
                    recommended = current - reduction
                else:
                    recommended = current + reduction
                reason = (
                    f"Not discriminating winners from losers "
                    f"(correlation={score.win_correlation:+.2f}). "
                    f"Consider reducing weight to free up signal space."
                )

            if recommended != current:
                change = recommended - current
                recs.append(WeightRecommendation(
                    indicator=score.name,
                    current_weight=round(current, 4),
                    recommended_weight=round(recommended, 4),
                    change=round(change, 4),
                    reason=reason,
                    confidence=min(1.0, abs(score.win_correlation) * 1.5),
                ))

        # Sort by absolute change (biggest changes first)
        recs.sort(key=lambda r: abs(r.change), reverse=True)
        return recs

    def _diagnose_regime(self, trades: List[Dict]) -> Dict[str, Any]:
        """Basic regime diagnosis."""
        closed = [t for t in trades if t.get("pnl") is not None]
        if not closed:
            return {"detected_regime": "unknown", "regime_fit": "unknown"}

        pnls = [t["pnl"] for t in closed]
        total_pnl = sum(pnls)
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)

        # Simple heuristic: detect day character
        if win_rate > 0.6 and total_pnl > 0:
            regime = "trending"
            fit = "good"
        elif win_rate < 0.35:
            regime = "choppy"
            fit = "poor"
        else:
            regime = "mixed"
            fit = "mixed"

        return {
            "detected_regime": regime,
            "regime_fit": fit,
            "day_win_rate": round(win_rate * 100, 1),
            "day_pnl": round(total_pnl, 2),
        }

    def _generate_rule_opportunities(
        self, summary: Dict, mistakes: List[Mistake]
    ) -> List[Opportunity]:
        """Generate opportunities from rule-based analysis."""
        opps = []

        if summary.get("win_rate", 0) < 40:
            opps.append(Opportunity(
                hypothesis="Win rate is low — tighten entry threshold to filter weak signals",
                expected_mechanism="Raising entry threshold from 0.70 to 0.75 would skip borderline signals",
                suggested_change_type="threshold_change",
                confidence=0.6,
            ))

        if summary.get("profit_factor", 0) < 1.0 and summary.get("profit_factor", 0) > 0:
            opps.append(Opportunity(
                hypothesis="Profit factor < 1 — average losses exceed average wins",
                expected_mechanism="Tighten stop loss or widen take profit to improve risk/reward",
                suggested_change_type="risk_param",
                confidence=0.5,
            ))

        mistake_types = {m.type for m in mistakes}

        if MistakeType.WHIPSAW.value in mistake_types:
            opps.append(Opportunity(
                hypothesis="Whipsaw trades suggest choppy market — add a confirmation filter",
                expected_mechanism="Require SI to stay above threshold for 2 consecutive bars before entry",
                suggested_change_type="filter_add",
                confidence=0.5,
            ))

        if MistakeType.OVERTRADING.value in mistake_types:
            opps.append(Opportunity(
                hypothesis="Overtrading in concentrated hours — reduce max trades/hour",
                expected_mechanism="Cap trades per hour to 3 to prevent overtrading in choppy periods",
                suggested_change_type="risk_param",
                confidence=0.6,
            ))

        if MistakeType.EOD_FORCED_EXIT.value in mistake_types:
            opps.append(Opportunity(
                hypothesis="EOD forced exits are losing money — stop new entries after 2:30 PM",
                expected_mechanism="No new positions after 14:30 IST to avoid forced flatten losses",
                suggested_change_type="risk_param",
                confidence=0.7,
            ))

        return opps

    # ------------------------------------------------------------------
    # LLM diagnosis with strict JSON validation + retry
    # ------------------------------------------------------------------

    def _llm_diagnose(
        self,
        trades: List[Dict],
        news_items: List[Dict],
        analysis_date: date,
        strategy_version: str,
        summary: Dict,
    ) -> Optional[Dict]:
        """Call LLM for diagnosis with retry on invalid JSON."""
        trades_detail = self._format_trades_for_prompt(trades)
        news_context = self._format_news_for_prompt(news_items)

        prompt = self.DIAGNOSIS_PROMPT.format(
            analysis_date=analysis_date.isoformat(),
            strategy_version=strategy_version,
            total_trades=summary["total_trades"],
            winners=summary["winners"],
            losers=summary["losers"],
            win_rate=summary["win_rate"],
            gross_pnl=summary["gross_pnl"],
            net_pnl=summary["net_pnl"],
            total_costs=summary["total_costs"],
            trades_detail=trades_detail,
            news_context=news_context,
        )

        for attempt in range(1, self.MAX_LLM_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.config.llm.model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.1,
                        "max_output_tokens": 4096,
                    },
                )
                raw = response.text.strip()
                
                # Enhanced cleaning
                # 1. Strip markdown
                if "```" in raw:
                    import re
                    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                    if match:
                        raw = match.group(1)
                    else:
                        # Fallback simple strip
                        if raw.startswith("```"):
                            first_newline = raw.find("\n")
                            last_fence = raw.rfind("```")
                            if first_newline != -1 and last_fence != -1:
                                raw = raw[first_newline + 1:last_fence].strip()

                # 2. Extract first JSON object if surrounded by text
                start_idx = raw.find('{')
                end_idx = raw.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    raw = raw[start_idx:end_idx+1]
                
                # 3. Clean control characters
                raw = raw.replace('\n', ' ').replace('\r', '')

                data = json.loads(raw)

                # Validate structure
                is_valid, errors = CoachDiagnosis.validate_dict({
                    "analysis_date": analysis_date.isoformat(),
                    "strategy_version": strategy_version,
                    "summary": summary,
                    **data,
                })

                if is_valid:
                    logger.info(f"LLM diagnosis valid on attempt {attempt}")
                    return data
                else:
                    logger.warning(
                        f"LLM attempt {attempt}: validation errors: {errors}"
                    )
                    # Add validation errors to next prompt for correction
                    prompt += (
                        f"\n\nYour previous response had these errors: {errors}. "
                        "Fix them and respond with valid JSON only."
                    )

            except json.JSONDecodeError as e:
                logger.warning(f"LLM attempt {attempt}: invalid JSON: {e}")
                prompt += (
                    "\n\nYour previous response was not valid JSON. "
                    "Respond with ONLY a valid JSON object, no other text."
                )
            except Exception as e:
                logger.error(f"LLM attempt {attempt} failed: {e}")
                break

        logger.warning("All LLM retries exhausted, falling back to rule-based")
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_mistake(
        self,
        mistakes: Dict[str, Mistake],
        mtype: MistakeType,
        pnl: float,
        trade_id: str,
        description: str,
    ):
        """Add or update a mistake entry."""
        key = mtype.value
        if key in mistakes:
            mistakes[key].count += 1
            mistakes[key].pnl_impact += pnl
            if trade_id not in mistakes[key].examples:
                mistakes[key].examples.append(trade_id)
        else:
            mistakes[key] = Mistake(
                type=key,
                count=1,
                pnl_impact=round(pnl, 2),
                description=description,
                examples=[trade_id],
            )

    def _merge_mistakes(
        self, rule_mistakes: List[Mistake], llm_mistakes: List[Dict]
    ) -> List[Mistake]:
        """Merge rule-based and LLM mistakes, preferring LLM descriptions."""
        merged: Dict[str, Mistake] = {m.type: m for m in rule_mistakes}

        for lm in llm_mistakes:
            mtype = lm.get("type", "unknown")
            if mtype in merged:
                # LLM enriches description
                merged[mtype].description = lm.get("description", merged[mtype].description)
            else:
                merged[mtype] = Mistake(
                    type=mtype,
                    count=lm.get("count", 1),
                    pnl_impact=lm.get("pnl_impact", 0),
                    description=lm.get("description", ""),
                    examples=lm.get("examples", []),
                )

        return list(merged.values())

    def _parse_time(self, ts: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                return None
        return None

    def _count_trades_by_hour(self, trades: List[Dict]) -> Dict[int, int]:
        """Count trades per hour of day."""
        counts: Dict[int, int] = {}
        for t in trades:
            hour = self._get_hour(t.get("timestamp"))
            if hour is not None:
                counts[hour] = counts.get(hour, 0) + 1
        return counts

    def _get_hour(self, ts: Any) -> Optional[int]:
        """Extract hour from timestamp."""
        dt = self._parse_time(ts)
        return dt.hour if dt else None

    def _cluster_by_time(self, trades: List[Dict]) -> Dict[str, List[Dict]]:
        """Cluster trades into time-of-day buckets."""
        buckets = {
            "morning_open (9:15-10:00)": (9, 10),
            "mid_morning (10:00-11:30)": (10, 12),
            "afternoon (12:00-14:00)": (12, 14),
            "closing_hour (14:00-15:30)": (14, 16),
        }
        clusters: Dict[str, List[Dict]] = {}
        for label, (start_h, end_h) in buckets.items():
            matching = [
                t for t in trades
                if self._get_hour(t.get("timestamp")) is not None
                and start_h <= self._get_hour(t.get("timestamp")) < end_h
            ]
            if matching:
                clusters[label] = matching
        return clusters

    @staticmethod
    def _news_dict_to_event(n: Dict) -> NewsEvent:
        """Convert a news dict (various formats) into a NewsEvent."""
        return NewsEvent(
            event=n.get("event", n.get("headline", "Unknown event")),
            impact_level=n.get("impact_level", n.get("impact", "medium")),
            affected_symbols=n.get("affected_symbols", n.get("symbols", [])),
            timestamp=n.get("timestamp", ""),
        )

    def _format_trades_for_prompt(self, trades: List[Dict]) -> str:
        """Format trades for the LLM prompt."""
        if not trades:
            return "No trades today."

        lines = []
        for t in trades:
            pnl = t.get("pnl")
            pnl_str = f"₹{pnl:,.0f}" if pnl is not None else "OPEN"
            result = ""
            if pnl is not None:
                result = " WIN" if pnl > 0 else " LOSS"

            line = (
                f"  {t.get('trade_id', '?')} | {t.get('symbol', '?')} | "
                f"{t.get('side', '?')} | ₹{t.get('price', 0):,.2f}"
            )
            if t.get("exit_price"):
                line += f" → ₹{t['exit_price']:,.2f}"
            line += f" | SI={t.get('si_value', 0):.3f}"
            line += f" | P&L: {pnl_str}{result}"
            line += f" | Entry: {t.get('entry_reason', '?')}"
            if t.get("exit_reason"):
                line += f" | Exit: {t['exit_reason']}"

            lines.append(line)

        return "\n".join(lines)

    def _format_news_for_prompt(self, news_items: List[Dict]) -> str:
        """Format news for the LLM prompt."""
        if not news_items:
            return "No market news available for this day."

        lines = []
        for n in news_items[:15]:  # limit to 15 items
            line = f"  [{n.get('impact', n.get('impact_level', 'medium'))}] "
            line += n.get("headline", n.get("event", "?"))
            syms = n.get("symbols", n.get("affected_symbols", []))
            if syms:
                line += f" ({', '.join(syms)})"
            lines.append(line)

        return "\n".join(lines)
