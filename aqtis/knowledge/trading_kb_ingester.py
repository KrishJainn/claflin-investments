"""
Trading Knowledge Base Ingester.

Imports the curated trading knowledge from the 5-player coach model's
knowledge base into the AQTIS vector store for semantic retrieval.

This includes indicator combinations, regime detection rules, risk management
rules, VWAP strategies, Bollinger Band squeeze rules, SuperTrend rules,
volume analysis, intraday time rules, Indian market knowledge, and book wisdom.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from .base_ingester import BaseIngester

logger = logging.getLogger(__name__)

# Attempt to import the 5-player knowledge base
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    KB_AVAILABLE = True
except ImportError:
    KB_AVAILABLE = False
    logger.warning("5-player trading knowledge base not available")


class TradingKBIngester(BaseIngester):
    """Ingests the 5-player coach trading knowledge base into AQTIS vectors."""

    def __init__(self, vector_store):
        super().__init__(vector_store, source_name="trading_knowledge_base")

    def ingest(self, **kwargs) -> Dict[str, Any]:
        """Ingest all trading knowledge into the vector store."""
        if not KB_AVAILABLE:
            return {
                "documents_ingested": 0,
                "chunks_created": 0,
                "errors": ["trading_knowledge_base module not available"],
            }

        total_chunks = 0
        errors = []

        sections = [
            ("indicator_combinations", self._ingest_indicator_combinations),
            ("regime_detection", self._ingest_regime_detection),
            ("risk_management", self._ingest_risk_management),
            ("vwap_strategies", self._ingest_vwap_strategies),
            ("bb_squeeze", self._ingest_bb_squeeze),
            ("supertrend", self._ingest_supertrend),
            ("volume_rules", self._ingest_volume_rules),
            ("intraday_time_rules", self._ingest_intraday_time),
            ("indian_market", self._ingest_indian_market),
            ("book_wisdom", self._ingest_book_wisdom),
            ("coach_optimization", self._ingest_coach_optimization),
            ("indicator_settings", self._ingest_indicator_settings),
            ("rsi_divergence", self._ingest_rsi_divergence),
            ("strategy_matrix", self._ingest_strategy_matrix),
        ]

        for section_name, ingest_fn in sections:
            try:
                count = ingest_fn()
                total_chunks += count
                self.logger.info(f"Ingested {count} chunks from {section_name}")
            except Exception as e:
                errors.append(f"{section_name}: {e}")
                self.logger.error(f"Error ingesting {section_name}: {e}")

        return {
            "documents_ingested": len(sections),
            "chunks_created": total_chunks,
            "errors": errors,
        }

    def _dict_to_text(self, d: Any, indent: int = 0) -> str:
        """Recursively convert a dict/list to readable text."""
        prefix = "  " * indent
        if isinstance(d, dict):
            lines = []
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{k}:")
                    lines.append(self._dict_to_text(v, indent + 1))
                else:
                    lines.append(f"{prefix}{k}: {v}")
            return "\n".join(lines)
        elif isinstance(d, (list, tuple)):
            return "\n".join(f"{prefix}- {item}" for item in d)
        return f"{prefix}{d}"

    def _ingest_indicator_combinations(self) -> int:
        count = 0
        for name, combo in INDICATOR_COMBINATIONS.items():
            text = f"INDICATOR COMBINATION: {combo.get('name', name)}\n"
            text += f"Components: {combo.get('components', [])}\n"
            text += self._dict_to_text(combo)

            self._store_chunk(text, {
                "category": "indicator_combination",
                "combination_name": name,
                "components": json.dumps(combo.get("components", [])),
            })
            count += 1
        return count

    def _ingest_regime_detection(self) -> int:
        text = "REGIME DETECTION RULES\n"
        text += self._dict_to_text(REGIME_DETECTION_RULES)
        chunks = self._chunk_text(text, max_tokens=400)
        for i, chunk in enumerate(chunks):
            self._store_chunk(chunk, {
                "category": "regime_detection",
                "part": i + 1,
            })
        return len(chunks)

    def _ingest_risk_management(self) -> int:
        text = "RISK MANAGEMENT RULES\n"
        text += self._dict_to_text(RISK_MANAGEMENT_RULES)
        chunks = self._chunk_text(text, max_tokens=400)
        for i, chunk in enumerate(chunks):
            self._store_chunk(chunk, {
                "category": "risk_management",
                "part": i + 1,
            })
        return len(chunks)

    def _ingest_vwap_strategies(self) -> int:
        count = 0
        for name, strat in VWAP_STRATEGIES.items():
            text = f"VWAP STRATEGY: {strat.get('name', name)}\n"
            text += self._dict_to_text(strat)
            self._store_chunk(text, {
                "category": "vwap_strategy",
                "strategy_name": name,
            })
            count += 1
        return count

    def _ingest_bb_squeeze(self) -> int:
        text = "BOLLINGER BAND SQUEEZE RULES\n"
        text += self._dict_to_text(BB_SQUEEZE_RULES)
        chunks = self._chunk_text(text, max_tokens=400)
        for i, chunk in enumerate(chunks):
            self._store_chunk(chunk, {
                "category": "bb_squeeze",
                "part": i + 1,
            })
        return len(chunks)

    def _ingest_supertrend(self) -> int:
        text = "SUPERTREND RULES (15-MIN)\n"
        text += self._dict_to_text(SUPERTREND_RULES)
        self._store_chunk(text, {"category": "supertrend"})
        return 1

    def _ingest_volume_rules(self) -> int:
        text = "VOLUME ANALYSIS RULES\n"
        text += self._dict_to_text(VOLUME_RULES)
        chunks = self._chunk_text(text, max_tokens=400)
        for i, chunk in enumerate(chunks):
            self._store_chunk(chunk, {
                "category": "volume_rules",
                "part": i + 1,
            })
        return len(chunks)

    def _ingest_intraday_time(self) -> int:
        count = 0
        phases = INTRADAY_TIME_RULES.get("trading_phases", {})
        for phase_key, phase in phases.items():
            if isinstance(phase, dict):
                text = f"INTRADAY TIME PHASE: {phase_key}\n"
                text += self._dict_to_text(phase)
                self._store_chunk(text, {
                    "category": "intraday_time",
                    "phase": phase_key,
                    "time_range": phase.get("time", ""),
                })
                count += 1

        # Gap strategies
        gap = INTRADAY_TIME_RULES.get("gap_opening_strategy", {})
        if gap:
            text = "GAP OPENING STRATEGIES\n" + self._dict_to_text(gap)
            self._store_chunk(text, {"category": "intraday_time", "phase": "gap_strategy"})
            count += 1
        return count

    def _ingest_indian_market(self) -> int:
        text = "INDIAN MARKET (NIFTY 50) KNOWLEDGE\n"
        text += self._dict_to_text(INDIAN_MARKET_KNOWLEDGE)
        chunks = self._chunk_text(text, max_tokens=400)
        for i, chunk in enumerate(chunks):
            self._store_chunk(chunk, {
                "category": "indian_market",
                "part": i + 1,
            })
        return len(chunks)

    def _ingest_book_wisdom(self) -> int:
        count = 0
        for book_key, book in BOOK_WISDOM.items():
            text = f"BOOK WISDOM: {book_key}\n"
            text += self._dict_to_text(book)
            chunks = self._chunk_text(text, max_tokens=400)
            for i, chunk in enumerate(chunks):
                self._store_chunk(chunk, {
                    "category": "book_wisdom",
                    "book": book_key,
                    "part": i + 1,
                })
                count += 1
        return count

    def _ingest_coach_optimization(self) -> int:
        text = "COACH OPTIMIZATION PARAMETERS\n"
        text += self._dict_to_text(COACH_OPTIMIZATION_PARAMS)
        chunks = self._chunk_text(text, max_tokens=400)
        for i, chunk in enumerate(chunks):
            self._store_chunk(chunk, {
                "category": "coach_optimization",
                "part": i + 1,
            })
        return len(chunks)

    def _ingest_indicator_settings(self) -> int:
        count = 0
        for ind_name, settings in INDICATOR_OPTIMAL_SETTINGS.items():
            text = f"INDICATOR SETTINGS: {ind_name}\n"
            text += self._dict_to_text(settings)
            self._store_chunk(text, {
                "category": "indicator_settings",
                "indicator": ind_name,
            })
            count += 1
        return count

    def _ingest_rsi_divergence(self) -> int:
        text = "RSI DIVERGENCE RULES\n"
        text += self._dict_to_text(RSI_DIVERGENCE_RULES)
        self._store_chunk(text, {"category": "rsi_divergence"})
        return 1

    def _ingest_strategy_matrix(self) -> int:
        text = "STRATEGY DECISION MATRIX\n"
        text += str(STRATEGY_DECISION_MATRIX.get("description", ""))
        matrix = STRATEGY_DECISION_MATRIX.get("matrix", {})
        for key, config in matrix.items():
            text += f"\n\nRegime+Time: {key}\n"
            text += self._dict_to_text(config)
        chunks = self._chunk_text(text, max_tokens=400)
        for i, chunk in enumerate(chunks):
            self._store_chunk(chunk, {
                "category": "strategy_matrix",
                "part": i + 1,
            })
        return len(chunks)
