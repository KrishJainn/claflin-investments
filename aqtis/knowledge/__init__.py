"""
AQTIS Knowledge Ingestion System.

Provides a unified interface for ingesting knowledge from multiple sources
into the ChromaDB vector store for use by trading agents.
"""

from .knowledge_manager import KnowledgeManager

__all__ = ["KnowledgeManager"]
