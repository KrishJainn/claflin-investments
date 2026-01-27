"""
Hall of Fame module.

Tracks the best-performing DNA configurations across all generations.
"""

from typing import List, Optional, Dict
import json
from pathlib import Path
import logging

from ..super_indicator.dna import SuperIndicatorDNA
from ..journal.database import Database

logger = logging.getLogger(__name__)


class HallOfFame:
    """
    Tracks the best DNA configurations across all evolution runs.

    Features:
    - Maintains top N strategies of all time
    - Prevents duplicates
    - Persists to database
    - Provides comparison and analysis
    """

    def __init__(self,
                 database: Database = None,
                 max_size: int = 10):
        """
        Initialize Hall of Fame.

        Args:
            database: Database for persistence
            max_size: Maximum entries to track
        """
        self.db = database
        self.max_size = max_size
        self._entries: List[SuperIndicatorDNA] = []
        self._loaded = False

    def _load_from_db(self):
        """Load Hall of Fame from database."""
        if self._loaded or self.db is None:
            return

        try:
            entries = self.db.get_hall_of_fame()
            for entry in entries:
                dna = SuperIndicatorDNA.from_dict({
                    'dna_id': entry['dna_id'],
                    'generation': entry['generation_num'],
                    'run_id': entry['run_id'],
                    'genes': {},  # Would need to load from dna_configs
                    'fitness_score': entry['fitness_score'],
                    'sharpe_ratio': entry['sharpe_ratio'],
                    'max_drawdown': entry['max_drawdown'],
                    'net_profit': entry['net_profit'],
                    'win_rate': entry.get('win_rate', 0),
                    'validation_fitness': entry.get('validation_fitness'),
                    'holdout_fitness': entry.get('holdout_fitness')
                })
                self._entries.append(dna)

            self._loaded = True
            logger.info(f"Loaded {len(self._entries)} Hall of Fame entries")

        except Exception as e:
            logger.warning(f"Could not load Hall of Fame: {e}")

    def update(self, dna: SuperIndicatorDNA) -> bool:
        """
        Update Hall of Fame with a new DNA if it qualifies.

        Args:
            dna: DNA to potentially add

        Returns:
            True if DNA was added to Hall of Fame
        """
        self._load_from_db()

        # Check if already in hall of fame
        if any(e.dna_id == dna.dna_id for e in self._entries):
            return False

        # Check if it qualifies
        if len(self._entries) >= self.max_size:
            min_fitness = min(e.fitness_score for e in self._entries)
            if dna.fitness_score <= min_fitness:
                return False

            # Remove worst entry
            self._entries = sorted(
                self._entries,
                key=lambda d: d.fitness_score,
                reverse=True
            )[:self.max_size - 1]

        # Add new entry
        self._entries.append(dna.copy())
        self._entries = sorted(
            self._entries,
            key=lambda d: d.fitness_score,
            reverse=True
        )

        # Save to database
        if self.db:
            try:
                self.db.update_hall_of_fame(
                    dna_id=dna.dna_id,
                    run_id=dna.run_id,
                    generation_num=dna.generation,
                    fitness_score=dna.fitness_score,
                    metrics={
                        'sharpe_ratio': dna.sharpe_ratio,
                        'max_drawdown': dna.max_drawdown,
                        'net_profit': dna.net_profit,
                        'win_rate': dna.win_rate,
                        'total_trades': dna.total_trades,
                        'validation_fitness': dna.validation_fitness,
                        'holdout_fitness': dna.holdout_fitness
                    },
                    weights_json=json.dumps(dna.get_weights()),
                    max_size=self.max_size
                )
            except Exception as e:
                logger.warning(f"Could not save to Hall of Fame: {e}")

        logger.info(f"Added DNA {dna.dna_id} to Hall of Fame (fitness: {dna.fitness_score:.4f})")
        return True

    def get_best(self) -> Optional[SuperIndicatorDNA]:
        """Get the best DNA configuration."""
        self._load_from_db()
        return self._entries[0] if self._entries else None

    def get_top(self, n: int = 5) -> List[SuperIndicatorDNA]:
        """Get top N DNA configurations."""
        self._load_from_db()
        return self._entries[:n]

    def get_all(self) -> List[SuperIndicatorDNA]:
        """Get all Hall of Fame entries."""
        self._load_from_db()
        return self._entries.copy()

    def get_rank(self, dna_id: str) -> Optional[int]:
        """Get rank of a DNA in Hall of Fame."""
        self._load_from_db()
        for i, entry in enumerate(self._entries):
            if entry.dna_id == dna_id:
                return i + 1
        return None

    def contains(self, dna_id: str) -> bool:
        """Check if DNA is in Hall of Fame."""
        self._load_from_db()
        return any(e.dna_id == dna_id for e in self._entries)

    def get_stats(self) -> Dict:
        """Get Hall of Fame statistics."""
        self._load_from_db()

        if not self._entries:
            return {'count': 0}

        return {
            'count': len(self._entries),
            'best_fitness': self._entries[0].fitness_score,
            'worst_fitness': self._entries[-1].fitness_score,
            'avg_fitness': sum(e.fitness_score for e in self._entries) / len(self._entries),
            'best_sharpe': max(e.sharpe_ratio for e in self._entries),
            'best_profit': max(e.net_profit for e in self._entries),
            'generations_represented': len(set(e.generation for e in self._entries)),
            'runs_represented': len(set(e.run_id for e in self._entries))
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        self._load_from_db()

        if not self._entries:
            return "Hall of Fame is empty"

        lines = [
            "Hall of Fame",
            "=" * 50,
        ]

        for i, entry in enumerate(self._entries, 1):
            lines.append(
                f"{i}. DNA {entry.dna_id} (Gen {entry.generation}) | "
                f"Fitness: {entry.fitness_score:.4f} | "
                f"Sharpe: {entry.sharpe_ratio:.2f} | "
                f"Profit: ${entry.net_profit:.0f}"
            )

        return "\n".join(lines)

    def save_to_file(self, filepath: str):
        """Save Hall of Fame to JSON file."""
        self._load_from_db()

        data = {
            'entries': [e.to_dict() for e in self._entries],
            'stats': self.get_stats()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved Hall of Fame to {filepath}")

    def load_from_file(self, filepath: str):
        """Load Hall of Fame from JSON file."""
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Hall of Fame file not found: {filepath}")
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self._entries = [
            SuperIndicatorDNA.from_dict(e)
            for e in data.get('entries', [])
        ]
        self._loaded = True

        logger.info(f"Loaded {len(self._entries)} entries from {filepath}")

    def clear(self):
        """Clear the Hall of Fame."""
        self._entries = []
        logger.info("Hall of Fame cleared")
