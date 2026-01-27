"""
Super Indicator DNA module.

Defines the genetic representation of a Super Indicator configuration.
Supports serialization, comparison, and genetic operations.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np


def generate_dna_id() -> str:
    """Generate unique DNA identifier."""
    return str(uuid.uuid4())[:8]


@dataclass
class IndicatorGene:
    """
    Single indicator gene in the DNA.

    Represents one indicator's configuration in the Super Indicator.
    """
    name: str  # Indicator name (e.g., 'RSI_14')
    weight: float  # Weight in [-1, 1], negative = inverse signal
    active: bool = True  # Whether this indicator is used
    category: str = ""  # Indicator category


@dataclass
class SuperIndicatorDNA:
    """
    Genetic representation of a Super Indicator configuration.

    The DNA encodes:
    - Which indicators are active
    - Weight of each indicator
    - Performance metrics from backtesting

    This is the "genotype" that evolves over generations.
    """

    # Identity
    dna_id: str = field(default_factory=generate_dna_id)
    generation: int = 0
    run_id: int = 0

    # Genetic content
    genes: Dict[str, IndicatorGene] = field(default_factory=dict)

    # Lineage
    parent_ids: Tuple[str, str] = ("", "")
    mutation_history: List[str] = field(default_factory=list)

    # Performance metrics (filled after evaluation)
    fitness_score: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    net_profit: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    profit_factor: float = 0.0

    # Validation scores
    validation_fitness: Optional[float] = None
    holdout_fitness: Optional[float] = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_active_indicators(self) -> List[str]:
        """Get list of active indicator names."""
        return [name for name, gene in self.genes.items() if gene.active]

    def get_weights(self) -> Dict[str, float]:
        """Get weights for active indicators."""
        return {
            name: gene.weight
            for name, gene in self.genes.items()
            if gene.active and gene.weight != 0
        }

    def to_weight_vector(self) -> np.ndarray:
        """Convert active weights to numpy array for genetic operations."""
        active_genes = [g for g in self.genes.values() if g.active]
        return np.array([g.weight for g in active_genes])

    def from_weight_vector(self, vector: np.ndarray):
        """Update weights from numpy array."""
        active_names = [n for n, g in self.genes.items() if g.active]
        for i, name in enumerate(active_names):
            if i < len(vector):
                self.genes[name].weight = float(np.clip(vector[i], -1.0, 1.0))

    def copy(self) -> 'SuperIndicatorDNA':
        """Create a deep copy of this DNA."""
        new_dna = SuperIndicatorDNA(
            dna_id=generate_dna_id(),
            generation=self.generation,
            run_id=self.run_id,
            parent_ids=(self.dna_id, ""),
            created_at=datetime.now().isoformat()
        )

        # Deep copy genes
        for name, gene in self.genes.items():
            new_dna.genes[name] = IndicatorGene(
                name=gene.name,
                weight=gene.weight,
                active=gene.active,
                category=gene.category
            )

        return new_dna

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dna_id': self.dna_id,
            'generation': self.generation,
            'run_id': self.run_id,
            'genes': {
                name: {
                    'name': gene.name,
                    'weight': gene.weight,
                    'active': gene.active,
                    'category': gene.category
                }
                for name, gene in self.genes.items()
            },
            'parent_ids': self.parent_ids,
            'mutation_history': self.mutation_history,
            'fitness_score': self.fitness_score,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'net_profit': self.net_profit,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'profit_factor': self.profit_factor,
            'validation_fitness': self.validation_fitness,
            'holdout_fitness': self.holdout_fitness,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuperIndicatorDNA':
        """Create DNA from dictionary."""
        dna = cls(
            dna_id=data.get('dna_id', generate_dna_id()),
            generation=data.get('generation', 0),
            run_id=data.get('run_id', 0),
            parent_ids=tuple(data.get('parent_ids', ("", ""))),
            mutation_history=data.get('mutation_history', []),
            fitness_score=data.get('fitness_score', 0.0),
            sharpe_ratio=data.get('sharpe_ratio', 0.0),
            max_drawdown=data.get('max_drawdown', 0.0),
            net_profit=data.get('net_profit', 0.0),
            win_rate=data.get('win_rate', 0.0),
            total_trades=data.get('total_trades', 0),
            long_trades=data.get('long_trades', 0),
            short_trades=data.get('short_trades', 0),
            profit_factor=data.get('profit_factor', 0.0),
            validation_fitness=data.get('validation_fitness'),
            holdout_fitness=data.get('holdout_fitness'),
            created_at=data.get('created_at', datetime.now().isoformat())
        )

        # Load genes
        genes_data = data.get('genes', {})
        for name, gene_data in genes_data.items():
            dna.genes[name] = IndicatorGene(
                name=gene_data.get('name', name),
                weight=gene_data.get('weight', 0.0),
                active=gene_data.get('active', True),
                category=gene_data.get('category', '')
            )

        return dna

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'SuperIndicatorDNA':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: str):
        """Save DNA to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: str) -> 'SuperIndicatorDNA':
        """Load DNA from file."""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())

    def summary(self) -> str:
        """Get human-readable summary of DNA."""
        active = self.get_active_indicators()
        top_weights = sorted(
            [(n, g.weight) for n, g in self.genes.items() if g.active],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        lines = [
            f"DNA: {self.dna_id} (Gen {self.generation})",
            f"Active Indicators: {len(active)}",
            f"Fitness: {self.fitness_score:.4f}",
            f"Sharpe: {self.sharpe_ratio:.2f}",
            f"Max DD: {self.max_drawdown:.2%}",
            f"Win Rate: {self.win_rate:.1%}",
            f"Trades: {self.total_trades} (L:{self.long_trades} S:{self.short_trades})",
            f"Top Weights:"
        ]
        for name, weight in top_weights:
            lines.append(f"  {name}: {weight:.3f}")

        return "\n".join(lines)


def create_random_dna(indicator_names: List[str],
                      categories: Dict[str, str] = None,
                      num_active: int = 20,
                      run_id: int = 0,
                      generation: int = 0) -> SuperIndicatorDNA:
    """
    Create a random DNA configuration.

    Args:
        indicator_names: List of all available indicators
        categories: Dict mapping indicator -> category
        num_active: Number of indicators to activate
        run_id: Evolution run ID
        generation: Generation number

    Returns:
        Randomly initialized DNA
    """
    dna = SuperIndicatorDNA(
        run_id=run_id,
        generation=generation
    )

    # Randomly select which indicators to activate
    num_active = min(num_active, len(indicator_names))
    active_indices = np.random.choice(
        len(indicator_names),
        size=num_active,
        replace=False
    )

    for i, name in enumerate(indicator_names):
        # Random weight in [-1, 1]
        weight = np.random.uniform(-1, 1) if i in active_indices else 0.0
        active = i in active_indices

        dna.genes[name] = IndicatorGene(
            name=name,
            weight=weight,
            active=active,
            category=categories.get(name, '') if categories else ''
        )

    return dna


def create_dna_from_weights(weights: Dict[str, float],
                            categories: Dict[str, str] = None,
                            run_id: int = 0,
                            generation: int = 0) -> SuperIndicatorDNA:
    """
    Create DNA from a weight dictionary.

    Args:
        weights: Dict of indicator_name -> weight
        categories: Dict mapping indicator -> category
        run_id: Evolution run ID
        generation: Generation number

    Returns:
        DNA with specified weights
    """
    dna = SuperIndicatorDNA(
        run_id=run_id,
        generation=generation
    )

    for name, weight in weights.items():
        dna.genes[name] = IndicatorGene(
            name=name,
            weight=weight,
            active=weight != 0,
            category=categories.get(name, '') if categories else ''
        )

    return dna
