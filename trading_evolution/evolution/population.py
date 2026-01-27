"""
Population Management module.

Manages populations of Super Indicator DNA configurations.
"""

import random
from typing import List, Dict, Optional, Set
import numpy as np
import logging

from ..super_indicator.dna import SuperIndicatorDNA, IndicatorGene, create_random_dna
from ..indicators.universe import IndicatorUniverse

logger = logging.getLogger(__name__)


class PopulationManager:
    """
    Manages populations of DNA configurations for genetic evolution.

    Responsibilities:
    - Create initial population
    - Track population statistics
    - Manage diversity
    - Handle population selection
    """

    def __init__(self,
                 universe: IndicatorUniverse = None,
                 min_indicators: int = 10,
                 max_indicators: int = 30):
        """
        Initialize population manager.

        Args:
            universe: Indicator universe for creating DNA
            min_indicators: Minimum active indicators per DNA
            max_indicators: Maximum active indicators per DNA
        """
        self.universe = universe or IndicatorUniverse()
        self.universe.load_all()

        self.min_indicators = min_indicators
        self.max_indicators = max_indicators

        # Get all indicator names and categories
        self.all_indicators = self.universe.get_all()
        self.categories = {
            name: self.universe.get_definition(name).category
            for name in self.all_indicators
            if self.universe.get_definition(name)
        }

    def create_initial_population(self,
                                  size: int,
                                  run_id: int = 0,
                                  ensure_diversity: bool = True) -> List[SuperIndicatorDNA]:
        """
        Create initial population of random DNA configurations.

        Args:
            size: Population size
            run_id: Evolution run ID
            ensure_diversity: Ensure category diversity in each DNA

        Returns:
            List of DNA configurations
        """
        population = []

        for i in range(size):
            if ensure_diversity:
                dna = self._create_diverse_dna(run_id, generation=0)
            else:
                dna = create_random_dna(
                    self.all_indicators,
                    self.categories,
                    num_active=random.randint(self.min_indicators, self.max_indicators),
                    run_id=run_id,
                    generation=0
                )
            population.append(dna)

        logger.info(f"Created initial population of {size} DNA configurations")
        return population

    def _create_diverse_dna(self, run_id: int, generation: int) -> SuperIndicatorDNA:
        """Create DNA with diverse category representation."""
        dna = SuperIndicatorDNA(run_id=run_id, generation=generation)

        # Get indicators by category
        by_category = {}
        for name in self.all_indicators:
            cat = self.categories.get(name, 'other')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(name)

        # Select some from each category
        num_per_category = max(2, self.max_indicators // len(by_category))
        selected = set()

        for category, indicators in by_category.items():
            n_select = min(num_per_category, len(indicators))
            chosen = random.sample(indicators, n_select)
            selected.update(chosen)

        # Add all indicators to DNA
        for name in self.all_indicators:
            active = name in selected
            weight = np.random.uniform(-1, 1) if active else 0.0

            dna.genes[name] = IndicatorGene(
                name=name,
                weight=weight,
                active=active,
                category=self.categories.get(name, '')
            )

        return dna

    def get_population_stats(self, population: List[SuperIndicatorDNA]) -> Dict:
        """Get statistics about the population."""
        if not population:
            return {}

        fitnesses = [d.fitness_score for d in population]
        active_counts = [len(d.get_active_indicators()) for d in population]

        return {
            'size': len(population),
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'worst_fitness': min(fitnesses),
            'avg_active_indicators': np.mean(active_counts),
            'diversity': self._calculate_diversity(population)
        }

    def _calculate_diversity(self, population: List[SuperIndicatorDNA]) -> float:
        """
        Calculate population diversity.

        Higher diversity = population members are more different.
        """
        if len(population) < 2:
            return 1.0

        # Compare weight vectors
        distances = []
        for i, dna1 in enumerate(population[:50]):  # Sample for efficiency
            for dna2 in population[i + 1:50]:
                vec1 = dna1.to_weight_vector()
                vec2 = dna2.to_weight_vector()

                if len(vec1) == len(vec2):
                    dist = np.linalg.norm(vec1 - vec2)
                    distances.append(dist)

        if not distances:
            return 0.0

        return np.mean(distances)

    def select_top(self, population: List[SuperIndicatorDNA],
                   n: int) -> List[SuperIndicatorDNA]:
        """Select top N by fitness."""
        sorted_pop = sorted(population, key=lambda d: d.fitness_score, reverse=True)
        return sorted_pop[:n]

    def select_bottom(self, population: List[SuperIndicatorDNA],
                      n: int) -> List[SuperIndicatorDNA]:
        """Select bottom N by fitness (for removal/replacement)."""
        sorted_pop = sorted(population, key=lambda d: d.fitness_score)
        return sorted_pop[:n]

    def maintain_diversity(self,
                           population: List[SuperIndicatorDNA],
                           min_diversity: float = 0.5) -> List[SuperIndicatorDNA]:
        """
        Ensure population maintains minimum diversity.

        If diversity is too low, inject random DNA.
        """
        diversity = self._calculate_diversity(population)

        if diversity >= min_diversity:
            return population

        logger.info(f"Low diversity ({diversity:.3f}), injecting random DNA")

        # Replace bottom 10% with random DNA
        n_replace = max(1, len(population) // 10)
        keep = self.select_top(population, len(population) - n_replace)

        for _ in range(n_replace):
            new_dna = self._create_diverse_dna(
                run_id=population[0].run_id if population else 0,
                generation=population[0].generation + 1 if population else 0
            )
            keep.append(new_dna)

        return keep

    def get_unused_indicators(self, population: List[SuperIndicatorDNA]) -> List[str]:
        """Get indicators not currently used in any DNA."""
        used = set()
        for dna in population:
            used.update(dna.get_active_indicators())

        unused = [name for name in self.all_indicators if name not in used]
        return unused

    def inject_new_indicators(self,
                              population: List[SuperIndicatorDNA],
                              n_indicators: int = 3) -> List[SuperIndicatorDNA]:
        """
        Inject previously unused indicators into some population members.

        Promotes exploration of new indicator combinations.
        """
        unused = self.get_unused_indicators(population)

        if not unused:
            logger.info("All indicators are in use")
            return population

        # Select random unused indicators to try
        to_inject = random.sample(unused, min(n_indicators, len(unused)))
        logger.info(f"Injecting {len(to_inject)} new indicators: {to_inject}")

        # Add to bottom 10% of population
        bottom = self.select_bottom(population, max(1, len(population) // 10))

        for dna in bottom:
            for name in to_inject:
                if name in dna.genes:
                    dna.genes[name].active = True
                    dna.genes[name].weight = np.random.uniform(-1, 1)

        return population

    def create_checkpoint(self, population: List[SuperIndicatorDNA]) -> List[Dict]:
        """Create a serializable checkpoint of population."""
        return [dna.to_dict() for dna in population]

    def restore_checkpoint(self, checkpoint: List[Dict]) -> List[SuperIndicatorDNA]:
        """Restore population from checkpoint."""
        return [SuperIndicatorDNA.from_dict(d) for d in checkpoint]
