"""
Super Indicator Evolution module.

Implements genetic operators for evolving Super Indicator DNA:
- Mutation: Random weight adjustments
- Crossover: Combine DNA from two parents
- Selection: Choose best performing DNA
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from copy import deepcopy

from .dna import SuperIndicatorDNA, IndicatorGene, generate_dna_id


class MutationOperator:
    """
    Mutation operators for DNA evolution.

    Available mutations:
    - Weight adjustment: Random change to weights
    - Indicator swap: Enable/disable indicators
    - Reset: Reset weight to random value
    """

    def __init__(self,
                 weight_mutation_rate: float = 0.2,
                 weight_mutation_strength: float = 0.15,
                 indicator_swap_rate: float = 0.1,
                 reset_rate: float = 0.05):
        """
        Initialize mutation operator.

        Args:
            weight_mutation_rate: Probability of mutating each weight
            weight_mutation_strength: Maximum weight change (as fraction)
            indicator_swap_rate: Probability of toggling indicator active state
            reset_rate: Probability of resetting weight to random value
        """
        self.weight_mutation_rate = weight_mutation_rate
        self.weight_mutation_strength = weight_mutation_strength
        self.indicator_swap_rate = indicator_swap_rate
        self.reset_rate = reset_rate

    def mutate(self, dna: SuperIndicatorDNA,
               generation: int = None) -> SuperIndicatorDNA:
        """
        Apply mutations to DNA.

        Args:
            dna: DNA to mutate
            generation: New generation number

        Returns:
            Mutated DNA (new instance)
        """
        # Create copy
        mutated = dna.copy()
        if generation is not None:
            mutated.generation = generation
        mutated.parent_ids = (dna.dna_id, "")

        mutations_applied = []

        for name, gene in mutated.genes.items():
            # Weight mutation
            if random.random() < self.weight_mutation_rate:
                if random.random() < self.reset_rate:
                    # Reset to random value
                    gene.weight = np.random.uniform(-1, 1)
                    mutations_applied.append(f"reset_{name}")
                else:
                    # Gaussian mutation
                    delta = np.random.normal(0, self.weight_mutation_strength)
                    gene.weight = np.clip(gene.weight + delta, -1, 1)
                    mutations_applied.append(f"weight_{name}")

            # Indicator swap
            if random.random() < self.indicator_swap_rate:
                gene.active = not gene.active
                if not gene.active:
                    gene.weight = 0.0
                else:
                    gene.weight = np.random.uniform(-1, 1)
                mutations_applied.append(f"swap_{name}")

        mutated.mutation_history = mutations_applied
        return mutated

    def mutate_weights_only(self, dna: SuperIndicatorDNA,
                            mutation_rate: float = None,
                            mutation_strength: float = None) -> SuperIndicatorDNA:
        """Mutate only weights (no indicator swapping)."""
        mutated = dna.copy()
        rate = mutation_rate or self.weight_mutation_rate
        strength = mutation_strength or self.weight_mutation_strength

        for name, gene in mutated.genes.items():
            if gene.active and random.random() < rate:
                delta = np.random.normal(0, strength)
                gene.weight = np.clip(gene.weight + delta, -1, 1)

        return mutated


class CrossoverOperator:
    """
    Crossover operators for combining DNA from two parents.

    Available crossover types:
    - Uniform: Each gene randomly from parent1 or parent2
    - Blend: Weighted average of parent weights
    - Category-based: Preserve category groupings
    """

    def __init__(self, blend_alpha: float = 0.5):
        """
        Initialize crossover operator.

        Args:
            blend_alpha: Blending factor for blend crossover
        """
        self.blend_alpha = blend_alpha

    def uniform_crossover(self, parent1: SuperIndicatorDNA,
                          parent2: SuperIndicatorDNA,
                          generation: int = None) -> SuperIndicatorDNA:
        """
        Uniform crossover: each gene randomly from parent1 or parent2.

        Args:
            parent1: First parent DNA
            parent2: Second parent DNA
            generation: New generation number

        Returns:
            Child DNA
        """
        child = SuperIndicatorDNA(
            generation=generation or max(parent1.generation, parent2.generation) + 1,
            run_id=parent1.run_id,
            parent_ids=(parent1.dna_id, parent2.dna_id)
        )

        # Get all gene names from both parents
        all_names = set(parent1.genes.keys()) | set(parent2.genes.keys())

        for name in all_names:
            gene1 = parent1.genes.get(name)
            gene2 = parent2.genes.get(name)

            if gene1 and gene2:
                # Both parents have this gene - randomly select
                source_gene = gene1 if random.random() < 0.5 else gene2
            elif gene1:
                source_gene = gene1
            else:
                source_gene = gene2

            child.genes[name] = IndicatorGene(
                name=source_gene.name,
                weight=source_gene.weight,
                active=source_gene.active,
                category=source_gene.category
            )

        return child

    def blend_crossover(self, parent1: SuperIndicatorDNA,
                        parent2: SuperIndicatorDNA,
                        generation: int = None) -> SuperIndicatorDNA:
        """
        Blend crossover: weighted average of parent weights.

        Args:
            parent1: First parent DNA
            parent2: Second parent DNA
            generation: New generation number

        Returns:
            Child DNA with blended weights
        """
        child = SuperIndicatorDNA(
            generation=generation or max(parent1.generation, parent2.generation) + 1,
            run_id=parent1.run_id,
            parent_ids=(parent1.dna_id, parent2.dna_id)
        )

        all_names = set(parent1.genes.keys()) | set(parent2.genes.keys())

        for name in all_names:
            gene1 = parent1.genes.get(name)
            gene2 = parent2.genes.get(name)

            if gene1 and gene2:
                # Blend weights with random alpha
                alpha = np.random.uniform(
                    max(0, self.blend_alpha - 0.25),
                    min(1, self.blend_alpha + 0.25)
                )
                blended_weight = alpha * gene1.weight + (1 - alpha) * gene2.weight

                # Active if either parent is active
                active = gene1.active or gene2.active

                child.genes[name] = IndicatorGene(
                    name=name,
                    weight=blended_weight if active else 0.0,
                    active=active,
                    category=gene1.category or gene2.category
                )
            elif gene1:
                child.genes[name] = IndicatorGene(
                    name=gene1.name,
                    weight=gene1.weight,
                    active=gene1.active,
                    category=gene1.category
                )
            else:
                child.genes[name] = IndicatorGene(
                    name=gene2.name,
                    weight=gene2.weight,
                    active=gene2.active,
                    category=gene2.category
                )

        return child

    def category_crossover(self, parent1: SuperIndicatorDNA,
                           parent2: SuperIndicatorDNA,
                           generation: int = None) -> SuperIndicatorDNA:
        """
        Category-based crossover: entire categories from one parent.

        Preserves indicator relationships within categories.
        """
        child = SuperIndicatorDNA(
            generation=generation or max(parent1.generation, parent2.generation) + 1,
            run_id=parent1.run_id,
            parent_ids=(parent1.dna_id, parent2.dna_id)
        )

        # Group genes by category
        categories1 = {}
        categories2 = {}

        for name, gene in parent1.genes.items():
            cat = gene.category or 'other'
            if cat not in categories1:
                categories1[cat] = {}
            categories1[cat][name] = gene

        for name, gene in parent2.genes.items():
            cat = gene.category or 'other'
            if cat not in categories2:
                categories2[cat] = {}
            categories2[cat][name] = gene

        # For each category, randomly select from parent1 or parent2
        all_categories = set(categories1.keys()) | set(categories2.keys())

        for category in all_categories:
            if random.random() < 0.5 and category in categories1:
                source_genes = categories1[category]
            elif category in categories2:
                source_genes = categories2[category]
            elif category in categories1:
                source_genes = categories1[category]
            else:
                continue

            for name, gene in source_genes.items():
                child.genes[name] = IndicatorGene(
                    name=gene.name,
                    weight=gene.weight,
                    active=gene.active,
                    category=gene.category
                )

        return child


class SelectionOperator:
    """
    Selection operators for choosing parents from population.
    """

    @staticmethod
    def tournament_selection(population: List[SuperIndicatorDNA],
                             tournament_size: int = 3) -> SuperIndicatorDNA:
        """
        Tournament selection: random subset, return best.

        Args:
            population: List of DNA to select from
            tournament_size: Number of competitors

        Returns:
            Winner of tournament
        """
        competitors = random.sample(population, min(tournament_size, len(population)))
        return max(competitors, key=lambda d: d.fitness_score)

    @staticmethod
    def roulette_selection(population: List[SuperIndicatorDNA]) -> SuperIndicatorDNA:
        """
        Roulette wheel selection: probability proportional to fitness.
        """
        fitnesses = [max(0.001, d.fitness_score) for d in population]
        total = sum(fitnesses)
        probabilities = [f / total for f in fitnesses]

        return np.random.choice(population, p=probabilities)

    @staticmethod
    def elite_selection(population: List[SuperIndicatorDNA],
                        n_elite: int = 5) -> List[SuperIndicatorDNA]:
        """
        Elite selection: return top N performers.

        Args:
            population: List of DNA
            n_elite: Number of elite to return

        Returns:
            List of top performing DNA
        """
        sorted_pop = sorted(population, key=lambda d: d.fitness_score, reverse=True)
        return sorted_pop[:n_elite]


def apply_coach_guidance(dna: SuperIndicatorDNA,
                         promoted: List[str],
                         demoted: List[str],
                         removed: List[str],
                         weight_adjustments: Dict[str, float]) -> SuperIndicatorDNA:
    """
    Apply Coach recommendations to DNA.

    Args:
        dna: DNA to modify
        promoted: Indicators to increase weight
        demoted: Indicators to decrease weight
        removed: Indicators to deactivate
        weight_adjustments: Specific weight changes

    Returns:
        Modified DNA
    """
    modified = dna.copy()

    # Apply removals
    for name in removed:
        if name in modified.genes:
            modified.genes[name].active = False
            modified.genes[name].weight = 0.0

    # Apply demotions (reduce weight by 30%)
    for name in demoted:
        if name in modified.genes and modified.genes[name].active:
            modified.genes[name].weight *= 0.7

    # Apply promotions (increase weight by 30%)
    for name in promoted:
        if name in modified.genes and modified.genes[name].active:
            current = modified.genes[name].weight
            # Increase magnitude while preserving sign
            if current > 0:
                modified.genes[name].weight = min(1.0, current * 1.3)
            elif current < 0:
                modified.genes[name].weight = max(-1.0, current * 1.3)

    # Apply specific adjustments
    for name, new_weight in weight_adjustments.items():
        if name in modified.genes:
            modified.genes[name].weight = np.clip(new_weight, -1, 1)
            modified.genes[name].active = new_weight != 0

    return modified
