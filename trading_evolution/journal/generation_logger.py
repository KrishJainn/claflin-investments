"""
Generation Logger module.

Logs generation-level statistics and summaries.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

from .database import Database
from ..super_indicator.dna import SuperIndicatorDNA

logger = logging.getLogger(__name__)


class GenerationLogger:
    """
    Logs generation-level statistics to the database.

    Tracks:
    - Population fitness statistics
    - Best DNA configuration
    - Evolution progress
    - Coach recommendations applied
    """

    def __init__(self, database: Database):
        """
        Initialize generation logger.

        Args:
            database: Database instance for persistence
        """
        self.db = database

    def log_generation(self,
                       run_id: int,
                       generation: int,
                       best_dna: SuperIndicatorDNA,
                       population_stats: Dict,
                       trade_stats: Dict,
                       coach_applied: bool = False,
                       validation_fitness: float = None,
                       notes: str = None):
        """
        Log a generation's results.

        Args:
            run_id: Evolution run ID
            generation: Generation number
            best_dna: Best DNA from this generation
            population_stats: Population statistics (avg, std, etc.)
            trade_stats: Trading statistics (win rate, profit, etc.)
            coach_applied: Whether Coach recommendations were applied
            validation_fitness: Validation set fitness (if checked)
            notes: Additional notes
        """
        # Prepare generation data
        gen_data = {
            'run_id': run_id,
            'generation_num': generation,
            'best_fitness': best_dna.fitness_score,
            'avg_fitness': population_stats.get('avg_fitness', 0),
            'std_fitness': population_stats.get('std_fitness', 0),
            'best_sharpe': best_dna.sharpe_ratio,
            'best_profit': best_dna.net_profit,
            'best_drawdown': best_dna.max_drawdown,
            'best_win_rate': best_dna.win_rate,
            'total_trades': trade_stats.get('total_trades', 0),
            'diversity_score': population_stats.get('diversity', 0),
            'validation_fitness': validation_fitness,
            'created_at': datetime.now().isoformat()
        }

        # Save generation stats
        self.db.save_generation(gen_data)

        # Save best DNA configuration
        self.db.save_dna_config(
            run_id=run_id,
            generation=generation,
            dna=best_dna
        )

        logger.info(
            f"Gen {generation}: Best={best_dna.fitness_score:.4f}, "
            f"Avg={population_stats.get('avg_fitness', 0):.4f}, "
            f"Sharpe={best_dna.sharpe_ratio:.2f}, "
            f"Profit=${best_dna.net_profit:.0f}"
        )

    def log_generation_from_evolution(self,
                                       run_id: int,
                                       generation: int,
                                       population: List[SuperIndicatorDNA],
                                       trades: List[Dict],
                                       validation_fitness: float = None):
        """
        Log generation from evolution data (convenience method).

        Args:
            run_id: Evolution run ID
            generation: Generation number
            population: Current population
            trades: Trades from this generation
            validation_fitness: Validation set fitness
        """
        import numpy as np

        # Calculate population stats
        fitnesses = [d.fitness_score for d in population]
        population_stats = {
            'avg_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'min_fitness': np.min(fitnesses),
            'max_fitness': np.max(fitnesses),
            'diversity': self._calculate_diversity(population)
        }

        # Calculate trade stats
        if trades:
            pnls = [t.get('net_pnl', 0) for t in trades]
            winners = [p for p in pnls if p > 0]
            trade_stats = {
                'total_trades': len(trades),
                'win_rate': len(winners) / len(trades) if trades else 0,
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls) if pnls else 0
            }
        else:
            trade_stats = {'total_trades': 0}

        # Get best DNA
        best_dna = max(population, key=lambda d: d.fitness_score)

        self.log_generation(
            run_id=run_id,
            generation=generation,
            best_dna=best_dna,
            population_stats=population_stats,
            trade_stats=trade_stats,
            validation_fitness=validation_fitness
        )

    def _calculate_diversity(self, population: List[SuperIndicatorDNA]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 1.0

        import numpy as np

        distances = []
        for i, dna1 in enumerate(population[:20]):  # Sample for efficiency
            for dna2 in population[i + 1:20]:
                vec1 = dna1.to_weight_vector()
                vec2 = dna2.to_weight_vector()
                if len(vec1) == len(vec2):
                    dist = np.linalg.norm(vec1 - vec2)
                    distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def get_generation(self, run_id: int, generation: int) -> Optional[Dict]:
        """
        Get a specific generation's data.

        Args:
            run_id: Evolution run ID
            generation: Generation number

        Returns:
            Generation data dictionary or None
        """
        generations = self.db.get_generations(run_id)
        for gen in generations:
            if gen.get('generation_num') == generation:
                return gen
        return None

    def get_all_generations(self, run_id: int) -> List[Dict]:
        """
        Get all generations for a run.

        Args:
            run_id: Evolution run ID

        Returns:
            List of generation data dictionaries
        """
        return self.db.get_generations(run_id)

    def get_evolution_progress(self, run_id: int) -> Dict:
        """
        Get evolution progress summary.

        Args:
            run_id: Evolution run ID

        Returns:
            Dictionary with progress metrics
        """
        generations = self.get_all_generations(run_id)

        if not generations:
            return {'generations': 0}

        first = generations[0]
        last = generations[-1]

        fitnesses = [g.get('best_fitness', 0) for g in generations]
        sharpes = [g.get('best_sharpe', 0) for g in generations]
        profits = [g.get('best_profit', 0) for g in generations]

        return {
            'generations': len(generations),
            'starting_fitness': first.get('best_fitness', 0),
            'current_fitness': last.get('best_fitness', 0),
            'fitness_improvement': last.get('best_fitness', 0) - first.get('best_fitness', 0),
            'best_fitness_ever': max(fitnesses),
            'best_sharpe_ever': max(sharpes),
            'best_profit_ever': max(profits),
            'avg_fitness_trend': self._calculate_trend(fitnesses),
            'current_sharpe': last.get('best_sharpe', 0),
            'current_profit': last.get('best_profit', 0),
            'current_win_rate': last.get('best_win_rate', 0)
        }

    def _calculate_trend(self, values: List[float], window: int = 5) -> str:
        """Calculate trend direction from values."""
        if len(values) < window:
            return 'insufficient_data'

        recent = values[-window:]
        earlier = values[-2*window:-window] if len(values) >= 2*window else values[:window]

        import numpy as np
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)

        if recent_avg > earlier_avg * 1.05:
            return 'improving'
        elif recent_avg < earlier_avg * 0.95:
            return 'declining'
        else:
            return 'stable'

    def get_generation_summary(self, run_id: int, generation: int) -> str:
        """
        Get human-readable generation summary.

        Args:
            run_id: Evolution run ID
            generation: Generation number

        Returns:
            Formatted summary string
        """
        gen = self.get_generation(run_id, generation)

        if not gen:
            return f"Generation {generation} not found"

        lines = [
            f"Generation {generation} Summary",
            "=" * 40,
            f"Best Fitness: {gen.get('best_fitness', 0):.4f}",
            f"Avg Fitness:  {gen.get('avg_fitness', 0):.4f}",
            f"Std Fitness:  {gen.get('std_fitness', 0):.4f}",
            "",
            f"Best Sharpe:    {gen.get('best_sharpe', 0):.2f}",
            f"Best Profit:    ${gen.get('best_profit', 0):.0f}",
            f"Best Drawdown:  {gen.get('best_drawdown', 0):.1%}",
            f"Best Win Rate:  {gen.get('best_win_rate', 0):.1%}",
            "",
            f"Total Trades:   {gen.get('total_trades', 0)}",
            f"Diversity:      {gen.get('diversity_score', 0):.3f}",
        ]

        if gen.get('validation_fitness'):
            lines.append(f"Validation:     {gen.get('validation_fitness'):.4f}")

        return "\n".join(lines)

    def get_run_summary(self, run_id: int) -> str:
        """
        Get complete run summary.

        Args:
            run_id: Evolution run ID

        Returns:
            Formatted summary string
        """
        progress = self.get_evolution_progress(run_id)

        if progress.get('generations', 0) == 0:
            return "No generations found for this run"

        lines = [
            f"Evolution Run {run_id} Summary",
            "=" * 50,
            f"Generations Completed: {progress['generations']}",
            "",
            "Fitness Progress:",
            f"  Starting: {progress['starting_fitness']:.4f}",
            f"  Current:  {progress['current_fitness']:.4f}",
            f"  Best:     {progress['best_fitness_ever']:.4f}",
            f"  Change:   {progress['fitness_improvement']:+.4f}",
            f"  Trend:    {progress['avg_fitness_trend']}",
            "",
            "Current Best Strategy:",
            f"  Sharpe Ratio: {progress['current_sharpe']:.2f}",
            f"  Net Profit:   ${progress['current_profit']:.0f}",
            f"  Win Rate:     {progress['current_win_rate']:.1%}",
            "",
            "All-Time Bests:",
            f"  Best Sharpe:  {progress['best_sharpe_ever']:.2f}",
            f"  Best Profit:  ${progress['best_profit_ever']:.0f}",
        ]

        return "\n".join(lines)

    def compare_generations(self, run_id: int, gen1: int, gen2: int) -> Dict:
        """
        Compare two generations.

        Args:
            run_id: Evolution run ID
            gen1: First generation
            gen2: Second generation

        Returns:
            Dictionary with comparison metrics
        """
        g1 = self.get_generation(run_id, gen1)
        g2 = self.get_generation(run_id, gen2)

        if not g1 or not g2:
            return {'error': 'Generation not found'}

        return {
            'fitness_change': g2.get('best_fitness', 0) - g1.get('best_fitness', 0),
            'sharpe_change': g2.get('best_sharpe', 0) - g1.get('best_sharpe', 0),
            'profit_change': g2.get('best_profit', 0) - g1.get('best_profit', 0),
            'win_rate_change': g2.get('best_win_rate', 0) - g1.get('best_win_rate', 0),
            'drawdown_change': g2.get('best_drawdown', 0) - g1.get('best_drawdown', 0),
            'diversity_change': g2.get('diversity_score', 0) - g1.get('diversity_score', 0),
            'improved': g2.get('best_fitness', 0) > g1.get('best_fitness', 0)
        }
