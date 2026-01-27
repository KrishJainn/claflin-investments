"""
Indicator Importance Visualization module.

Visualizes indicator weights and importance over generations.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class IndicatorImportanceChart:
    """
    Generates visualizations for indicator importance and evolution.

    Features:
    - Weight heatmaps over generations
    - Top indicator bar charts
    - Weight evolution line plots
    - Category breakdown
    """

    def __init__(self,
                 figsize: Tuple[int, int] = (14, 10),
                 colormap: str = 'RdYlGn'):
        """
        Initialize indicator importance chart generator.

        Args:
            figsize: Figure size (width, height)
            colormap: Colormap for heatmaps
        """
        self.figsize = figsize
        self.colormap = colormap

    def plot_weight_heatmap(self,
                            weight_history: Dict[int, Dict[str, float]],
                            save_path: str = None,
                            title: str = "Indicator Weights Over Generations",
                            top_n: int = 30) -> str:
        """
        Plot heatmap of indicator weights over generations.

        Args:
            weight_history: Dict mapping generation to weights dict
            save_path: Path to save PNG
            title: Chart title
            top_n: Number of top indicators to show

        Returns:
            Path to saved file
        """
        # Convert to DataFrame
        generations = sorted(weight_history.keys())
        all_indicators = set()
        for weights in weight_history.values():
            all_indicators.update(weights.keys())

        # Calculate average absolute weight for each indicator
        avg_weights = {}
        for indicator in all_indicators:
            values = [abs(weight_history[g].get(indicator, 0)) for g in generations]
            avg_weights[indicator] = np.mean(values)

        # Select top N by average weight
        top_indicators = sorted(avg_weights.keys(),
                               key=lambda x: avg_weights[x],
                               reverse=True)[:top_n]

        # Build data matrix
        data = np.zeros((len(top_indicators), len(generations)))
        for j, gen in enumerate(generations):
            for i, indicator in enumerate(top_indicators):
                data[i, j] = weight_history[gen].get(indicator, 0)

        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(data, aspect='auto', cmap=self.colormap,
                      vmin=-1, vmax=1)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Weight')
        cbar.ax.set_ylabel('Weight', fontsize=11)

        # Labels
        ax.set_yticks(range(len(top_indicators)))
        ax.set_yticklabels(top_indicators, fontsize=8)

        # Show every N generations on x-axis
        gen_step = max(1, len(generations) // 15)
        ax.set_xticks(range(0, len(generations), gen_step))
        ax.set_xticklabels([str(generations[i]) for i in range(0, len(generations), gen_step)])

        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Indicator', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path is None:
            save_path = 'weight_heatmap.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        logger.info(f"Saved weight heatmap to {save_path}")
        return save_path

    def plot_top_indicators(self,
                            weights: Dict[str, float],
                            save_path: str = None,
                            title: str = "Top Indicators by Weight",
                            top_n: int = 20) -> str:
        """
        Plot horizontal bar chart of top indicators.

        Args:
            weights: Dictionary of indicator weights
            save_path: Path to save PNG
            title: Chart title
            top_n: Number of indicators to show

        Returns:
            Path to saved file
        """
        # Sort by absolute weight
        sorted_indicators = sorted(weights.items(),
                                  key=lambda x: abs(x[1]),
                                  reverse=True)[:top_n]

        names = [x[0] for x in sorted_indicators]
        values = [x[1] for x in sorted_indicators]

        # Color by sign
        colors = ['green' if v > 0 else 'red' for v in values]

        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()

        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Weight', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, (name, value) in enumerate(zip(names, values)):
            ax.text(value + 0.02 if value >= 0 else value - 0.02,
                   i, f'{value:.3f}',
                   va='center', ha='left' if value >= 0 else 'right',
                   fontsize=8)

        plt.tight_layout()

        if save_path is None:
            save_path = 'top_indicators.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        logger.info(f"Saved top indicators chart to {save_path}")
        return save_path

    def plot_weight_evolution(self,
                              weight_history: Dict[int, Dict[str, float]],
                              indicators: List[str] = None,
                              save_path: str = None,
                              title: str = "Indicator Weight Evolution",
                              top_n: int = 10) -> str:
        """
        Plot line chart of indicator weight evolution.

        Args:
            weight_history: Dict mapping generation to weights dict
            indicators: Specific indicators to plot (or auto-select top)
            save_path: Path to save PNG
            title: Chart title
            top_n: Number of indicators if auto-selecting

        Returns:
            Path to saved file
        """
        generations = sorted(weight_history.keys())

        # Auto-select top indicators if not specified
        if indicators is None:
            # Get indicators with most variation
            all_indicators = set()
            for weights in weight_history.values():
                all_indicators.update(weights.keys())

            variations = {}
            for indicator in all_indicators:
                values = [weight_history[g].get(indicator, 0) for g in generations]
                variations[indicator] = np.std(values) + abs(np.mean(values))

            indicators = sorted(variations.keys(),
                              key=lambda x: variations[x],
                              reverse=True)[:top_n]

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.tab20(np.linspace(0, 1, len(indicators)))

        for i, indicator in enumerate(indicators):
            values = [weight_history[g].get(indicator, 0) for g in generations]
            ax.plot(generations, values, color=colors[i], linewidth=1.5,
                   marker='o', markersize=3, label=indicator)

        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = 'weight_evolution.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        logger.info(f"Saved weight evolution to {save_path}")
        return save_path

    def plot_category_breakdown(self,
                                weights: Dict[str, float],
                                categories: Dict[str, str],
                                save_path: str = None,
                                title: str = "Indicator Category Breakdown") -> str:
        """
        Plot breakdown of weights by indicator category.

        Args:
            weights: Dictionary of indicator weights
            categories: Dict mapping indicator name to category
            save_path: Path to save PNG
            title: Chart title

        Returns:
            Path to saved file
        """
        # Aggregate by category
        category_weights = {}
        category_counts = {}

        for indicator, weight in weights.items():
            if abs(weight) > 0.01:  # Only count active indicators
                category = categories.get(indicator, 'other')
                if category not in category_weights:
                    category_weights[category] = 0
                    category_counts[category] = 0
                category_weights[category] += abs(weight)
                category_counts[category] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Pie chart of total weights
        cats = list(category_weights.keys())
        weights_list = [category_weights[c] for c in cats]

        colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))

        ax1.pie(weights_list, labels=cats, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax1.set_title('Total Weight by Category', fontsize=12, fontweight='bold')

        # Bar chart of counts
        ax2.bar(cats, [category_counts[c] for c in cats], color=colors,
               edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Number of Active Indicators', fontsize=11)
        ax2.set_title('Active Indicators by Category', fontsize=12, fontweight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path is None:
            save_path = 'category_breakdown.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        logger.info(f"Saved category breakdown to {save_path}")
        return save_path

    def plot_indicator_performance(self,
                                   performance_data: List[Dict],
                                   save_path: str = None,
                                   title: str = "Indicator Performance Scores",
                                   top_n: int = 20) -> str:
        """
        Plot indicator performance scores.

        Args:
            performance_data: List of dicts with 'name', 'score', 'accuracy' etc.
            save_path: Path to save PNG
            title: Chart title
            top_n: Number of indicators to show

        Returns:
            Path to saved file
        """
        # Sort by score
        sorted_data = sorted(performance_data,
                            key=lambda x: x.get('score', 0),
                            reverse=True)[:top_n]

        names = [d['name'] for d in sorted_data]
        scores = [d.get('score', 0) for d in sorted_data]
        accuracies = [d.get('accuracy', 0) for d in sorted_data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        y_pos = np.arange(len(names))

        # Score bar chart
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        ax1.barh(y_pos, scores, color=colors, alpha=0.8,
                edgecolor='black', linewidth=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Performance Score', fontsize=11)
        ax1.set_title('Quality Score', fontsize=12, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)

        # Accuracy bar chart
        colors_acc = ['green' if a > 0.5 else 'red' for a in accuracies]
        ax2.barh(y_pos, [a * 100 for a in accuracies], color=colors_acc,
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axvline(x=50, color='black', linewidth=1, linestyle='--')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Prediction Accuracy (%)', fontsize=11)
        ax2.set_title('Predictive Accuracy', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='x', alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path is None:
            save_path = 'indicator_performance.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        logger.info(f"Saved indicator performance to {save_path}")
        return save_path

    def plot_fitness_evolution(self,
                               generation_stats: List[Dict],
                               save_path: str = None,
                               title: str = "Fitness Evolution") -> str:
        """
        Plot fitness metrics over generations.

        Args:
            generation_stats: List of dicts with 'generation', 'best_fitness', etc.
            save_path: Path to save PNG
            title: Chart title

        Returns:
            Path to saved file
        """
        generations = [s['generation'] for s in generation_stats]
        best_fitness = [s.get('best_fitness', 0) for s in generation_stats]
        avg_fitness = [s.get('avg_fitness', 0) for s in generation_stats]
        std_fitness = [s.get('std_fitness', 0) for s in generation_stats]

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot best fitness
        ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best', marker='o', markersize=4)

        # Plot average with std band
        avg = np.array(avg_fitness)
        std = np.array(std_fitness)
        ax.plot(generations, avg_fitness, 'g--', linewidth=1.5, label='Average')
        ax.fill_between(generations, avg - std, avg + std, alpha=0.2, color='green')

        # Validation fitness if available
        val_fitness = [s.get('validation_fitness') for s in generation_stats]
        if any(v is not None for v in val_fitness):
            valid_gens = [g for g, v in zip(generations, val_fitness) if v is not None]
            valid_vals = [v for v in val_fitness if v is not None]
            ax.plot(valid_gens, valid_vals, 'r^', markersize=8, label='Validation')

        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Fitness', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = 'fitness_evolution.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        logger.info(f"Saved fitness evolution to {save_path}")
        return save_path


def create_indicator_report(weight_history: Dict[int, Dict[str, float]],
                            categories: Dict[str, str],
                            generation_stats: List[Dict],
                            output_dir: str) -> Dict[str, str]:
    """
    Create complete indicator importance report.

    Args:
        weight_history: Dict mapping generation to weights
        categories: Dict mapping indicator to category
        generation_stats: List of generation statistics
        output_dir: Directory to save charts

    Returns:
        Dictionary mapping chart name to file path
    """
    chart = IndicatorImportanceChart()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {}

    # Get final weights
    final_gen = max(weight_history.keys())
    final_weights = weight_history[final_gen]

    # Weight heatmap
    files['weight_heatmap'] = chart.plot_weight_heatmap(
        weight_history=weight_history,
        save_path=str(output_path / 'weight_heatmap.png'),
        top_n=30
    )

    # Top indicators
    files['top_indicators'] = chart.plot_top_indicators(
        weights=final_weights,
        save_path=str(output_path / 'top_indicators.png'),
        top_n=20
    )

    # Weight evolution
    files['weight_evolution'] = chart.plot_weight_evolution(
        weight_history=weight_history,
        save_path=str(output_path / 'weight_evolution.png'),
        top_n=10
    )

    # Category breakdown
    files['category_breakdown'] = chart.plot_category_breakdown(
        weights=final_weights,
        categories=categories,
        save_path=str(output_path / 'category_breakdown.png')
    )

    # Fitness evolution
    files['fitness_evolution'] = chart.plot_fitness_evolution(
        generation_stats=generation_stats,
        save_path=str(output_path / 'fitness_evolution.png')
    )

    logger.info(f"Created indicator report with {len(files)} charts in {output_dir}")

    return files
