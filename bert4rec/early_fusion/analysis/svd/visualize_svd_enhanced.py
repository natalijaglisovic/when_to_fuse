#!/usr/bin/env python3
"""
Enhanced SVD visualization utilities for modality fusion analysis.

This module provides comprehensive visualizations for analyzing fusion-level
modality interactions including:
1. Singular Value Spectrum
2. Effective Rank Bar Plot
3. Modality Attention Distribution (Histogram/KDE)
4. Combined analysis dashboard

Usage:
    from visualize_svd_enhanced import create_comprehensive_visualizations
    create_comprehensive_visualizations(results, output_dir)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


def plot_singular_value_spectrum(singular_values_dict, output_path=None):
    """
    Primary Visualization: Singular Value Spectrum

    Shows how many independent modality interaction patterns exist.
    - Sharp drop (rank ≈ 1): Addition fusion, fixed weighting
    - Flatter spectrum: Adaptive attention-based fusion

    Args:
        singular_values_dict: Dict of {fusion_name: singular_values_array}
        output_path: Path to save figure (None = display only)
    """
    plt.figure(figsize=(10, 6))

    for name, S in singular_values_dict.items():
        if isinstance(S, np.ndarray):
            values = S
        else:
            values = S.cpu().numpy() if hasattr(S, 'cpu') else np.array(S)

        plt.plot(
            range(1, len(values) + 1),
            values,
            marker='o',
            linewidth=2,
            markersize=8,
            label=name
        )

    plt.xlabel("Singular Value Index", fontsize=12)
    plt.ylabel("Singular Value", fontsize=12)
    plt.title("Fusion-Level Singular Value Spectrum", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_effective_rank_bar(ranks, output_path=None):
    """
    Strong Summary Visualization: Effective Rank Bar Plot

    Single scalar capturing fusion adaptivity.
    r_eff = exp(-Σ p_i log p_i)

    Interpretation:
    - Addition ≈ 1: Fixed weighting
    - Attention ≈ M: Adaptive fusion (M = number of modalities)

    Args:
        ranks: Dict {fusion_name: effective_rank_value}
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    names = list(ranks.keys())
    values = [ranks[n].item() if hasattr(ranks[n], 'item') else ranks[n] for n in names]

    colors = ['#FF6B6B' if v < 1.2 else '#4ECDC4' if v < 1.5 else '#95E1D3' if v < 1.8 else '#45B7D1' for v in values]

    bars = plt.bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylabel("Effective Rank", fontsize=12)
    plt.title("Fusion-Level Effective Rank", fontsize=14, fontweight='bold')
    plt.ylim(0, max(values) * 1.3)
    plt.grid(axis='y', alpha=0.3)

    # Add interpretation guide
    plt.axhline(y=1.2, color='red', linestyle='--', alpha=0.5, label='Poor (< 1.2)')
    plt.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='Fair (1.2-1.5)')
    plt.axhline(y=1.8, color='green', linestyle='--', alpha=0.5, label='Good (1.5-1.8)')
    plt.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_attention_distribution(alpha_matrix, modality_names=['Text', 'Image'], output_path=None):
    """
    Highly Interpretable: Modality Attention Distribution

    Shows how often the model prefers each modality across all items.

    What to look for:
    - Sharp peak: Modality dominance
    - Wide distribution: Adaptivity
    - Bimodality: Item-level specialization

    Args:
        alpha_matrix: [N, M] attention matrix (N items, M modalities)
        modality_names: List of modality names
        output_path: Path to save figure
    """
    if isinstance(alpha_matrix, dict):
        alpha_matrix = alpha_matrix.get('alpha_matrix', alpha_matrix)

    if hasattr(alpha_matrix, 'cpu'):
        alpha_matrix = alpha_matrix.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3'][:alpha_matrix.shape[1]]

    # 1. Overlapping Histograms
    ax = axes[0, 0]
    for i, name in enumerate(modality_names[:alpha_matrix.shape[1]]):
        # Determine number of bins based on data range and unique values
        data = alpha_matrix[:, i]

        # Filter out NaN/inf values
        data_clean = data[np.isfinite(data)]

        if len(data_clean) == 0:
            continue

        data_range = data_clean.max() - data_clean.min()
        num_unique = len(np.unique(data_clean))

        # Handle edge case: constant or near-constant data
        if data_range < 1e-6 or num_unique <= 1:
            # For constant data, just use 1 bin
            bins = 1
        elif num_unique <= 10:
            # For very few unique values, use explicit bin edges
            bins = min(num_unique, 50)
        else:
            # For normal data, calculate bins based on range
            # Ensure bins is reasonable: at least 10, at most 50
            bins = min(50, max(10, int(np.sqrt(len(data_clean)))))

        ax.hist(data_clean, bins=bins, alpha=0.6, label=name,
               color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Attention Weight", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Modality Attention Weights", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. KDE (Kernel Density Estimation)
    ax = axes[0, 1]
    kde_success = False
    for i, name in enumerate(modality_names[:alpha_matrix.shape[1]]):
        data = alpha_matrix[:, i]

        # Filter out NaN/inf values
        data_clean = data[np.isfinite(data)]

        if len(data_clean) == 0:
            continue

        # Check if data has sufficient variance for KDE
        if len(data_clean) > 1 and np.std(data_clean) > 1e-6:
            try:
                kde = stats.gaussian_kde(data_clean)
                x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
                ax.plot(x_range, kde(x_range), linewidth=2.5, label=name, color=colors[i])
                ax.fill_between(x_range, kde(x_range), alpha=0.3, color=colors[i])
                kde_success = True
            except (np.linalg.LinAlgError, ValueError):
                # If KDE fails, skip this modality
                pass

    if not kde_success:
        # If no KDE plots were created, show a message
        ax.text(0.5, 0.5, 'KDE unavailable\n(constant or near-constant values)',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel("Attention Weight", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Kernel Density Estimation", fontsize=12, fontweight='bold')
    if kde_success:
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # 3. Box Plot
    ax = axes[1, 0]
    # Filter out NaN/inf values for each modality
    data_for_box = [alpha_matrix[:, i][np.isfinite(alpha_matrix[:, i])] for i in range(alpha_matrix.shape[1])]

    # Only create box plot if we have valid data
    if all(len(d) > 0 for d in data_for_box):
        bp = ax.boxplot(data_for_box, labels=modality_names[:alpha_matrix.shape[1]],
                        patch_artist=True, showmeans=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel("Attention Weight", fontsize=11)
        ax.set_title("Attention Weight Distribution (Box Plot)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'No valid data\nfor box plot',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title("Attention Weight Distribution (Box Plot)", fontsize=12, fontweight='bold')
        ax.axis('off')

    # 4. Scatter Plot (for 2 modalities) or Violin Plot
    ax = axes[1, 1]

    if alpha_matrix.shape[1] == 2:
        # Filter out NaN/inf values before plotting
        text_vals = alpha_matrix[:, 0]
        image_vals = alpha_matrix[:, 1]
        valid_mask = np.isfinite(text_vals) & np.isfinite(image_vals)

        if valid_mask.sum() > 0:
            text_vals_clean = text_vals[valid_mask]
            image_vals_clean = image_vals[valid_mask]
            ax.hexbin(text_vals_clean, image_vals_clean, gridsize=30, cmap='YlOrRd', mincnt=1)
            ax.plot([0, 1], [1, 0], 'k--', alpha=0.5, linewidth=2, label='α_text + α_image = 1')
            ax.set_xlabel(f"{modality_names[0]} Attention", fontsize=11)
            ax.set_ylabel(f"{modality_names[1]} Attention", fontsize=11)
            ax.set_title("Joint Attention Distribution", fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid data\nfor joint distribution',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title("Joint Attention Distribution", fontsize=12, fontweight='bold')
            ax.axis('off')
    else:
        # Violin plot for multiple modalities
        if all(len(d) > 0 for d in data_for_box):
            parts = ax.violinplot(data_for_box, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.6)
            ax.set_xticks(range(1, len(modality_names[:alpha_matrix.shape[1]]) + 1))
            ax.set_xticklabels(modality_names[:alpha_matrix.shape[1]])
            ax.set_ylabel("Attention Weight", fontsize=11)
            ax.set_title("Attention Weight Distribution (Violin)", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No valid data\nfor violin plot',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title("Attention Weight Distribution (Violin)", fontsize=12, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_attention_statistics_summary(alpha_matrix, modality_names=['Text', 'Image'], output_path=None):
    """
    Statistical summary of attention distributions.

    Args:
        alpha_matrix: [N, M] attention matrix
        modality_names: List of modality names
        output_path: Path to save figure
    """
    if isinstance(alpha_matrix, dict):
        alpha_matrix = alpha_matrix.get('alpha_matrix', alpha_matrix)

    if hasattr(alpha_matrix, 'cpu'):
        alpha_matrix = alpha_matrix.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Mean with std error bars
    ax = axes[0]
    means = [alpha_matrix[:, i].mean() for i in range(alpha_matrix.shape[1])]
    stds = [alpha_matrix[:, i].std() for i in range(alpha_matrix.shape[1])]

    # Clean means and stds: replace NaN/inf with 0
    means = [m if np.isfinite(m) else 0.0 for m in means]
    stds = [s if np.isfinite(s) else 0.0 for s in stds]

    colors = ['coral', 'skyblue', 'lightgreen', 'plum'][:alpha_matrix.shape[1]]

    x_pos = np.arange(len(modality_names[:alpha_matrix.shape[1]]))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}\n±{std:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Mean Attention Weight', fontsize=12)
    ax.set_title('Modality Attention Statistics', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modality_names[:alpha_matrix.shape[1]], fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Pie chart with robust handling
    ax = axes[1]

    # Ensure all values are non-negative and finite for pie chart
    pie_values = np.array([max(0.0, m) if np.isfinite(m) else 0.0 for m in means])

    # Only create pie chart if we have valid, non-zero values
    if pie_values.sum() > 0:
        ax.pie(pie_values, labels=modality_names[:alpha_matrix.shape[1]], autopct='%1.1f%%',
              colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
              wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        ax.set_title('Average Modality Distribution', fontsize=13, fontweight='bold')
    else:
        # If all values are zero or invalid, show a message instead
        ax.text(0.5, 0.5, 'No valid attention\nweights available',
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Average Modality Distribution', fontsize=13, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def create_comprehensive_dashboard(results, output_path=None):
    """
    Create a comprehensive 4-panel dashboard with all key visualizations.

    Args:
        results: Dict from analyze_modality_fusion() containing:
            - effective_rank
            - singular_values
            - alpha_matrix
            - mean_text_weight, mean_image_weight
            - std_text_weight, std_image_weight
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    alpha_matrix = results['alpha_matrix']
    singular_values = results['singular_values']
    eff_rank = results['effective_rank']

    colors = ['#FF6B6B', '#4ECDC4']

    # 1. Singular Values (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(singular_values) + 1), singular_values,
            marker='o', linewidth=3, markersize=12, color='#45B7D1')
    ax1.set_xlabel("Singular Value Index", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Singular Value", fontsize=12, fontweight='bold')
    ax1.set_title("Singular Value Spectrum", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add text annotation
    ax1.text(0.95, 0.95, f'Effective Rank: {eff_rank:.3f}',
            transform=ax1.transAxes, fontsize=11, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 2. Effective Rank Bar (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    color = '#FF6B6B' if eff_rank < 1.2 else '#FFA07A' if eff_rank < 1.5 else '#90EE90' if eff_rank < 1.8 else '#45B7D1'
    bar = ax2.bar(['Effective Rank'], [eff_rank], color=color, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.text(0, eff_rank + 0.05, f'{eff_rank:.3f}', ha='center', va='bottom',
            fontsize=14, fontweight='bold')
    ax2.set_ylabel("Effective Rank", fontsize=12, fontweight='bold')
    ax2.set_title("Fusion Adaptivity", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 2.2)
    ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Max (balanced)')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Min (collapse)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Attention Distribution Histogram (Middle - Full Width)
    ax3 = fig.add_subplot(gs[1, :])
    for i, name in enumerate(['Text', 'Image'][:alpha_matrix.shape[1]]):
        # Determine number of bins based on data range and unique values
        data = alpha_matrix[:, i]

        # Filter out NaN/inf values
        data_clean = data[np.isfinite(data)]

        if len(data_clean) == 0:
            continue

        data_range = data_clean.max() - data_clean.min()
        num_unique = len(np.unique(data_clean))

        # Handle edge case: constant or near-constant data
        if data_range < 1e-6 or num_unique <= 1:
            # For constant data, just use 1 bin
            bins = 1
        elif num_unique <= 10:
            # For very few unique values, use explicit bin edges
            bins = min(num_unique, 50)
        else:
            # For normal data, calculate bins based on range
            # Ensure bins is reasonable: at least 10, at most 50
            bins = min(50, max(10, int(np.sqrt(len(data_clean)))))

        ax3.hist(data_clean, bins=bins, alpha=0.6, label=name,
                color=colors[i], edgecolor='black', linewidth=0.5)

        # Add KDE overlay if data has sufficient variance
        if len(data_clean) > 1 and np.std(data_clean) > 1e-6:
            try:
                kde = stats.gaussian_kde(data_clean)
                x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
                ax3_twin = ax3.twinx()
                ax3_twin.plot(x_range, kde(x_range), linewidth=2.5, color=colors[i], alpha=0.8)
                ax3_twin.set_ylabel("")
                ax3_twin.set_yticks([])
            except (np.linalg.LinAlgError, ValueError):
                # Skip KDE if it fails
                pass

    ax3.set_xlabel("Attention Weight", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax3.set_title("Modality Attention Distribution", fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    # 4. Statistics Summary (Bottom Left)
    ax4 = fig.add_subplot(gs[2, 0])
    means = [alpha_matrix[:, i].mean() for i in range(alpha_matrix.shape[1])]
    stds = [alpha_matrix[:, i].std() for i in range(alpha_matrix.shape[1])]

    # Clean means and stds: replace NaN/inf with 0
    means = [m if np.isfinite(m) else 0.0 for m in means]
    stds = [s if np.isfinite(s) else 0.0 for s in stds]

    x_pos = np.arange(2)
    bars = ax4.bar(x_pos, means, yerr=stds, capsize=10, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5)

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax4.text(i, mean + std + 0.02, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax4.set_ylabel('Mean Attention Weight', fontsize=12, fontweight='bold')
    ax4.set_title('Modality Statistics', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['Text', 'Image'], fontsize=11)
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Joint Distribution (Bottom Right)
    ax5 = fig.add_subplot(gs[2, 1])
    if alpha_matrix.shape[1] == 2:
        # Filter out NaN/inf values before plotting
        text_vals = alpha_matrix[:, 0]
        image_vals = alpha_matrix[:, 1]
        valid_mask = np.isfinite(text_vals) & np.isfinite(image_vals)

        if valid_mask.sum() > 0:
            text_vals_clean = text_vals[valid_mask]
            image_vals_clean = image_vals[valid_mask]
            h = ax5.hexbin(text_vals_clean, image_vals_clean, gridsize=25, cmap='YlOrRd', mincnt=1)
            ax5.plot([0, 1], [1, 0], 'k--', alpha=0.7, linewidth=2.5, label='α_text + α_image = 1')
            ax5.set_xlabel("Text Attention", fontsize=12, fontweight='bold')
            ax5.set_ylabel("Image Attention", fontsize=12, fontweight='bold')
            ax5.set_title("Joint Attention Distribution", fontsize=13, fontweight='bold')
            ax5.legend(fontsize=10)
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)
            plt.colorbar(h, ax=ax5, label='Count')
        else:
            ax5.text(0.5, 0.5, 'No valid data\nfor joint distribution',
                    ha='center', va='center', fontsize=12, transform=ax5.transAxes)
            ax5.set_title("Joint Attention Distribution", fontsize=13, fontweight='bold')
            ax5.axis('off')

    # Overall title
    fig.suptitle(f'SVD Modality Fusion Analysis Dashboard\n'
                f'Items Analyzed: {results["num_items_analyzed"]:,} | '
                f'Effective Rank: {eff_rank:.3f}/2.0',
                fontsize=15, fontweight='bold', y=0.995)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def create_comprehensive_visualizations(results, output_dir, modality_names=['Text', 'Image']):
    """
    Create all visualizations for SVD modality analysis.

    Args:
        results: Dict from analyze_modality_fusion()
        output_dir: Directory to save all visualizations
        modality_names: List of modality names
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCreating comprehensive visualizations in: {output_dir}")
    print("="*60)

    # 1. Singular value spectrum
    singular_values_dict = {'Attention Fusion': results['singular_values']}
    plot_singular_value_spectrum(
        singular_values_dict,
        os.path.join(output_dir, '1_singular_value_spectrum.png')
    )

    # 2. Effective rank bar
    ranks = {'Fusion Model': results['effective_rank']}
    plot_effective_rank_bar(
        ranks,
        os.path.join(output_dir, '2_effective_rank_bar.png')
    )

    # 3. Attention distribution
    plot_attention_distribution(
        results['alpha_matrix'],
        modality_names,
        os.path.join(output_dir, '3_attention_distribution.png')
    )

    # 4. Statistics summary
    plot_attention_statistics_summary(
        results['alpha_matrix'],
        modality_names,
        os.path.join(output_dir, '4_statistics_summary.png')
    )

    # 5. Comprehensive dashboard
    create_comprehensive_dashboard(
        results,
        os.path.join(output_dir, '5_comprehensive_dashboard.png')
    )

    print("="*60)
    print(f"All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. 1_singular_value_spectrum.png - Shows rank structure")
    print("  2. 2_effective_rank_bar.png - Summary metric")
    print("  3. 3_attention_distribution.png - Detailed distributions")
    print("  4. 4_statistics_summary.png - Statistical summary")
    print("  5. 5_comprehensive_dashboard.png - All-in-one view")


if __name__ == "__main__":
    # Example usage with dummy data
    print("Example: Creating visualizations with dummy data")

    # Simulate results from analyze_modality_fusion()
    np.random.seed(42)

    # Simulate attention matrix: some items prefer text, some prefer image
    N = 5000
    alpha_text = np.concatenate([
        np.random.beta(5, 2, N//2),  # Text-preferring items
        np.random.beta(2, 5, N//2)   # Image-preferring items
    ])
    alpha_image = 1 - alpha_text
    alpha_matrix = np.stack([alpha_text, alpha_image], axis=1)

    # Compute SVD
    U, S, Vt = np.linalg.svd(alpha_matrix, full_matrices=False)
    p = S / S.sum()
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-12)))

    results = {
        'effective_rank': eff_rank,
        'singular_values': S,
        'alpha_matrix': alpha_matrix,
        'mean_text_weight': alpha_text.mean(),
        'mean_image_weight': alpha_image.mean(),
        'std_text_weight': alpha_text.std(),
        'std_image_weight': alpha_image.std(),
        'num_items_analyzed': N
    }

    create_comprehensive_visualizations(results, 'example_visualizations')
