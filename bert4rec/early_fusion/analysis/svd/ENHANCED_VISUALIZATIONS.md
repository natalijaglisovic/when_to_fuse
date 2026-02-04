# Enhanced SVD Visualizations

This document describes the enhanced visualization suite for SVD-based modality fusion analysis.

## Overview

The enhanced visualization module provides comprehensive, publication-ready plots for analyzing fusion-level modality interactions. All visualizations are based on best practices for understanding multimodal attention patterns.

## New Files

### `visualize_svd_enhanced.py`
Complete visualization toolkit with 5 types of plots:

1. **Singular Value Spectrum** - Primary metric showing rank structure
2. **Effective Rank Bar Plot** - Summary scalar for fusion adaptivity
3. **Modality Attention Distribution** - Detailed histograms and KDE
4. **Attention Statistics Summary** - Mean/std with error bars
5. **Comprehensive Dashboard** - All-in-one multi-panel view

## Visualization Types

### 1. Singular Value Spectrum (Primary)

**What it shows:**
- How many independent modality interaction patterns exist
- Sharp drop → rank ≈ 1 (fixed weighting, similar to addition)
- Flat spectrum → adaptive fusion with balanced modality usage

**File:** `1_singular_value_spectrum.png`

```python
plot_singular_value_spectrum(
    singular_values_dict={'Attention Fusion': singular_values},
    output_path='singular_values.png'
)
```

**Interpretation:**
- For M=2 modalities:
  - σ₁ ≈ σ₂: Both modalities contribute equally (balanced)
  - σ₁ >> σ₂: One modality dominates (collapse)

### 2. Effective Rank Bar Plot (Summary Metric)

**What it shows:**
- Single scalar capturing fusion adaptivity
- r_eff = exp(-Σ p_i log p_i) where p = σ/Σσ

**File:** `2_effective_rank_bar.png`

**Interpretation:**
- r_eff ≈ 1.0: Fixed weighting (like addition)
- r_eff ≈ 2.0: Adaptive fusion (for M=2)
- Color-coded thresholds:
  - Red (< 1.2): Poor - rank collapse
  - Orange (1.2-1.5): Fair - some preference
  - Green (1.5-1.8): Good - moderate diversity
  - Blue (> 1.8): Excellent - balanced fusion

### 3. Modality Attention Distribution (Highly Interpretable)

**What it shows:**
- How often the model prefers each modality across all items
- Distribution of α_text and α_image values

**File:** `3_attention_distribution.png`

**Contains 4 subplots:**
1. **Overlapping Histograms**: Frequency distribution of attention weights
2. **Kernel Density Estimation (KDE)**: Smooth probability density
3. **Box Plot**: Statistical summary (median, quartiles, outliers)
4. **Joint Distribution**: Hexbin plot showing α_text vs α_image

**What to look for:**
- **Sharp peak**: Modality dominance (most items use one modality)
- **Wide distribution**: Adaptivity (different items use different modalities)
- **Bimodal**: Item-level specialization (some items use text, others use image)
- **Uniform**: Balanced but non-adaptive (always 50/50 split)

### 4. Attention Statistics Summary

**What it shows:**
- Mean attention weights with standard deviation error bars
- Pie chart showing overall modality distribution

**File:** `4_statistics_summary.png`

**Useful for:**
- Quick summary of which modality is preferred overall
- Understanding variance in modality usage
- Comparing multiple models side-by-side

### 5. Comprehensive Dashboard (All-in-One)

**What it shows:**
- 5-panel comprehensive view combining all key metrics

**File:** `5_comprehensive_dashboard.png`

**Layout:**
```
┌─────────────────────────┬─────────────────────────┐
│  Singular Values        │  Effective Rank Bar     │
│  (line plot)            │  (bar with thresholds)  │
├─────────────────────────┴─────────────────────────┤
│  Attention Distribution (histogram + KDE overlay) │
├─────────────────────────┬─────────────────────────┤
│  Statistics Summary     │  Joint Distribution     │
│  (bar with error bars)  │  (hexbin heatmap)       │
└─────────────────────────┴─────────────────────────┘
```

**Ideal for:**
- Papers (main figure)
- Presentations (comprehensive overview)
- Quick model diagnosis

## Usage

### During Training

The training script automatically generates visualizations at key epochs:

```bash
python train_early_fusion_attention_svd.py \
    --data_path data.csv \
    --text_embeddings_dir text_emb/ \
    --image_embeddings_dir img_emb/ \
    --save_dir results/ \
    --svd_analysis_freq 5  # Analysis every 5 epochs
```

**Generated files:**
- `results/visualizations_epoch_1/` - Initial state
- `results/visualizations_epoch_10/` - Every 10 epochs (2× analysis freq)
- `results/visualizations_epoch_20/` - Final epoch
- `results/final_visualizations/` - Complete analysis on full dataset

### Standalone Analysis

For existing trained models:

```bash
python analyze_svd_standalone.py \
    --model_path model.pt \
    --data_path data.csv \
    --text_embeddings_dir text_emb/ \
    --image_embeddings_dir img_emb/ \
    --output_dir analysis_output/
```

**Output directory structure:**
```
analysis_output/
├── 1_singular_value_spectrum.png
├── 2_effective_rank_bar.png
├── 3_attention_distribution.png
├── 4_statistics_summary.png
├── 5_comprehensive_dashboard.png
└── svd_analysis.json
```

### Programmatic Usage

```python
from bert4rec.early_fusion.visualize_svd_enhanced import (
    create_comprehensive_visualizations,
    plot_singular_value_spectrum,
    plot_effective_rank_bar,
    plot_attention_distribution
)

# After analyzing model
results = analyze_modality_fusion(model, dataloader, device)

# Create all visualizations
create_comprehensive_visualizations(
    results,
    output_dir='my_analysis',
    modality_names=['Text', 'Image']
)

# Or create individual plots
plot_singular_value_spectrum(
    {'Model A': results['singular_values']},
    'singular_values.png'
)
```

## Interpretation Examples

### Example 1: Balanced Adaptive Fusion
```
Effective Rank: 1.92 / 2.0
Singular Values: [1.41, 1.38]

Modality Statistics:
  Text  - Mean: 0.51, Std: 0.23
  Image - Mean: 0.49, Std: 0.23
```

**Interpretation:** EXCELLENT
- High effective rank (1.92 ≈ 2.0)
- Similar singular values (1.41 ≈ 1.38)
- Balanced means (0.51 ≈ 0.49)
- High std (0.23) indicates adaptivity across items
- **Conclusion:** Model uses both modalities adaptively

### Example 2: Text Dominance (Rank Collapse)
```
Effective Rank: 1.08 / 2.0
Singular Values: [1.98, 0.21]

Modality Statistics:
  Text  - Mean: 0.92, Std: 0.05
  Image - Mean: 0.08, Std: 0.05
```

**Interpretation:** POOR
- Low effective rank (1.08 ≈ 1.0)
- One singular value dominates (1.98 >> 0.21)
- Heavily skewed means (0.92 >> 0.08)
- Low std (0.05) indicates no adaptivity
- **Conclusion:** Model ignores image modality (rank collapse)

### Example 3: Moderate Diversity
```
Effective Rank: 1.67 / 2.0
Singular Values: [1.52, 1.12]

Modality Statistics:
  Text  - Mean: 0.61, Std: 0.18
  Image - Mean: 0.39, Std: 0.18
```

**Interpretation:** GOOD
- Moderate effective rank (1.67)
- Both singular values substantial (1.52 and 1.12)
- Some text preference (0.61 > 0.39) but not extreme
- Moderate std (0.18) shows some adaptivity
- **Conclusion:** Model prefers text but uses image when helpful

## What to Look For in Visualizations

### Red Flags (Poor Fusion)
1. **Singular value spectrum**: Sharp drop, second value near zero
2. **Effective rank**: < 1.2
3. **Distribution histogram**: Sharp peak at 0 or 1 for one modality
4. **Joint distribution**: All points along edges (no middle ground)

### Green Flags (Good Fusion)
1. **Singular value spectrum**: Flat, values close to each other
2. **Effective rank**: > 1.7
3. **Distribution histogram**: Wide, overlapping distributions
4. **Joint distribution**: Points spread across diagonal

### Interesting Patterns
1. **Bimodal distribution**: Two peaks in histogram
   - Some items prefer text, others prefer image
   - Item-level specialization (good!)

2. **U-shaped distribution**: Peaks at 0 and 1
   - Model makes strong decisions
   - Either/or behavior (not always bad)

3. **Narrow gaussian**: Peak at 0.5
   - Always 50/50 split
   - Balanced but NOT adaptive (questionable)

## Comparing Multiple Models

You can compare different fusion strategies:

```python
# Collect results from different models
results_attention = analyze_modality_fusion(model_attention, loader, device)
results_concat = analyze_modality_fusion(model_concat, loader, device)

# Compare singular values
plot_singular_value_spectrum({
    'Attention Fusion': results_attention['singular_values'],
    'Concat Fusion': results_concat['singular_values']
})

# Compare effective ranks
plot_effective_rank_bar({
    'Attention': results_attention['effective_rank'],
    'Concat': results_concat['effective_rank']
})
```

## Tips for Papers/Presentations

### For Main Paper
- Use **Comprehensive Dashboard** (5-panel figure)
- Include in main results section
- Reference effective rank in text

### For Appendix
- Include individual plots
- Show evolution over training (multiple epochs)
- Compare different architectures/hyperparameters

### For Presentations
- Start with **Effective Rank Bar** (simple, interpretable)
- Follow with **Attention Distribution** (shows adaptivity)
- End with **Comprehensive Dashboard** (complete picture)

### For Ablation Studies
- **Singular Value Spectrum**: Compare different fusion methods
- **Effective Rank Bar**: Summary comparison across 5+ models
- **Distribution Histogram**: Show impact of architectural choices

## Color Scheme

All visualizations use a consistent color scheme:
- **Text modality**: Coral/Red (`#FF6B6B`)
- **Image modality**: Sky Blue (`#4ECDC4`)
- **Effective rank thresholds**:
  - Poor (< 1.2): Red
  - Fair (1.2-1.5): Orange
  - Good (1.5-1.8): Light Green
  - Excellent (> 1.8): Blue

## File Naming Convention

Generated files follow this pattern:
```
{number}_{description}.png
```

This ensures:
1. Alphabetical ordering matches logical flow
2. Easy to identify visualization type
3. Consistent across different analyses

## Dependencies

Required packages:
```python
matplotlib >= 3.3.0
numpy >= 1.19.0
scipy >= 1.5.0  # For KDE
seaborn >= 0.11.0  # Optional, for enhanced styling
```

## Example Output

When running the visualization suite, you'll see:

```
Creating comprehensive visualizations in: results/final_visualizations
============================================================
Saved: results/final_visualizations/1_singular_value_spectrum.png
Saved: results/final_visualizations/2_effective_rank_bar.png
Saved: results/final_visualizations/3_attention_distribution.png
Saved: results/final_visualizations/4_statistics_summary.png
Saved: results/final_visualizations/5_comprehensive_dashboard.png
============================================================
All visualizations saved to: results/final_visualizations

Generated files:
  1. 1_singular_value_spectrum.png - Shows rank structure
  2. 2_effective_rank_bar.png - Summary metric
  3. 3_attention_distribution.png - Detailed distributions
  4. 4_statistics_summary.png - Statistical summary
  5. 5_comprehensive_dashboard.png - All-in-one view
```

## Testing the Visualizations

You can test the visualization module with dummy data:

```bash
python visualize_svd_enhanced.py
```

This will generate example visualizations in `example_visualizations/` directory using simulated attention data.
