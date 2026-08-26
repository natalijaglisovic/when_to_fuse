# When to Fuse: A Systematic Comparison of Early, Intermediate and Late Fusion
Strategies for Multimodal Sequential Recommendation

Code for the paper **"When to Fuse"**, which systematically compares multimodal fusion strategies early, intermediate, and late fusion across three element-wise methods (sum, concatenation, cross-modal attention) in a BERT4Rec-based sequential recommender system.

## Overview

Modern recommender systems can draw on multiple modalities: item IDs, product images, and textual descriptions. This paper asks *when* these signals should be combined. We compare:

- **Fusion stage**: early (input), intermediate (mid-encoder), late (post-encoder)
- **Fusion method**: element-wise sum, concatenation, cross-modal attention
- **Modalities**: item ID embeddings, CLIP image embeddings, Sentence-Transformer text embeddings

Beyond ranking metrics (HR@K, NDCG@K), we analyse each strategy via:
- **Robustness evaluation**: performance under progressive modality dropout (10–90%)
- **Gradient attribution**: how much each modality drives predictions

## Project Structure

```
.
├── data/
│   ├── data_preprocessing.py        # Download and preprocess Amazon review data
│   └── process_multimodal_data.py   # Merge item metadata with interaction data
│
├── text_encoders/
│   └── sentence_transformer_embeddings.py   # Extract text embeddings (all-MiniLM-L6-v2)
│
├── vision_encoders/
│   └── clip_embeddings.py           # Extract image embeddings (CLIP ViT-B/32)
│
├── model/
│   ├── early_fusion/                # Fusion at the input layer
│   │   ├── model_early_fusion.py          # Sum and concat fusion
│   │   ├── model_early_fusion_attention.py # Cross-modal attention fusion
│   │   ├── train_early_fusion.py
│   │   ├── train_early_fusion_attention.py
│   │   └── analysis/
│   │       ├── robustness_eval.py
│   │       ├── model_early_fusion_gradient_attribution.py
│   │       ├── train_early_fusion_gradient_attribution.py
representational analysis
│   ├── intermediate_fusion/         # Fusion after per-modality transformer layers
│   │   ├── model_intermediate_fusion.py
│   │   ├── model_intermediate_fusion_attention.py
│   │   ├── train_intermediate_fusion.py
│   │   ├── train_intermediate_fusion_attention.py
│   │   └── analysis/
│   │       ├── robustness_eval.py
│   │       ├── model_intermediate_fusion_gradient_attribution.py
│   │       └── train_intermediate_fusion_gradient_attribution.py
│   │
│   └── late_fusion/                 # Fusion after full per-modality encoding
│       ├── model_late_fusion.py
│       ├── model_late_fusion_attention.py
│       ├── train_late_fusion.py
│       ├── train_late_fusion_attention.py
│       └── analysis/
│           ├── robustness_eval.py
│           ├── model_late_fusion_gradient_attribution.py
│           └── train_late_fusion_gradient_attribution.py
└── requirements.txt
```

## Datasets

We evaluate on three Amazon product review datasets:

| Dataset | Domain |
|---|---|
| Amazon Games | Video games |
| Amazon Baby | Baby products |
| Amazon Science | Science & education |

Raw data is available from the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) benchmark. For each domain, download the per-category review file (`<Category>.jsonl`) and metadata file (`meta_<Category>.jsonl`) into `data/<dataset>/` and run them through `data/process_multimodal_data.py` (see below) to produce the CSV consumed by the rest of the pipeline.

Expected CSV format: one row per user–item interaction with columns including `user_id`, `item_id`, `image`, and text metadata (title, description), this is what `process_multimodal_data.py` outputs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 1.10+. GPU recommended for training.

## Usage

### 1. Preprocess data

Preprocessing is a two-step pipeline:

1. **`process_multimodal_data.py`** — reads the raw `<Category>.jsonl` / `meta_<Category>.jsonl` files (place them in `data/<dataset>/`), applies k-core filtering (and, for Baby, random sampling down to a target size), merges reviews with item metadata (images, text), and writes `amazon_<category>_user_item_image_text.csv`.
2. **`data_preprocessing.py`** — turns that CSV into per-user interaction sequences, a train/test split, and item-id vocabulary, saved as pickles under `preprocessed_data/<dataset>/`.

The exact k-core threshold and sampling settings used per domain (**Games: 5-core, Science: 4-core, Baby: 5-core + random sample to ~9k users / ~2k items**) are what make the reported results reproducible; every `process_multimodal_data.py` run also writes a `<output>.manifest.json` next to the CSV recording the parameters (including the random seed) actually used.

```bash
# Games (5-core, no sampling)
python -m data.process_multimodal_data \
    --data-dir data/games \
    --category Video_Games \
    --min-interactions 5 \
    --output data/games/amazon_games_user_item_image_text.csv

# Science (4-core, no sampling)
python -m data.process_multimodal_data \
    --data-dir data/science \
    --category Industrial_and_Scientific \
    --min-interactions 4 \
    --output data/science/amazon_science_user_item_image.csv

# Baby (5-core, then random sample to ~9k users / ~2k items, fixed seed for reproducibility)
python -m data.process_multimodal_data \
    --data-dir data/baby \
    --category Baby_Products \
    --min-interactions 5 \
    --target-users 9000 \
    --target-items 2000 \
    --sampling-strategy random \
    --seed 42 \
    --output data/baby/amazon_baby_products_user_item_image_text.csv
```

```bash
python -m data.data_preprocessing
```

`data_preprocessing.py`'s `__main__` iterates over the three domains (with paths and k-core values matching the commands above) and writes each domain's preprocessed pickles to `preprocessed_data/<dataset>/`.

### 2. Extract multimodal embeddings

**Text** (Sentence-Transformers, `all-MiniLM-L6-v2`):
```bash
python text_encoders/sentence_transformer_embeddings.py \
    --csv_file data/games/amazon_games_user_item_image_text.csv \
    --dataset_name amazon_games
```

**Image** (CLIP ViT-B/32):
```bash
python vision_encoders/clip_embeddings.py \
    --image_folder data/games/item_images \
    --dataset_name amazon_games
```

Embeddings are saved to `text_encoders/embeddings/<dataset>/` and `vision_encoders/embeddings/<dataset>/`.

### 3. Train fusion models

All training scripts share the same core arguments. Run from the repo root with `-m` so Python resolves the `model.*` package imports correctly.

**Early fusion — sum or concat:**
```bash
python -m model.early_fusion.train_early_fusion \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games \
    --fusion_mode add        # or concat
```

**Early fusion — attention:**
```bash
python -m model.early_fusion.train_early_fusion_attention \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games
```

Replace `early_fusion` with `intermediate_fusion` or `late_fusion` for the other two stages; the arguments are identical.

### 4. Robustness evaluation

Evaluates model performance as an increasing fraction of items have their text/image embeddings replaced with the modality mean (simulating missing data):

```bash
python -m model.early_fusion.analysis.robustness_eval \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games
```

### 5. Gradient attribution

Trains an attribution-instrumented version of the model and reports how much each modality (item ID, text, image) contributes to the ranking score:

```bash
python -m model.early_fusion.analysis.train_early_fusion_gradient_attribution \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games
```
