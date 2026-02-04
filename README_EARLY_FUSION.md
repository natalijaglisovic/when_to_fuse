# BERT4Rec with Early Fusion

This implementation provides a BERT4Rec model with early fusion of item ID embeddings, text embeddings, and image embeddings using learnable weights.

## Model Architecture

The early fusion model combines three types of embeddings:

```
fused_embedding = item_id_emb + α·text_emb + β·image_emb
```

Where:
- `item_id_emb`: Learnable item ID embeddings (trained from scratch)
- `text_emb`: Pre-computed text embeddings from sentence transformers (projected to hidden_dim)
- `image_emb`: Pre-computed CLIP image embeddings (projected to hidden_dim)
- `α` (alpha): Learnable weight for text embeddings
- `β` (beta): Learnable weight for image embeddings

The fused embeddings are then passed through the BERT4Rec transformer architecture.

## Files

- `bert4rec/model_early_fusion.py`: Model implementation with early fusion
- `bert4rec/train_early_fusion.py`: Training script
- `text_encoders/sentence_transformer_embeddings.py`: Script to extract text embeddings
- `vision_encoders/clip_embeddings.py`: Script to extract image embeddings

## Prerequisites

Ensure you have:
1. Processed data with `user_id` and `item_id` columns
2. Pre-computed text embeddings (from sentence transformers)
3. Pre-computed image embeddings (from CLIP)

## Usage

### Step 1: Extract Text Embeddings (if not already done)

```bash
python text_encoders/sentence_transformer_embeddings.py \
    --csv_file data/games/amazon_games_user_item_image_text.csv \
    --dataset_name amazon_games
```

This creates embeddings in: `text_encoders/embeddings/amazon_games/sentence_transformer/`

### Step 2: Extract Image Embeddings (if not already done)

```bash
python vision_encoders/clip_embeddings.py \
    --image_folder data/games/item_images_games \
    --dataset_name amazon_games
```

This creates embeddings in: `vision_encoders/embeddings/amazon_games/clip_games/`

### Step 3: Train Early Fusion Model

```bash
python bert4rec/train_early_fusion.py \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games \
    --max_seq_len 50 \
    --hidden_dim 64 \
    --num_layers 2 \
    --num_heads 2 \
    --batch_size 128 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --init_alpha 1.0 \
    --init_beta 1.0
```

## Training Arguments

### Required Arguments
- `--data_path`: Path to CSV with user-item interactions
- `--text_embeddings_dir`: Directory with text embedding .npz files
- `--image_embeddings_dir`: Directory with image embedding .npz files
- `--dataset_name`: Name of dataset (for logging)

### Model Architecture Arguments
- `--max_seq_len`: Maximum sequence length (default: 50)
- `--hidden_dim`: Hidden dimension for transformer (default: 64)
- `--num_layers`: Number of transformer layers (default: 2)
- `--num_heads`: Number of attention heads (default: 2)
- `--dropout_rate`: Dropout rate (default: 0.1)
- `--mask_prob`: Masking probability for BERT training (default: 0.15)

### Fusion Arguments
- `--init_alpha`: Initial value for text weight α (default: 1.0)
- `--init_beta`: Initial value for image weight β (default: 1.0)
- `--freeze_text_embeddings`: Freeze text embeddings during training
- `--freeze_image_embeddings`: Freeze image embeddings during training

### Training Arguments
- `--batch_size`: Batch size (default: 128)
- `--num_epochs`: Number of training epochs (default: 20)
- `--learning_rate`: Learning rate (default: 0.001)
- `--test_ratio`: Ratio of data for testing (default: 0.2)
- `--seed`: Random seed (default: 42)
- `--device`: Device to use: 'cuda', 'cpu', or 'auto' (default: auto)

### Data Arguments
- `--user_col`: Name of user column (auto-detected if not specified)
- `--item_col`: Name of item column (auto-detected if not specified)
- `--num_users`: Subsample N users for faster experimentation (default: None = all users)

### Output Arguments
- `--save_dir`: Directory to save model checkpoints (default: None)

## Example: Training on Baby Products Dataset

```bash
python bert4rec/train_early_fusion.py \
    --data_path data/baby/amazon_baby_products_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_baby/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_baby/clip_baby \
    --dataset_name amazon_baby \
    --num_epochs 30 \
    --learning_rate 0.0005 \
    --save_dir models/early_fusion_baby
```

## Example: Training on Science Dataset

```bash
python bert4rec/train_early_fusion.py \
    --data_path data/science/amazon_science_user_item_image.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_science/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_science/clip_science \
    --dataset_name amazon_science \
    --max_seq_len 25 \
    --hidden_dim 128 \
    --num_layers 4 \
    --save_dir models/early_fusion_science
```

## Model Output

During training, you'll see:
- Loss per epoch
- Current values of α and β fusion weights
- Periodic evaluation metrics (HR@10, NDCG@10)

Final output includes:
- HR@5, HR@10, HR@20 (Hit Rate)
- NDCG@5, NDCG@10, NDCG@20 (Normalized Discounted Cumulative Gain)
- Final learned values of α and β

Example output:
```
Final Results for amazon_games:
  HR@5: 0.2341
  NDCG@5: 0.1523
  HR@10: 0.3128
  NDCG@10: 0.1789
  HR@20: 0.4012
  NDCG@20: 0.1998

Learned Fusion Weights:
  α (text weight): 0.8234
  β (image weight): 1.2456
```

## Comparison with Baselines

To compare with baseline models:

### 1. Vanilla BERT4Rec (ID-only)
```bash
python bert4rec/train.py \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --dataset_name amazon_games
```

### 2. BERT4Rec with Text
```bash
python bert4rec/train_text.py \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --dataset_name amazon_games
```

### 3. BERT4Rec with Image
```bash
python bert4rec/train_image.py \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games
```

### 4. BERT4Rec with Early Fusion (This Implementation)
```bash
python bert4rec/train_early_fusion.py \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games
```

## Key Features

1. **Learnable Fusion Weights**: α and β are learned during training, allowing the model to automatically balance text and image information
2. **Flexible Embedding Freezing**: Can freeze text/image embeddings to reduce training parameters
3. **BERT4Rec Architecture**: Uses masked language modeling for sequential recommendation
4. **Early Fusion**: Combines multimodal features at the input level before transformer processing

## Implementation Details

### Embedding Loading
- Text embeddings: Loaded from `.npz` files with key `'sentence_transformer'`
- Image embeddings: Loaded from `.npz` files with key `'clip'`
- Item ID embeddings: Initialized randomly and trained

### Projection Layers
- Text embeddings projected to `hidden_dim` via linear layer
- Image embeddings projected to `hidden_dim` via linear layer
- Item ID embeddings have dimension `hidden_dim` directly

### Fusion Formula
The fusion is computed as:
```python
fused_emb = item_id_emb + alpha * text_proj(text_emb) + beta * image_proj(image_emb)
```

### Training
- Optimizer: Adam with weight decay 1e-4
- Loss: Cross-entropy loss on masked positions
- Gradient clipping: Max norm of 1.0
- Evaluation: Every 5 epochs

## Troubleshooting

### Missing Embeddings
If you see warnings about missing embeddings, ensure:
1. All items in your CSV have corresponding embedding files
2. Embedding file names match item IDs exactly
3. Embedding files are in the correct directories

### Memory Issues
If you encounter OOM errors:
- Reduce `--batch_size`
- Reduce `--max_seq_len`
- Reduce `--hidden_dim`
- Use `--num_users` to subsample users for experimentation

### Fusion Weight Instability
If α or β become very large or negative:
- Reduce learning rate
- Add regularization on fusion weights (modify model code)
- Try different initialization values

## Citation

If you use this implementation, please cite the original BERT4Rec paper:

```
@inproceedings{sun2019bert4rec,
  title={BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer},
  author={Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={1441--1450},
  year={2019}
}
```
