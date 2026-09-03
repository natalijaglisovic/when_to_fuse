#!/usr/bin/env python3
"""
PyTorch BERT4Rec training script with late fusion (decision-level fusion).

Each modality (item ID, text, image) is processed by its own independent
transformer encoder. The final representations are combined after encoding.

Fusion Modes:
1. Element-wise addition (--fusion_mode add): u = u_id + u_text + u_image
   - Simple element-wise sum of independently encoded representations
   - Each modality contributes equally to the final representation

2. Concatenation (--fusion_mode concat): u = [u_id || u_text || u_image] -> projection
   - Concatenates representations and projects back to hidden_dim
   - Preserves all modality-specific information

Usage (Addition):
python train_late_fusion.py \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games \
    --fusion_mode add

Usage (Concatenation):
python train_late_fusion.py \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games \
    --fusion_mode concat
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os
import sys
from tqdm import tqdm
from model.early_fusion.model_early_fusion import TextEmbeddingLoader, VisionEmbeddingLoader
from model.late_fusion.model_late_fusion import BERT4RecLateFusion


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BERT4RecLateFusionDataset(Dataset):
    """Dataset class for BERT4Rec training with late fusion"""

    def __init__(self, sequences, model):
        self.sequences = sequences
        self.model = model

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        masked_data = self.model.create_masked_sequences([sequence])

        return {
            'input_ids': masked_data['input_ids'][0],
            'masked_positions': masked_data['masked_positions'][0],
            'masked_labels': masked_data['masked_labels'][0]
        }


def load_and_process_data(csv_path, user_col=None, item_col=None,
                         num_users=None):
    """Load and process CSV data"""
    print(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    if user_col is None:
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['user', 'customer', 'buyer']):
                user_col = col
                break

    if item_col is None:
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['item', 'product', 'asin', 'movie', 'book']):
                item_col = col
                break

    if user_col is None or item_col is None:
        print(f"Available columns: {df.columns.tolist()}")
        if user_col is None:
            user_col = input("Enter the user column name: ")
        if item_col is None:
            item_col = input("Enter the item column name: ")

    print(f"Using columns: user='{user_col}', item='{item_col}'")

    valid_users = df[user_col].unique()
    print(f"Total users found: {len(valid_users)}")

    if num_users and len(valid_users) > num_users:
        sampled_users = np.random.choice(valid_users, num_users, replace=False)
        df = df[df[user_col].isin(sampled_users)]
        print(f"Sampled {num_users} random users: {df.shape}")

    unique_users = sorted(df[user_col].unique())
    unique_items = sorted(df[item_col].unique())

    user_to_id = {user: i for i, user in enumerate(unique_users)}
    item_to_id = {item: i + 1 for i, item in enumerate(unique_items)}
    id_to_item = {i + 1: item for i, item in enumerate(unique_items)}

    df['user_idx'] = df[user_col].map(user_to_id)
    df['item_idx'] = df[item_col].map(item_to_id)

    if 'timestamp' in df.columns:
        df = df.sort_values(['user_idx', 'timestamp'])
    else:
        df = df.sort_values(['user_idx'])

    user_sequences = defaultdict(list)
    for _, row in df.iterrows():
        user_sequences[row['user_idx']].append(row['item_idx'])

    interaction_lengths = [len(seq) for seq in user_sequences.values()]
    avg_interaction_length = np.mean(interaction_lengths) if interaction_lengths else 0

    # Leave-one-out split: most recent item = test, second-most-recent = validation,
    # remaining prefix = training. Val is evaluated against the train-only prefix;
    # test is evaluated against train+val history (so the val item is visible as context).
    train_sequences = []
    val_data = []
    test_data = []

    for user_id, sequence in user_sequences.items():
        if len(sequence) < 3:
            continue

        train_seq = sequence[:-2]
        val_item = sequence[-2]
        test_item = sequence[-1]

        train_sequences.append(train_seq)
        val_data.append((train_seq, val_item))
        test_data.append((sequence[:-1], test_item))

    print(f"Data processing complete:")
    print(f"  Users: {len(unique_users)}")
    print(f"  Items: {len(unique_items)}")
    print(f"  Average interaction length: {avg_interaction_length:.2f}")
    print(f"  Train sequences: {len(train_sequences)}")
    print(f"  Val sequences: {len(val_data)}")
    print(f"  Test sequences: {len(test_data)}")

    return {
        'train_sequences': train_sequences,
        'val_data': val_data,
        'test_data': test_data,
        'num_users': len(unique_users),
        'num_items': len(unique_items),
        'avg_interaction_length': avg_interaction_length,
        'user_to_id': user_to_id,
        'item_to_id': item_to_id,
        'id_to_item': id_to_item
    }


def train_model(model, train_loader, val_data, num_epochs, learning_rate, device, save_dir=None, num_neg=99):
    """Train the BERT4Rec model with late fusion"""

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    model.to(device)

    print(f"\nLate Fusion mode: {model.fusion_mode}")

    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            masked_positions = batch['masked_positions'].to(device)
            masked_labels = batch['masked_labels'].to(device)

            logits = model(input_ids, masked_positions)

            loss_mask = (masked_labels != -1)
            if loss_mask.sum() > 0:
                active_logits = logits.view(-1, logits.size(-1))[loss_mask.view(-1)]
                active_labels = masked_labels.view(-1)[loss_mask.view(-1)]
                loss = criterion(active_logits, active_labels)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Mode': f'late_{model.fusion_mode}'
            })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_losses.append(avg_loss)

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Mode: late_{model.fusion_mode}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("Evaluating on validation set (sampled negatives)...")
            metrics = evaluate_model(model, val_data, device, num_neg=num_neg, full_catalog=False)
            print(f"Val HR@10: {metrics['HR@10']:.4f}, Val NDCG@10: {metrics['NDCG@10']:.4f}")

    return model, epoch_losses


def compute_beyond_accuracy_metrics(model, test_data, train_sequences, device, k=[5, 10, 20]):
    """Compute Coverage@K and Popularity Lift (ΔGAP) using full-catalog ranking.

    Coverage@K: fraction of the item catalog that appears in at least one user's top-K list.
    Popularity Lift: (GAP_recs - GAP_profile) / GAP_profile, reported globally and per user
    group (niche/diverse/blockbuster) split by profile mainstreaminess (Abdollahpouri et al. 2019/2020).
    """
    model.eval()

    item_pop = defaultdict(int)
    for seq in train_sequences:
        for item in seq:
            item_pop[item] += 1

    all_items = list(range(1, model.item_num + 1))

    rec_lists = {}
    user_profiles = {}

    with torch.no_grad():
        for user_idx, (user_seq, _) in enumerate(tqdm(test_data, desc="Full-catalog scoring")):
            user_profiles[user_idx] = user_seq

            trunc = user_seq[-model.max_seq_len:] if len(user_seq) > model.max_seq_len else user_seq
            padded = trunc + [0] * (model.max_seq_len - len(trunc))

            seq_tensor = torch.tensor([padded], dtype=torch.long, device=device)
            cand_tensor = torch.tensor([all_items], dtype=torch.long, device=device)

            scores = model.predict(seq_tensor, cand_tensor).cpu().numpy()[0]
            top_indices = np.argsort(-scores)[:max(k)]
            rec_lists[user_idx] = [all_items[i] for i in top_indices]

    results = {}

    for ki in k:
        recommended = set()
        for items in rec_lists.values():
            recommended.update(items[:ki])
        results[f'Coverage@{ki}'] = len(recommended) / len(all_items)

    mainstream_scores = {
        uid: np.mean([item_pop[item] for item in profile]) if profile else 0.0
        for uid, profile in user_profiles.items()
    }
    vals = sorted(mainstream_scores.values())
    t33 = np.percentile(vals, 33)
    t66 = np.percentile(vals, 66)

    groups = {'niche': [], 'diverse': [], 'blockbuster': []}
    for uid, score in mainstream_scores.items():
        if score <= t33:
            groups['niche'].append(uid)
        elif score <= t66:
            groups['diverse'].append(uid)
        else:
            groups['blockbuster'].append(uid)

    def gap(user_set, item_lists):
        total, count = 0, 0
        for uid in user_set:
            for item in item_lists[uid]:
                total += item_pop[item]
                count += 1
        return total / count if count > 0 else 0.0

    all_users = list(user_profiles.keys())
    for ki in k:
        rec_k = {uid: items[:ki] for uid, items in rec_lists.items()}
        gap_p = gap(all_users, user_profiles)
        gap_q = gap(all_users, rec_k)
        results[f'PopLift@{ki}'] = (gap_q - gap_p) / gap_p if gap_p > 0 else 0.0

        for gname, gusers in groups.items():
            if not gusers:
                results[f'PopLift_{gname}@{ki}'] = 0.0
                continue
            gp = gap(gusers, user_profiles)
            gq = gap(gusers, rec_k)
            results[f'PopLift_{gname}@{ki}'] = (gq - gp) / gp if gp > 0 else 0.0

    return results


def evaluate_model(model, test_data, device, num_neg=99, k=[5, 10, 20], full_catalog=True):
    """Evaluate ranking performance (HR@K, NDCG@K).

    full_catalog=True  -> rank the held-out target against the whole item
                          catalog, excluding items already in the user's history
                          (num_neg is ignored).
    full_catalog=False -> rank the target against num_neg sampled negatives.
    """
    model.eval()

    ndcg_sums = {ki: 0 for ki in k}
    hr_sums = {ki: 0 for ki in k}
    num_users = 0

    all_items = list(range(1, model.item_num + 1))

    with torch.no_grad():
        eval_bar = tqdm(test_data, desc="Evaluating")
        for user_seq, target_item in eval_bar:
            if full_catalog:
                seen = set(user_seq)
                candidates = [target_item] + [it for it in all_items
                                              if it != target_item and it not in seen]
            else:
                candidates = [target_item]
                while len(candidates) <= num_neg:
                    neg_item = random.randint(1, model.item_num)
                    if neg_item != target_item and neg_item not in user_seq:
                        candidates.append(neg_item)

            if len(user_seq) > model.max_seq_len:
                user_seq = user_seq[-model.max_seq_len:]

            padded_seq = user_seq + [0] * (model.max_seq_len - len(user_seq))

            sequences = torch.tensor([padded_seq], dtype=torch.long, device=device)
            candidates_tensor = torch.tensor([candidates], dtype=torch.long, device=device)

            scores = model.predict(sequences, candidates_tensor)
            scores = scores.cpu().numpy()[0]

            # 0-indexed rank of the held-out target (candidate 0); ties favour the target
            rank = int(np.sum(scores > scores[0]))

            for ki in k:
                if rank < ki:
                    hr_sums[ki] += 1
                    ndcg_sums[ki] += 1 / np.log2(rank + 2)

            num_users += 1

    results = {}
    for ki in k:
        hr = hr_sums[ki] / num_users if num_users > 0 else 0
        ndcg = ndcg_sums[ki] / num_users if num_users > 0 else 0
        results[f'HR@{ki}'] = hr
        results[f'NDCG@{ki}'] = ndcg

    return results


def main():
    parser = argparse.ArgumentParser(description='Train BERT4Rec with late fusion')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with user_id and item_id')
    parser.add_argument('--text_embeddings_dir', type=str, required=True,
                        help='Path to directory containing text embeddings (.npz files)')
    parser.add_argument('--image_embeddings_dir', type=str, required=True,
                        help='Path to directory containing image embeddings (.npz files)')
    parser.add_argument('--dataset_name', type=str, default='amazon_games',
                        help='Dataset name (used for logging)')
    parser.add_argument('--user_col', type=str, default=None,
                        help='Name of user column (auto-detect if not specified)')
    parser.add_argument('--item_col', type=str, default=None,
                        help='Name of item column (auto-detect if not specified)')
    parser.add_argument('--num_users', type=int, default=None,
                        help='Number of users to sample (default: None = all users)')
    parser.add_argument('--max_seq_len', type=int, default=100,
                        help='Maximum sequence length (default: 100)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers per modality (default: 4)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads (default: 2)')
    parser.add_argument('--dropout_rate', type=float, default=0.432,
                        help='Dropout rate (default: 0.206)')
    parser.add_argument('--mask_prob', type=float, default=0.149,
                        help='Masking probability (default: 0.149)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--freeze_text_embeddings', action='store_true',
                        help='Freeze text embeddings during training')
    parser.add_argument('--freeze_image_embeddings', action='store_true',
                        help='Freeze image embeddings during training')
    parser.add_argument('--fusion_mode', type=str, default='add', choices=['add', 'concat'],
                        help='Fusion mode: add for element-wise sum, concat for concatenation (default: add)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save model checkpoints')
    parser.add_argument('--loss_history_dir', type=str, default=None,
                        help='Directory to save per-epoch loss history as JSON')
    parser.add_argument('--eval_mode', type=str, default='full', choices=['full', 'sampled'],
                        help="Final-evaluation ranking mode: 'full' ranks the held-out target "
                             "against the whole catalog (minus items already seen by the user); "
                             "'sampled' ranks against --num_neg sampled negatives (default: full)")
    parser.add_argument('--num_neg', type=int, default=99,
                        help='Number of sampled negatives, used for --eval_mode sampled and for '
                             'the in-training validation checks (default: 99)')

    args = parser.parse_args()

    set_random_seeds(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    data = load_and_process_data(
        csv_path=args.data_path,
        user_col=args.user_col,
        item_col=args.item_col,
        num_users=args.num_users
    )

    print("\nLoading text embeddings...")
    text_loader = TextEmbeddingLoader(
        embeddings_dir=args.text_embeddings_dir,
        embedding_key='sentence_transformer'
    )

    text_embedding_matrix = text_loader.build_embedding_matrix(
        item_to_id=data['item_to_id'],
        id_to_item=data['id_to_item']
    )

    print(f"Text embedding matrix shape: {text_embedding_matrix.shape}")

    print("\nLoading image embeddings...")
    image_loader = VisionEmbeddingLoader(
        embeddings_dir=args.image_embeddings_dir,
        embedding_key='clip'
    )

    image_embedding_matrix = image_loader.build_embedding_matrix(
        item_to_id=data['item_to_id'],
        id_to_item=data['id_to_item']
    )

    print(f"Image embedding matrix shape: {image_embedding_matrix.shape}")

    print(f"\nCreating BERT4Rec model with late fusion (mode: {args.fusion_mode})...")
    model = BERT4RecLateFusion(
        item_num=data['num_items'],
        text_embedding_matrix=text_embedding_matrix,
        image_embedding_matrix=image_embedding_matrix,
        max_seq_len=args.max_seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        mask_prob=args.mask_prob,
        freeze_text_embeddings=args.freeze_text_embeddings,
        freeze_image_embeddings=args.freeze_image_embeddings,
        fusion_mode=args.fusion_mode,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params} total parameters")
    print(f"Trainable parameters: {trainable_params}")

    train_dataset = BERT4RecLateFusionDataset(data['train_sequences'], model)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0
    )

    print(f"\nTraining dataset size: {len(train_dataset)}")
    print(f"Number of batches per epoch: {len(train_loader)}")

    print("\nStarting training...")
    trained_model, epoch_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_data=data['val_data'],
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.save_dir,
        num_neg=args.num_neg
    )

    if args.loss_history_dir:
        import json
        os.makedirs(args.loss_history_dir, exist_ok=True)
        loss_file = os.path.join(
            args.loss_history_dir,
            f"{args.dataset_name}_late_{args.fusion_mode}_seed{args.seed}.json"
        )
        with open(loss_file, 'w') as f:
            json.dump({
                'dataset': args.dataset_name,
                'fusion_stage': 'late',
                'fusion_mode': args.fusion_mode,
                'seed': args.seed,
                'epoch_losses': epoch_losses
            }, f)
        print(f"Loss history saved to: {loss_file}")

    print(f"\nFinal evaluation ({args.eval_mode}-catalog ranking)...")
    final_metrics = evaluate_model(trained_model, data['test_data'], device,
                                   num_neg=args.num_neg,
                                   full_catalog=(args.eval_mode == 'full'))
    final_weights = trained_model.get_fusion_weights()

    print(f"\nFinal Results for {args.dataset_name} (Late Fusion - {args.fusion_mode}):")
    print(f"  HR@5: {final_metrics['HR@5']:.4f}")
    print(f"  NDCG@5: {final_metrics['NDCG@5']:.4f}")
    print(f"  HR@10: {final_metrics['HR@10']:.4f}")
    print(f"  NDCG@10: {final_metrics['NDCG@10']:.4f}")
    print(f"  HR@20: {final_metrics['HR@20']:.4f}")
    print(f"  NDCG@20: {final_metrics['NDCG@20']:.4f}")

    print(f"\nFusion Mode: late_{final_weights['fusion_mode']}")

    print("\nComputing beyond-accuracy metrics (full-catalog ranking)...")
    beyond_metrics = compute_beyond_accuracy_metrics(
        trained_model, data['test_data'], data['train_sequences'], device
    )

    print(f"\nBeyond-Accuracy Metrics for {args.dataset_name} (Late Fusion - {args.fusion_mode}):")
    for ki in [5, 10, 20]:
        print(f"  Coverage@{ki}: {beyond_metrics[f'Coverage@{ki}']:.4f}")
        print(f"  PopLift@{ki} (global):      {beyond_metrics[f'PopLift@{ki}']:+.4f}")
        print(f"  PopLift_niche@{ki}:         {beyond_metrics[f'PopLift_niche@{ki}']:+.4f}")
        print(f"  PopLift_diverse@{ki}:       {beyond_metrics[f'PopLift_diverse@{ki}']:+.4f}")
        print(f"  PopLift_blockbuster@{ki}:   {beyond_metrics[f'PopLift_blockbuster@{ki}']:+.4f}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"bert4rec_late_fusion_{args.fusion_mode}_{args.dataset_name}_final.pt")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'fusion_weights': final_weights,
            'metrics': final_metrics,
            'beyond_metrics': beyond_metrics,
            'args': vars(args)
        }, save_path)
        print(f"\nModel saved to: {save_path}")

    print("Training completed")


if __name__ == "__main__":
    main()
