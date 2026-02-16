#!/usr/bin/env python3
"""
Robustness evaluation framework for BERT4Rec early fusion models.

Evaluates how well each fusion method (concat, add, attention) handles missing
multimodal features. For each missing percentage [10%, 20%, ..., 90%], randomly
selects that fraction of items and replaces their text/image embeddings with the
global mean of the non-missing items. Each setting is repeated 5 times with
different random seeds.

Usage:
python -m bert4rec.early_fusion.robustness_eval \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games
"""
import argparse
import json
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

from bert4rec.early_fusion.model_early_fusion import (
    BERT4RecEarlyFusion, TextEmbeddingLoader, VisionEmbeddingLoader
)
from bert4rec.early_fusion.model_early_fusion_attention import (
    BERT4RecEarlyFusionAttention
)


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BERT4RecDatasetWrapper(Dataset):
    def __init__(self, sequences, model):
        self.sequences = sequences
        self.model = model

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        masked_data = self.model.create_masked_sequences([self.sequences[idx]])
        return {
            'input_ids': masked_data['input_ids'][0],
            'masked_positions': masked_data['masked_positions'][0],
            'masked_labels': masked_data['masked_labels'][0]
        }


def load_and_process_data(csv_path, user_col=None, item_col=None,
                          test_ratio=0.2, num_users=None):
    df = pd.read_csv(csv_path)
    print(f"Data shape: {df.shape}, Columns: {df.columns.tolist()}")

    if user_col is None:
        for col in df.columns:
            if any(k in col.lower() for k in ['user', 'customer', 'buyer']):
                user_col = col
                break
    if item_col is None:
        for col in df.columns:
            if any(k in col.lower() for k in ['item', 'product', 'asin', 'movie', 'book']):
                item_col = col
                break
    if user_col is None or item_col is None:
        raise ValueError(f"Cannot auto-detect columns. Available: {df.columns.tolist()}")

    print(f"Using columns: user='{user_col}', item='{item_col}'")

    if num_users and len(df[user_col].unique()) > num_users:
        sampled = np.random.choice(df[user_col].unique(), num_users, replace=False)
        df = df[df[user_col].isin(sampled)]

    unique_users = sorted(df[user_col].unique())
    unique_items = sorted(df[item_col].unique())
    user_to_id = {u: i for i, u in enumerate(unique_users)}
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

    train_sequences, test_data = [], []
    for user_id, seq in user_sequences.items():
        if len(seq) < 2:
            continue
        split = max(1, int(len(seq) * (1 - test_ratio)))
        train_seq = seq[:split]
        test_item = seq[split] if split < len(seq) else seq[-1]
        train_sequences.append(train_seq)
        test_data.append((train_seq, test_item))

    print(f"Users: {len(unique_users)}, Items: {len(unique_items)}, "
          f"Train: {len(train_sequences)}, Test: {len(test_data)}")

    return {
        'train_sequences': train_sequences,
        'test_data': test_data,
        'num_items': len(unique_items),
        'item_to_id': item_to_id,
        'id_to_item': id_to_item,
    }


def apply_global_mean_imputation(embedding_matrix, missing_pct, seed):
    """Replace `missing_pct` fraction of item embeddings with the global mean.

    Items are 1-indexed; row 0 is padding and the last row is the mask token.
    Only real item rows (1..num_items) are candidates for removal.
    """
    rng = np.random.RandomState(seed)
    matrix = embedding_matrix.copy()
    num_items = matrix.shape[0] - 2  # exclude padding (0) and mask token (last)
    item_indices = np.arange(1, num_items + 1)

    num_missing = int(num_items * missing_pct)
    missing_items = rng.choice(item_indices, size=num_missing, replace=False)

    non_missing_mask = np.ones(matrix.shape[0], dtype=bool)
    non_missing_mask[0] = False              # padding
    non_missing_mask[-1] = False             # mask token
    non_missing_mask[missing_items] = False

    global_mean = matrix[non_missing_mask].mean(axis=0)
    matrix[missing_items] = global_mean

    return matrix, missing_items


def train_and_evaluate(model, train_sequences, test_data, eval_candidates, args, device):
    """Train model and return final evaluation metrics."""
    dataset = BERT4RecDatasetWrapper(train_sequences, model)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    model.to(device)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss, num_batches = 0, 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False):
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

        avg_loss = total_loss / max(num_batches, 1)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} loss: {avg_loss:.4f}")

    # Final evaluation
    metrics = evaluate_model(model, test_data, eval_candidates, device)
    return metrics


def pregenerate_eval_candidates(test_data, num_items, num_neg=99, seed=123):
    """Pre-generate fixed negative candidates for all test users.

    This ensures every experiment is evaluated on the exact same candidates,
    removing the dependency on Python random state after training.
    """
    rng = random.Random(seed)
    all_candidates = []
    for user_seq, target_item in test_data:
        candidates = [target_item]
        user_set = set(user_seq)
        while len(candidates) <= num_neg:
            neg = rng.randint(1, num_items)
            if neg != target_item and neg not in user_set:
                candidates.append(neg)
        all_candidates.append(candidates)
    return all_candidates


def evaluate_model(model, test_data, eval_candidates, device, k_values=[5, 10, 20]):
    model.eval()
    hr_sums = {ki: 0 for ki in k_values}
    ndcg_sums = {ki: 0 for ki in k_values}
    num_users = 0

    with torch.no_grad():
        for (user_seq, target_item), candidates in tqdm(
            zip(test_data, eval_candidates), total=len(test_data),
            desc="Evaluating", leave=False
        ):
            seq = user_seq[-model.max_seq_len:]
            padded = seq + [0] * (model.max_seq_len - len(seq))

            sequences = torch.tensor([padded], dtype=torch.long, device=device)
            candidates_t = torch.tensor([candidates], dtype=torch.long, device=device)
            scores = model.predict(sequences, candidates_t).cpu().numpy()[0]

            rank = np.argsort(-scores)[0]
            for ki in k_values:
                if rank < ki:
                    hr_sums[ki] += 1
                    ndcg_sums[ki] += 1 / np.log2(rank + 2)
            num_users += 1

    results = {}
    for ki in k_values:
        results[f'HR@{ki}'] = hr_sums[ki] / max(num_users, 1)
        results[f'NDCG@{ki}'] = ndcg_sums[ki] / max(num_users, 1)
    return results


def create_model(fusion_method, num_items, text_matrix, image_matrix, args, device):
    """Create a fresh model for the given fusion method."""
    if fusion_method in ('add', 'concat'):
        model = BERT4RecEarlyFusion(
            item_num=num_items,
            text_embedding_matrix=text_matrix,
            image_embedding_matrix=image_matrix,
            max_seq_len=args.max_seq_len,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            mask_prob=args.mask_prob,
            freeze_text_embeddings=args.freeze_text_embeddings,
            freeze_image_embeddings=args.freeze_image_embeddings,
            fusion_mode=fusion_method,
        )
    elif fusion_method == 'attention':
        model = BERT4RecEarlyFusionAttention(
            item_num=num_items,
            text_embedding_matrix=text_matrix,
            image_embedding_matrix=image_matrix,
            max_seq_len=args.max_seq_len,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            mask_prob=args.mask_prob,
            freeze_text_embeddings=args.freeze_text_embeddings,
            freeze_image_embeddings=args.freeze_image_embeddings,
        )
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Robustness evaluation with global mean imputation')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--text_embeddings_dir', type=str, required=True)
    parser.add_argument('--image_embeddings_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='amazon_games')
    parser.add_argument('--user_col', type=str, default=None)
    parser.add_argument('--item_col', type=str, default=None)
    parser.add_argument('--num_users', type=int, default=None)
    parser.add_argument('--max_seq_len', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.206)
    parser.add_argument('--mask_prob', type=float, default=0.149)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--freeze_text_embeddings', action='store_true')
    parser.add_argument('--freeze_image_embeddings', action='store_true')
    parser.add_argument('--output_dir', type=str, default='robustness_results',
                        help='Directory to save results JSON and plots')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs per setting (default: 1)')
    parser.add_argument('--missing_percentages', type=str, default='10,20,30,40,50,60,70,80,90',
                        help='Comma-separated missing percentages (default: 10,20,...,90)')
    parser.add_argument('--fusion_methods', type=str, default='concat,add,attention',
                        help='Comma-separated fusion methods (default: concat,add,attention)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    missing_pcts = [int(x) / 100.0 for x in args.missing_percentages.split(',')]
    fusion_methods = args.fusion_methods.split(',')
    print(f"Missing percentages: {[f'{p:.0%}' for p in missing_pcts]}")
    print(f"Fusion methods: {fusion_methods}")
    print(f"Runs per setting: {args.num_runs}")

    # Load data once
    set_random_seeds(42)
    data = load_and_process_data(
        csv_path=args.data_path,
        user_col=args.user_col,
        item_col=args.item_col,
        test_ratio=args.test_ratio,
        num_users=args.num_users
    )

    # Load original embedding matrices once
    print("\nLoading text embeddings...")
    text_loader = TextEmbeddingLoader(
        embeddings_dir=args.text_embeddings_dir,
        embedding_key='sentence_transformer'
    )
    text_matrix_orig = text_loader.build_embedding_matrix(
        item_to_id=data['item_to_id'],
        id_to_item=data['id_to_item']
    )
    print(f"Text embedding matrix shape: {text_matrix_orig.shape}")

    print("Loading image embeddings...")
    image_loader = VisionEmbeddingLoader(
        embeddings_dir=args.image_embeddings_dir,
        embedding_key='clip'
    )
    image_matrix_orig = image_loader.build_embedding_matrix(
        item_to_id=data['item_to_id'],
        id_to_item=data['id_to_item']
    )
    print(f"Image embedding matrix shape: {image_matrix_orig.shape}")

    # Pre-generate evaluation candidates ONCE so all experiments use the same negatives
    print("\nPre-generating evaluation candidates...")
    eval_candidates = pregenerate_eval_candidates(
        data['test_data'], data['num_items'], num_neg=99, seed=123
    )
    print(f"Generated candidates for {len(eval_candidates)} test users")

    # Results storage: {fusion_method: {missing_pct: [hr@20 for each run]}}
    results = {fm: {pct: [] for pct in missing_pcts} for fm in fusion_methods}
    all_metrics = []  # detailed log

    total_experiments = len(missing_pcts) * len(fusion_methods) * args.num_runs
    experiment_num = 0

    for pct in missing_pcts:
        for run_idx in range(args.num_runs):
            # Use a different seed per run but same seed across fusion methods
            # so they share the same missing items
            imputation_seed = 1000 * run_idx + int(pct * 100)

            text_matrix_imputed, missing_items = apply_global_mean_imputation(
                text_matrix_orig, pct, seed=imputation_seed
            )
            image_matrix_imputed, _ = apply_global_mean_imputation(
                image_matrix_orig, pct, seed=imputation_seed
            )

            for fm in fusion_methods:
                experiment_num += 1
                print(f"\n{'='*60}")
                print(f"[{experiment_num}/{total_experiments}] "
                      f"Missing: {pct:.0%}, Run: {run_idx+1}/{args.num_runs}, "
                      f"Fusion: {fm}")
                print(f"{'='*60}")

                # Set a training seed that varies per run but is consistent
                training_seed = 42 + run_idx
                set_random_seeds(training_seed)

                model = create_model(
                    fm, data['num_items'],
                    text_matrix_imputed, image_matrix_imputed,
                    args, device
                )

                metrics = train_and_evaluate(
                    model, data['train_sequences'], data['test_data'],
                    eval_candidates, args, device
                )

                hr20 = metrics['HR@20']
                results[fm][pct].append(hr20)
                all_metrics.append({
                    'fusion_method': fm,
                    'missing_pct': pct,
                    'run': run_idx,
                    'seed': imputation_seed,
                    **metrics
                })

                print(f"  HR@20: {hr20:.4f}")

                # Free memory
                del model
                torch.cuda.empty_cache() if device.type == 'cuda' else None

    # Compute summary statistics
    summary = {}
    for fm in fusion_methods:
        summary[fm] = {}
        for pct in missing_pcts:
            vals = results[fm][pct]
            summary[fm][f'{pct:.0%}'] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'runs': [float(v) for v in vals]
            }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f'robustness_{args.dataset_name}.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': summary,
            'all_metrics': all_metrics,
            'config': {
                'missing_percentages': [f'{p:.0%}' for p in missing_pcts],
                'fusion_methods': fusion_methods,
                'num_runs': args.num_runs,
                'dataset': args.dataset_name,
            }
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: HR@20 (mean +/- std over {args.num_runs} runs)")
    print(f"{'='*70}")
    header = f"{'Missing %':>10}"
    for fm in fusion_methods:
        header += f"  {fm:>20}"
    print(header)
    print('-' * 70)
    for pct in missing_pcts:
        row = f"{pct:>9.0%}"
        for fm in fusion_methods:
            m = np.mean(results[fm][pct])
            s = np.std(results[fm][pct])
            row += f"  {m:>8.4f} +/- {s:.4f}"
        print(row)

    # Generate plot
    plot_results(results, missing_pcts, fusion_methods, args.dataset_name, args.output_dir)


def plot_results(results, missing_pcts, fusion_methods, dataset_name, output_dir):
    """Generate line plot of HR@20 vs missing percentage for each fusion method."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    style_map = {
        'concat': {'color': '#1f77b4', 'marker': 'o', 'label': 'Concatenation'},
        'add': {'color': '#ff7f0e', 'marker': 's', 'label': 'Addition'},
        'attention': {'color': '#2ca02c', 'marker': '^', 'label': 'Attention'},
    }

    pct_labels = [int(p * 100) for p in missing_pcts]

    for fm in fusion_methods:
        means = [np.mean(results[fm][p]) for p in missing_pcts]
        stds = [np.std(results[fm][p]) for p in missing_pcts]

        style = style_map.get(fm, {'color': 'gray', 'marker': 'D', 'label': fm})
        ax.errorbar(
            pct_labels, means, yerr=stds,
            marker=style['marker'], color=style['color'],
            label=style['label'], linewidth=2, markersize=7,
            capsize=4, capthick=1.5
        )

    ax.set_xlabel('Missing Multimodal Features (%)', fontsize=13)
    ax.set_ylabel('Recall@20', fontsize=13)
    ax.set_title(f'Robustness to Missing Features — {dataset_name}\n'
                 f'(Global Mean Imputation, BERT4Rec Early Fusion)', fontsize=13)
    ax.set_xticks(pct_labels)
    ax.set_xticklabels([f'{p}%' for p in pct_labels])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, f'robustness_{dataset_name}.pdf')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plot_path_png = os.path.join(output_dir, f'robustness_{dataset_name}.png')
    fig.savefig(plot_path_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {plot_path} and {plot_path_png}")


if __name__ == '__main__':
    main()
