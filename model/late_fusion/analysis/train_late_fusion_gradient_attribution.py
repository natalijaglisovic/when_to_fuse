#!/usr/bin/env python3
"""
PyTorch BERT4Rec training script with gradient-based modality attribution analysis
for late fusion.

Each modality (item ID, text, image) has its own independent transformer encoder.
After encoding, representations are combined via the chosen fusion mode.

Supports ALL fusion modes:
- 'add': u_id + u_text + u_image (element-wise sum)
- 'concat': [u_id || u_text || u_image] -> projection
- 'attention': attention-weighted fusion of encoded modalities

Attribution Methods:
1. gradient_norm: ||d_score/d_embedding|| - measures gradient magnitude
2. gradient_input: (embedding * gradient).sum() - gradient x input product
3. integrated_gradients: approximation using absolute gradient x input

Usage (single fusion mode):
python train_late_fusion_gradient_attribution.py \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games \
    --fusion_mode add \
    --attribution_method gradient_norm \
    --num_attribution_samples 500

Usage (compare all fusion modes):
python train_late_fusion_gradient_attribution.py \
    --data_path data/games/amazon_games_user_item_image_text.csv \
    --text_embeddings_dir text_encoders/embeddings/amazon_games/sentence_transformer \
    --image_embeddings_dir vision_encoders/embeddings/amazon_games/clip_games \
    --dataset_name amazon_games \
    --compare_fusion_modes \
    --num_attribution_samples 500
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
import json
import matplotlib.pyplot as plt
import seaborn as sns

from model.early_fusion.model_early_fusion import TextEmbeddingLoader, VisionEmbeddingLoader
from model.late_fusion.analysis.model_late_fusion_gradient_attribution import (
    BERT4RecLateFusionGradientAttribution,
    analyze_attribution_results,
)


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BERT4RecLateFusionGradAttrDataset(Dataset):
    """Dataset class for BERT4Rec training with gradient attribution"""

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


def train_model(model, train_loader, test_data, val_data, num_epochs, learning_rate, device,
                attribution_method='gradient_norm', attribution_freq=5,
                num_attribution_samples=100, save_dir=None):
    """Train the BERT4Rec late fusion model with gradient attribution analysis"""

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    model.to(device)

    attribution_history = []

    print(f"\nTraining with gradient attribution analysis (Late Fusion)")
    print(f"Fusion mode: late_{model.fusion_mode}")
    print(f"Attribution method: {attribution_method}")
    print(f"Attribution frequency: every {attribution_freq} epochs")

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
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Mode: late_{model.fusion_mode}")

        # Compute attribution analysis at specified frequency
        if (epoch + 1) % attribution_freq == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"\nComputing gradient attribution (epoch {epoch+1})...")
            attr_results = model.compute_attribution_batch(
                test_data, device,
                method=attribution_method,
                num_samples=num_attribution_samples,
                verbose=True
            )

            if attr_results:
                attribution_history.append({
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'id_mean': attr_results['id']['mean'],
                    'id_std': attr_results['id']['std'],
                    'text_mean': attr_results['text']['mean'],
                    'text_std': attr_results['text']['std'],
                    'image_mean': attr_results['image']['mean'],
                    'image_std': attr_results['image']['std'],
                    'id_raw_mean': attr_results['id'].get('raw_mean', 0),
                    'text_raw_mean': attr_results['text'].get('raw_mean', 0),
                    'image_raw_mean': attr_results['image'].get('raw_mean', 0),
                    'zero_gradient_samples': attr_results.get('zero_gradient_samples', 0)
                })

                def format_contribution(val, std):
                    if val < 0.0001 and val > 0:
                        return f"{val:.2e} +/- {std:.2e}"
                    return f"{val:.4f} +/- {std:.4f}"

                print(f"  ID contribution:    {format_contribution(attr_results['id']['mean'], attr_results['id']['std'])}")
                print(f"  Text contribution:  {format_contribution(attr_results['text']['mean'], attr_results['text']['std'])}")
                print(f"  Image contribution: {format_contribution(attr_results['image']['mean'], attr_results['image']['std'])}")

                total_prop = attr_results['id']['mean'] + attr_results['text']['mean'] + attr_results['image']['mean']
                if abs(total_prop - 1.0) > 0.01:
                    print(f"  Proportions sum to {total_prop:.4f} (expected ~1.0)")

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("Evaluating on validation set...")
            metrics = evaluate_model(model, val_data, device)
            print(f"Val HR@10: {metrics['HR@10']:.4f}, Val NDCG@10: {metrics['NDCG@10']:.4f}")

    return model, attribution_history


def evaluate_model(model, test_data, device, num_neg=99, k=[5, 10, 20]):
    """Evaluate model performance"""
    model.eval()

    ndcg_sums = {ki: 0 for ki in k}
    hr_sums = {ki: 0 for ki in k}
    num_users = 0

    with torch.no_grad():
        eval_bar = tqdm(test_data, desc="Evaluating")
        for user_seq, target_item in eval_bar:
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

            rank = np.argsort(-scores)[0]

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


def visualize_attribution_history(attribution_history, save_path=None, use_stderr=True, num_samples=500):
    """Visualize attribution changes over training"""
    if not attribution_history:
        print("No attribution history to visualize")
        return

    epochs = [h['epoch'] for h in attribution_history]
    id_means = [h['id_mean'] for h in attribution_history]
    text_means = [h['text_mean'] for h in attribution_history]
    image_means = [h['image_mean'] for h in attribution_history]

    id_stds = [h['id_std'] for h in attribution_history]
    text_stds = [h['text_std'] for h in attribution_history]
    image_stds = [h['image_std'] for h in attribution_history]

    if use_stderr:
        sqrt_n = np.sqrt(num_samples)
        id_errs = [s / sqrt_n for s in id_stds]
        text_errs = [s / sqrt_n for s in text_stds]
        image_errs = [s / sqrt_n for s in image_stds]
    else:
        id_errs = id_stds
        text_errs = text_stds
        image_errs = image_stds

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.errorbar(epochs, id_means, yerr=id_errs, label='Item ID', marker='o', capsize=3)
    ax1.errorbar(epochs, text_means, yerr=text_errs, label='Text', marker='s', capsize=3)
    ax1.errorbar(epochs, image_means, yerr=image_errs, label='Image', marker='^', capsize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Attribution Proportion')
    ax1.set_title('Late Fusion: Modality Attribution Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.stackplot(epochs, id_means, text_means, image_means,
                  labels=['Item ID', 'Text', 'Image'],
                  colors=['#2ecc71', '#3498db', '#e74c3c'],
                  alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Attribution Proportion')
    ax2.set_title('Late Fusion: Stacked Modality Contributions')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attribution history plot saved to: {save_path}")
    plt.show()


def visualize_final_attribution(results, save_path=None, use_stderr=True):
    """Visualize final attribution distribution"""
    if results is None:
        print("No results to visualize")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    modalities = ['Item ID', 'Text', 'Image']
    means = [results['id']['mean'], results['text']['mean'], results['image']['mean']]
    stds = [results['id']['std'], results['text']['std'], results['image']['std']]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    if use_stderr and 'values' in results['id']:
        n = len(results['id']['values'])
        errs = [s / np.sqrt(n) for s in stds]
    else:
        errs = stds

    bars = ax1.bar(modalities, means, yerr=errs, capsize=5, color=colors, alpha=0.7)
    ax1.set_ylabel('Attribution Proportion')
    ax1.set_title(f'Late Fusion: Mean Modality Attribution')
    ax1.set_ylim(0, max(means) * 1.3)

    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    ax2 = axes[1]
    ax2.pie(means, labels=modalities, autopct='%1.1f%%', colors=colors,
            startangle=90, explode=(0.05, 0.05, 0.05))
    ax2.set_title('Contribution Distribution')

    ax3 = axes[2]
    id_vals = results['id']['values']
    text_vals = results['text']['values']
    image_vals = results['image']['values']

    ax3.hist(id_vals, bins=30, alpha=0.5, label='Item ID', color='#2ecc71')
    ax3.hist(text_vals, bins=30, alpha=0.5, label='Text', color='#3498db')
    ax3.hist(image_vals, bins=30, alpha=0.5, label='Image', color='#e74c3c')
    ax3.set_xlabel('Attribution Proportion')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Attribution Distribution')
    ax3.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Final attribution plot saved to: {save_path}")
    plt.show()


def visualize_fusion_comparison(all_results, save_path=None):
    """Visualize comparison of attribution across fusion modes."""
    if not all_results:
        print("No results to visualize")
        return

    fusion_modes = list(all_results.keys())
    modalities = ['Item ID', 'Text', 'Image']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    x = np.arange(len(fusion_modes))
    width = 0.25

    id_means = [all_results[fm]['id']['mean'] for fm in fusion_modes]
    text_means = [all_results[fm]['text']['mean'] for fm in fusion_modes]
    image_means = [all_results[fm]['image']['mean'] for fm in fusion_modes]

    ax1.bar(x - width, id_means, width, label='Item ID', color=colors[0], alpha=0.7)
    ax1.bar(x, text_means, width, label='Text', color=colors[1], alpha=0.7)
    ax1.bar(x + width, image_means, width, label='Image', color=colors[2], alpha=0.7)

    ax1.set_xlabel('Fusion Mode')
    ax1.set_ylabel('Attribution Proportion')
    ax1.set_title('Late Fusion: Modality Attribution by Mode')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Late {fm.capitalize()}' for fm in fusion_modes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    bottom_text = id_means
    bottom_image = [id_means[i] + text_means[i] for i in range(len(fusion_modes))]

    labels = [f'Late {fm.capitalize()}' for fm in fusion_modes]
    ax2.bar(labels, id_means, label='Item ID', color=colors[0], alpha=0.7)
    ax2.bar(labels, text_means, bottom=bottom_text, label='Text', color=colors[1], alpha=0.7)
    ax2.bar(labels, image_means, bottom=bottom_image, label='Image', color=colors[2], alpha=0.7)

    ax2.set_xlabel('Fusion Mode')
    ax2.set_ylabel('Attribution Proportion')
    ax2.set_title('Stacked Attribution by Fusion Mode')
    ax2.legend()

    ax3 = axes[2]
    for i, fm in enumerate(fusion_modes):
        vals = [all_results[fm]['id']['mean'],
                all_results[fm]['text']['mean'],
                all_results[fm]['image']['mean']]
        ax3.plot(modalities + [modalities[0]], vals + [vals[0]],
                 marker='o', label=f'Late {fm.capitalize()}', linewidth=2)

    ax3.set_ylabel('Attribution Proportion')
    ax3.set_title('Attribution Profile by Fusion Mode')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Fusion comparison plot saved to: {save_path}")
    plt.show()


def compare_all_fusion_modes(args, data, text_embedding_matrix, image_embedding_matrix, device):
    """Train and compare all fusion modes."""
    print("\n" + "=" * 70)
    print("COMPARING ALL LATE FUSION MODES: add, concat, attention")
    print("=" * 70)

    all_results = {}
    all_metrics = {}

    for fusion_mode in ['add', 'concat', 'attention']:
        print(f"\n{'='*70}")
        print(f"Training with late fusion_mode = '{fusion_mode}'")
        print(f"{'='*70}")

        set_random_seeds(args.seed)

        model = BERT4RecLateFusionGradientAttribution(
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
            fusion_mode=fusion_mode
        )

        train_dataset = BERT4RecLateFusionGradAttrDataset(data['train_sequences'], model)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4 if device.type == 'cuda' else 0
        )

        model, _ = train_model(
            model=model,
            train_loader=train_loader,
            test_data=data['test_data'],
            val_data=data['val_data'],
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=device,
            attribution_method=args.attribution_method,
            attribution_freq=args.num_epochs + 1,  # Only at end
            num_attribution_samples=args.num_attribution_samples
        )

        metrics = evaluate_model(model, data['test_data'], device)
        all_metrics[fusion_mode] = metrics

        attr_results = model.compute_attribution_batch(
            data['test_data'], device,
            method=args.attribution_method,
            num_samples=args.num_attribution_samples
        )
        all_results[fusion_mode] = attr_results

        analyze_attribution_results(attr_results)

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"bert4rec_late_grad_attr_{fusion_mode}_{args.dataset_name}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'attribution': {
                    'id_mean': attr_results['id']['mean'],
                    'text_mean': attr_results['text']['mean'],
                    'image_mean': attr_results['image']['mean']
                },
                'args': vars(args)
            }, save_path)

    # Print comparison summary
    print("\n" + "=" * 70)
    print("LATE FUSION MODE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Mode':<12} {'HR@10':<10} {'NDCG@10':<10} {'ID':<10} {'Text':<10} {'Image':<10}")
    print("-" * 70)
    for fm in ['add', 'concat', 'attention']:
        print(f"{fm:<12} "
              f"{all_metrics[fm]['HR@10']:<10.4f} "
              f"{all_metrics[fm]['NDCG@10']:<10.4f} "
              f"{all_results[fm]['id']['mean']:<10.4f} "
              f"{all_results[fm]['text']['mean']:<10.4f} "
              f"{all_results[fm]['image']['mean']:<10.4f}")
    print("=" * 70)

    if args.save_dir:
        comparison_path = os.path.join(args.save_dir, f"late_fusion_comparison_{args.dataset_name}.png")
        visualize_fusion_comparison(all_results, save_path=comparison_path)

        comparison_json = os.path.join(args.save_dir, f"late_fusion_comparison_{args.dataset_name}.json")
        comparison_data = {
            fm: {
                'metrics': all_metrics[fm],
                'attribution': {
                    'id_mean': float(all_results[fm]['id']['mean']),
                    'id_std': float(all_results[fm]['id']['std']),
                    'text_mean': float(all_results[fm]['text']['mean']),
                    'text_std': float(all_results[fm]['text']['std']),
                    'image_mean': float(all_results[fm]['image']['mean']),
                    'image_std': float(all_results[fm]['image']['std'])
                }
            }
            for fm in ['add', 'concat', 'attention']
        }
        with open(comparison_json, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"Comparison results saved to: {comparison_json}")
    else:
        visualize_fusion_comparison(all_results)

    return all_results, all_metrics


def main():
    parser = argparse.ArgumentParser(description='Train BERT4Rec late fusion with gradient attribution analysis')
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
                        help='Maximum sequence length (default: 25)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers per modality (default: 4)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads (default: 1)')
    parser.add_argument('--dropout_rate', type=float, default=0.206,
                        help='Dropout rate (default: 0.432)')
    parser.add_argument('--mask_prob', type=float, default=0.149,
                        help='Masking probability (default: 0.224)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--freeze_text_embeddings', action='store_true',
                        help='Freeze text embeddings during training')
    parser.add_argument('--freeze_image_embeddings', action='store_true',
                        help='Freeze image embeddings during training')
    parser.add_argument('--attribution_method', type=str, default='gradient_norm',
                        choices=['gradient_norm', 'gradient_input', 'integrated_gradients'],
                        help='Attribution method (default: gradient_norm)')
    parser.add_argument('--attribution_freq', type=int, default=5,
                        help='Compute attribution every N epochs (default: 5)')
    parser.add_argument('--num_attribution_samples', type=int, default=500,
                        help='Number of samples for attribution analysis (default: 500)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save model and results')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pre-trained model to load (skip training)')
    parser.add_argument('--fusion_mode', type=str, default='add',
                        choices=['add', 'concat', 'attention'],
                        help='Fusion mode: add, concat, or attention (default: add)')
    parser.add_argument('--compare_fusion_modes', action='store_true',
                        help='Train and compare all fusion modes (add, concat, attention)')

    args = parser.parse_args()

    set_random_seeds(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load and process data
    data = load_and_process_data(
        csv_path=args.data_path,
        user_col=args.user_col,
        item_col=args.item_col,
        num_users=args.num_users
    )

    # Load text embeddings
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

    # Load image embeddings
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

    # Handle compare_fusion_modes flag
    if args.compare_fusion_modes:
        compare_all_fusion_modes(args, data, text_embedding_matrix, image_embedding_matrix, device)
        return

    # Create model
    print(f"\nCreating BERT4Rec late fusion model with gradient attribution support (fusion_mode={args.fusion_mode})...")
    model = BERT4RecLateFusionGradientAttribution(
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
        fusion_mode=args.fusion_mode
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params} total parameters")
    print(f"Trainable parameters: {trainable_params}")

    # Load pre-trained model if specified
    if args.load_model:
        print(f"\nLoading pre-trained model from: {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        attribution_history = []
    else:
        train_dataset = BERT4RecLateFusionGradAttrDataset(data['train_sequences'], model)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4 if device.type == 'cuda' else 0
        )

        print(f"\nTraining dataset size: {len(train_dataset)}")
        print(f"Number of batches per epoch: {len(train_loader)}")

        print("\nStarting training...")
        model, attribution_history = train_model(
            model=model,
            train_loader=train_loader,
            test_data=data['test_data'],
            val_data=data['val_data'],
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=device,
            attribution_method=args.attribution_method,
            attribution_freq=args.attribution_freq,
            num_attribution_samples=args.num_attribution_samples,
            save_dir=args.save_dir
        )

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_model(model, data['test_data'], device)

    print(f"\nFinal Results for {args.dataset_name} (Late Fusion - {args.fusion_mode}):")
    print(f"  Fusion Mode: late_{args.fusion_mode}")
    print(f"  HR@5: {final_metrics['HR@5']:.4f}")
    print(f"  NDCG@5: {final_metrics['NDCG@5']:.4f}")
    print(f"  HR@10: {final_metrics['HR@10']:.4f}")
    print(f"  NDCG@10: {final_metrics['NDCG@10']:.4f}")
    print(f"  HR@20: {final_metrics['HR@20']:.4f}")
    print(f"  NDCG@20: {final_metrics['NDCG@20']:.4f}")

    # Comprehensive final attribution analysis
    print("\n" + "=" * 60)
    print("FINAL GRADIENT ATTRIBUTION ANALYSIS (LATE FUSION)")
    print("=" * 60)

    for method in ['gradient_norm', 'gradient_input', 'integrated_gradients']:
        print(f"\nMethod: {method}")
        final_attr = model.compute_attribution_batch(
            data['test_data'], device,
            method=method,
            num_samples=args.num_attribution_samples
        )

        if final_attr:
            analysis = analyze_attribution_results(final_attr)

    # Use the specified method for visualization
    final_attr_for_viz = model.compute_attribution_batch(
        data['test_data'], device,
        method=args.attribution_method,
        num_samples=args.num_attribution_samples
    )

    # Save results
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

        save_path = os.path.join(args.save_dir, f"bert4rec_late_grad_attr_{args.fusion_mode}_{args.dataset_name}_final.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': final_metrics,
            'fusion_mode': args.fusion_mode,
            'args': vars(args)
        }, save_path)
        print(f"\nModel saved to: {save_path}")

        if attribution_history:
            history_path = os.path.join(args.save_dir, f"late_attribution_history_{args.fusion_mode}_{args.dataset_name}.json")
            with open(history_path, 'w') as f:
                json.dump(attribution_history, f, indent=2)
            print(f"Attribution history saved to: {history_path}")

            history_plot_path = os.path.join(args.save_dir, f"late_attribution_history_{args.fusion_mode}_{args.dataset_name}.png")
            visualize_attribution_history(attribution_history, save_path=history_plot_path,
                                         use_stderr=True, num_samples=args.num_attribution_samples)

        if final_attr_for_viz:
            attr_path = os.path.join(args.save_dir, f"late_final_attribution_{args.fusion_mode}_{args.dataset_name}.json")
            analyze_attribution_results(final_attr_for_viz, save_path=attr_path)

            final_plot_path = os.path.join(args.save_dir, f"late_final_attribution_{args.fusion_mode}_{args.dataset_name}.png")
            visualize_final_attribution(final_attr_for_viz, save_path=final_plot_path)
    else:
        if attribution_history:
            visualize_attribution_history(attribution_history, use_stderr=True,
                                         num_samples=args.num_attribution_samples)
        if final_attr_for_viz:
            visualize_final_attribution(final_attr_for_viz)

    print("\nTraining and analysis completed")


if __name__ == "__main__":
    main()
