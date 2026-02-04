#!/usr/bin/env python3
"""
PyTorch BERT4Rec training script with CLIP vision embeddings
Usage: python train_image.py --data_path data.csv --embeddings_dir path/to/embeddings --dataset_name amazon_games
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
from bert4rec.single_modality.model_image import BERT4RecImage, VisionEmbeddingLoader

def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BERT4RecImageDataset(Dataset):
    """Dataset class for BERT4Rec training with CLIP vision embeddings"""

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
                         test_ratio=0.2, num_users=None):
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
    test_data = []

    for user_id, sequence in user_sequences.items():
        if len(sequence) < 2:
            continue

        split_point = max(1, int(len(sequence) * (1 - test_ratio)))
        train_seq = sequence[:split_point]
        test_item = sequence[split_point] if split_point < len(sequence) else sequence[-1]

        if len(train_seq) > 0:
            train_sequences.append(train_seq)
            test_data.append((train_seq, test_item))

    print(f"Data processing complete:")
    print(f"  Users: {len(unique_users)}")
    print(f"  Items: {len(unique_items)}")
    print(f"  Average interaction length: {avg_interaction_length:.2f}")
    print(f"  Train sequences: {len(train_sequences)}")
    print(f"  Test sequences: {len(test_data)}")

    return {
        'train_sequences': train_sequences,
        'test_data': test_data,
        'num_users': len(unique_users),
        'num_items': len(unique_items),
        'avg_interaction_length': avg_interaction_length,
        'user_to_id': user_to_id,
        'item_to_id': item_to_id,
        'id_to_item': id_to_item
    }


def train_model(model, train_loader, test_data, num_epochs, learning_rate, device, save_dir=None):
    """Train the BERT4Rec model with CLIP vision embeddings"""

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    model.to(device)

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

            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            print("Evaluating...")
            metrics = evaluate_model(model, test_data, device)
            print(f"HR@10: {metrics['HR@10']:.4f}, NDCG@10: {metrics['NDCG@10']:.4f}")

    return model


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


def main():
    parser = argparse.ArgumentParser(description='Train BERT4Rec with CLIP vision embeddings')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with user_id and item_id')
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Path to directory containing CLIP vision embeddings (.npz files)')
    parser.add_argument('--dataset_name', type=str, default='amazon_games',
                        help='Dataset name (used for logging)')
    parser.add_argument('--user_col', type=str, default=None,
                        help='Name of user column (auto-detect if not specified)')
    parser.add_argument('--item_col', type=str, default=None,
                        help='Name of item column (auto-detect if not specified)')
    parser.add_argument('--num_users', type=int, default=None,
                        help='Number of users to sample (default: None = all users)')
    parser.add_argument('--max_seq_len', type=int, default=100,
                        help='Maximum sequence length (default: 50)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers (default: 3)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads (default: 1)')
    parser.add_argument('--dropout_rate', type=float, default=0.206,
                        help='Dropout rate (default: 0.395)')
    parser.add_argument('--mask_prob', type=float, default=0.149,
                        help='Masking probability (default: 0.235)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate (default: 0.0005)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--projection_dim', type=int, default=None,
                        help='Dimension to project CLIP embeddings to (default: None, uses CLIP embedding dim)')
    parser.add_argument('--freeze_embeddings', action='store_true',
                        help='Freeze CLIP embeddings during training')

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
        test_ratio=args.test_ratio,
        num_users=args.num_users
    )

    print("\nLoading CLIP vision embeddings...")
    embedding_loader = VisionEmbeddingLoader(
        embeddings_dir=args.embeddings_dir,
        embedding_key='clip'
    )

    image_embedding_matrix = embedding_loader.build_embedding_matrix(
        item_to_id=data['item_to_id'],
        id_to_item=data['id_to_item']
    )

    print(f"CLIP embedding matrix shape: {image_embedding_matrix.shape}")

    print("\nCreating BERT4Rec model with CLIP vision embeddings...")
    model = BERT4RecImage(
        item_num=data['num_items'],
        image_embedding_matrix=image_embedding_matrix,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        mask_prob=args.mask_prob,
        projection_dim=args.projection_dim,
        freeze_embeddings=args.freeze_embeddings
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    train_dataset = BERT4RecImageDataset(data['train_sequences'], model)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0
    )

    print(f"\nTraining dataset size: {len(train_dataset)}")
    print(f"Number of batches per epoch: {len(train_loader)}")

    print("\nStarting training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        test_data=data['test_data'],
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=None
    )

    print("\nFinal evaluation...")
    final_metrics = evaluate_model(trained_model, data['test_data'], device)
    print(f"Final Results for {args.dataset_name}:")
    print(f"  HR@5: {final_metrics['HR@5']:.4f}")
    print(f"  NDCG@5: {final_metrics['NDCG@5']:.4f}")
    print(f"  HR@10: {final_metrics['HR@10']:.4f}")
    print(f"  NDCG@10: {final_metrics['NDCG@10']:.4f}")
    print(f"  HR@20: {final_metrics['HR@20']:.4f}")
    print(f"  NDCG@20: {final_metrics['NDCG@20']:.4f}")
    print("Training completed")


if __name__ == "__main__":
    main()
