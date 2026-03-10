#!/usr/bin/env python3
"""
BERT4Rec with Intermediate Fusion and Gradient-Based Modality Attribution

This module extends the intermediate fusion BERT4Rec model to support gradient-based
attribution analysis for ALL fusion types:
- 'add': u_id + u_text + u_image (element-wise sum after independent pre-fusion encoding)
- 'concat': [u_id || u_text || u_image] -> projection (after independent pre-fusion encoding)
- 'attention': attention-weighted fusion (query from ID encoder, key/value from text/image encoders)

Key difference from early and late fusion attribution:
In intermediate fusion, each modality has its OWN pre-fusion transformer encoder,
followed by a fusion layer, then SHARED post-fusion transformer layers. Gradients flow:
    score -> post_fusion_layers -> fusion_layer -> pre_fusion_encoder_m -> raw_embedding_m
So attribution captures the pre-fusion encoder's learned representations, the fusion
mechanism, AND the post-fusion shared processing.

Attribution Methods:
1. gradient_norm: ||d_score/d_embedding|| - measures gradient magnitude
2. gradient_input: (embedding * gradient).sum() - gradient x input product
3. integrated_gradients: approximation using absolute gradient x input
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import math
import os

from bert4rec.early_fusion.model_early_fusion import (
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    TextEmbeddingLoader,
    VisionEmbeddingLoader,
)
from bert4rec.intermediate_fusion.model_intermediate_fusion import ModalityEncoder
from bert4rec.intermediate_fusion.model_intermediate_fusion_attention import IntermediateFusionAttention


class BERT4RecIntermediateFusionGradientAttribution(nn.Module):
    """
    BERT4Rec with intermediate fusion supporting gradient-based modality attribution.

    Each modality (item ID, text, image) has its own independent pre-fusion transformer
    encoder. After encoding, representations are combined via the chosen fusion mode,
    then processed by shared post-fusion transformer layers.

    Fusion modes:
    - 'add': u_id + u_text + u_image
    - 'concat': [u_id || u_text || u_image] -> projection to hidden_dim
    - 'attention': attention-weighted fusion (query from ID, key/value from text/image)
    """

    def __init__(self,
                 item_num,
                 text_embedding_matrix,
                 image_embedding_matrix,
                 max_seq_len=50,
                 hidden_dim=64,
                 num_pre_fusion_layers=2,
                 num_post_fusion_layers=2,
                 num_heads=2,
                 dropout_rate=0.1,
                 mask_prob=0.15,
                 freeze_text_embeddings=True,
                 freeze_image_embeddings=True,
                 fusion_mode='add'):
        super(BERT4RecIntermediateFusionGradientAttribution, self).__init__()

        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob
        self.fusion_mode = fusion_mode
        self.num_pre_fusion_layers = num_pre_fusion_layers
        self.num_post_fusion_layers = num_post_fusion_layers

        if fusion_mode not in ['add', 'concat', 'attention']:
            raise ValueError(f"fusion_mode must be 'add', 'concat', or 'attention', got '{fusion_mode}'")

        self.pad_token = 0
        self.mask_token = item_num + 1
        self.vocab_size = item_num + 2

        self.text_dim = text_embedding_matrix.shape[1]
        self.image_dim = image_embedding_matrix.shape[1]

        # --- Embedding layers ---
        self.item_id_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)

        self.text_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(text_embedding_matrix),
            freeze=freeze_text_embeddings,
            padding_idx=0
        )

        self.image_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(image_embedding_matrix),
            freeze=freeze_image_embeddings,
            padding_idx=0
        )

        # --- Stage 1: Independent modality-specific transformer encoders (pre-fusion) ---
        self.id_encoder = ModalityEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_layers=num_pre_fusion_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        self.text_encoder = ModalityEncoder(
            input_dim=self.text_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_layers=num_pre_fusion_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        self.image_encoder = ModalityEncoder(
            input_dim=self.image_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_layers=num_pre_fusion_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        # --- Stage 2: Fusion layer ---
        if fusion_mode == 'concat':
            self.fusion_projection = nn.Linear(3 * hidden_dim, hidden_dim)
        elif fusion_mode == 'attention':
            self.fusion_attention = IntermediateFusionAttention(
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
            )

        # --- Stage 3: Shared post-fusion transformer layers ---
        ff_dim = hidden_dim * 4
        self.post_fusion_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_post_fusion_layers)
        ])
        self.post_fusion_norm = nn.LayerNorm(hidden_dim)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, item_num)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if module is self.text_embedding or module is self.image_embedding:
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def fuse_representations(self, id_hidden, text_hidden, image_hidden):
        """Intermediate fusion: combine modality representations after pre-fusion encoding."""
        if self.fusion_mode == 'add':
            fused = id_hidden + text_hidden + image_hidden
        elif self.fusion_mode == 'concat':
            concat = torch.cat([id_hidden, text_hidden, image_hidden], dim=-1)
            fused = self.fusion_projection(concat)
        elif self.fusion_mode == 'attention':
            fused = self.fusion_attention(id_hidden, text_hidden, image_hidden)
        return fused

    def forward(self, input_ids, masked_positions=None):
        batch_size, seq_len = input_ids.size()
        attention_mask = (input_ids != self.pad_token).long()

        # Get embeddings for each modality
        id_emb = self.item_id_embedding(input_ids)
        text_emb = self.text_embedding(input_ids)
        image_emb = self.image_embedding(input_ids)

        # Stage 1: Encode each modality independently (pre-fusion)
        id_hidden = self.id_encoder(id_emb, attention_mask)
        text_hidden = self.text_encoder(text_emb, attention_mask)
        image_hidden = self.image_encoder(image_emb, attention_mask)

        # Stage 2: Intermediate fusion
        hidden_states = self.fuse_representations(id_hidden, text_hidden, image_hidden)

        # Stage 3: Shared post-fusion transformer layers
        for transformer in self.post_fusion_blocks:
            hidden_states = transformer(hidden_states, attention_mask)

        hidden_states = self.post_fusion_norm(hidden_states)

        if masked_positions is not None:
            batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
            batch_indices = batch_indices.expand(-1, masked_positions.size(1))

            gather_indices = torch.stack([batch_indices, masked_positions], dim=-1)

            batch_flat = gather_indices[:, :, 0].flatten()
            pos_flat = gather_indices[:, :, 1].flatten()

            masked_hidden = hidden_states[batch_flat, pos_flat]
            masked_hidden = masked_hidden.view(batch_size, masked_positions.size(1), self.hidden_dim)

            logits = self.output_layer(masked_hidden)
            return logits
        else:
            return hidden_states

    def forward_with_separate_embeddings(self, id_emb, text_emb, image_emb, input_ids):
        """
        Forward pass with pre-computed (detached) embeddings for gradient attribution.

        Each embedding is passed through its own pre-fusion encoder, fused,
        then processed by shared post-fusion layers.
        """
        batch_size, seq_len = input_ids.size()
        attention_mask = (input_ids != self.pad_token).long()

        # Stage 1: Encode each modality independently through its own transformer
        id_hidden = self.id_encoder(id_emb, attention_mask)
        text_hidden = self.text_encoder(text_emb, attention_mask)
        image_hidden = self.image_encoder(image_emb, attention_mask)

        # Stage 2: Intermediate fusion
        hidden_states = self.fuse_representations(id_hidden, text_hidden, image_hidden)

        # Stage 3: Shared post-fusion transformer layers
        for transformer in self.post_fusion_blocks:
            hidden_states = transformer(hidden_states, attention_mask)

        hidden_states = self.post_fusion_norm(hidden_states)

        return hidden_states

    def compute_modality_attribution(self, input_ids, target_item, method='gradient_norm'):
        """
        Compute gradient-based attribution for each modality in intermediate fusion.

        In intermediate fusion, gradients flow through:
            score -> post_fusion_layers -> fusion -> pre_fusion_encoder_m -> raw_embedding_m

        Args:
            input_ids: (batch_size, seq_len) - input sequences
            target_item: int - target item ID
            method: str - attribution method

        Returns:
            dict with 'id', 'text', 'image' attribution scores
        """
        self.eval()

        # Zero model gradients to prevent accumulation across samples
        self.zero_grad()

        batch_size, seq_len = input_ids.size()

        with torch.enable_grad():
            # Get raw embeddings
            id_emb = self.item_id_embedding(input_ids)
            text_emb_raw = self.text_embedding(input_ids)
            image_emb_raw = self.image_embedding(input_ids)

            # Enable gradients on raw embeddings
            id_emb = id_emb.detach().requires_grad_(True)
            text_emb_raw = text_emb_raw.detach().requires_grad_(True)
            image_emb_raw = image_emb_raw.detach().requires_grad_(True)

            # Forward pass through pre-fusion encoders, fusion, and post-fusion layers
            hidden_states = self.forward_with_separate_embeddings(
                id_emb, text_emb_raw, image_emb_raw, input_ids
            )

            # Get last position hidden states
            last_positions = []
            for i in range(batch_size):
                valid_positions = (input_ids[i] != self.pad_token).nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    last_positions.append(valid_positions[-1].item())
                else:
                    last_positions.append(0)

            batch_indices = torch.arange(batch_size, device=input_ids.device)
            last_hidden = hidden_states[batch_indices, last_positions]

            # Get target embedding (fused through pre-fusion encoders)
            target_ids = torch.tensor([[target_item]], device=input_ids.device).expand(batch_size, 1)
            target_id_emb = self.item_id_embedding(target_ids)
            target_text_emb = self.text_embedding(target_ids)
            target_image_emb = self.image_embedding(target_ids)

            attention_mask_target = torch.ones(batch_size, 1, device=input_ids.device).long()
            target_id_hidden = self.id_encoder(target_id_emb, attention_mask_target)
            target_text_hidden = self.text_encoder(target_text_emb, attention_mask_target)
            target_image_hidden = self.image_encoder(target_image_emb, attention_mask_target)

            target_emb = self.fuse_representations(target_id_hidden, target_text_hidden, target_image_hidden).squeeze(1)

            # Normalize and compute cosine similarity (bounded [-1, 1] for stable gradients)
            last_hidden_norm = F.normalize(last_hidden, p=2, dim=-1)
            target_emb_norm = F.normalize(target_emb, p=2, dim=-1)
            score = (last_hidden_norm * target_emb_norm).sum()

            # Backpropagate
            score.backward()

        # Get gradients (handle None for frozen embeddings)
        id_grad = id_emb.grad if id_emb.grad is not None else torch.zeros_like(id_emb)
        text_grad = text_emb_raw.grad if text_emb_raw.grad is not None else torch.zeros_like(text_emb_raw)
        image_grad = image_emb_raw.grad if image_emb_raw.grad is not None else torch.zeros_like(image_emb_raw)

        if method == 'gradient_norm':
            id_attr = id_grad.double().norm(dim=-1).mean().item()
            text_attr = text_grad.double().norm(dim=-1).mean().item()
            image_attr = image_grad.double().norm(dim=-1).mean().item()

        elif method == 'gradient_input':
            id_attr = (id_grad.double() * id_emb.double()).sum().item()
            text_attr = (text_grad.double() * text_emb_raw.double()).sum().item()
            image_attr = (image_grad.double() * image_emb_raw.double()).sum().item()

        elif method == 'integrated_gradients':
            id_attr = (id_grad.double() * id_emb.double()).abs().sum().item()
            text_attr = (text_grad.double() * text_emb_raw.double()).abs().sum().item()
            image_attr = (image_grad.double() * image_emb_raw.double()).abs().sum().item()

        else:
            raise ValueError(f"Unknown attribution method: {method}")

        # Handle numerical edge cases
        min_threshold = 1e-12
        if abs(id_attr) < min_threshold and abs(text_attr) < min_threshold and abs(image_attr) < min_threshold:
            return {
                'id': 0.0,
                'text': 0.0,
                'image': 0.0,
                'id_proportion': 1/3,
                'text_proportion': 1/3,
                'image_proportion': 1/3,
                'total': 0.0,
                'warning': 'all_gradients_near_zero'
            }

        # Normalize to get proportions
        total = abs(id_attr) + abs(text_attr) + abs(image_attr)

        if total < min_threshold:
            total = min_threshold

        return {
            'id': id_attr,
            'text': text_attr,
            'image': image_attr,
            'id_proportion': abs(id_attr) / total,
            'text_proportion': abs(text_attr) / total,
            'image_proportion': abs(image_attr) / total,
            'total': total
        }

    def compute_attribution_batch(self, test_data, device, method='gradient_norm', num_samples=None, verbose=False):
        """
        Compute attribution scores across multiple test samples.

        Args:
            test_data: list of (sequence, target_item) tuples
            device: torch device
            method: attribution method
            num_samples: number of samples to process (None = all)
            verbose: if True, print diagnostic info

        Returns:
            dict with aggregated attribution statistics
        """
        self.eval()

        all_attributions = []
        zero_gradient_count = 0
        samples = test_data[:num_samples] if num_samples else test_data

        for user_seq, target_item in tqdm(samples, desc=f"Computing attributions (intermediate_{self.fusion_mode})"):
            if len(user_seq) > self.max_seq_len:
                user_seq = user_seq[-self.max_seq_len:]

            padded_seq = user_seq + [0] * (self.max_seq_len - len(user_seq))
            input_ids = torch.tensor([padded_seq], dtype=torch.long, device=device)

            try:
                attr = self.compute_modality_attribution(input_ids, target_item, method=method)
                all_attributions.append(attr)

                if attr.get('warning') == 'all_gradients_near_zero':
                    zero_gradient_count += 1

            except Exception as e:
                print(f"Warning: Attribution computation failed: {e}")
                continue

        if not all_attributions:
            return None

        if zero_gradient_count > 0:
            print(f"  Warning: {zero_gradient_count}/{len(all_attributions)} samples had near-zero gradients for all modalities")

        # Aggregate statistics
        id_attrs = [a['id_proportion'] for a in all_attributions]
        text_attrs = [a['text_proportion'] for a in all_attributions]
        image_attrs = [a['image_proportion'] for a in all_attributions]

        id_raw = [a['id'] for a in all_attributions]
        text_raw = [a['text'] for a in all_attributions]
        image_raw = [a['image'] for a in all_attributions]

        if verbose:
            print(f"  Raw gradient magnitudes:")
            print(f"    ID:    mean={np.mean(np.abs(id_raw)):.2e}, min={np.min(np.abs(id_raw)):.2e}, max={np.max(np.abs(id_raw)):.2e}")
            print(f"    Text:  mean={np.mean(np.abs(text_raw)):.2e}, min={np.min(np.abs(text_raw)):.2e}, max={np.max(np.abs(text_raw)):.2e}")
            print(f"    Image: mean={np.mean(np.abs(image_raw)):.2e}, min={np.min(np.abs(image_raw)):.2e}, max={np.max(np.abs(image_raw)):.2e}")

        sum_check = np.mean(id_attrs) + np.mean(text_attrs) + np.mean(image_attrs)
        if abs(sum_check - 1.0) > 0.01:
            print(f"  Warning: Mean proportions sum to {sum_check:.4f} (expected ~1.0)")

        return {
            'fusion_mode': f'intermediate_{self.fusion_mode}',
            'method': method,
            'num_samples': len(all_attributions),
            'zero_gradient_samples': zero_gradient_count,
            'id': {
                'mean': np.mean(id_attrs),
                'std': np.std(id_attrs),
                'min': np.min(id_attrs),
                'max': np.max(id_attrs),
                'values': id_attrs,
                'raw_mean': np.mean(np.abs(id_raw)),
                'raw_std': np.std(np.abs(id_raw))
            },
            'text': {
                'mean': np.mean(text_attrs),
                'std': np.std(text_attrs),
                'min': np.min(text_attrs),
                'max': np.max(text_attrs),
                'values': text_attrs,
                'raw_mean': np.mean(np.abs(text_raw)),
                'raw_std': np.std(np.abs(text_raw))
            },
            'image': {
                'mean': np.mean(image_attrs),
                'std': np.std(image_attrs),
                'min': np.min(image_attrs),
                'max': np.max(image_attrs),
                'values': image_attrs,
                'raw_mean': np.mean(np.abs(image_raw)),
                'raw_std': np.std(np.abs(image_raw))
            },
            'raw_attributions': all_attributions
        }

    def create_masked_sequences(self, sequences):
        """Create masked sequences for training."""
        batch_data = []

        for seq in sequences:
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]

            padded_seq = seq + [self.pad_token] * (self.max_seq_len - len(seq))

            masked_seq = padded_seq.copy()
            masked_positions = []
            masked_labels = []

            valid_positions = [i for i, item in enumerate(padded_seq) if item != self.pad_token]

            if valid_positions:
                num_to_mask = max(1, int(len(valid_positions) * self.mask_prob))
                mask_positions = random.sample(valid_positions, min(num_to_mask, len(valid_positions)))

                for pos in mask_positions:
                    masked_labels.append(padded_seq[pos] - 1)
                    masked_seq[pos] = self.mask_token
                    masked_positions.append(pos)

            max_masked = max(1, int(self.max_seq_len * self.mask_prob))
            while len(masked_positions) < max_masked:
                masked_positions.append(0)
                masked_labels.append(-1)

            batch_data.append({
                'input_ids': masked_seq,
                'masked_positions': masked_positions[:max_masked],
                'masked_labels': masked_labels[:max_masked]
            })

        input_ids = torch.tensor([item['input_ids'] for item in batch_data], dtype=torch.long)
        masked_positions = torch.tensor([item['masked_positions'] for item in batch_data], dtype=torch.long)
        masked_labels = torch.tensor([item['masked_labels'] for item in batch_data], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'masked_positions': masked_positions,
            'masked_labels': masked_labels
        }

    def predict(self, sequences, target_items):
        """Predict scores for target items given sequences."""
        self.eval()
        with torch.no_grad():
            hidden_states = self.forward(sequences)

            batch_size = sequences.size(0)
            last_positions = []

            for i in range(batch_size):
                valid_positions = (sequences[i] != self.pad_token).nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    last_positions.append(valid_positions[-1].item())
                else:
                    last_positions.append(0)

            batch_indices = torch.arange(batch_size, device=sequences.device)
            last_hidden = hidden_states[batch_indices, last_positions]

            # Use output layer to score all items, then gather for candidates
            # target_items are 1-indexed (0 is pad), output_layer is 0-indexed
            all_logits = self.output_layer(last_hidden)  # (B, item_num)
            scores = all_logits.gather(1, target_items - 1)

            return scores

    def get_fusion_weights(self):
        """Return current fusion mode info."""
        result = {
            'fusion_mode': self.fusion_mode,
            'num_pre_fusion_layers': self.num_pre_fusion_layers,
            'num_post_fusion_layers': self.num_post_fusion_layers,
        }
        return result


def analyze_attribution_results(results, save_path=None):
    """Analyze and optionally save attribution results."""
    if results is None:
        return None

    summary = {
        'fusion_mode': results['fusion_mode'],
        'method': results['method'],
        'num_samples': results['num_samples'],
        'modality_contributions': {
            'id': {
                'mean': results['id']['mean'],
                'std': results['id']['std']
            },
            'text': {
                'mean': results['text']['mean'],
                'std': results['text']['std']
            },
            'image': {
                'mean': results['image']['mean'],
                'std': results['image']['std']
            }
        },
        'dominant_modality': max(
            ['id', 'text', 'image'],
            key=lambda x: results[x]['mean']
        )
    }

    print("\n" + "=" * 60)
    print("GRADIENT-BASED MODALITY ATTRIBUTION ANALYSIS (INTERMEDIATE FUSION)")
    print("=" * 60)
    print(f"Fusion Mode: {results['fusion_mode']}")
    print(f"Method: {results['method']}")
    print(f"Number of samples: {results['num_samples']}")
    print("-" * 60)
    print("Modality Contributions (proportions):")
    print(f"  Item ID:  {results['id']['mean']:.4f} +/- {results['id']['std']:.4f}")
    print(f"  Text:     {results['text']['mean']:.4f} +/- {results['text']['std']:.4f}")
    print(f"  Image:    {results['image']['mean']:.4f} +/- {results['image']['std']:.4f}")
    print("-" * 60)
    print(f"Dominant modality: {summary['dominant_modality']}")
    print("=" * 60)

    if save_path:
        import json
        save_data = {
            'fusion_mode': results['fusion_mode'],
            'method': results['method'],
            'num_samples': results['num_samples'],
            'id_mean': float(results['id']['mean']),
            'id_std': float(results['id']['std']),
            'text_mean': float(results['text']['mean']),
            'text_std': float(results['text']['std']),
            'image_mean': float(results['image']['mean']),
            'image_std': float(results['image']['std']),
            'dominant_modality': summary['dominant_modality']
        }
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to: {save_path}")

    return summary
