import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from tqdm import tqdm
import math
import os


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module for BERT4Rec"""

    def __init__(self, hidden_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )

        output = self.output_linear(attention_output)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, hidden_dim, ff_dim, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward"""

    def __init__(self, hidden_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.feed_forward = FeedForward(hidden_dim, ff_dim, dropout_rate)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class ModalityAttentionFusion(nn.Module):
    """
    Attention-based modality fusion module with SVD analysis support.

    For each item at position t:
        q_t = W_q * e_t^id
        k_m = W_k * e_t^m, m ∈ {text, image}
        v_m = W_v * e_t^m
        α_m = softmax(q_t^T k_m / sqrt(d))
        h_t = e_t^id + Σ_m α_m * v_m
    """

    def __init__(self, hidden_dim, text_dim, image_dim, dropout_rate=0.1):
        super(ModalityAttentionFusion, self).__init__()

        self.hidden_dim = hidden_dim

        # LayerNorm to normalize embeddings before projection (critical for stable attention)
        self.ln_id = nn.LayerNorm(hidden_dim)
        self.ln_text = nn.LayerNorm(text_dim)
        self.ln_image = nn.LayerNorm(image_dim)

        # Query projection from item ID embedding
        self.W_q = nn.Linear(hidden_dim, hidden_dim)

        # Key projections from each modality
        self.W_k_text = nn.Linear(text_dim, hidden_dim)
        self.W_k_image = nn.Linear(image_dim, hidden_dim)

        # Value projections from each modality
        self.W_v_text = nn.Linear(text_dim, hidden_dim)
        self.W_v_image = nn.Linear(image_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        # Use smaller scale for modality attention (not multi-head, so sqrt(d) is too aggressive)
        # Learnable temperature allows model to tune score sharpness
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, item_id_emb, text_emb, image_emb, return_alpha=False):
        """
        Args:
            item_id_emb: (batch_size, seq_len, hidden_dim)
            text_emb: (batch_size, seq_len, text_dim)
            image_emb: (batch_size, seq_len, image_dim)
            return_alpha: If True, return fusion attention weights for SVD analysis

        Returns:
            fused_emb: (batch_size, seq_len, hidden_dim)
            alpha: (batch_size, seq_len, 2) - attention weights if return_alpha=True
        """
        # Normalize embeddings before projection (ensures stable attention scores)
        item_id_emb_norm = self.ln_id(item_id_emb)
        text_emb_norm = self.ln_text(text_emb)
        image_emb_norm = self.ln_image(image_emb)

        # Compute query from item ID embedding
        q = self.W_q(item_id_emb_norm)  # (batch_size, seq_len, hidden_dim)

        # Compute keys from each modality
        k_text = self.W_k_text(text_emb_norm)  # (batch_size, seq_len, hidden_dim)
        k_image = self.W_k_image(image_emb_norm)  # (batch_size, seq_len, hidden_dim)

        # Compute values from each modality
        v_text = self.W_v_text(text_emb_norm)  # (batch_size, seq_len, hidden_dim)
        v_image = self.W_v_image(image_emb_norm)  # (batch_size, seq_len, hidden_dim)

        # Compute attention scores for each modality
        # score_m = q^T k_m (scaled by learnable temperature)
        score_text = torch.sum(q * k_text, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        score_image = torch.sum(q * k_image, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)

        # Stack scores and apply softmax to get normalized attention weights
        scores = torch.cat([score_text, score_image], dim=-1)  # (batch_size, seq_len, 2)

        # DEBUG: Check if scores are too small (causes softmax to output ~0.5)
        if not hasattr(self, '_debug_printed') or not self._debug_printed:
            score_diff = (score_text - score_image).abs()
            print(f"\n[DEBUG] Attention score diagnostics:")
            print(f"  score_text  - mean: {score_text.mean().item():.6f}, std: {score_text.std().item():.6f}")
            print(f"  score_image - mean: {score_image.mean().item():.6f}, std: {score_image.std().item():.6f}")
            print(f"  |score_text - score_image| - mean: {score_diff.mean().item():.6f}, std: {score_diff.std().item():.6f}")
            print(f"  If score difference is near 0, softmax → [0.5, 0.5] for all items\n")
            self._debug_printed = True

        # Use softplus to ensure temperature stays positive (prevents collapse to 0)
        # Add minimum of 0.1 to prevent near-zero temperature
        effective_temp = F.softplus(self.temperature) + 0.1
        alpha = F.softmax(scores * effective_temp, dim=-1)  # (batch_size, seq_len, 2)
        alpha_no_dropout = alpha.clone()  # Save before dropout for SVD analysis
        alpha = self.dropout(alpha)

        alpha_text = alpha[:, :, 0:1]  # (batch_size, seq_len, 1)
        alpha_image = alpha[:, :, 1:2]  # (batch_size, seq_len, 1)

        # Compute weighted sum of values
        weighted_text = alpha_text * v_text  # (batch_size, seq_len, hidden_dim)
        weighted_image = alpha_image * v_image  # (batch_size, seq_len, hidden_dim)

        # Fuse: h_t = e_t^id + Σ_m α_m * v_m
        fused_emb = item_id_emb + weighted_text + weighted_image

        if return_alpha:
            return fused_emb, alpha_no_dropout
        return fused_emb


class TextEmbeddingLoader:
    """Utility class to load pre-computed text embeddings"""

    def __init__(self, embeddings_dir, embedding_key='sentence_transformer'):
        self.embeddings_dir = embeddings_dir
        self.embedding_key = embedding_key
        self.cache = {}

    def load_embedding(self, item_id):
        if item_id in self.cache:
            return self.cache[item_id]

        npz_path = os.path.join(self.embeddings_dir, f"{item_id}.npz")

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Embedding file not found: {npz_path}")

        data = np.load(npz_path)
        embedding = data[self.embedding_key]
        self.cache[item_id] = embedding

        return embedding

    def build_embedding_matrix(self, item_to_id, id_to_item):
        num_items = len(item_to_id)

        sample_item = list(id_to_item.values())[0] if id_to_item else None
        if sample_item is None:
            raise ValueError("No items found in id_to_item mapping")

        sample_emb = self.load_embedding(sample_item)
        embedding_dim = sample_emb.shape[0]

        embedding_matrix = np.zeros((num_items + 2, embedding_dim), dtype=np.float32)

        print(f"Building text embedding matrix for {num_items} items...")
        missing_items = []

        for idx in tqdm(range(1, num_items + 1), desc="Loading text embeddings"):
            if idx in id_to_item:
                original_item_id = id_to_item[idx]
                try:
                    embedding_matrix[idx] = self.load_embedding(original_item_id)
                except FileNotFoundError:
                    missing_items.append(original_item_id)
                    embedding_matrix[idx] = np.random.randn(embedding_dim).astype(np.float32) * 0.02

        embedding_matrix[num_items + 1] = np.random.randn(embedding_dim).astype(np.float32) * 0.02

        if missing_items:
            print(f"Warning: {len(missing_items)} items had missing text embeddings (using random init)")

        return embedding_matrix


class VisionEmbeddingLoader:
    """Utility class to load pre-computed CLIP vision embeddings"""

    def __init__(self, embeddings_dir, embedding_key='clip'):
        self.embeddings_dir = embeddings_dir
        self.embedding_key = embedding_key
        self.cache = {}

    def load_embedding(self, item_id):
        if item_id in self.cache:
            return self.cache[item_id]

        npz_path = os.path.join(self.embeddings_dir, f"{item_id}.npz")

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Embedding file not found: {npz_path}")

        data = np.load(npz_path)
        embedding = data[self.embedding_key]
        self.cache[item_id] = embedding

        return embedding

    def build_embedding_matrix(self, item_to_id, id_to_item):
        num_items = len(item_to_id)

        sample_item = list(id_to_item.values())[0] if id_to_item else None
        if sample_item is None:
            raise ValueError("No items found in id_to_item mapping")

        sample_emb = self.load_embedding(sample_item)
        embedding_dim = sample_emb.shape[0]

        embedding_matrix = np.zeros((num_items + 2, embedding_dim), dtype=np.float32)

        print(f"Building image embedding matrix for {num_items} items...")
        missing_items = []

        for idx in tqdm(range(1, num_items + 1), desc="Loading image embeddings"):
            if idx in id_to_item:
                original_item_id = id_to_item[idx]
                try:
                    embedding_matrix[idx] = self.load_embedding(original_item_id)
                except FileNotFoundError:
                    missing_items.append(original_item_id)
                    embedding_matrix[idx] = np.random.randn(embedding_dim).astype(np.float32) * 0.02

        embedding_matrix[num_items + 1] = np.random.randn(embedding_dim).astype(np.float32) * 0.02

        if missing_items:
            print(f"Warning: {len(missing_items)} items had missing image embeddings (using random init)")

        return embedding_matrix


class BERT4RecEarlyFusionAttentionSVD(nn.Module):
    """
    BERT4Rec with attention-based early fusion and SVD-based modality interaction analysis.

    Uses canonical attention mechanism:
        q_t = W_q * e_t^id
        k_m = W_k * e_t^m, m ∈ {text, image}
        v_m = W_v * e_t^m
        α_m = softmax(q_t^T k_m / sqrt(d))
        h_t = e_t^id + Σ_m α_m * v_m

    SVD Analysis:
    - Collects modality attention matrix A ∈ R^(N×2) across all items
    - Performs SVD: A = U Σ V^T
    - Computes effective rank to quantify multimodal fusion diversity
    - Higher effective rank indicates adaptive, balanced modality usage
    - Lower rank indicates one modality dominates (rank collapse)
    """

    def __init__(self,
                 item_num,
                 text_embedding_matrix,
                 image_embedding_matrix,
                 max_seq_len=50,
                 hidden_dim=64,
                 num_layers=2,
                 num_heads=2,
                 dropout_rate=0.1,
                 mask_prob=0.15,
                 freeze_text_embeddings=True,
                 freeze_image_embeddings=True):
        """
        Args:
            item_num: Number of items
            text_embedding_matrix: Pre-computed text embeddings (num_items+2, text_dim)
            image_embedding_matrix: Pre-computed image embeddings (num_items+2, image_dim)
            max_seq_len: Maximum sequence length
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            mask_prob: Masking probability
            freeze_text_embeddings: Whether to freeze text embeddings
            freeze_image_embeddings: Whether to freeze image embeddings
        """
        super(BERT4RecEarlyFusionAttentionSVD, self).__init__()

        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob

        self.pad_token = 0
        self.mask_token = item_num + 1
        self.vocab_size = item_num + 2

        text_dim = text_embedding_matrix.shape[1]
        image_dim = image_embedding_matrix.shape[1]

        # Item ID embeddings (learnable)
        self.item_id_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)

        # Text embeddings (optional freeze)
        self.text_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(text_embedding_matrix),
            freeze=freeze_text_embeddings,
            padding_idx=0
        )

        # Image embeddings (optional freeze)
        self.image_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(image_embedding_matrix),
            freeze=freeze_image_embeddings,
            padding_idx=0
        )

        # Attention-based fusion module
        self.modality_fusion = ModalityAttentionFusion(
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            image_dim=image_dim,
            dropout_rate=dropout_rate
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer blocks
        ff_dim = hidden_dim * 4
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, item_num)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Skip pre-trained embeddings — they must not be overwritten
        if module is self.text_embedding or module is self.image_embedding:
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_fused_embeddings(self, input_ids, return_alpha=False):
        """
        Compute attention-based fused embeddings.

        Args:
            input_ids: (batch_size, seq_len) - item indices
            return_alpha: If True, return fusion attention weights

        Returns:
            fused_emb: (batch_size, seq_len, hidden_dim) - fused embeddings
            alpha: (batch_size, seq_len, 2) - attention weights if return_alpha=True
        """
        # Get item ID embeddings
        item_id_emb = self.item_id_embedding(input_ids)

        # Get text and image embeddings
        text_emb = self.text_embedding(input_ids)
        image_emb = self.image_embedding(input_ids)

        # Apply attention-based fusion
        if return_alpha:
            fused_emb, alpha = self.modality_fusion(item_id_emb, text_emb, image_emb, return_alpha=True)
            return fused_emb, alpha
        else:
            fused_emb = self.modality_fusion(item_id_emb, text_emb, image_emb, return_alpha=False)
            return fused_emb

    def forward(self, input_ids, masked_positions=None, return_alpha=False):
        """
        Args:
            input_ids: (batch_size, seq_len) - item sequence with some items masked
            masked_positions: (batch_size, num_masked) - positions of masked items
            return_alpha: If True, return fusion attention weights for SVD analysis

        Returns:
            logits: (batch_size, num_masked, item_num) - predictions for masked positions
            alpha: (batch_size, seq_len, 2) - fusion attention weights if return_alpha=True
        """
        batch_size, seq_len = input_ids.size()

        # Attention mask (1 for valid tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token).long()

        # Get fused embeddings
        if return_alpha:
            embeddings, alpha = self.get_fused_embeddings(input_ids, return_alpha=True)
        else:
            embeddings = self.get_fused_embeddings(input_ids, return_alpha=False)

        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)

        # Add position embeddings
        embeddings = embeddings + position_emb
        embeddings = self.dropout(embeddings)

        # Pass through transformer blocks
        hidden_states = embeddings
        for transformer in self.transformer_blocks:
            hidden_states = transformer(hidden_states, attention_mask)

        hidden_states = self.layer_norm(hidden_states)

        if masked_positions is not None:
            # Extract hidden states at masked positions
            batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
            batch_indices = batch_indices.expand(-1, masked_positions.size(1))

            gather_indices = torch.stack([batch_indices, masked_positions], dim=-1)

            batch_flat = gather_indices[:, :, 0].flatten()
            pos_flat = gather_indices[:, :, 1].flatten()

            masked_hidden = hidden_states[batch_flat, pos_flat]
            masked_hidden = masked_hidden.view(batch_size, masked_positions.size(1), self.hidden_dim)

            # Predict items
            logits = self.output_layer(masked_hidden)

            if return_alpha:
                return logits, alpha
            return logits
        else:
            if return_alpha:
                return hidden_states, alpha
            return hidden_states

    def create_masked_sequences(self, sequences):
        """
        Create masked sequences for training

        Args:
            sequences: list of item sequences (list of lists)

        Returns:
            dict with 'input_ids', 'masked_positions', 'masked_labels'
        """
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
        """
        Predict scores for target items given sequences

        Args:
            sequences: (batch_size, seq_len) - input sequences
            target_items: (batch_size, num_candidates) - candidate items to score

        Returns:
            scores: (batch_size, num_candidates) - prediction scores
        """
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

            # Get fused embeddings for target items
            target_emb = self.get_fused_embeddings(target_items)

            # Compute similarity scores
            scores = torch.bmm(
                last_hidden.unsqueeze(1),
                target_emb.transpose(1, 2)
            ).squeeze(1)

            return scores


def collect_fusion_attention(model, dataloader, device, max_batches=None):
    """
    Collect fusion-level attention weights across the dataset.

    This function extracts the modality attention matrix A ∈ R^(N×2) where:
    - Each row represents an item's attention distribution over text and image modalities
    - α_t = [α_text, α_image] where α_text + α_image = 1

    Args:
        model: BERT4RecEarlyFusionAttentionSVD model
        dataloader: DataLoader for sequences
        device: torch device
        max_batches: Maximum number of batches to process (None = all)

    Returns:
        torch.Tensor: Attention matrix of shape (N, 2) where N is total items processed
    """
    model.eval()
    all_alpha = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Collecting fusion attention weights")
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)

            # Get fusion attention weights
            _, alpha = model.forward(input_ids, return_alpha=True)  # [B, T, 2]

            # Filter out padding AND mask token positions
            valid_mask = (input_ids != model.pad_token) & (input_ids != model.mask_token)  # [B, T]

            # Flatten and filter
            for b in range(alpha.size(0)):
                valid_positions = valid_mask[b]
                valid_alpha = alpha[b][valid_positions]  # [num_valid, 2]
                if valid_alpha.size(0) > 0:
                    all_alpha.append(valid_alpha)

    # Concatenate all attention weights
    return torch.cat(all_alpha, dim=0)  # [N, 2]


def effective_rank(matrix, eps=1e-12):
    """
    Compute effective rank of modality attention matrix using SVD.

    Effective rank measures the diversity of modality usage:
    - High effective rank (close to M=2): Balanced, adaptive multimodal fusion
    - Low effective rank (close to 1): One modality dominates (rank collapse)

    Formula: exp(H(p)) where H is entropy of normalized singular values

    Args:
        matrix: Attention matrix of shape (N, M) where M=2 (text, image)
        eps: Small constant for numerical stability

    Returns:
        r_eff: Effective rank (scalar)
        singular_values: Singular values from SVD
    """
    # Perform SVD
    U, S, V = torch.linalg.svd(matrix, full_matrices=False)

    # Normalize singular values to get probability distribution
    p = S / (S.sum() + eps)

    # Compute entropy
    entropy = -(p * torch.log(p + eps)).sum()

    # Effective rank is exp(entropy)
    r_eff = torch.exp(entropy)

    return r_eff, S


def analyze_modality_fusion(model, dataloader, device, max_batches=None):
    """
    Analyze modality fusion using SVD on attention weights.

    Returns:
        dict: Analysis results containing:
            - effective_rank: Effective rank of attention matrix
            - singular_values: Singular values from SVD
            - mean_text_weight: Average attention weight for text modality
            - mean_image_weight: Average attention weight for image modality
            - std_text_weight: Standard deviation of text attention
            - std_image_weight: Standard deviation of image attention
            - alpha_matrix: Full attention matrix for detailed visualization
    """
    # Collect attention weights
    alpha_matrix = collect_fusion_attention(model, dataloader, device, max_batches)

    # Compute effective rank
    r_eff, singular_values = effective_rank(alpha_matrix)

    # Compute statistics
    text_weights = alpha_matrix[:, 0].cpu().numpy()
    image_weights = alpha_matrix[:, 1].cpu().numpy()

    results = {
        'effective_rank': r_eff.item(),
        'singular_values': singular_values.cpu().numpy(),
        'mean_text_weight': text_weights.mean(),
        'mean_image_weight': image_weights.mean(),
        'std_text_weight': text_weights.std(),
        'std_image_weight': image_weights.std(),
        'num_items_analyzed': alpha_matrix.shape[0],
        'alpha_matrix': alpha_matrix.cpu().numpy()  # Include full matrix for visualization
    }

    return results
