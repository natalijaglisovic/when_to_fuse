#!/usr/bin/env python3
"""
BERT4Rec with Early Fusion and Gradient-Based Modality Attribution

This module extends the standard BERT4RecEarlyFusion model to support
gradient-based attribution analysis for ALL fusion types:
- 'add': item_id_emb + text_emb_proj + image_emb_proj
- 'concat': [item_id_emb; text_emb; image_emb] -> projection
- 'attention': attention-weighted fusion of modalities

Attribution Methods:
1. Gradient magnitude: ||∂score/∂embedding||
2. Gradient × input (integrated gradients approximation): embedding * gradient
3. Layer-wise relevance propagation (optional)
"""
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
    Attention-based modality fusion module.

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

        # LayerNorm to normalize embeddings before projection
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
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, item_id_emb, text_emb, image_emb, return_alpha=False):
        """
        Args:
            item_id_emb: (batch_size, seq_len, hidden_dim)
            text_emb: (batch_size, seq_len, text_dim)
            image_emb: (batch_size, seq_len, image_dim)
            return_alpha: If True, return fusion attention weights

        Returns:
            fused_emb: (batch_size, seq_len, hidden_dim)
            alpha: (batch_size, seq_len, 2) - attention weights if return_alpha=True
        """
        # Normalize embeddings before projection
        item_id_emb_norm = self.ln_id(item_id_emb)
        text_emb_norm = self.ln_text(text_emb)
        image_emb_norm = self.ln_image(image_emb)

        # Compute query from item ID embedding
        q = self.W_q(item_id_emb_norm)

        # Compute keys from each modality
        k_text = self.W_k_text(text_emb_norm)
        k_image = self.W_k_image(image_emb_norm)

        # Compute values from each modality
        v_text = self.W_v_text(text_emb_norm)
        v_image = self.W_v_image(image_emb_norm)

        # Compute attention scores for each modality
        score_text = torch.sum(q * k_text, dim=-1, keepdim=True)
        score_image = torch.sum(q * k_image, dim=-1, keepdim=True)

        # Stack scores and apply softmax
        scores = torch.cat([score_text, score_image], dim=-1)

        effective_temp = F.softplus(self.temperature) + 0.1
        alpha = F.softmax(scores * effective_temp, dim=-1)
        alpha_no_dropout = alpha.clone()
        alpha = self.dropout(alpha)

        alpha_text = alpha[:, :, 0:1]
        alpha_image = alpha[:, :, 1:2]

        # Compute weighted sum of values
        weighted_text = alpha_text * v_text
        weighted_image = alpha_image * v_image

        # Fuse: h_t = e_t^id + Σ_m α_m * v_m
        fused_emb = item_id_emb + weighted_text + weighted_image

        if return_alpha:
            return fused_emb, alpha_no_dropout
        return fused_emb

    def forward_with_embeddings(self, item_id_emb, text_emb, image_emb):
        """
        Forward pass for gradient attribution - returns intermediate values.

        Returns:
            fused_emb: fused embeddings
            v_text: projected text values
            v_image: projected image values
            alpha_text: text attention weight
            alpha_image: image attention weight
        """
        # Normalize embeddings before projection
        item_id_emb_norm = self.ln_id(item_id_emb)
        text_emb_norm = self.ln_text(text_emb)
        image_emb_norm = self.ln_image(image_emb)

        # Compute query from item ID embedding
        q = self.W_q(item_id_emb_norm)

        # Compute keys from each modality
        k_text = self.W_k_text(text_emb_norm)
        k_image = self.W_k_image(image_emb_norm)

        # Compute values from each modality
        v_text = self.W_v_text(text_emb_norm)
        v_image = self.W_v_image(image_emb_norm)

        # Compute attention scores
        score_text = torch.sum(q * k_text, dim=-1, keepdim=True)
        score_image = torch.sum(q * k_image, dim=-1, keepdim=True)

        scores = torch.cat([score_text, score_image], dim=-1)

        effective_temp = F.softplus(self.temperature) + 0.1
        alpha = F.softmax(scores * effective_temp, dim=-1)

        alpha_text = alpha[:, :, 0:1]
        alpha_image = alpha[:, :, 1:2]

        # Compute weighted sum
        weighted_text = alpha_text * v_text
        weighted_image = alpha_image * v_image

        fused_emb = item_id_emb + weighted_text + weighted_image

        return fused_emb, v_text, v_image, alpha_text, alpha_image


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


class BERT4RecGradientAttribution(nn.Module):
    """
    BERT4Rec with early fusion supporting gradient-based modality attribution.

    This model supports computing gradient-based attribution to understand
    how much each modality (item_id, text, image) contributes to predictions.

    Fusion modes:
    - 'add': item_id_emb + text_emb_proj + image_emb_proj
    - 'concat': [item_id_emb; text_emb; image_emb] -> projection to hidden_dim
    - 'attention': attention-weighted fusion (query from ID, key/value from text/image)
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
                 freeze_image_embeddings=True,
                 fusion_mode='add'):
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
            fusion_mode: 'add', 'concat', or 'attention'
        """
        super(BERT4RecGradientAttribution, self).__init__()

        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob
        self.fusion_mode = fusion_mode

        if fusion_mode not in ['add', 'concat', 'attention']:
            raise ValueError(f"fusion_mode must be 'add', 'concat', or 'attention', got '{fusion_mode}'")

        self.pad_token = 0
        self.mask_token = item_num + 1
        self.vocab_size = item_num + 2

        self.text_dim = text_embedding_matrix.shape[1]
        self.image_dim = image_embedding_matrix.shape[1]

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

        # Fusion-specific layers
        if fusion_mode == 'add':
            self.text_projection = nn.Linear(self.text_dim, hidden_dim)
            self.image_projection = nn.Linear(self.image_dim, hidden_dim)
        elif fusion_mode == 'concat':
            # Add LayerNorm to normalize each modality before concatenation
            # This prevents embeddings with different scales from dominating
            self.ln_id_concat = nn.LayerNorm(hidden_dim)
            self.ln_text_concat = nn.LayerNorm(self.text_dim)
            self.ln_image_concat = nn.LayerNorm(self.image_dim)
            self.fusion_projection = nn.Linear(hidden_dim + self.text_dim + self.image_dim, hidden_dim)
        elif fusion_mode == 'attention':
            self.modality_attention = ModalityAttentionFusion(
                hidden_dim=hidden_dim,
                text_dim=self.text_dim,
                image_dim=self.image_dim,
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
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_id_embedding(self, input_ids):
        """Get item ID embeddings for given input IDs"""
        return self.item_id_embedding(input_ids)

    def get_text_embedding(self, input_ids):
        """Get text embeddings (raw, before projection) for given input IDs"""
        return self.text_embedding(input_ids)

    def get_image_embedding(self, input_ids):
        """Get image embeddings (raw, before projection) for given input IDs"""
        return self.image_embedding(input_ids)

    def get_text_embedding_projected(self, input_ids):
        """Get projected text embeddings for given input IDs (only for 'add' mode)"""
        if self.fusion_mode != 'add':
            raise ValueError("get_text_embedding_projected only available for 'add' fusion mode")
        text_emb = self.text_embedding(input_ids)
        return self.text_projection(text_emb)

    def get_image_embedding_projected(self, input_ids):
        """Get projected image embeddings for given input IDs (only for 'add' mode)"""
        if self.fusion_mode != 'add':
            raise ValueError("get_image_embedding_projected only available for 'add' fusion mode")
        image_emb = self.image_embedding(input_ids)
        return self.image_projection(image_emb)

    def forward_with_embeddings_add(self, id_emb, text_emb_proj, image_emb_proj, input_ids):
        """Forward pass for additive fusion with pre-computed embeddings."""
        batch_size, seq_len = input_ids.size()
        attention_mask = (input_ids != self.pad_token).long()

        # Fused embeddings via addition
        fused_emb = id_emb + text_emb_proj + image_emb_proj

        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)

        embeddings = fused_emb + position_emb
        embeddings = self.dropout(embeddings)

        hidden_states = embeddings
        for transformer in self.transformer_blocks:
            hidden_states = transformer(hidden_states, attention_mask)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def forward_with_embeddings_concat(self, id_emb, text_emb, image_emb, input_ids):
        """Forward pass for concatenation fusion with pre-computed embeddings."""
        batch_size, seq_len = input_ids.size()
        attention_mask = (input_ids != self.pad_token).long()

        # Normalize each modality before concatenation to prevent scale mismatch
        id_emb_norm = self.ln_id_concat(id_emb)
        text_emb_norm = self.ln_text_concat(text_emb)
        image_emb_norm = self.ln_image_concat(image_emb)

        # Concatenate and project
        concat_emb = torch.cat([id_emb_norm, text_emb_norm, image_emb_norm], dim=-1)
        fused_emb = self.fusion_projection(concat_emb)

        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)

        embeddings = fused_emb + position_emb
        embeddings = self.dropout(embeddings)

        hidden_states = embeddings
        for transformer in self.transformer_blocks:
            hidden_states = transformer(hidden_states, attention_mask)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def forward_with_embeddings_attention(self, id_emb, text_emb, image_emb, input_ids):
        """Forward pass for attention fusion with pre-computed embeddings."""
        batch_size, seq_len = input_ids.size()
        attention_mask = (input_ids != self.pad_token).long()

        # Attention-based fusion
        fused_emb = self.modality_attention(id_emb, text_emb, image_emb)

        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)

        embeddings = fused_emb + position_emb
        embeddings = self.dropout(embeddings)

        hidden_states = embeddings
        for transformer in self.transformer_blocks:
            hidden_states = transformer(hidden_states, attention_mask)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def get_fused_embeddings(self, input_ids):
        """
        Compute fused embeddings based on fusion_mode.

        Args:
            input_ids: (batch_size, seq_len) - item indices

        Returns:
            fused_emb: (batch_size, seq_len, hidden_dim) - fused embeddings
        """
        item_id_emb = self.item_id_embedding(input_ids)
        text_emb = self.text_embedding(input_ids)
        image_emb = self.image_embedding(input_ids)

        if self.fusion_mode == 'add':
            text_emb_proj = self.text_projection(text_emb)
            image_emb_proj = self.image_projection(image_emb)
            fused_emb = item_id_emb + text_emb_proj + image_emb_proj
        elif self.fusion_mode == 'concat':
            # Normalize before concatenation
            item_id_emb_norm = self.ln_id_concat(item_id_emb)
            text_emb_norm = self.ln_text_concat(text_emb)
            image_emb_norm = self.ln_image_concat(image_emb)
            concat_emb = torch.cat([item_id_emb_norm, text_emb_norm, image_emb_norm], dim=-1)
            fused_emb = self.fusion_projection(concat_emb)
        elif self.fusion_mode == 'attention':
            fused_emb = self.modality_attention(item_id_emb, text_emb, image_emb)

        return fused_emb

    def forward(self, input_ids, masked_positions=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - item sequence with some items masked
            masked_positions: (batch_size, num_masked) - positions of masked items

        Returns:
            logits: (batch_size, num_masked, item_num) - predictions for masked positions
        """
        batch_size, seq_len = input_ids.size()
        attention_mask = (input_ids != self.pad_token).long()

        embeddings = self.get_fused_embeddings(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)

        embeddings = embeddings + position_emb
        embeddings = self.dropout(embeddings)

        hidden_states = embeddings
        for transformer in self.transformer_blocks:
            hidden_states = transformer(hidden_states, attention_mask)

        hidden_states = self.layer_norm(hidden_states)

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

    def compute_modality_attribution(self, input_ids, target_item, method='gradient_norm'):
        """
        Compute gradient-based attribution for each modality.

        Works for ALL fusion modes: add, concat, attention.

        Args:
            input_ids: (batch_size, seq_len) - input sequences
            target_item: int - target item ID
            method: str - attribution method:
                - 'gradient_norm': ||∂score/∂embedding||
                - 'gradient_input': (embedding * gradient).sum()
                - 'integrated_gradients': approximation using baseline

        Returns:
            dict with 'id', 'text', 'image' attribution scores
        """
        self.eval()

        batch_size, seq_len = input_ids.size()

        # Get raw embeddings
        id_emb = self.item_id_embedding(input_ids)
        text_emb_raw = self.text_embedding(input_ids)
        image_emb_raw = self.image_embedding(input_ids)

        # Enable gradients on raw embeddings
        id_emb = id_emb.detach().requires_grad_(True)
        text_emb_raw = text_emb_raw.detach().requires_grad_(True)
        image_emb_raw = image_emb_raw.detach().requires_grad_(True)

        # Forward pass based on fusion mode
        if self.fusion_mode == 'add':
            text_emb_proj = self.text_projection(text_emb_raw)
            image_emb_proj = self.image_projection(image_emb_raw)
            hidden_states = self.forward_with_embeddings_add(id_emb, text_emb_proj, image_emb_proj, input_ids)
        elif self.fusion_mode == 'concat':
            hidden_states = self.forward_with_embeddings_concat(id_emb, text_emb_raw, image_emb_raw, input_ids)
        elif self.fusion_mode == 'attention':
            hidden_states = self.forward_with_embeddings_attention(id_emb, text_emb_raw, image_emb_raw, input_ids)

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

        # Project the final hidden state through the untied output layer to get
        # full logits over the item catalog (same projection used at train/inference time).
        logits = self.output_layer(last_hidden)  # (batch_size, item_num)

        # Select the logit for the true next item as the scalar to backpropagate from.
        # Apply the -1 offset used elsewhere (label construction, candidate scoring):
        # output_layer is 0-indexed over item_num, while target_item lives in the
        # ID-embedding vocab space (1-indexed, with pad/mask offsets).
        score = logits[:, target_item - 1].sum()

        # Backpropagate
        score.backward()

        # Check for None gradients (can happen if embeddings are frozen or disconnected)
        id_grad = id_emb.grad if id_emb.grad is not None else torch.zeros_like(id_emb)
        text_grad = text_emb_raw.grad if text_emb_raw.grad is not None else torch.zeros_like(text_emb_raw)
        image_grad = image_emb_raw.grad if image_emb_raw.grad is not None else torch.zeros_like(image_emb_raw)

        if method == 'gradient_norm':
            # Use double precision for numerical stability
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
        # If all attributes are essentially zero, return equal attribution
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

        # Sanity check: total should not be extremely small
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
            verbose: if True, print diagnostic info about gradient magnitudes

        Returns:
            dict with aggregated attribution statistics
        """
        self.eval()

        all_attributions = []
        zero_gradient_count = 0
        samples = test_data[:num_samples] if num_samples else test_data

        for user_seq, target_item in tqdm(samples, desc=f"Computing attributions ({self.fusion_mode})"):
            if len(user_seq) > self.max_seq_len:
                user_seq = user_seq[-self.max_seq_len:]

            padded_seq = user_seq + [0] * (self.max_seq_len - len(user_seq))
            input_ids = torch.tensor([padded_seq], dtype=torch.long, device=device)

            try:
                attr = self.compute_modality_attribution(input_ids, target_item, method=method)
                all_attributions.append(attr)

                # Track samples with near-zero gradients
                if attr.get('warning') == 'all_gradients_near_zero':
                    zero_gradient_count += 1

            except Exception as e:
                print(f"Warning: Attribution computation failed: {e}")
                continue

        if not all_attributions:
            return None

        # Report gradient health
        if zero_gradient_count > 0:
            print(f"  Warning: {zero_gradient_count}/{len(all_attributions)} samples had near-zero gradients for all modalities")

        # Aggregate statistics
        id_attrs = [a['id_proportion'] for a in all_attributions]
        text_attrs = [a['text_proportion'] for a in all_attributions]
        image_attrs = [a['image_proportion'] for a in all_attributions]

        # Collect raw gradient magnitudes for diagnostics
        id_raw = [a['id'] for a in all_attributions]
        text_raw = [a['text'] for a in all_attributions]
        image_raw = [a['image'] for a in all_attributions]

        if verbose:
            print(f"  Raw gradient magnitudes:")
            print(f"    ID:    mean={np.mean(np.abs(id_raw)):.2e}, min={np.min(np.abs(id_raw)):.2e}, max={np.max(np.abs(id_raw)):.2e}")
            print(f"    Text:  mean={np.mean(np.abs(text_raw)):.2e}, min={np.min(np.abs(text_raw)):.2e}, max={np.max(np.abs(text_raw)):.2e}")
            print(f"    Image: mean={np.mean(np.abs(image_raw)):.2e}, min={np.min(np.abs(image_raw)):.2e}, max={np.max(np.abs(image_raw)):.2e}")

        # Verify proportions sum to ~1.0 (sanity check)
        sum_check = np.mean(id_attrs) + np.mean(text_attrs) + np.mean(image_attrs)
        if abs(sum_check - 1.0) > 0.01:
            print(f"  Warning: Mean proportions sum to {sum_check:.4f} (expected ~1.0)")

        return {
            'fusion_mode': self.fusion_mode,
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

            target_emb = self.get_fused_embeddings(target_items)

            scores = torch.bmm(
                last_hidden.unsqueeze(1),
                target_emb.transpose(1, 2)
            ).squeeze(1)

            return scores

    def get_fusion_weights(self):
        """Return current fusion mode info."""
        result = {
            'fusion_mode': self.fusion_mode,
            'alpha': None,
            'beta': None
        }
        if self.fusion_mode == 'attention':
            result['temperature'] = self.modality_attention.temperature.item()
        return result


def analyze_attribution_results(results, save_path=None):
    """
    Analyze and optionally save attribution results.

    Args:
        results: dict from compute_attribution_batch
        save_path: optional path to save analysis

    Returns:
        analysis_summary: dict with analysis summary
    """
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
    print("GRADIENT-BASED MODALITY ATTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Fusion Mode: {results['fusion_mode']}")
    print(f"Method: {results['method']}")
    print(f"Number of samples: {results['num_samples']}")
    print("-" * 60)
    print("Modality Contributions (proportions):")
    print(f"  Item ID:  {results['id']['mean']:.4f} ± {results['id']['std']:.4f}")
    print(f"  Text:     {results['text']['mean']:.4f} ± {results['text']['std']:.4f}")
    print(f"  Image:    {results['image']['mean']:.4f} ± {results['image']['std']:.4f}")
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
