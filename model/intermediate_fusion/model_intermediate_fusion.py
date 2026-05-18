import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import math
import os

from model.early_fusion.model_early_fusion import (
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    TextEmbeddingLoader,
    VisionEmbeddingLoader,
)


class ModalityEncoder(nn.Module):
    """
    Independent transformer encoder for a single modality (pre-fusion layers).
    Each modality gets its own transformer stack to learn modality-specific
    high-dimensional representations before fusion.
    """

    def __init__(self, input_dim, hidden_dim, max_seq_len, num_layers, num_heads, dropout_rate=0.1):
        super(ModalityEncoder, self).__init__()

        self.hidden_dim = hidden_dim

        # Project modality input to hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Position embeddings
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer blocks (pre-fusion)
        ff_dim = hidden_dim * 4
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, embeddings, attention_mask):
        """
        Args:
            embeddings: (batch_size, seq_len, input_dim) - modality embeddings
            attention_mask: (batch_size, seq_len) - 1 for valid, 0 for padding

        Returns:
            hidden_states: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = embeddings.size()

        # Project to hidden_dim
        x = self.input_projection(embeddings)

        # Add position embeddings
        position_ids = torch.arange(seq_len, device=embeddings.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(position_ids)
        x = self.dropout(x)

        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, attention_mask)

        x = self.layer_norm(x)
        return x


class BERT4RecIntermediateFusion(nn.Module):
    """
    BERT4Rec with intermediate fusion (feature-level fusion).

    Architecture:
    1. Each modality (item ID, text, image) gets independent transformer layers
       to build high-dimensional modality-specific representations.
    2. The representations are fused in a middle layer (element-wise sum or concatenation).
    3. Shared transformer layers process the fused representation for final prediction.

    This sits between early fusion (fuse before any processing) and late fusion
    (fuse after all processing), allowing each modality to first learn its own
    representations before combining them for joint reasoning.

    Fusion modes:
    - 'add': u = u_id + u_text + u_image (element-wise sum)
    - 'concat': u = [u_id || u_text || u_image] -> projection to hidden_dim
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
        """
        Args:
            item_num: Number of items
            text_embedding_matrix: Pre-computed text embeddings (num_items+2, text_dim)
            image_embedding_matrix: Pre-computed image embeddings (num_items+2, image_dim)
            max_seq_len: Maximum sequence length
            hidden_dim: Hidden dimension for transformer layers
            num_pre_fusion_layers: Number of modality-specific transformer layers (before fusion)
            num_post_fusion_layers: Number of shared transformer layers (after fusion)
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            mask_prob: Masking probability
            freeze_text_embeddings: Whether to freeze text embeddings
            freeze_image_embeddings: Whether to freeze image embeddings
            fusion_mode: 'add' for element-wise sum or 'concat' for concatenation
        """
        super(BERT4RecIntermediateFusion, self).__init__()

        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob
        self.fusion_mode = fusion_mode
        self.num_pre_fusion_layers = num_pre_fusion_layers
        self.num_post_fusion_layers = num_post_fusion_layers

        if fusion_mode not in ['add', 'concat']:
            raise ValueError(f"fusion_mode must be 'add' or 'concat', got '{fusion_mode}'")

        self.pad_token = 0
        self.mask_token = item_num + 1
        self.vocab_size = item_num + 2

        text_dim = text_embedding_matrix.shape[1]
        image_dim = image_embedding_matrix.shape[1]

        # --- Embedding layers ---
        # Item ID embeddings (learnable)
        self.item_id_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)

        # Text embeddings (pre-computed, optionally frozen)
        self.text_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(text_embedding_matrix),
            freeze=freeze_text_embeddings,
            padding_idx=0
        )

        # Image embeddings (pre-computed, optionally frozen)
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
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_layers=num_pre_fusion_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        self.image_encoder = ModalityEncoder(
            input_dim=image_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_layers=num_pre_fusion_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        # --- Stage 2: Fusion layer ---
        if fusion_mode == 'concat':
            # After concatenation: 3 * hidden_dim -> hidden_dim
            self.fusion_projection = nn.Linear(3 * hidden_dim, hidden_dim)

        # --- Stage 3: Shared transformer layers (post-fusion) ---
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
        """
        Intermediate fusion: combine modality representations after modality-specific encoding,
        before shared post-fusion layers.

        Args:
            id_hidden: (batch_size, seq_len, hidden_dim)
            text_hidden: (batch_size, seq_len, hidden_dim)
            image_hidden: (batch_size, seq_len, hidden_dim)

        Returns:
            fused: (batch_size, seq_len, hidden_dim)
        """
        if self.fusion_mode == 'add':
            # Element-wise sum: u = u_id + u_text + u_image
            fused = id_hidden + text_hidden + image_hidden
        elif self.fusion_mode == 'concat':
            # Concatenation: u = [u_id || u_text || u_image] -> projection
            concat = torch.cat([id_hidden, text_hidden, image_hidden], dim=-1)
            fused = self.fusion_projection(concat)

        return fused

    def forward(self, input_ids, masked_positions=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - item sequence with some items masked
            masked_positions: (batch_size, num_masked) - positions of masked items

        Returns:
            logits: (batch_size, num_masked, item_num) - predictions for masked positions
        """
        batch_size, seq_len = input_ids.size()

        # Attention mask (1 for valid tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token).long()

        # Get embeddings for each modality
        id_emb = self.item_id_embedding(input_ids)     # (B, S, hidden_dim)
        text_emb = self.text_embedding(input_ids)       # (B, S, text_dim)
        image_emb = self.image_embedding(input_ids)     # (B, S, image_dim)

        # Stage 1: Encode each modality independently (pre-fusion)
        id_hidden = self.id_encoder(id_emb, attention_mask)       # (B, S, hidden_dim)
        text_hidden = self.text_encoder(text_emb, attention_mask)  # (B, S, hidden_dim)
        image_hidden = self.image_encoder(image_emb, attention_mask)  # (B, S, hidden_dim)

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

    def create_masked_sequences(self, sequences):
        """
        Create masked sequences for training.

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
        Predict scores for target items given sequences.

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

            # For scoring, encode target items through modality encoders, fuse, then post-fusion
            target_id_emb = self.item_id_embedding(target_items)
            target_text_emb = self.text_embedding(target_items)
            target_image_emb = self.image_embedding(target_items)

            # Use projection layers for single-item targets (no transformer needed)
            target_id_proj = self.id_encoder.input_projection(target_id_emb)
            target_text_proj = self.text_encoder.input_projection(target_text_emb)
            target_image_proj = self.image_encoder.input_projection(target_image_emb)

            # Fuse target representations
            if self.fusion_mode == 'add':
                target_fused = target_id_proj + target_text_proj + target_image_proj
            elif self.fusion_mode == 'concat':
                target_concat = torch.cat([target_id_proj, target_text_proj, target_image_proj], dim=-1)
                target_fused = self.fusion_projection(target_concat)

            # Compute similarity scores
            scores = torch.bmm(
                last_hidden.unsqueeze(1),
                target_fused.transpose(1, 2)
            ).squeeze(1)

            return scores

    def get_fusion_weights(self):
        """Return current fusion mode info"""
        return {
            'fusion_mode': self.fusion_mode,
            'num_pre_fusion_layers': self.num_pre_fusion_layers,
            'num_post_fusion_layers': self.num_post_fusion_layers,
        }
