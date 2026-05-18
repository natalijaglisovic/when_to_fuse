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


class BERT4RecEarlyFusion(nn.Module):
    """
    BERT4Rec with early fusion of item ID, text, and image embeddings.

    Fusion modes:
    - 'add': item_id_emb + text_emb_proj + image_emb_proj (simple additive fusion without weights)
    - 'concat': [item_id_emb; text_emb; image_emb] (concatenation fusion with original embeddings)
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
            fusion_mode: 'add' for additive fusion or 'concat' for concatenation fusion
        """
        super(BERT4RecEarlyFusion, self).__init__()

        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob
        self.fusion_mode = fusion_mode

        if fusion_mode not in ['add', 'concat']:
            raise ValueError(f"fusion_mode must be 'add' or 'concat', got '{fusion_mode}'")

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

        # Projection layers to map text and image to hidden_dim (only for additive mode)
        if fusion_mode == 'add':
            self.text_projection = nn.Linear(text_dim, hidden_dim)
            self.image_projection = nn.Linear(image_dim, hidden_dim)
        elif fusion_mode == 'concat':
                       # For concatenation mode, concatenate original embeddings: hidden_dim + text_dim + image_dim
            self.fusion_projection = nn.Linear(hidden_dim + text_dim + image_dim, hidden_dim)

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
        # Skip pretrained embedding layers to preserve loaded weights
        if module is self.text_embedding or module is self.image_embedding:
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_fused_embeddings(self, input_ids):
        """
        Compute fused embeddings based on fusion_mode:
        - 'add': item_id_emb + text_emb_proj + image_emb_proj
        - 'concat': [item_id_emb; text_emb; image_emb] -> projection to hidden_dim

        Args:
            input_ids: (batch_size, seq_len) - item indices

        Returns:
            fused_emb: (batch_size, seq_len, hidden_dim) - fused embeddings
        """
        # Get item ID embeddings
        item_id_emb = self.item_id_embedding(input_ids)

        # Get text and image embeddings
        text_emb = self.text_embedding(input_ids)
        image_emb = self.image_embedding(input_ids)

        if self.fusion_mode == 'concat':
            # Concatenate original embeddings: [item_id_emb; text_emb; image_emb]
            concat_emb = torch.cat([item_id_emb, text_emb, image_emb], dim=-1)
            # Project back to hidden_dim
            fused_emb = self.fusion_projection(concat_emb)
        else:
            # Additive fusion: project text and image to hidden_dim, then add
            text_emb_proj = self.text_projection(text_emb)
            image_emb_proj = self.image_projection(image_emb)
            fused_emb = item_id_emb + text_emb_proj + image_emb_proj

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

        # Attention mask (1 for valid tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token).long()

        # Get fused embeddings
        embeddings = self.get_fused_embeddings(input_ids)

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
            return logits
        else:
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

    def get_fusion_weights(self):
        """Return current fusion mode (no learnable weights in this implementation)"""
        return {
            'fusion_mode': self.fusion_mode,
            'alpha': None,
            'beta': None
        }
