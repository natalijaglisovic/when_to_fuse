import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

from model.early_fusion.model_early_fusion import (
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
)
from model.late_fusion.model_late_fusion import ModalityEncoder


class LateFusionAttention(nn.Module):
    """
    Attention-based late fusion module.

    After each modality has been independently encoded by its own transformer,
    this module computes attention weights to combine them:

        q = W_q * u_id
        k_m = W_k_m * u_m,  m ∈ {text, image}
        v_m = W_v_m * u_m
        α_m = softmax(q^T k_m / sqrt(d))
        u = u_id + Σ_m α_m * v_m

    All inputs are already in hidden_dim space (post-encoder).
    """

    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(LateFusionAttention, self).__init__()

        self.hidden_dim = hidden_dim

        # Query projection from item ID representation
        self.W_q = nn.Linear(hidden_dim, hidden_dim)

        # Key projections from each modality
        self.W_k_text = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_image = nn.Linear(hidden_dim, hidden_dim)

        # Value projections from each modality
        self.W_v_text = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_image = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, id_hidden, text_hidden, image_hidden):
        """
        Args:
            id_hidden: (batch_size, seq_len, hidden_dim) - encoded item ID representations
            text_hidden: (batch_size, seq_len, hidden_dim) - encoded text representations
            image_hidden: (batch_size, seq_len, hidden_dim) - encoded image representations

        Returns:
            fused: (batch_size, seq_len, hidden_dim) - attention-fused representation
        """
        # Compute query from item ID representation
        q = self.W_q(id_hidden)  # (batch_size, seq_len, hidden_dim)

        # Compute keys from each modality
        k_text = self.W_k_text(text_hidden)    # (batch_size, seq_len, hidden_dim)
        k_image = self.W_k_image(image_hidden)  # (batch_size, seq_len, hidden_dim)

        # Compute values from each modality
        v_text = self.W_v_text(text_hidden)    # (batch_size, seq_len, hidden_dim)
        v_image = self.W_v_image(image_hidden)  # (batch_size, seq_len, hidden_dim)

        # Compute attention scores: score_m = q^T k_m / sqrt(d)
        score_text = torch.sum(q * k_text, dim=-1, keepdim=True) / self.scale    # (B, S, 1)
        score_image = torch.sum(q * k_image, dim=-1, keepdim=True) / self.scale  # (B, S, 1)

        # Softmax over modalities
        scores = torch.cat([score_text, score_image], dim=-1)  # (B, S, 2)
        alpha = F.softmax(scores, dim=-1)  # (B, S, 2)
        alpha = self.dropout(alpha)

        alpha_text = alpha[:, :, 0:1]   # (B, S, 1)
        alpha_image = alpha[:, :, 1:2]  # (B, S, 1)

        # Weighted sum of values
        weighted_text = alpha_text * v_text      # (B, S, hidden_dim)
        weighted_image = alpha_image * v_image   # (B, S, hidden_dim)

        # Fuse: u = u_id + Σ_m α_m * v_m
        fused = id_hidden + weighted_text + weighted_image

        return fused


class BERT4RecLateFusionAttention(nn.Module):
    """
    BERT4Rec with attention-based late fusion (decision-level fusion).

    Each modality (item ID, text, image) has its own independent transformer encoder.
    After encoding, representations are combined using attention-based fusion:
        u = u_id + Σ_m α_m * v_m
    where α_m = softmax(q^T k_m / sqrt(d)), with q derived from item ID representation.
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
        super(BERT4RecLateFusionAttention, self).__init__()

        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob
        self.fusion_mode = 'attention'

        self.pad_token = 0
        self.mask_token = item_num + 1
        self.vocab_size = item_num + 2

        text_dim = text_embedding_matrix.shape[1]
        image_dim = image_embedding_matrix.shape[1]

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

        # --- Independent transformer encoders per modality ---
        self.id_encoder = ModalityEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        self.text_encoder = ModalityEncoder(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        self.image_encoder = ModalityEncoder(
            input_dim=image_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        # --- Attention-based late fusion ---
        self.late_fusion_attention = LateFusionAttention(
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

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

    def forward(self, input_ids, masked_positions=None):
        batch_size, seq_len = input_ids.size()

        attention_mask = (input_ids != self.pad_token).long()

        # Get embeddings for each modality
        id_emb = self.item_id_embedding(input_ids)
        text_emb = self.text_embedding(input_ids)
        image_emb = self.image_embedding(input_ids)

        # Encode each modality independently
        id_hidden = self.id_encoder(id_emb, attention_mask)
        text_hidden = self.text_encoder(text_emb, attention_mask)
        image_hidden = self.image_encoder(image_emb, attention_mask)

        # Attention-based late fusion
        hidden_states = self.late_fusion_attention(id_hidden, text_hidden, image_hidden)

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

            # For target items, project through encoder input projections and fuse with attention
            target_id_emb = self.item_id_embedding(target_items)
            target_text_emb = self.text_embedding(target_items)
            target_image_emb = self.image_embedding(target_items)

            target_id_proj = self.id_encoder.input_projection(target_id_emb)
            target_text_proj = self.text_encoder.input_projection(target_text_emb)
            target_image_proj = self.image_encoder.input_projection(target_image_emb)

            # Use the attention fusion for target items too
            target_fused = self.late_fusion_attention(target_id_proj, target_text_proj, target_image_proj)

            scores = torch.bmm(
                last_hidden.unsqueeze(1),
                target_fused.transpose(1, 2)
            ).squeeze(1)

            return scores

    def get_attention_weights(self, input_ids):
        """
        Return attention weights for each modality.
        Useful for analysis and visualization.

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            dict with 'text_weights' and 'image_weights' tensors of shape (batch_size, seq_len)
        """
        self.eval()
        with torch.no_grad():
            attention_mask = (input_ids != self.pad_token).long()

            id_emb = self.item_id_embedding(input_ids)
            text_emb = self.text_embedding(input_ids)
            image_emb = self.image_embedding(input_ids)

            id_hidden = self.id_encoder(id_emb, attention_mask)
            text_hidden = self.text_encoder(text_emb, attention_mask)
            image_hidden = self.image_encoder(image_emb, attention_mask)

            # Compute attention scores
            q = self.late_fusion_attention.W_q(id_hidden)
            k_text = self.late_fusion_attention.W_k_text(text_hidden)
            k_image = self.late_fusion_attention.W_k_image(image_hidden)

            score_text = torch.sum(q * k_text, dim=-1, keepdim=True) / self.late_fusion_attention.scale
            score_image = torch.sum(q * k_image, dim=-1, keepdim=True) / self.late_fusion_attention.scale

            scores = torch.cat([score_text, score_image], dim=-1)
            alpha = F.softmax(scores, dim=-1)

            return {
                'text_weights': alpha[:, :, 0],
                'image_weights': alpha[:, :, 1]
            }

    def get_fusion_weights(self):
        """Return current fusion mode info"""
        return {
            'fusion_mode': self.fusion_mode,
        }
