"""
Concatenation-based fusion with SVD analysis.

Fusion: h = projection([e_id; e_text; e_image])

SVD Analysis:
- Uses norm-based importance: α_m = ||e_m|| / Σ||e_m||
- Expected: Moderate rank (depends on embedding magnitudes)
"""
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm

# Reuse components from attention model
from bert4rec.early_fusion.analysis.svd.model_early_fusion_attention_svd import (
    TransformerBlock, TextEmbeddingLoader, VisionEmbeddingLoader,
    effective_rank
)


class ModalityConcatFusion(nn.Module):
    """Concatenation-based fusion: h = projection([e_id; e_text; e_image])"""

    def __init__(self, hidden_dim, text_dim, image_dim, dropout_rate=0.1):
        super().__init__()
        concat_dim = hidden_dim + text_dim + image_dim
        self.projection = nn.Linear(concat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, item_id_emb, text_emb, image_emb, return_alpha=False):
        concat = torch.cat([item_id_emb, text_emb, image_emb], dim=-1)
        fused_emb = self.dropout(self.projection(concat))

        if return_alpha:
            # Norm-based importance
            text_norm = torch.norm(text_emb, dim=-1, keepdim=True)
            image_norm = torch.norm(image_emb, dim=-1, keepdim=True)
            norms = torch.cat([text_norm, image_norm], dim=-1)
            alpha = norms / (norms.sum(dim=-1, keepdim=True) + 1e-8)
            return fused_emb, alpha
        return fused_emb


class BERT4RecConcatFusionSVD(nn.Module):
    """BERT4Rec with concatenation fusion"""

    def __init__(self, item_num, text_embedding_matrix, image_embedding_matrix,
                 max_seq_len=50, hidden_dim=64, num_layers=2, num_heads=2,
                 dropout_rate=0.1, mask_prob=0.15,
                 freeze_text_embeddings=True, freeze_image_embeddings=True):
        super().__init__()

        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob
        self.pad_token = 0
        self.mask_token = item_num + 1
        self.vocab_size = item_num + 2

        text_dim = text_embedding_matrix.shape[1]
        image_dim = image_embedding_matrix.shape[1]

        self.item_id_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)
        self.text_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(text_embedding_matrix), freeze=freeze_text_embeddings, padding_idx=0)
        self.image_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(image_embedding_matrix), freeze=freeze_image_embeddings, padding_idx=0)

        self.modality_fusion = ModalityConcatFusion(hidden_dim, text_dim, image_dim, dropout_rate)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        ff_dim = hidden_dim * 4
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
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

    def get_fused_embeddings(self, input_ids, return_alpha=False):
        item_id_emb = self.item_id_embedding(input_ids)
        text_emb = self.text_embedding(input_ids)
        image_emb = self.image_embedding(input_ids)
        return self.modality_fusion(item_id_emb, text_emb, image_emb, return_alpha)

    def forward(self, input_ids, masked_positions=None, return_alpha=False):
        batch_size, seq_len = input_ids.size()
        attention_mask = (input_ids != self.pad_token).long()

        if return_alpha:
            embeddings, alpha = self.get_fused_embeddings(input_ids, return_alpha=True)
        else:
            embeddings = self.get_fused_embeddings(input_ids, return_alpha=False)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = embeddings + self.position_embedding(position_ids)
        embeddings = self.dropout(embeddings)

        hidden_states = embeddings
        for transformer in self.transformer_blocks:
            hidden_states = transformer(hidden_states, attention_mask)
        hidden_states = self.layer_norm(hidden_states)

        if masked_positions is not None:
            batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand(-1, masked_positions.size(1))
            batch_flat = torch.stack([batch_indices, masked_positions], dim=-1)[:, :, 0].flatten()
            pos_flat = torch.stack([batch_indices, masked_positions], dim=-1)[:, :, 1].flatten()
            masked_hidden = hidden_states[batch_flat, pos_flat].view(batch_size, masked_positions.size(1), self.hidden_dim)
            logits = self.output_layer(masked_hidden)
            return (logits, alpha) if return_alpha else logits
        else:
            return (hidden_states, alpha) if return_alpha else hidden_states

    def create_masked_sequences(self, sequences):
        """Same as attention model"""
        batch_data = []
        for seq in sequences:
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]
            padded_seq = seq + [self.pad_token] * (self.max_seq_len - len(seq))
            masked_seq, masked_positions, masked_labels = padded_seq.copy(), [], []
            valid_positions = [i for i, item in enumerate(padded_seq) if item != self.pad_token]

            if valid_positions:
                num_to_mask = max(1, int(len(valid_positions) * self.mask_prob))
                for pos in random.sample(valid_positions, min(num_to_mask, len(valid_positions))):
                    masked_labels.append(padded_seq[pos] - 1)
                    masked_seq[pos] = self.mask_token
                    masked_positions.append(pos)

            max_masked = max(1, int(self.max_seq_len * self.mask_prob))
            while len(masked_positions) < max_masked:
                masked_positions.append(0)
                masked_labels.append(-1)

            batch_data.append({'input_ids': masked_seq, 'masked_positions': masked_positions[:max_masked],
                             'masked_labels': masked_labels[:max_masked]})

        return {k: torch.tensor([item[k] for item in batch_data], dtype=torch.long)
                for k in ['input_ids', 'masked_positions', 'masked_labels']}

    def predict(self, sequences, target_items):
        """Same as attention model"""
        self.eval()
        with torch.no_grad():
            hidden_states = self.forward(sequences)
            batch_size = sequences.size(0)
            last_positions = [(sequences[i] != self.pad_token).nonzero(as_tuple=True)[0][-1].item()
                            if len((sequences[i] != self.pad_token).nonzero(as_tuple=True)[0]) > 0 else 0
                            for i in range(batch_size)]
            last_hidden = hidden_states[torch.arange(batch_size, device=sequences.device), last_positions]
            target_emb = self.get_fused_embeddings(target_items)
            return torch.bmm(last_hidden.unsqueeze(1), target_emb.transpose(1, 2)).squeeze(1)


def collect_fusion_attention(model, dataloader, device, max_batches=None):
    """Collect norm-based modality importance"""
    model.eval()
    all_alpha = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting modality importance")):
            if max_batches and batch_idx >= max_batches:
                break
            input_ids = batch['input_ids'].to(device)
            _, alpha = model.forward(input_ids, return_alpha=True)
            attention_mask = (input_ids != model.pad_token)
            for b in range(alpha.size(0)):
                all_alpha.append(alpha[b][attention_mask[b]])
    return torch.cat(all_alpha, dim=0)


def analyze_modality_fusion(model, dataloader, device, max_batches=None):
    """SVD analysis for concat fusion"""
    alpha_matrix = collect_fusion_attention(model, dataloader, device, max_batches)
    r_eff, singular_values = effective_rank(alpha_matrix)
    text_weights, image_weights = alpha_matrix[:, 0].cpu().numpy(), alpha_matrix[:, 1].cpu().numpy()

    return {
        'effective_rank': r_eff.item(),
        'singular_values': singular_values.cpu().numpy(),
        'mean_text_weight': text_weights.mean(),
        'mean_image_weight': image_weights.mean(),
        'std_text_weight': text_weights.std(),
        'std_image_weight': image_weights.std(),
        'num_items_analyzed': alpha_matrix.shape[0],
        'alpha_matrix': alpha_matrix.cpu().numpy()
    }
