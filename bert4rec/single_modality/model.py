import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from tqdm import tqdm
import math


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
        
        #linear transformations
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


class BERT4Rec(nn.Module):
    """BERT4Rec model for sequential recommendation"""
    
    def __init__(self, 
                 item_num,
                 max_seq_len=50,
                 hidden_dim=64,
                 num_layers=2,
                 num_heads=2,
                 dropout_rate=0.1,
                 mask_prob=0.15):
        super(BERT4Rec, self).__init__()
        
        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob
        
        self.pad_token = 0
        self.mask_token = item_num + 1
        self.vocab_size = item_num + 2 
        
        self.item_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)
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
    
    def forward(self, input_ids, masked_positions=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - item sequence with some items masked
            masked_positions: (batch_size, num_masked) - positions of masked items
            
        Returns:
            logits: (batch_size, num_masked, item_num) - predictions for masked positions
        """
        batch_size, seq_len = input_ids.size()
        
        #attention mask (1 for valid tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token).long()
        
        #item embeddings
        item_emb = self.item_embedding(input_ids) 
        
        #position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)
        
        embeddings = item_emb + position_emb
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
            #return full sequence representations
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
                masked_positions.append(0)  #pad with 0
                masked_labels.append(-1)    
            
            batch_data.append({
                'input_ids': masked_seq,
                'masked_positions': masked_positions[:max_masked],
                'masked_labels': masked_labels[:max_masked]
            })
        
        #to tensors
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
            
            target_emb = self.item_embedding(target_items) 
            
            #similarity scores
            scores = torch.bmm(
                last_hidden.unsqueeze(1),
                target_emb.transpose(1, 2) 
            ).squeeze(1)
            
            return scores


class BERT4RecTrainer:
    """Trainer class for BERT4Rec"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, data_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(data_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            masked_positions = batch['masked_positions'].to(self.device)
            masked_labels = batch['masked_labels'].to(self.device)
            
            #Forward pass
            logits = self.model(input_ids, masked_positions)
            
            loss_mask = (masked_labels != -1)
            if loss_mask.sum() > 0:
                active_logits = logits[loss_mask]
                active_labels = masked_labels[loss_mask]
                loss = criterion(active_logits, active_labels)
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def evaluate(self, test_data, num_neg=99, k=10):
        """Evaluate model performance"""
        self.model.eval()
        
        ndcg_sum = 0
        hr_sum = 0
        num_users = 0
        
        with torch.no_grad():
            for user_seq, target_item in tqdm(test_data, desc="Evaluating"):
                candidates = [target_item]
                
                while len(candidates) <= num_neg:
                    neg_item = random.randint(1, self.model.item_num)
                    if neg_item != target_item and neg_item not in user_seq:
                        candidates.append(neg_item)

                if len(user_seq) > self.model.max_seq_len:
                    user_seq = user_seq[-self.model.max_seq_len:]
                
                padded_seq = user_seq + [0] * (self.model.max_seq_len - len(user_seq))
                
                sequences = torch.tensor([padded_seq], dtype=torch.long, device=self.device)
                candidates_tensor = torch.tensor([candidates], dtype=torch.long, device=self.device)
                
                scores = self.model.predict(sequences, candidates_tensor)
                scores = scores.cpu().numpy()[0]
                
                rank = np.argsort(-scores)[0]  
                
                if rank < k:
                    hr_sum += 1
                    ndcg_sum += 1 / np.log2(rank + 2)
                
                num_users += 1
        
        hr = hr_sum / num_users if num_users > 0 else 0
        ndcg = ndcg_sum / num_users if num_users > 0 else 0
        
        return {'HR@10': hr, 'NDCG@10': ndcg}


def prepare_data_from_csv(csv_path, user_col='user_id', item_col='item_id', 
                         min_interactions=5, test_ratio=0.2):
    """
    Prepare data from CSV file with user_id and item_id columns
    
    Args:
        csv_path: path to CSV file
        user_col: name of user column
        item_col: name of item column
        min_interactions: minimum interactions per user
        test_ratio: ratio of data for testing
        
    Returns:
        dict with train_data, test_data, num_users, num_items
    """
    df = pd.read_csv(csv_path)
    
    user_counts = df[user_col].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df[user_col].isin(valid_users)]
    
    #ID mappings (1-indexed for items, 0-indexed for users)
    unique_users = sorted(df[user_col].unique())
    unique_items = sorted(df[item_col].unique())
    
    user_to_id = {user: i for i, user in enumerate(unique_users)}
    item_to_id = {item: i + 1 for i, item in enumerate(unique_items)}  
    
    df['user_idx'] = df[user_col].map(user_to_id)
    df['item_idx'] = df[item_col].map(item_to_id)
    
    user_sequences = defaultdict(list)
    for _, row in df.iterrows():
        user_sequences[row['user_idx']].append(row['item_idx'])
    
    train_data = []
    test_data = []
    
    for user_id, sequence in user_sequences.items():
        if len(sequence) < 2:
            continue
            
        split_point = int(len(sequence) * (1 - test_ratio))
        train_seq = sequence[:split_point]
        test_item = sequence[split_point]  #first test item
        
        if len(train_seq) > 0:
            train_data.append(train_seq)
            test_data.append((train_seq, test_item))
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'num_users': len(unique_users),
        'num_items': len(unique_items),
        'user_to_id': user_to_id,
        'item_to_id': item_to_id
    }
