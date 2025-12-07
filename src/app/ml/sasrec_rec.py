import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
import random
from typing import List, Tuple, Set, Optional

class SASRecDataset(Dataset):
    def __init__(self, sequences, max_len, item_num, interaction_types=4, augment=False):
        self.sequences = sequences
        self.max_len = max_len
        self.item_num = item_num
        self.interaction_types = interaction_types
        self.augment = augment
        
    def __len__(self):
        return len(self.sequences)
    
    def _augment_sequence(self, seq):
        """Apply data augmentation: random crop, mask, or reorder"""
        if len(seq) <= 3 or random.random() > 0.5:
            return seq
        
        aug_type = random.choice(['crop', 'mask', 'reorder'])
        
        if aug_type == 'crop':
            # Randomly crop sequence
            crop_len = random.randint(max(3, len(seq)//2), len(seq))
            start_idx = random.randint(0, len(seq) - crop_len)
            return seq[start_idx:start_idx + crop_len]
        
        elif aug_type == 'mask':
            # Randomly mask some items (replace with padding)
            seq_copy = seq.copy()
            mask_ratio = 0.2
            num_mask = max(1, int(len(seq) * mask_ratio))
            mask_indices = random.sample(range(len(seq)), num_mask)
            for idx in mask_indices:
                seq_copy[idx] = (0, 0)
            return seq_copy
        
        else:  # reorder
            # Slightly reorder sequence (swap adjacent items)
            seq_copy = seq.copy()
            num_swaps = max(1, len(seq) // 10)
            for _ in range(num_swaps):
                if len(seq_copy) < 2:
                    break
                idx = random.randint(0, len(seq_copy) - 2)
                seq_copy[idx], seq_copy[idx + 1] = seq_copy[idx + 1], seq_copy[idx]
            return seq_copy
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Apply augmentation if enabled
        if self.augment:
            seq = self._augment_sequence(seq)
        
        # Create input and target sequences
        if len(seq) <= self.max_len:
            # Pad sequence
            padded_seq = [(0, 0)] * (self.max_len - len(seq)) + seq
            input_seq = padded_seq[:-1] + [(0, 0)]  # Last item becomes padding
            target_seq = padded_seq[1:] + [(0, 0)]  # Shift by 1
        else:
            # Truncate sequence
            input_seq = seq[-(self.max_len+1):-1]
            target_seq = seq[-self.max_len:]
        
        # Separate items and interactions
        input_items = [item for item, _ in input_seq]
        input_interactions = [inter for _, inter in input_seq]
        target_items = [item for item, _ in target_seq]
        
        return {
            'input_items': torch.LongTensor(input_items),
            'input_interactions': torch.LongTensor(input_interactions),
            'target_items': torch.LongTensor(target_items)
        }

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        residual = query
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask is already [batch, 1, seq_len, seq_len], expand for n_heads
            mask = mask.expand(-1, self.n_heads, -1, -1)
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)

class SASRecBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        x = self.attention(x, x, x, mask)
        x = self.feed_forward(x)
        return x

class SASRec(nn.Module):
    def __init__(self, item_num, interaction_types=4, max_len=50, d_model=64, 
                 n_blocks=2, n_heads=1, dropout=0.1):
        super().__init__()
        
        self.item_num = item_num
        self.max_len = max_len
        self.d_model = d_model
        self.interaction_types = interaction_types
        
        # Embeddings
        self.item_emb = nn.Embedding(item_num + 1, d_model, padding_idx=0)
        self.interaction_emb = nn.Embedding(interaction_types + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_blocks)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, input_items, input_interactions):
        batch_size, seq_len = input_items.size()
        
        # Create position indices
        pos_ids = torch.arange(seq_len, device=input_items.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        item_embs = self.item_emb(input_items)
        interaction_embs = self.interaction_emb(input_interactions)
        pos_embs = self.pos_emb(pos_ids)
        
        # Combine embeddings
        x = item_embs + interaction_embs + pos_embs
        x = self.dropout(x)
        
        # Create attention mask (hide padding tokens)
        attention_mask = (input_items != 0).unsqueeze(1).unsqueeze(2)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_items.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine masks
        mask = attention_mask & causal_mask
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.layer_norm(x)
        
        return x
    
    def predict(self, input_items, input_interactions, candidate_items=None):
        """Predict next items for given sequences"""
        seq_output = self.forward(input_items, input_interactions)
        
        # Use last non-padding position for prediction
        seq_emb = seq_output[torch.arange(seq_output.size(0)), 
                           (input_items != 0).sum(1) - 1]  # [batch_size, d_model]
        
        if candidate_items is not None:
            # Score specific candidate items
            candidate_embs = self.item_emb(candidate_items)  # [batch_size, num_candidates, d_model]
            scores = torch.matmul(candidate_embs, seq_emb.unsqueeze(-1)).squeeze(-1)
        else:
            # Score all items
            item_embs = self.item_emb.weight[1:]  # Exclude padding item
            scores = torch.matmul(seq_emb, item_embs.transpose(0, 1))
        
        return scores

def prepare_data(df, min_interactions=5):
    """
    Prepare sequential data from DataFrame
    
    Args:
        df: DataFrame with columns ['user_idx', 'item_idx', 'interaction_idx', 'timestamp']
        min_interactions: Minimum number of interactions per user
    
    Returns:
        sequences: List of sequences for each user
        item_num: Number of unique items
        user_map: Mapping from original user_idx to sequential indices
        item_map: Mapping from original item_idx to sequential indices
    """
    
    # Sort by user and timestamp
    df = df.sort_values(['user_idx', 'timestamp']).reset_index(drop=True)
    
    # Filter users with minimum interactions
    user_counts = df['user_idx'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['user_idx'].isin(valid_users)].reset_index(drop=True)
    
    # Create mappings (starting from 1, 0 is reserved for padding)
    unique_users = df['user_idx'].unique()
    unique_items = df['item_idx'].unique()
    
    user_map = {user: idx + 1 for idx, user in enumerate(unique_users)}
    item_map = {item: idx + 1 for idx, item in enumerate(unique_items)}
    
    # Map to new indices
    df['user_mapped'] = df['user_idx'].map(user_map)
    df['item_mapped'] = df['item_idx'].map(item_map)
    
    # Group by user to create sequences
    sequences = []
    user_sequences = df.groupby('user_mapped')
    
    for user_id, group in user_sequences:
        # Create sequence of (item, interaction) pairs
        sequence = list(zip(group['item_mapped'].tolist(), 
                           group['interaction_idx'].tolist()))
        sequences.append(sequence)
    
    return sequences, len(unique_items), user_map, item_map

def train_sasrec(sequences, item_num, max_len=50, batch_size=256, 
                lr=0.001, epochs=100, device='cpu', d_model=32, n_blocks=1, 
                n_heads=1, dropout=0.5, weight_decay=1e-4, patience=10, 
                use_augmentation=False):
    """
    Train SASRec model
    
    Args:
        sequences: List of user sequences
        item_num: Number of unique items
        max_len: Maximum sequence length
        batch_size: Training batch size
        lr: Learning rate
        epochs: Number of training epochs
        device: Training device
        d_model: Model dimension (reduce for small datasets)
        n_blocks: Number of transformer blocks (reduce for small datasets)
        n_heads: Number of attention heads
        dropout: Dropout rate (increase for regularization)
        weight_decay: L2 regularization (increase for regularization)
        patience: Early stopping patience
        use_augmentation: Whether to use data augmentation (set False for fair comparison)
    
    Returns:
        Trained SASRec model
    """
    
    # Split sequences into train/validation
    train_sequences, val_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
    
    # Create datasets - only augment training data if specified
    train_dataset = SASRecDataset(train_sequences, max_len, item_num, augment=use_augmentation)
    val_dataset = SASRecDataset(val_sequences, max_len, item_num, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with regularization-friendly hyperparameters
    model = SASRec(item_num, max_len=max_len, d_model=d_model, 
                   n_blocks=n_blocks, n_heads=n_heads, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_items = batch['input_items'].to(device)
            input_interactions = batch['input_interactions'].to(device)
            target_items = batch['target_items'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            seq_output = model(input_items, input_interactions)
            
            # Compute loss for all positions
            # Note: item embeddings start from index 1, so we need to adjust targets
            logits = torch.matmul(seq_output, model.item_emb.weight[1:].transpose(0, 1))
            
            # Adjust target items: subtract 1 to align with logits indices (0 to item_num-1)
            # Targets with value 0 (padding) become -1 and will be ignored by criterion
            adjusted_targets = target_items - 1
            
            loss = criterion(logits.view(-1, item_num), adjusted_targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_items = batch['input_items'].to(device)
                input_interactions = batch['input_interactions'].to(device)
                target_items = batch['target_items'].to(device)
                
                seq_output = model(input_items, input_interactions)
                logits = torch.matmul(seq_output, model.item_emb.weight[1:].transpose(0, 1))
                
                # Adjust target items: subtract 1 to align with logits indices
                # Targets with value 0 (padding) become -1 and will be ignored by criterion
                adjusted_targets = target_items - 1
                
                loss = criterion(logits.view(-1, item_num), adjusted_targets.view(-1))
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_sasrec_model.pth')
            print(f'  → Saved best model (val_loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  → No improvement ({patience_counter}/{patience})')
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_sasrec_model.pth'))
    return model

class SASRecRecommender:
    """
    Wrapper class for SASRec model to integrate with hybrid system
    Handles item mapping and provides clean prediction interface
    """
    
    def __init__(self, 
                 model,
                 item_map: dict,
                 reverse_item_map: dict,
                 device: torch.device,
                 max_seq_len: int = 50):
        """
        Args:
            model: Trained SASRec model
            item_map: Dict mapping original item_idx -> model item_idx (1-indexed)
            reverse_item_map: Dict mapping model item_idx -> original item_idx
            device: torch device
            max_seq_len: Maximum sequence length
        """
        self.model = model
        self.item_map = item_map
        self.reverse_item_map = reverse_item_map
        self.device = device
        self.max_seq_len = max_seq_len
        self.model.eval()
    
    def predict(self, 
                user_sequence: List[Tuple[int, int]], 
                top_k: int = 10,
                exclude_items: Optional[Set[int]] = None) -> List[int]:
        """
        Get top-K recommendations for a user sequence
        
        Args:
            user_sequence: List of (item_idx, interaction_idx) tuples
                          item_idx: original item ID from dataset
                          interaction_idx: 0=click, 1=like, 2=comment, 3=share
            top_k: Number of recommendations to return
            exclude_items: Set of items to exclude (already interacted)
        
        Returns:
            List of recommended item indices (original item IDs)
        """
        if exclude_items is None:
            exclude_items = set()
        
        # Convert sequence to model indices
        mapped_items = []
        interactions = []
        
        for item, interaction in user_sequence:
            if item in self.item_map:
                mapped_items.append(self.item_map[item])
                # Note: Interactions stay as-is because training handles them correctly
                interactions.append(interaction)
        
        if not mapped_items:
            return []
        
        # Truncate to max length (keep most recent)
        if len(mapped_items) > self.max_seq_len:
            mapped_items = mapped_items[-self.max_seq_len:]
            interactions = interactions[-self.max_seq_len:]
        
        # Pad if necessary
        if len(mapped_items) < self.max_seq_len:
            pad_len = self.max_seq_len - len(mapped_items)
            mapped_items = [0] * pad_len + mapped_items
            interactions = [0] * pad_len + interactions
        
        # Prepare input tensors
        input_items = torch.LongTensor([mapped_items]).to(self.device)
        input_interactions = torch.LongTensor([interactions]).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            scores = self.model.predict(input_items, input_interactions)  # [1, item_num]
            scores = scores.squeeze(0)  # [item_num]
        
        # Get top-K items (scores are for items 1 to item_num)
        top_scores, top_indices = torch.topk(scores, min(top_k * 3, len(scores)))
        
        # Convert back to original item IDs and filter
        recommendations = []
        for idx in top_indices.cpu().numpy():
            model_item_id = idx + 1  # Model items are 1-indexed
            
            if model_item_id in self.reverse_item_map:
                original_item_id = self.reverse_item_map[model_item_id]
                
                # Skip excluded items
                if original_item_id not in exclude_items:
                    recommendations.append(original_item_id)
                
                if len(recommendations) >= top_k:
                    break
        
        return recommendations
    
    def predict_batch(self, 
                     user_sequences: List[List[Tuple[int, int]]], 
                     top_k: int = 10) -> List[List[int]]:
        """
        Get recommendations for multiple users at once
        
        Args:
            user_sequences: List of user sequences
            top_k: Number of recommendations per user
        
        Returns:
            List of recommendation lists
        """
        all_recommendations = []
        for sequence in user_sequences:
            recs = self.predict(sequence, top_k=top_k)
            all_recommendations.append(recs)
        
        return all_recommendations


def load_sasrec_recommender(path: str, device: torch.device):
    """
    Load a saved SASRec recommender from checkpoint
    
    Args:
        path: Path to saved model checkpoint
        device: Torch device
    
    Returns:
        SASRecRecommender instance
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Reconstruct model with saved architecture config
    config = checkpoint['model_config']
    model = SASRec(
        item_num=config['item_num'],
        interaction_types=config.get('interaction_types', 4),
        max_len=config.get('max_len', 50),
        d_model=config.get('d_model', 32),
        n_blocks=config.get('n_blocks', 1),
        n_heads=config.get('n_heads', 1),
        dropout=config.get('dropout', 0.5)
    )
    
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    return SASRecRecommender(
        model=model,
        item_map=checkpoint['item_map'],
        reverse_item_map=checkpoint['reverse_item_map'],
        device=device,
        max_seq_len=checkpoint.get('max_seq_len', 50)
    )



if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("/Users/allisonlawlixuan/Documents/repos/recommendation_system/src/data/mock_interactions.csv")
    
    # Prepare data
    sequences, item_num, user_map, item_map = prepare_data(df)
    
    # Create reverse mapping for wrapper
    reverse_item_map = {v: k for k, v in item_map.items()}
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_sasrec(
        sequences, item_num, 
        d_model=32, n_blocks=1, dropout=0.5, weight_decay=1e-3,
        use_augmentation=False,  # Keep False for fair hybrid comparison
        device=device
    )
    
    # Create wrapper
    sasrec_recommender = SASRecRecommender(
        model=model,
        item_map=item_map,
        reverse_item_map=reverse_item_map,
        device=device,
        max_seq_len=50
    )
    
    # Test it
    test_sequence = [(101, 0), (102, 1)]  # Original item IDs
    recommendations = sasrec_recommender.predict(test_sequence, top_k=10)
    print(f"✓ Test recommendations: {recommendations}")
    
    # Save the complete recommender
    torch.save({
        'model_state': model.state_dict(),
        'item_map': item_map,
        'reverse_item_map': reverse_item_map,
        'max_seq_len': 50,
        'model_config': {
            'item_num': item_num,
            'interaction_types': 4,
            'max_len': 50,
            'd_model': 32,
            'n_blocks': 1,
            'n_heads': 1,
            'dropout': 0.5,
        }
    }, 'models/weights/sasrec_recommender.pt')
    
    print("✓ SASRec recommender saved to 'sasrec_recommender.pt'")
    print("\nNext steps:")
    print("1. Use this sasrec_recommender in your hybrid system")
    print("2. Import: from sasrec_rec import load_sasrec_recommender")
    print("3. Load: sasrec = load_sasrec_recommender('sasrec_recommender.pt', device)")