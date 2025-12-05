import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Self-attention over temporal dimension."""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x: (B, T, D)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class MILModel(nn.Module):
    """
    Enhanced MIL Network with temporal attention.
    Input: Bag of Features (Batch, 32, Input_Dim)
    Output: Anomaly Scores (Batch, 32, 1)
    """

    def __init__(self, input_dim=512, hidden_dim=128, use_attention=True):
        super(MILModel, self).__init__()
        
        self.use_attention = use_attention
        
        # Optional temporal attention
        if use_attention:
            self.temporal_attn = TemporalAttention(input_dim, num_heads=4)
        
        # Deeper MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        self.sigmoid = nn.Sigmoid()

        # Weight Initialization
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x shape: (Batch, Segments, Input_Dim) -> e.g. (30, 32, 512)
        Returns: (Batch, Segments, 1) -> e.g. (30, 32, 1)
        """
        B, T, D = x.shape
        
        # Optional attention
        if self.use_attention:
            x = self.temporal_attn(x)  # (B, T, D)
        
        # Flatten for MLP
        x = x.view(B * T, D)

        # Forward Pass
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))

        # Reshape back
        return x.view(B, T, 1)