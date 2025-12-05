import torch.nn as nn


class TemporalAttention(nn.Module):
    """Temporal self-attention module."""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class MILModel(nn.Module):
    """MIL network with temporal attention."""

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
        """Forward pass for MIL model."""
        B, T, D = x.shape
        if self.use_attention:
            x = self.temporal_attn(x)
        x = x.view(B * T, D)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x.view(B, T, 1)