import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """Self-attention over temporal dimension with Pre-Norm stability."""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T, D)
        # Pre-Norm (x + Attention(Norm(x))) is often more stable than Post-Norm
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + self.dropout(attn_out)

class MILModel(nn.Module):
    """
    Enhanced MIL Network with Temporal Attention and Positional Encoding.
    """
    def __init__(self, input_dim=512, hidden_dim=128, use_attention=True, segments=32):
        super(MILModel, self).__init__()
        
        self.use_attention = use_attention
        
        # 1. Positional Encoding (The "Order Matters" Fix)
        # We learn a unique vector for Segment 1, Segment 2, ... Segment 32
        if use_attention:
            self.pos_embed = nn.Parameter(torch.zeros(1, segments, input_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02) # Standard transformer init
            
            self.temporal_attn = TemporalAttention(input_dim, num_heads=4)
        
        # 2. Deeper MLP (feature projection)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32) # Bottleneck
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6) 
        self.sigmoid = nn.Sigmoid()

        # Weight Initialization
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x shape: (Batch, Segments, Input_Dim)
        """
        B, T, D = x.shape
        
        # Apply Attention
        if self.use_attention:
            # Add Position Info: "This feature is from Time Step 5"
            # We interpolate pos_embed if T doesn't match segments (robustness)
            if T == self.pos_embed.shape[1]:
                x = x + self.pos_embed
            else:
                # Handle test time if we don't compress (variable length)
                # For variable length, we usually skip pos_embed or interpolate it.
                # For now, let's skip adding pos_embed if sizes mismatch to avoid crash
                pass 

            x = self.temporal_attn(x)
        
        # Flatten for MLP
        x = x.view(B * T, D)

        # Forward Pass (MLP)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))

        # Reshape back
        return x.view(B, T, 1)