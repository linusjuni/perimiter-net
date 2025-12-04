import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Lightweight temporal self-attention block used in the latest MIL model."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out + x)


class MILModel(nn.Module):
    """
    The 'Detective' Network with temporal attention (newest training setup).
    Input: Bag of Features (Batch, Segments, Input_Dim)
    Output: Anomaly Scores (Batch, Segments, 1)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden1: int = 128,
        hidden2: int = 64,
        num_heads: int = 4,
        dropout: float = 0.6,
    ):
        super().__init__()

        # Two-layer MLP before temporal attention
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.temporal_attn = TemporalAttention(dim=hidden2, num_heads=num_heads)
        self.fc3 = nn.Linear(hidden2, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # Weight Initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (Batch, Segments, Input_Dim) -> e.g. (30, 32, 512)
        Returns: (Batch, Segments, 1) -> e.g. (30, 32, 1)
        """
        B, T, D = x.shape

        # 1. Flatten Bag Dimension (Merge Batch and Time)
        x = x.view(B * T, D)  # (B*T, D)

        # 2. Per-clip MLP
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))

        # 3. Reshape back to Bag and apply temporal attention
        x = x.view(B, T, -1)  # (B, T, hidden2)
        x = self.temporal_attn(x)  # (B, T, hidden2)

        # 4. Clip-level scoring
        x = self.sigmoid(self.fc3(x))  # (B, T, 1)

        return x
