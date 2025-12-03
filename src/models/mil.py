import torch.nn as nn


class MILModel(nn.Module):
    """
    The 'Detective' Network.
    Input: Bag of Features (Batch, 32, Input_Dim)
    Output: Anomaly Scores (Batch, 32, 1)
    """

    def __init__(self, input_dim=512, dropout=0.6):
        super(MILModel, self).__init__()

        # Architecture following Sultani et al. (CVPR 2018)
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # Weight Initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        """
        x shape: (Batch, Segments, Input_Dim) -> e.g. (30, 32, 512)
        Returns: (Batch, Segments, 1) -> e.g. (30, 32, 1)
        """
        B, T, D = x.shape

        # 1. Flatten Bag Dimension (Merge Batch and Time)
        x = x.view(B * T, D)  # (960, 512)

        # 2. Forward Pass
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))  # (960, 1)

        # 3. Reshape back to Bag
        x = x.view(B, T, 1)  # (30, 32, 1)

        return x
