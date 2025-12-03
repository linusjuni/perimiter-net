import torch
import torch.nn as nn

class MILModel(nn.Module):
    """
    The 'Detective' Network.
    Input: Bag of Features (Batch, 32, Input_Dim)
    Output: Anomaly Scores (Batch, 32, 1)
    """

    def __init__(self, input_dim=512):
        super(MILModel, self).__init__()

        # Architecture following Sultani et al. (CVPR 2018)
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
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


class TwoStreamMIL(nn.Module):
    """
    Two-Stream MIL for Video Anomaly Detection.

    Combines RGB and Motion features using either:
    - Early fusion: Concatenate features → single MIL head
    - Late fusion: Separate MIL heads → combine scores
    """

    def __init__(
        self,
        input_dim: int = 512,
        fusion_mode: str = "early",
        rgb_checkpoint: str = None,
        motion_checkpoint: str = None,
        freeze_streams: bool = True,
    ):
        """
        Args:
            input_dim: Feature dimension per stream (default: 512)
            fusion_mode: 'early' or 'late'
            rgb_checkpoint: Path to pre-trained RGB MIL checkpoint (for late fusion)
            motion_checkpoint: Path to pre-trained Motion MIL checkpoint (for late fusion)
            freeze_streams: Whether to freeze pre-trained streams in late fusion
        """
        super(TwoStreamMIL, self).__init__()

        self.fusion_mode = fusion_mode
        self.input_dim = input_dim

        if fusion_mode == "early":
            # Single MIL head with concatenated features (1024-dim)
            self.mil_head = MILModel(input_dim=input_dim * 2)

        elif fusion_mode == "late":
            # Separate MIL heads for each stream
            self.rgb_stream = MILModel(input_dim=input_dim)
            self.motion_stream = MILModel(input_dim=input_dim)

            # Load pre-trained weights if provided
            if rgb_checkpoint:
                self._load_stream_checkpoint(self.rgb_stream, rgb_checkpoint)
            if motion_checkpoint:
                self._load_stream_checkpoint(self.motion_stream, motion_checkpoint)

            # Freeze streams if requested
            if freeze_streams:
                self._freeze_stream(self.rgb_stream)
                self._freeze_stream(self.motion_stream)

            # Learnable fusion weights (initialized to equal weighting)
            self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        else:
            raise ValueError(
                f"Unknown fusion_mode: {fusion_mode}. Use 'early' or 'late'."
            )

    def _load_stream_checkpoint(self, stream: nn.Module, checkpoint_path: str):
        """Load pre-trained weights into a stream."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        stream.load_state_dict(state_dict)

    def _freeze_stream(self, stream: nn.Module):
        """Freeze all parameters in a stream."""
        for param in stream.parameters():
            param.requires_grad = False

    def forward(
        self, rgb_features: torch.Tensor, motion_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the two-stream MIL model."""
        if self.fusion_mode == "early":
            # Concatenate along feature dimension
            fused = torch.cat([rgb_features, motion_features], dim=-1)  # (B, 32, 1024)
            return self.mil_head(fused)

        elif self.fusion_mode == "late":
            # Get scores from each stream
            rgb_scores = self.rgb_stream(rgb_features)  # (B, 32, 1)
            motion_scores = self.motion_stream(motion_features)  # (B, 32, 1)

            # Weighted combination (softmax ensures weights sum to 1)
            weights = torch.softmax(self.fusion_weight, dim=0)
            fused_scores = weights[0] * rgb_scores + weights[1] * motion_scores

            return fused_scores

    def get_stream_scores(
        self, rgb_features: torch.Tensor, motion_features: torch.Tensor
    ) -> dict:
        """
        Get individual stream scores (useful for analysis).
        Only available in late fusion mode.
        """
        if self.fusion_mode != "late":
            raise ValueError("get_stream_scores only available in late fusion mode")

        with torch.no_grad():
            rgb_scores = self.rgb_stream(rgb_features)
            motion_scores = self.motion_stream(motion_features)
            weights = torch.softmax(self.fusion_weight, dim=0)

        return {
            "rgb": rgb_scores,
            "motion": motion_scores,
            "weights": weights.detach().cpu().numpy(),
        }