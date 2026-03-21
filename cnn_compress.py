import torch
import torch.nn as nn


class VideoCompressor(nn.Module):
    def __init__(
        self,
        input_channels=3,
        feature_dim=128,
        kernel_size=3,
        target_keep_ratio=0.2,
    ):
        super(VideoCompressor, self).__init__()

        self.target_keep_ratio = target_keep_ratio

        # 1. Spatial feature extractor (per frame)
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
        )

        # 2. Temporal scoring network
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=1,   # one importance score per frame
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def build_topk_mask(self, logits):
        """
        logits: (B, T) raw per-frame scores

        Returns:
            soft_mask: sigmoid(logits), shape (B, T)
            hard_mask: binary top-k mask, shape (B, T)
            st_mask:   straight-through mask
                       forward = hard mask
                       backward = soft mask gradients
        """
        B, T = logits.shape
        k = max(1, int(round(T * self.target_keep_ratio)))

        soft_mask = torch.sigmoid(logits)

        topk_idx = logits.topk(k, dim=1).indices
        hard_mask = torch.zeros_like(logits)
        hard_mask.scatter_(1, topk_idx, 1.0)

        # Straight-through estimator
        st_mask = hard_mask.detach() - soft_mask.detach() + soft_mask

        return soft_mask, hard_mask, st_mask

    def forward(self, x):
        """
        x shape: (B, T, C, H, W)

        Returns:
            mask:      (B, T) straight-through top-k mask for use in forward pass
            logits:    (B, T) raw frame scores for loss/logging
            hard_mask: (B, T) actual binary selected frames
        """
        B, T, C, H, W = x.shape

        # Step 1: spatial features per frame
        x_folded = x.view(B * T, C, H, W)
        features = self.spatial_cnn(x_folded)

        # Global average pool -> (B*T, feature_dim)
        pooled_features = torch.mean(features, dim=[2, 3])

        # Reshape to sequence -> (B, T, feature_dim)
        sequence_features = pooled_features.view(B, T, -1)

        # Conv1d expects (B, feature_dim, T)
        sequence_features = sequence_features.transpose(1, 2)

        # Temporal logits -> (B, 1, T) -> (B, T)
        logits = self.temporal_conv(sequence_features).squeeze(1)

        # Build top-k mask
        soft_mask, hard_mask, st_mask = self.build_topk_mask(logits)

        return st_mask, logits, hard_mask