import torch
import torch.nn as nn


class VideoCompressor(nn.Module):
    def __init__(self, input_channels=3, feature_dim=128, kernel_size=3):
        super(VideoCompressor, self).__init__()

        # 1. The Spatial Squeeze (Frame-by-Frame Feature Extractor)
        # I added stride=2 here to actually achieve the "aggressive downsampling"
        # we talked about. This cuts the resolution in half at each step.
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1)
            # Removed ReLU here to allow the final feature vector to be more expressive
        )

        # 2. The Temporal Network (Sliding Window across Time)
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=1,  # Output 1 drop probability per frame
            kernel_size=kernel_size,
            padding=kernel_size // 2  # Keeps output length identical to input length
        )

    def forward(self, x):
        # x shape: (Batch, Time, Channels, Height, Width)
        B, T, C, H, W = x.shape

        # --- STEP 1: Process Spatial Features ---
        # PyTorch Conv2d expects 4D inputs. We temporarily fold the Batch and Time
        # dimensions together to push all frames through the CNN in parallel.
        # Shape becomes: (B * T, C, H, W)
        x_folded = x.view(B * T, C, H, W)

        features = self.spatial_cnn(x_folded)

        # Global average pooling: Crushes (B*T, feature_dim, H', W') -> (B*T, feature_dim)
        pooled_features = torch.mean(features, dim=[2, 3])

        # --- STEP 2: Process Temporal Features ---
        # Unfold back to our sequence format.
        # Shape: (Batch, Time, feature_dim)
        sequence_features = pooled_features.view(B, T, -1)

        # Conv1d expects the sequence length to be the LAST dimension.
        # Shape: (Batch, feature_dim, Time)
        sequence_features = sequence_features.transpose(1, 2)

        # Slide the kernel across the timeline.
        # Shape: (Batch, 1, Time)
        logits = self.temporal_conv(sequence_features)

        # Apply sigmoid to get probabilities between 0 and 1,
        # and squeeze out the unnecessary channel dimension.
        # Final mask shape: (Batch, Time)
        mask = torch.sigmoid(logits).squeeze(1)

        return mask