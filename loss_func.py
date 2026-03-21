import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticCompressionLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        beta=0.1,
        gamma=0.1,
        eps=1e-8,
    ):
        """
        alpha: Weight for preserving semantic meaning
        beta:  Weight for overall compression (smaller average masks)
        gamma: Weight for discouraging uniform masks across time
        eps:   Small constant for numerical stability
        """
        super(SemanticCompressionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, orig_embeds, masked_embeds, masks):
        """
        orig_embeds:   (Batch, Feature_Dim)
        masked_embeds: (Batch, Feature_Dim)
        masks:         (Batch, Time)
        """

        # --- 1. Semantic Loss ---
        # Cosine similarity preserves semantic direction, but by itself it
        # allows the "dimmer switch" exploit because it ignores scale.
        cos_sim = F.cosine_similarity(orig_embeds, masked_embeds, dim=1)
        semantic_loss = (1.0 - cos_sim).mean()

        # --- 2. Compression Loss ---
        # Keep the original compression objective: lower mean mask = stronger compression.
        compression_loss = masks.mean()

        # --- 3. Shape Loss ---
        # Normalize masks across time so they form a distribution over frames.
        # This removes absolute scale and lets us measure how UNIFORMLY the mask
        # mass is spread over time.
        #
        # If the model uses a flat mask like [0.2, 0.2, 0.2, 0.2], then after
        # normalization p is uniform, which has HIGH entropy -> high loss.
        #
        # If the model concentrates mass on a subset of frames, p becomes peaky,
        # which has LOWER entropy -> lower loss.
        p = masks / (masks.sum(dim=1, keepdim=True) + self.eps)
        shape_loss = -(p * torch.log(p + self.eps)).sum(dim=1).mean()

        # --- 4. Aggregate ---
        total_loss = (
            self.alpha * semantic_loss
            + self.beta * compression_loss
            + self.gamma * shape_loss
        )

        # Keep original return style: total + the two main logged components.
        return total_loss, semantic_loss, compression_loss