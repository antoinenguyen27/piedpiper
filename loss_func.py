import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticCompressionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        """
        alpha: Weight for preserving semantic meaning (The Shield)
        beta: Weight for dropping frames (The Sledgehammer)
        """
        super(SemanticCompressionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, orig_embeds, masked_embeds, masks):
        """
        orig_embeds:   (Batch, Feature_Dim) - Output of VideoPrism on raw video
        masked_embeds: (Batch, Feature_Dim) - Output of VideoPrism on masked video
        masks:         (Batch, Time) - The probability masks from our lightweight network
        """

        # --- 1. The Semantic Loss (Alpha) ---
        # Cosine similarity returns values from -1 to 1.
        # We do (1 - cos_sim) so that identical embeddings = 0 loss.
        cos_sim = F.cosine_similarity(orig_embeds, masked_embeds, dim=1)
        semantic_loss = (1.0 - cos_sim).mean()

        # --- 2. The Compression Loss (Beta) ---
        # Instead of taking the absolute sum, we take the mean of the masks.
        # This gives us the *proportion* of frames kept (a value from 0 to 1).
        # PRO TIP: Using mean() instead of sum() makes your beta hyperparameter
        # scale-invariant! It won't break if you change the video length from 32 to 64.
        compression_loss = masks.mean()

        # --- 3. The Aggregate Loss ---
        total_loss = (self.alpha * semantic_loss) + (self.beta * compression_loss)

        # We return all three so you can log them to Weights & Biases or TensorBoard
        # to watch the two competing forces fight during training.
        return total_loss, semantic_loss, compression_loss

