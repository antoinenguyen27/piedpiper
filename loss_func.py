import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticCompressionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, target_sparsity=0.2):
        """
        alpha: Weight for preserving semantic meaning.
        beta:  Weight for the compression constraint.
        target_sparsity: The ideal % of frames to KEEP (e.g., 0.2 = keep 20%).
        """
        super(SemanticCompressionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.target_sparsity = target_sparsity

    def forward(self, orig_embeds, masked_embeds, masks):
        # --- 1. The Semantic Loss (The Shield) ---
        cos_sim = F.cosine_similarity(orig_embeds, masked_embeds, dim=1)
        semantic_loss = (1.0 - cos_sim).mean()

        # --- 2. The Budget Constraint (The Balance) ---
        # We calculate the actual proportion of frames kept
        current_sparsity = masks.mean()

        # Option A: MSE Loss - Pulls the model toward exactly 20%
        compression_loss = (current_sparsity - self.target_sparsity) ** 2

        # Option B: Hinge Loss - Only penalizes if we exceed the budget
        # compression_loss = torch.max(torch.tensor(0.0), current_sparsity - self.target_sparsity)

        # --- 3. The Aggregate Loss ---
        total_loss = (self.alpha * semantic_loss) + (self.beta * compression_loss)

        return total_loss, semantic_loss, compression_loss