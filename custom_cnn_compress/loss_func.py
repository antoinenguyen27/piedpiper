import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticCompressionLoss(nn.Module):
    def __init__(
        self,
        alpha=20.0,          # semantic dominates
        beta=0.01,           # small auxiliary compression regulariser
        target_keep_ratio=0.2,
        semantic_threshold=0.05,   # kept for compatibility
        gate_strength=10.0,        # kept for compatibility
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.target_keep_ratio = target_keep_ratio
        self.semantic_threshold = semantic_threshold
        self.gate_strength = gate_strength

    def build_topk_mask(self, scores):
        """
        scores: (B, T) raw mask logits / scores

        Returns:
            soft_mask: (B, T) sigmoid(scores)
            hard_mask: (B, T) binary top-k mask
            st_mask:   (B, T) straight-through version:
                       forward = hard, backward = soft
        """
        B, T = scores.shape
        k = max(1, int(round(T * self.target_keep_ratio)))

        soft_mask = torch.sigmoid(scores)

        topk_idx = scores.topk(k, dim=1).indices
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, topk_idx, 1.0)

        # Straight-through estimator:
        # forward uses hard mask, backward flows through soft mask
        st_mask = hard_mask.detach() - soft_mask.detach() + soft_mask

        return soft_mask, hard_mask, st_mask

    def forward(self, orig_embeds, masked_embeds, masks):
        """
        orig_embeds:   (B, D)
        masked_embeds: (B, D)
        masks:         (B, T) raw logits / scores, NOT sigmoid outputs

        IMPORTANT:
        Your MODEL should use the same top-k masking logic in its forward pass
        when producing masked_embeds. This loss does not re-mask embeddings;
        it just builds the same top-k mask for regularisation / logging.
        """

        # 1. Build top-k mask from raw scores
        soft_mask, hard_mask, st_mask = self.build_topk_mask(masks)

        # 2. Semantic preservation loss
        mse = F.mse_loss(masked_embeds, orig_embeds)
        cos = (1.0 - F.cosine_similarity(orig_embeds, masked_embeds, dim=1)).mean()
        semantic_loss = mse + 0.1 * cos

        # 3. Since top-k already enforces exact compression,
        # compression loss just encourages confident scores
        # (not all logits hovering near 0 => sigmoid ~ 0.5)
        binary_loss = (soft_mask * (1.0 - soft_mask)).mean()

        compression_loss = binary_loss

        total_loss = self.alpha * semantic_loss + self.beta * compression_loss

        return total_loss, semantic_loss, compression_loss