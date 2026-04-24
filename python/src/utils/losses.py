from __future__ import annotations
import torch
import torch.nn as nn



class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).

    Given L2-normalized features z of shape (B, D) and integer labels y of shape (B,),
    pulls same-class samples together and pushes different-class apart in the embedding.

    Usage:
        loss = supcon(z, y)   # z is already F.normalize'd, labels are long tensor

    Notes for small batch sizes:
        With batch_size=32 and 10 classes, some classes may have only one sample in the
        batch. Such samples contribute 0 loss (no positives to pull toward). That's fine
        and handled gracefully — just expect slightly noisier gradients than SupCon with
        large batches.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B = features.size(0)

        # similarity matrix (B, B), scaled by temperature
        sim = torch.matmul(features, features.T) / self.temperature

        # numerical stability: subtract per-row max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # mask[i, j] = 1 if labels[i] == labels[j] else 0
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # remove self-similarity from both mask and denominator
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        # log softmax over non-self entries
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # mean log-prob over positives for each anchor
        pos_count = mask.sum(dim=1)
        # avoid div-by-zero for anchors with no positives in the batch
        valid = pos_count > 0
        mean_log_prob_pos = (mask * log_prob).sum(dim=1)
        mean_log_prob_pos = torch.where(valid, mean_log_prob_pos / pos_count.clamp(min=1), torch.zeros_like(mean_log_prob_pos))

        loss = -mean_log_prob_pos[valid].mean() if valid.any() else torch.tensor(0.0, device=device)
        return loss