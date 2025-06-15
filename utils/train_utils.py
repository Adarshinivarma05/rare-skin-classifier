# utils/train_utils.py  (add)
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    CE‑based focal loss to focus learning on harder / minority samples.
    gamma: 2.0 is standard; higher = more focus on hard examples.
    """
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)           # pt = 1‑p_t
        return torch.mean((1 - pt) ** self.gamma * ce)
