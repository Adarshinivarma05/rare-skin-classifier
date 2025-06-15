import torch
import torch.nn as nn
import torchvision.models as models


class ProtoPNet(nn.Module):
    """
    ProtoPNet with a ResNet50 backbone, dropout, and 90 prototypes
    """
    def __init__(self, num_prototypes: int = 90, num_classes: int = 7):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()           # 2048‑d global feature
        self.dropout = nn.Dropout(0.4)

        self.prototype_layer = nn.Linear(2048, num_prototypes)
        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        feats = self.backbone(x)          # [B, 2048]
        feats = self.dropout(feats)
        proto_scores = self.prototype_layer(feats)  # [B, P]
        logits = self.classifier(proto_scores)      # [B, C]
        return logits
