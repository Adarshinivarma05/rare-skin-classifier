import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes=30, num_classes=7):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove last fc layer
        self.prototype_vectors = nn.Parameter(torch.rand(num_prototypes, 512))  # 512-dim features from ResNet18
        self.linear = nn.Linear(num_prototypes, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone(x)              # Extract features
        dists = torch.cdist(features, self.prototype_vectors)  # Compute distance to prototypes
        sims = -dists                           # Similarities (negative distance)
        logits = self.linear(sims)              # Classification logits
        return logits, sims

