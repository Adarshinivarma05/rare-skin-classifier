import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes=70, num_classes=7):
        super(ProtoPNet, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.add_dropout = nn.Dropout(0.5)

        self.prototype_layer = nn.Conv2d(512, num_prototypes, kernel_size=1)
        self.last_layer = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.add_dropout(features)
        proto_distances = self.prototype_layer(features)
        pooled = nn.functional.adaptive_avg_pool2d(proto_distances, (1, 1)).squeeze()
        logits = self.last_layer(pooled)
        return logits
