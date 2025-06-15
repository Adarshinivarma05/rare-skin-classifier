import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes: int = 60, num_classes: int = 7):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(0.4)
        self.prototype_layer = nn.Linear(2048, num_prototypes)
        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.dropout(feats)
        proto_scores = self.prototype_layer(feats)
        logits = self.classifier(proto_scores)
        return logits







