import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes=10, num_classes=7, dropout_prob=0.3):
        super(ProtoPNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(dropout_prob)
        self.prototype_layer = nn.Linear(512, num_prototypes)
        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        prototype_scores = self.prototype_layer(features)
        logits = self.classifier(prototype_scores)
        return logits
