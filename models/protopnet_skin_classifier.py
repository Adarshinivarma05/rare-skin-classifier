import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes=70, num_classes=7):  # keep prototypes high for better representation
        super(ProtoPNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        
        self.dropout = nn.Dropout(p=0.5)  # dropout layer with 50% rate
        self.prototype_layer = nn.Linear(512, num_prototypes)
        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        prototype_scores = self.prototype_layer(features)
        logits = self.classifier(prototype_scores)
        return logits
