import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes=10, num_classes=7):
        super(ProtoPNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()

        self.prototype_layer = nn.Linear(512, num_prototypes)
        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        prototype_scores = self.prototype_layer(features)
        logits = self.classifier(prototype_scores)
        return logits

