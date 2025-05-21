import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes=70, num_classes=7):
        super(ProtoPNet, self).__init__()
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove FC layers
        self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Better context
        self.prototype_vectors = nn.Parameter(torch.rand(num_prototypes, 512, 1, 1))
        self.last_layer = nn.Linear(num_prototypes, num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        distances = torch.cdist(x.view(x.size(0), -1), self.prototype_vectors.view(self.prototype_vectors.size(0), -1))
        similarity = -distances  # Higher similarity = closer
        logits = self.last_layer(similarity)
        return logits
