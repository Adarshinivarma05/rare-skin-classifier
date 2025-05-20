import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPSkinClassifier(nn.Module):
    def __init__(self, num_classes=7, num_prototypes=70):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove original FC

        self.dropout = nn.Dropout(p=0.5)
        self.prototype_layer = nn.Linear(512, num_prototypes)
        self.output_layer = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.prototype_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x
