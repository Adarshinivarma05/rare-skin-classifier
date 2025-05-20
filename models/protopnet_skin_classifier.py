import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ProtoPSkinClassifier(nn.Module):
    def __init__(self, num_prototypes=70, num_classes=7, prototype_dim=256):
        super(ProtoPSkinClassifier, self).__init__()

        # Pretrained ResNet18 as feature extractor
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification layer
        self.feature_dim = self.backbone.fc.in_features if hasattr(self.backbone.fc, 'in_features') else 512

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

        # Prototypes (num_prototypes x prototype_dim)
        self.prototype_layer = nn.Linear(self.feature_dim, num_prototypes, bias=False)

        # Classification layer
        self.last_layer = nn.Linear(num_prototypes, num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        proto_activations = self.prototype_layer(x)
        logits = self.last_layer(F.relu(proto_activations))
        return logits
