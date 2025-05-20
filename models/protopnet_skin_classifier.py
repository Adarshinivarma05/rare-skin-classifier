# models/protopnet_skin_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ProtoPSkinClassifier(nn.Module):
    def __init__(self, num_classes=3, num_prototypes=70, prototype_dim=256):
        super(ProtoPSkinClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove original classification layer

        self.dropout = nn.Dropout(p=0.5)
        self.prototype_layer = nn.Linear(512, prototype_dim)
        self.prototype_vectors = nn.Parameter(torch.rand(num_prototypes, prototype_dim))
        self.classifier = nn.Linear(num_prototypes, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        proto_features = self.prototype_layer(features)
        proto_features = F.normalize(proto_features, dim=1)
        prototypes = F.normalize(self.prototype_vectors, dim=1)
        distances = torch.cdist(proto_features.unsqueeze(1), prototypes.unsqueeze(0))
        similarity = -distances.squeeze(1)
        out = self.classifier(similarity)
        return out
