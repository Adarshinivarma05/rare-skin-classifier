import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPSkinClassifier(nn.Module):
    def __init__(self, num_prototypes=70, num_classes=7):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # remove original classifier

        # Add dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

        # Prototype layer (learnable prototype vectors)
        self.prototype_vectors = nn.Parameter(torch.randn(num_prototypes, 512))

        # Classification layer
        self.classifier = nn.Linear(num_prototypes, num_classes)

        # Optional BatchNorm before classifier
        self.batchnorm = nn.BatchNorm1d(num_prototypes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)

        # Compute distances to prototypes (Euclidean)
        # features shape: (batch_size, 512)
        # prototype_vectors shape: (num_prototypes, 512)
        # Expand for broadcasting
        features_exp = features.unsqueeze(1)  # (batch_size, 1, 512)
        protos_exp = self.prototype_vectors.unsqueeze(0)  # (1, num_prototypes, 512)
        distances = torch.sum((features_exp - protos_exp) ** 2, dim=2)  # (batch_size, num_prototypes)

        # Convert distances to similarity scores (you may use negative distances)
        sim_scores = -distances

        normed_scores = self.batchnorm(sim_scores)
        logits = self.classifier(normed_scores)
        return logits
