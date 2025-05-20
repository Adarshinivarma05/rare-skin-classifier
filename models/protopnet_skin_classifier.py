import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes=30, num_classes=7, prototype_shape=(512, 1, 1)):
        super(ProtoPNet, self).__init__()
        # Backbone: ResNet18 pretrained feature extractor without final fc layer
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()

        # Prototype vectors: (num_prototypes, feature_dim, 1, 1)
        # Initialized randomly; learnable parameters
        self.prototype_vectors = nn.Parameter(torch.rand(num_prototypes, *prototype_shape))

        # Linear layer: connect prototype activations to class logits
        self.last_layer = nn.Linear(num_prototypes, num_classes, bias=True)

    def forward(self, x):
        # Extract features: shape (batch_size, feature_dim)
        features = self.backbone(x)  # (B, 512)

        # Reshape features for distance calculation: (B, 512, 1, 1)
        features = features.unsqueeze(-1).unsqueeze(-1)

        # Calculate squared L2 distance between features and each prototype
        # Expand dims for broadcasting
        # features: (B, C, 1, 1), prototype_vectors: (P, C, 1, 1)
        # distance: (B, P)
        distances = torch.sum((features - self.prototype_vectors) ** 2, dim=(1, 2, 3))

        # Convert distances to similarity scores (negative distances)
        prototype_activations = -distances

        # Classification logits from prototype activations
        logits = self.last_layer(prototype_activations)

        return logits, prototype_activations, distances

    def prototype_vectors_normalized(self):
        # Optional helper: return normalized prototype vectors for visualization
        return F.normalize(self.prototype_vectors, p=2, dim=1)
