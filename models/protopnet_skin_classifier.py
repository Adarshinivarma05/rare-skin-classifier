# models/protopnet_skin_classifier.py
import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes_per_class=5, num_classes=7):
        super(ProtoPNet, self).__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes_per_class * num_classes

        # Backbone: Pretrained ResNet-18 without final FC
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()

        # Prototype layer
        self.prototype_vectors = nn.Parameter(torch.rand(self.num_prototypes, 512))  # 512 from ResNet18

        # Classifier: fixed prototype-class association
        self.prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        for j in range(num_classes):
            for i in range(num_prototypes_per_class):
                self.prototype_class_identity[j * num_prototypes_per_class + i, j] = 1
        self.prototype_class_identity = nn.Parameter(self.prototype_class_identity, requires_grad=False)

    def forward(self, x):
        features = self.backbone(x)  # Shape: (B, 512)
        x_exp = features.unsqueeze(1)  # (B, 1, 512)
        p_exp = self.prototype_vectors.unsqueeze(0)  # (1, num_prototypes, 512)

        # Compute L2 distances
        distances = torch.norm(x_exp - p_exp, dim=2)  # (B, num_prototypes)

        # Convert distances to similarity scores
        similarity = torch.exp(-distances)

        # Classification scores
        logits = similarity @ self.prototype_class_identity  # (B, num_classes)
        return logits
