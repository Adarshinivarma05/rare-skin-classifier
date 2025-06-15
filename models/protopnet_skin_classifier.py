import torch.nn as nn
import torchvision.models as models

class ProtoPNet(nn.Module):
    def __init__(self, num_prototypes: int = 60, num_classes: int = 7, p_drop: float = 0.5):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()           # 2048‑d feature
        self.dropout = nn.Dropout(p_drop)
        self.prototype_layer = nn.Linear(2048, num_prototypes)
        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        ps = self.prototype_layer(x)
        return self.classifier(ps)



