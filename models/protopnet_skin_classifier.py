import torch.nn as nn
import torchvision.models as models


class ProtoPNet(nn.Module):
    """
    ProtoPNet with ResNet‑50 backbone, dropout, and 80 prototypes
    Supports prototype-based linear or direct classification based on use_proto flag
    """
    def __init__(self, num_prototypes: int = 80, num_classes: int = 7, p_drop=0.4, use_proto=True):
        super().__init__()
        self.use_proto = use_proto
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()          # 2048‑d features
        self.dropout = nn.Dropout(p_drop)
        self.prototype_layer = nn.Linear(2048, num_prototypes)
        self.classifier = nn.Linear(num_prototypes if use_proto else 2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        
        if self.use_proto:
            proto_output = self.prototype_layer(x)
            out = self.classifier(proto_output)
        else:
            out = self.classifier(x)
        
        return out
