import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPNet(nn.Module):
   def __init__(self, num_prototypes=50, num_classes=7):
       super(ProtoPNet, self).__init__()
       self.backbone = models.resnet50(pretrained=True)
       self.backbone.fc = nn.Identity()

       self.prototype_layer = nn.Linear(2048, num_prototypes)
       self.classifier = nn.Linear(num_prototypes, num_classes)

   def forward(self, x):
       features = self.backbone(x)
       proto_scores = self.prototype_layer(features)
       logits = self.classifier(proto_scores)
       return logits
