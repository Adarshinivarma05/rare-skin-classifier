import torch
import torch.nn as nn
import torchvision.models as models

class ProtoPNet(nn.Module):
def init(self, num_prototypes, num_classes):
super().init()
self.backbone = models.resnet18(pretrained=True)
self.backbone.fc = nn.Identity() # Remove last FC
self.prototype_vectors = nn.Parameter(torch.rand(num_prototypes, 512))
self.linear = nn.Linear(num_prototypes, num_classes, bias=False)

def forward(self, x):
    features = self.backbone(x)  # shape: (B, 512)
    dists = torch.cdist(features, self.prototype_vectors)  # shape: (B, P)
    sims = -dists  # convert distance to similarity
    logits = self.linear(sims)
    return logits, sims

