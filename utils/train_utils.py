import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def compute_loss_weights(train_loader):
   all_labels = []
   for _, labels in train_loader:
       all_labels.extend(labels.squeeze().numpy())
   class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
   return torch.tensor(class_weights, dtype=torch.float)

def get_optimizer(model, lr=1e-4):
   return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

def get_scheduler(optimizer):
   return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
