
import torchvision.models as models
import torch.nn as nn


model = models.mobilenet_v3_small(pretrained=True)
model.fc = nn.Linear(1024, 4)

print(model)