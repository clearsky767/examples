import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms

import os
import json
from PIL import Image, ImageDraw


model = models.__dict__["resnet18"]()
model = model.cuda(0)

print("model")
print(model)