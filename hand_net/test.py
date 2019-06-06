import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import os
import json
import time
import cv2
import shutil
import argparse
import numpy as np
from PIL import Image, ImageDraw
from resnet import resnet18
from roi_pooling import roi_pooling,adaptive_max_pool


def show(img,points):
    draw = ImageDraw.Draw(img)
    pts = [tuple(point )for point in points]
    draw.point(pts, fill = (255, 0, 0))
    draw.text((100,100), "hand", fill=(0,255,0))
    img.show()

def show2(img,target_tensor,w,h):
    target = target_tensor.cpu().detach().numpy()
    target = target.tolist()
    target = target[0]
    points = []
    for i in range(0,len(target),2):
        p_w = target[i]*w
        p_h = target[i+1]*h
        points.append([p_w,p_h])
    print(points)
    show(img,points)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

img_transforms = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    normalize,
    ])

gpu = 0
checkpoint_path = "checkpoint_99.pth"

torch.manual_seed(1)
if gpu is not None:
    torch.cuda.manual_seed(1)

model = resnet18()
print("loaded model!")

if gpu is not None:
    model = model.cuda(gpu)
    print("model to gpu")
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint {}".format(checkpoint_path))

class HandNet(nn.Module):
    def __init__(self):
        super(HandNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
        )
        self.fc = nn.Linear(512, 2)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 512)
        output = self.fc(output)
        output = self.sig(output)
        return output

model2 = HandNet()
if gpu is not None:
    model2 = model2.cuda(gpu)

def main():
    print("model")
    img = "test/2.jpg"
    #img_path,img_path2 = generate_edges(img)
    print(model)
    img = Image.open(img).convert('RGB')
    img_tensor = img_transforms(img)
    input = torch.unsqueeze(img_tensor,0)
    if gpu is not None:
        input = input.cuda(gpu, non_blocking=True)
    x1,x2,x3,f_map = model(input)
    print(x1.shape)
    print(f_map.shape)
    output2 = model2(f_map)
    print(output2.shape)

    out = adaptive_max_pool(f_map,(4,4))
    print(out.shape)

    #img = transforms.ToPILImage()(img_tensor)
    #w = 320
    #h = 320
    #img = img.resize((w, h),Image.ANTIALIAS)
    #show2(img,output,w,h)

if __name__ == '__main__':
    main()
