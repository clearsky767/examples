import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import os
import json
import time
import shutil
import argparse
import numpy as np
from PIL import Image, ImageDraw
from unet.unet_model import UNet

def show_heats(img,heats):
    """
    heats numpy ndarray shape (31,320,320)
    """
    print("show heats")
    print(heats.shape)
    c = heats.shape[0]
    points = []
    for i in range(c):
        heat = heats[i]
        heat_max = np.max(heat)
        print(heat_max)
        heat_coor = np.where(heat == heat_max)
        p_w = heat_coor[0][0]
        p_h = heat_coor[1][0]
        points.append([p_w,p_h])
    print(points)
    show(img,points)

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
checkpoint_path = "3/checkpoint_80.pth"

torch.manual_seed(1)
if gpu is not None:
    torch.cuda.manual_seed(1)

model = UNet(3,31)
print("loaded model!")

if gpu is not None:
    model = model.cuda(gpu)
    print("model to gpu")
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint {}".format(checkpoint_path))


def main():
    print("model")
    print(model)
    img = Image.open("test/zjf_7.jpg").convert('RGB')
    img_tensor = img_transforms(img)
    input = torch.unsqueeze(img_tensor,0)
    if gpu is not None:
        input = input.cuda(gpu, non_blocking=True)
    output = model(input)
    print(output.shape)

    img = img.resize((320, 320),Image.ANTIALIAS)
    output = output.cpu().detach().numpy()
    show_heats(img,output[0])

if __name__ == '__main__':
    main()
