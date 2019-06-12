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
from hand_net import create_model


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

model = create_model(gpu)
print("loaded model!")

if gpu is not None:
    model = model.cuda(gpu)
    print("model to gpu")
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint {}".format(checkpoint_path))

def generate_edges(img):
    img_path = os.path.join(os.path.realpath("test/"),img)
    img_cv = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray,60,65,apertureSize = 3)
    img_path2 = os.path.join(os.path.realpath("test/edges/"),img)
    cv2.imwrite(img_path2,img_edges)
    return img_path,img_path2

def main():
    print("model")
    img = "test/2.jpg"
    #img_path,img_path2 = generate_edges(img)
    #print(model)
    img = Image.open(img).convert('RGB')
    img_tensor = img_transforms(img)
    input = torch.unsqueeze(img_tensor,0)
    if gpu is not None:
        input = input.cuda(gpu, non_blocking=True)
    output = model(input,target = None)
    p1,p2,p3,p4,p5 = output
    print(p1.shape)
    print(p2.shape)
    print(p3.shape)
    print(p4.shape)
    print(p5.shape)
    p = torch.cat((p1,p2),dim = -1)
    p = torch.cat((p,p3),dim = -1)
    p = torch.cat((p,p4),dim = -1)
    p = torch.cat((p,p5),dim = -1)
    print(p.shape)
    #img = transforms.ToPILImage()(img_tensor)
    w = 320
    h = 320
    img = img.resize((w, h),Image.ANTIALIAS)
    show2(img,p,w,h)

if __name__ == '__main__':
    main()
