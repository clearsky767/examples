import numpy as np
import time
import os
import PIL.Image as Image
import argparse

import torch
import torchvision.transforms as transforms
import torchvision.models as models

parser = argparse.ArgumentParser(description='WaveGo')
parser.add_argument('--csvfile', default='BB0001B900420D5F_1.csv', type=str, metavar='PATH',help='path to dataset')
parser.add_argument('--batchsize', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',help='path to checkpoint')
parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')

args = parser.parse_args()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return

def LoadImgs(filepath,batchsize):
    filename = "./log/{}.list".format(filepath)
    fo = open(filename, "r")
    filelist = fo.readlines()
    fo.close()
    size = 0
    filelength = len(filelist)
    batchs = []
    onebatch = []
    for filename in filelist:
        img = Image.open(filename.strip("\n")).convert("RGB")
        imgtensor = transforms(img)

        size = size + 1
        onebatch.append(imgtensor)
        if size%batchsize != 0:
            if size >= filelength:
                batchs.append(torch.stack(onebatch ,dim = 0))
                onebatch = []
        else:
            batchs.append(torch.stack(onebatch ,dim = 0))
            onebatch = []
    return batchs

def main():
    start_tm = time.time()
    mkdir('./log')
    print("now start ready data to wavenet!")
    tm = time.time()
    filename = os.path.basename(args.csvfile)
    filename = filename.split(".")[0]
    batchs = LoadImgs(filename,args.batchsize)
    print("time is {}".format(time.time()-tm))
    
    print("now wavenet start!")
    tm = time.time()
    model = models.__dict__["resnet18"]()
    print("loaded model!")
    if args.gpu is not None:
        model = model.cuda(args.gpu)
        print("model to gpu")
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("loaded checkpoint '{}'".format(args.checkpoint))
    
    ret = []
    model.eval()
    with torch.no_grad():
        for input in batchs:
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            output = model(input)
            _, pred = output.topk(3, 1, True, True)
            for i in range(0,pred.size(0)):
                ret.append(pred[i][0].cpu().item())
    print("time is {}".format(time.time()-tm))

    print(ret)
    ret = [max(ret[2*i],ret[2*i+1]) for i in range(len(ret)/2)]
    filename = os.path.basename(args.csvfile)
    filename = filename.split(".")[0]
    filename = "./log/{}.log".format(filename)
    ret = [str(s) for s in ret]
    sep = ','
    fo = open(filename, "w+")
    fo.write(sep.join(ret))
    fo.close()

    print("total time is {}".format(time.time()-start_tm))

if __name__ == '__main__':
    main()