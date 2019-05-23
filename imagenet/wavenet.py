import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import PIL.Image as Image
import argparse
import shutil
import multiprocessing

import torch
import torchvision.transforms as transforms
import torchvision.models as models

parser = argparse.ArgumentParser(description='WaveNet')
parser.add_argument('--csvfile', default='BB0001B900420D5F_1.csv', type=str, metavar='PATH',help='path to dataset')
parser.add_argument('--batchsize', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',help='path to checkpoint')
parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')

args = parser.parse_args()

fs = 125 #sampling frequency
T = 1.0/fs #sampling period
L = 15*fs #length of signal
t = np.arange(0,L/fs,T) #time vector

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

def LoadImgs(filelist,batchsize):
    size = 0
    filelength = len(filelist)
    batchs = []
    onebatch = []
    for filename in filelist:
        img = Image.open(filename).convert("RGB")
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
    #print(batchs[-1].size())
    return batchs

def Filter(data, fs = 125):
    #bandpass 0.6Hz~25Hz
    b, a = signal.butter(2, [40.0/60.0/fs*2,1500.0/60.0/fs*2] ,'bandpass')
    filteddata = signal.filtfilt(b, a, data)
    #filteddata = signal.lfilter(b, a, data)
    return filteddata

def GenImgFunc(data,i):
    print multiprocessing.current_process().name  + "   msg %d"%(i)
    plt.figure()
    idx_start = i*L
    plt.plot(data[idx_start+125:idx_start+L],linewidth=0.5,color="black")
    plt.xlim(0,L)
    plt.ylim(-1500,1500)
    plt.axis('off')
    plt.savefig('./img/img_{}.png'.format(i))
    plt.clf()
    plt.close()

def GenImgs(data,L):
    data_length = len(data)
    img_count = data_length/L
    imglist = []
    pool = multiprocessing.Pool(processes=4)
    for i in range(img_count):
        imglist.append('./img/img_{}.png'.format(i))
        pool.apply_async(GenImgFunc,(data,i))
    pool.close()
    pool.join()
    return imglist

def main():
    try:
        start_tm = time.time()
        print("now start read csv and filter data!")
        tm = time.time()
        mkdir('./img')
        mkdir('./log')
        filedata = pd.read_csv(args.csvfile)
        wavedata = [elem for elem in filedata.iloc[:,1]]
        filterdata = Filter(wavedata)
        print("time is {}".format(time.time()-tm))

        print("now start generate wave images!")
        tm = time.time()
        imagelist = GenImgs(filterdata,L)
        print("time is {}".format(time.time()-tm))
        #imagelist = ['./img/img_1.png','./img/img_2.png','./img/img_3.png','./img/img_4.png']

        print("now start ready data to wavenet!")
        tm = time.time()
        batchs = LoadImgs(imagelist,args.batchsize)
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
        filename = os.path.basename(args.csvfile)
        filename = filename.split(".")[0]
        filename = "./log/{}.txt".format(filename)
        ret = [str(s) for s in ret]
        sep = ','
        fo = open(filename, "w+")
        fo.write(sep.join(ret))
        fo.close()
    finally:
        print("now start delete imgs!")
        tm = time.time()
        #shutil.rmtree("./img")
        print("time is {}".format(time.time()-tm))

        print("total time is {}".format(time.time()-start_tm))

if __name__ == '__main__':
    main()
