import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import argparse

parser = argparse.ArgumentParser(description='txt2imgs')
parser.add_argument('--path', default='test', type=str, metavar='PATH',help='txts path')
args = parser.parse_args()

fs = 125 #sampling frequency
T = 1.0/fs #sampling period
L = 15*fs #length of signal
t = np.arange(0,L/fs,T) #time vector

def ReadTxt(filename):
    fo = open(filename, "r")
    fl = fo.readlines()
    fo.close()
    return [float(e.strip("\n")) for e in fl]

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return

def GenImgFunc(data,i,filepath,L,filename):
    idx_start = i*L
    plt.plot(data[idx_start+125:idx_start+L],linewidth=0.5,color="black")
    plt.xlim(0,L)
    plt.ylim(-1500,1500)
    plt.axis('off')
    plt.savefig('./{}/{}_{}.png'.format(filepath,filename,i))
    plt.clf()

def GenImgs(data,L,filepath,filename):
    data_length = len(data)
    img_count = data_length/L
    plt.figure()
    for i in range(img_count):
        GenImgFunc(data,i,filepath,L,filename)
    plt.close()
    return

def main():
    start_tm = time.time()
    print("now start read txt files!")
    tm = time.time()
    mkdir("./imgs")
    filepath = args.path
    txtlist = [txtfile for txtfile in os.listdir(filepath) if os.path.splitext(txtfile)[1] == '.txt']
    for txt in txtlist:
        txt_path = os.path.join(os.path.realpath('.'), filepath, txt)
        data = ReadTxt(txt_path)
        filename = os.path.splitext(txt)[0]
        GenImgs(data,L,"imgs",filename)

    print("time is {}".format(time.time()-tm))
    print("total time is {}".format(time.time()-start_tm))

if __name__ == '__main__':
    main()
