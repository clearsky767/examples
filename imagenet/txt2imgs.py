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
L = 30*fs #length of signal
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

def Filter(data, fs = 125):
    #bandpass 0.6Hz~25Hz
    b, a = signal.butter(2, [40.0/60.0/fs*2,1500.0/60.0/fs*2] ,'bandpass')
    filteddata = signal.filtfilt(b, a, data)
    #filteddata = signal.lfilter(b, a, data)
    return filteddata

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
    txtlist = [os.path.join(os.path.realpath('.'), filepath, txtfile) for txtfile in os.listdir(filepath) if os.path.splitext(txtfile)[1] == '.txt']
    for txt in txtlist:
        data = ReadTxt(txt)
        #data = Filter(data)
        filename = os.path.basename(txt)
        filename = filename.split(".")[0]
        GenImgs(data,L/2,"imgs",filename)

    print("time is {}".format(time.time()-tm))
    print("total time is {}".format(time.time()-start_tm))

if __name__ == '__main__':
    main()