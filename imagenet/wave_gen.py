import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import argparse

parser = argparse.ArgumentParser(description='WaveGen')
parser.add_argument('--csvfile', default='BB0001B900420D5F_1.csv', type=str, metavar='PATH',help='path to dataset')
args = parser.parse_args()

fs = 125 #sampling frequency
T = 1.0/fs #sampling period
L = 30*fs #length of signal
t = np.arange(0,L/fs,T) #time vector

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

def GenImgFunc(data,i,filepath):
    idx_start = i*L
    plt.plot(data[idx_start+125:idx_start+L],linewidth=0.5,color="black")
    plt.xlim(0,3750)
    plt.ylim(-30,30)
    plt.axis('off')
    plt.savefig('./{}/img_{}.png'.format(filepath,i))
    plt.clf()

def GenImgs(data,L,filepath):
    data_length = len(data)
    img_count = data_length/L
    imglist = []
    plt.figure()
    filename = "./log/{}.list".format(filepath)
    fo = open(filename, "w+")
    for i in range(img_count):
        fo.writelines('./{}/img_{}.png\n'.format(filepath,i))
        GenImgFunc(data,i,filepath)
    plt.close()
    fo.close()
    return imglist

def main():
    start_tm = time.time()
    print("now start read csv and filter data!")
    tm = time.time()
    filename = os.path.basename(args.csvfile)
    filepath = filename.split(".")[0]
    mkdir(filepath)
    mkdir("./log")
    filedata = pd.read_csv(args.csvfile)
    wavedata = [elem for elem in filedata.iloc[:,1]]
    filterdata = Filter(wavedata)
    print("time is {}".format(time.time()-tm))

    print("now start generate wave images!")
    tm = time.time()
    imagelist = GenImgs(filterdata,L,filepath)
    print("time is {}".format(time.time()-tm))
    #imagelist = ['./img/img_1.png','./img/img_2.png','./img/img_3.png','./img/img_4.png']
    print("total time is {}".format(time.time()-start_tm))

if __name__ == '__main__':
    main()
