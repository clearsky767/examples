import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import argparse

parser = argparse.ArgumentParser(description='txt2imgs')
parser.add_argument('--path', default='test2', type=str, metavar='PATH',help='txts path')
args = parser.parse_args()

def ReadCsv(filename):
    filedata = pd.read_csv(filename)
    wavedata = [elem for elem in filedata.iloc[:,1]]
    return wavedata

def WriteTxt(filename,data):
    fo = open(filename, "w+")
    sep = "\n"
    fl = fo.write(sep.join(data))
    fo.close()

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

def main():
    start_tm = time.time()
    print("now start read csv files!")
    tm = time.time()
    filepath = args.path
    csvlist = [os.path.join(os.path.realpath('.'), filepath, file) for file in os.listdir(filepath) if os.path.splitext(file)[1] == '.csv']
    for csv in csvlist:
        data = ReadCsv(csv)
        data = Filter(data)
        filename = os.path.basename(csv)
        filename = filename.split(".")[0]
        filename = os.path.join(os.path.realpath('.'), filepath, "{}.txt".format(filename))
        data = [str(s) for s in data]
        WriteTxt(filename,data)

    print("time is {}".format(time.time()-tm))
    print("total time is {}".format(time.time()-start_tm))

if __name__ == '__main__':
    main()
