import os
from scipy import signal

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

def ReadTxt(filename):
    fo = open(filename, "r")
    fl = fo.readlines()
    fo.close()
    return [float(e.strip("\n")) for e in fl]

data = ReadTxt("/home/alex/github/examples/imagenet/test/a0001-0.txt")
print(data[0:20])

filterdata = Filter(data)
print(filterdata[0:20])
filterdata2 = Filter(filterdata)
print(filterdata2[0:20])