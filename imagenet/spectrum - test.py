# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:21:46 2019
@author: DRivch
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import pandas as pd
import os
#
#Filename1 = 'noice/bb0201b9001a066a_2019-3-30-11-50_2019-3-30-11-55.ch1.csv'
#Filename2 = 'noice/bb0201b9001a066a_2019-3-30-11-50_2019-3-30-11-55.ch2.csv'
#两信号做对比
Filename1 = 'in/zhengchangbcgxunlian.csv'
Filename2 = 'in/zhengchangbcgxunlian.csv'

#Filename1 = 'in_far/bb0201b9001a066a_2019-3-26-21-38_2019-3-26-21-41.ch1.csv'
#Filename2 = 'in_far/bb0201b9001a066a_2019-3-26-21-38_2019-3-26-21-41.ch2.csv'

def entropy(c): #熵值计算步骤
    c=(c - min(c))
    c=c/max(abs(c))
    result=-1
#    print(max(c))
    if(len(c)>0):
        result=0
    for x in c:
        if x == 0 or x == 1:
            result+=0
        else:
            result+=(-x)*np.log2(x)
    return result

def mkdir(path):
 
	folder = os.path.exists(path)

	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
	else:return

def Filter(bcg, fs = 125):
    Nyquist = fs/2
    bcgRawMean = sum(bcg)/len(bcg)
    bcg = [elem-bcgRawMean for elem in bcg]

    #高通滤波 0.6Hz
    b, a = signal.butter(2, 0.1/Nyquist ,'high')
    bcg = signal.filtfilt(b, a, bcg)
    #低通滤波 60Hz(<fs/2)
#    b2, a2 = signal.butter(2, 60/Nyquist )
#    bcg = signal.filtfilt(b2, a2, bcg)
    
    return bcg

def GetEnt(data , start ,end, steplengh): 
    xf_ent = []
    max_mean = []
    for i in range(start,end,steplengh): #快速傅里叶变换求频谱
        if (end-i) < 500:break
        xf1 = np.fft.rfft(data[i:i+steplengh])/steplengh
        out1 = np.abs(xf1[int(len(xf1)/100):])
        out1 = out1/max(out1)
        max_mean.append(max(out1[100:]))  #探测有无高频噪声
        ent1 = entropy(out1) #熵值计算
        xf_ent.append(ent1)

    plt.figure()
    plt.plot(out1)
    return xf_ent,max_mean
        

sampling_rate = 125 #采样频率

posin1 = pd.read_csv(Filename1) 
pos1 = [elem for elem in posin1.iloc[:,1]]

posin2 = pd.read_csv(Filename2) 
pos2 = [elem for elem in posin2.iloc[:,1]]
#滤波
pos1 = Filter(pos1)
pos2 = Filter(pos2)

plt.figure()
plt.plot(pos1)
plt.figure()
plt.plot(pos2)

steplengh = 500 #步长4s
start , end = 0,len(pos2)


#命名和建立文件
#
xf1_ent,max_mean1 = GetEnt(pos1 , start ,end, steplengh)
xf2_ent,max_mean2 = GetEnt(pos2 , start ,end, steplengh)

xf_in = 0
xf_leave = 0
for i in range(len(xf1_ent)-1):
    if  max(xf1_ent[i:i+2])<30 and max(max_mean1[i:i+2])<0.05 and max(xf2_ent[i:i+2])<30 and max(max_mean2[i:i+2])<0.05:
#        print('in')
        xf_in += 1
    else:
#        print('leave')
        xf_leave += 1
print('in:'+str(xf_in))
print('leave:'+str(xf_leave))
#doc.close()

