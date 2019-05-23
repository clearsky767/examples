import time
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='plot txt')
parser.add_argument('--path1', default='loss_test.log', type=str, metavar='PATH',help='txt path')
parser.add_argument('--path2', default='loss_train.log', type=str, metavar='PATH',help='txt path')
args = parser.parse_args()

def read_ls_txt(filename):
    fo = open(filename, "r")
    fl = fo.readlines()
    fo.close()
    return [float(e.strip("\n"))*1000 for e in fl]

#data is str list
def write_ls_txt(filename,data):
    fo = open(filename, "w+")
    sep = "\n"
    data = [str(e) for e in data]
    fl = fo.write(sep.join(data))
    fo.close()

def read_txt(filename):
    fo = open(filename, "r")
    fl = fo.read()
    fo.close()
    return fl

def write_txt(filename,data):
    fo = open(filename, "w+")
    fl = fo.write(data)
    fo.close()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def plot(data1,data2):
    fig = plt.figure()
    plt.plot(data1,linewidth=0.5,color="black")
    plt.plot(data2,linewidth=0.5,color="red")
    #plt.xlim(0,max(len(data1),len(data2)))
    #plt.ylim(0,100)
    plt.show()

def main():
    print("start!")
    data1 = read_ls_txt(args.path1)
    data2 = read_ls_txt(args.path2)
    plot(data1,data2)
    print("end!")

if __name__ == '__main__':
    main()
