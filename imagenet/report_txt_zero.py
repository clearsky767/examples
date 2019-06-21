#coding=utf-8
import time
import os
import argparse

parser = argparse.ArgumentParser(description='log report')
parser.add_argument('--path', default='log/BB0201C001DD8465.log', type=str, metavar='PATH',help='log path')
args = parser.parse_args()


def read_ls_txt(filename):
    fo = open(filename, "r")
    fl = fo.readlines()
    fo.close()
    return [int(e.strip("\n")) for e in fl]

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
    fo = open(filename, "a+")
    fl = fo.write(data)
    fo.close()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def get_loglist(path):
    loglist = []
    for f in os.listdir(path):
        if os.path.splitext(f)[1] == ".log":
            loglist.append(f)
    return loglist

def main():
    print("start!")
    log_path = args.path
    log = read_txt(log_path)
    log = [int(e) for e in log.split(",")]
    
    for i,v in enumerate(log):
        if v == 0:
            print(i)
   
    print("end!")

if __name__ == '__main__':
    main()