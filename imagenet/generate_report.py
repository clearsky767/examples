#coding=utf-8
import time
import os
import argparse

parser = argparse.ArgumentParser(description='log report')
parser.add_argument('--path', default='.', type=str, metavar='PATH',help='log path')
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
    loglist = get_loglist(log_path)
    print(loglist)
    out_file = os.path.join(os.path.realpath(log_path),"report.txt")
    tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    for log in loglist:
        c = read_txt(os.path.join(os.path.realpath(log_path),log))
        c = c.split(",")
        c = [int(e) for e in c]
        zero_ls = []
        zero_count = 0
        count = len(c)
        last_4 = 0
        last_5 = 0
        last_6 = 0
        last_7 = 0
        last_8 = 0
        last_9 = 0
        last_10 = 0
        hour = count*15/3600
        minute = (count*15%3600)/60
        for v in c:
            if v == 0:
                zero_count += 1
                zero_ls.append(v)
            else:
                if len(zero_ls)>0:
                    if len(zero_ls) == 4:
                        last_4 += 1
                    if len(zero_ls) == 5:
                        last_5 += 1
                    if len(zero_ls) == 6:
                        last_6 += 1
                    if len(zero_ls) == 7:
                        last_7 += 1
                    if len(zero_ls) == 8:
                        last_8 += 1
                    zero_ls = []
        report = "文件:" + log + "  时间: " + tm + "\n"\
             + "总睡眠时间:" + str(hour) + " 小时 " + str(minute) + " 分钟  "\
             + "房颤比非房颤 0/1 = " + str(float(zero_count)/float(count)) + "(" + str(zero_count) + "/" + str(count) + ")\n"\
             + "持续一分钟次数: " + str(last_4) + "\n"\
             + "持续一分15秒次数: " + str(last_5) + "\n"\
             + "持续一分30秒次数: " + str(last_6) + "\n"\
             + "持续一分45秒次数: " + str(last_7) + "\n"\
             + "持续两分钟次数: " + str(last_8) + "\n"\
             + "持续两分15秒次数: " + str(last_9) + "\n"\
             + "持续两分30秒次数: " + str(last_10) + "\n\n\n"
        write_txt(out_file, report)
    print("end!")

if __name__ == '__main__':
    main()