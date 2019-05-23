import time
import os
import shutil

def GetFileList(path):
    filelist = [os.path.join(os.path.realpath('.'), path, file) for file in os.listdir(path) if os.path.splitext(file)[1] == '.png']
    return filelist

def main():
    start_tm = time.time()
    print("now start move files!")
    filelist = GetFileList("zjf")
    i = 0
    for file in filelist:
        os.rename(file,"./zjf/zjf_{}.png".format(i))
        i += 1
    print("total time is {}".format(time.time()-start_tm))

if __name__ == '__main__':
    main()
