import time
import os
import shutil
import random

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return

def GetFileList(path):
    filelist = [os.path.join(os.path.realpath('.'), path, file) for file in os.listdir(path) if os.path.splitext(file)[1] == '.png']
    return filelist

def Move2Path(file,path):
    filename = os.path.basename(file)
    full_path = os.path.join(path, filename)
    shutil.move(file,full_path)

def main():
    start_tm = time.time()
    print("now start move files!")
    mkdir("./zjfval")
    filelist = GetFileList("data_n")
    for file in filelist:
        rd = random.randint(1,10)
        if rd < 4:
            Move2Path(file,"./zjfval")
    print("total time is {}".format(time.time()-start_tm))

if __name__ == '__main__':
    main()
