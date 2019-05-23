import time
import os

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

def get_imglist(path):
    jsonfile = None
    imglist = []
    for f in os.listdir(path):
        if os.path.splitext(f)[1] == ".jpg" or os.path.splitext(f)[1] == ".JPG":
            imglist.append(f)
        if os.path.splitext(f)[1] == ".png" or os.path.splitext(f)[1] == ".jpeg":
            print("find png or jpeg")
        if os.path.splitext(f)[1] == ".json":
            jsonfile = f
    return imglist,jsonfile

def get_imglist2(path):
    imglist = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1] == '.jpg':
                #imglist.append(os.path.join(root, f))
                imglist.append(f)
    return imglist


def main():
    print("please input img prefix")
    prefix = input("prefix: ")
    print("prefix is {}".format(prefix))
    idx = 0
    print("start!")
    imglist,jsonfile = get_imglist(".")
    json = read_txt(jsonfile)
    for img in imglist:
        idx += 1
        rp_name = "{}{}.jpg".format(prefix, idx)
        oldfile = os.path.join(os.path.realpath("."),img)
        newfile = os.path.join(os.path.realpath("."),rp_name)
        os.rename(oldfile, newfile)
        json = json.replace(img,rp_name,2)
    print("end!")
    write_txt("out.json", json)

if __name__ == '__main__':
    main()