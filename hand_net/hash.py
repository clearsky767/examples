import hashlib
import os

#string MD5
def GetStrMd5(src):
    m = hashlib.md5()
    m.update(src)
    md5 = m.hexdigest()
    return md5
 
#big file MD5
def GetFileMd5(filename):
    if not os.path.isfile(filename):
        return
    m = hashlib.md5()
    f = open(filename,'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        m.update(b)
    f.close()
    md5 = m.hexdigest()
    return md5
 
def CalcSha1(filepath):
    with open(filepath,'rb') as f:
        m = hashlib.sha1()
        m.update(f.read())
        sha1 = m.hexdigest()
        print(sha1)
        return sha1
 
def CalcMD5(filepath):
    with open(filepath,'rb') as f:
        m = hashlib.md5()
        m.update(f.read())
        md5 = m.hexdigest()
        print(md5)
        return md5

def get_imglist(path):
    imglist = []
    for f in os.listdir(path):
        if os.path.splitext(f)[1] == ".jpg" or os.path.splitext(f)[1] == ".JPG":
            imglist.append(f)
    return imglist

def main():
    print("start")
    md5_ls = []
    imglist = get_imglist("data")
    for img in imglist:
        img_path = os.path.join(os.path.realpath("data/"),img)
        md5 = GetFileMd5(img_path)
        if md5 is None:
            print("md5 is None!")
        else:
            if md5 in md5_ls:
                print("md5 is in md5_ls")
                print(img_path)
            else:
                md5_ls.append(md5)

if __name__ == "__main__":
    main()