import os
import cv2
import numpy as np

def get_imglist(path):
    imglist = []
    for f in os.listdir(path):
        if os.path.splitext(f)[1] == ".jpg" or os.path.splitext(f)[1] == ".JPG":
            imglist.append(f)
    return imglist

def cut(img):
    px = img[150,200]

    blue = img[150,200,0]
    green = img[150,200,1]
    red = img[150,200,2]

    img[150,200] = [0,0,0]

    blue = img.item(100,200,0)
    green = img.item(100,200,1)
    red = img.item(100,200,2)

    img.itemset((100,200,1),255)
    green = img.item(100,200,1)
    rows,cols,channels = img.shape
 
    imgSkin = np.zeros(img.shape, np.uint8)
    imgSkin = img.copy()
 
    for r in range(rows):
        for c in range(cols):
            B = img.item(r,c,0)
            G = img.item(r,c,1)
            R = img.item(r,c,2)
            skin = 0
            if (abs(R - G) > 15) and (R > G) and (R > B):
                if (R > 95) and (G > 40) and (B > 20) and (max(R,G,B) - min(R,G,B) > 15):               
                    skin = 1
                elif (R > 220) and (G > 210) and (B > 170):
                    skin = 1
            if 0 == skin:
                imgSkin.itemset((r,c,0),0)
                imgSkin.itemset((r,c,1),0)            
                imgSkin.itemset((r,c,2),0)

    imgSkin = cv2.cvtColor(imgSkin, cv2.COLOR_BGR2RGB)
    return imgSkin

def main():
    print("start")
    imglist = get_imglist("data")
    for img in imglist:
        img_path = os.path.join(os.path.realpath("data/"),img)
        img_cv = cv2.imread(img_path)
        imgSkin = cut(img_cv)
        img_path2 = os.path.join(os.path.realpath("skin/"),img)
        cv2.imwrite(img_path2,imgSkin)

if __name__ == "__main__":
    main()
