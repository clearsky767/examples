import cv2
import numpy as np
from matplotlib import pyplot as plt


imgFile = 'hand_test/1 (7).jpg'

img = cv2.imread(imgFile)
rows,cols,channels = img.shape
gamma = 0.95
 
for r in range(rows):
    for c in range(cols):
        
        # get values of blue, green, red     
        B = img.item(r,c,0)
        G = img.item(r,c,1)
        R = img.item(r,c,2)
        
        # gamma correction
        B = int(B ** gamma)  
        G = int(G ** gamma) 
        R = int(R ** gamma)
        
        # set values of blue, green, red
        img.itemset((r,c,0), B)
        img.itemset((r,c,1), G)
        img.itemset((r,c,2), R)

# convert color space from rgb to ycbcr
imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
 
# convert color space from bgr to rgb                    
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# prepare an empty image space
imgSkin = np.zeros(img.shape, np.uint8)
# copy original image
imgSkin = img.copy()

# define variables for skin rules
 
Wcb = 46.97
Wcr = 38.76
 
WHCb = 14
WHCr = 10
WLCb = 23
WLCr = 20
 
Ymin = 16
Ymax = 235
 
Kl = 125
Kh = 188
 
WCb = 0
WCr = 0
 
CbCenter = 0
CrCenter = 0

for r in range(rows):
    for c in range(cols):
        
        # non-skin area if skin equals 0, skin area otherwise        
        skin = 0

        # color space transformation
        
        # get values from ycbcr color space     
        Y = imgYcc.item(r,c,0)
        Cr = imgYcc.item(r,c,1)
        Cb = imgYcc.item(r,c,2)

        if Y < Kl:
            WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin)
            WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin)
            
            CrCenter = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
            CbCenter = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)
            
        elif Y > Kh:
            WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh)
            WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh)
 
            CrCenter = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
            CbCenter = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh) 
        
        if Y < Kl or Y > Kh:
            Cr = (Cr - CrCenter) * Wcr / WCr + 154
            Cb = (Cb - CbCenter) * Wcb / WCb + 108

        # skin color detection
        
        if Cb > 77 and Cb < 127 and Cr > 133 and Cr < 173:
            skin = 1
            # print 'Skin detected!'
        
        if 0 == skin:
            imgSkin.itemset((r,c,0),0)
            imgSkin.itemset((r,c,1),0)
            imgSkin.itemset((r,c,2),0)
 
# display original image and skin image
plt.subplot(1,2,1), plt.imshow(img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(imgSkin), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
plt.show()

print('Goodbye!')