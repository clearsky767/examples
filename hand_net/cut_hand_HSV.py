import cv2
import numpy as np
from matplotlib import pyplot as plt


imgFile = 'hand_test/1 (7).jpg'

# load an original image
img = cv2.imread(imgFile)

rows,cols,channels = img.shape
 
# convert color space from bgr to rgb        
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# prepare an empty image space
imgSkin = np.zeros(img.shape, np.uint8)
# copy original image
imgSkin = img.copy()
 
# convert color space from rgb to hsv
imgHsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
 
for r in range(rows):
    for c in range(cols):
        # get values of hue, saturation and value
        # standard -- h range: [0,360]; s range: [0,1]; v range: [0,255]
        # opencv -- h range: [0,180]; s range: [0,255]; v range: [0,255]
        H = imgHsv.item(r,c,0)
        S = imgHsv.item(r,c,1)
        V = imgHsv.item(r,c,2)
        
        # non-skin area if skin equals 0, skin area otherwise        
        skin = 0
                
        if ((H >= 0) and (H <= 25 / 2)) or ((H >= 335 / 2) and (H <= 360 / 2)):
            if ((S >= 0.2 * 255) and (S <= 0.6 * 255)) and (V >= 0.4 * 255):
                skin = 1

        if 0 == skin:
            imgSkin.itemset((r,c,0),0)
            imgSkin.itemset((r,c,1),0)
            imgSkin.itemset((r,c,2),0)
 
# display original image and skin image
plt.subplot(1,2,1), plt.imshow(img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(imgSkin), plt.title('HSV Skin Image'), plt.xticks([]), plt.yticks([])
plt.show()

print('Goodbye!')