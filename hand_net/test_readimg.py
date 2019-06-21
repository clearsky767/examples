# -*- coding: UTF-8 -*-


"""
https://blog.csdn.net/m0_37673307/article/details/81271155
https://www.jianshu.com/p/dd08418c306f
https://www.cnblogs.com/ocean1100/p/9494640.html
"""

import cv2


#opencv坐标系  原点在左上角
#img_cv is numpy ndarray uint8 0-255  shape is (H,W,C)  BGR
img_cv = cv2.imread("data/aip_2504.png")
print("cv2 ",img_cv.shape)
#way 1           convert BGR to RGB
b,g,r = cv2.split(img_cv)
img_cv = cv2.merge([r,g,b])
#way 2           convert BGR to RGB
#img_cv = img_cv[:,:,::-1]
#way 3           convert BGR to RGB
#img = cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)

#show
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
cv2.imshow("OpenCV",img)
cv2.waitKey()
cv2.imwrite('new_image.jpg',img)

gray = cv2.imread('image.jpg',cv2.IMREAD_GRAYSCALE)  #灰度图读取
image = cv2.resize(img,(100,200),interpolation=cv2.INTER_LINEAR) #resize

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  #BGR转RGB
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #BGR转灰度图

import matplotlib.pyplot as plt

#img_plt is numpy ndarray uint8 0-255  shape is (H,W,C)  RGB
img_plt = plt.imread("data/aip_2504.png")
print("plt ",img_plt.shape)



from PIL import Image
# PIL 坐标系  原点在左上角
# PIL bug iphone image  =========>>>>     png width height not same as jpg

#img_pil is PIL.Image.Image obj, img_np is ndarray uint8 0-255  shape is (H,W,C) RGB
img_pil = Image.open("data/aip_2504.png").convert('RGB')
print(type(img_pil))  #<class 'PIL.Image.Image'>
img_np = np.array(img_pil)
print(type(img_np))  #<class 'numpy.ndarray'>
print('PIL',img_np.shape)  #(H,W,C)







