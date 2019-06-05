import cv2
import numpy as np

image = cv2.imread("hand_test/1 (7).jpg")
image = cv2.resize(image,(512,512),interpolation=cv2.INTER_CUBIC)
cv2.imshow("palm",image) #to view the palm in python
cv2.waitKey(0)

colors = [[0,70,70],[100,255,255]] # [b,g,r]

lower = np.array(colors[0],dtype="uint")
upper = np.array(colors[1],dtype="uint")

mask = cv2.inRange(image, lower, upper)
out = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("mask", np.hstack([image, out]))
cv2.waitKey(0)