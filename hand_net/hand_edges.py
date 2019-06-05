import cv2
image = cv2.imread("hand_test\\1 (10).jpg")
image = cv2.resize(image,(512,512),interpolation=cv2.INTER_CUBIC)
cv2.imshow("palm",image) #to view the palm in python
cv2.waitKey(0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,50,52,apertureSize = 3)
cv2.imshow("edges",edges)
cv2.waitKey(0)

edges = cv2.bitwise_not(edges)
cv2.imshow("change black and white",edges)
cv2.waitKey(0)

cv2.imwrite("palmlines.jpg", edges)
palmlines = cv2.imread("palmlines.jpg")
img = cv2.addWeighted(palmlines, 0.3, image, 0.7, 0)
cv2.imshow("lines in palm", img)
cv2.waitKey(0)