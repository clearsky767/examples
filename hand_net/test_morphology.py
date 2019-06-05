import cv2


img = cv2.imread("C:/Users/DELL/Desktop/img4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gauss = cv2.GaussianBlur(gray, (3, 3), 1)
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
erode = cv2.erode(gauss, element)
dilate = cv2.dilate(gauss, element)
open = cv2.morphologyEx(gauss, cv2.MORPH_OPEN, element)
close = cv2.morphologyEx(gauss, cv2.MORPH_CLOSE, element)
morphological_gradient = cv2.morphologyEx(gauss, cv2.MORPH_GRADIENT, element)
top_hat = cv2.morphologyEx(gauss, cv2.MORPH_TOPHAT, element)
black_hat = cv2.morphologyEx(gauss, cv2.MORPH_BLACKHAT, element)

cv2.imshow("src", gauss)
cv2.imshow("dilate", dilate)
cv2.imshow("erode", erode)
cv2.imshow("open", open)
cv2.imshow("close", close)
cv2.imshow("morphological_gradient", morphological_gradient)
cv2.imshow("top_hat", top_hat)
cv2.imshow("black_hat", black_hat)

cv2.waitKey(0)
cv2.destroyAllWindows()