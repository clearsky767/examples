import cv2 as cv
import numpy as np

def shape_image():
    target = cv.imread("F:\\WeiYi\\6.jpg")
    src = cv.imread("F:\\WeiYi\\7.jpg")

    targetGray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    srcGray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    kernel = np.ones((2, 2),np.uint8)
    targetErosion = cv.erode(targetGray, kernel, iterations = 1)
    srcErosion = cv.erode(srcGray, kernel, iterations = 1)

    thresh = 100
    targetEdges = cv.Canny(targetErosion, thresh, thresh+100)
    srcEdges = cv.Canny(srcErosion, thresh, thresh+100)

    cv.imshow("targetEdges", targetEdges)
    cv.imshow("srcEdges", srcEdges)

    targetContours, targetHierarchy = cv.findContours(targetEdges, 3, cv.CHAIN_APPROX_SIMPLE)
    srcContours, srcHierarchy = cv.findContours(srcEdges, 3, cv.CHAIN_APPROX_SIMPLE)

    return cv.matchShapes(srcContours[0],targetContours[0],1, 0.0)

def template_image():
    target = cv.imread("F:\\WeiYi\\6.jpg")
    src = cv.imread("F:\\WeiYi\\7.jpg")

    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = src.shape[:2]
    for md in methods:
        result = cv.matchTemplate(target, src, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, [0, 0, 0])
        cv.imshow("pipei"+np.str(md), target)


print(shape_image())
#template_image()
cv.waitKey(0)
cv.destroyAllWindows()