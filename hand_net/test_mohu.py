import cv2
import os
import numpy as np

def get_imglist(path):
    imglist = []
    for f in os.listdir(path):
        if os.path.splitext(f)[1] == ".jpg" or os.path.splitext(f)[1] == ".JPG":
            imglist.append(f)
        if os.path.splitext(f)[1] == ".png":
            imglist.append(f)
    return imglist

def motion_blur(image, degree=12, angle=45):
  image = np.array(image)
  M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
  motion_blur_kernel = np.diag(np.ones(degree))
  motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
  motion_blur_kernel = motion_blur_kernel / degree
  blurred = cv2.filter2D(image, -1, motion_blur_kernel)
  # convert to uint8
  cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
  blurred = np.array(blurred, dtype=np.uint8)
  return blurred

def main():
    print("start")
    path = "data"
    path2 = "test_gs"
    path3 = "test_yd"
    imglist = get_imglist(path)
    for img in imglist:
        img_path = os.path.join(os.path.realpath(path),img)
        img_cv = cv2.imread(img_path)
        img_GaussianBlur = cv2.GaussianBlur(img_cv, ksize=(9, 9), sigmaX=0, sigmaY=0)
        cv2.imwrite(os.path.join(os.path.realpath(path2),img),img_GaussianBlur)

        img_MotionBlur = motion_blur(img_cv,10,30)
        cv2.imwrite(os.path.join(os.path.realpath(path3),img),img_MotionBlur)

if __name__ == "__main__":
    main()