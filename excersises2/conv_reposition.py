import cv2

import numpy as np

img = cv2.imread("images2/lastsupper.jpg")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gauss = np.array((
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1]), dtype=int) / 49

sharp = np.array((
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]), dtype=int)

# imgGray = cv2.filter2D(imgGray, -1, sharp)

x = np.array((
    [1,1,1],
    [0,0,0],
    [1,1,1]), dtype=int) / 6

x = np.array((
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]), dtype=int)

y = np.array((
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]), dtype=int)

filteredImgX = cv2.filter2D(imgGray, -1, x)
filteredImgY = cv2.filter2D(imgGray, -1, y)

filteredImgRes = filteredImgX + filteredImgY

gradient = cv2.normalize(filteredImgRes,None,0,255,cv2.NORM_MINMAX,cv2.CV_8S)

cv2.imshow("Filtered", filteredImgRes)
cv2.imshow("gradient", gradient)

cv2.waitKey(0)
cv2.destroyAllWindows()