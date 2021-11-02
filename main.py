import cv2

import numpy as np

img = cv2.imread("img/lenna.png")

shape = img.shape

heightStart = shape[0] // 4
lengthStart = shape[1] // 4

heightEnd = shape[0] // 4 * 3
lengthEnd = shape[1] // 4 * 3

imgCrop = img[heightStart:heightEnd, lengthStart:lengthEnd]

imgCropInv = 255 - imgCrop

img[heightStart:heightEnd, lengthStart:lengthEnd] = imgCropInv

cv2.imshow("Mixed1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
