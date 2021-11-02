import numpy as np
import cv2


# Image operation using thresholding
img = cv2.imread("images2/coins102.jpg")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

ret, thresh = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV +
                            cv2.THRESH_OTSU)
# cv2.imshow('image', thresh)

# Noise removal using Morphological
# closing operation
# kernel = np.ones((3, 3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                           kernel)

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,
                           kernel1)

# Background area using Dialation

# bg = cv2.erode(opening, kernel)

# Finding foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 0)
ret, fg = cv2.threshold(dist_transform, 0.02
                        * dist_transform.max(), 255, 0)
fg = fg.astype(np.uint8)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
print(nb_components)
# cv2.imshow('eroded', bg)
cv2.imshow('image', fg)
cv2.waitKey(0)
cv2.destroyAllWindows()