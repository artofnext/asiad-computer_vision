import cv2

import numpy as np

h = np.zeros((255, 180), np.uint8)
s = np.zeros((255, 180), np.uint8)
v = np.zeros((255, 180), np.uint8)

for i in range(180):
    h[:, i] = i

for i in range(255):
    s[i, :] = 255 - i

for i in range(255):
    v[i, :] = 255 - i

imgHSV = cv2.merge((h, s, v))

img = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)

cv2.imshow("Gradient", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
