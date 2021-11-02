import cv2

import numpy as np
from os import listdir
from os.path import isfile, join
path = "img/challenge1"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
# img1 = cv2.imread("img/challenge1/beach00.jpg")

def getColorArea(img, h, delta):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = (h, 10, 10)
    upper = (h + delta, 255, 255)
    mask = cv2.inRange(img, lower, upper)

    pixels = cv2.countNonZero(mask)
    image_area = img.shape[0] * img.shape[1]
    area_ratio = (pixels / image_area) * 100

    return area_ratio

for file in onlyfiles:
    img = cv2.imread(path + "/" + file)
    colorArea = getColorArea(img, 10, 50)
    isBeach = "nonbeach" if colorArea > 45 else "beach"
    res = file + " - " + isBeach
    print(res)

# cv2.imshow("Image: ", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
