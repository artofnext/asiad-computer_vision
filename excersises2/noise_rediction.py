import cv2

img = cv2.imread("images2/corruptedRect.png")

strel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))


# erodedImg = cv2.erode(img, strel)
# dilatedImg = cv2.dilate(erodedImg, strel)

imgOpened = cv2.morphologyEx(img, cv2.MORPH_OPEN, strel)
imgClosed = cv2.morphologyEx(imgOpened, cv2.MORPH_CLOSE, strel)

cv2.imshow('image', img)
# cv2.imshow('image dilated', dilatedImg)
# cv2.imshow('image erode', erodedImg)
cv2.imshow('image opened', imgOpened)
cv2.imshow('image closed', imgClosed)
cv2.waitKey(0)
cv2.destroyAllWindows()