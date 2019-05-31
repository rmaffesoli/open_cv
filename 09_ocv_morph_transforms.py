import cv2
import numpy as np


cap = cv2.VideoCapture(0)
lower_bound = np.array([200, 200, 200])
upper_bound = np.array([255, 255, 255])

while True:
    _, frame = cap.read()
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(mask, kernel, iterations=1)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # tophat = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # blackhat = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow('frame', frame)
    # cv2.imshow('res', res)
    # cv2.imshow('erosion', erosion)
    # cv2.imshow('dilation', dilation)

    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)

    # cv2.imshow('tophat', tophat)
    # cv2.imshow('blackhat', blackhat)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
