import cv2
import numpy as np


img = cv2.imread('/Users/rmaffesoli/Documents/personal/openCV/cup.jpg', cv2.IMREAD_COLOR)
img[55, 55] = [255, 255, 255]
px = img[55, 55]
print(px)

cup = img[70:250, 5:200]
img[0:180, 0:195] = cup

# cup
print(cup)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
