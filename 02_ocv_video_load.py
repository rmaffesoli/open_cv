import cv2
import numpy as np


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('/Users/rmaffesoli/Documents/personal/openCV/ouput.mp4v', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()