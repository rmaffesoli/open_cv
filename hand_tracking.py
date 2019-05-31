import cv2
import numpy as np
import imutils

from sklearn.metrics import pairwise


class HandDetector(object):
    def __init__(self):
        self.bg = None

    def run_avg(self, image):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, self.bg, self.aWeight)

    def segment(self, image, threshold=25):
        diff = cv2.absdiff(self.bg.astype("uint8"), image)
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        if len(contours) == 0:
            return
        else:
            segmented = max(contours, key=cv2.contourArea)
            return (thresholded, segmented)

    def run(self):
        self.aWeight = 0.5
        self.capture = cv2.VideoCapture(0)
        top, right, bottom, left = 10, 350, 225, 590
        num_frames = 0
        while True:
            (ret, frame) = self.capture.read()
            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)
            clone = frame.copy()
            (height, width) = frame.shape[:2]
            roi = frame[top:bottom, right:left]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if num_frames < 30:
                self.run_avg(gray)
            else:
                hand = self.segment(gray)
                if hand is not None:
                    (thresholded, segmented) = hand

                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                    fingers = self.count(thresholded, segmented)
                    cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # cv2.imshow("Thresholded", thresholded)
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            num_frames += 1

            cv2.imshow('Capture', clone)
            keypress = cv2.waitKey(1 & 0xFF)
            if keypress == ord('q'):
                break
        self.capture.release()
        cv2.destroyAllWindows()

    def count(self, thresholded, segmented):
        chull = cv2.convexHull(segmented)

        extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
        extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
        extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
        extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

        cX = int((extreme_left[0] + extreme_right[0])/2)
        cY = int((extreme_top[1] + extreme_bottom[1])/2)

        distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
        maximum_distance = distance[distance.argmax()]

        radius = int(0.8 * maximum_distance)
        circumference = (2 * np.pi * radius)

        circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

        cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
        circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

        contours = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        count = 0

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
                count += 1
        return count


hand_tracker = HandDetector()
hand_tracker.run()
