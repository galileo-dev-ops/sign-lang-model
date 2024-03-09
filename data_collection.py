import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

labels = ['A', 'B', 'C']

folder = "Data/C"
counter = 0
while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 480))
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the cropping dimensions are within the image dimensions
        y = max(0, y - offset)
        x = max(0, x - offset)
        h = min(img.shape[0] - y, h + offset)
        w = min(img.shape[1] - x, w + offset)

        if h > 0 and w > 0:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            if imgCrop.size > 0:
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = int(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = int((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = int(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = int((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
