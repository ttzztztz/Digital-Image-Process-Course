# -*- coding: UTF-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

m1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
m2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
img = cv2.imread("./test.jpg", 0)
img = cv2.GaussianBlur(img, (3, 3), 2)

img1 = np.zeros(img.shape, dtype="uint8")
theta = np.zeros(img.shape, dtype="float")
img = cv2.copyMakeBorder(img, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)
rows, cols = img.shape
for i in range(1, rows-1):
    for j in range(1, cols-1):
        Gy = (np.dot(np.array([1, 1, 1]), (m1 * img[i - 1:i +
                                                    2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]]))
        Gx = (np.dot(np.array([1, 1, 1]), (m2 * img[i - 1:i +
                                                    2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]]))
        if Gx[0] == 0:
            theta[i-1, j-1] = 90
            continue
        else:
            temp = (np.arctan(Gy[0] / Gx[0])) * 180 / np.pi
        if Gx[0]*Gy[0] > 0:
            if Gx[0] > 0:
                theta[i-1, j-1] = np.abs(temp)
            else:
                theta[i-1, j-1] = (np.abs(temp) - 180)
        if Gx[0] * Gy[0] < 0:
            if Gx[0] > 0:
                theta[i-1, j-1] = (-1) * np.abs(temp)
            else:
                theta[i-1, j-1] = 180 - np.abs(temp)
        img1[i-1, j-1] = (np.sqrt(Gx**2 + Gy**2))
for i in range(1, rows - 2):
    for j in range(1, cols - 2):
        if (((theta[i, j] >= -22.5) and (theta[i, j] < 22.5)) or
                ((theta[i, j] <= -157.5) and (theta[i, j] >= -180)) or
                ((theta[i, j] >= 157.5) and (theta[i, j] < 180))):
            theta[i, j] = 0.0
        elif (((theta[i, j] >= 22.5) and (theta[i, j] < 67.5)) or
              ((theta[i, j] <= -112.5) and (theta[i, j] >= -157.5))):
            theta[i, j] = 45.0
        elif (((theta[i, j] >= 67.5) and (theta[i, j] < 112.5)) or
              ((theta[i, j] <= -67.5) and (theta[i, j] >= -112.5))):
            theta[i, j] = 90.0
        elif (((theta[i, j] >= 112.5) and (theta[i, j] < 157.5)) or
              ((theta[i, j] <= -22.5) and (theta[i, j] >= -67.5))):
            theta[i, j] = -45.0

img2 = np.zeros(img1.shape)

for i in range(1, img2.shape[0]-1):
    for j in range(1, img2.shape[1]-1):
        if (theta[i, j] == 0.0) and (img1[i, j] == np.max([img1[i, j], img1[i+1, j], img1[i-1, j]])):
            img2[i, j] = img1[i, j]

        if (theta[i, j] == -45.0) and img1[i, j] == np.max([img1[i, j], img1[i-1, j-1], img1[i+1, j+1]]):
            img2[i, j] = img1[i, j]

        if (theta[i, j] == 90.0) and img1[i, j] == np.max([img1[i, j], img1[i, j+1], img1[i, j-1]]):
            img2[i, j] = img1[i, j]

        if (theta[i, j] == 45.0) and img1[i, j] == np.max([img1[i, j], img1[i-1, j+1], img1[i+1, j-1]]):
            img2[i, j] = img1[i, j]

img3 = np.zeros(img2.shape)
TL = 50
TH = 100
for i in range(1, img3.shape[0]-1):
    for j in range(1, img3.shape[1]-1):
        if img2[i, j] < TL:
            img3[i, j] = 0
        elif img2[i, j] > TH:
            img3[i, j] = 255
        elif ((img2[i+1, j] < TH) or (img2[i-1, j] < TH) or (img2[i, j+1] < TH) or
                (img2[i, j-1] < TH) or (img2[i-1, j-1] < TH) or (img2[i-1, j+1] < TH) or
              (img2[i+1, j+1] < TH) or (img2[i+1, j-1] < TH)):
            img3[i, j] = 255


cv2.imshow("Original", img)
cv2.imshow("Gradient magnitude", img1)
cv2.imshow("Non-maximum suppression grayscale", img2)
cv2.imshow("Final", img3)
cv2.imshow("Angle grayscale", theta)
cv2.waitKey(0)
