# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weight = 1.0/pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_threshold = -1
    final_value = -1
    for t in bins[1:-1]:
        wb = np.sum(his[:t]) * mean_weight
        wf = np.sum(his[t:]) * mean_weight

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])

        value = wb * wf * (mub - muf) ** 2
        if value > final_value:
            final_threshold = t
            final_value = value
    final_img = gray.copy()
    print(final_threshold)
    final_img[gray > final_threshold] = 255
    final_img[gray < final_threshold] = 0
    return final_img


image = cv2.imread("./test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(131)
plt.imshow(image, "gray")
plt.title("source image")
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.hist(image.ravel(), 256)

plt.title("Histogram")
plt.xticks([])
plt.yticks([])

ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
plt.subplot(133)
plt.imshow(th1, "gray")
plt.title("threshold=%s" % str(ret1))
plt.xticks([])
plt.yticks([])
plt.show()
