# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]:
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])

        value = Wb * Wf * (mub - muf) ** 2

        print("Wb", Wb, "Wf", Wf)
        print("t", t, "value", value)

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    print(final_thresh)
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
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
