import cv2
import numpy as np
from matplotlib import pyplot as plt
import random


def expand(k, m):
    for i in range(len(k)):
        k[i] = k[i][:2] + '0'*(8*m+2 - len(k[i])) + k[i][2:len(k[i])]
    return k


def Hist(image):
    a = [0]*256
    h = image.shape[0]
    w = image.shape[1]
    MN = h*w
    average = 0.0
    for i in range(w):
        for j in range(h):
            pixel = int(image[j][i])
            a[pixel] = a[pixel]+1
    for i in range(256):
        a[i] = a[i]/float(MN)
        average = average+a[i]*i
    return a, average


def getTh(seed, m):
    th = [0, 256]
    seedInt = int(seed, 2)
    for i in range(0, m):
        tmp = seedInt & 255
        if tmp != 0:
            th.append(tmp)
        seedInt = seedInt >> 8
    th.sort()
    return th


def fitness(seed, p, average, m):
    Var = [0.0] * len(seed)
    g_muT = 0.0

    for i in range(256):
        g_muT = g_muT + i * p[i]

    for i in range(len(seed)):
        th = getTh(seed[i], m)
        for j in range(len(th)-1):
            w = [0.0] * (len(th)-1)
            muT = [0.0] * (len(th)-1)
            mu = [0.0] * (len(th)-1)
            for k in range(th[j], th[j+1]):
                w[j] = w[j] + p[k]
                muT[j] = muT[j] + + p[k] * k
            if w[j] > 0:
                mu[j] = muT[j] / w[j]
                Var[i] = Var[i] + w[j] * pow(mu[j] - g_muT, 2)
    return Var


def wheel_selection(seed, Var):
    var = [0.0]*len(Var)
    s = 0.0
    n = ['']*len(seed)
    sumV = sum(Var)
    for i in range(len(Var)):
        var[i] = Var[i]/sumV
    for i in range(1, len(Var)):
        var[i] = var[i] + var[i-1]
    for i in range(len(seed)):
        s = random.random()
        for j in range(len(var)):
            if s <= var[j]:
                n[i] = seed[j]
    return n


def Cross(Next, m):
    for i in range(0, len(Next) - 1, 2):
        if random.random() < 0.7:
            if m > 2:
                tmp = Next[i][10:]
                Next[i] = Next[i][:10] + Next[i+1][10:]
                Next[i+1] = Next[i+1][:10] + tmp
            else:
                tmp = Next[i][6:]
                Next[i] = Next[i][:6] + Next[i+1][6:]
                Next[i+1] = Next[i+1][:6] + tmp
    return Next


def Variation(Next):
    for i in range(len(Next)):
        if random.random() < 0.06:
            Next[i] = bin(int(Next[i], 2)+2)
    return Next


def genetic_thres(image, k, m):
    th = image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for t in range(1, len(k)-1):
                if k[t-1] <= image[i][j] < k[t]:
                    th[i][j] = int(k[t-1])
    return th


imagesrc = cv2.imread("./test.jpg")
gray = cv2.cvtColor(imagesrc, cv2.COLOR_BGR2GRAY)

m = 3
items_x = [i for i in range(0, imagesrc.shape[0])]
items_y = [i for i in range(0, imagesrc.shape[1])]

random.shuffle(items_x)
random.shuffle(items_y)
x = items_x[0:20*m]
y = items_y[0:20*m]
seed = []
Var = 0.0
times = 0
k = 0
P, average = Hist(gray)
for i in range(0, 20):
    code = int(0)
    for j in range(0, m):
        code = code + gray[x[i*j]][y[i*j]] << j*8
    seed.append(bin(code))

while times < 2000:
    Var = fitness(seed, P, average, m)
    Next = wheel_selection(seed, Var)
    Next = Cross(Next, m)
    Next = expand(Variation(Next), m)
    seed = Next
    times = times + 1

    for j in range(len(Var)):
        if Var[j] == max(Var):
            k = getTh(Next[j], m)

plt.subplot(131)
plt.imshow(imagesrc, "gray")

plt.title("source image")
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.hist(imagesrc.ravel(), 256)

plt.title("Histogram")
plt.xticks([])
plt.yticks([])

th1 = genetic_thres(gray, k, m)
plt.subplot(133)
plt.imshow(th1, "gray")

titleName = ''
for i in range(1, len(k)-1):
    titleName = titleName + str(k[i]) + ', '

titleName = titleName[:len(titleName)-2]
plt.title("threshold=%s" % titleName)
plt.xticks([])
plt.yticks([])
plt.show()
