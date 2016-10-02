# Visualize differences between two images at the same day

import cv2
import numpy as np

import os
import random

from datetime import datetime

path = '/home/bonnibel/ic/deeplearning/pless/testing/2013.05/'
res_path = path + 'results/'

files = sorted(os.listdir(path))
cur_f = files[0][0:8]

days  = []
cur_d = []

if not os.path.exists(res_path):
    os.makedirs(res_path)

# separate by days
for f in files:
    if f[0:8] == cur_f:
        cur_d.append(f)
    else:
        days.append(cur_d)
        cur_d = []
        cur_f = f[0:8]

for day in days:
    random.shuffle(day)

    order = range(0, len(day))
    random.shuffle(order)

    j = len(day) - 2

    while j > 0:
        a = cv2.imread(path + day[j])
        b = cv2.imread(path + day[j + 1])

        g = a.copy()
        gpA = [g]
        for i in xrange(6):
            g = cv2.pyrDown(g)
            gpA.append(g)

        g = b.copy()
        gpB = [g]
        for i in xrange(6):
            g = cv2.pyrDown(g)
            gpB.append(g)

        lpA = [gpA[5]]
        for i in xrange(5, 0, -1):
            GE = cv2.pyrUp(gpA[i])
            L = cv2.subtract(gpA[i-1], GE)
            lpA.append(L)

        lpB = [gpB[5]]
        for i in xrange(5,0,-1):
            GE = cv2.pyrUp(gpB[i])
            L = cv2.subtract(gpB[i-1],GE)
            lpB.append(L)   

        # Now add left and right halves of images in each level
        LS = []
        for la, lb in zip(lpA, lpB):
            rows ,cols, dpt = la.shape
            ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
            LS.append(ls)

        # now reconstruct
        ls_ = LS[0]
        for i in xrange(1,6):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[i])

        time_one = day[j][9:11] + ':' + day[j][11:13] + ':' + day[j][13:15] 
        time_two = day[j + 1][9:11] + ':' + day[j + 1][11:13] + ':' + day[j + 1][13:15] 

        FMT = '%H:%M:%S'

        tdelta = datetime.strptime(time_two, FMT) - datetime.strptime(time_one, FMT)
        tdelta = tdelta.seconds/60

        name_blend = 'blending_' + str(tdelta) + '.jpg'
        name_diff  = 'diff_' + str(tdelta) + '.jpg'

        c = cv2.absdiff(a, b)

        cv2.imwrite(res_path + name_blend, ls_)
        cv2.imwrite(res_path + name_diff, c)

        j -= 2
