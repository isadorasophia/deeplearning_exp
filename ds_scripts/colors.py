# references: http://giusedroid.blogspot.com.br/2015/04/using-python-and-k-means-in-hsv-color.html

import cv2
import numpy as np
import os
import random

from matplotlib import pyplot as plt

path = '/home/bonnibel/ic/deeplearning/pless/testing/2013.05/'
res_path = path + 'results/histogram/'

IMG_SIZE = 224

# initialize stuff
files = sorted(os.listdir(path))
cur_f = files[0][0:8]

days  = []
cur_d = []

if not os.path.exists(res_path):
    os.makedirs(res_path)

# separate by days
for f in files:
    if os.path.isdir(path + f):
        continue

    if f[0:8] == cur_f:
        cur_d.append(f)
    else:
        days.append(cur_d)
        cur_d = []
        cur_f = f[0:8]

# append last day, if it is the case
if cur_d is not None and len(cur_d) > 1:
    days.append(cur_d)

random.shuffle(days)

day = days[0]

random.shuffle(day)

order = range(0, len(day))
random.shuffle(order)

j = len(day) - 1

# create proper directory
cur_res_path = res_path + str(day[0][0:8]) + '/'

if not os.path.exists(cur_res_path):
    os.makedirs(cur_res_path)

# iterate over files
while j >= 0:
    img = cv2.imread(path + day[j])

    # plot rgb!
    color = ('b','g','r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color = col)
        plt.xlim([0, 256])
        plt.ylim([0, IMG_SIZE * IMG_SIZE * 3])

    j -= 1

    name = cur_res_path + 'rgb_' + day[j][9:15] + '.jpg'

    plt.savefig(name)
    plt.clf()
    plt.close()

    # plot hsv!
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]

    plt.figure(figsize=(10,8))
    plt.subplot(311)                             # plot in the first cell
    plt.ylim([0, 100000])
    plt.subplots_adjust(hspace=.5)
    plt.title("Hue")
    plt.hist(np.ndarray.flatten(hue), bins=180)
    plt.subplot(312)                             # plot in the second cell
    plt.ylim([0, 20000])  
    plt.title("Saturation")
    plt.hist(np.ndarray.flatten(sat), bins=128)
    plt.subplot(313)                             # plot in the third cell
    plt.ylim([0, 60000])
    plt.title("Luminosity Value")
    plt.hist(np.ndarray.flatten(val), bins=128)
    
    name = cur_res_path + 'hsv_' + day[j][9:15] + '.jpg'

    plt.savefig(name)
    plt.clf()
    plt.close()

plt.close()