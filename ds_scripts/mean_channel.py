# -*- coding: utf-8 -*-
# find mean value from dataset channels

import cv2
import os
import numpy as np

path = "/media/bonnibel/Jerônimo/AMOS_Data/tr_raw_data/"
files = []

# check if is it a valid path
if os.path.exists(path):
    subd = []

    # list all valid subdirectories
    for dirpath, dirnames, filenames in os.walk(path):
        if not dirnames and len(filenames) is not 0:
            subd.append(dirpath)

    # get all the valid files from each directory
    for dir_ in subd:
        for f in sorted(os.listdir(dir_)):
            files.append(dir_ + '/' + f)

print ("Done collecting files!")

# mean arrays
r_mean = []
g_mean = []
b_mean = []

step = 0

# for each of the images...
for i in files:
    try:
        f = cv2.imread(i)
    except IOError:
        continue

    if f is None:
        continue

    # find the mean value from channels
    b_mean.append(np.mean(f[:, :, 0]))
    g_mean.append(np.mean(f[:, :, 1]))
    r_mean.append(np.mean(f[:, :, 2]))

    del f

    if step % 1000 == 0 and step > 0:
        print("Iteration no. %d" % step)

    step += 1

print ("Done collecting mean values!")

# finally, find the mean value from each image of the dataset
AMOS_MEAN = [np.mean(b_mean), np.mean(g_mean), np.mean(r_mean)]

print (AMOS_MEAN)