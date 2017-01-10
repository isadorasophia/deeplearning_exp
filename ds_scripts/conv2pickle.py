# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from scipy import ndimage
from six.moves import cPickle as pickle

import numpy as np
import cv2
import scipy

import os
import sys

import random

DB_NAME = '/datasets/isophia/AMOS/tr_raw_data/'

# constant values
MAX_IMG  = 18000
STR_SIZE = 20

""" 
    load data from a given path (i.e. 00000001/0/) and returns it as:
   - dataset_x1: input of model (string to be loaded as image)
   - labels:     1 if first is older and
                 0 if first is newer.
""" 
def load_data(path):
    dataset_x  = ['' for x in xrange(MAX_IMG)]
    labels     = [0]  * MAX_IMG

    # iterator for both datasets
    x_it   = 0

    img_f = os.listdir(path)

    # sort images
    img_f.sort()

    # initialize first day
    cur_day = img_f[0][0:8]
    process = False

    start = 0
    end   = 0

    for img in img_f:
        dataset_x[x_it] = path + '/' + img

        labels[x_it] = 0
        labels[x_it] += img[9:11] * 60 * 60  # hours
        labels[x_it] += img[11:13] * 60      # minutes
        labels[x_it] += img[13:15]           # seconds

        x_it += 1

    # set everything up!
    dataset_x  = dataset_x[0:x_it]
    labels     = labels[0:x_it]

    return (dataset_x, labels)

def try_pickle(path, destination, force=False):
    # list all the camera directories, according to path
    camera_dir = os.listdir(path)

    # read camera locations
    for d in camera_dir:
        com_d = path + '/' + d

        img_f = os.listdir(com_d)

        destination_d = destination + '/' + d

        # check if it already exists, otherwise create it!
        if not os.path.exists(destination_d):
            os.makedirs(destination_d   )

        input_d1 = destination_d + '/x.pickle'
        output_d = destination_d + '/y.pickle'

        if os.path.exists(input_d1) and not force:
            print ('Pickle already present:\n\t%s;\n\t%s\n' % input_d1)
        else:
            print ('Pickling %s;\n\t %s;\n\t %s...\n' % (input_d1, output_d))

        dataset_x, labels = load_data(com_d)

        # save both input and output
        try:
            with open(input_d1, 'wb') as f:
                pickle.dump(dataset_x1, f, pickle.HIGHEST_PROTOCOL)

            with open(output_d, 'wb') as f:
                pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            print('Unable to save data to ', destination, ' : ', e)

        del dataset_x
        del labels

    return

if __name__ == "__main__":
    database = os.listdir(DB_NAME)
    destination = os.path.join(DB_NAME, 'pickle_files/')

    # check if it already exists, otherwise create it!
    if not os.path.exists(destination):
        os.makedirs(destination)

    # start checking for each camera!
    for dataset in database:
        # skip pickle database
        if dataset == 'pickle_files':
            continue

        # destination of pickle files
        new_destination = os.path.join(destination, dataset)

        # check if it already exists, otherwise create it!
        if not os.path.exists(new_destination):
            os.makedirs(new_destination)

        # finally, try to create pickle file
        try_pickle(os.path.join(DB_NAME, dataset), \
                   new_destination)
