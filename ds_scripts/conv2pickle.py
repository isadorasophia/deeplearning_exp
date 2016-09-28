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
   - dataset_x1: first input of siamese model (string to be loaded as image)
   - dataset_x2: second input of siamese model (string to be loaded as image)
   - labels:     1 if first is older and
                 0 if first is newer.
""" 
def load_data(path):
    dataset_x1 = ['' for x in xrange(MAX_IMG)]
    dataset_x2 = ['' for x in xrange(MAX_IMG)]
    labels     = [0]  * MAX_IMG

    # iterator for both datasets
    x1_it   = 0
    x2_it   = 0

    img_f = os.listdir(path)

    # sort images
    img_f.sort()

    # initialize first day
    cur_day = img_f[0][0:7]
    process = False

    start = 0
    end   = 0

    for img in img_f:
        # if image is still in the same day
        if img[0:7] == cur_day[0:7]:
            end += 1
        # day has changed! process!
        else:
            cur_day = img[0:7]
            process = True

        if process:
            # get files for that day
            day_f = img_f[start:end]

            random.shuffle(day_f)

            for f in day_f:
                try:
                    # a pair is done!
                    if x1_it != x2_it:
                        dataset_x2[x2_it] = path + '/' + f

                        # first = x1; second = x2
                        # if the first is older: 1
                        #              is newer: 0
                        ## year
                        if f[0:3] < l_label[0:3]:
                            labels[x2_it] = 1

                        elif f[0:3] > l_label[0:3]:
                            labels[x2_it] = 0

                        ## month
                        elif f[4:5] > l_label[4:5]:
                            labels[x2_it] = 1

                        elif f[4:5] < l_label[4:5]:
                            labels[x2_it] = 0

                        ### day
                        elif f[6:7] > l_label[6:7]:
                            labels[x2_it] = 1

                        elif f[6:7] < l_label[6:7]:
                            labels[x2_it] = 0

                        ### hour
                        elif f[9:10] > l_label[9:10]:
                            labels[x2_it] = 1

                        elif f[9:10] < l_label[9:10]:
                            labels[x2_it] = 0

                        ### minute
                        elif f[11:12] > l_label[11:12]:
                            labels[x2_it] = 1

                        elif f[11:12] < l_label[11:12]:
                            labels[x2_it] = 0

                        ### second
                        elif f[13:14] > l_label[13:14]:
                            labels[x2_it] = 1

                        elif f[13:14] < l_label[13:14]:
                            labels[x2_it] = 0

                        x2_it += 1

                    else:
                        # get data!
                        dataset_x1[x1_it] = path + '/' + f
                        l_label = f

                        x1_it += 1

                except IOError as e:
                    print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

            # set everything back to normal!
            process = False
            start   = end
            end     = start + 1

            # if it is an odd number of pics, get away with it! take least number
            if x1_it != x2_it:
                x1_it = x2_it

            # clean up the mess...
            del day_f

    # set everything up!
    dataset_x1 = dataset_x1[0:x1_it]
    dataset_x2 = dataset_x2[0:x1_it]
    labels     = labels[0:x1_it]

    return (dataset_x1, dataset_x2, labels)

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

        input_d1 = destination_d + '/x1.pickle'
        input_d2 = destination_d + '/x2.pickle'
        output_d = destination_d + '/y.pickle'

        if os.path.exists(input_d1) or os.path.exists(input_d2) and not force:
            print ('Pickle already present:\n\t%s;\n\t%s\n' % (input_d1, input_d2))
        else:
            print ('Pickling %s;\n\t %s;\n\t %s...\n' % (input_d1, input_d2, output_d))

        dataset_x1, dataset_x2, labels = load_data(com_d)

        # save both input and output
        try:
            with open(input_d1, 'wb') as f:
                pickle.dump(dataset_x1, f, pickle.HIGHEST_PROTOCOL)

            with open(input_d2, 'wb') as f:
                pickle.dump(dataset_x2, f, pickle.HIGHEST_PROTOCOL)

            with open(output_d, 'wb') as f:
                pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            print('Unable to save data to ', destination, ' : ', e)

        del dataset_x1
        del dataset_x2
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
