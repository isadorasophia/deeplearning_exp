# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from scipy import ndimage
from six.moves import cPickle as pickle
from keras.utils.io_utils import HDF5Matrix

import numpy as np
import cv2
import scipy

import os
import sys

DB_NAME = '/media/bonnibel/Jerônimo/AMOS_Data/left/'

# constant values
MAX_IMG  = 18000
IMG_SIZE = 224

def cv2keras(img):
    return np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)

def load_data(path):
    dataset_x1 = np.ndarray(shape = (MAX_IMG, 3, IMG_SIZE, IMG_SIZE),
                            dtype = np.uint8)
    dataset_x2 = np.ndarray(shape = (MAX_IMG, 3, IMG_SIZE, IMG_SIZE),
                            dtype = np.uint8)

    labels  = [0] * MAX_IMG
    l_label = ''
    
    x1_it = 0
    x2_it = 0

    img_f = os.listdir(path)

    # randomize images
    img_f = np.random.permutation(img_f)

    for img in img_f:
        try:
            data = cv2.imread(path + '/' + img)

            if data is None or data.shape < 3:
                continue
            else:
                data = cv2keras(data)

            if data.shape != (3, IMG_SIZE, IMG_SIZE):
                raise Exception('Unexpected image shape: %s' % str(data.shape))

            # a pair is done!
            if x1_it != x2_it:
                dataset_x2[x2_it, :, :, :] = data

                # first = x1; second = x2
                # if the first is older: 1
                #              is newer: 0
                ## year
                if int(img[0:3]) < int(l_label[0:3]):
                    labels[x2_it] = 1

                elif int(img[0:3]) > int(l_label[0:3]):
                    labels[x2_it] = 0

                ## month
                elif int(img[4:5]) > int(l_label[4:5]):
                    labels[x2_it] = 1

                elif int(img[4:5]) < int(l_label[4:5]):
                    labels[x2_it] = 0

                ### day
                elif int(img[6:7]) > int(l_label[6:7]):
                    labels[x2_it] = 1

                elif int(img[6:7]) < int(l_label[6:7]):
                    labels[x2_it] = 0

                ### hour
                elif int(img[9:10]) > int(l_label[9:10]):
                    labels[x2_it] = 1

                elif int(img[9:10]) < int(l_label[9:10]):
                    labels[x2_it] = 0

                ### minute
                elif int(img[11:12]) > int(l_label[11:12]):
                    labels[x2_it] = 1

                elif int(img[11:12]) < int(l_label[11:12]):
                    labels[x2_it] = 0

                ### second
                elif int(img[13:14]) > int(l_label[13:14]):
                    labels[x2_it] = 1

                elif int(img[13:14]) < int(l_label[13:14]):
                    labels[x2_it] = 0

                x2_it += 1

            else:
                # get data!
                dataset_x1[x1_it, :, :, :] = data
                labels[x1_it] = 0

                l_label = img

                x1_it += 1

        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    # if it is an odd number of pics, get away with it! take least number
    if x1_it != x2_it:
        x1_it = x2_it

    dataset_x1 = dataset_x1[0:x1_it, :, :, :]
    dataset_x2 = dataset_x2[0:x1_it, :, :, :]
    labels  = labels[0:x1_it]

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
            print ('Pickling for %s or %s already present!' % (input_d1, input_d2))
        else:
            print ('Pickling %s;\n\t %s...\n' % (input_d1, output_d))

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
