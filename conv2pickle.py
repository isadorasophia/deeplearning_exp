from matplotlib import pyplot as plt
from scipy import ndimage
from six.moves import cPickle as pickle
from keras.utils.io_utils import HDF5Matrix

import numpy as np
import cv2
import scipy

import os
import sys

DB_NAME = '/home/bonnibel/ic/deeplearning/pless/treatment/testing/final/'

# constant values
MAX_IMG  = 10000
IMG_SIZE = 224

def cv2keras(img):
    return np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)

def load_data(path):
    dataset = np.ndarray(shape = (MAX_IMG, 3, IMG_SIZE, IMG_SIZE),
                         dtype = np.uint8)

    labels  = [0] * MAX_IMG
    l_label = ''
    
    num_img = 0

    # list all the camera directories, according to path
    camera_dir = os.listdir(path)

    # read camera locations
    for d in camera_dir:
        com_d = path + '/' + d

        img_f = os.listdir(com_d)

        # randomize images
        img_f = np.random.permutation(img_f)

        for img in img_f:
            try:
                data = cv2keras(cv2.imread(com_d + '/' + img))

                if data.shape != (3, IMG_SIZE, IMG_SIZE):
                    raise Exception('Unexpected image shape: %s' % str(data.shape))

                # get data!
                dataset[num_img, :, :, :] = data

                labels[num_img] = 0

                # a pair is done!
                if num_img != 0 and num_img % 2 != 0:
                    # if the second is older: 1
                    #               is newer: 0
                    ## year
                    if int(img[0:3]) > int(l_label[0:3]):
                        labels[num_img]     = 1
                        labels[num_img - 1] = 1
                    elif int(img[0:3]) < int(l_label[0:3]):
                        labels[num_img]     = 0
                        labels[num_img - 1] = 0
                    ## month
                    elif int(img[4:5]) > int(l_label[4:5]):
                        labels[num_img]     = 1
                        labels[num_img - 1] = 1
                    elif int(img[4:5]) < int(l_label[4:5]):
                        labels[num_img]     = 0
                        labels[num_img - 1] = 0
                    ### day
                    elif int(img[6:7]) > int(l_label[6:7]):
                        labels[num_img]     = 1
                        labels[num_img - 1] = 1
                    elif int(img[6:7]) < int(l_label[6:7]):
                        labels[num_img]     = 0
                        labels[num_img - 1] = 0
                    ### hour
                    elif int(img[9:10]) > int(l_label[9:10]):
                        labels[num_img]     = 1
                        labels[num_img - 1] = 1
                    elif int(img[9:10]) < int(l_label[9:10]):
                        labels[num_img]     = 0
                        labels[num_img - 1] = 0
                    ### minute
                    elif int(img[11:12]) > int(l_label[11:12]):
                        labels[num_img]     = 1
                        labels[num_img - 1] = 1
                    elif int(img[11:12]) < int(l_label[11:12]):
                        labels[num_img]     = 0
                        labels[num_img - 1] = 0
                    ### second
                    elif int(img[13:14]) > int(l_label[13:14]):
                        labels[num_img]     = 1
                        labels[num_img - 1] = 1
                    elif int(img[13:14]) < int(l_label[13:14]):
                        labels[num_img]     = 0
                        labels[num_img - 1] = 0
                else:
                    l_label = img

                num_img += 1

            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    # if it is an odd number of pics, get away with it!
    if num_img % 2 != 0:
        num_img -= 1

    dataset = dataset[0:num_img, :, :, :]
    labels  = labels[0:num_img]

    print labels

    return (dataset, labels)

def try_pickle(path, destination, force=False):
    input_d  = destination + '/x.pickle'
    output_d = destination + '/y.pickle'

    if os.path.exists(input_d) and not force:
        print ('Pickling for %s already present!' % input_d)
    else:
        print ('Pickling %s;\n\t %s...\n' % (input_d, output_d))

    dataset, labels = load_data(path)

    # save both input and output
    try:
        with open(input_d, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

        with open(output_d, 'wb') as f:
            pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print('Unable to save data to ', set_filename, ' : ', e)

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
