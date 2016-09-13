# -*- coding: utf-8 -*-

import numpy as np
import os
import gc
    
# python serialization
from six.moves import cPickle as pickle

# standard definition
X1 = 0
X2 = 1
Y  = 2

# get files from both test and train set
class dataset:
    def __init__(self, path, batch_size):
        self.files = []

        # check if is it a valid path
        if os.path.exists(path):
            subd = []

            # list all valid subdirectories
            for dirpath, dirnames, filenames in os.walk(path):
                if not dirnames and len(filenames) is not 0:
                    subd.append(dirpath)

            # get all the valid files from each directory
            for dir_ in subd:
                t = [dir_ + '/' + f for f in sorted(os.listdir(dir_))]

                self.files.append(t)

        # iterator for total files and batch files
        self.current_file = 0
        self.batch_counter = 0

        # get batch size
        self.batch_size = batch_size

        self.current_batches = {'x1': None, 'x2': None, 'y': None}

    # keep getting next (available) batch
    def get_next_batch(self):
        filename = self.files[self.current_file]

        # check if there is a valid candidate as a next extraction file
        if len(self.files) > self.current_file + 1:
            next_filename = self.files[self.current_file + 1]

        else:
            # take a None object for each candidate
            next_filename = {'x1': None, 'x2': None, 'y': None}

        x1 = self.get_batch(filename[X1], next_filename[X1], 'x1')
        x2 = self.get_batch(filename[X2], next_filename[X2], 'x2')
        y  = self.get_batch(filename[Y], next_filename[Y], 'y')

        assert len(x1) != 100 or len(x2) != 100 or y != 100, 'Incorrect batch files!'

        self.batch_counter += 1

        return x1, x2, y

    # get a valid batch, given the filenames and current id
    def get_batch(self, filename, next_filename, cur_id):
        if self.current_batches[cur_id] is None:
            self.current_batches[cur_id] = self.open_pickle(filename)

            print ('Unpickling file...')

        size = len(self.current_batches[cur_id])

        start = self.batch_counter * self.batch_size
        end = start + self.batch_size

        split = False
        extra = 0

        if end - size > 0:
            extra = end - size
            end =  size

            split = True

            self.current_file += 1
            self.batch_counter = 0

        # assign variable and (possibly) clean up space!
        final_b = self.current_batches[cur_id][int(start):int(end)]
        
        # del self.current_batches[cur_id][start:end]

        if end - size == 0:
            self.current_file += 1
            self.batch_counter = 0

            del self.current_batches[cur_id]

        if split:
            # if we are over it
            if not next_filename:
                return None

            self.current_batches[cur_id] = self.open_pickle(next_filename)

            start = 0
            end = extra

            final_b[batch_size - extra:batch_size] = self.self.current_batches[cur_id][start:end]

            # clean up space!
            # del self.current_batches[cur_id][start:end]

        return final_b

    # extract pickle file
    def open_pickle(self, path):
        f = open(path, "rb")

        # disable garbage collector
        gc.disable()

        p = pickle.load(f)

        # enable garbage collector again
        gc.enable()
        f.close()

        return p
