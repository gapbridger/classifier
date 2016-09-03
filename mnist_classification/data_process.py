from __future__ import print_function
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit


import numpy as np
from skimage import io, color
# parameters for images size: to be adjusted
# kRwidth = 128
# kRheight = 128

# animal_types: 0, ..., 11 are used for labels
animal_types = ['cat', 'chipmunk', 'Dog', 'fox',
                'giraffe', 'guinea pig', 'hyena', 'reindeer',
                'sikadeer', 'squirrel', 'weasel', 'wolf']


def ImageSearch(dir):
    '''
    find the images in the sub dir
    :param dir: the file dir that contains images
    :return: list of image names
    '''
    files = []
    imgs = []
    for root, subdir, file in os.walk(dir):
        files.append(file)
    for file in files[0]:
        ext = os.path.splitext(file)[1].lower()
        if ext in ('.jpg', ',jpeg', '.png', '.bmp', ',gif'):
            imgs.append(file)
    return imgs

def ProcessData(dir):
    ''' Load the dataset, and store them in the pickle block
    :param dir: file dir
    :save: pickle of of images and labels
    '''
    # train data: image
    type = -1
    for animal in animal_types:
        subdir = os.path.join(dir, animal)
        img_lists = ImageSearch(subdir)   # search images
        type = type+1;
        images = []
        labels = []
        for im in img_lists:   # each files
            img_dir = os.path.join(subdir, im)
            # in RGB: dtype = int8
            # images.append(io.imread(img_dir))
            # in grayscale: dtype = int8
            images.append( (color.rgb2gray(io.imread(img_dir))).astype(np.int8) )

            labels.append(type)
        labels = np.asarray(labels, dtype = np.int8)
        # pickle
        pickle_name = 'bot_train_'+str(type)
        pickle_name = pickle_name + '.pkl'
        pickle_dir = os.path.join(dir, pickle_name)
        write_file = open(pickle_dir, 'wb')
        pickle.dump(images, write_file, -1)
        pickle.dump(labels, write_file, -1)
        write_file.close()


###############################
# save and load pickles
###############################

dir = 'C:\zzdata\bot_train'    # to be changed to local address

# generate the pickles (in 12 pieces)
ProcessData(dir)
# read the pickles
i = 0
while i<1:
    pickle_name = 'bot_train_' + str(i)
    pickle_name = pickle_name + '.pkl'
    subdir = os.path.join(dir, pickle_name)
    read_file = open(subdir,'rb')
    images = pickle.load(read_file)
    labels = pickle.load(read_file)
    read_file.close()
    i = i+1

os.system("pause")
