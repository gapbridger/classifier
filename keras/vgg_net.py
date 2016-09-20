"""
CNN model:
1. Train the model based on data set
2. Predict the animals, give the probability distribution



"""
from __future__ import print_function
import numpy as np
from skimage.feature.corner_cy import img_as_float
np.random.seed(1234)

import argparse
import os
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import cifar10

from image_preprocess import ImagePreprocess
from image_preprocess import read_cifar_10
import gc


def vgg_net_16(n_class, vgg_16_weights_path=None, vgg_11_weights_path=None):
    
    rectified_linear_unit = 'relu'
    softmax = 'softmax'
    
    model = Sequential()

    model.add(ZeroPadding2D((1, 1),input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation=rectified_linear_unit, name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation=rectified_linear_unit, name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation=rectified_linear_unit, name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation=rectified_linear_unit, name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation=rectified_linear_unit, name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation=rectified_linear_unit, name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation=rectified_linear_unit, name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # fully connected layers
    model.add(Flatten(name="flatten"))
    model.add(Dense(2048, activation=rectified_linear_unit, name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation=rectified_linear_unit, name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, name='dense_3'))
    model.add(Activation(softmax, name="softmax"))

    if vgg_16_weights_path:
        model.load_weights(vgg_16_weights_path)
        
    if vgg_11_weights_path:
        # use vgg 11 weights to initialize vgg 16 network
        vgg_11 = vgg_net_11(n_class)
        vgg_11.load_weights(vgg_11_weights_path)
     
        vgg_11_layer_dict = dict([(layer.name, layer) for layer in vgg_11.layers])
        conv_1_1_weights = vgg_11_layer_dict['conv1_1'].get_weights()
        conv_2_1_weights = vgg_11_layer_dict['conv2_1'].get_weights()
        conv_3_1_weights = vgg_11_layer_dict['conv3_1'].get_weights()
        conv_4_1_weights = vgg_11_layer_dict['conv4_1'].get_weights()
        dense_1_weights = vgg_11_layer_dict['dense_1'].get_weights()
        dense_2_weights = vgg_11_layer_dict['dense_2'].get_weights()
        
        vgg_16_layer_dict = dict([(layer.name, layer) for layer in model.layers])
        vgg_16_layer_dict['conv1_1'].set_weights(conv_1_1_weights)
        vgg_16_layer_dict['conv2_1'].set_weights(conv_2_1_weights)
        vgg_16_layer_dict['conv3_1'].set_weights(conv_3_1_weights)
        vgg_16_layer_dict['conv4_1'].set_weights(conv_4_1_weights)
        vgg_16_layer_dict['dense_1'].set_weights(dense_1_weights)
        vgg_16_layer_dict['dense_2'].set_weights(dense_2_weights)
        

    return model


def vgg_net_11(n_class, weights_path=None):
    
    rectified_linear_unit = 'relu'
    softmax = 'softmax'
    
    model = Sequential()

    model.add(ZeroPadding2D((1, 1),input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation=rectified_linear_unit, name='conv1_1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation=rectified_linear_unit, name='conv2_1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation=rectified_linear_unit, name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation=rectified_linear_unit, name='conv3_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv4_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv5_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # fully connected layers
    model.add(Flatten(name="flatten"))
    model.add(Dense(2048, activation=rectified_linear_unit, name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation=rectified_linear_unit, name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, name='dense_3'))
    model.add(Activation(softmax, name="softmax"))

    if weights_path:
        model.load_weights(weights_path)
        
    return model


def extract_images(image_preprocessor, partition_index, data_type, avg_rgb):
    images, labels = image_preprocessor.load_data_partition(partition_index, data_type)
    images = image_preprocessor.crop_images(images, mode='random')
    images = np.stack(images, axis=0)
    images = img_as_float(images) - avg_rgb
    # images = np.divide(images, 255.0)
    images = np.rollaxis(images, 3, 1)
    return images, labels


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='vgg_net')
    parser.add_argument("-i", "--input", help="input data directory", default="")
    parser.add_argument("-o", "--output", help="output data directory", default="")
    
    args = parser.parse_args()
    
    input_root_dir = args.input
    output_root_dir = args.output
    
    n_partition = 24
    n_epoch = 100
    n_class = 12
    
    avg_rgb = np.asarray([132.4509, 123.5161, 105.4855], dtype=float) / 255.0
    print('avg_rgb %f %f %f' %(avg_rgb[0], avg_rgb[1], avg_rgb[2]))
    
    image_preprocessor = ImagePreprocess(input_root_dir, output_root_dir)
    
    # averaged over training set
    sgd = SGD(lr=0.004, decay=1e-6, momentum=0.9, nesterov=True)
#     vgg = vgg_net_11(n_class)
    vgg = vgg_net_16(n_class, vgg_11_weights_path='/home/tao/Projects/bot-match/weights/2016-09-18/vgg-11/vgg_11_weights_8.h5')
    vgg.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])
    
    for epoch_index in range(n_epoch):
        for partition_index in range(n_partition):
             
            print('epoch %d, partition %d' % (epoch_index, partition_index))
             
            train_images, train_labels = extract_images(image_preprocessor, partition_index, 'train', avg_rgb)
            validation_images, validation_labels = extract_images(image_preprocessor, partition_index, 'validation', avg_rgb)
             
            train_labels = np_utils.to_categorical(train_labels, n_class)
            validation_labels = np_utils.to_categorical(validation_labels, n_class)
            
            vgg.fit(train_images,
                    train_labels,
                    batch_size = 20,
                    nb_epoch= 1,
                    verbose = 1,
                    validation_data = (validation_images, validation_labels))
             
            print("release memory...")
            del train_images
            del train_labels
             
            del validation_images
            del validation_labels
             
            gc.collect()
            
            if(partition_index % 6 == 5):
                print('saving weights for epoch %d, parittion %d...' % (epoch_index, partition_index))
                vgg.save_weights(('vgg_11_weights_%d_%d.h5' % (epoch_index, partition_index)), overwrite = True)        
                
    print('training finished...')

   
#     data_len = images.shape[0]
#     n_slice = 10
#     slice_len = data_len / n_slice
#     for slice_idx in range(n_slice):
#         if slice_idx < n_slice - 1:
#             images[slice_idx * slice_len:(slice_idx + 1) * slice_len,] = np.divide(images[slice_idx * slice_len:(slice_idx + 1) * slice_len,], 255.0)
#         else:
#             images[slice_idx * slice_len:,] = np.divide(images[slice_idx * slice_len:,], 255.0)
    

