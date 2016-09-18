"""
CNN model:
1. Train the model based on data set
2. Predict the animals, give the probability distribution



"""
from __future__ import print_function
import numpy as np
from skimage.util.dtype import img_as_ubyte
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
from keras.preprocessing.image import ImageDataGenerator
from skimage import img_as_ubyte
from image_preprocess import ImagePreprocess
from image_preprocess import read_cifar_10
import gc


def vgg_net_16(n_class, weights_path=None):
    
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

    if weights_path:
        model.load_weights(weights_path)
        
    return model


def vgg_net_11(n_class, weights_path=None):
    
    rectified_linear_unit = 'relu'
    softmax = 'softmax'
    
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation=rectified_linear_unit, name='conv1_1', dim_ordering='tf'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation=rectified_linear_unit, name='conv2_1', dim_ordering='tf'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation=rectified_linear_unit, name='conv3_1', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation=rectified_linear_unit, name='conv3_2', dim_ordering='tf'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv4_1', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv4_2', dim_ordering='tf'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv5_1', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation=rectified_linear_unit, name='conv5_2', dim_ordering='tf'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='tf'))
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


def extract_images(image_preprocessor, partition_index, data_type):
    image_list, labels = image_preprocessor.load_data_partition(partition_index, data_type)
    return image_list, labels
#     image_list = image_list.astype('float32')
#     image_list = (image_list.astype('float32') - avg_rgb)
#     image_list = np.divide(image_list, 255.0)
#     image_list = np.rollaxis(image_list, 3, 1)
    


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
    
    avg_rgb = np.asarray([132.4509, 123.5161, 105.4855], dtype=float)
    
    image_preprocessor = ImagePreprocess(input_root_dir, output_root_dir)
    
    # averaged over training set
    sgd = SGD(lr=0.008, decay=1e-6, momentum=0.9, nesterov=True)
    vgg = vgg_net_11(n_class)
    vgg.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])
    
    train_data_gen = ImageDataGenerator(rotation_range=10.0,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        shear_range=0.,
                                        zoom_range=0.,
                                        horizontal_flip=True,
                                        rescale=1.0/255.0,
                                        dim_ordering='tf')  # randomly flip images
    
    validation_data_gen = ImageDataGenerator(rotation_range=10.0,
                                       width_shift_range=0.0,
                                       height_shift_range=0.0,
                                       shear_range=0.,
                                       zoom_range=0.,
                                       horizontal_flip=True,
                                       rescale=1.0/255.0,
                                       dim_ordering='tf')  # randomly flip images
    
    
    for epoch_index in range(4,n_epoch):
        for partition_index in range(n_partition):
            print('epoch %d, partition %d' % (epoch_index, partition_index))
            print('loading images...') 
            train_images, train_labels = extract_images(image_preprocessor, partition_index, 'train')
            train_labels = np_utils.to_categorical(train_labels, n_class)
            
            validation_images, validation_labels = extract_images(image_preprocessor, partition_index, 'validation')
            validation_labels = np_utils.to_categorical(validation_labels, n_class)
            print('cropping images...')
            train_images = image_preprocessor.crop_images(train_images, mode='random')
            train_images = np.stack(train_images, axis=0)
            validation_images = image_preprocessor.crop_images(validation_images, mode='random')
            validation_images = np.stack(validation_images, axis=0)
            print('image crop finished...')
                        
            train_data_gen.fit(train_images)
            validation_data_gen.fit(validation_images)
            
            vgg.fit_generator(generator=train_data_gen.flow(train_images, train_labels, batch_size=32), 
                              samples_per_epoch=train_images.shape[0], 
                              nb_epoch=1, 
                              verbose=1, 
                              validation_data=validation_data_gen.flow(validation_images, validation_labels, batch_size=32), 
                              nb_val_samples=validation_images.shape[0])
            
#             vgg.fit(train_images, 
#                 train_labels, 
#                 batch_size = 32, 
#                 nb_epoch= 1, 
#                 verbose = 1, 
#                 validation_data = (validation_images, validation_labels))
             
            print("release memory...")
            del train_images
            del train_labels
             
            del validation_images
            del validation_labels
             
            gc.collect()
            
            if partition_index % 6 == 5:
                print('saving weights for epoch %d partition index %d...' % (epoch_index, n_partition))
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
    
       
