"""
Plan: 
1. try lenet 5 using theano first, maybe get a smaller set of train / validation data, like 60 % of the data
2. try keras and its model zoo 
"""

"""
CNN model:
1. Train the model based on data set
2. Predict the animals, give the probability distribution
"""
from __future__ import print_function
import numpy as np
np.random.seed(1234)

import os
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


# basics
nb_class = 12
nb_epoch = 15
batch_size = 100

# image dimensions
img_height = 180
img_width = 240

# number of convolutional filters
nb_filter1 = 4
nb_filter2 = 8
nb_filter3 = 16
# pooling size for max pooling
nb_pool = 2  # to be adjusted
# convolution kernel size
nb_conv1 = 25
nb_conv2 = 15
nb_conv3 = 15


def net_model(lr = 0.05, decay = 1e-6, momentum = 0.9):
    model = Sequential()

    # convolutional layer 1
    model.add(Convolution2D(nb_filter1, nb_conv1, nb_conv1,
                            border_mode = 'valid',
                            input_shape = (1, img_width, img_height)))
    model.add(Activation('tanh'))
    # model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool)))
    model.add(Dropout(0.5))   # can be comment

    # convolutional layer 2
    model.add(Convolution2D(nb_filter2, nb_conv2, nb_conv2))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    # convolutional layer 3
    model.add(Convolution2D(nb_filter3, nb_conv3, nb_conv3))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    # fully_connected layer
    model.add(Flatten())
    model.add(Dense(128, init = 'normal'))
    model.add(Activation('tanh'))

    # softmax
    model.add(Dense(nb_class, init = 'normal'))
    model.add(Activation('softmax'))

    # compile
    sgd = SGD(lr = lr, decay = decay, momentum = momentum, nesterov = True)
    model.compile(loss = 'categorical_crossentropy', optimizer=sgd,
                  metrics = ['accuracy'])

    return model

def train_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train,
              batch_size = batch_size,
              nb_epoch= nb_epoch,
              show_accuracy = True,
              verbose = 1,
              validation_data = (x_val, y_val))
    model.save_weights('model_weights.h5', overwrite = True)
    return model

def test_model(model, x, y):
    model.load_weights('model_weights.h5')
    score = model.evaluate(x, y, show_accuracy = True, verbose = 0)
    print('Test score: ', score[0])
    print('Test accuracy', score[1])
    return score


def load_data(dataset_path):
    animal_types = ['cat', 'chipmunk', 'Dog', 'fox',
                    'giraffe', 'guinea pig', 'hyena', 'reindeer',
                    'sikadeer', 'squirrel', 'weasel', 'wolf']
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    for i in range(12):
        pickle_name = 'labeled_raw_data_' + animal_types[i]
        pickle_name = pickle_name + '.pkl'
        subdir = os.path.join(dataset_path, pickle_name)
        read_file = open(subdir, 'rb')
        output = pickle.load(read_file)
        x_train.extend(output[0][0])
        y_train.extend(output[0][1])
        x_val.extend(output[1][0])
        y_val.extend(output[1][1])
        x_test.extend(output[2][0])
        y_test.extend(output[2][1])
        read_file.close()
    return x_train, y_train, x_val, y_val, x_test, y_test



if __name__ == '__main__':
    # change the file path
    data_path = 'C:\\zzdata\\bot_train\pickle'
    x_train, y_train, \
    x_val, y_val, \
    x_test, y_test = load_data(data_path)

    x_train = np.asarray(x_train).astype(float)
    y_train = np.asarray(y_train)
    x_val = np.asarray(x_val).astype(float)
    y_val = np.asarray(y_val)
    x_test = np.asarray(x_test).astype(float)
    y_test = np.asarray(y_test)

    # reshape for x, binary format for y
    x_train = x_train.reshape(x_train.shape[0], 1, img_width, img_height)
    x_val = x_val.reshape(x_val.shape[0], 1, img_width, img_height)
    x_test = x_test.reshape(x_test.shape[0], 1, img_width, img_height)
    y_train = np_utils.to_categorical(y_train, nb_class)
    y_val = np_utils.to_categorical(y_val, nb_class)
    y_test = np_utils.to_categorical(y_test, nb_class)


    model = net_model()
    # training
    # comment when doing testing
    train_model(model, x_train, y_train, x_val, y_val)
    score = test_model(model, x_test, y_test)

    # testing
    model.load_weights('model_weights.h5')
    classes = model.predict_classes(x_test, verbose = 0)
    test_accuracy = np.mean(np.equal(y_test, classes))
    print("accuracy: ", test_accuracy)


