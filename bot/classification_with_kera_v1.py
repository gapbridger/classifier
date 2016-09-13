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
from keras.datasets import cifar10

# basics
nb_class = 12
nb_epoch = 50
batch_size = 32
data_augmentation = True
channel = 3

# image dimensions
img_height = 64
img_width = 64

# number of convolutional filters
nb_filter1 = 32
nb_filter2 = 32
nb_filter3 = 32
# pooling size for max pooling
nb_pool = 2  # to be adjusted
# convolution kernel size
nb_conv1 = 5
nb_conv2 = 3
nb_conv3 = 3


def net_model(lr = 0.01, decay = 1e-6, momentum = 0.9):
    model = Sequential()

    # convolutional block1
    model.add(Convolution2D(32, 3, 3,
                            border_mode = 'same',
                            input_shape = (channel, img_height, img_width)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    # convolutional block2
    model.add(Convolution2D(64, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # convolutional block3
    # model.add(Convolution2D(128, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(128, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # fully_connected layer
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    # compile
    sgd = SGD(lr = lr, decay = decay, momentum = momentum, nesterov = True)
    model.compile(loss = 'categorical_crossentropy', optimizer=sgd,
                  metrics = ['accuracy'])

    return model

# def net_model2(lr = 0.05, decay = 1e-6, momentum = 0.9):



def train_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train,
              batch_size = batch_size,
              nb_epoch= nb_epoch,
              show_accuracy = True,
              verbose = 1,
              validation_data = (x_val, y_val),
              shuffle = True)
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
        train_data, val_data, test_data = pickle.load(read_file)
        x_train.extend(train_data[0][0:len(train_data[0])])
        y_train.extend(train_data[1][0:len(train_data[0])])
        x_val.extend(val_data[0][0:len(val_data[0])])
        y_val.extend(val_data[1][0:len(val_data[0])])
        x_test.extend(test_data[0][0:len(test_data[0])])
        y_test.extend(test_data[1][0:len(test_data[0])])
        read_file.close()
    return x_train, y_train, x_val, y_val, x_test, y_test



if __name__ == '__main__':

    # change the file path
    data_path = 'D:\Jon\\botchina\\bot_train\\pickle'
    x_train, y_train, \
    x_val, y_val, \
    x_test, y_test = load_data(data_path)

    # reshape for x, binary format for y
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)
    y_train = np_utils.to_categorical(y_train, nb_class)
    y_val = np_utils.to_categorical(y_val, nb_class)
    y_test = np_utils.to_categorical(y_test, nb_class)


    # to be removed to data_process
    # ugly: to be modified and removed
    x_train_nd = np.zeros((len(x_train), channel, img_height, img_width), dtype='float32')
    x_val_nd = np.zeros((len(x_val), channel, img_height, img_width), dtype='float32')
    x_test_nd = np.zeros((len(x_test), channel, img_height, img_width), dtype='float32')
    for i in range(len(x_train)):
        x_train_nd[i, 0] = x_train[i][:, :, 0]
        x_train_nd[i, 1] = x_train[i][:, :, 1]
        x_train_nd[i, 2] = x_train[i][:, :, 2]
    for i in range(len(x_val)):
        x_val_nd[i, 0] = x_val[i][:, :, 0]
        x_val_nd[i, 1] = x_val[i][:, :, 1]
        x_val_nd[i, 2] = x_val[i][:, :, 2]
    for i in range(len(x_test)):
        x_test_nd[i, 0] = x_test[i][:, :, 0]
        x_test_nd[i, 1] = x_test[i][:, :, 1]
        x_test_nd[i, 2] = x_test[i][:, :, 2]


    # # debug
    # for i in range(len(x_train)):
    #     image = np.zeros(img_width, img_height, channel)
    #     image[:, :, 0] = x_train[i, 0, :, :]
    #     image[:, :, 1] = x_train[i, 1, :, :]
    #     image[:, :, 2] = x_train[i, 2, :, :]
    #     plt.imshow(image, cmap='Greys_r')
    #     plt.pause(2)


    x_train = x_train_nd.astype('float32')
    x_val = x_val_nd.astype('float32')
    x_test = x_test_nd.astype('float32')
    print('X_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'val samples')
    print(x_test.shape[0], 'test samples')
    # end



    '''
    nb_classes = 10

    (x_train, y_train), (x_val, y_val) = cifar10.load_data()
    print('X_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_val = np_utils.to_categorical(y_val, nb_classes)
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    '''

    x_train /= 255
    x_val /= 255
    # x_test /= 255


    model = net_model()
    # training
    # comment when doing testing
    train_model(model, x_train, y_train, x_val, y_val)
    # score = test_model(model, x_test, y_test)

    # # testing
    # model.load_weights('model_weights.h5')
    # classes = model.predict_classes(x_test, verbose=0)
    # test_accuracy = np.mean(np.equal(y_test, classes))
    # print("accuracy: ", test_accuracy)

