from __future__ import print_function
import numpy as np
import copy
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model2 = copy.deepcopy(model)

# train on the whole data
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, verbose=0)


# train on batch
def ImageNet():
    """
    # image generator
    """
    # loading data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = np_utils.to_categorical(y_train, 10)
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # divide the dataset into two parts (as if loaded from two pickles)
    X_train_pk = []
    X_train_pk.append(X_train[0:len(X_train)/3*2])
    X_train_pk.append(X_train[len(X_train)/3*2+1:])
    y_train_pk = []
    y_train_pk.append(y_train[0:len(y_train)/3*2])
    y_train_pk.append(y_train[len(y_train)/3*2+1:])
    nb_sample = len(X_train)
    chunk_size = 10000

    while 1:   #  to be decide by data size
        for i in range(nb_sample/chunk_size):  # 60000/10000
            if i<4:
                yield X_train_pk[0][chunk_size*i:chunk_size*(i+1)], y_train_pk[0][chunk_size*i:chunk_size*(i+1)]   # load pickle
            else:
                j = i-4
                yield X_train_pk[1][chunk_size*j:chunk_size*(j+1)], y_train_pk[1][chunk_size*j:chunk_size*(j+1)]


datagen = ImageDataGenerator(
        featurewise_center=True, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False) # randomly flip images

idx = range(60000)
np.random.shuffle(idx)
X_sample = X_train[idx[1:1000]]
datagen.fit(X_sample) # let's say X_sample is a small-ish but statistically representative sample of your data

# With data augmentation

for X_train, Y_train in ImageNet(): # load a chunk of pictures first
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                            samples_per_epoch = X_train.shape[0],
                            nb_epoch = nb_epoch,
                            validation_data = (X_test, Y_test)
                            )#  pick a batch from the chunk, and do augmentation



# Alternatively, without data augmentation
# for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
#     model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)


