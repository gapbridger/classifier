"""
Plan:
1. need to work out how to convert data... 
2. try lenet 5 using theano first, maybe get a smaller set of train / validation data, like 60 % of the data
3. try keras and its model zoo 

"""


from __future__ import print_function

import os, sys, timeit
import six.moves.cPickle as pickle
import numpy
import six.moves.cPickle as pickle
import gzip

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from logistic_regression import LogisticRegression
from multilayer_perceptron import HiddenLayer
from convolutional_neural_network import LeNetConvPoolLayer
from utilities import load_data
from data_preprocess import DataPreProcess

def shared_dataset(data_x, data_y):
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=True)
    
    return shared_x, T.cast(shared_y, 'int32')


def build_cnn_model(image_height, image_width, n_kernel, batch_size, learning_rate, rng):
    # learning_rate, n_kernel, batch_size, print, rng, T,
    print('... building the model')
    x = T.matrix('x', dtype=theano.config.floatX)
    y = T.ivector('y')
    layer_1_input = x.reshape((batch_size, 1, image_height, image_width))
    layer_1 = LeNetConvPoolLayer(rng, input=layer_1_input, image_shape=(batch_size, 1, image_height, image_width), filter_shape=(n_kernel[0], 1, 5, 5), poolsize=(2, 2))
    layer_2 = LeNetConvPoolLayer(rng, input=layer_1.output, image_shape=(batch_size, n_kernel[0], 12, 12), filter_shape=(n_kernel[1], n_kernel[0], 5, 5), poolsize=(2, 2))
#     layer_3 = LeNetConvPoolLayer(rng, input=layer_2.output, image_shape=(batch_size, n_kernel[1], 26, 36), filter_shape=(n_kernel[2], n_kernel[1], 5, 5), poolsize=(2, 2))
    layer_4 = HiddenLayer(rng, input=layer_2.output.flatten(2), n_in=n_kernel[1] * 4 * 4, n_out=batch_size, activation=T.tanh)
    layer_5 = LogisticRegression(input=layer_4.output, input_dim=batch_size, output_dim=10)
    cost = layer_5.negative_log_likelihood(y)
    error = layer_5.errors(y)
#     params = layer_5.params + layer_4.params + layer_3.params + layer_2.params + layer_1.params
    params = layer_5.params + layer_4.params + layer_2.params + layer_1.params
    grads = T.grad(cost, params)
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]
    train_model = theano.function([x, y], cost, updates=updates)
    validation_model = theano.function([x, y], error)
    
    return train_model, validation_model

def train_lenet6():
    # parameters
    learning_rate=0.05
    n_epoch=200
#     batch_size=20
#     image_height = 120
#     image_width = 160
#     n_kernel=[20, 40, 60]
    batch_size=500
    image_height = 28
    image_width = 28
    n_kernel=[20, 40]
    rng = numpy.random.RandomState(23455)
    
#     image_root_dir = '/home/tao/Projects/bot-match/train_data/images/'
#     pickle_root_dir = '/home/tao/Projects/bot-match/train_data/data/120_160'
#     
#     label_map = {'cat': 0, 'chipmunk': 1, 'dog': 2, 'fox': 3, 
#                  'giraffe': 4, 'guinea pig': 5, 'hyena': 6, 'reindeer': 7, 
#                  'sikadeer': 8, 'squirrel': 9, 'weasel': 10, 'wolf': 11}
#     
#     data_preprocessor = DataPreProcess(image_root_dir, pickle_root_dir, label_map, target_height=120, target_width=160)
# 
#     print('loading training data...')
#     train_images, train_labels = data_preprocessor.load_all_data('train')
#     print('training data loaded...')
#     
#     print('loading validation data...')
#     validation_images, validation_labels = data_preprocessor.load_all_data('validation')
#     print('validation data loaded...')
    
    data_path = '/home/tao/Projects/machine-learning/data/mnist.pkl.gz'
    with gzip.open(data_path, 'rb') as f:
        try:
            train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, validation_set, test_set = pickle.load(f)

    train_images, train_labels = train_set
    validation_images, validation_labels = validation_set
    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_images)
    n_validation_batches = len(validation_images)
    n_train_batches //= batch_size
    n_validation_batches //= batch_size


    # start-snippet-1
    train_model, validation_model = build_cnn_model(image_height, image_width, n_kernel, batch_size, learning_rate, rng)


    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 1  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = 1000
    print('validation frequency %d' % validation_frequency)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epoch) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ epoch %d iter %d' % (epoch, iter))
                
#             print('converting current batch of training images and labels')
#             current_train_images = numpy.concatenate(train_images[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], axis=0).astype('float32')
            current_train_images = train_images[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            current_train_labels = train_labels[minibatch_index * batch_size: (minibatch_index + 1) * batch_size].astype('int32')
            # current_train_set_x, current_train_set_y = shared_dataset(numpy.concatenate(current_train_images, axis=0), current_train_labels, borrow=False)
            
            
            cost_ij = train_model(current_train_images, current_train_labels)

            if (iter + 1) % validation_frequency == 0:
                print('validating model')
                # compute zero-one loss on validation set
                validation_losses = numpy.ndarray(shape=(n_validation_batches,), dtype=float)
                for validation_index in range(n_validation_batches):
#                     current_validation_images = numpy.concatenate(validation_images[validation_index * batch_size: (validation_index + 1) * batch_size], axis=0).astype('float32')
                    current_validation_images = validation_images[validation_index * batch_size: (validation_index + 1) * batch_size]
                    current_validation_labels = validation_labels[validation_index * batch_size: (validation_index + 1) * batch_size].astype('int32')
                    validation_losses[validation_index] = validation_model(current_validation_images, current_validation_labels)
                                     
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))
                print('saving model')
                with open('bot.pkl', 'wb') as f:
                        pickle.dump([train_model, validation_model], f)
                        
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    
#     print('Best validation score of %f %% obtained at iteration %i, '
#           'with test performance %f %%' %
#           (best_validation_loss * 100., best_iter + 1, test_score * 100.))
#     print(('The code for file ' +
#            os.path.split(__file__)[1] +
#            ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    label_map = {'cat': 0, 'chipmunk': 1, 'dog': 2, 'fox': 3, 'giraffe': 4, 'guinea pig': 5, 
                 'hyena': 6, 'reindeer': 7, 'sikadeer': 8, 'squirrel': 9, 'weasel': 10, 'wolf': 11}
    
    image_root_dir = '/home/tao/Projects/bot-match/train_data/images/'
    pickle_root_dir = '/home/tao/Projects/bot-match/train_data/data/120_160'
    
    data_preprocessor = DataPreProcess(image_root_dir, pickle_root_dir, label_map, target_height=120, target_width=160)
    
    train_lenet6()
