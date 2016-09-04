from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from logistic_regression import LogisticRegression
from utilities import load_data




def stochastic_gradient_descent_mnist(learning_rate=0.13, 
                                      n_epochs=1000, 
                                      path='/home/tao/Projects/machine-learning/data/mnist.pkl.gz', 
                                      batch_size=600):
    
    datasets = load_data(path)

    train_set_data, train_set_label = datasets[0]
    validation_set_data, validation_set_label = datasets[1]
    test_set_data, test_set_label = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_data.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = validation_set_data.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_data.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    index = T.lscalar()  # index to a [mini]batch

    data = T.matrix('x')  # data, presented as rasterized images
    label = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    classifier = LogisticRegression(input=data, input_dim=28 * 28, output_dim=10)

    objective_function = classifier.negative_log_likelihood(label)

    # testing model
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(label),
        givens={
            data: test_set_data[index * batch_size: (index + 1) * batch_size],
            label: test_set_label[index * batch_size: (index + 1) * batch_size]
        }
    )
    # validation model
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(label),
        givens={
            data: validation_set_data[index * batch_size: (index + 1) * batch_size],
            label: validation_set_label[index * batch_size: (index + 1) * batch_size]
        }
    )

    # gradients
    g_W = T.grad(cost=objective_function, wrt=classifier.W)
    g_b = T.grad(cost=objective_function, wrt=classifier.b)

    # update rule
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

   # training model
    train_model = theano.function(
        inputs=[index],
        outputs=objective_function,
        updates=updates,
        givens={
            data: train_set_data[index * batch_size: (index + 1) * batch_size],
            label: train_set_label[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    # go through this many minibatche before checking the network on the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)] # grammar sugar
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % 
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        
                    best_validation_loss = this_validation_loss
                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f %%, with test performance %f %%' % (best_validation_loss * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


def predict():
    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.predicted_label)

    # We can test it on some examples from test test
    dataset='/home/tao/Projects/machine-learning/data/mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
    print("Ground truth label values for the first 10 examples in test set:")
    print(test_set_y.eval()[:10])

if __name__ == '__main__':
    stochastic_gradient_descent_mnist()
    # predict()