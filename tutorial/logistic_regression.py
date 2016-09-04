import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    
    def __init__(self, input, input_dim, output_dim):
        # initialize with 0 the weights W as a matrix of shape (input_dim, output_dim)
        # TODO understand how to initialize with random value
        self.W = theano.shared(
            value=numpy.zeros((input_dim, output_dim), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of output_dim 0s
        self.b = theano.shared(value=numpy.zeros((output_dim,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        # class probability vector calculated by softmax
        self.likelihood = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # predict the class label by taking the maximum
        self.predicted_label = T.argmax(self.likelihood, axis=1)
        # parameters of the model
        self.parameters = [self.W, self.b]
        # keep track of model input
        self.input = input
        

    def negative_log_likelihood(self, label):
        return -T.mean(T.log(self.likelihood)[T.arange(label.shape[0]), label])


    def errors(self, groud_truth):
        if groud_truth.ndim != self.predicted_label.ndim:
            raise TypeError(
                'groud_truth should have the same shape as self.predicted_label',
                ('groud_truth', groud_truth.type, 'predicted_label', self.predicted_label.type)
            )
            
        if groud_truth.dtype.startswith('int'):
            return T.mean(T.neq(self.predicted_label, groud_truth))
        else:
            raise NotImplementedError()
