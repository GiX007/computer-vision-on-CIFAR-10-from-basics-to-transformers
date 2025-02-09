from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        # First layer initialization.
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale   
        self.params['b1'] = np.zeros(hidden_dim)

        # Second layer initialization.
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        # Get the params.
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        reg = self.reg

        # Flatten the input.
        X_ = X.reshape(X.shape[0], -1)
        
        # Forward pass: input->affine->relu->affine->and then to softmax.
        first_hidden_out, cache1 = affine_forward(X_, W1, b1)
        relu_out, relu_cache = relu_forward(first_hidden_out)
        scores, cache2 = affine_forward(relu_out, W2, b2)

        # If y is None then we are in test mode so just return scores.
        if y is None:
            return scores

        loss, grads = 0, {}

        # Compute the loss.
        loss, dy = softmax_loss(scores, y) 
        
        # Add regularization to the loss.
        loss +=  0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
        
        # Backward step:
        
        # Backprop through the second affine layer.
        final_dout, grads['W2'], grads['b2'] = affine_backward(dy, cache2)
        grads['W2'] += reg * W2 # Add regularization to W2.
        
        # Backprop through the relu laayer.
        to_first_layer = relu_backward(final_dout, relu_cache)
        
        # Backprop through first affine layer.
        _, grads['W1'], grads['b1'] = affine_backward(to_first_layer, cache1)
        grads['W1'] += reg * W1 # Add regularization to W1.
        
        return loss, grads