from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    # Dimensions
    num_train = X.shape[0]
    num_classes = W.shape[1]

    # Loop over all training samples
    for i in range(num_train):
        
        # Compute scores and stabilize them by subtracting the max score for numerical stability
        scores = X[i].dot(W)
        scores -= np.max(scores)  # subtract the max score to prevent numerical issues
        
        # Compute softmax probabilities
        probs = np.exp(scores) / np.sum(np.exp(scores))
        
        # Compute the loss: the negative log likelihood of the correct class
        loss += -np.log(probs[y[i]])
        
        # Compute the gradient for each class
        for j in range(num_classes):
            # Update gradient
            if j == y[i]:
                dW[:, j] += (probs[j] - 1) * X[i]
            else:
                dW[:, j] += probs[j] * X[i]

    # Average loss over the batch
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    num_train = X.shape[0]
    num_classes = W.shape[1]

    # Compute scores
    scores = X.dot(W)
    
    # Stabilize scores to avoid numerical instability by subtracting max score
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # Compute softmax probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute loss with regularization
    correct_log_probs = -np.log(probs[np.arange(num_train), y])
    loss = np.sum(correct_log_probs) / num_train + reg * np.sum(W * W)
    
    # Compute the gradient
    probs[np.arange(num_train), y] -= 1  # Adjust probs for correct classes( it's (probs-1)*X )
    dW = X.T.dot(probs) / num_train + 2 * reg * W

    return loss, dW