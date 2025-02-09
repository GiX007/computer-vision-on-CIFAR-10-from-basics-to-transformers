from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    # Initialize the loss and gradient as zero
    loss = 0.0
    dW = np.zeros(W.shape)  # Gradient of the weights, initialize to zero

    # Dimensions
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # Forward pass: loop over training examples
    for i in range(num_train):
        
        # Compute scores
        scores = np.dot(W.T, X[i])
        
        # Loop over all classes
        for j in range(num_classes):
            
            # For every j != y(i)
            if j != y[i]:   
                # Compute total scores
                total_loss_score = scores[j] - scores[y[i]] + 1    
                
                # Loss is max(0, total_loss_score)
                if total_loss_score>0:
                    loss += total_loss_score

                    # Backward pass with gradient calculations 
                    dW[:,j] += X[i]     # (incorrect class)
                    dW[:,y[i]] -= X[i]  # (correct class)

    # Final loss function with regularization
    loss /= num_train
    loss += reg * np.sum(W**2)
    
    # Gradient calculation with regularization
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # Initialize the gradient as zero

    # Forward pass:

    # Compute all scores
    scores = X.dot(W)  # shape of (X.shape[0], W.shape[1]) -> (500, 10)
    
    # Scores for given y(correct classes)
    scores_y = scores[np.arange(X.shape[0]), y].reshape(-1, 1)  # shape of (500,1)
    
    # Compute the total_score
    total_score = np.maximum(0, scores - scores_y + 1)  # (500,10) - (500,1) = (500,10) with broadcasting
    
    # Exclude correct class score total_score
    total_score[np.arange(X.shape[0]), y] = 0  # set total_score for the correct class to 0
    
    # Compute the loss
    loss = np.sum(total_score) / X.shape[0] + reg * np.sum(W * W)

    # 2. Backward pass
    
    # Create a mask for the '+1, -1 condition'
    mask = np.zeros(total_score.shape)
    mask[total_score > 0] = 1  # Set the total_score greater than 0 to 1
    
    # Calculate the number of positive margins for each example
    num_positive_margins = np.sum(mask, axis=1)
    
    # Adjust the mask for the correct class
    mask[np.arange(X.shape[0]), y] -= num_positive_margins
    
    # Compute gradient dW with regularization
    dW = X.T.dot(mask) / X.shape[0] + 2 * reg * W

    return loss, dW