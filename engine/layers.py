from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    # Reshape the input x(N, d1, d2, ...) -> (N, D).
    x_ = x.reshape(x.shape[0], -1)
    
    # Compute the output of shape (N, M).
    out = x_.dot(w) + b 
    
    #print(out, out.shape)
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    # Reshape the input x(N, d1, d2, ...) -> (N, D).
    x_ = x.reshape(x.shape[0], -1)
    
    # Compute gradients (dy = dout).
    dx = dout.dot(w.T).reshape(x.shape) # Reshape (10,2,3)->(10,6).
    dw = x_.T.dot(dout)
    db = np.sum(dout, axis=0)
    #print("x_ shape:", x_.shape, "dout shape:", dout.shape, "w shape:", w.shape)
    #print("dx_ shape:", dx.shape, "dw shape:", dw.shape, "b shape:", db.shape)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    # Relu activation is max(0, x).
    out = np.maximum(0, x)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    # dx = dout*1 if x > 0
    dx = dout * (x > 0)

    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    # Number of training examples.
    num_train = x.shape[0]
    
    # Compute the scores.
    correct_class_scores = x[np.arange(num_train), y].reshape(-1, 1)
    
    # Compute the margins.
    margins = np.maximum(0, x - correct_class_scores + 1)
    margins[np.arange(num_train), y] = 0    # no contribution from correct classes scores in loss.
    
    # Compute the loss.
    loss = np.sum(margins) / num_train
    
    # Compute the gradient.
    margins[margins > 0] = 1
    row_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] -= row_sum
    dx = margins / num_train

    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    # Stabilize the logits by shifting by max(score) to prevent overflow.
    logits = x - np.max(x, axis=1, keepdims=True)
    
    # Convert logits to softmax probabilities.
    exp_logits = np.exp(logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Compute cross-entropy loss.
    correct_log_probs = -np.log(softmax_probs[np.arange(x.shape[0]), y]) # Correct class probabilities.
    loss = np.sum(correct_log_probs) / x.shape[0] # Average over the total x examples.
    
    # Compute the gradient
    softmax_probs[np.arange(x.shape[0]), y] -= 1  # -1 from correct classes probabilities.
    dx = softmax_probs / x.shape[0]  # Average over the total x examples.

    return loss, dx