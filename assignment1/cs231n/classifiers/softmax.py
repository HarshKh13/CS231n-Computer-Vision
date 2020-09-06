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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        score = X[i].dot(W)
        score -= score.max()
        score = np.exp(score)
        loss += -np.log(score[y[i]]/np.sum(score,axis=0))
        for j in range(num_classes):
            if(j==y[i]):
                grad = -X[i] + (X[i]*score[j])/np.sum(score,axis=0)
                dW[list(range(W.shape[0])),j] += grad
                
            else:
                grad = (X[i]*score[j])/np.sum(score,axis=0)
                dW[list(range(W.shape[0])),j] += grad
            
    loss /= num_train
    dW /= num_train
    loss += reg*np.sum(W*W)
    dW += 2*reg*W
            
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= scores.max()
    scores_ex = np.exp(scores)
    scores_sum = np.sum(scores_ex,axis=1)
    correct_score = scores_ex[list(range(num_train)),y]
    loss = correct_score/scores_sum
    loss = -(np.log(loss)).sum()/num_train
    scores_ex = scores_ex/np.sum(scores_ex,axis=1).reshape(num_train,1)
    scores_ex[list(range(num_train)),y] -= 1
    dW = X.T.dot(scores_ex)
    dW /= num_train
    loss += reg*np.sum(W*W)
    dW += 2*reg*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
