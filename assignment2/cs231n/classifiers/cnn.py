from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale*np.random.randn(num_filters*H*W//4,hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b3'] = np.zeros(num_classes)
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        fs = self.filter_size
        conv_param = {'stride': 1, 'pad': (fs - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        N,C,H,W = X.shape
        F = W1.shape[0]
        st1 = conv_param['stride']
        p = conv_param['pad']
        st2 = pool_param['stride']
        ph = pool_param['pool_height']
        pw = pool_param['pool_width']
        H1 = 1 + (H + 2*p - fs)//st1
        L1 = 1 + (W + 2*p - fs)//st1
        xpad = np.pad(X,((0,0),(0,0),(p,p),(p,p)),'constant',constant_values=0)
        conv_out = np.zeros((N,F,H1,L1))
        for n in range(N):
            for f in range(F):
                for i in range(H1):
                    for j in range(L1):
                        v = xpad[n,:,i*st1:i*st1+fs,j*st1:j*st1+fs]*W1[f,:,:,:]
                        g = v.sum() + b1[f]
                        conv_out[n,f,i,j] = g
                        
        conv_relu_out = np.maximum(conv_out,0)
        h1 = H//2
        w1 = W//2
        maxpool_out = np.zeros((N,F,h1,w1))
        for n in range(N):
            for f in range(F):
                for i in range(h1):
                    for j in range(w1):
                        v = np.max(conv_relu_out[n,f,i*st2:i*st2+pw,j*st2:j*st2+ph])
                        maxpool_out[n,f,i,j] = v
                        
        affine_in = np.reshape(maxpool_out,(maxpool_out.shape[0],-1))
        first_layer = affine_in.dot(W2) + b2
        first_layer_relu = np.maximum(first_layer,0)
        scores = first_layer_relu.dot(W3) + b3
        
         
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        scores = np.exp(scores)
        correct_scores = scores[list(range(N)),y]
        scores_sum = np.sum(scores,axis=1)
        loss = -np.sum(np.log(correct_scores/scores_sum))/N
        loss += self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))/N
        dout = scores/scores_sum.reshape(N,1)
        dout[list(range(N)),y] -= 1
        db3 = np.sum(dout,axis=0)/N
        dW3 = first_layer_relu.T.dot(dout)/N + self.reg*(W3)
        dfirstlayer = dout.dot(W3.T)
        dfirstlayer[first_layer_relu==0] = 0
        db2 = np.sum(dfirstlayer,axis=0)/N
        dW2 = affine_in.T.dot(dfirstlayer)/N + self.reg*(W2)
        daff = dfirstlayer.dot(W2.T)
        dmaxpool = daff.reshape(maxpool_out.shape)
        
        dconv = np.zeros(conv_out.shape)
        for n in range(N):
            for f in range(F):
                for i in range(h1):
                    for j in range(w1):
                        ind = np.argmax(conv_relu_out[n,f,i*st2:i*st2+pw,j*st2:j*st2+ph])
                        ind1,ind2 = np.unravel_index(ind,(pw,ph))
                        v = dmaxpool[n,f,i,j]
                        dconv[n,f,i*st2:i*st2+pw,j*st2:j*st2+ph][ind1,ind2] = v
                        
        dconv[conv_relu_out==0] = 0
        dW1 = np.zeros(W1.shape)
        db1 = np.zeros(b1.shape)
        for n in range(N):
            for f in range(F):
                db1[f] += dconv[n,f].sum()
                for i in range(H1):
                    for j in range(L1):
                        v = xpad[n,:,i*st1:i*st1+fs,j*st1:j*st1+fs]*dconv[n,f,i,j]
                        dW1[f,:,:,:] += v
                    
        dW1 /= N
        db1 /= N
        dW1 += self.reg*(W1)
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3
        
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
