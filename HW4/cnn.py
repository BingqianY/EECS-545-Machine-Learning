import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax
  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights for the convolutional layer using the keys 'W1' (here      #
    # we do not consider the bias term in the convolutional layer);            #
    # use keys 'W2' and 'b2' for the weights and biases of the                 #
    # hidden fully-connected layer, and keys 'W3' and 'b3' for the weights     #
    # and biases of the output affine layer.                                   #
    ############################################################################
    C,H,W = input_dim

    self.params['W1'] = np.random.normal(scale = weight_scale, size = (num_filters,C,filter_size, filter_size))
    
    Hh = (H - filter_size) + 1 #stride is taken as 1
    Ww = (W - filter_size) + 1
    #after the pooling operation sizes
    H1 = (Hh - 2) // 2 + 1 #since 2x2 max pooling
    W1 = (Ww - 2) // 2 + 1

    rescaled_size = num_filters * H1 * W1

    self.params['W2'] = np.random.normal(scale = weight_scale, size = (rescaled_size, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    """
    W1 = self.params['W1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    a, convfcache = conv_forward(X, W1)
    s, relucache = relu_forward(a)
    conv_op, pool_cache = max_pool_forward(s, pool_param)
    cache_1 = (convfcache, relucache, pool_cache)
    a, fc_cache = fc_forward(conv_op, W2, b2)
    rl_output, relucache = relu_forward(a)
    cache_2 = (fc_cache, relucache)
    #afn_rl_output, affn_rl_cache = affine_relu_forward(conv_output, W2, b2)
    scores, affine_cache = fc_forward(rl_output, W3, b3)
    #scores, affine_cache = fc_forward(conv_output, W3, b3)

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
    # for self.params[k].                                                      #
    ############################################################################
    loss, dout = softmax_loss(scores, y)

    dout, grads["W3"], grads["b3"] = fc_backward(dout, affine_cache)

    fc_cache, relu_cache = cache_2
    da = relu_backward(dout, relu_cache)
    dout, grads["W2"], grads["b2"] = fc_backward(da, fc_cache)
    #dout, grads["W2"], grads["b2"] = affine_relu_backward(dout, cache2)
    #loss += 0.5 * self.reg * np.sum(W2 * W2)
    #grads["W2"] += self.reg * W2

    conv_cache, relu_cache, pool_cache = cache_1
    ds = max_pool_backward(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dout, grads["W1"] = conv_backward(da, conv_cache)
    #dout, grads["W1"], grads["b1"] = conv_relu_pool_backward(dout, cache1)
    #loss += 0.5 * self.reg * np.sum(W1 * W1)
    #grads["W1"] += self.reg * W1


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

