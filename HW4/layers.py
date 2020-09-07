from builtins import range
import numpy as np
import math


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in the variable out. #
    ###########################################################################
    x1=x.reshape(x.shape[0], -1)
    out = np.dot(x1, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x1=x.reshape(x.shape[0], -1)
    dx = np.dot(dout, w.T).reshape(x.shape)

    dw = np.dot(x1.T, dout)
    db = np.sum(dout, axis = 0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout
    x = cache
    #dx = dx.reshape(-1,1)
    A = np.maximum(0,x)
    A[A>0] = 1
    dx = np.multiply(A, dx)   

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height H' and width W'. 
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, H', W')
    Returns a tuple of:
    - out: Output data, of shape (N, F, HH, WW) where H' and W' are given by
      HH = H - H' + 1
      WW = W - W' + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass based on the definition  #
    # of Y in Q1(c).                                                          #
    ###########################################################################
    N, C, H, W = x.shape
    F, _, Hw, Ww = w.shape
    HH = H - Hw + 1  
    WW = W - Ww + 1
    out = np.zeros((N, F, HH, WW))

    '''for n in range(N):
    	for f in range(F):
    		for h_out in range(HH):
    			for w_out in range(WW):
    				out[n, f, h_out, w_out] = np.sum(x[n, :, h_out:h_out+HH-1, w_out:w_out+WW-1]*w[f, :])'''
    xlocal = np.expand_dims(x, axis = 1)
    wlocal = np.expand_dims(w, axis = 0)
    for i in range(HH):
        for j in range(WW):
            out[:, :, i, j] = np.sum(xlocal[:,:,:, i: i+ Hw, j:j+Ww]*wlocal, axis = (2,3,4))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    N, F, HH, WW = dout.shape
    (x, w) = cache
    _, C, Hw, Ww = w.shape
    _, _, H, W = x.shape
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    wlocal = np.expand_dims(w, axis = 0)
    xlocal = np.expand_dims(x, axis = 1)
    doutlocal = np.expand_dims(dout, axis = 2)
    npad = ((0, 0), (0,0), (0,0), (HH-1, HH-1), (WW-1, WW-1))
    wpad = np.pad(wlocal, pad_width=npad, mode='constant', constant_values=0)
    dflip = np.flip(doutlocal, (3,4))


    for i in range(Hw):
        for j in range(Ww):
            dw[:, :, i, j] = np.sum(xlocal[:,:,:, i: i+ HH, j:j+WW]*doutlocal, axis = (0,3,4))
    for i in range(H):
        for j in range(W):
            dx[:, :, i, j] = np.sum(wpad[:,:,:, i: i+ HH, j:j+WW]*dflip, axis = (1,3,4))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here. Output size is given by 
    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    height = pool_param['pool_height']
    width = pool_param['pool_width']
    stride = pool_param['stride']
    HH = 1+(H-height)//stride
    WW = 1+(W-width)//stride
  
    out=np.zeros((N,HH,WW,C))
    x1=x.reshape(N,H,W,C).copy()
    
    for i in range(HH):
        start = i * stride
        end = start + height

        for j in range(WW):
            h_begin = j * stride
            h_end = h_begin + width
            prevslice_a = x1[:, start:end, h_begin:h_end, :]
            out[:, i, j, :] = np.max(prevslice_a, axis=(1, 2))

    out=out.reshape(N, C, HH, WW)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    height = pool_param['pool_height']
    width = pool_param['pool_width']
    stride = pool_param['stride']
    HH = ((H-height)//stride) + 1
    WW = ((W-width)//stride) + 1
  
    dx=np.zeros((N,H,W,C))
    
    x1=x.reshape(N,H,W,C).copy()
    doutlocal=dout.reshape(dout.shape[0],dout.shape[2],dout.shape[3],dout.shape[1]).copy()

    xx=-1
    yy=-1
    for i in range(HH):
        start = i * stride
        end = start + height
        yy+=1

        for j in range(WW):
            h_begin = j * stride
            h_end = h_begin + width
            xx +=1
            
            x2 = x1[:, start:end, h_begin:h_end, :].copy()
            mask = np.zeros_like(x2)
            reshaped_x = x2.reshape(x2.shape[0], x2.shape[1] * x2.shape[2], x2.shape[3])
            idx = np.argmax(reshaped_x, axis=1)
            
            ax1, ax2 = np.indices((x2.shape[0], x2.shape[3]))
            mask.reshape(mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3])[ax1, idx, ax2] = 1

            dx[:, start:end, h_begin:h_end, :] += doutlocal[:, i:i+1, j:j+1, :] * mask
        xx = -1
            
    dx=dx.reshape(N,C,H,W)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  loss, dx = None, None
  x_max = np.max(x, axis = 1, keepdims = True)
  p = np.exp(x - x_max)
  p = p/np.sum(p, axis = 1, keepdims = True)
  loss = 0
    #dx = np.zeros(p.shape[0])
  N = x.shape[0]
  '''for i in range(N):
      loss =- y[i]*np.log(p[i])
      dx[i] = y[i] - p[i]'''
  loss = -np.sum(np.log(p[range(N), y])) / N
    #m = y.shape[0]
  grad = p.copy()
  grad[range(N),y] -= 1
  grad = grad/N
  dx = grad
  return loss, dx
