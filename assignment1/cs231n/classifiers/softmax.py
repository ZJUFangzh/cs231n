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
  (N, D) = X.shape
  C = W.shape[1]
  #遍历每个样本
  for i in range(N):
    f_i = X[i].dot(W)
    #进行公式的指数修正
    f_i -= np.max(f_i)
    sum_j = np.sum(np.exp(f_i))
    #得到样本中每个类别的概率
    p = lambda k : np.exp(f_i[k]) / sum_j
    loss += - np.log(p(y[i]))
    #根据softmax求导公式
    for k in range(C):
      p_k = p(k)
      dW[:, k] += (p_k - (k == y[i])) * X[i]
  
  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW /= N
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  (N, D) = X.shape
  C = W.shape[1]
  f = X.dot(W)
  #在列方向进行指数修正
  f -= np.max(f,axis=1,keepdims=True)
  #求得softmax各个类的概率
  p = np.exp(f) / np.sum(np.exp(f),axis=1,keepdims=True)
  y_lable = np.zeros((N,C))
  #y_lable就是(N,C)维的矩阵，每一行中只有对应的那个正确类别 = 1，其他都是0
  y_lable[np.arange(N),y] = 1
  #cross entropy
  loss = -1 * np.sum(np.multiply(np.log(p),y_lable)) / N
  loss += 0.5 * reg * np.sum( W * W)
  #求导公式，很清晰
  dW = X.T.dot(p-y_lable)
  dW /= N
  dW += reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

