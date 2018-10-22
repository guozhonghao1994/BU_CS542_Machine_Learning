#import cPickle as pickle
import numpy as np
import os
from math import sqrt, ceil
from random import randrange
from scipy.misc import imread
from sklearn.datasets import fetch_mldata

def get_MNIST_data(num_training=50000, num_validation=10000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # fix random seed
    np.random.seed(0)

    # Load the raw MNIST data
    mnist = fetch_mldata('MNIST-original', data_home='datasets/mnist')
    X = mnist['data'].astype(float)
    y = mnist['target'].astype(int)
    I = np.random.permutation(X.shape[0])
    X = X[I, :]
    y = y[I]
        
    # Subsample the data
    mask = range(num_training)
    X_train = X[mask]
    y_train = y[mask]
    mask = range(num_training, num_training + num_validation)
    X_val = X[mask]
    y_val = y[mask]
    num_trainval = num_training + num_validation
    mask = range(num_trainval, num_trainval + num_test)
    X_test = X[mask]
    y_test = y[mask]

    # Normalize the data: subtract the mean
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Package data into a dictionary
    return X_train, y_train, X_val, y_val, X_test, y_test


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 
  fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[ix])
    it.iternext() # step to next dimension

  return grad


def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 4D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (N, H, W, C)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  try:
      (N, H, W, C) = Xs.shape
  except:
      (N, H, W) = Xs.shape
      C = 1
  grid_size = int(ceil(sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  if C == 1:
      grid = np.zeros((grid_height, grid_width))
  else:
      grid = np.zeros((grid_height, grid_width, C))
  next_idx = 0
  y0, y1 = 0, H
  for y in range(grid_size):
    x0, x1 = 0, W
    for x in range(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  # grid_max = np.max(grid)
  # grid_min = np.min(grid)
  # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
  return grid
