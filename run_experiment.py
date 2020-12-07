# Written by Thomas Noel
# for CS 539 Convex Optimization
#
# SVRG Algorithm is described in "Accelerating Stochastic
# Gradient Descent using Predictive Variance Reduction" by
# Johnson and Zhang


import mnist_reader
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def sgd(batch_size: int, lr_init: float = 1e-3) -> np.ndarray:
    ''' Implements SGD using mini-batch stochastic vector.

    Args:
        batch_size -- The number of training examples used per batch
        lr_init    -- The initial learning rate (default 1e-3)

    Returns:
        The model's weights.   
    '''
    # Using a t-inverse learning rate schedule


def svrg(update_freq: int, learning_rate: float = 1e-3) -> np.ndarray:
    ''' Implements SVRG as described by Johnson and Zhang (2013).
    
    Args:
        update_freq -- The number of iterations between each average gradient
                       update.
        learning_rate -- The fixed learning rate of the SVRG algorithm

    Returns:
        The model's weights

    .. R. Johnson, T. Zhang. Accelerating Stochastic Gradient Descent using
           Predictive Variance Reduction. NIPS, 2013.
    '''
    # Initialize \tilde{w}_0: sgd with a certain number of iterations

    print("The learning rate is {}".format(learning_rate))
    return np.array([[1, 0], [0, 1]])


def main():
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')    
    svrg(10)

if __name__ == '__main__':
    main()
