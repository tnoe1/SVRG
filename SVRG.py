# Written by Thomas Noel
# for CS 539 Convex Optimization
#
# SVRG Algorithm is described in "Accelerating Stochastic
# Gradient Descent using Predictive Variance Reduction" by
# Johnson and Zhang

import mnist_reader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def svrg(update_freq: int, learning_rate: float = 1e-3) -> None:
    pass

def main():
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')    
    print(X_train.shape)

if __name__ == '__main__':
    main()
