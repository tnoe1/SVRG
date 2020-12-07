# Written by Thomas Noel
# for CS 539 Convex Optimization
#
# SVRG Algorithm is described in "Accelerating Stochastic
# Gradient Descent using Predictive Variance Reduction" by
# Johnson and Zhang

import torch.optim.Optimizer


class SVRG(Optimizer):
    ''' Implements SVRG as described by Johnson and Zhang (2013).
    
    Parameters:
        params (list): An iterable of parameters to be optimized
        update_freq (int): The number of iterations between each average
                               gradient update.
        lr (float): learning rate. (default 1e-3)

    .. R. Johnson, T. Zhang. Accelerating Stochastic Gradient Descent using
           Predictive Variance Reduction. NIPS, 2013.
    '''
    def __init__(self, params, update_freq, lr=1e-3):
        if lr < 0.0:
            raise ValueError("Learning rate invalid; must be positive")
        if update_freq < 1:
            raise ValueError("Update frequency invalid; must be integer >= 1")
        defaults = dict(lr=lr, update_freq=update_freq)
        super(SVRG, self).__init__(params, defaults)


    def step(self, closure=None):
        pass
        

