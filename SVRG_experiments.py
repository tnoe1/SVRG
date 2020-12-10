# Thomas Noel
# CS 539
#
# Implements the SVRG algorithm for simple logistic regression

import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import mnist_reader
import random
import math

def sigmoid(x):
    '''
    NumPy friendly numerically stable sigmoid
    https://stackoverflow.com/a/62860170
    '''
    return np.piecewise(x, [x > 0], [
        lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))
    ])


def get_logistic_loss(X, y, w, reg):
    ''' Compute loss given the current weights

    Parameters:
        X (np.ndarray) : The training set
        y (np.ndarray) : The target values
        w (np.ndarray) : The model parameters
        reg (float) : The regularization parameter

    Returns:
        A float representing the loss.
    '''
    N = len(X)
    d = len(X[0])-1
    loss = 0
    for i in range(N):
        loss += (1/N)*(-y[i]*math.log(sigmoid(np.dot(w,X[i]))) 
                - (1-y[i])*math.log(1-sigmoid(np.dot(w,X[i]))))
    
    for j in range(d):
        loss += reg*w[j]**2
    
    return loss


def SGD_logistic(X, y, init_lr=0.001, a=1.1, eps=0.01):
    ''' Implements SGD using exponential decay learning rate scheduling.
    
    Parameters:
        X (np.ndarray) : The training set
        y (np.ndarray) : The target values
        init_lr (float) : The schedulers initial learning rate (Default: 1e-3)
        a (float) : The base of the learning rate scheduler; gets
                    exponentiated. (Default: 2.0)

    Returns:
        Model weights (np.ndarray)
    '''
    N = len(X)
    m = len(X[0])
    num_epochs = 30
    w = np.random.rand(m)
    y = y.reshape((y.shape[0],))
    grad_norm = eps + 1
    iter_num = 0
    # If the gradient l2 norm falls below our specified epsilon value,
    # then we have attained convergence 
    while grad_norm > eps:
        rand_idx = random.randint(0,N-1)
        grad = (y[rand_idx]-sigmoid(np.dot(w,X[rand_idx])))*X[rand_idx]
        grad_norm = LA.norm(grad)
        print(grad_norm)
        # Updating our learning rate from schedule
        alpha_j = init_lr*a**(iter_num//N)
        # Adding because objective function is concave
        w = w + alpha_j*grad
        iter_num += 1
    
    return w


def GD_logistic(X, y, reg, lr=1e-3, max_iter=10000, epsilon=0.1):
    '''Trains a logistic regression classifier using L2 regularization'''
    N = len(X)
    num_features = len(X[0])
    w = np.ones(num_features)
    grad_norm = epsilon + 1
    iter_num = 0 
    while grad_norm > epsilon and iter_num < max_iter:
        y = y.reshape((y.shape[0],))
        grad = (1/N)*X.T@(y-sigmoid(X@w))
        grad_norm = LA.norm(grad)
        w = w + lr*grad
        # excludes the dummy variable
        for i in range(num_features-1):
            w[i] -= lr*reg*w[i]
        iter_num += 1
    
    print('\n{} iterations until convergence'.format(iter_num))
    print('Grad. Norm at Convergence: {}'.format(grad_norm))
    print('Training Data Loss: {}'.format(get_logistic_loss(X, y, w, reg)))
    return w


def SVRG_mc_logistic(X, y, update_freq, lr=0.5, eps=0.01):
    ''' Implements SVRG for multiclass logistic regression
    
    Parameters:
        X (np.ndarray) : The training set
        y (np.ndarray) : The target values
        update_freq (int) : The number of iterations between every
                            average gradient update
        lr (float) : The learning rate (Default: 1e-3)

    Returns:
        Model weights (np.ndarray)
    
    .. R. Johnson, T. Zhang. Accelerating Stochastic Gradient Descent using
           Predictive Variance Reduction. NIPS, 2013.

    '''
    
    pass


def SVRG_logistic(X, y, update_freq, lr=0.5, eps=0.01):
    ''' Implements SVRG for logistic regression objective

    Parameters:
        X (np.ndarray) : The training set
        y (np.ndarray) : The target values
        update_freq (int) : The number of iterations between every
                            average gradient update
        lr (float) : The learning rate (Default: 1e-3)

    Returns:
        Model weights (np.ndarray)
    
    .. R. Johnson, T. Zhang. Accelerating Stochastic Gradient Descent using
           Predictive Variance Reduction. NIPS, 2013.
    '''
    grad_norm = eps + 1         
    N = len(X)
    m = len(X[0])
    w = np.ones(m)
    s_iter_count = 0
    tot_iter_count = 0
    loss = []
    loss_s = []

    # Initialize SVRG weights by performing a single SGD iteration on
    # a random training example.
    rand_ind = random.randint(0,N-1)
    w_tilde_prev = lr*(y[rand_ind]-sigmoid(np.dot(w,X[rand_ind])))*X[rand_ind]
    
    # Choosing to optimize until convergence, I maintain a count on the
    # outer iteration number, but semantically rely on convergence to
    # determine when it is appropriate to stop
    while grad_norm > eps:
        w_tilde = w_tilde_prev
        # Logging loss for this outer iteration
        #loss_s.append(get_logistic_loss(X, y, w_tilde, 0))
        mu_tilde = 0
        # Calculate mu_tilde
        for j in range(N):
            # adding (1/N) times the gradient associated with the ith example
            mu_tilde += (1/N)*(y[j]-sigmoid(np.dot(w,X[j])))*X[j]
        # Finding the Euclidian norm of our objective's gradient
        grad_norm = LA.norm(mu_tilde)
        print(grad_norm)
        # If the convergence is reached, then skip the next inner loop
        if grad_norm > eps:
            w = w_tilde
            #loss.append(get_logistic_loss(X, y, w, 0))
            for j in range(update_freq):
                rand_ind = random.randint(0,N-1)
                w_grad = (y[rand_ind]-sigmoid(np.dot(w,X[rand_ind])))*X[rand_ind]
                w_tilde_grad = (y[rand_ind]-sigmoid(np.dot(w_tilde,X[rand_ind])))*X[rand_ind]
                # SVRG Update step. Note that our objective function is concave, so we are
                # using gradient ascent
                w = w + lr*(w_grad - w_tilde_grad + mu_tilde)
                #loss.append(get_logistic_loss(X, y, w, 0))
                tot_iter_count += 1
        
            # Updating the weights using option I from the paper
            w_tilde_prev = w
            s_iter_count += 1

    return w_tilde, tot_iter_count, s_iter_count


def SVRG_testbed(X_train, y_train, X_test, y_test):
    # Iterate over frequencies (for each model)
    # Comparison with CVX perhaps?
    update_freqs = [2,5,10,15,20,25,30,35,40,45,50,75,100]
    SVRG_ws = []
    SVRG_tot_iters = []
    SVRG_s_iters = []
    SVRG_accuracies = []
    for freq in update_freqs:
        w, tot_iters, s_iters = SVRG_logistic(X_train.to_numpy(), y_train.to_numpy(), freq)
        SVRG_ws.append(w)
        SVRG_tot_iters.append(tot_iters)
        SVRG_s_iters.append(s_iters)
        SVRG_accuracies.append(accuracy(X_test.to_numpy(), y_test.to_numpy(), w))

    plt.figure(1)
    plt.plot(update_freqs, SVRG_s_iters)
    plt.xlabel('Update Frequency, m')
    plt.ylabel('\~\mu Calculations')
    plt.title('Health Insurance: Number of \~\mu Calculations Until Convergence')
    plt.savefig('SVRG_avg_grad_calcs_v_freq.png')
    #w = SGD_logistic(X_train.to_numpy(), y_train.to_numpy())
    #w = GD_logistic(X_train.to_numpy(), y_train.to_numpy(), 1e-4)
    #print('Accuracy: {}'.format(accuracy(X_test.to_numpy(), y_test.to_numpy(), w)))


def data_normalize(X_raw, exempt_labels=[]):
    ''' Normalizes the given data

    Parameters:
        X_raw (pd.DataFrame): The raw dataset
        exempt_labels (list): Labels to be left unnormalized (Default: [])
    
    Returns:
        The normalized data in a dataframe.
    '''
    features = X_raw.columns.tolist()
    X = X_raw.copy()
    for feature in X_raw.columns.tolist():
        if feature not in exempt_labels:
            stats = X_raw[feature].describe()
            l_min = stats['min']
            l_max = stats['max']
            # Normalize this column
            X[feature] = (X[feature].sub(l_min)).div(l_max - l_min)

    return X


def data_split(data, train_prop=0.7):
    ''' Binary classification data split into training and test sets '''
    N = len(data)
    data_pos = data[(data['target']==1)]
    data_neg = data[(data['target']==0)]
    npos = len(data_pos)
    nneg = len(data_neg)
    data_train = pd.DataFrame(columns=data.columns.tolist())
    data_test = pd.DataFrame(columns=data.columns.tolist())
    for i in range(npos):
        if i < train_prop*npos:
            data_train = data_train.append(data_pos.iloc[i])
        else:
            data_test = data_test.append(data_pos.iloc[i])

    for i in range(nneg):
        if i < train_prop*nneg:
            data_train = data_train.append(data_neg.iloc[i])
        else:
            data_test = data_test.append(data_neg.iloc[i])

    # Randomly shuffle rows of both dataframes
    data_train.sample(frac=1)
    data_test.sample(frac=1)
    
    X_train = data_train.drop(columns=['target'])
    y_train = data_train['target']
    X_test = data_test.drop(columns=['target'])
    y_test = data_test['target']

    return X_train, y_train, X_test, y_test 



def accuracy(X, y, w):
    '''Computes prediction accuracy of given set (for binary logistic regression)'''
    N = len(X)
    y = y.reshape((y.shape[0],))
    # 0 difference indicates correct prediction
    correct = np.equal(np.round(sigmoid(X@w)),y)
    num_correct = np.count_nonzero(correct == True)
    return num_correct / N


def normalize_verif(X, num_stats=None):
    '''Normalizes numerical features in given dataset'''
    NUMERICAL = ['Age', 'Annual_Premium', 'Vintage']
    if num_stats is None:
        num_stats = X[NUMERICAL].describe()
 
    for label in NUMERICAL:
        l_min = num_stats[label]['min']
        l_max = num_stats[label]['max']
        X[label] = (X[label].subtract(l_min)).div(l_max - l_min)
        
    return X, num_stats


def load_clean_verif_data():
    '''Loads and normalizes dataset'''
    X_train = pd.read_csv('data/health_insurance/insurance_train_X.csv')
    X_dev   = pd.read_csv('data/health_insurance/insurance_test_X.csv')
    y_train = pd.read_csv('data/health_insurance/insurance_train_y.csv')
    y_dev   = pd.read_csv('data/health_insurance/insurance_test_y.csv')

    X_train, num_stats = normalize_verif(X_train)
    X_dev, _ = normalize_verif(X_dev, num_stats=num_stats)

    return X_train, y_train, X_dev, y_dev
    

def main():
    HEALTH_INSURANCE = 0
    HEART_ATTACK = 1
    FASHION_MNIST = 2

    dataset = HEALTH_INSURANCE #[0, 1, 2]

    if dataset == HEALTH_INSURANCE:
        X_train, y_train, X_test, y_test = load_clean_verif_data()       
 
    elif dataset == HEART_ATTACK:
        data = pd.read_csv('data/heart/heart.csv')
        n_data = data_normalize(data, exempt_labels=['target'])
        X_train, y_train, X_test, y_test = data_split(n_data)

    elif dataset == FASHION_MNIST:
        X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
        X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')    
    

    # Pass along dataframes to experiment runner
    SVRG_testbed(X_train, y_train, X_test, y_test) 

    # If another model is built/used, can call SVRG_testbed on that
    # data again here

if __name__ == '__main__':
    main()
