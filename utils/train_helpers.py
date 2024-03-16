import numpy as np

def log_likelihood(X, y, beta):
    """Log likelihood function."""
    return -np.sum(X @ beta * y - np.log(1 + np.exp(X @ beta))) 

def calc_pi(X, beta):
    """Calculation of odds."""
    exp = np.exp(X @ beta)
    return exp / (1 + exp)
