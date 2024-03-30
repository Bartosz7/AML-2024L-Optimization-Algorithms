"""
util.py

Utility functions for optimization algorithms.
"""
import numpy as np


def calc_pi(X, beta):
    """Calculation of odds."""
    exp = np.exp(X @ beta)
    return exp / (1 + exp)


# pylint: disable=invalid-name
def log_likelihood(X, y, beta):
    """Log likelihood function."""
    return -np.sum(X @ beta * y - np.log(1 + np.exp(X @ beta)))


# pylint: disable=invalid-name
def make_batches(X, y, batch_size):
    """Function creates batches for gradient descent algorithm."""
    perm = np.random.permutation(len(y))
    X_perm = X[perm, :]  # pylint: disable=invalid-name
    y_perm = y[perm]
    return (np.array_split(X_perm, int(X_perm.shape[0] / batch_size)),
            np.array_split(y_perm, int(len(y_perm) / batch_size)))


def sigmoid(z):
    """
    Sigmoid activation function.

    Arguments:
    z : Input scalar or batch of scalars

    Returns:
    activation : Sigmoid activation(s) on z
    """
    activation = 1 / (1 + np.exp(-z))
    return activation


def logistic_loss(preds, targets):
    """
    Logistic loss function for binary classification.

    Arguments:
    preds : Predicted values
    targets : Target values

    Returns :
    cost : The mean logistic loss value between preds and targets
    """
    # mean logistic loss
    eps = 1e-14
    y = targets
    y_hat = preds
    cost = np.mean(-y * np.log(y_hat + eps) - (1 - y) * np.log(1 - y_hat + eps))
    return cost


# pylint: disable=invalid-name
def dlogistic(preds, X, Y, W=[]):  # TODO: dangerous default value?
    """
    Gradient/derivative of the logistic loss.

    Arguments:
    preds : Predicted values
    X : Input data matrix
    Y : True target values
    W : The weights, optional argument, may/may not be needed depending on the loss function
    """
    y_pred = sigmoid(np.dot(W, X.T))
    J = X.T @ (np.expand_dims(y_pred, 1) - Y)
    J = np.mean(J, axis=1)
    return J
