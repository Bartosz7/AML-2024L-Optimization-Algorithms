import numpy as np
from tqdm import tqdm
from utils.train_helpers import (
    log_likelihood,
    calc_pi
)

def make_batches(X, y, batch_size):
    """Function creates batches for gradient descent algorithm."""
    perm = np.random.permutation(len(y))
    X_perm = X[perm, :]
    y_perm = y[perm]
    return np.array_split(X_perm, int(X_perm.shape[0]/batch_size)), np.array_split(y_perm, int(len(y_perm)/batch_size))

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
    cost = np.mean(-y*np.log(y_hat+eps)-(1-y)*np.log(1-y_hat+eps))
    return cost

def dlogistic(preds, X, Y, W=[]):
    """
    Gradient/derivative of the logistic loss.

    Arguments:
    preds : Predicted values
    X : Input data matrix
    Y : True target values
    W : The weights, optional argument, may/may not be needed depending on the loss function
    """
    y_pred = sigmoid(np.dot(W, X.T))
    J = X.T @ (np.expand_dims(y_pred, 1)-Y)

    J = np.mean(J, axis=1)
    return J


def GD(
        X, 
        y, 
        learning_rate, 
        n_epoch=1, 
        batch_size=1, 
        print_every=50, 
        print_likeli=True, 
        use_adam=True,
        beta1 = 0.9,
        beta2 = 0.999
    ):
    """
    Stochastic Gradient Descent with logistic loss for binary classification [0,1].
    Print the loss values at interval of your choice.

    Arguments:
    X : Data matrix
    Y : Labels
    W : Weights, previously initialized outside the function
    learning_rate : Float value of step size to take.
    n_epoch : Maximum number of epochs, after which to stop
    batch_size : Size of the batch to use for each iteration
    print_every : if positive prints loss after every `record_every` iteration
                  (1=record all losses), otherwise record nothing

    Returns:
    history : A list containing the loss value at each iteration
    best_w : The best weights corresponding to the best loss value
    """

    history = []  # to keep track of loss values
#     w_init = np.ones(X.shape[1])
    w_init = (np.linalg.inv(X.T @ X) @ X.T @ y).T[0]
    best_w = w_init.copy()
    n_samples = X.shape[0]
    log_like = log_likelihood(X, y, np.expand_dims(best_w, 1))
    history.append(log_like)
    
    #ADAM initialization
    M = np.zeros(len(w_init))
    R = np.zeros(len(w_init))
    t = 0
    eps = 1e-8
    
#     if print_likeli:
#         print("Minus log likelihood", log_like)
    for i in tqdm(range(n_epoch), "Epochs"):
        batches = make_batches(X, y, batch_size)
        for j in range(len(batches[0])):
            t += 1
            X_sample = batches[0][j]
            Y_sample = batches[1][j]

            # compute y_hat (preds) and then loss (L)
            # sigmoid, probability of class 1
            preds = 1 / (1 + np.exp(-np.dot(best_w, X_sample.T)))

            # compute loss gradient (J) and update weights
            J = dlogistic(preds, X_sample, Y_sample, W=best_w)
            if use_adam:
                M = beta1 * M + (1 - beta1) * J
                R = beta2 * R + (1 - beta2) * (J**2)

                M_bias = M / (1 - beta1**t)
                R_bias = R / (1 - beta2**t)

                best_w = best_w - learning_rate * (M_bias / np.sqrt(R_bias + eps))
            else:
                best_w = best_w - learning_rate*J
        log_like = log_likelihood(X, y, np.expand_dims(best_w, 1))
        history.append(log_like)
#         if print_likeli:
#             print("Minus log likelihood", log_like)
    return history, best_w