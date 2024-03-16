from tqdm import tqdm
import numpy as np


def sgd(X, Y, step_size, max_it=10000, batch_size=1, print_every=50):
    """
    Stochastic Gradient Descent with logistic loss for binary classification [0,1].
    Print the loss values at interval of your choice.

    Arguments:
    X : Data matrix
    Y : Labels
    W : Weights, previously initialized outside the function
    step_size : Float value of step size to take.
    max_it : Maximum number of iterations, after which to stop
    batch_size : Size of the batch to use for each iteration
    print_every : if positive prints loss after every `record_every` iteration
                  (1=record all losses), otherwise record nothing

    Returns:
    history : A list containing the loss value at each iteration
    best_w : The best weights corresponding to the best loss value
    """
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
        J = X.T*(y_pred-Y)
        J = np.mean(J, axis=1)
        return J

    history = []  # to keep track of loss values
    w_init = np.ones(X.shape[1])
    best_w = w_init.copy()
    n_samples = X.shape[0]

    for i in tqdm(range(max_it), desc="Stochastic Gradient Descent"):
        # create a random sample
        indices = np.random.choice(n_samples, batch_size, replace=False)
        X_sample = X[indices]
        Y_sample = Y[indices]

        # compute y_hat (preds) and then loss (L)
        # sigmoid, probability of class 1
        preds = 1 / (1 + np.exp(-np.dot(best_w, X_sample.T)))
        L = logistic_loss(preds, Y_sample)
        history.append(L)
        if print_every > 0 and i % print_every == 0:
            print("Loss", L)

        # compute loss gradient (J) and update weights
        J = dlogistic(preds, X_sample, Y_sample, W=best_w)
        best_w = best_w - step_size*J

    return history, best_w
