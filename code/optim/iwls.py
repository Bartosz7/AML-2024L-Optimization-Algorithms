"""
iwls.py

Iterative Weighted Least Squares (IWLS) optimizer implementation.
"""
import numpy as np
from tqdm import tqdm
from .util import calc_pi, log_likelihood
from .optim import Optimizer


# pylint: disable=invalid-name
class IWLS(Optimizer):
    """Iterative Weighted Least Squares (IWLS) optimizer implementation
    with logistic loss for binary classification [0,1]."""

    def __init__(self, n_iter: int, tolerance: int = 5) -> None:
        """
        Initializes ADAM optimizer with the given parameters.

        Arguments:
        n_iter : Maximum number of iterations, after which to stop
        tolerance: Number of iterations to wait before stopping
                   if no improvement
        """
        super().__init__()
        self.n_iter = n_iter
        self.tolerance = tolerance

    def optimize(self, X, y, standardize=False):
        """
        Runs the IWLS optimizer on the given data.

        Returns:
        loss_history : A list containing the loss value at each iteration
        best_w : The best weights corresponding to the best loss value
        """
        self.reset()  # resets history and best weights
        if standardize:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        # beta = np.zeros((X.shape[1], 1))
        pi = calc_pi(X, beta)
        self._loss_history = [log_likelihood(X, y, beta)]

        best_log_like = float('inf')
        no_change_counter = 0

        for _ in tqdm(range(self.n_iter), "IWLS"):
            W = np.diag((pi * (1 - pi)).T[0])
            beta = beta + np.linalg.inv(X.T @ W @ X) @ X.T @ (y - pi)
            pi = calc_pi(X, beta)
            log_like = log_likelihood(X, y, beta)
            self._loss_history.append(log_like)

            if log_like < best_log_like:
                best_log_like = log_like
                self._global_best_weights = beta
                no_change_counter = 0
            else:
                no_change_counter += 1

            if no_change_counter > 5:
                break

        return self.loss_history, self.global_best_weights
