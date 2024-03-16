import numpy as np
from tqdm import tqdm

from utils.train_helpers import calc_pi, log_likelihood


def IWLS(X, y, n_iter=10, print_likeli=True):
    """Iterative Weighted Least Squares method."""
    #     X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    # beta = np.zeros((X.shape[1], 1))
    pi = calc_pi(X, beta)
    l_prev = log_likelihood(X, y, beta)
    #     if print_likeli:
    #         print(f"Starting Log likelihood is: {l_prev}")
    l_vals = [l_prev]
    for i in tqdm(range(n_iter), "Iterations"):
        W = np.diag((pi * (1 - pi)).T[0])
        beta = beta + np.linalg.inv(X.T @ W @ X) @ X.T @ (y - pi)
        pi = calc_pi(X, beta)
        l = log_likelihood(X, y, beta)
        l_diff = abs(l - l_prev)
        l_prev = l
        #         if print_likeli:
        #             print(f"Log likelihood is: {l}")
        l_vals.append(l)
    #         if l_diff < 1:
    #             break

    return l_vals, beta
