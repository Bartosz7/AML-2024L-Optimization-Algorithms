import numpy as np
from tqdm import tqdm

from utils.train_helpers import calc_pi, log_likelihood


def IWLS(X, y, n_iter=10, print_likeli=True):
    """Iterative Weighted Least Squares method."""
    #     X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    # beta = np.zeros((X.shape[1], 1))
    pi = calc_pi(X, beta)
    l_vals = [log_likelihood(X, y, beta)]

    best_l = 100000000
    no_change_counter = 0

    for i in tqdm(range(n_iter), "IWLS"):
        W = np.diag((pi * (1 - pi)).T[0])
        beta = beta + np.linalg.inv(X.T @ W @ X) @ X.T @ (y - pi)
        pi = calc_pi(X, beta)
        l = log_likelihood(X, y, beta)
        l_vals.append(l)

        if l < best_l:
            best_l = l
            no_change_counter = 0
        else:
            no_change_counter += 1

        if no_change_counter > 5:
            break

    return l_vals, beta
