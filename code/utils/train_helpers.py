import numpy as np


def log_likelihood(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Log likelihood function."""
    return -np.sum(X @ beta * y - np.log(1 + np.exp(X @ beta)))


def calc_pi(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Calculation of odds."""
    exp = np.exp(X @ beta)
    return exp / (1 + exp)
