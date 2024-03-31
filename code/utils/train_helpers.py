import numpy as np
from sklearn.metrics import balanced_accuracy_score


def log_likelihood(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Log likelihood function."""
    return -np.sum(X @ beta * y - np.log(1 + np.exp(X @ beta)))


def calc_pi(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Calculation of odds."""
    exp = np.exp(X @ beta)
    return exp / (1 + exp)


def train_eval_scikit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scikit_model,  # TODO: type?,
):
    """TODO"""
    model = scikit_model
    model.fit(X_train, y_train.T[0])
    lr_preds = model.predict(X_test)
    lr_acc = balanced_accuracy_score(y_test, np.expand_dims(lr_preds, 1))
    return lr_acc
