import numpy as np
from sklearn.metrics import balanced_accuracy_score


def log_likelihood(X, y, beta):
    """Log likelihood function."""
    return -np.sum(X @ beta * y - np.log(1 + np.exp(X @ beta)))


def calc_pi(X, beta):
    """Calculation of odds."""
    exp = np.exp(X @ beta)
    return exp / (1 + exp)


def train_eval_scikit_model(X_train, y_train, X_test, y_test, scikit_model):
    model = scikit_model
    model.fit(X_train, y_train.T[0])
    lr_preds = model.predict(X_test)
    lr_acc = balanced_accuracy_score(y_test, np.expand_dims(lr_preds, 1))
    return lr_acc
