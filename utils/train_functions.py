import time
import warnings
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from utils.GD import GD
from utils.IWLS import IWLS
from utils.train_helpers import calc_pi

warnings.filterwarnings("ignore")


def train_and_eval(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
):
    """Train models for a given train and test sets"""
    # IWLS
    l_iwls_vals, best_beta_iwls = IWLS(X_train, y_train, n_iter=100)
    iwls_test_preds = 1 * (calc_pi(X_test, best_beta_iwls) > 0.5)
    iwls_acc = balanced_accuracy_score(y_test, iwls_test_preds)
    # SGD
    l_sgd_vals, best_beta_sgd = GD(
        X_train, y_train, 0.0002, n_epoch=100, use_adam=False
    )
    sgd_test_preds = 1 * (calc_pi(X_test, best_beta_sgd) > 0.5)
    sgd_acc = balanced_accuracy_score(y_test, sgd_test_preds)
    # ADAM
    l_adam_vals, best_beta_adam = GD(X_train, y_train, 0.0002, n_epoch=100)
    adam_test_preds = 1 * (calc_pi(X_test, best_beta_adam) > 0.5)
    adam_acc = balanced_accuracy_score(y_test, adam_test_preds)

    # LR from scikit
    lr = LogisticRegression()
    lr.fit(X_train, y_train.T[0])
    lr_preds = lr.predict(X_test)
    lr_acc = balanced_accuracy_score(y_test, np.expand_dims(lr_preds, 1))

    print(f"Balanced accuracy of GD without optimizer is: {sgd_acc}")
    print(f"Balanced accuracy of SGD with ADAM is: {adam_acc}")
    print(f"Balanced accuracy of IWLS is: {iwls_acc}")
    print(f"Balanced accuracy of LR from Scikit is {lr_acc}")

    return l_iwls_vals, l_sgd_vals, l_adam_vals, iwls_acc, sgd_acc, adam_acc


def cv(preprocess_fun: Callable, n_splits: int = 5, **kwargs):
    """Cross-validation for every model used to evaluate balanced accuracy.

    Arguments:
    preprocess_fun : function to perform preprocessing
    n_splits : number of different splits of data

    Keyword Arguments:
    filename : path to file with data
    interactions : if True the interactions between variables are added during preprocessing

    Returns:

    """
    sgd_acc_list = []
    adam_acc_list = []
    iwls_acc_list = []
    l_iwls_vals_list = []
    l_sgd_vals_list = []
    l_adam_vals_list = []
    for i in range(n_splits):
        print(f"CV split {i+1}")

        X_train, y_train, X_test, y_test = preprocess_fun(**kwargs)
        time.sleep(1)  # to remove visual bug with tqdm
        l_iwls_vals, l_sgd_vals, l_adam_vals, iwls_acc, sgd_acc, adam_acc = (
            train_and_eval(X_train, y_train, X_test, y_test)
        )

        sgd_acc_list.append(sgd_acc)
        adam_acc_list.append(adam_acc)
        iwls_acc_list.append(iwls_acc)
        l_iwls_vals_list.append(l_iwls_vals)
        l_sgd_vals_list.append(l_sgd_vals)
        l_adam_vals_list.append(l_adam_vals)

    return (
        np.mean(sgd_acc_list),
        np.mean(adam_acc_list),
        np.mean(iwls_acc_list),
        l_iwls_vals_list,
        l_sgd_vals_list,
        l_adam_vals_list,
    )
