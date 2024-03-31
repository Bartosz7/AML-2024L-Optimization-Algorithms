import time
import warnings
from typing import Callable

import numpy as np
from optim import ADAM, GD, IWLS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from utils.train_helpers import calc_pi, train_eval_scikit_model

warnings.filterwarnings("ignore")


def train_and_eval(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[list[float], list[float], list[float], float, float, float]:
    """Train models for a given train and test sets"""
    # IWLS
    iwls = IWLS(n_iter=500)
    l_iwls_vals, best_beta_iwls = iwls.optimize(X_train, y_train)
    iwls_test_preds = 1 * (calc_pi(X_test, best_beta_iwls) > 0.5)
    iwls_acc = balanced_accuracy_score(y_test, iwls_test_preds)

    # SGD
    gd = GD(learning_rate=0.0002, n_epoch=500)
    l_sgd_vals, best_beta_sgd = gd.optimize(X_train, y_train)
    sgd_test_preds = 1 * (calc_pi(X_test, best_beta_sgd) > 0.5)
    sgd_acc = balanced_accuracy_score(y_test, sgd_test_preds)

    # ADAM
    adam = ADAM(learning_rate=0.0002, n_epoch=500)
    l_adam_vals, best_beta_adam = adam.optimize(X_train, y_train)
    adam_test_preds = 1 * (calc_pi(X_test, best_beta_adam) > 0.5)
    adam_acc = balanced_accuracy_score(y_test, adam_test_preds)

    # LR from scikit
    lr_acc = train_eval_scikit_model(
        X_train, y_train, X_test, y_test, scikit_model=LogisticRegression()
    )
    # QDA from scikit
    qda_acc = train_eval_scikit_model(
        X_train, y_train, X_test, y_test, scikit_model=QDA()
    )
    # LDA from scikit
    lda_acc = train_eval_scikit_model(
        X_train, y_train, X_test, y_test, scikit_model=LDA()
    )
    # Decision tree from scikit
    dt_acc = train_eval_scikit_model(
        X_train,
        y_train,
        X_test,
        y_test,
        scikit_model=DecisionTreeClassifier(max_depth=5),
    )
    # Random forest from scikit
    rf_acc = train_eval_scikit_model(
        X_train,
        y_train,
        X_test,
        y_test,
        scikit_model=RandomForestClassifier(max_depth=5),
    )

    print(f"Balanced accuracy of SGD without optimizer is: {sgd_acc}")
    print(f"Balanced accuracy of GD with ADAM is: {adam_acc}")
    print(f"Balanced accuracy of IWLS is: {iwls_acc}")
    print(f"Balanced accuracy of LR from Scikit is {lr_acc}")
    print(f"Balanced accuracy of QDA from Scikit is {qda_acc}")
    print(f"Balanced accuracy of LDA with ADAM is: {lda_acc}")
    print(f"Balanced accuracy of Decision Tree is: {dt_acc}")
    print(f"Balanced accuracy of Random Forest from Scikit is {rf_acc}")

    l_vals_dict = {"iwls": l_iwls_vals, "sgd": l_sgd_vals, "adam": l_adam_vals}

    acc_vals_dict = {
        "iwls": iwls_acc,
        "sgd": sgd_acc,
        "adam": adam_acc,
        "lr": lr_acc,
        "qda": qda_acc,
        "lda": lda_acc,
        "dt": dt_acc,
        "rf": rf_acc,
    }

    return l_vals_dict, acc_vals_dict


def cv(preprocess_fun: Callable, n_splits: int = 5, **kwargs) -> tuple[
    list[float],
    list[float],
    list[float],
    list[list[float]],
    list[list[float]],
    list[list[float]],
]:
    """Cross-validation for every model used to evaluate balanced accuracy.

    Arguments:
    preprocess_fun : function to perform preprocessing
    n_splits : number of different splits of data

    Keyword Arguments:
    filename : path to file with data
    interactions : if True the interactions between variables are added during preprocessing

    Returns:
        TODO
    """
    acc_vals_splits_dict = {
        "iwls": [None for i in range(n_splits)],
        "sgd": [None for i in range(n_splits)],
        "adam": [None for i in range(n_splits)],
        "lr": [None for i in range(n_splits)],
        "qda": [None for i in range(n_splits)],
        "lda": [None for i in range(n_splits)],
        "dt": [None for i in range(n_splits)],
        "rf": [None for i in range(n_splits)],
    }

    l_vals_splits_dict = {
        "iwls": [None for i in range(n_splits)],
        "sgd": [None for i in range(n_splits)],
        "adam": [None for i in range(n_splits)],
    }

    for i in range(n_splits):
        print(f"CV split {i+1}")

        X_train, y_train, X_test, y_test = preprocess_fun(**kwargs)
        time.sleep(1)  # to remove visual bug with tqdm
        l_vals_dict, acc_vals_dict = train_and_eval(X_train, y_train, X_test, y_test)

        for key in l_vals_splits_dict:
            l_vals_splits_dict[key][i] = l_vals_dict[key]

        for key in acc_vals_splits_dict:
            acc_vals_splits_dict[key][i] = acc_vals_dict[key]

    return l_vals_splits_dict, acc_vals_splits_dict
