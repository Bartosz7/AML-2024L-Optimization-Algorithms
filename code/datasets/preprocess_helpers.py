from typing import Callable, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datasets.dataset_model import Dataset
from sklearn.preprocessing import MinMaxScaler


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One hot encoding for object type columns in given data frame."""
    for column in df:
        if is_object_dtype(df[column]):
            dummies = pd.get_dummies(df[column], prefix=column) * 1
            if np.sum(df[column].isna()) == 0:
                dummies = dummies.iloc[:, :-1]
            df = df.drop(column, axis=1)
            df = df.join(dummies)
    return df


def _calculate_vif(X: pd.DataFrame, thresh: float = 5.0) -> list[int]:
    """
    Removal of multicolinear columns in given data frame using VIF. Based on:
    https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-pandas-columns-in-python
    """
    X = X.assign(const=1)  # faster than add_constant from statsmodels
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [
            variance_inflation_factor(X.iloc[:, variables].values, ix)
            for ix in range(X.iloc[:, variables].shape[1])
        ]
        vif = vif[:-1]  # don't let the constant be removed in the loop.
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            #             print('dropping \'' + X.iloc[:, variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    return variables[:-1]


def _make_interactions(X: np.ndarray) -> np.ndarray:
    """Generates interactions as the product of each 2 variables."""
    p = X.shape[1]
    for i in range(p - 1):
        for j in range(i + 1, p):
            X = np.append(X, np.expand_dims(X[:, i] * X[:, j], 1), axis=1)
    return X


def split_with_preprocess(
    dataset: Dataset,
    interactions: bool = False,
    test_size: int = 0.2,
    random_state: int = None,
    vif=True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train test split for given data frame including some additional preprocessing,
    removal of multicolinear columns using VIF, generating interactions, and adding column of ones.
    """
    train, test = train_test_split(
        dataset.df, test_size=test_size, random_state=random_state
    )

    if dataset.additional_preprocess:
        train, test = dataset.additional_preprocess(train, test)

    y_train = train[dataset.target_colname].to_numpy()
    y_test = test[dataset.target_colname].to_numpy()
    X_train = train.drop(dataset.target_colname, axis=1)
    X_test = test.drop(dataset.target_colname, axis=1)

    print(f"Removing multicolinear columns in {dataset.name} dataset...")
    if vif:
        indices_to_drop = _calculate_vif(X_train)
        X_train = X_train.iloc[:, indices_to_drop].to_numpy()
        X_test = X_test.iloc[:, indices_to_drop].to_numpy()

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if interactions:
        X_train = _make_interactions(X_train)
        X_test = _make_interactions(X_test)

    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)

    return X_train, np.expand_dims(y_train, 1), X_test, np.expand_dims(y_test, 1)
