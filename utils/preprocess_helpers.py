import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


def one_hot_encode(df):
    """One hot encoding for object type columns in given data frame."""
    for column in df:
        if is_object_dtype(df[column]):
            dummies = pd.get_dummies(df[column], prefix=column)
            if np.sum(df[column].isna()) == 0:
                dummies = dummies.iloc[:, :-1]
            df = df.drop(column, axis=1)
            df = df.join(dummies)
    return df


def calculate_vif(X, thresh=5.0):
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


def replace_water_nans(df):
    """Imputation of #NUM! values in water_quality data frame."""
    df["ammonia"] = df["ammonia"].replace("#NUM!", -100)
    df["ammonia"] = df["ammonia"].astype(float)
    df["ammonia"] = df["ammonia"].replace(
        -100, df.loc[df["ammonia"] != -100, "ammonia"].mean()
    )

    df["is_safe"] = df["is_safe"].replace("#NUM!", -100)
    df["is_safe"] = df["is_safe"].astype(int)
    return df


def impute_water(water_train, water_test):
    """Data imputation in water_quality data frame using column dominant from training set."""
    water_train = replace_water_nans(water_train)
    water_test = replace_water_nans(water_test)

    if np.mean(water_train.loc[water_train["is_safe"] != -100, "is_safe"]) > 0.5:
        dominant = 1
    else:
        dominant = 0

    water_train["is_safe"] = water_train["is_safe"].replace(-100, dominant)
    water_test["is_safe"] = water_test["is_safe"].replace(-100, dominant)

    return water_train, water_test


def split_with_preprocess(
    df, target_col_name, dataset_name, additional_preprocess=None, interactions=False
):
    """Train test split for given data frame including removal of multicolinear columns."""
    print(f"Removing multicolinear columns in {dataset_name} dataset...")
    train, test = train_test_split(df, test_size=0.2)  # random_state=RANDOM_STATE

    y_train = train[target_col_name].to_numpy()
    y_test = test[target_col_name].to_numpy()
    _X_train = train.drop(target_col_name, axis=1)
    _X_test = test.drop(target_col_name, axis=1)

    if additional_preprocess:
        _X_train, _X_test = additional_preprocess(_X_train, _X_test)

    indices_to_drop = calculate_vif(_X_train)
    X_train = _X_train.iloc[:, indices_to_drop]
    X_test = _X_test.iloc[:, indices_to_drop]

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    if interactions:
        X_train = make_interactions(X_train)
        X_test = make_interactions(X_test)

    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)

    return X_train, np.expand_dims(y_train, 1), X_test, np.expand_dims(y_test, 1)


def make_interactions(X):
    # TODO
    return X
