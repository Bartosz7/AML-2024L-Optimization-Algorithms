from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from utils.preprocess_helpers import impute_water, one_hot_encode, split_with_preprocess

RANDOM_STATE = 123  # TODO: where is it used?
DATA_DIR = Path("../data")


def preprocess_booking(
    filename: str = DATA_DIR / "booking.csv", interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for booking.csv dataset."""
    booking = pd.read_csv(filename).drop(["Booking_ID", "date of reservation"], axis=1)
    booking["market segment type"] = 1 * (booking["market segment type"] == "Online")
    booking["booking status"] = 1 * (booking["booking status"] == "Canceled")
    label_encoder = preprocessing.LabelEncoder()
    booking["room type"] = label_encoder.fit_transform(booking["room type"])
    booking = one_hot_encode(booking)
    return split_with_preprocess(
        df=booking,
        target_col_name="booking status",
        dataset_name="booking",
        interactions=interactions,
    )


def preprocess_churn(
    filename: str = DATA_DIR / "churn.csv", interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for churn.csv dataset."""
    churn = pd.read_csv(filename)
    churn["FrequentFlyer"] = 1 * (churn["FrequentFlyer"] == "Yes")
    churn["BookedHotelOrNot"] = 1 * (churn["BookedHotelOrNot"] == "Yes")
    churn["AccountSyncedToSocialMedia"] = 1 * (
        churn["AccountSyncedToSocialMedia"] == "Yes"
    )
    churn.loc[churn["AnnualIncomeClass"] == "Low Income", "AnnualIncomeClass"] = 0
    churn.loc[churn["AnnualIncomeClass"] == "Middle Income", "AnnualIncomeClass"] = 1
    churn.loc[churn["AnnualIncomeClass"] == "High Income", "AnnualIncomeClass"] = 2
    churn.AnnualIncomeClass = churn.AnnualIncomeClass.astype(int)

    return split_with_preprocess(
        df=churn,
        target_col_name="Target",
        dataset_name="churn",
        interactions=interactions,
    )


def preprocess_diabetes(
    filename: str = DATA_DIR / "diabetes.arff", interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for diabetes.arff dataset."""
    df = pd.DataFrame(arff.loadarff(filename)[0])
    str_df = df.select_dtypes([object]).astype(str)
    df[str_df.columns] = str_df
    df["class"] = df["class"].apply(lambda x: 1 if x == "tested_positive" else 0)

    return split_with_preprocess(
        df=df,
        target_col_name="class",
        dataset_name="diabetes",
        interactions=interactions,
    )


def preprocess_employee(
    filename: str = DATA_DIR / "employee.csv", interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for employee.csv dataset."""
    df = pd.read_csv(filename)
    df["EducationBachelors"] = 1 * (df["Education"] == "Bachelors")
    df["EducationMasters"] = 1 * (df["Education"] == "Masters")
    df["Gender"] = df["Gender"].map({"Female": 1, "Male": 0})
    df["EverBenched"] = df["EverBenched"].map({"No": 0, "Yes": 1})
    df.drop(["Education", "City"], axis=1, inplace=True)

    return split_with_preprocess(
        df=df,
        target_col_name="LeaveOrNot",
        dataset_name="employee",
        interactions=interactions,
    )


def preprocess_challenger(
    filename: str = DATA_DIR / "challenger_lol.csv", interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for challenger_lol.csv dataset."""
    df = pd.read_csv(filename)
    df.drop(["blueFirstBlood", "redFirstBlood", "gameId"], axis=1, inplace=True)
    for col in ["blue", "red"]:
        for lane in ["BOT_LANE", "MID_LANE", "TOP_LANE"]:
            df[f"{col}FirstTowerLane_{lane}"] = df[f"{col}FirstTowerLane"].apply(
                lambda x: int(lane in x)
            )
        for dragon in ["AIR_DRAGON", "WATER_DRAGON", "FIRE_DRAGON", "EARTH_DRAGON"]:
            df[f"{col}DragnoType_{dragon}"] = df[f"{col}DragnoType"].apply(
                lambda x: int(dragon in x)
            )
        df.drop(f"{col}FirstTowerLane", axis=1, inplace=True)
        df.drop(f"{col}DragnoType", axis=1, inplace=True)

    return split_with_preprocess(
        df=df,
        target_col_name="blueWins",
        dataset_name="challenger lol",
        interactions=interactions,
    )


def preprocess_jungle(
    filename: str = DATA_DIR / "jungle_chess.arff", interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for jungle_chess.arff dataset."""
    df = arff.loadarff(filename)
    df = pd.DataFrame(df[0])
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode("utf-8").unstack()
    for col in str_df:
        df[col] = str_df[col]
    df = df[df["class"] != "d"]
    df[["highest_strength", "closest_to_den", "fastest_to_den", "class"]] = df.copy()[
        ["highest_strength", "closest_to_den", "fastest_to_den", "class"]
    ].applymap(lambda x: int(x == "w"))
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[["white_piece0_advanced", "black_piece0_advanced"]], drop_first=True
            )
            * 1,
        ],
        axis=1,
    )
    df.drop(
        [
            "white_piece0_advanced",
            "black_piece0_advanced",
            "white_piece0_in_water",
            "black_piece0_in_water",
        ],
        axis=1,
        inplace=True,
    )
    df = df.apply(pd.to_numeric)

    return split_with_preprocess(
        df=df, target_col_name="class", dataset_name="jungle", interactions=interactions
    )


def preprocess_ionosphere(
    filename: str = DATA_DIR / "ionosphere.data", interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for ionosphere.data dataset."""
    df = pd.read_csv(filename, header=None)
    df = df.rename(columns={34: "class"})
    df["class"] = df["class"].map({"g": 0, "b": 1})
    return split_with_preprocess(
        df=df,
        target_col_name="class",
        dataset_name="ionosphere",
        interactions=interactions,
    )


def preprocess_water(
    filename: str = DATA_DIR / "water_quality.csv", interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for water_quality.csv dataset."""
    water = pd.read_csv(filename)

    return split_with_preprocess(
        df=water,
        target_col_name="is_safe",
        dataset_name="water",
        additional_preprocess=impute_water,
        interactions=interactions,
    )


def preprocess_seeds(
    filename: str = DATA_DIR / "seeds.txt", interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for seeds.txt dataset."""
    cols = [
        "A",
        "P",
        "C",
        "kernel_length",
        "kernel_width",
        "asymmetry_coef",
        "kernel_groove_length",
        "class",
    ]
    df = pd.read_csv(filename, sep=r"\s+", header=None, names=cols)
    # combine classes 1, 3 (similar) and 2 (different) based on pairplot
    df["class"] = df["class"].map({1: 0, 2: 1, 3: 2})
    return split_with_preprocess(
        df=df,
        target_col_name="class",
        dataset_name="seeds",
        interactions=interactions,
    )


def preprocess_sonar(
    filename: str = DATA_DIR / "sonar.data", interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for sonar.data dataset."""
    df = pd.read_csv(filename, header=None)
    df = df.rename(columns={60: "class"})
    df["class"] = df["class"].map({"R": 0, "M": 1})
    return split_with_preprocess(
        df=df,
        target_col_name="class",
        dataset_name="sonar",
        interactions=interactions,
    )
