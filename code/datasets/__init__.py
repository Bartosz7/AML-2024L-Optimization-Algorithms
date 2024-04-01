"""
This module contains the dataset classes, their
preprocessing functions, the function to load the datasets.
"""

from datasets.datasets import (
    Dataset,
    Booking,
    Churn,
    Diabetes,
    Employee,
    Challenger,
    Jungle,
    Ionosphere,
    Water,
    Seeds,
    Sonar,
)
from datasets.preprocess_helpers import split_with_preprocess


_INSTALLED_DATASETS = {
    "booking": Booking,
    "churn": Churn,
    "diabetes": Diabetes,
    "employee": Employee,
    "challenger": Challenger,
    "jungle": Jungle,
    "ionosphere": Ionosphere,
    "water": Water,
    "seeds": Seeds,
    "sonar": Sonar,
}


def load_dataset(name: str) -> Dataset:
    """Loads the dataset(s) and preprocesses it."""
    return _INSTALLED_DATASETS[name]()
