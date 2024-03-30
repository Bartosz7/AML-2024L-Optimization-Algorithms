"""
optim.py

Provides the abstract class for the optimizer interface.
"""
from abc import ABC, abstractmethod
import copy
import numpy as np


# pylint: disable=invalid-name
class Optimizer(ABC):
    """Defines the general optimizer interface 
    for the logistic regression problem."""

    def __init__(self) -> None:
        self._loss_history: list[float] = []
        self._global_best_weights: np.ndarray | None = None

    @abstractmethod
    def optimize(self,
                 X: np.ndarray,
                 y: np.ndarray) -> tuple[list[float], np.ndarray]:
        """Optimize the given problem.
        To be overridden by the derived classes."""

    def reset(self) -> None:
        """Reset the optimizer's state."""
        self._loss_history = []
        self._global_best_weights = None

    @property
    def loss_history(self) -> list[float]:
        """Returns the loss history of the optimizer

        Returns:
            list[float]: The loss history of the optimizer
        """
        return copy.deepcopy(self._loss_history)

    @property
    def global_best_weights(self) -> np.ndarray | None:
        """Returns the globally best weights of the optimizer

        Returns:
            np.ndarray: The globally best weights of the optimizer
        """
        return copy.deepcopy(self._global_best_weights)
