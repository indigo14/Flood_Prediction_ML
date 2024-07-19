from enum import Enum, auto
from typing import Protocol, List
import numpy as np

class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()  # Uncommented for inclusion

class ExperimentTracker(Protocol):
    def set_stage(self, stage: Stage) -> None:
        """
        Sets the current stage of the experiment.

        Parameters:
        stage (Stage): The stage to set (TRAIN, TEST, VAL).
        """
        ...

    def add_batch_metric(self, name: str, value: float, step: int) -> None:
        """
        Logs a batch-level metric.

        Parameters:
        name (str): The name of the metric.
        value (float): The value of the metric.
        step (int): The step (batch number).
        """
        ...

    def add_epoch_metric(self, name: str, value: float, step: int) -> None:
        """
        Logs an epoch-level metric.

        Parameters:
        name (str): The name of the metric.
        value (float): The value of the metric.
        step (int): The step (epoch number).
        """
        ...

    # def add_epoch_confusion_matrix(
    #     self, y_true: List[np.array], y_pred: List[np.array], step: int
    # ) -> None:
    #     """
    #     Logs a confusion matrix at epoch-level.

    #     Parameters:
    #     y_true (List[np.array]): True labels.
    #     y_pred (List[np.array]): Predicted labels.
    #     step (int): The step (epoch number).
    #     """
        ...

# Example concrete implementation
class SimpleExperimentTracker:
    def __init__(self):
        self.current_stage = None

    def set_stage(self, stage: Stage) -> None:
        self.current_stage = stage
        print(f"Stage set to: {self.current_stage}")

    def add_batch_metric(self, name: str, value: float, step: int) -> None:
        print(f"Batch metric - {name}: {value} at step {step}")

    def add_epoch_metric(self, name: str, value: float, step: int) -> None:
        print(f"Epoch metric - {name}: {value} at step {step}")

    # def add_epoch_confusion_matrix(
    #     self, y_true: List[np.array], y_pred: List[np.array], step: int
    # ) -> None:
    #     # Simple example, you might want to calculate and print the actual confusion matrix
    #     print(f"Confusion matrix at step {step}")


