from dataclasses import dataclass, field
from typing import List

@dataclass
class Metric:
    values: List[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: int = 0
    average: float = 0.0

    def update(self, value: float, batch_size: int) -> None:
        """
        Update the metric with a new value and batch size.

        Parameters:
        value (float): The new value to update the metric with.
        batch_size (int): The size of the batch associated with the new value.
        """
        if batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")

        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates
