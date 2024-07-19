import torch
import torch.nn as nn

class FloodPredictionNet(nn.Module):
    """
    A neural network for predicting flood probability.

    Parameters:
    input_size (int): The number of input features.
    """
    def __init__(self, input_size: int):
        super(FloodPredictionNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)  # No activation function in the output layer for regression
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        return self.network(x)

# Example usage
# Assuming input_size is the number of features in your dataset
# model = FloodPredictionNet(input_size=<number_of_features>)
