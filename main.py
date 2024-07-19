import pathlib
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import torch
# from src.dataset import create_dataloader
# from src.models import LinearNet
# from src.runner import Runner, run_epoch
# from src.tensorboard import TensorboardExperiment
from src.dataset import create_dataloader
from src.models import FloodPredictionNet
from src.runner import Runner, run_epoch
from src.tensorboard import TensorboardExperiment

# Hyperparameters
EPOCH_COUNT = 10#changed to save time
LR = 5e-5
BATCH_SIZE = 128

# Log configuration
LOG_PATH = "./logs"

# Data configuration
DATA_DIR = "./data/processed"
TEST_DATA = pathlib.Path(f"{DATA_DIR}/preprocessed_X_test.csv")
TEST_LABELS = pathlib.Path(f"{DATA_DIR}/preprocessed_y_test.csv")
TRAIN_DATA = pathlib.Path(f"{DATA_DIR}/preprocessed_X_train.csv")
TRAIN_LABELS = pathlib.Path(f"{DATA_DIR}/preprocessed_y_train.csv")

# Define the input size (number of features in your dataset)
INPUT_SIZE = 20  # Replace with the actual number of features in your dataset

def main():
    """
    Main function to run the training and evaluation of the model.
    """

    # Initialize the model and optimizer
    model = FloodPredictionNet(input_size=INPUT_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create the data loaders
    test_loader = create_dataloader(BATCH_SIZE, TEST_DATA, TEST_LABELS)
    train_loader = create_dataloader(BATCH_SIZE, TRAIN_DATA, TRAIN_LABELS, shuffle=True)

    # Create the runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    # Setup the experiment tracker
    tracker = TensorboardExperiment(log_path=LOG_PATH)

    # Run the epochs
    for epoch_id in range(EPOCH_COUNT):
        run_epoch(test_runner, train_runner, tracker, epoch_id)

        # Compute Average Epoch Metrics
        summary = ", ".join(
            [
                f"[Epoch: {epoch_id + 1}/{EPOCH_COUNT}]",
                f"Test MSE: {test_runner.avg_mse: 0.4f}",
                f"Train MSE: {train_runner.avg_mse: 0.4f}",
                f"Test MAE: {test_runner.avg_mae: 0.4f}",
                f"Train MAE: {train_runner.avg_mae: 0.4f}",
                f"Test R2: {test_runner.avg_r2: 0.4f}",
                f"Train R2: {train_runner.avg_r2: 0.4f}",
            ]
        )
        print("\n" + summary + "\n")

        # Reset the runners
        train_runner.reset()
        test_runner.reset()

        # Flush the tracker after every epoch for live updates
        tracker.flush()


if __name__ == "__main__":
    main()
