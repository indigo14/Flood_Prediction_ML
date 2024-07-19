from typing import Any, Optional, List
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Added r2_score import
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from metrics import Metric
from tracking import ExperimentTracker, Stage

class Runner:
    def __init__( 
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """
        Initializes the Runner.

        Parameters:
        loader (DataLoader): The data loader for fetching batches of data.
        model (torch.nn.Module): The neural network model to train or evaluate.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer for training the model. If None, the runner is in evaluation mode.
        """
        self.run_count = 0
        self.loader = loader
        self.mse_metric = Metric()
        self.mae_metric = Metric()
        self.r2_metric = Metric()  # Add R-squared metric
        self.model = model
        self.optimizer = optimizer
        # Objective (loss) function
        self.compute_loss = torch.nn.MSELoss(reduction="mean")
        self.y_true_batches: List[List[Any]] = []
        self.y_pred_batches: List[List[Any]] = []
        # Assume Stage based on presence of optimizer
        self.stage = Stage.VAL if optimizer is None else Stage.TRAIN

    @property
    def avg_mse(self) -> float:
        """Returns the average Mean Squared Error."""
        return self.mse_metric.average

    @property
    def avg_mae(self) -> float:
        """Returns the average Mean Absolute Error."""
        return self.mae_metric.average
    
    @property
    def avg_r2(self) -> float:
        """Returns the average R-squared value."""
        return self.r2_metric.average #Added r2_metric property

    def run(self, desc: str, experiment: ExperimentTracker) -> None:
        """
        Runs the training or evaluation loop over the data loader.

        Parameters:
        desc (str): Description for the progress bar.
        experiment (ExperimentTracker): The experiment tracker for logging metrics.
        """
        self.model.train(self.stage is Stage.TRAIN)

        for x, y in tqdm(self.loader, desc=desc, ncols=80):
            if torch.isnan(x).any() or torch.isnan(y).any():
                print("NaN values found in input data")
                continue  # Skip this batch if it contains NaN values

            loss, batch_mse, batch_mae, batch_r2 = self._run_single(x, y)# Add batch_r2

            if experiment is not None:
                experiment.add_batch_metric("mse", batch_mse, self.run_count)
                experiment.add_batch_metric("mae", batch_mae, self.run_count)
                experiment.add_batch_metric("r2", batch_r2, self.run_count) # Add r2 metric

            if self.optimizer:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _run_single(self, x: Any, y: Any) -> tuple[torch.Tensor, float, float]:
        """
        Processes a single batch of data.

        Parameters:
        x (Any): Input features.
        y (Any): Target labels.

        Returns:
        tuple: Tuple containing the loss, batch MSE, and batch MAE.
        """
        self.run_count += 1
        batch_size: int = x.shape[0]
        prediction = self.model(x).squeeze()

        # Flatten y to match the shape of prediction
        y = y.view(-1)

        if torch.isnan(prediction).any():
            print("NaN values found in predictions")
            return torch.tensor(0.0), float('nan'), float('nan')  # Skip this batch if it contains NaN values

        loss = self.compute_loss(prediction, y.float())

        y_np = y.detach().numpy()
        y_prediction_np = prediction.detach().numpy()
        
        if np.isnan(y_np).any() or np.isnan(y_prediction_np).any():
            print("NaN values found in true labels or predictions")
            return loss, float('nan'), float('nan')  # Skip this batch if it contains NaN values

        batch_mse: float = mean_squared_error(y_np, y_prediction_np)
        batch_mae: float = mean_absolute_error(y_np, y_prediction_np)
        batch_r2: float = r2_score(y_np, y_prediction_np)# Add r2_score calculation

        self.mse_metric.update(batch_mse, batch_size)
        self.mae_metric.update(batch_mae, batch_size)
        self.r2_metric.update(batch_r2, batch_size)# Add r2_metric update

        self.y_true_batches += [y_np]
        self.y_pred_batches += [y_prediction_np]
        return loss, batch_mse, batch_mae, batch_r2 # Add batch_r2

    def reset(self) -> None:
        """Resets the metrics and stored batches."""
        self.mse_metric = Metric()
        self.mae_metric = Metric()
        self.r2_metric = Metric()  # Reset R-squared metric
        self.y_true_batches = []
        self.y_pred_batches = []

def run_epoch(
    test_runner: Runner,
    train_runner: Runner,
    experiment: ExperimentTracker,
    epoch_id: int,
) -> None:
    """
    Orchestrates the training and validation process for a single epoch.

    Parameters:
    test_runner (Runner): The runner for the validation phase.
    train_runner (Runner): The runner for the training phase.
    experiment (ExperimentTracker): The experiment tracker for logging metrics.
    epoch_id (int): The current epoch ID.
    """
    experiment.set_stage(Stage.TRAIN)
    train_runner.run("Train Batches", experiment)

    experiment.add_epoch_metric("mse", train_runner.avg_mse, epoch_id)
    experiment.add_epoch_metric("mae", train_runner.avg_mae, epoch_id)
    experiment.add_epoch_metric("r2", train_runner.avg_r2, epoch_id)  # Add R-squared metric
    

    experiment.set_stage(Stage.VAL)
    test_runner.run("Validation Batches", experiment)

    experiment.add_epoch_metric("mse", test_runner.avg_mse, epoch_id)
    experiment.add_epoch_metric("mae", test_runner.avg_mae, epoch_id)
    experiment.add_epoch_metric("r2", test_runner.avg_r2, epoch_id)# Add R-squared metric
    experiment.add_epoch_scatter_plot(
        test_runner.y_true_batches, test_runner.y_pred_batches, epoch_id
    )# Add scatter plot
    #experiment.add_epoch_confusion_matrix(
        #test_runner.y_true_batches, test_runner.y_pred_batches, epoch_id
   # )
