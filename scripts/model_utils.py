import time
import os
import torch
from typing import List, Tuple
from tqdm.auto import tqdm


class EarlyStopping:
    """
    Implements early stopping logic for model training.

    This class monitors the validation loss over epochs and triggers
    early stopping if no sufficient improvement is observed for a specified number
    of consecutive epochs.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as improvement.
        early_stop (bool): Whether training should be stopped early.
        best_loss (float): Best validation loss seen so far.
        counter (int): Number of consecutive epochs without improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.00001):
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (float): Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def __call__(self, val_loss: float):
        """
        Updates the early stopping state based on the current validation loss.

        Args:
            val_loss (float): Current epoch's validation loss.

        Notes:
            If the validation loss improves more than `min_delta`, the counter resets.
            Otherwise, the counter increases. If it reaches `patience`, early stopping is triggered.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self):
        """
        Resets the early stopping state.
        Useful if the same EarlyStopping instance is reused across multiple training runs.
        """
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str = 'cpu') -> float:
    """
    Executes a single training epoch over the provided DataLoader.

    This function performs a full training pass using the given model, loss function,
    and optimizer. After each batch, gradients are computed and model parameters are updated.
    The function computes the average loss across all training samples in this epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): Dataloader providing the training data.
        loss_fn (torch.nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer that updates model parameters.
        device (str, optional): Device on which computations will be performed ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        float: average loss across all batches in this epoch.

    Notes:
        - This function computes the average of the **per-batch** loss values.
        - The average is taken over the number of batches, not the number of individual samples.
        - If batch sizes vary and precise loss per sample is desired, you may need a weighted approach.
        - Gradient clipping with max_norm=5 is applied to prevent exploding gradients.
    """
    avg_loss = 0

    model.train()

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

    avg_loss /= len(dataloader)

    return avg_loss


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str = 'cpu') -> Tuple[float, float]:
    """
    Evaluates the model on a test or validation set for a single epoch.

    This function disables gradient computation and switches the model to evaluation mode
    using `model.eval()` and `torch.inference_mode()`. It  computes the average loss
    across all samples in the given DataLoader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader providing the evaluation data.
        loss_fn (torch.nn.Module): Loss function used for evaluation.
        device (str, optional): Device on which computations will be performed ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        float: average loss across all batches in this epoch.

    Notes:
        - This function computes the average of the **per-batch** loss values.
        - The average is taken over the number of batches, not the number of individual samples.
        - If batch sizes vary and precise loss per sample is desired, you may need a weighted approach.
        - No parameter updates or gradient calculations are performed during this step.
    """
    avg_loss = 0

    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            avg_loss += loss.item()

    avg_loss /= len(dataloader)

    return avg_loss


def training_loop(model: torch.nn.Module,
                  train_dataloader: torch.utils.data.DataLoader,
                  test_dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  num_epochs: int,
                  log_every_n_epochs: int = None,
                  early_stopping=None,
                  device: str = 'cpu') -> Tuple[List[float], List[float], List[float]]:
    """
    Runs the training and evaluation process over multiple epochs.

    This function executes a full training loop consisting of:
    - forward/backward passes on the training set,
    - evaluation on the test or val set,
    - optional logging of losses.

    Args:
        model (torch.nn.Module): The model to be trained and evaluated.
        train_dataloader (DataLoader): DataLoader providing training batches.
        test_dataloader (DataLoader): DataLoader providing test/validation batches.
        loss_fn (nn.Module): Loss function used for optimization and evaluation.
        optimizer (Optimizer): Optimizer to update model parameters.
        num_epochs (int): Total number of training epochs.
        log_every_n_epochs (int, optional): If set to a positive integer, prints metrics every N epochs.
        early_stopping (Optional[EarlyStopping], optional): An instance of EarlyStopping. If provided, training stops early when validation loss stops improving.
        device (str): Device on which computations will be performed ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Tuple [List, List, List]: A tuple containing:
            - train_losses: average training loss per epoch,
            - test_losses: average test/validation loss per epoch,
            - epoch_times: duration of each epoch in seconds.

    Raises:
        AssertionError: If `num_epochs` or `log_every_n_epochs` are invalid types or values.

    Notes:
        - Losses are tracked at the epoch level using `train_step()` and `test_step()` functions.
        - `log_every_n_epochs` controls how often epoch summaries are printed.
        - The model is assumed to already be on the correct device.
        - If early_stopping is provided and triggered, training loop will terminate before reaching num_epochs.
        - The best model weights (based on test/validation loss) are saved automatically to the models/ directory.
    """

    assert log_every_n_epochs is None or (log_every_n_epochs > 0 and isinstance(log_every_n_epochs, int)), \
        'log_every_n_epochs must be an integer number greater than zero'
    assert num_epochs > 0 and isinstance(num_epochs, int), \
        'epochs must be an integer number greater than zero'

    epoch_pad = len(str(num_epochs+1))

    train_losses = []
    test_losses = []
    epoch_times = []
    best_test_loss = 10e8

    os.makedirs("models", exist_ok=True)

    for epoch in tqdm(range(num_epochs)):

        start = time.perf_counter()

        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if log_every_n_epochs is not None and (epoch % log_every_n_epochs) == 0:
            print(f'Epoch: {epoch+1:<{epoch_pad}} | ', end='')
            print(f'Train loss: {round(train_loss, 4):<6} | ', end='')
            print(f'Test loss: {round(test_loss, 4):<6} | ', end='')

        if early_stopping is not None:
            early_stopping(test_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model_name = model.__class__.__name__
            save_path = f"models/{model_name}_best.pth"
            model.save(save_path)

        duration = time.perf_counter() - start
        epoch_times.append(duration)

    return train_losses, test_losses, epoch_times