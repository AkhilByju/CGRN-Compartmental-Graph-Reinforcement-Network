import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.regression import mae
from src.training.trainer import evaluate_loop, train_loop
from src.utilities.device import get_device


def test_train_loop_reduces_loss_on_linear_regression() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 1)
    y = 3 * x + 2

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    device = get_device(prefer="cpu")
    loss_fn = torch.nn.MSELoss()

    initial_loss = loss_fn(model(x), y).item()
    train_loop(model, dataloader, optimizer, loss_fn, device, max_epochs=50)
    final_loss = loss_fn(model(x), y).item()

    assert final_loss < initial_loss


def test_evaluate_loop_computes_metrics() -> None:
    x = torch.randn(32, 1)
    y = 3 * x + 2
    model = torch.nn.Linear(1, 1)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=8)
    device = get_device(prefer="cpu")

    metrics = evaluate_loop(model, dataloader, {"mae": mae}, device)

    assert "mae" in metrics
    assert metrics["mae"] >= 0.0
