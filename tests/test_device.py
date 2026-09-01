import torch

from src.utilities.device import get_device


def test_get_device_returns_torch_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_honors_override() -> None:
    assert get_device(prefer="cpu") == torch.device("cpu")
