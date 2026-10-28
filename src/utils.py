"""Reproducibility, seeding, device management, and checkpoint utilities."""

import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
  """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility.

  Args:
      seed: Integer seed value (default: 42).
  """
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
  """Worker initialization function for deterministic PyTorch DataLoaders."""
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def get_device(device_str: str = "auto") -> torch.device:
  """Selects computation device (CUDA GPU if available, otherwise CPU)."""
  if device_str == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(device_str)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    epoch: int,
    val_acc: float,
    val_loss: float,
    filepath: str,
) -> None:
  """Saves model weights, optimizer state, and configuration to disk."""
  os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
  state = {
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "val_acc": val_acc,
      "val_loss": val_loss,
      "config": config,
  }
  torch.save(state, filepath)


def save_json(data: Any, filepath: str) -> None:
  """Saves dictionary or list data as formatted JSON."""
  os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
  with open(filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
