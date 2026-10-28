"""Standalone and library evaluation module for VGG6 on CIFAR-10."""

import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import get_cifar10_dataloaders
from src.model import CFG_VGG6, vgg
from src.utils import get_device, set_seed


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
  """Evaluates average Cross-Entropy loss and Top-1 accuracy on a DataLoader.

  Args:
      model: Trained VGG6 model.
      data_loader: Validation/test or training DataLoader.
      criterion: Loss function (e.g., nn.CrossEntropyLoss()).
      device: Computation device (CPU or CUDA).

  Returns:
      Tuple of (avg_loss, top1_accuracy_percent).
  """
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for inputs, targets in data_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      running_loss += loss.item() * targets.size(0)
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

  avg_loss = running_loss / max(total, 1)
  acc = 100.0 * correct / max(total, 1)
  return avg_loss, acc


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Evaluate a trained VGG6 checkpoint on CIFAR-10 test set."
  )
  parser.add_argument(
      "--checkpoint",
      type=str,
      default="checkpoints/best_vgg6_cifar10.pth",
      help="Path to the saved .pth model checkpoint.",
  )
  parser.add_argument(
      "--data-dir",
      type=str,
      default="./data",
      help="Path to CIFAR-10 dataset directory.",
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=128,
      help="Evaluation mini-batch size.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for reproducibility.",
  )
  args = parser.parse_args()

  set_seed(args.seed)
  device = get_device("auto")
  ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
  config = ckpt.get("config", {})
  activation = config.get("activation", "gelu")
  batch_norm = config.get("batch_norm", True)

  model = vgg(
      CFG_VGG6,
      num_classes=10,
      batch_norm=batch_norm,
      activation=activation,
  ).to(device)
  model.load_state_dict(ckpt["model_state_dict"])

  _, test_loader = get_cifar10_dataloaders(
      batch_size=args.batch_size,
      data_dir=args.data_dir,
      seed=args.seed,
  )
  criterion = nn.CrossEntropyLoss()
  test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

  print("=" * 70)
  print(f"Evaluated Checkpoint : {args.checkpoint}")
  print(f"Configuration        : {config}")
  print(f"Saved Epoch          : {ckpt.get('epoch', 'N/A')}")
  print(f"Test Loss            : {test_loss:.4f}")
  print(f"Test Top-1 Accuracy  : {test_acc:.2f}%")
  print("=" * 70)


if __name__ == "__main__":
  main()
