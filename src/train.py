"""Training module and CLI entry point for VGG6 on CIFAR-10."""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import get_cifar10_dataloaders
from src.evaluate import evaluate_model
from src.model import CFG_VGG6, vgg
from src.utils import get_device, save_checkpoint, save_json, set_seed


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
) -> optim.Optimizer:
  """Instantiates the requested PyTorch optimizer for VGG6 parameters.

  Supported optimizers:
  - sgd: Standard Stochastic Gradient Descent (with momentum)
  - nesterov: Nesterov Accelerated Gradient Descent (nesterov=True)
  - adam: Adaptive Moment Estimation
  - adagrad: Adaptive Gradient Algorithm
  - rmsprop: Root Mean Square Propagation
  - nadam: Nesterov-accelerated Adaptive Moment Estimation
  """
  opt = optimizer_name.lower().strip()
  params = model.parameters()

  if opt == "sgd":
    return optim.SGD(
        params,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=False,
    )
  elif opt in ("nesterov", "nesterov-sgd", "nesterov_sgd"):
    eff_momentum = momentum if momentum > 0.0 else 0.9
    return optim.SGD(
        params,
        lr=learning_rate,
        momentum=eff_momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
  elif opt == "adam":
    return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
  elif opt == "adagrad":
    return optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
  elif opt == "rmsprop":
    return optim.RMSprop(
        params,
        lr=learning_rate,
        momentum=momentum if momentum > 0.0 else 0.0,
        weight_decay=weight_decay,
    )
  elif opt == "nadam":
    return optim.NAdam(params, lr=learning_rate, weight_decay=weight_decay)
  else:
    raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. Choose from: sgd, nesterov,"
        " adam, adagrad, rmsprop, nadam."
    )


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
  """Trains VGG6 for one epoch and returns (train_loss, train_acc)."""
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * targets.size(0)
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

  avg_loss = running_loss / max(total, 1)
  acc = 100.0 * correct / max(total, 1)
  return avg_loss, acc


def run_training_experiment(
    config: Dict[str, Any],
    data_dir: str = "./data",
    num_workers: int = 0,
    save_best_path: Optional[str] = None,
    use_wandb: bool = False,
    wandb_mode: str = "offline",
    wandb_project: str = "CS6886W-VGG6-CIFAR10",
    subset_size: int = 0,
    val_subset_size: int = 0,
) -> Dict[str, Any]:
  """Executes a full VGG6 training and validation run for a given configuration.

  Args:
      config: Dictionary containing hyperparameters ('activation', 'batch_norm',
        'batch_size', 'epochs', 'learning_rate', 'momentum', 'optimizer',
        'weight_decay', 'seed', 'use_cutout').
      data_dir: Directory containing CIFAR-10 dataset.
      num_workers: DataLoader workers.
      save_best_path: Optional path to save best model checkpoint (.pth).
      use_wandb: Whether to initialize and log metrics to Weights & Biases.
      wandb_mode: W&B mode ('online', 'offline', 'disabled').
      wandb_project: W&B project name.
      subset_size: Optional training subset size (0 uses all 50,000 samples).

  Returns:
      Dictionary containing run metadata, best metrics, and per-epoch history.
  """
  seed = int(config.get("seed", 42))
  set_seed(seed)
  device = get_device("auto")

  activation = str(config.get("activation", "relu"))
  batch_norm = bool(config.get("batch_norm", True))
  batch_size = int(config.get("batch_size", 64))
  epochs = int(config.get("epochs", 20))
  learning_rate = float(config.get("learning_rate", 0.001))
  momentum = float(config.get("momentum", 0.9))
  optimizer_name = str(config.get("optimizer", "adam"))
  weight_decay = float(config.get("weight_decay", 0.0005))
  use_cutout = bool(config.get("use_cutout", True))
  use_cosine_lr = bool(config.get("use_cosine_lr", True))

  run_name = config.get(
      "run_name",
      f"act_{activation}_bs_{batch_size}_opt_{optimizer_name}_lr_{learning_rate}_wd_{weight_decay}_bn_{batch_norm}",
  )

  wandb_run = None
  if use_wandb:
    import wandb

    os.environ["WANDB_MODE"] = wandb_mode
    os.environ["WANDB_SILENT"] = "true"
    wandb_run = wandb.init(
        project=wandb_project,
        name=run_name,
        config=config,
        mode=wandb_mode,
        reinit=True,
    )

  train_loader, val_loader = get_cifar10_dataloaders(
      batch_size=batch_size,
      data_dir=data_dir,
      num_workers=num_workers,
      use_cutout=use_cutout,
      seed=seed,
      subset_size=subset_size,
      val_subset_size=val_subset_size,
  )

  model = vgg(
      CFG_VGG6,
      num_classes=10,
      batch_norm=batch_norm,
      activation=activation,
  ).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = build_optimizer(
      model=model,
      optimizer_name=optimizer_name,
      learning_rate=learning_rate,
      momentum=momentum,
      weight_decay=weight_decay,
  )
  scheduler = (
      optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
      if use_cosine_lr
      else None
  )

  history: List[Dict[str, float]] = []
  best_val_acc = -1.0
  best_val_loss = float("inf")
  best_epoch = -1
  start_time = time.time()

  print(
      f"\n[RUN START] {run_name} | device={device} | epochs={epochs} |"
      f" batch_size={batch_size} | opt={optimizer_name} | lr={learning_rate} |"
      f" act={activation} | bn={batch_norm}",
      flush=True,
  )

  for epoch in range(epochs):
    ep_start = time.time()
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    if scheduler is not None:
      scheduler.step()
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    ep_time = time.time() - ep_start

    step_record = {
        "step": epoch,
        "epoch": epoch + 1,
        "train_loss": round(train_loss, 5),
        "train_acc": round(train_acc, 2),
        "val_loss": round(val_loss, 5),
        "val_acc": round(val_acc, 2),
        "epoch_time_sec": round(ep_time, 2),
    }
    history.append(step_record)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      best_val_loss = val_loss
      best_epoch = epoch + 1
      if save_best_path:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=best_epoch,
            val_acc=best_val_acc,
            val_loss=best_val_loss,
            filepath=save_best_path,
        )

    if wandb_run is not None:
      import wandb

      wandb.log({
          "step": epoch,
          "epoch": epoch + 1,
          "train_loss": train_loss,
          "train_acc": train_acc,
          "val_loss": val_loss,
          "val_acc": val_acc,
          "best_val_acc": best_val_acc,
      })

    print(
        f"  Epoch {epoch+1:02d}/{epochs:02d} - Train_Loss: {train_loss:.4f} ,"
        f" Train_acc: {train_acc:.2f}% , Val_Loss: {val_loss:.4f} , Val_acc:"
        f" {val_acc:.2f}% (Best: {best_val_acc:.2f}%)"
        f" [{ep_time:.1f}s]"
    )

  total_time = time.time() - start_time
  if wandb_run is not None:
    wandb_run.summary["best_val_acc"] = best_val_acc
    wandb_run.summary["best_val_loss"] = best_val_loss
    wandb_run.summary["best_epoch"] = best_epoch
    wandb_run.finish()

  return {
      "run_name": run_name,
      "config": config,
      "best_val_acc": round(best_val_acc, 2),
      "best_val_loss": round(best_val_loss, 5),
      "best_epoch": best_epoch,
      "final_train_acc": history[-1]["train_acc"] if history else 0.0,
      "final_train_loss": history[-1]["train_loss"] if history else 0.0,
      "final_val_acc": history[-1]["val_acc"] if history else 0.0,
      "final_val_loss": history[-1]["val_loss"] if history else 0.0,
      "total_time_sec": round(total_time, 2),
      "created_timestamp": time.strftime("%b %d '%y %H:%M:%S"),
      "history": history,
  }


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Train VGG6 on CIFAR-10 with configurable hyperparameters."
  )
  parser.add_argument(
      "--activation",
      type=str,
      default="gelu",
      choices=[
          "relu",
          "sigmoid",
          "tanh",
          "silu",
          "gelu",
          "leaky_relu",
          "selu",
      ],
      help="Activation function inside VGG6 convolutional blocks.",
  )
  parser.add_argument(
      "--optimizer",
      type=str,
      default="nadam",
      choices=["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"],
      help="Optimization algorithm.",
  )
  parser.add_argument(
      "--batch-size", type=int, default=64, help="Training mini-batch size."
  )
  parser.add_argument(
      "--epochs", type=int, default=8, help="Number of training epochs."
  )
  parser.add_argument(
      "--lr", type=float, default=0.001, help="Initial learning rate."
  )
  parser.add_argument(
      "--momentum",
      type=float,
      default=0.9,
      help="Momentum factor (for SGD, Nesterov-SGD, RMSprop).",
  )
  parser.add_argument(
      "--weight-decay",
      type=float,
      default=0.0005,
      help="L2 weight decay regularization.",
  )
  parser.add_argument(
      "--batch-norm",
      type=str,
      default="true",
      choices=["true", "false"],
      help="Whether to use BatchNorm2d after Conv2d layers.",
  )
  parser.add_argument(
      "--use-cutout",
      type=str,
      default="true",
      choices=["true", "false"],
      help="Whether to use Cutout(1, 16) data augmentation.",
  )
  parser.add_argument(
      "--subset-size",
      type=int,
      default=15000,
      help="Training subset size (0 uses all 50,000 CIFAR-10 images).",
  )
  parser.add_argument(
      "--seed", type=int, default=42, help="Random seed for reproducibility."
  )
  parser.add_argument(
      "--data-dir",
      type=str,
      default="./data",
      help="Directory to download/load CIFAR-10.",
  )
  parser.add_argument(
      "--save-checkpoint",
      type=str,
      default="checkpoints/best_vgg6_cifar10.pth",
      help="Filepath to save best validation accuracy checkpoint.",
  )
  parser.add_argument(
      "--save-metrics",
      type=str,
      default="results/single_run_metrics.json",
      help="Filepath to save JSON training history.",
  )
  parser.add_argument(
      "--use-wandb",
      action="store_true",
      help="Enable Weights & Biases logging.",
  )
  parser.add_argument(
      "--wandb-mode",
      type=str,
      default="offline",
      choices=["online", "offline", "disabled"],
      help="W&B logging mode.",
  )
  args = parser.parse_args()

  config = {
      "activation": args.activation,
      "optimizer": args.optimizer,
      "batch_size": args.batch_size,
      "epochs": args.epochs,
      "learning_rate": args.lr,
      "momentum": args.momentum,
      "weight_decay": args.weight_decay,
      "batch_norm": args.batch_norm.lower() == "true",
      "use_cutout": args.use_cutout.lower() == "true",
      "seed": args.seed,
  }

  result = run_training_experiment(
      config=config,
      data_dir=args.data_dir,
      save_best_path=args.save_checkpoint,
      use_wandb=args.use_wandb,
      wandb_mode=args.wandb_mode,
      subset_size=args.subset_size,
  )
  save_json(result, args.save_metrics)
  print(
      f"\n[DONE] Best Validation/Test Top-1 Accuracy: {result['best_val_acc']:.2f}%"
      f" (Epoch {result['best_epoch']})"
  )


if __name__ == "__main__":
  main()
