"""Weights & Biases (W&B) Sweep Agent Entry Point for VGG6 on CIFAR-10."""

import wandb
from src.train import run_training_experiment


def sweep_train() -> None:
  """Single W&B sweep trial entry point."""
  run = wandb.init()
  cfg = dict(wandb.config)
  config = {
      "activation": cfg.get("activation", "gelu"),
      "batch_norm": bool(cfg.get("batch_norm", True)),
      "batch_size": int(cfg.get("batch_size", 64)),
      "epochs": int(cfg.get("epochs", 50)),
      "learning_rate": float(cfg.get("learning_rate", 0.001)),
      "momentum": float(cfg.get("momentum", 0.9)),
      "optimizer": cfg.get("optimizer", "nadam"),
      "weight_decay": float(cfg.get("weight_decay", 0.0005)),
      "use_cutout": True,
      "seed": int(cfg.get("seed", 42)),
  }
  run.name = (
      f"act_{config['activation']}_bs_{config['batch_size']}_"
      f"opt_{config['optimizer']}_lr_{config['learning_rate']}_"
      f"wd_{config['weight_decay']}_bn_{config['batch_norm']}"
  )
  run.finish()

  run_training_experiment(
      config=config,
      use_wandb=True,
      wandb_mode="online",
  )


if __name__ == "__main__":
  sweep_train()
