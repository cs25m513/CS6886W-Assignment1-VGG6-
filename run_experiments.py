"""Orchestrates the full VGG6 CIFAR-10 experiment suite and W&B sweep logging."""

import concurrent.futures
import multiprocessing
import os
import time
from typing import Any, Dict, List


def _worker_run(task: Dict[str, Any]) -> Dict[str, Any]:
  """Executes a single training run inside a clean spawned worker process."""
  threads = int(task.get("threads", 4))
  os.environ["OMP_NUM_THREADS"] = str(threads)
  os.environ["MKL_NUM_THREADS"] = str(threads)

  import torch
  from src.train import run_training_experiment

  torch.set_num_threads(threads)
  torch.set_flush_denormal(True)

  config = task["config"]
  res = run_training_experiment(
      config=config,
      data_dir="./data",
      num_workers=0,
      save_best_path=task.get("save_best_path"),
      use_wandb=True,
      wandb_mode="offline",
      subset_size=int(task.get("subset_size", 3000)),
      val_subset_size=int(task.get("val_subset_size", 1500)),
  )
  res["group"] = task.get("group", "sweep")
  return res


def build_ablation_tasks() -> List[Dict[str, Any]]:
  """Builds the comparative ablation tasks for Q2(a,b,c) and Q3."""
  tasks: List[Dict[str, Any]] = []

  # Question 2(a): Activation Function Ablation (fixing opt=nadam, lr=0.001, bs=64, bn=True, epochs=8)
  for act in ["relu", "sigmoid", "tanh", "silu", "gelu", "leaky_relu", "selu"]:
    tasks.append({
        "group": "activation_ablation",
        "threads": 4,
        "subset_size": 3000,
        "val_subset_size": 1500,
        "save_best_path": None,
        "config": {
            "run_name": f"act_{act}_bs_64_opt_nadam_lr_0.001_wd_0.0005_bn_True",
            "activation": act,
            "optimizer": "nadam",
            "batch_size": 64,
            "epochs": 8,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "batch_norm": True,
            "use_cutout": True,
            "use_cosine_lr": True,
            "seed": 42,
        },
    })

  # Question 2(b): Optimizer Ablation (fixing act=gelu, bs=64, bn=True, epochs=8)
  opt_lrs = [
      ("sgd", 0.01, 0.9),
      ("nesterov", 0.01, 0.9),
      ("adam", 0.001, 0.9),
      ("adagrad", 0.01, 0.0),
      ("rmsprop", 0.001, 0.0),
      ("nadam", 0.001, 0.9),
  ]
  for opt_name, lr_val, mom_val in opt_lrs:
    tasks.append({
        "group": "optimizer_ablation",
        "threads": 4,
        "subset_size": 3000,
        "val_subset_size": 1500,
        "save_best_path": None,
        "config": {
            "run_name": f"act_gelu_bs_64_opt_{opt_name}_lr_{lr_val}_wd_0.0005_bn_True",
            "activation": "gelu",
            "optimizer": opt_name,
            "batch_size": 64,
            "epochs": 8,
            "learning_rate": lr_val,
            "momentum": mom_val,
            "weight_decay": 0.0005,
            "batch_norm": True,
            "use_cutout": True,
            "use_cosine_lr": True,
            "seed": 42,
        },
    })

  # Question 2(c)-i: Batch Size Ablation (32, 64, 128, 256, 512)
  for bs in [32, 64, 128, 256, 512]:
    tasks.append({
        "group": "batch_size_ablation",
        "threads": 4,
        "subset_size": 3000,
        "val_subset_size": 1500,
        "save_best_path": None,
        "config": {
            "run_name": f"act_gelu_bs_{bs}_opt_nadam_lr_0.001_wd_0.0005_bn_True",
            "activation": "gelu",
            "optimizer": "nadam",
            "batch_size": bs,
            "epochs": 8,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "batch_norm": True,
            "use_cutout": True,
            "use_cosine_lr": True,
            "seed": 42,
        },
    })

  # Question 2(c)-ii: Learning Rate Ablation (0.0001, 0.0005, 0.001, 0.005, 0.01)
  for lr_val in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
    tasks.append({
        "group": "lr_ablation",
        "threads": 4,
        "subset_size": 3000,
        "val_subset_size": 1500,
        "save_best_path": None,
        "config": {
            "run_name": f"act_gelu_bs_64_opt_nadam_lr_{lr_val}_wd_0.0005_bn_True",
            "activation": "gelu",
            "optimizer": "nadam",
            "batch_size": 64,
            "epochs": 8,
            "learning_rate": lr_val,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "batch_norm": True,
            "use_cutout": True,
            "use_cosine_lr": True,
            "seed": 42,
        },
    })

  # Question 2(c)-iii: Epoch Count Ablation (4, 6, 8, 12)
  for ep_val in [4, 6, 8, 12]:
    tasks.append({
        "group": "epoch_ablation",
        "threads": 4,
        "subset_size": 3000,
        "val_subset_size": 1500,
        "save_best_path": None,
        "config": {
            "run_name": f"act_gelu_bs_64_opt_nadam_lr_0.001_ep_{ep_val}_bn_True",
            "activation": "gelu",
            "optimizer": "nadam",
            "batch_size": 64,
            "epochs": ep_val,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "batch_norm": True,
            "use_cutout": True,
            "use_cosine_lr": True,
            "seed": 42,
        },
    })

  # Extra diverse W&B sweep configurations (including BatchNorm=False)
  extra_sweeps = [
      ("silu", False, 512, 8, 0.0001, 0.0, "nadam", 0.00005),
      ("leaky_relu", False, 256, 8, 0.001, 0.5, "adam", 0.0001),
      ("relu", False, 128, 8, 0.005, 0.9, "sgd", 0.0002),
  ]
  for act, bn, bs, ep, lr_v, mom, opt_n, wd in extra_sweeps:
    tasks.append({
        "group": "extra_sweep",
        "threads": 4,
        "subset_size": 3000,
        "val_subset_size": 1500,
        "save_best_path": None,
        "config": {
            "run_name": f"act_{act}_bs_{bs}_opt_{opt_n}_lr_{lr_v}_wd_{wd}_bn_{bn}",
            "activation": act,
            "optimizer": opt_n,
            "batch_size": bs,
            "epochs": ep,
            "learning_rate": lr_v,
            "momentum": mom,
            "weight_decay": wd,
            "batch_norm": bn,
            "use_cutout": True,
            "use_cosine_lr": True,
            "seed": 42,
        },
    })

  return tasks


def main() -> None:
  t0 = time.time()
  mp_ctx = multiprocessing.get_context("spawn")

  # Step 1: Run Best Baseline Configuration with all 24 CPU cores on full 10,000 test set
  best_task = {
      "group": "best_baseline",
      "threads": 24,
      "subset_size": 15000,
      "val_subset_size": 0,  # Full 10,000 CIFAR-10 test set!
      "save_best_path": "checkpoints/best_vgg6_cifar10.pth",
      "config": {
          "run_name": "act_gelu_bs_64_opt_nadam_lr_0.001_wd_0.0005_bn_True",
          "activation": "gelu",
          "optimizer": "nadam",
          "batch_size": 64,
          "epochs": 8,
          "learning_rate": 0.001,
          "momentum": 0.9,
          "weight_decay": 0.0005,
          "batch_norm": True,
          "use_cutout": True,
          "use_cosine_lr": True,
          "seed": 42,
      },
  }

  print("Step 1/2: Training Best Baseline Model on 24 cores and saving checkpoint...", flush=True)
  with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as ex:
    best_res = ex.submit(_worker_run, best_task).result()
  print(
      f"[BEST BASELINE DONE] {best_res['run_name']} -> Best Test Acc:"
      f" {best_res['best_val_acc']:.2f}% ({best_res['total_time_sec']:.1f}s)",
      flush=True,
  )

  # Step 2: Run all comparative ablation runs in parallel across 6 spawned processes (4 threads each)
  ablation_tasks = build_ablation_tasks()
  print(
      f"\nStep 2/2: Launching {len(ablation_tasks)} ablation/sweep runs across"
      " 6 spawned workers (4 threads each)...",
      flush=True,
  )
  results: List[Dict[str, Any]] = [best_res]
  with concurrent.futures.ProcessPoolExecutor(max_workers=6, mp_context=mp_ctx) as executor:
    futures = [executor.submit(_worker_run, task) for task in ablation_tasks]
    for fut in concurrent.futures.as_completed(futures):
      res = fut.result()
      results.append(res)
      print(
          f"[COMPLETED] ({len(results)}/{len(ablation_tasks)+1})"
          f" {res['run_name']} [{res['group']}] -> Best Val Acc:"
          f" {res['best_val_acc']:.2f}% ({res['total_time_sec']:.1f}s)",
          flush=True,
      )

  all_tasks = [best_task] + ablation_tasks
  task_order = [(t["group"], t["config"]["run_name"]) for t in all_tasks]
  results.sort(
      key=lambda r: task_order.index((r["group"], r["run_name"]))
      if (r["group"], r["run_name"]) in task_order
      else 999
  )

  from generate_plots import generate_all_plots
  from src.utils import save_json

  best_overall = max(results, key=lambda r: r["best_val_acc"])
  summary = {
      "total_runs": len(results),
      "total_wall_time_sec": round(time.time() - t0, 2),
      "best_run": best_overall,
      "runs": results,
  }
  save_json(summary, "results/all_experiments.json")
  generate_all_plots("results/all_experiments.json", "plots")
  print(
      f"\n=== ALL {len(results)} EXPERIMENTS COMPLETED IN"
      f" {time.time() - t0:.1f}s ===",
      flush=True,
  )
  print(
      f"Best Configuration: {best_overall['run_name']} ->"
      f" {best_overall['best_val_acc']:.2f}%",
      flush=True,
  )


if __name__ == "__main__":
  main()
