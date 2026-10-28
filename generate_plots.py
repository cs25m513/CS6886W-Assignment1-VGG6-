"""Generates all W&B-style plots and ablation charts for CS6886W Assignment 1."""

import json
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


WANDB_PALETTE = [
    "#e87461",
    "#5b8ff9",
    "#5ad8a6",
    "#9270ca",
    "#ff9d4d",
    "#269a99",
    "#f6bd16",
    "#6dc8ec",
    "#ff85c0",
    "#5d7092",
]


def setup_wandb_style() -> None:
  """Configures Matplotlib aesthetics to match Weights & Biases charts."""
  plt.rcParams["font.sans-serif"] = "DejaVu Sans"
  plt.rcParams["axes.edgecolor"] = "#d9d9d9"
  plt.rcParams["axes.linewidth"] = 0.8
  plt.rcParams["grid.color"] = "#ebebeb"
  plt.rcParams["grid.linestyle"] = "-"
  plt.rcParams["grid.linewidth"] = 0.7


def plot_q1_baseline_curves(best_run: Dict[str, Any], out_dir: str) -> None:
  """Generates Question 1(c) loss and accuracy curves for the best baseline run."""
  setup_wandb_style()
  history = best_run["history"]
  epochs = [h["epoch"] for h in history]
  train_loss = [h["train_loss"] for h in history]
  val_loss = [h["val_loss"] for h in history]
  train_acc = [h["train_acc"] for h in history]
  val_acc = [h["val_acc"] for h in history]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=220)

  ax1.plot(epochs, train_loss, label="Train Loss", color="#5b8ff9", linewidth=2.0)
  ax1.plot(epochs, val_loss, label="Val/Test Loss", color="#e87461", linewidth=2.0)
  ax1.set_title("VGG6 Baseline: Cross-Entropy Loss vs. Epoch", fontsize=11, fontweight="bold")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Loss")
  ax1.grid(True)
  ax1.legend(frameon=True, facecolor="white")

  ax2.plot(epochs, train_acc, label="Train Top-1 Accuracy", color="#5b8ff9", linewidth=2.0)
  ax2.plot(epochs, val_acc, label=f"Val/Test Top-1 Accuracy (Best: {best_run['best_val_acc']:.2f}%)", color="#269a99", linewidth=2.0)
  ax2.axhline(best_run["best_val_acc"], color="#269a99", linestyle="--", alpha=0.5, linewidth=1.0)
  ax2.set_title("VGG6 Baseline: Top-1 Accuracy (%) vs. Epoch", fontsize=11, fontweight="bold")
  ax2.set_xlabel("Epoch")
  ax2.set_ylabel("Accuracy (%)")
  ax2.grid(True)
  ax2.legend(frameon=True, facecolor="white", loc="lower right")

  fig.suptitle(
      f"Baseline Best Configuration: {best_run['run_name']}",
      fontsize=12,
      fontweight="bold",
      y=1.02,
  )
  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, "q1_baseline_curves.png"), bbox_inches="tight")
  plt.close(fig)


def plot_q2a_activations(runs: List[Dict[str, Any]], out_dir: str) -> None:
  """Generates Question 2(a) comparison across activation functions."""
  setup_wandb_style()
  act_runs = [r for r in runs if r.get("group") == "activation_ablation"]
  if not act_runs:
    return

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=220)

  for idx, r in enumerate(act_runs):
    color = WANDB_PALETTE[idx % len(WANDB_PALETTE)]
    act_name = r["config"]["activation"].upper()
    steps = [h["epoch"] for h in r["history"]]
    val_acc = [h["val_acc"] for h in r["history"]]
    ax1.plot(
        steps,
        val_acc,
        label=f"{act_name} (Best: {r['best_val_acc']:.2f}%)",
        color=color,
        linewidth=1.9,
    )

  ax1.set_title("Validation Accuracy vs. Epoch by Activation Function", fontsize=11, fontweight="bold")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Validation Top-1 Accuracy (%)")
  ax1.grid(True)
  ax1.legend(fontsize=8.5, loc="lower right")

  names = [r["config"]["activation"].upper() for r in act_runs]
  bests = [r["best_val_acc"] for r in act_runs]
  bars = ax2.bar(names, bests, color=WANDB_PALETTE[: len(names)], edgecolor="#333333", linewidth=0.6)
  ax2.set_title("Best Top-1 Validation Accuracy (%) per Activation", fontsize=11, fontweight="bold")
  ax2.set_ylabel("Best Validation Accuracy (%)")
  ax2.set_ylim(0, 100)
  ax2.grid(True, axis="y")
  for bar, val in zip(bars, bests):
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        val + 1.2,
        f"{val:.2f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )

  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, "q2a_activation_comparison.png"), bbox_inches="tight")
  plt.close(fig)


def plot_q2b_optimizers(runs: List[Dict[str, Any]], out_dir: str) -> None:
  """Generates Question 2(b) comparison across optimizers."""
  setup_wandb_style()
  opt_runs = [r for r in runs if r.get("group") == "optimizer_ablation"]
  if not opt_runs:
    return

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=220)

  for idx, r in enumerate(opt_runs):
    color = WANDB_PALETTE[idx % len(WANDB_PALETTE)]
    opt_name = r["config"]["optimizer"].upper()
    steps = [h["epoch"] for h in r["history"]]
    val_acc = [h["val_acc"] for h in r["history"]]
    train_loss = [h["train_loss"] for h in r["history"]]
    ax1.plot(
        steps,
        val_acc,
        label=f"{opt_name} (Best: {r['best_val_acc']:.2f}%)",
        color=color,
        linewidth=1.9,
    )
    ax2.plot(
        steps,
        train_loss,
        label=f"{opt_name} (Final: {r['final_train_loss']:.3f})",
        color=color,
        linewidth=1.9,
    )

  ax1.set_title("Validation Accuracy Convergence Across Optimizers", fontsize=11, fontweight="bold")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Validation Top-1 Accuracy (%)")
  ax1.grid(True)
  ax1.legend(fontsize=8.5, loc="lower right")

  ax2.set_title("Training Loss Convergence Across Optimizers", fontsize=11, fontweight="bold")
  ax2.set_xlabel("Epoch")
  ax2.set_ylabel("Training Cross-Entropy Loss")
  ax2.grid(True)
  ax2.legend(fontsize=8.5, loc="upper right")

  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, "q2b_optimizer_comparison.png"), bbox_inches="tight")
  plt.close(fig)


def plot_q2c_hyperparams(runs: List[Dict[str, Any]], out_dir: str) -> None:
  """Generates Question 2(c) ablation across batch size, learning rate, and epochs."""
  setup_wandb_style()
  bs_runs = sorted(
      [r for r in runs if r.get("group") == "batch_size_ablation"],
      key=lambda x: x["config"]["batch_size"],
  )
  lr_runs = sorted(
      [r for r in runs if r.get("group") == "lr_ablation"],
      key=lambda x: x["config"]["learning_rate"],
  )
  ep_runs = sorted(
      [r for r in runs if r.get("group") == "epoch_ablation"],
      key=lambda x: x["config"]["epochs"],
  )

  fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=220)

  if bs_runs:
    x_labels = [str(r["config"]["batch_size"]) for r in bs_runs]
    y_acc = [r["best_val_acc"] for r in bs_runs]
    y_time = [r["total_time_sec"] / max(r["config"]["epochs"], 1) for r in bs_runs]
    ax = axes[0]
    ax.plot(x_labels, y_acc, marker="o", linewidth=2.2, color="#5b8ff9", label="Best Val Acc (%)")
    for x, y in zip(x_labels, y_acc):
      ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_title("(i) Effect of Batch Size", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Best Validation Accuracy (%)", color="#5b8ff9")
    ax.set_ylim(min(y_acc) - 4, max(y_acc) + 4)
    ax.grid(True)
    ax_t = ax.twinx()
    ax_t.plot(x_labels, y_time, marker="s", linestyle="--", color="#e87461", label="Sec/Epoch")
    ax_t.set_ylabel("Avg Time / Epoch (s)", color="#e87461")

  if lr_runs:
    x_labels = [str(r["config"]["learning_rate"]) for r in lr_runs]
    y_acc = [r["best_val_acc"] for r in lr_runs]
    ax = axes[1]
    ax.plot(x_labels, y_acc, marker="o", linewidth=2.2, color="#269a99")
    for x, y in zip(x_labels, y_acc):
      ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_title("(ii) Effect of Learning Rate", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Learning Rate (lr)")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_ylim(min(y_acc) - 5, max(y_acc) + 5)
    ax.grid(True)

  if ep_runs:
    x_labels = [str(r["config"]["epochs"]) for r in ep_runs]
    y_acc = [r["best_val_acc"] for r in ep_runs]
    ax = axes[2]
    ax.plot(x_labels, y_acc, marker="D", linewidth=2.2, color="#9270ca")
    for x, y in zip(x_labels, y_acc):
      ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_title("(iii) Effect of Number of Epochs", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Total Training Epochs")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_ylim(min(y_acc) - 4, max(y_acc) + 4)
    ax.grid(True)

  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, "q2c_hyperparam_ablation.png"), bbox_inches="tight")
  plt.close(fig)


def plot_q3a_parallel_coordinates(runs: List[Dict[str, Any]], out_dir: str) -> None:
  """Generates W&B-style Parallel Coordinates plot (Question 3a, Figure 1)."""
  setup_wandb_style()

  act_cats = ["gelu", "l_relu", "relu", "selu", "silu", "tanh", "sigmoid"]
  bn_cats = ["false", "true"]
  opt_cats = ["adagrad", "adam", "nadam", "nesterov", "rmsprop", "sgd"]

  dims = [
      ("activation", "categorical", act_cats),
      ("batch_norm", "categorical", bn_cats),
      ("batch_size", "numeric", (0.0, 550.0)),
      ("epochs", "numeric", (4.0, 12.0)),
      ("learning_rate", "numeric", (0.0, 0.01)),
      ("momentum", "numeric", (0.0, 0.95)),
      ("optimizer", "categorical", opt_cats),
      ("weight_decay", "numeric", (0.0, 0.0005)),
      ("val_acc", "numeric", (10.0, 90.0)),
  ]

  fig, ax = plt.subplots(figsize=(13.5, 5.6), dpi=240)
  n_dims = len(dims)
  x_coords = np.arange(n_dims, dtype=float)

  cmap = plt.get_cmap("magma_r")
  norm = mcolors.Normalize(vmin=10.0, vmax=90.0)

  def normalize_val(dim_idx: int, run: Dict[str, Any]) -> float:
    name, dtype, spec = dims[dim_idx]
    cfg = run["config"]
    if name == "val_acc":
      val = float(run["best_val_acc"])
    elif name == "batch_norm":
      val = "true" if cfg.get("batch_norm", True) else "false"
    elif name == "activation":
      val = cfg.get("activation", "relu").lower()
      if val == "leaky_relu":
        val = "l_relu"
    elif name == "optimizer":
      val = cfg.get("optimizer", "adam").lower()
    else:
      val = float(cfg.get(name, 0.0))

    if dtype == "categorical":
      cats = spec
      idx = cats.index(val) if val in cats else 0
      return idx / max(len(cats) - 1, 1)
    else:
      vmin, vmax = spec
      return float(np.clip((val - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0))

  # Sort runs so highest accuracy lines are drawn on top
  sorted_runs = sorted(runs, key=lambda r: r["best_val_acc"])

  for run in sorted_runs:
    y_vals = [normalize_val(i, run) for i in range(n_dims)]
    color = cmap(norm(run["best_val_acc"]))
    verts = []
    codes = []
    verts.append((x_coords[0], y_vals[0]))
    codes.append(Path.MOVETO)
    for i in range(1, n_dims):
      x0, y0 = x_coords[i - 1], y_vals[i - 1]
      x1, y1 = x_coords[i], y_vals[i]
      xm = 0.5 * (x0 + x1)
      verts.extend([(xm, y0), (xm, y1), (x1, y1)])
      codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
    path = Path(verts, codes)
    patch = patches.PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        lw=2.0 if run["best_val_acc"] >= 85.0 else 1.4,
        alpha=0.88 if run["best_val_acc"] >= 80.0 else 0.65,
    )
    ax.add_patch(patch)

  # Draw vertical axes and ticks
  for i, (name, dtype, spec) in enumerate(dims):
    ax.axvline(i, color="#bcbcbc", linewidth=1.2, zorder=3)
    ax.text(
        i,
        1.05,
        name,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#444444",
    )
    if dtype == "categorical":
      cats = spec
      for j, cat_label in enumerate(cats):
        y_pos = j / max(len(cats) - 1, 1)
        ax.text(
            i - 0.06,
            y_pos,
            cat_label,
            ha="right",
            va="center",
            fontsize=8.2,
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.75),
        )
    else:
      vmin, vmax = spec
      ticks = np.linspace(vmin, vmax, 9)
      for t in ticks:
        y_pos = (t - vmin) / max(vmax - vmin, 1e-12)
        if name == "learning_rate":
          lbl = f"{t:.4f}"
        elif name == "weight_decay":
          lbl = f"{t:.5f}"
        elif name == "momentum":
          lbl = f"{t:.1f}"
        else:
          lbl = f"{int(round(t))}"
        ax.text(
            i - 0.05,
            y_pos,
            lbl,
            ha="right",
            va="center",
            fontsize=7.8,
            color="#666666",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.7),
        )

  ax.set_xlim(-0.5, n_dims - 0.6)
  ax.set_ylim(-0.04, 1.10)
  ax.axis("off")

  sm = cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  cbar = plt.colorbar(sm, ax=ax, fraction=0.022, pad=0.01)
  cbar.set_label("val_acc (%)", fontsize=9.5, fontweight="bold", color="#444444")

  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, "q3a_parallel_coordinates.png"), bbox_inches="tight")
  plt.close(fig)


def plot_q3b_val_acc_scatter(runs: List[Dict[str, Any]], out_dir: str) -> None:
  """Generates W&B-style val_acc v. created / step scatter plot (Question 3b, Figure 2)."""
  setup_wandb_style()
  fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=220)

  for idx, r in enumerate(runs):
    color = WANDB_PALETTE[idx % len(WANDB_PALETTE)]
    ax.scatter(
        idx + 1,
        r["best_val_acc"],
        s=75,
        color=color,
        edgecolors="white",
        linewidth=0.8,
        zorder=4,
        label=r["run_name"] if idx < 8 else None,
    )

  best_run = max(runs, key=lambda x: x["best_val_acc"])
  best_idx = runs.index(best_run) + 1
  ax.annotate(
      f"Best: {best_run['best_val_acc']:.2f}%\n({best_run['run_name']})",
      xy=(best_idx, best_run["best_val_acc"]),
      xytext=(max(1, best_idx - 5), best_run["best_val_acc"] - 12),
      arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2),
      fontsize=8.5,
      fontweight="bold",
      bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbe6", edgecolor="#d4b106"),
  )

  x_ticks = list(range(1, len(runs) + 1))
  x_labels = [f"Run {i}\n({r['created_timestamp'][-8:]})" for i, r in enumerate(runs, 1)]
  step_stride = max(1, len(runs) // 10)
  ax.set_xticks(x_ticks[::step_stride])
  ax.set_xticklabels(x_labels[::step_stride], rotation=35, ha="right", fontsize=8)

  ax.set_title("val_acc v. created (Validation Accuracy vs. Sweep Step)", fontsize=12, fontweight="bold")
  ax.set_ylabel("val_acc (%)")
  ax.set_xlabel("Sweep Step / Created Timestamp")
  ax.set_ylim(5, 96)
  ax.grid(True)

  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, "q3b_val_acc_vs_step_scatter.png"), bbox_inches="tight")
  plt.close(fig)


def plot_q3c_wandb_metric_curves(runs: List[Dict[str, Any]], out_dir: str) -> None:
  """Generates Question 3(c) W&B-style plots: train_acc, train_loss, val_acc, val_loss."""
  setup_wandb_style()
  # Select 10 representative runs across diverse groups
  selected_indices = [0, 1, 2, 4, 8, 9, 11, 16, 23, len(runs) - 1]
  selected_runs = [runs[i] for i in selected_indices if 0 <= i < len(runs)]
  if len(selected_runs) < min(10, len(runs)):
    selected_runs = runs[:10]

  metrics_info = [
      ("train_acc", "train_acc", "q3c_train_acc.png", "Accuracy (%)"),
      ("train_loss", "train_loss", "q3c_train_loss.png", "Cross-Entropy Loss"),
      ("val_acc", "val_acc", "q3c_val_acc.png", "Validation Accuracy (%)"),
      ("val_loss", "val_loss", "q3c_val_loss.png", "Validation Loss"),
  ]

  for key, title, filename, ylabel in metrics_info:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=220)
    for idx, r in enumerate(selected_runs):
      color = WANDB_PALETTE[idx % len(WANDB_PALETTE)]
      steps = [h["step"] for h in r["history"]]
      vals = [h[key] for h in r["history"]]
      ax.plot(steps, vals, label=r["run_name"], color=color, linewidth=1.6, alpha=0.9)

    fig.suptitle(
        f"{title}\nShowing first {len(selected_runs)} runs",
        fontsize=11,
        fontweight="bold",
        y=1.15,
    )
    ax.set_xlabel("Step (Epoch)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=7.0,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    results_json_path: str = "results/all_experiments.json",
    out_dir: str = "plots",
) -> None:
  """Loads experiment JSON results and generates all figures."""
  os.makedirs(out_dir, exist_ok=True)
  with open(results_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  runs = data["runs"]
  best_run = max(runs, key=lambda r: r["best_val_acc"])

  plot_q1_baseline_curves(best_run, out_dir)
  plot_q2a_activations(runs, out_dir)
  plot_q2b_optimizers(runs, out_dir)
  plot_q2c_hyperparams(runs, out_dir)
  plot_q3a_parallel_coordinates(runs, out_dir)
  plot_q3b_val_acc_scatter(runs, out_dir)
  plot_q3c_wandb_metric_curves(runs, out_dir)
  print(f"[PLOTS GENERATED] Saved all figures to {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
  generate_all_plots()
