# CS6886W — Systems Engineering for Deep Learning (Assignment 1)
## Exploring VGG6 on CIFAR-10 Across Activations, Optimizers, and Hyperparameters

- **Student Name:** Kiran T K
- **Roll Number:** `cs25m513`
- **Course:** CS6886W — Systems Engineering for Deep Learning, IIT Madras
- **GitHub Repository:** [https://github.com/cs25m513/CS6886W-Assignment1-VGG6](https://github.com/cs25m513/CS6886W-Assignment1-VGG6)

---

## 1. Repository Structure (Modular Architecture — Question 5a)

Training, evaluation, dataset preparation, model definition, and visualization are cleanly separated into modular Python packages:

```text
CS6886W_Assignment1/
├── README.md                              # Exact reproduction instructions & experimental analysis
├── requirements.txt                       # Exact pinned library dependency versions
├── sweep_config.yaml                      # Weights & Biases (W&B) hyperparameter sweep definition
├── run_sweep.py                           # W&B sweep agent entry point
├── run_experiments.py                     # Automated runner for baseline + all Q2/Q3 ablations
├── generate_plots.py                      # Generates all W&B parallel-coordinate, scatter, and curve plots
├── checkpoints/
│   └── best_vgg6_cifar10.pth              # Trained VGG6 model checkpoint for the best configuration
├── results/
│   └── all_experiments.json               # Full epoch-by-epoch training/validation metrics
├── plots/                                 # Generated W&B and ablation figures
│   ├── q1_baseline_curves.png
│   ├── q2a_activation_comparison.png
│   ├── q2b_optimizer_comparison.png
│   ├── q2c_hyperparam_ablation.png
│   ├── q3a_parallel_coordinates.png
│   ├── q3b_val_acc_vs_step_scatter.png
│   ├── q3c_train_acc.png
│   ├── q3c_train_loss.png
│   ├── q3c_val_acc.png
│   └── q3c_val_loss.png
└── src/
    ├── __init__.py
    ├── dataset.py                         # CIFAR-10 transforms, Cutout(1,16), AutoAugment, DataLoaders
    ├── model.py                           # VGG6 architecture with configurable activations & BatchNorm
    ├── train.py                           # Modular training loop, optimizer builder, CLI entry point
    ├── evaluate.py                        # Standalone & library evaluation module on CIFAR-10 test set
    └── utils.py                           # Deterministic seeding (seed=42), checkpoint & JSON utilities
```

---

## 2. Environment Details, Dependency Versions & Seed Configuration (Question 5b)

### Hardware & Operating System
- **OS:** Linux (x86_64, Debian/Ubuntu-based environment)
- **Processor:** 24-core AMD EPYC 7B12 (supports CUDA GPU automatically if available via `torch.cuda.is_available()`)
- **Python Version:** `Python 3.13` (compatible with Python `3.10+`)

### Exact Dependency Versions (`requirements.txt`)
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `torch` | `2.12.1` | Deep learning framework, VGG6 layers, autograd, optimizers |
| `torchvision` | `0.27.1` | CIFAR-10 dataset & image transformations |
| `numpy` | `2.5.4` | Array operations, Cutout mask generation, deterministic RNG |
| `pillow` | `13.0.0` | Image processing & AutoAugment operations |
| `matplotlib` | `3.11.2` | Loss/accuracy curves & W&B visualization plots |
| `wandb` | `0.31.0` | Weights & Biases experiment tracking and sweeps |
| `pyyaml` | `6.0.3` | Sweep YAML configuration parsing |

### Deterministic Seed Configuration (`src/utils.py`)
All experiments enforce a strict deterministic seed (`seed = 42`) across every random number generator:
```python
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Additionally, `torch.utils.data.DataLoader` uses an explicit seeded `torch.Generator().manual_seed(42)` and `worker_init_fn` so batch ordering and data subset selection are 100% reproducible across runs.

---

## 3. Exact Commands to Reproduce Results

### Step 1: Setup Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Verify the Saved Trained Model Checkpoint (`checkpoints/best_vgg6_cifar10.pth`)
To evaluate the pre-trained best VGG6 model checkpoint directly on the 10,000-image CIFAR-10 test set:
```bash
python -m src.evaluate --checkpoint checkpoints/best_vgg6_cifar10.pth --seed 42
```
- **Reproduced Test Top-1 Accuracy:** **`67.02%`** (Test Loss: `0.9378`)

### Step 3: Re-Train the Best Configuration from Scratch (Question 1 & Question 4)
```bash
python -m src.train \
  --activation gelu \
  --optimizer nadam \
  --batch-size 64 \
  --epochs 8 \
  --lr 0.001 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --batch-norm true \
  --use-cutout true
  --subset-size 15000 \
  --seed 42 \
  --save-checkpoint checkpoints/best_vgg6_cifar10.pth
```
*(Tip: Pass `--subset-size 0` on a GPU machine to train on all 50,000 images for 50–100 epochs.)*

### Step 4: Reproduce the Full Ablation Suite & Regenerate All Plots (Questions 2 & 3)
```bash
python run_experiments.py
python generate_plots.py
```

### Step 5: Run a Live Weights & Biases (W&B) Sweep
```bash
wandb sweep sweep_config.yaml
wandb agent <SWEEP_ID>
```

---

## 4. Explanation of How Performance Varies Across Configurations

### 4.1 Baseline Data Augmentation & Normalization (Question 1)
- **Training Transforms:**
  1. `transforms.RandomCrop(32, padding=4)`: Pads each $32 \times 32$ image by 4 pixels on each side ($40 \times 40$) and randomly crops back to $32 \times 32$, enforcing translation invariance.
  2. `transforms.RandomHorizontalFlip(p=0.5)`: Horizontally mirrors images with $50\%$ probability.
  3. `transforms.ToTensor()`: Scales `[0, 255]` uint8 pixels to `[0.0, 1.0]` float32 tensors.
  4. `transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))`: Standardizes each RGB channel to zero mean and unit variance across CIFAR-10.
  5. `Cutout(n_holes=1, length=16)`: Randomly masks one $16 \times 16$ square patch to zero, preventing co-adaptation on single local features.
- **Test/Validation Transforms:** Deterministic `ToTensor()` followed by `Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))`.
- **Best Baseline Performance:** **`67.02%`** Top-1 Test Accuracy (`val_loss = 0.9378`).

### 4.2 Varying the Activation Function (Question 2a)
| Activation | Final Train Loss | Final Train Acc | Best Val Loss | Best Val Top-1 Acc | Best Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `GELU` | `1.3313` | `53.00%` | `1.3156` | **`52.53%`** | `8` |
| `RELU` | `1.3531` | `52.23%` | `1.3462` | **`51.60%`** | `8` |
| `SILU` | `1.3744` | `51.37%` | `1.3370` | **`50.93%`** | `8` |
| `LEAKY_RELU` | `1.3854` | `50.60%` | `1.3587` | **`50.93%`** | `8` |
| `SELU` | `1.5367` | `44.77%` | `1.4790` | **`46.93%`** | `8` |
| `TANH` | `1.5962` | `43.60%` | `1.5127` | **`45.00%`** | `8` |
| `SIGMOID` | `1.7668` | `39.17%` | `1.6594` | **`41.93%`** | `8` |

- **Smooth Non-Monotonic Gated Activations (`GELU`, `SiLU`) & Non-Saturating Rectifiers (`LeakyReLU`, `ReLU`):** Achieve the highest validation accuracies because their derivatives remain near $1$ for positive activations without hard zero-gradient dead zones (`GELU` and `SiLU` provide smooth probabilistic gating $x \cdot \Phi(x)$ and $x \cdot \sigma(x)$, while `LeakyReLU` preserves a $0.1$ negative slope).
- **Zero-Centered Saturating (`Tanh`) & Self-Normalizing (`SELU`):** Perform moderately well when paired with BatchNorm, though `Tanh` saturates for $|x| > 2$, attenuating gradients in deeper conv blocks.
- **Non-Zero-Centered Saturating (`Sigmoid`):** Performs worst because $\sigma(x) \in (0, 1)$ has a maximum derivative of $0.25$ at $x=0$ and saturates at both tails, causing severe gradient attenuation across the 4 convolutional layers.

### 4.3 Varying the Optimizer (Question 2b)
| Optimizer | Learning Rate | Momentum | Final Train Loss | Final Train Acc | Best Val Top-1 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `NADAM` | `0.001` | `0.9` | `1.3313` | `53.00%` | **`52.53%`** |
| `RMSPROP` | `0.001` | `0.0` | `1.3670` | `51.70%` | **`51.73%`** |
| `ADAM` | `0.001` | `0.9` | `1.3684` | `51.93%` | **`50.80%`** |
| `ADAGRAD` | `0.01` | `0.0` | `1.3928` | `50.63%` | **`49.73%`** |
| `SGD` | `0.01` | `0.9` | `1.4929` | `47.03%` | **`46.93%`** |
| `NESTEROV` | `0.01` | `0.9` | `1.4775` | `47.47%` | **`46.73%`** |

- **`Nadam` & `Adam`:** Combine first-moment momentum with per-parameter second-moment ($v_t$) adaptive scaling. `Nadam` incorporates Nesterov lookahead directly into the bias-corrected first moment update, achieving the fastest initial loss reduction and highest validation accuracy.
- **`Nesterov-SGD` vs. `SGD`:** Standard `SGD` with momentum ($\mu=0.9$) accumulates velocity along consistent gradient directions; `Nesterov-SGD` evaluates the gradient at the lookahead position $\theta_t - \mu v_t$, providing a corrective brake before overshooting sharp curvature valleys.
- **`RMSprop`:** Maintains an exponential moving average of squared gradients without a momentum bias-correction term; it adapts step sizes effectively but exhibits slightly higher validation oscillation unless paired with a lower learning rate.
- **`Adagrad`:** Accumulates squared gradients monotonically from step $0$ ($G_t = G_{t-1} + g_t^2$) without exponential decay, causing the effective learning rate $\eta / \sqrt{G_t + \epsilon}$ to shrink prematurely and stall before reaching the optimum.

### 4.4 Varying Batch Size, Learning Rate, and Epochs (Question 2c)

#### (i) Batch Size Ablation (`32, 64, 128, 256, 512`)
| Batch Size | Final Train Loss | Final Train Acc | Best Val Top-1 Acc | Avg Time / Epoch |
| :---: | :---: | :---: | :---: | :---: |
| `32` | `1.3202` | `53.83%` | **`53.67%`** | `24.0s` |
| `64` | `1.3313` | `53.00%` | **`52.53%`** | `23.5s` |
| `128` | `1.3879` | `51.23%` | **`50.20%`** | `23.1s` |
| `256` | `1.4635` | `49.10%` | **`48.00%`** | `22.2s` |
| `512` | `1.5744` | `45.57%` | **`42.40%`** | `21.7s` |

- Smaller batch sizes (`32`, `64`) execute more gradient updates per epoch ($47$–$94\times$ per 3,000 samples) and inject beneficial stochastic gradient noise that acts as an implicit regularizer, escaping sharp minima and yielding higher generalization accuracy. Large batch sizes (`256`, `512`) reduce per-epoch compute time via vectorization but perform 4–8$\times$ fewer parameter updates per epoch, underfitting when epoch count and learning rate are held fixed.

#### (ii) Learning Rate Ablation (`0.0001, 0.0005, 0.001, 0.005, 0.01`)
| Learning Rate | Final Train Loss | Final Train Acc | Best Val Top-1 Acc |
| :---: | :---: | :---: | :---: |
| `0.0001` | `1.7168` | `40.80%` | **`38.53%`** |
| `0.0005` | `1.4155` | `50.53%` | **`49.60%`** |
| `0.001` | `1.3313` | `53.00%` | **`52.53%`** |
| `0.005` | `1.2945` | `54.50%` | **`54.47%`** |
| `0.01` | `1.3553` | `51.53%` | **`52.00%`** |

- An intermediate learning rate (`lr = 0.001` to `0.005` for adaptive optimizers) balances rapid descent with stable convergence. A very small learning rate (`0.0001`) makes overly conservative steps and underfits within the epoch budget, whereas an excessively large learning rate (`0.01` with `Nadam`) causes large parameter oscillations across mini-batches.

#### (iii) Number of Epochs Ablation (`4, 6, 8, 12`)
| Epochs | Final Train Loss | Final Train Acc | Best Val Top-1 Acc | Total Wall Time |
| :---: | :---: | :---: | :---: | :---: |
| `4` | `1.4939` | `46.40%` | **`47.13%`** | `94.5s` |
| `6` | `1.4006` | `50.87%` | **`49.60%`** | `137.9s` |
| `8` | `1.3313` | `53.00%` | **`52.53%`** | `166.3s` |
| `12` | `1.2235` | `57.63%` | **`55.67%`** | `211.0s` |

- Increasing the training budget from `4` to `12` epochs steadily lowers training cross-entropy loss and improves validation accuracy as the cosine annealing schedule refines feature representations in the later epochs, with diminishing returns once validation loss begins to plateau.
