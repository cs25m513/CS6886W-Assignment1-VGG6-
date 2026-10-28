"""CIFAR-10 dataset preparation, normalization, and data augmentation module.

Adapted from the provided CS6886W Assignment 1 template code, supporting:
- RandomCrop(32, padding=4)
- RandomHorizontalFlip(p=0.5)
- Optional AutoAugment (CIFAR10Policy)
- Channel-wise CIFAR-10 Normalization:
    mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
- Cutout(n_holes=1, length=16) regularization
"""

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils import seed_worker

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class Cutout(object):
  """Randomly mask out one or more square patches from an image tensor.

  Args:
      n_holes (int): Number of patches to cut out of each image.
      length (int): The side length (in pixels) of each square patch.
  """

  def __init__(self, n_holes: int = 1, length: int = 16):
    self.n_holes = n_holes
    self.length = length

  def __call__(self, img: torch.Tensor) -> torch.Tensor:
    """Applies Cutout mask to a normalized tensor image of shape (C, H, W)."""
    h = img.size(1)
    w = img.size(2)

    mask = np.ones((h, w), np.float32)

    for _ in range(self.n_holes):
      y = np.random.randint(h)
      x = np.random.randint(w)

      y1 = np.clip(y - self.length // 2, 0, h)
      y2 = np.clip(y + self.length // 2, 0, h)
      x1 = np.clip(x - self.length // 2, 0, w)
      x2 = np.clip(x + self.length // 2, 0, w)

      mask[y1:y2, x1:x2] = 0.0

    mask_tensor = torch.from_numpy(mask).expand_as(img)
    return img * mask_tensor


class SubPolicy(object):
  """Sub-policy for CIFAR-10 AutoAugment."""

  def __init__(
      self,
      p1: float,
      operation1: str,
      magnitude_idx1: int,
      p2: float,
      operation2: str,
      magnitude_idx2: int,
      fillcolor: Tuple[int, int, int] = (128, 128, 128),
  ):
    self.p1 = p1
    self.op1 = operation1
    self.magnitude_idx1 = magnitude_idx1
    self.p2 = p2
    self.op2 = operation2
    self.magnitude_idx2 = magnitude_idx2
    self.fillcolor = fillcolor
    self.init = 0

  def gen(self, operation1, magnitude_idx1, operation2, magnitude_idx2, fillcolor):
    ranges = {
        "shearX": np.linspace(0, 0.3, 10),
        "shearY": np.linspace(0, 0.3, 10),
        "translateX": np.linspace(0, 150 / 331, 10),
        "translateY": np.linspace(0, 150 / 331, 10),
        "rotate": np.linspace(0, 30, 10),
        "color": np.linspace(0.0, 0.9, 10),
        "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
        "solarize": np.linspace(256, 0, 10),
        "contrast": np.linspace(0.0, 0.9, 10),
        "sharpness": np.linspace(0.0, 0.9, 10),
        "brightness": np.linspace(0.0, 0.9, 10),
        "autocontrast": [0] * 10,
        "equalize": [0] * 10,
        "invert": [0] * 10,
    }

    def rotate_with_fill(img, magnitude):
      rot = img.convert("RGBA").rotate(magnitude)
      return Image.composite(
          rot, Image.new("RGBA", rot.size, (128,) * 4), rot
      ).convert(img.mode)

    func = {
        "shearX": lambda img, magnitude: img.transform(
            img.size,
            Image.AFFINE,
            (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC,
            fillcolor=fillcolor,
        ),
        "shearY": lambda img, magnitude: img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC,
            fillcolor=fillcolor,
        ),
        "translateX": lambda img, magnitude: img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=fillcolor,
        ),
        "translateY": lambda img, magnitude: img.transform(
            img.size,
            Image.AFFINE,
            (0, 1, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
            fillcolor=fillcolor,
        ),
        "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
        "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
            1 + magnitude * random.choice([-1, 1])
        ),
        "posterize": lambda img, magnitude: ImageOps.posterize(img, int(magnitude)),
        "solarize": lambda img, magnitude: ImageOps.solarize(img, int(magnitude)),
        "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
            1 + magnitude * random.choice([-1, 1])
        ),
        "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
            1 + magnitude * random.choice([-1, 1])
        ),
        "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
            1 + magnitude * random.choice([-1, 1])
        ),
        "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
        "equalize": lambda img, magnitude: ImageOps.equalize(img),
        "invert": lambda img, magnitude: ImageOps.invert(img),
    }

    self.operation1 = func[operation1]
    self.magnitude1 = ranges[operation1][magnitude_idx1]
    self.operation2 = func[operation2]
    self.magnitude2 = ranges[operation2][magnitude_idx2]

  def __call__(self, img):
    if self.init == 0:
      self.gen(
          self.op1,
          self.magnitude_idx1,
          self.op2,
          self.magnitude_idx2,
          self.fillcolor,
      )
      self.init = 1
    if random.random() < self.p1:
      img = self.operation1(img, self.magnitude1)
    if random.random() < self.p2:
      img = self.operation2(img, self.magnitude2)
    return img


class CIFAR10Policy(object):
  """AutoAugment Policy learned on CIFAR-10 (Cubuk et al., CVPR 2019)."""

  def __init__(self, fillcolor: Tuple[int, int, int] = (128, 128, 128)):
    self.policies = [
        SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
        SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
        SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
        SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
        SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
        SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
        SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
        SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
        SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
        SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
        SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
        SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
        SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
        SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
        SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
        SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
        SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
        SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
        SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
        SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
        SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
        SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
        SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
        SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
        SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
    ]

  def __call__(self, img):
    policy_idx = random.randint(0, len(self.policies) - 1)
    return self.policies[policy_idx](img)

  def __repr__(self) -> str:
    return "AutoAugment CIFAR10 Policy"


def get_cifar10_dataloaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 0,
    use_cutout: bool = True,
    use_autoaugment: bool = False,
    seed: int = 42,
    subset_size: int = 0,
    val_subset_size: int = 0,
) -> Tuple[DataLoader, DataLoader]:
  """Creates training and test/validation DataLoaders for CIFAR-10.

  Args:
      batch_size: Mini-batch size for training and evaluation.
      data_dir: Directory where CIFAR-10 dataset is stored/downloaded.
      num_workers: Number of subprocesses for data loading.
      use_cutout: Whether to apply Cutout(n_holes=1, length=16) augmentation.
      use_autoaugment: Whether to apply CIFAR10Policy() AutoAugment.
      seed: Random seed for deterministic DataLoader shuffling.
      subset_size: If > 0, uses a deterministic subset of training samples for
        fast sweep exploration (0 uses all 50,000 training samples).
      val_subset_size: If > 0, uses a deterministic subset of validation/test
        samples for fast sweep exploration (0 uses all 10,000 test samples).

  Returns:
      Tuple of (train_dataloader, test_dataloader).
  """
  train_transform_list = [
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
  ]
  if use_autoaugment:
    train_transform_list.append(CIFAR10Policy())
  train_transform_list.extend([
      transforms.ToTensor(),
      transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
  ])
  if use_cutout:
    train_transform_list.append(Cutout(n_holes=1, length=16))

  trans_train = transforms.Compose(train_transform_list)
  trans_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
  ])

  train_data = datasets.CIFAR10(
      root=data_dir, train=True, transform=trans_train, download=True
  )
  test_data = datasets.CIFAR10(
      root=data_dir, train=False, transform=trans_test, download=True
  )

  if subset_size and 0 < subset_size < len(train_data):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_data), size=subset_size, replace=False)
    train_data = torch.utils.data.Subset(train_data, indices.tolist())

  if val_subset_size and 0 < val_subset_size < len(test_data):
    rng_val = np.random.default_rng(seed)
    val_indices = rng_val.choice(
        len(test_data), size=val_subset_size, replace=False
    )
    test_data = torch.utils.data.Subset(test_data, val_indices.tolist())

  generator = torch.Generator()
  generator.manual_seed(seed)

  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      worker_init_fn=seed_worker if num_workers > 0 else None,
      generator=generator,
      pin_memory=torch.cuda.is_available(),
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=max(batch_size, 256),
      shuffle=False,
      num_workers=num_workers,
      pin_memory=torch.cuda.is_available(),
  )
  return train_dataloader, test_dataloader


# Alias matching the template notebook function name
GetCifar10 = get_cifar10_dataloaders
