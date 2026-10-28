"""VGG6 model architecture on CIFAR-10 with configurable activations and BatchNorm.

Adapted from the provided CS6886W Assignment 1 template code:
- Feature extractor configuration: cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']
- Global pooling: F.adaptive_avg_pool2d(x, (1, 1))
- Classifier head: nn.Linear(128, num_classes)
- Supported activations: relu, sigmoid, tanh, silu, gelu, leaky_relu, selu
"""

import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

CFG_VGG6: List[Union[int, str]] = [64, 64, "M", 128, 128, "M"]


def get_activation(name: str) -> nn.Module:
  """Returns the requested PyTorch activation module.

  Args:
      name: Activation name ('relu', 'sigmoid', 'tanh', 'silu', 'gelu',
        'leaky_relu', 'selu').

  Returns:
      Instantiated torch.nn.Module activation function.
  """
  act = name.lower().strip()
  if act == "relu":
    return nn.ReLU(inplace=True)
  elif act == "sigmoid":
    return nn.Sigmoid()
  elif act == "tanh":
    return nn.Tanh()
  elif act == "silu":
    return nn.SiLU(inplace=True)
  elif act == "gelu":
    return nn.GELU()
  elif act in ("leaky_relu", "l_relu", "leakyrelu"):
    return nn.LeakyReLU(negative_slope=0.1, inplace=True)
  elif act == "selu":
    return nn.SELU(inplace=True)
  else:
    raise ValueError(
        f"Unsupported activation '{name}'. Choose from: relu, sigmoid, tanh,"
        " silu, gelu, leaky_relu, selu."
    )


class VGG(nn.Module):
  """VGG-style Convolutional Neural Network for CIFAR-10 classification."""

  def __init__(self, features: nn.Module, num_classes: int = 10):
    super(VGG, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
        nn.Linear(128, num_classes),
    )
    self._initialize_weights()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

  def _initialize_weights(self) -> None:
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def make_layers(
    cfg: List[Union[int, str]],
    batch_norm: bool = True,
    activation: str = "relu",
) -> nn.Sequential:
  """Constructs VGG convolutional feature layers with configurable activation."""
  layers: List[nn.Module] = []
  in_channels = 3
  for v in cfg:
    if v == "M":
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      out_channels = int(v)
      conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
      act_layer = get_activation(activation)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(out_channels), act_layer]
      else:
        layers += [conv2d, act_layer]
      in_channels = out_channels
  return nn.Sequential(*layers)


def vgg(
    cfg: List[Union[int, str]] = None,
    num_classes: int = 10,
    batch_norm: bool = True,
    activation: str = "relu",
) -> VGG:
  """Factory function to build VGG6 with the specified configuration."""
  if cfg is None:
    cfg = CFG_VGG6
  return VGG(
      make_layers(cfg, batch_norm=batch_norm, activation=activation),
      num_classes=num_classes,
  )
