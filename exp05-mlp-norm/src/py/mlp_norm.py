"""
MLP with an optional normalisation layer between Linear and activation — exp05's
subject of study.

Block layout (repeated for every hidden layer; the output Linear is bare):

    Linear -> Norm -> Activation -> Dropout

Why the norm goes *before* the activation: exp03/exp04 showed that `Quadratic`
(x^2) — the only CKKS-cheap activation available — blows the activation scale up
multiplicatively with depth, which is what limits how deep an exportable MLP can
get. Normalising the pre-activations is the direct lever on that.

CKKS relevance of each norm mode (this is the point of the experiment):

  "none"   baseline, identical to exp03's MLP.
  "batch"  `nn.BatchNorm1d`. In *eval* mode it is a fixed per-feature affine map
           y = gamma*(x - mu)/sqrt(var + eps) + beta, so it costs nothing under
           CKKS: it folds into the preceding Linear's (W, b) at export time.
  "layer"  `nn.LayerNorm`. Data-dependent — needs a per-sample mean, variance and
           inverse square root of the ciphertext. NOT foldable and not CKKS-cheap;
           kept as a plaintext upper bound to see what "batch" gives up.

`is_ckks_compatible()` encodes that rule, mirroring exp04's `train_cnn.py`.
"""

from typing import Callable

import torch
import torch.nn as nn


class Quadratic(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


ACTIVATION_MAP: dict[str, Callable[[], nn.Module]] = {
    "GELU":      nn.GELU,
    "ReLU":      nn.ReLU,
    "Sigmoid":   nn.Sigmoid,
    "Quadratic": Quadratic,
    "Tanh":      nn.Tanh,
}

NORM_MODES = ("none", "batch", "layer")

# norm modes that stay linear at inference time -> foldable into the previous Linear
CKKS_NORM_MODES = ("none", "batch")


def is_ckks_compatible(activation: str, norm: str) -> bool:
    """Only x^2 activations and inference-time-linear norms can be exported to CKKS."""
    return activation == "Quadratic" and norm in CKKS_NORM_MODES


def make_norm(mode: str, dim: int) -> list[nn.Module]:
    if mode == "none":
        return []
    if mode == "batch":
        return [nn.BatchNorm1d(dim)]
    if mode == "layer":
        return [nn.LayerNorm(dim)]
    raise ValueError(f"unknown norm mode {mode!r}, expected one of {NORM_MODES}")


class MlpNorm(nn.Module):
    def __init__(
        self,
        layers:     list[int],
        activation: Callable[[], nn.Module],
        norm:       str = "none",
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()
        self.layers, self.norm = layers, norm
        mods: list[nn.Module] = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:                    # hidden layer -> norm/act/dropout
                mods += make_norm(norm, layers[i + 1])
                mods.append(activation())
                if dropout > 0.0:
                    mods.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
