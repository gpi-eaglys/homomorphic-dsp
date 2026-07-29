# exp04 CNN topology

Defined in `src/py/train_cnn.py` (`CNN` class), swept by `src/py/grid_search_cnn.py`.
Input is always a single 28x28 grayscale MNIST image (784 raw pixels, normalized by
per-pixel mean/std computed from the training set).

## Per-stage structure

Each conv stage is: `Conv2d(k=kernel_size, padding=kernel_size//2)` -> `activation()` ->
`pool(2)` (kernel 2, stride 2) -> optional `Dropout`. `padding=kernel_size//2` is SAME
padding, so a conv stage never changes the spatial size on its own — only `pool` does,
by floor-dividing each spatial dimension by 2.

After the conv stack, the tensor is flattened (channel-major, then row-major spatial —
matches `nn.Linear`'s implicit column order) and fed through the FC stack: `Linear` ->
`activation()` (except after the last `Linear`, which has none — that's the logits
layer).

## Grid search space (`grid_search_cnn.py`)

| Axis            | Values swept                                    | Fixed? |
|------------------|--------------------------------------------------|--------|
| `conv_channels`  | `[1,16,32]`, `[1,32,64]`, `[1,16,32,64]`         | swept  |
| `kernel_size`    | `3`                                              | fixed  |
| `fc_layers`      | `[128, 10]`                                      | fixed  |
| `activation`     | `GELU`, `ReLU`, `Sigmoid`, `Quadratic` (`x*x`)   | swept  |
| `pool`           | `max`, `avg`                                     | swept  |
| `lr`             | `1e-3`, `5e-3`                                   | swept  |
| `dropout`        | `0.0`, `0.2`, `0.4`                              | swept  |

144 total combinations (3 x 4 x 2 x 2 x 3).

Only `activation=Quadratic` (polynomial, CKKS-evaluable) and `pool=avg` (linear,
CKKS-evaluable) combinations are exportable to CKKS — see
[export_mdl.py](src/py/export_mdl.py) and the main [README](README.md)'s CKKS workflow
section. `MaxPool2d` needs comparisons, and `GELU`/`ReLU`/`Sigmoid` aren't polynomial, so
neither is homomorphically evaluable as-is.

## Worked shapes, per `conv_channels` option (all with `kernel_size=3`, `fc_layers=[128,10]`)

**`[1, 16, 32]`** (2 conv stages):
```
input        1x28x28
conv1 (1->16)  16x28x28   -> pool ->  16x14x14
conv2 (16->32) 32x14x14   -> pool ->  32x7x7
flatten        1568                              (32*7*7)
fc1            1568 -> 128
fc2 (logits)   128  -> 10
```

**`[1, 32, 64]`** (2 conv stages, wider):
```
input        1x28x28
conv1 (1->32)  32x28x28   -> pool ->  32x14x14
conv2 (32->64) 64x14x14   -> pool ->  64x7x7
flatten        3136                              (64*7*7)
fc1            3136 -> 128
fc2 (logits)   128  -> 10
```

**`[1, 16, 32, 64]`** (3 conv stages, deeper):
```
input        1x28x28
conv1 (1->16)   16x28x28  -> pool ->  16x14x14
conv2 (16->32)  32x14x14  -> pool ->  32x7x7
conv3 (32->64)  64x7x7    -> pool ->  64x3x3     (7 is odd -> floor(7/2)=3, last row/col dropped)
flatten          576                             (64*3*3)
fc1              576 -> 128
fc2 (logits)     128 -> 10
```

The `[1,16,32,64]` topology's odd 7x7 -> 3x3 pool (dropping a row/col) is exactly the
case `export_mdl.py`'s stride/floor-division bookkeeping was written to handle robustly
(see its module docstring) — it isn't a special case in the export logic, just a smaller
`h_active`/`w_active` than the even-dimension topologies.
