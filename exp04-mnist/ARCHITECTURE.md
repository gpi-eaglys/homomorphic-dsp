# exp04 CNN topology

* defined in `exp04-mnist/src/py/train_cnn.py` 
   * hyperparameters are swept by `exp04-mnist/src/py/grid_search_cnn.py`.
* input: 28x28 grayscale MNIST image (784 raw pixels)
* normalization: per-pixel mean/std computed over the training set


## Grid search space (`grid_search_cnn.py`)

| Axis             | Values swept                                     | Fixed? |
|------------------|--------------------------------------------------|--------|
| `conv_channels`  | `[1,16,32]`, `[1,32,64]`, `[1,16,32,64]`         | swept  |
| `kernel_size`    | `3`                                              | fixed  |
| `fc_layers`      | `[128, 10]`                                      | fixed  |
| `activation`     | `GELU`, `ReLU`, `Sigmoid`, `Quadratic` (`x*x`)   | swept  |
| `pool`           | `max`, `avg`                                     | swept  |
| `lr`             | `1e-3`, `5e-3`                                   | swept  |
| `dropout`        | `0.0`, `0.2`, `0.4`                              | swept  |

144 total combinations (3 x 4 x 2 x 2 x 3).

### CKKS compatibility 
* only `activation=Quadratic` (polynomial, CKKS-evaluable) and `pool=avg` (linear, CKKS-evaluable) combinations are exportable to CKKS
* cf. [export_mdl.py](src/py/export_mdl.py) and the main [README](README.md)'s CKKS workflow

## Per-stage CNN topology

* Each conv stage consists of
  * `Conv2d(k=kernel_size, padding=kernel_size//2)` ->
  * `activation()`
  * `pool(2)` (kernel 2, stride 2)
  *  optional `Dropout`
* a conv stage never changes the spatial size on its own — only `pool` does
* after the CNN stack, the tensor is flattened and fed through the FC layer



## Network topologies per convolution channels 

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

