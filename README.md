# LiFT classification

A trainable PyTorch image classifier that preserves the LiFT recurrent architecture:

```text
RGB patches → Prelude (once) → shared Core (K loops) → conditional token RMSNorm → GAP → Linear
```

Each loop computes `h = core(RMSNorm(h) + h0, c_base + depth_embedding(s))`.
The core has one set of weights, regardless of K. Labels are used only by the loss.
This is a classification adaptation, with final cross-entropy instead of LiFT's
generative trajectory objective. Increasing K is supported, but improved accuracy
at larger K is an open experiment.

## Install and run

Python 3.10+ and PyTorch 2.2+ are required. PyTorch is the only runtime dependency.

```sh
python -m pip install -e '.[test]'
python -m pytest -q
```

```python
import torch
from lift_classify import LiFTClassifier, tiny_config

model = LiFTClassifier(tiny_config(num_classes=10))
images = torch.randn(2, 3, 224, 224)
labels = torch.tensor([2, 7])

logits = model(images)                         # [2, 10], default K=2
loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()

model.eval()
with torch.inference_mode():
    logits = model(images, num_loops=8)         # same parameters, more computation
```

Inputs are floating point `[B,in_channels,H,W]`. Choose image preprocessing and
normalization for the classification dataset; the model performs neither resizing
nor pixel-value normalization. Actual H/W may differ from `image_size`, but must
be positive multiples of `patch_size`. Rectangular images and patches are supported.

## Intermediate supervision

```python
output = model(images, num_loops=4,
               return_intermediates=True,
               return_intermediate_logits=True)
final_logits = output["logits"]
states = output["hidden_states"]              # tuple (h0, ..., h4), each [B,N,D]
logits_by_depth = output["intermediate_logits"]  # tuple (z0, ..., z4)
depths = output["depth_coordinates"]           # [B,4], prelude coordinate 0 omitted

# An example research loss, not the default training objective:
auxiliary_ce = sum(torch.nn.functional.cross_entropy(z, labels)
                   for z in logits_by_depth) / len(logits_by_depth)
```

With neither return flag, forward returns only a logits tensor. With either flag,
it returns a dictionary with the four keys above; an unrequested intermediate
field is `None`. States retain autograd connectivity. The coda reads states without
changing them. State-only requests compute the head once; logits requests compute
one shared head per state and reuse the final result. Logits-only requests do not
retain an additional tuple of hidden states.

`model.classify_features(state)` applies that same coda to a selected returned
state, allowing the caller to supervise only chosen depths. Keeping all states
retains their autograd graphs and increases memory use.

Default training and inference both use `s_k = k/K`. An explicit floating-point
`depth_coordinates` tensor can have shape `[K]` or `[B,K]`; values must be finite,
strictly increasing in `(0,1]`, and end at exactly 1. K must match `num_loops` (or
the config default). Coordinates do not determine K implicitly.

```python
from lift_classify import sample_depth_coordinates

depths = sample_depth_coordinates(images.shape[0], 4, device=images.device,
                                  generator=torch.Generator().manual_seed(42))
logits = model(images, num_loops=4, depth_coordinates=depths)
```

The random helper implements the paper's sorted interior sampling and spacing
correction on CPU in FP64, then transfers FP32 coordinates to the requested device.
Its optional generator is a CPU generator. Random coordinates are opt-in; using
them with final CE alone does not reproduce LiFT trajectory supervision.

## Model sizes

`LiFTConfig` accepts `image_size`, `in_channels`, `patch_size`, `num_classes`,
`hidden_dim`, `num_heads`, `mlp_ratio`, `prelude_depth`, `core_depth`, `num_loops`,
`dropout`, and `attention_dropout`. Image/patch sizes accept an integer or `(H,W)`.
Head dimension must be divisible by four for axial RoPE. There are no transformer
coda blocks or CLS tokens. Dropout defaults to zero; `dropout` controls attention
output/MLP dropout and `attention_dropout` controls attention probabilities.

```python
from lift_classify import LiFTConfig, small_config, paper_reference_config

custom = LiFTConfig(image_size=224, patch_size=16, num_classes=100,
                    hidden_dim=256, num_heads=4, prelude_depth=2,
                    core_depth=3, num_loops=4)
small = small_config(num_classes=100)          # D=384, heads=6, P=2, R=2, K=2
deeper_tiny = tiny_config(num_loops=4)          # D=192, heads=3, P=2, R=2
paper_b = paper_reference_config("B", core_depth=2)  # 4x32x32 input, patch=2
rgb_b = paper_reference_config("B", core_depth=2,
                               image_size=224, in_channels=3, patch_size=16)
```

Paper presets preserve backbone dimensions and the table's P/R/K choices, but
still use the classification conditioning and head. They are not pretrained
models or complete generation models. Their total parameter counts should not
be compared for equality with the paper's generative models.

## Initialization and validation

Blocks use zero-initialized AdaLN modulation, making them identity maps initially.
The classifier has nonzero Xavier weights so CE reaches the patch embedding on
the first backward. Attention/MLP and depth-embedding gradients are initially zero
behind the gates; the tests verify that optimizer updates unlock these paths.
Zero-initialization is not evidence of disconnected weights, and initial logits
need not differ across K.

The test suite covers equations, QK norm/RoPE ordering, recurrence and first
reinjection, weight identity, shapes, intermediate outputs, CE gradients, numerical
stability, CPU BF16 autocast, presets, exact parameter counts, and cost accounting.
See [architecture and source decisions](docs/architecture.md) and
[validation results](docs/validation.md).

## Parameters, compute, and CPU timing

Run from the repository root:

```sh
python -m tools.analyze_model --preset tiny --loops 1 2 4 8
python -m tools.analyze_model --preset tiny --loops 1 2 4 8 --latency --threads 1 --json
python -m tools.analyze_model --preset paper-L --rgb --loops 4
python -m tools.analyze_model --hidden-dim 256 --num-heads 4 --core-depth 3 --loops 2 4
```

Analytic comparisons allocate no models. `--latency` constructs and times only the
LiFT classifier on CPU, with warmup, inference mode, median/IQR, and environment
metadata. FLOPs count multiply-add as two operations and omit elementwise work;
they are not measured hardware FLOPs or training costs. Dense/looped ViT rows are
analytic baselines, not additional trainable model implementations.

Tiny with 224x224 RGB, patch=16, and 1,000 classes has **3,167,592 parameters**.
K=2/4/8 execute 6/10/18 blocks at approximately **1.279/2.093/3.720 GFLOPs**.
AdaLN adds parameters: this Tiny K=2 model is slightly larger than a conventional
6-block ViT of the same width. Shared weights do not guarantee lower latency,
activation memory, or better accuracy.
