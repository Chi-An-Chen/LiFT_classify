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

Python 3.10+ and PyTorch 2.2+ are required. The model itself only depends on PyTorch.
The optional FAS training/evaluation tools also use torchvision, Pillow, NumPy and scikit-learn.

```sh
python -m pip install -e '.[test]'
python -m pytest -q
```

## FAS training and inference

The FAS trainer preserves the original `train/{fake,live}` and `test/{fake,live}`
directories, using `test` as **validation for checkpoint selection**. It retains
duplicates as requested and records their hashes and cross-split overlap. These
scores are not independent test/generalization results.

Run these tools from the repository root. For RTX 5090, first install a matching
official PyTorch/torchvision CUDA 12.8+ build that supports Blackwell; the model's
minimum PyTorch version alone does not ensure GPU compatibility.

```sh
python -m pip install -e '.[train,test]'
python tools.train.py --config configs/fas_lift_tiny.json \
  --data-root /path/to/FAS_all --output-dir outputs/fas_lift_tiny --device cuda
python tools.train.py --data-root /path/to/FAS_all \
  --output-dir outputs/fas_lift_tiny --resume outputs/fas_lift_tiny/last.pt --device cuda
python -m tools.predict --checkpoint outputs/fas_lift_tiny/best.pt \
  --input /path/to/image.jpg --output outputs/predictions.csv --device auto
python -m tools.evaluate --checkpoint outputs/fas_lift_tiny/best.pt \
  --data-root /path/to/FAS_all --output-dir outputs/fas_evaluation --loops 1 2 4 8
```

Default recipe: full-image 256px letterbox, patch16, D192, P2/R2/K4, weighted final
CE, 100 epochs, AdamW, batch128, CUDA BF16. Only mild color/lighting augmentation
is enabled; no crop, rotation, flip, blur, or extra compression. Auto inference
selects CUDA → MPS → CPU and uses FP32. See [the FAS training guide](docs/fas_training.md)
for exact settings, resume requirements, output files and validation limits.

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
