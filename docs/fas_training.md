# FAS_all training

This is a from-scratch classification baseline, not LiFT trajectory training or a
demonstrated optimal FAS recipe. The backbone equations are unchanged. All commands
run from the repository root; the `tools` entrypoints and example config are source
checkout utilities, not wheel-installed console commands.

## Data and protocol

Pass the root containing `train/fake`, `train/live`, `test/fake`, `test/live`.
Optional `manifest.csv` must match the directory contents, split and class names;
its legacy remote paths are not used. Images are sorted by relative path. Class
IDs are always `fake=0`, `live=1`.

The startup audit fully decodes and SHA256-hashes every image, recording size,
mode and class. Corrupt images and conflicting labels on identical bytes raise
errors with their paths. No original files are deleted, renamed or rewritten.
There is no random train/validation split or deduplication.

The user chose to keep `test` as validation for model selection, including known
duplicates. On 2026-10-14 the full local audit decoded all **44,039** images:

| Split | Fake | Live | Total |
|---|---:|---:|---:|
| train | 26,415 | 8,982 | 35,397 |
| test (validation) | 6,024 | 2,618 | 8,642 |

There are 43,274 distinct byte hashes and 176 cross-split duplicate groups,
involving 185 training and 194 validation files. There are 445 within-training
duplicate groups. All are retained. Image bytes have no conflicting labels.
The manifest fingerprint is
`e19dcf77b0f61de9c4113e634cea6ab408d80bdd802f53a659c1805449f7e812`.

These validation scores include images seen during training and may be optimistic.
Byte hashing does not detect every re-encoded/near-duplicate image; subject, video,
session and device independence are unknown. No independent test is provided.

## Shared preprocessing and augmentation

Training, validation and prediction use the same checkpoint-recorded preprocessing:
EXIF orientation → RGB → resize longest side to 256 (bilinear antialias, preserve
aspect ratio) → center-pad to 256×256 with RGB (124,116,104) → normalize with mean
(0.485,0.456,0.406), std (0.229,0.224,0.225). Original borders remain. There is no
face detector, alignment or crop, and ImageNet normalization does not imply
pretrained weights. Padding/background correlations can still become shortcuts.

Training applies photometric augmentation **after resize and before new padding**:

- With probability 0.5: ColorJitter brightness/contrast ±15%, saturation ±10%, hue 0.
- Independently with probability 0.3: choose exactly one of gamma U(0.9,1.1),
  independent RGB gains U(0.95,1.05), or smooth directional illumination.
- Directional illumination multiplies pixels by
  `1 + a*(x*cos(theta)+y*sin(theta))/sqrt(2)`, with normalized x/y in [-1,1],
  theta uniform in [0,2π), and a uniform in [-0.15,0.15]. Output is clamped to [0,1].

Both classes use identical distributions. Validation/prediction are deterministic.
No random/center crop, rotation, flip, perspective, erasing, grayscale, blur,
extra JPEG compression, background replacement, MixUp or CutMix is applied.
Lighting changes are exposure/color proxies, not physically rendered new scenes.
Recipe versions and every numerical setting are stored in checkpoints; the fixed
v1 recipe lives in `fas_data.py` rather than adding another augmentation framework.

## Default configuration

`configs/fas_lift_tiny.json` is the starting configuration. It uses 256² RGB,
patch16, hidden192, heads3, MLP ratio4, P2/R2/K4, dropout0.1, attention dropout0.
There are 2,974,978 parameters and approximately 2.849 forward GFLOPs/image,
excluding elementwise operations. Depth coordinates remain uniform in train/eval.

| Setting | Value |
|---|---|
| Epochs / batch | 100 / 128 (no gradient accumulation) |
| Objective | Class-weighted final CE only |
| Class weights | N/(2*n_class), computed from original train rows |
| Sampling | Shuffled once per epoch; no oversampling; keep final partial batch |
| Optimizer | AdamW, LR 3e-4, betas (0.9,0.999), eps 1e-8 |
| Weight decay | 0.05 on matrices; 0 on biases, norm weights and c_base |
| Schedule | 5 warmup epochs (0.2/0.4/0.6/0.8/1 × base), then cosine to 3e-6 |
| Gradient clip | Global norm 1.0, fail on nonfinite loss/gradient norm |
| Precision | CUDA BF16 autocast, FP32 parameters/loss; MPS/CPU FP32 |
| Seed / workers | 42 / 4; workers reseeded each epoch |
| Selection | Lowest validation ACER at K=4 and threshold0.5 |

Ties use lower **unweighted** validation CE, then retain the earlier epoch. Logged
training CE is a class-weight-denominator aggregate across all batches. No EMA,
label smoothing, distillation, intermediate loss or early stopping is used.

Install the official matching torch/torchvision build for the target hardware,
then `python -m pip install -e '.[train,test]'`. RTX 5090 requires a Blackwell-capable
build (CUDA 12.8+), not merely the package's model-only minimum torch version.
Check the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

```sh
python tools.train.py --config configs/fas_lift_tiny.json \
  --data-root /path/to/FAS_all --output-dir outputs/fas_lift_tiny --device cuda
```

The data root is portable; no local absolute dataset path is baked into code.
`--epochs`, `--batch-size`, `--num-workers` override JSON values for a new run.
For shorter runs, set `warmup_epochs < epochs` in the JSON. Model dimensions and
optimizer settings are configurable in JSON. GPU OOM is surfaced; use a smaller
batch in a new run rather than silently changing the recipe.

`python tools.train.py`, `python tools/train.py`, and `python -m tools.train`
use the same training entrypoint. Rich displays separate Train and Val progress
bars per epoch, updated after each completed batch. Train shows cumulative
weighted loss and accuracy; Val shows cumulative accuracy and fixed-0.5 ACER
(ACER stays `--` until both classes have appeared). Each bar also shows images/s,
ETA and peak CUDA allocated memory for that phase (`--` on CPU/MPS).
The epoch summary follows the MobileCLIP trainer's field names and formatting:
`loss`, `train acc`, `val acc`, `val ACER`, `best`, plus validation loss, APCER,
BPCER, AUC and best epoch. No threshold calibration or AMP-skip fields are shown.
The final best-checkpoint K sweep also displays validation progress. Redirected
output retains completed bars and summaries without per-batch log spam.
No extra optimizer warmup updates are performed and no 5090 latency claim is made.

## Checkpoints, resume and reports

Outputs include config/environment JSON, audited data manifest/overlap JSON,
epoch CSV, `last.pt`, `best.pt`, final validation CSVs for K=1/2/4/8, and a report.
All outputs stay outside the source dataset. New runs reject nonempty output
directories, so existing experiments are not silently overwritten.

`last.pt` owns the committed epoch-boundary run state: current/best weights,
optimizer/scheduler, RNG states, history, best metrics, class weights, configuration,
class order, preprocessing, augmentation and dataset fingerprint. It is written
via a temporary file and atomic replace. `best.pt` is an inference export; it is
regenerated from `last.pt` on resume if interruption left the export missing/stale.

```sh
python tools.train.py --data-root /path/to/FAS_all \
  --output-dir outputs/fas_lift_tiny --resume outputs/fas_lift_tiny/last.pt --device cuda
```

Resume uses the saved config unless one is explicitly provided. It requires the
same config, augmentation and dataset content, and `last.pt` in the same output
directory. Changing the total epoch budget changes the cosine schedule and is a
new experiment, not an exact resume. An interrupted epoch restarts from its last
completed boundary; a first-epoch failure before `last.pt` exists can be restarted
in a new output directory. CPU resume equivalence is tested; different hardware,
PyTorch builds and CUDA kernels are not promised bitwise identical. Driver and
runtime versions, device/precision and GPU memory are recorded when available.

The fixed threshold accepts live when `p_live >= 0.5`. APCER is the fraction of
fake images accepted as live; BPCER is the fraction of live images rejected;
ACER is their mean. Accuracy and ROC-AUC (live positive) are also reported. Scores
are image-level and not source/attack-type-specific official PAD benchmark results.
There is no validation-optimized threshold. K sweeps reuse the same selected
checkpoint and do not automatically change its default K.

## Evaluation and prediction

```sh
python -m tools.evaluate --checkpoint outputs/fas_lift_tiny/best.pt \
  --data-root /path/to/FAS_all --output-dir outputs/fas_evaluation --loops 1 2 4 8
python -m tools.predict --checkpoint outputs/fas_lift_tiny/best.pt \
  --input /path/to/image_or_directory --output outputs/predictions.csv --device auto
```

`auto` selects CUDA → MPS → CPU according to availability. Explicit unavailable
devices fail, as do runtime errors/OOM; they are not silently rerouted. Evaluation
and prediction use FP32, `eval()` and `inference_mode()`. No stochastic test-time
augmentation is used. `--num-loops` optionally overrides the checkpoint default
in prediction. CSVs include p_live, p_fake and the fixed-threshold label. Standalone
evaluation reads `test/` as validation; it reports checkpoint overlap provenance
but does not assert a fresh train/test hash audit on a different supplied dataset.

## Validation boundary

Run `python -m pytest -q` with the train/test extras. Without training dependencies,
FAS test modules skip and the model-only tests remain available. Tests include
full-frame preprocessing, augmentation, audit failure paths, class/metric direction,
weighted CE, decay exclusions, LR schedule, train/save/load/CLI integration,
exact CPU interrupted-resume equivalence, and checkpoint write-failure recovery.
CUDA/MPS parity and CUDA BF16 backward tests run only with those accelerators.

On 2026-10-14 the complete suite passed **149 tests, with 2 CUDA tests skipped**, in
36.31 seconds (Python 3.12.13, torch 2.14.0, torchvision 0.29.0). The sandbox blocks
PyTorch shared-memory workers and hides MPS; the full suite ran outside that sandbox.
Two-worker augmentation reproducibility and MPS FP32 inference against CPU passed,
including nonzero attention/MLP gates (atol/rtol 1e-4). Both CUDA tests remain
unverified because no NVIDIA GPU is available locally.

The full data decode/hash audit succeeded. A separate default-config CPU smoke check
used four actual FAS training images, original class weights (0.670017,1.970441),
and three optimizer updates; loss/gradients were finite and the depth embedding
received a nonzero gradient. No weights were saved from that check. Wheel building
and all documentation local links also passed. No 5090 training benchmark or full
FAS training was performed. These checks establish functionality, not classification
accuracy or a benefit from increasing loops.
