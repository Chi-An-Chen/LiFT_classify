# Validation

## Public-release preparation (2026-10-15)

After the attribution/license and report-path changes, the complete suite passed
**149 tests with 2 CUDA tests skipped** in 43.23 seconds. This run was outside the
sandbox so multiprocessing shared memory and MPS were available. It includes the
existing exact CPU resume check and new assertions for single-image filenames,
directory-relative prediction paths with duplicate basenames, and evaluation
reports without absolute dataset/checkpoint paths. No full training or CUDA
benchmark was performed.

Wheel building succeeded; its metadata names Chi-An Chen and includes both the
MIT LICENSE and THIRD_PARTY_NOTICES.md. Local Markdown links and diff whitespace
checks passed. The isolated release history retains five commits with public
noreply identities; all reachable historical blobs and current source files were
scanned for known private paths, private email addresses and common credential
formats, with no matches. This is a targeted scan, not proof that every possible
secret format is absent. The original paper PDF's bytes were preserved.

These checks apply to the prepared local release copy. Remote history replacement
and a fresh-clone verification remain required before changing visibility.

## Earlier validation records

The original model-only results below predate the FAS trainer. The added FAS
training, resume and inference suite now passes **149 tests with 2 CUDA skips**;
MPS FP32 inference and multi-worker loading were checked outside the sandbox.
See [FAS validation and dataset audit](fas_training.md#validation-boundary).

Validated on 2026-10-14 with Python 3.12.13, PyTorch 2.14.0, and pytest 9.1.1,
using an existing environment without installing/upgrading its dependencies.
Runtime platform: macOS 27.0.1, arm64, CPU. The exact CPU SKU was not available
through the sandbox. MPS was unavailable; CUDA and MPS were not tested.

## Automated checks

```sh
python -m pytest -q
```

Result: **130 passed in 2.76 seconds**. Tests run with one CPU thread and fixed
seeds; the thread setting is restored after the test session.

- Manual attention softmax and spatial rotation references verify per-head Q/K
  normalization, affine weights, RoPE ordering, row-major indexing, and noncausal
  attention.
- Nonzero modulation fixtures verify attention-then-MLP residual ordering,
  conditional token normalization before GAP, and depth dependence.
- Manual recurrent unrolling verifies Eq. (9), including the first reinjection,
  distinct boundary/block norms, per-example coordinates, and the final readout.
- K=1/2/4/8 cover batch sizes 1/2/4 and 32/224/256/rectangular resolutions. Core
  parameter object IDs, counts, and state-dict keys stay fixed across K.
- Intermediate states preserve autograd; final logits do not change when readouts
  are requested. Head-call counts confirm state-only requests avoid extra heads.
- CE initially reaches the patch embed and classifier while attention/MLP/depth
  gradients are zero behind gates. Five SGD updates on a synthetic multiclass
  batch unlock all required module groups, including the depth embedding and
  learned constant condition.
- FP32 inputs at scales 0, 1e-10, 1, and 1e4 remain finite through K=8 with activated
  gates, including backward. CPU BF16 autocast forward/backward and native FP16
  forward are checked. RMS statistics avoid FP16 squaring overflow.
- All paper preset configurations match the P/R/K table; default B/L/XL models
  are constructed on the meta device to verify parameter counts without allocating
  large real weights. These are not full-size trained-model execution tests.
- Exact analytic parameter counts match actual model tensors; noninteger MLP
  ratios, invalid configurations/grids, random-grid CPU sampling, CLI output,
  checkpoint compatibility across loop counts, and benchmark metadata are checked.

A separate default-Tiny integration check uses RGB 224x224, batch=2, ten classes,
final CE backward, and a K=8 inference forward. This is a synthetic training-interface
check, not a dataset evaluation or convergence result.

The package also builds successfully as a wheel with no dependency installation:

```sh
python -m pip wheel . --no-deps --no-build-isolation --no-index --wheel-dir /tmp/lift-classify-wheel
```

## Costs and CPU timing

```sh
python -m tools.analyze_model --loops 1 2 4 8 --latency --threads 1 --min-run-time 2 --json
```

Actual report: [cpu_benchmark.json](cpu_benchmark.json). Configuration: Tiny,
RGB 224x224, patch=16, D=192, heads=3, P=2, R=2, 1,000 classes, batch=1,
CPU FP32, one thread, eval + inference mode, three warmup forwards per K.
Timing uses `torch.utils.benchmark` with a minimum two-second measurement window.

| K | Unique blocks | Executed blocks | Actual parameters | Estimated forward GFLOPs | Median ms | IQR ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 3,167,592 | 0.872 | 5.938 | 2.270 |
| 2 | 4 | 6 | 3,167,592 | 1.279 | 7.425 | 0.988 |
| 4 | 4 | 10 | 3,167,592 | 2.093 | 11.917 | 1.137 |
| 8 | 4 | 18 | 3,167,592 | 3.720 | 20.905 | 1.922 |

Timings are descriptive measurements on this shared machine; the IQR shows
material variability. They are not deployment guarantees or comparisons to
executed dense/looped ViT baselines. FLOPs count multiply-add=2 and omit
elementwise work, intermediate readouts, and backward, as defined in
[architecture.md](architecture.md).

No real classification dataset, long training run, accuracy-versus-K curve,
official LiFT checkpoint/implementation parity, or GPU benchmark was evaluated.
