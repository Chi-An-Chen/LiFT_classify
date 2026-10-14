# Architecture and source contract

## References and verification boundary

Ground truth is [LiFT v1](https://arxiv.org/html/2610.05538v1), especially sections
3.1, 3.2, 4.1, D.1, D.3, and the compute expressions in D.5.1. The downloaded
`2610.05538v1.pdf` remains unchanged.

The block reference is the MIT-licensed
[SpeedrunDiT repository](https://github.com/SwayStar123/SpeedrunDiT), specifically
[models/sit.py](https://raw.githubusercontent.com/SwayStar123/SpeedrunDiT/main/models/sit.py).
On 2026-10-14 the accessible raw web snapshot contained 640 lines. Inspected symbols
were `Attention` (lines 98–168), `SiTBlock` (170–216), `FinalLayer` (218–241),
`TimestepEmbedder` (31–67), and `SiT.initialize_weights` (337–371).

No upstream source file is vendored or imported. The equations and choices below
freeze the behavioral contract used for this implementation. GitHub commit metadata
was inaccessible, so an immutable upstream commit/hash is **not verified**. A cached
GitHub HTML view reported a different file length; `main` links alone are not a
version pin. Recheck upstream against this contract before claiming exact code
parity or loading an official checkpoint. No official LiFT code or checkpoint was
verified: the paper links say Coming Soon and the author's demo source was
inaccessible. This package does not claim official checkpoint compatibility.

## Block contract

For tokens `x [B,N,D]` and condition `c [B,D]`, each block computes:

```text
shift_a, scale_a, gate_a, shift_m, scale_m, gate_m
    = chunk_6(Linear_D_to_6D(SiLU(c)))
u = RMSNorm_1(x) * (1 + scale_a) + shift_a
x1 = x + gate_a * Attention(u)
v = RMSNorm_2(x1) * (1 + scale_m) + shift_m
x2 = x1 + gate_m * MLP(v)
```

Conditions broadcast over tokens; gates are unconstrained, without sigmoid or
extra residual scaling. Attention uses biased QKV, separate per-head Q/K RMSNorm,
Q/K RoPE, noncausal scaled-dot-product attention, and biased output projection.
V is not normalized or rotated. The Q/K norm weights each have shape `[head_dim]`
and are shared across heads. MLP is biased Linear → tanh-approximate GELU → biased
Linear, with optional dropout after GELU and the second projection. No
`TransformerEncoderLayer`, SwiGLU, value residual, SPRINT routing/fusion, external
CLS representation, REPA projector, or generative training mechanisms are used.

### Normalization and position assumptions

- Token, boundary, and readout norms are non-affine RMSNorm. Q/K norms are affine.
- All norms use `eps=1e-6` and FP32 statistics, returning the activation dtype.
  The upstream Q/K norm defaults to `eps=None`; the explicit epsilon is a numerical
  adaptation, not a verified LiFT detail.
- RoPE is axial: first half of head channels encodes row, second half column;
  adjacent pairs rotate together. Pair frequency is `10000**(-j/(head_dim/4))`.
  Patch coordinates start at zero and flatten in row-major order. The original
  helper was inaccessible, so pairing, theta, axis convention, and resizing
  behavior are assumptions. No absolute positions or CLS token are added.
- Actual grid coordinates are regenerated for each spatial shape. Changing
  resolution does not interpolate a learned position table; accuracy across
  resolutions is untested.

### Initialization

- QKV/output/MLP and classifier weights: Xavier uniform, biases zero.
- Patch Conv2d kernel: Xavier uniform after flattening to `[D,patch_dimension]`;
  bias zero.
- Q/K affine RMS weights: one.
- Block and coda modulation final Linear weights/biases: zero.
- Learned `c_base`: normal standard deviation 0.02.
- Depth embedding MLP weights: normal standard deviation 0.02, biases zero.
- Classification projection stays nonzero, unlike the generative output zeroing.

## Recurrence and classification changes

Equation (9) is implemented as:

```text
h0 = Prelude(PatchEmbed(images), c_base)
h = h0
for s in coordinates:
    h = Core(RMSNorm(h) + h0, c_base + DepthEmbedding(s))
logits = Coda(h, c_base)
```

The first loop also reinjects h0. A single Core stack owns its blocks and is reused
at every loop. All operations between loops preserve hidden-state width. There is
no boundary projection, extra skip, step-size multiplier, hidden-state detach, or
normalization after reinjection beyond the block's own norms. The boundary norm is
separate from the block norms. No state persists between separate image forwards.

`c_base` is learned but constant across examples and depths. It replaces the
generative timestep/class condition without accepting class labels. Prelude and
Coda use only `c_base`; only Core receives depth. The depth embedding takes raw
continuous s, uses 256 Fourier features with period 10,000, then Linear → SiLU →
Linear. The Fourier width follows D.5.1; the precise depth frequency/scaling
implementation is unverified. There is no learned discrete loop-index table.

The prediction-only Coda contains zero Transformer blocks, then:

```text
shift, scale = chunk_2(Linear_D_to_2D(SiLU(c_base)))
features = mean_over_tokens(RMSNorm(h) * (1 + scale) + shift)
logits = Linear_D_to_num_classes(features)
```

Normalization occurs before pooling. All depths share the readout. Reading a state
does not alter the recurrence. GAP keeps the grid purely spatial; CLS would require
an additional positional convention and is not implemented.

### Reproduction matrix

| Design | Status | Implementation/reason |
|---|---|---|
| Eq. (9), weight sharing, input reinjection | Preserved | Same state transition and boundary order |
| RMS/QK norms, RoPE, GELU, gated AdaLN | Preserved components | Reference block equations; numerical/positional details above are assumptions |
| Prelude/Core/Coda conditions | Adapted | Fixed learned condition replaces generation time and labels |
| Patch embedding | Adapted | RGB default; latent-shaped reference inputs still supported |
| Shared normalized/modulated readout | Adapted | GAP + classification projection replaces velocity/unpatchify |
| Uniform inference grid | Preserved | k/K, endpoint 1 |
| Training grid | Adapted default | Uniform with final CE; explicit random-grid helper is opt-in |
| LiFT trajectory/prelude losses | Removed | No velocity target or straight-path classification target assumed |
| Flow noise, sampler, VAE, CFG | Removed | No generation pipeline |
| Exact official initialization/position implementation | Unverified | No official code parity claim |

The random helper follows D.3's FP64 sorted draws with
`s_i=(1-K*1e-6)*r_sorted_i+i*1e-6`, then casts to FP32 and appends 1. It does not
multiply the transition by interval length. The initial s=0 corresponds to h0
and is not a Core call. The helper bounds K below 1,000,000 to keep the correction
valid.

Final CE is the only baseline loss. Deep supervision and self-distillation can use
returned states/readouts, but no detachment policy or loss weights are imposed on
those experiments. The paper's detached prediction anchor is not a detached h0;
hidden states always retain gradients here.

## Presets and analytic cost

| Paper scale | Width | Heads | Dense depth | Prelude | (Core, training K) |
|---|---:|---:|---:|---:|---|
| B/2 | 768 | 12 | 12 | 4 | (1,8), (2,4), (4,2) |
| L/2 | 1024 | 16 | 24 | 4 | (1,20), (2,10), (4,5), (5,4), (10,2) |
| XL/2 | 1152 | 16 | 28 | 4 | (1,24), (2,12), (3,8), (4,6), (6,4), (8,3), (12,2) |

Paper reference inputs are `[B,4,32,32]` with 2x2 patches, not raw RGB patch=2.
Defaults select Core=2/5/6 respectively and K=4. MLP ratio=4, Coda depth=0, and
dropout=0. Overrides are allowed and may deliberately leave the reference regime.

Let N be tokens, D width, M=`int(D*mlp_ratio)`, q input patch dimension, C classes,
d head dimension, U=P+R unique blocks, E=P+KR executed blocks. The exact trainable
parameter expressions for this implementation are:

```text
LiFT block:      10D² + 2DM + 11D + M + 2d
patch embed:    Dq + D
base condition: D
depth embed:    D² + 258D
coda modulation:2D² + 2D
classifier:     DC + C
```

Forward FLOP estimates count multiply-add=2:

```text
block = N(8D² + 4DM + 4ND) + 12D²
total = E*block + 2NqD + K*2(256D+D²) + 4D² + 2DC
```

This excludes biases, norms, softmax, GELU, rotary arithmetic, pooling, residual
arithmetic, and dropout. It includes final readout only, not intermediate heads or
backward. It also includes dense Linear costs even when modulation is initialized
to zero; no branches are skipped. Activation memory and latency are separate.

The tool compares Standard ViT (pre-LayerNorm, absolute positions, GAP), Looped
ViT with the same blocks and boundary reinjection but no depth conditioning, LiFT
classification, and an untied dense model using the same LiFT blocks. These are
analytic comparisons; the extra baselines are not implemented training pipelines.
The dense same-block row omits depth embedding because it has no recurrent depth
condition. A matched-depth parameter comparison is not a controlled accuracy
experiment.

## Known limits

Longer rollouts can drift or amplify gradients. Shared weights reduce unique
parameters without removing repeated compute or training activations. Initial
identity blocks do not prove learned stability; tests also activate gates before
checking recurrent gradients and extreme inputs. No training on a real dataset,
accuracy-versus-K conclusion, checkpoint parity, GPU/MPS benchmark, automatic early
exit, checkpointed backward, or compile/throughput optimization is included.
