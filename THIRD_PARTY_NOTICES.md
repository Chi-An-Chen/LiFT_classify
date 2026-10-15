# Third-party notices

## License scope

Original code and documentation in this repository are licensed under the MIT
License in [LICENSE](LICENSE), copyright (c) 2026 Chi-An Chen. Third-party
materials retain their own licenses. In particular, the paper PDF below is not
relicensed under MIT. Dataset and pretrained-weight rights are not granted by
this repository; neither datasets nor pretrained weights are distributed here.

## LiFT paper — CC BY 4.0

Mohammad Mahdi Derakhshani, Pedro M. P. Curvo, Gertjan J. Burghouts,
Jan-Willem van de Meent, and Cees G. M. Snoek. **LiFT: Loop Flow Transformers**,
2026, arXiv:2610.05538v1.

- Source: <https://arxiv.org/abs/2610.05538v1>
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Included material: [2610.05538v1.pdf](2610.05538v1.pdf), retained unchanged.

The classification architecture is inspired by this paper. The classification
conditioning, readout, final cross-entropy objective and FAS tools are adaptations
by Chi-An Chen; they do not reproduce the paper's generative trajectory training.
This is an unofficial project, without affiliation with or endorsement by the
paper's authors. The authors retain rights in their paper.

## SpeedrunDiT — MIT

[SpeedrunDiT](https://github.com/SwayStar123/SpeedrunDiT) by Swayam Bhanded was the
block implementation reference. Adapted design areas include attention and
modulation in `lift_classify/layers.py` and `lift_classify/blocks.py`, the Fourier
embedding in `lift_classify/conditioning.py`, and readout/initialization choices
in `lift_classify/model.py`. No complete upstream file is vendored or imported.
The upstream source also credits GLIDE, MAE and SPRINT; see its source header.

Reference inspected for this release preparation on 2026-10-15:
[models/sit.py at c24c2ff25699cce63174ca56c2afcfeeb225e367](https://github.com/SwayStar123/SpeedrunDiT/blob/c24c2ff25699cce63174ca56c2afcfeeb225e367/models/sit.py).
This identifies the current reference check, not a verified original implementation
revision or a claim of exact code parity. See [architecture](docs/architecture.md)
for numerical/positional assumptions and omitted upstream features.

The [upstream MIT notice](https://github.com/SwayStar123/SpeedrunDiT/blob/c24c2ff25699cce63174ca56c2afcfeeb225e367/LICENSE)
is reproduced in full below:

```text
MIT License

Copyright (c) 2025 Swayam Bhanded

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
