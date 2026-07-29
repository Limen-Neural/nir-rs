# NIR test fixtures

Real `.nir` (HDF5) files written by the **Python** reference implementation,
vendored here so `nir-rs` can prove wire compatibility from pure-Rust tests —
no Python interpreter is needed to build, test, or use this crate.

## Provenance

Source: [neuromorphs/NIR](https://github.com/neuromorphs/NIR)
Commit: `7883c3c85f1be27ed113ccc9e8d6ab47ab541df4`

| File | Upstream path | Covers |
|------|---------------|--------|
| `lif_norse.nir` | `paper/01_lif/lif_norse.nir` | `Input` / `Affine` / `LIF` / `Output`, `f32`, **`v_reset` absent on the wire** |
| `two_lif_neurons.nir` | `paper/01_lif/debug_spike_representation/two_lif_neurons.nir` | `Linear` + `LIF` chain, `f64` |
| `braille_noDelay_bias_zero.nir` | `paper/03_rnn/braille_noDelay_bias_zero.nir` | `CubaLIF`, recurrence, multi-edge, dotted node names (`lif1.lif`) |
| `braille_noDelay_bias_zero_subgraph.nir` | `paper/03_rnn/extras/braille_noDelay_bias_zero_subgraph.nir` | nested `NIRGraph` nodes |
| `cnn_sinabs.nir` | `paper/02_cnn/cnn_sinabs.nir` | `Conv2d`, `IF`, `SumPool2d`, `Flatten`, `Affine` |

SHA-256 of the vendored copies:

```
0dd9143ef624892d4a6461f474d93653a337b3490a62e23ec9ec5d18fde9b7b6  lif_norse.nir
806c7c1cfae72b5be92ce15a670f37014b82a5d4fba8a077efce46d193c21a82  two_lif_neurons.nir
f1aab3ce74024e7feac508a03b04c58d483ee3a01e293dba4b53d0b519b8650e  braille_noDelay_bias_zero.nir
58132a83a42d614b411948a9f2bc9817e648577b58b70b5b2c4c86021d437a1b  braille_noDelay_bias_zero_subgraph.nir
e2fa55bda7aab5a772485e1b690358bcb825b303eca7dc426e3973937fcb5bcb  cnn_sinabs.nir
```

These files are **inputs to tests only**. They are not part of the published
library API, and nothing in `src/` depends on them.

## License of the vendored files

The upstream NIR project is BSD 3-Clause licensed. That license, reproduced in
full as required by clause 2, governs the `.nir` files in this directory (it
does **not** apply to the rest of `nir-rs`, which is MIT OR Apache-2.0).

```
BSD 3-Clause License

Copyright (c) 2023, NIR Team.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
