# NIR test fixtures

Real `.nir` (HDF5) files written by the **Python** reference implementation and
ecosystem exporters (Norse, Rockpool, Sinabs paper pipelines), vendored here so
`nir-rs` can prove wire compatibility from pure-Rust tests — no Python
interpreter is needed to build, test, or use this crate.

## Provenance

| Field | Value |
|-------|--------|
| Source repository | [neuromorphs/NIR](https://github.com/neuromorphs/NIR) |
| Source commit | `7883c3c85f1be27ed113ccc9e8d6ab47ab541df4` |
| License of `.nir` files | BSD 3-Clause (see below) |
| Modified? | **No** — byte-identical copies of upstream paper artifacts |
| Producing NIR package version | Embedded `/version` is `0.1.1` for most paper files; `lif_rockpool.nir` is `0.2.0` |

Machine-readable catalog: [`MANIFEST.toml`](MANIFEST.toml).

## Fixture table

| File | Upstream path | Producer flavour | Notable coverage |
|------|---------------|------------------|------------------|
| `lif_norse.nir` | `paper/01_lif/lif_norse.nir` | Norse | `Input` / `Affine` / `LIF` / `Output`; `f32`; `v_reset` **absent** |
| `lif_rockpool.nir` | `paper/01_lif/lif_rockpool.nir` | Rockpool | Same topology with **`Linear` (no bias)**; underscore node names (`0_LinearTorch`) |
| `two_lif_neurons.nir` | `paper/01_lif/debug_spike_representation/two_lif_neurons.nir` | Norse debug | `Linear` + `LIF` chain; **`f64` throughout** |
| `braille_noDelay_bias_zero.nir` | `paper/03_rnn/braille_noDelay_bias_zero.nir` | RNN / braille | `CubaLIF`, recurrence, multi-edge, dotted names (`lif1.lif`); `Affine` with zero bias |
| `braille_noDelay_noBias_subtract.nir` | `paper/03_rnn/braille_noDelay_noBias_subtract.nir` | RNN / braille | Same family with **`Linear` (no bias)** and different hidden size (40) |
| `braille_noDelay_bias_zero_subgraph.nir` | `paper/03_rnn/extras/braille_noDelay_bias_zero_subgraph.nir` | RNN / nested | Nested `NIRGraph`; unresolved inner edge (upstream quirk) |
| `braille_noDelay_noBias_subtract_subgraph.nir` | `paper/03_rnn/extras/braille_noDelay_noBias_subtract_subgraph.nir` | RNN / nested | Nested graph + inner **`Linear`**; same unresolved-edge quirk |
| `cnn_sinabs.nir` | `paper/02_cnn/cnn_sinabs.nir` | Sinabs CNN | `Conv2d`, `IF`, `SumPool2d`, `Flatten`, `Affine`; multi-axis shapes |

## Coverage checklist (#29)

What this **interoperability** corpus exercises vs. what is only covered by
**synthetic** round-trips in `tests/hdf5_roundtrip.rs`:

| Wire family / structure | Real fixture? | Notes |
|-------------------------|---------------|--------|
| Input / Output | yes | All LIF + braille + CNN |
| Affine | yes | Norse LIF, bias-zero braille, CNN |
| Linear | yes | Rockpool, two_lif, noBias braille |
| Scale | synthetic | No licensed paper `.nir` yet |
| Conv1d | synthetic | — |
| Conv2d | yes | `cnn_sinabs.nir` |
| I / LI / IF / LIF | IF+LIF yes; I/LI synthetic | — |
| CubaLI / CubaLIF | CubaLIF yes; CubaLI synthetic | braille family |
| Delay / Threshold | synthetic | — |
| Flatten | yes | CNN |
| SumPool2d | yes | CNN |
| AvgPool2d | synthetic | — |
| Nested `NIRGraph` | yes | both braille subgraph files |
| Explicit padding (pairs) | yes | CNN Conv2d |
| Symbolic padding string | synthetic | — |
| Optional `v_reset` absent | yes | LIF + CubaLIF paper files |
| Optional `w_in` present | yes | braille CubaLIF |
| `f32` / `f64` mix | yes | braille (Linear f32 + Cuba f64) |
| Integer shape metadata | yes | Input/Output shapes |
| Empty-edge / minimal graph | partial | smallest: rockpool / norse LIF (3 edges) |
| Multi-layer nontrivial | yes | CNN + braille RNN |

Do **not** claim support for node families that appear only in the synthetic
column until a licensed real-world fixture is added.

## SHA-256 of the vendored copies

```
0dd9143ef624892d4a6461f474d93653a337b3490a62e23ec9ec5d18fde9b7b6  lif_norse.nir
f493a2cd00e20be6305acd7808faa1d4558993c91f3af43454bf637371e543d6  lif_rockpool.nir
806c7c1cfae72b5be92ce15a670f37014b82a5d4fba8a077efce46d193c21a82  two_lif_neurons.nir
f1aab3ce74024e7feac508a03b04c58d483ee3a01e293dba4b53d0b519b8650e  braille_noDelay_bias_zero.nir
14f59e0903d76437be3cb23d7e4be6b4e08ea476e482f7478dbb4592b29377d3  braille_noDelay_noBias_subtract.nir
58132a83a42d614b411948a9f2bc9817e648577b58b70b5b2c4c86021d437a1b  braille_noDelay_bias_zero_subgraph.nir
8038ab6b095554950ec966da0f9cb72d5a97593afd6bb4b37a7887c4bfd81b9a  braille_noDelay_noBias_subtract_subgraph.nir
e2fa55bda7aab5a772485e1b690358bcb825b303eca7dc426e3973937fcb5bcb  cnn_sinabs.nir
```

These files are **inputs to tests only**. They are not part of the published
library API, and nothing in `src/` depends on them.

Hostile / malformed inputs for security tests live under the HDF5 untrusted
test suite (generated at runtime) — **not** in this directory.

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
