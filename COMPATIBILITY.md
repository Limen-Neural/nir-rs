# Compatibility

How `nir-rs` relates to upstream [neuromorphs/NIR](https://github.com/neuromorphs/NIR)
and what each release line claims.

## Release matrix

| nir-rs line | Git tag / crates.io | Upstream NIR (fixtures) | Notes |
|-------------|---------------------|-------------------------|--------|
| **0.4.x** | `v0.4.0` (git; crates.io pending #26) | Vendored from neuromorphs/NIR @ `7883c3c` (see `tests/fixtures/README.md`) | Graph model + HDF5 I/O + serde DX |
| Unreleased post-0.4.0 | `main` | Same fixture corpus | Multi-OS CI + package/semver gates |

**Only fixture-backed claims are made.** A newer upstream NIR release is **not**
automatically supported until fixtures (and tests) are updated.

### Fixture coverage (0.4.x)

| Fixture | Exercises (node families / structure) |
|---------|----------------------------------------|
| `lif_norse.nir` | Input / Affine / LIF / Output; `f32`; optional `v_reset` **absent** |
| `two_lif_neurons.nir` | Linear + LIF; `f64` |
| `braille_noDelay_bias_zero.nir` | CubaLIF, multi-edge, dotted names |
| `braille_noDelay_bias_zero_subgraph.nir` | Nested `NIRGraph` |
| `cnn_sinabs.nir` | Conv2d, IF, SumPool2d, Flatten, Affine |

Provenance, SHAs, and license for fixtures: [`tests/fixtures/README.md`](tests/fixtures/README.md).

## Fidelity semantics

These are intentional, documented behaviors — not bugs:

| Topic | Behavior |
|-------|----------|
| Round-trip | **Graph-level**, not byte-identical HDF5. Field values and topology match after read → write → read. |
| Node order | Insertion / iteration order is stable in-memory (`IndexMap`); wire order may differ from Python producers. |
| Dtype widening | Some metadata may widen (e.g. integer/scalar forms) within documented IO rules; equality checks use graph semantics, not raw HDF5 types. |
| Optional fields | e.g. LIF `v_reset`, Cuba `w_in` — absent on the wire means default / `None` in Rust; presence is preserved. |
| Wire type strings | Must match neuromorphs/NIR exactly (`CubaLIF`, `Conv2d`, `SumPool2d`, …). Marketing aliases are rejected. |

## Features

| Feature | Default | Requires | What you get |
|---------|---------|----------|--------------|
| *(none)* | yes | No native libs | Graph model, validation, errors |
| `serde` | no | — | `Serialize` / `Deserialize` for graph types |
| `hdf5` | no | System **libhdf5** (or static recipe) | `.nir` read/write + fixture tests |

Default builds and default-feature CI need **no** Python and **no** libhdf5.

## Toolchain (MSRV)

| Pin | Version | Role |
|-----|---------|------|
| Dev / CI | **1.97.1** | Quality bar (`rust-toolchain.toml`, GitHub Actions OS matrix) |
| `package.rust-version` | **1.85.1** | Cargo/crates.io floor (edition 2024 + `hdf5-metno` 0.14) |

We do not matrix CI on the cargo floor. Raise the floor when dependencies or
language features require it; document the bump in the changelog.

## Updating claims when upstream NIR moves

1. Add or refresh licensed fixtures under `tests/fixtures/` (with provenance).
2. Extend HDF5 / graph tests so claims stay **test-backed**.
3. Update the matrix table above and fixture list.
4. Note the change under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
5. Do **not** claim support for upstream versions without fixtures.

## Related issues

- #26 — first crates.io publish  
- #27 / #32 — multi-OS CI  
- #28 / #33 — package + semver CI  
- #29 — expand fixture corpus  
- #31 — this document and changelog policy  
