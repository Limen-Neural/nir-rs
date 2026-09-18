# Compatibility

How `nir-rs` relates to upstream [neuromorphs/NIR](https://github.com/neuromorphs/NIR)
and what each release line claims.

## Release matrix

| nir-rs line | Git tag / crates.io | Upstream NIR (fixtures) | Notes |
|-------------|---------------------|-------------------------|--------|
| **0.4.4 (candidate)** | Release candidate; not yet tagged or published | Vendored from neuromorphs/NIR @ `7883c3c` (see `tests/fixtures/README.md`) | Graph model, validation, HDF5 I/O, serde DX, and release hardening |
| 0.4.3 | `v0.4.3` / crates.io **0.4.3** | Same fixture commit | Current published release |
| 0.4.1 | crates.io **0.4.1** | Same fixture commit | First crates.io release; README still org-oriented |
| `v0.4.0` | git tag only (pre-crates.io) | Same fixture commit | First git-tagged consumer pin |

**Only fixture-backed claims are made.** A newer upstream NIR release is **not**
automatically supported until fixtures (and tests) are updated.

### Fixture coverage (0.4.x)

| Fixture | Exercises (node families / structure) |
|---------|----------------------------------------|
| `lif_norse.nir` | Input / Affine / LIF / Output; `f32`; optional `v_reset` **absent** |
| `lif_rockpool.nir` | Rockpool: Input / **Linear** / LIF / Output; underscore names |
| `two_lif_neurons.nir` | Linear + LIF; `f64` |
| `braille_noDelay_bias_zero.nir` | CubaLIF, multi-edge, dotted names; Affine + zero bias |
| `braille_noDelay_noBias_subtract.nir` | CubaLIF RNN with **Linear** (no bias) |
| `braille_noDelay_bias_zero_subgraph.nir` | Nested `NIRGraph` (Affine inner) |
| `braille_noDelay_noBias_subtract_subgraph.nir` | Nested `NIRGraph` (Linear inner) |
| `cnn_sinabs.nir` | Conv2d, IF, SumPool2d, Flatten, Affine |

Provenance, SHAs, MANIFEST, and real-vs-synthetic coverage checklist:
[`tests/fixtures/README.md`](tests/fixtures/README.md).

### Hugging Face–derived fixtures (0.4.x)

Separate from the paper corpus and from Synfire (#43). Converted with
upstream `nir==1.0.8` `nir.write` from pinned Hub checkpoints; CI loads the
vendored files only (no network). See
[`tests/fixtures/huggingface/README.md`](tests/fixtures/huggingface/README.md).

| Fixture | Hub source (pinned revision) | Exercises |
|---------|------------------------------|-----------|
| `huggingface/neurocuda_mlp_mnist.nir` | `Krishnav1234/neurocuda-mlp-mnist-snn` `@5a24224` | Affine + IF MLP; NIR `/version` **1.0.8** |
| `huggingface/neurocuda_cnn_nmnist.nir` | `Krishnav1234/neurocuda-cnn-nmnist-snn` `@1ee6ba2` | Conv2d, IF, **AvgPool2d**, Flatten, Affine |

## Fidelity semantics

These are intentional, documented behaviors — not bugs:

| Topic | Behavior |
|-------|----------|
| Round-trip | **Graph-level**, not byte-identical HDF5. Field values and topology match after read → write → read. |
| Node order | Insertion / iteration order is stable in-memory (`IndexMap`); wire order may differ from Python producers. |
| Dtype widening | Some metadata may widen (e.g. integer/scalar forms) within documented IO rules; equality checks use graph semantics, not raw HDF5 types. |
| Optional fields | e.g. LIF `v_reset`, Cuba `w_in` — absent on the wire means default / `None` in Rust; presence is preserved. |
| Wire type strings | Must match neuromorphs/NIR exactly (`CubaLIF`, `Conv2d`, `SumPool2d`, …). Marketing aliases are rejected. |
| Parameter validation | Opt-in via `NirGraph::validate_parameters`. Reads stay permissive; default writes do not run convolution/pooling checks. |
| `/version` on read | Default `read` is **permissive** (missing or arbitrary strings stored verbatim). Opt in with `ReadOptions::version_policy`: `RequirePresent`, or `CompatibleMajor` with caller-supplied majors (typically `[0, 1]` for paper fixtures and 1.x writers). Prerelease/build suffixes are parsed then ignored for the major check; the original string is stored. Policy failures use `NirError::IncompatibleVersion` and run before the graph body is decoded. |

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
| Dev / CI / `package.rust-version` | **1.98.1** | Same pin in `rust-toolchain.toml`, GitHub Actions, and crates.io metadata |

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
- #44 / LIM-1086 — Hugging Face SNN → NIR → nir-rs load (this corpus)  
