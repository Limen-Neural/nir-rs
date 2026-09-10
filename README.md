# nir-rs

**Pure-Rust implementation of the Neuromorphic Intermediate Representation (NIR)**

[![CI](https://github.com/Limen-Neural/nir-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/nir-rs/actions)
[![Docker](https://github.com/Limen-Neural/nir-rs/actions/workflows/docker.yml/badge.svg)](https://github.com/Limen-Neural/nir-rs/actions/workflows/docker.yml)
[![crates.io](https://img.shields.io/crates/v/nir-rs.svg)](https://crates.io/crates/nir-rs)
[![docs.rs](https://docs.rs/nir-rs/badge.svg)](https://docs.rs/nir-rs)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Typed NIR graphs in Rust, with opt-in HDF5 `.nir` read/write that interoperates
with the official Python reference. The graph model has **no** system
dependencies; only the `hdf5` feature links native libhdf5.

NIR is to spiking neural networks what ONNX is to conventional nets (or GGUF to
LLMs): a framework-agnostic graph format so models can move between simulators
and hardware without being rewritten.

## Why nir-rs?

- Official NIR is primarily Python ([neuromorphs/NIR](https://github.com/neuromorphs/NIR))
- This crate is a pure-Rust graph model with the same wire types and HDF5 layout
- Suitable for embedded, server, and tooling pipelines that should not embed a Python runtime
- Opt-in HDF5 I/O and Serde for debug serialization — enable only what you need

## Upstream

- Spec / reference: [github.com/neuromorphs/NIR](https://github.com/neuromorphs/NIR)
- Primitives docs: [neuroir.org](https://neuroir.org/docs/)
- Paper: [Nature Communications (2024)](https://www.nature.com/articles/s41467-024-52259-9)
  (DOI [10.1038/s41467-024-52259-9](https://doi.org/10.1038/s41467-024-52259-9)) —
  [cite](#citing-nir)

**Wire compatibility:** HDF5 node `type` strings must match the Python IR
(`CubaLIF`, `Conv2d`, `SumPool2d`, …), not informal aliases
(`CurrLIF`, `Convolution`, …).

## Scope

**This crate provides:**

- The NIR graph model and standard node types
- Reading and writing `.nir` (HDF5) files
- Round-trip fidelity checks and structural validation
- An idiomatic Rust API (`NirGraph`, closed `NirNode` enum, tensors, errors)

**This crate does not provide:**

- SNN training or simulation
- Mapping graphs onto specific neuromorphic hardware
- Framework-specific importers/exporters (those belong in the tools that
  produce or consume NIR)

## Status

| Version | Focus |
|---------|--------|
| **0.4.x** (current) | Graph model, HDF5 I/O, Serde/debug DX, Docker image, crates.io |
| Earlier | Dual license, typed nodes, fixtures, CI hardening |

Release notes and the upstream compatibility matrix:

| Doc | Purpose |
|-----|---------|
| [CHANGELOG.md](CHANGELOG.md) | Keep a Changelog notes + 0.x versioning policy |
| [COMPATIBILITY.md](COMPATIBILITY.md) | Release ↔ upstream NIR, fidelity rules, features, MSRV |

Compatibility claims are **fixture-backed** (`tests/fixtures/`).

## Install

```toml
[dependencies]
nir-rs = "0.4.2"
```

HDF5 `.nir` I/O (needs a system libhdf5, or a static build — see [File I/O](#file-io)):

```toml
[dependencies]
nir-rs = { version = "0.4.2", features = ["hdf5"] }
```

Debug Serde (JSON / RON / etc.; not a wire standard):

```toml
[dependencies]
nir-rs = { version = "0.4.2", features = ["serde"] }
```

From git — pin a release tag (same tree as the matching crates.io release once
the tag exists):

```toml
nir-rs = { git = "https://github.com/Limen-Neural/nir-rs", tag = "v0.4.2" }
```

For unreleased work on the default branch:

```toml
nir-rs = { git = "https://github.com/Limen-Neural/nir-rs", branch = "main" }
```

## Quick start

```rust
use nir_rs::nodes::{Input, Output};
use nir_rs::{NirGraph, NirNode};

fn main() -> nir_rs::Result<()> {
    let mut g = NirGraph::new();
    g.insert_node(
        "input",
        NirNode::Input(Input {
            shape: vec![4],
            metadata: Default::default(),
        }),
    )?;
    g.insert_node(
        "output",
        NirNode::Output(Output {
            shape: vec![4],
            metadata: Default::default(),
        }),
    )?;
    g.add_edge("input", "output");
    g.validate_structure()?;
    Ok(())
}
```

## File I/O

`.nir` is the official NIR interchange format: an HDF5 container whose layout is
fixed by upstream. Files written here load in Python `nir.read`, and files
written by `nir.write` load here.

```rust
fn main() -> nir_rs::Result<()> {
    let graph = nir_rs::io::read("model.nir")?;
    for (name, node) in &graph.nodes {
        println!("{name}: {}", node.type_name());
    }
    nir_rs::io::write("copy.nir", &graph)?;
    Ok(())
}
```

I/O is behind the opt-in **`hdf5`** feature, which links native libhdf5.
Without that feature the crate has no system dependencies:

| Platform | System dependency |
|----------|-------------------|
| Debian / Ubuntu | `apt install libhdf5-dev` |
| Fedora | `dnf install hdf5-devel` |
| macOS | `brew install hdf5` |
| Anywhere | depend on `hdf5-metno = { version = "0.14", features = ["static", "zlib"] }` — Cargo feature unification applies it to this crate's copy. A dependency cannot enable `hdf5/static` via this crate's feature list alone; without `zlib` the vendored build has no gzip filter. |

Without the feature, `io::read` / `io::write` still exist and return
`NirError::Unimplemented`, so downstream code compiles either way.

Round-trip fidelity is **graph-level**, not byte-level: node names and types,
ordered edges, and parameter values are preserved; HDF5 group order and chunk
layout may differ from h5py. In-memory dtypes (`f32`, `f64`, `i64`, `bool`)
round-trip exactly; narrower on-disk integers widen to `i64` on read. Absent
optional fields (`v_reset`, `w_in`) use the same defaults as Python so graphs
match `nir.read` in memory.

### Example: load, inspect, save a LIF graph

```bash
cargo run --example load_inspect_lif --features hdf5
# optional paths:
cargo run --example load_inspect_lif --features hdf5 -- model.nir copy.nir
```

Default input is `tests/fixtures/lif_norse.nir`; default output is a
PID-qualified file in the system temp directory.

## Docker

Images ship a **Rust 1.98 + libhdf5** environment with the crate sources and the
`load_inspect_lif` example binary (not an SNN simulator).

```bash
docker pull ghcr.io/limen-neural/nir-rs:0.4.2
docker pull ghcr.io/limen-neural/nir-rs:latest

docker run --rm ghcr.io/limen-neural/nir-rs:latest rustc --version
docker run --rm ghcr.io/limen-neural/nir-rs:latest load_inspect_lif
```

The same tags may also appear on **Docker Hub**; prefer GHCR for a stable,
documented image path.

Local image:

```bash
docker build -t nir-rs:local .
```

## Debug serialization

The opt-in `serde` feature implements `Serialize` / `Deserialize` for the graph
model. It is independent of `hdf5`:

```toml
[dependencies]
nir-rs = { version = "0.4.2", features = ["serde"] }
serde_json = "1"
```

**JSON is debug/test output, not a NIR interchange standard.** Use HDF5 `.nir`
for Python and hardware tooling. JSON cannot represent NaN/infinities faithfully.

## Toolchain

**Rust 1.98.1** — `rust-toolchain.toml`, `package.rust-version`, and CI
(Linux / macOS / Windows) all pin the same version.

## Develop

```bash
cargo fmt --check
cargo test                 # graph model only — no libhdf5
cargo test --features serde
cargo test --all-features  # + HDF5 fixtures / round-trip
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features
```

Wire compatibility is checked against real Python-written `.nir` fixtures under
`tests/fixtures/` (BSD-3; see that directory's README). No Python interpreter is
required for default builds, tests, or CI.

API and fidelity details: [COMPATIBILITY.md](COMPATIBILITY.md),
[docs.rs/nir-rs](https://docs.rs/nir-rs).

## Citing NIR

If you use NIR in your work (including via this crate), please cite the
[Nature Communications paper](https://www.nature.com/articles/s41467-024-52259-9):

```bibtex
@article{NIR2024,
    title={Neuromorphic intermediate representation: A unified instruction set for interoperable brain-inspired computing},
    author={Pedersen, Jens E. and Abreu, Steven and Jobst, Matthias and Lenz, Gregor and Fra, Vittorio and Bauer, Felix Christian and Muir, Dylan Richard and Zhou, Peng and Vogginger, Bernhard and Heckel, Kade and Urgese, Gianvito and Shankar, Sadasivan and Stewart, Terrence C. and Sheik, Sadique and Eshraghian, Jason K.},
    rights={2024 The Author(s)},
    DOI={10.1038/s41467-024-52259-9},
    number={1},
    journal={Nature Communications},
    volume={15},
    year={2024},
    month=sep,
    pages={8122},
}
```

Machine-readable form: [CITATION.cff](CITATION.cff) (GitHub “Cite this repository”).

## Acknowledgements

NIR was originally conceived at the **Telluride Neuromorphic Workshop 2023** by
the authors below (alphabetical order), as listed by upstream
[neuromorphs/NIR](https://github.com/neuromorphs/NIR):

- Steven Abreu
- Felix Bauer
- Jason Eshraghian
- Matthias Jobst
- Gregor Lenz
- Jens Egholm Pedersen
- Sadique Sheik
- Peng Zhou

This crate is an independent pure-Rust implementation of that IR; it is not the
official Python reference package.

## License

Dual-licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or
  https://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or
  https://opensource.org/licenses/MIT)

at your option.
