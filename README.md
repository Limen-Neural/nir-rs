# nir-rs

**Pure-Rust implementation of the Neuromorphic Intermediate Representation (NIR)**

[![CI](https://github.com/Limen-Neural/nir-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/nir-rs/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

> Pure-Rust NIR graph model (typed nodes, edges, validation), plus opt-in HDF5 `.nir` read/write that interoperates with the Python reference implementation. The graph model has no system dependencies; the `hdf5` feature is the one part that links native libhdf5.

NIR is to SNNs what ONNX is to conventional neural networks (or GGUF to LLMs): a framework-agnostic graph format that lets models move between simulators and hardware without being rewritten.

## Why nir-rs?

- Official NIR is primarily Python-based ([neuromorphs/NIR](https://github.com/neuromorphs/NIR))
- No mature shared Rust IR crate for the Limen stack
- Enables pure-Rust, embedded, and high-performance pipelines
- Native integration with the rest of Limen Neural (`axon-encoder`, `silicon-bridge`, `neuromod`, …)

## Upstream

- Spec / reference: [github.com/neuromorphs/NIR](https://github.com/neuromorphs/NIR)
- Primitives docs: [neuroir.org](https://neuroir.org/docs/)
- Paper: [Nature Communications (2024)](https://www.nature.com/articles/s41467-024-52259-9) (DOI [10.1038/s41467-024-52259-9](https://doi.org/10.1038/s41467-024-52259-9))

**Wire compatibility:** HDF5 node `type` strings must match the Python IR (`CubaLIF`, `Conv2d`, `SumPool2d`, …), not informal aliases (`CurrLIF`, `Convolution`, …).

## Scope

This crate **owns**:

- The NIR graph model and standard node types
- Reading and writing `.nir` (HDF5) files (v0.3+)
- Round-trip fidelity and basic validation
- A clean, idiomatic Rust API

This crate does **not** own:

- Training or simulation of SNNs
- Mapping to specific hardware (that lives in `silicon-bridge`)
- Framework-specific converters (those live in producing/consuming crates)

## Status / roadmap

| Milestone | Focus | Status |
|-----------|--------|--------|
| **v0.1** | Dual license, module skeleton, CI, agent docs | Done |
| **v0.2** | Typed graph, wire-accurate nodes, structured errors | Done |
| **v0.3** | HDF5 read/write via `hdf5-metno`, fixtures, round-trip | Done |
| **v0.4** | Serde/debug DX, examples | **This release** |
| **v0.5** | Wire consumers (silicon-bridge, axon-encoder, engram-parser) | Planned |

Tracking: [GitHub milestones](https://github.com/Limen-Neural/nir-rs/milestones) · [LIM-822](https://linear.app/rpd-34/issue/LIM-822)

## Quick start

Not published to crates.io yet. Use a git or path dependency:

```toml
[dependencies]
nir-rs = { git = "https://github.com/Limen-Neural/nir-rs", branch = "main" }
```

To also get HDF5 `.nir` I/O, enable the `hdf5` feature (see [File I/O](#file-io)
for the system dependency it brings):

```toml
[dependencies]
nir-rs = { git = "https://github.com/Limen-Neural/nir-rs", branch = "main", features = ["hdf5"] }
```

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

I/O is behind the opt-in **`hdf5`** feature, which links the native libhdf5
library. Without this feature, the crate requires no system dependencies:

| Platform | System dependency |
|----------|-------------------|
| Debian / Ubuntu | `apt install libhdf5-dev` |
| Fedora | `dnf install hdf5-devel` |
| macOS | `brew install hdf5` |
| Anywhere | depend on `hdf5-metno = { version = "0.14", features = ["static", "zlib"] }` directly — Cargo's feature unification applies it to this crate's copy. A dependency's feature list cannot name `hdf5/static`, and without `zlib` the vendored build has no gzip filter. |

Without the feature, `io::read` / `io::write` still exist and return
`NirError::Unimplemented`, so downstream code compiles either way.

Round-trip fidelity is graph-level, not byte-level: node names and types,
ordered edges, and exact parameter values are preserved, while HDF5 details
such as group ordering and chunk layout may differ from h5py. In-memory dtypes
(`f32`, `f64`, `i64`, `bool`) round-trip exactly; narrower on-disk integer
types are widened to `i64` on read. Absent optional fields (`v_reset`, `w_in`)
are filled with the same defaults Python uses, so a graph read here matches
what `nir.read` produces in memory.

### Load, inspect, and save a LIF graph

The public example loads the vendored LIF fixture, prints its graph structure
and a preview of every `LIF` parameter tensor (first eight elements, with a
total count when longer), writes a copy, then reads it back and checks graph
equality on finite values:

```bash
cargo run --example load_inspect_lif --features hdf5
```

Pass optional input and output paths to use your own model:

```bash
cargo run --example load_inspect_lif --features hdf5 -- model.nir copy.nir
```

When omitted, the input is `tests/fixtures/lif_norse.nir` and the output is a
PID-qualified file in the system temporary directory.

### Develop

```bash
cargo fmt --check
cargo test                 # graph model only, no libhdf5 required
cargo test --all-features  # + HDF5 I/O, fixtures and round-trip
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features
```

Wire compatibility is checked against real `.nir` files written by the Python
implementation and vendored under `tests/fixtures/` (BSD-3, see the README
there). Nothing in the build, tests, or CI needs a Python interpreter.

See [REVIEW.md](REVIEW.md) and [AGENTS.md](AGENTS.md).

## License

This project is dual-licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or https://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.
