# nir-rs

**Pure-Rust implementation of the Neuromorphic Intermediate Representation (NIR)**

[![CI](https://github.com/Limen-Neural/nir-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/nir-rs/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

> Pure-Rust library for the NIR graph model (typed nodes, edges, validation). HDF5 `.nir` I/O lands in v0.3.

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
| **v0.2** | Typed graph, wire-accurate nodes, structured errors | **This release** |
| **v0.3** | HDF5 read/write via `hdf5-metno`, fixtures, round-trip | Planned |
| **v0.4** | Serde/debug DX, examples | Planned |
| **v0.5** | Wire consumers (silicon-bridge, axon-encoder, engram-parser) | Planned |

Tracking: [GitHub milestones](https://github.com/Limen-Neural/nir-rs/milestones) · [LIM-822](https://linear.app/rpd-34/issue/LIM-822)

## Quick start

Not published to crates.io yet. Use a git or path dependency:

```toml
[dependencies]
nir-rs = { git = "https://github.com/Limen-Neural/nir-rs", branch = "Main" }
```

```rust
use nir_rs::nodes::{Input, Output};
use nir_rs::{NirGraph, NirNode};

fn main() -> nir_rs::Result<()> {
    let mut g = NirGraph::new();
    g.insert_node("input", NirNode::Input(Input { shape: vec![4] }))?;
    g.insert_node("output", NirNode::Output(Output { shape: vec![4] }))?;
    g.add_edge("input", "output");
    g.validate_structure()?;
    // HDF5 I/O arrives in v0.3: nir_rs::io::read("model.nir")
    Ok(())
}
```

### Develop

```bash
cargo fmt --check
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features
```

See [REVIEW.md](REVIEW.md) and [AGENTS.md](AGENTS.md).

## License

This project is dual-licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or https://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.
