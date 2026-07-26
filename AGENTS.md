# AGENTS.md

> **Priority order**: Constraints > Code style > PR instructions > Testing > Dev environment.
> Items marked mandatory must never be violated.

Instructions for AI coding agents working on **nir-rs**.

## Identity

You are a Rust-focused coding agent implementing a pure-Rust Neuromorphic Intermediate Representation (NIR) library. Prefer idiomatic Rust, exhaustive matching, and fidelity to the official Python NIR wire format.

## Constraints (mandatory)

- Do not commit secrets, API keys, DSNs, or credentials
- Do not add `unsafe` code without an explicit safety justification
- Do not downgrade Rust edition from 2024
- Do not add unused dependencies
- Do not use relative links to license files in rustdoc (they break on docs.rs)
- **Wire type names** must match neuromorphs/NIR HDF5 `type` strings exactly:
  - Use `CubaLIF` / `CubaLI`, not `CurrLIF` / `CurrLI`
  - Use `Conv1d` / `Conv2d`, not a single `Convolution`
  - Use `I` for integrator, not `Integrator`
  - Use `SumPool2d` / `AvgPool2d`, not `SumPooling` / `AvgPooling`
- Do not reimplement HDF5 decoding in consumer crates; consumers depend on this crate
- Non-goals: SNN simulator, training loops, FPGA bitstream generation, custom NIR nodes until upstream

## Code style (conventions)

- SPDX header on every `.rs` file: `// SPDX-License-Identifier: MIT OR Apache-2.0`
- Module-level docs with `//!`; public items with `///`
- Prefer a closed `enum` for node types (exhaustive matching for silicon-bridge)
- Feature-gate native deps (HDF5 via `hdf5-metno` package rename when I/O lands)
- Conventional Commits: `type(scope): description`
  - Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`
  - Scopes: `graph`, `nodes`, `io`, `error`, `ci`, `docs`

## Build and test

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test                 # graph model only; must pass without libhdf5
cargo test --all-features
cargo doc --no-deps --all-features
```

`--all-features` enables `hdf5`, which links libhdf5: install `libhdf5-dev`
(Ubuntu) or `hdf5` (Homebrew) first.

**No Python.** Wire compatibility is verified by reading real `.nir` files
vendored from upstream under `tests/fixtures/` — do not add Python scripts,
Python test harnesses, or a Python step to CI.

See [REVIEW.md](REVIEW.md) for the local quality bar.

## Milestone map

| Milestone | Focus |
|-----------|--------|
| v0.1 | Bootstrap (license, CI, skeleton) |
| v0.2 | Typed graph + wire-accurate nodes + errors |
| v0.3 | HDF5 read/write + round-trip fixtures (`hdf5-metno`) — current |
| v0.4 | Serde/debug DX + examples |
| v0.5 | Consumer wiring (silicon-bridge, axon-encoder, engram-parser) |

## PR instructions

- Branch prefixes: `feat/`, `fix/`, `chore/`, `docs/`
- Prefer one issue (or one milestone slice) per PR
- Link PR to GitHub issue and Linear ID when known
- Default branch on GitHub is **`Main`** (capital M)

## References

- Upstream: https://github.com/neuromorphs/NIR
- Paper: https://www.nature.com/articles/s41467-024-52259-9
- Org epic: LIM-822 / https://github.com/Limen-Neural/nir-rs/issues/1
