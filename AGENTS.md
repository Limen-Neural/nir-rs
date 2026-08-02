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

Hermetic fallback (no system libhdf5):

```bash
cargo clippy --all-targets --features hdf5,hdf5/static,hdf5/zlib -- -D warnings
cargo test --features hdf5,hdf5/static,hdf5/zlib
cargo doc --no-deps --features hdf5,hdf5/static,hdf5/zlib
```

### Toolchain for PR / cloud agents (mandatory)

If `cargo` is **not** on `PATH` (common in bare PR-fix sandboxes), install it
**before** editing or claiming a fix:

```bash
bash scripts/agent-bootstrap.sh
# then in the same shell:
source "${CARGO_HOME:-$HOME/.cargo}/env"
```

Do **not** skip verification with “cargo not available”. Bootstrap, then run
the quality bar above. Never post a fix commit without at least:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --features hdf5,hdf5/static,hdf5/zlib -- -D warnings
cargo test --features hdf5,hdf5/static,hdf5/zlib
```

Configured channel/components: [`rust-toolchain.toml`](rust-toolchain.toml)
(`stable` + `rustfmt` + `clippy`; `stable` tracks the latest stable release).

### Dev container / cloud agent images

Prefer a prebuilt image when the host supports it (Rust 1.97 + `libhdf5-dev`):

| Path | Consumer |
|------|----------|
| [`.cursor/environment.json`](.cursor/environment.json) → [`.cursor/Dockerfile`](.cursor/Dockerfile) | Cursor cloud agents |
| [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) | VS Code / Cursor Desktop |
| [`scripts/agent-bootstrap.sh`](scripts/agent-bootstrap.sh) | cubic / Claude / other bare sandboxes |
| [`cubic.yaml`](cubic.yaml) | cubic review + fix instructions |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | GitHub Actions quality bar (`fmt` / `test` / `clippy` / `doc`) |
| [`.github/workflows/opencode.yml`](.github/workflows/opencode.yml) | OpenCode on `/opencode` or `/oc` (`opencode-go/gpt-5.6-luna`; `OPENCODE_API_KEY`) |

**No Python.** Wire compatibility is verified by reading real `.nir` files
vendored from upstream under `tests/fixtures/` — do not add Python scripts,
Python test harnesses, or a Python step to CI.

See [REVIEW.md](REVIEW.md) for the local quality bar.

## Milestone map

| Milestone | Focus |
|-----------|--------|
| v0.1 | Bootstrap (license, CI, skeleton) |
| v0.2 | Typed graph + wire-accurate nodes + errors |
| v0.3 | HDF5 read/write + round-trip fixtures (`hdf5-metno`) |
| v0.4 | Serde/debug DX + examples — current |
| v0.5 | Consumer wiring (silicon-bridge, axon-encoder, engram-parser) |

## PR instructions

- Branch prefixes: `feat/`, `fix/`, `chore/`, `docs/`
- Prefer one issue (or one milestone slice) per PR
- Link PR to GitHub issue and Linear ID when known
- Default branch on GitHub is **`main`**

## References

- Upstream: https://github.com/neuromorphs/NIR
- Paper: https://www.nature.com/articles/s41467-024-52259-9
- Org epic: LIM-822 / https://github.com/Limen-Neural/nir-rs/issues/1
