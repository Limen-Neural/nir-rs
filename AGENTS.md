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
cargo test --features serde
cargo test --all-features
cargo doc --no-deps --all-features
```

Property tests (`proptest`, `tests/prop_*.rs`) run inside `cargo test`. Optional
nightly fuzz harnesses: see [TESTING.md](TESTING.md) (`cargo +nightly fuzz run …`).

`--all-features` enables `hdf5`, which links libhdf5: install `libhdf5-dev`
(Ubuntu) or `hdf5` (Homebrew) first.

**Toolchain:** CI and agents pin **Rust 1.97.1** (`rust-toolchain.toml`).
`package.rust-version` (**1.85.1**) is a cargo floor only — do **not** point the
OS matrix at an old rustc. Default/`serde` and `--all-features` tests run on
**Linux, macOS, and Windows** with `toolchain: "1.97.1"`.

### Package / semver CI (release gate)

Separate workflow: [`.github/workflows/package.yml`](.github/workflows/package.yml)
(`cargo package` + required-file checks + `cargo-semver-checks`).

- **Package job** validates the crates.io *artifact* boundary, not only the
  checkout (licenses, README, fixtures, default-feature tests from the pack).
- **Semver job** compares the public API to tag **`v0.4.0`** (git baseline until
  the crate is on crates.io — see #26). After the first publish, prefer a
  registry baseline (`baseline-version` / crates.io) and bump the floor when
  cutting releases.

**Escape hatch (deliberate breaks):** during `0.x`, intentional public API
breaks require a **minor** bump (e.g. `0.4.0` → `0.5.0`), not a silent patch.
Document the break in the PR/changelog, update `package.version`, and either:

1. temporarily point the semver job at the new baseline tag after merge, or
2. land the break in a PR that also updates the `baseline-rev` / version once
   the intentional change is accepted.

Do **not** disable the job to force a green PR. Consumer-only work (#15 / #16)
does not change crate semver.

Hermetic fallback (no system libhdf5) — may lag if vendored HDF5 and
`hdf5-metno-sys` disagree; prefer system libhdf5 in CI:

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
(`1.97.1` + `rustfmt` + `clippy`).

### Dev container / cloud agent images

Prefer a prebuilt image when the host supports it (Rust 1.97 + `libhdf5-dev`):

| Path | Consumer |
|------|----------|
| [`Dockerfile`](Dockerfile) → `ghcr.io/limen-neural/nir-rs` + Docker Hub | Published toolchain image (GHCR + Hub) |
| [`.github/workflows/docker.yml`](.github/workflows/docker.yml) | PR verify; `main`/tag dual-publish |
| [`.cursor/environment.json`](.cursor/environment.json) → [`.cursor/Dockerfile`](.cursor/Dockerfile) | Cursor cloud agents |
| [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) | VS Code / Cursor Desktop |
| [`scripts/agent-bootstrap.sh`](scripts/agent-bootstrap.sh) | cubic / Claude / other bare sandboxes |
| [`cubic.yaml`](cubic.yaml) | cubic review + fix instructions |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | GitHub Actions quality bar (`fmt` / `test` / `clippy` / `doc`) |
| [`.github/workflows/package.yml`](.github/workflows/package.yml) | Package artifact + public-API semver gate |

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
- Use Conventional Commits: `type(scope): description` (allowed types and scopes
  are listed under **Code style**)
- Prefer one issue (or one milestone slice) per PR
- Link PR to GitHub issue and Linear ID when known
- Default branch on GitHub is **`main`**

## References

- Upstream: https://github.com/neuromorphs/NIR
- Paper: https://www.nature.com/articles/s41467-024-52259-9
- Org epic: LIM-822 / https://github.com/Limen-Neural/nir-rs/issues/1
