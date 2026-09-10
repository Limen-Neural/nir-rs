# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for the **0.x** series as described under [Versioning](#versioning) below.

## [Unreleased]

### Changed

- Toolchain pin raised **1.97.1 → 1.98.1** in lockstep (`rust-toolchain.toml`,
  `package.rust-version`, CI, `rust:1.98-bookworm` images).
- Package `exclude` list: drop git-tracked Docker/VCS/agent paths that were
  still packing (`.dockerignore`, `.gitignore`, `Dockerfile`,
  `rust-toolchain.toml`, `TESTING.md`). Existing agent/CI excludes kept;
  licenses, README, CHANGELOG, CITATION, COMPATIBILITY, sources, examples, and
  fixtures still ship.

### Documentation

- Cite the NIR Nature Communications paper (BibTeX + `CITATION.cff`) and credit
  Telluride 2023 acknowledgements from upstream neuromorphs/NIR.
- Align toolchain docs with the **1.98.1** pin.

## [0.4.2] - 2026-08-13

Docs-only patch so crates.io serves a consumer-facing README (0.4.1 is
immutable on the registry).

### Changed

- `package.rust-version` raised **1.85.1 → 1.97.1** to match the supported CI /
  `rust-toolchain.toml` pin (stop advertising an untested older floor).

### Documentation

- Rewrite README for public consumers: drop internal sibling-crate roadmap,
  Linear tracking, and CI-only Hub variable language; keep crates.io/docs.rs
  install path, GHCR pulls, and scope clear without org-private context.
- Align crate rustdoc non-goals and a few comments with the same public wording.
- Bump advertised install / Docker tags to **0.4.2**.
- Toolchain docs: single **1.97.1** pin (no separate MSRV floor story).

## [0.4.1] - 2026-08-12

First **crates.io** release of the 0.4 line (package name `nir-rs`).

### Added

- Multi-OS CI: default and HDF5 tests on Linux, macOS, and Windows (#27, #32).
- Package workflow: `cargo package` artifact checks + public-API semver gate vs
  git tag `v0.4.0` (#28, #33).
- Compatibility matrix and changelog policy (#31).
- Expanded real-world fixture corpus: Rockpool LIF + noBias braille variants,
  `MANIFEST.toml`, coverage checklist (#29).
- Property tests (`proptest`) for tensor/graph/wire invariants and HDF5
  write→read; optional `cargo-fuzz` harnesses under `fuzz/` (#30).
- Docker image + dual publish to **GHCR** (`ghcr.io/limen-neural/nir-rs`) and
  **Docker Hub** (`$DOCKER_USER/nir-rs`); PR verify / main+tag push (#26).

### Documentation

- README crates.io / docs.rs badges and dependency snippets (#26).
- README Docker (GHCR + Hub) install/run notes (#26).
- [TESTING.md](TESTING.md) for property tests and local fuzz invocation.

## [0.4.0] - 2026-08-06

First **git-tagged** library release (`tag = "v0.4.0"`). Superseded as the
crates.io baseline by **0.4.1**.

### Added

- Typed NIR graph model and wire-accurate node types.
- Opt-in HDF5 `.nir` read/write (`hdf5` feature, `hdf5-metno`).
- Opt-in `serde` feature for graph/debug export.
- Vendored real-world fixtures under `tests/fixtures/` for wire compatibility.
- Example `load_inspect_lif` (requires `hdf5`).

### Toolchain

- `package.rust-version = "1.85.1"` (cargo floor).
- Dev / CI pin **Rust 1.97.1**.

## Versioning

| Change class | Version impact (0.x) |
|--------------|----------------------|
| Bug fix, docs, CI, non-public refactors | Patch (`0.4.y`) |
| Additive public API | Prefer patch if compatible; minor if the surface is large |
| Breaking public API | **Minor** (`0.5.0`), never a silent patch |
| Org-only consumer wiring | Does not change crate semver |

Deliberate breaks must update `package.version`, changelog, and the package
workflow baseline (`baseline-rev` / crates.io version) once accepted. Do not
disable the semver job to force green CI.

## Compatibility

See [COMPATIBILITY.md](COMPATIBILITY.md) for the release ↔ upstream NIR matrix,
fidelity semantics, and feature availability.
