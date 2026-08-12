# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for the **0.x** series as described under [Versioning](#versioning) below.

## [Unreleased]

### Added

- Multi-OS CI: default and HDF5 tests on Linux, macOS, and Windows (#27, #32).
- Package workflow: `cargo package` artifact checks + public-API semver gate vs
  git tag `v0.4.0` (#28, #33).
- Compatibility matrix and changelog policy (this document) (#31).
- Expanded real-world fixture corpus: Rockpool LIF + noBias braille variants,
  `MANIFEST.toml`, coverage checklist (#29).

## [0.4.0] - 2026-08-06

First **git-tagged** library release intended as the consumer baseline
(`tag = "v0.4.0"`). **Not yet published to crates.io** (see #26).

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
