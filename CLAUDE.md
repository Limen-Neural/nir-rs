# CLAUDE.md

@AGENTS.md

Companion entrypoint for Claude Code and similar agents. [AGENTS.md](AGENTS.md)
is authoritative for repository constraints, code style, PR scope, and the full
quality bar. Read it before editing code or workflows.

## Release-sensitive commands

- Use Rust **1.98.1**. Keep [`rust-toolchain.toml`](rust-toolchain.toml),
  `package.rust-version`, CI, and development images in lockstep.
- Preserve the Rust 2024 edition and a default build free of native/system
  library dependencies. The `hdf5` feature stays opt-in, with system and static
  test paths maintained.
- For relevant Rust or CI work, run `cargo fmt --check`.
- Run `cargo test --locked`.
- Run `cargo test --locked --features serde`.
- Run `cargo clippy --all-targets --all-features -- -D warnings`.
- Run `cargo test --all-features`.
- Run `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`.
  See [REVIEW.md](REVIEW.md) for the local checklist.
- Package work runs with `--locked`. The semver gate compares the candidate with
  the latest published crates.io API, currently `0.4.3`, for default and Serde
  feature configurations.
- `0.4.4` is a release candidate until it is published. Keep public installation
  snippets on published `0.4.3`; do not publish, tag, merge, or use
  `--allow-dirty` without explicit authorization.

## Coverage integrations

[`.github/coverage-integrations.md`](.github/coverage-integrations.md) records
how Codecov organization-token and Codacy coverage uploads are enabled. Do not add provider
tokens to repository files, logs, or pull-request text.
