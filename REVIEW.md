# Local review quality gate

These commands are the **human quality bar** beyond GitHub Actions.
Run them before claiming a PR is ready when the change touches `src/`,
`Cargo.toml`, public APIs, or CI.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test                 # default features: no libhdf5 needed
cargo test --all-features
cargo doc --no-deps --all-features
```

`--all-features` turns on `hdf5`, which links the native library. Install it
first, or build a hermetic copy:

```bash
sudo apt-get install -y libhdf5-dev   # Ubuntu/Debian; brew install hdf5 on macOS
# Vendored source; zlib carries the gzip filter. The slash forms enable
# features of the `hdf5` (hdf5-metno) dependency directly — this is cargo's
# documented dep/feature syntax, verified with
# `cargo metadata --features hdf5,hdf5/static,hdf5/zlib` on cargo 1.98.
cargo test --features hdf5,hdf5/static,hdf5/zlib
```

## Checklist

- [ ] No secrets or credentials
- [ ] Wire type names match upstream NIR (not marketing aliases)
- [ ] Dual license files present when packaging (`LICENSE-MIT`, `LICENSE-APACHE-2.0`)
- [ ] New public API has rustdoc
- [ ] CI workflow still targets `Main` and `main` if changed
- [ ] Graph model still builds and tests green **without** the `hdf5` feature
- [ ] No Python added to the repo, tests, or CI
- [ ] Read and write paths stay symmetric — a new wire field needs both sides
      plus a round-trip assertion
