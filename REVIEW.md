# Local review quality gate

These commands are the **human quality bar** beyond GitHub Actions.
Run them before claiming a PR is ready when the change touches `src/`,
`Cargo.toml`, public APIs, or CI.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo doc --no-deps --all-features
```

## Checklist

- [ ] No secrets or credentials
- [ ] Wire type names match upstream NIR (not marketing aliases)
- [ ] Dual license files present when packaging (`LICENSE-MIT`, `LICENSE-APACHE-2.0`)
- [ ] New public API has rustdoc
- [ ] CI workflow still targets `Main` and `main` if changed

## Optional (when HDF5 lands)

```bash
# system package or static feature as documented in AGENTS.md
cargo test --features hdf5
```
