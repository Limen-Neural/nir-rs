# Testing

Quality bar for contributors and agents. Also see [REVIEW.md](REVIEW.md) and
[AGENTS.md](AGENTS.md).

## Default CI matrix

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --features serde
cargo test --all-features   # needs libhdf5
cargo doc --no-deps --all-features
```

Deterministic property tests (`tests/prop_*.rs`, powered by **proptest**) run as
part of `cargo test` / `cargo test --features serde` / `--all-features`. They
use a fixed case budget suitable for CI.

## Property tests

| File | Feature | Focus |
|------|---------|--------|
| `tests/prop_invariants.rs` | default | Tensor shape/data, graph insert/validate, link-name preflight |
| `tests/prop_invariants.rs` | `serde` | Finite tensor + simple graph JSON round-trip |
| `tests/prop_hdf5.rs` | `hdf5` | Write → read graph fidelity, illegal name preflight (no clobber) |

Re-run with more cases locally:

```bash
PROPTEST_CASES=1024 cargo test --test prop_invariants
PROPTEST_CASES=256 cargo test --features hdf5 --test prop_hdf5
```

## Fuzz harnesses (`cargo-fuzz`)

Tooling: **libFuzzer** via [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz)
(nightly). Targets live under [`fuzz/`](fuzz/) and do **not** require network
access or system libhdf5.

| Target | What it hammers |
|--------|-----------------|
| `tensor_new` | `Tensor::new` shape × length / overflow |
| `graph_structure` | insert / edges / `validate_structure` panic freedom |
| `link_names` | `check_link_name` / `check_hdf5_string` |

### Local invocation

```bash
# once
cargo install cargo-fuzz
rustup toolchain install nightly

# run (example: 60 seconds)
cargo +nightly fuzz run tensor_new -- -max_total_time=60
cargo +nightly fuzz run graph_structure -- -max_total_time=60
cargo +nightly fuzz run link_names -- -max_total_time=60
```

Corpus artifacts (auto-created by libFuzzer):

```text
fuzz/corpus/<target>/
fuzz/artifacts/<target>/   # crashing inputs if any
```

**Policy:** any confirmed panic or invariant break found by fuzzing must be
turned into a minimal deterministic regression test under `tests/` or a unit
test next to the code, then re-run the harness to confirm the fix.

Fuzz jobs are **not** part of the default GitHub Actions matrix (nondeterministic
runtime, nightly). They are a local / optional hardening path for release work.

## Fixture interoperability

Known-good Python-written `.nir` files: [`tests/fixtures/`](tests/fixtures/).
See that directory’s `README.md` and `MANIFEST.toml`. Hostile / malformed HDF5
layouts are generated at runtime in `tests/hdf5_untrusted.rs` — keep those
separate from the interoperability corpus.
