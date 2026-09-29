# Synfire registry → NIR fixtures

`.nir` (HDF5) files pulled from the public [Synfire](https://synfire.dev)
registry and vendored byte-identically. They prove `nir-rs` can consume NIR
artifacts published through an **independent registry**, not only the
neuromorphs/NIR paper corpus or locally converted files.

CI never talks to Synfire. These files are vendored;
`tests/synfire_fixture_checksums.rs` checks their SHA-256 against
[`MANIFEST.toml`](MANIFEST.toml), and `tests/hdf5_synfire.rs` (feature `hdf5`)
loads each one and asserts the pinned topology and wire-type inventory.

This is the pinned corpus for GitHub
[#43](https://github.com/Limen-Neural/nir-rs/issues/43) / LIM-1085: five
representative releases. It is intentionally finite — do **not** add further
registry models here without a separate corpus-expansion ticket.

## Pinned corpus (as of 2026-09-29)

| File | Synfire release | Purpose | `/version` | Nodes / Edges | Wire types |
|------|-----------------|---------|-----------|---------------|------------|
| `lifneuron_1.0.0.nir` | `pabogdan/lifneuron:1.0.0` | Minimal LIF graph (Norse) | 0.1.1 | 4 / 3 | Input, Affine, LIF, Output |
| `ifsynfire_0.1.0.nir` | `pabogdan/ifsynfire:0.1.0` | IF synfire-chain, 50 neurons in groups of 10 | 1.0.7 | 5 / 5 | Input, Linear, IF, Output; recurrent self-loop |
| `nmnistcnn_1.0.0.nir` | `pabogdan/nmnistcnn:1.0.0` | N-MNIST spiking CNN (Sinabs) | 0.2.0 | 15 / 14 | Input, Conv2d, IF, SumPool2d, Flatten, Affine, Output |
| `brailernn_1.0.1.nir` | `pabogdan/brailernn:1.0.1` | Recurrent Braille RNN (snnTorch) | 0.2.0 | 7 / 7 | Input, Linear, CubaLIF, Output; `w_in` recurrence |
| `swavelet_1.0.0.nir` | `jegp/swavelet:1.0.0` | 16-ch spiking DoE wavelet encoder | 1.0.7 | 6 / 5 | Input, Affine, LI, LIF, Output |

Three releases redistribute neuromorphs/NIR paper artifacts byte-identically
(`lifneuron` = `lif_norse.nir`, `nmnistcnn` = `cnn_sinabs.nir`,
`brailernn` = `braille_noDelay_noBias_subtract.nir`); the Synfire copies are
kept anyway so registry provenance, not file novelty, is what the baseline
covers.

## Maintainer-only refresh procedure

Normal tests and CI consume **committed fixtures only** — no Synfire network
access, authentication, or live download. Refresh is a deliberate maintainer
step:

```bash
pip install synfire
synfire pull pabogdan/lifneuron:1.0.0
synfire pull pabogdan/ifsynfire:0.1.0
synfire pull pabogdan/nmnistcnn:1.0.0
synfire pull pabogdan/brailernn:1.0.1
synfire pull jegp/swavelet:1.0.0
```

Each pull writes `models/<org>/<model>/<version>/model.nir` plus a
`nir-card.json` (license, source repo, upstream path/SHA-256). To refresh:

1. Pull the **exact pinned versions** above (never substitute newer releases).
2. Copy each `model.nir` over the file named in [`MANIFEST.toml`](MANIFEST.toml).
3. Recompute `sha256sum` and update `MANIFEST.toml`, the table above, and
   `tests/synfire_fixture_checksums.rs` expectations change only through the
   manifest — the test reads digests from it.
4. If a refresh changes node/edge counts, versions, or wire-type inventories,
   update `MANIFEST.toml` **and** the structural assertions in
   `tests/hdf5_synfire.rs`; unexplained drift means the pull was not the pinned
   artifact.
5. Pulling requires no login. Publishing does (`synfire login` /
   `SYNFIRE_TOKEN`) and is out of scope.

## License of the vendored files

All five registry cards declare **BSD-3-Clause**. Four artifacts originate in
the NIR paper corpus or the registry itself; `swavelet` is published by
Jens E. Pedersen et al. from
[`jegp/swavelet`](https://github.com/jegp/swavelet) (upstream path
`examples/spiking_doe_16ch.nir`). [`LICENSE-BSD-3-Clause`](LICENSE-BSD-3-Clause)
reproduces the BSD-3 text; it governs the `.nir` files here, not the rest of
`nir-rs` (MIT OR Apache-2.0).

These files are **inputs to tests only** — not part of the published library
API; nothing in `src/` depends on them.
