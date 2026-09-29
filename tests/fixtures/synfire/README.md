# Synfire registry -> NIR fixtures

This directory holds the first Synfire compatibility baseline for milestone
`v0.4.5`: FIVE pinned `.nir` (HDF5) artifacts acquired from the Synfire model
registry (see GitHub [#43](https://github.com/Limen-Neural/nir-rs/issues/43) /
LIM-1085). The bytes are vendored so `nir-rs` can prove it loads real registry
artifacts from pure-Rust tests. CI and the default test suite never contact the
registry or authenticate; acquisition is maintainer-only.

Machine-readable catalog: [`MANIFEST.toml`](MANIFEST.toml). Redistribution
notice: [`LICENSE-Synfire`](LICENSE-Synfire).

## Provenance

Every value below was observed by loading the committed bytes through
`nir_rs::io::read` + `validate_structure()`. It must match
[`MANIFEST.toml`](MANIFEST.toml) exactly.

| Model | Version | File | Retrieved | Embedded NIR version | License (basis) | Nodes | Edges | SHA-256 |
|-------|---------|------|-----------|----------------------|-----------------|-------|-------|---------|
| pabogdan/lifneuron | 1.0.0 | `lifneuron-1.0.0.nir` | 2026-09-29 | 0.1.1 | BSD-3-Clause (nir-card.json) | 4 | 3 | `0dd9143ef624892d4a6461f474d93653a337b3490a62e23ec9ec5d18fde9b7b6` |
| pabogdan/ifsynfire | 0.1.0 | `ifsynfire-0.1.0.nir` | 2026-09-29 | 1.0.7 | BSD-3-Clause (nir-card.json) | 5 | 5 | `62be4ef638af7ba5dba85f2254083ec98257c838d9dcd6c7c0a7dbe1a3c20d70` |
| pabogdan/nmnistcnn | 1.0.0 | `nmnistcnn-1.0.0.nir` | 2026-09-29 | 0.2.0 | BSD-3-Clause (nir-card.json) | 15 | 14 | `e2fa55bda7aab5a772485e1b690358bcb825b303eca7dc426e3973937fcb5bcb` |
| pabogdan/brailernn | 1.0.1 | `brailernn-1.0.1.nir` | 2026-09-29 | 0.2.0 | BSD-3-Clause (nir-card.json) | 7 | 7 | `14f59e0903d76437be3cb23d7e4be6b4e08ea476e482f7478dbb4592b29377d3` |
| jegp/swavelet | 1.0.0 | `swavelet-1.0.0.nir` | 2026-09-29 | 1.0.7 | BSD-3-Clause (nir-card.json) | 6 | 5 | `d453379b30cb86a05ed7f73587fed971753744c8a2bfe7fe22bd3e9802135f69` |

All five `nir-card.json` cards declare BSD-3-Clause, which permits
redistribution with attribution plus notice, so the bytes are committed. The
registry publishes no per-release checksum, so the locally computed SHA-256 of
the committed bytes is authoritative (`registry_checksum = "none"`).

## Credits and citation

The people and citations below are drawn verbatim from each model's own
`nir-card.json` (observed at acquisition, 2026-09-29) and mirror the `authors`
and `citation` keys in [`MANIFEST.toml`](MANIFEST.toml). Four of the five cards
carry the NIR-paper citation (DOI
[10.1038/s41467-024-52259-9](https://doi.org/10.1038/s41467-024-52259-9)); the
`ifsynfire` card carries no citation field.

| Model | Card authors | Citation / paper | Model links |
|-------|--------------|------------------|-------------|
| pabogdan/lifneuron | NIR Team | NIR paper, DOI 10.1038/s41467-024-52259-9 | source: github.com/neuromorphs/NIR (`paper/01_lif`) |
| pabogdan/ifsynfire | Petrut Bogdan | none (no citation field in card) | none (card declares no source repository) |
| pabogdan/nmnistcnn | NIR Team | NIR paper, DOI 10.1038/s41467-024-52259-9 | source: github.com/neuromorphs/NIR (`paper/02_cnn`, Sinabs export) |
| pabogdan/brailernn | NIR Team | NIR paper, DOI 10.1038/s41467-024-52259-9 | source: github.com/neuromorphs/NIR (`paper/03_rnn`, snnTorch export); Braille dataset paper: <https://www.frontiersin.org/articles/10.3389/fnins.2022.951164/full> |
| jegp/swavelet | Jens E. Pedersen, Tony Lindeberg, Peter Gerstoft | NIR paper, DOI 10.1038/s41467-024-52259-9 | repo: <https://github.com/jegp/swavelet>; paper: <https://arxiv.org/abs/2605.09770> |

The full NIR-paper citation recorded for the four NIR-cited models is:

> Pedersen JE, Abreu S, Jobst M, et al. Neuromorphic Intermediate
> Representation: A Unified Instruction Set for Interoperable Brain-Inspired
> Computing. Nature Communications. DOI: 10.1038/s41467-024-52259-9.

Redistribution basis and copyright holders are recorded separately in
[`LICENSE-Synfire`](LICENSE-Synfire).

## Node-type inventory

| Model | Node-type inventory | Nested graphs | Key topology feature |
|-------|---------------------|---------------|----------------------|
| pabogdan/lifneuron | `Affine:1`, `Input:1`, `LIF:1`, `Output:1` | none | Minimal feed-forward `Input -> Affine -> LIF -> Output` |
| pabogdan/ifsynfire | `IF:1`, `Input:1`, `Linear:2`, `Output:1` | none | Single `IF` population of shape [50] with a recurrent `Linear` loop; NOT a chain of separate IF nodes |
| pabogdan/nmnistcnn | `Affine:2`, `Conv2d:3`, `Flatten:1`, `IF:5`, `Input:1`, `Output:1`, `SumPool2d:2` | none | Convolutional event-vision graph using `SumPool2d` (not AvgPool2d) with explicit-padding Conv2d and `Flatten` |
| pabogdan/brailernn | `CubaLIF:2`, `Input:1`, `Linear:3`, `Output:1` | none | Recurrent CubaLIF loop (`lif1.lif -> lif1.w_rec -> lif1.lif`) with dotted node names |
| jegp/swavelet | `Affine:2`, `Input:1`, `LI:1`, `LIF:1`, `Output:1` | none | `Input -> fanout (Affine) -> li_stage_0 (LI) -> connectivity (Affine) -> lif (LIF) -> Output` |

None of the five contains a nested `NIRGraph`.

## Same bytes, different registry provenance

Confirmed by SHA-256, three of the five Synfire artifacts are byte-identical to
existing neuromorphs/NIR paper fixtures already vendored in the parent
directory:

- `lifneuron-1.0.0.nir` == [`../lif_norse.nir`](../lif_norse.nir) (Norse, `paper/01_lif`)
- `nmnistcnn-1.0.0.nir` == [`../cnn_sinabs.nir`](../cnn_sinabs.nir) (Sinabs, `paper/02_cnn`, roughly 267 KB)
- `brailernn-1.0.1.nir` == [`../braille_noDelay_noBias_subtract.nir`](../braille_noDelay_noBias_subtract.nir) (snnTorch, `paper/03_rnn`)

For those three, this corpus proves independent-registry provenance rather than
new topology coverage: the same bytes reached us through the Synfire registry
instead of the paper repository. `ifsynfire-0.1.0.nir` and `swavelet-1.0.0.nir`
have no existing counterpart and add genuinely new coverage. All five are still
vendored under their stable `synfire/` names and tested independently.

## Maintainer retrieval (maintainer-only)

Acquisition is acquisition-only and never part of CI or the default test suite.
A maintainer refreshing the corpus uses the Synfire CLI to pull each pinned
version by `synfire pull <org>/<model>:<version>` into a scratch directory
OUTSIDE the repository, then copies verified bytes in under the stable names.
Public access is unauthenticated (`auth_required = false`); keyring warnings
from the CLI are harmless.

The FIVE exact commands for this pinned baseline:

```bash
synfire pull pabogdan/lifneuron:1.0.0
synfire pull pabogdan/ifsynfire:0.1.0
synfire pull pabogdan/nmnistcnn:1.0.0
synfire pull pabogdan/brailernn:1.0.1
synfire pull jegp/swavelet:1.0.0
```

Retrieval workflow:

1. Pull into a scratch directory outside the repository (never pull directly
   into `tests/fixtures/synfire/`).
2. Compute `sha256sum <file>.nir` and confirm it matches the SHA-256 recorded
   in [`MANIFEST.toml`](MANIFEST.toml) before copying the bytes in.
3. Load each file through `nir_rs::io::read` + `validate_structure()` and record
   the node count, edge count, and sorted node-type inventory.
4. Record the retrieval date (UTC) as `retrieved` / `corpus_as_of`, and record
   the observed `synfire --version` string.
5. Same-pinned-version rule: a re-pull of the same `<org>/<model>:<version>`
   must reproduce the same SHA-256. If it differs, treat it as a registry
   regression and investigate rather than silently overwriting the fixture.
6. CI and default tests must never contact the registry or authenticate. Only
   the committed bytes and their recorded SHA-256 are used by tests.

Note the CLI/package version discrepancy: the published PyPI package is `0.0.2`
while the observed `synfire --version` string is `synfire version 0.0.1` (the
CLI's internal `_version.py` lags the published package). Both are recorded in
[`MANIFEST.toml`](MANIFEST.toml).

## Review checklist

For a fixture PR touching this directory, a reviewer should confirm:

- [ ] Each `.nir` file's `sha256sum` matches [`MANIFEST.toml`](MANIFEST.toml) and the provenance table above.
- [ ] `MANIFEST.toml` keeps each key and each list on ONE line (the hand-written single-line parser depends on it).
- [ ] Every wire type name in the inventory matches NIR exactly (for example `Affine`, `Conv2d`, `SumPool2d`, `CubaLIF`, `LI`, `LIF`, `IF`, `Linear`, `Input`, `Output`).
- [ ] `node_count` and `edge_count` were observed from the actual file, not assumed, and node-type counts sum to `node_count`.
- [ ] Versions are the exact pinned versions; no substitutions and no added models.
- [ ] Redistribution basis is documented and [`LICENSE-Synfire`](LICENSE-Synfire) carries the correct copyright holders.
- [ ] No default-CI code path contacts the registry or authenticates.

## Compatibility outcomes

Status classes (four-way scheme):

- `supported`: the file loads through `nir_rs::io::read` and passes `validate_structure()` with no `NirError`.
- `unsupported-valid`: the file is well-formed HDF5/NIR but uses a NIR construct `nir-rs` does not yet represent.
- `malformed`: the file is structurally invalid or fails to parse (a `NirError` on read).
- `inaccessible`: the artifact could not be redistributed or retrieved, so no bytes are vendored.

| Model | Status | Load result | Follow-up |
|-------|--------|-------------|-----------|
| pabogdan/lifneuron | supported | loads and validates, no NirError | no follow-up required |
| pabogdan/ifsynfire | supported | loads and validates, no NirError | no follow-up required |
| pabogdan/nmnistcnn | supported | loads and validates, no NirError | no follow-up required |
| pabogdan/brailernn | supported | loads and validates, no NirError | no follow-up required |
| jegp/swavelet | supported | loads and validates, no NirError | no follow-up required |

All five are `supported`, so no non-supported model exists and no follow-up
issue is required. The `status` key is kept in the manifest so a later re-pull
can record a regression against a different class.

## Relationships

- Parent: [LIM-822](https://linear.app/rpd-34/issue/LIM-822) (fixture compatibility program).
- Twin: GitHub [#43](https://github.com/Limen-Neural/nir-rs/issues/43) / LIM-1085 (this Synfire registry compatibility work).
- Related real-world corpus: GitHub [#29](https://github.com/Limen-Neural/nir-rs/issues/29), vendored in [`../`](../) (neuromorphs/NIR paper artifacts).
- Related Hugging Face corpus: GitHub [#44](https://github.com/Limen-Neural/nir-rs/issues/44), vendored in [`../huggingface/`](../huggingface/).
- Downstream: [Limen-Neural/neuromod#125](https://github.com/Limen-Neural/neuromod/issues/125) is the neuromod post-load handoff pattern, kept as a separate follow-up and out of scope here.

## Tracker conflict

The CodeRabbit execution plan targeted milestone `v0.4.4` and three models. The
issue [#43](https://github.com/Limen-Neural/nir-rs/issues/43) body and the
`v0.4.5` milestone (Linear LIM-1085, "v0.4.5 - Fixture compatibility and read
hardening") are authoritative, so this baseline implements FIVE models at
`v0.4.5`. Maintainers should reconcile the `v0.4.4`-vs-`v0.4.5` and
three-vs-five discrepancy in the tracker.

## Scope

Out of scope for this work (from issue [#43](https://github.com/Limen-Neural/nir-rs/issues/43)):

- No supporting every registry model.
- No crawling the registry.
- No live registry tests in default CI.
- No model execution.
- No neuromod simulation or handoff.
- No axon-encoder exporter.
- No publishing Limen models to Synfire.
- No Spikenaut Q8.8 conversion.
- No implementing missing NIR operators to force a pass.
