// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guards against `.nir` files that try to reach outside their own container.

#![cfg(feature = "hdf5")]

mod common;
use common::{assert_err, write_then};
use nir_rs::NirError;
use nir_rs::io::ReadOptions;
use tempfile::TempDir;

//
// HDF5 resolves external links, external raw storage and virtual-dataset
// sources against the host filesystem, which is how a crafted model file turns
// a reader into an arbitrary-file-read primitive (the Keras advisories
// GHSA-3m4q-jmj6-r34q / CVE-2026-12480 are exactly this). `read` rejects all
// three. The cases below construct each one so the guard is actually executed
// rather than merely present.

/// Build a well-formed file, then let `mutate` add something hostile to it.
fn tampered(dir: &TempDir, name: &str, mutate: impl FnOnce(&hdf5::File)) -> std::path::PathBuf {
    write_then(dir, name, mutate)
}

fn bounded(max_bytes: usize) -> ReadOptions {
    ReadOptions::default().with_max_bytes(Some(max_bytes))
}

fn assert_limit<T: std::fmt::Debug>(result: Result<T, NirError>, max_bytes: usize) -> NirError {
    let err = result.expect_err("bounded read should exceed its allocation budget");
    match &err {
        NirError::ReadLimitExceeded {
            limit,
            used,
            requested,
            ..
        } => {
            assert_eq!(*limit, max_bytes);
            assert!(*requested > 0);
            assert!(
                used.checked_add(*requested)
                    .is_none_or(|total| total > *limit)
            );
        }
        other => panic!("expected ReadLimitExceeded, got {other:?}"),
    }
    err
}

/// A second HDF5 file standing in for the attacker's target.
fn decoy(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("decoy.h5");
    let file = hdf5::File::create(&path).unwrap();
    let ds = file
        .new_dataset::<f32>()
        .shape([2])
        .create("secret")
        .unwrap();
    ds.write_raw(&[1.0f32, 2.0]).unwrap();
    path
}

/// Run one external-link case: the decoy's `/secret` is linked into the
/// container at `parent`/`link_name`, and reading the graph must refuse it.
fn assert_external_link_rejected(parent: &str, name: &str, link_name: &str, needles: &[&str]) {
    let dir = TempDir::new().unwrap();
    let target = decoy(&dir);
    let path = tampered(&dir, name, |file| {
        file.group(parent)
            .unwrap()
            .link_external(target.to_str().unwrap(), "/secret", link_name)
            .unwrap();
    });

    assert_err(nir_rs::io::read(&path), NirError::InvalidGraph, needles);
}

#[test]
fn external_link_in_the_file_root_is_rejected() {
    assert_external_link_rejected(
        "/",
        "external_root.nir",
        "elsewhere",
        &["external link", "elsewhere"],
    );
}

#[test]
fn external_node_type_is_rejected_before_it_is_read() {
    // `/node/type` is opened by `read` itself, before `read_graph_body` gets a
    // chance to validate the `/node` group. The decoy's `type` holds a value the
    // type check would reject, so the *message* proves the ordering: if the
    // external link were followed first, this would fail with "must be a
    // NIRGraph, found \"NotAGraph\"" — i.e. after the external file had already
    // been opened and read.
    let dir = TempDir::new().unwrap();
    let target = dir.path().join("type_decoy.h5");
    {
        let file = hdf5::File::create(&target).unwrap();
        let ds = file
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("type")
            .unwrap();
        ds.write_scalar(&"NotAGraph".parse::<hdf5::types::VarLenUnicode>().unwrap())
            .unwrap();
    }

    let path = tampered(&dir, "external_type.nir", |file| {
        let node = file.group("node").unwrap();
        node.unlink("type").unwrap();
        node.link_external(target.to_str().unwrap(), "/type", "type")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["external link", "type"],
    );
}

#[test]
fn external_link_inside_a_node_group_is_rejected() {
    assert_external_link_rejected(
        "node/nodes/input",
        "external_node.nir",
        "shape_alias",
        &["external link"],
    );
}

#[test]
fn external_storage_dataset_is_rejected() {
    let dir = TempDir::new().unwrap();
    // Raw data held in a plain file outside the container.
    let raw = dir.path().join("raw.bin");
    std::fs::write(&raw, vec![0u8; 8]).unwrap();

    let path = tampered(&dir, "external_storage.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset::<i64>()
            .external(raw.to_str().unwrap(), 0, 8)
            .shape([1])
            .create("shape")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["external storage"],
    );
}

#[test]
fn virtual_dataset_is_rejected() {
    // The VDS bypass: a guard that only checks external storage misses this.
    let dir = TempDir::new().unwrap();
    let target = decoy(&dir);

    let path = tampered(&dir, "virtual.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset::<f32>()
            .virtual_map(target.to_str().unwrap(), "/secret", 2, .., 2, ..)
            .shape([2])
            .create("shape")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["virtual dataset"],
    );
}

#[test]
fn a_virtual_string_dataset_is_rejected_too() {
    // String datasets go through a different reader, so `/version` and the
    // `type` fields need the same guard as tensors.
    let dir = TempDir::new().unwrap();
    let target = dir.path().join("decoy_strings.h5");
    {
        let file = hdf5::File::create(&target).unwrap();
        let ds = file
            .new_dataset::<f32>()
            .shape([1])
            .create("secret")
            .unwrap();
        ds.write_raw(&[1.0f32]).unwrap();
    }

    let path = tampered(&dir, "virtual_string.nir", |file| {
        file.unlink("version").unwrap();
        file.new_dataset::<f32>()
            .virtual_map(target.to_str().unwrap(), "/secret", 1, .., 1, ..)
            .shape([1])
            .create("version")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read_version(&path),
        NirError::InvalidGraph,
        &["virtual dataset"],
    );
}

#[test]
fn soft_link_is_rejected() {
    // Soft links can resolve to external targets, so they are banned at the
    // same gate as external links. A sibling alias under `nodes/` is the
    // realistic place for one to appear.
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "soft_link.nir", |file| {
        file.group("node/nodes")
            .unwrap()
            .link_soft("input", "alias")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["soft link", "alias"],
    );
}

#[test]
fn oversized_dataset_is_rejected() {
    // Explicit element×decoded-width product above the caller's 800 MB budget.
    // 101M i64 values would decode to ~808 MB; creating the dataspace alone is
    // enough — rejection happens before `read_raw`. Complements the
    // fixed-string / narrow-integer cases that catch under-counting modes.
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "oversized.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset::<i64>()
            .shape([101_000_000])
            .chunk([1024])
            .create("shape")
            .unwrap();
    });

    let err = assert_limit(
        nir_rs::io::read_with(&path, &bounded(800_000_000)),
        800_000_000,
    );
    assert!(err.to_string().contains("input.shape"));
}

#[test]
fn ordinary_files_still_pass_the_guards() {
    // The guards must not reject the real thing.
    for name in ["lif_norse.nir", "cnn_sinabs.nir"] {
        let path = format!("tests/fixtures/{name}");
        nir_rs::io::read(&path).unwrap_or_else(|e| panic!("{name} should still read: {e}"));
        nir_rs::io::read_with(&path, &bounded(1_000_000_000))
            .unwrap_or_else(|e| panic!("{name} should fit the generous budget: {e}"));
    }
}

#[test]
fn a_wide_fixed_string_dataset_is_rejected_on_bytes_not_element_count() {
    // 200K elements is far below any plausible element-count cap, but each is
    // read through the capacity ladder as `FixedAscii<4096>`, so decoding
    // would materialize ~4 GB. A guard that counts elements accepts this; one
    // that charges bytes does not. No data is written, and the rejection
    // happens before `read_raw`, so the test never allocates it.
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "wide_strings.nir", |file| {
        let root = file.group("node").unwrap();
        root.unlink("edges").unwrap();
        root.new_dataset::<hdf5::types::FixedAscii<4096>>()
            .shape([100_000, 2])
            .chunk([16, 2])
            .create("edges")
            .unwrap();
    });

    let err = assert_limit(
        nir_rs::io::read_with(&path, &bounded(800_000_000)),
        800_000_000,
    );
    assert!(err.to_string().contains("edges"));
}

#[test]
fn a_narrow_integer_dataset_is_charged_at_its_decoded_width() {
    // `read_tensor` materializes every integer width as `i64`, so a 1-byte
    // dataset costs eight times its stored size. 150M `i8` elements is 150 MB
    // on the descriptor but 1.2 GB decoded: charging the source width accepts
    // it, charging the destination width does not.
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "narrow_ints.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset::<i8>()
            .shape([150_000_000])
            .chunk([4096])
            .create("shape")
            .unwrap();
    });

    let err = assert_limit(
        nir_rs::io::read_with(&path, &bounded(800_000_000)),
        800_000_000,
    );
    assert!(err.to_string().contains("input.shape"));
}

#[test]
fn cumulative_budget_is_shared_across_datasets() {
    // Two equal shape payloads under a limit that cannot cover both (plus the
    // fixed graph overhead). The shared `ReadBudget` must reject the read
    // rather than allocating both shapes unbounded.
    //
    // On HDF5 builds where scalar VLEN sizing falls back to the containing
    // file size, the first over-budget charge can be an early type/version
    // string with `used == 0`. That is still a correct rejection; when the
    // overrun happens later, `used > 0` proves earlier charges applied.
    const N: usize = 16;
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "cumulative.nir", |file| {
        for node_name in ["input", "output"] {
            let node = file.group(&format!("node/nodes/{node_name}")).unwrap();
            node.unlink("shape").unwrap();
            let ds = node
                .new_dataset::<i64>()
                .shape([N])
                .create("shape")
                .unwrap();
            ds.write_raw(&[1_i64; N]).unwrap();
        }
    });

    let known_data_size = 2 * N * std::mem::size_of::<i64>();
    let limit = known_data_size + known_data_size / 2;

    let err = assert_limit(nir_rs::io::read_with(&path, &bounded(limit)), limit);
    let NirError::ReadLimitExceeded {
        used,
        requested,
        limit: reported_limit,
        ..
    } = err
    else {
        unreachable!()
    };
    assert_eq!(reported_limit, limit);
    assert!(requested > 0);
    // Prefer the stronger cumulative signal when available.
    if used > 0 {
        assert!(
            used.saturating_add(requested) > limit || used.checked_add(requested).is_none(),
            "cumulative overrun expected: used={used} requested={requested} limit={limit}"
        );
    }
}

#[test]
fn variable_length_payload_is_charged_before_read() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "large_version.nir", |file| {
        file.unlink("version").unwrap();
        let ds = file
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("version")
            .unwrap();
        let payload = "v".repeat(16_384);
        ds.write_scalar(&payload.parse::<hdf5::types::VarLenUnicode>().unwrap())
            .unwrap();
    });

    let limit = 4096;
    let err = assert_limit(nir_rs::io::read_version_with(&path, &bounded(limit)), limit);
    assert!(err.to_string().contains("version"));

    // The compatibility entry point remains intentionally unbounded.
    assert_eq!(nir_rs::io::read_version(&path).unwrap(), "v".repeat(16_384));
}
