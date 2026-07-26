// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guards against `.nir` files that try to reach outside their own container.

#![cfg(feature = "hdf5")]

mod common;
use common::{assert_err, write_then};
use nir_rs::NirError;
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

#[test]
fn external_link_in_the_file_root_is_rejected() {
    let dir = TempDir::new().unwrap();
    let target = decoy(&dir);
    let path = tampered(&dir, "external_root.nir", |file| {
        file.link_external(target.to_str().unwrap(), "/secret", "elsewhere")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
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
    let dir = TempDir::new().unwrap();
    let target = decoy(&dir);
    let path = tampered(&dir, "external_node.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.link_external(target.to_str().unwrap(), "/secret", "shape_alias")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
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
fn ordinary_files_still_pass_the_guards() {
    // The guards must not reject the real thing.
    for name in ["lif_norse.nir", "cnn_sinabs.nir"] {
        nir_rs::io::read(format!("tests/fixtures/{name}"))
            .unwrap_or_else(|e| panic!("{name} should still read: {e}"));
    }
}
