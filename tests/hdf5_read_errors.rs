// SPDX-License-Identifier: MIT OR Apache-2.0

//! Failure modes when **reading** malformed `.nir` files.
//!
//! Downstream tools need to tell "this file is not NIR" apart
//! from "this node type is not supported yet", so each case below asserts the
//! specific [`NirError`] variant, not merely that an error occurred.

#![cfg(feature = "hdf5")]

mod common;
use common::{assert_err, write_then};
use nir_rs::NirError;
use tempfile::TempDir;

#[test]
fn missing_file_is_an_io_error() {
    assert_err(
        nir_rs::io::read("definitely/not/here.nir"),
        NirError::Io,
        &["cannot open"],
    );
}

#[test]
fn non_hdf5_file_is_an_io_error() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("not_hdf5.nir");
    std::fs::write(&path, b"this is plain text, not an HDF5 container").unwrap();

    assert_err(nir_rs::io::read(&path), NirError::Io, &[]);
}

#[test]
fn hdf5_file_without_a_node_group_is_rejected() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("no_node.nir");
    {
        let file = hdf5::File::create(&path).unwrap();
        file.create_group("something_else").unwrap();
    }

    let err = nir_rs::io::read(&path).unwrap_err();
    match err {
        NirError::MissingField(field) => {
            assert!(field.contains("/node"), "got {field}");
            assert!(field.contains("not a NIR graph file"), "got {field}");
        }
        other => panic!("expected MissingField, got {other:?}"),
    }
}

#[test]
fn hdf5_file_whose_node_group_is_not_a_graph_is_rejected() {
    // A `node` group alone does not make a file NIR; `/node/type` decides.
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "wrong_root_type.nir", |file| {
        let root = file.group("node").unwrap();
        root.unlink("type").unwrap();
        let ds = root
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("type")
            .unwrap();
        ds.write_scalar(&"LIF".parse::<hdf5::types::VarLenUnicode>().unwrap())
            .unwrap();
    });

    // The message must name both what was expected and what was actually found.
    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["NIRGraph", "LIF"],
    );
}

#[test]
fn root_without_a_type_is_rejected() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "no_root_type.nir", |file| {
        file.group("node").unwrap().unlink("type").unwrap();
    });

    let err = nir_rs::io::read(&path).unwrap_err();
    assert_eq!(err, NirError::MissingField("/node/type".into()));
}

#[test]
fn missing_edges_is_an_error_rather_than_a_disconnected_graph() {
    // Upstream `NIRGraph.from_dict` asserts the key is present even for an
    // empty graph, so a truncated file must not decode as "no connections".
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "no_edges.nir", |file| {
        file.group("node").unwrap().unlink("edges").unwrap();
    });

    let err = nir_rs::io::read(&path).unwrap_err();
    assert_eq!(err, NirError::MissingField("/node/edges".into()));
}

#[test]
fn missing_nodes_group_is_an_error() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "no_nodes.nir", |file| {
        file.group("node").unwrap().unlink("nodes").unwrap();
    });

    let err = nir_rs::io::read(&path).unwrap_err();
    assert_eq!(err, NirError::MissingField("/node/nodes".into()));
}

#[test]
fn unknown_node_type_is_reported_verbatim() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "unknown_type.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("type").unwrap();
        let ds = node
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("type")
            .unwrap();
        ds.write_scalar(&"CurrLIF".parse::<hdf5::types::VarLenUnicode>().unwrap())
            .unwrap();
    });

    let err = nir_rs::io::read(&path).unwrap_err();
    assert_eq!(err, NirError::UnknownNodeType("CurrLIF".into()));
}

#[test]
fn missing_required_field_names_the_node_and_field() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "missing_shape.nir", |file| {
        file.group("node/nodes/input")
            .unwrap()
            .unlink("shape")
            .unwrap();
    });

    let err = nir_rs::io::read(&path).unwrap_err();
    assert_eq!(err, NirError::MissingField("input.shape".into()));
}

#[test]
fn negative_axis_length_is_an_invalid_tensor() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "negative_shape.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        let ds = node
            .new_dataset::<i64>()
            .shape([1])
            .create("shape")
            .unwrap();
        ds.write_raw(&[-3i64]).unwrap();
    });

    let err = nir_rs::io::read(&path).unwrap_err();
    match err {
        NirError::InvalidTensor(message) => {
            assert!(message.contains("input.shape"), "got {message}");
            assert!(message.contains("-3"), "got {message}");
        }
        other => panic!("expected InvalidTensor, got {other:?}"),
    }
}

#[test]
fn malformed_edges_shape_is_an_invalid_graph() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "bad_edges.nir", |file| {
        let root = file.group("node").unwrap();
        root.unlink("edges").unwrap();
        let ds = root
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape([3])
            .create("edges")
            .unwrap();
        let values: Vec<hdf5::types::VarLenUnicode> =
            ["a", "b", "c"].iter().map(|s| s.parse().unwrap()).collect();
        ds.write_raw(&values).unwrap();
    });

    assert_err(nir_rs::io::read(&path), NirError::InvalidGraph, &["(E, 2)"]);
}

#[test]
fn read_version_errors_when_absent_but_read_does_not() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "no_version.nir", |file| {
        file.unlink("version").unwrap();
    });

    let err = nir_rs::io::read_version(&path).unwrap_err();
    assert_eq!(err, NirError::MissingField("/version".into()));

    // Reading the graph itself still succeeds; the version is simply unknown.
    let graph = nir_rs::io::read(&path).unwrap();
    assert_eq!(graph.version, None);
    assert_eq!(graph.len(), 2);
}

#[test]
fn a_malformed_version_is_not_reported_as_a_missing_one() {
    // `/version` present but a group, not a dataset. Both entry points must
    // call that what it is: reporting `MissingField` would tell a caller to
    // go add a version string that is already there.
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "version_group.nir", |file| {
        file.unlink("version").unwrap();
        file.create_group("version").unwrap();
    });

    assert_err(
        nir_rs::io::read_version(&path),
        NirError::Io,
        &["/version", "expected a dataset"],
    );
    // `read` reaches the same link by a different path; it must agree.
    assert_err(
        nir_rs::io::read(&path),
        NirError::Io,
        &["/version", "expected a dataset"],
    );
}

#[test]
fn rank_2_string_metadata_is_refused_rather_than_flattened() {
    // A Python `list[list[str]]`. `MetadataValue::StringList` is flat, so
    // decoding this would silently drop the nesting and write it back as a
    // rank-1 dataset. Refusing keeps the failure loud.
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "nested_strings.nir", |file| {
        let md = file
            .group("node/nodes/input")
            .unwrap()
            .create_group("metadata")
            .unwrap();
        md.new_dataset::<hdf5::types::VarLenUnicode>()
            .shape([2, 2])
            .create("grid")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidTensor,
        &["grid", "rank-1", "[2, 2]"],
    );
}
