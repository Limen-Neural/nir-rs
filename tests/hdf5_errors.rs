// SPDX-License-Identifier: MIT OR Apache-2.0

//! Failure modes of HDF5 `.nir` I/O.
//!
//! Consumers such as `silicon-bridge` need to tell "this file is not NIR" apart
//! from "this node type is not supported yet", so each case below asserts the
//! specific [`NirError`] variant, not merely that an error occurred.

#![cfg(feature = "hdf5")]

use nir_rs::io::WriteOptions;
use nir_rs::nodes::{Input, Output};
use nir_rs::types::Tensor;
use nir_rs::{NirError, NirGraph, NirNode};
use tempfile::TempDir;

fn input(shape: Vec<usize>) -> NirNode {
    NirNode::Input(Input {
        shape,
        metadata: Default::default(),
    })
}

/// Build a minimal well-formed file, then hand it to `mutate` for corruption.
fn write_then(dir: &TempDir, name: &str, mutate: impl FnOnce(&hdf5::File)) -> std::path::PathBuf {
    let mut graph = NirGraph::new();
    graph.insert_node("input", input(vec![1])).unwrap();
    graph
        .insert_node(
            "output",
            NirNode::Output(Output {
                shape: vec![1],
                metadata: Default::default(),
            }),
        )
        .unwrap();
    graph.add_edge("input", "output");

    let path = dir.path().join(name);
    nir_rs::io::write(&path, &graph).unwrap();

    let file = hdf5::File::open_rw(&path).unwrap();
    mutate(&file);
    drop(file);
    path
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

#[test]
fn missing_file_is_an_io_error() {
    let err = nir_rs::io::read("definitely/not/here.nir").unwrap_err();
    assert!(matches!(err, NirError::Io(_)), "got {err:?}");
    assert!(err.to_string().contains("cannot open"));
}

#[test]
fn non_hdf5_file_is_an_io_error() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("not_hdf5.nir");
    std::fs::write(&path, b"this is plain text, not an HDF5 container").unwrap();

    let err = nir_rs::io::read(&path).unwrap_err();
    assert!(matches!(err, NirError::Io(_)), "got {err:?}");
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

    let err = nir_rs::io::read(&path).unwrap_err();
    match err {
        NirError::InvalidGraph(message) => assert!(message.contains("(E, 2)"), "got {message}"),
        other => panic!("expected InvalidGraph, got {other:?}"),
    }
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

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

#[test]
fn node_name_with_a_slash_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut graph = NirGraph::new();
    graph.insert_node("layer/one", input(vec![1])).unwrap();

    let err = nir_rs::io::write(dir.path().join("slash.nir"), &graph).unwrap_err();
    match err {
        NirError::InvalidGraph(message) => {
            assert!(message.contains("layer/one"), "got {message}");
            assert!(message.contains('/'), "got {message}");
        }
        other => panic!("expected InvalidGraph, got {other:?}"),
    }
}

#[test]
fn illegal_name_inside_a_subgraph_is_rejected_too() {
    let dir = TempDir::new().unwrap();
    let mut inner = NirGraph::new();
    inner.insert_node("bad/name", input(vec![1])).unwrap();

    let mut outer = NirGraph::new();
    outer
        .insert_node("sub", NirNode::Graph(Box::new(inner)))
        .unwrap();

    let err = nir_rs::io::write(dir.path().join("nested_slash.nir"), &outer).unwrap_err();
    assert!(matches!(err, NirError::InvalidGraph(_)), "got {err:?}");
}

#[test]
fn dangling_edge_is_rejected_unless_validation_is_disabled() {
    let dir = TempDir::new().unwrap();
    let mut graph = NirGraph::new();
    graph.insert_node("input", input(vec![1])).unwrap();
    graph.add_edge("input", "ghost");

    let path = dir.path().join("dangling.nir");
    let err = nir_rs::io::write(&path, &graph).unwrap_err();
    assert_eq!(err, NirError::MissingNode("ghost".into()));
    assert!(!path.exists(), "a rejected write must not leave a file");

    // The escape hatch exists because upstream ships such a file.
    nir_rs::io::write_with(
        &path,
        &graph,
        &WriteOptions::default().with_validation(false),
    )
    .unwrap();
    assert_eq!(nir_rs::io::read(&path).unwrap().edges, graph.edges);
}

#[test]
fn conv1d_extent_with_two_values_is_rejected() {
    // Conv1d writes scalars on the wire, so a two-element stride has no valid
    // encoding and must fail loudly rather than silently drop a value.
    let dir = TempDir::new().unwrap();
    let mut graph = NirGraph::new();
    graph
        .insert_node(
            "conv",
            NirNode::Conv1d(nir_rs::nodes::Conv1d {
                weight: Tensor::from_f32(vec![1, 1, 3], vec![1., 0., -1.]).unwrap(),
                stride: vec![1, 1],
                padding: nir_rs::nodes::Padding::single(0),
                dilation: vec![1],
                groups: 1,
                bias: Tensor::from_f32([1], vec![0.]).unwrap(),
                input_shape: None,
                metadata: Default::default(),
            }),
        )
        .unwrap();

    let err = nir_rs::io::write(dir.path().join("conv1d.nir"), &graph).unwrap_err();
    match err {
        NirError::InvalidGraph(message) => {
            assert!(message.contains("Conv1d stride"), "got {message}");
        }
        other => panic!("expected InvalidGraph, got {other:?}"),
    }
}

#[test]
fn write_to_an_unwritable_path_is_an_io_error() {
    let err = nir_rs::io::write("no/such/directory/model.nir", &NirGraph::new()).unwrap_err();
    assert!(matches!(err, NirError::Io(_)), "got {err:?}");
    assert!(err.to_string().contains("cannot create"));
}
