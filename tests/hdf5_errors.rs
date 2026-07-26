// SPDX-License-Identifier: MIT OR Apache-2.0

//! Failure modes of HDF5 `.nir` I/O.
//!
//! Consumers such as `silicon-bridge` need to tell "this file is not NIR" apart
//! from "this node type is not supported yet", so each case below asserts the
//! specific [`NirError`] variant, not merely that an error occurred.

#![cfg(feature = "hdf5")]

use nir_rs::io::WriteOptions;
use nir_rs::nodes::{Input, Output};
use nir_rs::types::{MetadataValue, Tensor};
use nir_rs::{NirError, NirGraph, NirNode};
use tempfile::TempDir;

/// Assert that a Result is an error of the expected variant with a message containing the needle.
fn assert_err<T>(
    result: Result<T, NirError>,
    expected_variant: fn(String) -> NirError,
    needle: &str,
) {
    let err = result.unwrap_err();
    // Check the variant by constructing a dummy instance and comparing discriminants
    let dummy = expected_variant(String::new());
    assert_eq!(
        std::mem::discriminant(&err),
        std::mem::discriminant(&dummy),
        "expected {:?} variant, got {:?}",
        dummy,
        err
    );
    let msg = err.to_string();
    assert!(
        msg.contains(needle),
        "expected message to contain {:?}, got {:?}",
        needle,
        msg
    );
}

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
    assert_err(
        nir_rs::io::read("definitely/not/here.nir"),
        NirError::Io,
        "cannot open",
    );
}

#[test]
fn non_hdf5_file_is_an_io_error() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("not_hdf5.nir");
    std::fs::write(&path, b"this is plain text, not an HDF5 container").unwrap();

    assert_err(nir_rs::io::read(&path), NirError::Io, "");
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

    assert_err(nir_rs::io::read(&path), NirError::InvalidGraph, "NIRGraph");
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

    assert_err(nir_rs::io::read(&path), NirError::InvalidGraph, "(E, 2)");
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

    assert_err(
        nir_rs::io::write(dir.path().join("slash.nir"), &graph),
        NirError::InvalidGraph,
        "layer/one",
    );
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

    assert_err(
        nir_rs::io::write(dir.path().join("nested_slash.nir"), &outer),
        NirError::InvalidGraph,
        "",
    );
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
fn node_name_with_a_nul_byte_is_rejected_before_the_file_is_created() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("existing.nir");

    // Something valuable is already at the destination.
    let mut good = NirGraph::new();
    good.insert_node("input", input(vec![1])).unwrap();
    nir_rs::io::write(&path, &good).unwrap();
    let before = std::fs::metadata(&path).unwrap().len();

    let mut bad = NirGraph::new();
    bad.insert_node("na\0me", input(vec![1])).unwrap();
    assert_err(
        nir_rs::io::write(&path, &bad),
        NirError::InvalidGraph,
        "NUL",
    );

    // The rejected write must not have truncated the existing file.
    assert_eq!(std::fs::metadata(&path).unwrap().len(), before);
    assert_eq!(nir_rs::io::read(&path).unwrap().len(), 1);
}

#[test]
fn metadata_key_with_a_slash_is_rejected_before_the_file_is_created() {
    // Metadata keys become HDF5 link names too, so they need the same
    // preflight as node names — otherwise the failure lands after truncation.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("existing.nir");

    let mut good = NirGraph::new();
    good.insert_node("input", input(vec![1])).unwrap();
    nir_rs::io::write(&path, &good).unwrap();
    let before = std::fs::metadata(&path).unwrap().len();

    let mut bad = NirGraph::new();
    bad.metadata
        .insert("nested/key".into(), MetadataValue::String("boom".into()));
    let err = nir_rs::io::write(&path, &bad).unwrap_err();
    match err {
        NirError::InvalidGraph(message) => {
            assert!(message.contains("metadata key"), "got {message}");
            assert!(message.contains("nested/key"), "got {message}");
        }
        other => panic!("expected InvalidGraph, got {other:?}"),
    }
    assert_eq!(std::fs::metadata(&path).unwrap().len(), before);
}

#[test]
fn node_metadata_keys_are_checked_too() {
    let dir = TempDir::new().unwrap();
    let mut node_metadata = nir_rs::types::MetadataMap::new();
    node_metadata.insert("bad\0key".into(), MetadataValue::I64(1));

    let mut graph = NirGraph::new();
    graph
        .insert_node(
            "input",
            NirNode::Input(Input {
                shape: vec![1],
                metadata: node_metadata,
            }),
        )
        .unwrap();

    let err = nir_rs::io::write(dir.path().join("bad_node_key.nir"), &graph).unwrap_err();
    assert!(err.to_string().contains("metadata key"), "got {err}");
    assert!(err.to_string().contains("NUL"), "got {err}");
}

#[test]
fn rank_0_metadata_tensor_is_rejected_as_ambiguous() {
    // A rank-0 tensor and a MetadataValue::F64 are the same dataset on the
    // wire, so the tensor would decode back as the scalar variant.
    let dir = TempDir::new().unwrap();
    let mut graph = NirGraph::new();
    graph.metadata.insert(
        "scalar".into(),
        MetadataValue::Tensor(Tensor::scalar_f32(1.5)),
    );

    let path = dir.path().join("scalar_metadata.nir");
    assert_err(
        nir_rs::io::write(&path, &graph),
        NirError::InvalidGraph,
        "rank-0",
    );

    // With validation off it is written, and decodes as the scalar variant —
    // which is exactly the lossiness the check exists to surface.
    nir_rs::io::write_with(
        &path,
        &graph,
        &WriteOptions::default().with_validation(false),
    )
    .unwrap();
    assert_eq!(
        nir_rs::io::read(&path).unwrap().metadata.get("scalar"),
        Some(&MetadataValue::F64(1.5))
    );
}

#[test]
fn optional_field_of_the_wrong_link_kind_is_not_treated_as_absent() {
    // A `v_reset` group is malformed, not missing: defaulting it to zeros
    // would silently change the model.
    let dir = TempDir::new().unwrap();
    let mut graph = NirGraph::new();
    graph
        .insert_node(
            "lif",
            NirNode::Lif(nir_rs::nodes::Lif {
                tau: Tensor::from_f64([1], vec![10.0]).unwrap(),
                r: Tensor::from_f64([1], vec![1.0]).unwrap(),
                v_leak: Tensor::from_f64([1], vec![0.0]).unwrap(),
                v_threshold: Tensor::from_f64([1], vec![1.0]).unwrap(),
                v_reset: None,
                metadata: Default::default(),
            }),
        )
        .unwrap();

    let path = dir.path().join("bad_v_reset.nir");
    nir_rs::io::write(&path, &graph).unwrap();
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        file.group("node/nodes/lif")
            .unwrap()
            .create_group("v_reset")
            .unwrap();
    }

    let err = nir_rs::io::read(&path).unwrap_err();
    match err {
        NirError::Io(message) => {
            assert!(message.contains("lif.v_reset"), "got {message}");
            assert!(message.contains("expected a dataset"), "got {message}");
        }
        other => panic!("expected Io, got {other:?}"),
    }
}

#[test]
fn subgraph_carrying_a_version_is_rejected() {
    // The wire format has a single root `/version`, so a nested one would be
    // silently dropped on write and absent again on read.
    let dir = TempDir::new().unwrap();
    let mut inner = NirGraph::new();
    inner.version = Some("0.2.0".into());
    inner.insert_node("input", input(vec![1])).unwrap();

    let mut outer = NirGraph::new();
    outer
        .insert_node("sub", NirNode::Graph(Box::new(inner)))
        .unwrap();

    let path = dir.path().join("nested_version.nir");
    let err = nir_rs::io::write(&path, &outer).unwrap_err();
    match err {
        NirError::InvalidGraph(message) => {
            assert!(message.contains("sub"), "got {message}");
            assert!(message.contains("version"), "got {message}");
        }
        other => panic!("expected InvalidGraph, got {other:?}"),
    }

    // Opting out of validation writes it anyway, dropping the nested version.
    nir_rs::io::write_with(
        &path,
        &outer,
        &WriteOptions::default().with_validation(false),
    )
    .unwrap();
    let decoded = nir_rs::io::read(&path).unwrap();
    let NirNode::Graph(sub) = decoded.get("sub").unwrap() else {
        panic!("expected a nested graph");
    };
    assert_eq!(sub.version, None);
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

    assert_err(
        nir_rs::io::write(dir.path().join("conv1d.nir"), &graph),
        NirError::InvalidGraph,
        "Conv1d stride",
    );
}

#[test]
fn write_to_an_unwritable_path_is_an_io_error() {
    assert_err(
        nir_rs::io::write("no/such/directory/model.nir", &NirGraph::new()),
        NirError::Io,
        "cannot create",
    );
}

#[test]
fn version_with_a_nul_byte_is_rejected_before_the_file_is_created() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("existing.nir");
    nir_rs::io::write(&path, &NirGraph::new()).unwrap();
    let before = std::fs::metadata(&path).unwrap().len();

    assert_err(
        nir_rs::io::write_with(
            &path,
            &NirGraph::new(),
            &WriteOptions::default().with_version("0.2\0.0"),
        ),
        NirError::InvalidGraph,
        "NUL",
    );
    assert_eq!(std::fs::metadata(&path).unwrap().len(), before);
}

#[test]
fn metadata_string_with_a_nul_byte_is_rejected_before_the_file_is_created() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("existing.nir");
    nir_rs::io::write(&path, &NirGraph::new()).unwrap();
    let before = std::fs::metadata(&path).unwrap().len();

    let mut bad = NirGraph::new();
    bad.metadata
        .insert("note".into(), MetadataValue::String("bad\0value".into()));
    assert_err(
        nir_rs::io::write(&path, &bad),
        NirError::InvalidGraph,
        "NUL",
    );
    assert_eq!(std::fs::metadata(&path).unwrap().len(), before);
}

#[test]
fn conv2d_input_shape_must_be_a_pair() {
    let dir = TempDir::new().unwrap();
    let mut graph = NirGraph::new();
    graph
        .insert_node(
            "conv",
            NirNode::Conv2d(nir_rs::nodes::Conv2d {
                weight: Tensor::from_f32(vec![1, 1, 2, 2], vec![0.; 4]).unwrap(),
                stride: vec![1, 1],
                padding: nir_rs::nodes::Padding::Valid,
                dilation: vec![1, 1],
                groups: 1,
                bias: Tensor::from_f32([1], vec![0.]).unwrap(),
                input_shape: Some(vec![8]),
                metadata: Default::default(),
            }),
        )
        .unwrap();

    assert_err(
        nir_rs::io::write(dir.path().join("conv2d.nir"), &graph),
        NirError::InvalidGraph,
        "Conv2d",
    );
}

#[test]
fn metadata_link_of_the_wrong_kind_is_not_treated_as_absent() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "bad_metadata_link.nir", |file| {
        let root = file.group("node").unwrap();
        let ds = root
            .new_dataset::<i64>()
            .shape(())
            .create("metadata")
            .unwrap();
        ds.write_scalar(&1i64).unwrap();
    });

    let err = nir_rs::io::read(&path).unwrap_err();
    match err {
        NirError::Io(message) => {
            assert!(message.contains("metadata"), "got {message}");
            assert!(message.contains("expected a group"), "got {message}");
        }
        other => panic!("expected Io, got {other:?}"),
    }
}

#[test]
fn scalar_u64_metadata_above_i64_max_is_rejected() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "big_u64_meta.nir", |file| {
        let root = file.group("node").unwrap();
        let md = root.create_group("metadata").unwrap();
        let ds = md.new_dataset::<u64>().shape(()).create("huge").unwrap();
        ds.write_scalar(&(i64::MAX as u64 + 1)).unwrap();
    });

    let err = nir_rs::io::read(&path).unwrap_err();
    match err {
        NirError::InvalidTensor(message) => {
            assert!(message.contains("metadata.huge"), "got {message}");
            assert!(message.contains("does not fit in i64"), "got {message}");
        }
        other => panic!("expected InvalidTensor, got {other:?}"),
    }
}

#[test]
fn nested_graph_hard_link_cycle_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut inner = NirGraph::new();
    inner.insert_node("input", input(vec![1])).unwrap();
    let mut outer = NirGraph::new();
    outer
        .insert_node("sub", NirNode::Graph(Box::new(inner)))
        .unwrap();

    let path = dir.path().join("cycle.nir");
    nir_rs::io::write(&path, &outer).unwrap();
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        // Hard-link the nested graph back onto itself under nodes/.
        file.group("node/nodes/sub")
            .unwrap()
            .link_hard(".", "nodes/loop")
            .unwrap();
        // Force the hard-linked child to look like a NIRGraph node.
        let looped = file.group("node/nodes/sub/nodes/loop").unwrap();
        if !looped.link_exists("type") {
            let ds = looped
                .new_dataset::<hdf5::types::VarLenUnicode>()
                .shape(())
                .create("type")
                .unwrap();
            ds.write_scalar(&"NIRGraph".parse::<hdf5::types::VarLenUnicode>().unwrap())
                .unwrap();
        }
    }

    let err = nir_rs::io::read(&path).unwrap_err();
    match err {
        NirError::InvalidGraph(message) => {
            assert!(message.contains("cycle"), "got {message}");
        }
        other => panic!("expected InvalidGraph, got {other:?}"),
    }
}
