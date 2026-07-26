// SPDX-License-Identifier: MIT OR Apache-2.0

//! Round-trip fidelity for HDF5 `.nir` I/O.
//!
//! Two directions are covered:
//!
//! - **write → read** on graphs constructed in Rust, exercising every wire node
//!   type, every metadata variant, nesting, and both compression settings.
//! - **read → write → read** on the vendored Python-written fixtures, which is
//!   what proves the writer emits the same structure the files themselves use.
//!
//! Equality is defined at the **graph** level, not file bytes: node names and
//! types, ordered edges, exact parameter values and dtypes, recursing into
//! nested graphs. `NirGraph` already derives `PartialEq` with those semantics
//! (`IndexMap` compares as a map, so node ordering is not significant, while
//! `edges` is a `Vec` and is).

#![cfg(feature = "hdf5")]

use nir_rs::io::WriteOptions;
use nir_rs::nodes::{
    Affine, AvgPool2d, Conv1d, Conv2d, CubaLi, CubaLif, Delay, Flatten, I, If, Input, Li, Lif,
    Linear, Output, Padding, Scale, SumPool2d, Threshold,
};
use nir_rs::types::{MetadataValue, Tensor};
use nir_rs::{NirGraph, NirNode};
use tempfile::TempDir;

const FIXTURES: [&str; 5] = [
    "lif_norse.nir",
    "two_lif_neurons.nir",
    "braille_noDelay_bias_zero.nir",
    "braille_noDelay_bias_zero_subgraph.nir",
    "cnn_sinabs.nir",
];

fn scratch(dir: &TempDir, name: &str) -> std::path::PathBuf {
    dir.path().join(name)
}

/// Write then read back, returning the decoded graph.
fn round_trip(graph: &NirGraph, opts: &WriteOptions) -> NirGraph {
    let dir = TempDir::new().unwrap();
    let path = scratch(&dir, "round_trip.nir");
    nir_rs::io::write_with(&path, graph, opts).expect("write");
    nir_rs::io::read(&path).expect("read")
}

// ---------------------------------------------------------------------------
// Graphs constructed in Rust
// ---------------------------------------------------------------------------

fn vec3_f64(values: [f64; 3]) -> Tensor {
    Tensor::from_f64([3], values.to_vec()).unwrap()
}

/// A graph containing one of every wire node type, wired in a single chain so
/// it also validates. Node kinds that cannot legally follow one another are
/// still fine here: `validate_structure` checks endpoints, not shapes.
fn graph_with_every_node_type() -> NirGraph {
    let mut inner = NirGraph::new();
    inner
        .insert_node(
            "in",
            NirNode::Input(Input {
                shape: vec![3],
                metadata: Default::default(),
            }),
        )
        .unwrap();
    inner
        .insert_node(
            "out",
            NirNode::Output(Output {
                shape: vec![3],
                metadata: Default::default(),
            }),
        )
        .unwrap();
    inner.add_edge("in", "out");

    let nodes: Vec<(&str, NirNode)> = vec![
        (
            "input",
            NirNode::Input(Input {
                shape: vec![2, 3],
                metadata: Default::default(),
            }),
        ),
        (
            "affine",
            NirNode::Affine(Affine {
                weight: Tensor::from_f32(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
                bias: Tensor::from_f32([2], vec![0.5, -0.5]).unwrap(),
                metadata: Default::default(),
            }),
        ),
        (
            "linear",
            NirNode::Linear(Linear {
                weight: Tensor::from_f64(vec![2, 2], vec![1., 0., 0., 1.]).unwrap(),
                metadata: Default::default(),
            }),
        ),
        (
            "scale",
            NirNode::Scale(Scale {
                scale: vec3_f64([0.5, 1.5, 2.5]),
                metadata: Default::default(),
            }),
        ),
        (
            "conv1d",
            NirNode::Conv1d(Conv1d {
                weight: Tensor::from_f32(vec![1, 1, 3], vec![1., 0., -1.]).unwrap(),
                stride: vec![2],
                padding: Padding::single(1),
                dilation: vec![1],
                groups: 1,
                bias: Tensor::from_f32([1], vec![0.25]).unwrap(),
                input_shape: Some(10),
                metadata: Default::default(),
            }),
        ),
        (
            "conv2d",
            NirNode::Conv2d(Conv2d {
                weight: Tensor::from_f32(vec![2, 1, 2, 2], vec![0.1; 8]).unwrap(),
                stride: vec![1, 2],
                padding: Padding::pair(1, 0),
                dilation: vec![1, 1],
                groups: 1,
                bias: Tensor::from_f32([2], vec![0., 1.]).unwrap(),
                input_shape: Some(vec![28, 28]),
                metadata: Default::default(),
            }),
        ),
        (
            "conv2d_same",
            NirNode::Conv2d(Conv2d {
                weight: Tensor::from_f32(vec![1, 1, 1, 1], vec![2.0]).unwrap(),
                stride: vec![1, 1],
                padding: Padding::Same,
                dilation: vec![1, 1],
                groups: 2,
                bias: Tensor::from_f32([1], vec![0.]).unwrap(),
                input_shape: None,
                metadata: Default::default(),
            }),
        ),
        (
            "conv1d_valid",
            NirNode::Conv1d(Conv1d {
                weight: Tensor::from_f32(vec![1, 1, 2], vec![1., 1.]).unwrap(),
                stride: vec![1],
                padding: Padding::Valid,
                dilation: vec![2],
                groups: 1,
                bias: Tensor::from_f32([1], vec![0.]).unwrap(),
                input_shape: None,
                metadata: Default::default(),
            }),
        ),
        (
            "cuba_li",
            NirNode::CubaLi(CubaLi {
                tau_syn: vec3_f64([1., 2., 3.]),
                tau_mem: vec3_f64([4., 5., 6.]),
                r: vec3_f64([1., 1., 1.]),
                v_leak: vec3_f64([0., 0., 0.]),
                w_in: Some(vec3_f64([2., 2., 2.])),
                metadata: Default::default(),
            }),
        ),
        (
            "cuba_lif",
            NirNode::CubaLif(CubaLif {
                tau_syn: vec3_f64([1., 2., 3.]),
                tau_mem: vec3_f64([4., 5., 6.]),
                r: vec3_f64([1., 1., 1.]),
                v_leak: vec3_f64([0., 0., 0.]),
                v_threshold: vec3_f64([1., 1., 1.]),
                v_reset: Some(vec3_f64([0.1, 0.2, 0.3])),
                w_in: Some(vec3_f64([1., 1., 1.])),
                metadata: Default::default(),
            }),
        ),
        (
            "delay",
            NirNode::Delay(Delay {
                delay: vec3_f64([1., 2., 3.]),
                metadata: Default::default(),
            }),
        ),
        (
            "flatten",
            NirNode::Flatten(Flatten {
                start_dim: 1,
                end_dim: -1,
                input_type: Some(vec![1, 4, 4]),
                metadata: Default::default(),
            }),
        ),
        (
            "flatten_bare",
            NirNode::Flatten(Flatten {
                start_dim: 0,
                end_dim: 2,
                input_type: None,
                metadata: Default::default(),
            }),
        ),
        (
            "i",
            NirNode::I(I {
                r: vec3_f64([1., 1., 1.]),
                metadata: Default::default(),
            }),
        ),
        (
            "if",
            NirNode::If(If {
                r: vec3_f64([1., 1., 1.]),
                v_threshold: vec3_f64([1., 1., 1.]),
                v_reset: Some(vec3_f64([0., 0., 0.])),
                metadata: Default::default(),
            }),
        ),
        (
            "li",
            NirNode::Li(Li {
                tau: vec3_f64([10., 10., 10.]),
                r: vec3_f64([1., 1., 1.]),
                v_leak: vec3_f64([0., 0., 0.]),
                metadata: Default::default(),
            }),
        ),
        (
            "lif",
            NirNode::Lif(Lif {
                tau: vec3_f64([10., 10., 10.]),
                r: vec3_f64([1., 1., 1.]),
                v_leak: vec3_f64([0., 0., 0.]),
                v_threshold: vec3_f64([1., 1., 1.]),
                v_reset: Some(vec3_f64([0., 0., 0.])),
                metadata: Default::default(),
            }),
        ),
        (
            "sum_pool",
            NirNode::SumPool2d(SumPool2d {
                kernel_size: Tensor::from_i64([2], vec![2, 2]).unwrap(),
                stride: Tensor::from_i64([2], vec![2, 2]).unwrap(),
                padding: Tensor::from_i64([2], vec![0, 0]).unwrap(),
                metadata: Default::default(),
            }),
        ),
        (
            "avg_pool",
            NirNode::AvgPool2d(AvgPool2d {
                kernel_size: Tensor::from_i64([2], vec![3, 3]).unwrap(),
                stride: Tensor::from_i64([2], vec![1, 1]).unwrap(),
                padding: Tensor::from_i64([2], vec![1, 1]).unwrap(),
                metadata: Default::default(),
            }),
        ),
        (
            "threshold",
            NirNode::Threshold(Threshold {
                threshold: Tensor::scalar_f64(0.75),
                metadata: Default::default(),
            }),
        ),
        ("subgraph", NirNode::Graph(Box::new(inner))),
        (
            "output",
            NirNode::Output(Output {
                shape: vec![3],
                metadata: Default::default(),
            }),
        ),
    ];

    let mut graph = NirGraph::new();
    graph.version = Some("1.0.8".into());
    for (name, node) in nodes {
        graph.insert_node(name, node).unwrap();
    }
    let names: Vec<String> = graph.nodes.keys().cloned().collect();
    for pair in names.windows(2) {
        graph.add_edge(&pair[0], &pair[1]);
    }
    graph
}

#[test]
fn every_wire_node_type_round_trips() {
    let original = graph_with_every_node_type();
    // Sanity: the fixture really does cover the whole enum.
    let covered: std::collections::HashSet<&str> = original
        .nodes
        .values()
        .map(nir_rs::NirNode::type_name)
        .collect();
    assert_eq!(
        covered.len(),
        nir_rs::io::wire::WIRE_TYPES.len(),
        "test graph must exercise every wire node type"
    );

    let decoded = round_trip(&original, &WriteOptions::default());
    assert_eq!(decoded, original);
}

#[test]
fn scalar_and_multi_axis_tensors_round_trip() {
    let original = graph_with_every_node_type();
    let decoded = round_trip(&original, &WriteOptions::default());

    let NirNode::Threshold(threshold) = decoded.get("threshold").unwrap() else {
        panic!("expected Threshold");
    };
    assert!(
        threshold.threshold.shape().is_empty(),
        "rank-0 stays rank-0"
    );

    let NirNode::Conv2d(conv) = decoded.get("conv2d").unwrap() else {
        panic!("expected Conv2d");
    };
    assert_eq!(conv.weight.shape(), [2, 1, 2, 2]);
}

#[test]
fn symbolic_padding_round_trips_as_a_string() {
    let decoded = round_trip(&graph_with_every_node_type(), &WriteOptions::default());

    let NirNode::Conv2d(same) = decoded.get("conv2d_same").unwrap() else {
        panic!("expected Conv2d");
    };
    assert_eq!(same.padding, Padding::Same);
    assert_eq!(same.input_shape, None, "absent input_shape stays absent");
    assert_eq!(same.groups, 2);

    let NirNode::Conv1d(valid) = decoded.get("conv1d_valid").unwrap() else {
        panic!("expected Conv1d");
    };
    assert_eq!(valid.padding, Padding::Valid);
    assert_eq!(valid.input_shape, None);

    let NirNode::Conv1d(conv1d) = decoded.get("conv1d").unwrap() else {
        panic!("expected Conv1d");
    };
    assert_eq!(
        conv1d.padding,
        Padding::single(1),
        "scalar extent on the wire"
    );
    assert_eq!(conv1d.stride, [2]);
    assert_eq!(conv1d.input_shape, Some(10));
}

#[test]
fn metadata_round_trips_on_graph_and_nodes() {
    let mut graph = NirGraph::new();
    // A graph with no version acquires DEFAULT_NIR_VERSION on write; set it up
    // front so the comparison below is about metadata only.
    graph.version = Some(nir_rs::io::DEFAULT_NIR_VERSION.into());
    graph.metadata.insert(
        "origin".into(),
        MetadataValue::String("nir-rs round-trip".into()),
    );
    graph
        .metadata
        .insert("float".into(), MetadataValue::F64(0.125));
    graph.metadata.insert("int".into(), MetadataValue::I64(-42));
    graph
        .metadata
        .insert("flag".into(), MetadataValue::Bool(true));
    graph.metadata.insert(
        "array".into(),
        MetadataValue::Tensor(Tensor::from_f32(vec![2, 2], vec![1., 2., 3., 4.]).unwrap()),
    );

    let mut node_metadata = nir_rs::types::MetadataMap::new();
    node_metadata.insert("layer".into(), MetadataValue::String("dense".into()));
    node_metadata.insert("index".into(), MetadataValue::I64(3));
    graph
        .insert_node(
            "input",
            NirNode::Input(Input {
                shape: vec![4],
                metadata: node_metadata,
            }),
        )
        .unwrap();

    assert_eq!(round_trip(&graph, &WriteOptions::default()), graph);
}

#[test]
fn empty_metadata_is_omitted_from_the_file() {
    let mut graph = NirGraph::new();
    graph.version = Some(nir_rs::io::DEFAULT_NIR_VERSION.into());
    graph
        .insert_node(
            "input",
            NirNode::Input(Input {
                shape: vec![1],
                metadata: Default::default(),
            }),
        )
        .unwrap();

    let dir = TempDir::new().unwrap();
    let path = scratch(&dir, "no_metadata.nir");
    nir_rs::io::write(&path, &graph).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    assert!(
        file.group("node").unwrap().group("metadata").is_err(),
        "empty graph metadata must not create a group"
    );
    assert!(
        file.group("node/nodes/input")
            .unwrap()
            .group("metadata")
            .is_err(),
        "empty node metadata must not create a group"
    );
    assert_eq!(nir_rs::io::read(&path).unwrap(), graph);
}

#[test]
fn empty_graph_round_trips() {
    let graph = NirGraph::new();
    let decoded = round_trip(&graph, &WriteOptions::default());
    assert!(decoded.is_empty());
    assert!(decoded.edges.is_empty());
    // A graph with no version of its own gets the crate default.
    assert_eq!(
        decoded.version.as_deref(),
        Some(nir_rs::io::DEFAULT_NIR_VERSION)
    );
}

#[test]
fn compression_settings_do_not_change_the_decoded_graph() {
    let original = graph_with_every_node_type();
    for compression in [None, Some(0), Some(1), Some(9)] {
        let decoded = round_trip(
            &original,
            &WriteOptions::default().with_compression(compression),
        );
        assert_eq!(decoded, original, "compression = {compression:?}");
    }
}

#[test]
fn compression_actually_shrinks_the_file() {
    let mut graph = NirGraph::new();
    graph.version = Some(nir_rs::io::DEFAULT_NIR_VERSION.into());
    graph
        .insert_node(
            "affine",
            NirNode::Affine(Affine {
                // Highly compressible: a large constant weight matrix.
                weight: Tensor::from_f32(vec![256, 256], vec![1.0; 256 * 256]).unwrap(),
                bias: Tensor::from_f32([256], vec![0.0; 256]).unwrap(),
                metadata: Default::default(),
            }),
        )
        .unwrap();

    let dir = TempDir::new().unwrap();
    let plain = scratch(&dir, "plain.nir");
    let gzipped = scratch(&dir, "gzipped.nir");
    nir_rs::io::write_with(
        &plain,
        &graph,
        &WriteOptions::default().with_compression(None),
    )
    .unwrap();
    nir_rs::io::write_with(&gzipped, &graph, &WriteOptions::default()).unwrap();

    let plain_size = std::fs::metadata(&plain).unwrap().len();
    let gzipped_size = std::fs::metadata(&gzipped).unwrap().len();
    assert!(
        gzipped_size < plain_size / 2,
        "deflate should shrink a constant matrix: {gzipped_size} vs {plain_size}"
    );
    assert_eq!(nir_rs::io::read(&gzipped).unwrap(), graph);
}

#[test]
fn version_precedence_is_option_then_graph_then_default() {
    let dir = TempDir::new().unwrap();

    let mut graph = NirGraph::new();
    graph.version = Some("0.2.0".into());

    let from_graph = scratch(&dir, "from_graph.nir");
    nir_rs::io::write(&from_graph, &graph).unwrap();
    assert_eq!(nir_rs::io::read_version(&from_graph).unwrap(), "0.2.0");

    let overridden = scratch(&dir, "overridden.nir");
    nir_rs::io::write_with(
        &overridden,
        &graph,
        &WriteOptions::default().with_version("9.9.9"),
    )
    .unwrap();
    assert_eq!(nir_rs::io::read_version(&overridden).unwrap(), "9.9.9");

    let defaulted = scratch(&dir, "defaulted.nir");
    nir_rs::io::write(&defaulted, &NirGraph::new()).unwrap();
    assert_eq!(
        nir_rs::io::read_version(&defaulted).unwrap(),
        nir_rs::io::DEFAULT_NIR_VERSION
    );
}

// ---------------------------------------------------------------------------
// read → write → read on the Python-written fixtures
// ---------------------------------------------------------------------------

#[test]
fn fixtures_survive_read_write_read() {
    let dir = TempDir::new().unwrap();
    for name in FIXTURES {
        let original = nir_rs::io::read(format!("tests/fixtures/{name}")).unwrap();

        // One upstream fixture has an unresolved inner edge (see
        // hdf5_fixtures.rs); rewriting it needs the validation opt-out.
        let opts = WriteOptions::default().with_validation(false);
        let path = scratch(&dir, name);
        nir_rs::io::write_with(&path, &original, &opts)
            .unwrap_or_else(|e| panic!("writing {name}: {e}"));

        let decoded = nir_rs::io::read(&path).unwrap_or_else(|e| panic!("re-reading {name}: {e}"));
        assert_eq!(decoded, original, "{name} did not survive a round trip");
    }
}

#[test]
fn rewritten_fixture_keeps_its_original_version_string() {
    let dir = TempDir::new().unwrap();
    let original = nir_rs::io::read("tests/fixtures/lif_norse.nir").unwrap();
    let path = scratch(&dir, "lif_norse.nir");
    nir_rs::io::write(&path, &original).unwrap();

    assert_eq!(nir_rs::io::read_version(&path).unwrap(), "0.1.1");
}

#[test]
fn defaulted_v_reset_is_written_back_explicitly() {
    // `read` fills the absent v_reset with zeros, so the rewritten file has a
    // dataset the original lacked. That is the documented consequence of
    // matching Python's in-memory defaults, and the values still agree.
    let dir = TempDir::new().unwrap();
    let original = nir_rs::io::read("tests/fixtures/lif_norse.nir").unwrap();
    let path = scratch(&dir, "lif_norse.nir");
    nir_rs::io::write(&path, &original).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let lif = file.group("node/nodes/1").unwrap();
    assert!(
        lif.dataset("v_reset").is_ok(),
        "v_reset should now be present"
    );
    assert_eq!(nir_rs::io::read(&path).unwrap(), original);
}

#[test]
fn fixed_length_strings_from_other_producers_are_readable() {
    // This crate writes variable-length UTF-8, but h5py has historically
    // encoded lists of `str` as fixed-length bytes, so the reader accepts both.
    // Build such a file by hand, since our own writer cannot produce one.
    use hdf5::types::FixedAscii;

    let dir = TempDir::new().unwrap();
    let path = scratch(&dir, "fixed_strings.nir");
    {
        let ascii = |s: &str| FixedAscii::<8>::from_ascii(s.as_bytes()).unwrap();
        let file = hdf5::File::create(&path).unwrap();
        let version = file
            .new_dataset::<FixedAscii<8>>()
            .shape(())
            .create("version")
            .unwrap();
        version.write_scalar(&ascii("0.2.0")).unwrap();

        let root = file.create_group("node").unwrap();
        let ty = root
            .new_dataset::<FixedAscii<8>>()
            .shape(())
            .create("type")
            .unwrap();
        ty.write_scalar(&ascii("NIRGraph")).unwrap();

        let nodes = root.create_group("nodes").unwrap();
        for name in ["input", "output"] {
            let node = nodes.create_group(name).unwrap();
            let ty = node
                .new_dataset::<FixedAscii<8>>()
                .shape(())
                .create("type")
                .unwrap();
            ty.write_scalar(&ascii(if name == "input" { "Input" } else { "Output" }))
                .unwrap();
            let shape = node
                .new_dataset::<i64>()
                .shape([1])
                .create("shape")
                .unwrap();
            shape.write_raw(&[4i64]).unwrap();
        }

        let edges = root
            .new_dataset::<FixedAscii<8>>()
            .shape([1, 2])
            .create("edges")
            .unwrap();
        edges.write_raw(&[ascii("input"), ascii("output")]).unwrap();
    }

    let graph = nir_rs::io::read(&path).unwrap();
    assert_eq!(graph.version.as_deref(), Some("0.2.0"));
    assert_eq!(graph.get("input").unwrap().type_name(), "Input");
    assert_eq!(
        graph.edges,
        [("input".to_owned(), "output".to_owned())],
        "fixed-length names must not keep their NUL padding"
    );
    graph.validate_structure().unwrap();
}

#[test]
fn a_written_file_uses_the_upstream_layout() {
    let dir = TempDir::new().unwrap();
    let path = scratch(&dir, "layout.nir");
    nir_rs::io::write(&path, &graph_with_every_node_type()).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    assert!(file.dataset("version").is_ok(), "/version");
    let root = file.group("node").expect("/node");
    assert!(root.dataset("type").is_ok(), "/node/type");
    assert!(root.dataset("edges").is_ok(), "/node/edges");
    assert_eq!(root.dataset("edges").unwrap().shape()[1], 2);
    let nodes = root.group("nodes").expect("/node/nodes");
    assert!(nodes.group("lif").unwrap().dataset("v_threshold").is_ok());
    // A nested graph is a node group that itself carries nodes and edges.
    let sub = nodes.group("subgraph").unwrap();
    assert!(sub.group("nodes").is_ok());
    assert!(sub.dataset("edges").is_ok());
}
