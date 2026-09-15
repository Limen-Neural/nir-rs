// SPDX-License-Identifier: MIT OR Apache-2.0

//! Load/inspect tests for Hugging Face–derived `.nir` fixtures.
//!
//! These files are **converted** from public HF SNN checkpoints with upstream
//! Python `nir.write` (see `tests/fixtures/huggingface/README.md`). They are
//! kept separate from the neuromorphs/NIR paper corpus so converter issues
//! cannot be mistaken for `nir-rs` parser bugs, and so Synfire work (#43)
//! does not grow a Hugging Face download surface in CI.

#![cfg(feature = "hdf5")]

use nir_rs::io::WriteOptions;
use nir_rs::nodes::Padding;
use nir_rs::types::{DType, TensorData};
use nir_rs::{NirGraph, NirNode};
use tempfile::TempDir;

const DIR: &str = "tests/fixtures/huggingface";

const MLP: &str = "neurocuda_mlp_mnist.nir";
const CNN: &str = "neurocuda_cnn_nmnist.nir";

const ALL_HF_FIXTURES: [&str; 2] = [MLP, CNN];

fn read(name: &str) -> NirGraph {
    nir_rs::io::read(format!("{DIR}/{name}")).unwrap_or_else(|e| panic!("reading {name}: {e}"))
}

fn type_names(graph: &NirGraph) -> Vec<(&str, &'static str)> {
    graph
        .nodes
        .iter()
        .map(|(name, node)| (name.as_str(), node.type_name()))
        .collect()
}

fn f32s(data: &TensorData) -> &[f32] {
    match data {
        TensorData::F32(v) => v,
        other => panic!("expected f32 data, found {:?}", other.dtype()),
    }
}

fn i64s(data: &TensorData) -> &[i64] {
    match data {
        TensorData::I64(v) => v,
        other => panic!("expected i64 data, found {:?}", other.dtype()),
    }
}

fn bits(x: f32) -> u32 {
    x.to_bits()
}

// ---------------------------------------------------------------------------
// neurocuda_mlp_mnist.nir — HF MLP 784→256 IF →256 IF →10
// ---------------------------------------------------------------------------

#[test]
fn mlp_mnist_structure_and_values() {
    let g = read(MLP);

    assert_eq!(g.version.as_deref(), Some("1.0.8"));
    assert_eq!(
        type_names(&g),
        [
            ("fc1", "Affine"),
            ("fc2", "Affine"),
            ("fc3", "Affine"),
            ("if1", "IF"),
            ("if2", "IF"),
            ("input", "Input"),
            ("output", "Output"),
        ]
    );
    assert_eq!(
        g.edges,
        [
            ("input".to_owned(), "fc1".to_owned()),
            ("fc1".to_owned(), "if1".to_owned()),
            ("if1".to_owned(), "fc2".to_owned()),
            ("fc2".to_owned(), "if2".to_owned()),
            ("if2".to_owned(), "fc3".to_owned()),
            ("fc3".to_owned(), "output".to_owned()),
        ]
    );
    g.validate_structure().unwrap();

    let NirNode::Input(input) = g.get("input").unwrap() else {
        panic!("expected Input");
    };
    assert_eq!(input.shape, [784]);

    let NirNode::Affine(fc1) = g.get("fc1").unwrap() else {
        panic!("expected Affine fc1");
    };
    assert_eq!(fc1.weight.shape(), [256, 784]);
    assert_eq!(fc1.weight.dtype(), DType::F32);
    assert_eq!(bits(f32s(fc1.weight.data())[0]), 0x3c80_0e7b);
    assert_eq!(fc1.bias.shape(), [256]);
    assert_eq!(bits(f32s(fc1.bias.data())[0]), 0xbc3c_2c5a);

    let NirNode::If(if1) = g.get("if1").unwrap() else {
        panic!("expected IF");
    };
    assert_eq!(if1.r.shape(), [256]);
    assert_eq!(f32s(if1.r.data())[0], 1.0);
    // QCFS threshold from Hub key relu1.thresh, broadcast to 256 units.
    assert_eq!(bits(f32s(if1.v_threshold.data())[0]), 0x409f_2a4d);
    let v_reset = if1.v_reset.as_ref().expect("v_reset present on this wire");
    assert_eq!(v_reset.shape(), [256]);
    assert_eq!(f32s(v_reset.data())[0], 0.0);

    let NirNode::If(if2) = g.get("if2").unwrap() else {
        panic!("expected IF");
    };
    assert_eq!(bits(f32s(if2.v_threshold.data())[0]), 0x4071_05cb);

    let NirNode::Affine(fc3) = g.get("fc3").unwrap() else {
        panic!("expected Affine fc3");
    };
    assert_eq!(fc3.weight.shape(), [10, 256]);
    assert_eq!(bits(f32s(fc3.weight.data())[0]), 0xbdb0_db12);

    let NirNode::Output(output) = g.get("output").unwrap() else {
        panic!("expected Output");
    };
    assert_eq!(output.shape, [10]);
}

// ---------------------------------------------------------------------------
// neurocuda_cnn_nmnist.nir — HF event-camera CNN
// ---------------------------------------------------------------------------

#[test]
fn cnn_nmnist_structure_and_values() {
    let g = read(CNN);

    assert_eq!(g.version.as_deref(), Some("1.0.8"));
    assert_eq!(
        type_names(&g),
        [
            ("conv1", "Conv2d"),
            ("conv2", "Conv2d"),
            ("conv3", "Conv2d"),
            ("fc", "Affine"),
            ("flatten", "Flatten"),
            ("if1", "IF"),
            ("if2", "IF"),
            ("if3", "IF"),
            ("input", "Input"),
            ("output", "Output"),
            ("pool1", "AvgPool2d"),
            ("pool2", "AvgPool2d"),
            ("pool3", "AvgPool2d"),
        ]
    );
    assert_eq!(g.edges.len(), 12);
    assert!(g.edges.contains(&("input".to_owned(), "conv1".to_owned())));
    assert!(g.edges.contains(&("conv1".to_owned(), "if1".to_owned())));
    assert!(g.edges.contains(&("if1".to_owned(), "pool1".to_owned())));
    assert!(
        g.edges
            .contains(&("pool3".to_owned(), "flatten".to_owned()))
    );
    assert!(g.edges.contains(&("flatten".to_owned(), "fc".to_owned())));
    assert!(g.edges.contains(&("fc".to_owned(), "output".to_owned())));
    g.validate_structure().unwrap();

    let NirNode::Input(input) = g.get("input").unwrap() else {
        panic!("expected Input");
    };
    assert_eq!(input.shape, [2, 34, 34]);

    let NirNode::Conv2d(conv1) = g.get("conv1").unwrap() else {
        panic!("expected Conv2d");
    };
    assert_eq!(conv1.weight.shape(), [32, 2, 5, 5]);
    assert_eq!(conv1.weight.dtype(), DType::F32);
    assert_eq!(bits(f32s(conv1.weight.data())[0]), 0xbdeb_d4a4);
    assert_eq!(conv1.stride, [1, 1]);
    assert_eq!(conv1.dilation, [1, 1]);
    assert_eq!(conv1.padding, Padding::pair(2, 2));
    assert_eq!(conv1.groups, 1);
    assert_eq!(conv1.bias.shape(), [32]);
    assert_eq!(
        conv1.input_shape.as_deref(),
        Some([34, 34].as_slice()),
        "spatial input used for Conv2d shape inference"
    );

    let NirNode::If(if1) = g.get("if1").unwrap() else {
        panic!("expected IF after conv1");
    };
    assert_eq!(if1.r.shape(), [32, 34, 34]);
    assert_eq!(if1.v_threshold.shape(), [32, 34, 34]);
    assert_eq!(bits(f32s(if1.v_threshold.data())[0]), 0x3ff1_b57e);

    let NirNode::AvgPool2d(pool1) = g.get("pool1").unwrap() else {
        panic!("expected AvgPool2d");
    };
    assert_eq!(i64s(pool1.kernel_size.data()), [2, 2]);
    assert_eq!(i64s(pool1.stride.data()), [2, 2]);
    assert_eq!(i64s(pool1.padding.data()), [0, 0]);

    let NirNode::Conv2d(conv2) = g.get("conv2").unwrap() else {
        panic!("expected Conv2d conv2");
    };
    assert_eq!(conv2.weight.shape(), [64, 32, 5, 5]);
    assert_eq!(conv2.input_shape.as_deref(), Some([17, 17].as_slice()));

    let NirNode::Conv2d(conv3) = g.get("conv3").unwrap() else {
        panic!("expected Conv2d conv3");
    };
    assert_eq!(conv3.weight.shape(), [128, 64, 3, 3]);
    assert_eq!(conv3.padding, Padding::pair(1, 1));
    assert_eq!(conv3.input_shape.as_deref(), Some([8, 8].as_slice()));

    let NirNode::Flatten(flatten) = g.get("flatten").unwrap() else {
        panic!("expected Flatten");
    };
    assert_eq!(flatten.start_dim, 0);
    assert_eq!(flatten.end_dim, -1);
    assert_eq!(
        flatten.input_type.as_deref(),
        Some([128, 4, 4].as_slice()),
        "128×4×4 = 2048 matches fc.weight"
    );

    let NirNode::Affine(fc) = g.get("fc").unwrap() else {
        panic!("expected Affine readout");
    };
    assert_eq!(fc.weight.shape(), [10, 2048]);
    assert_eq!(bits(f32s(fc.weight.data())[0]), 0x3ca9_ee50);

    let NirNode::Output(output) = g.get("output").unwrap() else {
        panic!("expected Output");
    };
    assert_eq!(output.shape, [10]);
}

#[test]
fn every_hf_fixture_node_is_a_known_wire_type() {
    for name in ALL_HF_FIXTURES {
        let g = read(name);
        for node in g.nodes.values() {
            assert!(
                nir_rs::io::wire::is_wire_type(node.type_name()),
                "{name}: {} is not a wire type",
                node.type_name()
            );
        }
    }
}

#[test]
fn every_hf_fixture_loads_and_validates() {
    for name in ALL_HF_FIXTURES {
        let g = read(name);
        assert!(!g.nodes.is_empty(), "{name} should have nodes");
        assert_eq!(
            g.version.as_deref(),
            Some("1.0.8"),
            "{name}: HF conversions are written by nir 1.0.8"
        );
        g.validate_structure()
            .unwrap_or_else(|e| panic!("{name} failed validate_structure: {e}"));
    }
}

#[test]
fn hf_fixtures_survive_read_write_read() {
    let dir = TempDir::new().unwrap();
    for name in ALL_HF_FIXTURES {
        let original = read(name);
        let path = dir.path().join(name);
        nir_rs::io::write_with(&path, &original, &WriteOptions::default())
            .unwrap_or_else(|e| panic!("writing {name}: {e}"));
        let decoded = nir_rs::io::read(&path).unwrap_or_else(|e| panic!("re-reading {name}: {e}"));
        assert_eq!(decoded, original, "{name} did not survive a round trip");
    }
}

#[test]
fn read_version_matches_graph_version() {
    for name in ALL_HF_FIXTURES {
        let version = nir_rs::io::read_version(format!("{DIR}/{name}")).unwrap();
        assert_eq!(version, "1.0.8");
        assert_eq!(read(name).version, Some(version));
    }
}
