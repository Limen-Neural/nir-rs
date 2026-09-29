// SPDX-License-Identifier: MIT OR Apache-2.0
//
//! Load/inspect tests for Synfire registry `.nir` fixtures.
//!
//! These files were pulled from the public Synfire registry with
//! `synfire pull <org>/<model>:<version>` (see
//! `tests/fixtures/synfire/README.md`). They are kept separate from the
//! paper corpus and the Hugging Face conversions so that registry-provenance
//! failures are distinguishable, and so CI never touches the network.
//!
//! A compatibility failure must identify the model, its pinned Synfire
//! version, and the failure class: `read` (malformed/unreadable artifact),
//! `unsupported` (valid file, unsupported wire family), or `drift`
//! (structural mismatch against the pinned inventory).

#![cfg(feature = "hdf5")]

use nir_rs::io::WriteOptions;
use nir_rs::{NirGraph, NirNode};
use tempfile::TempDir;

const DIR: &str = "tests/fixtures/synfire";

struct Spec {
    file: &'static str,
    /// `org/model:version` — always reported on failure.
    release: &'static str,
    nir_version: &'static str,
    nodes: usize,
    edges: usize,
    /// Sorted, deduplicated expected `type_name` inventory.
    types: &'static [&'static str],
}

const LIFNEURON: Spec = Spec {
    file: "lifneuron_1.0.0.nir",
    release: "pabogdan/lifneuron:1.0.0",
    nir_version: "0.1.1",
    nodes: 4,
    edges: 3,
    types: &["Affine", "Input", "LIF", "Output"],
};

const IFSYNFIRE: Spec = Spec {
    file: "ifsynfire_0.1.0.nir",
    release: "pabogdan/ifsynfire:0.1.0",
    nir_version: "1.0.7",
    nodes: 5,
    edges: 5,
    types: &["IF", "Input", "Linear", "Output"],
};

const NMNISTCNN: Spec = Spec {
    file: "nmnistcnn_1.0.0.nir",
    release: "pabogdan/nmnistcnn:1.0.0",
    nir_version: "0.2.0",
    nodes: 15,
    edges: 14,
    types: &[
        "Affine",
        "Conv2d",
        "Flatten",
        "IF",
        "Input",
        "Output",
        "SumPool2d",
    ],
};

const BRAILERNN: Spec = Spec {
    file: "brailernn_1.0.1.nir",
    release: "pabogdan/brailernn:1.0.1",
    nir_version: "0.2.0",
    nodes: 7,
    edges: 7,
    types: &["CubaLIF", "Input", "Linear", "Output"],
};

const SWAVELET: Spec = Spec {
    file: "swavelet_1.0.0.nir",
    release: "jegp/swavelet:1.0.0",
    nir_version: "1.0.7",
    nodes: 6,
    edges: 5,
    types: &["Affine", "Input", "LI", "LIF", "Output"],
};

const ALL: [Spec; 5] = [LIFNEURON, IFSYNFIRE, NMNISTCNN, BRAILERNN, SWAVELET];

fn read(spec: &Spec) -> NirGraph {
    nir_rs::io::read(format!("{DIR}/{}", spec.file)).unwrap_or_else(|e| {
        panic!("{} read failure: {e}", spec.release);
    })
}

fn type_inventory(graph: &NirGraph) -> Vec<&'static str> {
    let mut types: Vec<_> = graph.nodes.values().map(|n| n.type_name()).collect();
    types.sort_unstable();
    types.dedup();
    types
}

fn check(spec: &Spec) -> NirGraph {
    let g = read(spec);
    assert_eq!(
        g.version.as_deref(),
        Some(spec.nir_version),
        "{} drift: embedded /version",
        spec.release
    );
    assert_eq!(
        g.nodes.len(),
        spec.nodes,
        "{} drift: node count",
        spec.release
    );
    assert_eq!(
        g.edges.len(),
        spec.edges,
        "{} drift: edge count",
        spec.release
    );
    assert_eq!(
        type_inventory(&g),
        spec.types,
        "{} drift: wire-type inventory",
        spec.release
    );
    for node in g.nodes.values() {
        assert!(
            nir_rs::io::wire::is_wire_type(node.type_name()),
            "{} unsupported wire family: {}",
            spec.release,
            node.type_name()
        );
    }
    g.validate_structure()
        .unwrap_or_else(|e| panic!("{} read failure: validate_structure: {e}", spec.release));
    g
}

#[test]
fn every_synfire_fixture_loads_with_pinned_inventory() {
    for spec in &ALL {
        check(spec);
    }
}

#[test]
fn read_version_matches_embedded_version() {
    for spec in &ALL {
        let version = nir_rs::io::read_version(format!("{DIR}/{}", spec.file))
            .unwrap_or_else(|e| panic!("{} read failure: read_version: {e}", spec.release));
        assert_eq!(
            version, spec.nir_version,
            "{} drift: /version",
            spec.release
        );
    }
}

#[test]
fn synfire_fixtures_survive_read_write_read() {
    let dir = TempDir::new().unwrap();
    for spec in &ALL {
        let original = read(spec);
        let path = dir.path().join(spec.file);
        nir_rs::io::write_with(&path, &original, &WriteOptions::default())
            .unwrap_or_else(|e| panic!("{} read failure: writing: {e}", spec.release));
        let decoded = nir_rs::io::read(&path)
            .unwrap_or_else(|e| panic!("{} read failure: re-reading: {e}", spec.release));
        assert_eq!(decoded, original, "{} drift: round trip", spec.release);
    }
}

// ---------------------------------------------------------------------------
// Pinned topology assertions per release.
// ---------------------------------------------------------------------------

#[test]
fn lifneuron_topology() {
    let g = check(&LIFNEURON);
    assert_eq!(
        g.edges,
        [
            ("input".to_owned(), "0".to_owned()),
            ("0".to_owned(), "1".to_owned()),
            ("1".to_owned(), "output".to_owned()),
        ]
    );
    let NirNode::Input(input) = g.get("input").unwrap() else {
        panic!("{} drift: input node", LIFNEURON.release);
    };
    assert_eq!(input.shape, [1]);
    let NirNode::Lif(lif) = g.get("1").unwrap() else {
        panic!("{} drift: node 1 is not LIF", LIFNEURON.release);
    };
    assert_eq!(lif.tau.shape(), [1]);
    let NirNode::Output(output) = g.get("output").unwrap() else {
        panic!("{} drift: output node", LIFNEURON.release);
    };
    assert_eq!(output.shape, [1]);
}

#[test]
fn ifsynfire_topology() {
    let g = check(&IFSYNFIRE);
    assert_eq!(
        g.edges,
        [
            ("input".to_owned(), "synapses".to_owned()),
            ("synapses".to_owned(), "neurons".to_owned()),
            ("neurons".to_owned(), "recurrent_synapses".to_owned()),
            ("recurrent_synapses".to_owned(), "neurons".to_owned()),
            ("neurons".to_owned(), "output".to_owned()),
        ],
        "{} drift: synfire-chain wiring",
        IFSYNFIRE.release
    );
    let NirNode::If(neurons) = g.get("neurons").unwrap() else {
        panic!("{} drift: neurons is not IF", IFSYNFIRE.release);
    };
    assert_eq!(neurons.v_threshold.shape(), [50]);
    let NirNode::Linear(rec) = g.get("recurrent_synapses").unwrap() else {
        panic!(
            "{} drift: recurrent_synapses is not Linear",
            IFSYNFIRE.release
        );
    };
    assert_eq!(rec.weight.shape(), [50, 50]);
    let NirNode::Linear(fwd) = g.get("synapses").unwrap() else {
        panic!("{} drift: synapses is not Linear", IFSYNFIRE.release);
    };
    assert_eq!(fwd.weight.shape()[1], 50);
}

#[test]
fn nmnistcnn_topology() {
    let g = check(&NMNISTCNN);
    assert_eq!(g.nodes.len(), 15);
    assert_eq!(g.edges.len(), 14);
    let NirNode::Conv2d(conv0) = g.get("0").unwrap() else {
        panic!("{} drift: node 0 is not Conv2d", NMNISTCNN.release);
    };
    assert_eq!(conv0.weight.shape()[1], 2);
    let NirNode::Input(input) = g.get("input").unwrap() else {
        panic!("{} drift: input node", NMNISTCNN.release);
    };
    assert_eq!(input.shape, [2, 34, 34]);
    let NirNode::Output(output) = g.get("output").unwrap() else {
        panic!("{} drift: output node", NMNISTCNN.release);
    };
    assert_eq!(output.shape, [10]);
}

#[test]
fn brailernn_topology() {
    let g = check(&BRAILERNN);
    // Recurrent CubaLIF core: lif1.lif <-> lif1.w_rec.
    for (a, b) in [("lif1.w_rec", "lif1.lif"), ("lif1.lif", "lif1.w_rec")] {
        assert!(
            g.edges.iter().any(|(x, y)| x == a && y == b),
            "{} drift: missing recurrence edge {a} -> {b}",
            BRAILERNN.release
        );
    }
    let NirNode::CubaLif(lif1) = g.get("lif1.lif").unwrap() else {
        panic!("{} drift: lif1.lif is not CubaLIF", BRAILERNN.release);
    };
    assert_eq!(lif1.tau_mem.shape(), [40]);
    let NirNode::Input(input) = g.get("input").unwrap() else {
        panic!("{} drift: input node", BRAILERNN.release);
    };
    assert_eq!(input.shape, [12]);
    let NirNode::Output(output) = g.get("output").unwrap() else {
        panic!("{} drift: output node", BRAILERNN.release);
    };
    assert_eq!(output.shape, [7]);
}

#[test]
fn swavelet_topology() {
    let g = check(&SWAVELET);
    assert_eq!(
        g.edges,
        [
            ("input".to_owned(), "fanout".to_owned()),
            ("fanout".to_owned(), "li_stage_0".to_owned()),
            ("li_stage_0".to_owned(), "connectivity".to_owned()),
            ("connectivity".to_owned(), "lif".to_owned()),
            ("lif".to_owned(), "output".to_owned()),
        ]
    );
    let NirNode::Li(li) = g.get("li_stage_0").unwrap() else {
        panic!("{} drift: li_stage_0 is not LI", SWAVELET.release);
    };
    // 15 LI DoE channels drive the 16-unit LIF readout via connectivity (16×15).
    assert_eq!(li.tau.shape(), [15]);
    let NirNode::Affine(connectivity) = g.get("connectivity").unwrap() else {
        panic!("{} drift: connectivity is not Affine", SWAVELET.release);
    };
    assert_eq!(connectivity.weight.shape(), [16, 15]);
    let NirNode::Lif(lif) = g.get("lif").unwrap() else {
        panic!("{} drift: lif is not LIF", SWAVELET.release);
    };
    assert_eq!(lif.v_threshold.shape(), [16]);
    let NirNode::Output(output) = g.get("output").unwrap() else {
        panic!("{} drift: output node", SWAVELET.release);
    };
    assert_eq!(output.shape, [16]);
}
