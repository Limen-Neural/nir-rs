// SPDX-License-Identifier: MIT OR Apache-2.0

//! Load / topology / inventory / round-trip tests for the vendored Synfire
//! registry corpus (GitHub #43 / LIM-1085).
//!
//! These tests use **no network, no registry, and no authentication**: they
//! read only the committed `.nir` bytes under `tests/fixtures/synfire/`. Every
//! asserted value (versions, node/edge counts, inventories, ordered edges,
//! selected weight shapes) is taken verbatim from the observed ground truth
//! recorded in `MANIFEST.toml`.
//!
//! `nir_rs::io::read` returns nodes in **sorted name order**, so all node-name
//! sequences below are asserted in that sorted order (e.g. nmnistcnn:
//! `0, 1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9, input, output`).

#![cfg(feature = "hdf5")]

mod common;

use common::synfire_manifest::{self, SynfireRecord, inventory, label};
use nir_rs::io::WriteOptions;
use nir_rs::{NirGraph, NirNode};
use std::fs;
use tempfile::TempDir;

const DIR: &str = "tests/fixtures/synfire";
const MANIFEST: &str = "tests/fixtures/synfire/MANIFEST.toml";

const LIFNEURON: &str = "lifneuron-1.0.0.nir";
const IFSYNFIRE: &str = "ifsynfire-0.1.0.nir";
const NMNISTCNN: &str = "nmnistcnn-1.0.0.nir";
const BRAILERNN: &str = "brailernn-1.0.1.nir";
const SWAVELET: &str = "swavelet-1.0.0.nir";

const ALL_SYNFIRE_FIXTURES: [&str; 5] = [LIFNEURON, IFSYNFIRE, NMNISTCNN, BRAILERNN, SWAVELET];

fn read(name: &str) -> NirGraph {
    nir_rs::io::read(format!("{DIR}/{name}")).unwrap_or_else(|e| panic!("reading {name}: {e}"))
}

fn records() -> Vec<SynfireRecord> {
    let text = fs::read_to_string(MANIFEST).unwrap_or_else(|e| panic!("reading {MANIFEST}: {e}"));
    synfire_manifest::read_manifest(&text)
}

/// Sorted `(name, wire type)` pairs, matching the `read` sort order.
fn type_names(graph: &NirGraph) -> Vec<(&str, &'static str)> {
    graph
        .nodes
        .iter()
        .map(|(name, node)| (name.as_str(), node.type_name()))
        .collect()
}

/// Convert `&[(&str, &str)]` edge expectations into owned pairs for comparison.
fn edges(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect()
}

// ---------------------------------------------------------------------------
// Manifest-driven loop over every record.
// ---------------------------------------------------------------------------

#[test]
fn manifest_records_load_and_match_observed_facts() {
    for rec in records() {
        match rec.status.as_str() {
            "supported" => assert_supported(&rec),
            // The following branches do not fire in the all-`supported`
            // corpus, but exist for a future re-pull per the four-way scheme.
            "unsupported-valid" => assert_unsupported_valid(&rec),
            "malformed" => assert_malformed(&rec),
            "inaccessible" => assert_inaccessible(&rec),
            other => panic!("{} has an unknown status {other:?}", label(&rec)),
        }
    }
}

/// Assert a `supported` record: the committed file loads, validates, and its
/// observed topology matches every value recorded in the manifest.
fn assert_supported(rec: &SynfireRecord) {
    let l = label(rec);
    let file = rec
        .file
        .as_ref()
        .unwrap_or_else(|| panic!("{l} supported record is missing a file key"));
    let g = read(file);
    g.validate_structure()
        .unwrap_or_else(|e| panic!("{l} failed validate_structure: {e}"));

    assert_eq!(
        g.version.as_deref(),
        rec.nir_version.as_deref(),
        "{l} embedded /version mismatch"
    );
    assert_eq!(
        Some(g.nodes.len()),
        rec.node_count,
        "{l} node_count mismatch"
    );

    assert_nested_graphs(rec, &g, &l);

    assert_eq!(
        Some(g.edges.len()),
        rec.edge_count,
        "{l} edge_count mismatch"
    );
    assert_eq!(
        inventory(&g),
        rec.node_types,
        "{l} node_types inventory mismatch"
    );
}

/// Assert the nested-subgraph facts for a `supported` record.
///
/// No nested subgraphs exist in the current corpus, but assert each nested
/// graph's counts + inventory if a future re-pull adds one (the record must
/// then carry `nested` / `nested_node_types`). Track which declared nested
/// names are actually found on load so that declared-and-found graphs pass
/// while a stale manifest entry (declared but never found) still fails below.
fn assert_nested_graphs(rec: &SynfireRecord, g: &NirGraph, l: &str) {
    let mut pending_nested: std::collections::BTreeSet<&str> =
        rec.nested.iter().map(String::as_str).collect();
    for (name, node) in &g.nodes {
        if let NirNode::Graph(sub) = node {
            check_nested_node(rec, name, sub, l);
            pending_nested.remove(name.as_str());
        }
    }
    assert!(
        pending_nested.is_empty(),
        "{l} declares nested graphs {pending_nested:?} that were not found on load"
    );
}

/// Verify one nested subgraph found on load against the record's declarations.
fn check_nested_node(rec: &SynfireRecord, name: &str, sub: &NirGraph, l: &str) {
    assert!(
        rec.nested.iter().any(|n| n == name),
        "{l} has undeclared nested graph {name:?}"
    );
    assert!(
        !rec.nested_node_types.is_empty(),
        "{l} nested graph {name:?} present but nested_node_types is empty"
    );
    assert_eq!(
        inventory(sub),
        rec.nested_node_types,
        "{l} nested graph {name:?} inventory mismatch"
    );
}

/// Assert an `unsupported-valid` record: the committed file reads back a
/// `UnknownNodeType` error whose message carries the recorded fragment.
fn assert_unsupported_valid(rec: &SynfireRecord) {
    let l = label(rec);
    let file = rec
        .file
        .as_ref()
        .unwrap_or_else(|| panic!("{l} unsupported-valid record is missing a file"));
    let err =
        nir_rs::io::read(format!("{DIR}/{file}")).expect_err(&format!("{l} expected read to fail"));
    assert_eq!(
        synfire_manifest::nir_error_variant_name(&err),
        "UnknownNodeType",
        "{l} unsupported-valid must fail with UnknownNodeType"
    );
    assert_eq!(
        rec.error_class.as_deref(),
        Some("UnknownNodeType"),
        "{l} unsupported-valid must record error_class = UnknownNodeType"
    );
    let fragment = rec
        .error_fragment
        .as_ref()
        .unwrap_or_else(|| panic!("{l} unsupported-valid is missing error_fragment"));
    assert!(
        err.to_string().contains(fragment.as_str()),
        "{l} error string must contain {fragment:?}, got {err}"
    );
}

/// Assert a `malformed` record: reading or validating the committed file fails
/// with the recorded error variant.
fn assert_malformed(rec: &SynfireRecord) {
    let l = label(rec);
    let file = rec
        .file
        .as_ref()
        .unwrap_or_else(|| panic!("{l} malformed record is missing a file"));
    let expected = rec
        .error_class
        .as_ref()
        .unwrap_or_else(|| panic!("{l} malformed record is missing error_class"));
    let err = match nir_rs::io::read(format!("{DIR}/{file}")) {
        Ok(g) => g
            .validate_structure()
            .expect_err(&format!("{l} expected malformed file to fail")),
        Err(e) => e,
    };
    assert_eq!(
        synfire_manifest::nir_error_variant_name(&err),
        expected.as_str(),
        "{l} malformed error variant mismatch"
    );
}

/// Assert an `inaccessible` record: if it names a file, that file must not be
/// committed to the fixture directory.
fn assert_inaccessible(rec: &SynfireRecord) {
    let l = label(rec);
    if let Some(file) = &rec.file {
        let path = format!("{DIR}/{file}");
        assert!(
            !std::path::Path::new(&path).exists(),
            "{l} inaccessible record must not commit a file, found {path}"
        );
    }
}

// ---------------------------------------------------------------------------
// Detailed per-model tests. Every value is from GROUND_TRUTH.md / MANIFEST.toml.
// ---------------------------------------------------------------------------

#[test]
fn lifneuron_structure() {
    let g = read(LIFNEURON);
    g.validate_structure().unwrap();
    assert_eq!(g.version.as_deref(), Some("0.1.1"));
    assert_eq!(
        type_names(&g),
        [
            ("0", "Affine"),
            ("1", "LIF"),
            ("input", "Input"),
            ("output", "Output"),
        ]
    );
    assert_eq!(
        g.edges,
        edges(&[("input", "0"), ("0", "1"), ("1", "output")])
    );

    let NirNode::Affine(a) = g.get("0").unwrap() else {
        panic!("expected Affine at 0");
    };
    assert_eq!(inventory(&g), ["Affine:1", "Input:1", "LIF:1", "Output:1"]);
    assert_eq!(a.weight.shape(), [1, 1]);
    assert_eq!(a.bias.shape(), [1]);
}

#[test]
fn ifsynfire_structure_is_single_if_recurrent_loop() {
    let g = read(IFSYNFIRE);
    g.validate_structure().unwrap();
    assert_eq!(g.version.as_deref(), Some("1.0.7"));
    assert_eq!(
        type_names(&g),
        [
            ("input", "Input"),
            ("neurons", "IF"),
            ("output", "Output"),
            ("recurrent_synapses", "Linear"),
            ("synapses", "Linear"),
        ]
    );
    // The "synfire chain of 50 IF neurons" is a SINGLE IF population of shape
    // [50] with a recurrent Linear feeding back into it — not a multi-IF chain.
    assert_eq!(
        g.edges,
        edges(&[
            ("input", "synapses"),
            ("synapses", "neurons"),
            ("neurons", "recurrent_synapses"),
            ("recurrent_synapses", "neurons"),
            ("neurons", "output"),
        ])
    );
    let NirNode::If(neurons) = g.get("neurons").unwrap() else {
        panic!("expected IF at neurons");
    };
    assert_eq!(inventory(&g), ["IF:1", "Input:1", "Linear:2", "Output:1"]);
    assert_eq!(neurons.v_threshold.shape(), [50]);

    let NirNode::Linear(rec) = g.get("recurrent_synapses").unwrap() else {
        panic!("expected Linear at recurrent_synapses");
    };
    assert_eq!(rec.weight.shape(), [50, 50]);
}

/// Full ordered wire-type inventory of `nmnistcnn-1.0.0.nir` in `read` sort
/// order (`0, 1, 10, 11, 12, 2, 3, ...`).
fn nmnistcnn_type_names() -> [(&'static str, &'static str); 15] {
    [
        ("0", "Conv2d"),
        ("1", "IF"),
        ("10", "IF"),
        ("11", "Affine"),
        ("12", "IF"),
        ("2", "Conv2d"),
        ("3", "IF"),
        ("4", "SumPool2d"),
        ("5", "Conv2d"),
        ("6", "IF"),
        ("7", "SumPool2d"),
        ("8", "Flatten"),
        ("9", "Affine"),
        ("input", "Input"),
        ("output", "Output"),
    ]
}

/// Selected weight shapes + operator counts (substitution detection).
fn nmnistcnn_shapes_and_operators(g: &NirGraph) {
    let NirNode::Conv2d(c0) = g.get("0").unwrap() else {
        panic!("expected Conv2d at 0");
    };
    assert_eq!(c0.weight.shape(), [16, 2, 5, 5]);
    let NirNode::Conv2d(c2) = g.get("2").unwrap() else {
        panic!("expected Conv2d at 2");
    };
    assert_eq!(c2.weight.shape(), [16, 16, 3, 3]);
    let NirNode::Conv2d(c5) = g.get("5").unwrap() else {
        panic!("expected Conv2d at 5");
    };
    assert_eq!(c5.weight.shape(), [8, 16, 3, 3]);
    let NirNode::Affine(a11) = g.get("11").unwrap() else {
        panic!("expected Affine at 11");
    };
    assert_eq!(a11.weight.shape(), [10, 256]);

    // Operator presence + counts. Pooling is SumPool2d (NOT AvgPool2d).
    let count = |ty: &str| g.nodes.values().filter(|n| n.type_name() == ty).count();
    assert_eq!(
        [
            count("Conv2d"),
            count("SumPool2d"),
            count("AvgPool2d"),
            count("Flatten")
        ],
        [3, 2, 0, 1],
        "operator counts mismatch (Conv2d, SumPool2d, AvgPool2d, Flatten)"
    );
}

#[test]
fn nmnistcnn_structure_and_selected_shapes() {
    let g = read(NMNISTCNN);
    g.validate_structure().unwrap();
    assert_eq!(g.version.as_deref(), Some("0.2.0"));
    assert_eq!(type_names(&g), nmnistcnn_type_names());

    nmnistcnn_shapes_and_operators(&g);

    assert_eq!(
        g.edges,
        edges(&[
            ("8", "9"),
            ("11", "12"),
            ("5", "6"),
            ("0", "1"),
            ("9", "10"),
            ("12", "output"),
            ("4", "5"),
            ("3", "4"),
            ("2", "3"),
            ("input", "0"),
            ("6", "7"),
            ("1", "2"),
            ("7", "8"),
            ("10", "11"),
        ])
    );
    assert_eq!(
        inventory(&g),
        [
            "Affine:2",
            "Conv2d:3",
            "Flatten:1",
            "IF:5",
            "Input:1",
            "Output:1",
            "SumPool2d:2"
        ]
    );

    let NirNode::Input(input) = g.get("input").unwrap() else {
        panic!("expected Input");
    };
    assert_eq!(input.shape, [2, 34, 34]);
}

#[test]
fn brailernn_structure_and_recurrent_pair() {
    let g = read(BRAILERNN);
    g.validate_structure().unwrap();
    assert_eq!(g.version.as_deref(), Some("0.2.0"));
    assert_eq!(
        type_names(&g),
        [
            ("fc1", "Linear"),
            ("fc2", "Linear"),
            ("input", "Input"),
            ("lif1.lif", "CubaLIF"),
            ("lif1.w_rec", "Linear"),
            ("lif2", "CubaLIF"),
            ("output", "Output"),
        ]
    );
    assert_eq!(
        g.edges,
        edges(&[
            ("lif1.w_rec", "lif1.lif"),
            ("fc2", "lif2"),
            ("lif1.lif", "lif1.w_rec"),
            ("lif2", "output"),
            ("input", "fc1"),
            ("fc1", "lif1.lif"),
            ("lif1.lif", "fc2"),
        ])
    );
    // The exact-edge assertion above already covers the recurrent loop
    // lif1.lif <-> lif1.w_rec.
    let NirNode::Linear(w_rec) = g.get("lif1.w_rec").unwrap() else {
        panic!("expected Linear at lif1.w_rec");
    };
    assert_eq!(
        inventory(&g),
        ["CubaLIF:2", "Input:1", "Linear:3", "Output:1"]
    );
    assert_eq!(w_rec.weight.shape(), [40, 40]);
}

#[test]
fn swavelet_structure_is_li_lif_chain() {
    let g = read(SWAVELET);
    g.validate_structure().unwrap();
    assert_eq!(g.version.as_deref(), Some("1.0.7"));
    assert_eq!(
        type_names(&g),
        [
            ("connectivity", "Affine"),
            ("fanout", "Affine"),
            ("input", "Input"),
            ("li_stage_0", "LI"),
            ("lif", "LIF"),
            ("output", "Output"),
        ]
    );
    // Chain: Input -> fanout -> li_stage_0 -> connectivity -> lif -> Output.
    assert_eq!(
        g.edges,
        edges(&[
            ("input", "fanout"),
            ("fanout", "li_stage_0"),
            ("li_stage_0", "connectivity"),
            ("connectivity", "lif"),
            ("lif", "output"),
        ])
    );
    let NirNode::Affine(fanout) = g.get("fanout").unwrap() else {
        panic!("expected Affine at fanout");
    };
    assert_eq!(fanout.weight.shape(), [15, 1]);
    assert_eq!(fanout.bias.shape(), [15]);
    let NirNode::Affine(conn) = g.get("connectivity").unwrap() else {
        panic!("expected Affine at connectivity");
    };
    assert_eq!(
        inventory(&g),
        ["Affine:2", "Input:1", "LI:1", "LIF:1", "Output:1"]
    );
    assert_eq!(conn.weight.shape(), [16, 15]);
    assert_eq!(conn.bias.shape(), [16]);
}

// ---------------------------------------------------------------------------
// Corpus-wide invariants.
// ---------------------------------------------------------------------------

#[test]
fn every_synfire_fixture_node_is_a_known_wire_type() {
    for name in ALL_SYNFIRE_FIXTURES {
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
fn synfire_fixtures_survive_read_write_read() {
    let dir = TempDir::new().unwrap();
    for name in ALL_SYNFIRE_FIXTURES {
        let original = read(name);
        let path = dir.path().join(name);
        nir_rs::io::write_with(&path, &original, &WriteOptions::default())
            .unwrap_or_else(|e| panic!("writing {name}: {e}"));
        let decoded = nir_rs::io::read(&path).unwrap_or_else(|e| panic!("re-reading {name}: {e}"));
        assert_eq!(decoded, original, "{name} did not survive a round trip");
    }
}
