// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unknown-field policy on HDF5 read (LIM-1312).
//!
//! Python `from_dict` rejects an extra dataset on a node; the default reader
//! drops it. [`UnknownFieldPolicy::Reject`] closes that gap on request.

#![cfg(feature = "hdf5")]

mod common;
use hdf5::types::VarLenUnicode;
use nir_rs::io::{ReadOptions, UnknownFieldPolicy};
use nir_rs::nodes::Lif;
use nir_rs::{NirError, NirGraph, NirNode, Tensor};
use tempfile::TempDir;

fn lif() -> NirNode {
    let t = |v: f32| Tensor::from_f32([1], vec![v]).unwrap();
    NirNode::Lif(Lif {
        tau: t(1.0),
        r: t(1.0),
        v_leak: t(0.0),
        v_reset: Some(t(0.0)),
        v_threshold: t(1.0),
        metadata: Default::default(),
    })
}

fn strict() -> ReadOptions {
    ReadOptions::default().with_unknown_fields(UnknownFieldPolicy::Reject)
}

/// Write `graph`, then add a scalar dataset `field` under `group_path`.
fn write_with_extra(
    dir: &TempDir,
    graph: &NirGraph,
    group_path: &str,
    field: &str,
) -> std::path::PathBuf {
    let path = dir.path().join("extra.nir");
    nir_rs::io::write(&path, graph).unwrap();
    let file = hdf5::File::open_rw(&path).unwrap();
    file.group(group_path)
        .unwrap()
        .new_dataset_builder()
        .with_data(&[42.0f64])
        .create(field)
        .unwrap();
    drop(file);
    path
}

fn lif_graph() -> NirGraph {
    let mut g = NirGraph::new();
    g.insert_node("lif", lif()).unwrap();
    g
}

fn assert_unknown(err: NirError, ctx: &str, field: &str) {
    match err {
        NirError::UnknownField { context, field: f } => {
            assert_eq!((context.as_str(), f.as_str()), (ctx, field));
        }
        other => panic!("expected UnknownField, got {other:?}"),
    }
}

#[test]
fn default_policy_ignores_extra_dataset_on_lif() {
    let dir = TempDir::new().unwrap();
    let path = write_with_extra(&dir, &lif_graph(), "node/nodes/lif", "mystery");
    assert_eq!(
        ReadOptions::default().unknown_fields,
        UnknownFieldPolicy::Ignore
    );
    let graph = nir_rs::io::read(&path).unwrap();
    assert_eq!(graph.len(), 1);
}

#[test]
fn reject_policy_fails_on_extra_dataset_on_lif() {
    let dir = TempDir::new().unwrap();
    let path = write_with_extra(&dir, &lif_graph(), "node/nodes/lif", "mystery");
    let err = nir_rs::io::read_with(&path, &strict()).unwrap_err();
    assert_unknown(err, "lif", "mystery");
}

#[test]
fn reject_policy_accepts_a_clean_file_and_defaults() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("clean.nir");
    nir_rs::io::write(&path, &lif_graph()).unwrap();
    nir_rs::io::read_with(&path, &strict()).unwrap();
}

#[test]
fn extra_metadata_keys_stay_allowed_under_reject() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("md.nir");
    nir_rs::io::write(&path, &lif_graph()).unwrap();
    let file = hdf5::File::open_rw(&path).unwrap();
    let node = file.group("node/nodes/lif").unwrap();
    let md = node
        .group("metadata")
        .or_else(|_| node.create_group("metadata"))
        .unwrap();
    let ds = md
        .new_dataset::<VarLenUnicode>()
        .shape(())
        .create("anything_goes")
        .unwrap();
    ds.write_scalar(&"hello".parse::<VarLenUnicode>().unwrap())
        .unwrap();
    drop(file);

    let graph = nir_rs::io::read_with(&path, &strict()).unwrap();
    let NirNode::Lif(l) = graph.get("lif").unwrap() else {
        panic!("lif")
    };
    assert!(l.metadata.contains_key("anything_goes"));
}

#[test]
fn reject_policy_recurses_into_nested_graphs() {
    let mut inner = NirGraph::new();
    inner.insert_node("lif", lif()).unwrap();
    let mut outer = NirGraph::new();
    outer
        .insert_node("sub", NirNode::Graph(Box::new(inner)))
        .unwrap();

    let dir = TempDir::new().unwrap();
    let path = write_with_extra(&dir, &outer, "node/nodes/sub/nodes/lif", "mystery");
    nir_rs::io::read(&path).unwrap();
    let err = nir_rs::io::read_with(&path, &strict()).unwrap_err();
    assert_unknown(err, "lif", "mystery");

    // An extra member on the nested graph group itself.
    let path = write_with_extra(&dir, &outer, "node/nodes/sub", "mystery");
    nir_rs::io::read(&path).unwrap();
    let err = nir_rs::io::read_with(&path, &strict()).unwrap_err();
    assert_unknown(err, "/node/nodes/sub", "mystery");
}

#[test]
fn reject_policy_applies_to_the_root_graph() {
    let dir = TempDir::new().unwrap();
    let path = write_with_extra(&dir, &lif_graph(), "node", "mystery");
    nir_rs::io::read(&path).unwrap();
    let err = nir_rs::io::read_with(&path, &strict()).unwrap_err();
    assert_unknown(err, "/node", "mystery");
}

#[test]
fn reject_policy_accepts_every_vendored_python_fixture() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut seen = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "nir") {
            nir_rs::io::read_with(&path, &strict())
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            seen += 1;
        }
    }
    assert!(seen > 0);
}
