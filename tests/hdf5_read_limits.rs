// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural node / edge / nested-graph budgets on [`ReadOptions`].

#![cfg(feature = "hdf5")]

mod common;
use common::{input, write_then};
use nir_rs::io::ReadOptions;
use nir_rs::nodes::Output;
use nir_rs::types::MetadataMap;
use nir_rs::{NirError, NirGraph, NirNode, ReadLimitResource};
use tempfile::TempDir;

fn output(shape: Vec<usize>) -> NirNode {
    NirNode::Output(Output {
        shape,
        metadata: MetadataMap::new(),
    })
}

fn write_graph(dir: &TempDir, name: &str, graph: &NirGraph) -> std::path::PathBuf {
    let path = dir.path().join(name);
    nir_rs::io::write(&path, graph).unwrap();
    path
}

fn assert_count_limit(
    result: Result<NirGraph, NirError>,
    resource: ReadLimitResource,
    limit: usize,
) -> NirError {
    let err = result.expect_err("bounded read should exceed its collection budget");
    match &err {
        NirError::ReadCountLimitExceeded {
            resource: got,
            limit: reported,
            used,
            requested,
            context,
        } => {
            assert_eq!(*got, resource, "context={context}");
            assert_eq!(*reported, limit, "context={context}");
            assert!(*requested > 0, "context={context}");
            assert!(
                used.checked_add(*requested)
                    .is_none_or(|total| total > limit),
                "overrun expected: used={used} requested={requested} limit={limit} context={context}"
            );
        }
        other => panic!("expected ReadCountLimitExceeded, got {other:?}"),
    }
    err
}

fn chain(node_count: usize) -> NirGraph {
    let mut graph = NirGraph::new();
    for i in 0..node_count {
        let name = format!("n{i}");
        if i == 0 {
            graph.insert_node(name, input(vec![1])).unwrap();
        } else {
            graph.insert_node(name, output(vec![1])).unwrap();
        }
    }
    for i in 0..node_count.saturating_sub(1) {
        graph.add_edge(format!("n{i}"), format!("n{}", i + 1));
    }
    graph
}

fn nested_pair() -> NirGraph {
    let mut inner = NirGraph::new();
    inner.insert_node("in", input(vec![1])).unwrap();
    inner.insert_node("out", output(vec![1])).unwrap();
    inner.add_edge("in", "out");

    let mut outer = NirGraph::new();
    outer.insert_node("input", input(vec![1])).unwrap();
    outer
        .insert_node("sub", NirNode::Graph(Box::new(inner)))
        .unwrap();
    outer.insert_node("output", output(vec![1])).unwrap();
    outer.add_edge("input", "sub");
    outer.add_edge("sub", "output");
    outer
}

#[test]
fn read_limit_exact_nodes_succeed_and_one_more_fails() {
    let dir = TempDir::new().unwrap();
    let path = write_graph(&dir, "nodes.nir", &chain(3));

    let exact = ReadOptions::default().with_max_nodes(Some(3));
    assert_eq!(nir_rs::io::read_with(&path, &exact).unwrap().len(), 3);

    let err = assert_count_limit(
        nir_rs::io::read_with(&path, &ReadOptions::default().with_max_nodes(Some(2))),
        ReadLimitResource::Nodes,
        2,
    );
    match err {
        NirError::ReadCountLimitExceeded {
            used,
            requested,
            context,
            ..
        } => {
            assert_eq!(used, 0, "collection is charged as a whole before decode");
            assert_eq!(requested, 3);
            assert!(context.contains("nodes"), "{context}");
        }
        other => panic!("expected ReadCountLimitExceeded, got {other:?}"),
    }
}

#[test]
fn read_limit_exact_edges_succeed_and_one_more_fails() {
    let dir = TempDir::new().unwrap();
    let path = write_graph(&dir, "edges.nir", &chain(3));

    let exact = ReadOptions::default().with_max_edges(Some(2));
    assert_eq!(nir_rs::io::read_with(&path, &exact).unwrap().edges.len(), 2);

    let err = assert_count_limit(
        nir_rs::io::read_with(&path, &ReadOptions::default().with_max_edges(Some(1))),
        ReadLimitResource::Edges,
        1,
    );
    match err {
        NirError::ReadCountLimitExceeded {
            used,
            requested,
            context,
            ..
        } => {
            assert_eq!(used, 0, "edge count is charged from shape before decode");
            assert_eq!(requested, 2);
            assert!(context.contains("edges"), "{context}");
        }
        other => panic!("expected ReadCountLimitExceeded, got {other:?}"),
    }
}

#[test]
fn read_limit_exact_nested_graphs_succeed_and_one_more_fails() {
    let dir = TempDir::new().unwrap();
    let path = write_graph(&dir, "nested.nir", &nested_pair());

    let exact = ReadOptions::default().with_max_nested_graphs(Some(2));
    let decoded = nir_rs::io::read_with(&path, &exact).unwrap();
    assert!(matches!(decoded.get("sub"), Some(NirNode::Graph(_))));

    let err = assert_count_limit(
        nir_rs::io::read_with(
            &path,
            &ReadOptions::default().with_max_nested_graphs(Some(1)),
        ),
        ReadLimitResource::NestedGraphs,
        1,
    );
    match err {
        NirError::ReadCountLimitExceeded {
            used,
            requested,
            context,
            ..
        } => {
            assert_eq!(used, 1);
            assert_eq!(requested, 1);
            assert_eq!(context, "/node/nodes/sub");
        }
        other => panic!("expected ReadCountLimitExceeded, got {other:?}"),
    }
}

#[test]
fn read_limit_nested_graph_is_charged_before_metadata() {
    // A rejected nested graph must not decode its metadata group first: a
    // large declared payload with only a count budget would otherwise hit
    // `max_bytes` before the structured graph-count error.
    let dir = TempDir::new().unwrap();
    let path = write_graph(&dir, "nested_meta.nir", &nested_pair());
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let md = file
            .group("node/nodes/sub")
            .unwrap()
            .create_group("metadata")
            .unwrap();
        md.new_dataset::<i64>()
            .shape([10_000_000])
            .chunk([1024])
            .create("blob")
            .unwrap();
    }

    let opts = ReadOptions::default()
        .with_max_bytes(Some(1_000_000))
        .with_max_nested_graphs(Some(1));
    let err = assert_count_limit(
        nir_rs::io::read_with(&path, &opts),
        ReadLimitResource::NestedGraphs,
        1,
    );
    match err {
        NirError::ReadCountLimitExceeded { context, .. } => {
            assert_eq!(context, "/node/nodes/sub");
        }
        other => panic!("expected ReadCountLimitExceeded, got {other:?}"),
    }
}

#[test]
fn read_limit_nested_nodes_and_edges_accumulate_globally() {
    let dir = TempDir::new().unwrap();
    let path = write_graph(&dir, "accumulate.nir", &nested_pair());
    // outer: input, sub, output (3) + inner: in, out (2) = 5 nodes
    // outer: 2 edges + inner: 1 edge = 3 edges
    // graphs: root + sub = 2

    nir_rs::io::read_with(&path, &ReadOptions::default().with_max_nodes(Some(5))).unwrap();
    assert_count_limit(
        nir_rs::io::read_with(&path, &ReadOptions::default().with_max_nodes(Some(4))),
        ReadLimitResource::Nodes,
        4,
    );

    nir_rs::io::read_with(&path, &ReadOptions::default().with_max_edges(Some(3))).unwrap();
    assert_count_limit(
        nir_rs::io::read_with(&path, &ReadOptions::default().with_max_edges(Some(2))),
        ReadLimitResource::Edges,
        2,
    );
}

#[test]
fn read_limit_hard_link_alias_does_not_bypass_the_budget() {
    let dir = TempDir::new().unwrap();
    let path = write_graph(&dir, "alias.nir", &nested_pair());
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        file.group("node/nodes")
            .unwrap()
            .link_hard("sub", "alias")
            .unwrap();
    }

    // Alias detection must fire even when the nested-graph budget would allow
    // a second expansion of the same object.
    let opts = ReadOptions::default()
        .with_max_nested_graphs(Some(16))
        .with_max_nodes(Some(10_000));
    let err = nir_rs::io::read_with(&path, &opts).unwrap_err();
    match err {
        NirError::InvalidGraph(message) => {
            assert!(
                message.contains("alias") || message.contains("cycle"),
                "got {message}"
            );
        }
        other => panic!("expected InvalidGraph for hard-link alias, got {other:?}"),
    }
}

#[test]
fn read_limit_default_options_still_load_fixtures() {
    for name in [
        "lif_norse.nir",
        "cnn_sinabs.nir",
        "braille_noDelay_bias_zero_subgraph.nir",
    ] {
        let path = format!("tests/fixtures/{name}");
        nir_rs::io::read(&path).unwrap_or_else(|e| panic!("{name} should still read: {e}"));
        nir_rs::io::read_with(&path, &ReadOptions::default())
            .unwrap_or_else(|e| panic!("{name} should load with default ReadOptions: {e}"));
        let modest = ReadOptions::default()
            .with_max_nodes(Some(10_000))
            .with_max_edges(Some(50_000))
            .with_max_nested_graphs(Some(64));
        nir_rs::io::read_with(&path, &modest)
            .unwrap_or_else(|e| panic!("{name} should fit modest structure budgets: {e}"));
    }
}

#[test]
fn read_limit_empty_edges_fit_a_zero_edge_budget() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "no_edges.nir", |file| {
        let root = file.group("node").unwrap();
        root.unlink("edges").unwrap();
        root.new_dataset::<f32>()
            .shape([0])
            .create("edges")
            .unwrap();
    });

    let graph = nir_rs::io::read_with(&path, &ReadOptions::default().with_max_edges(Some(0)))
        .expect("empty edges must fit a zero budget");
    assert!(graph.edges.is_empty());
}
