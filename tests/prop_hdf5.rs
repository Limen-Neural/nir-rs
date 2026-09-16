// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property tests at the HDF5 write → read boundary (graph-level fidelity).
//!
//! Structured generation only — no blind byte mutation of HDF5 containers.
//! Hostile layout cases stay in `hdf5_untrusted.rs`. Requires `--features hdf5`.

#![cfg(feature = "hdf5")]

use nir_rs::io::wire::check_link_name;
use nir_rs::io::{DEFAULT_NIR_VERSION, ReadOptions, WriteOptions};
use nir_rs::nodes::{Input, Linear, Output};
use nir_rs::types::Tensor;
use nir_rs::{NirGraph, NirNode};
use proptest::prelude::*;
use tempfile::TempDir;

/// Writers fill absent `NirGraph::version` with [`DEFAULT_NIR_VERSION`].
fn with_default_version(mut graph: NirGraph) -> NirGraph {
    if graph.version.is_none() {
        graph.version = Some(DEFAULT_NIR_VERSION.into());
    }
    graph
}

fn input(shape: Vec<usize>) -> NirNode {
    NirNode::Input(Input {
        shape,
        metadata: Default::default(),
    })
}

fn output(shape: Vec<usize>) -> NirNode {
    NirNode::Output(Output {
        shape,
        metadata: Default::default(),
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Small valid graphs survive write → read at graph equality.
    #[test]
    fn valid_io_chain_round_trips(
        width in 1usize..4,
        compress in prop::option::of(0u8..5),
    ) {
        let mut g = NirGraph::new();
        g.insert_node("input", input(vec![width])).unwrap();
        g.insert_node("output", output(vec![width])).unwrap();
        g.add_edge("input", "output");
        g.validate_structure().unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("prop.nir");
        let opts = WriteOptions::default().with_compression(compress);
        nir_rs::io::write_with(&path, &g, &opts).unwrap();
        let decoded = nir_rs::io::read(&path).unwrap();
        assert_eq!(decoded.nodes.len(), g.nodes.len());
        assert_eq!(decoded.edges, g.edges);
        decoded.validate_structure().unwrap();
    }

    /// Linear + Input/Output with a small weight matrix round-trips values.
    #[test]
    fn linear_weight_values_round_trip(
        rows in 1usize..3,
        cols in 1usize..3,
        seed in any::<u16>(),
    ) {
        let n = rows * cols;
        let data: Vec<f32> = (0..n)
            .map(|i| ((seed as u32).wrapping_add(i as u32) % 50) as f32 * 0.1)
            .collect();
        let weight = Tensor::from_f32([rows, cols], data).unwrap();

        let mut g = NirGraph::new();
        g.insert_node("input", input(vec![cols])).unwrap();
        g.insert_node(
            "lin",
            NirNode::Linear(Linear {
                weight,
                metadata: Default::default(),
            }),
        )
        .unwrap();
        g.insert_node("output", output(vec![rows])).unwrap();
        g.add_edge("input", "lin");
        g.add_edge("lin", "output");

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("lin.nir");
        nir_rs::io::write(&path, &g).unwrap();
        let decoded = nir_rs::io::read(&path).unwrap();
        assert_eq!(decoded, with_default_version(g));
    }

    /// Illegal link names fail **before** the destination path is replaced.
    #[test]
    fn illegal_node_name_does_not_clobber_destination(
        bad in prop_oneof![
            Just(String::new()),
            Just("a/b".into()),
            Just(".".into()),
            Just("..".into()),
            Just("x\0y".into()),
        ],
    ) {
        assert!(check_link_name("node name", &bad).is_err());

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("keep.nir");

        // Seed a good file at the destination.
        let mut good = NirGraph::new();
        good.insert_node("input", input(vec![1])).unwrap();
        good.insert_node("output", output(vec![1])).unwrap();
        good.add_edge("input", "output");
        nir_rs::io::write(&path, &good).unwrap();
        let before = std::fs::read(&path).unwrap();

        let mut bad_graph = NirGraph::new();
        bad_graph.insert_node(bad, input(vec![1])).unwrap();
        let err = nir_rs::io::write(&path, &bad_graph).unwrap_err();
        assert!(
            matches!(err, nir_rs::NirError::InvalidGraph(_)),
            "expected preflight InvalidGraph, got {err:?}"
        );
        let after = std::fs::read(&path).unwrap();
        assert_eq!(before, after, "destination must be unchanged after preflight reject");
    }

    /// Nested subgraphs charge the same node/edge budgets as the root.
    #[test]
    fn read_limit_nested_counts_are_global(
        inner_nodes in 1usize..4,
        extra_root in 0usize..3,
    ) {
        let mut inner = NirGraph::new();
        for i in 0..inner_nodes {
            inner.insert_node(format!("i{i}"), input(vec![1])).unwrap();
        }
        for i in 0..inner_nodes.saturating_sub(1) {
            inner.add_edge(format!("i{i}"), format!("i{}", i + 1));
        }

        let mut outer = NirGraph::new();
        outer
            .insert_node("sub", NirNode::Graph(Box::new(inner)))
            .unwrap();
        for i in 0..extra_root {
            outer.insert_node(format!("r{i}"), output(vec![1])).unwrap();
            outer.add_edge("sub", format!("r{i}"));
        }

        let total_nodes = 1 + inner_nodes + extra_root;
        let total_edges = inner_nodes.saturating_sub(1) + extra_root;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested.nir");
        nir_rs::io::write(&path, &outer).unwrap();

        let ok = ReadOptions::default()
            .with_max_nodes(Some(total_nodes))
            .with_max_edges(Some(total_edges))
            .with_max_nested_graphs(Some(2));
        nir_rs::io::read_with(&path, &ok).unwrap();

        let is_count_limit = |err: &nir_rs::NirError| {
            matches!(err, nir_rs::NirError::ReadCountLimitExceeded { .. })
        };

        if total_nodes > 0 {
            let err = nir_rs::io::read_with(
                &path,
                &ReadOptions::default().with_max_nodes(Some(total_nodes - 1)),
            )
            .unwrap_err();
            prop_assert!(is_count_limit(&err), "nodes: {err:?}");
        }
        if total_edges > 0 {
            let err = nir_rs::io::read_with(
                &path,
                &ReadOptions::default().with_max_edges(Some(total_edges - 1)),
            )
            .unwrap_err();
            prop_assert!(is_count_limit(&err), "edges: {err:?}");
        }
        let err = nir_rs::io::read_with(
            &path,
            &ReadOptions::default().with_max_nested_graphs(Some(1)),
        )
        .unwrap_err();
        prop_assert!(is_count_limit(&err), "graphs: {err:?}");
    }
}
