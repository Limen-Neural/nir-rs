// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic property tests for pure-Rust graph / tensor invariants.
//!
//! These run in normal CI (`cargo test`) with a fixed case budget. Structured
//! fuzz harnesses live under `fuzz/` and are optional / nightly-only — see
//! [`TESTING.md`](https://github.com/Limen-Neural/nir-rs/blob/main/TESTING.md).

use nir_rs::io::wire::{check_hdf5_string, check_link_name};
use nir_rs::nodes::{Input, Output};
use nir_rs::types::{Tensor, TensorData};
use nir_rs::{NirError, NirGraph, NirNode};
use proptest::prelude::*;

/// Checked shape product; empty rank is a scalar (1), matching [`Tensor`].
fn shape_product(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        Some(1)
    } else {
        shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
    }
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

// ---------------------------------------------------------------------------
// Tensor construction
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `Tensor::new` succeeds iff `len == product(shape)` (with overflow → Err).
    #[test]
    fn tensor_new_enforces_shape_data_len(
        shape in prop::collection::vec(0usize..12, 0..5),
        len in 0usize..64,
    ) {
        let data = TensorData::F32(vec![0.0; len]);
        let expected = shape_product(&shape);
        match Tensor::new(shape.clone(), data) {
            Ok(t) => {
                assert_eq!(expected, Some(len));
                assert_eq!(t.shape(), shape.as_slice());
                assert_eq!(t.data().len(), len);
                assert_eq!(t.numel(), len);
            }
            Err(NirError::InvalidTensor(_)) => {
                assert_ne!(expected, Some(len));
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    /// Overflowing shape products never construct a tensor and never panic.
    #[test]
    fn tensor_new_rejects_overflowing_shapes(
        // Large dims so the product overflows usize quickly.
        dims in prop::collection::vec(2usize..=usize::MAX / 2, 8..16),
    ) {
        assert!(shape_product(&dims).is_none());
        let err = Tensor::new(dims, TensorData::I64(vec![0])).unwrap_err();
        assert!(matches!(err, NirError::InvalidTensor(_)));
    }

    /// `zeros_like` / `ones_like` preserve shape and length.
    #[test]
    fn zeros_and_ones_like_preserve_shape(
        shape in prop::collection::vec(1usize..6, 0..4),
    ) {
        let n = shape_product(&shape).unwrap();
        let t = Tensor::from_f32(shape.clone(), vec![3.0; n]).unwrap();
        let z = t.zeros_like();
        let o = t.ones_like();
        assert_eq!(z.shape(), t.shape());
        assert_eq!(o.shape(), t.shape());
        assert_eq!(z.data().len(), n);
        assert_eq!(o.data().len(), n);
        if let TensorData::F32(v) = z.data() {
            assert!(v.iter().all(|&x| x == 0.0));
        } else {
            panic!("expected f32");
        }
        if let TensorData::F32(v) = o.data() {
            assert!(v.iter().all(|&x| x == 1.0));
        } else {
            panic!("expected f32");
        }
    }
}

// ---------------------------------------------------------------------------
// Graph structure
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Duplicate node names always error; successful inserts increase `len`.
    #[test]
    fn insert_node_rejects_duplicates(
        names in prop::collection::vec("[a-z]{1,4}", 1..12),
    ) {
        let mut g = NirGraph::new();
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            let before = g.len();
            match g.insert_node(name.clone(), input(vec![1])) {
                Ok(()) => {
                    assert!(seen.insert(name.clone()));
                    assert_eq!(g.len(), before + 1);
                }
                Err(NirError::DuplicateNode(n)) => {
                    assert_eq!(&n, name);
                    assert!(seen.contains(name));
                    assert_eq!(g.len(), before);
                }
                Err(other) => panic!("unexpected: {other:?}"),
            }
        }
    }

    /// Valid graphs (only endpoints that exist, no duplicate directed edges)
    /// always pass `validate_structure`, including with cycles.
    #[test]
    fn valid_edge_sets_validate(
        n_nodes in 1usize..8,
        edge_pairs in prop::collection::vec((0usize..8, 0usize..8), 0..20),
    ) {
        let mut g = NirGraph::new();
        for i in 0..n_nodes {
            g.insert_node(format!("n{i}"), input(vec![1])).unwrap();
        }
        let mut seen = std::collections::HashSet::new();
        for (a, b) in edge_pairs {
            let src = format!("n{}", a % n_nodes);
            let dst = format!("n{}", b % n_nodes);
            if seen.insert((src.clone(), dst.clone())) {
                g.add_edge(src, dst);
            }
        }
        g.validate_structure().expect("valid edge set must pass");
    }

    /// Missing endpoints always fail validation.
    #[test]
    fn missing_endpoints_fail_validation(
        n_nodes in 1usize..6,
        ghost in "[a-z]{5,8}",
    ) {
        prop_assume!(!ghost.starts_with('n') || ghost.len() > 2);
        let mut g = NirGraph::new();
        for i in 0..n_nodes {
            g.insert_node(format!("n{i}"), output(vec![1])).unwrap();
        }
        // Force a missing endpoint.
        g.add_edge("n0", ghost.clone());
        let err = g.validate_structure().unwrap_err();
        assert!(matches!(err, NirError::MissingNode(n) if n == ghost));
    }

    /// Nested subgraph structure errors are re-prefixed, never panic.
    #[test]
    fn nested_missing_edge_is_invalid_graph(
        ghost in "[a-z]{3,6}",
    ) {
        let mut inner = NirGraph::new();
        inner.insert_node("in", input(vec![1])).unwrap();
        inner.add_edge("in", ghost.clone());

        let mut outer = NirGraph::new();
        outer
            .insert_node("sub", NirNode::Graph(Box::new(inner)))
            .unwrap();
        let err = outer.validate_structure().unwrap_err();
        match err {
            NirError::InvalidGraph(msg) => {
                assert!(msg.contains("sub"), "{msg}");
                assert!(msg.contains(&ghost) || msg.contains("missing"), "{msg}");
            }
            other => panic!("expected InvalidGraph, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Wire name / string preflight (pure, no libhdf5)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `check_link_name` accepts only non-empty names without `/`, NUL, `.`, `..`.
    #[test]
    fn link_name_preflight(name in ".*") {
        // Cap length so we do not spend time on multi-megabyte strings.
        prop_assume!(name.len() < 64);
        let result = check_link_name("node name", &name);
        let bad = name.is_empty()
            || name.contains('/')
            || name.contains('\0')
            || name == "."
            || name == "..";
        if bad {
            assert!(result.is_err(), "should reject {name:?}");
        } else {
            assert!(result.is_ok(), "should accept {name:?}");
        }
    }

    /// Embedded NUL is always rejected for HDF5 string payloads.
    #[test]
    fn hdf5_string_rejects_nul(
        prefix in "[a-zA-Z0-9 ]{0,16}",
        suffix in "[a-zA-Z0-9 ]{0,16}",
    ) {
        let with_nul = format!("{prefix}\0{suffix}");
        assert!(check_hdf5_string("meta", &with_nul).is_err());
        // Without NUL, any UTF-8 of modest length is fine.
        let clean = format!("{prefix}{suffix}");
        assert!(check_hdf5_string("meta", &clean).is_ok());
    }
}

// ---------------------------------------------------------------------------
// Serde round-trip (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Finite f32 tensors that construct successfully round-trip through JSON
    /// when the `serde` feature is enabled.
    #[test]
    fn finite_tensor_json_round_trip(
        shape in prop::collection::vec(1usize..4, 0..3),
        seed in any::<u32>(),
    ) {
        let n = shape_product(&shape).unwrap();
        let data: Vec<f32> = (0..n)
            .map(|i| ((seed.wrapping_add(i as u32) % 1000) as f32) * 0.01)
            .collect();
        let t = Tensor::from_f32(shape, data).unwrap();
        let json = serde_json::to_string(&t).unwrap();
        let back: Tensor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    /// Structurally valid Input→Output graphs survive JSON round-trip.
    #[test]
    fn simple_graph_json_round_trip(n in 1usize..5) {
        let mut g = NirGraph::new();
        g.insert_node("in", input(vec![n])).unwrap();
        g.insert_node("out", output(vec![n])).unwrap();
        g.add_edge("in", "out");
        g.validate_structure().unwrap();
        let json = serde_json::to_string(&g).unwrap();
        let back: NirGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back, g);
        back.validate_structure().unwrap();
    }
}
