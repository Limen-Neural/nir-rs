// SPDX-License-Identifier: MIT OR Apache-2.0

//! In-memory NIR graph model.
//!
//! A graph is a named set of computational nodes plus a list of directed
//! identity edges, matching neuromorphs/NIR (`nodes`, `edges`, `metadata`,
//! optional `version`). Cycles are allowed; structure validation only checks
//! edge endpoints, duplicate directed edges, and nested [`NirNode::Graph`]
//! subgraphs. Validation walks those subgraphs on a heap-allocated work
//! list so process-stack usage does not grow with nesting depth.

use crate::error::{NirError, Result};
use crate::nodes::NirNode;
use crate::types::MetadataValue;
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

/// Directed NIR computation graph.
///
/// Node insertion order is preserved via [`IndexMap`] (stable iteration for
/// serialization and debugging). Edges are an ordered list of `(src, dst)`
/// name pairs.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NirGraph {
    /// Named computational nodes (insertion-ordered).
    pub nodes: IndexMap<String, NirNode>,
    /// Directed edges as `(source_name, destination_name)`.
    pub edges: Vec<(String, String)>,
    /// Free-form graph metadata.
    pub metadata: HashMap<String, MetadataValue>,
    /// Optional NIR version string (set when loading from HDF5 in v0.3).
    pub version: Option<String>,
}

impl NirGraph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a node under `name`.
    ///
    /// # Errors
    ///
    /// Returns [`NirError::DuplicateNode`] if `name` is already present.
    pub fn insert_node(&mut self, name: impl Into<String>, node: NirNode) -> Result<()> {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            return Err(NirError::DuplicateNode(name));
        }
        self.nodes.insert(name, node);
        Ok(())
    }

    /// Append a directed edge `(from → to)` without validating endpoints.
    ///
    /// Call [`validate_structure`](Self::validate_structure) to check that
    /// endpoints exist and that the edge is not duplicated.
    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) {
        self.edges.push((from.into(), to.into()));
    }

    /// Borrow the node named `name`, if present.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&NirNode> {
        self.nodes.get(name)
    }

    /// Mutably borrow the node named `name`, if present.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut NirNode> {
        self.nodes.get_mut(name)
    }

    /// Number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Maximum nesting depth of [`NirNode::Graph`] subgraphs that
    /// [`Self::validate_structure`] will walk.
    ///
    /// The root graph has depth `0`. Each nested [`NirNode::Graph`] increments
    /// the depth. A subgraph whose depth would exceed this limit is rejected
    /// with [`NirError::InvalidGraph`] instead of growing the process stack.
    ///
    /// This bound is independent of the HDF5 reader/writer nested-graph
    /// budget: it applies to adversarial in-memory graphs after
    /// deserialization succeeds; parsing depth is controlled by the serde
    /// format and deserializer.
    pub const MAX_NESTING_DEPTH: usize = 1024;

    /// Validate structural integrity.
    ///
    /// Checks:
    /// - every edge endpoint names an existing node
    /// - no duplicate directed edges `(src, dst)`
    /// - nested [`NirNode::Graph`] subgraphs also validate, in node insertion
    ///   order, depth-first
    ///
    /// Nested subgraphs are walked with an explicit heap-allocated work list, so
    /// process-stack usage does not grow with nesting depth. A chain deeper than
    /// [`Self::MAX_NESTING_DEPTH`] fails with [`NirError::InvalidGraph`].
    ///
    /// Cycles are **allowed**. Type/shape inference is out of scope for v0.2.
    ///
    /// First-error ordering matches a recursive walk: missing endpoints, then
    /// duplicate edges, then nested subgraphs in insertion order. Nested failures
    /// are re-prefixed so the message names each enclosing subgraph.
    ///
    /// # Errors
    ///
    /// - [`NirError::MissingNode`] if an endpoint is unknown
    /// - [`NirError::DuplicateEdge`] if the same directed edge appears twice
    /// - [`NirError::InvalidGraph`] if a nested subgraph fails validation, or if
    ///   nesting exceeds [`Self::MAX_NESTING_DEPTH`]
    pub fn validate_structure(&self) -> Result<()> {
        // Heap DFS: process-stack usage is O(1) in nesting depth. Frames are
        // pushed in reverse insertion order so the first nested graph is
        // popped next, matching the historical recursive walk.
        // Path ancestry is stored in a flat arena (`Vec<PathFrame>`) with
        // index-based parent links. Each frame is Copy (no recursive Drop glue
        // on worker threads with small stacks) and child nodes share prefix
        // ancestry in O(1) space.
        let mut frames: Vec<PathFrame<'_>> = Vec::new();
        let mut work: Vec<(&NirGraph, Option<usize>)> = vec![(self, None)];

        while let Some((graph, frame_idx)) = work.pop() {
            graph
                .validate_local_structure()
                .map_err(|err| wrap_frame_error(&frames, frame_idx, err))?;

            let parent_depth = frame_idx.map_or(0, |idx| frames[idx].depth);

            let mut nested = Vec::new();
            for (name, node) in &graph.nodes {
                let NirNode::Graph(sub) = node else {
                    continue;
                };
                let child_depth = parent_depth + 1;
                let child_idx = frames.len();
                frames.push(PathFrame {
                    name: name.as_str(),
                    parent: frame_idx,
                    depth: child_depth,
                });
                if parent_depth >= Self::MAX_NESTING_DEPTH {
                    return Err(wrap_frame_error(
                        &frames,
                        Some(child_idx),
                        NirError::InvalidGraph(format!(
                            "graph nesting depth exceeds {}",
                            Self::MAX_NESTING_DEPTH
                        )),
                    ));
                }
                nested.push((sub.as_ref(), Some(child_idx)));
            }
            work.extend(nested.into_iter().rev());
        }

        Ok(())
    }

    /// Endpoint and duplicate-edge checks for a single graph, ignoring nesting.
    fn validate_local_structure(&self) -> Result<()> {
        let node_keys: HashSet<&str> = self.nodes.keys().map(String::as_str).collect();

        for (src, dst) in &self.edges {
            if !node_keys.contains(src.as_str()) {
                return Err(NirError::MissingNode(src.clone()));
            }
            if !node_keys.contains(dst.as_str()) {
                return Err(NirError::MissingNode(dst.clone()));
            }
        }

        let mut seen_edges: HashSet<(&str, &str)> = HashSet::new();
        for (src, dst) in &self.edges {
            let key = (src.as_str(), dst.as_str());
            if !seen_edges.insert(key) {
                return Err(NirError::DuplicateEdge(src.clone(), dst.clone()));
            }
        }

        Ok(())
    }
}

/// Parent-linked frame tracking subgraph ancestry with O(1) prefix sharing.
///
/// Implements `Copy` so tearing down deep path chains performs no recursive drop
/// calls on the process stack.
#[derive(Clone, Copy)]
struct PathFrame<'a> {
    name: &'a str,
    parent: Option<usize>,
    depth: usize,
}

/// Prefix a structure-validation error with `in subgraph {name:?}: …`.
///
/// Nested [`NirError::InvalidGraph`] payloads are unwrapped so path prefixes
/// compose the way the historical recursive walk did. Other variants keep their
/// Display text so future validation errors still surface with path context.
fn prefix_subgraph_error(name: &str, err: NirError) -> NirError {
    match err {
        NirError::MissingNode(n) => {
            NirError::InvalidGraph(format!("in subgraph {name:?}: missing node: {n}"))
        }
        NirError::DuplicateEdge(a, b) => {
            NirError::InvalidGraph(format!("in subgraph {name:?}: duplicate edge: ({a}, {b})"))
        }
        NirError::DuplicateNode(n) => {
            NirError::InvalidGraph(format!("in subgraph {name:?}: duplicate node: {n}"))
        }
        NirError::InvalidGraph(msg) => {
            NirError::InvalidGraph(format!("in subgraph {name:?}: {msg}"))
        }
        other => NirError::InvalidGraph(format!("in subgraph {name:?}: {other}")),
    }
}

/// Apply [`prefix_subgraph_error`] from the innermost subgraph out to the root.
fn wrap_frame_error(
    frames: &[PathFrame<'_>],
    mut curr: Option<usize>,
    mut err: NirError,
) -> NirError {
    while let Some(idx) = curr {
        let frame = &frames[idx];
        err = prefix_subgraph_error(frame.name, err);
        curr = frame.parent;
    }
    err
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{Affine, Input, Lif, Output};
    use crate::types::Tensor;

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

    #[test]
    fn empty_graph_default() {
        let g = NirGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert!(g.validate_structure().is_ok());
    }

    #[test]
    fn insert_node_rejects_duplicate() {
        let mut g = NirGraph::new();
        g.insert_node("a", input(vec![4])).unwrap();
        let err = g.insert_node("a", input(vec![2])).unwrap_err();
        assert_eq!(err, NirError::DuplicateNode("a".into()));
    }

    #[test]
    fn get_returns_inserted_node() {
        let mut g = NirGraph::new();
        g.insert_node("in", input(vec![3])).unwrap();
        assert!(matches!(g.get("in"), Some(NirNode::Input(_))));
        assert!(g.get("missing").is_none());
    }

    #[test]
    fn validate_missing_source_endpoint() {
        let mut g = NirGraph::new();
        g.insert_node("b", output(vec![1])).unwrap();
        g.add_edge("ghost", "b");
        let err = g.validate_structure().unwrap_err();
        assert_eq!(err, NirError::MissingNode("ghost".into()));
    }

    #[test]
    fn validate_missing_dest_endpoint() {
        let mut g = NirGraph::new();
        g.insert_node("a", input(vec![1])).unwrap();
        g.add_edge("a", "ghost");
        let err = g.validate_structure().unwrap_err();
        assert_eq!(err, NirError::MissingNode("ghost".into()));
    }

    #[test]
    fn validate_duplicate_edge() {
        let mut g = NirGraph::new();
        g.insert_node("a", input(vec![1])).unwrap();
        g.insert_node("b", output(vec![1])).unwrap();
        g.add_edge("a", "b");
        g.add_edge("a", "b");
        let err = g.validate_structure().unwrap_err();
        assert_eq!(err, NirError::DuplicateEdge("a".into(), "b".into()));
    }

    #[test]
    fn cycles_are_allowed() {
        let mut g = NirGraph::new();
        g.insert_node("a", input(vec![1])).unwrap();
        g.insert_node("b", output(vec![1])).unwrap();
        g.add_edge("a", "b");
        g.add_edge("b", "a");
        assert!(g.validate_structure().is_ok());
    }

    #[test]
    fn integration_input_affine_lif_output() {
        let weight = Tensor::from_f32(vec![2, 4], vec![0.1; 8]).unwrap();
        let bias = Tensor::from_f32(vec![2], vec![0.0, 0.0]).unwrap();
        let tau = Tensor::from_f64(vec![2], vec![10.0, 10.0]).unwrap();
        let r = Tensor::from_f64(vec![2], vec![1.0, 1.0]).unwrap();
        let v_leak = Tensor::from_f64(vec![2], vec![0.0, 0.0]).unwrap();
        let v_th = Tensor::from_f64(vec![2], vec![1.0, 1.0]).unwrap();

        let mut g = NirGraph::new();
        g.version = Some("1.0.0".into());
        g.metadata.insert(
            "origin".into(),
            MetadataValue::String("integration-test".into()),
        );

        g.insert_node("input", input(vec![4])).unwrap();
        g.insert_node(
            "fc",
            NirNode::Affine(Affine {
                weight,
                bias,
                metadata: Default::default(),
            }),
        )
        .unwrap();
        g.insert_node(
            "lif",
            NirNode::Lif(Lif {
                tau,
                r,
                v_leak,
                v_threshold: v_th,
                v_reset: None,
                metadata: Default::default(),
            }),
        )
        .unwrap();
        g.insert_node("output", output(vec![2])).unwrap();

        g.add_edge("input", "fc");
        g.add_edge("fc", "lif");
        g.add_edge("lif", "output");

        assert!(g.validate_structure().is_ok());
        assert_eq!(g.len(), 4);
        assert_eq!(g.edges.len(), 3);
        assert_eq!(g.get("lif").unwrap().type_name(), "LIF");
        assert_eq!(g.get("fc").unwrap().type_name(), "Affine");
    }

    #[test]
    fn nested_graph_validates() {
        let mut inner = NirGraph::default();
        inner.insert_node("i", input(vec![1])).unwrap();
        inner.insert_node("o", output(vec![1])).unwrap();
        inner.add_edge("i", "o");

        let mut outer = NirGraph::default();
        outer
            .insert_node("sub", NirNode::Graph(Box::new(inner)))
            .unwrap();
        outer.insert_node("out", output(vec![1])).unwrap();
        outer.add_edge("sub", "out");
        assert!(outer.validate_structure().is_ok());
    }

    #[test]
    fn nested_graph_reports_inner_failure() {
        let mut inner = NirGraph::default();
        inner.insert_node("i", input(vec![1])).unwrap();
        // edge to missing node
        inner.add_edge("i", "missing");

        let mut outer = NirGraph::default();
        outer
            .insert_node("sub", NirNode::Graph(Box::new(inner)))
            .unwrap();
        let err = outer.validate_structure().unwrap_err();
        assert_eq!(
            err,
            NirError::InvalidGraph("in subgraph \"sub\": missing node: missing".into())
        );
    }

    #[test]
    fn nested_duplicate_edge_keeps_path_context() {
        let mut inner = NirGraph::default();
        inner.insert_node("a", input(vec![1])).unwrap();
        inner.insert_node("b", output(vec![1])).unwrap();
        inner.add_edge("a", "b");
        inner.add_edge("a", "b");

        let mut outer = NirGraph::default();
        outer
            .insert_node("sub", NirNode::Graph(Box::new(inner)))
            .unwrap();
        let err = outer.validate_structure().unwrap_err();
        assert_eq!(
            err,
            NirError::InvalidGraph("in subgraph \"sub\": duplicate edge: (a, b)".into())
        );
    }

    #[test]
    fn nested_path_context_is_outermost_first() {
        let mut inner = NirGraph::default();
        inner.insert_node("i", input(vec![1])).unwrap();
        inner.add_edge("i", "ghost");

        let mut mid = NirGraph::default();
        mid.insert_node("inner", NirNode::Graph(Box::new(inner)))
            .unwrap();

        let mut outer = NirGraph::default();
        outer
            .insert_node("mid", NirNode::Graph(Box::new(mid)))
            .unwrap();
        let err = outer.validate_structure().unwrap_err();
        assert_eq!(
            err,
            NirError::InvalidGraph(
                "in subgraph \"mid\": in subgraph \"inner\": missing node: ghost".into()
            )
        );
    }

    #[test]
    fn outer_endpoint_error_precedes_nested_failure() {
        let mut inner = NirGraph::default();
        inner.insert_node("i", input(vec![1])).unwrap();
        inner.add_edge("i", "missing_inner");

        let mut outer = NirGraph::default();
        outer
            .insert_node("sub", NirNode::Graph(Box::new(inner)))
            .unwrap();
        outer.add_edge("ghost", "sub");
        let err = outer.validate_structure().unwrap_err();
        assert_eq!(err, NirError::MissingNode("ghost".into()));
    }

    #[test]
    fn first_nested_sibling_error_is_reported() {
        let mut first = NirGraph::default();
        first.insert_node("i", input(vec![1])).unwrap();
        first.add_edge("i", "missing_a");

        let mut second = NirGraph::default();
        second.insert_node("i", input(vec![1])).unwrap();
        second.add_edge("i", "missing_b");

        let mut outer = NirGraph::default();
        outer
            .insert_node("sub_a", NirNode::Graph(Box::new(first)))
            .unwrap();
        outer
            .insert_node("sub_b", NirNode::Graph(Box::new(second)))
            .unwrap();
        let err = outer.validate_structure().unwrap_err();
        assert_eq!(
            err,
            NirError::InvalidGraph("in subgraph \"sub_a\": missing node: missing_a".into())
        );
    }

    #[test]
    fn nested_cycles_are_allowed() {
        let mut inner = NirGraph::default();
        inner.insert_node("a", input(vec![1])).unwrap();
        inner.insert_node("b", output(vec![1])).unwrap();
        inner.add_edge("a", "b");
        inner.add_edge("b", "a");

        let mut outer = NirGraph::default();
        outer
            .insert_node("loop", NirNode::Graph(Box::new(inner)))
            .unwrap();
        assert!(outer.validate_structure().is_ok());
    }

    /// Wrap `leaf` in `depth` enclosing [`NirNode::Graph`] nodes named `n0`…`n{depth-1}`.
    fn wrap_depth(depth: usize, leaf: NirGraph) -> NirGraph {
        let mut g = leaf;
        for i in (0..depth).rev() {
            let mut outer = NirGraph::default();
            outer
                .insert_node(format!("n{i}"), NirNode::Graph(Box::new(g)))
                .unwrap();
            g = outer;
        }
        g
    }

    fn leaf_graph() -> NirGraph {
        let mut g = NirGraph::default();
        g.insert_node("leaf", input(vec![1])).unwrap();
        g
    }

    #[test]
    fn nesting_at_max_depth_validates() {
        let g = wrap_depth(NirGraph::MAX_NESTING_DEPTH, leaf_graph());
        assert!(g.validate_structure().is_ok());
    }

    #[test]
    fn nesting_beyond_max_depth_is_invalid_graph() {
        let g = wrap_depth(NirGraph::MAX_NESTING_DEPTH + 1, leaf_graph());
        let err = g.validate_structure().unwrap_err();
        match err {
            NirError::InvalidGraph(msg) => {
                assert!(msg.contains("graph nesting depth exceeds"), "{msg}");
                assert!(
                    msg.contains(&NirGraph::MAX_NESTING_DEPTH.to_string()),
                    "{msg}"
                );
                assert!(
                    msg.contains(&format!("n{}", NirGraph::MAX_NESTING_DEPTH)),
                    "{msg}"
                );
                assert!(msg.contains("n0"), "{msg}");
            }
            other => panic!("expected InvalidGraph, got {other:?}"),
        }
    }

    #[test]
    fn wide_nesting_is_not_a_depth_limit() {
        let mut outer = NirGraph::default();
        for i in 0..64 {
            outer
                .insert_node(format!("sub{i}"), NirNode::Graph(Box::new(leaf_graph())))
                .unwrap();
        }
        assert!(outer.validate_structure().is_ok());
    }

    #[test]
    fn wide_frontier_with_shared_path_frames_validates() {
        let mut root = NirGraph::default();
        for i in 0..50 {
            let mut sub = NirGraph::default();
            for j in 0..20 {
                sub.insert_node(format!("leaf_{j}"), input(vec![1]))
                    .unwrap();
            }
            root.insert_node(format!("branch_{i}"), NirNode::Graph(Box::new(sub)))
                .unwrap();
        }
        assert!(root.validate_structure().is_ok());
    }

    /// Historical recursive walk used as an oracle for shallow graphs.
    fn validate_structure_recursive(graph: &NirGraph) -> Result<()> {
        graph.validate_local_structure()?;
        for (name, node) in &graph.nodes {
            if let NirNode::Graph(sub) = node {
                validate_structure_recursive(sub).map_err(|e| prefix_subgraph_error(name, e))?;
            }
        }
        Ok(())
    }

    fn assert_matches_recursive(graph: &NirGraph) {
        assert_eq!(
            graph.validate_structure(),
            validate_structure_recursive(graph)
        );
    }

    #[test]
    fn shallow_graphs_match_recursive_oracle() {
        assert_matches_recursive(&NirGraph::default());
        assert_matches_recursive(&leaf_graph());

        let mut missing = NirGraph::default();
        missing.insert_node("a", input(vec![1])).unwrap();
        missing.add_edge("a", "ghost");
        assert_matches_recursive(&missing);

        let mut dup = NirGraph::default();
        dup.insert_node("a", input(vec![1])).unwrap();
        dup.insert_node("b", output(vec![1])).unwrap();
        dup.add_edge("a", "b");
        dup.add_edge("a", "b");
        assert_matches_recursive(&dup);

        let mut cyc = NirGraph::default();
        cyc.insert_node("a", input(vec![1])).unwrap();
        cyc.insert_node("b", output(vec![1])).unwrap();
        cyc.add_edge("a", "b");
        cyc.add_edge("b", "a");
        assert_matches_recursive(&cyc);

        for depth in 0..=4 {
            let mut inner = NirGraph::default();
            inner.insert_node("i", input(vec![1])).unwrap();
            inner.add_edge("i", "ghost");
            assert_matches_recursive(&wrap_depth(depth, inner));
            assert_matches_recursive(&wrap_depth(depth, leaf_graph()));
        }

        let mut first = NirGraph::default();
        first.insert_node("i", input(vec![1])).unwrap();
        first.add_edge("i", "missing_a");
        let mut second = NirGraph::default();
        second.insert_node("i", input(vec![1])).unwrap();
        second.add_edge("i", "missing_b");
        let mut outer = NirGraph::default();
        outer
            .insert_node("sub_a", NirNode::Graph(Box::new(first)))
            .unwrap();
        outer
            .insert_node("sub_b", NirNode::Graph(Box::new(second)))
            .unwrap();
        outer.add_edge("nope", "sub_a");
        assert_matches_recursive(&outer);
    }

    #[test]
    fn insertion_order_preserved() {
        let mut g = NirGraph::default();
        g.insert_node("z", input(vec![1])).unwrap();
        g.insert_node("a", output(vec![1])).unwrap();
        let keys: Vec<&str> = g.nodes.keys().map(String::as_str).collect();
        assert_eq!(keys, ["z", "a"]);
    }

    #[test]
    fn small_stack_thread_validates_max_depth_without_overflow() {
        // Runs validate_structure on a thread with a 64 KiB stack to prove
        // that neither traversal nor PathFrame teardown recurses on the process stack.
        let g = wrap_depth(NirGraph::MAX_NESTING_DEPTH, leaf_graph());
        let builder = std::thread::Builder::new().stack_size(64 * 1024);
        std::thread::scope(|s| {
            builder
                .spawn_scoped(s, || {
                    assert!(g.validate_structure().is_ok());
                })
                .unwrap()
                .join()
                .unwrap();
        });
    }
}
