// SPDX-License-Identifier: MIT OR Apache-2.0

//! In-memory NIR graph model.
//!
//! A graph is a named set of computational nodes plus a list of directed
//! identity edges, matching neuromorphs/NIR (`nodes`, `edges`, `metadata`,
//! optional `version`). Cycles are allowed; structure validation only checks
//! edge endpoints and duplicate directed edges.

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

    /// Validate structural integrity.
    ///
    /// Checks:
    /// - every edge endpoint names an existing node
    /// - no duplicate directed edges `(src, dst)`
    /// - nested [`NirNode::Graph`] subgraphs also validate
    ///
    /// Cycles are **allowed**. Type/shape inference is out of scope for v0.2.
    ///
    /// # Errors
    ///
    /// - [`NirError::MissingNode`] if an endpoint is unknown
    /// - [`NirError::DuplicateEdge`] if the same directed edge appears twice
    /// - [`NirError::InvalidGraph`] if a nested subgraph fails validation
    pub fn validate_structure(&self) -> Result<()> {
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

        // Nested graphs: re-prefix structure errors so callers see which subgraph failed.
        // Today this method only returns MissingNode / DuplicateEdge / InvalidGraph;
        // the `other` arm preserves any future validation variants as InvalidGraph.
        for (name, node) in &self.nodes {
            if let NirNode::Graph(sub) = node {
                sub.validate_structure().map_err(|e| match e {
                    NirError::MissingNode(n) => {
                        NirError::InvalidGraph(format!("in subgraph {name:?}: missing node: {n}"))
                    }
                    NirError::DuplicateEdge(a, b) => NirError::InvalidGraph(format!(
                        "in subgraph {name:?}: duplicate edge: ({a}, {b})"
                    )),
                    NirError::DuplicateNode(n) => {
                        NirError::InvalidGraph(format!("in subgraph {name:?}: duplicate node: {n}"))
                    }
                    NirError::InvalidGraph(msg) => {
                        NirError::InvalidGraph(format!("in subgraph {name:?}: {msg}"))
                    }
                    other => NirError::InvalidGraph(format!("in subgraph {name:?}: {other}")),
                })?;
            }
        }

        Ok(())
    }
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
        let mut inner = NirGraph::new();
        inner.insert_node("i", input(vec![1])).unwrap();
        inner.insert_node("o", output(vec![1])).unwrap();
        inner.add_edge("i", "o");

        let mut outer = NirGraph::new();
        outer
            .insert_node("sub", NirNode::Graph(Box::new(inner)))
            .unwrap();
        outer.insert_node("out", output(vec![1])).unwrap();
        outer.add_edge("sub", "out");
        assert!(outer.validate_structure().is_ok());
    }

    #[test]
    fn nested_graph_reports_inner_failure() {
        let mut inner = NirGraph::new();
        inner.insert_node("i", input(vec![1])).unwrap();
        // edge to missing node
        inner.add_edge("i", "missing");

        let mut outer = NirGraph::new();
        outer
            .insert_node("sub", NirNode::Graph(Box::new(inner)))
            .unwrap();
        let err = outer.validate_structure().unwrap_err();
        match err {
            NirError::InvalidGraph(msg) => {
                assert!(msg.contains("sub"));
                assert!(msg.contains("missing"));
            }
            other => panic!("expected InvalidGraph, got {other:?}"),
        }
    }

    #[test]
    fn insertion_order_preserved() {
        let mut g = NirGraph::new();
        g.insert_node("z", input(vec![1])).unwrap();
        g.insert_node("a", output(vec![1])).unwrap();
        let keys: Vec<&str> = g.nodes.keys().map(String::as_str).collect();
        assert_eq!(keys, ["z", "a"]);
    }
}
