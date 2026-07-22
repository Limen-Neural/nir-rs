// SPDX-License-Identifier: MIT OR Apache-2.0

//! In-memory NIR graph model.
//!
//! v0.1 is a scaffold only. Typed nodes, edges, metadata, and version fields
//! are filled in **v0.2 — Core IR** (see LIM-826 / GH#7).

/// Directed NIR computation graph (scaffold).
///
/// Upstream NIR represents graphs as named nodes plus an edge list. The full
/// shape will match neuromorphs/NIR wire layout once Core IR lands.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NirGraph {
    // Fields reserved for v0.2 (nodes, edges, metadata, version).
}

impl NirGraph {
    /// Create an empty graph scaffold.
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_default() {
        let g = NirGraph::new();
        assert_eq!(g, NirGraph::default());
    }
}
