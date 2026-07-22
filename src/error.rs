// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structured errors for `nir-rs`.
//!
//! Covers graph construction/validation, tensor shape checks, and placeholders
//! for I/O and version handling that land in later milestones.

use thiserror::Error;

/// Result type used across the crate.
pub type Result<T> = std::result::Result<T, NirError>;

/// Public error type for NIR operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum NirError {
    /// Feature not yet implemented (e.g. HDF5 I/O until v0.3).
    #[error("not implemented: {0}")]
    Unimplemented(&'static str),

    /// HDF5 / wire `type` string is not a known NIR node.
    #[error("unknown node type: {0}")]
    UnknownNodeType(String),

    /// A node name was inserted twice into the same graph.
    #[error("duplicate node: {0}")]
    DuplicateNode(String),

    /// An edge or lookup referenced a node name that is not in the graph.
    #[error("missing node: {0}")]
    MissingNode(String),

    /// The same directed edge `(src, dst)` appears more than once.
    #[error("duplicate edge: ({0}, {1})")]
    DuplicateEdge(String, String),

    /// Structural or semantic problem with a graph that is not covered above.
    #[error("invalid graph: {0}")]
    InvalidGraph(String),

    /// NIR file / graph version is not supported by this crate.
    #[error("unsupported version: {0}")]
    UnsupportedVersion(String),

    /// A required wire field was absent when decoding a node or graph.
    #[error("missing field: {0}")]
    MissingField(String),

    /// Tensor shape / data length mismatch or other tensor invariant failure.
    #[error("invalid tensor: {0}")]
    InvalidTensor(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unimplemented_display() {
        let err = NirError::Unimplemented("hdf5 read");
        assert_eq!(err.to_string(), "not implemented: hdf5 read");
    }

    #[test]
    fn unknown_node_type_display() {
        let err = NirError::UnknownNodeType("CurrLIF".into());
        assert_eq!(err.to_string(), "unknown node type: CurrLIF");
    }

    #[test]
    fn duplicate_node_display() {
        let err = NirError::DuplicateNode("lif".into());
        assert_eq!(err.to_string(), "duplicate node: lif");
    }

    #[test]
    fn missing_node_display() {
        let err = NirError::MissingNode("missing".into());
        assert_eq!(err.to_string(), "missing node: missing");
    }

    #[test]
    fn duplicate_edge_display() {
        let err = NirError::DuplicateEdge("a".into(), "b".into());
        assert_eq!(err.to_string(), "duplicate edge: (a, b)");
    }

    #[test]
    fn invalid_graph_display() {
        let err = NirError::InvalidGraph("empty subgraph".into());
        assert_eq!(err.to_string(), "invalid graph: empty subgraph");
    }

    #[test]
    fn unsupported_version_display() {
        let err = NirError::UnsupportedVersion("99.0".into());
        assert_eq!(err.to_string(), "unsupported version: 99.0");
    }

    #[test]
    fn missing_field_display() {
        let err = NirError::MissingField("weight".into());
        assert_eq!(err.to_string(), "missing field: weight");
    }

    #[test]
    fn invalid_tensor_display() {
        let err = NirError::InvalidTensor("shape product 4 != data len 3".into());
        assert_eq!(
            err.to_string(),
            "invalid tensor: shape product 4 != data len 3"
        );
    }

    #[test]
    fn error_trait_implemented() {
        let err: Box<dyn std::error::Error> = Box::new(NirError::Unimplemented("x"));
        assert!(err.to_string().contains("not implemented"));
    }
}
