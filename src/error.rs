// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structured errors for `nir-rs`.
//!
//! Covers graph construction/validation, tensor shape checks, and HDF5 `.nir`
//! I/O.

use thiserror::Error;

/// Result type used across the crate.
pub type Result<T> = std::result::Result<T, NirError>;

/// Local convolution or pooling parameter invariant that failed.
///
/// Each variant is unambiguous from the node's own fields (no whole-graph
/// shape inference). Returned inside [`NirError::InvalidNodeParameters`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ParameterError {
    /// Weight tensor rank does not match the convolution kind.
    #[error("weight rank {found} is not {expected}")]
    WeightRank {
        /// Observed rank (`shape.len()`).
        found: usize,
        /// Required rank (3 for `Conv1d`, 4 for `Conv2d`).
        expected: usize,
    },
    /// `groups` is not a positive count.
    #[error("groups must be > 0, found {found}")]
    Groups {
        /// Observed `groups` value.
        found: i64,
    },
    /// Output-channel count is not divisible by `groups`.
    #[error("output channels {channels} are not divisible by groups {groups}")]
    ChannelGroupDivisibility {
        /// `weight` output-channel axis (`shape[0]`).
        channels: usize,
        /// Observed `groups` value.
        groups: i64,
    },
    /// A stride, dilation, padding, or pooling window has the wrong length.
    #[error("{field} arity {found} is not {expected}")]
    ExtentArity {
        /// Field name (`stride`, `dilation`, `padding`, `kernel_size`, …).
        field: &'static str,
        /// Human-readable allowed arity (`1`, `1 or 2`).
        expected: &'static str,
        /// Observed number of extents.
        found: usize,
    },
    /// A pooling window tensor is not a scalar or a 1-D vector of extents.
    #[error("{field} rank {found} is not 0 or 1")]
    ExtentRank {
        /// Field name (`kernel_size`, `stride`, `padding`).
        field: &'static str,
        /// Observed rank (`shape.len()`).
        found: usize,
    },
    /// A pooling window tensor is not an integer payload.
    #[error("{field} must contain i64 extents, found {dtype}")]
    ExtentDType {
        /// Field name (`kernel_size`, `stride`, `padding`).
        field: &'static str,
        /// Observed payload dtype label (`f32`, `f64`, `bool`).
        dtype: &'static str,
    },
    /// A stride, dilation, or kernel extent is not strictly positive.
    #[error("{field} extents must be strictly positive, found {value} at index {index}")]
    ExtentPositive {
        /// Field name (`stride`, `dilation`, `kernel_size`).
        field: &'static str,
        /// Index of the offending extent.
        index: usize,
        /// Observed value.
        value: i64,
    },
    /// An explicit padding extent is negative.
    #[error("{field} extents must be non-negative, found {value} at index {index}")]
    ExtentNonNegative {
        /// Field name (`padding`).
        field: &'static str,
        /// Index of the offending extent.
        index: usize,
        /// Observed value.
        value: i64,
    },
    /// Bias element count does not match the convolution's output channels.
    #[error("bias length {found} is incompatible with {expected} output channels")]
    BiasLength {
        /// `bias.numel()`.
        found: usize,
        /// Output-channel count from `weight.shape[0]`.
        expected: usize,
    },
}

/// Public error type for NIR operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum NirError {
    /// Feature not yet implemented.
    ///
    /// Returned by HDF5 I/O function stubs when the `hdf5` feature is not enabled.
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

    /// Local convolution or pooling parameter invariant failed.
    ///
    /// Returned by [`crate::NirNode::validate_parameters`] and
    /// [`crate::NirGraph::validate_parameters`]. HDF5 reads never emit this
    /// variant; callers opt in after import. The default writer also does not
    /// run parameter validation.
    #[error("invalid parameters for {node_type} node {node}: {kind}")]
    InvalidNodeParameters {
        /// Graph node name, or a `/`-separated path through nested subgraphs.
        ///
        /// Isolated [`crate::NirNode::validate_parameters`] uses `"<node>"`.
        node: String,
        /// Wire type string (`Conv1d`, `SumPool2d`, …).
        node_type: &'static str,
        /// The invariant that failed.
        kind: ParameterError,
    },

    /// A bounded read would exceed its decoded-allocation budget.
    #[error(
        "read allocation limit exceeded at {context}: limit {limit} bytes, used {used} bytes, requested {requested} bytes"
    )]
    ReadLimitExceeded {
        /// Dataset or synthesized field being charged.
        context: String,
        /// Configured decoded-allocation limit in bytes.
        limit: usize,
        /// Bytes already charged by earlier allocations.
        used: usize,
        /// Bytes requested by the allocation that was rejected.
        requested: usize,
    },

    /// Filesystem or HDF5 library failure while reading or writing a `.nir` file.
    ///
    /// The underlying `hdf5::Error` is rendered into the message rather than
    /// carried, so [`NirError`] stays `Clone + Eq`.
    #[error("io error: {0}")]
    Io(String),
}

/// Render an HDF5 library failure into [`NirError::Io`].
///
/// The message is flattened into a `String` because `hdf5::Error` is neither
/// `Clone` nor `Eq`, and this enum is both.
#[cfg(feature = "hdf5")]
impl From<hdf5::Error> for NirError {
    fn from(err: hdf5::Error) -> Self {
        Self::Io(err.to_string())
    }
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
    fn invalid_node_parameters_display() {
        let err = NirError::InvalidNodeParameters {
            node: "conv".into(),
            node_type: "Conv1d",
            kind: ParameterError::WeightRank {
                found: 2,
                expected: 3,
            },
        };
        assert_eq!(
            err.to_string(),
            "invalid parameters for Conv1d node conv: weight rank 2 is not 3"
        );
    }

    #[test]
    fn parameter_error_displays() {
        let cases = [
            (
                ParameterError::Groups { found: 0 },
                "groups must be > 0, found 0",
            ),
            (
                ParameterError::ChannelGroupDivisibility {
                    channels: 3,
                    groups: 2,
                },
                "output channels 3 are not divisible by groups 2",
            ),
            (
                ParameterError::ExtentArity {
                    field: "stride",
                    expected: "1",
                    found: 2,
                },
                "stride arity 2 is not 1",
            ),
            (
                ParameterError::ExtentRank {
                    field: "kernel_size",
                    found: 2,
                },
                "kernel_size rank 2 is not 0 or 1",
            ),
            (
                ParameterError::ExtentDType {
                    field: "padding",
                    dtype: "f32",
                },
                "padding must contain i64 extents, found f32",
            ),
            (
                ParameterError::ExtentPositive {
                    field: "dilation",
                    index: 0,
                    value: 0,
                },
                "dilation extents must be strictly positive, found 0 at index 0",
            ),
            (
                ParameterError::ExtentNonNegative {
                    field: "padding",
                    index: 1,
                    value: -1,
                },
                "padding extents must be non-negative, found -1 at index 1",
            ),
            (
                ParameterError::BiasLength {
                    found: 1,
                    expected: 2,
                },
                "bias length 1 is incompatible with 2 output channels",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn read_limit_display() {
        let err = NirError::ReadLimitExceeded {
            context: "lif.tau".into(),
            limit: 1024,
            used: 768,
            requested: 512,
        };
        assert_eq!(
            err.to_string(),
            "read allocation limit exceeded at lif.tau: limit 1024 bytes, used 768 bytes, requested 512 bytes"
        );
    }

    #[test]
    fn io_display() {
        let err = NirError::Io("unable to open file: model.nir".into());
        assert_eq!(err.to_string(), "io error: unable to open file: model.nir");
    }

    #[test]
    fn error_trait_implemented() {
        let err: Box<dyn std::error::Error> = Box::new(NirError::Unimplemented("x"));
        assert!(err.to_string().contains("not implemented"));
    }
}
