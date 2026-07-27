// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vocabulary of the NIR HDF5 wire format.
//!
//! Everything here is pure Rust and always compiled, with or without the
//! `hdf5` feature: it is the shared agreement between the reader and the
//! writer, and it is useful on its own to tooling that inspects `.nir` files
//! without going through [`crate::io::read`].
//!
//! The structural layout these names describe:
//!
//! ```text
//! /version                 scalar string, e.g. "0.2.0"
//! /node                    group — the root NIRGraph
//!   type                   scalar string "NIRGraph"
//!   edges                  (E, 2) string dataset
//!   nodes/<name>/          one group per node
//!     type                 scalar string, e.g. "LIF"
//!     <field>              one dataset per wire field
//!     metadata/            group; omitted when empty
//!   metadata/              group; omitted when empty
//! ```

use crate::error::{NirError, Result};
use crate::nodes::Padding;

/// Dataset holding the NIR format version at the root of the file.
pub const KEY_VERSION: &str = "version";
/// Group holding the root graph.
pub const KEY_NODE: &str = "node";
/// Group holding a graph's named nodes.
pub const KEY_NODES: &str = "nodes";
/// Dataset holding a graph's `(E, 2)` edge list.
pub const KEY_EDGES: &str = "edges";
/// Group holding free-form metadata; omitted from the file when empty.
pub const KEY_METADATA: &str = "metadata";
/// Dataset holding a node's wire `type` string.
pub const KEY_TYPE: &str = "type";

/// Every node `type` string that can appear in a `.nir` file.
///
/// These match [`crate::NirNode::type_name`] exactly — a unit test asserts the
/// two stay in step, so this list cannot drift from the enum.
///
/// Upstream's `Identity` node is deliberately absent: it is not in the Python
/// serializer registry (`nir.ir.__all_ir`) and so never reaches the wire.
pub const WIRE_TYPES: [&str; 19] = [
    "Input",
    "Output",
    "Affine",
    "Linear",
    "Scale",
    "Conv1d",
    "Conv2d",
    "CubaLI",
    "CubaLIF",
    "Delay",
    "Flatten",
    "I",
    "IF",
    "LI",
    "LIF",
    "SumPool2d",
    "AvgPool2d",
    "Threshold",
    "NIRGraph",
];

/// Whether `name` is a NIR wire node type this crate understands.
#[must_use]
pub fn is_wire_type(name: &str) -> bool {
    WIRE_TYPES.contains(&name)
}

/// Wire spelling of the symbolic padding modes.
///
/// Returns [`None`] for [`Padding::Explicit`], which is written as an integer
/// dataset rather than a string.
#[must_use]
pub fn padding_as_wire_str(padding: &Padding) -> Option<&'static str> {
    match padding {
        Padding::Same => Some("same"),
        Padding::Valid => Some("valid"),
        Padding::Explicit(_) => None,
    }
}

/// Parse a symbolic padding mode from its wire spelling.
///
/// # Errors
///
/// Returns [`NirError::InvalidGraph`] for anything other than `"same"` or
/// `"valid"` — the only two strings upstream `Conv1d` / `Conv2d` accept.
pub fn padding_from_wire_str(s: &str) -> Result<Padding> {
    match s {
        "same" => Ok(Padding::Same),
        "valid" => Ok(Padding::Valid),
        other => Err(NirError::InvalidGraph(format!(
            "padding must be \"same\", \"valid\", or integer extents, not {other:?}"
        ))),
    }
}

/// Check that `name` can be used as an HDF5 link name.
///
/// Applies to every caller-supplied string that becomes a link in the file:
/// graph node names and metadata keys. HDF5 splits paths on `/`, so a name
/// containing one would silently nest and change the graph on the next read;
/// `.` and `..` are reserved path components; link names are C strings and so
/// cannot carry an embedded NUL; and an empty name has no valid encoding.
///
/// Callers should run this **before** creating the destination file. Every
/// rejected name otherwise fails at link-creation time, by which point an
/// existing file at that path has already been truncated.
///
/// `kind` names what is being checked (`"node name"`, `"metadata key"`) and
/// appears in the error.
///
/// # Errors
///
/// Returns [`NirError::InvalidGraph`] describing the offending name.
pub fn check_link_name(kind: &str, name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(NirError::InvalidGraph(format!("{kind} must not be empty")));
    }
    if name.contains('/') {
        return Err(NirError::InvalidGraph(format!(
            "{kind} {name:?} must not contain '/' (HDF5 path separator)"
        )));
    }
    if name.contains('\0') {
        return Err(NirError::InvalidGraph(format!(
            "{kind} {name:?} must not contain a NUL byte (HDF5 link names are C strings)"
        )));
    }
    if name == "." || name == ".." {
        return Err(NirError::InvalidGraph(format!(
            "{kind} {name:?} is a reserved HDF5 path component"
        )));
    }
    Ok(())
}

/// Check that `value` can be stored in an HDF5 string dataset.
///
/// Link names and string payloads share the C-string constraint: an embedded
/// NUL cannot be encoded. Call this **before** creating the destination file,
/// alongside [`check_link_name`].
///
/// # Errors
///
/// Returns [`NirError::InvalidGraph`] when `value` contains a NUL byte.
pub fn check_hdf5_string(kind: &str, value: &str) -> Result<()> {
    if value.contains('\0') {
        return Err(NirError::InvalidGraph(format!(
            "{kind} {value:?} must not contain a NUL byte (HDF5 strings are C strings)"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NirGraph;
    use crate::nodes::{
        Affine, AvgPool2d, Conv1d, Conv2d, CubaLi, CubaLif, Delay, Flatten, I, If, Input, Li, Lif,
        Linear, NirNode, Output, Scale, SumPool2d, Threshold,
    };
    use crate::types::Tensor;

    /// A length-2 `f64` vector, the shape every neuron parameter below uses.
    fn v() -> Tensor {
        Tensor::from_f64([2], vec![1.0, 1.0]).unwrap()
    }

    /// A length-2 `i64` vector, for pooling windows.
    fn pool() -> Tensor {
        Tensor::from_i64([2], vec![2, 2]).unwrap()
    }

    fn port_and_linear_nodes() -> Vec<NirNode> {
        let weight = || Tensor::from_f32(vec![2, 2], vec![1., 0., 0., 1.]).unwrap();
        vec![
            NirNode::Input(Input {
                shape: vec![2],
                metadata: Default::default(),
            }),
            NirNode::Output(Output {
                shape: vec![2],
                metadata: Default::default(),
            }),
            NirNode::Affine(Affine {
                weight: weight(),
                bias: Tensor::from_f32([2], vec![0., 0.]).unwrap(),
                metadata: Default::default(),
            }),
            NirNode::Linear(Linear {
                weight: weight(),
                metadata: Default::default(),
            }),
            NirNode::Scale(Scale {
                scale: v(),
                metadata: Default::default(),
            }),
        ]
    }

    fn conv_nodes() -> Vec<NirNode> {
        vec![
            NirNode::Conv1d(Conv1d {
                weight: Tensor::from_f32(vec![1, 1, 3], vec![1., 0., -1.]).unwrap(),
                stride: vec![1],
                padding: Padding::single(0),
                dilation: vec![1],
                groups: 1,
                bias: Tensor::from_f32([1], vec![0.]).unwrap(),
                input_shape: Some(10),
                metadata: Default::default(),
            }),
            NirNode::Conv2d(Conv2d {
                weight: Tensor::from_f32(vec![1, 1, 2, 2], vec![0.; 4]).unwrap(),
                stride: vec![1, 1],
                padding: Padding::Same,
                dilation: vec![1, 1],
                groups: 1,
                bias: Tensor::from_f32([1], vec![0.]).unwrap(),
                input_shape: Some(vec![8, 8]),
                metadata: Default::default(),
            }),
        ]
    }

    fn cuba_nodes() -> Vec<NirNode> {
        vec![
            NirNode::CubaLi(CubaLi {
                tau_syn: v(),
                tau_mem: v(),
                r: v(),
                v_leak: v(),
                w_in: None,
                metadata: Default::default(),
            }),
            NirNode::CubaLif(CubaLif {
                tau_syn: v(),
                tau_mem: v(),
                r: v(),
                v_leak: v(),
                v_threshold: v(),
                v_reset: None,
                w_in: None,
                metadata: Default::default(),
            }),
        ]
    }

    fn neuron_nodes() -> Vec<NirNode> {
        vec![
            NirNode::I(I {
                r: v(),
                metadata: Default::default(),
            }),
            NirNode::If(If {
                r: v(),
                v_threshold: v(),
                v_reset: None,
                metadata: Default::default(),
            }),
            NirNode::Li(Li {
                tau: v(),
                r: v(),
                v_leak: v(),
                metadata: Default::default(),
            }),
            NirNode::Lif(Lif {
                tau: v(),
                r: v(),
                v_leak: v(),
                v_threshold: v(),
                v_reset: None,
                metadata: Default::default(),
            }),
        ]
    }

    fn pool_nodes() -> Vec<NirNode> {
        let no_pad = || Tensor::from_i64([2], vec![0, 0]).unwrap();
        vec![
            NirNode::SumPool2d(SumPool2d {
                kernel_size: pool(),
                stride: pool(),
                padding: no_pad(),
                metadata: Default::default(),
            }),
            NirNode::AvgPool2d(AvgPool2d {
                kernel_size: pool(),
                stride: pool(),
                padding: no_pad(),
                metadata: Default::default(),
            }),
        ]
    }

    /// Compile-time notification that a `NirNode` variant was added.
    ///
    /// The match has no `_` arm, so a new variant stops the crate compiling
    /// here. That is all it guarantees: the fix is *reviewed*, not that the
    /// lists get updated — adding the variant to this arm satisfies the
    /// compiler on its own, and two lists that both omit it still compare
    /// equal in `wire_types_matches_every_node_variant`.
    ///
    /// Closing that gap properly needs the variants, their wire strings and
    /// their sample values generated from one definition, which means a macro
    /// owning `NirNode` itself — stable Rust has no way to enumerate an enum's
    /// variants otherwise. Out of scope here; the compile error plus the
    /// name-vs-`WIRE_TYPES` comparison is the practical net.
    ///
    /// It takes a real node rather than matching a fixed dummy, which would
    /// leave every other arm unreachable and prove nothing at all.
    fn assert_variant_covered(node: &NirNode) {
        match node {
            NirNode::Input(_)
            | NirNode::Output(_)
            | NirNode::Affine(_)
            | NirNode::Linear(_)
            | NirNode::Scale(_)
            | NirNode::Conv1d(_)
            | NirNode::Conv2d(_)
            | NirNode::CubaLi(_)
            | NirNode::CubaLif(_)
            | NirNode::Delay(_)
            | NirNode::Flatten(_)
            | NirNode::I(_)
            | NirNode::If(_)
            | NirNode::Li(_)
            | NirNode::Lif(_)
            | NirNode::SumPool2d(_)
            | NirNode::AvgPool2d(_)
            | NirNode::Threshold(_)
            | NirNode::Graph(_) => {}
        }
    }

    /// One value per `NirNode` variant, in [`WIRE_TYPES`] order.
    ///
    /// Assembled from the per-family helpers above so the constructors live in
    /// one place; `assert_variant_covered` supplies the compile-time coverage.
    fn one_of_each() -> Vec<NirNode> {
        let mut nodes = port_and_linear_nodes();
        nodes.extend(conv_nodes());
        nodes.extend(cuba_nodes());
        nodes.push(NirNode::Delay(Delay {
            delay: v(),
            metadata: Default::default(),
        }));
        nodes.push(NirNode::Flatten(Flatten {
            start_dim: 1,
            end_dim: -1,
            input_type: None,
            metadata: Default::default(),
        }));
        nodes.extend(neuron_nodes());
        nodes.extend(pool_nodes());
        nodes.push(NirNode::Threshold(Threshold {
            threshold: v(),
            metadata: Default::default(),
        }));
        nodes.push(NirNode::Graph(Box::new(NirGraph::new())));
        for node in &nodes {
            assert_variant_covered(node);
        }
        nodes
    }

    #[test]
    fn wire_types_matches_every_node_variant() {
        let names: Vec<&str> = one_of_each().iter().map(NirNode::type_name).collect();
        assert_eq!(
            names, WIRE_TYPES,
            "WIRE_TYPES must list exactly the NirNode variants, in order"
        );
    }

    #[test]
    fn is_wire_type_rejects_marketing_aliases() {
        for good in WIRE_TYPES {
            assert!(is_wire_type(good), "{good} should be a wire type");
        }
        for bad in ["CurrLIF", "Convolution", "Integrator", "SumPooling", ""] {
            assert!(!is_wire_type(bad), "{bad} must not be a wire type");
        }
    }

    #[test]
    fn padding_wire_strings_round_trip() {
        assert_eq!(padding_as_wire_str(&Padding::Same), Some("same"));
        assert_eq!(padding_as_wire_str(&Padding::Valid), Some("valid"));
        assert_eq!(padding_as_wire_str(&Padding::pair(1, 1)), None);
        assert_eq!(padding_from_wire_str("same").unwrap(), Padding::Same);
        assert_eq!(padding_from_wire_str("valid").unwrap(), Padding::Valid);
    }

    #[test]
    fn padding_from_unknown_string_is_rejected() {
        let err = padding_from_wire_str("SAME").unwrap_err();
        assert!(matches!(err, NirError::InvalidGraph(_)));
        assert!(err.to_string().contains("\"SAME\""));
    }

    #[test]
    fn node_names_with_dots_are_allowed() {
        // Real upstream fixtures use names like "lif1.lif".
        assert!(check_link_name("node name", "lif1.lif").is_ok());
        assert!(check_link_name("node name", "0").is_ok());
    }

    #[test]
    fn illegal_link_names_are_rejected() {
        for bad in ["", "a/b", ".", "..", "nul\0inside"] {
            let err = check_link_name("node name", bad).unwrap_err();
            assert!(
                matches!(err, NirError::InvalidGraph(_)),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn the_kind_label_appears_in_the_error() {
        let err = check_link_name("metadata key", "a/b").unwrap_err();
        assert!(err.to_string().contains("metadata key"), "got {err}");
    }

    #[test]
    fn nul_bytes_in_string_values_are_rejected() {
        let err = check_hdf5_string("version", "0.2\0.0").unwrap_err();
        assert!(matches!(err, NirError::InvalidGraph(_)));
        assert!(err.to_string().contains("version"), "got {err}");
        assert!(err.to_string().contains("NUL"), "got {err}");
    }
}
