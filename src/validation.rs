// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in local convolution and pooling parameter checks.
//!
//! [`NirGraph::validate_structure`] only inspects edge endpoints and duplicate
//! directed edges. This module checks invariants that are unambiguous from a
//! single node's NIR fields — weight rank, grouped-convolution divisibility,
//! stride/dilation/padding extents, bias length, and pooling windows.
//!
//! HDF5 reads stay permissive: callers invoke
//! [`NirNode::validate_parameters`] / [`NirGraph::validate_parameters`] after
//! import if they want these checks. The writer does not run them.

use crate::error::{NirError, ParameterError, Result};
use crate::graph::NirGraph;
use crate::nodes::{AvgPool2d, Conv1d, Conv2d, NirNode, Padding, SumPool2d};
use crate::types::{DType, Tensor, TensorData};

/// Sentinel node name used by isolated [`NirNode::validate_parameters`].
pub(crate) const ANONYMOUS_NODE: &str = "<node>";

/// Validate one node. Nested graphs recurse under `name` as a path prefix.
pub(crate) fn validate_node(node: &NirNode, name: &str) -> Result<()> {
    validate_node_with_depth(node, name, 0)
}

fn validate_node_with_depth(node: &NirNode, name: &str, depth: usize) -> Result<()> {
    if depth > NirGraph::MAX_NESTING_DEPTH {
        return Err(NirError::InvalidGraph(format!(
            "graph nesting depth exceeds {}",
            NirGraph::MAX_NESTING_DEPTH
        )));
    }
    match node {
        NirNode::Conv1d(conv) => validate_conv1d(name, conv),
        NirNode::Conv2d(conv) => validate_conv2d(name, conv),
        NirNode::SumPool2d(pool) => validate_sum_pool2d(name, pool),
        NirNode::AvgPool2d(pool) => validate_avg_pool2d(name, pool),
        NirNode::Graph(sub) => validate_graph_with_prefix_and_depth(sub, Some(name), depth + 1),
        NirNode::Input(_)
        | NirNode::Output(_)
        | NirNode::Affine(_)
        | NirNode::Linear(_)
        | NirNode::Scale(_)
        | NirNode::CubaLi(_)
        | NirNode::CubaLif(_)
        | NirNode::Delay(_)
        | NirNode::Flatten(_)
        | NirNode::I(_)
        | NirNode::If(_)
        | NirNode::Li(_)
        | NirNode::Lif(_)
        | NirNode::Threshold(_) => Ok(()),
    }
}

/// Validate every node in `graph`, qualifying nested subgraphs with `/`.
pub(crate) fn validate_graph(graph: &NirGraph) -> Result<()> {
    validate_graph_with_prefix_and_depth(graph, None, 0)
}

fn validate_graph_with_prefix_and_depth(
    graph: &NirGraph,
    prefix: Option<&str>,
    depth: usize,
) -> Result<()> {
    if depth > NirGraph::MAX_NESTING_DEPTH {
        return Err(NirError::InvalidGraph(format!(
            "graph nesting depth exceeds {}",
            NirGraph::MAX_NESTING_DEPTH
        )));
    }
    for (name, node) in &graph.nodes {
        let qualified = match prefix {
            Some(parent) => format!("{parent}/{name}"),
            None => name.clone(),
        };
        validate_node_with_depth(node, &qualified, depth)?;
    }
    Ok(())
}

fn invalid(node: &str, node_type: &'static str, kind: ParameterError) -> NirError {
    NirError::InvalidNodeParameters {
        node: node.to_owned(),
        node_type,
        kind,
    }
}

fn validate_conv1d(node: &str, conv: &Conv1d) -> Result<()> {
    validate_conv(
        node,
        "Conv1d",
        &conv.weight,
        3,
        &conv.stride,
        &conv.padding,
        &conv.dilation,
        conv.groups,
        &conv.bias,
        &[1],
    )
}

fn validate_conv2d(node: &str, conv: &Conv2d) -> Result<()> {
    validate_conv(
        node,
        "Conv2d",
        &conv.weight,
        4,
        &conv.stride,
        &conv.padding,
        &conv.dilation,
        conv.groups,
        &conv.bias,
        &[1, 2],
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_conv(
    node: &str,
    node_type: &'static str,
    weight: &Tensor,
    expected_rank: usize,
    stride: &[i64],
    padding: &Padding,
    dilation: &[i64],
    groups: i64,
    bias: &Tensor,
    spatial_arity: &'static [usize],
) -> Result<()> {
    if groups <= 0 {
        return Err(invalid(
            node,
            node_type,
            ParameterError::Groups { found: groups },
        ));
    }

    let rank = weight.ndim();
    if rank != expected_rank {
        return Err(invalid(
            node,
            node_type,
            ParameterError::WeightRank {
                found: rank,
                expected: expected_rank,
            },
        ));
    }

    for (axis, &extent) in weight.shape().iter().enumerate() {
        if extent == 0 {
            return Err(invalid(
                node,
                node_type,
                ParameterError::WeightExtentPositive { axis },
            ));
        }
    }

    let out_channels = weight.shape()[0];
    if !(out_channels as u64).is_multiple_of(groups as u64) {
        return Err(invalid(
            node,
            node_type,
            ParameterError::ChannelGroupDivisibility {
                channels: out_channels,
                groups,
            },
        ));
    }

    check_vec_extents(
        node,
        node_type,
        "stride",
        stride,
        spatial_arity,
        ExtentSign::Positive,
    )?;
    check_vec_extents(
        node,
        node_type,
        "dilation",
        dilation,
        spatial_arity,
        ExtentSign::Positive,
    )?;
    check_padding(node, node_type, padding, spatial_arity)?;

    let bias_rank = bias.ndim();
    if bias_rank != 1 {
        return Err(invalid(
            node,
            node_type,
            ParameterError::BiasRank { found: bias_rank },
        ));
    }

    let bias_len = bias.numel();
    if bias_len != out_channels {
        return Err(invalid(
            node,
            node_type,
            ParameterError::BiasLength {
                found: bias_len,
                expected: out_channels,
            },
        ));
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum ExtentSign {
    Positive,
    NonNegative,
}

fn arity_label(allowed: &[usize]) -> &'static str {
    match allowed {
        [1] => "1",
        [1, 2] => "1 or 2",
        _ => "a valid arity",
    }
}

fn check_vec_extents(
    node: &str,
    node_type: &'static str,
    field: &'static str,
    extents: &[i64],
    allowed: &[usize],
    sign: ExtentSign,
) -> Result<()> {
    if !allowed.contains(&extents.len()) {
        return Err(invalid(
            node,
            node_type,
            ParameterError::ExtentArity {
                field,
                expected: arity_label(allowed),
                found: extents.len(),
            },
        ));
    }
    check_extent_values(node, node_type, field, extents, sign)
}

fn check_padding(
    node: &str,
    node_type: &'static str,
    padding: &Padding,
    spatial_arity: &[usize],
) -> Result<()> {
    match padding {
        Padding::Same | Padding::Valid => Ok(()),
        Padding::Explicit(extents) => check_vec_extents(
            node,
            node_type,
            "padding",
            extents,
            spatial_arity,
            ExtentSign::NonNegative,
        ),
    }
}

fn check_extent_values(
    node: &str,
    node_type: &'static str,
    field: &'static str,
    extents: &[i64],
    sign: ExtentSign,
) -> Result<()> {
    for (index, &value) in extents.iter().enumerate() {
        match sign {
            ExtentSign::Positive if value <= 0 => {
                return Err(invalid(
                    node,
                    node_type,
                    ParameterError::ExtentPositive {
                        field,
                        index,
                        value,
                    },
                ));
            }
            ExtentSign::NonNegative if value < 0 => {
                return Err(invalid(
                    node,
                    node_type,
                    ParameterError::ExtentNonNegative {
                        field,
                        index,
                        value,
                    },
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_sum_pool2d(node: &str, pool: &SumPool2d) -> Result<()> {
    validate_pool2d(
        node,
        "SumPool2d",
        &pool.kernel_size,
        &pool.stride,
        &pool.padding,
    )
}

fn validate_avg_pool2d(node: &str, pool: &AvgPool2d) -> Result<()> {
    validate_pool2d(
        node,
        "AvgPool2d",
        &pool.kernel_size,
        &pool.stride,
        &pool.padding,
    )
}

fn validate_pool2d(
    node: &str,
    node_type: &'static str,
    kernel_size: &Tensor,
    stride: &Tensor,
    padding: &Tensor,
) -> Result<()> {
    let kernel = pool_extents(node, node_type, "kernel_size", kernel_size)?;
    check_extent_values(node, node_type, "kernel_size", kernel, ExtentSign::Positive)?;
    let stride = pool_extents(node, node_type, "stride", stride)?;
    check_extent_values(node, node_type, "stride", stride, ExtentSign::Positive)?;
    let padding = pool_extents(node, node_type, "padding", padding)?;
    check_extent_values(node, node_type, "padding", padding, ExtentSign::NonNegative)
}

fn pool_extents<'a>(
    node: &str,
    node_type: &'static str,
    field: &'static str,
    tensor: &'a Tensor,
) -> Result<&'a [i64]> {
    let TensorData::I64(values) = tensor.data() else {
        return Err(invalid(
            node,
            node_type,
            ParameterError::ExtentDType {
                field,
                dtype: dtype_label(tensor.dtype()),
            },
        ));
    };
    if tensor.ndim() > 1 {
        return Err(invalid(
            node,
            node_type,
            ParameterError::ExtentRank {
                field,
                found: tensor.ndim(),
            },
        ));
    }
    if !matches!(values.len(), 1 | 2) {
        return Err(invalid(
            node,
            node_type,
            ParameterError::ExtentArity {
                field,
                expected: "1 or 2",
                found: values.len(),
            },
        ));
    }
    Ok(values)
}

fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::I64 => "i64",
        DType::Bool => "bool",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{Input, Lif};
    use crate::types::{MetadataMap, Tensor};

    fn conv1d_ok() -> Conv1d {
        Conv1d {
            weight: Tensor::from_f32(vec![2, 1, 3], vec![0.; 6]).unwrap(),
            stride: vec![1],
            padding: Padding::Valid,
            dilation: vec![1],
            groups: 1,
            bias: Tensor::from_f32(vec![2], vec![0., 0.]).unwrap(),
            input_shape: None,
            metadata: MetadataMap::default(),
        }
    }

    #[test]
    fn anonymous_leaf_uses_sentinel_name() {
        let mut conv = conv1d_ok();
        conv.groups = 0;
        let err = NirNode::Conv1d(conv).validate_parameters().unwrap_err();
        match err {
            NirError::InvalidNodeParameters {
                node, node_type, ..
            } => {
                assert_eq!(node, ANONYMOUS_NODE);
                assert_eq!(node_type, "Conv1d");
            }
            other => panic!("expected InvalidNodeParameters, got {other:?}"),
        }
    }

    #[test]
    fn non_conv_nodes_are_skipped() {
        let node = NirNode::Input(Input {
            shape: vec![4],
            metadata: MetadataMap::default(),
        });
        node.validate_parameters().unwrap();

        let lif = NirNode::Lif(Lif {
            tau: Tensor::scalar_f64(1.0),
            r: Tensor::scalar_f64(1.0),
            v_leak: Tensor::scalar_f64(0.0),
            v_threshold: Tensor::scalar_f64(1.0),
            v_reset: None,
            metadata: MetadataMap::default(),
        });
        lif.validate_parameters().unwrap();
    }

    #[test]
    fn zero_extent_in_weight_is_rejected() {
        let mut conv = conv1d_ok();
        conv.weight = Tensor::from_f32(vec![2, 1, 0], vec![]).unwrap();
        let err = NirNode::Conv1d(conv).validate_parameters().unwrap_err();
        match err {
            NirError::InvalidNodeParameters { kind, .. } => {
                assert_eq!(kind, ParameterError::WeightExtentPositive { axis: 2 });
            }
            other => panic!("expected InvalidNodeParameters, got {other:?}"),
        }
    }

    #[test]
    fn non_1d_bias_is_rejected_even_with_matching_numel() {
        let mut conv = conv1d_ok();
        // shape [2, 1] has numel 2 which equals out_channels (2), but rank is 2
        conv.bias = Tensor::from_f32(vec![2, 1], vec![0., 0.]).unwrap();
        let err = NirNode::Conv1d(conv).validate_parameters().unwrap_err();
        match err {
            NirError::InvalidNodeParameters { kind, .. } => {
                assert_eq!(kind, ParameterError::BiasRank { found: 2 });
            }
            other => panic!("expected InvalidNodeParameters, got {other:?}"),
        }
    }

    #[test]
    fn nesting_beyond_limit_in_validate_parameters_is_rejected() {
        let mut curr = NirGraph::default();
        for i in 0..=NirGraph::MAX_NESTING_DEPTH {
            let mut parent = NirGraph::default();
            parent
                .nodes
                .insert(format!("sub_{i}"), NirNode::Graph(Box::new(curr)));
            curr = parent;
        }
        let err = curr.validate_parameters().unwrap_err();
        match err {
            NirError::InvalidGraph(msg) => {
                assert!(msg.contains("nesting depth"));
            }
            other => panic!("expected InvalidGraph, got {other:?}"),
        }
    }
}
