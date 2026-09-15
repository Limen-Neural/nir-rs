// SPDX-License-Identifier: MIT OR Apache-2.0

//! Table-driven coverage of convolution and pooling parameter validation.
//!
//! Every rejection and acceptance class from LIM-1239 lives here so
//! `cargo test validation` is a complete filter.

use nir_rs::nodes::{
    Affine, AvgPool2d, Conv1d, Conv2d, Input, NirNode, Output, Padding, SumPool2d,
};
use nir_rs::types::Tensor;
use nir_rs::{NirError, NirGraph, ParameterError};

fn conv1d_ok() -> Conv1d {
    Conv1d {
        weight: Tensor::from_f32(vec![2, 1, 3], vec![0.; 6]).unwrap(),
        stride: vec![1],
        padding: Padding::Valid,
        dilation: vec![1],
        groups: 1,
        bias: Tensor::from_f32(vec![2], vec![0., 0.]).unwrap(),
        input_shape: Some(8),
        metadata: Default::default(),
    }
}

fn conv2d_ok() -> Conv2d {
    Conv2d {
        weight: Tensor::from_f32(vec![4, 2, 3, 3], vec![0.; 72]).unwrap(),
        stride: vec![1, 1],
        padding: Padding::pair(0, 0),
        dilation: vec![1, 1],
        groups: 1,
        bias: Tensor::from_f32(vec![4], vec![0.; 4]).unwrap(),
        input_shape: Some(vec![8, 8]),
        metadata: Default::default(),
    }
}

fn pool_ok() -> (Tensor, Tensor, Tensor) {
    (
        Tensor::from_i64(vec![2], vec![2, 2]).unwrap(),
        Tensor::from_i64(vec![2], vec![2, 2]).unwrap(),
        Tensor::from_i64(vec![2], vec![0, 0]).unwrap(),
    )
}

fn sum_pool_ok() -> SumPool2d {
    let (kernel_size, stride, padding) = pool_ok();
    SumPool2d {
        kernel_size,
        stride,
        padding,
        metadata: Default::default(),
    }
}

fn avg_pool_ok() -> AvgPool2d {
    let (kernel_size, stride, padding) = pool_ok();
    AvgPool2d {
        kernel_size,
        stride,
        padding,
        metadata: Default::default(),
    }
}

fn assert_kind(node: NirNode, expected: ParameterError) {
    let err = node.validate_parameters().unwrap_err();
    match err {
        NirError::InvalidNodeParameters { kind, .. } => {
            assert_eq!(kind, expected);
        }
        other => panic!("expected InvalidNodeParameters, got {other:?}"),
    }
}

fn assert_ok(node: NirNode) {
    node.validate_parameters()
        .unwrap_or_else(|e| panic!("expected Ok, got {e}"));
}

#[derive(Clone, Copy)]
enum Expect {
    Ok,
    Err(&'static str),
}

/// Rejection and acceptance classes, keyed by a stable case name.
fn cases() -> Vec<(&'static str, NirNode, Expect)> {
    vec![
        // --- acceptance ---
        (
            "conv1d_valid_padding",
            NirNode::Conv1d(conv1d_ok()),
            Expect::Ok,
        ),
        (
            "conv1d_same_padding",
            {
                let mut conv = conv1d_ok();
                conv.padding = Padding::Same;
                NirNode::Conv1d(conv)
            },
            Expect::Ok,
        ),
        (
            "conv1d_explicit_padding",
            {
                let mut conv = conv1d_ok();
                conv.padding = Padding::single(1);
                NirNode::Conv1d(conv)
            },
            Expect::Ok,
        ),
        (
            "conv1d_depthwise",
            {
                let mut conv = conv1d_ok();
                conv.weight = Tensor::from_f32(vec![4, 1, 3], vec![0.; 12]).unwrap();
                conv.groups = 4;
                conv.bias = Tensor::from_f32(vec![4], vec![0.; 4]).unwrap();
                NirNode::Conv1d(conv)
            },
            Expect::Ok,
        ),
        (
            "conv2d_explicit_pair",
            NirNode::Conv2d(conv2d_ok()),
            Expect::Ok,
        ),
        (
            "conv2d_same_padding",
            {
                let mut conv = conv2d_ok();
                conv.padding = Padding::Same;
                NirNode::Conv2d(conv)
            },
            Expect::Ok,
        ),
        (
            "conv2d_valid_padding",
            {
                let mut conv = conv2d_ok();
                conv.padding = Padding::Valid;
                NirNode::Conv2d(conv)
            },
            Expect::Ok,
        ),
        (
            "conv2d_scalar_stride_dilation_padding",
            {
                let mut conv = conv2d_ok();
                conv.stride = vec![2];
                conv.dilation = vec![1];
                conv.padding = Padding::single(0);
                NirNode::Conv2d(conv)
            },
            Expect::Ok,
        ),
        (
            "conv2d_grouped",
            {
                let mut conv = conv2d_ok();
                conv.weight = Tensor::from_f32(vec![8, 2, 3, 3], vec![0.; 144]).unwrap();
                conv.groups = 4;
                conv.bias = Tensor::from_f32(vec![8], vec![0.; 8]).unwrap();
                NirNode::Conv2d(conv)
            },
            Expect::Ok,
        ),
        (
            "conv2d_depthwise",
            {
                let mut conv = conv2d_ok();
                conv.weight = Tensor::from_f32(vec![6, 1, 3, 3], vec![0.; 54]).unwrap();
                conv.groups = 6;
                conv.bias = Tensor::from_f32(vec![6], vec![0.; 6]).unwrap();
                NirNode::Conv2d(conv)
            },
            Expect::Ok,
        ),
        (
            "sum_pool2d_ok",
            NirNode::SumPool2d(sum_pool_ok()),
            Expect::Ok,
        ),
        (
            "avg_pool2d_ok",
            NirNode::AvgPool2d(avg_pool_ok()),
            Expect::Ok,
        ),
        (
            "pool_scalar_windows",
            {
                NirNode::SumPool2d(SumPool2d {
                    kernel_size: Tensor::scalar_i64(3),
                    stride: Tensor::scalar_i64(2),
                    padding: Tensor::scalar_i64(0),
                    metadata: Default::default(),
                })
            },
            Expect::Ok,
        ),
        (
            "affine_skipped",
            NirNode::Affine(Affine {
                weight: Tensor::from_f32(vec![2, 2], vec![0.; 4]).unwrap(),
                bias: Tensor::from_f32(vec![2], vec![0., 0.]).unwrap(),
                metadata: Default::default(),
            }),
            Expect::Ok,
        ),
        (
            "input_skipped",
            NirNode::Input(Input {
                shape: vec![1],
                metadata: Default::default(),
            }),
            Expect::Ok,
        ),
        // --- rejection: conv rank ---
        (
            "conv1d_weight_rank",
            {
                let mut conv = conv1d_ok();
                conv.weight = Tensor::from_f32(vec![2, 3], vec![0.; 6]).unwrap();
                NirNode::Conv1d(conv)
            },
            Expect::Err("weight rank"),
        ),
        (
            "conv2d_weight_rank",
            {
                let mut conv = conv2d_ok();
                conv.weight = Tensor::from_f32(vec![4, 2, 3], vec![0.; 24]).unwrap();
                NirNode::Conv2d(conv)
            },
            Expect::Err("weight rank"),
        ),
        // --- rejection: groups ---
        (
            "conv1d_groups_zero",
            {
                let mut conv = conv1d_ok();
                conv.groups = 0;
                NirNode::Conv1d(conv)
            },
            Expect::Err("groups must be > 0"),
        ),
        (
            "conv2d_groups_negative",
            {
                let mut conv = conv2d_ok();
                conv.groups = -1;
                NirNode::Conv2d(conv)
            },
            Expect::Err("groups must be > 0"),
        ),
        (
            "conv1d_groups_not_dividing_out_channels",
            {
                let mut conv = conv1d_ok();
                conv.weight = Tensor::from_f32(vec![3, 1, 3], vec![0.; 9]).unwrap();
                conv.groups = 2;
                conv.bias = Tensor::from_f32(vec![3], vec![0.; 3]).unwrap();
                NirNode::Conv1d(conv)
            },
            Expect::Err("not divisible by groups"),
        ),
        (
            "conv2d_groups_not_dividing_out_channels",
            {
                let mut conv = conv2d_ok();
                conv.weight = Tensor::from_f32(vec![3, 1, 3, 3], vec![0.; 27]).unwrap();
                conv.groups = 2;
                conv.bias = Tensor::from_f32(vec![3], vec![0.; 3]).unwrap();
                NirNode::Conv2d(conv)
            },
            Expect::Err("not divisible by groups"),
        ),
        // --- rejection: stride / dilation arity and sign ---
        (
            "conv1d_stride_arity",
            {
                let mut conv = conv1d_ok();
                conv.stride = vec![1, 1];
                NirNode::Conv1d(conv)
            },
            Expect::Err("stride arity"),
        ),
        (
            "conv1d_stride_zero",
            {
                let mut conv = conv1d_ok();
                conv.stride = vec![0];
                NirNode::Conv1d(conv)
            },
            Expect::Err("strictly positive"),
        ),
        (
            "conv1d_dilation_empty",
            {
                let mut conv = conv1d_ok();
                conv.dilation = vec![];
                NirNode::Conv1d(conv)
            },
            Expect::Err("dilation arity"),
        ),
        (
            "conv1d_dilation_negative",
            {
                let mut conv = conv1d_ok();
                conv.dilation = vec![-1];
                NirNode::Conv1d(conv)
            },
            Expect::Err("strictly positive"),
        ),
        (
            "conv2d_stride_arity",
            {
                let mut conv = conv2d_ok();
                conv.stride = vec![1, 1, 1];
                NirNode::Conv2d(conv)
            },
            Expect::Err("stride arity"),
        ),
        (
            "conv2d_dilation_zero",
            {
                let mut conv = conv2d_ok();
                conv.dilation = vec![1, 0];
                NirNode::Conv2d(conv)
            },
            Expect::Err("strictly positive"),
        ),
        // --- rejection: explicit padding ---
        (
            "conv1d_padding_arity",
            {
                let mut conv = conv1d_ok();
                conv.padding = Padding::pair(0, 0);
                NirNode::Conv1d(conv)
            },
            Expect::Err("padding arity"),
        ),
        (
            "conv1d_padding_negative",
            {
                let mut conv = conv1d_ok();
                conv.padding = Padding::single(-1);
                NirNode::Conv1d(conv)
            },
            Expect::Err("non-negative"),
        ),
        (
            "conv2d_padding_arity",
            {
                let mut conv = conv2d_ok();
                conv.padding = Padding::Explicit(vec![1, 1, 1]);
                NirNode::Conv2d(conv)
            },
            Expect::Err("padding arity"),
        ),
        (
            "conv2d_padding_negative",
            {
                let mut conv = conv2d_ok();
                conv.padding = Padding::pair(0, -2);
                NirNode::Conv2d(conv)
            },
            Expect::Err("non-negative"),
        ),
        // --- rejection: bias ---
        (
            "conv1d_bias_length",
            {
                let mut conv = conv1d_ok();
                conv.bias = Tensor::from_f32(vec![1], vec![0.]).unwrap();
                NirNode::Conv1d(conv)
            },
            Expect::Err("bias length"),
        ),
        (
            "conv2d_bias_length",
            {
                let mut conv = conv2d_ok();
                conv.bias = Tensor::from_f32(vec![2], vec![0., 0.]).unwrap();
                NirNode::Conv2d(conv)
            },
            Expect::Err("bias length"),
        ),
        // --- rejection: pooling ---
        (
            "sum_pool_kernel_zero",
            {
                let mut pool = sum_pool_ok();
                pool.kernel_size = Tensor::from_i64(vec![2], vec![0, 2]).unwrap();
                NirNode::SumPool2d(pool)
            },
            Expect::Err("strictly positive"),
        ),
        (
            "sum_pool_stride_negative",
            {
                let mut pool = sum_pool_ok();
                pool.stride = Tensor::from_i64(vec![2], vec![2, -1]).unwrap();
                NirNode::SumPool2d(pool)
            },
            Expect::Err("strictly positive"),
        ),
        (
            "sum_pool_padding_negative",
            {
                let mut pool = sum_pool_ok();
                pool.padding = Tensor::from_i64(vec![2], vec![0, -1]).unwrap();
                NirNode::SumPool2d(pool)
            },
            Expect::Err("non-negative"),
        ),
        (
            "sum_pool_kernel_arity",
            {
                let mut pool = sum_pool_ok();
                pool.kernel_size = Tensor::from_i64(vec![3], vec![2, 2, 2]).unwrap();
                NirNode::SumPool2d(pool)
            },
            Expect::Err("kernel_size arity"),
        ),
        (
            "sum_pool_kernel_rank",
            {
                let mut pool = sum_pool_ok();
                pool.kernel_size = Tensor::from_i64(vec![1, 2], vec![2, 2]).unwrap();
                NirNode::SumPool2d(pool)
            },
            Expect::Err("kernel_size rank"),
        ),
        (
            "avg_pool_stride_dtype",
            {
                let mut pool = avg_pool_ok();
                pool.stride = Tensor::from_f32(vec![2], vec![2.0, 2.0]).unwrap();
                NirNode::AvgPool2d(pool)
            },
            Expect::Err("i64 extents"),
        ),
        (
            "avg_pool_padding_arity",
            {
                let mut pool = avg_pool_ok();
                pool.padding = Tensor::from_i64(vec![0], vec![]).unwrap();
                NirNode::AvgPool2d(pool)
            },
            Expect::Err("padding arity"),
        ),
    ]
}

#[test]
fn validation_table_covers_every_acceptance_and_rejection_class() {
    let table = cases();
    let names: Vec<&str> = table.iter().map(|(name, _, _)| *name).collect();
    assert!(names.iter().any(|n| n.contains("depthwise")));
    assert!(names.iter().any(|n| n.contains("grouped")));
    assert!(names.iter().any(|n| n.contains("same_padding")));
    assert!(names.iter().any(|n| n.contains("valid_padding")));
    assert!(names.iter().any(|n| n.contains("weight_rank")));
    assert!(names.iter().any(|n| n.contains("groups_zero")));
    assert!(names.iter().any(|n| n.contains("not_dividing")));
    assert!(names.iter().any(|n| n.contains("stride")));
    assert!(names.iter().any(|n| n.contains("dilation")));
    assert!(names.iter().any(|n| n.contains("padding_negative")));
    assert!(names.iter().any(|n| n.contains("bias_length")));
    assert!(names.iter().any(|n| n.contains("kernel_zero")));
    assert!(names.iter().any(|n| n.contains("dtype")));

    for (name, node, expect) in table {
        match expect {
            Expect::Ok => {
                assert_ok(node);
                eprintln!("{name}: ok");
            }
            Expect::Err(needle) => {
                let err = node.validate_parameters().unwrap_err();
                let msg = err.to_string();
                assert!(
                    msg.contains(needle),
                    "{name}: expected {needle:?} in {msg:?}"
                );
                assert!(
                    matches!(err, NirError::InvalidNodeParameters { .. }),
                    "{name}: typed InvalidNodeParameters, got {err:?}"
                );
                eprintln!("{name}: rejected ({msg})");
            }
        }
    }
}

#[test]
fn validation_empty_graph_ok() {
    NirGraph::new().validate_parameters().unwrap();
}

#[test]
fn validation_error_is_node_qualified() {
    let mut conv = conv1d_ok();
    conv.groups = 0;
    let mut g = NirGraph::new();
    g.insert_node("encoder", NirNode::Conv1d(conv)).unwrap();
    g.validate_structure().unwrap();
    let err = g.validate_parameters().unwrap_err();
    match err {
        NirError::InvalidNodeParameters {
            node,
            node_type,
            kind: ParameterError::Groups { found: 0 },
        } => {
            assert_eq!(node, "encoder");
            assert_eq!(node_type, "Conv1d");
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn validation_nested_graph_uses_slash_path() {
    let mut conv = conv2d_ok();
    conv.padding = Padding::pair(-1, 0);
    let mut inner = NirGraph::new();
    inner.insert_node("conv", NirNode::Conv2d(conv)).unwrap();

    let mut outer = NirGraph::new();
    outer
        .insert_node("encoder", NirNode::Graph(Box::new(inner)))
        .unwrap();
    outer
        .insert_node(
            "out",
            NirNode::Output(Output {
                shape: vec![1],
                metadata: Default::default(),
            }),
        )
        .unwrap();

    let err = outer.validate_parameters().unwrap_err();
    match err {
        NirError::InvalidNodeParameters {
            node, node_type, ..
        } => {
            assert_eq!(node, "encoder/conv");
            assert_eq!(node_type, "Conv2d");
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn structure_validation_ignores_bad_conv_parameters() {
    let mut conv = conv1d_ok();
    conv.groups = 0;
    conv.stride = vec![0];
    let mut g = NirGraph::new();
    g.insert_node(
        "in",
        NirNode::Input(Input {
            shape: vec![1],
            metadata: Default::default(),
        }),
    )
    .unwrap();
    g.insert_node("conv", NirNode::Conv1d(conv)).unwrap();
    g.add_edge("in", "conv");
    g.validate_structure()
        .expect("structure must stay independent of parameter invariants");
    assert!(g.validate_parameters().is_err());
}

#[test]
fn validation_typed_kinds_match_specific_invariants() {
    let mut conv = conv1d_ok();
    conv.weight = Tensor::from_f32(vec![2, 3], vec![0.; 6]).unwrap();
    assert_kind(
        NirNode::Conv1d(conv),
        ParameterError::WeightRank {
            found: 2,
            expected: 3,
        },
    );

    let mut conv = conv2d_ok();
    conv.bias = Tensor::from_f32(vec![1], vec![0.]).unwrap();
    assert_kind(
        NirNode::Conv2d(conv),
        ParameterError::BiasLength {
            found: 1,
            expected: 4,
        },
    );

    let mut pool = sum_pool_ok();
    pool.kernel_size = Tensor::from_i64(vec![1, 2], vec![2, 2]).unwrap();
    assert_kind(
        NirNode::SumPool2d(pool),
        ParameterError::ExtentRank {
            field: "kernel_size",
            found: 2,
        },
    );
}

#[cfg(feature = "hdf5")]
mod hdf5 {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn read_stays_permissive_for_parameter_validation() {
        let mut conv = conv1d_ok();
        conv.groups = 0;
        let mut g = NirGraph::new();
        g.insert_node("conv", NirNode::Conv1d(conv)).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("zero_groups.nir");
        nir_rs::io::write(&path, &g).expect("writer must not require parameter validation");

        let decoded = nir_rs::io::read(&path).expect("HDF5 read stays permissive");
        decoded.validate_structure().unwrap();
        let err = decoded.validate_parameters().unwrap_err();
        match err {
            NirError::InvalidNodeParameters {
                node,
                kind: ParameterError::Groups { found: 0 },
                ..
            } => assert_eq!(node, "conv"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn cnn_sinabs_fixture_passes_parameter_validation() {
        let g = nir_rs::io::read("tests/fixtures/cnn_sinabs.nir").unwrap();
        g.validate_structure().unwrap();
        g.validate_parameters()
            .expect("vendored cnn_sinabs.nir must satisfy local conv/pool invariants");
    }
}
