// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::typecheck::shape::{Pad, Window, flatten_output, matrix_types, pool_output, py_slice};

#[test]
fn non_square_kernel_uses_each_axis() {
    // 8 - (3-1) = 6 rows, 8 - (5-1) = 4 cols.
    let g = chain(vec![
        ("in", input(&[1, 8, 8])),
        ("conv", conv2d(2, 1, (3, 5), None)),
        ("out", output(&[2, 6, 4])),
    ]);
    g.check_types().unwrap();
}

#[test]
fn conv1d_same_padding_keeps_length() {
    let conv = NirNode::Conv1d(Conv1d {
        weight: Tensor::from_f32(vec![3, 2, 5], vec![0.0; 30]).unwrap(),
        stride: vec![1],
        padding: Padding::Same,
        dilation: vec![1],
        groups: 1,
        bias: Tensor::from_f32(vec![3], vec![0.0; 3]).unwrap(),
        input_shape: None,
        metadata: MetadataMap::default(),
    });
    let g = chain(vec![
        ("in", input(&[2, 10])),
        ("conv", conv),
        ("out", output(&[3, 10])),
    ]);
    g.check_types().unwrap();
}

#[test]
fn flatten_python_dim_semantics() {
    assert_eq!(flatten_output(&[2, 3, 4], 1, -1).unwrap(), vec![2, 12]);
    assert_eq!(flatten_output(&[2, 3, 4], 0, -1).unwrap(), vec![24]);
    assert_eq!(flatten_output(&[2, 3, 4, 5], 1, 2).unwrap(), vec![2, 12, 5]);
    assert_eq!(flatten_output(&[2, 3, 4], 0, 2).unwrap(), vec![24]);
    assert_eq!(
        flatten_output(&[2, 3, 4, 5], 1, -2).unwrap(),
        vec![2, 12, 5]
    );
}

#[test]
fn affine_ports_follow_weight_axes() {
    let t = matrix_types(&Tensor::from_f32(vec![4, 3], vec![0.0; 12]).unwrap()).unwrap();
    assert_eq!(t, NodeTypes::leaf(Some(vec![3]), Some(vec![4])));
    assert!(matrix_types(&Tensor::from_f32(vec![3], vec![0.0; 3]).unwrap()).is_err());
}

#[test]
fn hostile_extents_do_not_panic() {
    let conv = |stride: i64, padding: i64, dilation: i64| {
        NirNode::Conv2d(Conv2d {
            weight: Tensor::from_f32(vec![1, 1, 1, 1], vec![0.0]).unwrap(),
            stride: vec![stride],
            padding: Padding::Explicit(vec![padding]),
            dilation: vec![dilation],
            groups: 1,
            bias: Tensor::from_f32(vec![1], vec![0.0]).unwrap(),
            input_shape: None,
            metadata: MetadataMap::default(),
        })
    };
    // Huge explicit padding drives the extent past `usize` bounds; the `i128`
    // arithmetic itself must stay panic-free.
    let g = chain(vec![
        ("in", input(&[1, usize::MAX, 4])),
        ("conv", conv(1, i64::MAX, 1)),
        ("out", output(&[1])),
    ]);
    assert!(
        inference_reason(&g).contains("out of range"),
        "extent overflow"
    );
    // Non-positive stride is rejected before any extent arithmetic runs.
    let g = chain(vec![
        ("in", input(&[1, usize::MAX, 4])),
        ("conv", conv(i64::MIN, 0, 1)),
        ("out", output(&[1])),
    ]);
    assert!(
        inference_reason(&g).contains("stride must be positive"),
        "stride guard"
    );
}

#[test]
fn pool_extent_errors() {
    let ints = |v: &[i64]| Tensor::from_i64(vec![v.len()], v.to_vec()).unwrap();
    let ok = ints(&[1, 1]);
    assert!(
        pool_output(&[1, 4, 4], &ints(&[0, 2]), &ok, &ok)
            .unwrap_err()
            .contains("positive")
    );
    assert!(
        pool_output(&[1, 4, 4], &ints(&[]), &ok, &ok)
            .unwrap_err()
            .contains("is empty")
    );
    assert_eq!(
        pool_output(&[1, 4, 4], &ints(&[2, 2]), &ints(&[2, 2]), &ints(&[0, 0])).unwrap(),
        vec![1, 2, 2]
    );
}

#[test]
fn window_extent_errors() {
    let window = |padding, dilation, kernel, stride| Window {
        padding,
        dilation,
        kernel,
        stride,
    };
    let ok = window(Pad::Valid, &[1], &[3], &[1]);
    assert_eq!(ok.output(&[5]).unwrap(), vec![3]);
    assert_eq!(
        window(Pad::Same, &[1], &[3], &[2]).output(&[5]).unwrap(),
        vec![5]
    );
    assert!(window(Pad::Valid, &[1], &[3], &[0]).output(&[5]).is_err());
    assert!(
        window(Pad::Valid, &[0], &[3], &[1])
            .output(&[5])
            .unwrap_err()
            .contains("dilation must be positive")
    );
    assert!(
        window(Pad::Valid, &[1], &[3, 3, 3], &[1])
            .output(&[5, 5])
            .unwrap_err()
            .contains("kernel has 3")
    );
    assert!(
        window(Pad::Explicit(&[1, 2, 3]), &[1], &[3], &[1])
            .output(&[5, 5])
            .unwrap_err()
            .contains("padding has 3")
    );
    assert!(
        window(Pad::Valid, &[1], &[9], &[1])
            .output(&[5])
            .unwrap_err()
            .contains("out of range")
    );
}

#[test]
fn flatten_overflow_is_an_error() {
    assert!(
        flatten_output(&[usize::MAX, 2], 0, -1)
            .unwrap_err()
            .contains("overflows")
    );
    assert_eq!(flatten_output(&[2, 3, 4], 0, 1).unwrap(), vec![6, 4]);
    assert_eq!(py_slice(3, Some(-1), None), (2, 3));
}
