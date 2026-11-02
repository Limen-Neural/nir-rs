// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::nodes::{Conv1d, Conv2d, Flatten, Lif, SumPool2d};
use crate::types::MetadataMap;

mod graph;
mod repair;
mod shape_math;

fn input(shape: &[usize]) -> NirNode {
    NirNode::Input(Input {
        shape: shape.to_vec(),
        metadata: MetadataMap::default(),
    })
}

fn output(shape: &[usize]) -> NirNode {
    NirNode::Output(Output {
        shape: shape.to_vec(),
        metadata: MetadataMap::default(),
    })
}

fn lif(shape: &[usize]) -> NirNode {
    let t = |v: f32| {
        let n: usize = shape.iter().product();
        Tensor::from_f32(shape.to_vec(), vec![v; n]).unwrap()
    };
    NirNode::Lif(Lif {
        tau: t(1.0),
        r: t(1.0),
        v_leak: t(0.0),
        v_threshold: t(1.0),
        v_reset: None,
        metadata: MetadataMap::default(),
    })
}

fn conv2d(
    c_out: usize,
    c_in: usize,
    k: (usize, usize),
    input_shape: Option<Vec<usize>>,
) -> NirNode {
    NirNode::Conv2d(Conv2d {
        weight: Tensor::from_f32(
            vec![c_out, c_in, k.0, k.1],
            vec![0.0; c_out * c_in * k.0 * k.1],
        )
        .unwrap(),
        stride: vec![1],
        padding: Padding::Explicit(vec![0]),
        dilation: vec![1],
        groups: 1,
        bias: Tensor::from_f32(vec![c_out], vec![0.0; c_out]).unwrap(),
        input_shape,
        metadata: MetadataMap::default(),
    })
}

fn pool(kernel: i64) -> NirNode {
    let t = |v: i64| Tensor::from_i64(vec![2], vec![v, v]).unwrap();
    NirNode::SumPool2d(SumPool2d {
        kernel_size: t(kernel),
        stride: t(kernel),
        padding: t(0),
        metadata: MetadataMap::default(),
    })
}

fn flatten(input_type: Option<Vec<usize>>) -> NirNode {
    NirNode::Flatten(Flatten {
        start_dim: 1,
        end_dim: -1,
        input_type,
        metadata: MetadataMap::default(),
    })
}

fn chain(nodes: Vec<(&str, NirNode)>) -> NirGraph {
    let mut g = NirGraph::new();
    let names: Vec<String> = nodes.iter().map(|(n, _)| (*n).to_owned()).collect();
    for (name, node) in nodes {
        g.insert_node(name, node).unwrap();
    }
    for w in names.windows(2) {
        g.add_edge(w[0].clone(), w[1].clone());
    }
    g
}

fn type_error(g: &NirGraph) -> TypeError {
    match g.check_types().unwrap_err() {
        NirError::TypeCheck(e) => *e,
        other => panic!("expected TypeCheck, got {other:?}"),
    }
}

fn conv1d(weight: Tensor, padding: Padding, input_shape: Option<usize>) -> NirNode {
    NirNode::Conv1d(Conv1d {
        weight,
        stride: vec![1],
        padding,
        dilation: vec![1],
        groups: 1,
        bias: Tensor::from_f32(vec![1], vec![0.0]).unwrap(),
        input_shape,
        metadata: MetadataMap::default(),
    })
}

fn zeros(shape: &[usize]) -> Tensor {
    Tensor::from_f32(shape.to_vec(), vec![0.0; shape.iter().product()]).unwrap()
}

fn inference_reason(g: &NirGraph) -> String {
    match type_error(g) {
        TypeError::Inference { reason, .. } => reason,
        other => panic!("expected Inference, got {other:?}"),
    }
}

fn two_input_graph() -> NirGraph {
    let mut sub = NirGraph::new();
    sub.insert_node("a", input(&[1])).unwrap();
    sub.insert_node("b", input(&[1])).unwrap();
    sub.insert_node("o", output(&[1])).unwrap();
    sub.add_edge("a", "o");
    sub.add_edge("b", "o");
    sub
}
