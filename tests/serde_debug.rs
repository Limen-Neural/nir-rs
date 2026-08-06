// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature-gated Serde coverage for finite, debug-oriented graph exports.

#![cfg(feature = "serde")]

use nir_rs::nodes::{Input, Lif, Output};
use nir_rs::{MetadataValue, NirGraph, NirNode, Tensor};

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

fn lif() -> NirNode {
    NirNode::Lif(Lif {
        tau: Tensor::from_f64(vec![2], vec![10.0, 12.0]).unwrap(),
        r: Tensor::from_f64(vec![2], vec![1.0, 1.5]).unwrap(),
        v_leak: Tensor::from_f64(vec![2], vec![0.0, 0.0]).unwrap(),
        v_threshold: Tensor::from_f64(vec![2], vec![1.0, 1.25]).unwrap(),
        v_reset: Some(Tensor::from_f64(vec![2], vec![0.0, 0.0]).unwrap()),
        metadata: Default::default(),
    })
}

fn finite_nested_graph() -> NirGraph {
    let mut inner = NirGraph::new();
    inner.insert_node("inner_input", input(vec![2])).unwrap();
    inner.insert_node("inner_output", output(vec![2])).unwrap();
    inner.add_edge("inner_input", "inner_output");

    let mut graph = NirGraph::new();
    graph.version = Some("1.0.8".into());
    graph
        .metadata
        .insert("purpose".into(), MetadataValue::String("debug-only".into()));
    graph.insert_node("input", input(vec![2])).unwrap();
    graph.insert_node("lif", lif()).unwrap();
    graph
        .insert_node("subgraph", NirNode::Graph(Box::new(inner)))
        .unwrap();
    graph.insert_node("output", output(vec![2])).unwrap();
    graph.add_edge("input", "lif");
    graph.add_edge("lif", "subgraph");
    graph.add_edge("subgraph", "output");
    graph
}

#[test]
fn finite_graph_round_trips_through_debug_json() {
    let graph = finite_nested_graph();
    let json = serde_json::to_string_pretty(&graph).unwrap();
    let decoded: NirGraph = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded, graph);
    decoded.validate_structure().unwrap();
}

#[test]
fn representation_uses_wire_tags_and_shape_plus_typed_data() {
    let value = serde_json::to_value(finite_nested_graph()).unwrap();

    assert_eq!(value["nodes"]["lif"]["type"], "LIF");
    assert_eq!(
        value["nodes"]["lif"]["tau"]["shape"],
        serde_json::json!([2])
    );
    assert_eq!(
        value["nodes"]["lif"]["tau"]["data"],
        serde_json::json!({"F64": [10.0, 12.0]})
    );
    assert_eq!(value["nodes"]["subgraph"]["type"], "NIRGraph");
    assert_eq!(
        value["nodes"]["subgraph"]["nodes"]["inner_input"]["type"],
        "Input"
    );
}

#[test]
fn malformed_tensor_cannot_bypass_shape_invariant() {
    let malformed = serde_json::json!({
        "shape": [2],
        "data": {"F64": [1.0]}
    });
    let err = serde_json::from_value::<Tensor>(malformed).unwrap_err();
    assert!(err.to_string().contains("shape product 2"));
    assert!(err.to_string().contains("data len 1"));
}
