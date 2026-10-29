// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn matching_chain_passes_and_is_not_mutated() {
    let g = chain(vec![
        ("in", input(&[2])),
        ("lif", lif(&[2])),
        ("out", output(&[2])),
    ]);
    let before = g.clone();
    g.check_types().unwrap();
    assert_eq!(g, before);
}

#[test]
fn mismatch_names_both_nodes_types_and_shapes() {
    let g = chain(vec![
        ("in", input(&[1])),
        ("lif", lif(&[1])),
        ("out", output(&[1, 1, 1])),
    ]);
    let err = g.check_types().unwrap_err().to_string();
    assert!(err.contains("lif (LIF).output [1]"), "{err}");
    assert!(err.contains("out (Output).input [1, 1, 1]"), "{err}");
}

#[test]
fn conv_pool_flatten_chain_is_inferred() {
    // 1x8x8 -conv3x3-> 4x6x6 -pool2-> 4x3x3 -flatten-> 4x9
    let g = chain(vec![
        ("in", input(&[1, 8, 8])),
        ("conv", conv2d(4, 1, (3, 3), None)),
        ("pool", pool(2)),
        ("flat", flatten(None)),
        ("out", output(&[4, 9])),
    ]);
    let inferred = g.infer_types().unwrap();
    assert_eq!(inferred.nodes["conv"].output, vec![Some(vec![4, 6, 6])]);
    assert_eq!(inferred.nodes["pool"].output, vec![Some(vec![4, 3, 3])]);
    assert_eq!(inferred.nodes["flat"].output, vec![Some(vec![4, 9])]);
    g.check_types().unwrap();
}

#[test]
fn conv_with_declared_input_shape_conflicts_with_producer() {
    let g = chain(vec![
        ("in", input(&[1, 9, 9])),
        ("conv", conv2d(2, 1, (3, 3), Some(vec![8, 8]))),
        ("out", output(&[2, 6, 6])),
    ]);
    assert!(matches!(type_error(&g), TypeError::EdgeMismatch { .. }));
}

#[test]
fn root_without_defined_input_is_rejected() {
    let g = chain(vec![
        ("conv", conv2d(2, 1, (3, 3), None)),
        ("out", output(&[2, 6, 6])),
    ]);
    assert!(matches!(type_error(&g), TypeError::UndefinedInput { .. }));
}

#[test]
fn lone_input_node_cannot_start_inference() {
    let mut g = NirGraph::new();
    g.insert_node("in", input(&[1])).unwrap();
    assert!(matches!(type_error(&g), TypeError::NoInputs { .. }));
}

#[test]
fn cycle_without_exit_is_rejected() {
    let mut g = chain(vec![
        ("in", input(&[1])),
        ("a", lif(&[1])),
        ("b", lif(&[1])),
    ]);
    g.add_edge("b", "a");
    assert!(matches!(type_error(&g), TypeError::NoOutputs { .. }));
}

#[test]
fn empty_graph_passes() {
    NirGraph::new().check_types().unwrap();
}

#[test]
fn structure_errors_come_first() {
    let mut g = chain(vec![("in", input(&[1])), ("out", output(&[1]))]);
    g.add_edge("in", "ghost");
    assert_eq!(
        g.check_types().unwrap_err(),
        NirError::MissingNode("ghost".into())
    );
}

#[test]
fn nested_graph_ports_are_checked() {
    let sub = chain(vec![
        ("sin", input(&[2])),
        ("lif", lif(&[2])),
        ("sout", output(&[2])),
    ]);
    let mut g = NirGraph::new();
    g.insert_node("in", input(&[2])).unwrap();
    g.insert_node("sub", NirNode::Graph(Box::new(sub))).unwrap();
    g.insert_node("out", output(&[2])).unwrap();
    g.add_edge("in", "sub");
    g.add_edge("sub", "out");
    g.check_types().unwrap();

    let mut bad = g.clone();
    bad.nodes.insert("out".into(), output(&[5]));
    assert!(matches!(type_error(&bad), TypeError::EdgeMismatch { .. }));
}

#[test]
fn nested_mismatch_is_path_qualified() {
    let sub = chain(vec![
        ("sin", input(&[2])),
        ("lif", lif(&[2])),
        ("sout", output(&[3])),
    ]);
    let mut g = NirGraph::new();
    g.insert_node("in", input(&[2])).unwrap();
    g.insert_node("sub", NirNode::Graph(Box::new(sub))).unwrap();
    g.add_edge("in", "sub");
    let err = g.check_types().unwrap_err().to_string();
    assert!(err.contains("sub/lif"), "{err}");
}

#[test]
fn deep_nesting_is_bounded() {
    let mut g = chain(vec![("in", input(&[1])), ("out", output(&[1]))]);
    for _ in 0..(NirGraph::MAX_TYPE_CHECK_DEPTH + 2) {
        let mut outer = NirGraph::new();
        outer
            .insert_node("sub", NirNode::Graph(Box::new(g)))
            .unwrap();
        g = outer;
    }
    assert!(matches!(type_error(&g), TypeError::NestingTooDeep { .. }));
}

#[test]
fn invalid_node_parameters_become_inference_errors() {
    let bad_declared = chain(vec![
        ("in", input(&[5])),
        ("conv", conv1d(zeros(&[2, 2]), Padding::Valid, Some(5))),
    ]);
    assert!(inference_reason(&bad_declared).contains("weight rank"));

    let float_pool = NirNode::SumPool2d(SumPool2d {
        kernel_size: zeros(&[2]),
        stride: Tensor::from_i64(vec![2], vec![1, 1]).unwrap(),
        padding: Tensor::from_i64(vec![2], vec![0, 0]).unwrap(),
        metadata: Default::default(),
    });
    let g = chain(vec![("in", input(&[1, 4, 4])), ("pool", float_pool)]);
    assert!(inference_reason(&g).contains("integer tensor"));

    let g = chain(vec![("in", input(&[4, 4])), ("pool", pool(2))]);
    assert!(inference_reason(&g).contains("is not 3"));

    let g = chain(vec![
        ("in", input(&[1, 4, 4])),
        ("conv", conv2d(1, 1, (3, 3), None)),
        ("flat", flatten(None)),
    ]);
    g.check_types().unwrap();
    let g = chain(vec![
        ("in", input(&[4])),
        ("conv", conv2d(1, 1, (3, 3), None)),
    ]);
    assert!(inference_reason(&g).contains("input rank"));
}

#[test]
fn multi_port_subgraph_as_root_is_unsupported() {
    let mut g = NirGraph::new();
    g.insert_node("sub", NirNode::Graph(Box::new(two_input_graph())))
        .unwrap();
    assert!(matches!(type_error(&g), TypeError::MultiPort { .. }));
}

#[test]
fn multi_port_subgraph_as_leaf_is_unsupported() {
    let mut sub = NirGraph::new();
    sub.insert_node("i", input(&[1])).unwrap();
    sub.insert_node("a", output(&[1])).unwrap();
    sub.insert_node("b", output(&[1])).unwrap();
    sub.add_edge("i", "a");
    sub.add_edge("i", "b");
    let mut g = chain(vec![("in", input(&[1]))]);
    g.insert_node("sub", NirNode::Graph(Box::new(sub))).unwrap();
    g.add_edge("in", "sub");
    assert!(matches!(type_error(&g), TypeError::MultiPort { .. }));
}

#[test]
fn check_edge_reports_each_port_defect() {
    let g = chain(vec![("in", input(&[1])), ("out", output(&[1]))]);
    let scope = Scope {
        graph: &g,
        prefix: "p",
    };
    let run = |pre: NodeTypes, post: NodeTypes| {
        let mut inferred = InferredTypes::default();
        inferred.nodes.insert("in".into(), pre);
        inferred.nodes.insert("out".into(), post);
        match scope.check_edge(&inferred, "in", "out").unwrap_err() {
            NirError::TypeCheck(e) => *e,
            other => panic!("{other:?}"),
        }
    };
    let known = || NodeTypes::leaf(Some(vec![1]), Some(vec![1]));
    assert!(matches!(
        run(NodeTypes::leaf(Some(vec![1]), None), known()),
        TypeError::UndefinedOutput { node, .. } if node == "p/in"
    ));
    assert!(matches!(
        run(known(), NodeTypes::leaf(None, Some(vec![1]))),
        TypeError::UndefinedInput { .. }
    ));
    let two = NodeTypes {
        input: vec![Some(vec![1]); 2],
        output: vec![Some(vec![1]); 2],
    };
    assert!(matches!(
        run(known(), two.clone()),
        TypeError::PortCountMismatch { .. }
    ));
    assert!(matches!(run(two.clone(), two), TypeError::MultiPort { .. }));
}
