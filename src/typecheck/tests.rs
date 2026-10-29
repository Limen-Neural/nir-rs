use super::*;
use crate::nodes::{Conv1d, Conv2d, Flatten, Lif, SumPool2d};

fn input(shape: &[usize]) -> NirNode {
    NirNode::Input(Input {
        shape: shape.to_vec(),
        metadata: Default::default(),
    })
}

fn output(shape: &[usize]) -> NirNode {
    NirNode::Output(Output {
        shape: shape.to_vec(),
        metadata: Default::default(),
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
        metadata: Default::default(),
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
        metadata: Default::default(),
    })
}

fn pool(kernel: i64) -> NirNode {
    let t = |v: i64| Tensor::from_i64(vec![2], vec![v, v]).unwrap();
    NirNode::SumPool2d(SumPool2d {
        kernel_size: t(kernel),
        stride: t(kernel),
        padding: t(0),
        metadata: Default::default(),
    })
}

fn flatten(input_type: Option<Vec<usize>>) -> NirNode {
    NirNode::Flatten(Flatten {
        start_dim: 1,
        end_dim: -1,
        input_type,
        metadata: Default::default(),
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
fn conv_with_declared_input_shape_conflicts_with_producer() {
    let g = chain(vec![
        ("in", input(&[1, 9, 9])),
        ("conv", conv2d(2, 1, (3, 3), Some(vec![8, 8]))),
        ("out", output(&[2, 6, 6])),
    ]);
    assert!(matches!(type_error(&g), TypeError::EdgeMismatch { .. }));
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
        metadata: Default::default(),
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
fn root_without_defined_input_is_rejected() {
    let g = chain(vec![
        ("conv", conv2d(2, 1, (3, 3), None)),
        ("out", output(&[2, 6, 6])),
    ]);
    assert!(matches!(type_error(&g), TypeError::UndefinedInput { .. }));
}

#[test]
fn implicit_ports_are_reported_not_inserted() {
    // Root and leaf are plain LIFs with defined types.
    let g = chain(vec![("a", lif(&[3])), ("b", lif(&[3]))]);
    let inferred = g.infer_types().unwrap();
    assert_eq!(inferred.implicit_inputs.len(), 1);
    assert_eq!(inferred.implicit_inputs[0].name, "input_a");
    assert_eq!(inferred.implicit_inputs[0].node, "a");
    assert_eq!(inferred.implicit_outputs[0].name, "output_b");
    g.check_types().unwrap();
    assert_eq!(g.len(), 2);
}

#[test]
fn repair_is_idempotent_and_makes_check_pass() {
    let mut g = chain(vec![
        ("conv", conv2d(2, 1, (3, 3), Some(vec![8, 8]))),
        ("flat", flatten(None)),
    ]);
    g.infer_and_repair().unwrap();
    assert!(matches!(g.get("input_conv"), Some(NirNode::Input(i)) if i.shape == [1, 8, 8]));
    assert!(matches!(g.get("output_flat"), Some(NirNode::Output(o)) if o.shape == [2, 36]));
    let NirNode::Flatten(f) = g.get("flat").unwrap() else {
        unreachable!()
    };
    assert_eq!(f.input_type, Some(vec![2, 6, 6]));
    g.check_types().unwrap();
    let once = g.clone();
    g.infer_and_repair().unwrap();
    assert_eq!(g, once);
}

#[test]
fn repair_names_avoid_collisions() {
    let mut g = NirGraph::new();
    g.insert_node("input_a", input(&[3])).unwrap();
    g.insert_node("a", lif(&[3])).unwrap();
    g.insert_node("z", lif(&[3])).unwrap();
    g.insert_node("b", lif(&[3])).unwrap();
    g.add_edge("input_a", "z");
    g.add_edge("z", "b");
    // `a` is an isolated root *and* leaf: needs input_a_0 / output_a.
    g.infer_and_repair().unwrap();
    assert!(g.get("input_a_0").is_some());
    assert!(g.get("output_a").is_some());
}

#[test]
fn failed_repair_leaves_graph_untouched() {
    let mut g = chain(vec![
        ("in", input(&[1])),
        ("lif", lif(&[1])),
        ("out", output(&[2])),
    ]);
    let before = g.clone();
    assert!(g.infer_and_repair().is_err());
    assert_eq!(g, before);
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
fn affine_ports_follow_weight_axes() {
    let t = matrix_types(&Tensor::from_f32(vec![4, 3], vec![0.0; 12]).unwrap()).unwrap();
    assert_eq!(t, NodeTypes::leaf(Some(vec![3]), Some(vec![4])));
    assert!(matrix_types(&Tensor::from_f32(vec![3], vec![0.0; 3]).unwrap()).is_err());
}

#[test]
fn hostile_extents_do_not_panic() {
    let g = chain(vec![
        ("in", input(&[1, usize::MAX, 4])),
        (
            "conv",
            NirNode::Conv2d(Conv2d {
                weight: Tensor::from_f32(vec![1, 1, 1, 1], vec![0.0]).unwrap(),
                stride: vec![i64::MIN],
                padding: Padding::Explicit(vec![i64::MAX]),
                dilation: vec![i64::MAX],
                groups: 1,
                bias: Tensor::from_f32(vec![1], vec![0.0]).unwrap(),
                input_shape: None,
                metadata: Default::default(),
            }),
        ),
        ("out", output(&[1])),
    ]);
    assert!(g.check_types().is_err());
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

fn conv1d(weight: Tensor, padding: Padding, input_shape: Option<usize>) -> NirNode {
    NirNode::Conv1d(Conv1d {
        weight,
        stride: vec![1],
        padding,
        dilation: vec![1],
        groups: 1,
        bias: Tensor::from_f32(vec![1], vec![0.0]).unwrap(),
        input_shape,
        metadata: Default::default(),
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

#[test]
fn conv1d_is_inferred_and_repaired() {
    let mut g = chain(vec![
        ("in", input(&[2, 10])),
        ("conv", conv1d(zeros(&[3, 2, 3]), Padding::Valid, None)),
        ("out", output(&[3, 8])),
    ]);
    g.check_types().unwrap();
    g.infer_and_repair().unwrap();
    let NirNode::Conv1d(conv) = g.get("conv").unwrap() else {
        unreachable!()
    };
    assert_eq!(conv.input_shape, Some(10));
}

#[test]
fn nested_graph_is_repaired() {
    let mut g = chain(vec![
        ("in", input(&[1, 8, 8])),
        (
            "sub",
            NirNode::Graph(Box::new(chain(vec![
                ("sin", input(&[1, 8, 8])),
                ("conv", conv2d(2, 1, (3, 3), None)),
                ("sout", output(&[2, 6, 6])),
            ]))),
        ),
    ]);
    g.infer_and_repair().unwrap();
    let Some(NirNode::Graph(sub)) = g.get("sub") else {
        unreachable!()
    };
    let NirNode::Conv2d(conv) = sub.get("conv").unwrap() else {
        unreachable!()
    };
    assert_eq!(conv.input_shape, Some(vec![8, 8]));
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

fn two_input_graph() -> NirGraph {
    let mut sub = NirGraph::new();
    sub.insert_node("a", input(&[1])).unwrap();
    sub.insert_node("b", input(&[1])).unwrap();
    sub.insert_node("o", output(&[1])).unwrap();
    sub.add_edge("a", "o");
    sub.add_edge("b", "o");
    sub
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
