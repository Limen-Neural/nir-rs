// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

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
