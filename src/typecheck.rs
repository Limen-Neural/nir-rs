// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in type/shape inference and checking, aligned with Python `NIRGraph`.
//!
//! Python `NIRGraph(type_check=True)` (the default) runs `infer_types()` and
//! `check_types()` from `nir/ir/graph.py`. This module reproduces that
//! algorithm, with one deliberate difference: **it never mutates the graph**.
//! Python prepends an `Input` node to every root and appends an `Output` node
//! to every leaf as a side effect of loading; here those implicit ports are
//! reported in [`InferredTypes`] and only materialized by the explicit
//! [`NirGraph::infer_and_repair`].
//!
//! Shapes are the per-node `input_type` / `output_type` of Python NIR:
//!
//! - `Input` / `Output`: their `shape`
//! - `Affine` / `Linear`: `weight.shape[:-2] + [weight.shape[-1]]` →
//!   `weight.shape[:-2] + [weight.shape[-2]]`
//! - `Scale`, `Delay`, `Threshold`: the parameter's shape
//! - `I`, `IF`, `LI`, `LIF`: the shape of `r`; `CubaLI`: of `v_leak`;
//!   `CubaLIF`: of `v_threshold`
//! - `Conv1d` / `Conv2d`: defined only when `input_shape` is set, otherwise
//!   inferred from the producer
//! - `SumPool2d` / `AvgPool2d` / `Flatten`: inferred from the producer
//!   (`Flatten` may carry `input_type`)
//! - `NIRGraph`: one port per `Input` / `Output` node, in node order
//!
//! Known difference: Python's `Conv2d` shape inference reads only
//! `weight.shape[2]` as the kernel size for **both** spatial axes. This crate
//! uses `weight.shape[2]` and `weight.shape[3]` per axis, so non-square
//! kernels are inferred correctly instead of reproducing that quirk.

use crate::error::{NirError, Result, TypeError};
use crate::graph::NirGraph;
use crate::nodes::{Input, NirNode, Output, Padding};
use crate::types::{Tensor, TensorData};
use indexmap::IndexMap;
use std::collections::HashSet;

/// Tensor shape of one port (axis lengths).
pub type Shape = Vec<usize>;

/// Input and output ports of one node. `None` is an undefined type.
///
/// Leaf nodes have exactly one input and one output port. A nested graph has
/// one port per `Input` / `Output` node it contains.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NodeTypes {
    /// Input port shapes.
    pub input: Vec<Option<Shape>>,
    /// Output port shapes.
    pub output: Vec<Option<Shape>>,
}

impl NodeTypes {
    fn leaf(input: Option<Shape>, output: Option<Shape>) -> Self {
        Self {
            input: vec![input],
            output: vec![output],
        }
    }

    fn undefined() -> Self {
        Self::leaf(None, None)
    }

    /// Python: `input_type is None or any(v is None ...)`. A graph without
    /// `Input` nodes has no input ports and counts as undefined.
    fn input_undefined(&self) -> bool {
        self.input.is_empty() || self.input.iter().any(Option::is_none)
    }

    /// Python: `any(v is None for v in output_type.values())`.
    fn output_undefined(&self) -> bool {
        self.output.iter().any(Option::is_none)
    }
}

/// An `Input` or `Output` node Python would insert but this crate does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplicitPort {
    /// Name Python would give the new node (`input_<root>` / `output_<leaf>`,
    /// with a `_<i>` suffix on collision).
    pub name: String,
    /// The existing root (for an input) or leaf (for an output) it attaches to.
    pub node: String,
    /// Shape of the new port.
    pub shape: Shape,
}

/// Result of [`NirGraph::infer_types`]. Read-only; the graph is unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InferredTypes {
    /// Resolved ports for every node of the graph, in node order.
    pub nodes: IndexMap<String, NodeTypes>,
    /// Resolved types for each nested [`NirNode::Graph`], keyed by node name.
    pub subgraphs: IndexMap<String, InferredTypes>,
    /// `Input` nodes Python would prepend to non-`Input` roots.
    pub implicit_inputs: Vec<ImplicitPort>,
    /// `Output` nodes Python would append to non-`Output` leaves.
    pub implicit_outputs: Vec<ImplicitPort>,
}

impl NirGraph {
    /// Maximum subgraph nesting that type inference recurses into.
    ///
    /// Inference recurses on the process stack, unlike
    /// [`validate_structure`](Self::validate_structure), so it uses a much
    /// smaller bound than [`MAX_NESTING_DEPTH`](Self::MAX_NESTING_DEPTH).
    /// Deeper graphs fail with [`TypeError::NestingTooDeep`].
    pub const MAX_TYPE_CHECK_DEPTH: usize = 64;

    /// Infer every node's input/output shape without mutating the graph.
    ///
    /// Mirrors Python `NIRGraph.infer_types()`: it runs
    /// [`validate_structure`](Self::validate_structure), requires every root
    /// to be an `Input` or have a defined input type, propagates shapes from
    /// the `Input` nodes along edges, derives `Conv1d`/`Conv2d`/`SumPool2d`/
    /// `AvgPool2d`/`Flatten` outputs, and fails on a producer/consumer
    /// mismatch it meets on the way. Roots and leaves that Python would wrap
    /// in a new `Input` / `Output` are listed in
    /// [`InferredTypes::implicit_inputs`] / [`implicit_outputs`](InferredTypes::implicit_outputs).
    ///
    /// Edges to nodes already reached are only compared by
    /// [`check_types`](Self::check_types).
    ///
    /// # Errors
    ///
    /// Structure errors from `validate_structure`, otherwise
    /// [`NirError::TypeCheck`].
    pub fn infer_types(&self) -> Result<InferredTypes> {
        self.validate_structure()?;
        infer_graph(self, "", 0)
    }

    /// Check that shapes agree along every edge, without mutating the graph.
    ///
    /// Equivalent to Python's default `type_check=True` acceptance test:
    /// [`infer_types`](Self::infer_types) followed by an every-edge shape
    /// comparison in the graph and all nested subgraphs. Not run by
    /// [`validate_structure`](Self::validate_structure) or by default
    /// [`crate::io::read`].
    ///
    /// ```
    /// use nir_rs::{NirGraph, NirNode, NirError};
    /// use nir_rs::nodes::{Input, Output};
    ///
    /// let mut g = NirGraph::new();
    /// g.insert_node("in", NirNode::Input(Input { shape: vec![3], metadata: Default::default() }))?;
    /// g.insert_node("out", NirNode::Output(Output { shape: vec![4], metadata: Default::default() }))?;
    /// g.add_edge("in", "out");
    /// assert!(g.validate_structure().is_ok());
    /// assert!(matches!(g.check_types(), Err(NirError::TypeCheck(_))));
    /// # Ok::<(), NirError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Structure errors from `validate_structure`, otherwise
    /// [`NirError::TypeCheck`].
    pub fn check_types(&self) -> Result<()> {
        let inferred = self.infer_types()?;
        check_graph(self, &inferred, "", 0)
    }

    /// Run the checks and then apply Python's graph repair in place.
    ///
    /// Python does this implicitly on construction; here it is explicit and
    /// atomic (the graph is untouched when an error is returned). On success:
    ///
    /// - an `Input` is prepended to every non-`Input` root and an `Output`
    ///   appended to every non-`Output` leaf (named as in Python)
    /// - `Conv1d` / `Conv2d` get `input_shape`, and `Flatten` gets
    ///   `input_type`, where those were `None`
    ///
    /// Nested subgraphs are repaired too. A repaired graph passes
    /// [`check_types`](Self::check_types), and repairing it again is a no-op.
    ///
    /// # Errors
    ///
    /// As [`check_types`](Self::check_types).
    pub fn infer_and_repair(&mut self) -> Result<()> {
        self.validate_structure()?;
        let inferred = infer_graph(self, "", 0)?;
        check_graph(self, &inferred, "", 0)?;
        apply_repair(self, &inferred);
        Ok(())
    }
}

fn qualify(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}/{name}")
    }
}

fn type_err(err: TypeError) -> NirError {
    NirError::TypeCheck(Box::new(err))
}

fn unique_name(base: &str, taken: &mut HashSet<String>) -> String {
    let mut name = base.to_owned();
    let mut i = 0usize;
    while taken.contains(&name) {
        name = format!("{base}_{i}");
        i += 1;
    }
    taken.insert(name.clone());
    name
}

fn infer_graph(graph: &NirGraph, prefix: &str, depth: usize) -> Result<InferredTypes> {
    if depth > NirGraph::MAX_TYPE_CHECK_DEPTH {
        return Err(type_err(TypeError::NestingTooDeep {
            limit: NirGraph::MAX_TYPE_CHECK_DEPTH,
        }));
    }
    let mut out = InferredTypes::default();
    if graph.nodes.is_empty() {
        return Ok(out);
    }

    // Declared types (what each node knows on its own).
    let mut types: IndexMap<String, NodeTypes> = IndexMap::with_capacity(graph.nodes.len());
    for (name, node) in &graph.nodes {
        let declared = if let NirNode::Graph(sub) = node {
            let inner = infer_graph(sub, &qualify(prefix, name), depth + 1)?;
            let ports = graph_ports(sub, &inner);
            out.subgraphs.insert(name.clone(), inner);
            ports
        } else {
            declared_types(node).map_err(|reason| {
                type_err(TypeError::Inference {
                    node: qualify(prefix, name),
                    node_type: node.type_name(),
                    reason,
                })
            })?
        };
        types.insert(name.clone(), declared);
    }

    // Working edge list: real edges plus the implicit `Input -> root` ones.
    let mut edges: Vec<(String, String)> = graph.edges.clone();
    let mut taken: HashSet<String> = graph.nodes.keys().cloned().collect();
    let mut input_names: HashSet<String> = graph
        .nodes
        .iter()
        .filter(|(_, n)| matches!(n, NirNode::Input(_)))
        .map(|(k, _)| k.clone())
        .collect();
    let dests: HashSet<&str> = graph.edges.iter().map(|(_, d)| d.as_str()).collect();
    let srcs: HashSet<&str> = graph.edges.iter().map(|(s, _)| s.as_str()).collect();

    // Every root must be an Input, or have a type to build one from.
    for (name, node) in &graph.nodes {
        if dests.contains(name.as_str()) || matches!(node, NirNode::Input(_)) {
            continue;
        }
        let declared = &types[name];
        if declared.input_undefined() {
            return Err(type_err(TypeError::UndefinedInput {
                node: qualify(prefix, name),
                node_type: node.type_name(),
            }));
        }
        let shape = single_port(&declared.input, &qualify(prefix, name), name)?;
        let new_name = unique_name(&format!("input_{name}"), &mut taken);
        types.insert(
            new_name.clone(),
            NodeTypes::leaf(Some(shape.clone()), Some(shape.clone())),
        );
        input_names.insert(new_name.clone());
        edges.push((new_name.clone(), name.clone()));
        out.implicit_inputs.push(ImplicitPort {
            name: new_name,
            node: name.clone(),
            shape,
        });
    }

    let mut ready: Vec<(String, String)> = edges
        .iter()
        .filter(|(s, _)| input_names.contains(s))
        .cloned()
        .collect();
    if ready.is_empty() {
        return Err(type_err(TypeError::NoInputs {
            graph: prefix.to_owned(),
        }));
    }
    let mut seen: HashSet<String> = ready.iter().map(|(s, _)| s.clone()).collect();

    while let Some((pre, post)) = ready.pop() {
        let pre_output = types[&pre].output.clone();
        let post_node = &graph.nodes[&post];
        let post_types = types.get_mut(&post).expect("edge endpoints validated");

        if post_types.input_undefined() {
            post_types.input = pre_output;
        } else if post_types.input != pre_output {
            return Err(type_err(TypeError::EdgeMismatch {
                src: qualify(prefix, &pre),
                src_type: graph.nodes.get(&pre).map_or("Input", NirNode::type_name),
                dst: qualify(prefix, &post),
                dst_type: post_node.type_name(),
                output: pre_output,
                input: post_types.input.clone(),
            }));
        }

        if matches!(post_node, NirNode::Output(_)) {
            post_types.output = post_types.input.clone();
        }

        if post_types.output_undefined() {
            let inferred = infer_output(post_node, &post_types.input).map_err(|reason| {
                type_err(TypeError::Inference {
                    node: qualify(prefix, &post),
                    node_type: post_node.type_name(),
                    reason,
                })
            })?;
            if let Some(shape) = inferred {
                post_types.output = vec![Some(shape)];
            }
        }

        seen.insert(post.clone());
        ready.extend(
            edges
                .iter()
                .filter(|(s, d)| *s == post && !seen.contains(d))
                .cloned(),
        );
    }

    // Every leaf must be an Output, or have a type to build one from.
    let leaves: Vec<&String> = graph
        .nodes
        .keys()
        .filter(|k| !srcs.contains(k.as_str()))
        .collect();
    if leaves.is_empty() {
        return Err(type_err(TypeError::NoOutputs {
            graph: prefix.to_owned(),
        }));
    }
    for name in leaves {
        let node = &graph.nodes[name];
        if matches!(node, NirNode::Output(_)) {
            continue;
        }
        let declared = &types[name];
        if declared.output_undefined() {
            return Err(type_err(TypeError::UndefinedOutput {
                node: qualify(prefix, name),
                node_type: node.type_name(),
            }));
        }
        let shape = single_port(&declared.output, &qualify(prefix, name), name)?;
        let new_name = unique_name(&format!("output_{name}"), &mut taken);
        out.implicit_outputs.push(ImplicitPort {
            name: new_name,
            node: name.clone(),
            shape,
        });
    }

    for name in graph.nodes.keys() {
        out.nodes.insert(name.clone(), types[name].clone());
    }
    Ok(out)
}

/// The sole shape of a port list, or the unsupported-multi-port error.
fn single_port(ports: &[Option<Shape>], qualified: &str, name: &str) -> Result<Shape> {
    match ports {
        [Some(shape)] => Ok(shape.clone()),
        _ => Err(type_err(TypeError::MultiPort {
            src: qualified.to_owned(),
            dst: name.to_owned(),
        })),
    }
}

/// Ports of a nested graph after it has been inferred: its `Input` /
/// `Output` nodes in node order, then the implicit ones.
fn graph_ports(sub: &NirGraph, inner: &InferredTypes) -> NodeTypes {
    let mut input = Vec::new();
    let mut output = Vec::new();
    for node in sub.nodes.values() {
        match node {
            NirNode::Input(Input { shape, .. }) => input.push(Some(shape.clone())),
            NirNode::Output(Output { shape, .. }) => output.push(Some(shape.clone())),
            _ => {}
        }
    }
    input.extend(inner.implicit_inputs.iter().map(|p| Some(p.shape.clone())));
    output.extend(inner.implicit_outputs.iter().map(|p| Some(p.shape.clone())));
    NodeTypes { input, output }
}

fn check_graph(
    graph: &NirGraph,
    inferred: &InferredTypes,
    prefix: &str,
    depth: usize,
) -> Result<()> {
    if depth > NirGraph::MAX_TYPE_CHECK_DEPTH {
        return Err(type_err(TypeError::NestingTooDeep {
            limit: NirGraph::MAX_TYPE_CHECK_DEPTH,
        }));
    }
    for (name, node) in &graph.nodes {
        if let NirNode::Graph(sub) = node {
            check_graph(
                sub,
                &inferred.subgraphs[name],
                &qualify(prefix, name),
                depth + 1,
            )?;
        }
    }
    for (src, dst) in &graph.edges {
        let (pre, post) = (&inferred.nodes[src], &inferred.nodes[dst]);
        let (pre_node, post_node) = (&graph.nodes[src], &graph.nodes[dst]);
        if pre.output_undefined() {
            return Err(type_err(TypeError::UndefinedOutput {
                node: qualify(prefix, src),
                node_type: pre_node.type_name(),
            }));
        }
        if post.input_undefined() {
            return Err(type_err(TypeError::UndefinedInput {
                node: qualify(prefix, dst),
                node_type: post_node.type_name(),
            }));
        }
        if pre.output.len() != post.input.len() {
            return Err(type_err(TypeError::PortCountMismatch {
                src: qualify(prefix, src),
                dst: qualify(prefix, dst),
                outputs: pre.output.len(),
                inputs: post.input.len(),
            }));
        }
        if pre.output.len() != 1 {
            return Err(type_err(TypeError::MultiPort {
                src: qualify(prefix, src),
                dst: qualify(prefix, dst),
            }));
        }
        if pre.output != post.input {
            return Err(type_err(TypeError::EdgeMismatch {
                src: qualify(prefix, src),
                src_type: pre_node.type_name(),
                dst: qualify(prefix, dst),
                dst_type: post_node.type_name(),
                output: pre.output.clone(),
                input: post.input.clone(),
            }));
        }
    }
    Ok(())
}

fn apply_repair(graph: &mut NirGraph, inferred: &InferredTypes) {
    if graph.nodes.is_empty() {
        return;
    }
    for (name, node) in &mut graph.nodes {
        let input = inferred.nodes[name].input.first().cloned().flatten();
        match node {
            NirNode::Graph(sub) => apply_repair(sub, &inferred.subgraphs[name]),
            NirNode::Conv1d(conv) if conv.input_shape.is_none() => {
                conv.input_shape = input.and_then(|s| s.get(1).copied());
            }
            NirNode::Conv2d(conv) if conv.input_shape.is_none() => {
                conv.input_shape = input.map(|s| s[1..].to_vec());
            }
            NirNode::Flatten(flat) if flat.input_type.is_none() => flat.input_type = input,
            _ => {}
        }
    }
    for port in &inferred.implicit_inputs {
        graph.nodes.insert(
            port.name.clone(),
            NirNode::Input(Input {
                shape: port.shape.clone(),
                metadata: Default::default(),
            }),
        );
        graph.edges.push((port.name.clone(), port.node.clone()));
    }
    for port in &inferred.implicit_outputs {
        graph.nodes.insert(
            port.name.clone(),
            NirNode::Output(Output {
                shape: port.shape.clone(),
                metadata: Default::default(),
            }),
        );
        graph.edges.push((port.node.clone(), port.name.clone()));
    }
}

/// Types a node knows from its own fields (Python `__post_init__`).
fn declared_types(node: &NirNode) -> std::result::Result<NodeTypes, String> {
    let same = |shape: &[usize]| NodeTypes::leaf(Some(shape.to_vec()), Some(shape.to_vec()));
    Ok(match node {
        NirNode::Input(n) => same(&n.shape),
        NirNode::Output(n) => same(&n.shape),
        NirNode::Affine(n) => matrix_types(&n.weight)?,
        NirNode::Linear(n) => matrix_types(&n.weight)?,
        NirNode::Scale(n) => same(n.scale.shape()),
        NirNode::Delay(n) => same(n.delay.shape()),
        NirNode::Threshold(n) => same(n.threshold.shape()),
        NirNode::CubaLi(n) => same(n.v_leak.shape()),
        NirNode::CubaLif(n) => same(n.v_threshold.shape()),
        NirNode::I(n) => same(n.r.shape()),
        NirNode::If(n) => same(n.r.shape()),
        NirNode::Li(n) => same(n.r.shape()),
        NirNode::Lif(n) => same(n.r.shape()),
        NirNode::Conv1d(n) => match n.input_shape {
            None => NodeTypes::undefined(),
            Some(len) => conv_types(&n.weight, 3, &[len], &n.padding, &n.dilation, &n.stride)?,
        },
        NirNode::Conv2d(n) => match &n.input_shape {
            None => NodeTypes::undefined(),
            Some(spatial) => conv_types(&n.weight, 4, spatial, &n.padding, &n.dilation, &n.stride)?,
        },
        NirNode::SumPool2d(_) | NirNode::AvgPool2d(_) => NodeTypes::undefined(),
        NirNode::Flatten(n) => match &n.input_type {
            None => NodeTypes::undefined(),
            Some(shape) => NodeTypes::leaf(
                Some(shape.clone()),
                Some(flatten_output(shape, n.start_dim, n.end_dim)?),
            ),
        },
        // Handled by the caller (needs the recursive result).
        NirNode::Graph(_) => NodeTypes::undefined(),
    })
}

/// Output shape for nodes whose output follows from their (now known) input.
/// `Ok(None)` for nodes that cannot be inferred this way.
fn infer_output(
    node: &NirNode,
    input: &[Option<Shape>],
) -> std::result::Result<Option<Shape>, String> {
    let known = || match input {
        [Some(shape)] => Ok(shape.clone()),
        _ => Err("input type is undefined".to_owned()),
    };
    match node {
        NirNode::Conv1d(n) => {
            conv_output(&n.weight, 3, &n.padding, &n.dilation, &n.stride, &known()?).map(Some)
        }
        NirNode::Conv2d(n) => {
            conv_output(&n.weight, 4, &n.padding, &n.dilation, &n.stride, &known()?).map(Some)
        }
        NirNode::SumPool2d(n) => {
            pool_output(&known()?, &n.kernel_size, &n.stride, &n.padding).map(Some)
        }
        NirNode::AvgPool2d(n) => {
            pool_output(&known()?, &n.kernel_size, &n.stride, &n.padding).map(Some)
        }
        NirNode::Flatten(n) => flatten_output(&known()?, n.start_dim, n.end_dim).map(Some),
        _ => Ok(None),
    }
}

fn matrix_types(weight: &Tensor) -> std::result::Result<NodeTypes, String> {
    let shape = weight.shape();
    if shape.len() < 2 {
        return Err(format!(
            "weight must be at least 2-D, found rank {}",
            shape.len()
        ));
    }
    let batch = &shape[..shape.len() - 2];
    let with = |last: usize| {
        let mut s = batch.to_vec();
        s.push(last);
        s
    };
    Ok(NodeTypes::leaf(
        Some(with(shape[shape.len() - 1])),
        Some(with(shape[shape.len() - 2])),
    ))
}

fn conv_types(
    weight: &Tensor,
    rank: usize,
    spatial: &[usize],
    padding: &Padding,
    dilation: &[i64],
    stride: &[i64],
) -> std::result::Result<NodeTypes, String> {
    let channels = *weight
        .shape()
        .get(1)
        .ok_or_else(|| format!("weight rank {} is not {rank}", weight.shape().len()))?;
    let mut input = vec![channels];
    input.extend_from_slice(spatial);
    let output = conv_output(weight, rank, padding, dilation, stride, &input)?;
    Ok(NodeTypes::leaf(Some(input), Some(output)))
}

/// `[C_out, *spatial_out]` for a convolution whose input is
/// `[C_in, *spatial]`.
fn conv_output(
    weight: &Tensor,
    rank: usize,
    padding: &Padding,
    dilation: &[i64],
    stride: &[i64],
    input: &[usize],
) -> std::result::Result<Shape, String> {
    if weight.shape().len() != rank {
        return Err(format!(
            "weight rank {} is not {rank}",
            weight.shape().len()
        ));
    }
    if input.len() != rank - 1 {
        return Err(format!(
            "input rank {} is not {} (channels plus {} spatial axes)",
            input.len(),
            rank - 1,
            rank - 2
        ));
    }
    let pad = match padding {
        Padding::Explicit(p) => Pad::Explicit(p),
        Padding::Same => Pad::Same,
        Padding::Valid => Pad::Valid,
    };
    let mut out = vec![weight.shape()[0]];
    out.extend(spatial_output(
        &input[1..],
        pad,
        dilation,
        &weight.shape()[2..],
        stride,
    )?);
    Ok(out)
}

#[derive(Clone, Copy)]
enum Pad<'a> {
    Explicit(&'a [i64]),
    Same,
    Valid,
}

/// One value per spatial axis; a single value broadcasts (Python int).
fn axis_value(
    values: &[i64],
    axis: usize,
    ndim: usize,
    field: &str,
) -> std::result::Result<i64, String> {
    match values.len() {
        1 => Ok(values[0]),
        n if n == ndim => Ok(values[axis]),
        n => Err(format!("{field} has {n} extents for {ndim} spatial axes")),
    }
}

/// `floor((in + 2p - d(k-1) - 1) / s + 1)` per axis (PyTorch / NIR formula).
fn spatial_output(
    input: &[usize],
    padding: Pad<'_>,
    dilation: &[i64],
    kernel: &[usize],
    stride: &[i64],
) -> std::result::Result<Shape, String> {
    let ndim = input.len();
    let mut out = Vec::with_capacity(ndim);
    for (axis, &n) in input.iter().enumerate() {
        let extent = match padding {
            Pad::Same => i128::from(n as u64),
            Pad::Valid | Pad::Explicit(_) => {
                let p = match padding {
                    Pad::Explicit(p) => axis_value(p, axis, ndim, "padding")?,
                    _ => 0,
                };
                let d = axis_value(dilation, axis, ndim, "dilation")?;
                let s = axis_value(stride, axis, ndim, "stride")?;
                if s <= 0 {
                    return Err(format!("stride must be positive, found {s}"));
                }
                let k = match kernel.len() {
                    1 => kernel[0],
                    len if len == ndim => kernel[axis],
                    len => return Err(format!("kernel has {len} extents for {ndim} spatial axes")),
                };
                let numerator = i128::from(n as u64) + 2 * i128::from(p)
                    - i128::from(d) * (i128::from(k as u64) - 1)
                    - 1;
                numerator.div_euclid(i128::from(s)) + 1
            }
        };
        let extent = usize::try_from(extent).map_err(|_| {
            format!("output extent {extent} on spatial axis {axis} is out of range")
        })?;
        out.push(extent);
    }
    Ok(out)
}

fn pool_extents<'a>(tensor: &'a Tensor, field: &str) -> std::result::Result<&'a [i64], String> {
    match tensor.data() {
        TensorData::I64(values) if !values.is_empty() => Ok(values),
        TensorData::I64(_) => Err(format!("{field} is empty")),
        _ => Err(format!("{field} must be an integer tensor")),
    }
}

fn pool_output(
    input: &[usize],
    kernel: &Tensor,
    stride: &Tensor,
    padding: &Tensor,
) -> std::result::Result<Shape, String> {
    if input.len() != 3 {
        return Err(format!(
            "input rank {} is not 3 (channels, height, width)",
            input.len()
        ));
    }
    let kernel = pool_extents(kernel, "kernel_size")?;
    if kernel.iter().any(|&k| k < 1) {
        return Err("kernel_size must be positive".to_owned());
    }
    let kernel: Vec<usize> = kernel.iter().map(|&k| k as usize).collect();
    let stride = pool_extents(stride, "stride")?;
    let padding = pool_extents(padding, "padding")?;
    let mut out = vec![input[0]];
    out.extend(spatial_output(
        &input[1..],
        Pad::Explicit(padding),
        &[1],
        &kernel,
        stride,
    )?);
    Ok(out)
}

/// Python slice `values[start:stop]` on a list of length `len`, as bounds.
fn py_slice(len: usize, start: Option<i64>, stop: Option<i64>) -> (usize, usize) {
    let clamp = |i: i64| -> usize {
        let len = len as i64;
        let i = if i < 0 { i + len } else { i };
        i.clamp(0, len) as usize
    };
    let start = start.map_or(0, clamp);
    let stop = stop.map_or(len, clamp);
    (start, stop.max(start))
}

/// Python `calc_flatten_output` plus its element-count assertion.
fn flatten_output(
    input: &[usize],
    start_dim: i64,
    end_dim: i64,
) -> std::result::Result<Shape, String> {
    let len = input.len();
    let mut out: Shape = Vec::new();
    if start_dim != 0 {
        let (a, b) = py_slice(len, None, Some(start_dim));
        out.extend_from_slice(&input[a..b]);
    }
    let (a, b) = if end_dim != -1 {
        py_slice(len, Some(start_dim), Some(end_dim.saturating_add(1)))
    } else {
        py_slice(len, Some(start_dim), None)
    };
    let middle = input[a..b]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .ok_or("flattened extent overflows usize")?;
    out.push(middle);
    if end_dim != -1 && i64::try_from(len).map_or(true, |l| end_dim != l - 1) {
        let (a, b) = py_slice(len, Some(end_dim.saturating_add(1)), None);
        out.extend_from_slice(&input[a..b]);
    }
    let count = |s: &[usize]| s.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x));
    if count(input) != count(&out) {
        return Err(format!(
            "Flatten must not change the number of elements: {input:?} -> {out:?}"
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
