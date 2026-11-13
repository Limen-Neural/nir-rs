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
//!
//! Inherited as-is: with `Padding::Same`, upstream `calculate_conv_output`
//! returns the input extent and ignores `stride`, so a strided `same`
//! convolution keeps its spatial size here too (PyTorch would reject it).

mod shape;

use crate::error::{NirError, Result, TypeError};
use crate::graph::NirGraph;
use crate::nodes::{Input, NirNode, Output};
use crate::types::MetadataMap;
use indexmap::IndexMap;
use shape::{declared_types, infer_output};
use std::collections::{HashMap, HashSet};

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
    /// Maximum number of graph levels type inference recurses through, root
    /// included.
    ///
    /// Inference recurses on the process stack, unlike
    /// [`validate_structure`](Self::validate_structure), so it uses a much
    /// smaller bound than [`MAX_NESTING_DEPTH`](Self::MAX_NESTING_DEPTH).
    /// Subgraphs nested `MAX_TYPE_CHECK_DEPTH` levels deep or deeper fail
    /// with [`TypeError::NestingTooDeep`]; one level less still passes.
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
    /// use nir_rs::types::MetadataMap;
    ///
    /// let mut g = NirGraph::new();
    /// g.insert_node("in", NirNode::Input(Input { shape: vec![3], metadata: MetadataMap::default() }))?;
    /// g.insert_node("out", NirNode::Output(Output { shape: vec![4], metadata: MetadataMap::default() }))?;
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
    /// - an `Input` is added for every non-`Input` root and an `Output` for
    ///   every non-`Output` leaf, appended in node order and named as in Python
    /// - `Conv1d` / `Conv2d` get `input_shape`, and `Flatten` gets
    ///   `input_type`, where those were `None`
    ///
    /// Nested subgraphs are repaired too, and repairing again is a no-op.
    /// A repaired graph passes [`check_types`](Self::check_types) again, with
    /// one Python-inherited exception: while a `Conv*` `input_shape` is unset,
    /// channel counts are never compared (Python propagates shapes wholesale),
    /// so a producer whose channels differ from the weight passes the first
    /// check. Once repair fills `input_shape`, the convolution declares the
    /// weight's channel count and the next check flags the mismatch, exactly
    /// as re-reading the repaired graph in Python would.
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

/// A graph under inference, with its path for error messages.
struct Scope<'a> {
    graph: &'a NirGraph,
    prefix: &'a str,
}

/// Edge list and name bookkeeping, extended with the implicit ports Python adds.
struct Wiring {
    edges: Vec<(String, String)>,
    taken: HashSet<String>,
    input_names: HashSet<String>,
}

impl Wiring {
    fn new(graph: &NirGraph) -> Self {
        Self {
            edges: graph.edges.clone(),
            taken: graph.nodes.keys().cloned().collect(),
            input_names: graph
                .nodes
                .iter()
                .filter(|(_, n)| matches!(n, NirNode::Input(_)))
                .map(|(k, _)| k.clone())
                .collect(),
        }
    }
}

fn infer_graph(graph: &NirGraph, prefix: &str, depth: usize) -> Result<InferredTypes> {
    if depth >= NirGraph::MAX_TYPE_CHECK_DEPTH {
        return Err(type_err(TypeError::NestingTooDeep {
            limit: NirGraph::MAX_TYPE_CHECK_DEPTH,
        }));
    }
    let mut out = InferredTypes::default();
    if graph.nodes.is_empty() {
        return Ok(out);
    }
    let scope = Scope { graph, prefix };
    let mut types = scope.declared(depth, &mut out)?;
    let mut wiring = Wiring::new(graph);
    out.implicit_inputs = scope.implicit_inputs(&mut types, &mut wiring)?;
    scope.propagate(&mut types, &wiring)?;
    out.implicit_outputs = scope.implicit_outputs(&types, &mut wiring.taken)?;
    out.nodes = graph
        .nodes
        .keys()
        .map(|name| (name.clone(), types[name].clone()))
        .collect();
    Ok(out)
}

impl Scope<'_> {
    fn qualified(&self, name: &str) -> String {
        qualify(self.prefix, name)
    }

    fn inference_error(&self, name: &str, node: &NirNode, reason: String) -> NirError {
        type_err(TypeError::Inference {
            node: self.qualified(name),
            node_type: node.type_name(),
            reason,
        })
    }

    /// Types each node knows on its own; nested graphs are inferred here.
    fn declared(
        &self,
        depth: usize,
        out: &mut InferredTypes,
    ) -> Result<IndexMap<String, NodeTypes>> {
        let mut types = IndexMap::with_capacity(self.graph.nodes.len());
        for (name, node) in &self.graph.nodes {
            let declared = if let NirNode::Graph(sub) = node {
                let inner = infer_graph(sub, &self.qualified(name), depth + 1)?;
                let ports = graph_ports(sub, &inner);
                out.subgraphs.insert(name.clone(), inner);
                ports
            } else {
                declared_types(node).map_err(|reason| self.inference_error(name, node, reason))?
            };
            types.insert(name.clone(), declared);
        }
        Ok(types)
    }

    /// Every root must be an `Input`, or have a type to build one from.
    fn implicit_inputs(
        &self,
        types: &mut IndexMap<String, NodeTypes>,
        wiring: &mut Wiring,
    ) -> Result<Vec<ImplicitPort>> {
        let dests: HashSet<&str> = self.graph.edges.iter().map(|(_, d)| d.as_str()).collect();
        let mut ports = Vec::new();
        for (name, node) in &self.graph.nodes {
            if dests.contains(name.as_str()) || matches!(node, NirNode::Input(_)) {
                continue;
            }
            let declared = &types[name];
            if declared.input_undefined() {
                return Err(type_err(TypeError::UndefinedInput {
                    node: self.qualified(name),
                    node_type: node.type_name(),
                }));
            }
            let shape = single_port(&declared.input, &self.qualified(name), name)?;
            let new_name = unique_name(&format!("input_{name}"), &mut wiring.taken);
            types.insert(
                new_name.clone(),
                NodeTypes::leaf(Some(shape.clone()), Some(shape.clone())),
            );
            wiring.input_names.insert(new_name.clone());
            wiring.edges.push((new_name.clone(), name.clone()));
            ports.push(ImplicitPort {
                name: new_name,
                node: name.clone(),
                shape,
            });
        }
        Ok(ports)
    }

    /// Walk the edges from the `Input` nodes, filling in undefined types.
    ///
    /// Outgoing edges are indexed up front, in edge order, so each edge is
    /// queued at most once and propagation stays linear in the edge count.
    fn propagate(&self, types: &mut IndexMap<String, NodeTypes>, wiring: &Wiring) -> Result<()> {
        let mut outgoing: HashMap<&str, Vec<&str>> = HashMap::with_capacity(wiring.edges.len());
        for (src, dst) in &wiring.edges {
            outgoing.entry(src.as_str()).or_default().push(dst.as_str());
        }
        let mut ready: Vec<(&str, &str)> = wiring
            .edges
            .iter()
            .filter(|(s, _)| wiring.input_names.contains(s))
            .map(|(s, d)| (s.as_str(), d.as_str()))
            .collect();
        if ready.is_empty() {
            return Err(type_err(TypeError::NoInputs {
                graph: self.prefix.to_owned(),
            }));
        }
        let mut seen: HashSet<&str> = ready.iter().map(|(s, _)| *s).collect();
        while let Some((pre, post)) = ready.pop() {
            self.propagate_edge(types, pre, post)?;
            // Expand only on the first visit to `post`: a repeat visit (a
            // node with several incoming edges) would only re-queue edges
            // the first expansion already queued.
            if seen.insert(post)
                && let Some(dsts) = outgoing.get(post)
            {
                ready.extend(
                    dsts.iter()
                        .filter(|dst| !seen.contains(*dst))
                        .map(|dst| (post, *dst)),
                );
            }
        }
        Ok(())
    }

    fn propagate_edge(
        &self,
        types: &mut IndexMap<String, NodeTypes>,
        pre: &str,
        post: &str,
    ) -> Result<()> {
        let pre_output = types[pre].output.clone();
        let post_node = &self.graph.nodes[post];
        let post_types = types.get_mut(post).expect("edge endpoints validated");

        if post_types.input_undefined() {
            post_types.input = pre_output;
        } else if post_types.input != pre_output {
            return Err(type_err(TypeError::EdgeMismatch {
                src: self.qualified(pre),
                src_type: self
                    .graph
                    .nodes
                    .get(pre)
                    .map_or("Input", NirNode::type_name),
                dst: self.qualified(post),
                dst_type: post_node.type_name(),
                output: pre_output,
                input: post_types.input.clone(),
            }));
        }

        if matches!(post_node, NirNode::Output(_)) {
            post_types.output.clone_from(&post_types.input);
        }
        if post_types.output_undefined() {
            let inferred = infer_output(post_node, &post_types.input)
                .map_err(|reason| self.inference_error(post, post_node, reason))?;
            if let Some(shape) = inferred {
                post_types.output = vec![Some(shape)];
            }
        }
        Ok(())
    }

    /// Every leaf must be an `Output`, or have a type to build one from.
    fn implicit_outputs(
        &self,
        types: &IndexMap<String, NodeTypes>,
        taken: &mut HashSet<String>,
    ) -> Result<Vec<ImplicitPort>> {
        let srcs: HashSet<&str> = self.graph.edges.iter().map(|(s, _)| s.as_str()).collect();
        let leaves: Vec<&String> = self
            .graph
            .nodes
            .keys()
            .filter(|k| !srcs.contains(k.as_str()))
            .collect();
        if leaves.is_empty() {
            return Err(type_err(TypeError::NoOutputs {
                graph: self.prefix.to_owned(),
            }));
        }
        let mut ports = Vec::new();
        for name in leaves {
            let node = &self.graph.nodes[name];
            if matches!(node, NirNode::Output(_)) {
                continue;
            }
            let declared = &types[name];
            if declared.output_undefined() {
                return Err(type_err(TypeError::UndefinedOutput {
                    node: self.qualified(name),
                    node_type: node.type_name(),
                }));
            }
            let shape = single_port(&declared.output, &self.qualified(name), name)?;
            ports.push(ImplicitPort {
                name: unique_name(&format!("output_{name}"), taken),
                node: name.clone(),
                shape,
            });
        }
        Ok(ports)
    }

    /// Both ends of one edge must have a single, equal port.
    fn check_edge(&self, inferred: &InferredTypes, src: &str, dst: &str) -> Result<()> {
        let (pre, post) = (&inferred.nodes[src], &inferred.nodes[dst]);
        let (pre_node, post_node) = (&self.graph.nodes[src], &self.graph.nodes[dst]);
        if pre.output_undefined() {
            return Err(type_err(TypeError::UndefinedOutput {
                node: self.qualified(src),
                node_type: pre_node.type_name(),
            }));
        }
        if post.input_undefined() {
            return Err(type_err(TypeError::UndefinedInput {
                node: self.qualified(dst),
                node_type: post_node.type_name(),
            }));
        }
        if pre.output.len() != post.input.len() {
            return Err(type_err(TypeError::PortCountMismatch {
                src: self.qualified(src),
                dst: self.qualified(dst),
                outputs: pre.output.len(),
                inputs: post.input.len(),
            }));
        }
        if pre.output.len() != 1 {
            return Err(type_err(TypeError::MultiPort {
                src: self.qualified(src),
                dst: self.qualified(dst),
            }));
        }
        if pre.output != post.input {
            return Err(type_err(TypeError::EdgeMismatch {
                src: self.qualified(src),
                src_type: pre_node.type_name(),
                dst: self.qualified(dst),
                dst_type: post_node.type_name(),
                output: pre.output.clone(),
                input: post.input.clone(),
            }));
        }
        Ok(())
    }
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
    if depth >= NirGraph::MAX_TYPE_CHECK_DEPTH {
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
    let scope = Scope { graph, prefix };
    graph
        .edges
        .iter()
        .try_for_each(|(src, dst)| scope.check_edge(inferred, src, dst))
}

fn apply_repair(graph: &mut NirGraph, inferred: &InferredTypes) {
    if graph.nodes.is_empty() {
        return;
    }
    for (name, node) in &mut graph.nodes {
        if let NirNode::Graph(sub) = node {
            apply_repair(sub, &inferred.subgraphs[name]);
        } else {
            repair_node(node, inferred.nodes[name].input.first().cloned().flatten());
        }
    }
    insert_implicit_ports(graph, inferred);
}

/// Fill in the optional input type a node was missing.
fn repair_node(node: &mut NirNode, input: Option<Shape>) {
    match node {
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

fn insert_implicit_ports(graph: &mut NirGraph, inferred: &InferredTypes) {
    for port in &inferred.implicit_inputs {
        graph.nodes.insert(
            port.name.clone(),
            NirNode::Input(Input {
                shape: port.shape.clone(),
                metadata: MetadataMap::default(),
            }),
        );
        graph.edges.push((port.name.clone(), port.node.clone()));
    }
    for port in &inferred.implicit_outputs {
        graph.nodes.insert(
            port.name.clone(),
            NirNode::Output(Output {
                shape: port.shape.clone(),
                metadata: MetadataMap::default(),
            }),
        );
        graph.edges.push((port.node.clone(), port.name.clone()));
    }
}

#[cfg(test)]
mod tests;
