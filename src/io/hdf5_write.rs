// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encoding of [`NirGraph`] into HDF5 `.nir` files.
//!
//! Mirrors upstream `nir/serialization.py` (`write` / `write_recursive`) and
//! the `to_dict` of each node dataclass, so the result loads in Python
//! `nir.read`. Byte-identical output versus h5py is not a goal: group ordering,
//! chunk layout and filter parameters are HDF5 implementation details.
//!
//! Two asymmetries are wire requirements rather than oversights:
//!
//! - `Conv1d` writes `stride` / `padding` / `dilation` / `input_shape` as
//!   **scalars** while `Conv2d` writes **length-2 arrays** — upstream keeps the
//!   1-D versions as plain `int` and promotes only the 2-D ones to tuples.
//! - Fields that are `None` are **omitted** rather than written as null. That
//!   is what the Python reader expects, and `create_dataset(k, data=None)`
//!   would in fact raise on the writing side.

use super::wire::{self, KEY_EDGES, KEY_METADATA, KEY_NODE, KEY_NODES, KEY_TYPE, KEY_VERSION};
use super::{DEFAULT_NIR_VERSION, WriteOptions};
use crate::error::{NirError, Result};
use crate::graph::NirGraph;
use crate::nodes::{
    Affine, AvgPool2d, Conv1d, Conv2d, CubaLi, CubaLif, Delay, Flatten, I, If, Input, Li, Lif,
    Linear, NirNode, Output, Padding, Scale, SumPool2d, Threshold,
};
use crate::types::{MetadataMap, MetadataValue, Tensor, TensorData};
use hdf5::H5Type;
use hdf5::types::VarLenUnicode;
use hdf5::{File, Group};
use std::path::Path;
use std::str::FromStr;

/// Write `graph` to `path`, truncating any existing file.
pub(super) fn write(path: &Path, graph: &NirGraph, opts: &WriteOptions) -> Result<()> {
    // Validate before touching the filesystem so a rejected graph never leaves
    // a half-written file behind. Name legality is not optional — HDF5 cannot
    // represent the rejected names at all.
    if opts.validate {
        graph.validate_structure()?;
    }
    check_names(graph)?;

    let version = opts
        .version
        .clone()
        .or_else(|| graph.version.clone())
        .unwrap_or_else(|| DEFAULT_NIR_VERSION.to_owned());

    let file = File::create(path)
        .map_err(|e| NirError::Io(format!("cannot create {}: {e}", path.display())))?;
    write_string(&file, KEY_VERSION, &version)?;

    let root = file.create_group(KEY_NODE)?;
    write_string(&root, KEY_TYPE, "NIRGraph")?;
    write_graph_body(&Writer::new(&root, opts), graph)
}

/// Reject node names that HDF5 cannot represent as a link, at any nesting depth.
fn check_names(graph: &NirGraph) -> Result<()> {
    for (name, node) in &graph.nodes {
        wire::check_node_name(name)?;
        if let NirNode::Graph(sub) = node {
            check_names(sub)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Graph structure
// ---------------------------------------------------------------------------

/// Write a graph's `nodes`, `edges` and `metadata` into `w`'s group.
///
/// The `type` dataset is the caller's responsibility: for a nested graph it has
/// already been written by [`write_node`], and writing it twice into the same
/// group is an HDF5 error.
fn write_graph_body(w: &Writer, graph: &NirGraph) -> Result<()> {
    let nodes = w.group.create_group(KEY_NODES)?;
    for (name, node) in &graph.nodes {
        let node_group = nodes.create_group(name)?;
        write_node(&w.rebind(&node_group), node)?;
    }

    write_edges(w.group, &graph.edges)?;
    write_metadata(w, &graph.metadata)
}

/// Always written, even when empty: upstream `NIRGraph.from_dict` asserts the
/// key is present.
fn write_edges(group: &Group, edges: &[(String, String)]) -> Result<()> {
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (src, dst) in edges {
        flat.push(var_str(src)?);
        flat.push(var_str(dst)?);
    }
    let ds = group
        .new_dataset::<VarLenUnicode>()
        .shape([edges.len(), 2])
        .create(KEY_EDGES)?;
    ds.write_raw(&flat)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Node dispatch
// ---------------------------------------------------------------------------

fn write_node(w: &Writer, node: &NirNode) -> Result<()> {
    write_string(w.group, KEY_TYPE, node.type_name())?;

    // A nested graph is a node group that also carries nodes and edges; its
    // metadata is written by `write_graph_body`.
    if let NirNode::Graph(sub) = node {
        return write_graph_body(w, sub);
    }

    let metadata = match node {
        NirNode::Input(n) => write_input(w, n).map(|()| &n.metadata),
        NirNode::Output(n) => write_output(w, n).map(|()| &n.metadata),
        NirNode::Affine(n) => write_affine(w, n).map(|()| &n.metadata),
        NirNode::Linear(n) => write_linear(w, n).map(|()| &n.metadata),
        NirNode::Scale(n) => write_scale(w, n).map(|()| &n.metadata),
        NirNode::Conv1d(n) => write_conv1d(w, n).map(|()| &n.metadata),
        NirNode::Conv2d(n) => write_conv2d(w, n).map(|()| &n.metadata),
        NirNode::CubaLi(n) => write_cuba_li(w, n).map(|()| &n.metadata),
        NirNode::CubaLif(n) => write_cuba_lif(w, n).map(|()| &n.metadata),
        NirNode::Delay(n) => write_delay(w, n).map(|()| &n.metadata),
        NirNode::Flatten(n) => write_flatten(w, n).map(|()| &n.metadata),
        NirNode::I(n) => write_i(w, n).map(|()| &n.metadata),
        NirNode::If(n) => write_if(w, n).map(|()| &n.metadata),
        NirNode::Li(n) => write_li(w, n).map(|()| &n.metadata),
        NirNode::Lif(n) => write_lif(w, n).map(|()| &n.metadata),
        NirNode::SumPool2d(n) => write_sum_pool2d(w, n).map(|()| &n.metadata),
        NirNode::AvgPool2d(n) => write_avg_pool2d(w, n).map(|()| &n.metadata),
        NirNode::Threshold(n) => write_threshold(w, n).map(|()| &n.metadata),
        NirNode::Graph(_) => unreachable!("handled above"),
    }?;

    write_metadata(w, metadata)
}

// ---------------------------------------------------------------------------
// Ports and linear maps
// ---------------------------------------------------------------------------

fn write_input(w: &Writer, node: &Input) -> Result<()> {
    w.usizes("shape", &node.shape)
}

fn write_output(w: &Writer, node: &Output) -> Result<()> {
    w.usizes("shape", &node.shape)
}

fn write_affine(w: &Writer, node: &Affine) -> Result<()> {
    w.tensor("weight", &node.weight)?;
    w.tensor("bias", &node.bias)
}

fn write_linear(w: &Writer, node: &Linear) -> Result<()> {
    w.tensor("weight", &node.weight)
}

fn write_scale(w: &Writer, node: &Scale) -> Result<()> {
    w.tensor("scale", &node.scale)
}

// ---------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------

fn write_conv1d(w: &Writer, node: &Conv1d) -> Result<()> {
    w.tensor("weight", &node.weight)?;
    w.conv_extent("stride", &node.stride, Rank::One)?;
    w.padding(&node.padding, Rank::One)?;
    w.conv_extent("dilation", &node.dilation, Rank::One)?;
    w.scalar("groups", node.groups)?;
    w.tensor("bias", &node.bias)?;
    if let Some(extent) = node.input_shape {
        w.scalar("input_shape", to_i64(extent, "input_shape")?)?;
    }
    Ok(())
}

fn write_conv2d(w: &Writer, node: &Conv2d) -> Result<()> {
    w.tensor("weight", &node.weight)?;
    w.conv_extent("stride", &node.stride, Rank::Two)?;
    w.padding(&node.padding, Rank::Two)?;
    w.conv_extent("dilation", &node.dilation, Rank::Two)?;
    w.scalar("groups", node.groups)?;
    w.tensor("bias", &node.bias)?;
    if let Some(shape) = &node.input_shape {
        w.usizes("input_shape", shape)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Neuron models
// ---------------------------------------------------------------------------

fn write_cuba_li(w: &Writer, node: &CubaLi) -> Result<()> {
    w.tensor("tau_syn", &node.tau_syn)?;
    w.tensor("tau_mem", &node.tau_mem)?;
    w.tensor("r", &node.r)?;
    w.tensor("v_leak", &node.v_leak)?;
    w.opt_tensor("w_in", node.w_in.as_ref())
}

fn write_cuba_lif(w: &Writer, node: &CubaLif) -> Result<()> {
    w.tensor("tau_syn", &node.tau_syn)?;
    w.tensor("tau_mem", &node.tau_mem)?;
    w.tensor("r", &node.r)?;
    w.tensor("v_leak", &node.v_leak)?;
    w.tensor("v_threshold", &node.v_threshold)?;
    w.opt_tensor("v_reset", node.v_reset.as_ref())?;
    w.opt_tensor("w_in", node.w_in.as_ref())
}

fn write_i(w: &Writer, node: &I) -> Result<()> {
    w.tensor("r", &node.r)
}

fn write_if(w: &Writer, node: &If) -> Result<()> {
    w.tensor("r", &node.r)?;
    w.tensor("v_threshold", &node.v_threshold)?;
    w.opt_tensor("v_reset", node.v_reset.as_ref())
}

fn write_li(w: &Writer, node: &Li) -> Result<()> {
    w.tensor("tau", &node.tau)?;
    w.tensor("r", &node.r)?;
    w.tensor("v_leak", &node.v_leak)
}

fn write_lif(w: &Writer, node: &Lif) -> Result<()> {
    w.tensor("tau", &node.tau)?;
    w.tensor("r", &node.r)?;
    w.tensor("v_leak", &node.v_leak)?;
    w.tensor("v_threshold", &node.v_threshold)?;
    w.opt_tensor("v_reset", node.v_reset.as_ref())
}

// ---------------------------------------------------------------------------
// Pooling and the remaining leaf nodes
// ---------------------------------------------------------------------------

/// `SumPool2d` and `AvgPool2d` carry an identical field set.
fn write_pool_window(
    w: &Writer,
    kernel_size: &Tensor,
    stride: &Tensor,
    pad: &Tensor,
) -> Result<()> {
    w.tensor("kernel_size", kernel_size)?;
    w.tensor("stride", stride)?;
    w.tensor("padding", pad)
}

fn write_sum_pool2d(w: &Writer, node: &SumPool2d) -> Result<()> {
    write_pool_window(w, &node.kernel_size, &node.stride, &node.padding)
}

fn write_avg_pool2d(w: &Writer, node: &AvgPool2d) -> Result<()> {
    write_pool_window(w, &node.kernel_size, &node.stride, &node.padding)
}

fn write_delay(w: &Writer, node: &Delay) -> Result<()> {
    w.tensor("delay", &node.delay)
}

fn write_flatten(w: &Writer, node: &Flatten) -> Result<()> {
    w.scalar("start_dim", node.start_dim)?;
    w.scalar("end_dim", node.end_dim)?;
    // Flatten stores its input shape under the key `input_type`.
    match &node.input_type {
        Some(shape) => w.usizes("input_type", shape),
        None => Ok(()),
    }
}

fn write_threshold(w: &Writer, node: &Threshold) -> Result<()> {
    w.tensor("threshold", &node.threshold)
}

/// Omitted entirely when empty, matching upstream's `if not v == {}` guard.
fn write_metadata(w: &Writer, metadata: &MetadataMap) -> Result<()> {
    if metadata.is_empty() {
        return Ok(());
    }
    let group = w.group.create_group(KEY_METADATA)?;
    let md = w.rebind(&group);
    for (key, value) in metadata {
        match value {
            MetadataValue::String(s) => write_string(md.group, key, s)?,
            MetadataValue::F64(v) => md.scalar(key, *v)?,
            MetadataValue::I64(v) => md.scalar(key, *v)?,
            MetadataValue::Bool(v) => md.scalar(key, *v)?,
            MetadataValue::Tensor(t) => md.tensor(key, t)?,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dataset writers
// ---------------------------------------------------------------------------

/// Which convolution an extent belongs to, and therefore how it is shaped.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rank {
    /// `Conv1d`: a bare scalar on the wire.
    One,
    /// `Conv2d`: a length-2 array on the wire.
    Two,
}

/// A destination group paired with the options that govern how datasets in it
/// are stored, so field writers need only a name and a value.
struct Writer<'a> {
    group: &'a Group,
    opts: &'a WriteOptions,
}

impl<'a> Writer<'a> {
    fn new(group: &'a Group, opts: &'a WriteOptions) -> Self {
        Self { group, opts }
    }

    /// The same options aimed at a different group.
    fn rebind<'b>(&self, group: &'b Group) -> Writer<'b>
    where
        'a: 'b,
    {
        Writer {
            group,
            opts: self.opts,
        }
    }

    fn tensor(&self, name: &str, tensor: &Tensor) -> Result<()> {
        let shape = tensor.shape();
        match tensor.data() {
            TensorData::F32(v) => self.array(name, shape, v),
            TensorData::F64(v) => self.array(name, shape, v),
            TensorData::I64(v) => self.array(name, shape, v),
            TensorData::Bool(v) => self.array(name, shape, v),
        }
    }

    fn opt_tensor(&self, name: &str, tensor: Option<&Tensor>) -> Result<()> {
        match tensor {
            Some(t) => self.tensor(name, t),
            None => Ok(()),
        }
    }

    fn usizes(&self, name: &str, values: &[usize]) -> Result<()> {
        let converted: Vec<i64> = values
            .iter()
            .map(|&v| to_i64(v, name))
            .collect::<Result<_>>()?;
        self.array(name, &[converted.len()], &converted)
    }

    fn conv_extent(&self, name: &str, values: &[i64], rank: Rank) -> Result<()> {
        match rank {
            Rank::One => match values {
                [only] => self.scalar(name, *only),
                other => Err(NirError::InvalidGraph(format!(
                    "Conv1d {name} must hold exactly one extent, found {}",
                    other.len()
                ))),
            },
            Rank::Two => self.array(name, &[values.len()], values),
        }
    }

    fn padding(&self, padding: &Padding, rank: Rank) -> Result<()> {
        match wire::padding_as_wire_str(padding) {
            Some(mode) => write_string(self.group, "padding", mode),
            None => {
                let Padding::Explicit(extents) = padding else {
                    unreachable!("padding_as_wire_str returns None only for Explicit");
                };
                self.conv_extent("padding", extents, rank)
            }
        }
    }

    /// Write an n-dimensional dataset, compressing only when it can be chunked.
    ///
    /// HDF5 requires a chunked layout for any filter, and scalar dataspaces
    /// cannot be chunked — so rank-0 datasets are always stored contiguously
    /// regardless of [`WriteOptions::compression`].
    fn array<T: H5Type>(&self, name: &str, shape: &[usize], data: &[T]) -> Result<()> {
        let mut builder = self.group.new_dataset::<T>();
        let compressible = !shape.is_empty() && !data.is_empty();
        if let Some(level) = self.opts.compression
            && compressible
        {
            builder = builder.deflate(level);
        }
        let ds = builder.shape(shape).create(name)?;

        if shape.is_empty() {
            // `Tensor` guarantees `shape product == data.len()`, so an empty
            // shape means exactly one element; `first` keeps that an error
            // rather than a panic if a future caller bypasses the invariant.
            let value = data.first().ok_or_else(|| {
                NirError::InvalidTensor(format!("{name}: scalar dataset needs one element, got 0"))
            })?;
            ds.write_scalar(value)?;
        } else {
            ds.write_raw(data)?;
        }
        Ok(())
    }

    fn scalar<T: H5Type>(&self, name: &str, value: T) -> Result<()> {
        let ds = self.group.new_dataset::<T>().shape(()).create(name)?;
        ds.write_scalar(&value)?;
        Ok(())
    }
}

fn to_i64(value: usize, field: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        NirError::InvalidTensor(format!("{field}: axis length {value} does not fit in i64"))
    })
}

fn write_string(group: &Group, name: &str, value: &str) -> Result<()> {
    let ds = group
        .new_dataset::<VarLenUnicode>()
        .shape(())
        .create(name)?;
    ds.write_scalar(&var_str(value)?)?;
    Ok(())
}

fn var_str(value: &str) -> Result<VarLenUnicode> {
    VarLenUnicode::from_str(value).map_err(|e| {
        NirError::Io(format!(
            "{value:?} cannot be encoded as an HDF5 string: {e}"
        ))
    })
}
