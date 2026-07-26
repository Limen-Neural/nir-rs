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
use crate::nodes::{NirNode, Padding};
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
    write_graph_body(&root, graph, opts)
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
// Graph and node structure
// ---------------------------------------------------------------------------

/// Write a graph's `nodes`, `edges` and `metadata` into `group`.
///
/// The `type` dataset is the caller's responsibility: for a nested graph it has
/// already been written by [`write_node`], and writing it twice into the same
/// group is an HDF5 error.
fn write_graph_body(group: &Group, graph: &NirGraph, opts: &WriteOptions) -> Result<()> {
    let nodes = group.create_group(KEY_NODES)?;
    for (name, node) in &graph.nodes {
        let node_group = nodes.create_group(name)?;
        write_node(&node_group, node, opts)?;
    }

    write_edges(group, &graph.edges)?;
    write_metadata(group, &graph.metadata, opts)
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

fn write_node(group: &Group, node: &NirNode, opts: &WriteOptions) -> Result<()> {
    write_string(group, KEY_TYPE, node.type_name())?;

    let metadata = match node {
        NirNode::Input(n) => {
            write_usizes(group, "shape", &n.shape, opts)?;
            &n.metadata
        }
        NirNode::Output(n) => {
            write_usizes(group, "shape", &n.shape, opts)?;
            &n.metadata
        }
        NirNode::Affine(n) => {
            write_tensor(group, "weight", &n.weight, opts)?;
            write_tensor(group, "bias", &n.bias, opts)?;
            &n.metadata
        }
        NirNode::Linear(n) => {
            write_tensor(group, "weight", &n.weight, opts)?;
            &n.metadata
        }
        NirNode::Scale(n) => {
            write_tensor(group, "scale", &n.scale, opts)?;
            &n.metadata
        }
        NirNode::Conv1d(n) => {
            write_tensor(group, "weight", &n.weight, opts)?;
            write_conv_extent(group, "stride", &n.stride, Rank::One, opts)?;
            write_padding(group, &n.padding, Rank::One, opts)?;
            write_conv_extent(group, "dilation", &n.dilation, Rank::One, opts)?;
            write_scalar(group, "groups", n.groups)?;
            write_tensor(group, "bias", &n.bias, opts)?;
            if let Some(extent) = n.input_shape {
                write_scalar(group, "input_shape", to_i64(extent, "input_shape")?)?;
            }
            &n.metadata
        }
        NirNode::Conv2d(n) => {
            write_tensor(group, "weight", &n.weight, opts)?;
            write_conv_extent(group, "stride", &n.stride, Rank::Two, opts)?;
            write_padding(group, &n.padding, Rank::Two, opts)?;
            write_conv_extent(group, "dilation", &n.dilation, Rank::Two, opts)?;
            write_scalar(group, "groups", n.groups)?;
            write_tensor(group, "bias", &n.bias, opts)?;
            if let Some(shape) = &n.input_shape {
                write_usizes(group, "input_shape", shape, opts)?;
            }
            &n.metadata
        }
        NirNode::CubaLi(n) => {
            write_tensor(group, "tau_syn", &n.tau_syn, opts)?;
            write_tensor(group, "tau_mem", &n.tau_mem, opts)?;
            write_tensor(group, "r", &n.r, opts)?;
            write_tensor(group, "v_leak", &n.v_leak, opts)?;
            write_opt_tensor(group, "w_in", n.w_in.as_ref(), opts)?;
            &n.metadata
        }
        NirNode::CubaLif(n) => {
            write_tensor(group, "tau_syn", &n.tau_syn, opts)?;
            write_tensor(group, "tau_mem", &n.tau_mem, opts)?;
            write_tensor(group, "r", &n.r, opts)?;
            write_tensor(group, "v_leak", &n.v_leak, opts)?;
            write_tensor(group, "v_threshold", &n.v_threshold, opts)?;
            write_opt_tensor(group, "v_reset", n.v_reset.as_ref(), opts)?;
            write_opt_tensor(group, "w_in", n.w_in.as_ref(), opts)?;
            &n.metadata
        }
        NirNode::Delay(n) => {
            write_tensor(group, "delay", &n.delay, opts)?;
            &n.metadata
        }
        NirNode::Flatten(n) => {
            write_scalar(group, "start_dim", n.start_dim)?;
            write_scalar(group, "end_dim", n.end_dim)?;
            if let Some(shape) = &n.input_type {
                write_usizes(group, "input_type", shape, opts)?;
            }
            &n.metadata
        }
        NirNode::I(n) => {
            write_tensor(group, "r", &n.r, opts)?;
            &n.metadata
        }
        NirNode::If(n) => {
            write_tensor(group, "r", &n.r, opts)?;
            write_tensor(group, "v_threshold", &n.v_threshold, opts)?;
            write_opt_tensor(group, "v_reset", n.v_reset.as_ref(), opts)?;
            &n.metadata
        }
        NirNode::Li(n) => {
            write_tensor(group, "tau", &n.tau, opts)?;
            write_tensor(group, "r", &n.r, opts)?;
            write_tensor(group, "v_leak", &n.v_leak, opts)?;
            &n.metadata
        }
        NirNode::Lif(n) => {
            write_tensor(group, "tau", &n.tau, opts)?;
            write_tensor(group, "r", &n.r, opts)?;
            write_tensor(group, "v_leak", &n.v_leak, opts)?;
            write_tensor(group, "v_threshold", &n.v_threshold, opts)?;
            write_opt_tensor(group, "v_reset", n.v_reset.as_ref(), opts)?;
            &n.metadata
        }
        NirNode::SumPool2d(n) => {
            write_tensor(group, "kernel_size", &n.kernel_size, opts)?;
            write_tensor(group, "stride", &n.stride, opts)?;
            write_tensor(group, "padding", &n.padding, opts)?;
            &n.metadata
        }
        NirNode::AvgPool2d(n) => {
            write_tensor(group, "kernel_size", &n.kernel_size, opts)?;
            write_tensor(group, "stride", &n.stride, opts)?;
            write_tensor(group, "padding", &n.padding, opts)?;
            &n.metadata
        }
        NirNode::Threshold(n) => {
            write_tensor(group, "threshold", &n.threshold, opts)?;
            &n.metadata
        }
        NirNode::Graph(sub) => {
            // A nested graph is a node group that also carries nodes and edges;
            // its metadata is written by `write_graph_body`.
            return write_graph_body(group, sub, opts);
        }
    };

    write_metadata(group, metadata, opts)
}

/// Which convolution the extent belongs to, and therefore how it is shaped.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rank {
    /// `Conv1d`: a bare scalar on the wire.
    One,
    /// `Conv2d`: a length-2 array on the wire.
    Two,
}

fn write_conv_extent(
    group: &Group,
    field: &str,
    values: &[i64],
    rank: Rank,
    opts: &WriteOptions,
) -> Result<()> {
    match rank {
        Rank::One => match values {
            [only] => write_scalar(group, field, *only),
            other => Err(NirError::InvalidGraph(format!(
                "Conv1d {field} must hold exactly one extent, found {}",
                other.len()
            ))),
        },
        Rank::Two => write_array(group, field, &[values.len()], values, opts),
    }
}

fn write_padding(group: &Group, padding: &Padding, rank: Rank, opts: &WriteOptions) -> Result<()> {
    match wire::padding_as_wire_str(padding) {
        Some(mode) => write_string(group, "padding", mode),
        None => {
            let Padding::Explicit(extents) = padding else {
                unreachable!("padding_as_wire_str returns None only for Explicit");
            };
            write_conv_extent(group, "padding", extents, rank, opts)
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Omitted entirely when empty, matching upstream's `if not v == {}` guard.
fn write_metadata(group: &Group, metadata: &MetadataMap, opts: &WriteOptions) -> Result<()> {
    if metadata.is_empty() {
        return Ok(());
    }
    let md = group.create_group(KEY_METADATA)?;
    for (key, value) in metadata {
        match value {
            MetadataValue::String(s) => write_string(&md, key, s)?,
            MetadataValue::F64(v) => write_scalar(&md, key, *v)?,
            MetadataValue::I64(v) => write_scalar(&md, key, *v)?,
            MetadataValue::Bool(v) => write_scalar(&md, key, *v)?,
            MetadataValue::Tensor(t) => write_tensor(&md, key, t, opts)?,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dataset writers
// ---------------------------------------------------------------------------

fn write_tensor(group: &Group, name: &str, tensor: &Tensor, opts: &WriteOptions) -> Result<()> {
    let shape = tensor.shape();
    match tensor.data() {
        TensorData::F32(v) => write_array(group, name, shape, v, opts),
        TensorData::F64(v) => write_array(group, name, shape, v, opts),
        TensorData::I64(v) => write_array(group, name, shape, v, opts),
        TensorData::Bool(v) => write_array(group, name, shape, v, opts),
    }
}

fn write_opt_tensor(
    group: &Group,
    name: &str,
    tensor: Option<&Tensor>,
    opts: &WriteOptions,
) -> Result<()> {
    match tensor {
        Some(t) => write_tensor(group, name, t, opts),
        None => Ok(()),
    }
}

fn write_usizes(group: &Group, name: &str, values: &[usize], opts: &WriteOptions) -> Result<()> {
    let converted: Vec<i64> = values
        .iter()
        .map(|&v| to_i64(v, name))
        .collect::<Result<_>>()?;
    write_array(group, name, &[converted.len()], &converted, opts)
}

fn to_i64(value: usize, field: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        NirError::InvalidTensor(format!("{field}: axis length {value} does not fit in i64"))
    })
}

/// Write an n-dimensional dataset, compressing only when it can be chunked.
///
/// HDF5 requires a chunked layout for any filter, and scalar dataspaces cannot
/// be chunked — so rank-0 datasets are always stored contiguously regardless of
/// [`WriteOptions::compression`].
fn write_array<T: H5Type>(
    group: &Group,
    name: &str,
    shape: &[usize],
    data: &[T],
    opts: &WriteOptions,
) -> Result<()> {
    let mut builder = group.new_dataset::<T>();
    let compressible = !shape.is_empty() && !data.is_empty();
    if let Some(level) = opts.compression
        && compressible
    {
        builder = builder.deflate(level);
    }
    let ds = builder.shape(shape).create(name)?;
    if shape.is_empty() {
        ds.write_scalar(&data[0])?;
    } else {
        ds.write_raw(data)?;
    }
    Ok(())
}

fn write_scalar<T: H5Type>(group: &Group, name: &str, value: T) -> Result<()> {
    let ds = group.new_dataset::<T>().shape(()).create(name)?;
    ds.write_scalar(&value)?;
    Ok(())
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
