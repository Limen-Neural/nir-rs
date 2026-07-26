// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoding of HDF5 `.nir` files into [`NirGraph`].
//!
//! Mirrors upstream `nir/serialization.py` (`read` / `hdf2dict`) and the
//! `from_dict` of each node dataclass. Reading is deliberately permissive: the
//! graph is returned as the file describes it, without running
//! [`NirGraph::validate_structure`] — callers decide whether a structurally
//! odd file is a problem.
//!
//! The only normalization performed is documented on [`super::read`]: element
//! types are widened losslessly into [`DType`](crate::DType), and absent
//! optional fields are filled with the upstream Python defaults.
//!
//! [`read_node`] is a flat dispatch table over the wire `type` string; the
//! per-type functions below it own one node kind each.

use super::wire::{self, KEY_EDGES, KEY_METADATA, KEY_NODE, KEY_NODES, KEY_TYPE, KEY_VERSION};
use crate::error::{NirError, Result};
use crate::graph::NirGraph;
use crate::nodes::{
    Affine, AvgPool2d, Conv1d, Conv2d, CubaLi, CubaLif, Delay, Flatten, I, If, Input, Li, Lif,
    Linear, NirNode, Output, Padding, Scale, SumPool2d, Threshold,
};
use crate::types::{MetadataMap, MetadataValue, Tensor, TensorData};
use hdf5::types::{
    FixedAscii, FixedUnicode, FloatSize, IntSize, TypeDescriptor as Td, VarLenAscii, VarLenUnicode,
};
use hdf5::{Dataset, File, Group, LocationToken};
use std::path::Path;

/// Read a whole `.nir` file.
pub(super) fn read(path: &Path) -> Result<NirGraph> {
    let file = open(path)?;
    let root = file.group(KEY_NODE).map_err(|_| {
        NirError::MissingField(format!(
            "/{KEY_NODE} (not a NIR graph file: {})",
            path.display()
        ))
    })?;

    // `/node` is a serialized NIRGraph. Checking its type is what separates a
    // NIR file from unrelated HDF5 that happens to have a `node` group.
    let root_type = read_string_scalar(
        &root
            .dataset(KEY_TYPE)
            .map_err(|_| NirError::MissingField(format!("/{KEY_NODE}/{KEY_TYPE}")))?,
        KEY_NODE,
    )?;
    if root_type != "NIRGraph" {
        return Err(NirError::InvalidGraph(format!(
            "/{KEY_NODE} must be a NIRGraph, found {root_type:?}"
        )));
    }

    let mut graph = read_graph_body(&root, &format!("/{KEY_NODE}"), &mut Vec::new())?;
    graph.metadata = read_metadata(&root)?;
    // A missing /version is not an error; `read_version` is the strict accessor.
    graph.version = match file.dataset(KEY_VERSION) {
        Ok(ds) => Some(read_string_scalar(&ds, KEY_VERSION)?),
        Err(_) => None,
    };
    Ok(graph)
}

/// Read only `/version`.
pub(super) fn read_version(path: &Path) -> Result<String> {
    let file = open(path)?;
    let ds = file
        .dataset(KEY_VERSION)
        .map_err(|_| NirError::MissingField(format!("/{KEY_VERSION}")))?;
    read_string_scalar(&ds, KEY_VERSION)
}

fn open(path: &Path) -> Result<File> {
    File::open(path).map_err(|e| NirError::Io(format!("cannot open {}: {e}", path.display())))
}

// ---------------------------------------------------------------------------
// Graph structure
// ---------------------------------------------------------------------------

/// Read a graph's `nodes` and `edges`, leaving metadata to the caller.
///
/// Split out because a nested `NIRGraph` node and the root graph share this
/// body but source their metadata from different places. `context` is the
/// group's path, used to say *which* graph a missing key belongs to.
///
/// Both keys are **required**, matching upstream `NIRGraph.from_dict`, which
/// asserts their presence even for an empty graph. Defaulting a missing
/// `edges` to "no connections" would turn a truncated file into a silently
/// disconnected model.
fn read_graph_body(
    group: &Group,
    context: &str,
    visited: &mut Vec<LocationToken>,
) -> Result<NirGraph> {
    let token = group.loc_info()?.token;
    if visited.contains(&token) {
        return Err(NirError::InvalidGraph(format!(
            "{context}: nested NIRGraph hard-link cycle detected"
        )));
    }
    visited.push(token);
    // Pop on every exit path so a diamond of hard-links (same subgraph reached
    // via two parents) is allowed, while a link back to an ancestor is not.
    let result = read_graph_body_uncycled(group, context, visited);
    visited.pop();
    result
}

fn read_graph_body_uncycled(
    group: &Group,
    context: &str,
    visited: &mut Vec<LocationToken>,
) -> Result<NirGraph> {
    let mut graph = NirGraph::new();

    validate_group_links(group, context)?;

    let nodes = group
        .group(KEY_NODES)
        .map_err(|_| NirError::MissingField(format!("{context}/{KEY_NODES}")))?;
    validate_group_links(&nodes, &format!("{context}/{KEY_NODES}"))?;

    // HDF5 iterates links in name order; sorting makes that explicit and
    // keeps repeated reads of the same file identical.
    let mut names = nodes.member_names()?;
    names.sort();
    for name in names {
        let node_group = nodes
            .group(&name)
            .map_err(|e| NirError::Io(format!("node {name:?} is not a group: {e}")))?;
        validate_group_links(&node_group, &name)?;
        let node = read_node(&node_group, &name, visited)?;
        graph.insert_node(name, node)?;
    }

    let edges = group
        .dataset(KEY_EDGES)
        .map_err(|_| NirError::MissingField(format!("{context}/{KEY_EDGES}")))?;
    graph.edges = read_edges(&edges)?;

    Ok(graph)
}

fn read_edges(ds: &Dataset) -> Result<Vec<(String, String)>> {
    // h5py encodes an empty edge list as a zero-length dataset whose element
    // type is float, not string; treat any empty dataset as "no edges".
    if ds.size() == 0 {
        return Ok(Vec::new());
    }
    let shape = ds.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(NirError::InvalidGraph(format!(
            "{KEY_EDGES} must have shape (E, 2), found {shape:?}"
        )));
    }
    let flat = read_strings(ds, KEY_EDGES)?;
    Ok(flat
        .chunks_exact(2)
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect())
}

// ---------------------------------------------------------------------------
// Node dispatch
// ---------------------------------------------------------------------------

fn read_node(group: &Group, name: &str, visited: &mut Vec<LocationToken>) -> Result<NirNode> {
    let type_ds = group
        .dataset(KEY_TYPE)
        .map_err(|_| NirError::MissingField(format!("{name}.{KEY_TYPE}")))?;
    let ty = read_string_scalar(&type_ds, name)?;
    let metadata = read_metadata(group)?;
    let r = NodeReader { group, name };

    let node = match ty.as_str() {
        "Input" => NirNode::Input(read_input(&r, metadata)?),
        "Output" => NirNode::Output(read_output(&r, metadata)?),
        "Affine" => NirNode::Affine(read_affine(&r, metadata)?),
        "Linear" => NirNode::Linear(read_linear(&r, metadata)?),
        "Scale" => NirNode::Scale(read_scale(&r, metadata)?),
        "Conv1d" => NirNode::Conv1d(read_conv1d(&r, metadata)?),
        "Conv2d" => NirNode::Conv2d(read_conv2d(&r, metadata)?),
        "CubaLI" => NirNode::CubaLi(read_cuba_li(&r, metadata)?),
        "CubaLIF" => NirNode::CubaLif(read_cuba_lif(&r, metadata)?),
        "Delay" => NirNode::Delay(read_delay(&r, metadata)?),
        "Flatten" => NirNode::Flatten(read_flatten(&r, metadata)?),
        "I" => NirNode::I(read_i(&r, metadata)?),
        "IF" => NirNode::If(read_if(&r, metadata)?),
        "LI" => NirNode::Li(read_li(&r, metadata)?),
        "LIF" => NirNode::Lif(read_lif(&r, metadata)?),
        "SumPool2d" => NirNode::SumPool2d(read_sum_pool2d(&r, metadata)?),
        "AvgPool2d" => NirNode::AvgPool2d(read_avg_pool2d(&r, metadata)?),
        "Threshold" => NirNode::Threshold(read_threshold(&r, metadata)?),
        "NIRGraph" => {
            let mut sub = read_graph_body(group, name, visited)?;
            sub.metadata = metadata;
            NirNode::Graph(Box::new(sub))
        }
        other => return Err(NirError::UnknownNodeType(other.to_owned())),
    };

    debug_assert!(
        wire::is_wire_type(node.type_name()),
        "decoded a node whose type is not in WIRE_TYPES"
    );
    Ok(node)
}

// ---------------------------------------------------------------------------
// Ports and linear maps
// ---------------------------------------------------------------------------

fn read_input(r: &NodeReader, metadata: MetadataMap) -> Result<Input> {
    Ok(Input {
        shape: r.usizes("shape")?,
        metadata,
    })
}

fn read_output(r: &NodeReader, metadata: MetadataMap) -> Result<Output> {
    Ok(Output {
        shape: r.usizes("shape")?,
        metadata,
    })
}

fn read_affine(r: &NodeReader, metadata: MetadataMap) -> Result<Affine> {
    Ok(Affine {
        weight: r.tensor("weight")?,
        bias: r.tensor("bias")?,
        metadata,
    })
}

fn read_linear(r: &NodeReader, metadata: MetadataMap) -> Result<Linear> {
    Ok(Linear {
        weight: r.tensor("weight")?,
        metadata,
    })
}

fn read_scale(r: &NodeReader, metadata: MetadataMap) -> Result<Scale> {
    Ok(Scale {
        scale: r.tensor("scale")?,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------

fn read_conv1d(r: &NodeReader, metadata: MetadataMap) -> Result<Conv1d> {
    Ok(Conv1d {
        weight: r.tensor("weight")?,
        stride: r.ints("stride")?,
        padding: r.padding()?,
        dilation: r.ints("dilation")?,
        groups: r.int_scalar("groups")?,
        bias: r.tensor("bias")?,
        // Upstream `Conv1d.input_shape` is a bare `int`.
        input_shape: r.opt_dim("input_shape")?,
        metadata,
    })
}

fn read_conv2d(r: &NodeReader, metadata: MetadataMap) -> Result<Conv2d> {
    Ok(Conv2d {
        weight: r.tensor("weight")?,
        stride: r.ints("stride")?,
        padding: r.padding()?,
        dilation: r.ints("dilation")?,
        groups: r.int_scalar("groups")?,
        bias: r.tensor("bias")?,
        // Upstream `Conv2d.input_shape` is a `(N_x, N_y)` tuple.
        input_shape: r.opt_usizes("input_shape")?,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Neuron models
// ---------------------------------------------------------------------------

fn read_cuba_li(r: &NodeReader, metadata: MetadataMap) -> Result<CubaLi> {
    let v_leak = r.tensor("v_leak")?;
    Ok(CubaLi {
        tau_syn: r.tensor("tau_syn")?,
        tau_mem: r.tensor("tau_mem")?,
        r: r.tensor("r")?,
        w_in: r.w_in(&v_leak)?,
        v_leak,
        metadata,
    })
}

fn read_cuba_lif(r: &NodeReader, metadata: MetadataMap) -> Result<CubaLif> {
    let v_leak = r.tensor("v_leak")?;
    let v_threshold = r.tensor("v_threshold")?;
    Ok(CubaLif {
        tau_syn: r.tensor("tau_syn")?,
        tau_mem: r.tensor("tau_mem")?,
        r: r.tensor("r")?,
        v_reset: r.v_reset(&v_threshold)?,
        w_in: r.w_in(&v_leak)?,
        v_leak,
        v_threshold,
        metadata,
    })
}

fn read_i(r: &NodeReader, metadata: MetadataMap) -> Result<I> {
    Ok(I {
        r: r.tensor("r")?,
        metadata,
    })
}

fn read_if(r: &NodeReader, metadata: MetadataMap) -> Result<If> {
    let v_threshold = r.tensor("v_threshold")?;
    Ok(If {
        r: r.tensor("r")?,
        v_reset: r.v_reset(&v_threshold)?,
        v_threshold,
        metadata,
    })
}

fn read_li(r: &NodeReader, metadata: MetadataMap) -> Result<Li> {
    Ok(Li {
        tau: r.tensor("tau")?,
        r: r.tensor("r")?,
        v_leak: r.tensor("v_leak")?,
        metadata,
    })
}

fn read_lif(r: &NodeReader, metadata: MetadataMap) -> Result<Lif> {
    let v_threshold = r.tensor("v_threshold")?;
    Ok(Lif {
        tau: r.tensor("tau")?,
        r: r.tensor("r")?,
        v_leak: r.tensor("v_leak")?,
        v_reset: r.v_reset(&v_threshold)?,
        v_threshold,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Pooling and the remaining leaf nodes
// ---------------------------------------------------------------------------

/// `SumPool2d` and `AvgPool2d` carry an identical field set.
fn read_pool_window(r: &NodeReader) -> Result<(Tensor, Tensor, Tensor)> {
    Ok((
        r.tensor("kernel_size")?,
        r.tensor("stride")?,
        r.tensor("padding")?,
    ))
}

fn read_sum_pool2d(r: &NodeReader, metadata: MetadataMap) -> Result<SumPool2d> {
    let (kernel_size, stride, padding) = read_pool_window(r)?;
    Ok(SumPool2d {
        kernel_size,
        stride,
        padding,
        metadata,
    })
}

fn read_avg_pool2d(r: &NodeReader, metadata: MetadataMap) -> Result<AvgPool2d> {
    let (kernel_size, stride, padding) = read_pool_window(r)?;
    Ok(AvgPool2d {
        kernel_size,
        stride,
        padding,
        metadata,
    })
}

fn read_delay(r: &NodeReader, metadata: MetadataMap) -> Result<Delay> {
    Ok(Delay {
        delay: r.tensor("delay")?,
        metadata,
    })
}

fn read_flatten(r: &NodeReader, metadata: MetadataMap) -> Result<Flatten> {
    Ok(Flatten {
        // Upstream's dataclass defaults are start_dim = 1, end_dim = -1.
        start_dim: r.opt_int_scalar("start_dim")?.unwrap_or(1),
        end_dim: r.opt_int_scalar("end_dim")?.unwrap_or(-1),
        // Flatten stores its input shape under the key `input_type`.
        input_type: r.opt_usizes("input_type")?,
        metadata,
    })
}

fn read_threshold(r: &NodeReader, metadata: MetadataMap) -> Result<Threshold> {
    Ok(Threshold {
        threshold: r.tensor("threshold")?,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Field access
// ---------------------------------------------------------------------------

/// A node's HDF5 group paired with its name.
///
/// Bundling the two means each accessor takes only a field name, and every
/// error message can say which node it came from without the caller repeating
/// itself.
struct NodeReader<'a> {
    group: &'a Group,
    name: &'a str,
}

impl NodeReader<'_> {
    /// `node.field`, the prefix every error message from this node uses.
    fn context(&self, field: &str) -> String {
        format!("{}.{field}", self.name)
    }

    /// The dataset for an optional field, or [`None`] when it is absent.
    ///
    /// A link that *exists* but is not a dataset is an error rather than an
    /// absent field: treating a malformed `v_reset` group as "not there" would
    /// silently synthesize a default and change the model.
    fn optional(&self, field: &str) -> Result<Option<Dataset>> {
        if !self.group.link_exists(field) {
            return Ok(None);
        }
        self.group.dataset(field).map(Some).map_err(|e| {
            NirError::Io(format!(
                "{}: expected a dataset, found another link kind: {e}",
                self.context(field)
            ))
        })
    }

    fn required(&self, field: &str) -> Result<Dataset> {
        self.optional(field)?
            .ok_or_else(|| NirError::MissingField(self.context(field)))
    }

    fn tensor(&self, field: &str) -> Result<Tensor> {
        read_tensor(&self.required(field)?, &self.context(field))
    }

    fn opt_tensor(&self, field: &str) -> Result<Option<Tensor>> {
        match self.optional(field)? {
            Some(ds) => read_tensor(&ds, &self.context(field)).map(Some),
            None => Ok(None),
        }
    }

    fn ints(&self, field: &str) -> Result<Vec<i64>> {
        read_ints(&self.required(field)?, &self.context(field))
    }

    fn opt_ints(&self, field: &str) -> Result<Option<Vec<i64>>> {
        match self.optional(field)? {
            Some(ds) => read_ints(&ds, &self.context(field)).map(Some),
            None => Ok(None),
        }
    }

    fn int_scalar(&self, field: &str) -> Result<i64> {
        single_int(self.ints(field)?, &self.context(field))
    }

    fn opt_int_scalar(&self, field: &str) -> Result<Option<i64>> {
        match self.opt_ints(field)? {
            Some(values) => single_int(values, &self.context(field)).map(Some),
            None => Ok(None),
        }
    }

    fn usizes(&self, field: &str) -> Result<Vec<usize>> {
        to_usizes(self.ints(field)?, &self.context(field))
    }

    fn opt_usizes(&self, field: &str) -> Result<Option<Vec<usize>>> {
        match self.opt_ints(field)? {
            Some(values) => to_usizes(values, &self.context(field)).map(Some),
            None => Ok(None),
        }
    }

    /// A single non-negative extent, for `Conv1d.input_shape`.
    fn opt_dim(&self, field: &str) -> Result<Option<usize>> {
        let Some(values) = self.opt_usizes(field)? else {
            return Ok(None);
        };
        match values.as_slice() {
            [only] => Ok(Some(*only)),
            other => Err(NirError::InvalidTensor(format!(
                "{}: expected a single extent, found {} values",
                self.context(field),
                other.len()
            ))),
        }
    }

    /// `padding` is an integer extent list, or the string `"same"` / `"valid"`.
    fn padding(&self) -> Result<Padding> {
        let ds = self.required("padding")?;
        if is_string(&ds)? {
            wire::padding_from_wire_str(&read_string_scalar(&ds, self.name)?)
        } else {
            Ok(Padding::Explicit(read_ints(&ds, &self.context("padding"))?))
        }
    }

    /// `v_reset`, defaulting to `zeros_like(v_threshold)` as Python does.
    fn v_reset(&self, v_threshold: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(
            self.opt_tensor("v_reset")?
                .unwrap_or_else(|| v_threshold.zeros_like()),
        ))
    }

    /// `w_in`, defaulting to `ones_like(v_leak)` as Python does.
    fn w_in(&self, v_leak: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(
            self.opt_tensor("w_in")?
                .unwrap_or_else(|| v_leak.ones_like()),
        ))
    }
}

fn to_usizes(values: Vec<i64>, context: &str) -> Result<Vec<usize>> {
    values
        .into_iter()
        .map(|v| {
            usize::try_from(v).map_err(|_| {
                NirError::InvalidTensor(format!("{context}: negative axis length {v}"))
            })
        })
        .collect()
}

fn single_int(values: Vec<i64>, context: &str) -> Result<i64> {
    match values.as_slice() {
        [only] => Ok(*only),
        other => Err(NirError::InvalidTensor(format!(
            "{context}: expected a single integer, found {} values",
            other.len()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Security validation (reject external links and VDS)
// ---------------------------------------------------------------------------

/// Validate that a group does not contain external links before traversing it.
///
/// Checks each member link to ensure it's not an external link that could
/// reference files outside the current `.nir` file.
fn validate_group_links(group: &Group, context: &str) -> Result<()> {
    // H5L_TYPE_EXTERNAL links reference objects in other files; hard and soft
    // links stay inside this one, so they are fine. `iter_visit_default`
    // visits the group's immediate members; returning false stops the scan
    // early once an external link is found.
    let external = group
        .iter_visit_default(None, |_group, name, info, found: &mut Option<String>| {
            if info.link_type == hdf5::LinkType::External {
                *found = Some(name.to_owned());
                false
            } else {
                true
            }
        })
        .map_err(|e| NirError::Io(format!("{context}: cannot inspect group links: {e}")))?;
    if let Some(name) = external {
        return Err(NirError::InvalidGraph(format!(
            "{context}: external link '{name}' is not allowed"
        )));
    }
    Ok(())
}

/// Validate that a dataset does not use disallowed storage layouts.
///
/// Rejects external storage (H5D_EXTERNAL) so dataset data cannot be pulled
/// from files outside the `.nir` file. Fail closed when the dataset creation
/// property list cannot be read. Note: hdf5-metno 0.14 does not compile its
/// `Layout::Virtual` variant, so VDS layouts cannot be detected through the
/// high-level API; external links and external storage are the enforced
/// boundaries.
fn validate_dataset_security(ds: &Dataset, context: &str) -> Result<()> {
    let dcpl = ds.dcpl().map_err(|e| {
        NirError::Io(format!(
            "{context}: cannot read dataset creation property list: {e}"
        ))
    })?;
    let external = dcpl.external();
    if !external.is_empty() {
        return Err(NirError::InvalidGraph(format!(
            "{context}: external storage is not allowed ({} external file(s))",
            external.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

fn read_metadata(group: &Group) -> Result<MetadataMap> {
    // Absent is fine (empty map). A present link of the wrong kind — e.g. a
    // dataset named `metadata` — is an error, not silent data loss.
    if !group.link_exists(KEY_METADATA) {
        return Ok(MetadataMap::new());
    }
    let md = group.group(KEY_METADATA).map_err(|e| {
        NirError::Io(format!(
            "{KEY_METADATA}: expected a group, found another link kind: {e}"
        ))
    })?;
    validate_group_links(&md, KEY_METADATA)?;
    let mut out = MetadataMap::new();
    for key in md.member_names()? {
        let ds = md.dataset(&key).map_err(|e| {
            NirError::Io(format!(
                "metadata {key:?} must be a dataset, not a group: {e}"
            ))
        })?;
        let value = read_metadata_value(&ds, &key)?;
        out.insert(key, value);
    }
    Ok(out)
}

fn read_metadata_value(ds: &Dataset, key: &str) -> Result<MetadataValue> {
    validate_dataset_security(ds, &format!("{KEY_METADATA}.{key}"))?;
    let scalar = ds.shape().is_empty();
    let context = format!("{KEY_METADATA}.{key}");
    let value = match ds.dtype()?.to_descriptor()? {
        Td::VarLenUnicode | Td::VarLenAscii | Td::FixedAscii(_) | Td::FixedUnicode(_) => {
            MetadataValue::String(read_string_scalar(ds, key)?)
        }
        Td::Boolean if scalar => MetadataValue::Bool(ds.read_scalar::<bool>()?),
        Td::Float(_) if scalar => MetadataValue::F64(ds.read_scalar::<f64>()?),
        Td::Integer(_) if scalar => MetadataValue::I64(ds.read_scalar::<i64>()?),
        Td::Unsigned(IntSize::U8) if scalar => {
            // Same checked conversion as tensor `u64` payloads — HDF5's own
            // i64 cast can saturate above i64::MAX.
            let v = ds.read_scalar::<u64>()?;
            MetadataValue::I64(i64::try_from(v).map_err(|_| {
                NirError::InvalidTensor(format!("{context}: u64 value {v} does not fit in i64"))
            })?)
        }
        Td::Unsigned(IntSize::U1 | IntSize::U2 | IntSize::U4) if scalar => {
            MetadataValue::I64(ds.read_scalar::<i64>()?)
        }
        _ => MetadataValue::Tensor(read_tensor(ds, &context)?),
    };
    Ok(value)
}

// ---------------------------------------------------------------------------
// Typed dataset decoding
// ---------------------------------------------------------------------------

/// Decode a numeric dataset, preserving its on-disk element type.
///
/// Narrower integers widen into [`DType::I64`](crate::DType::I64); floats keep
/// their width so an `f32` file never silently becomes `f64` (or worse, the
/// reverse). Anything else is rejected rather than guessed at.
fn read_tensor(ds: &Dataset, context: &str) -> Result<Tensor> {
    validate_dataset_security(ds, context)?;
    let descriptor = ds.dtype()?.to_descriptor()?;
    let data = match descriptor {
        Td::Float(FloatSize::U4) => TensorData::F32(ds.read_raw::<f32>()?),
        Td::Float(FloatSize::U8) => TensorData::F64(ds.read_raw::<f64>()?),
        Td::Integer(_) | Td::Unsigned(IntSize::U1 | IntSize::U2 | IntSize::U4) => {
            TensorData::I64(ds.read_raw::<i64>()?)
        }
        Td::Unsigned(IntSize::U8) => TensorData::I64(read_u64_as_i64(ds, context)?),
        Td::Boolean => TensorData::Bool(ds.read_raw::<bool>()?),
        other => {
            return Err(NirError::InvalidTensor(format!(
                "{context}: element type {other} has no NIR dtype"
            )));
        }
    };
    Tensor::new(ds.shape(), data)
}

/// `u64` is the one integer width that does not fit losslessly in `i64`.
fn read_u64_as_i64(ds: &Dataset, context: &str) -> Result<Vec<i64>> {
    ds.read_raw::<u64>()?
        .into_iter()
        .map(|v| {
            i64::try_from(v).map_err(|_| {
                NirError::InvalidTensor(format!("{context}: u64 value {v} does not fit in i64"))
            })
        })
        .collect()
}

fn read_ints(ds: &Dataset, context: &str) -> Result<Vec<i64>> {
    match read_tensor(ds, context)?.data() {
        TensorData::I64(values) => Ok(values.clone()),
        other => Err(NirError::InvalidTensor(format!(
            "{context}: expected integer data, found {:?}",
            other.dtype()
        ))),
    }
}

fn is_string(ds: &Dataset) -> Result<bool> {
    Ok(matches!(
        ds.dtype()?.to_descriptor()?,
        Td::VarLenUnicode | Td::VarLenAscii | Td::FixedAscii(_) | Td::FixedUnicode(_)
    ))
}

fn read_string_scalar(ds: &Dataset, context: &str) -> Result<String> {
    let mut values = read_strings(ds, context)?;
    match values.len() {
        1 => Ok(values.remove(0)),
        n => Err(NirError::Io(format!(
            "{context}: expected a single string, found {n}"
        ))),
    }
}

/// Read every element of a string dataset, in C order.
///
/// h5py can emit any of the four HDF5 string flavors depending on version and
/// how the value was passed, so all four are handled. Fixed-length strings are
/// read through a capacity ladder because `FixedAscii<N>` is const-generic:
/// HDF5 converts the on-disk width up to the requested one.
fn read_strings(ds: &Dataset, context: &str) -> Result<Vec<String>> {
    macro_rules! read_fixed {
        ($ty:ident, $width:expr, $($cap:literal),+) => {
            $(if $width <= $cap {
                return Ok(ds
                    .read_raw::<$ty<$cap>>()?
                    .iter()
                    .map(ToString::to_string)
                    .collect());
            })+
        };
    }

    match ds.dtype()?.to_descriptor()? {
        Td::VarLenUnicode => Ok(ds
            .read_raw::<VarLenUnicode>()?
            .iter()
            .map(ToString::to_string)
            .collect()),
        Td::VarLenAscii => Ok(ds
            .read_raw::<VarLenAscii>()?
            .iter()
            .map(ToString::to_string)
            .collect()),
        Td::FixedAscii(width) => {
            read_fixed!(FixedAscii, width, 64, 256, 4096);
            Err(too_wide(context, width))
        }
        Td::FixedUnicode(width) => {
            read_fixed!(FixedUnicode, width, 64, 256, 4096);
            Err(too_wide(context, width))
        }
        other => Err(NirError::Io(format!(
            "{context}: expected a string dataset, found {other}"
        ))),
    }
}

fn too_wide(context: &str, width: usize) -> NirError {
    NirError::Io(format!(
        "{context}: fixed-length string of {width} bytes exceeds the supported maximum of 4096"
    ))
}
