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
use hdf5::{Dataset, File, Group};
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

    let mut graph = read_graph_body(&root)?;
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
// Graph and node structure
// ---------------------------------------------------------------------------

/// Read a graph's `nodes` and `edges`, leaving metadata to the caller.
///
/// Split out because a nested `NIRGraph` node and the root graph share this
/// body but source their metadata from different places.
fn read_graph_body(group: &Group) -> Result<NirGraph> {
    let mut graph = NirGraph::new();

    if let Ok(nodes) = group.group(KEY_NODES) {
        // HDF5 iterates links in name order; sorting makes that explicit and
        // keeps repeated reads of the same file identical.
        let mut names = nodes.member_names()?;
        names.sort();
        for name in names {
            let node_group = nodes
                .group(&name)
                .map_err(|e| NirError::Io(format!("node {name:?} is not a group: {e}")))?;
            let node = read_node(&node_group, &name)?;
            graph.insert_node(name, node)?;
        }
    }

    graph.edges = match group.dataset(KEY_EDGES) {
        Ok(ds) => read_edges(&ds)?,
        Err(_) => Vec::new(),
    };

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

fn read_node(group: &Group, name: &str) -> Result<NirNode> {
    let type_ds = group
        .dataset(KEY_TYPE)
        .map_err(|_| NirError::MissingField(format!("{name}.{KEY_TYPE}")))?;
    let ty = read_string_scalar(&type_ds, name)?;
    let metadata = read_metadata(group)?;

    let node = match ty.as_str() {
        "Input" => NirNode::Input(Input {
            shape: usizes(group, "shape", name)?,
            metadata,
        }),
        "Output" => NirNode::Output(Output {
            shape: usizes(group, "shape", name)?,
            metadata,
        }),
        "Affine" => NirNode::Affine(Affine {
            weight: tensor(group, "weight", name)?,
            bias: tensor(group, "bias", name)?,
            metadata,
        }),
        "Linear" => NirNode::Linear(Linear {
            weight: tensor(group, "weight", name)?,
            metadata,
        }),
        "Scale" => NirNode::Scale(Scale {
            scale: tensor(group, "scale", name)?,
            metadata,
        }),
        "Conv1d" => NirNode::Conv1d(Conv1d {
            weight: tensor(group, "weight", name)?,
            stride: ints(group, "stride", name)?,
            padding: read_padding(group, name)?,
            dilation: ints(group, "dilation", name)?,
            groups: int_scalar(group, "groups", name)?,
            bias: tensor(group, "bias", name)?,
            input_shape: match opt_usizes(group, "input_shape", name)? {
                Some(dims) => Some(single_dim(dims, name, "input_shape")?),
                None => None,
            },
            metadata,
        }),
        "Conv2d" => NirNode::Conv2d(Conv2d {
            weight: tensor(group, "weight", name)?,
            stride: ints(group, "stride", name)?,
            padding: read_padding(group, name)?,
            dilation: ints(group, "dilation", name)?,
            groups: int_scalar(group, "groups", name)?,
            bias: tensor(group, "bias", name)?,
            input_shape: opt_usizes(group, "input_shape", name)?,
            metadata,
        }),
        "CubaLI" => {
            let v_leak = tensor(group, "v_leak", name)?;
            NirNode::CubaLi(CubaLi {
                tau_syn: tensor(group, "tau_syn", name)?,
                tau_mem: tensor(group, "tau_mem", name)?,
                r: tensor(group, "r", name)?,
                w_in: Some(opt_tensor(group, "w_in", name)?.unwrap_or_else(|| v_leak.ones_like())),
                v_leak,
                metadata,
            })
        }
        "CubaLIF" => {
            let v_leak = tensor(group, "v_leak", name)?;
            let v_threshold = tensor(group, "v_threshold", name)?;
            NirNode::CubaLif(CubaLif {
                tau_syn: tensor(group, "tau_syn", name)?,
                tau_mem: tensor(group, "tau_mem", name)?,
                r: tensor(group, "r", name)?,
                v_reset: Some(
                    opt_tensor(group, "v_reset", name)?.unwrap_or_else(|| v_threshold.zeros_like()),
                ),
                w_in: Some(opt_tensor(group, "w_in", name)?.unwrap_or_else(|| v_leak.ones_like())),
                v_leak,
                v_threshold,
                metadata,
            })
        }
        "Delay" => NirNode::Delay(Delay {
            delay: tensor(group, "delay", name)?,
            metadata,
        }),
        "Flatten" => NirNode::Flatten(Flatten {
            // Upstream's dataclass defaults are start_dim = 1, end_dim = -1.
            start_dim: opt_int_scalar(group, "start_dim", name)?.unwrap_or(1),
            end_dim: opt_int_scalar(group, "end_dim", name)?.unwrap_or(-1),
            input_type: opt_usizes(group, "input_type", name)?,
            metadata,
        }),
        "I" => NirNode::I(I {
            r: tensor(group, "r", name)?,
            metadata,
        }),
        "IF" => {
            let v_threshold = tensor(group, "v_threshold", name)?;
            NirNode::If(If {
                r: tensor(group, "r", name)?,
                v_reset: Some(
                    opt_tensor(group, "v_reset", name)?.unwrap_or_else(|| v_threshold.zeros_like()),
                ),
                v_threshold,
                metadata,
            })
        }
        "LI" => NirNode::Li(Li {
            tau: tensor(group, "tau", name)?,
            r: tensor(group, "r", name)?,
            v_leak: tensor(group, "v_leak", name)?,
            metadata,
        }),
        "LIF" => {
            let v_threshold = tensor(group, "v_threshold", name)?;
            NirNode::Lif(Lif {
                tau: tensor(group, "tau", name)?,
                r: tensor(group, "r", name)?,
                v_leak: tensor(group, "v_leak", name)?,
                v_reset: Some(
                    opt_tensor(group, "v_reset", name)?.unwrap_or_else(|| v_threshold.zeros_like()),
                ),
                v_threshold,
                metadata,
            })
        }
        "SumPool2d" => NirNode::SumPool2d(SumPool2d {
            kernel_size: tensor(group, "kernel_size", name)?,
            stride: tensor(group, "stride", name)?,
            padding: tensor(group, "padding", name)?,
            metadata,
        }),
        "AvgPool2d" => NirNode::AvgPool2d(AvgPool2d {
            kernel_size: tensor(group, "kernel_size", name)?,
            stride: tensor(group, "stride", name)?,
            padding: tensor(group, "padding", name)?,
            metadata,
        }),
        "Threshold" => NirNode::Threshold(Threshold {
            threshold: tensor(group, "threshold", name)?,
            metadata,
        }),
        "NIRGraph" => {
            let mut sub = read_graph_body(group)?;
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

/// `padding` is an integer extent list, or the string `"same"` / `"valid"`.
fn read_padding(group: &Group, node: &str) -> Result<Padding> {
    let ds = required(group, "padding", node)?;
    if is_string(&ds)? {
        wire::padding_from_wire_str(&read_string_scalar(&ds, node)?)
    } else {
        Ok(Padding::Explicit(read_ints(&ds, node, "padding")?))
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

fn read_metadata(group: &Group) -> Result<MetadataMap> {
    let Ok(md) = group.group(KEY_METADATA) else {
        return Ok(MetadataMap::new());
    };
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
    let scalar = ds.shape().is_empty();
    let value = match ds.dtype()?.to_descriptor()? {
        Td::VarLenUnicode | Td::VarLenAscii | Td::FixedAscii(_) | Td::FixedUnicode(_) => {
            MetadataValue::String(read_string_scalar(ds, key)?)
        }
        Td::Boolean if scalar => MetadataValue::Bool(ds.read_scalar::<bool>()?),
        Td::Float(_) if scalar => MetadataValue::F64(ds.read_scalar::<f64>()?),
        Td::Integer(_) | Td::Unsigned(_) if scalar => MetadataValue::I64(ds.read_scalar::<i64>()?),
        _ => MetadataValue::Tensor(read_tensor(ds, "metadata", key)?),
    };
    Ok(value)
}

// ---------------------------------------------------------------------------
// Dataset accessors
// ---------------------------------------------------------------------------

fn required(group: &Group, field: &str, node: &str) -> Result<Dataset> {
    group
        .dataset(field)
        .map_err(|_| NirError::MissingField(format!("{node}.{field}")))
}

fn tensor(group: &Group, field: &str, node: &str) -> Result<Tensor> {
    read_tensor(&required(group, field, node)?, node, field)
}

fn opt_tensor(group: &Group, field: &str, node: &str) -> Result<Option<Tensor>> {
    match group.dataset(field) {
        Ok(ds) => read_tensor(&ds, node, field).map(Some),
        Err(_) => Ok(None),
    }
}

fn ints(group: &Group, field: &str, node: &str) -> Result<Vec<i64>> {
    read_ints(&required(group, field, node)?, node, field)
}

fn int_scalar(group: &Group, field: &str, node: &str) -> Result<i64> {
    let values = ints(group, field, node)?;
    single_int(values, node, field)
}

fn opt_int_scalar(group: &Group, field: &str, node: &str) -> Result<Option<i64>> {
    match group.dataset(field) {
        Ok(ds) => single_int(read_ints(&ds, node, field)?, node, field).map(Some),
        Err(_) => Ok(None),
    }
}

fn usizes(group: &Group, field: &str, node: &str) -> Result<Vec<usize>> {
    to_usizes(ints(group, field, node)?, node, field)
}

fn opt_usizes(group: &Group, field: &str, node: &str) -> Result<Option<Vec<usize>>> {
    match group.dataset(field) {
        Ok(ds) => to_usizes(read_ints(&ds, node, field)?, node, field).map(Some),
        Err(_) => Ok(None),
    }
}

fn to_usizes(values: Vec<i64>, node: &str, field: &str) -> Result<Vec<usize>> {
    values
        .into_iter()
        .map(|v| {
            usize::try_from(v).map_err(|_| {
                NirError::InvalidTensor(format!("{node}.{field}: negative axis length {v}"))
            })
        })
        .collect()
}

fn single_int(values: Vec<i64>, node: &str, field: &str) -> Result<i64> {
    match values.as_slice() {
        [only] => Ok(*only),
        other => Err(NirError::InvalidTensor(format!(
            "{node}.{field}: expected a single integer, found {} values",
            other.len()
        ))),
    }
}

fn single_dim(values: Vec<usize>, node: &str, field: &str) -> Result<usize> {
    match values.as_slice() {
        [only] => Ok(*only),
        other => Err(NirError::InvalidTensor(format!(
            "{node}.{field}: expected a single extent, found {} values",
            other.len()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Typed dataset decoding
// ---------------------------------------------------------------------------

/// Decode a numeric dataset, preserving its on-disk element type.
///
/// Narrower integers widen into [`DType::I64`](crate::DType::I64); floats keep
/// their width so an `f32` file never silently becomes `f64` (or worse, the
/// reverse). Anything else is rejected rather than guessed at.
fn read_tensor(ds: &Dataset, node: &str, field: &str) -> Result<Tensor> {
    let descriptor = ds.dtype()?.to_descriptor()?;
    let data = match descriptor {
        Td::Float(FloatSize::U4) => TensorData::F32(ds.read_raw::<f32>()?),
        Td::Float(FloatSize::U8) => TensorData::F64(ds.read_raw::<f64>()?),
        Td::Integer(_) | Td::Unsigned(IntSize::U1 | IntSize::U2 | IntSize::U4) => {
            TensorData::I64(ds.read_raw::<i64>()?)
        }
        Td::Unsigned(IntSize::U8) => {
            let raw = ds.read_raw::<u64>()?;
            let mut values = Vec::with_capacity(raw.len());
            for v in raw {
                values.push(i64::try_from(v).map_err(|_| {
                    NirError::InvalidTensor(format!(
                        "{node}.{field}: u64 value {v} does not fit in i64"
                    ))
                })?);
            }
            TensorData::I64(values)
        }
        Td::Boolean => TensorData::Bool(ds.read_raw::<bool>()?),
        other => {
            return Err(NirError::InvalidTensor(format!(
                "{node}.{field}: element type {other} has no NIR dtype"
            )));
        }
    };
    Tensor::new(ds.shape(), data)
}

fn read_ints(ds: &Dataset, node: &str, field: &str) -> Result<Vec<i64>> {
    match read_tensor(ds, node, field)?.data() {
        TensorData::I64(values) => Ok(values.clone()),
        other => Err(NirError::InvalidTensor(format!(
            "{node}.{field}: expected integer data, found {:?}",
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
