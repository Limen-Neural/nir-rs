// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-node port shapes and the sliding-window / flatten arithmetic behind
//! them. Graph traversal lives in the parent module.

use super::{NodeTypes, Shape};
use crate::nodes::{Conv1d, Conv2d, NirNode, Padding};
use crate::types::{Tensor, TensorData};

/// Types a node knows from its own fields (Python `__post_init__`).
///
/// Pooling is inferred from its producer, and nested graphs are handled by
/// the caller (they need the recursive result), so both are undefined here.
pub(super) fn declared_types(node: &NirNode) -> std::result::Result<NodeTypes, String> {
    if let Some(shape) = parameter_shape(node) {
        return Ok(NodeTypes::leaf(Some(shape.to_vec()), Some(shape.to_vec())));
    }
    match node {
        NirNode::Affine(n) => matrix_types(&n.weight),
        NirNode::Linear(n) => matrix_types(&n.weight),
        _ => declared_input_types(node).unwrap_or_else(|| Ok(NodeTypes::undefined())),
    }
}

/// `Conv*` / `Flatten` ports, when the node declares its own input shape.
fn declared_input_types(node: &NirNode) -> Option<std::result::Result<NodeTypes, String>> {
    match node {
        NirNode::Conv1d(n) => n.input_shape.map(|len| ConvParams::conv1d(n).types(&[len])),
        NirNode::Conv2d(n) => n
            .input_shape
            .as_deref()
            .map(|spatial| ConvParams::conv2d(n).types(spatial)),
        NirNode::Flatten(n) => n
            .input_type
            .as_deref()
            .map(|shape| flatten_types(shape, n.start_dim, n.end_dim)),
        _ => None,
    }
}

/// Shape shared by every port of a node, taken from one field.
fn parameter_shape(node: &NirNode) -> Option<&[usize]> {
    match node {
        NirNode::Input(n) => Some(n.shape.as_slice()),
        NirNode::Output(n) => Some(n.shape.as_slice()),
        NirNode::Scale(n) => Some(n.scale.shape()),
        NirNode::Delay(n) => Some(n.delay.shape()),
        NirNode::Threshold(n) => Some(n.threshold.shape()),
        _ => neuron_shape(node),
    }
}

/// Port shape of a neuron node.
fn neuron_shape(node: &NirNode) -> Option<&[usize]> {
    match node {
        NirNode::CubaLi(n) => Some(n.v_leak.shape()),
        NirNode::CubaLif(n) => Some(n.v_threshold.shape()),
        NirNode::I(n) => Some(n.r.shape()),
        NirNode::If(n) => Some(n.r.shape()),
        NirNode::Li(n) => Some(n.r.shape()),
        NirNode::Lif(n) => Some(n.r.shape()),
        _ => None,
    }
}

/// Output shape for nodes whose output follows from their (now known) input.
/// `Ok(None)` for nodes that cannot be inferred this way.
pub(super) fn infer_output(
    node: &NirNode,
    input: &[Option<Shape>],
) -> std::result::Result<Option<Shape>, String> {
    let Some(derived) = Derived::of(node) else {
        return Ok(None);
    };
    match input {
        [Some(shape)] => derived.output(shape).map(Some),
        _ => Err("input type is undefined".to_owned()),
    }
}

/// How a node's output shape follows from its input shape.
enum Derived<'a> {
    Conv(ConvParams<'a>),
    Pool {
        kernel: &'a Tensor,
        stride: &'a Tensor,
        padding: &'a Tensor,
    },
    Flatten {
        start_dim: i64,
        end_dim: i64,
    },
}

impl<'a> Derived<'a> {
    fn of(node: &'a NirNode) -> Option<Self> {
        Some(match node {
            NirNode::Conv1d(n) => Self::Conv(ConvParams::conv1d(n)),
            NirNode::Conv2d(n) => Self::Conv(ConvParams::conv2d(n)),
            NirNode::SumPool2d(n) => Self::Pool {
                kernel: &n.kernel_size,
                stride: &n.stride,
                padding: &n.padding,
            },
            NirNode::AvgPool2d(n) => Self::Pool {
                kernel: &n.kernel_size,
                stride: &n.stride,
                padding: &n.padding,
            },
            NirNode::Flatten(n) => Self::Flatten {
                start_dim: n.start_dim,
                end_dim: n.end_dim,
            },
            _ => return None,
        })
    }

    fn output(&self, input: &[usize]) -> std::result::Result<Shape, String> {
        match self {
            Self::Conv(conv) => conv.output(input),
            Self::Pool {
                kernel,
                stride,
                padding,
            } => pool_output(input, kernel, stride, padding),
            Self::Flatten { start_dim, end_dim } => flatten_output(input, *start_dim, *end_dim),
        }
    }
}

pub(super) fn matrix_types(weight: &Tensor) -> std::result::Result<NodeTypes, String> {
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

/// Convolution fields that fix its shapes.
struct ConvParams<'a> {
    weight: &'a Tensor,
    /// Weight rank: 3 for `Conv1d`, 4 for `Conv2d`.
    rank: usize,
    padding: &'a Padding,
    dilation: &'a [i64],
    stride: &'a [i64],
}

impl<'a> ConvParams<'a> {
    fn conv1d(n: &'a Conv1d) -> Self {
        Self {
            weight: &n.weight,
            rank: 3,
            padding: &n.padding,
            dilation: &n.dilation,
            stride: &n.stride,
        }
    }

    fn conv2d(n: &'a Conv2d) -> Self {
        Self {
            weight: &n.weight,
            rank: 4,
            padding: &n.padding,
            dilation: &n.dilation,
            stride: &n.stride,
        }
    }

    /// Ports for a convolution whose spatial input extents are declared.
    fn types(&self, spatial: &[usize]) -> std::result::Result<NodeTypes, String> {
        let channels = *self.weight.shape().get(1).ok_or_else(|| {
            format!(
                "weight rank {} is not {}",
                self.weight.shape().len(),
                self.rank
            )
        })?;
        let mut input = vec![channels];
        input.extend_from_slice(spatial);
        let output = self.output(&input)?;
        Ok(NodeTypes::leaf(Some(input), Some(output)))
    }

    /// `[C_out, *spatial_out]` for an input of `[C_in, *spatial]`.
    fn output(&self, input: &[usize]) -> std::result::Result<Shape, String> {
        let rank = self.rank;
        let weight = self.weight.shape();
        if weight.len() != rank {
            return Err(format!("weight rank {} is not {rank}", weight.len()));
        }
        if input.len() != rank - 1 {
            return Err(format!(
                "input rank {} is not {} (channels plus {} spatial axes)",
                input.len(),
                rank - 1,
                rank - 2
            ));
        }
        let padding = match self.padding {
            Padding::Explicit(p) => Pad::Explicit(p),
            Padding::Same => Pad::Same,
            Padding::Valid => Pad::Valid,
        };
        let window = Window {
            padding,
            dilation: self.dilation,
            kernel: &weight[2..],
            stride: self.stride,
        };
        let mut out = vec![weight[0]];
        out.extend(window.output(&input[1..])?);
        Ok(out)
    }
}

#[derive(Clone, Copy)]
pub(super) enum Pad<'a> {
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

/// Sliding-window parameters shared by convolution and pooling.
pub(super) struct Window<'a> {
    pub(super) padding: Pad<'a>,
    pub(super) dilation: &'a [i64],
    pub(super) kernel: &'a [usize],
    pub(super) stride: &'a [i64],
}

impl Window<'_> {
    /// `floor((in + 2p - d(k-1) - 1) / s + 1)` per axis (PyTorch / NIR formula).
    pub(super) fn output(&self, input: &[usize]) -> std::result::Result<Shape, String> {
        let ndim = input.len();
        input
            .iter()
            .enumerate()
            .map(|(axis, &n)| self.extent(n, axis, ndim))
            .collect()
    }

    fn extent(&self, n: usize, axis: usize, ndim: usize) -> std::result::Result<usize, String> {
        let extent = match self.padding {
            Pad::Same => i128::from(n as u64),
            Pad::Valid | Pad::Explicit(_) => self.sliding_extent(n, axis, ndim)?,
        };
        usize::try_from(extent)
            .map_err(|_| format!("output extent {extent} on spatial axis {axis} is out of range"))
    }

    fn sliding_extent(
        &self,
        n: usize,
        axis: usize,
        ndim: usize,
    ) -> std::result::Result<i128, String> {
        let p = match self.padding {
            Pad::Explicit(p) => axis_value(p, axis, ndim, "padding")?,
            _ => 0,
        };
        let d = axis_value(self.dilation, axis, ndim, "dilation")?;
        if d <= 0 {
            return Err(format!("dilation must be positive, found {d}"));
        }
        let s = axis_value(self.stride, axis, ndim, "stride")?;
        if s <= 0 {
            return Err(format!("stride must be positive, found {s}"));
        }
        let k = match self.kernel.len() {
            1 => self.kernel[0],
            len if len == ndim => self.kernel[axis],
            len => return Err(format!("kernel has {len} extents for {ndim} spatial axes")),
        };
        let numerator = i128::from(n as u64) + 2 * i128::from(p)
            - i128::from(d) * (i128::from(k as u64) - 1)
            - 1;
        Ok(numerator.div_euclid(i128::from(s)) + 1)
    }
}

fn pool_extents<'a>(tensor: &'a Tensor, field: &str) -> std::result::Result<&'a [i64], String> {
    match tensor.data() {
        TensorData::I64(values) if !values.is_empty() => Ok(values),
        TensorData::I64(_) => Err(format!("{field} is empty")),
        _ => Err(format!("{field} must be an integer tensor")),
    }
}

pub(super) fn pool_output(
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
    let window = Window {
        padding: Pad::Explicit(padding),
        dilation: &[1],
        kernel: &kernel,
        stride,
    };
    out.extend(window.output(&input[1..])?);
    Ok(out)
}

/// Python slice `values[start:stop]` on a list of length `len`, as bounds.
pub(super) fn py_slice(len: usize, start: Option<i64>, stop: Option<i64>) -> (usize, usize) {
    let clamp = |i: i64| -> usize {
        let len = len as i64;
        let i = if i < 0 { i + len } else { i };
        i.clamp(0, len) as usize
    };
    let start = start.map_or(0, clamp);
    let stop = stop.map_or(len, clamp);
    (start, stop.max(start))
}

/// `Flatten` ports for a declared input shape.
fn flatten_types(
    input: &[usize],
    start_dim: i64,
    end_dim: i64,
) -> std::result::Result<NodeTypes, String> {
    let output = flatten_output(input, start_dim, end_dim)?;
    Ok(NodeTypes::leaf(Some(input.to_vec()), Some(output)))
}

/// Python `calc_flatten_output` plus its element-count assertion.
pub(super) fn flatten_output(
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
