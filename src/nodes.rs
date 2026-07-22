// SPDX-License-Identifier: MIT OR Apache-2.0

//! Wire-accurate NIR computational node types.
//!
//! The closed [`NirNode`] enum mirrors upstream HDF5 `type` strings exactly
//! (`CubaLIF`, `Conv2d`, `SumPool2d`, `I`, …). Do **not** invent marketing
//! aliases (`CurrLIF`, `Convolution`, `Integrator`) — those break
//! interoperability with Python NIR.
//!
//! Field names use snake_case matching the neuromorphs/NIR Python dataclasses.
//! Numeric array parameters are [`Tensor`] values (Python: `numpy.ndarray`).

use crate::graph::NirGraph;
use crate::types::{MetadataMap, Tensor};

/// Convolution / pooling padding specification.
///
/// Upstream NIR accepts integer extents or the string modes `"same"` / `"valid"`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Padding {
    /// Explicit per-axis padding extents (length 1 for 1d, 2 for 2d, …).
    Explicit(Vec<i64>),
    /// Pad so that spatial output size matches input (`"same"`).
    Same,
    /// No padding (`"valid"`).
    Valid,
}

impl Padding {
    /// Single-axis explicit padding.
    #[must_use]
    pub fn single(value: i64) -> Self {
        Self::Explicit(vec![value])
    }

    /// Two-axis explicit padding `(h, w)`.
    #[must_use]
    pub fn pair(h: i64, w: i64) -> Self {
        Self::Explicit(vec![h, w])
    }
}

/// Closed set of NIR computational nodes.
///
/// Exhaustive matching is intentional for silicon-bridge and other consumers.
/// This enum is **not** `#[non_exhaustive]` so downstream mappers can cover all
/// wire types without a wildcard arm (new wire types are a major API change).
#[derive(Debug, Clone, PartialEq)]
pub enum NirNode {
    /// Graph input port (`type = "Input"`).
    Input(Input),
    /// Graph output port (`type = "Output"`).
    Output(Output),
    /// Affine transform `y = W x + b` (`type = "Affine"`).
    Affine(Affine),
    /// Linear transform without bias (`type = "Linear"`).
    Linear(Linear),
    /// Elementwise scale (`type = "Scale"`).
    Scale(Scale),
    /// 1-D convolution (`type = "Conv1d"`).
    Conv1d(Conv1d),
    /// 2-D convolution (`type = "Conv2d"`).
    Conv2d(Conv2d),
    /// Current-based leaky integrator (`type = "CubaLI"`).
    CubaLi(CubaLi),
    /// Current-based LIF (`type = "CubaLIF"`).
    CubaLif(CubaLif),
    /// Pure delay (`type = "Delay"`).
    Delay(Delay),
    /// Flatten (`type = "Flatten"`).
    Flatten(Flatten),
    /// Integrator (`type = "I"`).
    I(I),
    /// Integrate-and-fire (`type = "IF"`).
    If(If),
    /// Leaky integrator (`type = "LI"`).
    Li(Li),
    /// Leaky integrate-and-fire (`type = "LIF"`).
    Lif(Lif),
    /// Sum pooling 2-D (`type = "SumPool2d"`).
    SumPool2d(SumPool2d),
    /// Average pooling 2-D (`type = "AvgPool2d"`).
    AvgPool2d(AvgPool2d),
    /// Heaviside threshold (`type = "Threshold"`).
    Threshold(Threshold),
    /// Nested subgraph (`type = "NIRGraph"`).
    Graph(Box<NirGraph>),
}

impl NirNode {
    /// Exact upstream HDF5 / Python wire `type` string.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Input(_) => "Input",
            Self::Output(_) => "Output",
            Self::Affine(_) => "Affine",
            Self::Linear(_) => "Linear",
            Self::Scale(_) => "Scale",
            Self::Conv1d(_) => "Conv1d",
            Self::Conv2d(_) => "Conv2d",
            Self::CubaLi(_) => "CubaLI",
            Self::CubaLif(_) => "CubaLIF",
            Self::Delay(_) => "Delay",
            Self::Flatten(_) => "Flatten",
            Self::I(_) => "I",
            Self::If(_) => "IF",
            Self::Li(_) => "LI",
            Self::Lif(_) => "LIF",
            Self::SumPool2d(_) => "SumPool2d",
            Self::AvgPool2d(_) => "AvgPool2d",
            Self::Threshold(_) => "Threshold",
            Self::Graph(_) => "NIRGraph",
        }
    }
}

/// Input port: virtual node feeding data into the graph.
///
/// Wire field: `shape` (array of axis lengths).
#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    /// Shape of the input tensor.
    pub shape: Vec<usize>,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Output port: virtual node collecting graph results.
///
/// Wire field: `shape` (array of axis lengths).
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    /// Shape of the output tensor.
    pub shape: Vec<usize>,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Affine map `y = W x + b`.
#[derive(Debug, Clone, PartialEq)]
pub struct Affine {
    /// Weight matrix / tensor.
    pub weight: Tensor,
    /// Bias vector / tensor.
    pub bias: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Linear map without bias `y = W x`.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    /// Weight matrix / tensor.
    pub weight: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Elementwise scale `y = x ⊙ s`.
#[derive(Debug, Clone, PartialEq)]
pub struct Scale {
    /// Per-element scale factors.
    pub scale: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// 1-D convolution.
#[derive(Debug, Clone, PartialEq)]
pub struct Conv1d {
    /// Kernel weights, typically `(C_out, C_in, K)`.
    pub weight: Tensor,
    /// Stride (scalar or length-1).
    pub stride: Vec<i64>,
    /// Padding specification.
    pub padding: Padding,
    /// Dilation (scalar or length-1).
    pub dilation: Vec<i64>,
    /// Grouped convolution groups.
    pub groups: i64,
    /// Bias of shape `(C_out,)` (required on the NIR wire).
    pub bias: Tensor,
    /// Optional spatial input length `N` used for shape inference.
    pub input_shape: Option<usize>,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// 2-D convolution.
#[derive(Debug, Clone, PartialEq)]
pub struct Conv2d {
    /// Kernel weights, typically `(C_out, C_in, Kh, Kw)`.
    pub weight: Tensor,
    /// Stride per spatial axis (or single value expanded later).
    pub stride: Vec<i64>,
    /// Padding specification.
    pub padding: Padding,
    /// Dilation per spatial axis.
    pub dilation: Vec<i64>,
    /// Grouped convolution groups.
    pub groups: i64,
    /// Bias of shape `(C_out,)` (required on the NIR wire).
    pub bias: Tensor,
    /// Optional spatial input `(N_x, N_y)`.
    pub input_shape: Option<Vec<usize>>,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Current-based leaky integrator (`CubaLI`).
#[derive(Debug, Clone, PartialEq)]
pub struct CubaLi {
    /// Synaptic time constant.
    pub tau_syn: Tensor,
    /// Membrane time constant.
    pub tau_mem: Tensor,
    /// Resistance.
    pub r: Tensor,
    /// Leak voltage.
    pub v_leak: Tensor,
    /// Input current weight (elementwise); defaults to ones in Python.
    pub w_in: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Current-based leaky integrate-and-fire (`CubaLIF`).
#[derive(Debug, Clone, PartialEq)]
pub struct CubaLif {
    /// Synaptic time constant.
    pub tau_syn: Tensor,
    /// Membrane time constant.
    pub tau_mem: Tensor,
    /// Resistance.
    pub r: Tensor,
    /// Leak voltage.
    pub v_leak: Tensor,
    /// Firing threshold.
    pub v_threshold: Tensor,
    /// Reset potential (optional; Python defaults to zeros).
    pub v_reset: Option<Tensor>,
    /// Input current weight (elementwise).
    pub w_in: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Pure delay `y(t) = x(t − τ)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Delay {
    /// Delay amount(s).
    pub delay: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Flatten a contiguous range of dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct Flatten {
    /// First dimension to flatten (Python default: 1).
    pub start_dim: i64,
    /// Last dimension to flatten (Python default: −1).
    pub end_dim: i64,
    /// Optional input shape used for shape inference / wire `input_type`.
    pub input_type: Option<Vec<usize>>,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Integrator neuron (`I`): `dv/dt = R I`.
#[derive(Debug, Clone, PartialEq)]
pub struct I {
    /// Resistance.
    pub r: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Integrate-and-fire neuron (`IF`).
#[derive(Debug, Clone, PartialEq)]
pub struct If {
    /// Resistance.
    pub r: Tensor,
    /// Firing threshold.
    pub v_threshold: Tensor,
    /// Reset potential (optional).
    pub v_reset: Option<Tensor>,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Leaky integrator (`LI`).
#[derive(Debug, Clone, PartialEq)]
pub struct Li {
    /// Membrane time constant.
    pub tau: Tensor,
    /// Resistance.
    pub r: Tensor,
    /// Leak voltage.
    pub v_leak: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Leaky integrate-and-fire (`LIF`).
#[derive(Debug, Clone, PartialEq)]
pub struct Lif {
    /// Membrane time constant.
    pub tau: Tensor,
    /// Resistance.
    pub r: Tensor,
    /// Leak voltage.
    pub v_leak: Tensor,
    /// Firing threshold.
    pub v_threshold: Tensor,
    /// Reset potential (optional).
    pub v_reset: Option<Tensor>,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// 2-D sum pooling.
#[derive(Debug, Clone, PartialEq)]
pub struct SumPool2d {
    /// Kernel size `(H, W)`.
    pub kernel_size: Tensor,
    /// Stride `(H, W)`.
    pub stride: Tensor,
    /// Padding `(H, W)`.
    pub padding: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// 2-D average pooling.
#[derive(Debug, Clone, PartialEq)]
pub struct AvgPool2d {
    /// Kernel size `(H, W)`.
    pub kernel_size: Tensor,
    /// Stride `(H, W)`.
    pub stride: Tensor,
    /// Padding `(H, W)`.
    pub padding: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

/// Heaviside threshold / surrogate step.
#[derive(Debug, Clone, PartialEq)]
pub struct Threshold {
    /// Threshold value(s).
    pub threshold: Tensor,
    /// Free-form node metadata (Python `metadata` dict).
    pub metadata: MetadataMap,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Tensor;

    fn sample_weight() -> Tensor {
        Tensor::from_f32(vec![2, 3], vec![1., 0., 0., 0., 1., 0.]).unwrap()
    }

    fn sample_bias() -> Tensor {
        Tensor::from_f32(vec![2], vec![0., 0.]).unwrap()
    }

    fn sample_vec3() -> Tensor {
        Tensor::from_f64(vec![3], vec![1.0, 1.0, 1.0]).unwrap()
    }

    #[test]
    fn all_type_names_match_wire_strings() {
        let cases: Vec<(&str, NirNode)> = vec![
            (
                "Input",
                NirNode::Input(Input {
                    shape: vec![1, 4],
                    metadata: Default::default(),
                }),
            ),
            (
                "Output",
                NirNode::Output(Output {
                    shape: vec![1, 2],
                    metadata: Default::default(),
                }),
            ),
            (
                "Affine",
                NirNode::Affine(Affine {
                    weight: sample_weight(),
                    bias: sample_bias(),
                    metadata: Default::default(),
                }),
            ),
            (
                "Linear",
                NirNode::Linear(Linear {
                    weight: sample_weight(),
                    metadata: Default::default(),
                }),
            ),
            (
                "Scale",
                NirNode::Scale(Scale {
                    scale: sample_vec3(),
                    metadata: Default::default(),
                }),
            ),
            (
                "Conv1d",
                NirNode::Conv1d(Conv1d {
                    weight: Tensor::from_f32(vec![1, 1, 3], vec![1., 0., -1.]).unwrap(),
                    stride: vec![1],
                    padding: Padding::single(0),
                    dilation: vec![1],
                    groups: 1,
                    bias: Tensor::from_f32(vec![1], vec![0.]).unwrap(),
                    input_shape: Some(10),
                    metadata: Default::default(),
                }),
            ),
            (
                "Conv2d",
                NirNode::Conv2d(Conv2d {
                    weight: Tensor::from_f32(vec![1, 1, 3, 3], vec![0.; 9]).unwrap(),
                    stride: vec![1, 1],
                    padding: Padding::Same,
                    dilation: vec![1, 1],
                    groups: 1,
                    bias: Tensor::from_f32(vec![1], vec![0.]).unwrap(),
                    input_shape: Some(vec![28, 28]),
                    metadata: Default::default(),
                }),
            ),
            (
                "CubaLI",
                NirNode::CubaLi(CubaLi {
                    tau_syn: sample_vec3(),
                    tau_mem: sample_vec3(),
                    r: sample_vec3(),
                    v_leak: Tensor::from_f64(vec![3], vec![0., 0., 0.]).unwrap(),
                    w_in: Tensor::from_f64(vec![3], vec![1., 1., 1.]).unwrap(),
                    metadata: Default::default(),
                }),
            ),
            (
                "CubaLIF",
                NirNode::CubaLif(CubaLif {
                    tau_syn: sample_vec3(),
                    tau_mem: sample_vec3(),
                    r: sample_vec3(),
                    v_leak: Tensor::from_f64(vec![3], vec![0., 0., 0.]).unwrap(),
                    v_threshold: Tensor::from_f64(vec![3], vec![1., 1., 1.]).unwrap(),
                    v_reset: None,
                    w_in: Tensor::from_f64(vec![3], vec![1., 1., 1.]).unwrap(),
                    metadata: Default::default(),
                }),
            ),
            (
                "Delay",
                NirNode::Delay(Delay {
                    delay: Tensor::scalar_f64(1.0),
                    metadata: Default::default(),
                }),
            ),
            (
                "Flatten",
                NirNode::Flatten(Flatten {
                    start_dim: 1,
                    end_dim: -1,
                    input_type: Some(vec![1, 4, 4]),
                    metadata: Default::default(),
                }),
            ),
            (
                "I",
                NirNode::I(I {
                    r: sample_vec3(),
                    metadata: Default::default(),
                }),
            ),
            (
                "IF",
                NirNode::If(If {
                    r: sample_vec3(),
                    v_threshold: Tensor::from_f64(vec![3], vec![1., 1., 1.]).unwrap(),
                    v_reset: None,
                    metadata: Default::default(),
                }),
            ),
            (
                "LI",
                NirNode::Li(Li {
                    tau: sample_vec3(),
                    r: sample_vec3(),
                    v_leak: Tensor::from_f64(vec![3], vec![0., 0., 0.]).unwrap(),
                    metadata: Default::default(),
                }),
            ),
            (
                "LIF",
                NirNode::Lif(Lif {
                    tau: sample_vec3(),
                    r: sample_vec3(),
                    v_leak: Tensor::from_f64(vec![3], vec![0., 0., 0.]).unwrap(),
                    v_threshold: Tensor::from_f64(vec![3], vec![1., 1., 1.]).unwrap(),
                    v_reset: Some(Tensor::from_f64(vec![3], vec![0., 0., 0.]).unwrap()),
                    metadata: Default::default(),
                }),
            ),
            (
                "SumPool2d",
                NirNode::SumPool2d(SumPool2d {
                    kernel_size: Tensor::from_i64(vec![2], vec![2, 2]).unwrap(),
                    stride: Tensor::from_i64(vec![2], vec![2, 2]).unwrap(),
                    padding: Tensor::from_i64(vec![2], vec![0, 0]).unwrap(),
                    metadata: Default::default(),
                }),
            ),
            (
                "AvgPool2d",
                NirNode::AvgPool2d(AvgPool2d {
                    kernel_size: Tensor::from_i64(vec![2], vec![2, 2]).unwrap(),
                    stride: Tensor::from_i64(vec![2], vec![2, 2]).unwrap(),
                    padding: Tensor::from_i64(vec![2], vec![0, 0]).unwrap(),
                    metadata: Default::default(),
                }),
            ),
            (
                "Threshold",
                NirNode::Threshold(Threshold {
                    threshold: Tensor::scalar_f64(1.0),
                    metadata: Default::default(),
                }),
            ),
            ("NIRGraph", NirNode::Graph(Box::new(NirGraph::new()))),
        ];

        assert_eq!(cases.len(), 19, "expected all wire node types");
        for (wire, node) in cases {
            assert_eq!(node.type_name(), wire);
        }
    }

    #[test]
    fn padding_helpers() {
        assert_eq!(Padding::single(1), Padding::Explicit(vec![1]));
        assert_eq!(Padding::pair(1, 2), Padding::Explicit(vec![1, 2]));
        let _ = Padding::Same;
        let _ = Padding::Valid;
    }

    #[test]
    fn never_use_marketing_aliases() {
        // Guard against accidental CurrLIF / Convolution / Integrator names.
        let names: Vec<&str> = [
            NirNode::CubaLif(CubaLif {
                tau_syn: Tensor::scalar_f64(1.0),
                tau_mem: Tensor::scalar_f64(1.0),
                r: Tensor::scalar_f64(1.0),
                v_leak: Tensor::scalar_f64(0.0),
                v_threshold: Tensor::scalar_f64(1.0),
                v_reset: None,
                w_in: Tensor::scalar_f64(1.0),
                metadata: Default::default(),
            }),
            NirNode::Conv2d(Conv2d {
                weight: Tensor::from_f32(vec![1, 1, 1, 1], vec![1.]).unwrap(),
                stride: vec![1, 1],
                padding: Padding::Valid,
                dilation: vec![1, 1],
                groups: 1,
                bias: Tensor::from_f32(vec![1], vec![0.]).unwrap(),
                input_shape: None,
                metadata: Default::default(),
            }),
            NirNode::I(I {
                r: Tensor::scalar_f64(1.0),
                metadata: Default::default(),
            }),
            NirNode::SumPool2d(SumPool2d {
                kernel_size: Tensor::from_i64(vec![2], vec![2, 2]).unwrap(),
                stride: Tensor::from_i64(vec![2], vec![2, 2]).unwrap(),
                padding: Tensor::from_i64(vec![2], vec![0, 0]).unwrap(),
                metadata: Default::default(),
            }),
        ]
        .into_iter()
        .map(|n| n.type_name())
        .collect();

        assert_eq!(names, ["CubaLIF", "Conv2d", "I", "SumPool2d"]);
        for n in names {
            assert!(!n.contains("Curr"));
            assert!(!n.contains("Convolution"));
            assert!(!n.contains("Integrator"));
            assert!(!n.contains("Pooling"));
        }
    }
}
