// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared value types for NIR tensors and metadata.
//!
//! Python NIR stores parameters as `numpy.ndarray`. This module provides a
//! compact, owned representation suitable for an in-memory Rust IR. HDF5
//! decoding (v0.3) will map wire arrays into [`Tensor`].

use crate::error::{NirError, Result};

/// Element type of a contiguous tensor buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DType {
    /// 32-bit IEEE floating point.
    F32,
    /// 64-bit IEEE floating point.
    F64,
    /// 64-bit signed integer.
    I64,
    /// Boolean.
    Bool,
}

impl DType {
    /// Size of one element in bytes.
    #[must_use]
    pub const fn size_of(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::I64 => 8,
            Self::Bool => 1,
        }
    }
}

/// Contiguous numeric payload backing a [`Tensor`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TensorData {
    /// `f32` elements.
    F32(Vec<f32>),
    /// `f64` elements.
    F64(Vec<f64>),
    /// `i64` elements.
    I64(Vec<i64>),
    /// `bool` elements.
    Bool(Vec<bool>),
}

impl TensorData {
    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element dtype of this payload.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
        }
    }
}

/// Dense, row-major tensor (shape + contiguous typed data).
///
/// Shape is listed outer-to-inner (C-order), matching typical NumPy layout.
///
/// Element dtype is always derived from the payload via [`Tensor::dtype`].
///
/// Shape and data are **private** so callers cannot break the
/// `shape product == data.len()` invariant after construction. Use
/// [`Tensor::shape`] / [`Tensor::data`] accessors.
///
/// # PartialEq
///
/// Equality is exact element-wise (IEEE). In particular, `NaN != NaN`, matching
/// Rust's default float `PartialEq`.
///
/// # Serde
///
/// With the `serde` feature, tensors encode as `{ "shape": ..., "data": ... }`.
/// Deserialization always calls [`Tensor::new`], preserving the private-field
/// invariant. JSON is debug-only and cannot faithfully represent non-finite
/// floats; HDF5 `.nir` remains the NIR interchange format.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Axis lengths.
    shape: Vec<usize>,
    /// Contiguous elements in C-order.
    data: TensorData,
}

#[cfg(feature = "serde")]
impl serde::Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(serde::Serialize)]
        struct TensorRef<'a> {
            shape: &'a [usize],
            data: &'a TensorData,
        }

        serde::Serialize::serialize(
            &TensorRef {
                shape: self.shape(),
                data: self.data(),
            },
            serializer,
        )
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct TensorOwned {
            shape: Vec<usize>,
            data: TensorData,
        }

        let tensor = <TensorOwned as serde::Deserialize>::deserialize(deserializer)?;
        Self::new(tensor.shape, tensor.data).map_err(serde::de::Error::custom)
    }
}

impl Tensor {
    /// Build a tensor from shape and typed data, checking length against shape.
    pub fn new(shape: impl Into<Vec<usize>>, data: TensorData) -> Result<Self> {
        let shape = shape.into();
        check_shape_len(&shape, data.len())?;
        Ok(Self { shape, data })
    }

    /// Element dtype of this tensor (derived from the payload).
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Axis lengths (outer-to-inner / C-order).
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Contiguous payload.
    #[must_use]
    pub fn data(&self) -> &TensorData {
        &self.data
    }

    /// Take the payload, consuming the tensor.
    ///
    /// Crate-internal: the HDF5 reader decodes integer wire fields through a
    /// `Tensor` and then wants the `Vec` itself. Cloning out of [`data`] would
    /// hold both buffers live at once, doubling peak memory on every `shape`,
    /// `stride` and `dilation` read for no benefit.
    ///
    /// [`data`]: Self::data
    #[must_use]
    #[cfg(feature = "hdf5")]
    pub(crate) fn into_data(self) -> TensorData {
        self.data
    }

    /// `f32` tensor; `data.len()` must equal the product of `shape`.
    pub fn from_f32(shape: impl Into<Vec<usize>>, data: impl Into<Vec<f32>>) -> Result<Self> {
        Self::new(shape, TensorData::F32(data.into()))
    }

    /// `f64` tensor; `data.len()` must equal the product of `shape`.
    pub fn from_f64(shape: impl Into<Vec<usize>>, data: impl Into<Vec<f64>>) -> Result<Self> {
        Self::new(shape, TensorData::F64(data.into()))
    }

    /// `i64` tensor; `data.len()` must equal the product of `shape`.
    pub fn from_i64(shape: impl Into<Vec<usize>>, data: impl Into<Vec<i64>>) -> Result<Self> {
        Self::new(shape, TensorData::I64(data.into()))
    }

    /// `bool` tensor; `data.len()` must equal the product of `shape`.
    pub fn from_bool(shape: impl Into<Vec<usize>>, data: impl Into<Vec<bool>>) -> Result<Self> {
        Self::new(shape, TensorData::Bool(data.into()))
    }

    /// Rank-0 (scalar) `f32` tensor.
    #[must_use]
    pub fn scalar_f32(value: f32) -> Self {
        Self {
            shape: vec![],
            data: TensorData::F32(vec![value]),
        }
    }

    /// Rank-0 (scalar) `f64` tensor.
    #[must_use]
    pub fn scalar_f64(value: f64) -> Self {
        Self {
            shape: vec![],
            data: TensorData::F64(vec![value]),
        }
    }

    /// Rank-0 (scalar) `i64` tensor.
    #[must_use]
    pub fn scalar_i64(value: i64) -> Self {
        Self {
            shape: vec![],
            data: TensorData::I64(vec![value]),
        }
    }

    /// A tensor of zeros with the same shape and dtype as `self`.
    ///
    /// Mirrors `numpy.zeros_like`, which upstream NIR uses to default absent
    /// optional wire fields (`v_reset`). [`DType::Bool`] zeroes to `false`.
    #[must_use]
    pub fn zeros_like(&self) -> Self {
        let n = self.data.len();
        let data = match self.dtype() {
            DType::F32 => TensorData::F32(vec![0.0; n]),
            DType::F64 => TensorData::F64(vec![0.0; n]),
            DType::I64 => TensorData::I64(vec![0; n]),
            DType::Bool => TensorData::Bool(vec![false; n]),
        };
        Self {
            shape: self.shape.clone(),
            data,
        }
    }

    /// A tensor of ones with the same shape and dtype as `self`.
    ///
    /// Mirrors `numpy.ones_like`, which upstream NIR uses to default an absent
    /// `w_in`. [`DType::Bool`] ones to `true`.
    #[must_use]
    pub fn ones_like(&self) -> Self {
        let n = self.data.len();
        let data = match self.dtype() {
            DType::F32 => TensorData::F32(vec![1.0; n]),
            DType::F64 => TensorData::F64(vec![1.0; n]),
            DType::I64 => TensorData::I64(vec![1; n]),
            DType::Bool => TensorData::Bool(vec![true; n]),
        };
        Self {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Number of elements implied by `shape` (empty shape → 1).
    #[must_use]
    pub fn numel(&self) -> usize {
        // Invariant: construction ensures product fits and matches data.len().
        shape_product(&self.shape).expect("tensor shape product overflow")
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// Free-form metadata map (Python NIR `metadata: Dict[str, Any]`).
pub type MetadataMap = std::collections::HashMap<String, MetadataValue>;

/// Free-form metadata values attached to graphs and nodes.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum MetadataValue {
    /// UTF-8 string.
    String(String),
    /// List of UTF-8 strings.
    ///
    /// Python's `metadata: Dict[str, Any]` admits a `list[str]`, which h5py
    /// stores as a multi-element string dataset. Without this variant such a
    /// file cannot be decoded at all, since [`Tensor`] carries only numeric
    /// and boolean payloads.
    StringList(Vec<String>),
    /// 64-bit float.
    F64(f64),
    /// 64-bit signed integer.
    I64(i64),
    /// Boolean.
    Bool(bool),
    /// Dense tensor.
    Tensor(Tensor),
}

/// Product of shape dims; empty shape is a scalar (1 element).
/// Returns `None` if the product overflows `usize`.
fn shape_product(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        Some(1)
    } else {
        shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
    }
}

fn check_shape_len(shape: &[usize], len: usize) -> Result<()> {
    let expected = shape_product(shape).ok_or_else(|| {
        NirError::InvalidTensor(format!("shape product overflows usize (shape={shape:?})"))
    })?;
    if expected != len {
        return Err(NirError::InvalidTensor(format!(
            "shape product {expected} != data len {len} (shape={shape:?})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f32_ok() {
        let t = Tensor::from_f32(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn from_f64_ok() {
        let t = Tensor::from_f64([2], vec![1.0, 2.0]).unwrap();
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(t.numel(), 2);
    }

    #[test]
    fn length_mismatch_f32() {
        let err = Tensor::from_f32(vec![2, 2], vec![1., 2., 3.]).unwrap_err();
        assert!(matches!(err, NirError::InvalidTensor(_)));
        assert!(err.to_string().contains("shape product 4 != data len 3"));
    }

    #[test]
    fn length_mismatch_f64() {
        let err = Tensor::from_f64(vec![3], vec![1.0]).unwrap_err();
        assert!(matches!(err, NirError::InvalidTensor(_)));
    }

    #[test]
    fn scalar_has_empty_shape_one_element() {
        let t = Tensor::scalar_f64(0.5);
        assert!(t.shape().is_empty());
        assert_eq!(t.numel(), 1);
        assert_eq!(t.data().len(), 1);
    }

    #[test]
    fn empty_shape_rejects_wrong_len() {
        let err = Tensor::from_f32(Vec::<usize>::new(), vec![1., 2.]).unwrap_err();
        assert!(matches!(err, NirError::InvalidTensor(_)));
    }

    #[test]
    fn i64_and_bool_constructors() {
        let i = Tensor::from_i64([2], vec![1, 2]).unwrap();
        assert_eq!(i.dtype(), DType::I64);
        let b = Tensor::from_bool([2], vec![true, false]).unwrap();
        assert_eq!(b.dtype(), DType::Bool);
    }

    #[test]
    fn dtype_size_of() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F64.size_of(), 8);
        assert_eq!(DType::I64.size_of(), 8);
        assert_eq!(DType::Bool.size_of(), 1);
    }

    #[test]
    fn shape_product_overflow_rejected() {
        let err = Tensor::from_f32(vec![usize::MAX, usize::MAX], vec![1.0]).unwrap_err();
        assert!(matches!(err, NirError::InvalidTensor(_)));
        assert!(err.to_string().contains("overflows"));
    }

    #[test]
    fn zeros_like_preserves_shape_and_dtype() {
        let t = Tensor::from_f32(vec![2, 2], vec![1., 2., 3., 4.]).unwrap();
        let z = t.zeros_like();
        assert_eq!(z.shape(), t.shape());
        assert_eq!(z.dtype(), DType::F32);
        assert_eq!(z.data(), &TensorData::F32(vec![0.0; 4]));
    }

    #[test]
    fn ones_like_preserves_shape_and_dtype() {
        let t = Tensor::from_f64(vec![3], vec![7.0, 8.0, 9.0]).unwrap();
        let o = t.ones_like();
        assert_eq!(o.shape(), [3]);
        assert_eq!(o.data(), &TensorData::F64(vec![1.0; 3]));
    }

    #[test]
    fn zeros_and_ones_like_cover_int_and_bool() {
        let i = Tensor::from_i64([2], vec![5, 6]).unwrap();
        assert_eq!(i.zeros_like().data(), &TensorData::I64(vec![0, 0]));
        assert_eq!(i.ones_like().data(), &TensorData::I64(vec![1, 1]));

        let b = Tensor::from_bool([2], vec![true, false]).unwrap();
        assert_eq!(b.zeros_like().data(), &TensorData::Bool(vec![false, false]));
        assert_eq!(b.ones_like().data(), &TensorData::Bool(vec![true, true]));
    }

    #[test]
    fn zeros_like_of_scalar_is_scalar() {
        let z = Tensor::scalar_f64(3.5).zeros_like();
        assert!(z.shape().is_empty());
        assert_eq!(z.numel(), 1);
    }

    #[test]
    fn metadata_variants() {
        let m = MetadataValue::String("note".into());
        assert!(matches!(m, MetadataValue::String(_)));
        let _ = MetadataValue::F64(1.0);
        let _ = MetadataValue::I64(2);
        let _ = MetadataValue::Bool(true);
        let _ = MetadataValue::Tensor(Tensor::scalar_f32(0.0));
    }
}
