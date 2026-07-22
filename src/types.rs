// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared value types for NIR tensors and metadata.
//!
//! Python NIR stores parameters as `numpy.ndarray`. This module provides a
//! compact, owned representation suitable for an in-memory Rust IR. HDF5
//! decoding (v0.3) will map wire arrays into [`Tensor`].

use crate::error::{NirError, Result};

/// Element type of a contiguous tensor buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Dense, row-major tensor (shape + dtype + contiguous data).
///
/// Shape is listed outer-to-inner (C-order), matching typical NumPy layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Axis lengths.
    pub shape: Vec<usize>,
    /// Element type (must match [`TensorData`] variant).
    pub dtype: DType,
    /// Contiguous elements in C-order.
    pub data: TensorData,
}

impl Tensor {
    /// Build a tensor from shape and typed data, checking length against shape.
    pub fn new(shape: impl Into<Vec<usize>>, data: TensorData) -> Result<Self> {
        let shape = shape.into();
        let dtype = data.dtype();
        check_shape_len(&shape, data.len())?;
        Ok(Self { shape, dtype, data })
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
    pub fn scalar_f32(value: f32) -> Self {
        Self {
            shape: vec![],
            dtype: DType::F32,
            data: TensorData::F32(vec![value]),
        }
    }

    /// Rank-0 (scalar) `f64` tensor.
    pub fn scalar_f64(value: f64) -> Self {
        Self {
            shape: vec![],
            dtype: DType::F64,
            data: TensorData::F64(vec![value]),
        }
    }

    /// Rank-0 (scalar) `i64` tensor.
    pub fn scalar_i64(value: i64) -> Self {
        Self {
            shape: vec![],
            dtype: DType::I64,
            data: TensorData::I64(vec![value]),
        }
    }

    /// Number of elements implied by `shape` (empty shape → 1).
    #[must_use]
    pub fn numel(&self) -> usize {
        shape_product(&self.shape)
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// Free-form metadata values attached to graphs (and later nodes).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MetadataValue {
    /// UTF-8 string.
    String(String),
    /// 64-bit float.
    F64(f64),
    /// 64-bit signed integer.
    I64(i64),
    /// Boolean.
    Bool(bool),
    /// Dense tensor.
    Tensor(Tensor),
}

fn shape_product(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().copied().product()
    }
}

fn check_shape_len(shape: &[usize], len: usize) -> Result<()> {
    let expected = shape_product(shape);
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
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn from_f64_ok() {
        let t = Tensor::from_f64([2], vec![1.0, 2.0]).unwrap();
        assert_eq!(t.dtype, DType::F64);
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
        assert!(t.shape.is_empty());
        assert_eq!(t.numel(), 1);
        assert_eq!(t.data.len(), 1);
    }

    #[test]
    fn empty_shape_rejects_wrong_len() {
        let err = Tensor::from_f32(Vec::<usize>::new(), vec![1., 2.]).unwrap_err();
        assert!(matches!(err, NirError::InvalidTensor(_)));
    }

    #[test]
    fn i64_and_bool_constructors() {
        let i = Tensor::from_i64([2], vec![1, 2]).unwrap();
        assert_eq!(i.dtype, DType::I64);
        let b = Tensor::from_bool([2], vec![true, false]).unwrap();
        assert_eq!(b.dtype, DType::Bool);
    }

    #[test]
    fn dtype_size_of() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F64.size_of(), 8);
        assert_eq!(DType::I64.size_of(), 8);
        assert_eq!(DType::Bool.size_of(), 1);
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
