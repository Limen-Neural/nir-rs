// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure-Rust implementation of the Neuromorphic Intermediate Representation (NIR).
//!
//! NIR is a framework-agnostic graph format for spiking neural networks
//! (analogous to ONNX for conventional nets). This crate provides a typed
//! in-memory graph model; HDF5 `.nir` read/write lands in **v0.3**.
//!
//! # Status
//!
//! **v0.2 — Core IR**: closed [`NirNode`] enum (wire-accurate type strings),
//! [`NirGraph`] with ordered nodes/edges, [`Tensor`] / metadata types, and
//! structured [`NirError`]. HDF5 I/O remains unimplemented
//! ([`io::read`] / [`io::write`] return [`NirError::Unimplemented`]).
//!
//! # Example
//!
//! ```
//! use nir_rs::nodes::{Affine, Input, Lif, Output};
//! use nir_rs::types::Tensor;
//! use nir_rs::{NirGraph, NirNode};
//!
//! let mut g = NirGraph::new();
//! g.insert_node(
//!     "input",
//!     NirNode::Input(Input {
//!         shape: vec![4],
//!         metadata: Default::default(),
//!     }),
//! )?;
//! g.insert_node(
//!     "fc",
//!     NirNode::Affine(Affine {
//!         weight: Tensor::from_f32(vec![2, 4], vec![0.1; 8])?,
//!         bias: Tensor::from_f32(vec![2], vec![0.0, 0.0])?,
//!         metadata: Default::default(),
//!     }),
//! )?;
//! g.insert_node(
//!     "lif",
//!     NirNode::Lif(Lif {
//!         tau: Tensor::from_f64(vec![2], vec![10.0, 10.0])?,
//!         r: Tensor::from_f64(vec![2], vec![1.0, 1.0])?,
//!         v_leak: Tensor::from_f64(vec![2], vec![0.0, 0.0])?,
//!         v_threshold: Tensor::from_f64(vec![2], vec![1.0, 1.0])?,
//!         v_reset: None,
//!         metadata: Default::default(),
//!     }),
//! )?;
//! g.insert_node(
//!     "output",
//!     NirNode::Output(Output {
//!         shape: vec![2],
//!         metadata: Default::default(),
//!     }),
//! )?;
//! g.add_edge("input", "fc");
//! g.add_edge("fc", "lif");
//! g.add_edge("lif", "output");
//! g.validate_structure()?;
//! # Ok::<(), nir_rs::NirError>(())
//! ```
//!
//! # Non-goals
//!
//! - SNN training or simulation
//! - FPGA / hardware mapping (see `silicon-bridge`)
//! - Framework-specific converters (live in producer/consumer crates)
//!
//! # Upstream
//!
//! - [neuromorphs/NIR](https://github.com/neuromorphs/NIR)
//! - [neuroir.org](https://neuroir.org/)
//!
//! Wire type names must match the Python IR (`CubaLIF`, `Conv2d`, …), not
//! informal marketing aliases.

#![warn(missing_docs)]

pub mod error;
pub mod graph;
pub mod io;
pub mod nodes;
pub mod types;

pub use error::{NirError, Result};
pub use graph::NirGraph;
pub use nodes::NirNode;
pub use types::{DType, MetadataMap, MetadataValue, Tensor, TensorData};
