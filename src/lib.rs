// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure-Rust implementation of the Neuromorphic Intermediate Representation (NIR).
//!
//! NIR is a framework-agnostic graph format for spiking neural networks
//! (analogous to ONNX for conventional nets). This crate provides a typed
//! in-memory graph model plus HDF5 `.nir` read/write that interoperates with
//! the Python reference implementation.
//!
//! # Status
//!
//! **v0.4 — developer experience**: a public load-inspect-save example makes
//! the HDF5 workflow executable end to end, on top of the v0.3 I/O and v0.2
//! graph model — the closed [`NirNode`] enum (wire-accurate type strings),
//! [`NirGraph`] with ordered nodes/edges, [`Tensor`] / metadata types, and
//! structured [`NirError`].
//!
//! I/O lives behind the opt-in **`hdf5`** feature because it links the native
//! libhdf5 library; the graph model itself has no system dependencies. See the
//! [`io`] module for the feature gate, the file layout, and the version-string
//! policy.
//!
//! The independent opt-in **`serde`** feature implements Serde traits for the
//! graph model. Formats such as JSON are useful for debugging and tests only:
//! they are not a stable NIR schema or an interchange format. Use HDF5 `.nir`
//! through [`io::read`] / [`io::write`] for interoperability. JSON also cannot
//! represent non-finite floats faithfully, so NaN and infinities are not
//! guaranteed to round-trip.
//!
//! ```toml
//! nir-rs = { version = "0.4", features = ["hdf5"] }
//! ```
//!
//! Run the complete fixture workflow from a checkout with:
//!
//! ```text
//! cargo run --example load_inspect_lif --features hdf5
//! ```
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
//!
//! // With `features = ["hdf5"]`, the graph exchanges as a `.nir` file:
//! // nir_rs::io::write("model.nir", &g)?;
//! // let reloaded = nir_rs::io::read("model.nir")?;
//! # Ok::<(), nir_rs::NirError>(())
//! ```
//!
//! # Non-goals
//!
//! - SNN training or simulation
//! - Mapping graphs onto specific neuromorphic hardware
//! - Framework-specific converters (belong in producer/consumer tools)
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
