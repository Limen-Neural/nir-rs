// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure-Rust implementation of the Neuromorphic Intermediate Representation (NIR).
//!
//! NIR is a framework-agnostic graph format for spiking neural networks
//! (analogous to ONNX for conventional nets). This crate will provide a typed
//! graph model and HDF5 `.nir` read/write.
//!
//! # Status
//!
//! **v0.1 scaffold** — dual license, module layout, CI. Typed IR (v0.2) and
//! HDF5 I/O (v0.3) are not implemented yet.
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

pub use error::{NirError, Result};
pub use graph::NirGraph;
pub use nodes::NodePlaceholder;
