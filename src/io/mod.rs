// SPDX-License-Identifier: MIT OR Apache-2.0

//! HDF5 `.nir` read/write.
//!
//! `.nir` is the official NIR interchange format: an HDF5 container whose
//! layout is fixed by upstream [neuromorphs/NIR](https://github.com/neuromorphs/NIR).
//! Files written here load in Python `nir.read`, and files written by Python
//! `nir.write` load in [`read`]. See [`wire`] for the layout itself.
//!
//! # Feature gate
//!
//! The implementation lives behind the **`hdf5`** feature, which links the
//! native libhdf5 library. It is off by default so that the graph model stays
//! dependency-free for consumers that only build or inspect graphs:
//!
//! ```toml
//! nir-rs = { version = "0.3", features = ["hdf5"] }
//! ```
//!
//! System dependency: `libhdf5-dev` (Debian/Ubuntu), `hdf5` (Homebrew), or add
//! `hdf5-metno` as a direct dependency with `features = ["static", "zlib"]`
//! for a hermetic build from vendored source.
//!
//! Every item in this module exists in both builds — only the bodies are gated.
//! Without the feature, [`read`], [`write()`] and [`read_version`] return
//! [`NirError::Unimplemented`] rather than failing to compile, so downstream
//! code can be written once and feature-gated at the call site if it wants to.
//!
//! # Version string
//!
//! Upstream writes the version of the Python `nir` package into `/version` and
//! never validates it on read. This crate follows suit:
//!
//! - [`read`] stores `/version` in [`NirGraph::version`], and leaves it `None`
//!   when the dataset is absent. It is never an error.
//! - [`read_version`] is the strict accessor and *does* error when absent.
//! - [`write()`] emits [`NirGraph::version`] when set, else
//!   [`DEFAULT_NIR_VERSION`].
//!
//! # Non-goals
//!
//! Byte-identical output versus h5py (group ordering, chunk layout and filter
//! parameters may differ), and the separate `NIRGraphData` observables layout
//! that upstream `read_data` / `write_data` handle.

pub mod wire;

#[cfg(feature = "hdf5")]
mod hdf5_read;
#[cfg(feature = "hdf5")]
mod hdf5_write;

use crate::graph::NirGraph;
use std::path::Path;

// `NirError` is constructed only by the feature-off backend, but the rustdoc
// links throughout this module reference it in both builds.
#[cfg_attr(feature = "hdf5", allow(unused_imports))]
use crate::error::{NirError, Result};

/// The three primitives the public functions delegate to.
///
/// Selecting the implementation once, here, is what keeps `#[cfg]` out of the
/// public functions below — they have one body each regardless of features.
#[cfg(feature = "hdf5")]
mod backend {
    pub(super) use super::hdf5_read::{read, read_version};
    pub(super) use super::hdf5_write::write;
}

/// Stand-ins used when the `hdf5` feature is off.
///
/// The signatures match the real backend so the public API is identical in
/// both builds; only the outcome differs.
#[cfg(not(feature = "hdf5"))]
mod backend {
    use super::{NirError, NirGraph, Path, Result, WriteOptions};

    pub(super) fn read(_path: &Path) -> Result<NirGraph> {
        Err(NirError::Unimplemented(
            "io::read (enable feature \"hdf5\")",
        ))
    }

    pub(super) fn read_version(_path: &Path) -> Result<String> {
        Err(NirError::Unimplemented(
            "io::read_version (enable feature \"hdf5\")",
        ))
    }

    pub(super) fn write(_path: &Path, _graph: &NirGraph, _opts: &WriteOptions) -> Result<()> {
        Err(NirError::Unimplemented(
            "io::write (enable feature \"hdf5\")",
        ))
    }
}

/// Version string written to `/version` when a graph carries none.
///
/// Tracks the upstream `nir` release this crate's wire format is validated
/// against. Consumers that need a specific value should set
/// [`NirGraph::version`] or [`WriteOptions::with_version`].
pub const DEFAULT_NIR_VERSION: &str = "1.0.8";

/// Default gzip level, matching h5py's `compression="gzip"` default.
const DEFAULT_COMPRESSION: u8 = 4;

/// Tuning knobs for [`write_with`].
///
/// Construct from [`Default`] and adjust:
///
/// ```
/// use nir_rs::io::WriteOptions;
///
/// let opts = WriteOptions::default().with_compression(None);
/// assert_eq!(opts.compression, None);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WriteOptions {
    /// Deflate (gzip) level for array datasets, clamped to `0..=9`.
    ///
    /// `None` writes arrays uncompressed. Scalar datasets are never
    /// compressed — HDF5 cannot chunk them, and chunking is a prerequisite for
    /// any filter.
    pub compression: Option<u8>,
    /// Version string to write, overriding [`NirGraph::version`].
    pub version: Option<String>,
    /// Run [`NirGraph::validate_structure`] before writing. Defaults to `true`.
    ///
    /// Set this to `false` to rewrite a graph whose edges do not all resolve.
    /// Such files exist in the wild — upstream's own
    /// `braille_noDelay_bias_zero_subgraph.nir` has a subgraph edge naming a
    /// node that is not in that subgraph — and [`read`] loads them faithfully,
    /// so writing them back has to be possible. The resulting file will not
    /// load in Python `nir.read` with its default type checking.
    pub validate: bool,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            compression: Some(DEFAULT_COMPRESSION),
            version: None,
            validate: true,
        }
    }
}

impl WriteOptions {
    /// Set the deflate level for array datasets; `None` disables compression.
    #[must_use]
    pub fn with_compression(mut self, level: Option<u8>) -> Self {
        self.compression = level.map(|l| l.min(9));
        self
    }

    /// Override the version string written to `/version`.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Enable or disable the pre-write structure check.
    #[must_use]
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }
}

/// Read a NIR graph from a `.nir` (HDF5) path.
///
/// Absent optional fields are filled with the upstream Python defaults, so a
/// graph read here matches what `nir.read` produces in memory: a missing
/// `v_reset` becomes zeros shaped like `v_threshold`, and a missing `w_in`
/// becomes ones shaped like `v_leak`.
///
/// # Errors
///
/// - [`NirError::Io`] if the file cannot be opened or is not valid HDF5
/// - [`NirError::MissingField`] if `/node` or a required node field is absent
/// - [`NirError::UnknownNodeType`] for a `type` string outside
///   [`wire::WIRE_TYPES`]
/// - [`NirError::InvalidTensor`] for a dataset whose element type has no
///   [`DType`](crate::DType) representation
/// - [`NirError::Unimplemented`] if the `hdf5` feature is off
///
/// # Examples
///
/// ```no_run
/// let graph = nir_rs::io::read("model.nir")?;
/// for (name, node) in &graph.nodes {
///     println!("{name}: {}", node.type_name());
/// }
/// # Ok::<(), nir_rs::NirError>(())
/// ```
pub fn read(path: impl AsRef<Path>) -> Result<NirGraph> {
    backend::read(path.as_ref())
}

/// Read only the `/version` string from a `.nir` file.
///
/// # Errors
///
/// [`NirError::MissingField`] when the file has no `/version` dataset;
/// otherwise as [`read`].
pub fn read_version(path: impl AsRef<Path>) -> Result<String> {
    backend::read_version(path.as_ref())
}

/// Write a NIR graph to a `.nir` (HDF5) path, truncating any existing file.
///
/// Equivalent to [`write_with`] using [`WriteOptions::default`] (gzip level 4,
/// matching h5py).
///
/// The graph is validated with
/// [`NirGraph::validate_structure`](crate::NirGraph::validate_structure) first:
/// a graph with dangling edges would produce a file that upstream refuses to
/// load, so it is rejected here instead. Opt out with
/// [`WriteOptions::with_validation`].
///
/// # Errors
///
/// As [`write_with`].
///
/// # Examples
///
/// ```no_run
/// # let graph = nir_rs::NirGraph::new();
/// nir_rs::io::write("model.nir", &graph)?;
/// # Ok::<(), nir_rs::NirError>(())
/// ```
pub fn write(path: impl AsRef<Path>, graph: &NirGraph) -> Result<()> {
    write_with(path, graph, &WriteOptions::default())
}

/// Write a NIR graph to a `.nir` (HDF5) path with explicit options.
///
/// # Errors
///
/// - [`NirError::MissingNode`] / [`NirError::DuplicateEdge`] /
///   [`NirError::InvalidGraph`] if the graph does not validate, if a node name
///   or metadata key is not a legal HDF5 link name (see
///   [`wire::check_link_name`]), if a string payload / edge endpoint contains a
///   NUL byte, if a `Conv2d.input_shape` is not a length-2 pair, or if it holds
///   a value the wire format cannot carry back unchanged (a nested graph
///   version, or a rank-0 metadata tensor)
/// - [`NirError::Io`] if the file cannot be created or a dataset cannot be
///   written
/// - [`NirError::Unimplemented`] if the `hdf5` feature is off
pub fn write_with(path: impl AsRef<Path>, graph: &NirGraph, opts: &WriteOptions) -> Result<()> {
    backend::write(path.as_ref(), graph, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_options_default_matches_h5py_gzip() {
        let opts = WriteOptions::default();
        assert_eq!(opts.compression, Some(4));
        assert_eq!(opts.version, None);
        assert!(opts.validate);
    }

    #[test]
    fn write_options_builders() {
        let opts = WriteOptions::default()
            .with_compression(Some(200))
            .with_version("0.2.0")
            .with_validation(false);
        assert_eq!(opts.compression, Some(9), "level should clamp to 9");
        assert_eq!(opts.version.as_deref(), Some("0.2.0"));
        assert!(!opts.validate);

        let off = WriteOptions::default().with_compression(None);
        assert_eq!(off.compression, None);
    }

    #[cfg(not(feature = "hdf5"))]
    mod without_feature {
        use super::*;

        #[test]
        fn read_is_unimplemented() {
            let err = read("model.nir").unwrap_err();
            assert!(matches!(err, NirError::Unimplemented(_)));
        }

        #[test]
        fn read_version_is_unimplemented() {
            let err = read_version("model.nir").unwrap_err();
            assert!(matches!(err, NirError::Unimplemented(_)));
        }

        #[test]
        fn write_is_unimplemented() {
            let g = NirGraph::new();
            let err = write("out.nir", &g).unwrap_err();
            assert!(matches!(err, NirError::Unimplemented(_)));
        }
    }
}
