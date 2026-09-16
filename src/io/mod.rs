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
//! nir-rs = { version = "0.4", features = ["hdf5"] }
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
//! never validates it on read. Default [`read`] follows suit:
//!
//! - [`read`] stores `/version` in [`NirGraph::version`], and leaves it `None`
//!   when the dataset is absent. It is never an error.
//! - [`read_version`] is the strict accessor and *does* error when absent.
//! - [`write()`] emits [`NirGraph::version`] when set, else
//!   [`DEFAULT_NIR_VERSION`].
//!
//! Production importers can opt into an envelope check via
//! [`ReadOptions::version_policy`] without changing that default:
//! [`VersionPolicy::RequirePresent`] rejects a missing `/version`;
//! [`VersionPolicy::CompatibleMajor`] parses a SemVer-compatible string and
//! accepts caller-supplied majors (typically `0` for paper fixtures and `1`
//! for current writers). Policy failures use
//! [`NirError::IncompatibleVersion`] and run **before** the graph body is
//! decoded.
//!
//! # Non-goals
//!
//! Byte-identical output versus h5py (group ordering, chunk layout and filter
//! parameters may differ), and the separate `NIRGraphData` observables layout
//! that upstream `read_data` / `write_data` handle.

pub mod wire;

mod version;
pub use version::VersionPolicy;

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
    use super::{NirError, NirGraph, Path, ReadOptions, Result, WriteOptions};

    pub(super) fn read(_path: &Path, _opts: &ReadOptions) -> Result<NirGraph> {
        Err(NirError::Unimplemented(
            "io::read (enable feature \"hdf5\")",
        ))
    }

    pub(super) fn read_version(_path: &Path, _opts: &ReadOptions) -> Result<String> {
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

/// Allocation and collection policy for decoding an untrusted `.nir` file
/// with [`read_with`].
///
/// Defaults are **permissive**: every field is `None`, so [`read`] stays
/// unbounded aside from the existing hard cap of 1024 `NIRGraph` groups
/// (root plus nested). That cap is a stack/alias safety bound, not a
/// substitute for a caller-chosen collection budget.
///
/// `max_bytes` is a **decoded-allocation budget**, not an on-disk file-size
/// limit and not a bound on the returned graph's exact resident size. Charging
/// is monotonic and conservative: temporary allocations stay charged after
/// they are released. The exact rules are:
///
/// - numeric datasets: element count times decoded width;
/// - `u64` datasets: both the temporary `Vec<u64>` and converted `Vec<i64>`;
/// - `i64` extent lists converted to `Vec<usize>` (e.g. `Input.shape`): both the
///   source `Vec<i64>` and the destination `Vec<usize>`;
/// - fixed strings: fixed-capacity HDF5 buffers, resulting [`String`] headers,
///   and the worst-case copied payload;
/// - variable-length strings: descriptor buffers, payload bytes reported by
///   `H5Dvlen_get_buf_size`, resulting [`String`] headers, and copied payload.
///   Scalar VLEN strings use the containing file size as a payload bound
///   because `H5Dvlen_get_buf_size` can abort on scalar VLEN;
/// - scalar metadata: its decoded width;
/// - missing `v_reset` and `w_in`: the synthesized tensor payload.
///
/// `max_nodes`, `max_edges`, and `max_nested_graphs` are **global count
/// budgets** for the whole file: nested subgraphs add to the same totals
/// rather than resetting per group. Counts are charged from HDF5 metadata
/// (`H5Gget_info` link counts, `edges` shape, one charge per `NIRGraph`
/// group) **before** the corresponding `Vec` / map is materialized. Hard-link
/// aliases are rejected before they can charge a second time.
///
/// All arithmetic is checked; overflow is treated as over budget. Node and
/// link names, collection bookkeeping, allocator overhead, and libhdf5's own
/// caches are not charged against `max_bytes`.
///
/// # Untrusted inputs
///
/// Conservative starting points when the file is not from a trusted
/// producer — tighten further for your threat model:
///
/// ```
/// use nir_rs::io::ReadOptions;
///
/// let opts = ReadOptions::default()
///     .with_max_bytes(Some(64 * 1024 * 1024))
///     .with_max_nodes(Some(10_000))
///     .with_max_edges(Some(50_000))
///     .with_max_nested_graphs(Some(64));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ReadOptions {
    /// Maximum total bytes charged by decoded allocations, or `None` for no
    /// allocation budget.
    pub max_bytes: Option<usize>,
    /// Maximum total nodes across root and nested graphs, or `None` for no
    /// node-count budget.
    pub max_nodes: Option<usize>,
    /// Maximum total edges across root and nested graphs, or `None` for no
    /// edge-count budget.
    pub max_edges: Option<usize>,
    /// Maximum total `NIRGraph` groups decoded from the file (root included),
    /// or `None` to use only the hard cap of 1024 groups.
    pub max_nested_graphs: Option<usize>,
    /// How to treat the root `/version` dataset. Defaults to
    /// [`VersionPolicy::Permissive`], which is the Python `nir.read` behaviour
    /// and keeps [`read`] byte-for-byte compatible with earlier crate
    /// versions.
    pub version_policy: VersionPolicy,
}

impl ReadOptions {
    /// Set the decoded-allocation budget in bytes; `None` makes it unbounded.
    #[must_use]
    pub fn with_max_bytes(mut self, max_bytes: Option<usize>) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Set the global node-count budget; `None` makes it unbounded.
    #[must_use]
    pub fn with_max_nodes(mut self, max_nodes: Option<usize>) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    /// Set the global edge-count budget; `None` makes it unbounded.
    #[must_use]
    pub fn with_max_edges(mut self, max_edges: Option<usize>) -> Self {
        self.max_edges = max_edges;
        self
    }

    /// Set the global nested-graph budget; `None` keeps only the hard cap of
    /// 1024 groups.
    #[must_use]
    pub fn with_max_nested_graphs(mut self, max_nested_graphs: Option<usize>) -> Self {
        self.max_nested_graphs = max_nested_graphs;
        self
    }

    /// Set the `/version` compatibility policy.
    ///
    /// ```
    /// use nir_rs::io::{ReadOptions, VersionPolicy};
    ///
    /// // Permissive tooling (default): accept missing or arbitrary versions.
    /// let tool = ReadOptions::default();
    /// assert_eq!(tool.version_policy, VersionPolicy::Permissive);
    ///
    /// // Fail-closed importer: paper 0.x fixtures and 1.x writers.
    /// let importer = ReadOptions::default()
    ///     .with_version_policy(VersionPolicy::compatible_major([0, 1]));
    /// ```
    #[must_use]
    pub fn with_version_policy(mut self, version_policy: VersionPolicy) -> Self {
        self.version_policy = version_policy;
        self
    }
}

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
    /// Deflate (gzip) level for **numeric** array datasets, must be in `0..=9`.
    ///
    /// `None` writes arrays uncompressed. Values above 9 are rejected; use
    /// [`Self::with_compression`] to clamp automatically.
    ///
    /// Three kinds of dataset are never compressed regardless of this setting:
    ///
    /// - **Scalars.** HDF5 cannot chunk them, and chunking is a prerequisite
    ///   for any filter.
    /// - **Empty arrays** — any tensor with a zero-length axis. There are no
    ///   bytes to compress, and a filter would still cost a chunked layout.
    /// - **String arrays** — `edges` and [`MetadataValue::StringList`]. These
    ///   are variable-length, so the dataset holds only heap descriptors and
    ///   the characters live on HDF5's global heap. A filter applies to the
    ///   descriptors, not to the payload, so deflating them would add chunking
    ///   overhead while compressing almost nothing.
    ///
    /// [`MetadataValue::StringList`]: crate::types::MetadataValue::StringList
    pub compression: Option<u8>,
    /// Version string to write, overriding [`NirGraph::version`].
    pub version: Option<String>,
    /// Run [`NirGraph::validate_structure`] and lossless-representation checks
    /// before writing. Defaults to `true`.
    ///
    /// Set this to `false` to rewrite a graph whose edges do not all resolve,
    /// or that contains values the wire format cannot preserve (nested graph
    /// versions, rank-0 metadata tensors). Such files exist in the wild —
    /// upstream's own `braille_noDelay_bias_zero_subgraph.nir` has a subgraph
    /// edge naming a node that is not in that subgraph — and [`read`] loads
    /// them faithfully, so writing them back has to be possible. The resulting
    /// file will not load in Python `nir.read` with its default type checking,
    /// and may lose or change the unrepresentable values on readback.
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
/// Node **parameters** keep their on-disk float width — an `f32` weight never
/// becomes `f64`. **Scalar metadata** is the one exception: [`MetadataValue`]
/// has no `F32` variant, so a scalar `float32` metadata value decodes as
/// [`MetadataValue::F64`] and is written back as a 64-bit dataset. The value
/// survives exactly, since `f32` widens to `f64` losslessly; only the wire
/// dtype of that one dataset changes. Narrower integers likewise widen into
/// [`MetadataValue::I64`].
///
/// **Node order is not preserved.** [`NirGraph::nodes`] is an order-preserving
/// map, but this reads names in sorted order so that decoding one file twice
/// gives the same order both times — HDF5 does not promise a link ordering
/// worth carrying. `edges` is a `Vec` and *is* order-significant, so it is
/// preserved exactly.
///
/// [`MetadataValue`]: crate::types::MetadataValue
/// [`MetadataValue::F64`]: crate::types::MetadataValue::F64
/// [`MetadataValue::I64`]: crate::types::MetadataValue::I64
///
/// # Errors
///
/// - [`NirError::Io`] if the file cannot be opened or is not valid HDF5
/// - [`NirError::MissingField`] if `/node` or a required node field is absent
/// - [`NirError::UnknownNodeType`] for a `type` string outside
///   [`wire::WIRE_TYPES`]
/// - [`NirError::InvalidTensor`] for a dataset whose element type has no
///   [`DType`](crate::DType) representation
/// - [`NirError::InvalidGraph`] for a file that reaches outside its own
///   container (external links, external raw storage, virtual datasets)
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
    read_with(path, &ReadOptions::default())
}

/// Read a NIR graph with an explicit decoded-allocation and collection budget.
///
/// See [`ReadOptions`] for the exact charging rules. Use this entry point for
/// untrusted files. Plain [`read`] is intentionally unbounded for trusted
/// callers and backward compatibility.
///
/// # Errors
///
/// As [`read`], plus [`NirError::ReadLimitExceeded`] when the next decoded
/// allocation would cross `opts.max_bytes`,
/// [`NirError::ReadCountLimitExceeded`] when a node, edge, or nested-graph
/// count would cross the corresponding limit, and
/// [`NirError::IncompatibleVersion`] when `opts.version_policy` rejects
/// `/version`. Version-policy checks run before the graph body is decoded.
pub fn read_with(path: impl AsRef<Path>, opts: &ReadOptions) -> Result<NirGraph> {
    backend::read(path.as_ref(), opts)
}

/// Read only the `/version` string from a `.nir` file.
///
/// # Errors
///
/// [`NirError::MissingField`] when the file has no `/version` dataset;
/// otherwise as [`read`].
pub fn read_version(path: impl AsRef<Path>) -> Result<String> {
    read_version_with(path, &ReadOptions::default())
}

/// Read only `/version` with an explicit decoded-allocation budget.
///
/// # Errors
///
/// As [`read_version`], plus [`NirError::ReadLimitExceeded`] when decoding the
/// version string would cross `opts.max_bytes`, and
/// [`NirError::IncompatibleVersion`] when `opts.version_policy` rejects the
/// value. Parsing and policy errors share the graph-reader path.
pub fn read_version_with(path: impl AsRef<Path>, opts: &ReadOptions) -> Result<String> {
    backend::read_version(path.as_ref(), opts)
}

/// Write a NIR graph to a `.nir` (HDF5) path atomically.
///
/// Equivalent to [`write_with`] using [`WriteOptions::default`] (gzip level 4,
/// matching h5py).
///
/// Data is written to a temporary file inside a private staging directory
/// (mode `0700` on Unix), flushed, closed, and then atomically renamed over the
/// destination. A failed write leaves an existing destination unchanged.
///
/// **Staging base (Unix):** when the destination parent is untrusted —
/// group/world-writable without the sticky bit, a symlink path component, or
/// owned by a UID other than the process effective UID or root — staging
/// attempts to use sticky temp (if owned by the current user or root, writable,
/// and with verified symlink-free ancestry) or a private per-user runtime/cache
/// directory (if all ancestors are owned by the current user or root, non-symlink,
/// and free of non-sticky group/world-writable modes) so other local users cannot
/// rename the staging directory away and plant a path for the HDF5 reopen.
/// Foreign-owned parents are treated as untrusted even at mode `0755`, because
/// the directory owner can always rename entries (including under a sticky bit).
/// If no safe staging base is found, the write fails rather than falling back to
/// the untrusted destination parent. The final replace into a multi-user
/// non-sticky parent still has residual rename races — prefer private destination
/// directories on shared hosts.
///
/// Existing Unix file permissions (mode bits) are preserved, but **ownership
/// and group are changed** to those of the writing process, and POSIX ACLs are
/// not preserved. A new Unix destination uses mode `0o666` filtered by the
/// process umask.
///
/// **SELinux context (Unix):** On SELinux-enforcing hosts, same-filesystem renames
/// preserve the source inode's security context. When staging under a secure base
/// such as `/tmp` and renaming onto the destination, the written file may keep
/// the staging label rather than the destination directory's file-creation
/// context, which can make it inaccessible to a confined consumer. Creating the
/// final inode under a hostile (shared/non-sticky) destination parent would
/// reintroduce path-swap races, so this residual is accepted: apply `restorecon`
/// or `chcon` after a successful write when a specific context is required.
///
/// This does not fsync the file or containing directory, so it
/// is not a power-loss durability guarantee.
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
/// Uses the same atomic staging and replacement protocol as [`write()`].
///
/// **Symlink handling**: When `path` is a symlink, the atomic rename replaces
/// the symlink itself rather than updating its target. To update the target
/// file, pass a resolved path: use [`std::fs::canonicalize`] for a fully
/// resolved absolute path, or join a relative [`std::fs::read_link`] result
/// with the symlink's parent before writing (raw `read_link` alone is not
/// enough when the stored target is relative).
///
/// **ACL preservation**: Only basic Unix permission bits (mode) are preserved
/// from an existing destination. POSIX ACLs and Windows DACLs are **not copied**
/// to the new inode. If the destination is ACL-protected, the replacement may
/// change who can access it.
///
/// **Multi-user destination directories**: Staging is hardened against parent
/// directory rename races when the destination parent is shared and non-sticky
/// (see [`write()`]). If no safe staging base can be found (sticky temp owned by
/// current user, or private per-user directories with verified ownership ancestry),
/// the write fails. Cross-device promotion is also rejected when the destination
/// parent is shared and non-sticky to prevent path-swap vulnerabilities during
/// local staging. The final `rename` into a multi-user non-sticky parent still
/// cannot be made fully race-free while HDF5 requires a path reopen; use private
/// directories when untrusted local users can write the parent.
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

    #[test]
    fn read_options_default_is_unbounded() {
        let opts = ReadOptions::default();
        assert_eq!(opts.max_bytes, None);
        assert_eq!(opts.max_nodes, None);
        assert_eq!(opts.max_edges, None);
        assert_eq!(opts.max_nested_graphs, None);
        assert_eq!(opts.version_policy, VersionPolicy::Permissive);
        let configured = opts
            .with_max_bytes(Some(4096))
            .with_max_nodes(Some(8))
            .with_max_edges(Some(16))
            .with_max_nested_graphs(Some(4));
        assert_eq!(configured.max_bytes, Some(4096));
        assert_eq!(configured.max_nodes, Some(8));
        assert_eq!(configured.max_edges, Some(16));
        assert_eq!(configured.max_nested_graphs, Some(4));
    }

    #[test]
    fn read_options_version_policy_builder() {
        let opts = ReadOptions::default()
            .with_version_policy(VersionPolicy::RequirePresent)
            .with_max_bytes(Some(1024));
        assert_eq!(opts.version_policy, VersionPolicy::RequirePresent);
        assert_eq!(opts.max_bytes, Some(1024));
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
        fn bounded_read_is_unimplemented() {
            let opts = ReadOptions::default()
                .with_max_bytes(Some(1024))
                .with_max_nodes(Some(8))
                .with_max_edges(Some(16))
                .with_max_nested_graphs(Some(4));
            assert!(matches!(
                read_with("model.nir", &opts).unwrap_err(),
                NirError::Unimplemented(_)
            ));
            assert!(matches!(
                read_version_with("model.nir", &opts).unwrap_err(),
                NirError::Unimplemented(_)
            ));
        }

        #[test]
        fn write_is_unimplemented() {
            let g = NirGraph::new();
            let err = write("out.nir", &g).unwrap_err();
            assert!(matches!(err, NirError::Unimplemented(_)));
        }
    }
}
