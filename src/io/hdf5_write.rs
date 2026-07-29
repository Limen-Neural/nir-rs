// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encoding of [`NirGraph`] into HDF5 `.nir` files.
//!
//! Mirrors upstream `nir/serialization.py` (`write` / `write_recursive`) and
//! the `to_dict` of each node dataclass, so the result loads in Python
//! `nir.read`. Byte-identical output versus h5py is not a goal: group ordering,
//! chunk layout and filter parameters are HDF5 implementation details.
//!
//! Two asymmetries are wire requirements rather than oversights:
//!
//! - `Conv1d` writes `stride` / `padding` / `dilation` / `input_shape` as
//!   **scalars** while `Conv2d` writes **length-2 arrays** — upstream keeps the
//!   1-D versions as plain `int` and promotes only the 2-D ones to tuples.
//! - Fields that are `None` are **omitted** rather than written as null. That
//!   is what the Python reader expects, and `create_dataset(k, data=None)`
//!   would in fact raise on the writing side.

use super::wire::{self, KEY_EDGES, KEY_METADATA, KEY_NODE, KEY_NODES, KEY_TYPE, KEY_VERSION};
use super::{DEFAULT_NIR_VERSION, WriteOptions};
use crate::error::{NirError, Result};
use crate::graph::NirGraph;
use crate::nodes::{
    Affine, AvgPool2d, Conv1d, Conv2d, CubaLi, CubaLif, Delay, Flatten, I, If, Input, Li, Lif,
    Linear, NirNode, Output, Padding, Scale, SumPool2d, Threshold,
};
use crate::types::{MetadataMap, MetadataValue, Tensor, TensorData};
use hdf5::H5Type;
use hdf5::types::VarLenUnicode;
use hdf5::{File, Group};
use std::path::Path;
use std::str::FromStr;

/// Write `graph` to `path` by atomically replacing it after a successful flush.
pub(super) fn write(path: &Path, graph: &NirGraph, opts: &WriteOptions) -> Result<()> {
    // Validate before touching the filesystem for precise caller-facing
    // errors. Name legality is not optional — HDF5 cannot represent the
    // rejected names at all.
    // Depth first: `validate_structure` recurses through nested subgraphs
    // without a bound, so an over-deep graph would overflow the stack before
    // the guard inside `check_names` could reject it.
    check_names(graph)?;
    if opts.validate {
        graph.validate_structure()?;
        check_representable(graph)?;
    }

    let version = opts
        .version
        .clone()
        .or_else(|| graph.version.clone())
        .unwrap_or_else(|| DEFAULT_NIR_VERSION.to_owned());

    check_string_values(graph, &version)?;
    check_usize_fields(graph)?;
    check_tensor_ranks(graph)?;
    check_compression(opts.compression)?;

    write_atomically(path, graph, opts, &version, |_| Ok(()))
}

/// Stage a complete HDF5 file inside a private directory, then replace `path` in one rename.
///
/// The staging file is created inside a mode-0700 directory (Unix) to prevent
/// TOCTOU attacks. The callback exists solely to let the unit test inject an
/// error after the temporary file exists and prove cleanup/preservation behavior.
///
/// **Ownership change on Unix**: The atomic write replaces the destination inode,
/// so the new file's owner and group become those of the writing process. Mode bits
/// (permissions) are preserved when the destination exists, but ownership/ACL metadata
/// is not. A privileged writer can leave the original owner unable to update the model
/// despite keeping mode bits intact. Callers that require ownership preservation should
/// explicitly `chown` the destination after this returns, or stage and rename manually.
fn write_atomically(
    path: &Path,
    graph: &NirGraph,
    opts: &WriteOptions,
    version: &str,
    after_temp_created: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    let (temp_path, staging_dir) = temporary_path(path)?;
    after_temp_created(&temp_path)?;

    let result = (|| {
        write_file(&temp_path, graph, opts, version)?;

        match std::fs::metadata(path) {
            Ok(metadata) => {
                std::fs::set_permissions(&temp_path, metadata.permissions()).map_err(|e| {
                    NirError::Io(format!(
                        "cannot preserve permissions for {}: {e}",
                        path.display()
                    ))
                })?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(NirError::Io(format!(
                    "cannot stat {}: {e}",
                    path.display()
                )));
            }
        }

        std::fs::rename(&temp_path, path).map_err(|e| {
            NirError::Io(format!(
                "cannot atomically replace {}: {e}",
                path.display()
            ))
        })
    })();

    let _ = std::fs::remove_file(&temp_path);
    let _ = std::fs::remove_dir(&staging_dir);

    result
}

/// Create the staging file inside a private temporary directory.
///
/// Returns the staging file path and the staging directory path. The staging
/// file is created inside a directory with mode 0o700 (Unix) to prevent TOCTOU
/// attacks: between closing the temp file handle and HDF5 reopening it, a local
/// attacker with write access to the destination directory cannot swap the staging
/// path for a symlink because they cannot access the private staging directory.
fn temporary_path(path: &Path) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.nir");

    let staging_dir = {
        let mut builder = tempfile::Builder::new();
        builder.prefix(".nir_staging.");
        let dir = builder.tempdir_in(parent).map_err(|e| {
            NirError::Io(format!(
                "cannot create staging directory beside {}: {e}",
                path.display()
            ))
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700))
                .map_err(|e| {
                    NirError::Io(format!(
                        "cannot set staging directory permissions: {e}"
                    ))
                })?;
        }

        dir.into_path()
    };

    let staging_path = staging_dir.join(name);
    std::fs::File::create(&staging_path).map_err(|e| {
        let _ = std::fs::remove_dir(&staging_dir);
        NirError::Io(format!(
            "cannot create staging file {}: {e}",
            staging_path.display()
        ))
    })?;

    Ok((staging_path, staging_dir))
}

/// Encode and flush one complete HDF5 file.
fn write_file(path: &Path, graph: &NirGraph, opts: &WriteOptions, version: &str) -> Result<()> {
    let file = File::create(path)
        .map_err(|e| NirError::Io(format!("cannot create {}: {e}", path.display())))?;
    write_string(&file, KEY_VERSION, version)?;

    let root = file.create_group(KEY_NODE)?;
    write_string(&root, KEY_TYPE, "NIRGraph")?;
    write_graph_body(&Writer::new(&root, opts), graph)?;

    // HDF5 buffers metadata and raw data, so a failure while committing them —
    // a full filesystem, for instance — would otherwise surface during `Drop`,
    // where it cannot be returned. Flushing here is what makes `Ok(())` mean
    // the bytes actually reached the file.
    file.flush()
        .map_err(|e| NirError::Io(format!("cannot flush {}: {e}", path.display())))?;
    drop(file);
    Ok(())
}

/// Reject every caller-supplied string that HDF5 cannot use as a link name,
/// at any nesting depth.
///
/// This runs before staging because HDF5's link-creation error is less useful
/// than identifying the invalid caller-supplied name directly.
fn check_names(graph: &NirGraph) -> Result<()> {
    check_names_at(graph, &mut 0)
}

/// `seen` counts every graph including the root, matching the reader's bound so
/// a graph this crate writes is always one it can read back.
fn check_names_at(graph: &NirGraph, seen: &mut usize) -> Result<()> {
    *seen += 1;
    if *seen > super::hdf5_read::MAX_NESTED_GRAPHS {
        return Err(NirError::InvalidGraph(format!(
            "more than {} nested NIRGraph groups",
            super::hdf5_read::MAX_NESTED_GRAPHS
        )));
    }
    check_metadata_keys(&graph.metadata)?;
    for (name, node) in &graph.nodes {
        wire::check_link_name("node name", name)?;
        check_metadata_keys(node_metadata(node))?;
        if let NirNode::Graph(sub) = node {
            check_names_at(sub, seen)?;
        }
    }
    Ok(())
}

/// Reject a deflate level HDF5 will not accept.
///
/// [`WriteOptions::with_compression`] clamps, but `compression` is a public
/// field a caller can set directly. Without this, `H5Pset_deflate` rejects the
/// level only at dataset-creation time, after more work has already occurred.
fn check_compression(level: Option<u8>) -> Result<()> {
    match level {
        Some(level) if level > 9 => Err(NirError::InvalidGraph(format!(
            "compression level {level} is out of range (expected 0..=9)"
        ))),
        _ => Ok(()),
    }
}

fn check_metadata_keys(metadata: &MetadataMap) -> Result<()> {
    for key in metadata.keys() {
        wire::check_link_name("metadata key", key)?;
    }
    Ok(())
}

/// Reject HDF5 string payloads before staging begins.
fn check_string_values(graph: &NirGraph, version: &str) -> Result<()> {
    wire::check_hdf5_string("version", version)?;
    check_graph_string_values(graph)
}

fn check_graph_string_values(graph: &NirGraph) -> Result<()> {
    check_metadata_string_values(&graph.metadata, "graph metadata")?;
    for (src, dst) in &graph.edges {
        wire::check_hdf5_string("edge source", src)?;
        wire::check_hdf5_string("edge destination", dst)?;
    }
    for (name, node) in &graph.nodes {
        if let NirNode::Graph(sub) = node {
            check_graph_string_values(sub)?;
        } else {
            check_metadata_string_values(
                node_metadata(node),
                &format!("metadata of node {name:?}"),
            )?;
        }
    }
    Ok(())
}

fn check_metadata_string_values(metadata: &MetadataMap, context: &str) -> Result<()> {
    for (key, value) in metadata {
        match value {
            MetadataValue::String(s) => wire::check_hdf5_string(&format!("{context} {key:?}"), s)?,
            MetadataValue::StringList(v) => {
                for s in v {
                    wire::check_hdf5_string(&format!("{context} {key:?}"), s)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Reject usize fields and convolution extents that the wire cannot carry.
///
/// Everything here fails during `write_graph_body` too, but checking up front
/// produces a focused representation error before creating a staging file.
fn check_usize_fields(graph: &NirGraph) -> Result<()> {
    for (name, node) in &graph.nodes {
        match node {
            NirNode::Input(n) => check_extents(&format!("Input {name:?}"), "shape", &n.shape)?,
            NirNode::Output(n) => check_extents(&format!("Output {name:?}"), "shape", &n.shape)?,
            NirNode::Flatten(n) => check_extents(
                &format!("Flatten {name:?}"),
                "input_type",
                n.input_type.as_deref().unwrap_or(&[]),
            )?,
            NirNode::Conv1d(conv) => check_conv1d_extents(&format!("Conv1d {name:?}"), conv)?,
            NirNode::Conv2d(conv) => check_conv2d_extents(&format!("Conv2d {name:?}"), conv)?,
            NirNode::Graph(sub) => check_usize_fields(sub)?,
            _ => {}
        }
    }
    Ok(())
}

/// Range-check every extent in one `usize`-backed wire field.
fn check_extents(who: &str, field: &str, extents: &[usize]) -> Result<()> {
    for extent in extents {
        check_extent_range(who, field, *extent)?;
    }
    Ok(())
}

/// `Conv1d` extents are bare scalars on the wire.
fn check_conv1d_extents(who: &str, conv: &Conv1d) -> Result<()> {
    check_extent_arity(who, "stride", conv.stride.len(), &[1])?;
    check_extent_arity(who, "dilation", conv.dilation.len(), &[1])?;
    if let Padding::Explicit(extents) = &conv.padding {
        check_extent_arity(who, "padding", extents.len(), &[1])?;
    }
    match conv.input_shape {
        Some(extent) => check_extent_range(who, "input_shape", extent),
        None => Ok(()),
    }
}

/// `Conv2d` extents are pairs; a single value is the scalar form and is
/// expanded to a pair by the writer.
fn check_conv2d_extents(who: &str, conv: &Conv2d) -> Result<()> {
    check_extent_arity(who, "stride", conv.stride.len(), &[1, 2])?;
    check_extent_arity(who, "dilation", conv.dilation.len(), &[1, 2])?;
    if let Padding::Explicit(extents) = &conv.padding {
        check_extent_arity(who, "padding", extents.len(), &[1, 2])?;
    }
    let Some(shape) = &conv.input_shape else {
        return Ok(());
    };
    check_extent_arity(who, "input_shape", shape.len(), &[2])?;
    for extent in shape {
        check_extent_range(who, "input_shape", *extent)?;
    }
    Ok(())
}

fn check_extent_arity(who: &str, field: &str, found: usize, allowed: &[usize]) -> Result<()> {
    if allowed.contains(&found) {
        return Ok(());
    }
    let expected = match allowed {
        [1] => "exactly one extent",
        [2] => "a (N_x, N_y) pair",
        _ => "one or two extents",
    };
    Err(NirError::InvalidGraph(format!(
        "{who} {field} must hold {expected}, found {found} values"
    )))
}

/// `usize` is wider than the `i64` the wire uses on 64-bit targets.
fn check_extent_range(who: &str, field: &str, extent: usize) -> Result<()> {
    if i64::try_from(extent).is_err() {
        return Err(NirError::InvalidTensor(format!(
            "{who} {field}: extent {extent} does not fit in i64"
        )));
    }
    Ok(())
}

/// Reject tensors whose rank exceeds HDF5's 32-dimension limit.
fn check_tensor_ranks(graph: &NirGraph) -> Result<()> {
    check_metadata_tensor_ranks(&graph.metadata, "graph metadata")?;
    for (name, node) in &graph.nodes {
        let node_context = format!("node {name:?}");
        if let NirNode::Graph(sub) = node {
            check_tensor_ranks(sub)?;
        } else {
            check_node_tensor_ranks(node, &node_context)?;
            check_metadata_tensor_ranks(node_metadata(node), &node_context)?;
        }
    }
    Ok(())
}

/// Every `Tensor` field of one non-graph node, named against `src/nodes.rs`.
fn check_node_tensor_ranks(node: &NirNode, node_context: &str) -> Result<()> {
    match node {
        NirNode::Affine(n) => {
            check_tensor_rank(&n.weight, node_context, "weight")?;
            check_tensor_rank(&n.bias, node_context, "bias")?;
        }
        NirNode::Linear(n) => check_tensor_rank(&n.weight, node_context, "weight")?,
        NirNode::Scale(n) => check_tensor_rank(&n.scale, node_context, "scale")?,
        NirNode::Conv1d(n) => {
            check_tensor_rank(&n.weight, node_context, "weight")?;
            check_tensor_rank(&n.bias, node_context, "bias")?;
        }
        NirNode::Conv2d(n) => {
            check_tensor_rank(&n.weight, node_context, "weight")?;
            check_tensor_rank(&n.bias, node_context, "bias")?;
        }
        NirNode::CubaLi(n) => check_cuba_li_ranks(n, node_context)?,
        NirNode::CubaLif(n) => check_cuba_lif_ranks(n, node_context)?,
        NirNode::Delay(n) => check_tensor_rank(&n.delay, node_context, "delay")?,
        NirNode::I(n) => check_tensor_rank(&n.r, node_context, "r")?,
        NirNode::If(n) => {
            check_tensor_rank(&n.r, node_context, "r")?;
            check_tensor_rank(&n.v_threshold, node_context, "v_threshold")?;
            check_opt_tensor_rank(n.v_reset.as_ref(), node_context, "v_reset")?;
        }
        NirNode::Li(n) => {
            check_tensor_rank(&n.tau, node_context, "tau")?;
            check_tensor_rank(&n.r, node_context, "r")?;
            check_tensor_rank(&n.v_leak, node_context, "v_leak")?;
        }
        // `Lif` has no `w_in` — that field is CubaLI/CubaLIF only.
        NirNode::Lif(n) => {
            check_tensor_rank(&n.tau, node_context, "tau")?;
            check_tensor_rank(&n.r, node_context, "r")?;
            check_tensor_rank(&n.v_leak, node_context, "v_leak")?;
            check_tensor_rank(&n.v_threshold, node_context, "v_threshold")?;
            check_opt_tensor_rank(n.v_reset.as_ref(), node_context, "v_reset")?;
        }
        NirNode::SumPool2d(n) => {
            check_pool_window_ranks(&n.kernel_size, &n.stride, &n.padding, node_context)?;
        }
        NirNode::AvgPool2d(n) => {
            check_pool_window_ranks(&n.kernel_size, &n.stride, &n.padding, node_context)?;
        }
        NirNode::Threshold(n) => check_tensor_rank(&n.threshold, node_context, "threshold")?,
        // Listed rather than swept into `_` so a new variant with tensor
        // fields fails to compile here instead of silently skipping the
        // check. `shape` / `input_type` are `Vec<usize>`, not tensors, and
        // are range-checked by `check_usize_fields`. `Graph` is handled by
        // the caller, which recurses.
        NirNode::Input(_) | NirNode::Output(_) | NirNode::Flatten(_) | NirNode::Graph(_) => {}
    }
    Ok(())
}

/// `CubaLI` carries the synaptic/membrane pair plus an optional input weight.
fn check_cuba_li_ranks(n: &CubaLi, ctx: &str) -> Result<()> {
    check_tensor_rank(&n.tau_syn, ctx, "tau_syn")?;
    check_tensor_rank(&n.tau_mem, ctx, "tau_mem")?;
    check_tensor_rank(&n.r, ctx, "r")?;
    check_tensor_rank(&n.v_leak, ctx, "v_leak")?;
    check_opt_tensor_rank(n.w_in.as_ref(), ctx, "w_in")
}

/// `CubaLIF` is `CubaLI` plus the firing threshold and its reset.
fn check_cuba_lif_ranks(n: &CubaLif, ctx: &str) -> Result<()> {
    check_tensor_rank(&n.tau_syn, ctx, "tau_syn")?;
    check_tensor_rank(&n.tau_mem, ctx, "tau_mem")?;
    check_tensor_rank(&n.r, ctx, "r")?;
    check_tensor_rank(&n.v_leak, ctx, "v_leak")?;
    check_tensor_rank(&n.v_threshold, ctx, "v_threshold")?;
    check_opt_tensor_rank(n.v_reset.as_ref(), ctx, "v_reset")?;
    check_opt_tensor_rank(n.w_in.as_ref(), ctx, "w_in")
}

/// `SumPool2d` and `AvgPool2d` share one window field set.
fn check_pool_window_ranks(
    kernel_size: &Tensor,
    stride: &Tensor,
    padding: &Tensor,
    ctx: &str,
) -> Result<()> {
    check_tensor_rank(kernel_size, ctx, "kernel_size")?;
    check_tensor_rank(stride, ctx, "stride")?;
    check_tensor_rank(padding, ctx, "padding")
}

fn check_opt_tensor_rank(tensor: Option<&Tensor>, context: &str, field: &str) -> Result<()> {
    match tensor {
        Some(t) => check_tensor_rank(t, context, field),
        None => Ok(()),
    }
}

fn check_tensor_rank(tensor: &Tensor, context: &str, field: &str) -> Result<()> {
    let rank = tensor.shape().len();
    if rank > 32 {
        return Err(NirError::InvalidTensor(format!(
            "{context} {field}: rank {rank} exceeds HDF5 limit of 32"
        )));
    }
    Ok(())
}

fn check_metadata_tensor_ranks(metadata: &MetadataMap, context: &str) -> Result<()> {
    for (key, value) in metadata {
        if let MetadataValue::Tensor(t) = value {
            check_tensor_rank(t, context, &format!("metadata.{key}"))?;
        }
    }
    Ok(())
}

/// Reject in-memory values the wire format cannot carry back unchanged.
///
/// Both cases below would otherwise make `read(write(g)) != g` for the
/// caller's own graph, silently. Callers who want the value dropped anyway can
/// turn the check off with [`WriteOptions::with_validation`].
fn check_representable(graph: &NirGraph) -> Result<()> {
    check_metadata_values(&graph.metadata, "graph metadata")?;
    for (name, node) in &graph.nodes {
        check_metadata_values(node_metadata(node), &format!("metadata of node {name:?}"))?;
        if let NirNode::Graph(sub) = node {
            // The wire format has exactly one `/version`, at the root, so a
            // version on a nested graph has nowhere to go.
            if sub.version.is_some() {
                return Err(NirError::InvalidGraph(format!(
                    "subgraph {name:?} carries a version, but the wire format has \
                     only the root /version; clear it or set it on the root graph"
                )));
            }
            check_representable(sub)?;
        }
    }
    Ok(())
}

/// A rank-0 metadata tensor is indistinguishable on the wire from a plain
/// [`MetadataValue::F64`] / `I64` / `Bool`, so it would decode as the scalar
/// variant instead of a tensor.
fn check_metadata_values(metadata: &MetadataMap, context: &str) -> Result<()> {
    for (key, value) in metadata {
        if let MetadataValue::Tensor(t) = value
            && t.shape().is_empty()
        {
            return Err(NirError::InvalidGraph(format!(
                "{context}: {key:?} is a rank-0 tensor, which the wire format cannot \
                 tell apart from a scalar; use MetadataValue::F64/I64/Bool instead"
            )));
        }
    }
    Ok(())
}

/// The metadata map of any node, including a nested graph's own.
fn node_metadata(node: &NirNode) -> &MetadataMap {
    match node {
        NirNode::Input(n) => &n.metadata,
        NirNode::Output(n) => &n.metadata,
        NirNode::Affine(n) => &n.metadata,
        NirNode::Linear(n) => &n.metadata,
        NirNode::Scale(n) => &n.metadata,
        NirNode::Conv1d(n) => &n.metadata,
        NirNode::Conv2d(n) => &n.metadata,
        NirNode::CubaLi(n) => &n.metadata,
        NirNode::CubaLif(n) => &n.metadata,
        NirNode::Delay(n) => &n.metadata,
        NirNode::Flatten(n) => &n.metadata,
        NirNode::I(n) => &n.metadata,
        NirNode::If(n) => &n.metadata,
        NirNode::Li(n) => &n.metadata,
        NirNode::Lif(n) => &n.metadata,
        NirNode::SumPool2d(n) => &n.metadata,
        NirNode::AvgPool2d(n) => &n.metadata,
        NirNode::Threshold(n) => &n.metadata,
        NirNode::Graph(sub) => &sub.metadata,
    }
}

// ---------------------------------------------------------------------------
// Graph structure
// ---------------------------------------------------------------------------

/// Write a graph's `nodes`, `edges` and `metadata` into `w`'s group.
///
/// The `type` dataset is the caller's responsibility: for a nested graph it has
/// already been written by [`write_node`], and writing it twice into the same
/// group is an HDF5 error.
fn write_graph_body(w: &Writer, graph: &NirGraph) -> Result<()> {
    let nodes = w.group.create_group(KEY_NODES)?;
    for (name, node) in &graph.nodes {
        let node_group = nodes.create_group(name)?;
        write_node(&w.rebind(&node_group), node)?;
    }

    write_edges(w.group, &graph.edges)?;
    write_metadata(w, &graph.metadata)
}

/// Always written, even when empty: upstream `NIRGraph.from_dict` asserts the
/// key is present.
fn write_edges(group: &Group, edges: &[(String, String)]) -> Result<()> {
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for (src, dst) in edges {
        flat.push(var_str(src)?);
        flat.push(var_str(dst)?);
    }
    let ds = group
        .new_dataset::<VarLenUnicode>()
        .shape([edges.len(), 2])
        .create(KEY_EDGES)?;
    ds.write_raw(&flat)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Node dispatch
// ---------------------------------------------------------------------------

fn write_node(w: &Writer, node: &NirNode) -> Result<()> {
    write_string(w.group, KEY_TYPE, node.type_name())?;

    // A nested graph is a node group that also carries nodes and edges; its
    // metadata is written by `write_graph_body`.
    if let NirNode::Graph(sub) = node {
        return write_graph_body(w, sub);
    }

    match node {
        NirNode::Input(n) => write_input(w, n)?,
        NirNode::Output(n) => write_output(w, n)?,
        NirNode::Affine(n) => write_affine(w, n)?,
        NirNode::Linear(n) => write_linear(w, n)?,
        NirNode::Scale(n) => write_scale(w, n)?,
        NirNode::Conv1d(n) => write_conv1d(w, n)?,
        NirNode::Conv2d(n) => write_conv2d(w, n)?,
        NirNode::CubaLi(n) => write_cuba_li(w, n)?,
        NirNode::CubaLif(n) => write_cuba_lif(w, n)?,
        NirNode::Delay(n) => write_delay(w, n)?,
        NirNode::Flatten(n) => write_flatten(w, n)?,
        NirNode::I(n) => write_i(w, n)?,
        NirNode::If(n) => write_if(w, n)?,
        NirNode::Li(n) => write_li(w, n)?,
        NirNode::Lif(n) => write_lif(w, n)?,
        NirNode::SumPool2d(n) => write_sum_pool2d(w, n)?,
        NirNode::AvgPool2d(n) => write_avg_pool2d(w, n)?,
        NirNode::Threshold(n) => write_threshold(w, n)?,
        NirNode::Graph(_) => unreachable!("handled above"),
    }

    write_metadata(w, node_metadata(node))
}

// ---------------------------------------------------------------------------
// Ports and linear maps
// ---------------------------------------------------------------------------

fn write_input(w: &Writer, node: &Input) -> Result<()> {
    w.usizes("shape", &node.shape)
}

fn write_output(w: &Writer, node: &Output) -> Result<()> {
    w.usizes("shape", &node.shape)
}

fn write_affine(w: &Writer, node: &Affine) -> Result<()> {
    w.tensor("weight", &node.weight)?;
    w.tensor("bias", &node.bias)
}

fn write_linear(w: &Writer, node: &Linear) -> Result<()> {
    w.tensor("weight", &node.weight)
}

fn write_scale(w: &Writer, node: &Scale) -> Result<()> {
    w.tensor("scale", &node.scale)
}

// ---------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------

fn write_conv1d(w: &Writer, node: &Conv1d) -> Result<()> {
    w.tensor("weight", &node.weight)?;
    w.conv_extent("stride", &node.stride, Rank::One)?;
    w.padding(&node.padding, Rank::One)?;
    w.conv_extent("dilation", &node.dilation, Rank::One)?;
    w.scalar("groups", node.groups)?;
    w.tensor("bias", &node.bias)?;
    if let Some(extent) = node.input_shape {
        w.scalar("input_shape", to_i64(extent, "input_shape")?)?;
    }
    Ok(())
}

fn write_conv2d(w: &Writer, node: &Conv2d) -> Result<()> {
    w.tensor("weight", &node.weight)?;
    w.conv_extent("stride", &node.stride, Rank::Two)?;
    w.padding(&node.padding, Rank::Two)?;
    w.conv_extent("dilation", &node.dilation, Rank::Two)?;
    w.scalar("groups", node.groups)?;
    w.tensor("bias", &node.bias)?;
    if let Some(shape) = &node.input_shape {
        w.usizes("input_shape", shape)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Neuron models
// ---------------------------------------------------------------------------

fn write_cuba_li(w: &Writer, node: &CubaLi) -> Result<()> {
    w.tensor("tau_syn", &node.tau_syn)?;
    w.tensor("tau_mem", &node.tau_mem)?;
    w.tensor("r", &node.r)?;
    w.tensor("v_leak", &node.v_leak)?;
    w.opt_tensor("w_in", node.w_in.as_ref())
}

fn write_cuba_lif(w: &Writer, node: &CubaLif) -> Result<()> {
    w.tensor("tau_syn", &node.tau_syn)?;
    w.tensor("tau_mem", &node.tau_mem)?;
    w.tensor("r", &node.r)?;
    w.tensor("v_leak", &node.v_leak)?;
    w.tensor("v_threshold", &node.v_threshold)?;
    w.opt_tensor("v_reset", node.v_reset.as_ref())?;
    w.opt_tensor("w_in", node.w_in.as_ref())
}

fn write_i(w: &Writer, node: &I) -> Result<()> {
    w.tensor("r", &node.r)
}

fn write_if(w: &Writer, node: &If) -> Result<()> {
    w.tensor("r", &node.r)?;
    w.tensor("v_threshold", &node.v_threshold)?;
    w.opt_tensor("v_reset", node.v_reset.as_ref())
}

fn write_li(w: &Writer, node: &Li) -> Result<()> {
    w.tensor("tau", &node.tau)?;
    w.tensor("r", &node.r)?;
    w.tensor("v_leak", &node.v_leak)
}

fn write_lif(w: &Writer, node: &Lif) -> Result<()> {
    w.tensor("tau", &node.tau)?;
    w.tensor("r", &node.r)?;
    w.tensor("v_leak", &node.v_leak)?;
    w.tensor("v_threshold", &node.v_threshold)?;
    w.opt_tensor("v_reset", node.v_reset.as_ref())
}

// ---------------------------------------------------------------------------
// Pooling and the remaining leaf nodes
// ---------------------------------------------------------------------------

/// `SumPool2d` and `AvgPool2d` carry an identical field set.
fn write_pool_window(
    w: &Writer,
    kernel_size: &Tensor,
    stride: &Tensor,
    pad: &Tensor,
) -> Result<()> {
    w.tensor("kernel_size", kernel_size)?;
    w.tensor("stride", stride)?;
    w.tensor("padding", pad)
}

fn write_sum_pool2d(w: &Writer, node: &SumPool2d) -> Result<()> {
    write_pool_window(w, &node.kernel_size, &node.stride, &node.padding)
}

fn write_avg_pool2d(w: &Writer, node: &AvgPool2d) -> Result<()> {
    write_pool_window(w, &node.kernel_size, &node.stride, &node.padding)
}

fn write_delay(w: &Writer, node: &Delay) -> Result<()> {
    w.tensor("delay", &node.delay)
}

fn write_flatten(w: &Writer, node: &Flatten) -> Result<()> {
    w.scalar("start_dim", node.start_dim)?;
    w.scalar("end_dim", node.end_dim)?;
    // Flatten stores its input shape under the key `input_type`.
    match &node.input_type {
        Some(shape) => w.usizes("input_type", shape),
        None => Ok(()),
    }
}

fn write_threshold(w: &Writer, node: &Threshold) -> Result<()> {
    w.tensor("threshold", &node.threshold)
}

/// Omitted entirely when empty, matching upstream's `if not v == {}` guard.
fn write_metadata(w: &Writer, metadata: &MetadataMap) -> Result<()> {
    if metadata.is_empty() {
        return Ok(());
    }
    let group = w.group.create_group(KEY_METADATA)?;
    let md = w.rebind(&group);
    for (key, value) in metadata {
        match value {
            MetadataValue::String(s) => write_string(md.group, key, s)?,
            MetadataValue::StringList(v) => write_string_list(md.group, key, v)?,
            MetadataValue::F64(v) => md.scalar(key, *v)?,
            MetadataValue::I64(v) => md.scalar(key, *v)?,
            MetadataValue::Bool(v) => md.scalar(key, *v)?,
            MetadataValue::Tensor(t) => md.tensor(key, t)?,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dataset writers
// ---------------------------------------------------------------------------

/// Which convolution an extent belongs to, and therefore how it is shaped.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rank {
    /// `Conv1d`: a bare scalar on the wire.
    One,
    /// `Conv2d`: a length-2 array on the wire.
    Two,
}

impl Rank {
    fn node_type(self) -> &'static str {
        match self {
            Self::One => "Conv1d",
            Self::Two => "Conv2d",
        }
    }

    fn expected_extents(self) -> &'static str {
        match self {
            Self::One => "exactly one extent",
            Self::Two => "one or two extents",
        }
    }
}

/// A destination group paired with the options that govern how datasets in it
/// are stored, so field writers need only a name and a value.
struct Writer<'a> {
    group: &'a Group,
    opts: &'a WriteOptions,
}

impl<'a> Writer<'a> {
    fn new(group: &'a Group, opts: &'a WriteOptions) -> Self {
        Self { group, opts }
    }

    /// The same options aimed at a different group.
    fn rebind<'b>(&self, group: &'b Group) -> Writer<'b>
    where
        'a: 'b,
    {
        Writer {
            group,
            opts: self.opts,
        }
    }

    fn tensor(&self, name: &str, tensor: &Tensor) -> Result<()> {
        let shape = tensor.shape();
        match tensor.data() {
            TensorData::F32(v) => self.array(name, shape, v),
            TensorData::F64(v) => self.array(name, shape, v),
            TensorData::I64(v) => self.array(name, shape, v),
            TensorData::Bool(v) => self.array(name, shape, v),
        }
    }

    fn opt_tensor(&self, name: &str, tensor: Option<&Tensor>) -> Result<()> {
        match tensor {
            Some(t) => self.tensor(name, t),
            None => Ok(()),
        }
    }

    fn usizes(&self, name: &str, values: &[usize]) -> Result<()> {
        let converted: Vec<i64> = values
            .iter()
            .map(|&v| to_i64(v, name))
            .collect::<Result<_>>()?;
        self.array(name, &[converted.len()], &converted)
    }

    /// Write a convolution extent in the shape its rank requires.
    ///
    /// `Conv1d` extents are bare scalars upstream. `Conv2d` extents are always
    /// length-2 tuples: Python's `__post_init__` promotes a scalar `s` to
    /// `(s, s)`, so a single value here is the scalar form and is expanded the
    /// same way rather than written as a length-1 array Python never produces.
    fn conv_extent(&self, name: &str, values: &[i64], rank: Rank) -> Result<()> {
        match (rank, values) {
            (Rank::One, [only]) => self.scalar(name, *only),
            (Rank::Two, &[only]) => self.array(name, &[2], &[only, only]),
            (Rank::Two, [_, _]) => self.array(name, &[values.len()], values),
            (rank, other) => Err(NirError::InvalidGraph(format!(
                "{} {name} must hold {}, found {} values",
                rank.node_type(),
                rank.expected_extents(),
                other.len()
            ))),
        }
    }

    fn padding(&self, padding: &Padding, rank: Rank) -> Result<()> {
        match wire::padding_as_wire_str(padding) {
            Some(mode) => write_string(self.group, "padding", mode),
            None => {
                let Padding::Explicit(extents) = padding else {
                    unreachable!("padding_as_wire_str returns None only for Explicit");
                };
                self.conv_extent("padding", extents, rank)
            }
        }
    }

    /// Write an n-dimensional dataset, compressing only when it can be chunked.
    ///
    /// HDF5 requires a chunked layout for any filter, and scalar dataspaces
    /// cannot be chunked — so rank-0 datasets are always stored contiguously
    /// regardless of [`WriteOptions::compression`].
    fn array<T: H5Type>(&self, name: &str, shape: &[usize], data: &[T]) -> Result<()> {
        let mut builder = self.group.new_dataset::<T>();
        let compressible = !shape.is_empty() && !data.is_empty();
        if let Some(level) = self.opts.compression
            && compressible
        {
            builder = builder.deflate(level);
        }
        let ds = builder.shape(shape).create(name)?;

        if shape.is_empty() {
            // `Tensor` guarantees `shape product == data.len()`, so an empty
            // shape means exactly one element; `first` keeps that an error
            // rather than a panic if a future caller bypasses the invariant.
            let value = data.first().ok_or_else(|| {
                NirError::InvalidTensor(format!("{name}: scalar dataset needs one element, got 0"))
            })?;
            ds.write_scalar(value)?;
        } else {
            ds.write_raw(data)?;
        }
        Ok(())
    }

    fn scalar<T: H5Type>(&self, name: &str, value: T) -> Result<()> {
        let ds = self.group.new_dataset::<T>().shape(()).create(name)?;
        ds.write_scalar(&value)?;
        Ok(())
    }
}

fn to_i64(value: usize, field: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        NirError::InvalidTensor(format!("{field}: axis length {value} does not fit in i64"))
    })
}

fn write_string(group: &Group, name: &str, value: &str) -> Result<()> {
    let ds = group
        .new_dataset::<VarLenUnicode>()
        .shape(())
        .create(name)?;
    ds.write_scalar(&var_str(value)?)?;
    Ok(())
}

/// Rank-1 variable-length string dataset, matching what h5py emits for a
/// Python `list[str]`. Uncompressed like the other metadata writers.
fn write_string_list(group: &Group, name: &str, values: &[String]) -> Result<()> {
    let encoded = values
        .iter()
        .map(|s| var_str(s))
        .collect::<Result<Vec<_>>>()?;
    let ds = group
        .new_dataset::<VarLenUnicode>()
        .shape([encoded.len()])
        .create(name)?;
    ds.write_raw(&encoded)?;
    Ok(())
}

fn var_str(value: &str) -> Result<VarLenUnicode> {
    VarLenUnicode::from_str(value).map_err(|e| {
        NirError::Io(format!(
            "{value:?} cannot be encoded as an HDF5 string: {e}"
        ))
    })
}

#[cfg(test)]
mod atomic_tests {
    use super::*;
    use tempfile::TempDir;

    fn residue(dir: &Path) -> Vec<String> {
        let mut names: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn injected_failure_preserves_existing_file_and_cleans_temp() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.nir");
        let original = b"existing model bytes";
        std::fs::write(&path, original).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        }

        let graph = NirGraph::new();
        let before = residue(dir.path());
        let err = write_atomically(
            &path,
            &graph,
            &WriteOptions::default(),
            DEFAULT_NIR_VERSION,
            |_| Err(NirError::Io("injected failure after temp creation".into())),
        )
        .unwrap_err();

        assert!(err.to_string().contains("injected failure"));
        assert_eq!(std::fs::read(&path).unwrap(), original);
        assert_eq!(residue(dir.path()), before);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o640
            );
        }
    }

    #[test]
    fn successful_atomic_write_replaces_and_preserves_permissions() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.nir");
        std::fs::write(&path, b"old bytes").unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o604)).unwrap();
        }

        write(&path, &NirGraph::new(), &WriteOptions::default()).unwrap();
        let decoded =
            super::super::hdf5_read::read(&path, &super::super::ReadOptions::default()).unwrap();
        assert!(decoded.nodes.is_empty());
        assert!(decoded.edges.is_empty());
        assert_eq!(decoded.version.as_deref(), Some(DEFAULT_NIR_VERSION));
        assert_eq!(residue(dir.path()), vec!["model.nir"]);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o604
            );
        }
    }

    #[test]
    #[cfg(unix)]
    fn write_succeeds_when_destination_lacks_write_permission() {
        use std::os::unix::fs::PermissionsExt;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("readonly.nir");
        std::fs::write(&path, b"placeholder").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();

        write(&path, &NirGraph::new(), &WriteOptions::default()).unwrap();

        let decoded =
            super::super::hdf5_read::read(&path, &super::super::ReadOptions::default()).unwrap();
        assert!(decoded.nodes.is_empty());
        assert!(decoded.edges.is_empty());
        assert_eq!(decoded.version.as_deref(), Some(DEFAULT_NIR_VERSION));

        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o444
        );
        assert_eq!(residue(dir.path()), vec!["readonly.nir"]);
    }
}
