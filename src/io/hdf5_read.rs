// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoding of HDF5 `.nir` files into [`NirGraph`].
//!
//! Mirrors upstream `nir/serialization.py` (`read` / `hdf2dict`) and the
//! `from_dict` of each node dataclass. Reading is deliberately permissive: the
//! graph is returned as the file describes it, without running
//! [`NirGraph::validate_structure`] — callers decide whether a structurally
//! odd file is a problem.
//!
//! The only normalization performed is documented on [`super::read`]: element
//! types are widened losslessly into [`DType`](crate::DType), and absent
//! optional fields are filled with the upstream Python defaults.
//!
//! [`read_node`] is a flat dispatch table over the wire `type` string; the
//! per-type functions below it own one node kind each.

use super::ReadOptions;
use super::wire::{self, KEY_EDGES, KEY_METADATA, KEY_NODE, KEY_NODES, KEY_TYPE, KEY_VERSION};
use crate::error::{NirError, Result};
use crate::graph::NirGraph;
use crate::nodes::{
    Affine, AvgPool2d, Conv1d, Conv2d, CubaLi, CubaLif, Delay, Flatten, I, If, Input, Li, Lif,
    Linear, NirNode, Output, Padding, Scale, SumPool2d, Threshold,
};
use crate::types::{MetadataMap, MetadataValue, Tensor, TensorData};
use hdf5::plist::dataset_create::Layout;
use hdf5::types::{
    FixedAscii, FixedUnicode, FloatSize, IntSize, TypeDescriptor as Td, VarLenAscii, VarLenUnicode,
};
use hdf5::{Dataset, File, Group, LocationToken};
use std::cell::Cell;
use std::path::Path;

/// Total `NIRGraph` groups decoded from one file, counting the root.
///
/// Bounds total work rather than depth, which covers both failure modes: an
/// unbounded chain would overflow the stack, and hard-link aliases could
/// re-expand a shared subtree exponentially. The writer enforces the same
/// bound, so nesting alone never makes this crate emit a file it would then
/// refuse to read.
pub(super) const MAX_NESTED_GRAPHS: usize = 1024;

/// Monotonic decoded-allocation ledger shared by one read operation.
struct ReadBudget {
    limit: Option<usize>,
    used: Cell<usize>,
}

impl ReadBudget {
    fn new(opts: &ReadOptions) -> Self {
        Self {
            limit: opts.max_bytes,
            used: Cell::new(0),
        }
    }

    /// Charge before allocation. Overflow is reported through the same
    /// structured limit error because no finite `usize` budget can admit it.
    fn charge(&self, context: &str, requested: Option<usize>) -> Result<()> {
        let Some(limit) = self.limit else {
            return Ok(());
        };
        let used = self.used.get();
        let Some(requested) = requested else {
            return Err(NirError::ReadLimitExceeded {
                context: context.to_owned(),
                limit,
                used,
                requested: usize::MAX,
            });
        };
        let Some(next) = used.checked_add(requested) else {
            return Err(NirError::ReadLimitExceeded {
                context: context.to_owned(),
                limit,
                used,
                requested,
            });
        };
        if next > limit {
            return Err(NirError::ReadLimitExceeded {
                context: context.to_owned(),
                limit,
                used,
                requested,
            });
        }
        self.used.set(next);
        Ok(())
    }

    /// Dry-run check for a charge that must fit before an expensive operation.
    fn would_fit(&self, context: &str, requested: Option<usize>) -> Result<()> {
        let Some(limit) = self.limit else {
            return Ok(());
        };
        let used = self.used.get();
        let Some(requested) = requested else {
            return Err(NirError::ReadLimitExceeded {
                context: context.to_owned(),
                limit,
                used,
                requested: usize::MAX,
            });
        };
        let Some(next) = used.checked_add(requested) else {
            return Err(NirError::ReadLimitExceeded {
                context: context.to_owned(),
                limit,
                used,
                requested,
            });
        };
        if next > limit {
            return Err(NirError::ReadLimitExceeded {
                context: context.to_owned(),
                limit,
                used,
                requested,
            });
        }
        Ok(())
    }
}

/// Requested capacities for fixed-length strings, smallest first.
const FIXED_STRING_CAPS: [usize; 3] = [64, 256, 4096];

// The ladder is spelled out again in the `read_fixed!` calls inside
// `read_strings_unchecked`, because the macro needs literals. Pin them here so
// a change to one side fails the build rather than silently mischarging the guard.
const _: () = assert!(
    FIXED_STRING_CAPS[0] == 64 && FIXED_STRING_CAPS[1] == 256 && FIXED_STRING_CAPS[2] == 4096,
    "FIXED_STRING_CAPS and the read_fixed! rungs in read_strings_unchecked must match"
);

/// Read a whole `.nir` file.
pub(super) fn read(path: &Path, opts: &ReadOptions) -> Result<NirGraph> {
    let budget = ReadBudget::new(opts);
    let file = open(path)?;
    // Check the file root before following `/node` or `/version`: opening an
    // external link is what leaves the container, so the check has to happen
    // on the link, not on the group it resolves to.
    validate_group_links(&file, "/")?;
    let root = file.group(KEY_NODE).map_err(|_| {
        NirError::MissingField(format!(
            "/{KEY_NODE} (not a NIR graph file: {})",
            path.display()
        ))
    })?;

    // `/node`'s own members must be cleared before any of them is opened.
    // `read_graph_body` validates this group too, but only after the `type`
    // read below — and by then HDF5 would already have opened and read an
    // external `/node/type`, which is the access the policy forbids. Failing
    // afterwards is not the same as not looking.
    validate_group_links(&root, &format!("/{KEY_NODE}"))?;

    // `/node` is a serialized NIRGraph. Checking its type is what separates a
    // NIR file from unrelated HDF5 that happens to have a `node` group.
    let root_type = read_string_scalar(
        &root
            .dataset(KEY_TYPE)
            .map_err(|_| NirError::MissingField(format!("/{KEY_NODE}/{KEY_TYPE}")))?,
        KEY_NODE,
        &budget,
    )?;
    if root_type != "NIRGraph" {
        return Err(NirError::InvalidGraph(format!(
            "/{KEY_NODE} must be a NIRGraph, found {root_type:?}"
        )));
    }

    let mut graph = read_graph_body(&root, &format!("/{KEY_NODE}"), &mut Vec::new(), &budget)?;
    graph.metadata = read_metadata(&root, &budget)?;
    graph.version = match version_dataset(&file)? {
        Some(ds) => Some(read_string_scalar(&ds, KEY_VERSION, &budget)?),
        None => None,
    };
    Ok(graph)
}

/// `/version` as a dataset, or [`None`] when the link is absent.
///
/// Shared by [`read`] and [`read_version`] so the two cannot disagree about
/// which error a *malformed* `/version` produces — a link that exists but is
/// not a dataset is [`NirError::Io`] either way. They differ only in whether
/// absence is an error, which is each caller's own decision. Same split as
/// [`NodeReader::optional`] / [`NodeReader::required`] for node fields.
fn version_dataset(file: &File) -> Result<Option<Dataset>> {
    if !file.link_exists(KEY_VERSION) {
        return Ok(None);
    }
    file.dataset(KEY_VERSION).map(Some).map_err(|e| {
        NirError::Io(format!(
            "/{KEY_VERSION}: expected a dataset, found another link kind: {e}"
        ))
    })
}

/// Read only `/version`.
pub(super) fn read_version(path: &Path, opts: &ReadOptions) -> Result<String> {
    let budget = ReadBudget::new(opts);
    let file = open(path)?;
    validate_group_links(&file, "/")?;
    let ds =
        version_dataset(&file)?.ok_or_else(|| NirError::MissingField(format!("/{KEY_VERSION}")))?;
    read_string_scalar(&ds, KEY_VERSION, &budget)
}

fn open(path: &Path) -> Result<File> {
    File::open(path).map_err(|e| NirError::Io(format!("cannot open {}: {e}", path.display())))
}

// ---------------------------------------------------------------------------
// Graph structure
// ---------------------------------------------------------------------------

/// Read a graph's `nodes` and `edges`, leaving metadata to the caller.
///
/// Split out because a nested `NIRGraph` node and the root graph share this
/// body but source their metadata from different places. `context` is the
/// group's path, used to say *which* graph a missing key belongs to.
///
/// Both keys are **required**, matching upstream `NIRGraph.from_dict`, which
/// asserts their presence even for an empty graph. Defaulting a missing
/// `edges` to "no connections" would turn a truncated file into a silently
/// disconnected model.
fn read_graph_body(
    group: &Group,
    context: &str,
    visited: &mut Vec<LocationToken>,
    budget: &ReadBudget,
) -> Result<NirGraph> {
    // A deep but *acyclic* chain gives every level a unique token, so the
    // repeat check below cannot stop it; unbounded recursion would abort the
    // process.
    if visited.len() >= MAX_NESTED_GRAPHS {
        return Err(NirError::InvalidGraph(format!(
            "{context}: more than {MAX_NESTED_GRAPHS} nested NIRGraph groups"
        )));
    }
    let token = group.loc_info()?.token;
    if visited.contains(&token) {
        return Err(NirError::InvalidGraph(format!(
            "{context}: nested NIRGraph group is already being decoded (hard-link cycle or alias)"
        )));
    }
    // Kept for the whole read, not popped on exit. Popping would allow a
    // "diamond" of hard-links to the same subgraph, and two aliases per level
    // re-expand the shared subtree, so a few dozen groups can decode into
    // exponentially many graphs and exhaust memory. Every graph object is
    // therefore decoded at most once per file.
    visited.push(token);
    read_graph_body_inner(group, context, visited, budget)
}

fn read_graph_body_inner(
    group: &Group,
    context: &str,
    visited: &mut Vec<LocationToken>,
    budget: &ReadBudget,
) -> Result<NirGraph> {
    let mut graph = NirGraph::new();

    validate_group_links(group, context)?;

    let nodes = group
        .group(KEY_NODES)
        .map_err(|_| NirError::MissingField(format!("{context}/{KEY_NODES}")))?;
    validate_group_links(&nodes, &format!("{context}/{KEY_NODES}"))?;

    // HDF5 iterates links in name order; sorting makes that explicit and
    // keeps repeated reads of the same file identical.
    let mut names = nodes.member_names()?;
    names.sort();
    for name in names {
        let node_group = nodes
            .group(&name)
            .map_err(|e| NirError::Io(format!("node {name:?} is not a group: {e}")))?;
        validate_group_links(&node_group, &name)?;
        let node = read_node(&node_group, &name, visited, budget)?;
        graph.insert_node(name, node)?;
    }

    let edges = group
        .dataset(KEY_EDGES)
        .map_err(|_| NirError::MissingField(format!("{context}/{KEY_EDGES}")))?;
    graph.edges = read_edges(&edges, budget)?;

    Ok(graph)
}

fn read_edges(ds: &Dataset, budget: &ReadBudget) -> Result<Vec<(String, String)>> {
    validate_dataset_security(ds, KEY_EDGES)?;
    // h5py encodes an empty edge list as a zero-length dataset whose element
    // type is float, not string; treat any empty dataset as "no edges".
    if ds.size() == 0 {
        return Ok(Vec::new());
    }
    let shape = ds.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(NirError::InvalidGraph(format!(
            "{KEY_EDGES} must have shape (E, 2), found {shape:?}"
        )));
    }
    // Already validated above; skip the second property-list walk.
    let flat = read_strings_unchecked(ds, KEY_EDGES, budget)?;
    let mut strings = flat.into_iter();
    Ok(std::iter::from_fn(|| Some((strings.next()?, strings.next()?))).collect())
}

// ---------------------------------------------------------------------------
// Node dispatch
// ---------------------------------------------------------------------------

fn read_node(
    group: &Group,
    name: &str,
    visited: &mut Vec<LocationToken>,
    budget: &ReadBudget,
) -> Result<NirNode> {
    let type_ds = group
        .dataset(KEY_TYPE)
        .map_err(|_| NirError::MissingField(format!("{name}.{KEY_TYPE}")))?;
    let ty = read_string_scalar(&type_ds, name, budget)?;
    let metadata = read_metadata(group, budget)?;
    let r = NodeReader {
        group,
        name,
        budget,
    };

    let node = match ty.as_str() {
        "Input" => NirNode::Input(read_input(&r, metadata)?),
        "Output" => NirNode::Output(read_output(&r, metadata)?),
        "Affine" => NirNode::Affine(read_affine(&r, metadata)?),
        "Linear" => NirNode::Linear(read_linear(&r, metadata)?),
        "Scale" => NirNode::Scale(read_scale(&r, metadata)?),
        "Conv1d" => NirNode::Conv1d(read_conv1d(&r, metadata)?),
        "Conv2d" => NirNode::Conv2d(read_conv2d(&r, metadata)?),
        "CubaLI" => NirNode::CubaLi(read_cuba_li(&r, metadata)?),
        "CubaLIF" => NirNode::CubaLif(read_cuba_lif(&r, metadata)?),
        "Delay" => NirNode::Delay(read_delay(&r, metadata)?),
        "Flatten" => NirNode::Flatten(read_flatten(&r, metadata)?),
        "I" => NirNode::I(read_i(&r, metadata)?),
        "IF" => NirNode::If(read_if(&r, metadata)?),
        "LI" => NirNode::Li(read_li(&r, metadata)?),
        "LIF" => NirNode::Lif(read_lif(&r, metadata)?),
        "SumPool2d" => NirNode::SumPool2d(read_sum_pool2d(&r, metadata)?),
        "AvgPool2d" => NirNode::AvgPool2d(read_avg_pool2d(&r, metadata)?),
        "Threshold" => NirNode::Threshold(read_threshold(&r, metadata)?),
        "NIRGraph" => {
            let mut sub = read_graph_body(group, name, visited, budget)?;
            sub.metadata = metadata;
            NirNode::Graph(Box::new(sub))
        }
        other => return Err(NirError::UnknownNodeType(other.to_owned())),
    };

    debug_assert!(
        wire::is_wire_type(node.type_name()),
        "decoded a node whose type is not in WIRE_TYPES"
    );
    Ok(node)
}

// ---------------------------------------------------------------------------
// Ports and linear maps
// ---------------------------------------------------------------------------

fn read_input(r: &NodeReader, metadata: MetadataMap) -> Result<Input> {
    Ok(Input {
        shape: r.usizes("shape")?,
        metadata,
    })
}

fn read_output(r: &NodeReader, metadata: MetadataMap) -> Result<Output> {
    Ok(Output {
        shape: r.usizes("shape")?,
        metadata,
    })
}

fn read_affine(r: &NodeReader, metadata: MetadataMap) -> Result<Affine> {
    Ok(Affine {
        weight: r.tensor("weight")?,
        bias: r.tensor("bias")?,
        metadata,
    })
}

fn read_linear(r: &NodeReader, metadata: MetadataMap) -> Result<Linear> {
    Ok(Linear {
        weight: r.tensor("weight")?,
        metadata,
    })
}

fn read_scale(r: &NodeReader, metadata: MetadataMap) -> Result<Scale> {
    Ok(Scale {
        scale: r.tensor("scale")?,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------

/// The field set `Conv1d` and `Conv2d` share on the wire — everything except
/// `input_shape`, whose arity differs between the two and so stays with the
/// per-type readers below.
struct ConvGeometry {
    weight: Tensor,
    stride: Vec<i64>,
    padding: Padding,
    dilation: Vec<i64>,
    groups: i64,
    bias: Tensor,
}

impl ConvGeometry {
    fn read(r: &NodeReader) -> Result<Self> {
        Ok(Self {
            weight: r.tensor("weight")?,
            stride: r.ints("stride")?,
            padding: r.padding()?,
            dilation: r.ints("dilation")?,
            groups: r.int_scalar("groups")?,
            bias: r.tensor("bias")?,
        })
    }

    fn into_conv1d(self, input_shape: Option<usize>, metadata: MetadataMap) -> Conv1d {
        let Self {
            weight,
            stride,
            padding,
            dilation,
            groups,
            bias,
        } = self;
        Conv1d {
            weight,
            stride,
            padding,
            dilation,
            groups,
            bias,
            input_shape,
            metadata,
        }
    }

    fn into_conv2d(self, input_shape: Option<Vec<usize>>, metadata: MetadataMap) -> Conv2d {
        Conv2d {
            weight: self.weight,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
            bias: self.bias,
            input_shape,
            metadata,
        }
    }
}

fn read_conv1d(r: &NodeReader, metadata: MetadataMap) -> Result<Conv1d> {
    // Upstream `Conv1d.input_shape` is a bare `int`.
    Ok(ConvGeometry::read(r)?.into_conv1d(r.opt_dim("input_shape")?, metadata))
}

fn read_conv2d(r: &NodeReader, metadata: MetadataMap) -> Result<Conv2d> {
    // Upstream `Conv2d.input_shape` is a `(N_x, N_y)` tuple.
    Ok(ConvGeometry::read(r)?.into_conv2d(r.opt_usizes("input_shape")?, metadata))
}

// ---------------------------------------------------------------------------
// Neuron models
// ---------------------------------------------------------------------------

/// `v_threshold` together with the reset potential, defaulted as Python does.
fn threshold_and_reset(r: &NodeReader) -> Result<(Tensor, Option<Tensor>)> {
    let v_threshold = r.tensor("v_threshold")?;
    let v_reset = r.v_reset(&v_threshold)?;
    Ok((v_threshold, v_reset))
}

/// The `tau` / `r` / `v_leak` membrane-dynamics group `LI` and `LIF` share.
struct LeakDynamics {
    tau: Tensor,
    r: Tensor,
    v_leak: Tensor,
}

impl LeakDynamics {
    fn read(r: &NodeReader) -> Result<Self> {
        Ok(Self {
            tau: r.tensor("tau")?,
            r: r.tensor("r")?,
            v_leak: r.tensor("v_leak")?,
        })
    }

    fn into_li(self, metadata: MetadataMap) -> Li {
        let Self { tau, r, v_leak } = self;
        Li {
            tau,
            r,
            v_leak,
            metadata,
        }
    }

    fn into_lif(self, v_threshold: Tensor, v_reset: Option<Tensor>, metadata: MetadataMap) -> Lif {
        let Self { tau, r, v_leak } = self;
        Lif {
            tau,
            r,
            v_leak,
            v_reset,
            v_threshold,
            metadata,
        }
    }
}

/// The `tau_syn` / `tau_mem` / `r` / `w_in` / `v_leak` group `CubaLI` and
/// `CubaLIF` share, with `w_in` carrying its Python default.
struct CubaDynamics {
    tau_syn: Tensor,
    tau_mem: Tensor,
    r: Tensor,
    w_in: Option<Tensor>,
    v_leak: Tensor,
}

impl CubaDynamics {
    fn read(r: &NodeReader) -> Result<Self> {
        let v_leak = r.tensor("v_leak")?;
        Ok(Self {
            tau_syn: r.tensor("tau_syn")?,
            tau_mem: r.tensor("tau_mem")?,
            r: r.tensor("r")?,
            w_in: r.w_in(&v_leak)?,
            v_leak,
        })
    }

    fn into_li(self, metadata: MetadataMap) -> CubaLi {
        let Self {
            tau_syn,
            tau_mem,
            r,
            w_in,
            v_leak,
        } = self;
        CubaLi {
            tau_syn,
            tau_mem,
            r,
            w_in,
            v_leak,
            metadata,
        }
    }

    fn into_lif(
        self,
        v_threshold: Tensor,
        v_reset: Option<Tensor>,
        metadata: MetadataMap,
    ) -> CubaLif {
        let Self {
            tau_syn,
            tau_mem,
            r,
            w_in,
            v_leak,
        } = self;
        CubaLif {
            tau_syn,
            tau_mem,
            r,
            v_reset,
            w_in,
            v_leak,
            v_threshold,
            metadata,
        }
    }
}

fn read_cuba_li(r: &NodeReader, metadata: MetadataMap) -> Result<CubaLi> {
    Ok(CubaDynamics::read(r)?.into_li(metadata))
}

fn read_cuba_lif(r: &NodeReader, metadata: MetadataMap) -> Result<CubaLif> {
    let (v_threshold, v_reset) = threshold_and_reset(r)?;
    Ok(CubaDynamics::read(r)?.into_lif(v_threshold, v_reset, metadata))
}

fn read_i(r: &NodeReader, metadata: MetadataMap) -> Result<I> {
    Ok(I {
        r: r.tensor("r")?,
        metadata,
    })
}

fn read_if(r: &NodeReader, metadata: MetadataMap) -> Result<If> {
    let (v_threshold, v_reset) = threshold_and_reset(r)?;
    Ok(If {
        r: r.tensor("r")?,
        v_reset,
        v_threshold,
        metadata,
    })
}

fn read_li(r: &NodeReader, metadata: MetadataMap) -> Result<Li> {
    Ok(LeakDynamics::read(r)?.into_li(metadata))
}

fn read_lif(r: &NodeReader, metadata: MetadataMap) -> Result<Lif> {
    let (v_threshold, v_reset) = threshold_and_reset(r)?;
    Ok(LeakDynamics::read(r)?.into_lif(v_threshold, v_reset, metadata))
}

// ---------------------------------------------------------------------------
// Pooling and the remaining leaf nodes
// ---------------------------------------------------------------------------

/// `SumPool2d` and `AvgPool2d` carry an identical field set.
fn read_pool_window(r: &NodeReader) -> Result<(Tensor, Tensor, Tensor)> {
    Ok((
        r.tensor("kernel_size")?,
        r.tensor("stride")?,
        r.tensor("padding")?,
    ))
}

fn read_sum_pool2d(r: &NodeReader, metadata: MetadataMap) -> Result<SumPool2d> {
    let (kernel_size, stride, padding) = read_pool_window(r)?;
    Ok(SumPool2d {
        kernel_size,
        stride,
        padding,
        metadata,
    })
}

fn read_avg_pool2d(r: &NodeReader, metadata: MetadataMap) -> Result<AvgPool2d> {
    let (kernel_size, stride, padding) = read_pool_window(r)?;
    Ok(AvgPool2d {
        kernel_size,
        stride,
        padding,
        metadata,
    })
}

fn read_delay(r: &NodeReader, metadata: MetadataMap) -> Result<Delay> {
    Ok(Delay {
        delay: r.tensor("delay")?,
        metadata,
    })
}

fn read_flatten(r: &NodeReader, metadata: MetadataMap) -> Result<Flatten> {
    Ok(Flatten {
        // Upstream's dataclass defaults are start_dim = 1, end_dim = -1.
        start_dim: r.opt_int_scalar("start_dim")?.unwrap_or(1),
        end_dim: r.opt_int_scalar("end_dim")?.unwrap_or(-1),
        // Flatten stores its input shape under the key `input_type`.
        input_type: r.opt_usizes("input_type")?,
        metadata,
    })
}

fn read_threshold(r: &NodeReader, metadata: MetadataMap) -> Result<Threshold> {
    Ok(Threshold {
        threshold: r.tensor("threshold")?,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Field access
// ---------------------------------------------------------------------------

/// A node's HDF5 group paired with its name.
///
/// Bundling the two means each accessor takes only a field name, and every
/// error message can say which node it came from without the caller repeating
/// itself.
struct NodeReader<'a> {
    group: &'a Group,
    name: &'a str,
    budget: &'a ReadBudget,
}

impl NodeReader<'_> {
    /// `node.field`, the prefix every error message from this node uses.
    fn context(&self, field: &str) -> String {
        format!("{}.{field}", self.name)
    }

    /// The dataset for an optional field, or [`None`] when it is absent.
    ///
    /// A link that *exists* but is not a dataset is an error rather than an
    /// absent field: treating a malformed `v_reset` group as "not there" would
    /// silently synthesize a default and change the model.
    fn optional(&self, field: &str) -> Result<Option<Dataset>> {
        if !self.group.link_exists(field) {
            return Ok(None);
        }
        self.group.dataset(field).map(Some).map_err(|e| {
            NirError::Io(format!(
                "{}: expected a dataset, found another link kind: {e}",
                self.context(field)
            ))
        })
    }

    fn required(&self, field: &str) -> Result<Dataset> {
        self.optional(field)?
            .ok_or_else(|| NirError::MissingField(self.context(field)))
    }

    fn tensor(&self, field: &str) -> Result<Tensor> {
        read_tensor(&self.required(field)?, &self.context(field), self.budget)
    }

    fn opt_tensor(&self, field: &str) -> Result<Option<Tensor>> {
        match self.optional(field)? {
            Some(ds) => read_tensor(&ds, &self.context(field), self.budget).map(Some),
            None => Ok(None),
        }
    }

    fn ints(&self, field: &str) -> Result<Vec<i64>> {
        read_ints(&self.required(field)?, &self.context(field), self.budget)
    }

    fn opt_ints(&self, field: &str) -> Result<Option<Vec<i64>>> {
        match self.optional(field)? {
            Some(ds) => read_ints(&ds, &self.context(field), self.budget).map(Some),
            None => Ok(None),
        }
    }

    fn int_scalar(&self, field: &str) -> Result<i64> {
        single_int(self.ints(field)?, &self.context(field))
    }

    fn opt_int_scalar(&self, field: &str) -> Result<Option<i64>> {
        match self.opt_ints(field)? {
            Some(values) => single_int(values, &self.context(field)).map(Some),
            None => Ok(None),
        }
    }

    fn usizes(&self, field: &str) -> Result<Vec<usize>> {
        to_usizes(self.ints(field)?, &self.context(field), self.budget)
    }

    fn opt_usizes(&self, field: &str) -> Result<Option<Vec<usize>>> {
        match self.opt_ints(field)? {
            Some(values) => to_usizes(values, &self.context(field), self.budget).map(Some),
            None => Ok(None),
        }
    }

    /// A single non-negative extent, for `Conv1d.input_shape`.
    fn opt_dim(&self, field: &str) -> Result<Option<usize>> {
        let Some(values) = self.opt_usizes(field)? else {
            return Ok(None);
        };
        match values.as_slice() {
            [only] => Ok(Some(*only)),
            other => Err(NirError::InvalidTensor(format!(
                "{}: expected a single extent, found {} values",
                self.context(field),
                other.len()
            ))),
        }
    }

    /// `padding` is an integer extent list, or the string `"same"` / `"valid"`.
    fn padding(&self) -> Result<Padding> {
        let ds = self.required("padding")?;
        if is_string(&ds)? {
            wire::padding_from_wire_str(&read_string_scalar(&ds, self.name, self.budget)?)
        } else {
            Ok(Padding::Explicit(read_ints(
                &ds,
                &self.context("padding"),
                self.budget,
            )?))
        }
    }

    /// `v_reset`, defaulting to `zeros_like(v_threshold)` as Python does.
    fn v_reset(&self, v_threshold: &Tensor) -> Result<Option<Tensor>> {
        match self.opt_tensor("v_reset")? {
            Some(tensor) => Ok(Some(tensor)),
            None => {
                self.budget.charge(
                    &self.context("v_reset (synthesized)"),
                    v_threshold
                        .data()
                        .len()
                        .checked_mul(v_threshold.dtype().size_of()),
                )?;
                Ok(Some(v_threshold.zeros_like()))
            }
        }
    }

    /// `w_in`, defaulting to `ones_like(v_leak)` as Python does.
    fn w_in(&self, v_leak: &Tensor) -> Result<Option<Tensor>> {
        match self.opt_tensor("w_in")? {
            Some(tensor) => Ok(Some(tensor)),
            None => {
                self.budget.charge(
                    &self.context("w_in (synthesized)"),
                    v_leak.data().len().checked_mul(v_leak.dtype().size_of()),
                )?;
                Ok(Some(v_leak.ones_like()))
            }
        }
    }
}

fn to_usizes(values: Vec<i64>, context: &str, budget: &ReadBudget) -> Result<Vec<usize>> {
    // `values` is still alive while `collect` allocates the `Vec<usize>`, so
    // charge the destination buffer before conversion to keep the budget
    // accurate. The `u64` -> `i64` path in `read_tensor` already charges both
    // buffers; this mirrors that for `usize` extents.
    budget.charge(context, values.len().checked_mul(size_of::<usize>()))?;
    values
        .into_iter()
        .map(|v| {
            usize::try_from(v).map_err(|_| {
                NirError::InvalidTensor(format!("{context}: negative axis length {v}"))
            })
        })
        .collect()
}

fn single_int(values: Vec<i64>, context: &str) -> Result<i64> {
    match values.as_slice() {
        [only] => Ok(*only),
        other => Err(NirError::InvalidTensor(format!(
            "{context}: expected a single integer, found {} values",
            other.len()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Security validation (reject external links and VDS)
// ---------------------------------------------------------------------------

/// Validate that a group does not contain external or soft links before traversing it.
///
/// Checks each member link to ensure it's a hard link only. Soft links can
/// resolve to external links, so both are rejected.
fn validate_group_links(group: &Group, context: &str) -> Result<()> {
    let bad_link = group
        .iter_visit_default(
            None,
            |_group, name, info, found: &mut Option<(String, &'static str)>| match info.link_type {
                hdf5::LinkType::External => {
                    *found = Some((name.to_owned(), "external"));
                    false
                }
                hdf5::LinkType::Soft => {
                    *found = Some((name.to_owned(), "soft"));
                    false
                }
                hdf5::LinkType::Hard => true,
            },
        )
        .map_err(|e| NirError::Io(format!("{context}: cannot inspect group links: {e}")))?;
    if let Some((name, kind)) = bad_link {
        return Err(NirError::InvalidGraph(format!(
            "{context}: {kind} link '{name}' is not allowed"
        )));
    }
    Ok(())
}

/// Validate that a dataset does not use disallowed storage layouts.
///
/// Rejects external storage (H5D_EXTERNAL) and virtual (VDS) layouts, so a
/// dataset's raw data can only come from inside the `.nir` file itself. Fails
/// closed when the dataset creation property list cannot be read.
fn validate_dataset_security(ds: &Dataset, context: &str) -> Result<()> {
    let dcpl = ds.dcpl().map_err(|e| {
        NirError::Io(format!(
            "{context}: cannot read dataset creation property list: {e}"
        ))
    })?;

    let external = dcpl.external();
    if !external.is_empty() {
        return Err(NirError::InvalidGraph(format!(
            "{context}: external storage is not allowed ({} external file(s))",
            external.len()
        )));
    }

    if !matches!(
        ds.layout(),
        Layout::Compact | Layout::Contiguous | Layout::Chunked
    ) {
        return Err(NirError::InvalidGraph(format!(
            "{context}: virtual dataset layouts are not allowed"
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

fn read_metadata(group: &Group, budget: &ReadBudget) -> Result<MetadataMap> {
    // Absent is fine (empty map). A present link of the wrong kind — e.g. a
    // dataset named `metadata` — is an error, not silent data loss.
    if !group.link_exists(KEY_METADATA) {
        return Ok(MetadataMap::new());
    }
    let md = group.group(KEY_METADATA).map_err(|e| {
        NirError::Io(format!(
            "{KEY_METADATA}: expected a group, found another link kind: {e}"
        ))
    })?;
    validate_group_links(&md, KEY_METADATA)?;
    let mut out = MetadataMap::new();
    for key in md.member_names()? {
        let ds = md.dataset(&key).map_err(|e| {
            NirError::Io(format!(
                "metadata {key:?} must be a dataset, not a group: {e}"
            ))
        })?;
        let value = read_metadata_value(&ds, &key, budget)?;
        out.insert(key, value);
    }
    Ok(out)
}

/// Decode a string-typed metadata dataset.
///
/// Security is already checked by the caller; what is left is arity. A
/// rank-1 string dataset is a Python `list[str]` — reading it as one string
/// would fail the whole file, and [`Tensor`] holds no strings, so it needs
/// [`MetadataValue::StringList`].
///
/// The split is on **rank, not element count**: rank 0 is `String`, rank 1 is
/// `StringList` even when it holds one element. That is what h5py emits —
/// a scalar `str` is shape `()`, a `list[str]` is shape `[n]` — and it makes
/// the round-trip exact, since a one-element `StringList` would otherwise be
/// written as `[1]` and read back as `String`. Counting elements instead
/// would collapse the two.
///
/// The looser size-based rule in [`read_string_scalar`] is deliberate and
/// stays: `/version`, node `type` and symbolic `padding` accept a single
/// element at any rank, because producers disagree on whether a lone string
/// is `()` or `[1]`. Metadata is the one place where the distinction carries
/// meaning, so it is the one place that reads rank.
///
/// Rank 2+ is a `list[list[str]]`, which `StringList` cannot hold: decoding
/// one would drop the nesting and write it back as rank-1, a silent reshape.
/// Refusing loses nothing, since such a file did not load before `StringList`
/// existed either — it just failed with a message about the wrong problem.
fn read_string_metadata(
    ds: &Dataset,
    key: &str,
    context: &str,
    budget: &ReadBudget,
) -> Result<MetadataValue> {
    match ds.shape().as_slice() {
        [] => Ok(MetadataValue::String(read_string_scalar_validated(
            ds, key, budget,
        )?)),
        [_] => Ok(MetadataValue::StringList(read_strings_unchecked(
            ds, context, budget,
        )?)),
        shape => Err(NirError::InvalidTensor(format!(
            "{context}: string metadata must be scalar or rank-1, found shape {shape:?}"
        ))),
    }
}

fn read_metadata_value(ds: &Dataset, key: &str, budget: &ReadBudget) -> Result<MetadataValue> {
    validate_dataset_security(ds, &format!("{KEY_METADATA}.{key}"))?;
    let scalar = ds.shape().is_empty();
    let context = format!("{KEY_METADATA}.{key}");
    let value = match ds.dtype()?.to_descriptor()? {
        Td::VarLenUnicode | Td::VarLenAscii | Td::FixedAscii(_) | Td::FixedUnicode(_) => {
            read_string_metadata(ds, key, &context, budget)?
        }
        Td::Boolean if scalar => {
            budget.charge(&context, Some(size_of::<bool>()))?;
            MetadataValue::Bool(ds.read_scalar::<bool>()?)
        }
        Td::Float(_) if scalar => {
            budget.charge(&context, Some(size_of::<f64>()))?;
            MetadataValue::F64(ds.read_scalar::<f64>()?)
        }
        Td::Integer(_) if scalar => {
            budget.charge(&context, Some(size_of::<i64>()))?;
            MetadataValue::I64(ds.read_scalar::<i64>()?)
        }
        Td::Unsigned(IntSize::U8) if scalar => {
            // Same checked conversion as tensor `u64` payloads — HDF5's own
            // i64 cast can saturate above i64::MAX.
            budget.charge(&context, Some(size_of::<i64>()))?;
            let v = ds.read_scalar::<u64>()?;
            MetadataValue::I64(i64::try_from(v).map_err(|_| {
                NirError::InvalidTensor(format!("{context}: u64 value {v} does not fit in i64"))
            })?)
        }
        Td::Unsigned(IntSize::U1 | IntSize::U2 | IntSize::U4) if scalar => {
            budget.charge(&context, Some(size_of::<i64>()))?;
            MetadataValue::I64(ds.read_scalar::<i64>()?)
        }
        _ => MetadataValue::Tensor(read_tensor(ds, &context, budget)?),
    };
    Ok(value)
}

// ---------------------------------------------------------------------------
// Typed dataset decoding
// ---------------------------------------------------------------------------

/// Decode a numeric dataset, preserving its on-disk element type.
///
/// Narrower integers widen into [`DType::I64`](crate::DType::I64); floats keep
/// their width so an `f32` file never silently becomes `f64` (or worse, the
/// reverse). Anything else is rejected rather than guessed at.
fn read_tensor(ds: &Dataset, context: &str, budget: &ReadBudget) -> Result<Tensor> {
    validate_dataset_security(ds, context)?;
    let descriptor = ds.dtype()?.to_descriptor()?;
    let data = match descriptor {
        Td::Float(FloatSize::U4) => {
            charge_elements(budget, ds, context, size_of::<f32>())?;
            TensorData::F32(ds.read_raw::<f32>()?)
        }
        Td::Float(FloatSize::U8) => {
            charge_elements(budget, ds, context, size_of::<f64>())?;
            TensorData::F64(ds.read_raw::<f64>()?)
        }
        Td::Integer(_) | Td::Unsigned(IntSize::U1 | IntSize::U2 | IntSize::U4) => {
            charge_elements(budget, ds, context, size_of::<i64>())?;
            TensorData::I64(ds.read_raw::<i64>()?)
        }
        Td::Unsigned(IntSize::U8) => {
            let two_buffers = size_of::<u64>().checked_add(size_of::<i64>());
            budget.charge(
                context,
                two_buffers.and_then(|width| ds.size().checked_mul(width)),
            )?;
            TensorData::I64(read_u64_as_i64(ds, context)?)
        }
        Td::Boolean => {
            charge_elements(budget, ds, context, size_of::<bool>())?;
            TensorData::Bool(ds.read_raw::<bool>()?)
        }
        other => {
            return Err(NirError::InvalidTensor(format!(
                "{context}: element type {other} has no NIR dtype"
            )));
        }
    };
    Tensor::new(ds.shape(), data)
}

fn charge_elements(
    budget: &ReadBudget,
    ds: &Dataset,
    context: &str,
    decoded_width: usize,
) -> Result<()> {
    budget.charge(context, ds.size().checked_mul(decoded_width))
}

/// `u64` is the one integer width that does not fit losslessly in `i64`.
fn read_u64_as_i64(ds: &Dataset, context: &str) -> Result<Vec<i64>> {
    ds.read_raw::<u64>()?
        .into_iter()
        .map(|v| {
            i64::try_from(v).map_err(|_| {
                NirError::InvalidTensor(format!("{context}: u64 value {v} does not fit in i64"))
            })
        })
        .collect()
}

fn read_ints(ds: &Dataset, context: &str, budget: &ReadBudget) -> Result<Vec<i64>> {
    // `into_data` rather than cloning out of `data()`: the decoded buffer is
    // handed over instead of duplicated, so peak use on this path is one copy
    // rather than two.
    match read_tensor(ds, context, budget)?.into_data() {
        TensorData::I64(values) => Ok(values),
        other => Err(NirError::InvalidTensor(format!(
            "{context}: expected integer data, found {:?}",
            other.dtype()
        ))),
    }
}

fn is_string(ds: &Dataset) -> Result<bool> {
    Ok(matches!(
        ds.dtype()?.to_descriptor()?,
        Td::VarLenUnicode | Td::VarLenAscii | Td::FixedAscii(_) | Td::FixedUnicode(_)
    ))
}

fn read_string_scalar(ds: &Dataset, context: &str, budget: &ReadBudget) -> Result<String> {
    validate_dataset_security(ds, context)?;
    read_string_scalar_validated(ds, context, budget)
}

/// Cardinality check + decode for a string dataset that has already passed
/// [`validate_dataset_security`].
fn read_string_scalar_validated(
    ds: &Dataset,
    context: &str,
    budget: &ReadBudget,
) -> Result<String> {
    let size = ds.size();
    if size != 1 {
        return Err(NirError::Io(format!(
            "{context}: expected a single string, found {size} elements"
        )));
    }
    let mut values = read_strings_unchecked(ds, context, budget)?;
    match values.len() {
        1 => Ok(values.remove(0)),
        n => Err(NirError::Io(format!(
            "{context}: expected a single string, found {n}"
        ))),
    }
}

/// Decode a string dataset without re-running [`validate_dataset_security`].
///
/// Callers that already inspected the property list (edges, scalar strings)
/// use this so the HDF5 walk is not repeated for the same dataset. The sole
/// entry points that still need a validating wrapper are
/// [`read_string_scalar`] and the edges path, both of which call
/// `validate_dataset_security` themselves first.
fn read_strings_unchecked(ds: &Dataset, context: &str, budget: &ReadBudget) -> Result<Vec<String>> {
    charge_strings(ds, context, budget)?;

    macro_rules! read_fixed {
        ($ty:ident, $width:expr, $($cap:literal),+) => {
            $(if $width <= $cap {
                return Ok(ds
                    .read_raw::<$ty<$cap>>()?
                    .iter()
                    .map(ToString::to_string)
                    .collect());
            })+
        };
    }

    match ds.dtype()?.to_descriptor()? {
        Td::VarLenUnicode => Ok(ds
            .read_raw::<VarLenUnicode>()?
            .iter()
            .map(ToString::to_string)
            .collect()),
        Td::VarLenAscii => Ok(ds
            .read_raw::<VarLenAscii>()?
            .iter()
            .map(ToString::to_string)
            .collect()),
        Td::FixedAscii(width) => {
            // Rungs must match `FIXED_STRING_CAPS`, which is what
            // `decoded_element_bytes` charges the size guard for.
            read_fixed!(FixedAscii, width, 64, 256, 4096);
            Err(too_wide(context, width))
        }
        Td::FixedUnicode(width) => {
            // Rungs must match `FIXED_STRING_CAPS` — see above.
            read_fixed!(FixedUnicode, width, 64, 256, 4096);
            Err(too_wide(context, width))
        }
        other => Err(NirError::Io(format!(
            "{context}: expected a string dataset, found {other}"
        ))),
    }
}

/// Charge every allocation performed while decoding a string dataset.
fn charge_strings(ds: &Dataset, context: &str, budget: &ReadBudget) -> Result<()> {
    if budget.limit.is_none() {
        return Ok(());
    }

    let count = ds.size();
    let headers = count.checked_mul(size_of::<String>());
    let requested = match ds.dtype()?.to_descriptor()? {
        Td::VarLenUnicode => {
            let descriptors = count.checked_mul(size_of::<VarLenUnicode>());
            // Reject enormous declared counts before asking HDF5 to size the
            // heap, since the VLEN sizing call itself must traverse the data.
            let min = checked_sum([descriptors, headers]);
            budget.would_fit(context, min)?;
            let payload = vlen_payload_bytes(ds, context)?;
            checked_sum([descriptors, Some(payload), headers, Some(payload)])
        }
        Td::VarLenAscii => {
            let descriptors = count.checked_mul(size_of::<VarLenAscii>());
            let min = checked_sum([descriptors, headers]);
            budget.would_fit(context, min)?;
            let payload = vlen_payload_bytes(ds, context)?;
            checked_sum([descriptors, Some(payload), headers, Some(payload)])
        }
        Td::FixedAscii(width) | Td::FixedUnicode(width) => {
            let Some(capacity) = FIXED_STRING_CAPS.iter().copied().find(|cap| width <= *cap) else {
                return Ok(());
            };
            checked_sum([
                count.checked_mul(capacity),
                headers,
                count.checked_mul(width),
            ])
        }
        _ => return Ok(()),
    };
    budget.charge(context, requested)
}

fn checked_sum<const N: usize>(parts: [Option<usize>; N]) -> Option<usize> {
    parts
        .into_iter()
        .try_fold(0usize, |sum, part| sum.checked_add(part?))
}

/// Ask HDF5 how many heap bytes a VLEN dataset will allocate during `H5Dread`.
#[allow(deprecated)]
fn vlen_payload_bytes(ds: &Dataset, context: &str) -> Result<usize> {
    // `H5Dvlen_get_buf_size` aborts on scalar VLEN across multiple HDF5
    // releases, including the 1.10 series in CI. The containing file size is
    // a safe upper bound for a single in-container scalar heap payload;
    // external storage and VDS have already been rejected.
    if ds.shape().is_empty() {
        return usize::try_from(ds.file()?.size()).map_err(|_| {
            NirError::Io(format!(
                "{context}: containing file size does not fit usize"
            ))
        });
    }

    let dtype = ds.dtype()?;
    let space = ds.space()?;
    let mut bytes: hdf5_sys::h5::hsize_t = 0;
    let status = hdf5::sync::sync(|| {
        // SAFETY: `ds`, `dtype`, and `space` keep live borrowed HDF5 IDs for
        // this call; `&mut bytes` is a valid output pointer, no ownership is
        // transferred, and the hdf5-metno global lock is held by `sync`.
        unsafe { hdf5_sys::h5d::H5Dvlen_get_buf_size(ds.id(), dtype.id(), space.id(), &mut bytes) }
    });
    hdf5::h5check(status).map_err(|e| {
        NirError::Io(format!(
            "{context}: cannot determine variable-length string allocation: {e}"
        ))
    })?;
    usize::try_from(bytes).map_err(|_| {
        NirError::Io(format!(
            "{context}: variable-length string allocation does not fit usize"
        ))
    })
}

fn too_wide(context: &str, width: usize) -> NirError {
    NirError::Io(format!(
        "{context}: fixed-length string of {width} bytes exceeds the supported maximum of 4096"
    ))
}
