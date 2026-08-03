// SPDX-License-Identifier: MIT OR Apache-2.0

//! Load a NIR fixture, inspect every LIF parameter, save it, and verify the copy.

use nir_rs::io::DEFAULT_NIR_VERSION;
use nir_rs::nodes::Padding;
use nir_rs::types::{MetadataMap, MetadataValue, Tensor, TensorData};
use nir_rs::{NirError, NirGraph, NirNode, io};
use std::ffi::OsString;
use std::path::PathBuf;

const PREVIEW_ELEMENTS: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (input, output) = paths_from_args()?;
    let graph = io::read(&input)?;

    println!(
        "loaded {} (NIR version {}, {} nodes, {} edges)",
        input.display(),
        graph.version.as_deref().unwrap_or("<absent>"),
        graph.nodes.len(),
        graph.edges.len()
    );

    print_graph_structure("", &graph);

    let mut lif_count = 0;
    inspect_lifs("", &graph, &mut lif_count);

    if lif_count == 0 {
        println!("no LIF nodes found");
    }

    // Try a validating write first. If the graph contains values the wire
    // format cannot preserve exactly (dangling edges, nested graph versions,
    // rank-0 metadata tensors, etc.), `io::write` rejects it even though
    // `io::read` loaded it. Fall back to preservation mode for those files,
    // verifying round-trip equality whenever the file can be preserved exactly.
    let mut expected = graph.clone();
    if expected.version.is_none() {
        expected.version = Some(DEFAULT_NIR_VERSION.to_owned());
    }
    let has_nan = graph_has_nan(&expected);
    let has_lossy = graph_has_lossy_values(&graph);

    if let Err(e) = io::write(&output, &graph) {
        if is_representability_or_structure_error(&e) {
            io::write_with(
                &output,
                &graph,
                &io::WriteOptions::default().with_validation(false),
            )?;
            if has_nan || has_lossy {
                println!(
                    "saved {} (validation skipped: {e}; round-trip verification not possible)",
                    output.display()
                );
            } else {
                let reloaded = io::read(&output)?;
                assert_eq!(
                    reloaded, expected,
                    "saved graph did not round-trip exactly (finite values)"
                );
                println!("saved and verified {}", output.display());
            }
        } else {
            return Err(e.into());
        }
    } else if has_nan {
        println!(
            "saved {} (skipped equality check: graph has NaN; IEEE PartialEq cannot verify)",
            output.display()
        );
    } else if has_lossy {
        println!(
            "saved {} (skipped equality check: graph contains values the NIR wire cannot preserve exactly)",
            output.display()
        );
    } else {
        let reloaded = io::read(&output)?;
        assert_eq!(
            reloaded, expected,
            "saved graph did not round-trip exactly (finite values)"
        );
        println!("saved and verified {}", output.display());
    }

    Ok(())
}

fn paths_from_args() -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let input = args.next().map_or_else(default_input, PathBuf::from);
    let output = args.next().map_or_else(default_output, PathBuf::from);

    if let Some(extra) = args.next() {
        return Err(invalid_arguments(extra).into());
    }

    Ok((input, output))
}

fn default_input() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lif_norse.nir")
}

fn default_output() -> PathBuf {
    std::env::temp_dir().join(format!("nir-rs-lif-copy-{}.nir", std::process::id()))
}

fn invalid_arguments(extra: OsString) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "unexpected third argument {:?}; usage: load_inspect_lif [INPUT.nir] [OUTPUT.nir]",
            extra
        ),
    )
}

fn print_graph_structure(prefix: &str, graph: &NirGraph) {
    for (name, node) in &graph.nodes {
        println!("{prefix}node {name:?}: {}", node.type_name());
        if let NirNode::Graph(sub) = node {
            print_graph_structure(&format!("{prefix}  "), sub);
        }
    }
    for (source, target) in &graph.edges {
        println!("{prefix}edge {source:?} -> {target:?}");
    }
}

fn inspect_lifs(path: &str, graph: &NirGraph, lif_count: &mut usize) {
    for (name, node) in &graph.nodes {
        let label = if path.is_empty() {
            name.clone()
        } else {
            format!("{path}/{name}")
        };
        match node {
            NirNode::Lif(lif) => {
                *lif_count += 1;
                println!("LIF node {label:?} parameters:");
                print_tensor("tau", &lif.tau);
                print_tensor("r", &lif.r);
                print_tensor("v_leak", &lif.v_leak);
                print_tensor("v_threshold", &lif.v_threshold);
                match &lif.v_reset {
                    Some(v_reset) => print_tensor("v_reset", v_reset),
                    None => println!("  v_reset: <absent>"),
                }
            }
            NirNode::Graph(sub) => inspect_lifs(&label, sub, lif_count),
            _ => {}
        }
    }
}

fn graph_has_nan(graph: &NirGraph) -> bool {
    if metadata_has_nan(&graph.metadata) {
        return true;
    }
    for node in graph.nodes.values() {
        if node_has_nan(node) {
            return true;
        }
    }
    false
}

fn node_has_nan(node: &NirNode) -> bool {
    match node {
        NirNode::Input(n) => metadata_has_nan(&n.metadata),
        NirNode::Output(n) => metadata_has_nan(&n.metadata),
        NirNode::Affine(n) => {
            metadata_has_nan(&n.metadata) || tensor_has_nan(&n.weight) || tensor_has_nan(&n.bias)
        }
        NirNode::Linear(n) => metadata_has_nan(&n.metadata) || tensor_has_nan(&n.weight),
        NirNode::Scale(n) => metadata_has_nan(&n.metadata) || tensor_has_nan(&n.scale),
        NirNode::Conv1d(n) => {
            metadata_has_nan(&n.metadata) || tensor_has_nan(&n.weight) || tensor_has_nan(&n.bias)
        }
        NirNode::Conv2d(n) => {
            metadata_has_nan(&n.metadata) || tensor_has_nan(&n.weight) || tensor_has_nan(&n.bias)
        }
        NirNode::CubaLi(n) => {
            metadata_has_nan(&n.metadata)
                || tensor_has_nan(&n.tau_syn)
                || tensor_has_nan(&n.tau_mem)
                || tensor_has_nan(&n.r)
                || tensor_has_nan(&n.v_leak)
                || n.w_in.as_ref().is_some_and(tensor_has_nan)
        }
        NirNode::CubaLif(n) => {
            metadata_has_nan(&n.metadata)
                || tensor_has_nan(&n.tau_syn)
                || tensor_has_nan(&n.tau_mem)
                || tensor_has_nan(&n.r)
                || tensor_has_nan(&n.v_leak)
                || tensor_has_nan(&n.v_threshold)
                || n.v_reset.as_ref().is_some_and(tensor_has_nan)
                || n.w_in.as_ref().is_some_and(tensor_has_nan)
        }
        NirNode::Delay(n) => metadata_has_nan(&n.metadata) || tensor_has_nan(&n.delay),
        NirNode::Flatten(n) => metadata_has_nan(&n.metadata),
        NirNode::I(n) => metadata_has_nan(&n.metadata) || tensor_has_nan(&n.r),
        NirNode::If(n) => {
            metadata_has_nan(&n.metadata)
                || tensor_has_nan(&n.r)
                || tensor_has_nan(&n.v_threshold)
                || n.v_reset.as_ref().is_some_and(tensor_has_nan)
        }
        NirNode::Li(n) => {
            metadata_has_nan(&n.metadata)
                || tensor_has_nan(&n.tau)
                || tensor_has_nan(&n.r)
                || tensor_has_nan(&n.v_leak)
        }
        NirNode::Lif(n) => {
            metadata_has_nan(&n.metadata)
                || tensor_has_nan(&n.tau)
                || tensor_has_nan(&n.r)
                || tensor_has_nan(&n.v_leak)
                || tensor_has_nan(&n.v_threshold)
                || n.v_reset.as_ref().is_some_and(tensor_has_nan)
        }
        NirNode::SumPool2d(n) => {
            metadata_has_nan(&n.metadata)
                || tensor_has_nan(&n.kernel_size)
                || tensor_has_nan(&n.stride)
                || tensor_has_nan(&n.padding)
        }
        NirNode::AvgPool2d(n) => {
            metadata_has_nan(&n.metadata)
                || tensor_has_nan(&n.kernel_size)
                || tensor_has_nan(&n.stride)
                || tensor_has_nan(&n.padding)
        }
        NirNode::Threshold(n) => metadata_has_nan(&n.metadata) || tensor_has_nan(&n.threshold),
        NirNode::Graph(n) => graph_has_nan(n),
    }
}

fn metadata_has_nan(metadata: &MetadataMap) -> bool {
    for value in metadata.values() {
        if let MetadataValue::F64(v) = value
            && v.is_nan()
        {
            return true;
        }
        if let MetadataValue::Tensor(t) = value
            && tensor_has_nan(t)
        {
            return true;
        }
    }
    false
}

fn is_representability_or_structure_error(e: &NirError) -> bool {
    matches!(
        e,
        NirError::InvalidGraph(_) | NirError::MissingNode(_) | NirError::DuplicateEdge(..)
    )
}

fn graph_has_lossy_values(graph: &NirGraph) -> bool {
    if metadata_has_lossy_values(&graph.metadata) {
        return true;
    }
    for node in graph.nodes.values() {
        if node_has_lossy_values(node) {
            return true;
        }
    }
    false
}

fn node_has_lossy_values(node: &NirNode) -> bool {
    let node_lossy = |metadata: &MetadataMap| metadata_has_lossy_values(metadata);
    match node {
        NirNode::Input(n) => node_lossy(&n.metadata),
        NirNode::Output(n) => node_lossy(&n.metadata),
        NirNode::Affine(n) => node_lossy(&n.metadata),
        NirNode::Linear(n) => node_lossy(&n.metadata),
        NirNode::Scale(n) => node_lossy(&n.metadata),
        NirNode::Conv1d(n) => node_lossy(&n.metadata),
        NirNode::Conv2d(n) => conv2d_has_lossy_extents(n) || node_lossy(&n.metadata),
        NirNode::CubaLi(n) => node_lossy(&n.metadata),
        NirNode::CubaLif(n) => node_lossy(&n.metadata),
        NirNode::Delay(n) => node_lossy(&n.metadata),
        NirNode::Flatten(n) => node_lossy(&n.metadata),
        NirNode::I(n) => node_lossy(&n.metadata),
        NirNode::If(n) => node_lossy(&n.metadata),
        NirNode::Li(n) => node_lossy(&n.metadata),
        NirNode::Lif(n) => node_lossy(&n.metadata),
        NirNode::SumPool2d(n) => node_lossy(&n.metadata),
        NirNode::AvgPool2d(n) => node_lossy(&n.metadata),
        NirNode::Threshold(n) => node_lossy(&n.metadata),
        NirNode::Graph(n) => n.version.is_some() || graph_has_lossy_values(n),
    }
}

fn metadata_has_lossy_values(metadata: &MetadataMap) -> bool {
    for value in metadata.values() {
        if let MetadataValue::Tensor(t) = value
            && t.shape().is_empty()
        {
            return true;
        }
    }
    false
}

fn conv2d_has_lossy_extents(conv: &nir_rs::nodes::Conv2d) -> bool {
    conv.stride.len() == 1
        || conv.dilation.len() == 1
        || matches!(&conv.padding, Padding::Explicit(extents) if extents.len() == 1)
}

fn tensor_has_nan(tensor: &Tensor) -> bool {
    match tensor.data() {
        TensorData::F32(values) => values.iter().any(|v| v.is_nan()),
        TensorData::F64(values) => values.iter().any(|v| v.is_nan()),
        TensorData::I64(_) | TensorData::Bool(_) => false,
    }
}

fn print_tensor(name: &str, tensor: &Tensor) {
    print!(
        "  {name}: dtype={:?}, shape={:?}, values=",
        tensor.dtype(),
        tensor.shape()
    );

    match tensor.data() {
        TensorData::F32(values) => println!("{}", format_preview(values)),
        TensorData::F64(values) => println!("{}", format_preview(values)),
        TensorData::I64(values) => println!("{}", format_preview(values)),
        TensorData::Bool(values) => println!("{}", format_preview(values)),
    }
}

/// First `PREVIEW_ELEMENTS` values, with an ellipsis and total length when truncated.
fn format_preview<T: std::fmt::Debug>(values: &[T]) -> String {
    if values.len() <= PREVIEW_ELEMENTS {
        format!("{values:?}")
    } else {
        let head = &values[..PREVIEW_ELEMENTS];
        format!("{head:?}… ({} elements)", values.len())
    }
}
