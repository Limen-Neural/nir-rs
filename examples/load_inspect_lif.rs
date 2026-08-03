// SPDX-License-Identifier: MIT OR Apache-2.0

//! Load a NIR fixture, inspect every LIF parameter, save it, and verify the copy.

use nir_rs::io::DEFAULT_NIR_VERSION;
use nir_rs::types::{MetadataMap, MetadataValue, Tensor, TensorData};
use nir_rs::{NirGraph, NirNode, io};
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

    io::write(&output, &graph)?;
    let reloaded = io::read(&output)?;
    let mut expected = graph.clone();
    if expected.version.is_none() {
        expected.version = Some(DEFAULT_NIR_VERSION.to_owned());
    }

    // IEEE PartialEq: NaN != NaN. User models may contain NaNs; skip assert then.
    if graph_has_nan(&expected) {
        println!(
            "saved {} (skipped equality check: graph has NaN; IEEE PartialEq cannot verify)",
            output.display()
        );
    } else {
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
