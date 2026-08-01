// SPDX-License-Identifier: MIT OR Apache-2.0

//! Load a NIR fixture, inspect every LIF parameter, save it, and verify the copy.

use nir_rs::types::{Tensor, TensorData};
use nir_rs::{NirNode, io};
use nir_rs::io::DEFAULT_NIR_VERSION;
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

    for (name, node) in &graph.nodes {
        println!("node {name:?}: {}", node.type_name());
    }
    for (source, target) in &graph.edges {
        println!("edge {source:?} -> {target:?}");
    }

    let mut lif_count = 0;
    for (name, node) in &graph.nodes {
        if let NirNode::Lif(lif) = node {
            lif_count += 1;
            println!("LIF node {name:?} parameters:");
            print_tensor("tau", &lif.tau);
            print_tensor("r", &lif.r);
            print_tensor("v_leak", &lif.v_leak);
            print_tensor("v_threshold", &lif.v_threshold);
            match &lif.v_reset {
                Some(v_reset) => print_tensor("v_reset", v_reset),
                None => println!("  v_reset: <absent>"),
            }
        }
    }

    if lif_count == 0 {
        println!("no LIF nodes found");
    }

    io::write(&output, &graph)?;
    let reloaded = io::read(&output)?;
    // `PartialEq` on tensors is IEEE equality: graphs with NaN will not compare
    // equal to themselves. The vendored LIF fixture is finite-only, which is
    // the intended path for this demo.
    let mut expected = graph.clone();
    if expected.version.is_none() {
        expected.version = Some(DEFAULT_NIR_VERSION.to_owned());
    }
    assert_eq!(
        reloaded, expected,
        "saved graph did not round-trip exactly (finite values only)"
    );
    println!("saved and verified {}", output.display());

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
