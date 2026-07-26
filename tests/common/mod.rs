// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared helpers for the HDF5 failure-mode test binaries.
//!
//! Rust compiles each file in `tests/` into its own binary, so the error
//! suites are split by responsibility — reading malformed files, rejecting
//! bad graphs on write, and refusing files that reach outside the container —
//! and share this module rather than repeating the scaffolding.

#![cfg(feature = "hdf5")]
// Each test binary uses a different subset of these helpers.
#![allow(dead_code)]

use nir_rs::nodes::{Input, Output};
use nir_rs::{NirError, NirGraph, NirNode};
use tempfile::TempDir;

/// Assert that `result` failed with `expected_variant`, and that its message
/// mentions every one of `needles`.
///
/// `needles` is a slice rather than a single `&str` so that a test can demand
/// several things of one message, and so that "check the variant only" has to
/// be written as an explicit empty slice — an accidental `""` would otherwise
/// pass vacuously, since `str::contains("")` is always true.
pub fn assert_err<T>(
    result: Result<T, NirError>,
    expected_variant: fn(String) -> NirError,
    needles: &[&str],
) {
    let Err(err) = result else {
        panic!("expected an error, got Ok");
    };
    // Compare discriminants against a dummy so the variant is checked without
    // depending on the payload.
    let dummy = expected_variant(String::new());
    assert_eq!(
        std::mem::discriminant(&err),
        std::mem::discriminant(&dummy),
        "expected {dummy:?} variant, got {err:?}"
    );
    let msg = err.to_string();
    for needle in needles {
        assert!(
            !needle.is_empty(),
            "empty needle is vacuous; pass &[] to skip the message check"
        );
        assert!(
            msg.contains(needle),
            "expected message to contain {needle:?}, got {msg:?}"
        );
    }
}

pub fn input(shape: Vec<usize>) -> NirNode {
    NirNode::Input(Input {
        shape,
        metadata: Default::default(),
    })
}

/// Build a minimal well-formed file, then hand it to `mutate` for corruption.
pub fn write_then(
    dir: &TempDir,
    name: &str,
    mutate: impl FnOnce(&hdf5::File),
) -> std::path::PathBuf {
    let mut graph = NirGraph::new();
    graph.insert_node("input", input(vec![1])).unwrap();
    graph
        .insert_node(
            "output",
            NirNode::Output(Output {
                shape: vec![1],
                metadata: Default::default(),
            }),
        )
        .unwrap();
    graph.add_edge("input", "output");

    let path = dir.path().join(name);
    nir_rs::io::write(&path, &graph).unwrap();

    let file = hdf5::File::open_rw(&path).unwrap();
    mutate(&file);
    drop(file);
    path
}
