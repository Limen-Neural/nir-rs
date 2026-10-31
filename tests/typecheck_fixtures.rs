// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in type/shape checking against the vendored Python-written fixtures.
//!
//! Expected outcomes are Python `nir==1.0.8` `NIRGraph(type_check=True)`
//! results (LIM-1313): everything loads except `lif_rockpool.nir` (Output
//! `[1, 1, 1]` vs LIF `[1]`) and the `*_subgraph.nir` files (dangling inner
//! edge, which `validate_structure` already rejects).

#![cfg(feature = "hdf5")]

use nir_rs::io::{ReadOptions, read, read_with};
use nir_rs::{NirError, TypeError};

fn path(name: &str) -> String {
    format!("tests/fixtures/{name}")
}

#[test]
fn well_typed_paper_models_pass() {
    for name in [
        "lif_norse.nir",
        "two_lif_neurons.nir",
        "cnn_sinabs.nir",
        "braille_noDelay_bias_zero.nir",
        "braille_noDelay_noBias_subtract.nir",
    ] {
        let g = read(path(name)).unwrap();
        g.check_types()
            .unwrap_or_else(|e| panic!("{name} should pass: {e}"));
    }
}

#[test]
fn check_types_never_mutates_the_graph() {
    for name in ["cnn_sinabs.nir", "lif_norse.nir", "lif_rockpool.nir"] {
        let g = read(path(name)).unwrap();
        let before = g.clone();
        let _ = g.check_types();
        let _ = g.infer_types();
        assert_eq!(g, before, "{name}");
    }
}

#[test]
fn lif_rockpool_fails_naming_output_and_lif_shapes() {
    let g = read(path("lif_rockpool.nir")).unwrap();
    g.validate_structure().unwrap();
    let err = g.check_types().unwrap_err();
    let NirError::TypeCheck(inner) = &err else {
        panic!("expected TypeCheck, got {err:?}");
    };
    let TypeError::EdgeMismatch {
        src,
        src_type,
        dst_type,
        output,
        input,
        ..
    } = &**inner
    else {
        panic!("expected EdgeMismatch, got {inner:?}");
    };
    assert_eq!(src, "1_LIFNeuronTorch");
    assert_eq!(*src_type, "LIF");
    assert_eq!(*dst_type, "Output");
    assert_eq!(output, &vec![Some(vec![1])]);
    assert_eq!(input, &vec![Some(vec![1, 1, 1])]);
    let msg = err.to_string();
    assert!(msg.contains("(LIF).output [1]"), "{msg}");
    assert!(msg.contains("(Output).input [1, 1, 1]"), "{msg}");
}

#[test]
fn repair_of_a_well_typed_model_is_a_no_op() {
    let mut g = read(path("cnn_sinabs.nir")).unwrap();
    let before = g.clone();
    g.infer_and_repair().unwrap();
    assert_eq!(g, before);
}

#[test]
fn subgraph_dangling_edge_fails_via_validate_structure() {
    for name in [
        "braille_noDelay_bias_zero_subgraph.nir",
        "braille_noDelay_noBias_subtract_subgraph.nir",
    ] {
        let g = read(path(name)).unwrap();
        let structure = g.validate_structure().unwrap_err();
        assert_eq!(g.check_types().unwrap_err(), structure, "{name}");
        assert_eq!(g.infer_and_repair_clone().unwrap_err(), structure, "{name}");
    }
}

trait RepairClone {
    fn infer_and_repair_clone(&self) -> nir_rs::Result<()>;
}

impl RepairClone for nir_rs::NirGraph {
    fn infer_and_repair_clone(&self) -> nir_rs::Result<()> {
        self.clone().infer_and_repair()
    }
}

#[test]
fn default_read_stays_permissive() {
    assert!(read(path("lif_rockpool.nir")).is_ok());
    assert!(read(path("braille_noDelay_bias_zero_subgraph.nir")).is_ok());
    let opts = ReadOptions::default();
    assert!(!opts.check_types);
    assert!(read_with(path("lif_rockpool.nir"), &opts).is_ok());
}

#[test]
fn read_option_runs_the_checker_after_decode() {
    let strict = ReadOptions::default().with_check_types(true);
    assert!(read_with(path("cnn_sinabs.nir"), &strict).is_ok());
    assert!(read_with(path("lif_norse.nir"), &strict).is_ok());
    let Err(NirError::TypeCheck(inner)) = read_with(path("lif_rockpool.nir"), &strict) else {
        panic!("lif_rockpool must fail the opt-in check");
    };
    assert!(matches!(*inner, TypeError::EdgeMismatch { .. }));
    assert!(matches!(
        read_with(path("braille_noDelay_bias_zero_subgraph.nir"), &strict),
        Err(NirError::InvalidGraph(_) | NirError::MissingNode(_))
    ));
}
