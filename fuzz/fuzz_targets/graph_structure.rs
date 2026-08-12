// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz graph insert / edge / validate_structure for panic freedom.
//!
//! Byte layout:
//! - byte 0: node count `n` in 1..=16
//! - following pairs of bytes: edge endpoints mod `n`
//!
//! Must never panic; `validate_structure` returns Ok or a documented error.

#![no_main]

use libfuzzer_sys::fuzz_target;
use nir_rs::nodes::Input;
use nir_rs::{NirError, NirGraph, NirNode};

fn input() -> NirNode {
    NirNode::Input(Input {
        shape: vec![1],
        metadata: Default::default(),
    })
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let n = (data[0] as usize % 16).max(1);
    let mut g = NirGraph::new();
    for i in 0..n {
        g.insert_node(format!("n{i}"), input()).unwrap();
    }
    // Duplicate insert must fail closed.
    let err = g.insert_node("n0", input()).unwrap_err();
    assert!(matches!(err, NirError::DuplicateNode(_)));

    for chunk in data[1..].chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let a = chunk[0] as usize % n;
        let b = chunk[1] as usize % n;
        g.add_edge(format!("n{a}"), format!("n{b}"));
    }

    // Maybe inject a ghost endpoint.
    if data.len() > 3 && data[1] % 5 == 0 {
        g.add_edge("n0", "ghost");
    }

    match g.validate_structure() {
        Ok(()) => {}
        Err(NirError::MissingNode(_)) | Err(NirError::DuplicateEdge(_, _)) => {}
        Err(other) => panic!("unexpected validation error: {other:?}"),
    }
});
