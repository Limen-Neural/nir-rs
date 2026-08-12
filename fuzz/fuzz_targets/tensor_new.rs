// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz `Tensor::new` shape / data-length accounting.
//!
//! Structured input: up to 8 axis lengths + a data length. Never panics; every
//! result is either a tensor whose `numel()` matches `data.len()` or
//! `InvalidTensor`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use nir_rs::types::{Tensor, TensorData};
use nir_rs::NirError;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let rank = (data[0] as usize) % 9; // 0..=8
    if data.len() < 1 + rank {
        return;
    }
    // Map bytes to small dims so we exercise product / overflow without OOM.
    let shape: Vec<usize> = data[1..1 + rank]
        .iter()
        .map(|&b| match b % 8 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5 => 8,
            6 => 16,
            _ => 1usize << ((b % 12) + 8), // large-ish, may overflow product
        })
        .collect();
    let len = if data.len() > 1 + rank {
        (data[1 + rank] as usize) % 65
    } else {
        0
    };
    let payload = TensorData::F32(vec![0.0; len]);
    match Tensor::new(shape, payload) {
        Ok(t) => {
            assert_eq!(t.data().len(), t.numel());
        }
        Err(NirError::InvalidTensor(_)) => {}
        Err(other) => panic!("unexpected error: {other:?}"),
    }
});
