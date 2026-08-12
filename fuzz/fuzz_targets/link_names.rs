// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz HDF5 link-name / string preflight helpers (pure Rust, no libhdf5).

#![no_main]

use libfuzzer_sys::fuzz_target;
use nir_rs::io::wire::{check_hdf5_string, check_link_name};

fuzz_target!(|data: &[u8]| {
    // Interpret as lossy UTF-8 so we cover NULs and path separators.
    let s = String::from_utf8_lossy(data);
    let name = if s.len() > 128 { &s[..128] } else { &s };
    let _ = check_link_name("node name", name);
    let _ = check_hdf5_string("metadata", name);
});
