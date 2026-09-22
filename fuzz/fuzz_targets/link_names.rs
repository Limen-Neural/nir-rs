// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz HDF5 link-name / string preflight helpers (pure Rust, no libhdf5).

#![no_main]

use libfuzzer_sys::fuzz_target;
use nir_rs_fuzz::check_link_name_input;

fuzz_target!(|data: &[u8]| {
    check_link_name_input(data);
});
