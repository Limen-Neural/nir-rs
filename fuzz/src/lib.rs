// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared helpers for the `nir-rs` fuzz targets.

use nir_rs::io::wire::{check_hdf5_string, check_link_name};

const MAX_NAME_BYTES: usize = 128;

fn truncate_at_char_boundary(value: &str, max_bytes: usize) -> &str {
    let mut end = value.len().min(max_bytes);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

/// Exercises link-name and HDF5-string validation with a bounded lossy string.
pub fn check_link_name_input(data: &[u8]) {
    // Interpret as lossy UTF-8 so we cover NULs and path separators.
    let value = String::from_utf8_lossy(data);
    let name = truncate_at_char_boundary(&value, MAX_NAME_BYTES);
    let _ = check_link_name("node name", name);
    let _ = check_hdf5_string("metadata", name);
}

#[cfg(test)]
mod tests {
    use super::{MAX_NAME_BYTES, check_link_name_input, truncate_at_char_boundary};

    #[test]
    fn truncates_lossy_utf8_reproducer_at_char_boundary() {
        let mut data = b"\n\r\n".to_vec();
        data.extend(std::iter::repeat_n(0xe6, 42));

        let value = String::from_utf8_lossy(&data);
        let truncated = truncate_at_char_boundary(&value, MAX_NAME_BYTES);

        assert!(truncated.len() <= MAX_NAME_BYTES);
        assert!(value.is_char_boundary(truncated.len()));
        check_link_name_input(&data);
    }
}
