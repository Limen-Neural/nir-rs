// SPDX-License-Identifier: MIT OR Apache-2.0

//! Byte-integrity checks for Hugging Face–derived `.nir` fixtures.
//!
//! SHA-256 values match `tests/fixtures/huggingface/MANIFEST.toml`. These
//! tests do not need libhdf5, so default-feature CI still catches a swapped
//! or truncated file.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const DIR: &str = "tests/fixtures/huggingface";

/// `(file name, lowercase hex SHA-256)` from [`DIR`]/MANIFEST.toml.
const EXPECTED: &[(&str, &str)] = &[
    (
        "neurocuda_mlp_mnist.nir",
        "fc0b1a1e0c4caeb9d1f7be8700de0212a76ec5f441cae13038887411fd9a1ef0",
    ),
    (
        "neurocuda_cnn_nmnist.nir",
        "972b45984094606b83b5b19173a653524550a5aa416f8a46398baa4250c34f2f",
    ),
];

fn sha256_hex(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn huggingface_fixtures_match_manifest_sha256() {
    for (name, expected) in EXPECTED {
        let path = Path::new(DIR).join(name);
        let actual = sha256_hex(&path);
        assert_eq!(
            actual, *expected,
            "{name} SHA-256 does not match MANIFEST.toml"
        );
    }
}
