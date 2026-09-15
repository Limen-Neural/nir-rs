// SPDX-License-Identifier: MIT OR Apache-2.0

//! Byte-integrity checks for Hugging Face–derived `.nir` fixtures.
//!
//! Digests are read from `tests/fixtures/huggingface/MANIFEST.toml` so that
//! file is the source of truth. These tests do not need libhdf5, so
//! default-feature CI still catches a swapped or truncated file.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const DIR: &str = "tests/fixtures/huggingface";
const MANIFEST: &str = "tests/fixtures/huggingface/MANIFEST.toml";

fn quoted_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key} = \"");
    line.trim().strip_prefix(&prefix)?.strip_suffix('"')
}

/// Parse `file` / `sha256` pairs from the catalog. Not a full TOML parser.
fn fixtures_from_manifest(text: &str) -> Vec<(&str, &str)> {
    let mut out = Vec::new();
    let mut file = None;
    for line in text.lines() {
        if let Some(name) = quoted_value(line, "file") {
            assert!(
                file.is_none(),
                "file {name} started before sha256 for the previous entry"
            );
            file = Some(name);
        } else if let Some(hash) = quoted_value(line, "sha256") {
            let name = file.take().expect("sha256 without a preceding file key");
            out.push((name, hash));
        }
    }
    if let Some(name) = file {
        panic!("file {name} is missing a sha256 key");
    }
    out
}

fn sha256_hex(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn huggingface_fixtures_match_manifest_sha256() {
    let text = fs::read_to_string(MANIFEST).unwrap();
    let expected = fixtures_from_manifest(&text);
    assert!(
        expected.len() >= 2,
        "MANIFEST.toml should list at least two fixtures, found {}",
        expected.len()
    );
    for (name, digest) in expected {
        let path = Path::new(DIR).join(name);
        let actual = sha256_hex(&path);
        assert_eq!(
            actual, digest,
            "{name} SHA-256 does not match MANIFEST.toml"
        );
    }
}
