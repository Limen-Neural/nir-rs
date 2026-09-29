// SPDX-License-Identifier: MIT OR Apache-2.0

//! Byte-integrity and record-completeness checks for the vendored Synfire
//! registry corpus (GitHub #43 / LIM-1085).
//!
//! Digests and provenance come from `tests/fixtures/synfire/MANIFEST.toml`, so
//! that file is the source of truth. These checks need no libhdf5, so
//! default-feature CI still catches a swapped, truncated, substituted, or
//! undeclared `.nir` file. Nothing here contacts the registry or authenticates.

mod common;

use common::synfire_manifest::{self, SynfireRecord, label};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const DIR: &str = "tests/fixtures/synfire";
const MANIFEST: &str = "tests/fixtures/synfire/MANIFEST.toml";

/// The five pinned models for the v0.4.5 baseline. Do not substitute versions
/// or add models (issue #43 body + user instruction are authoritative).
const PINNED: [(&str, &str); 5] = [
    ("pabogdan/lifneuron", "1.0.0"),
    ("pabogdan/ifsynfire", "0.1.0"),
    ("pabogdan/nmnistcnn", "1.0.0"),
    ("pabogdan/brailernn", "1.0.1"),
    ("jegp/swavelet", "1.0.0"),
];

fn sha256_hex(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn records() -> Vec<SynfireRecord> {
    let text = fs::read_to_string(MANIFEST).unwrap_or_else(|e| panic!("reading {MANIFEST}: {e}"));
    synfire_manifest::read_manifest(&text)
}

/// Sum of the `:<count>` suffixes in a sorted inventory list.
fn inventory_total(node_types: &[String]) -> usize {
    node_types
        .iter()
        .map(|entry| {
            let (_, count) = entry
                .rsplit_once(':')
                .unwrap_or_else(|| panic!("node_types entry {entry:?} is not \"<Type>:<count>\""));
            count
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("node_types entry {entry:?} has a non-integer count"))
        })
        .sum()
}

#[test]
fn manifest_records_exactly_the_five_pinned_models() {
    let recs = records();
    let got: Vec<(&str, &str)> = recs
        .iter()
        .map(|r| (r.synfire_model.as_str(), r.synfire_version.as_str()))
        .collect();
    assert_eq!(
        got,
        PINNED.to_vec(),
        "MANIFEST.toml must record exactly the five pinned (model, version) pairs, in order"
    );
}

#[test]
fn synfire_fixtures_match_manifest_sha256() {
    for rec in records() {
        let file = rec
            .file
            .as_ref()
            .unwrap_or_else(|| panic!("{} is missing a file key", label(&rec)));
        let digest = rec
            .sha256
            .as_ref()
            .unwrap_or_else(|| panic!("{} is missing a sha256 key", label(&rec)));
        let path = Path::new(DIR).join(file);
        let actual = sha256_hex(&path);
        assert_eq!(
            &actual,
            digest,
            "{} SHA-256 does not match MANIFEST.toml",
            label(&rec)
        );
    }
}

#[test]
fn every_record_has_provenance_fields() {
    for rec in records() {
        let l = label(&rec);
        assert!(rec.retrieved.is_some(), "{l} is missing a retrieved key");
        assert!(rec.license.is_some(), "{l} is missing a license key");
        assert!(
            rec.license_source.is_some(),
            "{l} is missing a license_source key"
        );
    }
}

#[test]
fn record_completeness_matches_status() {
    for rec in records() {
        let l = label(&rec);
        match rec.status.as_str() {
            "supported" => {
                let node_count = rec
                    .node_count
                    .unwrap_or_else(|| panic!("{l} supported record is missing node_count"));
                assert!(
                    rec.edge_count.is_some(),
                    "{l} supported record is missing edge_count"
                );
                assert!(
                    !rec.node_types.is_empty(),
                    "{l} supported record has empty node_types"
                );
                assert_eq!(
                    inventory_total(&rec.node_types),
                    node_count,
                    "{l} node_types counts must sum to node_count"
                );
            }
            // These branches will not fire in the current all-`supported`
            // corpus, but must exist so a future re-pull that records a
            // regression is checked, not silently accepted.
            "unsupported-valid" => {
                assert!(
                    rec.error_class.is_some(),
                    "{l} unsupported-valid record is missing error_class"
                );
                assert!(
                    !rec.follow_up.is_empty(),
                    "{l} unsupported-valid record is missing follow_up"
                );
                assert!(
                    rec.evidence.is_some(),
                    "{l} unsupported-valid record is missing evidence"
                );
            }
            "malformed" => {
                assert!(
                    rec.error_class.is_some(),
                    "{l} malformed record is missing error_class"
                );
                assert!(
                    !rec.follow_up.is_empty(),
                    "{l} malformed record is missing follow_up"
                );
            }
            "inaccessible" => {
                assert!(
                    rec.error_class.is_some(),
                    "{l} inaccessible record is missing error_class"
                );
                assert!(
                    !rec.follow_up.is_empty(),
                    "{l} inaccessible record is missing follow_up"
                );
                assert!(
                    rec.reason.is_some(),
                    "{l} inaccessible record is missing reason"
                );
            }
            other => panic!("{l} has an unknown status {other:?}"),
        }
    }
}

#[test]
fn no_orphan_nir_file_in_dir() {
    let declared: BTreeSet<String> = records().iter().filter_map(|r| r.file.clone()).collect();
    for entry in fs::read_dir(DIR).unwrap_or_else(|e| panic!("reading {DIR}: {e}")) {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".nir") {
            assert!(
                declared.contains(&name),
                "{name} is present in {DIR} but not declared in MANIFEST.toml"
            );
        }
    }
}
