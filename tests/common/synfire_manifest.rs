// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-written reader + helpers for `tests/fixtures/synfire/MANIFEST.toml`.
//!
//! The Synfire manifest is the provenance source of truth for the vendored
//! registry corpus (see GitHub #43 / LIM-1085). It is deliberately parsed by
//! a small hand-written reader rather than a TOML crate: `AGENTS.md` forbids
//! adding dependencies, so this mirrors the `quoted_value` line-parser idiom
//! from `tests/hf_fixture_checksums.rs`.
//!
//! Every record field is observed by loading the committed `.nir` bytes
//! through `nir_rs::io::read` + `validate_structure()`; nothing here contacts
//! the registry or authenticates. The helpers ([`label`],
//! [`nir_error_variant_name`], [`inventory`]) use only the always-compiled
//! graph model, so this module needs no `hdf5` feature gate.

use nir_rs::error::NirError;
use nir_rs::graph::NirGraph;

/// One `[[fixture]]` block from the manifest.
///
/// String keys that every record carries (`synfire_model`, `synfire_version`,
/// `status`) are non-optional; the rest are optional because the four-way
/// classification scheme (supported / unsupported-valid / malformed /
/// inaccessible) populates different subsets. All five current fixtures are
/// `supported`, but the optional error/evidence fields exist so a future
/// re-pull can record a regression without a schema change.
#[derive(Debug, Clone, Default)]
pub struct SynfireRecord {
    /// `<org>/<model>` registry identifier (e.g. `pabogdan/lifneuron`).
    pub synfire_model: String,
    /// Registry release string (e.g. `1.0.0`).
    pub synfire_version: String,
    /// Classification: `supported`, `unsupported-valid`, `malformed`, or
    /// `inaccessible`.
    pub status: String,
    /// Vendored file name relative to the fixture dir (committed models only).
    pub file: Option<String>,
    /// SHA-256 of the committed bytes (committed models only).
    pub sha256: Option<String>,
    /// Embedded `/version` string observed on load.
    pub nir_version: Option<String>,
    /// Observed node count.
    pub node_count: Option<usize>,
    /// Observed edge count.
    pub edge_count: Option<usize>,
    /// Sorted `"<WireType>:<count>"` inventory of the root graph.
    pub node_types: Vec<String>,
    /// Names of nested subgraphs, if any (none in the current corpus).
    pub nested: Vec<String>,
    /// Sorted inventory of each nested subgraph, if any.
    pub nested_node_types: Vec<String>,
    /// Retrieval date (UTC).
    pub retrieved: Option<String>,
    /// Declared license identifier.
    pub license: Option<String>,
    /// Where the license was declared (e.g. `nir-card.json`).
    pub license_source: Option<String>,
    /// Classification error class for non-`supported` records.
    pub error_class: Option<String>,
    /// Substring expected in the read/validate error (unsupported-valid /
    /// malformed).
    pub error_fragment: Option<String>,
    /// Evidence recorded for an `unsupported-valid` classification.
    pub evidence: Option<String>,
    /// Reason recorded for an `inaccessible` classification.
    pub reason: Option<String>,
    /// Follow-up issue references for non-`supported` records.
    pub follow_up: Vec<String>,
}

/// Extract the payload of a single-line `key = "value"` pair.
///
/// Mirrors `quoted_value` in `tests/hf_fixture_checksums.rs`. Not a full TOML
/// parser: it only understands one double-quoted value on the line.
fn quoted_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key} = \"");
    line.trim().strip_prefix(&prefix)?.strip_suffix('"')
}

/// Extract a `key = <int>` value (unquoted integer).
fn int_value(line: &str, key: &str) -> Option<usize> {
    let prefix = format!("{key} = ");
    let rest = line.trim().strip_prefix(&prefix)?;
    rest.parse().ok()
}

/// Extract a `key = true|false` value.
fn bool_value(line: &str, key: &str) -> Option<bool> {
    let prefix = format!("{key} = ");
    match line.trim().strip_prefix(&prefix)? {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Extract a single-line `key = ["a", "b", ...]` list of double-quoted items.
///
/// Returns `None` when the key is not on this line; returns an empty vec for
/// `key = []`. Only handles the single-line array form the manifest uses.
fn list_value(line: &str, key: &str) -> Option<Vec<String>> {
    let prefix = format!("{key} = [");
    let inner = line.trim().strip_prefix(&prefix)?.strip_suffix(']')?;
    let inner = inner.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    for item in inner.split(',') {
        let item = item.trim();
        let item = item
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or_else(|| panic!("list item {item:?} for key {key} is not double-quoted"));
        out.push(item.to_owned());
    }
    Some(out)
}

/// Plain `key = "value"` string fields, keyed by manifest name to the record
/// slot each one fills.
///
/// Order matters: `license_source` precedes `license` and `nested_node_types`
/// precedes `nested` because each pair shares a prefix and [`quoted_value`]
/// matches on `"{key} = \""`, so the longer, more specific key must be tried
/// first (the same precedence the original `else if` chain relied on).
type StringField = (&'static str, fn(&mut SynfireRecord, String));

const STRING_FIELDS: &[StringField] = &[
    ("synfire_model", |rec, v| rec.synfire_model = v),
    ("synfire_version", |rec, v| rec.synfire_version = v),
    ("status", |rec, v| rec.status = v),
    ("file", |rec, v| rec.file = Some(v)),
    ("sha256", |rec, v| rec.sha256 = Some(v)),
    ("nir_version", |rec, v| rec.nir_version = Some(v)),
    ("retrieved", |rec, v| rec.retrieved = Some(v)),
    // `license_source` before `license`: shared `license` prefix.
    ("license_source", |rec, v| rec.license_source = Some(v)),
    ("license", |rec, v| rec.license = Some(v)),
    ("error_class", |rec, v| rec.error_class = Some(v)),
    ("error_fragment", |rec, v| rec.error_fragment = Some(v)),
    ("evidence", |rec, v| rec.evidence = Some(v)),
    ("reason", |rec, v| rec.reason = Some(v)),
];

/// Single-line `key = ["a", ...]` list fields, keyed by manifest name.
///
/// `nested_node_types` precedes `nested` for the same shared-prefix reason as
/// the string fields above.
type ListField = (&'static str, fn(&mut SynfireRecord, Vec<String>));

const LIST_FIELDS: &[ListField] = &[
    ("node_types", |rec, v| rec.node_types = v),
    // `nested_node_types` before `nested`: shared `nested` prefix.
    ("nested_node_types", |rec, v| rec.nested_node_types = v),
    ("nested", |rec, v| rec.nested = v),
    ("follow_up", |rec, v| rec.follow_up = v),
];

/// Apply a single `key = value` manifest line to the record under construction.
///
/// Tries each recognized key in the same precedence order as the original
/// `else if` dispatch chain: string fields, then the two integer counts, then
/// the list fields. Unrecognized keys (`registry_url`, `producer`, `covers`,
/// `auth_required`, ...) are provenance-only and left untouched.
fn apply_field(rec: &mut SynfireRecord, line: &str) {
    for (key, set) in STRING_FIELDS {
        if let Some(v) = quoted_value(line, key) {
            set(rec, v.to_owned());
            return;
        }
    }
    if let Some(v) = int_value(line, "node_count") {
        rec.node_count = Some(v);
        return;
    }
    if let Some(v) = int_value(line, "edge_count") {
        rec.edge_count = Some(v);
        return;
    }
    for (key, set) in LIST_FIELDS {
        if let Some(v) = list_value(line, key) {
            set(rec, v);
            return;
        }
    }
}

/// Parse the manifest text into one [`SynfireRecord`] per `[[fixture]]` block.
///
/// This is not a general TOML parser: it splits on `[[fixture]]` header lines
/// and reads one `key = value` pair per line inside each block. File-level
/// keys above the first `[[fixture]]` header are ignored.
#[must_use]
pub fn read_manifest(text: &str) -> Vec<SynfireRecord> {
    let mut records = Vec::new();
    let mut current: Option<SynfireRecord> = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "[[fixture]]" {
            if let Some(rec) = current.take() {
                records.push(rec);
            }
            current = Some(SynfireRecord::default());
            continue;
        }
        let Some(rec) = current.as_mut() else {
            // Still in the file-level header; skip.
            continue;
        };
        apply_field(rec, trimmed);
    }
    if let Some(rec) = current.take() {
        records.push(rec);
    }
    records
}

/// Human label prefixing every Synfire test assertion:
/// `"<org>/<model>:<version> [<status>]"`.
///
/// `synfire_model` already holds `<org>/<model>`.
#[must_use]
pub fn label(rec: &SynfireRecord) -> String {
    format!(
        "{}:{} [{}]",
        rec.synfire_model, rec.synfire_version, rec.status
    )
}

/// Stable variant name for a [`NirError`], for error-class assertions.
///
/// Matches every variant explicitly so a future enum change is a visible diff
/// here. `NirError` is `#[non_exhaustive]` outside its crate, so a wildcard arm
/// is required to compile; it panics rather than inventing a name, which keeps
/// the mapping honest if a new variant ever reaches a test.
#[must_use]
pub fn nir_error_variant_name(err: &NirError) -> &'static str {
    match err {
        NirError::Unimplemented(_) => "Unimplemented",
        NirError::UnknownNodeType(_) => "UnknownNodeType",
        NirError::DuplicateNode(_) => "DuplicateNode",
        NirError::MissingNode(_) => "MissingNode",
        NirError::DuplicateEdge(_, _) => "DuplicateEdge",
        NirError::InvalidGraph(_) => "InvalidGraph",
        NirError::UnsupportedVersion(_) => "UnsupportedVersion",
        NirError::IncompatibleVersion { .. } => "IncompatibleVersion",
        NirError::MissingField(_) => "MissingField",
        NirError::InvalidTensor(_) => "InvalidTensor",
        NirError::InvalidNodeParameters { .. } => "InvalidNodeParameters",
        NirError::ReadLimitExceeded { .. } => "ReadLimitExceeded",
        NirError::ReadCountLimitExceeded { .. } => "ReadCountLimitExceeded",
        NirError::Io(_) => "Io",
        other => panic!("unmapped NirError variant: {other:?}"),
    }
}

/// Sorted `"<WireType>:<count>"` inventory of a graph's nodes.
///
/// Uses [`nir_rs::NirNode::type_name`] as the wire mapping and counts every
/// node in the graph. The sum of the `:<count>` values therefore equals
/// `graph.nodes.len()`, which the checksum test cross-checks against the
/// manifest `node_count`.
#[must_use]
pub fn inventory(graph: &NirGraph) -> Vec<String> {
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for node in graph.nodes.values() {
        *counts.entry(node.type_name()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(ty, count)| format!("{ty}:{count}"))
        .collect()
}
