// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in `/version` compatibility policy on HDF5 read.
//!
//! Default [`nir_rs::io::read`] stays permissive. These cases cover missing,
//! malformed, accepted, and rejected envelopes, plus the early-fail path
//! shared with [`nir_rs::io::read_version_with`].

#![cfg(feature = "hdf5")]

mod common;
use common::write_then;
use hdf5::types::VarLenUnicode;
use nir_rs::NirError;
use nir_rs::io::{DEFAULT_NIR_VERSION, ReadOptions, VersionPolicy};
use tempfile::TempDir;

fn set_version(file: &hdf5::File, value: &str) {
    file.unlink("version").unwrap();
    let ds = file
        .new_dataset::<VarLenUnicode>()
        .shape(())
        .create("version")
        .unwrap();
    ds.write_scalar(&value.parse::<VarLenUnicode>().unwrap())
        .unwrap();
}

fn importer() -> ReadOptions {
    ReadOptions::default().with_version_policy(VersionPolicy::compatible_major([0, 1]))
}

fn assert_incompatible(err: NirError, expect_observed: Option<&str>, expect_policy: &str) {
    match err {
        NirError::IncompatibleVersion { observed, policy } => {
            assert_eq!(observed.as_deref(), expect_observed);
            assert_eq!(policy, expect_policy);
        }
        other => panic!("expected IncompatibleVersion, got {other:?}"),
    }
}

#[test]
fn default_read_still_accepts_a_missing_version() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "no_version.nir", |file| {
        file.unlink("version").unwrap();
    });

    let graph = nir_rs::io::read(&path).unwrap();
    assert_eq!(graph.version, None);
    assert_eq!(graph.len(), 2);
}

#[test]
fn require_present_rejects_a_missing_version() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "require_present_missing.nir", |file| {
        file.unlink("version").unwrap();
    });

    let opts = ReadOptions::default().with_version_policy(VersionPolicy::RequirePresent);
    assert_incompatible(
        nir_rs::io::read_with(&path, &opts).unwrap_err(),
        None,
        "require-present",
    );
    assert_incompatible(
        nir_rs::io::read_version_with(&path, &opts).unwrap_err(),
        None,
        "require-present",
    );
    // The historical strict accessor still uses MissingField under the default
    // (permissive) policy.
    assert_eq!(
        nir_rs::io::read_version(&path).unwrap_err(),
        NirError::MissingField("/version".into())
    );
}

#[test]
fn require_present_accepts_an_arbitrary_version_string() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "require_present_garbage.nir", |file| {
        set_version(file, "not-a-semver");
    });

    let opts = ReadOptions::default().with_version_policy(VersionPolicy::RequirePresent);
    let graph = nir_rs::io::read_with(&path, &opts).unwrap();
    assert_eq!(graph.version.as_deref(), Some("not-a-semver"));
    assert_eq!(
        nir_rs::io::read_version_with(&path, &opts).unwrap(),
        "not-a-semver"
    );
}

#[test]
fn compatible_major_accepts_paper_and_writer_versions() {
    let opts = importer();
    for (name, expected) in [
        ("tests/fixtures/lif_norse.nir", "0.1.1"),
        ("tests/fixtures/lif_rockpool.nir", "0.2.0"),
    ] {
        let graph = nir_rs::io::read_with(name, &opts).unwrap();
        assert_eq!(graph.version.as_deref(), Some(expected), "{name}");
        assert_eq!(
            nir_rs::io::read_version_with(name, &opts).unwrap(),
            expected
        );
        // Default read is unchanged.
        assert_eq!(
            nir_rs::io::read(name).unwrap().version.as_deref(),
            Some(expected)
        );
    }
}

#[test]
fn compatible_major_accepts_the_default_writer_version() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "writer_default.nir", |_| {});
    let opts = importer();
    let graph = nir_rs::io::read_with(&path, &opts).unwrap();
    assert_eq!(graph.version.as_deref(), Some(DEFAULT_NIR_VERSION));
    assert_eq!(
        nir_rs::io::read_version_with(&path, &opts).unwrap(),
        DEFAULT_NIR_VERSION
    );
}

#[test]
fn compatible_major_accepts_prerelease_and_build_version_suffixes() {
    let dir = TempDir::new().unwrap();
    let opts = importer();
    for value in ["1.0.0-rc.1", "1.0.0+build.5", "0.2.0-alpha.1+exp"] {
        let path = write_then(&dir, &format!("{value}.nir"), |file| {
            set_version(file, value);
        });
        let graph = nir_rs::io::read_with(&path, &opts).unwrap();
        assert_eq!(graph.version.as_deref(), Some(value));
        assert_eq!(nir_rs::io::read_version_with(&path, &opts).unwrap(), value);
    }
}

#[test]
fn compatible_major_rejects_missing_malformed_and_other_version_majors() {
    let dir = TempDir::new().unwrap();
    let opts = importer();
    let policy = "compatible-major majors=[0, 1]";

    let missing = write_then(&dir, "compat_missing.nir", |file| {
        file.unlink("version").unwrap();
    });
    assert_incompatible(
        nir_rs::io::read_with(&missing, &opts).unwrap_err(),
        None,
        policy,
    );
    assert_incompatible(
        nir_rs::io::read_version_with(&missing, &opts).unwrap_err(),
        None,
        policy,
    );

    for (name, value) in [
        ("malformed.nir", "not-a-semver"),
        ("incomplete.nir", "1.0"),
        ("v_prefix.nir", "v1.0.0"),
        ("other_major.nir", "2.0.0"),
        ("empty_prerelease.nir", "1.0.0-"),
    ] {
        let path = write_then(&dir, name, |file| set_version(file, value));
        assert_incompatible(
            nir_rs::io::read_with(&path, &opts).unwrap_err(),
            Some(value),
            policy,
        );
        assert_incompatible(
            nir_rs::io::read_version_with(&path, &opts).unwrap_err(),
            Some(value),
            policy,
        );
        // Permissive default still stores the string.
        assert_eq!(
            nir_rs::io::read(&path).unwrap().version.as_deref(),
            Some(value)
        );
        assert_eq!(nir_rs::io::read_version(&path).unwrap(), value);
    }
}

#[test]
fn rejected_version_fails_before_an_invalid_graph_body() {
    // Unknown node type + missing required field: if the body were decoded
    // first, the error would be UnknownNodeType / MissingField, not the
    // version policy.
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "bad_version_and_body.nir", |file| {
        set_version(file, "99.0.0");
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("type").unwrap();
        let ds = node
            .new_dataset::<VarLenUnicode>()
            .shape(())
            .create("type")
            .unwrap();
        ds.write_scalar(&"CurrLIF".parse::<VarLenUnicode>().unwrap())
            .unwrap();
        node.unlink("shape").unwrap();
    });

    assert_eq!(
        nir_rs::io::read(&path).unwrap_err(),
        NirError::UnknownNodeType("CurrLIF".into())
    );

    let opts = importer();
    assert_incompatible(
        nir_rs::io::read_with(&path, &opts).unwrap_err(),
        Some("99.0.0"),
        "compatible-major majors=[0, 1]",
    );
    assert_incompatible(
        nir_rs::io::read_version_with(&path, &opts).unwrap_err(),
        Some("99.0.0"),
        "compatible-major majors=[0, 1]",
    );
}

#[test]
fn rejected_version_fails_before_tensor_allocation() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "huge_body_bad_version.nir", |file| {
        set_version(file, "99.0.0");
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset::<i64>()
            .shape([101_000_000])
            .chunk([1024])
            .create("shape")
            .unwrap();
    });

    let opts = importer().with_max_bytes(Some(800_000_000));
    let err = nir_rs::io::read_with(&path, &opts).unwrap_err();
    assert_incompatible(err, Some("99.0.0"), "compatible-major majors=[0, 1]");
}

#[test]
fn malformed_version_dataset_is_still_io_under_any_policy() {
    let dir = TempDir::new().unwrap();
    let path = write_then(&dir, "version_group.nir", |file| {
        file.unlink("version").unwrap();
        file.create_group("version").unwrap();
    });

    let opts = importer();
    for result in [
        nir_rs::io::read(&path).map(|_| ()),
        nir_rs::io::read_with(&path, &opts).map(|_| ()),
        nir_rs::io::read_version(&path).map(|_| ()),
        nir_rs::io::read_version_with(&path, &opts).map(|_| ()),
    ] {
        match result {
            Err(NirError::Io(message)) => {
                assert!(message.contains("/version"), "{message}");
                assert!(message.contains("expected a dataset"), "{message}");
            }
            other => panic!("expected Io, got {other:?}"),
        }
    }
}
