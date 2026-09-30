// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guards against `.nir` files that try to reach outside their own container.

#![cfg(feature = "hdf5")]

mod common;
use common::{assert_err, decode_hex_fixture, write_then};
use nir_rs::NirError;
use nir_rs::io::ReadOptions;
use tempfile::TempDir;

//
// HDF5 resolves external links, external raw storage and virtual-dataset
// sources against the host filesystem, which is how a crafted model file turns
// a reader into an arbitrary-file-read primitive (the Keras advisories
// GHSA-3m4q-jmj6-r34q / CVE-2026-12480 are exactly this). `read` rejects all
// three. The cases below construct each one so the guard is actually executed
// rather than merely present.

/// Build a well-formed file, then let `mutate` add something hostile to it.
fn tampered(dir: &TempDir, name: &str, mutate: impl FnOnce(&hdf5::File)) -> std::path::PathBuf {
    write_then(dir, name, mutate)
}

fn bounded(max_bytes: usize) -> ReadOptions {
    ReadOptions::default().with_max_bytes(Some(max_bytes))
}

fn filter_dcpl(filter_id: i32, values: &[u32]) -> hdf5::plist::DatasetCreate {
    let dcpl = hdf5::plist::DatasetCreate::build()
        .chunk([1])
        .finish()
        .unwrap();
    let status = hdf5::sync::sync(|| {
        // SAFETY: `dcpl` owns a live dataset-creation property list and
        // `values` remains alive for the call. HDF5 access is serialized by
        // `sync`.
        unsafe {
            hdf5_sys::h5p::H5Pset_filter(
                dcpl.id(),
                filter_id,
                hdf5_sys::h5z::H5Z_FLAG_OPTIONAL,
                values.len(),
                values.as_ptr(),
            )
        }
    });
    hdf5::h5check(status).unwrap();
    dcpl
}

fn user_filter_dcpl(values: &[u32]) -> hdf5::plist::DatasetCreate {
    filter_dcpl(32_000, values)
}

fn replace_shape_with_raw_dcpl(file: &hdf5::File, dcpl: &hdf5::plist::DatasetCreate) {
    let node = file.group("node/nodes/input").unwrap();
    let original = node.dataset("shape").unwrap();
    let dtype = original.dtype().unwrap();
    let space = original.space().unwrap();
    node.unlink("shape").unwrap();
    let name = std::ffi::CString::new("shape").unwrap();
    let dataset_id = hdf5::sync::sync(|| {
        // SAFETY: all HDF5 objects own live IDs and `name` remains alive
        // throughout the call. The dataset is closed below, and `sync`
        // serializes calls with hdf5-metno. Using the C API here avoids
        // the upstream builder's own 32-parameter panic during setup.
        unsafe {
            hdf5_sys::h5d::H5Dcreate2(
                node.id(),
                name.as_ptr(),
                dtype.id(),
                space.id(),
                hdf5_sys::h5p::H5P_DEFAULT,
                dcpl.id(),
                hdf5_sys::h5p::H5P_DEFAULT,
            )
        }
    });
    let dataset_id = hdf5::h5check(dataset_id).unwrap();
    let close_status = hdf5::sync::sync(|| {
        // SAFETY: `dataset_id` is the live ID returned by H5Dcreate2 and
        // is closed exactly once while the HDF5 global lock is held.
        unsafe { hdf5_sys::h5d::H5Dclose(dataset_id) }
    });
    hdf5::h5check(close_status).unwrap();
}

#[test]
fn nbit_filter_is_rejected_before_tensor_decode() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "nbit_shape.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset_builder()
            .with_data(&[1_i64])
            .nbit()
            .create("shape")
            .unwrap();
    });

    for options in [ReadOptions::default(), bounded(1_000_000)] {
        assert_err(
            nir_rs::io::read_with(&path, &options),
            NirError::InvalidGraph,
            &["input.shape", "NBit"],
        );
    }
}

#[test]
fn user_filter_is_rejected_before_tensor_decode() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "user_filter_shape.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();

        let dcpl = user_filter_dcpl(&[]);

        node.new_dataset_builder()
            .set_dcpl(&dcpl)
            .empty::<i64>()
            .shape([1])
            .create("shape")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["input.shape", "filter id 32000"],
    );
}

#[test]
fn oversized_user_filter_parameters_cannot_panic_the_reader() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "user_filter_many_parameters.nir", |file| {
        let dcpl = user_filter_dcpl(&[7; 33]);
        replace_shape_with_raw_dcpl(file, &dcpl);
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["input.shape", "filter"],
    );
}

#[test]
fn allowed_filter_with_oversized_parameters_cannot_panic_the_reader() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "deflate_many_parameters.nir", |file| {
        let dcpl = filter_dcpl(hdf5_sys::h5z::H5Z_FILTER_DEFLATE, &[7; 33]);
        replace_shape_with_raw_dcpl(file, &dcpl);
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["input.shape", "at most 32"],
    );
}

#[test]
fn user_filter_on_version_is_rejected_by_both_readers() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "user_filter_version.nir", |file| {
        file.unlink("version").unwrap();
        let dcpl = user_filter_dcpl(&[]);
        file.new_dataset_builder()
            .set_dcpl(&dcpl)
            .empty::<hdf5::types::FixedAscii<8>>()
            .shape([1])
            .create("version")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["version", "filter id 32000"],
    );
    assert_err(
        nir_rs::io::read_version(&path),
        NirError::InvalidGraph,
        &["version", "filter id 32000"],
    );
}

#[test]
fn non_allowlisted_metadata_filter_is_rejected() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "nbit_metadata.nir", |file| {
        let metadata = file
            .group("node")
            .unwrap()
            .create_group("metadata")
            .unwrap();
        metadata
            .new_dataset_builder()
            .with_data(&[7_i64])
            .nbit()
            .create("untrusted")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["metadata.untrusted", "NBit"],
    );
}

#[test]
fn built_in_filter_combination_remains_readable() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "allowed_filters.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset_builder()
            .with_data(&[1_i64])
            .shuffle()
            .deflate(4)
            .fletcher32()
            .create("shape")
            .unwrap();
    });

    let graph = nir_rs::io::read(&path).unwrap();
    assert_eq!(graph.nodes.len(), 2);
    graph.validate_structure().unwrap();
}

fn assert_limit<T: std::fmt::Debug>(result: Result<T, NirError>, max_bytes: usize) -> NirError {
    let err = result.expect_err("bounded read should exceed its allocation budget");
    match &err {
        NirError::ReadLimitExceeded {
            limit,
            used,
            requested,
            ..
        } => {
            assert_eq!(*limit, max_bytes);
            assert!(*requested > 0);
            assert!(
                used.checked_add(*requested)
                    .is_none_or(|total| total > *limit)
            );
        }
        other => panic!("expected ReadLimitExceeded, got {other:?}"),
    }
    err
}

/// A second HDF5 file standing in for the attacker's target.
fn decoy(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("decoy.h5");
    let file = hdf5::File::create(&path).unwrap();
    let ds = file
        .new_dataset::<f32>()
        .shape([2])
        .create("secret")
        .unwrap();
    ds.write_raw(&[1.0f32, 2.0]).unwrap();
    path
}

/// Run one external-link case: the decoy's `/secret` is linked into the
/// container at `parent`/`link_name`, and reading the graph must refuse it.
fn assert_external_link_rejected(parent: &str, name: &str, link_name: &str, needles: &[&str]) {
    let dir = TempDir::new().unwrap();
    let target = decoy(&dir);
    let path = tampered(&dir, name, |file| {
        file.group(parent)
            .unwrap()
            .link_external(target.to_str().unwrap(), "/secret", link_name)
            .unwrap();
    });

    assert_err(nir_rs::io::read(&path), NirError::InvalidGraph, needles);
}

#[test]
fn external_link_in_the_file_root_is_rejected() {
    assert_external_link_rejected(
        "/",
        "external_root.nir",
        "elsewhere",
        &["external link", "elsewhere"],
    );
}

#[test]
fn external_node_type_is_rejected_before_it_is_read() {
    // `/node/type` is opened by `read` itself, before `read_graph_body` gets a
    // chance to validate the `/node` group. The decoy's `type` holds a value the
    // type check would reject, so the *message* proves the ordering: if the
    // external link were followed first, this would fail with "must be a
    // NIRGraph, found \"NotAGraph\"" — i.e. after the external file had already
    // been opened and read.
    let dir = TempDir::new().unwrap();
    let target = dir.path().join("type_decoy.h5");
    {
        let file = hdf5::File::create(&target).unwrap();
        let ds = file
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("type")
            .unwrap();
        ds.write_scalar(&"NotAGraph".parse::<hdf5::types::VarLenUnicode>().unwrap())
            .unwrap();
    }

    let path = tampered(&dir, "external_type.nir", |file| {
        let node = file.group("node").unwrap();
        node.unlink("type").unwrap();
        node.link_external(target.to_str().unwrap(), "/type", "type")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["external link", "type"],
    );
}

#[test]
fn external_link_inside_a_node_group_is_rejected() {
    assert_external_link_rejected(
        "node/nodes/input",
        "external_node.nir",
        "shape_alias",
        &["external link"],
    );
}

#[test]
fn external_storage_dataset_is_rejected() {
    let dir = TempDir::new().unwrap();
    // Raw data held in a plain file outside the container.
    let raw = dir.path().join("raw.bin");
    std::fs::write(&raw, vec![0u8; 8]).unwrap();

    let path = tampered(&dir, "external_storage.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset::<i64>()
            .external(raw.to_str().unwrap(), 0, 8)
            .shape([1])
            .create("shape")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["external storage"],
    );
}

#[test]
fn virtual_dataset_is_rejected() {
    // The VDS bypass: a guard that only checks external storage misses this.
    let dir = TempDir::new().unwrap();
    let target = decoy(&dir);

    let path = tampered(&dir, "virtual.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset::<f32>()
            .virtual_map(target.to_str().unwrap(), "/secret", 2, .., 2, ..)
            .shape([2])
            .create("shape")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["virtual dataset"],
    );
}

#[test]
fn a_virtual_string_dataset_is_rejected_too() {
    // String datasets go through a different reader, so `/version` and the
    // `type` fields need the same guard as tensors.
    let dir = TempDir::new().unwrap();
    let target = dir.path().join("decoy_strings.h5");
    {
        let file = hdf5::File::create(&target).unwrap();
        let ds = file
            .new_dataset::<f32>()
            .shape([1])
            .create("secret")
            .unwrap();
        ds.write_raw(&[1.0f32]).unwrap();
    }

    let path = tampered(&dir, "virtual_string.nir", |file| {
        file.unlink("version").unwrap();
        file.new_dataset::<f32>()
            .virtual_map(target.to_str().unwrap(), "/secret", 1, .., 1, ..)
            .shape([1])
            .create("version")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read_version(&path),
        NirError::InvalidGraph,
        &["virtual dataset"],
    );
}

#[test]
fn soft_link_is_rejected() {
    // Soft links can resolve to external targets, so they are banned at the
    // same gate as external links. A sibling alias under `nodes/` is the
    // realistic place for one to appear.
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "soft_link.nir", |file| {
        file.group("node/nodes")
            .unwrap()
            .link_soft("input", "alias")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["soft link", "alias"],
    );
}

#[test]
fn oversized_dataset_is_rejected() {
    // Explicit element×decoded-width product above the caller's 800 MB budget.
    // 101M i64 values would decode to ~808 MB; creating the dataspace alone is
    // enough — rejection happens before `read_raw`. Complements the
    // fixed-string / narrow-integer cases that catch under-counting modes.
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "oversized.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset::<i64>()
            .shape([101_000_000])
            .chunk([1024])
            .create("shape")
            .unwrap();
    });

    let err = assert_limit(
        nir_rs::io::read_with(&path, &bounded(800_000_000)),
        800_000_000,
    );
    assert!(err.to_string().contains("input.shape"));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn overflowing_dataset_extents_are_rejected_before_reading() {
    // The declared element count is 2^64, but the chunked dataset has no
    // allocated chunks. The file remains small and `read_raw` must never be
    // reached. In particular, do not call `Dataset::size` here: its unchecked
    // extent product is the behavior this regression exercises through the
    // public reader.
    let dir = TempDir::new().unwrap();
    let path = decode_hex_fixture(
        &dir,
        "tests/fixtures/overflowing_extents.nir.hex",
        "overflowing_extents.nir",
    );

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidTensor,
        &["input.shape", "shape product overflows usize"],
    );

    let err = assert_limit(
        nir_rs::io::read_with(&path, &bounded(usize::MAX)),
        usize::MAX,
    );
    assert!(err.to_string().contains("input.shape"));
}

#[test]
fn ordinary_files_still_pass_the_guards() {
    // The guards must not reject the real thing.
    for name in ["lif_norse.nir", "cnn_sinabs.nir"] {
        let path = format!("tests/fixtures/{name}");
        nir_rs::io::read(&path).unwrap_or_else(|e| panic!("{name} should still read: {e}"));
        nir_rs::io::read_with(&path, &bounded(1_000_000_000))
            .unwrap_or_else(|e| panic!("{name} should fit the generous budget: {e}"));
    }
}

#[test]
fn a_wide_fixed_string_dataset_is_rejected_on_bytes_not_element_count() {
    // 200K elements is far below any plausible element-count cap, but each is
    // read through the capacity ladder as `FixedAscii<4096>`, so decoding
    // would materialize ~4 GB. A guard that counts elements accepts this; one
    // that charges bytes does not. No data is written, and the rejection
    // happens before `read_raw`, so the test never allocates it.
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "wide_strings.nir", |file| {
        let root = file.group("node").unwrap();
        root.unlink("edges").unwrap();
        root.new_dataset::<hdf5::types::FixedAscii<4096>>()
            .shape([100_000, 2])
            .chunk([16, 2])
            .create("edges")
            .unwrap();
    });

    let err = assert_limit(
        nir_rs::io::read_with(&path, &bounded(800_000_000)),
        800_000_000,
    );
    assert!(err.to_string().contains("edges"));
}

#[test]
fn a_narrow_integer_dataset_is_charged_at_its_decoded_width() {
    // `read_tensor` materializes every integer width as `i64`, so a 1-byte
    // dataset costs eight times its stored size. 150M `i8` elements is 150 MB
    // on the descriptor but 1.2 GB decoded: charging the source width accepts
    // it, charging the destination width does not.
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "narrow_ints.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset::<i8>()
            .shape([150_000_000])
            .chunk([4096])
            .create("shape")
            .unwrap();
    });

    let err = assert_limit(
        nir_rs::io::read_with(&path, &bounded(800_000_000)),
        800_000_000,
    );
    assert!(err.to_string().contains("input.shape"));
}

#[test]
fn cumulative_budget_is_shared_across_datasets() {
    // Two equal shape payloads under a limit that cannot cover both (plus the
    // fixed graph overhead). The shared `ReadBudget` must reject the read
    // rather than allocating both shapes unbounded.
    //
    // On HDF5 builds where scalar VLEN sizing falls back to the containing
    // file size, the first over-budget charge can be an early type/version
    // string with `used == 0`. That is still a correct rejection; when the
    // overrun happens later, `used > 0` proves earlier charges applied.
    const N: usize = 16;
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "cumulative.nir", |file| {
        for node_name in ["input", "output"] {
            let node = file.group(&format!("node/nodes/{node_name}")).unwrap();
            node.unlink("shape").unwrap();
            let ds = node
                .new_dataset::<i64>()
                .shape([N])
                .create("shape")
                .unwrap();
            ds.write_raw(&[1_i64; N]).unwrap();
        }
    });

    let known_data_size = 2 * N * std::mem::size_of::<i64>();
    let limit = known_data_size + known_data_size / 2;

    let err = assert_limit(nir_rs::io::read_with(&path, &bounded(limit)), limit);
    let NirError::ReadLimitExceeded {
        used,
        requested,
        limit: reported_limit,
        ..
    } = err
    else {
        unreachable!()
    };
    assert_eq!(reported_limit, limit);
    assert!(requested > 0);
    // Prefer the stronger cumulative signal when available.
    if used > 0 {
        assert!(
            used.saturating_add(requested) > limit || used.checked_add(requested).is_none(),
            "cumulative overrun expected: used={used} requested={requested} limit={limit}"
        );
    }
}

#[test]
fn variable_length_payload_is_charged_before_read() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "large_version.nir", |file| {
        file.unlink("version").unwrap();
        let ds = file
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("version")
            .unwrap();
        let payload = "v".repeat(16_384);
        ds.write_scalar(&payload.parse::<hdf5::types::VarLenUnicode>().unwrap())
            .unwrap();
    });

    let limit = 4096;
    let err = assert_limit(nir_rs::io::read_version_with(&path, &bounded(limit)), limit);
    assert!(err.to_string().contains("version"));

    // The compatibility entry point remains intentionally unbounded.
    assert_eq!(nir_rs::io::read_version(&path).unwrap(), "v".repeat(16_384));
}

#[test]
fn ascii_version_charges_both_payload_copies() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "ascii_version.nir", |file| {
        file.unlink("version").unwrap();
        let value = hdf5::types::VarLenAscii::from_ascii("1.0.0").unwrap();
        file.new_dataset_builder()
            .with_data(&[value])
            .create("version")
            .unwrap();
    });

    // One HDF5 descriptor, one Rust String header, and two payload charges.
    // HDF5 reports six bytes including the trailing NUL; the Rust copy is
    // conservatively charged at that same size.
    let bytes = size_of::<hdf5::types::VarLenAscii>() + size_of::<String>() + 12;
    assert_eq!(nir_rs::io::read_version(&path).unwrap(), "1.0.0");
    assert_eq!(
        nir_rs::io::read_version_with(&path, &bounded(bytes)).unwrap(),
        "1.0.0"
    );
    let err = assert_limit(
        nir_rs::io::read_version_with(&path, &bounded(bytes - 1)),
        bytes - 1,
    );
    assert!(err.to_string().contains("version"));
}

#[test]
fn numeric_version_is_rejected_as_a_non_string() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "numeric_version.nir", |file| {
        file.unlink("version").unwrap();
        file.new_dataset_builder()
            .with_data(&[1_i32])
            .create("version")
            .unwrap();
    });

    for options in [ReadOptions::default(), bounded(1024)] {
        assert_err(
            nir_rs::io::read_version_with(&path, &options),
            NirError::Io,
            &["version", "expected a string dataset"],
        );
    }
}

#[test]
fn u64_shape_is_decoded_losslessly() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "u64_shape.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        node.new_dataset_builder()
            .with_data(&[7_u64])
            .create("shape")
            .unwrap();
    });

    for options in [ReadOptions::default(), bounded(1_000_000)] {
        let graph = nir_rs::io::read_with(&path, &options).unwrap();
        let Some(nir_rs::NirNode::Input(input)) = graph.nodes.get("input") else {
            panic!("expected an Input node");
        };
        assert_eq!(input.shape, vec![7]);
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn u64_combined_buffers_overflow_is_rejected_before_reading() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "u64_allocation_overflow.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        // Each eight-byte buffer fits in usize, but their combined charge
        // does not. Leave all chunks unallocated so the fixture stays small.
        node.new_dataset::<u64>()
            .shape([1_usize << 60])
            .chunk([1])
            .create("shape")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidTensor,
        &["input.shape", "decoded allocation size overflows usize"],
    );
    let err = assert_limit(
        nir_rs::io::read_with(&path, &bounded(usize::MAX)),
        usize::MAX,
    );
    assert!(err.to_string().contains("input.shape"));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn rust_object_size_limit_is_checked_before_reading() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "invalid_vec_layout.nir", |file| {
        let node = file.group("node/nodes/input").unwrap();
        node.unlink("shape").unwrap();
        // The byte count fits usize but cannot be represented by a Rust
        // allocation layout. No chunks are written or read by this test.
        node.new_dataset::<f32>()
            .shape([1_usize << 61])
            .chunk([1])
            .create("shape")
            .unwrap();
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidTensor,
        &["input.shape", "object-size limit"],
    );
    let err = assert_limit(
        nir_rs::io::read_with(&path, &bounded(usize::MAX)),
        usize::MAX,
    );
    assert!(err.to_string().contains("input.shape"));
}

#[test]
fn null_dataspace_edges_are_empty() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "null_edges.nir", |file| {
        let root = file.group("node").unwrap();
        root.unlink("edges").unwrap();
        root.new_dataset::<f32>()
            .shape(hdf5::Extents::Null)
            .create("edges")
            .unwrap();
    });

    for options in [ReadOptions::default(), bounded(1_000_000)] {
        let graph = nir_rs::io::read_with(&path, &options).unwrap();
        assert!(graph.edges.is_empty());
        assert_eq!(graph.nodes.len(), 2);
    }
}
