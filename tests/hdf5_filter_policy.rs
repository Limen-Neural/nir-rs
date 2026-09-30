// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dataset-filter admission tests for HDF5 `.nir` reads.

#![cfg(feature = "hdf5")]

mod common;
use common::{assert_err, write_then};
use nir_rs::NirError;
use nir_rs::io::ReadOptions;
use tempfile::TempDir;

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
fn other_builtin_filters_are_rejected_before_decode() {
    for (filter_id, name) in [
        (hdf5_sys::h5z::H5Z_FILTER_SZIP, "SZip"),
        (hdf5_sys::h5z::H5Z_FILTER_SCALEOFFSET, "ScaleOffset"),
    ] {
        let dir = TempDir::new().unwrap();
        let path = tampered(&dir, "other_builtin_filter.nir", |file| {
            let dcpl = filter_dcpl(filter_id, &[]);
            replace_shape_with_raw_dcpl(file, &dcpl);
        });
        assert_err(
            nir_rs::io::read(&path),
            NirError::InvalidGraph,
            &["input.shape", name],
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
fn oversized_filter_parameters_cannot_panic_the_reader() {
    for (filename, filter_id, error_fragment) in [
        ("user_many_parameters.nir", 32_000, "not allowed"),
        (
            "deflate_many_parameters.nir",
            hdf5_sys::h5z::H5Z_FILTER_DEFLATE,
            "at most 32",
        ),
    ] {
        let dir = TempDir::new().unwrap();
        let path = tampered(&dir, filename, |file| {
            let dcpl = filter_dcpl(filter_id, &[7; 33]);
            replace_shape_with_raw_dcpl(file, &dcpl);
        });
        assert_err(
            nir_rs::io::read(&path),
            NirError::InvalidGraph,
            &["input.shape", error_fragment],
        );
    }
}

#[test]
fn malformed_deflate_parameters_fail_closed() {
    let dir = TempDir::new().unwrap();
    let path = tampered(&dir, "deflate_no_parameters.nir", |file| {
        let dcpl = filter_dcpl(hdf5_sys::h5z::H5Z_FILTER_DEFLATE, &[]);
        replace_shape_with_raw_dcpl(file, &dcpl);
    });

    assert_err(
        nir_rs::io::read(&path),
        NirError::InvalidGraph,
        &["input.shape", "cannot inspect dataset filters"],
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
