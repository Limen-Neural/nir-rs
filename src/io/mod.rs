// SPDX-License-Identifier: MIT OR Apache-2.0

//! HDF5 `.nir` read/write.
//!
//! **v0.3 — HDF5 I/O** will implement read/write using the maintained
//! `hdf5-metno` crate (feature-gated), not the abandoned `hdf5` 0.8 package.
//!
//! This module is intentionally empty in v0.1 so pure-Rust graph work can
//! land without a system libhdf5 dependency.

use crate::error::{NirError, Result};
use crate::graph::NirGraph;
use std::path::Path;

/// Read a NIR graph from a `.nir` (HDF5) path.
///
/// Unimplemented until v0.3.
pub fn read(_path: impl AsRef<Path>) -> Result<NirGraph> {
    Err(NirError::Unimplemented("hdf5 read (v0.3)"))
}

/// Write a NIR graph to a `.nir` (HDF5) path.
///
/// Unimplemented until v0.3.
pub fn write(_path: impl AsRef<Path>, _graph: &NirGraph) -> Result<()> {
    Err(NirError::Unimplemented("hdf5 write (v0.3)"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_is_unimplemented() {
        let err = read("model.nir").unwrap_err();
        assert!(matches!(err, NirError::Unimplemented(_)));
    }

    #[test]
    fn write_is_unimplemented() {
        let g = NirGraph::new();
        let err = write("out.nir", &g).unwrap_err();
        assert!(matches!(err, NirError::Unimplemented(_)));
    }
}
