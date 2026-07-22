// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error types for `nir-rs`.
//!
//! Full I/O and schema errors land in later milestones; this module provides a
//! stable public error surface for the scaffold.

use std::fmt;

/// Result type used across the crate.
pub type Result<T> = std::result::Result<T, NirError>;

/// Public error type for NIR operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NirError {
    /// Placeholder for features not yet implemented (v0.2+).
    Unimplemented(&'static str),
}

impl fmt::Display for NirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unimplemented(what) => write!(f, "not implemented: {what}"),
        }
    }
}

impl std::error::Error for NirError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unimplemented_display() {
        let err = NirError::Unimplemented("hdf5 read");
        assert_eq!(err.to_string(), "not implemented: hdf5 read");
    }
}
