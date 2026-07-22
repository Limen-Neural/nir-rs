// SPDX-License-Identifier: MIT OR Apache-2.0

//! NIR computational node types.
//!
//! v0.1 only reserves the module. Implement a closed enum matching **upstream
//! HDF5 `type` strings** in **v0.2** (LIM-829 / GH#8):
//!
//! `Input`, `Output`, `Affine`, `Linear`, `Scale`, `Conv1d`, `Conv2d`,
//! `CubaLI`, `CubaLIF`, `Delay`, `Flatten`, `I`, `IF`, `LI`, `LIF`,
//! `SumPool2d`, `AvgPool2d`, `Threshold` (plus nestable `NIRGraph`).
//!
//! Do **not** invent marketing names (`CurrLIF`, `Convolution`, `Integrator`)
//! for wire types — those break interoperability with Python NIR.

/// Placeholder node handle until the full enum lands in v0.2.
///
/// This type exists so the module is public and re-exportable without claiming
/// a complete IR surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NodePlaceholder;

impl NodePlaceholder {
    /// Construct the scaffold placeholder.
    pub const fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_constructs() {
        assert_eq!(NodePlaceholder::new(), NodePlaceholder);
    }
}
