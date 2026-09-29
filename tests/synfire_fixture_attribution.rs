// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packaging checks for the Synfire-derived fixture attribution.
//!
//! The vendored `.nir` files declare `"license": "BSD-3-Clause"`, which permits
//! redistribution with attribution and this notice. `LICENSE-Synfire` reproduces
//! both copyright holders and the full BSD 3-Clause body, following the
//! `tests/fixtures/huggingface/LICENSE-NeuroCUDA` precedent. This check needs no
//! libhdf5, so default-feature CI guards the redistribution basis.

use std::fs;

const NOTICE: &str = "tests/fixtures/synfire/LICENSE-Synfire";

#[test]
fn synfire_fixture_notice_preserves_bsd_3_clause_terms() {
    let notice = fs::read_to_string(NOTICE).expect("Synfire fixture notice must be present");

    // BSD 3-Clause header.
    assert!(notice.contains("BSD 3-Clause License"));

    // Both copyright holders.
    assert!(notice.contains("Copyright (c) 2023, NIR Team."));
    assert!(notice.contains(
        "Copyright (c) the swavelet authors (Jens E. Pedersen, Tony Lindeberg, Peter Gerstoft)."
    ));
    assert!(notice.contains("Jens E. Pedersen"));
    assert!(notice.contains("swavelet"));

    // Clause 1: retain the copyright notice.
    assert!(
        notice.contains(
            "Redistributions of source code must retain the above copyright notice, this"
        )
    );

    // Clause 3: no endorsement.
    assert!(notice.contains("Neither the name of the copyright holder nor the names of its"));
    assert!(
        notice.contains("contributors may be used to endorse or promote products derived from")
    );

    // "AS IS" disclaimer.
    assert!(
        notice.contains(
            "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\""
        )
    );
}
