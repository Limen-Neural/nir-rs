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
const README: &str = "tests/fixtures/synfire/README.md";

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

/// Guards that the individual model creators are credited, drawn from each
/// model's `nir-card.json`. The notice names the copyright holders; the README
/// `Credits and citation` section names the per-model authors and records the
/// NIR-paper citation where the card declares one. This check needs no libhdf5.
#[test]
fn synfire_fixture_credits_every_model_creator() {
    let notice = fs::read_to_string(NOTICE).expect("Synfire fixture notice must be present");
    let readme = fs::read_to_string(README).expect("Synfire fixture README must be present");

    // The swavelet authors are credited in both the notice and the README.
    for author in ["Jens E. Pedersen", "Tony Lindeberg", "Peter Gerstoft"] {
        assert!(
            notice.contains(author),
            "LICENSE-Synfire must credit swavelet author {author}"
        );
        assert!(
            readme.contains(author),
            "README must credit swavelet author {author}"
        );
    }

    // The ifsynfire publisher (Petrut Bogdan) is credited in both files.
    assert!(
        notice.contains("Petrut Bogdan"),
        "LICENSE-Synfire must credit Petrut Bogdan (ifsynfire)"
    );
    assert!(
        readme.contains("Petrut Bogdan"),
        "README must credit Petrut Bogdan (ifsynfire)"
    );

    // The NIR Team authored lifneuron, nmnistcnn, and brailernn.
    assert!(
        notice.contains("NIR Team"),
        "LICENSE-Synfire must credit the NIR Team"
    );
    assert!(
        readme.contains("NIR Team"),
        "README must credit the NIR Team"
    );

    // The NIR-paper DOI is recorded where the card citation was added.
    assert!(
        readme.contains("10.1038/s41467-024-52259-9"),
        "README must record the NIR-paper DOI for the NIR-cited models"
    );
}
