// SPDX-License-Identifier: MIT OR Apache-2.0
//! Packaging checks for the Synfire fixture provenance and license notice.

use std::fs;

const README: &str = "tests/fixtures/synfire/README.md";
const LICENSE: &str = "tests/fixtures/synfire/LICENSE-BSD-3-Clause";

#[test]
fn synfire_license_notice_preserves_bsd3_terms() {
    let notice =
        fs::read_to_string(LICENSE).expect("Synfire fixture license notice must be present");
    assert!(notice.contains("BSD 3-Clause License"));
    assert!(notice.contains("Copyright (c) 2023, NIR Team."));
    assert!(notice.contains("Redistribution and use in source and binary forms"));
    assert!(notice.contains("THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS"));
}

#[test]
fn synfire_readme_records_refresh_procedure_and_provenance() {
    let readme = fs::read_to_string(README).expect("Synfire fixture README must be present");
    for needle in [
        "synfire pull pabogdan/lifneuron:1.0.0",
        "synfire pull pabogdan/ifsynfire:0.1.0",
        "synfire pull pabogdan/nmnistcnn:1.0.0",
        "synfire pull pabogdan/brailernn:1.0.1",
        "synfire pull jegp/swavelet:1.0.0",
        "BSD-3-Clause",
        "committed fixtures only",
    ] {
        assert!(
            readme.contains(needle),
            "synfire README must document {needle:?}"
        );
    }
}
