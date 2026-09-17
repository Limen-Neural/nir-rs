// SPDX-License-Identifier: MIT OR Apache-2.0
//! Packaging checks for the NeuroCUDA-derived fixture attribution.

use std::fs;

const NOTICE: &str = "tests/fixtures/huggingface/LICENSE-NeuroCUDA";

#[test]
fn neurocuda_fixture_notice_preserves_the_upstream_mit_terms() {
    let notice = fs::read_to_string(NOTICE).expect("NeuroCUDA fixture notice must be present");

    assert!(notice.contains("MIT License"));
    assert!(notice.contains("Copyright (c) 2026 NeuroCUDA"));
    assert!(notice.contains("The above copyright notice and this permission notice"));
}
