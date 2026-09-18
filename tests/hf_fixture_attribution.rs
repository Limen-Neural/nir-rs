// SPDX-License-Identifier: MIT OR Apache-2.0
//! Packaging checks for the NeuroCUDA-derived fixture attribution.

use std::fs;

const NOTICE: &str = "tests/fixtures/huggingface/LICENSE-NeuroCUDA";

#[test]
fn neurocuda_fixture_notice_preserves_the_upstream_mit_terms() {
    let notice = fs::read_to_string(NOTICE).expect("NeuroCUDA fixture notice must be present");

    assert!(notice.contains("MIT License"));
    assert!(notice.contains("Copyright (c) 2026 NeuroCUDA"));
    assert!(
        notice.contains(
            "Permission is hereby granted, free of charge, to any person obtaining a copy"
        )
    );
    assert!(notice.contains("The above copyright notice and this permission notice"));
    assert!(
        notice.contains(
            "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR"
        )
    );
    assert!(
        notice.contains(
            "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE"
        )
    );
    assert!(notice.contains("AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM"));
}
