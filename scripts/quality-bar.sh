#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Shared quality bar used by the OpenCode agent workflow before and after the
# agent runs, to ensure the repo starts green and stays green.

set -euo pipefail

cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --all-features
cargo doc --no-deps --all-features
