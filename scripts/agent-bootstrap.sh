#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Bootstrap a bare agent/sandbox so cargo fmt, clippy, test, and doc work.
# Safe to re-run (idempotent). Prefer the prebuilt .cursor/Dockerfile when
# available; this script is for PR bots (cubic/Claude sandboxes, Copilot, etc.)
# that start without a Rust toolchain.
#
# Usage (from repo root):
#   bash scripts/agent-bootstrap.sh
#   bash scripts/agent-bootstrap.sh --no-hdf5   # skip libhdf5-dev (graph-only tests)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_HDF5=1
for arg in "$@"; do
  case "$arg" in
    --no-hdf5) WITH_HDF5=0 ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
  esac
done

log() { printf '==> %s\n' "$*"; }

export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export PATH="$CARGO_HOME/bin:$PATH"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_rustup() {
  if need_cmd rustup && need_cmd cargo && need_cmd rustc; then
    log "rustup/cargo already present: $(rustc --version 2>/dev/null || true)"
    return 0
  fi

  log "Installing rustup (stable + rustfmt + clippy)…"
  # Non-interactive install; toolchain file will select channel/components.
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain none --profile minimal

  # shellcheck disable=SC1091
  source "$CARGO_HOME/env"
}

ensure_toolchain() {
  # rust-toolchain.toml drives channel + components when present.
  if [[ -f rust-toolchain.toml ]]; then
    log "Installing toolchain from rust-toolchain.toml…"
    rustup show active-toolchain || true
    # Force component install even if channel already present.
    rustup component add rustfmt clippy 2>/dev/null || true
  else
    log "No rust-toolchain.toml; installing stable with fmt/clippy…"
    rustup toolchain install stable --component rustfmt,clippy
    rustup default stable
  fi

  # Verify the quality bar tools exist.
  cargo --version
  rustc --version
  cargo fmt --version
  cargo clippy --version
}

install_system_deps() {
  if [[ "$WITH_HDF5" != "1" ]]; then
    log "Skipping libhdf5-dev (--no-hdf5)"
    return 0
  fi

  if pkg-config --exists hdf5 2>/dev/null; then
    log "libhdf5 already available via pkg-config"
    return 0
  fi

  if need_cmd apt-get; then
    log "Installing build deps + libhdf5-dev (apt)…"
    if need_cmd sudo && sudo -n true 2>/dev/null; then
      sudo apt-get update -qq
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential pkg-config libhdf5-dev ca-certificates curl
    elif [[ "$(id -u)" -eq 0 ]]; then
      apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential pkg-config libhdf5-dev ca-certificates curl
    else
      log "WARN: no passwordless sudo; install libhdf5-dev manually for --all-features"
      log "      Hermetic fallback: cargo test --features hdf5,hdf5/static,hdf5/zlib"
    fi
  elif need_cmd brew; then
    log "Installing hdf5 (Homebrew)…"
    brew list hdf5 >/dev/null 2>&1 || brew install hdf5
  else
    log "WARN: unknown package manager; using hermetic hdf5/static path for tests"
  fi
}

fetch_deps() {
  log "cargo fetch (default features)…"
  cargo fetch
}

main() {
  log "nir-rs agent bootstrap (root=$ROOT)"
  install_rustup
  # shellcheck disable=SC1091
  [[ -f "$CARGO_HOME/env" ]] && source "$CARGO_HOME/env"
  ensure_toolchain
  install_system_deps
  fetch_deps
  log "Bootstrap complete. Quality bar:"
  cat <<'EOF'
  cargo fmt --check
  cargo test
  # With system libhdf5:
  cargo clippy --all-targets --all-features -- -D warnings
  cargo test --all-features
  cargo doc --no-deps --all-features
  # Hermetic (no libhdf5-dev):
  cargo clippy --all-targets --features hdf5,hdf5/static,hdf5/zlib -- -D warnings
  cargo test --features hdf5,hdf5/static,hdf5/zlib
EOF
}

main "$@"
