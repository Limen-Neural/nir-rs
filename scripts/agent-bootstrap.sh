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
#   bash scripts/agent-bootstrap.sh --no-hdf5   # skip libhdf5-dev only
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Bootstrap a bare agent/sandbox so cargo fmt, clippy, test, and doc work.

Usage (from repo root):
  bash scripts/agent-bootstrap.sh
  bash scripts/agent-bootstrap.sh --no-hdf5   # skip libhdf5-dev only

Always installs build tools (and curl/cmake when needed). --no-hdf5 only skips
the optional system libhdf5-dev package; hermetic hdf5/static still needs cmake.
EOF
}

WITH_HDF5=1
for arg in "$@"; do
  case "$arg" in
    --no-hdf5) WITH_HDF5=0 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $arg" >&2
      usage >&2
      exit 2
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

apt_install() {
  local packages=("$@")
  if need_cmd sudo && sudo -n true 2>/dev/null; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"
  elif [[ "$(id -u)" -eq 0 ]]; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"
  else
    return 1
  fi
}

ensure_downloader() {
  if need_cmd curl || need_cmd wget; then
    return 0
  fi
  if need_cmd apt-get; then
    log "Installing curl (required to fetch rustup)…"
    if ! apt_install ca-certificates curl; then
      log "ERROR: need curl or wget, and cannot install packages without root/passwordless sudo"
      exit 1
    fi
  else
    log "ERROR: need curl or wget to install rustup"
    exit 1
  fi
}

download_rustup() {
  # Prefer curl; fall back to wget when curl is still missing.
  if need_cmd curl; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs
  elif need_cmd wget; then
    wget -qO- https://sh.rustup.rs
  else
    log "ERROR: no curl/wget after ensure_downloader"
    exit 1
  fi
}

install_rustup() {
  if need_cmd rustup && need_cmd cargo && need_cmd rustc; then
    log "rustup/cargo already present: $(rustc --version 2>/dev/null || true)"
    return 0
  fi

  ensure_downloader
  log "Installing rustup (stable + rustfmt + clippy)…"
  # Non-interactive install; toolchain file will select channel/components.
  download_rustup | sh -s -- -y --default-toolchain none --profile minimal

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
  # Build tools are always required (graph-only crates still need a linker;
  # hermetic hdf5/static needs cmake). libhdf5-dev is optional.
  if need_cmd apt-get; then
    local packages=(build-essential pkg-config ca-certificates curl cmake)
    if [[ "$WITH_HDF5" == "1" ]]; then
      if pkg-config --exists hdf5 2>/dev/null; then
        log "libhdf5 already available via pkg-config"
      else
        packages+=(libhdf5-dev)
      fi
    else
      log "Skipping libhdf5-dev (--no-hdf5); hermetic path uses hdf5/static + cmake"
    fi
    log "Installing system packages: ${packages[*]}"
    if ! apt_install "${packages[@]}"; then
      log "WARN: no passwordless sudo; install build tools manually"
      log "      Hermetic fallback: cargo test --features hdf5,hdf5/static,hdf5/zlib"
    fi
  elif need_cmd brew; then
    log "Installing cmake (Homebrew)…"
    brew list cmake >/dev/null 2>&1 || brew install cmake
    if [[ "$WITH_HDF5" == "1" ]]; then
      log "Installing hdf5 (Homebrew)…"
      brew list hdf5 >/dev/null 2>&1 || brew install hdf5
    fi
  else
    log "WARN: unknown package manager; ensure a C toolchain (+ cmake for static HDF5) is present"
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
  # Hermetic (no libhdf5-dev; needs cmake):
  cargo clippy --all-targets --features hdf5,hdf5/static,hdf5/zlib -- -D warnings
  cargo test --features hdf5,hdf5/static,hdf5/zlib
  cargo doc --no-deps --features hdf5,hdf5/static,hdf5/zlib
EOF
}

main "$@"
