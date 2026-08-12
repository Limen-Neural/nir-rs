# syntax=docker/dockerfile:1
# nir-rs — published image for GHCR + Docker Hub.
#
# Builder stage: compile/test with system libhdf5 (matches CI quality bar).
# Runtime stage: toolchain + libhdf5 for agents/consumers (not an SNN simulator).
# No Python in either stage (AGENTS.md).

ARG RUST_IMAGE=rust:1.97-bookworm

FROM ${RUST_IMAGE} AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        pkg-config \
        libhdf5-dev \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN rustup component add clippy rustfmt \
    && cargo test --all-features \
    && cargo build --release --example load_inspect_lif --features hdf5

# ---------------------------------------------------------------------------
# Published image: Rust pin + libhdf5 + crate source + release example binary
# ---------------------------------------------------------------------------
FROM ${RUST_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        pkg-config \
        libhdf5-dev \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && rustup component add clippy rustfmt

WORKDIR /workspace
COPY --from=builder /src /workspace
COPY --from=builder /src/target/release/examples/load_inspect_lif /usr/local/bin/load_inspect_lif

# Pre-warm registry cache for offline-ish agent use; ignore failure if offline.
RUN cargo fetch || true

ENV CARGO_TERM_COLOR=always
LABEL org.opencontainers.image.title="nir-rs" \
      org.opencontainers.image.description="Pure-Rust NIR graph + HDF5 I/O toolchain image" \
      org.opencontainers.image.source="https://github.com/Limen-Neural/nir-rs" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0"

CMD ["rustc", "--version"]
