# syntax=docker/dockerfile:1
# nir-rs — published image for GHCR + Docker Hub.
#
# Builder: cargo test --all-features + release example with system libhdf5.
# Runtime: Rust pin + libhdf5 + source (no target/) + example binary.
# WORKDIR stays /src so env!(CARGO_MANIFEST_DIR) from the builder still finds
# tests/fixtures. Non-root user. No Python (AGENTS.md).

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
    && cargo build --release --example load_inspect_lif --features hdf5 \
    && cp target/release/examples/load_inspect_lif /tmp/load_inspect_lif \
    && rm -rf target

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
    && rustup component add clippy rustfmt \
    && useradd --create-home --uid 10001 --shell /bin/bash nir

# Match builder path for CARGO_MANIFEST_DIR baked into the example binary.
WORKDIR /src
COPY --from=builder --chown=nir:nir /src /src
COPY --from=builder /tmp/load_inspect_lif /usr/local/bin/load_inspect_lif
RUN chmod 755 /usr/local/bin/load_inspect_lif

USER nir
ENV CARGO_HOME=/home/nir/.cargo \
    CARGO_TERM_COLOR=always
# Pre-warm crate index for agent use; tolerate offline builders.
RUN cargo fetch || true

LABEL org.opencontainers.image.title="nir-rs" \
      org.opencontainers.image.description="Pure-Rust NIR graph + HDF5 I/O toolchain image" \
      org.opencontainers.image.source="https://github.com/Limen-Neural/nir-rs" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0"

CMD ["rustc", "--version"]
