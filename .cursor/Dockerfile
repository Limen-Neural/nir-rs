# Cloud-agent / local-dev image for nir-rs.
# Do not COPY the project — Cursor checks out the correct commit itself.
FROM rust:1.97-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        sudo \
        pkg-config \
        libhdf5-dev \
        build-essential \
        ca-certificates \
        clang \
        llvm-dev \
    && rm -rf /var/lib/apt/lists/*

RUN rustup component add clippy rustfmt

# Passwordless sudo is required by some Cursor cloud-agent tooling.
RUN useradd -m -s /bin/bash ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

USER ubuntu
WORKDIR /workspace
