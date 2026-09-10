# Cloud-agent / local-dev image for nir-rs.
# Do not COPY the project — Cursor checks out the correct commit itself.
# Keeps cargo/rustc/clippy/rustfmt + libhdf5 so PR/cloud agents can run the
# full quality bar without a separate bootstrap step.
FROM rust:1.98-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        sudo \
        curl \
        pkg-config \
        libhdf5-dev \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Match rust-toolchain.toml components (fmt + clippy) on the image toolchain.
RUN rustup component add clippy rustfmt \
    && cargo --version \
    && rustc --version \
    && cargo fmt --version \
    && cargo clippy --version

# Passwordless sudo is required by some Cursor cloud-agent tooling.
RUN useradd -m -s /bin/bash ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && mkdir -p /workspace \
    && chown ubuntu:ubuntu /workspace

USER ubuntu
WORKDIR /workspace
ENV CARGO_HOME=/home/ubuntu/.cargo \
    RUSTUP_HOME=/home/ubuntu/.rustup \
    PATH=/home/ubuntu/.cargo/bin:/usr/local/cargo/bin:$PATH
