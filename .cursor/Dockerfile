# Cloud-agent / local-dev image for nir-rs.
# Do not COPY the project — Cursor checks out the correct commit itself.
# Keeps cargo/rustc/clippy/rustfmt + libhdf5 so PR/cloud agents can run the
# full quality bar without a separate bootstrap step.
FROM rust:1.98.1-bookworm

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

# Non-login shells and profile.d login shells must see the image toolchain
# under /usr/local/{cargo,rustup} (do not redirect CARGO_HOME into ~ubuntu).
RUN for f in /usr/local/cargo/bin/*; do ln -sf "$f" /usr/local/bin/; done \
    && printf '%s\n' \
        'export RUSTUP_HOME=/usr/local/rustup' \
        'export CARGO_HOME=/usr/local/cargo' \
        'export PATH=/usr/local/cargo/bin:$PATH' \
        > /etc/profile.d/cargo.sh \
    && chmod 0644 /etc/profile.d/cargo.sh

# Passwordless sudo is required by some Cursor cloud-agent tooling.
RUN useradd -m -s /bin/bash ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && mkdir -p /workspace \
    && chown ubuntu:ubuntu /workspace \
    && chown -R ubuntu:ubuntu /usr/local/cargo /usr/local/rustup

USER ubuntu
WORKDIR /workspace
