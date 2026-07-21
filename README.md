# nir-rs

**Pure-Rust implementation of the Neuromorphic Intermediate Representation (NIR)**

[![CI](https://github.com/Limen-Neural/nir-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/nir-rs/actions)
[![crates.io](https://img.shields.io/crates/v/nir-rs.svg)](https://crates.io/crates/nir-rs)
[![docs.rs](https://docs.rs/nir-rs/badge.svg)](https://docs.rs/nir-rs)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](LICENSE)

> The first production-quality Rust library for reading, writing, and working with NIR — the standard interchange format for spiking neural networks.

NIR is to SNNs what ONNX is to conventional neural networks (or GGUF to LLMs): a framework-agnostic graph format that lets models move between simulators and hardware without being rewritten.

## Why nir-rs?

- Official NIR is Python-only
- No mature Rust implementation existed
- Enables pure-Rust, embedded, and high-performance pipelines
- Native integration with the rest of the Limen Neural stack (`axon-encoder`, `silicon-bridge`, `neuromod`, etc.)

## Scope

This crate **owns**:
- The NIR graph model and the ~17 standard node types
- Reading and writing `.nir` (HDF5) files
- Round-trip fidelity and basic validation
- A clean, idiomatic Rust API

This crate does **not** own:
- Training or simulation of SNNs
- Mapping to specific hardware (that lives in `silicon-bridge`)
- Framework-specific converters (those live in the producing/consuming crates)

## Quick Start

```toml
[dependencies]
nir-rs = "0.1"
