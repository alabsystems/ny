// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#![forbid(unsafe_code)]

//! File format loaders for ny.
//!
//! This crate provides parsers for reading model weights from various file
//! formats. It sits below `ny-onnx` in the dependency graph so that
//! parsing logic can be shared without pulling in the full ONNX conversion
//! and analysis stack.
//!
//! ## Supported formats
//!
//! - **SafeTensors** — Hugging Face weight files (`.safetensors`)
//!
//! Additional formats (PyTorch, CoreML, GGUF, NNet, ONNX protobuf) will
//! migrate here from `ny-onnx` in subsequent iterations.

// Link macOS Accelerate BLAS for ndarray::dot() acceleration (#4259).
#[cfg(target_os = "macos")]
extern crate blas_src;

/// Shared file I/O helpers (plain and gzip).
pub mod io;

/// SafeTensors format loader for reading Hugging Face model weights.
pub mod safetensors;
