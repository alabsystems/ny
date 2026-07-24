// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! Shared test utilities for ny workspace.
//!
//! Provides common helpers for integration tests across crates:
//! - Path utilities for locating test models
//! - Assertion helpers for model existence
//! - Shared GPU regression comparison helpers and tolerances

#[cfg(feature = "tensor")]
mod bounds;
// Serialized env-var mutation choke point (clippy env wall). Dependency-free
// like `scalar` so every crate can reuse it regardless of features.
pub mod env;
#[cfg(feature = "core")]
mod gemm;
// Keep scalar assertions dependency-free so downstream test crates can reuse
// them without pulling ny-core/ny-tensor back through a dev-dependency
// cycle.
mod scalar;

#[cfg(feature = "tensor")]
pub use bounds::{
    assert_bounded_tensor_close, assert_bounds_do_not_loosen, assert_slice_close,
    assert_slice_close_relative, GPU_REGRESSION_NEAR_EXACT_EPSILON, GPU_REGRESSION_RELAXED_EPSILON,
    GPU_REGRESSION_STRICT_EPSILON, GPU_REGRESSION_TINY_EPSILON,
};
#[cfg(feature = "core")]
pub use gemm::CountingGemmEngine;
pub use scalar::{assert_f32_close, assert_f32_nan};

use std::path::PathBuf;

/// Get the workspace root directory.
///
/// Works from any crate in the workspace by traversing up from CARGO_MANIFEST_DIR.
pub fn workspace_root() -> PathBuf {
    // When called from a crate's tests, CARGO_MANIFEST_DIR points to that crate.
    // We traverse up to find the workspace root (where the root Cargo.toml lives).
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set - are you running under cargo?");

    let mut path = PathBuf::from(manifest_dir);

    // Walk up until we find the workspace root (has members in Cargo.toml)
    // For safety, limit to 10 levels
    for _ in 0..10 {
        let cargo_toml = path.join("Cargo.toml");
        if cargo_toml.exists() {
            // Check if this is the workspace root by looking for [workspace]
            if let Ok(contents) = std::fs::read_to_string(&cargo_toml) {
                if contents.contains("[workspace]") {
                    return path;
                }
            }
        }

        // Move up one level
        if let Some(parent) = path.parent() {
            path = parent.to_path_buf();
        } else {
            break;
        }
    }

    // Fallback: assume we're already at workspace root
    PathBuf::from(".")
}

/// Get path to in-repo test models directory (`tests/models/`).
///
/// This directory contains small test models committed to the repository.
pub fn test_models_dir() -> PathBuf {
    workspace_root().join("tests/models")
}

/// Get path to external models directory (`models/`).
///
/// This directory contains larger models not committed to git.
/// These are typically downloaded separately for benchmarking.
pub fn external_models_dir() -> PathBuf {
    workspace_root().join("models")
}

/// Assert that a test model exists, failing with a clear error if missing.
///
/// Use this for in-repo test models that should always exist.
pub fn require_model(model_path: &std::path::Path) {
    assert!(
        model_path.exists(),
        "Test model missing: {}. This is an in-repo test model that should exist.",
        model_path.display()
    );
}

/// Assert that an external model exists, failing with a descriptive error.
///
/// Use this for models not committed to git that require separate download.
pub fn require_external_model(model_path: &std::path::Path) {
    assert!(
        model_path.exists(),
        "External model missing: {}. Download the model or disable the external-models feature.",
        model_path.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_root_exists() {
        let root = workspace_root();
        assert!(
            root.join("Cargo.toml").exists(),
            "Workspace root should contain Cargo.toml"
        );
        assert!(
            root.join("crates").is_dir(),
            "Workspace root should contain crates/"
        );
    }

    #[test]
    fn test_models_dir_is_under_workspace() {
        let models = test_models_dir();
        let root = workspace_root();
        assert!(
            models.starts_with(&root),
            "test_models_dir should be under workspace root"
        );
    }
}
