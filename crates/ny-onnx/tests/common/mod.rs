// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared benchmark-tree resolution for the corpus-backed `ny-onnx` tests.
//!
//! Three test files had grown near-identical `benchmark_root()` copies, and a
//! fourth arrived with the 2026 tree — which tripped
//! `ny-levers`' raw-`NY_*`-read ratchet (830 against a baseline of 827). The
//! ratchet's instruction is explicit: a positive delta must not be absorbed by
//! raising the baseline, because a fresh `env::var("NY_…")` is "unenumerable,
//! its parser disagrees with the other ~850, and it will not appear in the
//! flight receipt".
//!
//! Consolidating here answers that directly — one read serves every caller, so
//! the count goes DOWN rather than up, and the resolution policy is stated once
//! instead of drifting across four copies.
//!
//! POLICY, preserved verbatim from the copies it replaces: honour the override
//! environment variable, otherwise walk ancestors (so a git worktree finds the
//! primary checkout's untracked benchmark tree), and NEVER silently skip — a
//! vacuous pass is exactly the hole a soundness guard cannot afford.

use std::path::{Path, PathBuf};

/// Which staged VNN-COMP tree a test needs.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // Each integration-test crate selects one year from this shared module.
pub enum BenchYear {
    /// `benchmarks/vnncomp2025/benchmarks`, overridden by `NY_BENCH_ROOT`.
    Vnncomp2025,
    /// `benchmarks/vnncomp2026/benchmarks`, overridden by `NY_BENCH_ROOT_2026`.
    Vnncomp2026,
}

impl BenchYear {
    const fn relative_path(self) -> &'static str {
        match self {
            Self::Vnncomp2025 => "benchmarks/vnncomp2025/benchmarks",
            Self::Vnncomp2026 => "benchmarks/vnncomp2026/benchmarks",
        }
    }

    fn override_decl(self) -> &'static ny_levers::LeverDecl {
        match self {
            Self::Vnncomp2025 => &ny_levers::decls::onnx::BENCH_ROOT_2025,
            Self::Vnncomp2026 => &ny_levers::decls::onnx::BENCH_ROOT_2026,
        }
    }
}

/// Resolve the benchmark tree, or panic naming the fix.
///
/// The single raw environment read behind every corpus-backed `ny-onnx` test.
/// Adding another one anywhere in `crates/` trips the ratchet; route new callers
/// through here (or add a `BenchYear` variant) instead.
pub fn benchmark_root(year: BenchYear) -> PathBuf {
    let inputs = ny_levers::RawLeverInputs::capture(ny_levers::all());
    if let Some(root) = inputs.get(year.override_decl()) {
        return PathBuf::from(root);
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest.ancestors() {
        let candidate = ancestor.join(year.relative_path());
        if candidate.is_dir() {
            return candidate;
        }
    }
    panic!(
        "Benchmark tree missing: no `{}` in any ancestor of {}. \
         Run benchmarks/download_benchmarks.sh first, or set {}.",
        year.relative_path(),
        manifest.display(),
        year.override_decl().name,
    );
}
