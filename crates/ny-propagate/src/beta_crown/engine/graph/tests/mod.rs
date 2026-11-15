// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for graph-specific β-CROWN verifier logic.
//!
//! Extracted from inline `#[cfg(test)]` blocks for maintainability.

mod clip_complete_selector;
mod conv_mode;
mod domain_conversion;
mod domain_process;
mod gpu_bab;
mod kfsb_production;
mod kfsb_single_shot;
mod kfsb_single_shot_review;
mod multi_objective_alpha;
mod objectives;
mod warm_cache_rounding_error;
mod warm_frontier;
mod warm_start_seed;
