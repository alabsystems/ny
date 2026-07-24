// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `batched_domain` module, split by behavior family.
//!
//! - `builder`: BatchedDomainsBuilder construction and shape checks
//! - `bounds_access`: Input bounds, static bounds, domain operations, utilities
//! - `sparse_bounds`: Sparse intermediate bounds extraction and merge
//! - `serialization`: Constraint serialization roundtrip tests
//! - `branching`: Unstable neuron detection and branch selection

mod bounds_access;
mod branching;
mod builder;
mod error_paths;
mod serialization;
mod sparse_bounds;

use ndarray::ArrayD;

/// Result of evaluating a domain after bound propagation (test-only).
#[derive(Debug, Clone)]
pub(super) enum DomainResult {
    Verified,
    Falsified(Box<Counterexample>),
    Continue,
}

/// A counterexample showing property violation (test-only).
#[derive(Debug, Clone)]
pub(super) struct Counterexample {
    pub input: ArrayD<f32>,
    pub output: ArrayD<f32>,
}
