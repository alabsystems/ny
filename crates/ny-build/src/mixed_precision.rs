// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Mixed-precision policy for verifying models at their deployed precision (P8).
//!
//! NY proves bounds under an f32 idealization. A model that runs in f16/bf16 (or
//! accumulates in a different precision than it computes in) needs those bounds
//! widened to remain SOUND for the bits actually executed. [`MixedPrecisionPolicy`]
//! records the target *compute* and *accumulate* precisions so downstream
//! widening can key off them.
//!
//! ADDITIVE + OPT-IN: the [`Default`] is `{ compute: F32, accumulate: F32 }`,
//! which is exactly today's behavior (no widening, no regression). A policy only
//! influences verification when a caller sets a non-F32 precision.

use ny_core::FloatPrecision;
use serde::{Deserialize, Serialize};

/// Target precisions for a mixed-precision verification run.
///
/// `compute` is the precision in which per-element products / activations are
/// represented; `accumulate` is the precision of the running sum in reductions
/// (e.g. the dot-product accumulator in a GEMM). They are tracked separately
/// because real hardware commonly multiplies in a low precision but accumulates
/// in a wider one (f16 multiply, f32 accumulate).
///
/// SOUNDNESS: this type is purely a declaration of intent. It carries no bounds
/// and can never narrow one. Widening logic that consumes it is responsible for
/// producing a superset of every deployed-precision value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MixedPrecisionPolicy {
    /// Precision of element-wise products and activations.
    pub compute: FloatPrecision,
    /// Precision of reduction accumulators (dot-product running sums).
    pub accumulate: FloatPrecision,
}

impl MixedPrecisionPolicy {
    /// Construct a policy with explicit compute and accumulate precisions.
    #[must_use]
    pub const fn new(compute: FloatPrecision, accumulate: FloatPrecision) -> Self {
        Self {
            compute,
            accumulate,
        }
    }

    /// The idealized all-f32 policy: identical to today's default behavior.
    ///
    /// Equivalent to [`MixedPrecisionPolicy::default`]; provided as a named
    /// constructor for call sites that want to be explicit that no widening is
    /// requested.
    #[must_use]
    pub const fn f32_idealized() -> Self {
        Self::new(FloatPrecision::F32, FloatPrecision::F32)
    }

    /// A uniform policy where compute and accumulate share one precision.
    #[must_use]
    pub const fn uniform(precision: FloatPrecision) -> Self {
        Self::new(precision, precision)
    }

    /// Whether this policy is the idealized all-f32 case requiring no widening.
    ///
    /// Downstream verification can fast-path on this to guarantee the legacy
    /// f32 path is byte-for-byte unchanged when no mixed precision is requested.
    #[must_use]
    pub const fn is_idealized_f32(self) -> bool {
        self.compute.is_idealized_f32() && self.accumulate.is_idealized_f32()
    }
}

impl Default for MixedPrecisionPolicy {
    /// Both compute and accumulate default to f32 — today's exact behavior.
    fn default() -> Self {
        Self::f32_idealized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_f32_idealization() {
        let policy = MixedPrecisionPolicy::default();
        assert_eq!(policy.compute, FloatPrecision::F32);
        assert_eq!(policy.accumulate, FloatPrecision::F32);
        assert!(policy.is_idealized_f32());
    }

    #[test]
    fn f32_idealized_constructor_matches_default() {
        assert_eq!(
            MixedPrecisionPolicy::f32_idealized(),
            MixedPrecisionPolicy::default()
        );
    }

    #[test]
    fn uniform_sets_both_fields() {
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        assert_eq!(policy.compute, FloatPrecision::F16);
        assert_eq!(policy.accumulate, FloatPrecision::F16);
        assert!(!policy.is_idealized_f32());
    }

    #[test]
    fn mixed_compute_low_accumulate_high_is_not_idealized() {
        // f16 multiply, f32 accumulate — a common real hardware configuration.
        let policy = MixedPrecisionPolicy::new(FloatPrecision::F16, FloatPrecision::F32);
        assert!(!policy.is_idealized_f32());
        assert_eq!(policy.compute, FloatPrecision::F16);
        assert_eq!(policy.accumulate, FloatPrecision::F32);
    }

    #[test]
    fn policy_round_trips_through_serde() {
        let policy = MixedPrecisionPolicy::new(FloatPrecision::Bf16, FloatPrecision::F32);
        let json = serde_json::to_string(&policy).expect("serialize");
        let back: MixedPrecisionPolicy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(policy, back);
    }
}
