// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Result types for β-CROWN verification.

use std::cell::Cell;
use std::time::{Duration, Instant};

use ny_core::{Bound, MethodUsed, SoundnessProvenance, UnknownReason, VerificationResult};
use ny_tensor::{repair_inverted_bounds, BoundedTensor, InversionRepair};

#[derive(Clone, Copy, Default)]
struct GlobalTimeoutDeadlineCapture {
    armed: bool,
    invocations: u32,
    source: Option<Instant>,
    poisoned: bool,
}

thread_local! {
    /// Exact global deadline observed by the current thread's β-CROWN engine.
    ///
    /// This is receipt provenance, not verdict authority. Only engine seams
    /// that themselves classify an expiry against their global deadline may
    /// record it; late publication gates and phase-local caps must not. More
    /// than one engine invocation in a top-level capture poisons the source so
    /// a timeout from one relational row cannot be attributed to another.
    static GLOBAL_TIMEOUT_DEADLINE_SOURCE: Cell<GlobalTimeoutDeadlineCapture> =
        const { Cell::new(GlobalTimeoutDeadlineCapture {
            armed: false,
            invocations: 0,
            source: None,
            poisoned: false,
        }) };
}

/// Reset typed timeout provenance before one top-level β-CROWN invocation.
pub fn reset_global_timeout_deadline_source() {
    GLOBAL_TIMEOUT_DEADLINE_SOURCE.with(|capture| {
        capture.set(GlobalTimeoutDeadlineCapture {
            armed: true,
            ..GlobalTimeoutDeadlineCapture::default()
        });
    });
}

/// Register one engine invocation inside the active top-level capture.
pub(crate) fn note_global_timeout_engine_invocation() {
    GLOBAL_TIMEOUT_DEADLINE_SOURCE.with(|capture| {
        let mut state = capture.get();
        if !state.armed {
            return;
        }
        state.invocations = state.invocations.saturating_add(1);
        if state.invocations != 1 {
            state.source = None;
            state.poisoned = true;
        }
        capture.set(state);
    });
}

/// Record a timeout only where the engine itself observed its global deadline.
pub(crate) fn record_global_timeout_deadline_source(deadline: Option<Instant>) {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        GLOBAL_TIMEOUT_DEADLINE_SOURCE.with(|capture| {
            let mut state = capture.get();
            if state.armed && state.invocations == 1 && !state.poisoned {
                state.source = deadline;
                capture.set(state);
            }
        });
    }
}

/// Take typed timeout provenance after the corresponding engine invocation.
pub fn take_global_timeout_deadline_source() -> Option<Instant> {
    GLOBAL_TIMEOUT_DEADLINE_SOURCE.with(|capture| {
        let state = capture.take();
        (state.armed && state.invocations == 1 && !state.poisoned)
            .then_some(state.source)
            .flatten()
    })
}

#[derive(Debug, Clone)]
pub struct BetaCrownResult {
    /// Verification result.
    pub result: BabVerificationStatus,
    /// Number of domains explored.
    pub domains_explored: usize,
    /// Total time taken.
    pub time_elapsed: Duration,
    /// Maximum depth reached.
    pub max_depth_reached: usize,
    /// Final output bounds (if available).
    pub output_bounds: Option<BoundedTensor>,
    /// Number of cutting planes generated (GCP-CROWN).
    pub cuts_generated: usize,
    /// Number of domains verified (contributes to cut generation).
    pub domains_verified: usize,
}

/// A concrete point a BaB stage already evaluated with its own exact concrete
/// forward and scored as violating (#advcheck-witness).
///
/// SOUNDNESS CONTRACT — this is a CANDIDATE, never a verdict. It rides along
/// a [`BabVerificationStatus::PotentialViolation`] purely so the post-BaB
/// confirmer can VERIFY THIS POINT instead of re-searching the root box for a
/// point the search already held. It confers no trust: the confirmer still
/// re-evaluates the model at the point and still checks the FULL VNN-LIB
/// output constraints, and the emitted `sat` still passes the unchanged
/// trusted ONNX-Runtime gate. A witness that fails either check is discarded
/// and the caller behaves exactly as it did without one.
#[derive(Debug, Clone, PartialEq)]
pub struct ViolationWitness {
    /// Flattened input coordinates of the candidate point.
    pub input: Vec<f32>,
    /// Shape of `input` as the network consumed it.
    pub input_shape: Vec<usize>,
    /// Network output at `input` per the producing stage's concrete forward.
    /// Diagnostic only — the confirmer re-evaluates rather than trusting it.
    pub output: Vec<f32>,
}

/// Status of β-CROWN verification.
#[derive(Debug, Clone, PartialEq)]
pub enum BabVerificationStatus {
    /// Property verified: all domains have lower bound > threshold.
    Verified,
    /// Property violated: concrete counterexample found via PGD attack.
    Violated {
        /// Counterexample input that violates the property.
        counterexample: Vec<f32>,
        /// Output at the counterexample.
        output: Vec<f32>,
    },
    /// Property potentially violated: found a domain where upper bound < threshold.
    ///
    /// `witness` is `Some` only when the producing stage held a concrete point
    /// its own exact forward scored as violating (currently the input-split
    /// `adv_check` PGD probe). `None` is the historical payloadless case: a
    /// bounds-only violation with no point in hand. The verdict semantics of
    /// the two are identical — the payload only saves the confirmer from
    /// re-searching for something already found.
    PotentialViolation {
        /// Concrete candidate point, when the producing stage held one.
        witness: Option<Box<ViolationWitness>>,
    },
    /// Inconclusive: timed out or hit domain limit.
    Unknown { reason: String },
    /// Verification timed out before completion.
    Timeout,
}

impl BabVerificationStatus {
    /// Payloadless `PotentialViolation` — a bounds-only violation with no
    /// concrete point in hand. This is the historical behaviour and remains
    /// the right constructor for every producer that only has bounds.
    pub const fn potential_violation() -> Self {
        Self::PotentialViolation { witness: None }
    }

    /// `PotentialViolation` carrying a concrete candidate point.
    ///
    /// Callers must only pass a point their own exact concrete forward scored
    /// as violating. It is still re-verified downstream (see
    /// [`ViolationWitness`]).
    pub fn potential_violation_with(witness: ViolationWitness) -> Self {
        Self::PotentialViolation {
            witness: Some(Box::new(witness)),
        }
    }

    /// The carried candidate point, if any.
    pub fn potential_violation_witness(&self) -> Option<&ViolationWitness> {
        match self {
            Self::PotentialViolation { witness } => witness.as_deref(),
            _ => None,
        }
    }
}

/// Convert a `BoundedTensor` to a flat `Vec<Bound>`.
///
/// Uses `Bound::new_allow_infinite` since β-CROWN may produce infinite bounds
/// in timeout or unknown cases (e.g., bounds that exploded during propagation).
///
/// NaN sanitization (#2663): If any element is NaN (from upstream numerical
/// instability that escaped the optimization loop guards), it is replaced with
/// the conservative bound (-inf, +inf). `Bound::new_allow_infinite` asserts
/// non-NaN, so passing NaN through would panic.
///
/// Inverted bound repair (#3216, #3307): finite inversions are repaired via the
/// shared `ny_tensor::repair_inverted_bounds(..., InversionRepair::Swap)`
/// helper so Beta-CROWN export uses the same swap strategy as SMT export and
/// bounded-tensor constructors.
fn bounded_tensor_to_bounds(tensor: &BoundedTensor) -> Vec<Bound> {
    let flat = tensor.flatten();
    let (lower, upper) = flat.lower_upper();
    let mut inversions = 0usize;
    let bounds: Vec<Bound> = lower
        .iter()
        .zip(upper.iter())
        .map(|(&l, &u)| {
            let mut repaired_lower = [if l.is_nan() { f32::NEG_INFINITY } else { l }];
            let mut repaired_upper = [if u.is_nan() { f32::INFINITY } else { u }];
            inversions += repair_inverted_bounds(
                &mut repaired_lower,
                &mut repaired_upper,
                InversionRepair::Swap,
            );
            Bound::new_allow_infinite(repaired_lower[0], repaired_upper[0])
        })
        .collect();
    if inversions > 0 {
        tracing::warn!(
            inversions,
            total = bounds.len(),
            "Inverted bounds from CROWN — swapped to valid intervals (#3216)"
        );
    }
    bounds
}

impl From<BetaCrownResult> for VerificationResult {
    fn from(result: BetaCrownResult) -> Self {
        // Use "BetaCrown" for consistency with verifier.rs and ny-python
        let method: Option<MethodUsed> = Some(MethodUsed::BetaCrown);
        let provenance = SoundnessProvenance::sound();

        match result.result {
            BabVerificationStatus::Verified => {
                let output_bounds = result
                    .output_bounds
                    .as_ref()
                    .map(bounded_tensor_to_bounds)
                    .unwrap_or_default();
                VerificationResult::Verified {
                    provenance,
                    output_bounds,
                    proof: None,
                    actual_method: method,
                }
            }
            BabVerificationStatus::Violated {
                counterexample,
                output,
            } => VerificationResult::Violated {
                provenance,
                counterexample,
                output,
                details: None,
                actual_method: method,
            },
            BabVerificationStatus::PotentialViolation { .. } => {
                let bounds = result
                    .output_bounds
                    .as_ref()
                    .map(bounded_tensor_to_bounds)
                    .unwrap_or_default();
                VerificationResult::Unknown {
                    provenance,
                    bounds,
                    reason: UnknownReason::PotentialViolation,
                    actual_method: method,
                }
            }
            BabVerificationStatus::Unknown { reason } => {
                let bounds = result
                    .output_bounds
                    .as_ref()
                    .map(bounded_tensor_to_bounds)
                    .unwrap_or_default();
                VerificationResult::Unknown {
                    provenance,
                    bounds,
                    reason: UnknownReason::from(reason),
                    actual_method: method,
                }
            }
            BabVerificationStatus::Timeout => {
                let partial_bounds = result.output_bounds.as_ref().map(bounded_tensor_to_bounds);
                VerificationResult::Timeout {
                    provenance,
                    partial_bounds,
                    actual_method: method,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    fn make_beta_result(status: BabVerificationStatus) -> BetaCrownResult {
        BetaCrownResult {
            result: status,
            domains_explored: 100,
            time_elapsed: Duration::from_secs(1),
            max_depth_reached: 5,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 50,
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_from_verified() {
        let beta = make_beta_result(BabVerificationStatus::Verified);
        let result: VerificationResult = beta.into();
        assert!(matches!(result, VerificationResult::Verified { .. }));
        if let VerificationResult::Verified { actual_method, .. } = result {
            assert_eq!(actual_method, Some(MethodUsed::BetaCrown));
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_from_violated() {
        let beta = make_beta_result(BabVerificationStatus::Violated {
            counterexample: vec![1.0, 2.0],
            output: vec![0.5],
        });
        let result: VerificationResult = beta.into();
        if let VerificationResult::Violated {
            counterexample,
            output,
            actual_method,
            ..
        } = result
        {
            assert_eq!(counterexample, vec![1.0, 2.0]);
            assert_eq!(output, vec![0.5]);
            assert_eq!(actual_method, Some(MethodUsed::BetaCrown));
        } else {
            unreachable!("Expected Violated");
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_from_potential_violation() {
        let beta = make_beta_result(BabVerificationStatus::potential_violation());
        let result: VerificationResult = beta.into();
        if let VerificationResult::Unknown { reason, .. } = result {
            assert_eq!(reason, UnknownReason::PotentialViolation);
        } else {
            unreachable!("Expected Unknown with PotentialViolation");
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_from_unknown() {
        let beta = make_beta_result(BabVerificationStatus::Unknown {
            reason: "domain limit".to_string(),
        });
        let result: VerificationResult = beta.into();
        assert!(matches!(result, VerificationResult::Unknown { .. }));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_from_timeout() {
        let beta = make_beta_result(BabVerificationStatus::Timeout);
        let result: VerificationResult = beta.into();
        assert!(matches!(result, VerificationResult::Timeout { .. }));
    }

    #[test]
    fn global_timeout_deadline_source_is_exact_scoped_and_fail_closed() {
        reset_global_timeout_deadline_source();
        note_global_timeout_engine_invocation();
        let future = Instant::now() + Duration::from_secs(1);
        record_global_timeout_deadline_source(Some(future));
        assert_eq!(take_global_timeout_deadline_source(), None);

        reset_global_timeout_deadline_source();
        note_global_timeout_engine_invocation();
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond before now is representable");
        record_global_timeout_deadline_source(Some(expired));
        assert_eq!(take_global_timeout_deadline_source(), Some(expired));
        assert_eq!(take_global_timeout_deadline_source(), None);

        reset_global_timeout_deadline_source();
        note_global_timeout_engine_invocation();
        note_global_timeout_engine_invocation();
        record_global_timeout_deadline_source(Some(expired));
        assert_eq!(
            take_global_timeout_deadline_source(),
            None,
            "multiple engine calls must poison cross-invocation attribution"
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_bounded_tensor_to_bounds_with_infinite() {
        // Test that infinite bounds don't panic
        let lower = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::NEG_INFINITY);
        let upper = ArrayD::from_elem(ndarray::IxDyn(&[2]), f32::INFINITY);
        let tensor = BoundedTensor::new_unchecked(lower, upper).unwrap();

        let bounds = bounded_tensor_to_bounds(&tensor);
        assert_eq!(bounds.len(), 2);
        assert!(bounds[0].lower().is_infinite());
        assert!(bounds[0].upper().is_infinite());
    }

    /// Regression test for #2663: NaN bounds must be sanitized to conservative
    /// (-inf, +inf) instead of panicking in Bound::new_allow_infinite.
    #[ntest::timeout(5000)]
    #[test]
    fn test_bounded_tensor_to_bounds_nan_sanitized_2663() {
        // BoundedTensor::new rejects NaN, so use new_unchecked to inject NaN
        // (simulating upstream corruption that escaped optimization guards).
        let lower =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![f32::NAN, -1.0, 0.5]).unwrap();
        let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, f32::NAN, 1.5]).unwrap();
        let tensor = BoundedTensor::new_unchecked(lower, upper).unwrap();

        let bounds = bounded_tensor_to_bounds(&tensor);
        assert_eq!(bounds.len(), 3);

        // Element 0: NaN lower → -inf, normal upper preserved
        assert_eq!(bounds[0].lower(), f32::NEG_INFINITY);
        assert_eq!(bounds[0].upper(), 1.0);

        // Element 1: normal lower preserved, NaN upper → +inf
        assert_eq!(bounds[1].lower(), -1.0);
        assert_eq!(bounds[1].upper(), f32::INFINITY);

        // Element 2: both normal, preserved as-is
        assert!((bounds[2].lower() - 0.5).abs() < 1e-6);
        assert!((bounds[2].upper() - 1.5).abs() < 1e-6);
    }

    /// Regression test for #3216: inverted bounds from CROWN numerical instability
    /// must be swapped (sound over-approximation), not panic.
    #[ntest::timeout(5000)]
    #[test]
    fn test_bounded_tensor_to_bounds_inverted_swaps_3216() {
        // Simulate CROWN computing lower=0.126 > upper=0.123 (inverted, both finite)
        let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![-1.0, 0.126, 2.0]).unwrap();
        let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 0.123, 3.0]).unwrap();
        let tensor = BoundedTensor::new_unchecked(lower, upper).unwrap();

        let bounds = bounded_tensor_to_bounds(&tensor);
        assert_eq!(bounds.len(), 3);

        // Element 0: normal, preserved
        assert!((bounds[0].lower() - (-1.0)).abs() < 1e-6);
        assert!((bounds[0].upper() - 1.0).abs() < 1e-6);

        // Element 1: inverted [0.126, 0.123] swapped to [0.123, 0.126]
        assert!((bounds[1].lower() - 0.123).abs() < 1e-6);
        assert!((bounds[1].upper() - 0.126).abs() < 1e-6);

        // Element 2: normal, preserved
        assert!((bounds[2].lower() - 2.0).abs() < 1e-6);
        assert!((bounds[2].upper() - 3.0).abs() < 1e-6);
    }

    /// Regression test for #3216: end-to-end — BetaCrownResult with inverted output
    /// bounds converts to VerificationResult without panicking.
    #[ntest::timeout(5000)]
    #[test]
    fn test_beta_crown_result_with_inverted_bounds_3216() {
        let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.5, 0.126]).unwrap();
        let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 0.123]).unwrap();
        let tensor = BoundedTensor::new_unchecked(lower, upper).unwrap();

        let beta = BetaCrownResult {
            result: BabVerificationStatus::Unknown {
                reason: "test".to_string(),
            },
            domains_explored: 1,
            time_elapsed: Duration::from_millis(100),
            max_depth_reached: 1,
            output_bounds: Some(tensor),
            cuts_generated: 0,
            domains_verified: 0,
        };

        // Must not panic — inverted bounds should be swapped
        let result: VerificationResult = beta.into();
        if let VerificationResult::Unknown { bounds, .. } = result {
            assert_eq!(bounds.len(), 2);
            // First bound: normal
            assert!((bounds[0].lower() - 0.5).abs() < 1e-6);
            assert!((bounds[0].upper() - 1.0).abs() < 1e-6);
            // Second bound: swapped from [0.126, 0.123] to [0.123, 0.126]
            assert!((bounds[1].lower() - 0.123).abs() < 1e-6);
            assert!((bounds[1].upper() - 0.126).abs() < 1e-6);
        } else {
            unreachable!("Expected Unknown result");
        }
    }
}
