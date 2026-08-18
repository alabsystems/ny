// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Arithmetic-only transport for a call-local resident Cut-CROWN shadow.
//!
//! These values are deliberately **not proof authority**.  They know nothing
//! about a graph, input domain, intermediate-bound generation, or exact facet
//! certificate.  A private `ny-propagate` authority boundary must bind those
//! semantic objects, build this transport, and consume it synchronously.
//!
//! The first slice is lower-only and one-target:
//!
//! - one ordered pair of neurons at one resident ReLU;
//! - one nonnegative multiplier vector per lower objective row;
//! - post, pre, and bias channels carried atomically; and
//! - a mandatory call-local deadline.
//!
//! No upper channel, cache token, `lA`, verdict, registry, environment gate, or
//! persistent identity is representable here.

use std::time::Instant;

use crate::{GpuCrownResult, NyError, Result};

/// Maximum exact-certified facets in the first resident shadow slice.
pub const RESIDENT_CUT_SHADOW_MAX_FACETS: usize = 8;

/// Maximum lower-objective rows in the first resident shadow slice.
pub const RESIDENT_CUT_SHADOW_MAX_ROWS: usize = 64;

/// Typed policy for the non-authoritative resident Cut-CROWN experiment.
///
/// `Shadow` can request an observation, but the backend contract always returns
/// the unchanged baseline as the only consumable bound.  In particular, this
/// enum is not proof authority and cannot authorize a verdict.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidentCutShadowPolicy {
    /// Call the pre-existing resident method without inspecting a cut carrier.
    #[default]
    Disabled,
    /// Request an observation-only cut evaluation when a backend implements it.
    Shadow,
}

/// Result disposition for one observation-only resident cut request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentCutShadowDisposition {
    /// The typed policy was [`ResidentCutShadowPolicy::Disabled`].
    Disabled,
    /// Shadow inputs were absent, malformed, stale at the API boundary, or late.
    Rejected,
    /// The backend has no audited resident cut arithmetic kernel yet.
    BackendUnavailable,
    /// A complete shadow observation was produced; the returned bound is still baseline.
    Observed,
}

/// One complete observation of a binding lower-objective row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentCutShadowObservation {
    binding_row: usize,
    baseline_lower: f32,
    shadow_lower: f32,
    delta: f32,
}

impl ResidentCutShadowObservation {
    /// Build a finite observation and recompute its delta in one place.
    pub fn try_new(binding_row: usize, baseline_lower: f32, shadow_lower: f32) -> Result<Self> {
        if !baseline_lower.is_finite() || !shadow_lower.is_finite() {
            return Err(NyError::NumericalInstability(
                "resident cut shadow observation contains a non-finite lower bound".into(),
            ));
        }
        let delta = shadow_lower - baseline_lower;
        if !delta.is_finite() {
            return Err(NyError::NumericalInstability(
                "resident cut shadow observation delta is non-finite".into(),
            ));
        }
        Ok(Self {
            binding_row,
            baseline_lower,
            shadow_lower,
            delta,
        })
    }

    /// Objective-row position in the exact seed ordering.
    pub const fn binding_row(self) -> usize {
        self.binding_row
    }

    /// Lower bound returned by the unchanged resident call.
    pub const fn baseline_lower(self) -> f32 {
        self.baseline_lower
    }

    /// Observation-only lower bound returned by the shadow arithmetic.
    pub const fn shadow_lower(self) -> f32 {
        self.shadow_lower
    }

    /// `shadow_lower - baseline_lower`.
    pub const fn delta(self) -> f32 {
        self.delta
    }
}

/// Bounds returned from the resident Cut-CROWN shadow seam.
///
/// `baseline` is always the unchanged result of the pre-existing resident
/// method.  An observation is telemetry only and has no conversion into a
/// verdict-bearing result.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidentCutShadowOutcome {
    baseline: GpuCrownResult,
    disposition: ResidentCutShadowDisposition,
    observation: Option<ResidentCutShadowObservation>,
}

impl ResidentCutShadowOutcome {
    const fn new(
        baseline: GpuCrownResult,
        disposition: ResidentCutShadowDisposition,
        observation: Option<ResidentCutShadowObservation>,
    ) -> Self {
        Self {
            baseline,
            disposition,
            observation,
        }
    }

    /// Wrap an unchanged baseline for a disabled request.
    pub const fn disabled(baseline: GpuCrownResult) -> Self {
        Self::new(baseline, ResidentCutShadowDisposition::Disabled, None)
    }

    /// Wrap an unchanged baseline after rejecting shadow inputs.
    pub const fn rejected(baseline: GpuCrownResult) -> Self {
        Self::new(baseline, ResidentCutShadowDisposition::Rejected, None)
    }

    /// Wrap an unchanged baseline when the resident cut kernel is unavailable.
    pub const fn backend_unavailable(baseline: GpuCrownResult) -> Self {
        Self::new(
            baseline,
            ResidentCutShadowDisposition::BackendUnavailable,
            None,
        )
    }

    /// Attach complete telemetry after binding it to the exact baseline row.
    pub fn try_observed(
        baseline: GpuCrownResult,
        observation: ResidentCutShadowObservation,
    ) -> Result<Self> {
        let baseline_row = baseline
            .lower_bounds
            .get(observation.binding_row)
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "resident cut shadow observation row is outside the baseline".into(),
                )
            })?;
        if baseline_row.to_bits() != observation.baseline_lower.to_bits() {
            return Err(NyError::SoundnessRefusal(
                "resident cut shadow observation is not bound to the exact baseline row".into(),
            ));
        }
        Ok(Self::new(
            baseline,
            ResidentCutShadowDisposition::Observed,
            Some(observation),
        ))
    }

    /// Borrow the unchanged resident result.
    pub const fn baseline(&self) -> &GpuCrownResult {
        &self.baseline
    }

    /// Consume the wrapper and recover the unchanged resident result.
    pub fn into_baseline(self) -> GpuCrownResult {
        self.baseline
    }

    /// Whether the request was disabled, rejected, unsupported, or observed.
    pub const fn disposition(&self) -> ResidentCutShadowDisposition {
        self.disposition
    }

    /// Completed telemetry, if and only if disposition is `Observed`.
    pub const fn observation(&self) -> Option<ResidentCutShadowObservation> {
        self.observation
    }
}

/// Stored lower-channel center plus an outward absolute source-error charge.
///
/// If `q` is the exact real cut reduction represented by this value, then
/// `q ∈ [value - source_abs_error, value + source_abs_error]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentLowerCutChannel {
    value: f32,
    source_abs_error: f32,
}

impl ResidentLowerCutChannel {
    /// Validate and construct one complete lower channel.
    pub fn try_new(value: f32, source_abs_error: f32) -> Result<Self> {
        if !value.is_finite() {
            return Err(NyError::NumericalInstability(
                "resident lower-cut channel value is non-finite".into(),
            ));
        }
        if !source_abs_error.is_finite() || source_abs_error < 0.0 {
            return Err(NyError::NumericalInstability(
                "resident lower-cut source error must be finite and nonnegative".into(),
            ));
        }
        Ok(Self {
            value: if value == 0.0 { 0.0 } else { value },
            source_abs_error: if source_abs_error == 0.0 {
                0.0
            } else {
                source_abs_error
            },
        })
    }

    /// Stored f32 channel center.
    pub const fn value(self) -> f32 {
        self.value
    }

    /// Outward absolute error already incurred by source reduction/conversion.
    pub const fn source_abs_error(self) -> f32 {
        self.source_abs_error
    }
}

/// Complete lower-only contribution for one exact seed row.
///
/// Zero multipliers are retained in their exact facet positions.  The private
/// semantic builder, not this arithmetic value, proves that the five reduced
/// channels correspond to these multipliers and exact-certified facets.
#[derive(Debug, PartialEq)]
pub struct ResidentLowerCutRow {
    multipliers: Box<[f32]>,
    pre: [ResidentLowerCutChannel; 2],
    post: [ResidentLowerCutChannel; 2],
    bias: ResidentLowerCutChannel,
}

impl ResidentLowerCutRow {
    /// Validate a complete row before it can enter a carrier.
    pub fn try_new(
        multipliers: Vec<f32>,
        pre: [ResidentLowerCutChannel; 2],
        post: [ResidentLowerCutChannel; 2],
        bias: ResidentLowerCutChannel,
    ) -> Result<Self> {
        if multipliers.is_empty() || multipliers.len() > RESIDENT_CUT_SHADOW_MAX_FACETS {
            return Err(NyError::InvalidSpec(format!(
                "resident lower-cut multiplier count {} is outside 1..={RESIDENT_CUT_SHADOW_MAX_FACETS}",
                multipliers.len()
            )));
        }
        if let Some(value) = multipliers
            .iter()
            .copied()
            .find(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(NyError::InvalidSpec(format!(
                "resident lower-cut multiplier must be finite and nonnegative, got {value}"
            )));
        }
        let multipliers = multipliers
            .into_iter()
            .map(|value| if value == 0.0 { 0.0 } else { value })
            .collect();
        Ok(Self {
            multipliers,
            pre,
            post,
            bias,
        })
    }

    /// Exact facet-order multiplier snapshot, including zero entries.
    pub fn multipliers(&self) -> &[f32] {
        &self.multipliers
    }

    /// Two pre-activation lower-channel contributions.
    pub const fn pre(&self) -> &[ResidentLowerCutChannel; 2] {
        &self.pre
    }

    /// Two post-activation lower-channel contributions.
    pub const fn post(&self) -> &[ResidentLowerCutChannel; 2] {
        &self.post
    }

    /// The single `-lambda * b` lower-bias contribution.
    pub const fn bias(&self) -> ResidentLowerCutChannel {
        self.bias
    }
}

/// Arithmetic-only carrier for one target ReLU in one resident call.
///
/// It is intentionally owned, non-`Clone`, and non-serializable.  Semantic
/// proof remains in a private call-local wrapper that retains the exact facet
/// certificates and an opaque request seal.
#[derive(Debug, PartialEq)]
pub struct ResidentLowerCutCarrier {
    target_activation: usize,
    target_width: usize,
    ordered_neurons: [usize; 2],
    rows: Box<[ResidentLowerCutRow]>,
    deadline: Instant,
}

impl ResidentLowerCutCarrier {
    /// Validate every arithmetic component before constructing the carrier.
    pub fn try_new(
        target_activation: usize,
        target_width: usize,
        ordered_neurons: [usize; 2],
        rows: Vec<ResidentLowerCutRow>,
        deadline: Instant,
    ) -> Result<Self> {
        if target_width == 0 {
            return Err(NyError::InvalidSpec(
                "resident lower-cut target width is zero".into(),
            ));
        }
        let [first, second] = ordered_neurons;
        if first == second || first >= target_width || second >= target_width {
            return Err(NyError::InvalidSpec(format!(
                "resident lower-cut ordered pair ({first}, {second}) is invalid for width {target_width}"
            )));
        }
        if rows.is_empty() || rows.len() > RESIDENT_CUT_SHADOW_MAX_ROWS {
            return Err(NyError::InvalidSpec(format!(
                "resident lower-cut row count {} is outside 1..={RESIDENT_CUT_SHADOW_MAX_ROWS}",
                rows.len()
            )));
        }
        let facet_count = rows[0].multipliers.len();
        if rows.iter().any(|row| row.multipliers.len() != facet_count) {
            return Err(NyError::InvalidSpec(
                "resident lower-cut rows do not share one exact facet ordering".into(),
            ));
        }
        Ok(Self {
            target_activation,
            target_width,
            ordered_neurons,
            rows: rows.into_boxed_slice(),
            deadline,
        })
    }

    /// Validate this carrier against the exact backend-call shape and deadline.
    pub fn validate_for_call(
        &self,
        activation_count: usize,
        activation_width: usize,
        expected_rows: usize,
        deadline: Instant,
    ) -> Result<()> {
        if self.target_activation >= activation_count {
            return Err(NyError::InvalidSpec(format!(
                "resident lower-cut target activation {} is outside activation count {activation_count}",
                self.target_activation
            )));
        }
        if self.target_width != activation_width {
            return Err(NyError::InvalidSpec(format!(
                "resident lower-cut target width {} does not match resident width {activation_width}",
                self.target_width
            )));
        }
        if self.rows.len() != expected_rows {
            return Err(NyError::InvalidSpec(format!(
                "resident lower-cut row count {} does not match seed rows {expected_rows}",
                self.rows.len()
            )));
        }
        if self.deadline != deadline {
            return Err(NyError::SoundnessRefusal(
                "resident lower-cut deadline is not the exact call-local deadline".into(),
            ));
        }
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "resident lower-cut deadline expired before backend mutation".into(),
            ));
        }
        Ok(())
    }

    /// Target activation in the exact resident fold order.
    pub const fn target_activation(&self) -> usize {
        self.target_activation
    }

    /// Declared width of the exact target ReLU.
    pub const fn target_width(&self) -> usize {
        self.target_width
    }

    /// Exact ordered pair within the target ReLU.
    pub const fn ordered_neurons(&self) -> [usize; 2] {
        self.ordered_neurons
    }

    /// Lower rows in the exact objective/seed ordering.
    pub fn rows(&self) -> &[ResidentLowerCutRow] {
        &self.rows
    }

    /// Mandatory call-local deadline.
    pub const fn deadline(&self) -> Instant {
        self.deadline
    }

    /// Whether at least one retained multiplier is strictly positive.
    pub fn has_nonzero_multiplier(&self) -> bool {
        self.rows
            .iter()
            .flat_map(|row| row.multipliers.iter())
            .any(|&value| value > 0.0)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::{GpuCrownBackward, GpuCrownLayer, GpuCrownSeed, GpuResnetSegment};

    fn channel(value: f32) -> ResidentLowerCutChannel {
        ResidentLowerCutChannel::try_new(value, 0.0).expect("finite exact test channel")
    }

    fn carrier(deadline: Instant) -> ResidentLowerCutCarrier {
        ResidentLowerCutCarrier::try_new(
            0,
            2,
            [0, 1],
            vec![ResidentLowerCutRow::try_new(
                vec![0.0, 1.0],
                [channel(-0.5), channel(-0.5)],
                [channel(1.0), channel(1.0)],
                channel(-1.0),
            )
            .expect("valid test row")],
            deadline,
        )
        .expect("valid test carrier")
    }

    fn seed() -> GpuCrownSeed {
        GpuCrownSeed {
            lower_a: Arc::from([1.0_f32]),
            upper_a: Arc::from([1.0_f32]),
            lower_b: Arc::from([0.0_f32]),
            upper_b: Arc::from([0.0_f32]),
            num_specs: 1,
            current_dim: 1,
        }
    }

    struct BaselineEngine {
        calls: AtomicUsize,
        result: GpuCrownResult,
    }

    impl GpuCrownBackward for BaselineEngine {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp("not used by shadow test".into()))
        }

        fn crown_backward_gpu_resnet_sound_beta(
            &self,
            _segments: &[GpuResnetSegment],
            _seed: &GpuCrownSeed,
            _input_lower: &[f32],
            _input_upper: &[f32],
            _beta_signed: &[Vec<f32>],
            _frontier_abs: &[Vec<f32>],
            _node_abs: &[Vec<f32>],
        ) -> Result<GpuCrownResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.result.clone())
        }
    }

    #[test]
    fn disabled_policy_calls_unchanged_baseline_before_shadow_inspection() {
        let expected = GpuCrownResult {
            lower_bounds: vec![f32::from_bits(0xbead_beef)],
            upper_bounds: vec![f32::from_bits(0x3f12_3456)],
        };
        let engine = BaselineEngine {
            calls: AtomicUsize::new(0),
            result: expected.clone(),
        };
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("one-second monotonic clock history");
        let outcome = engine
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Disabled,
                &[],
                &seed(),
                &[],
                &[],
                &[],
                &[],
                &[],
                None,
                usize::MAX,
                expired,
            )
            .expect("disabled wrapper must delegate");

        assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            outcome.disposition(),
            ResidentCutShadowDisposition::Disabled
        );
        assert!(outcome.observation().is_none());
        assert_eq!(outcome.into_baseline(), expected);
    }

    #[test]
    fn valid_shadow_is_explicitly_unavailable_and_still_returns_baseline() {
        let expected = GpuCrownResult {
            lower_bounds: vec![-3.0],
            upper_bounds: vec![0.0],
        };
        let engine = BaselineEngine {
            calls: AtomicUsize::new(0),
            result: expected.clone(),
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        let future_carrier = carrier(deadline);
        let outcome = engine
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Shadow,
                &[GpuResnetSegment::Chain(vec![GpuCrownLayer::Activation {
                    lower_slope: vec![0.0, 0.0],
                    upper_slope: vec![1.0, 1.0],
                    lower_intercept: vec![0.0, 0.0],
                    upper_intercept: vec![0.0, 0.0],
                    num_neurons: 2,
                }])],
                &seed(),
                &[],
                &[],
                &[],
                &[],
                &[],
                Some(&future_carrier),
                0,
                deadline,
            )
            .expect("unsupported shadow must retain baseline");

        assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            outcome.disposition(),
            ResidentCutShadowDisposition::BackendUnavailable
        );
        assert!(outcome.observation().is_none());
        assert_eq!(outcome.into_baseline(), expected);
    }

    #[test]
    fn malformed_or_expired_shadow_is_rejected_without_bound_publication() {
        let expected = GpuCrownResult {
            lower_bounds: vec![-3.0],
            upper_bounds: vec![0.0],
        };
        let engine = BaselineEngine {
            calls: AtomicUsize::new(0),
            result: expected.clone(),
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        let malformed_carrier = carrier(deadline);
        let outcome = engine
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Shadow,
                &[GpuResnetSegment::Chain(vec![GpuCrownLayer::Activation {
                    lower_slope: vec![0.0, 0.0],
                    upper_slope: vec![1.0, 1.0],
                    lower_intercept: vec![0.0, 0.0],
                    upper_intercept: vec![0.0, 0.0],
                    num_neurons: 2,
                }])],
                &seed(),
                &[],
                &[],
                &[],
                &[],
                &[],
                Some(&malformed_carrier),
                7,
                deadline,
            )
            .expect("malformed shadow must retain baseline");

        assert_eq!(
            outcome.disposition(),
            ResidentCutShadowDisposition::Rejected
        );
        assert!(outcome.observation().is_none());
        assert_eq!(outcome.into_baseline(), expected);

        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("one-second monotonic clock history");
        let expired_carrier = carrier(expired);
        let outcome = engine
            .crown_backward_gpu_resnet_sound_beta_cut_shadow(
                ResidentCutShadowPolicy::Shadow,
                &[GpuResnetSegment::Chain(vec![GpuCrownLayer::Activation {
                    lower_slope: vec![0.0, 0.0],
                    upper_slope: vec![1.0, 1.0],
                    lower_intercept: vec![0.0, 0.0],
                    upper_intercept: vec![0.0, 0.0],
                    num_neurons: 2,
                }])],
                &seed(),
                &[],
                &[],
                &[],
                &[],
                &[],
                Some(&expired_carrier),
                0,
                expired,
            )
            .expect("expired shadow must retain baseline");
        assert_eq!(
            outcome.disposition(),
            ResidentCutShadowDisposition::Rejected
        );
        assert!(outcome.observation().is_none());
        assert_eq!(outcome.into_baseline(), engine.result);
    }

    #[test]
    fn lower_transport_validates_sign_finiteness_shape_and_retains_zero() {
        for invalid in [f32::NAN, f32::INFINITY, -0.25] {
            assert!(ResidentLowerCutRow::try_new(
                vec![invalid],
                [channel(0.0), channel(0.0)],
                [channel(0.0), channel(0.0)],
                channel(0.0),
            )
            .is_err());
        }
        assert!(ResidentLowerCutChannel::try_new(f32::NAN, 0.0).is_err());
        assert!(ResidentLowerCutChannel::try_new(0.0, -f32::from_bits(1)).is_err());

        let carrier = carrier(Instant::now() + Duration::from_secs(30));
        assert_eq!(carrier.rows()[0].multipliers(), &[0.0, 1.0]);
        assert!(carrier.has_nonzero_multiplier());
        assert!(
            carrier
                .validate_for_call(1, 3, 1, carrier.deadline())
                .is_err(),
            "target width mismatch must reject the whole carrier"
        );
        assert!(
            carrier
                .validate_for_call(1, 2, 2, carrier.deadline())
                .is_err(),
            "seed row mismatch must reject the whole carrier"
        );
    }

    #[test]
    fn completed_observation_is_bound_to_the_exact_baseline_row() {
        let baseline = GpuCrownResult {
            lower_bounds: vec![-3.0],
            upper_bounds: vec![0.0],
        };
        let observation =
            ResidentCutShadowObservation::try_new(0, -3.0, -2.0).expect("finite observation");
        let outcome = ResidentCutShadowOutcome::try_observed(baseline.clone(), observation)
            .expect("exact baseline binding");
        assert_eq!(
            outcome.disposition(),
            ResidentCutShadowDisposition::Observed
        );
        assert_eq!(outcome.observation(), Some(observation));
        assert_eq!(outcome.into_baseline(), baseline);

        let mismatched =
            ResidentCutShadowObservation::try_new(0, -4.0, -2.0).expect("finite mismatch");
        assert!(ResidentCutShadowOutcome::try_observed(
            GpuCrownResult {
                lower_bounds: vec![-3.0],
                upper_bounds: vec![0.0],
            },
            mismatched,
        )
        .is_err());
    }
}
