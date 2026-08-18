// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified inference BatchNorm for an unwired constrained zonotope.
//!
//! ONNX inference BatchNorm is a diagonal affine map
//! `y = gamma * (x - mean) / sqrt(variance + epsilon) + beta`.  This
//! primitive derives one nominal binary64 scale and bias per channel while
//! enclosing their distance from the real coefficients.  It then preserves
//! every constrained-zonotope predicate and sparse generator column, charging
//! both arithmetic width and
//! `scale_error * max_abs_input + bias_error`
//! to the independent output box remainder.
//!
//! The flat value order is described explicitly by `input_shape` and
//! `channel_axis`.  Thus `[N, C, H, W]` uses axis 1, while a squeezed or
//! flattened channel-major `[C, ...]` tensor uses axis 0.  An explicit axis is
//! preferable to guessing from equal extents: a mistaken batch/channel choice
//! would apply the wrong affine and is therefore a proof-soundness issue.
//!
//! This module is deliberately **unwired**.  It does not read ONNX nodes,
//! normalize attributes, run on CUDA, or affect a verdict.  The raw finite
//! binary64 parameters supplied here are interpreted as exact real values.

use std::cmp::Ordering;

use ndarray::Array2;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::constrained_zonotope64::ConstrainedZonotope64CallGateError;
use crate::constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallBudget, ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallGate,
    ConstrainedZonotopeCallOutcome, ConstrainedZonotopeCallTracker,
    ConstrainedZonotopePeakLiveBytes, InertConstrainedZonotopeCallGate,
};
use crate::{ConstrainedZonotope64, ConstrainedZonotope64Error};

const MAX_SQRT_REFINEMENTS_PER_CHANNEL: usize = 8;
// Exact-rational certification processes one channel at a time.  This covers
// all simultaneously live <=4,300-bit numerator/denominator payloads with a
// deliberately wide moat; allocator metadata remains in the caller baseline.
const BATCH_NORM_RATIONAL_SCRATCH_BYTES: usize = 64 * 1024;
// A published exact error owns a numerator and denominator whose sizes depend
// on the raw binary64 inputs.  Charge the same deliberately wide moat for each
// retained rational instead of relying on `size_of::<BigRational>()`, which
// accounts only for its inline handles.
const BATCH_NORM_RETAINED_RATIONAL_BYTES: usize = 64 * 1024;

/// BatchNorm execution semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstrainedZonotopeBatchNormMode {
    /// Use fixed running mean and variance.
    Inference,
    /// Derive statistics from the current batch.
    ///
    /// This nonlinear, cross-coordinate operation is intentionally rejected by
    /// [`constrained_zonotope_batch_norm_unwired`].
    Training,
}

/// Raw inference BatchNorm data and explicit row-major layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstrainedZonotopeBatchNormSpec<'a> {
    /// Logical input tensor shape.  Its checked product must equal the domain's
    /// flat value dimension.
    pub input_shape: &'a [usize],
    /// Channel dimension within `input_shape`.
    pub channel_axis: usize,
    /// ONNX scale (`gamma`), one exact binary64 value per channel.
    pub gamma: &'a [f64],
    /// ONNX additive parameter (`beta`), one value per channel.
    pub beta: &'a [f64],
    /// Fixed running mean, one value per channel.
    pub mean: &'a [f64],
    /// Fixed running variance, one nonnegative value per channel.
    pub variance: &'a [f64],
    /// Strictly positive inference epsilon.
    pub epsilon: f64,
    /// Requested execution semantics.
    pub mode: ConstrainedZonotopeBatchNormMode,
}

/// Resource limits for certifying a caller-declared BatchNorm affine
/// surrogate.
///
/// There is intentionally no `Default`.  The declared surrogate contributes
/// two parameter elements per channel in addition to the four authored
/// BatchNorm arrays.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeBatchNormAffineCertificateLimits {
    /// Maximum logical input rank carried by the BatchNorm specification.
    pub max_rank: usize,
    /// Maximum BatchNorm channel count.
    pub max_channel_count: usize,
    /// Maximum authored plus declared-surrogate elements (`6 * channels`).
    pub max_parameter_elements: usize,
}

/// Exact error of one caller-declared affine surrogate from the authored real
/// BatchNorm affine.
///
/// Each value is the maximum exact rational distance from the declared
/// binary64 nominal to the corresponding lower and upper affine endpoints.
/// Those endpoints are derived from an exact rational bracket of
/// `sqrt(variance + epsilon)`; no rounded graph error certificate is trusted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactBatchNormChannelAffineCertificate {
    scale_error: BigRational,
    bias_error: BigRational,
}

impl ExactBatchNormChannelAffineCertificate {
    /// Exact nonnegative error of the declared scale surrogate.
    #[must_use]
    pub const fn scale_error(&self) -> &BigRational {
        &self.scale_error
    }

    /// Exact nonnegative error of the declared bias surrogate.
    #[must_use]
    pub const fn bias_error(&self) -> &BigRational {
        &self.bias_error
    }
}

/// Completed exact certificate for one declared BatchNorm affine surrogate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactBatchNormAffineSurrogateCertificate {
    channels: Vec<ExactBatchNormChannelAffineCertificate>,
    sqrt_refinements: usize,
    conservative_live_bytes: usize,
}

impl ExactBatchNormAffineSurrogateCertificate {
    /// Per-channel exact surrogate errors in authored channel order.
    #[must_use]
    pub fn channels(&self) -> &[ExactBatchNormChannelAffineCertificate] {
        &self.channels
    }

    /// Adjacent-float steps used only to locate exact rational square-root
    /// brackets.  The published errors themselves remain exact rationals.
    #[must_use]
    pub const fn sqrt_refinements(&self) -> usize {
        self.sqrt_refinements
    }

    /// Conservative logical bytes retained by this certificate after its call
    /// returns.  A caller retaining it across another budgeted call must add
    /// these bytes to that call's baseline.
    #[must_use]
    pub const fn conservative_live_bytes(&self) -> usize {
        self.conservative_live_bytes
    }
}

/// Explicit resource limits for
/// [`constrained_zonotope_batch_norm_unwired`].
///
/// There is intentionally no `Default`: an experimental caller must price
/// every retained structure and every explicit transform walk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeBatchNormLimits {
    /// Maximum flat input/output value dimension.
    pub max_value_count: usize,
    /// Maximum logical input rank.
    pub max_rank: usize,
    /// Maximum BatchNorm channel count.
    pub max_channel_count: usize,
    /// Maximum alpha dimension.
    pub max_alpha_dim: usize,
    /// Maximum sparse generator nonzeros in either input or output.
    pub max_generator_nonzeros: usize,
    /// Maximum raw parameter/statistic elements (`4 * channels`).
    pub max_parameter_elements: usize,
    /// Maximum full-coordinate visits.  The transform makes two.
    pub max_coordinate_visits: usize,
    /// Maximum sparse-entry visits.  The transform makes two.
    pub max_generator_visits: usize,
    /// Maximum interval multiplications performed by the transform.
    pub max_interval_products: usize,
    /// Maximum retained predicate rows, including zero-width rows.
    pub max_constraint_count: usize,
    /// Maximum retained constraint-matrix elements.
    pub max_constraint_elements: usize,
}

/// Checked shape and work accounting for one completed BatchNorm transform.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeBatchNormPlan {
    /// Logical input rank.
    pub input_rank: usize,
    /// Explicit channel axis.
    pub channel_axis: usize,
    /// Product of dimensions before the channel axis.
    pub outer_count: usize,
    /// Channel count.
    pub channel_count: usize,
    /// Product of dimensions after the channel axis.
    pub elements_per_channel: usize,
    /// Flat input/output value count.
    pub value_count: usize,
    /// Alpha dimension, unchanged by this diagonal affine.
    pub alpha_dim: usize,
    /// Constraint count, unchanged by this diagonal affine.
    pub constraint_count: usize,
    /// Retained constraint-matrix elements.
    pub constraint_elements: usize,
    /// Raw parameter/statistic elements inspected.
    pub parameter_elements: usize,
    /// Full-coordinate visits.
    pub coordinate_visits: usize,
    /// Sparse generator-entry visits.
    pub generator_visits: usize,
    /// Sparse nonzeros in the input generators.
    pub input_generator_nonzeros: usize,
    /// Sparse nonzeros retained in the output generators.
    pub output_generator_nonzeros: usize,
    /// Interval products actually evaluated.
    pub interval_products: usize,
    /// Adjacent-float steps used to bracket exact square roots.
    pub sqrt_refinements: usize,
}

/// Invalid BatchNorm data, exhausted resources, or arithmetic that could not
/// be enclosed by finite binary64 endpoints.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotopeBatchNormError {
    /// A tensor/vector has the wrong shape.
    #[error("shape mismatch for {field}: expected {expected:?}, got {got:?}")]
    Shape {
        /// Rejected input.
        field: &'static str,
        /// Required shape.
        expected: Vec<usize>,
        /// Supplied shape.
        got: Vec<usize>,
    },

    /// A layout or other structural field is invalid.
    #[error("invalid BatchNorm specification: {message}")]
    InvalidSpec {
        /// Concrete rejected precondition.
        message: String,
    },

    /// Training BatchNorm is not a fixed diagonal affine.
    #[error("unsupported BatchNorm semantics: {semantics}")]
    UnsupportedSemantics {
        /// Rejected semantics.
        semantics: &'static str,
    },

    /// A supplied parameter or statistic is NaN or infinite.
    #[error("{field}[{index}] must be finite")]
    NonFinite {
        /// Rejected input.
        field: &'static str,
        /// Flattened index.
        index: usize,
    },

    /// Epsilon is not strictly positive and finite.
    #[error("BatchNorm epsilon must be finite and strictly positive")]
    InvalidEpsilon,

    /// Running variance is negative.
    #[error("variance[{index}] must be nonnegative")]
    InvalidVariance {
        /// Rejected channel.
        index: usize,
    },

    /// Checked resource arithmetic overflowed.
    #[error("resource size overflow while computing {operation}")]
    ResourceOverflow {
        /// Failed calculation.
        operation: &'static str,
    },

    /// A caller-selected cap was exceeded.
    #[error("resource limit exceeded for {resource}: required {required}, limit {limit}")]
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Required count.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },

    /// A bounded allocation failed.
    #[error("unable to reserve storage for {resource}")]
    AllocationFailure {
        /// Requested buffer.
        resource: &'static str,
    },

    /// Outward arithmetic reached infinity, NaN, or an unverifiable endpoint.
    #[error("non-finite outward arithmetic while computing {operation}")]
    NonFiniteArithmetic {
        /// Failed operation.
        operation: &'static str,
    },

    /// The host does not preserve binary64 subnormals.
    #[error("unsupported floating-point environment: {requirement}")]
    UnsupportedFloatingPoint {
        /// Required IEEE behavior.
        requirement: &'static str,
    },

    /// The validated result could not be materialized as a domain.
    #[error(transparent)]
    Domain(#[from] ConstrainedZonotope64Error),
}

/// Primitive or call-firewall refusal from budgeted BatchNorm.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConstrainedZonotopeBatchNormBudgetError {
    /// BatchNorm data, limits, or outward arithmetic were invalid.
    #[error(transparent)]
    Transform(#[from] ConstrainedZonotopeBatchNormError),

    /// The caller's deadline or aggregate peak-memory ceiling refused work.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
}

/// Apply inference BatchNorm to a flat constrained zonotope.
///
/// The input and output use row-major `spec.input_shape` order.  For each flat
/// coordinate `p`, the channel is
/// `(p / product(shape[channel_axis + 1..])) % shape[channel_axis]`.
/// `C alpha <= d` is copied unchanged.
pub fn constrained_zonotope_batch_norm_unwired(
    input: &ConstrainedZonotope64,
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    limits: ConstrainedZonotopeBatchNormLimits,
) -> Result<
    (ConstrainedZonotope64, ConstrainedZonotopeBatchNormPlan),
    ConstrainedZonotopeBatchNormError,
> {
    let mut gate = InertConstrainedZonotopeCallGate;
    match constrained_zonotope_batch_norm_impl(input, spec, limits, &mut gate) {
        Ok(value) => Ok(value),
        Err(ConstrainedZonotopeBatchNormBudgetError::Transform(error)) => Err(error),
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(_)) => {
            unreachable!("the inert BatchNorm call gate cannot refuse work")
        }
    }
}

/// Apply inference BatchNorm behind a synchronous call-local execution
/// firewall.
///
/// The complete transform-owned logical peak is preflighted before scratch or
/// output allocation.  `budget.baseline_live_bytes()` must include the input,
/// parameters, and any other caller-retained storage that shares the hard
/// ceiling.  A completed domain remains private until the final deadline
/// checkpoint.
pub fn constrained_zonotope_batch_norm_unwired_with_budget(
    input: &ConstrainedZonotope64,
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    limits: ConstrainedZonotopeBatchNormLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<
    ConstrainedZonotopeCallOutcome<(ConstrainedZonotope64, ConstrainedZonotopeBatchNormPlan)>,
    ConstrainedZonotopeBatchNormBudgetError,
> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let value = constrained_zonotope_batch_norm_impl(input, spec, limits, &mut gate)?;
    let report = gate.report();
    Ok(ConstrainedZonotopeCallOutcome::new(value, report))
}

/// Certify exact errors for a caller-declared inference BatchNorm affine
/// surrogate.
///
/// `nominal_scale` and `nominal_bias` are interpreted as exact binary64
/// values.  They may, for example, be exact promotions of normalized graph
/// binary32 coefficients.  For every channel this function brackets the
/// authored real square root with exact dyadic endpoints, derives exact
/// rational affine endpoints, and returns the maximum exact distance from the
/// declared nominal to those endpoints.
pub fn certify_batch_norm_affine_surrogate_unwired(
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    nominal_scale: &[f64],
    nominal_bias: &[f64],
    limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
) -> Result<ExactBatchNormAffineSurrogateCertificate, ConstrainedZonotopeBatchNormError> {
    let mut gate = InertConstrainedZonotopeCallGate;
    match certify_batch_norm_affine_surrogate_impl(
        spec,
        nominal_scale,
        nominal_bias,
        limits,
        &mut gate,
    ) {
        Ok(value) => Ok(value),
        Err(ConstrainedZonotopeBatchNormBudgetError::Transform(error)) => Err(error),
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(_)) => {
            unreachable!("the inert BatchNorm call gate cannot refuse work")
        }
    }
}

/// Certify a declared inference BatchNorm affine surrogate behind the shared
/// synchronous call-local execution firewall.
///
/// `budget.baseline_live_bytes()` must include the borrowed specification and
/// declared-surrogate slices plus any other caller-retained storage.  The
/// returned certificate reports its conservative retained size for pricing
/// subsequent calls while it remains live.
pub fn certify_batch_norm_affine_surrogate_unwired_with_budget(
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    nominal_scale: &[f64],
    nominal_bias: &[f64],
    limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<
    ConstrainedZonotopeCallOutcome<ExactBatchNormAffineSurrogateCertificate>,
    ConstrainedZonotopeBatchNormBudgetError,
> {
    let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
    let value = certify_batch_norm_affine_surrogate_impl(
        spec,
        nominal_scale,
        nominal_bias,
        limits,
        &mut gate,
    )?;
    let report = gate.report();
    Ok(ConstrainedZonotopeCallOutcome::new(value, report))
}

#[cfg(test)]
fn certify_batch_norm_affine_surrogate_unwired_with_clock<N>(
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    nominal_scale: &[f64],
    nominal_bias: &[f64],
    limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<
    ConstrainedZonotopeCallOutcome<ExactBatchNormAffineSurrogateCertificate>,
    ConstrainedZonotopeBatchNormBudgetError,
>
where
    N: FnMut(&'static str) -> std::time::Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let value = certify_batch_norm_affine_surrogate_impl(
        spec,
        nominal_scale,
        nominal_bias,
        limits,
        &mut gate,
    )?;
    let report = gate.report();
    Ok(ConstrainedZonotopeCallOutcome::new(value, report))
}

#[cfg(test)]
fn constrained_zonotope_batch_norm_unwired_with_clock<N>(
    input: &ConstrainedZonotope64,
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    limits: ConstrainedZonotopeBatchNormLimits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<
    ConstrainedZonotopeCallOutcome<(ConstrainedZonotope64, ConstrainedZonotopeBatchNormPlan)>,
    ConstrainedZonotopeBatchNormBudgetError,
>
where
    N: FnMut(&'static str) -> std::time::Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let value = constrained_zonotope_batch_norm_impl(input, spec, limits, &mut gate)?;
    let report = gate.report();
    Ok(ConstrainedZonotopeCallOutcome::new(value, report))
}

pub(crate) fn certify_batch_norm_affine_surrogate_impl<G>(
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    nominal_scale: &[f64],
    nominal_bias: &[f64],
    limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
    gate: &mut G,
) -> Result<ExactBatchNormAffineSurrogateCertificate, ConstrainedZonotopeBatchNormBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    require_gradual_underflow()?;
    gate.checkpoint("BatchNorm surrogate-certificate floating-point preflight")?;
    let channel_count = validate_affine_certificate_spec_with_gate(
        spec,
        nominal_scale,
        nominal_bias,
        limits,
        gate,
    )?;
    gate.checkpoint("BatchNorm surrogate-certificate validation complete")?;
    let conservative_live_bytes = batch_norm_affine_certificate_live_bytes(channel_count)?;
    if gate.is_enforcing() {
        let transform_peak = batch_norm_affine_certificate_peak_live_bytes(channel_count)?;
        gate.preflight_peak_live_bytes(transform_peak)?;
    }
    gate.checkpoint("BatchNorm surrogate-certificate peak-memory preflight complete")?;
    let (channels, sqrt_refinements) =
        certify_declared_channel_affines_with_gate(spec, nominal_scale, nominal_bias, gate)?;
    gate.checkpoint("BatchNorm surrogate-certificate coefficient certification complete")?;
    let certificate = ExactBatchNormAffineSurrogateCertificate {
        channels,
        sqrt_refinements,
        conservative_live_bytes,
    };
    gate.checkpoint("BatchNorm surrogate-certificate publication")?;
    Ok(certificate)
}

fn constrained_zonotope_batch_norm_impl<G>(
    input: &ConstrainedZonotope64,
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    limits: ConstrainedZonotopeBatchNormLimits,
    gate: &mut G,
) -> Result<
    (ConstrainedZonotope64, ConstrainedZonotopeBatchNormPlan),
    ConstrainedZonotopeBatchNormBudgetError,
>
where
    G: ConstrainedZonotopeCallGate,
{
    require_gradual_underflow()?;
    gate.checkpoint("BatchNorm floating-point preflight")?;
    let geometry = validate_geometry_with_gate(input, spec, limits, gate)?;
    gate.checkpoint("BatchNorm geometry validation complete")?;
    if gate.is_enforcing() {
        gate.preflight_peak_live_bytes(batch_norm_peak_live_bytes(input, geometry)?)?;
    }
    gate.checkpoint("BatchNorm peak-memory preflight complete")?;
    let (channel_affines, sqrt_refinements) = certify_channel_affines_with_gate(spec, gate)?;
    gate.checkpoint("BatchNorm coefficient certification complete")?;

    let mut input_magnitude = Vec::new();
    gate.checkpoint("BatchNorm input-magnitude allocation")?;
    try_reserve(
        &mut input_magnitude,
        geometry.value_count,
        "input magnitude bounds",
    )?;
    for index in 0..geometry.value_count {
        gate.charge_items(1, "BatchNorm input-magnitude coordinates")?;
        input_magnitude.push(add_nonnegative_upper(
            input.center()[index].abs(),
            input.box_remainder()[index],
        )?);
    }
    for generator in input.generators() {
        gate.charge_items(1, "BatchNorm input-magnitude generators")?;
        for (value_index, coefficient) in generator.entries() {
            gate.charge_items(1, "BatchNorm input-magnitude generators")?;
            input_magnitude[value_index] =
                add_nonnegative_upper(input_magnitude[value_index], coefficient.abs())?;
        }
    }
    gate.checkpoint("BatchNorm input-magnitude phase complete")?;

    let mut output_center = Vec::new();
    let mut output_remainder = Vec::new();
    gate.checkpoint("BatchNorm output-center allocation")?;
    try_reserve(
        &mut output_center,
        geometry.value_count,
        "BatchNorm output center",
    )?;
    gate.checkpoint("BatchNorm output-remainder allocation")?;
    try_reserve(
        &mut output_remainder,
        geometry.value_count,
        "BatchNorm output remainder",
    )?;

    let mut interval_products = 0_usize;
    for value_index in 0..geometry.value_count {
        gate.charge_items(1, "BatchNorm coordinate transform")?;
        let channel = geometry.channel_for_flat_index(value_index);
        let affine = channel_affines[channel];

        let center_interval = if affine.scale == 0.0 {
            OutwardInterval::exact(affine.bias)
        } else {
            consume_product(&mut interval_products, limits.max_interval_products)?;
            outward_product(affine.scale, input.center()[value_index])?
                .add(OutwardInterval::exact(affine.bias))?
        };
        let (nominal_center, center_error) = center_interval.nominal_and_error()?;

        let mut total_remainder = 0.0;
        let input_remainder = input.box_remainder()[value_index];
        if affine.scale != 0.0 && input_remainder != 0.0 {
            consume_product(&mut interval_products, limits.max_interval_products)?;
            total_remainder = outward_product(affine.scale.abs(), input_remainder)?
                .abs()
                .hi;
        }

        let mut parameter_error = affine.bias_error;
        if affine.scale_error != 0.0 && input_magnitude[value_index] != 0.0 {
            consume_product(&mut interval_products, limits.max_interval_products)?;
            let scale_error = outward_product(affine.scale_error, input_magnitude[value_index])?
                .abs()
                .hi;
            parameter_error = add_nonnegative_upper(parameter_error, scale_error)?;
        }
        total_remainder = add_nonnegative_upper(total_remainder, parameter_error)?;
        total_remainder = add_nonnegative_upper(total_remainder, center_error)?;

        output_center.push(nominal_center);
        output_remainder.push(total_remainder);
    }
    gate.checkpoint("BatchNorm coordinate transform complete")?;

    let mut output_generators = Vec::new();
    gate.checkpoint("BatchNorm generator-column allocation")?;
    try_reserve(
        &mut output_generators,
        input.alpha_dim(),
        "BatchNorm output generator columns",
    )?;
    let mut output_generator_nonzeros = 0_usize;
    for generator in input.generators() {
        let mut output_entries = Vec::new();
        gate.checkpoint("BatchNorm generator-entry allocation")?;
        try_reserve(
            &mut output_entries,
            generator.nnz(),
            "BatchNorm output generator coefficients",
        )?;
        for (value_index, coefficient) in generator.entries() {
            gate.charge_items(1, "BatchNorm generator transform")?;
            let channel = geometry.channel_for_flat_index(value_index);
            let scale = channel_affines[channel].scale;
            if scale == 0.0 {
                continue;
            }
            consume_product(&mut interval_products, limits.max_interval_products)?;
            let (nominal, representation_error) =
                outward_product(scale, coefficient)?.nominal_and_error()?;
            output_remainder[value_index] =
                add_nonnegative_upper(output_remainder[value_index], representation_error)?;
            if nominal != 0.0 {
                output_generator_nonzeros = output_generator_nonzeros.checked_add(1).ok_or(
                    ConstrainedZonotopeBatchNormError::ResourceOverflow {
                        operation: "output generator nonzeros",
                    },
                )?;
                check_limit(
                    "output generator nonzeros",
                    output_generator_nonzeros,
                    limits.max_generator_nonzeros,
                )?;
                output_entries.push((value_index, nominal));
            }
        }
        output_generators.push(output_entries);
    }
    gate.checkpoint("BatchNorm generator transform complete")?;

    let constraints = clone_constraints(input, gate)?;
    gate.checkpoint("BatchNorm constraint clone complete")?;
    gate.checkpoint("BatchNorm right-hand-side allocation")?;
    let rhs = clone_slice_with_gate(input.rhs(), "BatchNorm constraint right-hand side", gate)?;
    gate.checkpoint("BatchNorm right-hand-side clone complete")?;
    gate.checkpoint("BatchNorm domain materialization")?;
    let output = ConstrainedZonotope64::try_new_with_call_gate(
        output_center,
        output_generators,
        constraints,
        rhs,
        output_remainder,
        gate,
    )
    .map_err(|error| match error {
        ConstrainedZonotope64CallGateError::Domain(error) => {
            ConstrainedZonotopeBatchNormBudgetError::Transform(
                ConstrainedZonotopeBatchNormError::Domain(error),
            )
        }
        ConstrainedZonotope64CallGateError::Budget(error) => {
            ConstrainedZonotopeBatchNormBudgetError::Budget(error)
        }
    })?;
    gate.checkpoint("BatchNorm domain materialization complete")?;
    let plan = ConstrainedZonotopeBatchNormPlan {
        input_rank: spec.input_shape.len(),
        channel_axis: spec.channel_axis,
        outer_count: geometry.outer_count,
        channel_count: geometry.channel_count,
        elements_per_channel: geometry.elements_per_channel,
        value_count: geometry.value_count,
        alpha_dim: input.alpha_dim(),
        constraint_count: input.constraint_count(),
        constraint_elements: geometry.constraint_elements,
        parameter_elements: geometry.parameter_elements,
        coordinate_visits: geometry.coordinate_visits,
        generator_visits: geometry.generator_visits,
        input_generator_nonzeros: geometry.input_generator_nonzeros,
        output_generator_nonzeros,
        interval_products,
        sqrt_refinements,
    };
    gate.checkpoint("BatchNorm publication")?;
    Ok((output, plan))
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    outer_count: usize,
    channel_count: usize,
    elements_per_channel: usize,
    value_count: usize,
    constraint_elements: usize,
    parameter_elements: usize,
    coordinate_visits: usize,
    generator_visits: usize,
    input_generator_nonzeros: usize,
}

impl Geometry {
    fn channel_for_flat_index(self, flat_index: usize) -> usize {
        debug_assert!(flat_index < self.value_count);
        (flat_index / self.elements_per_channel) % self.channel_count
    }
}

/// Conservative transform-owned peak.  This intentionally sums scratch from
/// disjoint phases.  It counts both the caller-form generator representation
/// and the private validated representation because `try_new` can transiently
/// overlap them.  The retained input and borrowed BatchNorm arrays belong in
/// the caller's baseline.
fn batch_norm_peak_live_bytes(
    input: &ConstrainedZonotope64,
    geometry: Geometry,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_bytes(
        BATCH_NORM_RATIONAL_SCRATCH_BYTES,
        "BatchNorm rational scratch bytes",
    )?;
    peak.add_elements::<CertifiedChannelAffine>(
        geometry.channel_count,
        "BatchNorm certified channel-affine bytes",
    )?;
    peak.add_elements::<f64>(geometry.value_count, "BatchNorm input-magnitude bytes")?;
    peak.add_elements::<f64>(geometry.value_count, "BatchNorm output-center bytes")?;
    peak.add_elements::<f64>(geometry.value_count, "BatchNorm output-remainder bytes")?;
    peak.add_elements::<Vec<(usize, f64)>>(
        input.alpha_dim(),
        "BatchNorm candidate generator-column bytes",
    )?;
    peak.add_elements::<(usize, f64)>(
        geometry.input_generator_nonzeros,
        "BatchNorm candidate generator-entry bytes",
    )?;
    peak.add_elements::<f64>(
        geometry.constraint_elements,
        "BatchNorm constraint-matrix bytes",
    )?;
    peak.add_elements::<f64>(
        input.constraint_count(),
        "BatchNorm constraint right-hand-side bytes",
    )?;
    peak.add_elements::<Vec<(usize, f64)>>(
        input.alpha_dim(),
        "BatchNorm validated generator-column bytes",
    )?;
    peak.add_elements::<(usize, f64)>(
        geometry.input_generator_nonzeros,
        "BatchNorm validated generator-entry bytes",
    )?;
    Ok(peak.finish())
}

fn batch_norm_affine_certificate_live_bytes(
    channel_count: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    let retained_rationals = channel_count.checked_mul(2).ok_or(
        ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "BatchNorm surrogate-certificate retained rational count",
        },
    )?;
    let rational_payload_bytes = retained_rationals
        .checked_mul(BATCH_NORM_RETAINED_RATIONAL_BYTES)
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "BatchNorm surrogate-certificate retained rational bytes",
        })?;
    let mut live = ConstrainedZonotopePeakLiveBytes::new();
    live.add_bytes(
        size_of::<ExactBatchNormAffineSurrogateCertificate>(),
        "BatchNorm surrogate-certificate header bytes",
    )?;
    live.add_elements::<ExactBatchNormChannelAffineCertificate>(
        channel_count,
        "BatchNorm surrogate-certificate channel bytes",
    )?;
    live.add_bytes(
        rational_payload_bytes,
        "BatchNorm surrogate-certificate rational payload bytes",
    )?;
    Ok(live.finish())
}

pub(crate) fn batch_norm_affine_certificate_peak_live_bytes(
    channel_count: usize,
) -> Result<usize, ConstrainedZonotopeCallBudgetError> {
    batch_norm_affine_certificate_live_bytes(channel_count)?
        .checked_add(BATCH_NORM_RATIONAL_SCRATCH_BYTES)
        .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
            operation: "BatchNorm surrogate-certificate peak bytes",
        })
}

fn validate_affine_certificate_spec_with_gate<G>(
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    nominal_scale: &[f64],
    nominal_bias: &[f64],
    limits: ConstrainedZonotopeBatchNormAffineCertificateLimits,
    gate: &mut G,
) -> Result<usize, ConstrainedZonotopeBatchNormBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    if spec.mode != ConstrainedZonotopeBatchNormMode::Inference {
        return Err(ConstrainedZonotopeBatchNormError::UnsupportedSemantics {
            semantics: "training mode requires batch-dependent statistics",
        }
        .into());
    }
    if spec.input_shape.is_empty() {
        return Err(ConstrainedZonotopeBatchNormError::InvalidSpec {
            message: "rank-zero input is unsupported".to_string(),
        }
        .into());
    }
    check_limit("input rank", spec.input_shape.len(), limits.max_rank)?;
    if spec.channel_axis >= spec.input_shape.len() {
        return Err(ConstrainedZonotopeBatchNormError::InvalidSpec {
            message: format!(
                "channel axis {} is outside rank {}",
                spec.channel_axis,
                spec.input_shape.len()
            ),
        }
        .into());
    }
    for (axis, &dimension) in spec.input_shape.iter().enumerate() {
        gate.charge_items(1, "BatchNorm surrogate-certificate shape validation")?;
        if dimension == 0 {
            return Err(ConstrainedZonotopeBatchNormError::InvalidSpec {
                message: format!("input shape axis {axis} has zero extent"),
            }
            .into());
        }
    }

    let channel_count = spec.input_shape[spec.channel_axis];
    check_limit("channel count", channel_count, limits.max_channel_count)?;
    for (field, values) in [
        ("gamma", spec.gamma),
        ("beta", spec.beta),
        ("mean", spec.mean),
        ("variance", spec.variance),
        ("nominal scale", nominal_scale),
        ("nominal bias", nominal_bias),
    ] {
        validate_parameter_shape(field, values, channel_count)?;
    }
    let parameter_elements = channel_count.checked_mul(6).ok_or(
        ConstrainedZonotopeBatchNormError::ResourceOverflow {
            operation: "BatchNorm surrogate-certificate parameter elements",
        },
    )?;
    check_limit(
        "parameter elements",
        parameter_elements,
        limits.max_parameter_elements,
    )?;

    validate_finite_with_gate("gamma", spec.gamma.iter().copied(), gate)?;
    validate_finite_with_gate("beta", spec.beta.iter().copied(), gate)?;
    validate_finite_with_gate("mean", spec.mean.iter().copied(), gate)?;
    validate_finite_with_gate("variance", spec.variance.iter().copied(), gate)?;
    validate_finite_with_gate("nominal scale", nominal_scale.iter().copied(), gate)?;
    validate_finite_with_gate("nominal bias", nominal_bias.iter().copied(), gate)?;
    if !spec.epsilon.is_finite() || spec.epsilon <= 0.0 {
        return Err(ConstrainedZonotopeBatchNormError::InvalidEpsilon.into());
    }
    for (index, &variance) in spec.variance.iter().enumerate() {
        gate.charge_items(1, "BatchNorm surrogate-certificate variance validation")?;
        if variance < 0.0 {
            return Err(ConstrainedZonotopeBatchNormError::InvalidVariance { index }.into());
        }
    }
    Ok(channel_count)
}

fn validate_geometry_with_gate<G>(
    input: &ConstrainedZonotope64,
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    limits: ConstrainedZonotopeBatchNormLimits,
    gate: &mut G,
) -> Result<Geometry, ConstrainedZonotopeBatchNormBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    if spec.mode != ConstrainedZonotopeBatchNormMode::Inference {
        return Err(ConstrainedZonotopeBatchNormError::UnsupportedSemantics {
            semantics: "training mode requires batch-dependent statistics",
        }
        .into());
    }
    if spec.input_shape.is_empty() {
        return Err(ConstrainedZonotopeBatchNormError::InvalidSpec {
            message: "rank-zero input is unsupported".to_string(),
        }
        .into());
    }
    check_limit("input rank", spec.input_shape.len(), limits.max_rank)?;
    if spec.channel_axis >= spec.input_shape.len() {
        return Err(ConstrainedZonotopeBatchNormError::InvalidSpec {
            message: format!(
                "channel axis {} is outside rank {}",
                spec.channel_axis,
                spec.input_shape.len()
            ),
        }
        .into());
    }
    for (axis, &dimension) in spec.input_shape.iter().enumerate() {
        gate.charge_items(1, "BatchNorm shape validation")?;
        if dimension == 0 {
            return Err(ConstrainedZonotopeBatchNormError::InvalidSpec {
                message: format!("input shape axis {axis} has zero extent"),
            }
            .into());
        }
    }

    let value_count = checked_product(spec.input_shape, "BatchNorm input value count")?;
    if input.value_dim() != value_count {
        return Err(ConstrainedZonotopeBatchNormError::Shape {
            field: "input domain",
            expected: vec![value_count],
            got: vec![input.value_dim()],
        }
        .into());
    }
    check_limit("value count", value_count, limits.max_value_count)?;
    check_limit("alpha dimension", input.alpha_dim(), limits.max_alpha_dim)?;
    check_limit(
        "constraint count",
        input.constraint_count(),
        limits.max_constraint_count,
    )?;

    let outer_count = checked_product(
        &spec.input_shape[..spec.channel_axis],
        "BatchNorm outer element count",
    )?;
    let channel_count = spec.input_shape[spec.channel_axis];
    let elements_per_channel = checked_product(
        &spec.input_shape[spec.channel_axis + 1..],
        "BatchNorm elements per channel",
    )?;
    check_limit("channel count", channel_count, limits.max_channel_count)?;

    validate_parameter_shape("gamma", spec.gamma, channel_count)?;
    validate_parameter_shape("beta", spec.beta, channel_count)?;
    validate_parameter_shape("mean", spec.mean, channel_count)?;
    validate_parameter_shape("variance", spec.variance, channel_count)?;
    let parameter_elements = channel_count.checked_mul(4).ok_or(
        ConstrainedZonotopeBatchNormError::ResourceOverflow {
            operation: "BatchNorm parameter elements",
        },
    )?;
    check_limit(
        "parameter elements",
        parameter_elements,
        limits.max_parameter_elements,
    )?;

    let constraint_elements = input
        .constraint_count()
        .checked_mul(input.alpha_dim())
        .ok_or(ConstrainedZonotopeBatchNormError::ResourceOverflow {
            operation: "constraint matrix elements",
        })?;
    check_limit(
        "constraint matrix elements",
        constraint_elements,
        limits.max_constraint_elements,
    )?;

    let mut input_generator_nonzeros = 0_usize;
    for generator in input.generators() {
        gate.charge_items(1, "BatchNorm generator geometry validation")?;
        input_generator_nonzeros = input_generator_nonzeros
            .checked_add(generator.nnz())
            .ok_or(ConstrainedZonotopeBatchNormError::ResourceOverflow {
                operation: "input generator nonzeros",
            })?;
    }
    check_limit(
        "input generator nonzeros",
        input_generator_nonzeros,
        limits.max_generator_nonzeros,
    )?;

    let coordinate_visits =
        value_count
            .checked_mul(2)
            .ok_or(ConstrainedZonotopeBatchNormError::ResourceOverflow {
                operation: "BatchNorm coordinate visits",
            })?;
    check_limit(
        "coordinate visits",
        coordinate_visits,
        limits.max_coordinate_visits,
    )?;
    let generator_visits = input_generator_nonzeros.checked_mul(2).ok_or(
        ConstrainedZonotopeBatchNormError::ResourceOverflow {
            operation: "BatchNorm generator visits",
        },
    )?;
    check_limit(
        "generator visits",
        generator_visits,
        limits.max_generator_visits,
    )?;

    validate_finite_with_gate("gamma", spec.gamma.iter().copied(), gate)?;
    validate_finite_with_gate("beta", spec.beta.iter().copied(), gate)?;
    validate_finite_with_gate("mean", spec.mean.iter().copied(), gate)?;
    validate_finite_with_gate("variance", spec.variance.iter().copied(), gate)?;
    if !spec.epsilon.is_finite() || spec.epsilon <= 0.0 {
        return Err(ConstrainedZonotopeBatchNormError::InvalidEpsilon.into());
    }
    for (index, &variance) in spec.variance.iter().enumerate() {
        gate.charge_items(1, "BatchNorm variance validation")?;
        if variance < 0.0 {
            return Err(ConstrainedZonotopeBatchNormError::InvalidVariance { index }.into());
        }
    }

    Ok(Geometry {
        outer_count,
        channel_count,
        elements_per_channel,
        value_count,
        constraint_elements,
        parameter_elements,
        coordinate_visits,
        generator_visits,
        input_generator_nonzeros,
    })
}

fn validate_parameter_shape(
    field: &'static str,
    values: &[f64],
    channel_count: usize,
) -> Result<(), ConstrainedZonotopeBatchNormError> {
    if values.len() != channel_count {
        return Err(ConstrainedZonotopeBatchNormError::Shape {
            field,
            expected: vec![channel_count],
            got: vec![values.len()],
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct CertifiedChannelAffine {
    scale: f64,
    bias: f64,
    scale_error: f64,
    bias_error: f64,
}

#[derive(Clone, Debug)]
struct ExactChannelAffineBounds {
    root_candidate: f64,
    scale_lower: BigRational,
    scale_upper: BigRational,
    bias_lower: BigRational,
    bias_upper: BigRational,
}

#[cfg(test)]
fn certify_channel_affines(
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
) -> Result<(Vec<CertifiedChannelAffine>, usize), ConstrainedZonotopeBatchNormError> {
    let mut gate = InertConstrainedZonotopeCallGate;
    match certify_channel_affines_with_gate(spec, &mut gate) {
        Ok(value) => Ok(value),
        Err(ConstrainedZonotopeBatchNormBudgetError::Transform(error)) => Err(error),
        Err(ConstrainedZonotopeBatchNormBudgetError::Budget(_)) => {
            unreachable!("the inert BatchNorm call gate cannot refuse work")
        }
    }
}

fn certify_channel_affines_with_gate<G>(
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    gate: &mut G,
) -> Result<(Vec<CertifiedChannelAffine>, usize), ConstrainedZonotopeBatchNormBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut affines = Vec::new();
    gate.checkpoint("BatchNorm certified-affine allocation")?;
    try_reserve(
        &mut affines,
        spec.gamma.len(),
        "certified BatchNorm channel affines",
    )?;
    let epsilon = exact_rational(spec.epsilon, "BatchNorm epsilon conversion")?;
    let mut total_refinements = 0_usize;
    for channel in 0..spec.gamma.len() {
        gate.charge_items(1, "BatchNorm coefficient certification")?;
        gate.checkpoint("BatchNorm rational coefficient allocation")?;
        let (bounds, refinements) = certify_exact_channel_affine_bounds(spec, channel, &epsilon)?;
        total_refinements = total_refinements.checked_add(refinements).ok_or(
            ConstrainedZonotopeBatchNormError::ResourceOverflow {
                operation: "BatchNorm square-root refinements",
            },
        )?;

        let scale = spec.gamma[channel] / bounds.root_candidate;
        if !scale.is_finite() {
            return Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic {
                operation: "BatchNorm nominal scale",
            }
            .into());
        }
        let scale_error = enclosing_error(
            scale,
            &bounds.scale_lower,
            &bounds.scale_upper,
            "BatchNorm scale error",
        )?;

        let bias = spec.beta[channel] - spec.mean[channel] * scale;
        if !bias.is_finite() {
            return Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic {
                operation: "BatchNorm nominal bias",
            }
            .into());
        }
        let bias_error = enclosing_error(
            bias,
            &bounds.bias_lower,
            &bounds.bias_upper,
            "BatchNorm bias error",
        )?;

        affines.push(CertifiedChannelAffine {
            scale,
            bias,
            scale_error,
            bias_error,
        });
    }
    Ok((affines, total_refinements))
}

fn certify_declared_channel_affines_with_gate<G>(
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    nominal_scale: &[f64],
    nominal_bias: &[f64],
    gate: &mut G,
) -> Result<
    (Vec<ExactBatchNormChannelAffineCertificate>, usize),
    ConstrainedZonotopeBatchNormBudgetError,
>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut certificates = Vec::new();
    gate.checkpoint("BatchNorm surrogate-certificate channel allocation")?;
    try_reserve(
        &mut certificates,
        spec.gamma.len(),
        "BatchNorm surrogate-certificate channels",
    )?;
    let epsilon = exact_rational(spec.epsilon, "BatchNorm epsilon conversion")?;
    let mut total_refinements = 0_usize;
    for channel in 0..spec.gamma.len() {
        gate.charge_items(1, "BatchNorm surrogate coefficient certification")?;
        gate.checkpoint("BatchNorm surrogate rational coefficient allocation")?;
        let (bounds, refinements) = certify_exact_channel_affine_bounds(spec, channel, &epsilon)?;
        total_refinements = total_refinements.checked_add(refinements).ok_or(
            ConstrainedZonotopeBatchNormError::ResourceOverflow {
                operation: "BatchNorm surrogate square-root refinements",
            },
        )?;
        let scale_error = exact_enclosing_error(
            nominal_scale[channel],
            &bounds.scale_lower,
            &bounds.scale_upper,
            "BatchNorm declared scale error",
        )?;
        let bias_error = exact_enclosing_error(
            nominal_bias[channel],
            &bounds.bias_lower,
            &bounds.bias_upper,
            "BatchNorm declared bias error",
        )?;
        certificates.push(ExactBatchNormChannelAffineCertificate {
            scale_error,
            bias_error,
        });
    }
    Ok((certificates, total_refinements))
}

fn certify_exact_channel_affine_bounds(
    spec: ConstrainedZonotopeBatchNormSpec<'_>,
    channel: usize,
    epsilon: &BigRational,
) -> Result<(ExactChannelAffineBounds, usize), ConstrainedZonotopeBatchNormError> {
    let variance = exact_rational(spec.variance[channel], "BatchNorm variance conversion")?;
    let denominator_squared = variance + epsilon;
    let rounded_sum = spec.variance[channel] + spec.epsilon;
    if !rounded_sum.is_finite() || rounded_sum <= 0.0 {
        return Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic {
            operation: "variance plus epsilon",
        });
    }
    let root_candidate = rounded_sum.sqrt();
    if !root_candidate.is_finite() || root_candidate <= 0.0 {
        return Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic {
            operation: "BatchNorm square root candidate",
        });
    }
    let (root_lower, root_upper, refinements) =
        bracket_positive_sqrt(&denominator_squared, root_candidate)?;
    let gamma = exact_rational(spec.gamma[channel], "BatchNorm gamma conversion")?;
    let root_lower = exact_rational(root_lower, "BatchNorm root lower conversion")?;
    let root_upper = exact_rational(root_upper, "BatchNorm root upper conversion")?;
    let (scale_lower, scale_upper) = if spec.gamma[channel] >= 0.0 {
        (&gamma / &root_upper, &gamma / &root_lower)
    } else {
        (&gamma / &root_lower, &gamma / &root_upper)
    };
    let beta = exact_rational(spec.beta[channel], "BatchNorm beta conversion")?;
    let mean = exact_rational(spec.mean[channel], "BatchNorm mean conversion")?;
    let (bias_lower, bias_upper) = if spec.mean[channel] >= 0.0 {
        (&beta - &mean * &scale_upper, &beta - &mean * &scale_lower)
    } else {
        (&beta - &mean * &scale_lower, &beta - &mean * &scale_upper)
    };
    Ok((
        ExactChannelAffineBounds {
            root_candidate,
            scale_lower,
            scale_upper,
            bias_lower,
            bias_upper,
        },
        refinements,
    ))
}

fn bracket_positive_sqrt(
    squared: &BigRational,
    candidate: f64,
) -> Result<(f64, f64, usize), ConstrainedZonotopeBatchNormError> {
    let candidate_rational =
        exact_rational(candidate, "BatchNorm square-root candidate conversion")?;
    match (&candidate_rational * &candidate_rational).cmp(squared) {
        Ordering::Equal => Ok((candidate, candidate, 0)),
        Ordering::Less => {
            let lower = candidate;
            let mut upper = candidate;
            for refinements in 1..=MAX_SQRT_REFINEMENTS_PER_CHANNEL {
                upper = upper.next_up();
                if !upper.is_finite() {
                    break;
                }
                let upper_rational =
                    exact_rational(upper, "BatchNorm square-root upper conversion")?;
                if &upper_rational * &upper_rational >= *squared {
                    return Ok((lower, upper, refinements));
                }
            }
            Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic {
                operation: "certified BatchNorm square-root upper bracket",
            })
        }
        Ordering::Greater => {
            let upper = candidate;
            let mut lower = candidate;
            for refinements in 1..=MAX_SQRT_REFINEMENTS_PER_CHANNEL {
                lower = lower.next_down();
                if !lower.is_finite() || lower <= 0.0 {
                    break;
                }
                let lower_rational =
                    exact_rational(lower, "BatchNorm square-root lower conversion")?;
                if &lower_rational * &lower_rational <= *squared {
                    return Ok((lower, upper, refinements));
                }
            }
            Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic {
                operation: "certified BatchNorm square-root lower bracket",
            })
        }
    }
}

fn enclosing_error(
    nominal: f64,
    lower: &BigRational,
    upper: &BigRational,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeBatchNormError> {
    ceil_nonnegative_rational_to_f64(
        exact_enclosing_error(nominal, lower, upper, operation)?,
        operation,
    )
}

fn exact_enclosing_error(
    nominal: f64,
    lower: &BigRational,
    upper: &BigRational,
    operation: &'static str,
) -> Result<BigRational, ConstrainedZonotopeBatchNormError> {
    let nominal = exact_rational(nominal, operation)?;
    let lower_error = (&nominal - lower).abs();
    let upper_error = (upper - &nominal).abs();
    Ok(lower_error.max(upper_error))
}

fn ceil_nonnegative_rational_to_f64(
    value: BigRational,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeBatchNormError> {
    if value.is_zero() {
        return Ok(0.0);
    }
    let mut candidate = value
        .to_f64()
        .ok_or(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic { operation })?;
    if !candidate.is_finite() || candidate < 0.0 {
        return Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic { operation });
    }
    let candidate_rational = exact_rational(candidate, operation)?;
    if candidate_rational < value {
        candidate = candidate.next_up();
        if !candidate.is_finite() || exact_rational(candidate, operation)? < value {
            return Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic { operation });
        }
    }
    Ok(candidate)
}

fn exact_rational(
    value: f64,
    operation: &'static str,
) -> Result<BigRational, ConstrainedZonotopeBatchNormError> {
    BigRational::from_float(value)
        .ok_or(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic { operation })
}

fn clone_constraints<G>(
    input: &ConstrainedZonotope64,
    gate: &mut G,
) -> Result<Array2<f64>, ConstrainedZonotopeBatchNormBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let shape = (input.constraint_count(), input.alpha_dim());
    let element_count = shape.0.checked_mul(shape.1).ok_or(
        ConstrainedZonotopeBatchNormError::ResourceOverflow {
            operation: "constraint matrix elements",
        },
    )?;
    let constraints = input.constraints();
    let mut values = Vec::new();
    gate.checkpoint("BatchNorm constraint-matrix allocation")?;
    try_reserve(&mut values, element_count, "BatchNorm constraint matrix")?;
    for row in 0..shape.0 {
        gate.charge_items(1, "BatchNorm constraint-matrix clone")?;
        for column in 0..shape.1 {
            gate.charge_items(1, "BatchNorm constraint-matrix clone")?;
            values.push(constraints[[row, column]]);
        }
    }
    Array2::from_shape_vec(shape, values).map_err(|_| {
        ConstrainedZonotopeBatchNormError::ResourceOverflow {
            operation: "constraint matrix shape",
        }
        .into()
    })
}

fn clone_slice_with_gate<T: Copy, G>(
    source: &[T],
    resource: &'static str,
    gate: &mut G,
) -> Result<Vec<T>, ConstrainedZonotopeBatchNormBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    let mut output = Vec::new();
    try_reserve(&mut output, source.len(), resource)?;
    for &value in source {
        gate.charge_items(1, "BatchNorm right-hand-side clone")?;
        output.push(value);
    }
    Ok(output)
}

fn validate_finite_with_gate<G>(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
    gate: &mut G,
) -> Result<(), ConstrainedZonotopeBatchNormBudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    for (index, value) in values.into_iter().enumerate() {
        gate.charge_items(1, "BatchNorm finite-parameter validation")?;
        if !value.is_finite() {
            return Err(ConstrainedZonotopeBatchNormError::NonFinite { field, index }.into());
        }
    }
    Ok(())
}

fn checked_product(
    dimensions: &[usize],
    operation: &'static str,
) -> Result<usize, ConstrainedZonotopeBatchNormError> {
    dimensions.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(ConstrainedZonotopeBatchNormError::ResourceOverflow { operation })
    })
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeBatchNormError> {
    if required > limit {
        return Err(ConstrainedZonotopeBatchNormError::ResourceLimit {
            resource,
            required,
            limit,
        });
    }
    Ok(())
}

fn consume_product(
    count: &mut usize,
    limit: usize,
) -> Result<(), ConstrainedZonotopeBatchNormError> {
    *count = count
        .checked_add(1)
        .ok_or(ConstrainedZonotopeBatchNormError::ResourceOverflow {
            operation: "interval product count",
        })?;
    check_limit("interval products", *count, limit)
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), ConstrainedZonotopeBatchNormError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ConstrainedZonotopeBatchNormError::AllocationFailure { resource })
}

/// Reject FTZ/DAZ before adjacent-float intervals are used as proof objects.
fn require_gradual_underflow() -> Result<(), ConstrainedZonotopeBatchNormError> {
    let half = std::hint::black_box(0.5_f64);
    let min_normal = std::hint::black_box(f64::MIN_POSITIVE);
    let min_subnormal = std::hint::black_box(f64::from_bits(1));
    let two_subnormals = std::hint::black_box(f64::from_bits(2));
    if std::hint::black_box(min_normal * half).to_bits() != 0x0008_0000_0000_0000
        || std::hint::black_box(two_subnormals * half).to_bits() != 1
        || std::hint::black_box(min_subnormal + min_subnormal).to_bits() != 2
    {
        return Err(
            ConstrainedZonotopeBatchNormError::UnsupportedFloatingPoint {
                requirement: "IEEE-754 binary64 gradual underflow (FTZ/DAZ disabled)",
            },
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct OutwardInterval {
    lo: f64,
    hi: f64,
}

impl OutwardInterval {
    const fn exact(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    fn is_exact_zero(self) -> bool {
        self.lo == 0.0 && self.hi == 0.0
    }

    fn add(self, rhs: Self) -> Result<Self, ConstrainedZonotopeBatchNormError> {
        if self.is_exact_zero() {
            return Ok(rhs);
        }
        if rhs.is_exact_zero() {
            return Ok(self);
        }
        Ok(Self {
            lo: round_down(self.lo + rhs.lo, "BatchNorm interval sum")?,
            hi: round_up(self.hi + rhs.hi, "BatchNorm interval sum")?,
        })
    }

    const fn abs(self) -> Self {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            Self {
                lo: -self.hi,
                hi: -self.lo,
            }
        } else {
            Self {
                lo: 0.0,
                hi: if -self.lo > self.hi {
                    -self.lo
                } else {
                    self.hi
                },
            }
        }
    }

    fn nominal_and_error(self) -> Result<(f64, f64), ConstrainedZonotopeBatchNormError> {
        if self.lo == self.hi {
            return Ok((self.lo, 0.0));
        }
        let midpoint = self.lo * 0.5 + self.hi * 0.5;
        let nominal = if midpoint.is_finite() && midpoint >= self.lo && midpoint <= self.hi {
            midpoint
        } else {
            self.lo
        };
        let lower_error = upper_difference(nominal, self.lo)?;
        let upper_error = upper_difference(self.hi, nominal)?;
        Ok((nominal, lower_error.max(upper_error)))
    }
}

fn outward_product(
    left: f64,
    right: f64,
) -> Result<OutwardInterval, ConstrainedZonotopeBatchNormError> {
    if left == 0.0 || right == 0.0 {
        return Ok(OutwardInterval::exact(0.0));
    }
    if left == 1.0 {
        return Ok(OutwardInterval::exact(right));
    }
    if right == 1.0 {
        return Ok(OutwardInterval::exact(left));
    }
    if left == -1.0 {
        return Ok(OutwardInterval::exact(-right));
    }
    if right == -1.0 {
        return Ok(OutwardInterval::exact(-left));
    }
    let product = left * right;
    Ok(OutwardInterval {
        lo: round_down(product, "BatchNorm interval product")?,
        hi: round_up(product, "BatchNorm interval product")?,
    })
}

fn upper_difference(upper: f64, lower: f64) -> Result<f64, ConstrainedZonotopeBatchNormError> {
    if upper == lower {
        return Ok(0.0);
    }
    round_up(upper - lower, "BatchNorm representation error")
}

fn add_nonnegative_upper(left: f64, right: f64) -> Result<f64, ConstrainedZonotopeBatchNormError> {
    if left == 0.0 {
        return Ok(right);
    }
    if right == 0.0 {
        return Ok(left);
    }
    round_up(left + right, "BatchNorm box remainder sum")
}

fn round_down(
    value: f64,
    operation: &'static str,
) -> Result<f64, ConstrainedZonotopeBatchNormError> {
    if !value.is_finite() {
        return Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic { operation });
    }
    let outward = value.next_down();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

fn round_up(value: f64, operation: &'static str) -> Result<f64, ConstrainedZonotopeBatchNormError> {
    if !value.is_finite() {
        return Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic { operation });
    }
    let outward = value.next_up();
    if !outward.is_finite() {
        return Err(ConstrainedZonotopeBatchNormError::NonFiniteArithmetic { operation });
    }
    Ok(outward)
}

#[cfg(test)]
#[path = "constrained_zonotope_batch_norm_tests.rs"]
mod tests;
