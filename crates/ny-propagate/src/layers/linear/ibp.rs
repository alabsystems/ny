// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP (Interval Bound Propagation) for linear layers.
//!
//! For x in [l, u] and y = Wx + b:
//! - W+ = max(W, 0), W- = min(W, 0)
//! - lower_y = W+ @ l + W- @ u + b
//! - upper_y = W+ @ u + W- @ l + b

use std::borrow::Cow;
use std::time::Instant;

use faer::Mat;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_dim_product, GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, L2Constraint, RepairStrategy};
use tracing::debug;

use super::bias::{
    add_f64_down, add_f64_up, binary32_min_normal_as_f64, f32_to_f64_exact,
    f64_to_f32_down_no_subnormal, f64_to_f32_up_no_subnormal, publish_error_up_normal,
};
use super::LinearLayer;
use crate::faer_parallelism::mat_mul;
use crate::layers::common::BoundPropagation;

const DEADLINE_LINEAR_IBP_POLL_OPS: usize = 4_096;
const DEADLINE_LINEAR_IBP_MAX_ELEMENTS: usize = 4 * 1024 * 1024;
const DEADLINE_LINEAR_IBP_MAX_RANK: usize = 1_024;

impl LinearLayer {
    /// IBP propagation with optional GEMM-engine acceleration.
    ///
    /// When `engine` is provided, both 1-D and N-D inputs are flattened to a
    /// `[batch, in_features]` matrix and evaluated via `GemmEngine::gemm_f32`.
    /// This mirrors the existing CROWN engine path and is intended for
    /// performance-sensitive callers such as PGD.
    ///
    /// The engine path may differ slightly from `propagate_ibp()` on 1-D inputs
    /// because it accumulates in f32 instead of the scalar path's f64 dot
    /// products. `propagate_ibp_sound()` remains the soundness-first API.
    pub fn propagate_ibp_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let Some(engine) = engine else {
            return self.propagate_ibp(input);
        };
        if engine.forbids_unbounded_cpu_fallback() {
            return Err(NyError::UnsupportedOp(
                "bounded Linear IBP requires the explicit deadline-aware entry".into(),
            ));
        }

        match propagate_ibp_via_gemm(self, input, engine) {
            // The GEMM path produces the same box interval as the CPU path, so
            // apply the same sound Cauchy–Schwarz intersection when the input
            // carries an L2 constraint (only tightens; no-op otherwise).
            Ok(bounds) => Ok(intersect_with_l2_cauchy_schwarz(self, input, bounds)),
            Err(err) => {
                debug!("GEMM engine failed for Linear IBP, falling back to CPU: {err}");
                self.propagate_ibp(input)
            }
        }
    }

    /// Deadline-authoritative Linear interval forward.
    ///
    /// `deadline: None` delegates to [`Self::propagate_ibp_with_engine`]
    /// exactly. Under a finite authority, the opaque engine helper is refused:
    /// its surrounding host copies, lazy cache construction, bookkeeping, and
    /// allocations do not have a cooperative deadline contract. Finite calls
    /// use a capped direct contraction that polls between bounded work quanta.
    pub fn propagate_ibp_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let Some(deadline) = deadline else {
            return self.propagate_ibp_with_engine(input, engine);
        };
        check_linear_ibp_deadline(deadline, "before entry")?;

        let geometry = linear_deadline_geometry(self, input)?;
        cap_linear_deadline_elements(geometry.input_elements, "input")?;
        cap_linear_deadline_elements(geometry.output_elements, "output")?;

        // No existing capability covers the host-side copies, lazy transpose
        // cache construction, multi-call bookkeeping, and result allocations
        // around the opaque engine calls. A finite authority therefore refuses
        // the entire engine helper, including engines whose inner GEMM is
        // deadline-safe.
        propagate_ibp_pollable_f64(self, input, &geometry, deadline)
    }

    /// Sound IBP propagation: [`BoundPropagation::propagate_ibp`] plus a further
    /// `in_features + 2` ULP margin.
    ///
    /// # The ULP charge is a MARGIN, not the certificate (#linear-ibp-higham)
    ///
    /// This docstring used to justify the charge as: "Reference: Higham, Theorem
    /// 3.1. For a sum of n non-negative terms, the accumulated rounding error is at
    /// most (n-1) ULPs. Each matmul produces a sum of `in_features` non-negative
    /// terms (since w_pos >= 0 and x_lower >= any)." That is FALSE, and stating it
    /// is what let the defect survive review:
    ///
    /// - The terms of `W⁺·l` are `W⁺_oj · l_j`, and `l_j` has ARBITRARY sign, so
    ///   `W⁺ ≥ 0` does NOT make them non-negative. The parenthetical
    ///   "(since w_pos >= 0 and x_lower >= any)" asserts nothing.
    /// - Higham Thm 3.1 bounds the error by `γ_n · Σ_j |t_j|` — the MAGNITUDE sum —
    ///   NOT by `γ_n · |Σ_j t_j|`. An "n ULPs of the result" charge is the second
    ///   quantity. They coincide only when nothing cancels.
    /// - The two matmuls are then COMBINED, so error accrued at each matmul's own
    ///   magnitude is being measured against the post-cancellation result.
    ///   `shift_down_n_ulps` scales with that result: at `0.0` the whole charge
    ///   degenerates to `n` subnormals (~1e-45).
    ///
    /// The enclosure is now established by the abssum-Higham certificate inside
    /// [`BoundPropagation::propagate_ibp`] itself — `γ^f32_{n+2}·S_safe + 2u·|y|`
    /// with `S = |W| @ max(|l|,|u|)` for the f32 (N-D / GEMM) paths, and its f64
    /// twin for the rank-1 path. This method's extra ULP margin is retained on top:
    /// it costs nothing that was not already being paid, and every caller that
    /// reaches here is a verdict path that prefers width to risk.
    pub fn propagate_ibp_sound(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut result = self.propagate_ibp(input)?;
        // in_features MACs + 1 for combining two matmul results + 1 for bias
        let rounding_ulps = u32::try_from(self.in_features())
            .unwrap_or(u32::MAX)
            .saturating_add(2);
        result.round_for_soundness_n_ulps_inplace(rounding_ulps);
        Ok(result)
    }
}

struct LinearDeadlineGeometry {
    batch_size: usize,
    in_features: usize,
    out_features: usize,
    input_elements: usize,
    output_elements: usize,
    output_shape: Vec<usize>,
}

#[inline]
fn check_linear_ibp_deadline(deadline: Instant, stage: &str) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(format!(
            "Linear IBP forward: deadline exceeded {stage}"
        )));
    }
    Ok(())
}

fn linear_deadline_geometry(
    layer: &LinearLayer,
    input: &BoundedTensor,
) -> Result<LinearDeadlineGeometry> {
    let shape = input.shape();
    let ndim = shape.len();
    if ndim == 0 {
        return Err(NyError::InvalidSpec(
            "Linear IBP: rank-0 (scalar) input not supported".to_string(),
        ));
    }
    if ndim > DEADLINE_LINEAR_IBP_MAX_RANK {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: ndim.saturating_mul(size_of::<usize>()),
            budget_bytes: DEADLINE_LINEAR_IBP_MAX_RANK * size_of::<usize>(),
            site: "Linear finite-deadline IBP shape metadata",
        });
    }
    let in_features = shape[ndim - 1];
    if in_features != layer.in_features() {
        return Err(NyError::ShapeMismatch {
            expected: vec![layer.in_features()],
            got: vec![in_features],
        });
    }
    let batch_size = checked_batch_size(&shape[..ndim - 1])?;
    let out_features = layer.out_features();
    let input_elements = batch_size.checked_mul(in_features).ok_or_else(|| {
        NyError::InvalidSpec(
            "Linear finite-deadline IBP input element count overflows usize".to_string(),
        )
    })?;
    let output_elements = batch_size.checked_mul(out_features).ok_or_else(|| {
        NyError::InvalidSpec(
            "Linear finite-deadline IBP output element count overflows usize".to_string(),
        )
    })?;
    let mut output_shape = shape[..ndim - 1].to_vec();
    output_shape.push(out_features);
    Ok(LinearDeadlineGeometry {
        batch_size,
        in_features,
        out_features,
        input_elements,
        output_elements,
        output_shape,
    })
}

fn cap_linear_deadline_elements(elements: usize, kind: &str) -> Result<()> {
    if elements > DEADLINE_LINEAR_IBP_MAX_ELEMENTS {
        let required_bytes = elements.saturating_mul(size_of::<f32>());
        let budget_bytes = DEADLINE_LINEAR_IBP_MAX_ELEMENTS * size_of::<f32>();
        let site = if kind == "input" {
            "Linear finite-deadline IBP input buffer"
        } else {
            "Linear finite-deadline IBP output buffer"
        };
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site,
        });
    }
    Ok(())
}

#[inline]
fn is_binary32_subnormal(value: f32) -> bool {
    let magnitude = value.to_bits() & 0x7fff_ffff;
    magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits()
}

fn linear_deadline_contains_subnormal<I>(values: I, deadline: Instant, stage: &str) -> Result<bool>
where
    I: IntoIterator<Item = f32>,
{
    for (index, value) in values.into_iter().enumerate() {
        if index.is_multiple_of(DEADLINE_LINEAR_IBP_POLL_OPS) {
            check_linear_ibp_deadline(deadline, stage)?;
        }
        if is_binary32_subnormal(value) {
            return Ok(true);
        }
    }
    check_linear_ibp_deadline(deadline, stage)?;
    Ok(false)
}

fn reserve_linear_deadline_vec<T>(len: usize, deadline: Instant, name: &str) -> Result<Vec<T>> {
    check_linear_ibp_deadline(deadline, "before bounded CPU allocation")?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NyError::InvalidSpec(format!(
            "Linear finite-deadline IBP {name} allocation failed for {len} elements: {error}"
        ))
    })?;
    check_linear_ibp_deadline(deadline, "after bounded CPU allocation")?;
    Ok(values)
}

#[inline]
fn linear_f64_to_f32_down(value: f64) -> f32 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == f64::INFINITY {
        return f32::MAX;
    }
    if value.abs() < f64::from(f32::MIN_POSITIVE) {
        return if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        };
    }
    let narrowed = value as f32;
    if narrowed == f32::INFINITY {
        // A FINITE f64 above `f32::MAX` narrows to `+inf`, and `next_down_f32`
        // returns infinities UNCHANGED (they are its infeasibility sentinel), so the
        // bare cast published `+inf` as a LOWER bound on a finite quantity — the one
        // direction that can exclude the true value. `f32::MAX < value`, so it is the
        // largest sound f32 lower bound available. Matches the `value == INFINITY`
        // arm above; this is that arm's finite twin.
        return f32::MAX;
    }
    next_down_f32(narrowed)
}

#[inline]
fn linear_f64_to_f32_up(value: f64) -> f32 {
    if value.is_nan() || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == f64::NEG_INFINITY {
        return f32::MIN;
    }
    if value.abs() < f64::from(f32::MIN_POSITIVE) {
        return if value.is_sign_negative() {
            0.0
        } else {
            f32::MIN_POSITIVE
        };
    }
    let narrowed = value as f32;
    if narrowed == f32::NEG_INFINITY {
        // Mirror of the `linear_f64_to_f32_down` guard: a finite f64 below `f32::MIN`
        // narrows to `-inf`, which `next_up_f32` passes through, publishing `-inf` as
        // an UPPER bound on a finite quantity. `f32::MIN > value`, so it is the
        // smallest sound f32 upper bound available.
        return f32::MIN;
    }
    next_up_f32(narrowed)
}

fn linear_deadline_universal(
    geometry: &LinearDeadlineGeometry,
    deadline: Instant,
) -> Result<BoundedTensor> {
    let mut lower =
        reserve_linear_deadline_vec(geometry.output_elements, deadline, "universal lower output")?;
    let mut upper =
        reserve_linear_deadline_vec(geometry.output_elements, deadline, "universal upper output")?;
    while lower.len() < geometry.output_elements {
        let chunk = (geometry.output_elements - lower.len()).min(DEADLINE_LINEAR_IBP_POLL_OPS);
        lower.extend(std::iter::repeat_n(f32::NEG_INFINITY, chunk));
        upper.extend(std::iter::repeat_n(f32::INFINITY, chunk));
        check_linear_ibp_deadline(deadline, "while initializing universal output")?;
    }
    let lower = ArrayD::from_shape_vec(IxDyn(&geometry.output_shape), lower)
        .map_err(|error| NyError::InternalError(format!("Linear IBP lower reshape: {error}")))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&geometry.output_shape), upper)
        .map_err(|error| NyError::InternalError(format!("Linear IBP upper reshape: {error}")))?;
    let result =
        BoundedTensor::new_repaired_with_poll(lower, upper, RepairStrategy::Conservative, || {
            check_linear_ibp_deadline(deadline, "during universal output repair")
        })?;
    check_linear_ibp_deadline(deadline, "immediately before publishing universal output")?;
    Ok(result)
}

fn propagate_ibp_pollable_f64(
    layer: &LinearLayer,
    input: &BoundedTensor,
    geometry: &LinearDeadlineGeometry,
    deadline: Instant,
) -> Result<BoundedTensor> {
    check_linear_ibp_deadline(deadline, "before pollable CPU contraction")?;

    // A DAZ-enabled host can erase a subnormal source operand before the f64
    // conversion. Fail open rather than relying on that conversion.
    let weight_has_subnormal = linear_deadline_contains_subnormal(
        layer.weight.iter().copied(),
        deadline,
        "while scanning weights for subnormal operands",
    )?;
    let bias_has_subnormal = if let Some(bias) = &layer.bias {
        linear_deadline_contains_subnormal(
            bias.iter().copied(),
            deadline,
            "while scanning bias for subnormal operands",
        )?
    } else {
        false
    };
    let lower_has_subnormal = linear_deadline_contains_subnormal(
        input.lower().iter().copied(),
        deadline,
        "while scanning lower input for subnormal operands",
    )?;
    let upper_has_subnormal = linear_deadline_contains_subnormal(
        input.upper().iter().copied(),
        deadline,
        "while scanning upper input for subnormal operands",
    )?;
    if weight_has_subnormal || bias_has_subnormal || lower_has_subnormal || upper_has_subnormal {
        debug!(
            "Linear finite-deadline IBP: subnormal source operand; \
             returning universal bounds for DAZ independence"
        );
        return linear_deadline_universal(geometry, deadline);
    }

    let mut lower_input =
        reserve_linear_deadline_vec(geometry.input_elements, deadline, "lower input")?;
    let mut upper_input =
        reserve_linear_deadline_vec(geometry.input_elements, deadline, "upper input")?;
    for (index, (&lower, &upper)) in input.lower().iter().zip(input.upper().iter()).enumerate() {
        if index.is_multiple_of(DEADLINE_LINEAR_IBP_POLL_OPS) {
            check_linear_ibp_deadline(deadline, "while flattening input bounds")?;
        }
        lower_input.push(lower);
        upper_input.push(upper);
    }
    if lower_input.len() != geometry.input_elements || upper_input.len() != geometry.input_elements
    {
        return Err(NyError::InternalError(format!(
            "Linear finite-deadline IBP flattened {} elements, expected {}",
            lower_input.len(),
            geometry.input_elements
        )));
    }

    let mut lower_output =
        reserve_linear_deadline_vec(geometry.output_elements, deadline, "lower output")?;
    let mut upper_output =
        reserve_linear_deadline_vec(geometry.output_elements, deadline, "upper output")?;
    let mut operations = 0usize;
    let mut output_cells = 0usize;
    for batch_index in 0..geometry.batch_size {
        let input_base = batch_index * geometry.in_features;
        for output_index in 0..geometry.out_features {
            output_cells += 1;
            if output_cells.is_multiple_of(DEADLINE_LINEAR_IBP_POLL_OPS) {
                check_linear_ibp_deadline(deadline, "while traversing output cells")?;
            }
            let bias = layer
                .bias
                .as_ref()
                .map_or(0.0_f64, |values| f64::from(values[output_index]));
            let mut lower_sum = bias;
            let mut upper_sum = bias;
            for input_index in 0..geometry.in_features {
                operations += 1;
                if operations == DEADLINE_LINEAR_IBP_POLL_OPS {
                    check_linear_ibp_deadline(deadline, "during directed CPU contraction")?;
                    operations = 0;
                }
                let weight = f64::from(layer.weight[[output_index, input_index]]);
                let flat_index = input_base + input_index;
                let (lower_factor, upper_factor) = if weight >= 0.0 {
                    (
                        f64::from(lower_input[flat_index]),
                        f64::from(upper_input[flat_index]),
                    )
                } else {
                    (
                        f64::from(upper_input[flat_index]),
                        f64::from(lower_input[flat_index]),
                    )
                };
                lower_sum = ny_core::dd::next_down_f64(lower_sum + weight * lower_factor);
                upper_sum = ny_core::dd::next_up_f64(upper_sum + weight * upper_factor);
            }
            lower_output.push(linear_f64_to_f32_down(lower_sum));
            upper_output.push(linear_f64_to_f32_up(upper_sum));
        }
    }
    check_linear_ibp_deadline(deadline, "after directed CPU contraction")?;
    drop(lower_input);
    drop(upper_input);

    let lower = ArrayD::from_shape_vec(IxDyn(&geometry.output_shape), lower_output)
        .map_err(|error| NyError::InternalError(format!("Linear IBP lower reshape: {error}")))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&geometry.output_shape), upper_output)
        .map_err(|error| NyError::InternalError(format!("Linear IBP upper reshape: {error}")))?;
    let result =
        BoundedTensor::new_repaired_with_poll(lower, upper, RepairStrategy::Conservative, || {
            check_linear_ibp_deadline(deadline, "during result repair")
        })?;
    check_linear_ibp_deadline(deadline, "immediately before publishing result")?;
    Ok(result)
}

impl BoundPropagation for LinearLayer {
    /// IBP for linear layer: y = Wx + b
    ///
    /// Supports N-D batched inputs where the last dimension must match in_features().
    /// For input shape [...batch_dims..., in_features], output is [...batch_dims..., out_features].
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let ndim = input.shape().len();
        if ndim == 0 {
            return Err(NyError::InvalidSpec(
                "Linear IBP: rank-0 (scalar) input not supported".to_string(),
            ));
        }
        let box_result = if ndim == 1 {
            propagate_ibp_1d(self, input)?
        } else {
            propagate_ibp_nd(self, input)?
        };
        // THE LEVER: if the input carries a per-slice L2 (Euclidean-ball)
        // constraint from an upstream normalization, intersect the decorrelated
        // box interval with the EXACT Cauchy–Schwarz row bound. Intersection
        // only tightens; a non-applicable / malformed constraint is ignored.
        Ok(intersect_with_l2_cauchy_schwarz(self, input, box_result))
    }

    /// CROWN backward propagation through linear layer.
    ///
    /// Delegates to the crown_single module.
    #[inline]
    fn propagate_linear<'a>(
        &self,
        bounds: &'a crate::LinearBounds,
    ) -> Result<Cow<'a, crate::LinearBounds>> {
        super::crown_single::propagate_linear_cpu(self, bounds)
    }
}

/// f32 unit roundoff `u = 2^-24`.
const LINEAR_IBP_F32_UNIT_ROUNDOFF: f64 = 1.0 / (1u64 << 24) as f64;

/// f64 unit roundoff `u = 2^-53`.
const LINEAR_IBP_F64_UNIT_ROUNDOFF: f64 = 1.0 / (1u64 << 53) as f64;

/// `xmax_j = max(|l_j|, |u_j|)`: the per-coordinate magnitude envelope of the input
/// box, and the operand of every abssum certificate below. `f32::abs` and `f32::max`
/// round nothing, so this array is EXACT.
///
/// A NaN endpoint is absorbed by `f32::max` (which returns the non-NaN side), which
/// cannot under-cover: an output whose contraction touches a NaN coordinate is itself
/// NaN, and both the widening below and `BoundedTensor::new_repaired` open such a row
/// to `[-inf, +inf]`.
fn linear_ibp_xmax_2d(lower: &Array2<f32>, upper: &Array2<f32>) -> Array2<f32> {
    let mut xmax = lower.mapv(f32::abs);
    ndarray::Zip::from(&mut xmax)
        .and(upper)
        .for_each(|m, &u| *m = m.max(u.abs()));
    xmax
}

/// `S_o = Σ_j |W_oj| · xmax_j`, the operand of the certificate in
/// [`widen_linear_ibp_higham`], contracted as `Sᵀ = |W| · xmaxᵀ` in ONE matmul against
/// the layer's cached `|W|` (#linear-abs-weight-cache). Shared by ALL THREE certified
/// paths — the N-D faer path, the engine interval path, and the engine point path.
///
/// # Cost
///
/// The previous form was `S = (xmax·W⁺ᵀ) − (xmax·W⁻ᵀ)`: TWO contractions of
/// `batch·in·out` MACs, because `W⁺ = max(W,0)` and `W⁻ = min(W,0)` have DISJOINT
/// supports and therefore reconstruct `|W|` exactly. `LinearLayer` now caches `|W|`
/// itself as an owned `(out, in)` faer `Mat`, so the SAME quantity is ONE
/// contraction.
///
/// The orientation is picked so that no transpose is materialized. The cached `|W|`
/// is `(out, in)` — `weight_faer`'s own shape, not the `(in, out)` shape the bound
/// matmuls' `w_pos_t_faer`/`w_neg_t_faer` operands have — so transposing IT would
/// cost `in·out` per call, which is the whole matmul it saves at `batch = 1`.
/// Contracting it on the LEFT instead moves the transpose onto `xmax`:
/// `Mat::from_fn(in, batch, |j, i| xmax[[i, j]])` visits exactly the `batch·in`
/// elements the old `Mat::from_fn(batch, in, ..)` pack visited, so the pack cost is
/// unchanged and only the matmul count falls, 2 → 1.
///
/// The engine paths call this too, so their certificate no longer issues ANY
/// `GemmEngine::gemm_f32` dispatch: the point path is back to ONE engine GEMM and the
/// interval path to FOUR, exactly the counts the batching pins were written
/// against. SOUNDNESS IS UNAFFECTED by moving the contraction off the engine:
/// [`widen_linear_ibp_higham`] requires only that `S_f32` be a round-to-nearest f32
/// accumulation of the non-negative terms `|W_oj|·xmax_j` in ANY summation order, and
/// faer's blocked SIMD kernel is exactly that — the same contract compliance
/// `FaerCpuGemmEngine` states for its own `gemm_f32`. (An FMA-fused kernel drops a
/// rounding per term rather than adding one, so it stays inside the same
/// `γ_n` envelope in both directions.) It is NOT the same wall-clock cost on every
/// engine: against a GPU or a scalar-loop engine this trades a device dispatch for a
/// CPU matmul.
///
/// # Direction: this may not publish a narrower box than the two-matmul form
///
/// The two forms round differently, and a bound may only ever move OUTWARD, so the
/// one-matmul result is inflated by [`linear_ibp_abs_sum_split_bump`] before it is
/// returned. Write `S` for the exact `Σ_j |W_oj|·xmax_j`, `n = in_features`,
/// `u = 2^-24`.
///
/// - Higham Thm 3.1 on a sum of NON-NEGATIVE terms gives
///   `fl(xmax·W⁺ᵀ) ≤ S⁺·(1+γ_n)` and `|fl(xmax·W⁻ᵀ)| ≤ S⁻·(1+γ_n)`, with
///   `S⁺ + S⁻ = S` by the disjoint supports. Their f32 difference is an ADDITION of
///   two non-negative magnitudes (no cancellation), so it rounds by at most `(1+u)`
///   and `S_old ≤ S·(1+γ_n)(1+u)`.
/// - The new value is a single round-to-nearest f32 accumulation of the same
///   non-negative terms, so `S_new ≥ S·(1−γ_n)`.
/// - Hence `S_old ≤ S_new · (1+γ_n)(1+u)/(1−γ_n)`, which is the bump. When
///   `S_new = 0` the chain gives `S = 0` hence `S_old = 0`, and the `next_up` in
///   [`linear_ibp_inflate_abs_sum`] returns a positive subnormal, so the inequality
///   still holds.
///
/// Every downstream step — [`widen_linear_ibp_higham`], then `next_down_f32` /
/// `next_up_f32`, then the `n+2` ULP charge — is MONOTONE in `S`, so `S_bumped ≥
/// S_old` implies the published box CONTAINS the box the two-matmul form published.
///
/// The bump is `≈ 1 + 2γ_n`: `1.0000935` at `in_features = 784`, i.e. it widens the
/// certificate term by under one part in ten thousand. It is NOT a soundness margin — the
/// `1/(1−γ)` inflation inside [`widen_linear_ibp_higham`] already covers `S_new`'s
/// own accumulation deficit — it exists solely so that a COST change cannot move any
/// endpoint inward.
fn linear_ibp_abs_sum_faer(layer: &LinearLayer, xmax: &Array2<f32>) -> Array2<f32> {
    let (batch_size, in_features) = xmax.dim();
    let out_features = layer.out_features();
    let xmax_t = Mat::<f32>::from_fn(in_features, batch_size, |j, i| xmax[[i, j]]);
    let s_t = mat_mul(layer.weight_abs_faer(), &xmax_t);
    let bump = linear_ibp_abs_sum_split_bump(in_features);
    Array2::<f32>::from_shape_fn((batch_size, out_features), |(i, j)| {
        linear_ibp_inflate_abs_sum(s_t[(j, i)], bump)
    })
}

/// `(1+γ_n)(1+u)/(1−γ_n)`: the factor that makes the one-matmul abssum provably at or
/// above the two-matmul abssum it replaced. Derivation and the monotonicity argument
/// are on [`linear_ibp_abs_sum_faer`].
///
/// Saturates to `+inf` once `γ_n ≥ 1` (`n ≥ 2^24`), which
/// [`widen_linear_ibp_higham`] already reads as a non-finite certificate and answers
/// with `[-inf, +inf]`. Computed ONCE per call, not per output element.
fn linear_ibp_abs_sum_split_bump(in_features: usize) -> f64 {
    let gamma = crate::layers::linear::crown_single_gamma_n_f32(in_features);
    if gamma < 1.0 {
        (1.0 + gamma) * (1.0 + LINEAR_IBP_F32_UNIT_ROUNDOFF) / (1.0 - gamma)
    } else {
        f64::INFINITY
    }
}

/// Apply [`linear_ibp_abs_sum_split_bump`] to one abssum entry, rounding OUTWARD at
/// every step so the returned f32 is `≥ s · bump` exactly.
///
/// `f32→f64` is exact. The f64 multiply rounds to nearest, so `next_up_f64` (≥ ½ ULP,
/// the round-to-nearest deficit) restores it; the f64→f32 narrowing rounds to
/// nearest, so `next_up_f32` restores that. Two directed steps, no fudge factor.
///
/// FAILS OUTWARD on every degenerate input: an f32 overflow narrows to `+inf`, an
/// infinite `bump` against a zero `s` produces `NaN`, and a `NaN` `xmax` propagates —
/// [`widen_linear_ibp_higham`] answers all three with `[-inf, +inf]`.
#[inline]
fn linear_ibp_inflate_abs_sum(s: f32, bump: f64) -> f32 {
    next_up_f32(ny_core::dd::next_up_f64(f64::from(s) * bump) as f32)
}

/// THE CERTIFICATE for a round-to-nearest f32 Linear IBP contraction
/// (#linear-ibp-higham). Widens `[lower, upper]` OUTWARD, in place.
///
/// # Why the n-ULP charge it supplements is not a bound
///
/// Both f32 paths (`propagate_ibp_nd`, `propagate_ibp_via_gemm`) accumulate
/// `W⁺·l + W⁻·u` in round-to-nearest f32 and then charge
/// `round_for_soundness_n_ulps_inplace(in_features + 2)` — `n + 2` ULPs OF THE
/// RESULT. Higham Thm 3.1 bounds the error of an `n`-term f32 contraction by
/// `γ_n · Σ_j |t_j|`, the MAGNITUDE sum, NOT by `γ_n · |Σ_j t_j|`. The two coincide
/// only without cancellation, and the terms here are `l_j · W⁺_oj` with `l_j` of
/// ARBITRARY sign, so `W⁺ ≥ 0` does not make them same-sign. `shift_down_n_ulps`
/// scales with the result, so as the result cancels toward zero the charge collapses
/// with it — at `0.0` it degenerates to `n` subnormals (~1e-45). Hand-verified:
/// `W = [2^24, 1, -2^24]`, no bias, `x = [1, 1, 1]` gives
/// `fl(2^24 + 1) = 2^24` (ties-to-even), `fl(y) = 0`, a `±7e-45` charge, and a true
/// `y = 1` that lies OUTSIDE the published box. A ReLU reading `u ≤ 0` then pins a
/// live neuron to `0` — a false constraint, i.e. a false `unsat`.
///
/// # The bound that IS charged
///
/// `err_o = up( γ^f32_{n+2} · S_safe_o + 2u·|y_o| + U_abs )` with
/// `S_o = Σ_j |W_oj| · max(|l_j|, |u_j|)`,
/// `S_safe = (S_f32 + U_abs) / (1 − γ^f32_{n+2})`, and the binary32 FTZ floor
/// `U_abs = n·2^-126 / (1 − n·u)`, `n = 6·in_features + 2`. This mirrors
/// `convolution::crown_helpers::higham_widen_ibp`, which is the canonical form
/// (`conv2d::bound` now delegates to it rather than hand-rolling this).
///
/// WHY `U_abs` IS NOT OPTIONAL. `γ·S` is a purely RELATIVE bound, so it provably
/// cannot see a flush-to-zero event: if every product in a row underflows then
/// `S_f32` is itself ~0 and `γ·S` charges ~0, while the true discarded mass is up
/// to `n·2^-126`. It is charged TWICE — once inside `S_safe`, once in the published
/// error — because the abssum pass can flush independently of the value pass. `n`
/// deliberately over-counts (6 ops per MAC) so the term does not depend on the gemm
/// backend's summation order.
///
/// SOUNDNESS. Write `A = Σ_j W⁺_oj·l_j`, `B = Σ_j W⁻_oj·u_j`, `y = A + B + b`.
/// `W⁺`/`W⁻` have disjoint supports, so the two contractions' term magnitudes SUM to
/// at most `S_o`, and Higham gives `|fl(A) − A| + |fl(B) − B| ≤ γ_n · S_o`. The
/// combine rounds by at most `u·|fl(A)+fl(B)| ≤ u·S_o·(1 + γ_n)` and the bias add by
/// at most `u·|y_o|`. Since `γ_{n+2} − γ_n ≥ 2u ≥ u·(1 + γ_n)` whenever `γ_n ≤ 1`
/// (and `γ_n > 1` saturates the whole certificate to `+inf`), `γ_{n+2}·S_o` covers
/// the contractions AND the combine, and `2u·|y_o|` covers the bias add twice over.
/// `S_f32` is itself a round-to-nearest f32 accumulation of non-negative terms, so it
/// can fall short of the true abssum by up to `γ_n·S`; dividing by `(1 − γ_{n+2})`
/// restores `S_safe ≥ S_true`. `err` is published by `publish_error_up_normal`, which
/// lands on the NORMAL binary32 grid (never a subnormal that the very FTZ event
/// `U_abs` accounts for could itself flush), and is applied through the directed f64
/// adds `add_f64_down` / `add_f64_up` before a `no_subnormal` cast — idiom (b),
/// strictly stronger than the f32 subtract plus one directed step this replaced.
/// So the published box strictly encloses the true `y_o`.
///
/// NOT COVERED — DAZ. A subnormal WEIGHT or input endpoint erased before
/// multiplication contributes an error the other (normal) operand can amplify to
/// O(1), which no after-the-fact widening can repair. `ConvTranspose2d` and (as of
/// the same change) `Conv2d` refuse to bounds in that case. Linear does NOT yet
/// screen for it; see the audit note. Trained-network weights are essentially never
/// subnormal, so this is a latent gap rather than a live one.
///
/// DIRECTION. `err ≥ 0` always (γ ≥ 0, `S ≥ 0`, `|y| ≥ 0`), so this only ever moves
/// endpoints OUTWARD. A non-finite or NaN certificate opens the row to
/// `[-inf, +inf]` rather than publishing a tight lie.
///
/// COST. `S` is a MAGNITUDE-scaled floor that input splitting cannot erode
/// (#cgan-conv-ibp-magnitude-floor). Where a deadline exists, prefer
/// [`propagate_ibp_pollable_f64`], whose directed-f64 contraction is strictly
/// tighter than any f32-plus-widening.
fn widen_linear_ibp_higham(
    lower: &mut ArrayD<f32>,
    upper: &mut ArrayD<f32>,
    abs_sum: &Array2<f32>,
    in_features: usize,
) -> Result<()> {
    if lower.len() != abs_sum.len() || upper.len() != abs_sum.len() {
        return Err(NyError::InternalError(format!(
            "Linear IBP Higham widening: {} lower / {} upper endpoints against {} abssum entries",
            lower.len(),
            upper.len(),
            abs_sum.len()
        )));
    }
    let gamma = crate::layers::linear::crown_single_gamma_n_f32(in_features.saturating_add(2));
    let s_inflate = if gamma < 1.0 {
        1.0 / (1.0 - gamma)
    } else {
        f64::INFINITY
    };
    // FTZ ABSOLUTE FLOOR. `gamma * S` is a purely RELATIVE bound and therefore
    // provably cannot see a flush-to-zero event: when every product in a row
    // underflows, `S` itself is ~0 and `gamma * S` charges ~0, while the true
    // discarded mass is up to n * 2^-126. Charge that absolutely, both inside
    // `S_safe` and again in the published error. Over-count the op budget the
    // same way the conv sibling does (`crown_helpers.rs:401`) so the term does
    // not depend on the gemm backend's summation order.
    let underflow_n = in_features.saturating_mul(6).saturating_add(2) as f64;
    let underflow_nu = underflow_n * LINEAR_IBP_F32_UNIT_ROUNDOFF;
    let underflow_abs = if underflow_nu < 1.0 {
        (underflow_n * binary32_min_normal_as_f64()) / (1.0 - underflow_nu)
    } else {
        f64::INFINITY
    };
    for ((lo, up), &s) in lower.iter_mut().zip(upper.iter_mut()).zip(abs_sum.iter()) {
        let lo64 = f32_to_f64_exact(*lo);
        let up64 = f32_to_f64_exact(*up);
        let mag = lo64.abs().max(up64.abs());
        let s_observed = f32_to_f64_exact(s);
        // A non-finite or negative abssum is untrusted metadata, not a small
        // number: poison it rather than letting `gamma * s_safe` come out small.
        let s_safe = if s_observed.is_finite() && s_observed >= 0.0 {
            (s_observed + underflow_abs) * s_inflate
        } else {
            f64::INFINITY
        };
        let err_estimate =
            gamma * s_safe + 2.0 * LINEAR_IBP_F32_UNIT_ROUNDOFF * mag + underflow_abs;
        // Publishes on the NORMAL binary32 grid, and maps NaN/negative/inf to
        // +inf, which the `is_finite` test below then opens to the universal row.
        let err = publish_error_up_normal(err_estimate);
        if err.is_finite() {
            let err64 = f32_to_f64_exact(err);
            *lo = f64_to_f32_down_no_subnormal(add_f64_down(lo64, -err64));
            *up = f64_to_f32_up_no_subnormal(add_f64_up(up64, err64));
        } else {
            *lo = f32::NEG_INFINITY;
            *up = f32::INFINITY;
        }
    }
    Ok(())
}

/// IBP for 1D (unbatched) input.
fn propagate_ibp_1d(layer: &LinearLayer, input: &BoundedTensor) -> Result<BoundedTensor> {
    let shape = input.shape();
    let in_len = shape[0];
    if in_len != layer.in_features() {
        return Err(NyError::ShapeMismatch {
            expected: vec![layer.in_features()],
            got: vec![in_len],
        });
    }

    let x_lower = input
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![in_len],
            got: input.lower().shape().to_vec(),
        })?;

    let x_upper = input
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![in_len],
            got: input.upper().shape().to_vec(),
        })?;

    // f64 accumulation for IBP dot products to prevent precision loss for large
    // in_features. Each dot product accumulates in_features multiply-add terms;
    // f32 sum loses ~log2(n) bits of precision.
    // Reference: Higham, "Accuracy and Stability of Numerical Algorithms", Theorem 3.1.
    let w_pos_f64 = layer.w_pos().mapv(|x| x as f64);
    let w_neg_f64 = layer.w_neg().mapv(|x| x as f64);
    let xl_f64 = x_lower.mapv(|x| x as f64);
    let xu_f64 = x_upper.mapv(|x| x as f64);

    let lower_y_f64 = w_pos_f64.dot(&xl_f64) + w_neg_f64.dot(&xu_f64);
    let upper_y_f64 = w_pos_f64.dot(&xu_f64) + w_neg_f64.dot(&xl_f64);

    let (lower_y_f64, upper_y_f64) = if let Some(ref b) = layer.bias {
        let b_f64 = b.mapv(|x| x as f64);
        (lower_y_f64 + &b_f64, upper_y_f64 + b_f64)
    } else {
        (lower_y_f64, upper_y_f64)
    };

    // SOUNDNESS (#linear-ibp-higham), the f64 twin of the certificate the two f32
    // paths carry. The directed f64→f32 cast below certifies the CAST ONLY — it does
    // NOT cover the f64 accumulation residual, which Higham Thm 3.1 bounds by
    // `γ^f64_n · Σ_j |W_oj|·|x_j|`, an S-scaled quantity that does NOT shrink with the
    // result. One ULP of an f32 result is `~2^-24·|y|`, so the old widening
    // under-covers as soon as `S/|y| ≳ 2^24/(n·2^-29)`; concretely, a 50-wide acasxu
    // row with `S ≈ 50` and a near-zero pre-activation `|y| ≈ 1e-6` has a residual of
    // `~2.8e-13` against `ulp(1e-6) ≈ 1.1e-13` — 2.5x short, in exactly the
    // straddling-zero regime where a ReLU stability decision is made. Charge the
    // residual explicitly. The cost is invisible in f32: for that row the term is
    // 2.8e-13 on a bound of order 1.
    //
    // Charge is `up(γ^f64_{n+2}·S_safe + 2u₆₄·|y_o|)` with
    // `S_safe = S/(1 - γ^f64_{n+2})`, the same derivation as
    // [`widen_linear_ibp_higham`]: `W⁺`/`W⁻` have disjoint supports so the two dots'
    // term magnitudes sum to at most `S_o`; `γ_{n+2} - γ_n ≥ 2u` covers the combine;
    // `2u·|y_o|` covers the bias add twice over. The `(n+2)/n` slack in `γ` exceeds
    // the ~4 f64 roundings spent computing `err` itself by ~13 orders of magnitude,
    // and `next_up_f64` is applied on top of that.
    let xmax_f64 = {
        let mut magnitudes = xl_f64.mapv(f64::abs);
        ndarray::Zip::from(&mut magnitudes)
            .and(&xu_f64)
            .for_each(|m, &u| *m = m.max(u.abs()));
        magnitudes
    };
    // |W| = W⁺ - W⁻ exactly (disjoint supports), and both dots are non-negative, so
    // `s_f64` carries no cancellation of its own.
    let s_f64 = w_pos_f64.dot(&xmax_f64) - w_neg_f64.dot(&xmax_f64);
    let gamma = crate::layers::linear::crown_single_gamma_n_f64(in_len.saturating_add(2));
    let s_inflate = if gamma < 1.0 {
        1.0 / (1.0 - gamma)
    } else {
        f64::INFINITY
    };

    // Directed rounding on the f64→f32 cast: lower → next_down, upper → next_up, via
    // the guarded casts (which, unlike a bare `next_down_f32(x as f32)`, do not
    // publish the ±inf infeasibility SENTINEL as a bound on a finite quantity).
    let mut lower_y = Array1::<f32>::zeros(lower_y_f64.len());
    let mut upper_y = Array1::<f32>::zeros(upper_y_f64.len());
    ndarray::Zip::from(&mut lower_y)
        .and(&mut upper_y)
        .and(&lower_y_f64)
        .and(&upper_y_f64)
        .and(&s_f64)
        .for_each(|lo_out, up_out, &lo, &up, &s| {
            let s_safe = s.abs() * s_inflate;
            let err_lo = ny_core::dd::next_up_f64(
                gamma * s_safe + 2.0 * LINEAR_IBP_F64_UNIT_ROUNDOFF * lo.abs(),
            );
            let err_up = ny_core::dd::next_up_f64(
                gamma * s_safe + 2.0 * LINEAR_IBP_F64_UNIT_ROUNDOFF * up.abs(),
            );
            *lo_out = linear_f64_to_f32_down(ny_core::dd::next_down_f64(lo - err_lo));
            *up_out = linear_f64_to_f32_up(ny_core::dd::next_up_f64(up + err_up));
        });

    // Centralized NaN/Inf repair at constructor.
    BoundedTensor::new_repaired(
        lower_y.into_dyn(),
        upper_y.into_dyn(),
        RepairStrategy::Conservative,
    )
}

/// Compute batch_size from batch dimensions with overflow check.
fn checked_batch_size(batch_dims: &[usize]) -> Result<usize> {
    checked_dim_product(batch_dims, "Linear IBP batch dimensions")
}

fn contiguous_2d_slice(array: &Array2<f32>) -> Cow<'_, [f32]> {
    match array.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(array.iter().copied().collect()),
    }
}

fn reshape_gemm_output(
    output: Vec<f32>,
    batch_size: usize,
    out_features: usize,
) -> Result<Array2<f32>> {
    let got_len = output.len();
    Array2::from_shape_vec((batch_size, out_features), output).map_err(|_| NyError::ShapeMismatch {
        expected: vec![batch_size, out_features],
        got: vec![got_len],
    })
}

fn broadcast_bias_2d(
    bias: &Array1<f32>,
    batch_size: usize,
    out_features: usize,
) -> Result<Array2<f32>> {
    bias.broadcast((batch_size, out_features))
        .ok_or_else(|| NyError::ShapeMismatch {
            expected: vec![batch_size, out_features],
            got: bias.shape().to_vec(),
        })
        .map(|view| view.to_owned())
}

fn propagate_concrete_via_gemm(
    layer: &LinearLayer,
    input_2d: &Array2<f32>,
    batch_size: usize,
    out_shape: &[usize],
    engine: &dyn GemmEngine,
) -> Result<BoundedTensor> {
    let output = reshape_gemm_output(
        engine.gemm_f32(
            batch_size,
            layer.in_features(),
            layer.out_features(),
            contiguous_2d_slice(input_2d).as_ref(),
            // #cora-transpose-cache: cached once on the layer — the per-call
            // rebuild was 62% of ALL samples on a cora PGD profile.
            layer.weight_t_row_major(),
        )?,
        batch_size,
        layer.out_features(),
    )?;

    let output = if let Some(ref bias) = layer.bias {
        output + broadcast_bias_2d(bias, batch_size, layer.out_features())?
    } else {
        output
    };

    BoundedTensor::concrete(reshape_2d_to_nd(output, out_shape)?)
}

/// Certified Higham widening for the POINT-input engine GEMM (#degenerate-ibp-gemm).
///
/// `propagate_concrete_via_gemm` evaluates `y = W*x + b` with ONE round-to-nearest
/// f32 GEMM over `K = in_features` products plus one f32 bias add. A degenerate
/// (`lower == upper`) input box does NOT make that evaluation exact: Higham Thm 3.1
/// bounds the deviation by `gamma^f32_{K+1} * S_o + u*|y_o|` with
/// `S_o = sum_j |W_oj| * |x_j|` — the MAGNITUDE sum, not `|y_o|`, so the charge does
/// not shrink when the row cancels. Publishing `[fl(y_o), fl(y_o)]` can therefore
/// exclude `y_o` entirely (`W = [2^24, 1, -2^24]`, `x = [1, 1, 1]`: true `y = 1`,
/// `fl(y) = 0`, published width 0).
///
/// A point is just a box with `l == u`, so this is the SAME certificate the widened
/// paths use, on the SAME operands: `xmax = max(|l|, |u|) = |x|` from
/// [`linear_ibp_xmax_2d`], the `|W|` abssum from [`linear_ibp_abs_sum_faer`], and the
/// `up(gamma^f32_{n+2} * S_safe + 2u*|y|)` charge from [`widen_linear_ibp_higham`] —
/// which carries the full soundness and direction argument. Sharing one
/// implementation is deliberate: two certificates for one contraction is how the
/// degenerate arm came to have none at all.
///
/// COST. The abssum takes NO `GemmEngine` dispatch — it contracts the layer's
/// cached `|W|` with one faer matmul (see [`linear_ibp_abs_sum_faer`]) — so the whole
/// point path still issues exactly ONE engine GEMM, the `y = W*x + b` forward itself,
/// independent of batch size. That is the property the batching pins assert, and it is
/// the one that matters on the PGD hot path, which re-enters this arm once per Linear
/// layer per concrete evaluation.
fn certify_point_gemm(
    layer: &LinearLayer,
    input_2d: &Array2<f32>,
    point: BoundedTensor,
) -> Result<BoundedTensor> {
    // A point box: `max(|l_j|, |u_j|)` collapses to `|x_j|`.
    let xmax = linear_ibp_xmax_2d(input_2d, input_2d);
    let abs_sum = linear_ibp_abs_sum_faer(layer, &xmax);
    let mut lower = point.lower().to_owned();
    let mut upper = point.upper().to_owned();
    widen_linear_ibp_higham(&mut lower, &mut upper, &abs_sum, layer.in_features())?;
    let mut result = BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)?;
    // Arm parity: the widened path keeps its historical `in_features + 2` ULP charge
    // ON TOP of the same certificate. The charge is non-negative and
    // `shift_down_n_ulps` / `shift_up_n_ulps` are monotone, so applying it here too
    // only moves endpoints OUTWARD — and it keeps a point input from receiving a
    // box that is TIGHTER than the one this layer publishes for a hair-width input
    // box around that same point.
    let rounding_ulps = u32::try_from(layer.in_features())
        .unwrap_or(u32::MAX)
        .saturating_add(2);
    result.round_for_soundness_n_ulps_inplace(rounding_ulps);
    Ok(result)
}

fn propagate_ibp_via_gemm(
    layer: &LinearLayer,
    input: &BoundedTensor,
    engine: &dyn GemmEngine,
) -> Result<BoundedTensor> {
    let shape = input.shape();
    let ndim = shape.len();
    if ndim == 0 {
        return Err(NyError::InvalidSpec(
            "Linear IBP: rank-0 (scalar) input not supported".to_string(),
        ));
    }

    let in_features = shape[ndim - 1];
    if in_features != layer.in_features() {
        return Err(NyError::ShapeMismatch {
            expected: vec![layer.in_features()],
            got: vec![in_features],
        });
    }

    let batch_size = checked_batch_size(&shape[..ndim - 1])?;
    let out_features = layer.out_features();

    let mut out_shape = shape[..ndim - 1].to_vec();
    out_shape.push(out_features);

    // Normalize to standard (row-major) layout before the reshape — a
    // non-leading-axis Concat yields a non-contiguous owned array that
    // `into_shape_with_order` would reject (mislabeled as ShapeMismatch).
    // `as_standard_layout` copies to C-order only when needed; sound.
    let x_lower_2d = input
        .lower()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch_size, in_features],
            got: input.lower().shape().to_vec(),
        })?;
    if input.lower() == input.upper() {
        // SOUNDNESS (#degenerate-ibp-gemm): a POINT input does NOT make the engine
        // GEMM exact, so its output may not be published as an exact point.
        // `propagate_concrete_via_gemm` contracts `K = in_features` products in
        // round-to-nearest f32 (the `GemmEngine::gemm_f32` precision contract) and
        // then adds the bias in f32; by Higham (Thm 3.1) the deviation is bounded by
        // `gamma^f32_{K+1} * S_o + u*|y_o|` with `S_o = sum_k |W_ok| * |x_k|`, an
        // S-scaled quantity that is UNBOUNDED in ULPs of `y_o` once the row cancels
        // (`W = [2^24, 1, -2^24]`, `x = [1, 1, 1]`: true `y = 1`, `fl(y) = 0`).
        // Returning `[fl(y), fl(y)]` therefore publishes a ZERO-WIDTH box that
        // EXCLUDES the true value — a false VERIFIED if it feeds a pre-activation or
        // a verdict row. `propagate_ibp_with_engine` is the SOUND box API and is
        // documented (see `Network::propagate_concrete_point`) to return a
        // non-trivially-wide box for a point input; the faithful point forward is
        // `propagate_concrete_point`, which collapses to the interval centre itself.
        let point =
            propagate_concrete_via_gemm(layer, &x_lower_2d, batch_size, &out_shape, engine)?;
        return certify_point_gemm(layer, &x_lower_2d, point);
    }
    let x_upper_2d = input
        .upper()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch_size, in_features],
            got: input.upper().shape().to_vec(),
        })?;

    let x_lower = contiguous_2d_slice(&x_lower_2d);
    let x_upper = contiguous_2d_slice(&x_upper_2d);
    // #cora-transpose-cache: cached once on the layer (see mod.rs).
    let w_pos_t = layer.w_pos_t_row_major();
    let w_neg_t = layer.w_neg_t_row_major();

    let lower_pos = reshape_gemm_output(
        engine.gemm_f32(
            batch_size,
            in_features,
            out_features,
            x_lower.as_ref(),
            w_pos_t,
        )?,
        batch_size,
        out_features,
    )?;
    let lower_neg = reshape_gemm_output(
        engine.gemm_f32(
            batch_size,
            in_features,
            out_features,
            x_upper.as_ref(),
            w_neg_t,
        )?,
        batch_size,
        out_features,
    )?;
    let upper_pos = reshape_gemm_output(
        engine.gemm_f32(
            batch_size,
            in_features,
            out_features,
            x_upper.as_ref(),
            w_pos_t,
        )?,
        batch_size,
        out_features,
    )?;
    let upper_neg = reshape_gemm_output(
        engine.gemm_f32(
            batch_size,
            in_features,
            out_features,
            x_lower.as_ref(),
            w_neg_t,
        )?,
        batch_size,
        out_features,
    )?;

    let (lower_y_2d, upper_y_2d) = if let Some(ref bias) = layer.bias {
        let bias_broadcast = broadcast_bias_2d(bias, batch_size, out_features)?;
        (
            lower_pos + lower_neg + &bias_broadcast,
            upper_pos + upper_neg + bias_broadcast,
        )
    } else {
        (lower_pos + lower_neg, upper_pos + upper_neg)
    };

    let mut out_lower = reshape_2d_to_nd(lower_y_2d, &out_shape)?;
    let mut out_upper = reshape_2d_to_nd(upper_y_2d, &out_shape)?;
    // SOUNDNESS (#linear-ibp-higham). The `in_features + 2` ULP charge below used to
    // be the ONLY widening here, described as "the same Higham widening as the N-D
    // faer path". It is the same charge, and it is under-strength for the same
    // reason: `n` ULPs OF THE RESULT is `γ_n·|Σ t_j|`, while Higham Thm 3.1 bounds
    // the error by `γ_n·Σ|t_j|` — the MAGNITUDE sum, which the four combined
    // round-to-nearest f32 GEMMs make unboundedly larger than the result under
    // cancellation. The `GemmEngine::gemm_f32` precision contract is plain IEEE
    // round-to-nearest f32 in ANY summation order, which is exactly the hypothesis of
    // the order-independent abssum certificate applied here.
    let xmax = linear_ibp_xmax_2d(&x_lower_2d, &x_upper_2d);
    // The abssum itself takes NO fifth/sixth engine dispatch: it contracts the layer's
    // cached `|W|` with ONE faer matmul (see `linear_ibp_abs_sum_faer`), which halves
    // the certificate's MACs against the `xmax·W⁺ᵀ − xmax·W⁻ᵀ` pair it replaced and
    // leaves this path at the FOUR engine GEMMs it has always issued.
    let abs_sum = linear_ibp_abs_sum_faer(layer, &xmax);
    widen_linear_ibp_higham(&mut out_lower, &mut out_upper, &abs_sum, in_features)?;

    let mut result =
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)?;
    // KEPT on top of the certificate above, not replaced by it: the ULP charge is
    // non-negative and `shift_down_n_ulps` / `shift_up_n_ulps` are monotone, so the
    // composition lands at or outside every endpoint this path published before.
    let rounding_ulps = u32::try_from(layer.in_features())
        .unwrap_or(u32::MAX)
        .saturating_add(2);
    result.round_for_soundness_n_ulps_inplace(rounding_ulps);
    Ok(result)
}

/// IBP for N-D batched input (last dimension is in_features).
fn propagate_ibp_nd(layer: &LinearLayer, input: &BoundedTensor) -> Result<BoundedTensor> {
    let shape = input.shape();
    let ndim = shape.len();
    let in_features = shape[ndim - 1];
    if in_features != layer.in_features() {
        return Err(NyError::ShapeMismatch {
            expected: vec![layer.in_features()],
            got: vec![in_features],
        });
    }

    // Output shape: [...batch_dims..., out_features]
    let mut out_shape: Vec<usize> = shape[..ndim - 1].to_vec();
    out_shape.push(layer.out_features());
    let batch_size = checked_batch_size(&shape[..ndim - 1])?;

    // Reshape input to [batch_size, in_features]. Normalize to standard
    // (row-major) layout first: a non-leading-axis Concat (e.g. channel concat)
    // produces an OWNED but non-contiguous (column-major) array, and
    // `into_shape_with_order` errors on a non-contiguous buffer even for an
    // identity reshape — which was mislabeled as a ShapeMismatch (expected ==
    // got). `as_standard_layout` copies to C-order only when needed; values and
    // bounds are unchanged, so this is sound.
    let x_lower_2d = input
        .lower()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch_size, in_features],
            got: input.lower().shape().to_vec(),
        })?;
    let x_upper_2d = input
        .upper()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch_size, in_features],
            got: input.upper().shape().to_vec(),
        })?;

    // Convert ndarray to faer matrices (copy into column-major format for optimal matmul)
    // Note: Attempted zero-copy MatRef::from_row_major_slice but it causes 3-5x regression
    // due to strided memory access in matmul. The copy cost is amortized by faster matmul.
    let x_lower_faer = Mat::<f32>::from_fn(batch_size, in_features, |i, j| x_lower_2d[[i, j]]);
    let x_upper_faer = Mat::<f32>::from_fn(batch_size, in_features, |i, j| x_upper_2d[[i, j]]);

    // IBP matmul: [batch, in] @ [in, out] = [batch, out]
    let lower_pos = mat_mul(&x_lower_faer, layer.w_pos_t_faer());
    let lower_neg = mat_mul(&x_upper_faer, layer.w_neg_t_faer());
    let upper_pos = mat_mul(&x_upper_faer, layer.w_pos_t_faer());
    let upper_neg = mat_mul(&x_lower_faer, layer.w_neg_t_faer());
    let lower_y_faer = &lower_pos + &lower_neg;
    let upper_y_faer = &upper_pos + &upper_neg;

    // Convert faer results back to ndarray
    let out_features = layer.out_features();
    let lower_y_2d =
        Array2::<f32>::from_shape_fn((batch_size, out_features), |(i, j)| lower_y_faer[(i, j)]);
    let upper_y_2d =
        Array2::<f32>::from_shape_fn((batch_size, out_features), |(i, j)| upper_y_faer[(i, j)]);

    // Add bias if present (broadcast across batch)
    let (lower_y_2d, upper_y_2d) = if let Some(ref b) = layer.bias {
        let b_broadcast = b
            .broadcast((batch_size, layer.out_features()))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: vec![batch_size, layer.out_features()],
                got: b.shape().to_vec(),
            })?
            .to_owned();
        (lower_y_2d + &b_broadcast, upper_y_2d + b_broadcast)
    } else {
        (lower_y_2d, upper_y_2d)
    };

    // Reshape back to original batch dimensions + out_features
    let mut out_lower = reshape_2d_to_nd(lower_y_2d, &out_shape)?;
    let mut out_upper = reshape_2d_to_nd(upper_y_2d, &out_shape)?;

    // SOUNDNESS (#linear-ibp-higham). This block used to carry ONLY the comment
    // "each output element is a sum of `in_features` same-sign terms per matmul ...
    // so by Higham Thm 3.1 the accumulated error is bounded by `in_features + 2`
    // ULPs of the result". BOTH halves of that were wrong. (1) The terms are
    // `x_lower_j · W⁺_oj` and `x_lower` has ARBITRARY sign, so `W⁺ ≥ 0` does not
    // make them same-sign — and the four matmuls are then COMBINED, so rounding lost
    // at each matmul's own magnitude is being measured against the POST-cancellation
    // result. (2) Higham Thm 3.1 bounds the error by `γ_n · Σ|t_j|`, the MAGNITUDE
    // sum, not by `γ_n · |Σ t_j|`; an n-ULP-of-the-result charge is the second
    // quantity. `W = [2^24, 1, -2^24]`, `x = [1, 1, 1]`: `fl(y) = 0`, the charge is
    // `shift_down_n_ulps(0.0, 5) ≈ 7e-45`, and the true `y = 1` is OUTSIDE the
    // published box — a false pre-activation that a ReLU turns into a false `unsat`.
    // Charge the certified abssum bound instead; see [`widen_linear_ibp_higham`].
    let xmax = linear_ibp_xmax_2d(&x_lower_2d, &x_upper_2d);
    let abs_sum = linear_ibp_abs_sum_faer(layer, &xmax);
    widen_linear_ibp_higham(&mut out_lower, &mut out_upper, &abs_sum, in_features)?;

    // Centralized NaN/Inf repair at constructor.
    let mut result =
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)?;
    // The historical `in_features + 2` ULP charge is KEPT, applied ON TOP of the
    // certificate above rather than replaced by it. It is not a bound on its own (see
    // the paragraph above), but it is non-negative, and `shift_down_n_ulps` /
    // `shift_up_n_ulps` are monotone in their argument — so composing it with an
    // already-widened box lands at or OUTSIDE every endpoint this path published
    // before. That makes the change outward-only at this site by construction, with
    // no case analysis over which of the two charges dominates.
    let rounding_ulps = u32::try_from(layer.in_features())
        .unwrap_or(u32::MAX)
        .saturating_add(2);
    result.round_for_soundness_n_ulps_inplace(rounding_ulps);
    Ok(result)
}

/// Reshape a 2D array back to N-D target shape.
fn reshape_2d_to_nd(arr: Array2<f32>, target_shape: &[usize]) -> Result<ArrayD<f32>> {
    let src_shape: Vec<usize> = arr.shape().to_vec();
    arr.into_shape_with_order(IxDyn(target_shape))
        .map_err(|_| NyError::ShapeMismatch {
            expected: target_shape.to_vec(),
            got: src_shape,
        })
}

/// Intersect the box IBP result with the exact Cauchy–Schwarz row bound implied
/// by an input L2 (Euclidean-ball) constraint. **Only ever tightens.**
///
/// For `y = W·x + b` and a per-slice constraint `‖x_slice − c_slice‖₂ ≤ r_slice`
/// (the ball axis being the contracted `in_features` axis), each output obeys
///   y[s,o] = W[o,:]·c[s,:] + b[o] + W[o,:]·(x[s,:] − c[s,:]),
///   |W[o,:]·(x[s,:] − c[s,:])| ≤ ‖W[o,:]‖₂ · r_slice    (Cauchy–Schwarz, EXACT),
/// so y[s,o] ∈ nominal[s,o] ± ‖W_o‖₂·r_slice. This interval and the box interval
/// are BOTH sound enclosures of the same true value, so their per-element
/// intersection (max of lowers, min of uppers) is sound and non-empty.
///
/// SOUNDNESS / DIRECTED ROUNDING. `nominal` and `‖W_o‖₂` accumulate in f64; the
/// radius term is rounded UP (`next_up_f32`) and the interval endpoints rounded
/// OUTWARD (`next_down_f32` low / `next_up_f32` high) before intersecting, so the
/// CS interval can never under-cover. The radius supplied by the producer is a
/// proven OUTWARD bound on the true distance. If the constraint is absent, its
/// ball axis is not the contracted axis, or any shape/feature check fails, the
/// box result is returned unchanged (sound: no tightening).
fn intersect_with_l2_cauchy_schwarz(
    layer: &LinearLayer,
    input: &BoundedTensor,
    box_result: BoundedTensor,
) -> BoundedTensor {
    // THE GATE: the lever is active ONLY in a top-level plain IBP pass. Inside
    // any iterative CROWN bound recomputation (alpha-/beta-CROWN) the gate is OFF
    // and this is byte-identical to the pre-lever box result. Skipping is sound
    // (the box already encloses the true value). See `crate::l2_lever_gate`.
    if !crate::l2_lever_gate::l2_lever_active() {
        return box_result;
    }
    let Some(constraint) = input.l2_constraint() else {
        return box_result;
    };
    match try_cauchy_schwarz_tighten(layer, input, constraint, &box_result) {
        Some(tightened) => tightened,
        None => box_result,
    }
}

/// Core Cauchy–Schwarz tightening; returns `None` to fall back to the box.
///
/// ## Cheap O(out + in) nominal — the midpoint identity
///
/// The exact CS bound is `y[s,o] ∈ (W_o·center[s] + b_o) ± ‖W_o‖₂·r[s]`. Computing
/// the nominal `W_o·center[s]` directly is O(out·in) per slice (the old hot loop
/// that hung deep-CROWN). We avoid it with an identity on the box-IBP result.
///
/// For the box IBP of `y = Wz + b` over the Linear-input box `z ∈ [zl, zu]`,
/// `W = W⁺ + W⁻` and (in real arithmetic)
///   box_lo[o] + box_hi[o]
///     = (W⁺·zl + W⁻·zu + b) + (W⁺·zu + W⁻·zl + b)
///     = (W⁺ + W⁻)·(zl + zu) + 2b = W·(zl + zu) + 2b,
/// so the box-result midpoint `mid_o := (box_lo[o]+box_hi[o])/2 = W_o·z_mid + b_o`
/// EXACTLY, where `z_mid = (zl+zu)/2` is the input-box midpoint — O(1) per output,
/// no inner loop. This is the nominal centred at `z_mid` instead of `center`.
///
/// ## Recentring margin `d` (sound for ANY centre — not just symmetric boxes)
///
/// `z_mid` need not equal the ball centre. Writing `y_o = W_o·z_mid + W_o·(z−z_mid)`
/// and using `‖z − z_mid‖₂ ≤ ‖z − center‖₂ + ‖center − z_mid‖₂ ≤ r + d` (triangle
/// inequality, with `d := ‖z_mid − center‖₂` computed ONCE per slice — O(in), not
/// O(out·in)), Cauchy–Schwarz gives
///   |W_o·(z − z_mid)| ≤ ‖W_o‖₂·(r + d),
/// hence `y_o ∈ mid_o ± ‖W_o‖₂·(r + d)`. EXACT in real arithmetic. For the
/// Standard-LayerNorm IBP box `d ≈ 0` (the centred-(x−mean) interval is symmetric
/// about `beta`), so this is byte-for-byte as tight as the old `W·center` nominal;
/// for RMSNorm (origin-centred, asymmetric box) `d > 0` recovers soundness at a
/// modest widening. Either way the result is ⊆ the box (intersection) and never
/// looser than before.
///
/// ## f32 rounding of the midpoint (extra outward margin `μ`)
///
/// `mid_o` is computed from the *directed-rounded* f32 box endpoints, which were
/// widened OUTWARD by `in_features + 2` ULPs (Higham) from the real box endpoints
/// `true_lo, true_hi` (whose midpoint is the real `W_o·z_mid + b_o`). Since
/// `box_lo ≤ true_lo` and `box_hi ≥ true_hi`,
///   |fl(mid_o) − (W_o·z_mid + b_o)|
///     ≤ ((box_hi − true_hi) + (true_lo − box_lo))/2 + ½·ulp(mid_o)
///     ≤ (in_features + 2.5)·ulp(max(|box_lo|,|box_hi|)),
/// so we fold a generous `μ_o = (in_features + 4)·ulp(max(|box_lo[o]|,|box_hi[o]|))`
/// into the per-output delta (rounded up). Combined with the radius/`d` term rounded
/// outward, the CS interval provably encloses the true `y_o`, so intersecting it
/// with the box only removes infeasible mass.
fn try_cauchy_schwarz_tighten(
    layer: &LinearLayer,
    input: &BoundedTensor,
    constraint: &L2Constraint,
    box_result: &BoundedTensor,
) -> Option<BoundedTensor> {
    let in_shape = input.shape();
    let ndim = in_shape.len();
    if ndim == 0 {
        return None;
    }
    // The ball must be taken over the axis the Linear contracts (last axis).
    if constraint.axis() != ndim - 1 {
        return None;
    }
    let in_features = in_shape[ndim - 1];
    if in_features != layer.in_features() {
        return None;
    }
    let out_features = layer.out_features();

    // The nominal is now O(out) via the box-midpoint identity (see doc above) plus
    // an O(in) per-slice recentring norm `d`, so there is no O(out·in) per-call cost
    // to guard against — the old `CS_MAX_WEIGHT_ELEMS` cap is gone. This makes the
    // lever affordable even inside CROWN's intermediate-bound IBP passes.

    // Flatten batch dims; the constraint center matches the input shape and its
    // radius is per-batch-slice (rank ndim-1).
    let batch_size = checked_batch_size(&in_shape[..ndim - 1]).ok()?;
    if constraint.center().shape() != in_shape {
        return None;
    }
    let radius = constraint.radius();
    let center_2d = constraint
        .center()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .ok()?;
    let radius_flat = radius
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(batch_size)
        .ok()?;

    // Per-output-row ‖W[o,:]‖₂, precomputed once on the layer (this was an O(out·in)
    // hot loop recomputed every IBP call — the table-transformer deep-CROWN hang).
    // Already rounded outward at construction; reject the whole tightening if any
    // row norm is non-finite.
    let w_row_l2 = layer.row_l2_norms();
    if w_row_l2.iter().any(|n| !n.is_finite()) {
        return None;
    }

    // The Linear INPUT box `z` (flattened) — its per-slice midpoint `z_mid` is the
    // implicit centre of the box-result midpoint identity, and the per-slice
    // distance `d = ‖z_mid − center‖₂` is the recentring margin (computed O(in)
    // per slice below, never O(out·in)).
    let (in_lower, in_upper) = input.lower_upper();
    let in_lower_2d = in_lower
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .ok()?;
    let in_upper_2d = in_upper
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, in_features))
        .ok()?;

    let (box_lower, box_upper) = box_result.lower_upper();
    // Operate on flat [batch_size, out_features] views of the box result.
    let mut out_lower = box_lower
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, out_features))
        .ok()?;
    let mut out_upper = box_upper
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, out_features))
        .ok()?;

    for s in 0..batch_size {
        let r = radius_flat[s];
        if !(r.is_finite() && r >= 0.0) {
            // Malformed slice radius: leave this slice's box bound untouched.
            continue;
        }

        // d = ‖z_mid − center‖₂ for THIS slice, in f64, rounded UP to f32 (it adds
        // to the radius → round outward). O(in) per slice — paid once, not per
        // output. z_mid = (in_lo + in_hi)/2 (the Linear-input box midpoint).
        let mut d_sumsq = 0.0_f64;
        let mut center_finite = true;
        for j in 0..in_features {
            let zl = in_lower_2d[[s, j]] as f64;
            let zu = in_upper_2d[[s, j]] as f64;
            let cj = center_2d[[s, j]] as f64;
            if !(zl.is_finite() && zu.is_finite() && cj.is_finite()) {
                center_finite = false;
                break;
            }
            // Bit-identical to `0.5 * (zl + zu)`: finite f32-cast operands stay on
            // f64::midpoint's non-overflow `(a + b) * 0.5` path.
            let z_mid = f64::midpoint(zl, zu);
            let diff = z_mid - cj;
            d_sumsq += diff * diff;
        }
        if !center_finite {
            // Non-finite input/center on this slice: leave its box bound untouched.
            continue;
        }
        let d = next_up_f32(d_sumsq.sqrt() as f32);
        // r_eff = r + d, rounded UP (the total CS radius from z_mid).
        let r_eff = next_up_f32(r + d);
        if !r_eff.is_finite() {
            continue;
        }

        for o in 0..out_features {
            let bl = out_lower[[s, o]];
            let bu = out_upper[[s, o]];
            if !(bl.is_finite() && bu.is_finite()) {
                continue;
            }
            // nominal `mid_o = (box_lo + box_hi)/2 = W_o·z_mid + b_o` EXACTLY in
            // real arithmetic (the box-midpoint identity). f32 add then ×0.5.
            // Kept verbatim: `bl + bu` overflowing to ±inf must reach the guard
            // below (f32::midpoint would fabricate a finite center instead).
            #[allow(clippy::manual_midpoint)]
            let mid = 0.5 * (bl + bu);
            if !mid.is_finite() {
                // bl + bu overflowed to ±inf: skip (leave box untouched — sound).
                continue;
            }

            // delta = ‖W_o‖₂·(r + d) + μ_o, rounded UP (outward). μ_o accounts for
            // the f32 rounding of `mid` vs the real `W_o·z_mid + b_o`: the box
            // endpoints were widened OUTWARD by `in_features + 2` ULPs, so the
            // midpoint is off by ≤ (in_features + 2.5)·ulp(max|endpoint|); we use a
            // generous (in_features + 4)·ulp margin.
            let radius_term = next_up_f32(w_row_l2[o] * r_eff);
            let max_abs = bl.abs().max(bu.abs());
            let ulp_max = next_up_f32(max_abs) - max_abs; // 1 ULP at the larger endpoint
            let mu = next_up_f32(((in_features as f32) + 4.0) * ulp_max);
            let delta = next_up_f32(radius_term + mu);
            if !delta.is_finite() {
                continue;
            }
            let cs_lower = next_down_f32(next_down_f32(mid) - delta);
            let cs_upper = next_up_f32(next_up_f32(mid) + delta);

            // Intersect: keep the tighter (higher lower / lower upper). Both the
            // box and CS endpoints are sound enclosures of the true value, so the
            // intersection still contains it (no inversion against the box).
            let new_l = if cs_lower > bl { cs_lower } else { bl };
            let new_u = if cs_upper < bu { cs_upper } else { bu };
            // Final guard: never emit an inverted interval (rounding paranoia).
            if new_l <= new_u {
                out_lower[[s, o]] = new_l;
                out_upper[[s, o]] = new_u;
            }
        }
    }

    let mut out_shape: Vec<usize> = in_shape[..ndim - 1].to_vec();
    out_shape.push(out_features);
    let lower_nd = reshape_2d_to_nd(out_lower, &out_shape).ok()?;
    let upper_nd = reshape_2d_to_nd(out_upper, &out_shape).ok()?;
    // Tightened box only; the output's own L2 sphere is not propagated here.
    BoundedTensor::new_repaired(lower_nd, upper_nd, RepairStrategy::Conservative).ok()
}

#[cfg(test)]
mod degenerate_point_gemm_certificate_tests {
    use ndarray::array;
    use ny_core::{GemmEngine, Result};
    use ny_tensor::BoundedTensor;

    use crate::layers::linear::LinearLayer;

    /// Reference `GemmEngine`: plain IEEE round-to-nearest f32, sequential over the
    /// contraction axis. This is exactly the `GemmEngine::gemm_f32` precision
    /// contract that `certify_point_gemm`'s Higham bound is derived against.
    struct NaiveF32Gemm;

    impl GemmEngine for NaiveF32Gemm {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            let mut out = vec![0.0_f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0_f32;
                    for l in 0..k {
                        acc += a[i * k + l] * b[l * n + j];
                    }
                    out[i * n + j] = acc;
                }
            }
            Ok(out)
        }
    }

    /// A POINT input must NOT be published as an exact point (#degenerate-ibp-gemm).
    ///
    /// COUNTEREXAMPLE: `W = [2^24, 1, -2^24]`, `x = [1, 1, 1]`, no bias. The true
    /// value is `1`. The f32 contraction computes `2^24 + 1 -> 2^24` (an exact tie,
    /// rounded to even) and then `2^24 - 2^24 = 0`, so `fl(y) = 0`. Before the fix,
    /// `propagate_ibp_via_gemm`'s degenerate short-circuit returned
    /// `BoundedTensor::concrete(fl(y))` = `[0, 0]` — a ZERO-WIDTH box that EXCLUDES
    /// the true value. Feeding that to a pre-activation or a verdict row is a false
    /// `unsat`. This assertion FAILS on the pre-fix code.
    #[ntest::timeout(10000)]
    #[test]
    fn point_input_box_encloses_the_true_value_under_cancellation() {
        const BIG: f32 = 16_777_216.0; // 2^24

        // Pin the counterexample's own arithmetic: if this fold ever stops
        // cancelling, the test would silently stop testing anything.
        // x is all ones, so the contraction is just the weight fold.
        let mut fl = 0.0_f32;
        for w in [BIG, 1.0_f32, -BIG] {
            fl += w;
        }
        assert_eq!(
            fl, 0.0,
            "counterexample precondition: the f32 contraction must cancel to 0"
        );

        let layer = LinearLayer::new(array![[BIG, 1.0_f32, -BIG]], None).expect("layer");
        let point = BoundedTensor::concrete(array![1.0_f32, 1.0, 1.0].into_dyn()).expect("point");

        let out = layer
            .propagate_ibp_with_engine(&point, Some(&NaiveF32Gemm))
            .expect("point IBP");

        let lo = out.lower()[[0]];
        let hi = out.upper()[[0]];
        assert!(
            lo <= 1.0 && 1.0 <= hi,
            "point-input IBP box [{lo}, {hi}] must enclose the true value 1.0"
        );
        // The certificate is S-scaled (`S ~ 2^25` here), NOT result-scaled: `|y| = 0`,
        // so ANY "widen by n ULPs of the result" patch still excludes 1.0. Pin that
        // the charge survives full cancellation.
        assert!(
            hi - lo >= 2.0,
            "certificate must be S-scaled, not result-scaled; got width {}",
            hi - lo
        );
    }

    /// The certificate is proportional to the row's MAGNITUDE sum, so a
    /// well-conditioned point stays tight — the fix must not be a blunt instrument.
    #[ntest::timeout(10000)]
    #[test]
    fn point_input_box_stays_tight_when_the_row_does_not_cancel() {
        let layer = LinearLayer::new(array![[1.0_f32, 2.0]], Some(array![0.5_f32])).expect("layer");
        let point = BoundedTensor::concrete(array![3.0_f32, 4.0].into_dyn()).expect("point");

        let out = layer
            .propagate_ibp_with_engine(&point, Some(&NaiveF32Gemm))
            .expect("point IBP");

        let lo = out.lower()[[0]];
        let hi = out.upper()[[0]];
        // y = 1*3 + 2*4 + 0.5 = 11.5, and every step of that is exact in f32.
        assert!(
            lo <= 11.5 && 11.5 <= hi,
            "box [{lo}, {hi}] must enclose 11.5"
        );
        assert!(
            hi > lo,
            "a point input must still receive a non-degenerate box"
        );
        // S = 1*3 + 2*4 = 11, so err ~ gamma_4 * 11 + 2u * 11.5 ~ 4e-6.
        assert!(
            hi - lo < 1e-4,
            "well-conditioned point must stay tight; got width {}",
            hi - lo
        );
    }
}

#[cfg(test)]
mod linear_ibp_higham_certificate_tests {
    use super::*;

    /// `2^24`, the smallest f32 magnitude whose ULP is 2: `fl32(2^24 + 1) = 2^24`
    /// under round-half-to-even, in ANY summation order.
    fn two_pow_24() -> f32 {
        (1u32 << 24) as f32
    }

    /// `2^60`. Its f64 ULP is `2^7`, so `fl64(2^60 + 1) = 2^60` with no reliance on
    /// tie-breaking at all — the rank-1 path accumulates in f64.
    fn two_pow_60() -> f32 {
        (1u64 << 60) as f32
    }

    fn only_element(values: &ArrayD<f32>) -> f32 {
        assert_eq!(values.len(), 1, "test rows are 1x1");
        *values.iter().next().expect("one element")
    }

    /// N-D (rank >= 2) faer path. THE hand-verified counterexample of
    /// #linear-ibp-higham: `W = [2^24, 1, -2^24]`, no bias, `x = [1, 1, 1]`.
    ///
    /// `w_pos = [2^24, 1, 0]` contracts to `fl32(2^24 + 1) = 2^24`;
    /// `w_neg = [0, 0, -2^24]` contracts to `-2^24`; their f32 sum is `0.0`, while the
    /// true value is `2^24 + 1 - 2^24 = 1`. Before the fix the ONLY widening was
    /// `round_for_soundness_n_ulps_inplace(3 + 2)`, and `shift_down_n_ulps(0.0, 5)` is
    /// five subnormals (~7e-45), so the published box EXCLUDED the true value — which
    /// a downstream ReLU turns into "stable INACTIVE, output pinned to 0" when the
    /// truth is `ReLU(1) = 1`: a false constraint, i.e. a false `unsat`.
    #[test]
    fn nd_linear_ibp_encloses_true_value_under_catastrophic_cancellation() {
        let big = two_pow_24();
        let weight = Array2::from_shape_vec((1, 3), vec![big, 1.0, -big]).expect("weight shape");
        let layer = LinearLayer::new(weight, None).expect("layer");
        let point =
            ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0_f32, 1.0, 1.0]).expect("input shape");
        let input = BoundedTensor::concrete(point).expect("input box");

        let out = layer.propagate_ibp(&input).expect("ibp");
        let lower = only_element(out.lower());
        let upper = only_element(out.upper());

        assert!(
            lower <= 1.0 && upper >= 1.0,
            "rank-2 Linear IBP published [{lower}, {upper}], which EXCLUDES the true \
             value 1.0 (#linear-ibp-higham): an n-ULP-OF-THE-RESULT charge collapses \
             together with the cancelling result"
        );
        // The certificate's OWN observable, independent of how the matmul rounded:
        // `gamma^f32_5 * S` with `S = |W| @ |x| = 2^25` is ~10, so a box narrower than
        // 1.0 means the abssum term never engaged.
        assert!(
            upper - lower > 1.0,
            "abssum certificate did not engage: width {} is far below gamma_5*S ~ 10",
            upper - lower
        );
    }

    /// Engine-GEMM path (`propagate_ibp_via_gemm`), same row.
    ///
    /// A fourth coordinate with weight ZERO and a WIDE box keeps `lower != upper`, so
    /// this exercises the four-GEMM interval arm rather than the point arm
    /// (#degenerate-ibp-gemm), while leaving the true value at exactly 1 for every
    /// point of the box.
    #[test]
    fn engine_linear_ibp_encloses_true_value_under_catastrophic_cancellation() {
        let big = two_pow_24();
        let weight =
            Array2::from_shape_vec((1, 4), vec![big, 1.0, -big, 0.0]).expect("weight shape");
        let layer = LinearLayer::new(weight, None).expect("layer");
        let lower_in = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0_f32, 1.0, 1.0, -1.0])
            .expect("lower shape");
        let upper_in = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0_f32, 1.0, 1.0, 1.0])
            .expect("upper shape");
        let input = BoundedTensor::new(lower_in, upper_in).expect("input box");
        let engine = crate::faer_parallelism::FaerCpuGemmEngine;

        let out = layer
            .propagate_ibp_with_engine(&input, Some(&engine))
            .expect("engine ibp");
        let lower = only_element(out.lower());
        let upper = only_element(out.upper());

        assert!(
            lower <= 1.0 && upper >= 1.0,
            "engine-GEMM Linear IBP published [{lower}, {upper}], which EXCLUDES the \
             true value 1.0 (#linear-ibp-higham)"
        );
        assert!(
            upper - lower > 1.0,
            "abssum certificate did not engage on the engine path: width {}",
            upper - lower
        );
    }

    /// Rank-1 path (`propagate_ibp_1d`), which accumulates in f64 and then widened by
    /// exactly ONE f32 ULP of the RESULT. That certifies the CAST, not the f64
    /// accumulation residual `gamma^f64_n * S`. With `W = [2^60, 1, -2^60]` the f64
    /// contraction publishes `0.0` where the true value is 1, and one ULP of `0.0` is
    /// a single subnormal (~1.4e-45).
    #[test]
    fn rank1_linear_ibp_encloses_true_value_under_f64_cancellation() {
        let big = two_pow_60();
        let weight = Array2::from_shape_vec((1, 3), vec![big, 1.0, -big]).expect("weight shape");
        let layer = LinearLayer::new(weight, None).expect("layer");
        let point =
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0_f32, 1.0, 1.0]).expect("input shape");
        let input = BoundedTensor::concrete(point).expect("input box");

        let out = layer.propagate_ibp(&input).expect("ibp");
        let lower = only_element(out.lower());
        let upper = only_element(out.upper());

        assert!(
            lower <= 1.0 && upper >= 1.0,
            "rank-1 Linear IBP published [{lower}, {upper}], which EXCLUDES the true \
             value 1.0: the f64 accumulation residual is gamma^f64_n * S, not one f32 \
             ULP of the result (#linear-ibp-higham)"
        );
    }

    /// The guarded casts must never publish an infinity SENTINEL as a bound on a
    /// FINITE quantity: `next_down_f32(1e39_f64 as f32) == +inf` would be a LOWER
    /// bound of `+inf`, and its mirror an UPPER bound of `-inf`.
    #[test]
    fn guarded_f64_to_f32_casts_saturate_finite_overflow_outward() {
        let above = 1.0e39_f64;
        assert_eq!(linear_f64_to_f32_down(above), f32::MAX);
        assert_eq!(linear_f64_to_f32_up(-above), f32::MIN);
        // In-range values keep their historical directed result exactly.
        assert_eq!(linear_f64_to_f32_down(1.0), next_down_f32(1.0));
        assert_eq!(linear_f64_to_f32_up(1.0), next_up_f32(1.0));
        assert_eq!(
            linear_f64_to_f32_down(f64::from(f32::MAX)),
            next_down_f32(f32::MAX)
        );
    }
}
#[cfg(test)]
mod linear_ibp_abs_sum_cost_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// Counting `GemmEngine`: plain IEEE round-to-nearest f32, sequential over the
    /// contraction axis (the `GemmEngine::gemm_f32` precision contract), plus a
    /// dispatch counter. The counter is the observable the batching pins are
    /// written against.
    struct CountingF32Gemm {
        calls: AtomicUsize,
    }

    impl CountingF32Gemm {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    impl GemmEngine for CountingF32Gemm {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut out = vec![0.0_f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0_f32;
                    for l in 0..k {
                        acc += a[i * k + l] * b[l * n + j];
                    }
                    out[i * n + j] = acc;
                }
            }
            Ok(out)
        }
    }

    /// Byte-for-byte the abssum body that `linear_ibp_abs_sum_faer` replaced:
    /// `S = (xmax·W⁺ᵀ) − (xmax·W⁻ᵀ)`, two matmuls against the same cached faer
    /// transposes the four bound matmuls use. Kept here ONLY as the reference the
    /// direction pin compares against.
    fn two_matmul_abs_sum_reference(layer: &LinearLayer, xmax: &Array2<f32>) -> Array2<f32> {
        let (batch_size, in_features) = xmax.dim();
        let out_features = layer.out_features();
        let xmax_faer = Mat::<f32>::from_fn(batch_size, in_features, |i, j| xmax[[i, j]]);
        let s_pos = mat_mul(&xmax_faer, layer.w_pos_t_faer());
        let s_neg = mat_mul(&xmax_faer, layer.w_neg_t_faer());
        Array2::<f32>::from_shape_fn((batch_size, out_features), |(i, j)| {
            s_pos[(i, j)] - s_neg[(i, j)]
        })
    }

    /// Deterministic pseudo-random f32 in `[-scale, scale]`, so the direction pin
    /// exercises mixed signs and mixed magnitudes without a dependency.
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((*state >> 40) as f32) / ((1u32 << 24) as f32);
        2.0 * unit - 1.0
    }

    fn mixed_sign_layer(out_features: usize, in_features: usize) -> LinearLayer {
        let mut state = 0x5eed_1234_9abc_def0_u64;
        let mut weight = Vec::with_capacity(out_features * in_features);
        for k in 0..out_features * in_features {
            // Spread over four binades so the two accumulations round differently.
            let scale = [1.0_f32, 64.0, 1.0 / 512.0, 4096.0][k % 4];
            weight.push(lcg_next(&mut state) * scale);
        }
        let weight = Array2::from_shape_vec((out_features, in_features), weight).expect("weight");
        LinearLayer::new(weight, None).expect("layer")
    }

    /// DIRECTION PIN (#linear-ibp-higham). The one-matmul abssum is a COST change, so
    /// it must never publish a value below the two-matmul form it replaced — a lower
    /// `S` is a narrower certificate, which is the one direction that can exclude a
    /// true value. `linear_ibp_abs_sum_split_bump` is what guarantees this; delete or
    /// shrink it and this fails.
    #[ntest::timeout(10000)]
    #[test]
    fn one_matmul_abs_sum_never_falls_below_the_two_matmul_form() {
        let layer = mixed_sign_layer(5, 61);
        let mut state = 0xfeed_face_dead_beef_u64;
        let xmax = Array2::from_shape_fn((7, 61), |_| lcg_next(&mut state).abs() * 32.0);

        let one = linear_ibp_abs_sum_faer(&layer, &xmax);
        let two = two_matmul_abs_sum_reference(&layer, &xmax);
        assert_eq!(one.dim(), two.dim(), "abssum shape must be unchanged");

        let mut positive_entries = 0usize;
        for ((idx, &new), &old) in one.indexed_iter().zip(two.iter()) {
            assert!(
                new >= old,
                "one-matmul abssum {new} fell BELOW the two-matmul abssum {old} at {idx:?}: \
                 the certificate would publish a narrower box than before"
            );
            if old > 0.0 {
                positive_entries += 1;
            }
        }
        assert_eq!(
            positive_entries,
            one.len(),
            "vacuous pin: every reference abssum entry must be strictly positive"
        );
    }

    /// The bump is the ONLY reason the pin above holds, so pin the factor itself
    /// rather than only its effect: it must be at least `1`, and at most `1 + 4γ_n`
    /// (a bump larger than that would be widening the certificate for no derivation).
    #[ntest::timeout(10000)]
    #[test]
    fn abs_sum_split_bump_is_at_least_one_and_barely_above_it() {
        for n in [1_usize, 2, 61, 784, 4096, 1 << 20] {
            let gamma = crate::layers::linear::crown_single_gamma_n_f32(n);
            let bump = linear_ibp_abs_sum_split_bump(n);
            assert!(
                bump >= 1.0,
                "bump {bump} at n={n} is below 1: the one-matmul abssum could fall \
                 below the two-matmul form"
            );
            assert!(
                bump <= 1.0 + 4.0 * gamma,
                "bump {bump} at n={n} exceeds 1 + 4*gamma_n ({}): unjustified widening",
                1.0 + 4.0 * gamma
            );
        }
        assert!(
            linear_ibp_abs_sum_split_bump(1 << 24).is_infinite(),
            "gamma_n >= 1 must saturate the bump so widen_linear_ibp_higham opens the row"
        );
    }

    /// COST PIN. A POINT input must cost exactly ONE `GemmEngine` dispatch —
    /// the `y = W*x + b` forward. The abssum certificate rides the layer's cached
    /// `|W|` through faer and adds no dispatch. This is the assertion the
    /// `test_linear_ibp_with_engine_uses_single_gemm_for_concrete_input_3954` and
    /// `test_pgd_evaluate_with_engine_threads_ibp_gemm_3954` pins make from outside;
    /// it lives here too so a change inside this file fails at its own site.
    ///
    /// The enclosure assertion is deliberately in the SAME test: a future "fix" that
    /// restores the count by deleting the certificate would satisfy the count alone.
    #[ntest::timeout(10000)]
    #[test]
    fn point_input_certificate_adds_no_engine_gemm_and_still_encloses() {
        const BIG: f32 = 16_777_216.0; // 2^24
        let layer = LinearLayer::new(
            Array2::from_shape_vec((1, 3), vec![BIG, 1.0, -BIG]).expect("weight"),
            None,
        )
        .expect("layer");
        let point = BoundedTensor::concrete(
            ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0_f32, 1.0, 1.0]).expect("point"),
        )
        .expect("point box");

        let engine = CountingF32Gemm::new();
        let out = layer
            .propagate_ibp_with_engine(&point, Some(&engine))
            .expect("point IBP");

        assert_eq!(
            engine.calls(),
            1,
            "point Linear IBP must issue exactly one engine GEMM (the forward); the \
             abssum certificate contracts the cached |W| through faer"
        );

        let lower = out.lower().iter().copied().next().expect("lower");
        let upper = out.upper().iter().copied().next().expect("upper");
        assert!(
            lower <= 1.0 && 1.0 <= upper,
            "point box [{lower}, {upper}] must still enclose the true value 1.0"
        );
        assert!(
            upper - lower >= 2.0,
            "certificate must stay S-scaled under full cancellation; width {}",
            upper - lower
        );
    }

    /// COST PIN, interval arm: FOUR engine dispatches — `W⁺·l`, `W⁻·u`,
    /// `W⁺·u`, `W⁻·l` — and no fifth or sixth for the abssum.
    #[ntest::timeout(10000)]
    #[test]
    fn interval_input_certificate_adds_no_engine_gemm_and_still_encloses() {
        // A fourth coordinate with weight ZERO and a wide box keeps lower != upper,
        // so this is the interval arm, while the true value stays exactly 1.
        const BIG: f32 = 16_777_216.0;
        let layer = LinearLayer::new(
            Array2::from_shape_vec((1, 4), vec![BIG, 1.0, -BIG, 0.0]).expect("weight"),
            None,
        )
        .expect("layer");
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0_f32, 1.0, 1.0, -1.0]).expect("lower"),
            ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0_f32, 1.0, 1.0, 1.0]).expect("upper"),
        )
        .expect("input box");

        let engine = CountingF32Gemm::new();
        let out = layer
            .propagate_ibp_with_engine(&input, Some(&engine))
            .expect("interval IBP");

        assert_eq!(
            engine.calls(),
            4,
            "interval Linear IBP must issue exactly four engine GEMMs (the sign-split \
             bounds); the abssum certificate contracts the cached |W| through faer"
        );

        let lower = out.lower().iter().copied().next().expect("lower");
        let upper = out.upper().iter().copied().next().expect("upper");
        assert!(
            lower <= 1.0 && 1.0 <= upper,
            "interval box [{lower}, {upper}] must still enclose the true value 1.0"
        );
        assert!(
            upper - lower > 1.0,
            "abssum certificate did not engage on the interval arm: width {}",
            upper - lower
        );
    }
}
