// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::patches::{
    CrownBounds, PatchesMaterializationDeadline, PatchesMaterializationPurpose,
};
#[cfg(test)]
use crate::bounds::patches_batched::BatchedCrownBounds;
use crate::bounds::{safe_mul_for_bounds_f64, BatchedLinearBounds, LinearBounds};

use ndarray::{Array, Array1, ArrayBase, ArrayView1, Data, Dimension, Ix1, Zip};
use ny_core::dd::{dd_fma, gamma_n_dd, next_down_f64, next_up_f64, Dd};
use ny_core::dd_selfcheck::dd_selfcheck_ok;
use ny_core::{f64_to_f32_down, f64_to_f32_up, NyError, Result};
use ny_tensor::BoundedTensor;
use std::mem::size_of;
use std::time::Instant;

use super::super::core::{GraphNetwork, NETWORK_INPUT};
use crate::network::CrownMergeAccumulator;

const SPEC_FALLBACK_SITE: &str = "spec-guided CROWN cooperative IBP fallback";

/// One request-local accounting receipt for the finite spec fallback. The
/// borrowed output box and specification matrix remain resident while both
/// sanitized endpoint vectors and both result vectors are staged, so all of
/// them participate in the same admission decision.
struct SpecFallbackAdmission {
    nominal_required_bytes: usize,
    capacity_overage_bytes: usize,
    budget_bytes: usize,
}

impl SpecFallbackAdmission {
    fn new(
        output_elements: usize,
        spec_elements: usize,
        result_elements: usize,
        budget_bytes: usize,
    ) -> Result<Self> {
        let resident_elements = output_elements
            .saturating_mul(2)
            .saturating_add(spec_elements);
        let staged_elements = output_elements
            .saturating_mul(2)
            .saturating_add(result_elements.saturating_mul(2));
        let nominal_required_bytes = resident_elements
            .saturating_add(staged_elements)
            .saturating_mul(size_of::<f32>());
        if nominal_required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: nominal_required_bytes,
                budget_bytes,
                site: SPEC_FALLBACK_SITE,
            });
        }
        Ok(Self {
            nominal_required_bytes,
            capacity_overage_bytes: 0,
            budget_bytes,
        })
    }

    fn allocation_error(&self) -> NyError {
        NyError::CpuMemoryExceeded {
            required_bytes: self
                .nominal_required_bytes
                .saturating_add(self.capacity_overage_bytes),
            budget_bytes: self.budget_bytes,
            site: SPEC_FALLBACK_SITE,
        }
    }

    fn reconcile_f32_capacity(&mut self, requested: usize, actual: usize) -> Result<()> {
        self.capacity_overage_bytes = self.capacity_overage_bytes.saturating_add(
            actual
                .saturating_sub(requested)
                .saturating_mul(size_of::<f32>()),
        );
        let required_bytes = self
            .nominal_required_bytes
            .saturating_add(self.capacity_overage_bytes);
        if required_bytes > self.budget_bytes {
            return Err(self.allocation_error());
        }
        Ok(())
    }
}

fn try_spec_fallback_vec(
    elements: usize,
    admission: &mut SpecFallbackAdmission,
    authority: &mut PatchesMaterializationDeadline,
) -> Result<Vec<f32>> {
    authority.checkpoint("before spec fallback allocation")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| admission.allocation_error())?;
    admission.reconcile_f32_capacity(elements, values.capacity())?;
    authority.checkpoint("after spec fallback allocation")?;
    Ok(values)
}

impl GraphNetwork {
    /// Fallback for spec-guided CROWN: compute spec bounds from IBP output bounds.
    // Generic over the map value (`BoundedTensor` or `Arc<BoundedTensor>` via
    // `Borrow`): the batched BaB caches are `Arc`-shared (#cone-delta
    // increment 2) while the graph-CROWN spec lanes pass owned maps.
    pub(crate) fn propagate_crown_with_specs_fallback_ibp<V: std::borrow::Borrow<BoundedTensor>>(
        &self,
        _input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        node_bounds: &std::collections::HashMap<String, V>,
        output_node_name: &str,
    ) -> Result<BoundedTensor> {
        let output_bounds = node_bounds
            .get(output_node_name)
            .map(std::borrow::Borrow::borrow)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Output node {} not found for IBP fallback",
                    output_node_name
                ))
            })?;

        Self::apply_spec_matrix_to_bounds_fallback(spec_matrix, output_bounds)
    }

    /// Deadline-aware form of [`Self::propagate_crown_with_specs_fallback_ibp`].
    /// The unlimited lane delegates to the historical implementation exactly.
    pub(crate) fn propagate_crown_with_specs_fallback_ibp_with_deadline<
        V: std::borrow::Borrow<BoundedTensor>,
    >(
        &self,
        input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        node_bounds: &std::collections::HashMap<String, V>,
        output_node_name: &str,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let Some(deadline) = deadline else {
            return self.propagate_crown_with_specs_fallback_ibp(
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
            );
        };
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "spec-guided CROWN IBP fallback: deadline exceeded before output lookup".into(),
            ));
        }
        let output_bounds = node_bounds
            .get(output_node_name)
            .map(std::borrow::Borrow::borrow)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Output node {} not found for IBP fallback",
                    output_node_name
                ))
            })?;
        Self::apply_spec_matrix_to_bounds_fallback_with_deadline(
            spec_matrix,
            output_bounds,
            Some(deadline),
        )
    }

    /// Apply a spec matrix directly to one sound output box. This is the IBP
    /// fallback primitive used both by graph-backed and empty-graph spec paths.
    pub(crate) fn apply_spec_matrix_to_bounds_fallback(
        spec_matrix: &ndarray::Array2<f32>,
        output_bounds: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let sanitized = Self::sanitize_bounds_for_fallback(output_bounds);
        let flat = sanitized.flatten();
        let lower_values: Vec<f32> = flat.lower().iter().copied().collect();
        let upper_values: Vec<f32> = flat.upper().iter().copied().collect();
        let num_specs = spec_matrix.nrows();
        let mut lower = Array1::<f32>::zeros(num_specs);
        let mut upper = Array1::<f32>::zeros(num_specs);

        for (i, spec_row) in spec_matrix.rows().into_iter().enumerate() {
            let (l, u) = Self::spec_row_interval_bounds(spec_row, &lower_values, &upper_values);
            lower[i] = l;
            upper[i] = u;
        }

        // Inf bounds are sound (conservative); NaN has been replaced above.
        BoundedTensor::new_allow_infinite(lower.into_dyn(), upper.into_dyn())
    }

    /// Apply a spec matrix to a box under one absolute deadline. All endpoint
    /// sanitation, copies, row reductions, and final invariant scans poll after
    /// at most 4,096 values. Publication is transactional: only fully built and
    /// validated arrays escape this method.
    pub(crate) fn apply_spec_matrix_to_bounds_fallback_with_deadline(
        spec_matrix: &ndarray::Array2<f32>,
        output_bounds: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let Some(deadline) = deadline else {
            return Self::apply_spec_matrix_to_bounds_fallback(spec_matrix, output_bounds);
        };
        Self::apply_spec_matrix_to_bounds_fallback_finite(
            spec_matrix,
            output_bounds,
            deadline,
            crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
        )
    }

    fn apply_spec_matrix_to_bounds_fallback_finite(
        spec_matrix: &ndarray::Array2<f32>,
        output_bounds: &BoundedTensor,
        deadline: Instant,
        budget_bytes: usize,
    ) -> Result<BoundedTensor> {
        let mut authority = PatchesMaterializationDeadline::new(Some(deadline));
        authority.checkpoint("before spec fallback validation")?;

        let output_elements = output_bounds.len();
        if spec_matrix.ncols() > output_elements {
            return Err(NyError::shape_mismatch(
                vec![spec_matrix.ncols()],
                vec![output_elements],
            ));
        }
        let result_elements = spec_matrix.nrows();
        let mut admission = SpecFallbackAdmission::new(
            output_elements,
            spec_matrix.len(),
            result_elements,
            budget_bytes,
        )?;

        // Sanitize directly into flat row-major vectors. This is identical to
        // sanitize_bounds_for_fallback(...).flatten(), without retaining a
        // second intermediate BoundedTensor.
        let mut lower_values =
            try_spec_fallback_vec(output_elements, &mut admission, &mut authority)?;
        let mut upper_values =
            try_spec_fallback_vec(output_elements, &mut admission, &mut authority)?;
        for (&source_lower, &source_upper) in output_bounds
            .lower()
            .iter()
            .zip(output_bounds.upper().iter())
        {
            let mut lower = source_lower;
            let mut upper = source_upper;
            if !lower.is_finite() {
                lower = f32::NEG_INFINITY;
            }
            if !upper.is_finite() {
                upper = f32::INFINITY;
            }
            if lower > upper {
                lower = f32::NEG_INFINITY;
                upper = f32::INFINITY;
            }
            lower_values.push(lower);
            upper_values.push(upper);
            authority.work(4, "while sanitizing spec fallback endpoints")?;
        }
        authority.checkpoint("after sanitizing spec fallback endpoints")?;

        let mut lower = try_spec_fallback_vec(result_elements, &mut admission, &mut authority)?;
        let mut upper = try_spec_fallback_vec(result_elements, &mut admission, &mut authority)?;
        for spec_row in spec_matrix.rows() {
            authority.checkpoint("before spec fallback row reduction")?;
            let (row_lower, row_upper) = Self::spec_row_interval_bounds_with_authority(
                spec_row,
                &lower_values,
                &upper_values,
                &mut authority,
            )?;
            lower.push(row_lower);
            upper.push(row_upper);
        }
        authority.checkpoint("after spec fallback row reductions")?;

        let lower = Array1::from_vec(lower).into_dyn();
        let upper = Array1::from_vec(upper).into_dyn();
        BoundedTensor::new_allow_infinite_with_poll(lower, upper, || {
            authority.checkpoint("while validating spec fallback publication")
        })
    }

    #[cfg(test)]
    pub(crate) fn apply_spec_matrix_to_bounds_fallback_with_budget_for_test(
        spec_matrix: &ndarray::Array2<f32>,
        output_bounds: &BoundedTensor,
        deadline: Instant,
        budget_bytes: usize,
    ) -> Result<BoundedTensor> {
        Self::apply_spec_matrix_to_bounds_fallback_finite(
            spec_matrix,
            output_bounds,
            deadline,
            budget_bytes,
        )
    }

    /// Reduce one spec row against an output box with interval arithmetic, rounding
    /// the returned endpoints OUTWARD.
    ///
    /// Sign-split rule: a coefficient `c >= 0` pairs with the box's lower endpoint for
    /// the lower bound and its upper endpoint for the upper bound; `c < 0` swaps them.
    /// Accumulated as an outward-rounded f64 interval and closed with a
    /// directed f32 conversion, mirroring [`LinearBounds::concretize_sound`].
    ///
    /// SOUNDNESS (#concretize-soundness-hardening): reducing a row with plain f32
    /// multiply-accumulate rounds to nearest at every product and every add, so the
    /// returned `lower` can land ABOVE the row's true minimum over the box (and `upper`
    /// below its true maximum) by up to `gamma_n * sum |c_j * y_j|` — a bias that
    /// cancellation makes unbounded relative to the result. The verdict consumers of
    /// this reduction (`domain_is_verified_for_mode`, `disjunctive_domain_verified`)
    /// compare against their threshold with no error budget and then permanently prune
    /// the domain, so an inward endpoint is undetectable and terminal. `f32 x f32`
    /// promoted to f64 is exact (48 < 53 significand bits). Each addition still
    /// needs directed rounding: a large positive term, a small term, and a large
    /// negative term can lose the small term even in f64. Widening only the final
    /// f32 cast does not cover that cancellation.
    ///
    /// `safe_mul_for_bounds_f64` keeps the interval convention that a zero coefficient
    /// contributes exactly 0 even against a saturated (infinite) endpoint, so one Inf
    /// output cannot poison the whole row (#3044). A NaN accumulator (0*Inf with a NaN
    /// coefficient, or Inf-Inf) collapses that side to its saturating bound: NaN lower
    /// → -Inf, NaN upper → +Inf, both sound over-approximations.
    ///
    /// REQUIRES: `lower_values.len() == upper_values.len() >= spec_row.len()`.
    /// ENSURES: `lower <= c^T y <= upper` for every `y` in the box.
    pub(crate) fn spec_row_interval_bounds(
        spec_row: ArrayView1<'_, f32>,
        lower_values: &[f32],
        upper_values: &[f32],
    ) -> (f32, f32) {
        let mut authority = PatchesMaterializationDeadline::new(None);
        Self::spec_row_interval_bounds_with_authority(
            spec_row,
            lower_values,
            upper_values,
            &mut authority,
        )
        .expect("unlimited spec-row reduction cannot observe deadline failure")
    }

    fn spec_row_interval_bounds_with_authority(
        spec_row: ArrayView1<'_, f32>,
        lower_values: &[f32],
        upper_values: &[f32],
        authority: &mut PatchesMaterializationDeadline,
    ) -> Result<(f32, f32)> {
        // Double-double keeps cancellation-heavy rows useful while its
        // certified gamma channel encloses the tiny residual arithmetic error.
        // The EFT self-check is mandatory: if the compiler/target breaks the
        // transforms, fall back to the wider per-addition interval below.
        let mut dd_authorized = dd_selfcheck_ok();
        for &value in &spec_row {
            dd_authorized &= value.is_finite();
            authority.work(1, "while scanning finite spec coefficients")?;
        }
        for values in [lower_values, upper_values] {
            for &value in &values[..spec_row.len()] {
                dd_authorized &= value.is_finite();
                authority.work(1, "while scanning finite fallback endpoints")?;
            }
        }
        if dd_authorized {
            let mut lower_acc = Dd::ZERO;
            let mut upper_acc = Dd::ZERO;
            let mut lower_abs_sum = 0.0_f64;
            let mut upper_abs_sum = 0.0_f64;
            for (j, &coefficient) in spec_row.iter().enumerate() {
                let coefficient = coefficient as f64;
                let input_l = lower_values[j] as f64;
                let input_u = upper_values[j] as f64;
                let (lower_input, upper_input) = if coefficient >= 0.0 {
                    (input_l, input_u)
                } else {
                    (input_u, input_l)
                };
                lower_acc = dd_fma(lower_acc, coefficient, lower_input);
                upper_acc = dd_fma(upper_acc, coefficient, upper_input);
                lower_abs_sum = next_up_f64(lower_abs_sum + (coefficient * lower_input).abs());
                upper_abs_sum = next_up_f64(upper_abs_sum + (coefficient * upper_input).abs());
                authority.work(3, "during double-double spec reduction")?;
            }

            let gamma = gamma_n_dd(spec_row.len());
            let lower_error = if gamma == 0.0 || lower_abs_sum == 0.0 {
                0.0
            } else {
                next_up_f64(gamma * lower_abs_sum)
            };
            let upper_error = if gamma == 0.0 || upper_abs_sum == 0.0 {
                0.0
            } else {
                next_up_f64(gamma * upper_abs_sum)
            };
            let represented_lower = next_down_f64(lower_acc.hi + lower_acc.lo);
            let represented_upper = next_up_f64(upper_acc.hi + upper_acc.lo);
            let result = (
                f64_to_f32_down(next_down_f64(represented_lower - lower_error)),
                f64_to_f32_up(next_up_f64(represented_upper + upper_error)),
            );
            authority.checkpoint("after double-double spec reduction")?;
            return Ok(result);
        }

        let mut l = 0.0f64;
        let mut u = 0.0f64;
        for (j, &c) in spec_row.iter().enumerate() {
            let c = c as f64;
            let input_l = lower_values[j] as f64;
            let input_u = upper_values[j] as f64;
            let (lower_term, upper_term) = if c >= 0.0 {
                (
                    safe_mul_for_bounds_f64(c, input_l),
                    safe_mul_for_bounds_f64(c, input_u),
                )
            } else {
                (
                    safe_mul_for_bounds_f64(c, input_u),
                    safe_mul_for_bounds_f64(c, input_l),
                )
            };

            let lower_sum = l + lower_term;
            l = if lower_sum.is_nan() {
                f64::NEG_INFINITY
            } else {
                next_down_f64(lower_sum)
            };
            let upper_sum = u + upper_term;
            u = if upper_sum.is_nan() {
                f64::INFINITY
            } else {
                next_up_f64(upper_sum)
            };
            authority.work(3, "during interval spec reduction")?;
        }

        let lower = f64_to_f32_down(l);
        let upper = f64_to_f32_up(u);
        authority.checkpoint("after interval spec reduction")?;
        Ok((lower, upper))
    }

    /// Accumulate CrownBounds for a given input node (Phase 1b, #2613).
    ///
    /// On first insertion, preserves the CrownBounds variant (Patches stays Patches
    /// so the next layer backward can operate on sparse structure). On subsequent
    /// accumulations (merge points), delegates to the checked Crown merge, which preserves
    /// compatible Patches carriers in-place and falls back to Dense+f64 for
    /// incompatible or mixed carriers (#4382).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accumulate_crown_bounds_to_input_with_deadline(
        &self,
        input_name: &str,
        new_bounds: CrownBounds,
        node_crown_bounds: &mut CrownMergeAccumulator,
        _output_dim: usize,
        _input_dim: usize,
        input_accumulated: &mut bool,
        deadline: Option<Instant>,
    ) -> Result<()> {
        self.accumulate_crown_bounds_to_input_with_deadline_authority(
            input_name,
            new_bounds,
            node_crown_bounds,
            _output_dim,
            _input_dim,
            input_accumulated,
            deadline,
            deadline.is_some(),
        )
    }

    /// Authority-aware accumulation used by collectors whose internal Patches
    /// scheduling timestamp is not a caller-visible hard deadline.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accumulate_crown_bounds_to_input_with_deadline_authority(
        &self,
        input_name: &str,
        new_bounds: CrownBounds,
        node_crown_bounds: &mut CrownMergeAccumulator,
        _output_dim: usize,
        _input_dim: usize,
        input_accumulated: &mut bool,
        deadline: Option<Instant>,
        deadline_is_hard: bool,
    ) -> Result<()> {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(
                "graph CROWN accumulation: deadline exceeded before publication".into(),
            ));
        }
        let is_network_input = input_name == NETWORK_INPUT;
        let is_first = if is_network_input {
            !*input_accumulated
        } else {
            !node_crown_bounds.contains_key(input_name)
        };

        if is_first {
            // First insertion: preserve Patches if applicable.
            if is_network_input {
                node_crown_bounds.insert(NETWORK_INPUT.to_string(), new_bounds);
                *input_accumulated = true;
            } else {
                node_crown_bounds.insert(input_name.to_string(), new_bounds);
            }
        } else {
            let key = if is_network_input {
                NETWORK_INPUT
            } else {
                input_name
            };
            node_crown_bounds.merge_crown_with_deadline_authority(
                key,
                new_bounds,
                deadline,
                deadline_is_hard,
            )?;
        }
        Ok(())
    }

    /// Concretize the current DAG/spec-CROWN frontier and collapse it onto the
    /// network input accumulator.
    ///
    /// This is the shared truncation primitive for #3813: the backward pass can
    /// stop at any frontier as long as each remaining node contribution is
    /// concretized with sound forward bounds before the final input concretization.
    /// Mirrors alpha-beta-CROWN's use of fixed intermediate bounds during BaB.
    /// Transactionally collapse a graph frontier under one absolute deadline.
    /// A complete Dense snapshot and replacement NETWORK_INPUT relation are
    /// built before the source accumulator is cleared, so any typed refusal
    /// leaves the original frontier and caller flag untouched.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn concretize_crown_frontier_to_network_input_with_deadline(
        &self,
        node_crown_bounds: &mut CrownMergeAccumulator,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        output_dim: usize,
        input_dim: usize,
        input_accumulated: &mut bool,
        deadline: Option<Instant>,
    ) -> Result<LinearBounds> {
        let staged_frontier = node_crown_bounds.snapshot_dense_with_deadline(deadline)?;
        let mut staged_accumulator = CrownMergeAccumulator::new();
        let mut staged_input_accumulated = false;
        let mut network_input_bounds = None;

        for (node_name, bounds) in staged_frontier {
            if node_name == NETWORK_INPUT {
                network_input_bounds = Some(CrownBounds::Dense(bounds));
                continue;
            }

            let node_ibp = node_bounds.get(&node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Truncated CROWN frontier missing forward bounds for node '{}'",
                    node_name
                ))
            })?;
            let concretized = bounds.concretize_sound_with_deadline(node_ibp, deadline)?;
            let (lower, upper) = concretized.into_parts();
            let lower = lower.into_dimensionality::<Ix1>().map_err(|error| {
                NyError::InternalError(format!(
                    "truncated CROWN frontier lower result was not 1D: {error}"
                ))
            })?;
            let upper = upper.into_dimensionality::<Ix1>().map_err(|error| {
                NyError::InternalError(format!(
                    "truncated CROWN frontier upper result was not 1D: {error}"
                ))
            })?;
            Self::accumulate_bias_to_network_input_crown_with_deadline(
                &lower,
                &upper,
                &mut staged_accumulator,
                output_dim,
                input_dim,
                &mut staged_input_accumulated,
                deadline,
            )?;
        }

        if let Some(bounds) = network_input_bounds {
            self.accumulate_crown_bounds_to_input_with_deadline(
                NETWORK_INPUT,
                bounds,
                &mut staged_accumulator,
                output_dim,
                input_dim,
                &mut staged_input_accumulated,
                deadline,
            )?;
        }

        let final_bounds = staged_accumulator
            .take_with_deadline(NETWORK_INPUT, deadline)?
            .ok_or_else(|| NyError::InvalidSpec("No path to network input found".to_string()))?
            .into_dense_with_deadline_for_purpose(
                deadline,
                PatchesMaterializationPurpose::NetworkInputTerminal,
            )?;
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(
                "graph CROWN frontier: deadline exceeded before transactional publication".into(),
            ));
        }

        node_crown_bounds.clear();
        *input_accumulated = staged_input_accumulated;
        Ok(final_bounds)
    }

    /// Element-wise addition of two arrays with NaN→conservative-infinity
    /// fallback (#2951). Generic over array dimension: works for `Array1`,
    /// `Array2`, `ArrayD`, etc. Returns a conservative (all ±Inf) array if
    /// shapes mismatch instead of panicking (#2907).
    ///
    /// # Inf coefficient invariant (#3032)
    ///
    /// When accumulating CROWN backward contributions across DAG merge points,
    /// NaN results (from inf + (-inf) cancellation) are replaced with ±Inf.
    /// This means `LinearBounds` coefficients may contain ±Inf after
    /// accumulation, which is intentional: Inf coefficients produce
    /// maximally-loose-but-sound bounds at concretization time. See the
    /// "Two-Phase Invariant" documentation on [`LinearBounds`].
    pub(crate) fn safe_add<D: Dimension>(
        existing: &Array<f32, D>,
        new: &Array<f32, D>,
        is_lower: bool,
    ) -> Array<f32, D> {
        if existing.shape() != new.shape() {
            if existing.len() == new.len() {
                if let Ok(reshaped) = new.view().into_shape_with_order(existing.raw_dim()) {
                    tracing::debug!(
                        existing_shape = ?existing.shape(),
                        new_shape = ?new.shape(),
                        "safe_add reshaping same-count contribution before accumulation"
                    );
                    return Self::safe_add_same_shape(existing, &reshaped, is_lower);
                }
            }
            tracing::warn!(
                existing_shape = ?existing.shape(),
                new_shape = ?new.shape(),
                "safe_add shape mismatch; returning conservative bounds"
            );
            let conservative = if is_lower {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
            return Array::from_elem(existing.raw_dim(), conservative);
        }
        Self::safe_add_same_shape(existing, new, is_lower)
    }

    /// Single element of [`safe_add`](Self::safe_add): NaN-safe f32 addition
    /// with the conservative-infinity degrade. Extracted so the flat SoA
    /// batched-backward merge (#lsnc-batched-bwd,
    /// `propagation/batched/batched_bwd.rs`) applies the IDENTICAL per-element
    /// arithmetic as the scalar reference (bit-parity by construction).
    #[inline]
    pub(crate) fn safe_add_elem(e: f32, n: f32, is_lower: bool) -> f32 {
        // NaN check: sum.is_nan() catches all cases:
        //   - NaN + x = NaN, x + NaN = NaN (input NaN)
        //   - inf + (-inf) = NaN (cancellation)
        let sum = e + n;
        if sum.is_nan() {
            if is_lower {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            sum
        }
    }

    /// Inner implementation of `safe_add` for shape-matched arrays.
    ///
    /// Replaces every NaN element in the sum with the conservative bound
    /// (−Inf for lower, +Inf for upper). This is the source of ±Inf
    /// coefficients in accumulated `LinearBounds` (see #3032).
    fn safe_add_same_shape<D, S>(
        existing: &Array<f32, D>,
        new: &ArrayBase<S, D>,
        is_lower: bool,
    ) -> Array<f32, D>
    where
        D: Dimension,
        S: Data<Elem = f32>,
    {
        // Use map_collect to construct the output directly, avoiding a
        // clone() + overwrite cycle that doubles allocation (#2220 F13).
        Zip::from(existing)
            .and(new)
            .map_collect(|&e, &n| Self::safe_add_elem(e, n, is_lower))
    }

    pub(crate) fn sanitize_bounds_for_fallback(bounds: &BoundedTensor) -> BoundedTensor {
        let mut lower = bounds.lower().clone();
        let mut upper = bounds.upper().clone();

        Zip::from(&mut lower).and(&mut upper).for_each(|l, u| {
            if !l.is_finite() {
                *l = f32::NEG_INFINITY;
            }
            if !u.is_finite() {
                *u = f32::INFINITY;
            }
            if *l > *u {
                *l = f32::NEG_INFINITY;
                *u = f32::INFINITY;
            }
        });

        // NaN replaced with ±Inf above; Inf is sound conservative fallback.
        match BoundedTensor::new_allow_infinite(lower, upper) {
            Ok(sanitized) => sanitized,
            Err(err) => {
                // #2812 Finding 49: Return conservative infinity bounds, not the
                // original unsanitized bounds which may contain NaN or inverted pairs.
                tracing::warn!(
                    "sanitize_bounds_for_fallback: shape mismatch {err}; \
                     returning conservative infinity bounds"
                );
                BoundedTensor::new_conservative(bounds.lower().shape())
            }
        }
    }

    /// Concretize accumulated linear bounds using IBP bounds as a partial CROWN fallback.
    ///
    /// When a layer's CROWN backward propagation fails or produces non-finite coefficients,
    /// we fall back to concretizing the accumulated linear bounds at this node using the
    /// pre-computed IBP bounds.
    ///
    /// Checks both IBP bounds AND accumulated linear bounds for Inf/NaN. If either is
    /// non-finite, sanitizes to IBP-only bounds. concretize_sound() guarantees no
    /// NaN/inversion (#2287).
    ///
    /// Part of design: crown-batched-partial-fallback-dedup (Slice 1).
    pub(crate) fn partial_crown_fallback(
        node_lb: &BatchedLinearBounds,
        ibp_bounds: &BoundedTensor,
        output_shape: &[usize],
    ) -> Result<BoundedTensor> {
        let ibp_has_bad = ibp_bounds
            .lower()
            .iter()
            .chain(ibp_bounds.upper().iter())
            .any(|v: &f32| !v.is_finite());

        let lb_has_bad = Self::has_non_finite_coefficients(node_lb);

        let result = if ibp_has_bad || lb_has_bad {
            Self::sanitize_bounds_for_fallback(ibp_bounds)
        } else {
            node_lb.concretize_sound(ibp_bounds)?
        };

        if result.shape() != output_shape {
            result.reshape(output_shape)
        } else {
            Ok(result)
        }
    }

    /// Check if any coefficient in the batched linear bounds is Inf or NaN.
    ///
    /// Part of design: crown-batched-partial-fallback-dedup (Slice 5).
    pub(crate) fn has_non_finite_coefficients(lb: &BatchedLinearBounds) -> bool {
        lb.lower_a
            .iter()
            .chain(lb.upper_a.iter())
            .chain(lb.lower_b.iter())
            .chain(lb.upper_b.iter())
            .any(|v: &f32| !v.is_finite())
    }

    /// Accumulate `BatchedCrownBounds` for a given input node (Phase 4, #2613).
    ///
    /// Mirrors [`accumulate_crown_bounds_to_input_with_deadline`] for the batched path.
    /// On first insertion, preserves the variant (Patches stays Patches).
    /// On merge (multiple paths converge), converts both to Dense and keeps the
    /// merged dense carrier in f64 until the node is consumed.
    #[cfg(test)]
    pub(crate) fn accumulate_batched_crown_bounds_to_input(
        &self,
        input_name: &str,
        new_bounds: BatchedCrownBounds,
        node_bounds: &mut std::collections::HashMap<String, BatchedCrownBounds>,
        input_accumulated: &mut bool,
    ) -> Result<()> {
        let is_network_input = input_name == NETWORK_INPUT;
        let is_first = if is_network_input {
            !*input_accumulated
        } else {
            !node_bounds.contains_key(input_name)
        };

        if is_first {
            // First insertion: preserve variant (Patches stays Patches).
            if is_network_input {
                node_bounds.insert(NETWORK_INPUT.to_string(), new_bounds);
                *input_accumulated = true;
            } else {
                node_bounds.insert(input_name.to_string(), new_bounds);
            }
        } else {
            // Merge point: convert both to Dense, then safe_add (#3550: checked).
            let new_blb = new_bounds
                .into_batched_dense_checked("accumulate_batched_crown_bounds_to_input:new")?;
            let key = if is_network_input {
                NETWORK_INPUT
            } else {
                input_name
            };
            if let Some(existing_bcb) = node_bounds.get_mut(key) {
                existing_bcb.merge_dense_checked(
                    new_blb,
                    "accumulate_batched_crown_bounds_to_input:existing",
                )?;
            } else {
                tracing::error!(
                    "accumulate_batched_crown_bounds_to_input: merge expected but {} \
                     missing — bounds dropped",
                    key
                );
                debug_assert!(false, "BatchedCrownBounds entry missing during merge");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "utils_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "utils_fallback_tests.rs"]
mod fallback_tests;

#[cfg(test)]
#[path = "utils_batched_merge_tests.rs"]
mod batched_merge_tests;
