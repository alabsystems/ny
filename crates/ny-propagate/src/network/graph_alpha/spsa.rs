// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SPSA gradient estimation and alpha selection for alpha-CROWN on `GraphNetwork`.

use crate::bounds::alpha_reciprocal::ReciprocalGradients;
use crate::bounds::{GraphAlphaState, MonotoneSShapedGradients, SqrtGradients};
use crate::faer_parallelism::RayonTaskGuard;
use crate::network::alpha_crown_loop::finite_lower_sum;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};
use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::BTreeMap;

use super::spsa_accumulate::{
    accumulate_monotone_gradients, accumulate_reciprocal_gradients, accumulate_sqrt_gradients,
};

/// Compute the lower bound of `spec^T output` using interval arithmetic (#4355).
///
/// For each coefficient `c_i` in the spec vector:
/// - If `c_i >= 0`, the lower bound contribution is `c_i * lower_i`
/// - If `c_i < 0`, the lower bound contribution is `c_i * upper_i`
///
/// This is the standard interval arithmetic formula for a linear combination.
pub(super) fn spec_guided_lower(bounds: &BoundedTensor, spec: &[f32]) -> f32 {
    // Flatten to handle arbitrary-dimensional output bounds (e.g. [1, 10]).
    let lower_iter = bounds.lower().iter().copied();
    let upper_iter = bounds.upper().iter().copied();
    let mut result = 0.0f32;
    for (&c, (lo, hi)) in spec.iter().zip(lower_iter.zip(upper_iter)) {
        let contribution = if c >= 0.0 { c * lo } else { c * hi };
        if contribution.is_finite() {
            result += contribution;
        }
    }
    result
}

#[derive(Debug)]
pub(super) struct DagSpsaGradients {
    pub(super) relu: BTreeMap<String, Array1<f32>>,
    pub(super) monotone: BTreeMap<String, MonotoneSShapedGradients>,
    pub(super) sqrt: BTreeMap<String, SqrtGradients>,
    pub(super) reciprocal: BTreeMap<String, ReciprocalGradients>,
}

impl GraphNetwork {
    /// Compute SPSA gradients for DAG α-CROWN with optional sparse mask.
    ///
    /// When sparse_mask is provided, only perturb alphas where mask[name][i] is true.
    /// This reduces SPSA variance by focusing perturbations on influential alphas.
    // Justification: SPSA gradient estimation needs input, IBP bounds, alpha state, output node,
    // output dim, perturbation magnitude, sparse mask, and engine — all independent parameters.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_spsa_gradients_dag_for_output_sparse(
        &self,
        input: &BoundedTensor,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        output_node: &str,
        eps: f32,
        num_samples: usize,
        sparse_mask: Option<&BTreeMap<String, Array1<bool>>>,
        engine: Option<&dyn ny_core::GemmEngine>,
    ) -> Result<DagSpsaGradients> {
        self.compute_spsa_gradients_impl(
            input,
            ibp_bounds,
            alpha_state,
            output_node,
            eps,
            num_samples,
            sparse_mask,
            engine,
            None, // No spec objective — optimize sum of output lower bounds
        )
    }

    /// Compute SPSA gradients targeting a specific spec-guided objective (#4355).
    ///
    /// Instead of maximizing `sum(output_lower)`, maximizes the lower bound of
    /// `c^T output` where `c` is the spec vector. Used for per-disjunct alpha
    /// optimization in `optimize_disjuncts_separately` mode.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_spsa_gradients_for_spec_objective(
        &self,
        input: &BoundedTensor,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        output_node: &str,
        eps: f32,
        num_samples: usize,
        sparse_mask: Option<&BTreeMap<String, Array1<bool>>>,
        engine: Option<&dyn ny_core::GemmEngine>,
        spec_row: &[f32],
    ) -> Result<DagSpsaGradients> {
        self.compute_spsa_gradients_impl(
            input,
            ibp_bounds,
            alpha_state,
            output_node,
            eps,
            num_samples,
            sparse_mask,
            engine,
            Some(spec_row),
        )
    }

    /// Internal SPSA gradient computation with configurable objective.
    ///
    /// When `spec_objective` is `None`, targets `sum(output_lower)` (standard).
    /// When `Some(spec)`, targets the lower bound of `spec^T output` (#4355).
    #[allow(clippy::too_many_arguments)]
    fn compute_spsa_gradients_impl(
        &self,
        input: &BoundedTensor,
        ibp_bounds: &std::collections::HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        output_node: &str,
        eps: f32,
        num_samples: usize,
        sparse_mask: Option<&BTreeMap<String, Array1<bool>>>,
        engine: Option<&dyn ny_core::GemmEngine>,
        spec_objective: Option<&[f32]>,
    ) -> Result<DagSpsaGradients> {
        use rand::RngExt;
        use rayon::prelude::*;

        // Step 1: Pre-generate all perturbations for all samples
        let mut rng = crate::random::rng();
        let mut all_relu_perturbations: Vec<BTreeMap<String, Array1<f32>>> =
            Vec::with_capacity(num_samples);
        let mut all_monotone_perturbations: Vec<BTreeMap<String, MonotoneSShapedGradients>> =
            Vec::with_capacity(num_samples);
        let mut all_sqrt_perturbations: Vec<BTreeMap<String, SqrtGradients>> =
            Vec::with_capacity(num_samples);
        let mut all_reciprocal_perturbations: Vec<BTreeMap<String, ReciprocalGradients>> =
            Vec::with_capacity(num_samples);

        // GraphAlphaState.alphas is a BTreeMap, so iteration is deterministic
        // (sorted by key). This ensures the seeded RNG consumes random numbers
        // in a consistent order across runs. See #1976.
        for _ in 0..num_samples {
            let mut relu_perturbations: BTreeMap<String, Array1<f32>> = BTreeMap::new();
            for (name, alpha) in &alpha_state.alphas {
                let unstable_mask = alpha_state.unstable_mask.get(name);
                let active_mask = sparse_mask.and_then(|m| m.get(name));

                let pert = Array1::from_iter((0..alpha.len()).map(|i| {
                    let is_unstable = unstable_mask.map(|m| m[i]).unwrap_or(false);
                    // If sparse_mask is provided, only perturb if also active
                    let is_active = active_mask.map(|m| m[i]).unwrap_or(true);
                    if is_unstable && is_active {
                        if rng.random_bool(0.5) {
                            1.0
                        } else {
                            -1.0
                        }
                    } else {
                        0.0
                    }
                }));
                relu_perturbations.insert(name.clone(), pert);
            }

            let monotone_perturbations = alpha_state
                .monotone_alpha_names()
                .map(|name| {
                    let alpha = alpha_state.monotone_s_shaped_alpha(name).ok_or_else(|| {
                        NyError::InternalError(
                            "monotone alpha iterator must point to existing state".into(),
                        )
                    })?;
                    Ok((name.clone(), alpha.spsa_perturbations(&mut rng)))
                })
                .collect::<Result<_>>()?;
            let sqrt_perturbations = alpha_state
                .sqrt_alpha_names()
                .map(|name| {
                    let alpha = alpha_state.sqrt_alpha(name).ok_or_else(|| {
                        NyError::InternalError(
                            "sqrt alpha iterator must point to existing state".into(),
                        )
                    })?;
                    Ok((name.clone(), alpha.spsa_perturbations(&mut rng)))
                })
                .collect::<Result<_>>()?;
            let reciprocal_perturbations = alpha_state
                .reciprocal_alpha_names()
                .map(|name| {
                    let alpha = alpha_state.reciprocal_alpha(name).ok_or_else(|| {
                        NyError::InternalError(
                            "reciprocal alpha iterator must point to existing state".into(),
                        )
                    })?;
                    Ok((name.clone(), alpha.spsa_perturbations(&mut rng)))
                })
                .collect::<Result<_>>()?;

            all_relu_perturbations.push(relu_perturbations);
            all_monotone_perturbations.push(monotone_perturbations);
            all_sqrt_perturbations.push(sqrt_perturbations);
            all_reciprocal_perturbations.push(reciprocal_perturbations);
        }

        // Step 2: Create all perturbed alpha states FLATTENED
        let original_alphas = &alpha_state.alphas;
        let original_alphas_upper = &alpha_state.alphas_upper;
        let mut all_tasks: Vec<(usize, bool, GraphAlphaState)> =
            Vec::with_capacity(num_samples * 2);

        for (sample_idx, relu_perturbations) in all_relu_perturbations.iter().enumerate() {
            let monotone_perturbations = &all_monotone_perturbations[sample_idx];
            let sqrt_perturbations = &all_sqrt_perturbations[sample_idx];
            let reciprocal_perturbations = &all_reciprocal_perturbations[sample_idx];
            // +eps perturbation (both lower and upper paths, #3393).
            // `clone_for_backward` clones only the fields the backward pass reads
            // and leaves the unused Adam/SGD optimizer maps empty — these tasks are
            // only ever evaluated (never optimizer-updated), so the bound is
            // bit-identical while skipping six deep map copies per perturbation.
            let mut alpha_plus = alpha_state.clone_for_backward();
            for (name, pert) in relu_perturbations {
                if let (Some(orig), Some(plus_alpha)) =
                    (original_alphas.get(name), alpha_plus.alphas.get_mut(name))
                {
                    for i in 0..orig.len() {
                        plus_alpha[i] = (orig[i] + eps * pert[i]).clamp(0.0, 1.0);
                    }
                }
                if let (Some(orig_u), Some(plus_alpha_u)) = (
                    original_alphas_upper.get(name),
                    alpha_plus.alphas_upper.get_mut(name),
                ) {
                    for i in 0..orig_u.len() {
                        plus_alpha_u[i] = (orig_u[i] + eps * pert[i]).clamp(0.0, 1.0);
                    }
                }
            }
            for (name, perturbation) in monotone_perturbations {
                if let Some(alpha) = alpha_plus.monotone_s_shaped_alpha_mut(name) {
                    alpha.apply_perturbation(perturbation, eps);
                }
            }
            for (name, perturbation) in sqrt_perturbations {
                if let Some(alpha) = alpha_plus.sqrt_alpha_mut(name) {
                    alpha.apply_perturbation(perturbation, eps);
                }
            }
            for (name, perturbation) in reciprocal_perturbations {
                if let Some(alpha) = alpha_plus.reciprocal_alpha_mut(name) {
                    alpha.apply_perturbation(perturbation, eps);
                }
            }
            all_tasks.push((sample_idx, true, alpha_plus));

            // -eps perturbation (both lower and upper paths, #3393).
            // Same lightweight clone as the +eps task above (optimizer maps unused).
            let mut alpha_minus = alpha_state.clone_for_backward();
            for (name, pert) in relu_perturbations {
                if let (Some(orig), Some(minus_alpha)) =
                    (original_alphas.get(name), alpha_minus.alphas.get_mut(name))
                {
                    for i in 0..orig.len() {
                        minus_alpha[i] = (orig[i] - eps * pert[i]).clamp(0.0, 1.0);
                    }
                }
                if let (Some(orig_u), Some(minus_alpha_u)) = (
                    original_alphas_upper.get(name),
                    alpha_minus.alphas_upper.get_mut(name),
                ) {
                    for i in 0..orig_u.len() {
                        minus_alpha_u[i] = (orig_u[i] - eps * pert[i]).clamp(0.0, 1.0);
                    }
                }
            }
            for (name, perturbation) in monotone_perturbations {
                if let Some(alpha) = alpha_minus.monotone_s_shaped_alpha_mut(name) {
                    alpha.apply_perturbation(perturbation, -eps);
                }
            }
            for (name, perturbation) in sqrt_perturbations {
                if let Some(alpha) = alpha_minus.sqrt_alpha_mut(name) {
                    alpha.apply_perturbation(perturbation, -eps);
                }
            }
            for (name, perturbation) in reciprocal_perturbations {
                if let Some(alpha) = alpha_minus.reciprocal_alpha_mut(name) {
                    alpha.apply_perturbation(perturbation, -eps);
                }
            }
            all_tasks.push((sample_idx, false, alpha_minus));
        }

        // Step 3: Compute all CROWN bounds
        // Use sequential evaluation for small task counts (≤4) to ensure
        // deterministic floating-point results. Rayon parallelism can cause
        // non-determinism via CPU cache effects and thread scheduling (#1975).
        // For typical spsa_samples=1 (2 tasks), sequential is also faster
        // since it avoids thread pool overhead.
        let empty_map: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        let eval_task = |(sample_idx, is_plus, perturbed_alpha): &(
            usize,
            bool,
            GraphAlphaState,
        )|
         -> Result<(usize, bool, f32)> {
            // Use layout-agnostic iter().sum() instead of as_slice() which returns
            // None for non-contiguous arrays (#1939, #2024).
            // Propagate CROWN errors instead of swallowing them (#2063).
            // No per-node deadline for SPSA bound-evals: each eval is one of a
            // small fixed batch already bounded by the caller's iteration-level
            // deadline checks, and a mid-eval DeadlineExceeded here would be a
            // hard error (this closure propagates Err, #2063).
            let bounds = self.propagate_crown_to_node_with_alpha(
                input,
                output_node,
                &empty_map,
                ibp_bounds,
                perturbed_alpha,
                engine,
                None,
            )?;
            let lower: f32 = match spec_objective {
                Some(spec) => spec_guided_lower(&bounds, spec),
                None => finite_lower_sum(bounds.lower()),
            };
            Ok((*sample_idx, *is_plus, lower))
        };

        let all_results: Vec<(usize, bool, f32)> = if all_tasks.len() <= 4 {
            all_tasks
                .iter()
                .map(eval_task)
                .collect::<Result<Vec<_>>>()?
        } else {
            all_tasks
                .par_iter()
                .map(|task| {
                    let _rayon_task_guard = RayonTaskGuard::new();
                    eval_task(task)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        // Step 4: Reconstruct (lower_plus, lower_minus) pairs
        let mut sample_results: Vec<(f32, f32)> = vec![(0.0, 0.0); num_samples];
        for (sample_idx, is_plus, lower) in all_results {
            if is_plus {
                sample_results[sample_idx].0 = lower;
            } else {
                sample_results[sample_idx].1 = lower;
            }
        }

        // Step 5: Aggregate gradients
        let mut avg_relu_grads: BTreeMap<String, Array1<f32>> = BTreeMap::new();
        for (name, alpha) in &alpha_state.alphas {
            avg_relu_grads.insert(name.clone(), Array1::zeros(alpha.len()));
        }
        let mut avg_monotone_grads: BTreeMap<String, MonotoneSShapedGradients> = alpha_state
            .monotone_alpha_names()
            .map(|name| {
                let alpha = alpha_state.monotone_s_shaped_alpha(name).ok_or_else(|| {
                    NyError::InternalError(
                        "monotone alpha iterator must point to existing state".into(),
                    )
                })?;
                Ok((name.clone(), alpha.zeros_gradients()))
            })
            .collect::<Result<_>>()?;
        let mut avg_sqrt_grads: BTreeMap<String, SqrtGradients> = alpha_state
            .sqrt_alpha_names()
            .map(|name| {
                let alpha = alpha_state.sqrt_alpha(name).ok_or_else(|| {
                    NyError::InternalError(
                        "sqrt alpha iterator must point to existing state".into(),
                    )
                })?;
                Ok((name.clone(), alpha.zeros_gradients()))
            })
            .collect::<Result<_>>()?;
        let mut avg_reciprocal_grads: BTreeMap<String, ReciprocalGradients> = alpha_state
            .reciprocal_alpha_names()
            .map(|name| {
                let alpha = alpha_state.reciprocal_alpha(name).ok_or_else(|| {
                    NyError::InternalError(
                        "reciprocal alpha iterator must point to existing state".into(),
                    )
                })?;
                Ok((name.clone(), alpha.zeros_gradients()))
            })
            .collect::<Result<_>>()?;

        for (sample_idx, (lower_plus, lower_minus)) in sample_results.iter().enumerate() {
            let relu_perturbations = &all_relu_perturbations[sample_idx];
            let monotone_perturbations = &all_monotone_perturbations[sample_idx];
            let sqrt_perturbations = &all_sqrt_perturbations[sample_idx];
            let reciprocal_perturbations = &all_reciprocal_perturbations[sample_idx];
            let diff = lower_plus - lower_minus;

            // SPSA gradient estimate: g_i = (f+ - f-) / (2 * eps * Δ_i)
            for (name, pert) in relu_perturbations {
                if let Some(grad) = avg_relu_grads.get_mut(name) {
                    for i in 0..grad.len() {
                        if pert[i].abs() > 0.5 {
                            grad[i] += diff / (2.0 * eps * pert[i]);
                        }
                    }
                }
            }
            for (name, perturbation) in monotone_perturbations {
                if let Some(grad) = avg_monotone_grads.get_mut(name) {
                    accumulate_monotone_gradients(grad, perturbation, diff, eps);
                }
            }
            for (name, perturbation) in sqrt_perturbations {
                if let Some(grad) = avg_sqrt_grads.get_mut(name) {
                    accumulate_sqrt_gradients(grad, perturbation, diff, eps);
                }
            }
            for (name, perturbation) in reciprocal_perturbations {
                if let Some(grad) = avg_reciprocal_grads.get_mut(name) {
                    accumulate_reciprocal_gradients(grad, perturbation, diff, eps);
                }
            }
        }

        // Average the gradients
        // Guard against num_samples=0 to prevent NaN from division by zero (#2245, #2079).
        let num_samples_f32 = num_samples.max(1) as f32;
        for grad in avg_relu_grads.values_mut() {
            *grad /= num_samples_f32;
        }
        for grad in avg_monotone_grads.values_mut() {
            grad.scale_in_place(1.0 / num_samples_f32);
        }
        for grad in avg_sqrt_grads.values_mut() {
            grad.scale_in_place(1.0 / num_samples_f32);
        }
        for grad in avg_reciprocal_grads.values_mut() {
            grad.scale_in_place(1.0 / num_samples_f32);
        }

        Ok(DagSpsaGradients {
            relu: avg_relu_grads,
            monotone: avg_monotone_grads,
            sqrt: avg_sqrt_grads,
            reciprocal: avg_reciprocal_grads,
        })
    }

    /// Select top `ratio` fraction of alphas by gradient magnitude.
    ///
    /// Returns a mask where true indicates the alpha should be optimized.
    pub(super) fn select_top_alphas(
        gradients: &BTreeMap<String, Array1<f32>>,
        ratio: f32,
    ) -> BTreeMap<String, Array1<bool>> {
        // Collect all (gradient_magnitude, name, idx) tuples
        // BTreeMap iteration is sorted by key, so collection order is deterministic.
        let mut all_grads: Vec<(f32, &str, usize)> = Vec::new();
        for (name, grad) in gradients {
            for (i, &g) in grad.iter().enumerate() {
                all_grads.push((g.abs(), name.as_str(), i));
            }
        }

        // Sort by magnitude (descending, NaN last — #2995), with name+idx tiebreaker for determinism.
        all_grads.sort_by(|a, b| {
            crate::cmp_utils::nan_last_descending_cmp(&a.0, &b.0)
                .then_with(|| a.1.cmp(b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        // Select top `ratio` fraction.
        // Guard: NaN/Inf/negative ratio would produce garbage via `as usize` (saturates
        // to 0 or usize::MAX). Clamp to [0,1] and check NaN (clamp preserves NaN).
        let ratio = ratio.clamp(0.0, 1.0);
        let keep_count = if ratio.is_nan() || all_grads.is_empty() {
            1
        } else {
            // SAFETY(as usize): ratio is finite (NaN filtered), in [0,1] from clamp.
            // Result is in [0, all_grads.len()], non-negative and in-bounds.
            ((all_grads.len() as f32 * ratio).ceil() as usize).max(1)
        };

        // Build mask
        let mut mask: BTreeMap<String, Array1<bool>> = BTreeMap::new();
        for (name, grad) in gradients {
            mask.insert(name.clone(), Array1::from_elem(grad.len(), false));
        }

        for (_, name, idx) in all_grads.iter().take(keep_count) {
            if let Some(m) = mask.get_mut(*name) {
                m[*idx] = true;
            }
        }

        mask
    }

    /// Get bounds for a node reference, either from cache or network input.
    ///
    /// Returns a reference to avoid cloning BoundedTensor.
    ///
    /// Returns a reference to avoid cloning BoundedTensor.
    #[inline]
    pub(crate) fn bounds_ref<'a>(
        &self,
        name: &str,
        input: &'a BoundedTensor,
        cache: &'a std::collections::HashMap<String, BoundedTensor>,
    ) -> Result<&'a BoundedTensor> {
        if name == NETWORK_INPUT {
            Ok(input)
        } else {
            cache.get(name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Bounds for node {} not yet computed (dependency order error)",
                    name
                ))
            })
        }
    }
}

#[cfg(test)]
#[path = "spsa_tests.rs"]
mod tests;
