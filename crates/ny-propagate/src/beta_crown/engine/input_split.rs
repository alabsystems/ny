// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Input-splitting helpers for β-CROWN.

use std::sync::Arc;
use std::time::Instant;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::trace;

use crate::beta_crown::domain::{BabDomain, IntermediateLinearBounds};
use crate::beta_crown::state::{BetaState, DomainAlphaState};
use crate::bounds::LinearBounds;
use crate::Network;

use super::bounds::crown_ibp_budget_exceeded;
use super::tensor_ext::BoundedTensorExt;
use super::BetaCrownVerifier;

/// Outcome of input-split branching on one BaB domain.
///
/// `Split` and an empty child list mean different things: the first is a
/// completed branch whose children cover the parent box, the second is a
/// parent box that no input dimension could divide. Collapsing them into a
/// bare `Vec` lets the BaB loop drain an unexplored box and call it Verified,
/// so the two stay distinct all the way to the unresolved flags.
pub(crate) enum InputSplitChildren {
    /// The parent box was midpoint-split; the children exactly cover it.
    /// Empty when every child was pruned (verified by clipping).
    Split(Vec<BabDomain>),
    /// No input dimension admits a split, so the parent box stays unexplored.
    Unsplittable,
}

impl BetaCrownVerifier {
    /// Select input dimension with margin-weighted SB heuristic.
    ///
    /// `domain_bounds` are the active verification-direction output bounds:
    /// lower bounds when `verify_upper_bound=false`, upper bounds otherwise.
    /// Together with `thresholds`, they enable per-spec margin-weighted scoring
    /// per the baseline SB heuristic. When `None`, margin weighting is disabled.
    ///
    /// Reference: alpha-beta-CROWN `branching_heuristics.py:input_split_heuristic_sb`
    pub(crate) fn select_input_dimension_sb(
        &self,
        input_bounds: &BoundedTensor,
        linear_bounds: Option<&LinearBounds>,
        domain_bounds: Option<&[f32]>,
        thresholds: Option<&[f32]>,
    ) -> usize {
        let scores =
            self.input_split_scores(input_bounds, linear_bounds, domain_bounds, thresholds);
        if scores.is_empty() {
            return 0;
        }
        let mut best_dim = scores[0].0;
        let mut best_score = scores[0].1;
        for (dim, score) in scores {
            // Skip NaN scores — corrupt scores must not win (#2588).
            if score.is_nan() {
                continue;
            }
            if best_score.is_nan() || score > best_score {
                best_score = score;
                best_dim = dim;
            }
        }
        best_dim
    }

    /// Select top-k input dimensions with margin-weighted SB heuristic.
    ///
    /// `domain_bounds` and `thresholds` enable per-spec margin-weighted scoring.
    pub(crate) fn select_input_dimensions_sb(
        &self,
        input_bounds: &BoundedTensor,
        linear_bounds: Option<&LinearBounds>,
        domain_bounds: Option<&[f32]>,
        thresholds: Option<&[f32]>,
    ) -> Vec<usize> {
        let mut scores =
            self.input_split_scores(input_bounds, linear_bounds, domain_bounds, thresholds);
        if scores.is_empty() {
            return Vec::new();
        }
        scores.sort_by(|a, b| {
            crate::cmp_utils::nan_last_descending_cmp(&a.1, &b.1).then_with(|| a.0.cmp(&b.0))
        });
        let depth = self.config.input_split_depth.min(scores.len());
        if depth == 0 {
            return Vec::new();
        }
        scores.into_iter().take(depth).map(|(dim, _)| dim).collect()
    }

    /// Compute SB heuristic scores for each input dimension.
    ///
    /// Implements the full baseline SB heuristic with three modes:
    /// - `sb_sum=true`: sum `|lA[s,d]|.clamp(min=thresh) * width/2` across specs
    /// - `sb_primary_spec=Some(s)`: use only spec row `s`
    /// - default: `max_s(|lA[s,d]|.clamp(min=thresh) * width/2 + margin[s] * weight)`
    ///
    /// `domain_bounds` are the active verification-direction output bounds:
    /// lower bounds when `verify_upper_bound=false`, upper bounds otherwise.
    /// With `thresholds`, they produce per-spec margins matching the baseline
    /// heuristic. `None` disables margin weighting.
    ///
    /// Reference: alpha-beta-CROWN `branching_heuristics.py:input_split_heuristic_sb`
    fn input_split_scores(
        &self,
        input_bounds: &BoundedTensor,
        linear_bounds: Option<&LinearBounds>,
        domain_bounds: Option<&[f32]>,
        thresholds: Option<&[f32]>,
    ) -> Vec<(usize, f32)> {
        let flat = input_bounds.flatten();
        let len = flat.len();
        if len == 0 {
            return Vec::new();
        }

        let width_only = || {
            let mut scores = Vec::with_capacity(len);
            for dim in 0..len {
                let width = flat.upper()[[dim]] - flat.lower()[[dim]];
                if width.is_finite() && width > 0.0 {
                    scores.push((dim, width));
                }
            }
            scores
        };

        let coeff_thresh = self.config.input_split_coeff_thresh;
        let touch_zero_score = self.config.input_split_touch_zero_score;
        let sb_sum = self.config.input_split_sb_sum;
        let sb_margin_weight = self.config.input_split_sb_margin_weight;
        let sb_primary_spec = self.config.input_split_sb_primary_spec;

        let coeffs = linear_bounds.map(|linear| {
            if self.config.verify_upper_bound {
                linear.upper_a()
            } else {
                linear.lower_a()
            }
        });
        if let Some(a) = coeffs {
            if a.ncols() != len {
                return width_only();
            }
        }

        let spec_margin = |spec_idx: usize| -> f32 {
            let Some(bounds) = domain_bounds else {
                return 0.0;
            };
            let Some(thresholds) = thresholds else {
                return 0.0;
            };
            if spec_idx >= bounds.len() || spec_idx >= thresholds.len() {
                return 0.0;
            }
            if self.config.verify_upper_bound {
                thresholds[spec_idx] - bounds[spec_idx]
            } else {
                bounds[spec_idx] - thresholds[spec_idx]
            }
        };

        let mut scores = Vec::with_capacity(len);
        let mut any_linear_signal = false;
        for dim in 0..len {
            let width = flat.upper()[[dim]] - flat.lower()[[dim]];
            let mut score = f32::NEG_INFINITY;
            if width.is_finite() && width > 0.0 {
                score = width;
                if let Some(a) = coeffs {
                    let num_specs = a.nrows();

                    if sb_sum {
                        // sb_sum mode: sum |A[s,d]|.clamp(min=thresh) * width/2 across specs.
                        // Margin weighting is NOT applied in sb_sum mode (matches baseline).
                        // Reference: branching_heuristics.py:79-81
                        let mut sum_score = 0.0f32;
                        for s in 0..num_specs {
                            let coeff = a[[s, dim]].abs();
                            if coeff > 0.0 {
                                any_linear_signal = true;
                            }
                            sum_score += coeff.max(coeff_thresh) * width * 0.5;
                        }
                        score = sum_score;
                    } else if let Some(primary) = sb_primary_spec {
                        // sb_primary_spec: score from a single spec row.
                        // Reference: branching_heuristics.py:91-93
                        if primary < num_specs {
                            let coeff = a[[primary, dim]].abs();
                            if coeff > 0.0 {
                                any_linear_signal = true;
                            }
                            score = coeff.max(coeff_thresh) * width * 0.5
                                + spec_margin(primary) * sb_margin_weight;
                        }
                        // If primary >= num_specs, fall through to width-only score.
                    } else {
                        // Default: max across specs of per-spec SB score.
                        // per_spec_score = |A[s,d]|.clamp(min=thresh) * width/2 + margin * weight
                        // Reference: branching_heuristics.py:84-95
                        let mut best_spec_score = f32::NEG_INFINITY;
                        for s in 0..num_specs {
                            let coeff = a[[s, dim]].abs();
                            if coeff > 0.0 {
                                any_linear_signal = true;
                            }
                            let spec_score = coeff.max(coeff_thresh) * width * 0.5
                                + spec_margin(s) * sb_margin_weight;
                            if spec_score > best_spec_score {
                                best_spec_score = spec_score;
                            }
                        }
                        score = best_spec_score;
                    }
                }
                if sb_sum
                    && touch_zero_score > 0.0
                    && (flat.lower()[[dim]] == 0.0 || flat.upper()[[dim]] == 0.0)
                {
                    score += width * touch_zero_score;
                }
            }
            if score.is_finite() && score > f32::NEG_INFINITY {
                scores.push((dim, score));
            }
        }

        if coeffs.is_some() && !any_linear_signal {
            // All coefficients are zero: fall back to width-only selection.
            return width_only();
        }

        scores
    }

    /// Create child domains by midpoint-splitting the top-scored input
    /// dimensions: each selected dim `d` splits every current domain into
    /// `input[d] in [l, mid]` and `input[d] in [mid, u]`.
    ///
    /// Returns `Unsplittable` when no input dimension admits a split (empty
    /// input, or every dimension is zero-width/non-finite/deselected). The
    /// parent box is then still unexplored, so the caller must mark the
    /// domain unresolved rather than treat it as fully branched.
    pub(crate) fn create_input_split_children(
        &self,
        network: &Network,
        input: &BoundedTensor,
        parent: &BabDomain,
        threshold: f32,
        deadline: Option<Instant>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<InputSplitChildren> {
        // Get the input bounds for this domain
        let domain_input = parent.input_bounds().unwrap_or(input);
        let flat = domain_input.flatten();
        if flat.is_empty() {
            trace!("Input split skipped: empty input bounds (len=0)");
            return Ok(InputSplitChildren::Unsplittable);
        }

        let linear_bounds = match network.propagate_crown_with_linear(domain_input) {
            Ok((_, linear)) => Some(linear),
            Err(_) => None,
        };
        // Thread the active verification-direction output bound through the SB
        // scorer so per-spec margin weighting matches alpha-beta-CROWN (#1074).
        let domain_bounds = [if self.config.verify_upper_bound {
            parent.upper_bound
        } else {
            parent.lower_bound
        }];
        let thresholds = [threshold];
        let split_dims = self.select_input_dimensions_sb(
            domain_input,
            linear_bounds.as_ref(),
            Some(&domain_bounds),
            Some(&thresholds),
        );
        if split_dims.is_empty() {
            trace!("Input split skipped: no valid input dimensions to split");
            return Ok(InputSplitChildren::Unsplittable);
        }

        let mut current_domains = vec![parent.clone()];
        for split_dim in split_dims {
            let mut next_domains = Vec::with_capacity(current_domains.len() * 2);
            for domain in current_domains {
                // A domain this dim cannot divide (e.g. clipping collapsed it
                // since scoring) is carried forward unsplit — dropping it would
                // leave part of the parent box uncovered.
                let domain_input = domain.input_bounds().unwrap_or(input);
                let flat = domain_input.flatten();
                if split_dim >= flat.len() {
                    trace!(
                        "Input split dim {} out of range (len={}); carrying domain unsplit",
                        split_dim,
                        flat.len()
                    );
                    next_domains.push(domain);
                    continue;
                }

                let l = flat.lower()[[split_dim]];
                let u = flat.upper()[[split_dim]];
                if !l.is_finite() || !u.is_finite() {
                    trace!(
                        "Input split dim {} non-finite (l={:?}, u={:?}); carrying domain unsplit",
                        split_dim,
                        l,
                        u
                    );
                    next_domains.push(domain);
                    continue;
                }

                if u <= l {
                    trace!(
                        "Input split dim {} non-positive width (l={:.6}, u={:.6}); carrying domain unsplit",
                        split_dim,
                        l,
                        u
                    );
                    next_domains.push(domain);
                    continue;
                }

                let mid = l + (u - l) / 2.0; // overflow-safe midpoint

                trace!(
                    "Input split on dim {}: [{:.4}, {:.4}] -> [{:.4}, {:.4}] and [{:.4}, {:.4}]",
                    split_dim,
                    l,
                    u,
                    l,
                    mid,
                    mid,
                    u
                );

                if let Some(left_child) = self.create_input_split_child(
                    network, input, &domain, split_dim, l, mid, threshold, deadline, engine,
                )? {
                    next_domains.push(left_child);
                }

                if let Some(right_child) = self.create_input_split_child(
                    network, input, &domain, split_dim, mid, u, threshold, deadline, engine,
                )? {
                    next_domains.push(right_child);
                }
            }

            current_domains = next_domains;
            if current_domains.is_empty() {
                break;
            }
        }

        Ok(InputSplitChildren::Split(current_domains))
    }

    /// Create a single child domain with tightened input bounds on one dimension.
    // Justification: Input split child needs network, original input, parent domain,
    // split dimension, bound value, side flag, and engine for bound recomputation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_input_split_child(
        &self,
        network: &Network,
        original_input: &BoundedTensor,
        parent: &BabDomain,
        split_dim: usize,
        new_lower: f32,
        new_upper: f32,
        threshold: f32,
        deadline: Option<Instant>,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Option<BabDomain>> {
        // Get the input bounds for this domain
        let domain_input = parent.input_bounds().unwrap_or(original_input);

        // Create new input bounds with tightened dimension
        let flat = domain_input.flatten();
        if flat.is_empty() {
            trace!("Input split child skipped: empty input bounds (len=0)");
            return Ok(None);
        }
        let shape = domain_input.lower().shape().to_vec();
        if split_dim >= flat.len() {
            trace!(
                "Input split child skipped: split_dim {} out of range (len={})",
                split_dim,
                flat.len()
            );
            return Ok(None);
        }
        if !new_lower.is_finite() || !new_upper.is_finite() {
            trace!(
                "Input split child skipped: non-finite bounds (dim={}, l={:?}, u={:?})",
                split_dim,
                new_lower,
                new_upper
            );
            return Ok(None);
        }
        if new_upper <= new_lower {
            trace!(
                "Input split child skipped: non-positive width (dim={}, l={:.6}, u={:.6})",
                split_dim,
                new_lower,
                new_upper
            );
            return Ok(None);
        }

        let mut new_lower_arr = flat.lower().clone();
        let mut new_upper_arr = flat.upper().clone();

        new_lower_arr[[split_dim]] = new_lower;
        new_upper_arr[[split_dim]] = new_upper;

        // Reshape back to original shape
        let new_lower_arr = new_lower_arr
            .into_shape_clone(ndarray::IxDyn(&shape))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape lower: {}", e)))?;
        let new_upper_arr = new_upper_arr
            .into_shape_clone(ndarray::IxDyn(&shape))
            .map_err(|e| ny_core::NyError::InvalidSpec(format!("reshape upper: {}", e)))?;

        let mut new_input_bounds = BoundedTensor::new(new_lower_arr, new_upper_arr)?;

        // Apply input-domain clipping if enabled (Clip-and-Verify)
        // Dispatches to relaxed or complete clipping based on config.
        if self.config.enable_relaxed_clip {
            use crate::beta_crown::config::InputClipType;
            let clip_outcome = match self.config.input_clip_type {
                InputClipType::Relaxed => self.apply_relaxed_clipping(
                    network,
                    new_input_bounds,
                    &shape,
                    threshold,
                    engine,
                )?,
                InputClipType::Complete => self.apply_complete_clipping(
                    network,
                    new_input_bounds,
                    &shape,
                    threshold,
                    engine,
                )?,
            };
            if clip_outcome.verified {
                trace!("Input-split child verified by clipping, skipping");
                return Ok(None);
            }
            new_input_bounds = clip_outcome.bounds;
        }

        // #tll-nested-collect-par: permit nested faer (Rayon) parallelism for the
        // per-domain CROWN-IBP intermediate collection + output CROWN below. In
        // the shallow phase of the input-split tree only 1–few domains run
        // concurrently, so the wide TLL f64 `A·W` / `|A|·|W|` backward GEMMs
        // would otherwise pin ONE Rayon worker at a time while the rest of the
        // machine sits idle (profiled: warmup collection 3.2s/wide-layer
        // multi-threaded vs 18.7s/wide-layer single-threaded on a worker). The
        // guard's work-stealing scope self-balances: it fills idle cores early
        // and stays effectively sequential once the batch saturates the machine
        // (deep phase), so the saturated regime is unaffected. Sound: only the
        // GEMM summation order changes; the certified γ_n·S envelope is
        // order-independent. Held to end of function to cover the output-CROWN
        // pass too. `NY_INPUT_SPLIT_NESTED_PAR=0` disables (byte-identical A/B).
        let _nested_par_guard = crate::faer_parallelism::NestedFaerParGuard::new();

        // Recompute all layer bounds with new input bounds
        let budget_exceeded = crown_ibp_budget_exceeded(&self.config, network);
        let use_crown_ibp_layer_bounds = self.config.use_crown_ibp && !budget_exceeded;

        let mut new_layer_bounds = if self.config.use_crown_ibp {
            if use_crown_ibp_layer_bounds {
                network.collect_crown_ibp_bounds_with_engine_and_deadline(
                    &new_input_bounds,
                    engine,
                    deadline,
                )?
            } else {
                network.collect_ibp_bounds_with_deadline(&new_input_bounds, deadline)?
            }
        } else {
            network.collect_ibp_bounds(&new_input_bounds)?
        };

        // Apply intermediate constrained tightening for complete clipping.
        // When clip_type == Complete and we have spec constraints, tighten
        // hidden-layer bounds using the Lagrangian dual LP solver.
        //
        // Reference: auto_LiRPA/concretize_bounds.py:concretize_bounds (two-pass)
        // Part of #3552 Packet 3
        if self.config.enable_relaxed_clip {
            use crate::beta_crown::config::InputClipType;
            if self.config.input_clip_type == InputClipType::Complete {
                if let Err(e) = self.apply_intermediate_complete_clipping(
                    network,
                    &new_input_bounds,
                    &mut new_layer_bounds,
                    threshold,
                    engine,
                ) {
                    trace!("complete_clip_intermediate: skipping due to error: {}", e);
                }
            }
        }

        // Compute output bounds with CROWN
        // Thread deadline into alpha config so per-domain α-CROWN bails early
        // when the BaB timeout budget is exhausted (#2724).
        let output_bounds = if self.config.use_alpha_crown {
            let mut alpha_config = self.config.alpha_config.clone();
            alpha_config.deadline = deadline;
            network.propagate_alpha_crown_with_config_and_engine(
                &new_input_bounds,
                &alpha_config,
                engine,
            )?
        } else if use_crown_ibp_layer_bounds {
            network.propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
                &new_input_bounds,
                &new_layer_bounds,
                engine,
                deadline,
                self.config.crown_backward_layers,
            )?
        } else {
            // Reuse the just-computed IBP bounds so split domains keep the GPU
            // fast-path without paying for a second internal IBP forward pass.
            network.propagate_crown_with_precomputed_ibp_and_limits(
                &new_input_bounds,
                new_layer_bounds.clone(),
                engine,
                deadline,
                self.config.crown_backward_layers,
            )?
        };

        let mut new_lower_bound = output_bounds.lower_scalar();
        let new_upper_bound = output_bounds.upper_scalar();

        // JOINT-MARGIN closer (same-LHS conjunctive max-diff, acasxu prop_2/3/4).
        // CROWN's scalar bound above routed the MaxPool *lower* relaxation through
        // a single conjunct (argmax l_j), which cannot certify a box where
        // different conjuncts dominate different sub-regions — the divergence
        // root cause (diag c7126554). When a closer is attached, recover a tighter
        // JOINT lower bound over ALL conjuncts and raise `new_lower_bound` to it.
        // Sound: the closer returns a certified lower bound on the same max-diff
        // objective, so `max(crown_lb, joint_lb)` is still a valid lower bound
        // (see `joint_margin::JointMarginCloser`). Only run on the unverified
        // lower-bound direction — a verified domain needs no help, and the closer
        // certifies a *lower* bound (useless for upper-bound verification).
        if !self.config.verify_upper_bound && new_lower_bound.is_finite() {
            if let Some(closer) = self.joint_margin_closer() {
                if !self
                    .config
                    .domain_is_verified(new_lower_bound, new_upper_bound, threshold)
                {
                    if let Some(joint_lb) = closer.certified_joint_lower_bound(
                        &new_input_bounds,
                        engine,
                        new_lower_bound,
                        threshold,
                    ) {
                        if joint_lb.is_finite() && joint_lb > new_lower_bound {
                            new_lower_bound = joint_lb;
                        }
                    }
                }
            }
        }

        let history = parent.history.clone();
        let beta_state = BetaState::from_history(&history)?;
        let new_layer_bounds: Vec<Arc<BoundedTensor>> =
            new_layer_bounds.into_iter().map(Arc::new).collect();
        let domain_alpha_state = if self.config.use_alpha_crown {
            DomainAlphaState::from_layer_bounds_and_constraints(
                network,
                &new_layer_bounds,
                &history,
            )
        } else {
            DomainAlphaState::empty()
        };

        // Use validated child constructor to enforce NaN rejection (#3125).
        // Priority defaults to lower_bound; BaB loop overwrites via
        // set_priority()/violation_priority() before queue insertion (#2682).
        Ok(Some(BabDomain::child(
            history,
            new_lower_bound,
            new_upper_bound,
            new_layer_bounds,
            parent.alpha_state.clone(),
            domain_alpha_state,
            beta_state,
            Some(Arc::new(new_input_bounds)),
            parent.input_split_count + 1,
            // Input splits change the input space, so intermediate bounds are reset
            IntermediateLinearBounds::empty(),
        )?))
    }

    /// Apply intermediate constrained tightening for complete clipping.
    ///
    /// Gets CROWN linear bounds at the output, builds spec constraints, and
    /// tightens intermediate layer bounds using the Lagrangian dual LP solver.
    ///
    /// Reference: auto_LiRPA/concretize_bounds.py:concretize_bounds
    /// Part of #3552 Packet 3
    fn apply_intermediate_complete_clipping(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        layer_bounds: &mut [BoundedTensor],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<()> {
        // Get CROWN linear bounds at the output for constraint construction
        let (_output_bounds, mut linear_bounds) =
            network.propagate_crown_with_linear_and_engine(input_bounds, engine)?;
        Self::discharge_coeff_err_for_clip(&mut linear_bounds, input_bounds);

        // Build spec constraints from output linear bounds
        let spec_constraints =
            super::complete_clip_intermediate::build_spec_constraints_for_intermediate(
                &linear_bounds,
                threshold,
                self.config.verify_upper_bound,
            )?;

        // Preprocess: filter infeasible/covered, compute d offsets
        let x_flat = input_bounds.flatten();
        let x_l = ndarray::Array1::from_vec(x_flat.lower().iter().copied().collect());
        let x_u = ndarray::Array1::from_vec(x_flat.upper().iter().copied().collect());
        let preprocessed =
            crate::clip_interm_domain::sort_out_constraints(&spec_constraints, &x_l, &x_u)?;

        // Tighten intermediate bounds
        super::complete_clip_intermediate::tighten_intermediate_with_spec_constraints(
            network,
            input_bounds,
            layer_bounds,
            &preprocessed,
            self.config.clip_neuron_selection_ratio,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_crown::config::BetaCrownConfig;
    use ndarray::array;

    #[test]
    fn test_select_input_dimension_with_linear_bounds_prefers_sensitive_dim() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input =
            BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![1.0, 2.0].into_dyn()).unwrap();

        let linear_bounds = LinearBounds {
            lower_a: array![[10.0, 0.5]],
            lower_b: array![0.0],
            upper_a: array![[10.0, 0.5]],
            upper_b: array![0.0],
            lower_a_err: None,
            upper_a_err: None,
        };

        let dim_with_linear =
            verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        let dim_with_width = verifier.select_input_dimension_sb(&input, None, None, None);

        assert_eq!(dim_with_linear, 0);
        assert_eq!(dim_with_width, 1);
    }

    #[test]
    fn test_select_input_dimension_falls_back_to_width_when_linear_is_zero() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input =
            BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![1.0, 3.0].into_dyn()).unwrap();

        let linear_bounds = LinearBounds {
            lower_a: array![[0.0, 0.0]],
            lower_b: array![0.0],
            upper_a: array![[0.0, 0.0]],
            upper_b: array![0.0],
            lower_a_err: None,
            upper_a_err: None,
        };

        let dim = verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        assert_eq!(dim, 1);
    }

    #[test]
    fn test_select_input_dimension_uses_upper_bounds_for_upper_verification() {
        let config = BetaCrownConfig {
            verify_upper_bound: true,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let input =
            BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

        let linear_bounds = LinearBounds {
            lower_a: array![[10.0, 1.0]],
            lower_b: array![0.0],
            upper_a: array![[1.0, 10.0]],
            upper_b: array![0.0],
            lower_a_err: None,
            upper_a_err: None,
        };

        let dim = verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        assert_eq!(dim, 1);
    }

    #[test]
    fn test_select_input_dimension_uses_lower_bounds_for_lower_verification() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input =
            BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

        let linear_bounds = LinearBounds {
            lower_a: array![[1.0, 10.0]],
            lower_b: array![0.0],
            upper_a: array![[10.0, 1.0]],
            upper_b: array![0.0],
            lower_a_err: None,
            upper_a_err: None,
        };

        let dim = verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        assert_eq!(dim, 1);
    }

    #[test]
    fn test_select_input_dimension_falls_back_on_shape_mismatch() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input = BoundedTensor::new(
            array![0.0, 0.0, 0.0].into_dyn(),
            array![1.0, 3.0, 2.0].into_dyn(),
        )
        .unwrap();

        let linear_bounds = LinearBounds {
            lower_a: array![[1.0, 2.0]],
            lower_b: array![0.0],
            upper_a: array![[1.0, 2.0]],
            upper_b: array![0.0],
            lower_a_err: None,
            upper_a_err: None,
        };

        let dim = verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        assert_eq!(dim, 1);
    }

    #[test]
    fn test_select_input_dimension_handles_empty_input() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input = BoundedTensor::new(
            ndarray::Array1::<f32>::from_vec(Vec::new()).into_dyn(),
            ndarray::Array1::<f32>::from_vec(Vec::new()).into_dyn(),
        )
        .unwrap();

        // Empty inputs default to the first dimension.
        let dim = verifier.select_input_dimension_sb(&input, None, None, None);
        assert_eq!(dim, 0);
    }

    #[test]
    fn test_select_input_dimensions_topk_by_width() {
        let config = BetaCrownConfig {
            input_split_depth: 2,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let input = BoundedTensor::new(
            array![0.0, 0.0, 0.0].into_dyn(),
            array![1.0, 3.0, 2.0].into_dyn(),
        )
        .unwrap();

        let dims = verifier.select_input_dimensions_sb(&input, None, None, None);
        assert_eq!(dims, vec![1, 2]);
    }

    #[test]
    #[ntest::timeout(10000)]
    fn test_create_input_split_children_multi_dim() {
        let config = BetaCrownConfig {
            input_split_depth: 2,
            use_alpha_crown: false,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let input =
            BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![2.0, 4.0].into_dyn()).unwrap();
        let network = Network::new();
        let parent = BabDomain::root_with_input(Vec::new(), 0.0, 0.0, &input).unwrap();

        let children = match verifier
            .create_input_split_children(&network, &input, &parent, 0.0, None, None)
            .unwrap()
        {
            InputSplitChildren::Split(children) => children,
            InputSplitChildren::Unsplittable => panic!("positive-width box must be splittable"),
        };
        assert_eq!(children.len(), 4);
        for child in &children {
            assert_eq!(child.input_split_count, 2);
        }
    }

    #[test]
    #[ntest::timeout(10000)]
    fn test_create_input_split_children_empty_input_is_unsplittable() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input = BoundedTensor::new(
            ndarray::Array1::<f32>::from_vec(Vec::new()).into_dyn(),
            ndarray::Array1::<f32>::from_vec(Vec::new()).into_dyn(),
        )
        .unwrap();

        let network = Network::new();
        let parent = BabDomain::root_with_input(Vec::new(), 0.0, 0.0, &input).unwrap();

        let outcome = verifier
            .create_input_split_children(&network, &input, &parent, 0.0, None, None)
            .unwrap();
        assert!(
            matches!(outcome, InputSplitChildren::Unsplittable),
            "empty input must report Unsplittable, not a clean empty branch"
        );
    }

    /// A fully-degenerate (point) box has no positive-width dimension, so it
    /// must report `Unsplittable`. An empty `Split` here would let the BaB
    /// loop drop the unexplored box and claim Verified.
    #[test]
    #[ntest::timeout(10000)]
    fn test_create_input_split_children_point_box_is_unsplittable() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input =
            BoundedTensor::new(array![0.0, 1.0].into_dyn(), array![0.0, 1.0].into_dyn()).unwrap();

        let network = Network::new();
        let parent = BabDomain::root_with_input(Vec::new(), 0.0, 0.0, &input).unwrap();

        let outcome = verifier
            .create_input_split_children(&network, &input, &parent, 0.0, None, None)
            .unwrap();
        assert!(
            matches!(outcome, InputSplitChildren::Unsplittable),
            "point box must report Unsplittable, not a clean empty branch"
        );
    }

    #[test]
    #[ntest::timeout(10000)]
    fn test_create_input_split_child_skips_empty_input() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input = BoundedTensor::new(
            ndarray::Array1::<f32>::from_vec(Vec::new()).into_dyn(),
            ndarray::Array1::<f32>::from_vec(Vec::new()).into_dyn(),
        )
        .unwrap();

        let network = Network::new();
        let parent = BabDomain::root_with_input(Vec::new(), 0.0, 0.0, &input).unwrap();

        let child = verifier
            .create_input_split_child(&network, &input, &parent, 0, 0.0, 1.0, 0.0, None, None)
            .unwrap();
        assert!(child.is_none());
    }

    #[test]
    #[ntest::timeout(10000)]
    fn test_create_input_split_child_skips_out_of_range_dim() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input = BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap();
        let network = Network::new();
        let parent = BabDomain::root_with_input(Vec::new(), 0.0, 0.0, &input).unwrap();

        let child = verifier
            .create_input_split_child(&network, &input, &parent, 3, 0.0, 1.0, 0.0, None, None)
            .unwrap();
        assert!(child.is_none());
    }

    #[test]
    #[ntest::timeout(10000)]
    fn test_create_input_split_child_skips_non_finite_bounds() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input = BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap();
        let network = Network::new();
        let parent = BabDomain::root_with_input(Vec::new(), 0.0, 0.0, &input).unwrap();

        let child = verifier
            .create_input_split_child(&network, &input, &parent, 0, f32::NAN, 1.0, 0.0, None, None)
            .unwrap();
        assert!(child.is_none());
    }

    #[test]
    #[ntest::timeout(10000)]
    fn test_create_input_split_child_skips_invalid_width() {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let input = BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap();
        let network = Network::new();
        let parent = BabDomain::root_with_input(Vec::new(), 0.0, 0.0, &input).unwrap();

        let child = verifier
            .create_input_split_child(&network, &input, &parent, 0, 1.0, 0.5, 0.0, None, None)
            .unwrap();
        assert!(child.is_none());
    }

    /// sb_sum mode: sum across specs values dimensions that affect many specs,
    /// even if no single spec has a dominant coefficient.
    ///
    /// Spec 0: large coeff on dim 0, small on dim 1.
    /// Spec 1: small coeff on dim 0, large on dim 1.
    /// Default mode (max): both dims get the same max coeff (10.0), so dim 1
    /// wins on width (2.0 vs 1.0).
    /// sb_sum mode: both dims get sum = (10+1)*w/2. dim 0 = 5.5, dim 1 = 11.0.
    /// dim 1 still wins because width dominates, but the ranking mechanism is different.
    ///
    /// Reference: alpha-beta-CROWN branching_heuristics.py:79-81
    #[test]
    fn test_sb_sum_sums_across_specs() {
        let config = BetaCrownConfig {
            input_split_sb_sum: true,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        // dim 0: width=1, dim 1: width=1 (equal widths)
        let input =
            BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

        // Spec 0: coeff [10, 1], Spec 1: coeff [1, 10]
        // Default max mode: dim 0 score = 10 * 0.5 = 5.0, dim 1 score = 10 * 0.5 = 5.0 (tie)
        // sb_sum mode: dim 0 score = (10 + 1) * 0.5 = 5.5, dim 1 score = (1 + 10) * 0.5 = 5.5
        let linear_bounds = LinearBounds {
            lower_a: array![[10.0, 1.0], [1.0, 10.0]],
            lower_b: array![0.0, 0.0],
            upper_a: array![[10.0, 1.0], [1.0, 10.0]],
            upper_b: array![0.0, 0.0],
            lower_a_err: None,
            upper_a_err: None,
        };
        // Both dims score 5.5, tie broken by dimension index (dim 0 wins).
        let dim = verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        assert_eq!(dim, 0);

        // With asymmetric coefficients: spec 0 [10, 2], spec 1 [1, 10]
        // sb_sum: dim 0 = (10+1)*0.5=5.5, dim 1 = (2+10)*0.5=6.0 → dim 1 wins
        let linear_asymmetric = LinearBounds {
            lower_a: array![[10.0, 2.0], [1.0, 10.0]],
            lower_b: array![0.0, 0.0],
            upper_a: array![[10.0, 2.0], [1.0, 10.0]],
            upper_b: array![0.0, 0.0],
            lower_a_err: None,
            upper_a_err: None,
        };
        let dim = verifier.select_input_dimension_sb(&input, Some(&linear_asymmetric), None, None);
        assert_eq!(dim, 1);
    }

    /// sb_primary_spec selects a single spec row for scoring.
    ///
    /// Reference: alpha-beta-CROWN branching_heuristics.py:91-93
    #[test]
    fn test_sb_primary_spec_uses_single_row() {
        // Use spec 1 only.
        let config = BetaCrownConfig {
            input_split_sb_primary_spec: Some(1),
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let input =
            BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

        // Spec 0: large on dim 0. Spec 1: large on dim 1.
        // Default would pick max across specs → tie.
        // Primary spec 1 → dim 1 wins (coeff 10 vs 1).
        let linear_bounds = LinearBounds {
            lower_a: array![[10.0, 1.0], [1.0, 10.0]],
            lower_b: array![0.0, 0.0],
            upper_a: array![[10.0, 1.0], [1.0, 10.0]],
            upper_b: array![0.0, 0.0],
            lower_a_err: None,
            upper_a_err: None,
        };
        let dim = verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        assert_eq!(dim, 1);
    }

    /// Per-spec SB margins can change the selected split dimension by favoring
    /// nearly-verified specs over far-from-verified ones.
    ///
    /// Reference: alpha-beta-CROWN branching_heuristics.py:84-89
    #[test]
    fn test_sb_margin_weight_prefers_nearly_verified_spec_dimension() {
        let config = BetaCrownConfig {
            input_split_sb_margin_weight: 1.0,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let input =
            BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

        let linear_bounds = LinearBounds {
            lower_a: array![[2.0, 1.0], [1.0, 10.0]],
            lower_b: array![0.0, 0.0],
            upper_a: array![[2.0, 1.0], [1.0, 10.0]],
            upper_b: array![0.0, 0.0],
            lower_a_err: None,
            upper_a_err: None,
        };
        let domain_bounds = [0.9_f32, -5.0_f32];
        let thresholds = [1.0_f32, 1.0_f32];

        // Without per-spec margins, the far spec's strong coefficient makes dim 1 win.
        // With per-spec margins, spec 0 is nearly verified while spec 1 is far away, so
        // dim 0 becomes the better split.
        let dim_no_margin =
            verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        let dim_with_margin = verifier.select_input_dimension_sb(
            &input,
            Some(&linear_bounds),
            Some(&domain_bounds),
            Some(&thresholds),
        );
        assert_eq!(dim_no_margin, 1);
        assert_eq!(dim_with_margin, 0);
    }

    #[test]
    fn test_sb_margin_weight_respects_upper_bound_verification_mode() {
        let config = BetaCrownConfig {
            verify_upper_bound: true,
            input_split_sb_margin_weight: 1.0,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let input =
            BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

        let linear_bounds = LinearBounds {
            lower_a: array![[0.0, 0.0], [0.0, 0.0]],
            lower_b: array![0.0, 0.0],
            upper_a: array![[2.0, 1.0], [1.0, 10.0]],
            upper_b: array![0.0, 0.0],
            lower_a_err: None,
            upper_a_err: None,
        };
        let upper_bounds = [1.1_f32, 7.0_f32];
        let thresholds = [1.0_f32, 1.0_f32];

        let dim_no_margin =
            verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        let dim_with_margin = verifier.select_input_dimension_sb(
            &input,
            Some(&linear_bounds),
            Some(&upper_bounds),
            Some(&thresholds),
        );
        assert_eq!(dim_no_margin, 1);
        assert_eq!(dim_with_margin, 0);
    }

    #[test]
    fn test_sb_margin_weight_default_matches_reference() {
        assert_eq!(BetaCrownConfig::default().input_split_sb_margin_weight, 1.0);
    }

    #[test]
    fn test_touch_zero_score_is_ignored_in_default_sb_mode() {
        let config = BetaCrownConfig {
            input_split_touch_zero_score: 10.0,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let input =
            BoundedTensor::new(array![0.0, 0.1].into_dyn(), array![1.0, 1.1].into_dyn()).unwrap();

        let linear_bounds = LinearBounds {
            lower_a: array![[1.0, 2.0]],
            lower_b: array![0.0],
            upper_a: array![[1.0, 2.0]],
            upper_b: array![0.0],
            lower_a_err: None,
            upper_a_err: None,
        };

        let dim = verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        assert_eq!(dim, 1);
    }

    #[test]
    fn test_touch_zero_score_applies_in_sb_sum_mode() {
        let config = BetaCrownConfig {
            input_split_sb_sum: true,
            input_split_touch_zero_score: 10.0,
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let input =
            BoundedTensor::new(array![0.0, 0.1].into_dyn(), array![1.0, 1.1].into_dyn()).unwrap();

        let linear_bounds = LinearBounds {
            lower_a: array![[1.0, 2.0]],
            lower_b: array![0.0],
            upper_a: array![[1.0, 2.0]],
            upper_b: array![0.0],
            lower_a_err: None,
            upper_a_err: None,
        };

        let dim = verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        assert_eq!(dim, 0);
    }

    /// margin=None is backward compatible — equivalent to margin_weight=0.
    #[test]
    fn test_margin_none_is_backward_compatible() {
        let config = BetaCrownConfig {
            input_split_sb_margin_weight: 5.0, // nonzero weight
            ..Default::default()
        };
        let verifier = BetaCrownVerifier::new(config);
        let input =
            BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![1.0, 2.0].into_dyn()).unwrap();

        let linear_bounds = LinearBounds {
            lower_a: array![[10.0, 0.5]],
            lower_b: array![0.0],
            upper_a: array![[10.0, 0.5]],
            upper_b: array![0.0],
            lower_a_err: None,
            upper_a_err: None,
        };

        // Missing domain bounds/thresholds disable margin weighting.
        let dim = verifier.select_input_dimension_sb(&input, Some(&linear_bounds), None, None);
        assert_eq!(dim, 0);
    }
}
