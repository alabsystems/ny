// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constrained intermediate tightening for complete clipping.
//!
//! After input-space complete clipping (output-level), applies spec-derived
//! constraints to tighten intermediate layer bounds via Lagrangian dual LP.
//!
//! This implements the hidden-layer component of `clip_type: complete` from
//! alpha-beta-CROWN. The spec constraints (from CROWN output linear bounds)
//! constrain the optimization when computing intermediate neuron bounds.
//!
//! ## Two-pass approach (matching alpha-beta-CROWN)
//!
//! 1. **Naive pass**: Compute unconstrained intermediate bounds (already done
//!    by CROWN-IBP collection before this function is called).
//! 2. **Constrained pass**: For selected unstable neurons, run CROWN backward
//!    from each layer to the input, then apply `tighten_with_constraints` with
//!    spec constraints. Merge tightened values into the existing bounds.
//!
//! ## References
//!
//! - `auto_LiRPA/concretize_bounds.py:concretize_bounds` — two-pass approach
//! - `auto_LiRPA/concretize_func.py:constraints_solving` — LP solver call
//! - `designs/2026-03-17-issue-3552-complete-clipping-semantics-execution-packet.md` Packet 3

use ndarray::{Array1, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use tracing::{debug, trace};

use crate::cmp_utils::nan_last_descending_cmp;

use crate::clip_interm_domain::{
    merge_bounds, sub_f32_down, tighten_with_constraints, PreprocessedConstraints, SplitConstraints,
};
use crate::layers::common::BoundPropagation;
use crate::{LinearBounds, Network};

use super::BetaCrownVerifier;

/// Build spec-derived constraints from CROWN output linear bounds.
///
/// Converts CROWN linear bounds `lA @ x + lbias` and verification threshold
/// into standard-form constraints `A @ x + b <= 0`.
///
/// Reference: `auto_LiRPA/concretize_func.py:construct_constraints` (line 50)
pub(in crate::beta_crown::engine) fn build_spec_constraints_for_intermediate(
    linear_bounds: &LinearBounds,
    threshold: f32,
    verify_upper: bool,
) -> Result<SplitConstraints> {
    let (coeffs, biases, threshold_value): (Array2<f32>, Array1<f32>, f32) = if verify_upper {
        (
            linear_bounds.upper_a().mapv(|v| -v),
            linear_bounds.upper_b().mapv(|v| -v),
            -threshold,
        )
    } else {
        (
            linear_bounds.lower_a().clone(),
            linear_bounds.lower_b().clone(),
            threshold,
        )
    };

    let n_spec = coeffs.nrows();

    // Standard form: lA @ x + (lbias - threshold) <= 0
    let mut b_vector = Array1::zeros(n_spec);
    for (result, &bias) in b_vector.iter_mut().zip(&biases) {
        *result = sub_f32_down(bias, threshold_value).ok_or_else(|| {
            ny_core::NyError::InvalidSpec(format!(
                "intermediate complete-clip constraints require finite bias and threshold, got {bias} and {threshold_value}"
            ))
        })?;
    }

    Ok(SplitConstraints {
        a_matrix: coeffs,
        b_vector,
        num_constraints: n_spec,
    })
}

/// Tighten intermediate layer bounds using spec-derived constraints.
///
/// For each layer with unstable neurons:
/// 1. Select neurons using `clip_neuron_selection_ratio`
/// 2. Run CROWN backward from that layer to the input (per-neuron linear bounds)
/// 3. Apply `tighten_with_constraints` with spec constraints
/// 4. Merge tightened bounds back
///
/// Returns: `Ok(true)` if any bounds were tightened, `Ok(false)` if not.
///
/// Reference: `auto_LiRPA/concretize_bounds.py:concretize_bounds` (two-pass)
pub(in crate::beta_crown::engine) fn tighten_intermediate_with_spec_constraints(
    network: &Network,
    input_bounds: &BoundedTensor,
    layer_bounds: &mut [BoundedTensor],
    preprocessed: &PreprocessedConstraints,
    selection_ratio: f32,
) -> Result<bool> {
    // Skip if no active constraints
    if preprocessed.a_active.nrows() == 0 {
        trace!("complete_clip_intermediate: no active constraints, skipping");
        return Ok(false);
    }

    let x_flat = input_bounds.flatten();
    let x_l: Array1<f32> = Array1::from_vec(x_flat.lower().iter().copied().collect());
    let x_u: Array1<f32> = Array1::from_vec(x_flat.upper().iter().copied().collect());

    let mut any_tightened = false;

    for layer_idx in 0..layer_bounds.len() {
        let bt = &layer_bounds[layer_idx];
        let flat = bt.flatten();
        let lower: Array1<f32> = Array1::from_vec(flat.lower().iter().copied().collect());
        let upper: Array1<f32> = Array1::from_vec(flat.upper().iter().copied().collect());
        // Count unstable neurons (l < 0 and u > 0)
        let n_unstable = lower
            .iter()
            .zip(upper.iter())
            .filter(|(&l, &u)| l < 0.0 && u > 0.0)
            .count();

        if n_unstable == 0 {
            continue;
        }

        // Determine budget from selection_ratio
        // Reference: designs/2026-03-17-issue-3552-complete-clipping-semantics-execution-packet.md
        //   ratio < 0: tighten all unstable neurons
        //   ratio in [0, 1]: ceil(unstable_count * ratio), clamped to >= 1
        let budget = if selection_ratio < 0.0 {
            n_unstable
        } else {
            let k = (n_unstable as f32 * selection_ratio).ceil() as usize;
            k.max(1).min(n_unstable)
        };

        // Select neurons by uncertainty (gap = upper - lower, descending)
        let selected = select_neurons_by_uncertainty(&lower, &upper, budget);
        if selected.is_empty() {
            continue;
        }

        // Run CROWN backward from this layer to the input for selected neurons
        let mut lin_bounds = match crown_backward_for_neurons(
            network,
            input_bounds,
            layer_bounds,
            layer_idx,
            &selected,
        ) {
            Ok(lb) => lb,
            Err(e) => {
                debug!(
                    "complete_clip_intermediate: backward failed at layer {}: {}",
                    layer_idx, e
                );
                continue;
            }
        };
        // The backward pass may attach a certified coefficient-error envelope;
        // fold it into the bias over the input box before the raw coefficients
        // drive the constrained tightening below, otherwise the merged bounds
        // could be tighter than the true coefficients entail.
        BetaCrownVerifier::discharge_coeff_err_for_clip(&mut lin_bounds, input_bounds);

        // Apply constrained tightening using spec constraints
        let (tightened_lower, tightened_upper) = match tighten_with_constraints(
            preprocessed,
            lin_bounds.lower_a(),
            lin_bounds.lower_b(),
            lin_bounds.upper_a(),
            lin_bounds.upper_b(),
            &x_l,
            &x_u,
        ) {
            Ok(result) => result,
            Err(e) => {
                debug!(
                    "complete_clip_intermediate: tighten failed at layer {}: {}",
                    layer_idx, e
                );
                continue;
            }
        };

        // Merge tightened bounds back into original
        let (merged_lower, merged_upper) = merge_bounds(
            &lower,
            &upper,
            &tightened_lower,
            &tightened_upper,
            &selected,
        );

        // Check if anything actually changed
        let changed = selected
            .iter()
            .any(|&i| merged_lower[i] > lower[i] || merged_upper[i] < upper[i]);

        if changed {
            let shape = bt.lower().shape().to_vec();
            let new_lower = merged_lower
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| {
                    ny_core::NyError::InternalError(format!(
                        "complete_clip_intermediate: reshape lower at layer {}: {}",
                        layer_idx, e
                    ))
                })?;
            let new_upper = merged_upper
                .into_shape_clone(ndarray::IxDyn(&shape))
                .map_err(|e| {
                    ny_core::NyError::InternalError(format!(
                        "complete_clip_intermediate: reshape upper at layer {}: {}",
                        layer_idx, e
                    ))
                })?;
            layer_bounds[layer_idx] = BoundedTensor::new(new_lower, new_upper)?;
            any_tightened = true;
            trace!(
                "complete_clip_intermediate: tightened {} neurons at layer {}",
                selected.len(),
                layer_idx
            );
        }
    }

    Ok(any_tightened)
}

/// Select unstable neurons by uncertainty (gap = upper - lower).
///
/// Differs from `select_objective_neurons` in `clip_interm_domain/objectives.rs`:
/// - Uses pure uncertainty (gap) instead of kFSB intercept * coeff_mag
/// - Matches the alpha-beta-CROWN `clip_neuron_selection_type: ratio` contract
///
/// Reference: `auto_LiRPA/concretize_bounds.py:concretize_bounds` (lines 115-136)
fn select_neurons_by_uncertainty(
    lower: &Array1<f32>,
    upper: &Array1<f32>,
    budget: usize,
) -> Vec<usize> {
    let mut unstable: Vec<(usize, f32)> = lower
        .iter()
        .zip(upper.iter())
        .enumerate()
        .filter_map(|(i, (&l, &u))| {
            if l < 0.0 && u > 0.0 {
                let gap = u - l;
                if gap.is_finite() {
                    Some((i, gap))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    // Sort descending by gap (largest uncertainty first).
    // NaN-safe: nan_last_descending_cmp sorts NaN last (#4288).
    // Defense-in-depth: the is_finite() filter above excludes NaN gaps,
    // but the safe comparator prevents silent corruption if the filter changes.
    unstable.sort_by(|a, b| nan_last_descending_cmp(&a.1, &b.1));

    unstable.iter().take(budget).map(|(idx, _)| *idx).collect()
}

/// Run CROWN backward from a specific layer to the input for selected neurons.
///
/// Creates an identity-like seed at the target layer (with rows only for
/// selected neurons) and propagates backward through all preceding layers
/// using existing pre-activation bounds.
///
/// Returns `LinearBounds` of shape `(n_selected, x_dim)` expressing each
/// selected neuron as a linear function of the network input.
///
/// This is THE input-relative provenance primitive for the sequential engine:
/// `layer_bounds[target_layer]` is the box of `h_target_layer`, and the returned
/// rows bound that same quantity as an affine function of the network input.
/// `domain::clip_provenance` reuses it (over `Arc`-shared boxes, hence the
/// `Borrow` bound) rather than growing a second copy of this composition.
///
/// Reference: This is equivalent to `propagate_crown_partial_with_engine` but
/// starting from an intermediate identity rather than the output identity.
pub(in crate::beta_crown::engine) fn crown_backward_for_neurons<
    B: std::borrow::Borrow<BoundedTensor>,
>(
    network: &Network,
    input: &BoundedTensor,
    layer_bounds: &[B],
    target_layer: usize,
    selected_neurons: &[usize],
) -> Result<LinearBounds> {
    let n_selected = selected_neurons.len();
    if target_layer >= layer_bounds.len() || target_layer >= network.layers.len() {
        return Err(ny_core::NyError::InternalError(format!(
            "crown_backward_for_neurons: target layer {} out of range (bounds={}, layers={})",
            target_layer,
            layer_bounds.len(),
            network.layers.len()
        )));
    }
    let target_dim = layer_bounds[target_layer].borrow().len();

    // Create seed: each row selects one neuron at the target layer
    let mut lower_a = Array2::zeros((n_selected, target_dim));
    let mut upper_a = Array2::zeros((n_selected, target_dim));
    for (row, &col) in selected_neurons.iter().enumerate() {
        if col >= target_dim {
            return Err(ny_core::NyError::InternalError(format!(
                "crown_backward_for_neurons: neuron {} out of range (dim={})",
                col, target_dim
            )));
        }
        lower_a[(row, col)] = 1.0;
        upper_a[(row, col)] = 1.0;
    }

    let mut lin_bounds = LinearBounds::new(
        lower_a,
        Array1::zeros(n_selected),
        upper_a,
        Array1::zeros(n_selected),
    )?;

    // Propagate backward through layers target_layer down to 0
    for layer_idx in (0..=target_layer).rev() {
        let pre_activation = if layer_idx == 0 {
            input
        } else if layer_idx - 1 < layer_bounds.len() {
            layer_bounds[layer_idx - 1].borrow()
        } else {
            return Err(ny_core::NyError::InternalError(format!(
                "crown_backward_for_neurons: missing bounds for layer {}",
                layer_idx - 1
            )));
        };

        lin_bounds = network.layers[layer_idx]
            .propagate_crown_backward(&lin_bounds, Some(pre_activation))?;
    }

    Ok(lin_bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_select_neurons_by_uncertainty_picks_widest_unstable() {
        let lower = array![-1.0, -2.0, 0.5, -0.5, -3.0];
        let upper = array![1.0, 2.0, 1.5, 0.5, 3.0];
        // Unstable: idx 0 (gap=2), idx 1 (gap=4), idx 3 (gap=1), idx 4 (gap=6)
        // idx 2 is stable (l=0.5 > 0)
        let selected = select_neurons_by_uncertainty(&lower, &upper, 2);
        assert_eq!(selected, vec![4, 1]); // Largest gaps first
    }

    #[test]
    fn test_select_neurons_by_uncertainty_all_stable_returns_empty() {
        let lower = array![0.1, 0.2, 0.3];
        let upper = array![1.0, 2.0, 3.0];
        let selected = select_neurons_by_uncertainty(&lower, &upper, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_build_spec_constraints_lower_bound() {
        let linear_bounds = LinearBounds::new(
            array![[1.0, 2.0], [3.0, 4.0]], // lower_a
            array![0.5, 1.0],               // lower_b
            array![[1.0, 2.0], [3.0, 4.0]], // upper_a
            array![0.5, 1.0],               // upper_b
        )
        .unwrap();
        let threshold = 0.0;

        let constraints =
            build_spec_constraints_for_intermediate(&linear_bounds, threshold, false).unwrap();

        // Standard form: lA @ x + (lbias - threshold) <= 0
        assert_eq!(constraints.num_constraints, 2);
        assert_eq!(constraints.a_matrix, array![[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(constraints.b_vector, array![0.5, 1.0]); // 0.5 - 0, 1.0 - 0
    }

    #[test]
    fn test_build_spec_constraints_rounds_half_ulp_down() {
        let linear_bounds =
            LinearBounds::new(array![[0.0]], array![1.0], array![[0.0]], array![1.0]).unwrap();
        let half_ulp_below_one = 2.0_f32.powi(-25);

        let constraints =
            build_spec_constraints_for_intermediate(&linear_bounds, half_ulp_below_one, false)
                .unwrap();

        assert_eq!(constraints.b_vector[0], ny_tensor::next_down_f32(1.0));
        assert!(f64::from(constraints.b_vector[0]) <= 1.0_f64 - f64::from(half_ulp_below_one));
    }

    #[test]
    fn test_tighten_intermediate_skips_when_no_active_constraints() {
        let network = Network {
            layers: vec![],
            gpu_crown_cache: std::sync::Mutex::new(None),
        };
        let input = BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap();
        let mut layer_bounds = vec![];
        let preprocessed = PreprocessedConstraints {
            a_active: Array2::zeros((0, 1)),
            b_active: Array1::zeros(0),
            d_active: Array1::zeros(0),
            infeasible_mask: vec![],
            fully_covered_mask: vec![],
        };

        let result = tighten_intermediate_with_spec_constraints(
            &network,
            &input,
            &mut layer_bounds,
            &preprocessed,
            -1.0,
        );

        assert!(result.is_ok());
        assert!(!result.unwrap()); // No tightening
    }
}
