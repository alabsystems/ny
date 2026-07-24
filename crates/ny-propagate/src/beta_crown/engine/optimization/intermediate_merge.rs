// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Best-intermediate merge helpers for joint optimization loops.

use std::sync::Arc;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::domain::IntermediateLinearBounds;

/// Accumulate tighter intermediate bounds from a finite optimization step only.
///
/// Non-finite scalar lower bounds indicate an unusable iteration for "best"
/// tracking, so the loop must not seed or merge `best_intermediate` from that
/// step. Once a finite iteration exists, keep accumulating row-wise tightening.
pub(super) fn accumulate_tightest_intermediate_bounds(
    best: &mut Option<IntermediateLinearBounds>,
    candidate: &IntermediateLinearBounds,
    current_lower: f32,
    layer_bounds: &[Arc<BoundedTensor>],
) -> Result<()> {
    if !current_lower.is_finite() {
        return Ok(());
    }

    if let Some(best_bounds) = best {
        merge_tightest_intermediate_bounds(best_bounds, candidate, layer_bounds)
    } else {
        *best = Some(candidate.clone());
        Ok(())
    }
}

/// Merge tighter intermediate bounds from another optimization iteration.
///
/// alpha-beta-CROWN stores concrete intermediate tensors and keeps the
/// tightest value per element via `max(lower)` / `min(upper)` across
/// iterations (`optimized_bounds.py:336-361`). ny stores the
/// corresponding `LinearBounds` rows instead, so the sound analogue is to
/// compare concretized row tightness against the box the rows are actually
/// expressed over and keep whichever lower/upper row yields the tighter
/// concrete bound.
///
/// Convention (see `IntermediateLinearBounds` docs and
/// `compute_bounds_capturing_intermediate`): `bounds_at_layer[i]` holds the
/// backward-pass state BEFORE processing layer `i`, i.e. its coefficient
/// columns range over layer `i`'s OUTPUT activation. The concrete box for
/// that activation is `layer_bounds[i]` (layer `i`'s output bounds), NOT
/// `layer_bounds[i-1]` / the network input, which bound layer `i`'s INPUT.
/// Concretizing against the previous box is dimensionally silent on
/// uniform-width networks (rows ranked in the wrong space) and fails closed
/// to `[-inf, +inf]` on width-changing networks (merge becomes a permanent
/// no-op), so a dimension mismatch here is treated as a hard error.
pub(super) fn merge_tightest_intermediate_bounds(
    best: &mut IntermediateLinearBounds,
    candidate: &IntermediateLinearBounds,
    layer_bounds: &[Arc<BoundedTensor>],
) -> Result<()> {
    if best.start_layer != candidate.start_layer {
        return Err(NyError::InvalidSpec(format!(
            "IntermediateLinearBounds start_layer mismatch: {} != {}",
            best.start_layer, candidate.start_layer
        )));
    }
    if best.bounds_at_layer.len() != candidate.bounds_at_layer.len() {
        return Err(NyError::InvalidSpec(format!(
            "IntermediateLinearBounds length mismatch: {} != {}",
            best.bounds_at_layer.len(),
            candidate.bounds_at_layer.len()
        )));
    }

    for layer_idx in 0..best.bounds_at_layer.len() {
        // bounds_at_layer[layer_idx] is expressed over layer `layer_idx`'s
        // OUTPUT activation, whose concrete box is layer_bounds[layer_idx].
        let activation_box = layer_bounds
            .get(layer_idx)
            .map(Arc::as_ref)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "IntermediateLinearBounds merge missing layer_bounds[{layer_idx}]: \
                     bounds_at_layer[{layer_idx}] is expressed over layer {layer_idx}'s \
                     output activation",
                ))
            })?;

        let best_layer = best.bounds_at_layer[layer_idx].as_ref();
        let candidate_layer = candidate.bounds_at_layer[layer_idx].as_ref();

        if best_layer.num_outputs() != candidate_layer.num_outputs()
            || best_layer.num_inputs() != candidate_layer.num_inputs()
        {
            return Err(NyError::InvalidSpec(format!(
                "IntermediateLinearBounds layer {} shape mismatch: best=({}x{}), candidate=({}x{})",
                layer_idx,
                best_layer.num_outputs(),
                best_layer.num_inputs(),
                candidate_layer.num_outputs(),
                candidate_layer.num_inputs()
            )));
        }

        // A box/coefficient dimension mismatch means the indexing convention
        // above was violated. It must surface as a bug, not degrade into
        // concretize_sound's [-inf, +inf] fail-closed fallback that silently
        // turns the merge into a no-op (observed 51k times on non-uniform
        // RSPLITTER nets with the old `layer_idx - 1` indexing).
        debug_assert_eq!(
            best_layer.num_inputs(),
            activation_box.len(),
            "bounds_at_layer[{layer_idx}] has {} input columns but \
             layer_bounds[{layer_idx}] (layer {layer_idx}'s output box) has {} \
             elements; IntermediateLinearBounds indexing convention violated",
            best_layer.num_inputs(),
            activation_box.len(),
        );
        if best_layer.num_inputs() != activation_box.len() {
            return Err(NyError::InvalidSpec(format!(
                "IntermediateLinearBounds merge box mismatch at layer {layer_idx}: \
                 bounds have {} input columns but layer_bounds[{layer_idx}] has {} \
                 elements (bounds_at_layer[i] must be concretized against layer i's \
                 output box)",
                best_layer.num_inputs(),
                activation_box.len(),
            )));
        }

        let best_concrete = best_layer.concretize_sound(activation_box);
        let candidate_concrete = candidate_layer.concretize_sound(activation_box);
        let mut merged_layer = best_layer.clone();
        let mut changed = false;

        for row_idx in 0..best_layer.num_outputs() {
            let best_lower = best_concrete.lower()[[row_idx]];
            let candidate_lower = candidate_concrete.lower()[[row_idx]];
            if candidate_lower > best_lower {
                merged_layer
                    .lower_a_mut()
                    .row_mut(row_idx)
                    .assign(&candidate_layer.lower_a().row(row_idx));
                merged_layer.lower_b_mut()[row_idx] = candidate_layer.lower_b()[row_idx];
                changed = true;
            }

            let best_upper = best_concrete.upper()[[row_idx]];
            let candidate_upper = candidate_concrete.upper()[[row_idx]];
            if candidate_upper < best_upper {
                merged_layer
                    .upper_a_mut()
                    .row_mut(row_idx)
                    .assign(&candidate_layer.upper_a().row(row_idx));
                merged_layer.upper_b_mut()[row_idx] = candidate_layer.upper_b()[row_idx];
                changed = true;
            }
        }

        if changed {
            best.bounds_at_layer[layer_idx] = Arc::new(merged_layer);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::*;
    use crate::LinearBounds;

    fn box1(lo: f32, hi: f32) -> BoundedTensor {
        BoundedTensor::new(arr1(&[lo]).into_dyn(), arr1(&[hi]).into_dyn()).expect("valid 1-d box")
    }

    #[test]
    fn test_merge_tightest_intermediate_bounds_2417() {
        // bounds_at_layer[0] is expressed over layer 0's OUTPUT activation, so
        // the merge ranks rows against layer_bounds[0] = [0, 1].
        let layer0_box = box1(0.0, 1.0);
        let layer_bounds = vec![Arc::new(layer0_box.clone())];

        let best = LinearBounds::new(
            arr2(&[[0.0_f32], [0.0_f32]]),
            arr1(&[0.0_f32, 0.0_f32]),
            arr2(&[[0.0_f32], [0.0_f32]]),
            arr1(&[4.0_f32, 5.0_f32]),
        )
        .expect("valid best linear bounds");
        let candidate = LinearBounds::new(
            arr2(&[[0.0_f32], [0.0_f32]]),
            arr1(&[1.0_f32, -1.0_f32]),
            arr2(&[[0.0_f32], [0.0_f32]]),
            arr1(&[6.0_f32, 3.0_f32]),
        )
        .expect("valid candidate linear bounds");

        let mut best_intermediate = IntermediateLinearBounds {
            bounds_at_layer: vec![Arc::new(best)],
            start_layer: 0,
        };
        let candidate_intermediate = IntermediateLinearBounds {
            bounds_at_layer: vec![Arc::new(candidate)],
            start_layer: 0,
        };

        merge_tightest_intermediate_bounds(
            &mut best_intermediate,
            &candidate_intermediate,
            &layer_bounds,
        )
        .expect("merge should succeed");

        let merged = best_intermediate
            .get(0)
            .expect("merged layer 0 should exist");
        let concretized = merged.concretize_sound(&layer0_box);

        assert_eq!(merged.lower_b()[0], 1.0);
        assert_eq!(merged.upper_b()[0], 4.0);
        assert_eq!(merged.lower_b()[1], 0.0);
        assert_eq!(merged.upper_b()[1], 3.0);

        assert!(
            concretized.lower()[[0]] <= 1.0 && concretized.lower()[[0]] > 0.9999,
            "directed rounding should keep the merged lower bound just below 1.0"
        );
        assert!(
            concretized.upper()[[0]] >= 4.0 && concretized.upper()[[0]] < 4.0001,
            "directed rounding should keep the merged upper bound just above 4.0"
        );
        assert!(
            concretized.lower()[[1]] <= 0.0 && concretized.lower()[[1]] > -0.0001,
            "directed rounding should keep the merged lower bound at or just below 0.0"
        );
        assert!(
            concretized.upper()[[1]] >= 3.0 && concretized.upper()[[1]] < 3.0001,
            "directed rounding should keep the merged upper bound just above 3.0"
        );
    }

    #[test]
    fn test_accumulate_tightest_intermediate_bounds_skips_non_finite_lower_2417() {
        let layer_bounds = vec![Arc::new(box1(0.0, 1.0))];
        let candidate = LinearBounds::new(
            arr2(&[[0.0_f32]]),
            arr1(&[1.0_f32]),
            arr2(&[[0.0_f32]]),
            arr1(&[2.0_f32]),
        )
        .expect("valid candidate linear bounds");
        let candidate_intermediate = IntermediateLinearBounds {
            bounds_at_layer: vec![Arc::new(candidate)],
            start_layer: 0,
        };

        let mut best_intermediate = None;
        accumulate_tightest_intermediate_bounds(
            &mut best_intermediate,
            &candidate_intermediate,
            f32::NEG_INFINITY,
            &layer_bounds,
        )
        .expect("non-finite lower bound should skip accumulation");
        assert!(
            best_intermediate.is_none(),
            "non-finite optimization iterations must not seed best_intermediate"
        );

        accumulate_tightest_intermediate_bounds(
            &mut best_intermediate,
            &candidate_intermediate,
            0.5,
            &layer_bounds,
        )
        .expect("finite lower bound should seed best_intermediate");
        assert!(
            best_intermediate.is_some(),
            "first finite optimization iteration should seed best_intermediate"
        );
    }

    /// Discriminating oracle for the wrong-box indexing bug (non-uniform widths).
    ///
    /// Miniature of the RSPLITTER 784->271->256 failure: layer 0's output has
    /// width 3, layer 1's output has width 2. The old code concretized
    /// bounds_at_layer[0] against the network input box and bounds_at_layer[1]
    /// against layer_bounds[0] (width 3) — both dimension mismatches, so
    /// concretize_sound failed closed to [-inf, +inf] and the merge was a
    /// permanent no-op (candidate rows never adopted). With the correct
    /// convention (bounds_at_layer[i] over layer i's output box =
    /// layer_bounds[i]) the merge adopts every hand-computed tighter row.
    #[test]
    fn test_merge_non_uniform_widths_tightens_with_correct_box() {
        // layer 0 output box: [0, 1]^3; layer 1 output box: [-1, 2]^2.
        let layer_bounds = vec![
            Arc::new(
                BoundedTensor::new(
                    arr1(&[0.0_f32, 0.0, 0.0]).into_dyn(),
                    arr1(&[1.0_f32, 1.0, 1.0]).into_dyn(),
                )
                .expect("valid layer 0 output box"),
            ),
            Arc::new(
                BoundedTensor::new(
                    arr1(&[-1.0_f32, -1.0]).into_dyn(),
                    arr1(&[2.0_f32, 2.0]).into_dyn(),
                )
                .expect("valid layer 1 output box"),
            ),
        ];

        // Entry 0 (1 output row over 3 inputs), over [0,1]^3:
        //   best:      lower = x0            -> min 0;   upper = x0        -> max 1
        //   candidate: lower = x1 + x2 + 0.5 -> min 0.5; upper = 0.75      -> max 0.75
        // candidate is tighter on both sides.
        let best0 = LinearBounds::new(
            arr2(&[[1.0_f32, 0.0, 0.0]]),
            arr1(&[0.0_f32]),
            arr2(&[[1.0_f32, 0.0, 0.0]]),
            arr1(&[0.0_f32]),
        )
        .expect("valid best layer-0 bounds");
        let candidate0 = LinearBounds::new(
            arr2(&[[0.0_f32, 1.0, 1.0]]),
            arr1(&[0.5_f32]),
            arr2(&[[0.0_f32, 0.0, 0.0]]),
            arr1(&[0.75_f32]),
        )
        .expect("valid candidate layer-0 bounds");

        // Entry 1 (1 output row over 2 inputs), over [-1,2]^2:
        //   best:      lower = x0    -> min -1;    upper = x1  -> max 2
        //   candidate: lower = -0.25 -> min -0.25; upper = 2.5 -> max 2.5
        // candidate lower is tighter; candidate upper is LOOSER, so the merge
        // must mix rows: candidate lower + best upper.
        let best1 = LinearBounds::new(
            arr2(&[[1.0_f32, 0.0]]),
            arr1(&[0.0_f32]),
            arr2(&[[0.0_f32, 1.0]]),
            arr1(&[0.0_f32]),
        )
        .expect("valid best layer-1 bounds");
        let candidate1 = LinearBounds::new(
            arr2(&[[0.0_f32, 0.0]]),
            arr1(&[-0.25_f32]),
            arr2(&[[0.0_f32, 0.0]]),
            arr1(&[2.5_f32]),
        )
        .expect("valid candidate layer-1 bounds");

        let mut best_intermediate = IntermediateLinearBounds {
            bounds_at_layer: vec![Arc::new(best0), Arc::new(best1)],
            start_layer: 1,
        };
        let candidate_intermediate = IntermediateLinearBounds {
            bounds_at_layer: vec![Arc::new(candidate0), Arc::new(candidate1)],
            start_layer: 1,
        };

        merge_tightest_intermediate_bounds(
            &mut best_intermediate,
            &candidate_intermediate,
            &layer_bounds,
        )
        .expect("non-uniform merge should succeed with correct per-layer boxes");

        let merged0 = best_intermediate.get(0).expect("merged layer 0");
        assert_eq!(
            merged0.lower_b()[0],
            0.5,
            "layer 0 lower row must come from the tighter candidate (min 0.5 > 0)"
        );
        assert_eq!(merged0.lower_a().row(0).to_vec(), vec![0.0, 1.0, 1.0]);
        assert_eq!(
            merged0.upper_b()[0],
            0.75,
            "layer 0 upper row must come from the tighter candidate (max 0.75 < 1)"
        );
        assert_eq!(merged0.upper_a().row(0).to_vec(), vec![0.0, 0.0, 0.0]);

        let merged1 = best_intermediate.get(1).expect("merged layer 1");
        assert_eq!(
            merged1.lower_b()[0],
            -0.25,
            "layer 1 lower row must come from the tighter candidate (-0.25 > -1)"
        );
        assert_eq!(merged1.lower_a().row(0).to_vec(), vec![0.0, 0.0]);
        assert_eq!(
            merged1.upper_b()[0],
            0.0,
            "layer 1 upper row must stay from best (max 2 < 2.5): rows must mix"
        );
        assert_eq!(merged1.upper_a().row(0).to_vec(), vec![0.0, 1.0]);

        // Sanity: concrete tightness against the CORRECT boxes (directed
        // rounding permits one-ulp slack).
        let concrete0 = merged0.concretize_sound(layer_bounds[0].as_ref());
        assert!(concrete0.lower()[[0]] > 0.4999 && concrete0.lower()[[0]] <= 0.5);
        assert!(concrete0.upper()[[0]] >= 0.75 && concrete0.upper()[[0]] < 0.7501);
        let concrete1 = merged1.concretize_sound(layer_bounds[1].as_ref());
        assert!(concrete1.lower()[[0]] > -0.2501 && concrete1.lower()[[0]] <= -0.25);
        assert!(concrete1.upper()[[0]] >= 2.0 && concrete1.upper()[[0]] < 2.0001);
    }

    /// Discriminating oracle for the uniform-width wrong-box ranking.
    ///
    /// All widths are 1, so the old `layer_idx - 1` indexing was dimensionally
    /// silent — but it ranked rows against the box of layer 0's INPUT
    /// (historically the network input box, [0, 1] here) instead of layer 0's
    /// OUTPUT box ([-2, -1]). Over the wrong box the best lower row (x -> x,
    /// min 0) looks tighter than the candidate constant row (-1.5), so the old
    /// code kept the looser row. Over the correct box, x -> x has min -2 and
    /// the candidate's -1.5 is tighter and must win.
    #[test]
    fn test_merge_uniform_width_ranks_rows_in_output_box_not_input_box() {
        let layer0_output_box = box1(-2.0, -1.0);
        let layer_bounds = vec![Arc::new(layer0_output_box)];

        let best = LinearBounds::new(
            arr2(&[[1.0_f32]]),
            arr1(&[0.0_f32]),
            arr2(&[[0.0_f32]]),
            arr1(&[5.0_f32]),
        )
        .expect("valid best linear bounds");
        let candidate = LinearBounds::new(
            arr2(&[[0.0_f32]]),
            arr1(&[-1.5_f32]),
            arr2(&[[1.0_f32]]),
            arr1(&[0.0_f32]),
        )
        .expect("valid candidate linear bounds");

        let mut best_intermediate = IntermediateLinearBounds {
            bounds_at_layer: vec![Arc::new(best)],
            start_layer: 0,
        };
        let candidate_intermediate = IntermediateLinearBounds {
            bounds_at_layer: vec![Arc::new(candidate)],
            start_layer: 0,
        };

        merge_tightest_intermediate_bounds(
            &mut best_intermediate,
            &candidate_intermediate,
            &layer_bounds,
        )
        .expect("uniform-width merge should succeed");

        let merged = best_intermediate.get(0).expect("merged layer 0");
        assert_eq!(
            merged.lower_b()[0],
            -1.5,
            "lower row must be ranked in layer 0's OUTPUT box [-2,-1] where the \
             candidate constant -1.5 beats best's min of -2 (the old input-box \
             ranking over [0,1] kept the looser best row)"
        );
        assert_eq!(merged.lower_a().row(0).to_vec(), vec![0.0]);
        assert_eq!(
            merged.upper_b()[0],
            0.0,
            "upper row: candidate x -> x has max -1 over [-2,-1], tighter than 5"
        );
        assert_eq!(merged.upper_a().row(0).to_vec(), vec![1.0]);
    }

    /// A box/coefficient dimension mismatch is a convention violation and must
    /// fail loudly instead of silently no-oping via the [-inf, +inf] fallback.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "indexing convention"))]
    fn test_merge_box_dimension_mismatch_is_loud() {
        // 1-input rows but a 2-element "box": convention violated.
        let bad_box = BoundedTensor::new(
            arr1(&[0.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("valid 2-d box");
        let layer_bounds = vec![Arc::new(bad_box)];

        let row = || {
            LinearBounds::new(
                arr2(&[[1.0_f32]]),
                arr1(&[0.0_f32]),
                arr2(&[[1.0_f32]]),
                arr1(&[0.0_f32]),
            )
            .expect("valid linear bounds")
        };
        let mut best_intermediate = IntermediateLinearBounds {
            bounds_at_layer: vec![Arc::new(row())],
            start_layer: 0,
        };
        let candidate_intermediate = IntermediateLinearBounds {
            bounds_at_layer: vec![Arc::new(row())],
            start_layer: 0,
        };

        // Debug builds panic on the debug_assert; release builds get an Err.
        let result = merge_tightest_intermediate_bounds(
            &mut best_intermediate,
            &candidate_intermediate,
            &layer_bounds,
        );
        assert!(
            result.is_err(),
            "dimension mismatch must be an error, never a silent no-op"
        );
    }
}
