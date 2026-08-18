// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #spec-axis-alpha slice 2c: slot assignment and per-slot gradients
//! (`docs/SPEC_AXIS_ALPHA_DESIGN.md` §3–§5).
//!
//! Pure functions only — the optimize loop calls these; nothing here touches
//! the walk. Slot assignment picks the K WORST margins at loop entry (the
//! rows joint ascent measurably sacrifices,
//! `CIFAR100_ROOT_ALPHA_DEGRADES_SPEC_BOUNDS_2026-07-26.md`); per-slot
//! gradients replay the SAME stored A-matrices the AnalyticChain gradient
//! uses (`backward/gradients.rs`), just without the row sum that creates the
//! shared-α conflict. Carrier rows resolve to original output rows through
//! the seed's margin subset — a k-row subset seed makes the carrier index a
//! compact id, never a spec id (consult-#5's hard correction; same rule as
//! `AlphaRowScope`).

use std::collections::BTreeMap;

use ndarray::Array2;

use crate::bounds::{GraphAlphaCrownIntermediate, GraphAlphaState};

/// Choose the K worst (lowest lower-bound) spec rows and install zeroed δ
/// state for them across every ReLU node.
///
/// `carrier_lowers[j]` is carrier row `j`'s lower bound from the iteration-0
/// fold; `subset` maps carrier→original row when the seed was a margin
/// subset (`None` = identity). δ starts at exactly 0.0, so the installed
/// slots reproduce the baseline bit-for-bit until the first update — the
/// parity anchor extends through slot installation (pinned by the slice-2
/// walk tests).
///
/// A carrier the `subset` map does not describe is SKIPPED rather than
/// resolved to some default row — the same rule [`compute_spec_gradients`]
/// applies to the identical lookup, so the two stay consistent about which
/// rows own slots.
///
/// Returns the number of slots installed. Installing zero slots (K=0, empty
/// bounds, non-finite-only rows, or a map that describes no selected carrier)
/// leaves the state untouched.
pub(crate) fn assign_spec_slots(
    state: &mut GraphAlphaState,
    carrier_lowers: &[f32],
    subset: Option<&[usize]>,
    requested_slots: usize,
) -> usize {
    if requested_slots == 0 || carrier_lowers.is_empty() {
        return 0;
    }
    // Order carrier rows by ascending lower bound; non-finite rows are
    // excluded outright (a NaN/-inf row's "gradient" is noise — the shared
    // path's sanitize story, applied at selection instead).
    let mut order: Vec<usize> = (0..carrier_lowers.len())
        .filter(|&j| carrier_lowers[j].is_finite())
        .collect();
    if order.is_empty() {
        return 0;
    }
    order.sort_by(|&a, &b| {
        carrier_lowers[a]
            .partial_cmp(&carrier_lowers[b])
            .expect("finite by filter")
    });
    order.truncate(requested_slots);

    // Carrier → original spec row (subset seeds), deduplicated defensively:
    // a malformed subset with repeated ids must not create two slots for one
    // row (`slot_for_spec_row` is first-wins; two slots would shadow).
    let mut slot_rows: Vec<usize> = Vec::with_capacity(order.len());
    for carrier in order {
        // A carrier past the end of the map means the seed and the subset
        // disagree on how many rows there are. SKIP it — this is the identical
        // lookup `compute_spec_gradients` performs below, and that one skips
        // too, so a row invented here could never be fed: it would sit at δ=0
        // burning one of the K slots a genuinely bad row needed. (This used to
        // fold such carriers onto row 0, which the comment already called
        // "install nothing" without doing it.)
        let Some(spec_row) = subset.map_or(Some(carrier), |map| map.get(carrier).copied()) else {
            continue;
        };
        if !slot_rows.contains(&spec_row) {
            slot_rows.push(spec_row);
        }
    }
    if slot_rows.is_empty() {
        return 0; // no carrier survived the map ⇒ install nothing, fail closed
    }

    let node_widths: Vec<(String, usize)> = state
        .alphas
        .iter()
        .map(|(name, alpha)| (name.clone(), alpha.len()))
        .collect();
    state.spec_slot_rows = slot_rows;
    let slots = state.spec_slot_rows.len();
    for (name, width) in node_widths {
        state
            .spec_deltas
            .insert(name, Array2::<f32>::zeros((slots, width)));
    }
    slots
}

/// Per-slot AnalyticChain gradients from the stored per-ReLU A-matrices.
///
/// Identical guards and arithmetic to
/// `compute_graph_chain_rule_gradients` (`backward/gradients.rs`) — same
/// non-finite skips, same unstable-only rule, same `a_ji > 0 ⇒ a_ji · l`
/// contribution — but accumulated PER SLOT instead of row-summed. Carrier
/// row `j` maps through `subset` to its original row, then through the
/// state's slot table; rows without slots contribute nothing here (they
/// keep feeding the shared gradient exactly as before).
///
/// Returns base-width `[K, N]` gradients per node name. Channel-shared
/// nodes are NOT reduced here — the caller mirrors `reduce_gradient` per
/// slot row, keeping this function's contract identical to its sibling's
/// (full width in, full width out).
#[allow(dead_code)] // Per-slot analytic oracle is currently exercised only by module tests.
pub(crate) fn compute_spec_gradients(
    state: &GraphAlphaState,
    relu_nodes: &[String],
    intermediate: &GraphAlphaCrownIntermediate,
    subset: Option<&[usize]>,
) -> BTreeMap<String, Array2<f32>> {
    let slots = state.spec_slot_rows.len();
    let mut out = BTreeMap::new();
    if slots == 0 {
        return out;
    }
    for relu_name in relu_nodes {
        let Some(a_at_relu) = intermediate.a_at_relu(relu_name) else {
            continue;
        };
        let Some((pre_lower, pre_upper)) = intermediate.pre_relu_bounds(relu_name) else {
            continue;
        };
        let n_neurons = pre_lower.len();
        if a_at_relu.ncols() != n_neurons {
            continue; // shape drift ⇒ skip node, fail closed
        }
        let mut grads = Array2::<f32>::zeros((slots, n_neurons));
        let mut touched = false;
        for carrier in 0..a_at_relu.nrows() {
            let spec_row = subset.map_or(Some(carrier), |map| map.get(carrier).copied());
            let Some(spec_row) = spec_row else { continue };
            let Some(slot) = state.slot_for_spec_row(spec_row) else {
                continue;
            };
            for i in 0..n_neurons {
                let l = pre_lower[i];
                let u = pre_upper[i];
                if !l.is_finite() || !u.is_finite() {
                    continue;
                }
                if l >= 0.0 || u <= 0.0 {
                    continue; // stable neurons carry no α gradient
                }
                let a_ji = a_at_relu[[carrier, i]];
                if !a_ji.is_finite() || a_ji <= 0.0 {
                    continue; // upper relaxation is α-independent
                }
                grads[[slot, i]] += a_ji * l;
                touched = true;
            }
        }
        if touched {
            out.insert(relu_name.clone(), grads);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    use super::*;

    fn state_with_relu(width: usize) -> GraphAlphaState {
        let lows: Vec<f32> = (0..width).map(|i| -1.0 - i as f32 * 0.1).collect();
        let highs: Vec<f32> = (0..width).map(|i| 0.5 + i as f32 * 0.1).collect();
        let pre = BoundedTensor::new(arr1(&lows).into_dyn(), arr1(&highs).into_dyn())
            .expect("unstable bounds");
        let mut state = GraphAlphaState::new();
        state
            .add_relu_node("relu", &pre, false)
            .expect("relu state");
        state
    }

    /// Worst-K selection is by ascending lower bound, mapped through the
    /// subset to ORIGINAL row ids — the consult-#5 correction, pinned.
    #[test]
    fn slot_assignment_selects_worst_rows_and_maps_carrier_ids_through_the_subset() {
        let mut state = state_with_relu(4);
        // Carrier rows 0..4 with lowers: row 2 worst, then row 0.
        let lowers = [-5.0_f32, -1.0, -9.0, f32::NAN];
        // Subset: carrier j -> original row (compact seed of rows 10,20,30,40).
        let subset = [10_usize, 20, 30, 40];
        let installed = assign_spec_slots(&mut state, &lowers, Some(&subset), 2);
        assert_eq!(installed, 2);
        assert_eq!(
            state.spec_slot_rows,
            vec![30, 10],
            "worst first (carrier 2 -> original 30), NaN row excluded"
        );
        let deltas = &state.spec_deltas["relu"];
        assert_eq!(deltas.dim(), (2, 4));
        assert!(
            deltas.iter().all(|&d| d == 0.0),
            "δ starts at the parity anchor"
        );
    }

    /// A subset map shorter than the carrier vector is a caller inconsistency
    /// (seed and map disagree on the row count). Carriers past its end are
    /// SKIPPED, never folded onto a default row: `compute_spec_gradients`
    /// skips the identical lookup, so an invented slot could never be fed and
    /// would just burn one of the K slots on a row no carrier owns.
    #[test]
    fn carriers_outside_the_subset_map_are_skipped_not_folded_onto_row_zero() {
        let mut state = state_with_relu(3);
        // Four carriers, but the map only describes the first two.
        let lowers = [-1.0_f32, -9.0, -5.0, -7.0];
        let subset = [10_usize, 20];
        let installed = assign_spec_slots(&mut state, &lowers, Some(&subset), 3);
        assert_eq!(
            state.spec_slot_rows,
            vec![20],
            "worst carrier 1 -> row 20; carriers 3 and 2 fall outside the map and \
             must not appear at all — folding them onto row 0 would install 2 slots"
        );
        assert_eq!(installed, 1);
    }

    /// K=0 and non-finite-only inputs install nothing and leave state
    /// untouched — the dark default costs zero. A map that describes no
    /// selected carrier is the third way to install nothing.
    #[test]
    fn zero_slots_or_unusable_bounds_install_nothing() {
        let mut state = state_with_relu(3);
        assert_eq!(assign_spec_slots(&mut state, &[-1.0, -2.0], None, 0), 0);
        assert_eq!(
            assign_spec_slots(&mut state, &[f32::NAN, f32::NEG_INFINITY], None, 4),
            0
        );
        assert_eq!(
            assign_spec_slots(&mut state, &[-1.0, -2.0], Some(&[]), 2),
            0,
            "an empty map describes no carrier, so nothing is installed"
        );
        assert!(state.spec_slot_rows.is_empty());
        assert!(state.spec_deltas.is_empty());
    }

    /// The per-slot gradient is the UN-SUMMED sibling of the AnalyticChain
    /// gradient: each slot row receives exactly its own carrier row's
    /// `a_ji · l` mass, with the same stability/sign guards.
    // Expected values stay in the `a_ji · l` product form the comments name, so
    // the l = -1.0 factors read the same as their l = -1.1 siblings.
    #[allow(clippy::neg_multiply)]
    #[test]
    fn spec_gradients_replay_the_chain_rule_per_slot_without_the_row_sum() {
        let mut state = state_with_relu(2);
        // Identity seed: carrier == spec row. Slots for rows 1 (slot 0) and 0 (slot 1).
        assert_eq!(
            assign_spec_slots(&mut state, &[-1.0, -4.0, -0.5], None, 2),
            2
        );
        assert_eq!(state.spec_slot_rows, vec![1, 0]);

        let mut intermediate = GraphAlphaCrownIntermediate::new();
        // 3 carrier rows × 2 neurons; both neurons unstable (from fixture:
        // l=-1.0/-1.1, u=0.5/0.6). The intermediate's maps are public —
        // populate them the way the backward pass does.
        let a = ndarray::arr2(&[[2.0_f32, -1.0], [0.5, 3.0], [1.0, 1.0]]);
        intermediate.a_at_relu.insert("relu".to_string(), a);
        intermediate.pre_relu_bounds.insert(
            "relu".to_string(),
            (arr1(&[-1.0_f32, -1.1]), arr1(&[0.5_f32, 0.6])),
        );

        let grads = compute_spec_gradients(&state, &["relu".to_string()], &intermediate, None);
        let g = &grads["relu"];
        // Slot 0 = spec row 1 = carrier 1: [0.5·(-1.0), 3.0·(-1.1)].
        assert_eq!(g[[0, 0]], 0.5 * -1.0);
        assert_eq!(g[[0, 1]], 3.0 * -1.1);
        // Slot 1 = spec row 0 = carrier 0: a_ji>0 only for neuron 0
        // (a=2.0); neuron 1's a=-1.0 is the α-independent upper relaxation.
        assert_eq!(g[[1, 0]], 2.0 * -1.0);
        assert_eq!(g[[1, 1]], 0.0);
        // Carrier 2 owns no slot: contributes nowhere (row isolation).
    }
}
