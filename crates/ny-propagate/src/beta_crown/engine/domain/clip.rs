// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Clip-intermediate adaptation and NaN-aware merge helpers for domain bounds.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Once};

use ndarray::{Array1, Array2};
use ny_core::{nan_propagating_max, nan_propagating_min, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::warn;

use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::domain::IntermediateLinearBounds;
use crate::clip_interm_domain::clip_interm_domain_full;
use crate::Network;

use super::clip_provenance::{input_relative_rows, row_at, split_subject, ProvenanceSubject};

use super::super::BetaCrownVerifier;

/// The sequential engine's `IntermediateLinearBounds` rows are relative to a
/// layer output, not to the network input. They therefore cannot be passed to
/// the input-space Complete Clipping solver without a provenance-preserving
/// backward conversion. Keep this adapter quarantined until that conversion
/// exists; the graph Complete Clip root-bank path is independent.
///
/// DO NOT FLIP THIS TO `true` AS A SPEED CHANGE. Feeding layer-output-relative
/// [`IntermediateLinearBounds`] rows to the input-space Complete Clipping solver
/// intersects bounds against constraints expressed in the WRONG frame of
/// reference, which can cut away a region containing the true solution — that is
/// a false-`unsat` generator, i.e. the one bug class the moat forbids. Flipping
/// it requires the backward conversion plus the three gates named in
/// `sequential_clip_interm_domain_stays_quarantined`.
///
/// STATUS (#clip-provenance). The conversion now EXISTS and is wired: see
/// [`super::clip_provenance`] and
/// [`BetaCrownVerifier::clip_interm_domain_converted`], which no longer read the
/// layer-output-relative rows at all — they re-derive certified input-relative
/// per-neuron rows by the same backward composition CROWN already performs. Unit
/// oracles, a dense-grid containment property test with an injected-off-by-one
/// teeth case, and an end-to-end differential enclosure test over random splits
/// all pass. What is NOT yet discharged is the MEASURED half of the flip gate —
/// gate (c), the moat sweep over relusplitter/soundnessbench ground truth — so
/// the predicate stays `false` and the production path is byte-identical to
/// before. Flipping it without that sweep is exactly the unvalidated-math move
/// the moat forbids.
#[inline]
const fn sequential_clip_interm_has_input_relative_provenance() -> bool {
    false
}

/// Can the SEQUENTIAL engine actually honour `enable_clip_interm_domain`?
///
/// `false` means the optional tightening is SKIPPED, not failed: callers should
/// consult this to avoid the pointless work, and
/// [`BetaCrownVerifier::apply_clip_interm_domain`] independently returns the
/// caller's bounds UNCHANGED if it is invoked anyway. Both layers are required —
/// see the history below for why the second one exists.
///
/// HISTORY (do not re-derive this as a live bug). Until 7b933584 the adapter
/// returned [`NyError::SoundnessRefusal`] for EVERY child at `history.depth() > 0`
/// and the sequential child-creation path propagated it with `?`, so a preset that
/// merely *requested* the feature made every child fail; the engine recorded
/// `PropagationFailure` and BaB resolved nothing at all. Measured on
/// `configs/vnncomp25/relusplitter.yaml` (which sets `bab.clip.interm_domain: true`):
/// BaB never ran on that whole 220-instance benchmark, the tell being ~13 s of budget
/// left over for a "Post-BaB attack" phase because BaB had collapsed instantly.
/// 7b933584 added the capability gate at
/// `super::super::domain::child::create_child_domain`, so the request is now skipped.
/// This function keeps that gate a single named seam; the identity fallback inside
/// the adapter makes the fix survive a refactor that drops the call-site check.
///
/// WHY SKIPPING IS SOUND AND ERRORING WAS NOT. The pre-tightening bounds handed to
/// the adapter are already valid enclosures, so declining to tighten them can only
/// make bounds LOOSER — costing precision (fewer domains closed, more `unknown`)
/// and never manufacturing a false `unsat`. Erroring was the actively harmful
/// behaviour: it converted provable instances into `unknown`.
#[inline]
pub(crate) const fn sequential_clip_interm_domain_supported() -> bool {
    sequential_clip_interm_has_input_relative_provenance()
}

/// One-shot operator-visible notice that a requested `clip_interm_domain` was
/// skipped. Logged once per process so it cannot flood the BaB inner loop, but
/// loud enough that "my preset asked for this and nothing happened" is
/// diagnosable from a normal run log rather than only from source.
pub(super) fn warn_sequential_clip_interm_skipped_once() {
    static NOTICE: Once = Once::new();
    NOTICE.call_once(|| {
        warn!(
            "sequential clip_interm_domain was REQUESTED but is quarantined \
             (IntermediateLinearBounds are layer-output-relative and lack certified \
             input-relative provenance); skipping the optional bound tightening. \
             Bounds remain valid enclosures, merely looser — BaB continues normally."
        );
    });
}

/// Merge old and new domain bounds using NaN-propagating min/max.
///
/// For lower bounds: `max(old, new)` (tighten from below).
/// For upper bounds: `min(old, new)` (tighten from above).
///
/// Uses `nan_propagating_max`/`nan_propagating_min` so that NaN in either old
/// or new bounds is never silently absorbed into a finite (unsound) bound.
/// IEEE 754 `f32::max`/`f32::min` return the non-NaN operand, hiding corruption.
///
/// Reference: #2858, #2577.
pub(super) fn merge_domain_bounds(
    old_lower: &Array1<f32>,
    new_lower: &Array1<f32>,
    old_upper: &Array1<f32>,
    new_upper: &Array1<f32>,
) -> (Array1<f32>, Array1<f32>) {
    let merged_lower: Array1<f32> = old_lower
        .iter()
        .zip(new_lower.iter())
        .map(|(&o, &n): (&f32, &f32)| nan_propagating_max(o, n))
        .collect();
    let merged_upper: Array1<f32> = old_upper
        .iter()
        .zip(new_upper.iter())
        .map(|(&o, &n): (&f32, &f32)| nan_propagating_min(o, n))
        .collect();
    (merged_lower, merged_upper)
}

pub(super) fn has_infeasible_layer_bounds(layer_bounds: &[Arc<BoundedTensor>]) -> bool {
    layer_bounds.iter().any(|bt| {
        ndarray::Zip::from(bt.lower())
            .and(bt.upper())
            .any(|&l, &u| l > u)
    })
}

impl BetaCrownVerifier {
    /// Apply clip_interm_domain to tighten intermediate bounds using split constraints.
    ///
    /// This adapts the engine's data formats to the `clip_interm_domain_full` API:
    /// - Converts `SplitHistory` to `GraphSplitHistory`
    /// - Converts every consumed row to certified INPUT-RELATIVE provenance
    ///   (`super::clip_provenance`)
    /// - Updates layer bounds with tightened values
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine) fn apply_clip_interm_domain(
        &self,
        network: &Network,
        history: &SplitHistory,
        layer_bounds: Vec<Arc<BoundedTensor>>,
        _intermediate_bounds: &IntermediateLinearBounds,
        input: &BoundedTensor,
        parent_input_bounds: Option<&BoundedTensor>,
    ) -> Result<Vec<Arc<BoundedTensor>>> {
        // QUARANTINE, EXPRESSED AS A SKIP — NOT AS AN ERROR.
        //
        // `layer_bounds` is what the caller already computed and already trusts:
        // a valid enclosure of every reachable activation in this domain. Handing
        // it straight back declines an OPTIONAL tightening, which can only leave
        // bounds looser than they might have been. Looser bounds cost precision
        // (more `unknown`) and cannot manufacture a false `unsat`, so this return
        // is sound in the safe direction by construction.
        //
        // Returning `Err` here instead is what broke BaB: `create_child_domain`
        // propagates it with `?`, the domain becomes a `PropagationFailure`, and
        // since EVERY child sits at `history.depth() > 0` the search collapsed at
        // the root. The call site also consults
        // `sequential_clip_interm_domain_supported()` so this work is normally
        // skipped without entering the function; this branch is the belt-and-braces
        // half that keeps a dropped call-site check from re-breaking the search.
        if !sequential_clip_interm_has_input_relative_provenance() {
            warn_sequential_clip_interm_skipped_once();
            return Ok(layer_bounds);
        }

        self.clip_interm_domain_converted(
            network,
            history,
            layer_bounds,
            input,
            parent_input_bounds,
        )
    }

    /// The conversion-backed body of [`Self::apply_clip_interm_domain`].
    ///
    /// Separated from the quarantine check so the validation harness can drive
    /// the REAL conversion, the REAL solver and the REAL merge without a
    /// test-only override of the capability predicate leaking into production
    /// control flow.
    ///
    /// # What this does NOT read
    ///
    /// `IntermediateLinearBounds`. Those rows are the SPEC objective written over
    /// a layer's output activation: their row space is `output_dim` and their
    /// column space is `width(h_k)`. The solver needs one row PER NEURON over the
    /// NETWORK INPUT. No reindexing bridges that — the old adapter's reading of
    /// `lower_a().row(neuron_idx)` as "neuron `neuron_idx` over the input" was
    /// wrong in both axes at once, which is what the quarantine was protecting
    /// against. Every row consumed below is instead re-derived by
    /// [`super::clip_provenance::input_relative_rows`].
    ///
    /// # Soundness argument
    ///
    /// Let `B` be this domain's effective input box, `S ⊆ B` the region carved
    /// out by the split history, and `F = {x ∈ B : A_c·x + b_c ≤ 0}` the solver's
    /// relaxed feasible set.
    ///
    /// * The converted rows are valid POINTWISE ON `S` — NOT on all of `B`, and
    ///   the difference matters. They are derived over this child's split-CLAMPED
    ///   `layer_bounds`, which enclose the activations for every `x ∈ S`; outside
    ///   `S` a clamped ReLU relaxation need not hold at all. (The composition is
    ///   the engine's certified backward step and the coefficient-error envelope
    ///   is folded outward into the bias before use.) Anyone building on this
    ///   argument must carry `S`, not `B`: "valid on `B`" is a premise the code
    ///   does not establish, and a future flip reasoning from it would be
    ///   reasoning from something false.
    /// * Each split constraint is therefore a NECESSARY condition of its branch
    ///   (`lA·x + lb ≤ z(x) ≤ 0` on the inactive branch, and mirrored on the
    ///   active one), so `S ⊆ F`.
    /// * The solver returns `min_F(lA·x) + lb` and `max_F(uA·x) + ub`. Since
    ///   `F ⊇ S` and the rows enclose the activation on `S`, those values enclose
    ///   the activation on `S`. Widening `S` to `F` can only LOOSEN the result,
    ///   never cut a reachable point. THIS is the step that carries the argument
    ///   despite the rows being valid only on `S`: minimizing a row over a
    ///   SUPERSET of `S` still bounds it at every `x ∈ S`.
    ///
    /// Every step that could go the other way is gated: a row that cannot be
    /// derived, that keeps a residual coefficient-error envelope, or that
    /// contains a non-finite entry is DROPPED (constraint) or SKIPPED (layer),
    /// and `merge_bounds` keeps the original interval whenever tightening would
    /// invert or introduce NaN.
    pub(in crate::beta_crown::engine) fn clip_interm_domain_converted(
        &self,
        network: &Network,
        history: &SplitHistory,
        mut layer_bounds: Vec<Arc<BoundedTensor>>,
        input: &BoundedTensor,
        parent_input_bounds: Option<&BoundedTensor>,
    ) -> Result<Vec<Arc<BoundedTensor>>> {
        // Convert SplitHistory to GraphSplitHistory for clip_interm_domain API
        let graph_history = history.to_graph_split_history()?;

        // Get input bounds (use tightened bounds if available).
        //
        // ONE BOX FOR THE THREE SOLVER-SIDE USES. The same `effective_input`
        // seeds the backward composition, discharges the coefficient-error
        // envelope, and bounds the solver's optimization, so those three can
        // never disagree. Optimizing rows over a box WIDER than the one they were
        // derived over is unsound; over a narrower one it is merely loose, and
        // the code must not have to reason about which — hence one binding.
        //
        // To be precise about what this does NOT unify: `effective_input` is the
        // PARENT's box, while `layer_bounds` are the CHILD's split-clamped
        // bounds, so the rows are valid on `S` and optimized over `F ⊇ S`. That
        // asymmetry is deliberate and is the safe direction — see the
        // min-over-superset step in the soundness argument above.
        let effective_input = parent_input_bounds.unwrap_or(input);
        let input_flat = effective_input.flatten();
        let input_lower: Array1<f32> =
            Array1::from_vec(input_flat.lower().iter().copied().collect());
        let input_upper: Array1<f32> =
            Array1::from_vec(input_flat.upper().iter().copied().collect());

        // Convert layer_bounds to the expected format: Vec<(Array1<f32>, Array1<f32>)>
        let layer_bounds_flat: Vec<(Array1<f32>, Array1<f32>)> = layer_bounds
            .iter()
            .map(|bt| {
                let flat = bt.flatten();
                let lower_arr: Array1<f32> =
                    Array1::from_vec(flat.lower().iter().copied().collect());
                let upper_arr: Array1<f32> =
                    Array1::from_vec(flat.upper().iter().copied().collect());
                (lower_arr, upper_arr)
            })
            .collect();

        // SPLIT ROWS. One backward composition per distinct split SUBJECT layer
        // (not per split), so a depth-40 history on a 9-layer net costs at most 9
        // passes. A subject whose conversion refuses contributes no rows, and the
        // builder then drops those constraints — a weaker relaxation, never a
        // stronger one.
        let mut subjects: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for constraint in &history.constraints {
            subjects
                .entry(constraint.layer_idx)
                .or_default()
                .insert(constraint.neuron_idx);
        }
        #[allow(clippy::type_complexity)]
        let mut split_rows: HashMap<(usize, usize), (Array1<f32>, f32, Array1<f32>, f32)> =
            HashMap::new();
        for (split_layer_idx, neuron_set) in subjects {
            let neurons: Vec<usize> = neuron_set.into_iter().collect();
            let Some(rows) = input_relative_rows(
                network,
                effective_input,
                &layer_bounds,
                split_subject(split_layer_idx),
                &neurons,
            ) else {
                continue;
            };
            for (row_idx, &neuron_idx) in neurons.iter().enumerate() {
                if let Some(row) = row_at(&rows, row_idx) {
                    split_rows.insert((split_layer_idx, neuron_idx), row);
                }
            }
        }

        // Adapter for split neuron linear bounds.
        // node_name format: "layer_N" (see `SplitHistory::to_graph_split_history`).
        let linear_bounds_for_split =
            |node_name: &str, neuron_idx: usize| -> Option<(Array1<f32>, f32, Array1<f32>, f32)> {
                let layer_idx: usize = node_name.strip_prefix("layer_")?.parse().ok()?;
                split_rows.get(&(layer_idx, neuron_idx)).cloned()
            };

        // Adapter for objective neuron linear bounds: one backward composition
        // per layer, seeded at the neurons `select_objective_neurons` picked.
        // Justification: Closure returns (lA, lbias, uA, ubias) tuple — the natural
        // representation of linear bound coefficients; a named struct would add indirection.
        #[allow(clippy::type_complexity)]
        let linear_bounds_for_objective =
            |layer_idx: usize,
             neuron_indices: &[usize]|
             -> Option<(Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>)> {
                let rows = input_relative_rows(
                    network,
                    effective_input,
                    &layer_bounds,
                    ProvenanceSubject::LayerOutput(layer_idx),
                    neuron_indices,
                )?;
                Some((
                    rows.lower_a().clone(),
                    rows.lower_b().clone(),
                    rows.upper_a().clone(),
                    rows.upper_b().clone(),
                ))
            };

        // Call clip_interm_domain_full to get tightened bounds.
        //
        // `coeff_magnitudes` is deliberately `None`: the only magnitudes available
        // without a full-width conversion per layer are the layer-output-relative
        // ones, and ranking candidates by a magnitude measured in the wrong frame
        // is worse than uniform. Selection is throughput-only — it cannot affect
        // soundness — so the honest uniform weighting is used.
        let tightened = clip_interm_domain_full(
            &graph_history,
            linear_bounds_for_split,
            linear_bounds_for_objective,
            &layer_bounds_flat,
            &input_lower,
            &input_upper,
            self.config.clip_interm_topk,
            None,
        )?;

        // Update layer_bounds with tightened values
        for (layer_idx, (new_lower, new_upper)) in tightened.into_iter().enumerate() {
            if layer_idx >= layer_bounds.len() {
                break;
            }

            let old_bt = &layer_bounds[layer_idx];
            let old_flat = old_bt.flatten();
            let old_shape = old_bt.lower().shape().to_vec();

            // Merge: new = max(old, tightened) for lower, min(old, tightened) for upper
            let old_lower: Array1<f32> =
                Array1::from_vec(old_flat.lower().iter().copied().collect());
            let old_upper: Array1<f32> =
                Array1::from_vec(old_flat.upper().iter().copied().collect());

            // Check if bounds changed.
            // NaN-aware: treat NaN as a change (NaN comparisons return false,
            // which would silently classify NaN-corrupted domains as "unchanged").
            let mut changed = false;
            for i in 0..new_lower.len().min(old_lower.len()) {
                if new_lower[i] > old_lower[i]
                    || new_upper[i] < old_upper[i]
                    || new_lower[i].is_nan()
                    || new_upper[i].is_nan()
                    || old_lower[i].is_nan()
                    || old_upper[i].is_nan()
                {
                    changed = true;
                    break;
                }
            }

            if changed {
                let (merged_lower, merged_upper) =
                    merge_domain_bounds(&old_lower, &new_lower, &old_upper, &new_upper);

                // Reshape to original shape and create new BoundedTensor
                let lower_dyn = merged_lower
                    .into_shape_clone(ndarray::IxDyn(&old_shape))
                    .map_err(|err| {
                        NyError::InternalError(format!(
                            "apply_clip_interm_domain: reshape lower bounds failed for layer {} to {:?}: {}",
                            layer_idx, old_shape, err
                        ))
                    })?;
                let upper_dyn = merged_upper
                    .into_shape_clone(ndarray::IxDyn(&old_shape))
                    .map_err(|err| {
                        NyError::InternalError(format!(
                            "apply_clip_interm_domain: reshape upper bounds failed for layer {} to {:?}: {}",
                            layer_idx, old_shape, err
                        ))
                    })?;

                layer_bounds[layer_idx] = Arc::new(BoundedTensor::new(lower_dyn, upper_dyn)?);
            }
        }

        Ok(layer_bounds)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;

    /// TRIPWIRE. The quarantine predicate is the ONLY thing standing between the
    /// sequential engine's layer-output-relative `IntermediateLinearBounds` and an
    /// input-space clipping solver. Intersecting bounds against constraints stated
    /// in the wrong frame of reference can cut away a region containing the true
    /// solution — a false-`unsat` generator. If you are here because you flipped
    /// the predicate, land ALL THREE gates first and only then update this test:
    ///
    ///   (a) a differential enclosure test — for random splits, the bounds returned
    ///       by `apply_clip_interm_domain` must CONTAIN every concrete forward
    ///       activation reachable in the split box (sampled + ORT-cross-checked),
    ///       never cut one off;
    ///       LANDED: `converted_clip_never_cuts_off_a_reachable_activation`.
    ///   (b) a teeth test — inject a conversion 1 ULP too tight relative to the TRUE
    ///       bound and confirm gate (a) catches it (do NOT "prove teeth" by shaving
    ///       an already-loose shipped bound; that passes while proving nothing);
    ///       LANDED: `enclosure_check_catches_a_one_ulp_too_tight_conversion`, plus
    ///       `clip_provenance::containment_check_catches_a_one_layer_subject_slip`
    ///       for the frame-slip variant.
    ///   (c) a moat sweep — every relusplitter row has ground truth, so assert zero
    ///       verdicts contradicting it and in particular no GT-`sat` row returning
    ///       `unsat`.
    ///       NOT LANDED. This is the gate that keeps the predicate `false`: it is a
    ///       MEASURED gate, not a unit test, and nothing in this repository can
    ///       discharge it. Until a sweep exists, the conversion below is code that
    ///       is proven correct in the small and unproven at benchmark scale.
    ///
    /// Flipping it ALSO retires an entry from the CLI preset/engine contract
    /// registry (`ny-cli` `preset::contract`, field `bab.clip.interm_domain`) and
    /// from the shipped-preset allow-list in `preset::contract_tests`; both read
    /// this predicate through `ny_propagate::sequential_clip_interm_domain_supported`,
    /// so they follow automatically but their allow-list entries must be deleted.
    #[test]
    fn sequential_clip_interm_domain_stays_quarantined() {
        assert!(
            !sequential_clip_interm_domain_supported(),
            "sequential clip_interm_domain must stay quarantined until the \
             provenance-preserving backward conversion exists; see this test's docs \
             for the three gates required to flip it",
        );
    }

    /// TEETH for the BaB-unblocking fix. While the capability is unsupported, the
    /// adapter must SKIP the optional tightening — returning the caller's bounds
    /// unchanged — and must NOT return an error.
    ///
    /// Reverting the fix (restoring `Err(NyError::SoundnessRefusal(..))`) fails this
    /// test at `expect`. Subtly breaking it — returning bounds that are not the
    /// caller's, e.g. an empty vector, a reordered vector, or silently widened or
    /// narrowed tensors — fails the length and element-wise identity assertions.
    /// Narrowing in particular MUST fail: an unbacked tightening applied under a
    /// quarantine is exactly the false-`unsat` mechanism the quarantine exists to
    /// prevent.
    #[test]
    fn sequential_clip_interm_skip_is_identity_not_error() {
        let verifier = BetaCrownVerifier::new(Default::default());
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

        // Deliberately ASYMMETRIC and multi-layer so a transposed, reordered, or
        // partially-copied result cannot coincidentally compare equal.
        let layer_bounds = vec![
            Arc::new(
                BoundedTensor::new(
                    arr1(&[-3.0, 0.5, -0.25]).into_dyn(),
                    arr1(&[4.0, 7.0, 0.125]).into_dyn(),
                )
                .unwrap(),
            ),
            Arc::new(
                BoundedTensor::new(arr1(&[-1.5, 2.0]).into_dyn(), arr1(&[9.0, 11.0]).into_dyn())
                    .unwrap(),
            ),
        ];

        let returned = verifier
            .apply_clip_interm_domain(
                &Network::new(),
                &SplitHistory::new(),
                layer_bounds.clone(),
                &IntermediateLinearBounds::empty(),
                &input,
                None,
            )
            .expect(
                "a quarantined clip_interm_domain must SKIP the optional tightening, not fail \
                 the domain — erroring here is what collapsed sequential BaB at the root",
            );

        assert_eq!(
            returned.len(),
            layer_bounds.len(),
            "skip path must return every layer's bounds, unchanged in count",
        );
        for (layer_idx, (got, want)) in returned.iter().zip(layer_bounds.iter()).enumerate() {
            assert_eq!(
                got.lower(),
                want.lower(),
                "layer {layer_idx}: skipped tightening must return the caller's LOWER bounds \
                 byte-for-byte; any change here is an unbacked bound edit under quarantine",
            );
            assert_eq!(
                got.upper(),
                want.upper(),
                "layer {layer_idx}: skipped tightening must return the caller's UPPER bounds \
                 byte-for-byte; any change here is an unbacked bound edit under quarantine",
            );
        }
    }

    /// The skip must be shape-agnostic: an empty bound set is the degenerate case
    /// the old refusal also hit, and it must now succeed rather than error.
    #[test]
    fn sequential_clip_interm_skip_handles_empty_bounds() {
        let verifier = BetaCrownVerifier::new(Default::default());
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
        let returned = verifier
            .apply_clip_interm_domain(
                &Network::new(),
                &SplitHistory::new(),
                Vec::new(),
                &IntermediateLinearBounds::empty(),
                &input,
                None,
            )
            .expect("empty bound set must skip cleanly");
        assert!(returned.is_empty());
    }
}

/// GATE (a) + GATE (b) for the flip: the differential enclosure harness.
///
/// These drive [`BetaCrownVerifier::clip_interm_domain_converted`] — the real
/// conversion, the real Lagrangian-dual solver, the real merge — directly, so
/// they measure the code the flip would enable rather than the quarantined skip.
/// The oracle is a hand-written interval/forward evaluator in this module, kept
/// deliberately independent of the engine's own propagation so a shared bug
/// cannot make both sides agree.
#[cfg(test)]
mod enclosure_tests {
    use ndarray::{arr1, Array1, Array2};
    use ny_tensor::BoundedTensor;
    use std::sync::Arc;

    use super::*;
    use crate::beta_crown::branching::NeuronConstraint;
    use crate::{Layer, LinearLayer, ReLULayer};

    /// Deterministic LCG so the "random splits" are reproducible in CI.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }
        fn next_unit(&mut self) -> f32 {
            (self.next_u32() % 2001) as f32 / 1000.0 - 1.0
        }
    }

    /// `Linear(3->4) -> ReLU -> Linear(4->3) -> ReLU -> Linear(3->2)`.
    /// Layer indices 0..4; ReLUs sit at 1 and 3, so their split subjects are
    /// `LayerOutput(0)` and `LayerOutput(2)`.
    struct Fixture {
        network: Network,
        w1: Array2<f32>,
        b1: Array1<f32>,
        w2: Array2<f32>,
        b2: Array1<f32>,
        w3: Array2<f32>,
        b3: Array1<f32>,
    }

    fn fixture() -> Fixture {
        let mut rng = Lcg(0x5eed_1234);
        let mut sample_matrix = |rows: usize, cols: usize| -> Array2<f32> {
            Array2::from_shape_fn((rows, cols), |_| rng.next_unit())
        };
        let w1 = sample_matrix(4, 3);
        let w2 = sample_matrix(3, 4);
        let w3 = sample_matrix(2, 3);
        let b1 = arr1(&[0.1, -0.2, 0.3, -0.05]);
        let b2 = arr1(&[-0.15, 0.25, 0.05]);
        let b3 = arr1(&[0.2, -0.1]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(w1.clone(), Some(b1.clone())).expect("layer 0"),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(w2.clone(), Some(b2.clone())).expect("layer 2"),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(w3.clone(), Some(b3.clone())).expect("layer 4"),
        ));

        Fixture {
            network,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        }
    }

    impl Fixture {
        /// Independent concrete forward pass: `activations()[k]` is `h_k`.
        fn activations(&self, x: &[f32]) -> Vec<Vec<f32>> {
            let affine = |w: &Array2<f32>, b: &Array1<f32>, v: &[f32]| -> Vec<f32> {
                (0..w.nrows())
                    .map(|i| {
                        let mut acc = b[i];
                        for (j, &value) in v.iter().enumerate() {
                            acc += w[[i, j]] * value;
                        }
                        acc
                    })
                    .collect()
            };
            let relu = |v: &[f32]| -> Vec<f32> { v.iter().map(|value| value.max(0.0)).collect() };

            let h0 = affine(&self.w1, &self.b1, x);
            let h1 = relu(&h0);
            let h2 = affine(&self.w2, &self.b2, &h1);
            let h3 = relu(&h2);
            let h4 = affine(&self.w3, &self.b3, &h3);
            vec![h0, h1, h2, h3, h4]
        }

        /// Independent interval propagation, padded OUTWARD, so the boxes handed
        /// to the clip are honest enclosures without depending on engine code.
        fn root_layer_bounds(&self, input: &BoundedTensor) -> Vec<Arc<BoundedTensor>> {
            const PAD: f32 = 1e-3;
            let flat = input.flatten();
            let mut lower: Vec<f32> = flat.lower().iter().copied().collect();
            let mut upper: Vec<f32> = flat.upper().iter().copied().collect();
            let mut out = Vec::new();

            let affine = |w: &Array2<f32>, b: &Array1<f32>, l: &[f32], u: &[f32]| {
                let mut nl = vec![0.0f32; w.nrows()];
                let mut nu = vec![0.0f32; w.nrows()];
                for i in 0..w.nrows() {
                    let mut lo = b[i];
                    let mut hi = b[i];
                    for j in 0..w.ncols() {
                        let c = w[[i, j]];
                        if c >= 0.0 {
                            lo += c * l[j];
                            hi += c * u[j];
                        } else {
                            lo += c * u[j];
                            hi += c * l[j];
                        }
                    }
                    nl[i] = lo - PAD;
                    nu[i] = hi + PAD;
                }
                (nl, nu)
            };

            for step in 0..5 {
                let (nl, nu) = match step {
                    0 => affine(&self.w1, &self.b1, &lower, &upper),
                    2 => affine(&self.w2, &self.b2, &lower, &upper),
                    4 => affine(&self.w3, &self.b3, &lower, &upper),
                    _ => (
                        lower.iter().map(|v| v.max(0.0)).collect(),
                        upper.iter().map(|v| v.max(0.0)).collect(),
                    ),
                };
                lower = nl;
                upper = nu;
                out.push(Arc::new(
                    BoundedTensor::new(
                        Array1::from(lower.clone()).into_dyn(),
                        Array1::from(upper.clone()).into_dyn(),
                    )
                    .expect("valid oracle box"),
                ));
            }
            out
        }
    }

    fn input_box() -> BoundedTensor {
        BoundedTensor::new(
            arr1(&[-1.0, -1.0, -1.0]).into_dyn(),
            arr1(&[1.0, 1.0, 1.0]).into_dyn(),
        )
        .expect("valid input box")
    }

    /// Apply a split to the layer boxes exactly as `create_child_domain` does:
    /// clamp `layer_bounds[layer_idx - 1][neuron_idx]` at 0.
    fn clamp_for_split(
        layer_bounds: &mut [Arc<BoundedTensor>],
        layer_idx: usize,
        neuron_idx: usize,
        is_active: bool,
    ) {
        let target = layer_idx - 1;
        let flat = layer_bounds[target].flatten();
        let mut lower: Vec<f32> = flat.lower().iter().copied().collect();
        let mut upper: Vec<f32> = flat.upper().iter().copied().collect();
        if is_active {
            lower[neuron_idx] = lower[neuron_idx].max(0.0);
        } else {
            upper[neuron_idx] = upper[neuron_idx].min(0.0);
        }
        layer_bounds[target] = Arc::new(
            BoundedTensor::new(
                Array1::from(lower).into_dyn(),
                Array1::from(upper).into_dyn(),
            )
            .expect("valid clamped box"),
        );
    }

    /// Every returned interval must contain the true activation at every sampled
    /// point of the SPLIT REGION. Returns the number of violations plus the
    /// worst one, so the teeth test can assert the checker actually fires.
    fn enclosure_violations(
        fixture: &Fixture,
        clipped: &[Arc<BoundedTensor>],
        samples: &[Vec<f32>],
    ) -> (usize, f32) {
        // Directed-rounding slack: the solver closes with `next_down`/`next_up`
        // and the oracle evaluates in plain f32, so only violations larger than a
        // few ULP at these magnitudes are real.
        const TOL: f32 = 1e-4;
        let mut violations = 0usize;
        let mut worst = 0.0f32;
        for x in samples {
            let truth = fixture.activations(x);
            for (layer_idx, box_k) in clipped.iter().enumerate() {
                let flat = box_k.flatten();
                let lower: Vec<f32> = flat.lower().iter().copied().collect();
                let upper: Vec<f32> = flat.upper().iter().copied().collect();
                for (neuron_idx, &value) in truth[layer_idx].iter().enumerate() {
                    let low_gap = lower[neuron_idx] - value;
                    let high_gap = value - upper[neuron_idx];
                    let gap = low_gap.max(high_gap);
                    if gap > TOL {
                        violations += 1;
                        worst = worst.max(gap);
                    }
                }
            }
        }
        (violations, worst)
    }

    /// Sample the input box and keep only points inside the split region `S`.
    fn samples_in_split_region(
        fixture: &Fixture,
        history: &SplitHistory,
        count: usize,
        seed: u64,
    ) -> Vec<Vec<f32>> {
        let mut rng = Lcg(seed);
        let mut kept = Vec::new();
        for _ in 0..count {
            let x = vec![rng.next_unit(), rng.next_unit(), rng.next_unit()];
            let acts = fixture.activations(&x);
            let inside = history.constraints.iter().all(|c| {
                // The split subject is layer `layer_idx`'s PRE-activation, i.e.
                // `h_{layer_idx - 1}` — the same one-back indexing the conversion
                // uses. Getting this wrong HERE would weaken the test, not the
                // code, so it is spelled out rather than shared.
                let value = acts[c.layer_idx - 1][c.neuron_idx];
                if c.is_active {
                    value >= 0.0
                } else {
                    value <= 0.0
                }
            });
            if inside {
                kept.push(x);
            }
        }
        kept
    }

    /// GATE (a). Over many random split histories, the converted clip must never
    /// return an interval that excludes a reachable activation.
    #[test]
    fn converted_clip_never_cuts_off_a_reachable_activation() {
        let fixture = fixture();
        let input = input_box();
        let verifier = BetaCrownVerifier::new(Default::default());
        let mut rng = Lcg(0xabcd_0001);

        let mut tightened_somewhere = false;
        let mut regions_checked = 0usize;

        for trial in 0..40u64 {
            let root = fixture.root_layer_bounds(&input);
            let mut layer_bounds = root.clone();
            let mut history = SplitHistory::new();

            let n_splits = 1 + (rng.next_u32() % 3) as usize;
            for _ in 0..n_splits {
                // ReLU layers are at indices 1 and 3.
                let layer_idx = if rng.next_u32().is_multiple_of(2) {
                    1
                } else {
                    3
                };
                let width = layer_bounds[layer_idx - 1].len();
                let neuron_idx = (rng.next_u32() as usize) % width;
                if history.is_constrained(layer_idx, neuron_idx).is_some() {
                    continue;
                }
                let is_active = rng.next_u32().is_multiple_of(2);
                history.add_constraint(
                    NeuronConstraint::new(layer_idx, neuron_idx, is_active, 1.0)
                        .expect("finite score"),
                );
                clamp_for_split(&mut layer_bounds, layer_idx, neuron_idx, is_active);
            }
            if history.depth() == 0 {
                continue;
            }

            let clipped = verifier
                .clip_interm_domain_converted(
                    &fixture.network,
                    &history,
                    layer_bounds.clone(),
                    &input,
                    None,
                )
                .expect("conversion-backed clip must not error");

            // Did it do anything? A harness that silently no-ops proves nothing.
            for (before, after) in layer_bounds.iter().zip(clipped.iter()) {
                let bl = before.flatten();
                let al = after.flatten();
                let lower_tightened = bl
                    .lower()
                    .iter()
                    .zip(al.lower().iter())
                    .any(|(&b_lo, &a_lo)| a_lo > b_lo);
                let upper_tightened = bl
                    .upper()
                    .iter()
                    .zip(al.upper().iter())
                    .any(|(&b_hi, &a_hi)| a_hi < b_hi);
                tightened_somewhere |= lower_tightened || upper_tightened;
            }

            let samples = samples_in_split_region(&fixture, &history, 4000, 0x1000 + trial);
            if samples.is_empty() {
                continue;
            }
            regions_checked += 1;
            let (violations, worst) = enclosure_violations(&fixture, &clipped, &samples);
            assert_eq!(
                violations, 0,
                "trial {trial}: the converted clip EXCLUDED {violations} reachable activations \
                 (worst excursion {worst}); a tightened intermediate bound that cuts off a \
                 reachable point is the false-`unsat` mechanism this gate exists to catch",
            );
        }

        assert!(
            regions_checked >= 10,
            "the enclosure gate needs non-empty split regions to have content; only \
             {regions_checked} were checked",
        );
        assert!(
            tightened_somewhere,
            "the converted clip never tightened ANY bound across 40 split histories — the \
             enclosure assertions above would then pass vacuously",
        );
    }

    /// GATE (b), TEETH. The checker used by gate (a) must fail on a bound that is
    /// one ULP tighter than the TRUE reachable extremum — not merely tighter than
    /// some already-loose shipped bound. Built by measuring the sampled extremum
    /// and then stepping one ULP inward, so the injected error is defined against
    /// ground truth.
    #[test]
    fn enclosure_check_catches_a_one_ulp_too_tight_conversion() {
        let fixture = fixture();
        let input = input_box();
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint::new(1, 0, true, 1.0).expect("finite score"));

        let samples = samples_in_split_region(&fixture, &history, 4000, 0x2000);
        assert!(!samples.is_empty(), "need a non-empty split region");

        // TRUE reachable extremum of h_0[0] over the sampled split region.
        let mut true_max = f32::NEG_INFINITY;
        for x in &samples {
            true_max = true_max.max(fixture.activations(x)[0][0]);
        }

        let mut boxes = fixture.root_layer_bounds(&input);
        let flat = boxes[0].flatten();
        let lower: Vec<f32> = flat.lower().iter().copied().collect();
        let mut upper: Vec<f32> = flat.upper().iter().copied().collect();

        // Sanity: the honest box must ENCLOSE the true extremum, so the only
        // difference between the two arms below is the injected error.
        assert!(
            upper[0] >= true_max,
            "oracle box must enclose the true maximum before the injection",
        );
        let honest = boxes.clone();
        let (violations, _) = enclosure_violations(&fixture, &honest, &samples);
        assert_eq!(
            violations, 0,
            "the honest oracle box must pass the checker, else the teeth below prove nothing",
        );

        // Inject: one ULP tighter than the truth (scaled past the checker's
        // directed-rounding tolerance so the assertion is about the injection,
        // not about float noise).
        upper[0] = true_max - 1e-3;
        boxes[0] = Arc::new(
            BoundedTensor::new(
                Array1::from(lower).into_dyn(),
                Array1::from(upper).into_dyn(),
            )
            .expect("valid injected box"),
        );

        let (violations, worst) = enclosure_violations(&fixture, &boxes, &samples);
        assert!(
            violations > 0 && worst > 0.0,
            "the enclosure checker MISSED a bound tightened past the true reachable maximum; \
             gate (a) would then be decorative",
        );
    }

    /// A history whose split subject cannot be converted must leave the caller's
    /// bounds alone rather than tighten on nothing. Uses an empty network, where
    /// every conversion refuses.
    #[test]
    fn unconvertible_history_leaves_bounds_untouched() {
        let verifier = BetaCrownVerifier::new(Default::default());
        let input = input_box();
        let layer_bounds = vec![Arc::new(
            BoundedTensor::new(arr1(&[-2.0, -1.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn())
                .expect("valid box"),
        )];
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint::new(1, 0, true, 1.0).expect("finite score"));

        let returned = verifier
            .clip_interm_domain_converted(
                &Network::new(),
                &history,
                layer_bounds.clone(),
                &input,
                None,
            )
            .expect("an unconvertible history must skip, not error");
        assert_eq!(returned.len(), layer_bounds.len());
        assert_eq!(returned[0].lower(), layer_bounds[0].lower());
        assert_eq!(returned[0].upper(), layer_bounds[0].upper());
    }
}
