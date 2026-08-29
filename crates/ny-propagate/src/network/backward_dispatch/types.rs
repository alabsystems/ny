// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Types for backward CROWN dispatch.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use ndarray::Array1;

use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::MulBinaryRelaxationMode;

/// Borrowed view over a node-bounds map whose values are either owned
/// `BoundedTensor`s or `Arc`-shared ones (#cone-delta increment 2: the
/// constrained/batched BaB caches share out-of-cone entries with the parent
/// domain via `Arc`). One dispatch surface serves both worlds without
/// materializing a converted map.
#[derive(Clone, Copy)]
pub(crate) enum NodeBoundsView<'a> {
    /// Map of owned tensors (graph-CROWN / graph-alpha / sequential callers).
    Plain(&'a HashMap<String, BoundedTensor>),
    /// Map of `Arc`-shared tensors (constrained / batched BaB callers).
    Shared(&'a HashMap<String, Arc<BoundedTensor>>),
}

impl<'a> NodeBoundsView<'a> {
    /// Look up a node's bounds by name.
    pub(crate) fn get(&self, name: &str) -> Option<&'a BoundedTensor> {
        match self {
            Self::Plain(m) => m.get(name),
            Self::Shared(m) => m.get(name).map(|a| a.as_ref()),
        }
    }
}

impl<'a> From<&'a HashMap<String, BoundedTensor>> for NodeBoundsView<'a> {
    fn from(m: &'a HashMap<String, BoundedTensor>) -> Self {
        Self::Plain(m)
    }
}

impl<'a> From<&'a HashMap<String, Arc<BoundedTensor>>> for NodeBoundsView<'a> {
    fn from(m: &'a HashMap<String, Arc<BoundedTensor>>) -> Self {
        Self::Shared(m)
    }
}

/// Context needed for backward dispatch through a single node.
pub(crate) struct DispatchContext<'a> {
    /// Name of the node being processed (for error messages).
    pub node_name: &'a str,
    /// The layer to dispatch through.
    pub layer: &'a Layer,
    /// Input node names for this node.
    pub inputs: &'a [String],
    /// Pre-activation bounds for the first input.
    pub pre_activation: &'a BoundedTensor,
    /// Network input bounds (for resolving `_input` references).
    pub network_input: &'a BoundedTensor,
    /// Cached node bounds (for resolving multi-input node references).
    pub node_bounds: NodeBoundsView<'a>,
    /// Optional GEMM engine for accelerated linear layer propagation.
    pub engine: Option<&'a dyn GemmEngine>,
    /// Optional wall-clock deadline for deadline-aware backward kernels.
    pub deadline: Option<Instant>,
    /// Optional bilinear alpha parameters for McCormick interpolation (#3287).
    /// Maps node name → [4, m, n, k] alpha array. When present and a
    /// BilinearCrown node is found in the map, uses `interpolated_mccormick`
    /// with optimizable r_l/r_u instead of the fixed midpoint heuristic.
    pub bilinear_alphas: Option<&'a HashMap<String, ndarray::Array4<f32>>>,
    /// Relaxation mode for MulBinary CROWN backward (#3439).
    /// Controls which McCormick envelope selection strategy to use for
    /// element-wise multiplication layers. Used when no alpha is available.
    pub mul_binary_relaxation: MulBinaryRelaxationMode,
    /// Optional MulBinary alpha parameters for interpolated McCormick (#3439).
    /// Maps node name → [2, n] alpha array where row 0 = r_l, row 1 = r_u.
    /// When present and a MulBinary node is found in the map, uses
    /// `propagate_linear_binary_with_alpha` with optimizable interpolation.
    pub mul_binary_alphas: Option<&'a HashMap<String, ndarray::Array2<f32>>>,
    /// Optional per-(node, group) `inv_rms` range override for GenBaB norm
    /// branching (#norm-genbab). Maps a `Layer::RmsNorm` node name → a vector of
    /// per-normalization-group windows (indexed by batch row); `None` leaves a
    /// group unconstrained. When present for the node being dispatched, the
    /// decomposed RmsNorm CROWN backward INTERSECTS each group's IBP-derived
    /// `inv_rms` interval with its window (never widening — see `InvRmsOverride`),
    /// tightening the reciprocal/sqrt relaxation so it survives the fused-IBP
    /// validation on this subdomain. Per-group (not per-node) windows are
    /// required for soundness; see `InvRmsOverride`.
    pub norm_inv_rms_override: Option<&'a HashMap<String, Vec<Option<(f32, f32)>>>>,
}

/// Result of dispatching a single node's backward CROWN propagation.
///
/// Each variant tells the caller how to distribute the propagated linear
/// bounds to the node's input(s). The caller handles accumulation and
/// fallback logic.
#[derive(Debug)]
pub(crate) enum BackwardDispatchResult {
    /// Propagated bounds to accumulate to `node.inputs[0]`.
    Single(Box<LinearBounds>),
    /// Two sets of A-matrix bounds for `node.inputs[0]` and `node.inputs[1]`,
    /// plus a single bias pair accumulated exactly once by the engine.
    ///
    /// # Separate bias channel (#2617, #2530)
    ///
    /// Each `LinearBounds` in `bounds_a`/`bounds_b` MUST have `lower_b = upper_b = 0`.
    /// All bias goes through `bias_lower`/`bias_upper`. The engine accumulates
    /// bias exactly once, eliminating the class of bias-splitting bugs (#2520,
    /// #2527, #2529, #2530).
    ///
    /// Reference: alpha-beta-CROWN `backward_bound.py:253-296` — engine
    /// accumulates bias once via `lb = lb + lower_b`.
    Binary {
        bounds_a: Box<LinearBounds>,
        bounds_b: Box<LinearBounds>,
        bias_lower: Array1<f32>,
        bias_upper: Array1<f32>,
    },
    /// N sets of A-matrix bounds in ORIGINAL ONNX input order (`None` =
    /// constant/skipped input), plus a single bias pair accumulated exactly
    /// once by the engine.
    ///
    /// # Separate bias channel (#2617, #2530)
    ///
    /// Each `LinearBounds` MUST have `lower_b = upper_b = 0`.
    /// All bias goes through `bias_lower`/`bias_upper`.
    ///
    /// # Pairing (#nary-zip)
    ///
    /// The slot list is ONNX-ordered while `node.inputs` is dynamic-only, so
    /// the two MUST NOT be paired positionally. [`NaryInputBounds`] owns that
    /// pairing and makes the wrong one unrepresentable.
    Nary {
        bounds: NaryInputBounds,
        bias_lower: Array1<f32>,
        bias_upper: Array1<f32>,
    },
    /// Forward `node_lb` directly to `node.inputs[0]` unchanged (SkipMerge).
    PassThrough,
    /// Layer doesn't support CROWN backward; caller decides fallback.
    Unsupported(String),
}

/// ONNX-ordered per-input A-matrix bounds for an N-ary node, paired with the
/// node's dynamic input edges by the ONLY method that can see inside.
///
/// # Why this is a newtype and not a bare `Vec<Option<LinearBounds>>`
///
/// The slots are indexed by ORIGINAL ONNX input order, with `None` marking an
/// embedded-constant position whose `W_i · c_i` contribution has already been
/// folded into the separate bias channel. A graph node's `node.inputs`, by
/// contrast, lists ONLY the DYNAMIC edges: `ny-build`'s
/// `find_graph_input_nodes` filters every `weights` / `evaluated_constants`
/// name out of the list *before* the Concat keep-branch, and
/// `network/core/graph/ibp/dispatch.rs` states the convention verbatim
/// ("`node.inputs` only contains names for the dynamic (non-constant)
/// inputs").
///
/// Pairing those two lists positionally — `node.inputs.iter().zip(bounds)` —
/// is therefore WRONG whenever a constant sits at an ONNX position before a
/// dynamic one. `zip` (a) hands the dynamic input name the constant's `None`
/// slot, which the `if let Some` skips, and (b) truncates at
/// `node.inputs.len()`, discarding the trailing `Some(LinearBounds)`
/// altogether. The live branch then contributes ZERO coefficients and ZERO
/// bias to the backward walk, and the walk publishes a certified interval that
/// does not contain the true range — the same failure shape as the
/// `try_alpha_backward_gpu_suffix` false proof. Two β-CROWN consumers (the
/// batched and constrained BaB lanes, i.e. the ones that mint scored verdicts)
/// did exactly this on the shipped default path (#nary-zip).
///
/// So the slot vector is PRIVATE. A consumer cannot obtain owned bounds *with
/// the input name they belong to* except through
/// [`NaryInputBounds::pair_with_dynamic_inputs`], which performs the
/// compaction and REFUSES, fail-closed, when the counts disagree. This is a
/// precondition a caller cannot forget, in the spirit of
/// `SoleCarrierWitness(())`.
#[derive(Debug)]
pub(crate) struct NaryInputBounds {
    /// ONNX-ordered slots; `None` = embedded constant already folded into bias.
    slots: Vec<Option<LinearBounds>>,
}

impl NaryInputBounds {
    /// Build from slots indexed by ORIGINAL ONNX input order.
    ///
    /// Every producer of [`BackwardDispatchResult::Nary`] builds its vector in
    /// this order: `dispatch_concat` walks `concat.constant_inputs` (ONNX
    /// order) and emits `None` at each embedded-constant position, while the
    /// OpaqueSkip / AdaIN / self-attention producers emit one `Some` per
    /// dynamic input (no `None`s at all, so ONNX order and dynamic order
    /// coincide for them).
    pub(crate) fn from_onnx_order(slots: Vec<Option<LinearBounds>>) -> Self {
        Self { slots }
    }

    /// Build from bounds that are already one-per-dynamic-input (no embedded
    /// constants). Equivalent to `from_onnx_order` over all-`Some` slots.
    pub(crate) fn all_dynamic(bounds: Vec<LinearBounds>) -> Self {
        Self {
            slots: bounds.into_iter().map(Some).collect(),
        }
    }

    /// Number of slots that carry bounds, i.e. the number of dynamic edges the
    /// producer believes this node has.
    pub(crate) fn dynamic_len(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Borrow the carried bounds in dynamic order, skipping constant slots.
    ///
    /// For pre-publication authentication scans (bias-zero checks) that do not
    /// need the input names. Never use the position of an item yielded here as
    /// an index into anything ONNX-ordered.
    pub(crate) fn iter_dynamic(&self) -> impl Iterator<Item = &LinearBounds> {
        self.slots.iter().flatten()
    }

    /// Pair each carried bound with the dynamic input edge it belongs to.
    ///
    /// Compacts the ONNX-ordered slots onto `inputs` (which is dynamic-only)
    /// and refuses when the counts disagree. A mismatch in either direction
    /// means a live branch would be dropped or a bound misattributed to the
    /// wrong edge, which publishes a non-containing certified interval — so
    /// this is a [`NyError::SoundnessRefusal`], not a silent skip and not an
    /// `UnsupportedOp` that would fall back with the bad bound in hand.
    pub(crate) fn pair_with_dynamic_inputs<'a>(
        self,
        inputs: &'a [String],
        context: &str,
    ) -> Result<Vec<(&'a str, LinearBounds)>> {
        let carried = self.dynamic_len();
        if carried != inputs.len() {
            return Err(NyError::SoundnessRefusal(format!(
                "{context}: n-ary dispatch produced {carried} dynamic bound set(s) over {} ONNX \
                 slot(s) but the node has {} dynamic input edge(s) {:?}. Pairing them would drop \
                 or misattribute a live branch, publishing coefficients that do not cover it.",
                self.slots.len(),
                inputs.len(),
                inputs,
            )));
        }
        Ok(inputs
            .iter()
            .map(String::as_str)
            .zip(self.slots.into_iter().flatten())
            .collect())
    }

    /// Number of ONNX slots, constants included. Only meaningful against
    /// another slot list for the SAME node (e.g. real vs error-carrying
    /// shadow); never against `node.inputs`, which is dynamic-only.
    pub(crate) fn slot_len(&self) -> usize {
        self.slots.len()
    }

    /// Mutably borrow the carried bounds in dynamic order, skipping constant
    /// slots. For in-place error-channel maintenance that needs no names.
    pub(crate) fn iter_dynamic_mut(&mut self) -> impl Iterator<Item = &mut LinearBounds> {
        self.slots.iter_mut().flatten()
    }

    /// Pair this node's slots with the SAME node's shadow slot list by ONNX
    /// slot index, yielding only positions where both carry bounds.
    ///
    /// Sound because both lists come from one dispatch of one node and so
    /// share the ONNX ordering. This is NOT a substitute for
    /// [`Self::pair_with_dynamic_inputs`] — it never touches `node.inputs`.
    pub(crate) fn zip_aligned_slots_mut<'b>(
        &'b mut self,
        shadow: &'b Self,
    ) -> impl Iterator<Item = (&'b mut LinearBounds, &'b LinearBounds)> {
        self.slots
            .iter_mut()
            .zip(shadow.slots.iter())
            .filter_map(|(a, b)| match (a.as_mut(), b.as_ref()) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            })
    }

    /// Read the raw ONNX-ordered slots. Test-only: production code must go
    /// through [`Self::pair_with_dynamic_inputs`].
    #[cfg(test)]
    pub(crate) fn onnx_slots(&self) -> &[Option<LinearBounds>] {
        &self.slots
    }
}

#[cfg(test)]
mod nary_pairing_tests {
    use super::*;
    use ndarray::arr2;

    /// A distinguishable relation: `A = [[k]]`, zero bias.
    fn tagged(k: f32) -> LinearBounds {
        LinearBounds::new(
            arr2(&[[k]]),
            Array1::zeros(1),
            arr2(&[[k]]),
            Array1::zeros(1),
        )
        .expect("tagged relation")
    }

    /// The exact shape that produced the false proof: ONNX `Cx` — an embedded
    /// constant at position 0, a live dynamic branch at position 1, and ONE
    /// dynamic edge in `node.inputs`.
    ///
    /// The positional `zip` this type exists to forbid would pair the dynamic
    /// name with the constant's `None` slot, skip it, and truncate the live
    /// `Some` — publishing zero coefficients for the branch that carries all
    /// the input dependence. Compaction must instead hand the DYNAMIC bound to
    /// the dynamic name.
    #[test]
    fn constant_before_dynamic_pairs_the_live_branch_not_the_hole() {
        let bounds = NaryInputBounds::from_onnx_order(vec![None, Some(tagged(7.0))]);
        let inputs = vec!["patches".to_string()];

        // What the forbidden pairing would have produced, spelled out so the
        // contrast is testable rather than asserted in prose.
        let zip_would_yield: Vec<&str> = inputs
            .iter()
            .map(String::as_str)
            .zip(vec![None, Some(())])
            .filter(|(_, slot)| slot.is_some())
            .map(|(name, _)| name)
            .collect();
        assert!(
            zip_would_yield.is_empty(),
            "fixture is not the defect's shape: a positional zip must drop the branch here"
        );

        let paired = bounds
            .pair_with_dynamic_inputs(&inputs, "test")
            .expect("Cx pattern must pair, not refuse");
        assert_eq!(paired.len(), 1, "the live branch must survive compaction");
        assert_eq!(paired[0].0, "patches");
        assert_eq!(
            paired[0].1.lower_a()[[0, 0]],
            7.0,
            "the dynamic name must receive the DYNAMIC slot's coefficients"
        );
    }

    /// ONNX `xCx`: the compaction must not shift a later branch onto an
    /// earlier name.
    #[test]
    fn interior_constant_does_not_misattribute_later_branches() {
        let bounds =
            NaryInputBounds::from_onnx_order(vec![Some(tagged(1.0)), None, Some(tagged(3.0))]);
        let inputs = vec!["a".to_string(), "b".to_string()];
        let paired = bounds
            .pair_with_dynamic_inputs(&inputs, "test")
            .expect("xCx pattern must pair");
        assert_eq!(paired[0].0, "a");
        assert_eq!(paired[0].1.lower_a()[[0, 0]], 1.0);
        assert_eq!(paired[1].0, "b");
        assert_eq!(
            paired[1].1.lower_a()[[0, 0]],
            3.0,
            "slot 2 belongs to dynamic edge 1, not to edge 0"
        );
    }

    /// Fail-closed: more carried branches than edges means one would be
    /// silently dropped. That is the unsound direction, so it must REFUSE.
    #[test]
    fn more_carried_branches_than_edges_is_refused() {
        let bounds = NaryInputBounds::from_onnx_order(vec![Some(tagged(1.0)), Some(tagged(2.0))]);
        let inputs = vec!["only".to_string()];
        let err = bounds
            .pair_with_dynamic_inputs(&inputs, "test ctx")
            .expect_err("dropping a live branch must be refused, not silently truncated");
        assert!(
            matches!(err, NyError::SoundnessRefusal(_)),
            "must be a SoundnessRefusal (no silent IBP fallback with the bad bound in hand), got {err:?}"
        );
    }

    /// Fail-closed the other way: an edge with no bound contributes zero
    /// coefficients, the same non-containment shape.
    #[test]
    fn more_edges_than_carried_branches_is_refused() {
        let bounds = NaryInputBounds::from_onnx_order(vec![Some(tagged(1.0))]);
        let inputs = vec!["a".to_string(), "b".to_string()];
        let err = bounds
            .pair_with_dynamic_inputs(&inputs, "test ctx")
            .expect_err("an unbounded edge must be refused");
        assert!(matches!(err, NyError::SoundnessRefusal(_)), "got {err:?}");
    }

    /// All-`Some` producers (OpaqueSkip / AdaIN / self-attention) are
    /// unaffected: ONNX order and dynamic order coincide.
    #[test]
    fn all_dynamic_pairs_positionally() {
        let bounds = NaryInputBounds::all_dynamic(vec![tagged(1.0), tagged(2.0), tagged(3.0)]);
        let inputs = vec!["q".to_string(), "k".to_string(), "v".to_string()];
        let paired = bounds.pair_with_dynamic_inputs(&inputs, "test").unwrap();
        let names: Vec<&str> = paired.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["q", "k", "v"]);
        for (i, (_, lb)) in paired.iter().enumerate() {
            assert_eq!(lb.lower_a()[[0, 0]], (i + 1) as f32);
        }
    }
}
