// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#cgan-unstable-rows`: refine ONLY a target's baseline-unstable rows.
//!
//! alpha-beta-CROWN bounds each pre-ReLU node over the rows whose reference
//! interval straddles zero and keeps the reference bound verbatim everywhere
//! else (`auto_LiRPA/backward_bound.py:568-604` builds the one-hot `C` over
//! `get_unstable_locations`' index set, `:608-645` scatters the result back).
//! ny already owns both halves of that mechanism — the selector in
//! [`super::demand`] and the k-row backward in [`super::target_backward`] — but
//! they were reachable only through a raw environment variable. This module is
//! the typed home for the pair: one selector wrapper that carries the abc
//! `minimum_sparsity` ceiling, and one kernel that turns a row list into a
//! shrink-only refinement of an existing certified enclosure.
//!
//! SOUNDNESS. The kernel never produces a bound; it INTERSECTS. Unselected
//! rows are copied from `base` verbatim, selected rows are
//! `base ∩ CROWN_row` where `CROWN_row` comes from the same
//! `run_target_backward_pass` the full-width walk uses, seeded with an exact
//! identity row. Any disjoint element rejects the whole target, and the caller
//! keeps `base`. Row selection therefore changes only WHICH rows are
//! recomputed, never the value written to a row that is not: a wrong mask
//! yields a LOOSER map, never an unsound one.

use std::collections::HashMap;
use std::time::Instant;

use ny_core::NyError;
use ny_tensor::BoundedTensor;

use super::demand::{
    relu_only_intermediate_target_is_eligible, sparse_relu_row_plan_for_target_with_max_permille,
    sparse_relu_threshold_allows, SparseReluRowPlan,
};
use crate::network::core::GraphNetwork;
use crate::output_margin_seed::scatter_margin_rows_over_bounds;

/// Outcome of the typed unstable-row selector for ONE demanded target.
///
/// The decline variants exist so a null reads as "this target declined" rather
/// than "sparse rows were measured and worthless" — the collector counts each
/// one into its `[unstable-rows] root …` summary.
/// `pub(super)`, not `pub(crate)`: the variant carries a `SparseReluRowPlan`,
/// which is private to this module tree. The child-arm increment that wants it
/// outside `bounds` widens both together.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum RowSelection {
    /// A usable plan. `Rows` is never empty: an empty selection is `AllStable`.
    Rows(SparseReluRowPlan),
    /// Every row is sign-stable under the baseline, so no CROWN walk can move
    /// any ReLU decision this target feeds.
    AllStable,
    /// More than `max_frac` of the rows are unstable — abc's `minimum_sparsity`
    /// fallback to the full-width identity seed
    /// (`backward_bound.py:580-589`). `k` and `dim` are reported so a decline
    /// null can be told apart from a cost null.
    AboveMaxFrac { k: usize, dim: usize },
    /// The target has a bound-demanding consumer that is not a ReLU (or is the
    /// collection output), so a sign-stable row's relaxation is NOT
    /// magnitude-independent and omitting it is not admissible.
    NotReluOnly,
}

/// Outcome of one k-row refinement over one box.
#[derive(Debug)]
pub(crate) enum RowRefineOutcome {
    /// `bounds` is elementwise no wider than the `base` handed in.
    Refined {
        /// Boxed: a `BoundedTensor` made `Refined` 280 bytes against 56
        /// (clippy::large_enum_variant, workspace threshold 64).
        bounds: Box<BoundedTensor>,
        k: usize,
        secs: f64,
    },
    /// The shrink-only intersection found `count` disjoint elements. Both
    /// inputs are supposed to be valid enclosures, so this is a numerical
    /// anomaly: the caller keeps `base`.
    Disjoint { count: usize },
    /// Refused before any work. The caller keeps `base` (and may fall open to
    /// its existing full-width route).
    Declined(&'static str),
    /// The backward walk or the scatter failed (deadline included).
    Failed(NyError),
}

/// Everything the kernel needs to refine `rows` of `target` over one box.
pub(crate) struct RowRefineRequest<'a> {
    pub(crate) input: &'a BoundedTensor,
    pub(crate) target: &'a str,
    /// Already-finalised CROWN bounds for this target's ancestors.
    pub(crate) crown_bounds: &'a HashMap<String, BoundedTensor>,
    /// The certified baseline MAP the backward walk reads for its ancestors'
    /// relaxations: the original one-shot map, exactly as the historical
    /// collector passed it.
    pub(crate) ibp_bounds: &'a HashMap<String, BoundedTensor>,
    /// The enclosure the scatter writes over, threaded EXPLICITLY instead of
    /// looked up as `ibp_bounds[target]`.
    ///
    /// The two are NOT the same box. Under `#crown-ibp-downstream-resweep` the
    /// collector composes already-finalised predecessor tightening through this
    /// node and publishes over that composed box (`crown_tighten.rs`: `let
    /// ibp_bound = composed_ibp_bound.as_ref().unwrap_or(original_ibp_bound)`),
    /// while the walk keeps reading the original map. A lookup here would
    /// silently publish over the PRE-composition box whenever that gate is
    /// armed, so the caller states its publication intent rather than having
    /// the kernel guess it.
    pub(crate) base: &'a BoundedTensor,
    pub(crate) engine: Option<&'a dyn ny_core::GemmEngine>,
    pub(crate) rows: &'a [usize],
    pub(crate) per_node_deadline: Option<Instant>,
    pub(crate) deadline_is_hard: bool,
    /// Forwarded to the walk as `collector_patches_override`. Dense cGAN
    /// cascades pass `false`.
    pub(crate) collector_patches_override: bool,
    pub(crate) label: &'a str,
    pub(crate) cut_ctx: Option<&'a super::target_backward::CrownCutContext>,
    /// Dense-materialisation budget, in bytes, for the `[k x dim]` seed pair.
    /// Only consulted when `enforce_dense_budget` is set.
    pub(crate) dense_budget_bytes: usize,
    /// Apply the dense-seed budget refusal at all.
    ///
    /// The collector's planner guards that refusal with `subset_is_sparse`
    /// (`crown_tighten::resolve_subset_indices`): a margin/cone subset is
    /// routed by other rules entirely and has never been budget-refused. The
    /// caller therefore passes its own `subset_is_sparse`, which makes this
    /// precheck a re-statement of a decision already taken with the identical
    /// inputs — it can only agree. Applying it unconditionally would refuse
    /// subsets the historical collector ran, i.e. the typed key would not be
    /// dark.
    pub(crate) enforce_dense_budget: bool,
    /// Would this target's seed be a VIRTUAL patches identity rather than a
    /// materialised `[k x dim]` pair?
    ///
    /// `subset_seed_fits_dense_budget` short-circuits on it, so it is the
    /// difference between agreeing with the planner and refusing a target the
    /// planner admitted. Pass the collector's real `is_patches_target`.
    pub(crate) patches_seed: bool,
    /// Apply the shrink-only intersection with `base` inside the kernel.
    ///
    /// `false` reproduces the historical collector exactly: the scattered rows
    /// are handed back raw and the caller's own publication step performs the
    /// intersection. `true` is for shrink-only typed transactions, where the
    /// extra intersection is idempotent with that step
    /// (`base ∩ (base ∩ crown) = base ∩ crown`).
    pub(crate) shrink_only: bool,
}

impl GraphNetwork {
    /// Typed wrapper over the existing unstable-row selector.
    ///
    /// This is deliberately NOT a second implementation: it calls
    /// [`sparse_relu_row_plan_for_target_with_max_permille`], which owns both
    /// the structural eligibility predicate and the row predicate. The only
    /// thing added here is the classification of a decline.
    pub(super) fn select_unstable_rows(
        &self,
        target: &str,
        output: Option<&str>,
        base: &BoundedTensor,
        max_permille: u16,
    ) -> RowSelection {
        if !relu_only_intermediate_target_is_eligible(self, target, output) {
            return RowSelection::NotReluOnly;
        }
        match sparse_relu_row_plan_for_target_with_max_permille(
            self,
            target,
            output,
            base,
            max_permille,
        ) {
            Some(plan) if plan.is_all_stable() => RowSelection::AllStable,
            Some(plan) => RowSelection::Rows(plan),
            None => {
                // Structural eligibility already passed, so the only remaining
                // refusal is the `minimum_sparsity` ceiling. Recompute the
                // count from the same predicate rather than guessing.
                let dim = base.len();
                let k = super::demand::relu_stability_profile(base).retained();
                debug_assert!(!sparse_relu_threshold_allows(k, dim, max_permille));
                RowSelection::AboveMaxFrac { k, dim }
            }
        }
    }

    /// Refine exactly `rows` of `target` by certified CROWN backward and write
    /// the result back over the baseline enclosure.
    ///
    /// Order of operations mirrors the collector's historical inline pair:
    /// dense-budget check, k-row identity backward, scatter over the baseline,
    /// optional shrink-only intersection.
    pub(crate) fn refine_rows_over_box(&self, request: RowRefineRequest<'_>) -> RowRefineOutcome {
        let base = request.base;
        // Unreachable on every historical route, and deliberately so: the
        // collector's own empty guard `continue`s a sparse plan before it gets
        // here, `margin_subset_indices_from_published` rejects an empty
        // publication, and the influence cone returns `None` rather than an
        // empty row list. Kept as a precondition, not as a behaviour.
        if request.rows.is_empty() {
            return RowRefineOutcome::Declined("no-rows");
        }
        // Refuse BEFORE the walk rather than letting the `[k x dim]` seed pair
        // silently overrun the dense envelope — but refuse on EXACTLY the
        // planner's terms (`crown_tighten::resolve_subset_indices`): the rule
        // is scoped to sparse-ReLU routes and short-circuits on a virtual
        // patches seed. Same predicate, same budget, same inputs, so this can
        // only agree with the routing decision already taken.
        if request.enforce_dense_budget
            && !super::crown_tighten::subset_seed_fits_dense_budget(
                request.patches_seed,
                request.rows.len(),
                base.len(),
                request.dense_budget_bytes,
            )
        {
            return RowRefineOutcome::Declined("dense-budget");
        }

        let started = Instant::now();
        let walked = self.propagate_crown_to_node_subset(
            request.input,
            request.target,
            request.crown_bounds,
            request.ibp_bounds,
            request.engine,
            request.label,
            request.per_node_deadline,
            request.deadline_is_hard,
            request.collector_patches_override,
            request.rows,
            request.cut_ctx,
        );
        let secs = started.elapsed().as_secs_f64();
        let (lower_rows, upper_rows) = match walked {
            Ok(rows) => rows,
            Err(error) => return RowRefineOutcome::Failed(error),
        };
        let scattered =
            match scatter_margin_rows_over_bounds(base, request.rows, &lower_rows, &upper_rows) {
                Ok(bounds) => bounds,
                Err(error) => return RowRefineOutcome::Failed(error),
            };
        if !request.shrink_only {
            return RowRefineOutcome::Refined {
                bounds: Box::new(scattered),
                k: request.rows.len(),
                secs,
            };
        }
        // Shrink-only write-back, identical in contract to the collector's
        // `retain_partial_crown_rows`: a disjoint element is a numerical
        // anomaly and rejects the WHOLE target, because the sound diagnostic
        // union `intersection_per_element` returns for such an element can be
        // LOOSER than the certified baseline.
        match base.intersection_per_element(&scattered) {
            Some((tightened, 0)) => RowRefineOutcome::Refined {
                bounds: Box::new(tightened),
                k: request.rows.len(),
                secs,
            },
            Some((_diagnostic_union, count)) => RowRefineOutcome::Disjoint { count },
            None => RowRefineOutcome::Declined("intersection-unavailable"),
        }
    }
}

/// Scheduling weight for a target that received a TYPED sparse row plan.
///
/// The per-node time slice is proportional to a target's demanded row count
/// (`crown_tighten.rs` weighted budget). A sparse plan shrinks that count by
/// `dim/k` while the ancestor PREFIX the walk still traverses is unchanged, so
/// pricing a typed sparse target at `k` starves it into exactly the
/// `PerNodeDeadlineExceeded` this mechanism exists to remove. The share is
/// re-read from the remaining window at every node, so an over-allocated slice
/// that finishes early returns to the pool for free; an under-allocated one
/// does not.
///
/// Env-armed and atomic (`CganSparseTargetComplete`) plans keep their
/// historical `selected_len()` weight, so their frontier rows do not move.
pub(crate) const fn typed_sparse_schedule_rows(
    typed_plan: bool,
    subset_len: usize,
    full_len: usize,
) -> usize {
    if typed_plan {
        full_len
    } else {
        subset_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::core::GraphNode;
    use ndarray::{arr1, arr2};

    /// Tiny sequential Linear/ReLU chain whose `producer` node has two
    /// sign-UNSTABLE rows and two sign-stable ones under IBP, so a row plan is
    /// a strict, non-empty subset.
    fn mixed_stability_net() -> (GraphNetwork, BoundedTensor) {
        let pre = LinearLayer::new(arr2(&[[1.0_f32], [-1.0]]), None).expect("pre");
        let producer = LinearLayer::new(
            arr2(&[[1.0_f32, -1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            Some(arr1(&[0.0_f32, -0.5, 1.0, -3.0])),
        )
        .expect("producer");
        let out = LinearLayer::new(
            arr2(&[[1.0_f32, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]),
            None,
        )
        .expect("out");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("pre", Layer::Linear(pre)));
        graph.add_node(GraphNode::new(
            "act1",
            Layer::ReLU(ReLULayer),
            vec!["pre".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "producer",
            Layer::Linear(producer),
            vec!["act1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "act2",
            Layer::ReLU(ReLULayer),
            vec!["producer".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(out),
            vec!["act2".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("input");
        (graph, input)
    }

    fn request<'a>(
        input: &'a BoundedTensor,
        ibp: &'a HashMap<String, BoundedTensor>,
        crown: &'a HashMap<String, BoundedTensor>,
        rows: &'a [usize],
        shrink_only: bool,
        dense_budget_bytes: usize,
    ) -> RowRefineRequest<'a> {
        // The historical sparse route: the budget is enforced, and the seed is
        // a materialised dense pair.
        request_over_base(
            input,
            ibp.get("producer").expect("producer baseline"),
            ibp,
            crown,
            rows,
            shrink_only,
            dense_budget_bytes,
            true,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn request_over_base<'a>(
        input: &'a BoundedTensor,
        base: &'a BoundedTensor,
        ibp: &'a HashMap<String, BoundedTensor>,
        crown: &'a HashMap<String, BoundedTensor>,
        rows: &'a [usize],
        shrink_only: bool,
        dense_budget_bytes: usize,
        enforce_dense_budget: bool,
        patches_seed: bool,
    ) -> RowRefineRequest<'a> {
        RowRefineRequest {
            input,
            target: "producer",
            crown_bounds: crown,
            ibp_bounds: ibp,
            base,
            engine: None,
            rows,
            per_node_deadline: None,
            deadline_is_hard: false,
            collector_patches_override: false,
            label: "row-refine-test",
            cut_ctx: None,
            dense_budget_bytes,
            enforce_dense_budget,
            patches_seed,
            shrink_only,
        }
    }

    fn refined(outcome: RowRefineOutcome) -> BoundedTensor {
        match outcome {
            RowRefineOutcome::Refined { bounds, .. } => *bounds,
            other => panic!("expected a refinement, got {other:?}"),
        }
    }

    /// The selector reports the four dispositions the collector's decline
    /// census counts, and `Rows` carries exactly the unstable rows.
    #[test]
    fn select_unstable_rows_classifies_every_disposition() {
        let (graph, input) = mixed_stability_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let producer = ibp.get("producer").expect("producer baseline");

        let RowSelection::Rows(plan) =
            graph.select_unstable_rows("producer", Some("out"), producer, 900)
        else {
            panic!("expected a strict non-empty subset");
        };
        assert_eq!(plan.selected_rows(), &[0, 1]);
        assert_eq!(plan.total_rows(), 4);

        // The collection OUTPUT is never row-sparse: its rows are the proof
        // objective, not a ReLU sign decision.
        assert_eq!(
            graph.select_unstable_rows("out", Some("out"), ibp.get("out").expect("out"), 900),
            RowSelection::NotReluOnly
        );

        // A ceiling below the target's own unstable fraction (2/4 = 500 per
        // mille) declines it to the full-width walk — abc's `minimum_sparsity`
        // fallback — and says so with counts.
        assert_eq!(
            graph.select_unstable_rows("producer", Some("out"), producer, 400),
            RowSelection::AboveMaxFrac { k: 2, dim: 4 }
        );
    }

    /// SOUNDNESS CORE: every element the kernel writes is elementwise within
    /// the certified baseline, unselected rows are copied verbatim, and the
    /// selected rows agree with the full-row certified walk.
    #[test]
    fn refine_rows_is_within_the_certified_full_row_result() {
        let (graph, input) = mixed_stability_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let crown: HashMap<String, BoundedTensor> = HashMap::new();
        let base = ibp.get("producer").expect("producer baseline").clone();

        let all_rows: Vec<usize> = (0..base.len()).collect();
        let full = refined(graph.refine_rows_over_box(request(
            &input,
            &ibp,
            &crown,
            &all_rows,
            true,
            usize::MAX,
        )));
        let sparse = refined(graph.refine_rows_over_box(request(
            &input,
            &ibp,
            &crown,
            &[0, 1],
            true,
            usize::MAX,
        )));

        for row in 0..base.len() {
            // Shrink-only against the certified baseline, on EVERY row.
            assert!(
                sparse.lower()[[row]] >= base.lower()[[row]]
                    && sparse.upper()[[row]] <= base.upper()[[row]],
                "row {row} widened the certified baseline"
            );
        }
        for row in [0usize, 1] {
            // Selected rows: the same value the full-row certified walk stores.
            assert_eq!(
                sparse.lower()[[row]],
                full.lower()[[row]],
                "row {row} lower"
            );
            assert_eq!(
                sparse.upper()[[row]],
                full.upper()[[row]],
                "row {row} upper"
            );
        }
        for row in [2usize, 3] {
            // Unselected rows: the certified baseline, copied verbatim.
            assert_eq!(
                sparse.lower()[[row]],
                base.lower()[[row]],
                "row {row} lower"
            );
            assert_eq!(
                sparse.upper()[[row]],
                base.upper()[[row]],
                "row {row} upper"
            );
        }
    }

    /// `shrink_only=false` is the historical collector contract (the caller's
    /// own publication step intersects), so the raw scatter travels back.
    /// It must still never be published without that intersection, which is
    /// why the collector passes `true` on the typed route.
    #[test]
    fn shrink_only_is_idempotent_with_a_second_baseline_intersection() {
        let (graph, input) = mixed_stability_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let crown: HashMap<String, BoundedTensor> = HashMap::new();
        let base = ibp.get("producer").expect("producer baseline").clone();

        let raw = refined(graph.refine_rows_over_box(request(
            &input,
            &ibp,
            &crown,
            &[0, 1],
            false,
            usize::MAX,
        )));
        let shrunk = refined(graph.refine_rows_over_box(request(
            &input,
            &ibp,
            &crown,
            &[0, 1],
            true,
            usize::MAX,
        )));
        let (raw_then_base, disjoint) = base
            .intersection_per_element(&raw)
            .expect("baseline intersection");
        assert_eq!(disjoint, 0);
        assert_eq!(raw_then_base.lower(), shrunk.lower());
        assert_eq!(raw_then_base.upper(), shrunk.upper());
    }

    /// A `[k x dim]` seed pair over the dense envelope is REFUSED before any
    /// work, so the caller keeps its baseline and can fall open to the
    /// existing full-width/chunked route. A decline null is not a cost null.
    #[test]
    fn an_over_budget_row_set_is_declined_before_the_walk() {
        let (graph, input) = mixed_stability_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let crown: HashMap<String, BoundedTensor> = HashMap::new();
        assert!(matches!(
            graph.refine_rows_over_box(request(&input, &ibp, &crown, &[0, 1], true, 1)),
            RowRefineOutcome::Declined("dense-budget")
        ));
        assert!(matches!(
            graph.refine_rows_over_box(request(&input, &ibp, &crown, &[], true, usize::MAX)),
            RowRefineOutcome::Declined("no-rows")
        ));
    }

    /// DARK GUARANTEE. The dense-seed refusal belongs to the SPARSE route
    /// only, and short-circuits on a virtual patches seed — exactly the two
    /// scopes `crown_tighten::resolve_subset_indices` applies. A margin/cone
    /// subset (`enforce_dense_budget=false`) must therefore be bit-identical
    /// under a 1-byte budget and under an unbounded one: with the key dark the
    /// kernel may not refuse anything the historical collector ran.
    #[test]
    fn the_dense_budget_is_enforced_only_where_the_planner_enforces_it() {
        let (graph, input) = mixed_stability_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let crown: HashMap<String, BoundedTensor> = HashMap::new();
        let base = ibp.get("producer").expect("producer baseline").clone();

        // Non-sparse subset route, 1-byte budget: NOT refused.
        let unbudgeted = refined(graph.refine_rows_over_box(request_over_base(
            &input,
            &base,
            &ibp,
            &crown,
            &[0, 1],
            false,
            1,
            false,
            false,
        )));
        let unbounded = refined(graph.refine_rows_over_box(request_over_base(
            &input,
            &base,
            &ibp,
            &crown,
            &[0, 1],
            false,
            usize::MAX,
            false,
            false,
        )));
        assert_eq!(unbudgeted.lower(), unbounded.lower());
        assert_eq!(unbudgeted.upper(), unbounded.upper());

        // Sparse route, same 1-byte budget: refused before the walk.
        assert!(matches!(
            graph.refine_rows_over_box(request_over_base(
                &input,
                &base,
                &ibp,
                &crown,
                &[0, 1],
                false,
                1,
                true,
                false,
            )),
            RowRefineOutcome::Declined("dense-budget")
        ));

        // Sparse route whose full-width fallback would start in PATCHES: the
        // seed is virtual, so the same 1-byte budget does not refuse it. A
        // hard-coded `patches_seed=false` fails this case.
        let patches = refined(graph.refine_rows_over_box(request_over_base(
            &input,
            &base,
            &ibp,
            &crown,
            &[0, 1],
            false,
            1,
            true,
            true,
        )));
        assert_eq!(patches.lower(), unbounded.lower());
        assert_eq!(patches.upper(), unbounded.upper());
    }

    /// The scatter writes over the baseline the CALLER threads in, not over
    /// `ibp_bounds[target]`. `#crown-ibp-downstream-resweep` publishes over a
    /// composed box that is strictly tighter than the original map entry, and
    /// a lookup here would silently discard that composition.
    #[test]
    fn the_write_back_baseline_is_the_caller_threaded_box() {
        let (graph, input) = mixed_stability_net();
        let ibp = graph.collect_node_bounds(&input).expect("IBP");
        let crown: HashMap<String, BoundedTensor> = HashMap::new();
        let map_entry = ibp.get("producer").expect("producer baseline").clone();

        // Stand in for a downstream-reswept composition: strictly tighter than
        // the map entry on the UNSELECTED rows.
        let mut lower = map_entry.lower().clone();
        let mut upper = map_entry.upper().clone();
        for row in [2usize, 3] {
            lower[[row]] += 0.125;
            upper[[row]] -= 0.125;
        }
        let composed = BoundedTensor::new(lower, upper).expect("composed box");

        let refined_bounds = refined(graph.refine_rows_over_box(request_over_base(
            &input,
            &composed,
            &ibp,
            &crown,
            &[0, 1],
            true,
            usize::MAX,
            true,
            false,
        )));
        for row in [2usize, 3] {
            assert_eq!(
                refined_bounds.lower()[[row]],
                composed.lower()[[row]],
                "row {row} must keep the composed box, not the map entry"
            );
            assert_eq!(refined_bounds.upper()[[row]], composed.upper()[[row]]);
            assert_ne!(refined_bounds.lower()[[row]], map_entry.lower()[[row]]);
        }
    }

    /// The typed schedule weight is the FULL row count, not `k`. Pricing a
    /// typed sparse target at `k` shrinks its deadline by dim/k while the
    /// ancestor prefix it walks is unchanged, which manufactures the
    /// PerNodeDeadlineExceeded this mechanism removes.
    #[test]
    fn typed_sparse_plans_keep_the_full_row_schedule_weight() {
        // The measured 64-px cGAN case: BatchNormalization_13, 1090 unstable
        // rows of 61,504.
        assert_eq!(typed_sparse_schedule_rows(true, 1_090, 61_504), 61_504);
        // Env-armed and atomic plans keep their historical weight, so their
        // frontier rows do not move.
        assert_eq!(typed_sparse_schedule_rows(false, 1_090, 61_504), 1_090);
        assert_eq!(typed_sparse_schedule_rows(true, 0, 0), 0);
    }
}
