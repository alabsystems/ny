// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! cGAN shared-prefix STACKED backward planner (#cgan-stacked-backward).
//!
//! MEASURED mechanism (docs/CGAN_COLLECTION_CACHE_DEFECTS_2026-08-03.md,
//! docs/CGAN_BOUND_QUALITY_FIX_2026-08-18.md): each demanded cgan target's
//! CROWN backward costs 95-125 s, of which 98.4-99.7 % is the SHARED upstream
//! ConvTranspose generator prefix, and 7-8 targets are demanded per collection
//! — ~700 s of near-duplicate walk inside a 900 s budget. Both scored graphs
//! (`cGAN_imgSz32_nCh_{1,3}.onnx`) are PURE CHAINS (28 nodes, zero fan-out
//! tensors — verified against the ONNX bytes), so every demanded target sits
//! on one trunk and a walk from the deepest target passes through every
//! shallower target's node: stacked seeds never diverge.
//!
//! The lane (env `NY_CGAN_STACKED_BACKWARD`, exact `"1"`, default OFF): plan
//! ONE dense backward walk from the deepest admissible demanded target whose
//! seed carries every stacked member's identity block
//! (`target_backward::StackedSeedInjectionPlan`), so the shared prefix is
//! walked once instead of once per member.
//!
//! WHY the plan does NOT compose with the 28,800-row objective chunking
//! (investigated, deliberately not built): the chunked driver's own contract
//! says "widening changes only how many times the ancestor prefix is
//! re-walked" (`target_backward.rs`, #patches-obj-chunk docs) — cost tracks
//! the number of chunk-walks. A stacked pass split into mixed-target chunks
//! conserves the total row count, and the bytes-per-row bound on a chunk is
//! identical either way, so the chunk count — and with it the prefix re-walk
//! count — would be conserved too: parity, not a win. The winning form is
//! maximal rows per SINGLE walk, which is exactly what the peak-bytes gate
//! below prices (stacked rows scale memory linearly). Members that do not fit
//! stay on the existing per-target path.
//!
//! SOUNDNESS: pure batching — each stacked row's arithmetic is identical to
//! its solo pass (see the injection's exact-zero precondition and the
//! bit-identity test in `target_backward.rs`). Every refusal in this planner
//! falls back to the historical per-target path and can only cost tightness.

use crate::layers::Layer;
use crate::network::core::GraphNetwork;
use crate::network::crown_memory::{cpu_crown_dense_budget_bytes, dense_pair_bytes};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Exact-"1" arming gate. Anything else (unset, "0", "true", " 1") is OFF and
/// the collector is byte-identical to the historical path.
///
/// Declared as `ny_levers::decls::cgan_stacked::CGAN_STACKED_BACKWARD`. Only the
/// ENV ACQUISITION goes through the chokepoint (`read_raw` is the same
/// `env::var(..).ok()` lookup this used to do inline, so a non-UTF-8 value still
/// reads as absent); the arming RULE stays in the pure predicate below, which is
/// unit-tested and is therefore the spec.
pub(super) fn stacked_backward_enabled() -> bool {
    stacked_backward_enabled_from_raw(
        ny_levers::read_raw(&ny_levers::decls::cgan_stacked::CGAN_STACKED_BACKWARD).as_deref(),
    )
}

fn stacked_backward_enabled_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Peak-bytes budget for the single stacked walk. Default: the same CPU dense
/// budget every other dense CROWN transient is priced against. Research
/// override: `NY_CGAN_STACKED_BUDGET_MB` (whole MiB) — the GB10's 121 GiB
/// unified memory admits far larger single-walk stacks than the conservative
/// default, and Phase 1 of the runbook sweeps this.
///
/// Declared as `ny_levers::decls::cgan_stacked::CGAN_STACKED_BUDGET_MB` (`U64`,
/// NOT trimmed — a padded value is rejected and leaves the host budget, exactly
/// as `parse::<usize>()` did here). The MiB->bytes multiplication and its
/// overflow fallback stay at this reader, and so does the meaning of an explicit
/// `0`: a zero-byte budget refuses every member, which is a different thing from
/// absence.
pub(super) fn stacked_budget_bytes() -> usize {
    ny_levers::read(&ny_levers::decls::cgan_stacked::CGAN_STACKED_BUDGET_MB)
        .value
        .as_u64()
        .and_then(|mib| usize::try_from(mib).ok())
        .and_then(|mib| mib.checked_mul(1024 * 1024))
        .unwrap_or_else(cpu_crown_dense_budget_bytes)
}

/// Engagement telemetry (rule R9: a null from an inert lever is vacuous).
/// Armed-only, rate-limited: the first 64 events, then powers of two.
pub(super) fn stacked_event(args: std::fmt::Arguments<'_>) {
    if !stacked_backward_enabled() {
        return;
    }
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    if n < 64 || n.is_power_of_two() {
        eprintln!("[NY_CGAN_STACKED] {args}");
    }
}

/// A collector candidate eligible for stacking: a demanded, non-sparse,
/// non-subset tightening target with its exec-order position and flat rows.
pub(super) struct StackedCandidate {
    pub(super) exec_index: usize,
    pub(super) node_name: String,
    pub(super) rows: usize,
}

/// The planned stack. `stack` is in walk-encounter order (deepest target
/// first — it becomes the walk root); `member_exec_indices` lets the
/// collector zero the members' scheduling weights once the pass has run.
pub(super) struct StackedBackwardPlan {
    pub(super) stack: Vec<(String, usize)>,
    pub(super) member_exec_indices: Vec<usize>,
    pub(super) total_rows: usize,
    pub(super) estimated_peak_bytes: usize,
}

/// Conservative multiplier over one dense `[rows x width]` lower/upper
/// coefficient pair for the walk's live transients: the A-pair itself, the
/// certified-error pair, and one step-transient copy. A planning estimate
/// (MEASURED peak belongs to Phase 0 of the runbook); over-estimating only
/// refuses more, never breaks soundness.
const STACKED_TRANSIENT_PAIRS: usize = 3;

/// Minimum share of the deepest member's walk a candidate's own solo walk
/// must cover to join the stack. Riding the FULL deep walk to save a short
/// shallow one is a net loss (the rows pay every deep step); on the scored
/// cgan chains this keeps the shallow generator targets (BatchNormalization_5
/// walks 5 of 26 nodes) off a discriminator-rooted stack.
const MIN_WALK_SHARE: f64 = 0.5;

/// Dense-walk audit for the stacked lane. Deliberately explicit, mirroring
/// `output_conditioned_additional_seed_node_is_audited` but extended with the
/// ConvTranspose generators this lane exists for:
///
/// - `Linear`, dense `Conv2d`, `ConvTranspose1d/2d`, and `BatchNorm` run the
///   canonical `dispatch_backward_layer` route, which preserves the certified
///   coefficient-error channel and is strictly per-row;
/// - fixed-slope `ReLU` (the executor passes `alpha_state = None`) scales
///   coefficients and adds intercept*coefficient bias terms per element —
///   row-local and exactly zero on zero rows;
/// - `Flatten`/`Reshape` are exact pass-through relations.
///
/// Everything else is refused until it has an equally explicit audit; refusal
/// falls back to the per-target path.
fn stacked_walk_node_is_audited(layer: &Layer, input_count: usize) -> bool {
    input_count == 1
        && matches!(
            layer,
            Layer::Linear(_)
                | Layer::Conv2d(_)
                | Layer::ConvTranspose1d(_)
                | Layer::ConvTranspose2d(_)
                | Layer::BatchNorm(_)
                | Layer::ReLU(_)
                | Layer::Flatten(_)
                | Layer::Reshape(_)
        )
}

fn estimated_peak_bytes(total_rows: usize, max_width: usize) -> Option<usize> {
    dense_pair_bytes(total_rows, max_width)?.checked_mul(STACKED_TRANSIENT_PAIRS)
}

/// Build the stacked plan, or decline (`None`) with a telemetry reason.
/// Every decline leaves the collector byte-identical to the historical path.
pub(super) fn plan_stacked_backward(
    graph: &GraphNetwork,
    exec_order: &[String],
    candidates: &[StackedCandidate],
    ibp_bounds: &HashMap<String, BoundedTensor>,
) -> Option<StackedBackwardPlan> {
    plan_stacked_backward_with_budget(
        graph,
        exec_order,
        candidates,
        ibp_bounds,
        stacked_budget_bytes(),
    )
}

pub(super) fn plan_stacked_backward_with_budget(
    graph: &GraphNetwork,
    exec_order: &[String],
    candidates: &[StackedCandidate],
    ibp_bounds: &HashMap<String, BoundedTensor>,
    budget_bytes: usize,
) -> Option<StackedBackwardPlan> {
    if candidates.len() < 2 {
        stacked_event(format_args!(
            "stage=declined reason=fewer-than-two-candidates count={}",
            candidates.len()
        ));
        return None;
    }
    // Deepest-first: the first admissible candidate roots the walk and fixes
    // the audited ancestry every other member must lie on.
    let mut ordered: Vec<&StackedCandidate> = candidates.iter().collect();
    ordered.sort_by_key(|c| std::cmp::Reverse(c.exec_index));
    let deepest = ordered[0];

    let ancestry = graph.ancestors(&deepest.node_name).ok()?;
    // Trunk + per-node audit over the WHOLE walk. Fan-out anywhere on the
    // ancestry disqualifies the lane: the shared-prefix claim (and the
    // "members never diverge" premise) is a chain property.
    for node_name in ancestry.iter() {
        let Some(node) = graph.nodes.get(node_name) else {
            stacked_event(format_args!(
                "stage=declined reason=missing-node node='{node_name}'"
            ));
            return None;
        };
        if !stacked_walk_node_is_audited(&node.layer, node.inputs().len()) {
            stacked_event(format_args!(
                "stage=declined reason=unaudited-layer node='{node_name}' layer={}",
                node.layer.layer_type()
            ));
            return None;
        }
        let consumers = exec_order
            .iter()
            .filter_map(|name| graph.nodes.get(name))
            .filter(|candidate| candidate.inputs().iter().any(|input| input == node_name))
            .count();
        if consumers > 1 {
            stacked_event(format_args!(
                "stage=declined reason=fan-out node='{node_name}' consumers={consumers}"
            ));
            return None;
        }
    }

    let position_in_ancestry: HashMap<&str, usize> = ancestry
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();
    let deepest_walk_len = ancestry.len();
    let max_walk_width = ancestry
        .iter()
        .filter_map(|name| ibp_bounds.get(name))
        .map(BoundedTensor::len)
        .max()
        .unwrap_or(deepest.rows)
        .max(deepest.rows);

    // Greedy deepest-first accumulation under the peak-bytes gate. The stack
    // rows ride the deepest walk, so the peak transient is
    // `[total_rows x max_walk_width]` regardless of where a member injects —
    // conservative for shallow members (their rows exist only below their
    // node), which can only refuse more.
    let mut selected: Vec<&StackedCandidate> = Vec::new();
    let mut total_rows = 0usize;
    for &candidate in &ordered {
        let Some(&position) = position_in_ancestry.get(candidate.node_name.as_str()) else {
            // Not on the deepest walk (impossible on a chain; defensive).
            continue;
        };
        let candidate_walk_len = position + 1;
        if !selected.is_empty()
            && (candidate_walk_len as f64) < MIN_WALK_SHARE * deepest_walk_len as f64
        {
            stacked_event(format_args!(
                "stage=member-skipped reason=short-walk node='{}' walk={candidate_walk_len}/{deepest_walk_len}",
                candidate.node_name
            ));
            continue;
        }
        let Some(next_rows) = total_rows.checked_add(candidate.rows) else {
            continue;
        };
        match estimated_peak_bytes(next_rows, max_walk_width) {
            Some(peak) if peak <= budget_bytes => {
                selected.push(candidate);
                total_rows = next_rows;
            }
            peak => {
                stacked_event(format_args!(
                    "stage=member-skipped reason=over-budget node='{}' rows={} \
                     projected_peak_bytes={} budget_bytes={budget_bytes}",
                    candidate.node_name,
                    candidate.rows,
                    peak.unwrap_or(usize::MAX),
                ));
            }
        }
    }
    if selected.len() < 2 {
        stacked_event(format_args!(
            "stage=declined reason=budget-leaves-single-member budget_bytes={budget_bytes} \
             max_walk_width={max_walk_width}"
        ));
        return None;
    }
    // The greedy loop always admits the deepest candidate first (its solo
    // walk IS the full walk), so `selected[0]` is the walk root.
    let estimated = estimated_peak_bytes(total_rows, max_walk_width).unwrap_or(usize::MAX);
    let plan = StackedBackwardPlan {
        stack: selected
            .iter()
            .map(|candidate| (candidate.node_name.clone(), candidate.rows))
            .collect(),
        member_exec_indices: selected
            .iter()
            .map(|candidate| candidate.exec_index)
            .collect(),
        total_rows,
        estimated_peak_bytes: estimated,
    };
    stacked_event(format_args!(
        "stage=planned members={} total_rows={} estimated_peak_bytes={} budget_bytes={} root='{}'",
        plan.stack.len(),
        plan.total_rows,
        plan.estimated_peak_bytes,
        budget_bytes,
        plan.stack[0].0,
    ));
    Some(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::network::core::GraphNode;
    use ndarray::arr1;

    fn linear(rows: usize, cols: usize) -> LinearLayer {
        let weights = ndarray::Array2::from_shape_fn((rows, cols), |(i, j)| {
            0.25 + 0.5 * ((i + 2 * j) % 3) as f32
        });
        LinearLayer::new(weights, None).expect("linear layer")
    }

    /// in -> l1 -> relu1 -> l2 -> relu2 -> l3 -> relu3. Long enough that l2's
    /// own walk (3 of l3's 5 ancestry nodes) clears `MIN_WALK_SHARE` while
    /// l1's (1 of 5) does not.
    fn chain_graph() -> (GraphNetwork, HashMap<String, BoundedTensor>) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("l1", Layer::Linear(linear(3, 2))));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l2",
            Layer::Linear(linear(4, 3)),
            vec!["relu1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu2",
            Layer::ReLU(ReLULayer),
            vec!["l2".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l3",
            Layer::Linear(linear(2, 4)),
            vec!["relu2".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu3",
            Layer::ReLU(ReLULayer),
            vec!["l3".to_string()],
        ));
        graph.set_output("relu3");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[1.0_f32, 0.75]).into_dyn(),
        )
        .expect("input");
        let bounds = graph.collect_node_bounds(&input).expect("forward bounds");
        (graph, bounds)
    }

    fn candidate(exec_index: usize, node_name: &str, rows: usize) -> StackedCandidate {
        StackedCandidate {
            exec_index,
            node_name: node_name.to_string(),
            rows,
        }
    }

    #[test]
    fn gate_parser_is_exact_and_default_dark() {
        assert!(!stacked_backward_enabled_from_raw(None));
        assert!(stacked_backward_enabled_from_raw(Some("1")));
        for raw in ["", "0", "true", "01", " 1", "2"] {
            assert!(!stacked_backward_enabled_from_raw(Some(raw)), "raw={raw:?}");
        }
    }

    #[test]
    fn plans_deepest_first_on_a_chain() {
        let (graph, bounds) = chain_graph();
        let exec_order = graph.topological_sort().expect("order");
        let candidates = [candidate(2, "l2", 4), candidate(4, "l3", 2)];
        let plan = plan_stacked_backward_with_budget(
            &graph,
            &exec_order,
            &candidates,
            &bounds,
            usize::MAX,
        )
        .expect("chain must plan");
        assert_eq!(
            plan.stack,
            vec![("l3".to_string(), 2), ("l2".to_string(), 4)],
            "walk-encounter order is deepest first"
        );
        assert_eq!(plan.total_rows, 6);
        assert_eq!(plan.member_exec_indices, vec![4, 2]);
        assert!(plan.estimated_peak_bytes > 0);
    }

    #[test]
    fn declines_on_fan_out() {
        // Diamond: l1 feeds BOTH relu1 and l2b — the trunk premise fails.
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("l1", Layer::Linear(linear(3, 2))));
        graph.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "l2b",
            Layer::Linear(linear(2, 3)),
            vec!["l1".to_string()],
        ));
        graph.set_output("l2b");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[1.0_f32, 0.75]).into_dyn(),
        )
        .expect("input");
        let bounds = graph.collect_node_bounds(&input).expect("forward bounds");
        let exec_order = graph.topological_sort().expect("order");
        let candidates = [candidate(0, "l1", 3), candidate(2, "l2b", 2)];
        assert!(plan_stacked_backward_with_budget(
            &graph,
            &exec_order,
            &candidates,
            &bounds,
            usize::MAX,
        )
        .is_none());
    }

    #[test]
    fn budget_gate_refuses_and_partial_stacks() {
        let (graph, bounds) = chain_graph();
        let exec_order = graph.topological_sort().expect("order");
        let candidates = [candidate(2, "l2", 4), candidate(4, "l3", 2)];
        // A one-byte budget cannot admit even the deepest member.
        assert!(
            plan_stacked_backward_with_budget(&graph, &exec_order, &candidates, &bounds, 1,)
                .is_none()
        );
        // Exactly the two-member peak admits both (boundary inclusive).
        let two_member_peak = estimated_peak_bytes(6, 4).expect("peak");
        let plan = plan_stacked_backward_with_budget(
            &graph,
            &exec_order,
            &candidates,
            &bounds,
            two_member_peak,
        )
        .expect("boundary budget must plan");
        assert_eq!(plan.total_rows, 6);
        assert_eq!(plan.estimated_peak_bytes, two_member_peak);
    }

    #[test]
    fn single_candidate_declines() {
        let (graph, bounds) = chain_graph();
        let exec_order = graph.topological_sort().expect("order");
        let candidates = [candidate(4, "l3", 2)];
        assert!(plan_stacked_backward_with_budget(
            &graph,
            &exec_order,
            &candidates,
            &bounds,
            usize::MAX,
        )
        .is_none());
    }

    #[test]
    fn short_walk_member_is_excluded() {
        let (graph, bounds) = chain_graph();
        let exec_order = graph.topological_sort().expect("order");
        // l1's walk is 1 of 5 ancestry nodes of l3 (< MIN_WALK_SHARE):
        // excluded, and with l2 + l3 remaining the plan still forms.
        let candidates = [
            candidate(0, "l1", 3),
            candidate(2, "l2", 4),
            candidate(4, "l3", 2),
        ];
        let plan = plan_stacked_backward_with_budget(
            &graph,
            &exec_order,
            &candidates,
            &bounds,
            usize::MAX,
        )
        .expect("deep pair must plan");
        assert_eq!(
            plan.stack,
            vec![("l3".to_string(), 2), ("l2".to_string(), 4)],
            "the short-walk shallow member must be excluded"
        );
    }

    #[test]
    fn peak_estimate_is_pair_bytes_times_transients() {
        let expected = dense_pair_bytes(7, 4).unwrap() * STACKED_TRANSIENT_PAIRS;
        assert_eq!(estimated_peak_bytes(7, 4), Some(expected));
        assert_eq!(estimated_peak_bytes(usize::MAX, 2), None);
    }
}
