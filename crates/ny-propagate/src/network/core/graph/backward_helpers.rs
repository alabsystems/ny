// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph-wide backward helpers for CROWN bound accumulation.
//!
//! These helpers are used across graph-CROWN, graph-alpha, and beta-CROWN graph
//! constraint coordinators. Hoisted from `graph_crown/utils.rs` (#3936) to make
//! the cross-engine dependency explicit and eliminate module-boundary leaks.

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::network::backward_dispatch::BackwardDispatchResult;

use super::merge_accumulator::CrownMergeAccumulator;
use super::{GraphNetwork, GraphNode, NETWORK_INPUT};

use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;
use tracing::debug;

fn clone_pass_through_bounds_for_dispatch(
    pass_through_bounds: &LinearBounds,
    _node_crown_bounds: &CrownMergeAccumulator,
    deadline: Option<Instant>,
) -> Result<LinearBounds> {
    if let Some(limit) = deadline {
        if Instant::now() >= limit {
            return Err(NyError::DeadlineExceeded(
                "graph CROWN pass-through: deadline exceeded before clone".into(),
            ));
        }
        // Computing the retained-frontier charge is itself an unpollable
        // O(node-count) scan. Until the caller can transfer ownership instead
        // of cloning, a finite request must decline before inspecting either
        // the frontier or coefficient payload.
        return Err(NyError::UnsupportedConfiguration(
            "finite graph CROWN pass-through cloning is unavailable".into(),
        ));
    }

    // Preserve the historical unbounded behavior exactly.
    Ok(pass_through_bounds.clone())
}

/// Return the fixed boolean select mask for a Where condition iff that condition
/// is bound-independent (a constant), i.e. `lower == upper` at every position.
///
/// `mask[i] == true` means the condition selects the true branch at flat
/// position `i` (ONNX treats any non-zero condition value as true; here the
/// bounds are integer-coded 0/1 so we threshold at 0.5). Returns `None` when the
/// condition is data-dependent (any position has a non-degenerate interval),
/// signalling the caller to keep the sound IBP/concretize fallback.
///
/// Shared by the graph-CROWN and graph-α backward Where arms (#Where-const-cond).
pub(crate) fn where_constant_mask(cond: &BoundedTensor) -> Option<Vec<bool>> {
    let lower = cond.lower();
    let upper = cond.upper();
    let mut mask = Vec::with_capacity(lower.len());
    for (&lo, &hi) in lower.iter().zip(upper.iter()) {
        // Data-dependent condition: the interval straddles the 0.5 decision
        // boundary or is otherwise non-degenerate. Bail to the loose fallback.
        if lo != hi {
            return None;
        }
        if !lo.is_finite() {
            return None;
        }
        mask.push(lo >= 0.5);
    }
    Some(mask)
}

/// Build the exact backward `LinearBounds` for one branch of a constant-condition
/// Where by zeroing the A-matrix columns that belong to the other branch.
///
/// Column `i` of the incoming `node_lb` corresponds to flat output position `i`
/// of the Where. For the true branch (`keep_true == true`) we keep column `i`
/// only where `mask[i]` is true; for the false branch we keep it where `mask[i]`
/// is false.
///
/// The two branch contributions are accumulated separately by the caller, so the
/// incoming bias must NOT be applied twice. We carry it on the true branch only
/// and zero it on the false branch.
pub(crate) fn mask_linear_bounds_columns(
    node_lb: &LinearBounds,
    mask: &[bool],
    keep_true: bool,
) -> LinearBounds {
    let mut lower_a = node_lb.lower_a().clone();
    let mut upper_a = node_lb.upper_a().clone();
    for (col, &m) in mask.iter().enumerate() {
        if m != keep_true {
            // This output position belongs to the other branch — zero the column
            // so this branch contributes nothing through it.
            for row in 0..lower_a.nrows() {
                lower_a[[row, col]] = 0.0;
                upper_a[[row, col]] = 0.0;
            }
        }
    }
    // Bias is a per-output constant independent of the split; apply it on exactly
    // one of the two branch paths to avoid double-counting.
    let (lower_b, upper_b) = if keep_true {
        (node_lb.lower_b().clone(), node_lb.upper_b().clone())
    } else {
        (
            Array1::zeros(node_lb.lower_b().len()),
            Array1::zeros(node_lb.upper_b().len()),
        )
    };
    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
        .expect("masking preserves A/bias shapes; new_or_conservative cannot fail on shape")
}

/// Minimum dense row count for Dense->Patches re-entry
/// (#cgan-alpha-on-tight-refs). The patches-mode ReLU backward selects the
/// lower/upper envelope PER TAP; when conv receptive fields overlap
/// (stride < kernel), taps of one input neuron with mixed signs each pay
/// their own relaxation intercept, while the dense backward sign-selects on
/// the SUMMED coefficient — strictly tighter. Measured on cGAN_imgSz32_nCh_1
/// prop_1: the 1-row objective backward concretizes to -7.7e7 through the
/// patches segment vs -96.9 in matrix mode (~8e5x looser), which froze the
/// alpha warmup and the per-domain BaB rebound at the root bound. Thin seeds
/// are also CHEAPER dense (rows x in_dim matrices), so route them to matrix
/// mode; patches remains for many-row seeds where dense materialization is
/// the memory wall. `NY_PATCHES_REENTRY_MIN_ROWS` overrides (1 restores the
/// pre-fix always-re-enter behavior).
pub(crate) fn patches_reentry_min_rows() -> usize {
    static MIN_ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_ROWS.get_or_init(|| {
        std::env::var("NY_PATCHES_REENTRY_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(5)
    })
}

/// `NY_PATCHES_REENTRY_MEASURED=0` restores the pure row-count heuristic, i.e.
/// re-enter patches whenever `rows >= patches_reentry_min_rows()` regardless of
/// whether the dense pair would actually have fit (#reentry-measure-dont-guess).
fn patches_reentry_measured_enabled() -> bool {
    std::env::var("NY_PATCHES_REENTRY_MEASURED").ok().as_deref() != Some("0")
}

pub(crate) fn try_dense_spatial_patches_reentry(
    node_cb: &mut CrownBounds,
    node: &GraphNode,
    node_name: &str,
    current_bounds: &HashMap<String, BoundedTensor>,
    use_patches_mode: bool,
    label: &str,
) -> bool {
    if !use_patches_mode
        || !matches!(node_cb, CrownBounds::Dense(_))
        || node.inputs.len() != 1
        || !matches!(&node.layer, Layer::Conv2d(_))
    {
        return matches!(node_cb, CrownBounds::Patches(_));
    }
    if let CrownBounds::Dense(lb) = &*node_cb {
        let rows = lb.num_outputs();
        if rows < patches_reentry_min_rows() {
            debug!(
                "{}: Dense->Patches re-entry skipped at {} ({} rows < min {}): \
                 matrix mode is tighter for thin seeds through overlapping \
                 receptive fields (#cgan-alpha-on-tight-refs)",
                label,
                node_name,
                rows,
                patches_reentry_min_rows()
            );
            return false;
        }
        // #reentry-measure-dont-guess: the row FLOOR above answers "is this seed
        // thin enough that matrix mode is tighter". It does NOT answer the
        // question this function's own docstring says decides the route --
        // "patches remains for many-row seeds WHERE DENSE MATERIALIZATION IS THE
        // MEMORY WALL". Row count is not that measurement, and on a wide-but-thin
        // relation the two answers diverge badly.
        //
        // Measured on yolo_2023 (2026-07-30): Conv_25's cone seed is 576 rows over
        // a 10816-wide input. The dense pair is 576*10816*4*2 = ~50 MB -- three
        // orders of magnitude under the budget, nowhere near any wall. The row
        // floor of 5 nonetheless routed it into a 7-D patches re-entry, whose
        // backward took **611.9 s**, blew the per-node deadline, returned NO bound,
        // and starved `Flatten_30` (the OUTPUT node, whose bound is the only one
        // the verdict reads). Declining that re-entry: **9.4 s** (65x faster) and
        // crown width 511.0 instead of 116611 (228x tighter -- and 8.3x tighter
        // than the node's own IBP, where before it was 27x WORSE).
        // Y[269] over the whole run: -42053.75 -> -5246.26.
        //
        // So decide it by MEASURING the thing the comment names. Re-enter patches
        // only when the dense pair genuinely does not fit; otherwise dense is both
        // cheaper AND tighter and there is no reason to leave it.
        //
        // SOUNDNESS-NEUTRAL: this only chooses a REPRESENTATION for the same
        // relation. Both routes are certified; neither can narrow a bound. The
        // existing `Err` arm below already falls back to staying dense, so this is
        // the same decision taken earlier and on better evidence.
        // MARGIN, and why it is not cosmetic. `dense_pair_bytes` measures the
        // COEFFICIENT PAIR only. The dense conv backward additionally allocates
        // transients the pair does not account for -- notably
        // `conv2d::ops_transpose_gemm::backward_result`, sized on `conv_in_size`
        // rather than on this relation's `in_dim`. So "the pair fits" does NOT
        // imply "the backward fits", and declining re-entry on the pair alone
        // turns a survivable patches route into a hard `CpuMemoryExceeded`.
        // `input_linear_captures_patches_conv_chain` caught exactly that:
        // required 640_000 B against a 524_288 B budget, a ratio of only 1.22.
        // Require the pair to clear the budget by this factor before declining,
        // so a relation that is merely *near* the wall keeps the patches route.
        // Conv_25 on yolo_2023 clears it by ~128x, so the case this exists for is
        // unaffected. This is a stated safety factor on a measurement, not a
        // guess at the answer -- if the transient estimate is ever threaded
        // through to here, replace it with the real number and drop the factor.
        const DENSE_ROUTE_HEADROOM: usize = 8;
        if let Some(dense_pair_bytes) = lb
            .num_outputs()
            .checked_mul(lb.num_inputs())
            .and_then(|cells| cells.checked_mul(2 * size_of::<f32>()))
        {
            let budget = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
            let fits_with_headroom = dense_pair_bytes
                .checked_mul(DENSE_ROUTE_HEADROOM)
                .is_some_and(|needed| needed <= budget);
            if fits_with_headroom && patches_reentry_measured_enabled() {
                debug!(
                    "{}: #reentry-measure-dont-guess: Dense->Patches re-entry declined at {} \
                     ({} rows x {} in; dense pair {} B fits the {} B budget). Dense is cheaper \
                     AND tighter here; patches re-entry exists for when dense does NOT fit.",
                    label,
                    node_name,
                    rows,
                    lb.num_inputs(),
                    dense_pair_bytes,
                    budget
                );
                return false;
            }
        }
    }

    let Some(current_bounds) = current_bounds.get(node_name) else {
        return false;
    };

    let current_shape = current_bounds.shape();
    if current_shape.len() != 3 {
        return false;
    }

    let spatial = (current_shape[0], current_shape[1], current_shape[2]);
    let spatial_dim = spatial.0 * spatial.1 * spatial.2;
    if let CrownBounds::Dense(lb) = node_cb {
        if lb.num_inputs() == spatial_dim {
            match PatchesLinearBounds::from_dense_spatial_rows(lb, spatial) {
                Ok(pb) => {
                    debug!(
                        "{}: Dense->Patches re-entry at {} with {} rows over {:?}",
                        label, node_name, pb.row_count, spatial
                    );
                    *node_cb = CrownBounds::Patches(Box::new(pb));
                }
                Err(err) => {
                    debug!(
                        "{}: Dense->Patches re-entry skipped at {}: {}",
                        label, node_name, err
                    );
                }
            }
        }
    }

    matches!(node_cb, CrownBounds::Patches(_))
}

/// Finite graph walks must not enter the legacy Dense→Patches converter: it
/// allocates and fills two full coefficient tensors without a cooperative
/// deadline or operation receipt. `None` preserves the historical heuristic;
/// `Some` declines in O(1) before inspecting/copying the dense carrier.
pub(crate) fn try_dense_spatial_patches_reentry_with_deadline(
    node_cb: &mut CrownBounds,
    node: &GraphNode,
    node_name: &str,
    current_bounds: &HashMap<String, BoundedTensor>,
    use_patches_mode: bool,
    label: &str,
    deadline: Option<Instant>,
) -> bool {
    try_dense_spatial_patches_reentry_with_deadline_authority(
        node_cb,
        node,
        node_name,
        current_bounds,
        use_patches_mode,
        label,
        deadline,
        deadline.is_some(),
    )
}

/// Authority-aware variant for collectors that retain an internal scheduling
/// timestamp even when the caller supplied no hard outer deadline. The soft
/// timestamp remains available to cooperative work after re-entry; it cannot
/// authorize declining this historical unbounded conversion.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_dense_spatial_patches_reentry_with_deadline_authority(
    node_cb: &mut CrownBounds,
    node: &GraphNode,
    node_name: &str,
    current_bounds: &HashMap<String, BoundedTensor>,
    use_patches_mode: bool,
    label: &str,
    _deadline: Option<Instant>,
    deadline_is_hard: bool,
) -> bool {
    if deadline_is_hard {
        return matches!(node_cb, CrownBounds::Patches(_));
    }
    try_dense_spatial_patches_reentry(
        node_cb,
        node,
        node_name,
        current_bounds,
        use_patches_mode,
        label,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_dense_backward_dispatch_result_with_deadline(
    graph: &GraphNetwork,
    node: &GraphNode,
    first_input: &str,
    pass_through_bounds: &LinearBounds,
    result: BackwardDispatchResult,
    node_crown_bounds: &mut CrownMergeAccumulator,
    output_dim: usize,
    input_dim: usize,
    input_accumulated: &mut bool,
    context_prefix: &str,
    deadline: Option<Instant>,
) -> Result<()> {
    match result {
        BackwardDispatchResult::Single(new_lb) => graph
            .accumulate_dense_bounds_to_input_with_deadline(
                first_input,
                *new_lb,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
                deadline,
            ),
        BackwardDispatchResult::Binary {
            bounds_a,
            bounds_b,
            bias_lower,
            bias_upper,
        } => {
            let (input_a_name, input_b_name) = node.require_binary_inputs()?;
            GraphNetwork::accumulate_bias_to_network_input_crown_with_deadline(
                &bias_lower,
                &bias_upper,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
                deadline,
            )?;
            GraphNetwork::verify_split_path_bias_zero(
                &bounds_a,
                &format!("{context_prefix} binary lhs split path"),
            )?;
            GraphNetwork::verify_split_path_bias_zero(
                &bounds_b,
                &format!("{context_prefix} binary rhs split path"),
            )?;
            graph.accumulate_dense_bounds_to_input_with_deadline(
                input_a_name,
                *bounds_a,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
                deadline,
            )?;
            graph.accumulate_dense_bounds_to_input_with_deadline(
                input_b_name,
                *bounds_b,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
                deadline,
            )
        }
        BackwardDispatchResult::Nary {
            bounds,
            bias_lower,
            bias_upper,
        } => {
            GraphNetwork::accumulate_bias_to_network_input_crown_with_deadline(
                &bias_lower,
                &bias_upper,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
                deadline,
            )?;
            for (graph_idx, lb) in bounds.into_iter().flatten().enumerate() {
                GraphNetwork::verify_split_path_bias_zero(
                    &lb,
                    &format!("{context_prefix} n-ary split path"),
                )?;
                if let Some(inp_name) = node.inputs.get(graph_idx) {
                    graph.accumulate_dense_bounds_to_input_with_deadline(
                        inp_name,
                        lb,
                        node_crown_bounds,
                        output_dim,
                        input_dim,
                        input_accumulated,
                        deadline,
                    )?;
                }
            }
            Ok(())
        }
        BackwardDispatchResult::PassThrough => {
            let pass_through_bounds = clone_pass_through_bounds_for_dispatch(
                pass_through_bounds,
                node_crown_bounds,
                deadline,
            )?;
            graph.accumulate_dense_bounds_to_input_with_deadline(
                first_input,
                pass_through_bounds,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
                deadline,
            )
        }
        BackwardDispatchResult::Unsupported(reason) => Err(NyError::UnsupportedOp(reason)),
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod deadline_tests {
    use super::*;
    use crate::layers::AddLayer;
    use ndarray::{ArrayD, IxDyn};
    use std::time::Duration;

    fn residual_operand_bounds() -> HashMap<String, BoundedTensor> {
        let lower = ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0_f32);
        let bounds = BoundedTensor::new(lower, upper).expect("valid residual operand bounds");
        HashMap::from([
            ("left".to_string(), bounds.clone()),
            ("right".to_string(), bounds),
        ])
    }

    #[test]
    fn expired_pass_through_clone_preserves_accumulator_frontier() {
        let pass_through = LinearBounds::identity(8);
        let mut accumulator = CrownMergeAccumulator::new();
        accumulator.insert(
            "retained".to_string(),
            CrownBounds::Dense(LinearBounds::identity(4)),
        );
        let retained_bytes = accumulator.logical_frontier_payload_bytes();
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before now");

        let error =
            clone_pass_through_bounds_for_dispatch(&pass_through, &accumulator, Some(expired))
                .expect_err("an expired finite authority must refuse before cloning");

        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert!(accumulator.contains_key("retained"));
        assert_eq!(
            accumulator.logical_frontier_payload_bytes(),
            retained_bytes,
            "failed staging must not drain or replace the live frontier"
        );
    }

    #[test]
    fn live_finite_pass_through_declines_without_mutating_frontier() {
        let pass_through = LinearBounds::identity(8);
        let mut accumulator = CrownMergeAccumulator::new();
        accumulator.insert(
            "retained".to_string(),
            CrownBounds::Dense(LinearBounds::identity(4)),
        );
        let retained_bytes = accumulator.logical_frontier_payload_bytes();

        let error = clone_pass_through_bounds_for_dispatch(
            &pass_through,
            &accumulator,
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .expect_err("a live finite authority must decline before scanning the frontier");

        assert!(matches!(error, NyError::UnsupportedConfiguration(_)));
        assert!(accumulator.contains_key("retained"));
        assert_eq!(
            accumulator.logical_frontier_payload_bytes(),
            retained_bytes,
            "typed refusal must leave the live frontier unchanged"
        );
    }

    #[test]
    fn finite_patches_residual_declines_before_cloning_or_publication() {
        let graph = GraphNetwork::new();
        let node = GraphNode::binary("residual", Layer::Add(AddLayer), "left", "right");
        let mut carrier = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
            (1, 2, 2),
            (1, 2, 2),
        )));
        let (lower_bias_ptr, upper_bias_ptr) = match &carrier {
            CrownBounds::Patches(bounds) => (bounds.lower_b.as_ptr(), bounds.upper_b.as_ptr()),
            CrownBounds::Dense(_) => unreachable!("test starts with a Patches carrier"),
        };
        let mut accumulator = CrownMergeAccumulator::new();
        let mut input_accumulated = false;

        let handled = try_apply_patches_residual_passthrough_with_deadline(
            &graph,
            &node,
            &mut carrier,
            &HashMap::new(),
            &mut accumulator,
            4,
            4,
            &mut input_accumulated,
            "finite residual test",
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .expect("finite residual passthrough should decline to Dense");

        assert!(!handled);
        assert!(accumulator.is_empty());
        assert!(!input_accumulated);
        match carrier {
            CrownBounds::Patches(bounds) => {
                assert_eq!(bounds.lower_b.as_ptr(), lower_bias_ptr);
                assert_eq!(bounds.upper_b.as_ptr(), upper_bias_ptr);
            }
            CrownBounds::Dense(_) => panic!("finite decline must retain the Patches source"),
        }
    }

    #[test]
    fn expired_soft_residual_unit_refuses_before_clone_or_publication() {
        let graph = GraphNetwork::new();
        let node = GraphNode::binary("residual", Layer::Add(AddLayer), "left", "right");
        let mut carrier = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
            (1, 2, 2),
            (1, 2, 2),
        )));
        let (lower_bias_ptr, upper_bias_ptr) = match &carrier {
            CrownBounds::Patches(bounds) => (bounds.lower_b.as_ptr(), bounds.upper_b.as_ptr()),
            CrownBounds::Dense(_) => unreachable!("test starts with a Patches carrier"),
        };
        let mut accumulator = CrownMergeAccumulator::new();
        let mut input_accumulated = false;
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits before now");

        let error = try_apply_patches_residual_passthrough_with_deadline_authority(
            &graph,
            &node,
            &mut carrier,
            &residual_operand_bounds(),
            &mut accumulator,
            4,
            4,
            &mut input_accumulated,
            "soft residual test",
            Some(expired),
            false,
        )
        .expect_err("an expired soft unit must refuse before its residual clone");

        assert!(matches!(error, NyError::DeadlineExceeded(_)));
        assert!(accumulator.is_empty());
        assert!(!input_accumulated);
        match carrier {
            CrownBounds::Patches(bounds) => {
                assert_eq!(bounds.lower_b.as_ptr(), lower_bias_ptr);
                assert_eq!(bounds.upper_b.as_ptr(), upper_bias_ptr);
            }
            CrownBounds::Dense(_) => panic!("expired soft admission must retain the source"),
        }
    }

    #[test]
    fn live_soft_residual_unit_publishes_both_branches() {
        let graph = GraphNetwork::new();
        let node = GraphNode::binary("residual", Layer::Add(AddLayer), "left", "right");
        let mut carrier = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
            (1, 2, 2),
            (1, 2, 2),
        )));
        let mut accumulator = CrownMergeAccumulator::new();
        let mut input_accumulated = false;

        let handled = try_apply_patches_residual_passthrough_with_deadline_authority(
            &graph,
            &node,
            &mut carrier,
            &residual_operand_bounds(),
            &mut accumulator,
            4,
            4,
            &mut input_accumulated,
            "soft residual test",
            Some(Instant::now() + Duration::from_secs(30)),
            false,
        )
        .expect("a live soft unit must publish its complete residual fan-out");

        assert!(handled);
        assert!(accumulator.contains_key("left"));
        assert!(accumulator.contains_key("right"));
        assert!(!input_accumulated);
    }
}
/// Can the patches CROWN-backward walk consume this node's own layer without
/// densifying (#conv-crown-residual)?
///
/// Single-input nodes are the historical case: the patches step dispatches on
/// the layer and hands one relation to the one input. A **2-input elementwise
/// `Add`/`Sub`** is admitted too, because
/// [`try_apply_patches_residual_passthrough_with_deadline`] consumes it natively: for
/// `y = u + v` a backward relation `A` on `y` applies unchanged to both `u` and
/// `v`, so the relation is duplicated down the two branches and summed where
/// they rejoin.
///
/// This predicate is the structural half of the fix; the passthrough itself
/// still re-checks that the operand shapes match (no broadcast) against the
/// live bounds map and declines otherwise, so admitting a node here can never
/// force an unsound step — only a dense fallback, exactly as today.
///
/// `Sub` is NOT admitted, even though
/// [`try_apply_patches_residual_passthrough_with_deadline`] handles it — but purely as scope
/// control, NOT for correctness. An earlier revision of this comment claimed
/// the right-operand carrier used a "negate and swap" convention
/// (`lower_a' = -upper_a`, `upper_a' = -lower_a`); that was wrong about this
/// path. `PatchesLinearBounds::negated_zero_bias` negates each relation's own
/// coefficients with NO swap (`bounds/patches/merge/mod.rs:66-72`), which is
/// the correct substitution rule: from `obj >= lower_a·y + lower_b` and
/// `y = u - v`, the right operand's LOWER coefficient is `-lower_a`. The dense
/// path (`binary_ops/sub.rs`) genuinely did swap, which was a FALSE-BOUND bug,
/// and it has been fixed to match this one — so the two agree, and neither
/// swaps.
///
/// Admitting `Sub` here would therefore be sound (and the passthrough re-checks
/// shapes anyway, so it could only fall back, never mis-step). It is left out
/// because it is unmeasured: residual blocks in the ResNets this targets are
/// `Add`, so there was no throughput case for widening the predicate. A graph
/// with `Sub` inside a conv DAG is the case that would justify revisiting it.
///
/// Every other multi-input node (`Concat`, `MulBinary`, `Where`, broadcast
/// arithmetic, …) stays excluded too: the patches representation has no
/// branch/merge rule for them, and the generic single-input patches step would
/// misattribute the whole relation to `inputs[0]` — an unsound bound, not a
/// loose one.
pub(crate) fn node_admits_patches_backward_step(node: &GraphNode) -> bool {
    match node.inputs.len() {
        1 => true,
        2 => matches!(&node.layer, Layer::Add(_)),
        _ => false,
    }
}

/// Try patches-native residual passthrough for Add/Sub nodes (#4382).
///
/// Returns `Ok(true)` if this node was handled in patches form (caller should
/// skip the Dense fallback). Returns `Ok(false)` if not applicable.
///
/// Only handles same-shape `Add`/`Sub` (no broadcast). The original carrier is
/// moved to the left branch with its bias intact; the one fallibly-cloned right
/// branch has zero bias so the merge counts the affine offset exactly once.
///
/// Reference: alpha-beta-CROWN `operators/add_sub.py:37-47`
// Kept only for direct regression coverage of the historical unlimited route.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_apply_patches_residual_passthrough(
    graph: &GraphNetwork,
    node: &GraphNode,
    node_cb: &mut CrownBounds,
    node_bounds: &HashMap<String, BoundedTensor>,
    node_crown_bounds: &mut CrownMergeAccumulator,
    output_dim: usize,
    input_dim: usize,
    input_accumulated: &mut bool,
    context_prefix: &str,
) -> Result<bool> {
    try_apply_patches_residual_passthrough_with_deadline(
        graph,
        node,
        node_cb,
        node_bounds,
        node_crown_bounds,
        output_dim,
        input_dim,
        input_accumulated,
        context_prefix,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_apply_patches_residual_passthrough_with_deadline(
    graph: &GraphNetwork,
    node: &GraphNode,
    node_cb: &mut CrownBounds,
    node_bounds: &HashMap<String, BoundedTensor>,
    node_crown_bounds: &mut CrownMergeAccumulator,
    output_dim: usize,
    input_dim: usize,
    input_accumulated: &mut bool,
    context_prefix: &str,
    deadline: Option<Instant>,
) -> Result<bool> {
    try_apply_patches_residual_passthrough_with_deadline_authority(
        graph,
        node,
        node_cb,
        node_bounds,
        node_crown_bounds,
        output_dim,
        input_dim,
        input_accumulated,
        context_prefix,
        deadline,
        deadline.is_some(),
    )
}

/// #residual-patches-expiry: does finite authority refuse the native residual
/// Patches fan-out?
///
/// Shares `NY_PATCHES_FINITE_EXPIRY` with the sequential twin
/// (`patches_step::hard_finite_authority_refuses_patches`, `4d0257ba9`) and the
/// graph-lane alpha twin, so the set cannot drift apart. Lever off => refuses on
/// deadline PRESENCE, byte-identical to the historical `if deadline_is_hard`.
/// Shared by every set-mate (`backward_helpers`, `graph_alpha/backward/nonlinear`,
/// `graph_alpha/bounds/target_backward_patches`) so the set cannot drift apart —
/// which matters because fixing these one at a time measures exactly zero.
pub(crate) fn patches_finite_authority_refuses(
    deadline_is_hard: bool,
    deadline: Option<Instant>,
) -> bool {
    if !deadline_is_hard {
        return false;
    }
    if crate::network::core::sequential::crown::patches_step::expiry_authority_armed() {
        return deadline.is_some_and(|limit| Instant::now() >= limit);
    }
    true
}

/// Conservative retained-footprint admission for the native residual fan-out.
///
/// `try_clone_residual_branch` charges only the CLONE against
/// `cpu_crown_dense_budget_bytes()`. During the fan-out the SOURCE carrier is
/// still live and the accumulator frontier is still retained, so peak residency
/// is at least twice the clone. We require 3x to leave the frontier headroom.
///
/// Deliberately conservative: refusing here costs a densified node (the exact
/// status quo), whereas over-admitting on a unified-memory host costs the
/// process. Asymmetric penalties, asymmetric threshold.
fn residual_retained_footprint_admits(pb: &PatchesLinearBounds) -> bool {
    let elements = [
        pb.lower_a
            .patches
            .as_ref()
            .map_or(0, ndarray::ArrayBase::len),
        pb.lower_a.coeff_err.as_ref().map_or(0, Array1::len),
        pb.upper_a
            .patches
            .as_ref()
            .map_or(0, ndarray::ArrayBase::len),
        pb.upper_a.coeff_err.as_ref().map_or(0, Array1::len),
        pb.lower_b.len(),
        pb.upper_b.len(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .unwrap_or(usize::MAX);
    let retained = elements
        .saturating_mul(std::mem::size_of::<f32>())
        .saturating_mul(3);
    retained <= crate::network::crown_memory::cpu_crown_dense_budget_bytes()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_apply_patches_residual_passthrough_with_deadline_authority(
    graph: &GraphNetwork,
    node: &GraphNode,
    node_cb: &mut CrownBounds,
    node_bounds: &HashMap<String, BoundedTensor>,
    node_crown_bounds: &mut CrownMergeAccumulator,
    output_dim: usize,
    input_dim: usize,
    input_accumulated: &mut bool,
    context_prefix: &str,
    deadline: Option<Instant>,
    deadline_is_hard: bool,
) -> Result<bool> {
    // The native residual fan-out clones a complete Patches coefficient
    // carrier. Its legacy helper is fallible but neither polls a deadline nor
    // charges the simultaneously retained source/frontier. A hard finite caller
    // therefore declines before inspecting or mutating the carrier and takes
    // the existing deadline-aware Dense path. An internal soft scheduling
    // timestamp admits the complete legacy fan-out as one indivisible
    // scheduling unit, but does not revoke the historical native route.
    // #residual-patches-expiry: with the lever off this is `deadline_is_hard`
    // exactly — byte-identical to the historical guard. Armed, it refuses only on
    // real EXPIRY, so a scored run with budget left keeps its structured carrier.
    //
    // WHY THIS SITE IS THE ONE THAT MATTERS. The residual `Add` fan-out is where
    // a cifar100 resnet's carrier dies: the demanded pre-activation targets
    // frequently ARE the residual Adds, so the decline fires at step 1 of the
    // target's own walk and every later patches gate is then dead code behind an
    // already-Dense carrier. Measured consequences of the densification:
    // 14400x14400 f32 = 1.659 GB for the pair, and a CPU dense conv backward at
    // 257.56s against 12.97s on GPU. That is the dominant cost-per-alpha-
    // iteration term on the flagship benchmark.
    //
    // This must move TOGETHER with its set-mates (patches_step.rs:269,
    // dispatch.rs:85, nonlinear.rs, target_backward_patches.rs). Partial fixes
    // are worthless here and the repo has the receipt: "the Conv2d decline
    // disappears and the SAME decline reappears one node later at
    // /layers.2/Reshape, then at the next family, until all five sites sharing
    // the predicate are switched together" (REGRESSION_FC_UNSAT_LOST_2026-08-14).
    if patches_finite_authority_refuses(deadline_is_hard, deadline) {
        return Ok(false);
    }

    let pb = match node_cb {
        CrownBounds::Patches(pb) => pb,
        CrownBounds::Dense(_) => return Ok(false),
    };

    // The invariant the original guard was protecting, restated and kept: the
    // legacy fan-out helper neither polls a deadline nor charges the
    // SIMULTANEOUSLY RETAINED source carrier and accumulator frontier —
    // `try_clone_residual_branch` admits against the clone alone. Under a live
    // deadline we therefore pre-charge the retained total before entering it.
    // On this host that matters more than usual: the 121 GiB is shared CPU+GPU
    // unified memory, and an over-admission here is a global OOM, not a refusal.
    if deadline_is_hard && !residual_retained_footprint_admits(pb) {
        return Ok(false);
    }

    let is_add = matches!(&node.layer, Layer::Add(_));
    let is_sub = matches!(&node.layer, Layer::Sub(_));
    if !is_add && !is_sub {
        return Ok(false);
    }

    let (input_a_name, input_b_name) = node.require_binary_inputs()?;

    // Reject broadcasted Add/Sub — only same-shape residual fan-in
    if !residual_shapes_match(node, node_bounds) {
        debug!("{context_prefix}: patches residual passthrough skipped (shape mismatch)");
        return Ok(false);
    }

    // The clone plus its two branch publications form one legacy native unit.
    // Check a soft collector timestamp exactly once before entering it. Passing
    // that timestamp separately to each branch would allow the left insertion
    // to publish and the right insertion to fail solely because the same soft
    // unit crossed its scheduling timestamp in between.
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "graph CROWN residual fan-out: soft scheduling deadline exceeded before admission"
                .into(),
        ));
    }

    // The bias rides the LEFT carrier; the right carrier is zero-bias
    // (#conv-crown-residual). For `y = u + v` a row's relation
    // `A·y + b = A·u + A·v + b` needs `b` counted exactly once, and the two
    // branches are summed where they rejoin — biases outward-rounded in patches
    // form (`bounds/patches/merge/mod.rs:106`) or in f64 after a dense
    // promotion — so `b` on one side and `0` on the other reconstructs it
    // exactly. `Sub` is the same with the right carrier NEGATED (each relation
    // keeps its own coefficients; no lower/upper swap — see
    // `negated_zero_bias` and the false-bound note on the predicate above).
    //
    // The previous route sent `b` to the network-input accumulator instead.
    // That is equally correct, but it allocates a pair of
    // `(output_dim x input_dim)` ZERO matrices to carry a bias vector
    // (`accumulate_bias_to_network_input_crown`), and pins `NETWORK_INPUT` to
    // `Dense` so every later patches carrier arriving there must densify. In
    // the network-output walks that helper was written for, `output_dim` is the
    // handful of spec rows and the cost is invisible. In the per-target walk
    // `output_dim` IS the target dim, so a residual `Add` on
    // CIFAR100_resnet_medium (14400 x 3072) burns 354 MB per Add to hold a
    // 14400-element vector — the exact dense materialization this path exists
    // to avoid.
    let right_branch = match pb.try_clone_residual_branch(is_sub) {
        Ok(branch) => branch,
        Err(NyError::UnsupportedConfiguration(_)) => return Ok(false),
        Err(error) => return Err(error),
    };
    // The right branch above is the only full-size copy. Move the original
    // relation (including its one bias contribution) into the left branch only
    // after every fallible clone/allocation has succeeded, so a refusal leaves
    // the caller's carrier bit-for-bit unchanged.
    let left_carrier = std::mem::replace(node_cb, CrownBounds::Dense(LinearBounds::identity(0)));
    let right_carrier = CrownBounds::Patches(Box::new(right_branch));
    graph.accumulate_crown_bounds_to_input_with_deadline_authority(
        input_a_name,
        left_carrier,
        node_crown_bounds,
        output_dim,
        input_dim,
        input_accumulated,
        None,
        false,
    )?;
    graph.accumulate_crown_bounds_to_input_with_deadline_authority(
        input_b_name,
        right_carrier,
        node_crown_bounds,
        output_dim,
        input_dim,
        input_accumulated,
        None,
        false,
    )?;

    Ok(true)
}

/// Check that both inputs and the node output have the same spatial shape.
fn residual_shapes_match(node: &GraphNode, node_bounds: &HashMap<String, BoundedTensor>) -> bool {
    if node.inputs.len() != 2 {
        return false;
    }
    let a_shape = node_bounds.get(&node.inputs[0]).map(|b| b.shape().to_vec());
    let b_shape = node_bounds.get(&node.inputs[1]).map(|b| b.shape().to_vec());
    match (a_shape, b_shape) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

impl GraphNetwork {
    /// Convenience wrapper: accumulate Dense LinearBounds into a CrownBounds map.
    ///
    /// Wraps the LinearBounds as CrownBounds::Dense and delegates to
    /// [`accumulate_crown_bounds_to_input_with_deadline`]. Used by the Dense dispatch paths
    /// (ReLU, MulBinary, Where, shared dispatch) in the graph engine.
    #[cfg(test)]
    pub(crate) fn accumulate_dense_bounds_to_input(
        &self,
        input_name: &str,
        new_bounds: LinearBounds,
        node_crown_bounds: &mut CrownMergeAccumulator,
        output_dim: usize,
        input_dim: usize,
        input_accumulated: &mut bool,
    ) -> Result<()> {
        self.accumulate_dense_bounds_to_input_with_deadline(
            input_name,
            new_bounds,
            node_crown_bounds,
            output_dim,
            input_dim,
            input_accumulated,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accumulate_dense_bounds_to_input_with_deadline(
        &self,
        input_name: &str,
        new_bounds: LinearBounds,
        node_crown_bounds: &mut CrownMergeAccumulator,
        output_dim: usize,
        input_dim: usize,
        input_accumulated: &mut bool,
        deadline: Option<Instant>,
    ) -> Result<()> {
        self.accumulate_crown_bounds_to_input_with_deadline(
            input_name,
            CrownBounds::Dense(new_bounds),
            node_crown_bounds,
            output_dim,
            input_dim,
            input_accumulated,
            deadline,
        )
    }

    /// Runtime invariant check for #2617/#2530 split-path bounds (#2656).
    ///
    /// `BackwardDispatchResult::Binary`/`Nary` must carry bias only in the
    /// separate channel; each per-input `LinearBounds` path must have zero bias.
    /// Violation would silently double-count bias, producing unsound (too-tight) bounds.
    ///
    /// Originally `debug_assert!`, upgraded to runtime check (#2656): the cost of
    /// iterating the bias vector is negligible vs. the CROWN backward matrix
    /// multiplications, and this invariant guards against a class of severe
    /// soundness bugs (#2520, #2527, #2529, #2530).
    ///
    /// Converted from `assert!` to `Result` (#2907) to eliminate a production
    /// panic cliff — callers can now propagate the error cleanly instead of
    /// crashing the entire verification run.
    #[inline]
    pub(crate) fn verify_split_path_bias_zero(bounds: &LinearBounds, context: &str) -> Result<()> {
        // Tolerance for floating-point rounding artifacts (#2700).
        // Split-path bias must be exactly zero by construction, but accumulated
        // float ops can produce negligible artifacts (e.g., 1e-38).
        const TOLERANCE: f32 = 1e-30;

        for (label, bias) in [("lower_b", bounds.lower_b()), ("upper_b", bounds.upper_b())] {
            // Check NaN explicitly first — IEEE 754 NaN == 0.0 returns false,
            // which would produce the misleading "non-zero" message (#2700).
            if bias.iter().any(|v| v.is_nan()) {
                return Err(NyError::InvalidSpec(format!(
                    "{context} produced NaN in {label} split-path bounds \
                     (NaN corruption in dispatch layer)"
                )));
            }
            let max_abs = bias.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            if max_abs >= TOLERANCE {
                return Err(NyError::InvalidSpec(format!(
                    "{context} produced non-zero {label} in split-path bounds \
                     (max |v| = {max_abs:.2e})"
                )));
            }
        }
        Ok(())
    }

    /// CrownBounds-aware bias accumulation for DAG-CROWN (Phase 1b, #2613).
    ///
    /// Same logic as [`accumulate_bias_to_network_input`] but operates on a
    /// `CrownBounds` map. When inserting a new NETWORK_INPUT entry, wraps as
    /// `CrownBounds::Dense`. When updating existing, the merge accumulator
    /// preflights any required Patches→Dense promotion before allocating the
    /// zero coefficient pair.
    #[cfg(test)]
    pub(crate) fn accumulate_bias_to_network_input_crown(
        bias_lower: &Array1<f32>,
        bias_upper: &Array1<f32>,
        node_crown_bounds: &mut CrownMergeAccumulator,
        output_dim: usize,
        input_dim: usize,
        input_accumulated: &mut bool,
    ) -> Result<()> {
        Self::accumulate_bias_to_network_input_crown_with_deadline(
            bias_lower,
            bias_upper,
            node_crown_bounds,
            output_dim,
            input_dim,
            input_accumulated,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accumulate_bias_to_network_input_crown_with_deadline(
        bias_lower: &Array1<f32>,
        bias_upper: &Array1<f32>,
        node_crown_bounds: &mut CrownMergeAccumulator,
        output_dim: usize,
        input_dim: usize,
        input_accumulated: &mut bool,
        deadline: Option<Instant>,
    ) -> Result<()> {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(
                "graph CROWN bias accumulation: deadline exceeded before publication".into(),
            ));
        }
        if *input_accumulated {
            node_crown_bounds.merge_dense_bias_with_deadline(
                NETWORK_INPUT,
                bias_lower,
                bias_upper,
                output_dim,
                input_dim,
                deadline,
            )?;
        } else {
            let lb = CrownMergeAccumulator::try_dense_bias_bounds_with_deadline(
                bias_lower, bias_upper, output_dim, input_dim, deadline,
            )?;
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return Err(NyError::DeadlineExceeded(
                    "graph CROWN bias accumulation: deadline exceeded before publication".into(),
                ));
            }
            node_crown_bounds.insert(NETWORK_INPUT.to_string(), CrownBounds::Dense(lb));
            *input_accumulated = true;
        }
        Ok(())
    }
}
