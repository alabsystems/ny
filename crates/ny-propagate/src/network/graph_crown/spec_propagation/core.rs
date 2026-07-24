// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core backward coordinator loop for spec-guided CROWN.
//!
//! This module owns the main backward propagation loop with explicit
//! `Layer::{Div, Linear, MulBinary, ReLU}` checks for dispatch-coverage
//! tooling visibility. IBP fallback/finalization lives in [`super::fallback`],
//! patches flow control in [`super::patches`]. Split from the original
//! monolithic `spec_propagation.rs` as part of #3960.

use crate::batched_domain::CachedLinearBounds;
use crate::bounds::patches::CrownBounds;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::Layer;
use crate::network::core::{apply_dense_backward_dispatch_result, GraphNetwork};
use crate::network::{merge_reference_bound_maps, CrownMergeAccumulator};
use crate::types::{CrownBackwardResult, CrownIbpFallbackReason};
use crate::MulBinaryRelaxationMode;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::{debug, info};

use super::super::helpers::is_softmax_decomposition_mul;
use super::fallback::fallback_to_ibp_with_reason;
use super::patches::PatchesDispatchOutcome;

/// Batteries-included gate for the C-matrix-seeded GPU resnet ROOT pass
/// (#w4-root-gpu): ON by default, opt out with `NY_SPEC_ROOT_GPU=0` for A/B
/// measurement (disable-flag principle).
fn spec_root_gpu_enabled() -> bool {
    !matches!(std::env::var("NY_SPEC_ROOT_GPU").ok().as_deref(), Some("0"))
}

/// Batteries-included gate for the forward-linear C-margin ROOT composition
/// (#w4-root-margin): ON by default, opt out with `NY_SPEC_ROOT_MARGIN=0`.
fn spec_root_margin_enabled() -> bool {
    !matches!(
        std::env::var("NY_SPEC_ROOT_MARGIN").ok().as_deref(),
        Some("0")
    )
}

/// Batteries-included gate for the ALPHA-FED forward-linear C-margin rebuild
/// (#w4-root-alpha): ON by default, opt out with `NY_SPEC_ROOT_ALPHA=0`.
fn spec_root_alpha_enabled() -> bool {
    !matches!(
        std::env::var("NY_SPEC_ROOT_ALPHA").ok().as_deref(),
        Some("0")
    )
}

/// Sound per-element intersection of two enclosures of the same spec values.
/// Falls back to `a` on shape mismatch or NaN (both operands are sound, so
/// keeping either is sound).
fn intersect_sound(a: BoundedTensor, b: &BoundedTensor) -> BoundedTensor {
    if a.shape() == b.shape() {
        a.intersection_per_element(b).map(|(t, _)| t).unwrap_or(a)
    } else {
        a
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn propagate_crown_with_specs_and_engine_with_linear_and_reference_bounds_and_deadline_and_truncation(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec_matrix: &ndarray::Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    mul_binary_relaxation: MulBinaryRelaxationMode,
    precomputed_node_bounds: Option<&std::collections::HashMap<String, BoundedTensor>>,
    reference_node_bounds: Option<&std::collections::HashMap<String, BoundedTensor>>,
    alpha_state: Option<&GraphAlphaState>,
    deadline: Option<Instant>,
    mul_binary_alphas: Option<&std::collections::HashMap<String, ndarray::Array2<f32>>>,
    capture_linear_cache: bool,
    crown_backward_layers: Option<usize>,
    wants_input_linear: bool,
    mn_pool: Option<&crate::multineuron::MultiNeuronPool>,
) -> Result<(
    CrownBackwardResult,
    Option<LinearBounds>,
    Option<CachedLinearBounds>,
)> {
    if precomputed_node_bounds.is_some() && reference_node_bounds.is_some() {
        return Err(NyError::UnsupportedConfiguration(
            "spec-guided CROWN cannot combine fixed precomputed_node_bounds with reference_node_bounds"
                .to_string(),
        ));
    }

    // Empty graph fast path: spec matrix applied directly to input bounds.
    if let Some(result) = super::fallback::empty_graph_fast_path(graph, spec_matrix, input)? {
        return Ok(result);
    }

    let num_specs = spec_matrix.nrows();
    let spec_output_dim = spec_matrix.ncols();

    let exec_order = graph.exec_order()?;
    let plan = graph.dispatch_plan()?;

    // Use pre-computed bounds if provided (e.g., alpha-CROWN optimized bounds),
    // otherwise compute internally via CROWN-IBP or IBP.
    let computed_node_bounds;
    let reference_merged_node_bounds;
    let node_bounds = if let Some(precomputed) = precomputed_node_bounds {
        precomputed
    } else {
        computed_node_bounds =
            super::setup::collect_intermediate_bounds(graph, input, deadline, engine)?;
        if let Some(reference) = reference_node_bounds {
            reference_merged_node_bounds =
                merge_reference_bound_maps(Some(&computed_node_bounds), Some(reference))?
                    .ok_or_else(|| {
                        NyError::InternalError(
                            "merging fresh and reference node bounds produced None".into(),
                        )
                    })?;
            &reference_merged_node_bounds
        } else {
            &computed_node_bounds
        }
    };

    let output_node_name =
        super::setup::resolve_output_contract(graph, exec_order, node_bounds, spec_output_dim)?;
    debug_assert_eq!(plan.index_of(output_node_name), Some(plan.output_node_idx));

    let nodes_by_idx = super::setup::collect_nodes_by_idx(graph, exec_order)?;
    let seed_lb = LinearBounds::from_spec_matrix(spec_matrix.clone())?;

    // #w4-root-gpu: C-matrix-seeded sound GPU resnet ROOT pass. The multi-objective
    // root evaluation (99-row C matrix on cifar100) previously had NO GPU route:
    // the CPU backward loop below deadline-died mid-graph and fell back to IBP, so
    // the root objective bounds came from the per-logit forward-linear projection —
    // which loses the pairwise logit correlation and can never verify margin
    // objectives. Seeding the proven sound GPU-resident resnet backward with the
    // FULL spec matrix (the reference's approach) keeps that correlation and runs
    // in <1s. Same certified machinery as every resnet suffix call: sound-only
    // engine, certified f32 error, explosion auto-fallback inside the backend, and
    // `Ok(None)`/`Err` → the proven CPU loop below (fail-closed, 0-wrong moat).
    // Skipped under explicit backward truncation (a caller-requested semantic).
    // The IBP intersection below guarantees the result is never looser than IBP,
    // mirroring `finalize_backward_output`. Linear/cache capture is not available
    // on this route (concrete bounds only) — honest `None`s are returned.
    //
    // ALSO skipped when the caller asked for the input `LinearBounds`
    // (`run_with_linear`, #w5-bab-throughput): the root-candidate early return
    // carries `None` linear, which silently defeats every linear-extraction
    // caller — measured on cifar100: the PGD exact-gradient path (#4274) ran a
    // full ~13-25s certified forward-linear walk at a CONCRETE point (single-use
    // cache key, polluting the root-box cache) only to receive `None` and fall
    // back to SPSA. Those callers need the CPU backward loop below, which is the
    // only route that produces the linear map. Bounds-only callers (root passes,
    // prechecks) keep the fast root candidates.
    // Multi-neuron injection (increment 3, §2.2): a group facet can only be
    // carried by the CPU backward ReLU arm below, so a non-empty pool DISABLES
    // the bounds-only GPU/forward-linear root fast paths (which have no ReLU arm
    // to inject into). Sound either way — the fast paths are a tighter-or-equal
    // enclosure of the SAME margin; forcing the CPU loop only forgoes their speed
    // to gain the coupling-facet tightening.
    let has_mn_pool = mn_pool.is_some_and(|p| !p.is_empty());
    if crown_backward_layers.is_none() && !wants_input_linear && !has_mn_pool {
        let mut root_candidate: Option<BoundedTensor> = None;

        // (a) Forward-linear C-margin composition (#w4-root-margin): compose the
        // spec matrix with the OUTPUT node's certified forward-linear affine map
        // (cached; the composition itself is a tiny certified f64 GEMM). This
        // keeps the cross-output correlation the per-logit projection loses —
        // margin rows cancel coefficients BEFORE concretization. Conv2d-DAG only
        // (the cifar100/tinyimagenet image surface, where the CPU backward loop
        // below deadline-dies); fail-closed on refusal.
        //
        // Cost-gated to the IMAGE surface (frac-head audit, 4726b45b): on
        // conv1d-only DAGs (nn4sys pensieve pow graphs) this candidate is
        // measured BOTH looser (~1.13-1.30x per-node width growth after the
        // first crossing ReLU, compounding through Pow to ~4 units/head of
        // root slack — enough to flip a 105-instance root-verifiable family
        // to timeout) AND slower (fresh O(L) dense forward state per input,
        // 41ms vs the 7ms full backward). Those graphs keep the proven
        // spec-CROWN backward loop below, which is feasible at their scale.
        let conv_dag = graph.has_conv_layers()
            && graph
                .exec_order()
                .map(|order| !graph.is_sequential_graph(order))
                .unwrap_or(false);
        // #cgan-fwdlin-ref (DARK, `NY_FORWARD_LINEAR_CONV_TRANSPOSE_REF=1`):
        // sequential ConvTranspose chains (cgan) are image-capable too — the
        // looser/slower measurement above predates the certified ConvTranspose
        // surface and covered sequential families WITHOUT it. Scoped to the
        // dark surface gate + actual ConvTranspose presence, so those measured
        // families keep the proven spec-CROWN backward loop; gate-off is
        // byte-identical. This is what lets every PER-DOMAIN input-split spec
        // evaluation pick up the forward-linear C-margin candidate on cgan.
        let seq_conv_transpose_chain =
            GraphNetwork::forward_linear_conv_transpose_reference_enabled()
                && graph.has_conv2d_layers()
                && graph.has_conv_transpose2d_layers();
        let image_dag = (conv_dag && graph.has_conv2d_layers()) || seq_conv_transpose_chain;
        if image_dag
            && spec_root_margin_enabled()
            && GraphNetwork::forward_linear_reference_enabled()
        {
            match graph.forward_linear_spec_margin_bounds(input, spec_matrix, engine, deadline) {
                Ok(bounds) => {
                    info!(
                        num_specs,
                        "Spec-guided CROWN: forward-linear C-margin root bounds (#w4-root-margin)"
                    );
                    // #cgan-fwdlin-ref diagnostics (dark; probe-gated): margin
                    // tightness per evaluation, sampled so a deep BaB run stays
                    // readable — first 20 calls, then every 500th. Answers
                    // "how close does the per-domain C-margin get with depth".
                    if std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_PROBE")
                        .ok()
                        .as_deref()
                        == Some("1")
                    {
                        use std::sync::atomic::{AtomicUsize, Ordering};
                        static CMARGIN_CALLS: AtomicUsize = AtomicUsize::new(0);
                        let n = CMARGIN_CALLS.fetch_add(1, Ordering::Relaxed);
                        if n < 20 || n.is_multiple_of(500) {
                            let worst =
                                bounds.lower().iter().copied().fold(f32::INFINITY, f32::min);
                            let width: f32 = bounds
                                .lower()
                                .iter()
                                .zip(bounds.upper().iter())
                                .map(|(&l, &u)| u - l)
                                .fold(0.0_f32, f32::max);
                            eprintln!(
                                "[fwdlin-cmargin] call={n} worst_lower={worst:.6} max_width={width:.6} in_w={:.6}",
                                input
                                    .lower()
                                    .iter()
                                    .zip(input.upper().iter())
                                    .map(|(&l, &u)| u - l)
                                    .fold(0.0_f32, f32::max)
                            );
                        }
                    }
                    root_candidate = Some(bounds);
                }
                Err(
                    error @ (NyError::UnsupportedOp(_)
                    | NyError::UnsupportedConfiguration(_)
                    | NyError::DeadlineExceeded(_)
                    | NyError::ShapeMismatch { .. }
                    | NyError::CpuMemoryExceeded { .. }),
                ) => {
                    debug!(
                        %error,
                        "Spec-guided CROWN: forward-linear C-margin unavailable (fail-closed)"
                    );
                }
                Err(error) => return Err(error),
            }
        }

        // (b) C-matrix-seeded sound GPU resnet backward (#w4-root-gpu).
        if spec_root_gpu_enabled() {
            match crate::network::graph_alpha::resnet_decompose::try_resnet_gpu_suffix(
                graph,
                input,
                output_node_name,
                node_bounds,
                node_bounds,
                alpha_state,
                engine,
                deadline,
                &seed_lb,
            ) {
                Ok(Some(gpu_bounds)) => {
                    info!(
                        num_specs,
                        "Spec-guided CROWN: C-matrix root pass decided on sound GPU resnet backward (#w4-root-gpu)"
                    );
                    root_candidate = Some(match root_candidate {
                        // Both are sound enclosures of the same spec values —
                        // the per-element intersection is sound and tightest.
                        Some(margin) => intersect_sound(margin, &gpu_bounds),
                        None => gpu_bounds,
                    });
                }
                Ok(None) => {}
                Err(error) => {
                    // Unexpected internal error from the GPU route (reshape/
                    // repair): fail closed onto the proven CPU backward loop.
                    debug!(
                        %error,
                        "Spec-guided CROWN: GPU resnet root pass errored; taking CPU backward"
                    );
                }
            }
        }

        // (c) Forward-map ALPHA OPTIMIZER + certified rebuild
        // (#w4-root-alpha-opt): the fixed-slope map in (a) uses the ADAPTIVE
        // lower ReLU slopes. W4-7 measured that the warmup's alphas (optimized
        // for the GPU-backward relaxation) are ~8-10x LOOSER for the forward
        // map, so this optimizes per-neuron slopes directly against the
        // forward-linear C-margin objective of the unverified rows (cheap
        // point-evaluation surrogate + one certified rebuild — see
        // `forward_linear::alpha_opt`). Every candidate map is sound for any
        // α ∈ [0, 1], so the element-wise intersection with (a)/(b) is sound
        // and never-worse. Self-budgeted: skips without cost when the fixed
        // cache is cold, headroom cannot fit the rebuild, or the optimizer
        // predicts no improvement. Deliberately LAST: the cheap (a)/(b)
        // candidates must never be starved of deadline by it.
        if image_dag
            && spec_root_margin_enabled()
            && spec_root_alpha_enabled()
            && GraphNetwork::forward_linear_reference_enabled()
        {
            let rebuild_start = Instant::now();
            match graph.forward_linear_alpha_optimized_spec_margin_bounds(
                input,
                spec_matrix,
                root_candidate.as_ref(),
                engine,
                deadline,
            ) {
                Ok(Some((bounds, stats))) => {
                    let worst =
                        |b: &BoundedTensor| b.lower().iter().copied().fold(f32::INFINITY, f32::min);
                    info!(
                        num_specs,
                        elapsed_ms = rebuild_start.elapsed().as_millis() as u64,
                        rows = stats.rows,
                        sweeps = stats.sweeps,
                        moved = stats.moved,
                        interior = stats.interior,
                        surrogate_baseline_min = stats.baseline_min,
                        surrogate_predicted_min = stats.predicted_min,
                        alpha_worst_lower = worst(&bounds),
                        fixed_worst_lower =
                            root_candidate.as_ref().map(worst).unwrap_or(f32::NAN),
                        "Spec-guided CROWN: alpha-OPTIMIZED forward-linear C-margin root bounds (#w4-root-alpha-opt)"
                    );
                    root_candidate = Some(match root_candidate {
                        Some(fixed) => intersect_sound(fixed, &bounds),
                        None => bounds,
                    });
                }
                Ok(None) => {}
                Err(
                    error @ (NyError::UnsupportedOp(_)
                    | NyError::UnsupportedConfiguration(_)
                    | NyError::DeadlineExceeded(_)
                    | NyError::ShapeMismatch { .. }
                    | NyError::CpuMemoryExceeded { .. }),
                ) => {
                    debug!(
                        %error,
                        elapsed_ms = rebuild_start.elapsed().as_millis() as u64,
                        "Spec-guided CROWN: alpha-optimized C-margin unavailable (fail-closed)"
                    );
                }
                Err(error) => return Err(error),
            }
        }

        if let Some(bounds) = root_candidate {
            let ibp_spec_bounds = graph.propagate_crown_with_specs_fallback_ibp(
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
            )?;
            let tightened = crate::network::tighten_crown_output(
                bounds,
                &ibp_spec_bounds,
                "Spec-guided CROWN (root candidates)",
            )?;
            return Ok((
                CrownBackwardResult {
                    bounds: tightened,
                    provenance: crate::types::BoundsProvenance::Crown,
                },
                None,
                None,
            ));
        }
    }

    let mut node_crown_bounds = CrownMergeAccumulator::new_indexed(exec_order);

    node_crown_bounds.insert(output_node_name.to_string(), CrownBounds::Dense(seed_lb));

    // Shared IBP fallback closure — every fallback path needs the same
    // graph/input/spec/bounds context, only the reason differs.
    let ibp_fallback = |reason: CrownIbpFallbackReason| {
        fallback_to_ibp_with_reason(
            graph,
            input,
            spec_matrix,
            node_bounds,
            output_node_name,
            reason,
        )
    };

    let input_dim = input.len();
    let mut input_accumulated = false;
    let mut captured_linear_bounds = capture_linear_cache.then(std::collections::HashMap::new);
    let mut cache_capture_valid = capture_linear_cache;

    // Per-node deadline budgeting (#3795): same policy as propagation.rs.
    const SPEC_CROWN_MAX_BUDGET_FRACTION: f64 = 0.25;
    const SPEC_CROWN_MIN_NODE_BUDGET_SECS: f64 = 2.0;
    let total_backward_nodes = plan.node_count();
    let mut backward_steps = 0usize;

    for (rev_pos, &idx) in plan.reverse_order.iter().enumerate() {
        let node_name = plan.name_of(idx);

        // Deadline enforcement: check before each node's backward pass.
        // For deep models (e.g., malbeware 16-25: 16 layers x 24 specs),
        // the full backward pass can take 100-200s. Falling back to IBP
        // when the deadline is exceeded ensures timeout compliance. #3218/#3328
        if deadline.is_some_and(|d| Instant::now() >= d) {
            info!(
                "Spec-guided CROWN: deadline exceeded at node '{}', falling back to IBP",
                node_name
            );
            return ibp_fallback(CrownIbpFallbackReason::DeadlineExceeded);
        }

        // Compute per-node deadline for this backward step (#3795).
        let node_deadline = super::super::backward_node_dispatch::compute_node_deadline(
            deadline,
            rev_pos,
            total_backward_nodes,
            SPEC_CROWN_MAX_BUDGET_FRACTION,
            SPEC_CROWN_MIN_NODE_BUDGET_SECS,
        );

        // If the overall deadline expires during budget calculation, bail to IBP for
        // the remaining backward pass. Sub-floor node shares keep the global deadline
        // so CROWN LinearBounds are preserved on short-budget tiny graphs (#3881).
        if deadline.is_some() && node_deadline.is_none() {
            info!(
                "Spec-guided CROWN: deadline expired while budgeting '{}' ({}/{} nodes), falling back to IBP",
                node_name,
                rev_pos + 1,
                total_backward_nodes,
            );
            return ibp_fallback(CrownIbpFallbackReason::DeadlineExceeded);
        }

        if crown_backward_layers.is_some_and(|max_layers| backward_steps >= max_layers) {
            info!(
                "Spec-guided CROWN: truncating backward after {} nodes at frontier '{}'",
                backward_steps, node_name
            );
            return super::fallback::truncation_early_return(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                &mut node_crown_bounds,
                num_specs,
                input_dim,
                &mut input_accumulated,
            );
        }

        let node = nodes_by_idx[idx];
        let mut node_cb = match node_crown_bounds.take_by_idx(idx)? {
            Some(cb) => cb,
            None => continue,
        };
        backward_steps += 1;

        if let Some(ref mut linear_bounds_map) = captured_linear_bounds {
            let captured_lb = match &node_cb {
                CrownBounds::Dense(lb) => lb.clone(),
                CrownBounds::Patches(_) => node_cb.clone().into_dense()?,
            };
            linear_bounds_map.insert(node_name.to_string(), captured_lb);
        }

        let first_input_idx = plan.first_input_idx(idx);
        let first_input = plan.name_of(first_input_idx);
        let pre_activation = if plan.is_network_input(first_input_idx) {
            input
        } else {
            node_bounds.get(first_input).ok_or_else(|| {
                NyError::InvalidSpec(format!("Pre-activation bounds for {first_input} not found"))
            })?
        };

        // #3813: Dense→Patches re-entry at unary Conv2d boundaries.
        super::super::backward_node_dispatch::try_patches_reentry(
            &mut node_cb,
            node,
            node_bounds,
            node_name,
            graph.use_patches_mode,
            "Spec-guided CROWN",
        );

        // Patches fast-path: dispatch in patches mode if applicable, with
        // ensure_dense() downgrade on failure. Flow control extracted to
        // patches.rs as part of #3960.
        if matches!(&node_cb, CrownBounds::Patches(_)) && node.inputs.len() == 1 {
            match super::patches::dispatch_patches_or_fallback(
                &mut node_cb,
                &node.layer,
                pre_activation,
                engine,
                node_deadline,
                node_name,
                node.layer.layer_type(),
            ) {
                PatchesDispatchOutcome::AccumulateToInput => {
                    graph.accumulate_crown_bounds_to_input(
                        first_input,
                        node_cb,
                        &mut node_crown_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    )?;
                    continue;
                }
                PatchesDispatchOutcome::IbpFallback(reason) => {
                    return ibp_fallback(reason);
                }
                PatchesDispatchOutcome::FallThroughDense => {}
            }
        }

        let mut node_lb = node_cb.into_dense()?;

        // === Linear: pre-dispatch dimension check with IBP fallback (#2817, #3935) ===
        // Explicit Layer::Linear guard kept for dispatch-coverage tooling visibility.
        if matches!(&node.layer, Layer::Linear(_))
            && super::super::backward_node_dispatch::linear_dimension_mismatch(node, &node_lb)
        {
            return ibp_fallback(CrownIbpFallbackReason::ShapeMismatch);
        }

        // === ReLU: heuristic relaxation via shared dispatch (#3935) ===
        if matches!(&node.layer, Layer::ReLU(_)) {
            use super::super::backward_node_dispatch::{
                dispatch_relu_backward, NodeDispatchResult,
            };
            // Multi-neuron §2.2 step 1: inject each group facet's post-activation
            // terms `+β_c·g_i` onto this ReLU's OUTPUT columns BEFORE relaxation,
            // so they ride `propagate_linear_with_alpha` exactly like a β-split.
            // Only groups anchored at THIS ReLU node inject (the term filter). No
            // effect when `mn_pool` is None or every β_c is 0 (default).
            if let Some(pool) = mn_pool {
                for g in pool.groups() {
                    g.inject_post_terms_before_relu(&mut node_lb, node_name, g.beta());
                }
            }
            let expanded_relu_alpha = alpha_state.and_then(|state| {
                state.relu_alpha_pair(node_name).map(|(lower, upper)| {
                    (
                        state.expand_alpha(node_name, lower),
                        state.expand_alpha(node_name, upper),
                    )
                })
            });
            let (alpha_lower, alpha_upper) = expanded_relu_alpha
                .as_ref()
                .map_or((None, None), |(lower, upper)| (Some(lower), Some(upper)));
            match dispatch_relu_backward(
                graph.cut_fold_scope(),
                node,
                &node_lb,
                pre_activation,
                node_name,
                "Spec-guided CROWN",
                alpha_lower,
                alpha_upper,
            )? {
                NodeDispatchResult::SingleDense(mut bounds) => {
                    // Multi-neuron §2.2 steps 2+3: inject the pre-activation terms
                    // `+β_c·a_i` directly onto this ReLU's INPUT columns of the
                    // relaxed carrier (bypassing the ReLU) and fold `−β_c·b_c` into
                    // the lower bias (outward). Same anchored-node filter.
                    if let Some(pool) = mn_pool {
                        for g in pool.groups() {
                            g.inject_pre_terms_after_relu(&mut bounds, node_name, g.beta());
                        }
                    }
                    graph.accumulate_crown_bounds_to_input(
                        first_input,
                        CrownBounds::Dense(*bounds),
                        &mut node_crown_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    )?;
                }
                NodeDispatchResult::IbpFallback(reason) => {
                    return ibp_fallback(reason);
                }
            }
            continue;
        }

        // === MulBinary: site-specific (relaxation mode, IBP fallback) ===
        // Shared dispatch returns Unsupported for MulBinary because it requires
        // a relaxation mode parameter. Handle here to use the caller-provided
        // mode instead of falling back to IBP. (#3389)
        if matches!(&node.layer, Layer::MulBinary(_)) {
            use super::super::backward_node_dispatch::{
                concretized_node_bias, dispatch_mul_binary_backward, MulBinaryDispatchCtx,
                MulBinaryDispatchResult,
            };

            let (input_a_name, input_b_name) = node.require_binary_inputs()?;
            let input_a_bounds = graph.bounds_ref(input_a_name, input, node_bounds)?;
            let input_b_bounds = graph.bounds_ref(input_b_name, input, node_bounds)?;

            let dispatch_ctx = MulBinaryDispatchCtx {
                node,
                node_name,
                node_lb: &node_lb,
                input_a_bounds,
                input_b_bounds,
                mul_binary_relaxation,
                mul_binary_alpha: mul_binary_alphas.and_then(|m| m.get(node_name)),
                softmax_decomposition: is_softmax_decomposition_mul(graph, node),
                label: "Spec-guided CROWN",
            };
            match dispatch_mul_binary_backward(&dispatch_ctx)? {
                MulBinaryDispatchResult::BinaryDense {
                    bounds_a,
                    bounds_b,
                    bias_lower,
                    bias_upper,
                } => {
                    GraphNetwork::accumulate_bias_to_network_input_crown(
                        &bias_lower,
                        &bias_upper,
                        &mut node_crown_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                    graph.accumulate_crown_bounds_to_input(
                        input_a_name,
                        CrownBounds::Dense(*bounds_a),
                        &mut node_crown_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    )?;
                    graph.accumulate_crown_bounds_to_input(
                        input_b_name,
                        CrownBounds::Dense(*bounds_b),
                        &mut node_crown_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    )?;
                }
                MulBinaryDispatchResult::SoftmaxNonFinite => {
                    return ibp_fallback(CrownIbpFallbackReason::CrownPropagationError);
                }
                // #3602/#3596: Per-node IBP concretization for unsupported/error cases.
                // Concretize this MulBinary node's contribution using its IBP bounds
                // instead of falling back the entire spec CROWN to pure IBP.
                MulBinaryDispatchResult::RecoverableError(err) => {
                    let node_ibp = node_bounds.get(node_name).ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "IBP bounds for MulBinary node '{}' not found",
                            node_name
                        ))
                    })?;
                    debug!(
                        "Spec-guided CROWN: MulBinary '{}' recoverable error ({}), concretizing per-node IBP",
                        node_name, err,
                    );
                    cache_capture_valid = false;
                    let bias = concretized_node_bias(&node_lb, node_ibp);
                    GraphNetwork::accumulate_bias_to_network_input_crown(
                        &bias.lower,
                        &bias.upper,
                        &mut node_crown_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
            }
            continue;
        }

        // === Binary Div: numerator-only backward with reciprocal scaling (#3626, #3499) ===
        // Math documented in backward_node_dispatch::backward_div_to_numerator.
        if matches!(&node.layer, Layer::Div(_)) {
            use super::super::backward_node_dispatch::{
                backward_div_to_numerator, DivBackwardResult,
            };

            let (input_a_name, input_b_name) = node.require_binary_inputs()?;
            let input_a_bounds = graph.bounds_ref(input_a_name, input, node_bounds)?;
            let input_b_bounds = graph.bounds_ref(input_b_name, input, node_bounds)?;
            let node_ibp = graph.bounds_ref(node_name, input, node_bounds)?;

            match backward_div_to_numerator(&node_lb, input_a_bounds, input_b_bounds, node_ibp)? {
                DivBackwardResult::PropagateNumerator(bounds) => {
                    graph.accumulate_crown_bounds_to_input(
                        input_a_name,
                        CrownBounds::Dense(*bounds),
                        &mut node_crown_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    )?;
                }
                DivBackwardResult::ConcretizeCurrentNode(bias) => {
                    cache_capture_valid = false;
                    GraphNetwork::accumulate_bias_to_network_input_crown(
                        &bias.lower,
                        &bias.upper,
                        &mut node_crown_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                    );
                }
            }
            continue;
        }

        // === All other layers: shared dispatch core (#1949 Step B, #3935) ===
        use super::super::backward_node_dispatch::{
            concretized_node_bias, dispatch_shared_core, SharedDispatchCtx, SharedDispatchResult,
        };
        let shared_ctx = SharedDispatchCtx {
            node,
            node_name,
            node_lb: &node_lb,
            pre_activation,
            network_input: input,
            node_bounds,
            engine,
            node_deadline,
            mul_binary_relaxation,
            label: "Spec-guided CROWN",
        };
        match dispatch_shared_core(&shared_ctx)? {
            SharedDispatchResult::Dispatch(result) => {
                apply_dense_backward_dispatch_result(
                    graph,
                    node,
                    first_input,
                    &node_lb,
                    *result,
                    &mut node_crown_bounds,
                    num_specs,
                    input_dim,
                    &mut input_accumulated,
                    "Spec dispatch",
                )?;
            }
            SharedDispatchResult::IbpFallback(reason) => {
                // PerNodeDeadlineExceeded: full IBP fallback (don't continue per-node).
                if reason == CrownIbpFallbackReason::PerNodeDeadlineExceeded {
                    return ibp_fallback(reason);
                }
                // Per-node IBP concretization for unsupported/error cases (#3596).
                // Concretize this node's contribution using pre-computed IBP bounds
                // instead of falling back the entire spec CROWN to pure IBP.
                // Sound: concretize_sound computes min/max(A * x + b) for x ∈ [l, u].
                let node_ibp = node_bounds.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!("IBP bounds for node '{}' not found", node_name))
                })?;
                debug!(
                    "Spec-guided CROWN: {} ({}) fallback {:?}, concretizing per-node IBP",
                    node_name,
                    node.layer.layer_type(),
                    reason,
                );
                cache_capture_valid = false;
                let bias = concretized_node_bias(&node_lb, node_ibp);
                GraphNetwork::accumulate_bias_to_network_input_crown(
                    &bias.lower,
                    &bias.upper,
                    &mut node_crown_bounds,
                    num_specs,
                    input_dim,
                    &mut input_accumulated,
                );
            }
        }
    }

    // Final output assembly: extract NETWORK_INPUT bounds, concretize, guard
    // against non-finite, tighten with IBP, and package cached linear bounds.
    super::fallback::finalize_backward_output(
        graph,
        input,
        spec_matrix,
        node_bounds,
        output_node_name,
        node_crown_bounds,
        captured_linear_bounds,
        cache_capture_valid,
        num_specs,
    )
}
