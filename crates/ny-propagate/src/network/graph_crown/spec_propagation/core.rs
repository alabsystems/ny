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
use crate::bounds::patches::{CrownBounds, PatchesMaterializationPurpose};
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::Layer;
use crate::network::core::{apply_dense_backward_dispatch_result_with_deadline, GraphNetwork};
use crate::network::crown_memory::{cpu_crown_dense_budget_bytes, DenseMaterializationEstimate};
use crate::network::{merge_reference_bound_maps, CrownMergeAccumulator};
use crate::types::{CrownBackwardResult, CrownIbpFallbackReason};
use crate::MulBinaryRelaxationMode;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::mem::size_of;
use std::time::{Duration, Instant};
use tracing::{debug, info};

use super::super::helpers::is_softmax_decomposition_mul;
use super::fallback::fallback_to_ibp_with_reason;
use super::patches::PatchesDispatchOutcome;

type MnReluDenseFailure = (CrownIbpFallbackReason, Option<DenseMaterializationEstimate>);

/// Materialize an incoming Patches carrier only after the established dense
/// pair budget admits it. The returned estimate lets the caller report the
/// exact refusal without attempting the allocation.
fn ensure_multineuron_relu_dense(
    node_cb: &mut CrownBounds,
    budget: usize,
    deadline: Option<Instant>,
) -> std::result::Result<(), MnReluDenseFailure> {
    if let CrownBounds::Patches(patches) = &*node_cb {
        let (rows, cols) = patches
            .dense_pair_shape()
            .map_err(|_| (CrownIbpFallbackReason::CrownPropagationError, None))?;
        let estimate = DenseMaterializationEstimate::new("spec_crown_multineuron_relu", rows, cols);
        if estimate.exceeds_budget(budget) {
            return Err((CrownIbpFallbackReason::MemoryBudgetExceeded, Some(estimate)));
        }
        node_cb
            .ensure_dense_with_deadline_for_purpose(deadline, PatchesMaterializationPurpose::Other)
            .map_err(|error| match error {
                NyError::CpuMemoryExceeded { .. } => {
                    (CrownIbpFallbackReason::MemoryBudgetExceeded, None)
                }
                NyError::DeadlineExceeded(_) => {
                    (CrownIbpFallbackReason::PerNodeDeadlineExceeded, None)
                }
                _ => (CrownIbpFallbackReason::CrownPropagationError, None),
            })?;
    }
    Ok(())
}

/// Capture one backward carrier without cloning structured Patches before its
/// checked Dense materialization.  Publication occurs only after conversion
/// succeeds, so a typed resource/malformed refusal leaves both source and map
/// untouched.
fn capture_node_linear_bounds(
    linear_bounds_map: &mut std::collections::HashMap<String, LinearBounds>,
    node_name: &str,
    node_cb: &CrownBounds,
    node_box: Option<&BoundedTensor>,
    deadline: Option<Instant>,
) -> Result<()> {
    if deadline.is_none() {
        let captured = match node_cb {
            CrownBounds::Dense(bounds) => bounds.clone(),
            CrownBounds::Patches(bounds) => {
                bounds.to_dense_for_purpose(PatchesMaterializationPurpose::Other)?
            }
        };
        linear_bounds_map.insert(node_name.to_string(), captured);
        return Ok(());
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(format!(
            "spec-guided CROWN: deadline exceeded before capturing node '{node_name}'"
        )));
    }
    const SITE: &str = "spec-guided CROWN finite linear-cache capture";
    let budget_bytes = cpu_crown_dense_budget_bytes();
    let entry_bytes = size_of::<(String, LinearBounds)>().saturating_add(size_of::<usize>());
    let mut retained_map_bytes = linear_bounds_map.capacity().saturating_mul(entry_bytes);
    for (index, (name, bounds)) in linear_bounds_map.iter().enumerate() {
        retained_map_bytes = retained_map_bytes
            .saturating_add(name.capacity())
            .saturating_add(bounds.memory_bytes());
        if index.is_multiple_of(4_096) && deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(format!(
                "{SITE}: deadline exceeded while scanning retained captures"
            )));
        }
    }
    let nominal_entry_bytes = entry_bytes.saturating_add(node_name.len());
    if retained_map_bytes.saturating_add(nominal_entry_bytes) > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: retained_map_bytes.saturating_add(nominal_entry_bytes),
            budget_bytes,
            site: SITE,
        });
    }
    linear_bounds_map
        .try_reserve(1)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: retained_map_bytes.saturating_add(nominal_entry_bytes),
            budget_bytes,
            site: SITE,
        })?;
    retained_map_bytes = linear_bounds_map
        .capacity()
        .saturating_mul(entry_bytes)
        .saturating_add(
            linear_bounds_map
                .iter()
                .fold(0usize, |sum, (name, bounds)| {
                    sum.saturating_add(name.capacity())
                        .saturating_add(bounds.memory_bytes())
                }),
        );
    let mut captured_name = String::new();
    captured_name
        .try_reserve_exact(node_name.len())
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: retained_map_bytes.saturating_add(nominal_entry_bytes),
            budget_bytes,
            site: SITE,
        })?;
    captured_name.push_str(node_name);
    let retained_base_bytes = retained_map_bytes.saturating_add(captured_name.capacity());
    if retained_base_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: retained_base_bytes,
            budget_bytes,
            site: SITE,
        });
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(format!(
            "{SITE}: deadline exceeded before carrier copy"
        )));
    }
    let mut captured = match node_cb {
        CrownBounds::Dense(bounds) => {
            bounds.try_clone_with_deadline(deadline, retained_base_bytes)?
        }
        CrownBounds::Patches(bounds) => bounds.to_dense_with_deadline_and_resident_for_purpose(
            deadline,
            retained_base_bytes,
            PatchesMaterializationPurpose::Other,
        )?,
    };
    if captured.has_coeff_err() {
        let node_box = node_box.ok_or_else(|| {
            NyError::UnsupportedConfiguration(format!(
                "{SITE}: no output box is available to fold coefficient-error state at '{node_name}'"
            ))
        })?;
        captured.fold_coeff_err_over_box_eager_with_deadline(node_box, deadline)?;
        if captured.has_coeff_err() {
            return Err(NyError::UnsupportedConfiguration(format!(
                "{SITE}: CachedLinearBounds cannot preserve non-finite coefficient-error state at '{node_name}'"
            )));
        }
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(format!(
            "spec-guided CROWN: deadline exceeded after capturing node '{node_name}'"
        )));
    }
    linear_bounds_map.insert(captured_name, captured);
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        linear_bounds_map.remove(node_name);
        return Err(NyError::DeadlineExceeded(format!(
            "spec-guided CROWN: deadline exceeded before publishing node capture '{node_name}'"
        )));
    }
    Ok(())
}

/// Preserve semantic errors while routing structured resource refusals to the
/// established whole-request IBP fallback.
fn resource_ibp_fallback_reason(result: Result<()>) -> Result<Option<CrownIbpFallbackReason>> {
    match result {
        Ok(()) => Ok(None),
        Err(NyError::CpuMemoryExceeded { .. }) => {
            Ok(Some(CrownIbpFallbackReason::MemoryBudgetExceeded))
        }
        Err(NyError::DeadlineExceeded(_)) => {
            Ok(Some(CrownIbpFallbackReason::PerNodeDeadlineExceeded))
        }
        Err(error) => Err(error),
    }
}

/// Batteries-included gate for the C-matrix-seeded GPU resnet ROOT pass
/// (#w4-root-gpu): ON by default, opt out with `NY_SPEC_ROOT_GPU=0` for A/B
/// measurement (disable-flag principle).
pub(super) fn spec_root_gpu_enabled() -> bool {
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

/// Batteries-included gate for the alpha-fed forward-linear C-margin rebuild
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

/// Exact f32 endpoint bits for the sealed cGAN forward-alpha mechanism marker.
/// This is called only behind `NY_PHASE_TELEMETRY=1`.
fn cgan_forward_alpha_endpoint_bits<'a>(values: impl Iterator<Item = &'a f32>) -> String {
    use std::fmt::Write as _;

    let mut out = String::from("[");
    for (index, value) in values.enumerate() {
        if index != 0 {
            out.push(',');
        }
        write!(&mut out, "0x{:08x}", value.to_bits())
            .expect("writing endpoint bits to String cannot fail");
    }
    out.push(']');
    out
}

/// Pure formatter for the typed cGAN forward-alpha canary marker.
///
/// `None` is returned before any formatting when telemetry is dark. When it is
/// enabled, the helper independently checks that `intersected` is the exact
/// per-element intersection/union-fallback of `prior` and `rebuilt`; malformed
/// shapes, NaNs, or a mismatched result decline the marker rather than claiming
/// `certified-rebuild-intersected`. This is observation-only and never feeds a
/// bound, schedule, or verdict.
#[allow(clippy::too_many_arguments)]
fn cgan_forward_alpha_marker_line_if(
    enabled: bool,
    specs: usize,
    rows: usize,
    sweeps: usize,
    moved: usize,
    interior: usize,
    baseline: f64,
    predicted: f64,
    prior: &BoundedTensor,
    rebuilt: &BoundedTensor,
    intersected: &BoundedTensor,
    elapsed: Duration,
) -> Option<String> {
    if !enabled {
        return None;
    }
    if prior.shape() != rebuilt.shape() || prior.shape() != intersected.shape() {
        return None;
    }

    let mut tightened_lower = 0usize;
    let mut tightened_upper = 0usize;
    let mut disjoint = 0usize;
    for (
        ((&prior_lower, &prior_upper), (&rebuilt_lower, &rebuilt_upper)),
        (&merged_lower, &merged_upper),
    ) in prior
        .lower()
        .iter()
        .zip(prior.upper().iter())
        .zip(rebuilt.lower().iter().zip(rebuilt.upper().iter()))
        .zip(intersected.lower().iter().zip(intersected.upper().iter()))
    {
        if [
            prior_lower,
            prior_upper,
            rebuilt_lower,
            rebuilt_upper,
            merged_lower,
            merged_upper,
        ]
        .iter()
        .any(|value| value.is_nan())
        {
            return None;
        }

        let overlap_lower = prior_lower.max(rebuilt_lower);
        let overlap_upper = prior_upper.min(rebuilt_upper);
        let (expected_lower, expected_upper) = if overlap_lower <= overlap_upper {
            (overlap_lower, overlap_upper)
        } else {
            disjoint += 1;
            (
                prior_lower.min(rebuilt_lower),
                prior_upper.max(rebuilt_upper),
            )
        };
        if merged_lower.to_bits() != expected_lower.to_bits()
            || merged_upper.to_bits() != expected_upper.to_bits()
        {
            return None;
        }
        tightened_lower += usize::from(merged_lower > prior_lower);
        tightened_upper += usize::from(merged_upper < prior_upper);
    }

    Some(format!(
        "[phase] cgan-forward-alpha status=certified-rebuild-intersected \
         specs={specs} rows={rows} sweeps={sweeps} moved={moved} interior={interior} \
         baseline={baseline:.17e} predicted={predicted:.17e} \
         tightened_lower={tightened_lower} tightened_upper={tightened_upper} \
         disjoint={disjoint} \
         prior_lower_bits={} prior_upper_bits={} \
         rebuilt_lower_bits={} rebuilt_upper_bits={} \
         intersected_lower_bits={} intersected_upper_bits={} elapsed_ms={}",
        cgan_forward_alpha_endpoint_bits(prior.lower().iter()),
        cgan_forward_alpha_endpoint_bits(prior.upper().iter()),
        cgan_forward_alpha_endpoint_bits(rebuilt.lower().iter()),
        cgan_forward_alpha_endpoint_bits(rebuilt.upper().iter()),
        cgan_forward_alpha_endpoint_bits(intersected.lower().iter()),
        cgan_forward_alpha_endpoint_bits(intersected.upper().iter()),
        elapsed.as_millis(),
    ))
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
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        // Even with precomputed node boxes, producing specification bounds
        // requires an O(spec_rows * output_width) projection. Do not start
        // that fallback (or execution-plan/seed allocation) after expiry.
        return Err(NyError::DeadlineExceeded(
            "spec-guided CROWN: deadline exceeded before request setup".to_string(),
        ));
    }

    // Empty graph fast path: spec matrix applied directly to input bounds.
    if let Some(result) =
        super::fallback::empty_graph_fast_path(graph, spec_matrix, input, deadline)?
    {
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
    let seed_lb = match super::fallback::spec_seed_with_deadline(spec_matrix, deadline) {
        Ok(bounds) => bounds,
        Err(NyError::CpuMemoryExceeded { .. }) => {
            return fallback_to_ibp_with_reason(
                graph,
                input,
                spec_matrix,
                node_bounds,
                output_node_name,
                CrownIbpFallbackReason::MemoryBudgetExceeded,
                deadline,
            );
        }
        Err(error) => return Err(error),
    };

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
    if deadline.is_none() && crown_backward_layers.is_none() && !wants_input_linear && !has_mn_pool
    {
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
        // #cgan-fwdlin-ref (default-ON, opt out with the shared or
        // ConvTranspose-specific reference kill switch):
        // sequential ConvTranspose chains (cgan) are image-capable too — the
        // looser/slower measurement above predates the certified ConvTranspose
        // surface and covered sequential families WITHOUT it. Scoped to the
        // shared image policy, so those measured families keep the proven
        // spec-CROWN backward loop. This is what lets every PER-DOMAIN
        // input-split spec evaluation pick up the forward-linear C-margin
        // candidate on cgan.
        let image_forward_linear = graph.should_collect_forward_linear_image_reference();
        if image_forward_linear && spec_root_margin_enabled() {
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
        if image_forward_linear
            && spec_root_margin_enabled()
            && spec_root_alpha_enabled()
            && graph.forward_linear_spec_alpha_enabled()
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
                        Some(fixed) => {
                            // Default-dark, observation-only mechanism marker.
                            // Preserve the pre-intersection candidate only when
                            // telemetry is enabled; the production path keeps
                            // the original single intersection and allocation
                            // behavior. Emit only after the certified rebuild
                            // has passed through `intersect_sound`.
                            if crate::phase_telemetry::phase_telemetry_enabled() {
                                let intersected = intersect_sound(fixed.clone(), &bounds);
                                if let Some(line) = cgan_forward_alpha_marker_line_if(
                                    true,
                                    num_specs,
                                    stats.rows,
                                    stats.sweeps,
                                    stats.moved,
                                    stats.interior,
                                    stats.baseline_min,
                                    stats.predicted_min,
                                    &fixed,
                                    &bounds,
                                    &intersected,
                                    rebuild_start.elapsed(),
                                ) {
                                    eprintln!("{line}");
                                }
                                intersected
                            } else {
                                intersect_sound(fixed, &bounds)
                            }
                        }
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
            deadline,
        )
    };

    let input_dim = input.len();
    let mut input_accumulated = false;
    // CachedLinearBounds does not carry coefficient-error matrices. A live
    // multi-neuron pre-activation term can create exactly such an error after
    // the ReLU, so publishing that cache would silently drop proof state.
    // Refuse cache capture for the whole request while a nonempty pool is live.
    // Error-carrying multi-neuron relations cannot be represented by
    // `CachedLinearBounds`. Ordinary finite captures use the checked copy and
    // publication seams below; any resource refusal drops the whole optional
    // candidate rather than returning a partial cache.
    let cache_capture_allowed = capture_linear_cache && !has_mn_pool;
    let mut captured_linear_bounds = cache_capture_allowed.then(std::collections::HashMap::new);
    let mut cache_capture_valid = cache_capture_allowed;

    // Per-node deadline budgeting (#3795): same policy as propagation.rs.
    const SPEC_CROWN_MAX_BUDGET_FRACTION: f64 = 0.25;
    const SPEC_CROWN_MIN_NODE_BUDGET_SECS: f64 = 2.0;
    let total_backward_nodes = plan.node_count();
    let mut backward_steps = 0usize;
    // #patches-drop (dark, NY_PATCHES_CARRIER_TRACE=1, print-only): publish this
    // walk's position so a `[patches-drop]` line emitted deep inside the
    // materializer names the node whose carrier densified.
    let carrier_trace = crate::patches_carrier_trace::enabled();

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
                deadline,
            );
        }

        let node = nodes_by_idx[idx];
        let mut node_cb = match node_crown_bounds.take_by_idx_with_deadline(idx, node_deadline) {
            Ok(Some(cb)) => cb,
            Ok(None) => continue,
            Err(NyError::CpuMemoryExceeded { .. }) => {
                return ibp_fallback(CrownIbpFallbackReason::MemoryBudgetExceeded);
            }
            Err(NyError::DeadlineExceeded(_)) => {
                return ibp_fallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded);
            }
            Err(error) => return Err(error),
        };
        backward_steps += 1;
        if carrier_trace {
            crate::patches_carrier_trace::enter_node("spec-crown", node_name);
        }

        if let Some(ref mut linear_bounds_map) = captured_linear_bounds {
            match capture_node_linear_bounds(
                linear_bounds_map,
                node_name,
                &node_cb,
                node_bounds.get(node_name),
                node_deadline,
            ) {
                Ok(()) => {}
                Err(NyError::CpuMemoryExceeded { .. }) => {
                    return ibp_fallback(CrownIbpFallbackReason::MemoryBudgetExceeded);
                }
                Err(NyError::DeadlineExceeded(_)) => {
                    return ibp_fallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded);
                }
                Err(error) => return Err(error),
            }
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

        // #3813: Dense→Patches re-entry at unary Conv2d boundaries. A live
        // multi-neuron group anchored at a ReLU requires the dense pre/post
        // handshake below; the Patches ReLU fast path has no representation for
        // those extra terms. Preserve an incoming Patches relation through the
        // conv chain, then materialize it exactly at the anchored ReLU instead
        // of silently skipping the pool.
        let mn_relu_requires_dense = matches!(&node.layer, Layer::ReLU(_))
            && mn_pool.is_some_and(|pool| {
                pool.groups().iter().any(|group| {
                    group.beta().is_finite() && group.beta() > 0.0 && group.anchor() == node_name
                })
            });
        if !mn_relu_requires_dense {
            super::super::backward_node_dispatch::try_patches_reentry(
                &mut node_cb,
                node,
                node_bounds,
                node_name,
                graph.use_patches_mode,
                "Spec-guided CROWN",
                node_deadline,
            );
        }
        // This bit follows the carrier conversion itself. A verifier deadline
        // on an ordinary Dense relation is not structured-boundary authority.
        let mut finite_structured_boundary = false;
        if mn_relu_requires_dense {
            let materializes_patches = matches!(&node_cb, CrownBounds::Patches(_));
            let budget = cpu_crown_dense_budget_bytes();
            if let Err((reason, estimate)) =
                ensure_multineuron_relu_dense(&mut node_cb, budget, node_deadline)
            {
                if let Some(estimate) = estimate {
                    info!(
                        "Spec-guided CROWN: {}; falling back to IBP before multi-neuron ReLU densification",
                        estimate.budget_exceeded_details(budget)
                    );
                }
                return ibp_fallback(reason);
            }
            finite_structured_boundary |= materializes_patches && node_deadline.is_some();
        }

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
            )? {
                PatchesDispatchOutcome::AccumulateToInput => {
                    if let Some(reason) = resource_ibp_fallback_reason(
                        graph.accumulate_crown_bounds_to_input_with_deadline(
                            first_input,
                            node_cb,
                            &mut node_crown_bounds,
                            num_specs,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ),
                    )? {
                        return ibp_fallback(reason);
                    }
                    continue;
                }
                PatchesDispatchOutcome::IbpFallback(reason) => {
                    return ibp_fallback(reason);
                }
                PatchesDispatchOutcome::FallThroughDense => {
                    finite_structured_boundary = node_deadline.is_some();
                }
            }
        }

        let materializes_patches = matches!(&node_cb, CrownBounds::Patches(_));
        let mut node_lb = match node_cb.into_dense_with_deadline_for_purpose(
            node_deadline,
            PatchesMaterializationPurpose::Other,
        ) {
            Ok(bounds) => bounds,
            Err(NyError::CpuMemoryExceeded { .. }) => {
                return ibp_fallback(CrownIbpFallbackReason::MemoryBudgetExceeded);
            }
            Err(NyError::DeadlineExceeded(_)) => {
                return ibp_fallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded);
            }
            Err(error) => return Err(error),
        };
        finite_structured_boundary |= materializes_patches && node_deadline.is_some();

        // These coordinator-owned branches bypass the shared dispatcher.
        // Keep them historical for ordinary finite Dense carriers, but do not
        // let an actual finite Patches materialization enter their unchecked
        // scans/allocations.
        if finite_structured_boundary
            && matches!(
                &node.layer,
                Layer::ReLU(_) | Layer::MulBinary(_) | Layer::Div(_)
            )
        {
            return ibp_fallback(CrownIbpFallbackReason::CrownPropagationError);
        }

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
            // Track only groups whose first half actually committed. Skipped
            // groups must not receive the price after relaxation, while every
            // committed group must either complete or fail closed.
            let mut mn_committed = Vec::new();
            if let Some(pool) = mn_pool {
                for (group_index, group) in pool.groups().iter().enumerate() {
                    if group.inject_post_terms_before_relu(&mut node_lb, node_name, group.beta())
                        == crate::multineuron::MnInjectOutcome::Injected
                    {
                        mn_committed.push(group_index);
                    }
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
                        for &group_index in &mn_committed {
                            let group = &pool.groups()[group_index];
                            if !group.inject_pre_terms_after_relu(
                                &mut bounds,
                                node_name,
                                group.beta(),
                            ) {
                                bounds.degrade_lower_to_vacuous();
                            }
                        }
                    }
                    if let Some(reason) = resource_ibp_fallback_reason(
                        graph.accumulate_crown_bounds_to_input_with_deadline(
                            first_input,
                            CrownBounds::Dense(*bounds),
                            &mut node_crown_bounds,
                            num_specs,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ),
                    )? {
                        return ibp_fallback(reason);
                    }
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
                concretized_node_bias_with_deadline, dispatch_mul_binary_backward,
                MulBinaryDispatchCtx, MulBinaryDispatchResult,
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
                    if let Some(reason) = resource_ibp_fallback_reason(
                        GraphNetwork::accumulate_bias_to_network_input_crown_with_deadline(
                            &bias_lower,
                            &bias_upper,
                            &mut node_crown_bounds,
                            num_specs,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ),
                    )? {
                        return ibp_fallback(reason);
                    }
                    if let Some(reason) = resource_ibp_fallback_reason(
                        graph.accumulate_crown_bounds_to_input_with_deadline(
                            input_a_name,
                            CrownBounds::Dense(*bounds_a),
                            &mut node_crown_bounds,
                            num_specs,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ),
                    )? {
                        return ibp_fallback(reason);
                    }
                    if let Some(reason) = resource_ibp_fallback_reason(
                        graph.accumulate_crown_bounds_to_input_with_deadline(
                            input_b_name,
                            CrownBounds::Dense(*bounds_b),
                            &mut node_crown_bounds,
                            num_specs,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ),
                    )? {
                        return ibp_fallback(reason);
                    }
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
                    let bias = match concretized_node_bias_with_deadline(
                        &node_lb,
                        node_ibp,
                        node_deadline,
                    ) {
                        Ok(bias) => bias,
                        Err(NyError::CpuMemoryExceeded { .. }) => {
                            return ibp_fallback(CrownIbpFallbackReason::MemoryBudgetExceeded);
                        }
                        Err(NyError::DeadlineExceeded(_)) => {
                            return ibp_fallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded);
                        }
                        Err(error) => return Err(error),
                    };
                    if let Some(reason) = resource_ibp_fallback_reason(
                        GraphNetwork::accumulate_bias_to_network_input_crown_with_deadline(
                            &bias.lower,
                            &bias.upper,
                            &mut node_crown_bounds,
                            num_specs,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ),
                    )? {
                        return ibp_fallback(reason);
                    }
                }
            }
            continue;
        }

        // === Binary Div: numerator-only backward with reciprocal scaling (#3626, #3499) ===
        // Math documented in backward_node_dispatch::backward_div_to_numerator.
        if matches!(&node.layer, Layer::Div(_)) {
            use super::super::backward_node_dispatch::{
                backward_div_to_numerator_with_deadline, DivBackwardResult,
            };

            let (input_a_name, input_b_name) = node.require_binary_inputs()?;
            let input_a_bounds = graph.bounds_ref(input_a_name, input, node_bounds)?;
            let input_b_bounds = graph.bounds_ref(input_b_name, input, node_bounds)?;
            let node_ibp = graph.bounds_ref(node_name, input, node_bounds)?;

            let div_result = match backward_div_to_numerator_with_deadline(
                &node_lb,
                input_a_bounds,
                input_b_bounds,
                node_ibp,
                node_deadline,
            ) {
                Ok(result) => result,
                Err(NyError::CpuMemoryExceeded { .. }) => {
                    return ibp_fallback(CrownIbpFallbackReason::MemoryBudgetExceeded);
                }
                Err(NyError::DeadlineExceeded(_)) => {
                    return ibp_fallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded);
                }
                Err(error) => return Err(error),
            };
            match div_result {
                DivBackwardResult::PropagateNumerator(bounds) => {
                    if let Some(reason) = resource_ibp_fallback_reason(
                        graph.accumulate_crown_bounds_to_input_with_deadline(
                            input_a_name,
                            CrownBounds::Dense(*bounds),
                            &mut node_crown_bounds,
                            num_specs,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ),
                    )? {
                        return ibp_fallback(reason);
                    }
                }
                DivBackwardResult::ConcretizeCurrentNode(bias) => {
                    cache_capture_valid = false;
                    if let Some(reason) = resource_ibp_fallback_reason(
                        GraphNetwork::accumulate_bias_to_network_input_crown_with_deadline(
                            &bias.lower,
                            &bias.upper,
                            &mut node_crown_bounds,
                            num_specs,
                            input_dim,
                            &mut input_accumulated,
                            node_deadline,
                        ),
                    )? {
                        return ibp_fallback(reason);
                    }
                }
            }
            continue;
        }

        // === All other layers: shared dispatch core (#1949 Step B, #3935) ===
        use super::super::backward_node_dispatch::{
            concretized_node_bias_with_deadline, dispatch_shared_core, SharedDispatchCtx,
            SharedDispatchResult,
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
            finite_structured_boundary,
            mul_binary_relaxation,
            label: "Spec-guided CROWN",
        };
        match dispatch_shared_core(&shared_ctx)? {
            SharedDispatchResult::Dispatch(result) => {
                if let Some(reason) = resource_ibp_fallback_reason(
                    apply_dense_backward_dispatch_result_with_deadline(
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
                        node_deadline,
                    ),
                )? {
                    return ibp_fallback(reason);
                }
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
                let bias =
                    match concretized_node_bias_with_deadline(&node_lb, node_ibp, node_deadline) {
                        Ok(bias) => bias,
                        Err(NyError::CpuMemoryExceeded { .. }) => {
                            return ibp_fallback(CrownIbpFallbackReason::MemoryBudgetExceeded);
                        }
                        Err(NyError::DeadlineExceeded(_)) => {
                            return ibp_fallback(CrownIbpFallbackReason::PerNodeDeadlineExceeded);
                        }
                        Err(error) => return Err(error),
                    };
                if let Some(resource_reason) = resource_ibp_fallback_reason(
                    GraphNetwork::accumulate_bias_to_network_input_crown_with_deadline(
                        &bias.lower,
                        &bias.upper,
                        &mut node_crown_bounds,
                        num_specs,
                        input_dim,
                        &mut input_accumulated,
                        node_deadline,
                    ),
                )? {
                    return ibp_fallback(resource_reason);
                }
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
        deadline,
    )
}

#[cfg(test)]
mod telemetry_tests {
    use super::*;
    use crate::bounds::patches::{PatchGeometry, PatchesData, PatchesLinearBounds};
    use ndarray::{arr1, Array1, ArrayD, IxDyn};

    fn bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn()).unwrap()
    }

    #[test]
    fn spec_merge_error_classifier_preserves_resource_authority() {
        assert_eq!(resource_ibp_fallback_reason(Ok(())).unwrap(), None);
        assert_eq!(
            resource_ibp_fallback_reason(Err(NyError::CpuMemoryExceeded {
                required_bytes: 8,
                budget_bytes: 4,
                site: "test",
            }))
            .unwrap(),
            Some(CrownIbpFallbackReason::MemoryBudgetExceeded)
        );
        assert_eq!(
            resource_ibp_fallback_reason(Err(NyError::DeadlineExceeded("test".into()))).unwrap(),
            Some(CrownIbpFallbackReason::PerNodeDeadlineExceeded)
        );
        let semantic =
            resource_ibp_fallback_reason(Err(NyError::InvalidSpec("semantic".to_string())))
                .expect_err("semantic errors must remain typed");
        assert!(matches!(semantic, NyError::InvalidSpec(_)));
    }

    fn anchored_capture_fixture() -> PatchesLinearBounds {
        let geometry =
            PatchGeometry::anchored(vec![0, 1], vec![0, 1]).expect("fixture axes are non-empty");
        let data = PatchesData {
            coeff_err: None,
            patches: Some(
                ArrayD::from_shape_vec(IxDyn(&[1, 2, 2, 1, 1, 1]), vec![0.25, 0.5, 0.75, 1.0])
                    .expect("fixture shape and values agree"),
            ),
            geometry,
            identity: false,
            output_shape: (1, 2, 2),
            input_shape: (1, 2, 2),
            unstable_idx: None,
        };
        PatchesLinearBounds {
            row_count: 4,
            lower_a: data.clone(),
            lower_b: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
            upper_a: data,
            upper_b: Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]),
        }
    }

    fn assert_capture_patches_exact(actual: &PatchesLinearBounds, expected: &PatchesLinearBounds) {
        fn assert_data(actual: &PatchesData, expected: &PatchesData) {
            assert_eq!(actual.coeff_err, expected.coeff_err);
            assert_eq!(actual.patches, expected.patches);
            assert_eq!(actual.geometry, expected.geometry);
            assert_eq!(actual.identity, expected.identity);
            assert_eq!(actual.output_shape, expected.output_shape);
            assert_eq!(actual.input_shape, expected.input_shape);
            assert_eq!(actual.unstable_idx, expected.unstable_idx);
        }

        assert_eq!(actual.row_count, expected.row_count);
        assert_data(&actual.lower_a, &expected.lower_a);
        assert_eq!(actual.lower_b, expected.lower_b);
        assert_data(&actual.upper_a, &expected.upper_a);
        assert_eq!(actual.upper_b, expected.upper_b);
    }

    #[test]
    fn multineuron_relu_patches_densification_is_budget_transactional() {
        let make_patches = || {
            CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
                (1, 2, 2),
                (1, 2, 2),
            )))
        };

        let mut refused = make_patches();
        let (reason, estimate) = ensure_multineuron_relu_dense(&mut refused, 0, None).unwrap_err();
        assert_eq!(reason, CrownIbpFallbackReason::MemoryBudgetExceeded);
        assert!(estimate.is_some());
        assert!(
            matches!(refused, CrownBounds::Patches(_)),
            "an over-budget refusal must occur before Patches allocation state changes"
        );

        let mut admitted = make_patches();
        ensure_multineuron_relu_dense(&mut admitted, usize::MAX, None).unwrap();
        assert!(matches!(admitted, CrownBounds::Dense(_)));
    }

    #[test]
    fn captured_patches_budget_refusal_is_borrowed_and_atomic() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "0");

            let expected = anchored_capture_fixture();
            let carrier = CrownBounds::Patches(Box::new(expected.clone()));
            let mut captures = std::collections::HashMap::new();
            captures.insert("sentinel".to_string(), LinearBounds::identity(1));

            let error = capture_node_linear_bounds(&mut captures, "anchored", &carrier, None, None)
                .expect_err("zero budget must refuse captured Patches materialization");
            assert!(
                matches!(error, NyError::CpuMemoryExceeded { .. }),
                "expected typed memory refusal, got {error:?}"
            );
            match &carrier {
                CrownBounds::Patches(actual) => assert_capture_patches_exact(actual, &expected),
                CrownBounds::Dense(_) => panic!("borrowed capture changed the source carrier"),
            }
            assert_eq!(captures.len(), 1);
            assert!(captures.contains_key("sentinel"));
            assert!(!captures.contains_key("anchored"));
        });
    }

    /// #fl-alpha-composition monotone floor: the α-optimized certified
    /// rebuild is only ever published through `intersect_sound(fixed, …)`,
    /// so a poisoned/degraded α candidate can NEVER drag the root margins
    /// below the fixed FL bound — the fixed candidate is the floor — while
    /// a genuinely tighter candidate still tightens element-wise.
    #[test]
    fn alpha_rebuild_can_never_publish_below_the_fixed_fl_floor() {
        let fixed = bounds(&[-15.76, -8.0], &[10.0, 12.0]);

        // Poisoned ascent: a strictly WEAKER (wider) sound enclosure.
        let poisoned = bounds(&[-100.0, -90.0], &[100.0, 90.0]);
        let merged = intersect_sound(fixed.clone(), &poisoned);
        for (m, f) in merged.lower().iter().zip(fixed.lower().iter()) {
            assert_eq!(m.to_bits(), f.to_bits(), "floor: lower must stay fixed-FL");
        }
        for (m, f) in merged.upper().iter().zip(fixed.upper().iter()) {
            assert_eq!(m.to_bits(), f.to_bits(), "floor: upper must stay fixed-FL");
        }

        // Productive ascent: a strictly tighter candidate tightens.
        let tighter = bounds(&[-12.0, -9.0], &[8.0, 11.0]);
        let merged = intersect_sound(fixed.clone(), &tighter);
        assert_eq!(merged.lower()[[0]], -12.0, "row 0 tightened by α");
        assert_eq!(merged.lower()[[1]], -8.0, "row 1 keeps the FL floor");
        assert_eq!(merged.upper()[[0]], 8.0);
        assert_eq!(merged.upper()[[1]], 11.0);

        // Shape mismatch fails open to the fixed candidate (sound either way).
        let mismatched = bounds(&[-1.0], &[1.0]);
        let merged = intersect_sound(fixed.clone(), &mismatched);
        for (m, f) in merged.lower().iter().zip(fixed.lower().iter()) {
            assert_eq!(m.to_bits(), f.to_bits());
        }
    }

    #[test]
    fn cgan_forward_alpha_marker_is_default_dark_before_formatting() {
        let prior = bounds(&[-2.0, -1.0], &[3.0, 4.0]);
        let rebuilt = bounds(&[-1.5, -2.0], &[2.5, 3.0]);
        let intersected = intersect_sound(prior.clone(), &rebuilt);

        assert_eq!(
            cgan_forward_alpha_marker_line_if(
                false,
                2,
                2,
                3,
                7,
                4,
                1.0,
                2.0,
                &prior,
                &rebuilt,
                &intersected,
                Duration::from_millis(1234),
            ),
            None,
            "the default-dark path must decline before endpoint formatting"
        );
    }

    #[test]
    fn cgan_forward_alpha_marker_formats_post_intersection_evidence_exactly() {
        let prior = bounds(&[-2.0, -1.0], &[3.0, 4.0]);
        let rebuilt = bounds(&[-1.5, -2.0], &[2.5, 3.0]);
        let intersected = intersect_sound(prior.clone(), &rebuilt);

        let line = cgan_forward_alpha_marker_line_if(
            true,
            2,
            2,
            3,
            7,
            4,
            1.0,
            2.0,
            &prior,
            &rebuilt,
            &intersected,
            Duration::from_millis(1234),
        )
        .expect("valid intersected bounds must produce the mechanism marker");

        assert_eq!(
            line,
            concat!(
                "[phase] cgan-forward-alpha status=certified-rebuild-intersected ",
                "specs=2 rows=2 sweeps=3 moved=7 interior=4 ",
                "baseline=1.00000000000000000e0 predicted=2.00000000000000000e0 ",
                "tightened_lower=1 tightened_upper=2 disjoint=0 ",
                "prior_lower_bits=[0xc0000000,0xbf800000] ",
                "prior_upper_bits=[0x40400000,0x40800000] ",
                "rebuilt_lower_bits=[0xbfc00000,0xc0000000] ",
                "rebuilt_upper_bits=[0x40200000,0x40400000] ",
                "intersected_lower_bits=[0xbfc00000,0xbf800000] ",
                "intersected_upper_bits=[0x40200000,0x40400000] elapsed_ms=1234"
            )
        );
    }
}
