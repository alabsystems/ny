// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dense CROWN backward-step helpers and materialization budget guards.

use crate::bounds::patches::CrownBounds;
use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::network::backward_dispatch::{
    dispatch_backward_layer, dispatch_backward_layer_finite_boundary, BackwardDispatchResult,
    DispatchContext,
};
use crate::network::crown_memory::{cpu_crown_dense_budget_bytes, DenseMaterializationEstimate};
use crate::types::{CrownIbpFallbackReason, MulBinaryRelaxationMode};
use crate::BoundPropagation;
use ndarray::Ix1;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info, warn};

use super::{CrownStepFallback, CrownStepResult};

pub(super) fn dense_identity_budget_estimate(
    site: &'static str,
    dim: usize,
) -> Option<DenseMaterializationEstimate> {
    let estimate = DenseMaterializationEstimate {
        site,
        rows: dim,
        cols: dim,
        required_bytes: LinearBounds::identity_pair_bytes(dim).unwrap_or(usize::MAX),
    };
    estimate
        .exceeds_budget(cpu_crown_dense_budget_bytes())
        .then_some(estimate)
}

pub(super) fn dense_materialization_budget_estimate(
    crown_bounds: &CrownBounds,
    site: &'static str,
) -> Result<Option<DenseMaterializationEstimate>> {
    match crown_bounds {
        CrownBounds::Dense(_) => Ok(None),
        CrownBounds::Patches(pb) => {
            let (rows, cols) = pb.dense_pair_shape()?;
            let estimate = DenseMaterializationEstimate::new(site, rows, cols);
            Ok(estimate
                .exceeds_budget(cpu_crown_dense_budget_bytes())
                .then_some(estimate))
        }
    }
}

pub(super) fn log_dense_materialization_budget_fallback(
    label: &str,
    estimate: DenseMaterializationEstimate,
    layer_idx: Option<usize>,
    layer_type: Option<&str>,
) {
    let details = estimate.budget_exceeded_details(cpu_crown_dense_budget_bytes());
    match (layer_idx, layer_type) {
        (Some(idx), Some(kind)) => {
            info!("{label}: layer {idx} ({kind}) {details}; falling back to IBP")
        }
        _ => info!("{label}: {details}; falling back to IBP"),
    }
}

fn memory_budget_step_fallback(
    estimate: DenseMaterializationEstimate,
    layer_idx: usize,
    layer_type: &str,
) -> CrownStepFallback {
    CrownStepFallback {
        reason: CrownIbpFallbackReason::MemoryBudgetExceeded,
        details: format!(
            "layer {layer_idx} ({layer_type}) {}",
            estimate.budget_exceeded_details(cpu_crown_dense_budget_bytes())
        ),
    }
}

/// Whether a finite authority must refuse `crown_backward_ibp_concretize` on a
/// TYPED LAYER ERROR (`UnsupportedOp` / `UnsupportedConfiguration` /
/// `NumericalInstability` / `ShapeMismatch`).
///
/// EXPIRY, not existence — the same rule `crown_backward_relu` below already
/// follows. `0a6a9bfe2` moved only the ReLU arm on measured receipts and left
/// its siblings on presence, so this one step decided the identical question
/// two different ways. This closes that split for the typed-error arms; it is a
/// coherence repair with no separate measurement of its own (census and the
/// arm-by-arm reasoning: docs/DEADLINE_PRESENCE_GUARD_CENSUS_2026-08-17.md).
///
/// Scope, and it is the whole reason this is a named predicate rather than an
/// inlined test: these two arms are reachable ONLY from a real error raised
/// inside a layer kernel. `dispatch_backward_layer_finite_boundary_inner`
/// never raises a typed Err — its deliberate refusals arrive as
/// `Ok(BackwardDispatchResult::Unsupported)`, handled separately below and
/// deliberately still on presence.
///
/// Pollability argument, strictly weaker than the one already accepted for
/// ReLU: the declined work is `crown_backward_ibp_concretize`, already fully
/// deadline-plumbed — `check_crown_backward_deadline` brackets the per-layer
/// IBP on both sides, then `concretize_sound_with_deadline` and
/// `constant_linear_from_concretized_with_deadline` carry the authority through
/// allocation, arithmetic, and publication. The only uninterruptible window is
/// ONE single-layer `Layer::propagate_ibp`, smaller than the dense ReLU
/// backward window expiry already permits.
///
/// What declining costs: the caller maps `IbpFallback` to
/// `clone_crown_forward_fallback(output_bounds)` (crown.rs), so the whole
/// backward walk is discarded and the row keeps the CROWN-IBP output bounds it
/// entered with. Per-layer concretization instead keeps the CROWN prefix
/// accumulated so far — which is why it exists. Every no-deadline run has taken
/// that route for months.
///
/// ERROR-CHANNEL WIDENING, stated because it is the real cost of this flip.
/// Under a live deadline these two arms previously returned
/// `Ok(CrownStepResult::IbpFallback(..))` unconditionally; they can now return
/// `Err`. The `DeadlineExceeded` half is pre-existing — already reachable at the
/// per-layer check below and via the ReLU arm since `0a6a9bfe2` — but the
/// `layer.propagate_ibp(..)?` failure becomes NEWLY reachable under a deadline.
/// No test covers "typed dispatch error + live deadline + `propagate_ibp` also
/// fails"; the arm is entered precisely because a layer's CROWN backward is
/// unsupported, so an unsupported IBP on the same layer is not far-fetched. It
/// matches the no-deadline behaviour the tree has run for months, but that
/// behaviour now also runs on the lane that scores.
///
/// CALLER ASYMMETRY, a fact about existing code rather than a request to change
/// it: `sequential/crown.rs` consumes `crown_backward_step_patches(..)` with a
/// bare `?` while its sibling sites in the same function map
/// `Err(DeadlineExceeded)` to `clone_crown_forward_fallback`. With
/// `verifier/network.rs` also using `?`, an escaping `Err` surfaces as a hard
/// verification error rather than `NetworkBoundsResult::Timeout`.
#[inline]
fn finite_authority_refuses_per_layer_ibp(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|value| Instant::now() >= value)
}

fn memory_error_step_fallback(error: &NyError, layer_idx: usize, layer: &Layer) -> CrownStepResult {
    CrownStepResult::IbpFallback(CrownStepFallback {
        reason: CrownIbpFallbackReason::MemoryBudgetExceeded,
        details: format!(
            "layer {layer_idx} ({}) exceeded the CPU CROWN budget: {error}",
            layer.layer_type()
        ),
    })
}

pub(super) fn guard_dense_materialization_budget(
    crown_bounds: &CrownBounds,
    site: &'static str,
    layer: &Layer,
    layer_idx: usize,
    label: &str,
) -> Result<Option<CrownStepResult>> {
    let Some(estimate) = dense_materialization_budget_estimate(crown_bounds, site)? else {
        return Ok(None);
    };
    log_dense_materialization_budget_fallback(
        label,
        estimate,
        Some(layer_idx),
        Some(layer.layer_type()),
    );
    Ok(Some(CrownStepResult::IbpFallback(
        memory_budget_step_fallback(estimate, layer_idx, layer.layer_type()),
    )))
}

/// Execute one CROWN backward propagation step for a single layer (Dense only).
///
/// Called directly by `propagate_crown_fast` and `propagate_crown_with_linear`.
/// Called indirectly by `propagate_crown_with_engine` (and `propagate_crown_ibp`)
/// via [`super::crown_backward_step_patches`], which handles Conv2d Patches dispatch
/// and delegates all other layers to this function.
///
/// The only variation across callers was whether a
/// GEMM engine is passed for `Layer::Linear`; this is captured by the
/// `engine` parameter.
///
/// # Returns
///
/// - `Ok(CrownStepResult::Continue)` — propagation succeeded, `linear_bounds`
///   updated in-place.
/// - `Ok(CrownStepResult::IbpFallback(_))` — caller should fall back to IBP.
/// - `Err(e)` — propagation error.
///
/// # Design reference
///
/// See `designs/2026-03-13-crown-backward-step-dispatch-bridge.md` for rationale.
/// Delegates per-layer shape setup and trait dispatch to `dispatch_backward_layer`
/// (backward_dispatch module), with pre-filters for ReLU and multi-input ops.
pub(super) fn crown_backward_step(
    layer: &Layer,
    linear_bounds: &mut LinearBounds,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    layer_idx: usize,
    label: &str,
    deadline: Option<Instant>,
) -> Result<CrownStepResult> {
    crown_backward_step_with_dispatch_boundary(
        layer,
        linear_bounds,
        pre_activation,
        engine,
        layer_idx,
        label,
        deadline,
        false,
    )
}

/// Dense CROWN step with explicit provenance for a just-materialized Patches
/// carrier.  Ordinary Dense callers use [`crown_backward_step`]; a hard finite
/// Patches boundary opts into the stricter canonical dispatcher so legacy
/// uncooperative Dense operators decline before touching the carrier.
#[allow(clippy::too_many_arguments)]
pub(super) fn crown_backward_step_with_dispatch_boundary(
    layer: &Layer,
    linear_bounds: &mut LinearBounds,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    layer_idx: usize,
    label: &str,
    deadline: Option<Instant>,
    finite_structured_boundary: bool,
) -> Result<CrownStepResult> {
    check_crown_backward_deadline(deadline, layer_idx, layer, "before dense dispatch")?;

    // An embedded-constant Where normally has a useful sequential shortcut,
    // but its condition/broadcast preparation is a legacy O(N) operation. A
    // Dense carrier that was just materialized from Patches under hard finite
    // authority must decline before that shortcut, matching the canonical
    // strict-boundary policy used by graph coordinators.
    if finite_structured_boundary && matches!(layer, Layer::Where(_)) {
        return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
            reason: CrownIbpFallbackReason::CrownPropagationError,
            details: format!(
                "finite layer {layer_idx} ({}) declines structured-boundary Where dispatch",
                layer.layer_type()
            ),
        }));
    }

    // === Pre-filter 1: ReLU ===
    // dispatch_backward_layer returns Unsupported for ReLU because graph-CROWN
    // needs alpha/beta state. Sequential CROWN uses the simpler
    // propagate_crown_backward trait path (no alpha optimization).
    if matches!(layer, Layer::ReLU(_)) {
        return crown_backward_relu(
            layer,
            linear_bounds,
            pre_activation,
            layer_idx,
            label,
            deadline,
        );
    }

    // === Pre-filter 1b: embedded-constant Where ===
    // In a sequential chain the only well-formed Where is the embedded-constant
    // form: Where(cond, const_true, const_false) where `cond` is the chain input
    // (= pre_activation). Both branches are constants, so the output is a constant
    // vector w.r.t. the chain input. `embedded_constant_select_output` returns the
    // EXACT per-element select when `cond` is bound-independent (lower == upper)
    // and the sound IBP union otherwise — strictly no looser than the generic IBP
    // fallback that follows. Concretize the accumulated CROWN bounds against that
    // (zero-A, constant-bias result), tightening UNSAT for cctsdb/nn4sys/vit
    // masks. A non-embedded sequential Where (3 named edges) cannot exist, so it
    // still drops to the multi-input IBP fallback below.
    if let Layer::Where(where_layer) = layer {
        if where_layer.has_embedded_constants() {
            match where_layer.embedded_constant_select_output(pre_activation) {
                Ok(select) => {
                    let concretized =
                        match linear_bounds.concretize_sound_with_deadline(&select, deadline) {
                            Ok(bounds) => bounds,
                            Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                                return Ok(memory_error_step_fallback(&error, layer_idx, layer));
                            }
                            Err(error) => return Err(error),
                        };
                    let staged = match constant_linear_from_concretized_with_deadline(
                        concretized,
                        pre_activation.len(),
                        deadline,
                        layer_idx,
                        layer,
                    ) {
                        Ok(bounds) => bounds,
                        Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                            return Ok(memory_error_step_fallback(&error, layer_idx, layer));
                        }
                        Err(error) => return Err(error),
                    };
                    *linear_bounds = staged;
                    return Ok(CrownStepResult::Continue);
                }
                // Constants failed to broadcast / produced NaN: keep the sound
                // generic IBP fallback below by falling through.
                Err(e) => {
                    debug!(
                        "{}: embedded-constant Where at layer {} fell back to IBP: {}",
                        label, layer_idx, e
                    );
                }
            }
        }
    }

    // === Pre-filter 2: Multi-input ops ===
    // dispatch_backward_layer handles binary/n-ary ops by reading ctx.inputs[0..N]
    // and resolving node_bounds. Sequential networks pass inputs=&[] (no named graph
    // edges), so binary dispatch would fail with InvalidSpec. Pre-filter these to
    // the same IbpFallback that the inline code used.
    if matches!(
        layer,
        Layer::MatMul(_)
            | Layer::MulBinary(_)
            | Layer::Add(_)
            | Layer::Concat(_)
            | Layer::Sub(_)
            | Layer::Div(_)
            | Layer::Atan2(_)
            | Layer::BilinearCrown(_)
            | Layer::MinBinary(_)
            | Layer::MaxBinary(_)
            | Layer::ExpandLikeLastAxis(_)
            | Layer::ScatterAdd(_)
            | Layer::IndexAdd(_)
            | Layer::ScatterNd(_)
            | Layer::Where(_)
            | Layer::NonZero(_)
            | Layer::SelfAttention(_)
    ) {
        debug!(
            "{}: layer {} ({}) not supported in sequential network, using IBP",
            label,
            layer_idx,
            layer.layer_type()
        );
        return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
            reason: CrownIbpFallbackReason::CrownPropagationError,
            details: format!(
                "layer {} ({}) not supported in sequential network",
                layer_idx,
                layer.layer_type()
            ),
        }));
    }

    // === Pre-filter 3: SkipMerge pass-through ===
    // SkipMerge is identity in sequential CROWN — bounds pass through unchanged.
    // dispatch_backward_layer requires ctx.inputs.len() == 1 (graph edge),
    // but sequential networks pass inputs=&[] (no named edges). Handle directly.
    if matches!(layer, Layer::SkipMerge(_)) {
        return Ok(CrownStepResult::Continue);
    }

    // Build minimal context for unary dispatch. After pre-filtering ReLU,
    // multi-input ops, and SkipMerge, only unary shape-setup and
    // trait-dispatch layers remain.
    let empty_bounds: HashMap<String, BoundedTensor> = HashMap::new();
    let ctx = DispatchContext {
        node_name: label,
        layer,
        inputs: &[],
        pre_activation,
        network_input: pre_activation,
        node_bounds: (&empty_bounds).into(),
        engine,
        deadline,
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas: None,
        norm_inv_rms_override: None,
    };

    // dispatch_backward_layer may raise UnsupportedOp (e.g., Conv1d with
    // shape < 2 dims), NumericalInstability, ShapeMismatch (#3813: Dense
    // Conv2d backward when graph restructuring changes dimensions), or
    // DeadlineExceeded is verifier authority and must remain structured. Other
    // unsupported/unstable errors are graceful per-layer IBP fallbacks.
    let dispatched = if finite_structured_boundary {
        dispatch_backward_layer_finite_boundary(&ctx, linear_bounds)
    } else {
        dispatch_backward_layer(&ctx, linear_bounds)
    };
    let dispatch_result = match dispatched {
        Ok(result) => result,
        Err(
            NyError::UnsupportedOp(ref msg)
            | NyError::UnsupportedConfiguration(ref msg)
            | NyError::NumericalInstability(ref msg),
        ) => {
            if finite_authority_refuses_per_layer_ibp(deadline) {
                return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                    reason: CrownIbpFallbackReason::CrownPropagationError,
                    details: format!(
                        "finite layer {layer_idx} ({}) typed-refused before opaque per-layer IBP: {msg}",
                        layer.layer_type()
                    ),
                }));
            }
            return crown_backward_ibp_concretize(
                layer,
                linear_bounds,
                pre_activation,
                layer_idx,
                label,
                msg,
                deadline,
            );
        }
        Err(NyError::ShapeMismatch {
            ref expected,
            ref got,
        }) => {
            let msg = format!("shape mismatch: expected {:?}, got {:?}", expected, got);
            if finite_authority_refuses_per_layer_ibp(deadline) {
                return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                    reason: CrownIbpFallbackReason::ShapeMismatch,
                    details: format!(
                        "finite layer {layer_idx} ({}) refused before opaque per-layer IBP: {msg}",
                        layer.layer_type()
                    ),
                }));
            }
            return crown_backward_ibp_concretize(
                layer,
                linear_bounds,
                pre_activation,
                layer_idx,
                label,
                &msg,
                deadline,
            );
        }
        Err(error @ NyError::DeadlineExceeded(_)) => return Err(error),
        Err(e) => return Err(e),
    };
    check_crown_backward_deadline(deadline, layer_idx, layer, "after dense dispatch")?;

    match dispatch_result {
        BackwardDispatchResult::Single(new_lb) => {
            *linear_bounds = *new_lb;
            Ok(CrownStepResult::Continue)
        }
        BackwardDispatchResult::PassThrough => Ok(CrownStepResult::Continue),
        BackwardDispatchResult::Unsupported(msg) => {
            // Per-layer IBP concretization: when dispatch returns Unsupported,
            // concretize accumulated CROWN bounds through IBP at this layer
            // instead of discarding ALL CROWN work with whole-network IBP fallback.
            //
            // PRESENCE here, unlike the two typed-error arms above, and the
            // difference is which dispatcher can produce this variant.
            // `dispatch_backward_layer_finite_boundary_inner` never raises a
            // typed Err — every family outside SkipMerge/ReLU/Where/Div gets an
            // `Ok(Unsupported)` here as its DELIBERATE finite refusal ("has no
            // fully cooperative finite-deadline dispatch route"). So this arm
            // carries both real layer failures and the finite-routes policy's
            // own decision, and cannot distinguish them from the message.
            // Deciding it by expiry would silently rewrite that policy into
            // per-layer concretization; `patches_step_tests::
            // finite_deadline_conv_uses_typed_fallback_and_preserves_expiry`
            // pins the refusal and fails when this flips. Moving it belongs to
            // the finite-dispatch program (docs/DEADLINE_PRESENCE_GUARD_
            // CENSUS_2026-08-17.md, class UNPOLLABLE-ENTANGLED), not here.
            if deadline.is_some() {
                return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                    reason: CrownIbpFallbackReason::CrownPropagationError,
                    details: format!(
                        "finite layer {layer_idx} ({}) typed-refused before opaque per-layer IBP: {msg}",
                        layer.layer_type()
                    ),
                }));
            }
            crown_backward_ibp_concretize(
                layer,
                linear_bounds,
                pre_activation,
                layer_idx,
                label,
                &msg,
                deadline,
            )
        }
        // Binary/Nary: unreachable after pre-filter, but handle defensively.
        BackwardDispatchResult::Binary { .. } | BackwardDispatchResult::Nary { .. } => {
            Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                reason: CrownIbpFallbackReason::CrownPropagationError,
                details: format!(
                    "unexpected multi-input result for {} in sequential CROWN",
                    layer.layer_type()
                ),
            }))
        }
    }
}

#[inline]
fn check_crown_backward_deadline(
    deadline: Option<Instant>,
    layer_idx: usize,
    layer: &Layer,
    phase: &str,
) -> Result<()> {
    if deadline.is_some_and(|value| Instant::now() >= value) {
        Err(NyError::DeadlineExceeded(format!(
            "sequential CROWN backward: deadline exceeded {phase} at layer {layer_idx} ({})",
            layer.layer_type()
        )))
    } else {
        Ok(())
    }
}

/// ReLU backward via trait dispatch (no alpha optimization in sequential CROWN).
///
/// dispatch_backward_layer returns Unsupported for ReLU because graph-CROWN
/// needs alpha/beta state that varies per call site. Sequential CROWN uses
/// the simpler propagate_crown_backward trait path.
fn crown_backward_relu(
    layer: &Layer,
    linear_bounds: &mut LinearBounds,
    pre_activation: &BoundedTensor,
    layer_idx: usize,
    label: &str,
    deadline: Option<Instant>,
) -> Result<CrownStepResult> {
    // Decline only when the deadline has ALREADY expired. This guard shipped as
    // `deadline.is_some()` (rode in unreviewed via 7b0140d09, a test-titled
    // commit inside a non-compiling stack), which declined tight dense-ReLU
    // CROWN on EVERY deadlined run — the vnncomp lane always sets one —
    // silently downgrading all sequential-engine bounds to the CROWN-IBP
    // fallback. Measured cost (relusplitter 220, receipts in reports/sweeps/):
    // 43/220 -> 20/220 with unsat 23 -> 0; the alpha phase went verified=9/9
    // -> verified=0/9 on rows this engine had proved for months
    // (docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md).
    //
    // Pollability argument: the step is bracketed by
    // `check_crown_backward_deadline` before dispatch (caller) and after
    // dispatch (below), so the only unpollable window is one elementwise
    // dense-ReLU backward over the current row block — microseconds on the
    // sequential engine's FC layers. Falling back on EXPIRY keeps the
    // finite-routes tail-latency protection; falling back on EXISTENCE
    // deletes the proof path.
    if deadline.is_some_and(|value| Instant::now() >= value) {
        return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
            reason: CrownIbpFallbackReason::CrownPropagationError,
            details: format!(
                "finite layer {layer_idx} ({}) declines dense ReLU propagation past the deadline",
                layer.layer_type()
            ),
        }));
    }
    match layer.propagate_crown_backward(linear_bounds, Some(pre_activation)) {
        Ok(mut next) => {
            check_crown_backward_deadline(deadline, layer_idx, layer, "after ReLU dispatch")?;
            // Eager per-row discharge of the carried coefficient error over the
            // pre-activation cut (#cgan-conv-err-compose, see
            // LinearBounds::fold_coeff_err_over_box_eager). Rows with a
            // non-finite penalty keep carrying (prior behavior).
            next.fold_coeff_err_over_box_eager(pre_activation);
            check_crown_backward_deadline(
                deadline,
                layer_idx,
                layer,
                "after ReLU coefficient-error discharge",
            )?;
            *linear_bounds = next;
            Ok(CrownStepResult::Continue)
        }
        Err(
            NyError::UnsupportedOp(ref msg)
            | NyError::UnsupportedConfiguration(ref msg)
            | NyError::NumericalInstability(ref msg),
        ) => crown_backward_ibp_concretize(
            layer,
            linear_bounds,
            pre_activation,
            layer_idx,
            label,
            msg,
            deadline,
        ),
        // #3813: ShapeMismatch triggers per-layer IBP concretization (same as
        // crown_backward_step). RSPLITTER models change intermediate dimensions,
        // causing shape mismatches in ReLU backward propagation. IBP fallback
        // is always sound.
        Err(NyError::ShapeMismatch {
            ref expected,
            ref got,
        }) => {
            let msg = format!(
                "ReLU shape mismatch: expected {:?}, got {:?}",
                expected, got
            );
            crown_backward_ibp_concretize(
                layer,
                linear_bounds,
                pre_activation,
                layer_idx,
                label,
                &msg,
                deadline,
            )
        }
        Err(e) => Err(e),
    }
}

/// Per-layer IBP concretization fallback.
///
/// When CROWN backward fails for a single layer, concretize accumulated CROWN
/// bounds through IBP at that layer instead of discarding ALL CROWN work with
/// whole-network IBP fallback.
/// Pattern: beta_crown/engine/backward/layer_dispatch.rs:beta_crown_ibp_fallback.
fn crown_backward_ibp_concretize(
    layer: &Layer,
    linear_bounds: &mut LinearBounds,
    pre_activation: &BoundedTensor,
    layer_idx: usize,
    label: &str,
    reason: &str,
    deadline: Option<Instant>,
) -> Result<CrownStepResult> {
    warn!(
        "{}: layer {} ({}) unsupported/unstable, \
         concretizing to per-layer IBP bounds: {}",
        label,
        layer_idx,
        layer.layer_type(),
        reason,
    );
    check_crown_backward_deadline(deadline, layer_idx, layer, "before per-layer IBP")?;
    let post_bounds = layer.propagate_ibp(pre_activation)?;
    check_crown_backward_deadline(deadline, layer_idx, layer, "after per-layer IBP")?;
    let concretized = match linear_bounds.concretize_sound_with_deadline(&post_bounds, deadline) {
        Ok(bounds) => bounds,
        Err(error @ NyError::CpuMemoryExceeded { .. }) => {
            return Ok(memory_error_step_fallback(&error, layer_idx, layer));
        }
        Err(error) => return Err(error),
    };
    let staged = match constant_linear_from_concretized_with_deadline(
        concretized,
        pre_activation.len(),
        deadline,
        layer_idx,
        layer,
    ) {
        Ok(bounds) => bounds,
        Err(error @ NyError::CpuMemoryExceeded { .. }) => {
            return Ok(memory_error_step_fallback(&error, layer_idx, layer));
        }
        Err(error) => return Err(error),
    };
    *linear_bounds = staged;
    Ok(CrownStepResult::Continue)
}

fn constant_linear_from_concretized_with_deadline(
    concretized: BoundedTensor,
    num_inputs: usize,
    deadline: Option<Instant>,
    layer_idx: usize,
    layer: &Layer,
) -> Result<LinearBounds> {
    check_crown_backward_deadline(deadline, layer_idx, layer, "before constant-bound staging")?;
    let num_outputs = concretized.len();
    let (lower, upper) = concretized.into_parts();
    let lower = lower.into_dimensionality::<Ix1>().map_err(|error| {
        NyError::InternalError(format!(
            "sequential CROWN concretized lower bound was not one-dimensional: {error}"
        ))
    })?;
    let upper = upper.into_dimensionality::<Ix1>().map_err(|error| {
        NyError::InternalError(format!(
            "sequential CROWN concretized upper bound was not one-dimensional: {error}"
        ))
    })?;
    let staged = crate::network::CrownMergeAccumulator::try_dense_bias_bounds_with_deadline(
        &lower,
        &upper,
        num_outputs,
        num_inputs,
        deadline,
    )?;
    check_crown_backward_deadline(deadline, layer_idx, layer, "after constant-bound staging")?;
    Ok(staged)
}

#[cfg(test)]
#[path = "backward_step_tests.rs"]
mod tests;
