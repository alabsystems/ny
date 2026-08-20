// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-aware CROWN backward step dispatch.
//!
//! Extracted from `crown.rs` as Packet 1 of the three-module extraction (#4005).
//! Contains [`crown_backward_step_patches`] and the shared
//! [`try_patches_or_dense_fallback`] helper that deduplicates the
//! BatchNorm/AveragePool/MaxPool2d dense-fallback arms.

use crate::bounds::patches::{CrownBounds, PatchesMaterializationPurpose};
use crate::layers::common::PatchesPropagation;
use crate::layers::Layer;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use std::time::Instant;
use tracing::{debug, info};

use super::{
    crown_backward_step, crown_backward_step_with_dispatch_boundary,
    guard_dense_materialization_budget, CrownStepFallback, CrownStepResult,
};
use crate::types::CrownIbpFallbackReason;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReluDeadlineRequest {
    Disabled,
    SpecCrown,
}

/// Private error surface for the spec-guided Patches dispatcher.
///
/// `ReluDeadlineExceeded` is emitted only by the gated, materialized
/// explicit-row ReLU leaf. All ordinary `NyError`s remain `Ordinary`; the
/// caller may still classify a layer-specific resource authority (notably the
/// pollable ConvTranspose anchored planner) before deciding whether a Dense
/// retry is legal.
pub(crate) enum SpecPatchesStepError {
    ReluDeadlineExceeded,
    Ordinary(NyError),
}

impl From<NyError> for SpecPatchesStepError {
    fn from(error: NyError) -> Self {
        Self::Ordinary(error)
    }
}

/// Default-dark cooperative deadline polling for the explicit-row ReLU
/// Patches leaf. Exact `1` is the only enabling spelling.
fn patches_deadline_relu_enabled() -> bool {
    matches!(
        std::env::var("NY_PATCHES_DEADLINE_RELU").as_deref(),
        Ok("1")
    )
}

#[inline]
fn check_patches_scheduling_deadline(deadline: Instant, phase: &'static str) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(format!(
            "Patches scheduling deadline exceeded {phase}"
        )));
    }
    Ok(())
}

#[inline]
fn is_materialized_explicit_row_candidate(
    bounds: &crate::bounds::patches::PatchesLinearBounds,
) -> bool {
    let side_is_candidate = |side: &crate::bounds::patches::PatchesData| {
        !side.identity
            && side.unstable_idx.is_none()
            && side
                .patches
                .as_ref()
                .is_some_and(|patches| patches.ndim() == 7)
    };
    side_is_candidate(&bounds.lower_a) && side_is_candidate(&bounds.upper_a)
}

/// Try patches-native backward, fall back to Dense + [`crown_backward_step`] on error.
///
/// Used by BatchNorm, AveragePool, MaxPool2d arms — each supports patches natively
/// but falls back to Dense when propagation fails (shape mismatch, numerical issue).
///
/// `patches_fn` abstracts the method dispatch since BatchNorm uses
/// `propagate_patches(pb)` while pool layers use
/// `propagate_patches_with_bounds(pb, pre_activation)`.
#[allow(clippy::too_many_arguments)] // CROWN backward context: each param is a distinct propagation concern
fn try_patches_or_dense_fallback(
    patches_fn: impl FnOnce(&mut crate::bounds::patches::PatchesLinearBounds) -> Result<CrownBounds>,
    crown_bounds: &mut CrownBounds,
    pre_activation: &BoundedTensor,
    layer: &Layer,
    engine: Option<&dyn GemmEngine>,
    layer_idx: usize,
    label: &str,
    fallback_site: &'static str,
    dispatch_site: &'static str,
    deadline: Option<Instant>,
    deadline_is_hard: bool,
    materialized_from_finite_patches: bool,
) -> Result<CrownStepResult> {
    // The legacy affine BN/pool Patches kernels do not carry the absolute
    // deadline through every allocation and scan. Anchored BatchNorm has its
    // dedicated cooperative branch before this helper. Under hard finite
    // authority route every remaining carrier through the cooperative
    // materializer and shared finite dispatcher instead of entering an opaque
    // native leaf. A collector-local timestamp only admits the legacy native
    // operation as one indivisible scheduling unit.
    if hard_finite_authority_refuses_patches(deadline_is_hard, deadline)
        && matches!(crown_bounds, CrownBounds::Patches(_))
    {
        if let Some(fallback) = guard_dense_materialization_budget(
            crown_bounds,
            fallback_site,
            layer,
            layer_idx,
            label,
        )? {
            return Ok(fallback);
        }
        let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
        return crown_backward_step_with_dispatch_boundary(
            layer,
            lb,
            pre_activation,
            engine,
            layer_idx,
            label,
            deadline,
            true,
        );
    }
    if let CrownBounds::Patches(pb) = crown_bounds {
        if !deadline_is_hard {
            if let Some(limit) = deadline {
                check_patches_scheduling_deadline(limit, "before native BN/pool dispatch")?;
            }
        }
        match patches_fn(pb) {
            Ok(result) => {
                *crown_bounds = result;
                return Ok(CrownStepResult::Continue);
            }
            Err(e) => {
                debug!(
                    "{}: {} Patches backward failed at layer {}: {}, \
                     falling back to Dense",
                    label,
                    layer.layer_type(),
                    layer_idx,
                    e
                );
                if let Some(fallback) = guard_dense_materialization_budget(
                    crown_bounds,
                    fallback_site,
                    layer,
                    layer_idx,
                    label,
                )? {
                    return Ok(fallback);
                }
                let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
                return crown_backward_step(
                    layer,
                    lb,
                    pre_activation,
                    engine,
                    layer_idx,
                    label,
                    deadline,
                );
            }
        }
    }
    // Dense mode: use standard Dense dispatch
    if let Some(fallback) =
        guard_dense_materialization_budget(crown_bounds, dispatch_site, layer, layer_idx, label)?
    {
        return Ok(fallback);
    }
    let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
    crown_backward_step_with_dispatch_boundary(
        layer,
        lb,
        pre_activation,
        engine,
        layer_idx,
        label,
        deadline,
        materialized_from_finite_patches,
    )
}

/// Patches-aware CROWN backward step for sequential networks.
///
/// Wraps [`CrownBounds`] to dispatch layers through the Patches path when
/// already in Patches mode, falling back to Dense for unsupported layers.
///
/// Patches-aware layers (Phase 1: Conv2d, Phase 2: ReLU/activations) process
/// Patches coefficients natively, preserving the sparse structure. All other
/// layers call `ensure_dense()` first, then delegate to [`crown_backward_step`].
///
/// Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md
/// Part of #2613
pub(crate) fn crown_backward_step_patches(
    layer: &Layer,
    crown_bounds: &mut CrownBounds,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    layer_idx: usize,
    label: &str,
    deadline: Option<Instant>,
) -> Result<CrownStepResult> {
    crown_backward_step_patches_with_deadline_authority(
        layer,
        crown_bounds,
        pre_activation,
        engine,
        layer_idx,
        label,
        deadline,
        deadline.is_some(),
    )
}

/// Patches-aware CROWN step with an explicit distinction between a caller's
/// hard deadline and an internal scheduling deadline.
///
/// Both deadline classes remain visible to cooperative Patches kernels. Only a
/// hard deadline requires pre-emptive routing away from a legacy native kernel
/// that is not yet cooperative across every allocation/scanning phase.
/// Whether hard finite authority must refuse the native Patches route and hand
/// the carrier to the cooperative Dense materializer instead.
///
/// SHIPPED: presence. `deadline_is_hard` alone refuses, exactly as before.
///
/// That default is worth stating plainly, because it is not obviously the right
/// one. Every scored run carries a finite deadline, so the refusal fires on
/// every conv row, and it is a DEAD END rather than a fallback: the Dense
/// carrier it produces goes to `dispatch_backward_layer_finite_boundary`, which
/// declines every layer family except SkipMerge/ReLU/Where/Div. The node ends
/// with reference bounds and no CROWN at all — neither structured nor dense.
/// That is how `cifar_bias_field_46` reports
/// `'/layers.3/Conv' ... has no fully cooperative finite-deadline dispatch route`
/// and then times out where `97fb4bd6a` proved it in 35.7 s.
///
/// `NY_PATCHES_FINITE_EXPIRY=1` decides the same refusal by EXPIRY instead, so a
/// live deadline keeps the native route. STALENESS NOTE (2026-08-19): the
/// lever now SHIPS ARMED (default true, `=0` kill switch; Provenance::Measured
/// 8c393486c) — the "dark because measured not to convert" record below is the
/// original biasfield null kept for history, not the current state: 3 sat / 17
/// timeout on the 20-row biasfield subset in both arms, identical row by row. Armed, the declines do disappear from the logs and
/// tight work runs — the row then exhausts its budget inside
/// `FaerCpuGemmEngine::gemm_f64_with_deadline` instead, which says the remaining
/// gap is GPU routing for that backward, not this gate.
///
/// Note what expiry does NOT give up. For the sequential step an expired
/// deadline is already refused at the step's own entry check, ahead of this
/// predicate and ahead of any dispatch, so the stop-work behaviour a hard
/// deadline is owed does not come from the presence test either way. What the
/// armed arm gives up is interruptibility WITHIN a step: these kernels poll
/// their dominant contraction but own unreceipted allocation and scanning
/// phases, so an armed run can overrun by a bounded single layer step.
#[inline]
fn hard_finite_authority_refuses_patches(
    deadline_is_hard: bool,
    deadline: Option<Instant>,
) -> bool {
    if !deadline_is_hard {
        return false;
    }
    let refuses = if expiry_authority_armed() {
        deadline.is_some_and(|limit| Instant::now() >= limit)
    } else {
        true
    };
    // [deadline-preserve] BUG #18 (docs/DEADLINE_PRESENCE_FIX_2026-08-19.md):
    // saved-vs-discarded engagement counts for the sequential-lane mate of the
    // patches/alpha set. Recorded after the decision; routing unchanged, no
    // bound value touched — observation only. Rate-limited (power-of-two).
    static SEQ_PRESERVE: crate::network::core::graph::backward_helpers::DeadlinePreserveCounters =
        crate::network::core::graph::backward_helpers::DeadlinePreserveCounters::new(
            "patches-alpha-seq",
        );
    SEQ_PRESERVE.record(refuses);
    refuses
}

/// Latched once per process.
///
/// This predicate runs on EVERY patches layer step of every backward walk, so a
/// live `env::var_os` here is not free — it is a lock plus a scan of the whole
/// environment block, paid per layer per target per iteration. The repo's
/// existing hot-path lever idiom is exactly this latch (see the telemetry and
/// margin-row readers), and the value cannot change mid-run anyway.
pub(crate) fn expiry_authority_armed() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| {
        ny_levers::read(&ny_levers::decls::diagnostics::PATCHES_FINITE_EXPIRY)
            .value
            .as_bool()
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn crown_backward_step_patches_with_deadline_authority(
    layer: &Layer,
    crown_bounds: &mut CrownBounds,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    layer_idx: usize,
    label: &str,
    deadline: Option<Instant>,
    deadline_is_hard: bool,
) -> Result<CrownStepResult> {
    match crown_backward_step_patches_impl(
        layer,
        crown_bounds,
        pre_activation,
        engine,
        layer_idx,
        label,
        deadline,
        deadline_is_hard,
        ReluDeadlineRequest::Disabled,
    ) {
        Ok(result) => Ok(result),
        Err(SpecPatchesStepError::Ordinary(error @ NyError::CpuMemoryExceeded { .. })) => {
            Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                reason: CrownIbpFallbackReason::MemoryBudgetExceeded,
                details: format!(
                    "layer {layer_idx} ({}) Patches resource refusal: {error}",
                    layer.layer_type()
                ),
            }))
        }
        Err(SpecPatchesStepError::Ordinary(error)) => Err(error),
        Err(SpecPatchesStepError::ReluDeadlineExceeded) => {
            unreachable!("ordinary Patches dispatch cannot request the Spec-only ReLU deadline")
        }
    }
}

/// Patches-aware CROWN backward step for the spec-guided dispatcher.
///
/// This is the only seam that requests cooperative finite-deadline polling for
/// eligible explicit-row ReLU work.
pub(crate) fn crown_backward_step_patches_spec_crown(
    layer: &Layer,
    crown_bounds: &mut CrownBounds,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    layer_idx: usize,
    label: &str,
    deadline: Option<Instant>,
) -> std::result::Result<CrownStepResult, SpecPatchesStepError> {
    crown_backward_step_patches_impl(
        layer,
        crown_bounds,
        pre_activation,
        engine,
        layer_idx,
        label,
        deadline,
        deadline.is_some(),
        ReluDeadlineRequest::SpecCrown,
    )
}

#[allow(clippy::too_many_arguments)] // Private implementation also carries the typed ReLU deadline scope.
fn crown_backward_step_patches_impl(
    layer: &Layer,
    crown_bounds: &mut CrownBounds,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    layer_idx: usize,
    label: &str,
    deadline: Option<Instant>,
    deadline_is_hard: bool,
    relu_deadline_request: ReluDeadlineRequest,
) -> std::result::Result<CrownStepResult, SpecPatchesStepError> {
    if let Some(limit) = deadline {
        if Instant::now() >= limit {
            if relu_deadline_request == ReluDeadlineRequest::SpecCrown
                && matches!(layer, Layer::ReLU(_))
            {
                return Err(SpecPatchesStepError::ReluDeadlineExceeded);
            }
            return Err(NyError::DeadlineExceeded(format!(
                "{label}: Patches deadline exceeded before layer {layer_idx} ({}) dispatch",
                layer.layer_type()
            ))
            .into());
        }
    }
    let mut materialized_from_finite_patches = false;

    // SOUNDNESS SAFETY NET (#vnncomp-aw-soundness). The incoming Dense bounds may
    // carry a certified coefficient-error interval (from a linear/conv backward
    // composed earlier in the chain). Layers taught to propagate that error keep
    // it sound:
    //   - `propagates_coeff_err()` (Linear, Conv2d, element-wise activations,
    //     pass-through reshapes) — handled inside their own backward;
    //   - `is_exact_linear_coeff_err_carrier()` (Slice/Transpose/Tile/Gather/Pad/
    //     constant-arithmetic/Conv1d/ConvTranspose…) — carried by the EXACT-linear
    //     transform path in `dispatch_backward_layer`;
    //   - `SkipMerge` — a pure identity pass-through (`Continue`) that leaves the
    //     error matrices untouched.
    // For any OTHER layer, soundly discharge the error by degrading the affected
    // rows to conservative `[-inf, +inf]` before dispatch, so the layer proceeds
    // on plain (error-free) bounds and never silently drops the penalty.
    let carries_err = layer.propagates_coeff_err()
        || layer.is_exact_linear_coeff_err_carrier()
        || matches!(layer, Layer::SkipMerge(_));
    if !carries_err {
        if let CrownBounds::Dense(lb) = crown_bounds {
            if lb.has_coeff_err() {
                debug!(
                    "{}: layer {} ({}) does not propagate certified coefficient error; \
                     discharging affected rows to conservative bounds (#vnncomp-aw-soundness)",
                    label,
                    layer_idx,
                    layer.layer_type()
                );
                lb.discharge_coeff_err_to_conservative();
            }
        }
    }
    // Sparse patches fallback (#2613 Phase 4): structural layers (Conv2d,
    // BatchNorm, pooling, Linear, Flatten, Reshape) don't support sparse
    // patches natively yet. Convert sparse patches to dense before dispatch.
    // Element-wise activations handle sparse patches natively via
    // backward_patches_sparse in crown_patches.rs.
    let is_structural = matches!(
        layer,
        Layer::Conv2d(_)
            | Layer::ConvTranspose2d(_)
            | Layer::BatchNorm(_)
            | Layer::AveragePool(_)
            | Layer::MaxPool2d(_)
            | Layer::Linear(_)
            | Layer::Flatten(_)
            | Layer::Reshape(_)
    );
    let materialization_purpose = if matches!(layer, Layer::Reshape(_)) {
        PatchesMaterializationPurpose::LatentInputCrossover
    } else {
        PatchesMaterializationPurpose::Other
    };
    let sparse_structural_to_dense = is_structural
        && matches!(
            crown_bounds,
            CrownBounds::Patches(pb)
                if pb.lower_a.unstable_idx.is_some() || pb.upper_a.unstable_idx.is_some()
        );
    if sparse_structural_to_dense {
        if let Some(fallback) = guard_dense_materialization_budget(
            crown_bounds,
            "sparse_structural_to_dense",
            layer,
            layer_idx,
            label,
        )? {
            return Ok(fallback);
        }
        if let CrownBounds::Patches(pb) = crown_bounds {
            debug!(
                "{}: layer {} ({}) converting sparse patches to dense",
                label,
                layer_idx,
                layer.layer_type()
            );
            // This pre-dispatch conversion is part of the same node
            // transaction as the layer below.  Reusing the exact absolute
            // deadline makes an expiry terminal and leaves `crown_bounds`
            // unchanged; the pollable materializer publishes no partial pair.
            let dense = match pb
                .to_dense_with_deadline_for_purpose(deadline, materialization_purpose)
            {
                Ok(bounds) => bounds,
                Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                    return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                        reason: CrownIbpFallbackReason::MemoryBudgetExceeded,
                        details: format!(
                            "layer {layer_idx} ({}) sparse Patches materialization exceeded the CPU budget: {error}",
                            layer.layer_type()
                        ),
                    }));
                }
                Err(error) => return Err(error.into()),
            };
            materialized_from_finite_patches =
                hard_finite_authority_refuses_patches(deadline_is_hard, deadline);
            *crown_bounds = CrownBounds::Dense(dense);
        }
    }

    match layer {
        Layer::Conv2d(c) => {
            let input_shape = pre_activation.shape();
            let (in_h, in_w) = if input_shape.len() >= 3 {
                (
                    input_shape[input_shape.len() - 2],
                    input_shape[input_shape.len() - 1],
                )
            } else {
                debug!(
                    "{}: Conv2d input shape too small: {:?}, using IBP",
                    label, input_shape
                );
                return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                    reason: CrownIbpFallbackReason::CrownPropagationError,
                    details: format!("Conv2d input shape too small: {input_shape:?}"),
                }));
            };

            // Conv2d's legacy native Patches path polls its contraction but
            // still owns unreceipted allocation/scanning phases. Keep hard finite
            // authority on the cooperative Dense materializer + shared typed
            // dispatcher until that entire kernel has a hard authority seam.
            if hard_finite_authority_refuses_patches(deadline_is_hard, deadline)
                && matches!(crown_bounds, CrownBounds::Patches(_))
            {
                if let Some(fallback) = guard_dense_materialization_budget(
                    crown_bounds,
                    "conv2d_finite_dense_boundary",
                    layer,
                    layer_idx,
                    label,
                )? {
                    return Ok(fallback);
                }
                let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
                return crown_backward_step_with_dispatch_boundary(
                    layer,
                    lb,
                    pre_activation,
                    engine,
                    layer_idx,
                    label,
                    deadline,
                    true,
                )
                .map_err(Into::into);
            }
            // #3813 / #hotpath: Pre-check — if patches backward would produce a
            // composed kernel whose receptive-field AREA reaches the dense
            // A-matrix element count (the true memory crossover), convert to Dense
            // early. This avoids the expensive O(positions × channels × kernel²)
            // patches composition only when patches would no longer save memory
            // and `should_fallback_to_dense` would convert anyway. Patches stay
            // active for all earlier conv layers where new_kh*new_kw < in_h*in_w,
            // even past the old fixed 75%-per-dimension threshold.
            let bail_patches_to_dense = if let CrownBounds::Patches(pb) = &*crown_bounds {
                let (kh, kw) = c.kernel_size();
                pb.lower_a.would_conv_compose_cover_input(c.stride, (kh, kw), in_h, in_w)
                    || pb.upper_a.would_conv_compose_cover_input(c.stride, (kh, kw), in_h, in_w)
            } else {
                false
            };
            if bail_patches_to_dense {
                info!(
                    "{}: Conv2d Patches->Dense early bail at layer {}: \
                     composed kernel would cover {}x{} input (#3813)",
                    label, layer_idx, in_h, in_w
                );
                if let Some(fallback) = guard_dense_materialization_budget(
                    crown_bounds,
                    "conv2d_patches_early_bail",
                    layer,
                    layer_idx,
                    label,
                )? {
                    return Ok(fallback);
                }
                crown_bounds.ensure_dense_with_deadline(deadline)?;
            }

            match crown_bounds {
                CrownBounds::Patches(pb) => {
                    // Patches Conv2d backward. On UnsupportedConfiguration (dilation,
                    // or the soundness guard against composing through padded incoming
                    // patches — #hotpath), fall back to the exact dense CROWN path
                    // rather than propagating the error.
                    match c.propagate_patches_engine_and_deadline_for_input_shape(
                        pb,
                        engine,
                        deadline,
                        (in_h, in_w),
                    ) {
                        Ok(result) => *crown_bounds = result,
                        Err(NyError::UnsupportedConfiguration(msg)) => {
                            // EXPIRY, not hard-authority presence (sixth instance of
                            // the 7b0140d09 pattern, hidden behind the DERIVED
                            // `deadline_is_hard` bool and therefore invisible to both
                            // the guard census and the mechanical `is_some()` sweeps;
                            // receipts in REGRESSION_FC_UNSAT_LOST_2026-08-14.md).
                            // Aborting here refused the Dense retry BELOW — which is
                            // already deadline-plumbed end to end
                            // (`guard_dense_materialization_budget` +
                            // `ensure_dense_with_deadline` + the polled dense conv
                            // backward) — whenever the Patches kernel typed-refused,
                            // e.g. on every host whose wgpu proof adapter is
                            // unqualified. Measured on `cifar_bias_field_46`: every
                            // multi-conv walk discarded, first-conv widths 13x, alpha
                            // iter-0 -341M vs -168, 9 unsat proofs -> timeouts. An
                            // EXPIRED deadline still refuses instantly.
                            if deadline_is_hard
                                && deadline.is_some_and(|value| Instant::now() >= value)
                            {
                                return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                                    reason: CrownIbpFallbackReason::CrownPropagationError,
                                    details: format!(
                                        "finite layer {layer_idx} ({}) typed-refused Patches at the expired deadline: {msg}",
                                        layer.layer_type()
                                    ),
                                }));
                            }
                            debug!(
                                "{}: Conv2d Patches->Dense fallback at layer {}: {}",
                                label, layer_idx, msg
                            );
                            if let Some(fallback) = guard_dense_materialization_budget(
                                crown_bounds,
                                "conv2d_patches_unsupported_fallback",
                                layer,
                                layer_idx,
                                label,
                            )? {
                                return Ok(fallback);
                            }
                            let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
                            let next = c
                                .propagate_linear_with_engine_and_deadline_for_input_shape(
                                    lb,
                                    engine,
                                    deadline,
                                    (in_h, in_w),
                                )?;
                            if let Cow::Owned(next) = next {
                                *lb = next;
                            }
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
                CrownBounds::Dense(lb) => {
                    if materialized_from_finite_patches {
                        return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                            reason: CrownIbpFallbackReason::CrownPropagationError,
                            details: format!(
                                "finite layer {layer_idx} ({}) declines partially cooperative dense Conv2d kernel",
                                layer.layer_type()
                            ),
                        }));
                    }
                    // Thread deadline to enable intra-GEMM deadline checking (#3795).
                    let next = c.propagate_linear_with_engine_and_deadline_for_input_shape(
                        lb,
                        engine,
                        deadline,
                        (in_h, in_w),
                    )?;
                    if let Cow::Owned(next) = next {
                        *lb = next;
                    }
                }
            }
            Ok(CrownStepResult::Continue)
        }
        // Patches-native ConvTranspose2d CROWN backward (LEVER 2). The
        // historical no-deadline stride-1 route delegates to the proven
        // equivalent-Conv2d path. Finite stride 1 and every stride>1 route
        // admit authenticated full identities and materialized 6D/7D Affine
        // or Anchored composition below the stored-area crossover. Sparse,
        // mixed, and oversized relations return a typed refusal and take the
        // established fallback.
        // ConvTranspose2d is
        // `propagates_coeff_err = true`: every materialized compose carries its
        // certificate; an identity has `None` only when its copied coefficients
        // are DAZ-stable, otherwise zero centers carry a per-row flush certificate.
        Layer::ConvTranspose2d(ct) => {
            let input_shape = pre_activation.shape();
            let (in_h, in_w) = if input_shape.len() >= 3 {
                (
                    input_shape[input_shape.len() - 2],
                    input_shape[input_shape.len() - 1],
                )
            } else {
                debug!(
                    "{}: ConvTranspose2d input shape too small: {:?}, using IBP",
                    label, input_shape
                );
                return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                    reason: CrownIbpFallbackReason::CrownPropagationError,
                    details: format!("ConvTranspose2d input shape too small: {input_shape:?}"),
                }));
            };
            match crown_bounds {
                CrownBounds::Patches(pb) => {
                    match ct.propagate_patches_engine_and_deadline_for_input_shape(
                        pb,
                        engine,
                        deadline,
                        (in_h, in_w),
                    ) {
                        Ok(result) => *crown_bounds = result,
                        Err(NyError::UnsupportedConfiguration(msg)) => {
                            // EXPIRY, not hard-authority presence (sixth instance of
                            // the 7b0140d09 pattern, hidden behind the DERIVED
                            // `deadline_is_hard` bool and therefore invisible to both
                            // the guard census and the mechanical `is_some()` sweeps;
                            // receipts in REGRESSION_FC_UNSAT_LOST_2026-08-14.md).
                            // Aborting here refused the Dense retry BELOW — which is
                            // already deadline-plumbed end to end
                            // (`guard_dense_materialization_budget` +
                            // `ensure_dense_with_deadline` + the polled dense conv
                            // backward) — whenever the Patches kernel typed-refused,
                            // e.g. on every host whose wgpu proof adapter is
                            // unqualified. Measured on `cifar_bias_field_46`: every
                            // multi-conv walk discarded, first-conv widths 13x, alpha
                            // iter-0 -341M vs -168, 9 unsat proofs -> timeouts. An
                            // EXPIRED deadline still refuses instantly.
                            if deadline_is_hard
                                && deadline.is_some_and(|value| Instant::now() >= value)
                            {
                                return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                                    reason: CrownIbpFallbackReason::CrownPropagationError,
                                    details: format!(
                                        "finite layer {layer_idx} ({}) typed-refused Patches at the expired deadline: {msg}",
                                        layer.layer_type()
                                    ),
                                }));
                            }
                            debug!(
                                "{}: ConvTranspose2d Patches->Dense fallback at layer {}: {}",
                                label, layer_idx, msg
                            );
                            if let Some(fallback) = guard_dense_materialization_budget(
                                crown_bounds,
                                "convtranspose2d_patches_unsupported_fallback",
                                layer,
                                layer_idx,
                                label,
                            )? {
                                return Ok(fallback);
                            }
                            let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
                            let next = ct
                                .propagate_linear_with_engine_and_deadline_for_input_shape(
                                    lb,
                                    engine,
                                    deadline,
                                    (in_h, in_w),
                                )?;
                            if let Cow::Owned(next) = next {
                                *lb = next;
                            }
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
                CrownBounds::Dense(lb) => {
                    if materialized_from_finite_patches {
                        return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                            reason: CrownIbpFallbackReason::CrownPropagationError,
                            details: format!(
                                "finite layer {layer_idx} ({}) declines partially cooperative dense ConvTranspose2d kernel",
                                layer.layer_type()
                            ),
                        }));
                    }
                    let next = ct.propagate_linear_with_engine_and_deadline_for_input_shape(
                        lb,
                        engine,
                        deadline,
                        (in_h, in_w),
                    )?;
                    if let Cow::Owned(next) = next {
                        *lb = next;
                    }
                }
            }
            Ok(CrownStepResult::Continue)
        }
        // Phase 2: BatchNorm per-channel scaling natively on Patches.
        // BatchNorm is linear (y = scale*x + bias), so no pre-activation bounds
        // needed for the relaxation (there is no relaxation — exact substitution).
        // Part of #2613
        Layer::BatchNorm(bn) => {
            // Anchored carriers can be enormous, and their native BN path owns
            // a cooperative, operation-local allocation receipt. Thread the
            // exact node deadline directly and never reinterpret a resource
            // refusal as permission to materialize Dense. Numerical/structural
            // refusals retain the historical dense-fallback classification.
            if let CrownBounds::Patches(pb) = crown_bounds {
                let anchored = matches!(
                    &pb.lower_a.geometry,
                    crate::bounds::patches::PatchGeometry::Anchored(_)
                ) && matches!(
                    &pb.upper_a.geometry,
                    crate::bounds::patches::PatchGeometry::Anchored(_)
                );
                if anchored {
                    let native_result = if let Some(limit) = deadline {
                        bn.propagate_patches_with_deadline(pb, limit)
                    } else {
                        bn.propagate_patches(pb)
                    };
                    match native_result {
                        Ok(result) => {
                            *crown_bounds = result;
                            return Ok(CrownStepResult::Continue);
                        }
                        Err(NyError::DeadlineExceeded(_))
                            if relu_deadline_request == ReluDeadlineRequest::SpecCrown =>
                        {
                            // Legacy variant name; the spec caller treats it
                            // as node-local cooperative timeout authority.
                            return Err(SpecPatchesStepError::ReluDeadlineExceeded);
                        }
                        Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                            // The native total-live receipt is authoritative.
                            // In particular, no-deadline callers must not turn
                            // its refusal into permission to allocate Dense.
                            return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                                reason: CrownIbpFallbackReason::MemoryBudgetExceeded,
                                details: format!(
                                    "layer {layer_idx} ({}) Patches resource refusal: {error}",
                                    layer.layer_type()
                                ),
                            }));
                        }
                        Err(error @ NyError::DeadlineExceeded(_)) => {
                            return Err(error.into());
                        }
                        Err(
                            error @ (NyError::NumericalInstability(_)
                            | NyError::UnsupportedConfiguration(_)),
                        ) => {
                            debug!(
                                "{}: {} Patches backward failed at layer {}: {}, \
                                 falling back to Dense",
                                label,
                                layer.layer_type(),
                                layer_idx,
                                error
                            );
                            if let Some(fallback) = guard_dense_materialization_budget(
                                crown_bounds,
                                "batchnorm_dense_fallback",
                                layer,
                                layer_idx,
                                label,
                            )? {
                                return Ok(fallback);
                            }
                            let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
                            return crown_backward_step_with_dispatch_boundary(
                                layer,
                                lb,
                                pre_activation,
                                engine,
                                layer_idx,
                                label,
                                deadline,
                                deadline_is_hard,
                            )
                            .map_err(Into::into);
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            try_patches_or_dense_fallback(
                |pb| bn.propagate_patches(pb),
                crown_bounds,
                pre_activation,
                layer,
                engine,
                layer_idx,
                label,
                "batchnorm_dense_fallback",
                "batchnorm_dense_dispatch",
                deadline,
                deadline_is_hard,
                materialized_from_finite_patches,
            )
            .map_err(Into::into)
        }
        // Phase 3: Pooling layers natively support Patches backward.
        // AvgPool upsamples patches by pool kernel and divides by pool_size.
        // MaxPool applies relaxation slopes (argmax-lower / definite-winner).
        // Reference: designs/2026-03-01-patches-phase3-pooling-termination.md
        // Part of #2613
        Layer::AveragePool(ap) => try_patches_or_dense_fallback(
            |pb| ap.propagate_patches_with_bounds(pb, pre_activation),
            crown_bounds,
            pre_activation,
            layer,
            engine,
            layer_idx,
            label,
            "averagepool_dense_fallback",
            "averagepool_dense_dispatch",
            deadline,
            deadline_is_hard,
            materialized_from_finite_patches,
        )
        .map_err(Into::into),
        Layer::MaxPool2d(mp) => try_patches_or_dense_fallback(
            |pb| mp.propagate_patches_with_bounds(pb, pre_activation),
            crown_bounds,
            pre_activation,
            layer,
            engine,
            layer_idx,
            label,
            "maxpool_dense_fallback",
            "maxpool_dense_dispatch",
            deadline,
            deadline_is_hard,
            materialized_from_finite_patches,
        )
        .map_err(Into::into),
        // Phase 3: Linear/Flatten/Reshape terminate Patches mode — convert to Dense.
        // These layers flatten spatial structure, so Patches representation has no benefit.
        // Making this explicit rather than relying on the catch-all improves clarity.
        // Part of #2613 Phase 3.
        Layer::Linear(_) | Layer::Flatten(_) | Layer::Reshape(_) | Layer::Resize(_) => {
            if matches!(crown_bounds, CrownBounds::Patches(_)) {
                debug!(
                    "{}: Patches→Dense termination at layer {} ({})",
                    label,
                    layer_idx,
                    match layer {
                        Layer::Linear(_) => "Linear",
                        Layer::Flatten(_) => "Flatten",
                        Layer::Reshape(_) => "Reshape",
                        Layer::Resize(_) => "Resize",
                        _ => unreachable!(),
                    }
                );
            }
            if let Some(fallback) = guard_dense_materialization_budget(
                crown_bounds,
                "linear_flatten_reshape_resize_termination",
                layer,
                layer_idx,
                label,
            )? {
                return Ok(fallback);
            }
            let finite_structured_boundary = materialized_from_finite_patches
                || (hard_finite_authority_refuses_patches(deadline_is_hard, deadline)
                    && matches!(crown_bounds, CrownBounds::Patches(_)));
            let lb = crown_bounds
                .ensure_dense_with_deadline_for_purpose(deadline, materialization_purpose)?;
            crown_backward_step_with_dispatch_boundary(
                layer,
                lb,
                pre_activation,
                engine,
                layer_idx,
                label,
                deadline,
                finite_structured_boundary,
            )
            .map_err(Into::into)
        }
        // All remaining layers: try Patches activation dispatch, then Dense fallback.
        // Exhaustive listing — adding a new Layer variant is a compile error (#3424).
        //
        // Elementwise activations:
        Layer::ReLU(_) | Layer::GELU(_) | Layer::SiLU(_) | Layer::Tanh(_)
        | Layer::Sigmoid(_) | Layer::Erf(_) | Layer::Exp(_) | Layer::Log(_) | Layer::Sqrt(_)
        | Layer::Reciprocal(_) | Layer::Softplus(_) | Layer::HardSwish(_)
        | Layer::Mish(_) | Layer::Selu(_) | Layer::Softsign(_) | Layer::Arctan(_)
        | Layer::Tan(_) | Layer::Sin(_) | Layer::Cos(_) | Layer::Elu(_)
        | Layer::Celu(_) | Layer::LeakyReLU(_) | Layer::HardSigmoid(_)
        | Layer::Clip(_) | Layer::ThresholdedRelu(_) | Layer::Abs(_)
        | Layer::PowConstant(_) | Layer::Floor(_) | Layer::Ceil(_)
        | Layer::Round(_) | Layer::Trunc(_) | Layer::Sign(_) | Layer::PRelu(_) | Layer::Shrink(_)
        | Layer::Snake(_) | Layer::Compare(_)
        // Softmax family:
        | Layer::Softmax(_) | Layer::CausalSoftmax(_) | Layer::LogSoftmax(_)
        | Layer::LogSumExp(_)
        // Normalization:
        | Layer::LayerNorm(_) | Layer::RmsNorm(_) | Layer::InstanceNorm1d(_)
        | Layer::GroupNorm(_) | Layer::AdaIN1d(_)
        // Convolutions (non-2D):
        | Layer::Conv1d(_) | Layer::ConvTranspose1d(_)
        // Constant arithmetic:
        | Layer::AddConstant(_) | Layer::MulConstant(_) | Layer::DivConstant(_)
        | Layer::SubConstant(_)
        // Reductions:
        | Layer::ReduceMean(_) | Layer::ReduceSum(_) | Layer::CumSum(_)
        | Layer::ReduceMax(_) | Layer::ReduceMin(_)
        | Layer::Topk(_) | Layer::ArgMax(_) | Layer::ArgMin(_) | Layer::ArgSort(_)
        // Shape transforms:
        | Layer::Transpose(_) | Layer::Tile(_) | Layer::Gather(_)
        | Layer::ScatterAdd(_) | Layer::IndexAdd(_) | Layer::ScatterNd(_)
        | Layer::Slice(_) | Layer::Pad(_)
        | Layer::Squeeze(_) | Layer::Unsqueeze(_)
        | Layer::QdqPerturbation(_)
        // Positional encoding:
        | Layer::RoPE(_)
        // Binary / multi-input ops:
        | Layer::MatMul(_) | Layer::MulBinary(_) | Layer::Add(_) | Layer::Concat(_)
        | Layer::Sub(_) | Layer::Div(_) | Layer::Atan2(_) | Layer::BilinearCrown(_)
        | Layer::MinBinary(_)
        | Layer::MaxBinary(_) | Layer::ExpandLikeLastAxis(_)
        | Layer::CompareTensor(_)
        // Special:
        | Layer::SelfAttention(_) | Layer::Where(_) | Layer::NonZero(_)
        | Layer::SkipMerge(_) | Layer::OpaqueSkip(_) => {
            // Phase 2: Element-wise activations natively support Patches backward.
            // When in Patches mode, use propagate_patches_with_bounds to keep sparse
            // structure. When in Dense mode or for unsupported layers, fall through to
            // standard Dense dispatch. Generated via for_each_patches_capable_activation!
            // Part of #2613 Phase 2 step 11.
            if let CrownBounds::Patches(pb) = crown_bounds {
                // #patches-zero-pad-identity: a Pad whose every (before, after) is
                // (0, 0) adds no elements, so its output tensor IS its input tensor
                // — same shape, same values, for every `PadMode` (there is nothing
                // to fill). The CROWN backward relation with respect to the output
                // is therefore already the relation with respect to the input, and
                // the correct backward step is to do NOTHING. This is an identity,
                // not a relaxation: it cannot widen or narrow any bound.
                //
                // Without this, `Pad` falls through to `generic_dense_dispatch`
                // below, which calls `ensure_dense()` and DESTROYS the patches
                // representation for the whole remaining prefix. Measured on
                // TinyYOLO / yolo_2023 (2026-07-29): `Pad_10` and `Pad_17` are both
                // `pads=[0,0,0,0,0,0,0,0]` — pure no-ops — and each demanded a
                // 3_743_547_392-byte dense materialization for a 10816x43264
                // coefficient pair against a 2 GiB budget. The guard refused, the
                // CROWN backward for `Conv_12` and `Add_15` returned the
                // conservative relation, and both targets silently fell back to
                // IBP width while still being counted as CROWN successes.
                if let Layer::Pad(pad) = layer {
                    if pad
                        .pads
                        .iter()
                        .all(|&(before, after)| before == 0 && after == 0)
                    {
                        debug!(
                            "{}: #patches-zero-pad-identity — Pad at layer {} has all-zero \
                             pads; passing the patches relation through unchanged instead of \
                             materializing dense",
                            label, layer_idx
                        );
                        return Ok(CrownStepResult::Continue);
                    }
                }
                // Within this elementwise branch, a present deadline is
                // threaded to ReLU for Anchored carriers and for the existing
                // spec explicit-row gate. BatchNorm threads the same authority
                // in its dedicated branch above. Other activations and
                // no-deadline calls retain the historical trait dispatch.
                let anchored_relu_candidate = matches!(
                    &pb.lower_a.geometry,
                    crate::bounds::patches::PatchGeometry::Anchored(_)
                ) && matches!(
                    &pb.upper_a.geometry,
                    crate::bounds::patches::PatchGeometry::Anchored(_)
                );
                if anchored_relu_candidate
                    || (relu_deadline_request == ReluDeadlineRequest::SpecCrown
                        && patches_deadline_relu_enabled()
                        && is_materialized_explicit_row_candidate(pb))
                {
                    if let (Some(limit), Layer::ReLU(relu)) = (deadline, layer) {
                        match relu.propagate_patches_with_bounds_and_deadline(
                            pb,
                            pre_activation,
                            limit,
                        ) {
                            Ok(result) => {
                                *crown_bounds = result;
                                return Ok(CrownStepResult::Continue);
                            }
                            Err(error @ NyError::DeadlineExceeded(_)) => {
                                return if relu_deadline_request == ReluDeadlineRequest::SpecCrown {
                                    Err(SpecPatchesStepError::ReluDeadlineExceeded)
                                } else {
                                    Err(error.into())
                                };
                            }
                            Err(
                                NyError::NumericalInstability(msg)
                                | NyError::UnsupportedConfiguration(msg),
                            ) => {
                                debug!(
                                    "{}: ReLU Patches backward NumericalInstability \
                                     at layer {}: {}, falling back to Dense",
                                    label, layer_idx, msg
                                );
                                if let Some(fallback) = guard_dense_materialization_budget(
                                    crown_bounds,
                                    "activation_numerical_dense_fallback",
                                    layer,
                                    layer_idx,
                                    label,
                                )? {
                                    return Ok(fallback);
                                }
                                let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
                                return crown_backward_step_with_dispatch_boundary(
                                    layer,
                                    lb,
                                    pre_activation,
                                    engine,
                                    layer_idx,
                                    label,
                                    deadline,
                                    deadline_is_hard,
                                )
                                .map_err(Into::into);
                            }
                            Err(error) => return Err(error.into()),
                        }
                    }
                }

                // The graph collector also supplies an aggregate Patches
                // scheduling timestamp when there is no caller/cap deadline.
                // That timestamp is soft authority: it must bound the node,
                // but it must not force an otherwise exact affine ReLU Patches
                // walk through the O(rows * columns) Dense boundary. Run the
                // historical native ReLU transaction after one admission
                // check. The unpollable operation is one scheduling unit: once
                // admitted, its complete result publishes even if the soft
                // timestamp passes during work. Anchored carriers were handled
                // by the fully cooperative branch above.
                // THIS IS THE SITE THAT DENSIFIES THE WALK.
                //
                // Traced on `cifar_bias_field_46` with the expiry lever armed:
                // every backward walk ran `Conv2d(PATCHES) -> ReLU(PATCHES) ->
                // next(DENSE)`. The flip is here. `deadline_is_hard` is deadline
                // PRESENCE and every scored run carries one, so this branch was
                // dead in competition and the ReLU always fell through to the
                // Dense boundary — the exact outcome the comment above forbids.
                // Everything downstream then pays the dense bill: 27% of busy
                // time in the hand-written col2im scatter, ~21% more in scalar
                // certified-error folds, against 59.6% in BLAS microkernels at
                // the last commit that proved this row.
                //
                // Under the lever this asks about EXPIRY instead. With the lever
                // off `hard_finite_authority_refuses_patches` returns
                // `deadline_is_hard`, so the condition is byte-identical to what
                // shipped; armed and still live it takes the native transaction,
                // which is one scheduling unit exactly as the comment describes.
                if !hard_finite_authority_refuses_patches(deadline_is_hard, deadline)
                    && !anchored_relu_candidate
                {
                    if let (Some(limit), Layer::ReLU(relu)) = (deadline, layer) {
                        check_patches_scheduling_deadline(limit, "before native ReLU dispatch")?;
                        match relu.propagate_patches_with_bounds(pb, pre_activation) {
                            Ok(result) => {
                                *crown_bounds = result;
                                return Ok(CrownStepResult::Continue);
                            }
                            Err(
                                NyError::NumericalInstability(_)
                                | NyError::UnsupportedConfiguration(_),
                            ) => {
                                debug!(
                                    "{}: soft-authority ReLU Patches backward refused at layer {}, \
                                     falling back to Dense",
                                    label, layer_idx
                                );
                                if let Some(fallback) = guard_dense_materialization_budget(
                                    crown_bounds,
                                    "activation_numerical_dense_fallback",
                                    layer,
                                    layer_idx,
                                    label,
                                )? {
                                    return Ok(fallback);
                                }
                                let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
                                return crown_backward_step(
                                    layer,
                                    lb,
                                    pre_activation,
                                    engine,
                                    layer_idx,
                                    label,
                                    deadline,
                                )
                                .map_err(Into::into);
                            }
                            Err(error) => return Err(error.into()),
                        }
                    }
                }

                macro_rules! patches_activation_dispatch {
                    ($($Variant:ident),*) => {
                        match layer {
                            $(Layer::$Variant(inner) => {
                                match inner.propagate_patches_with_bounds(pb, pre_activation) {
                                    Ok(result) => {
                                        *crown_bounds = result;
                                        return Ok(CrownStepResult::Continue);
                                    }
                                    Err(
                                        NyError::NumericalInstability(_)
                                        | NyError::UnsupportedConfiguration(_),
                                    ) => {
                                        debug!(
                                            "{}: {} Patches backward refused at layer {}, \
                                             falling back to Dense",
                                            label, stringify!($Variant), layer_idx
                                        );
                                        if let Some(fallback) = guard_dense_materialization_budget(
                                            crown_bounds,
                                            "activation_numerical_dense_fallback",
                                            layer,
                                            layer_idx,
                                            label,
                                        )? {
                                            return Ok(fallback);
                                        }
                                        let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
                                        return crown_backward_step(
                                            layer, lb, pre_activation, engine,
                                            layer_idx, label, deadline,
                                        )
                                        .map_err(Into::into);
                                    }
                                    Err(e) => return Err(e.into()),
                                }
                            })*
                            _ => {} // Not a patches-capable activation, fall through
                        }
                    };
                }
                // The generic activation Patches traits do not carry the
                // absolute node deadline.  Only the explicit cooperative
                // ReLU path above may execute under finite authority; all
                // other hard-finite activations proceed to the deadline-aware
                // Dense/shared dispatcher below. A soft collector timestamp
                // admits the historical native activation as one indivisible
                // scheduling unit.
                // The SECOND door onto the same node, and the only one for every
                // non-ReLU patches-capable activation (Clip, Transpose, Slice,
                // AddConstant, MulConstant, Pad...). Same presence-only shape as
                // the ReLU gate above, so it gets the same expiry treatment:
                // byte-identical with the lever off, native transaction while
                // live when armed. Leaving this one on presence would let any
                // net whose trunk carries such a layer densify exactly as the
                // ReLU nets did.
                if !hard_finite_authority_refuses_patches(deadline_is_hard, deadline) {
                    if let Some(limit) = deadline {
                        check_patches_scheduling_deadline(
                            limit,
                            "before native activation dispatch",
                        )?;
                    }
                    crate::layers::layer_enum::for_each_patches_capable_activation!(
                        patches_activation_dispatch
                    );
                }
            }
            // All other layers (or Dense mode): ensure Dense, then delegate.
            // ensure_dense() is a no-op when already Dense.
            if let Some(fallback) = guard_dense_materialization_budget(
                crown_bounds,
                "generic_dense_dispatch",
                layer,
                layer_idx,
                label,
            )? {
                return Ok(fallback);
            }
            let finite_structured_boundary = materialized_from_finite_patches
                || (hard_finite_authority_refuses_patches(deadline_is_hard, deadline)
                    && matches!(crown_bounds, CrownBounds::Patches(_)));
            let lb = crown_bounds.ensure_dense_with_deadline(deadline)?;
            crown_backward_step_with_dispatch_boundary(
                layer,
                lb,
                pre_activation,
                engine,
                layer_idx,
                label,
                deadline,
                finite_structured_boundary,
            )
            .map_err(Into::into)
        }
    }
}

#[cfg(test)]
#[path = "patches_step_tests.rs"]
mod tests;
