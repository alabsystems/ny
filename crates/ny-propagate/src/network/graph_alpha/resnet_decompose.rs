// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Backward graph decomposition of ResNet suffixes into [`GpuResnetSegment`]s for
//! the sound GPU-resident CROWN backward (the cifar100/tinyimagenet win path).
//!
//! # Why this exists
//!
//! The unary GPU-suffix path ([`super::bounds::gpu_suffix`] / [`super::backward`])
//! bails to the CPU dense backward on any multi-input node — and a residual skip is
//! an `Add` (multi-input). So on ResNets the verdict-deciding alpha-CROWN suffix runs
//! on the slow CPU dense path (which materializes `[num_objectives × conv_dim]`
//! coefficient matrices on the host → the ~7 GB OOM / timeout on cifar100). This
//! module decomposes such a suffix into backward-order chains + residual blocks so it
//! can run on the proven sound GPU-resident resnet backward instead.
//!
//! # Soundness (the −150 firewall)
//!
//! A residual block is recognized ONLY when, by purely **local** checks:
//!
//! 1. the merge node is an exact element-wise `Add` (identity Jacobian on both
//!    inputs, no local relaxation bias), and
//! 2. both branches are **pure unary chains** of GPU-extractable layers terminating
//!    at a common boundary node `z` (the topologically-latest common ancestor).
//!
//! Then `z` provably **dominates** the merge (the only data reaching the merge flows
//! through `z` via the two branches), so `out = F(z) + z` (identity skip) or
//! `out = F(z) + P(z)` (projection skip) holds *exactly*. Independently relaxing each
//! branch as a sound function of `z` and summing is therefore always a valid
//! over-approximation (CROWN composition + interval addition are sound). Picking the
//! topo-latest common ancestor also guarantees the two branch paths are node-disjoint
//! (a shared interior node would itself be a later common ancestor) — we additionally
//! assert disjointness defensively.
//!
//! ANY deviation from this clean structure (a non-`Add` merge, a non-unary or
//! non-extractable branch node, a nested residual inside a branch, S-shaped/sqrt
//! alpha, a missing common ancestor) returns `None`, and the caller falls back to the
//! proven-sound CPU dense backward. The decomposition can only *refuse*; it can never
//! produce an unsound bound.

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
use ny_core::{
    GemmEngine, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, GpuResnetSegment, NyError, Result,
    DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
};
use ny_tensor::BoundedTensor;
use tracing::debug;

/// Process-global cumulative wall time (microseconds) spent inside the resnet GPU
/// backward. A hard ceiling (`resnet_gpu_time_budget_ms`) prevents the per-target
/// resnet attempts from accumulating past the verification deadline: once the
/// budget is spent, every further attempt bails to CPU. This is the runaway/hang
/// firewall — a single GPU call cannot be interrupted mid-flight, so the only safe
/// bound is to stop *starting* new ones.
static RESNET_GPU_MICROS: AtomicU64 = AtomicU64::new(0);

use super::resnet_skeleton::{ResnetSegmentSkeleton, SkeletonRecorder};
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::{Layer, ReLULayer};
use crate::network::core::{
    extract_relu_gpu_layer_with_alpha, try_extract_single_gpu_layer, GraphNetwork, NETWORK_INPUT,
};

/// Cap on the seed size (`num_specs × current_dim`) for the resnet GPU suffix.
///
/// The per-target backward also runs for wide *intermediate* nodes whose identity
/// seed would be enormous (`dim × dim`); a full dense GPU backward there is
/// infeasible and is better left to IBP/CPU. The verdict-deciding output-node seed
/// is tiny (objectives × objectives), so it always passes. Override with
/// `NY_RESNET_GPU_MAX_SEED`. (`run_gpu_checked` still catches any residual OOM and
/// falls back to CPU, so this is a fast-skip optimization, not a soundness gate.)
fn resnet_gpu_max_seed() -> usize {
    std::env::var("NY_RESNET_GPU_MAX_SEED")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1 << 24)
}

/// Master enable for the resnet GPU-resident suffix. Default ON; set
/// `NY_RESNET_GPU=0` to force the CPU dense path (for A/B measurement).
pub(in crate::network::graph_alpha) fn resnet_gpu_enabled() -> bool {
    !matches!(std::env::var("NY_RESNET_GPU").ok().as_deref(), Some("0"))
}

/// Opt-out env-gate predicate: enabled unless the raw value is exactly `"0"`.
/// Pure so the default-on semantics are unit-testable without mutating the
/// process environment (env-var tests are racy under parallel test threads).
fn env_gate_default_on(raw: Option<&str>) -> bool {
    !matches!(raw, Some("0"))
}

/// GPU-resident resnet WARMUP extension (#unsat-keystone step 3b): the
/// per-iteration alpha-warmup output BOUND (`try_gpu_warmup_bound`), the analytic
/// per-ReLU alpha GRADIENTS (`try_gpu_resnet_warmup_gradients`), and the
/// per-disjunct spec-objective gradients (`try_gpu_spec_objective_gradients`).
///
/// Default ON (opt out with `NY_RESNET_WARMUP_GPU=0` for A/B measurement),
/// matching [`resnet_gpu_enabled`]: the CPU dag-alpha backward is ~7 s/iteration
/// on cifar100's conv-resnet, so the warmup alone consumed the whole competition
/// budget and BaB explored 0 domains (measured; BaB > 0 domains with the GPU
/// warmup). Soundness: the warmup only steers alpha — any alpha ∈ [0,1] is a
/// sound relaxation — and the bound it returns is itself a sound GPU enclosure
/// (certified f32 error, directed rounding); every miss/Err/non-finite falls
/// back to the proven CPU path.
pub(crate) fn resnet_warmup_gpu_enabled() -> bool {
    env_gate_default_on(std::env::var("NY_RESNET_WARMUP_GPU").ok().as_deref())
}

/// #root-alpha-gpu dark gate (`NY_ROOT_ALPHA_GPU=1`, default OFF ⇒ byte-identical
/// to today): (A) build the dag-alpha warmup extraction ONCE per optimization
/// loop as a [`super::resnet_skeleton::ResnetSegmentSkeleton`] and re-fold only
/// the per-domain slots each iteration (static `Arc` payloads stay shared across
/// folds), and (B) reuse the loop-top full fold's `(bound, local-rule gradients,
/// segments)` at the gradient site instead of a second extraction + kernel run
/// per iteration.
///
/// Soundness (the 0-wrong moat): neither increment adds a bound channel — the
/// skeleton fold is oracle-proven bit-identical to the legacy extraction
/// whenever both succeed ([`super::resnet_skeleton`] tests) and EVERY miss falls
/// through to the legacy walk (fail closed); the reused gradients only steer α,
/// and any α ∈ [0,1] is a sound relaxation. `NY_RESNET_WARMUP_GPU=0` remains
/// the global kill switch above this gate.
pub(crate) fn root_alpha_gpu_enabled() -> bool {
    std::env::var("NY_ROOT_ALPHA_GPU").ok().as_deref() == Some("1")
}

/// Dark gate (`NY_MULTIOBJ_JOINT_ALPHA=1`, default OFF ⇒ byte-identical to today):
/// steer the warmup / per-disjunct spec-objective α-optimizer with the TRUE JOINT
/// α-gradient (`ny_core::joint_alpha_grad::joint_alpha_gradient`, the FD-validated
/// reverse-mode adjoint of the whole sound CROWN fold — see
/// `docs/BATCHED_BAB_JOINT_ALPHA_GRADIENT.md`) instead of the refuted LOCAL rule
/// `l_i·Σ_s max(A[s,i],0)` (which is stuck at a local optimum ~0.23 above ny's own
/// relaxation LP optimum on the cifar100/tinyimagenet resnet stragglers).
///
/// **Soundness (moat-safe).** The gradient ONLY proposes the next α; every α∈[0,1] is
/// a valid ReLU lower slope, and α is clamped to [0,1] on the write-back path before
/// it is used, so a wrong gradient can only propose a worse α ⇒ a looser-but-sound
/// bound. The reported verdict bound is ALWAYS the sound CROWN fold evaluated at the
/// α actually used — never a gradient-extrapolated value. A wrong gradient can never
/// make the bound unsound (design doc §4).
pub(crate) fn multiobj_joint_alpha_enabled() -> bool {
    std::env::var("NY_MULTIOBJ_JOINT_ALPHA").ok().as_deref() == Some("1")
}

/// Compute the TRUE joint α-gradient (fold order, one `Vec<f32>` per ReLU) for a seed
/// frontier over a domain's resnet `segments` (current α baked into the `Activation`
/// layers), then mask stable neurons (`pre_lowers[r][i] == 0` ⇒ grad 0) so the
/// channel-only α reduction is not corrupted by spurious gradients — matching the
/// local-rule warmup path.
///
/// `seed_lower_a` (`num_specs × output_dim`, row-major) + `seed_lower_b` (`num_specs`)
/// are the spec frontier at the network output — the identity seed for the shared
/// warmup (`num_specs = output_dim`, gradient of `Σ_r lower(output_r)` in ONE adjoint
/// pass), or a single spec row for the per-disjunct objective (`num_specs = 1`).
/// `in_lo/in_hi` are the domain's input box. Returns `None` on any shape mismatch,
/// ReLU-count mismatch, or unsupported topology — the caller then falls back to the
/// local rule (sound either way; the gradient only steers α, never the verdict bound).
#[allow(clippy::too_many_arguments)]
pub(crate) fn joint_alpha_grads_fold(
    segments: &[GpuResnetSegment],
    seed_lower_a: &[f32],
    seed_lower_b: &[f32],
    num_specs: usize,
    output_dim: usize,
    in_lo: &[f32],
    in_hi: &[f32],
    pre_lowers: &[Vec<f32>],
    n_relu_expected: usize,
) -> Option<Vec<Vec<f32>>> {
    // JointGradConfig::default() keeps the bias channel ON (required in production;
    // dropping it is the ~0.7× wrong-but-sound gradient — design doc §2).
    let mut g = ny_core::joint_alpha_grad::joint_alpha_gradient(
        segments,
        seed_lower_a,
        seed_lower_b,
        num_specs,
        output_dim,
        in_lo,
        in_hi,
        ny_core::joint_alpha_grad::JointGradConfig::default(),
    )?;
    if g.len() != n_relu_expected {
        return None;
    }
    for (r, gr) in g.iter_mut().enumerate() {
        if let Some(pl) = pre_lowers.get(r) {
            if pl.len() == gr.len() {
                for (gi, &plv) in gr.iter_mut().zip(pl.iter()) {
                    if plv == 0.0 {
                        *gi = 0.0;
                    }
                }
            }
        }
    }
    Some(g)
}

/// GPU-resident twin of [`joint_alpha_grads_fold`] (#multiobj-joint-alpha-gpu): compute
/// the SAME true-joint α-gradient via the on-device adjoint
/// (`crown_joint_alpha_gradient_resident`) instead of the CPU oracle's whole-network
/// re-fold, then apply the identical stable-neuron mask (`pre_lowers[r][i] == 0` ⇒ grad
/// 0). The GPU adjoint takes no `seed_lower_b`/`JointGradConfig` (it reads its knobs from
/// env and drops the bias accumulator the adjoint does not need). Returns `None` on any
/// `Err`/shape/ReLU-count mismatch so the caller falls back to the CPU oracle, then the
/// local rule.
///
/// MEASURED MOTIVATION (2026-07-14, cifar100 resnet_medium prop_4429, the fastest abc
/// unsat at 9.9 s): the CPU oracle tightens the straggler's root bound −1.31 → −0.689
/// (sound, deterministic, joint_alpha tests green) but its per-disjunct scalar re-fold is
/// slow enough to spend the BaB budget, so BaB reaches only depth 7–8 and times out. This
/// twin moves that same tightening onto the GPU-resident adjoint (already the default
/// joint path in the batched wide-α lane, `batched.rs`) so the root fit is ~instant and
/// BaB keeps its budget.
///
/// SOUNDNESS (moat-safe, identical argument to [`joint_alpha_grads_fold`]): the gradient
/// ONLY proposes the next α; every α∈[0,1] is a valid ReLU lower slope, α is clamped on
/// write-back, and the reported verdict bound is ALWAYS the sound CROWN fold at the α
/// actually used — never a gradient-extrapolated value. A CPU-vs-GPU gradient difference
/// can only change tightness (a worse α ⇒ looser-but-sound bound), never soundness.
#[allow(clippy::too_many_arguments)]
pub(crate) fn joint_alpha_grads_fold_gpu(
    gpu: &dyn ny_core::GpuCrownBackward,
    segments: &[GpuResnetSegment],
    seed_lower_a: &[f32],
    num_specs: usize,
    output_dim: usize,
    in_lo: &[f32],
    in_hi: &[f32],
    pre_lowers: &[Vec<f32>],
    n_relu_expected: usize,
) -> Option<Vec<Vec<f32>>> {
    let mut g = gpu
        .crown_joint_alpha_gradient_resident(
            segments,
            seed_lower_a,
            num_specs,
            output_dim,
            in_lo,
            in_hi,
        )
        .ok()?;
    if g.len() != n_relu_expected {
        return None;
    }
    for (r, gr) in g.iter_mut().enumerate() {
        if let Some(pl) = pre_lowers.get(r) {
            if pl.len() == gr.len() {
                for (gi, &plv) in gr.iter_mut().zip(pl.iter()) {
                    if plv == 0.0 {
                        *gi = 0.0;
                    }
                }
            }
        }
    }
    Some(g)
}

/// Deadline-scored twin of [`joint_alpha_grads_fold_gpu`].  This refuses unless
/// the backend advertises the exact joint-adjoint capability; a backend's
/// generic CROWN deadline flag is deliberately insufficient.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeadlineJointAlphaFoldError {
    DeadlineExpired,
    JointUnavailable,
    NonFiniteGradient,
    MappingMismatch,
}

fn deadline_joint_alpha_check(
    deadline: Instant,
) -> std::result::Result<(), DeadlineJointAlphaFoldError> {
    if Instant::now() >= deadline {
        Err(DeadlineJointAlphaFoldError::DeadlineExpired)
    } else {
        Ok(())
    }
}

fn deadline_joint_alpha_host_work(
    deadline: Instant,
    completed: &mut usize,
) -> std::result::Result<(), DeadlineJointAlphaFoldError> {
    if completed.is_multiple_of(4096) {
        deadline_joint_alpha_check(deadline)?;
    }
    *completed = completed
        .checked_add(1)
        .ok_or(DeadlineJointAlphaFoldError::MappingMismatch)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn joint_alpha_grads_fold_gpu_with_deadline(
    gpu: &dyn ny_core::GpuCrownBackward,
    segments: &[GpuResnetSegment],
    seed_lower_a: &[f32],
    num_specs: usize,
    output_dim: usize,
    in_lo: &[f32],
    in_hi: &[f32],
    pre_lowers: &[Vec<f32>],
    n_relu_expected: usize,
    deadline: Instant,
) -> std::result::Result<Vec<Vec<f32>>, DeadlineJointAlphaFoldError> {
    deadline_joint_alpha_check(deadline)?;
    if !gpu.provides_deadline_bounded_joint_alpha_gradient_resident() {
        return Err(DeadlineJointAlphaFoldError::JointUnavailable);
    }
    let mut gradients = match gpu.crown_joint_alpha_gradient_resident_with_deadline(
        segments,
        seed_lower_a,
        num_specs,
        output_dim,
        in_lo,
        in_hi,
        deadline,
    ) {
        Ok(gradients) => gradients,
        Err(error) if error.is_deadline_exceeded() || Instant::now() >= deadline => {
            return Err(DeadlineJointAlphaFoldError::DeadlineExpired);
        }
        Err(NyError::NumericalInstability(_)) => {
            return Err(DeadlineJointAlphaFoldError::NonFiniteGradient);
        }
        Err(_) => return Err(DeadlineJointAlphaFoldError::JointUnavailable),
    };
    deadline_joint_alpha_check(deadline)?;
    if gradients.len() != n_relu_expected {
        return Err(DeadlineJointAlphaFoldError::JointUnavailable);
    }
    if pre_lowers.len() != n_relu_expected {
        return Err(DeadlineJointAlphaFoldError::MappingMismatch);
    }
    if gradients
        .iter()
        .zip(pre_lowers)
        .any(|(gradient, pre_lower)| gradient.len() != pre_lower.len())
    {
        return Err(DeadlineJointAlphaFoldError::MappingMismatch);
    }

    // Validate the raw backend result before masking: a stable-neuron zero must
    // never launder NaN/Inf into an apparently valid gradient.
    let mut host_work = 0usize;
    for gradient in &gradients {
        for &value in gradient {
            deadline_joint_alpha_host_work(deadline, &mut host_work)?;
            if !value.is_finite() {
                return Err(DeadlineJointAlphaFoldError::NonFiniteGradient);
            }
        }
    }
    deadline_joint_alpha_check(deadline)?;

    for (gradient, pre_lower) in gradients.iter_mut().zip(pre_lowers) {
        for (value, &lower) in gradient.iter_mut().zip(pre_lower) {
            deadline_joint_alpha_host_work(deadline, &mut host_work)?;
            if lower == 0.0 {
                *value = 0.0;
            }
        }
    }
    deadline_joint_alpha_check(deadline)?;
    Ok(gradients)
}

/// Dark gate (`NY_MULTIOBJ_JOINT_ALPHA_GPU=1`, requires [`multiobj_joint_alpha_enabled`]
/// too; default OFF ⇒ CPU oracle, byte-identical to today): route the true-joint
/// α-gradient through the GPU-resident adjoint [`joint_alpha_grads_fold_gpu`] instead of
/// the CPU `joint_alpha_gradient`, so the cifar100/tinyimagenet root/straggler tightening
/// does not spend the BaB budget on the CPU whole-network re-fold. Non-soundness-critical
/// (steers α∈[0,1]; the sound fold recomputes the verdict).
pub(crate) fn multiobj_joint_alpha_gpu_enabled() -> bool {
    std::env::var("NY_MULTIOBJ_JOINT_ALPHA_GPU").ok().as_deref() == Some("1")
}

/// Beta-capable GPU resnet per-domain extension (#unsat-keystone step 4): the
/// BaB per-domain / constrained-backward bound with the β-CROWN split dual
/// folded in (`try_gpu_beta_perdomain_bound`, `try_gpu_beta_batched_resnet`,
/// `try_gpu_beta_constrained_backward`).
///
/// Default ON (opt out with `NY_RESNET_BETA_GPU=0` for A/B measurement): the CPU
/// dense per-domain backward is the cifar100/tinyimagenet UNSAT wall (~60 s per
/// domain vs ~180 ms on the GPU resnet backward). Soundness: a β-CROWN bound is
/// a valid Lagrangian dual for ANY β ≥ 0, the GPU resnet backward is a sound
/// enclosure (certified f32 error), and every miss/Err/non-finite falls back to
/// the proven CPU path — the 0-wrong moat is preserved.
pub(crate) fn resnet_beta_gpu_enabled() -> bool {
    env_gate_default_on(std::env::var("NY_RESNET_BETA_GPU").ok().as_deref())
}

/// #batched-bab: route the per-domain resnet GPU beta backward through the BATCHED
/// entry (`crown_backward_gpu_resnet_sound_beta_batched`) instead of one call per
/// domain. DEFAULT ON (opt out with `NY_BAB_RESNET_BATCHED=0`, matching
/// [`resnet_beta_gpu_enabled`]). The reference stacker is byte-identical to the
/// serial loop (differential oracle green); the β-opt-eligible scored path takes
/// the wide β lane below. Gate OFF → today's per-domain serial/rayon loop,
/// byte-identical.
///
/// #w5-bab-throughput A/B (2026-07-11, cifar100 resnet_medium fast-unsats,
/// release, `sample` + `NY_ACASXU_PROF`): the DEFAULT scored path was a serial
/// per-domain `gpu_beta_optimize_domain` loop whose main thread sat 63% BLOCKED
/// in the GPU device wait (`pollster::block_on`→`DynDevice::wait`) dispatching one
/// small per-domain backward at a time, +37% single-threaded CPU coeff-gather with
/// all 14 rayon workers PARKED, plus per-pass Metal buffer-alloc churn. Collapsing
/// the batch into ONE wide GPU pass per β iteration lifts explored domains 4× (63→
/// 255) at chunk 8 and 8× (63→511) at chunk 64 over the SAME 57s BaB budget — the
/// wide fold is β-parity with the serial lane (element-wise-tightest iterates,
/// iterate-0 = the serial single-shot bound), so verdicts are unchanged.
pub(crate) fn resnet_beta_gpu_batched_enabled() -> bool {
    env_gate_default_on(std::env::var("NY_BAB_RESNET_BATCHED").ok().as_deref())
}

/// #refold-guard (batched-BaB increment: class-C → class-B): runtime
/// wide↔serial spot-check on the batched resnet single-pass lane (batched.rs
/// `run_batched`). The batched trait entry internally prefers the ONE-pass
/// wide kernel (`NY_BAB_RESNET_WIDE`, default ON); its documented failure
/// modes are cross-domain row misassignment (HOLE-3/4 stacking) and the
/// wg-limit "silent over-tight bound" driver UB — either can prune a domain
/// as verified on another domain's bound (false-VERIFY, class-C, on a lane
/// that is DEFAULT ON in scored runs; today only test-time oracles check it).
/// The guard re-folds two domains per batch (a deterministic anchor + the
/// most verified-looking row) through the SERIAL sound backward on the SAME
/// backend and requires the oracle-proven two-sided 1e-3 relative contract
/// (wide reorders f32 GEMMs, so bitwise is only true for the internal
/// stacker fallback); ANY mismatch or serial refusal abandons the whole wide
/// result to the proven serial/rayon loop (downgrade-only — the guard can
/// only prevent verdicts, never create one). DEFAULT ON — the guard is the
/// price of trusting the wide fold; `NY_BAB_RESNET_REFOLD_GUARD=0` restores
/// the unguarded historical behavior for A/B measurement.
pub(crate) fn resnet_refold_guard_enabled() -> bool {
    env_gate_default_on(std::env::var("NY_BAB_RESNET_REFOLD_GUARD").ok().as_deref())
}

/// #batched-bab part A: route the per-domain β ascent through the WIDE batched grad
/// backward (`gpu_beta_optimize_wide`) instead of the serial per-domain
/// `gpu_beta_optimize_domain` loop. Requires `resnet_beta_gpu_batched_enabled()`
/// too (the outer batched gate). Any miss → serial fallback. SOUND either way:
/// every iterate is a valid Lagrangian dual for its β ≥ 0 over the SAME child
/// sub-domain and only the element-wise-tightest sound iterate is kept — never
/// looser than the single-shot lane.
///
/// **DEFAULT FLIPPED TO OFF 2026-08-13 — it was a MEASURED PESSIMISATION.** The
/// lane shipped default-ON to apply "wide-fold throughput to the β-opt-eligible
/// (scored) path". Measured through `[phase] mo-wave-stage` on cifar100
/// idx_7641, IN THE SCORED CONFIGURATION (`OMP_NUM_THREADS=1`, as
/// `vnncomp_scripts/run_instance.sh` exports it):
///
/// ```text
/// wide lane ON  : bwd 43.19s / 43.22s   per_child 5.41s
/// wide lane OFF : bwd 24.76s            per_child 3.11s
/// ```
///
/// **1.75x FASTER with the wide lane OFF**, on the phase that is 99.8% of a BaB
/// wave, with the frontier bound byte-identical (`worst=-11.02619`, 16 domains).
/// The two ON runs agree to 0.03 s, so this is far outside noise. A 14-row
/// cifar100 sweep at official budgets showed 14/14 verdicts unchanged — no
/// conversions, no regressions.
///
/// FIRST measured at 1.30x under `OMP_NUM_THREADS=20`; the effect is LARGER at
/// the scored thread count because the serial fallback is what actually benefits
/// from the wide fold's absence. Always A/B this at OMP_NUM_THREADS=1: measuring
/// it at 20 understates the gap.
///
/// Opt back in with `NY_BAB_RESNET_WIDE_BETA=1`.
/// #wide-beta-reflip: DEFAULT-ON again, and the reversal is the point.
///
/// This was flipped OFF earlier on a measured 1.76x pessimization (per_child
/// 5.41s -> 3.08s with it off). That measurement was CORRECT AT THE TIME and is
/// now STALE: it was taken when the root census was 0/99, so every child priced
/// all 99 objectives. With the row-chunked root sweep the census is 87/99,
/// `union_rows` collapses 99 -> 4, and the economics invert.
///
/// MEASURED, cifar100 idx_8600, official 100s budget:
///   off: mo-wave-stage children=8 union_rows=12 bwd=5.28s, frontier -1.01375,
///        `[wide-lane] declines: entry_wide_beta_gate_off=2`
///   on:  mo-wave-stage children=6 union_rows=4  bwd=2.09s, frontier -0.98610,
///        `[wide-lane] published=13, declines: none`
///
/// 2.5x faster per wave, it stops declining the domain-stacked GPU lane
/// entirely, and the frontier improves -- the first time this row has gone below
/// 1.0. Moat: 12/12 banked verdicts, then all six marginal unsat rows re-run 3x
/// each for 18/18, every one FASTER than with it off.
///
/// The lesson worth keeping: a lever's measurement is only valid for the
/// configuration it was taken in. This one became wrong the moment the root
/// improved, and nothing would have re-tested it automatically.
///
/// `NY_BAB_RESNET_WIDE_BETA=0` restores the previous default.
pub(crate) fn resnet_beta_gpu_wide_beta_enabled() -> bool {
    !matches!(
        std::env::var("NY_BAB_RESNET_WIDE_BETA").ok().as_deref(),
        Some("0") | Some("false")
    )
}

/// #w4 wide α+β ascent: per-domain α re-optimization inside the wide batched β
/// loop (the cifar100/tinyimagenet frozen-root-α tightness lever — the batched
/// lane folds warmup-inherited α and never adapts it to each sub-domain's tighter
/// post-split bounds). Dark until the `[converge]` A/B validates; requires
/// `NY_BAB_RESNET_WIDE_BETA=1` too (the ascent lives inside that loop).
/// Non-soundness-critical: any α ∈ [0,1] is a valid lower relaxation slope; the
/// bound is recomputed by the sound wide fold every iteration.
pub(crate) fn resnet_beta_gpu_wide_alpha_enabled() -> bool {
    matches!(
        std::env::var("NY_BAB_RESNET_WIDE_ALPHA").ok().as_deref(),
        Some("1")
    )
}

/// #metaroom-chain-wide: allow the BaB batched GPU β lane
/// (`try_gpu_beta_batched_resnet_opt`) to consume PURE-CHAIN suffix decompositions
/// (`segments = [Chain(..)]`, no residual block). The GPU kernel side is already
/// segment-kind-agnostic — the batched point-VJP lane drives the SAME wide resident
/// fold with pure-Chain segment lists — only the extractor refused them (metaroom's
/// 6cnn conv chains therefore fell to the dense node-by-node batched backward, the
/// BaB throughput wall). DEFAULT OFF (opt-in `NY_BAB_CHAIN_WIDE=1`) — dark until the
/// chain-only differential oracle + the metaroom A/B validate. OFF → the extractor
/// keeps the >=1-residual gate and routing is byte-identical. SOUND either way: the
/// chain decomposition feeds the same sound GPU-resident backward (certified f32
/// error, directed rounding), and every miss/Err still falls back to the proven
/// dense path.
pub(crate) fn bab_chain_wide_enabled() -> bool {
    matches!(
        std::env::var("NY_BAB_CHAIN_WIDE").ok().as_deref(),
        Some("1")
    )
}

// #clip-gather-probe L3: WHICH `None` exit of [`extract_gpu_resnet_segments_collect`]
// refused. Telemetry ONLY — it is written on refusal paths and never read by any
// decision, so no verdict, bound, segment list, or control-flow branch depends on
// it, and the success path never touches it. Thread-local because wave-batched
// preps run under rayon and a shared cell would cross-talk between domains.
//
// Sound to read straight after a `None`: every refusal below writes the slot before
// returning, so a `None` observed on this thread means this thread just wrote the
// reason. Callers that did not just observe a `None` must not read it.
thread_local! {
    static EXTRACT_LAST_REFUSAL: std::cell::Cell<&'static str> =
        const { std::cell::Cell::new("extract_not_recorded") };
}

fn note_extract_refusal(reason: &'static str) {
    EXTRACT_LAST_REFUSAL.with(|slot| slot.set(reason));
}

/// The reason [`extract_gpu_resnet_segments_collect`] last returned `None` on THIS
/// thread. Only meaningful immediately after observing that `None`.
pub(crate) fn extract_segments_last_refusal() -> &'static str {
    EXTRACT_LAST_REFUSAL.with(std::cell::Cell::get)
}

/// Cap on the number of objective rows (`num_specs`) the resnet GPU suffix will
/// attempt. The verdict-deciding backward has few objectives; very wide
/// *intermediate*-node backwards (whose `num_specs` is the node's flat dim) are far
/// better left to IBP/CPU than run as a full dense GPU resnet backward. Override
/// with `NY_RESNET_GPU_MAX_OBJECTIVES`.
pub(in crate::network::graph_alpha) fn resnet_gpu_max_objectives() -> usize {
    std::env::var("NY_RESNET_GPU_MAX_OBJECTIVES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512)
}

/// Hard ceiling on cumulative resnet-GPU-backward wall time (ms) per process.
/// Default 30s; override with `NY_RESNET_GPU_TIME_BUDGET_MS`. Set to 0 to disable
/// the resnet GPU path entirely via the budget.
fn resnet_gpu_time_budget_ms() -> u64 {
    std::env::var("NY_RESNET_GPU_TIME_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30_000)
}

/// Whether the per-call timing trace is enabled (`NY_RESNET_GPU_TRACE=1`). Uses
/// `eprintln!` (unbuffered) so the line survives even a hard wall-clock kill.
fn resnet_gpu_trace() -> bool {
    matches!(
        std::env::var("NY_RESNET_GPU_TRACE").ok().as_deref(),
        Some("1")
    )
}

/// Resolve a node's pre-activation bounds the same way the suffix backward does:
/// `NETWORK_INPUT` → the input box, else the crown bound, else the IBP bound.
// pub(in graph_alpha) so the #extract-skeleton fold resolves per-domain bounds
// through the IDENTICAL lookup chain (byte-identity by construction).
pub(in crate::network::graph_alpha) fn resolve_pre<
    'a,
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    input: &'a BoundedTensor,
    name: &str,
    crown_bounds: &'a HashMap<String, V1>,
    ibp_bounds: &'a HashMap<String, V2>,
) -> Option<&'a BoundedTensor> {
    if name == NETWORK_INPUT {
        Some(input)
    } else {
        crown_bounds
            .get(name)
            .map(Borrow::borrow)
            .or_else(|| ibp_bounds.get(name).map(Borrow::borrow))
    }
}

/// A node carries an active NON-ReLU alpha (S-shaped / sqrt / reciprocal) that is not
/// supported on the GPU backward — such a node forces CPU fallback.
// pub(in graph_alpha) so the #extract-skeleton fold re-runs the SAME per-domain
// refusal over the build walk's visited nodes (`None`-agreement is behavior).
pub(in crate::network::graph_alpha) fn has_active_non_relu_alpha(
    node_name: &str,
    alpha_state: Option<&GraphAlphaState>,
) -> bool {
    let Some(alpha_state) = alpha_state else {
        return false;
    };
    alpha_state.monotone_s_shaped_alpha(node_name).is_some()
        || alpha_state.sqrt_alpha(node_name).is_some()
        || alpha_state.reciprocal_alpha(node_name).is_some()
}

/// Bake ONE ReLU node's relaxation layer from its CURRENT pre-activation bounds
/// and (optional) alpha state — the per-domain half of the #extract-skeleton
/// static/dynamic extraction split.
///
/// Alpha present with a live pair + mask ⇒ the alpha-aware extraction, which
/// itself decides `Activation` vs `ActivationReluDualAlpha` PER DOMAIN (the
/// variant depends on this domain's bridged lower/upper alphas and must never be
/// frozen into a cross-domain skeleton). Otherwise ⇒ the default-CROWN
/// `relu_linear_relaxation` `Activation`, routed through the SAME
/// `try_extract_single_gpu_layer` ReLU arm the pre-split code used (`ReLULayer`
/// is a stateless unit struct, so a fresh value IS the node's layer). Returns
/// `None` (→ CPU fallback) on any non-contiguous bounds/alpha slice — this
/// helper can only refuse, never produce a wrong layer.
///
/// Called by BOTH the legacy per-domain extraction (`extract_node_layer`) and
/// `ResnetSegmentSkeleton::fold_for_domain`, so fold-vs-legacy byte-identity of
/// the ReLU relaxation holds by construction, not by parallel maintenance.
pub(in crate::network::graph_alpha) fn bake_relu_layer(
    node_name: &str,
    pre_activation: &BoundedTensor,
    alpha_state: Option<&GraphAlphaState>,
) -> Option<GpuCrownLayer> {
    if let Some(alpha_state) = alpha_state {
        if let Some((alpha_lower, alpha_upper)) = alpha_state.relu_alpha_pair(node_name) {
            if let Some(unstable_mask) = alpha_state.relu_unstable_mask(node_name) {
                let pre_l = pre_activation.lower().as_slice()?;
                let pre_u = pre_activation.upper().as_slice()?;
                // #4404: expand channel-only alpha to full spatial for GPU extraction.
                let al = alpha_state.expand_alpha(node_name, alpha_lower);
                let au = alpha_state.expand_alpha(node_name, alpha_upper);
                let mask = if alpha_state.spatial_shape(node_name).is_some() {
                    alpha_state.expand_mask(node_name, unstable_mask)
                } else {
                    unstable_mask.clone()
                };
                return Some(extract_relu_gpu_layer_with_alpha(
                    pre_l,
                    pre_u,
                    al.as_slice()?,
                    au.as_slice()?,
                    mask.as_slice()?,
                ));
            }
        }
    }
    let mut out = Vec::with_capacity(1);
    try_extract_single_gpu_layer(&Layer::ReLU(ReLULayer), pre_activation, &mut out)?;
    // Defensive: the ReLU arm pushes exactly one layer; anything else refuses.
    if out.len() != 1 {
        return None;
    }
    out.pop()
}

/// Extract the GPU layer descriptor(s) for a single node into `out`, using
/// alpha-aware ReLU slopes when alpha state is present (mirrors the unary
/// suffix's `try_extract_single_gpu_layer_with_alpha`). Returns `None` (→ CPU
/// fallback) for any layer that cannot be represented as a `GpuCrownLayer`.
fn extract_node_layer(
    node_name: &str,
    layer: &Layer,
    pre_activation: &BoundedTensor,
    alpha_state: Option<&GraphAlphaState>,
    out: &mut Vec<GpuCrownLayer>,
) -> Option<()> {
    if let Layer::ReLU(_) = layer {
        // Shared with the #extract-skeleton fold — see `bake_relu_layer`.
        out.push(bake_relu_layer(node_name, pre_activation, alpha_state)?);
        return Some(());
    }
    try_extract_single_gpu_layer(layer, pre_activation, out)
}

/// #cgan-bn-gpu-extract protocol: BatchNorm pushes the exact 1×1 diagonal conv
/// and stashes its discharge; the NEXT extracted node must be the feeding ReLU
/// (plain Activation), which absorbs it. Everything else refuses (fail-closed).
#[allow(clippy::too_many_arguments)]
fn extract_node_layer_bn(
    node_name: &str,
    layer: &Layer,
    pre_activation: &BoundedTensor,
    alpha_state: Option<&GraphAlphaState>,
    out: &mut Vec<GpuCrownLayer>,
    allow_bn: bool,
    pending_bn_werr: &mut Option<Vec<f32>>,
) -> Option<()> {
    use crate::network::core::{apply_bn_werr_to_host_relu, try_extract_batch_norm_conv1x1};
    if let Layer::BatchNorm(bn) = layer {
        if !allow_bn || pending_bn_werr.is_some() {
            return None; // lane not opted in, or BN→BN with no host ReLU between
        }
        let (conv, werr) = try_extract_batch_norm_conv1x1(bn, pre_activation)?;
        out.push(conv);
        *pending_bn_werr = Some(werr);
        return Some(());
    }
    if pending_bn_werr.is_some() {
        // The BN's producer must be its host ReLU. Clear `pending` ONLY after the
        // discharge fully applied: on any failure below the caller truncates the
        // partial push and the still-set marker VETOES a frozen stop here (a
        // stack ending in an undischarged BN would drop the certified widen).
        if !matches!(layer, Layer::ReLU(_)) {
            return None;
        }
        extract_node_layer(node_name, layer, pre_activation, alpha_state, out)?;
        apply_bn_werr_to_host_relu(out.last_mut()?, pending_bn_werr.as_ref()?)?;
        *pending_bn_werr = None;
        return Some(());
    }
    extract_node_layer(node_name, layer, pre_activation, alpha_state, out)
}

/// Walk a pure unary chain backward from `branch_start` until reaching `z`,
/// extracting each node's GPU layer(s) in backward order. Returns the layer vec and
/// the ordered set of visited node names, or `None` if the chain is not a pure unary
/// GPU-extractable path that terminates exactly at `z`.
///
/// `branch_start == z` yields an empty path (the identity-skip case).
///
/// `rec` is the #extract-skeleton BUILD recorder: `Some` only when
/// `build_resnet_segment_skeleton` drives this walk (all legacy callers pass
/// `None`, keeping the historical path byte-identical). Recording can only add
/// a REFUSAL (an unclassifiable layer refuses the build), never change a layer.
#[allow(clippy::too_many_arguments)]
fn extract_unary_path_to_z<V1: Borrow<BoundedTensor>, V2: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    branch_start: &str,
    z: &str,
    input: &BoundedTensor,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
    mut rec: Option<&mut SkeletonRecorder>,
) -> Option<(Vec<GpuCrownLayer>, Vec<String>)> {
    let mut layers = Vec::new();
    let mut visited = Vec::new();
    let mut current = branch_start.to_string();
    // A simple DAG chain cannot exceed the node count; bound the walk defensively.
    let max_steps = graph.nodes.len() + 1;
    let mut steps = 0;
    while current != z {
        steps += 1;
        if steps > max_steps {
            return None;
        }
        // Reaching the network input without hitting `z` means `z` is not on this
        // chain (we overshot the real divergence) — not a clean diamond.
        if current == NETWORK_INPUT {
            return None;
        }
        if has_active_non_relu_alpha(&current, alpha_state) {
            return None;
        }
        let node = graph.nodes.get(&current)?;
        if node.inputs.len() != 1 {
            // A multi-input node inside a branch (e.g. a nested residual) — bail.
            return None;
        }
        let input_name = node.require_unary_input().ok()?;
        let pre = resolve_pre(input, input_name, crown_bounds, ibp_bounds)?;
        let before = layers.len();
        extract_node_layer(&current, &node.layer, pre, alpha_state, &mut layers)?;
        if let Some(r) = rec.as_deref_mut() {
            r.record_visited(&current);
            r.record_resolved(input_name, pre);
            r.record_layer(&node.layer, &current, input_name, layers.len() - before)?;
        }
        visited.push(current.clone());
        current = input_name.to_string();
    }
    Some((layers, visited))
}

/// The topologically-latest common ancestor of `in_a` and `in_b`, or
/// `NETWORK_INPUT` when their only common origin is the network input. Returns
/// `None` if no common ancestor can be resolved.
fn common_ancestor(graph: &GraphNetwork, in_a: &str, in_b: &str) -> Option<String> {
    // An identity skip wired straight from the network input: the input box itself
    // is the block input `z`.
    if in_a == NETWORK_INPUT || in_b == NETWORK_INPUT {
        return Some(NETWORK_INPUT.to_string());
    }
    let anc = graph.all_ancestors().ok()?;
    // Ancestor lists are topologically ordered and include the node itself, and
    // exclude NETWORK_INPUT.
    let anc_a = anc.get(in_a)?;
    let set_b: HashSet<&str> = anc.get(in_b)?.iter().map(String::as_str).collect();
    // Topo-latest shared node = the closest (lowest) common ancestor.
    if let Some(z) = anc_a.iter().rev().find(|n| set_b.contains(n.as_str())) {
        return Some(z.clone());
    }
    // No shared *node*: both branches diverge at the network input itself.
    Some(NETWORK_INPUT.to_string())
}

/// The ReLU node names among `visited` (a branch's nodes in backward order), in that
/// order. Matches the order their `Activation` layers appear in the branch's GPU
/// layer vec, hence the fold order the resident backward captures gradients in.
fn relu_names_in(graph: &GraphNetwork, visited: &[String]) -> Vec<String> {
    visited
        .iter()
        .filter(|n| {
            matches!(
                graph.nodes.get(n.as_str()).map(|nd| &nd.layer),
                Some(Layer::ReLU(_))
            )
        })
        .cloned()
        .collect()
}

/// Decompose a residual `Add(in_a, in_b)` into a [`GpuResnetSegment`] plus the block
/// input node `z` to continue the backward walk from, and the block's per-ReLU node
/// names in FOLD order (F-branch ReLUs then P-branch ReLUs — the order
/// `crown_backward_sound_resident_resnet_seeded` captures their gradients). `None` →
/// CPU fallback.
#[allow(clippy::too_many_arguments)]
fn decompose_residual_block<V1: Borrow<BoundedTensor>, V2: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    in_a: &str,
    in_b: &str,
    input: &BoundedTensor,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
    // #extract-skeleton BUILD recorder (`None` on every legacy path). Branch
    // layer origins accumulate in the recorder's pending buffer during each
    // branch walk; they are staged here and committed (with the final segment
    // index) by the collect loop right after it pushes the block segment.
    mut rec: Option<&mut SkeletonRecorder>,
) -> Option<(GpuResnetSegment, String, Vec<String>)> {
    let z = common_ancestor(graph, in_a, in_b)?;

    if z == in_a || z == in_b {
        // One input IS the block input → identity skip; F is the other chain.
        let f_start = if z == in_a { in_b } else { in_a };
        let (f, visited) = extract_unary_path_to_z(
            graph,
            f_start,
            &z,
            input,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            rec.as_deref_mut(),
        )?;
        if f.is_empty() {
            // Both inputs are `z`: `out = z + z`, not a residual block we model.
            return None;
        }
        if let Some(r) = rec {
            let f_origins = r.take_pending();
            r.stage_block(f_origins, Vec::new());
        }
        let names = relu_names_in(graph, &visited);
        Some((GpuResnetSegment::Residual(f), z, names))
    } else {
        // Projection skip: both branches run from `z` to the merge.
        let (f, f_nodes) = extract_unary_path_to_z(
            graph,
            in_a,
            &z,
            input,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            rec.as_deref_mut(),
        )?;
        // Drain F's origins BEFORE the P walk refills the pending buffer.
        let f_origins = rec.as_deref_mut().map(SkeletonRecorder::take_pending);
        let (p, p_nodes) = extract_unary_path_to_z(
            graph,
            in_b,
            &z,
            input,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            rec.as_deref_mut(),
        )?;
        if f.is_empty() || p.is_empty() {
            return None;
        }
        // Defensive disjointness guard: the two branches must share no interior node
        // (guaranteed by the topo-latest `z`, asserted here so a structural surprise
        // can only refuse, never double-count).
        let f_set: HashSet<&str> = f_nodes.iter().map(String::as_str).collect();
        if p_nodes.iter().any(|n| f_set.contains(n.as_str())) {
            return None;
        }
        if let Some(r) = rec {
            let p_origins = r.take_pending();
            r.stage_block(f_origins.unwrap_or_default(), p_origins);
        }
        let mut names = relu_names_in(graph, &f_nodes);
        names.extend(relu_names_in(graph, &p_nodes));
        Some((GpuResnetSegment::ResidualProj(f, p), z, names))
    }
}

/// Decompose the backward suffix from `start_node` to `NETWORK_INPUT` into
/// backward-order [`GpuResnetSegment`]s (plain chains + identity/projection residual
/// blocks), or `None` when the suffix is not a clean chain/residual structure the
/// sound GPU-resident backward can handle.
///
/// Returns `Some` ONLY when the suffix contains at least one residual block; pure
/// unary suffixes are left to the existing unary GPU-suffix path (this is purely a
/// resnet *extension*, never a replacement of the proven unary path).
///
/// Test-only convenience wrapper: production callers go through
/// `extract_gpu_resnet_segments_with_relu_names` / the `_ext` variant.
#[cfg(test)]
pub(in crate::network::graph_alpha) fn extract_gpu_resnet_segments<
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    start_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
) -> Option<Vec<GpuResnetSegment>> {
    let mut names = Vec::new();
    let mut frontier_abs = Vec::new();
    let mut stopped_at: Option<String> = None;
    extract_gpu_resnet_segments_collect(
        graph,
        input,
        start_node,
        crown_bounds,
        ibp_bounds,
        alpha_state,
        &mut names,
        &mut frontier_abs,
        false,
        None,
        false,
        false,
        &mut stopped_at,
    )
}

/// Gradient-warmup variant: also returns the per-ReLU node names in FOLD order — the
/// order [`crown_backward_sound_resident_resnet_seeded`] captures their gradients,
/// so the caller can map GPU per-ReLU gradients back to DAG ReLU node alphas. Same
/// decomposition (same `Some`/`None` decisions) as `extract_gpu_resnet_segments`
/// (the test-only wrapper).
// pub(crate) so the beta_crown BaB engine can reuse it for the per-domain GPU beta
// backward (#unsat-keystone step 4); decomposition is logically shared (verdict suffix
// AND BaB per-domain), and it can only refuse (returns None) — never produce an unsound result.
//
// Returns `(segments, relu_names, frontier_abs, node_abs)`:
//  - `frontier_abs[seg]` = the per-SEGMENT frontier-node abs-max bound (input-side).
//  - `node_abs[k]` = the per-ReLU PRE-activation abs-max bound (`max(|pre_l|,|pre_u|)` per
//    dim) in the SAME FOLD order as `relu_names` — the finer per-ReLU concretization
//    frontier. Derived exactly like `pre_lowers` is built at the warmup/beta call sites
//    (the ReLU's INPUT-node bounds), but UNMASKED: `node_abs[k][j] ≥ |pre-activation j|`
//    holds for every neuron (the true pre-activation lies in `[pre_l[j],pre_u[j]]`), which
//    is the soundness contract the per-ReLU error fold needs. If any ReLU's input bounds
//    cannot be resolved, `node_abs` is left empty (the backend then degrades to the
//    per-segment fold — still sound).
pub(crate) fn extract_gpu_resnet_segments_with_relu_names<
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    start_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
) -> Option<(
    Vec<GpuResnetSegment>,
    Vec<String>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
)> {
    let (segments, names, frontier_abs, node_abs, stopped_at) =
        extract_gpu_segments_with_relu_names_ext(
            graph,
            input,
            start_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
            false,
            false,
            false,
        )?;
    debug_assert!(stopped_at.is_none(), "frozen_stop=false can never stop");
    Some((segments, names, frontier_abs, node_abs))
}

/// #metaroom-chain-wide extension of [`extract_gpu_resnet_segments_with_relu_names`]:
/// with `allow_pure_chain = true` a suffix WITHOUT any residual block decomposes to
/// `[Chain(layers)]` (plus fold-order ReLU names / frontier / node abs tables built
/// exactly as for resnets) instead of refusing. `allow_pure_chain = false` is
/// byte-identical to the resnet-only entry. Only the BaB batched β lane passes
/// `true` (behind [`bab_chain_wide_enabled`]); the verdict-suffix path keeps
/// the `>=1`-residual gate so the proven unary GPU-suffix path stays in charge
/// there. Like the resnet entry, this can only refuse (`None` → dense/CPU
/// fallback) — never produce an unsound decomposition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_gpu_segments_with_relu_names_ext<
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    start_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
    allow_pure_chain: bool,
    // #cgan-bn-gpu-extract: accept surviving BatchNorm as an exact 1x1 diagonal
    // conv (discharge folded into the feeding ReLU). Root-joint lane only.
    allow_bn: bool,
    // #root-joint-frozen-stop: allow the walk to terminate at the deepest
    // frozen-bounded node M it cannot walk past (5th tuple slot names M).
    frozen_stop: bool,
) -> Option<(
    Vec<GpuResnetSegment>,
    Vec<String>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Option<String>,
)> {
    let mut names = Vec::new();
    let mut frontier_abs = Vec::new();
    let mut stopped_at: Option<String> = None;
    let segments = extract_gpu_resnet_segments_collect(
        graph,
        input,
        start_node,
        crown_bounds,
        ibp_bounds,
        alpha_state,
        &mut names,
        &mut frontier_abs,
        allow_pure_chain,
        None,
        allow_bn,
        frozen_stop,
        &mut stopped_at,
    )?;
    // Per-ReLU PRE-activation abs-max bounds in FOLD order (same order as `names`). The
    // ReLU's pre-activation is its INPUT node's bounds — resolved exactly as the
    // warmup/beta paths resolve `pre` for `pre_lowers`. UNMASKED abs-max is a sound
    // over-approximation of `|pre-activation|` for the error fold. On any unresolvable
    // ReLU we leave node_abs EMPTY (the backend falls back to the per-segment fold).
    let node_abs =
        collect_relu_pre_abs(graph, input, &names, crown_bounds, ibp_bounds).unwrap_or_default();
    if std::env::var("NY_DECOMP_PROBE").ok().as_deref() == Some("1") {
        use std::fmt::Write as _;
        let mut s = String::new();
        for seg in &segments {
            let _ = match seg {
                GpuResnetSegment::Chain(c) => write!(s, " Chain({})", c.len()),
                GpuResnetSegment::Residual(c) => write!(s, " Res({})", c.len()),
                GpuResnetSegment::ResidualProj(f, p) => {
                    write!(s, " ResProj({},{})", f.len(), p.len())
                }
            };
        }
        eprintln!(
            "[decomp] start={start_node} segments={} relus={} node_abs={} stop={}:{}",
            segments.len(),
            names.len(),
            node_abs.len(),
            stopped_at.as_deref().unwrap_or("-"),
            s
        );
    }
    Some((segments, names, frontier_abs, node_abs, stopped_at))
}

/// Per-ReLU PRE-activation abs-max bound (`max(|pre_l|,|pre_u|)` per dim) for each ReLU in
/// `relu_names`, in that order. The ReLU's pre-activation is its single input node's bounds,
/// resolved the same way the suffix backward resolves them (`NETWORK_INPUT` → the input box,
/// else crown bound, else IBP bound). Returns `None` if any ReLU's input bounds cannot be
/// resolved (so the caller can degrade to the coarser per-segment concretization). The
/// abs-max is UNMASKED: `max(|l|,|u|) ≥ |pre-activation|` holds for every neuron, which is
/// exactly the soundness contract `concretize_error_into_bias` requires.
// pub(in graph_alpha) so the #extract-skeleton fold derives its per-domain
// `node_abs` through the IDENTICAL helper (byte-identity by construction,
// including the identical empty-on-unresolvable degradation).
pub(in crate::network::graph_alpha) fn collect_relu_pre_abs<
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    relu_names: &[String],
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
) -> Option<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(relu_names.len());
    for name in relu_names {
        let node = graph.nodes.get(name)?;
        let input_name = node.require_unary_input().ok()?;
        let pre = resolve_pre(input, input_name, crown_bounds, ibp_bounds)?;
        let abs: Vec<f32> = pre
            .lower()
            .iter()
            .zip(pre.upper().iter())
            .map(|(&l, &u)| l.abs().max(u.abs()))
            .collect();
        out.push(abs);
    }
    Some(out)
}

/// abs-max bound of a node's frontier output (max(|l|,|u|) per dim), resolved
/// exactly the way the walk resolves pre-activations.
// pub(in graph_alpha) so the #extract-skeleton fold rebuilds the per-domain
// `frontier_abs` tables through the IDENTICAL arithmetic (byte-identity by
// construction), including the identical `None`-on-unresolvable refusal.
pub(in crate::network::graph_alpha) fn frontier_abs_max_of<
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    input: &BoundedTensor,
    name: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
) -> Option<Vec<f32>> {
    let bt = resolve_pre(input, name, crown_bounds, ibp_bounds)?;
    Some(
        bt.lower()
            .iter()
            .zip(bt.upper().iter())
            .map(|(&l, &u)| l.abs().max(u.abs()))
            .collect(),
    )
}

/// Shared core: decompose the suffix into segments while collecting the fold-order
/// per-ReLU node names into `relu_names` (chain ReLUs as encountered, then each
/// residual block's F-then-P ReLUs — matching the segment order the fold processes).
// pub(in graph_alpha) so `build_resnet_segment_skeleton` (#extract-skeleton) can
// drive THIS walk (with its recorder attached) rather than a parallel one — the
// skeleton's structure is the legacy structure by construction.
#[allow(clippy::too_many_arguments)]
pub(in crate::network::graph_alpha) fn extract_gpu_resnet_segments_collect<
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    start_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
    relu_names: &mut Vec<String>,
    frontier_abs: &mut Vec<Vec<f32>>,
    allow_pure_chain: bool,
    // #extract-skeleton BUILD recorder. `None` on every legacy/production call —
    // every recording line below is guarded, so the historical path is untouched.
    // Recording can only add a REFUSAL (unclassifiable layer ⇒ build `None`),
    // never change a produced segment.
    mut rec: Option<&mut SkeletonRecorder>,
    // PHASE B (#cgan-bn-gpu-extract): accept surviving BatchNorm nodes as an exact
    // 1×1 diagonal Conv2d whose precompute-error discharge is folded into the
    // feeding ReLU's intercepts. Only the root-joint interm-α driver passes true;
    // false = byte-identical (BN keeps hitting the catch-all refusal).
    allow_bn: bool,
    // PHASE A (#root-joint-frozen-stop): permit the walk to TERMINATE at the
    // deepest node M it cannot walk past, PROVIDED M has a finite frozen bound in
    // the maps. The returned stack is then EXACTLY graph(M→L] and the caller MUST
    // concretize against box(M) — sound because reachable(M) ⊆ box(M).
    // false = byte-identical (stack must reach NETWORK_INPUT or refuse).
    frozen_stop: bool,
    // OUT: Some(M) iff the walk stopped at frozen node M (frozen_stop only).
    stopped_at: &mut Option<String>,
) -> Option<Vec<GpuResnetSegment>> {
    // #extract-skeleton x #image-node-crown fail-closed seam: the BUILD recorder
    // documents the LEGACY walk only. The gated new semantics (BN-as-1x1-conv,
    // frozen stop) must never be baked into a skeleton — refuse the combination
    // outright (the skeleton path itself declines whenever either flag is set;
    // see `prep_resnet_domain_with`).
    if rec.is_some() && (allow_bn || frozen_stop) {
        return None;
    }
    let mut segments: Vec<GpuResnetSegment> = Vec::new();
    let mut chain: Vec<GpuCrownLayer> = Vec::new();
    let mut chain_names: Vec<String> = Vec::new();
    let mut current = start_node.to_string();
    let mut saw_residual = false;
    let max_steps = graph.nodes.len() + 1;
    let mut steps = 0;
    // abs-max bound of a node's frontier output (max(|l|,|u|) per dim).
    let abs_max_of = |name: &str| frontier_abs_max_of(input, name, crown_bounds, ibp_bounds);
    // PHASE B: a just-extracted BatchNorm's pending error discharge; MUST be
    // absorbed by the next extracted node (its feeding ReLU) or the extraction
    // refuses — and it VETOES any frozen stop while set (a stack ending in an
    // undischarged BN would drop the certified scale_err/bias_err widen).
    let mut pending_bn_werr: Option<Vec<f32>> = None;
    // PHASE A stop eligibility: M must carry a NON-EMPTY, ALL-FINITE, NON-CROSSED
    // bound under the SAME crown→ibp preference the extraction resolves with
    // (resolve_pre; the loop breaks on NETWORK_INPUT before any refusal can name
    // it), and at least one layer must already be accumulated.
    // (#cone-delta: maps are Borrow-genericized on main — resolve through
    // `Borrow::borrow` exactly as `resolve_pre` does.)
    let stop_eligible = |name: &str, have_layers: bool| -> bool {
        if !frozen_stop || !have_layers {
            return false;
        }
        let bt: &BoundedTensor = if let Some(v) = crown_bounds.get(name) {
            v.borrow()
        } else if let Some(v) = ibp_bounds.get(name) {
            v.borrow()
        } else {
            return false;
        };
        !bt.lower().is_empty()
            && bt
                .lower()
                .iter()
                .zip(bt.upper().iter())
                .all(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u)
    };

    // #clip-gather-probe L3: telemetry-only sub-reason for the step-fail exit.
    let mut step_reason: &'static str = "extract_step_unclassified";
    let stop: Option<String> = loop {
        steps += 1;
        if steps > max_steps {
            note_extract_refusal("extract_cycle_guard");
            return None; // cycle guard: hard refusal, never a stop
        }
        if current == NETWORK_INPUT {
            if pending_bn_werr.is_some() {
                note_extract_refusal("extract_bn_at_input");
                return None; // BN fed directly by the input: no host ReLU
            }
            break None;
        }
        // Uniform frozen-stop conversion: any refusal to walk PAST `current`
        // becomes a stop AT `current` when eligible. At that point the segments
        // are exactly graph(current→L] — every extractor pushes only on success,
        // and the truncate below makes that independent of evaluation order.
        let chain_ckpt = chain.len();
        let step_ok = 'step: {
            if has_active_non_relu_alpha(&current, alpha_state) {
                step_reason = "extract_step_non_relu_alpha";
                break 'step false;
            }
            let Some(node) = graph.nodes.get(&current) else {
                step_reason = "extract_step_node_missing";
                break 'step false;
            };
            if let Some(r) = rec.as_deref_mut() {
                // Every walked node (Add merges included) is re-checked for a
                // per-domain non-ReLU alpha by the fold — same refusal set as above.
                r.record_visited(&current);
            }
            match node.inputs.len() {
                1 => {
                    let Ok(input_name) = node.require_unary_input().map(str::to_string) else {
                        step_reason = "extract_step_not_unary";
                        break 'step false;
                    };
                    let Some(pre) = resolve_pre(input, &input_name, crown_bounds, ibp_bounds)
                    else {
                        step_reason = "extract_step_pre_unresolved";
                        break 'step false;
                    };
                    let before = chain.len();
                    if extract_node_layer_bn(
                        &current,
                        &node.layer,
                        pre,
                        alpha_state,
                        &mut chain,
                        allow_bn,
                        &mut pending_bn_werr,
                    )
                    .is_none()
                    {
                        step_reason = "extract_step_layer_unsupported";
                        break 'step false;
                    }
                    if let Some(r) = rec.as_deref_mut() {
                        r.record_resolved(&input_name, pre);
                        if r.record_layer(&node.layer, &current, &input_name, chain.len() - before)
                            .is_none()
                        {
                            step_reason = "extract_step_recorder";
                            break 'step false;
                        }
                    }
                    if matches!(node.layer, Layer::ReLU(_)) {
                        chain_names.push(current.clone());
                    }
                    current = input_name;
                }
                2 => {
                    // Residual merge — Add only (exact identity Jacobian, zero local
                    // bias). Sub (negated second branch) and any other binary op bail
                    // to CPU. A pending BN discharge must never straddle a segment
                    // boundary.
                    if pending_bn_werr.is_some() || !matches!(node.layer, Layer::Add(_)) {
                        step_reason = "extract_step_binary_not_add";
                        break 'step false;
                    }
                    let in_a = node.inputs[0].clone();
                    let in_b = node.inputs[1].clone();
                    // #extract-skeleton x #image-node-crown: flush the plain chain
                    // accumulated downstream of this block BEFORE the block walk
                    // (main's historical order — the recorder's pending buffer must
                    // hold only ONE accumulation at a time; its ReLU names precede
                    // the block's in fold order). Frozen-stop semantics are
                    // unchanged: on a later failure in this arm the already-flushed
                    // Chain (frontier `current`) is exactly what the branch's
                    // stop-path final flush would have produced at stop node
                    // `current`, and the step-fail conversion below still stops
                    // AT `current` (segments non-empty keeps it eligible).
                    if !chain.is_empty() {
                        let Some(cur_abs) = abs_max_of(&current) else {
                            step_reason = "extract_step_chain_frontier_abs";
                            break 'step false;
                        };
                        frontier_abs.push(cur_abs);
                        let seg_idx = segments.len();
                        segments.push(GpuResnetSegment::Chain(std::mem::take(&mut chain)));
                        relu_names.append(&mut chain_names);
                        if let Some(r) = rec.as_deref_mut() {
                            r.commit_chain(seg_idx, &current);
                        }
                    }
                    let Some((seg, z, mut block_names)) = decompose_residual_block(
                        graph,
                        &in_a,
                        &in_b,
                        input,
                        crown_bounds,
                        ibp_bounds,
                        alpha_state,
                        rec.as_deref_mut(),
                    ) else {
                        step_reason = "extract_step_block_decompose";
                        break 'step false;
                    };
                    // The residual block's frontier (after its backward) is the block input z.
                    let Some(z_abs) = abs_max_of(&z) else {
                        step_reason = "extract_step_block_frontier_abs";
                        break 'step false;
                    };
                    frontier_abs.push(z_abs);
                    let seg_idx = segments.len();
                    segments.push(seg);
                    relu_names.append(&mut block_names);
                    if let Some(r) = rec.as_deref_mut() {
                        r.commit_block(seg_idx, &z);
                    }
                    saw_residual = true;
                    current = z;
                }
                _ => {
                    step_reason = "extract_step_arity";
                    break 'step false;
                }
            }
            true
        };
        if !step_ok {
            chain.truncate(chain_ckpt); // defensive: discard any partial push
            if pending_bn_werr.is_none()
                && stop_eligible(&current, !(chain.is_empty() && segments.is_empty()))
            {
                break Some(current.clone());
            }
            note_extract_refusal(step_reason);
            return None; // historical refusal — byte-identical when !frozen_stop
        }
    };

    if !chain.is_empty() {
        // Final Chain's frontier = the stack's INPUT: NETWORK_INPUT normally, or
        // the frozen stop node M. abs_max_of resolves both through the same maps,
        // so the frontier abs-max is the sup over the ACTUAL fold domain in both
        // cases (the concretize_error_into_bias contract, gemm.rs:1110-1124).
        // (rec ⇒ !frozen_stop ⇒ stop is None ⇒ the recorder always commits the
        // historical NETWORK_INPUT frontier.)
        let frontier_name: &str = stop.as_deref().unwrap_or(NETWORK_INPUT);
        let Some(final_abs) = abs_max_of(frontier_name) else {
            note_extract_refusal("extract_final_frontier_abs");
            return None;
        };
        frontier_abs.push(final_abs);
        let seg_idx = segments.len();
        segments.push(GpuResnetSegment::Chain(std::mem::take(&mut chain)));
        relu_names.append(&mut chain_names);
        if let Some(r) = rec {
            r.commit_chain(seg_idx, frontier_name);
        }
    }
    if segments.is_empty() {
        note_extract_refusal("extract_segments_empty");
        return None;
    }
    if !saw_residual && !allow_pure_chain {
        note_extract_refusal("extract_pure_chain_disallowed");
        return None;
    }
    debug_assert_eq!(frontier_abs.len(), segments.len());
    *stopped_at = stop;
    Some(segments)
}

/// Drive the sound GPU-resident **resnet** CROWN backward for a target node whose
/// ancestor suffix is a clean chain/residual structure, seeded from `seed_lb`.
///
/// Returns the concretized `[num_specs]` bounds, or `Ok(None)` to fall back to the
/// proven-sound CPU dense backward — on a missing/unsupported sound GPU engine, a
/// non-decomposable suffix, a non-finite/oversized seed, a GPU error, or NaN output.
/// The bounds are a sound enclosure (the GPU-resident backward carries directed,
/// over-bounded f32 error throughout), so this is safe to decide verdicts under the
/// soundness gate; CPU fallback on every other path preserves the 0-wrong moat.
// pub(crate) so the spec-guided CROWN root pass (graph_crown/spec_propagation) can
// seed the SAME sound GPU resnet backward with the multi-objective C matrix
// (#w4-root-gpu). Like the decomposition itself, this entry can only refuse
// (`Ok(None)` → CPU fallback) — it never produces an unsound bound.
#[derive(Clone, Copy)]
enum ResnetGpuDispatch {
    Ordinary,
    DeadlineBoundedSingleRow(Instant),
    DeadlineBoundedRows(Instant),
}

/// Apply the ordinary process-wide admission budget only to the ordinary route.
///
/// The deadline-bounded single-row experiment has its own private deadline and
/// must neither read nor inherit the state of this unrelated throughput guard.
fn resnet_gpu_dispatch_budget_allows_start_with(
    dispatch: ResnetGpuDispatch,
    ordinary_elapsed: &AtomicU64,
    ordinary_budget_ms: u64,
) -> bool {
    match dispatch {
        ResnetGpuDispatch::Ordinary => {
            ordinary_budget_ms != 0
                && ordinary_elapsed.load(Ordering::Relaxed) / 1000 < ordinary_budget_ms
        }
        ResnetGpuDispatch::DeadlineBoundedSingleRow(_)
        | ResnetGpuDispatch::DeadlineBoundedRows(_) => true,
    }
}

fn resnet_gpu_dispatch_budget_allows_start(dispatch: ResnetGpuDispatch) -> bool {
    match dispatch {
        ResnetGpuDispatch::Ordinary => resnet_gpu_dispatch_budget_allows_start_with(
            dispatch,
            &RESNET_GPU_MICROS,
            resnet_gpu_time_budget_ms(),
        ),
        ResnetGpuDispatch::DeadlineBoundedSingleRow(_)
        | ResnetGpuDispatch::DeadlineBoundedRows(_) => true,
    }
}

/// Charge elapsed GPU time only to the ordinary route's process-wide budget.
///
/// `None` identifies the deadline-bounded experiment: its elapsed time remains
/// available to call-local trace telemetry, but is never stored in or returned
/// as the ordinary cumulative counter.
fn account_resnet_gpu_elapsed_with(
    ordinary_elapsed: &AtomicU64,
    dispatch: ResnetGpuDispatch,
    elapsed_us: u64,
) -> Option<u64> {
    match dispatch {
        ResnetGpuDispatch::Ordinary => {
            Some(ordinary_elapsed.fetch_add(elapsed_us, Ordering::Relaxed) + elapsed_us)
        }
        ResnetGpuDispatch::DeadlineBoundedSingleRow(_)
        | ResnetGpuDispatch::DeadlineBoundedRows(_) => None,
    }
}

fn account_resnet_gpu_elapsed(dispatch: ResnetGpuDispatch, elapsed_us: u64) -> Option<u64> {
    account_resnet_gpu_elapsed_with(&RESNET_GPU_MICROS, dispatch, elapsed_us)
}

/// Validate and convert an optional backend result without letting repair
/// create a new authoritative interval. Every dispatch kind must return the
/// exact finite, already-ordered row payload; otherwise its caller restores
/// the established host fallback.
fn resnet_gpu_result_to_bounds(
    _dispatch: ResnetGpuDispatch,
    expected_rows: usize,
    result: GpuCrownResult,
) -> Result<Option<BoundedTensor>> {
    // All GPU routes are optional accelerators. A malformed ordinary payload
    // must take the same CPU fallback as a malformed private-deadline payload;
    // repairing it here would turn device validation failure into authority.
    if !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&result, expected_rows) {
        return Ok(None);
    }
    let (Ok(lower), Ok(upper)) = (
        ArrayD::from_shape_vec(IxDyn(&[expected_rows]), result.lower_bounds),
        ArrayD::from_shape_vec(IxDyn(&[expected_rows]), result.upper_bounds),
    ) else {
        return Ok(None);
    };
    Ok(BoundedTensor::new(lower, upper).ok())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_resnet_gpu_suffix<V1: Borrow<BoundedTensor>, V2: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    target_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    seed_lb: &LinearBounds,
) -> Result<Option<BoundedTensor>> {
    // The ordinary helper performs topology/segment walks and full seed/input
    // copies before its backend deadline begins. Finite callers must not enter
    // that unpollable host preparation; the dedicated private single/bounded-
    // row experiments remain separate APIs and are not production proof
    // authority for this route.
    // #gpu-suffix-expiry set-mate: default-off, byte-identical unarmed.
    if let Some(limit) = deadline {
        if Instant::now() >= limit {
            return Err(NyError::DeadlineExceeded(
                "ResNet GPU suffix deadline expired before host preparation".into(),
            ));
        }
        if crate::sound_gpu_gate::gpu_suffix_declines_under_finite_authority(limit) {
            return Ok(None);
        }
    }
    try_resnet_gpu_suffix_with_dispatch(
        graph,
        input,
        target_node,
        crown_bounds,
        ibp_bounds,
        alpha_state,
        engine,
        deadline,
        seed_lb,
        ResnetGpuDispatch::Ordinary,
    )
}

/// Narrow sibling for the critical-row experiment. It reuses the exact same
/// decomposition and result validation, but dispatches only the backend's
/// explicit call-local one-row API: no ordinary CUDA route, global deadline
/// lease, forward candidate, or CPU fallback can run here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_resnet_gpu_suffix_single_row_with_deadline<
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    target_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
    seed_lb: &LinearBounds,
) -> Result<Option<BoundedTensor>> {
    try_resnet_gpu_suffix_with_dispatch(
        graph,
        input,
        target_node,
        crown_bounds,
        ibp_bounds,
        None,
        engine,
        Some(deadline),
        seed_lb,
        ResnetGpuDispatch::DeadlineBoundedSingleRow(deadline),
    )
}

/// Alpha-bearing sibling of
/// [`try_resnet_gpu_suffix_single_row_with_deadline`].
///
/// It retains the dedicated call-local one-row dispatch and all of its strict
/// result/deadline validation, but bakes the caller's exact clamped alpha state
/// into the ResNet fold.  This is intentionally a separate typed surface:
/// ordinary callers cannot accidentally opt into the critical-row experiment,
/// and the historical fresh-slope surface above remains byte-identical.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_resnet_gpu_suffix_single_row_with_alpha_and_deadline<
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    target_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: &GraphAlphaState,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
    seed_lb: &LinearBounds,
) -> Result<Option<BoundedTensor>> {
    try_resnet_gpu_suffix_with_dispatch(
        graph,
        input,
        target_node,
        crown_bounds,
        ibp_bounds,
        Some(alpha_state),
        engine,
        Some(deadline),
        seed_lb,
        ResnetGpuDispatch::DeadlineBoundedSingleRow(deadline),
    )
}

/// Complete result of one strict, deadline-bounded small-row ResNet fold.
///
/// The folded segments and row-independent input box are the exact operands
/// used by the backend call. Active-set callers may therefore replay one
/// binding row on the host without running a second topology extraction.
#[allow(dead_code)] // Phase-1 plumbing: fields are consumed by the follow-up root integration.
pub(crate) struct DeadlineBoundedResnetRowsResult {
    pub(crate) bounds: BoundedTensor,
    pub(crate) segments: Vec<GpuResnetSegment>,
    pub(crate) relu_names: Vec<String>,
    pub(crate) input_lower: Vec<f32>,
    pub(crate) input_upper: Vec<f32>,
}

/// Alpha-bearing, strict-skeleton sibling for one atomic `2..=8` row
/// deadline-bounded sound ResNet backward.
///
/// This entry is deliberately additive. K=1 remains owned by
/// [`try_resnet_gpu_suffix_single_row_with_alpha_and_deadline`] and is refused
/// here, so wiring the active-set route cannot divert the sealed critical-row
/// path. The supplied skeleton must describe this exact output suffix and fold
/// successfully for the supplied alpha state; there is no legacy-extraction or
/// ordinary-GPU fallback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_resnet_gpu_suffix_bounded_rows_with_alpha_and_deadline<
    V1: Borrow<BoundedTensor>,
    V2: Borrow<BoundedTensor>,
>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    target_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: &GraphAlphaState,
    skeleton: &ResnetSegmentSkeleton,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
    seed_lb: &LinearBounds,
) -> Result<Option<DeadlineBoundedResnetRowsResult>> {
    if !resnet_gpu_enabled() || Instant::now() >= deadline {
        return Ok(None);
    }

    let num_specs = seed_lb.num_outputs();
    let current_dim = seed_lb.num_inputs();
    if !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&num_specs)
        || current_dim == 0
        || seed_lb
            .lower_a()
            .iter()
            .chain(seed_lb.upper_a().iter())
            .chain(seed_lb.lower_b().iter())
            .chain(seed_lb.upper_b().iter())
            .any(|value| !value.is_finite())
    {
        return Ok(None);
    }

    let Some(gpu) = engine
        .and_then(|candidate| candidate.as_gpu_crown_backward())
        .filter(|candidate| candidate.provides_sound_gpu_crown())
    else {
        return Ok(None);
    };
    let capacity = gpu.deadline_bounded_resnet_sound_max_rows();
    if capacity > DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS || capacity < num_specs {
        return Ok(None);
    }

    if skeleton.cache_key() != (target_node, false) || !skeleton.matches_graph(graph) {
        return Ok(None);
    }
    let Some((segments, relu_names, frontier_abs, node_abs)) =
        skeleton.fold_for_domain(graph, input, crown_bounds, ibp_bounds, Some(alpha_state))
    else {
        return Ok(None);
    };
    if Instant::now() >= deadline {
        return Ok(None);
    }

    let input_flat = input.flatten();
    let input_lower: Vec<f32> = input_flat.lower().iter().copied().collect();
    let input_upper: Vec<f32> = input_flat.upper().iter().copied().collect();
    if input_lower.is_empty()
        || input_lower.len() != input_upper.len()
        || input_lower
            .iter()
            .zip(&input_upper)
            .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
    {
        return Ok(None);
    }

    let seed = GpuCrownSeed {
        lower_a: seed_lb.lower_a().iter().copied().collect::<Vec<_>>().into(),
        upper_a: seed_lb.upper_a().iter().copied().collect::<Vec<_>>().into(),
        lower_b: seed_lb.lower_b().iter().copied().collect::<Vec<_>>().into(),
        upper_b: seed_lb.upper_b().iter().copied().collect::<Vec<_>>().into(),
        num_specs,
        current_dim,
    };
    if Instant::now() >= deadline {
        return Ok(None);
    }
    let dispatch = ResnetGpuDispatch::DeadlineBoundedRows(deadline);
    let started_at = Instant::now();
    let result = gpu.crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &segments,
        &seed,
        &input_lower,
        &input_upper,
        &frontier_abs,
        &node_abs,
        deadline,
    );
    let elapsed_us = started_at.elapsed().as_micros() as u64;
    let _ordinary_cumulative = account_resnet_gpu_elapsed(dispatch, elapsed_us);
    if Instant::now() >= deadline {
        return Ok(None);
    }
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            debug!(
                target_node,
                num_specs,
                error = %error,
                "deadline-bounded rows resnet GPU suffix refused"
            );
            return Ok(None);
        }
    };
    let Some(bounds) = resnet_gpu_result_to_bounds(dispatch, num_specs, result)? else {
        return Ok(None);
    };

    Ok(Some(DeadlineBoundedResnetRowsResult {
        bounds,
        segments,
        relu_names,
        input_lower,
        input_upper,
    }))
}

#[allow(clippy::too_many_arguments)]
fn try_resnet_gpu_suffix_with_dispatch<V1: Borrow<BoundedTensor>, V2: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    target_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    seed_lb: &LinearBounds,
    dispatch: ResnetGpuDispatch,
) -> Result<Option<BoundedTensor>> {
    if !resnet_gpu_enabled() {
        return Ok(None);
    }
    // Hang/runaway firewall for ordinary calls: never start another unbounded
    // ResNet backward once its process-wide budget is spent. The dedicated
    // one-row call has a stronger, call-local contract: bounded CUDA dispatches
    // plus an explicit private deadline. It neither reads nor charges the
    // ordinary throughput budget.
    if !resnet_gpu_dispatch_budget_allows_start(dispatch) {
        return Ok(None);
    }
    if let Some(deadline) = deadline {
        if Instant::now() >= deadline {
            return Ok(None);
        }
    }
    // Only the SOUND GPU-resident resnet backward is eligible; it is sound under
    // directed/over-bounded f32 error, so it is safe whether or not the soundness
    // gate is engaged (unlike the round-to-nearest fast path). A non-sound engine
    // → CPU fallback.
    let Some(gpu) = engine
        .and_then(|e| e.as_gpu_crown_backward())
        .filter(|g| g.provides_sound_gpu_crown())
    else {
        return Ok(None);
    };
    match dispatch {
        ResnetGpuDispatch::Ordinary => {
            if !crate::sound_gpu_gate::gpu_crown_backend_honors_deadline(gpu, deadline) {
                return Ok(None);
            }
        }
        ResnetGpuDispatch::DeadlineBoundedSingleRow(_) => {
            if !gpu.provides_deadline_bounded_single_row_resnet_sound() {
                return Ok(None);
            }
        }
        ResnetGpuDispatch::DeadlineBoundedRows(_) => {
            let capacity = gpu.deadline_bounded_resnet_sound_max_rows();
            if capacity > DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS {
                return Ok(None);
            }
        }
    }

    let num_specs = seed_lb.num_outputs();
    let current_dim = seed_lb.num_inputs();
    if num_specs == 0 || current_dim == 0 {
        return Ok(None);
    }
    match dispatch {
        ResnetGpuDispatch::Ordinary => {}
        ResnetGpuDispatch::DeadlineBoundedSingleRow(_) if num_specs != 1 => return Ok(None),
        ResnetGpuDispatch::DeadlineBoundedRows(_)
            if !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&num_specs)
                || gpu.deadline_bounded_resnet_sound_max_rows() < num_specs =>
        {
            return Ok(None);
        }
        ResnetGpuDispatch::DeadlineBoundedSingleRow(_)
        | ResnetGpuDispatch::DeadlineBoundedRows(_) => {}
    }
    if num_specs > resnet_gpu_max_objectives() {
        return Ok(None);
    }
    if num_specs.saturating_mul(current_dim) > resnet_gpu_max_seed() {
        return Ok(None);
    }

    // Reject non-finite seed coefficients/bias (Inf in the relaxation matrices would
    // make the GPU backward meaningless).
    if seed_lb.lower_a().iter().any(|v| !v.is_finite())
        || seed_lb.upper_a().iter().any(|v| !v.is_finite())
        || seed_lb.lower_b().iter().any(|v| !v.is_finite())
        || seed_lb.upper_b().iter().any(|v| !v.is_finite())
    {
        return Ok(None);
    }

    // Same decomposition as the unary path, but keep the per-segment `frontier_abs`
    // (frontier-node abs-max bounds) AND the per-ReLU `node_abs` (pre-activation abs-max
    // bounds, fold order) so the main bound gets the SAME #unsat-keystone error-
    // concretization the warmup (`_grad`) and BaB (`_beta`) paths use. With `node_abs`
    // threaded, the AUTO-FALLBACK prefers the FINER per-ReLU concretization on an error-
    // explosion (deep cifar100/tinyimagenet resnets), recovering a finite, capped bound.
    // Both are computed regardless; the gate-off (non-exploding) path is byte-for-byte
    // unchanged because the un-concretized first pass and its USELESS threshold are
    // untouched — the fine path only runs when that cheap bound already failed.
    let Some((segments, _relu_names, frontier_abs, node_abs)) =
        extract_gpu_resnet_segments_with_relu_names(
            graph,
            input,
            target_node,
            crown_bounds,
            ibp_bounds,
            alpha_state,
        )
    else {
        debug!(
            target_node = target_node,
            num_specs, "resnet GPU suffix: suffix not decomposable (→ CPU fallback)"
        );
        return Ok(None);
    };
    let total_layers: usize = segments
        .iter()
        .map(|s| match s {
            GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => l.len(),
            GpuResnetSegment::ResidualProj(f, p) => f.len() + p.len(),
        })
        .sum();
    debug!(
        target_node = target_node,
        num_specs,
        current_dim,
        segments = segments.len(),
        total_layers,
        "resnet GPU suffix: decomposed; dispatching sound GPU-resident backward"
    );

    let seed = GpuCrownSeed {
        lower_a: seed_lb.lower_a().iter().copied().collect::<Vec<_>>().into(),
        upper_a: seed_lb.upper_a().iter().copied().collect::<Vec<_>>().into(),
        lower_b: seed_lb.lower_b().iter().copied().collect::<Vec<_>>().into(),
        upper_b: seed_lb.upper_b().iter().copied().collect::<Vec<_>>().into(),
        num_specs,
        current_dim,
    };
    let input_lower: Vec<f32> = input.lower().iter().copied().collect();
    let input_upper: Vec<f32> = input.upper().iter().copied().collect();

    let trace = resnet_gpu_trace();
    if trace {
        match dispatch {
            ResnetGpuDispatch::Ordinary => eprintln!(
                "[resnet-gpu] dispatch target={target_node} num_specs={num_specs} \
                 current_dim={current_dim} segments={} layers={total_layers}",
                segments.len()
            ),
            ResnetGpuDispatch::DeadlineBoundedSingleRow(_) => eprintln!(
                "[resnet-gpu] dispatch target={target_node} num_specs={num_specs} \
                 current_dim={current_dim} segments={} layers={total_layers} \
                 accounting=critical-call-local ordinary_budget_charged=false",
                segments.len()
            ),
            ResnetGpuDispatch::DeadlineBoundedRows(_) => eprintln!(
                "[resnet-gpu] dispatch target={target_node} num_specs={num_specs} \
                 current_dim={current_dim} segments={} layers={total_layers} \
                 accounting=bounded-rows-call-local ordinary_budget_charged=false",
                segments.len()
            ),
        }
    }
    if deadline.is_some_and(|value| Instant::now() >= value) {
        return Ok(None);
    }
    let t0 = Instant::now();
    let result = match dispatch {
        ResnetGpuDispatch::Ordinary => {
            let _deadline_scope =
                crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, deadline);
            gpu.crown_backward_gpu_resnet_sound(
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &frontier_abs,
                &node_abs,
            )
        }
        ResnetGpuDispatch::DeadlineBoundedSingleRow(deadline) => gpu
            .crown_backward_gpu_resnet_sound_single_row_with_deadline(
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &frontier_abs,
                &node_abs,
                deadline,
            ),
        ResnetGpuDispatch::DeadlineBoundedRows(deadline) => gpu
            .crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                &segments,
                &seed,
                &input_lower,
                &input_upper,
                &frontier_abs,
                &node_abs,
                deadline,
            ),
    };
    if deadline.is_some_and(|value| Instant::now() >= value) {
        let us = t0.elapsed().as_micros() as u64;
        let _ordinary_cumulative = account_resnet_gpu_elapsed(dispatch, us);
        return Ok(None);
    }
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            let us = t0.elapsed().as_micros() as u64;
            let _ordinary_cumulative = account_resnet_gpu_elapsed(dispatch, us);
            if trace {
                eprintln!(
                    "[resnet-gpu] target={target_node} ERR after {}ms: {error}",
                    us / 1000
                );
            }
            debug!(
                target_node = target_node,
                error = %error,
                "resnet GPU suffix failed; falling back to CPU backward"
            );
            return Ok(None);
        }
    };
    let elapsed_us = t0.elapsed().as_micros() as u64;
    let ordinary_cumulative = account_resnet_gpu_elapsed(dispatch, elapsed_us);
    if trace {
        match ordinary_cumulative {
            Some(cumulative) => eprintln!(
                "[resnet-gpu] target={target_node} returned in {}ms (cumulative {}ms)",
                elapsed_us / 1000,
                cumulative / 1000
            ),
            None => eprintln!(
                "[resnet-gpu] target={target_node} returned in {}ms \
                 (accounting=critical-call-local ordinary_budget_charged=false)",
                elapsed_us / 1000
            ),
        }
    }
    debug!(
        target_node = target_node,
        elapsed_ms = elapsed_us / 1000,
        "resnet GPU suffix: GPU-resident backward returned"
    );

    let Some(bounds) = resnet_gpu_result_to_bounds(dispatch, num_specs, result)? else {
        match dispatch {
            ResnetGpuDispatch::Ordinary => debug!(
                target_node = target_node,
                "resnet GPU suffix produced NaN; falling back to CPU backward"
            ),
            ResnetGpuDispatch::DeadlineBoundedSingleRow(_) => debug!(
                target_node = target_node,
                "deadline-bounded resnet GPU suffix produced an invalid raw result; \
                 refusing publication"
            ),
            ResnetGpuDispatch::DeadlineBoundedRows(_) => debug!(
                target_node = target_node,
                "deadline-bounded rows resnet GPU suffix produced an invalid raw result; \
                 refusing publication"
            ),
        }
        return Ok(None);
    };
    debug!(
        target_node = target_node,
        num_specs, current_dim, "resnet GPU suffix decided bounds on GPU-resident sound backward"
    );
    Ok(Some(bounds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer, SubLayer};
    use crate::network::core::GraphNode;
    use crate::network::{build_resnet_segment_skeleton, SpecCrownRequest};
    use ndarray::{arr1, arr2, Array2, ArrayD, IxDyn};
    use ny_core::GpuCrownBackward;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use std::time::Duration;

    fn lin(name: &str, input: &str) -> GraphNode {
        // A simple 2→2 Linear with a small bias.
        let w = arr2(&[[0.7_f32, -0.3], [0.2, 0.6]]);
        let b = arr1(&[0.05_f32, -0.04]);
        let layer = Layer::Linear(LinearLayer::new(w, Some(b)).expect("valid linear"));
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }
    fn relu(name: &str, input: &str) -> GraphNode {
        GraphNode::new(name, Layer::ReLU(ReLULayer), vec![input.to_string()])
    }
    fn input_box() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[2]), -0.5_f32),
            ArrayD::from_elem(IxDyn(&[2]), 0.5_f32),
        )
        .expect("valid input")
    }
    fn decompose(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        start: &str,
    ) -> Option<Vec<GpuResnetSegment>> {
        let bounds = graph.collect_node_bounds(input).expect("node bounds");
        extract_gpu_resnet_segments(graph, input, start, &bounds, &bounds, None)
    }
    fn decompose_names(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        start: &str,
    ) -> Option<(Vec<GpuResnetSegment>, Vec<String>)> {
        let bounds = graph.collect_node_bounds(input).expect("node bounds");
        // extract_..._with_relu_names returns a 4-tuple (segments, relu_names,
        // frontier abstractions, per-ReLU node_abs); these tests only assert on
        // segments + names.
        extract_gpu_resnet_segments_with_relu_names(graph, input, start, &bounds, &bounds, None)
            .map(|(segs, names, _frontier, _node_abs)| (segs, names))
    }
    fn chain_len(seg: &GpuResnetSegment) -> usize {
        match seg {
            GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => l.len(),
            GpuResnetSegment::ResidualProj(f, p) => f.len() + p.len(),
        }
    }

    struct BoundedRowsMock {
        capacity: usize,
        bounded_calls: AtomicUsize,
        single_calls: AtomicUsize,
    }

    impl BoundedRowsMock {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                bounded_calls: AtomicUsize::new(0),
                single_calls: AtomicUsize::new(0),
            }
        }

        fn evaluate_seed(seed: &GpuCrownSeed) -> GpuCrownResult {
            assert_eq!(seed.lower_a.len(), seed.num_specs * seed.current_dim);
            assert_eq!(seed.upper_a.len(), seed.num_specs * seed.current_dim);
            assert_eq!(seed.lower_b.len(), seed.num_specs);
            assert_eq!(seed.upper_b.len(), seed.num_specs);

            let lower_bounds = seed
                .lower_a
                .chunks_exact(seed.current_dim)
                .zip(seed.lower_b.iter())
                .map(|(row, bias)| row.iter().copied().sum::<f32>() + *bias - 0.25)
                .collect();
            let upper_bounds = seed
                .upper_a
                .chunks_exact(seed.current_dim)
                .zip(seed.upper_b.iter())
                .map(|(row, bias)| row.iter().copied().sum::<f32>() + *bias + 0.25)
                .collect();
            GpuCrownResult {
                lower_bounds,
                upper_bounds,
            }
        }
    }

    impl GemmEngine for BoundedRowsMock {
        fn gemm_f32(
            &self,
            m: usize,
            _k: usize,
            n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Ok(vec![0.0; m * n])
        }

        fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
            Some(self)
        }
    }

    impl GpuCrownBackward for BoundedRowsMock {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            Err(NyError::UnsupportedOp(
                "ordinary GPU CROWN must not run in bounded-row tests".into(),
            ))
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }

        fn provides_deadline_bounded_single_row_resnet_sound(&self) -> bool {
            self.capacity >= 1
        }

        fn deadline_bounded_resnet_sound_max_rows(&self) -> usize {
            self.capacity
        }

        fn crown_backward_gpu_resnet_sound_single_row_with_deadline(
            &self,
            segments: &[GpuResnetSegment],
            seed: &GpuCrownSeed,
            input_lower: &[f32],
            input_upper: &[f32],
            _frontier_abs: &[Vec<f32>],
            _node_abs: &[Vec<f32>],
            deadline: Instant,
        ) -> Result<GpuCrownResult> {
            assert!(Instant::now() < deadline);
            assert!(!segments.is_empty());
            assert_eq!(seed.num_specs, 1);
            assert_eq!(input_lower.len(), input_upper.len());
            self.single_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Self::evaluate_seed(seed))
        }

        fn crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
            &self,
            segments: &[GpuResnetSegment],
            seed: &GpuCrownSeed,
            input_lower: &[f32],
            input_upper: &[f32],
            _frontier_abs: &[Vec<f32>],
            _node_abs: &[Vec<f32>],
            deadline: Instant,
        ) -> Result<GpuCrownResult> {
            assert!(Instant::now() < deadline);
            assert!(!segments.is_empty());
            assert!((2..=self.capacity).contains(&seed.num_specs));
            assert!(seed.num_specs <= DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS);
            assert_eq!(input_lower.len(), input_upper.len());
            self.bounded_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Self::evaluate_seed(seed))
        }
    }

    fn bounded_rows_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(lin("l0", NETWORK_INPUT));
        graph.add_node(relu("relu", "l0"));
        graph.add_node(lin("residual", "relu"));
        graph.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["residual".to_string(), "l0".to_string()],
        ));
        graph.add_node(lin("out", "add"));
        graph.set_output("out");
        graph
    }

    fn bounded_rows_spec(rows: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, 2), |(row, col)| {
            let row = row as f32;
            if col == 0 {
                row + 1.0
            } else {
                row.mul_add(-0.125, 0.5)
            }
        })
    }

    fn assert_bounded_rows_serial_parity(rows: usize) {
        let graph = bounded_rows_graph();
        let input = input_box();
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let alpha_state = GraphAlphaState::new();
        let skeleton = build_resnet_segment_skeleton(
            &graph,
            &input,
            "out",
            &node_bounds,
            &node_bounds,
            Some(&alpha_state),
            false,
        )
        .expect("test resnet must produce a reusable skeleton");
        let spec = bounded_rows_spec(rows);
        let engine = BoundedRowsMock::new(DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS);
        let deadline = Instant::now() + Duration::from_secs(10);

        let result = SpecCrownRequest::new(&graph, &input, &spec, Some(&engine))
            .node_bounds(&node_bounds)
            .alpha_state_opt(Some(&alpha_state))
            .deadline_opt(Some(deadline))
            .run_alpha_sound_gpu_bounded_rows_only(&skeleton)
            .expect("bounded-row request")
            .expect("eligible bounded-row request must dispatch");

        assert_eq!(engine.bounded_calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            engine.single_calls.load(Ordering::Relaxed),
            0,
            "the request itself must dispatch one K-row transaction, not K scalar calls"
        );
        assert!(!result.segments.is_empty());
        assert!(!result.relu_names.is_empty());

        let batched_lower: Vec<f32> = result.bounds.lower().iter().copied().collect();
        let batched_upper: Vec<f32> = result.bounds.upper().iter().copied().collect();
        for row in 0..rows {
            let coefficients: Vec<f32> = spec.row(row).iter().copied().collect();
            let scalar_seed = GpuCrownSeed {
                lower_a: Arc::from(coefficients.clone()),
                upper_a: Arc::from(coefficients),
                lower_b: Arc::from([0.0]),
                upper_b: Arc::from([0.0]),
                num_specs: 1,
                current_dim: spec.ncols(),
            };
            let scalar = engine
                .crown_backward_gpu_resnet_sound_single_row_with_deadline(
                    &result.segments,
                    &scalar_seed,
                    &result.input_lower,
                    &result.input_upper,
                    &[],
                    &[],
                    deadline,
                )
                .expect("serial one-row oracle");
            assert_eq!(
                batched_lower[row].to_bits(),
                scalar.lower_bounds[0].to_bits(),
                "lower row {row} differs from serial one-row dispatch"
            );
            assert_eq!(
                batched_upper[row].to_bits(),
                scalar.upper_bounds[0].to_bits(),
                "upper row {row} differs from serial one-row dispatch"
            );
        }
        assert_eq!(engine.bounded_calls.load(Ordering::Relaxed), 1);
        assert_eq!(engine.single_calls.load(Ordering::Relaxed), rows);
    }

    #[test]
    fn deadline_bounded_rows_k2_is_one_dispatch_and_matches_serial_rows() {
        assert_bounded_rows_serial_parity(2);
    }

    #[test]
    fn deadline_bounded_rows_k8_is_one_dispatch_and_matches_serial_rows() {
        assert_bounded_rows_serial_parity(DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS);
    }

    #[test]
    fn deadline_bounded_rows_refuses_invalid_k_without_dispatch() {
        let graph = bounded_rows_graph();
        let input = input_box();
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let alpha_state = GraphAlphaState::new();
        let skeleton = build_resnet_segment_skeleton(
            &graph,
            &input,
            "out",
            &node_bounds,
            &node_bounds,
            Some(&alpha_state),
            false,
        )
        .expect("test resnet skeleton");

        for rows in [0, 1, DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS + 1] {
            let spec = bounded_rows_spec(rows);
            let engine = BoundedRowsMock::new(DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS);
            let result = SpecCrownRequest::new(&graph, &input, &spec, Some(&engine))
                .node_bounds(&node_bounds)
                .alpha_state_opt(Some(&alpha_state))
                .deadline_opt(Some(Instant::now() + Duration::from_secs(10)))
                .run_alpha_sound_gpu_bounded_rows_only(&skeleton)
                .expect("invalid K must fail closed");
            assert!(result.is_none(), "K={rows} must be refused");
            assert_eq!(engine.bounded_calls.load(Ordering::Relaxed), 0);
            assert_eq!(engine.single_calls.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn deadline_bounded_rows_refuses_insufficient_or_invalid_capacity() {
        let graph = bounded_rows_graph();
        let input = input_box();
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let alpha_state = GraphAlphaState::new();
        let skeleton = build_resnet_segment_skeleton(
            &graph,
            &input,
            "out",
            &node_bounds,
            &node_bounds,
            Some(&alpha_state),
            false,
        )
        .expect("test resnet skeleton");

        for (rows, capacity) in [
            (2, 1),
            (DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS, 7),
            (2, DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS + 1),
        ] {
            let spec = bounded_rows_spec(rows);
            let engine = BoundedRowsMock::new(capacity);
            let result = SpecCrownRequest::new(&graph, &input, &spec, Some(&engine))
                .node_bounds(&node_bounds)
                .alpha_state_opt(Some(&alpha_state))
                .deadline_opt(Some(Instant::now() + Duration::from_secs(10)))
                .run_alpha_sound_gpu_bounded_rows_only(&skeleton)
                .expect("invalid capacity must fail closed");
            assert!(
                result.is_none(),
                "K={rows}, advertised capacity={capacity} must be refused"
            );
            assert_eq!(engine.bounded_calls.load(Ordering::Relaxed), 0);
            assert_eq!(engine.single_calls.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn deadline_bounded_rows_refuses_wrong_skeleton_without_dispatch() {
        let graph = bounded_rows_graph();
        let input = input_box();
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let alpha_state = GraphAlphaState::new();
        let wrong_skeleton = build_resnet_segment_skeleton(
            &graph,
            &input,
            "add",
            &node_bounds,
            &node_bounds,
            Some(&alpha_state),
            false,
        )
        .expect("alternate target still has a valid resnet skeleton");
        let spec = bounded_rows_spec(2);
        let engine = BoundedRowsMock::new(DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS);

        let result = SpecCrownRequest::new(&graph, &input, &spec, Some(&engine))
            .node_bounds(&node_bounds)
            .alpha_state_opt(Some(&alpha_state))
            .deadline_opt(Some(Instant::now() + Duration::from_secs(10)))
            .run_alpha_sound_gpu_bounded_rows_only(&wrong_skeleton)
            .expect("skeleton mismatch must fail closed");
        assert!(result.is_none());
        assert_eq!(engine.bounded_calls.load(Ordering::Relaxed), 0);
        assert_eq!(engine.single_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn deadline_bounded_rows_refuses_expired_deadline_without_dispatch() {
        let graph = bounded_rows_graph();
        let input = input_box();
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let alpha_state = GraphAlphaState::new();
        let skeleton = build_resnet_segment_skeleton(
            &graph,
            &input,
            "out",
            &node_bounds,
            &node_bounds,
            Some(&alpha_state),
            false,
        )
        .expect("test resnet skeleton");
        let spec = bounded_rows_spec(2);
        let engine = BoundedRowsMock::new(DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS);
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("Instant supports a one-second subtraction");

        let result = SpecCrownRequest::new(&graph, &input, &spec, Some(&engine))
            .node_bounds(&node_bounds)
            .alpha_state_opt(Some(&alpha_state))
            .deadline_opt(Some(expired))
            .run_alpha_sound_gpu_bounded_rows_only(&skeleton)
            .expect("expired deadline must fail closed");
        assert!(result.is_none());
        assert_eq!(engine.bounded_calls.load(Ordering::Relaxed), 0);
        assert_eq!(engine.single_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn deadline_bounded_rows_dispatch_neither_reads_nor_charges_ordinary_budget() {
        let ordinary_elapsed = AtomicU64::new(29_999_999);
        let dispatch = ResnetGpuDispatch::DeadlineBoundedRows(Instant::now());

        assert!(resnet_gpu_dispatch_budget_allows_start_with(
            dispatch,
            &ordinary_elapsed,
            0
        ));
        assert_eq!(
            account_resnet_gpu_elapsed_with(&ordinary_elapsed, dispatch, 10_000_000),
            None
        );
        assert_eq!(ordinary_elapsed.load(Ordering::Relaxed), 29_999_999);
    }

    #[test]
    fn deadline_bounded_rows_result_refuses_malformed_intervals_without_repair() {
        let dispatch = ResnetGpuDispatch::DeadlineBoundedRows(Instant::now());
        for malformed in [
            GpuCrownResult {
                lower_bounds: vec![0.0],
                upper_bounds: vec![1.0, 2.0],
            },
            GpuCrownResult {
                lower_bounds: vec![0.0, 5.0],
                upper_bounds: vec![1.0, 4.0],
            },
            GpuCrownResult {
                lower_bounds: vec![0.0, f32::NEG_INFINITY],
                upper_bounds: vec![1.0, 2.0],
            },
        ] {
            assert!(resnet_gpu_result_to_bounds(dispatch, 2, malformed)
                .expect("malformed backend output must fail closed")
                .is_none());
        }
    }

    #[test]
    fn deadline_single_row_dispatch_neither_reads_nor_charges_ordinary_budget() {
        let ordinary_elapsed = AtomicU64::new(29_999_999);
        let dispatch = ResnetGpuDispatch::DeadlineBoundedSingleRow(Instant::now());

        assert!(
            resnet_gpu_dispatch_budget_allows_start_with(dispatch, &ordinary_elapsed, 0),
            "ordinary budget disablement must not admit or refuse the private-deadline route"
        );
        assert_eq!(
            account_resnet_gpu_elapsed_with(&ordinary_elapsed, dispatch, 10_000_000),
            None,
            "the private-deadline route must not expose an ordinary cumulative value"
        );
        assert_eq!(
            ordinary_elapsed.load(Ordering::Relaxed),
            29_999_999,
            "the private-deadline route must leave ordinary accounting byte-for-byte unchanged"
        );
    }

    #[test]
    fn ordinary_dispatch_budget_and_accounting_semantics_are_unchanged() {
        let ordinary_elapsed = AtomicU64::new(29_999_999);
        let dispatch = ResnetGpuDispatch::Ordinary;

        assert!(
            resnet_gpu_dispatch_budget_allows_start_with(dispatch, &ordinary_elapsed, 30_000),
            "the historical floor-to-milliseconds check admits just below the ceiling"
        );
        assert!(
            !resnet_gpu_dispatch_budget_allows_start_with(dispatch, &ordinary_elapsed, 0),
            "a zero ordinary budget must continue to disable the ordinary route"
        );
        assert_eq!(
            account_resnet_gpu_elapsed_with(&ordinary_elapsed, dispatch, 1),
            Some(30_000_000),
            "ordinary elapsed time must continue to return the updated cumulative value"
        );
        assert_eq!(ordinary_elapsed.load(Ordering::Relaxed), 30_000_000);
        assert!(
            !resnet_gpu_dispatch_budget_allows_start_with(dispatch, &ordinary_elapsed, 30_000),
            "the ordinary route must continue to refuse starts at the exact ceiling"
        );
    }

    #[test]
    fn deadline_single_row_result_refuses_inverted_pair_without_publication() {
        let dispatch = ResnetGpuDispatch::DeadlineBoundedSingleRow(Instant::now());
        let malicious = GpuCrownResult {
            lower_bounds: vec![5.0],
            upper_bounds: vec![4.0],
        };

        assert!(
            resnet_gpu_result_to_bounds(dispatch, 1, malicious)
                .expect("malformed backend output must fail closed, not error outward")
                .is_none(),
            "strict candidate conversion must not repair and publish an inverted raw pair"
        );
    }

    #[test]
    fn deadline_single_row_result_refuses_non_finite_pairs_without_publication() {
        let dispatch = ResnetGpuDispatch::DeadlineBoundedSingleRow(Instant::now());
        for (lower, upper) in [
            (f32::NAN, 1.0),
            (0.0, f32::NAN),
            (f32::NEG_INFINITY, 1.0),
            (0.0, f32::INFINITY),
        ] {
            let malicious = GpuCrownResult {
                lower_bounds: vec![lower],
                upper_bounds: vec![upper],
            };
            assert!(
                resnet_gpu_result_to_bounds(dispatch, 1, malicious)
                    .expect("malformed backend output must fail closed, not error outward")
                    .is_none(),
                "strict candidate conversion published non-finite pair [{lower}, {upper}]"
            );
        }
    }

    #[test]
    fn deadline_single_row_result_refuses_wrong_shape_without_publication() {
        let dispatch = ResnetGpuDispatch::DeadlineBoundedSingleRow(Instant::now());
        for malicious in [
            GpuCrownResult {
                lower_bounds: vec![],
                upper_bounds: vec![1.0],
            },
            GpuCrownResult {
                lower_bounds: vec![0.0],
                upper_bounds: vec![1.0, 2.0],
            },
            GpuCrownResult {
                lower_bounds: vec![0.0, 1.0],
                upper_bounds: vec![1.0, 2.0],
            },
        ] {
            assert!(
                resnet_gpu_result_to_bounds(dispatch, 1, malicious)
                    .expect("malformed backend output must fail closed, not error outward")
                    .is_none(),
                "strict candidate conversion must not publish a non-scalar raw result"
            );
        }
    }

    #[test]
    fn result_conversion_rejects_malformed_payloads_for_every_dispatch() {
        let valid = GpuCrownResult {
            lower_bounds: vec![-0.25],
            upper_bounds: vec![1.0],
        };
        let strict = resnet_gpu_result_to_bounds(
            ResnetGpuDispatch::DeadlineBoundedSingleRow(Instant::now()),
            1,
            valid,
        )
        .expect("valid strict result")
        .expect("valid strict result must publish");
        assert_eq!(strict.lower()[[0]], -0.25);
        assert_eq!(strict.upper()[[0]], 1.0);

        let ordinary_refusal = resnet_gpu_result_to_bounds(
            ResnetGpuDispatch::Ordinary,
            1,
            GpuCrownResult {
                lower_bounds: vec![5.0],
                upper_bounds: vec![4.0],
            },
        )
        .expect("ordinary malformed conversion must fail closed");
        assert!(
            ordinary_refusal.is_none(),
            "ordinary GPU refusal must preserve the caller's CPU fallback"
        );
    }

    #[test]
    fn env_gate_default_on_semantics() {
        // Unset / empty / any non-"0" value → enabled; only exactly "0" opts out.
        // (Pure predicate — the racy env-var read is a thin wrapper around this.)
        assert!(env_gate_default_on(None), "unset must default ON");
        assert!(!env_gate_default_on(Some("0")), "\"0\" must opt out");
        assert!(env_gate_default_on(Some("1")), "\"1\" stays enabled");
        assert!(env_gate_default_on(Some("")), "empty string stays enabled");
        assert!(
            env_gate_default_on(Some("off")),
            "non-\"0\" text stays enabled"
        );
    }

    #[test]
    fn identity_skip_block_is_recognized() {
        // input → l1 → relu1 → l2 → add(l2, l1)   (z = l1, identity skip)
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2", "relu1"));
        g.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["l2".to_string(), "l1".to_string()],
        ));
        g.set_output("add");
        let input = input_box();

        let segs = decompose(&g, &input, "add").expect("identity skip should decompose");
        assert_eq!(segs.len(), 2, "expected [Residual(F), Chain(l1)]");
        assert!(
            matches!(segs[0], GpuResnetSegment::Residual(_)),
            "first segment must be the identity-skip Residual block"
        );
        assert_eq!(chain_len(&segs[0]), 2, "F = [l2, relu1]");
        assert!(matches!(segs[1], GpuResnetSegment::Chain(_)));
        assert_eq!(chain_len(&segs[1]), 1, "trailing chain = [l1]");
    }

    #[test]
    fn projection_skip_block_is_recognized() {
        // input → l1 → relu1 → {l2a, l2b} → add(l2a, l2b)  (z = relu1, projection)
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2a", "relu1"));
        g.add_node(lin("l2b", "relu1"));
        g.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["l2a".to_string(), "l2b".to_string()],
        ));
        g.set_output("add");
        let input = input_box();

        let segs = decompose(&g, &input, "add").expect("projection skip should decompose");
        assert_eq!(
            segs.len(),
            2,
            "expected [ResidualProj(F,P), Chain(relu1,l1)]"
        );
        assert!(
            matches!(segs[0], GpuResnetSegment::ResidualProj(_, _)),
            "first segment must be the projection ResidualProj block"
        );
        assert_eq!(chain_len(&segs[0]), 2, "F=[l2a], P=[l2b]");
        assert!(matches!(segs[1], GpuResnetSegment::Chain(_)));
        assert_eq!(chain_len(&segs[1]), 2, "trailing chain = [relu1, l1]");
    }

    #[test]
    fn stacked_identity_blocks_compose() {
        // input → l0 → block1 → block2 → lout
        //  block1: l0 → relu1 → l1a → add1(l1a, l0)        (z=l0)
        //  block2: add1 → relu2 → l2a → add2(l2a, add1)    (z=add1)
        let mut g = GraphNetwork::new();
        g.add_node(lin("l0", NETWORK_INPUT));
        g.add_node(relu("relu1", "l0"));
        g.add_node(lin("l1a", "relu1"));
        g.add_node(GraphNode::new(
            "add1",
            Layer::Add(AddLayer),
            vec!["l1a".to_string(), "l0".to_string()],
        ));
        g.add_node(relu("relu2", "add1"));
        g.add_node(lin("l2a", "relu2"));
        g.add_node(GraphNode::new(
            "add2",
            Layer::Add(AddLayer),
            vec!["l2a".to_string(), "add1".to_string()],
        ));
        g.add_node(lin("lout", "add2"));
        g.set_output("lout");
        let input = input_box();

        let segs = decompose(&g, &input, "lout").expect("stacked resnet should decompose");
        // [Chain(lout), Residual(block2 F), Residual(block1 F), Chain(l0)]
        assert_eq!(segs.len(), 4, "got {} segments", segs.len());
        assert!(matches!(segs[0], GpuResnetSegment::Chain(_)));
        assert!(matches!(segs[1], GpuResnetSegment::Residual(_)));
        assert!(matches!(segs[2], GpuResnetSegment::Residual(_)));
        assert!(matches!(segs[3], GpuResnetSegment::Chain(_)));
        assert_eq!(chain_len(&segs[0]), 1, "Chain=[lout]");
        assert_eq!(chain_len(&segs[1]), 2, "block2 F=[l2a, relu2]");
        assert_eq!(chain_len(&segs[2]), 2, "block1 F=[l1a, relu1]");
        assert_eq!(chain_len(&segs[3]), 1, "Chain=[l0]");
    }

    #[test]
    fn fold_order_relu_names_match_segment_order() {
        // Stacked identity blocks: segments [Chain(lout), Residual(block2 F=[l2a,relu2]),
        // Residual(block1 F=[l1a,relu1]), Chain(l0)]. The resident fold captures ReLU
        // gradients in segment/branch order → fold-order names = [relu2, relu1].
        let mut g = GraphNetwork::new();
        g.add_node(lin("l0", NETWORK_INPUT));
        g.add_node(relu("relu1", "l0"));
        g.add_node(lin("l1a", "relu1"));
        g.add_node(GraphNode::new(
            "add1",
            Layer::Add(AddLayer),
            vec!["l1a".to_string(), "l0".to_string()],
        ));
        g.add_node(relu("relu2", "add1"));
        g.add_node(lin("l2a", "relu2"));
        g.add_node(GraphNode::new(
            "add2",
            Layer::Add(AddLayer),
            vec!["l2a".to_string(), "add1".to_string()],
        ));
        g.add_node(lin("lout", "add2"));
        g.set_output("lout");
        let input = input_box();

        let (segs, names) = decompose_names(&g, &input, "lout").expect("decompose with names");
        assert_eq!(segs.len(), 4);
        assert_eq!(
            names,
            vec!["relu2".to_string(), "relu1".to_string()],
            "fold-order ReLU names must follow segment/branch order"
        );
    }

    #[test]
    fn projection_block_fold_order_is_f_then_p() {
        // input → l1 → relu1 → {l2a→reluF, l2b→reluP} → add  (z = relu1, projection).
        // Fold order: F-branch ReLU (reluF) before P-branch ReLU (reluP), then trailing
        // chain's relu1.
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2a", "relu1"));
        g.add_node(relu("reluF", "l2a"));
        g.add_node(lin("l2b", "relu1"));
        g.add_node(relu("reluP", "l2b"));
        g.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["reluF".to_string(), "reluP".to_string()],
        ));
        g.set_output("add");
        let input = input_box();

        let (_segs, names) = decompose_names(&g, &input, "add").expect("decompose proj with names");
        assert_eq!(
            names,
            vec![
                "reluF".to_string(),
                "reluP".to_string(),
                "relu1".to_string()
            ],
            "projection fold order: F-branch ReLU, then P-branch ReLU, then trailing chain"
        );
    }

    #[test]
    fn pure_unary_suffix_is_left_to_unary_path() {
        // No residual → None (the existing unary GPU-suffix path handles it).
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2", "relu1"));
        g.set_output("l2");
        let input = input_box();
        assert!(
            decompose(&g, &input, "l2").is_none(),
            "pure unary suffix must return None (no residual)"
        );
    }

    #[test]
    fn pure_chain_suffix_decomposes_when_allowed() {
        // #metaroom-chain-wide: the SAME pure unary suffix decomposes to ONE
        // Chain segment (with fold-order ReLU names + per-segment frontier +
        // per-ReLU node_abs tables) when `allow_pure_chain = true` — the BaB
        // batched β lane's extraction. `false` stays None (previous test).
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2", "relu1"));
        g.set_output("l2");
        let input = input_box();
        let bounds = g.collect_node_bounds(&input).expect("node bounds");
        let (segs, names, frontier, node_abs, _stop) = extract_gpu_segments_with_relu_names_ext(
            &g, &input, "l2", &bounds, &bounds, None, true, false, false,
        )
        .expect("pure chain must decompose with allow_pure_chain");
        assert_eq!(segs.len(), 1, "one Chain segment");
        assert!(matches!(segs[0], GpuResnetSegment::Chain(_)));
        assert_eq!(chain_len(&segs[0]), 3, "Chain=[l2, relu1, l1]");
        assert_eq!(names, vec!["relu1".to_string()], "fold-order ReLU names");
        assert_eq!(frontier.len(), 1, "one frontier entry per segment");
        assert_eq!(node_abs.len(), 1, "one node_abs entry per ReLU");
        // The false-flag ext call must match the resnet-only entry byte-identically.
        assert!(
            extract_gpu_segments_with_relu_names_ext(
                &g, &input, "l2", &bounds, &bounds, None, false, false, false,
            )
            .is_none(),
            "allow_pure_chain=false keeps the >=1-residual refusal"
        );
    }

    #[test]
    fn bab_chain_wide_gate_is_opt_in() {
        // Dark by default: only exactly "1" enables (mirror of the batched gate).
        // Racy env mutation is avoided — this only checks the unset default.
        if std::env::var("NY_BAB_CHAIN_WIDE").is_err() {
            assert!(!bab_chain_wide_enabled(), "must default OFF");
        }
    }

    #[test]
    fn sub_merge_bails_to_cpu() {
        // A Sub merge (negated second branch) is NOT modeled → None (CPU fallback).
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2", "relu1"));
        g.add_node(GraphNode::new(
            "sub",
            Layer::Sub(SubLayer),
            vec!["l2".to_string(), "l1".to_string()],
        ));
        g.set_output("sub");
        let input = input_box();
        assert!(
            decompose(&g, &input, "sub").is_none(),
            "Sub merge must bail to CPU (only Add is modeled)"
        );
    }

    #[test]
    fn nested_residual_inside_branch_bails() {
        // The F branch of the outer Add itself contains an inner Add → not a pure
        // unary branch → bail (conservative; CPU handles it soundly).
        // input → l0 → relu1 → l1a → addInner(l1a, l0)   (inner block, z=l0)
        //        addInner → l3 → addOuter(l3, l0)
        // addOuter's F branch (l3 → addInner) hits a multi-input node → bail.
        let mut g = GraphNetwork::new();
        g.add_node(lin("l0", NETWORK_INPUT));
        g.add_node(relu("relu1", "l0"));
        g.add_node(lin("l1a", "relu1"));
        g.add_node(GraphNode::new(
            "addInner",
            Layer::Add(AddLayer),
            vec!["l1a".to_string(), "l0".to_string()],
        ));
        g.add_node(lin("l3", "addInner"));
        g.add_node(GraphNode::new(
            "addOuter",
            Layer::Add(AddLayer),
            vec!["l3".to_string(), "l0".to_string()],
        ));
        g.set_output("addOuter");
        let input = input_box();
        // addOuter: z = common(l3, l0) = l0 (l0 ∈ anc(l3) via addInner). F = path(l3 → l0)
        // must be pure unary, but l3's input addInner is multi-input → extract bails → None.
        assert!(
            decompose(&g, &input, "addOuter").is_none(),
            "nested residual inside a branch must bail to CPU"
        );
    }
}
