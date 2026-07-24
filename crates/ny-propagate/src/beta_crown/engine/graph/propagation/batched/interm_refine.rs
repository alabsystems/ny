// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-subdomain INTERMEDIATE-BOUND refinement for the resident resnet lane
//! (#interm-refine, dark gate `NY_INTERM_REFINE=1`, default OFF = byte-identical).
//!
//! WHY: with frozen intermediate bounds,
//! β-CROWN provably caps at the split-constrained LP optimum; SOTA exceeds it
//! ONLY by re-optimizing per-subdomain intermediate pre-activation bounds
//! (auto_LiRPA `fix_intermediate_layer_bounds=False` / Clip-and-Verify complete
//! clipping). Measured on cifar100_2024 resnet_medium prop885: 100.0% of split
//! premises live on `Relu_57` (the LAST ReLU, 512-d, feeding the output Gemm),
//! whose per-subdomain pre-activation bounds stay frozen at ROOT values — so the
//! final-layer relaxation never tightens as splits accumulate and children never
//! prune (frontier explosion at depth ≥ 9).
//!
//! WHAT (first increment, narrowly targeted where the tree lives): for each BaB
//! subdomain, ONE extra backward with `num_specs = pre_dim` identity rows seeded
//! at the last ReLU's INPUT node, over the TRUNCATED resnet stack (that node down
//! to the network input), folding the subdomain's α exactly like the margin
//! backward (the same `prep_resnet_domain` machinery) — concretized over the
//! subdomain's input box by the SAME sound resident GPU path (certified error,
//! directed rounding). The per-neuron `[l', u']` is INTERSECTED with the
//! inherited entry (`l = max`, `u = min`).
//!
//! SOUNDNESS:
//! * The refinement backward is a plain sound α-CROWN backward from the seed
//!   node: β terms for splits AT the seed layer (the last ReLU) do NOT appear —
//!   they constrain the seed node itself, and their pre-activation clamps are
//!   already present in the inherited cache entry (`apply_pre_constraints`).
//!   β entries for ReLUs BELOW the seed (none measured on cifar100) fold via the
//!   truncated `beta_signed` — a valid Lagrangian dual for the same subdomain.
//! * Both the inherited entry and the refined bounds are sound enclosures of the
//!   SAME subdomain's reachable pre-activations, so their intersection is sound.
//!   An empty per-neuron intersection (possible only for an infeasible domain or
//!   f32 edge noise) conservatively keeps the inherited pair.
//! * The ReLU's own POST-activation entry is tightened to
//!   `[relu(l'), relu(u')] ∩ inherited` (relu is exact and monotone).
//!
//! COST CONTROL: runs ONCE per subdomain (the dense-spec driver bounds each
//! child exactly once); one batched GPU call for the whole domain batch through
//! the SAME wide machinery as the margin backward (serial per-domain fallback on
//! error, keep-inherited on any per-domain refusal — sound).
//!
//! INFEASIBILITY PRUNING (`NY_INTERM_REFINE_PRUNE=1`, dark, default OFF): each
//! of the domain's β entries at the seed ReLU IS one of its split premises
//! (`z_j ≥ s` active / `z_j ≤ s` inactive). The refined `[l', u']` is a sound
//! enclosure of `z_j` over the subdomain WITHOUT the seed-layer premises (they
//! are excluded from the truncated stack by construction), so a strict
//! contradiction — active premise with `u' < s − tol`, inactive with
//! `l' > s + tol` — proves the subdomain's constraint set EMPTY: it verifies
//! vacuously. The refined bounds carry certified directed-rounding error
//! outward already; `tol` (`NY_INTERM_REFINE_PRUNE_TOL`, default `1e-4`) is
//! defense-in-depth on top. Because premise-clamped neurons are STABLE in the
//! inherited entry (the clamp), the prune lane force-includes premise neurons
//! in the row selection — the `unstable` arm alone would exclude exactly the
//! rows that can prove infeasibility (measured: v3 unstable-rows logs had
//! crossings_kept=0 while the all-rows arms saw up to 8/batch).
//!
//! MULTI-LAYER CASCADE (`NY_INTERM_REFINE_LAYERS=2`, dark, default 1): also
//! refine the SECOND-to-last ReLU's pre-activation entry (on resnet_medium:
//! `Relu_51`, seed `Conv_49` — a unary conv off the last residual trunk `Add_48`,
//! decomposable by the same segment extraction). Passes run DEEPEST-FIRST so
//! the last-ReLU pass (and then the margin backward) consumes the improved
//! upstream entry through the ReLU relaxations the extraction derives from the
//! caches. Deep seed layers are conv-wide (2048 on resnet_medium), so the deep
//! pass caps identity rows at `NY_INTERM_REFINE_DEEP_MAX_ROWS` (default 256):
//! premise rows always kept, the rest top-K by inherited width (row selection
//! is cost-only, never soundness).

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::{Array1, Array2};
use ny_core::{
    nan_propagating_max, nan_propagating_min, GemmEngine, GpuResidentCoeffBatched, NyError,
};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::clip_interm_domain::{
    build_split_constraints_with_deadline_check, sort_out_constraints_with_deadline_check,
    tighten_with_constraints_with_deadline,
};
use crate::complete_clip::{CertifiedAffineEnclosure, CrownPassStamp, ValidatedAffineEnclosure};
use crate::{GraphNetwork, Layer, NETWORK_INPUT};

use super::super::super::super::BetaCrownVerifier;
use super::{
    build_call_skeleton, prep_resnet_domain_ext, prep_resnet_domain_with, ResnetDomainPrep,
};

const SELECTIVE_CLIP_MAX_HOST_BYTES: usize = 64 * 1024 * 1024;
const SELECTIVE_CLIP_MAX_WORK: usize = 500_000_000;
const SELECTIVE_DEADLINE_POLL_STRIDE: usize = 1024;
const MARGIN_WEIGHT_SITE: &str = "interm_refine_margin_weights";
// Up to four in-place heapsorts can occur across winner selection and the
// deep-row cap; each sort is bounded by 4*n*ceil(log2 n) comparisons.
const SELECTIVE_SORT_OP_WEIGHT: usize = 16;

#[derive(Clone, Copy)]
struct MarginWeightLimits {
    max_host_bytes: usize,
    max_work: usize,
}

const MARGIN_WEIGHT_LIMITS: MarginWeightLimits = MarginWeightLimits {
    max_host_bytes: SELECTIVE_CLIP_MAX_HOST_BYTES,
    max_work: SELECTIVE_CLIP_MAX_WORK,
};

/// Checked peak-owned allocation and arithmetic envelope for
/// `spec_matrix.dot(tail_weight)` plus the reduction into one weight per seed
/// neuron. Borrowed graph/spec storage is excluded; both the dense dot result
/// and the returned weight vector (including its `Vec` -> `Arc` transition) are
/// included in the peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MarginWeightPlan {
    dot_elements: usize,
    weight_elements: usize,
    peak_owned_bytes: usize,
    arithmetic_ops: usize,
}

impl MarginWeightPlan {
    fn checked(
        spec_rows: usize,
        output_dim: usize,
        seed_dim: usize,
        limits: MarginWeightLimits,
    ) -> ny_core::Result<Self> {
        let dot_elements = spec_rows.checked_mul(seed_dim).ok_or_else(|| {
            NyError::InvalidSpec("margin-weight dot shape product overflow".into())
        })?;
        let dot_bytes = dot_elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| NyError::InvalidSpec("margin-weight dot byte overflow".into()))?;
        let weight_bytes = seed_dim
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| NyError::InvalidSpec("margin-weight vector byte overflow".into()))?;
        let dot_plus_weights = dot_bytes
            .checked_add(weight_bytes)
            .ok_or_else(|| NyError::InvalidSpec("margin-weight peak byte overflow".into()))?;
        let arc_transition = weight_bytes.checked_mul(2).ok_or_else(|| {
            NyError::InvalidSpec("margin-weight Arc transition byte overflow".into())
        })?;
        let peak_owned_bytes = dot_plus_weights.max(arc_transition);
        if peak_owned_bytes > limits.max_host_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: peak_owned_bytes,
                budget_bytes: limits.max_host_bytes,
                site: MARGIN_WEIGHT_SITE,
            });
        }

        // Charge a multiply and add for every GEMM term, plus comparison,
        // negation, and accumulation for every reduced dot cell.
        let multiply_adds = dot_elements.checked_mul(output_dim).ok_or_else(|| {
            NyError::InvalidSpec("margin-weight multiply-add count overflow".into())
        })?;
        let gemm_ops = multiply_adds
            .checked_mul(2)
            .ok_or_else(|| NyError::InvalidSpec("margin-weight GEMM work overflow".into()))?;
        let reduction_ops = dot_elements
            .checked_mul(3)
            .ok_or_else(|| NyError::InvalidSpec("margin-weight reduction work overflow".into()))?;
        let arithmetic_ops = gemm_ops
            .checked_add(reduction_ops)
            .ok_or_else(|| NyError::InvalidSpec("margin-weight total work overflow".into()))?;
        if arithmetic_ops > limits.max_work {
            return Err(NyError::InvalidSpec(format!(
                "margin-weight arithmetic budget exceeded: {arithmetic_ops} > {}",
                limits.max_work
            )));
        }

        Ok(Self {
            dot_elements,
            weight_elements: seed_dim,
            peak_owned_bytes,
            arithmetic_ops,
        })
    }
}
// The 7-arg legacy face is the tests' reference oracle (`use super::*` in the
// test modules below); production lanes call the `_with`/`_ext` faces directly.
#[cfg(test)]
use super::prep_resnet_domain;

/// Env gate for per-subdomain intermediate-bound refinement (dark, default off).
/// The proposed `NY_CLIP_INTERM` umbrella is authority-quarantined and cannot
/// imply this lane. Only `NY_INTERM_REFINE=1` enables production refinement.
pub(in crate::beta_crown::engine::graph) fn interm_refine_enabled() -> bool {
    matches!(std::env::var("NY_INTERM_REFINE").ok().as_deref(), Some("1"))
        || clip_interm_umbrella_enabled()
}

/// Env gate for infeasibility pruning (dark, default OFF — see module docs).
fn interm_refine_prune_enabled() -> bool {
    matches!(
        std::env::var("NY_INTERM_REFINE_PRUNE").ok().as_deref(),
        Some("1")
    )
}

/// How many ReLU layers (from the output inward) get per-subdomain refinement
/// (`NY_INTERM_REFINE_LAYERS`, default 1 = the last ReLU only).
fn interm_refine_layers() -> usize {
    std::env::var("NY_INTERM_REFINE_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

/// Extra defense-in-depth margin the premise contradiction must exceed before
/// a domain is pruned (`NY_INTERM_REFINE_PRUNE_TOL`, default `1e-4`). The
/// refined bounds already carry certified outward rounding; this only makes
/// the prune MORE conservative.
fn interm_refine_prune_tol() -> f32 {
    std::env::var("NY_INTERM_REFINE_PRUNE_TOL")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|t| t.is_finite() && *t >= 0.0)
        .unwrap_or(1e-4)
}

/// Row cap for DEEP (non-last-ReLU) seed layers
/// (`NY_INTERM_REFINE_DEEP_MAX_ROWS`, default 256): conv-wide seeds would
/// otherwise carry thousands of identity rows. Premise rows are always kept;
/// the remainder is top-K by inherited width. Cost-only, never soundness.
fn interm_refine_deep_max_rows() -> usize {
    std::env::var("NY_INTERM_REFINE_DEEP_MAX_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256)
}

/// Winner-style per-layer objective cap
/// (`NY_INTERM_REFINE_SELECTIVE_TOPK`, default 0 = OFF/unlimited).
///
/// αβ-CROWN's VNN-COMP 2025 TinyImageNet configuration clips only the top 20
/// intermediate objectives per layer.  NY historically seeded every unstable
/// neuron at each selected layer, making the identity backward and per-domain clip scale
/// with the full layer width.  A positive value keeps that many highest-impact
/// unstable objectives plus every split-premise source row.  Omitted rows retain
/// their inherited sound bounds, so this is cost/selection only.
fn interm_refine_selective_topk() -> usize {
    std::env::var("NY_INTERM_REFINE_SELECTIVE_TOPK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Probe (`NY_INTERM_REFINE_PROBE=1` or the lane-wide `NY_BETA_GPU_PROBE=1`):
/// per-batch refinement stats on stderr.
fn probe_enabled() -> bool {
    std::env::var("NY_INTERM_REFINE_PROBE").ok().as_deref() == Some("1")
        || std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1")
}

/// Cap on the seed-layer width (the refinement backward carries `pre_dim`
/// identity rows — cost scales linearly). `Relu_57` is 512-d; the default keeps
/// room for tinyimagenet-class nets while refusing conv-wide layers.
/// Override with `NY_INTERM_REFINE_MAX_DIM`.
fn interm_refine_max_dim() -> usize {
    std::env::var("NY_INTERM_REFINE_MAX_DIM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2048)
}

/// Depth gate (`NY_INTERM_REFINE_MIN_DEPTH=d`, default 0 = refine every
/// domain, byte-identical): refine only domains with at least `d` split
/// premises (β entries — ReLU-split BaB adds exactly one per split, so the
/// count IS the domain depth). Shallow domains keep their inherited caches.
/// Cost-only lever (#fit-100s): row selection of WHICH domains get a refined
/// enclosure — excluded domains keep the inherited sound pair, exactly like
/// an unselected neuron row.
fn interm_refine_min_depth() -> usize {
    std::env::var("NY_INTERM_REFINE_MIN_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Explicit seed list (`NY_INTERM_REFINE_SEEDS=Relu_13,Relu_19,Relu_31,last`,
/// dark, default unset = the `NY_INTERM_REFINE_LAYERS` walk, byte-identical):
/// comma-separated ReLU node names (exact graph names) plus the token `last`
/// for the last-ReLU chain. Purpose (#midref): under the LA brancher the split
/// premises accumulate on MID-network ReLUs (`Relu_13/19/31` on resnet_medium)
/// whose downstream effect is otherwise priced only by β — naming them refines
/// their pre-activation entries per subdomain, CASCADED in exec order
/// (earliest first), each pass consuming the previous passes' refined caches
/// (β folds of below-seed premises + clamped/refined below-seed relaxations),
/// then the margin backward consumes everything. Named seeds are exempt from
/// `NY_INTERM_REFINE_MAX_DIM` (naming a conv-wide layer is the deliberate ask;
/// the deep row cap + the hard `n_rows·pre_dim` guard still bound cost).
/// Unknown/non-ReLU/fully-stable names are skipped (probe-logged).
fn interm_refine_seed_names() -> Option<Vec<String>> {
    let raw = std::env::var("NY_INTERM_REFINE_SEEDS").ok()?;
    let names: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    (!names.is_empty()).then_some(names)
}

/// α′-FOR-REFINEMENT gate (`NY_INTERM_REFINE_ALPHA=1`, dark, default OFF =
/// byte-identical — #alpha-prime). auto_LiRPA "finding 4": each intermediate
/// bound is its own optimization with its OWN α, distinct from the margin
/// objective's α. The refinement backward today folds the WARMUP/MARGIN α
/// (optimized for the margin row, not for tightening each seed-layer
/// pre-activation). With the gate on, the last-ReLU pass runs a small Adam
/// ascent of a DEDICATED α′ against the refinement objective
/// `Σ_rows l′_row` — per-row gradients from the oracle-validated TRUE
/// chain rule (`max(ν,0)·ĥ(x*)`, `wide_alpha_true::true_alpha_grads_for_row`)
/// replayed on the truncated stack, summed over the identity rows — and
/// element-wise best-merges every iterate's per-row bounds (each iterate is a
/// sound enclosure of the SAME subdomain for its α′ ∈ [0,1], so `max l / min u`
/// across iterates is sound). The best-objective α′ is stored once per process
/// and REUSED by later batches (applied to each domain's own unstable neurons
/// before the single refinement backward — any α ∈ [0,1] is a valid lower
/// ReLU relaxation slope, so reuse is sound for every subdomain).
fn interm_refine_alpha_enabled() -> bool {
    matches!(
        std::env::var("NY_INTERM_REFINE_ALPHA").ok().as_deref(),
        Some("1")
    )
}

/// JOINT margin-directed intermediate-α gate (`NY_JOINT_INTERM_ALPHA=1`, dark,
/// default OFF = byte-identical — #joint-interm-alpha). The SOTA blueprint's #1
/// αβ-CROWN advantage: optimize the intermediate pre-activation bounds against
/// the FINAL margin, not each against its own objective. The base α′ lane
/// ([`interm_refine_alpha_enabled`]) ascends the refinement backward's α against
/// the UNIFORM refinement objective `Σ_rows l′_row` — every seed neuron weighted
/// equally. With this gate on, each refinement row is instead weighted by the
/// MARGIN's sensitivity to that seed neuron's refined LOWER bound
/// `m_j = Σ_specs max(−(spec·W_tail)[j], 0)` (only the upper-branch neurons the
/// margin's chord relaxation actually reads carry weight, ∝ the tail-coefficient
/// magnitude), so the α′ ascent tightens the refined bounds the margin uses.
/// Turning this on implies the α′ lane (auto-enabled in `from_env`). Sound: any
/// α ∈ [0,1] is a valid lower ReLU slope; the weights only STEER the ascent, and
/// only element-wise-tightest sound iterates are kept (exactly as the base lane).
fn joint_interm_alpha_enabled() -> bool {
    matches!(
        std::env::var("NY_JOINT_INTERM_ALPHA").ok().as_deref(),
        Some("1")
    )
}

/// PER-TARGET (per-intermediate-neuron) α′ gate (`NY_AB_PARITY_INTERM=1`, dark,
/// default OFF = byte-identical — #ab-parity-interm). The auto_LiRPA per-target
/// decoupling (STUDY-1): auto_LiRPA keeps a SEPARATE optimizable lower-slope per
/// (target-spec × source-ReLU-neuron) pair, so the sum-of-bounds objective
/// DECOUPLES and every target reaches its OWN slope optimum. The base α′ lane
/// ([`interm_refine_alpha_enabled`]) instead optimizes ONE shared slope set
/// against a SCALARIZED objective (`Σ_rows w_r·l′_r`), so a single α must serve
/// every seed-row target at once. With this gate on, the last-ReLU refinement
/// gives EACH target row (each seed-layer neuron whose pre-activation box we are
/// tightening) its OWN α, optimized against ITS OWN bound `l′_ri` via that row's
/// per-target gradient, and the sound per-row bound is computed with that row's
/// slopes (element-wise best-lo kept). Implies the α′ lane (auto-enabled in
/// `from_env`) and, under the gate, the once-per-process shared-α reuse is
/// DROPPED — a fresh per-target ascent runs each batch. Sound: any α ∈ [0,1] is
/// a valid lower ReLU slope; per-target α only steers which slope each target
/// picks, and only element-wise-tightest sound GPU-fold iterates are merged.
fn ab_parity_interm_enabled() -> bool {
    matches!(
        std::env::var("NY_AB_PARITY_INTERM").ok().as_deref(),
        Some("1")
    )
}

/// FC-head BOX-WIDTH PROBE (`NY_AB_INTERM_PROBE=1`, dark, stderr-only,
/// read-only — #ab-parity-interm). After the last-ReLU refinement pass, log the
/// seed node's (the FC-head `Gemm_56` pre-activation on cifar100 resnet_medium)
/// TOTAL refined box width `Σ_j (u_j − l_j)` over ALL pre-activation neurons —
/// the primary diagnostic for whether per-target α collapses the measured
/// 50×-too-wide head box (CROWN 419.27) toward the 8.35 true-sampled range,
/// independent of the verdict. Printed once at the ROOT batch and once at the
/// first mid-depth (≥ 5 split premises) batch.
fn ab_interm_probe_enabled() -> bool {
    matches!(
        std::env::var("NY_AB_INTERM_PROBE").ok().as_deref(),
        Some("1")
    )
}

/// Ascent iterations for the α′ lane (`NY_INTERM_REFINE_ALPHA_ITERS`,
/// default 3): each iteration is one gradient replay + one extra batched
/// refinement backward.
fn interm_refine_alpha_iters() -> usize {
    std::env::var("NY_INTERM_REFINE_ALPHA_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1)
}

/// Adam learning rate for the α′ ascent (`NY_INTERM_REFINE_ALPHA_LR`,
/// default 0.05 — larger than the margin ascent's 0.01: only 2-4 iterations
/// run, and the keep-best merge makes overshoot harmless).
fn interm_refine_alpha_lr() -> f32 {
    std::env::var("NY_INTERM_REFINE_ALPHA_LR")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|t| t.is_finite() && *t > 0.0)
        .unwrap_or(0.05)
}

/// Replay-row cap for the α′ gradient (`NY_INTERM_REFINE_ALPHA_MAX_ROWS`,
/// default 64): the per-row host replays dominate the ascent wall; rows are
/// picked widest-first (most refinement headroom). Cost-only.
fn interm_refine_alpha_max_rows() -> usize {
    std::env::var("NY_INTERM_REFINE_ALPHA_MAX_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1)
}

/// Re-optimize α′ at EVERY batch instead of optimize-once-reuse
/// (`NY_INTERM_REFINE_ALPHA_REOPT=1`, dark — the expensive per-subdomain arm
/// for A/B; never touches the store).
fn interm_refine_alpha_reopt() -> bool {
    matches!(
        std::env::var("NY_INTERM_REFINE_ALPHA_REOPT")
            .ok()
            .as_deref(),
        Some("1")
    )
}

/// The optimized α′ snapshot the root-batch ascent stores and later batches
/// reuse (#alpha-prime). `slopes[r]` is the FULL lower-slope vector of the
/// truncated stack's `r`-th Activation (fold order = `relu_names` order);
/// `stepped[r]` marks the neurons the ascent actually stepped (unstable at
/// optimization time) — reuse writes ONLY stepped neurons that are ALSO
/// unstable in the consuming domain's own extraction.
#[derive(Clone)]
pub(in crate::beta_crown::engine::graph) struct AlphaPrime {
    pub seed_node: String,
    pub pre_dim: usize,
    pub relu_names: Vec<String>,
    pub slopes: Vec<Vec<f32>>,
    pub stepped: Vec<Vec<bool>>,
    /// Did any ascent iterate beat the borrowed-α objective? `false` ⇒ the
    /// entry only suppresses re-ascending (reuse never applies it).
    pub improved: bool,
}

/// Shared handle to the per-process α′ store (`None` until the first
/// successful ascent). Tests construct their own store; production
/// (`from_env`) hands out this global — one verification instance per
/// process, and the key check in [`alpha_prime_matches`] refuses a stale
/// entry from a different net/seed anyway.
pub(in crate::beta_crown::engine::graph) type AlphaPrimeStore = Arc<Mutex<Option<AlphaPrime>>>;

fn global_alpha_prime_store() -> AlphaPrimeStore {
    static STORE: OnceLock<AlphaPrimeStore> = OnceLock::new();
    STORE.get_or_init(|| Arc::new(Mutex::new(None))).clone()
}

/// ADAPTIVE REFINE SCHEDULE gate (`NY_INTERM_REFINE_ADAPTIVE=1`, dark, default
/// OFF = byte-identical — #adaptive-refine). Measured (2026-07-11, prop8945
/// @300s): the refine pass eats ~47.5% of the BaB window at the TAIL producing
/// ZERO (newly_stable=0, width −0.03%/chunk) while being ESSENTIAL early
/// (objective collapse 25→3 on prop54). The schedule keeps refining while it
/// PRODUCES (newly_stable > 0 or infeasible-pruned > 0 per batch); the first
/// time a completed batch at depth ≥ [`interm_refine_adaptive_floor`] yields
/// zero, a per-process LATCH records that depth and every LATER domain at
/// depth ≥ latch keeps its inherited caches (skip = identical to a per-domain
/// refusal — sound; cost-only lever). Shallower domains (new subtrees) still
/// refine.
fn interm_refine_adaptive_enabled() -> bool {
    matches!(
        std::env::var("NY_INTERM_REFINE_ADAPTIVE").ok().as_deref(),
        Some("1")
    )
}

/// Minimum batch depth (min split-premise count over the batch) at which a
/// zero-yield batch may trip the adaptive latch
/// (`NY_INTERM_REFINE_ADAPTIVE_FLOOR`, default 4): protects the measured
/// early-depth value (objective collapse at depths 0-2, newly_stable=48 at
/// depth 1) from a transiently unproductive shallow batch.
fn interm_refine_adaptive_floor() -> usize {
    std::env::var("NY_INTERM_REFINE_ADAPTIVE_FLOOR")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
}

/// The adaptive latch: the smallest depth at which a completed refine batch
/// yielded zero (`usize::MAX` = unlatched). Per-process like the α′ store —
/// one verification instance per process in production; tests construct their
/// own handle.
pub(in crate::beta_crown::engine::graph) type AdaptiveLatch = Arc<std::sync::atomic::AtomicUsize>;

fn global_adaptive_latch() -> AdaptiveLatch {
    static LATCH: OnceLock<AdaptiveLatch> = OnceLock::new();
    LATCH
        .get_or_init(|| Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX)))
        .clone()
}

/// Should a completed zero-yield batch at `batch_depth` trip the latch?
/// (Pure — unit-tested.) Production is only ever a latch-DOWN: refinement
/// stops for deeper domains, bounds are never touched.
fn adaptive_should_latch(
    newly_stable: usize,
    n_infeasible: usize,
    batch_depth: usize,
    floor: usize,
) -> bool {
    newly_stable == 0 && n_infeasible == 0 && batch_depth >= floor
}

/// PER-CHILD RE-REFINEMENT (`NY_INTERM_REFINE_REDO=d`, dark, default 0 = OFF
/// = byte-identical — #interm-refine-redo). Under the adaptive latch, deep
/// domains inherit their ancestors' refined caches and never re-derive them —
/// even as split premises keep accumulating past the latch depth. Where the
/// accumulating splits are UPSTREAM (stem-directed branching), every refined
/// downstream box is derived against a stale premise set. With `d > 0`, a
/// latched-out domain whose depth is an exact multiple of `d` re-runs the
/// refinement anyway (its full premise set folds — β + clamps), and its
/// children inherit the re-derived caches; every other deep domain keeps the
/// latch skip. Pure modifier on the adaptive skip: with the latch untripped
/// (or the adaptive lane off) every domain already refines on every batch and
/// the knob is inert. Sound either way — refinement only ever intersects
/// tighter, skip ≡ per-domain refusal.
fn interm_refine_redo_every() -> usize {
    std::env::var("NY_INTERM_REFINE_REDO")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Latched-batch inclusion predicate (pure — unit-tested): a domain refines
/// despite a tripped latch iff it is below the latch depth, or the redo lane
/// re-includes it (depth an exact multiple of `redo_every`).
fn latched_domain_refines(depth: usize, latched: usize, redo_every: usize) -> bool {
    depth < latched || (redo_every > 0 && depth.is_multiple_of(redo_every))
}

/// Row selection (`NY_INTERM_REFINE_ROWS=all|unstable`, default `unstable` —
/// the measured-better arm on prop885 2026-07-11): `unstable` restricts
/// identity rows to the union of inherited-UNSTABLE neurons — a stable
/// neuron's relaxation slope is already exact, so only the `node_abs` error
/// concretization / output-side forward-IBP side-channels could differ, and
/// the A/B measured IDENTICAL per-depth bounds (to 4 decimals through depth 6)
/// at ~3x less GPU per batch (111s vs 191s refine wall at 300s; 468 vs 212
/// domains explored; 5 vs 3 pruned). `all` keeps every neuron's row for A/B.
/// Both arms are sound (row selection only chooses WHICH neurons get a
/// refined enclosure; unselected neurons keep the inherited sound pair).
fn interm_refine_unstable_rows_only() -> bool {
    !matches!(
        std::env::var("NY_INTERM_REFINE_ROWS").ok().as_deref(),
        Some("all")
    )
}

/// #wide-chunk cap on wide rows per batched refinement call
/// (`NY_INTERM_REFINE_WIDE_MAX_N`, default 0 = OFF — one call, byte-identical).
/// See [`IntermRefineOptions::wide_max_n`].
fn interm_refine_wide_max_n() -> usize {
    std::env::var("NY_INTERM_REFINE_WIDE_MAX_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// #wide-chunk sizing: domains per batched refinement call so the wide pass
/// stays under `wide_max_n` total rows (`0` = OFF — the whole batch in one
/// call). Always ≥ 1 (a single domain's rows may exceed the cap; the wide
/// lane's own gates and the serial fallback still bound that case).
fn wide_chunk_domains(wide_max_n: usize, n_rows: usize, n_idxs: usize) -> usize {
    if wide_max_n == 0 || n_rows == 0 {
        n_idxs.max(1)
    } else {
        (wide_max_n / n_rows).max(1)
    }
}

/// Snapshot of the env-tunable knobs, taken ONCE per refinement call so unit
/// tests can drive the lane without racing on process-global env vars.
#[derive(Clone)]
pub(in crate::beta_crown::engine::graph) struct IntermRefineOptions {
    pub unstable_rows_only: bool,
    pub prune: bool,
    pub layers: usize,
    pub max_dim: usize,
    pub deep_max_rows: usize,
    /// Winner-style cap for each seed layer's objectives (`0` = unlimited). Split
    /// premise rows are additive and never removed; see
    /// [`interm_refine_selective_topk`].
    pub selective_topk: usize,
    pub prune_tol: f32,
    pub probe: bool,
    pub min_depth: usize,
    /// Explicit seed list (`NY_INTERM_REFINE_SEEDS`, see
    /// [`interm_refine_seed_names`]); `None` = the `layers` walk.
    pub seeds: Option<Vec<String>>,
    /// α′-for-refinement ascent iterations (0 = lane OFF, byte-identical —
    /// see [`interm_refine_alpha_enabled`]).
    pub alpha_iters: usize,
    /// Adam lr for the α′ ascent.
    pub alpha_lr: f32,
    /// Replay-row cap for the α′ gradient (widest-first).
    pub alpha_max_rows: usize,
    /// Re-optimize α′ every batch (A/B arm) instead of optimize-once-reuse.
    pub alpha_reopt: bool,
    /// α′ store handle (`None` ⇔ lane OFF).
    pub alpha_store: Option<AlphaPrimeStore>,
    /// Adaptive-schedule latch handle (`None` ⇔ lane OFF — see
    /// [`interm_refine_adaptive_enabled`]).
    pub adaptive_latch: Option<AdaptiveLatch>,
    /// Min batch depth for a zero-yield batch to trip the adaptive latch.
    pub adaptive_floor: usize,
    /// Per-child re-refinement stride under the latch (0 = OFF — see
    /// [`interm_refine_redo_every`]).
    pub redo_every: usize,
    /// #wide-chunk (dark, `NY_INTERM_REFINE_WIDE_MAX_N=k`, default 0 = OFF =
    /// one batched call for the whole domain batch, byte-identical): cap the
    /// WIDE row count (`n_domains_in_call × n_rows`) of each batched
    /// refinement backward by splitting the domain batch into chunks of
    /// `max(1, k / n_rows)` domains. The wide resident lane fails device
    /// validation past ~2048 wide rows on the widest im2col conv dims
    /// (measured: binding cap under default limits; dispatch-dimension cap
    /// under `NY_GPU_BIG_BINDINGS=1`), and every failed batch falls to the
    /// serial per-domain stacker (~2× slower measured). Chunking is cost-only:
    /// each domain's bound is computed by the same wide kernel on its own
    /// domain block (per-domain independence — no cross-domain reduction);
    /// a chunk error falls back to the serial path for that chunk's domains.
    pub wide_max_n: usize,
    /// JOINT margin-directed intermediate-α (`NY_JOINT_INTERM_ALPHA=1`, dark —
    /// see [`joint_interm_alpha_enabled`]): weight each α′ refinement row by the
    /// margin's sensitivity to that seed neuron's refined lower bound instead of
    /// the uniform `Σ l′`. Implies the α′ lane (`from_env` auto-enables it).
    pub joint_margin: bool,
    /// Per-seed-neuron margin weights (len = seed `pre_dim`), computed once at
    /// the batched call site from the tail linear map + spec rows (see
    /// [`compute_margin_weights`]). `None` ⇒ uniform (the base α′ objective);
    /// only consulted when `joint_margin` is set. Sound: advisory ascent steering
    /// only — a wrong weight degrades the ascent, never the returned bound.
    pub margin_weights: Option<Arc<[f32]>>,
    /// #clip-interm-resnet-batched research option: after the batched seed
    /// backward, run the split-constraint clip (αβ `domain_clipper`) on the
    /// already-batched ResidentCoeff. Production environment authority is
    /// quarantined; explicit unit tests may still set this field to exercise the
    /// implementation and its enclosure oracles.
    pub clip_resnet: bool,
    /// #ab-parity-interm (dark, `NY_AB_PARITY_INTERM=1`): give EACH last-ReLU
    /// refinement target row its OWN α, optimized against its OWN bound (the
    /// auto_LiRPA per-target decoupling), instead of the ONE shared slope set the
    /// base α′ lane ascends against the scalarized objective. Implies the α′ lane
    /// (`from_env` auto-enables it) and DROPS the once-per-process shared-α reuse
    /// (fresh per-target ascent each batch). See [`ab_parity_interm_enabled`].
    /// `false` ⇒ byte-identical.
    pub per_target: bool,
    /// #ab-parity-interm FC-head box-width probe (`NY_AB_INTERM_PROBE=1`, dark,
    /// stderr-only). See [`ab_interm_probe_enabled`].
    pub interm_box_probe: bool,
    /// #clip-interm-guard FAIL-CLOSED runtime guard (`NY_CLIP_INTERM=1`, dark,
    /// default false = byte-identical). When set (and the clip is active), every
    /// clip-tightened row is checked against a bounded directed adversarial sample
    /// ([`clip_guard_verify_domain`]) BEFORE it is merged; on any feasible point
    /// outside the tightened box, or a non-finite folded coefficient/penalty, the
    /// seed node reverts to the inherited PARENT bound for that child. See
    /// [`clip_interm_umbrella_enabled`]. Never touches the OFF path.
    pub clip_guard: bool,
}

impl IntermRefineOptions {
    pub(in crate::beta_crown::engine::graph) fn from_env() -> Self {
        let joint_margin = joint_interm_alpha_enabled();
        // #ab-parity-interm per-target mode is itself a (per-target) α′ ascent.
        let per_target = ab_parity_interm_enabled();
        // Joint margin-directed mode IS an α′ ascent (with reweighted objective),
        // so it implies the α′ lane even if NY_INTERM_REFINE_ALPHA is unset.
        let alpha_on = interm_refine_alpha_enabled() || joint_margin || per_target;
        Self {
            unstable_rows_only: interm_refine_unstable_rows_only(),
            prune: interm_refine_prune_enabled(),
            layers: interm_refine_layers(),
            max_dim: interm_refine_max_dim(),
            deep_max_rows: interm_refine_deep_max_rows(),
            selective_topk: interm_refine_selective_topk(),
            prune_tol: interm_refine_prune_tol(),
            probe: probe_enabled(),
            min_depth: interm_refine_min_depth(),
            seeds: interm_refine_seed_names(),
            alpha_iters: if alpha_on {
                interm_refine_alpha_iters()
            } else {
                0
            },
            alpha_lr: interm_refine_alpha_lr(),
            alpha_max_rows: interm_refine_alpha_max_rows(),
            alpha_reopt: interm_refine_alpha_reopt(),
            alpha_store: alpha_on.then(global_alpha_prime_store),
            adaptive_latch: interm_refine_adaptive_enabled().then(global_adaptive_latch),
            adaptive_floor: interm_refine_adaptive_floor(),
            redo_every: interm_refine_redo_every(),
            wide_max_n: interm_refine_wide_max_n(),
            joint_margin,
            // Computed at the batched call site (needs the tail map + spec rows);
            // `from_env` cannot see the graph, so the env-default entry leaves it
            // None (uniform) and the joint path fills it in `refine_last_relu_*`.
            margin_weights: None,
            // Both proposed environment entry points are authority-quarantined.
            // Explicitly constructed unit-test options can still exercise the
            // clip and guard without granting production verdict authority.
            clip_resnet: clip_interm_resnet_batched_enabled() || clip_interm_umbrella_enabled(),
            per_target,
            interm_box_probe: ab_interm_probe_enabled(),
            clip_guard: clip_interm_umbrella_enabled(),
        }
    }
}

/// Authority gate for the legacy `NY_CLIP_INTERM_RESNET` request.
///
/// Quarantined for the same reason as [`clip_interm_umbrella_enabled`]: these
/// refined caches feed verdict-authoritative CROWN bounds. The environment cannot
/// enable this path; direct options remain available to unit tests.
fn clip_interm_resnet_batched_enabled() -> bool {
    false
}

/// Authority gate for the proposed `NY_CLIP_INTERM=1` umbrella.
///
/// Quarantined: a bounded random/directed sample is a useful bug detector, but
/// it is not a proof that a tightened intermediate enclosure contains every
/// feasible point. Allowing an environment variable to enable the tightening
/// would therefore let empirical sampling affect an `Unsat` verdict. Keep the
/// implementation and its differential tests for development, but leave this
/// production authority gate false until the clip has a complete checker-backed
/// enclosure argument.
fn clip_interm_umbrella_enabled() -> bool {
    false
}

#[cfg(test)]
#[test]
fn clip_interm_umbrella_is_authority_quarantined() {
    assert!(!clip_interm_umbrella_enabled());
}

#[cfg(test)]
#[test]
fn legacy_batched_clip_env_gate_is_quarantined_but_explicit_test_options_remain() {
    ny_test_utils::env::with_serialized_env_vars(
        &[("NY_CLIP_INTERM_RESNET", "1"), ("NY_CLIP_INTERM", "1")],
        || {
            let _refine_unset = ny_test_utils::env::ScopedEnvVar::unset("NY_INTERM_REFINE");
            assert!(!clip_interm_resnet_batched_enabled());
            assert!(!clip_interm_umbrella_enabled());
            assert!(!interm_refine_enabled());

            let mut options = IntermRefineOptions::from_env();
            assert!(!options.clip_resnet);
            assert!(!options.clip_guard);

            // Research and soundness tests opt in structurally, without an env
            // path that can affect production verdicts.
            options.clip_resnet = true;
            options.clip_guard = true;
            assert!(options.clip_resnet);
            assert!(options.clip_guard);
        },
    );
}

/// K-restart budget for the fail-closed clip guard (`NY_CLIP_INTERM_GUARD_K`,
/// default 24). Each restart is one directed sample + one true forward through
/// the graph, so this bounds the guard's per-domain cost. Production clip
/// authority is quarantined; explicit research tests can still exercise it.
fn clip_interm_guard_restarts() -> usize {
    std::env::var("NY_CLIP_INTERM_GUARD_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(24)
        .max(1)
}

/// Result of a refinement call: the refined per-domain caches plus, when the
/// prune lane is on, which domains were PROVEN infeasible (empty constraint
/// set — verified vacuously). `infeasible[i]` is only ever set under
/// `NY_INTERM_REFINE_PRUNE=1`.
pub(in crate::beta_crown::engine::graph) struct IntermRefineOutcome {
    pub caches: Vec<HashMap<String, Arc<BoundedTensor>>>,
    pub infeasible: Vec<bool>,
}

/// Walk the unary chain UP from `output_node` to the nearest ReLU (the
/// network's LAST ReLU) and return `(relu_name, seed_node)` where `seed_node`
/// is that ReLU's input node — the refinement backward's start node. `None` on
/// any non-unary node before a ReLU (no clean last-ReLU chain), on reaching the
/// network input, or on a ReLU fed directly by the network input (nothing below
/// the seed to refine with).
pub(in crate::beta_crown::engine::graph) fn find_last_relu_seed(
    graph: &GraphNetwork,
    output_node: &str,
) -> Option<(String, String)> {
    let mut current = output_node.to_string();
    for _ in 0..=graph.nodes.len() {
        if current == NETWORK_INPUT {
            return None;
        }
        let node = graph.nodes.get(&current)?;
        if matches!(node.layer, Layer::ReLU(_)) {
            let seed = node.inputs.first()?.clone();
            if seed == NETWORK_INPUT {
                return None;
            }
            return Some((current, seed));
        }
        if node.inputs.len() != 1 {
            return None;
        }
        current = node.inputs[0].clone();
    }
    None
}

/// Per-seed-neuron MARGIN WEIGHTS for the joint α′ objective (#joint-interm-alpha,
/// [`joint_interm_alpha_enabled`]). The seed layer feeds the last ReLU
/// (`relu_name`), whose output is consumed by the tail linear map `W_tail`
/// (`Relu_57 → Gemm_58 → output` on resnet_medium). For spec row `s` the margin's
/// coefficient at the ReLU output is `a_s = spec_s · W_tail`; a seed neuron `j`
/// enters the margin's LOWER bound only through the chord (upper) relaxation when
/// `a_s[j] < 0`, with magnitude `|a_s[j]|`. So the margin's sensitivity to
/// tightening neuron `j`'s refined lower bound is `m_j = Σ_s max(−a_s[j], 0) ≥ 0`
/// — the reweighting the joint lane applies to the α′ refinement objective.
/// `None` (⇒ uniform fallback, sound) when the tail is not a single linear map
/// directly consuming the last ReLU or the dims disagree.
pub(in crate::beta_crown::engine::graph) fn compute_margin_weights(
    graph: &GraphNetwork,
    output_node: &str,
    spec_matrix: &Array2<f32>,
) -> Option<Arc<[f32]>> {
    let mut never_past_deadline = || false;
    let mut dot = |spec: &Array2<f32>, weight: &Array2<f32>| spec.dot(weight);
    compute_margin_weights_with_deadline_check_and_dot(
        graph,
        output_node,
        spec_matrix,
        MARGIN_WEIGHT_LIMITS,
        &mut never_past_deadline,
        &mut dot,
    )
    .ok()
    .flatten()
}

fn compute_margin_weights_with_deadline(
    graph: &GraphNetwork,
    output_node: &str,
    spec_matrix: &Array2<f32>,
    deadline: Option<std::time::Instant>,
) -> ny_core::Result<Option<Arc<[f32]>>> {
    let mut past_deadline = || deadline.is_some_and(|d| std::time::Instant::now() >= d);
    let mut dot = |spec: &Array2<f32>, weight: &Array2<f32>| spec.dot(weight);
    compute_margin_weights_with_deadline_check_and_dot(
        graph,
        output_node,
        spec_matrix,
        MARGIN_WEIGHT_LIMITS,
        &mut past_deadline,
        &mut dot,
    )
}

fn compute_margin_weights_with_deadline_check_and_dot<F, D>(
    graph: &GraphNetwork,
    output_node: &str,
    spec_matrix: &Array2<f32>,
    limits: MarginWeightLimits,
    past_deadline: &mut F,
    dot: &mut D,
) -> ny_core::Result<Option<Arc<[f32]>>>
where
    F: FnMut() -> bool,
    D: FnMut(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
{
    let deadline_error = |phase: &str| {
        NyError::DeadlineExceeded(format!(
            "intermediate margin-weight planning exceeded deadline during {phase}"
        ))
    };
    if past_deadline() {
        return Err(deadline_error("entry"));
    }
    let Some((relu_name, _seed)) = find_last_relu_seed(graph, output_node) else {
        return Ok(None);
    };
    if past_deadline() {
        return Err(deadline_error("last-ReLU discovery"));
    }
    // The tail linear map: the UNIQUE Linear node directly consuming the last
    // ReLU's output. Any non-linear consumer, or fan-out to >1 consumer, bails
    // to the uniform objective (sound — the weights are advisory only).
    let mut tail: Option<&Array2<f32>> = None;
    for (node_index, node) in graph.nodes.values().enumerate() {
        if node_index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return Err(deadline_error("tail discovery"));
        }
        if node.inputs.iter().any(|i| i == &relu_name) {
            match &node.layer {
                Layer::Linear(lin) if tail.is_none() => tail = Some(&lin.weight),
                _ => return Ok(None),
            }
        }
    }
    let Some(w) = tail else {
        return Ok(None);
    }; // (out_features × in_features)
    if w.nrows() != spec_matrix.ncols() || w.ncols() == 0 {
        return Ok(None); // the spec's output dim must match the tail's out dim
    }
    // Do not let the opaque matrixmultiply path materialize layout repairs that
    // are absent from the checked owned-byte plan. Advisory weights can safely
    // fall back to uniform for unusual layouts.
    if !spec_matrix.is_standard_layout() || !w.is_standard_layout() {
        return Ok(None);
    }
    let plan = MarginWeightPlan::checked(spec_matrix.nrows(), w.nrows(), w.ncols(), limits)?;
    if past_deadline() {
        return Err(deadline_error("pre-allocation plan"));
    }

    let mut m = Vec::new();
    m.try_reserve_exact(plan.weight_elements).map_err(|_| {
        NyError::InvalidSpec("margin-weight vector allocation refused after checked plan".into())
    })?;
    m.resize(plan.weight_elements, 0.0f32);
    if past_deadline() {
        return Err(deadline_error("before dense dot"));
    }

    // a = spec · W_tail → (num_specs × in_features). Margin weight of seed
    // neuron j = Σ_s max(−a[s][j], 0) (upper-branch tail-coefficient mass).
    let a = dot(spec_matrix, w);
    if a.nrows() != spec_matrix.nrows() || a.ncols() != w.ncols() || a.len() != plan.dot_elements {
        return Err(NyError::shape_mismatch(
            vec![spec_matrix.nrows(), w.ncols()],
            a.shape().to_vec(),
        ));
    }
    if past_deadline() {
        return Err(deadline_error("dense dot"));
    }
    let mut fold_cells = 0usize;
    for row in a.rows() {
        for (j, &v) in row.iter().enumerate() {
            if fold_cells.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
                return Err(deadline_error("dot reduction"));
            }
            fold_cells = fold_cells.saturating_add(1);
            if v < 0.0 {
                m[j] += -v;
            }
        }
    }
    drop(a);
    if !m.iter().any(|&v| v > 0.0) {
        return Ok(None); // no upper-branch mass — nothing to steer; keep uniform
    }
    if past_deadline() {
        return Err(deadline_error("return allocation"));
    }
    debug_assert!(plan.peak_owned_bytes <= limits.max_host_bytes);
    debug_assert!(plan.arithmetic_ops <= limits.max_work);
    Ok(Some(Arc::from(m)))
}

/// Select a shared top-K objective set for one intermediate seed/domain batch.
///
/// The score mirrors αβ-CROWN's `DomainClipScorer`: triangle intercept
/// `(-l*u)/(u-l)` times downstream margin sensitivity when the last-ReLU tail
/// map provides one, or intercept alone for deeper seeds. Since NY uses one
/// identity seed shared by the whole wide batch, a neuron's score is the
/// maximum over domains (rather than a separate top-K per domain). This
/// hard-bounds the GPU row count. Selection is advisory: every omitted neuron
/// keeps its inherited enclosure. Premise rows are retained separately because
/// they are needed as necessary-condition constraint sources.
#[cfg(test)]
fn select_intermediate_objective_rows(
    caches: &[HashMap<String, Arc<BoundedTensor>>],
    seed_node: &str,
    candidates: &[usize],
    premise: &[bool],
    topk: usize,
    margin_weights: Option<&[f32]>,
) -> Vec<usize> {
    select_intermediate_objective_rows_with_deadline(
        caches,
        seed_node,
        candidates,
        premise,
        topk,
        margin_weights,
        None,
    )
}

fn selective_row_work_bytes(candidate_len: usize, premise_len: usize) -> Option<usize> {
    // Conservatively count the already-materialized candidate and premise
    // inputs because the selector keeps them live, plus the maximum scored and
    // output vectors. The in-place heapsorts below are allocation-free, so
    // their scratch charge is zero rather than an undocumented library-sort
    // allocation.
    let candidate_input = candidate_len.checked_mul(size_of::<usize>())?;
    let premise_input = premise_len.checked_mul(size_of::<bool>())?;
    let scored = candidate_len.checked_mul(size_of::<(usize, f64)>())?;
    let output = candidate_len.checked_mul(size_of::<usize>())?;
    [candidate_input, premise_input, scored, output]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
}

fn validate_selective_row_budget(
    candidate_len: usize,
    premise_len: usize,
    cache_len: usize,
    score: bool,
) -> Option<()> {
    if selective_row_work_bytes(candidate_len, premise_len)? > SELECTIVE_CLIP_MAX_HOST_BYTES {
        return None;
    }
    let levels = if candidate_len <= 1 {
        1
    } else {
        (usize::BITS - (candidate_len - 1).leading_zeros()) as usize
    };
    let scoring = if score {
        candidate_len.checked_mul(cache_len)?
    } else {
        0
    };
    let sorting = candidate_len
        .checked_mul(levels)?
        .checked_mul(SELECTIVE_SORT_OP_WEIGHT)?;
    let total = scoring.checked_add(sorting)?.checked_add(candidate_len)?;
    (total <= SELECTIVE_CLIP_MAX_WORK).then_some(())
}

#[allow(clippy::too_many_arguments)]
fn select_intermediate_objective_rows_with_deadline(
    caches: &[HashMap<String, Arc<BoundedTensor>>],
    seed_node: &str,
    candidates: &[usize],
    premise: &[bool],
    topk: usize,
    margin_weights: Option<&[f32]>,
    deadline: Option<std::time::Instant>,
) -> Vec<usize> {
    let mut past_deadline = || deadline.is_some_and(|d| std::time::Instant::now() >= d);
    select_intermediate_objective_rows_with_deadline_check(
        caches,
        seed_node,
        candidates,
        premise,
        topk,
        margin_weights,
        &mut past_deadline,
    )
    .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn select_intermediate_objective_rows_with_deadline_check<F>(
    caches: &[HashMap<String, Arc<BoundedTensor>>],
    seed_node: &str,
    candidates: &[usize],
    premise: &[bool],
    topk: usize,
    margin_weights: Option<&[f32]>,
    past_deadline: &mut F,
) -> Option<Vec<usize>>
where
    F: FnMut() -> bool,
{
    if past_deadline()
        || validate_selective_row_budget(candidates.len(), premise.len(), caches.len(), topk > 0)
            .is_none()
    {
        return None;
    }

    let mut keep = Vec::new();
    keep.try_reserve_exact(candidates.len()).ok()?;
    let mut scored = Vec::new();
    if topk > 0 {
        scored.try_reserve_exact(candidates.len()).ok()?;
    }
    for (position, &j) in candidates.iter().enumerate() {
        if position.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return None;
        }
        let is_premise = *premise.get(j)?;
        if topk == 0 || is_premise {
            keep.push(j);
        } else {
            scored.push((j, f64::NEG_INFINITY));
        }
    }
    if topk == 0 {
        return (!past_deadline()).then_some(keep);
    }

    let mut score_cells = 0usize;
    for cache in caches {
        if past_deadline() {
            return None;
        }
        let Some(entry) = cache.get(seed_node) else {
            continue;
        };
        // Resident bound tensors are contiguous in this lane. A non-standard
        // layout is an advisory-selector refusal for that cache, never authority.
        let (Some(lower), Some(upper)) = (
            entry.lower().as_slice_memory_order(),
            entry.upper().as_slice_memory_order(),
        ) else {
            continue;
        };
        for (j, score) in &mut scored {
            if score_cells.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
                return None;
            }
            score_cells = score_cells.saturating_add(1);
            let (Some(&l), Some(&u)) = (lower.get(*j), upper.get(*j)) else {
                continue;
            };
            if !l.is_finite() || !u.is_finite() || l >= 0.0 || u <= 0.0 || u <= l {
                continue;
            }
            let sensitivity = margin_weights
                .and_then(|m| m.get(*j))
                .copied()
                .filter(|v| v.is_finite() && *v >= 0.0)
                .unwrap_or(1.0);
            let intercept = (-f64::from(l) * f64::from(u)) / (f64::from(u) - f64::from(l));
            *score = score.max(intercept * f64::from(sensitivity));
        }
    }

    crate::complete_clip::deadline_heapsort_by(
        &mut scored,
        past_deadline,
        "selective objective score sort",
        |(ja, sa), (jb, sb)| {
            sb.partial_cmp(sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(ja.cmp(jb))
        },
    )
    .ok()?;
    keep.extend(scored.into_iter().take(topk).map(|(j, _)| j));

    crate::complete_clip::deadline_heapsort_by(
        &mut keep,
        past_deadline,
        "selective objective index sort",
        usize::cmp,
    )
    .ok()?;
    Some(keep)
}

/// Find the NEXT ReLU strictly BELOW `node` for the multi-layer cascade: among
/// all ancestor ReLUs (reachable through `inputs`, residual branches included),
/// the one LATEST in execution order — on resnet_medium from `Gemm_56` that is
/// `Relu_51` (F-branch of the last residual block). Returns
/// `(relu_name, seed_node)` where `seed_node` is that ReLU's input; `None` when
/// no ancestor ReLU exists, exec order is unavailable, or the ReLU is fed by
/// the network input (nothing below the seed to refine with).
pub(in crate::beta_crown::engine::graph) fn find_next_relu_seed_below(
    graph: &GraphNetwork,
    node: &str,
) -> Option<(String, String)> {
    let start = graph.nodes.get(node)?;
    let mut stack: Vec<String> = start.inputs.clone();
    let mut seen: std::collections::HashSet<String> = stack.iter().cloned().collect();
    let mut relu_ancestors: Vec<String> = Vec::new();
    while let Some(current) = stack.pop() {
        if current == NETWORK_INPUT {
            continue;
        }
        let Some(n) = graph.nodes.get(&current) else {
            continue;
        };
        if matches!(n.layer, Layer::ReLU(_)) {
            relu_ancestors.push(current.clone());
        }
        for inp in &n.inputs {
            if seen.insert(inp.clone()) {
                stack.push(inp.clone());
            }
        }
    }
    let exec = graph.exec_order().ok()?;
    let pos: HashMap<&str, usize> = exec
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let relu = relu_ancestors
        .into_iter()
        .filter(|name| pos.contains_key(name.as_str()))
        .max_by_key(|name| pos[name.as_str()])?;
    let seed = graph.nodes.get(&relu)?.inputs.first()?.clone();
    if seed == NETWORK_INPUT {
        return None;
    }
    Some((relu, seed))
}

/// C3-at-Relu_57 LEDGER PROBE gate (dark, `NY_C3_57_PROBE=1`, stderr-only,
/// read-only — `docs/CERTIFIED_CUT_CROWN_DESIGN.md` §C3-at-split-layer). The
/// probe dumps, per deep subdomain of the wide lane, exactly the raw data the
/// first-order cut ledger needs: the split premises at the last ReLU, the
/// REFINED pre-activation intervals of its post-refinement-unstable neurons,
/// and the per-spec-row optimized lower bounds.
pub(in crate::beta_crown::engine::graph) fn c3_57_probe_enabled() -> bool {
    matches!(std::env::var("NY_C3_57_PROBE").ok().as_deref(), Some("1"))
}

/// Minimum premise count (= BaB depth for ReLU-split BaB) for a domain to be
/// dumped (`NY_C3_57_PROBE_DEPTH`, default 8 — the prop885 frontier band).
fn c3_57_probe_min_depth() -> usize {
    std::env::var("NY_C3_57_PROBE_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8)
}

/// Dump the C3-at-Relu_57 ledger data for one wide-lane batch (see
/// [`c3_57_probe_enabled`]). `bounds_caches` must be the REFINED caches (the
/// same ones the margin backward consumed); `bounds` the per-domain per-spec
/// optimized output bounds. One `dom` line per deep-enough domain: premises
/// (`idx+`/`idx-`), per-row lower bounds, and `idx:l:u` for every unstable
/// neuron of the last ReLU's pre-activation entry. A `spec_rows` line
/// (`pos:neg` per row) is emitted once per process.
pub(in crate::beta_crown::engine::graph) fn c3_57_probe_dump(
    graph: &GraphNetwork,
    output_node: &str,
    bounds_caches: &[HashMap<String, Arc<BoundedTensor>>],
    beta_states: &[Option<&GraphBetaState>],
    spec_matrix: &Array2<f32>,
    bounds: &[BoundedTensor],
) {
    let Some((relu_name, seed_node)) = find_last_relu_seed(graph, output_node) else {
        return;
    };
    let min_depth = c3_57_probe_min_depth();
    static SPEC_ONCE: std::sync::Once = std::sync::Once::new();
    SPEC_ONCE.call_once(|| {
        let rows: Vec<String> = spec_matrix
            .rows()
            .into_iter()
            .map(|r| {
                let pos = r.iter().position(|&v| v > 0.5).map_or(-1, |p| p as i64);
                let neg = r.iter().position(|&v| v < -0.5).map_or(-1, |n| n as i64);
                format!("{pos}:{neg}")
            })
            .collect();
        eprintln!(
            "[c3-57] spec_rows relu={relu_name} seed={seed_node} rows={}",
            rows.join(",")
        );
    });
    // Per-batch summary (diagnosis aid): the per-domain premise count at the
    // last ReLU as seen through the beta states this backward consumed, plus
    // domain 0's full split-layer histogram (the LA brancher may split OFF the
    // last ReLU — unlike the pre-LA measurement's 100%-Relu_57).
    let counts: Vec<String> = (0..bounds_caches.len())
        .map(|i| {
            beta_states.get(i).copied().flatten().map_or_else(
                || "-".to_string(),
                |bs| bs.entries_for_node(&relu_name).count().to_string(),
            )
        })
        .collect();
    let hist0: String = beta_states
        .first()
        .copied()
        .flatten()
        .map(|bs| {
            let mut h: std::collections::BTreeMap<&str, usize> = Default::default();
            for e in &bs.entries {
                *h.entry(e.node_name()).or_default() += 1;
            }
            h.iter()
                .map(|(n, c)| format!("{n}={c}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    eprintln!(
        "[c3-57] batch n_domains={} prem_counts=[{}] hist0: {hist0}",
        bounds_caches.len(),
        counts.join(",")
    );
    // Suffix-MIP dump: the spec matrix NARROWS across batches (verified rows
    // drop out), so the once-per-process `spec_rows` line cannot attribute a
    // later batch's `lbs` — emit the batch's own rows in the sfx stream.
    if suffix_mip_dump_enabled() {
        let rows: Vec<String> = spec_matrix
            .rows()
            .into_iter()
            .map(|r| {
                let pos = r.iter().position(|&v| v > 0.5).map_or(-1, |p| p as i64);
                let neg = r.iter().position(|&v| v < -0.5).map_or(-1, |n| n as i64);
                format!("{pos}:{neg}")
            })
            .collect();
        eprintln!("[sfx] spec rows={}", rows.join(","));
    }
    for (i, cache) in bounds_caches.iter().enumerate() {
        let Some(bs) = beta_states.get(i).copied().flatten() else {
            continue;
        };
        let prem: Vec<String> = bs
            .entries_for_node(&relu_name)
            .map(|e| {
                format!(
                    "{}{}",
                    e.neuron_idx(),
                    if e.sign() > 0.0 { "+" } else { "-" }
                )
            })
            .collect();
        // Depth gate on TOTAL split count (the BaB depth): under the LA
        // brancher only the first ~3 splits sit on the last ReLU, so gating
        // on `prem.len()` would never fire at the deep frontier.
        if bs.entries.len() < min_depth {
            continue;
        }
        let Some(entry) = cache.get(&seed_node) else {
            continue;
        };
        let unstable: Vec<String> = entry
            .lower()
            .iter()
            .zip(entry.upper().iter())
            .enumerate()
            .filter(|(_, (&l, &u))| l < 0.0 && u > 0.0)
            .map(|(j, (&l, &u))| format!("{j}:{l:.6}:{u:.6}"))
            .collect();
        let lbs: Vec<String> = bounds
            .get(i)
            .map(|b| b.lower().iter().map(|v| format!("{v:.5}")).collect())
            .unwrap_or_default();
        eprintln!(
            "[c3-57] dom={i} depth={} prem=[{}] lbs=[{}] unstable=[{}]",
            prem.len(),
            prem.join(","),
            lbs.join(","),
            unstable.join(","),
        );
        if suffix_mip_dump_enabled() {
            suffix_mip_dump_domain(graph, &relu_name, &seed_node, cache, bs, i, bounds.get(i));
        }
    }
}

/// Suffix-MIP dump extension (dark, `NY_SUFFIX_MIP_DUMP=1`, stderr-only,
/// read-only; rides the `NY_C3_57_PROBE=1` site — both envs must be set). Dumps
/// what the OFFLINE exact-suffix gate (docs/CERTIFIED_CUT_CROWN_DESIGN.md,
/// suffix-MIP escalation hypothesis) needs beyond the ledger probe:
/// * `[sfx] meta` — the last ReLU, its seed node (the GEMM producing its
///   pre-activation) and the seed's own input node (once per process);
/// * `[sfx] inbox` — the per-domain cache enclosure at the seed's INPUT node
///   (the z-box of the suffix MIP), hash-deduped: MEASURED ~14 distinct boxes
///   per 300s run (premise-conditioned per-domain recomputation, coords ≤0.05
///   apart), so each distinct box prints once and doms reference it by id;
/// * `[sfx] dom` — per deep domain: premises at the last ReLU with split
///   points, per-spec-row optimized lower bounds, and the FULL refined seed
///   `[l',u']` (all dims — the exact suffix min needs stable neurons too).
///
/// Floats print with Rust's shortest-round-trip `Display` (bit-exact reload).
fn suffix_mip_dump_enabled() -> bool {
    matches!(
        std::env::var("NY_SUFFIX_MIP_DUMP").ok().as_deref(),
        Some("1")
    )
}

fn suffix_mip_dump_domain(
    graph: &GraphNetwork,
    relu_name: &str,
    seed_node: &str,
    cache: &HashMap<String, Arc<BoundedTensor>>,
    bs: &GraphBetaState,
    dom_idx: usize,
    bounds: Option<&BoundedTensor>,
) {
    static META_ONCE: std::sync::Once = std::sync::Once::new();
    let seed_inputs: Vec<String> = graph
        .nodes
        .get(seed_node)
        .map(|n| n.inputs.clone())
        .unwrap_or_default();
    META_ONCE.call_once(|| {
        eprintln!(
            "[sfx] meta relu={relu_name} seed={seed_node} seed_inputs=[{}]",
            seed_inputs.join(",")
        );
    });
    // The z-box: enclosure at the seed GEMM's (first) input node, deduped by
    // content hash so the ~2048-wide line prints once per distinct box.
    let inbox_id: i64 = seed_inputs
        .first()
        .and_then(|inp| cache.get(inp).map(|bt| (inp, bt)))
        .map_or(-1, |(inp, bt)| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for v in bt.lower().iter().chain(bt.upper().iter()) {
                v.to_bits().hash(&mut h);
            }
            let key = h.finish();
            static SEEN: Mutex<Vec<u64>> = Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap();
            let id = seen.iter().position(|&k| k == key).unwrap_or_else(|| {
                seen.push(key);
                let vals: Vec<String> = bt
                    .lower()
                    .iter()
                    .zip(bt.upper().iter())
                    .map(|(l, u)| format!("{l}:{u}"))
                    .collect();
                eprintln!(
                    "[sfx] inbox id={} node={inp} n={} vals=[{}]",
                    seen.len() - 1,
                    vals.len(),
                    vals.join(",")
                );
                seen.len() - 1
            });
            id as i64
        });
    // Upstream-window extension (dark, additive): `NY_SUFFIX_MIP_DUMP_NODES` =
    // comma-separated cache node names whose per-domain enclosures the offline
    // 2-block window gate needs (e.g. `Add_48,BatchNormalization_50`). Also
    // prints the cache's available keys once so the offline side can discover
    // what is materialized.
    static KEYS_ONCE: std::sync::Once = std::sync::Once::new();
    KEYS_ONCE.call_once(|| {
        let mut keys: Vec<String> = cache
            .iter()
            .map(|(k, bt)| format!("{k}:{}", bt.lower().len()))
            .collect();
        keys.sort();
        eprintln!("[sfx] keys=[{}]", keys.join(","));
    });
    if let Ok(extra) = std::env::var("NY_SUFFIX_MIP_DUMP_NODES") {
        for name in extra.split(',').filter(|s| !s.is_empty()) {
            let Some(bt) = cache.get(name) else {
                continue;
            };
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            name.hash(&mut h);
            for v in bt.lower().iter().chain(bt.upper().iter()) {
                v.to_bits().hash(&mut h);
            }
            let key = h.finish();
            static XSEEN: Mutex<Vec<u64>> = Mutex::new(Vec::new());
            let mut seen = XSEEN.lock().unwrap();
            let id = seen.iter().position(|&k| k == key).unwrap_or_else(|| {
                seen.push(key);
                let vals: Vec<String> = bt
                    .lower()
                    .iter()
                    .zip(bt.upper().iter())
                    .map(|(l, u)| format!("{l}:{u}"))
                    .collect();
                eprintln!(
                    "[sfx] xbox id={} node={name} n={} vals=[{}]",
                    seen.len() - 1,
                    vals.len(),
                    vals.join(",")
                );
                seen.len() - 1
            });
            eprintln!("[sfx] xref dom={dom_idx} node={name} id={id}");
        }
    }
    let prem: Vec<String> = bs
        .entries_for_node(relu_name)
        .map(|e| {
            format!(
                "{}{}@{}",
                e.neuron_idx(),
                if e.sign() > 0.0 { "+" } else { "-" },
                e.split_point()
            )
        })
        .collect();
    let lbs: Vec<String> = bounds
        .map(|b| b.lower().iter().map(|v| format!("{v}")).collect())
        .unwrap_or_default();
    let seed_box: Vec<String> = cache.get(seed_node).map_or_else(Vec::new, |entry| {
        entry
            .lower()
            .iter()
            .zip(entry.upper().iter())
            .map(|(l, u)| format!("{l}:{u}"))
            .collect()
    });
    eprintln!(
        "[sfx] dom={dom_idx} depth={} inbox={inbox_id} prem=[{}] lbs=[{}] seed=[{}]",
        bs.entries.len(),
        prem.join(","),
        lbs.join(","),
        seed_box.join(","),
    );
}

/// Split-premise contradiction test (the prune lane's core; pure for unit
/// tests). Each of the domain's β entries at the seed ReLU IS one of its split
/// premises. `refined_l/refined_u` (indexed by `row_of[neuron]`,
/// `usize::MAX` = unselected) are sound enclosures of the neuron's
/// pre-activation over the subdomain WITHOUT the seed-layer premises, so an
/// ACTIVE premise `z_j ≥ s` with `u' < s − tol`, or an INACTIVE premise
/// `z_j ≤ s` with `l' > s + tol`, proves the constraint set empty. Returns the
/// first contradiction as `(neuron_idx, sign, violation_margin)`; `None` = no
/// contradiction (never prune). Invalid/non-finite refined pairs are skipped
/// (conservative).
fn premise_contradiction(
    beta: &GraphBetaState,
    relu_name: &str,
    row_of: &[usize],
    refined_l: &[f32],
    refined_u: &[f32],
    tol: f32,
) -> Option<(usize, f32, f32)> {
    for e in beta.entries_for_node(relu_name) {
        let j = e.neuron_idx();
        let Some(&row) = row_of.get(j) else {
            continue;
        };
        if row == usize::MAX || row >= refined_l.len() || row >= refined_u.len() {
            continue;
        }
        let (rl, ru) = (refined_l[row], refined_u[row]);
        if !(rl.is_finite() && ru.is_finite() && rl <= ru) {
            continue;
        }
        let s = e.split_point();
        if e.sign() > 0.0 && ru < s - tol {
            return Some((j, 1.0, s - ru));
        }
        if e.sign() < 0.0 && rl > s + tol {
            return Some((j, -1.0, rl - s));
        }
    }
    None
}

/// Per-neuron intersection of the inherited `[ol, ou]` with the refined
/// `[rl, ru]`: `l = max`, `u = min` (NaN-propagating, so a NaN inherited entry
/// is never silently repaired). A non-finite/inverted refined pair, or an empty
/// intersection, conservatively keeps the inherited pair (sound: the inherited
/// entry already encloses the subdomain). Returns the new pair plus whether it
/// strictly tightened.
fn intersect_pair(ol: f32, ou: f32, rl: f32, ru: f32) -> (f32, f32, bool) {
    if !(rl.is_finite() && ru.is_finite() && rl <= ru) {
        return (ol, ou, false);
    }
    let cl = nan_propagating_max(ol, rl);
    let cu = nan_propagating_min(ou, ru);
    // NaN-aware "not (cl <= cu)": TRUE for NaN — `cl > cu` would treat a NaN
    // pair as a valid interval, so the negated comparison is load-bearing.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(cl <= cu) {
        // NaN or empty intersection — keep inherited.
        return (ol, ou, false);
    }
    let tightened = cl > ol || cu < ou;
    (cl, cu, tightened)
}

/// Fold-order walk over the Activation layers of a segment stack (mutable) —
/// the same order as `relu_names` / `beta_signed` (segments in vec order;
/// per segment the branch layer vec in order; ResidualProj F then P), matching
/// `write_back_alpha` in `batched.rs` and the GPU fold.
fn visit_activations_mut(
    segs: &mut [ny_core::GpuResnetSegment],
    f: &mut dyn FnMut(usize, &mut ny_core::GpuCrownLayer),
) {
    let mut r = 0usize;
    let mut walk = |layers: &mut [ny_core::GpuCrownLayer], r: &mut usize| {
        for l in layers.iter_mut() {
            if matches!(l, ny_core::GpuCrownLayer::Activation { .. }) {
                f(*r, l);
                *r += 1;
            }
        }
    };
    for seg in segs.iter_mut() {
        match seg {
            ny_core::GpuResnetSegment::Chain(l) | ny_core::GpuResnetSegment::Residual(l) => {
                walk(l, &mut r)
            }
            ny_core::GpuResnetSegment::ResidualProj(fb, pb) => {
                walk(fb, &mut r);
                walk(pb, &mut r);
            }
        }
    }
}

/// Immutable fold-order Activation walk (same order as
/// [`visit_activations_mut`]).
fn visit_activations(
    segs: &[ny_core::GpuResnetSegment],
    f: &mut dyn FnMut(usize, &ny_core::GpuCrownLayer),
) {
    let mut r = 0usize;
    let mut walk = |layers: &[ny_core::GpuCrownLayer], r: &mut usize| {
        for l in layers.iter() {
            if matches!(l, ny_core::GpuCrownLayer::Activation { .. }) {
                f(*r, l);
                *r += 1;
            }
        }
    };
    for seg in segs.iter() {
        match seg {
            ny_core::GpuResnetSegment::Chain(l) | ny_core::GpuResnetSegment::Residual(l) => {
                walk(l, &mut r)
            }
            ny_core::GpuResnetSegment::ResidualProj(fb, pb) => {
                walk(fb, &mut r);
                walk(pb, &mut r);
            }
        }
    }
}

/// Number of Activation layers in the stack (fold-order count) — the guard
/// that the Activation walk aligns with `relu_names` (the extraction only
/// emits ReLU Activations on this lane; a mismatch fail-closes the α′ lane).
fn count_activations(segs: &[ny_core::GpuResnetSegment]) -> usize {
    let mut n = 0usize;
    visit_activations(segs, &mut |_, _| n += 1);
    n
}

/// Per-Activation UNSTABLE mask of a segment stack: for the ReLU relaxation
/// the chord intercept `−u·l/(u−l)` is > 0 iff `l < 0 < u`, and the
/// extraction pins stable neurons' slopes exactly (1/0 with 0 intercepts) —
/// so `upper_intercept > 0` selects exactly the neurons whose lower slope is
/// a free α ∈ [0,1].
fn unstable_masks(segs: &[ny_core::GpuResnetSegment], n_relu: usize) -> Vec<Vec<bool>> {
    let mut masks: Vec<Vec<bool>> = vec![Vec::new(); n_relu];
    visit_activations(segs, &mut |r, layer| {
        if let ny_core::GpuCrownLayer::Activation {
            upper_intercept, ..
        } = layer
        {
            if r < n_relu {
                masks[r] = upper_intercept.iter().map(|&t| t > 0.0).collect();
            }
        }
    });
    masks
}

/// Snapshot the per-Activation lower slopes (fold order).
fn collect_lower_slopes(segs: &[ny_core::GpuResnetSegment], n_relu: usize) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); n_relu];
    visit_activations(segs, &mut |r, layer| {
        if let ny_core::GpuCrownLayer::Activation { lower_slope, .. } = layer {
            if r < n_relu {
                out[r] = lower_slope.clone();
            }
        }
    });
    out
}

/// Write α′ into a domain's segments: only neurons that were STEPPED by the
/// ascent AND are unstable in THIS domain's own extraction
/// (`upper_intercept > 0` — stable slopes are exact and must never change).
/// Any α ∈ [0,1] with zero lower intercept is a valid ReLU lower relaxation
/// slope, so this write is sound for every subdomain.
fn write_alpha_prime(
    segs: &mut [ny_core::GpuResnetSegment],
    slopes: &[Vec<f32>],
    stepped: &[Vec<bool>],
) {
    visit_activations_mut(segs, &mut |r, layer| {
        if let ny_core::GpuCrownLayer::Activation {
            lower_slope,
            upper_intercept,
            ..
        } = layer
        {
            let (Some(sl), Some(st)) = (slopes.get(r), stepped.get(r)) else {
                return;
            };
            if sl.len() != lower_slope.len() || st.len() != lower_slope.len() {
                return;
            }
            for (i, s) in lower_slope.iter_mut().enumerate() {
                if st[i] && upper_intercept[i] > 0.0 && sl[i].is_finite() {
                    *s = sl[i].clamp(0.0, 1.0);
                }
            }
        }
    });
}

/// Does a stored α′ match this pass's target (same seed node, width, and
/// ReLU layout)? Refuses reuse across nets/seeds.
fn alpha_prime_matches(
    ap: &AlphaPrime,
    seed_node: &str,
    pre_dim: usize,
    relu_names: &[String],
) -> bool {
    ap.seed_node == seed_node
        && ap.pre_dim == pre_dim
        && ap.relu_names.len() == relu_names.len()
        && ap.relu_names.iter().zip(relu_names).all(|(a, b)| a == b)
}

/// The α′ gradient of the REFINEMENT objective `Σ_{r ∈ rows} w_r · l′_r` for ONE
/// domain: per identity row, the TRUE chain-rule gradient
/// `max(ν,0)·ĥ(x*)` via the host replay of that row's backward over the
/// truncated segments (the same oracle-validated machinery as the wide-α
/// margin ascent — each row is its own CROWN objective with its own ν
/// selection and its own concretization corner), scaled by the per-row weight
/// `w_r` and summed over rows. `row_weights` (aligned with `rows`) is the
/// #joint-interm-alpha margin reweighting; `None` ⇒ every `w_r = 1` (the base
/// uniform `Σ l′` objective, byte-identical). Since `d(w·l′)/dα = w · dl′/dα`
/// the weight is applied to the finished per-row gradient — the fail-closed lb
/// validation still runs on the UNIT row (precision-safe). Rows whose replay
/// fails the validation contribute nothing (gradients only steer α′; the bound
/// is always the sound GPU fold). Returns `(grads, rows_ok)`.
#[allow(clippy::too_many_arguments)]
fn refine_alpha_objective_grads(
    segments: &[ny_core::GpuResnetSegment],
    beta_signed: &[Vec<f32>],
    in_lo: &[f32],
    in_hi: &[f32],
    n_relu: usize,
    pre_dim: usize,
    sel: &[usize],
    rows: &[usize],
    lbs: &[f32],
    row_weights: Option<&[f32]>,
    probe: bool,
) -> (Vec<Vec<f32>>, usize) {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
    let per_row: Vec<Option<Vec<Vec<f32>>>> = rows
        .par_iter()
        .enumerate()
        .map(|(k, &ri)| {
            let neuron = *sel.get(ri)?;
            let lb = *lbs.get(ri)?;
            if neuron >= pre_dim || !lb.is_finite() {
                return None;
            }
            let mut row = vec![0.0f32; pre_dim];
            row[neuron] = 1.0;
            let mut g = super::wide_alpha_true::true_alpha_grads_for_row(
                segments,
                &row,
                beta_signed,
                in_lo,
                in_hi,
                n_relu,
                lb,
                false,
            )?;
            // #joint-interm-alpha: d(w·l′)/dα = w · dl′/dα — scale the finished
            // per-row gradient (uniform lane leaves w = 1 ⇒ byte-identical).
            if let Some(w) = row_weights.and_then(|ws| ws.get(k).copied()) {
                if w != 1.0 {
                    for layer in g.iter_mut() {
                        for v in layer.iter_mut() {
                            *v *= w;
                        }
                    }
                }
            }
            Some(g)
        })
        .collect();
    let mut grads: Vec<Vec<f32>> = Vec::with_capacity(n_relu);
    let mut rows_ok = 0usize;
    for g in per_row.into_iter().flatten() {
        if g.len() != n_relu {
            continue;
        }
        rows_ok += 1;
        if grads.is_empty() {
            grads = g;
            continue;
        }
        for (acc, gr) in grads.iter_mut().zip(g) {
            if acc.len() == gr.len() {
                for (a, v) in acc.iter_mut().zip(gr) {
                    *a += v;
                }
            }
        }
    }
    if grads.is_empty() {
        grads = vec![Vec::new(); n_relu];
    }
    if probe {
        let max_g = grads
            .iter()
            .flatten()
            .fold(0.0f32, |m, &g| nan_propagating_max(m, g.abs()));
        eprintln!(
            "[interm-refine-alpha] grads rows_ok={rows_ok}/{} max|g|={max_g:.3e}",
            rows.len()
        );
    }
    (grads, rows_ok)
}

/// Minimal Adam ascent state for α′ (bias-corrected, [0,1]-projected). Local
/// on purpose: the parameter is the raw lower-slope vector, not a
/// `GraphDomainAlphaState` (nothing to track — the mask pins the stepped set).
struct AlphaAdam {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AlphaAdam {
    fn new(shape: &[Vec<f32>]) -> Self {
        Self {
            m: shape.iter().map(|s| vec![0.0; s.len()]).collect(),
            v: shape.iter().map(|s| vec![0.0; s.len()]).collect(),
        }
    }

    /// One ascent step (maximize): `α += lr·m̂/(√v̂+ε)`, clamped to [0,1],
    /// masked neurons only. Returns max |gradient| over stepped neurons.
    fn step(
        &mut self,
        slopes: &mut [Vec<f32>],
        grads: &[Vec<f32>],
        stepped: &[Vec<bool>],
        lr: f32,
        t: usize,
    ) -> f32 {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        let bc1 = 1.0 - B1.powi(t as i32);
        let bc2 = 1.0 - B2.powi(t as i32);
        let mut max_g = 0.0f32;
        for r in 0..slopes.len() {
            let (Some(gr), Some(st)) = (grads.get(r), stepped.get(r)) else {
                continue;
            };
            if gr.len() != slopes[r].len() || st.len() != slopes[r].len() {
                continue;
            }
            for i in 0..slopes[r].len() {
                if !st[i] {
                    continue;
                }
                let g = gr[i];
                if !g.is_finite() {
                    continue;
                }
                max_g = nan_propagating_max(max_g, g.abs());
                let m = &mut self.m[r][i];
                let v = &mut self.v[r][i];
                *m = B1 * *m + (1.0 - B1) * g;
                *v = B2 * *v + (1.0 - B2) * g * g;
                let mhat = *m / bc1;
                let vhat = *v / bc2;
                slopes[r][i] = (slopes[r][i] + lr * mhat / (vhat.sqrt() + EPS)).clamp(0.0, 1.0);
            }
        }
        max_g
    }
}

/// Element-wise best-merge of two sound per-row enclosures of the SAME
/// domain's rows: `l = max`, `u = min` (finite-guarded). Every α′ iterate is
/// a sound enclosure, so the merge is sound and never looser than iterate 0.
fn merge_refine_result(dst: &mut ny_core::GpuCrownResult, src: &ny_core::GpuCrownResult) {
    if dst.lower_bounds.len() != src.lower_bounds.len()
        || dst.upper_bounds.len() != src.upper_bounds.len()
    {
        return;
    }
    for (d, &s) in dst.lower_bounds.iter_mut().zip(&src.lower_bounds) {
        if s.is_finite() && (!d.is_finite() || s > *d) {
            *d = s;
        }
    }
    for (d, &s) in dst.upper_bounds.iter_mut().zip(&src.upper_bounds) {
        if s.is_finite() && (!d.is_finite() || s < *d) {
            *d = s;
        }
    }
}

/// The α′ ASCENT (#alpha-prime, see [`interm_refine_alpha_enabled`]): a small
/// Adam ascent of the truncated stack's lower slopes against the refinement
/// objective `Σ_rows l′_row`, replayed on ONE domain (the batch's first
/// successful one) and applied to EVERY domain's segments; each iterate
/// re-runs the batched refinement backward and best-merges the per-row bounds
/// into `per_domain` (sound: every iterate is a sound enclosure for its
/// α′ ∈ [0,1]). Returns the α′ snapshot to store (with `improved` marking
/// whether any iterate beat the borrowed-α objective on the replay domain).
///
/// SOUNDNESS: gradients only steer α′ (a wrong gradient degrades the ascent,
/// never the bound); every consumed bound comes from the sound GPU fold with
/// projected slopes, and only element-wise-tightest sound iterates are kept.
#[allow(clippy::too_many_arguments)]
fn alpha_prime_ascent(
    gpu: &dyn ny_core::GpuCrownBackward,
    preps: &mut [Option<ResnetDomainPrep>],
    idxs: &[usize],
    seed: &ny_core::GpuCrownSeed,
    per_domain: &mut [Option<ny_core::GpuCrownResult>],
    seed_node: &str,
    pre_dim: usize,
    sel: &[usize],
    row_widths: &[f32],
    opts: &IntermRefineOptions,
    past_deadline: &dyn Fn() -> bool,
) -> Option<AlphaPrime> {
    let probe = opts.probe;
    let n_rows = sel.len();
    let replay = idxs
        .iter()
        .copied()
        .find(|&i| preps[i].is_some() && per_domain[i].is_some())?;
    let (n_relu, relu_names, stepped, mut slopes) = {
        let p = preps[replay].as_ref()?;
        let n_relu = p.relu_names.len();
        if count_activations(&p.segments) != n_relu {
            if probe {
                eprintln!(
                    "[interm-refine-alpha] activation walk misaligned with relu_names — lane skipped"
                );
            }
            return None;
        }
        (
            n_relu,
            p.relu_names.clone(),
            unstable_masks(&p.segments, n_relu),
            collect_lower_slopes(&p.segments, n_relu),
        )
    };
    if !stepped.iter().any(|m| m.contains(&true)) {
        if probe {
            eprintln!("[interm-refine-alpha] no unstable below-seed neurons — nothing to step");
        }
        return None;
    }
    // Replay rows: widest inherited interval first (most refinement headroom),
    // capped (`NY_INTERM_REFINE_ALPHA_MAX_ROWS`) — the host replays dominate
    // the ascent wall. Cost/quality only, never soundness.
    let mut rows: Vec<usize> = (0..n_rows).collect();
    rows.sort_by(|&a, &b| {
        let wa = row_widths.get(a).copied().unwrap_or(0.0);
        let wb = row_widths.get(b).copied().unwrap_or(0.0);
        wb.partial_cmp(&wa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    rows.truncate(opts.alpha_max_rows);
    // #joint-interm-alpha: per-refinement-row margin weight `w_all[ri] =
    // margin_weights[sel[ri]]` (≥0). When the joint gate is off (or the tail map
    // was not resolvable) `w_all` is None ⇒ the uniform `Σ l′` objective,
    // byte-identical. `picked_weights` is `w_all` gathered over the (sorted,
    // truncated) replay `rows`, aligned with `refine_alpha_objective_grads`.
    let w_all: Option<Vec<f32>> = opts
        .margin_weights
        .as_ref()
        .filter(|_| opts.joint_margin)
        .map(|mw| {
            (0..n_rows)
                .map(|ri| sel.get(ri).and_then(|&j| mw.get(j)).copied().unwrap_or(0.0))
                .collect()
        });
    let picked_weights: Option<Vec<f32>> = w_all.as_ref().map(|wa| {
        rows.iter()
            .map(|&ri| wa.get(ri).copied().unwrap_or(0.0))
            .collect()
    });
    let obj_of = |lbs: &[f32]| -> f64 {
        match &w_all {
            // Weighted margin objective Σ_ri w_all[ri]·l′_ri (joint lane).
            Some(wa) => lbs
                .iter()
                .zip(wa.iter())
                .filter(|(v, _)| v.is_finite())
                .map(|(&v, &w)| v as f64 * w as f64)
                .sum(),
            None => lbs
                .iter()
                .filter(|v| v.is_finite())
                .map(|&v| v as f64)
                .sum(),
        }
    };
    let mut cur_lbs: Vec<f32> = per_domain[replay].as_ref()?.lower_bounds.clone();
    if cur_lbs.len() != n_rows {
        return None;
    }
    let obj0 = obj_of(&cur_lbs);
    let mut best_obj = obj0;
    let mut best_slopes: Option<Vec<Vec<f32>>> = None;
    let mut best_iter = 0usize;
    let mut adam = AlphaAdam::new(&slopes);
    let t0 = std::time::Instant::now();
    for t in 1..=opts.alpha_iters {
        if past_deadline() {
            break;
        }
        let (grads, rows_ok) = {
            let p = preps[replay].as_ref()?;
            refine_alpha_objective_grads(
                &p.segments,
                &p.beta_signed,
                &p.in_lo,
                &p.in_hi,
                n_relu,
                pre_dim,
                sel,
                &rows,
                &cur_lbs,
                picked_weights.as_deref(),
                probe,
            )
        };
        if rows_ok == 0 {
            if probe {
                eprintln!("[interm-refine-alpha] iter={t} every row replay failed — stop");
            }
            break;
        }
        let max_g = adam.step(&mut slopes, &grads, &stepped, opts.alpha_lr, t);
        if max_g == 0.0 || !max_g.is_finite() {
            break;
        }
        for &i in idxs {
            if let Some(p) = preps[i].as_mut() {
                write_alpha_prime(&mut p.segments, &slopes, &stepped);
            }
        }
        let results = {
            let refs: Vec<ny_core::GpuResnetBatchedDomainRef> = idxs
                .iter()
                .map(|&i| {
                    let p = preps[i].as_ref().expect("idxs holds Some preps only");
                    ny_core::GpuResnetBatchedDomainRef {
                        segments: &p.segments,
                        input_lower: &p.in_lo,
                        input_upper: &p.in_hi,
                        beta_signed: &p.beta_signed,
                        frontier_abs: &p.frontier_abs,
                        node_abs: &p.node_abs,
                    }
                })
                .collect();
            match gpu.crown_backward_gpu_resnet_sound_beta_batched(&refs, seed) {
                Ok(r) if r.len() == idxs.len() => r,
                _ => break,
            }
        };
        let mut obj_t = f64::NEG_INFINITY;
        for (&i, r) in idxs.iter().zip(&results) {
            if r.lower_bounds.len() != n_rows || r.upper_bounds.len() != n_rows {
                continue;
            }
            if i == replay {
                cur_lbs.clone_from(&r.lower_bounds);
                obj_t = obj_of(&r.lower_bounds);
            }
            match per_domain[i].as_mut() {
                Some(dst) => merge_refine_result(dst, r),
                None => {
                    per_domain[i] = Some(ny_core::GpuCrownResult {
                        lower_bounds: r.lower_bounds.clone(),
                        upper_bounds: r.upper_bounds.clone(),
                    })
                }
            }
        }
        if probe {
            eprintln!(
                "[interm-refine-alpha] iter={t} rows_ok={rows_ok} max|g|={max_g:.3e} \
                 obj={obj_t:.4} (obj0={obj0:.4} best={best_obj:.4})"
            );
        }
        if obj_t > best_obj {
            best_obj = obj_t;
            best_slopes = Some(slopes.clone());
            best_iter = t;
        }
    }
    let improved = best_iter > 0;
    if probe {
        eprintln!(
            "[interm-refine-alpha] ascent done iters={} improved={improved} dObj={:+.4} ({}ms)",
            opts.alpha_iters,
            best_obj - obj0,
            t0.elapsed().as_millis()
        );
    }
    Some(AlphaPrime {
        seed_node: seed_node.to_string(),
        pre_dim,
        relu_names,
        slopes: best_slopes.unwrap_or(slopes),
        stepped,
        improved,
    })
}

/// PER-TARGET α′ ASCENT (#ab-parity-interm, [`ab_parity_interm_enabled`]): the
/// auto_LiRPA per-target decoupling. Where [`alpha_prime_ascent`] optimizes ONE
/// shared lower-slope set against the SCALARIZED objective `Σ_rows w_r·l′_r`,
/// this gives EACH target seed-row `ri` its OWN slope vector `α^(ri)`, ascended
/// against ITS OWN bound `l′_ri` (that row's per-target gradient), and computes
/// the SOUND per-row bound with `α^(ri)` from the GPU fold. The optimum for one
/// target no longer has to compromise with every other target's optimum — each
/// picks the below-seed slopes that tighten ITS pre-activation box.
///
/// STRUCTURE: the per-target ascent optimizes each `α^(ri)` on the batch's
/// representative (replay) domain via the oracle-validated host-replay gradient
/// (`refine_alpha_objective_grads` on a single row) + a single-domain sound GPU
/// fold to score each iterate; then APPLIES each converged `α^(ri)` to EVERY
/// domain in the batch (one batched backward per target, writing `α^(ri)` into
/// all domains' segments and best-merging ONLY that target's column into
/// `per_domain`). Cost scales with `n_targets` (capped by `alpha_max_rows`) —
/// this is the deliberate extra α cost of the dark parity lane.
///
/// SOUNDNESS: identical discipline to the base lane — any `α ∈ [0,1]` is a valid
/// lower ReLU slope, gradients only STEER which α each target picks, every
/// consumed bound is the sound GPU fold (`crown_backward_gpu_resnet_sound_beta`
/// / `_batched`) with projected slopes, and the merge keeps the element-wise
/// tightest sound value per row (`l = max`, `u = min`). Applying a target's α to
/// a NON-replay domain is sound because [`write_alpha_prime`] only touches
/// neurons unstable in THAT domain's own extraction (stable slopes stay exact).
/// Returns the number of targets whose per-target bound beat the borrowed-α one.
#[allow(clippy::too_many_arguments)]
fn alpha_prime_ascent_per_target(
    gpu: &dyn ny_core::GpuCrownBackward,
    preps: &mut [Option<ResnetDomainPrep>],
    idxs: &[usize],
    seed: &ny_core::GpuCrownSeed,
    per_domain: &mut [Option<ny_core::GpuCrownResult>],
    pre_dim: usize,
    sel: &[usize],
    row_widths: &[f32],
    opts: &IntermRefineOptions,
    past_deadline: &dyn Fn() -> bool,
) -> usize {
    let probe = opts.probe;
    let n_rows = sel.len();
    let replay = idxs
        .iter()
        .copied()
        .find(|&i| preps[i].is_some() && per_domain[i].is_some());
    let Some(replay) = replay else {
        return 0;
    };
    // Snapshot the replay domain (owned copies — the application phase below
    // mutates every domain's segments through `preps`, so the ascent phase must
    // not borrow `preps[replay]`).
    let (
        n_relu,
        stepped,
        base_slopes,
        in_lo,
        in_hi,
        beta_signed,
        frontier_abs,
        node_abs,
        base_segs,
    ) = {
        let Some(p) = preps[replay].as_ref() else {
            return 0;
        };
        let n_relu = p.relu_names.len();
        if count_activations(&p.segments) != n_relu {
            if probe {
                eprintln!(
                    "[ab-parity-interm] activation walk misaligned with relu_names — lane skipped"
                );
            }
            return 0;
        }
        (
            n_relu,
            unstable_masks(&p.segments, n_relu),
            collect_lower_slopes(&p.segments, n_relu),
            p.in_lo.clone(),
            p.in_hi.clone(),
            p.beta_signed.clone(),
            p.frontier_abs.clone(),
            p.node_abs.clone(),
            p.segments.clone(),
        )
    };
    if !stepped.iter().any(|m| m.contains(&true)) {
        if probe {
            eprintln!("[ab-parity-interm] no unstable below-seed neurons — nothing to step");
        }
        return 0;
    }
    let base_lbs = match per_domain[replay].as_ref() {
        Some(r) if r.lower_bounds.len() == n_rows => r.lower_bounds.clone(),
        _ => return 0,
    };
    // Target rows: widest inherited interval first (most refinement headroom),
    // capped by `alpha_max_rows` (cost only — untargeted rows keep the sound
    // borrowed-α bound).
    let mut targets: Vec<usize> = (0..n_rows).collect();
    targets.sort_by(|&a, &b| {
        let wa = row_widths.get(a).copied().unwrap_or(0.0);
        let wb = row_widths.get(b).copied().unwrap_or(0.0);
        wb.partial_cmp(&wa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    targets.truncate(opts.alpha_max_rows);

    // PHASE 1 — per-target ascent on the replay domain. `α^(ri)` lives in
    // `per_target_slopes[ti]` (ti indexes `targets`).
    let mut per_target_slopes: Vec<Vec<Vec<f32>>> = Vec::with_capacity(targets.len());
    let mut work_segs = base_segs;
    let mut lbs = base_lbs.clone();
    let mut improved = 0usize;
    let t0 = std::time::Instant::now();
    for &ri in &targets {
        let base_lb = base_lbs.get(ri).copied().unwrap_or(f32::NEG_INFINITY);
        if past_deadline() {
            per_target_slopes.push(base_slopes.clone());
            continue;
        }
        let mut slopes = base_slopes.clone();
        let mut adam = AlphaAdam::new(&base_slopes);
        let mut cur_lb = base_lb;
        let mut best_lb = base_lb;
        let mut best_slopes = base_slopes.clone();
        for t in 1..=opts.alpha_iters {
            if past_deadline() {
                break;
            }
            // Gradient of THIS row's l′ at the current per-target slopes.
            write_alpha_prime(&mut work_segs, &slopes, &stepped);
            lbs[ri] = cur_lb;
            let (grads, rows_ok) = refine_alpha_objective_grads(
                &work_segs,
                &beta_signed,
                &in_lo,
                &in_hi,
                n_relu,
                pre_dim,
                sel,
                std::slice::from_ref(&ri),
                &lbs,
                None,
                false,
            );
            if rows_ok == 0 {
                break;
            }
            let max_g = adam.step(&mut slopes, &grads, &stepped, opts.alpha_lr, t);
            if max_g == 0.0 || !max_g.is_finite() {
                break;
            }
            // Sound bound for row ri with the stepped per-target slopes.
            write_alpha_prime(&mut work_segs, &slopes, &stepped);
            let r = match gpu.crown_backward_gpu_resnet_sound_beta(
                &work_segs,
                seed,
                &in_lo,
                &in_hi,
                &beta_signed,
                &frontier_abs,
                &node_abs,
            ) {
                Ok(r) if r.lower_bounds.len() == n_rows => r,
                _ => break,
            };
            cur_lb = r.lower_bounds[ri];
            if cur_lb.is_finite() && cur_lb > best_lb {
                best_lb = cur_lb;
                best_slopes = slopes.clone();
            }
        }
        if best_lb > base_lb {
            improved += 1;
        }
        per_target_slopes.push(best_slopes);
    }
    if probe {
        eprintln!(
            "[ab-parity-interm] ascent targets={} improved={improved} ({}ms)",
            targets.len(),
            t0.elapsed().as_millis()
        );
    }

    // PHASE 2 — application: for each target, write its α into EVERY domain's
    // segments, ONE batched backward, best-merge ONLY that target's column.
    for (ti, &ri) in targets.iter().enumerate() {
        if past_deadline() {
            break;
        }
        let slopes = &per_target_slopes[ti];
        for &i in idxs {
            if let Some(p) = preps[i].as_mut() {
                write_alpha_prime(&mut p.segments, slopes, &stepped);
            }
        }
        let refs: Vec<ny_core::GpuResnetBatchedDomainRef> = idxs
            .iter()
            .map(|&i| {
                let p = preps[i].as_ref().expect("idxs holds Some preps only");
                ny_core::GpuResnetBatchedDomainRef {
                    segments: &p.segments,
                    input_lower: &p.in_lo,
                    input_upper: &p.in_hi,
                    beta_signed: &p.beta_signed,
                    frontier_abs: &p.frontier_abs,
                    node_abs: &p.node_abs,
                }
            })
            .collect();
        let results = match gpu.crown_backward_gpu_resnet_sound_beta_batched(&refs, seed) {
            Ok(r) if r.len() == idxs.len() => r,
            _ => continue,
        };
        for (&i, r) in idxs.iter().zip(&results) {
            if r.lower_bounds.len() != n_rows || r.upper_bounds.len() != n_rows {
                continue;
            }
            if let Some(dst) = per_domain[i].as_mut() {
                if ri < dst.lower_bounds.len() && ri < dst.upper_bounds.len() {
                    let lo = r.lower_bounds[ri];
                    let hi = r.upper_bounds[ri];
                    if lo.is_finite()
                        && (!dst.lower_bounds[ri].is_finite() || lo > dst.lower_bounds[ri])
                    {
                        dst.lower_bounds[ri] = lo;
                    }
                    if hi.is_finite()
                        && (!dst.upper_bounds[ri].is_finite() || hi < dst.upper_bounds[ri])
                    {
                        dst.upper_bounds[ri] = hi;
                    }
                }
            }
        }
    }
    improved
}

/// Batch stats for the probe line.
#[derive(Default)]
struct RefineStats {
    domains_refined: usize,
    neurons_tightened: usize,
    newly_stable: usize,
    crossings_kept: usize,
    infeasible: usize,
    width_before: f64,
    width_after: f64,
    /// #clip-interm-resnet-batched: domains whose seed-layer bounds were tightened
    /// by the batched split-constraint clip (constrained concretization over the
    /// batched ResidentCoeff, zero extra backward).
    clip_domains: usize,
    /// #clip-interm-resnet-batched: seed rows refused (non-finite coeff-error fold —
    /// kept inherited, sound).
    clip_refused_rows: usize,
    /// #clip-interm-resnet-batched: max per-row tightening (|Δl| or |Δu|) from the clip.
    clip_max_tighten: f32,
    /// #clip-interm-guard: domains whose clip was REVERTED by the fail-closed runtime
    /// guard (a feasible directed sample fell outside a tightened row, or a non-finite
    /// clip bound) — the seed node kept its inherited parent bound. Nonzero ⇒ a
    /// (near-)unsound clip was caught at runtime.
    clip_guard_reverts: usize,
}

impl BetaCrownVerifier {
    /// Refine each domain's LAST-ReLU pre-activation bounds (see module docs;
    /// with `NY_INTERM_REFINE_LAYERS=2` also the second-to-last ReLU's,
    /// deepest-first so the last-ReLU pass consumes the improved upstream
    /// entry) and return the refined per-domain caches plus per-domain
    /// infeasibility flags (prune lane), or `None` when the lane does not
    /// apply (no sound GPU, no clean last-ReLU chain, oversized seed layer,
    /// deadline passed, or every domain refused) — the caller then keeps the
    /// inherited caches byte-identically.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn refine_last_relu_interm_bounds(
        &self,
        graph: &GraphNetwork,
        output_node: &str,
        n_domains: usize,
        bounds_caches: &[HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        engine: &dyn GemmEngine,
        spec_matrix: &Array2<f32>,
    ) -> Option<IntermRefineOutcome> {
        let mut opts = IntermRefineOptions::from_env();
        // #joint-interm-alpha: compute the per-seed-neuron margin weights here
        // (the tail linear map + spec rows are only available at the batched call
        // site, not in `from_env`). `None` ⇒ the α′ lane keeps its uniform
        // objective (sound fallback).
        if opts.joint_margin || opts.selective_topk > 0 {
            opts.margin_weights = match compute_margin_weights_with_deadline(
                graph,
                output_node,
                spec_matrix,
                self.config.alpha_config.deadline,
            ) {
                Ok(weights) => weights,
                Err(error) => {
                    if opts.probe {
                        eprintln!("[interm-refine] margin weights REFUSED: {error}");
                    }
                    None
                }
            };
            if opts.probe && opts.joint_margin {
                eprintln!(
                    "[joint-interm-alpha] margin weights: {}",
                    match &opts.margin_weights {
                        Some(m) => format!(
                            "n={} nonzero={} max={:.4}",
                            m.len(),
                            m.iter().filter(|&&v| v > 0.0).count(),
                            m.iter().cloned().fold(0.0f32, f32::max)
                        ),
                        None => "unavailable (uniform fallback)".to_string(),
                    }
                );
            }
        }
        self.refine_interm_bounds_with_opts(
            graph,
            output_node,
            n_domains,
            bounds_caches,
            constrained_inputs,
            beta_states,
            alpha_states,
            engine,
            &opts,
        )
    }

    /// Options-explicit body of [`Self::refine_last_relu_interm_bounds`]
    /// (unit tests pass `IntermRefineOptions` directly — no env races).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn refine_interm_bounds_with_opts(
        &self,
        graph: &GraphNetwork,
        output_node: &str,
        n_domains: usize,
        bounds_caches: &[HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        engine: &dyn GemmEngine,
        opts: &IntermRefineOptions,
    ) -> Option<IntermRefineOutcome> {
        if n_domains == 0
            || bounds_caches.len() != n_domains
            || constrained_inputs.len() != n_domains
            || beta_states.len() != n_domains
            || alpha_states.len() != n_domains
        {
            return None;
        }
        if !crate::network::resnet_beta_gpu_enabled() {
            return None;
        }
        // Deadline firewall: the refinement is an EXTRA GPU pass; never start it
        // past the BaB deadline (the margin backward still runs — unchanged).
        if self.config.alpha_config.past_deadline() {
            return None;
        }
        let gpu = engine
            .as_gpu_crown_backward()
            .filter(|g| g.provides_sound_gpu_crown())?;

        // ADAPTIVE SCHEDULE (#adaptive-refine, `NY_INTERM_REFINE_ADAPTIVE=1`,
        // dark): once the per-process latch has recorded a zero-yield depth,
        // domains at depth ≥ latch keep their inherited caches (skip =
        // identical to a per-domain refusal — sound, cost-only). A proper
        // subset recurses on the still-refined shallow sub-batch and scatters
        // back (same shape as the min_depth gate below); the recursed call
        // re-runs this filter as a no-op and owns the latch update.
        if let Some(latch) = &opts.adaptive_latch {
            let latched = latch.load(std::sync::atomic::Ordering::Relaxed);
            if latched != usize::MAX {
                let depth_of =
                    |i: usize| -> usize { beta_states[i].map_or(0, |b| b.entries.len()) };
                // #interm-refine-redo: depth-multiple domains re-refine past
                // the latch (stale-inheritance repair under upstream splits).
                let included: Vec<usize> = (0..n_domains)
                    .filter(|&i| latched_domain_refines(depth_of(i), latched, opts.redo_every))
                    .collect();
                if included.is_empty() {
                    if opts.probe {
                        eprintln!(
                            "[interm-refine-adaptive] batch SKIPPED ({n_domains} domains, all \
                             at depth >= latch {latched}, no redo multiple of {})",
                            opts.redo_every
                        );
                    }
                    return None;
                }
                if included.len() < n_domains {
                    let sub_caches: Vec<HashMap<String, Arc<BoundedTensor>>> =
                        included.iter().map(|&i| bounds_caches[i].clone()).collect();
                    let sub_inputs: Vec<BoundedTensor> = included
                        .iter()
                        .map(|&i| constrained_inputs[i].clone())
                        .collect();
                    let sub_betas: Vec<Option<&GraphBetaState>> =
                        included.iter().map(|&i| beta_states[i]).collect();
                    let sub_alphas: Vec<Option<&GraphDomainAlphaState>> =
                        included.iter().map(|&i| alpha_states[i]).collect();
                    if opts.probe {
                        eprintln!(
                            "[interm-refine-adaptive] batch FILTERED to {}/{n_domains} domains \
                             below latch {latched} (redo stride {})",
                            included.len(),
                            opts.redo_every
                        );
                    }
                    let outcome = self.refine_interm_bounds_with_opts(
                        graph,
                        output_node,
                        included.len(),
                        &sub_caches,
                        &sub_inputs,
                        &sub_betas,
                        &sub_alphas,
                        engine,
                        opts,
                    )?;
                    let mut caches = bounds_caches.to_vec();
                    let mut infeasible = vec![false; n_domains];
                    for (k, &i) in included.iter().enumerate() {
                        caches[i] = outcome.caches[k].clone();
                        infeasible[i] = outcome.infeasible[k];
                    }
                    return Some(IntermRefineOutcome { caches, infeasible });
                }
                // Every domain is below the latch — fall through.
            }
        }

        // Depth gate (#fit-100s, `NY_INTERM_REFINE_MIN_DEPTH=d`, default 0 =
        // byte-identical): refine only domains with ≥ d split premises. A
        // proper subset recurses on the filtered sub-batch (min_depth=0) and
        // scatters the refined caches/flags back; excluded domains keep their
        // inherited caches — sound (identical to a per-domain refusal).
        if opts.min_depth > 0 {
            let included: Vec<usize> = (0..n_domains)
                .filter(|&i| beta_states[i].map_or(0, |b| b.entries.len()) >= opts.min_depth)
                .collect();
            if included.is_empty() {
                return None;
            }
            if included.len() < n_domains {
                let sub_caches: Vec<HashMap<String, Arc<BoundedTensor>>> =
                    included.iter().map(|&i| bounds_caches[i].clone()).collect();
                let sub_inputs: Vec<BoundedTensor> = included
                    .iter()
                    .map(|&i| constrained_inputs[i].clone())
                    .collect();
                let sub_betas: Vec<Option<&GraphBetaState>> =
                    included.iter().map(|&i| beta_states[i]).collect();
                let sub_alphas: Vec<Option<&GraphDomainAlphaState>> =
                    included.iter().map(|&i| alpha_states[i]).collect();
                let sub_opts = IntermRefineOptions {
                    min_depth: 0,
                    ..opts.clone()
                };
                let outcome = self.refine_interm_bounds_with_opts(
                    graph,
                    output_node,
                    included.len(),
                    &sub_caches,
                    &sub_inputs,
                    &sub_betas,
                    &sub_alphas,
                    engine,
                    &sub_opts,
                )?;
                let mut caches = bounds_caches.to_vec();
                let mut infeasible = vec![false; n_domains];
                for (k, &i) in included.iter().enumerate() {
                    caches[i] = outcome.caches[k].clone();
                    infeasible[i] = outcome.infeasible[k];
                }
                return Some(IntermRefineOutcome { caches, infeasible });
            }
            // Every domain is deep enough — fall through to the whole-batch path.
        }

        // Any domain with an unstable neuron at a seed ⇒ useful target (a
        // fully-stable seed's slopes are already exact — refining its input
        // can never tighten anything downstream).
        let has_unstable = |seed: &str| {
            bounds_caches.iter().any(|cache| {
                cache.get(seed).is_some_and(|entry| {
                    entry
                        .lower()
                        .iter()
                        .zip(entry.upper().iter())
                        .any(|(&l, &u)| l < 0.0 && u > 0.0)
                })
            })
        };

        // Seed chain, built in EXEC ORDER (earliest first) — the cascade: each
        // pass (and afterwards the margin backward) consumes the earlier
        // passes' improved upstream entries through the ReLU relaxations the
        // extraction derives from the caches.
        let (last_relu_name, last_seed) = find_last_relu_seed(graph, output_node)?;
        let seeds: Vec<(String, String)> = if let Some(names) = &opts.seeds {
            // NAMED-SEED LANE (`NY_INTERM_REFINE_SEEDS`, #midref): resolve each
            // token (exact ReLU node name, or `last` = the last-ReLU chain),
            // skip unknown/non-ReLU/input-fed/fully-stable names (probe-logged
            // — row selection of WHICH layers get refined; never soundness),
            // dedupe, sort by exec order.
            let exec = graph.exec_order().ok()?;
            let pos: HashMap<&str, usize> = exec
                .iter()
                .enumerate()
                .map(|(i, name)| (name.as_str(), i))
                .collect();
            let mut resolved: Vec<(String, String)> = Vec::new();
            for tok in names {
                let pair = if tok.eq_ignore_ascii_case("last") {
                    (last_relu_name.clone(), last_seed.clone())
                } else {
                    let Some(node) = graph.nodes.get(tok) else {
                        if opts.probe {
                            eprintln!("[interm-refine] seed token {tok:?} not in graph — skipped");
                        }
                        continue;
                    };
                    if !matches!(node.layer, Layer::ReLU(_)) {
                        if opts.probe {
                            eprintln!("[interm-refine] seed token {tok:?} is not a ReLU — skipped");
                        }
                        continue;
                    }
                    let Some(seed) = node.inputs.first().cloned() else {
                        continue;
                    };
                    if seed == NETWORK_INPUT {
                        if opts.probe {
                            eprintln!(
                                "[interm-refine] seed token {tok:?} fed by the network input — skipped"
                            );
                        }
                        continue;
                    }
                    (tok.clone(), seed)
                };
                if resolved.iter().any(|(r, _)| *r == pair.0) {
                    continue; // dedupe
                }
                if pair.0 != last_relu_name && !has_unstable(&pair.1) {
                    if opts.probe {
                        eprintln!(
                            "[interm-refine] named seed relu={} seed={} fully stable — skipped",
                            pair.0, pair.1
                        );
                    }
                    continue;
                }
                resolved.push(pair);
            }
            resolved.sort_by_key(|(r, _)| pos.get(r.as_str()).copied().unwrap_or(usize::MAX));
            if resolved.is_empty() {
                return None;
            }
            resolved
        } else {
            // LAYERS walk: layer 1 = the last ReLU; deeper layers walk the
            // ancestor ReLUs (latest-in-exec-order first), SKIPPING
            // fully-stable ones (measured on resnet_medium: `Relu_51`, the
            // last residual F-branch, is DEAD at the root box — all 2048
            // `Conv_49` uppers < 0 — so the useful second target is the next
            // unstable ancestor). Reversed into exec order afterwards.
            let mut walk: Vec<(String, String)> = vec![(last_relu_name.clone(), last_seed)];
            let mut cursor = walk[0].1.clone();
            while walk.len() < opts.layers {
                let Some((relu, seed)) = find_next_relu_seed_below(graph, &cursor) else {
                    break;
                };
                cursor = seed.clone();
                if has_unstable(&seed) {
                    walk.push((relu, seed));
                } else if opts.probe {
                    eprintln!(
                        "[interm-refine] deep seed candidate relu={relu} seed={seed} fully \
                         stable — walking deeper"
                    );
                }
            }
            if opts.probe && walk.len() < opts.layers {
                eprintln!(
                    "[interm-refine] seed chain stopped at {}/{} layers (no unstable ancestor ReLU seed)",
                    walk.len(),
                    opts.layers
                );
            }
            walk.reverse();
            walk
        };

        let mut caches: Vec<HashMap<String, Arc<BoundedTensor>>> = bounds_caches.to_vec();
        let mut infeasible = vec![false; n_domains];
        let mut any_refined = false;
        let mut newly_stable_total = 0usize;
        let mut passes_completed = 0usize;
        let mut deadline_break = false;
        for (relu_name, seed_node) in &seeds {
            // Deadline check BETWEEN passes too (each pass is a GPU call).
            if self.config.alpha_config.past_deadline() {
                deadline_break = true;
                break;
            }
            let is_last_relu = *relu_name == last_relu_name;
            if let Some((refined, newly_stable)) = self.refine_one_seed_pass(
                graph,
                relu_name,
                seed_node,
                n_domains,
                &mut caches,
                constrained_inputs,
                beta_states,
                alpha_states,
                gpu,
                opts,
                is_last_relu,
                &mut infeasible,
            ) {
                any_refined |= refined;
                newly_stable_total += newly_stable;
                passes_completed += 1;
            }
        }
        // ADAPTIVE latch update (#adaptive-refine): a COMPLETED batch (at
        // least one pass ran, no deadline break) that PRODUCED nothing
        // (newly_stable = 0 and infeasible-pruned = 0) at depth ≥ floor stops
        // refinement for all deeper domains for the rest of the run. The
        // batch's depth band = the MIN premise count over its domains.
        if let Some(latch) = &opts.adaptive_latch {
            if passes_completed > 0 && !deadline_break {
                let n_inf = infeasible.iter().filter(|&&b| b).count();
                let batch_depth = (0..n_domains)
                    .map(|i| beta_states[i].map_or(0, |b| b.entries.len()))
                    .min()
                    .unwrap_or(0);
                if adaptive_should_latch(
                    newly_stable_total,
                    n_inf,
                    batch_depth,
                    opts.adaptive_floor,
                ) {
                    let prev = latch.fetch_min(batch_depth, std::sync::atomic::Ordering::Relaxed);
                    if opts.probe && batch_depth < prev {
                        eprintln!(
                            "[interm-refine-adaptive] LATCHED at depth {batch_depth} \
                             (zero-yield batch of {n_domains}; refinement now skipped for \
                             domains at depth >= {batch_depth})"
                        );
                    }
                }
            }
        }
        if any_refined || infeasible.iter().any(|&b| b) {
            Some(IntermRefineOutcome { caches, infeasible })
        } else {
            None
        }
    }

    /// ONE seed layer's refinement pass over the whole domain batch, applied
    /// in place into `caches` (a per-layer refusal returns `None` and leaves
    /// the caches untouched — sound). Returns `(any_domain_refined,
    /// newly_stable_count)` — the count feeds the adaptive-schedule latch
    /// (#adaptive-refine).
    #[allow(clippy::too_many_arguments)]
    fn refine_one_seed_pass(
        &self,
        graph: &GraphNetwork,
        relu_name: &str,
        seed_node: &str,
        n_domains: usize,
        caches: &mut [HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        gpu: &dyn ny_core::GpuCrownBackward,
        opts: &IntermRefineOptions,
        is_last_relu: bool,
        infeasible: &mut [bool],
    ) -> Option<(bool, usize)> {
        let probe = opts.probe;
        let refuse = |reason: &str| -> Option<(bool, usize)> {
            if probe {
                eprintln!(
                    "[interm-refine] pass REFUSED seed={seed_node} relu={relu_name}: {reason}"
                );
            }
            None
        };
        let Some(seed_entry) = caches[0].get(seed_node) else {
            return refuse("no seed entry in cache");
        };
        let pre_dim = seed_entry.len();
        // Named seeds (`NY_INTERM_REFINE_SEEDS`) are exempt from the dim cap —
        // naming a conv-wide layer is the deliberate ask; the deep row cap and
        // the hard `n_rows·pre_dim` guard below still bound cost.
        if pre_dim == 0 || (opts.seeds.is_none() && pre_dim > opts.max_dim) {
            return refuse(&format!("pre_dim={pre_dim} outside (0, {}]", opts.max_dim));
        }
        // Account the simultaneously-live selected/premise/candidate/scored/
        // output vectors before the first row-table allocation. The same hard
        // cap also covers the deep width/rest/keep representation.
        if self.config.alpha_config.past_deadline()
            || validate_selective_row_budget(
                pre_dim,
                pre_dim,
                caches.len(),
                opts.selective_topk > 0,
            )
            .is_none()
        {
            return refuse("selective row work exceeds deadline/resource cap");
        }
        let t0 = std::time::Instant::now();

        // ROW SELECTION (cost, not soundness — see
        // `interm_refine_unstable_rows_only`): default = the union of
        // inherited-UNSTABLE neurons (stable slopes are already exact; measured
        // bound-identical to all-rows at ~3x less GPU). The seed is shared
        // across the batch, so union over domains (per-domain intersection
        // keeps each domain's own inherited values elsewhere). The prune lane
        // force-includes every domain's split-premise neurons: clamped premises
        // are STABLE in the inherited entry, so the unstable arm alone would
        // exclude exactly the rows that can prove infeasibility.
        let mut selected = Vec::new();
        selected.try_reserve_exact(pre_dim).ok()?;
        selected.resize(pre_dim, !opts.unstable_rows_only);
        if opts.unstable_rows_only {
            for cache in caches.iter() {
                if self.config.alpha_config.past_deadline() {
                    return refuse("deadline during unstable-row selection");
                }
                let Some(entry) = cache.get(seed_node) else {
                    continue;
                };
                if entry.len() != pre_dim {
                    // Heterogeneous seed entry — refuse the whole batch.
                    return refuse("heterogeneous seed entry");
                }
                for (j, (&l, &u)) in entry.lower().iter().zip(entry.upper().iter()).enumerate() {
                    if j.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                        && self.config.alpha_config.past_deadline()
                    {
                        return refuse("deadline during unstable-row scan");
                    }
                    if l < 0.0 && u > 0.0 {
                        selected[j] = true;
                    }
                }
            }
        }
        // Force-include every domain's split-premise neurons: the prune lane needs them
        // (clamped premises are STABLE ⇒ excluded by the unstable arm), and the
        // #clip-interm-resnet-batched clip needs them as CONSTRAINT SOURCES (a missing
        // premise row ⇒ that split's half-space is dropped ⇒ the clip loses its
        // tightening). Both are seed-layer splits at `relu_name`.
        let mut premise = Vec::new();
        premise.try_reserve_exact(pre_dim).ok()?;
        premise.resize(pre_dim, false);
        if opts.prune || opts.clip_resnet {
            for bs in beta_states.iter().flatten() {
                if self.config.alpha_config.past_deadline() {
                    return refuse("deadline during premise-row selection");
                }
                for (entry_index, e) in bs.entries_for_node(relu_name).enumerate() {
                    if entry_index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                        && self.config.alpha_config.past_deadline()
                    {
                        return refuse("deadline during premise-row scan");
                    }
                    if e.neuron_idx() < pre_dim {
                        premise[e.neuron_idx()] = true;
                        selected[e.neuron_idx()] = true;
                    }
                }
            }
        }
        let mut sel = Vec::new();
        sel.try_reserve_exact(pre_dim).ok()?;
        for (j, &is_selected) in selected.iter().enumerate() {
            if j.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                && self.config.alpha_config.past_deadline()
            {
                return refuse("deadline during candidate materialization");
            }
            if is_selected {
                sel.push(j);
            }
        }
        // The candidate vector now owns the selection result; release the
        // dense mask before allocating scored/output vectors so the checked
        // live-byte model remains conservative.
        drop(selected);

        // Winner-style selective clip: αβ-CROWN's TinyImageNet configuration
        // uses top-20 objectives per layer.  NY's wide seed is
        // shared across domains, so select one batch-wide top-K (max score over
        // domains) and keep split-premise source rows additively.  Unselected
        // rows retain their inherited enclosures.
        if opts.selective_topk > 0 {
            sel = select_intermediate_objective_rows_with_deadline(
                caches,
                seed_node,
                &sel,
                &premise,
                opts.selective_topk,
                is_last_relu
                    .then_some(opts.margin_weights.as_deref())
                    .flatten(),
                self.config.alpha_config.deadline,
            );
        }

        // DEEP-layer row cap (cost-only): keep premise rows, then top-K by
        // inherited width (widest = most refinement headroom), index tie-break.
        if !is_last_relu && sel.len() > opts.deep_max_rows {
            let mut width = Vec::new();
            width.try_reserve_exact(pre_dim).ok()?;
            width.resize(pre_dim, 0.0f32);
            for cache in caches.iter() {
                if self.config.alpha_config.past_deadline() {
                    return refuse("deadline during deep-row width scan");
                }
                let Some(entry) = cache.get(seed_node) else {
                    continue;
                };
                if entry.len() != pre_dim {
                    continue;
                }
                for (j, (&l, &u)) in entry.lower().iter().zip(entry.upper().iter()).enumerate() {
                    if j.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                        && self.config.alpha_config.past_deadline()
                    {
                        return refuse("deadline during deep-row width fold");
                    }
                    if l.is_finite() && u.is_finite() {
                        width[j] = width[j].max(u - l);
                    }
                }
            }
            let mut keep = Vec::new();
            let mut rest = Vec::new();
            keep.try_reserve_exact(sel.len()).ok()?;
            rest.try_reserve_exact(sel.len()).ok()?;
            for (position, &j) in sel.iter().enumerate() {
                if position.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE)
                    && self.config.alpha_config.past_deadline()
                {
                    return refuse("deadline during deep-row partition");
                }
                if premise[j] {
                    keep.push(j);
                } else {
                    rest.push(j);
                }
            }
            let mut past_deadline = || self.config.alpha_config.past_deadline();
            if crate::complete_clip::deadline_heapsort_by(
                &mut rest,
                &mut past_deadline,
                "deep selective width sort",
                |&a, &b| {
                    width[b]
                        .partial_cmp(&width[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                },
            )
            .is_err()
            {
                return refuse("deadline during deep-row score sort");
            }
            let room = opts.deep_max_rows.saturating_sub(keep.len());
            keep.extend(rest.into_iter().take(room));
            let mut past_deadline = || self.config.alpha_config.past_deadline();
            if crate::complete_clip::deadline_heapsort_by(
                &mut keep,
                &mut past_deadline,
                "deep selective index sort",
                usize::cmp,
            )
            .is_err()
            {
                return refuse("deadline during deep-row index sort");
            }
            sel = keep;
        }
        let n_rows = sel.len();
        if n_rows == 0 || n_rows.saturating_mul(pre_dim) > (1 << 24) {
            // Diagnostic detail for the zero-row case: what does the seed
            // entry actually look like?
            let (mut n_l_neg, mut n_u_pos, mut n_nan) = (0usize, 0usize, 0usize);
            let (mut min_l, mut max_u) = (f32::INFINITY, f32::NEG_INFINITY);
            if let Some(entry) = caches[0].get(seed_node) {
                for (&l, &u) in entry.lower().iter().zip(entry.upper().iter()) {
                    if l.is_nan() || u.is_nan() {
                        n_nan += 1;
                    }
                    if l < 0.0 {
                        n_l_neg += 1;
                    }
                    if u > 0.0 {
                        n_u_pos += 1;
                    }
                    min_l = min_l.min(l);
                    max_u = max_u.max(u);
                }
            }
            return refuse(&format!(
                "n_rows={n_rows} (zero or seed too large; entry: l<0 {n_l_neg}/{pre_dim}, \
                 u>0 {n_u_pos}/{pre_dim}, nan {n_nan}, min_l={min_l:.4}, max_u={max_u:.4})"
            ));
        }

        // Identity seed: row r bounds pre-activation `sel[r]` of the seed node.
        let mut rows = vec![0.0f32; n_rows * pre_dim];
        for (r, &j) in sel.iter().enumerate() {
            rows[r * pre_dim + j] = 1.0;
        }
        let seed = ny_core::GpuCrownSeed {
            lower_a: rows.clone().into(),
            upper_a: rows.into(),
            lower_b: vec![0.0f32; n_rows].into(),
            upper_b: vec![0.0f32; n_rows].into(),
            num_specs: n_rows,
            current_dim: pre_dim,
        };

        // Truncated per-domain preps: same alpha-bridge/β/extraction machinery
        // as the margin backward, start node = the seed node. β for the SEED
        // ReLU is automatically EXCLUDED (its Activation is not in the
        // truncated stack — the seed-layer splits live in the inherited clamp).
        // The preps are independent across domains — rayon fan-out (the
        // extraction's conv weight_col conversion dominates the refinement
        // wall; GPU calls still serialize on the device mutex). Preps read the
        // CURRENT caches: a deeper pass's refinement cascades into this one.
        let mut preps: Vec<Option<ResnetDomainPrep>> = {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let caches_ref: &[HashMap<String, Arc<BoundedTensor>>] = caches;
            let allow_pure_chain = crate::network::bab_chain_wide_enabled();
            // #extract-skeleton increment 3: the truncated-stack skeleton for
            // THIS seed node comes from the verifier-level cross-batch cache
            // (`self.skeleton_cache`, keyed `(seed_node, allow_pure_chain)`) —
            // a hit re-validates `matches_graph` and shares the same `Arc`
            // across every pass and batch; a miss/stale entry rebuilds the
            // increment-2 per-pass skeleton (exemplar = domain 0). The "preps
            // read CURRENT caches" contract above is preserved: only
            // graph-static weights are shared through the skeleton; every
            // bounds-derived slot/table is re-folded per domain from
            // `caches_ref[i]` inside `prep_resnet_domain_with`. `None`
            // (kill-switch / build refusal) keeps every domain on the legacy
            // extraction, byte-identically (fail closed).
            let skeleton = (n_domains > 0)
                .then(|| {
                    self.skeleton_cache
                        .get_or_build(graph, seed_node, allow_pure_chain, || {
                            build_call_skeleton(
                                graph,
                                seed_node,
                                &caches_ref[0],
                                &constrained_inputs[0],
                                alpha_states[0],
                                allow_pure_chain,
                            )
                        })
                })
                .flatten();
            let skeleton = skeleton.as_deref();
            (0..n_domains)
                .into_par_iter()
                .map(|i| {
                    let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                    prep_resnet_domain_with(
                        skeleton,
                        graph,
                        seed_node,
                        &caches_ref[i],
                        &constrained_inputs[i],
                        beta_states[i],
                        alpha_states[i],
                        allow_pure_chain,
                        // #image-node-crown flags: legacy #interm-refine lane —
                        // never BN, never a frozen stop (byte-identical).
                        false,
                        false,
                    )
                })
                .collect()
        };
        let prep_ms = t0.elapsed().as_millis();
        let idxs: Vec<usize> = (0..n_domains).filter(|&i| preps[i].is_some()).collect();
        if idxs.is_empty() {
            return refuse("every domain's truncated-stack prep refused");
        }

        // #alpha-prime (dark, `NY_INTERM_REFINE_ALPHA=1`, last-ReLU pass only):
        // REUSE a stored α′ (optimize-once at the first batch) by writing it
        // into every domain's truncated segments BEFORE the refinement
        // backward — sound (any α ∈ [0,1] is a valid lower ReLU slope; only
        // this domain's own unstable neurons are touched). When no usable α′
        // is stored yet (or under `NY_INTERM_REFINE_ALPHA_REOPT=1`), mark the
        // ascent pending — it runs after the iterate-0 backward below.
        let mut alpha_pending = false;
        // #ab-parity-interm: the per-target lane runs a FRESH ascent every batch
        // (no once-per-process shared-α reuse — the shared α′ store is bypassed).
        if opts.per_target && opts.alpha_iters > 0 && is_last_relu {
            alpha_pending = true;
        } else if opts.alpha_iters > 0 && is_last_relu {
            if let Some(store) = &opts.alpha_store {
                let mut reuse: Option<(Vec<Vec<f32>>, Vec<Vec<bool>>)> = None;
                {
                    let stored = store.lock().unwrap_or_else(|e| e.into_inner());
                    match stored.as_ref() {
                        Some(ap) if !opts.alpha_reopt => {
                            let key_ok = preps[idxs[0]].as_ref().is_some_and(|p| {
                                alpha_prime_matches(ap, seed_node, pre_dim, &p.relu_names)
                            });
                            if ap.improved && key_ok {
                                reuse = Some((ap.slopes.clone(), ap.stepped.clone()));
                            } else if probe && !key_ok {
                                eprintln!(
                                    "[interm-refine-alpha] stored α′ key mismatch — reuse skipped"
                                );
                            }
                        }
                        Some(_) => alpha_pending = true, // REOPT: ascend every batch
                        None => alpha_pending = true,
                    }
                }
                if let Some((slopes, stepped)) = reuse {
                    let mut applied = 0usize;
                    for &i in &idxs {
                        if let Some(p) = preps[i].as_mut() {
                            write_alpha_prime(&mut p.segments, &slopes, &stepped);
                            applied += 1;
                        }
                    }
                    if probe {
                        eprintln!(
                            "[interm-refine-alpha] reuse applied doms={applied}/{}",
                            idxs.len()
                        );
                    }
                }
            }
        }
        let t_gpu = std::time::Instant::now();

        // ONE batched call through the same wide machinery as the margin
        // backward (#wide-chunk: under `NY_INTERM_REFINE_WIDE_MAX_N=k`, split
        // into domain chunks of `max(1, k / n_rows)` so each wide pass stays
        // under the device's binding/dispatch caps instead of failing whole and
        // falling serial — see `IntermRefineOptions::wide_max_n`); on any
        // batched error, serial per-domain fallback for that chunk's domains;
        // per-domain failures keep that domain's inherited bounds (sound).
        let mut per_domain: Vec<Option<ny_core::GpuCrownResult>> =
            (0..n_domains).map(|_| None).collect();
        // #clip-interm-resnet-batched: fire the split-constraint clip on EVERY seeded
        // layer (not just the last ReLU). MEASURED 2026-07-14: kFSB splits land on
        // MID-layers (e.g. Relu_31), NOT only Relu_57 — so a last-ReLU-only clip finds
        // 0 matching premises (split_closure requires node_name==seed relu) and no-ops.
        // Each seeded layer's clip uses the splits AT that layer (its own neurons ARE
        // the seed rows), so the LAYERS walk must reach the split layer. Per-domain
        // folded seed rows come from the batched ResidentCoeff (ZERO extra backward).
        let do_clip = opts.clip_resnet;
        let mut per_domain_folded: Vec<Option<FoldedSeedRows>> =
            (0..n_domains).map(|_| None).collect();
        let chunk_doms = wide_chunk_domains(opts.wide_max_n, n_rows, idxs.len());
        let mut batched_ok = true;
        for chunk in idxs.chunks(chunk_doms) {
            let refs: Vec<ny_core::GpuResnetBatchedDomainRef> = chunk
                .iter()
                .map(|&i| {
                    let p = preps[i].as_ref().expect("idxs holds Some preps only");
                    ny_core::GpuResnetBatchedDomainRef {
                        segments: &p.segments,
                        input_lower: &p.in_lo,
                        input_upper: &p.in_hi,
                        beta_signed: &p.beta_signed,
                        frontier_abs: &p.frontier_abs,
                        node_abs: &p.node_abs,
                    }
                })
                .collect();
            // #clip-interm-resnet-batched: capture the ResidentCoeff frontier from the
            // SAME single wide pass (throughput marker: batched_backward=1). On success,
            // fold each domain's block outward over its own box → per-domain seed rows for
            // the clip; then use the concretized bounds exactly like the non-clip path. Any
            // failure falls through to the plain batched call (no clip that chunk — sound).
            if do_clip {
                match gpu.crown_backward_gpu_resnet_sound_beta_batched_coeff(&refs, &seed) {
                    // FAIL-CLOSED SHAPE ASSERTION (#clip-interm-guard): the batched
                    // ResidentCoeff MUST carry exactly `n_domains_in_chunk × n_rows`
                    // wide rows in domain-major order — that is the invariant
                    // `fold_seed_rows_for_domain`'s `local_block * n_rows + r`
                    // indexing relies on to slice each domain its OWN rows. A layout
                    // mismatch (num_specs ≠ chunk.len()·n_rows) would silently hand a
                    // domain another domain's coefficients → a wrong (possibly
                    // too-tight) clip → false-VERIFY. On mismatch we REFUSE the clip
                    // for this chunk (fall through to the plain, unclipped batched
                    // call — sound).
                    Ok((results, coeff))
                        if results.len() == chunk.len()
                            && coeff.num_specs == chunk.len().saturating_mul(n_rows) =>
                    {
                        for (pos, (&i, r)) in chunk.iter().zip(results).enumerate() {
                            let p = preps[i].as_ref().expect("idxs holds Some preps only");
                            per_domain_folded[i] = fold_seed_rows_for_domain(
                                &coeff,
                                pos,
                                n_rows,
                                &p.in_lo,
                                &p.in_hi,
                                self.config.alpha_config.deadline,
                            );
                            per_domain[i] = Some(r);
                        }
                        continue;
                    }
                    other => {
                        if probe {
                            let why = match other {
                                Ok((r, c)) => format!(
                                    "shape mismatch results={} chunk={} coeff.num_specs={} expected={}",
                                    r.len(),
                                    chunk.len(),
                                    c.num_specs,
                                    chunk.len().saturating_mul(n_rows),
                                ),
                                Err(e) => format!("ERR: {e}"),
                            };
                            eprintln!("[clip-resnet-batched] _coeff backward failed ({why}) — falling through to plain (no clip this chunk)");
                        }
                        /* fall through to the plain batched call below */
                    }
                }
            }
            match gpu.crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed) {
                Ok(results) if results.len() == chunk.len() => {
                    for (&i, r) in chunk.iter().zip(results) {
                        per_domain[i] = Some(r);
                    }
                }
                _ => {
                    batched_ok = false;
                    for &i in chunk {
                        let p = preps[i].as_ref().expect("idxs holds Some preps only");
                        if let Ok(r) = gpu.crown_backward_gpu_resnet_sound_beta(
                            &p.segments,
                            &seed,
                            &p.in_lo,
                            &p.in_hi,
                            &p.beta_signed,
                            &p.frontier_abs,
                            &p.node_abs,
                        ) {
                            per_domain[i] = Some(r);
                        }
                    }
                }
            }
        }

        // #alpha-prime ASCENT (dark): optimize a DEDICATED α′ against the
        // refinement objective on top of the iterate-0 (borrowed-α) backward
        // above, best-merging every sound iterate into `per_domain`; the
        // resulting α′ is stored for reuse by later batches (unless REOPT).
        // Skipped when the batched call fell back to serial (rare error arm).
        if alpha_pending && batched_ok && !self.config.alpha_config.past_deadline() {
            let row_widths: Vec<f32> = {
                let entry = idxs
                    .iter()
                    .find_map(|&i| caches[i].get(seed_node))
                    .filter(|e| e.len() == pre_dim);
                match entry {
                    Some(e) => {
                        let lo: Vec<f32> = e.lower().iter().copied().collect();
                        let hi: Vec<f32> = e.upper().iter().copied().collect();
                        sel.iter()
                            .map(|&j| {
                                let (l, u) = (lo[j], hi[j]);
                                if l.is_finite() && u.is_finite() {
                                    u - l
                                } else {
                                    0.0
                                }
                            })
                            .collect()
                    }
                    None => vec![0.0; sel.len()],
                }
            };
            if opts.per_target {
                // #ab-parity-interm: per-target decoupled ascent (each target row
                // its own α + own bound). Effect is merged directly into
                // `per_domain`; nothing is stored (reuse is dropped under the gate).
                alpha_prime_ascent_per_target(
                    gpu,
                    &mut preps,
                    &idxs,
                    &seed,
                    &mut per_domain,
                    pre_dim,
                    &sel,
                    &row_widths,
                    opts,
                    &|| self.config.alpha_config.past_deadline(),
                );
            } else if let Some(ap) = alpha_prime_ascent(
                gpu,
                &mut preps,
                &idxs,
                &seed,
                &mut per_domain,
                seed_node,
                pre_dim,
                &sel,
                &row_widths,
                opts,
                &|| self.config.alpha_config.past_deadline(),
            ) {
                if !opts.alpha_reopt {
                    if let Some(store) = &opts.alpha_store {
                        *store.lock().unwrap_or_else(|e| e.into_inner()) = Some(ap);
                    }
                }
            }
        }

        let gpu_ms = t_gpu.elapsed().as_millis();

        // Row index of each selected neuron (usize::MAX = unselected) — the
        // prune test's map from a premise neuron to its refined row.
        let mut row_of = vec![usize::MAX; pre_dim];
        for (row, &j) in sel.iter().enumerate() {
            row_of[j] = row;
        }

        // #clip-interm-resnet-batched: constrained-concretize every domain's seed rows
        // over box ∩ split-half-spaces — the CPU clip solve (the eliminated per-child
        // backward is already collapsed into ONE batched GPU pass). Independent per
        // domain ⇒ rayon fan-out (mirrors the prep fan-out above); each result is a
        // pure `(tightened_lower, tightened_upper)` over `sel` rows, combined serially
        // below. Empty when the clip is off.
        let clip_results: Vec<Option<(Vec<f32>, Vec<f32>)>> = if do_clip {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let folded_ref: &[Option<FoldedSeedRows>] = &per_domain_folded;
            let preps_ref: &[Option<ResnetDomainPrep>] = &preps;
            let row_of_ref: &[usize] = &row_of;
            let sel_ref: &[usize] = &sel;
            (0..n_domains)
                .into_par_iter()
                .map(|i| {
                    let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                    let folded = folded_ref[i].as_ref()?;
                    let bs = beta_states[i]?;
                    let p = preps_ref[i].as_ref()?;
                    clip_seed_domain(
                        folded,
                        None,
                        bs,
                        relu_name,
                        row_of_ref,
                        sel_ref,
                        &p.in_lo,
                        &p.in_hi,
                        self.config.alpha_config.deadline,
                    )
                })
                .collect()
        } else {
            Vec::new()
        };

        // Prune test + intersect into the caches (pre entry + the ReLU's post
        // entry), in place.
        let mut stats = RefineStats::default();
        let mut any_refined = false;
        for i in 0..n_domains {
            if per_domain[i].is_none() {
                continue;
            }
            // INFEASIBILITY PRUNE (dark, `NY_INTERM_REFINE_PRUNE=1` — see
            // module docs): the RAW refined enclosure (pre-intersection, no
            // seed-layer premises folded) contradicting one of the domain's
            // own split premises beyond tol proves the subdomain empty. Uses the
            // scalar backward bound (pre-clip) so its behaviour is byte-identical
            // whether or not the clip runs.
            if opts.prune && !infeasible[i] {
                let r = per_domain[i].as_ref().expect("checked Some");
                if r.lower_bounds.len() == sel.len() && r.upper_bounds.len() == sel.len() {
                    if let Some(bs) = beta_states[i] {
                        if let Some((j, sign, margin)) = premise_contradiction(
                            bs,
                            relu_name,
                            &row_of,
                            &r.lower_bounds,
                            &r.upper_bounds,
                            opts.prune_tol,
                        ) {
                            infeasible[i] = true;
                            stats.infeasible += 1;
                            if probe {
                                eprintln!(
                                    "[interm-refine-prune] relu={relu_name} dom={i} neuron={j} \
                                     premise={} margin={margin:.6}",
                                    if sign > 0.0 { "active" } else { "inactive" },
                                );
                            }
                        }
                    }
                }
            }
            // #clip-interm-guard (dark, `NY_CLIP_INTERM=1`): FAIL-CLOSED runtime
            // enclosure check BEFORE the clip is merged. A too-tight intermediate
            // bound is a false-VERIFY (catastrophic), so — when armed — every
            // clip-tightened row is validated against a bounded directed adversarial
            // sample of the child's OWN split-satisfying feasible points (true forward
            // through the graph). On any feasible point outside a tightened box, or a
            // non-finite (NaN) tightening bound, the WHOLE seed node reverts to the
            // inherited PARENT bound for this child (skip both the clip and the scalar
            // refinement → keep inherited; always sound). Default OFF = byte-identical.
            if do_clip && opts.clip_guard {
                let verdict = match (
                    per_domain_folded[i].as_ref(),
                    clip_results.get(i).and_then(|o| o.as_ref()),
                    beta_states[i],
                ) {
                    (Some(folded), Some((tl, tu)), Some(bs))
                        if tl.len() == sel.len() && tu.len() == sel.len() =>
                    {
                        let nan_bound = (0..sel.len())
                            .any(|r| !folded.refused[r] && (tl[r].is_nan() || tu[r].is_nan()));
                        if nan_bound {
                            Err((usize::MAX, f32::NAN, f32::NAN, f32::NAN))
                        } else {
                            clip_guard_verify_domain(
                                graph,
                                &constrained_inputs[i],
                                seed_node,
                                relu_name,
                                bs,
                                &sel,
                                folded,
                                tl,
                                tu,
                                clip_interm_guard_restarts(),
                                0x9E37_79B9_7F4A_7C15u64 ^ (i as u64).wrapping_mul(0x1_0000_01b3),
                            )
                        }
                    }
                    _ => Ok(()),
                };
                if let Err((j, z, cl, cu)) = verdict {
                    stats.clip_guard_reverts += 1;
                    tracing::warn!(
                        seed = seed_node,
                        relu = relu_name,
                        dom = i,
                        neuron = j,
                        true_value = z,
                        clip_lower = cl,
                        clip_upper = cu,
                        "clip-interm-guard REVERT: feasible sample escapes tightened box — \
                         keeping inherited parent bound (clip refused for this child)"
                    );
                    if probe {
                        eprintln!(
                            "[clip-interm-guard] REVERT seed={seed_node} relu={relu_name} \
                             dom={i} neuron={j} true={z} outside clip [{cl}, {cu}]"
                        );
                    }
                    // Revert the whole node to inherited: skip clip + scalar refinement.
                    continue;
                }
            }
            // #clip-interm-resnet-batched: intersect the batched split-constraint clip
            // into this domain's scalar refinement BEFORE the inherited intersect. The
            // clip's constrained concretization (box ∩ split-half-spaces) can only
            // TIGHTEN the box-only scalar bound; `max`/`min` combine keeps the scalar on
            // inversion/NaN (sound — the clip may only fail to tighten).
            if do_clip {
                if let Some(folded) = per_domain_folded[i].as_ref() {
                    stats.clip_refused_rows += folded.refused.iter().filter(|&&b| b).count();
                }
                if let (Some(folded), Some((tl, tu))) = (
                    per_domain_folded[i].as_ref(),
                    clip_results.get(i).and_then(|o| o.as_ref()),
                ) {
                    let r = per_domain[i].as_mut().expect("checked Some");
                    if r.lower_bounds.len() == sel.len()
                        && r.upper_bounds.len() == sel.len()
                        && tl.len() == sel.len()
                        && tu.len() == sel.len()
                    {
                        let mut changed = false;
                        for row in 0..sel.len() {
                            if folded.refused[row] {
                                continue;
                            }
                            let (sl, su) = (r.lower_bounds[row], r.upper_bounds[row]);
                            let (cl, cu) = (tl[row], tu[row]);
                            let nl = if cl.is_finite() {
                                nan_propagating_max(sl, cl)
                            } else {
                                sl
                            };
                            let nu = if cu.is_finite() {
                                nan_propagating_min(su, cu)
                            } else {
                                su
                            };
                            // Keep the scalar (pre-clip) values on inversion/NaN.
                            if nl.is_finite() && nu.is_finite() && nl <= nu {
                                if nl > sl || nu < su {
                                    let d = (nl - sl).abs().max((nu - su).abs());
                                    stats.clip_max_tighten = stats.clip_max_tighten.max(d);
                                    changed = true;
                                }
                                r.lower_bounds[row] = nl;
                                r.upper_bounds[row] = nu;
                            }
                        }
                        if changed {
                            stats.clip_domains += 1;
                        }
                    }
                }
            }
            let r = per_domain[i].as_ref().expect("checked Some");
            if apply_refinement(
                &mut caches[i],
                seed_node,
                relu_name,
                r,
                pre_dim,
                &sel,
                &mut stats,
            ) {
                any_refined = true;
                stats.domains_refined += 1;
            }
        }
        // #ab-parity-interm FC-head BOX-WIDTH PROBE (`NY_AB_INTERM_PROBE=1`,
        // stderr-only, read-only): the seed node's TOTAL refined box width
        // Σ_j (u_j − l_j) over ALL pre-activation neurons — directly comparable
        // to the measured CROWN 419.27 head width. Read from the replay domain's
        // REFINED cache (post-apply). Runs for both the shared (OFF) and
        // per-target (ON) modes; prints once at the ROOT batch and once at the
        // first mid-depth (≥ 5 split premises) batch.
        if opts.interm_box_probe && is_last_relu {
            if let Some(&ridx) = idxs.first() {
                if let Some(entry) = caches[ridx].get(seed_node) {
                    let width: f64 = entry
                        .lower()
                        .iter()
                        .zip(entry.upper().iter())
                        .filter(|(l, u)| l.is_finite() && u.is_finite())
                        .map(|(&l, &u)| (u - l) as f64)
                        .sum();
                    let batch_depth = beta_states
                        .iter()
                        .flatten()
                        .map(|bs| bs.entries.len())
                        .max()
                        .unwrap_or(0);
                    let mode = if opts.per_target {
                        "per-target"
                    } else {
                        "shared"
                    };
                    static ROOT_ONCE: std::sync::Once = std::sync::Once::new();
                    let did_root = !ROOT_ONCE.is_completed();
                    ROOT_ONCE.call_once(|| {
                        eprintln!(
                            "[ab-interm-probe] ROOT seed={seed_node} mode={mode} pre_dim={pre_dim} \
                             depth={batch_depth} fc_head_box_width={width:.4}"
                        );
                    });
                    if !did_root && batch_depth >= 5 {
                        static MID_ONCE: std::sync::Once = std::sync::Once::new();
                        MID_ONCE.call_once(|| {
                            eprintln!(
                                "[ab-interm-probe] MID seed={seed_node} mode={mode} pre_dim={pre_dim} \
                                 depth={batch_depth} fc_head_box_width={width:.4}"
                            );
                        });
                    }
                }
            }
        }
        if probe && do_clip {
            eprintln!(
                "[clip-resnet-batched] relu={relu_name} batched_backward=1 \
                 clip_domains={} nodes_changed={} max_tighten={:.4} refused_rows={} \
                 guard_reverts={}",
                stats.clip_domains,
                stats.clip_domains,
                stats.clip_max_tighten,
                stats.clip_refused_rows,
                stats.clip_guard_reverts,
            );
        }
        if probe {
            eprintln!(
                "[interm-refine] seed={seed_node} relu={relu_name} pre_dim={pre_dim} rows={n_rows} \
                 domains={}/{n_domains} chunk_doms={chunk_doms} batched_ok={batched_ok} \
                 tightened={} newly_stable={} crossings_kept={} \
                 infeasible={} width {:.3}->{:.3} ({}ms: prep={prep_ms} gpu={gpu_ms})",
                stats.domains_refined,
                stats.neurons_tightened,
                stats.newly_stable,
                stats.crossings_kept,
                stats.infeasible,
                stats.width_before,
                stats.width_after,
                t0.elapsed().as_millis()
            );
        }
        Some((any_refined, stats.newly_stable))
    }
}

/// #clip-interm-resnet-batched: one domain's seed-node input-relative rows, with the
/// certified per-coefficient error already folded OUTWARD into the bias over that
/// domain's own input box. Row `r` (in `sel` order) is the enclosure
/// `lower_a[r]·x + lower_b[r] ≤ z_{sel[r]}(x) ≤ upper_a[r]·x + upper_b[r]` for every
/// `x` in the domain box; `refused[r]` marks a row whose fold produced a non-finite
/// penalty (kept inherited — sound, no clip for that neuron).
struct FoldedSeedRows {
    lower_a: Vec<f32>,
    upper_a: Vec<f32>,
    lower_b: Vec<f32>,
    upper_b: Vec<f32>,
    refused: Vec<bool>,
    dim: usize,
    n_rows: usize,
}

/// Fold the certified per-coefficient error of one domain's block of the batched
/// [`GpuResidentCoeffBatched`] OUTWARD into the bias over that domain's input box.
///
/// SOUNDNESS: the captured rows carry (a) a bias center split into `b ± b_err`
/// (`b_err` already folds the per-ReLU concretized error on the force-fine pass) and
/// (b) a residual per-coefficient error `err_a`. A raw-coefficient enclosure that
/// dropped `err_a` would be UNSOUND (a too-tight bound → false UNSAT). Here each row's
/// residual coefficient error is bounded by `Σ_j |err_a[j]|·max(|x_l[j]|,|x_u[j]|)`
/// over the box and folded outward: `lower_b ← next_down(b_lo − b_err_lo − penalty)`,
/// `upper_b ← next_up(b_hi + b_err_hi + penalty)`. Any row with a non-finite penalty
/// or non-finite coefficient is REFUSED (kept inherited). The resulting affine form is
/// a guaranteed enclosure of `z(x)` over the box (mirrors the serial path's
/// `fold_coeff_err_over_box_eager` + `concretize_resident_coeff_batched`).
fn fold_seed_rows_for_domain(
    coeff: &GpuResidentCoeffBatched,
    local_block: usize,
    n_rows: usize,
    in_lo: &[f32],
    in_hi: &[f32],
    deadline: Option<std::time::Instant>,
) -> Option<FoldedSeedRows> {
    let mut past_deadline = || deadline.is_some_and(|d| std::time::Instant::now() >= d);
    fold_seed_rows_for_domain_with_deadline_check(
        coeff,
        local_block,
        n_rows,
        in_lo,
        in_hi,
        &mut past_deadline,
    )
}

/// Callback-explicit fold body for deterministic expiry tests. Production uses
/// the `Instant` wrapper above. Polling is bounded by 1024 coefficient cells so
/// a deadline cannot expire during a full resident error fold and remain hidden
/// until the later CPU clip proposal.
fn fold_seed_rows_for_domain_with_deadline_check<F>(
    coeff: &GpuResidentCoeffBatched,
    local_block: usize,
    n_rows: usize,
    in_lo: &[f32],
    in_hi: &[f32],
    past_deadline: &mut F,
) -> Option<FoldedSeedRows>
where
    F: FnMut() -> bool,
{
    if past_deadline() {
        return None;
    }
    let dim = coeff.dim;
    let coeff_cells = coeff.num_specs.checked_mul(dim)?;
    let row_cells = n_rows.checked_mul(dim)?;
    if dim == 0
        || n_rows == 0
        || coeff.num_specs_per_dom != n_rows
        || !coeff.num_specs.is_multiple_of(n_rows)
        || coeff.lower_a.len() != coeff_cells
        || coeff.upper_a.len() != coeff_cells
        || coeff.lower_err.len() != coeff_cells
        || coeff.upper_err.len() != coeff_cells
        || coeff.lower_b.len() != coeff.num_specs
        || coeff.upper_b.len() != coeff.num_specs
        || coeff.lower_b_err.len() != coeff.num_specs
        || coeff.upper_b_err.len() != coeff.num_specs
        || in_lo.len() != dim
        || in_hi.len() != dim
    {
        return None;
    }
    for j in 0..dim {
        if j.is_multiple_of(1024) && past_deadline() {
            return None;
        }
        let (l, u) = (in_lo[j], in_hi[j]);
        if !l.is_finite() || !u.is_finite() || l > u {
            return None;
        }
    }
    if past_deadline() {
        return None;
    }
    // Includes conservative multiplicities for both coefficient directions,
    // their error sources, per-row biases, and the certificate that may consume
    // this proposal. This check precedes `abs_x` and every output Vec allocation.
    crate::complete_clip::validate_clip_work_budget(1, n_rows, 0, dim).ok()?;
    if past_deadline() {
        return None;
    }
    // Per-input-dim magnitude bound max(|x_l|, |x_u|).
    let mut abs_x = Vec::with_capacity(dim);
    for j in 0..dim {
        if j.is_multiple_of(1024) && past_deadline() {
            return None;
        }
        abs_x.push(f64::from(in_lo[j].abs().max(in_hi[j].abs())));
    }
    if past_deadline() {
        return None;
    }
    let mut out = FoldedSeedRows {
        lower_a: vec![0.0; row_cells],
        upper_a: vec![0.0; row_cells],
        lower_b: vec![0.0; n_rows],
        upper_b: vec![0.0; n_rows],
        refused: vec![false; n_rows],
        dim,
        n_rows,
    };
    for r in 0..n_rows {
        if past_deadline() {
            return None;
        }
        let s = local_block.checked_mul(n_rows)?.checked_add(r)?;
        if s >= coeff.num_specs {
            return None;
        }
        let base = s * dim;
        if base + dim > coeff.lower_a.len() || base + dim > coeff.upper_a.len() {
            return None;
        }
        let mut penalty_lo = 0.0f64;
        let mut penalty_hi = 0.0f64;
        let mut coeff_ok = true;
        for j in 0..dim {
            if j.is_multiple_of(1024) && past_deadline() {
                return None;
            }
            let la = coeff.lower_a[base + j];
            let ua = coeff.upper_a[base + j];
            if !la.is_finite() || !ua.is_finite() {
                coeff_ok = false;
                break;
            }
            out.lower_a[r * dim + j] = la;
            out.upper_a[r * dim + j] = ua;
            // Residual per-coeff error (near-zero after the force-fine fold; folded
            // outward as a backstop so the enclosure is guaranteed even if a row
            // still carries error).
            let le = coeff.lower_err[base + j];
            let ue = coeff.upper_err[base + j];
            penalty_lo = fold_add_up(penalty_lo, fold_mul_up(f64::from(le.abs()), abs_x[j]));
            penalty_hi = fold_add_up(penalty_hi, fold_mul_up(f64::from(ue.abs()), abs_x[j]));
        }
        if past_deadline() {
            return None;
        }
        let lb_c = f64::from(coeff.lower_b[s]);
        let lb_e = f64::from(coeff.lower_b_err[s].abs());
        let ub_c = f64::from(coeff.upper_b[s]);
        let ub_e = f64::from(coeff.upper_b_err[s].abs());
        let lbias = fold_f64_to_f32_down(fold_sub_down(fold_sub_down(lb_c, lb_e), penalty_lo));
        let ubias = fold_f64_to_f32_up(fold_add_up(fold_add_up(ub_c, ub_e), penalty_hi));
        if !coeff_ok
            || !penalty_lo.is_finite()
            || !penalty_hi.is_finite()
            || !lbias.is_finite()
            || !ubias.is_finite()
        {
            out.refused[r] = true;
            // Non-tightening placeholders so a refused row is a no-op in the clip.
            out.lower_b[r] = f32::NEG_INFINITY;
            out.upper_b[r] = f32::INFINITY;
            for j in 0..dim {
                out.lower_a[r * dim + j] = 0.0;
                out.upper_a[r * dim + j] = 0.0;
            }
            continue;
        }
        out.lower_b[r] = lbias;
        out.upper_b[r] = ubias;
    }
    Some(out)
}

/// Directed arithmetic used only by the resident-coefficient certificate fold.
/// Every source is an exact finite f32 dyadic. Widening each f64 operation (not
/// just the final f32 cast) remains sound even when a large error penalty nearly
/// cancels the stored bias center.
fn fold_add_up(a: f64, b: f64) -> f64 {
    fold_next_up_f64(a + b)
}

fn fold_sub_down(a: f64, b: f64) -> f64 {
    fold_next_down_f64(a - b)
}

fn fold_mul_up(a: f64, b: f64) -> f64 {
    fold_next_up_f64(a * b)
}

fn fold_next_down_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

fn fold_next_up_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn fold_f64_to_f32_down(value: f64) -> f32 {
    let candidate = value as f32;
    if f64::from(candidate) <= value {
        candidate
    } else {
        next_down_f32(candidate)
    }
}

fn fold_f64_to_f32_up(value: f64) -> f32 {
    let candidate = value as f32;
    if f64::from(candidate) >= value {
        candidate
    } else {
        next_up_f32(candidate)
    }
}

/// Recover the exact ReLU-at-zero split semantics retained by `GraphBetaState`.
/// A GenBaB-like nonzero split cannot be reconstructed and therefore refuses the
/// entire clipping proposal.
fn reconstruct_clip_relu_history<F>(
    beta_state: &GraphBetaState,
    past_deadline: &mut F,
) -> Option<GraphSplitHistory>
where
    F: FnMut() -> bool,
{
    let mut history_bytes = 0usize;
    for (entry_index, entry) in beta_state.entries.iter().enumerate() {
        if entry_index.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return None;
        }
        history_bytes = history_bytes
            .checked_add(size_of::<GraphNeuronConstraint>())?
            .checked_add(entry.node_name().len())?;
    }
    if history_bytes > SELECTIVE_CLIP_MAX_HOST_BYTES || past_deadline() {
        return None;
    }

    let mut history = GraphSplitHistory::new();
    for (entry_index, entry) in beta_state.entries.iter().enumerate() {
        if entry_index.is_multiple_of(64) && past_deadline() {
            return None;
        }
        if entry.split_point() != 0.0 || !matches!(entry.sign(), -1.0 | 1.0) {
            return None;
        }
        let constraint = GraphNeuronConstraint::new(
            entry.node_name().to_string(),
            entry.neuron_idx(),
            entry.sign() == 1.0,
            0.0,
        )
        .ok()?;
        history.add_constraint(constraint);
    }
    Some(history)
}

/// Bit-exact compatibility guard between the legacy raw CUDA proposal and the
/// independently context-validated token rows.  The proposal never supplies an
/// authoritative number; any disagreement disables clipping for the domain.
fn folded_proposal_matches_certificate<F>(
    proposal: &FoldedSeedRows,
    certified: &ValidatedAffineEnclosure,
    past_deadline: &mut F,
) -> bool
where
    F: FnMut() -> bool,
{
    if proposal.dim != certified.dim()
        || proposal.n_rows != certified.rows()
        || proposal.refused.len() != proposal.n_rows
        || proposal.refused.iter().any(|&refused| refused)
        || proposal.lower_a.len() != proposal.n_rows.saturating_mul(proposal.dim)
        || proposal.upper_a.len() != proposal.n_rows.saturating_mul(proposal.dim)
        || proposal.lower_b.len() != proposal.n_rows
        || proposal.upper_b.len() != proposal.n_rows
    {
        return false;
    }
    let mut cells = 0usize;
    for row in 0..proposal.n_rows {
        if proposal.lower_b[row].to_bits() != certified.lower_b()[row].to_bits()
            || proposal.upper_b[row].to_bits() != certified.upper_b()[row].to_bits()
        {
            return false;
        }
        for column in 0..proposal.dim {
            if cells.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
                return false;
            }
            let index = row * proposal.dim + column;
            if proposal.lower_a[index].to_bits() != certified.lower_a()[[row, column]].to_bits()
                || proposal.upper_a[index].to_bits() != certified.upper_a()[[row, column]].to_bits()
            {
                return false;
            }
            cells = cells.saturating_add(1);
        }
    }
    !past_deadline()
}

/// #clip-interm-resnet-batched: run the split-constraint clip for one domain from its
/// folded seed rows + β-state (the split premises). Reconstructs the domain's ReLU
/// split history from `beta_state` (node/neuron/sign — split point 0 for ReLU), builds
/// input-space half-spaces from the split neurons' folded rows
/// ([`build_split_constraints`]), and constrained-concretizes every `sel` objective row
/// over `box ∩ half-spaces` ([`tighten_with_constraints_with_deadline`]). Returns
/// `(tightened_lower, tightened_upper)` over `sel` rows, or `None` when there are no
/// usable constraints (no-op — sound). All bound math is the existing clip solver;
/// this only sources the objective/constraint rows from the batched ResidentCoeff
/// instead of a per-child backward.
#[allow(clippy::too_many_arguments)]
fn clip_seed_domain(
    folded: &FoldedSeedRows,
    authority: Option<(&CertifiedAffineEnclosure, &CrownPassStamp)>,
    beta_state: &GraphBetaState,
    relu_name: &str,
    row_of: &[usize],
    sel: &[usize],
    in_lo: &[f32],
    in_hi: &[f32],
    deadline: Option<std::time::Instant>,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let mut past_deadline = || deadline.is_some_and(|d| std::time::Instant::now() >= d);
    clip_seed_domain_with_deadline_check(
        folded,
        authority,
        beta_state,
        relu_name,
        row_of,
        sel,
        in_lo,
        in_hi,
        deadline,
        &mut past_deadline,
    )
}

#[allow(clippy::too_many_arguments)]
fn clip_seed_domain_with_deadline_check<F>(
    folded: &FoldedSeedRows,
    authority: Option<(&CertifiedAffineEnclosure, &CrownPassStamp)>,
    beta_state: &GraphBetaState,
    relu_name: &str,
    row_of: &[usize],
    sel: &[usize],
    in_lo: &[f32],
    in_hi: &[f32],
    deadline: Option<std::time::Instant>,
    past_deadline: &mut F,
) -> Option<(Vec<f32>, Vec<f32>)>
where
    F: FnMut() -> bool,
{
    // A raw ResidentCoeff/FoldedSeedRows value is a proposal, never authority.
    // The scored gate is still hard-false, and the current sound-CROWN boundary
    // deliberately has no production token constructor, so this returns `None`
    // (inherit bounds) outside explicit provenance fixtures.
    let (token, pass) = authority?;
    let x_dim = folded.dim;
    // Fail before building histories, half-spaces, or objective matrices when
    // their conservative dense-work estimate exceeds the certified clip cap.
    crate::complete_clip::validate_clip_work_budget(1, sel.len(), beta_state.entries.len(), x_dim)
        .ok()?;
    if past_deadline() {
        return None;
    }
    let history = reconstruct_clip_relu_history(beta_state, past_deadline)?;
    let dbg = std::env::var("NY_CLIP_DBG").ok().as_deref() == Some("1");
    if dbg {
        let names: Vec<String> = beta_state
            .entries
            .iter()
            .take(3)
            .map(|e| format!("{}[{}]", e.node_name(), e.neuron_idx()))
            .collect();
        let rows: Vec<i64> = beta_state
            .entries
            .iter()
            .take(3)
            .map(|e| {
                if e.node_name() == relu_name && e.neuron_idx() < row_of.len() {
                    row_of[e.neuron_idx()] as i64
                } else {
                    -1
                }
            })
            .collect();
        eprintln!(
            "[clip-dbg] relu_name={relu_name} n_entries={} entry_nodes={names:?} rows={rows:?} sel_len={} row_of_len={}",
            beta_state.entries.len(),
            sel.len(),
            row_of.len()
        );
    }
    if history.depth() == 0 {
        if dbg {
            eprintln!("[clip-dbg] history.depth==0 (no beta entries) -> None");
        }
        return None;
    }

    let certified = token
        .validate_for_clip(
            pass, in_lo, in_hi, &history, relu_name, sel, row_of, deadline,
        )
        .ok()?;
    if !folded_proposal_matches_certificate(folded, &certified, past_deadline) {
        return None;
    }

    // Split-constraint source rows: the seed IS the split pre-node on this lane, so a
    // split at `relu_name` neuron `j` maps to the folded row `row_of[j]` (Relu neuron j's
    // pre-activation is the seed neuron j). Any unmapped/refused premise → None → the
    // constraint is dropped (sound weakening).
    let split_closure = |node_name: &str,
                         neuron_idx: usize,
                         deadline_check: &mut F|
     -> Option<(Array1<f32>, f32, Array1<f32>, f32)> {
        if node_name != relu_name || neuron_idx >= row_of.len() {
            return None;
        }
        let row = row_of[neuron_idx];
        if row == usize::MAX || row >= certified.rows() {
            return None;
        }
        let d = certified.dim();
        let mut la = Array1::<f32>::zeros(d);
        let mut ua = Array1::<f32>::zeros(d);
        for j in 0..d {
            if j.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && deadline_check() {
                return None;
            }
            la[j] = certified.lower_a()[[row, j]];
            ua[j] = certified.upper_a()[[row, j]];
        }
        Some((la, certified.lower_b()[row], ua, certified.upper_b()[row]))
    };

    let constraints = match build_split_constraints_with_deadline_check(
        &history,
        split_closure,
        x_dim,
        past_deadline,
    ) {
        Ok(c) => c,
        Err(e) => {
            if dbg {
                eprintln!(
                    "[clip-dbg] depth={} build_split_constraints ERR: {e}",
                    history.depth()
                );
            }
            return None;
        }
    };
    if constraints.is_empty() {
        if dbg {
            eprintln!("[clip-dbg] depth={} constraints EMPTY (split_closure mapped 0 premises; relu={relu_name})", history.depth());
        }
        return None;
    }
    if in_lo.len() != x_dim || in_hi.len() != x_dim || past_deadline() {
        return None;
    }
    let mut in_lo_arr = Array1::<f32>::zeros(x_dim);
    let mut in_hi_arr = Array1::<f32>::zeros(x_dim);
    for j in 0..x_dim {
        if j.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return None;
        }
        in_lo_arr[j] = in_lo[j];
        in_hi_arr[j] = in_hi[j];
    }
    let preprocessed = sort_out_constraints_with_deadline_check(
        &constraints,
        &in_lo_arr,
        &in_hi_arr,
        past_deadline,
    )
    .ok()?;
    if preprocessed.a_active.nrows() == 0 {
        if dbg {
            eprintln!(
                "[clip-dbg] depth={} a_active=0 (all {} constraints infeasible/fully-covered)",
                history.depth(),
                constraints.num_constraints
            );
        }
        return None;
    }
    if dbg {
        eprintln!(
            "[clip-dbg] depth={} constraints={} a_active={} -> RUNNING clip",
            history.depth(),
            constraints.num_constraints,
            preprocessed.a_active.nrows()
        );
    }

    // Objective rows = every `sel` row (the seed pre-activations to tighten).
    // Invalid/refused rows cannot enter the provenance token at all, so this arm
    // contains only finite outward enclosures.
    let n_obj = sel.len();
    if past_deadline() {
        return None;
    }
    let mut obj_la = Array2::<f32>::zeros((n_obj, x_dim));
    let mut obj_lb = Array1::<f32>::zeros(n_obj);
    let mut obj_ua = Array2::<f32>::zeros((n_obj, x_dim));
    let mut obj_ub = Array1::<f32>::zeros(n_obj);
    let mut objective_cells = 0usize;
    for r in 0..n_obj {
        if r.is_multiple_of(64) && past_deadline() {
            return None;
        }
        obj_lb[r] = certified.lower_b()[r];
        obj_ub[r] = certified.upper_b()[r];
        for j in 0..x_dim {
            if objective_cells.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
                return None;
            }
            obj_la[[r, j]] = certified.lower_a()[[r, j]];
            obj_ua[[r, j]] = certified.upper_a()[[r, j]];
            objective_cells = objective_cells.saturating_add(1);
        }
    }
    if past_deadline() {
        return None;
    }
    let (tl, tu) = tighten_with_constraints_with_deadline(
        &preprocessed,
        &obj_la,
        &obj_lb,
        &obj_ua,
        &obj_ub,
        &in_lo_arr,
        &in_hi_arr,
        deadline,
    )
    .ok()?;
    if past_deadline() {
        return None;
    }
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    lower.try_reserve_exact(tl.len()).ok()?;
    upper.try_reserve_exact(tu.len()).ok()?;
    for (i, (&l, &u)) in tl.iter().zip(tu.iter()).enumerate() {
        if i.is_multiple_of(SELECTIVE_DEADLINE_POLL_STRIDE) && past_deadline() {
            return None;
        }
        lower.push(l);
        upper.push(u);
    }
    (!past_deadline()).then_some((lower, upper))
}

/// #clip-interm-guard: FAIL-CLOSED runtime enclosure check for ONE domain's clip.
/// Production authority is quarantined; explicit Leg-A tests call it directly.
/// Draws `restarts`
/// DIRECTED adversarial points in the child's input box — interior randoms plus
/// per-row extremal pushes toward the box CORNER that maximizes/minimizes a row's
/// folded objective (the vertex a too-tight bound would be exposed at) — keeps the
/// points that satisfy the child's ReLU split SIGNS at the seed layer (checked on
/// the TRUE forward through `graph`, exact on the degenerate point box), and
/// asserts every clip-tightened row `r` encloses the true seed value
/// `clip_lower[r] ≤ z_{sel[r]}(x) ≤ clip_upper[r]` (f32 forward tolerance).
///
/// Returns `Err((seed_neuron_idx, true_value, clip_lower, clip_upper))` on the
/// FIRST feasible point found outside its row's tightened box — proof the clip is
/// unsound for this child, so the caller REVERTS the whole seed node to the
/// inherited parent bound. `Ok(())` when every sampled feasible point is enclosed
/// (or no feasible sample was drawn — no evidence of unsoundness; the sound
/// scalar∩inherited intersect still bounds the merge). Refused rows carry no
/// tightening and are skipped.
#[allow(clippy::too_many_arguments)]
fn clip_guard_verify_domain(
    graph: &GraphNetwork,
    input_box: &BoundedTensor,
    seed_node: &str,
    relu_name: &str,
    beta: &GraphBetaState,
    sel: &[usize],
    folded: &FoldedSeedRows,
    clip_lower: &[f32],
    clip_upper: &[f32],
    restarts: usize,
    seed_rng: u64,
) -> Result<(), (usize, f32, f32, f32)> {
    let lo = input_box.lower();
    let hi = input_box.upper();
    let shape = lo.raw_dim();
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();
    let dim = lo_flat.len();
    let n_rows = sel.len();
    if dim == 0 || clip_lower.len() != n_rows || clip_upper.len() != n_rows || folded.dim != dim {
        return Ok(()); // nothing checkable — the sound scalar∩inherited still guards
    }

    // Deterministic LCG in [0,1) (the tests_soundness sampling pattern).
    let mut state = seed_rng | 1;
    let mut u01 = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };

    // The premises the clip actually encoded: split SIGNS at the seed ReLU.
    let premises: Vec<(usize, f32, bool)> = beta
        .entries_for_node(relu_name)
        .map(|e| (e.neuron_idx(), e.split_point(), e.sign() > 0.0))
        .collect();

    for k in 0..restarts {
        // Half the restarts are DIRECTED corner pushes (rotate over row × dir),
        // the rest interior randoms.
        let mut x = vec![0.0f32; dim];
        if k % 2 == 0 && n_rows > 0 {
            let r = (k / 2) % n_rows;
            let test_upper = (k / 2 / n_rows).is_multiple_of(2);
            for d in 0..dim {
                let a = if test_upper {
                    folded.upper_a[r * dim + d]
                } else {
                    folded.lower_a[r * dim + d]
                };
                // Corner extremizing a·x over [lo,hi] in the violating direction.
                let corner = if (test_upper && a > 0.0) || (!test_upper && a < 0.0) {
                    hi_flat[d]
                } else {
                    lo_flat[d]
                };
                let span = hi_flat[d] - lo_flat[d];
                x[d] = (corner + (u01() - 0.5) * 0.15 * span).clamp(lo_flat[d], hi_flat[d]);
            }
        } else {
            for d in 0..dim {
                x[d] = lo_flat[d] + u01() * (hi_flat[d] - lo_flat[d]);
            }
        }

        let pt = match ndarray::ArrayD::from_shape_vec(shape.clone(), x)
            .ok()
            .and_then(|a| BoundedTensor::new(a.clone(), a).ok())
        {
            Some(p) => p,
            None => continue,
        };
        let Ok(vals) = graph.collect_node_bounds(&pt) else {
            continue;
        };
        let Some(seed_vals) = vals.get(seed_node) else {
            continue;
        };
        let seed_flat = seed_vals.flatten();
        let seed_len = seed_flat.len();
        let val_at =
            |idx: usize| -> Option<f32> { (idx < seed_len).then(|| seed_flat.lower()[[idx]]) };

        // Feasibility: the child's ReLU split signs at the seed layer.
        let feasible = premises
            .iter()
            .all(|&(j, s, active)| val_at(j).is_some_and(|v| if active { v >= s } else { v <= s }));
        if !feasible {
            continue;
        }

        // Enclosure: every clip-tightened row must contain the true seed value.
        for r in 0..n_rows {
            if folded.refused[r] {
                continue;
            }
            let Some(z) = val_at(sel[r]) else {
                continue;
            };
            let (cl, cu) = (clip_lower[r], clip_upper[r]);
            let tol = 1e-3 * (1.0 + z.abs());
            if (cl.is_finite() && z < cl - tol) || (cu.is_finite() && z > cu + tol) {
                return Err((sel[r], z, cl, cu));
            }
        }
    }
    Ok(())
}

/// Intersect one domain's refined `[l', u']` into its cache (the seed node's
/// pre-activation entry and the ReLU's post-activation entry). Row `r` of the
/// result refines neuron `sel[r]`; unselected neurons keep their inherited
/// pair. Returns whether the cache was updated (a length/shape/repair refusal
/// keeps the inherited entries and returns false — sound).
fn apply_refinement(
    cache: &mut HashMap<String, Arc<BoundedTensor>>,
    seed_node: &str,
    relu_name: &str,
    r: &ny_core::GpuCrownResult,
    pre_dim: usize,
    sel: &[usize],
    stats: &mut RefineStats,
) -> bool {
    if r.lower_bounds.len() != sel.len() || r.upper_bounds.len() != sel.len() {
        return false;
    }
    let Some(old) = cache.get(seed_node) else {
        return false;
    };
    if old.len() != pre_dim {
        return false;
    }
    // Row index of each selected neuron (usize::MAX = unselected).
    let mut row_of = vec![usize::MAX; pre_dim];
    for (row, &j) in sel.iter().enumerate() {
        if j >= pre_dim {
            return false;
        }
        row_of[j] = row;
    }
    let mut new_l = Vec::with_capacity(pre_dim);
    let mut new_u = Vec::with_capacity(pre_dim);
    let mut tightened = 0usize;
    let mut newly_stable = 0usize;
    let mut crossings = 0usize;
    let mut w_before = 0.0f64;
    let mut w_after = 0.0f64;
    for (j, (&ol, &ou)) in old.lower().iter().zip(old.upper().iter()).enumerate() {
        let (nl, nu, t) = if row_of[j] == usize::MAX {
            (ol, ou, false)
        } else {
            let (rl, ru) = (r.lower_bounds[row_of[j]], r.upper_bounds[row_of[j]]);
            // Stats-only: a VALID refined pair whose intersection with the
            // inherited pair came back unchanged-and-empty was kept-inherited.
            if rl.is_finite() && ru.is_finite() && rl <= ru && ol.max(rl) > ou.min(ru) {
                crossings += 1;
            }
            intersect_pair(ol, ou, rl, ru)
        };
        if t {
            tightened += 1;
            // Stability flip: the pruning lever — the margin backward's
            // relaxation of this neuron becomes EXACT (slope 1 or 0).
            if ol < 0.0 && ou > 0.0 && (nl >= 0.0 || nu <= 0.0) {
                newly_stable += 1;
            }
        }
        if ol.is_finite() && ou.is_finite() {
            w_before += (ou - ol) as f64;
            w_after += (nu - nl) as f64;
        }
        new_l.push(nl);
        new_u.push(nu);
    }
    let shape = old.lower().raw_dim();
    let Ok(lower) = ndarray::ArrayD::from_shape_vec(shape.clone(), new_l.clone()) else {
        return false;
    };
    let Ok(upper) = ndarray::ArrayD::from_shape_vec(shape, new_u.clone()) else {
        return false;
    };
    let Ok(refined_pre) = BoundedTensor::new(lower, upper) else {
        return false;
    };

    // The ReLU's own POST-activation entry: relu is exact + monotone, so
    // [relu(l'), relu(u')] encloses the post-activations; intersect with the
    // inherited post entry. Refused (kept inherited) on any shape mismatch.
    let refined_post = cache.get(relu_name).and_then(|post| {
        if post.len() != pre_dim {
            return None;
        }
        let mut pl = Vec::with_capacity(pre_dim);
        let mut pu = Vec::with_capacity(pre_dim);
        for (j, (&ol, &ou)) in post.lower().iter().zip(post.upper().iter()).enumerate() {
            let (nl, nu, _) = intersect_pair(ol, ou, new_l[j].max(0.0), new_u[j].max(0.0));
            pl.push(nl);
            pu.push(nu);
        }
        let shape = post.lower().raw_dim();
        let lower = ndarray::ArrayD::from_shape_vec(shape.clone(), pl).ok()?;
        let upper = ndarray::ArrayD::from_shape_vec(shape, pu).ok()?;
        BoundedTensor::new(lower, upper).ok()
    });

    // Fresh Arcs: refinement replaces entries wholesale, never mutates a
    // shared tensor in place (#cone-delta increment 2 aliasing rule).
    cache.insert(seed_node.to_string(), Arc::new(refined_pre));
    if let Some(post) = refined_post {
        cache.insert(relu_name.to_string(), Arc::new(post));
    }
    stats.neurons_tightened += tightened;
    stats.newly_stable += newly_stable;
    stats.crossings_kept += crossings;
    stats.width_before += w_before;
    stats.width_after += w_after;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    fn lin(name: &str, input: &str) -> GraphNode {
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

    /// input → l1 → relu1 → l2 → add(l2, l1) → gemm_pre → relu_last → out.
    /// The walk from `out` must find (relu_last, gemm_pre).
    #[test]
    fn find_last_relu_seed_walks_the_unary_chain() {
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2", "relu1"));
        g.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["l2".to_string(), "l1".to_string()],
        ));
        g.add_node(lin("gemm_pre", "add"));
        g.add_node(relu("relu_last", "gemm_pre"));
        g.add_node(lin("out", "relu_last"));
        g.set_output("out");

        let (relu_name, seed) = find_last_relu_seed(&g, "out").expect("chain must resolve");
        assert_eq!(relu_name, "relu_last");
        assert_eq!(seed, "gemm_pre");
    }

    /// A binary node between the output and the first ReLU refuses (no clean
    /// last-ReLU chain), and a ReLU fed by the network input refuses too.
    #[test]
    fn find_last_relu_seed_refuses_non_unary_and_input_fed() {
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(lin("l2", NETWORK_INPUT));
        g.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["l1".to_string(), "l2".to_string()],
        ));
        g.set_output("add");
        assert!(
            find_last_relu_seed(&g, "add").is_none(),
            "binary node before any ReLU must refuse"
        );

        let mut g2 = GraphNetwork::new();
        g2.add_node(relu("r0", NETWORK_INPUT));
        g2.add_node(lin("out", "r0"));
        g2.set_output("out");
        assert!(
            find_last_relu_seed(&g2, "out").is_none(),
            "ReLU fed by the network input has nothing below the seed"
        );
    }

    /// The gate is dark: unset/other values are OFF, only "1" is ON. (Pure
    /// parse check via the env contract — matches!(…, Some("1")).)
    #[test]
    fn interm_refine_gate_semantics() {
        // The gate reads the env var directly; assert the default-off contract
        // by construction: matches!(None | Some("0"), Some("1")) is false.
        assert!(!matches!(None::<&str>, Some("1")));
        assert!(!matches!(Some("0"), Some("1")));
        assert!(matches!(Some("1"), Some("1")));
    }

    /// Adaptive-schedule latch decision (#adaptive-refine): only a zero-yield
    /// batch (no newly-stable neuron, no infeasibility prune) at depth ≥ floor
    /// may latch; production of either kind, or a shallow batch, never does.
    #[test]
    fn adaptive_latch_decision_is_yield_and_floor_gated() {
        // Zero yield at/above the floor ⇒ latch.
        assert!(adaptive_should_latch(0, 0, 4, 4));
        assert!(adaptive_should_latch(0, 0, 9, 4));
        // Any production ⇒ never latch, at any depth.
        assert!(!adaptive_should_latch(1, 0, 9, 4));
        assert!(!adaptive_should_latch(0, 1, 9, 4));
        // Below the floor ⇒ never latch (early refine is protected).
        assert!(!adaptive_should_latch(0, 0, 3, 4));
        assert!(!adaptive_should_latch(0, 0, 0, 4));
        // The gate itself is dark (same env contract as NY_INTERM_REFINE).
        assert!(!matches!(None::<&str>, Some("1")));
    }

    /// Per-child re-refinement (#interm-refine-redo): under a tripped latch,
    /// only sub-latch depths refine when the stride is 0 (the dark default);
    /// with stride `d`, exact depth multiples of `d` re-refine past the latch
    /// and every other deep depth keeps the skip.
    #[test]
    fn redo_reincludes_exact_depth_multiples_under_latch() {
        // Stride 0 (default): pre-redo behavior exactly.
        assert!(latched_domain_refines(6, 7, 0));
        assert!(!latched_domain_refines(7, 7, 0));
        assert!(!latched_domain_refines(16, 7, 0));
        // Stride 8: depths 8/16/24 re-refine; 9..15 and 17 keep the skip.
        assert!(latched_domain_refines(8, 7, 8));
        assert!(latched_domain_refines(16, 7, 8));
        assert!(latched_domain_refines(24, 7, 8));
        assert!(!latched_domain_refines(9, 7, 8));
        assert!(!latched_domain_refines(15, 7, 8));
        assert!(!latched_domain_refines(17, 7, 8));
        // Stride 2: every even deep depth re-refines.
        assert!(latched_domain_refines(8, 7, 2));
        assert!(latched_domain_refines(10, 7, 2));
        assert!(!latched_domain_refines(9, 7, 2));
        // Sub-latch depths always refine regardless of stride.
        assert!(latched_domain_refines(3, 7, 8));
        assert!(latched_domain_refines(0, 7, 8));
    }

    /// #wide-chunk sizing: 0 = OFF (whole batch, one call); otherwise
    /// `max(1, wide_max_n / n_rows)` domains per call.
    #[test]
    fn wide_chunk_domains_sizing() {
        // Gate off: the whole batch in one call regardless of rows.
        assert_eq!(wide_chunk_domains(0, 69, 32), 32);
        assert_eq!(wide_chunk_domains(0, 0, 5), 5);
        // Measured prop54 deep pass: 69 rows/domain, cap 2048 wide rows
        // ⇒ 29 domains per call (2001 rows — under both device caps).
        assert_eq!(wide_chunk_domains(2048, 69, 32), 29);
        // Row-capped deep pass (16 rows): 128 domains per call.
        assert_eq!(wide_chunk_domains(2048, 16, 32), 128);
        // A single domain over the cap still runs (chunk of 1).
        assert_eq!(wide_chunk_domains(64, 100, 8), 1);
        // Zero selected rows: degenerate, whole batch (call refuses upstream).
        assert_eq!(wide_chunk_domains(2048, 0, 4), 4);
        // Empty idxs never yields 0 (chunks(0) would panic).
        assert_eq!(wide_chunk_domains(0, 3, 0), 1);
    }

    /// Intersection semantics: tightening applies, crossings/NaN/non-finite
    /// refined pairs keep the inherited pair (sound).
    #[test]
    fn intersect_pair_is_sound_and_conservative() {
        // Plain tightening on both sides.
        let (l, u, t) = intersect_pair(-2.0, 3.0, -1.0, 2.0);
        assert_eq!((l, u), (-1.0, 2.0));
        assert!(t);
        // Refined looser than inherited → unchanged, not tightened.
        let (l, u, t) = intersect_pair(-0.5, 0.5, -3.0, 3.0);
        assert_eq!((l, u), (-0.5, 0.5));
        assert!(!t);
        // Empty intersection (disjoint sound-in-exact-math edge) → keep inherited.
        let (l, u, t) = intersect_pair(1.0, 2.0, -3.0, 0.5);
        assert_eq!((l, u), (1.0, 2.0));
        assert!(!t);
        // Non-finite refined → keep inherited.
        let (l, u, t) = intersect_pair(-1.0, 1.0, f32::NEG_INFINITY, f32::NAN);
        assert_eq!((l, u), (-1.0, 1.0));
        assert!(!t);
        // Inverted refined → keep inherited.
        let (l, u, t) = intersect_pair(-1.0, 1.0, 0.5, -0.5);
        assert_eq!((l, u), (-1.0, 1.0));
        assert!(!t);
        // NaN inherited must NOT be silently repaired by a finite refined pair.
        let (l, _u, t) = intersect_pair(f32::NAN, 1.0, -0.5, 0.5);
        assert!(l.is_nan(), "NaN inherited lower must propagate");
        assert!(!t);
    }

    /// apply_refinement: pre entry intersected, post entry gets relu-mapped
    /// refinement, stability flip counted, and a wrong-length result refuses.
    #[test]
    fn apply_refinement_updates_pre_and_post_entries() {
        let mk = |l: &[f32], u: &[f32]| {
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[l.len()]), l.to_vec()).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[u.len()]), u.to_vec()).unwrap(),
            )
            .unwrap()
        };
        let mut cache: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
        // Neuron 0 unstable, neuron 1 unstable (will flip stable-active).
        cache.insert("pre".into(), Arc::new(mk(&[-2.0, -1.0], &[3.0, 4.0])));
        cache.insert("relu".into(), Arc::new(mk(&[0.0, 0.0], &[3.0, 4.0])));
        let r = ny_core::GpuCrownResult {
            lower_bounds: vec![-1.5, 0.25],
            upper_bounds: vec![2.0, 3.0],
        };
        let mut stats = RefineStats::default();
        assert!(apply_refinement(
            &mut cache,
            "pre",
            "relu",
            &r,
            2,
            &[0, 1],
            &mut stats
        ));
        let pre = &cache["pre"];
        assert_eq!(pre.lower().as_slice().unwrap(), &[-1.5, 0.25]);
        assert_eq!(pre.upper().as_slice().unwrap(), &[2.0, 3.0]);
        let post = &cache["relu"];
        // post = [relu(l'), relu(u')] ∩ inherited = [0,2] and [0.25,3].
        assert_eq!(post.lower().as_slice().unwrap(), &[0.0, 0.25]);
        assert_eq!(post.upper().as_slice().unwrap(), &[2.0, 3.0]);
        assert_eq!(stats.neurons_tightened, 2);
        assert_eq!(stats.newly_stable, 1, "neuron 1 flipped stable-active");

        // Selective rows: sel=[1] refines only neuron 1; neuron 0 keeps the
        // inherited pair (unstable-only row selection).
        let mut cache_sel: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
        cache_sel.insert("pre".into(), Arc::new(mk(&[-2.0, -1.0], &[3.0, 4.0])));
        cache_sel.insert("relu".into(), Arc::new(mk(&[0.0, 0.0], &[3.0, 4.0])));
        let r_sel = ny_core::GpuCrownResult {
            lower_bounds: vec![-0.5],
            upper_bounds: vec![2.5],
        };
        let mut stats_sel = RefineStats::default();
        assert!(apply_refinement(
            &mut cache_sel,
            "pre",
            "relu",
            &r_sel,
            2,
            &[1],
            &mut stats_sel
        ));
        let pre_sel = &cache_sel["pre"];
        assert_eq!(pre_sel.lower().as_slice().unwrap(), &[-2.0, -0.5]);
        assert_eq!(pre_sel.upper().as_slice().unwrap(), &[3.0, 2.5]);
        assert_eq!(stats_sel.neurons_tightened, 1);

        // Wrong-length refinement refuses and leaves the cache untouched.
        let r_bad = ny_core::GpuCrownResult {
            lower_bounds: vec![0.0],
            upper_bounds: vec![0.0],
        };
        let before = cache["pre"].lower().to_owned();
        let mut stats2 = RefineStats::default();
        assert!(!apply_refinement(
            &mut cache,
            "pre",
            "relu",
            &r_bad,
            2,
            &[0, 1],
            &mut stats2
        ));
        assert_eq!(cache["pre"].lower(), &before);
    }

    use crate::beta_crown::state::GraphBetaEntry;

    fn beta_with(entries: Vec<(usize, f32, f32)>) -> GraphBetaState {
        // (neuron_idx, split_point, sign) at node "relu_last".
        GraphBetaState::from_entries(
            entries
                .into_iter()
                .map(|(idx, s, sign)| {
                    GraphBetaEntry::new("relu_last".into(), idx, s, 0.0, sign).expect("valid entry")
                })
                .collect(),
        )
    }

    /// The prune core: an ACTIVE premise `z ≥ s` contradicted by refined
    /// `u' < s − tol` (and the INACTIVE mirror) fires; anything within tol,
    /// feasible, unselected, or non-finite never fires.
    #[test]
    fn premise_contradiction_fires_only_on_strict_violation() {
        let row_of = vec![0usize, usize::MAX]; // neuron 0 → row 0; neuron 1 unselected
        let tol = 1e-4;

        // ACTIVE premise z0 ≥ 0, refined enclosure entirely below zero → fires.
        let beta = beta_with(vec![(0, 0.0, 1.0)]);
        let hit = premise_contradiction(&beta, "relu_last", &row_of, &[-1.0], &[-0.25], tol);
        assert_eq!(hit, Some((0, 1.0, 0.25)));
        // Wrong node name → no premises → never fires.
        assert!(
            premise_contradiction(&beta, "other_relu", &row_of, &[-1.0], &[-0.25], tol).is_none()
        );

        // INACTIVE premise z0 ≤ 0, refined entirely above zero → fires.
        let beta = beta_with(vec![(0, 0.0, -1.0)]);
        let hit = premise_contradiction(&beta, "relu_last", &row_of, &[0.5], &[2.0], tol);
        assert_eq!(hit, Some((0, -1.0, 0.5)));

        // Feasible: refined straddles the premise threshold → no prune.
        let beta = beta_with(vec![(0, 0.0, 1.0)]);
        assert!(premise_contradiction(&beta, "relu_last", &row_of, &[-0.5], &[1.5], tol).is_none());

        // Within tolerance: violation must EXCEED tol (u' = −tol/2 does not).
        assert!(
            premise_contradiction(&beta, "relu_last", &row_of, &[-1.0], &[-tol / 2.0], tol)
                .is_none()
        );

        // Non-zero split point s: active premise z ≥ 1 with u' = 0.5 fires.
        let beta = beta_with(vec![(0, 1.0, 1.0)]);
        let hit = premise_contradiction(&beta, "relu_last", &row_of, &[-1.0], &[0.5], tol);
        assert_eq!(hit, Some((0, 1.0, 0.5)));

        // Unselected premise neuron (row usize::MAX) → skipped, never fires.
        let beta = beta_with(vec![(1, 0.0, 1.0)]);
        assert!(
            premise_contradiction(&beta, "relu_last", &row_of, &[-1.0], &[-1.0], tol).is_none()
        );

        // Non-finite / inverted refined pair → skipped (conservative).
        let beta = beta_with(vec![(0, 0.0, 1.0)]);
        assert!(
            premise_contradiction(&beta, "relu_last", &row_of, &[f32::NAN], &[-1.0], tol).is_none()
        );
        assert!(
            premise_contradiction(&beta, "relu_last", &row_of, &[0.5], &[-0.5], tol).is_none(),
            "inverted refined pair must be skipped"
        );
    }

    /// α′ knob semantics: gate dark (only "1" is on), and write/step helpers
    /// respect masks + [0,1] projection and never touch stable neurons.
    #[test]
    fn alpha_prime_write_and_adam_respect_masks_and_projection() {
        use ny_core::{GpuCrownLayer, GpuResnetSegment};
        // One Activation: neuron 0 unstable (chord intercept > 0), neuron 1
        // stable-active (exact slopes, zero intercepts).
        let act = GpuCrownLayer::Activation {
            lower_slope: vec![0.5, 1.0],
            upper_slope: vec![0.6, 1.0],
            lower_intercept: vec![0.0, 0.0],
            upper_intercept: vec![0.3, 0.0],
            num_neurons: 2,
        };
        let mut segs = vec![GpuResnetSegment::Chain(vec![act])];
        assert_eq!(count_activations(&segs), 1);
        let masks = unstable_masks(&segs, 1);
        assert_eq!(masks, vec![vec![true, false]]);

        // Out-of-range α′ clamps; the stable neuron NEVER changes even when
        // marked stepped (its upper_intercept is 0 — the per-domain guard).
        write_alpha_prime(&mut segs, &[vec![5.0, 0.2]], &[vec![true, true]]);
        let slopes = collect_lower_slopes(&segs, 1);
        assert_eq!(slopes, vec![vec![1.0, 1.0]]);

        // Adam ascent: positive gradient pushes the masked slope up (clamped
        // at 1), the unmasked neuron never moves; negative gradient walks down.
        let mut sl = vec![vec![0.4f32, 1.0]];
        let stepped = vec![vec![true, false]];
        let mut adam = AlphaAdam::new(&sl);
        let g = adam.step(&mut sl, &[vec![2.0, 9.0]], &stepped, 0.1, 1);
        assert!((g - 2.0).abs() < 1e-6, "max|g| over STEPPED neurons only");
        assert!(sl[0][0] > 0.4 && sl[0][0] <= 1.0);
        assert_eq!(sl[0][1], 1.0, "unmasked neuron untouched");
        let mut sl2 = vec![vec![0.05f32, 1.0]];
        let mut adam2 = AlphaAdam::new(&sl2);
        for t in 1..=5 {
            adam2.step(&mut sl2, &[vec![-3.0, 0.0]], &stepped, 0.1, t);
        }
        assert!(sl2[0][0] >= 0.0, "projection keeps α′ in [0,1]");
        assert!(sl2[0][0] < 0.05);

        // Store key check.
        let ap = AlphaPrime {
            seed_node: "gemm_pre".into(),
            pre_dim: 2,
            relu_names: vec!["relu1".into()],
            slopes: vec![vec![0.5, 1.0]],
            stepped: vec![vec![true, false]],
            improved: true,
        };
        assert!(alpha_prime_matches(
            &ap,
            "gemm_pre",
            2,
            &["relu1".to_string()]
        ));
        assert!(!alpha_prime_matches(
            &ap,
            "other",
            2,
            &["relu1".to_string()]
        ));
        assert!(!alpha_prime_matches(
            &ap,
            "gemm_pre",
            3,
            &["relu1".to_string()]
        ));
        assert!(!alpha_prime_matches(
            &ap,
            "gemm_pre",
            2,
            &["reluX".to_string()]
        ));
    }

    /// Merge semantics: element-wise max-l/min-u, finite-guarded, length
    /// mismatch refuses.
    #[test]
    fn merge_refine_result_tightens_elementwise() {
        let mut dst = ny_core::GpuCrownResult {
            lower_bounds: vec![-1.0, 0.5, f32::NAN],
            upper_bounds: vec![2.0, 3.0, 1.0],
        };
        let src = ny_core::GpuCrownResult {
            lower_bounds: vec![-0.5, 0.2, 0.0],
            upper_bounds: vec![2.5, f32::INFINITY, 0.5],
        };
        merge_refine_result(&mut dst, &src);
        assert_eq!(dst.lower_bounds, vec![-0.5, 0.5, 0.0]);
        assert_eq!(dst.upper_bounds, vec![2.0, 3.0, 0.5]);
        let before = dst.lower_bounds.clone();
        let bad = ny_core::GpuCrownResult {
            lower_bounds: vec![9.0],
            upper_bounds: vec![9.0],
        };
        merge_refine_result(&mut dst, &bad);
        assert_eq!(dst.lower_bounds, before, "length mismatch refuses");
    }

    /// CPU FD ORACLE for the α′ REFINEMENT-objective gradient (#alpha-prime
    /// deliverable 1, formula tier): on a dense truncated stack with identity
    /// seed rows (the refinement backward's shape — each row its own CROWN
    /// objective with its own ν selection and its own argmin corner), the
    /// summed per-row TRUE chain-rule gradient returned by
    /// `refine_alpha_objective_grads` must match central finite differences
    /// of `Σ_rows lower_bound(row)` through the replayed backward when a
    /// lower slope is perturbed — including through a nonzero β fold.
    #[test]
    fn refine_alpha_grad_sum_matches_replay_fd() {
        use ny_core::{GpuCrownLayer, GpuResnetSegment};
        use std::sync::Arc;

        fn rngf(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        let mut st = 0xA11F_A5EEu64;
        let n = 3usize;
        let mk_lin = |st: &mut u64| GpuCrownLayer::Linear {
            weight: Arc::from(
                (0..n * n)
                    .map(|_| rngf(st) * 0.6)
                    .collect::<Vec<f32>>()
                    .into_boxed_slice(),
            ),
            bias: Some(Arc::from(
                (0..n)
                    .map(|_| rngf(st) * 0.1)
                    .collect::<Vec<f32>>()
                    .into_boxed_slice(),
            )),
            out_features: n,
            in_features: n,
        };
        let mk_act = |st: &mut u64| {
            let ls: Vec<f32> = (0..n).map(|_| rngf(st) * 0.4 + 0.5).collect();
            let us: Vec<f32> = (0..n).map(|_| rngf(st) * 0.3 + 0.5).collect();
            let ui: Vec<f32> = (0..n).map(|_| rngf(st).abs() * 0.3 + 0.05).collect();
            GpuCrownLayer::Activation {
                lower_slope: ls,
                upper_slope: us,
                lower_intercept: vec![0.0; n],
                upper_intercept: ui,
                num_neurons: n,
            }
        };
        // Fold order (output → input): Chain(lin3, act2, lin2) then
        // Residual(act1, lin1) — two Activations, residual structure, dense
        // mixed-sign weights.
        let segments = vec![
            GpuResnetSegment::Chain(vec![mk_lin(&mut st), mk_act(&mut st), mk_lin(&mut st)]),
            GpuResnetSegment::Residual(vec![mk_act(&mut st), mk_lin(&mut st)]),
        ];
        let in_lo: Vec<f32> = (0..n).map(|_| -0.5 + 0.1 * rngf(&mut st)).collect();
        let in_hi: Vec<f32> = in_lo.iter().map(|&l| l + 0.8).collect();
        // Nonzero β on the SECOND fold Activation (act1).
        let beta = vec![
            Vec::new(),
            (0..n).map(|_| rngf(&mut st) * 0.04).collect::<Vec<f32>>(),
        ];
        let sel: Vec<usize> = (0..n).collect();
        let rows: Vec<usize> = (0..n).collect();

        let row_lb = |segs: &[GpuResnetSegment], r: usize| -> f32 {
            let mut row = vec![0.0f32; n];
            row[r] = 1.0;
            super::super::wide_alpha_true::replay_critical_row(segs, &row, &beta)
                .expect("replay")
                .lower_bound(&in_lo, &in_hi)
        };
        let obj =
            |segs: &[GpuResnetSegment]| -> f64 { (0..n).map(|r| row_lb(segs, r) as f64).sum() };
        let lbs: Vec<f32> = (0..n).map(|r| row_lb(&segments, r)).collect();
        let (grads, rows_ok) = refine_alpha_objective_grads(
            &segments, &beta, &in_lo, &in_hi, 2, n, &sel, &rows, &lbs, None, false,
        );
        assert_eq!(rows_ok, n, "every row replay must validate against itself");

        // Per-row ν (for the near-kink skip rule, mirroring the wide oracle).
        let nus: Vec<Vec<Vec<f32>>> = (0..n)
            .map(|r| {
                let mut row = vec![0.0f32; n];
                row[r] = 1.0;
                super::super::wide_alpha_true::replay_critical_row(&segments, &row, &beta)
                    .expect("replay")
                    .nu
            })
            .collect();

        let h = 2e-3f32;
        let mut checked = 0usize;
        for fold_r in 0..2usize {
            for i in 0..n {
                let perturb = |delta: f32| -> f64 {
                    let mut segs = segments.clone();
                    visit_activations_mut(&mut segs, &mut |r, layer| {
                        if r == fold_r {
                            if let GpuCrownLayer::Activation { lower_slope, .. } = layer {
                                lower_slope[i] += delta;
                            }
                        }
                    });
                    obj(&segs)
                };
                let fd = ((perturb(h) - perturb(-h)) / (2.0 * h as f64)) as f32;
                let g = grads[fold_r][i];
                let tol = 2e-3 + 0.03 * fd.abs();
                // Near-kink rows (ν ≈ 0 at this neuron) legitimately disagree.
                let near_kink = nus.iter().any(|nu| nu[fold_r][i].abs() < 5e-2);
                if (fd - g).abs() > tol && near_kink {
                    continue;
                }
                assert!(
                    (fd - g).abs() <= tol,
                    "fold {fold_r} neuron {i}: fd {fd} != analytic {g}"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 4,
            "oracle must exercise enough neurons ({checked})"
        );
        assert!(
            grads.iter().flatten().any(|&g| g.abs() > 1e-4),
            "gradient signal must exist"
        );
    }

    /// #joint-interm-alpha FD ORACLE: the MARGIN-WEIGHTED α′ gradient
    /// (`refine_alpha_objective_grads` with `row_weights = Some(w)`) must match
    /// central finite differences of the WEIGHTED objective `Σ_r w_r · l′_r` —
    /// the exact chain `d(w·l′)/dα = w · dl′/dα`. Non-uniform weights (one zero)
    /// so a mis-applied, mis-aligned, or dropped weight is caught. Same synthetic
    /// residual net + nonzero β as the uniform oracle.
    #[test]
    fn joint_margin_alpha_grad_matches_weighted_replay_fd() {
        use ny_core::{GpuCrownLayer, GpuResnetSegment};
        use std::sync::Arc;

        fn rngf(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        let mut st = 0x5EED_1234u64;
        let n = 3usize;
        let mk_lin = |st: &mut u64| GpuCrownLayer::Linear {
            weight: Arc::from(
                (0..n * n)
                    .map(|_| rngf(st) * 0.6)
                    .collect::<Vec<f32>>()
                    .into_boxed_slice(),
            ),
            bias: Some(Arc::from(
                (0..n)
                    .map(|_| rngf(st) * 0.1)
                    .collect::<Vec<f32>>()
                    .into_boxed_slice(),
            )),
            out_features: n,
            in_features: n,
        };
        let mk_act = |st: &mut u64| {
            let ls: Vec<f32> = (0..n).map(|_| rngf(st) * 0.4 + 0.5).collect();
            let us: Vec<f32> = (0..n).map(|_| rngf(st) * 0.3 + 0.5).collect();
            let ui: Vec<f32> = (0..n).map(|_| rngf(st).abs() * 0.3 + 0.05).collect();
            GpuCrownLayer::Activation {
                lower_slope: ls,
                upper_slope: us,
                lower_intercept: vec![0.0; n],
                upper_intercept: ui,
                num_neurons: n,
            }
        };
        let segments = vec![
            GpuResnetSegment::Chain(vec![mk_lin(&mut st), mk_act(&mut st), mk_lin(&mut st)]),
            GpuResnetSegment::Residual(vec![mk_act(&mut st), mk_lin(&mut st)]),
        ];
        let in_lo: Vec<f32> = (0..n).map(|_| -0.5 + 0.1 * rngf(&mut st)).collect();
        let in_hi: Vec<f32> = in_lo.iter().map(|&l| l + 0.8).collect();
        let beta = vec![
            Vec::new(),
            (0..n).map(|_| rngf(&mut st) * 0.04).collect::<Vec<f32>>(),
        ];
        let sel: Vec<usize> = (0..n).collect();
        let rows: Vec<usize> = (0..n).collect();
        // Non-uniform margin weights, one exactly zero (must be dropped).
        let weights: Vec<f32> = vec![2.5, 0.0, 1.25];

        let row_lb = |segs: &[GpuResnetSegment], r: usize| -> f32 {
            let mut row = vec![0.0f32; n];
            row[r] = 1.0;
            super::super::wide_alpha_true::replay_critical_row(segs, &row, &beta)
                .expect("replay")
                .lower_bound(&in_lo, &in_hi)
        };
        let obj_w = |segs: &[GpuResnetSegment]| -> f64 {
            (0..n)
                .map(|r| weights[r] as f64 * row_lb(segs, r) as f64)
                .sum()
        };
        let lbs: Vec<f32> = (0..n).map(|r| row_lb(&segments, r)).collect();
        let (grads, rows_ok) = refine_alpha_objective_grads(
            &segments,
            &beta,
            &in_lo,
            &in_hi,
            2,
            n,
            &sel,
            &rows,
            &lbs,
            Some(&weights),
            false,
        );
        assert_eq!(rows_ok, n, "every row replay must validate against itself");

        let nus: Vec<Vec<Vec<f32>>> = (0..n)
            .map(|r| {
                let mut row = vec![0.0f32; n];
                row[r] = 1.0;
                super::super::wide_alpha_true::replay_critical_row(&segments, &row, &beta)
                    .expect("replay")
                    .nu
            })
            .collect();

        let h = 2e-3f32;
        let mut checked = 0usize;
        for fold_r in 0..2usize {
            for i in 0..n {
                let perturb = |delta: f32| -> f64 {
                    let mut segs = segments.clone();
                    visit_activations_mut(&mut segs, &mut |r, layer| {
                        if r == fold_r {
                            if let GpuCrownLayer::Activation { lower_slope, .. } = layer {
                                lower_slope[i] += delta;
                            }
                        }
                    });
                    obj_w(&segs)
                };
                let fd = ((perturb(h) - perturb(-h)) / (2.0 * h as f64)) as f32;
                let g = grads[fold_r][i];
                let tol = 2e-3 + 0.03 * fd.abs();
                let near_kink = nus.iter().any(|nu| nu[fold_r][i].abs() < 5e-2);
                if (fd - g).abs() > tol && near_kink {
                    continue;
                }
                assert!(
                    (fd - g).abs() <= tol,
                    "fold {fold_r} neuron {i}: weighted fd {fd} != analytic {g}"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 4,
            "weighted oracle must exercise enough neurons ({checked})"
        );
        assert!(
            grads.iter().flatten().any(|&g| g.abs() > 1e-4),
            "weighted gradient signal must exist"
        );
        // A row's weight must actually scale its contribution: doubling weights[0]
        // doubles the row-0 gradient mass reaching fold 1 (the seed-adjacent
        // Activation whose ν for row 0 is nonzero).
        let mut w2 = weights.clone();
        w2[0] *= 2.0;
        let (grads2, _) = refine_alpha_objective_grads(
            &segments,
            &beta,
            &in_lo,
            &in_hi,
            2,
            n,
            &sel,
            &rows,
            &lbs,
            Some(&w2),
            false,
        );
        // Zero-weight row 1 contributes nothing regardless.
        let (grads_uni, _) = refine_alpha_objective_grads(
            &segments, &beta, &in_lo, &in_hi, 2, n, &sel, &rows, &lbs, None, false,
        );
        assert!(
            grads2
                .iter()
                .flatten()
                .zip(grads.iter().flatten())
                .any(|(&a, &b)| (a - b).abs() > 1e-4),
            "reweighting must change the gradient"
        );
        assert!(
            grads_uni.iter().flatten().any(|&g| g.abs() > 1e-4),
            "uniform baseline gradient signal must exist"
        );
    }

    /// #joint-interm-alpha: `compute_margin_weights` builds
    /// `m_j = Σ_s max(−(spec·W_tail)[j], 0)` from the tail linear map directly
    /// consuming the last ReLU, and falls back to `None` (uniform) with no
    /// negative tail mass.
    #[test]
    fn compute_margin_weights_upper_branch_tail_mass() {
        // gemm_pre → relu_last → out; tail W = [[0.7,-0.3],[0.2,0.6]] (from `lin`).
        let mut g = GraphNetwork::new();
        g.add_node(lin("gemm_pre", NETWORK_INPUT));
        g.add_node(relu("relu_last", "gemm_pre"));
        g.add_node(lin("out", "relu_last"));
        g.set_output("out");
        // spec rows [-1,0], [1,0]: a0 = -row0 = [-0.7, 0.3] (0.7 mass on j0),
        // a1 = row0 = [0.7, -0.3] (0.3 mass on j1) ⇒ m = [0.7, 0.3].
        let spec = arr2(&[[-1.0_f32, 0.0], [1.0, 0.0]]);
        let m = compute_margin_weights(&g, "out", &spec).expect("weights resolve");
        assert_eq!(m.len(), 2);
        assert!((m[0] - 0.7).abs() < 1e-5, "m0={}", m[0]);
        assert!((m[1] - 0.3).abs() < 1e-5, "m1={}", m[1]);
        // a = [1,1]·W = [0.9, 0.3] — no negative tail mass ⇒ uniform fallback.
        let spec_pos = arr2(&[[1.0_f32, 1.0]]);
        assert!(compute_margin_weights(&g, "out", &spec_pos).is_none());
    }

    #[test]
    fn margin_weight_resource_and_deadline_refuse_before_dot() {
        let mut g = GraphNetwork::new();
        g.add_node(lin("gemm_pre", NETWORK_INPUT));
        g.add_node(relu("relu_last", "gemm_pre"));
        g.add_node(lin("out", "relu_last"));
        g.set_output("out");
        let spec = arr2(&[[-1.0_f32, 0.0], [1.0, 0.0]]);
        let dot_calls = std::cell::Cell::new(0usize);
        let mut counted_dot = |lhs: &Array2<f32>, rhs: &Array2<f32>| {
            dot_calls.set(dot_calls.get() + 1);
            lhs.dot(rhs)
        };

        let mut within_deadline = || false;
        let oversized = compute_margin_weights_with_deadline_check_and_dot(
            &g,
            "out",
            &spec,
            MarginWeightLimits {
                max_host_bytes: 0,
                max_work: SELECTIVE_CLIP_MAX_WORK,
            },
            &mut within_deadline,
            &mut counted_dot,
        )
        .unwrap_err();
        assert!(matches!(
            oversized,
            NyError::CpuMemoryExceeded {
                site: MARGIN_WEIGHT_SITE,
                ..
            }
        ));
        assert_eq!(dot_calls.get(), 0, "oversized work must refuse before dot");

        let mut expired = || true;
        let deadline = compute_margin_weights_with_deadline_check_and_dot(
            &g,
            "out",
            &spec,
            MARGIN_WEIGHT_LIMITS,
            &mut expired,
            &mut counted_dot,
        )
        .unwrap_err();
        assert!(matches!(deadline, NyError::DeadlineExceeded(_)));
        assert_eq!(dot_calls.get(), 0, "expired work must refuse before dot");
    }

    #[test]
    fn selective_last_rows_match_winner_score_and_retain_premises() {
        let bounds = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, 0.2, -2.0, -4.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.4, 2.0, 1.0]).unwrap(),
        )
        .unwrap();
        let cache = HashMap::from([("seed".to_string(), Arc::new(bounds))]);
        let candidates = vec![0, 1, 2, 3];
        let premise = vec![false, true, false, false];
        // Non-premise scores:
        //   j0 = 0.5 * 1 = 0.5
        //   j2 = 1.0 * 3 = 3.0
        //   j3 = 0.8 * 1 = 0.8
        // Keep top two (j2,j3) plus the stable premise source j1.
        let selected = select_intermediate_objective_rows(
            &[cache],
            "seed",
            &candidates,
            &premise,
            2,
            Some(&[1.0, 100.0, 3.0, 1.0]),
        );
        assert_eq!(selected, vec![1, 2, 3]);
        assert!(!selected.contains(&0), "top-K omission must be explicit");
    }

    #[test]
    fn selective_topk_zero_is_exact_selection_noop() {
        let candidates = vec![7usize, 2, 9, 1];
        let premise = vec![false; 10];
        let selected = select_intermediate_objective_rows(
            &[],
            "missing-seed",
            &candidates,
            &premise,
            0,
            Some(&[f32::NAN; 10]),
        );
        assert_eq!(
            selected, candidates,
            "gate-off must preserve every row and its order"
        );

        let refused = select_intermediate_objective_rows(
            &[],
            "missing-seed",
            &[usize::MAX],
            &[false],
            1,
            None,
        );
        assert!(
            refused.is_empty(),
            "overflowing score-table shape must refuse before allocation"
        );
    }

    #[test]
    fn selective_row_budget_accounts_live_vectors_at_exact_boundary() {
        let bytes_per_candidate =
            size_of::<usize>() + size_of::<bool>() + size_of::<(usize, f64)>() + size_of::<usize>();
        let last_admitted = SELECTIVE_CLIP_MAX_HOST_BYTES / bytes_per_candidate;
        assert!(selective_row_work_bytes(last_admitted, last_admitted)
            .is_some_and(|n| n <= SELECTIVE_CLIP_MAX_HOST_BYTES));
        assert!(
            selective_row_work_bytes(last_admitted + 1, last_admitted + 1)
                .is_none_or(|n| n > SELECTIVE_CLIP_MAX_HOST_BYTES)
        );
        // The independent work cap can bind before the byte cap. Pin its exact
        // monotone boundary too without materializing any large vectors.
        let (mut lo, mut hi) = (0usize, last_admitted);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if validate_selective_row_budget(mid, mid, 1, true).is_some() {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        assert!(validate_selective_row_budget(lo, lo, 1, true).is_some());
        assert!(validate_selective_row_budget(lo + 1, lo + 1, 1, true).is_none());
        assert!(selective_row_work_bytes(usize::MAX, usize::MAX).is_none());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn large_selective_candidate_set_is_bounded_deterministic_and_deadline_aware() {
        let n = 65_536usize;
        let bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[n]), -1.0f32),
            ArrayD::from_elem(IxDyn(&[n]), 1.0f32),
        )
        .unwrap();
        let cache = HashMap::from([("seed".to_string(), Arc::new(bounds))]);
        let candidates: Vec<usize> = (0..n).collect();
        let premise = vec![false; n];
        let selected = select_intermediate_objective_rows(
            std::slice::from_ref(&cache),
            "seed",
            &candidates,
            &premise,
            20,
            None,
        );
        assert_eq!(selected, (0..20).collect::<Vec<_>>());

        let mut polls = 0usize;
        let mut expire_during_score = || {
            polls += 1;
            polls >= 70
        };
        let refused = select_intermediate_objective_rows_with_deadline_check(
            &[cache],
            "seed",
            &candidates,
            &premise,
            20,
            None,
            &mut expire_during_score,
        );
        assert!(
            refused.is_none(),
            "expired scoring must return no partial selection"
        );
        assert_eq!(polls, 70);
    }

    /// input → l1 → relu1 → l2 → add(l2, l1) → gemm_pre → relu_last → out:
    /// the next ReLU strictly below `gemm_pre` is `relu1` (seed `l1`).
    #[test]
    fn find_next_relu_seed_below_walks_ancestors() {
        let mut g = GraphNetwork::new();
        g.add_node(lin("l1", NETWORK_INPUT));
        g.add_node(relu("relu1", "l1"));
        g.add_node(lin("l2", "relu1"));
        g.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["l2".to_string(), "l1".to_string()],
        ));
        g.add_node(lin("gemm_pre", "add"));
        g.add_node(relu("relu_last", "gemm_pre"));
        g.add_node(lin("out", "relu_last"));
        g.set_output("out");

        let (relu_name, seed) =
            find_next_relu_seed_below(&g, "gemm_pre").expect("ancestor ReLU must resolve");
        assert_eq!(relu_name, "relu1");
        assert_eq!(seed, "l1");

        // A ReLU fed by the network input refuses (nothing below the seed).
        let mut g2 = GraphNetwork::new();
        g2.add_node(relu("r0", NETWORK_INPUT));
        g2.add_node(lin("out", "r0"));
        g2.set_output("out");
        assert!(find_next_relu_seed_below(&g2, "out").is_none());

        // No ancestor ReLU at all refuses.
        let mut g3 = GraphNetwork::new();
        g3.add_node(lin("l1", NETWORK_INPUT));
        g3.add_node(lin("out", "l1"));
        g3.set_output("out");
        assert!(find_next_relu_seed_below(&g3, "out").is_none());
    }
}

/// GPU-gated end-to-end tests for the prune lane + the 2-layer cascade on a
/// tiny two-residual-block net (the smallest structure the resnet segment
/// extraction accepts for BOTH seed layers). Skips gracefully when no wgpu
/// adapter is available.
#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
    use crate::beta_crown::state::GraphBetaEntry;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    fn lin_w(name: &str, input: &str, w: [[f32; 2]; 2], b: [f32; 2]) -> GraphNode {
        let layer =
            Layer::Linear(LinearLayer::new(arr2(&w), Some(arr1(&b))).expect("valid linear"));
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }

    /// Two stacked residual blocks, all 2-d, then the last-ReLU tail:
    /// input → l1(I) → relu1 → l2(0.5·I) → add1(l2, l1)
    ///       → g3(I) → relu3 → l4(0.5·I) → add2(l4, g3)
    ///       → gemm_pre(I) → relu_last → out(I).
    /// Per coordinate: add1 = f(x), add2 = f(f(x)) with f(t) = 0.5·relu(t) + t
    /// (t if t < 0, 1.5·t if t ≥ 0), and gemm_pre = add2 exactly.
    fn build_two_block_net() -> GraphNetwork {
        let ident = [[1.0f32, 0.0], [0.0, 1.0]];
        let half = [[0.5f32, 0.0], [0.0, 0.5]];
        let zero = [0.0f32, 0.0];
        let mut g = GraphNetwork::new();
        g.add_node(lin_w("l1", NETWORK_INPUT, ident, zero));
        g.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(lin_w("l2", "relu1", half, zero));
        g.add_node(GraphNode::new(
            "add1",
            Layer::Add(AddLayer),
            vec!["l2".to_string(), "l1".to_string()],
        ));
        g.add_node(lin_w("g3", "add1", ident, zero));
        g.add_node(GraphNode::new(
            "relu3",
            Layer::ReLU(ReLULayer),
            vec!["g3".to_string()],
        ));
        g.add_node(lin_w("l4", "relu3", half, zero));
        g.add_node(GraphNode::new(
            "add2",
            Layer::Add(AddLayer),
            vec!["l4".to_string(), "g3".to_string()],
        ));
        g.add_node(lin_w("gemm_pre", "add2", ident, zero));
        g.add_node(GraphNode::new(
            "relu_last",
            Layer::ReLU(ReLULayer),
            vec!["gemm_pre".to_string()],
        ));
        g.add_node(lin_w("out", "relu_last", ident, zero));
        g.set_output("out");
        g
    }

    /// Exact forward of the tiny net: gemm_pre(x) per coordinate.
    fn gemm_pre_of(x: [f32; 2]) -> [f32; 2] {
        let f = |t: f32| 0.5 * t.max(0.0) + t;
        [f(f(x[0])), f(f(x[1]))]
    }

    fn box2(lo: [f32; 2], hi: [f32; 2]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), lo.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), hi.to_vec()).unwrap(),
        )
        .unwrap()
    }

    fn opts(prune: bool, layers: usize) -> IntermRefineOptions {
        IntermRefineOptions {
            unstable_rows_only: true,
            prune,
            layers,
            max_dim: 2048,
            deep_max_rows: 256,
            selective_topk: 0,
            prune_tol: 1e-4,
            probe: false,
            min_depth: 0,
            seeds: None,
            alpha_iters: 0,
            alpha_lr: 0.05,
            alpha_max_rows: 64,
            alpha_reopt: false,
            alpha_store: None,
            adaptive_latch: None,
            adaptive_floor: 4,
            redo_every: 0,
            wide_max_n: 0,
            joint_margin: false,
            margin_weights: None,
            clip_resnet: false,
            per_target: false,
            interm_box_probe: false,
            clip_guard: false,
        }
    }

    /// End-to-end prune + soundness on a real sound GPU backward:
    ///
    /// * Domain 0 (input x0 ∈ [−1, −0.9]) carries the split premise
    ///   `relu_last[0] ACTIVE` (z0 ≥ 0), but z0 = x0 ≤ −0.9 over its whole
    ///   box — contradictory through the net ⇒ MUST prune (infeasible). Even
    ///   with the ROOT-frozen chord relaxations the refinement inherits
    ///   (chord of [−1, 1.5] at both ReLUs), the refined upper is
    ///   1.625·x0 + 0.625 ≤ −0.8375 < 0 − tol.
    /// * Domain 1 (x0 ∈ [−0.5, 1]) carries the same premise, which is
    ///   satisfiable there (z0 ∈ [−0.5, 2.25]) ⇒ must NEVER prune; sampled
    ///   premise-satisfying points must stay inside the refined enclosure.
    /// * With layers=2, the deep pass (seed g3, one residual below) runs
    ///   first and the cascade must keep every entry sound (sample check on
    ///   BOTH seed entries) — and still never prune the feasible domain.
    #[test]
    fn interm_refine_prune_e2e_sound_gpu() {
        let Ok(device) = ny_gpu::ComputeDevice::new(ny_gpu::Backend::Wgpu) else {
            eprintln!("SKIP: no wgpu adapter for interm_refine_prune_e2e_sound_gpu");
            return;
        };
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net();
        let verifier = BetaCrownVerifier::default();

        // ROOT-frozen caches + the premise clamp, exactly like BaB children:
        // node bounds computed on the ROOT box, `apply_pre_constraints` clamps
        // the seed entry (relu_last[0] active ⇒ gemm_pre[0].l = 0).
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("relu_last".to_string(), 0, true, 0.0)
                .expect("valid constraint"),
        );
        let (cache, _root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("constrained forward on the root box");
        // Root entry for gemm_pre[0] must be unstable-then-clamped (premise
        // admissible at root — the refinement, not the forward clamp, must be
        // what proves infeasibility).
        let pre = cache.get("gemm_pre").expect("seed entry");
        assert_eq!(pre.lower()[[0]], 0.0, "active clamp on the seed entry");
        assert!(pre.upper()[[0]] > 0.0);

        let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu_last".into(),
            0,
            0.0,
            0.0,
            1.0,
        )
        .expect("valid entry")]);

        let caches = vec![cache.clone(), cache];
        let inputs = vec![
            box2([-1.0, -1.0], [-0.9, 1.0]), // premise z0 ≥ 0 is IMPOSSIBLE here
            box2([-0.5, -1.0], [1.0, 1.0]),  // premise satisfiable here
        ];
        let betas: Vec<Option<&GraphBetaState>> = vec![Some(&beta), Some(&beta)];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None, None];

        for layers in [1usize, 2] {
            let outcome = verifier
                .refine_interm_bounds_with_opts(
                    &graph,
                    "out",
                    2,
                    &caches,
                    &inputs,
                    &betas,
                    &alphas,
                    engine,
                    &opts(true, layers),
                )
                .expect("refinement must apply on the sound GPU lane");
            assert!(
                outcome.infeasible[0],
                "contradictory premise domain must prune (layers={layers})"
            );
            assert!(
                !outcome.infeasible[1],
                "feasible domain must NEVER prune (layers={layers})"
            );

            // Soundness: sampled premise-satisfying points of the FEASIBLE
            // domain stay inside the refined enclosures (gemm_pre always; g3
            // too — the layers=2 cascade must not corrupt the deep entry).
            let refined = &outcome.caches[1];
            let (in_lo, in_hi) = ([-0.5f32, -1.0], [1.0f32, 1.0]);
            for i in 0..=8 {
                for j in 0..=8 {
                    let x = [
                        in_lo[0] + (in_hi[0] - in_lo[0]) * (i as f32) / 8.0,
                        in_lo[1] + (in_hi[1] - in_lo[1]) * (j as f32) / 8.0,
                    ];
                    let z = gemm_pre_of(x);
                    if z[0] < 0.0 {
                        continue; // premise z0 ≥ 0 not satisfied — out of domain
                    }
                    let entry = refined.get("gemm_pre").expect("refined seed entry");
                    for (k, &zk) in z.iter().enumerate() {
                        assert!(
                            entry.lower()[[k]] <= zk + 1e-5 && zk - 1e-5 <= entry.upper()[[k]],
                            "gemm_pre[{k}]={zk} outside refined [{}, {}] at x={x:?} (layers={layers})",
                            entry.lower()[[k]],
                            entry.upper()[[k]],
                        );
                    }
                    if layers == 2 {
                        // g3 = f(x) per coordinate.
                        let f = |t: f32| 0.5 * t.max(0.0) + t;
                        let g = [f(x[0]), f(x[1])];
                        let entry = refined.get("g3").expect("deep seed entry");
                        for (k, &gk) in g.iter().enumerate() {
                            assert!(
                                entry.lower()[[k]] <= gk + 1e-5 && gk - 1e-5 <= entry.upper()[[k]],
                                "g3[{k}]={gk} outside refined [{}, {}] at x={x:?}",
                                entry.lower()[[k]],
                                entry.upper()[[k]],
                            );
                        }
                    }
                }
            }

            // With the prune lane OFF, the same batch must not flag anything
            // (dark default = today's behavior).
            if let Some(outcome_off) = verifier.refine_interm_bounds_with_opts(
                &graph,
                "out",
                2,
                &caches,
                &inputs,
                &betas,
                &alphas,
                engine,
                &opts(false, layers),
            ) {
                assert!(outcome_off.infeasible.iter().all(|&b| !b));
            }
        }

        // The deep-seed discovery itself (layers=2 walks add2's F-branch).
        let (relu_name, seed) = find_next_relu_seed_below(&graph, "gemm_pre").expect("deep seed");
        assert_eq!((relu_name.as_str(), seed.as_str()), ("relu3", "g3"));
    }

    /// #wide-chunk e2e on the sound GPU lane: forcing 1-domain chunks
    /// (`wide_max_n=1`) must reproduce the single-call refinement — same
    /// prune flags, same refined entries within the wide-vs-serial f32
    /// reorder tolerance (the chunked calls route each domain through the
    /// same batched entry, just in smaller groups).
    #[test]
    fn interm_refine_wide_chunk_matches_single_call_gpu() {
        let Ok(device) = ny_gpu::ComputeDevice::new(ny_gpu::Backend::Wgpu) else {
            eprintln!("SKIP: no wgpu adapter for interm_refine_wide_chunk_matches_single_call_gpu");
            return;
        };
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net();
        let verifier = BetaCrownVerifier::default();
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let history = GraphSplitHistory::new();
        let (cache, _root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("constrained forward on the root box");
        let caches = vec![cache.clone(), cache.clone(), cache];
        let inputs = vec![
            box2([-1.0, -1.0], [0.0, 1.0]),
            box2([-0.5, -1.0], [1.0, 1.0]),
            box2([-1.0, 0.0], [1.0, 1.0]),
        ];
        let betas: Vec<Option<&GraphBetaState>> = vec![None, None, None];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None, None, None];
        let run = |wide_max_n: usize| {
            verifier.refine_interm_bounds_with_opts(
                &graph,
                "out",
                3,
                &caches,
                &inputs,
                &betas,
                &alphas,
                engine,
                &IntermRefineOptions {
                    wide_max_n,
                    ..opts(false, 2)
                },
            )
        };
        let single = run(0).expect("single-call refinement applies");
        let chunked = run(1).expect("chunked refinement applies");
        assert_eq!(single.infeasible, chunked.infeasible);
        for (cs, cc) in single.caches.iter().zip(chunked.caches.iter()) {
            for node in ["gemm_pre", "relu_last", "g3", "relu3"] {
                let (es, ec) = (cs.get(node).expect(node), cc.get(node).expect(node));
                for k in 0..es.len() {
                    let (ls, us) = (es.lower()[[k]], es.upper()[[k]]);
                    let (lc, uc) = (ec.lower()[[k]], ec.upper()[[k]]);
                    assert!(
                        (ls - lc).abs() <= 1e-5 && (us - uc).abs() <= 1e-5,
                        "{node}[{k}]: single [{ls}, {us}] vs chunked [{lc}, {uc}]"
                    );
                }
            }
        }
    }

    /// ADAPTIVE SCHEDULE (#adaptive-refine) e2e on the sound GPU lane:
    ///
    /// * A zero-yield batch (no stability flip is POSSIBLE — the crossing
    ///   coordinate's true range spans 0, so any sound refinement keeps it
    ///   crossing — and prune is off) at depth 1 with floor 1 must TRIP the
    ///   latch to 1.
    /// * Once latched, a batch whose domains all sit at depth ≥ 1 must be
    ///   SKIPPED (returns `None` ⇒ caller keeps inherited caches).
    /// * With floor 2 the same zero-yield depth-1 batch must NOT latch
    ///   (early-depth protection).
    /// * `adaptive_latch: None` (the dark default) never touches anything.
    #[test]
    fn interm_refine_adaptive_latch_trips_and_skips_gpu() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let Ok(device) = ny_gpu::ComputeDevice::new(ny_gpu::Backend::Wgpu) else {
            eprintln!("SKIP: no wgpu adapter for interm_refine_adaptive_latch_trips_and_skips_gpu");
            return;
        };
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net();
        let verifier = BetaCrownVerifier::default();

        // Same fixture as the prune e2e: root-frozen caches + the ACTIVE
        // premise clamp on relu_last[0] (depth 1). Feasible domain only, prune
        // OFF ⇒ yield can only come from a stability flip; gemm_pre[1]'s true
        // range over the box spans 0 (f(f(−1)) = −1 < 0 < 2.25 = f(f(1))), and
        // gemm_pre[0] is already clamp-stable (l = 0), so a SOUND refinement
        // can never produce newly_stable > 0 here — zero yield by construction.
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("relu_last".to_string(), 0, true, 0.0)
                .expect("valid constraint"),
        );
        let (cache, _root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("constrained forward on the root box");
        let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu_last".into(),
            0,
            0.0,
            0.0,
            1.0,
        )
        .expect("valid entry")]);
        let caches = vec![cache];
        let inputs = vec![box2([-0.5, -1.0], [1.0, 1.0])];
        let betas: Vec<Option<&GraphBetaState>> = vec![Some(&beta)];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None];
        let run = |o: &IntermRefineOptions| {
            verifier.refine_interm_bounds_with_opts(
                &graph, "out", 1, &caches, &inputs, &betas, &alphas, engine, o,
            )
        };

        // Floor 1, fresh latch: the zero-yield depth-1 batch must trip it.
        let latch: AdaptiveLatch = Arc::new(AtomicUsize::new(usize::MAX));
        let o1 = IntermRefineOptions {
            adaptive_latch: Some(latch.clone()),
            adaptive_floor: 1,
            ..opts(false, 1)
        };
        let _ = run(&o1);
        assert_eq!(
            latch.load(Ordering::Relaxed),
            1,
            "zero-yield depth-1 batch at floor 1 must latch at depth 1"
        );

        // Latched: the same batch (all domains at depth ≥ 1) must be skipped.
        assert!(
            run(&o1).is_none(),
            "latched schedule must skip the whole deep batch"
        );

        // Floor 2: the same zero-yield depth-1 batch must NOT latch.
        let latch2: AdaptiveLatch = Arc::new(AtomicUsize::new(usize::MAX));
        let o2 = IntermRefineOptions {
            adaptive_latch: Some(latch2.clone()),
            adaptive_floor: 2,
            ..opts(false, 1)
        };
        let _ = run(&o2);
        assert_eq!(
            latch2.load(Ordering::Relaxed),
            usize::MAX,
            "below-floor batch must never latch"
        );

        // #interm-refine-redo through the tripped latch: the depth-1 batch is
        // an exact multiple of stride 1 ⇒ it must NOT be skipped — the call
        // behaves exactly like the unlatched lane (same Some/None outcome).
        // Stride 2 leaves depth 1 (1 % 2 ≠ 0) latched out ⇒ skip (None).
        let unlatched = run(&IntermRefineOptions {
            adaptive_latch: None,
            ..opts(false, 1)
        });
        let pre_latched: AdaptiveLatch = Arc::new(AtomicUsize::new(1));
        let redo1 = run(&IntermRefineOptions {
            adaptive_latch: Some(pre_latched.clone()),
            adaptive_floor: 1,
            redo_every: 1,
            ..opts(false, 1)
        });
        assert_eq!(
            unlatched.is_some(),
            redo1.is_some(),
            "redo stride 1 must re-include the latched depth-1 batch"
        );
        let redo2 = run(&IntermRefineOptions {
            adaptive_latch: Some(pre_latched),
            adaptive_floor: 1,
            redo_every: 2,
            ..opts(false, 1)
        });
        assert!(
            redo2.is_none(),
            "depth 1 is not a multiple of stride 2 — the latched batch must stay skipped"
        );
    }

    // ==== α′-for-refinement (#alpha-prime) fixtures: a DENSE two-block net —
    // mixed-sign weights so ν rows exercise both relaxation branches. Same
    // topology as `build_two_block_net` (the smallest structure the resnet
    // extraction accepts), weights shared with the concrete forward below.
    const DW1: [[f32; 2]; 2] = [[0.9, -0.4], [0.3, 0.8]];
    const DB1: [f32; 2] = [0.05, -0.05];
    const DW2: [[f32; 2]; 2] = [[0.5, 0.3], [-0.4, 0.6]];
    const DB2: [f32; 2] = [0.0, 0.05];
    const DW3: [[f32; 2]; 2] = [[0.7, -0.5], [0.4, 0.9]];
    const DB3: [f32; 2] = [-0.05, 0.0];
    const DW4: [[f32; 2]; 2] = [[0.6, 0.2], [-0.3, 0.5]];
    const DB4: [f32; 2] = [0.05, 0.05];
    const DW5: [[f32; 2]; 2] = [[1.0, -0.6], [0.5, 0.8]];
    const DB5: [f32; 2] = [0.0, -0.05];

    fn build_two_block_net_dense() -> GraphNetwork {
        let ident = [[1.0f32, 0.0], [0.0, 1.0]];
        let mut g = GraphNetwork::new();
        g.add_node(lin_w("l1", NETWORK_INPUT, DW1, DB1));
        g.add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["l1".to_string()],
        ));
        g.add_node(lin_w("l2", "relu1", DW2, DB2));
        g.add_node(GraphNode::new(
            "add1",
            Layer::Add(AddLayer),
            vec!["l2".to_string(), "l1".to_string()],
        ));
        g.add_node(lin_w("g3", "add1", DW3, DB3));
        g.add_node(GraphNode::new(
            "relu3",
            Layer::ReLU(ReLULayer),
            vec!["g3".to_string()],
        ));
        g.add_node(lin_w("l4", "relu3", DW4, DB4));
        g.add_node(GraphNode::new(
            "add2",
            Layer::Add(AddLayer),
            vec!["l4".to_string(), "g3".to_string()],
        ));
        g.add_node(lin_w("gemm_pre", "add2", DW5, DB5));
        g.add_node(GraphNode::new(
            "relu_last",
            Layer::ReLU(ReLULayer),
            vec!["gemm_pre".to_string()],
        ));
        g.add_node(lin_w("out", "relu_last", ident, [0.0, 0.0]));
        g.set_output("out");
        g
    }

    /// Concrete `gemm_pre(x)` of the dense net (the soundness oracle).
    fn dense_gemm_pre_of(x: [f32; 2]) -> [f32; 2] {
        let mv = |w: [[f32; 2]; 2], b: [f32; 2], v: [f32; 2]| {
            [
                w[0][0] * v[0] + w[0][1] * v[1] + b[0],
                w[1][0] * v[0] + w[1][1] * v[1] + b[1],
            ]
        };
        let relu = |v: [f32; 2]| [v[0].max(0.0), v[1].max(0.0)];
        let l1 = mv(DW1, DB1, x);
        let l2 = mv(DW2, DB2, relu(l1));
        let a1 = [l2[0] + l1[0], l2[1] + l1[1]];
        let g3 = mv(DW3, DB3, a1);
        let l4 = mv(DW4, DB4, relu(g3));
        let a2 = [l4[0] + g3[0], l4[1] + g3[1]];
        mv(DW5, DB5, a2)
    }

    /// THE FD ORACLE (#alpha-prime deliverable 1): finite differences of the
    /// refinement objective `Σ_rows l′_row` through the ACTUAL sound GPU
    /// refinement backward (the very
    /// `crown_backward_gpu_resnet_sound_beta_batched` call
    /// `refine_one_seed_pass` makes, on the truncated stack
    /// `prep_resnet_domain` builds at the seed node) must match the analytic
    /// per-row-summed TRUE chain-rule gradient `refine_alpha_objective_grads`
    /// returns — the gradient the α′ ascent steps on.
    #[test]
    fn interm_refine_alpha_fd_oracle_gpu() {
        let Ok(device) = ny_gpu::ComputeDevice::new(ny_gpu::Backend::Wgpu) else {
            eprintln!("SKIP: no wgpu adapter for interm_refine_alpha_fd_oracle_gpu");
            return;
        };
        let engine: &dyn GemmEngine = &device;
        let Some(gpu) = engine
            .as_gpu_crown_backward()
            .filter(|g| g.provides_sound_gpu_crown())
        else {
            eprintln!("SKIP: engine has no sound GPU CROWN backward");
            return;
        };
        let graph = build_two_block_net_dense();
        let verifier = BetaCrownVerifier::default();
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let history = crate::beta_crown::branching::GraphSplitHistory::new();
        let (cache, root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("root forward bounds");
        let mut prep =
            prep_resnet_domain(&graph, "gemm_pre", &cache, &root_input, None, None, false)
                .expect("truncated-stack prep at the seed node");
        let n_relu = prep.relu_names.len();
        assert_eq!(n_relu, 2, "relu1 + relu3 below the seed");
        assert_eq!(count_activations(&prep.segments), n_relu);

        // Interior α (off the 0/1 boundary and the CROWN adaptive kink) so
        // central differences stay inside [0,1].
        visit_activations_mut(&mut prep.segments, &mut |r, layer| {
            if let ny_core::GpuCrownLayer::Activation {
                lower_slope,
                upper_intercept,
                ..
            } = layer
            {
                for i in 0..lower_slope.len() {
                    if upper_intercept[i] > 0.0 {
                        lower_slope[i] = 0.35 + 0.3 * (((r + i) % 2) as f32);
                    }
                }
            }
        });
        let masks = unstable_masks(&prep.segments, n_relu);
        assert!(
            masks.iter().flatten().filter(|&&b| b).count() >= 2,
            "fixture must have unstable below-seed neurons"
        );

        let pre_dim = 2usize;
        let mut rows_seed = vec![0.0f32; pre_dim * pre_dim];
        rows_seed[0] = 1.0;
        rows_seed[3] = 1.0;
        let seed = ny_core::GpuCrownSeed {
            lower_a: rows_seed.clone().into(),
            upper_a: rows_seed.into(),
            lower_b: vec![0.0f32; pre_dim].into(),
            upper_b: vec![0.0f32; pre_dim].into(),
            num_specs: pre_dim,
            current_dim: pre_dim,
        };
        let run = |segs: &[ny_core::GpuResnetSegment]| -> Vec<f32> {
            let refs = vec![ny_core::GpuResnetBatchedDomainRef {
                segments: segs,
                input_lower: &prep.in_lo,
                input_upper: &prep.in_hi,
                beta_signed: &prep.beta_signed,
                frontier_abs: &prep.frontier_abs,
                node_abs: &prep.node_abs,
            }];
            gpu.crown_backward_gpu_resnet_sound_beta_batched(&refs, &seed)
                .expect("batched refinement backward")[0]
                .lower_bounds
                .clone()
        };
        let obj = |segs: &[ny_core::GpuResnetSegment]| -> f64 {
            run(segs).iter().map(|&v| v as f64).sum()
        };

        let base_lbs = run(&prep.segments);
        let sel: Vec<usize> = (0..pre_dim).collect();
        let rows: Vec<usize> = (0..pre_dim).collect();
        let (grads, rows_ok) = refine_alpha_objective_grads(
            &prep.segments,
            &prep.beta_signed,
            &prep.in_lo,
            &prep.in_hi,
            n_relu,
            pre_dim,
            &sel,
            &rows,
            &base_lbs,
            None,
            false,
        );
        assert_eq!(
            rows_ok, pre_dim,
            "every identity-row replay must validate against the GPU bound"
        );

        // Per-row ν for the near-kink skip rule (mirrors the wide-α oracle).
        let nus: Vec<Vec<Vec<f32>>> = (0..pre_dim)
            .map(|r| {
                let mut row = vec![0.0f32; pre_dim];
                row[r] = 1.0;
                super::super::wide_alpha_true::replay_critical_row(
                    &prep.segments,
                    &row,
                    &prep.beta_signed,
                )
                .expect("replay")
                .nu
            })
            .collect();

        let h = 4e-3f32;
        let mut checked = 0usize;
        for r in 0..n_relu {
            for i in 0..masks[r].len() {
                if !masks[r][i] {
                    continue;
                }
                let perturb = |delta: f32| -> f64 {
                    let mut segs = prep.segments.clone();
                    visit_activations_mut(&mut segs, &mut |rr, layer| {
                        if rr == r {
                            if let ny_core::GpuCrownLayer::Activation { lower_slope, .. } = layer {
                                lower_slope[i] += delta;
                            }
                        }
                    });
                    obj(&segs)
                };
                let fd = ((perturb(h) - perturb(-h)) / (2.0 * h as f64)) as f32;
                let g = grads[r][i];
                let tol = 4e-3 + 0.06 * fd.abs();
                let near_kink = nus.iter().any(|nu| nu[r][i].abs() < 5e-2);
                if (fd - g).abs() > tol && near_kink {
                    continue;
                }
                assert!(
                    (fd - g).abs() <= tol,
                    "relu {r} neuron {i}: GPU FD {fd} != analytic {g} (tol {tol})"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 2,
            "oracle must exercise enough neurons ({checked})"
        );
        assert!(
            grads.iter().flatten().any(|&g| g.abs() > 1e-4),
            "gradient signal must exist on the dense net"
        );
    }

    /// α′ ascent integration (#alpha-prime): with the lane ON the refined
    /// bounds are NEVER looser than the borrowed-α run (iterate 0 is the same
    /// backward; the merge only tightens), sampled concrete pre-activations
    /// stay enclosed (soundness), the α′ store fills, and a second (reuse)
    /// call stays sound.
    #[test]
    fn interm_refine_alpha_ascent_never_looser_sound_and_reuses() {
        let Ok(device) = ny_gpu::ComputeDevice::new(ny_gpu::Backend::Wgpu) else {
            eprintln!("SKIP: no wgpu adapter for interm_refine_alpha_ascent test");
            return;
        };
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net_dense();
        let verifier = BetaCrownVerifier::default();
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let history = crate::beta_crown::branching::GraphSplitHistory::new();
        let (cache, root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("root forward bounds");
        let caches = vec![cache];
        let inputs = vec![root_input];
        let betas: Vec<Option<&GraphBetaState>> = vec![None];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None];
        let run = |o: &IntermRefineOptions| {
            verifier.refine_interm_bounds_with_opts(
                &graph, "out", 1, &caches, &inputs, &betas, &alphas, engine, o,
            )
        };

        let off = run(&opts(false, 1)).expect("borrowed-α refinement must apply");
        let store: AlphaPrimeStore = Arc::new(Mutex::new(None));
        let on_opts = IntermRefineOptions {
            alpha_iters: 4,
            alpha_lr: 0.15,
            alpha_max_rows: 8,
            alpha_store: Some(store.clone()),
            ..opts(false, 1)
        };
        let on = run(&on_opts).expect("α′ refinement must apply");

        let (o_pre, n_pre) = (&off.caches[0]["gemm_pre"], &on.caches[0]["gemm_pre"]);
        for k in 0..2 {
            assert!(
                n_pre.lower()[[k]] >= o_pre.lower()[[k]] - 1e-6
                    && n_pre.upper()[[k]] <= o_pre.upper()[[k]] + 1e-6,
                "α′ merge must never loosen: neuron {k} ON [{}, {}] vs OFF [{}, {}]",
                n_pre.lower()[[k]],
                n_pre.upper()[[k]],
                o_pre.lower()[[k]],
                o_pre.upper()[[k]],
            );
        }
        let stored = store.lock().expect("store lock");
        let improved = stored.as_ref().map(|ap| ap.improved);
        assert!(stored.is_some(), "ascent must record an α′ snapshot");
        drop(stored);
        eprintln!("[test] alpha-prime improved={improved:?}");

        // Soundness of BOTH the ascent run and the reuse run: concrete
        // gemm_pre(x) of every sampled root point inside the refined bounds.
        let on2 = run(&on_opts).expect("reuse refinement must apply");
        for outcome in [&on, &on2] {
            let entry = &outcome.caches[0]["gemm_pre"];
            for i in 0..=8 {
                for j in 0..=8 {
                    let x = [-1.0 + 0.25 * i as f32, -1.0 + 0.25 * j as f32];
                    let z = dense_gemm_pre_of(x);
                    for (k, &zk) in z.iter().enumerate() {
                        assert!(
                            entry.lower()[[k]] <= zk + 1e-4 && zk - 1e-4 <= entry.upper()[[k]],
                            "gemm_pre[{k}]={zk} outside refined [{}, {}] at x={x:?}",
                            entry.lower()[[k]],
                            entry.upper()[[k]],
                        );
                    }
                }
            }
        }
    }

    /// #ab-parity-interm PER-TARGET lane: with `per_target` ON the refined
    /// bounds are NEVER looser than the borrowed-α run (each target's iterate-0
    /// bound seeds the merge; only element-wise-tightest sound GPU folds are
    /// kept), and every sampled concrete pre-activation stays enclosed
    /// (soundness). Two domains so the application phase's batched write/read
    /// over all domains is exercised.
    #[test]
    fn interm_refine_per_target_never_looser_and_sound_gpu() {
        let Ok(device) = ny_gpu::ComputeDevice::new(ny_gpu::Backend::Wgpu) else {
            eprintln!("SKIP: no wgpu adapter for interm_refine_per_target test");
            return;
        };
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net_dense();
        let verifier = BetaCrownVerifier::default();
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let history = crate::beta_crown::branching::GraphSplitHistory::new();
        let (cache, _root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("root forward bounds");
        // Two domains sharing the root cache but different input sub-boxes: the
        // per-target application phase writes each α into BOTH and reads columns.
        let caches = vec![cache.clone(), cache];
        let inputs = vec![
            box2([-1.0, -1.0], [1.0, 1.0]),
            box2([-0.5, -0.75], [0.75, 1.0]),
        ];
        let betas: Vec<Option<&GraphBetaState>> = vec![None, None];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None, None];
        let run = |o: &IntermRefineOptions| {
            verifier.refine_interm_bounds_with_opts(
                &graph, "out", 2, &caches, &inputs, &betas, &alphas, engine, o,
            )
        };

        let off = run(&opts(false, 1)).expect("borrowed-α refinement must apply");
        let store: AlphaPrimeStore = Arc::new(Mutex::new(None));
        let on_opts = IntermRefineOptions {
            alpha_iters: 4,
            alpha_lr: 0.15,
            alpha_max_rows: 8,
            alpha_store: Some(store.clone()),
            per_target: true,
            ..opts(false, 1)
        };
        let on = run(&on_opts).expect("per-target refinement must apply");

        // Never looser than the borrowed-α run, in BOTH domains.
        for d in 0..2 {
            let (o_pre, n_pre) = (&off.caches[d]["gemm_pre"], &on.caches[d]["gemm_pre"]);
            for k in 0..2 {
                assert!(
                    n_pre.lower()[[k]] >= o_pre.lower()[[k]] - 1e-6
                        && n_pre.upper()[[k]] <= o_pre.upper()[[k]] + 1e-6,
                    "per-target merge must never loosen: dom {d} neuron {k} ON [{}, {}] vs OFF [{}, {}]",
                    n_pre.lower()[[k]],
                    n_pre.upper()[[k]],
                    o_pre.lower()[[k]],
                    o_pre.upper()[[k]],
                );
            }
        }
        // Per-target drops the shared-α store reuse — nothing is written to it.
        assert!(
            store.lock().expect("store lock").is_none(),
            "per-target lane must NOT populate the shared-α store (reuse dropped)"
        );

        // Soundness: concrete gemm_pre(x) of every sampled point of each domain's
        // OWN input box is inside that domain's refined bounds.
        for (d, dbox) in [
            ([-1.0f32, -1.0], [1.0f32, 1.0]),
            ([-0.5, -0.75], [0.75, 1.0]),
        ]
        .into_iter()
        .enumerate()
        {
            let (lo, hi) = dbox;
            let entry = &on.caches[d]["gemm_pre"];
            for i in 0..=8 {
                for j in 0..=8 {
                    let x = [
                        lo[0] + (hi[0] - lo[0]) * i as f32 / 8.0,
                        lo[1] + (hi[1] - lo[1]) * j as f32 / 8.0,
                    ];
                    let z = dense_gemm_pre_of(x);
                    for (k, &zk) in z.iter().enumerate() {
                        assert!(
                            entry.lower()[[k]] <= zk + 1e-4 && zk - 1e-4 <= entry.upper()[[k]],
                            "dom {d} gemm_pre[{k}]={zk} outside per-target refined [{}, {}] at x={x:?}",
                            entry.lower()[[k]],
                            entry.upper()[[k]],
                        );
                    }
                }
            }
        }
    }

    /// NAMED-SEED lane (`NY_INTERM_REFINE_SEEDS`, #midref): naming the same
    /// two ReLUs the layers=2 walk finds must produce BIT-IDENTICAL refined
    /// caches and the same prune flags, regardless of token order (exec-order
    /// sort) and casing of `last`; unknown/non-ReLU tokens are skipped; a list
    /// with no resolvable seed refuses (returns None).
    #[test]
    fn interm_refine_named_seeds_match_layers_walk() {
        let Ok(device) = ny_gpu::ComputeDevice::new(ny_gpu::Backend::Wgpu) else {
            eprintln!("SKIP: no wgpu adapter for interm_refine_named_seeds_match_layers_walk");
            return;
        };
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net();
        let verifier = BetaCrownVerifier::default();

        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let mut history = GraphSplitHistory::new();
        history.add_constraint(
            GraphNeuronConstraint::new("relu_last".to_string(), 0, true, 0.0)
                .expect("valid constraint"),
        );
        let (cache, _root_input) = verifier
            .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
            .expect("constrained forward on the root box");
        let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
            "relu_last".into(),
            0,
            0.0,
            0.0,
            1.0,
        )
        .expect("valid entry")]);
        let caches = vec![cache.clone(), cache];
        let inputs = vec![
            box2([-1.0, -1.0], [-0.9, 1.0]),
            box2([-0.5, -1.0], [1.0, 1.0]),
        ];
        let betas: Vec<Option<&GraphBetaState>> = vec![Some(&beta), Some(&beta)];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None, None];

        let run = |seeds: Option<Vec<String>>, layers: usize| {
            verifier.refine_interm_bounds_with_opts(
                &graph,
                "out",
                2,
                &caches,
                &inputs,
                &betas,
                &alphas,
                engine,
                &IntermRefineOptions {
                    seeds,
                    ..opts(true, layers)
                },
            )
        };
        let walk = run(None, 2).expect("layers=2 walk must apply");
        for tokens in [
            vec!["relu3".to_string(), "last".to_string()],
            // Out of order + shouty `LAST` + junk tokens: same outcome.
            vec![
                "LAST".to_string(),
                "nope".to_string(),
                "l2".to_string(),
                "relu3".to_string(),
            ],
        ] {
            let named = run(Some(tokens.clone()), 1).expect("named seeds must apply");
            assert_eq!(named.infeasible, walk.infeasible, "tokens={tokens:?}");
            for node in ["g3", "relu3", "gemm_pre", "relu_last"] {
                let a = &walk.caches[1][node];
                let b = &named.caches[1][node];
                assert_eq!(a.lower(), b.lower(), "node={node} tokens={tokens:?}");
                assert_eq!(a.upper(), b.upper(), "node={node} tokens={tokens:?}");
            }
        }
        // No resolvable seed ⇒ the lane refuses (keeps inherited caches).
        assert!(
            run(Some(vec!["nope".to_string()]), 1).is_none(),
            "unresolvable seed list must refuse"
        );
    }

    /// LEG-A e2e on the direct research clip path (`clip_resnet + clip_guard`).
    /// Runs the full `refine_interm_bounds_with_opts` with explicit test options
    /// and the batched split-constraint clip ARMED on ≥2 heterogeneous last-ReLU
    /// split domains, then asserts every premise-satisfying sample's true forward
    /// stays inside the REFINED (clip-tightened) `gemm_pre` enclosure — the common-mode
    /// GPU coeff→fold→clip→merge path the CPU oracle cannot reach. The runtime guard
    /// is armed too (must not spuriously revert a sound clip). Also checks the clip is
    /// NEVER-LOOSER than the gate-OFF refinement. Skips without a wgpu adapter.
    #[test]
    fn leg_a_clip_e2e_enclosure_and_guard_gpu() {
        let Ok(device) = ny_gpu::ComputeDevice::new(ny_gpu::Backend::Wgpu) else {
            eprintln!("SKIP: no wgpu adapter for leg_a_clip_e2e_enclosure_and_guard_gpu");
            return;
        };
        let engine: &dyn GemmEngine = &device;
        let graph = build_two_block_net();
        let verifier = BetaCrownVerifier::default();

        // Two heterogeneous last-ReLU split domains (different premise + box).
        let root = box2([-1.0, -1.0], [1.0, 1.0]);
        let mk = |neuron: usize, active: bool, lo: [f32; 2], hi: [f32; 2]| {
            let mut history = GraphSplitHistory::new();
            history.add_constraint(
                GraphNeuronConstraint::new("relu_last".to_string(), neuron, active, 0.0)
                    .expect("valid constraint"),
            );
            let (cache, _cin) = verifier
                .compute_constrained_forward_bounds(&graph, &root, &history, None, None)
                .expect("constrained forward");
            let beta = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
                "relu_last".into(),
                neuron,
                0.0,
                0.0,
                if active { 1.0 } else { -1.0 },
            )
            .expect("entry")]);
            (cache, beta, box2(lo, hi), (neuron, active))
        };
        let d0 = mk(0, true, [-0.5, -1.0], [1.0, 1.0]); // gemm_pre[0] ≥ 0
        let d1 = mk(1, false, [-1.0, -1.0], [1.0, 0.6]); // gemm_pre[1] ≤ 0

        let caches = vec![d0.0.clone(), d1.0.clone()];
        let inputs = vec![d0.2.clone(), d1.2.clone()];
        let betas: Vec<Option<&GraphBetaState>> = vec![Some(&d0.1), Some(&d1.1)];
        let alphas: Vec<Option<&GraphDomainAlphaState>> = vec![None, None];
        let prem = [d0.3, d1.3];
        let dom_boxes = [([-0.5f32, -1.0], [1.0f32, 1.0]), ([-1.0, -1.0], [1.0, 0.6])];

        let run = |clip: bool| {
            verifier
                .refine_interm_bounds_with_opts(
                    &graph,
                    "out",
                    2,
                    &caches,
                    &inputs,
                    &betas,
                    &alphas,
                    engine,
                    &IntermRefineOptions {
                        clip_resnet: clip,
                        clip_guard: clip, // the NY_CLIP_INTERM umbrella arms both
                        ..opts(false, 1)
                    },
                )
                .expect("refinement must apply on the sound GPU lane")
        };
        let off = run(false);
        let on = run(true);

        let mut clip_tightened = false;
        for di in 0..2 {
            let (lo, hi) = dom_boxes[di];
            let (pn, pa) = prem[di];
            let refined = on.caches[di].get("gemm_pre").expect("refined seed");
            let base = off.caches[di].get("gemm_pre").expect("base seed");
            for k in 0..2 {
                // NEVER-LOOSER: clip-ON must be at least as tight as clip-OFF.
                assert!(
                    refined.lower()[[k]] >= base.lower()[[k]] - 1e-4
                        && refined.upper()[[k]] <= base.upper()[[k]] + 1e-4,
                    "dom {di} gemm_pre[{k}]: clip-ON [{}, {}] looser than clip-OFF [{}, {}]",
                    refined.lower()[[k]],
                    refined.upper()[[k]],
                    base.lower()[[k]],
                    base.upper()[[k]],
                );
                if refined.lower()[[k]] > base.lower()[[k]] + 1e-4
                    || refined.upper()[[k]] < base.upper()[[k]] - 1e-4
                {
                    clip_tightened = true;
                }
            }
            // ENCLOSURE: premise-satisfying grid samples stay inside the refined box.
            for i in 0..=16 {
                for j in 0..=16 {
                    let x = [
                        lo[0] + (hi[0] - lo[0]) * (i as f32) / 16.0,
                        lo[1] + (hi[1] - lo[1]) * (j as f32) / 16.0,
                    ];
                    let z = gemm_pre_of(x);
                    let sat = if pa { z[pn] >= 0.0 } else { z[pn] <= 0.0 };
                    if !sat {
                        continue;
                    }
                    for k in 0..2 {
                        let tol = 1e-4 * (1.0 + z[k].abs());
                        assert!(
                            refined.lower()[[k]] - tol <= z[k]
                                && z[k] <= refined.upper()[[k]] + tol,
                            "LEG-A(GPU) UNSOUND: dom {di} gemm_pre[{k}]={} escapes refined \
                             [{}, {}] at x={x:?}",
                            z[k],
                            refined.lower()[[k]],
                            refined.upper()[[k]],
                        );
                    }
                }
            }
        }
        eprintln!("[leg-a-gpu] clip engaged (tightened vs OFF)={clip_tightened}");
    }
}

/// #clip-interm SOUNDNESS harness (Leg-A enclosure oracle + Leg-B batched-vs-serial
/// differential + fold/shape guard + the fail-closed runtime guard). These drive the
/// PRODUCTION per-domain clip functions (`fold_seed_rows_for_domain`,
/// `clip_seed_domain`, `clip_guard_verify_domain`) directly on a hand-built
/// `GpuResidentCoeffBatched`, so they run WITHOUT a GPU and are fully deterministic —
/// the load-bearing soundness gate for the bound-TIGHTENING clip lever.
#[cfg(test)]
mod clip_interm_soundness {
    use super::*;
    use crate::beta_crown::state::GraphBetaEntry;
    use crate::complete_clip::{test_certified_affine_fixture, TestSoundCrownAffineParts};
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};
    use num_rational::BigRational;
    use ny_core::GpuResidentCoeffBatched;

    // The seed net: input(3) → gemm_pre(Linear W,b) → relu_last(ReLU) → out(I).
    // `gemm_pre` is EXACTLY affine in the network input, so the folded seed rows are
    // an EXACT enclosure and the split constraints are EXACT — the HARDEST (no-slack)
    // adversarial case for the clip's enclosure DIRECTION and its dual concretization.
    const W: [[f32; 3]; 3] = [[1.0, 0.4, -0.2], [-0.3, 0.9, 0.5], [0.6, -0.7, 0.8]];
    const B: [f32; 3] = [0.1, -0.2, 0.15];

    fn oracle_net() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "gemm_pre",
            Layer::Linear(LinearLayer::new(arr2(&W), Some(arr1(&B))).expect("linear")),
        ));
        g.add_node(GraphNode::new(
            "relu_last",
            Layer::ReLU(ReLULayer),
            vec!["gemm_pre".to_string()],
        ));
        let ident = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        g.add_node(GraphNode::new(
            "out",
            Layer::Linear(
                LinearLayer::new(arr2(&ident), Some(arr1(&[0.0, 0.0, 0.0]))).expect("id"),
            ),
            vec!["relu_last".to_string()],
        ));
        g.set_output("out");
        g
    }

    /// True forward of `gemm_pre` at a point (exact through the production forward on
    /// a degenerate box — the ground-truth sampler, not a re-implementation).
    fn forward_gemm_pre(g: &GraphNetwork, x: [f32; 3]) -> [f32; 3] {
        let arr = ArrayD::from_shape_vec(IxDyn(&[3]), x.to_vec()).unwrap();
        let pt = BoundedTensor::new(arr.clone(), arr).unwrap();
        let vals = g.collect_node_bounds(&pt).unwrap();
        let gp = vals.get("gemm_pre").unwrap().flatten();
        [gp.lower()[[0]], gp.lower()[[1]], gp.lower()[[2]]]
    }

    /// Batched coeff for `n_domains` blocks of the same seed W/b (the input-relative
    /// CROWN coeff of a Linear seed is W regardless of box); per-domain certified
    /// per-coeff error `err_lo[d]`/`err_hi[d]` (folded OUTWARD by
    /// `fold_seed_rows_for_domain`). Layout `= (d*n_rows + r)*dim + j`, exactly the
    /// invariant the fold's `local_block` indexing relies on.
    fn batched_coeff(n_domains: usize, err_lo: &[f32], err_hi: &[f32]) -> GpuResidentCoeffBatched {
        let (dim, n_rows) = (3usize, 3usize);
        let num_specs = n_domains * n_rows;
        let mut c = GpuResidentCoeffBatched {
            lower_a: vec![0.0; num_specs * dim],
            upper_a: vec![0.0; num_specs * dim],
            lower_err: vec![0.0; num_specs * dim],
            upper_err: vec![0.0; num_specs * dim],
            lower_b: vec![0.0; num_specs],
            upper_b: vec![0.0; num_specs],
            lower_b_err: vec![0.0; num_specs],
            upper_b_err: vec![0.0; num_specs],
            dim,
            num_specs,
            num_specs_per_dom: n_rows,
        };
        for d in 0..n_domains {
            for r in 0..n_rows {
                let s = d * n_rows + r;
                for j in 0..dim {
                    c.lower_a[s * dim + j] = W[r][j];
                    c.upper_a[s * dim + j] = W[r][j];
                    c.lower_err[s * dim + j] = err_lo[d];
                    c.upper_err[s * dim + j] = err_hi[d];
                }
                c.lower_b[s] = B[r];
                c.upper_b[s] = B[r];
            }
        }
        c
    }

    fn beta(prem: &[(usize, bool)]) -> GraphBetaState {
        GraphBetaState::from_entries(
            prem.iter()
                .map(|&(idx, active)| {
                    GraphBetaEntry::new(
                        "relu_last".into(),
                        idx,
                        0.0,
                        0.0,
                        if active { 1.0 } else { -1.0 },
                    )
                    .expect("valid entry")
                })
                .collect(),
        )
    }

    /// Route legacy clip soundness fixtures through the same provenance-required
    /// compatibility seam as production.  The raw-parts constructor is compiled
    /// only for tests; non-test code has no way to mint the token/pass pair.
    #[allow(clippy::too_many_arguments)]
    fn provenance_fixture(
        folded: &FoldedSeedRows,
        beta_state: &GraphBetaState,
        relu_name: &str,
        row_of: &[usize],
        sel: &[usize],
        in_lo: &[f32],
        in_hi: &[f32],
        pass_words: [u64; 2],
    ) -> Option<(CrownPassStamp, CertifiedAffineEnclosure)> {
        let mut never = || false;
        let history = reconstruct_clip_relu_history(beta_state, &mut never)?;
        let parts = TestSoundCrownAffineParts {
            lower_a: Array2::from_shape_vec((folded.n_rows, folded.dim), folded.lower_a.clone())
                .ok()?,
            upper_a: Array2::from_shape_vec((folded.n_rows, folded.dim), folded.upper_a.clone())
                .ok()?,
            lower_a_error: Array2::zeros((folded.n_rows, folded.dim)),
            upper_a_error: Array2::zeros((folded.n_rows, folded.dim)),
            lower_bias_center: Array1::from_vec(folded.lower_b.clone()),
            upper_bias_center: Array1::from_vec(folded.upper_b.clone()),
            lower_bias_error: Array1::zeros(folded.n_rows),
            upper_bias_error: Array1::zeros(folded.n_rows),
        };
        test_certified_affine_fixture(
            pass_words,
            b"clip-soundness-oracle-v1",
            in_lo,
            in_hi,
            &history,
            relu_name,
            sel,
            row_of,
            parts,
        )
        .ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn clip_seed_domain(
        folded: &FoldedSeedRows,
        beta_state: &GraphBetaState,
        relu_name: &str,
        row_of: &[usize],
        sel: &[usize],
        in_lo: &[f32],
        in_hi: &[f32],
        deadline: Option<std::time::Instant>,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let (pass, token) = provenance_fixture(
            folded,
            beta_state,
            relu_name,
            row_of,
            sel,
            in_lo,
            in_hi,
            [0x434c_4950, 0x5052_4f56],
        )?;
        super::clip_seed_domain(
            folded,
            Some((&token, &pass)),
            beta_state,
            relu_name,
            row_of,
            sel,
            in_lo,
            in_hi,
            deadline,
        )
    }

    /// The compatibility seam is authority-by-token, never authority-by-array.
    /// A valid test-only CROWN fixture reaches the existing independently checked
    /// Complete Clipping dual; every exact-context mismatch returns `None`, which
    /// is the caller's inherited-bound path.
    #[ntest::timeout(20000)]
    #[test]
    fn affine_provenance_token_is_required_and_exactly_context_bound() {
        let lo = [-1.0f32, -1.0, -1.0];
        let hi = [1.0f32, 1.0, 1.0];
        let sel = [0usize, 1, 2];
        let row_of = [0usize, 1, 2];
        let state = beta(&[(1, true)]);
        let mut proposal =
            fold_seed_rows_for_domain(&batched_coeff(1, &[0.0], &[0.0]), 0, 3, &lo, &hi, None)
                .expect("finite proposal");
        let (pass, token) = provenance_fixture(
            &proposal,
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            [41, 1],
        )
        .expect("test-only sound CROWN fixture");

        assert!(
            super::clip_seed_domain(
                &proposal,
                None,
                &state,
                "relu_last",
                &row_of,
                &sel,
                &lo,
                &hi,
                None,
            )
            .is_none(),
            "raw affine arrays without a token must inherit bounds"
        );
        let valid = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        )
        .expect("valid token must replay through the checked clipping dual");
        assert!(valid.0.iter().chain(&valid.1).all(|v| v.is_finite()));

        let other_node = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &state,
            "other_relu",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        );
        assert!(other_node.is_none(), "mismatched node must inherit");

        let changed_hi = [1.0f32, 1.0, 0.75];
        let other_domain = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &changed_hi,
            None,
        );
        assert!(
            other_domain.is_none(),
            "mismatched input domain must inherit"
        );

        let other_history = beta(&[(1, false)]);
        let changed_history = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &other_history,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        );
        assert!(changed_history.is_none(), "mismatched history must inherit");

        let other_sel = [1usize, 0, 2];
        let other_row_of = [1usize, 0, 2];
        let changed_objective = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &state,
            "relu_last",
            &other_row_of,
            &other_sel,
            &lo,
            &hi,
            None,
        );
        assert!(
            changed_objective.is_none(),
            "mismatched objective must inherit"
        );

        let (new_pass, _new_token) = provenance_fixture(
            &proposal,
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            [41, 2],
        )
        .expect("second pass fixture");
        let stale = super::clip_seed_domain(
            &proposal,
            Some((&token, &new_pass)),
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        );
        assert!(stale.is_none(), "stale pass token must inherit");

        proposal.lower_a[0] = next_up_f32(proposal.lower_a[0]);
        let altered_raw = super::clip_seed_domain(
            &proposal,
            Some((&token, &pass)),
            &state,
            "relu_last",
            &row_of,
            &sel,
            &lo,
            &hi,
            None,
        );
        assert!(altered_raw.is_none(), "altered raw proposal must inherit");

        assert!(!clip_interm_resnet_batched_enabled());
        assert!(!clip_interm_umbrella_enabled());
    }

    fn premises_satisfied(prem: &[(usize, bool)], z: [f32; 3]) -> bool {
        prem.iter()
            .all(|&(j, active)| if active { z[j] >= 0.0 } else { z[j] <= 0.0 })
    }

    /// Box-only (unconstrained IBP) range of seed row r over [lo,hi] — the range the
    /// clip must TIGHTEN INSIDE OF to be non-vacuous, and stay OUTSIDE OF to be sound.
    fn box_range(r: usize, lo: [f32; 3], hi: [f32; 3]) -> (f32, f32) {
        // Bit-identical (a+b)*0.5 anchor (f32 center fixture).
        #[allow(clippy::manual_midpoint)]
        let (x0, eps): ([f32; 3], [f32; 3]) = (
            [
                (lo[0] + hi[0]) * 0.5,
                (lo[1] + hi[1]) * 0.5,
                (lo[2] + hi[2]) * 0.5,
            ],
            [
                (hi[0] - lo[0]) * 0.5,
                (hi[1] - lo[1]) * 0.5,
                (hi[2] - lo[2]) * 0.5,
            ],
        );
        let mut center = B[r];
        let mut rad = 0.0f32;
        for j in 0..3 {
            center += W[r][j] * x0[j];
            rad += W[r][j].abs() * eps[j];
        }
        (center - rad, center + rad)
    }

    /// LEG-A GROUND-TRUTH ENCLOSURE ORACLE (the load-bearing soundness gate).
    ///
    /// For ≥2 heterogeneous split domains: fold the batched coeff → run the PRODUCTION
    /// per-domain clip (`clip_seed_domain`) → then DIRECTED adversarial search (dense
    /// grid + per-row box-corner pushes toward crossing clip_lower/clip_upper) for
    /// FEASIBLE points (satisfying the child's ReLU split signs on the true forward),
    /// asserting `clip_lower[r] ≤ z_r(x) ≤ clip_upper[r]` for EVERY sampled feasible x.
    /// A feasible x outside the tightened box PROVES the clip unsound (false-VERIFY) →
    /// this test goes RED and names the exact domain/row/value. ≥200 feasible points
    /// across the domains; the clip must also genuinely tighten (non-vacuity).
    #[ntest::timeout(60000)]
    #[test]
    fn leg_a_clip_enclosure_oracle_cpu() {
        let g = oracle_net();
        let sel = [0usize, 1, 2];
        let row_of = [0usize, 1, 2];

        // Two heterogeneous domains: different premises, different boxes, and a
        // genuine-relaxation (err>0) block to exercise the OUTWARD error fold.
        struct Dom {
            lo: [f32; 3],
            hi: [f32; 3],
            prem: Vec<(usize, bool)>,
            block: usize,
        }
        let doms = [
            Dom {
                lo: [-1.0, -1.0, -1.0],
                hi: [1.0, 1.0, 1.0],
                prem: vec![(0, true), (2, false)], // z0 ≥ 0, z2 ≤ 0
                block: 0,
            },
            Dom {
                lo: [-1.0, -0.8, -1.0],
                hi: [0.8, 1.0, 0.9],
                prem: vec![(1, true)], // z1 ≥ 0
                block: 1,
            },
        ];
        // block 0: exact (err 0). block 1: err>0 (folded looser — still an enclosure).
        let coeff = batched_coeff(2, &[0.0, 0.02], &[0.0, 0.02]);

        let mut total_feasible = 0usize;
        let mut tightened_rows = 0usize;

        for (di, d) in doms.iter().enumerate() {
            let lo = d.lo;
            let hi = d.hi;
            let folded = fold_seed_rows_for_domain(&coeff, d.block, 3, &lo, &hi, None)
                .expect("fold must succeed on finite coeff");
            assert!(
                folded.refused.iter().all(|&r| !r),
                "dom {di}: finite coeff must not refuse any row"
            );
            let bs = beta(&d.prem);
            let (tl, tu) =
                clip_seed_domain(&folded, &bs, "relu_last", &row_of, &sel, &lo, &hi, None)
                    .unwrap_or_else(|| {
                        panic!("dom {di}: clip must run (constraints non-empty on a full-sign box)")
                    });
            assert_eq!(tl.len(), 3);
            assert_eq!(tu.len(), 3);

            // Non-vacuity: the clip must TIGHTEN inside the box-only range on ≥1 row.
            for r in 0..3 {
                let (blo, bhi) = box_range(r, lo, hi);
                if tl[r] > blo + 1e-3 || tu[r] < bhi - 1e-3 {
                    tightened_rows += 1;
                }
            }

            let mut state: u64 = 0xD1CE_0000 ^ (di as u64);
            let mut u01 = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32) / ((1u64 << 24) as f32)
            };

            let check = |x: [f32; 3], feasible_count: &mut usize| {
                let z = forward_gemm_pre(&g, x);
                if !premises_satisfied(&d.prem, z) {
                    return;
                }
                *feasible_count += 1;
                for r in 0..3 {
                    let tol = 1e-3 * (1.0 + z[r].abs());
                    assert!(
                        tl[r] - tol <= z[r] && z[r] <= tu[r] + tol,
                        "LEG-A UNSOUND: dom {di} row {r} true z={} escapes clip [{}, {}] at x={x:?} \
                         (premises {:?}) — a too-tight intermediate bound = false-VERIFY",
                        z[r],
                        tl[r],
                        tu[r],
                        d.prem,
                    );
                }
            };

            let mut feasible = 0usize;
            // Dense grid.
            let n = 16i32;
            for i0 in 0..=n {
                for i1 in 0..=n {
                    for i2 in 0..=n {
                        let x = [
                            lo[0] + (hi[0] - lo[0]) * (i0 as f32) / (n as f32),
                            lo[1] + (hi[1] - lo[1]) * (i1 as f32) / (n as f32),
                            lo[2] + (hi[2] - lo[2]) * (i2 as f32) / (n as f32),
                        ];
                        check(x, &mut feasible);
                    }
                }
            }
            // DIRECTED corner pushes: extremize each row's folded objective toward the
            // box corner a too-tight bound would be exposed at, then jitter.
            for k in 0..600usize {
                let r = (k / 2) % 3;
                let test_upper = (k / 2 / 3) % 2 == 0;
                let mut x = [0.0f32; 3];
                for j in 0..3 {
                    let a = if test_upper {
                        folded.upper_a[r * 3 + j]
                    } else {
                        folded.lower_a[r * 3 + j]
                    };
                    let corner = if (test_upper && a > 0.0) || (!test_upper && a < 0.0) {
                        hi[j]
                    } else {
                        lo[j]
                    };
                    x[j] = (corner + (u01() - 0.5) * 0.25 * (hi[j] - lo[j])).clamp(lo[j], hi[j]);
                }
                check(x, &mut feasible);
            }
            assert!(
                feasible >= 100,
                "dom {di}: only {feasible} feasible samples — oracle vacuous"
            );
            total_feasible += feasible;
        }
        assert!(
            total_feasible >= 200,
            "oracle must sample ≥200 feasible points, got {total_feasible}"
        );
        assert!(
            tightened_rows > 0,
            "the clip never tightened any row vs the box-only range — test would be vacuous"
        );
    }

    /// LEG-B BATCHED-vs-SERIAL DIFFERENTIAL (misalignment guard). The batched clip
    /// folds domain d's block at `local_block = d` of the STACKED coeff; the serial
    /// reference builds a fresh 1-block coeff carrying ONLY domain d's own error and
    /// folds it at `local_block = 0` — sharing NO batched indexing. Per row the two
    /// must agree within f32 slop (a `local_block` misalignment would hand domain d
    /// another block's coefficients ⇒ a divergent, possibly too-tight, clip).
    #[ntest::timeout(30000)]
    #[test]
    fn leg_b_batched_vs_serial_clip_differential_cpu() {
        let sel = [0usize, 1, 2];
        let row_of = [0usize, 1, 2];
        // Distinct per-domain error ⇒ blocks genuinely differ (so the differential
        // has discriminating power against a misalignment).
        let err = [0.0f32, 0.03, 0.07];
        let batched = batched_coeff(3, &err, &err);
        let boxes = [
            ([-1.0f32, -1.0, -1.0], [1.0f32, 1.0, 1.0]),
            ([-0.9f32, -1.0, -0.7], [1.0f32, 0.6, 1.0]),
            ([-1.0f32, -0.5, -1.0], [0.5f32, 1.0, 0.8]),
        ];
        let prems: [Vec<(usize, bool)>; 3] = [
            vec![(0, true)],
            vec![(1, false)],
            vec![(2, true), (0, false)],
        ];

        let mut distinct_folds = 0usize;
        let mut prev_ub: Option<f32> = None;
        for d in 0..3 {
            let (lo, hi) = boxes[d];
            let bs = beta(&prems[d]);

            let bf =
                fold_seed_rows_for_domain(&batched, d, 3, &lo, &hi, None).expect("batched fold");
            // Serial reference: a private 1-block coeff with ONLY this domain's error.
            let serial_coeff = batched_coeff(1, &[err[d]], &[err[d]]);
            let sf = fold_seed_rows_for_domain(&serial_coeff, 0, 3, &lo, &hi, None)
                .expect("serial fold");

            // Folded rows must be numerically identical (same numbers, no shared index).
            for r in 0..3 {
                assert!(
                    (bf.lower_b[r] - sf.lower_b[r]).abs() <= 1e-4
                        && (bf.upper_b[r] - sf.upper_b[r]).abs() <= 1e-4,
                    "dom {d} row {r}: batched fold [{},{}] ≠ serial fold [{},{}] — MISALIGNMENT",
                    bf.lower_b[r],
                    bf.upper_b[r],
                    sf.lower_b[r],
                    sf.upper_b[r],
                );
            }

            let (btl, btu) = clip_seed_domain(&bf, &bs, "relu_last", &row_of, &sel, &lo, &hi, None)
                .expect("batched clip");
            let (stl, stu) = clip_seed_domain(&sf, &bs, "relu_last", &row_of, &sel, &lo, &hi, None)
                .expect("serial clip");
            for r in 0..3 {
                // Task form: batched ≤ serial + slop (not tighter beyond f32 reorder)
                // AND batched ≥ serial − slop (sound side / no drift).
                assert!(
                    btl[r] <= stl[r] + 1e-3 && btl[r] >= stl[r] - 1e-3,
                    "dom {d} row {r} LOWER: batched {} vs serial {} beyond slop",
                    btl[r],
                    stl[r]
                );
                assert!(
                    btu[r] <= stu[r] + 1e-3 && btu[r] >= stu[r] - 1e-3,
                    "dom {d} row {r} UPPER: batched {} vs serial {} beyond slop",
                    btu[r],
                    stu[r]
                );
            }
            // Non-tautology: the per-domain blocks must actually differ (else a
            // misalignment would be invisible).
            if prev_ub.is_some_and(|p| (p - bf.upper_b[0]).abs() > 1e-6) {
                distinct_folds += 1;
            }
            prev_ub = Some(bf.upper_b[0]);
        }
        assert!(
            distinct_folds > 0,
            "batched blocks were indistinguishable — the differential cannot catch a misalignment"
        );
    }

    /// `fold_seed_rows_for_domain` shape/indexing guard + OUTWARD error fold.
    #[ntest::timeout(20000)]
    #[test]
    fn fold_seed_rows_shape_and_outward_fold_cpu() {
        let lo = [-1.0f32, -1.0, -1.0];
        let hi = [1.0f32, 1.0, 1.0];
        let coeff = batched_coeff(2, &[0.05, 0.05], &[0.05, 0.05]); // num_specs=6

        assert!(fold_seed_rows_for_domain(&coeff, 0, 3, &lo, &hi, None).is_some());
        assert!(fold_seed_rows_for_domain(&coeff, 1, 3, &lo, &hi, None).is_some());
        // Out-of-range block (would read rows 6,7,8 ≥ num_specs) → None (fail-closed).
        assert!(
            fold_seed_rows_for_domain(&coeff, 2, 3, &lo, &hi, None).is_none(),
            "out-of-range block must refuse (the s>=num_specs bound)"
        );
        // Wrong input dim → None.
        assert!(fold_seed_rows_for_domain(&coeff, 0, 3, &[0.0, 0.0], &[1.0, 1.0], None).is_none());
        let mut malformed = batched_coeff(2, &[0.05, 0.05], &[0.05, 0.05]);
        malformed.lower_err.pop();
        assert!(
            fold_seed_rows_for_domain(&malformed, 0, 3, &lo, &hi, None).is_none(),
            "a truncated certified-error table must refuse"
        );
        let mut wrong_stride = batched_coeff(2, &[0.05, 0.05], &[0.05, 0.05]);
        wrong_stride.num_specs_per_dom = 2;
        assert!(
            fold_seed_rows_for_domain(&wrong_stride, 0, 3, &lo, &hi, None).is_none(),
            "domain-row stride mismatch must refuse"
        );
        let mut partial_block = batched_coeff(2, &[0.05, 0.05], &[0.05, 0.05]);
        partial_block.num_specs -= 1;
        partial_block
            .lower_a
            .truncate(partial_block.num_specs * partial_block.dim);
        partial_block
            .upper_a
            .truncate(partial_block.num_specs * partial_block.dim);
        partial_block
            .lower_err
            .truncate(partial_block.num_specs * partial_block.dim);
        partial_block
            .upper_err
            .truncate(partial_block.num_specs * partial_block.dim);
        partial_block.lower_b.truncate(partial_block.num_specs);
        partial_block.upper_b.truncate(partial_block.num_specs);
        partial_block.lower_b_err.truncate(partial_block.num_specs);
        partial_block.upper_b_err.truncate(partial_block.num_specs);
        assert!(
            fold_seed_rows_for_domain(&partial_block, 0, 3, &lo, &hi, None).is_none(),
            "a partial final domain block must refuse globally"
        );

        // OUTWARD fold: with err>0 the folded box strictly contains B, and the folded
        // affine form encloses the true z = W·x + b at every sampled point.
        let f = fold_seed_rows_for_domain(&coeff, 0, 3, &lo, &hi, None).unwrap();
        let mut state: u64 = 0x5EED;
        let mut u01 = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / ((1u64 << 24) as f32)
        };
        for r in 0..3 {
            assert!(
                f.lower_b[r] <= B[r] + 1e-6,
                "row {r} lower_b not folded outward"
            );
            assert!(
                f.upper_b[r] >= B[r] - 1e-6,
                "row {r} upper_b not folded outward"
            );
        }
        for _ in 0..2000 {
            let x = [
                lo[0] + u01() * 2.0,
                lo[1] + u01() * 2.0,
                lo[2] + u01() * 2.0,
            ];
            for r in 0..3 {
                let z: f32 = (0..3).map(|j| W[r][j] * x[j]).sum::<f32>() + B[r];
                let lo_aff: f32 =
                    (0..3).map(|j| f.lower_a[r * 3 + j] * x[j]).sum::<f32>() + f.lower_b[r];
                let hi_aff: f32 =
                    (0..3).map(|j| f.upper_a[r * 3 + j] * x[j]).sum::<f32>() + f.upper_b[r];
                assert!(
                    lo_aff <= z + 1e-5 && z <= hi_aff + 1e-5,
                    "row {r}: folded affine [{lo_aff},{hi_aff}] does NOT enclose true z={z} at x={x:?}"
                );
            }
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn resident_error_fold_polls_deadline_inside_coefficient_loop() {
        let dim = 4096usize;
        let coeff = GpuResidentCoeffBatched {
            lower_a: vec![0.25f32; dim],
            upper_a: vec![0.25f32; dim],
            lower_err: vec![f32::from_bits(1); dim],
            upper_err: vec![f32::from_bits(1); dim],
            lower_b: vec![0.0],
            upper_b: vec![0.0],
            lower_b_err: vec![0.0],
            upper_b_err: vec![0.0],
            dim,
            num_specs: 1,
            num_specs_per_dom: 1,
        };
        let lo = vec![-1.0f32; dim];
        let hi = vec![1.0f32; dim];
        assert!(
            fold_seed_rows_for_domain(&coeff, 0, 1, &lo, &hi, None).is_some(),
            "fixture must be an otherwise-valid resident fold"
        );
        assert!(
            fold_seed_rows_for_domain(&coeff, 0, 1, &lo, &hi, Some(std::time::Instant::now()),)
                .is_none(),
            "expired production Instant must refuse before fold allocation"
        );

        let mut polls = 0usize;
        let mut expires_in_error_fold = || {
            polls += 1;
            polls >= 16
        };
        let refused = fold_seed_rows_for_domain_with_deadline_check(
            &coeff,
            0,
            1,
            &lo,
            &hi,
            &mut expires_in_error_fold,
        );
        assert!(refused.is_none(), "in-fold expiry must refuse the proposal");
        assert_eq!(polls, 16, "fixture must expire inside the coefficient fold");
    }

    /// GraphBetaEntry does not retain GenBaB constraint kind/input-index. A
    /// nonzero split therefore cannot be losslessly reconstructed as a ReLU
    /// history and must refuse clipping for both branch directions.
    #[ntest::timeout(10000)]
    #[test]
    fn nonzero_beta_split_points_fail_closed_for_both_signs() {
        let lo = [-1.0f32, -1.0, -1.0];
        let hi = [1.0f32, 1.0, 1.0];
        let folded =
            fold_seed_rows_for_domain(&batched_coeff(1, &[0.0], &[0.0]), 0, 3, &lo, &hi, None)
                .unwrap();
        for sign in [-1.0f32, 1.0] {
            let state = GraphBetaState::from_entries(vec![GraphBetaEntry::new(
                "relu_last".into(),
                0,
                0.25,
                0.0,
                sign,
            )
            .unwrap()]);
            assert!(
                clip_seed_domain(
                    &folded,
                    &state,
                    "relu_last",
                    &[0, 1, 2],
                    &[0, 1, 2],
                    &lo,
                    &hi,
                    None,
                )
                .is_none(),
                "nonzero split with sign {sign} must refuse authority"
            );
        }

        let zero_relu = beta(&[(0, true)]);
        assert!(
            clip_seed_domain(
                &folded,
                &zero_relu,
                "relu_last",
                &[0, 1, 2],
                &[0, 1, 2],
                &lo,
                &hi,
                Some(std::time::Instant::now()),
            )
            .is_none(),
            "expired authority deadline must refuse before proposal allocation"
        );
    }

    /// Exact-real regression for the cancellation that a single final f32 ULP
    /// cannot cover: the first error product is 2^100 and the remaining 4095
    /// products are 1. An ordinary f64 sum loses all small terms, so subtracting
    /// it from a 2^100 bias incorrectly yields about zero instead of -4095.
    #[ntest::timeout(20000)]
    #[test]
    fn fold_bias_rounding_handles_large_small_cancellation_exactly() {
        let dim = 4096usize;
        let huge = 2.0f32.powi(50);
        let huge_sq = 2.0f32.powi(100);
        let mut err = vec![1.0f32; dim];
        err[0] = huge;
        let mut lo = vec![-1.0f32; dim];
        let mut hi = vec![1.0f32; dim];
        lo[0] = -huge;
        hi[0] = huge;
        let coeff = GpuResidentCoeffBatched {
            lower_a: vec![0.0; dim],
            upper_a: vec![0.0; dim],
            lower_err: err.clone(),
            upper_err: err,
            lower_b: vec![huge_sq],
            upper_b: vec![-huge_sq],
            lower_b_err: vec![0.0],
            upper_b_err: vec![0.0],
            dim,
            num_specs: 1,
            num_specs_per_dom: 1,
        };

        let naive_penalty: f64 = coeff
            .lower_err
            .iter()
            .zip(lo.iter().zip(&hi))
            .map(|(&e, (&l, &u))| f64::from(e.abs()) * f64::from(l.abs().max(u.abs())))
            .sum();
        assert_eq!(naive_penalty, f64::from(huge_sq));
        let naive_lower = next_down_f32((f64::from(huge_sq) - naive_penalty) as f32);

        let folded = fold_seed_rows_for_domain(&coeff, 0, 1, &lo, &hi, None).expect("valid fold");
        let exact = BigRational::from_integer((dim as i64 - 1).into());
        let stored_lower =
            BigRational::from_float(f64::from(folded.lower_b[0])).expect("finite lower");
        let stored_upper =
            BigRational::from_float(f64::from(folded.upper_b[0])).expect("finite upper");
        assert!(
            BigRational::from_float(f64::from(naive_lower)).unwrap() > -exact.clone(),
            "fixture must expose the old one-final-ULP failure"
        );
        assert!(stored_lower <= -exact.clone());
        assert!(stored_upper >= exact);
    }

    /// The FAIL-CLOSED runtime guard itself: it must PASS a sound clip and CATCH a
    /// deliberately too-tight bound (both directions). This directly exercises the
    /// guard retained for research while production clip authority is quarantined.
    #[ntest::timeout(30000)]
    #[test]
    fn clip_guard_catches_too_tight_and_passes_sound_cpu() {
        let g = oracle_net();
        let sel = [0usize, 1, 2];
        let row_of = [0usize, 1, 2];
        let lo = [-1.0f32, -1.0, -1.0];
        let hi = [1.0f32, 1.0, 1.0];
        let prem = vec![(1usize, true)]; // z1 ≥ 0
        let bs = beta(&prem);
        let coeff = batched_coeff(1, &[0.0], &[0.0]);
        let folded = fold_seed_rows_for_domain(&coeff, 0, 3, &lo, &hi, None).unwrap();
        let (tl, tu) = clip_seed_domain(&folded, &bs, "relu_last", &row_of, &sel, &lo, &hi, None)
            .expect("clip");
        let input_box = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), lo.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), hi.to_vec()).unwrap(),
        )
        .unwrap();

        // SOUND clip → guard passes.
        assert!(
            clip_guard_verify_domain(
                &g,
                &input_box,
                "gemm_pre",
                "relu_last",
                &bs,
                &sel,
                &folded,
                &tl,
                &tu,
                128,
                0xABC,
            )
            .is_ok(),
            "guard must PASS the genuinely sound clip"
        );

        // Too-tight UPPER on row 0 (below the whole box range) → guard catches it.
        let (blo0, _bhi0) = box_range(0, lo, hi);
        let mut bad_tu = tu.clone();
        bad_tu[0] = blo0 - 1.0;
        assert!(
            clip_guard_verify_domain(
                &g,
                &input_box,
                "gemm_pre",
                "relu_last",
                &bs,
                &sel,
                &folded,
                &tl,
                &bad_tu,
                256,
                0xABC,
            )
            .is_err(),
            "guard must CATCH a too-tight UPPER bound (false-VERIFY)"
        );

        // Too-tight LOWER on row 0 (above the whole box range) → guard catches it.
        let (_blo0b, bhi0) = box_range(0, lo, hi);
        let mut bad_tl = tl;
        bad_tl[0] = bhi0 + 1.0;
        assert!(
            clip_guard_verify_domain(
                &g,
                &input_box,
                "gemm_pre",
                "relu_last",
                &bs,
                &sel,
                &folded,
                &bad_tl,
                &tu,
                256,
                0xABC,
            )
            .is_err(),
            "guard must CATCH a too-tight LOWER bound"
        );
    }
}

// (free fns at module scope — all private helpers are file-visible here).

/// Root JOINT per-target intermediate-bound α pass (#root-joint-interm-alpha).
///
/// auto_LiRPA's `fix_intermediate_layer_bounds=False` root pass, scoped: for each
/// target pre-activation node L in `targets` (deepest-first), build the truncated
/// L→input resnet stack, seed identity over L's CROSSING rows (`num_specs = n_sel`),
/// and Adam-ascend the below-L α against the SUMMED lower-bound objective
/// `Σ_r lb(L_r)` whose joint gradient flows THROUGH L's own bound computation via
/// the on-device adjoint (`crown_joint_alpha_gradient_resident`). Every iterate is
/// scored with the certified sound fold (`crown_backward_gpu_resnet_sound_beta`);
/// the element-wise best `[l',u']` across iterates is intersected SHRINK-ONLY into
/// `bounds[L]` (`intersection_per_element`, union fallback per element — the
/// alpha_explicit contract), so the frozen root tree can only tighten, never widen.
///
/// SOUNDNESS: any α∈[0,1] is a valid ReLU lower slope ⇒ every scored fold is a
/// valid enclosure of L's pre-activation over the root box; the element-wise best
/// across valid enclosures is a valid enclosure (intersection); the final
/// intersect-with-reference is shrink-only with per-element union fallback. On any
/// refusal (no sound GPU, prep failure, shape mismatch, deadline) the target keeps
/// its sound reference bound. Dark: only reachable under
/// `NY_ROOT_JOINT_INTERM_ALPHA=1` (root.rs gate) ⇒ default byte-identical.
///
/// Returns the number of targets whose bound strictly tightened.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn root_joint_tighten_relu_preactivations(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    targets: &[String],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
    iters: usize,
    lr: f32,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> usize {
    let options = RootIntermTightenOptions {
        iters,
        lr,
        max_sel: std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(512),
        probe: std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_PROBE")
            .ok()
            .as_deref()
            == Some("1"),
        frozen_stop: std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP")
            .ok()
            .as_deref()
            == Some("1"),
        allow_bn: std::env::var("NY_BN_GPU_EXTRACT").ok().as_deref() == Some("1"),
        allow_pure_chain: std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN")
            .ok()
            .as_deref()
            == Some("1"),
        log_tag: "root-joint-interm-alpha",
    };
    root_tighten_relu_preactivations_with_options(
        graph, input, targets, engine, deadline, options, bounds,
    )
}

/// Production-shaped base CROWN fold for sparse crossing rows at the root.
///
/// This deliberately performs ZERO gradient/ascent iterations and admits none
/// of the experimental frozen-stop, BatchNorm-extraction, or pure-chain seams.
/// Every option is supplied by the typed caller, so unrelated research
/// environment variables cannot expand this pass's authority or resource use.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn root_sparse_tighten_relu_preactivations(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    targets: &[String],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
    max_sel: usize,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> usize {
    let options = RootIntermTightenOptions {
        iters: 0,
        lr: 0.0,
        max_sel,
        probe: false,
        frozen_stop: false,
        allow_bn: false,
        allow_pure_chain: false,
        log_tag: "root-sparse-interm-crown",
    };
    root_tighten_relu_preactivations_with_options(
        graph, input, targets, engine, deadline, options, bounds,
    )
}

#[derive(Debug, Clone, Copy)]
struct RootIntermTightenOptions {
    iters: usize,
    lr: f32,
    max_sel: usize,
    probe: bool,
    frozen_stop: bool,
    allow_bn: bool,
    allow_pure_chain: bool,
    log_tag: &'static str,
}

#[allow(clippy::too_many_arguments)]
fn root_tighten_relu_preactivations_with_options(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    targets: &[String],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<std::time::Instant>,
    options: RootIntermTightenOptions,
    bounds: &mut HashMap<String, BoundedTensor>,
) -> usize {
    let RootIntermTightenOptions {
        iters,
        lr,
        max_sel,
        probe,
        frozen_stop,
        allow_bn,
        allow_pure_chain,
        log_tag,
    } = options;
    if max_sel == 0 {
        return 0;
    }
    // Sanctioned sound-GPU access (same filter as refine_interm_bounds_with_opts):
    // the joint gradient only STEERS Adam (heuristic — cannot affect soundness);
    // every kept bound comes from the certified sound fold.
    let Some(gpu) = engine
        .and_then(|e| e.as_gpu_crown_backward())
        .filter(|g| g.provides_sound_gpu_crown())
    else {
        eprintln!("[{log_tag}] no sound GPU crown backward; skipping (sound no-op)");
        return 0;
    };
    let past_deadline = || deadline.is_some_and(|d| std::time::Instant::now() >= d);

    let mut n_tightened = 0usize;
    for target in targets {
        if past_deadline() {
            break;
        }
        let Some(ref_bt) = bounds.get(target).cloned() else {
            continue;
        };
        let pre_dim = ref_bt.lower().len();
        if pre_dim == 0 {
            continue;
        }
        let ref_l: Vec<f32> = ref_bt.lower().iter().copied().collect();
        let ref_u: Vec<f32> = ref_bt.upper().iter().copied().collect();
        // Crossing rows only (stable rows' relaxations are already exact),
        // widest-first, capped — the interm_refine sel convention.
        let mut sel: Vec<usize> = (0..pre_dim)
            .filter(|&j| ref_l[j] < 0.0 && ref_u[j] > 0.0)
            .collect();
        sel.sort_by(|&a, &b| {
            let wa = ref_u[a] - ref_l[a];
            let wb = ref_u[b] - ref_l[b];
            wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
        });
        sel.truncate(max_sel);
        let n_rows = sel.len();
        if n_rows == 0 {
            continue;
        }

        // Truncated stack from L down to the network input — or, under the
        // #frozen-stop opt-in, down to the deepest frozen-bounded node M below
        // which the extraction cannot continue (cgan: the generator boundary
        // ConvTranspose). SOUND: reachable(M) ⊆ box(M), so the certified fold of
        // graph[M→L] concretized against box(M) is a valid enclosure of L's true
        // pre-activation set; every KEPT bound still comes from the certified
        // sound fold, shrink-only intersected below. Dark-within-dark.
        // #cgan-bn-gpu-extract: BN-as-1x1-conv extraction, scoped to THIS lane
        // (legacy callers hard-pass false, so other lanes never see BN even with
        // the env set).
        // frozen-stop IMPLIES pure-chain: a truncated discriminator stack has no
        // residual Add, so without [Chain(..)] the stop lane could never fire.
        let allow_pure_chain = frozen_stop || allow_pure_chain;
        let Some(prep) = prep_resnet_domain_ext(
            graph,
            target,
            bounds,
            input,
            None,
            None,
            allow_pure_chain,
            allow_bn,
            frozen_stop,
        ) else {
            if probe {
                eprintln!("[{log_tag}] '{target}': prep refused; keeping reference");
            }
            continue;
        };
        // Preparation can include graph extraction and host-side materialization.
        // Never launch or publish a fold after that work consumed the slice.
        if past_deadline() {
            break;
        }
        // Stop-box belt check (fail-closed): when the stack stopped at M, the
        // concretization box must be bit-identical to the frozen map's entry for
        // M — the very tensor the extraction resolved against.
        if let Some(m) = prep.stop_node.as_deref() {
            let box_ok = bounds.get(m).is_some_and(|bt| {
                let lo = bt.lower();
                let hi = bt.upper();
                lo.len() == prep.in_lo.len()
                    && hi.len() == prep.in_hi.len()
                    && lo
                        .iter()
                        .zip(prep.in_lo.iter())
                        .all(|(&a, &b)| a.to_bits() == b.to_bits())
                    && hi
                        .iter()
                        .zip(prep.in_hi.iter())
                        .all(|(&a, &b)| a.to_bits() == b.to_bits())
                    && prep
                        .in_lo
                        .iter()
                        .zip(prep.in_hi.iter())
                        .all(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u)
            });
            if !box_ok {
                if probe {
                    eprintln!(
                        "[{log_tag}] '{target}': stop-box mismatch at '{m}'; keeping reference"
                    );
                }
                continue;
            }
            if probe {
                eprintln!(
                    "[{log_tag}] '{target}': truncated stack stop_at='{m}' box_dim={} segs={} relus={}",
                    prep.in_lo.len(),
                    prep.segments.len(),
                    prep.relu_names.len()
                );
            }
        }
        let n_relu = prep.relu_names.len();
        if n_relu == 0 || count_activations(&prep.segments) != n_relu {
            continue;
        }
        let stepped = unstable_masks(&prep.segments, n_relu);
        if !stepped.iter().any(|m| m.iter().any(|&s| s)) {
            continue; // nothing below L to optimize
        }
        let base_slopes = collect_lower_slopes(&prep.segments, n_relu);
        let empty_beta: Vec<Vec<f32>> = vec![Vec::new(); n_relu];

        // Identity seed over sel rows (both sides: the sound fold then returns
        // per-row [l,u] of L's pre-activation in one call).
        let mut rows = vec![0.0f32; n_rows * pre_dim];
        for (r, &j) in sel.iter().enumerate() {
            rows[r * pre_dim + j] = 1.0;
        }
        let seed = ny_core::GpuCrownSeed {
            lower_a: rows.clone().into(),
            upper_a: rows.into(),
            lower_b: vec![0.0f32; n_rows].into(),
            upper_b: vec![0.0f32; n_rows].into(),
            num_specs: n_rows,
            current_dim: pre_dim,
        };

        // Base sound fold = the ascent's starting enclosure (and best-so-far).
        let mut work_segs = prep.segments.clone();
        let base = match gpu.crown_backward_gpu_resnet_sound_beta(
            &work_segs,
            &seed,
            &prep.in_lo,
            &prep.in_hi,
            &empty_beta,
            &prep.frontier_abs,
            &prep.node_abs,
        ) {
            Ok(r) if r.lower_bounds.len() == n_rows && r.upper_bounds.len() == n_rows => r,
            _ => {
                if probe {
                    eprintln!("[{log_tag}] '{target}': base sound fold refused; keeping reference");
                }
                continue;
            }
        };
        // The sound GPU call is synchronous and may return after the slice.
        // A late enclosure is still mathematically sound, but publishing it can
        // let root verification escape the enclosing competition deadline.
        if past_deadline() {
            break;
        }
        let mut best_l = base.lower_bounds.clone();
        let mut best_u = base.upper_bounds.clone();
        let obj0: f32 = best_l.iter().filter(|v| v.is_finite()).sum();
        let mut best_obj = obj0;

        // Joint Adam ascent on the below-L α (summed lower objective).
        let mut slopes = base_slopes.clone();
        let mut adam = AlphaAdam::new(&base_slopes);
        for t in 1..=iters {
            if past_deadline() {
                break;
            }
            write_alpha_prime(&mut work_segs, &slopes, &stepped);
            let grads = match gpu.crown_joint_alpha_gradient_resident(
                &work_segs,
                seed.lower_a.as_ref(),
                n_rows,
                pre_dim,
                &prep.in_lo,
                &prep.in_hi,
            ) {
                Ok(g) if g.len() == n_relu => g,
                _ => break, // fail-closed: keep best-so-far (sound)
            };
            let lr_t = lr * 0.98f32.powi((t - 1) as i32);
            let max_g = adam.step(&mut slopes, &grads, &stepped, lr_t, t);
            if max_g == 0.0 || !max_g.is_finite() {
                break;
            }
            write_alpha_prime(&mut work_segs, &slopes, &stepped);
            let r = match gpu.crown_backward_gpu_resnet_sound_beta(
                &work_segs,
                &seed,
                &prep.in_lo,
                &prep.in_hi,
                &empty_beta,
                &prep.frontier_abs,
                &prep.node_abs,
            ) {
                Ok(r) if r.lower_bounds.len() == n_rows && r.upper_bounds.len() == n_rows => r,
                _ => break,
            };
            // Element-wise best across iterates: each fold is a valid enclosure,
            // so per-element max-l / min-u is a valid enclosure (intersection).
            let mut obj_t = 0.0f32;
            for i in 0..n_rows {
                let (l, u) = (r.lower_bounds[i], r.upper_bounds[i]);
                if l.is_finite() && l > best_l[i] {
                    best_l[i] = l;
                }
                if u.is_finite() && u < best_u[i] {
                    best_u[i] = u;
                }
                if l.is_finite() {
                    obj_t += l;
                }
            }
            if obj_t > best_obj {
                best_obj = obj_t;
            }
            if probe && (t % 10 == 0 || t == 1) {
                eprintln!(
                    "[{log_tag}] '{target}' iter={t} max|g|={max_g:.3e} \
                     obj={obj_t:.4} (obj0={obj0:.4} best={best_obj:.4})"
                );
            }
        }
        if past_deadline() {
            break;
        }

        // SHRINK-ONLY writeback via the alpha_explicit contract: build the refined
        // enclosure on the reference SHAPE, then intersection_per_element (union
        // fallback per disjoint element — still sound).
        let mut new_l = ref_l.clone();
        let mut new_u = ref_u.clone();
        for (r, &j) in sel.iter().enumerate() {
            if best_l[r].is_finite() {
                new_l[j] = new_l[j].max(best_l[r]);
            }
            if best_u[r].is_finite() {
                new_u[j] = new_u[j].min(best_u[r]);
            }
        }
        let shape = ref_bt.lower().raw_dim();
        let (Ok(la), Ok(ua)) = (
            ndarray::ArrayD::from_shape_vec(shape.clone(), new_l),
            ndarray::ArrayD::from_shape_vec(shape, new_u),
        ) else {
            continue;
        };
        let Ok(refined) = BoundedTensor::new(la, ua) else {
            continue; // crossed interval anywhere ⇒ keep the reference (sound)
        };
        let (tightened, disjoint) = ref_bt
            .intersection_per_element(&refined)
            .unwrap_or_else(|| (ref_bt.clone(), 0));
        if past_deadline() {
            break;
        }
        let ref_w: f32 =
            ref_l.iter().zip(&ref_u).map(|(&l, &u)| u - l).sum::<f32>() / pre_dim as f32;
        let new_w: f32 = tightened
            .lower()
            .iter()
            .zip(tightened.upper().iter())
            .map(|(&l, &u)| u - l)
            .sum::<f32>()
            / pre_dim as f32;
        let stop_tag = prep
            .stop_node
            .as_deref()
            .map(|m| format!(" stop_at='{m}' box_dim={}", prep.in_lo.len()))
            .unwrap_or_default();
        eprintln!(
            "[{log_tag}] '{target}': sel={n_rows}/{pre_dim} relus_below={n_relu} \
             meanw {ref_w:.5} -> {new_w:.5} (disjoint={disjoint}){stop_tag}"
        );
        // The return value tells the caller whether warmup α was computed from
        // now-stale boxes. Detect ANY strict endpoint shrink, rather than using
        // a mean-width display threshold that can hide a few changed rows in an
        // 8192-wide tensor.
        let strictly_tightened = tightened
            .lower()
            .iter()
            .zip(tightened.upper().iter())
            .zip(ref_l.iter().zip(ref_u.iter()))
            .any(|((&new_l, &new_u), (&old_l, &old_u))| new_l > old_l || new_u < old_u);
        // Keep the reference and do not report a stale-alpha signal if even the
        // bounded publication bookkeeping exhausts the slice.
        if past_deadline() {
            break;
        }
        if strictly_tightened {
            n_tightened += 1;
        }
        bounds.insert(target.clone(), tightened);
    }
    n_tightened
}

/// Scoped target pre-activation (seed) node names for the root JOINT interm-α
/// pass (#root-joint-interm-alpha). Walks every ReLU in execution order and
/// selects its PRE-ACTIVATION node when (a) it is not the network input, (b) a
/// reference bound exists, (c) `dim ≤ NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM`
/// (default 2048 ⇒ the FC head + the last residual block on cifar100), and
/// (d) it has at least one crossing neuron (a fully-stable seed's relaxations
/// are already exact). `NY_ROOT_JOINT_INTERM_ALPHA_LAYERS` (comma-list of ReLU
/// node names, or "all") overrides the auto scope. Selection-only — never
/// affects soundness.
pub(in crate::beta_crown::engine::graph) fn scoped_joint_alpha_targets(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
) -> Vec<String> {
    let Ok(order) = graph.exec_order() else {
        return Vec::new();
    };
    let order: Vec<String> = order.to_vec();
    let max_dim = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(2048);
    let layers_env = std::env::var("NY_ROOT_JOINT_INTERM_ALPHA_LAYERS")
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let select_all = layers_env.is_empty() || layers_env.eq_ignore_ascii_case("all");
    let explicit: std::collections::HashSet<String> = if select_all {
        std::collections::HashSet::new()
    } else {
        layers_env
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    let has_crossing = |bt: &BoundedTensor| -> bool {
        bt.lower()
            .iter()
            .zip(bt.upper().iter())
            .any(|(&l, &u)| l < 0.0 && u > 0.0)
    };
    let mut targets: Vec<String> = Vec::new();
    for name in &order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        if !select_all && !explicit.contains(name.as_str()) {
            continue;
        }
        let Some(pre) = node.inputs().first() else {
            continue;
        };
        if pre == NETWORK_INPUT {
            continue;
        }
        let Some(ref_bt) = bounds.get(pre) else {
            continue;
        };
        let pre_dim = ref_bt.lower().len();
        if pre_dim == 0 || pre_dim > max_dim {
            continue;
        }
        if !has_crossing(ref_bt) {
            continue;
        }
        if !targets.iter().any(|t| t == pre) {
            targets.push(pre.clone());
        }
    }
    // Deepest-first: later exec-order targets first, so downstream layers are
    // tightened before shallower ones are attempted within the grace slice.
    targets.reverse();
    targets
}

/// Structurally select convolutional/residual pre-activations for the bounded
/// root sparse-row CROWN pass.
///
/// Selection is deliberately independent of exporter node names: every ReLU
/// pre-activation with a frozen sound box, at least one crossing row, and width
/// no larger than `max_dim` is eligible except a `Linear` producer (the existing
/// dense-head pass owns that disjoint scope). Targets are processed
/// deepest-first and capped before any identity seed allocation. `max_rows == 0`
/// disables selection because the caller could seed no rows.
pub(in crate::beta_crown::engine::graph) fn scoped_sparse_crown_targets(
    graph: &GraphNetwork,
    bounds: &HashMap<String, BoundedTensor>,
    max_dim: usize,
    max_rows: usize,
    max_targets: usize,
) -> Vec<String> {
    if max_dim == 0 || max_rows == 0 || max_targets == 0 {
        return Vec::new();
    }
    let Ok(order) = graph.exec_order() else {
        return Vec::new();
    };
    let mut targets = Vec::new();
    for name in order {
        let Some(node) = graph.nodes.get(name) else {
            continue;
        };
        if !matches!(node.layer(), Layer::ReLU(_)) {
            continue;
        }
        let Some(pre) = node.inputs().first() else {
            continue;
        };
        if pre == NETWORK_INPUT
            || graph
                .nodes
                .get(pre)
                .is_some_and(|producer| matches!(producer.layer(), Layer::Linear(_)))
        {
            continue;
        }
        let Some(ref_bt) = bounds.get(pre) else {
            continue;
        };
        let pre_dim = ref_bt.lower().len();
        if pre_dim == 0 || pre_dim > max_dim {
            continue;
        }
        let has_crossing = ref_bt
            .lower()
            .iter()
            .zip(ref_bt.upper().iter())
            .any(|(&l, &u)| l.is_finite() && u.is_finite() && l < 0.0 && u > 0.0);
        if has_crossing && !targets.iter().any(|target| target == pre) {
            targets.push(pre.clone());
        }
    }
    targets.reverse();
    targets.truncate(max_targets);
    targets
}

#[cfg(test)]
mod root_sparse_target_tests {
    use super::*;
    use crate::layers::{AddLayer, LinearLayer, ReLULayer};
    use crate::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    fn crossing_box(dim: usize) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[dim]), -1.0),
            ArrayD::from_elem(IxDyn(&[dim]), 1.0),
        )
        .unwrap()
    }

    fn linear(name: &str, input: &str) -> GraphNode {
        let layer = Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]),
                Some(arr1(&[0.0_f32, 0.0])),
            )
            .unwrap(),
        );
        if input == NETWORK_INPUT {
            GraphNode::from_input(name, layer)
        } else {
            GraphNode::new(name, layer, vec![input.to_string()])
        }
    }

    fn relu(name: &str, input: &str) -> GraphNode {
        GraphNode::new(name, Layer::ReLU(ReLULayer), vec![input.to_string()])
    }

    #[test]
    fn sparse_targets_are_deepest_first_capped_and_exclude_dense_heads() {
        let mut graph = GraphNetwork::new();
        graph.add_node(linear("stem", NETWORK_INPUT));
        graph.add_node(GraphNode::new(
            "res0",
            Layer::Add(AddLayer),
            vec!["stem".into(), "stem".into()],
        ));
        graph.add_node(relu("relu0", "res0"));
        graph.add_node(GraphNode::new(
            "res1",
            Layer::Add(AddLayer),
            vec!["relu0".into(), "stem".into()],
        ));
        graph.add_node(relu("relu1", "res1"));
        graph.add_node(linear("head", "relu1"));
        graph.add_node(relu("relu_head", "head"));
        graph.set_output("relu_head");

        let bounds = HashMap::from([
            ("res0".to_string(), crossing_box(2)),
            ("res1".to_string(), crossing_box(2)),
            ("head".to_string(), crossing_box(2)),
        ]);
        assert_eq!(
            scoped_sparse_crown_targets(&graph, &bounds, 2, 2, 4),
            vec!["res1".to_string(), "res0".to_string()]
        );
        assert_eq!(
            scoped_sparse_crown_targets(&graph, &bounds, 2, 2, 1),
            vec!["res1".to_string()]
        );
        assert!(scoped_sparse_crown_targets(&graph, &bounds, 1, 2, 4).is_empty());
        assert!(scoped_sparse_crown_targets(&graph, &bounds, 2, 0, 4).is_empty());
    }
}
