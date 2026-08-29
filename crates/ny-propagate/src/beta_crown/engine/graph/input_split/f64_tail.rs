// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Lane hooks for the lsnc f64 tail pass (docs/LSNC_F64_TAIL_DESIGN.md §6).
//!
//! Two thin call sites, both ADDITIVE and gated `NY_F64_TAIL=1` (default OFF
//! => byte-identical lane, no f64 work, no logs):
//!
//! 1. **Batch seam** ([`f64_tail_escalate_batch`], design §6.3.1): at the end
//!    of `bound_deferred_disjunctive_domains_batch`, still-unverified domains
//!    whose grouped f32 gap is within the guard band (`NY_F64_TAIL_BAND`,
//!    default 5e-3) get a rayon-parallel certified f64 re-bound. On
//!    `Verified`, per-row certified lowers are cast DOWN to f32 and merged
//!    through the existing monotonic `tighten_obj_lower_bounds`, so the
//!    untouched downstream f32 verdict funnel passes on its own.
//! 2. **Pop-side last chance** ([`f64_tail_last_chance`], design §6.3.2):
//!    a domain about to be dropped `unresolved_due_to_unsplittable` gets one
//!    serial f64 pass regardless of band — these are exactly the queue-drain
//!    leaks of the precision-limited lsnc instances.
//!
//! Telemetry (design §6.5, greppable tag `[f64-tail]`): every escalated
//! domain logs `(gap_f32, gap_f64)` on STDERR (FIX B2 — the scored route
//! swallows `tracing::info!`; see `tail_eprintln!`) — `gap_f64 < -1e-9` on
//! the residue names the instance relaxation-limited (more precision cannot
//! help); `gap_f64 >= 0` means the blocker was fp noise (and the domain
//! verifies). `anchor_source=` tags where the pass got its per-node anchors
//! (`warm-refined` = the same refined boxes the f32 pass used, `collected` =
//! fresh `collect_intermediate_bounds`).
//!
//! # cgan endgame additions (design B, #cgan-f64-tail)
//!
//! A THIRD seam ([`F64TailHook`], survivor-polish site of the eager
//! grouped-disjunctive lane), a config chokepoint
//! ([`resolve_f64_tail_settings`]: typed preset key
//! `bab.branching.input_split.f64_tail.*` beneath the env overrides), a
//! shared per-run escalation budget ([`F64TailRunBudget`], FIX B5), and
//! box-bundled warm anchors ([`F64TailAnchors`], FIX B4).
//!
//! # Alpha-tail escalation (docs/LSNC_ALPHA_TAIL_DESIGN.md, options A+B)
//!
//! `NY_ALPHA_TAIL=1` (default OFF) arms the SAME seam with the refreshed
//! path — it composes with/supersedes `NY_F64_TAIL` (either gate arms the
//! escalation; the alpha gate switches it to the refreshed variant):
//!
//! - **Per-domain alpha refresh** (`f64_tail_verify_refreshed`): SPSA+Adam
//!   re-targeting of the root-frozen MulBinary alphas for THIS domain's box
//!   and blocking rows, keep-best-per-row (can only meet-or-beat the frozen
//!   pass). Telemetry gains `gap_f64_refreshed`.
//! - **Micro-BaB** ([`micro_bab_all_children_verified`]): a near-threshold
//!   domain the refreshed single-shot cannot close is midpoint-split (exact
//!   cover, SB-style dim choice from the domain's own `linear_bounds`) to
//!   depth `NY_ALPHA_TAIL_DEPTH` (default 3); every failing child splits
//!   further until the depth cap. ALL leaves must be certified-verified AND
//!   the per-row min-merge must still pass the grouped funnel for the parent
//!   to count; otherwise the whole escalation declines and the f32 verdict
//!   stands byte-identical (fail-closed). Soundness of the merge: the
//!   children exactly cover the parent, so each row's true min over the
//!   parent is the min over the cover, and the min of certified lowers is a
//!   certified lower. Telemetry gains `micro_bab{children,verified,depth}`.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::{Array2, ArrayD};
use ny_core::GemmEngine;
use ny_tensor::{next_down_f32, BoundedTensor};

use crate::network::graph_crown_f64_tail::{
    alpha_tail_band, alpha_tail_depth, alpha_tail_enabled, alpha_tail_iters, alpha_tail_micro_band,
    f64_tail_verify, f64_tail_verify_refreshed, AlphaTailEval, F64TailOutcome,
};
use crate::GraphNetwork;

use super::batching::tighten_obj_lower_bounds;
use super::grouped_semantics::{disjunctive_domain_priority, disjunctive_domain_verified};
use super::shared::{build_child_input_owned, MultiObjInputDomain};

/// Fixed SPSA seed for every per-domain refresh: outcomes stay deterministic
/// under rayon interleaving (each domain's refresh depends only on its own
/// inputs). Soundness never depends on the draws.
const ALPHA_TAIL_SPSA_SEED: u64 = 0xA1FA_7A11;

/// FIX B2 (#cgan-f64-tail, adversarial verdict): `[f64-tail]` telemetry goes
/// to STDERR. The scored vnncomp entry point swallows `tracing::info!`
/// (stated in-tree at `root_bounds.rs`; the ledger bought this lesson with a
/// vacuous stacked A/B and again with the `best_gap` blindness), and
/// `RUST_LOG` is not honoured by the scored binary, so `info!` lines can
/// never satisfy R9 on the route that matters. Rate discipline mirrors
/// `[disjunctive-trajectory]` / `[cgan-polish]`: verdict-bearing `verified`
/// lines are ALWAYS printed; per-domain not-verified/declined and per-batch
/// summary lines stop after `MAX_LINES` per process (a 900 s cgan run pops
/// ~600-1300 domains — the cap keeps a pathological queue from flooding the
/// log while leaving far more than any probe needs).
macro_rules! tail_eprintln {
    (always, $($arg:tt)*) => {
        eprintln!($($arg)*)
    };
    ($($arg:tt)*) => {{
        use std::sync::atomic::{AtomicUsize, Ordering};
        static EMITTED: AtomicUsize = AtomicUsize::new(0);
        const MAX_LINES: usize = 4000;
        if EMITTED.fetch_add(1, Ordering::Relaxed) < MAX_LINES {
            eprintln!($($arg)*);
        }
    }};
}

/// SB-style coefficient clamp for the micro-BaB split-dim score (mirrors the
/// lane heuristic's `input_split_coeff_thresh` default, `config/defaults.rs`).
const MICRO_SPLIT_COEFF_THRESH: f32 = 1e-3;

// ---------------------------------------------------------------------------
// Anchor parity (design B step B3 + FIX B4, #cgan-f64-tail).
// ---------------------------------------------------------------------------

/// Warm-refined per-node anchors BUNDLED WITH the child box they were
/// computed for (FIX B4): the tail's relaxations are only valid when anchored
/// on bounds that enclose THIS box — anchors from a smaller/sibling box are a
/// false-VERIFY path. Fields are private and the ONLY constructor is
/// [`F64TailAnchors::bundle_for`], called at the single site where
/// `compute_warm_start_crown_bounds_with_refined_alpha` has just computed the
/// node bounds FOR its own `input_bounds` argument — so cross-box reuse is
/// unrepresentable: the escalation reads BOTH the box and the anchors out of
/// one struct ([`F64TailHook::escalate_merge`] passes `self.child_input` as
/// the verified box). A wider box than the queued (clipped) child is sound:
/// verifying the superset verifies the subset, and its row lowers only merge
/// monotonically.
pub(crate) struct F64TailAnchors {
    child_input: BoundedTensor,
    node_bounds: HashMap<String, BoundedTensor>,
}

impl F64TailAnchors {
    /// Bundle node bounds with the exact box they were computed for. The two
    /// arguments MUST come from the same bound invocation (enforced by
    /// construction discipline: the only production caller sits inside
    /// `compute_warm_start_crown_bounds_with_refined_alpha`, which received
    /// `input_bounds` and derived `node_bounds` from it in the same call).
    pub(crate) fn bundle_for(
        input_bounds: &BoundedTensor,
        node_bounds: HashMap<String, BoundedTensor>,
    ) -> Self {
        Self {
            child_input: input_bounds.clone(),
            node_bounds,
        }
    }
}

// ---------------------------------------------------------------------------
// Config chokepoint (design B step B5, FIX B5): typed preset key
// `bab.branching.input_split.f64_tail.{enabled, band, max_escalations}`
// resolved ONCE against the legacy env gates, gating BOTH seams (the batch
// seam and the survivor-polish hook). Env cannot fire in competition
// (`run_instance.sh` exports no `NY_*`), so the shipped shape is the
// typed key; the env names keep working for diagnostics and override in both
// directions where they exist (`NY_F64_TAIL` 1/0, `NY_F64_TAIL_BAND`).
// `max_escalations` is config-only (0 = unlimited — the legacy env-armed lsnc
// behavior; adding a new raw env read would raise the ny-levers ratchet).
// ---------------------------------------------------------------------------

/// Effective f64-tail arming for one verifier run.
#[derive(Debug, Clone, Copy)]
pub(crate) struct F64TailSettings {
    pub(crate) enabled: bool,
    pub(crate) band: f32,
    /// Shared across BOTH seams per run; 0 = unlimited.
    pub(crate) max_escalations: usize,
    /// Resolution provenance for the R9 line: `env` / `config` / `default`.
    pub(crate) source: &'static str,
}

/// Resolve the run's f64-tail settings from the typed config + env overrides.
pub(crate) fn resolve_f64_tail_settings(
    config: &crate::beta_crown::config::BetaCrownConfig,
) -> F64TailSettings {
    use crate::network::graph_crown_f64_tail::{f64_tail_band_env, f64_tail_env_state};
    let (enabled, source) = match f64_tail_env_state() {
        Some(on) => (on, "env"),
        None => match config.input_split_f64_tail_enabled {
            Some(on) => (on, "config"),
            None => (false, "default"),
        },
    };
    let band = f64_tail_band_env()
        .or(config.input_split_f64_tail_band)
        .filter(|b| b.is_finite() && *b >= 0.0)
        .unwrap_or(5e-3);
    let max_escalations = config.input_split_f64_tail_max_escalations.unwrap_or(0);
    F64TailSettings {
        enabled,
        band,
        max_escalations,
        source,
    }
}

/// Per-run escalation budget shared by the batch seam and the polish hook
/// (FIX B5: the cap gates BOTH seams). Not `Sync` on purpose: candidate
/// selection happens on the coordinator thread before any rayon fan-out.
pub(crate) struct F64TailRunBudget {
    used: std::cell::Cell<usize>,
}

impl F64TailRunBudget {
    pub(crate) fn new() -> Self {
        Self {
            used: std::cell::Cell::new(0),
        }
    }

    /// How many escalations may still run under `cap` (0 = unlimited).
    fn remaining(&self, cap: usize) -> usize {
        if cap == 0 {
            usize::MAX
        } else {
            cap.saturating_sub(self.used.get())
        }
    }

    fn charge(&self, n: usize) {
        self.used.set(self.used.get().saturating_add(n));
    }

    pub(crate) fn used(&self) -> usize {
        self.used.get()
    }
}

// ---------------------------------------------------------------------------
// Survivor-polish-seam hook (design B step B4, #cgan-f64-tail).
// ---------------------------------------------------------------------------

/// Per-call wall slice for one hook escalation: the scalar f64 conv walk on a
/// 32×32 ConvTranspose stack is estimated ~1-10 s/domain (design §2 B4); the
/// slice bounds the worst case so a pathological domain cannot eat the run.
/// With the shipped cap of 8 escalations, worst case ≈ 8×10 s ≈ 9% of a
/// 900 s budget — the pre-registered wall-parity kill criterion owns that.
const F64_TAIL_HOOK_CALL_SLICE: Duration = Duration::from_secs(10);

/// Third injection point: certified f64 escalation AT the survivor-polish
/// seam of the eager grouped-disjunctive lane. Bundles every dependency so
/// `screen_disjunctive_child` needs a single optional reference; the shared
/// [`F64TailRunBudget`] gates this seam AND the batch seam under one cap
/// (FIX B5). Micro-BaB stays OFF here by construction (this hook calls the
/// plain single-shot [`f64_tail_verify`] — depth 0).
pub(crate) struct F64TailHook<'a> {
    graph: &'a GraphNetwork,
    spec_matrix: &'a Array2<f32>,
    thresholds: &'a [f32],
    clause_sizes: &'a [usize],
    mul_binary_alphas: Option<&'a HashMap<String, Array2<f32>>>,
    engine: Option<&'a dyn GemmEngine>,
    deadline: Option<Instant>,
    band: f32,
    max_escalations: usize,
    budget: &'a F64TailRunBudget,
    call_slice: Duration,
}

impl<'a> F64TailHook<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        graph: &'a GraphNetwork,
        spec_matrix: &'a Array2<f32>,
        thresholds: &'a [f32],
        clause_sizes: &'a [usize],
        mul_binary_alphas: Option<&'a HashMap<String, Array2<f32>>>,
        engine: Option<&'a dyn GemmEngine>,
        deadline: Option<Instant>,
        settings: F64TailSettings,
        budget: &'a F64TailRunBudget,
    ) -> Self {
        Self {
            graph,
            spec_matrix,
            thresholds,
            clause_sizes,
            mul_binary_alphas,
            engine,
            deadline,
            band: settings.band,
            max_escalations: settings.max_escalations,
            budget,
            call_slice: F64_TAIL_HOOK_CALL_SLICE,
        }
    }

    /// #cgan-dark-tail FIX-1: tighten this hook's per-call wall slice with the
    /// typed per-child cap (`bab.branching.input_split.child_bound_max_secs`).
    ///
    /// This seam was ALREADY capped — `escalate_merge` composes
    /// `min(self.deadline, now + call_slice)` with a 10 s
    /// [`F64_TAIL_HOOK_CALL_SLICE`] — so it is not the uncapped consumer FIX-1
    /// targets. This only keeps it consistent when a preset asks for something
    /// tighter: the slice can only SHRINK, and `None` (the shipped default)
    /// leaves it byte-for-byte at 10 s.
    ///
    /// NOTE the SHAPE. The cap is a `Duration` applied per call, never a
    /// pre-composed `Instant`. Composing an absolute deadline here would be a
    /// bug: `new` runs once at BaB start, so `start + cap` would silently
    /// disable the hook `cap` seconds into the run instead of capping each
    /// call.
    #[must_use]
    pub(crate) fn with_child_bound_cap(mut self, cap: Option<Duration>) -> Self {
        if let Some(cap) = cap {
            self.call_slice = self.call_slice.min(cap);
        }
        self
    }

    /// Escalate one in-band survivor. On a certified `Verified`, merges the
    /// per-row f64 lowers into `obj_bounds` through the EXISTING funnel
    /// (`certified_fresh_rows` + monotonic `tighten_obj_lower_bounds` — never
    /// hand-rolled) and returns `true`; every other outcome (out of band,
    /// cap exhausted, deadline, NotVerified, Unsupported) leaves `obj_bounds`
    /// byte-identical and returns `false`.
    pub(crate) fn escalate_merge(
        &self,
        anchors: &F64TailAnchors,
        obj_bounds: &mut Vec<(f32, f32)>,
        gap_f32: f32,
    ) -> bool {
        if !gap_f32.is_finite() || gap_f32 < -self.band {
            return false;
        }
        if self.budget.remaining(self.max_escalations) == 0 {
            tail_eprintln!(
                "[f64-tail] survivor-polish SKIPPED (escalation cap {} exhausted) gap_f32={:.6}",
                self.max_escalations,
                gap_f32
            );
            return false;
        }
        let now = Instant::now();
        if self.deadline.is_some_and(|d| now >= d) {
            return false;
        }
        let call_deadline = match self.deadline {
            Some(d) => d.min(now + self.call_slice),
            None => now + self.call_slice,
        };
        self.budget.charge(1);
        let used = self.budget.used();
        let outcome = f64_tail_verify(
            self.graph,
            anchors.child_input(),
            self.spec_matrix,
            self.thresholds,
            self.clause_sizes,
            self.mul_binary_alphas,
            Some(anchors.node_bounds()),
            self.engine,
            Some(call_deadline),
        );
        match outcome {
            F64TailOutcome::Verified { row_lowers } => {
                tail_eprintln!(
                    always,
                    "[f64-tail] survivor-polish verified gap_f32={:.6} anchor_source=warm-refined escalations={}/{} (certified f64)",
                    gap_f32,
                    used,
                    self.max_escalations
                );
                let fresh = certified_fresh_rows(obj_bounds, &row_lowers, self.thresholds);
                *obj_bounds = tighten_obj_lower_bounds(obj_bounds, fresh);
                true
            }
            F64TailOutcome::NotVerified { min_gap_f64 } => {
                tail_eprintln!(
                    "[f64-tail] survivor-polish not-verified gap_f32={:.6} gap_f64={:.3e} anchor_source=warm-refined escalations={}/{}",
                    gap_f32,
                    min_gap_f64,
                    used,
                    self.max_escalations
                );
                false
            }
            F64TailOutcome::Unsupported => {
                tail_eprintln!(
                    "[f64-tail] survivor-polish declined (unsupported) gap_f32={:.6} anchor_source=warm-refined escalations={}/{}",
                    gap_f32,
                    used,
                    self.max_escalations
                );
                false
            }
        }
    }
}

impl F64TailAnchors {
    fn child_input(&self) -> &BoundedTensor {
        &self.child_input
    }

    fn node_bounds(&self) -> &HashMap<String, BoundedTensor> {
        &self.node_bounds
    }
}

/// Cast a certified f64 lower bound DOWN into the f32 carrier: `f64 -> f32`
/// round-to-nearest followed by one directed step, so the result is `<=` the
/// certified f64 value (`next_down_f32(RN(x)) <= x` for every real `x`).
#[inline]
fn certified_lower_to_f32(l_cert: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    let cast = next_down_f32(l_cert as f32);
    if cast.is_finite() {
        cast
    } else {
        f32::NEG_INFINITY
    }
}

/// Certified-f64 rows -> f32 merge candidates: rows whose directed downcast
/// STILL clears the threshold get the cast value; every other row stays
/// `-inf` so the monotonic `tighten_obj_lower_bounds` leaves it untouched.
fn certified_fresh_rows(
    obj_bounds: &[(f32, f32)],
    row_lowers: &[f64],
    thresholds: &[f32],
) -> Vec<(f32, f32)> {
    obj_bounds
        .iter()
        .zip(row_lowers.iter())
        .zip(thresholds.iter())
        .map(|(((_old_l, old_u), &l_cert), &t)| {
            if l_cert.is_finite() {
                let cast = certified_lower_to_f32(l_cert);
                if cast > t {
                    return (cast, *old_u);
                }
            }
            (f32::NEG_INFINITY, *old_u)
        })
        .collect()
}

/// Merge certified per-row f64 lowers into a domain (the landed batch-seam
/// merge: monotonic tighten + priority recompute).
fn merge_certified_rows(
    domain: &mut MultiObjInputDomain,
    row_lowers: &[f64],
    thresholds: &[f32],
    clause_sizes: &[usize],
) {
    let fresh = certified_fresh_rows(&domain.obj_bounds, row_lowers, thresholds);
    domain.obj_bounds = tighten_obj_lower_bounds(&domain.obj_bounds, fresh);
    domain.priority = disjunctive_domain_priority(&domain.obj_bounds, thresholds, clause_sizes);
}

/// Micro-BaB telemetry counters.
struct MicroBabStats {
    /// Boxes evaluated (children at every depth, best-first short-circuit).
    children: usize,
    /// Boxes certified `Verified`.
    verified: usize,
    /// Deepest evaluated child depth (parent = depth 0).
    max_depth: usize,
}

/// Pick the micro-BaB split dimension: SB-style score
/// `width_d * max_rows(|lA[r,d]|).max(thresh)` from the domain's own CROWN
/// input coefficients when available, else width-only. Returns the exact
/// midpoint cover `(left_lo, left_hi, right_lo, right_hi)` or `None` when no
/// dimension admits a strict split (the caller declines — fail-closed).
/// Split-axis choice only selects WHICH exact cover is attempted; soundness
/// never depends on it.
fn micro_split(
    flat_lo: &ArrayD<f32>,
    flat_hi: &ArrayD<f32>,
    coeffs: Option<&Array2<f32>>,
) -> Option<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
    let len = flat_lo.len();
    let mut best: Option<(usize, f32)> = None;
    for d in 0..len {
        let l = flat_lo[[d]];
        let u = flat_hi[[d]];
        let width = u - l;
        if !width.is_finite() || width <= 0.0 {
            continue;
        }
        let coeff = coeffs
            .filter(|a| a.ncols() == len && a.nrows() > 0)
            .map(|a| {
                let mut m = 0.0f32;
                for r in 0..a.nrows() {
                    let v = a[[r, d]].abs();
                    if v.is_finite() && v > m {
                        m = v;
                    }
                }
                m.max(MICRO_SPLIT_COEFF_THRESH)
            })
            .unwrap_or(1.0);
        let score = width * coeff;
        if score.is_finite() && best.is_none_or(|(_, s)| score > s) {
            best = Some((d, score));
        }
    }
    let (dim, _) = best?;
    let l = flat_lo[[dim]];
    let u = flat_hi[[dim]];
    let mid = l + (u - l) / 2.0;
    if !(mid > l && mid < u) {
        return None;
    }
    let mut left_hi = flat_hi.clone();
    left_hi[[dim]] = mid;
    let mut right_lo = flat_lo.clone();
    right_lo[[dim]] = mid;
    Some((flat_lo.clone(), left_hi, right_lo, flat_hi.clone()))
}

/// Recursive micro-BaB verify of one child box: refreshed certified f64
/// pass; a failing child splits further while `depth < depth_cap`. `true`
/// ONLY when the entire subtree under this box is certified-verified; every
/// verified leaf's rows are min-folded into `acc_rows`.
#[allow(clippy::too_many_arguments)]
fn micro_verify_box(
    graph: &GraphNetwork,
    flat_lo: ArrayD<f32>,
    flat_hi: ArrayD<f32>,
    shape: &[usize],
    coeffs: Option<&Array2<f32>>,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    warm_alphas: Option<&HashMap<String, Array2<f32>>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    depth: usize,
    depth_cap: usize,
    iters: usize,
    max_evals: usize,
    stats: &mut MicroBabStats,
    acc_rows: &mut [f64],
) -> bool {
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return false;
    }
    if stats.children >= max_evals {
        return false;
    }
    let Ok(child) = build_child_input_owned(flat_lo.clone(), flat_hi.clone(), shape) else {
        return false;
    };
    stats.children += 1;
    if depth > stats.max_depth {
        stats.max_depth = depth;
    }
    let eval = f64_tail_verify_refreshed(
        graph,
        &child,
        spec_matrix,
        thresholds,
        clause_sizes,
        warm_alphas,
        None,
        engine,
        deadline,
        iters,
        ALPHA_TAIL_SPSA_SEED,
    );
    match eval.outcome {
        F64TailOutcome::Verified { ref row_lowers } => {
            stats.verified += 1;
            for (acc, &l) in acc_rows.iter_mut().zip(row_lowers.iter()) {
                if l < *acc {
                    *acc = l;
                }
            }
            true
        }
        F64TailOutcome::NotVerified { .. } if depth < depth_cap => {
            // Warm-start the grandchildren from this child's refreshed map.
            let warm_owned = eval.refreshed_alphas;
            let warm_next = warm_owned.as_ref().or(warm_alphas);
            let Some((ll, lh, rl, rh)) = micro_split(&flat_lo, &flat_hi, coeffs) else {
                return false;
            };
            micro_verify_box(
                graph,
                ll,
                lh,
                shape,
                coeffs,
                spec_matrix,
                thresholds,
                clause_sizes,
                warm_next,
                engine,
                deadline,
                depth + 1,
                depth_cap,
                iters,
                max_evals,
                stats,
                acc_rows,
            ) && micro_verify_box(
                graph,
                rl,
                rh,
                shape,
                coeffs,
                spec_matrix,
                thresholds,
                clause_sizes,
                warm_next,
                engine,
                deadline,
                depth + 1,
                depth_cap,
                iters,
                max_evals,
                stats,
                acc_rows,
            )
        }
        _ => false,
    }
}

/// Micro-BaB over one escalated parent (alpha-tail design option B): split
/// the parent into an exact midpoint cover and require EVERY leaf to be
/// certified-verified by the refreshed f64 pass. Returns the per-row
/// min-over-leaves certified lowers on success (sound: the true row min over
/// the parent is the min over the cover), `None` on ANY failure — the
/// escalation then declines and nothing is mutated (fail-closed).
#[allow(clippy::too_many_arguments)]
fn micro_bab_all_children_verified(
    graph: &GraphNetwork,
    parent_box: &BoundedTensor,
    coeffs: Option<&Array2<f32>>,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    warm_alphas: Option<&HashMap<String, Array2<f32>>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    depth_cap: usize,
    iters: usize,
) -> (MicroBabStats, Option<Vec<f64>>) {
    let mut stats = MicroBabStats {
        children: 0,
        verified: 0,
        max_depth: 0,
    };
    let total_rows: usize = clause_sizes.iter().sum();
    let mut acc_rows = vec![f64::INFINITY; total_rows];
    let flat = parent_box.flatten();
    let flat_lo = flat.lower().clone();
    let flat_hi = flat.upper().clone();
    let shape = parent_box.shape().to_vec();
    // Hard cap on evaluated boxes: complete best-first exploration to the
    // depth cap evaluates at most 2^(cap+1) - 2 boxes (interior + leaves).
    let max_evals = (1usize << (depth_cap + 1)).saturating_sub(2);
    let Some((ll, lh, rl, rh)) = micro_split(&flat_lo, &flat_hi, coeffs) else {
        return (stats, None);
    };
    let ok = micro_verify_box(
        graph,
        ll,
        lh,
        &shape,
        coeffs,
        spec_matrix,
        thresholds,
        clause_sizes,
        warm_alphas,
        engine,
        deadline,
        1,
        depth_cap,
        iters,
        max_evals,
        &mut stats,
        &mut acc_rows,
    ) && micro_verify_box(
        graph,
        rl,
        rh,
        &shape,
        coeffs,
        spec_matrix,
        thresholds,
        clause_sizes,
        warm_alphas,
        engine,
        deadline,
        1,
        depth_cap,
        iters,
        max_evals,
        &mut stats,
        &mut acc_rows,
    );
    if ok {
        (stats, Some(acc_rows))
    } else {
        (stats, None)
    }
}

/// Batch-seam escalation (design §6.3 call site 1). Gate-off => immediate
/// return, no reads, no logs (byte-identity leg of the parity contract).
///
/// Only `Verified` outcomes mutate a domain, and only by the monotonic
/// `tighten_obj_lower_bounds` merge of per-row certified lowers that STILL
/// clear their thresholds after the directed f32 downcast (rows that lose
/// the razor-edge cast are left untouched; the pop-side hook is the backstop).
///
/// With `NY_ALPHA_TAIL=1` the same seam runs the refreshed variant
/// (per-domain alpha refresh + micro-BaB, module doc); with only
/// `NY_F64_TAIL=1` the landed frozen-alpha path runs byte-identically.
///
/// #cgan-dark-tail FIX-1: `deadline` arrives as the ABSOLUTE BaB ceiling, so
/// before the fix a single escalation here was entitled to the whole remaining
/// budget — the same defect the per-child cap exists to close, on the other
/// seam. The typed cap is re-resolved from the SAME `config` and composed by
/// the SAME `disjunctive_multi_clause::child_bound_deadline` law, so both
/// seams obey one rule (`None` = no cap = byte-identical). The
/// fan-out is rayon-parallel, so one composed slice bounds every candidate's
/// WALL time, which is what a per-child wall cap means for a parallel batch.
/// MEASURED headroom: 363 escalations at a 0.1235 s mean / 5.68 s total, so a
/// cap of seconds never binds here — this is a ceiling on the pathological
/// case, not a schedule.
#[allow(clippy::too_many_arguments)]
pub(super) fn f64_tail_escalate_batch(
    domains: &mut [MultiObjInputDomain],
    graph: &GraphNetwork,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    config: &crate::beta_crown::config::BetaCrownConfig,
    budget: Option<&F64TailRunBudget>,
) {
    // FIX B5: one config chokepoint gates this seam and the polish hook —
    // typed key first, env override on top; the cap is shared via `budget`.
    let settings = resolve_f64_tail_settings(config);
    let alpha_armed = alpha_tail_enabled();
    if !settings.enabled && !alpha_armed {
        return;
    }
    // #cgan-dark-tail FIX-1. Composed AFTER the gate so the dark path still
    // reads nothing, and before candidate selection so every escalation below
    // shares one slice.
    let deadline = super::disjunctive_multi_clause::child_bound_deadline(
        deadline,
        super::disjunctive_multi_clause::resolve_child_bound_cap(config),
    );
    let band = if alpha_armed {
        alpha_tail_band()
    } else {
        settings.band
    };
    let mut candidates: Vec<(usize, f32)> = domains
        .iter()
        .enumerate()
        .filter(|(_, domain)| !domain.needs_bounding)
        .filter(|(_, domain)| {
            !disjunctive_domain_verified(&domain.obj_bounds, thresholds, clause_sizes)
        })
        .filter_map(|(idx, domain)| {
            let gap = disjunctive_domain_priority(&domain.obj_bounds, thresholds, clause_sizes);
            (gap.is_finite() && gap >= -band).then_some((idx, gap))
        })
        .collect();
    if candidates.is_empty() {
        return;
    }
    if let Some(budget) = budget {
        let remaining = budget.remaining(settings.max_escalations);
        if remaining == 0 {
            tail_eprintln!(
                "[f64-tail] batch-seam SKIPPED (escalation cap {} exhausted; {} in-band candidates)",
                settings.max_escalations,
                candidates.len()
            );
            return;
        }
        if candidates.len() > remaining {
            // Spend the residual budget on the knife edge: closest-to-zero
            // gaps first (selection only — every escalation stays certified).
            candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
            candidates.truncate(remaining);
        }
        budget.charge(candidates.len());
    }

    // B0 discriminator (design B): `Unsupported` conflates the OP GAP with a
    // deadline/anchor abort. Print the graph-level support verdict once per
    // process so a probe can tell "the walker cannot do this graph" (the
    // pre-B1/B2 state) from "a pass aborted" — and per-escalation wall time.
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static PRINTED: AtomicBool = AtomicBool::new(false);
        if !PRINTED.swap(true, Ordering::Relaxed) {
            let census = crate::network::graph_crown_f64_tail::f64_tail_unsupported_census(graph);
            let head: Vec<String> = census
                .iter()
                .take(8)
                .map(|(n, t)| format!("{n}:{t}"))
                .collect();
            eprintln!(
                "[f64-tail] batch-seam graph_supported={} spec_rows={} clauses={} unsupported={} [{}]",
                census.is_empty(),
                spec_matrix.nrows(),
                clause_sizes.len(),
                census.len(),
                head.join(", ")
            );
        }
    }

    // Rayon-parallel over the (immutable) escalated domains, mirroring the
    // rebound fallback's RayonTaskGuard discipline (`shared_specs.rs`).
    use rayon::prelude::*;
    let domains_ro: &[MultiObjInputDomain] = domains;

    if !alpha_armed {
        // Landed frozen-alpha path (byte-identical to the f64 tail as shipped).
        let outcomes: Vec<(usize, f32, F64TailOutcome, f64)> = candidates
            .par_iter()
            .map(|&(idx, gap_f32)| {
                let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                let domain = &domains_ro[idx];
                let started = Instant::now();
                let outcome = f64_tail_verify(
                    graph,
                    domain.input_bounds.as_ref(),
                    spec_matrix,
                    thresholds,
                    clause_sizes,
                    mul_binary_alphas,
                    None,
                    engine,
                    deadline,
                );
                (idx, gap_f32, outcome, started.elapsed().as_secs_f64())
            })
            .collect();

        let mut verified = 0usize;
        for (idx, gap_f32, outcome, secs) in outcomes {
            let domain = &mut domains[idx];
            match outcome {
                F64TailOutcome::Verified { row_lowers } => {
                    merge_certified_rows(domain, &row_lowers, thresholds, clause_sizes);
                    verified += 1;
                    tail_eprintln!(always,
                        "[f64-tail] batch-seam verified depth={} gap_f32={:.6} secs={:.2} anchor_source=collected (certified f64)",
                        domain.depth, gap_f32, secs
                    );
                }
                F64TailOutcome::NotVerified { min_gap_f64 } => {
                    tail_eprintln!(
                        "[f64-tail] batch-seam not-verified depth={} gap_f32={:.6} gap_f64={:.3e} secs={:.2} anchor_source=collected",
                        domain.depth, gap_f32, min_gap_f64, secs
                    );
                }
                F64TailOutcome::Unsupported => {
                    tail_eprintln!(
                        "[f64-tail] batch-seam declined (unsupported) depth={} gap_f32={:.6} secs={:.2} anchor_source=collected",
                        domain.depth, gap_f32, secs
                    );
                }
            }
        }
        // R9 engagement line: emitted whenever the seam escalated ANYTHING,
        // not only on a verify — a probe run must be able to distinguish
        // "seam fired and declined" from "seam never fired" (FIX B2).
        tail_eprintln!(
            "[f64-tail] batch-seam escalated={} verified={}",
            candidates.len(),
            verified
        );
        return;
    }

    // Alpha-tail path: refreshed single-shot, then micro-BaB for the
    // near-threshold refusals.
    let depth_cap = alpha_tail_depth();
    let iters = alpha_tail_iters();
    let micro_band = f64::from(alpha_tail_micro_band());
    type AlphaOutcome = (
        usize,
        f32,
        AlphaTailEval,
        Option<MicroBabStats>,
        Option<Vec<f64>>,
    );
    let outcomes: Vec<AlphaOutcome> = candidates
        .par_iter()
        .map(|&(idx, gap_f32)| {
            let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
            let domain = &domains_ro[idx];
            let eval = f64_tail_verify_refreshed(
                graph,
                domain.input_bounds.as_ref(),
                spec_matrix,
                thresholds,
                clause_sizes,
                mul_binary_alphas,
                None,
                engine,
                deadline,
                iters,
                ALPHA_TAIL_SPSA_SEED,
            );
            let mut micro_stats = None;
            let mut micro_rows = None;
            if matches!(eval.outcome, F64TailOutcome::NotVerified { .. })
                && depth_cap > 0
                && eval.gap_refreshed.is_finite()
                && eval.gap_refreshed >= -micro_band
                && deadline.is_none_or(|d| Instant::now() < d)
            {
                let warm = eval.refreshed_alphas.as_ref().or(mul_binary_alphas);
                let coeffs = domain.linear_bounds.as_ref().map(|lb| lb.lower_a());
                let (stats, rows) = micro_bab_all_children_verified(
                    graph,
                    domain.input_bounds.as_ref(),
                    coeffs,
                    spec_matrix,
                    thresholds,
                    clause_sizes,
                    warm,
                    engine,
                    deadline,
                    depth_cap,
                    iters,
                );
                micro_stats = Some(stats);
                micro_rows = rows;
            }
            (idx, gap_f32, eval, micro_stats, micro_rows)
        })
        .collect();

    let mut verified = 0usize;
    for (idx, gap_f32, eval, micro_stats, micro_rows) in outcomes {
        let domain = &mut domains[idx];
        let micro_tag = |stats: &MicroBabStats| {
            format!(
                " micro_bab{{children={} verified={} depth={}}}",
                stats.children, stats.verified, stats.max_depth
            )
        };
        match eval.outcome {
            F64TailOutcome::Verified { row_lowers } => {
                merge_certified_rows(domain, &row_lowers, thresholds, clause_sizes);
                verified += 1;
                tail_eprintln!(always,
                    "[f64-tail] batch-seam verified depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e} anchor_source=collected (refreshed certified f64)",
                    domain.depth, gap_f32, eval.gap_baseline, eval.gap_refreshed
                );
            }
            F64TailOutcome::NotVerified { .. } => {
                if let Some(rows) = micro_rows {
                    // ALL leaves certified. Commit ONLY when the min-merged
                    // rows still pass the grouped funnel after the directed
                    // downcast; a per-clause witness fragmented across
                    // children cannot be expressed in per-row bounds, so it
                    // declines fail-closed (byte-identical).
                    let fresh = certified_fresh_rows(&domain.obj_bounds, &rows, thresholds);
                    let tightened = tighten_obj_lower_bounds(&domain.obj_bounds, fresh);
                    if disjunctive_domain_verified(&tightened, thresholds, clause_sizes) {
                        domain.obj_bounds = tightened;
                        domain.priority = disjunctive_domain_priority(
                            &domain.obj_bounds,
                            thresholds,
                            clause_sizes,
                        );
                        verified += 1;
                        tail_eprintln!(
                            "[f64-tail] batch-seam verified depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e}{} (certified f64 micro-BaB)",
                            domain.depth,
                            gap_f32,
                            eval.gap_baseline,
                            eval.gap_refreshed,
                            micro_stats.as_ref().map(&micro_tag).unwrap_or_default()
                        );
                    } else {
                        tail_eprintln!(
                            "[f64-tail] batch-seam micro-bab declined (witness fragmentation) depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e}{}",
                            domain.depth,
                            gap_f32,
                            eval.gap_baseline,
                            eval.gap_refreshed,
                            micro_stats.as_ref().map(&micro_tag).unwrap_or_default()
                        );
                    }
                } else {
                    tail_eprintln!(
                        "[f64-tail] batch-seam not-verified depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e}{}",
                        domain.depth,
                        gap_f32,
                        eval.gap_baseline,
                        eval.gap_refreshed,
                        micro_stats.as_ref().map(&micro_tag).unwrap_or_default()
                    );
                }
            }
            F64TailOutcome::Unsupported => {
                tail_eprintln!(
                    "[f64-tail] batch-seam declined (unsupported) depth={} gap_f32={:.6} anchor_source=collected",
                    domain.depth, gap_f32
                );
            }
        }
    }
    // R9 engagement line (alpha-tail arm) — see the frozen-path twin above.
    tail_eprintln!(
        "[f64-tail] batch-seam escalated={} verified={} (alpha-tail)",
        candidates.len(),
        verified
    );
}

/// Pop-side last chance (design §6.3 call site 2): one serial f64 pass for a
/// domain about to be dropped `unresolved_due_to_unsplittable`, regardless of
/// band. Returns `true` ONLY on a certified f64 `Verified` — the caller then
/// counts the domain verified instead of dropping it. Gate-off => `false`
/// with no work (byte-identical control flow).
///
/// #cgan-dark-tail FIX-1: `deadline` is the ABSOLUTE BaB ceiling here too, and
/// this seam is band-FREE — it escalates whatever is about to be dropped, at
/// any gap — so it is the seam with the weakest prior on cost. The typed
/// per-child cap now bounds it by the same composition law as the other two.
/// "Rare and serial" was an assumption about the drop rate, not a measurement;
/// the cap makes it not need to hold.
#[allow(clippy::too_many_arguments)]
pub(super) fn f64_tail_last_chance(
    graph: &GraphNetwork,
    domain: &MultiObjInputDomain,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
    config: &crate::beta_crown::config::BetaCrownConfig,
) -> bool {
    // Same chokepoint as the batch seam (FIX B5): typed key first, env
    // override on top. The band is deliberately not applied here.
    let settings = resolve_f64_tail_settings(config);
    let alpha_armed = alpha_tail_enabled();
    if !settings.enabled && !alpha_armed {
        return false;
    }
    // #cgan-dark-tail FIX-1: composed after the gate, so the dark path reads
    // nothing. `None` cap yields the caller's ceiling verbatim; the helper
    // never returns `None` when a ceiling was supplied, so the `unwrap_or` is
    // unreachable defence rather than a fallback with meaning.
    let deadline = super::disjunctive_multi_clause::child_bound_deadline(
        Some(deadline),
        super::disjunctive_multi_clause::resolve_child_bound_cap(config),
    )
    .unwrap_or(deadline);
    let gap_f32 = disjunctive_domain_priority(&domain.obj_bounds, thresholds, clause_sizes);
    if alpha_armed {
        // Refreshed serial pass, no micro-BaB (design §4.1.4: drain mode is
        // rare and serial — the refresh is the cheap, high-value half).
        let eval = f64_tail_verify_refreshed(
            graph,
            domain.input_bounds.as_ref(),
            spec_matrix,
            thresholds,
            clause_sizes,
            mul_binary_alphas,
            None,
            engine,
            Some(deadline),
            alpha_tail_iters(),
            ALPHA_TAIL_SPSA_SEED,
        );
        return match eval.outcome {
            F64TailOutcome::Verified { .. } => {
                tail_eprintln!(always,
                    "[f64-tail] last-chance verified depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e} (refreshed certified f64)",
                    domain.depth, gap_f32, eval.gap_baseline, eval.gap_refreshed
                );
                true
            }
            F64TailOutcome::NotVerified { .. } => {
                tail_eprintln!(
                    "[f64-tail] last-chance not-verified depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e}",
                    domain.depth, gap_f32, eval.gap_baseline, eval.gap_refreshed
                );
                false
            }
            F64TailOutcome::Unsupported => {
                tail_eprintln!(
                    "[f64-tail] last-chance declined (unsupported) depth={} gap_f32={:.6} anchor_source=collected",
                    domain.depth, gap_f32
                );
                false
            }
        };
    }
    match f64_tail_verify(
        graph,
        domain.input_bounds.as_ref(),
        spec_matrix,
        thresholds,
        clause_sizes,
        mul_binary_alphas,
        None,
        engine,
        Some(deadline),
    ) {
        F64TailOutcome::Verified { .. } => {
            tail_eprintln!(always,
                "[f64-tail] last-chance verified depth={} gap_f32={:.6} anchor_source=collected (certified f64)",
                domain.depth, gap_f32
            );
            true
        }
        F64TailOutcome::NotVerified { min_gap_f64 } => {
            tail_eprintln!(
                "[f64-tail] last-chance not-verified depth={} gap_f32={:.6} gap_f64={:.3e} anchor_source=collected",
                domain.depth, gap_f32, min_gap_f64
            );
            false
        }
        F64TailOutcome::Unsupported => {
            tail_eprintln!(
                "[f64-tail] last-chance declined (unsupported) depth={} gap_f32={:.6} anchor_source=collected",
                domain.depth, gap_f32
            );
            false
        }
    }
}

#[cfg(test)]
mod tests;
