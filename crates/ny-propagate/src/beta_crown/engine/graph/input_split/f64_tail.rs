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
//! domain logs `(gap_f32, gap_f64)` at INFO — `gap_f64 < -1e-9` on the
//! residue names the instance relaxation-limited (more precision cannot
//! help); `gap_f64 >= 0` means the blocker was fp noise (and the domain
//! verifies).
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
use std::time::Instant;

use ndarray::{Array2, ArrayD};
use ny_core::GemmEngine;
use ny_tensor::{next_down_f32, BoundedTensor};
use tracing::info;

use crate::network::graph_crown_f64_tail::{
    alpha_tail_band, alpha_tail_depth, alpha_tail_enabled, alpha_tail_iters, alpha_tail_micro_band,
    f64_tail_band, f64_tail_enabled, f64_tail_verify, f64_tail_verify_refreshed, AlphaTailEval,
    F64TailOutcome,
};
use crate::GraphNetwork;

use super::batching::tighten_obj_lower_bounds;
use super::grouped_semantics::{disjunctive_domain_priority, disjunctive_domain_verified};
use super::shared::{build_child_input_owned, MultiObjInputDomain};

/// Fixed SPSA seed for every per-domain refresh: outcomes stay deterministic
/// under rayon interleaving (each domain's refresh depends only on its own
/// inputs). Soundness never depends on the draws.
const ALPHA_TAIL_SPSA_SEED: u64 = 0xA1FA_7A11;

/// SB-style coefficient clamp for the micro-BaB split-dim score (mirrors the
/// lane heuristic's `input_split_coeff_thresh` default, `config/defaults.rs`).
const MICRO_SPLIT_COEFF_THRESH: f32 = 1e-3;

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
) {
    let alpha_armed = alpha_tail_enabled();
    if !f64_tail_enabled() && !alpha_armed {
        return;
    }
    let band = if alpha_armed {
        alpha_tail_band()
    } else {
        f64_tail_band()
    };
    let candidates: Vec<(usize, f32)> = domains
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

    // Rayon-parallel over the (immutable) escalated domains, mirroring the
    // rebound fallback's RayonTaskGuard discipline (`shared_specs.rs`).
    use rayon::prelude::*;
    let domains_ro: &[MultiObjInputDomain] = domains;

    if !alpha_armed {
        // Landed frozen-alpha path (byte-identical to the f64 tail as shipped).
        let outcomes: Vec<(usize, f32, F64TailOutcome)> = candidates
            .par_iter()
            .map(|&(idx, gap_f32)| {
                let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                let domain = &domains_ro[idx];
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
                (idx, gap_f32, outcome)
            })
            .collect();

        let mut verified = 0usize;
        for (idx, gap_f32, outcome) in outcomes {
            let domain = &mut domains[idx];
            match outcome {
                F64TailOutcome::Verified { row_lowers } => {
                    merge_certified_rows(domain, &row_lowers, thresholds, clause_sizes);
                    verified += 1;
                    info!(
                        "[f64-tail] batch-seam verified depth={} gap_f32={:.6} (certified f64)",
                        domain.depth, gap_f32
                    );
                }
                F64TailOutcome::NotVerified { min_gap_f64 } => {
                    info!(
                        "[f64-tail] batch-seam not-verified depth={} gap_f32={:.6} gap_f64={:.3e}",
                        domain.depth, gap_f32, min_gap_f64
                    );
                }
                F64TailOutcome::Unsupported => {
                    info!(
                        "[f64-tail] batch-seam declined (unsupported) depth={} gap_f32={:.6}",
                        domain.depth, gap_f32
                    );
                }
            }
        }
        if verified > 0 {
            info!(
                "[f64-tail] batch-seam escalated={} verified={}",
                candidates.len(),
                verified
            );
        }
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
                info!(
                    "[f64-tail] batch-seam verified depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e} (refreshed certified f64)",
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
                        info!(
                            "[f64-tail] batch-seam verified depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e}{} (certified f64 micro-BaB)",
                            domain.depth,
                            gap_f32,
                            eval.gap_baseline,
                            eval.gap_refreshed,
                            micro_stats.as_ref().map(&micro_tag).unwrap_or_default()
                        );
                    } else {
                        info!(
                            "[f64-tail] batch-seam micro-bab declined (witness fragmentation) depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e}{}",
                            domain.depth,
                            gap_f32,
                            eval.gap_baseline,
                            eval.gap_refreshed,
                            micro_stats.as_ref().map(&micro_tag).unwrap_or_default()
                        );
                    }
                } else {
                    info!(
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
                info!(
                    "[f64-tail] batch-seam declined (unsupported) depth={} gap_f32={:.6}",
                    domain.depth, gap_f32
                );
            }
        }
    }
    if verified > 0 {
        info!(
            "[f64-tail] batch-seam escalated={} verified={} (alpha-tail)",
            candidates.len(),
            verified
        );
    }
}

/// Pop-side last chance (design §6.3 call site 2): one serial f64 pass for a
/// domain about to be dropped `unresolved_due_to_unsplittable`, regardless of
/// band. Returns `true` ONLY on a certified f64 `Verified` — the caller then
/// counts the domain verified instead of dropping it. Gate-off => `false`
/// with no work (byte-identical control flow).
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
) -> bool {
    let alpha_armed = alpha_tail_enabled();
    if !f64_tail_enabled() && !alpha_armed {
        return false;
    }
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
                info!(
                    "[f64-tail] last-chance verified depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e} (refreshed certified f64)",
                    domain.depth, gap_f32, eval.gap_baseline, eval.gap_refreshed
                );
                true
            }
            F64TailOutcome::NotVerified { .. } => {
                info!(
                    "[f64-tail] last-chance not-verified depth={} gap_f32={:.6} gap_f64={:.3e} gap_f64_refreshed={:.3e}",
                    domain.depth, gap_f32, eval.gap_baseline, eval.gap_refreshed
                );
                false
            }
            F64TailOutcome::Unsupported => {
                info!(
                    "[f64-tail] last-chance declined (unsupported) depth={} gap_f32={:.6}",
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
            info!(
                "[f64-tail] last-chance verified depth={} gap_f32={:.6} (certified f64)",
                domain.depth, gap_f32
            );
            true
        }
        F64TailOutcome::NotVerified { min_gap_f64 } => {
            info!(
                "[f64-tail] last-chance not-verified depth={} gap_f32={:.6} gap_f64={:.3e}",
                domain.depth, gap_f32, min_gap_f64
            );
            false
        }
        F64TailOutcome::Unsupported => {
            info!(
                "[f64-tail] last-chance declined (unsupported) depth={} gap_f32={:.6}",
                domain.depth, gap_f32
            );
            false
        }
    }
}

#[cfg(test)]
mod tests;
