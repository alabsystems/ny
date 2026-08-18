// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-batched domain processing for multi-objective graph BaB verification.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::{GemmEngine, GpuCrownBackward};
use rayon::prelude::*;
use tracing::debug;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::GraphNeuronConstraint;
use crate::beta_crown::domain::{
    GraphCrownContext, MultiObjDomainWithUnstable, MultiObjectiveGraphBabDomain,
    MultiObjectiveTargets, ObjectiveAggregation,
};
use crate::beta_crown::state::GraphDomainAlphaState;
use crate::faer_parallelism::RayonTaskGuard;
use crate::GraphNetwork;

use super::super::super::super::domain_results::MultiObjectiveGraphDomainResult;
use super::super::super::super::BetaCrownVerifier;
use super::super::bounded_shared_executor::MAX_BOUNDED_SHARED_HISTORY_CONSTRAINTS;
// `tighten_child_bounds_with_parent` now lives in `multi_objective::shared`
// (#bab-monotone-inherit) and is re-exported by `multi_objective::mod`; the call
// site below is unchanged.
use super::super::selective_root_alpha::{
    install_child_continuation_state, ChildContinuationStateProvenance,
};
use super::super::shared::{
    merge_pruned_cached_las, merge_pruned_objective_bounds,
    multi_objective_gpu_single_pass_enabled, prune_cached_las_for_targets,
    prune_verified_multi_objective_targets, violation_drop_is_certified,
};
use super::super::tighten_child_bounds_with_parent;
use super::batched_dense_specs::{BatchedMultiObjectiveAdapterError, MultiObjectiveChildEvalError};
use super::children::{
    collect_multi_objective_children, KfsbCertEffect, KfsbCertScope,
    MultiObjectiveChildCreationResult, KFSB_CERT_PARENT_ID_MAX_BYTES,
};
use super::kfsb_multi::kfsb_probe_enabled;

/// Chunk width for the GPU single-pass lane (#w5-bab-throughput): the batched
/// adapter folds the whole chunk into ONE wide GPU pass per β iteration (see
/// `gpu_beta_optimize_wide`), so a wider chunk amortizes the fixed per-pass Metal
/// dispatch + buffer-alloc + device-wait overhead across more domains. Chunking
/// still bounds the overrun past the deadline to ~one chunk of β-iteration passes
/// (the deadline is re-checked between chunks). 2026-07-11 cifar100 A/B: 8→255,
/// A 64-domain chunk was measured at roughly 25 seconds on the held-out
/// CIFAR-100 path, so it cannot preserve the verifier's five-second tail.
/// Deadline-scored calls hard-cap even an environment override to this width;
/// the cooperative backend deadline is the second line of defense.
const MO_GPU_SINGLE_PASS_CHUNK: usize = 8;

/// `NY_MO_GPU_CHUNK_DEADLINE=1` (exact) lets `NY_MO_GPU_CHUNK` raise the wide
/// batch width even under an authoritative deadline. Default OFF keeps every
/// scored run byte-identical; see the use site for the granularity trade.
fn deadline_chunk_override_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::wide_lane::MO_GPU_CHUNK_DEADLINE)
        .value
        .as_bool()
}
const MO_GPU_AUTHORITY_RESERVE: Duration = Duration::from_secs(5);

/// #adaptive-chunk: the last observed per-child cost of a dense-spec wave, in
/// microseconds. `0` means "never measured".
static OBSERVED_CHILD_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Record what one child actually cost in the wave that just completed. Called
/// unconditionally from the dense-spec adapter; one relaxed store.
pub(super) fn record_observed_child_cost(secs_per_child: f64) {
    if secs_per_child.is_finite() && secs_per_child > 0.0 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let micros = (secs_per_child * 1e6) as u64;
        OBSERVED_CHILD_MICROS.store(micros, std::sync::atomic::Ordering::Relaxed);
    }
}

/// How many children one UN-INTERRUPTIBLE chunk may hold without overrunning
/// `deadline`, given what a child measurably cost in the previous wave.
///
/// `MO_GPU_SINGLE_PASS_CHUNK = 8` is a deadline-GRANULARITY choice, not a
/// soundness one: a chunk cannot be interrupted, so one wider than the remaining
/// budget allows overruns it by a full pass. 8 happens to fit the CURRENTLY
/// measured ~3 s/child (8 x 3 s = 24 s inside a 100 s row) — but it is a constant
/// standing in for a quantity that is now measured, and it is wrong in BOTH
/// directions: dangerously wide if a child ever costs 30 s, and needlessly narrow
/// the moment per-child cost falls, which is precisely what the batched-BaB work
/// exists to achieve.
///
/// So derive it: fit one chunk inside HALF the remaining budget, leaving the rest
/// for the fold, the requeue and the next wave. `None` when there is no estimate
/// yet or no deadline, in which case the caller keeps its own constant. This only
/// ever moves WITHIN the caller's request — it can never widen past what the
/// operator asked for, so it cannot bypass the existing clamp.
fn adaptive_chunk_ceiling(now: Instant, deadline: Option<Instant>) -> Option<usize> {
    let deadline = deadline?;
    let micros = OBSERVED_CHILD_MICROS.load(std::sync::atomic::Ordering::Relaxed);
    if micros == 0 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let per_child_s = micros as f64 / 1e6;
    let remaining_s = deadline.saturating_duration_since(now).as_secs_f64();
    if per_child_s <= 0.0 || remaining_s <= 0.0 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let fits = ((remaining_s * 0.5) / per_child_s) as usize;
    Some(fits.max(1))
}

#[inline]
fn gpu_single_pass_backend_eligible(gpu: &dyn GpuCrownBackward, deadline: Option<Instant>) -> bool {
    gpu.provides_sound_gpu_crown() && (deadline.is_none() || gpu.honors_crown_backward_deadline())
}

#[inline]
fn child_uses_dense_spec_adapter(
    bounded_shared_lane: bool,
    gpu_single_pass_lane: bool,
    cpu_beta_optimization_applies: bool,
) -> bool {
    !bounded_shared_lane && (gpu_single_pass_lane || !cpu_beta_optimization_applies)
}

#[inline]
fn branch_selector_may_use_engine(bounded_shared_lane: bool) -> bool {
    !bounded_shared_lane
}

#[inline]
fn domain_state_may_branch(bounded_shared_lane: bool, constraint_count: usize) -> bool {
    !bounded_shared_lane || constraint_count < MAX_BOUNDED_SHARED_HISTORY_CONSTRAINTS
}

#[inline]
fn batched_child_refusal_has_cpu_fallback(
    bounded_shared_lane: bool,
    error: Option<MultiObjectiveChildEvalError>,
) -> bool {
    !bounded_shared_lane && error == Some(MultiObjectiveChildEvalError::PropagationFailure)
}

/// Return the root split committed by an exhaustive child cover.
///
/// A requested-depth cover appends one constraint per requested level.  The
/// last child constraint is therefore the deepest split, not the selector's
/// root decision.  Only accept an exact parent-history prefix and return the
/// first newly appended constraint so diagnostic receipts cannot silently
/// attribute a malformed child history to the parent.
#[inline]
fn committed_cover_root_constraint<'a>(
    parent_constraints: &[GraphNeuronConstraint],
    child_constraints: &'a [GraphNeuronConstraint],
) -> Option<&'a GraphNeuronConstraint> {
    let parent_len = parent_constraints.len();
    (child_constraints.get(..parent_len) == Some(parent_constraints))
        .then(|| child_constraints.get(parent_len))
        .flatten()
}

/// Require every surviving child to identify the same committed root split.
/// Active/inactive phases may differ, but a mixed-node cover is malformed and
/// must not produce an apparently precise diagnostic receipt.
fn common_committed_cover_root_constraint<'a>(
    parent_constraints: &[GraphNeuronConstraint],
    child_constraints: impl IntoIterator<Item = &'a [GraphNeuronConstraint]>,
) -> Option<&'a GraphNeuronConstraint> {
    let mut roots = child_constraints
        .into_iter()
        .map(|child| committed_cover_root_constraint(parent_constraints, child));
    let first = roots.next()??;
    roots
        .all(|root| {
            root.is_some_and(|root| {
                root.node_name == first.node_name && root.neuron_idx == first.neuron_idx
            })
        })
        .then_some(first)
}

#[inline]
fn child_uses_analytical_beta_optimizer(
    bounded_shared_lane: bool,
    beta_optimization_applies: bool,
) -> bool {
    !bounded_shared_lane && beta_optimization_applies
}

#[inline]
fn cut_pool_for_lane(
    bounded_shared_lane: bool,
    cut_pool: Option<&GraphCutPool>,
) -> Option<&GraphCutPool> {
    if bounded_shared_lane {
        None
    } else {
        cut_pool
    }
}

/// Run speculative advisory precompute only when it cannot consume the bounded
/// lane's authority before child verification begins.
///
/// Wave-kFSB simulates one objective row per candidate. The narrow CUDA beta
/// contract cannot service that historical one-row path without padding, and
/// the simulation constructs all candidate children before chunking. The
/// Complete-Clip preparation is likewise dead for this CPU-only facade because
/// its sole consumer requires a local broad GPU trait. Both are skipped.
#[inline]
fn maybe_run_unbounded_advisory<T>(
    bounded_shared_lane: bool,
    wave_enabled: bool,
    run: impl FnOnce() -> T,
) -> Option<T> {
    (!bounded_shared_lane && wave_enabled).then(run)
}

/// Publish a bounded-lane child's proof-side lower bound without trusting its
/// beta-derived upper endpoint.
///
/// The parent's interval encloses every child because a split child is a subset
/// of its parent. Keep that inherited (conservative) upper endpoint, and raise
/// only the lower endpoint. If the fresh lower crosses the inherited upper,
/// retain the complete inherited interval: it is ordered and sound, while
/// still preventing a regression below the parent's proven lower bound.
///
/// Non-finite fresh data is preserved so the existing publication validation
/// remains authoritative rather than masking a failed propagation.
fn inherit_bounded_child_bounds(
    inherited: &[(f32, f32)],
    fresh: Vec<(f32, f32)>,
) -> Vec<(f32, f32)> {
    if inherited.len() != fresh.len() {
        return fresh;
    }
    fresh
        .into_iter()
        .zip(inherited.iter())
        .map(
            |((fresh_lower, fresh_upper), &(parent_lower, parent_upper))| {
                if !fresh_lower.is_finite()
                    || !fresh_upper.is_finite()
                    || !parent_lower.is_finite()
                    || !parent_upper.is_finite()
                    || parent_lower > parent_upper
                {
                    return (fresh_lower, fresh_upper);
                }
                let lower = fresh_lower.max(parent_lower);
                if lower <= parent_upper {
                    (lower, parent_upper)
                } else {
                    (parent_lower, parent_upper)
                }
            },
        )
        .collect()
}

#[inline]
fn mo_gpu_chunk_start_allowed(now: Instant, deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|deadline| {
        now.checked_add(MO_GPU_AUTHORITY_RESERVE)
            .is_some_and(|reserved_until| reserved_until < deadline)
    })
}

#[inline]
fn mo_batch_chunk_start_allowed(
    now: Instant,
    deadline: Option<Instant>,
    gpu_single_pass_lane: bool,
) -> bool {
    deadline.is_none_or(|deadline| now < deadline)
        && (!gpu_single_pass_lane || mo_gpu_chunk_start_allowed(now, deadline))
}

/// Admit one fresh child-evaluation wave. A denied wave is represented in every
/// affected child slot and parent immediately so no caller can accidentally
/// reinterpret the missing work as a numerical propagation failure.
///
/// CPU fallback waves pass `gpu_single_pass_lane = false`: they remain
/// admissible throughout the five-second GPU reserve, but never at or after the
/// literal authoritative deadline.
fn admit_multi_objective_child_wave<T>(
    now: Instant,
    deadline: Option<Instant>,
    gpu_single_pass_lane: bool,
    positions: &[usize],
    parent_ids: &[usize],
    parents_with_deadline: &mut std::collections::HashSet<usize>,
    child_bounds: &mut [Option<Result<T, MultiObjectiveChildEvalError>>],
) -> bool {
    if mo_batch_chunk_start_allowed(now, deadline, gpu_single_pass_lane) {
        return true;
    }
    mark_multi_objective_positions_deadline_expired(
        positions,
        parent_ids,
        parents_with_deadline,
        child_bounds,
    );
    false
}

fn mark_multi_objective_positions_deadline_expired<T>(
    positions: &[usize],
    parent_ids: &[usize],
    parents_with_deadline: &mut std::collections::HashSet<usize>,
    child_bounds: &mut [Option<Result<T, MultiObjectiveChildEvalError>>],
) {
    for &pos in positions {
        parents_with_deadline.insert(parent_ids[pos]);
        child_bounds[pos] = Some(Err(MultiObjectiveChildEvalError::DeadlineExpired));
    }
}

/// Terminal-cause precedence for one parent. Deadline refusal dominates every
/// partial numerical/violation result because at least one child region remains
/// unevaluated and the verifier must return `Timeout`.
fn terminal_multi_objective_parent_result(
    deadline_expired: bool,
    propagation_failed: bool,
) -> Option<MultiObjectiveGraphDomainResult> {
    if deadline_expired {
        Some(MultiObjectiveGraphDomainResult::DeadlineExpired)
    } else if propagation_failed {
        Some(MultiObjectiveGraphDomainResult::PropagationFailure)
    } else {
        None
    }
}

type PreverifiedChildrenByParent =
    std::collections::HashMap<usize, Vec<(MultiObjectiveGraphBabDomain, bool)>>;

/// Receipt-specific work accounting for the KFSB reuse probe.
///
/// A typed receipt names exactly one newly verified row. Consequently both a
/// partial and a terminal receipt prune one specification row; a terminal
/// receipt additionally skips the whole child evaluation. Rows that were
/// already verified before the receipt are deliberately not attributed to
/// certificate reuse.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct KfsbCertAccounting {
    input_parents: usize,
    input_leaves: usize,
    partial_receipts: usize,
    complete_receipts: usize,
    parent_closes: usize,
    ordinary_preverified_leaves: usize,
    pruned_spec_rows: usize,
    invalid_parents: usize,
    invalid_leaves: usize,
    expired_parents: usize,
    expired_leaves: usize,
}

impl KfsbCertAccounting {
    fn observe_input_parent(&mut self, leaves: usize) {
        self.input_parents += 1;
        self.input_leaves += leaves;
    }

    fn observe_accepted_effect(&mut self, effect: &KfsbCertEffect) {
        match effect {
            KfsbCertEffect::None => {}
            KfsbCertEffect::RowVerified(_) => {
                self.partial_receipts += 1;
                self.pruned_spec_rows += 1;
            }
            KfsbCertEffect::ChildComplete(_) => {
                self.complete_receipts += 1;
                self.pruned_spec_rows += 1;
            }
            KfsbCertEffect::ParentComplete(_) => {
                self.parent_closes += 1;
                self.pruned_spec_rows += 1;
            }
        }
    }

    fn observe_invalid_parent(&mut self, leaves: usize) {
        self.invalid_parents += 1;
        self.invalid_leaves += leaves;
    }

    fn observe_ordinary_preverified(&mut self) {
        self.ordinary_preverified_leaves += 1;
    }

    fn observe_expired_parent(&mut self, leaves: usize) {
        self.expired_parents += 1;
        self.expired_leaves += leaves;
    }

    fn rejected_parents(self) -> usize {
        self.invalid_parents + self.expired_parents
    }

    fn rejected_leaves(self) -> usize {
        self.invalid_leaves + self.expired_leaves
    }
}

struct KfsbCertPartition {
    pending: Vec<MultiObjectiveChildCreationResult>,
    preverified: PreverifiedChildrenByParent,
    certified_parent_deadlines: std::collections::HashMap<usize, Instant>,
    expired_parents: std::collections::HashSet<usize>,
    invalid_parents: std::collections::HashSet<usize>,
    accounting: KfsbCertAccounting,
}

fn kfsb_cert_parent_publication_expired(
    now: Instant,
    parent_idx: usize,
    certified_parent_deadlines: &std::collections::HashMap<usize, Instant>,
) -> bool {
    certified_parent_deadlines
        .get(&parent_idx)
        .is_some_and(|deadline| now >= *deadline)
}

#[cfg(test)]
thread_local! {
    static KFSB_FINAL_PUBLICATION_NOW: std::cell::Cell<Option<Instant>> =
        const { std::cell::Cell::new(None) };
}

fn kfsb_final_publication_now() -> Instant {
    #[cfg(test)]
    if let Some(now) = KFSB_FINAL_PUBLICATION_NOW.with(std::cell::Cell::get) {
        return now;
    }
    Instant::now()
}

/// Deterministically exercise only the final receipt-publication boundary.
/// Earlier simulation, partition, and evaluator checks continue to use the
/// real clock, so tests cannot manufacture receipt authority.
#[cfg(test)]
pub(super) fn with_kfsb_final_publication_now<T>(now: Instant, f: impl FnOnce() -> T) -> T {
    struct Reset(Option<Instant>);
    impl Drop for Reset {
        fn drop(&mut self) {
            KFSB_FINAL_PUBLICATION_NOW.with(|clock| clock.set(self.0));
        }
    }

    let previous = KFSB_FINAL_PUBLICATION_NOW.with(|clock| clock.replace(Some(now)));
    let _reset = Reset(previous);
    f()
}

/// Revalidate the exact child state named by a typed KFSB receipt.
///
/// This is intentionally stronger than checking `all_verified`: every row's
/// cached mask is recomputed, the installed target lower must match bitwise,
/// and any target lA must retain the exact immutable parent allocation. The
/// latter preserves the full-spec Complete-Clip fast path when this child is a
/// parent in the next wave.
fn kfsb_cert_effect_matches_child(
    parent: &MultiObjectiveGraphBabDomain,
    parent_history_identity: &[u8],
    child: &MultiObjectiveGraphBabDomain,
    effect: &KfsbCertEffect,
    thresholds: &[f32],
) -> bool {
    let Some(receipt) = effect.receipt() else {
        return true;
    };
    if child.verify_upper()
        || child.aggregation() != ObjectiveAggregation::Disjunctive
        || child.objective_bounds().len() != thresholds.len()
        || child.verified().len() != thresholds.len()
        || child.cached_las().len() != thresholds.len()
        || receipt.row >= thresholds.len()
    {
        return false;
    }
    let parent_history = parent.history();
    let child_history = child.history();
    let parent_relu_len = parent_history.constraints.len();
    let Some(child_prefix) = child_history.constraints.get(..parent_relu_len) else {
        return false;
    };
    let Some(child_suffix) = child_history.constraints.get(parent_relu_len..) else {
        return false;
    };
    let history_and_scope_match = parent_history.is_pure_relu_at_zero()
        && child_history.is_pure_relu_at_zero()
        && receipt.parent_history_identity.as_ref() == parent_history_identity
        && child_prefix == parent_history.constraints.as_slice()
        && child.depth().checked_sub(parent.depth()) == Some(child_suffix.len())
        && child_history
            .split_count
            .checked_sub(parent_history.split_count)
            == Some(child_suffix.len())
        && match &receipt.scope {
            KfsbCertScope::ParentCover => true,
            KfsbCertScope::LiteralSide {
                node_name,
                neuron_idx,
                is_active,
            } => child_suffix.iter().any(|constraint| {
                constraint.node_name == *node_name
                    && constraint.neuron_idx == *neuron_idx
                    && constraint.is_active == *is_active
            }),
        };
    let mask_matches = child
        .objective_bounds()
        .iter()
        .zip(thresholds)
        .zip(child.verified())
        .all(|((&(lower, upper), &threshold), &verified)| {
            lower.is_finite()
                && upper.is_finite()
                && threshold.is_finite()
                && lower <= upper
                && verified
                    == crate::beta_crown::BetaCrownConfig::domain_is_verified_for_mode(
                        false, lower, upper, threshold,
                    )
        });
    let parent_complete_matches = match effect {
        KfsbCertEffect::ParentComplete(_) => {
            matches!(&receipt.scope, KfsbCertScope::ParentCover)
                && child_suffix.is_empty()
                && child.depth() == parent.depth()
                && child_history.split_count == parent_history.split_count
                && std::ptr::eq(child.input_bounds(), parent.input_bounds())
                && !parent.verify_upper()
                && parent.aggregation() == ObjectiveAggregation::Disjunctive
                && parent.objective_bounds().len() == child.objective_bounds().len()
                && parent.verified().len() == thresholds.len()
                && parent
                    .objective_bounds()
                    .iter()
                    .zip(thresholds)
                    .zip(parent.verified())
                    .all(|((&(lower, upper), &threshold), &verified)| {
                        lower.is_finite()
                            && upper.is_finite()
                            && threshold.is_finite()
                            && lower <= upper
                            && verified
                                == crate::beta_crown::BetaCrownConfig::domain_is_verified_for_mode(
                                    false, lower, upper, threshold,
                                )
                    })
                && parent
                    .verified()
                    .iter()
                    .enumerate()
                    .all(|(row, &verified)| verified == (row != receipt.row))
                && parent
                    .objective_bounds()
                    .iter()
                    .zip(child.objective_bounds())
                    .enumerate()
                    .all(
                        |(row, (&(parent_lower, parent_upper), &(child_lower, child_upper)))| {
                            if row == receipt.row {
                                child_lower >= parent_lower
                                    && child_upper.to_bits() == parent_upper.to_bits()
                            } else {
                                child_lower.to_bits() == parent_lower.to_bits()
                                    && child_upper.to_bits() == parent_upper.to_bits()
                            }
                        },
                    )
        }
        KfsbCertEffect::None
        | KfsbCertEffect::RowVerified(_)
        | KfsbCertEffect::ChildComplete(_) => true,
    };
    let receipt_cache_matches = match effect {
        KfsbCertEffect::RowVerified(_) | KfsbCertEffect::ChildComplete(_) => {
            match (
                parent.cached_las().get(receipt.row),
                child.cached_las().get(receipt.row),
            ) {
                (Some(Some(parent_cache)), Some(Some(child_cache))) => {
                    Arc::ptr_eq(parent_cache, child_cache)
                }
                (Some(None), Some(None)) => true,
                _ => false,
            }
        }
        // A parent-wide close deliberately publishes a lightweight terminal
        // shell with no continuation caches.
        KfsbCertEffect::ParentComplete(_) => child.cached_las().iter().all(Option::is_none),
        KfsbCertEffect::None => true,
    };
    history_and_scope_match
        && mask_matches
        && parent_complete_matches
        && receipt_cache_matches
        && child.verified()[receipt.row]
        && child.objective_bounds()[receipt.row].0.to_bits() == receipt.lower_bits
        && match effect {
            KfsbCertEffect::None => true,
            KfsbCertEffect::RowVerified(_) => !child.all_verified() && !child_suffix.is_empty(),
            KfsbCertEffect::ChildComplete(_) => child.all_verified() && !child_suffix.is_empty(),
            KfsbCertEffect::ParentComplete(_) => child.all_verified(),
        }
}

/// Validate and partition typed KFSB certificate effects atomically by parent.
///
/// Complete children bypass the expensive evaluator. Partial row receipts stay
/// pending so ordinary target pruning bounds every remaining row. Any malformed
/// or expired receipt poisons the whole parent before a child is moved, keeping
/// its exhaustive cover atomic.
fn partition_kfsb_certified_children(
    now: Instant,
    thresholds: &[f32],
    parent_lookup: &std::collections::HashMap<usize, &MultiObjectiveGraphBabDomain>,
    parents: Vec<MultiObjectiveChildCreationResult>,
) -> KfsbCertPartition {
    let mut pending_parents = Vec::with_capacity(parents.len());
    let mut preverified = PreverifiedChildrenByParent::new();
    let mut certified_parent_deadlines = std::collections::HashMap::new();
    let mut expired_parents = std::collections::HashSet::new();
    let mut invalid_parents = std::collections::HashSet::new();
    let mut accounting = KfsbCertAccounting::default();
    for (parent_idx, children) in parents {
        let parent_leaves = children.len();
        accounting.observe_input_parent(parent_leaves);
        // A parent-wide close replaces the complete split cover; it may never
        // coexist with another leaf or another close for the same parent.
        // Enforce this before moving any child so malformed groups fail
        // atomically through the ordinary parent-failure path.
        let parent_close_count = children
            .iter()
            .filter(|(_, _, _, effect)| matches!(effect, KfsbCertEffect::ParentComplete(_)))
            .count();
        if parent_close_count > 0 && (parent_close_count != 1 || children.len() != 1) {
            accounting.observe_invalid_parent(parent_leaves);
            invalid_parents.insert(parent_idx);
            continue;
        }
        let mut parent_deadline = None;
        let mut invalid = false;
        let has_receipt = children
            .iter()
            .any(|(_, _, _, effect)| effect.receipt().is_some());
        let parent = has_receipt
            .then(|| parent_lookup.get(&parent_idx).copied())
            .flatten();
        let parent_history_identity = parent.and_then(|parent| {
            parent
                .history()
                .exact_provenance_identity()
                .filter(|identity| identity.len() <= KFSB_CERT_PARENT_ID_MAX_BYTES)
        });
        if has_receipt && (parent.is_none() || parent_history_identity.is_none()) {
            accounting.observe_invalid_parent(parent_leaves);
            invalid_parents.insert(parent_idx);
            continue;
        }
        for (child_parent_idx, child, _, effect) in &children {
            if has_receipt && *child_parent_idx != parent_idx {
                invalid = true;
                break;
            }
            let Some(receipt) = effect.receipt() else {
                continue;
            };
            if !kfsb_cert_effect_matches_child(
                parent.expect("receipt parent prevalidated"),
                parent_history_identity
                    .as_deref()
                    .expect("receipt parent identity prevalidated"),
                child,
                effect,
                thresholds,
            ) || parent_deadline.is_some_and(|deadline| deadline != receipt.authority_deadline)
            {
                invalid = true;
                break;
            }
            parent_deadline = Some(receipt.authority_deadline);
        }
        if invalid {
            accounting.observe_invalid_parent(parent_leaves);
            invalid_parents.insert(parent_idx);
            continue;
        }
        if parent_deadline.is_some_and(|deadline| now >= deadline) {
            accounting.observe_expired_parent(parent_leaves);
            expired_parents.insert(parent_idx);
            continue;
        }
        if let Some(deadline) = parent_deadline {
            certified_parent_deadlines.insert(parent_idx, deadline);
        }

        let mut pending_children = Vec::with_capacity(children.len());
        for (child_parent_idx, child, is_active, effect) in children {
            accounting.observe_accepted_effect(&effect);
            match effect {
                KfsbCertEffect::ChildComplete(_) | KfsbCertEffect::ParentComplete(_) => {
                    preverified
                        .entry(parent_idx)
                        .or_default()
                        .push((child, true));
                }
                KfsbCertEffect::None if child.all_verified() => {
                    // The verification mask is authoritative independent of
                    // KFSB provenance. Preserve the historical empty-target
                    // bypass, but do not attach receipt deadline authority or
                    // count this leaf as certificate reuse.
                    accounting.observe_ordinary_preverified();
                    preverified
                        .entry(parent_idx)
                        .or_default()
                        .push((child, true));
                }
                KfsbCertEffect::None | KfsbCertEffect::RowVerified(_) => {
                    pending_children.push((child_parent_idx, child, is_active, effect));
                }
            }
        }
        pending_parents.push((parent_idx, pending_children));
    }
    KfsbCertPartition {
        pending: pending_parents,
        preverified,
        certified_parent_deadlines,
        expired_parents,
        invalid_parents,
        accounting,
    }
}

/// Validate every row consumed by the batched disjunctive executor before its
/// cached verification mask can authorize an `AlreadyVerified` result.
fn multi_objective_batch_layout_is_valid(
    domain: &MultiObjectiveGraphBabDomain,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
) -> bool {
    let rows = objectives.len();
    rows > 0
        && rows == thresholds.len()
        && thresholds.iter().all(|threshold| threshold.is_finite())
        && objectives.first().is_some_and(|first| {
            !first.is_empty()
                && objectives
                    .iter()
                    .all(|row| row.len() == first.len() && row.iter().all(|v| v.is_finite()))
        })
        && domain.aggregation() == ObjectiveAggregation::Disjunctive
        && !domain.verify_upper()
        && domain.objective_bounds().len() == rows
        && domain.verified().len() == rows
        && domain.cached_las().len() == rows
        && domain
            .per_disjunct_alphas()
            .is_none_or(|alphas| alphas.len() == rows)
        && domain
            .objective_bounds()
            .iter()
            .all(|&(lower, upper)| lower.is_finite() && upper.is_finite() && lower <= upper)
}

/// Final publication barrier for the bounded shared executor.
///
/// The outer loop deliberately lets a cleanly drained frontier win over a
/// deadline observed on its next iteration. Consequently, a bounded wave must
/// not return a completed last batch after its own authority expired during
/// host-side objective extraction, child assembly, or result ordering. Discard
/// the whole wave on a failed final poll so `apply_batched_results` records a
/// typed deadline and the verifier returns `Timeout`.
fn publish_bounded_wave_results(
    bounded_shared_lane: bool,
    engine: &dyn GemmEngine,
    mut results: Vec<MultiObjectiveGraphDomainResult>,
) -> Vec<MultiObjectiveGraphDomainResult> {
    if bounded_shared_lane && engine.poll_crown_backward_deadline().is_err() {
        for result in &mut results {
            *result = MultiObjectiveGraphDomainResult::DeadlineExpired;
        }
    }
    results
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiObjectiveBranchCreationError {
    PropagationFailure(usize),
    DeadlineExpired(usize),
}

fn classify_branch_creation_error(
    parent_idx: usize,
    error: &ny_core::NyError,
) -> MultiObjectiveBranchCreationError {
    if error.is_deadline_exceeded() {
        MultiObjectiveBranchCreationError::DeadlineExpired(parent_idx)
    } else {
        MultiObjectiveBranchCreationError::PropagationFailure(parent_idx)
    }
}

/// #metaroom-chain-wide: env override for the GPU single-pass chunk width
/// (`NY_MO_GPU_CHUNK=<n>`, default [`MO_GPU_SINGLE_PASS_CHUNK`]). Under an
/// authoritative deadline, larger overrides are capped back to the safe
/// scored width. An explicitly present malformed/zero/out-of-range value returns
/// `None`, which disables this experimental lane for the batch and routes every
/// child through the existing per-child fallback. On the WIDE batched lane the
/// whole chunk is ONE GPU pass per β iteration, so a wider chunk amortizes the
/// fixed per-pass cost across more domains (metaroom's 6cnn conv chains: 8 →
/// 32/64 packs 40 → 160/320 wide rows, still small for the device).
fn parse_mo_gpu_single_pass_chunk(raw: Option<&str>) -> Option<usize> {
    let Some(raw) = raw else {
        return Some(MO_GPU_SINGLE_PASS_CHUNK);
    };
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    raw.parse::<usize>().ok().filter(|&n| n > 0)
}

fn mo_gpu_single_pass_chunk() -> Option<usize> {
    match std::env::var("NY_MO_GPU_CHUNK") {
        Ok(raw) => parse_mo_gpu_single_pass_chunk(Some(&raw)),
        Err(std::env::VarError::NotPresent) => parse_mo_gpu_single_pass_chunk(None),
        Err(std::env::VarError::NotUnicode(_)) => None,
    }
}

impl BetaCrownVerifier {
    /// Process a batch of multi-objective domains with GPU-batched CROWN computation.
    ///
    /// Similar to `process_graph_domains_batched_gpu` but handles multiple objectives.
    /// This batches the CROWN computation across all child domains to improve GPU utilization.
    ///
    /// Part of #3813: `cut_pool` is a read-only view of the current cutting
    /// planes. The batched path applies existing cuts during CROWN backward
    /// propagation but does not generate or merge new cuts — that happens in
    /// the outer BaB loop after batch results return.
    // Justification: batched multi-objective processing needs graph, domains,
    // relu nodes, objective/threshold slices, engine, and the read-only cut
    // pool together; splitting this signature further would just mirror a
    // temporary context struct without reducing the actual call surface.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn process_graph_domains_batched_gpu_multi_objective(
        &self,
        bab_round: usize,
        graph: &GraphNetwork,
        domains: &[&MultiObjectiveGraphBabDomain],
        relu_nodes: &[String],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: &dyn GemmEngine,
        cut_pool: Option<&GraphCutPool>,
        selective_root_alpha_candidate: Option<&GraphDomainAlphaState>,
    ) -> Vec<MultiObjectiveGraphDomainResult> {
        // #phase-telemetry (dark, NY_PHASE_TELEMETRY=1, print-only): mark the
        // FIRST domain batch entering the resnet batched BaB lane, once per
        // process — the boundary where the root pipeline hands over to BaB
        // domain processing. Gate-off is a cached-bool load; the `Once` fires
        // only when the gate is on, so unset stays byte-identical.
        if crate::phase_telemetry::phase_telemetry_enabled() {
            static FIRST_BATCH: std::sync::Once = std::sync::Once::new();
            FIRST_BATCH
                .call_once(|| crate::phase_telemetry::phase_marker("bab-first-domain-batch start"));
        }
        if domains.is_empty() {
            return Vec::new();
        }
        let bounded_shared_lane = engine.forbids_unbounded_cpu_fallback();
        // Defense in depth: admission requires an empty pool and the outer
        // loop disables generation. Keep this executor incapable of observing
        // a pool even if a future caller passes one accidentally.
        let cut_pool = cut_pool_for_lane(bounded_shared_lane, cut_pool);

        // Pre-filter: separate already-verified, violation, and to-process domains
        let mut quick_results: std::collections::HashMap<usize, MultiObjectiveGraphDomainResult> =
            std::collections::HashMap::new();
        let mut domains_to_process: Vec<(usize, &MultiObjectiveGraphBabDomain)> = Vec::new();

        for (idx, domain) in domains.iter().enumerate() {
            if !multi_objective_batch_layout_is_valid(domain, objectives, thresholds) {
                quick_results.insert(idx, MultiObjectiveGraphDomainResult::PropagationFailure);
                continue;
            }
            // Quick verification check
            if domain.all_verified() {
                quick_results.insert(idx, MultiObjectiveGraphDomainResult::AlreadyVerified);
                continue;
            }

            // Quick violation check.
            //
            // #violdrop: only the ROOT may be abandoned on a `upper < threshold`
            // reading — a BaB child's interval is β-derived and its upper end
            // carries no certificate for the child's sub-region
            // (`violation_drop_is_certified`). Measured on vit_2023
            // ibp_3_3_8_3005: with only the child-assembly and pop-time sites
            // fixed, the two depth-1 children were still discarded HERE, so the
            // queue still emptied at `explored=3 queue=0 max_depth=1` after
            // 1.73 s of a 90.25 s grant — even though the single root split had
            // lifted the worst objective from −8.82 to −0.094 with 218 unstable
            // neurons still available to split.
            if domain.any_violated(thresholds, false) {
                super::super::shared::violdrop_site_probe("batched_entry_prefilter", domain.depth);
                if violation_drop_is_certified(domain.depth) {
                    quick_results.insert(idx, MultiObjectiveGraphDomainResult::Violation);
                    continue;
                }
            }
            if !domain_state_may_branch(bounded_shared_lane, domain.history().constraints.len()) {
                tracing::warn!(
                    depth = domain.depth(),
                    constraints = domain.history().constraints.len(),
                    "bounded shared executor retained a domain at its state-depth cap"
                );
                quick_results.insert(idx, MultiObjectiveGraphDomainResult::PropagationFailure);
                continue;
            }

            domains_to_process.push((idx, domain));
        }

        if domains_to_process.is_empty() {
            let results = (0..domains.len())
                .map(|idx| {
                    quick_results.remove(&idx).unwrap_or_else(|| {
                        tracing::warn!(
                            "process_graph_domains_batched_gpu_multi_objective: missing quick_result for idx {} (#1993)",
                            idx
                        );
                        MultiObjectiveGraphDomainResult::PropagationFailure
                    })
                })
                .collect();
            return publish_bounded_wave_results(bounded_shared_lane, engine, results);
        }

        // Unstable-neuron discovery is the first fresh CPU wave for unresolved
        // parents. Preserve already-completed quick siblings above, but do not
        // start the scan at or after the literal authoritative deadline.
        if !mo_batch_chunk_start_allowed(Instant::now(), self.effective_graph_bab_deadline(), false)
        {
            for (idx, _) in &domains_to_process {
                quick_results.insert(*idx, MultiObjectiveGraphDomainResult::DeadlineExpired);
            }
            return (0..domains.len())
                .map(|idx| {
                    quick_results.remove(&idx).unwrap_or_else(|| {
                        tracing::warn!(
                            "process_graph_domains_batched_gpu_multi_objective: missing deadline result for idx {} before unstable scan",
                            idx
                        );
                        MultiObjectiveGraphDomainResult::PropagationFailure
                    })
                })
                .collect();
        }

        // #phase-telemetry (dark, NY_PHASE_TELEMETRY=1, print-only): the
        // executor's PRE-wave stages are unlit — the phase timeline goes quiet
        // after `bab-first-domain-batch` and the next line is `mo-wave-stage`.
        // Measured on cifar100 idx_8600: only two waves complete inside a ~42 s
        // BaB window, so ~30 s (~70%) executes outside every timer. Each stage
        // below is timed separately; gate-off is one cached-bool load per stage
        // and no allocation.
        let __t_unstable_scan =
            crate::phase_telemetry::phase_telemetry_enabled().then(Instant::now);
        // Find unstable neurons. The bounded lane serializes its K<=8 wave and
        // uses the allocation-free, deadline-polled scanner; legacy lanes keep
        // their parallel helper unchanged.
        let unstable_per_domain: Vec<(usize, Vec<(String, usize)>)> = if bounded_shared_lane {
            let mut discovered = Vec::with_capacity(domains_to_process.len());
            for (idx, domain) in &domains_to_process {
                match self.find_unstable_graph_neurons_multi_bounded(
                    graph,
                    domain,
                    relu_nodes,
                    self.effective_graph_bab_deadline(),
                ) {
                    Ok(unstable) => discovered.push((*idx, unstable)),
                    Err(ref error) if error.is_deadline_exceeded() => {
                        quick_results
                            .insert(*idx, MultiObjectiveGraphDomainResult::DeadlineExpired);
                    }
                    Err(error) => {
                        tracing::warn!("bounded unstable discovery failed for idx {idx}: {error}");
                        quick_results
                            .insert(*idx, MultiObjectiveGraphDomainResult::PropagationFailure);
                    }
                }
            }
            discovered
        } else {
            domains_to_process
                .par_iter()
                .map(|(idx, domain)| {
                    let unstable =
                        self.find_unstable_graph_neurons_multi(graph, domain, relu_nodes);
                    (*idx, unstable)
                })
                .collect()
        };
        if let Some(t) = __t_unstable_scan {
            eprintln!(
                "[phase] mo-unstable-scan domains={} total_unstable={} secs={:.2}",
                domains_to_process.len(),
                unstable_per_domain
                    .iter()
                    .map(|(_, unstable)| unstable.len())
                    .sum::<usize>(),
                t.elapsed().as_secs_f64(),
            );
        }

        // Separate domains with/without unstable neurons
        let mut domains_with_unstable: Vec<MultiObjDomainWithUnstable<'_>> = Vec::new();

        // O(1) index from domain idx → domain ref, replacing a per-iteration
        // linear `.find()` over `domains_to_process` (was O(D²) for batch size D,
        // up to thousands of domains). `idx` is the unique `.enumerate()` index
        // assigned when `domains_to_process` was built, so each key maps to
        // exactly one domain — identical to the first-match `.find()` semantics.
        let domain_by_idx: std::collections::HashMap<usize, &MultiObjectiveGraphBabDomain> =
            domains_to_process.iter().map(|(i, d)| (*i, *d)).collect();

        for (idx, unstable) in unstable_per_domain {
            let Some(domain) = domain_by_idx.get(&idx).copied() else {
                tracing::warn!(
                    "process_graph_domains_batched_gpu_multi_objective: missing domain at idx {} while resolving unstable set (#1993)",
                    idx
                );
                quick_results.insert(idx, MultiObjectiveGraphDomainResult::PropagationFailure);
                continue;
            };

            if unstable.is_empty() {
                if !mo_batch_chunk_start_allowed(
                    Instant::now(),
                    self.effective_graph_bab_deadline(),
                    false,
                ) {
                    quick_results.insert(idx, MultiObjectiveGraphDomainResult::DeadlineExpired);
                    continue;
                }
                // No unstable neurons - compute final bounds
                let context = GraphCrownContext::new_with_node_bounds_map(
                    &domain.history,
                    cut_pool, // Part of #3813: apply existing cuts
                    Some(&domain.node_bounds),
                    Some(engine),
                )
                .with_alpha(&domain.alpha_state);
                match self.propagate_crown_with_graph_constraints(
                    graph,
                    domain.input_bounds.as_ref(),
                    &context,
                    None,
                    None, // Multi-objective: compute full output bounds
                ) {
                    Ok((output, _node_cache)) => {
                        if bounded_shared_lane {
                            match engine.poll_crown_backward_deadline() {
                                Ok(()) => {}
                                Err(ref error) if error.is_deadline_exceeded() => {
                                    quick_results.insert(
                                        idx,
                                        MultiObjectiveGraphDomainResult::DeadlineExpired,
                                    );
                                    continue;
                                }
                                Err(error) => {
                                    tracing::warn!(
                                        "bounded NoUnstable publication poll failed for idx {idx}: {error}"
                                    );
                                    quick_results.insert(
                                        idx,
                                        MultiObjectiveGraphDomainResult::PropagationFailure,
                                    );
                                    continue;
                                }
                            }
                        }
                        match Self::objective_bounds_multi(&output, objectives) {
                            Ok(new_bounds) => {
                                // Defense-in-depth: reject length mismatch instead of
                                // silent .zip() truncation (#3383).
                                if new_bounds.len() != thresholds.len() {
                                    debug!(
                                        "batched multi-objective NoUnstable: new_bounds/thresholds length mismatch ({} vs {}) (#3383)",
                                        new_bounds.len(),
                                        thresholds.len()
                                    );
                                    quick_results.insert(
                                        idx,
                                        MultiObjectiveGraphDomainResult::PropagationFailure,
                                    );
                                    continue;
                                }
                                let all_verified = new_bounds.iter().zip(thresholds.iter()).all(
                                    |(&(lower, upper), &threshold)| {
                                        crate::BetaCrownConfig::domain_is_verified_for_mode(
                                            false, lower, upper, threshold,
                                        )
                                    },
                                );
                                // #1866: Compute any_violated so the BaB loop can detect
                                // conclusive violations in fully-constrained domains.
                                //
                                // #violdrop: on a SPLIT leaf (`depth > 0`) this
                                // reading is not a certificate — the bounds above
                                // came from a β-augmented backward over the split
                                // history, which certifies the LOWER end only. A
                                // false `any_violated` here reports the run as
                                // "Some domains conclusively violated the property"
                                // instead of the TRUTHFUL "No unstable ReLU/Sign
                                // neurons left", hiding the real blocker. Both
                                // outcomes are `Unknown`, so this only changes the
                                // reason string — but that string is what the next
                                // investigation reads.
                                let any_violated = violation_drop_is_certified(domain.depth)
                                    && new_bounds.iter().zip(thresholds.iter()).any(
                                        |(&(lower, upper), &threshold)| {
                                            crate::BetaCrownConfig::domain_is_violation_for_mode(
                                                false, lower, upper, threshold,
                                            )
                                        },
                                    );
                                if bounded_shared_lane {
                                    match engine.poll_crown_backward_deadline() {
                                        Ok(()) => {}
                                        Err(ref error) if error.is_deadline_exceeded() => {
                                            quick_results.insert(
                                                idx,
                                                MultiObjectiveGraphDomainResult::DeadlineExpired,
                                            );
                                            continue;
                                        }
                                        Err(error) => {
                                            tracing::warn!(
                                                "bounded NoUnstable final publication poll failed for idx {idx}: {error}"
                                            );
                                            quick_results.insert(
                                                idx,
                                                MultiObjectiveGraphDomainResult::PropagationFailure,
                                            );
                                            continue;
                                        }
                                    }
                                }
                                quick_results.insert(
                                    idx,
                                    MultiObjectiveGraphDomainResult::NoUnstable {
                                        all_verified,
                                        any_violated,
                                    },
                                );
                            }
                            Err(e) => {
                                debug!(error = %e, "Multi-objective bounds extraction failed — returning PropagationFailure (#1978)");
                                quick_results.insert(
                                    idx,
                                    MultiObjectiveGraphDomainResult::PropagationFailure,
                                );
                            }
                        }
                    }
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: Infeasible domain = empty = trivially verified.
                        debug!(error = %e, "Multi-objective NoUnstable infeasible (empty)");
                        quick_results.insert(idx, MultiObjectiveGraphDomainResult::AlreadyVerified);
                    }
                    Err(ref e) if e.is_deadline_exceeded() => {
                        quick_results.insert(idx, MultiObjectiveGraphDomainResult::DeadlineExpired);
                    }
                    Err(e) => {
                        debug!(error = %e, "Multi-objective NoUnstable CROWN propagation failed — returning PropagationFailure (#1978)");
                        quick_results
                            .insert(idx, MultiObjectiveGraphDomainResult::PropagationFailure);
                    }
                }
            } else {
                domains_with_unstable.push((idx, domain, unstable));
            }
        }

        if domains_with_unstable.is_empty() {
            let results = (0..domains.len())
                .map(|idx| {
                    quick_results.remove(&idx).unwrap_or_else(|| {
                        tracing::warn!(
                            "process_graph_domains_batched_gpu_multi_objective: missing result for idx {} after unstable scan (#1993)",
                            idx
                        );
                        MultiObjectiveGraphDomainResult::PropagationFailure
                    })
                })
                .collect();
            return publish_bounded_wave_results(bounded_shared_lane, engine, results);
        }

        // kFSB scoring and the per-domain selector are a fresh CPU branch wave.
        // Do not enter either once the literal authoritative deadline has
        // expired; every still-open parent remains explicitly covered by a
        // typed terminal result.
        if !mo_batch_chunk_start_allowed(Instant::now(), self.effective_graph_bab_deadline(), false)
        {
            for (idx, _, _) in &domains_with_unstable {
                quick_results.insert(*idx, MultiObjectiveGraphDomainResult::DeadlineExpired);
            }
            return (0..domains.len())
                .map(|idx| {
                    quick_results.remove(&idx).unwrap_or_else(|| {
                        tracing::warn!(
                            "process_graph_domains_batched_gpu_multi_objective: missing deadline result for idx {} before branch wave",
                            idx
                        );
                        MultiObjectiveGraphDomainResult::PropagationFailure
                    })
                })
                .collect();
        }

        // #kfsb-multi (dark, NY_MO_KFSB=1): wave-batched kFSB branch selection.
        // Pre-scores + SIMULATES both children of the top-k∪backup candidates
        // for the whole wave in chunked dense-spec backward calls, picks per
        // domain by the configured reduce op on that domain's worst-straggler
        // row, and COMMITS the winner's already-built children (no rebuild).
        // By default only the split choice is advisory. The independent typed,
        // default-off certificate-reuse policy (exact env override
        // `NY_MO_KFSB_CERT_REUSE=1`) may additionally publish a
        // strictly-authorized scalar lower certificate; incomplete child
        // covers or per-domain misses fall back to `select_graph_branch_multi`
        // below. Both gates off ⇒ empty map ⇒ byte-identical to today.
        let kfsb_precomputed: std::sync::Mutex<
            std::collections::HashMap<usize, super::kfsb_multi::KfsbMultiChildren>,
        > = std::sync::Mutex::new(
            maybe_run_unbounded_advisory(
                bounded_shared_lane,
                self.kfsb_multi_wave_enabled_at_round(bab_round),
                || {
                    // #phase-telemetry: inside the closure so a skipped wave
                    // (bounded lane, or the gate off at this round) prints
                    // nothing at all — a zero-duration line would be
                    // indistinguishable from an instant wave.
                    let __t_kfsb_wave =
                        crate::phase_telemetry::phase_telemetry_enabled().then(Instant::now);
                    let committed = self.select_graph_branch_kfsb_multi_batched(
                        bab_round,
                        graph,
                        &domains_with_unstable,
                        relu_nodes,
                        objectives,
                        thresholds,
                        engine,
                    );
                    if let Some(t) = __t_kfsb_wave {
                        eprintln!(
                            "[phase] mo-kfsb-wave domains={} committed={} secs={:.2}",
                            domains_with_unstable.len(),
                            committed.len(),
                            t.elapsed().as_secs_f64(),
                        );
                    }
                    committed
                },
            )
            .unwrap_or_default(),
        );

        // Branch-selection and child-construction failures preserve whether
        // the cause was numerical or the authoritative deadline (#2143).
        let create_children = |(idx, domain, unstable): &MultiObjDomainWithUnstable<'_>| {
            // #kfsb-multi: committed winner children, if the wave selector
            // resolved this domain (children_info shape matches the
            // advisory path: 0..=2^d entries, infeasible leaves absent).
            if let Some(pre) = kfsb_precomputed.lock().ok().and_then(|mut m| m.remove(idx)) {
                let children_info: Vec<_> = pre
                    .into_iter()
                    .map(|(child, is_active, cert_effect)| (*idx, child, is_active, cert_effect))
                    .collect();
                return Ok((*idx, children_info));
            }
            if !mo_batch_chunk_start_allowed(
                Instant::now(),
                self.effective_graph_bab_deadline(),
                false,
            ) {
                return Err(MultiObjectiveBranchCreationError::DeadlineExpired(*idx));
            }
            let branch_selection = if branch_selector_may_use_engine(bounded_shared_lane) {
                self.select_graph_branch_multi(
                    graph,
                    domain,
                    unstable,
                    objectives,
                    thresholds,
                    Some(engine),
                )
            } else {
                self.select_graph_branch_multi_bounded_intercept(
                    graph,
                    domain,
                    unstable,
                    self.effective_graph_bab_deadline(),
                )
            };
            let (node_name, neuron_idx, score) = match branch_selection {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        "select_graph_branch_multi failed for idx {}: {e} (#1915)",
                        idx
                    );
                    return Err(classify_branch_creation_error(*idx, &e));
                }
            };

            let mut children_info = Vec::with_capacity(2);

            // Active child
            if !mo_batch_chunk_start_allowed(
                Instant::now(),
                self.effective_graph_bab_deadline(),
                false,
            ) {
                return Err(MultiObjectiveBranchCreationError::DeadlineExpired(*idx));
            }
            let active_constraint = GraphNeuronConstraint {
                node_name: node_name.clone(),
                neuron_idx,
                is_active: true,
                score,
            };
            let active_child = if bounded_shared_lane {
                domain.with_constraint_without_optional_warm_starts(
                    graph,
                    active_constraint,
                    false,
                    thresholds,
                )
            } else {
                domain.with_constraint(graph, active_constraint, false, thresholds)
            };
            match active_child {
                Ok(Some(child)) => children_info.push((*idx, child, true, KfsbCertEffect::None)),
                Ok(None) => {}
                Err(ref e) if e.is_infeasible_domain() => {
                    // #2926: Infeasible constraint = empty child, skip.
                }
                Err(e) => {
                    tracing::warn!("with_constraint (active) failed for idx {}: {e}", idx);
                    return Err(classify_branch_creation_error(*idx, &e));
                }
            }

            // Inactive child
            if !mo_batch_chunk_start_allowed(
                Instant::now(),
                self.effective_graph_bab_deadline(),
                false,
            ) {
                return Err(MultiObjectiveBranchCreationError::DeadlineExpired(*idx));
            }
            let inactive_constraint = GraphNeuronConstraint {
                node_name,
                neuron_idx,
                is_active: false,
                score,
            };
            let inactive_child = if bounded_shared_lane {
                domain.with_constraint_without_optional_warm_starts(
                    graph,
                    inactive_constraint,
                    false,
                    thresholds,
                )
            } else {
                domain.with_constraint(graph, inactive_constraint, false, thresholds)
            };
            match inactive_child {
                Ok(Some(child)) => children_info.push((*idx, child, false, KfsbCertEffect::None)),
                Ok(None) => {}
                Err(ref e) if e.is_infeasible_domain() => {
                    // #2926: Infeasible constraint = empty child, skip.
                }
                Err(e) => {
                    tracing::warn!("with_constraint (inactive) failed for idx {}: {e}", idx);
                    return Err(classify_branch_creation_error(*idx, &e));
                }
            }

            Ok((*idx, children_info))
        };
        // Optional alpha maps are rebuilt during child construction and have
        // no inner allocation seam. Serialize this small K<=8 wave for the
        // bounded facade; legacy lanes retain their parallel schedule.
        let __t_create_children =
            crate::phase_telemetry::phase_telemetry_enabled().then(Instant::now);
        let child_creation_results: Vec<_> = if bounded_shared_lane {
            domains_with_unstable.iter().map(create_children).collect()
        } else {
            domains_with_unstable
                .par_iter()
                .map(create_children)
                .collect()
        };
        if let Some(t) = __t_create_children {
            // `n` counts CHILDREN built (committed kFSB winners included), not
            // parents: the parent count is already on the two lines above and
            // every later stage is priced per child.
            eprintln!(
                "[phase] mo-create-children n={} secs={:.2}",
                child_creation_results
                    .iter()
                    .filter_map(|result| result.as_ref().ok())
                    .map(|(_, children)| children.len())
                    .sum::<usize>(),
                t.elapsed().as_secs_f64(),
            );
        }

        // Handle branch selection failures explicitly (#2143).
        let mut branch_selection_failures: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let successful_results: Vec<_> = child_creation_results
            .into_iter()
            .filter_map(|result| match result {
                Ok(v) => Some(v),
                Err(MultiObjectiveBranchCreationError::PropagationFailure(failed_idx)) => {
                    branch_selection_failures.insert(failed_idx);
                    quick_results.insert(
                        failed_idx,
                        MultiObjectiveGraphDomainResult::PropagationFailure,
                    );
                    None
                }
                Err(MultiObjectiveBranchCreationError::DeadlineExpired(failed_idx)) => {
                    branch_selection_failures.insert(failed_idx);
                    quick_results
                        .insert(failed_idx, MultiObjectiveGraphDomainResult::DeadlineExpired);
                    None
                }
            })
            .collect();

        // A typed receipt is the sole downstream authority. In particular this
        // boundary never re-reads the config/environment and never infers
        // provenance from an all-verified mask.
        let kfsb_has_cert_receipt = successful_results.iter().any(|(_, children)| {
            children
                .iter()
                .any(|(_, _, _, effect)| effect.receipt().is_some())
        });
        let kfsb_cert_probe = kfsb_has_cert_receipt && kfsb_probe_enabled();
        // Remove terminal certified leaves BEFORE Complete Clip and every
        // dense/scalar child evaluator. This is the critical empty-spec
        // short-circuit: routing an all-verified child through either fallback
        // would reject its zero-row target set as a propagation failure.
        let cert_partition = partition_kfsb_certified_children(
            Instant::now(),
            thresholds,
            &domain_by_idx,
            successful_results,
        );
        for &parent_idx in &cert_partition.invalid_parents {
            branch_selection_failures.insert(parent_idx);
            quick_results.insert(
                parent_idx,
                MultiObjectiveGraphDomainResult::PropagationFailure,
            );
        }
        for &parent_idx in &cert_partition.expired_parents {
            branch_selection_failures.insert(parent_idx);
            quick_results.insert(parent_idx, MultiObjectiveGraphDomainResult::DeadlineExpired);
        }
        let kfsb_cert_accounting = cert_partition.accounting;
        let kfsb_certified_parent_deadlines = cert_partition.certified_parent_deadlines;
        let successful_results = cert_partition.pending;
        let preverified_children_by_parent = cert_partition.preverified;
        if kfsb_cert_probe {
            let preverified_leaves = preverified_children_by_parent
                .values()
                .map(Vec::len)
                .sum::<usize>();
            let pending_leaves = successful_results
                .iter()
                .map(|(_, children)| children.len())
                .sum::<usize>();
            debug_assert_eq!(
                kfsb_cert_accounting.input_leaves,
                pending_leaves + preverified_leaves + kfsb_cert_accounting.rejected_leaves()
            );
            debug_assert_eq!(
                kfsb_cert_accounting.complete_receipts
                    + kfsb_cert_accounting.parent_closes
                    + kfsb_cert_accounting.ordinary_preverified_leaves,
                preverified_leaves
            );
            debug_assert_eq!(
                kfsb_cert_accounting.pruned_spec_rows,
                kfsb_cert_accounting.partial_receipts
                    + kfsb_cert_accounting.complete_receipts
                    + kfsb_cert_accounting.parent_closes
            );
            eprintln!(
                "[kfsb-cert-reuse] live={} parents={} preverified_parents={} input_leaves={} preverified_leaves={} pending_leaves={} skipped_child_evals={} partial_receipts={} complete_receipts={} parent_closes={} ordinary_preverified_leaves={} pruned_spec_rows={} rejected_parents={} rejected_leaves={} invalid_parents={} expired_parents={}",
                usize::from(!kfsb_certified_parent_deadlines.is_empty()),
                kfsb_cert_accounting.input_parents,
                preverified_children_by_parent.len(),
                kfsb_cert_accounting.input_leaves,
                preverified_leaves,
                pending_leaves,
                preverified_leaves,
                kfsb_cert_accounting.partial_receipts,
                kfsb_cert_accounting.complete_receipts,
                kfsb_cert_accounting.parent_closes,
                kfsb_cert_accounting.ordinary_preverified_leaves,
                kfsb_cert_accounting.pruned_spec_rows,
                kfsb_cert_accounting.rejected_parents(),
                kfsb_cert_accounting.rejected_leaves(),
                kfsb_cert_accounting.invalid_parents,
                kfsb_cert_accounting.expired_parents,
            );
        }

        // DomainClipper decision precompute belongs at the common committed-child
        // boundary, not inside one branching implementation. This covers
        // wave-batched kFSB winners, per-domain kFSB misses, kFSB-disabled
        // configurations, and every other ReLU branching heuristic handled by
        // this lane. The immutable parent lA snapshot is repeated across all of
        // its prospective children; no child CROWN pass runs here.
        maybe_run_unbounded_advisory(bounded_shared_lane, true, || {
            // #phase-telemetry: inside the closure — the bounded facade skips
            // this precompute entirely and must print no line for it.
            let __t_complete_clip =
                crate::phase_telemetry::phase_telemetry_enabled().then(Instant::now);
            let complete_clip_groups: Vec<(
                &MultiObjectiveGraphBabDomain,
                Vec<&MultiObjectiveGraphBabDomain>,
            )> = successful_results
                .iter()
                .filter_map(|(parent_idx, children)| {
                    let parent = domains_with_unstable
                        .iter()
                        .find_map(|(idx, parent, _)| (*idx == *parent_idx).then_some(*parent))?;
                    let child_refs: Vec<&MultiObjectiveGraphBabDomain> =
                        children.iter().map(|(_, child, _, _)| child).collect();
                    (!child_refs.is_empty()).then_some((parent, child_refs))
                })
                .collect();
            self.precompute_complete_clip_committed_decisions(
                graph,
                &complete_clip_groups,
                objectives,
                engine,
            );
            if let Some(t) = __t_complete_clip {
                eprintln!(
                    "[phase] mo-complete-clip groups={} children={} secs={:.2}",
                    complete_clip_groups.len(),
                    complete_clip_groups
                        .iter()
                        .map(|(_, children)| children.len())
                        .sum::<usize>(),
                    t.elapsed().as_secs_f64(),
                );
            }
        });

        // Collect all children that need CROWN bounds computation.
        let (all_children, parent_domain_lookup) = collect_multi_objective_children(
            &domains_with_unstable,
            successful_results,
            &mut quick_results,
        );

        // Per-child single-pass / beta-opt CROWN evaluation, unchanged from the
        // original closure. Used directly for the FALLBACK partition and as the
        // whole-batch fallback when the domain-batched primitive errors.
        let eval_child = |parent_idx: &usize, child: &MultiObjectiveGraphBabDomain| {
            let _rayon_task_guard = RayonTaskGuard::new();
            let Some(parent) = parent_domain_lookup.get(parent_idx) else {
                tracing::warn!(
                    "process_graph_domains_batched_gpu_multi_objective: missing parent lookup for child of idx {}",
                    parent_idx
                );
                return Err(MultiObjectiveChildEvalError::PropagationFailure);
            };

            // Use beta-CROWN with SPSA optimization for shallow domains
            let mut beta_state = child.beta_state.clone();
            let context = GraphCrownContext::new_with_node_bounds_map(
                &child.history,
                cut_pool, // Part of #3813: apply existing cuts (read-only)
                Some(&parent.node_bounds),
                Some(engine),
            )
            .with_alpha(&child.alpha_state);
            let pruned_targets =
                prune_verified_multi_objective_targets(objectives, thresholds, &child.verified);
            let targets = MultiObjectiveTargets::new(
                &pruned_targets.objectives,
                &pruned_targets.thresholds,
                &pruned_targets.verified_mask,
            );
            let pruned_cached_las =
                prune_cached_las_for_targets(child.cached_las(), &pruned_targets);
            // Only run beta optimization when enabled and for shallow domains.
            let result = if bounded_shared_lane {
                // The analytical optimizer's storing-intermediates mode both
                // bypasses the call-local bounded CUDA selector and captures
                // all-node coefficient state. This treatment is intrinsically
                // one Standard pass with inherited beta, independent of the
                // separately gated baseline-only experiment.
                self.propagate_multi_objective_with_beta_and_cache(
                    graph,
                    child.input_bounds.as_ref(),
                    &context,
                    &beta_state,
                    &targets,
                    &pruned_cached_las,
                    false,
                )
            } else if child_uses_analytical_beta_optimizer(
                bounded_shared_lane,
                self.config.beta_iterations > 0 && child.depth <= self.config.beta_max_depth,
            ) {
                self.optimize_graph_beta_analytical_multi_objective_with_cache(
                    graph,
                    child.input_bounds.as_ref(),
                    &context,
                    &mut beta_state,
                    &targets,
                    false, // conjunctive: batched path always uses disjunctive mode (#3334 closed)
                    &pruned_cached_las,
                    true,
                )
            } else {
                // Skip optimization, just propagate with inherited beta
                self.propagate_multi_objective_with_beta_and_cache(
                    graph,
                    child.input_bounds.as_ref(),
                    &context,
                    &beta_state,
                    &targets,
                    &pruned_cached_las,
                    true,
                )
            };
            match result {
                Ok((active_bounds, node_cache, active_cached_las)) => {
                    if bounded_shared_lane {
                        match engine.poll_crown_backward_deadline() {
                            Ok(()) => {}
                            Err(ref error) if error.is_deadline_exceeded() => {
                                return Err(MultiObjectiveChildEvalError::DeadlineExpired);
                            }
                            Err(error) => {
                                tracing::warn!(
                                    "bounded child pre-merge publication poll failed: {error}"
                                );
                                return Err(MultiObjectiveChildEvalError::PropagationFailure);
                            }
                        }
                    }
                    let obj_bounds = merge_pruned_objective_bounds(
                        &child.objective_bounds,
                        &pruned_targets,
                        active_bounds,
                    );
                    if bounded_shared_lane {
                        match engine.poll_crown_backward_deadline() {
                            Ok(()) => {}
                            Err(ref error) if error.is_deadline_exceeded() => {
                                return Err(MultiObjectiveChildEvalError::DeadlineExpired);
                            }
                            Err(error) => {
                                tracing::warn!(
                                    "bounded child final publication poll failed: {error}"
                                );
                                return Err(MultiObjectiveChildEvalError::PropagationFailure);
                            }
                        }
                    }
                    Ok((
                        obj_bounds,
                        node_cache,
                        beta_state,
                        // Per-child CPU path never persists ascent α — the
                        // child keeps its inherited α (#hard-six unshared-α).
                        None,
                        active_cached_las,
                        pruned_targets,
                        ChildContinuationStateProvenance::Established,
                    ))
                }
                // #2926: Preserve infeasibility signal through the parallel closure.
                Err(ref e) if e.is_infeasible_domain() => {
                    Err(MultiObjectiveChildEvalError::Infeasible)
                }
                Err(ref e) if e.is_deadline_exceeded() => {
                    Err(MultiObjectiveChildEvalError::DeadlineExpired)
                }
                Err(e) => {
                    tracing::warn!("Batched multi-objective child propagation failed: {e}");
                    Err(MultiObjectiveChildEvalError::PropagationFailure)
                }
            }
        };

        // GPU single-pass lane (#w5-bab-throughput): when the engine provides the
        // sound GPU CROWN backward on a conv graph, route beta-opt-eligible
        // children through the domain-batched single-pass adapter too. Measured
        // (cifar100 prop_idx_7641, release): the per-child CPU beta-opt inner
        // pass costs ~3s (conv2d_transpose_backward_coeff_f64-dominated), so ONE
        // domain consumed the whole BaB window; the adapter's whole-suffix GPU
        // sound backward (try_gpu_beta_batched_resnet, alpha-bridged, inherited-β
        // dual folded) bounds a child in a fraction of that. Trades per-domain β
        // OPTIMIZATION for ~10x domain throughput. Default ON; NY_MO_GPU_BATCH=0
        // restores the legacy per-child beta-opt lane byte-identically.
        let authority_deadline = self.effective_graph_bab_deadline();
        let gpu_single_pass_lane = multi_objective_gpu_single_pass_enabled()
            && graph.has_conv_layers()
            && (crate::sound_gpu_gate::sound_gpu_crown_for_wide_with_deadline(authority_deadline)
                .is_some_and(|gpu| gpu_single_pass_backend_eligible(gpu, authority_deadline))
                || engine
                    .as_gpu_crown_backward()
                    .is_some_and(|gpu| gpu_single_pass_backend_eligible(gpu, authority_deadline)));
        // The default-dark bounded CPU facade deliberately exposes no broad
        // GPU trait. Keep its children OUT of the dense-spec adapter: that
        // adapter's broad ResNet selector correctly rejects CUDA's narrower
        // contract. Serial per-child fallback enters constrained backward,
        // where the call-local K<=8 beta selector is the final authority.
        // Partition children into a DOMAIN-batchable single-pass set and a
        // FALLBACK set. A child is batchable iff ALL of:
        //   * cuts inactive (the dense-spec primitive does not apply cuts);
        //   * the single-pass branch applies (NOT beta-opt for this depth, OR
        //     the GPU single-pass lane is on);
        //   * no per-disjunct alphas (no GraphBabDomain equivalent);
        //   * every relu node is present in `child.node_bounds`
        //     (`from_graph_domains` errors otherwise);
        //   * the first objective bound is finite.
        // Everything else falls back to the EXACT per-child path.
        let cuts_inactive = cut_pool.map_or(true, |pool| pool.is_empty());
        let mut batchable_positions: Vec<usize> = Vec::new();
        for (pos, (_parent_idx, child, _is_active, _cert_effect)) in all_children.iter().enumerate()
        {
            let single_pass_branch = child_uses_dense_spec_adapter(
                bounded_shared_lane,
                gpu_single_pass_lane,
                self.config.beta_iterations > 0 && child.depth <= self.config.beta_max_depth,
            );
            let relu_nodes_present = relu_nodes
                .iter()
                .all(|name| child.node_bounds.contains_key(name));
            let first_obj_finite = child
                .objective_bounds
                .first()
                .is_some_and(|(l, u)| l.is_finite() && u.is_finite());
            let batchable = cuts_inactive
                && single_pass_branch
                && child.per_disjunct_alphas().is_none()
                && relu_nodes_present
                && first_obj_finite;
            if batchable {
                batchable_positions.push(pos);
            } else {
                // #wide-decline-tally: a child excluded here can NEVER appear in a
                // wide batch, so this is the outermost term of the coverage gap.
                // One relaxed atomic; nothing reads it.
                ny_core::wide_lane_telemetry::note_wide_lane_decline(
                    ny_core::wide_lane_telemetry::WideLaneDecline::WaveChildNotBatchable,
                );
            }
        }

        // Run the FALLBACK set through the existing per-child path (in parallel).
        let mut child_bounds: Vec<Option<_>> = (0..all_children.len()).map(|_| None).collect();
        let parent_ids: Vec<usize> = all_children
            .iter()
            .map(|(parent_idx, _, _, _)| *parent_idx)
            .collect();
        // Parent IDs whose GPU children were deliberately left unevaluated
        // because the authoritative deadline could not admit another bounded
        // chunk. Keep this typed separately from numerical propagation failure
        // so the outer lifecycle returns Timeout rather than pre-deadline
        // Unknown.
        let mut parents_with_deadline: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        // O(1) membership for the fallback complement, replacing a linear
        // `batchable_positions.contains(pos)` inside a filter over all child
        // positions (was O(C²) for C children ≈ 2D). `batchable_positions`
        // holds distinct positions, so a HashSet yields identical membership.
        let batchable_set: std::collections::HashSet<usize> =
            batchable_positions.iter().copied().collect();
        let fallback_positions: Vec<usize> = (0..all_children.len())
            .filter(|pos| !batchable_set.contains(pos))
            .collect();
        let evaluate_fallback_position = |&pos: &usize| {
            let (parent_idx, child, _is_active, _cert_effect) = &all_children[pos];
            (pos, eval_child(parent_idx, child))
        };
        if !fallback_positions.is_empty()
            && admit_multi_objective_child_wave(
                Instant::now(),
                self.effective_graph_bab_deadline(),
                false,
                &fallback_positions,
                &parent_ids,
                &mut parents_with_deadline,
                &mut child_bounds,
            )
        {
            // #phase-telemetry: inside the admitted branch — an empty or
            // deadline-refused fallback partition prints nothing.
            let __t_fallback_eval =
                crate::phase_telemetry::phase_telemetry_enabled().then(Instant::now);
            let fallback_results: Vec<_> = if bounded_shared_lane {
                // Do not multiply the facade's per-call host-memory cap across
                // an unbounded rayon fan-out.
                fallback_positions
                    .iter()
                    .map(&evaluate_fallback_position)
                    .collect()
            } else {
                fallback_positions
                    .par_iter()
                    .map(&evaluate_fallback_position)
                    .collect()
            };
            for (pos, result) in fallback_results {
                child_bounds[pos] = Some(result);
            }
            if let Some(t) = __t_fallback_eval {
                eprintln!(
                    "[phase] mo-fallback-eval n={} secs={:.2}",
                    fallback_positions.len(),
                    t.elapsed().as_secs_f64(),
                );
            }
        }

        // Run the BATCHABLE set through the domain-batched single-pass adapter,
        // with per-chunk fallback to the per-child path on any batched error.
        //
        // Chunking (#w5-bab-throughput): with the GPU single-pass lane on, a BaB
        // batch can hold up to `batch_size` (256) children, each one whole-suffix
        // GPU backward — un-interruptible for minutes as a single call. Bounded
        // chunks with a deadline check between them cap the overrun; children
        // left unprocessed at the deadline surface carry an explicit deadline
        // result (sound: their sub-regions stay unexplored and the outer loop
        // returns Timeout). Legacy lane: one chunk, exactly the previous single
        // call.
        if !batchable_positions.is_empty() {
            let authority_deadline = self.effective_graph_bab_deadline();
            let chunk_size = if gpu_single_pass_lane {
                mo_gpu_single_pass_chunk().map(|requested| {
                    if authority_deadline.is_some() {
                        // #wide-decline-tally: EVERY scored row runs under an
                        // authoritative deadline, so this is the branch that
                        // decides how many domains a wide pass can amortize over.
                        // Record only a real narrowing, so the counter means "the
                        // operator asked for a wider batch and the scored
                        // configuration refused", not "the default applied".
                        if requested > MO_GPU_SINGLE_PASS_CHUNK {
                            ny_core::wide_lane_telemetry::note_wide_lane_decline(
                                ny_core::wide_lane_telemetry::WideLaneDecline::WaveChunkCappedByDeadline,
                            );
                        }
                        // #w5-chunk-override (dark, NY_MO_GPU_CHUNK_DEADLINE=1):
                        // the clamp above is a DEADLINE-GRANULARITY choice, not a
                        // soundness one — a chunk is un-interruptible, so a wider
                        // chunk can overrun the deadline by one pass before the
                        // between-chunk check fires (children left unprocessed
                        // still carry an explicit deadline result, so the verdict
                        // stays sound either way). The in-tree A/B recorded
                        // 63 -> 255 domains at chunk 8 and 63 -> 511 at chunk 64,
                        // i.e. this clamp — not lane refusal — is what bounds
                        // wide-pass amortization in every scored run. Opting in
                        // lets a measured session trade granularity for width.
                        // #adaptive-chunk (2026-08-13, DEFAULT ON): the fixed
                        // ceiling below is a stand-in for "how many children fit
                        // in the remaining budget", which the wave now MEASURES.
                        // Use the measurement when there is one, and take the
                        // TIGHTER of the two so this can only ever narrow an
                        // over-wide request — it never widens past what the
                        // caller asked for, and never past the fixed ceiling
                        // unless the operator explicitly opted in below.
                        //
                        // Why this matters concretely: at the measured ~3 s per
                        // child, an un-interruptible 256-wide chunk is ~768 s and
                        // would blow a 100 s row outright. The constant 8 is only
                        // right by coincidence at today's cost.
                        let ceiling = adaptive_chunk_ceiling(Instant::now(), authority_deadline);
                        let base = if deadline_chunk_override_enabled() {
                            requested
                        } else {
                            requested.min(MO_GPU_SINGLE_PASS_CHUNK)
                        };
                        ceiling.map_or(base, |fits| base.min(fits).max(1))
                    } else {
                        requested
                    }
                })
            } else {
                Some(batchable_positions.len())
            };
            if let Some(chunk_size) = chunk_size {
                let mut deadline_abandoned = false;
                for chunk in batchable_positions.chunks(chunk_size.max(1)) {
                    let effective_deadline = self.effective_graph_bab_deadline();
                    let now = Instant::now();
                    let admitted = if deadline_abandoned {
                        mark_multi_objective_positions_deadline_expired(
                            chunk,
                            &parent_ids,
                            &mut parents_with_deadline,
                            &mut child_bounds,
                        );
                        false
                    } else {
                        admit_multi_objective_child_wave(
                            now,
                            effective_deadline,
                            gpu_single_pass_lane,
                            chunk,
                            &parent_ids,
                            &mut parents_with_deadline,
                            &mut child_bounds,
                        )
                    };
                    if !admitted {
                        if !deadline_abandoned
                            && std::env::var("NY_PROPFAIL_PROBE").ok().as_deref() == Some("1")
                        {
                            eprintln!(
                                "[propfail] site=deadline-drop chunk_dropped={}",
                                chunk.len()
                            );
                        }
                        deadline_abandoned = true;
                        continue;
                    }
                    let chunk_refs: Vec<&MultiObjectiveGraphBabDomain> =
                        chunk.iter().map(|&pos| &all_children[pos].1).collect();
                    match self.batched_selective_root_alpha_multi_objective_children(
                        graph,
                        &chunk_refs,
                        relu_nodes,
                        objectives,
                        thresholds,
                        engine,
                        gpu_single_pass_lane,
                        (bab_round == 0)
                            .then_some(selective_root_alpha_candidate)
                            .flatten(),
                        effective_deadline,
                    ) {
                        Ok(batched_results) => {
                            debug_assert_eq!(batched_results.len(), chunk.len());
                            for (&pos, result) in chunk.iter().zip(batched_results) {
                                // A batched `PropagationFailure` is a lane
                                // refusal, not evidence that this child cannot be
                                // evaluated. It covers heterogeneous-layout,
                                // result-shape, validation, device, and non-finite
                                // failures inside the dense-spec adapter. Always
                                // retry through the established per-child CPU
                                // path when that path is permitted; otherwise one
                                // refused GPU child would taint the whole run as
                                // pre-deadline Unknown despite a sound fallback.
                                let retry = batched_child_refusal_has_cpu_fallback(
                                    bounded_shared_lane,
                                    result.as_ref().err().copied(),
                                );
                                let result = if retry
                                    && admit_multi_objective_child_wave(
                                        Instant::now(),
                                        self.effective_graph_bab_deadline(),
                                        false,
                                        &[pos],
                                        &parent_ids,
                                        &mut parents_with_deadline,
                                        &mut child_bounds,
                                    ) {
                                    let (parent_idx, child, _, _) = &all_children[pos];
                                    eval_child(parent_idx, child)
                                } else if retry {
                                    deadline_abandoned = true;
                                    continue;
                                } else {
                                    result
                                };
                                child_bounds[pos] = Some(result);
                            }
                        }
                        Err(BatchedMultiObjectiveAdapterError::Fallback) => {
                            // Chunk fallback: route this chunk back through the
                            // exact per-child path (sound, mirrors
                            // batched_single.rs), but only while CPU work remains
                            // admissible before the literal deadline.
                            if admit_multi_objective_child_wave(
                                Instant::now(),
                                self.effective_graph_bab_deadline(),
                                false,
                                chunk,
                                &parent_ids,
                                &mut parents_with_deadline,
                                &mut child_bounds,
                            ) {
                                let fb: Vec<_> = if bounded_shared_lane {
                                    chunk.iter().map(&evaluate_fallback_position).collect()
                                } else {
                                    chunk.par_iter().map(&evaluate_fallback_position).collect()
                                };
                                for (pos, result) in fb {
                                    child_bounds[pos] = Some(result);
                                }
                            } else {
                                deadline_abandoned = true;
                            }
                        }
                        Err(BatchedMultiObjectiveAdapterError::ResourceRefused) => {
                            // A bounded host facade refused this allocation.
                            // Retrying through the scalar CPU path would recreate
                            // the same uncapped coefficient buffers, so preserve
                            // the unresolved child without launching more work.
                            tracing::warn!(
                                children = chunk.len(),
                                "bounded shared executor resource refusal; suppressing unbounded per-child fallback"
                            );
                            for &pos in chunk {
                                child_bounds[pos] =
                                    Some(Err(MultiObjectiveChildEvalError::PropagationFailure));
                            }
                        }
                        Err(BatchedMultiObjectiveAdapterError::DeadlineExpired) => {
                            mark_multi_objective_positions_deadline_expired(
                                chunk,
                                &parent_ids,
                                &mut parents_with_deadline,
                                &mut child_bounds,
                            );
                            deadline_abandoned = true;
                        }
                    }
                }
            } else {
                tracing::warn!(
                    "invalid NY_MO_GPU_CHUNK; disabling the GPU single-pass lane for this batch"
                );
                if admit_multi_objective_child_wave(
                    Instant::now(),
                    self.effective_graph_bab_deadline(),
                    false,
                    &batchable_positions,
                    &parent_ids,
                    &mut parents_with_deadline,
                    &mut child_bounds,
                ) {
                    let fb: Vec<_> = if bounded_shared_lane {
                        batchable_positions
                            .iter()
                            .map(&evaluate_fallback_position)
                            .collect()
                    } else {
                        batchable_positions
                            .par_iter()
                            .map(&evaluate_fallback_position)
                            .collect()
                    };
                    for (pos, result) in fb {
                        child_bounds[pos] = Some(result);
                    }
                }
            }
        }

        // Every position must now be filled (batchable ∪ fallback == all).
        let child_bounds: Vec<_> = child_bounds
            .into_iter()
            .map(|slot| {
                slot.unwrap_or_else(|| {
                    tracing::warn!(
                        "process_graph_domains_batched_gpu_multi_objective: unfilled child bound slot (#partition)"
                    );
                    Err(MultiObjectiveChildEvalError::PropagationFailure)
                })
            })
            .collect();

        // Build results from child bounds
        let mut children_by_parent: std::collections::HashMap<
            usize,
            Vec<(MultiObjectiveGraphBabDomain, bool)>,
        > = preverified_children_by_parent;

        // #1861: Track parents that had child propagation failures or violations.
        let mut parents_with_failure: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let mut parents_with_violation: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        // #boxlift [frontier] telemetry (dark, NY_PHASE_TELEMETRY=1, print-only):
        // per-depth worst-child accumulator for THIS batch. The resnet batched
        // lane is otherwise `[converge]`-silent; the BOXLIFT decision table
        // (docs/BOXLIFT_CHARTER.md §2.1(a)) needs a per-depth worst-child metric.
        // Gate checked FIRST and cached here — gate-off is one bool load before
        // the loop plus one predictable branch per child (no map, no arithmetic,
        // no allocation). Read-only: nothing downstream reads the accumulator,
        // the counter, or the frames; verdicts are byte-identical either way.
        let frontier_telemetry_on = crate::phase_telemetry::phase_telemetry_enabled();
        // #violdrop probe gate, read ONCE per batch (dark, print-only).
        let violdrop_probe_on = std::env::var("NY_VIOLDROP_PROBE").ok().as_deref() == Some("1");
        let mut frontier_batch_domains: u64 = 0;
        let mut frontier_worst_by_depth: std::collections::BTreeMap<usize, f32> =
            std::collections::BTreeMap::new();

        // #phase-telemetry: the fold below consumes `all_children`, so capture
        // its width alongside the start instant. Reuses the gate bool already
        // cached above — no second environment read.
        let __t_child_fold = frontier_telemetry_on.then(|| (Instant::now(), all_children.len()));
        for ((parent_idx, mut child, _is_active, _cert_effect), bounds_result) in
            all_children.into_iter().zip(child_bounds)
        {
            match bounds_result {
                Ok((
                    obj_bounds,
                    node_cache,
                    beta_state,
                    alpha_state,
                    mut active_cached_las,
                    pruned_targets,
                    state_provenance,
                )) => {
                    // #w5-bab-throughput: monotone BaB bound inheritance. The
                    // single-pass bound source can differ from the parent's
                    // (root bounds = margin ∩ GPU ∩ IBP); without this a child
                    // could REGRESS below the parent's already-proven bound and
                    // stall convergence. `child.objective_bounds` still holds
                    // the inherited parent bounds here (update_bounds not yet
                    // called). Applied only on the GPU lane so the legacy flag
                    // stays byte-identical.
                    // #violdrop probe (dark, NY_VIOLDROP_PROBE=1, print-only):
                    // keep the PRE-tighten fresh bounds so the probe below can
                    // tell "the single-pass backward already returned a crossed
                    // interval" apart from "the parent intersection crossed it".
                    let probe_fresh: Option<Vec<(f32, f32)>> = if violdrop_probe_on {
                        Some(obj_bounds.clone())
                    } else {
                        None
                    };
                    let obj_bounds = if bounded_shared_lane {
                        inherit_bounded_child_bounds(&child.objective_bounds, obj_bounds)
                    } else if gpu_single_pass_lane {
                        tighten_child_bounds_with_parent(&child.objective_bounds, obj_bounds)
                    } else {
                        obj_bounds
                    };
                    // Keep continuation state staged until objective bounds
                    // validate. A failed publication must leave the inherited
                    // child unchanged, especially before the f64 retry.
                    let mut continuation_state =
                        Some((node_cache, beta_state, alpha_state, state_provenance));
                    let mut bounds_ok = child.update_bounds(obj_bounds, thresholds, false).is_ok();
                    if !bounds_ok && !bounded_shared_lane {
                        // Publication validation failure (non-finite, inverted,
                        // wrong objective count, or another malformed GPU-batch
                        // product) is still an accelerator refusal. Recompute on
                        // the CPU f64 sound backward instead of converting an
                        // available fallback into pre-deadline Unknown. The retry
                        // starts from inherited child state; rejected GPU
                        // continuation data never crosses this boundary.
                        if mo_batch_chunk_start_allowed(
                            Instant::now(),
                            self.effective_graph_bab_deadline(),
                            false,
                        ) {
                            let f64_context = GraphCrownContext::new_with_node_bounds_map(
                                &child.history,
                                cut_pool,
                                Some(&child.node_bounds),
                                None,
                            )
                            .with_alpha(&child.alpha_state);
                            match self.propagate_crown_with_graph_constraints(
                                graph,
                                child.input_bounds.as_ref(),
                                &f64_context,
                                None,
                                None,
                            ) {
                                Ok((f64_output, _)) => {
                                    if let Ok(f64_bounds) =
                                        Self::objective_bounds_multi(&f64_output, objectives)
                                    {
                                        bounds_ok = child
                                            .update_bounds(f64_bounds, thresholds, false)
                                            .is_ok();
                                        if bounds_ok {
                                            // The retry was certified against
                                            // the inherited continuation state,
                                            // not the rejected f32 candidate.
                                            continuation_state = None;
                                            active_cached_las.fill(None);
                                        }
                                    }
                                }
                                Err(ref error) if error.is_deadline_exceeded() => {
                                    parents_with_deadline.insert(parent_idx);
                                    continue;
                                }
                                Err(_) => {}
                            }
                        } else {
                            parents_with_deadline.insert(parent_idx);
                            continue;
                        }
                    }
                    if !bounds_ok {
                        // NaN in objective bounds → treat as propagation failure (#2982)
                        if std::env::var("NY_PROPFAIL_PROBE").ok().as_deref() == Some("1") {
                            eprintln!("[propfail] site=NaN-bounds depth={}", child.depth);
                        }
                        parents_with_failure.insert(parent_idx);
                        continue;
                    }

                    if let Some((node_cache, beta_state, alpha_state, state_provenance)) =
                        continuation_state
                    {
                        // Node bounds are the selected arm's independently
                        // sound continuation enclosure; β/α are warm starts.
                        install_child_continuation_state(
                            &mut child,
                            node_cache,
                            beta_state,
                            alpha_state,
                            state_provenance,
                        );
                        debug_assert!(
                            !state_provenance.invalidates_all_cached_las()
                                || child.cached_las().iter().all(Option::is_none),
                            "selective W publication retained a stale lA"
                        );
                    }

                    let all_verified = child.all_verified();
                    let any_violated = child.any_violated(thresholds, false);

                    // #violdrop probe (dark, NY_VIOLDROP_PROBE=1, print-only):
                    // dump every objective row that reads as a "conclusive
                    // violation" on this child, so the two candidate mechanisms
                    // are distinguishable from the log alone:
                    //   crossed=true  => an INVERTED (l > u) interval, an f32/α
                    //                    merge-slop artifact that must be repaired.
                    //   crossed=false => an ordered `l <= u < t`, i.e. the child's
                    //                    UPPER is genuinely being trusted on a
                    //                    β-constrained subdomain (where only the
                    //                    LOWER direction is certified).
                    // Gate off => zero output, byte-identical.
                    if any_violated && violdrop_probe_on {
                        for (i, ((l, u), &t)) in child
                            .objective_bounds
                            .iter()
                            .zip(thresholds.iter())
                            .enumerate()
                        {
                            if !crate::beta_crown::BetaCrownConfig::domain_is_violation_for_mode(
                                false, *l, *u, t,
                            ) {
                                continue;
                            }
                            let fresh = probe_fresh
                                .as_ref()
                                .and_then(|f| f.get(i).copied())
                                .unwrap_or((f32::NAN, f32::NAN));
                            let parent_iv = parent_domain_lookup
                                .get(&parent_idx)
                                .and_then(|p| p.objective_bounds.get(i).copied())
                                .unwrap_or((f32::NAN, f32::NAN));
                            eprintln!(
                                "[violdrop] obj={i} l={l:.6} u={u:.6} t={t:.6} crossed={} depth={} fresh=[{:.6},{:.6}] fresh_crossed={} parent=[{:.6},{:.6}]",
                                l > u,
                                child.depth,
                                fresh.0,
                                fresh.1,
                                fresh.0 > fresh.1,
                                parent_iv.0,
                                parent_iv.1,
                            );
                        }
                    }

                    // #boxlift [frontier] hook: fold this child into the batch's
                    // per-depth worst-unverified-margin frame. depth = the
                    // child's split count (`depth` increments once per split
                    // constraint, so it IS the split_count); margin = lb − t on
                    // an unverified objective (this lane runs lower-bound mode:
                    // update_bounds/any_violated above pass verify_upper=false).
                    // Only surviving frontier children (neither all-verified nor
                    // violated) contribute a margin; every child that completed
                    // a bounds update counts toward the cumulative domain
                    // counter. Bounds here are finite: update_bounds rejects
                    // non-finite unverified bounds via its priority fold (#2982).
                    if frontier_telemetry_on {
                        frontier_batch_domains += 1;
                        if !all_verified && !any_violated {
                            let mut child_worst = f32::INFINITY;
                            for (((lb, _), &t), &v) in child
                                .objective_bounds
                                .iter()
                                .zip(thresholds.iter())
                                .zip(child.verified.iter())
                            {
                                if !v {
                                    child_worst = child_worst.min(lb - t);
                                }
                            }
                            if child_worst < f32::INFINITY {
                                let slot = frontier_worst_by_depth
                                    .entry(child.depth)
                                    .or_insert(f32::INFINITY);
                                *slot = slot.min(child_worst);
                            }
                        }
                    }

                    // #violdrop: a child whose objective interval reads
                    // `upper < threshold` is NOT conclusively violated — the
                    // β-carried split certifies only the LOWER bound, so that
                    // `upper` proves nothing about the child's sub-region (full
                    // measurement + GT-independent soundness argument in
                    // `bab_violated_child_drop_enabled`). Keeping the child makes
                    // it an ordinary unverified frontier domain; abandoning it
                    // used to raise `unresolved_due_to_violated_drop`, which
                    // forces the WHOLE run to `Unknown`. Measured on vit_2023
                    // ibp_3_3_8_3005: both root children were dropped and BaB
                    // returned after 2.58 s of a 90.25 s grant.
                    if any_violated && violation_drop_is_certified(child.depth) {
                        // #1861: Track violated children instead of silently dropping.
                        parents_with_violation.insert(parent_idx);
                    } else {
                        let merged_cached_las = merge_pruned_cached_las(
                            child.cached_las(),
                            &pruned_targets,
                            active_cached_las,
                        );
                        if child.set_shared_cached_las(merged_cached_las).is_err() {
                            if std::env::var("NY_PROPFAIL_PROBE").ok().as_deref() == Some("1") {
                                eprintln!("[propfail] site=cached_las depth={}", child.depth);
                            }
                            parents_with_failure.insert(parent_idx);
                            continue;
                        }
                        children_by_parent
                            .entry(parent_idx)
                            .or_default()
                            .push((child, all_verified));
                    }
                }
                Err(MultiObjectiveChildEvalError::Infeasible) => {
                    // #2926: Infeasible domain = empty = trivially verified.
                    // Ensure parent doesn't fall to PropagationFailure when both children infeasible.
                    children_by_parent.entry(parent_idx).or_default();
                }
                Err(MultiObjectiveChildEvalError::DeadlineExpired) => {
                    parents_with_deadline.insert(parent_idx);
                }
                Err(MultiObjectiveChildEvalError::PropagationFailure) => {
                    // #1861: child bounds computation failed — sub-region unexplored.
                    if std::env::var("NY_PROPFAIL_PROBE").ok().as_deref() == Some("1") {
                        eprintln!("[propfail] site=child-eval-failure parent={parent_idx}");
                    }
                    parents_with_failure.insert(parent_idx);
                }
            }
        }
        if let Some((t, children)) = __t_child_fold {
            eprintln!(
                "[phase] mo-child-fold children={} secs={:.2}",
                children,
                t.elapsed().as_secs_f64(),
            );
        }

        // #boxlift [frontier] emission: at most ONE line per distinct depth per
        // BATCH (batch-level frames, not per-domain — a mixed-depth batch gets
        // one line per depth present, in depth order via the BTreeMap). The
        // cumulative domain counter is a process-wide atomic (function-local
        // `static`, same idiom as FIRST_BATCH above); it only advances when the
        // gate is on, and nothing but this print ever reads it. Gate-off skips
        // the whole block — byte-identical, zero output.
        if frontier_telemetry_on && frontier_batch_domains > 0 {
            static FRONTIER_DOMAINS_CUMULATIVE: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let cumulative = FRONTIER_DOMAINS_CUMULATIVE
                .fetch_add(frontier_batch_domains, std::sync::atomic::Ordering::Relaxed)
                + frontier_batch_domains;
            for (&depth, &worst) in &frontier_worst_by_depth {
                crate::phase_telemetry::frontier_frame(depth, worst, cumulative);
            }
        }

        // NY_BRANCH_TRACE (dark, diagnostic-only): per-split frontier-bound lift.
        // For every parent that produced children, log the chosen split
        // (node,neuron), the parent's worst-unverified straggler LB, the
        // post-split bound on THAT objective (= min over the split's children of
        // their LB on the straggler row — the domain's effective bound after the
        // partition), and the lift. Aggregating these lines answers "does each
        // split move the frontier bound, or are most splits wasted?" and "is the
        // same layer repeatedly chosen?". Print-only; gate off ⇒ the whole block
        // is skipped ⇒ byte-identical. Advisory measurement ⇒ soundness-free.
        if std::env::var("NY_BRANCH_TRACE").ok().as_deref() == Some("1") {
            let unstable_count: std::collections::HashMap<usize, usize> = domains_with_unstable
                .iter()
                .map(|(i, _, u)| (*i, u.len()))
                .collect();
            for (parent_idx, children_vec) in &children_by_parent {
                let Some(parent) = parent_domain_lookup.get(parent_idx) else {
                    continue;
                };
                // A typed terminal parent close deliberately has no new split
                // constraint. Do not attribute the parent's historical last
                // constraint to that close in split telemetry.
                if children_vec.len() == 1
                    && children_vec[0].0.depth() == parent.depth()
                    && children_vec[0].0.history().split_count == parent.history().split_count
                    && children_vec[0].0.history().constraints == parent.history().constraints
                {
                    continue;
                }
                // Worst unverified straggler on the parent (mirrors the selector).
                let mut straggler: Option<(usize, f32)> = None;
                for (i, (lo, _)) in parent.objective_bounds.iter().enumerate() {
                    if parent.verified.get(i).copied().unwrap_or(false) {
                        continue;
                    }
                    let lo = if lo.is_nan() { f32::NEG_INFINITY } else { *lo };
                    if straggler.is_none_or(|(_, w)| lo < w) {
                        straggler = Some((i, lo));
                    }
                }
                let Some((s_idx, parent_lb)) = straggler else {
                    continue;
                };
                // Post-split bound = min over children of the child's LB on the
                // straggler objective (the partition's effective bound on `s_idx`).
                let mut post = f32::INFINITY;
                for (child, _) in children_vec {
                    if let Some((lo, _)) = child.objective_bounds.get(s_idx) {
                        let lo = if lo.is_nan() { f32::NEG_INFINITY } else { *lo };
                        post = post.min(lo);
                    }
                }
                // The chosen split is the first constraint appended after the
                // exact parent prefix. A multi-depth cover appends deeper
                // constraints too, so `.last()` would misreport its final
                // descendant split as the selector's committed root split.
                let (node, neuron) = common_committed_cover_root_constraint(
                    &parent.history().constraints,
                    children_vec
                        .iter()
                        .map(|(child, _)| child.history().constraints.as_slice()),
                )
                .map(|c| (c.node_name.clone(), c.neuron_idx))
                .unwrap_or_else(|| ("?".to_string(), usize::MAX));
                eprintln!(
                    "[branch-trace] depth={} node={} neuron={} straggler={} parent_lb={:.5} post_lb={:.5} lift={:.5} nchild={} nunstable={}",
                    parent.depth,
                    node,
                    neuron,
                    s_idx,
                    parent_lb,
                    post,
                    post - parent_lb,
                    children_vec.len(),
                    unstable_count.get(parent_idx).copied().unwrap_or(0),
                );
            }
        }

        // Assemble final results
        for (parent_idx, _, _) in &domains_with_unstable {
            // Final strict publication check for every parent whose exhaustive
            // cover consumed a typed KFSB effect. This includes partial-row
            // children: their fresh pass omitted the certified row, so a late
            // result still depends on the receipt and must become Timeout.
            if kfsb_cert_parent_publication_expired(
                kfsb_final_publication_now(),
                *parent_idx,
                &kfsb_certified_parent_deadlines,
            ) {
                quick_results.insert(
                    *parent_idx,
                    MultiObjectiveGraphDomainResult::DeadlineExpired,
                );
                continue;
            }
            // Branch selection failures are already in quick_results (#2143).
            if branch_selection_failures.contains(parent_idx) {
                continue;
            }
            if let Some(terminal) = terminal_multi_objective_parent_result(
                parents_with_deadline.contains(parent_idx),
                parents_with_failure.contains(parent_idx),
            ) {
                quick_results.insert(*parent_idx, terminal);
            } else if let Some(children) = children_by_parent.remove(parent_idx) {
                // #violdrop: SURVIVING SIBLINGS ARE NEVER DISCARDED. This arm
                // used to sit BELOW the violation arm, so a single dropped child
                // replaced the whole result with `Violation` and threw away every
                // sibling that had been successfully bounded. The drop is still
                // recorded (so the verdict cannot claim `Verified` for the
                // abandoned sub-region) — it just no longer takes the siblings
                // with it. Mirrors the sequential lane's per-child
                // `ChildOutcome::Dropped`.
                let result = if parents_with_violation.contains(parent_idx) {
                    MultiObjectiveGraphDomainResult::ChildrenWithViolatedDrop(children)
                } else {
                    MultiObjectiveGraphDomainResult::Children(children)
                };
                quick_results.insert(*parent_idx, result);
            } else if parents_with_violation.contains(parent_idx) {
                // #1861: every child violated (no survivor to enqueue) — track as
                // violation instead of silently dropping.
                quick_results.insert(*parent_idx, MultiObjectiveGraphDomainResult::Violation);
            } else {
                // Both children infeasible (with_constraint returned None for both).
                // This is a legitimate outcome, not an internal failure (#2143).
                tracing::debug!(
                    "process_graph_domains_batched_gpu_multi_objective: both children infeasible for parent idx {} (#2143)",
                    parent_idx
                );
                quick_results.insert(
                    *parent_idx,
                    MultiObjectiveGraphDomainResult::PropagationFailure,
                );
            }
        }

        // Return results in order
        let results = (0..domains.len())
            .map(|idx| {
                quick_results.remove(&idx).unwrap_or_else(|| {
                    tracing::warn!(
                        "process_graph_domains_batched_gpu_multi_objective: missing final result for idx {} (#1993)",
                        idx
                    );
                    MultiObjectiveGraphDomainResult::PropagationFailure
                })
            })
            .collect();
        publish_bounded_wave_results(bounded_shared_lane, engine, results)
    }
}

#[cfg(test)]
mod mo_gpu_chunk_tests {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use ny_core::{
        GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, NyError, Result as NyResult,
    };

    use super::super::children::{KfsbCertEffect, KfsbCertReceipt, KfsbCertScope};
    use super::{
        admit_multi_objective_child_wave, batched_child_refusal_has_cpu_fallback,
        branch_selector_may_use_engine, child_uses_analytical_beta_optimizer,
        child_uses_dense_spec_adapter, committed_cover_root_constraint,
        common_committed_cover_root_constraint, cut_pool_for_lane, deadline_chunk_override_enabled,
        domain_state_may_branch, gpu_single_pass_backend_eligible, inherit_bounded_child_bounds,
        kfsb_cert_effect_matches_child, kfsb_cert_parent_publication_expired,
        maybe_run_unbounded_advisory, mo_batch_chunk_start_allowed, mo_gpu_chunk_start_allowed,
        multi_objective_batch_layout_is_valid, parse_mo_gpu_single_pass_chunk,
        partition_kfsb_certified_children, publish_bounded_wave_results,
        terminal_multi_objective_parent_result, KfsbCertAccounting, MultiObjectiveChildEvalError,
        MultiObjectiveGraphDomainResult, MAX_BOUNDED_SHARED_HISTORY_CONSTRAINTS,
        MO_GPU_AUTHORITY_RESERVE, MO_GPU_SINGLE_PASS_CHUNK,
    };
    use crate::beta_crown::bab_cuts::GraphCutPool;
    use crate::beta_crown::branching::GraphNeuronConstraint;

    fn trace_constraint(node: &str, neuron: usize, is_active: bool) -> GraphNeuronConstraint {
        GraphNeuronConstraint::new(node.to_string(), neuron, is_active, 1.0)
            .expect("finite synthetic trace constraint")
    }

    #[test]
    fn committed_cover_trace_reports_first_new_split_after_parent_prefix() {
        let parent = vec![trace_constraint("old", 3, true)];
        let mut child = parent.clone();
        child.extend([
            trace_constraint("root", 50, false),
            trace_constraint("depth_2", 794, true),
            trace_constraint("depth_3", 11, false),
            trace_constraint("depth_4", 12, true),
        ]);

        let committed = committed_cover_root_constraint(&parent, &child)
            .expect("valid cover has a committed root split");
        assert_eq!(committed.node_name, "root");
        assert_eq!(committed.neuron_idx, 50);
        assert!(!committed.is_active);
    }

    #[test]
    fn committed_cover_trace_rejects_missing_or_mismatched_suffix() {
        let parent = vec![trace_constraint("old", 3, true)];
        assert!(committed_cover_root_constraint(&parent, &parent).is_none());

        let malformed = vec![
            trace_constraint("different", 3, true),
            trace_constraint("root", 50, false),
        ];
        assert!(committed_cover_root_constraint(&parent, &malformed).is_none());
    }

    #[test]
    fn committed_cover_trace_accepts_reversed_phases_but_rejects_mixed_roots() {
        let parent = vec![trace_constraint("old", 3, true)];
        let mut inactive = parent.clone();
        inactive.extend([
            trace_constraint("root", 50, false),
            trace_constraint("deep", 794, true),
        ]);
        let mut active = parent.clone();
        active.extend([
            trace_constraint("root", 50, true),
            trace_constraint("deep", 795, false),
        ]);

        for children in [
            vec![inactive.as_slice(), active.as_slice()],
            vec![active.as_slice(), inactive.as_slice()],
        ] {
            let committed = common_committed_cover_root_constraint(&parent, children)
                .expect("phase order does not change the common root split");
            assert_eq!(
                (committed.node_name.as_str(), committed.neuron_idx),
                ("root", 50)
            );
        }

        let mut mixed = parent.clone();
        mixed.push(trace_constraint("other_root", 51, true));
        assert!(common_committed_cover_root_constraint(
            &parent,
            [inactive.as_slice(), mixed.as_slice()],
        )
        .is_none());
    }

    #[test]
    fn batch_layout_gate_precedes_cached_verification_authority() {
        let input = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .unwrap();
        let mut domain = crate::beta_crown::domain::MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            vec![(-1.0, 1.0)],
            &input,
            &[0.0],
            false,
        )
        .unwrap();
        let objectives = vec![vec![1.0_f32]];
        assert!(multi_objective_batch_layout_is_valid(
            &domain,
            &objectives,
            &[0.0]
        ));
        assert!(!multi_objective_batch_layout_is_valid(
            &domain,
            &objectives,
            &[]
        ));

        domain.verified.clear();
        assert!(!multi_objective_batch_layout_is_valid(
            &domain,
            &objectives,
            &[0.0]
        ));
        domain.verified.push(true);
        domain.objective_bounds[0] = (1.0, 0.0);
        assert!(!multi_objective_batch_layout_is_valid(
            &domain,
            &objectives,
            &[0.0]
        ));
    }

    #[test]
    fn kfsb_receipt_partition_distinguishes_partial_complete_and_none() {
        let input = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .unwrap();
        let thresholds = [0.0, 0.0];
        let make = |bounds| {
            crate::beta_crown::domain::MultiObjectiveGraphBabDomain::root(
                std::collections::HashMap::new(),
                bounds,
                &input,
                &thresholds,
                false,
            )
            .unwrap()
        };
        let append_test_split =
            |domain: &mut crate::beta_crown::domain::MultiObjectiveGraphBabDomain,
             neuron_idx: usize,
             is_active: bool| {
                domain.history.add_constraint(
                    GraphNeuronConstraint::new(
                        "receipt_test_relu".to_string(),
                        neuron_idx,
                        is_active,
                        1.0,
                    )
                    .expect("finite synthetic split"),
                );
                domain.depth += 1;
            };
        let partial_source = make(vec![(-0.5, 1.0), (-0.5, 1.0)]);
        let mut partial = partial_source.clone();
        partial
            .update_bounds(vec![(0.5, 1.0), (-0.5, 1.0)], &thresholds, false)
            .expect("partial child proof bounds");
        append_test_split(&mut partial, 0, true);
        let complete_source = make(vec![(-0.5, 1.0), (0.25, 1.0)]);
        let mut complete = complete_source.clone();
        complete
            .update_bounds(vec![(0.5, 1.0), (0.25, 1.0)], &thresholds, false)
            .expect("complete child proof bounds");
        append_test_split(&mut complete, 1, false);
        let ordinary_complete = make(vec![(0.5, 1.0), (0.25, 1.0)]);
        let parent_close_source = make(vec![(-0.5, 1.0), (0.25, 1.0)]);
        let mut parent_close = parent_close_source.clone();
        parent_close
            .update_bounds(vec![(0.5, 1.0), (0.25, 1.0)], &thresholds, false)
            .expect("parent-wide proof bounds");
        let now = Instant::now();
        let deadline = now + Duration::from_secs(1);
        let receipt =
            |parent: &crate::beta_crown::domain::MultiObjectiveGraphBabDomain,
             child: &crate::beta_crown::domain::MultiObjectiveGraphBabDomain| {
                KfsbCertReceipt {
                    row: 0,
                    scope: KfsbCertScope::ParentCover,
                    parent_history_identity: parent
                        .history()
                        .exact_provenance_identity()
                        .expect("test root has bounded exact provenance")
                        .into(),
                    lower_bits: child.objective_bounds()[0].0.to_bits(),
                    authority_deadline: deadline,
                }
            };
        let parent_lookup = std::collections::HashMap::from([
            (7, &partial_source),
            (8, &complete_source),
            (9, &ordinary_complete),
            (10, &parent_close_source),
        ]);

        let partition = partition_kfsb_certified_children(
            now,
            &thresholds,
            &parent_lookup,
            vec![
                (
                    7,
                    vec![(
                        7,
                        partial.clone(),
                        true,
                        KfsbCertEffect::RowVerified(receipt(&partial_source, &partial)),
                    )],
                ),
                (
                    8,
                    vec![(
                        8,
                        complete.clone(),
                        false,
                        KfsbCertEffect::ChildComplete(receipt(&complete_source, &complete)),
                    )],
                ),
                (
                    9,
                    vec![(9, ordinary_complete.clone(), true, KfsbCertEffect::None)],
                ),
                (
                    10,
                    vec![(
                        10,
                        parent_close.clone(),
                        false,
                        KfsbCertEffect::ParentComplete(receipt(
                            &parent_close_source,
                            &parent_close,
                        )),
                    )],
                ),
            ],
        );
        assert!(partition.expired_parents.is_empty());
        assert!(partition.invalid_parents.is_empty());
        assert_eq!(partition.accounting.input_parents, 4);
        assert_eq!(partition.accounting.input_leaves, 4);
        assert_eq!(partition.accounting.partial_receipts, 1);
        assert_eq!(partition.accounting.complete_receipts, 1);
        assert_eq!(partition.accounting.parent_closes, 1);
        assert_eq!(partition.accounting.ordinary_preverified_leaves, 1);
        assert_eq!(partition.accounting.pruned_spec_rows, 3);
        assert_eq!(partition.accounting.rejected_parents(), 0);
        assert_eq!(partition.accounting.rejected_leaves(), 0);
        assert_eq!(
            partition.certified_parent_deadlines.get(&7),
            Some(&deadline)
        );
        assert_eq!(
            partition.certified_parent_deadlines.get(&8),
            Some(&deadline)
        );
        assert!(!kfsb_cert_parent_publication_expired(
            now,
            7,
            &partition.certified_parent_deadlines,
        ));
        assert!(kfsb_cert_parent_publication_expired(
            deadline,
            7,
            &partition.certified_parent_deadlines,
        ));
        assert!(!kfsb_cert_parent_publication_expired(
            deadline,
            9,
            &partition.certified_parent_deadlines,
        ));
        let pending: Vec<_> = partition
            .pending
            .into_iter()
            .flat_map(|(_, children)| children)
            .collect();
        assert_eq!(pending.len(), 1);
        assert!(pending.iter().any(|(parent, _, _, effect)| {
            *parent == 7 && matches!(effect, KfsbCertEffect::RowVerified(_))
        }));
        assert_eq!(partition.preverified.get(&8).map(Vec::len), Some(1));
        assert!(partition.preverified[&8][0].1);
        assert_eq!(partition.preverified.get(&9).map(Vec::len), Some(1));
        assert!(partition.preverified[&9][0].0.all_verified());
        assert!(partition.preverified[&9][0].1);
        assert_eq!(partition.preverified.get(&10).map(Vec::len), Some(1));
        assert_eq!(
            partition.preverified[&10][0].0.objective_bounds()[1],
            parent_close_source.objective_bounds()[1],
            "non-target row must remain bit-identical"
        );
    }

    #[test]
    fn kfsb_receipt_validator_requires_shared_cache_identity() {
        let input = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .unwrap();
        let thresholds = [0.0, 0.0];
        let mut parent = crate::beta_crown::domain::MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            vec![(-0.5, 1.0), (-0.5, 1.0)],
            &input,
            &thresholds,
            false,
        )
        .unwrap();
        let mut receipt_cache = crate::batched_domain::CachedLinearBounds::default();
        receipt_cache
            .lower_b
            .insert("receipt_arc_probe".to_string(), ndarray::arr1(&[7.25_f32]));
        parent
            .set_cached_las(vec![
                Some(receipt_cache),
                Some(crate::batched_domain::CachedLinearBounds::default()),
            ])
            .expect("full-spec cache shape");

        let mut child = parent.clone();
        child
            .update_bounds(vec![(0.5, 1.0), (-0.5, 1.0)], &thresholds, false)
            .expect("partial child proof bounds");
        child.history.add_constraint(
            GraphNeuronConstraint::new("receipt_test_relu".to_string(), 0, true, 1.0)
                .expect("finite synthetic split"),
        );
        child.depth += 1;
        let parent_identity = parent
            .history()
            .exact_provenance_identity()
            .expect("bounded parent identity");
        let effect = KfsbCertEffect::RowVerified(KfsbCertReceipt {
            row: 0,
            scope: KfsbCertScope::LiteralSide {
                node_name: "receipt_test_relu".to_string(),
                neuron_idx: 0,
                is_active: true,
            },
            parent_history_identity: Arc::from(parent_identity.as_slice()),
            lower_bits: child.objective_bounds()[0].0.to_bits(),
            authority_deadline: Instant::now() + Duration::from_secs(1),
        });

        assert!(Arc::ptr_eq(
            parent.cached_las()[0].as_ref().expect("parent cache"),
            child.cached_las()[0].as_ref().expect("shared child cache")
        ));
        assert!(kfsb_cert_effect_matches_child(
            &parent,
            &parent_identity,
            &child,
            &effect,
            &thresholds,
        ));

        let cloned_cache = child.cached_las()[0]
            .as_ref()
            .expect("shared child cache")
            .as_ref()
            .clone();
        child.cached_las[0] = Some(Arc::new(cloned_cache));
        assert_eq!(
            child.cached_las()[0]
                .as_ref()
                .expect("cloned cache")
                .lower_b["receipt_arc_probe"][0]
                .to_bits(),
            7.25_f32.to_bits()
        );
        assert!(
            !kfsb_cert_effect_matches_child(
                &parent,
                &parent_identity,
                &child,
                &effect,
                &thresholds,
            ),
            "bit-identical deep clone must not masquerade as inherited Arc transport"
        );
    }

    #[test]
    fn kfsb_receipt_accounting_counts_accepted_rows_and_rejected_covers() {
        let receipt = KfsbCertReceipt {
            row: 0,
            scope: KfsbCertScope::ParentCover,
            parent_history_identity: Arc::from(&b"accounting"[..]),
            lower_bits: 1.0_f32.to_bits(),
            authority_deadline: Instant::now() + Duration::from_secs(1),
        };
        let mut accounting = KfsbCertAccounting::default();
        accounting.observe_input_parent(4);
        accounting.observe_accepted_effect(&KfsbCertEffect::None);
        accounting.observe_accepted_effect(&KfsbCertEffect::RowVerified(receipt.clone()));
        accounting.observe_accepted_effect(&KfsbCertEffect::ChildComplete(receipt.clone()));
        accounting.observe_accepted_effect(&KfsbCertEffect::ParentComplete(receipt));
        accounting.observe_ordinary_preverified();
        accounting.observe_input_parent(2);
        accounting.observe_invalid_parent(2);
        accounting.observe_input_parent(4);
        accounting.observe_expired_parent(4);

        assert_eq!(accounting.input_parents, 3);
        assert_eq!(accounting.input_leaves, 10);
        assert_eq!(accounting.partial_receipts, 1);
        assert_eq!(accounting.complete_receipts, 1);
        assert_eq!(accounting.parent_closes, 1);
        assert_eq!(accounting.ordinary_preverified_leaves, 1);
        assert_eq!(accounting.pruned_spec_rows, 3);
        assert_eq!(accounting.invalid_parents, 1);
        assert_eq!(accounting.invalid_leaves, 2);
        assert_eq!(accounting.expired_parents, 1);
        assert_eq!(accounting.expired_leaves, 4);
        assert_eq!(accounting.rejected_parents(), 2);
        assert_eq!(accounting.rejected_leaves(), 6);
        assert_eq!(
            accounting.input_leaves,
            1 + 3 + accounting.rejected_leaves(),
            "one pending leaf plus three preverified leaves account for the accepted cover"
        );
    }

    #[test]
    fn kfsb_parent_complete_must_be_the_single_parent_entry() {
        let input = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .unwrap();
        let thresholds = [0.0];
        let parent = crate::beta_crown::domain::MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            vec![(-0.5, 1.0)],
            &input,
            &thresholds,
            false,
        )
        .unwrap();
        let mut close = parent.clone();
        close
            .update_bounds(vec![(0.5, 1.0)], &thresholds, false)
            .expect("parent close bounds");
        let now = Instant::now();
        let effect = KfsbCertEffect::ParentComplete(KfsbCertReceipt {
            row: 0,
            scope: KfsbCertScope::ParentCover,
            parent_history_identity: parent
                .history()
                .exact_provenance_identity()
                .expect("bounded root identity")
                .into(),
            lower_bits: close.objective_bounds()[0].0.to_bits(),
            authority_deadline: now + Duration::from_secs(1),
        });
        let parent_lookup = std::collections::HashMap::from([(17, &parent)]);
        let partition = partition_kfsb_certified_children(
            now,
            &thresholds,
            &parent_lookup,
            vec![(
                17,
                vec![
                    (17, close, false, effect),
                    (17, parent.clone(), true, KfsbCertEffect::None),
                ],
            )],
        );

        assert_eq!(partition.invalid_parents, HashSet::from([17]));
        assert!(partition.pending.is_empty());
        assert!(partition.preverified.is_empty());
        assert!(partition.certified_parent_deadlines.is_empty());
        assert_eq!(partition.accounting.invalid_parents, 1);
        assert_eq!(partition.accounting.invalid_leaves, 2);
    }

    #[test]
    fn kfsb_receipt_partition_rejects_expired_and_malformed_parents_atomically() {
        let input = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .unwrap();
        let thresholds = [0.0, 0.0];
        let parent = crate::beta_crown::domain::MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            vec![(-0.5, 1.0), (-0.5, 1.0)],
            &input,
            &thresholds,
            false,
        )
        .unwrap();
        let mut partial = parent.clone();
        partial
            .update_bounds(vec![(0.5, 1.0), (-0.5, 1.0)], &thresholds, false)
            .expect("synthetic child proof bounds");
        partial.history.add_constraint(
            GraphNeuronConstraint::new("receipt_test_relu".to_string(), 0, true, 1.0)
                .expect("finite synthetic split"),
        );
        partial.depth += 1;
        let now = Instant::now();
        let parent_history_identity: Arc<[u8]> = parent
            .history()
            .exact_provenance_identity()
            .expect("test root has bounded exact provenance")
            .into();
        let receipt = |deadline, lower_bits| KfsbCertReceipt {
            row: 0,
            scope: KfsbCertScope::ParentCover,
            parent_history_identity: parent_history_identity.clone(),
            lower_bits,
            authority_deadline: deadline,
        };
        let parent_lookup: std::collections::HashMap<_, _> =
            (10..=16).map(|parent_idx| (parent_idx, &parent)).collect();
        let partition = partition_kfsb_certified_children(
            now,
            &thresholds,
            &parent_lookup,
            vec![
                (
                    10,
                    vec![(
                        10,
                        partial.clone(),
                        true,
                        KfsbCertEffect::RowVerified(receipt(
                            now + Duration::from_secs(1),
                            partial.objective_bounds()[0].0.to_bits(),
                        )),
                    )],
                ),
                (
                    11,
                    vec![(
                        11,
                        partial.clone(),
                        true,
                        KfsbCertEffect::RowVerified(receipt(
                            now,
                            partial.objective_bounds()[0].0.to_bits(),
                        )),
                    )],
                ),
                (
                    12,
                    vec![
                        (12, partial.clone(), true, KfsbCertEffect::None),
                        (
                            12,
                            partial.clone(),
                            false,
                            KfsbCertEffect::RowVerified(receipt(
                                now + Duration::from_secs(1),
                                123_u32,
                            )),
                        ),
                    ],
                ),
                (
                    13,
                    vec![
                        (
                            13,
                            partial.clone(),
                            true,
                            KfsbCertEffect::RowVerified(receipt(
                                now + Duration::from_secs(1),
                                partial.objective_bounds()[0].0.to_bits(),
                            )),
                        ),
                        // Once any sibling carries authority, provenance for
                        // the whole exhaustive cover must agree on its parent.
                        (99, partial.clone(), false, KfsbCertEffect::None),
                    ],
                ),
                (
                    14,
                    vec![
                        (
                            14,
                            partial.clone(),
                            true,
                            KfsbCertEffect::RowVerified(receipt(
                                now + Duration::from_secs(1),
                                partial.objective_bounds()[0].0.to_bits(),
                            )),
                        ),
                        (
                            14,
                            partial.clone(),
                            false,
                            KfsbCertEffect::RowVerified(receipt(
                                now + Duration::from_secs(2),
                                0.5_f32.to_bits(),
                            )),
                        ),
                    ],
                ),
                (
                    15,
                    vec![(
                        15,
                        partial.clone(),
                        true,
                        KfsbCertEffect::RowVerified(KfsbCertReceipt {
                            scope: KfsbCertScope::LiteralSide {
                                node_name: "not-in-child-history".to_string(),
                                neuron_idx: 7,
                                is_active: true,
                            },
                            ..receipt(
                                now + Duration::from_secs(1),
                                partial.objective_bounds()[0].0.to_bits(),
                            )
                        }),
                    )],
                ),
                (
                    16,
                    vec![(
                        16,
                        partial.clone(),
                        false,
                        KfsbCertEffect::RowVerified(KfsbCertReceipt {
                            parent_history_identity: Arc::from(&b"wrong-parent-history"[..]),
                            ..receipt(
                                now + Duration::from_secs(1),
                                partial.objective_bounds()[0].0.to_bits(),
                            )
                        }),
                    )],
                ),
            ],
        );
        assert_eq!(partition.expired_parents, HashSet::from([11]));
        assert_eq!(
            partition.invalid_parents,
            HashSet::from([12, 13, 14, 15, 16])
        );
        assert_eq!(partition.pending.len(), 1);
        assert_eq!(partition.pending[0].0, 10);
        assert_eq!(partition.pending[0].1.len(), 1);
        assert!(partition.preverified.is_empty());
        assert_eq!(
            partition.certified_parent_deadlines.get(&10),
            Some(&(now + Duration::from_secs(1)))
        );
        assert_eq!(partition.accounting.input_parents, 7);
        assert_eq!(partition.accounting.input_leaves, 10);
        assert_eq!(partition.accounting.partial_receipts, 1);
        assert_eq!(partition.accounting.complete_receipts, 0);
        assert_eq!(partition.accounting.parent_closes, 0);
        assert_eq!(partition.accounting.ordinary_preverified_leaves, 0);
        assert_eq!(partition.accounting.pruned_spec_rows, 1);
        assert_eq!(partition.accounting.invalid_parents, 5);
        assert_eq!(partition.accounting.invalid_leaves, 8);
        assert_eq!(partition.accounting.expired_parents, 1);
        assert_eq!(partition.accounting.expired_leaves, 1);
        assert_eq!(partition.accounting.rejected_parents(), 6);
        assert_eq!(partition.accounting.rejected_leaves(), 9);
    }

    struct SinglePassGpu {
        sound: bool,
        honors_deadline: bool,
    }

    struct ExpiredWaveEngine;

    impl GemmEngine for ExpiredWaveEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> NyResult<Vec<f32>> {
            unreachable!("publication test must not enter GEMM")
        }

        fn poll_crown_backward_deadline(&self) -> NyResult<()> {
            Err(NyError::DeadlineExceeded(
                "injected final bounded-wave expiry".into(),
            ))
        }

        fn forbids_unbounded_cpu_fallback(&self) -> bool {
            true
        }
    }

    #[test]
    fn bounded_lane_never_enters_dense_spec_adapter() {
        for gpu_single_pass in [false, true] {
            for cpu_beta_applies in [false, true] {
                assert!(
                    !child_uses_dense_spec_adapter(true, gpu_single_pass, cpu_beta_applies),
                    "bounded facade must reach constrained per-child backward"
                );
                assert_eq!(
                    child_uses_dense_spec_adapter(false, gpu_single_pass, cpu_beta_applies),
                    gpu_single_pass || !cpu_beta_applies,
                    "legacy routing must remain unchanged"
                );
            }
        }
    }

    #[test]
    fn bounded_lane_withholds_engine_from_speculative_branch_selector() {
        assert!(!branch_selector_may_use_engine(true));
        assert!(branch_selector_may_use_engine(false));
    }

    #[test]
    fn bounded_lane_never_observes_a_cut_pool() {
        let pool = GraphCutPool::default();
        assert!(cut_pool_for_lane(true, Some(&pool)).is_none());
        assert!(cut_pool_for_lane(false, Some(&pool)).is_some());
        assert!(cut_pool_for_lane(false, None).is_none());
    }

    #[test]
    fn bounded_lane_caps_mandatory_history_and_beta_state() {
        assert!(domain_state_may_branch(
            true,
            MAX_BOUNDED_SHARED_HISTORY_CONSTRAINTS - 1
        ));
        assert!(!domain_state_may_branch(
            true,
            MAX_BOUNDED_SHARED_HISTORY_CONSTRAINTS
        ));
        assert!(domain_state_may_branch(false, usize::MAX));
    }

    #[test]
    fn bounded_lane_forces_standard_pass_even_when_beta_optimizer_applies() {
        assert!(!child_uses_analytical_beta_optimizer(true, true));
        assert!(!child_uses_analytical_beta_optimizer(true, false));
        assert!(child_uses_analytical_beta_optimizer(false, true));
        assert!(!child_uses_analytical_beta_optimizer(false, false));
    }

    #[test]
    fn bounded_lane_never_invokes_kfsb_or_complete_clip_advisory_callbacks() {
        let calls = std::cell::Cell::new(0usize);
        let result = maybe_run_unbounded_advisory(true, true, || {
            calls.set(calls.get() + 1);
            7usize
        });
        assert_eq!(result, None);
        let clip_result = maybe_run_unbounded_advisory(true, true, || {
            calls.set(calls.get() + 1);
        });
        assert_eq!(clip_result, None);
        assert_eq!(calls.get(), 0);

        assert_eq!(
            maybe_run_unbounded_advisory(false, true, || {
                calls.set(calls.get() + 1);
                7usize
            }),
            Some(7)
        );
        assert_eq!(calls.get(), 1);
        assert_eq!(maybe_run_unbounded_advisory(false, false, || 9usize), None);
    }

    #[test]
    fn bounded_publication_keeps_parent_lower_and_safe_upper_on_crossing() {
        let parent = [(-5.0_f32, 5.0_f32), (-3.0, 7.0)];
        let fresh = vec![(-10.0_f32, -6.0_f32), (-2.0, 4.0)];
        let published = inherit_bounded_child_bounds(&parent, fresh);

        assert_eq!(published, vec![(-5.0, 5.0), (-2.0, 7.0)]);
        assert!(published
            .iter()
            .zip(parent)
            .all(|(&(lower, upper), (parent_lower, _))| {
                lower >= parent_lower && lower <= upper
            }));
    }

    #[test]
    fn expired_final_bounded_wave_discards_all_completed_publications() {
        let results = publish_bounded_wave_results(
            true,
            &ExpiredWaveEngine,
            vec![
                MultiObjectiveGraphDomainResult::AlreadyVerified,
                MultiObjectiveGraphDomainResult::PropagationFailure,
            ],
        );
        assert!(results
            .iter()
            .all(|result| matches!(result, MultiObjectiveGraphDomainResult::DeadlineExpired)));
    }

    impl GpuCrownBackward for SinglePassGpu {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> NyResult<GpuCrownResult> {
            Err(NyError::UnsupportedOp("routing-only test backend".into()))
        }

        fn provides_sound_gpu_crown(&self) -> bool {
            self.sound
        }

        fn honors_crown_backward_deadline(&self) -> bool {
            self.honors_deadline
        }
    }

    #[test]
    fn finite_single_pass_precheck_matches_the_actual_backend_contract() {
        let finite = Some(Instant::now() + Duration::from_secs(1));
        for (sound, honors_deadline, deadline, expected) in [
            (false, false, None, false),
            (false, true, None, false),
            (true, false, None, true),
            (true, true, None, true),
            (false, false, finite, false),
            (false, true, finite, false),
            (true, false, finite, false),
            (true, true, finite, true),
        ] {
            let gpu = SinglePassGpu {
                sound,
                honors_deadline,
            };
            assert_eq!(
                gpu_single_pass_backend_eligible(&gpu, deadline),
                expected,
                "sound={sound} honors_deadline={honors_deadline} finite={}",
                deadline.is_some()
            );
        }
    }

    #[test]
    fn unset_uses_the_scored_default() {
        assert_eq!(
            parse_mo_gpu_single_pass_chunk(None),
            Some(MO_GPU_SINGLE_PASS_CHUNK)
        );
    }

    #[test]
    fn deadline_chunk_override_remains_an_exact_one_opt_in() {
        use ny_test_utils::env::{lock_env, ScopedEnvVar};

        let _env = lock_env();
        {
            let _unset = ScopedEnvVar::unset("NY_MO_GPU_CHUNK_DEADLINE");
            assert!(!deadline_chunk_override_enabled());
        }
        for value in ["", "0", "true", "yes", "2"] {
            let _set = ScopedEnvVar::set("NY_MO_GPU_CHUNK_DEADLINE", value);
            assert!(
                !deadline_chunk_override_enabled(),
                "{value:?} must not widen an authoritative-deadline chunk"
            );
        }
        let _on = ScopedEnvVar::set("NY_MO_GPU_CHUNK_DEADLINE", "1");
        assert!(deadline_chunk_override_enabled());
    }

    #[test]
    fn scored_chunk_requires_full_authority_reserve() {
        let now = Instant::now();
        assert!(mo_gpu_chunk_start_allowed(now, None));
        assert!(!mo_gpu_chunk_start_allowed(
            now,
            Some(now + Duration::from_secs(5))
        ));
        assert!(mo_gpu_chunk_start_allowed(
            now,
            Some(now + Duration::from_secs(6))
        ));
    }

    #[test]
    fn authority_reserve_applies_only_to_gpu_lane_but_deadline_applies_to_both() {
        let now = Instant::now();
        let short_deadline = (now + MO_GPU_AUTHORITY_RESERVE)
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond fits within the GPU authority reserve");

        assert!(
            !mo_batch_chunk_start_allowed(now, Some(short_deadline), true),
            "GPU work must preserve the full overrun reserve"
        );
        assert!(
            mo_batch_chunk_start_allowed(now, Some(short_deadline), false),
            "CPU work remains admissible before the literal deadline"
        );
        assert!(!mo_batch_chunk_start_allowed(now, Some(now), true));
        assert!(!mo_batch_chunk_start_allowed(now, Some(now), false));
        assert!(mo_batch_chunk_start_allowed(now, None, true));
        assert!(mo_batch_chunk_start_allowed(now, None, false));
    }

    #[test]
    fn expired_cpu_fallback_wave_is_typed_without_launching_work() {
        let now = Instant::now();
        let parent_ids = [7, 7, 9];
        let positions = [0, 2];
        let mut parents_with_deadline = HashSet::new();
        let mut child_bounds: Vec<Option<Result<(), MultiObjectiveChildEvalError>>> =
            vec![None, None, None];

        assert!(!admit_multi_objective_child_wave(
            now,
            Some(now),
            false,
            &positions,
            &parent_ids,
            &mut parents_with_deadline,
            &mut child_bounds,
        ));
        assert_eq!(parents_with_deadline, HashSet::from([7, 9]));
        assert!(matches!(
            child_bounds[0],
            Some(Err(MultiObjectiveChildEvalError::DeadlineExpired))
        ));
        assert!(child_bounds[1].is_none());
        assert!(matches!(
            child_bounds[2],
            Some(Err(MultiObjectiveChildEvalError::DeadlineExpired))
        ));
    }

    #[test]
    fn cpu_fallback_wave_remains_admissible_inside_gpu_reserve() {
        let now = Instant::now();
        let parent_ids = [7];
        let mut parents_with_deadline = HashSet::new();
        let mut child_bounds: Vec<Option<Result<(), MultiObjectiveChildEvalError>>> = vec![None];

        assert!(admit_multi_objective_child_wave(
            now,
            Some(now + Duration::from_millis(1)),
            false,
            &[0],
            &parent_ids,
            &mut parents_with_deadline,
            &mut child_bounds,
        ));
        assert!(parents_with_deadline.is_empty());
        assert!(child_bounds[0].is_none());
    }

    #[test]
    fn heterogeneous_gpu_child_refusals_retry_exactly_when_cpu_authority_exists() {
        assert!(batched_child_refusal_has_cpu_fallback(
            false,
            Some(MultiObjectiveChildEvalError::PropagationFailure),
        ));
        for terminal in [
            None,
            Some(MultiObjectiveChildEvalError::Infeasible),
            Some(MultiObjectiveChildEvalError::DeadlineExpired),
        ] {
            assert!(
                !batched_child_refusal_has_cpu_fallback(false, terminal),
                "a terminal/non-refusal disposition must not duplicate work"
            );
        }
        assert!(
            !batched_child_refusal_has_cpu_fallback(
                true,
                Some(MultiObjectiveChildEvalError::PropagationFailure),
            ),
            "the bounded facade explicitly forbids an unbounded CPU retry"
        );
    }

    #[test]
    fn producer_parent_assembly_gives_deadline_partial_result_precedence() {
        let result = terminal_multi_objective_parent_result(true, true)
            .expect("terminal cause must be selected");
        assert!(matches!(
            result,
            MultiObjectiveGraphDomainResult::DeadlineExpired
        ));
    }

    #[test]
    fn positive_native_usize_values_are_accepted() {
        assert_eq!(parse_mo_gpu_single_pass_chunk(Some("128")), Some(128));
        assert_eq!(parse_mo_gpu_single_pass_chunk(Some("00128")), Some(128));
        let native_max = usize::MAX.to_string();
        assert_eq!(
            parse_mo_gpu_single_pass_chunk(Some(&native_max)),
            Some(usize::MAX)
        );
    }

    #[test]
    fn explicitly_invalid_values_disable_the_experimental_lane() {
        for raw in ["", "0", "+64", "-1", " 64", "64 ", "64.0"] {
            assert_eq!(
                parse_mo_gpu_single_pass_chunk(Some(raw)),
                None,
                "{raw:?} must fail closed"
            );
        }
        let overflow = (usize::MAX as u128 + 1).to_string();
        assert_eq!(parse_mo_gpu_single_pass_chunk(Some(&overflow)), None);
    }
}
