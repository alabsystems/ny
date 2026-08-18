// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain-batched single-pass adapter for multi-objective graph BaB.
//!
//! For the single-pass (non-beta-optimizing) branch of multi-objective child
//! propagation, this adapter collapses N independent per-child CROWN backward
//! passes into ONE dense-spec batched backward (L kernel launches, not N×L),
//! reusing the existing dense domain×spec primitive
//! [`BetaCrownVerifier::propagate_crown_with_batched_domains_full_specs`].
//!
//! # Soundness
//!
//! The lower bounds this adapter produces are valid lower bounds on the
//! objective over each child sub-domain (verified independently by concrete
//! sampling in `tests.rs`), so swapping it in for the per-child single-pass path
//! is sound for `verify_upper=false` (where the lower bound drives verification).
//!
//! NOTE — NOT bit-equivalent to the old per-child path. The plan assumed the
//! per-child single-pass closure (`propagate_multi_objective_with_beta_and_cache`
//! -> `backward_crown_constrained`) and the dense-spec batched primitive used
//! here (`propagate_crown_with_batched_domains_full_specs`) compute identical
//! bounds modulo GEMM reassociation. They do NOT: they are two distinct CROWN
//! backward implementations that diverge for non-empty split histories (and when
//! per-domain alpha state is present). Empirically the batched primitive matches
//! the canonical direct CROWN oracle
//! (`propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline`), while
//! the per-child `backward_crown_constrained` is strictly LOOSER. Both are sound;
//! this adapter is the tighter of the two. See the SOUNDNESS FINDING note in
//! `tests.rs` for the empirical evidence.
//!
//! What this adapter preserves vs. the per-child path:
//!
//! * **Forward bounds** — Both run
//!   [`BetaCrownVerifier::compute_constrained_forward_bounds`] with
//!   `(child.input_bounds, child.history, parent.node_bounds)`. The child's
//!   `node_bounds` is `parent.node_bounds.clone()` (inherited verbatim by
//!   `with_constraint`, not yet recomputed), so seeding the shim's `node_bounds`
//!   from `child.node_bounds` reproduces the same base bounds.
//! * **Backward seeding** — Both seed a spec-guided CROWN backward with the same
//!   per-domain `beta_state` and `alpha_state`. The per-child path uses the
//!   *pruned* (unverified-only) spec rows; this adapter uses the *full* spec
//!   matrix but selects only the unverified rows via `active_indices`. CROWN
//!   backward is row-independent across spec rows, so the active-row selection
//!   yields the same per-row result as a pruned spec matrix.
//! * **Verified-objective latch** — `merge_pruned_objective_bounds` keeps the
//!   parent's bounds for already-verified objectives and places fresh batched
//!   bounds only for the unverified ones — exactly the per-child semantics.
//! * **Cuts** — Gated upstream: this adapter only runs when the cut pool is
//!   empty, because the dense-spec batched primitive does not apply cuts.
//! * **No warm-start** — `cached_la = None` and `active_cached_las = vec![None; …]`:
//!   sound (no warm-start), just no lA reuse for the minimal increment.

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{GemmEngine, NyError};
use ny_tensor::BoundedTensor;

use crate::batched_domain::{BatchedDomainOptions, BatchedDomains};
use crate::beta_crown::domain::{GraphBabDomain, MultiObjectiveGraphBabDomain};
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::GraphNetwork;

use super::super::super::super::BetaCrownVerifier;
use super::super::selective_root_alpha::{
    authoritative_deadline_expired, select_child_bounds_and_continuation_state,
    selective_candidate_deadline, selective_candidate_start_allowed, ChildArmEvaluation,
    ChildContinuationStateProvenance, SelectivePairDecision,
};
use super::super::shared::{
    build_spec_matrix, merge_pruned_objective_bounds, prune_verified_multi_objective_targets,
    spec_bounds_to_vec, PrunedMultiObjectiveTargets,
};

thread_local! {
    /// Per-(split pre-node, input-box) cache of the seeded backward input-relative
    /// enclosure — see the THROUGHPUT note in `clip_child_node_bounds`. `None`
    /// caches a known-failed backward. Keyed by input-box signature so it never
    /// leaks across instances/boxes. In ReLU-split BaB the input box is fixed, so
    /// each split pre-node's backward is computed once and reused across all
    /// children (sound: a wider-but-valid enclosure over the same box).
    static CLIP_BWD_CACHE: std::cell::RefCell<HashMap<(String, u64), Option<crate::LinearBounds>>> =
        std::cell::RefCell::new(HashMap::new());

    /// Per-input-box cache of the ROOT IBP node-bounds (valid over the full box),
    /// used as the ReLU-relaxation base for the cached clip backward so the
    /// enclosure is soundly reusable across all children of that box.
    static CLIP_ROOT_BOUNDS_CACHE: std::cell::RefCell<HashMap<u64, Arc<HashMap<String, Arc<BoundedTensor>>>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Cheap order-sensitive signature of the (fixed) input box for the clip backward
/// cache key — hashes the raw f32 bits of the flattened lower/upper bounds.
fn clip_input_box_signature(input: &BoundedTensor) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let flat = input.flatten();
    let lo = flat.lower();
    let up = flat.upper();
    lo.len().hash(&mut h);
    for v in lo.iter() {
        v.to_bits().hash(&mut h);
    }
    for v in up.iter() {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

/// Per-child single-pass result, identical in shape to the per-child closure in
/// `batched_multi.rs`:
/// `(obj_bounds, node_cache, beta_state, alpha_state, active_cached_las,
/// pruned_targets, state_provenance)`.
///
/// The `Option<GraphDomainAlphaState>` slot is the wide α ascent's persisted
/// best-margin α for the child (#hard-six, dark `NY_WIDE_ALPHA_UNSHARED=1`);
/// `None` = keep the child's inherited α (historical behavior, byte-identical
/// when the gate is off).
///
/// Preserve the semantic cause of a per-child failure across the batched and
/// fallback paths. In particular, an effective graph-BaB deadline can be
/// earlier than the outer loop's nominal timeout, so collapsing it into a
/// generic propagation failure can otherwise produce a pre-deadline `Unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) enum MultiObjectiveChildEvalError {
    Infeasible,
    PropagationFailure,
    DeadlineExpired,
}

/// Whole-adapter decline. Ordinary construction/numerical failures may retry
/// through the exact per-child path. Deadline expiry and a bounded facade's
/// host-memory refusal must not launch fresh work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) enum BatchedMultiObjectiveAdapterError {
    Fallback,
    ResourceRefused,
    DeadlineExpired,
}

fn classify_adapter_error(
    error: &NyError,
    engine: &dyn GemmEngine,
) -> BatchedMultiObjectiveAdapterError {
    if error.is_deadline_exceeded() {
        BatchedMultiObjectiveAdapterError::DeadlineExpired
    } else if error.is_cpu_memory_exceeded() && engine.forbids_unbounded_cpu_fallback() {
        BatchedMultiObjectiveAdapterError::ResourceRefused
    } else {
        BatchedMultiObjectiveAdapterError::Fallback
    }
}

fn classify_certified_adapter_fallback_error(error: &NyError) -> BatchedMultiObjectiveAdapterError {
    if error.is_deadline_exceeded() {
        BatchedMultiObjectiveAdapterError::DeadlineExpired
    } else if error.is_cpu_memory_exceeded() {
        BatchedMultiObjectiveAdapterError::ResourceRefused
    } else {
        BatchedMultiObjectiveAdapterError::Fallback
    }
}

fn certified_adapter_checkpoint(
    deadline: Instant,
) -> Result<(), BatchedMultiObjectiveAdapterError> {
    if Instant::now() >= deadline {
        Err(BatchedMultiObjectiveAdapterError::DeadlineExpired)
    } else {
        Ok(())
    }
}

fn certified_adapter_vec<T>(
    len: usize,
    deadline: Instant,
) -> Result<Vec<T>, BatchedMultiObjectiveAdapterError> {
    certified_adapter_checkpoint(deadline)?;
    let required_bytes = len.saturating_mul(size_of::<T>());
    if required_bytes > crate::network::crown_memory::cpu_crown_dense_budget_bytes() {
        return Err(BatchedMultiObjectiveAdapterError::ResourceRefused);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| BatchedMultiObjectiveAdapterError::ResourceRefused)?;
    certified_adapter_checkpoint(deadline)?;
    Ok(values)
}

fn finite_refusal_fallback_admitted(_child_count: usize) -> bool {
    // At this point the adapter still owns a dense spec matrix, graph-domain
    // shims, and `BatchedDomains`' opaque nested buffers. Their live retained
    // payload is not exposed, so even a one-child rescue cannot produce a
    // truthful aggregate memory receipt. Refuse before allocating any fallback
    // result; the higher-level caller retains the child as unresolved.
    false
}

fn prune_targets_for_certified_adapter_fallback(
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    verified_mask: &[bool],
    deadline: Instant,
) -> Result<PrunedMultiObjectiveTargets, BatchedMultiObjectiveAdapterError> {
    if objectives.len() != thresholds.len() || objectives.len() != verified_mask.len() {
        return Err(BatchedMultiObjectiveAdapterError::Fallback);
    }
    let active_count = verified_mask.iter().filter(|&&verified| !verified).count();
    let coefficient_count = objectives
        .iter()
        .zip(verified_mask)
        .filter(|(_, verified)| !**verified)
        .try_fold(0usize, |total, (objective, _)| {
            total.checked_add(objective.len())
        })
        .ok_or(BatchedMultiObjectiveAdapterError::ResourceRefused)?;
    let required_bytes = coefficient_count
        .saturating_mul(size_of::<f32>())
        .saturating_add(active_count.saturating_mul(
            size_of::<usize>() + size_of::<Vec<f32>>() + size_of::<f32>() + size_of::<bool>(),
        ));
    if required_bytes > crate::network::crown_memory::cpu_crown_dense_budget_bytes() {
        return Err(BatchedMultiObjectiveAdapterError::ResourceRefused);
    }

    let mut active_indices = certified_adapter_vec(active_count, deadline)?;
    let mut active_objectives = certified_adapter_vec(active_count, deadline)?;
    let mut active_thresholds = certified_adapter_vec(active_count, deadline)?;
    let mut active_verified_mask = certified_adapter_vec(active_count, deadline)?;
    for (index, ((objective, &threshold), &verified)) in objectives
        .iter()
        .zip(thresholds)
        .zip(verified_mask)
        .enumerate()
    {
        certified_adapter_checkpoint(deadline)?;
        if verified {
            continue;
        }
        let mut cloned_objective = certified_adapter_vec(objective.len(), deadline)?;
        for (coefficient_index, &coefficient) in objective.iter().enumerate() {
            cloned_objective.push(coefficient);
            if coefficient_index.is_multiple_of(1024) {
                certified_adapter_checkpoint(deadline)?;
            }
        }
        active_indices.push(index);
        active_objectives.push(cloned_objective);
        active_thresholds.push(threshold);
        active_verified_mask.push(false);
    }
    certified_adapter_checkpoint(deadline)?;
    Ok(PrunedMultiObjectiveTargets {
        active_indices,
        objectives: active_objectives,
        thresholds: active_thresholds,
        verified_mask: active_verified_mask,
    })
}

fn merge_certified_adapter_bounds(
    inherited: &[(f32, f32)],
    projected: &[(f32, f32)],
    active_indices: &[usize],
    deadline: Instant,
) -> Result<Vec<(f32, f32)>, BatchedMultiObjectiveAdapterError> {
    if inherited.len() != projected.len() {
        return Err(BatchedMultiObjectiveAdapterError::Fallback);
    }
    let mut merged = certified_adapter_vec(inherited.len(), deadline)?;
    for (index, &bounds) in inherited.iter().enumerate() {
        merged.push(bounds);
        if index.is_multiple_of(1024) {
            certified_adapter_checkpoint(deadline)?;
        }
    }
    for &index in active_indices {
        let Some(&bounds) = projected.get(index) else {
            return Err(BatchedMultiObjectiveAdapterError::Fallback);
        };
        merged[index] = bounds;
        certified_adapter_checkpoint(deadline)?;
    }
    Ok(merged)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectiveCandidateCompletion {
    CandidateAvailable,
    RetainEstablished(&'static str),
    HardDeadlineExpired,
}

/// Adjudicate W exactly once at the timestamp sampled immediately after its
/// evaluation returns.
///
/// Deadline authority precedes inspection of W's result. Consequently a
/// `Fallback` cannot mask a crossed hard deadline, and even an `Ok` result
/// finishing in the protected five-second tail is discarded before any W
/// bound, continuation state, or advisory write is read.
fn adjudicate_selective_candidate_completion<T>(
    finished_at: Instant,
    candidate_deadline: Option<Instant>,
    authoritative_deadline: Option<Instant>,
    candidate_call: &Result<T, BatchedMultiObjectiveAdapterError>,
) -> SelectiveCandidateCompletion {
    if authoritative_deadline_expired(finished_at, authoritative_deadline) {
        return SelectiveCandidateCompletion::HardDeadlineExpired;
    }
    if !selective_candidate_start_allowed(finished_at, candidate_deadline) {
        return SelectiveCandidateCompletion::RetainEstablished(
            "candidate_private_deadline_after_evaluation",
        );
    }
    match candidate_call {
        Ok(_) => SelectiveCandidateCompletion::CandidateAvailable,
        Err(BatchedMultiObjectiveAdapterError::Fallback) => {
            SelectiveCandidateCompletion::RetainEstablished("candidate_adapter_refusal")
        }
        Err(BatchedMultiObjectiveAdapterError::ResourceRefused) => {
            SelectiveCandidateCompletion::RetainEstablished("candidate_resource_refusal")
        }
        Err(BatchedMultiObjectiveAdapterError::DeadlineExpired) => {
            SelectiveCandidateCompletion::RetainEstablished("candidate_private_deadline")
        }
    }
}

fn evaluate_selective_candidate_with_scopes<T>(
    verifier: &BetaCrownVerifier,
    candidate_deadline: Option<Instant>,
    evaluate: impl FnOnce() -> T,
) -> (T, Instant) {
    let _candidate_deadline_scope = verifier
        .complete_clip_deadline_overrides
        .scoped(candidate_deadline);
    // Complete-Clipping affine/template caches are keyed without alpha
    // identity. W is speculative and may be rejected, so suppress that
    // optional lane for its entire evaluation rather than letting a W
    // template affect later H descendants.
    let _candidate_complete_clip_suppression = verifier
        .complete_clip_deadline_overrides
        .suppress_complete_clip_scoped();
    let result = evaluate();
    // Sample before either scope is dropped and before the result leaves this
    // seam, so no caller operation can enter the protected tail first.
    let finished_at = Instant::now();
    (result, finished_at)
}

pub(super) type BatchedChildResult = Result<
    (
        Vec<(f32, f32)>,
        HashMap<String, Arc<BoundedTensor>>,
        GraphBetaState,
        Option<GraphDomainAlphaState>,
        Vec<Option<crate::batched_domain::CachedLinearBounds>>,
        PrunedMultiObjectiveTargets,
        ChildContinuationStateProvenance,
    ),
    MultiObjectiveChildEvalError,
>;

#[cfg(test)]
std::thread_local! {
    static BATCHED_SINGLE_PASS_DISPATCH_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_batched_single_pass_dispatch_count_for_test() {
    BATCHED_SINGLE_PASS_DISPATCH_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(super) fn batched_single_pass_dispatch_count_for_test() -> usize {
    BATCHED_SINGLE_PASS_DISPATCH_COUNT.with(std::cell::Cell::get)
}

impl BetaCrownVerifier {
    /// Domain-batch the single-pass multi-objective child CROWN backward passes.
    ///
    /// Produces one result per `batchable` child, in order, matching the tuple
    /// the per-child `par_iter` closure produces in
    /// `process_graph_domains_batched_gpu_multi_objective`. Ordinary errors from
    /// the batched primitive (`BatchedDomains` construction or the dense-spec
    /// backward) return [`BatchedMultiObjectiveAdapterError::Fallback`], and the
    /// caller routes the batchable set through the existing per-child path.
    /// [`BatchedMultiObjectiveAdapterError::DeadlineExpired`] is terminal and
    /// must not launch a fresh fallback wave.
    ///
    /// Returns one result per child on success.
    // Justification: this adapter threads graph, the batchable child set, relu
    // node names, the full objective set, thresholds, and the engine together —
    // the same verification context the per-child path consumes.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn batched_single_pass_multi_objective_children(
        &self,
        graph: &GraphNetwork,
        batchable: &[&MultiObjectiveGraphBabDomain],
        relu_nodes: &[String],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: &dyn GemmEngine,
        prune_specs_to_union: bool,
    ) -> Result<Vec<BatchedChildResult>, BatchedMultiObjectiveAdapterError> {
        #[cfg(test)]
        BATCHED_SINGLE_PASS_DISPATCH_COUNT.with(|count| count.set(count.get() + 1));
        if batchable.is_empty() {
            return Ok(Vec::new());
        }

        // One UNIFORM spec matrix, shape (num_rows, output_dim). CROWN backward
        // is row-independent, so the unverified rows are computed identically to
        // the per-child pruned path.
        //
        // Union pruning (#w5-bab-throughput): the backward's per-layer GEMM cost
        // scales LINEARLY with the number of spec rows, and with 91/99 objectives
        // already root-verified (cifar100) only the union of each child's
        // unverified indices needs recomputation — the verified-latch merge below
        // keeps parent bounds for the rest anyway. Caller-gated (the GPU
        // single-pass lane requests it) so legacy callers keep the full-matrix
        // call byte-identically.
        let union_indices: Vec<usize> = if prune_specs_to_union {
            (0..objectives.len())
                .filter(|&j| {
                    batchable
                        .iter()
                        .any(|child| !child.verified().get(j).copied().unwrap_or(false))
                })
                .collect()
        } else {
            (0..objectives.len()).collect()
        };
        if union_indices.is_empty() {
            // Every objective verified in every child — nothing to recompute;
            // let the per-child path handle the (degenerate) batch.
            return Err(BatchedMultiObjectiveAdapterError::Fallback);
        }
        // Absolute objective index → spec row position.
        let union_pos: HashMap<usize, usize> = union_indices
            .iter()
            .enumerate()
            .map(|(pos, &j)| (j, pos))
            .collect();
        let union_objectives: Vec<Vec<f32>> = union_indices
            .iter()
            .map(|&j| objectives[j].clone())
            .collect();
        let Some(spec_matrix) = build_spec_matrix(&union_objectives) else {
            return Err(BatchedMultiObjectiveAdapterError::Fallback);
        };

        // Build owned GraphBabDomain shims that outlive the batched call.
        // BatchedDomains/BatchedBackwardContext borrow `&[&GraphBabDomain]`.
        //
        // #clip-interm-resnet research hook: before the dense backward reads the
        // shim's `node_bounds`, it can tighten each non-root child's inherited
        // (frozen) pre-activation bounds by constrained concretization over
        // `box ∩ split-half-spaces`. Production authority is quarantined below:
        // the legacy environment request is ignored, so this closure returns
        // `graph_bab_domain_shim(child)` unchanged.
        let clip_interm_resnet = clip_interm_resnet_enabled();
        let __clip_t = Instant::now();
        let build_shim = |child: &MultiObjectiveGraphBabDomain| -> GraphBabDomain {
            let mut shim = graph_bab_domain_shim(child);
            if clip_interm_resnet {
                if let Some(clipped) = self.clip_child_node_bounds(graph, child, engine) {
                    shim.node_bounds = clipped;
                    // #cone-delta: the clip replaced the map the delta was
                    // tracked against — the delta no longer describes it.
                    // Fail closed to full-history seeding for this shim.
                    shim.delta_pre_nodes = crate::beta_crown::domain::delta_pre_nodes_unknown();
                }
            }
            shim
        };
        // #clip-interm-par (M1): parallelize the per-child clip when armed.
        // rayon `collect` preserves order => `shim_refs` ordering into
        // `BatchedDomains` is unchanged vs the serial arm.
        let shims: Vec<GraphBabDomain> = if clip_interm_resnet && clip_interm_par_enabled() {
            use rayon::prelude::*;
            batchable
                .par_iter()
                .map(|child| {
                    let _g = crate::faer_parallelism::RayonTaskGuard::new();
                    build_shim(child)
                })
                .collect()
        } else {
            batchable.iter().map(|child| build_shim(child)).collect()
        };
        let shim_refs: Vec<&GraphBabDomain> = shims.iter().collect();
        if std::env::var("NY_CLIP_INTERM_RESNET_PROBE").ok().as_deref() == Some("1") {
            eprintln!(
                "[clip-resnet] stage=dense n={} par={} secs={:.3}",
                batchable.len(),
                (clip_interm_resnet && clip_interm_par_enabled()) as u8,
                __clip_t.elapsed().as_secs_f64()
            );
        }

        // Build batched domains (mirrors batched_single.rs ~300). Any error =>
        // whole-batch fallback.
        let batched = match BatchedDomains::from_graph_domains_with_options(
            &shim_refs,
            relu_nodes,
            BatchedDomainOptions {
                enable_interm_transfer: self.config.enable_interm_transfer,
            },
        ) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    "batched_single_pass_multi_objective_children: BatchedDomains build failed ({e}); falling back to per-child path"
                );
                return Err(classify_adapter_error(&e, engine));
            }
        };

        // Per-domain β OPTIMIZATION request (#w4-split-tightening): on the GPU
        // single-pass lane, β-eligible children run the analytic β ascent at GPU
        // speed inside the batched fast-path (measured wall: the single-shot lane
        // left β at 0 forever, so the worst straggler only improved ~0.1-0.4 per
        // split level). Eligibility mirrors the CPU beta-opt branch
        // (`beta_iterations > 0`, non-empty β state) but WITHOUT the CPU's
        // `beta_max_depth` cap by default — the GPU pass is ~20x cheaper than the
        // CPU inner pass that motivated the cap. `NY_MO_GPU_BETA_DEPTH=<n>` caps
        // the depth for A/B; `NY_MO_GPU_BETA=0` restores the single-shot lane
        // byte-identically.
        let gpu_beta_opt =
            prune_specs_to_union && super::super::shared::multi_objective_gpu_beta_enabled();
        let union_thresholds: Vec<f32> = union_indices.iter().map(|&j| thresholds[j]).collect();
        let row_verified: Vec<Vec<bool>> = batchable
            .iter()
            .map(|child| {
                union_indices
                    .iter()
                    .map(|&j| child.verified().get(j).copied().unwrap_or(false))
                    .collect()
            })
            .collect();
        let beta_depth_cap = std::env::var("NY_MO_GPU_BETA_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        let eligible: Vec<bool> = batchable
            .iter()
            .map(|child| {
                gpu_beta_opt
                    && self.config.beta_iterations > 0
                    && child.depth() <= beta_depth_cap
                    && !child.beta_state().is_empty()
            })
            .collect();
        // #hard-six tail-iters: per-domain BaB depths let the wide ascent
        // detect a pinned-tail batch (`NY_MO_GPU_BETA_ITERS_TAIL`).
        let depths: Vec<usize> = batchable.iter().map(|child| child.depth()).collect();
        let beta_opt_spec =
            crate::beta_crown::engine::graph::propagation::batched::GpuBetaOptSpec {
                thresholds: &union_thresholds,
                row_verified: &row_verified,
                eligible: &eligible,
                depths: &depths,
            };
        let beta_opt = eligible.iter().any(|&e| e).then_some(&beta_opt_spec);

        // Certified Cut-CROWN C3 diagnostic probe (dark, `NY_C3_PROBE=1`,
        // read-only): per deep child, log the split-layer histogram and — for
        // first-ReLU split premises — each candidate L1 group's root-B vs
        // split-strengthened B (`docs/CERTIFIED_CUT_CROWN_DESIGN.md` §C3).
        if crate::beta_crown::bab_cuts::c3_probe::c3_probe_enabled() {
            for child in batchable {
                crate::beta_crown::bab_cuts::c3_probe::c3_probe_domain(
                    graph,
                    child.depth(),
                    child.history(),
                    child.input_bounds(),
                );
            }
        }

        // ONE dense-spec batched backward replacing N per-child backward passes.
        let dense_out = match self.propagate_crown_with_batched_domains_full_specs_beta_opt(
            graph,
            &shim_refs,
            &batched,
            &spec_matrix,
            engine,
            beta_opt,
        ) {
            Ok(r) => r,
            Err(error)
                if crate::beta_crown::engine::graph::is_finite_constrained_crown_refusal(
                    &error,
                ) && self.effective_graph_bab_deadline().is_some() =>
            {
                let deadline = self
                    .effective_graph_bab_deadline()
                    .expect("finite refusal guard established a deadline");
                // Keep the bounded rescue receipt simple and authoritative:
                // multiple retained child maps have aggregate nested payload
                // that this adapter cannot yet account exactly.
                if !finite_refusal_fallback_admitted(batchable.len()) {
                    return Err(BatchedMultiObjectiveAdapterError::ResourceRefused);
                }
                let output_name = if graph.output_name().is_empty() {
                    graph
                        .exec_order()
                        .ok()
                        .and_then(|order| order.last())
                        .map(String::as_str)
                } else {
                    Some(graph.output_name())
                };
                let Some(output_name) = output_name else {
                    return Err(BatchedMultiObjectiveAdapterError::Fallback);
                };
                let mut fallback_results = certified_adapter_vec(batchable.len(), deadline)?;
                for child in batchable {
                    let Some(output) = child.node_bounds().get(output_name) else {
                        return Err(BatchedMultiObjectiveAdapterError::Fallback);
                    };
                    let projected =
                        crate::beta_crown::engine::graph::objectives::objective_bounds_multi_with_deadline(
                            output,
                            objectives,
                            deadline,
                        )
                        .map_err(|error| classify_certified_adapter_fallback_error(&error))?;
                    let pruned = prune_targets_for_certified_adapter_fallback(
                        objectives,
                        thresholds,
                        child.verified(),
                        deadline,
                    )?;
                    let merged = merge_certified_adapter_bounds(
                        child.objective_bounds(),
                        &projected,
                        &pruned.active_indices,
                        deadline,
                    )?;
                    let node_bounds =
                        crate::beta_crown::engine::graph::objectives::clone_arc_node_bounds_with_deadline(
                            child.node_bounds(),
                            deadline,
                        )
                        .map_err(|error| classify_certified_adapter_fallback_error(&error))?;
                    let mut caches = certified_adapter_vec(pruned.active_indices.len(), deadline)?;
                    caches.resize_with(pruned.active_indices.len(), || None);
                    certified_adapter_checkpoint(deadline)?;
                    fallback_results.push(Ok((
                        merged,
                        node_bounds,
                        GraphBetaState::empty(),
                        None,
                        caches,
                        pruned,
                        ChildContinuationStateProvenance::Established,
                    )));
                }
                tracing::debug!(
                    %error,
                    children = fallback_results.len(),
                    "Finite batched constrained CROWN declined; retaining certified child enclosures"
                );
                return Ok(fallback_results);
            }
            Err(e) => {
                let outcome = classify_adapter_error(&e, engine);
                if outcome == BatchedMultiObjectiveAdapterError::DeadlineExpired {
                    tracing::debug!(
                        "batched_single_pass_multi_objective_children: dense-spec backward reached the authoritative deadline"
                    );
                } else {
                    tracing::warn!(
                        "batched_single_pass_multi_objective_children: dense-spec backward failed ({e}); falling back to per-child path"
                    );
                }
                return Err(outcome);
            }
        };
        let mut optimized_betas = dense_out.optimized_betas.unwrap_or_default();
        optimized_betas.resize(batchable.len(), None);
        // #hard-six unshared-α: per-child persisted α (all-None unless
        // NY_WIDE_ALPHA_UNSHARED=1 and the wide α ascent ran).
        let mut optimized_alphas = dense_out.optimized_alphas.unwrap_or_default();
        optimized_alphas.resize(batchable.len(), None);
        // #interm-refine prune lane (dark, `NY_INTERM_REFINE_PRUNE=1`): domains
        // whose split-premise set the refinement pass PROVED empty verify
        // vacuously — surfaced below through the same `Infeasible` signal the
        // with_constraint infeasibility path uses (batched_multi: "Infeasible
        // domain = empty = trivially verified").
        // #bab-throughput: the ONLY view inside the wide pass. MEASURED
        // 2026-08-13 on cifar100 idx_7641 at the official budget: BaB gets ~44 s
        // and buys ONE outer iteration — a depth-4 kFSB wave whose 16 leaves are
        // bounded here — i.e. ~2.8 s per child even inside the "batched" lane.
        // That per-child cost IS the ~80x separating ny from the reference, and
        // until now the window was unlit: the phase timeline goes silent after
        // `bab-first-domain-batch` and the next output is the `[frontier]` frame
        // 42 s later.
        //
        // `stage_timing` is already computed by the primitive
        // (`propagation/batched/spec_adapters.rs`, `DenseSpecStageTiming`) and
        // was then DROPPED UNREAD here, so surfacing it costs one field read.
        // Splitting forward / backward / materialize is what says whether the
        // cost is the per-domain CPU constrained forward (whose shared and
        // batched fast arms both require `split_count == 0` and therefore NEVER
        // fire for BaB children) or the wide GPU backward itself.
        // #adaptive-chunk: record what a child ACTUALLY cost so the next wave can
        // size its chunk from measurement instead of a fixed constant. Always on
        // (one atomic store); the reader treats an unset value as "no estimate".
        if let Some(t) = dense_out.stage_timing {
            let n = batchable.len().max(1);
            let total = t.forward_elapsed_s + t.backward_elapsed_s + t.materialize_elapsed_s;
            super::batched_multi::record_observed_child_cost(total / n as f64);
        }
        if crate::phase_telemetry::phase_telemetry_enabled() {
            if let Some(t) = dense_out.stage_timing {
                let n = batchable.len().max(1);
                let total = t.forward_elapsed_s + t.backward_elapsed_s + t.materialize_elapsed_s;
                // `union_rows` is the load-bearing number: the backward's
                // per-layer GEMM cost scales LINEARLY with spec rows, so a wave
                // costs children x union_rows. The union-pruning comment above
                // records that on cifar100 91/99 objectives are already
                // root-verified, which would make union_rows ~8 rather than 99 —
                // an ~12x difference in the DOMINANT phase. Printing it is how
                // you tell whether the pruning is actually engaging.
                eprintln!(
                    "[phase] mo-wave-stage children={} union_rows={} pruned={} \
                     fwd={:.2}s bwd={:.2}s mat={:.2}s total={:.2}s per_child={:.2}s",
                    batchable.len(),
                    union_indices.len(),
                    prune_specs_to_union,
                    t.forward_elapsed_s,
                    t.backward_elapsed_s,
                    t.materialize_elapsed_s,
                    total,
                    total / n as f64,
                );
            }
        }
        let infeasible_domains = dense_out.infeasible_domains.unwrap_or_default();
        let dense_results = dense_out.results;

        // Defense-in-depth: the primitive must return one result per domain.
        if dense_results.len() != batchable.len() {
            tracing::warn!(
                "batched_single_pass_multi_objective_children: result count {} != child count {}; falling back",
                dense_results.len(),
                batchable.len()
            );
            return Err(BatchedMultiObjectiveAdapterError::Fallback);
        }

        let results: Vec<BatchedChildResult> = batchable
            .iter()
            .zip(dense_results)
            .zip(optimized_betas)
            .zip(optimized_alphas)
            .enumerate()
            .map(
                |(dom_idx, (((child, result), optimized_beta), optimized_alpha))| {
                    // Refinement-proven infeasible subdomain (see above): empty
                    // constraint set ⇒ trivially verified, drop the child.
                    if infeasible_domains.get(dom_idx).copied().unwrap_or(false) {
                        return Err(MultiObjectiveChildEvalError::Infeasible);
                    }
                    // Per-objective bounds for every spec row (length = union rows).
                    let full_bounds_i = spec_bounds_to_vec(&result.output_bounds);
                    if full_bounds_i.len() != union_indices.len() {
                        // Spec-row/objective count mismatch — drop (sound).
                        if std::env::var("NY_PROPFAIL_PROBE").ok().as_deref() == Some("1") {
                            eprintln!(
                                "[propfail] site=dense-specs-len full_bounds={} union={} nspec_rows={}",
                                full_bounds_i.len(),
                                union_indices.len(),
                                spec_matrix.nrows()
                            );
                        }
                        return Err(MultiObjectiveChildEvalError::PropagationFailure);
                    }

                    // Replicate per-child verified-latch pruning semantics.
                    let pruned_i = prune_verified_multi_objective_targets(
                        objectives,
                        thresholds,
                        child.verified(),
                    );

                    // Select only the unverified (active) rows, in order. Every
                    // active index is in the union by construction (a child's
                    // unverified objective is unverified in at least itself);
                    // a miss is defensive-only.
                    let mut active_bounds_i: Vec<(f32, f32)> =
                        Vec::with_capacity(pruned_i.active_indices.len());
                    for &j in &pruned_i.active_indices {
                        match union_pos.get(&j) {
                            Some(&pos) => active_bounds_i.push(full_bounds_i[pos]),
                            None => {
                                if std::env::var("NY_PROPFAIL_PROBE").ok().as_deref()
                                    == Some("1")
                                {
                                    eprintln!(
                                        "[propfail] site=dense-specs-union-miss active_obj={j} union_len={}",
                                        union_indices.len()
                                    );
                                }
                                return Err(MultiObjectiveChildEvalError::PropagationFailure);
                            }
                        }
                    }

                    // #unsat-keystone CONVERGENCE PROBE (NY_BETA_GPU_PROBE=1): does the best
                    // (max) active objective lower bound climb toward the verify threshold as
                    // BaB deepens? Climbing → converging (GPU speed wins); stuck → tightness gap.
                    if std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1")
                        && !active_bounds_i.is_empty()
                    {
                        let mn = active_bounds_i
                            .iter()
                            .map(|(l, _)| *l)
                            .fold(f32::INFINITY, f32::min);
                        let mx = active_bounds_i
                            .iter()
                            .map(|(l, _)| *l)
                            .fold(f32::NEG_INFINITY, f32::max);
                        eprintln!(
                            "[converge] depth={} n_active={} min_lower={:.5} max_lower={:.5}",
                            child.depth(),
                            active_bounds_i.len(),
                            mn,
                            mx
                        );
                    }
                    // #lpopt SPLIT-PREMISE DUMP (NY_LPOPT_SPLIT_DUMP=1): for a subdomain
                    // at/beyond NY_LPOPT_SPLIT_DEPTH (default 5) whose worst active
                    // margin is below NY_LPOPT_SPLIT_MAXLB (default 0.0 = still
                    // unverified), emit the full ReLU split premise set + the binding
                    // objective index + its beta-CROWN lower bound. This is exactly the
                    // data needed to rebuild NY's OWN triangle-LP WITH the subdomain's
                    // split constraints (p*_LP_sub) and check whether NY's beta-CROWN
                    // reaches it. Read-only / print-only. One line, parseable prefix.
                    if std::env::var("NY_LPOPT_SPLIT_DUMP").ok().as_deref() == Some("1")
                        && !active_bounds_i.is_empty()
                    {
                        let min_depth = std::env::var("NY_LPOPT_SPLIT_DEPTH")
                            .ok()
                            .and_then(|s| s.trim().parse::<usize>().ok())
                            .unwrap_or(5);
                        let max_lb = std::env::var("NY_LPOPT_SPLIT_MAXLB")
                            .ok()
                            .and_then(|s| s.trim().parse::<f32>().ok())
                            .unwrap_or(0.0);
                        // binding objective = argmin over active rows.
                        let mut bind_k = 0usize;
                        let mut bind_lb = f32::INFINITY;
                        for (k, (l, _)) in active_bounds_i.iter().enumerate() {
                            if *l < bind_lb {
                                bind_lb = *l;
                                bind_k = k;
                            }
                        }
                        let bind_obj = pruned_i
                            .active_indices
                            .get(bind_k)
                            .copied()
                            .unwrap_or(usize::MAX);
                        if child.depth() >= min_depth && bind_lb < max_lb {
                            let mut prem = String::new();
                            for c in child.history().iter_all() {
                                if let crate::beta_crown::branching::GraphConstraint::Relu(nc) = c {
                                    if !prem.is_empty() {
                                        prem.push(',');
                                    }
                                    prem.push_str(&format!(
                                        "{}:{}:{}",
                                        nc.node_name,
                                        nc.neuron_idx,
                                        if nc.is_active { "A" } else { "I" }
                                    ));
                                }
                            }
                            eprintln!(
                                "[lpopt-split] depth={} bind_obj={} bind_lb={:.5} premises={}",
                                child.depth(),
                                bind_obj,
                                bind_lb,
                                prem
                            );
                            // Optional: dump this child's EFFECTIVE (per-subdomain refined)
                            // node bounds + input box, so the off-line LP can be rebuilt
                            // over the SAME [l,u] NY's beta-CROWN backward uses at THIS
                            // subdomain (NY_INTERM_REFINE tightens them below root). Capped
                            // by NY_LPOPT_BOUNDS_MAX (default 3) to bound output volume;
                            // one file per child at `<NY_LPOPT_SPLIT_BOUNDS>.<n>`.
                            if let Ok(bpath) = std::env::var("NY_LPOPT_SPLIT_BOUNDS") {
                                use std::sync::atomic::{AtomicUsize, Ordering};
                                // WORST-PER-DEPTH mode (NY_LPOPT_SPLIT_WORST=1, diagnostic-
                                // only): for the LP-climb-with-depth measurement, keep for
                                // EACH depth the child with the lowest (worst) bind_lb and
                                // (over)write it to `<bpath>.d<depth>`. Bounded output (one
                                // file/depth); always the worst-lineage child at that depth.
                                if std::env::var("NY_LPOPT_SPLIT_WORST").ok().as_deref()
                                    == Some("1")
                                {
                                    use std::collections::HashMap;
                                    use std::sync::Mutex;
                                    static WORST: Mutex<Option<HashMap<usize, f32>>> =
                                        Mutex::new(None);
                                    let mut guard = WORST.lock().unwrap();
                                    let map = guard.get_or_insert_with(HashMap::new);
                                    let d = child.depth();
                                    let prev = map.get(&d).copied().unwrap_or(f32::INFINITY);
                                    if bind_lb < prev {
                                        map.insert(d, bind_lb);
                                        drop(guard);
                                        let f = format!("{bpath}.d{d}");
                                        lpopt_dump_child_bounds(
                                            &f, d, bind_obj, bind_lb, &prem, child,
                                        );
                                    }
                                } else {
                                    static N: AtomicUsize = AtomicUsize::new(0);
                                    let cap = std::env::var("NY_LPOPT_BOUNDS_MAX")
                                        .ok()
                                        .and_then(|s| s.trim().parse::<usize>().ok())
                                        .unwrap_or(3);
                                    let n = N.fetch_add(1, Ordering::Relaxed);
                                    if n < cap {
                                        let f = format!("{bpath}.{n}");
                                        lpopt_dump_child_bounds(
                                            &f,
                                            child.depth(),
                                            bind_obj,
                                            bind_lb,
                                            &prem,
                                            child,
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Keep parent bounds for already-verified objectives; fresh
                    // batched bounds for unverified ones.
                    let mut obj_bounds_i = merge_pruned_objective_bounds(
                        child.objective_bounds(),
                        &pruned_i,
                        active_bounds_i,
                    );

                    // #nobranch-f64 (dark, NY_NOBRANCH_F64=1, default OFF =>
                    // byte-identical): an active objective's fresh GPU sound-f32
                    // batched bound OVERFLOWED to non-finite (NaN/inf) on a deep
                    // subdomain — an f32-range failure, NOT a real infeasibility.
                    // Dropping the child (PropagationFailure below) taints the whole run as
                    // Unknown via has_unresolved even after the bound otherwise
                    // converges (measured: 224 such drops on cifar100 idx_9502).
                    // Restore the child's INHERITED (parent) bound for the overflowed
                    // objective: since the child region ⊆ the parent region,
                    // child_lower ≥ parent_lower and child_upper ≤ parent_upper, so
                    // the parent's [l,u] is a VALID (looser) sound bound for the
                    // child. BaB then keeps and re-splits the child instead of
                    // tainting the verdict. `merge_pruned_objective_bounds` seeds
                    // from `child.objective_bounds()`, so indices align 1:1.
                    if matches!(std::env::var("NY_NOBRANCH_F64").ok().as_deref(), Some("1")) {
                        let inherited = child.objective_bounds();
                        for (i, b) in obj_bounds_i.iter_mut().enumerate() {
                            if !b.0.is_finite() || !b.1.is_finite() {
                                if let Some(&inh) = inherited.get(i) {
                                    if inh.0.is_finite() && inh.1.is_finite() {
                                        *b = inh;
                                    }
                                }
                            }
                        }
                    }

                    // Guard: any non-finite merged bound => drop child (sound).
                    if obj_bounds_i
                        .iter()
                        .any(|(l, u)| !l.is_finite() || !u.is_finite())
                    {
                        return Err(MultiObjectiveChildEvalError::PropagationFailure);
                    }

                    // #cone-delta increment 2: Arc-shared map, moved through to the
                    // child install at batched_multi (no re-Arc deep clone).
                    let node_cache_i: HashMap<String, Arc<BoundedTensor>> = result.node_bounds;

                    // No warm-start lA (sound): one None per active objective.
                    let active_cached_las_i = vec![None; pruned_i.active_indices.len()];

                    // β warm-start (#w4-split-tightening): children inherit the GPU
                    // per-domain-optimized β when the fast-path optimized it; the
                    // inherited state otherwise (legacy behavior).
                    let beta_state_i = optimized_beta.unwrap_or_else(|| child.beta_state().clone());

                    Ok((
                        obj_bounds_i,
                        node_cache_i,
                        beta_state_i,
                        // α warm-start (#hard-six unshared-α): `Some` only under
                        // NY_WIDE_ALPHA_UNSHARED=1; `None` keeps inherited α.
                        optimized_alpha,
                        active_cached_las_i,
                        pruned_i,
                        ChildContinuationStateProvenance::Established,
                    ))
                },
            )
            .collect();

        Ok(results)
    }

    /// Evaluate established H first, then optionally evaluate expanded W under
    /// a private cutoff that preserves five seconds before the authoritative
    /// deadline.
    ///
    /// H is returned exactly on every optional-arm refusal. A strict W win
    /// publishes `max(H_lower, W_lower)` with H's upper endpoints and carries
    /// W's independently sound node enclosure plus cache-invalidated α/β
    /// continuation warm state for the next sound recomputation.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn batched_selective_root_alpha_multi_objective_children(
        &self,
        graph: &GraphNetwork,
        batchable: &[&MultiObjectiveGraphBabDomain],
        relu_nodes: &[String],
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: &dyn GemmEngine,
        prune_specs_to_union: bool,
        selective_root_alpha_candidate: Option<&GraphDomainAlphaState>,
        authoritative_deadline: Option<Instant>,
    ) -> Result<Vec<BatchedChildResult>, BatchedMultiObjectiveAdapterError> {
        if authoritative_deadline_expired(Instant::now(), authoritative_deadline) {
            return Err(BatchedMultiObjectiveAdapterError::DeadlineExpired);
        }
        let Some(root_candidate) = selective_root_alpha_candidate else {
            // Exact historical path: without W, H publishes gather advice at
            // its original sites and timing.
            return self.batched_single_pass_multi_objective_children(
                graph,
                batchable,
                relu_nodes,
                objectives,
                thresholds,
                engine,
                prune_specs_to_union,
            );
        };

        // H remains first, but its advisory gather writes are staged rather
        // than made globally visible. This lets per-child selection publish
        // exactly H or W with no transient rejected-W observation.
        let h_gather_stage = self.gather_score_cache.stage_writes();
        let (established_call, established_finished_at) = {
            let _h_deadline_scope = self
                .complete_clip_deadline_overrides
                .scoped(authoritative_deadline);
            let result = self.batched_single_pass_multi_objective_children(
                graph,
                batchable,
                relu_nodes,
                objectives,
                thresholds,
                engine,
                prune_specs_to_union,
            );
            (result, Instant::now())
        };
        if authoritative_deadline_expired(established_finished_at, authoritative_deadline) {
            drop(h_gather_stage);
            return Err(BatchedMultiObjectiveAdapterError::DeadlineExpired);
        }
        let h_gather_writes = h_gather_stage.finish();
        let established_results = established_call?;
        let Ok(candidate_deadline) = selective_candidate_deadline(authoritative_deadline) else {
            self.gather_score_cache.commit_all(&h_gather_writes);
            tracing::info!(
                status = "fallback",
                reason = "candidate_authority_reserve_unrepresentable",
                "Selective root alpha paired transport"
            );
            return Ok(established_results);
        };
        if !selective_candidate_start_allowed(Instant::now(), candidate_deadline) {
            self.gather_score_cache.commit_all(&h_gather_writes);
            tracing::info!(
                status = "fallback",
                reason = "candidate_five_second_reserve_after_h",
                "Selective root alpha paired transport"
            );
            return Ok(established_results);
        }

        // Rebase W directly from the root candidate onto each fully committed
        // first child. This filters newly stable/constrained neurons against the
        // child's authoritative boxes and history in one step.
        let mut candidate_children = Vec::with_capacity(batchable.len());
        for child in batchable {
            let now = Instant::now();
            if authoritative_deadline_expired(now, authoritative_deadline) {
                return Err(BatchedMultiObjectiveAdapterError::DeadlineExpired);
            }
            if !selective_candidate_start_allowed(now, candidate_deadline) {
                self.gather_score_cache.commit_all(&h_gather_writes);
                return Ok(established_results);
            }
            let mut candidate_child = (*child).clone();
            candidate_child.alpha_state = GraphDomainAlphaState::from_parent_node_bounds_map(
                root_candidate,
                graph,
                child.node_bounds(),
                child.history(),
                child.input_bounds(),
            );
            // W must never consume an lA produced under H.
            candidate_child.cached_las.fill(None);
            candidate_children.push(candidate_child);
        }
        if !selective_candidate_start_allowed(Instant::now(), candidate_deadline) {
            if authoritative_deadline_expired(Instant::now(), authoritative_deadline) {
                return Err(BatchedMultiObjectiveAdapterError::DeadlineExpired);
            }
            self.gather_score_cache.commit_all(&h_gather_writes);
            tracing::info!(
                status = "fallback",
                reason = "candidate_five_second_reserve_after_construction",
                "Selective root alpha paired transport"
            );
            return Ok(established_results);
        }
        let candidate_refs: Vec<_> = candidate_children.iter().collect();
        // H is still deferred, so W observes the exact pre-H cache. Its writes
        // receive a separate stage. The private deadline scope is
        // non-bypassable by nested dense/wide dispatch.
        let w_gather_stage = self.gather_score_cache.stage_writes();
        let (candidate_call, candidate_finished_at) =
            evaluate_selective_candidate_with_scopes(self, candidate_deadline, || {
                self.batched_single_pass_multi_objective_children(
                    graph,
                    &candidate_refs,
                    relu_nodes,
                    objectives,
                    thresholds,
                    engine,
                    prune_specs_to_union,
                )
            });
        // The helper sampled this timestamp inside W's scope immediately after
        // evaluation. Adjudicate before matching or reading candidate_call and
        // before extracting its staged writes.
        let candidate_completion = adjudicate_selective_candidate_completion(
            candidate_finished_at,
            candidate_deadline,
            authoritative_deadline,
            &candidate_call,
        );
        match candidate_completion {
            SelectiveCandidateCompletion::HardDeadlineExpired => {
                drop(w_gather_stage);
                return Err(BatchedMultiObjectiveAdapterError::DeadlineExpired);
            }
            SelectiveCandidateCompletion::RetainEstablished(reason) => {
                drop(w_gather_stage);
                self.gather_score_cache.commit_all(&h_gather_writes);
                tracing::info!(
                    status = "fallback",
                    reason,
                    "Selective root alpha paired transport"
                );
                return Ok(established_results);
            }
            SelectiveCandidateCompletion::CandidateAvailable => {}
        }
        let w_gather_writes = w_gather_stage.finish();
        let candidate_results =
            candidate_call.expect("candidate availability requires a successful W evaluation");
        if candidate_results.len() != established_results.len() {
            self.gather_score_cache.commit_all(&h_gather_writes);
            tracing::info!(
                status = "fallback",
                reason = "candidate_result_count_mismatch",
                established = established_results.len(),
                candidate = candidate_results.len(),
                "Selective root alpha paired transport"
            );
            return Ok(established_results);
        }

        let candidate_fingerprints: Vec<u64> = candidate_children
            .iter()
            .map(|child| {
                crate::beta_crown::engine::graph::propagation::batched::gather_score::beta_split_fingerprint(
                    child.beta_state(),
                )
            })
            .collect();
        let mut fingerprint_counts = HashMap::<u64, usize>::new();
        for &fingerprint in &candidate_fingerprints {
            *fingerprint_counts.entry(fingerprint).or_default() += 1;
        }

        let mut selected = 0usize;
        let mut retained = 0usize;
        let mut selected_fingerprints = Vec::new();
        let results: Vec<BatchedChildResult> = established_results
            .into_iter()
            .zip(candidate_results)
            .zip(candidate_children)
            .zip(candidate_fingerprints)
            .map(
                |(((established, candidate), candidate_child), candidate_fingerprint)| {
                    let (
                        h_bounds,
                        h_node_cache,
                        h_beta,
                        h_alpha,
                        h_cached_las,
                        h_pruned,
                        h_provenance,
                    ) = match established {
                        Ok(result) => result,
                        Err(error) => return Err(error),
                    };
                    let established_evaluation = ChildArmEvaluation {
                        bounds: h_bounds,
                        continuation_state: (
                            h_node_cache,
                            h_beta,
                            h_alpha,
                            h_cached_las,
                            h_pruned,
                            h_provenance,
                        ),
                    };
                    let (
                        w_bounds,
                        w_node_cache,
                        w_beta,
                        w_optimized_alpha,
                        _w_cached_las,
                        w_pruned,
                        _w_provenance,
                    ) = match candidate {
                        Ok(result) => result,
                        Err(error) => {
                            retained += 1;
                            tracing::debug!(
                                ?error,
                                reason = "candidate_child_refusal",
                                "Selective root alpha paired transport"
                            );
                            let evaluation = established_evaluation;
                            return Ok((
                                evaluation.bounds,
                                evaluation.continuation_state.0,
                                evaluation.continuation_state.1,
                                evaluation.continuation_state.2,
                                evaluation.continuation_state.3,
                                evaluation.continuation_state.4,
                                evaluation.continuation_state.5,
                            ));
                        }
                    };
                    if established_evaluation.continuation_state.4.active_indices
                        != w_pruned.active_indices
                        || established_evaluation.continuation_state.4.verified_mask
                            != w_pruned.verified_mask
                        || fingerprint_counts
                            .get(&candidate_fingerprint)
                            .copied()
                            .unwrap_or_default()
                            != 1
                    {
                        retained += 1;
                        tracing::debug!(
                            reason = "candidate_identity_or_pruned_target_mismatch",
                            "Selective root alpha paired transport"
                        );
                        let evaluation = established_evaluation;
                        return Ok((
                            evaluation.bounds,
                            evaluation.continuation_state.0,
                            evaluation.continuation_state.1,
                            evaluation.continuation_state.2,
                            evaluation.continuation_state.3,
                            evaluation.continuation_state.4,
                            evaluation.continuation_state.5,
                        ));
                    }

                    // If the wide adapter did not optimize α, the exact state that
                    // produced W is the explicitly transported candidate child.
                    let w_alpha =
                        w_optimized_alpha.unwrap_or_else(|| candidate_child.alpha_state().clone());
                    let active_indices = established_evaluation
                        .continuation_state
                        .4
                        .active_indices
                        .clone();
                    let candidate_evaluation = ChildArmEvaluation {
                        bounds: w_bounds,
                        continuation_state: (
                            w_node_cache,
                            w_beta,
                            Some(w_alpha),
                            vec![None; active_indices.len()],
                            w_pruned,
                            ChildContinuationStateProvenance::SelectiveExpandedWarmup,
                        ),
                    };
                    match select_child_bounds_and_continuation_state(
                        established_evaluation,
                        candidate_evaluation,
                        thresholds,
                        &active_indices,
                    ) {
                        SelectivePairDecision::Established {
                            evaluation,
                            refusal,
                        } => {
                            retained += 1;
                            tracing::debug!(
                                reason = refusal.telemetry_name(),
                                "Selective root alpha paired transport"
                            );
                            Ok((
                                evaluation.bounds,
                                evaluation.continuation_state.0,
                                evaluation.continuation_state.1,
                                evaluation.continuation_state.2,
                                evaluation.continuation_state.3,
                                evaluation.continuation_state.4,
                                evaluation.continuation_state.5,
                            ))
                        }
                        SelectivePairDecision::ExpandedWarmup {
                            published_bounds,
                            continuation_state,
                        } => {
                            selected += 1;
                            selected_fingerprints.push(candidate_fingerprint);
                            Ok((
                                published_bounds,
                                continuation_state.0,
                                continuation_state.1,
                                continuation_state.2,
                                continuation_state.3,
                                continuation_state.4,
                                continuation_state.5,
                            ))
                        }
                    }
                },
            )
            .collect();
        if authoritative_deadline_expired(Instant::now(), authoritative_deadline) {
            // Neither staged arm may publish after the hard authority expires.
            return Err(BatchedMultiObjectiveAdapterError::DeadlineExpired);
        }
        // Publish the per-child choice only after the terminal deadline check.
        // One cache lock prevents readers from observing an intermediate image.
        self.gather_score_cache.commit_pair_selection(
            &h_gather_writes,
            &w_gather_writes,
            &selected_fingerprints,
        );
        tracing::info!(
            status = "complete",
            selected,
            retained,
            "Selective root alpha paired transport"
        );
        Ok(results)
    }
}

#[cfg(test)]
mod deadline_outcome_tests {
    use super::*;

    struct BoundedRefusalEngine;

    impl GemmEngine for BoundedRefusalEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> ny_core::Result<Vec<f32>> {
            unreachable!("adapter classification test does not dispatch GEMM")
        }

        fn forbids_unbounded_cpu_fallback(&self) -> bool {
            true
        }
    }

    #[test]
    fn adapter_preserves_deadline_errors_as_terminal_outcomes() {
        let ordinary = ny_core::NaiveCpuGemmEngine;
        let deadline = NyError::DeadlineExceeded("test deadline".to_string());
        assert_eq!(
            classify_adapter_error(&deadline, &ordinary),
            BatchedMultiObjectiveAdapterError::DeadlineExpired
        );

        let numerical = NyError::InvalidSpec("test failure".to_string());
        assert_eq!(
            classify_adapter_error(&numerical, &ordinary),
            BatchedMultiObjectiveAdapterError::Fallback
        );

        let memory = NyError::CpuMemoryExceeded {
            required_bytes: 2,
            budget_bytes: 1,
            site: "adapter classification test",
        };
        assert_eq!(
            classify_adapter_error(&memory, &ordinary),
            BatchedMultiObjectiveAdapterError::Fallback
        );
        assert_eq!(
            classify_adapter_error(&memory, &BoundedRefusalEngine),
            BatchedMultiObjectiveAdapterError::ResourceRefused
        );
    }

    #[test]
    fn finite_refusal_fallback_conservatively_refuses_without_aggregate_receipt() {
        assert!(!finite_refusal_fallback_admitted(0));
        assert!(!finite_refusal_fallback_admitted(1));
        assert!(
            !finite_refusal_fallback_admitted(2),
            "one or more retained child maps require a truthful aggregate receipt"
        );
    }

    #[test]
    fn candidate_ok_after_private_cutoff_retains_h_before_reading_w() {
        let now = Instant::now();
        let private = now + std::time::Duration::from_secs(5);
        let hard = now + std::time::Duration::from_secs(10);
        assert_eq!(
            adjudicate_selective_candidate_completion(private, Some(private), Some(hard), &Ok(()),),
            SelectiveCandidateCompletion::RetainEstablished(
                "candidate_private_deadline_after_evaluation"
            )
        );
    }

    #[test]
    fn candidate_ok_after_hard_deadline_is_typed_expiry() {
        let now = Instant::now();
        let private = now + std::time::Duration::from_secs(5);
        let hard = now + std::time::Duration::from_secs(10);
        assert_eq!(
            adjudicate_selective_candidate_completion(hard, Some(private), Some(hard), &Ok(()),),
            SelectiveCandidateCompletion::HardDeadlineExpired
        );
    }

    #[test]
    fn candidate_fallback_cannot_mask_hard_deadline() {
        let now = Instant::now();
        let private = now + std::time::Duration::from_secs(5);
        let hard = now + std::time::Duration::from_secs(10);
        assert_eq!(
            adjudicate_selective_candidate_completion(
                hard,
                Some(private),
                Some(hard),
                &Err::<(), _>(BatchedMultiObjectiveAdapterError::Fallback),
            ),
            SelectiveCandidateCompletion::HardDeadlineExpired
        );
    }

    #[test]
    fn rejected_w_suppresses_complete_clip_and_discards_w_but_publishes_h() {
        let mut graph = GraphNetwork::new();
        graph.add_node(crate::GraphNode::from_input(
            "relu",
            crate::Layer::ReLU(crate::ReLULayer),
        ));
        graph.set_output("relu");
        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("valid input");
        let root_bounds = graph
            .collect_node_bounds(&input)
            .expect("root bounds")
            .into_iter()
            .map(|(name, bounds)| (name, Arc::new(bounds)))
            .collect();
        let verifier = BetaCrownVerifier::default();
        assert!(verifier.complete_clip_root_bounds_cache.store_finalized(
            &graph,
            &input,
            &root_bounds,
        ));
        let complete_clip_before = verifier
            .complete_clip_root_bounds_cache
            .test_identity_image();

        let h_stage = verifier.gather_score_cache.stage_writes();
        verifier
            .gather_score_cache
            .insert(901, vec![("h".to_string(), 1, 1.0)]);
        let h_writes = h_stage.finish();

        let w_stage = verifier.gather_score_cache.stage_writes();
        let (candidate_call, candidate_finished_at) =
            evaluate_selective_candidate_with_scopes(&verifier, None, || {
                assert!(
                    verifier
                        .complete_clip_deadline_overrides
                        .complete_clip_suppressed(),
                    "the full speculative W evaluation must suppress Complete Clipping"
                );
                verifier
                    .gather_score_cache
                    .insert(902, vec![("w".to_string(), 2, 2.0)]);
                Err::<(), _>(BatchedMultiObjectiveAdapterError::Fallback)
            });
        assert_eq!(
            adjudicate_selective_candidate_completion(
                candidate_finished_at,
                None,
                None,
                &candidate_call,
            ),
            SelectiveCandidateCompletion::RetainEstablished("candidate_adapter_refusal")
        );
        drop(w_stage);
        verifier.gather_score_cache.commit_all(&h_writes);

        assert!(verifier.gather_score_cache.get(901).is_some());
        assert!(
            verifier.gather_score_cache.get(902).is_none(),
            "rejected W advisory writes must be discarded"
        );
        assert_eq!(
            verifier
                .complete_clip_root_bounds_cache
                .test_identity_image(),
            complete_clip_before,
            "rejected W must leave Complete-Clipping cache bytes and identities unchanged"
        );
        assert!(
            !verifier
                .complete_clip_deadline_overrides
                .complete_clip_suppressed(),
            "suppression scope must restore after W"
        );
    }
}

/// Build a `GraphBabDomain` shim from a multi-objective child for the dense-spec
/// batched primitive.
///
/// Field mapping (`GraphBabDomain` <- `MultiObjectiveGraphBabDomain`):
/// * `history` <- `history.clone()`
/// * `node_bounds` <- `node_bounds.clone()` (== parent's, the forward base)
/// * `lower_bound`/`upper_bound` <- `objective_bounds[0]`
/// * `depth`/`priority` <- `depth`/`priority`
/// * `input_bounds` <- `input_bounds.clone()`
/// * `beta_state`/`alpha_state` <- cloned per-domain state
/// * `cached_la` <- `None` (no warm-start; sound)
///
/// `lower_bound`/`upper_bound` only seed `BatchedDomains` accounting metadata
/// (used for `extract_updates`); they do not affect the CROWN backward bounds,
/// which are driven by `node_bounds`/`input_bounds`/`history`/`beta`/`alpha`.
/// Production authority gate for the ReLU-split intermediate-bound clip.
///
/// Quarantined: clipped `node_bounds` feed the verdict-authoritative dense CROWN
/// backward, and the current sample-based guard is not a checker-backed enclosure
/// proof. The legacy `NY_CLIP_INTERM_RESNET` request is therefore intentionally
/// ignored. Private clip helpers remain available to explicit unit tests.
pub(super) fn clip_interm_resnet_enabled() -> bool {
    false
}

#[cfg(test)]
#[test]
fn legacy_dense_clip_env_gate_is_authority_quarantined() {
    ny_test_utils::env::with_serialized_env_vars(&[("NY_CLIP_INTERM_RESNET", "1")], || {
        assert!(!clip_interm_resnet_enabled());
    });
}

/// #clip-interm-par (GPU-throughput-port M1, dark, `NY_CLIP_INTERM_PAR=1`):
/// parallelize the per-child split-premise clip shim-build (the ledger's
/// \>385s / ~225s-per-depth SERIAL tail ahead of the already-GPU-batched child
/// backward) across rayon workers. Default OFF => the serial `.iter()` arm is
/// taken. This scheduling knob has no effect while clip production authority is
/// quarantined. If the clip is re-authorized with a checker-backed enclosure
/// argument, parallelization remains per-child independent and order-preserving;
/// it does not itself grant authority to tighten verdict inputs.
pub(super) fn clip_interm_par_enabled() -> bool {
    std::env::var("NY_CLIP_INTERM_PAR")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

impl BetaCrownVerifier {
    /// Split-premise clip of a child's inherited (frozen) `node_bounds` for the
    /// cifar100 ReLU-split research lane. The production authority gate is
    /// hard-quarantined; unit tests call this machinery explicitly.
    ///
    /// Seeds a bounds cache from the child's inherited per-node bounds, computes
    /// input-relative forward linear bounds, then runs the existing SOUND graph
    /// clip (`apply_graph_clip_in_alpha_crown`): each ReLU split premise becomes
    /// an input-space half-space (`build_split_constraints`), and every node's
    /// interval is re-concretized over `box ∩ half-spaces`
    /// (`tighten_with_constraints`) and INTERSECTED with the inherited interval
    /// (`merge_bounds`, `l=max`/`u=min`, keep-original on inversion). f32
    /// coefficient roundoff is folded OUTWARD inside `compute_forward_linear_bounds`,
    /// so the result is a sound over-approximation that can only ever narrow.
    ///
    /// Returns `Some(tightened_map)` (each entry `Arc`-wrapped, ready to drop onto
    /// `shim.node_bounds`) when the clip runs; `None` — carry inherited bounds
    /// unchanged — when the gate is off, the domain is the root (empty history),
    /// or any step fails. A clip miss is sound: it can only fail to tighten, never
    /// loosen a bound above truth. Cost is bounded by `config.clip_interm_topk`.
    pub(super) fn clip_child_node_bounds(
        &self,
        graph: &GraphNetwork,
        child: &MultiObjectiveGraphBabDomain,
        engine: &dyn GemmEngine,
    ) -> Option<HashMap<String, Arc<BoundedTensor>>> {
        if !clip_interm_resnet_enabled() || child.history().depth() == 0 {
            return None;
        }
        // #clip-interm-resnet-batched: when interm_refine is active it runs the BATCHED
        // split-constraint clip on the whole domain frontier in ONE GPU pass (reusing the
        // seed backward's ResidentCoeff, zero extra backward). This serial per-child clip
        // does a seeded backward PER CHILD (~225s/depth) — the throughput wall. Skip it so
        // the batched clip takes over; skipping is sound (carry inherited = pre-clip).
        if crate::beta_crown::engine::graph::propagation::batched::interm_refine::interm_refine_enabled()
        {
            if std::env::var("NY_CLIP_INTERM_RESNET_PROBE").ok().as_deref() == Some("1") {
                eprintln!(
                    "[clip-resnet] serial clip skipped (interm_refine batched clip active)"
                );
            }
            return None;
        }

        // Seed from the inherited/frozen per-node bounds. #cone-delta
        // increment 2: the seed is an Arc-clone of the child's map (no tensor
        // copies); the clip's `merge_bounds` intersects tightened-against-this
        // and replaces entries with fresh Arcs, so the result is
        // (inherited) ∩ (constrained concretization) — never looser.
        let mut bounds_cache: HashMap<String, Arc<BoundedTensor>> =
            child.node_bounds().to_shared_hash_map();
        let before: HashMap<String, (f32, f32)> = bounds_cache
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    (
                        v.lower().iter().copied().fold(f32::INFINITY, f32::min),
                        v.upper().iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    ),
                )
            })
            .collect();
        let constrained_input = child.input_bounds();
        let probe = std::env::var("NY_CLIP_INTERM_RESNET_PROBE").ok().as_deref() == Some("1");
        let exec_order = graph.exec_order().ok()?;

        // Input-relative forward linear bounds (objective rows are box-relative;
        // split constraints enter via the history inside the clip).
        let mut forward_bounds =
            match crate::beta_crown::engine::graph::clip_alpha::compute_forward_linear_bounds(
                graph,
                child.history(),
                exec_order,
                &bounds_cache,
                constrained_input,
            ) {
                Ok(fb) => fb,
                Err(e) => {
                    if probe {
                        eprintln!(
                            "[clip-resnet] depth={} forward_linear_bounds ERR: {e}",
                            child.history().depth()
                        );
                    }
                    return None;
                }
            };

        // #clip-interm-resnet BACKWARD-SPLIT-BOUNDS: the forward accumulation
        // above returns ±inf (`LinearBounds::conservative`) for the deep-resnet
        // split pre-activations — it cannot model conv / residual-Add / flatten
        // going forward — so every split constraint bias is non-finite and
        // `build_split_constraints` skips it (the clip no-ops; measured 100% of
        // Relu_57 splits). Replace each split ReLU's pre-activation node's forward
        // bound with a FINITE input-relative enclosure from a plain SOUND CROWN
        // backward seeded with the identity at that pre-node (NY's certified
        // verdict-path backward, default slopes, no β). This feeds BOTH Step 1
        // (the split constraints) and Step 2 (tightening the pre-node's own
        // pre-activations via the cross-neuron coupling that interm_refine
        // structurally cannot exploit at the seed layer). A per-node miss keeps
        // the forward/no-op entry — sound (the clip can only fail to tighten).
        let mut pre_nodes: std::collections::HashSet<String> = std::collections::HashSet::new();
        for constraint in child.history().iter_all() {
            let relu_name = constraint.node_name();
            if let Some(relu) = graph.nodes.get(relu_name) {
                if let Some(pre) = relu.inputs.first() {
                    if pre.as_str() != crate::NETWORK_INPUT {
                        pre_nodes.insert(pre.clone());
                    }
                }
            }
        }
        // THROUGHPUT (#clip-interm-resnet): the seeded backward at each split
        // pre-node is the dominant per-child cost (~20x slowdown). But the
        // input-relative enclosure `z_j(x) = A·x + b` computed from ANY child's
        // node_bounds over the FIXED input box is a valid enclosure for EVERY
        // child of that box: looser node_bounds only WIDEN A·x+b, and the bound
        // holds for all x in the box regardless of the split constraints (the
        // splits constrain x downstream in `tighten_with_constraints`, not the
        // enclosure). In ReLU-split BaB the input box is identical across all
        // children, so we compute each pre-node's backward ONCE and cache it,
        // keyed by (pre_node, input-box signature) — collapsing thousands of
        // per-child backwards to ~one-per-ReLU-layer. SOUND: reusing a
        // wider-but-valid enclosure can only WEAKEN the clip, never unsound it.
        // THROUGHPUT vs TIGHTNESS (#clip-interm-resnet, measured 2026-07-14):
        // caching the backward with ROOT IBP bounds (valid full-box, soundly
        // reusable) is fast but too LOOSE — it loses the 0.37 tightening the
        // per-child backward gives (prop913 depth-1 worst reverts -0.666 -> -1.038).
        // The useful (tight) linear bound is inherently per-child (valid only over
        // box∩splits, NOT reusable). So the per-child backward is the DEFAULT (it
        // actually tightens); NY_CLIP_INTERM_CACHE=1 selects the fast-but-loose
        // cached root-bounds path for experiments. The real throughput fix is to
        // BATCH the per-child backward across the domain frontier (as αβ-CROWN does),
        // not to cache — recorded in the design doc.
        let clip_cache = std::env::var("NY_CLIP_INTERM_CACHE").ok().as_deref() == Some("1");
        let box_sig = clip_input_box_signature(constrained_input);
        let root_bounds: Option<Arc<HashMap<String, Arc<BoundedTensor>>>> = if clip_cache {
            let cached = CLIP_ROOT_BOUNDS_CACHE.with(|c| c.borrow().get(&box_sig).cloned());
            match cached {
                Some(rb) => Some(rb),
                None => {
                    let rb = Arc::new(
                        graph
                            .collect_node_bounds_with_engine(constrained_input, Some(engine))
                            .ok()?
                            .into_iter()
                            .map(|(name, bounds)| (name, Arc::new(bounds)))
                            .collect::<HashMap<String, Arc<BoundedTensor>>>(),
                    );
                    CLIP_ROOT_BOUNDS_CACHE.with(|c| {
                        c.borrow_mut().insert(box_sig, rb.clone());
                    });
                    Some(rb)
                }
            }
        } else {
            None
        };
        let mut backward_seeded = 0usize;
        for pre in &pre_nodes {
            let bwd = if clip_cache {
                let key = (pre.clone(), box_sig);
                let cached = CLIP_BWD_CACHE.with(|c| c.borrow().get(&key).cloned());
                match cached {
                    Some(v) => v, // hit (v == None means a known-failed backward)
                    None => {
                        let computed =
                            crate::beta_crown::engine::graph::propagation::batched::backward_input_relative_bounds_at_node(
                                graph,
                                pre,
                                root_bounds.as_ref().unwrap(),
                                constrained_input,
                                engine,
                                None,
                                None,
                            );
                        CLIP_BWD_CACHE.with(|c| {
                            c.borrow_mut().insert(key, computed.clone());
                        });
                        computed
                    }
                }
            } else {
                // DEFAULT: per-child backward from the child's (tight) inherited
                // node_bounds — gives the real tightening, at per-child cost.
                crate::beta_crown::engine::graph::propagation::batched::backward_input_relative_bounds_at_node(
                    graph,
                    pre,
                    &bounds_cache,
                    constrained_input,
                    engine,
                    None,
                    None,
                )
            };
            if let Some(bwd) = bwd {
                forward_bounds.override_node(pre, bwd);
                backward_seeded += 1;
            }
        }
        if probe {
            eprintln!(
                "[clip-resnet] depth={} pre_nodes={} backward_seeded={}",
                child.history().depth(),
                pre_nodes.len(),
                backward_seeded
            );
        }

        // Constrained concretization + sound intersection, in place on the cache.
        // #clip-resnet: NY_CLIP_INTERM_TOPK overrides the per-layer neuron budget
        // (default config.clip_interm_topk=20) — cost-only (more neurons tightened),
        // never soundness (merge_bounds only intersects).
        let clip_topk = std::env::var("NY_CLIP_INTERM_TOPK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(self.config.clip_interm_topk);
        if let Err(e) =
            crate::beta_crown::engine::graph::clip_alpha::apply_graph_clip_in_alpha_crown(
                graph,
                child.history(),
                exec_order,
                &mut bounds_cache,
                constrained_input,
                &forward_bounds,
                clip_topk,
            )
        {
            if probe {
                eprintln!(
                    "[clip-resnet] depth={} apply_graph_clip ERR: {e}",
                    child.history().depth()
                );
            }
            return None;
        }

        if probe {
            let mut changed = 0usize;
            let mut max_tighten = 0.0f32;
            for (k, v) in bounds_cache.iter() {
                if let Some(&(bl, bu)) = before.get(k) {
                    let nl = v.lower().iter().copied().fold(f32::INFINITY, f32::min);
                    let nu = v.upper().iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    if (nl - bl).abs() > 1e-6 || (nu - bu).abs() > 1e-6 {
                        changed += 1;
                        max_tighten = max_tighten.max((nl - bl).abs()).max((nu - bu).abs());
                    }
                }
            }
            eprintln!(
                "[clip-resnet] depth={} history_splits={} topk={} nodes_changed={} max_tighten={:.4}",
                child.history().depth(),
                child.history().depth(),
                clip_topk,
                changed,
                max_tighten
            );
        }

        Some(bounds_cache)
    }
}

pub(in crate::beta_crown::engine::graph::multi_objective) fn graph_bab_domain_shim(
    child: &MultiObjectiveGraphBabDomain,
) -> GraphBabDomain {
    let (lower_bound, upper_bound) = child
        .objective_bounds()
        .first()
        .copied()
        .unwrap_or((0.0, 0.0));
    GraphBabDomain {
        history: child.history().clone(),
        node_bounds: child.node_bounds().to_shared_hash_map(),
        lower_bound,
        upper_bound,
        depth: child.depth(),
        priority: child.priority(),
        input_bounds: child.input_bounds_arc().clone(),
        beta_state: child.beta_state().clone(),
        alpha_state: child.alpha_state().clone(),
        cached_la: None,
        // #cone-delta: `node_bounds` and `history` transfer verbatim, so the
        // delta transfers verbatim with them. For a KFSB candidate child of a
        // freshly bounded parent this is exactly [candidate's pre-node].
        delta_pre_nodes: child.delta_pre_nodes().to_vec(),
    }
}

/// DIAGNOSTIC-ONLY (`NY_LPOPT_SPLIT_BOUNDS`): dump a BaB child's EFFECTIVE
/// per-subdomain node bounds (`node_bounds()`, refined by NY_INTERM_REFINE) plus
/// its input box + split premises, in the same text format as the root
/// `NY_LPOPT_DUMP` (see multi_objective/root.rs `run_lpopt_dump`), so the off-line
/// triangle-LP can be rebuilt over the SAME [l,u] this subdomain's beta-CROWN
/// backward uses. Read-only.
fn lpopt_dump_child_bounds(
    path: &str,
    depth: usize,
    bind_obj: usize,
    bind_lb: f32,
    premises: &str,
    child: &MultiObjectiveGraphBabDomain,
) {
    use std::io::Write;
    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[lpopt-split-bounds] create {path} failed: {e}");
            return;
        }
    };
    let mut w = std::io::BufWriter::new(file);
    let write_lu =
        |w: &mut std::io::BufWriter<std::fs::File>, bt: &BoundedTensor| -> std::io::Result<()> {
            let (lo, up) = bt.lower_upper();
            write!(w, "L")?;
            for v in lo.iter() {
                write!(w, " {v}")?;
            }
            writeln!(w)?;
            write!(w, "U")?;
            for v in up.iter() {
                write!(w, " {v}")?;
            }
            writeln!(w)
        };
    let res = (|| -> std::io::Result<()> {
        writeln!(w, "# ny lpopt child-bounds dump v1 depth={depth} bind_obj={bind_obj} bind_lb={bind_lb} premises={premises}")?;
        let ib = child.input_bounds();
        write!(w, "INPUT {}", ib.len())?;
        for d in ib.shape() {
            write!(w, " {d}")?;
        }
        writeln!(w)?;
        write_lu(&mut w, ib)?;
        // NOTE: no RELUMAP here (the root dump carries it; reuse that mapping).
        for (name, bt) in child.node_bounds().iter() {
            write!(w, "NODE {name} {}", bt.len())?;
            for d in bt.shape() {
                write!(w, " {d}")?;
            }
            writeln!(w)?;
            write_lu(&mut w, bt)?;
        }
        w.flush()
    })();
    match res {
        Ok(()) => eprintln!("[lpopt-split-bounds] wrote {} nodes to {path} (depth={depth} bind_obj={bind_obj} bind_lb={bind_lb:.5})", child.node_bounds().len()),
        Err(e) => eprintln!("[lpopt-split-bounds] write {path} error: {e}"),
    }
}
