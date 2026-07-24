// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-clause disjunctive (OR-of-AND) input-split BaB verifier (#3740 Packet B).
//! Forked from `multi_objective.rs`. Uses `disjunctive_domain_verified()` (all
//! clauses satisfied) and `disjunctive_domain_priority()` (worst clause's best row).
//! Reference: `stop_criterion_general` (`auto_LiRPA/utils.py:115-137`).

mod child_batch;
mod process_batch;
mod push_survivors;
mod screen_child;

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::bounds::GraphAlphaState;
use crate::GraphNetwork;

use self::process_batch::process_disjunctive_domain_batch;
use super::super::shared::state::GraphBabLifecycle;
use super::batching::{
    bound_deferred_disjunctive_domains_batch, input_split_loop_batch_size,
    pop_multi_obj_input_domain_batch,
};
use super::build_batches::compute_crown_or_ibp_bounds_in_build_batches;
use super::grouped_semantics::{
    disjunctive_domain_priority, disjunctive_domain_verified, valid_disjunctive_layout,
};
use super::metrics::{should_log_batch, InputSplitBatchSummary};
use super::mul_binary::maybe_optimize_mul_binary_alphas;
use super::root_bounds::collect_input_split_root_node_bounds;
use super::shared::{
    compute_crown_or_ibp_bounds_with_node_bounds, extract_obj_bounds, MultiObjBounds,
    MultiObjInputDomain,
};
use crate::beta_crown::engine::BetaCrownVerifier;

fn format_optional_seconds(value: Option<f64>) -> String {
    value
        .map(|seconds| format!("{seconds:.3}"))
        .unwrap_or_else(|| "n/a".to_string())
}

#[inline]
fn eager_warm_alpha_enabled(config: &BetaCrownConfig, root_alpha_available: bool) -> bool {
    !config.reorder_bab
        && config.input_split_alpha_iteration > 0
        && config.use_alpha_crown
        && root_alpha_available
}

/// Seed deferred children independently of the eager-screen gate.
///
/// Reordered BaB deliberately skips eager child refinement, but its root still
/// needs to carry the optimized slopes into the deferred rebound overlay.
#[inline]
fn deferred_root_alpha_seed(
    config: &BetaCrownConfig,
    root_alpha_state: Option<&GraphAlphaState>,
) -> Option<Arc<GraphAlphaState>> {
    (config.input_split_alpha_iteration > 0 && config.use_alpha_crown)
        .then(|| root_alpha_state.cloned())
        .flatten()
        .map(Arc::new)
}

/// Conservatively disable reused-bound relaxed clipping whenever the graph
/// cannot soundly batch-stack input-split domains.
///
/// `num_clauses` is intentionally not part of the decision: the reused CROWN
/// bound can be unsound for a batch-unsafe graph even when the property has a
/// single clause. Keeping it at this policy boundary makes that invariant
/// explicit and regression-testable.
#[inline]
fn should_disable_disjunctive_relaxed_clip(
    _num_clauses: usize,
    batch_stack_safe: bool,
    relaxed_clip_enabled: bool,
) -> bool {
    // MEASURED 2026-07-24: naively re-enabling this clip for the batch-unsafe
    // class (behind an NY_LSNC_LEVELSET_CLIP experiment flag) reproduced the KNOWN
    // false-unsats — state_34 AND state_45 both returned `unsat` despite their
    // ORT-confirmed in-box counterexamples. The clip's reused-CROWN bound source
    // is genuinely unsound on this net class (807bf511's disable is correct); a
    // sound level-set clip requires a per-domain, non-batch-stacked bound source,
    // not this reused one. Keeping the sound disable.
    relaxed_clip_enabled && !batch_stack_safe
}

fn maybe_emit_batch_summary(
    verifier: &BetaCrownVerifier,
    summary: &InputSplitBatchSummary,
    force: bool,
    queue_head: Option<(f32, usize)>,
) -> Result<()> {
    if !force && !should_log_batch(summary.batch_index) {
        return Ok(());
    }

    // Bound trajectory (#cgan lever-3 measurement): the queue head's priority
    // is the worst-clause best-row gap (`lower - threshold`) of the most
    // promising unresolved domain — the distance the bound must still climb
    // before the domain verifies. `best_gap`/`depth` across batches show
    // whether splits tighten the bound at all.
    let (best_gap, head_depth) = match queue_head {
        Some((gap, depth)) => (format!("{gap:.4}"), format!("{depth}")),
        None => ("n/a".to_string(), "n/a".to_string()),
    };
    let record = summary.to_record();
    info!(
        "[disjunctive-multi-clause] batch={} popped={} queue={}->{} rebound={} rebound_s={:.3} forward_s={} backward_s={} materialize_s={} split_screen_s={:.3} dps={:.1} verified={} clipped={} best_gap={} head_depth={}",
        record.batch_index,
        record.popped_domains,
        record.queue_len_before_pop,
        record.queue_len_after_batch,
        record.rebound_mode.as_str(),
        record.rebound_total_s,
        format_optional_seconds(record.forward_s),
        format_optional_seconds(record.backward_s),
        format_optional_seconds(record.materialize_s),
        record.split_screen_s,
        record.domains_per_second,
        record.domains_verified_in_batch,
        record.domains_clipped_in_batch,
        best_gap,
        head_depth,
    );

    if let Some(sink) = verifier.input_split_metrics_sink() {
        sink.record_batch_summary(&record)?;
    }
    Ok(())
}

impl BetaCrownVerifier {
    /// LEVER 1 — the IMB early fast-path as a PUBLIC hook, so the CLI can run it AHEAD
    /// of the ~365 s CROWN-IBP per-output precheck (which the IMB neither needs nor
    /// consumes) instead of DOWNSTREAM of it. This is the SAME full-recheck logic as
    /// the in-lane `#imb-early` block below (`imb_multi_objective_floors` →
    /// `disjunctive_domain_verified`) — only the call POSITION moves.
    ///
    /// Returns `Some(Verified)` iff every disjunctive clause is IMB-refuted (→ the
    /// caller returns `unsat` immediately, skipping the precheck + collection + BaB);
    /// else `None` (→ the caller runs the UNCHANGED precheck + BaB). Gated
    /// `NY_IMB=1 && NY_IMB_WIRE=1` (opt-out `NY_IMB_EARLY=0`); on a disarmed/mismatched
    /// instance it is a no-op `None`. Sets `imb::mark_early_attempted()` so the in-lane
    /// block does NOT repeat the (expensive) leaf-BaB / tail-opt.
    ///
    /// SOUND: identical to the in-lane block — IMB only proposes an exact-cover input
    /// partition; each row is independently re-bounded on the original full network.
    /// `disjunctive_domain_verified` concludes UNSAT only when every clause has such a
    /// replay-certified row; any miss falls through to the standard pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn try_imb_early_disjunctive(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        clause_sizes: &[usize],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Option<BetaCrownResult> {
        let imb_early_on = crate::imb::enabled()
            && matches!(std::env::var("NY_IMB_WIRE").ok().as_deref(), Some("1"))
            && !matches!(std::env::var("NY_IMB_EARLY").ok().as_deref(), Some("0"));
        self.try_imb_early_disjunctive_with_gate(
            graph,
            input,
            objectives,
            thresholds,
            clause_sizes,
            engine,
            deadline,
            imb_early_on,
        )
    }

    /// Gate-injected implementation so tests can exercise the armed preflight
    /// without mutating process-global environment variables.
    #[allow(clippy::too_many_arguments)]
    fn try_imb_early_disjunctive_with_gate(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        clause_sizes: &[usize],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        imb_early_on: bool,
    ) -> Option<BetaCrownResult> {
        if !imb_early_on
            || !valid_disjunctive_layout(objectives.len(), thresholds.len(), clause_sizes)
        {
            return None;
        }
        // Mark ATTEMPTED (verify or fall-through) so the downstream in-lane block skips.
        crate::imb::mark_early_attempted();
        let engine = self.resolve_engine(engine);
        // Configure the graph exactly as the in-lane path does (patches-mode + adopted
        // bound caches), so the seam-box CROWN and the IMB pipeline are byte-identical
        // to the in-lane fast-path — only the position differs.
        let configured = self.configured_graph_for_crown(graph);
        let graph = &configured;
        if !crate::imb::armed(graph, input) {
            return None;
        }
        let t_early = Instant::now();
        match graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline) {
            Ok(ibp_nb) => {
                let tightened = crate::imb::root_inject::imb_multi_objective_floors(
                    graph,
                    input,
                    objectives,
                    thresholds,
                    clause_sizes,
                    engine,
                    &ibp_nb,
                    deadline,
                );
                if disjunctive_domain_verified(&tightened, thresholds, clause_sizes) {
                    eprintln!(
                        "[imb] EARLY FAST-PATH (pre-precheck): VERIFIED in {:.1}s — skipping precheck + collection + BaB",
                        t_early.elapsed().as_secs_f64()
                    );
                    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
                    lifecycle.domains_explored = 1;
                    lifecycle.domains_verified = 1;
                    return Some(lifecycle.build_result(BabVerificationStatus::Verified));
                }
                eprintln!(
                    "[imb] EARLY FAST-PATH (pre-precheck): not verified in {:.1}s — falling through to precheck + BaB",
                    t_early.elapsed().as_secs_f64()
                );
            }
            Err(e) => {
                eprintln!(
                    "[imb] EARLY FAST-PATH (pre-precheck): IBP node-bounds failed ({e}); falling through"
                );
            }
        }
        None
    }

    /// Multi-clause disjunctive (OR-of-AND) input-split BaB.
    ///
    /// Spec rows packed clause-by-clause per `clause_sizes`. Verified when every
    /// clause has `lower > threshold` for at least one row. Part of #3740 Packet B.
    /// Reference: `stop_criterion_general` (`auto_LiRPA/utils.py:115-137`).
    /// `deadline`: If `Some`, the BaB engine derives its phase budgets from
    /// remaining wall-clock time instead of `self.config.timeout` (#4321).
    pub fn verify_graph_input_split_multi_clause_disjunctive(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        clause_sizes: &[usize],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        self.config.validate()?;
        let engine = self.resolve_engine(engine);

        // Validate dimensions without allowing malformed packed layouts to
        // panic on overflow or introduce vacuous zero-width clauses.
        if objectives.is_empty() || objectives.len() != thresholds.len() {
            return Err(NyError::InvalidSpec(format!(
                "Disjunctive multi-clause: {} objectives vs {} thresholds",
                objectives.len(),
                thresholds.len()
            )));
        }
        if clause_sizes.is_empty() {
            return Err(NyError::InvalidSpec(
                "Disjunctive multi-clause: empty clause_sizes".to_string(),
            ));
        }
        if clause_sizes.contains(&0) {
            return Err(NyError::InvalidSpec(
                "Disjunctive multi-clause: zero-sized clause".to_string(),
            ));
        }
        let total_rows = clause_sizes
            .iter()
            .try_fold(0usize, |acc, &size| acc.checked_add(size))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "Disjunctive multi-clause: clause_sizes total overflow".to_string(),
                )
            })?;
        if total_rows != objectives.len() {
            return Err(NyError::InvalidSpec(format!(
                "Disjunctive multi-clause: clause_sizes sum {} != {} objectives",
                total_rows,
                objectives.len()
            )));
        }

        // Reject non-finite thresholds early. Part of #3646.
        for (i, &t) in thresholds.iter().enumerate() {
            if !t.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "Disjunctive multi-clause threshold[{}] is non-finite ({}); \
                     BaB cannot make progress with NaN/Inf thresholds",
                    i, t
                )));
            }
        }

        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;

        // #disj-cross-clause-clip-unsat: the RELAXED INPUT CLIP verifies domains
        // (box-infeasibility + the `concretize_postclip_lower_bounds` grouped
        // check) using the parent's REUSED CROWN linear bounds. On the
        // batch-stack-UNSAFE disjunctive tracks (lsnc_relu's MulBinary/Gather
        // difference nets, cgan's Relu->BatchNorm, linearizenn's Concat/Slice)
        // those reused bounds are NOT a sound lower bound over the clipped
        // sub-box — decisively reproduced on lsnc quadrotor2d_state_34: over the
        // sub-box around the ORT counterexample X the clip's concretize claims
        // `max Y_1 <= 0.40` while ORT gives `Y_1(X)=0.725`, `max Y_1 = 1.33`, so
        // every clause is spuriously refuted and the DOMAIN (which contains X) is
        // marked verified — a FALSE UNSAT that the upfront falsifier only
        // sometimes masks. Making the clip clause-aware keeps X inside the carried
        // UNION box but does NOT fix this, because the concretize bound itself is
        // unsound. Clause count is not a soundness precondition: the same reused
        // bound can be invalid for a batch-unsafe single-clause graph. So for the
        // entire batch-unsafe class we DISABLE the relaxed clip and let the SOUND
        // deferred re-bound
        // (`bound_deferred_disjunctive_domains_batch` + the top-of-loop grouped
        // check on freshly recomputed obj bounds) do all verification.
        //
        // This is STRICTLY BETTER than fail-closing the UNSAT verdict (the prior
        // #disj gate): the re-bound is sound and complete-in-the-limit, so real
        // UNSATs still resolve — measured with the clip off: cgan_2023 stays
        // `unsat` (169 s), linearizenn stays `unsat` (~0 s), lsnc_relu's
        // false-unsat becomes `unknown`/`sat`. Batch-SAFE disjunctions
        // (acasxu prop_5/6/9/10) keep the CLAUSE-AWARE clip — their CROWN bounds
        // ARE sound, the clip is load-bearing (acasxu prop_9 times out without
        // it), and the clause-aware carve fixes the cross-clause box shrink that
        // was latently unsound there.
        let disable_clip_unsound_class = should_disable_disjunctive_relaxed_clip(
            clause_sizes.len(),
            graph.is_input_split_batch_stack_safe(),
            self.config.enable_relaxed_clip,
        );
        let clip_disabled_verifier;
        let bab: &BetaCrownVerifier = if disable_clip_unsound_class {
            let mut cfg = self.config.clone();
            cfg.enable_relaxed_clip = false;
            clip_disabled_verifier = self.with_config_from(cfg);
            &clip_disabled_verifier
        } else {
            self
        };

        let num_specs = objectives.len();
        let spec_dim = objectives[0].len();

        // Build multi-row spec matrix: each row is one C-matrix row (objective).
        let mut spec_data = Vec::with_capacity(num_specs * spec_dim);
        for obj in objectives {
            if obj.len() != spec_dim {
                return Err(NyError::InvalidSpec(format!(
                    "Objective dimension mismatch: {} vs {}",
                    obj.len(),
                    spec_dim
                )));
            }
            spec_data.extend_from_slice(obj);
        }
        let spec_matrix = Array2::from_shape_vec((num_specs, spec_dim), spec_data)
            .map_err(|e| NyError::InvalidSpec(format!("spec matrix: {}", e)))?;

        // EARLY IMB FAST-PATH (#imb-early). The self-contained Input-Manifold Bound
        // certificate is INDEPENDENT of the standard root collection (it rebuilds its
        // own tight prefix/tail anchors), but that collection is the pipeline's single
        // largest up-front cost (~313 s on cgan) and IMB needs ~250 s (180 s anchor +
        // ~40 s leaf-BaB + ~30 s tail-opt), so running IMB AFTER it can blow the wall
        // budget. Running IMB FIRST — before the collection — gives it the whole budget
        // and, when it refutes every clause, returns `Verified` immediately, skipping
        // the collection + BaB entirely.
        //
        // SOUND: IMB's sampled/decomposed floor only proposes terminal input boxes.
        // The wire validates their exact binary cover and independently re-bounds the
        // ORIGINAL full-network row on every leaf; only that replay lower can enter the
        // baseline. A candidate that fails any replay check returns baseline unchanged.
        // Therefore the tightened bounds are valid root lower bounds, and
        // `disjunctive_domain_verified` on them can only conclude `Verified` when every
        // clause is genuinely refuted. STRICTLY ADDITIVE: on any miss / disarmed / error
        // we fall through to the UNCHANGED standard pipeline (whose own late-stage
        // max-wiring stays as a second chance). Gate: `NY_IMB=1 && NY_IMB_WIRE=1`
        // (opt-out with `NY_IMB_EARLY=0`); default-OFF for every other benchmark.
        // SUPPRESSED when the CLI-level LEVER-1 hook already attempted the IMB (it runs
        // AHEAD of the precheck); re-running the leaf-BaB / tail-opt here would just
        // burn budget on the same result. When the hook never fired (non-Graph, etc.)
        // this stays the standalone in-lane fast-path.
        let imb_early_on = crate::imb::enabled()
            && matches!(std::env::var("NY_IMB_WIRE").ok().as_deref(), Some("1"))
            && !matches!(std::env::var("NY_IMB_EARLY").ok().as_deref(), Some("0"))
            && !crate::imb::early_attempted();
        if imb_early_on && crate::imb::armed(graph, input) {
            let t_early = Instant::now();
            match graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline) {
                Ok(ibp_nb) => {
                    // STEP 2 — MULTI-OBJECTIVE: use IMB to propose a partition for
                    // EVERY disjunctive clause's binding row, independently replay the
                    // original full-network row on all leaves, and raise only the
                    // replay-certified lowers, so
                    // `disjunctive_domain_verified` can conclude UNSAT only when every
                    // clause is genuinely refuted. Exact mode shares the prefix graph
                    // and tight anchor through one run-local graph/input-bound session;
                    // the tail (p,q) is per objective. For a single-clause prop
                    // (prop_0) this certifies exactly one objective — identical to the
                    // prior single-objective fast-path. A clause the IMB can't refute →
                    // not verified → the vacuous `-inf` baseline stands and we fall
                    // through unchanged.
                    let tightened = crate::imb::root_inject::imb_multi_objective_floors(
                        graph,
                        input,
                        objectives,
                        thresholds,
                        clause_sizes,
                        engine,
                        &ibp_nb,
                        deadline,
                    );
                    if disjunctive_domain_verified(&tightened, thresholds, clause_sizes) {
                        eprintln!(
                            "[imb] EARLY FAST-PATH: VERIFIED in {:.1}s — skipping standard collection + BaB",
                            t_early.elapsed().as_secs_f64()
                        );
                        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
                        lifecycle.domains_explored = 1;
                        lifecycle.domains_verified = 1;
                        return Ok(lifecycle.build_result(BabVerificationStatus::Verified));
                    }
                    eprintln!(
                        "[imb] EARLY FAST-PATH: not verified in {:.1}s — falling through to standard pipeline",
                        t_early.elapsed().as_secs_f64()
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[imb] EARLY FAST-PATH: IBP node-bounds failed ({e}); falling through"
                    );
                }
            }
        }

        let now = Instant::now();
        let mut lifecycle = GraphBabLifecycle::new(now);
        // When a wall-clock deadline is provided (#4321), derive the effective
        // timeout from remaining time instead of the configured timeout.
        let pgd_frac = self
            .config
            .phase_budget
            .post_bab_pgd_fraction
            .clamp(0.0, 0.5);
        let effective_total = match deadline {
            Some(dl) => dl.saturating_duration_since(now),
            None => self.config.timeout,
        };
        let bab_timeout = effective_total.mul_f32(1.0 - pgd_frac);
        let initial_deadline = {
            let frac = self
                .config
                .phase_budget
                .initial_bounds_fraction
                .clamp(0.0, 1.0);
            Some(now + bab_timeout.mul_f32(frac))
        };
        let crown_deadline = Some(now + bab_timeout);
        let mut domains_verified_by_clip = 0usize;

        let (root_node_bounds, root_alpha_state): (
            Option<HashMap<String, BoundedTensor>>,
            Option<GraphAlphaState>,
        ) = collect_input_split_root_node_bounds(
            graph,
            input,
            &self.config,
            engine,
            initial_deadline,
            "disjunctive-multi-clause input splitting",
            self.disjunctive_restart_root_cache(objectives, thresholds, clause_sizes, deadline)
                .map(|cache| (cache, deadline)),
        )?;

        // Phase 4 (#3439): MulBinary SPSA alpha optimization.
        let mul_binary_alphas_multi = maybe_optimize_mul_binary_alphas(
            graph,
            input,
            &spec_matrix,
            engine,
            initial_deadline,
            self.config.crown_backward_layers,
            "Graph input split (disjunctive-multi-clause)",
        )?;

        let crown_bkwd = self.config.crown_backward_layers;
        let compute_bounds = |input_bounds: &BoundedTensor,
                              node_bounds: Option<&HashMap<String, BoundedTensor>>|
         -> Result<MultiObjBounds> {
            let (bounds, linear) = compute_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                &spec_matrix,
                engine,
                root_node_bounds.as_ref(),
                node_bounds,
                root_alpha_state.as_ref(),
                mul_binary_alphas_multi.as_ref(),
                crown_deadline,
                crown_bkwd,
                self.config.input_split_ibp_enhancement,
            )?;
            Ok((extract_obj_bounds(&bounds, num_specs)?, linear))
        };

        // First safe slice of grouped-disjunctive per-domain alpha refinement:
        // eager BaB only. The reordered rebound is intentionally excluded because
        // its dense batch request currently carries one shared root alpha state,
        // not one inherited state per domain.
        let warm_alpha_enabled = eager_warm_alpha_enabled(&self.config, root_alpha_state.is_some());
        if self.config.input_split_alpha_iteration > 0 {
            if warm_alpha_enabled {
                eprintln!(
                    "NY_WARM_ALPHA route=grouped-disjunctive-eager status=enabled \
                     iterations={} lr={}",
                    self.config.input_split_alpha_iteration, self.config.input_split_lr_alpha
                );
            } else {
                eprintln!(
                    "NY_WARM_ALPHA route=grouped-disjunctive-eager status=inactive iterations={} \
                     reorder_bab={} use_alpha_crown={} root_alpha_available={}",
                    self.config.input_split_alpha_iteration,
                    self.config.reorder_bab,
                    self.config.use_alpha_crown,
                    root_alpha_state.is_some()
                );
            }
        }
        let warm_alpha_telemetry = screen_child::WarmAlphaTelemetry::new(warm_alpha_enabled);
        let warm_compute_bounds = |input_bounds: &BoundedTensor,
                                   node_bounds: Option<&HashMap<String, BoundedTensor>>,
                                   parent_alpha: &GraphAlphaState|
         -> Result<screen_child::WarmDisjunctiveBoundsResult> {
            let (bounds, linear, refined_alpha) =
                super::shared::compute_warm_start_crown_bounds_with_refined_alpha(
                    graph,
                    input_bounds,
                    &spec_matrix,
                    engine,
                    node_bounds,
                    parent_alpha,
                    mul_binary_alphas_multi.as_ref(),
                    crown_deadline,
                    crown_bkwd,
                    &self.config,
                )?;
            Ok((
                extract_obj_bounds(&bounds, num_specs)?,
                linear,
                refined_alpha,
            ))
        };
        let warm_compute_bounds_opt: Option<&screen_child::WarmDisjunctiveComputeBoundsFn<'_>> =
            if warm_alpha_enabled {
                Some(&warm_compute_bounds)
            } else {
                None
            };

        // Root domain bounds.
        let (root_bounds, root_linear) = compute_crown_or_ibp_bounds_in_build_batches(
            graph,
            input,
            &spec_matrix,
            self.config.build_batch_size,
            engine,
            root_node_bounds.as_ref(),
            root_alpha_state.as_ref(),
            mul_binary_alphas_multi.as_ref(),
            initial_deadline,
            crown_bkwd,
            self.config.input_split_ibp_enhancement,
        )?;
        let root_obj_bounds = extract_obj_bounds(&root_bounds, num_specs)?;

        // Multi-neuron (k-ReLU) ROOT injection (#multineuron, NY_MULTINEURON=1;
        // conv-fed discriminator ReLUs via NY_MULTINEURON_CONV=1). Wires the proven
        // cifar100 root injection (multi_objective/root.rs) into the DISJUNCTIVE
        // input-split root so the cgan objective slack — which lives in the conv-fed
        // discriminator ReLUs — can be tightened BEFORE BaB (measured: prop_0 is
        // margin-bound, ~2e-4 short and uniform, so a tighter root is the lever, not
        // more splitting). SOUND-BY-CONSTRUCTION (Invariant MN): the function returns
        // the per-objective MAX over the baseline and every injected β-candidate, so
        // it can only tighten the margin, never over-claim; it returns the baseline
        // unchanged when disarmed, on a non-conv net, without alpha, or when no
        // facet-carrying group is found. No-op unless NY_MULTINEURON=1.
        let root_obj_bounds = match root_node_bounds.as_ref() {
            Some(nb) => crate::multineuron::root_inject::tighten_root_objective_bounds(
                graph,
                input,
                objectives,
                engine,
                nb,
                root_alpha_state.as_ref(),
                &root_obj_bounds,
                initial_deadline,
            ),
            None => root_obj_bounds,
        };

        // Input-Manifold Bound (IMB) ROOT floor (#imb, NY_IMB=1; default-OFF).
        // Proposes a partition for the hardest objective by keeping the generator
        // + first-disc-block prefix
        // EXACT and relaxing the disc tail to ONE alpha-optimized affine lower
        // functional `p·h(x)+q`, then certifying `min_x[p·h(x)+q]` via per-leaf
        // backward-CROWN input-split BaB over the free input dims. Without
        // `NY_IMB_WIRE=1` this remains log-only. With wiring, the decomposed floor
        // still has no authority: every terminal box is replayed against the
        // original full-network objective before any lower bound can be raised.
        // No-op unless NY_IMB=1 (the `else` arm is the byte-identical default).
        let root_obj_bounds = if crate::imb::enabled() {
            match root_node_bounds.as_ref() {
                Some(nb) => crate::imb::root_inject::tighten_root_objective_bounds_imb(
                    graph,
                    input,
                    objectives,
                    thresholds,
                    engine,
                    nb,
                    root_alpha_state.as_ref(),
                    &root_obj_bounds,
                    initial_deadline,
                ),
                None => root_obj_bounds,
            }
        } else {
            root_obj_bounds
        };

        info!(
            "[disjunctive-multi-clause] {} clauses, {} total rows, root bounds (alpha={}, forward_bounds={}): {}",
            clause_sizes.len(),
            num_specs,
            self.config.use_alpha_crown,
            self.config.use_forward_bounds,
            root_obj_bounds
                .iter()
                .zip(thresholds.iter())
                .map(|((l, u), &t)| format!("[{:.6}, {:.6}] thr={:.6}", l, u, t))
                .collect::<Vec<_>>()
                .join(", ")
        );

        if disjunctive_domain_verified(&root_obj_bounds, thresholds, clause_sizes) {
            lifecycle.domains_explored = 1;
            lifecycle.domains_verified = 1;
            return Ok(lifecycle.build_result(BabVerificationStatus::Verified));
        }
        if lifecycle.start_time.elapsed() > bab_timeout {
            return Ok(lifecycle.timeout_result());
        }

        let root_priority = disjunctive_domain_priority(&root_obj_bounds, thresholds, clause_sizes);
        // Seed the root with its optimized α state for the deferred reordered
        // rebound. This gate is deliberately separate from `warm_alpha_enabled`:
        // that eager lane requires `!reorder_bab`, while reordered domains must
        // carry the seed without also running the eager refinement.
        let root_inherited_alpha =
            deferred_root_alpha_seed(&self.config, root_alpha_state.as_ref());
        let mut queue: BinaryHeap<MultiObjInputDomain> = BinaryHeap::new();
        queue.push(MultiObjInputDomain {
            input_bounds: Arc::new(input.clone()),
            obj_bounds: root_obj_bounds,
            linear_bounds: root_linear,
            depth: 0,
            priority: root_priority,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: root_inherited_alpha,
        });

        if self.config.reorder_bab {
            let loop_batch = input_split_loop_batch_size(self.config.batch_size, input.len())?;
            info!(
                requested_batch_size = loop_batch.requested_batch_size,
                effective_batch_size = loop_batch.effective_batch_size,
                clamp_reason = loop_batch.clamp_reason.as_str(),
                input_elems = input.len(),
                "[disjunctive-multi-clause] using reordered BaB (bound -> filter -> split -> clip)"
            );
        }

        let loop_batch_size = if self.config.reorder_bab {
            input_split_loop_batch_size(self.config.batch_size, input.len())?.effective_batch_size
        } else {
            1
        };
        let mut batch_index = 0usize;

        while !queue.is_empty() {
            if lifecycle.start_time.elapsed() > bab_timeout {
                return Ok(lifecycle.timeout_result());
            }
            if lifecycle.domains_explored >= self.config.max_domains {
                return Ok(lifecycle.build_result(BabVerificationStatus::Unknown {
                    reason: format!(
                        "Domain limit {}: {}/{} verified",
                        self.config.max_domains,
                        lifecycle.domains_verified,
                        lifecycle.domains_explored
                    ),
                }));
            }

            let queue_len_before_pop = queue.len();
            let mut domains = pop_multi_obj_input_domain_batch(&mut queue, loop_batch_size);
            let popped_domains = domains.len();
            let verified_before_batch = lifecycle.domains_verified;
            let clipped_before_batch = domains_verified_by_clip;
            let rebound = bound_deferred_disjunctive_domains_batch(
                &mut domains,
                graph,
                &spec_matrix,
                thresholds,
                clause_sizes,
                engine,
                root_node_bounds.as_ref(),
                root_alpha_state.as_ref(),
                mul_binary_alphas_multi.as_ref(),
                crown_deadline,
                crown_bkwd,
                &self.config,
                self.graph_domain_batch_metrics_sink(),
                batch_index,
            )?;
            let split_screen_start = Instant::now();

            let batch_result = process_disjunctive_domain_batch(
                bab,
                graph,
                domains,
                &spec_matrix,
                thresholds,
                clause_sizes,
                engine,
                &compute_bounds,
                warm_compute_bounds_opt,
                &warm_alpha_telemetry,
                mul_binary_alphas_multi.as_ref(),
                bab_timeout,
                &mut queue,
                &mut lifecycle,
                &mut domains_verified_by_clip,
            )?;

            let summary = InputSplitBatchSummary {
                batch_index,
                queue_len_before_pop,
                queue_len_after_batch: queue.len(),
                popped_domains,
                domains_explored_after_batch: lifecycle.domains_explored,
                domains_verified_in_batch: lifecycle.domains_verified - verified_before_batch,
                domains_clipped_in_batch: domains_verified_by_clip - clipped_before_batch,
                rebound,
                split_screen_elapsed_s: split_screen_start.elapsed().as_secs_f64(),
            };
            let should_force_emit = batch_result.is_some();
            let queue_head = queue.peek().map(|d| (d.priority, d.depth));
            maybe_emit_batch_summary(self, &summary, should_force_emit, queue_head)?;
            batch_index += 1;

            if let Some(result) = batch_result {
                return Ok(result);
            }
        }

        if domains_verified_by_clip > 0 {
            info!(
                "[disjunctive-multi-clause] domains_verified_by_clip={} out of {} verified ({} explored)",
                domains_verified_by_clip, lifecycle.domains_verified, lifecycle.domains_explored
            );
        }

        Ok(lifecycle.build_final_result())
    }
}

#[cfg(test)]
mod warm_alpha_gate_tests {
    use super::*;

    #[test]
    fn grouped_disjunctive_warm_alpha_gate_is_explicitly_eager_only() {
        let enabled = BetaCrownConfig {
            reorder_bab: false,
            use_alpha_crown: true,
            input_split_alpha_iteration: 5,
            ..Default::default()
        };
        assert!(eager_warm_alpha_enabled(&enabled, true));

        let cases = [
            BetaCrownConfig {
                reorder_bab: true,
                ..enabled.clone()
            },
            BetaCrownConfig {
                use_alpha_crown: false,
                ..enabled.clone()
            },
            BetaCrownConfig {
                input_split_alpha_iteration: 0,
                ..enabled.clone()
            },
        ];
        for config in cases {
            assert!(!eager_warm_alpha_enabled(&config, true));
        }
        assert!(!eager_warm_alpha_enabled(&enabled, false));
    }

    #[test]
    fn grouped_disjunctive_root_seed_includes_reorder_but_eager_gate_does_not_f8() {
        let root_alpha = GraphAlphaState::new();
        let reordered = BetaCrownConfig {
            reorder_bab: true,
            use_alpha_crown: true,
            input_split_alpha_iteration: 5,
            ..Default::default()
        };

        assert!(deferred_root_alpha_seed(&reordered, Some(&root_alpha)).is_some());
        assert!(!eager_warm_alpha_enabled(&reordered, true));

        let eager = BetaCrownConfig {
            reorder_bab: false,
            ..reordered.clone()
        };
        assert!(deferred_root_alpha_seed(&eager, Some(&root_alpha)).is_some());
        assert!(eager_warm_alpha_enabled(&eager, true));

        for disabled in [
            BetaCrownConfig {
                input_split_alpha_iteration: 0,
                ..reordered.clone()
            },
            BetaCrownConfig {
                use_alpha_crown: false,
                ..reordered.clone()
            },
        ] {
            assert!(deferred_root_alpha_seed(&disabled, Some(&root_alpha)).is_none());
        }
        assert!(deferred_root_alpha_seed(&reordered, None).is_none());
    }
}

#[cfg(test)]
mod relaxed_clip_safety_tests {
    use super::should_disable_disjunctive_relaxed_clip;

    #[test]
    fn batch_unsafe_graph_disables_relaxed_clip_regardless_of_clause_count() {
        for num_clauses in [1, 2, 8] {
            assert!(should_disable_disjunctive_relaxed_clip(
                num_clauses,
                false,
                true
            ));
        }
    }

    #[test]
    fn safe_or_already_disabled_relaxed_clip_keeps_existing_verifier() {
        for num_clauses in [1, 2, 8] {
            assert!(!should_disable_disjunctive_relaxed_clip(
                num_clauses,
                true,
                true
            ));
            assert!(!should_disable_disjunctive_relaxed_clip(
                num_clauses,
                false,
                false
            ));
            assert!(!should_disable_disjunctive_relaxed_clip(
                num_clauses,
                true,
                false
            ));
        }
    }
}

#[cfg(test)]
mod imb_early_layout_tests {
    use super::*;
    use ndarray::array;

    fn assert_armed_preflight_rejects(
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        clause_sizes: &[usize],
    ) {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let graph = GraphNetwork::new();
        let input = BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn())
            .expect("valid test input");
        crate::imb::reset_early_attempted();

        let result = verifier.try_imb_early_disjunctive_with_gate(
            &graph,
            &input,
            objectives,
            thresholds,
            clause_sizes,
            None,
            None,
            true,
        );

        assert!(result.is_none(), "malformed layout must never verify");
        assert!(
            !crate::imb::early_attempted(),
            "malformed layout must return before marking or expensive collection"
        );
    }

    #[test]
    fn armed_early_imb_rejects_empty_clauses_and_empty_objectives() {
        assert_armed_preflight_rejects(&[], &[], &[]);
    }

    #[test]
    fn armed_early_imb_rejects_empty_clauses_with_nonempty_objectives() {
        assert_armed_preflight_rejects(&[vec![1.0]], &[0.0], &[]);
    }

    #[test]
    fn armed_early_imb_rejects_empty_objectives_with_nonempty_clause() {
        assert_armed_preflight_rejects(&[], &[], &[1]);
    }

    #[test]
    fn armed_early_imb_rejects_zero_sized_clause() {
        assert_armed_preflight_rejects(&[vec![1.0]], &[0.0], &[0, 1]);
    }

    #[test]
    fn armed_early_imb_rejects_threshold_and_clause_total_mismatches() {
        assert_armed_preflight_rejects(&[vec![1.0]], &[], &[1]);
        assert_armed_preflight_rejects(&[vec![1.0]], &[0.0], &[2]);
    }

    #[test]
    fn armed_early_imb_rejects_clause_total_overflow() {
        assert_armed_preflight_rejects(&[vec![1.0]], &[0.0], &[usize::MAX, 1]);
    }
}
