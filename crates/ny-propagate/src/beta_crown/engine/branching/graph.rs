// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph-network branching helpers and BaBSR graph coefficient propagation.

use super::*;

impl BetaCrownVerifier {
    pub(in crate::beta_crown::engine) fn find_unstable_graph_neurons_multi(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        relu_nodes: &[String],
    ) -> Vec<(String, usize)> {
        let mut unstable = Vec::with_capacity(relu_nodes.len());

        for node_name in relu_nodes {
            let relu_node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            if !is_zero_threshold_binary_activation(&relu_node.layer) {
                continue;
            }
            // Skip nodes with empty inputs instead of fabricating NETWORK_INPUT.
            // A ReLU with no inputs is a graph construction bug — using network
            // input bounds would produce unsound branching decisions.
            let pre_name = match relu_node.inputs.first() {
                Some(s) => s.as_str(),
                None => {
                    tracing::warn!(node = %node_name, "ReLU node has empty inputs — skipping");
                    continue;
                }
            };

            let pre_bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
                domain.input_bounds.as_ref()
            } else {
                match domain.node_bounds.get(pre_name) {
                    Some(b) => b.as_ref(),
                    None => continue,
                }
            };

            // Hoisted out of the neuron loop: `is_constrained`
            // (`branching/graph_history.rs:172`) hashes `node_name` on every
            // call, against the two-level
            // `relu_lookup: HashMap<String, HashMap<usize, bool>>` (:33). Here
            // the loop below really does call it once per pre-activation
            // element — unlike the α-init twin in
            // `state/alpha/graph_init.rs`, this loop has no stable-neuron skip
            // ahead of the constraint check.
            //
            // The guard can never UNDER-trigger. `relu_lookup` has exactly one
            // writer, `GraphSplitHistory::add_constraint` (:117-118), which
            // pushes the same constraint onto the public `constraints` vec (:24)
            // in the same call, and no non-test code shrinks, reorders or
            // rewrites that vec — the only two `constraints[0].node_name = …`
            // sites are `engine/graph/resident_bab/operands.rs:1157` and `:1197`,
            // both inside the `#[cfg(test)] mod tests` that opens at :746. So a
            // node absent from `constraints` cannot hold a `relu_lookup` entry,
            // and `is_constrained` would have returned `None` for it anyway.
            //
            // It can over-trigger in exactly one way: a node carrying a ReLU
            // constraint on some OTHER neuron index. That merely runs the exact
            // lookup the original always ran, so the skip decision — and
            // therefore the returned candidate list — is unchanged. GenBaB is
            // NOT such a case: those constraints land in
            // `genbab_constraints`/`genbab_lookup` (:26, :38), which
            // `is_constrained` never reads, so a GenBaB-only node makes this
            // guard false and the exact lookup would have missed it too.
            let node_may_be_constrained = domain
                .history
                .constraints
                .iter()
                .any(|c| c.node_name() == node_name.as_str());

            // #flatten-to-read-a-scalar: this used to call `pre_bounds.flatten()`,
            // which COPIES both endpoints into fresh `Vec<f32>`s
            // (`ny-tensor bounded_tensor/core/shape_ops.rs:130-138`) purely to
            // read two scalars per neuron — a full activation-sized pair of
            // allocations per ReLU node, on a scan the batched lane maps over
            // every domain of the batch (`multi_objective/batched/batched_multi.rs`
            // runs it under `domains_to_process.par_iter()`). `flatten` fills
            // those vectors with `self.lower.iter()` — `TrackedArrayD` derefs to
            // `ArrayD<f32>` (`core/allocation_provenance.rs:217-224`), and
            // `lower()` hands back that same array (`core/mod.rs:110-112`) — then
            // wraps them with `from_parts_with_l2` (`core/mod.rs:162-172`), which
            // is a bare struct literal: no repair, no clamp, no arithmetic. Both
            // sides therefore go through the same `ArrayD::iter()`, so the zip
            // below yields the same f32 values in the same order the flattened
            // copy held, for any layout. This is the read the sibling
            // `find_unstable_graph_neurons_multi_bounded` already documents.
            //
            // Zip length caveat, stated precisely because it is a behaviour
            // change on one path: this scans `min(lower.len(), upper.len())`,
            // whereas the old loop ran to `flat.len()` = `lower.len()` and
            // indexed `upper`. Those agree only while the two endpoints have
            // equal shapes, and equal shapes are a CALLER CONTRACT here, not a
            // construction guarantee — `BoundedTensor::new`
            // (`ny-tensor .../core/constructors.rs:142-148`) does reject a
            // mismatch in every build, but the crate-internal
            // `from_parts_unchecked` (`core/mod.rs:136-143`) enforces it with
            // `debug_assert_eq!` only and `from_parts_with_l2` not at all. So in
            // a release build with a violated contract and a shorter `upper`,
            // the old code panicked on the index and this zip stops short
            // instead, leaving the tail neurons out of the returned candidate
            // list rather than aborting.
            for (neuron_idx, (&l, &u)) in pre_bounds
                .lower()
                .iter()
                .zip(pre_bounds.upper().iter())
                .enumerate()
            {
                if node_may_be_constrained
                    && domain
                        .history
                        .is_constrained(node_name, neuron_idx)
                        .is_some()
                {
                    continue;
                }

                if l < 0.0 && u > 0.0 {
                    unstable.push((node_name.clone(), neuron_idx));
                }
            }
        }

        unstable
    }

    /// Deadline-polled unstable discovery for the bounded shared executor.
    ///
    /// Unlike the historical helper, this reads already-owned arrays directly
    /// instead of flattening/cloning each producer tensor. The admission gate
    /// bounds the total candidate metadata; fallible reservation keeps an
    /// allocator refusal structured.
    pub(in crate::beta_crown::engine) fn find_unstable_graph_neurons_multi_bounded(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        relu_nodes: &[String],
        deadline: Option<std::time::Instant>,
    ) -> Result<Vec<(String, usize)>> {
        const METADATA_BUDGET_BYTES: usize = 256 * 1024 * 1024;

        let mut unstable = Vec::new();
        for node_name in relu_nodes {
            if deadline.is_some_and(|authority| std::time::Instant::now() >= authority) {
                return Err(ny_core::NyError::DeadlineExceeded(
                    "bounded unstable discovery exceeded its deadline".into(),
                ));
            }
            let Some(relu_node) = graph.nodes.get(node_name) else {
                continue;
            };
            if !is_zero_threshold_binary_activation(&relu_node.layer) {
                continue;
            }
            let Some(pre_name) = relu_node.inputs.first() else {
                continue;
            };
            let pre_bounds = if pre_name == NETWORK_INPUT {
                domain.input_bounds.as_ref()
            } else {
                let Some(bounds) = domain.node_bounds.get(pre_name) else {
                    continue;
                };
                bounds.as_ref()
            };
            let entry_bytes = size_of::<(String, usize)>()
                .checked_add(node_name.len())
                .ok_or_else(|| {
                    ny_core::NyError::InvalidSpec(
                        "bounded unstable metadata byte size overflow".into(),
                    )
                })?;
            let required_bytes = pre_bounds.len().checked_mul(entry_bytes).ok_or_else(|| {
                ny_core::NyError::InvalidSpec("bounded unstable metadata size overflow".into())
            })?;
            if required_bytes > METADATA_BUDGET_BYTES {
                return Err(ny_core::NyError::CpuMemoryExceeded {
                    required_bytes,
                    budget_bytes: METADATA_BUDGET_BYTES,
                    site: "bounded unstable discovery",
                });
            }
            unstable.try_reserve(pre_bounds.len()).map_err(|_| {
                ny_core::NyError::CpuMemoryExceeded {
                    required_bytes,
                    budget_bytes: METADATA_BUDGET_BYTES,
                    site: "bounded unstable discovery",
                }
            })?;

            for (neuron_idx, (&lower, &upper)) in pre_bounds
                .lower()
                .iter()
                .zip(pre_bounds.upper().iter())
                .enumerate()
            {
                if neuron_idx % 1_024 == 0
                    && deadline.is_some_and(|authority| std::time::Instant::now() >= authority)
                {
                    return Err(ny_core::NyError::DeadlineExceeded(
                        "bounded unstable discovery exceeded its deadline".into(),
                    ));
                }
                if domain
                    .history
                    .is_constrained(node_name, neuron_idx)
                    .is_some()
                {
                    continue;
                }
                if lower < 0.0 && upper > 0.0 {
                    unstable.push((node_name.clone(), neuron_idx));
                }
            }
        }
        if deadline.is_some_and(|authority| std::time::Instant::now() >= authority) {
            return Err(ny_core::NyError::DeadlineExceeded(
                "bounded unstable discovery exceeded its deadline".into(),
            ));
        }
        Ok(unstable)
    }

    /// Select branch point for multi-objective domain.
    ///
    /// When `BranchingHeuristic::BoundImpact` is configured, uses the shared
    /// BaBSR score kernel with signed lA and recoverable producer bias.
    ///
    /// `LargestBoundWidth` scores by real pre-activation width, and the
    /// kFSB family routes through the graph kFSB machinery (#mo-scorer-fix).
    /// Otherwise, falls back to intercept-only: intercept = (-l * u) / (u - l).
    pub(in crate::beta_crown::engine) fn select_graph_branch_multi(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        unstable: &[(String, usize)],
        // #branching-la: the multi-objective margin rows `c` (one per objective), so the
        // scorer can seed with the aggregation-critical objective (objective-directed BaBSR).
        // Empty (tests / callers without objectives) → legacy intercept-only behavior.
        objectives: &[Vec<f32>],
        // Thresholds are required because criticality is defined by proof
        // margin, not by the raw lower/upper bound.
        thresholds: &[f32],
        // #mo-scorer-fix: engine for the kFSB-family child-bound evaluation.
        // `None` skips child evaluation inside the kFSB machinery gracefully.
        engine: Option<&dyn ny_core::GemmEngine>,
    ) -> Result<(String, usize, f32)> {
        // #phase-cost: see `ny_core::phase_cost` (observability only).
        let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::BranchSelect);
        // Guard against empty unstable list.
        if unstable.is_empty() {
            return Err(ny_core::NyError::InternalError(
                "select_graph_branch_multi: no unstable neurons to branch on".into(),
            ));
        }

        // One aggregation- and direction-aware row from the domain drives
        // every objective-directed advisory scorer below. A malformed
        // scheduling view fails open to legacy
        // objective-agnostic scoring: branch choice must never become a proof
        // authority or erase a domain.
        let critical_objective_idx = match domain.critical_objective_index(thresholds) {
            Ok(index) => index,
            Err(error) => {
                tracing::warn!(
                    "multi-objective critical-row selection failed; \
                         using objective-agnostic branch scoring: {error}"
                );
                None
            }
        };

        // #gather-score (boxlift charter Inc 4 — DARK, NY_MO_GATHER_SCORE=1):
        // advisory candidate choice from the wide-β lane's harvested
        // |A_lower| scores (zero-cost kFSB surrogate from the already-paid
        // gather). Cache miss or empty intersection with `unstable` falls
        // through to the shipped scorer byte-identically; the split is an
        // exact partition either way, so score quality can never affect
        // soundness — only search order.
        {
            use crate::beta_crown::engine::graph::propagation::batched::gather_score;
            if let Some(mode) = gather_score::gather_score_mode() {
                let fp = gather_score::beta_split_fingerprint(&domain.beta_state);
                if let Some(rows) = self.gather_score_cache.get(fp) {
                    let picked = if mode == 2 {
                        // Mode 2: |A| × relaxation slack min(−l,u) — the kFSB
                        // improvement surrogate; slack comes from THIS domain's
                        // pre-activation bounds (the consumer owns them).
                        // The flatten is HOISTED per distinct pre-node (the
                        // first mode-2 tier run cloned the conv pre-activation
                        // tensor PER CANDIDATE inside rayon workers — measured
                        // ~10× per-domain slowdown and a 120GB allocator-arena
                        // OOM on cifar100 2477; one flatten per node per
                        // select call is bounded and cheap).
                        let mut flat_cache: std::collections::HashMap<
                            String,
                            Option<BoundedTensor>,
                        > = std::collections::HashMap::new();
                        gather_score::best_weighted_candidate(&rows, unstable, |name, idx| {
                            let flat = flat_cache
                                .entry(name.to_string())
                                .or_insert_with(|| {
                                    let pre_name = graph.nodes.get(name)?.inputs.first()?;
                                    let pre = if pre_name == NETWORK_INPUT {
                                        domain.input_bounds.as_ref()
                                    } else {
                                        domain.node_bounds.get(pre_name.as_str())?.as_ref()
                                    };
                                    Some(pre.flatten())
                                })
                                .as_ref()?;
                            let l = flat.lower().iter().nth(idx).copied()?;
                            let u = flat.upper().iter().nth(idx).copied()?;
                            (l < 0.0 && u > 0.0).then(|| (-l).min(u))
                        })
                    } else {
                        gather_score::best_scored_candidate(&rows, unstable)
                    };
                    if let Some((name, idx, score)) = picked {
                        // Pick telemetry (probe-gated): which layer class does
                        // the gather score choose — the OOM forensics need it.
                        if crate::beta_gpu_probe_armed() {
                            eprintln!(
                                "[gather-score] mode={mode} pick={name}:{idx} score={score:.3e} depth={}",
                                domain.depth
                            );
                        }
                        return Ok((name, idx, score));
                    }
                }
            }
        }

        // #mo-scorer-fix: kfsb / fsb / width previously ALL collapsed onto the
        // intercept-only fallback in this selector — only `BoundImpact` ever
        // computed a real score (`want_babsr` below), so the three heuristics
        // produced IDENTICAL branch picks on the multi-objective graph lane
        // (the measured metaroom degeneracy). Route each heuristic to its real
        // scorer: width → largest pre-activation width (mirrors
        // `select_largest_width_neuron`), kFSB family → the graph kFSB
        // machinery (BaBSR/intercept prescore + child-bound evaluation) seeded
        // with the aggregation-critical unverified objective row. The dark experiment gates
        // (NY_BRANCH_LA / NY_BRANCH_STEM) keep the legacy flow so their
        // measurements stay comparable; `NY_MO_SCORER_FIX=1` enables the fixed
        // scorers. Advisory-only (the pick only chooses WHICH ReLU to
        // partition) ⇒ soundness-free — but branching ORDER changes verdicts-
        // within-budget on every kfsb/auto track, and only metaroom was
        // regression-A/B'd (its preset uses `impact`, unaffected), so the fix
        // ships DEFAULT-OFF until the cross-track A/B lands
        // (#scorer-fix-default-off; the upfront-attack lane taught us not to
        // default-ON a behavior change verified on one track).
        //
        // A/B measured (relusplitter MO-kfsb lane) — the effect is INSTANCE-
        // DEPENDENT and does NOT support default-ON. ZERO verdict contradictions
        // in any run (soundness is identical, as expected for an advisory pick).
        // Single instance, 60s CPU (oval21 base/img395-eps0.0038): ON pruned far
        // more — 1716 domains / 402 subdomains verified / depth 30 vs OFF 15359 /
        // 52 / 92 — because OFF branched badly there (0.3% verify rate). But a
        // broader 9-instance wgpu sweep at 75s (base/deep) came out mixed-to-
        // negative: all 9 timed out both arms; verified-subdomains was 1 instance
        // OFF-better (img4386: 1024 vs 68), 0 ON-better, 8 ties at zero. The cause
        // is visible in domains_explored: kFSB's per-decision child-bound eval is
        // costly, so ON explores FAR fewer domains per unit time (127-253 vs OFF's
        // 2.6k-5.6k) — its better branch CHOICES win only when OFF is branching
        // badly, and its higher per-decision COST loses when OFF is already fine.
        // Net: no within-budget solve gain, one regression. Stays gated; the fix
        // is available opt-in via NY_MO_SCORER_FIX=1 for the cases it helps.
        //
        // FINAL 2026-07-18 (docs/MEASURED_KFSB_GATES.md): the full 3-arm matrix
        // at native 180s budgets (8 relusplitter MO instances; arms BASE / FIX /
        // FIX+NY_KFSB_BATCH_EVAL) resolved the question — solved-count 1=1=1
        // (the same PGD sat, same 2s), every other instance timed out in every
        // arm, ZERO contradictions (third independent confirmation of the
        // advisory-only claim). FIX's per-domain quality is real (191 vs 96
        // subdomains verified from 371 vs 18079 explored on img8258) but never
        // converted to a within-budget solve. Default stays OFF; re-open only
        // with a solved-count win on competition hardware.
        let scorer_fix = matches!(std::env::var("NY_MO_SCORER_FIX").ok().as_deref(), Some("1"));
        let typed_critical_kfsb = self.config.use_multi_objective_critical_kfsb
            && domain.aggregation() == ObjectiveAggregation::Conjunctive
            && matches!(
                self.config.branching_heuristic,
                BranchingHeuristic::Kfsb | BranchingHeuristic::KfsbInterceptOnly
            );
        let experiments_active = std::env::var("NY_BRANCH_LA").ok().as_deref() == Some("1")
            || std::env::var("NY_BRANCH_LA_PROBE").ok().as_deref() == Some("1")
            || std::env::var("NY_BRANCH_STEM").ok().as_deref() == Some("1");
        if !experiments_active {
            match self.config.branching_heuristic {
                BranchingHeuristic::LargestBoundWidth if scorer_fix => {
                    // FAIL-OPEN: no scorable width (all-NaN bounds) falls
                    // through to the historical intercept ranking.
                    if let Some(pick) =
                        self.select_graph_branch_multi_width(graph, domain, unstable)
                    {
                        return Ok(pick);
                    }
                }
                BranchingHeuristic::Kfsb
                | BranchingHeuristic::KfsbInterceptOnly
                | BranchingHeuristic::FilteredSmartBranching
                    if scorer_fix || typed_critical_kfsb =>
                {
                    // FAIL-OPEN: a scoring/child-eval error must degrade to the
                    // historical intercept ranking, not propagate — callers
                    // treat a branch-selection Err as a PropagationFailure
                    // (unresolved parent), which would be a regression for a
                    // merely-advisory scorer.
                    match self.select_graph_branch_multi_kfsb(
                        graph,
                        domain,
                        unstable,
                        objectives,
                        critical_objective_idx,
                        engine,
                    ) {
                        Ok(Some(pick)) => return Ok(pick),
                        // No unverified objective row (legacy/test callers) —
                        // fall through to the historical intercept ranking.
                        Ok(None) => {}
                        Err(e) => {
                            tracing::warn!(
                                "multi-objective kFSB scoring failed; falling back to intercept: {e}"
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        // #branching-la: OBJECTIVE-DIRECTED, CONV-CORRECT BaBSR scores (INC1/INC2). Seed
        // the coefficient backward with the aggregation-critical unverified objective
        // margin row `c`, so scores measure each candidate ReLU's signed influence on the
        // relevant row (not a 100-way-diluted average). Computed under legacy BoundImpact, or
        // when the lA branch (NY_BRANCH_LA) / its differential probe (NY_BRANCH_LA_PROBE)
        // is on. Advisory-only (the score only RANKS candidates, never read by any verdict)
        // ⇒ soundness-free.
        let la_enabled = std::env::var("NY_BRANCH_LA").ok().as_deref() == Some("1");
        let la_probe = std::env::var("NY_BRANCH_LA_PROBE").ok().as_deref() == Some("1");
        // #branching-la HYBRID: the objective-directed conv-adjoint backward is ~4x costlier
        // per decision, but it only PAYS OFF on the hard TAIL (few stragglers, where aiming
        // at the right neuron matters). Early BaB (many stragglers) closes fine with cheap
        // intercept and any split helps. So only spend lA when the number of unverified
        // objectives ≤ NY_BRANCH_LA_MAX_ACTIVE (default 3) — targeting the last straggler(s)
        // without slowing the easy part. Advisory-only ⇒ soundness-free.
        let n_active = domain
            .objective_bounds
            .iter()
            .enumerate()
            .filter(|(i, _)| !domain.verified.get(*i).copied().unwrap_or(false))
            .count();
        let la_max_active = std::env::var("NY_BRANCH_LA_MAX_ACTIVE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);
        let la_active = la_enabled && n_active <= la_max_active;
        // #branch-stem (dark, `NY_BRANCH_STEM=1`, default OFF = byte-identical):
        // STEM-FIRST branching experiment. The nonconvexity census (window-gate,
        // cifar100 resnet_medium deep domains) puts the unstable mass in the STEM
        // (Relu_2: 233, Relu_5: 135, Relu_13: 56 vs ≤ 37 anywhere later), and a
        // stem split constrains nearly the whole net DOWNSTREAM (neuron splits
        // carry information only below the split layer) — yet both the intercept
        // and the objective-lA selector always pick mid/late layers. The gate
        // restricts branch selection to the EARLIEST `NY_BRANCH_STEM_LAYERS`
        // (default 3) unstable ReLU layers in exec order — or an explicit
        // `NY_BRANCH_STEM_NODES=a,b,c` list — at ReLU-history depths below
        // `NY_BRANCH_STEM_K` (default 8). The restriction engages only while a
        // named candidate remains unstable and scorable; an empty/unscorable
        // restricted set fails open to the normal selector, as does every
        // depth at/above K. Within the stem the usual LEGACY score ranks (lA
        // when active, else intercept).
        //
        // Interaction with #mo-scorer-fix is deliberate: `NY_BRANCH_STEM=1`
        // makes `experiments_active` above, so `NY_MO_SCORER_FIX` is bypassed
        // even if separately present. Combining the fixed scorer with this
        // experiment needs its own reviewed runtime/provenance treatment; the
        // cGAN stem canary uses legacy intercept ranking. Advisory-only (picks
        // WHICH ReLU to partition) ⇒ soundness-free.
        let stem_enabled = std::env::var("NY_BRANCH_STEM").ok().as_deref() == Some("1");
        let stem_k = std::env::var("NY_BRANCH_STEM_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8);
        let stem_active = stem_enabled && domain.history.constraints.len() < stem_k;
        let stem_nodes: Option<std::collections::HashSet<String>> = if stem_active {
            let explicit = std::env::var("NY_BRANCH_STEM_NODES").ok().and_then(|raw| {
                let set: std::collections::HashSet<String> = raw
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                (!set.is_empty()).then_some(set)
            });
            match explicit {
                Some(set) => Some(set),
                None => {
                    // Earliest-N distinct candidate layers by exec order.
                    let n_layers = std::env::var("NY_BRANCH_STEM_LAYERS")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(3)
                        .max(1);
                    let candidate_layers: std::collections::HashSet<&str> =
                        unstable.iter().map(|(n, _)| n.as_str()).collect();
                    graph.exec_order().ok().map(|exec| {
                        exec.iter()
                            .filter(|n| candidate_layers.contains(n.as_str()))
                            .take(n_layers)
                            .cloned()
                            .collect::<std::collections::HashSet<String>>()
                    })
                }
            }
        } else {
            None
        };
        let want_babsr = la_active
            || la_probe
            || matches!(
                self.config.branching_heuristic,
                BranchingHeuristic::BoundImpact
            );
        let babsr_scores = if want_babsr {
            let seed_row: Option<&[f32]> = critical_objective_idx
                .and_then(|i| objectives.get(i))
                .map(|v| v.as_slice());
            // #branching-la stop-early: only the UNSTABLE ReLU nodes need a score, so stop
            // the backward once all are reached — skips the input-side (large-spatial) convs.
            let unstable_nodes: std::collections::HashSet<String> =
                unstable.iter().map(|(n, _)| n.clone()).collect();
            self.compute_graph_babsr_scores_from_bounds(
                graph,
                &domain.node_bounds,
                &domain.input_bounds,
                KfsbReduceOp::Min,
                seed_row,
                Some(&unstable_nodes),
            )?
        } else {
            std::collections::HashMap::new()
        };

        // Track the intercept-argmax AND the babsr-argmax SEPARATELY so the differential
        // probe (INC0) can log whether the objective-directed lA score picks a DIFFERENT
        // neuron than intercept-only — the decisive "is NY aiming at the wrong neuron?"
        // measurement — before any behavior change.
        let mut best_intercept = unstable[0].clone();
        let mut best_intercept_score = f32::NEG_INFINITY;
        let mut best_babsr = unstable[0].clone();
        let mut best_babsr_score = f32::NEG_INFINITY;
        // #branch-stem: bests restricted to the stem layer set (None until a
        // stem candidate is seen — an empty stem falls through to normal).
        let mut best_stem_intercept: Option<(String, usize, f32)> = None;
        let mut best_stem_babsr: Option<(String, usize, f32)> = None;
        // #branch-stem diagnosis probe (`NY_BRANCH_STEM_PROBE=1` or the lA
        // probe): per-LAYER max scores — the direct answer to "are stem scores
        // computed and dominated, or excluded?".
        let stem_probe = std::env::var("NY_BRANCH_STEM_PROBE").ok().as_deref() == Some("1");
        let mut layer_stats: std::collections::BTreeMap<String, (f32, f32, usize)> =
            std::collections::BTreeMap::new();

        // Cache flattened pre-activation bounds per producer node. Without this,
        // `pre_bounds.flatten()` ran once PER NEURON below — O(neurons × tensor)
        // ≈ O(neurons²) for conv layers (e.g. 43K-element activations × 43K
        // neurons), which made a single branch decision on conv-heavy models
        // (yolo/tinyimagenet) consume the whole timeout. Flattening once per
        // unique producer makes it O(unique_nodes × tensor + neurons). The
        // numerical result is identical. (#perf-branch-flatten)
        let mut flat_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        for (node_name, neuron_idx) in unstable {
            let relu_node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            if !is_zero_threshold_binary_activation(&relu_node.layer) {
                continue;
            }
            // Skip nodes with empty inputs instead of fabricating NETWORK_INPUT.
            // A ReLU with no inputs is a graph construction bug — using network
            // input bounds would produce unsound branching decisions.
            let pre_name = match relu_node.inputs.first() {
                Some(s) => s.as_str(),
                None => {
                    tracing::warn!(node = %node_name, "ReLU node has empty inputs — skipping");
                    continue;
                }
            };
            if !flat_cache.contains_key(pre_name) {
                let pre_bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
                    domain.input_bounds.as_ref()
                } else {
                    match domain.node_bounds.get(pre_name) {
                        Some(b) => b.as_ref(),
                        None => continue,
                    }
                };
                flat_cache.insert(pre_name.to_string(), pre_bounds.flatten());
            }
            let flat = &flat_cache[pre_name];
            if *neuron_idx >= flat.len() {
                continue;
            }

            let l = flat.lower()[[*neuron_idx]];
            let u = flat.upper()[[*neuron_idx]];
            if l < 0.0 && u > 0.0 {
                let intercept = relu_intercept_score(l, u);
                if intercept > best_intercept_score {
                    best_intercept_score = intercept;
                    best_intercept = (node_name.clone(), *neuron_idx);
                }
                let mut babsr_here: Option<f32> = None;
                if !babsr_scores.is_empty() {
                    babsr_here = babsr_scores
                        .get(&(node_name.clone(), *neuron_idx))
                        .map(|p| p.main_score);
                    // Missing score → 0.0 (legacy BoundImpact fallback), preserved.
                    let s = babsr_here.unwrap_or(0.0);
                    if s > best_babsr_score {
                        best_babsr_score = s;
                        best_babsr = (node_name.clone(), *neuron_idx);
                    }
                }
                // #branch-stem: stem-restricted bests.
                if stem_nodes
                    .as_ref()
                    .is_some_and(|stem| stem.contains(node_name))
                {
                    if best_stem_intercept
                        .as_ref()
                        .is_none_or(|(_, _, s)| intercept > *s)
                    {
                        best_stem_intercept = Some((node_name.clone(), *neuron_idx, intercept));
                    }
                    if let Some(s) = babsr_here {
                        if best_stem_babsr.as_ref().is_none_or(|(_, _, b)| s > *b) {
                            best_stem_babsr = Some((node_name.clone(), *neuron_idx, s));
                        }
                    }
                }
                if stem_probe || la_probe {
                    let e = layer_stats.entry(node_name.clone()).or_insert((
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY,
                        0,
                    ));
                    e.0 = e.0.max(babsr_here.unwrap_or(f32::NEG_INFINITY));
                    e.1 = e.1.max(intercept);
                    e.2 += 1;
                }
            }
        }

        // #branch-stem diagnosis probe: one line per decision with each layer's
        // candidate count and max scores. `babsr=-inf` for a layer with
        // candidates ⇒ the backward never scored it (EXCLUDED); a finite max
        // below the global best ⇒ computed-but-DOMINATED.
        if stem_probe || (la_probe && stem_enabled) {
            let per_layer: Vec<String> = layer_stats
                .iter()
                .map(|(n, (b, i, c))| format!("{n}:n={c}:babsr={b:.5}:icpt={i:.5}"))
                .collect();
            eprintln!(
                "[branch-stem] depth={} stem_active={stem_active} layers=[{}]",
                domain.history.constraints.len(),
                per_layer.join(" "),
            );
        }

        // INC0 differential probe: does the objective-directed lA score pick a DIFFERENT
        // neuron than intercept-only? Frequent disagreement confirms intercept is not
        // already aiming at the high-lA neuron (the convergence-speed defect).
        if la_probe {
            eprintln!(
                "[branch-la] agree={} intercept=({},{}) i_score={:.5} | la=({},{}) la_score={:.5} n_babsr={}",
                best_intercept == best_babsr,
                best_intercept.0,
                best_intercept.1,
                best_intercept_score,
                best_babsr.0,
                best_babsr.1,
                best_babsr_score,
                babsr_scores.len(),
            );
        }

        // Split by the lA-directed neuron when enabled (NY_BRANCH_LA or legacy BoundImpact)
        // AND a babsr score set exists; else the intercept pick (default UNCHANGED). Sound
        // either way (the returned (node,neuron) only selects which ReLU to partition on).
        let use_la = (la_active
            || matches!(
                self.config.branching_heuristic,
                BranchingHeuristic::BoundImpact
            ))
            && !babsr_scores.is_empty();

        // #branch-stem: within the stem window (ReLU-history depth < K),
        // return the best STEM candidate — ranked by the legacy score family
        // this experiment deliberately retains (lA when active with computed
        // stem scores, else intercept). An empty/unscorable named set falls
        // through to normal legacy selection.
        if stem_active {
            let stem_pick = if use_la && best_stem_babsr.is_some() {
                best_stem_babsr
            } else {
                best_stem_intercept
            };
            if let Some((node, neuron, score)) = stem_pick {
                if stem_probe {
                    eprintln!(
                        "[branch-stem] PICK ({node},{neuron}) score={score:.5} \
                         (global la=({},{}) {:.5} icpt=({},{}) {:.5})",
                        best_babsr.0,
                        best_babsr.1,
                        best_babsr_score,
                        best_intercept.0,
                        best_intercept.1,
                        best_intercept_score,
                    );
                }
                return Ok((node, neuron, score));
            } else if stem_probe {
                eprintln!("[branch-stem] no stem candidate — normal selection");
            }
        }

        if use_la {
            Ok((best_babsr.0, best_babsr.1, best_babsr_score))
        } else {
            Ok((best_intercept.0, best_intercept.1, best_intercept_score))
        }
    }

    /// Allocation-free, deadline-polled advisory selector for the bounded
    /// shared executor.
    ///
    /// Objective-directed BaBSR and kFSB retain coefficient maps or simulated
    /// children before the bounded GEMM facade can poll. Intercept ranking uses
    /// only the already-owned pre-activation arrays and chooses an equally
    /// exhaustive ReLU partition, so proof semantics are unchanged.
    pub(in crate::beta_crown::engine) fn select_graph_branch_multi_bounded_intercept(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        unstable: &[(String, usize)],
        deadline: Option<std::time::Instant>,
    ) -> Result<(String, usize, f32)> {
        // #phase-cost: see `ny_core::phase_cost` (observability only).
        let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::BranchSelect);
        if unstable.is_empty() {
            return Err(ny_core::NyError::InternalError(
                "bounded multi-objective branch selector has no unstable neurons".into(),
            ));
        }

        let mut best: Option<(String, usize, f32)> = None;
        let mut group_start = 0usize;
        while group_start < unstable.len() {
            if deadline.is_some_and(|authority| std::time::Instant::now() >= authority) {
                return Err(ny_core::NyError::DeadlineExceeded(
                    "bounded multi-objective intercept selection exceeded its deadline".into(),
                ));
            }
            let node_name = &unstable[group_start].0;
            let group_end = unstable[group_start..]
                .iter()
                .position(|(candidate, _)| candidate != node_name)
                .map_or(unstable.len(), |offset| group_start + offset);
            let Some(relu_node) = graph.nodes.get(node_name) else {
                group_start = group_end;
                continue;
            };
            if !is_zero_threshold_binary_activation(&relu_node.layer) {
                group_start = group_end;
                continue;
            }
            let Some(pre_name) = relu_node.inputs.first() else {
                group_start = group_end;
                continue;
            };
            let pre_bounds = if pre_name == NETWORK_INPUT {
                domain.input_bounds.as_ref()
            } else {
                let Some(bounds) = domain.node_bounds.get(pre_name) else {
                    group_start = group_end;
                    continue;
                };
                bounds.as_ref()
            };

            let mut candidates = unstable[group_start..group_end].iter().peekable();
            for (neuron_idx, (&lower, &upper)) in pre_bounds
                .lower()
                .iter()
                .zip(pre_bounds.upper().iter())
                .enumerate()
            {
                if neuron_idx % 1_024 == 0
                    && deadline.is_some_and(|authority| std::time::Instant::now() >= authority)
                {
                    return Err(ny_core::NyError::DeadlineExceeded(
                        "bounded multi-objective intercept selection exceeded its deadline".into(),
                    ));
                }
                while candidates
                    .peek()
                    .is_some_and(|(_, candidate_idx)| *candidate_idx < neuron_idx)
                {
                    candidates.next();
                }
                let Some((_, candidate_idx)) = candidates.peek() else {
                    break;
                };
                if *candidate_idx != neuron_idx {
                    continue;
                }
                candidates.next();
                if lower < 0.0 && upper > 0.0 {
                    let score = relu_intercept_score(lower, upper);
                    if best
                        .as_ref()
                        .is_none_or(|(_, _, incumbent)| score > *incumbent)
                    {
                        best = Some((node_name.clone(), neuron_idx, score));
                    }
                }
            }
            group_start = group_end;
        }
        if deadline.is_some_and(|authority| std::time::Instant::now() >= authority) {
            return Err(ny_core::NyError::DeadlineExceeded(
                "bounded multi-objective intercept selection exceeded its deadline".into(),
            ));
        }
        best.ok_or_else(|| {
            ny_core::NyError::InternalError(
                "bounded multi-objective branch selector found no scorable neuron".into(),
            )
        })
    }

    /// #mo-scorer-fix: real `LargestBoundWidth` scoring for the multi-objective
    /// selector — the neuron with the widest pre-activation interval (u - l),
    /// mirroring the sequential lane's `select_largest_width_neuron`. NaN
    /// widths are skipped; `None` (nothing scorable) lets the caller
    /// fall through to the intercept ranking. Advisory-only ⇒ soundness-free.
    fn select_graph_branch_multi_width(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        unstable: &[(String, usize)],
    ) -> Option<(String, usize, f32)> {
        let mut best: Option<(String, usize, f32)> = None;
        let mut flat_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();
        for (node_name, neuron_idx) in unstable {
            let relu_node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            if !is_zero_threshold_binary_activation(&relu_node.layer) {
                continue;
            }
            let pre_name = match relu_node.inputs.first() {
                Some(s) => s.as_str(),
                None => continue,
            };
            if !flat_cache.contains_key(pre_name) {
                let pre_bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
                    domain.input_bounds.as_ref()
                } else {
                    match domain.node_bounds.get(pre_name) {
                        Some(b) => b.as_ref(),
                        None => continue,
                    }
                };
                flat_cache.insert(pre_name.to_string(), pre_bounds.flatten());
            }
            let flat = &flat_cache[pre_name];
            if *neuron_idx >= flat.len() {
                continue;
            }
            let l = flat.lower()[[*neuron_idx]];
            let u = flat.upper()[[*neuron_idx]];
            if l < 0.0 && u > 0.0 {
                let width = u - l;
                if width.is_nan() {
                    continue;
                }
                if best.as_ref().is_none_or(|(_, _, s)| width > *s) {
                    best = Some((node_name.clone(), *neuron_idx, width));
                }
            }
        }
        best
    }

    /// #mo-scorer-fix: kFSB-family selection for the multi-objective lane.
    ///
    /// Seeds the graph kFSB machinery (`select_graph_branch_kfsb_in_gpu_batched`:
    /// BaBSR/intercept prescore, top-k candidate filtering, per-candidate child
    /// bound evaluation with the configured `kfsb_reduce_op`) with the
    /// aggregation-critical objective's margin row — the same rule the
    /// objective-directed BaBSR seed uses. Returns `Ok(None)` when no
    /// unverified objective row exists (legacy/test callers without
    /// objectives), letting the caller fall back to intercept ranking.
    ///
    /// The `GraphBabDomain` shim mirrors `graph_bab_domain_shim`
    /// (batched_dense_specs.rs): identical history / node_bounds / input /
    /// β / α; `cached_la = None` (no warm start — fine for scoring);
    /// lower/upper seeded from the critical objective's bounds (accounting
    /// metadata for child-bound evaluation pruning, not a verdict source).
    /// Advisory-only (branch choice) ⇒ soundness-free.
    fn select_graph_branch_multi_kfsb(
        &self,
        graph: &GraphNetwork,
        domain: &MultiObjectiveGraphBabDomain,
        unstable: &[(String, usize)],
        objectives: &[Vec<f32>],
        critical_objective_idx: Option<usize>,
        engine: Option<&dyn ny_core::GemmEngine>,
    ) -> Result<Option<(String, usize, f32)>> {
        let Some(critical_idx) = critical_objective_idx else {
            return Ok(None);
        };
        let Some(objective_row) = objectives.get(critical_idx) else {
            return Ok(None);
        };
        let (lower_bound, upper_bound) = domain
            .objective_bounds
            .get(critical_idx)
            .copied()
            .unwrap_or((0.0, 0.0));
        let shim = GraphBabDomain {
            history: domain.history.clone(),
            node_bounds: domain.node_bounds.to_shared_hash_map(),
            lower_bound,
            upper_bound,
            depth: domain.depth,
            priority: domain.priority,
            input_bounds: domain.input_bounds.clone(),
            beta_state: domain.beta_state.clone(),
            alpha_state: domain.alpha_state.clone(),
            cached_la: None,
            // #cone-delta: `node_bounds`/`history` transfer verbatim, so the
            // delta transfers verbatim with them.
            delta_pre_nodes: domain.delta_pre_nodes.clone(),
        };
        self.select_graph_branch_kfsb_in_gpu_batched(graph, &shim, unstable, objective_row, engine)
            .map(Some)
    }

    /// Find unstable neurons in graph ReLU nodes.
    ///
    /// Returns a list of (node_name, neuron_idx) pairs where the pre-activation
    /// bounds cross zero (l < 0 < u) and are not already constrained.
    pub(in crate::beta_crown::engine) fn find_unstable_graph_neurons(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        relu_nodes: &[String],
    ) -> Vec<(String, usize)> {
        let mut unstable = Vec::with_capacity(relu_nodes.len());
        // This node's ReLU-constrained neuron indices, ascending, refilled once
        // per node at the site below. Reused across nodes, so it keeps the
        // capacity the busiest node has needed and stops reallocating after the
        // first few refills. It holds at most one entry per ReLU constraint in
        // the history — BaB depth — against the thousands of pre-activation
        // elements the inner loop walks.
        let mut constrained_idx: Vec<usize> = Vec::new();

        for node_name in relu_nodes {
            let relu_node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            if !is_zero_threshold_binary_activation(&relu_node.layer) {
                continue;
            }
            // Skip nodes with empty inputs instead of fabricating NETWORK_INPUT.
            // A ReLU with no inputs is a graph construction bug — using network
            // input bounds would produce unsound branching decisions.
            let pre_name = match relu_node.inputs.first() {
                Some(s) => s.as_str(),
                None => {
                    tracing::warn!(node = %node_name, "ReLU node has empty inputs — skipping");
                    continue;
                }
            };

            let pre_bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
                domain.input_bounds.as_ref()
            } else {
                match domain.node_bounds.get(pre_name) {
                    Some(b) => b.as_ref(),
                    None => continue,
                }
            };

            // Resolve the node's ReLU constraints ONCE per node. The loop below
            // used to call `is_constrained` (`branching/graph_history.rs:172`)
            // once per pre-activation ELEMENT: unlike the alpha-init twin in
            // `state/alpha/graph_init.rs`, the historical loop had no
            // stable-neuron skip ahead of the constraint check, so it paid for
            // stable neurons too (the ordering below now supplies one). Every
            // such call hashes `node_name` against the two-level
            // `relu_lookup: HashMap<String, HashMap<usize, bool>>` (:33). That
            // is `std`'s `HashMap` (`:9`), i.e. a full SipHash of the name's
            // bytes, before the inner `usize` probe. What is left per neuron is
            // a binary search over `constrained_idx`.
            //
            // `constrained_idx` is a SUPERSET of this node's `relu_lookup` keys,
            // never a subset, and `is_constrained` still makes the final call —
            // so the skip decision, and therefore the returned candidate list,
            // is exactly what it was. The superset direction holds because
            // `relu_lookup` is a PRIVATE field with exactly two writers:
            // `GraphSplitHistory::new()`/`derive(Default)` (empty) and
            // `add_constraint` (:117-118), which pushes the same constraint onto
            // the public `constraints` vec (:24) in the same call (:122).
            // Nothing else can desynchronize them: there is no
            // `GraphSplitHistory` struct literal outside its own module (the
            // private field forbids one), the type derives no serde impl, and
            // the only non-test code that touches `constraints` is that single
            // `push` — the two `constraints[0].node_name = …` sites,
            // `engine/graph/resident_bab/operands.rs:1157` and `:1197`, are both
            // inside the `#[cfg(test)] mod tests` that opens at :746.
            //
            // GenBaB constraints are correctly absent: they land in
            // `genbab_constraints`/`genbab_lookup` (:26, :38), which
            // `is_constrained` never reads, so a GenBaB-only node yields an
            // empty `constrained_idx` and the old exact lookup would have missed
            // it too. The two stay in agreement rather than one covering for the
            // other.
            constrained_idx.clear();
            constrained_idx.extend(
                domain
                    .history
                    .constraints
                    .iter()
                    .filter(|c| c.node_name() == node_name.as_str())
                    .map(|c| c.neuron_idx()),
            );
            constrained_idx.sort_unstable();

            // #flatten-to-read-a-scalar: this used to call `pre_bounds.flatten()`,
            // which COPIES both endpoints into fresh `Vec<f32>`s
            // (`ny-tensor bounded_tensor/core/shape_ops.rs:130-138`) purely to
            // read two scalars per neuron — a full activation-sized pair of
            // allocations per ReLU node, on a scan that
            // `multi_objective/batched/batched_single.rs` maps over every domain
            // of the batch under `domains_to_process.par_iter()`. `flatten`
            // fills those vectors with `self.lower.iter()` — `TrackedArrayD`
            // derefs to `ArrayD<f32>` (`core/allocation_provenance.rs:217-224`),
            // and `lower()` hands back that same array (`core/mod.rs:110-112`) —
            // then wraps them with `from_parts_with_l2` (`core/mod.rs:162-172`),
            // which is a bare struct literal: no repair, no clamp, no
            // arithmetic. Both sides therefore go through the same
            // `ArrayD::iter()`, so the zip below yields the same f32 values in
            // the same order the flattened copy held, for ANY memory layout —
            // which `as_slice()` would NOT, since it is `None` on a
            // non-standard-layout endpoint. This is the read the sibling
            // `find_unstable_graph_neurons_multi_bounded` already documents
            // ("reads already-owned arrays directly instead of
            // flattening/cloning each producer tensor").
            //
            // Zip length caveat, stated precisely because it is a behaviour
            // change on one path: this scans `min(lower.len(), upper.len())`,
            // whereas the old loop ran to `flat.len()` = `lower.len()`
            // (`core/mod.rs:195-197`) and indexed `upper`. Those agree only
            // while the two endpoints have equal shapes, and equal shapes are a
            // CALLER CONTRACT here, not a construction guarantee —
            // `BoundedTensor::new` (`ny-tensor .../core/constructors.rs:142-148`)
            // does reject a mismatch in every build, but the crate-internal
            // `from_parts_unchecked` (`core/mod.rs:136-143`) enforces it with
            // `debug_assert_eq!` only and `from_parts_with_l2` not at all. So in
            // a release build with a violated contract and a shorter `upper`,
            // the old code panicked on the index and this zip stops short
            // instead, leaving the tail neurons out of the returned candidate
            // list rather than aborting.
            for (neuron_idx, (&l, &u)) in pre_bounds
                .lower()
                .iter()
                .zip(pre_bounds.upper().iter())
                .enumerate()
            {
                // Instability first. Both predicates are pure — `is_constrained`
                // takes `&self` and only reads two maps — so testing them in
                // either order selects the same indices, in the same ascending
                // order, and on a node whose neurons are mostly stable this
                // drops the constraint work for all of them. A NaN endpoint
                // takes the same branch under either order: `l < 0.0 && u > 0.0`
                // is false whenever either compare involves NaN.
                if !(l < 0.0 && u > 0.0) {
                    continue;
                }

                if !constrained_idx.is_empty() && constrained_idx.binary_search(&neuron_idx).is_ok()
                {
                    record_exact_constraint_lookup();
                    if domain
                        .history
                        .is_constrained(node_name, neuron_idx)
                        .is_some()
                    {
                        continue;
                    }
                }

                unstable.push((node_name.clone(), neuron_idx));
            }
        }

        unstable
    }

    /// Select which neuron to branch on using BaBSR or intercept-based scoring.
    ///
    /// Delegates to `select_graph_branches(k=1)` and returns the single best
    /// neuron. See `select_graph_branches` for scoring details.
    pub(in crate::beta_crown::engine) fn select_graph_branch(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
    ) -> Result<(String, usize, f32)> {
        let mut top = self.select_graph_branches(graph, domain, unstable, 1)?;
        // select_graph_branches guarantees non-empty result on success
        Ok(top.remove(0))
    }

    /// Select top-k neurons for multi-depth ReLU splitting.
    ///
    /// Returns up to `k` neurons sorted by descending score. Uses the shared
    /// BaBSR score kernel when `BoundImpact` is configured, otherwise
    /// intercept-only: (-l * u) / (u - l).
    ///
    /// When `k == 1`, returns the single best neuron (used by `select_graph_branch`).
    /// For k > 1, the top-k candidates enable multi-depth splitting.
    ///
    /// Reference: alpha-beta-CROWN `find_topk_scores()` in `base.py:97-168`.
    pub(in crate::beta_crown::engine) fn select_graph_branches(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
        k: usize,
    ) -> Result<Vec<(String, usize, f32)>> {
        // #phase-cost: see `ny_core::phase_cost` (observability only).
        let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::BranchSelect);
        if unstable.is_empty() {
            return Err(ny_core::NyError::InternalError(
                "select_graph_branches: no unstable neurons to branch on".into(),
            ));
        }
        if k == 0 {
            return Err(ny_core::NyError::InternalError(
                "select_graph_branches: k must be >= 1".into(),
            ));
        }

        let babsr_scores = if matches!(
            self.config.branching_heuristic,
            BranchingHeuristic::BoundImpact
        ) {
            self.compute_graph_babsr_scores(graph, domain, KfsbReduceOp::Min)?
        } else {
            std::collections::HashMap::new()
        };

        // Score all unstable neurons
        let mut scored: Vec<(String, usize, f32)> = Vec::with_capacity(unstable.len());

        // Flatten each producer's pre-activation bounds once, not per neuron
        // (O(neurons²) on conv layers otherwise). See select_graph_branch_multi.
        let mut flat_cache: std::collections::HashMap<String, BoundedTensor> =
            std::collections::HashMap::new();

        for (node_name, neuron_idx) in unstable {
            let relu_node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            if !is_zero_threshold_binary_activation(&relu_node.layer) {
                continue;
            }
            let pre_name = match relu_node.inputs.first() {
                Some(s) => s.as_str(),
                None => continue,
            };
            if !flat_cache.contains_key(pre_name) {
                let pre_bounds: &BoundedTensor = if pre_name == NETWORK_INPUT {
                    domain.input_bounds.as_ref()
                } else {
                    match domain.node_bounds.get(pre_name) {
                        Some(b) => b.as_ref(),
                        None => continue,
                    }
                };
                flat_cache.insert(pre_name.to_string(), pre_bounds.flatten());
            }
            let flat = &flat_cache[pre_name];
            if *neuron_idx >= flat.len() {
                continue;
            }

            let l = flat.lower()[[*neuron_idx]];
            let u = flat.upper()[[*neuron_idx]];
            if l < 0.0 && u > 0.0 {
                let intercept = relu_intercept_score(l, u);
                let score = if babsr_scores.is_empty() {
                    intercept
                } else {
                    babsr_scores
                        .get(&(node_name.clone(), *neuron_idx))
                        .copied()
                        .unwrap_or_else(|| {
                            debug!(
                                node = %node_name,
                                neuron = neuron_idx,
                                lower = l,
                                upper = u,
                                "BaBSR graph: no score parts for neuron, using 0.0 fallback"
                            );
                            BabsrScoreParts::default()
                        })
                        .main_score
                };
                scored.push((node_name.clone(), *neuron_idx, score));
            }
        }

        if scored.is_empty() {
            return Err(ny_core::NyError::InternalError(
                "select_graph_branches: no scorable unstable neurons found".into(),
            ));
        }

        // Sort by descending score and take top-k
        scored.sort_by(|a, b| crate::cmp_utils::nan_last_descending_cmp(&a.2, &b.2));
        scored.truncate(k);

        Ok(scored)
    }
}

// Test-only count of the EXACT `GraphSplitHistory::is_constrained` lookups
// performed by [`BetaCrownVerifier::find_unstable_graph_neurons`].
//
// The per-node constraint resolution in that scan is invisible from its return
// value — the candidate list is identical by construction — so reverting it to
// the historical per-neuron `is_constrained` call would break nothing that an
// output-comparing test could see. This counter is what fails instead: the
// fast lane reaches the exact map only for indices that survive BOTH the
// instability test and the sorted-index binary search, so it is bounded by the
// node's own ReLU-constraint count, while the historical loop reached it once
// per pre-activation element.
//
// Thread-local, so tests running in parallel cannot observe each other's
// counts, and `#[cfg(test)]`, so the shipped scan carries no counter at all.
#[cfg(test)]
thread_local! {
    static TEST_EXACT_CONSTRAINT_LOOKUPS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_exact_constraint_lookup() {
    TEST_EXACT_CONSTRAINT_LOOKUPS.with(|count| {
        count.set(count.get().saturating_add(1));
    });
}

// Production no-op twin of the counter above, so the call site is an ordinary
// statement in every build. Without it the `#[cfg(test)]` attribute would leave
// the release build with an `if` whose sole statement is another `if`, which is
// `clippy::collapsible_if`; an empty `#[inline]` function optimizes away
// instead.
#[cfg(not(test))]
#[inline(always)]
fn record_exact_constraint_lookup() {}

#[cfg(test)]
fn reset_test_exact_constraint_lookups() {
    TEST_EXACT_CONSTRAINT_LOOKUPS.with(|count| count.set(0));
}

#[cfg(test)]
fn test_exact_constraint_lookups() -> usize {
    TEST_EXACT_CONSTRAINT_LOOKUPS.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ny_tensor::BoundedTensor;

    use super::{reset_test_exact_constraint_lookups, test_exact_constraint_lookups};
    use crate::beta_crown::branching::GraphNeuronConstraint;
    use crate::beta_crown::domain::GraphBabDomain;
    use crate::beta_crown::{BetaCrownConfig, BetaCrownVerifier};
    use crate::layers::ReLULayer;
    use crate::{GraphNetwork, GraphNode, Layer};

    /// A one-ReLU graph whose producer is an ordinary named node, so the scan
    /// takes the `domain.node_bounds` branch rather than the network-input one.
    fn single_relu_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            "relu0",
            Layer::ReLU(ReLULayer),
            vec!["pre".to_string()],
        ));
        graph
    }

    fn domain_with(pre: BoundedTensor) -> GraphBabDomain {
        let input = BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .expect("input bounds");
        let mut node_bounds = HashMap::new();
        node_bounds.insert("pre".to_string(), pre);
        GraphBabDomain::root(node_bounds, -1.0, 1.0, &input, false).expect("root domain")
    }

    /// FAIL-CLOSED ENGAGEMENT PROOF for BOTH invisible halves of the scan: the
    /// per-node constraint resolution AND the instability-first ordering.
    ///
    /// Drives the shipped `find_unstable_graph_neurons` — not a private copy —
    /// over one node whose 64 pre-activation elements are a deliberate MIX:
    /// even indices unstable (`l < 0 < u`), odd indices stable (alternating
    /// stable-inactive `u <= 0` and stable-active `l >= 0`). The node carries
    /// three ReLU constraints — one on an unstable index, two on stable ones.
    ///
    /// Neither half is visible in the returned candidate list: `is_constrained`
    /// and the instability test are both pure, so either evaluation order and
    /// either resolution strategy selects the same indices in the same
    /// ascending order. The exact-lookup counter is the only observable, and
    /// this test pins it at 1 — the one constrained neuron that is also
    /// unstable. Each half breaks that number on its own:
    ///
    /// * restoring the per-neuron `is_constrained` call makes every neuron the
    ///   loop reaches consult the exact map: 32 lookups if the instability test
    ///   still stands in front of it, and 64 for the fully historical loop,
    ///   which walked `0..flat.len()` testing the constraint FIRST, with no
    ///   stable-neuron skip anywhere;
    /// * keeping the per-node resolution but testing the constraint before
    ///   `l < 0 && u > 0` lets the two constrained STABLE neurons reach the
    ///   exact map as well: 3 lookups.
    ///
    /// The mix is what makes the second bullet observable, so the fixture
    /// asserts its own stable/unstable split below. An all-unstable fixture —
    /// which this test used until the ordering assertion was audited — leaves
    /// the ordering unexercised and pins only the first bullet.
    #[test]
    fn find_unstable_graph_neurons_resolves_constraints_once_per_node() {
        // Even index -> unstable (l < 0 < u). Odd index -> stable, alternating
        // stable-inactive (u <= 0) and stable-active (l >= 0), so both stable
        // shapes are represented.
        fn bounds_for(idx: usize) -> (f32, f32) {
            if idx.is_multiple_of(2) {
                (-1.0, 1.0)
            } else if idx % 4 == 1 {
                (-2.0, -1.0)
            } else {
                (1.0, 2.0)
            }
        }
        fn is_unstable(idx: usize) -> bool {
            let (l, u) = bounds_for(idx);
            l < 0.0 && u > 0.0
        }

        const WIDTH: usize = 64;
        // Constrained AND unstable: the only index that may reach the exact map.
        const CONSTRAINED_UNSTABLE: usize = 8;
        // Constrained but stable: reachable only if the constraint test runs
        // before the instability test.
        const CONSTRAINED_STABLE_INACTIVE: usize = 9;
        const CONSTRAINED_STABLE_ACTIVE: usize = 11;

        let lower: Vec<f32> = (0..WIDTH).map(|idx| bounds_for(idx).0).collect();
        let upper: Vec<f32> = (0..WIDTH).map(|idx| bounds_for(idx).1).collect();

        assert!(
            is_unstable(CONSTRAINED_UNSTABLE),
            "the once-per-node pin needs its constrained neuron to be unstable, \
             so that exactly one lookup survives the instability test"
        );
        assert!(
            !is_unstable(CONSTRAINED_STABLE_INACTIVE) && !is_unstable(CONSTRAINED_STABLE_ACTIVE),
            "the ordering pin needs constrained neurons the instability test \
             REJECTS; with none, dropping the ordering costs no lookups and \
             this test stops seeing it"
        );
        assert_eq!(
            (0..WIDTH).filter(|&idx| is_unstable(idx)).count(),
            WIDTH / 2,
            "the fixture must stay a mix of stable and unstable neurons"
        );

        let pre = BoundedTensor::new(
            ndarray::Array1::from_vec(lower).into_dyn(),
            ndarray::Array1::from_vec(upper).into_dyn(),
        )
        .expect("mixed stable/unstable pre-activation bounds");

        let graph = single_relu_graph();
        let mut domain = domain_with(pre);
        for (neuron_idx, is_active) in [
            (CONSTRAINED_UNSTABLE, true),
            (CONSTRAINED_STABLE_INACTIVE, false),
            (CONSTRAINED_STABLE_ACTIVE, true),
        ] {
            domain.history.add_constraint(
                GraphNeuronConstraint::new("relu0".to_string(), neuron_idx, is_active, 1.0)
                    .expect("finite score"),
            );
        }

        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let relu_nodes = vec!["relu0".to_string()];

        reset_test_exact_constraint_lookups();
        let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);
        let lookups = test_exact_constraint_lookups();

        let expected: Vec<(String, usize)> = (0..WIDTH)
            .filter(|&idx| is_unstable(idx) && idx != CONSTRAINED_UNSTABLE)
            .map(|idx| ("relu0".to_string(), idx))
            .collect();
        assert_eq!(
            unstable, expected,
            "every unstable neuron but the constrained one, ascending; no stable \
             neuron appears, constrained or not"
        );

        assert_eq!(
            lookups, 1,
            "the scan must resolve relu0's constraints once per node and reach \
             is_constrained only for the constrained index that is ALSO \
             unstable; got {lookups} exact lookups over {WIDTH} neurons — 3 \
             means the instability-first ordering is gone, 32 or 64 mean the \
             per-neuron lookup is back"
        );
    }

    /// The zero-constraint case must never touch the exact map at all — the
    /// common shape in BaB, where most ReLU nodes carry no split.
    #[test]
    fn find_unstable_graph_neurons_skips_the_exact_map_for_an_unconstrained_node() {
        const WIDTH: usize = 32;

        let pre = BoundedTensor::new(
            ndarray::Array1::from_elem(WIDTH, -1.0_f32).into_dyn(),
            ndarray::Array1::from_elem(WIDTH, 1.0_f32).into_dyn(),
        )
        .expect("all-unstable pre-activation bounds");

        let graph = single_relu_graph();
        let mut domain = domain_with(pre);
        // A constraint on a DIFFERENT node: `constraints` is non-empty, so the
        // guard cannot pass merely because the history is empty.
        domain.history.add_constraint(
            GraphNeuronConstraint::new("relu_other".to_string(), 3, false, 1.0)
                .expect("finite score"),
        );

        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let relu_nodes = vec!["relu0".to_string()];

        reset_test_exact_constraint_lookups();
        let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);
        let lookups = test_exact_constraint_lookups();

        assert_eq!(unstable.len(), WIDTH, "all {WIDTH} neurons are unstable");
        assert_eq!(
            lookups, 0,
            "a node with no ReLU constraint of its own must never reach the \
             exact map; got {lookups} lookups"
        );
    }

    /// The endpoint read must follow `ArrayD::iter()` — logical row-major, which
    /// is exactly what the retired `flatten()` collected — and NOT the raw
    /// buffer. `reversed_axes()` keeps the original buffer with transposed
    /// strides, so `as_slice()` is `None` and buffer order differs from logical
    /// order; a substitution that read memory order would return a different
    /// index set here.
    #[test]
    fn find_unstable_graph_neurons_reads_logical_order_on_a_transposed_tensor() {
        // Row-major 2x3, then transposed to 3x2 without copying.
        //   lower buffer [-1, 1, -1,  1, -1, 1] -> iter [-1,  1,  1, -1, -1, 1]
        //   upper buffer [ 1, 2,  1,  2,  1, 2] -> iter [ 1,  2,  2,  1,  1, 2]
        // so l < 0 < u holds at logical indices 0, 3, 4. Reading the BUFFER
        // instead would make 0, 2, 4 unstable — a different index set, so the
        // assertion below separates the two.
        let lower =
            ndarray::Array2::from_shape_vec((2, 3), vec![-1.0_f32, 1.0, -1.0, 1.0, -1.0, 1.0])
                .expect("2x3 lower")
                .reversed_axes()
                .into_dyn();
        let upper = ndarray::Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 1.0, 2.0, 1.0, 2.0])
            .expect("2x3 upper")
            .reversed_axes()
            .into_dyn();
        assert!(
            lower.as_slice().is_none(),
            "the test needs a non-standard layout for the read to be observable"
        );

        let pre = BoundedTensor::new(lower, upper).expect("transposed pre-activation bounds");

        // Reference: what the retired `flatten()` read, computed the old way.
        let flat = pre.flatten();
        let reference: Vec<(String, usize)> = (0..flat.len())
            .filter(|&idx| flat.lower()[[idx]] < 0.0 && flat.upper()[[idx]] > 0.0)
            .map(|idx| ("relu0".to_string(), idx))
            .collect();

        let graph = single_relu_graph();
        let domain = domain_with(pre);
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let relu_nodes = vec!["relu0".to_string()];
        let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);

        assert_eq!(
            unstable,
            vec![
                ("relu0".to_string(), 0),
                ("relu0".to_string(), 3),
                ("relu0".to_string(), 4),
            ],
            "logical (iter) order, not buffer order"
        );
        assert_eq!(
            unstable, reference,
            "must match what flatten() would have read"
        );
    }
}
