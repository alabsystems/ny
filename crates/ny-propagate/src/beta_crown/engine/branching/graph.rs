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
            // #2098: Skip nodes with empty inputs instead of fabricating NETWORK_INPUT.
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

            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                if domain
                    .history
                    .is_constrained(node_name, neuron_idx)
                    .is_some()
                {
                    continue;
                }

                let l = flat.lower()[[neuron_idx]];
                let u = flat.upper()[[neuron_idx]];

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
        // Guard against empty unstable list (#1915).
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
                        if std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1") {
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
            // #2098: Skip nodes with empty inputs instead of fabricating NETWORK_INPUT.
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
    /// widths are skipped (#2588); `None` (nothing scorable) lets the caller
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

        for node_name in relu_nodes {
            let relu_node = match graph.nodes.get(node_name) {
                Some(n) => n,
                None => continue,
            };
            if !is_zero_threshold_binary_activation(&relu_node.layer) {
                continue;
            }
            // #2098: Skip nodes with empty inputs instead of fabricating NETWORK_INPUT.
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

            let flat = pre_bounds.flatten();
            for neuron_idx in 0..flat.len() {
                if domain
                    .history
                    .is_constrained(node_name, neuron_idx)
                    .is_some()
                {
                    continue;
                }

                let l = flat.lower()[[neuron_idx]];
                let u = flat.upper()[[neuron_idx]];

                if l < 0.0 && u > 0.0 {
                    unstable.push((node_name.clone(), neuron_idx));
                }
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
    /// For k > 1, the top-k candidates enable multi-depth splitting (#2767).
    ///
    /// Reference: alpha-beta-CROWN `find_topk_scores()` in `base.py:97-168`.
    pub(in crate::beta_crown::engine) fn select_graph_branches(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
        k: usize,
    ) -> Result<Vec<(String, usize, f32)>> {
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
