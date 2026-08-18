// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Graph-MIP escalation driver (increment 5): wires the DAG-aware encoder
// (`graph_mip`, increments 1-4) into the dispatch fallback's "no sequential
// form" arm, where Graph models (cifar100/tinyimagenet resnets) previously
// died with "MIP escalation ineligible". Default-on; `NY_GRAPH_MIP=0` restores
// the old path without this escalation. The phase ledger shares this enable
// gate and additionally requires Auto mode's category policy to request a
// nonzero MIP slice; zero-reservation categories keep the time and skip both
// the bounds stash and whole-net Auto escalation. Explicit MIP remains an
// override.
//
// Flow (mirrors `mip_highs::verify_with_mip`, graph-adapted):
//   1. Per-node bounds: REUSE the α-CROWN map already computed once per
//      property (`verify/graph.rs` stashes it here); only if no stash matches
//      (a lane that never precomputed) is a budgeted recompute performed.
//   2. Flatten `HashMap<String, BoundedTensor>` → `HashMap<String, Vec<Bound>>`
//      (the encoder's shape; ReLU reads its affine producer's box).
//   3. `encode_graph` (exact-or-certified-outward operator rows + DELTA box
//      inflation) → `MipParts` → the existing `MipSolver` certificate
//      machinery. Whole-net solves are deliberately serial: automatic phase
//      splitting clones the complete IR once per sibling and defeats the
//      pre-encode memory cap.
//   4. VNN-LIB spec stamping per clause (disjunctive: one solve per clause).
//   5. SOUNDNESS GATES, both stricter than the sequential lane:
//      * any Sat witness is clamped into the box and revalidated through the
//        GRAPH forward (`propagate_concrete_point`) before `Violated`;
//      * UNSAT is admitted ONLY as `MipResult::Unsat { certified: true }`
//        (verified Farkas evidence, ay_lib LG3) — an uncertified Unsat
//        degrades to Unknown. The 0-wrong moat for the new path.
//
// Escalation can only ever IMPROVE on the BaB verdict: it runs only when BaB
// was inconclusive (Timeout/Unknown), and every inconclusive/failed MIP arm
// degrades back to that BaB verdict.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use ndarray::ArrayD;
use ny_core::{Bound, VerificationResult};
use ny_mip::{MipBackend, MipConfig, MipResult, MipSolver};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{BabVerificationStatus, BetaCrownVerifier, GraphNetwork, PhaseBudgetConfig};
use ny_tensor::BoundedTensor;
use tracing::{info, warn};

use super::dispatch::{
    graph_unstable_relu_count, is_mip_encodable_graph, should_auto_escalate_to_mip,
};
use super::graph_mip::{graph_mip_enabled, GraphMipEncoding};
use super::mip_highs::{clamp_witness_to_box, mip_constraint_margin, print_result};
use super::mip_preprocess::bounded_tensor_to_bounds;
use super::output::{verification_result_exit_code, EffectiveTreatmentProjection};
use crate::{CompleteVerifierArg, MipSolverArg};

// ===========================================================================
// (a) Node-bounds reuse: the per-property α-CROWN map, stashed at its producer
// ===========================================================================

/// Stash the per-property graph node bounds for a later Graph-MIP escalation.
///
/// FIX 1 (stash coverage): the mailbox now lives in ny-propagate
/// (`beta_crown::graph_mip_leaf`), because cifar100's relational/multi-clause
/// lane computes its per-property bounds INSIDE ny-propagate's BaB bootstrap
/// (the multi-objective root evaluation), not at the ny-cli per-constraint
/// precompute — the propagate-side freeze points stash there directly, and
/// this wrapper keeps the ny-cli producer (`verify/graph.rs`) on the same
/// mailbox. The bounded one-slot stash follows its actual whole-network
/// consumer and is disabled by `NY_GRAPH_MIP=0` or a zero-reservation phase
/// policy; the leaf oracle receives node bounds directly. Last writer wins,
/// and the escalation checks exact graph and input identities so stale or
/// foreign bounds never match.
pub(super) fn stash_graph_bounds_for_mip(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    phase_budget: &PhaseBudgetConfig,
    node_bounds: &HashMap<String, BoundedTensor>,
) {
    ny_propagate::beta_crown::graph_mip_leaf::stash_root_bounds_for_mip(
        graph,
        input,
        phase_budget,
        node_bounds,
    );
}

/// Fetch the stashed bounds when they were computed for exactly this model and
/// input.
fn stashed_graph_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> Option<Arc<HashMap<String, BoundedTensor>>> {
    ny_propagate::beta_crown::graph_mip_leaf::stashed_root_bounds(graph, input)
}

/// Flatten the propagate-side node bounds (`BoundedTensor` per node) into the
/// encoder's shape (`Vec<Bound>` per node, flattened element order). Keys are
/// unchanged: the map is keyed by node name with the node's OUTPUT box, which
/// is exactly the encoder's contract — an affine node's own box bounds its
/// output columns, and a ReLU reads its input node's (the affine producer's)
/// box as the big-M pre-activation range.
pub(super) fn flatten_node_bounds(
    node_bounds: &HashMap<String, BoundedTensor>,
) -> Result<HashMap<String, Vec<Bound>>> {
    let mut out = HashMap::with_capacity(node_bounds.len());
    for (name, bt) in node_bounds {
        let bounds = bounded_tensor_to_bounds(bt)
            .map_err(|e| anyhow!("flatten node bounds for '{name}': {e}"))?;
        out.insert(name.clone(), bounds);
    }
    Ok(out)
}

// ===========================================================================
// (d) Escalation entry, called from the dispatch fallback
// ===========================================================================

/// Binary-count budget for Graph-MIP escalation: the number of UNSTABLE ReLU
/// neurons (= big-M binaries) the MIP may carry. `NY_GRAPH_MIP_MAX_BINARIES`
/// overrides; default 1024 (cifar100 root measures ~494 under α-CROWN bounds).
/// The MEMORY-safety gate is the encode-nnz cap (`NY_GRAPH_MIP_MAX_NNZ`, 5M), NOT
/// this binary count — cifar100's 494 binaries already passed 1024; it is the 44M
/// nnz that OOMs, so the nnz cap is what declines it. Kept at 1024 so the proven
/// `mscn_escalation_*` certifications (real instances ay solves) still arm.
pub(super) fn graph_mip_max_binaries() -> usize {
    std::env::var("NY_GRAPH_MIP_MAX_BINARIES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(1024)
}

/// Solver configuration for a whole-net Graph-MIP encode.
///
/// `MipConfig::default().parallel_split == 0` expands to a power-of-two set of
/// concurrent subproblems based on host parallelism, cloning the complete IR
/// for every sibling. The 5M-NNZ admission cap bounds one encode, not that
/// multiplicative clone set. Use the existing explicit serial setting (`1`)
/// so the pre-encode cap remains an enforced one-model memory envelope.
fn graph_mip_solver_config(
    backend: MipBackend,
    timeout_secs: f64,
    ay_node_warm_time_limit: Option<std::time::Duration>,
) -> MipConfig {
    MipConfig {
        backend,
        parallel_split: 1,
        timeout_secs,
        ay_node_warm_time_limit,
        ..MipConfig::default()
    }
}

fn graph_mip_min_slice_secs() -> u64 {
    std::env::var("NY_GRAPH_MIP_MIN_SLICE_S")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(20)
}

fn graph_mip_slice_admitted(mip_timeout: u64, min_slice: u64) -> bool {
    mip_timeout >= min_slice
}

fn graph_mip_policy_admitted(
    complete_verifier: CompleteVerifierArg,
    phase_budget: &PhaseBudgetConfig,
) -> bool {
    match complete_verifier {
        CompleteVerifierArg::Mip => true,
        CompleteVerifierArg::Auto => phase_budget.requests_mip_reservation(),
        CompleteVerifierArg::Bab => false,
    }
}

/// Try the Graph-MIP escalation. Returns `Ok(true)` when the MIP path took over
/// reporting (a verdict — possibly Timeout — was printed with the MIP method),
/// `Ok(false)` when ineligible (caller reports the BaB verdict unchanged), and
/// `Err` on an internal failure (caller logs + degrades to the BaB verdict).
#[cfg(test)]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_graph_mip_escalation(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    verifier: &BetaCrownVerifier,
    complete_verifier: CompleteVerifierArg,
    bab_status: &BabVerificationStatus,
    deadline: std::time::Instant,
    mip_solver: MipSolverArg,
    property: Option<&Path>,
    model: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    reporting_start: std::time::Instant,
    json: bool,
) -> Result<bool> {
    try_graph_mip_escalation_with_treatment(
        graph,
        input,
        vnnlib,
        verifier,
        complete_verifier,
        bab_status,
        deadline,
        mip_solver,
        property,
        model,
        epsilon,
        threshold,
        reporting_start,
        None,
        json,
    )
}

/// Production entry that carries the already-resolved treatment projection to
/// a Graph-MIP verdict. The compatibility wrapper above keeps direct unit tests
/// and non-reporting callers source-stable.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_graph_mip_escalation_with_treatment(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    verifier: &BetaCrownVerifier,
    complete_verifier: CompleteVerifierArg,
    bab_status: &BabVerificationStatus,
    deadline: std::time::Instant,
    mip_solver: MipSolverArg,
    property: Option<&Path>,
    model: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    reporting_start: std::time::Instant,
    effective_treatment: Option<&EffectiveTreatmentProjection>,
    json: bool,
) -> Result<bool> {
    if !graph_mip_enabled() {
        return Ok(false);
    }

    // A zero-reservation category is an explicit whole-net AUTO admission
    // decline, not merely a deadline refund. Otherwise an unusually early BaB
    // return could leave >=20s and trigger a 5--60s bounds recompute for a graph
    // the category already classified as unsuitable (CIFAR: ~44M NNZ > 5M).
    // Explicit `--complete-verifier mip` remains an override. Consume any stale
    // one-shot stash exactly as the downstream path would have done.
    if !graph_mip_policy_admitted(complete_verifier, &verifier.config.phase_budget) {
        let _ = stashed_graph_bounds(graph, input);
        info!(
            "Graph-MIP: NOT eligible (complete-verifier policy {:?}, nonzero reservation={}); keeping BaB {:?}",
            complete_verifier,
            verifier.config.phase_budget.requests_mip_reservation(),
            bab_status
        );
        return Ok(false);
    }

    // Whole-net SLICE gate (live-run residual): the whole-net encode + exact-
    // rational conversion needs a real slice. Check this BEFORE fetching or
    // recomputing node bounds: zero-reservation policies normally reach this
    // fallback with only the post-BaB tail left, and a five-second recompute
    // before rejecting the same sub-minimum slice would spend time on a lane
    // guaranteed not to encode. `NY_GRAPH_MIP_MIN_SLICE_S` (default 20 s) is
    // the floor. This is scheduling-only: the existing decline is unchanged.
    let min_slice = graph_mip_min_slice_secs();
    let mip_timeout = deadline
        .saturating_duration_since(std::time::Instant::now())
        .as_secs();
    if !graph_mip_slice_admitted(mip_timeout, min_slice) {
        // Preserve the one-shot mailbox lifecycle from the former ordering:
        // decline before any bounds work, but release an already-stashed map
        // instead of retaining its Arc until a later property or process exit.
        let _ = stashed_graph_bounds(graph, input);
        info!(
            "Graph-MIP: NOT eligible (slice {mip_timeout}s < min {min_slice}s — the whole-net encode cannot fit); keeping BaB {:?}",
            bab_status
        );
        return Ok(false);
    }

    // (a) Per-node bounds: reuse the per-property stash; recompute (budgeted)
    // only when no stash matches this input box.
    let node_bounds: Arc<HashMap<String, BoundedTensor>> = match stashed_graph_bounds(graph, input)
    {
        Some(b) => {
            info!("Graph-MIP: reusing the per-property α-CROWN node bounds (stash hit)");
            b
        }
        None => {
            // Budget the recompute to a fraction of the MIP slice so a slow
            // bound pass cannot eat the whole escalation budget.
            let live_remaining = deadline
                .saturating_duration_since(std::time::Instant::now())
                .as_secs();
            if live_remaining == 0 {
                return Ok(false);
            }
            let bounds_budget = (live_remaining / 4).clamp(1, 60);
            info!(
                "Graph-MIP: no stashed bounds for this box; recomputing (budget {bounds_budget}s)"
            );
            let deadline = (std::time::Instant::now()
                + std::time::Duration::from_secs(bounds_budget))
            .min(deadline);
            let (nb, _out) = verifier
                .compute_initial_graph_bounds(graph, input, Some(deadline))
                .map_err(|e| anyhow!("Graph-MIP bound recompute failed: {e}"))?;
            Arc::new(nb)
        }
    };
    let flat_bounds = flatten_node_bounds(&node_bounds)?;

    // Eligibility: layer set + binary-count budget (from the SOUND node bounds:
    // unstable ReLU count = big-M binary count). Explicit `--complete-verifier
    // mip` bypasses the budget (mirrors the sequential arm's unconditional
    // fallback); `auto` respects it.
    let max_binaries = graph_mip_max_binaries();
    let encodable = complete_verifier == CompleteVerifierArg::Mip
        || is_mip_encodable_graph(graph, &flat_bounds, max_binaries);
    if !should_auto_escalate_to_mip(complete_verifier, bab_status, encodable) {
        // Visibility parity (FIX 1): say WHY eligibility rejected — the
        // unstable-ReLU (= big-M binary) count vs the budget is the number
        // that inflates when the bounds are loose (truncated recompute).
        let unstable = graph_unstable_relu_count(graph, &flat_bounds);
        match unstable {
            Some(n) => info!(
                "Graph-MIP: NOT eligible (unstable={n} > budget={max_binaries} or unsupported layer); keeping BaB {:?}",
                bab_status
            ),
            None => info!(
                "Graph-MIP: NOT eligible (unsupported layer or a ReLU without a pre-activation box); keeping BaB {:?}",
                bab_status
            ),
        }
        return Ok(false);
    }

    // Whole-net NNZ gate (live-run residual): a cheap pre-encode estimate keeps
    // an over-scale encode from eating the slice / OOMing before the first
    // deadline check. MEMORY-SAFE default 5M (was 100M, which let cifar100's ~44M
    // encode through into the documented 24GB memory bomb). With the gate now
    // default-on, this makes the escalation DECLINE cheaply (pre-encode) on any
    // net ay cannot hold; raise `NY_GRAPH_MIP_MAX_NNZ` as ay's solver + memory
    // footprint grow.
    let max_nnz = std::env::var("NY_GRAPH_MIP_MAX_NNZ")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(5_000_000);
    match super::graph_mip_leaf::estimate_encode_nnz(graph, &flat_bounds) {
        Some(nnz) if nnz <= max_nnz => {
            info!("Graph-MIP: estimated encode nnz {nnz} (cap {max_nnz})");
        }
        Some(nnz) => {
            info!(
                "Graph-MIP: NOT eligible (estimated nnz {nnz} > cap {max_nnz}); keeping BaB {:?}",
                bab_status
            );
            return Ok(false);
        }
        None => {
            info!(
                "Graph-MIP: NOT eligible (nnz estimate unavailable — unsupported layer or                  missing box); keeping BaB {:?}",
                bab_status
            );
            return Ok(false);
        }
    }

    let solve_timeout = deadline
        .saturating_duration_since(std::time::Instant::now())
        .as_secs();
    if solve_timeout == 0 {
        return Ok(false);
    }
    info!(
        "BaB inconclusive ({:?}), escalating GRAPH model to MIP with {solve_timeout}s remaining (default-on Graph-MIP)",
        bab_status
    );
    verify_graph_with_mip(
        graph,
        input,
        vnnlib,
        &flat_bounds,
        deadline,
        mip_solver,
        property,
        model,
        epsilon,
        threshold,
        reporting_start,
        effective_treatment,
        json,
    )?;
    Ok(true)
}

// ===========================================================================
// The graph MIP verification driver (mirrors mip_highs::verify_with_mip)
// ===========================================================================

/// Solve the VNN-LIB property on the graph MIP encoding and report the result.
///
/// Encodes ONCE (the DAG walk + im2col unfolds re-scan every conv weight, so a
/// per-clause re-encode would multiply that cost), then stamps each clause's
/// constraints onto a clone. Disjunctive semantics mirror
/// `mip_highs::solve_disjunctive`: SAT on any clause (confirmed in-box through
/// the GRAPH forward) → Violated; certified UNSAT on ALL clauses → Verified;
/// anything else → Timeout/Unknown (never a verdict).
#[allow(clippy::too_many_arguments)]
fn verify_graph_with_mip(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    node_bounds: &HashMap<String, Vec<Bound>>,
    deadline: std::time::Instant,
    mip_solver: MipSolverArg,
    property: Option<&Path>,
    model: Option<&Path>,
    epsilon: f32,
    threshold: f32,
    reporting_start: std::time::Instant,
    effective_treatment: Option<&EffectiveTreatmentProjection>,
    json: bool,
) -> Result<()> {
    let backend = mip_solver.mip_backend();
    if !json {
        println!("\nRunning Graph-MIP verification (ay solver, serial whole-net model)...");
    }
    let start = std::time::Instant::now();
    if start >= deadline {
        anyhow::bail!("Graph-MIP deadline exhausted before encoding");
    }

    let input_bounds = bounded_tensor_to_bounds(input)?;
    // Encode under the slice deadline (live-run residual): the DAG walk checks
    // it per node and bails cleanly (degrading to the BaB verdict) instead of
    // running through the watchdog mid-encode.
    let base = super::graph_mip::encode_graph_with_deadline(
        graph,
        &input_bounds,
        node_bounds,
        Some(deadline),
    )?;
    let num_outputs = vnnlib.num_outputs;
    if !json {
        println!(
            "  Graph encoded: {} cols, {} binaries (unstable ReLUs), {:.2}s",
            base.problem.num_cols(),
            base.binary_vars.len(),
            start.elapsed().as_secs_f64()
        );
    }

    if std::time::Instant::now() >= deadline {
        anyhow::bail!("Graph-MIP deadline exhausted during encoding");
    }

    let clauses: Vec<Vec<OutputConstraint>> = if vnnlib.output_constraint_clauses.is_empty() {
        vec![vnnlib.output_constraints.clone()]
    } else {
        vnnlib.output_constraint_clauses.clone()
    };
    let disjunctive = vnnlib.is_disjunction && clauses.len() > 1;

    let result = if disjunctive {
        solve_graph_disjunctive(
            graph,
            input,
            &base,
            &clauses,
            deadline,
            backend,
            num_outputs,
            json,
        )?
    } else {
        // Conjunctive: stamp every constraint of every (single) clause.
        let mut enc = base;
        let all: Vec<&OutputConstraint> = clauses.iter().flatten().collect();
        for c in &all {
            enc.add_output_constraint(c)?;
        }
        let constraints: Vec<OutputConstraint> = all.into_iter().cloned().collect();
        let live_timeout = deadline
            .saturating_duration_since(std::time::Instant::now())
            .as_secs_f64();
        let mip_result = if live_timeout <= 0.0 {
            MipResult::Timeout
        } else {
            let ay_node_warm_time_limit = enc.ay_node_warm_time_limit();
            let solver = MipSolver::new(
                enc.into_parts(),
                graph_mip_solver_config(backend, live_timeout, ay_node_warm_time_limit),
            );
            solver
                .check_feasibility()
                .map_err(|e| anyhow!("Graph-MIP solve failed: {e}"))?
        };
        map_graph_mip_result(mip_result, graph, input, &constraints, num_outputs)
    };

    let result = if std::time::Instant::now() >= deadline {
        VerificationResult::Timeout {
            provenance: Default::default(),
            partial_bounds: Some(vec![
                Bound::new_allow_infinite(
                    f32::NEG_INFINITY,
                    f32::INFINITY
                );
                num_outputs
            ]),
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        }
    } else {
        result
    };
    let elapsed = reporting_start.elapsed();
    let publication_refused = print_result(
        &result,
        property,
        model,
        epsilon,
        threshold,
        elapsed,
        backend,
        effective_treatment,
        json,
    )?;
    let exit_code = if publication_refused {
        crate::commands::verify::exit_codes::UNKNOWN
    } else {
        verification_result_exit_code(&result)
    };
    if exit_code != crate::commands::verify::exit_codes::VERIFIED && !super::output::is_capturing()
    {
        std::process::exit(exit_code);
    }
    Ok(())
}

/// Disjunctive graph solve: one MIP per clause, equal remaining-budget slices.
/// UNSAT (certified) must hold on EVERY clause for Verified; a confirmed SAT on
/// any clause returns Violated immediately; anything else degrades to
/// Timeout/Unknown (sound).
#[allow(clippy::too_many_arguments)]
fn solve_graph_disjunctive(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    base: &GraphMipEncoding,
    clauses: &[Vec<OutputConstraint>],
    deadline: std::time::Instant,
    backend: MipBackend,
    num_outputs: usize,
    json: bool,
) -> Result<VerificationResult> {
    let num_clauses = clauses.len();
    if !json {
        println!("  Disjunctive property: {num_clauses} clauses, solving independently...");
    }
    let mut all_certified_unsat = true;
    let mut undecided_reason: Option<String> = None;

    for (idx, clause) in clauses.iter().enumerate() {
        if std::time::Instant::now() >= deadline {
            all_certified_unsat = false;
            undecided_reason = Some("budget exhausted before all clauses solved".into());
            break;
        }

        let mut enc = base.clone();
        for c in clause {
            enc.add_output_constraint(c)?;
        }
        let remaining = deadline
            .saturating_duration_since(std::time::Instant::now())
            .as_secs_f64();
        if remaining <= 0.0 {
            all_certified_unsat = false;
            undecided_reason = Some("budget exhausted while preparing a clause".into());
            break;
        }
        let clause_timeout = remaining / (num_clauses - idx).max(1) as f64;
        let ay_node_warm_time_limit = enc.ay_node_warm_time_limit();
        let solver = MipSolver::new(
            enc.into_parts(),
            graph_mip_solver_config(backend, clause_timeout, ay_node_warm_time_limit),
        );
        let mip_result = solver
            .check_feasibility()
            .map_err(|e| anyhow!("Graph-MIP solve failed on clause {idx}: {e}"))?;

        match &mip_result {
            MipResult::Sat { .. } => {
                let revalidated =
                    map_graph_mip_result(mip_result, graph, input, clause, num_outputs);
                if matches!(revalidated, VerificationResult::Violated { .. }) {
                    if !json {
                        println!(
                            "  Clause {}/{num_clauses}: SAT (counterexample confirmed in-box via graph forward)",
                            idx + 1
                        );
                    }
                    return Ok(revalidated);
                }
                // Unconfirmed witness: the clause's unsafe region may be
                // reachable — we can never conclude Verified.
                all_certified_unsat = false;
                undecided_reason = Some(format!(
                    "clause {idx}: MIP sat witness failed graph revalidation"
                ));
                if !json {
                    println!(
                        "  Clause {}/{num_clauses}: SAT but witness unconfirmed (demoted)",
                        idx + 1
                    );
                }
            }
            MipResult::Unsat { certified: true } => {
                if !json {
                    println!(
                        "  Clause {}/{num_clauses}: UNSAT (certified Farkas evidence)",
                        idx + 1
                    );
                }
            }
            MipResult::Unsat { certified: false } => {
                // (f) 0-wrong moat: the graph path admits UNSAT only with
                // verified certificate evidence.
                all_certified_unsat = false;
                undecided_reason = Some(format!(
                    "clause {idx}: MIP UNSAT lacked a verified certificate (graph path admits certified UNSAT only)"
                ));
                if !json {
                    println!(
                        "  Clause {}/{num_clauses}: UNSAT but uncertified (not admitted on the graph path)",
                        idx + 1
                    );
                }
            }
            MipResult::Timeout => {
                all_certified_unsat = false;
                undecided_reason = Some(format!("clause {idx}: MIP timeout"));
                if !json {
                    println!("  Clause {}/{num_clauses}: timeout", idx + 1);
                }
            }
            MipResult::Error(e) => {
                all_certified_unsat = false;
                undecided_reason = Some(format!("clause {idx}: MIP error: {e}"));
                warn!("Graph-MIP clause {idx} error (treated as undecided): {e}");
            }
        }
    }

    let inf_bounds =
        || vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); num_outputs];
    if all_certified_unsat {
        info!("Graph-MIP: every disjunctive clause certified UNSAT — property verified");
        Ok(VerificationResult::Verified {
            provenance: Default::default(),
            output_bounds: inf_bounds(),
            proof: None,
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        })
    } else {
        if let Some(reason) = &undecided_reason {
            info!("Graph-MIP undecided: {reason}");
        }
        Ok(VerificationResult::Timeout {
            provenance: Default::default(),
            partial_bounds: Some(inf_bounds()),
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        })
    }
}

// ===========================================================================
// (e)+(f) Result mapping: graph witness revalidation + certified-only UNSAT
// ===========================================================================

/// Map a graph-path MIP result to a `VerificationResult`.
///
/// * `Sat` → clamp + independent GRAPH forward revalidation (below); only a
///   confirmed in-box violation becomes `Violated`.
/// * `Unsat { certified: true }` → `Verified` (Farkas evidence verified at the
///   ny-mip seam, ay_lib LG3).
/// * `Unsat { certified: false }` → `Unknown` — the graph path's 0-wrong moat
///   admits ONLY certified UNSAT (stricter than the sequential lane).
/// * `Timeout`/`Error` → `Timeout`/`Unknown` (sound degrades).
fn map_graph_mip_result(
    result: MipResult,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    constraints: &[OutputConstraint],
    num_outputs: usize,
) -> VerificationResult {
    let inf_bounds =
        || vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); num_outputs];
    match result {
        MipResult::Sat { input_values, .. } => {
            revalidate_graph_mip_witness(graph, input, &input_values, constraints, num_outputs)
        }
        MipResult::Unsat { certified: true } => {
            info!("Graph-MIP UNSAT admitted with verified exact certificate");
            VerificationResult::Verified {
                provenance: Default::default(),
                output_bounds: inf_bounds(),
                proof: None,
                actual_method: Some(ny_core::MethodUsed::MipHiGHS),
            }
        }
        MipResult::Unsat { certified: false } => {
            warn!(
                "Graph-MIP UNSAT lacked a verified certificate; NOT admitted \
                 (graph path requires certified UNSAT) — degrading to Unknown"
            );
            VerificationResult::Unknown {
                provenance: Default::default(),
                bounds: inf_bounds(),
                reason: ny_core::UnknownReason::SmtUnknown {
                    solver_reason: Some(
                        "Graph-MIP UNSAT without verified certificate (admission requires one)"
                            .to_string(),
                    ),
                },
                actual_method: Some(ny_core::MethodUsed::MipHiGHS),
            }
        }
        MipResult::Timeout => VerificationResult::Timeout {
            provenance: Default::default(),
            partial_bounds: Some(inf_bounds()),
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        },
        MipResult::Error(msg) => VerificationResult::Unknown {
            provenance: Default::default(),
            bounds: inf_bounds(),
            reason: ny_core::UnknownReason::SmtUnknown {
                solver_reason: Some(msg),
            },
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        },
    }
}

/// Clamp a raw graph-MIP witness into the VNN-LIB box, re-validate it with an
/// independent forward pass through the ORIGINAL GRAPH, and emit `Violated`
/// only if the spec still holds at the clamped point (mirrors
/// `mip_highs::revalidate_mip_witness`, graph forward instead of sequential;
/// no f64 rescue — the graph has no f64 twin forward, so a ULP-borderline
/// witness demotes to Unknown, which is sound).
fn revalidate_graph_mip_witness(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    raw_input: &[f64],
    constraints: &[OutputConstraint],
    num_outputs: usize,
) -> VerificationResult {
    let unknown = |reason: &str| VerificationResult::Unknown {
        provenance: Default::default(),
        bounds: vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); num_outputs],
        reason: ny_core::UnknownReason::SmtUnknown {
            solver_reason: Some(reason.to_string()),
        },
        actual_method: Some(ny_core::MethodUsed::MipHiGHS),
    };

    if constraints.is_empty() {
        return unknown("Graph-MIP sat had no output constraints to revalidate against");
    }

    // 1. Clamp into the box (f64→f32 cast first — exactly the bytes ORT reads).
    let clamped = clamp_witness_to_box(raw_input, input);

    // 2. Independent forward through the ORIGINAL graph (engine=None; the
    //    point forward, NOT an IBP box — see verify/graph.rs `evaluate_graph`).
    let revalidated: ArrayD<f32> = match BoundedTensor::concrete(clamped.clone())
        .map_err(anyhow::Error::from)
        .and_then(|pt| {
            graph
                .propagate_concrete_point(&pt, None, None)
                .map_err(anyhow::Error::from)
        }) {
        Ok(out) => out.center(),
        Err(e) => {
            warn!("Graph-MIP sat witness revalidation forward failed: {e}");
            return unknown("Graph-MIP sat witness revalidation forward failed");
        }
    };

    // 3. Re-check the spec at the clamped point (exact SMT-LIB strictness
    //    semantics; the trusted-ORT vnncomp gate arbitrates any sub-eps margin
    //    downstream before a sat is scored).
    let confirmed = super::verify::check_unsafe_counterexample(&revalidated, constraints);
    if confirmed {
        VerificationResult::Violated {
            provenance: Default::default(),
            counterexample: clamped.iter().copied().collect(),
            output: revalidated.iter().copied().collect(),
            details: None,
            actual_method: Some(ny_core::MethodUsed::MipHiGHS),
        }
    } else {
        let min_margin = constraints
            .iter()
            .map(|c| mip_constraint_margin(c, &revalidated))
            .fold(f32::INFINITY, f32::min);
        warn!(
            "Graph-MIP sat witness failed in-box revalidation (min constraint margin \
             {min_margin:.3e}); demoting to Unknown"
        );
        unknown("Graph-MIP sat witness failed in-box revalidation")
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    fn whole_net_solver_config_enforces_one_model_at_a_time() {
        for (backend, timeout_secs) in [(MipBackend::Ay, 30.0), (MipBackend::AyProc, 7.5)] {
            let node_warm_limit = Some(std::time::Duration::from_secs(5));
            let config = graph_mip_solver_config(backend, timeout_secs, node_warm_limit);
            assert_eq!(config.backend, backend);
            assert_eq!(config.timeout_secs, timeout_secs);
            assert_eq!(config.ay_node_warm_time_limit, node_warm_limit);
            assert_eq!(
                config.parallel_split, 1,
                "whole-net Graph-MIP must not clone the full IR into sibling solves"
            );
        }
    }

    #[test]
    fn subminimum_slice_is_rejected_before_expensive_admission_work() {
        assert!(!graph_mip_slice_admitted(0, 20));
        assert!(!graph_mip_slice_admitted(19, 20));
        assert!(graph_mip_slice_admitted(20, 20));
        assert!(graph_mip_slice_admitted(21, 20));
    }

    #[test]
    fn whole_net_auto_requires_reservation_but_explicit_mip_overrides() {
        let zero_policy = PhaseBudgetConfig {
            mip_min_fraction: 0.0,
            mip_min_secs: 0,
            ..Default::default()
        };
        assert!(!graph_mip_policy_admitted(
            CompleteVerifierArg::Auto,
            &zero_policy
        ));
        assert!(graph_mip_policy_admitted(
            CompleteVerifierArg::Auto,
            &PhaseBudgetConfig::default()
        ));
        assert!(graph_mip_policy_admitted(
            CompleteVerifierArg::Mip,
            &zero_policy
        ));
        assert!(!graph_mip_policy_admitted(
            CompleteVerifierArg::Bab,
            &PhaseBudgetConfig::default()
        ));
    }
}
